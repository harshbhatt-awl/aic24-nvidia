"""SOLIDER Swin-Small ReID wrapper.

SOLIDER upstream: https://github.com/tinyvision/SOLIDER-REID
Architecture: build_transformer with TRANSFORMER_TYPE=swin_small_patch4_window7_224,
              NECK=bnneck, NECK_FEAT=before, no JPM, no SIE.

The vendored swin_transformer.py from SOLIDER requires mmcv (unavailable in .venv,
Python 3.14 / PyTorch 2.x).  Instead we use timm's SwinTransformer backbone
(same architecture, 768-dim features) with a key-remapping function to load
SOLIDER-trained checkpoints.

Checkpoint key format differences (SOLIDER → timm):
  base.patch_embed.projection.* → patch_embed.proj.*
  base.stages.N.blocks.M.*      → layers.N.blocks.M.*
  base.stages.N.downsample.*    → layers.(N+1).downsample.*
  base.stages.N.norm.*          → (dropped; timm handles per-layer norms internally)
  base.norm{N}.*                → norm.*  (only norm3, the final LayerNorm)
  attn.w_msa.qkv                → attn.qkv
  attn.w_msa.proj               → attn.proj
  attn.w_msa.relative_position* → attn.relative_position*
  ffn.layers.0.0.*              → mlp.fc1.*
  ffn.layers.1.*                → mlp.fc2.*
  semantic_embed_w/b.*          → (skipped; SOLIDER-specific pretraining keys)

Position bias table:
  SOLIDER checkpoints store 169-entry (13×13, for 7×7 window at 224×224 init).
  When the model operates at a different resolution the position bias tables are
  bicubic-interpolated to match the current model's table size (as done in
  SOLIDER's own init_weights).

Weight file: weights/solider_swin_small.pth
  Download the MSMT17 Swin-Small checkpoint:
    gdown 1C-aIZdFyjFsZX4W4feG-Ex39RU2Qvu3b -O weights/solider_swin_small.pth
  or see https://github.com/tinyvision/SOLIDER-REID for Market1501/other links.
"""

from __future__ import annotations
import math
import re
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["load_solider_swin_small"]

# Swin-Small output channels
_SWIN_SMALL_FEAT_DIM = 768


class _SOLIDERSwinSmall(nn.Module):
    """SOLIDER build_transformer wrapper: timm Swin-Small backbone + BN neck.

    In eval mode returns global features BEFORE BN (NECK_FEAT='before'),
    matching the upstream test configuration in configs/msmt17/swin_small.yml.
    Output shape: (B, 768).
    """

    def __init__(self, img_size=(256, 128)) -> None:
        super().__init__()
        import timm  # type: ignore

        # Use the target inference img_size so the model's attn masks and
        # patch-grid are consistent with actual input dimensions.
        # Relative position bias tables may differ from the checkpoint (which
        # was initialized at 224×224) — these are interpolated during load.
        self.base = timm.create_model(
            "swin_small_patch4_window7_224",
            pretrained=False,
            num_classes=0,        # strip classifier head → returns (B, 768)
            img_size=img_size,
        )
        self.bottleneck = nn.BatchNorm1d(_SWIN_SMALL_FEAT_DIM)
        self.bottleneck.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # timm swin with num_classes=0 returns (B, 768) global-avg-pooled features
        global_feat = self.base(x)
        # NECK_FEAT='before' → return global_feat (features before BN neck)
        return global_feat


# ---------------------------------------------------------------------------
# Checkpoint key remapping  (SOLIDER mmcv-swin → timm swin)
# ---------------------------------------------------------------------------

def _remap_key(k: str) -> str | None:
    """Remap one SOLIDER backbone key (with 'base.' stripped) to its timm equivalent.

    Returns None if the key should be dropped (no timm counterpart).
    """
    # --- semantic weight branches (SOLIDER-specific, no timm equivalent) ---
    if k.startswith("semantic_embed_w") or k.startswith("semantic_embed_b"):
        return None

    # --- patch_embed.projection → patch_embed.proj ---
    k = k.replace("patch_embed.projection.", "patch_embed.proj.")

    # --- norm{N} outside stages: only keep norm3 → norm ---
    #   (SOLIDER mmcv swin has norm0..norm3 per output stage; timm has a single
    #    final LayerNorm called 'norm' which corresponds to norm3)
    m = re.match(r"^norm(\d+)\.", k)
    if m:
        idx = int(m.group(1))
        if idx == 3:
            return "norm." + k[len(m.group(0)):]
        return None   # drop intermediate stage norms

    # --- stages.N → layers.N ---
    #   downsample: stages.N.downsample → layers.(N+1).downsample
    m = re.match(r"^stages\.(\d+)\.downsample\.(.*)", k)
    if m:
        n = int(m.group(1))
        rest = m.group(2)
        return f"layers.{n + 1}.downsample.{rest}"

    #   stage-level norms (stages.N.norm.*) — drop; timm handles them internally
    if re.match(r"^stages\.\d+\.norm\.", k):
        return None

    #   blocks: stages.N.blocks.M.* → layers.N.blocks.M.*
    m = re.match(r"^stages\.(\d+)\.blocks\.(\d+)\.(.*)", k)
    if m:
        n, b, rest = m.group(1), m.group(2), m.group(3)
        rest = _remap_block_key(rest)
        if rest is None:
            return None
        return f"layers.{n}.blocks.{b}.{rest}"

    # Everything else passes through unchanged
    return k


def _remap_block_key(rest: str) -> str | None:
    """Remap keys inside a single Swin block (SOLIDER → timm)."""
    # attn.w_msa.relative_position_bias_table → attn.relative_position_bias_table
    # attn.w_msa.relative_position_index      → attn.relative_position_index
    rest = rest.replace("attn.w_msa.relative_position_", "attn.relative_position_")
    # attn.w_msa.qkv  → attn.qkv
    # attn.w_msa.proj → attn.proj
    rest = rest.replace("attn.w_msa.", "attn.")

    # ffn.layers.0.0.* → mlp.fc1.*
    rest = re.sub(r"^ffn\.layers\.0\.0\.", "mlp.fc1.", rest)
    # ffn.layers.1.* → mlp.fc2.*
    rest = re.sub(r"^ffn\.layers\.1\.", "mlp.fc2.", rest)

    return rest


def _factor_table_size(L: int) -> tuple[int, int]:
    """Find (rows, cols) grid for a position bias table of size L.

    For a window of size (Wh, Ww), L = (2*Wh-1) * (2*Ww-1).
    We factor L into the closest-to-square pair whose factors are both odd
    (since 2*W-1 is always odd).

    Falls back to (int(sqrt(L)), L // int(sqrt(L))) if no clean factoring.
    """
    # Try all odd factors
    best = None
    for h in range(1, L + 1, 1):
        if L % h == 0:
            w = L // h
            if h % 2 == 1 and w % 2 == 1:
                if best is None or abs(h - w) < abs(best[0] - best[1]):
                    best = (h, w)
    if best is not None:
        return best
    s = int(math.sqrt(L))
    return s, (L // s)


def _interpolate_pos_bias(
    table_ckpt: torch.Tensor, table_model: torch.Tensor
) -> torch.Tensor:
    """Bicubic-interpolate a relative position bias table from ckpt size to model size.

    Both tensors have shape (L, num_heads).  L = (2*Wh-1)*(2*Ww-1) for a
    window of size (Wh, Ww).  When L values differ we bicubic-interpolate using
    the approach from SOLIDER/Swin's own init_weights.
    """
    L1, nH1 = table_ckpt.shape
    L2, nH2 = table_model.shape
    if L1 == L2:
        return table_ckpt
    if nH1 != nH2:
        warnings.warn(
            f"Cannot interpolate position bias: num_heads mismatch ({nH1} vs {nH2})",
            stacklevel=3,
        )
        return table_ckpt

    H1, W1 = _factor_table_size(L1)
    H2, W2 = _factor_table_size(L2)

    resized = F.interpolate(
        table_ckpt.permute(1, 0).reshape(1, nH1, H1, W1).float(),
        size=(H2, W2),
        mode="bicubic",
        align_corners=False,
    )
    return resized.reshape(nH2, L2).permute(1, 0).contiguous().to(table_ckpt.dtype)


def _build_timm_state_dict(
    ckpt: dict, model_sd: dict
) -> tuple[dict, list]:
    """Convert a SOLIDER checkpoint into a timm-compatible state dict.

    Args:
        ckpt:     raw SOLIDER checkpoint (OrderedDict).
        model_sd: current model's state_dict() (used for target shapes).

    Returns:
        (mapped_sd, skipped_keys)
    """
    mapped: dict[str, torch.Tensor] = {}
    skipped: list[str] = []

    for k, v in ckpt.items():
        if not k.startswith("base."):
            continue   # bottleneck, classifier — handled separately

        tk = _remap_key(k[5:])   # strip 'base.'
        if tk is None:
            skipped.append(k)
            continue

        # Skip relative_position_index (buffer — timm recomputes it)
        if "relative_position_index" in tk:
            skipped.append(k)
            continue

        # Interpolate position bias tables if sizes differ
        if "relative_position_bias_table" in tk and tk in model_sd:
            v = _interpolate_pos_bias(v, model_sd[tk])

        mapped[tk] = v

    return mapped, skipped


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_solider_swin_small(weights_path: str | Path, img_size=(256, 128)) -> nn.Module:
    """Load SOLIDER Swin-Small ReID model from a checkpoint file.

    Args:
        weights_path: Path to the .pth checkpoint (SOLIDER format).
        img_size:     (H, W) of the input images fed to this model.
                      Must match the T.Resize in _embed (default [256,128]).

    Returns:
        torch.nn.Module in eval mode (NOT on GPU yet — caller does .cuda()).

    Raises:
        NotImplementedError: if weights_path does not exist, with download instructions.
        RuntimeError: if the checkpoint cannot be loaded / keys are badly incompatible.
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise NotImplementedError(
            f"SOLIDER Swin-Small weights not found at '{weights_path}'.\n"
            "Download the MSMT17 Swin-Small checkpoint:\n"
            "  gdown 1C-aIZdFyjFsZX4W4feG-Ex39RU2Qvu3b -O weights/solider_swin_small.pth\n"
            "or see https://github.com/tinyvision/SOLIDER-REID for Market1501/other links.\n"
            "If SOLIDER make_model API changes, wire it in:\n"
            "  aic24_nvidia/models/solider/__init__.py :: load_solider_swin_small()"
        )

    model = _SOLIDERSwinSmall(img_size=img_size)
    model_sd = model.base.state_dict()

    raw_ckpt = torch.load(weights_path, map_location="cpu")
    # Normalise: accept bare state_dict or {'model': ...} / {'state_dict': ...}
    if isinstance(raw_ckpt, dict) and ("model" in raw_ckpt or "state_dict" in raw_ckpt):
        ckpt = raw_ckpt.get("model", raw_ckpt.get("state_dict"))
    else:
        ckpt = raw_ckpt

    # --- Load backbone (with key remapping + position bias interpolation) ---
    backbone_sd, skipped = _build_timm_state_dict(ckpt, model_sd)
    missing_b, unexpected_b = model.base.load_state_dict(backbone_sd, strict=False)

    if missing_b:
        warnings.warn(
            f"SOLIDER Swin-Small backbone: {len(missing_b)} missing keys "
            f"(e.g. {missing_b[:3]}). This may indicate a checkpoint format mismatch.",
            stacklevel=2,
        )
    # relative_position_index is already skipped in _build_timm_state_dict;
    # any remaining unexpected keys are genuinely unexpected.
    if unexpected_b:
        warnings.warn(
            f"SOLIDER Swin-Small backbone: {len(unexpected_b)} unexpected keys "
            f"(e.g. {unexpected_b[:3]}).",
            stacklevel=2,
        )

    # --- Load BN neck ---
    neck_sd = {k.replace("bottleneck.", ""): v
               for k, v in ckpt.items() if k.startswith("bottleneck.")}
    if neck_sd:
        model.bottleneck.load_state_dict(neck_sd, strict=True)

    model.eval()
    return model
