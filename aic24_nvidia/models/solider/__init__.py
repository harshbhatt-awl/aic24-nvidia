"""SOLIDER Swin-Small ReID wrapper.

SOLIDER upstream: https://github.com/tinyvision/SOLIDER-REID

We run SOLIDER's *own* Swin backbone (vendored verbatim as ``swin_transformer.py``)
rather than a re-implementation, so the forward graph — including the
``semantic_embed`` modulation that controls SOLIDER's appearance/semantic
trade-off — matches the trained checkpoint exactly. The only edits to the
vendored file are: the single unused ``mmcv.runner.load_checkpoint`` import is
stubbed out, and a hardcoded ``.cuda()`` is made device-aware. No mmcv needed.

Inference config (from configs/msmt17/swin_small.yml + config/defaults.py):
  • SIZE_TEST      = (384, 128)
  • PIXEL_MEAN/STD = 0.5 / 0.5   (NOT ImageNet)
  • SEMANTIC_WEIGHT = 1.0
  • NECK_FEAT      = 'after'     → return the BN-neck feature.

NOTE on NECK_FEAT: SOLIDER's own retrieval eval uses 'before' (pre-BN) with a
Euclidean + re-ranking metric. Our downstream YACHIYO tracker compares features
by **cosine** similarity, and the pre-BN feature carries a large common-mode that
inflates cosine to ~0.94 between *different* people. The BN neck centres the
feature, restoring discriminative cosine (different-person ≈ 0.16, same-person
≈ 0.38 on Warehouse_001 GT crops) — so we use 'after' for this pipeline.

A previous adapter ported the weights into a timm Swin; that silently dropped
``semantic_embed`` and used the wrong normalization/size, collapsing every
embedding to ~constant (random-pair cosine ≈ 0.99) and destroying tracking.
See tests/unit/test_reid_solider_discriminative.py for the regression gate.

Weight file: weights/solider_swin_small.pth
  gdown 1C-aIZdFyjFsZX4W4feG-Ex39RU2Qvu3b -O weights/solider_swin_small.pth
"""

from __future__ import annotations
import warnings
from pathlib import Path

import torch
import torch.nn as nn

__all__ = ["load_solider_swin_small", "SOLIDER_SIZE", "SOLIDER_MEAN", "SOLIDER_STD"]

_SWIN_SMALL_FEAT_DIM = 768

# Inference preprocessing constants (mirror SOLIDER's msmt17/swin_small.yml).
SOLIDER_SIZE = (384, 128)
SOLIDER_MEAN = (0.5, 0.5, 0.5)
SOLIDER_STD = (0.5, 0.5, 0.5)


class _SOLIDERSwinSmall(nn.Module):
    """SOLIDER's native Swin-Small backbone + BN neck.

    Returns the global feature BEFORE the BN neck (NECK_FEAT='before'), matching
    the upstream test config. Output shape: (B, 768).
    """

    def __init__(self, img_size=SOLIDER_SIZE, semantic_weight: float = 1.0,
                 neck_feat: str = "after") -> None:
        super().__init__()
        from .swin_transformer import swin_small_patch4_window7_224

        self.base = swin_small_patch4_window7_224(
            img_size=img_size, semantic_weight=semantic_weight
        )
        self.bottleneck = nn.BatchNorm1d(_SWIN_SMALL_FEAT_DIM)
        self.bottleneck.bias.requires_grad_(False)
        self.neck_feat = neck_feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        global_feat, _ = self.base(x)            # (B, 768), avg-pooled
        if self.neck_feat == "after":
            return self.bottleneck(global_feat)
        return global_feat


def load_solider_swin_small(
    weights_path: str | Path,
    img_size=SOLIDER_SIZE,
    semantic_weight: float = 1.0,
    neck_feat: str = "after",
) -> nn.Module:
    """Load SOLIDER Swin-Small from a native SOLIDER checkpoint.

    The checkpoint keys (``base.*``, ``bottleneck.*``, ``classifier.*``) map
    directly onto this module — no remapping — because we use SOLIDER's own
    backbone. ``classifier.*`` (training head) is dropped.

    Raises:
        NotImplementedError: if weights_path does not exist (with download hint).
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise NotImplementedError(
            f"SOLIDER Swin-Small weights not found at '{weights_path}'.\n"
            "  gdown 1C-aIZdFyjFsZX4W4feG-Ex39RU2Qvu3b -O weights/solider_swin_small.pth\n"
            "or see https://github.com/tinyvision/SOLIDER-REID for other links."
        )

    model = _SOLIDERSwinSmall(
        img_size=img_size, semantic_weight=semantic_weight, neck_feat=neck_feat
    )

    raw = torch.load(weights_path, map_location="cpu")
    if isinstance(raw, dict) and ("model" in raw or "state_dict" in raw):
        ckpt = raw.get("model", raw.get("state_dict"))
    else:
        ckpt = raw

    # Native keys: keep only backbone + BN neck; drop the training classifier head.
    sd = {k: v for k, v in ckpt.items()
          if k.startswith("base.") or k.startswith("bottleneck.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)

    # relative_position_index / attn_mask are recomputed buffers — ignore them.
    missing = [k for k in missing
               if "relative_position_index" not in k and "attn_mask" not in k]
    unexpected = [k for k in unexpected
                  if "relative_position_index" not in k and "attn_mask" not in k]
    if missing:
        warnings.warn(
            f"SOLIDER Swin-Small: {len(missing)} missing keys (e.g. {missing[:3]}).",
            stacklevel=2,
        )
    if unexpected:
        warnings.warn(
            f"SOLIDER Swin-Small: {len(unexpected)} unexpected keys "
            f"(e.g. {unexpected[:3]}).",
            stacklevel=2,
        )

    model.eval()
    return model
