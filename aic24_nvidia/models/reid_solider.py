"""SOLIDER Swin-Small ReID adapter — byte-compatible replacement for
external/deep-person-reid/torchreid/aic24_extract.py.

Key contract (mirrors upstream exactly):
  • Reads Detection/<scene>/<cam>.txt  (comma-sep, columns: cam,frame,cls,x1,y1,x2,y2,conf)
  • Reads Detection/<scene>/<cam>.json
  • For each detection at 0-based index idx:
      - u_num increments at TOP of loop (first detection → u_num=1)
      - filename: feature_<cur_frame>_<u_num>_<x1>_<x2>_<y1>_<y2>_<conf>.npy
          where coords are int(float(coord)), order is x1,x2,y1,y2 (x's then y's)
          and conf is the raw conf STRING with '.' removed
      - Saves a 1-D float32 .npy to EmbedFeature/<scene>/<cam>/<filename>
      - Updates json: jf[str(idx).zfill(8)]["NpyPath"] = "<scene>/<cam>/<filename>"
  • Writes updated json back.
"""

from __future__ import annotations
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

_MODEL = None
_TRANSFORM = None


def _get_transform():
    global _TRANSFORM
    if _TRANSFORM is None:
        import torchvision.transforms as T  # type: ignore
        from aic24_nvidia.models.solider import SOLIDER_SIZE, SOLIDER_MEAN, SOLIDER_STD

        _TRANSFORM = T.Compose([
            T.Resize(list(SOLIDER_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=list(SOLIDER_MEAN), std=list(SOLIDER_STD)),
        ])
    return _TRANSFORM


def _embed(crop: Image.Image) -> np.ndarray:
    """Embed a PIL crop → 1-D float32 vector via SOLIDER Swin-Small (Part B).

    Lazily loads the model on first call from weights/solider_swin_small.pth.
    Raises NotImplementedError if the weights file is absent (see
    aic24_nvidia/models/solider/__init__.py for download instructions).
    """
    import torch

    global _MODEL
    if _MODEL is None:
        from aic24_nvidia.models.solider import load_solider_swin_small

        _weights = Path(__file__).parent.parent.parent / "weights" / "solider_swin_small.pth"
        _MODEL = load_solider_swin_small(_weights)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _MODEL.eval().to(dev)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = _get_transform()(crop.convert("RGB")).unsqueeze(0).to(dev)
    with torch.no_grad():
        feat = _MODEL(x)
    return feat.cpu().numpy()[0].astype(np.float32)


def extract_camera(
    det_scene_dir,
    original_scene_dir,
    emb_out_dir,
    scene: str,
    cam: str,
    embed=None,
) -> None:
    """Extract and save ReID embeddings for one camera.

    Args:
        det_scene_dir:      Path to Detection/<scene>/ (contains <cam>.txt and <cam>.json).
        original_scene_dir: Path to Original/<scene>/ (contains <cam>/Frame/*.jpg).
        emb_out_dir:        Root output dir; files written to <emb_out_dir>/<scene>/<cam>/.
        scene:              Scene name string (used in NpyPath values).
        cam:                Camera name string (e.g. "camera_0390").
        embed:              Callable(PIL.Image) → np.ndarray[float32].  When None, uses the
                            module-global _embed (lazy SOLIDER loader).  Pass explicitly to
                            inject a mock during testing.
    """
    # NOTE: we use the PASSED embed callable directly (not the global) so that
    # monkeypatching reid_solider._embed and then passing reid_solider._embed as
    # the embed argument both work correctly.
    if embed is None:
        embed = _embed

    det_scene_dir = Path(det_scene_dir)
    dets = np.genfromtxt(det_scene_dir / f"{cam}.txt", dtype=str, delimiter=",")
    if dets.ndim == 1 and dets.shape[0] == 0:
        # Empty file — nothing to embed, NpyPaths unchanged, nothing to write back
        return
    if dets.ndim == 1:
        # Single detection row → reshape to (1, N)
        dets = dets.reshape(1, -1)

    json_path = det_scene_dir / f"{cam}.json"
    with open(json_path) as f:
        jf = json.load(f)

    out = Path(emb_out_dir) / scene / cam
    out.mkdir(parents=True, exist_ok=True)

    u_num = 0
    for idx, row in enumerate(dets):
        _cam, frame, _cls, x1, y1, x2, y2, conf = row
        u_num += 1  # incremented at TOP of loop; first detection → u_num=1

        cur_frame = int(frame)
        # coord order in filename: x1, x2, y1, y2  (x's then y's — matches upstream)
        xi1 = int(float(x1))
        xi2 = int(float(x2))
        yi1 = int(float(y1))
        yi2 = int(float(y2))
        conf_str = str(conf).replace(".", "")

        fname = f"feature_{cur_frame}_{u_num}_{xi1}_{xi2}_{yi1}_{yi2}_{conf_str}.npy"

        # Crop source image
        img_path = (
            Path(original_scene_dir) / cam / "Frame" / (frame.zfill(6) + ".jpg")
        )
        crop = Image.open(img_path).crop((float(x1), float(y1), float(x2), float(y2)))

        # Embed and save
        np.save(out / fname, embed(crop))

        # Update json
        jf[str(idx).zfill(8)]["NpyPath"] = os.path.join(scene, cam, fname)

    with open(json_path, "w") as f:
        json.dump(jf, f, ensure_ascii=False)


def _release_gpu():
    global _MODEL
    _MODEL = None
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_reid(
    det_scene_dir,
    original_scene_dir,
    emb_out_dir,
    scene: str,
    cams: list[str],
) -> None:
    """Run ReID embedding extraction for all cameras in a scene."""
    for cam in cams:
        extract_camera(det_scene_dir, original_scene_dir, emb_out_dir, scene, cam)
    _release_gpu()
