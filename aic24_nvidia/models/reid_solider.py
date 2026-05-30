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

class SoliderReID:
    """Default ReIDBackend: SOLIDER Swin-Small, 768-d embeddings.

    Mirrors _get_transform + _embed's lazy load. Default weights resolve to
    <weights_root>/solider_swin_small.pth (same file the module-relative path
    used when run from the repo root).
    """

    def __init__(self) -> None:
        self._model = None
        self._transform = None

    def load(self, cfg, weights_root) -> None:
        import torch
        import torchvision.transforms as T
        from aic24_nvidia.models.solider import (
            SOLIDER_MEAN, SOLIDER_SIZE, SOLIDER_STD, load_solider_swin_small,
        )
        weights = weights_root / (cfg.weights or "solider_swin_small.pth")
        self._model = load_solider_swin_small(weights)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.eval().to(dev)
        self._transform = T.Compose([
            T.Resize(list(SOLIDER_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=list(SOLIDER_MEAN), std=list(SOLIDER_STD)),
        ])

    def embed(self, crop):
        import torch
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = self._transform(crop.convert("RGB")).unsqueeze(0).to(dev)
        with torch.no_grad():
            feat = self._model(x)
        return feat.cpu().numpy()[0].astype(np.float32)

    def teardown(self) -> None:
        self._model = None
        self._transform = None
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def extract_camera(
    det_scene_dir,
    original_scene_dir,
    emb_out_dir,
    scene: str,
    cam: str,
    embed,
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


def run_reid(det_scene_dir, original_scene_dir, emb_out_dir, scene, cams,
             cfg, weights_root, backend=None):
    """Run ReID embedding extraction for all cameras in a scene.

    backend: a ReIDBackend. When None, resolved from cfg.model_name. Inject a
             fake in tests.
    """
    if backend is None:
        from .registry import get_reid
        backend = get_reid(cfg.model_name)
    backend.load(cfg, weights_root)
    try:
        for cam in cams:
            extract_camera(det_scene_dir, original_scene_dir, emb_out_dir,
                           scene, cam, embed=backend.embed)
    finally:
        backend.teardown()
