from __future__ import annotations
import json
from pathlib import Path


def adapt_calibration(src: Path, dst: Path, scene_mapping: dict[str, str]) -> None:
    """Pass NVIDIA-format calibration.json through to dst.

    Real NVIDIA Warehouse scenes (as pre-processed by the AIC23 sibling project)
    store calibration as `{"cameras": {camera_NNNN: {K, R, t}, ...}}`. Since our
    adapter preserves camera names (identity mapping), this is mostly a copy with
    a check that every camera listed in `scene_mapping` is present.

    scene_mapping is: {yachiyo_cam_name: source_cam_name}. With identity mapping
    these are equal, but we still validate presence.

    For backwards compatibility with the older `{camera_NNNN: {intrinsicMatrix: ...}}`
    schema (used in synthetic test fixtures), accept that form too.
    """
    src = Path(src)
    dst = Path(dst)
    body = json.loads(src.read_text())

    # Detect schema: real has top-level "cameras" key wrapping per-camera dicts.
    if isinstance(body, dict) and "cameras" in body and isinstance(body["cameras"], dict):
        per_cam = body["cameras"]
        wrap_under_cameras = True
    else:
        per_cam = body
        wrap_under_cameras = False

    out_per_cam: dict = {}
    for yachiyo, source in scene_mapping.items():
        if source not in per_cam:
            raise KeyError(f"calibration missing camera: {source}")
        out_per_cam[yachiyo] = per_cam[source]

    out: dict = {"cameras": out_per_cam} if wrap_under_cameras else out_per_cam
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2))
