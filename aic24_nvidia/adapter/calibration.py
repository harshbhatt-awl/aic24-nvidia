from __future__ import annotations
import json
from pathlib import Path


def adapt_calibration(src: Path, dst: Path, scene_mapping: dict[str, str]) -> None:
    """Pass NVIDIA calibration.json through, renaming Camera_NNNN keys
    to YACHIYO camera_nnnn keys per scene_mapping.

    scene_mapping is: {yachiyo_cam_name: nvidia_cam_name}
    """
    src = Path(src)
    dst = Path(dst)
    body = json.loads(src.read_text())
    out: dict = {}
    for yachiyo, nvidia in scene_mapping.items():
        if nvidia not in body:
            raise KeyError(f"calibration missing camera: {nvidia}")
        out[yachiyo] = body[nvidia]
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2))
