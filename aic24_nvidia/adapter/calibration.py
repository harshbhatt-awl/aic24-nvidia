from __future__ import annotations
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def write_per_camera_calibration(
    calib_body: dict,
    original_scene_dir: Path,
    cameras: list[str],
) -> list[Path]:
    """Write per-camera calibration.json files under original_scene_dir/cam/.

    Each file contains:
      - "camera projection matrix": 3x4 nested list  (P = K @ [R | t])
      - "homography matrix":        3x3 nested list  (H = K @ [r1 | r2 | t], Z=0 plane)

    This is the exact format expected by YACHIYO's tracking/src/utils.py
    load_calibration() at lines 158-163.

    Args:
        calib_body: NVIDIA calibration dict with {"cameras": {cam_name: {K, R, t}}}
        original_scene_dir: Path to Original/scene_NNN/ (camera dirs already exist)
        cameras: list of camera names to process (e.g. ["camera_0390", ...])

    Returns:
        List of paths to written calibration.json files.
    """
    import numpy as np

    per_cam = calib_body.get("cameras", {})
    written: list[Path] = []

    for cam in cameras:
        if cam not in per_cam:
            log.warning(
                "write_per_camera_calibration: camera %s missing from calib_body, skipping",
                cam,
            )
            continue

        cam_data = per_cam[cam]
        missing = [k for k in ("K", "R", "t") if k not in cam_data]
        if missing:
            log.warning(
                "write_per_camera_calibration: camera %s missing keys %s, skipping",
                cam, missing,
            )
            continue

        K = np.array(cam_data["K"], dtype=float)
        R = np.array(cam_data["R"], dtype=float)
        t = np.array(cam_data["t"], dtype=float).reshape(3)

        # P = K @ [R | t]  →  3x4 projection matrix
        Rt = np.hstack([R, t.reshape(3, 1)])  # 3x4
        P = K @ Rt                             # 3x4

        # H = K @ [r1 | r2 | t]  →  3x3 ground-plane (Z=0) homography
        # (world X,Y → image u,v for points on the Z=0 floor)
        H_inner = np.column_stack([R[:, 0], R[:, 1], t])  # 3x3
        H = K @ H_inner                                    # 3x3

        out = {
            "camera projection matrix": P.tolist(),
            "homography matrix": H.tolist(),
        }

        cam_dir = original_scene_dir / cam
        cam_dir.mkdir(parents=True, exist_ok=True)
        dst = cam_dir / "calibration.json"
        dst.write_text(json.dumps(out, indent=2))
        written.append(dst)

    return written


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
