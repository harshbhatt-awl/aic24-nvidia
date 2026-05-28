"""Override per-detection WorldCoordinate in SCT JSONs using pose ankle keypoints.

Called from stages/mct.py after the SCT outputs are staged into mct.tmp/ and
before the upstream MCT subprocess runs. Rewrites are in place and idempotent.

Method | What gets projected to world coords
-------|------------------------------------
bbox_bottom        | ((x1+x2)/2, y2)                                — no-op
ankle_avg          | score-weighted mean of left_ankle, right_ankle
ankle_lower        | the ankle with the larger pixel y (planted foot)
ankle_w_fallback   | ankle_avg if both ankle scores >= ankle_min_conf, else bbox_bottom

SCPT does not consume WorldCoordinate (verified via grep on
external/.../tracking/src/scpt.py), so doing this between SCT and MCT does not
change SCT decisions — only MCT clustering and the final eval see the override.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


# COCO-17 ordering used by RTMPose (see aic24_nvidia/models/pose_rtmpose.py:23).
COCO_LEFT_ANKLE = 15
COCO_RIGHT_ANKLE = 16


def _project_to_world(x_img: float, y_img: float, homography_matrix) -> tuple[float, float]:
    """Project an image-plane point (pixels) to world coordinates (metres).

    Matches the formula used in external/.../tracking/src/utils.py:170 —
    world->image homography H means image->world is inv(H) applied to
    [x, y, 1], with the result divided by its third component.
    """
    import numpy as np  # local import keeps the module import-time cheap

    H_inv = np.linalg.inv(np.asarray(homography_matrix, dtype=np.float64))
    v = H_inv @ np.array([x_img, y_img, 1.0])
    return float(v[0] / v[2]), float(v[1] / v[2])


def _build_pose_lookup(pose_json: Path) -> dict[tuple[int, tuple[int, int, int, int]], list[list[float]]]:
    """Index a per-camera pose JSON by (frame_int, bbox_ints) -> keypoints (17 x [x,y,score])."""
    body = json.loads(Path(pose_json).read_text())
    out: dict[tuple[int, tuple[int, int, int, int]], list[list[float]]] = {}
    for frame_str, entries in body.items():
        frame = int(frame_str)
        if not isinstance(entries, list):
            continue
        for e in entries:
            if not isinstance(e, dict):
                continue
            bbox = e.get("bbox")
            kps = e.get("keypoints")
            if not bbox or kps is None or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            out[(frame, (x1, y1, x2, y2))] = kps
    return out


def _bbox_bottom_point(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, _y1, x2, y2 = bbox
    return (float(x1) + float(x2)) / 2.0, float(y2)


def _compute_image_point(
    bbox: tuple[int, int, int, int],
    kps: list[list[float]] | None,
    method: str,
    ankle_min_conf: float,
) -> tuple[float, float]:
    """Pick the (x_img, y_img) pixel point to project to world coords.

    Falls back to bbox_bottom whenever the chosen method cannot be applied
    (no pose, both ankle scores zero, or low-confidence in ankle_w_fallback).
    """
    if method == "bbox_bottom" or kps is None:
        return _bbox_bottom_point(bbox)

    lx, ly, ls = kps[COCO_LEFT_ANKLE]
    rx, ry, rs = kps[COCO_RIGHT_ANKLE]

    if method == "ankle_avg":
        total = ls + rs
        if total <= 0.0:
            return _bbox_bottom_point(bbox)
        return (ls * lx + rs * rx) / total, (ls * ly + rs * ry) / total

    if method == "ankle_lower":
        if ls <= 0.0 and rs <= 0.0:
            return _bbox_bottom_point(bbox)
        # Pick the ankle with larger pixel y. Skip ankles with score 0.
        candidates = []
        if ls > 0.0:
            candidates.append((ly, lx, ly))
        if rs > 0.0:
            candidates.append((ry, rx, ry))
        # Sort by y descending, then take the first.
        candidates.sort(key=lambda t: -t[0])
        _y_key, cx, cy = candidates[0]
        return float(cx), float(cy)

    if method == "ankle_w_fallback":
        if ls >= ankle_min_conf and rs >= ankle_min_conf:
            total = ls + rs
            return (ls * lx + rs * rx) / total, (ls * ly + rs * ry) / total
        return _bbox_bottom_point(bbox)

    # Unknown method (defensive — config validation prevents this).
    return _bbox_bottom_point(bbox)


def _load_homography(calib_path: Path):
    import numpy as np
    body = json.loads(Path(calib_path).read_text())
    return np.array(body["homography matrix"], dtype=np.float64)


def _rewrite_one_file(
    sct_json: Path,
    pose_lookup: dict,
    homography,
    method: str,
    ankle_min_conf: float,
) -> int:
    """Rewrite `WorldCoordinate` for every detection in one SCT JSON. Returns count."""
    body = json.loads(sct_json.read_text())
    n = 0
    for _serial, entry in body.items():
        if not isinstance(entry, dict):
            continue
        coord = entry.get("Coordinate")
        frame = entry.get("Frame")
        if coord is None or frame is None:
            continue
        try:
            x1 = int(round(float(coord["x1"])))
            y1 = int(round(float(coord["y1"])))
            x2 = int(round(float(coord["x2"])))
            y2 = int(round(float(coord["y2"])))
            frame_i = int(frame)
        except (KeyError, TypeError, ValueError):
            continue

        kps = pose_lookup.get((frame_i, (x1, y1, x2, y2)))
        x_img, y_img = _compute_image_point((x1, y1, x2, y2), kps, method, ankle_min_conf)
        wx, wy = _project_to_world(x_img, y_img, homography)
        # Skip NaN/inf — keep the original WorldCoordinate if projection blew up.
        import math
        if not (math.isfinite(wx) and math.isfinite(wy)):
            continue
        entry["WorldCoordinate"] = {"x": wx, "y": wy}
        n += 1
    sct_json.write_text(json.dumps(body))
    return n


def rewrite_world_coordinates(
    *,
    sct_scene_dir: Path,
    pose_scene_dir: Path,
    calib_root: Path,
    camera_map: dict[int, str],
    method: str,
    ankle_min_conf: float,
) -> int:
    """Rewrite `WorldCoordinate` in every per-camera SCT JSON in place.

    Args:
        sct_scene_dir: e.g. `mct.tmp/scene_001/` — contains
            `camera{N}_tracking_results.json` and
            `fixed_camera{N}_tracking_results.json`.
        pose_scene_dir: e.g. `Pose/scene_001/` — contains
            `<nvidia_cam>/<nvidia_cam>_out_keypoint.json`.
        calib_root: e.g. `adapted/Original/scene_001/` — contains
            `<nvidia_cam>/calibration.json` with `"homography matrix"`.
        camera_map: numeric_id -> nvidia_cam_name, e.g. `{390: "camera_0390"}`.
        method: one of `bbox_bottom | ankle_avg | ankle_lower | ankle_w_fallback`.
        ankle_min_conf: per-keypoint confidence floor used by ankle_w_fallback.

    Returns:
        Total number of detection rewrites across all files. When
        `method == "bbox_bottom"` this is a true no-op and returns 0.
    """
    sct_scene_dir = Path(sct_scene_dir)
    pose_scene_dir = Path(pose_scene_dir)
    calib_root = Path(calib_root)

    if method == "bbox_bottom":
        return 0   # no-op contract; baseline must be byte-identical

    total = 0
    for cam_id, nvidia_name in camera_map.items():
        pose_json = pose_scene_dir / nvidia_name / f"{nvidia_name}_out_keypoint.json"
        calib_json = calib_root / nvidia_name / "calibration.json"
        if not pose_json.exists() or not calib_json.exists():
            missing = []
            if not pose_json.exists():
                missing.append(f"pose={pose_json}")
            if not calib_json.exists():
                missing.append(f"calibration={calib_json}")
            log.warning(
                "world_projection: skipping camera %s (method=%s) — missing: %s. "
                "This camera keeps SCT-derived (bbox-bottom) world coords while other cameras "
                "may get ankle-projected coords; results may be inconsistent across cameras.",
                nvidia_name, method, ", ".join(missing),
            )
            continue

        pose_lookup = _build_pose_lookup(pose_json)
        homography = _load_homography(calib_json)

        for stem in (f"camera{cam_id:03d}_tracking_results.json",
                     f"fixed_camera{cam_id:03d}_tracking_results.json"):
            sct_json = sct_scene_dir / stem
            if not sct_json.exists():
                continue
            total += _rewrite_one_file(
                sct_json, pose_lookup, homography, method, ankle_min_conf,
            )
    return total
