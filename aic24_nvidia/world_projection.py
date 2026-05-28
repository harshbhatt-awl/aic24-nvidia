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
from pathlib import Path
from typing import Iterable


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
