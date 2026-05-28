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
