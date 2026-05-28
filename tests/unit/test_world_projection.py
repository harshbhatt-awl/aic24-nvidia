"""Unit tests for aic24_nvidia.world_projection."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_pose_json(path: Path, body: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body))


def test_build_pose_lookup_indexes_by_frame_and_bbox(tmp_path):
    """Pose lookup table is keyed by (frame_int, (x1,y1,x2,y2)) tuple."""
    from aic24_nvidia.world_projection import _build_pose_lookup

    pose = {
        "5": [
            {"bbox": [100, 200, 150, 350, 1.0],
             "keypoints": [[0, 0, 0.0]] * 17},
            {"bbox": [400, 500, 460, 700, 1.0],
             "keypoints": [[0, 0, 0.0]] * 17},
        ],
        "6": [
            {"bbox": [110, 210, 160, 360, 1.0],
             "keypoints": [[0, 0, 0.0]] * 17},
        ],
    }
    pose_path = tmp_path / "camera_0390_out_keypoint.json"
    _write_pose_json(pose_path, pose)

    lookup = _build_pose_lookup(pose_path)

    assert (5, (100, 200, 150, 350)) in lookup
    assert (5, (400, 500, 460, 700)) in lookup
    assert (6, (110, 210, 160, 360)) in lookup
    assert len(lookup) == 3
    # Values are the keypoints list (17 entries).
    assert len(lookup[(5, (100, 200, 150, 350))]) == 17


def test_build_pose_lookup_empty_file(tmp_path):
    from aic24_nvidia.world_projection import _build_pose_lookup
    pose_path = tmp_path / "empty.json"
    _write_pose_json(pose_path, {})
    assert _build_pose_lookup(pose_path) == {}
