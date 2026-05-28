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


def test_project_to_world_identity_homography():
    """Identity homography means world = image coordinates."""
    import numpy as np
    from aic24_nvidia.world_projection import _project_to_world

    H = np.eye(3)
    wx, wy = _project_to_world(123.0, 456.0, H)
    assert wx == pytest.approx(123.0)
    assert wy == pytest.approx(456.0)


def test_project_to_world_translation_homography():
    """Pure translation homography shifts world coords."""
    import numpy as np
    from aic24_nvidia.world_projection import _project_to_world

    # world->image homography that adds (10, 20) when going world->image.
    # So image -> world subtracts (10, 20). H = [[1,0,10],[0,1,20],[0,0,1]].
    H = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]])
    wx, wy = _project_to_world(100.0, 200.0, H)
    assert wx == pytest.approx(90.0)
    assert wy == pytest.approx(180.0)
