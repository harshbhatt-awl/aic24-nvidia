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


@pytest.fixture
def _kps_two_ankles():
    """17-keypoint list with left_ankle=(100,800,0.9), right_ankle=(110,810,0.8)."""
    kps = [[0.0, 0.0, 0.0]] * 17
    kps = list(kps)
    kps[15] = [100.0, 800.0, 0.9]   # left_ankle
    kps[16] = [110.0, 810.0, 0.8]   # right_ankle
    return kps


def test_compute_image_point_bbox_bottom(_kps_two_ankles):
    from aic24_nvidia.world_projection import _compute_image_point

    bbox = (50, 100, 150, 850)   # bottom-center = (100, 850)
    # bbox_bottom must ignore keypoints and use ((x1+x2)/2, y2).
    x, y = _compute_image_point(bbox, kps=None, method="bbox_bottom", ankle_min_conf=0.3)
    assert (x, y) == pytest.approx((100.0, 850.0))


def test_compute_image_point_ankle_avg_score_weighted(_kps_two_ankles):
    from aic24_nvidia.world_projection import _compute_image_point

    bbox = (50, 100, 150, 850)
    # weighted avg of (100,800)@0.9 and (110,810)@0.8 =
    #   x = (0.9*100 + 0.8*110) / (0.9+0.8) = (90+88)/1.7 ≈ 104.7058...
    #   y = (0.9*800 + 0.8*810) / 1.7        = (720+648)/1.7 ≈ 804.7058...
    x, y = _compute_image_point(bbox, kps=_kps_two_ankles, method="ankle_avg",
                                ankle_min_conf=0.3)
    assert x == pytest.approx((0.9 * 100 + 0.8 * 110) / 1.7)
    assert y == pytest.approx((0.9 * 800 + 0.8 * 810) / 1.7)


def test_compute_image_point_ankle_avg_zero_scores_falls_back(_kps_two_ankles):
    """If both ankle scores are exactly zero, fall back to bbox_bottom."""
    from aic24_nvidia.world_projection import _compute_image_point

    kps = list(_kps_two_ankles)
    kps[15] = [100.0, 800.0, 0.0]
    kps[16] = [110.0, 810.0, 0.0]
    bbox = (50, 100, 150, 850)
    x, y = _compute_image_point(bbox, kps=kps, method="ankle_avg", ankle_min_conf=0.3)
    assert (x, y) == pytest.approx((100.0, 850.0))


def test_compute_image_point_ankle_lower(_kps_two_ankles):
    """ankle_lower picks the ankle with larger pixel y (the planted foot)."""
    from aic24_nvidia.world_projection import _compute_image_point

    bbox = (50, 100, 150, 850)
    # right_ankle at y=810 > left_ankle at y=800, so right_ankle wins.
    x, y = _compute_image_point(bbox, kps=_kps_two_ankles, method="ankle_lower",
                                ankle_min_conf=0.3)
    assert (x, y) == pytest.approx((110.0, 810.0))


def test_compute_image_point_ankle_w_fallback_uses_avg_when_confident(_kps_two_ankles):
    """ankle_w_fallback with both scores >= threshold uses ankle_avg."""
    from aic24_nvidia.world_projection import _compute_image_point

    bbox = (50, 100, 150, 850)
    x, y = _compute_image_point(bbox, kps=_kps_two_ankles, method="ankle_w_fallback",
                                ankle_min_conf=0.3)
    # Same as ankle_avg.
    assert x == pytest.approx((0.9 * 100 + 0.8 * 110) / 1.7)
    assert y == pytest.approx((0.9 * 800 + 0.8 * 810) / 1.7)


def test_compute_image_point_ankle_w_fallback_falls_back_when_low_conf():
    """ankle_w_fallback below threshold reverts to bbox_bottom."""
    from aic24_nvidia.world_projection import _compute_image_point

    kps = [[0.0, 0.0, 0.0]] * 17
    kps[15] = [100.0, 800.0, 0.1]  # below 0.3
    kps[16] = [110.0, 810.0, 0.2]  # below 0.3
    bbox = (50, 100, 150, 850)
    x, y = _compute_image_point(bbox, kps=kps, method="ankle_w_fallback",
                                ankle_min_conf=0.3)
    assert (x, y) == pytest.approx((100.0, 850.0))


def test_compute_image_point_no_pose_falls_back():
    """If pose lookup misses (kps is None) the method always falls back to bbox_bottom."""
    from aic24_nvidia.world_projection import _compute_image_point

    bbox = (50, 100, 150, 850)
    for method in ("ankle_avg", "ankle_lower", "ankle_w_fallback"):
        x, y = _compute_image_point(bbox, kps=None, method=method, ankle_min_conf=0.3)
        assert (x, y) == pytest.approx((100.0, 850.0)), f"method={method}"
