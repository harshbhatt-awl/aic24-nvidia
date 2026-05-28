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


def _write_sct_json(path: Path, body: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body))


def _identity_calib_json(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "camera projection matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
        "homography matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    }))


def test_rewrite_bbox_bottom_is_noop(tmp_path):
    """method=bbox_bottom leaves WorldCoordinate untouched (identity contract)."""
    from aic24_nvidia.world_projection import rewrite_world_coordinates

    sct_scene = tmp_path / "mct.tmp" / "scene_001"
    pose_scene = tmp_path / "Pose" / "scene_001"
    orig_scene = tmp_path / "adapted" / "Original" / "scene_001"

    sct_json = sct_scene / "camera390_tracking_results.json"
    original_body = {
        "00000001": {
            "Frame": 5, "NpyPath": "x",
            "Coordinate": {"x1": 50, "y1": 100, "x2": 150, "y2": 850},
            "WorldCoordinate": {"x": 7.0, "y": 11.0},   # arbitrary; should be preserved
            "OfflineID": 0,
        },
    }
    _write_sct_json(sct_json, original_body)

    pose_json = pose_scene / "camera_0390" / "camera_0390_out_keypoint.json"
    _write_pose_json(pose_json, {
        "5": [{"bbox": [50, 100, 150, 850, 1.0],
               "keypoints": [[0.0, 0.0, 0.0]] * 17}]
    })
    _identity_calib_json(orig_scene / "camera_0390" / "calibration.json")

    rewritten = rewrite_world_coordinates(
        sct_scene_dir=sct_scene,
        pose_scene_dir=pose_scene,
        calib_root=orig_scene,
        camera_map={390: "camera_0390"},
        method="bbox_bottom",
        ankle_min_conf=0.3,
    )

    assert rewritten == 0   # no detections were rewritten
    after = json.loads(sct_json.read_text())
    assert after["00000001"]["WorldCoordinate"] == {"x": 7.0, "y": 11.0}


def test_rewrite_ankle_avg_with_identity_homography(tmp_path):
    """method=ankle_avg with identity homography projects ankles directly."""
    from aic24_nvidia.world_projection import rewrite_world_coordinates

    sct_scene = tmp_path / "mct.tmp" / "scene_001"
    pose_scene = tmp_path / "Pose" / "scene_001"
    orig_scene = tmp_path / "adapted" / "Original" / "scene_001"

    sct_json = sct_scene / "camera390_tracking_results.json"
    _write_sct_json(sct_json, {
        "00000001": {
            "Frame": 5, "NpyPath": "x",
            "Coordinate": {"x1": 50, "y1": 100, "x2": 150, "y2": 850},
            "WorldCoordinate": {"x": -999.0, "y": -999.0},
            "OfflineID": 0,
        },
    })
    pose_json = pose_scene / "camera_0390" / "camera_0390_out_keypoint.json"
    kps = [[0.0, 0.0, 0.0]] * 17
    kps[15] = [100.0, 800.0, 1.0]
    kps[16] = [110.0, 810.0, 1.0]
    _write_pose_json(pose_json, {"5": [{"bbox": [50, 100, 150, 850, 1.0], "keypoints": kps}]})
    _identity_calib_json(orig_scene / "camera_0390" / "calibration.json")

    rewritten = rewrite_world_coordinates(
        sct_scene_dir=sct_scene,
        pose_scene_dir=pose_scene,
        calib_root=orig_scene,
        camera_map={390: "camera_0390"},
        method="ankle_avg",
        ankle_min_conf=0.3,
    )

    assert rewritten == 1
    after = json.loads(sct_json.read_text())
    # ankle_avg with equal weights = midpoint = (105, 805); identity homography -> world same.
    assert after["00000001"]["WorldCoordinate"]["x"] == pytest.approx(105.0)
    assert after["00000001"]["WorldCoordinate"]["y"] == pytest.approx(805.0)


def test_rewrite_pose_miss_falls_back_to_bbox_bottom(tmp_path):
    """Detection without a matching pose entry falls back silently."""
    from aic24_nvidia.world_projection import rewrite_world_coordinates

    sct_scene = tmp_path / "mct.tmp" / "scene_001"
    pose_scene = tmp_path / "Pose" / "scene_001"
    orig_scene = tmp_path / "adapted" / "Original" / "scene_001"

    sct_json = sct_scene / "camera390_tracking_results.json"
    _write_sct_json(sct_json, {
        "00000001": {
            "Frame": 5, "NpyPath": "x",
            "Coordinate": {"x1": 50, "y1": 100, "x2": 150, "y2": 850},
            "WorldCoordinate": {"x": 0.0, "y": 0.0},
            "OfflineID": 0,
        },
    })
    pose_json = pose_scene / "camera_0390" / "camera_0390_out_keypoint.json"
    # Pose JSON has a different bbox — miss on join.
    _write_pose_json(pose_json, {"5": [{"bbox": [999, 999, 9999, 9999, 1.0],
                                        "keypoints": [[0.0, 0.0, 0.0]] * 17}]})
    _identity_calib_json(orig_scene / "camera_0390" / "calibration.json")

    rewritten = rewrite_world_coordinates(
        sct_scene_dir=sct_scene,
        pose_scene_dir=pose_scene,
        calib_root=orig_scene,
        camera_map={390: "camera_0390"},
        method="ankle_avg",
        ankle_min_conf=0.3,
    )

    # Fallback to bbox_bottom = (100, 850); identity homography -> same.
    after = json.loads(sct_json.read_text())
    assert after["00000001"]["WorldCoordinate"]["x"] == pytest.approx(100.0)
    assert after["00000001"]["WorldCoordinate"]["y"] == pytest.approx(850.0)
    assert rewritten == 1   # still counts as "rewritten" (even if it landed on bbox_bottom)


def test_rewrite_processes_both_fixed_and_unfixed(tmp_path):
    """Both camera390_*.json and fixed_camera390_*.json are rewritten."""
    from aic24_nvidia.world_projection import rewrite_world_coordinates

    sct_scene = tmp_path / "mct.tmp" / "scene_001"
    pose_scene = tmp_path / "Pose" / "scene_001"
    orig_scene = tmp_path / "adapted" / "Original" / "scene_001"

    det_body = {
        "00000001": {
            "Frame": 5, "NpyPath": "x",
            "Coordinate": {"x1": 50, "y1": 100, "x2": 150, "y2": 850},
            "WorldCoordinate": {"x": -1.0, "y": -1.0},
            "OfflineID": 0,
        },
    }
    _write_sct_json(sct_scene / "camera390_tracking_results.json", det_body)
    _write_sct_json(sct_scene / "fixed_camera390_tracking_results.json", det_body)

    kps = [[0.0, 0.0, 0.0]] * 17
    kps[15] = [100.0, 800.0, 1.0]
    kps[16] = [110.0, 810.0, 1.0]
    _write_pose_json(
        pose_scene / "camera_0390" / "camera_0390_out_keypoint.json",
        {"5": [{"bbox": [50, 100, 150, 850, 1.0], "keypoints": kps}]},
    )
    _identity_calib_json(orig_scene / "camera_0390" / "calibration.json")

    rewritten = rewrite_world_coordinates(
        sct_scene_dir=sct_scene,
        pose_scene_dir=pose_scene,
        calib_root=orig_scene,
        camera_map={390: "camera_0390"},
        method="ankle_avg",
        ankle_min_conf=0.3,
    )
    assert rewritten == 2  # one detection in each file

    for fname in ("camera390_tracking_results.json", "fixed_camera390_tracking_results.json"):
        body = json.loads((sct_scene / fname).read_text())
        assert body["00000001"]["WorldCoordinate"]["x"] == pytest.approx(105.0)
        assert body["00000001"]["WorldCoordinate"]["y"] == pytest.approx(805.0)


def test_rewrite_missing_calibration_logs_warning(tmp_path, caplog):
    """Missing per-camera calibration logs a warning instead of failing silently."""
    import logging
    from aic24_nvidia.world_projection import rewrite_world_coordinates

    sct_scene = tmp_path / "mct.tmp" / "scene_001"
    pose_scene = tmp_path / "Pose" / "scene_001"
    orig_scene = tmp_path / "adapted" / "Original" / "scene_001"

    sct_scene.mkdir(parents=True)
    _write_pose_json(
        pose_scene / "camera_0390" / "camera_0390_out_keypoint.json",
        {"5": [{"bbox": [0, 0, 1, 1, 1.0], "keypoints": [[0.0, 0.0, 0.0]] * 17}]},
    )
    # Do NOT write calibration.json — that's the missing piece.

    with caplog.at_level(logging.WARNING, logger="aic24_nvidia.world_projection"):
        rewritten = rewrite_world_coordinates(
            sct_scene_dir=sct_scene,
            pose_scene_dir=pose_scene,
            calib_root=orig_scene,
            camera_map={390: "camera_0390"},
            method="ankle_avg",
            ankle_min_conf=0.3,
        )

    assert rewritten == 0
    assert any(
        "camera_0390" in rec.message and "calibration" in rec.message
        for rec in caplog.records
    ), f"expected warning about missing calibration; got: {[r.message for r in caplog.records]}"
