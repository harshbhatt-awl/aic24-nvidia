"""Unit tests for write_per_camera_calibration."""
from __future__ import annotations
import json
import math
from pathlib import Path

import pytest

from aic24_nvidia.adapter.calibration import write_per_camera_calibration


def _make_calib_body(cameras: dict) -> dict:
    return {"cameras": cameras}


def test_writes_calibration_json(tmp_path):
    """Basic: file is written with required keys."""
    calib_body = _make_calib_body({
        "camera_0001": {
            "K": [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]],
            "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "t": [0.0, 0.0, 10.0],
        }
    })
    scene_dir = tmp_path / "Original" / "scene_001"
    scene_dir.mkdir(parents=True)

    written = write_per_camera_calibration(calib_body, scene_dir, ["camera_0001"])

    assert len(written) == 1
    out_path = scene_dir / "camera_0001" / "calibration.json"
    assert out_path.exists()
    assert written[0] == out_path


def test_keys_present(tmp_path):
    """Output JSON has exactly the expected keys."""
    calib_body = _make_calib_body({
        "camera_0001": {
            "K": [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]],
            "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "t": [0.0, 0.0, 10.0],
        }
    })
    scene_dir = tmp_path / "scene_001"
    write_per_camera_calibration(calib_body, scene_dir, ["camera_0001"])

    data = json.loads((scene_dir / "camera_0001" / "calibration.json").read_text())
    assert "camera projection matrix" in data
    assert "homography matrix" in data


def test_projection_matrix_shape(tmp_path):
    """Projection matrix is 3 rows x 4 cols."""
    calib_body = _make_calib_body({
        "camera_0001": {
            "K": [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]],
            "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "t": [0.0, 0.0, 10.0],
        }
    })
    scene_dir = tmp_path / "scene_001"
    write_per_camera_calibration(calib_body, scene_dir, ["camera_0001"])

    data = json.loads((scene_dir / "camera_0001" / "calibration.json").read_text())
    P = data["camera projection matrix"]
    assert len(P) == 3
    assert all(len(row) == 4 for row in P)


def test_homography_matrix_shape(tmp_path):
    """Homography matrix is 3 rows x 3 cols."""
    calib_body = _make_calib_body({
        "camera_0001": {
            "K": [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]],
            "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "t": [0.0, 0.0, 10.0],
        }
    })
    scene_dir = tmp_path / "scene_001"
    write_per_camera_calibration(calib_body, scene_dir, ["camera_0001"])

    data = json.loads((scene_dir / "camera_0001" / "calibration.json").read_text())
    H = data["homography matrix"]
    assert len(H) == 3
    assert all(len(row) == 3 for row in H)


def test_numeric_values_identity_rotation(tmp_path):
    """Verify exact numeric values for K with R=I, t=[0,0,10].

    With R=identity and t=[0,0,10]:
      P = K @ [I | t] = [[1000,0,960,9600],[0,1000,540,5400],[0,0,1,10]]
      H = K @ [r1|r2|t] = [[1000,0,9600],[0,1000,5400],[0,0,10]]
    """
    K = [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]]
    R = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    t = [0.0, 0.0, 10.0]

    calib_body = _make_calib_body({"camera_0001": {"K": K, "R": R, "t": t}})
    scene_dir = tmp_path / "scene_001"
    write_per_camera_calibration(calib_body, scene_dir, ["camera_0001"])

    data = json.loads((scene_dir / "camera_0001" / "calibration.json").read_text())
    P = data["camera projection matrix"]
    H = data["homography matrix"]

    expected_P = [
        [1000.0, 0.0, 960.0, 9600.0],
        [0.0, 1000.0, 540.0, 5400.0],
        [0.0, 0.0, 1.0, 10.0],
    ]
    expected_H = [
        [1000.0, 0.0, 9600.0],
        [0.0, 1000.0, 5400.0],
        [0.0, 0.0, 10.0],
    ]

    for i in range(3):
        for j in range(4):
            assert math.isclose(P[i][j], expected_P[i][j], rel_tol=1e-9), \
                f"P[{i}][{j}]: got {P[i][j]}, expected {expected_P[i][j]}"
    for i in range(3):
        for j in range(3):
            assert math.isclose(H[i][j], expected_H[i][j], rel_tol=1e-9), \
                f"H[{i}][{j}]: got {H[i][j]}, expected {expected_H[i][j]}"


def test_multiple_cameras(tmp_path):
    """Multiple cameras all get calibration files written."""
    cam_data = {
        "K": [[800, 0, 640], [0, 800, 360], [0, 0, 1]],
        "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "t": [1.0, 2.0, 5.0],
    }
    calib_body = _make_calib_body({
        "camera_0390": cam_data,
        "camera_0391": cam_data,
        "camera_0392": cam_data,
    })
    scene_dir = tmp_path / "scene_001"
    cameras = ["camera_0390", "camera_0391", "camera_0392"]
    written = write_per_camera_calibration(calib_body, scene_dir, cameras)

    assert len(written) == 3
    for cam in cameras:
        assert (scene_dir / cam / "calibration.json").exists()


def test_missing_camera_is_skipped_with_warning(tmp_path, caplog):
    """Camera absent from calib_body is skipped (no crash) and a warning is logged."""
    import logging
    calib_body = _make_calib_body({
        "camera_0001": {
            "K": [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]],
            "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "t": [0.0, 0.0, 10.0],
        }
    })
    scene_dir = tmp_path / "scene_001"
    with caplog.at_level(logging.WARNING, logger="aic24_nvidia.adapter.calibration"):
        written = write_per_camera_calibration(
            calib_body, scene_dir, ["camera_0001", "camera_9999"]
        )
    assert len(written) == 1
    assert not (scene_dir / "camera_9999" / "calibration.json").exists()
    assert "camera_9999" in caplog.text


def test_missing_krt_keys_skipped(tmp_path, caplog):
    """Camera with missing K/R/t keys is skipped gracefully."""
    import logging
    calib_body = _make_calib_body({
        "camera_0001": {"K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},  # missing R, t
    })
    scene_dir = tmp_path / "scene_001"
    with caplog.at_level(logging.WARNING, logger="aic24_nvidia.adapter.calibration"):
        written = write_per_camera_calibration(calib_body, scene_dir, ["camera_0001"])
    assert written == []


def test_creates_camera_dir_if_missing(tmp_path):
    """Camera directory is created automatically if it does not exist."""
    calib_body = _make_calib_body({
        "camera_0001": {
            "K": [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]],
            "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "t": [0.0, 0.0, 10.0],
        }
    })
    scene_dir = tmp_path / "nonexistent" / "scene_001"
    # scene_dir itself doesn't exist yet
    written = write_per_camera_calibration(calib_body, scene_dir, ["camera_0001"])
    assert len(written) == 1
    assert (scene_dir / "camera_0001" / "calibration.json").exists()


def test_homography_is_projection_cols_0_1_3(tmp_path):
    """H should equal P[:, [0,1,3]] — verify the Z=0 ground plane relationship."""
    import numpy as np

    K = [[800.5, 0, 640.2], [0, 799.8, 360.1], [0, 0, 1]]
    # non-trivial rotation
    theta = 0.3
    c, s = math.cos(theta), math.sin(theta)
    R = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    t = [2.5, -1.3, 8.0]

    calib_body = _make_calib_body({"camera_0001": {"K": K, "R": R, "t": t}})
    scene_dir = tmp_path / "scene_001"
    write_per_camera_calibration(calib_body, scene_dir, ["camera_0001"])

    data = json.loads((scene_dir / "camera_0001" / "calibration.json").read_text())
    P = np.array(data["camera projection matrix"])
    H = np.array(data["homography matrix"])

    # H should match P[:, [0,1,3]]
    assert np.allclose(H, P[:, [0, 1, 3]], atol=1e-9), \
        f"H does not match P[:,[0,1,3]]:\nH={H}\nP[:,[0,1,3]]={P[:,[0,1,3]]}"
