import json
from pathlib import Path
import pytest
from aic24_nvidia.adapter.calibration import adapt_calibration


def test_adapt_calibration_renames_cameras(tmp_path):
    src = tmp_path / "calibration.json"
    src.write_text(json.dumps({
        "Camera_0001": {"intrinsicMatrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
        "Camera_0002": {"intrinsicMatrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
    }))

    dst = tmp_path / "out.json"
    mapping = {"camera_0001": "Camera_0001", "camera_0002": "Camera_0002"}
    adapt_calibration(src, dst, scene_mapping=mapping)

    body = json.loads(dst.read_text())
    assert "camera_0001" in body
    assert "camera_0002" in body
    assert "Camera_0001" not in body
    assert body["camera_0001"]["intrinsicMatrix"][0] == [1, 0, 0]


def test_adapt_calibration_fails_on_missing_camera(tmp_path):
    src = tmp_path / "calibration.json"
    src.write_text(json.dumps({"Camera_0001": {}}))
    dst = tmp_path / "out.json"
    mapping = {"camera_0001": "Camera_0001", "camera_0002": "Camera_0002"}
    with pytest.raises(KeyError, match="Camera_0002"):
        adapt_calibration(src, dst, scene_mapping=mapping)
