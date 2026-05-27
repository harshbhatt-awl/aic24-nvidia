import json
from pathlib import Path
import numpy as np
from aic24_nvidia.models import detect_yolo


def test_write_detection_outputs_matches_upstream_format(tmp_path):
    dets_by_frame = {
        1: [
            (10.4, 20.6, 110.2, 220.9, float(np.float32(0.91))),
            (300.0, 50.0, 360.0, 180.0, float(np.float32(0.55))),
        ],
        2: [(12.0, 22.0, 112.0, 222.0, float(np.float32(0.88)))],
        3: [(-5.0, -5.0, 5000.0, 5000.0, float(np.float32(0.7)))],
    }
    det_dir = tmp_path / "detect" / "scene_001"
    detect_yolo._write_camera(
        det_dir=det_dir, cam="camera_0390", dets_by_frame=dets_by_frame,
        img_rel_for_frame=lambda f: f"Original/scene_001/camera_0390/Frame/{f:06d}.jpg",
    )
    txt = (det_dir / "camera_0390.txt").read_text().splitlines()
    assert txt[0] == "camera_0390,1,1,10,20,110,220,0.91"
    assert txt[1] == "camera_0390,1,1,300,50,360,180,0.55"
    assert txt[2] == "camera_0390,2,1,12,22,112,222,0.88"
    # out-of-bounds box clamped to frame dimensions
    assert txt[3] == "camera_0390,3,1,0,0,1920,1080,0.7"
    j = json.loads((det_dir / "camera_0390.json").read_text())
    assert set(j.keys()) == {"00000000", "00000001", "00000002", "00000003"}
    assert j["00000000"] == {
        "Frame": 1, "ImgPath": "Original/scene_001/camera_0390/Frame/000001.jpg",
        "NpyPath": "", "Coordinate": {"x1": 10, "y1": 20, "x2": 110, "y2": 220},
        "ClusterID": None, "OfflineID": None,
    }
