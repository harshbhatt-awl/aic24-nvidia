import json
from pathlib import Path
import numpy as np
from PIL import Image
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


def test_run_detection_glue_writes_expected_files(tmp_path):
    # Build a tmp scene dir with two tiny jpgs
    frame_dir = tmp_path / "Original" / "scene_001" / "camera_0390" / "Frame"
    frame_dir.mkdir(parents=True)
    for name in ("000001.jpg", "000002.jpg"):
        Image.new("RGB", (64, 64)).save(frame_dir / name)

    # Fake backend: one detection for frame 000001, none for 000002.
    class FakeDetector:
        def load(self, cfg, weights_root):
            self.loaded = True

        def infer(self, img_path):
            if Path(img_path).stem == "000001":
                return [(1.0, 2.0, 3.0, 4.0, 0.9)]
            return []

        def teardown(self):
            self.torn = True

    from aic24_nvidia.config import DetectCfg
    detect_yolo.run_detection(
        scene_dir=tmp_path / "Original" / "scene_001",
        det_out_dir=tmp_path / "out",
        cams=["camera_0390"],
        cfg=DetectCfg(conf_thresh=0.5, nms_iou=0.5),
        weights_root=tmp_path / "weights",
        backend=FakeDetector(),
    )

    txt_path = tmp_path / "out" / "scene_001" / "camera_0390.txt"
    assert txt_path.exists(), "camera_0390.txt not written"
    lines = txt_path.read_text().splitlines()
    assert len(lines) == 1, f"expected 1 line, got {len(lines)}: {lines}"
    assert lines[0] == "camera_0390,1,1,1,2,3,4,0.9"

    j = json.loads((tmp_path / "out" / "scene_001" / "camera_0390.json").read_text())
    assert "00000000" in j
    assert j["00000000"]["Frame"] == 1
    assert len(j) == 1
