import json
import numpy as np
from PIL import Image
from aic24_nvidia.models import pose_rtmpose
from aic24_nvidia.config import PoseCfg


def _fake_pose(estimate_fn):
    class FakePose:
        def load(self, cfg, weights_root):
            pass

        def estimate(self, img, bboxes):
            return estimate_fn(img, bboxes)

        def teardown(self):
            pass
    return FakePose()


def test_pose_json_schema_and_bbox_keys(tmp_path):
    scene, cam = "scene_001", "camera_0390"
    det_dir = tmp_path / "Detection" / scene
    det_dir.mkdir(parents=True)
    (det_dir / f"{cam}.txt").write_text(
        "camera_0390,1,1,10,20,110,220,0.91\ncamera_0390,1,1,50,60,90,180,0.7\n")
    frame_dir = tmp_path / "Original" / scene / cam / "Frame"
    frame_dir.mkdir(parents=True)
    Image.new("RGB", (1920, 1080)).save(frame_dir / "000001.jpg")
    pose_out = tmp_path / "Pose"

    pose_rtmpose.run_pose(
        det_scene_dir=det_dir, original_scene_dir=tmp_path / "Original" / scene,
        pose_out_dir=pose_out, scene=scene, cams=[cam],
        cfg=PoseCfg(keypoint_conf=0.3), weights_root=tmp_path / "weights",
        backend=_fake_pose(lambda img, bboxes: [[[1.0, 2.0, 0.9]] * 17 for _ in bboxes]))

    out = pose_out / scene / cam / f"{cam}_out_keypoint.json"
    j = json.loads(out.read_text())
    assert list(j.keys()) == ["1"]
    assert len(j["1"]) == 2
    assert j["1"][0]["bbox"] == [10, 20, 110, 220, 1.0]
    assert len(j["1"][0]["keypoints"]) == 17
    assert j["1"][0]["keypoints"][0] == [1.0, 2.0, 0.9]


def test_empty_detection_file_writes_empty_json(tmp_path):
    scene, cam = "scene_001", "camera_0390"
    det_dir = tmp_path / "Detection" / scene
    det_dir.mkdir(parents=True)
    (det_dir / f"{cam}.txt").write_text("")
    pose_out = tmp_path / "Pose"

    pose_rtmpose.run_pose(
        det_scene_dir=det_dir, original_scene_dir=tmp_path / "Original" / scene,
        pose_out_dir=pose_out, scene=scene, cams=[cam],
        cfg=PoseCfg(keypoint_conf=0.3), weights_root=tmp_path / "weights",
        backend=_fake_pose(lambda img, bboxes: [[[0.0, 0.0, 0.5]] * 17 for _ in bboxes]))

    out = pose_out / scene / cam / f"{cam}_out_keypoint.json"
    assert out.exists()
    j = json.loads(out.read_text())
    assert j == {}


def test_single_detection_row(tmp_path):
    scene, cam = "scene_001", "camera_0390"
    det_dir = tmp_path / "Detection" / scene
    det_dir.mkdir(parents=True)
    (det_dir / f"{cam}.txt").write_text("camera_0390,3,1,5,6,7,8,0.5\n")
    frame_dir = tmp_path / "Original" / scene / cam / "Frame"
    frame_dir.mkdir(parents=True)
    Image.new("RGB", (1920, 1080)).save(frame_dir / "000003.jpg")
    pose_out = tmp_path / "Pose"

    pose_rtmpose.run_pose(
        det_scene_dir=det_dir, original_scene_dir=tmp_path / "Original" / scene,
        pose_out_dir=pose_out, scene=scene, cams=[cam],
        cfg=PoseCfg(keypoint_conf=0.3), weights_root=tmp_path / "weights",
        backend=_fake_pose(lambda img, bboxes: [[[0.0, 0.0, 0.5]] * 17 for _ in bboxes]))

    out = pose_out / scene / cam / f"{cam}_out_keypoint.json"
    j = json.loads(out.read_text())
    assert list(j.keys()) == ["3"]
    assert len(j["3"]) == 1
    assert j["3"][0]["bbox"] == [5, 6, 7, 8, 1.0]
