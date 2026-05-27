import json
import numpy as np
from PIL import Image
from aic24_nvidia.models import pose_rtmpose


def test_pose_json_schema_and_bbox_keys(tmp_path, monkeypatch):
    scene, cam = "scene_001", "camera_0390"
    det_dir = tmp_path / "Detection" / scene
    det_dir.mkdir(parents=True)
    (det_dir / f"{cam}.txt").write_text(
        "camera_0390,1,1,10,20,110,220,0.91\ncamera_0390,1,1,50,60,90,180,0.7\n")
    frame_dir = tmp_path / "Original" / scene / cam / "Frame"
    frame_dir.mkdir(parents=True)
    Image.new("RGB", (1920, 1080)).save(frame_dir / "000001.jpg")
    pose_out = tmp_path / "Pose"

    monkeypatch.setattr(pose_rtmpose, "_estimate",
                        lambda img, bboxes: [[[1.0, 2.0, 0.9]] * 17 for _ in bboxes])
    pose_rtmpose.run_pose(det_scene_dir=det_dir,
                          original_scene_dir=tmp_path / "Original" / scene,
                          pose_out_dir=pose_out, scene=scene, cams=[cam])

    out = pose_out / scene / cam / f"{cam}_out_keypoint.json"
    j = json.loads(out.read_text())
    assert list(j.keys()) == ["1"]
    assert len(j["1"]) == 2
    assert j["1"][0]["bbox"] == [10, 20, 110, 220, 1.0]
    assert len(j["1"][0]["keypoints"]) == 17
    assert j["1"][0]["keypoints"][0] == [1.0, 2.0, 0.9]
