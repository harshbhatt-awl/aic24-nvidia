import json
import numpy as np
from PIL import Image
from aic24_nvidia.models import reid_solider


def test_npy_filename_and_json_update_match_upstream(tmp_path):
    scene, cam = "scene_001", "camera_0390"
    det_dir = tmp_path / "Detection" / scene
    det_dir.mkdir(parents=True)
    (det_dir / f"{cam}.txt").write_text("camera_0390,5,1,10,20,110,220,0.91\n")
    (det_dir / f"{cam}.json").write_text(json.dumps(
        {"00000000": {"Frame": 5, "ImgPath": "x", "NpyPath": "",
                      "Coordinate": {"x1":10,"y1":20,"x2":110,"y2":220},
                      "ClusterID": None, "OfflineID": None}}))
    frame_dir = tmp_path / "Original" / scene / cam / "Frame"
    frame_dir.mkdir(parents=True)
    Image.new("RGB", (1920, 1080)).save(frame_dir / "000005.jpg")
    emb_dir = tmp_path / "EmbedFeature"

    reid_solider.extract_camera(
        det_scene_dir=det_dir, original_scene_dir=tmp_path / "Original" / scene,
        emb_out_dir=emb_dir, scene=scene, cam=cam,
        embed=lambda crop: np.ones(768, dtype=np.float32))

    files = list((emb_dir / scene / cam).glob("*.npy"))
    assert len(files) == 1
    assert files[0].name == "feature_5_1_10_110_20_220_091.npy"
    assert np.load(files[0]).shape == (768,)
    j = json.loads((det_dir / f"{cam}.json").read_text())
    assert j["00000000"]["NpyPath"] == f"{scene}/{cam}/feature_5_1_10_110_20_220_091.npy"


def test_empty_detection_file_is_noop(tmp_path):
    scene, cam = "scene_001", "camera_0390"
    det_dir = tmp_path / "Detection" / scene
    det_dir.mkdir(parents=True)
    (det_dir / f"{cam}.txt").write_text("")
    (det_dir / f"{cam}.json").write_text(json.dumps({}))
    emb_dir = tmp_path / "EmbedFeature"

    # Should not raise and should not create any .npy files
    reid_solider.extract_camera(
        det_scene_dir=det_dir, original_scene_dir=tmp_path / "Original" / scene,
        emb_out_dir=emb_dir, scene=scene, cam=cam,
        embed=lambda crop: np.ones(768, dtype=np.float32))

    # No output directory / npy files should have been created
    cam_out = emb_dir / scene / cam
    assert not cam_out.exists() or list(cam_out.glob("*.npy")) == []


def test_run_reid_drives_injected_backend(tmp_path):
    scene, cam = "scene_001", "camera_0390"
    det_dir = tmp_path / "Detection" / scene
    det_dir.mkdir(parents=True)
    (det_dir / f"{cam}.txt").write_text("camera_0390,5,1,10,20,110,220,0.91\n")
    (det_dir / f"{cam}.json").write_text(json.dumps(
        {"00000000": {"Frame": 5, "ImgPath": "x", "NpyPath": "",
                      "Coordinate": {"x1": 10, "y1": 20, "x2": 110, "y2": 220},
                      "ClusterID": None, "OfflineID": None}}))
    frame_dir = tmp_path / "Original" / scene / cam / "Frame"
    frame_dir.mkdir(parents=True)
    Image.new("RGB", (1920, 1080)).save(frame_dir / "000005.jpg")
    emb_dir = tmp_path / "EmbedFeature"

    events = []

    class FakeReID:
        def load(self, cfg, weights_root):
            events.append("load")

        def embed(self, crop):
            return np.ones(768, dtype=np.float32)

        def teardown(self):
            events.append("teardown")

    from aic24_nvidia.config import ReidCfg
    reid_solider.run_reid(
        det_scene_dir=det_dir, original_scene_dir=tmp_path / "Original" / scene,
        emb_out_dir=emb_dir, scene=scene, cams=[cam],
        cfg=ReidCfg(similarity_thresh=0.7), weights_root=tmp_path / "weights",
        backend=FakeReID())

    assert events == ["load", "teardown"]
    assert list((emb_dir / scene / cam).glob("*.npy"))
