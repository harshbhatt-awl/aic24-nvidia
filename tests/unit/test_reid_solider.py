import json
import numpy as np
from PIL import Image
from aic24_nvidia.models import reid_solider


def test_npy_filename_and_json_update_match_upstream(tmp_path, monkeypatch):
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

    monkeypatch.setattr(reid_solider, "_embed", lambda crop: np.ones(768, dtype=np.float32))
    reid_solider.extract_camera(
        det_scene_dir=det_dir, original_scene_dir=tmp_path / "Original" / scene,
        emb_out_dir=emb_dir, scene=scene, cam=cam, embed=reid_solider._embed)

    files = list((emb_dir / scene / cam).glob("*.npy"))
    assert len(files) == 1
    assert files[0].name == "feature_5_1_10_110_20_220_091.npy"
    assert np.load(files[0]).shape == (768,)
    j = json.loads((det_dir / f"{cam}.json").read_text())
    assert j["00000000"]["NpyPath"] == f"{scene}/{cam}/feature_5_1_10_110_20_220_091.npy"
