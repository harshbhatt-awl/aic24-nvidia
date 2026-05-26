import json
from pathlib import Path
from aic24_nvidia.stages.evaluate import (
    yachiyo_sct_json_to_mot,
    yachiyo_mct_json_to_mot,
    _build_mot_layout,
    _summarize_metrics,
)


def test_yachiyo_sct_json_to_mot(tmp_path):
    src = tmp_path / "fixed_camera001_tracking_results.json"
    # Upstream SCT JSON: nested dict keyed by serial; each entry has Frame, OfflineID,
    # Coordinate=[x1,y1,x2,y2].
    src.write_text(json.dumps({
        "0": {"Frame": 1, "OfflineID": 5, "Coordinate": [100, 200, 150, 300]},
        "1": {"Frame": 2, "OfflineID": 5, "Coordinate": [102, 201, 152, 301]},
        "2": {"Frame": 1, "OfflineID": 7, "Coordinate": [400, 200, 460, 310]},
    }))
    out = tmp_path / "out.txt"
    yachiyo_sct_json_to_mot(src, out)
    lines = sorted(out.read_text().splitlines())
    assert lines == [
        "1,5,100,200,50,100,1,-1,-1,-1",
        "1,7,400,200,60,110,1,-1,-1,-1",
        "2,5,102,201,50,100,1,-1,-1,-1",
    ]


def test_yachiyo_mct_json_to_mot_per_camera(tmp_path):
    src = tmp_path / "fixed_whole_tracking_results.json"
    src.write_text(json.dumps({
        "1": {  # camera id 1
            "0": {"Frame": 1, "GlobalOfflineID": 100, "Coordinate": [10, 20, 30, 50]},
        },
        "2": {
            "0": {"Frame": 1, "GlobalOfflineID": 100, "Coordinate": [40, 60, 80, 110]},
        },
    }))
    out_dir = tmp_path / "mot_mct"
    yachiyo_mct_json_to_mot(src, out_dir)
    c1 = (out_dir / "camera_0001.txt").read_text().splitlines()
    assert c1 == ["1,100,10,20,20,30,1,-1,-1,-1"]
    c2 = (out_dir / "camera_0002.txt").read_text().splitlines()
    assert c2 == ["1,100,40,60,40,50,1,-1,-1,-1"]


def test_build_mot_layout_creates_expected_tree(tmp_path):
    out = tmp_path / "mot"
    cam_gt = {"camera_0001": tmp_path / "gt1.txt"}
    cam_pred = {"camera_0001": tmp_path / "pred1.txt"}
    cam_gt["camera_0001"].write_text("1,1,100,200,50,100,1,1,1\n")
    cam_pred["camera_0001"].write_text("1,1,100,200,50,100,1,-1,-1,-1\n")
    _build_mot_layout(out, scene="S001", cam_gt=cam_gt, cam_pred=cam_pred)
    assert (out / "gt" / "S001" / "S001-camera_0001" / "gt" / "gt.txt").exists()
    assert (out / "trackers" / "S001" / "yachiyo" / "data" / "S001-camera_0001.txt").exists()


def test_summarize_metrics_extracts_hota():
    metrics = {"S001-camera_0001": {"HOTA": 0.65, "IDF1": 0.72, "MOTA": 0.60}}
    summary = _summarize_metrics(metrics)
    assert "HOTA" in summary
    assert "0.65" in summary
