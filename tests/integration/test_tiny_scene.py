# tests/integration/test_tiny_scene.py
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import yaml
import pytest

from aic24_nvidia.config import load_config
from aic24_nvidia.paths import make_run_id
from aic24_nvidia.stages import adapt


def _write_cfg(root: Path) -> Path:
    cfg = {
        "scene": "Warehouse_T01",
        "data_root": str(root / "data"),
        "weights_root": str(root / "weights"),
        "outputs_root": str(root / "outputs"),
        "external_root": str(root / "external"),
        "clip": {"start_sec": 0, "duration_sec": 1.0},
        "detect": {"conf_thresh": 0.5, "nms_iou": 0.5},
        "reid": {"similarity_thresh": 0.7},
        "pose": {"keypoint_conf": 0.3},
        "sct": {"track_buffer": 30, "match_thresh": 0.8},
        "mct": {"cluster_thresh": 0.6, "min_track_len": 10},
        "vram_min_free_gb": 0.0,
        "fps": 30,
    }
    p = root / "tiny.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_adapt_stage_on_tiny_scene(tiny_scene):
    cfg_path = _write_cfg(tiny_scene)
    cfg = load_config(cfg_path)
    rid = make_run_id(cfg.config_filename)
    rd = cfg.outputs_root / rid
    rd.mkdir(parents=True)

    adapt.run(cfg, rd, rid)

    assert (rd / "adapted" / "manifest.json").exists()
    assert (rd / "adapted" / "Original" / "scene_001" / "camera_0001" / "video.mp4").exists()
    assert (rd / "adapted" / "Original" / "scene_001" / "camera_0002" / "video.mp4").exists()
    assert (rd / "adapted" / "scene.json").exists()
    assert (rd / "adapted" / "gt_validation.json").exists()

    gt_c1 = (rd / "adapted" / "Original" / "scene_001" / "camera_0001" / "gt" / "gt.txt").read_text()
    assert gt_c1.splitlines()[0].startswith("1,1,")


@pytest.mark.skip(reason="full pipeline requires upstream YACHIYO scripts to be mocked; "
                         "tracked as follow-up after Task 11 documents exact entry points")
def test_full_pipeline_on_tiny_scene(tiny_scene):
    """Placeholder: enable once Task 11 produces the entry-point doc and we can
    mock each subprocess.run call with deterministic fixture outputs."""


def test_world_eval_end_to_end(tmp_path):
    # TrackEval (+scipy) must be importable, else skip cleanly.
    ext = Path(__file__).resolve().parents[2] / "external" / "TrackEval"
    sys.path.insert(0, str(ext))
    try:
        import trackeval  # noqa: F401
    except Exception as exc:
        pytest.skip(f"TrackEval not importable: {exc}")

    from aic24_nvidia.world_tracks import aggregate_world_tracks, write_world_pred
    from aic24_nvidia.world_metrics import run_world_eval

    # GT: id 1 walks (0,0)->(0,1)->(0,2) over frames 1..3
    gt = tmp_path / "scene_001_gt_world.txt"
    gt.write_text("1,1,0,0\n2,1,0,1\n3,1,0,2\n")
    # MCT JSON: one camera, global id 5 perfectly overlapping gt id 1
    mct = tmp_path / "mct.json"
    mct.write_text(json.dumps({"390": {
        "00000001": {"Frame": 1, "WorldCoordinate": {"x": 0, "y": 0}, "GlobalOfflineID": 5},
        "00000002": {"Frame": 2, "WorldCoordinate": {"x": 0, "y": 1}, "GlobalOfflineID": 5},
        "00000003": {"Frame": 3, "WorldCoordinate": {"x": 0, "y": 2}, "GlobalOfflineID": 5},
    }}))
    rows, _ = aggregate_world_tracks(mct)
    pred = tmp_path / "pred.txt"
    write_world_pred(rows, pred)
    m = run_world_eval(gt, pred, d_max=1.0, trackeval_root=ext, seq_name="scene_001")
    assert m["HOTA"] > 0.9        # perfect spatial+identity overlap
    assert m["IDF1"] > 0.9
