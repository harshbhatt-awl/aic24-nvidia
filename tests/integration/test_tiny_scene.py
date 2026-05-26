# tests/integration/test_tiny_scene.py
import json
import subprocess
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
