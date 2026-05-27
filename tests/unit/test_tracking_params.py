from pathlib import Path
import importlib.util

from aic24_nvidia.config import load_config
from aic24_nvidia.tracking_params import build_tracking_params, write_parameters_per_scene


def _write_cfg(tmp_path, extra_mct="", tracking_params_block="tracking_params: {}"):
    text = f"""
scene: Warehouse_001
data_root: ./data
weights_root: ./weights
outputs_root: ./outputs
external_root: ./external
clip: {{start_sec: 0, duration_sec: 30}}
detect: {{conf_thresh: 0.5, nms_iou: 0.5}}
reid: {{similarity_thresh: 0.7}}
pose: {{keypoint_conf: 0.3}}
sct: {{track_buffer: 30, match_thresh: 0.8}}
mct: {{cluster_thresh: 0.6, min_track_len: 10{extra_mct}}}
{tracking_params_block}
eval: {{world_d_max: 1.0}}
vram_min_free_gb: 4.0
fps: 30
"""
    p = tmp_path / "c.yaml"
    p.write_text(text)
    return load_config(p)


def test_build_uses_defaults_then_overrides(tmp_path):
    cfg = _write_cfg(tmp_path, tracking_params_block="tracking_params: {epsilon_mcpt: 0.25}")
    params = build_tracking_params(cfg)
    assert params["epsilon_scpt"] == 0.10
    assert params["replace_value"] == -10
    assert params["epsilon_mcpt"] == 0.25
    assert len(params) >= 10


def test_hard_world_gate_sets_large_negative_replace_value(tmp_path):
    cfg = _write_cfg(tmp_path, extra_mct=", hard_world_gate: true")
    params = build_tracking_params(cfg)
    assert params["replace_value"] <= -1e8
    assert params["replace_similarity_by_wcoordinate"] is True


def test_write_parameters_per_scene_is_importable(tmp_path):
    cfg = _write_cfg(tmp_path)
    cfg_dir = tmp_path / "tracking" / "config"
    write_parameters_per_scene(cfg, tmp_path, scene_int=1)
    out = cfg_dir / "parameters_per_scene.py"
    assert out.exists()
    spec = importlib.util.spec_from_file_location("pps_test", out)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert 1 in mod.parameters_per_scene
    assert mod.parameters_per_scene[1]["tracking_parameters"]["epsilon_mcpt"] == 0.37
