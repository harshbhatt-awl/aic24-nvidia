from pathlib import Path
from aic24_nvidia.config import load_config
from aic24_nvidia.tracking_params import write_parameters_per_scene


def test_propagation_writes_real_keys(tmp_path):
    text = """
scene: Warehouse_001
data_root: ./data
weights_root: ./weights
outputs_root: ./outputs
external_root: ./external
clip: {start_sec: 0, duration_sec: 30}
detect: {conf_thresh: 0.5, nms_iou: 0.5}
reid: {similarity_thresh: 0.7}
pose: {keypoint_conf: 0.3}
sct: {track_buffer: 30, match_thresh: 0.8}
mct: {cluster_thresh: 0.6, min_track_len: 10}
tracking_params: {epsilon_mcpt: 0.20}
eval: {world_d_max: 1.0}
vram_min_free_gb: 4.0
fps: 30
"""
    p = tmp_path / "c.yaml"
    p.write_text(text)
    cfg = load_config(p)
    out = write_parameters_per_scene(cfg, tmp_path, scene_int=1)
    import importlib.util
    spec = importlib.util.spec_from_file_location("pps_prop_test", out)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tp = mod.parameters_per_scene[1]["tracking_parameters"]
    assert "tracking_parameters" in out.read_text()
    assert tp["epsilon_mcpt"] == 0.20   # our override landed
