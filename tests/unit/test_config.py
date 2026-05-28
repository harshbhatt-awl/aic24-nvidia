from pathlib import Path
import pytest
import yaml
from aic24_nvidia.config import Config, load_config
from aic24_nvidia.errors import ConfigError


def _write(tmp_path: Path, body: dict) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(body))
    return p


def _minimal(tmp_path: Path) -> dict:
    return {
        "scene": "Warehouse_001",
        "data_root": str(tmp_path / "data"),
        "weights_root": str(tmp_path / "weights"),
        "outputs_root": str(tmp_path / "outputs"),
        "external_root": str(tmp_path / "external"),
        "clip": {"start_sec": 0, "duration_sec": 30},
        "detect": {"conf_thresh": 0.5, "nms_iou": 0.5},
        "reid": {"similarity_thresh": 0.7},
        "pose": {"keypoint_conf": 0.3},
        "sct": {"track_buffer": 30, "match_thresh": 0.8},
        "mct": {"cluster_thresh": 0.6, "min_track_len": 10},
        "vram_min_free_gb": 4.0,
        "fps": 30,
    }


def test_load_minimal(tmp_path):
    p = _write(tmp_path, _minimal(tmp_path))
    cfg = load_config(p)
    assert isinstance(cfg, Config)
    assert cfg.scene == "Warehouse_001"
    assert cfg.clip.duration_sec == 30
    assert cfg.detect.conf_thresh == 0.5
    assert cfg.sct.match_thresh == 0.8


def test_missing_required_raises(tmp_path):
    body = _minimal(tmp_path)
    del body["scene"]
    p = _write(tmp_path, body)
    with pytest.raises(ConfigError, match="scene"):
        load_config(p)


def test_negative_duration_rejected(tmp_path):
    body = _minimal(tmp_path)
    body["clip"]["duration_sec"] = -1
    p = _write(tmp_path, body)
    with pytest.raises(ConfigError, match="duration"):
        load_config(p)


def test_config_filename_property(tmp_path):
    p = _write(tmp_path, _minimal(tmp_path))
    cfg = load_config(p)
    assert cfg.config_filename == "cfg"


def test_config_loads_eval_and_tracking_params(tmp_path):
    from aic24_nvidia.config import load_config
    cfg_text = """
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
mct: {cluster_thresh: 0.6, min_track_len: 10, hard_world_gate: true}
tracking_params: {epsilon_mcpt: 0.37, distance_th: 10, short_track_th: 120}
eval: {world_d_max: 1.0}
vram_min_free_gb: 4.0
fps: 30
"""
    p = tmp_path / "c.yaml"
    p.write_text(cfg_text)
    cfg = load_config(p)
    assert cfg.eval.world_d_max == 1.0
    assert cfg.mct.hard_world_gate is True
    assert cfg.tracking_params["epsilon_mcpt"] == 0.37


def test_config_defaults_when_optional_blocks_missing(tmp_path):
    from aic24_nvidia.config import load_config
    cfg_text = """
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
vram_min_free_gb: 4.0
fps: 30
"""
    p = tmp_path / "c.yaml"
    p.write_text(cfg_text)
    cfg = load_config(p)
    assert cfg.eval.world_d_max == 1.0          # default
    assert cfg.mct.hard_world_gate is False     # default
    assert cfg.tracking_params == {}            # default empty


def test_world_projection_defaults(tmp_path):
    """When world_projection block is absent, defaults to bbox_bottom + 0.3."""
    body = _minimal(tmp_path)
    # Do NOT add world_projection
    cfg = load_config(_write(tmp_path, body))

    assert cfg.world_projection.method == "bbox_bottom"
    assert cfg.world_projection.ankle_min_conf == pytest.approx(0.3)


def test_world_projection_explicit(tmp_path):
    """world_projection block is parsed and validated."""
    body = _minimal(tmp_path)
    body["world_projection"] = {"method": "ankle_avg", "ankle_min_conf": 0.5}
    cfg = load_config(_write(tmp_path, body))

    assert cfg.world_projection.method == "ankle_avg"
    assert cfg.world_projection.ankle_min_conf == pytest.approx(0.5)


def test_world_projection_invalid_method(tmp_path):
    """Unknown method is rejected at load time."""
    body = _minimal(tmp_path)
    body["world_projection"] = {"method": "bogus"}
    with pytest.raises(ConfigError, match="world_projection.method"):
        load_config(_write(tmp_path, body))
