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
