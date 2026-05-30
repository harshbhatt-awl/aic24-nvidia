from pathlib import Path

import pytest
import yaml

from aic24_nvidia.config import load_config
from aic24_nvidia.errors import ConfigError

_BASE = {
    "scene": "Warehouse_001",
    "data_root": ".", "weights_root": "./weights", "outputs_root": "./outputs",
    "external_root": "./external",
    "clip": {"start_sec": 0, "duration_sec": 30},
    "detect": {"conf_thresh": 0.5, "nms_iou": 0.5},
    "reid": {"similarity_thresh": 0.7},
    "pose": {"keypoint_conf": 0.3},
    "sct": {"track_buffer": 30, "match_thresh": 0.8},
    "mct": {"cluster_thresh": 0.6, "min_track_len": 10},
    "vram_min_free_gb": 0.0, "fps": 30,
}


def _write(tmp_path, **patch):
    body = {k: dict(v) if isinstance(v, dict) else v for k, v in _BASE.items()}
    for k, v in patch.items():
        body[k] = v
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(body))
    return p


def test_model_names_default_to_current_stack(tmp_path):
    cfg = load_config(_write(tmp_path))
    assert cfg.detect.model_name == "yolo11x"
    assert cfg.reid.model_name == "solider_swin_small"
    assert cfg.pose.model_name == "rtmpose-l"
    assert cfg.detect.weights is None


def test_explicit_model_name_and_weights_are_read(tmp_path):
    cfg = load_config(_write(
        tmp_path,
        detect={"conf_thresh": 0.5, "nms_iou": 0.5, "model_name": "yolo11x",
                "weights": "custom.pt"},
    ))
    assert cfg.detect.weights == "custom.pt"


def test_unknown_detector_model_name_rejected(tmp_path):
    p = _write(tmp_path, detect={"conf_thresh": 0.5, "nms_iou": 0.5, "model_name": "bogus"})
    with pytest.raises(ConfigError, match="model_name"):
        load_config(p)


def test_unknown_reid_model_name_rejected(tmp_path):
    p = _write(tmp_path, reid={"similarity_thresh": 0.7, "model_name": "bogus"})
    with pytest.raises(ConfigError, match="model_name"):
        load_config(p)
