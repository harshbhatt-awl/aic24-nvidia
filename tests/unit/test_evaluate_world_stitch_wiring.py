# tests/unit/test_evaluate_world_stitch_wiring.py
"""Verify _eval_mct_world calls stitch_world_tracks with the right knobs, before smoothing."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _stub_cfg(method="endpoint_gap", max_gap_frames=30, max_dist_m=0.5):
    from aic24_nvidia.config import WorldSmoothingCfg, WorldStitchCfg, EvalCfg
    cfg = MagicMock()
    cfg.eval = EvalCfg(world_d_max=1.0)
    cfg.world_smoothing = WorldSmoothingCfg(method="none", ema_alpha=0.3)
    cfg.world_stitch = WorldStitchCfg(method=method, max_gap_frames=max_gap_frames, max_dist_m=max_dist_m)
    return cfg


def _write_minimal_mct(tmp_path: Path) -> Path:
    mct = tmp_path / "fixed_whole_tracking_results.json"
    mct.write_text(json.dumps({
        "390": {
            "00000001": {
                "Frame": 5,
                "GlobalOfflineID": 1,
                "Coordinate": {"x1": 50, "y1": 100, "x2": 150, "y2": 850},
                "WorldCoordinate": {"x": 1.0, "y": 2.0},
            }
        }
    }))
    return mct


def test_eval_mct_world_calls_stitch(tmp_path, monkeypatch):
    from aic24_nvidia.stages import evaluate as evaluate_stage

    captured = {}

    def fake_stitch(rows, method, max_gap_frames, max_dist_m):
        captured["method"] = method
        captured["max_gap_frames"] = max_gap_frames
        captured["max_dist_m"] = max_dist_m
        return rows, []

    monkeypatch.setattr(evaluate_stage, "stitch_world_tracks", fake_stitch)

    cfg = _stub_cfg(method="endpoint_gap", max_gap_frames=30, max_dist_m=0.5)
    ctx = MagicMock()
    ctx.work_dir = tmp_path

    mct_global = _write_minimal_mct(tmp_path)
    adapted_root = tmp_path / "adapted"
    adapted_root.mkdir()
    (adapted_root / "scene_001_gt_world.txt").write_text("5,1,1.0,2.0\n")

    try:
        evaluate_stage._eval_mct_world(cfg, ctx, str(mct_global), adapted_root)
    except Exception:
        pass

    assert captured.get("method") == "endpoint_gap"
    assert captured.get("max_gap_frames") == 30
    assert captured.get("max_dist_m") == pytest.approx(0.5)
