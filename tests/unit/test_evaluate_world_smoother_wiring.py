"""Verify _eval_mct_world calls smooth_world_tracks with the right knobs."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _stub_cfg(smoothing_method="none", ema_alpha=0.3):
    from aic24_nvidia.config import WorldSmoothingCfg, EvalCfg
    cfg = MagicMock()
    cfg.eval = EvalCfg(world_d_max=1.0)
    cfg.world_smoothing = WorldSmoothingCfg(method=smoothing_method, ema_alpha=ema_alpha)
    return cfg


def _write_minimal_mct(tmp_path: Path) -> Path:
    """One detection in one cam, valid for aggregate_world_tracks."""
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


def test_eval_mct_world_calls_smoother(tmp_path, monkeypatch):
    """smooth_world_tracks is invoked with the cfg's method + alpha."""
    from aic24_nvidia.stages import evaluate as evaluate_stage

    captured = {}

    def fake_smooth(rows, method, ema_alpha):
        captured["method"] = method
        captured["ema_alpha"] = ema_alpha
        return rows

    monkeypatch.setattr(evaluate_stage, "smooth_world_tracks", fake_smooth)

    cfg = _stub_cfg(smoothing_method="ema", ema_alpha=0.4)
    ctx = MagicMock()
    ctx.work_dir = tmp_path

    mct_global = _write_minimal_mct(tmp_path)
    adapted_root = tmp_path / "adapted"
    # We do not need a real gt_world.txt for this test — the smoothing call
    # happens before TrackEval; provide one to clear the early-exit guard.
    (adapted_root).mkdir()
    (adapted_root / "scene_001_gt_world.txt").write_text("5,1,1.0,2.0\n")

    # The function may exit early if TrackEval isn't available — that's fine;
    # we only assert the smoother was called.
    try:
        evaluate_stage._eval_mct_world(cfg, ctx, str(mct_global), adapted_root)
    except Exception:
        pass

    assert captured.get("method") == "ema"
    assert captured.get("ema_alpha") == pytest.approx(0.4)
