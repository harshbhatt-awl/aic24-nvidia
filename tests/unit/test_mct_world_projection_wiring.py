"""Verify stages/mct.py invokes rewrite_world_coordinates with the right args."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


def test_mct_does_not_call_rewrite_when_method_bbox_bottom(tmp_path, monkeypatch):
    """When world_projection.method is bbox_bottom (default), the rewrite hook is not invoked."""
    from aic24_nvidia.stages import mct as mct_stage
    from aic24_nvidia.config import WorldProjectionCfg

    called = {"count": 0}

    def fake_rewrite(**kwargs):
        called["count"] += 1
        return 0

    monkeypatch.setattr(mct_stage, "rewrite_world_coordinates", fake_rewrite)

    mct_stage._maybe_rewrite_world_coordinates(
        cfg_world_projection=WorldProjectionCfg(method="bbox_bottom", ankle_min_conf=0.3),
        sct_scene_dir=tmp_path / "mct.tmp" / "scene_001",
        pose_scene_dir=tmp_path / "Pose" / "scene_001",
        calib_root=tmp_path / "adapted" / "Original" / "scene_001",
        camera_map={390: "camera_0390"},
    )

    assert called["count"] == 0   # short-circuit on no-op method


def test_mct_calls_rewrite_when_method_ankle_avg(tmp_path, monkeypatch):
    from aic24_nvidia.stages import mct as mct_stage
    from aic24_nvidia.config import WorldProjectionCfg

    captured = {}

    def fake_rewrite(**kwargs):
        captured.update(kwargs)
        return 7

    monkeypatch.setattr(mct_stage, "rewrite_world_coordinates", fake_rewrite)

    mct_stage._maybe_rewrite_world_coordinates(
        cfg_world_projection=WorldProjectionCfg(method="ankle_avg", ankle_min_conf=0.4),
        sct_scene_dir=tmp_path / "a",
        pose_scene_dir=tmp_path / "b",
        calib_root=tmp_path / "c",
        camera_map={390: "camera_0390"},
    )

    assert captured["method"] == "ankle_avg"
    assert captured["ankle_min_conf"] == pytest.approx(0.4)
    assert captured["camera_map"] == {390: "camera_0390"}
    assert captured["sct_scene_dir"] == tmp_path / "a"
