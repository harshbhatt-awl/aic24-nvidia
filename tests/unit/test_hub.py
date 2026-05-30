import json
from pathlib import Path

from aic24_nvidia import hub


def _write_manifest(d: Path, status="ok", finished_at="2026-01-01T00:00:01Z"):
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps({
        "stage": d.name, "run_id": "r", "started_at": "x", "finished_at": finished_at,
        "runtime_sec": 1.0, "inputs": {}, "outputs": {}, "params": {},
        "upstream_manifests": [], "status": status,
    }))


def test_load_metrics_present_and_missing(tmp_path):
    run = tmp_path / "run1"
    (run / "evaluate").mkdir(parents=True)
    (run / "evaluate" / "metrics.json").write_text(json.dumps({"S001-camera_0001": {"HOTA": 0.5}}))
    assert hub.load_metrics(run) == {"S001-camera_0001": {"HOTA": 0.5}}
    assert hub.load_metrics(tmp_path / "nope") is None


def test_discover_runs_reads_stages_status_and_metrics(tmp_path):
    out = tmp_path / "outputs"
    run = out / "baseline_x"
    _write_manifest(run / "adapted")
    _write_manifest(run / "detect")
    _write_manifest(run / "evaluate")
    (run / "evaluate" / "metrics.json").write_text(json.dumps({
        "S001-camera_0001": {"HOTA": 0.70},
        "S001-camera_0002": {"HOTA": 0.80},
        "mct_world": {"HOTA": 0.5282},
    }))

    runs = hub.discover_runs(out)
    assert len(runs) == 1
    r = runs[0]
    assert r.run_id == "baseline_x"
    assert r.stages_present["adapt"] is True
    assert r.stages_present["detect"] is True
    assert r.stages_present["reid"] is False
    assert r.status == "ok"
    assert r.finished_at == "2026-01-01T00:00:01Z"
    assert abs(r.image_hota - 0.75) < 1e-9
    assert abs(r.world_hota - 0.5282) < 1e-9


def test_discover_runs_empty_when_no_outputs(tmp_path):
    assert hub.discover_runs(tmp_path / "nothing") == []
