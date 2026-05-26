import json
from pathlib import Path
import pytest
from aic24_nvidia.manifest import Manifest, write_manifest, read_manifest, gate


def test_roundtrip(tmp_path: Path):
    m = Manifest(
        stage="detect",
        run_id="r1",
        started_at="2026-05-26T14:30:00Z",
        finished_at="2026-05-26T14:35:00Z",
        runtime_sec=300.0,
        inputs={"adapted": str(tmp_path / "adapted")},
        outputs={"results": str(tmp_path / "detect/results")},
        params={"conf_thresh": 0.5},
        upstream_manifests=[],
        status="ok",
    )
    p = tmp_path / "manifest.json"
    write_manifest(m, p)
    loaded = read_manifest(p)
    assert loaded == m


def test_gate_skip_when_present_no_force(tmp_path):
    stage_dir = tmp_path / "detect"
    stage_dir.mkdir()
    (stage_dir / "manifest.json").write_text(json.dumps({
        "stage": "detect", "run_id": "r1", "started_at": "x", "finished_at": "y",
        "runtime_sec": 1.0, "inputs": {}, "outputs": {}, "params": {},
        "upstream_manifests": [], "status": "ok",
    }))
    decision = gate(stage_dir, upstream=[], force=False)
    assert decision == "skip"


def test_gate_run_when_not_present(tmp_path):
    stage_dir = tmp_path / "detect"
    decision = gate(stage_dir, upstream=[], force=False)
    assert decision == "run"


def test_gate_run_when_force(tmp_path):
    stage_dir = tmp_path / "detect"
    stage_dir.mkdir()
    (stage_dir / "manifest.json").write_text(json.dumps({
        "stage": "detect", "run_id": "r1", "started_at": "x", "finished_at": "y",
        "runtime_sec": 1.0, "inputs": {}, "outputs": {}, "params": {},
        "upstream_manifests": [], "status": "ok",
    }))
    assert gate(stage_dir, upstream=[], force=True) == "run"


def test_gate_error_when_upstream_missing(tmp_path):
    stage_dir = tmp_path / "detect"
    upstream_dir = tmp_path / "adapted"
    with pytest.raises(RuntimeError, match="upstream"):
        gate(stage_dir, upstream=[upstream_dir / "manifest.json"], force=False)


def test_gate_error_when_upstream_failed(tmp_path):
    stage_dir = tmp_path / "detect"
    upstream_dir = tmp_path / "adapted"
    upstream_dir.mkdir()
    (upstream_dir / "manifest.json").write_text(json.dumps({
        "stage": "adapt", "run_id": "r1", "started_at": "x", "finished_at": "y",
        "runtime_sec": 1.0, "inputs": {}, "outputs": {}, "params": {},
        "upstream_manifests": [], "status": "error",
    }))
    with pytest.raises(RuntimeError, match="status"):
        gate(stage_dir, upstream=[upstream_dir / "manifest.json"], force=False)
