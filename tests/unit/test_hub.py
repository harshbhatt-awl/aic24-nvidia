import json
import sys
from pathlib import Path

import pytest

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
        "COMBINED": {"HOTA": 0.78},
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
    assert abs(r.image_hota - 0.78) < 1e-9
    assert abs(r.world_hota - 0.5282) < 1e-9


def test_discover_runs_image_hota_falls_back_to_camera_mean_without_combined(tmp_path):
    out = tmp_path / "outputs"
    run = out / "run_nc"
    _write_manifest(run / "evaluate")
    (run / "evaluate" / "metrics.json").write_text(json.dumps({
        "S001-camera_0001": {"HOTA": 0.70},
        "S001-camera_0002": {"HOTA": 0.80},
        "mct_world": {"HOTA": 0.5},
    }))
    r = hub.discover_runs(out)[0]
    assert abs(r.image_hota - 0.75) < 1e-9


def test_discover_runs_empty_when_no_outputs(tmp_path):
    assert hub.discover_runs(tmp_path / "nothing") == []


def test_build_pipeline_cmd_all_is_single_command():
    cmds = hub.build_pipeline_cmd(Path("configs/baseline.yaml"), None, None, False)
    assert cmds == [[sys.executable, "pipeline.py", "all", "--config", "configs/baseline.yaml"]]


def test_build_pipeline_cmd_runid_and_force():
    cmds = hub.build_pipeline_cmd(Path("configs/baseline.yaml"), None, "baseline", True)
    assert cmds == [[
        sys.executable, "pipeline.py", "all",
        "--config", "configs/baseline.yaml", "--run-id", "baseline", "--force",
    ]]


def test_build_pipeline_cmd_specific_stages_in_registry_order():
    cmds = hub.build_pipeline_cmd(Path("c.yaml"), ["detect", "adapt"], None, False)
    assert [c[2] for c in cmds] == ["adapt", "detect"]
    assert all(c[0] == sys.executable and c[1] == "pipeline.py" for c in cmds)
    assert cmds[0][2:] == ["adapt", "--config", "c.yaml"]


def test_build_experiment_cmd_run_requires_experiment():
    with pytest.raises(ValueError):
        hub.build_experiment_cmd("run")


def test_build_experiment_cmd_simple_actions():
    assert hub.build_experiment_cmd("list") == [sys.executable, "experiments/run.py", "list"]
    assert hub.build_experiment_cmd("status") == [sys.executable, "experiments/run.py", "status"]


def test_build_experiment_cmd_ensure_baseline_force():
    assert hub.build_experiment_cmd("ensure-baseline", force=True) == [
        sys.executable, "experiments/run.py", "ensure-baseline", "--force"]


def test_build_experiment_cmd_run_with_variant_and_force():
    assert hub.build_experiment_cmd("run", experiment="eps_mcpt_sweep",
                                    variant="0.30", force=True) == [
        sys.executable, "experiments/run.py", "run", "eps_mcpt_sweep",
        "--variant", "0.30", "--force"]


def test_build_experiment_cmd_run_all_variants():
    assert hub.build_experiment_cmd("run", experiment="eps_mcpt_sweep") == [
        sys.executable, "experiments/run.py", "run", "eps_mcpt_sweep"]


def test_build_compare_cmd():
    assert hub.build_compare_cmd() == [sys.executable, "experiments/compare.py"]
    assert hub.build_compare_cmd("mct_world.HOTA") == [
        sys.executable, "experiments/compare.py", "--sort-by", "mct_world.HOTA"]


def test_require_interactive_missing_dep_exits_with_hint(monkeypatch, capsys):
    # Make `import questionary` raise ImportError without uninstalling it.
    monkeypatch.setitem(sys.modules, "questionary", None)
    with pytest.raises(SystemExit) as exc:
        hub._require_interactive()
    assert exc.value.code == 2
    assert 'pip install -e ".[hub]"' in capsys.readouterr().err


def test_menu_subcommand_dispatches_to_run_hub(monkeypatch):
    import pipeline
    called = []
    # cmd_menu does `from aic24_nvidia import hub; hub.run_hub()`, so patch the attr.
    monkeypatch.setattr("aic24_nvidia.hub.run_hub", lambda: called.append(True))
    rc = pipeline.main(["menu"])
    assert called == [True]
    assert rc == 0
