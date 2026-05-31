"""Unit tests for the durable results ledger (aic24_nvidia/results.py)."""

from __future__ import annotations

import json
from pathlib import Path

from aic24_nvidia import results as R


def _write(p: Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))


def _make_run(
    root: Path,
    run_id: str,
    *,
    world_hota: float = 0.60,
    image_hota: float = 0.77,
    finished: str = "2026-05-30T21:25:23Z",
    eps_scpt: float = 0.15,
) -> Path:
    """Create a synthetic run dir with metrics.json + the manifests we harvest."""
    run = root / run_id
    _write(
        run / "evaluate" / "metrics.json",
        {
            "S001-camera_0390": {"HOTA": 0.8, "IDF1": 0.78},  # per-camera (ignored by ledger)
            "COMBINED": {
                "HOTA": image_hota, "IDF1": 0.75, "MOTA": 0.78, "MOTP": 0.94, "CLR_F1": 0.89,
            },
            "mct_world": {
                "HOTA": world_hota, "DetA": 0.65, "AssA": 0.64, "IDF1": 0.80, "MOTA": 0.78,
                "d_max_m": 1.0, "dropped_detections": 6347, "frames_evaluated": 898,
            },
        },
    )
    _write(run / "evaluate" / "manifest.json",
           {"stage": "evaluate", "finished_at": finished, "runtime_sec": 2.5})
    _write(run / "detect" / "manifest.json",
           {"stage": "detect", "runtime_sec": 100.0, "params": {"model": "yolo11x", "conf_thresh": 0.5}})
    _write(run / "reid" / "manifest.json",
           {"stage": "reid", "runtime_sec": 50.0, "params": {"model": "solider_swin_small"}})
    _write(run / "pose" / "manifest.json",
           {"stage": "pose", "runtime_sec": 60.0, "params": {"model": "rtmpose-l"}})
    _write(run / "mct" / "manifest.json",
           {"stage": "mct", "runtime_sec": 5.0, "params": {
               "hard_world_gate": True,
               "tracking_params": {
                   "epsilon_scpt": eps_scpt, "epsilon_mcpt": 0.37,
                   "keypoint_condition_th": 3, "short_track_th": 120, "sim_th": 0.85,
               },
               "world_projection": {"method": "ankle_lower"},
           }})
    return run


def test_extract_record_reads_metrics_and_config(tmp_path):
    run = _make_run(tmp_path, "baseline")
    rec = R.extract_record(run, now="2026-05-31T00:00:00Z")
    assert rec is not None
    assert rec.run_id == "baseline"
    assert rec.experiment == "(baseline)"
    assert rec.date == "2026-05-30"
    assert rec.image["HOTA"] == 0.77
    assert rec.world["HOTA"] == 0.60
    assert rec.world["dropped_detections"] == 6347
    assert rec.runtime_sec == 2.5 + 100.0 + 50.0 + 60.0 + 5.0  # sum of stage manifests
    assert rec.config["epsilon_scpt"] == 0.15
    assert rec.config["projection"] == "ankle_lower"
    assert rec.config["detect"] == "yolo11x"


def test_extract_record_none_without_metrics(tmp_path):
    (tmp_path / "empty").mkdir()
    assert R.extract_record(tmp_path / "empty") is None


def test_infer_experiment_rules():
    known = frozenset({"eps_mcpt_sweep"})
    assert R.infer_experiment("baseline") == ("(baseline)", "", "")
    assert R.infer_experiment("eps_mcpt_sweep__0.45", known) == ("eps_mcpt_sweep", "0.45", "")
    assert R.infer_experiment("foo__bar", known)[0] == "ad-hoc"  # unknown prefix, not split
    assert R.infer_experiment("baseline_20260530_134259")[:2] == ("snapshot", "baseline")
    assert R.infer_experiment("v2_solider")[:2] == ("ad-hoc", "v2_solider")
    labels = {"v2_solider": {"experiment": "model-stack", "variant": "v2", "note": "n"}}
    assert R.infer_experiment("v2_solider", known, labels) == ("model-stack", "v2", "n")


def test_upsert_replaces_and_preserves_archived(tmp_path):
    run = _make_run(tmp_path, "x")
    rec1 = R.extract_record(run, archived={"remote": "onedrive", "remote_path": "p"})
    recs = R.upsert([], rec1)
    assert len(recs) == 1
    rec2 = R.extract_record(run)  # a fresh scan carries no archived marker...
    recs = R.upsert(recs, rec2)
    assert len(recs) == 1
    assert recs[0].archived == {"remote": "onedrive", "remote_path": "p"}  # ...must be preserved


def test_save_load_roundtrip(tmp_path):
    rec = R.extract_record(_make_run(tmp_path, "baseline"))
    ledger = tmp_path / "results" / "runs.jsonl"
    R.save_ledger(ledger, [rec])
    loaded = R.load_ledger(ledger)
    assert len(loaded) == 1
    assert loaded[0].run_id == "baseline"
    assert loaded[0].world["HOTA"] == 0.60


def test_render_markdown_has_both_views_and_deltas(tmp_path):
    base = R.extract_record(_make_run(tmp_path, "baseline", world_hota=0.60))
    variant = R.extract_record(
        _make_run(tmp_path, "eps_mcpt_sweep__0.45", world_hota=0.55,
                  finished="2026-05-31T10:00:00Z"),
        known_experiments=frozenset({"eps_mcpt_sweep"}),
    )
    md = R.render_markdown([base, variant], updated="2026-05-31")
    assert "## By day" in md
    assert "## By experiment" in md
    assert "### 2026-05-30" in md and "### 2026-05-31" in md
    assert "### (baseline)" in md and "### eps_mcpt_sweep" in md
    assert "0.6000" in md           # baseline world HOTA rendered
    assert "0.050" in md            # variant Δw magnitude (0.55 − 0.60)
