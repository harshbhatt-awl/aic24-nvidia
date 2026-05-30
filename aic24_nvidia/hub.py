"""Interactive operations hub for the aic24-nvidia pipeline.

Pure helpers (command-builders + run discovery) are I/O-free and unit-tested.
The interactive loop (run_hub) lazy-imports questionary + rich, so importing
this module for the helpers does not require the optional [hub] deps. The hub
shells out to the existing CLIs; it never reimplements pipeline logic.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from .registry import dir_name as _dir_name
from .registry import order as _stage_order

_REPO_ROOT = Path(__file__).resolve().parent.parent


def load_metrics(run_dir: Path) -> dict | None:
    p = Path(run_dir) / "evaluate" / "metrics.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _mean_image_hota(metrics: dict) -> float | None:
    vals = [
        m["HOTA"] for k, m in metrics.items()
        if k != "mct_world" and isinstance(m, dict) and isinstance(m.get("HOTA"), (int, float))
    ]
    return sum(vals) / len(vals) if vals else None


def _world_hota(metrics: dict) -> float | None:
    mw = metrics.get("mct_world")
    if isinstance(mw, dict) and isinstance(mw.get("HOTA"), (int, float)):
        return float(mw["HOTA"])
    return None


@dataclass
class RunInfo:
    run_id: str
    stages_present: dict[str, bool]
    status: str
    image_hota: float | None
    world_hota: float | None
    finished_at: str | None


def discover_runs(outputs_root: Path) -> list[RunInfo]:
    """Scan outputs/*/ for runs; read each stage manifest (presence) and the
    evaluate manifest + metrics.json (status + headline HOTA). Pure read."""
    outputs_root = Path(outputs_root)
    if not outputs_root.exists():
        return []
    runs: list[RunInfo] = []
    for d in sorted((p for p in outputs_root.iterdir() if p.is_dir()),
                    key=lambda p: p.name, reverse=True):
        present = {s: (d / _dir_name(s) / "manifest.json").exists() for s in _stage_order()}
        status, finished_at = "incomplete", None
        eval_manifest = d / "evaluate" / "manifest.json"
        if eval_manifest.exists():
            try:
                em = json.loads(eval_manifest.read_text())
                status = em.get("status", "ok")
                finished_at = em.get("finished_at")
            except (json.JSONDecodeError, OSError):
                status = "error"
        metrics = load_metrics(d)
        runs.append(RunInfo(
            run_id=d.name,
            stages_present=present,
            status=status,
            image_hota=_mean_image_hota(metrics) if metrics else None,
            world_hota=_world_hota(metrics) if metrics else None,
            finished_at=finished_at,
        ))
    return runs


def build_pipeline_cmd(config: Path, stages: list[str] | None,
                       run_id: str | None, force: bool) -> list[list[str]]:
    """Return argv lists for pipeline.py. stages=None -> a single 'all' command;
    otherwise one command per selected stage, in registry order. Each argv runs
    under sys.executable (same interpreter as the hub)."""
    common = ["--config", str(config)]
    if run_id:
        common += ["--run-id", run_id]
    if force:
        common += ["--force"]
    base = [sys.executable, "pipeline.py"]
    if not stages:
        return [base + ["all"] + common]
    selected = set(stages)
    ordered = [s for s in _stage_order() if s in selected]
    return [base + [s] + common for s in ordered]
