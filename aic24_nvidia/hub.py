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


def _image_hota(metrics: dict) -> float | None:
    combined = metrics.get("COMBINED")
    if isinstance(combined, dict) and isinstance(combined.get("HOTA"), (int, float)):
        return float(combined["HOTA"])
    vals = [
        m["HOTA"] for k, m in metrics.items()
        if "camera_" in k and isinstance(m, dict) and isinstance(m.get("HOTA"), (int, float))
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
            image_hota=_image_hota(metrics) if metrics else None,
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


def build_experiment_cmd(action: str, experiment: str | None = None,
                         variant: str | None = None, force: bool = False) -> list[str]:
    """action in {'list','status','ensure-baseline','run'}."""
    cmd = [sys.executable, "experiments/run.py", action]
    if action == "run":
        if experiment is None:
            raise ValueError("experiment is required for action 'run'")
        cmd.append(experiment)
        if variant:
            cmd += ["--variant", variant]
        if force:
            cmd += ["--force"]
    elif action == "ensure-baseline" and force:
        cmd += ["--force"]
    return cmd


def build_compare_cmd(sort_by: str | None = None) -> list[str]:
    cmd = [sys.executable, "experiments/compare.py"]
    if sort_by:
        cmd += ["--sort-by", sort_by]
    return cmd


def _require_interactive():
    """Import the optional interactive deps, or exit(2) with an install hint."""
    try:
        import questionary
        import rich  # noqa: F401  (flows import rich.console/table/panel lazily)
    except ImportError as e:
        print(
            'interactive hub needs extra deps:\n  pip install -e ".[hub]"',
            file=sys.stderr,
        )
        raise SystemExit(2) from e
    return questionary


def _run(argv_lists: list[list[str]], console) -> int:
    """Run each argv under the hub's interpreter from the repo root, streaming
    output live. Stops on the first non-zero exit."""
    import subprocess
    for argv in argv_lists:
        console.print(f"[bold cyan]$ {' '.join(argv)}[/]")
        rc = subprocess.run(argv, cwd=str(_REPO_ROOT)).returncode
        if rc != 0:
            console.print(f"[red]command failed (rc={rc})[/]")
            return rc
    console.print("[green]done[/]")
    return 0


def _flow_run_pipeline(q, console):
    configs = sorted((_REPO_ROOT / "configs").glob("*.yaml"))
    if not configs:
        console.print("[red]no configs/*.yaml found[/]")
        return
    cfg = q.select("Config:", choices=[str(c.relative_to(_REPO_ROOT)) for c in configs]).ask()
    if not cfg:
        return
    mode = q.select("Stages:", choices=["Run all", "Pick stages"]).ask()
    if not mode:
        return
    stages = None
    if mode == "Pick stages":
        stages = q.checkbox("Select stages:", choices=_stage_order()).ask()
        if not stages:
            return
    run_id = (q.text("Run-id (blank = auto):").ask() or "").strip() or None
    force = q.confirm("Force re-run?", default=False).ask()
    cmds = build_pipeline_cmd(Path(cfg), stages, run_id, force)
    from rich.panel import Panel
    console.print(Panel("\n".join(" ".join(c) for c in cmds), title="will run"))
    if q.confirm("Proceed?", default=True).ask():
        _run(cmds, console)


def _flow_experiments(q, console):
    action = q.select(
        "Experiments:",
        choices=["list", "status", "ensure-baseline", "run", "compare", "Back"],
    ).ask()
    if action in (None, "Back"):
        return
    if action == "compare":
        sort_by = (q.text("--sort-by (blank = default):").ask() or "").strip() or None
        _run([build_compare_cmd(sort_by)], console)
        return
    if action == "run":
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        from experiments._lib import load_registry
        exps = load_registry(_REPO_ROOT / "experiments" / "registry.yaml")
        if not exps:
            console.print("[red]no experiments defined[/]")
            return
        exp = q.select("Experiment:", choices=[e["id"] for e in exps]).ask()
        if not exp:
            return
        chosen = next(e for e in exps if e["id"] == exp)
        variant = q.select(
            "Variant:", choices=["(all)"] + [v["name"] for v in chosen["variants"]]
        ).ask()
        if not variant:
            return
        force = q.confirm("Force?", default=False).ask()
        cmd = build_experiment_cmd(
            "run", experiment=exp,
            variant=(None if variant == "(all)" else variant), force=force,
        )
        _run([cmd], console)
        return
    force = q.confirm("Force rebuild?", default=False).ask() if action == "ensure-baseline" else False
    _run([build_experiment_cmd(action, force=force)], console)


def _flow_browse_runs(console):
    from rich.table import Table
    runs = discover_runs(_REPO_ROOT / "outputs")
    if not runs:
        console.print("[yellow]no runs under outputs/[/]")
        return
    t = Table(title="runs")
    for col in ("run_id", "stages", "status", "image HOTA", "world HOTA", "finished"):
        t.add_column(col)
    for r in runs:
        stages = "".join("✓" if r.stages_present.get(s) else "·" for s in _stage_order())
        t.add_row(
            r.run_id, stages, r.status,
            f"{r.image_hota:.4f}" if r.image_hota is not None else "-",
            f"{r.world_hota:.4f}" if r.world_hota is not None else "-",
            r.finished_at or "-",
        )
    console.print(t)


def _flow_visualize(q, console):
    runs = discover_runs(_REPO_ROOT / "outputs")
    if not runs:
        console.print("[yellow]no runs under outputs/[/]")
        return
    run_id = q.select("Run-id:", choices=[r.run_id for r in runs]).ask()
    if not run_id:
        return
    stage = q.select("Stage:", choices=["detect", "sct", "mct"]).ask()
    if not stage:
        return
    run_cfg = _REPO_ROOT / "outputs" / run_id / "_config.yaml"
    if run_cfg.exists():
        cfg = str(run_cfg.relative_to(_REPO_ROOT))
    else:
        configs = sorted((_REPO_ROOT / "configs").glob("*.yaml"))
        cfg = q.select("Config:", choices=[str(c.relative_to(_REPO_ROOT)) for c in configs]).ask()
        if not cfg:
            return
    _run([[sys.executable, "pipeline.py", "viz",
           "--config", cfg, "--run-id", run_id, "--stage", stage]], console)


def _flow_dashboard(q, console):
    port = (q.text("Port:", default="8501").ask() or "").strip()
    if not port:
        return
    _run([[sys.executable, "pipeline.py", "dashboard", "--port", port]], console)


def run_hub() -> None:
    q = _require_interactive()
    from rich.console import Console
    console = Console()
    flows = {
        "Run pipeline": lambda: _flow_run_pipeline(q, console),
        "Experiments": lambda: _flow_experiments(q, console),
        "Browse runs": lambda: _flow_browse_runs(console),
        "Visualize": lambda: _flow_visualize(q, console),
        "Dashboard": lambda: _flow_dashboard(q, console),
    }
    while True:
        choice = q.select("aic24 — operations hub",
                          choices=[*flows.keys(), "Quit"]).ask()
        if choice in (None, "Quit"):
            break
        flows[choice]()
