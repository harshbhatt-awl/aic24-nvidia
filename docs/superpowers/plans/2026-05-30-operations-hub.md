# Interactive Operations Hub Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `python pipeline.py menu` — an interactive, arrow-key operations hub that guides the user through running the pipeline, running/comparing experiments, browsing runs, visualizing, and launching the dashboard, by shelling out to the existing CLIs.

**Architecture:** One module `aic24_nvidia/hub.py` holds pure, I/O-free command-builders + run-discovery (unit-tested) plus a thin `questionary`+`rich` interactive loop (lazy-imported, manually verified). The hub never reimplements pipeline logic — it builds exact argv lists and runs them via `subprocess.run([sys.executable, …], cwd=repo)`, streaming live.

**Tech Stack:** Python 3.14, `questionary` (arrow-key prompts) + `prompt_toolkit` + `rich` (rendering) in an optional `[hub]` extra; stdlib `subprocess`/`dataclasses`.

**Spec:** `docs/superpowers/specs/2026-05-30-operations-hub-cli-design.md`

**Conventions:** run `pytest` as `.venv/bin/python -m pytest`. Keep `.venv/bin/python -m pytest tests/unit -q` green after every task; run `.venv/bin/ruff check .` before each commit.

---

### Task 1: Add the `[hub]` optional-dependency extra

**Files:**
- Modify: `pyproject.toml`
- Modify: `requirements.lock`

- [ ] **Step 1: Add the extra**

In `pyproject.toml`, under `[project.optional-dependencies]` (next to the existing `dev` extra), add:

```toml
hub = [
    "questionary>=2.0",
    "prompt_toolkit>=3.0",
    "rich>=13",
]
```

- [ ] **Step 2: Install it**

Run: `.venv/bin/python -m pip install -e ".[hub]"`
Expected: installs `questionary`, `prompt_toolkit` (and confirms `rich` present).

- [ ] **Step 3: Verify imports**

Run: `.venv/bin/python -c "import questionary, prompt_toolkit, rich; print('hub deps ok')"`
Expected: `hub deps ok`

- [ ] **Step 4: Regenerate the lockfile**

Run:
```bash
{
  echo "# Exact environment lock for aic24-nvidia (generated via pip freeze on the verified stack)."
  echo "# Python 3.14, torch 2.12+cu130. Regenerate with: .venv/bin/python -m pip freeze"
  echo "# Install with: pip install -r requirements.lock  (after the torch cu130 index is configured)"
  .venv/bin/python -m pip freeze 2>/dev/null | grep -ivE '^(aic24-nvidia|-e )' | grep -vE '@ file://'
} > requirements.lock
```
Expected: `requirements.lock` now contains `questionary==` and `prompt_toolkit==` lines (`grep -E "questionary|prompt-toolkit" requirements.lock`).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml requirements.lock
git commit -m "feat(hub): add [hub] optional extra (questionary, prompt_toolkit, rich)"
```

---

### Task 2: Run discovery (`RunInfo`, `load_metrics`, `discover_runs`)

**Files:**
- Create: `aic24_nvidia/hub.py`
- Test: `tests/unit/test_hub.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_hub.py
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
    _write_manifest(run / "adapted")          # adapt stage dir is "adapted"
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
    assert abs(r.image_hota - 0.75) < 1e-9      # mean of 0.70, 0.80
    assert abs(r.world_hota - 0.5282) < 1e-9


def test_discover_runs_empty_when_no_outputs(tmp_path):
    assert hub.discover_runs(tmp_path / "nothing") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_hub.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'aic24_nvidia.hub'`.

- [ ] **Step 3: Create `hub.py` with the discovery helpers**

```python
# aic24_nvidia/hub.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_hub.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add aic24_nvidia/hub.py tests/unit/test_hub.py
git commit -m "feat(hub): run discovery (RunInfo, discover_runs, load_metrics)"
```

---

### Task 3: `build_pipeline_cmd`

**Files:**
- Modify: `aic24_nvidia/hub.py`
- Test: `tests/unit/test_hub.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_hub.py`:

```python
import sys


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
    # Pass out of order; must come back in registry order, one command per stage.
    cmds = hub.build_pipeline_cmd(Path("c.yaml"), ["detect", "adapt"], None, False)
    assert [c[2] for c in cmds] == ["adapt", "detect"]
    assert all(c[0] == sys.executable and c[1] == "pipeline.py" for c in cmds)
    assert cmds[0][2:] == ["adapt", "--config", "c.yaml"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_hub.py -k build_pipeline -q`
Expected: FAIL — `AttributeError: module 'aic24_nvidia.hub' has no attribute 'build_pipeline_cmd'`.

- [ ] **Step 3: Implement `build_pipeline_cmd`**

Add to `aic24_nvidia/hub.py` (after `discover_runs`):

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_hub.py -k build_pipeline -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add aic24_nvidia/hub.py tests/unit/test_hub.py
git commit -m "feat(hub): build_pipeline_cmd (all vs ordered stages, run-id, force)"
```

---

### Task 4: `build_experiment_cmd` + `build_compare_cmd`

**Files:**
- Modify: `aic24_nvidia/hub.py`
- Test: `tests/unit/test_hub.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_hub.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_hub.py -k "experiment_cmd or compare_cmd" -q`
Expected: FAIL — `AttributeError: ... has no attribute 'build_experiment_cmd'`.

- [ ] **Step 3: Implement the builders**

Add to `aic24_nvidia/hub.py`:

```python
def build_experiment_cmd(action: str, experiment: str | None = None,
                         variant: str | None = None, force: bool = False) -> list[str]:
    """action in {'list','status','ensure-baseline','run'}."""
    cmd = [sys.executable, "experiments/run.py", action]
    if action == "run":
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_hub.py -k "experiment_cmd or compare_cmd" -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add aic24_nvidia/hub.py tests/unit/test_hub.py
git commit -m "feat(hub): build_experiment_cmd + build_compare_cmd"
```

---

### Task 5: Interactive shell (`_require_interactive`, `run_hub`, flows)

**Files:**
- Modify: `aic24_nvidia/hub.py`
- Test: `tests/unit/test_hub.py`

The `run_hub` loop and flow functions are the thin I/O shell (manually verified, not unit-tested). Only `_require_interactive` has a unit test (the missing-dep guard).

- [ ] **Step 1: Write the failing test for the dependency guard**

Append to `tests/unit/test_hub.py`:

```python
import pytest


def test_require_interactive_missing_dep_exits_with_hint(monkeypatch, capsys):
    # Make `import questionary` raise ImportError without uninstalling it.
    monkeypatch.setitem(sys.modules, "questionary", None)
    with pytest.raises(SystemExit) as exc:
        hub._require_interactive()
    assert exc.value.code == 2
    assert 'pip install -e ".[hub]"' in capsys.readouterr().err
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_hub.py -k require_interactive -q`
Expected: FAIL — `AttributeError: ... has no attribute '_require_interactive'`.

- [ ] **Step 3: Implement the interactive shell**

Add to `aic24_nvidia/hub.py`:

```python
def _require_interactive():
    """Import the optional interactive deps, or exit(2) with an install hint."""
    try:
        import questionary
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
```

- [ ] **Step 4: Run the guard test + full suite**

Run: `.venv/bin/python -m pytest tests/unit/test_hub.py -q`
Expected: PASS (all hub tests, including `test_require_interactive_missing_dep_exits_with_hint`).

- [ ] **Step 5: Commit**

```bash
git add aic24_nvidia/hub.py tests/unit/test_hub.py
git commit -m "feat(hub): interactive run_hub loop + flows + missing-dep guard"
```

---

### Task 6: `menu` subcommand on `pipeline.py`

**Files:**
- Modify: `pipeline.py`
- Test: `tests/unit/test_hub.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_hub.py`:

```python
def test_menu_subcommand_dispatches_to_run_hub(monkeypatch):
    import pipeline
    called = []
    # cmd_menu does `from aic24_nvidia import hub; hub.run_hub()`, so patch the attr.
    monkeypatch.setattr("aic24_nvidia.hub.run_hub", lambda: called.append(True))
    rc = pipeline.main(["menu"])
    assert called == [True]
    assert rc == 0
```

(Note: `test_pipeline_registry_consistency.py` already puts the repo root on
`sys.path`, but this test imports `pipeline` directly; add the repo-root
`sys.path` insert at the top of `test_hub.py` if not already present:)

```python
# near the top of tests/unit/test_hub.py, after `import sys`:
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_hub.py -k menu_subcommand -q`
Expected: FAIL — argparse errors on the unknown `menu` choice (SystemExit) or `called == []`.

- [ ] **Step 3: Add the `menu` subcommand**

In `pipeline.py`, add the command handler (next to `cmd_dashboard`):

```python
def cmd_menu(args) -> None:
    from aic24_nvidia import hub
    hub.run_hub()
```

And register the subparser in `main()` (next to the `dashboard` subparser):

```python
    mp = sub.add_parser("menu")
    mp.set_defaults(func=cmd_menu)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_hub.py -k menu_subcommand -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline.py tests/unit/test_hub.py
git commit -m "feat(hub): add `python pipeline.py menu` subcommand"
```

---

### Task 7: Docs + final verification

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/commands.md`

- [ ] **Step 1: Document in CLAUDE.md**

In `CLAUDE.md`, under the "Running" section, add after the `dashboard` line:

```markdown
python pipeline.py menu                            # interactive operations hub (needs `pip install -e ".[hub]"`)
```

- [ ] **Step 2: Document in the runbook**

In `docs/commands.md`, under "## 2. Pipeline", add:

```markdown
# Interactive operations hub (arrow-key menu over pipeline/experiments/viz/dashboard).
# Needs the optional extra: pip install -e ".[hub]"
python pipeline.py menu
```

- [ ] **Step 3: Full suite + lint + import smoke**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: PASS (no failures; hub tests added).

Run: `.venv/bin/ruff check .`
Expected: `All checks passed!`

Run: `.venv/bin/python -c "import aic24_nvidia.hub as h; print(h.build_compare_cmd('mct_world.HOTA'))"`
Expected: prints the compare argv (confirms the module imports without the interactive deps loaded).

- [ ] **Step 4: Manual smoke (interactive — not automated)**

Run: `.venv/bin/python pipeline.py menu`
Expected: the arrow-key menu renders; "Browse runs" shows a table of `outputs/` runs; "Quit" exits cleanly. (Ctrl-C also exits.)

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md docs/commands.md
git commit -m "docs(hub): document the interactive operations hub"
```

---

## Self-review notes (for the executor)

- **Spec coverage:** `[hub]` extra + lazy import (T1, T5), pure command-builders (T3, T4), run discovery (T2), interactive flows for all five menu items (T5), `menu` entry point (T6), docs (T7). All spec sections covered.
- **Shell-out model:** every flow builds argv via the `build_*` helpers / inline `[sys.executable, "pipeline.py", …]` and runs through `_run`; no pipeline logic is duplicated.
- **Testability:** `hub.py` top-level imports are stdlib + `registry` only; `questionary`/`rich` are imported lazily inside `_require_interactive`/flows, so `import aic24_nvidia.hub` and all pure-helper tests work without the `[hub]` deps installed.
- **Type/name consistency:** `build_pipeline_cmd` (list[list[str]]), `build_experiment_cmd`/`build_compare_cmd` (list[str]), `discover_runs`→`list[RunInfo]`, `RunInfo(run_id, stages_present, status, image_hota, world_hota, finished_at)`, `run_hub`, `_require_interactive`, `_run` — used identically across tasks. Stage list always from `registry.order()`; dir mapping from `registry.dir_name`.
- **Lean base install:** interactive deps are opt-in (`[hub]`); CI/cron/headless installs and the unit suite don't need them.
