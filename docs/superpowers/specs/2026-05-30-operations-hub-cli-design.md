# Interactive Operations Hub (CLI) — design

- **Date:** 2026-05-30
- **Status:** approved (brainstorm), pending implementation plan
- **Scope:** an interactive, menu-driven operations hub that wraps the existing
  CLIs (`pipeline.py`, `experiments/run.py`, `experiments/compare.py`). A guided
  front-end, not a new execution path.

## Problem

Operating the pipeline today means remembering argparse flags across three CLIs
(`pipeline.py <stage> --config … --run-id … --force`, `experiments/run.py run …`,
`experiments/compare.py --sort-by …`) — captured in `docs/commands.md`, but still
manual. There's a read-only Streamlit dashboard for *viewing* runs, but nothing
interactive for *operating* them.

## Goals

1. A single interactive entry — `python pipeline.py menu` — that guides the user
   through the common operations (run a pipeline, run/compare experiments, browse
   runs, visualize, launch the dashboard) without memorizing flags.
2. **Reuse the existing CLIs as the single source of truth** — the hub builds and
   shells out the exact commands; it never reimplements pipeline logic.
3. Keep the base install lean: the interactive deps are an optional extra; the
   pipeline/CI/cron paths don't need them.

## Non-goals (YAGNI)

- No live-refresh monitor TUI (the Streamlit dashboard already views runs).
- No in-process pipeline execution — always shell out to the existing CLIs.
- No config *editing* from the hub (read/select only; `experiments/` is the
  mechanism for config variation).
- No remote/auth/multi-user.

## Design

### Module + entry point

- New module **`aic24_nvidia/hub.py`**. Input via **`questionary`** (arrow-key
  `select`/`checkbox`/`confirm`/`text`/`path`); output via **`rich`** (already a
  core dep) — `Table` for run lists, `Panel` for command confirmations.
- Entry: a **`menu`** subcommand added to `pipeline.py` → `python pipeline.py menu`.
  It lazy-imports `questionary`; on `ImportError` it prints
  `pip install -e ".[hub]"` and exits 2.

### Invocation model — shell out, never reimplement

The hub builds the exact argument lists for the existing CLIs and runs them via
`subprocess.run([sys.executable, …], cwd=<repo root>)` (same interpreter, per the
`sys.executable` fix), streaming output live so stage progress is visible. The
existing CLIs remain the single source of truth.

### Testability split

Pure, I/O-free helpers (unit-tested) vs a thin interactive shell (manually
verified). Helper signatures:

```python
def build_pipeline_cmd(config: Path, stages: list[str] | None, run_id: str | None,
                       force: bool) -> list[list[str]]:
    """stages=None -> a single `all` command; else one command per stage in
    registry order. Each inner list is argv for subprocess (starts with
    sys.executable, 'pipeline.py', ...)."""

def build_experiment_cmd(action: str, experiment: str | None = None,
                         variant: str | None = None, force: bool = False) -> list[str]:
    """action in {'list','status','ensure-baseline','run'}."""

def build_compare_cmd(sort_by: str | None = None) -> list[str]: ...

@dataclass
class RunInfo:
    run_id: str
    stages_present: dict[str, bool]      # stage name -> manifest.json exists & ok
    status: str                          # 'ok' | 'incomplete' | 'error'
    image_hota: float | None
    world_hota: float | None
    finished_at: str | None

def discover_runs(outputs_root: Path) -> list[RunInfo]:
    """Scan outputs/*/ ; read each stage manifest.json (status) and
    evaluate/metrics.json (headline metrics). Pure read, no subprocess."""

def load_metrics(run_dir: Path) -> dict | None: ...
```

The `questionary` loop calls these; tests inject choices / use a tmp `outputs/`
tree and assert on the returned argv lists / RunInfo rows.

### Top-level menu

```
aic24 — operations hub
❯ Run pipeline      guided: config → stages|all → run-id → force → confirm → run
  Experiments       list · ensure-baseline · run (experiment→variant) · compare
  Browse runs       rich table of outputs/ runs: stages, status, image/world HOTA
  Visualize         run-id → stage {detect,sct,mct} → viz
  Dashboard         launch streamlit (pipeline.py dashboard)
  Quit
```

### Per-menu flows

- **Run pipeline:** `select` config (glob `configs/*.yaml`, default
  `baseline.yaml`) → `select` "Run all" vs "Pick stages"; pick → `checkbox` over
  `registry.order()` → run-id (`text`, blank = auto, or `select` an existing run
  to resume) → `confirm` force → rich Panel of the resolved command(s) → final
  `confirm` → stream each command. "All" → one `all` command; specific → one
  command per selected stage in registry order.
- **Experiments:** submenu **List** / **Ensure-baseline** / **Run** / **Compare**.
  Run: `select` experiment (from `experiments._lib.load_registry`), then `select`
  variant or "all" → `experiments/run.py run <exp> [--variant V]`. Compare:
  `select` `--sort-by` from known metric keys → `experiments/compare.py`. All stream.
- **Browse runs:** `discover_runs(outputs/)` → rich Table
  `run_id │ stages ✓/· │ status │ image HOTA │ world HOTA │ finished_at`. No subprocess.
- **Visualize:** `select` run-id (from `discover_runs`) → `select` stage
  `{detect,sct,mct}` → use the run's `_config.yaml` if present else `select`
  config → `pipeline.py viz --config … --run-id … --stage …`.
- **Dashboard:** `text` port (default 8501) → `pipeline.py dashboard --port …`
  (streams until Ctrl-C).

### Dependencies

`pyproject.toml`:
```toml
[project.optional-dependencies]
hub = ["questionary>=2.0", "prompt_toolkit>=3.0"]
```
`rich` stays a core dep. Install the hub with `pip install -e ".[hub]"`.
`requirements.lock` is regenerated to include them.

## Testing

- **Unit (no TTY/subprocess):**
  - `build_pipeline_cmd`: all → one `all` argv; two stages → two argvs in registry
    order; `run_id`/`force` flags placed correctly; argv[0] == `sys.executable`.
  - `build_experiment_cmd` / `build_compare_cmd`: flags for each action.
  - `discover_runs`: tmp `outputs/` with fake stage manifests + an
    `evaluate/metrics.json` → asserts `RunInfo` rows (stages_present, status,
    image/world HOTA, finished_at).
  - `load_metrics`: present / missing.
- **Smoke (subprocess):** `pipeline.py menu` with `questionary` uninstalled prints
  the `pip install -e ".[hub]"` hint and exits 2 (simulate via a stubbed import).
- The `questionary` prompt loop + live streaming are the thin I/O shell — not
  unit-tested, verified manually.
- Full existing unit suite stays green.

## Risks & mitigations

- **New deps on a pinned stack.** Confined to the optional `[hub]` extra +
  `requirements.lock`; base/CI/cron installs are unaffected; lazy import with a
  friendly hint when absent.
- **Command drift** (hub builds wrong flags vs the real CLIs). Mitigated: the
  `build_*` helpers are the only place commands are constructed and are
  unit-tested against the known flag set; the hub never duplicates pipeline logic.
- **Stage list drift.** `build_pipeline_cmd` and the stages `checkbox` read
  `registry.order()` — the same single source of truth the pipeline uses.
- **Long-running streamed commands** (detect ~31 min). The hub streams live output
  and returns to the menu on completion; it does not capture/buffer.
