# Results ledger — durable, day-wise + experiment-wise

**Date:** 2026-05-31
**Status:** design (implemented same session)
**Related:** `scripts/archive_run.sh` (OneDrive archiving), `experiments/compare.py`
(live A/B snapshot).

## Problem

Every completed run writes its results *inside* its run dir:
`outputs/<run_id>/evaluate/metrics.json` (image `COMBINED` + scene `mct_world`)
plus per-stage `manifest.json` (params + `runtime_sec` + `finished_at`). That is
fine while the run lives on disk, but two things make it inadequate:

1. **Archiving deletes the run.** `scripts/archive_run.sh` tars a finished run to
   OneDrive and removes the local dir to reclaim space. The results — the whole
   point of the run — disappear from disk with it.
2. **No cross-run view.** There is no single place that answers "what did we try,
   on which day, and how did it score?" `experiments/compare.py` is a *live
   snapshot* of whatever is currently in `outputs/` and only knows
   registry-named variants (`<exp>__<variant>`) + `baseline`; ad-hoc runs
   (`v2_solider`, `scene041`, dated baseline snapshots) and any archived-away run
   are invisible to it, and it has no time axis.

We want results **stored in a durable format, organised day-wise AND
experiment-wise**, that survives archiving and is easy to read.

## Design

A small, git-tracked ledger under `results/`, plus a library + CLI to maintain it.

### Data — `results/runs.jsonl`

One JSON object per line (append-friendly, diff-friendly, merge-friendly), keyed
by `run_id`. Fields:

```jsonc
{
  "run_id": "baseline",
  "date": "2026-05-30",                 // from evaluate manifest finished_at (UTC date)
  "finished_at": "2026-05-30T21:25:23Z",
  "experiment": "(baseline)",           // inferred (see below)
  "variant": "",
  "image":  {"HOTA":..,"IDF1":..,"MOTA":..,"MOTP":..,"CLR_F1":..},   // from COMBINED
  "world":  {"HOTA":..,"DetA":..,"AssA":..,"IDF1":..,"MOTA":..,      // from mct_world
             "dropped_detections":.., "frames_evaluated":.., "d_max_m":..},
  "runtime_sec": 2.47,                  // sum of per-stage manifest runtime_sec
  "config": {"detect":"yolo11x","reid":"solider_swin_small","pose":"rtmpose-l",
             "epsilon_scpt":0.15,"epsilon_mcpt":0.37,"keypoint_condition_th":3,
             "short_track_th":120,"sim_th":0.85,"projection":"ankle_lower",
             "hard_world_gate":true},   // fingerprint harvested from stage manifests
  "note": "locked v3.3 baseline",
  "archived": {"remote":"onedrive","remote_path":"onedrive:aic24/outputs-archive/x.tar.zst",
               "archived_at":"2026-05-31T..Z"} | null,
  "git": {"branch":"main","commit":"6981bcc"},   // recorded-at-time, best-effort
  "recorded_at": "2026-05-31T..Z"
}
```

The record is built **entirely from files already inside the run dir** — no extra
state the pipeline must emit. Image metrics come from the `COMBINED` block
directly (not re-averaged); world metrics from `mct_world`; the config
fingerprint from `detect/reid/pose/sct/mct` manifest `params`; date + runtime
from manifests.

### Views — `results/README.md` (auto-rendered)

Generated from the JSONL; never hand-edited. Three parts:

- **Summary header** — baseline image/world HOTA, best world-HOTA run, run count,
  last-updated date, metric legend.
- **## By day** — `### YYYY-MM-DD` (newest first); each day a table of the runs
  that finished that day, sorted by world HOTA, columns:
  `run_id | experiment/variant | img HOTA | w HOTA | w AssA | w IDF1 | Δw | config | runtime`.
- **## By experiment** — `### <experiment>` (baseline first, then alpha); each a
  table of its variants:
  `run_id | variant | img HOTA | w HOTA | w DetA | w AssA | w IDF1 | Δw | config | date`.

`Δw` = world HOTA minus the baseline's world HOTA (world HOTA is the project's
priority metric per `pipeline-state`).

### Experiment / variant inference

`infer_experiment(run_id, known_experiments, labels)`:
1. `labels[run_id]` (from `results/labels.json`) wins if present — `{experiment,
   variant, note}`. This is how ad-hoc runs get human-meaningful grouping.
2. `run_id == "baseline"` → `("(baseline)", "")`.
3. `"<exp>__<variant>"` and `exp` in the registry → split (the
   `experiments/_lib.py:variant_run_id` convention).
4. `"<name>_YYYYMMDD_HHMMSS"` → `("snapshot", "<name>")`.
5. else → `("ad-hoc", run_id)`.

### Code

- **`aic24_nvidia/results.py`** (library, stdlib-only, mirrors `compare.py`'s
  no-deps style): `RunRecord`, `extract_record(run_dir, ...)`,
  `infer_experiment(...)`, `load_ledger`/`upsert`/`save_ledger`,
  `render_markdown(records)`. Pure functions, unit-tested, no GPU. Takes
  `known_experiments` as a parameter so it does not depend on `experiments/`.
  `upsert` preserves a prior record's `archived`/`note` when a fresh `scan`
  record lacks them (so re-scanning never wipes the archived marker).
- **`scripts/results.py`** (CLI, adds repo root to `sys.path` like `compare.py`):
  - `scan` — record every `outputs/*/evaluate/metrics.json`, refresh both files.
  - `add <run_id> [--archived-remote R --archived-path P]` — record one run;
    used by `archive_run.sh`.
  - `render` — re-render `README.md` from the JSONL.
- **`results/labels.json`** — seeded with the current ad-hoc runs.
- **`tests/unit/test_results.py`** — extraction from a synthetic run dir,
  inference cases, render contains both sections + correct deltas, upsert
  preserves `archived`.

### Integration with archiving

`scripts/archive_run.sh`, after upload+verify and **before** deleting the local
run dir, calls:

```
python scripts/results.py add "$run_id" --archived-remote "$REMOTE" --archived-path "$remote_path"
```

So the ledger captures the results (and the OneDrive location) before the bytes
leave the disk. The CLI tolerates a bare (non-venv) `python` — registry/label
lookups degrade to empty, metric extraction still works.

## Non-goals (YAGNI)

- No `pipeline.py results` subcommand yet (standalone script is enough; can add
  later if it earns its place).
- No DB / web UI — Markdown + JSONL is the right altitude for "understand
  properly" and is git-diffable.
- `compare.py` is **not** refactored; it keeps its registry-delta A/B role. The
  ledger is the durable cross-time record. Both coexist.

## Acceptance

- `python scripts/results.py scan` produces `results/runs.jsonl` (4 current runs)
  and a `results/README.md` with a By-day and a By-experiment section showing the
  v3.3 baseline (image HOTA 0.7677 / world HOTA 0.6479) and the others with
  correct Δw.
- `pytest tests/unit/test_results.py` passes.
- Archiving a run records it (with `archived` set) before deletion.
