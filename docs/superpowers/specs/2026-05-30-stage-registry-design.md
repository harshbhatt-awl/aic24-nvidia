# Stage Registry — Phase 1 design

- **Date:** 2026-05-30
- **Status:** approved (brainstorm), pending implementation plan
- **Scope:** "Registry core + wiring" only. Builds on the Phase 0 robustness branch
  (`phase0-robustness`). Precursor to the rest of the original Phase 1 bundle
  (typed I/O contracts, run-local `external/`, manifest path fix, provenance),
  which are explicitly out of scope here.

## Problem

The pipeline has no `Stage` abstraction. Its stage graph is encoded as **four
parallel module-level dicts** in `pipeline.py` (`STAGE_RUNNERS`, `ORDER`,
`UPSTREAM_OF`, `STAGE_DIR_NAME`) and **duplicated** in `experiments/_lib.py`
(`STAGES`, `STAGE_DIR`). Adding, removing, or reordering a stage means editing
4–6 separate structures kept in sync by hand; a typo in one (`'sct'` vs `'sct_'`)
silently mis-gates a stage.

Separately, each stage's **external-symlink wiring** (how its output is exposed
to the upstream YACHIYO tooling, and how its inputs are cross-wired) lives as
ad-hoc `make_symlink`/`ensure_dir_clean` calls inside each `run()`, and is
**hand-mirrored a third time** in `experiments/_lib.py:prime_external_symlinks`
— whose own comment admits "source-of-truth is the grep on `stages/*.py`". That
mirror is already incomplete (it omits `mct`).

## Goals

1. One ordered **single source of truth** for stage metadata, consumed by both
   `pipeline.py` and `experiments/_lib.py`. Adding a stage = write the module +
   one `StageSpec` entry.
2. Each stage declares its symlink **wiring once**, consumed by all three sites
   that maintain it today (pre-run setup, post-promotion re-point, experiment
   cache-reuse). Delete the grep-mirror.
3. **Zero behavior change.** Same stage order, same gating, same on-disk
   symlinks. Guarded by tests.

## Non-goals (later phases — registry is shaped to accept them)

Typed I/O contracts; run-local `external/` tree (the concurrency-race fix —
becomes a one-place change once wiring is centralized); manifest `.tmp` path
proper fix; run-level provenance; swappable model/dataset/tracking backends;
setuptools entry-point discovery (`REGISTRY` can later grow `.from_entry_points()`).

## Design

### `aic24_nvidia/registry.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import Config

Link = tuple[Path, Path]  # (symlink_path, target_path)

def _no_wiring(run_dir: Path, cfg: Config, output_dir: Path) -> list[Link]:
    return []

@dataclass(frozen=True)
class StageSpec:
    name: str                                    # logical stage name, e.g. "detect"
    dir_name: str                                # output subdir under run_dir ("adapted" for adapt; else == name)
    upstream: tuple[str, ...]                    # gating dependencies, by stage name
    run: Callable[[Config, Path, str], None]     # the EXISTING run(cfg, run_dir, run_id) — signature unchanged
    wiring: Callable[[Path, Config, Path], list[Link]] = _no_wiring

# Ordered: list position == pipeline execution order.
REGISTRY: tuple[StageSpec, ...] = (
    StageSpec("adapt",    "adapted",  (),           adapt.run,          adapt.WIRING),
    StageSpec("frames",   "frames",   ("adapt",),   extract_frames.run, extract_frames.WIRING),
    StageSpec("detect",   "detect",   ("frames",),  detect.run,         detect.WIRING),
    StageSpec("reid",     "reid",     ("detect",),  reid.run,           reid.WIRING),
    StageSpec("pose",     "pose",     ("reid",),    pose.run,           pose.WIRING),
    StageSpec("sct",      "sct",      ("pose",),    sct.run,            sct.WIRING),
    StageSpec("mct",      "mct",      ("sct",),     mct.run,            mct.WIRING),
    StageSpec("evaluate", "evaluate", ("mct",),     evaluate.run,       _no_wiring),
)
```

The `REGISTRY` above shows the **final** state. During migration the `wiring`
field starts at its `_no_wiring` default (step 1, metadata only) and is populated
with each stage's `WIRING` fn in step 4.

Derived accessors replace the dicts: `order() -> list[str]`,
`by_name(name) -> StageSpec`, `upstream_of(name) -> tuple[str, ...]`,
`dir_name(name) -> str`. A `validate_registry()` runs at import and asserts:
names unique; every `upstream` entry exists and appears earlier in the list
(valid topological order); `dir_name`s unique; `run` callable.

### The wiring contract

Each stage's symlink behavior is one pure function
`wiring(run_dir, cfg, output_dir) -> list[Link]`, where `output_dir` is where the
stage's output currently lives. It returns **both** the stage's output-exposure
links and its input cross-wiring. Full set, capturing today's behavior exactly:

| stage | `WIRING(run_dir, cfg, output_dir)` returns |
|---|---|
| adapt | `[(external/Original, output_dir/"Original")]` |
| frames | `[(yachiyo/Original, run_dir/adapted/Original)]` (input link; `output_dir` unused) |
| detect | `[(external/Detection, output_dir)]` |
| reid | `[(external/EmbedFeature, output_dir)]` |
| pose | `[(external/Pose, output_dir)]` |
| sct | `[(yachiyo/Tracking, output_dir), (yachiyo/EmbedFeature, run_dir/reid), (yachiyo/Detection, run_dir/detect)]` |
| mct | sct's three + `(yachiyo/Pose, run_dir/pose)` |
| evaluate | `[]` |

(`external/...` = `cfg.external_root`; `yachiyo/...` = `cfg.yachiyo_root`;
`run_dir/<stage>` = `stage_dir(run_dir, dir_name)`.) Each `WIRING` fn lives in
its **own stage module**, co-located with the code it describes.

One declaration feeds all three call sites maintained separately today:
1. **Pre-run** → `wiring(run_dir, cfg, work_dir)` (output→`.tmp`, inputs→finals).
2. **Post-promotion** → `wiring(run_dir, cfg, final_dir)` (output→final).
3. **Experiment cache-reuse** → `prime_external_symlinks` loops reused specs
   calling `wiring(run_dir, cfg, final_dir)`.

A stage never redeclares another stage's *output* link; the orchestrator (for a
full run) or `prime_external_symlinks` (for cache-reuse) ensures upstream links
exist by applying upstream stages' wiring — matching today (e.g. `reid` relies on
`adapt` having set `external/Original`).

### Integration: wiring-aware `atomic_stage`

`atomic_stage` owns the `work_dir`-create → body → write-manifest → rename
lifecycle, so it applies wiring at the two correct moments. To avoid an import
cycle (`base` → `registry` → stage modules → `base`), `base.py` does **not**
import `registry`; the wiring callable is passed in:

```python
@contextmanager
def atomic_stage(run_dir, stage, run_id, *, cfg=None, wiring=_no_wiring):
    tmp = stage_tmp_dir(run_dir, stage); ... ; tmp.mkdir(parents=True, exist_ok=True)
    if cfg is not None:
        _apply_links(wiring(run_dir, cfg, tmp))        # pre-run
    ctx = StageCtx(...); yield ctx
    ... write manifest; rename tmp -> final; rewrite .tmp paths in manifest ...
    if cfg is not None:
        _apply_links(wiring(run_dir, cfg, final))      # post-promotion
```

`_apply_links` is `ensure_dir_clean` + `make_symlink` per pair. `cfg`/`wiring`
are **optional** (default no-op), so existing `test_base_stage.py` calls work
unchanged. Each stage `run()` then passes `cfg=cfg, wiring=WIRING` and **deletes
its inline `ensure_dir_clean`/`make_symlink` calls** (in-body setup and the
trailing re-point line).

**Known harmless deviation:** `adapt` gains a pre-run link
`external/Original -> adapt.tmp/Original`. Nothing reads `external/Original`
during `adapt` (it is the first stage), and the post-promotion link that
downstream depends on is identical to today. The characterization test asserts
the post-promotion link set, which is unchanged.

### Consumer rewrites

- **`pipeline.py`** — delete the 4 dicts. `_gate_stage`, `cmd_stage`, `cmd_all`,
  `cmd_viz` read `registry.order()` / `upstream_of()` / `dir_name()` /
  `by_name(s).run`. (`cmd_viz`'s per-stage `if/elif` is untouched — that's a
  separate VizRegistry concern, deferred.)
- **`experiments/_lib.py`** — delete `STAGES`/`STAGE_DIR` (import from
  `registry`). `prime_external_symlinks` collapses to: for each reused stage,
  `_apply_links(spec.wiring(run_dir, cfg, stage_dir(run_dir, spec.dir_name)))`.
  The hand-maintained `plan` dict and its grep comment are deleted, and the
  `mct` gap it documented is closed for free.

## Migration plan (five independently-green commits)

1. Add `registry.py` (with `validate_registry()` at import) + a registry-invariant
   test. Metadata only — the `wiring` field defaults to no-op (populated in step
   4). Added *alongside* the old dicts; nothing consumes it yet.
2. Repoint `pipeline.py` and `experiments/_lib.py` to the registry; delete the
   dicts. `test_pipeline_registry_consistency.py` flips from "the 4 dicts agree"
   to "the registry satisfies its invariants"; the duplicate-drift assertion is
   retired (no duplicate remains). No behavior change (same order/deps).
3. Add a **characterization test** snapshotting each stage's `(link → target)`
   set from *current* behavior — the golden.
4. Extract each stage's inline symlinks into a module-level `WIRING` fn; wire them
   into the specs; make `atomic_stage` wiring-aware; strip the inline calls from
   each `run()`. The characterization test must reproduce the golden exactly.
   ← the only behavioral step, fully guarded.
5. Rewrite `prime_external_symlinks` over the registry; update experiment-harness
   tests.

## Testing

- **Registry invariants:** unique names; `upstream` refs exist and precede;
  unique `dir_name`s; `run` callable. (New test.)
- **Wiring characterization:** for each stage, assert
  `WIRING(run_dir, cfg, output_dir)` returns the expected `(link → target)` set,
  golden captured from current behavior. The safety net for step 4.
- **`atomic_stage` wiring:** exercise with a fake `wiring` fn asserting it is
  applied pre-run (target = `.tmp`) and post-promotion (target = final).
- All 121 existing unit tests stay green. Everything is GPU-free — wiring fns are
  pure path math; no model or subprocess involved.

## Risks & mitigations

- **Behavior drift in symlinks (step 4).** Mitigated by the golden
  characterization test captured *before* the refactor (step 3).
- **Import cycle.** Avoided by keeping `WIRING` fns in stage modules and passing
  the callable into `atomic_stage` (base never imports registry).
- **Experiment cache-reuse regression.** `prime_external_symlinks` is rewritten
  last (step 5) over the same wiring the live run uses, so reuse and live paths
  can no longer diverge; harness tests updated in the same commit.
