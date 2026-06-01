# Post-MCT Over-Split Stitch — Design

**Date:** 2026-06-01
**Status:** approved (brainstorming) → next: writing-plans
**Branch:** `feat/over-split-stitch`

## Problem

The v3.3 pipeline **over-segments** identities. On scene_044 it emits **32 predicted
world global IDs for 25 ground-truth people** (verified: `outputs/baseline/evaluate/mct_world_pred.txt`
has 32 unique ids; `outputs/baseline/adapted/scene_001_gt_world.txt` has 25). The 7 extra ids are
fragments — one real person split across multiple `GlobalOfflineID`s. This is a **structural**
greedy-clustering artifact (YACHIYO's MCPT merge is irreversible: once two tracks are not merged early,
they never reconsider), which is why the SOLIDER appearance fine-tune could not fix it (it sharpens
per-crop appearance margin; the splits are not an appearance-margin problem). World **AssA (0.6423)** is
the headroom these fragments cost, not the saturated appearance axis.

### Measured evidence (the gain this design captures)

Merging a single clean sequential fragment pair — `gid 3` (frames 1–320) → `gid 8` (frames 328–900),
endpoints **0.34 m / 8 frames** apart — and re-scoring lifted **world HOTA 0.6479 → 0.6567 (+0.0088)**,
**AssA +0.0174**, **IDF1 +0.0163**, **DetA flat**. The over-split-repair family has a measured oracle
ceiling of ≈ **+0.031** world HOTA (clean sequential ~+0.019 + long-gap sequential + 3 concurrent
cross-camera splits). **This design targets only the clean sequential subset** (see Scope).

## Goal

Add a config-gated, unit-tested, owned **post-MCT over-split stitch** that merges fragmented world
global IDs whose tracks are sequential (non-overlapping in time) and adjacent in the world plane, run as
a pure **evaluate-stage** transform. Recover the clean ~+0.009–0.019 world HOTA with zero retrain, zero
GPU, and no upstream patch. Default off; locked into the baseline only if an experiment sweep confirms a
world-HOTA gain with no AssA regression.

## Scope

**In scope (v1):** sequential, **strictly non-overlapping** fragment pairs joined by a **tight gate**
(close in time AND world distance). The strict non-overlap constraint structurally excludes concurrent
splits, keeping v1 conservative and low over-merge risk.

**Out of scope (deferred):**
- **Concurrent cross-camera splits** (GT14/GT17/GT20 — same person, two cameras, overlapping in time):
  these need a co-visibility + world-distance must-link, which is **lever #2** (the `mcpt.py:575`
  `max_distance_th` patch), a separate workstream.
- **Long-gap sequential pairs** (e.g. GT6 ~2.95 m/88 f, GT10 ~4.32 m/149 f): a looser, riskier gate;
  revisit only if v1 lands and the sweep shows the tight gate plateaus below the ceiling.
- **Velocity extrapolation** and **overlap tolerance** at the join: noted as future extensions.

## Architecture

One new pure module `aic24_nvidia/world_stitch.py`, invoked from `_eval_mct_world`
(`aic24_nvidia/stages/evaluate.py`) **between** `aggregate_world_tracks` and `smooth_world_tracks`.
It relabels over-split global IDs so a merged identity is a single track; smoothing then runs *across*
the join (smooth transition). Because it lives entirely in the evaluate stage and consumes only the
already-harvested world rows, it is `rerun_from: evaluate` (seconds per variant, no SCT/MCT re-run). This
mirrors exactly how `world_smoothing` is already wired.

### Data flow

```
aggregate_world_tracks(mct_json) → rows[(frame, gid, x, y)]
   → stitch_world_tracks(rows, method, max_gap_frames, max_dist_m) → (rows', merges)   ← NEW
   → smooth_world_tracks(rows', method, ema_alpha)
   → write_world_pred → run_world_eval
```

`world_stitch` precedes `world_smoothing` so EMA smooths over the merged join rather than locking in a
discontinuity.

## Algorithm (sequential, tight gate)

1. **Summarize** each gid: sort its rows by frame → `first_frame`, `last_frame`, `first_xy`, `last_xy`,
   `n` (frame count).
2. **Candidate edges** `A → B` accepted iff:
   - `A.last_frame < B.first_frame` (strict non-overlap — excludes concurrent splits by construction);
   - `gap = B.first_frame − A.last_frame ≤ max_gap_frames`;
   - `dist(A.last_xy, B.first_xy) ≤ max_dist_m` (Euclidean, world metres).
3. **Resolve** to a consistent relabel: sort candidate edges by `(dist, gap)` ascending; greedily accept
   an edge only if A's **end-slot** and B's **start-slot** are both unconsumed (1-in/1-out matching —
   prevents fan-in/fan-out, naturally forms chains `A→B→C`). Union accepted edges → connected components
   → canonical id = **min gid** in each component.
4. **Apply**: remap each row's gid to its canonical id; defensively re-average any duplicate
   `(frame, canonical_gid)` points (a no-op under strict non-overlap, robust against incidental dupes).
   Return rows sorted by `(frame, gid)` plus the merge list.

Fully **deterministic** — every ordering is total (`dist`, then `gap`, then `gid`); no randomness. Full-clip
ids (`[1..900]`) are inert (nothing ends before them or starts after them), so only genuine fragments chain.

## Module API — `aic24_nvidia/world_stitch.py`

```python
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class TrackSummary:
    gid: int
    first_frame: int
    last_frame: int
    first_xy: tuple[float, float]
    last_xy: tuple[float, float]
    n: int

def summarize_tracks(rows: list[tuple[int, int, float, float]]) -> dict[int, TrackSummary]:
    """Per-gid endpoints/counts from (frame, gid, x, y) rows."""

def find_stitch_edges(summaries: dict[int, TrackSummary], *, max_gap_frames: int, max_dist_m: float
                      ) -> list[tuple[float, int, int, int]]:
    """Candidate (dist, gap, gid_a, gid_b) sequential edges passing the tight gate."""

def resolve_merges(edges: list[tuple[float, int, int, int]]) -> dict[int, int]:
    """Greedy 1-in/1-out matching + union-find → {gid: canonical_gid} (canonical = min gid)."""

def stitch_world_tracks(rows: list[tuple[int, int, float, float]], *, method: str,
                        max_gap_frames: int, max_dist_m: float
                        ) -> tuple[list[tuple[int, int, float, float]], list[tuple[int, int]]]:
    """method='none' → (rows, []) identity; 'endpoint_gap' → (merged_rows, merges).
       merges = [(canonical_gid, absorbed_gid), ...] for logging/manifest."""
```

`stitch_world_tracks` mirrors `smooth_world_tracks`'s keyword call style; returning `(rows, merges)`
parallels `aggregate_world_tracks`'s `(rows, dropped)`.

## Config

New block in `configs/baseline.yaml` (default **off** — baseline unchanged):

```yaml
world_stitch:
  method: none          # none | endpoint_gap
  max_gap_frames: 45
  max_dist_m: 0.6
```

New `WorldStitchConfig` dataclass in `aic24_nvidia/config.py` with validation: `method ∈ {none, endpoint_gap}`,
`max_gap_frames > 0`, `max_dist_m > 0`; `load_config` rejects an unknown `method` (same pattern as
`world_smoothing`).

## Wiring — `aic24_nvidia/stages/evaluate.py`

In `_eval_mct_world`, one import + one insertion between aggregate (line ~237) and smooth (line ~240):

```python
rows, dropped = aggregate_world_tracks(Path(mct_global))
if not rows:
    return {"skipped": "MCT produced no valid world points"}
rows, merges = stitch_world_tracks(
    rows,
    method=cfg.world_stitch.method,
    max_gap_frames=cfg.world_stitch.max_gap_frames,
    max_dist_m=cfg.world_stitch.max_dist_m,
)
rows = smooth_world_tracks(rows, method=cfg.world_smoothing.method, ema_alpha=cfg.world_smoothing.ema_alpha)
```

- `log.info("world_stitch: merged %d fragment pairs: %s", len(merges), merges)`.
- Add a `world_stitch` block (`method`, `max_gap_frames`, `max_dist_m`, `n_merges`) to `ctx.set_params`.
- `experiments/registry.yaml` convention comment gains a row: `world_stitch.* -> evaluate`.

## Experiment — `world_stitch_sweep`

Append to `experiments/registry.yaml`. `base_config: configs/baseline.yaml`, `rerun_from: evaluate`
(seconds per variant — only evaluate re-runs).

| variant | `max_gap_frames` | `max_dist_m` |
|---|---|---|
| `none` (control) | — | — (method: none) |
| `g30_d0.5` | 30 | 0.5 |
| `g45_d0.6` | 45 | 0.6 |
| `g60_d0.75` | 60 | 0.75 |
| `g90_d1.0` | 90 | 1.0 |

### Decision rule

Run `python experiments/run.py run world_stitch_sweep`, then `python experiments/compare.py --sort-by
mct_world.HOTA`. Ship the variant with the **highest world HOTA** that also has **AssA ≥ baseline 0.6423**
and **DetA not down**. If a variant ships: set `configs/baseline.yaml` `world_stitch` to it, bump the
version comment, rebuild `outputs/baseline/`, and update the `pipeline-state` memory. The single-merge gain
was +0.0088; "both clean pairs" ≈ +0.0193 — the tight gates should capture the clean pairs without
over-merging. If no variant clears the guard, keep `none` and document the null result.

## Testing

**Unit** `tests/unit/test_world_stitch.py` (no GPU):
- `summarize_tracks` — endpoints/count correct on a multi-frame fixture.
- `find_stitch_edges` — accepts a close non-overlapping pair; rejects (a) temporally-overlapping pairs,
  (b) `dist > max_dist_m`, (c) `gap > max_gap_frames`.
- `resolve_merges` — chain `A→B→C` unions to one canonical (min gid); fan-in conflict (two predecessors,
  one start) keeps only the closer edge; empty edges → empty map.
- `stitch_world_tracks` — `none` is identity returning `[]` merges; `endpoint_gap` merges a synthetic
  `3→8`-style pair into one gid spanning both ranges and re-aggregates; merge list reported.
- Determinism — identical input → identical output.
- `config.py` — unknown `method` rejected; defaults are `none` / `45` / `0.6`.
- `tests/unit/test_world_stitch_sweep_registry.py` — block present, `rerun_from == evaluate`, has a `none`
  control + `endpoint_gap` variants.

**Edge cases:** empty rows → `([], [])`; single-frame gid (`first==last`) valid as either slot; full-clip
ids inert; strict non-overlap ⇒ no same-frame dupes (re-aggregation purely defensive).

**Execution (not unit-tested):** run `world_stitch_sweep` on `outputs/baseline/`, compare, apply the
decision rule; if a variant ships, lock baseline + update memory.

## Non-goals / YAGNI

- No appearance signal in the merge decision (post-aggregation rows carry none; pure geometry produced
  the measured gain).
- No automatic GT-dependent guard at inference (impossible without GT); the guard is the experiment +
  decision rule, the repo's standard discipline.
- No concurrent-split, long-gap, velocity, or overlap-tolerance handling in v1 (deferred).

## Self-review

- **Placeholders:** none — every section is concrete; gate values are explicit; the ship value is
  determined by the sweep (intentionally, not a placeholder).
- **Consistency:** the evaluate-stage injection (`rerun_from: evaluate`), the `(rows, merges)` return, and
  the config/experiment names are used identically across Architecture, API, Wiring, Config, Experiment,
  and Testing.
- **Scope:** single focused module + one wiring point + one config block + one experiment — fits one
  implementation plan.
- **Ambiguity:** "non-overlap" fixed as strict (`A.last_frame < B.first_frame`); canonical id fixed as min
  gid; edge ordering fixed as `(dist, gap, gid)`.
