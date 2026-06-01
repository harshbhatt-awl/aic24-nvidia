# Post-MCT Over-Split Stitch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a config-gated, evaluate-stage transform that merges sequential non-overlapping world-track fragments by world-endpoint proximity, recovering the measured +0.0088 (clean-subset ceiling ~+0.019) world HOTA from the 32-pred-vs-25-GT over-segmentation.

**Architecture:** One new pure module `aic24_nvidia/world_stitch.py` (summarize → find edges → resolve merges → relabel), invoked from `_eval_mct_world` in `aic24_nvidia/stages/evaluate.py` **between** `aggregate_world_tracks` and `smooth_world_tracks`. A `WorldStitchCfg` in `config.py` gates it (default `none`); a `world_stitch_sweep` experiment (`rerun_from: evaluate`) tunes the gate under an AssA-non-regression decision rule. No upstream patch, no GPU, no retrain.

**Tech Stack:** Python 3.14, stdlib only (`math`, `dataclasses`, `collections`), `pytest`, `pyyaml` (already a dep).

**Spec:** `docs/superpowers/specs/2026-06-01-over-split-stitch-design.md`

---

## File structure

- **Create** `aic24_nvidia/world_stitch.py` — pure functions: `TrackSummary`, `summarize_tracks`, `find_stitch_edges`, `resolve_merges`, `stitch_world_tracks`. No I/O, no pipeline imports.
- **Create** `tests/unit/test_world_stitch.py` — unit tests for the pure functions.
- **Modify** `aic24_nvidia/config.py` — add `WorldStitchCfg`, a `world_stitch` field on `Config`, and `load_config` parsing/validation.
- **Modify** `tests/unit/test_config.py` — defaults/explicit/invalid tests for `world_stitch`.
- **Modify** `aic24_nvidia/stages/evaluate.py` — import + call `stitch_world_tracks` between aggregate and smooth; log merges; record in metrics + `set_params`.
- **Create** `tests/unit/test_evaluate_world_stitch_wiring.py` — assert `_eval_mct_world` calls `stitch_world_tracks` with the cfg knobs.
- **Modify** `configs/baseline.yaml` — add `world_stitch` block (method `none`, gate defaults) — documentation + `set_params`; no behavior change.
- **Modify** `experiments/registry.yaml` — append `world_stitch_sweep`; add `world_stitch.* -> evaluate` to the convention comment.
- **Create** `tests/unit/test_world_stitch_sweep_registry.py` — registry parse test.

Naming note: the codebase config classes use the `Cfg` suffix (`WorldProjectionCfg`, `WorldSmoothingCfg`) — use **`WorldStitchCfg`** (not `...Config`).

---

### Task 1: Scaffold `world_stitch.py` — `TrackSummary` + `summarize_tracks`

**Files:**
- Create: `aic24_nvidia/world_stitch.py`
- Test: `tests/unit/test_world_stitch.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_world_stitch.py
"""Unit tests for aic24_nvidia.world_stitch."""
from __future__ import annotations

import pytest


def test_summarize_tracks_endpoints_and_count():
    from aic24_nvidia.world_stitch import summarize_tracks

    rows = [
        (3, 7, 0.0, 0.0),
        (1, 7, 10.0, 20.0),
        (2, 7, 11.0, 21.0),
        (5, 9, 100.0, 100.0),
    ]
    s = summarize_tracks(rows)
    assert set(s) == {7, 9}
    assert s[7].first_frame == 1 and s[7].last_frame == 3
    assert s[7].first_xy == (10.0, 20.0)
    assert s[7].last_xy == (0.0, 0.0)
    assert s[7].n == 3
    assert s[9].first_frame == s[9].last_frame == 5
    assert s[9].n == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_world_stitch.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aic24_nvidia.world_stitch'`

- [ ] **Step 3: Write minimal implementation**

```python
# aic24_nvidia/world_stitch.py
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

Row = tuple[int, int, float, float]  # (frame, gid, x, y)


@dataclass(frozen=True)
class TrackSummary:
    gid: int
    first_frame: int
    last_frame: int
    first_xy: tuple[float, float]
    last_xy: tuple[float, float]
    n: int


def summarize_tracks(rows: list[Row]) -> dict[int, TrackSummary]:
    """Per-gid endpoints/counts from (frame, gid, x, y) rows."""
    by_gid: dict[int, list[tuple[int, float, float]]] = {}
    for f, g, x, y in rows:
        by_gid.setdefault(g, []).append((f, x, y))
    out: dict[int, TrackSummary] = {}
    for g, seq in by_gid.items():
        seq.sort(key=lambda t: t[0])
        f0, x0, y0 = seq[0]
        f1, x1, y1 = seq[-1]
        out[g] = TrackSummary(
            gid=g, first_frame=f0, last_frame=f1,
            first_xy=(x0, y0), last_xy=(x1, y1), n=len(seq),
        )
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_world_stitch.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add aic24_nvidia/world_stitch.py tests/unit/test_world_stitch.py
git commit -m "feat(world_stitch): TrackSummary + summarize_tracks"
```

---

### Task 2: `find_stitch_edges` — tight-gate candidate pairs

**Files:**
- Modify: `aic24_nvidia/world_stitch.py`
- Test: `tests/unit/test_world_stitch.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/test_world_stitch.py
from aic24_nvidia.world_stitch import find_stitch_edges, summarize_tracks


def _summ(rows):
    return summarize_tracks(rows)


def test_find_edges_accepts_close_sequential_pair():
    # gid 3 ends frame 3 at (0,0); gid 8 starts frame 5 at (0.3,0): gap 2, dist 0.3
    rows = [(1, 3, 0.0, 0.0), (3, 3, 0.0, 0.0), (5, 8, 0.3, 0.0), (7, 8, 0.5, 0.0)]
    edges = find_stitch_edges(_summ(rows), max_gap_frames=45, max_dist_m=0.6)
    assert len(edges) == 1
    dist, gap, a, b = edges[0]
    assert (a, b) == (3, 8)
    assert gap == 2
    assert dist == pytest.approx(0.3)


def test_find_edges_rejects_temporal_overlap():
    # gid 3 spans 1..6, gid 8 spans 5..9 -> overlap, no edge either direction
    rows = [(1, 3, 0.0, 0.0), (6, 3, 0.0, 0.0), (5, 8, 0.1, 0.0), (9, 8, 0.1, 0.0)]
    assert find_stitch_edges(_summ(rows), max_gap_frames=45, max_dist_m=0.6) == []


def test_find_edges_rejects_too_far_in_space():
    rows = [(1, 3, 0.0, 0.0), (3, 3, 0.0, 0.0), (5, 8, 5.0, 0.0), (7, 8, 5.0, 0.0)]
    assert find_stitch_edges(_summ(rows), max_gap_frames=45, max_dist_m=0.6) == []


def test_find_edges_rejects_too_far_in_time():
    rows = [(1, 3, 0.0, 0.0), (3, 3, 0.0, 0.0), (100, 8, 0.1, 0.0), (110, 8, 0.1, 0.0)]
    assert find_stitch_edges(_summ(rows), max_gap_frames=45, max_dist_m=0.6) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_world_stitch.py -v`
Expected: FAIL — `ImportError: cannot import name 'find_stitch_edges'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to aic24_nvidia/world_stitch.py
def find_stitch_edges(
    summaries: dict[int, TrackSummary], *, max_gap_frames: int, max_dist_m: float
) -> list[tuple[float, int, int, int]]:
    """Candidate (dist, gap, gid_a, gid_b) sequential edges passing the tight gate.

    An edge A->B requires strict non-overlap (A.last_frame < B.first_frame), a
    frame gap <= max_gap_frames, and a world endpoint distance <= max_dist_m.
    Sorted by (dist, gap, gid_a, gid_b) ascending.
    """
    edges: list[tuple[float, int, int, int]] = []
    items = list(summaries.values())
    for a in items:
        for b in items:
            if a.gid == b.gid:
                continue
            if not (a.last_frame < b.first_frame):  # strict non-overlap, A before B
                continue
            gap = b.first_frame - a.last_frame
            if gap > max_gap_frames:
                continue
            dist = math.dist(a.last_xy, b.first_xy)
            if dist > max_dist_m:
                continue
            edges.append((dist, gap, a.gid, b.gid))
    edges.sort()
    return edges
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_world_stitch.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add aic24_nvidia/world_stitch.py tests/unit/test_world_stitch.py
git commit -m "feat(world_stitch): find_stitch_edges tight-gate candidate pairs"
```

---

### Task 3: `resolve_merges` — greedy 1-in/1-out matching + union-find

**Files:**
- Modify: `aic24_nvidia/world_stitch.py`
- Test: `tests/unit/test_world_stitch.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/test_world_stitch.py
from aic24_nvidia.world_stitch import resolve_merges


def test_resolve_chain_unions_to_min_gid():
    # edges as (dist, gap, a, b): 3->8 and 8->12 chain into one component, canonical 3
    edges = [(0.1, 2, 3, 8), (0.2, 2, 8, 12)]
    labels = resolve_merges(edges)
    assert labels[3] == 3
    assert labels[8] == 3
    assert labels[12] == 3


def test_resolve_fanin_keeps_closer_edge_only():
    # two predecessors (3, 5) both want successor-start 8; the closer (3) wins,
    # the other start-slot is consumed so 5->8 is skipped.
    edges = [(0.1, 2, 3, 8), (0.4, 2, 5, 8)]
    labels = resolve_merges(edges)
    assert labels[3] == 3 and labels[8] == 3
    assert 5 not in labels  # 5 never merged


def test_resolve_empty():
    assert resolve_merges([]) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_world_stitch.py -v`
Expected: FAIL — `ImportError: cannot import name 'resolve_merges'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to aic24_nvidia/world_stitch.py
def resolve_merges(edges: list[tuple[float, int, int, int]]) -> dict[int, int]:
    """Greedy 1-in/1-out matching + union-find -> {gid: canonical_gid}.

    Edges must be pre-sorted (best first). Each gid is consumed at most once as a
    predecessor end and once as a successor start, so a track continues into at
    most one other and is continued by at most one — preventing fan-in/fan-out
    while still forming chains A->B->C. Canonical id is the minimum gid in a
    component. Returns only gids touched by an accepted merge.
    """
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        hi, lo = (ra, rb) if ra > rb else (rb, ra)
        parent[hi] = lo  # attach larger root under smaller -> canonical = min

    used_end: set[int] = set()
    used_start: set[int] = set()
    for _dist, _gap, a, b in edges:
        if a in used_end or b in used_start:
            continue
        used_end.add(a)
        used_start.add(b)
        union(a, b)

    return {g: find(g) for g in parent}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_world_stitch.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add aic24_nvidia/world_stitch.py tests/unit/test_world_stitch.py
git commit -m "feat(world_stitch): resolve_merges greedy matching + union-find"
```

---

### Task 4: `stitch_world_tracks` — top-level transform (relabel + re-aggregate)

**Files:**
- Modify: `aic24_nvidia/world_stitch.py`
- Test: `tests/unit/test_world_stitch.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/test_world_stitch.py
from aic24_nvidia.world_stitch import stitch_world_tracks


def test_stitch_none_is_identity():
    rows = [(1, 3, 0.0, 0.0), (2, 3, 1.0, 1.0)]
    out, merges = stitch_world_tracks(rows, method="none", max_gap_frames=45, max_dist_m=0.6)
    assert out == rows
    assert merges == []


def test_stitch_endpoint_gap_merges_sequential_pair():
    # gid 3 frames 1..3, gid 8 frames 5..7, close -> merge into gid 3 spanning all
    rows = [
        (1, 3, 0.0, 0.0), (2, 3, 0.0, 0.0), (3, 3, 0.0, 0.0),
        (5, 8, 0.3, 0.0), (6, 8, 0.4, 0.0), (7, 8, 0.5, 0.0),
    ]
    out, merges = stitch_world_tracks(rows, method="endpoint_gap", max_gap_frames=45, max_dist_m=0.6)
    assert merges == [(3, 8)]
    assert {g for (_f, g, _x, _y) in out} == {3}
    assert sorted(f for (f, _g, _x, _y) in out) == [1, 2, 3, 5, 6, 7]


def test_stitch_no_candidates_returns_input_and_empty_merges():
    rows = [(1, 3, 0.0, 0.0), (5, 8, 9.0, 9.0)]  # far apart in space
    out, merges = stitch_world_tracks(rows, method="endpoint_gap", max_gap_frames=45, max_dist_m=0.6)
    assert merges == []
    assert {g for (_f, g, _x, _y) in out} == {3, 8}


def test_stitch_is_deterministic():
    rows = [
        (1, 3, 0.0, 0.0), (3, 3, 0.0, 0.0),
        (5, 8, 0.3, 0.0), (7, 8, 0.5, 0.0),
    ]
    a = stitch_world_tracks(rows, method="endpoint_gap", max_gap_frames=45, max_dist_m=0.6)
    b = stitch_world_tracks(rows, method="endpoint_gap", max_gap_frames=45, max_dist_m=0.6)
    assert a == b


def test_stitch_invalid_method():
    with pytest.raises(ValueError, match="method"):
        stitch_world_tracks([(1, 1, 0.0, 0.0)], method="kalman", max_gap_frames=45, max_dist_m=0.6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_world_stitch.py -v`
Expected: FAIL — `ImportError: cannot import name 'stitch_world_tracks'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to aic24_nvidia/world_stitch.py
def stitch_world_tracks(
    rows: list[Row], *, method: str, max_gap_frames: int, max_dist_m: float
) -> tuple[list[Row], list[tuple[int, int]]]:
    """Merge over-split world tracks. method='none' is identity.

    For 'endpoint_gap': summarize -> find tight-gate sequential edges -> resolve
    to a {gid: canonical} relabel -> remap rows and re-average any duplicate
    (frame, canonical_gid) points (defensive; a no-op under strict non-overlap).
    Returns (rows sorted by (frame, gid), merges) where merges = [(canonical,
    absorbed), ...].
    """
    if method == "none":
        return list(rows), []
    if method != "endpoint_gap":
        raise ValueError(f"stitch_world_tracks: unknown method {method!r}")

    summaries = summarize_tracks(rows)
    edges = find_stitch_edges(summaries, max_gap_frames=max_gap_frames, max_dist_m=max_dist_m)
    labels = resolve_merges(edges)
    if not labels:
        return list(rows), []

    acc: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)
    for f, g, x, y in rows:
        cg = labels.get(g, g)
        acc[(f, cg)].append((x, y))
    out: list[Row] = []
    for (f, g), pts in acc.items():
        mx = sum(p[0] for p in pts) / len(pts)
        my = sum(p[1] for p in pts) / len(pts)
        out.append((f, g, mx, my))
    out.sort()

    merges = sorted((labels[g], g) for g in labels if labels[g] != g)
    return out, merges
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_world_stitch.py -v`
Expected: PASS (13 passed)

- [ ] **Step 5: Commit**

```bash
git add aic24_nvidia/world_stitch.py tests/unit/test_world_stitch.py
git commit -m "feat(world_stitch): stitch_world_tracks top-level relabel + re-aggregate"
```

---

### Task 5: `WorldStitchCfg` + `load_config` parsing/validation

**Files:**
- Modify: `aic24_nvidia/config.py`
- Test: `tests/unit/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/test_config.py  (reuses _minimal and _write already in this file)
def test_world_stitch_defaults(tmp_path):
    body = _minimal(tmp_path)
    cfg = load_config(_write(tmp_path, body))
    assert cfg.world_stitch.method == "none"
    assert cfg.world_stitch.max_gap_frames == 45
    assert cfg.world_stitch.max_dist_m == pytest.approx(0.6)


def test_world_stitch_explicit(tmp_path):
    body = _minimal(tmp_path)
    body["world_stitch"] = {"method": "endpoint_gap", "max_gap_frames": 30, "max_dist_m": 0.5}
    cfg = load_config(_write(tmp_path, body))
    assert cfg.world_stitch.method == "endpoint_gap"
    assert cfg.world_stitch.max_gap_frames == 30
    assert cfg.world_stitch.max_dist_m == pytest.approx(0.5)


def test_world_stitch_invalid_method(tmp_path):
    body = _minimal(tmp_path)
    body["world_stitch"] = {"method": "magic"}
    with pytest.raises(ConfigError, match="world_stitch.method"):
        load_config(_write(tmp_path, body))


def test_world_stitch_gap_must_be_positive(tmp_path):
    body = _minimal(tmp_path)
    body["world_stitch"] = {"method": "endpoint_gap", "max_gap_frames": 0}
    with pytest.raises(ConfigError, match="max_gap_frames"):
        load_config(_write(tmp_path, body))


def test_world_stitch_dist_must_be_positive(tmp_path):
    body = _minimal(tmp_path)
    body["world_stitch"] = {"method": "endpoint_gap", "max_dist_m": 0.0}
    with pytest.raises(ConfigError, match="max_dist_m"):
        load_config(_write(tmp_path, body))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_config.py -k world_stitch -v`
Expected: FAIL — `AttributeError: ... object has no attribute 'world_stitch'`

- [ ] **Step 3: Write minimal implementation**

In `aic24_nvidia/config.py`, add the dataclass after `WorldSmoothingCfg` (after line 65):

```python
@dataclass(frozen=True)
class WorldStitchCfg:
    method: str = "none"              # none | endpoint_gap
    max_gap_frames: int = 45
    max_dist_m: float = 0.6
```

Add the field to `Config` immediately after `world_smoothing: WorldSmoothingCfg` (line 83):

```python
    world_stitch: WorldStitchCfg
```

In `load_config`, after the `world_smoothing = WorldSmoothingCfg(...)` line (line 161), add:

```python
    st_body = body.get("world_stitch") or {}
    st_method = st_body.get("method", "none")
    if st_method not in {"none", "endpoint_gap"}:
        raise ConfigError(f"world_stitch.method must be one of none|endpoint_gap, got {st_method!r}")
    st_gap = int(st_body.get("max_gap_frames", 45))
    if st_gap <= 0:
        raise ConfigError(f"world_stitch.max_gap_frames must be > 0, got {st_gap}")
    st_dist = float(st_body.get("max_dist_m", 0.6))
    if st_dist <= 0:
        raise ConfigError(f"world_stitch.max_dist_m must be > 0, got {st_dist}")
    world_stitch = WorldStitchCfg(method=st_method, max_gap_frames=st_gap, max_dist_m=st_dist)
```

Add to the `Config(...)` return (after `world_smoothing=world_smoothing,`, line 199):

```python
        world_stitch=world_stitch,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_config.py -v`
Expected: PASS (all, including the 5 new `world_stitch` tests)

- [ ] **Step 5: Guard against direct `Config(...)` construction (no-default field added)**

Run: `grep -rn "Config(" aic24_nvidia tests | grep -v "load_config\|WorldStitchCfg\|MagicMock\|class Config\|def "`
Expected: only call sites that go through `load_config`. `Config` is constructed solely in `load_config`; if the grep surfaces any direct `Config(...)` call, add `world_stitch=WorldStitchCfg()` to it. (Same pattern used when `world_smoothing` was added; none expected.)

- [ ] **Step 6: Commit**

```bash
git add aic24_nvidia/config.py tests/unit/test_config.py
git commit -m "feat(config): WorldStitchCfg + world_stitch parsing/validation"
```

---

### Task 6: Wire `stitch_world_tracks` into `_eval_mct_world`

**Files:**
- Modify: `aic24_nvidia/stages/evaluate.py`
- Test: `tests/unit/test_evaluate_world_stitch_wiring.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_evaluate_world_stitch_wiring.py
"""Verify _eval_mct_world calls stitch_world_tracks with the right knobs, before smoothing."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _stub_cfg(method="endpoint_gap", max_gap_frames=30, max_dist_m=0.5):
    from aic24_nvidia.config import WorldSmoothingCfg, WorldStitchCfg, EvalCfg
    cfg = MagicMock()
    cfg.eval = EvalCfg(world_d_max=1.0)
    cfg.world_smoothing = WorldSmoothingCfg(method="none", ema_alpha=0.3)
    cfg.world_stitch = WorldStitchCfg(method=method, max_gap_frames=max_gap_frames, max_dist_m=max_dist_m)
    return cfg


def _write_minimal_mct(tmp_path: Path) -> Path:
    mct = tmp_path / "fixed_whole_tracking_results.json"
    mct.write_text(json.dumps({
        "390": {
            "00000001": {
                "Frame": 5,
                "GlobalOfflineID": 1,
                "Coordinate": {"x1": 50, "y1": 100, "x2": 150, "y2": 850},
                "WorldCoordinate": {"x": 1.0, "y": 2.0},
            }
        }
    }))
    return mct


def test_eval_mct_world_calls_stitch(tmp_path, monkeypatch):
    from aic24_nvidia.stages import evaluate as evaluate_stage

    captured = {}

    def fake_stitch(rows, method, max_gap_frames, max_dist_m):
        captured["method"] = method
        captured["max_gap_frames"] = max_gap_frames
        captured["max_dist_m"] = max_dist_m
        return rows, []

    monkeypatch.setattr(evaluate_stage, "stitch_world_tracks", fake_stitch)

    cfg = _stub_cfg(method="endpoint_gap", max_gap_frames=30, max_dist_m=0.5)
    ctx = MagicMock()
    ctx.work_dir = tmp_path

    mct_global = _write_minimal_mct(tmp_path)
    adapted_root = tmp_path / "adapted"
    adapted_root.mkdir()
    (adapted_root / "scene_001_gt_world.txt").write_text("5,1,1.0,2.0\n")

    try:
        evaluate_stage._eval_mct_world(cfg, ctx, str(mct_global), adapted_root)
    except Exception:
        pass

    assert captured.get("method") == "endpoint_gap"
    assert captured.get("max_gap_frames") == 30
    assert captured.get("max_dist_m") == pytest.approx(0.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_evaluate_world_stitch_wiring.py -v`
Expected: FAIL — `AttributeError: <module 'aic24_nvidia.stages.evaluate'> does not have the attribute 'stitch_world_tracks'`

- [ ] **Step 3: Write minimal implementation**

In `aic24_nvidia/stages/evaluate.py`, extend the world_tracks import (line 16) and add the world_stitch import (line 17):

```python
from ..world_tracks import aggregate_world_tracks, smooth_world_tracks, write_world_pred
from ..world_stitch import stitch_world_tracks
```

In `_eval_mct_world`, replace the block from `rows, dropped = aggregate_world_tracks(...)` through the `rows = smooth_world_tracks(...)` call (lines 237–244) with:

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
        log.info("world_stitch: merged %d fragment pairs: %s", len(merges), merges)
        rows = smooth_world_tracks(
            rows,
            method=cfg.world_smoothing.method,
            ema_alpha=cfg.world_smoothing.ema_alpha,
        )
```

Then, where the world metrics dict `m` is finalized (after `m["frames_evaluated"] = ...`, line 260), add:

```python
        m["world_stitch_merges"] = len(merges)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_evaluate_world_stitch_wiring.py tests/unit/test_evaluate_world_smoother_wiring.py -v`
Expected: PASS (both the new stitch wiring test and the existing smoother wiring test)

- [ ] **Step 5: Record the cfg in `set_params`**

In `run(...)`, in the `ctx.set_params({...})` call, add a `world_stitch` block alongside `world_smoothing` (after the `world_smoothing` dict, ~line 339):

```python
            "world_stitch": {
                "method": cfg.world_stitch.method,
                "max_gap_frames": cfg.world_stitch.max_gap_frames,
                "max_dist_m": cfg.world_stitch.max_dist_m,
            },
```

- [ ] **Step 6: Commit**

```bash
git add aic24_nvidia/stages/evaluate.py tests/unit/test_evaluate_world_stitch_wiring.py
git commit -m "feat(evaluate): wire over-split stitch between aggregate and smooth"
```

---

### Task 7: `world_stitch_sweep` experiment + registry test + baseline block

**Files:**
- Modify: `experiments/registry.yaml`
- Modify: `configs/baseline.yaml`
- Test: `tests/unit/test_world_stitch_sweep_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_world_stitch_sweep_registry.py
from pathlib import Path
import yaml

REGISTRY = Path(__file__).resolve().parents[2] / "experiments" / "registry.yaml"


def _experiments():
    data = yaml.safe_load(REGISTRY.read_text())
    return {e["id"]: e for e in data["experiments"]}


def test_world_stitch_sweep_present_and_reruns_from_evaluate():
    exp = _experiments()["world_stitch_sweep"]
    assert exp["base_config"] == "configs/baseline.yaml"
    # world_stitch.* is an eval-only transform -> rerun_from evaluate
    assert exp["rerun_from"] == "evaluate"


def test_world_stitch_sweep_has_none_control_and_endpoint_gap_variants():
    exp = _experiments()["world_stitch_sweep"]
    names = {v["name"] for v in exp["variants"]}
    assert "none" in names
    methods, gaps, dists = set(), set(), set()
    for v in exp["variants"]:
        st = v["overrides"].get("world_stitch", {})
        if "method" in st:
            methods.add(st["method"])
        if "max_gap_frames" in st:
            gaps.add(st["max_gap_frames"])
        if "max_dist_m" in st:
            dists.add(st["max_dist_m"])
    assert "endpoint_gap" in methods
    assert {30, 45, 60, 90} <= gaps
    assert {0.5, 0.6, 0.75, 1.0} <= dists
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_world_stitch_sweep_registry.py -v`
Expected: FAIL — `KeyError: 'world_stitch_sweep'`

- [ ] **Step 3a: Append the experiment block to `experiments/registry.yaml`**

```yaml
  - id: world_stitch_sweep
    description: "Sweep the post-MCT over-split stitch gate (sequential non-overlapping fragments). Targets the 32-pred-vs-25-GT over-segmentation; recovers the measured +0.0088 (clean-subset ceiling ~+0.019) world HOTA."
    hypothesis: "Merging sequential fragments close in world+time raises world AssA/IDF1 (DetA flat); net world HOTA rises if the merges are correct (no over-merge)."
    base_config: configs/baseline.yaml
    rerun_from: evaluate     # world_stitch.* -> evaluate (eval-only transform)
    variants:
      - name: "none"          # control: stitch off (= baseline)
        overrides:
          world_stitch:
            method: none
      - name: "g30_d0.5"
        overrides:
          world_stitch:
            method: endpoint_gap
            max_gap_frames: 30
            max_dist_m: 0.5
      - name: "g45_d0.6"
        overrides:
          world_stitch:
            method: endpoint_gap
            max_gap_frames: 45
            max_dist_m: 0.6
      - name: "g60_d0.75"
        overrides:
          world_stitch:
            method: endpoint_gap
            max_gap_frames: 60
            max_dist_m: 0.75
      - name: "g90_d1.0"
        overrides:
          world_stitch:
            method: endpoint_gap
            max_gap_frames: 90
            max_dist_m: 1.0
```

- [ ] **Step 3b: Add the convention-comment row in `experiments/registry.yaml`**

In the `rerun_from` mapping table comment near the top, add a row beneath `world_smoothing.*`:

```
#   world_stitch.*                  -> evaluate
```

- [ ] **Step 3c: Add the `world_stitch` block to `configs/baseline.yaml`**

Immediately after the `world_smoothing:` block, add (default off → baseline output unchanged):

```yaml
world_stitch:
  method: none               # none | endpoint_gap  (off; tuned via world_stitch_sweep)
  max_gap_frames: 45
  max_dist_m: 0.6
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_world_stitch_sweep_registry.py tests/unit/test_config.py -v`
Expected: PASS. Also confirm `configs/baseline.yaml` still loads:
Run: `python -c "from aic24_nvidia.config import load_config; c=load_config('configs/baseline.yaml'); print(c.world_stitch)"`
Expected: `WorldStitchCfg(method='none', max_gap_frames=45, max_dist_m=0.6)`

- [ ] **Step 5: Commit**

```bash
git add experiments/registry.yaml configs/baseline.yaml tests/unit/test_world_stitch_sweep_registry.py
git commit -m "feat(experiments): world_stitch_sweep + baseline world_stitch block"
```

---

### Task 8: Full-suite regression + execute the sweep (execution, not unit-tested)

**Files:** none (produces `outputs/world_stitch_sweep__*/` + analysis).

- [ ] **Step 1: Run the full unit suite (no regressions)**

Run: `pytest tests/unit/ -q`
Expected: all pass — the new `world_stitch` field has a value in every `load_config` path, the registry-consistency test (`test_pipeline_registry_consistency.py`) is unaffected (no stage dicts changed). If a pre-existing unrelated failure appears (e.g. the known integration `discover_cameras` case mismatch), confirm it predates this branch.

- [ ] **Step 2: Ensure the baseline is built**

Run: `python experiments/run.py ensure-baseline`
Expected: `outputs/baseline/evaluate/metrics.json` present with `mct_world.HOTA` ≈ 0.6479. (Adding the `none` stitch block does not change baseline output.)

- [ ] **Step 3: Run the sweep**

Run: `python experiments/run.py run world_stitch_sweep`
Expected: one `outputs/world_stitch_sweep__<name>/evaluate/metrics.json` per variant (seconds each — only evaluate re-runs via `rerun_from: evaluate`). Each variant's `metrics.json` `mct_world` block carries `world_stitch_merges` (0 for `none`, ≥1 for the `endpoint_gap` variants).

- [ ] **Step 4: Read the metrics and apply the decision rule**

Run: `python experiments/compare.py --sort-by mct_world.HOTA`
Expected: a table of each variant's `mct_world.HOTA / DetA / AssA` vs baseline. Apply the spec's decision rule:
- Ship the variant with the highest world HOTA that also has **AssA ≥ baseline 0.6423** and **DetA not down**.
- If the best `endpoint_gap` variant ties or regresses vs `none`, keep `none` and record the null result in `pipeline-state` memory.
- Sanity-check `world_stitch_merges` is small (single digits) for the shipping variant — a large count signals over-merge; inspect the logged pairs.

- [ ] **Step 5 (only if a variant ships): lock the new baseline**

Edit `configs/baseline.yaml` `world_stitch` to the winning `method`/`max_gap_frames`/`max_dist_m`, bump the version comment with the new world HOTA, then:

```bash
rm -rf outputs/baseline/evaluate
python pipeline.py evaluate --config configs/baseline.yaml --run-id baseline --force
git add configs/baseline.yaml
git commit -m "feat(tracking): lock world_stitch_sweep winner into baseline"
```

Update the `pipeline-state` memory with the new world HOTA / AssA and the shipped gate.

---

## Self-review

**Spec coverage:**
- Module `world_stitch.py` (summarize/find/resolve/stitch) → Tasks 1–4. ✓
- `WorldStitchCfg` + validation (`method`, `max_gap_frames>0`, `max_dist_m>0`, unknown-method reject) → Task 5. ✓
- Evaluate-stage wiring between aggregate and smooth + log + metrics + set_params → Task 6. ✓
- `world_stitch_sweep` (`rerun_from: evaluate`) + convention comment + baseline block + registry test → Task 7. ✓
- Decision rule (highest world HOTA, AssA ≥ 0.6423, DetA not down; null-result fallback) → Task 8 Step 4. ✓
- Scope = sequential strict-non-overlap only; concurrent/long-gap/velocity explicitly out → enforced by the `a.last_frame < b.first_frame` gate in Task 2; no task admits the others. ✓
- Edge cases (empty rows, single-frame gid, full-clip inert, defensive re-aggregation, determinism) → Tasks 1/4 tests. ✓

**Placeholder scan:** every code step contains full code; every run step has an exact command + expected output; no TBD/TODO. ✓

**Type consistency:** `Row = (frame, gid, x, y)` and the edge tuple `(dist, gap, gid_a, gid_b)` are used identically across Tasks 1–4. `stitch_world_tracks(rows, *, method, max_gap_frames, max_dist_m) -> (rows, merges)` matches its call site in Task 6 and the fake in the wiring test. `WorldStitchCfg(method, max_gap_frames, max_dist_m)` fields match Task 5 (config), Task 6 (stub cfg + set_params), and Task 7 (YAML keys). Class suffix is `Cfg` (matches `WorldSmoothingCfg`), not `Config`. ✓
