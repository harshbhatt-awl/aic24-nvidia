# MCT Linking Attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested, offline tool that attributes every MCT-dropped detection to the exact eligibility gate that dropped it, and add the `tracking_params` threshold sweep that attacks those gates.

**Architecture:** A pure-Python diagnostics module (`aic24_nvidia/diagnostics/linking_attribution.py`) joins per-track records from the baseline MCT artifacts (`representative_nodes_scene1.json` for the gate inputs `all_serials`/`score`, `whole_tracking_results.json` for per-detection `OfflineID`/`GlobalOfflineID`), classifies each track by the first eligibility gate it fails (`G0` untracked → `E1` short → `E2` keypoint → `kept`), and rolls up detection counts per camera × gate with a hard reconciliation against the known totals (dropped 16,421 / kept 13,422). A thin CLI reuses the same functions on any run's `mct/` dir. A new `linking_gate_sweep` experiment varies `keypoint_condition_th` and `short_track_th` (`rerun_from: sct`) to attack the two gates.

**Tech Stack:** Python 3.14, stdlib only (`json`, `dataclasses`, `collections`, `argparse`), `pytest` for tests, `pyyaml` (already a dep) for the registry test. No GPU, no pipeline re-run for Phase 0.

---

## Verified model this plan rests on (from the spec)

A detection is **kept** iff its track passes BOTH `create_camera_dict` gates; this was verified exact on the current baseline:

- `E2` keypoint (`score > keypoint_condition_th`, baseline 1): **12,126 dropped (74%)**
- `E1` length (`len(all_serials) < short_track_th`, baseline 120): **4,254 dropped (26%)**
- `G0` untracked (`OfflineID < 0`): **41 dropped**
- **kept = 13,422** ; dropped total = 16,421 (no residual)

`eps_mcpt`/`distance_th`/reid are a separate *association* axis (they re-cluster the kept; they do not change the dropped count). This plan implements the **coverage axis** (the probe + the two threshold sweeps).

## File structure

- Create: `aic24_nvidia/diagnostics/__init__.py` — marks the diagnostics subpackage (diagnostic tools, not part of the stage flow).
- Create: `aic24_nvidia/diagnostics/linking_attribution.py` — pure functions (`classify_track`, `load_tracks`, `attribute`, `reconcile`) + `main()` CLI.
- Create: `tests/unit/test_linking_attribution.py` — unit tests for the pure functions.
- Modify: `experiments/registry.yaml` — append the `linking_gate_sweep` experiment.
- Modify: `tests/unit/test_experiment_harness.py` *(or new `tests/unit/test_linking_gate_sweep_registry.py`)* — assert the new block parses with the right `rerun_from` and variants.

---

### Task 1: Scaffold diagnostics package + `classify_track`

**Files:**
- Create: `aic24_nvidia/diagnostics/__init__.py`
- Create: `aic24_nvidia/diagnostics/linking_attribution.py`
- Test: `tests/unit/test_linking_attribution.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_linking_attribution.py
from aic24_nvidia.diagnostics.linking_attribution import (
    Track, classify_track, G0_UNTRACKED, E1_SHORT, E2_KEYPOINT, KEPT,
)


def _t(offline_id=5, n_all_serials=200, score=1, n_detections=10, linked=True):
    return Track(camera="395", offline_id=offline_id, n_detections=n_detections,
                 n_all_serials=n_all_serials, score=score, linked=linked)


def test_classify_kept_when_long_and_good_keypoints():
    assert classify_track(_t(n_all_serials=200, score=1), short_track_th=120,
                          keypoint_condition_th=1) == KEPT


def test_classify_e2_when_keypoint_score_exceeds_threshold():
    assert classify_track(_t(n_all_serials=200, score=2), short_track_th=120,
                          keypoint_condition_th=1) == E2_KEYPOINT


def test_classify_e1_when_track_too_short():
    assert classify_track(_t(n_all_serials=50, score=1), short_track_th=120,
                          keypoint_condition_th=1) == E1_SHORT


def test_classify_e1_takes_precedence_over_e2():
    # fails both gates; length is checked first
    assert classify_track(_t(n_all_serials=50, score=4), short_track_th=120,
                          keypoint_condition_th=1) == E1_SHORT


def test_classify_g0_when_untracked():
    assert classify_track(_t(offline_id=-1, n_all_serials=-1, score=-1),
                          short_track_th=120, keypoint_condition_th=1) == G0_UNTRACKED
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_linking_attribution.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aic24_nvidia.diagnostics'`

- [ ] **Step 3: Write minimal implementation**

```python
# aic24_nvidia/diagnostics/__init__.py
"""Offline diagnostics tools (not part of the pipeline stage flow)."""
```

```python
# aic24_nvidia/diagnostics/linking_attribution.py
from __future__ import annotations

from dataclasses import dataclass

# Gate labels in attribution (pipeline) order.
G0_UNTRACKED = "G0_untracked"
E1_SHORT = "E1_short_track"
E2_KEYPOINT = "E2_keypoint"
KEPT = "kept"
GATES = (G0_UNTRACKED, E1_SHORT, E2_KEYPOINT, KEPT)


@dataclass(frozen=True)
class Track:
    camera: str           # numeric camera key as in the MCT json, e.g. "395"
    offline_id: int       # SCT local id; -1 means untracked
    n_detections: int     # detections for this (camera, offline_id) in whole_tracking_results
    n_all_serials: int    # rep-node all_serials length (the E1 input); -1 if absent
    score: int            # rep-node representative score (the E2 input); -1 if absent
    linked: bool          # any detection of this track has a non-null GlobalOfflineID


def classify_track(track: Track, short_track_th: int, keypoint_condition_th: int) -> str:
    """First eligibility gate the track fails, mirroring create_camera_dict.

    Order: untracked (OfflineID<0) -> length (len(all_serials)<short_track_th)
    -> keypoint (score>keypoint_condition_th) -> kept.
    """
    if track.offline_id < 0:
        return G0_UNTRACKED
    if track.n_all_serials < short_track_th:
        return E1_SHORT
    if track.score > keypoint_condition_th:
        return E2_KEYPOINT
    return KEPT
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_linking_attribution.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add aic24_nvidia/diagnostics/__init__.py aic24_nvidia/diagnostics/linking_attribution.py tests/unit/test_linking_attribution.py
git commit -m "feat(diagnostics): linking-attribution gate classifier"
```

---

### Task 2: `load_tracks` — join MCT artifacts into Track records

**Files:**
- Modify: `aic24_nvidia/diagnostics/linking_attribution.py`
- Test: `tests/unit/test_linking_attribution.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/test_linking_attribution.py
import json
from aic24_nvidia.diagnostics.linking_attribution import load_tracks


def _write(tmp_path, whole, rep):
    (tmp_path / "whole_tracking_results.json").write_text(json.dumps(whole))
    (tmp_path / "representative_nodes_scene1.json").write_text(json.dumps(rep))
    return (tmp_path / "whole_tracking_results.json",
            tmp_path / "representative_nodes_scene1.json")


def test_load_tracks_joins_counts_scores_and_linked(tmp_path):
    whole = {
        "395": {
            "00000000": {"OfflineID": 7, "GlobalOfflineID": 14},
            "00000001": {"OfflineID": 7, "GlobalOfflineID": 14},
            "00000002": {"OfflineID": 8, "GlobalOfflineID": None},
            "00000003": {"OfflineID": -1, "GlobalOfflineID": None},
        }
    }
    rep = {
        "395": {
            "7": {"representative_node": {"score": 1}, "all_serials": ["a"] * 200},
            "8": {"representative_node": {"score": 3}, "all_serials": ["a"] * 50},
        }
    }
    wj, rj = _write(tmp_path, whole, rep)
    tracks = {(t.camera, t.offline_id): t for t in load_tracks(wj, rj)}

    assert tracks[("395", 7)].n_detections == 2
    assert tracks[("395", 7)].n_all_serials == 200
    assert tracks[("395", 7)].score == 1
    assert tracks[("395", 7)].linked is True

    assert tracks[("395", 8)].n_detections == 1
    assert tracks[("395", 8)].n_all_serials == 50
    assert tracks[("395", 8)].linked is False

    # untracked detection: offline_id normalized to -1, no rep node -> (-1, -1)
    assert tracks[("395", -1)].n_detections == 1
    assert tracks[("395", -1)].n_all_serials == -1
    assert tracks[("395", -1)].score == -1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_linking_attribution.py::test_load_tracks_joins_counts_scores_and_linked -v`
Expected: FAIL — `ImportError: cannot import name 'load_tracks'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to aic24_nvidia/diagnostics/linking_attribution.py
import json
from collections import defaultdict
from pathlib import Path


def load_tracks(whole_json: str | Path, rep_json: str | Path) -> list[Track]:
    """Build Track records by joining the post-assignment whole_tracking_results
    (per-detection OfflineID/GlobalOfflineID) with representative_nodes (the gate
    inputs all_serials/score) on (camera, offline_id)."""
    whole = json.loads(Path(whole_json).read_text())
    rep = json.loads(Path(rep_json).read_text())

    rep_by: dict[tuple[str, int], tuple[int, int]] = {}
    for cam, nodes in rep.items():
        if not isinstance(nodes, dict):
            continue
        for oid_str, node in nodes.items():
            n_all = len(node["all_serials"])
            score = int(node["representative_node"]["score"])
            rep_by[(cam, int(oid_str))] = (n_all, score)

    ndet: dict[tuple[str, int], int] = defaultdict(int)
    linked: dict[tuple[str, int], bool] = defaultdict(bool)
    for cam, entries in whole.items():
        if not isinstance(entries, dict):
            continue
        for entry in entries.values():
            if not isinstance(entry, dict):
                continue
            oid = entry.get("OfflineID")
            oid = -1 if oid is None else int(oid)
            ndet[(cam, oid)] += 1
            if entry.get("GlobalOfflineID") is not None:
                linked[(cam, oid)] = True

    tracks: list[Track] = []
    for (cam, oid), n in ndet.items():
        n_all, score = rep_by.get((cam, oid), (-1, -1))
        tracks.append(Track(cam, oid, n, n_all, score, linked[(cam, oid)]))
    return tracks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_linking_attribution.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add aic24_nvidia/diagnostics/linking_attribution.py tests/unit/test_linking_attribution.py
git commit -m "feat(diagnostics): load_tracks joins MCT artifacts into Track records"
```

---

### Task 3: `attribute` + `reconcile` — per-camera × per-gate rollup with consistency check

**Files:**
- Modify: `aic24_nvidia/diagnostics/linking_attribution.py`
- Test: `tests/unit/test_linking_attribution.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/test_linking_attribution.py
from aic24_nvidia.diagnostics.linking_attribution import attribute, reconcile


def _tracks():
    # two cameras; counts chosen so totals are easy to check
    return [
        Track("390", 1, n_detections=100, n_all_serials=200, score=1, linked=True),   # kept
        Track("390", 2, n_detections=30,  n_all_serials=50,  score=1, linked=False),  # E1
        Track("390", 3, n_detections=40,  n_all_serials=200, score=3, linked=False),  # E2
        Track("395", 4, n_detections=10,  n_all_serials=-1,  score=-1, linked=False), # E1 (absent rep)
        Track("395", -1, n_detections=5,  n_all_serials=-1,  score=-1, linked=False), # G0
    ]


def test_attribute_rolls_up_per_camera_and_gate():
    per_cam, totals = attribute(_tracks(), short_track_th=120, keypoint_condition_th=1)
    assert per_cam["390"][KEPT] == 100
    assert per_cam["390"][E1_SHORT] == 30
    assert per_cam["390"][E2_KEYPOINT] == 40
    assert per_cam["395"][E1_SHORT] == 10
    assert per_cam["395"][G0_UNTRACKED] == 5
    assert totals[KEPT] == 100
    assert totals[E1_SHORT] == 40
    assert totals[E2_KEYPOINT] == 40
    assert totals[G0_UNTRACKED] == 5


def test_reconcile_flags_clean_when_kept_matches_linked():
    rep = reconcile(_tracks(), short_track_th=120, keypoint_condition_th=1)
    assert rep["dropped"] == 85          # 30+40+10+5
    assert rep["kept"] == 100
    assert rep["kept_linked_mismatch"] == 0   # classify==KEPT iff linked, on this fixture
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_linking_attribution.py -v`
Expected: FAIL — `ImportError: cannot import name 'attribute'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to aic24_nvidia/diagnostics/linking_attribution.py


def attribute(tracks: list[Track], short_track_th: int, keypoint_condition_th: int):
    """Return (per_camera, totals) detection-count rollups keyed by gate label."""
    per_cam: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for t in tracks:
        gate = classify_track(t, short_track_th, keypoint_condition_th)
        per_cam[t.camera][gate] += t.n_detections
    totals: dict[str, int] = defaultdict(int)
    for gates in per_cam.values():
        for g, c in gates.items():
            totals[g] += c
    return {c: dict(g) for c, g in per_cam.items()}, dict(totals)


def reconcile(tracks: list[Track], short_track_th: int, keypoint_condition_th: int):
    """Roll up and verify the kept<->linked invariant the model rests on."""
    _, totals = attribute(tracks, short_track_th, keypoint_condition_th)
    kept = totals.get(KEPT, 0)
    dropped = sum(totals.get(g, 0) for g in (G0_UNTRACKED, E1_SHORT, E2_KEYPOINT))
    mismatch = 0
    for t in tracks:
        is_kept = classify_track(t, short_track_th, keypoint_condition_th) == KEPT
        if is_kept != t.linked:
            mismatch += t.n_detections
    return {"kept": kept, "dropped": dropped, "totals": totals,
            "kept_linked_mismatch": mismatch}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_linking_attribution.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add aic24_nvidia/diagnostics/linking_attribution.py tests/unit/test_linking_attribution.py
git commit -m "feat(diagnostics): per-camera gate rollup + kept/linked reconciliation"
```

---

### Task 4: CLI `main()` + verify against the real baseline

**Files:**
- Modify: `aic24_nvidia/diagnostics/linking_attribution.py`
- Test: `tests/unit/test_linking_attribution.py`

- [ ] **Step 1: Write the failing test (table formatting is pure, so test it)**

```python
# append to tests/unit/test_linking_attribution.py
from aic24_nvidia.diagnostics.linking_attribution import format_table


def test_format_table_contains_gate_columns_and_totals():
    per_cam, totals = attribute(_tracks(), short_track_th=120, keypoint_condition_th=1)
    text = format_table(per_cam, totals)
    assert "E2_keypoint" in text
    assert "E1_short_track" in text
    assert "ALL" in text
    # the kept total (100) and an E1 total (40) appear in the rendered table
    assert "100" in text and "40" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_linking_attribution.py::test_format_table_contains_gate_columns_and_totals -v`
Expected: FAIL — `ImportError: cannot import name 'format_table'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to aic24_nvidia/diagnostics/linking_attribution.py
import argparse


def format_table(per_cam: dict, totals: dict) -> str:
    cols = [E1_SHORT, E2_KEYPOINT, G0_UNTRACKED, KEPT]
    head = f"{'camera':<8}" + "".join(f"{c:>16}" for c in cols)
    lines = [head, "-" * len(head)]
    for cam in sorted(per_cam):
        row = f"{cam:<8}" + "".join(f"{per_cam[cam].get(c, 0):>16}" for c in cols)
        lines.append(row)
    lines.append("-" * len(head))
    lines.append(f"{'ALL':<8}" + "".join(f"{totals.get(c, 0):>16}" for c in cols))
    dropped = sum(totals.get(c, 0) for c in (G0_UNTRACKED, E1_SHORT, E2_KEYPOINT))
    lines.append(f"\ndropped (G0+E1+E2) = {dropped} | kept = {totals.get(KEPT, 0)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Attribute MCT-dropped detections to eligibility gates.")
    p.add_argument("--mct-dir", required=True,
                   help="path to outputs/<run>/mct/scene_001 (holds whole_tracking_results.json + representative_nodes_scene1.json)")
    p.add_argument("--short-track-th", type=int, default=120)
    p.add_argument("--keypoint-condition-th", type=int, default=1)
    args = p.parse_args(argv)

    mct = Path(args.mct_dir)
    tracks = load_tracks(mct / "whole_tracking_results.json",
                         mct / "representative_nodes_scene1.json")
    per_cam, totals = attribute(tracks, args.short_track_th, args.keypoint_condition_th)
    rep = reconcile(tracks, args.short_track_th, args.keypoint_condition_th)
    print(format_table(per_cam, totals))
    if rep["kept_linked_mismatch"]:
        print(f"WARNING: kept/linked mismatch on {rep['kept_linked_mismatch']} detections "
              "(the kept<->eligible invariant did not hold for this run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run unit tests, then run the CLI on the real baseline**

Run: `pytest tests/unit/test_linking_attribution.py -v`
Expected: PASS (9 passed)

Run: `python -m aic24_nvidia.diagnostics.linking_attribution --mct-dir outputs/baseline/mct/scene_001`
Expected: a per-camera table whose footer reads `dropped (G0+E1+E2) = 16421 | kept = 13422`, no mismatch warning, with `ALL` row `E1_short_track = 4254`, `E2_keypoint = 12126`, `G0_untracked = 41`, `kept = 13422`.

> If the dropped total is not 16,421 / kept not 13,422, STOP — the artifacts or the join are wrong; do not proceed to the sweep.

- [ ] **Step 5: Commit**

```bash
git add aic24_nvidia/diagnostics/linking_attribution.py tests/unit/test_linking_attribution.py
git commit -m "feat(diagnostics): linking-attribution CLI; reconciles baseline 16421/13422"
```

---

### Task 5: Add the `linking_gate_sweep` experiment + registry parse test

**Files:**
- Modify: `experiments/registry.yaml`
- Test: `tests/unit/test_linking_gate_sweep_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_linking_gate_sweep_registry.py
from pathlib import Path
import yaml

REGISTRY = Path(__file__).resolve().parents[2] / "experiments" / "registry.yaml"


def _experiments():
    data = yaml.safe_load(REGISTRY.read_text())
    return {e["id"]: e for e in data["experiments"]}


def test_linking_gate_sweep_present_and_reruns_from_sct():
    exp = _experiments()["linking_gate_sweep"]
    assert exp["base_config"] == "configs/baseline.yaml"
    # tracking_params changes must rerun from sct per registry convention
    assert exp["rerun_from"] == "sct"


def test_linking_gate_sweep_varies_the_two_eligibility_gates():
    exp = _experiments()["linking_gate_sweep"]
    names = {v["name"] for v in exp["variants"]}
    # baseline control + keypoint and short-track variants
    assert "baseline" in names
    kp_vals, st_vals = set(), set()
    for v in exp["variants"]:
        tp = v["overrides"].get("tracking_params", {})
        if "keypoint_condition_th" in tp:
            kp_vals.add(tp["keypoint_condition_th"])
        if "short_track_th" in tp:
            st_vals.add(tp["short_track_th"])
    assert {2, 3} <= kp_vals          # raises the keypoint gate (E2, 74% of drops)
    assert {60, 30} <= st_vals        # lowers the length gate (E1, 26% of drops)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_linking_gate_sweep_registry.py -v`
Expected: FAIL — `KeyError: 'linking_gate_sweep'`

- [ ] **Step 3: Append the experiment block to `experiments/registry.yaml`**

```yaml
  - id: linking_gate_sweep
    description: "Sweep the two MCPT eligibility gates that drop 100% of unlinked detections (E2 keypoint 74%, E1 length 26%)."
    hypothesis: "Admitting more tracks past keypoint_condition_th / short_track_th raises world DetA; net world HOTA rises if the admitted tracks aren't too noisy for AssA."
    base_config: configs/baseline.yaml
    rerun_from: sct          # tracking_params.* -> sct (registry convention); detect/reid/pose reused
    variants:
      - name: "baseline"     # control: keypoint_condition_th=1, short_track_th=120
        overrides:
          tracking_params:
            keypoint_condition_th: 1
            short_track_th: 120
      - name: "kp2"          # E2: admit score<=2
        overrides:
          tracking_params:
            keypoint_condition_th: 2
      - name: "kp3"          # E2: admit score<=3
        overrides:
          tracking_params:
            keypoint_condition_th: 3
      - name: "kp4"          # E2: admit all (score<=4)
        overrides:
          tracking_params:
            keypoint_condition_th: 4
      - name: "st60"         # E1: admit tracks >=60
        overrides:
          tracking_params:
            short_track_th: 60
      - name: "st30"         # E1: admit tracks >=30
        overrides:
          tracking_params:
            short_track_th: 30
      - name: "st15"         # E1: admit tracks >=15
        overrides:
          tracking_params:
            short_track_th: 15
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_linking_gate_sweep_registry.py -v`
Expected: PASS (2 passed)

Run (regression guard — the registry-consistency / harness tests must still pass): `pytest tests/unit/test_experiment_harness.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add experiments/registry.yaml tests/unit/test_linking_gate_sweep_registry.py
git commit -m "feat(experiments): linking_gate_sweep over keypoint_condition_th + short_track_th"
```

---

### Task 6: Execute the sweep and apply the decision rule (execution, not unit-tested)

**Files:** none (produces `outputs/linking_gate_sweep__*/` + analysis).

- [ ] **Step 1: Ensure the baseline is built**

Run: `python experiments/run.py ensure-baseline`
Expected: `outputs/baseline/` present with `evaluate/metrics.json` (`mct_world.HOTA` ≈ 0.5282).

- [ ] **Step 2: Run the sweep**

Run: `python experiments/run.py run linking_gate_sweep`
Expected: one `outputs/linking_gate_sweep__<name>/evaluate/metrics.json` per variant (minutes each — detect/reid/pose reused via symlink, only sct/mct/evaluate re-run).

- [ ] **Step 3: Attribute each variant (confirm the gate actually admitted detections)**

Run (per variant `<name>`):
`python -m aic24_nvidia.diagnostics.linking_attribution --mct-dir outputs/linking_gate_sweep__<name>/mct/scene_001 --keypoint-condition-th <kp> --short-track-th <st>`
Expected: `kept` rises vs the 13,422 baseline for the loosened variants (e.g. `kp4` admits most of the 12,126 E2 drops).

- [ ] **Step 4: Read the metrics table and apply the decision rule**

Run: `python experiments/compare.py --sort-by mct_world.HOTA`
Expected: a table of each variant's `mct_world.HOTA`, `AssA`, `DetA` vs baseline. Then apply the spec's decision rule:
- Any variant with **world HOTA ≥ 0.5482 (+0.02)** and **AssA not below baseline 0.6338** → ship it (next step).
- DetA up but world HOTA flat / AssA down → the residual is association-axis; stop here and open the reid/`eps_mcpt` follow-up (the 2026-05-28 reid spec, refreshed).
- Nothing moves DetA → re-check `parameters_per_scene.py` propagation.

- [ ] **Step 5 (only if a variant ships): lock the new baseline**

Edit `configs/baseline.yaml` `tracking_params` to the winning `keypoint_condition_th` / `short_track_th`, bump the version comment, then:

```bash
python pipeline.py all --config configs/baseline.yaml --run-id baseline --force
git add configs/baseline.yaml
git commit -m "feat(tracking): lock linking_gate_sweep winner as v3.2 baseline"
```

Update the `pipeline-state` memory with the new world HOTA and the shipped thresholds.

---

## Self-review

**Spec coverage:**
- Phase 0 offline exact probe → Tasks 1-4 (classify, load, attribute/reconcile, CLI + baseline verification reproducing 16,421/13,422). ✓
- Per-camera × per-gate table + reconciliation → Task 4 `format_table` + the CLI run. ✓
- Phase 1 coverage-axis sweep (`keypoint_condition_th`, `short_track_th`, `rerun_from: sct`) → Task 5. ✓
- Decision rule (+0.02 world HOTA, AssA guardrail; association-axis fallback) → Task 6 Step 4. ✓
- Association axis / reid fine-tune is explicitly *out of this plan* (fallback only) — matches the spec. ✓
- Keypoint-score histogram deliverable (spec Phase 0): NOT implemented as code here — it is a one-off analysis, not needed for the decision; noted as optional, omitted to keep the tool focused (YAGNI). If wanted, it is a trivial addition to the CLI later.

**Placeholder scan:** no TBD/TODO; every code step has complete code; every run step has an exact command + expected output. ✓

**Type consistency:** `Track` fields (`camera, offline_id, n_detections, n_all_serials, score, linked`) are used identically across Tasks 1-4; gate constants (`G0_UNTRACKED, E1_SHORT, E2_KEYPOINT, KEPT`) defined in Task 1 and reused verbatim in Tasks 3-4 tests and `format_table`. CLI flag names (`--mct-dir`, `--short-track-th`, `--keypoint-condition-th`) consistent between Task 4 impl and Task 6 usage. ✓
