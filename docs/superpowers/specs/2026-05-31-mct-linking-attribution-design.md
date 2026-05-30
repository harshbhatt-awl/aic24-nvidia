# MCT linking attribution → targeted fix

**Date:** 2026-05-31
**Branch:** `feat/linking-attribution`
**Status:** approved design (gate model verified exact), pending implementation plan
**Priority metric:** world HOTA (v3.1 baseline = **0.5282**)
**Reframes:** `docs/superpowers/specs/2026-05-28-reid-fine-tune-design.md` (that spec
is an **association-axis** workstream — it improves link *quality* among kept
detections; it does **not** reduce the dropped count, see below)
**Reads against:** `pipeline-state.md`, `detector-recall-workstream-paused.md`,
`reid-finetune-workstream-spec.md`, `linking-attribution-workstream-spec.md` memories

## Goal

Recover world-HOTA by reducing the ~55% of detections that the cross-camera
linking stage drops. Phase 0 attributes the loss to specific MCPT gates (done —
exact, below); Phase 1 sweeps the two threshold gates that cause it. Ship a Phase 1
fix when it gains **≥ +0.02 world HOTA** over the v3.1 baseline (0.5282) without
regressing AssA.

## Why linking, not detection (verified 2026-05-31)

The world metric loses **16,421 of 29,843 detections (55%)** at linking, and the
loss is **100% a linking failure** — every dropped detection has a valid
`WorldCoordinate` but a null `GlobalOfflineID` (detected *and* posed, then never
linked). Zero are dropped for detection or pose failures. The detector-recall
workstream was deprioritized on this same evidence
(`detector-recall-workstream-paused.md`): camera 0394 has 92% detection recall yet
only ~10% of its detections link.

(Verified via `outputs/baseline/mct/scene_001/whole_tracking_results.json` +
`aic24_nvidia/world_tracks.py:27` drop logic.)

## The exact drop model (verified 2026-05-31)

A detection is **kept** (gets a `GlobalOfflineID`) **if and only if** its SCT track
passes BOTH eligibility gates in `create_camera_dict`
(`external/AIC24_Track1_YACHIYO_RIIPS/tracking/src/mcpt.py` ~`:300-320`). Verified
exactly: `eligible_dets == kept_dets` for every camera; `kept = 13,422`;
zero eligible-but-unlinked, zero ineligible-but-linked. The full 16,421 dropped
decompose with no residual:

| Gate | What it is | Dropped dets | Share |
|---|---|---|---|
| **E2 — keypoint quality** | rep-node `score > keypoint_condition_th` (baseline **1**) → excluded | **12,126** | **74%** |
| **E1 — track length** | `len(all_serials) < short_track_th` (baseline 120) → excluded | **4,254** | 26% |
| **G0 — untracked** | detection's `OfflineID` is `-1`/None (SCT never tracked it) | 41 | 0.3% |
| **(kept)** | passes E1 ∧ E2 → assigned a GlobalOfflineID | 13,422 | — |

**Per-camera E1/E2 dropped detections** (baseline `short_track_th=120`,
`keypoint_condition_th=1`):

| Camera | kept | E1 (len) | E2 (keypoint) |
|---|---|---|---|
| 0390 | 4096 | 868 | 1530 |
| 0391 | 1661 | 867 | 1732 |
| 0392 | 4235 | 596 | 1656 |
| 0393 | 2883 | 459 | 2715 |
| 0394 | 327 | 514 | 2467 |
| 0395 | 220 | 880 | 1816 |
| 0396 | 0 | 70 | 210 |
| **ALL** | **13422** | **4254** | **12126** |

**The keypoint-quality gate (E2) is the dominant cause of the entire linking loss
(74%)** — and it was invisible to the 2026-05-28 reid spec. The track-length gate
(E1) is second (26%).

## Two distinct improvement axes (do not conflate)

The probe showed the dropped count is a function of *only* the two eligibility
thresholds. The other MCPT knobs operate on a different axis:

- **Coverage axis — reduces drops, raises world DetA/recall.** `keypoint_condition_th`
  (E2) and `short_track_th` (E1). Raising/lowering them admits more tracks into the
  kept set. **This is the Phase 1 lever** (it attacks 100% of the drops).
- **Association axis — link *quality* among the kept, raises AssA/IDF1, does NOT
  change drops.** `eps_mcpt` (clustering), `distance_th` (world cannot-link), and
  reid embedding quality (the 2026-05-28 fine-tune). Verified: `eps_mcpt` swings
  world HOTA via AssA (0.428→0.505→0.480) while the dropped count stays constant
  (16604); `sim_th` is fully inert.

The reid fine-tune is an **association-axis** workstream. It cannot recover the
16,421 dropped detections — it can only re-cluster the 13,422 already kept. So it
is pursued only as a *second* lever, after the coverage axis, if AssA among the
(enlarged) kept set is the residual limiter.

## Phase 0 — Linking Attribution Probe (DONE in design; productionize as a script)

**Offline and exact — no instrumentation, no pipeline re-run.** Inputs:
`outputs/baseline/mct/scene_001/{representative_nodes_scene1.json (all_serials +
score per track), whole_tracking_results.json (per-detection OfflineID +
GlobalOfflineID)}`.

**Method:** join tracks on `(camera, OfflineID)`; classify each track by the first
gate it fails in pipeline order — `OfflineID<0 → G0`, else
`len(all_serials) < short_track_th → E1`, else `score > keypoint_condition_th → E2`,
else `kept`; weight by the track's detection count from
`whole_tracking_results.json`. Reconcile: `E1 + E2 + G0` must equal the
null-`GlobalOfflineID` count (16,421) and `kept` must equal 13,422.

**Deliverables:** the per-camera × per-gate table above (regenerated from current
baseline), a per-camera keypoint-score histogram (how E2 scales with pose quality),
and the verdict (E2 dominant, then E1). Shipped as a reusable, tested module so the
same attribution reruns on any variant's `outputs/<run>/mct/`.

## Phase 1 — coverage-axis sweep (the fix attempt)

Three of the relevant knobs are pure `tracking_params` thresholds with cheap
`rerun_from: sct` sweeps (per the registry convention — `tracking_params.* → sct`;
detect/reid/pose are reused, so each variant is minutes). Add one experiment block
to `experiments/registry.yaml`:

- **`keypoint_condition_th`**: 1 (baseline) → 2 → 3 → 4. Targets E2 (74% of drops).
- **`short_track_th`**: 120 (baseline) → 60 → 30 → 15. Targets E1 (26% of drops).
- A small cross-product of the best of each, if singly-swept winners don't reach the
  bar.

Run the probe (Phase 0) on each variant to confirm the gate it targets actually
admitted the expected detections, and read world HOTA / AssA / DetA from each
variant's `metrics.json`.

**Guardrail / why this isn't a free win:** the E2-excluded tracks have *poor*
keypoint/pose quality (score 2–4) and E1-excluded tracks are short/fragmented.
Admitting them raises DetA but can inject noisy world points that *lower* precision
and AssA — net world HOTA may not rise. The +0.02 bar and an AssA-non-regression
check are the decision gate.

## Decision rule

| Outcome | Action |
|---|---|
| A coverage variant gains ≥ +0.02 world HOTA, AssA not regressed | **Ship** — lock as new baseline, update `pipeline-state.md`, bump `configs/baseline.yaml` |
| Coverage admits detections (DetA up) but world HOTA flat / AssA down | The kept set's *association* is now limiting → move to the **association axis**: cheap `eps_mcpt`/`distance_th` re-tune first, then the reid fine-tune (`2026-05-28-reid-fine-tune-design.md`, refreshed to v3.1) |
| No coverage variant moves DetA | Verify the threshold actually propagated (`parameters_per_scene.py`); the E2/E1 admitted tracks may be duplicates SCT already merged |
| Regression on every variant | Baseline thresholds are already optimal for this scene; the loss is intrinsic to track quality → SCT-fragmentation / pose-quality work (heavier, separate spec) |

## Out of scope

- **Detector recall** — deprioritized (`detector-recall-workstream-paused.md`).
- **The reid fine-tune as a drop fix** — it is association-axis only; it is the
  *fallback* lever per the decision rule, executed against its own 2026-05-28 spec.
- **Generic reid quality**, **Swin architecture changes**, **end-to-end joint
  training**, **multi-scene re-tuning** (follow-up if a fix ships).

## References

- Drop logic: `aic24_nvidia/world_tracks.py:27` (null `GlobalOfflineID` → dropped).
- Eligibility gates: `external/AIC24_Track1_YACHIYO_RIIPS/tracking/src/mcpt.py`
  `create_camera_dict` (E1 `len(all_serials) < short_track_th`; E2
  `score > keypoint_condition_th`) ~`:312`/`:317`; clustering (association axis)
  ~`:224`; `replace_similarity` world cannot-link ~`:228`; inert `sim_th` path ~`:378`.
- Param wiring: `aic24_nvidia/tracking_params.py` → `tracking/config/parameters_per_scene.py`.
  Baseline: `short_track_th=120`, `keypoint_condition_th=1`, `eps_mcpt=0.37`,
  `distance_th=10`, `distance_type=min`, `sim_th=0.85`.
- Association-axis spec: `docs/superpowers/specs/2026-05-28-reid-fine-tune-design.md`.

## Hand-off checklist

1. ☐ Confirm v3.1 baseline current: `jq '.mct_world' outputs/baseline/evaluate/metrics.json`.
2. ☐ Implement + test `aic24_nvidia/diagnostics/linking_attribution.py`; reproduce the
   table above from `outputs/baseline/` (E1+E2+G0 = 16,421; kept = 13,422).
3. ☐ Add the `linking_gate_sweep` experiment (`keypoint_condition_th`, `short_track_th`),
   `rerun_from: sct`.
4. ☐ Run the sweep; run the probe on each variant; read world HOTA / AssA / DetA.
5. ☐ Apply the decision rule; if shipping, update `pipeline-state.md` + `configs/baseline.yaml`.
6. ☐ If the residual is association-axis, refresh + execute the 2026-05-28 reid spec.
