# MCT linking attribution → targeted fix

**Date:** 2026-05-31
**Branch:** TBD (suggest `feat/linking-attribution` once execution starts)
**Status:** approved design, pending execution
**Priority metric:** world HOTA (v3.1 baseline = **0.5282**)
**Supersedes the framing of:** `docs/superpowers/specs/2026-05-28-reid-fine-tune-design.md`
(that spec is now the **conditional L1 branch** of this workstream, not the default)
**Reads against:** `pipeline-state.md`, `detector-recall-workstream-paused.md`,
`reid-finetune-workstream-spec.md` memories

## Goal

Recover world-HOTA by fixing the cross-camera **linking** stage, which today
discards ~55% of all detections. Phase 0 attributes the loss to a specific gate;
Phase 1 fixes the dominant gate(s). Ship a Phase 1 fix when it gains **≥ +0.02
world HOTA** over the v3.1 baseline (0.5282).

## Why linking, not detection (verified 2026-05-31)

The world metric loses **16,421 of 29,843 detections (55%)** at linking, and the
loss is **100% a linking failure** — every dropped detection has a valid
`WorldCoordinate` but a null `GlobalOfflineID` (detected *and* posed, then never
linked). Zero are dropped for detection or pose failures.

Per-camera detections lost at linking:

| Camera | Detections | Linked | Lost at linking | % lost |
|---|---|---|---|---|
| 0390 | 6501 | 4096 | 2405 | 37% |
| 0391 | 4262 | 1661 | 2601 | 61% |
| 0392 | 6493 | 4235 | 2258 | 35% |
| 0393 | 6061 | 2883 | 3178 | 52% |
| 0394 | 3328 | 327 | 3001 | **90%** |
| 0395 | 2918 | 220 | 2698 | **92%** |
| 0396 | 280 | 0 | 280 | **100%** |
| **ALL** | **29843** | **13422** | **16421** | **55%** |

(Verified via `outputs/baseline/mct/scene_001/whole_tracking_results.json` +
`aic24_nvidia/world_tracks.py:27` drop logic.) The detector-recall workstream was
deprioritized on the same evidence (`detector-recall-workstream-paused.md` VERDICT
2026-05-31): camera 0394 has 92% detection recall yet only ~10% of its detections
link — detection quality does not control linking.

## What the sweep evidence already rules out

From the v3-era tracker sweeps (`outputs/{sim_th_sweep,eps_mcpt_sweep}__*`):

- **`sim_th` is inert.** 0.75 / 0.85 / 0.92 → *bit-identical* world HOTA
  (0.505504), DetA, AssA, dropped. The reid cosine *threshold* does not bind.
- **`eps_mcpt` binds** world HOTA (0.428 → 0.505 → 0.480) but **does not change the
  dropped count** (constant 16604).
- Therefore **whether a detection links at all is decided upstream of both
  `sim_th` and `eps_mcpt`** — at the eligibility gates below.

This is the central correction to the 2026-05-28 reid spec, whose hypothesis was
"fine-tune reid to push cosine margin past `sim_th=0.85`." `sim_th` never fires,
and the bulk of the loss happens *before* appearance matching even runs. A reid
fine-tune is justified only if Phase 0 shows the **L1 (appearance)** gate is the
dominant residual — not before.

## The linking gates (verified against upstream MCPT code)

Source: `external/AIC24_Track1_YACHIYO_RIIPS/tracking/src/mcpt.py`. A track gets a
`GlobalOfflineID` only if it (a) passes BOTH eligibility gates in
`create_camera_dict`, then (b) joins a cross-camera cluster. Four gates, in order:

- **E1 — eligibility: track length** (`create_camera_dict`, `mcpt.py:312`):
  `len(all_serials) < short_track_th` → excluded. `all_serials` is the rep-node's
  serial list (≈ the track's detection count). Baseline `short_track_th=120`.
- **E2 — eligibility: keypoint quality** (`create_camera_dict`, `mcpt.py:317`):
  representative-node `score > keypoint_condition_th` → excluded. Baseline
  `keypoint_condition_th=1`, so any track whose representative pose score is 2/3/4
  is dropped. **This gate was missed by the 2026-05-28 spec.**
- **L1 — link: appearance** (`mcpt.py:224`+): cosine similarity matrix over the
  eligible rep nodes; `similarity_matrix[similarity_matrix < (1-eps_mcpt)] = 0`
  (baseline `eps_mcpt=0.37` → edges below cosine 0.63 cut), then hierarchical
  clustering at `eps_mcpt`.
- **L2 — link: world distance** (`replace_similarity`, `mcpt.py:228`): eligible
  pairs farther apart than `distance_th` (baseline 10 m, `distance_type=min`) have
  their similarity overwritten to `replace_value` (cannot-link). Calibration-
  dependent.

(`sim_th` participates only in a later leftover-assignment path that this scene
does not exercise — consistent with its measured inertness.)

## Gate taxonomy (what Phase 0 measures)

Each of the 16,421 unlinked detections is attributed to the **first** gate that
excluded its track, in pipeline order **E1 → E2 → L1 → L2**:

- **E1** — track shorter than `short_track_th` (fragmentation / threshold).
- **E2** — representative keypoint score exceeds `keypoint_condition_th` (pose
  quality / threshold).
- **L1** — eligible, but appearance never clusters cross-camera (reid / `eps_mcpt`).
- **L2** — eligible and appearance-matchable, but world-distance override blocks
  (calibration / `distance_th`).
- **linked** (sanity) — must reconcile to 13,422.

### Preliminary signal (offline, from `representative_nodes_scene1.json`)

A first-pass eligibility rollup (E1 then E2 precedence; L1/L2 not yet split) over
the rep-node `all_serials`/`score` fields, baseline `short_track_th=120`,
`keypoint_condition_th=1`:

| Camera | rep-dets | excl E1 (len) | excl E2 (keypoint) | pass to clustering |
|---|---|---|---|---|
| 0390 | ~6494 | 868 | (large) | majority |
| 0394 | ~3308 | 514 | ~1276+ | ~1518 |
| 0395 | ~2916 | 880 | ~1410+ | **~626 (21%)** |
| 0396 | ~280 | 70 | small | ~210 (then 100% lost at L1/L2) |

So the loss is a **mix**: on the back cams 0394/0395 the **eligibility gates
(E1+E2) dominate** (E2 — keypoint quality — often larger than E1), while 0396
passes eligibility but loses everything at L1/L2. Front cams pass most of
eligibility and lose a smaller residual at L1/L2. These offline numbers are
directional; Phase 0 produces the exact per-gate detection counts.

## Phase 0 — Linking Attribution Probe

**Environment:** local `.venv`, **no GPU training**, hours not days.

**Method — instrumentation-primary (corrected from offline-first).** The
eligibility/linking logic has four interacting gates with subtle inputs
(`len(all_serials)`, keypoint `score`, the cosine-cut + hierarchical clustering,
the world-distance override). Faithful offline reconstruction is error-prone, so
the source of truth is a single instrumented MCT run:

1. **Instrument `mcpt.py`** (probe-only patch, NOT committed as an upstream
   patch): at each gate emit one log row per track —
   `(camera_id, local_id, n_serials, kp_score, max_cross_cam_cosine,
   min_cross_cam_world_dist, outcome ∈ {E1,E2,L1,L2,linked})`.
2. **Re-run MCT + evaluate only** (`python pipeline.py mct ...` then `evaluate`;
   SCT cached, ~minutes).
3. **Roll up** the log to detections-per-gate-per-camera (weight each track by its
   detection count), reconciling the `linked` total to 13,422 and the four gate
   totals to 16,421.
4. **Offline cross-check**: recompute E1/E2 from `representative_nodes_scene1.json`
   (exact) and confirm they match the instrumented E1/E2.

**Deliverables:**

- A per-camera × per-gate (E1/E2/L1/L2/linked) detection-count table covering all
  16,421 unlinked detections, back cams (0394/0395/0396) called out, reconciled to
  the totals above.
- An SCT-fragmentation summary: per-camera track count and length histogram, and
  for short tracks whether their gaps fall in frames that *had* detections
  (association failure) vs *lacked* them (detection sparsity). The 2026-05-31
  adversarial check found 73.8% of 0395 SCT-gap frames have ≥2 detections at
  conf 0.5 — confirm/quantify per camera.
- A keypoint-score distribution per camera (how E2 exclusion scales with pose
  quality), since E2 is the newly-surfaced suspect.
- A one-paragraph verdict naming the dominant gate(s) and the Phase 1 branch(es).

**Artifacts to read:** `outputs/baseline/mct/scene_001/{camera{NNN}_tracking_results.json
(pre-global-assignment SCT-staged; carry no GlobalOfflineID),
representative_nodes_scene1.json (rep nodes with all_serials + score),
whole_tracking_results.json (post-assignment)}`,
`aic24_nvidia/world_tracks.py`, and the MCPT source above. Probe code lives in
`scripts/` (e.g. `scripts/linking_attribution.py`), not in the shipped package.

## Decision tree → Phase 1

Attribute by detection count, tackle the **largest gate first**, re-measure, then
re-attribute the residual (gates unmask one another — admitting more tracks at E1
may expose L1 for them). Every branch carries the +0.02 world-HOTA ship bar.

### Cheap pre-check FIRST (before any model/calibration work)

Three of the four gates are pure `tracking_params` thresholds with
seconds-per-variant `rerun_from: mct` sweeps. Run a coarse grid as the first
action of *any* branch — it may capture most of the win for near-zero cost:

- `keypoint_condition_th`: 1 → 2 → 3 (admits lower-pose-quality tracks; targets E2)
- `short_track_th`: 120 → 60 → 30 (admits shorter tracks; targets E1)
- `eps_mcpt`: re-tune around 0.37 (targets L1)
- `distance_th`: 10 → 15 → 20 (loosens world cannot-link; targets L2)

Add these as `experiments/registry.yaml` blocks, `rerun_from: mct`. Guardrail:
loosening thresholds can admit false links and *drop* AssA/world HOTA — the +0.02
bar and per-camera global-ID sanity catch that.

### Branch E1 — SCT length / fragmentation

If short tracks dominate after the threshold sweep: reduce SCT fragmentation
(`epsilon_scpt` re-tune; track gap-filling/interpolation — note upstream patch #3
already touches `interpolate_tracklet`). Local, GPU-free.

### Branch E2 — keypoint quality

If the keypoint gate dominates and raising `keypoint_condition_th` alone over-admits
junk: improve pose-score quality on back cams (the score derives from RTMPose
keypoint confidence; oblique back-cam poses score poorly). May overlap the pose
stage rather than the tracker.

### Branch L1 — appearance / reid (the existing spec, now conditional)

If, *after* eligibility is fixed, the residual is dominated by eligible tracks that
fail to cluster on appearance: re-tune `eps_mcpt` with current embeddings first;
if embeddings are genuinely the limiter, execute
`docs/superpowers/specs/2026-05-28-reid-fine-tune-design.md` (partial SOLIDER
Swin-Small fine-tune in Colab). Before executing, refresh that spec's stale
numbers (it cites v3 world HOTA 0.5055 and SCT counts 11/10/1 for 0394/0395/0396;
current v3.1 is 0.5282 with SCT 48/95/3 and MCT global IDs 1/1/0) and re-point its
decision rule at the v3.1 baseline + the +0.02 bar.

### Branch L2 — calibration / world distance

If the world-distance override dominates: audit per-camera projection/homography
for the back cams (adapter writes `Original/scene_NNN/camera_NNNN/calibration.json`
— 3×4 projection + 3×3 homography). Cross-check projected `WorldCoordinate` for
known same-person cross-camera pairs against the distance the override sees;
`distance_th` re-tune as the cheap version.

## Success bar & decision rule

| Outcome (per Phase 1 fix) | Action |
|---|---|
| world HOTA gain ≥ +0.02 over v3.1 (0.5282) | **Ship** — lock as new baseline, update `pipeline-state.md`, bump `configs/baseline.yaml` version comment |
| Back-cam global IDs rise (0394/0395/0396 from 1/1/0) but gain < +0.02 | Keep if non-regressing; re-attribute residual, try next-largest gate |
| No movement | Verify the fix engaged (re-log gate counts); if engaged but flat, move to next branch |
| Regression (AssA / world HOTA down) | Revert — a loosened threshold over-admitted false links; tighten or move on |

## Out of scope

- **Detector recall** — deprioritized (`detector-recall-workstream-paused.md`
  VERDICT 2026-05-31). Not revisited unless Phase 0 shows E2/L1/L2 negligible *and*
  E1 traces to genuine detection sparsity (2026-05-31 check suggests it does not).
- **Generic reid quality** beyond cross-cam linking; **Swin architecture changes**;
  **end-to-end joint detector+reid training**.
- **Multi-scene tracker re-tuning** after a Phase 1 change — a follow-up if a fix
  ships.

## References

- World-tracks drop logic: `aic24_nvidia/world_tracks.py` (null `GlobalOfflineID`
  or bad world coords).
- MCPT gates: `external/AIC24_Track1_YACHIYO_RIIPS/tracking/src/mcpt.py`
  (E1/E2 in `create_camera_dict` ~`:300-320`; L1 cosine cut + clustering ~`:224`;
  L2 `replace_similarity` ~`:228`; `sim_th` leftover path ~`:378`).
- Tracker param wiring: `aic24_nvidia/tracking_params.py` →
  `tracking/config/parameters_per_scene.py`. Baseline values:
  `short_track_th=120`, `keypoint_condition_th=1`, `eps_mcpt=0.37`,
  `distance_th=10`, `distance_type=min`, `sim_th=0.85`.
- Existing reid fine-tune spec (L1 branch):
  `docs/superpowers/specs/2026-05-28-reid-fine-tune-design.md`.
- Prior diagnostic: `docs/superpowers/notes/2026-05-28-world-projection-results.md`.

## Hand-off checklist

1. ☐ Confirm v3.1 baseline (`outputs/baseline/`, world HOTA 0.5282) current:
   `cat outputs/baseline/evaluate/metrics.json | jq '.mct_world'`.
2. ☐ Write `scripts/linking_attribution.py` (offline E1/E2 rollup + log parser).
3. ☐ Add the probe-only instrumentation to `mcpt.py`; run MCT+evaluate once.
4. ☐ Produce the per-camera × per-gate table; reconcile to 16,421 / 13,422.
5. ☐ Run the cheap threshold-sweep grid (`keypoint_condition_th`, `short_track_th`,
   `eps_mcpt`, `distance_th`) as `rerun_from: mct` experiments.
6. ☐ Select the Phase 1 branch(es) from the decision tree; apply the +0.02 ship bar.
7. ☐ If shipping, update `pipeline-state.md` and `configs/baseline.yaml`.
8. ☐ If the residual branch is L1, refresh + execute the 2026-05-28 reid spec
   against the v3.1 baseline.
