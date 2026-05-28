# World-projection sweep results

**Date:** 2026-05-28
**Branch:** `feat/world-projection`
**Spec:** `docs/superpowers/specs/2026-05-28-ankle-footpoints-and-world-smoother-design.md`
**Plan:** `docs/superpowers/plans/2026-05-28-ankle-footpoints-and-world-smoother.md`

## Headline

| Metric | v2 baseline | v3 (this branch) | Δ |
|---|---|---|---|
| Image HOTA (combined) | 0.7580 | 0.7580 | 0.0000 |
| Image IDF1 | 0.7495 | 0.7495 | 0.0000 |
| Image MOTA | 0.7806 | 0.7806 | 0.0000 |
| **World HOTA** | **0.4583** | **0.5055** | **+0.0472** (+10.3% relative) |
| World DetA | 0.3970 | 0.4301 | +0.0331 |
| **World AssA** | **0.5297** | **0.5946** | **+0.0649** ⭐ |
| World IDF1 | 0.6475 | 0.6587 | +0.0112 |
| World MOTA | 0.4927 | 0.5056 | +0.0129 |

Image-space metrics are byte-identical: SCPT does not consume `WorldCoordinate`,
so injecting the rewrite between SCT and MCT does not change single-camera
tracking — only the cross-camera association and the world-space eval.

The biggest gain is in **AssA** (+6.5 percentage points): better world
coordinates let MCT pair the same person across cameras more confidently.

## What was changed

Two improvements, both config-gated. Defaults in `configs/baseline.yaml` now
locked at the winning combo:

```yaml
world_projection:
  method: ankle_lower        # was: bbox_bottom
  ankle_min_conf: 0.3

world_smoothing:
  method: ema                # was: none
  ema_alpha: 0.3
```

1. **Ankle-lower footpoint** — RTMPose's left/right ankle keypoint with the
   larger pixel-y (the planted foot in image coordinates) is projected through
   the per-camera homography instead of bbox-bottom-center. Module:
   `aic24_nvidia/world_projection.py`. Hook point:
   `aic24_nvidia/stages/mct.py` after staging SCT output and before invoking
   upstream MCT. Pure no-op when method is `bbox_bottom`.
2. **EMA temporal smoother** — per-(frame, global_id) world-coord EMA with
   α=0.3. Pure post-process in `aic24_nvidia/world_tracks.py`; runs after
   MCT, before eval — does not affect tracker decisions.

## Sweep results (full)

Sorted by world HOTA. The `runtime` column reads off summed stage manifests
including the cache-symlinked stages; actual marginal variant cost was minutes,
not the ~3340 s shown.

| Variant | image HOTA | world HOTA | Δ world HOTA | Notes |
|---|---|---|---|---|
| `ankle_lower × ema_0.3` | 0.7085 | **0.5055** | **+0.0472** | new baseline |
| `ankle_lower × ema_0.25` | 0.7085 | 0.5053 | +0.0470 | within noise of 0.3 |
| `ankle_lower × ema_0.35` | 0.7085 | 0.5046 | +0.0463 | within noise of 0.3 |
| `ankle_lower × ema_0.4` | 0.7085 | 0.5034 | +0.0451 | |
| `ankle_lower × ema_0.2` | 0.7085 | 0.5014 | +0.0431 | |
| `ankle_lower × ema_0.5` | 0.7085 | 0.5009 | +0.0426 | |
| `ankle_lower` (no smoothing) | 0.7085 | 0.4932 | +0.0349 | |
| `ema_0.3` (no ankle) | 0.7085 | 0.4756 | +0.0173 | |
| `ema_0.5` (no ankle) | 0.7085 | 0.4665 | +0.0082 | |
| `ema_0.7` (no ankle) | 0.7085 | 0.4616 | +0.0033 | |
| `bbox_bottom`, `none` (controls) | 0.7085 | 0.4583 | 0.0000 | byte-identical to baseline |
| `ankle_lower × ema_0.1` | 0.7085 | 0.4458 | −0.0125 | **over-smoothed** |
| `ankle_avg` | 0.7085 | 0.4272 | −0.0312 | mid-stride foot is in air → projection above ground |
| `ankle_w_fallback` | 0.7085 | 0.4257 | −0.0327 | reverts to ankle_avg when both scores ≥ 0.3 |

## Why `ankle_lower` beats `ankle_avg`

The literature reflex is to *average* both ankles. It loses here because the
homography projects from the 2D pixel plane to the **floor** plane (Z=0).
During a walking gait, the swing foot is well above the floor. Averaging both
ankles puts the projection point above the floor too, and the homography then
places the resulting world point *far behind* the actual standing position
(because the homography "lifts" off-floor points along the camera ray, not
straight down).

Picking the planted foot (the ankle with the larger pixel-y — closer to the
bottom of the image) avoids this entirely. The planted ankle is on the floor
by definition, so the homography projects it accurately.

This is borne out in the numbers: `ankle_avg` regresses by −0.031 world HOTA;
`ankle_lower` wins by +0.035. A net swing of **0.066 HOTA between two methods
that sound equivalent on paper**.

## Composition pattern

Solo standalone gains: ankle_lower +0.0349; ema_0.3 +0.0173. Sum = +0.0522.
Combined (`ankle_lower × ema_0.3`): +0.0472. **Combined gain is ~91% of the
additive prediction** — slightly sub-additive, consistent with the two
improvements attacking mostly-orthogonal error modes:

- ankle_lower fixes *systematic* projection error (wrong projection point).
- EMA fixes *high-frequency* temporal noise in the projected coords.

The 9% shortfall from perfect additivity is the small residual overlap (some
of the per-frame noise was *itself* caused by bbox-bottom projection of
varying gait pose, which ankle_lower also reduces).

## Diagnostic: where the remaining world-HOTA gap lives

A breakdown of the 16,604 dropped detections in the baseline world eval
revealed two real bottlenecks **distinct from anything this branch addresses**:

### Detector recall on cams 0395 / 0396

| Cam | GT entries | YOLO detections | Detector recall |
|---|---|---|---|
| 0390 | 6990 | 6501 | 93% |
| 0391 | 4318 | 4262 | 99% |
| 0392 | 7659 | 6493 | 85% |
| 0393 | 6206 | 6061 | 98% |
| 0394 | 3476 | 3328 | 96% |
| **0395** | **4712** | **2918** | **62%** ⚠️ |
| **0396** | **791** | **280** | **35%** ⚠️ |

YOLO11-x misses 65% of cam 0396's GT instances and 38% of cam 0395's. This
caps the per-camera image HOTA mathematically (0395 = 0.50, 0396 = 0.42) and
also reduces what MCT can cluster.

### MCT cross-camera linking failure on cams 0394 / 0395 / 0396

Per-camera SCT produces healthy local tracks even on the laggard cameras
(0394 has 11 tracks ≥120 frames; 0395 has 10). MCT then refuses to link
those tracks into global IDs:

| Cam | Long SCT tracks (≥120 fr) | MCT global IDs that include this cam |
|---|---|---|
| 0390 | 12 | 8 |
| 0391 | 12 | 5 |
| 0392 | 16 | 9 |
| 0393 | 16 | 5 |
| **0394** | **11** | **1** |
| **0395** | **10** | **1** |
| **0396** | **1** | **0** |

Likely causes (not investigated, separate workstream):
- ReID embeddings from oblique back-camera angles fail `sim_th=0.85`.
- World-coord disagreement (calibration drift) blows `distance_th=10m`.
- `short_track_th=120` filters out cam 0396 candidates (only 1 long track).

GT contains 25 unique people; MCT assigns 20 global IDs, only 9 cross-camera.

## Suggested follow-up workstreams

These are out of scope for this branch but are the natural next ~10x wins:

1. **Per-camera detector tuning** — lower `detect.conf_thresh` per camera, or
   try a larger backbone on cams 0395/0396. Target: get 0395 to ≥85% recall.
2. **Per-camera MCT thresholds** — relax `sim_th` and/or `distance_th` for
   back-camera pairs only, so their tracks can join the cross-camera graph.
3. **Recover the 16,604 dropped detections** in `aggregate_world_tracks` —
   these are dropped purely because MCT didn't assign a `GlobalOfflineID`.
   Anything that gets more tracks through MCT directly recovers DetA.

The 0.30 image↔world HOTA gap is now down to 0.20. Closing it further requires
attacking these bottlenecks, not the projection point.

## How to reproduce / rebuild

After this branch lands on `main`:

```bash
source .venv/bin/activate
python experiments/run.py ensure-baseline --force      # rebuilds outputs/baseline/
python experiments/run.py run ankle_footpoint_sweep    # solo ankle sweep
python experiments/run.py run world_smoother_sweep     # solo EMA sweep
python experiments/run.py run ankle_lower_x_smoother   # cross-product (winners)
python experiments/compare.py --sort-by mct_world.HOTA
```

The control variants (`bbox_bottom` and `none`) reproduce the exact pre-branch
metrics — verified against the pre-branch baseline by floating-point equality.
