# World-projection improvements: ankle footpoints + EMA smoother

**Date:** 2026-05-28
**Branch:** TBD (likely `feat/world-projection`)
**Status:** approved design, pending implementation plan

## Goal

Close the gap between image HOTA (0.758) and 3D-world HOTA (0.458) in the
v2-tuned baseline by attacking the two parts of the world-projection pipeline
that are demonstrably weak:

1. **Ankle-keypoint footpoints** — replace bbox-bottom-center with RTMPose
   ankle keypoints as the 2D point projected through the homography. Bbox-bottom
   is pixel-inaccurate when feet are occluded or bboxes are tilted; ankles are
   pixel-accurate when visible.
2. **Per-(frame, gid) world-coord EMA smoother** — apply temporal smoothing in
   `world_tracks.py` to remove per-frame projection noise before HOTA scoring.

Both improvements ship as **config-gated, experiment-pipeline variants** with
the default behavior identical to today's baseline. No upstream YACHIYO code is
patched.

## Baseline reference

| Metric                | Tuned v2 baseline | Target after these changes |
|-----------------------|-------------------|----------------------------|
| Image HOTA (combined) | 0.758             | unchanged                  |
| Image IDF1            | 0.750             | unchanged                  |
| Image MOTA            | 0.781             | unchanged                  |
| **3D world HOTA**     | **0.458**         | **+0.05 to +0.10 target**  |

(Image-side metrics are not expected to move because SCPT does not use
`WorldCoordinate` — see "Why this works" below.)

## Why this works — key upstream facts

The 2D→world projection happens inside upstream YACHIYO in two places, both
using bbox-bottom-center `((x1+x2)/2, y2)`:

- `external/.../tracking/src/utils.py:92` — initial `WorldCoordinate` at
  detection ingestion (the live one).
- `external/.../tracking/src/mcpt.py:710` — re-computed inside
  `measure_world_coordinate`, but written to the misspelled key
  `WoorldCoordinate` (typo upstream — dead key, never read).

Critically: **`scpt.py` does not reference `WorldCoordinate` at all** (verified
via grep). Only `mcpt.py` reads it (lines 457, 618, 842). This means we can
override `WorldCoordinate` **after SCT finishes and before MCT starts** and
capture the full downstream benefit (MCT clustering + final eval) at zero
upstream-patch cost.

Pose JSON (`Pose/<scene>/<cam>/<cam>_out_keypoint.json`) is structured as
`{frame_str: [{"bbox": [x1,y1,x2,y2,1.0], "keypoints": [[x,y,score]*17]}, ...]}`.
The implementation builds a per-camera lookup table keyed by
`(frame:int, (x1,y1,x2,y2):tuple[int,int,int,int])` — exactly the join key
documented in `aic24_nvidia/models/pose_rtmpose.py:22`. COCO ordering gives
`15=left_ankle`, `16=right_ankle`. The per-camera `calibration.json` already
contains the 3×3 `homography matrix` field.

**Camera naming note:** upstream SCT writes per-camera tracking JSONs with
3-digit numeric IDs (`camera001_tracking_results.json`), while pose / detect /
calibration use the NVIDIA 4-digit names (`camera_0390`). The implementation
must thread through the camera-name → numeric-id mapping that the adapter
already maintains; this is a join detail, not a design knob.

## Architecture

### Change #1 — ankle footpoints

New module: `aic24_nvidia/world_projection.py`. Pure function:

```python
def rewrite_world_coordinates(
    sct_scene_dir: Path,    # contains camera_NNNN_tracking_results.json
    pose_scene_dir: Path,   # contains camera_NNNN/camera_NNNN_out_keypoint.json
    calib_root: Path,       # Original/scene_001/camera_NNNN/calibration.json
    cameras: list[str],
    method: str,            # bbox_bottom | ankle_avg | ankle_lower | ankle_w_fallback
    ankle_min_conf: float,
) -> int:                   # returns number of detections rewritten
    ...
```

Behavior:
- `bbox_bottom`: no-op (returns 0, leaves files untouched — byte-identical to baseline).
- `ankle_avg`: for each detection, look up pose by `(frame, x1, y1, x2, y2)`;
  take **score-weighted** average of L/R ankle pixel coords:
  `x = (s_L*x_L + s_R*x_R) / (s_L + s_R)` (same for y), where `s` is the
  per-keypoint confidence score. Project the resulting `(x, y)` through the
  homography (same formula as `utils.py:170` — `inv(H) @ [x, y, 1]`); overwrite
  `WorldCoordinate` in-place. If both scores are zero, fall back to bbox_bottom.
- `ankle_lower`: pick the ankle with the **larger** pixel-y value. Image origin
  is top-left, so larger y = lower in the image = closer to the ground plane =
  planted foot. Project that single keypoint.
- `ankle_w_fallback`: use `ankle_avg` when both ankle scores ≥ `ankle_min_conf`;
  otherwise fall back to `bbox_bottom` for that detection.

Edge cases:
- Pose join miss (detection without a corresponding pose entry): fall back to
  `bbox_bottom`. Count and log.
- Missing calibration: error (baseline already requires calibration).
- NaN/inf after projection: skip the rewrite for that detection.

Integration point: `aic24_nvidia/stages/mct.py:51`, immediately after
`shutil.copytree(src_scene, dst_scene)` and before the upstream subprocess.
Gated on `cfg.world_projection.method != "bbox_bottom"`.

Config schema (added to `configs/baseline.yaml`, default = no-op):

```yaml
world_projection:
  method: bbox_bottom        # bbox_bottom | ankle_avg | ankle_lower | ankle_w_fallback
  ankle_min_conf: 0.3
```

### Change #2 — EMA smoother

Modify `aic24_nvidia/world_tracks.py:aggregate_world_tracks` (or add a sibling
function). After the per-(frame, gid) rows are produced, optionally smooth them
per-gid timeseries:

```python
# Per gid, sort by frame, EMA:
#   s_t = alpha * obs_t + (1 - alpha) * s_{t-1}
# First observation = obs.
```

Pure post-process on the (frame, gid, x, y) rows; tracking decisions are
already made.

Config schema:

```yaml
world_smoothing:
  method: none               # none | ema
  ema_alpha: 0.3
```

(Kalman explicitly out of scope for v1 per design decision — add only if EMA
proves out.)

## Experiment registry additions

Two new experiments in `experiments/registry.yaml`:

```yaml
- id: ankle_footpoint_sweep
  description: "Use RTMPose ankle keypoints instead of bbox-bottom-center for world projection."
  hypothesis: "Ankle footpoints reduce 2D->world projection error, raising 3D-world HOTA."
  base_config: configs/baseline.yaml
  rerun_from: mct
  variants:
    - name: "bbox_bottom"          # = baseline (control)
      overrides:
        world_projection:
          method: bbox_bottom
    - name: "ankle_avg"
      overrides:
        world_projection:
          method: ankle_avg
    - name: "ankle_lower"
      overrides:
        world_projection:
          method: ankle_lower
    - name: "ankle_w_fallback"
      overrides:
        world_projection:
          method: ankle_w_fallback
          ankle_min_conf: 0.3      # explicit; matches the baseline default

- id: world_smoother_sweep
  description: "EMA temporal smoothing on per-(frame, gid) world coords (eval-only)."
  hypothesis: "Smoothing removes per-frame projection noise, raising world HOTA / IDF1."
  base_config: configs/baseline.yaml
  rerun_from: evaluate
  variants:
    - name: "none"                 # = baseline (control)
      overrides:
        world_smoothing:
          method: none
    - name: "ema_0.3"
      overrides:
        world_smoothing:
          method: ema
          ema_alpha: 0.3
    - name: "ema_0.5"
      overrides:
        world_smoothing:
          method: ema
          ema_alpha: 0.5
    - name: "ema_0.7"
      overrides:
        world_smoothing:
          method: ema
          ema_alpha: 0.7
```

`rerun_from: mct` means each ankle variant reuses the baseline's
`adapt/frames/detect/reid/pose/sct/` outputs via symlinks; only MCT and
evaluate re-run (~5 min per variant).

`rerun_from: evaluate` means smoother variants only re-run `evaluate` (~30 s
per variant).

## Run order

1. `python experiments/run.py ensure-baseline` — materialise
   `outputs/baseline/` (one-time, ~45-60 min, full pipeline run).
2. `python experiments/run.py run ankle_footpoint_sweep` — 4 variants × ~5 min
   ≈ 20 min.
3. `python experiments/run.py run world_smoother_sweep` — 4 variants × ~30 s
   ≈ 2 min.
4. `python experiments/compare.py --sort-by mct_world.HOTA` — read the table.
5. **Follow-up cross-product** (only if both improvements show positive
   deltas): one combined experiment that uses the best ankle method and best
   EMA alpha together — verifies they compose additively.

## Tests

### Unit (no GPU)

- `tests/unit/test_world_projection.py`
  - Synthetic SCT JSON (1 cam, 2 detections), synthetic pose JSON with known
    ankle positions, identity homography. Assert each method produces the
    expected world coords.
  - `bbox_bottom` mode: assert output JSON is byte-identical to input
    (no-op contract).
  - Pose-join miss: assert fallback to bbox_bottom + dropped-count logged.
  - Both ankle scores below `ankle_min_conf` in `ankle_w_fallback`: assert
    fallback to bbox_bottom.
- `tests/unit/test_world_smoother.py`
  - EMA on a known input series (step function, ramp, single point) —
    assert exact output values.
  - Per-gid isolation: rows for different gids do not influence each other.
  - `method: none`: assert byte-identical to input.

### Integration

- Existing `tests/integration/test_tiny_scene.py` must still pass with default
  config (proves no behavior change when both knobs are at defaults).

## Risk register

| Risk | Mitigation |
|---|---|
| Ankle keypoints are systematically biased (RTMPose ankle tends to be slightly above the actual contact point) | The sweep itself tests this — if all ankle variants underperform `bbox_bottom`, we revert. The control variant is in the sweep deliberately. |
| Pose join miss rate is high (detections without pose entries) | `ankle_w_fallback` is robust by construction. The unit test verifies fallback path. We will log the miss rate per camera. |
| Calibration error dominates and projection-point choice is noise | Lower-HOTA cameras (0395, 0396 in `v2_solider`) may have weak calibration. Per-camera world-HOTA breakdown will surface this; out of scope to fix here. |
| EMA introduces lag at track-start and track-end | Acceptable given the 30 s clip and the HOTA matching gate at 1.0 m. If problematic, future work can use symmetric/forward-backward smoothing. |

## Out of scope (explicitly)

- Kalman smoother (added only if EMA underperforms or the user requests).
- Per-camera detector tuning (the 0395/0396 IDR collapse is a separate
  workstream).
- Fixing the upstream `WoorldCoordinate` typo or removing the dead
  `measure_world_coordinate` re-computation.
- ReID fine-tuning (separate workstream, paused per current priority order).
- Cross-product ankle × smoother experiment (will be a follow-up only if both
  sweeps show positive deltas).
