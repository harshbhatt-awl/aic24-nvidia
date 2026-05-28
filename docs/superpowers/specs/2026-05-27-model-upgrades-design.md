# Model upgrades: detect / reid / pose

**Date:** 2026-05-27
**Branch:** `feat/model-upgrades`
**Status:** approved design, pending implementation plan

## Goal

Modernize the three model-bearing stages of the pipeline while leaving the
downstream tracking (`sct`, `mct`) and `evaluate` stages **100% untouched**.
Each new model is wrapped in an adapter that reproduces the existing on-disk
output format **byte-for-byte**, so YACHIYO's consumers cannot tell the model
changed. Also reclaim disk space (system is at 94% / 10 G free).

### Model decisions (locked)

| Stage  | Old model                              | New model                        |
|--------|----------------------------------------|----------------------------------|
| detect | YOLOX-X (BoT-SORT, MOT17 weights)      | **YOLO11-x** (ultralytics)       |
| reid   | OSNet-x1.0 (512-dim)                    | **SOLIDER Swin-Small** (~768-dim)|
| pose   | HRNet-W48 top-down (mmpose 0.29)        | **RTMPose-l** via **rtmlib** (ONNX) |

`sct`, `mct`, `evaluate`, `adapt`, `frames` — no model, unchanged.

## Why Approach B (standalone adapters in our package)

The three upstream injected scripts each do two jobs: *run a model* and
*write a format*. The injected sources are **upstream YACHIYO's own files**
(`external/AIC24_Track1_YACHIYO_RIIPS/{detector,embedder}/`, plus the mmpose
demo), copied into the sibling repos by bootstrap. `external/` is gitignored
and recreated on re-bootstrap.

Therefore we do **not** edit upstream scripts. Instead we add three adapter
modules to our **version-controlled, unit-testable** package and point the
stage modules at them. This also lets:

- `detect` drop its dependency on BoT-SORT/YOLOX entirely.
- `pose` drop mmpose + the entire `.venv-pose` (runs in main `.venv`).

## Architecture

New modules under `aic24_nvidia/models/`:

| Module                         | Model                          | Writes (exact format below)                                   |
|--------------------------------|--------------------------------|---------------------------------------------------------------|
| `models/detect_yolo.py`        | YOLO11-x, person class, 1920×1080 | `Detection/scene_001/camera_NNNN.{txt,json}`               |
| `models/reid_solider.py`       | SOLIDER Swin-Small             | `EmbedFeature/scene_001/camera_NNNN/feature_*.npy` + updates `.json` `NpyPath` |
| `models/pose_rtmpose.py`       | RTMPose-l (rtmlib, ONNX)       | `Pose/scene_001/camera_NNNN/camera_NNNN_out_keypoint.json`    |

Stage module changes (`stages/detect.py`, `reid.py`, `pose.py`):

- Replace the `subprocess.run([... external script ...])` block with a call into
  the new adapter.
- **Keep unchanged:** the `external/{Detection,EmbedFeature,Pose}` symlink trick,
  `atomic_stage`, manifest writing, per-camera validation, `assert_vram_free`.
- Update manifest `params` to record the new model name.
- `pose.py`: delete the `.venv-pose` python invocation; run in main `.venv`.

The adapters are pure functions of (scene dir, camera list, config) → files on
disk, so they are unit-testable against golden fixtures without GPU where
possible (CPU fallback for tiny inputs).

## Byte-compatible output contracts (must reproduce exactly)

### 1. Detection — `Detection/scene_001/camera_NNNN.txt` + `.json`

`.txt`, one line per detection, comma-separated, **no track id**:

```
cam,frame_id,cls,x1,y1,x2,y2,score
```

- `cam` = camera name string (e.g. `camera_0390`)
- `frame_id` = 1-indexed int
- `cls` = always `1` (person)
- `x1,y1,x2,y2` = **xyxy**, top-left→bottom-right, integer, clamped to [0,1920]×[0,1080]
- `score` = float

`.json` = dict keyed by 8-digit zero-padded detection index (`"00000000"`, …),
each value:

```json
{"Frame": <int>, "ImgPath": "<rel path>", "NpyPath": "",
 "Coordinate": {"x1": <int>, "y1": <int>, "x2": <int>, "y2": <int>},
 "ClusterID": null, "OfflineID": null}
```

### 2. ReID — `EmbedFeature/scene_001/camera_NNNN/feature_*.npy`

Filename convention (note coord order `x1_x2_y1_y2`, and `conf` has the decimal
point removed):

```
feature_<frame>_<u_num>_<x1>_<x2>_<y1>_<y2>_<conf>.npy
```

- `u_num` = global sequential detection index across all frames of the camera
- each `.npy` = a **single** float32 vector (dim = backbone output; OSNet was 512,
  SOLIDER Swin-S is ~768)
- the adapter reads the Detection `.txt` line-by-line (same iteration order used
  to assign `u_num`), crops the frame at `(x1,y1,x2,y2)`, embeds, and also sets
  the matching `.json` entry's `NpyPath`.

Consumed by YACHIYO `tracking/src/utils.py:84` (parses the filename) and
`tracking/src/scpt.py:324` (`np.load(NpyPath)`).

### 3. Pose — `Pose/scene_001/camera_NNNN/camera_NNNN_out_keypoint.json`

Dict keyed by `frame_id` → list of per-person dicts:

```json
{"<frame_id>": [
  {"bbox": [x1, y1, x2, y2, 1.0],
   "keypoints": [[x, y, score], ... 17 COCO keypoints ...]}
]}
```

- COCO-17 ordering (0=nose … 16=R_ankle) — must match what
  `tracking/src/pose.py` indexes.
- The adapter takes detection bboxes from the Detection `.txt` and runs RTMPose
  top-down per bbox.
- YACHIYO matches pose→detection by key `f"{frame}_{x1}_{y1}_{x2}_{y2}"`
  (`tracking/src/pose.py:116`), so bbox integers must match the detection bbox
  integers exactly.

## Dependencies & weights

Add to `pyproject.toml` dependencies:

- `ultralytics` — YOLO11-x weights (`yolo11x.pt`, ~110 MB) auto-download to cache.
- `rtmlib` + `onnxruntime-gpu` — RTMPose-l ONNX auto-downloads.
- SOLIDER deps as needed (timm/swin); SOLIDER Swin-Small ReID weights downloaded
  manually from the `tinyvision/SOLIDER-REID` release into `weights/`.

`detect` no longer needs BoT-SORT; `pose` no longer needs mmpose. Bootstrap may
stop cloning those siblings (follow-on cleanup, not required for correctness).

## Space reclamation (chosen ordering)

1. **Now**, gated on a ~10-second rtmlib RTMPose-l sanity inference succeeding in
   the main `.venv`: delete `.venv-pose` (**−4.1 G**). We never strand the pose
   stage — deletion happens only after confirming the replacement imports + runs.
2. **After** a full end-to-end run verifies all three new models produce valid
   SCT/MCT output and metrics: delete `external/BoT-SORT/bytetrack_x_mot17.pth.tar`
   (**−757 M**) and old OSNet/HRNet checkpoint caches.

## Verification gates

1. **Unit tests** per adapter: assert output matches golden-format fixtures
   (exact `.txt` columns, `.npy` filename pattern + dtype/shape, pose JSON schema).
   Use the existing synthetic 2-camera integration fixture where possible.
2. **rtmlib sanity** inference in `.venv` → unblocks `.venv-pose` deletion.
3. **End-to-end smoke run** on `Warehouse_001_30s`: all stages green, SCT/MCT
   produce tracks, evaluate emits metrics → unblocks old-weight deletion.

## Known risks

- **SOLIDER embedding dim** (512→~768): cosine similarity is dim-agnostic, but
  verify YACHIYO does not hardcode 512 anywhere; re-check SCT/MCT thresholds.
- **RTMPose keypoint ordering**: confirm COCO-17 order matches YACHIYO
  `src/pose.py` indices (nose/eyes/.../ankles).
- **Detection distribution shift**: YOLO11 person-class conf/NMS differ from the
  MOT17-trained YOLOX weights → detection counts shift, rippling into tracking.
  `detect.conf_thresh`/`nms_iou` in config become live, propagated knobs (unlike
  the old hardcoded-upstream situation).
- **Frame indexing**: detection `frame_id` is 1-indexed; pose JSON is keyed by
  frame as emitted by the demo. Adapters must preserve whatever indexing YACHIYO
  expects (verify against current outputs before swap).

## Out of scope

- Tuning the new models' hyperparameters beyond making the pipeline run.
- MCT 3D-world TrackEval adapter (tracked separately).
- Removing BoT-SORT/mmpose sibling clones from bootstrap (optional follow-on).
