# aic24-nvidia

Thin Python orchestrator that runs the **unmodified-as-possible** `riips/AIC24_Track1_YACHIYO_RIIPS`
multi-camera people-tracking pipeline on **NVIDIA PhysicalAI-SmartSpaces MTMC_Tracking_2024** data,
with per-stage manifests, TrackEval metrics, and a Streamlit dashboard.

Sibling project: `../aic23-nvidia/` (same idea, different upstream pipeline). Mirror its structure.

GitHub: https://github.com/harshbhatt-awl/aic24-nvidia

## Pipeline (8 stages)

`adapt → frames → detect → reid → pose → sct → mct → evaluate`

Each stage module is `aic24_nvidia/stages/<name>.py`, exposes `run(cfg, run_dir, run_id)`, writes to
`outputs/<run_id>/<stage>/` atomically (via `stages/base.py:atomic_stage`), and records a
`manifest.json`. Stages are manifest-gated (skip if done, refuse if upstream missing) and chained by
`pipeline.py`.

The orchestrator shells into the upstream repo + sibling repos under `external/` and harvests their
outputs into our run dir via **symlinks** (upstream honours no env-var path overrides). Upstream writes
to `external/{Original,Detection,EmbedFeature,Pose,Tracking}/`, which are symlinked to
`outputs/<run_id>/<stage>/` right before each stage runs.

## ⚠️ Dual-venv setup (the non-obvious part)

- **`.venv`** (Python 3.14, torch 2.12+cu130) — runs everything **except** pose.
- **`.venv-pose`** (Python 3.10, torch 1.13.1+cu117, mmcv-full 1.7.0, mmpose 0.29.0, mmdet 2.28.2)
  — runs **only** the pose stage. mmpose 0.x will not install on Python 3.14.

`stages/pose.py` auto-invokes `.venv-pose/bin/python`. Everything else uses `.venv`.

Created with `uv`:
```bash
uv python install 3.10
uv venv --python 3.10 .venv-pose
.venv-pose/bin/python -m ensurepip --upgrade
.venv-pose/bin/python -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
.venv-pose/bin/python -m pip install openmim 'numpy<2'
.venv-pose/bin/python -m mim install mmcv-full==1.7.0
.venv-pose/bin/python -m pip install --no-deps mmpose==0.29.0
.venv-pose/bin/python -m mim install mmdet==2.28.2
.venv-pose/bin/python -m pip install 'numpy<2'   # mim keeps pulling numpy 2.x back
```

## Running

```bash
source .venv/bin/activate
python pipeline.py bootstrap                       # clone BoT-SORT, deep-person-reid, mmpose siblings + copy injected files
python pipeline.py all   --config configs/warehouse_001_30s.yaml
python pipeline.py <stage> --config ... [--run-id <id>] [--force]
python pipeline.py viz   --config ... --stage {detect,sct,mct}
python pipeline.py dashboard --port 8501           # Streamlit (read-only viewer)
```

`detect` needs `external/BoT-SORT/bytetrack_x_mot17.pth.tar` (YOLOX-x, ~793 MB, from the ByteTrack
Google Drive: `gdown 1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5 -O external/BoT-SORT/bytetrack_x_mot17.pth.tar`).
OSNet (ReID) and HRNet (pose) checkpoints auto-download on first run.

Data lives at `data/nvidia_mtmc_2024/` (symlinked to `../aic23-nvidia/data/nvidia_mtmc_2024`). The real
scene is `MTMC_Tracking_2024/val/scene_044/`; `Warehouse_001/` is a flattened symlink view with
`videos/camera_0390.mp4`..`camera_0396.mp4` (7 cameras), `calibration.json`, `ground_truth.json`.

## Hardware

Verified on a 6 GB RTX 3050. Models are loaded sequentially per stage so peak VRAM stays ~3-4 GB.
Full 30s run ≈ 1 hour: detect ~26 min, reid ~8 min, pose ~15 min, the rest seconds.

## Gotchas / known issues

- **Hyperparameters in `configs/*.yaml` are recorded in manifests but NOT propagated to upstream** —
  upstream hardcodes them. Tuning is v2 work.
- **Manifest paths capture the `.tmp` dir** (written before the atomic rename). When a later stage reads
  an upstream manifest's `outputs` paths, they may point at `<stage>.tmp/...` which no longer exists.
  Workaround so far: sed-replace `<stage>.tmp` → `<stage>` in the manifests. **Proper fix needed:**
  rewrite the manifest after the rename in `atomic_stage`.
- **MCT depends on real pose data** AND on per-camera `Original/scene_NNN/camera_NNNN/calibration.json`
  containing `"camera projection matrix"` (3×4 K[R|t]) and `"homography matrix"` (3×3 world→image for
  Z=0). The adapter must write these; SCT reads them to populate `WorldCoordinate`, which MCT requires.
- **MCT/eval only cover per-camera SCT metrics in v1.** MCT HOTA/IDF1 needs a TrackEval dataset adapter
  for 3D world coords — not done.
- Detection hardcodes 1920×1080. NVIDIA Warehouse footage is 1080p, so fine here.

## Upstream patches applied (don't revert)

These live under `external/` (gitignored) and in our package. They fix import/runtime errors and
short-clip crashes — none change algorithm behaviour:

1. `external/BoT-SORT/fast_reid/.../testing.py` & `data/build.py`: `from collections import Mapping`
   → `collections.abc`; `from torch._six import string_classes` → `string_classes = (str, bytes)`.
2. `external/AIC24_Track1_YACHIYO_RIIPS/tracking/infer.py`: add
   `multiprocessing.set_start_method("fork", force=True)` — Python 3.14 defaults to spawn, which breaks
   the global-inheritance pattern in `single_tracking`.
3. `external/AIC24_Track1_YACHIYO_RIIPS/tracking/src/mcpt.py`: `assign_global_id` early-returns when no
   assigned tracks (avoids `feature_stack.T` on None); `interpolate_tracklet` skips cameras with empty
   `unique_local_ids` (avoids `min([])`).
4. `external/TrackEval/`: cloned separately (pip package lacks runner scripts); patched `np.float/int/bool`
   → built-ins.
5. Our `stages/reid.py`: PYTHONPATH override (deep-person-reid `setup.py` is broken).
6. Our adapter: consumes real NVIDIA schema `{cameras:{cam:{K,R,t}}, annotations:[{camera,frame,
   person_id,world_xy,bbox_2d}]}`, preserves real camera names, computes per-camera homography +
   projection matrices.

If `python pipeline.py bootstrap` re-clones the sibling repos, patches 1-4 must be re-applied.

## Tests

`pytest tests/unit/` (39 tests, no GPU). Integration test `tests/integration/test_tiny_scene.py` runs the
adapter on a synthetic 2-camera fixture; the full-pipeline path is skipped (needs GPU + siblings).
