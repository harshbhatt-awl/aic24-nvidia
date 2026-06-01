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

The stage graph is a **single source of truth** in `aic24_nvidia/registry.py` — an ordered `StageSpec`
list (`name, dir_name, upstream, run, wiring`) that both `pipeline.py` and `experiments/_lib.py`
consume (no more parallel dicts). Each stage also declares its external-symlink
`WIRING(run_dir, cfg, output_dir)` (output exposure + input cross-wiring); `atomic_stage` applies it
pre-run (→`.tmp`) and post-promotion (→final), and `experiments/_lib.py:prime_external_symlinks`
replays the same wiring for cache-reused stages.

The orchestrator shells into the upstream repo + sibling repos under `external/` and harvests their
outputs into our run dir via **symlinks** (upstream honours no env-var path overrides). Upstream writes
to `external/{Original,Detection,EmbedFeature,Pose,Tracking}/`, which are symlinked to
`outputs/<run_id>/<stage>/` right before each stage runs.

## Models (the learned components)

All three model stages run **in-process in the single `.venv`** (Python 3.14, torch 2.12+cu130) via
adapters under `aic24_nvidia/models/` that write byte-compatible output for the untouched SCT/MCT:

- **detect** → **YOLO11-x** (`ultralytics`, `aic24_nvidia/models/detect_yolo.py`); person class, GPU.
- **reid** → **SOLIDER Swin-Small** (timm-based, vendored in `aic24_nvidia/models/solider/`,
  `reid_solider.py`); weights at `weights/solider_swin_small.pth`; 768-d embeddings, GPU.
- **pose** → **RTMPose-l** via `rtmlib` (ONNX, `pose_rtmpose.py`); 17 COCO keypoints. ONNXRuntime has
  no CUDA provider on the cu130 stack, so pose runs on **CPU** (~11 min for the 30s clip — fine).

> Backends are swappable by config: `detect.model_name` / `reid.model_name` /
> `pose.model_name` (default to the v3 stack) resolve through
> `aic24_nvidia/models/registry.py` to a `*Backend` class implementing
> `load/infer/teardown`. The per-kind orchestrators (`run_detection`/`run_reid`/
> `run_pose`) keep the byte-compatible YACHIYO writers. `cfg.<stage>.weights`
> overrides the checkpoint (resolved against `weights_root`). `load_config`
> rejects an unknown `model_name`.

> Historical: pose used to need a separate `.venv-pose` (mmpose 0.29 / Python 3.10) — **removed**; rtmlib
> is pure ONNX so pose runs in the main `.venv`. detect dropped BoT-SORT/YOLOX; reid dropped
> deep-person-reid/OSNet. Each adapter releases GPU memory (`del`+`empty_cache`) after its stage since
> stages now run in one process rather than per-stage subprocesses.

## Running

```bash
source .venv/bin/activate
python pipeline.py bootstrap                       # clone YACHIYO + TrackEval siblings (detect/reid/pose models are pip/vendored, no sibling needed)
python pipeline.py all   --config configs/baseline.yaml
python pipeline.py <stage> --config ... [--run-id <id>] [--force]
python pipeline.py viz   --config ... --stage {detect,sct,mct}
python pipeline.py dashboard --port 8501           # Streamlit (read-only viewer)
python pipeline.py menu                            # interactive operations hub (needs `pip install -e ".[hub]"`)
```

### Renting a GPU box (remote training)

The 6 GB local card handles a 30s clip (~45 min); for faster iteration or reid
fine-tuning, rent a bigger GPU (RTX 4090 / A100). `scripts/remote_setup.sh` takes
a freshly-rented box from `git clone` to pipeline-ready in one command:
GPU/driver preflight (warns if the driver is < 580 — too old for torch cu130),
system deps (ffmpeg, rclone, tmux, …), a Python 3.14 `.venv` via `uv` with
**torch installed from the cu130 index first** (the lock pins `torch` bare, so a
plain `pip install -r` would grab the wrong wheel), `bootstrap_external.sh`, and
a best-effort `rclone` pull of the dataset + SOLIDER weight (the laptop's
`data/` + `weights/` symlinks don't exist on a fresh box).

```bash
# on the rented box (clone first — the script lives in the repo):
git clone https://github.com/harshbhatt-awl/aic24-nvidia && cd aic24-nvidia
scripts/remote_setup.sh         # --no-apt / --skip-data / --skip-weights / --remote NAME
```

For a hands-free data/weight pull, stage them on your rclone remote once
(`rclone copy data/nvidia_mtmc_2024 onedrive:aic24/data/nvidia_mtmc_2024`,
`rclone copy weights/solider_swin_small.pth onedrive:aic24/weights/`). Connect
from your laptop via VS Code Remote-SSH; `ssh -L 8501:localhost:8501` forwards
the Streamlit dashboard. Run long jobs under `tmux` so an SSH drop doesn't kill
them.

### Experiment harness (`experiments/`)

A/B variants against a locked baseline without re-running slow stages:

```bash
python experiments/run.py ensure-baseline                       # build outputs/baseline/ once (~1h)
python experiments/run.py list                                  # show defined experiments
python experiments/run.py run eps_mcpt_sweep                    # run all variants of one experiment
python experiments/run.py run eps_mcpt_sweep --variant 0.30     # single variant
python experiments/compare.py [--sort-by mct_world.HOTA]        # results table vs baseline
```

`configs/baseline.yaml` is the locked reference (v2 model stack + sweep-tuned tracker).
`experiments/registry.yaml` defines experiments; each variant is `base_config + overrides`.
Upstream stages (before `rerun_from`) are inherited from `outputs/baseline/` via symlinks, so
e.g. an `epsilon_mcpt` sweep takes seconds per variant (only SCT/MCT/evaluate re-run).
See `experiments/README.md` for full details and the worktree-vs-harness decision rule.

`detect` (YOLO11-x, `yolo11x.pt`) and `pose` (RTMPose-l ONNX) checkpoints auto-download on first run.
`reid` needs the SOLIDER Swin-Small weights at `weights/solider_swin_small.pth` (from the
`tinyvision/SOLIDER-REID` release).

Data lives at `data/nvidia_mtmc_2024/` (symlinked to `../aic23-nvidia/data/nvidia_mtmc_2024`). The real
scene is `MTMC_Tracking_2024/val/scene_044/`; `Warehouse_001/` is a flattened symlink view with
`videos/camera_0390.mp4`..`camera_0396.mp4` (7 cameras), `calibration.json`, `ground_truth.json`.

## Hardware

Verified on a 6 GB RTX 3050. Each adapter loads its model then frees GPU memory before the next stage,
so peak VRAM stays ~3-4 GB. Full 30s run ≈ 45 min: detect ~31 min (GPU), reid ~10 min (GPU),
pose ~11 min (CPU — no ONNX CUDA provider), the rest seconds.

## Gotchas / known issues

- **Hyperparameters ARE now propagated.** `tracking_params:` in `configs/*.yaml` (real YACHIYO keys:
  `epsilon_scpt`, `epsilon_mcpt`, `short_track_th`, `distance_th`, `sim_th`, `keypoint_condition_th`,
  `replace_similarity_by_wcoordinate`, `distance_type`, `delete_gid_th`, `time_period`,
  `replace_value`) are written to
  `external/AIC24_Track1_YACHIYO_RIIPS/tracking/config/parameters_per_scene.py` by
  `aic24_nvidia/tracking_params.py` before each SCT/MCT run; `infer.py` reads that file natively.
  `mct.hard_world_gate: true` pushes `replace_value` very negative for a hard world-distance
  cannot-link.
- **Manifest paths capture the `.tmp` dir** (written before the atomic rename). When a later stage reads
  an upstream manifest's `outputs` paths, they may point at `<stage>.tmp/...` which no longer exists.
  Workaround so far: sed-replace `<stage>.tmp` → `<stage>` in the manifests. **Proper fix needed:**
  rewrite the manifest after the rename in `atomic_stage`.
- **MCT depends on real pose data** AND on per-camera `Original/scene_NNN/camera_NNNN/calibration.json`
  containing `"camera projection matrix"` (3×4 K[R|t]) and `"homography matrix"` (3×3 world→image for
  Z=0). The adapter must write these; SCT reads them to populate `WorldCoordinate`, which MCT requires.
- **MCT 3D-world eval IS now done.** `evaluate` emits a scene-level `mct_world` block in
  `metrics.json` (HOTA/DetA/AssA/IDF1/MOTA), matching predicted vs GT world points by Euclidean
  distance with a gate of `eval.world_d_max` metres (default 1.0 m). Predictions come from averaging
  per-camera `WorldCoordinate` per `(frame, global_id)` (`aic24_nvidia/world_tracks.py`); scoring
  uses a custom TrackEval `NvidiaMTMCWorld` dataset adapter (`aic24_nvidia/world_metrics.py`). GT is
  the adapter's `adapted/scene_001_gt_world.txt`.
- Detection hardcodes 1920×1080. NVIDIA Warehouse footage is 1080p, so fine here.

## Upstream patches applied (don't revert)

These live under `external/` (gitignored) and in our package. They fix import/runtime errors and
short-clip crashes — none change algorithm behaviour:

1. ~~BoT-SORT fast_reid patches~~ — **obsolete** (detect uses YOLO11-x; BoT-SORT no longer used).
2. `external/AIC24_Track1_YACHIYO_RIIPS/tracking/infer.py`: add
   `multiprocessing.set_start_method("fork", force=True)` — Python 3.14 defaults to spawn, which breaks
   the global-inheritance pattern in `single_tracking`.
3. `external/AIC24_Track1_YACHIYO_RIIPS/tracking/src/mcpt.py`: `assign_global_id` early-returns when no
   assigned tracks (avoids `feature_stack.T` on None); `interpolate_tracklet` skips cameras with empty
   `unique_local_ids` (avoids `min([])`).
4. `external/TrackEval/`: cloned separately (pip package lacks runner scripts); patched `np.float/int/bool`
   → built-ins.
5. ~~reid PYTHONPATH override~~ — **obsolete** (reid uses SOLIDER; deep-person-reid no longer used).
6. Our adapter: consumes real NVIDIA schema `{cameras:{cam:{K,R,t}}, annotations:[{camera,frame,
   person_id,world_xy,bbox_2d}]}`, preserves real camera names, and writes per-camera
   `Original/scene_NNN/camera_NNNN/calibration.json` (3×4 projection + 3×3 homography) that SCT reads
   to populate `WorldCoordinate` (required by MCT).

Patches 2-4 are now **vendored** under `patches/{yachiyo,trackeval}.patch` and applied automatically by
`scripts/bootstrap_external.sh`, which pins the upstream commits (YACHIYO `f881fe0`, TrackEval
`12c8791`) and runs `scripts/verify_patches.sh` afterward (text-independent `git diff --quiet` check +
`set_start_method` sentinel). A re-clone no longer silently loses them. To regenerate a patch after
editing the upstream working tree: `git -C external/<repo> diff -- <files> > patches/<repo>.patch`
(for YACHIYO use only `tracking/infer.py tracking/src/mcpt.py` — the other modified tracked files are
runtime-generated by our pipeline, not patches).

## Tests

`pytest tests/unit/` (121 tests, no GPU). Integration test `tests/integration/test_tiny_scene.py` runs the
adapter on a synthetic 2-camera fixture; the full-pipeline path is skipped (needs GPU + siblings).

`tests/unit/test_pipeline_registry_consistency.py` guards the four stage dicts in `pipeline.py`
(`STAGE_RUNNERS`/`ORDER`/`UPSTREAM_OF`/`STAGE_DIR_NAME`) and the duplicate `STAGES`/`STAGE_DIR` in
`experiments/_lib.py` against drift until a real `StageRegistry` replaces them.

CI: `.github/workflows/tests.yml` runs the unit suite; `lint.yml` runs `ruff check` (high-signal rules,
blocking) + `mypy` (informational). Dev tools: `pip install -e ".[dev]"`. Deps are pinned in
`pyproject.toml`; `requirements.lock` is the exact verified environment (Python 3.14, torch 2.12+cu130).
