# ReID fine-tune: SOLIDER Swin-Small partial backbone adaptation for Warehouse cameras

**Date:** 2026-05-28
**Branch:** TBD (suggest `feat/reid-finetune` once execution starts)
**Status:** approved design, pending Colab execution in a future session
**Reads against:** `pipeline-state.md` memory, `docs/superpowers/notes/2026-05-28-world-projection-results.md`

## Goal

Fix the back-camera (0394 / 0395 / 0396) cross-camera linking failure in the v3
pipeline by adapting the SOLIDER Swin-Small ReID backbone to warehouse camera
geometry. Currently MCT assigns global IDs spanning back+front cameras for
**0 / 1 / 1** of cams 0394 / 0395 / 0396; the goal is **≥ 5 each**, with a
downstream world HOTA improvement of **≥ +0.01** over the v3 baseline (0.5055).

This is workstream #6 from the original session ranking — held for last because
it is the highest-cost item with the most failure modes. Execute only after
the detector-recall workstream (`detector-recall-workstream-paused`) and the
finer tracker tuning have been explored.

## Why fine-tune (vs other workstreams)

The same back cameras that fail MCT linking also have low YOLO detector
recall (0395: 62%, 0396: 35%). Two related but distinct workstreams:

| Workstream | What it fixes | Status |
|---|---|---|
| Detector recall (paused) | Cams 0395/0396 not producing detections at all | Phase 0 design approved, awaiting variant scope |
| **ReID fine-tune (this)** | **Cams 0394-0396 producing detections that fail cross-camera ReID matching** | This doc |
| Per-camera tracker thresholds | Existing detections rejected by MCPT distance / sim gates | Not yet scoped |

ReID fine-tune attacks the *quality* of embeddings produced for oblique views.
Even if the detector workstream finds more people on those cams, those people
will still fail to link cross-camera unless their embeddings are usable.

The diagnostic supporting this (in `docs/superpowers/notes/2026-05-28-world-
projection-results.md`):

- Cam 0394 produces **11 SCT tracks ≥120 frames**, MCT links **1** into a
  global ID. The other 10 die single-camera.
- Cam 0395 produces **10 long SCT tracks**, MCT links **1**.
- Cam 0396 produces **1 long SCT track**, MCT links **0**.

The current cosine margin on Warehouse_001 GT crops is same-id ≈ 0.38 vs
diff-id ≈ 0.16 (margin 0.22). This is discriminative but only marginally —
oblique-view embeddings sit near the decision boundary. Pushing the margin
to > 0.40 should let the MCPT sim_th = 0.85 gate start linking back-cam tracks.

## Environment

**Training:** Google Colab. Free T4 (12-15 GB) is the minimum; Colab Pro V100
or A100 (16-40 GB) is preferred for shorter iteration.

**Source repo:** This local checkout. Colab session pulls the SOLIDER vendored
files (`aic24_nvidia/models/solider/`) and the base checkpoint to bootstrap.

**Local integration:** This repo consumes only the final `.pth` checkpoint.
No code changes required beyond an optional config knob (described below).

## Design decisions (with reasoning)

### Decision 1: Partial fine-tune (Swin stage 4 + BN-neck)

Three intensity options were considered:

| Option | What's trainable | Compute | Expected | Risk |
|---|---|---|---|---|
| Head-only | New 768→K projection head, frozen backbone | ~30 min T4 | Near-zero on this problem | Lowest |
| **Partial** ⭐ | **Swin stage 4 + BN-neck + ID head** | **1-3 h T4 / 30-60 min A100** | **Moderate (target +0.05-0.10 cosine margin)** | **Low** |
| Full backbone | All Swin layers at low LR | 3-8 h T4 | High ceiling | High — MSMT17 forgetting |

**Why partial wins for this specific problem:**

The bottleneck is *camera-angle distribution shift*, not generic ReID quality.
MSMT17 (the pretraining data) is mostly street-level surveillance cameras; our
back cameras are oblique/overhead views. The backbone's high-level features
need to learn warehouse-camera-angle composition, but the low-level features
(edges, body parts, textures) are angle-invariant and don't need updating.

Concretely, by Swin layer:

| Layer | Frozen? | Reason |
|---|---|---|
| Patch embedding | ✅ frozen | Pixel-level features generalize trivially |
| Stage 1 (early features) | ✅ frozen | Edge / texture detectors are angle-invariant |
| Stage 2 (mid features) | ✅ frozen | Body-part detectors are angle-invariant |
| Stage 3 (mid-high features) | ✅ frozen | Person shape representations are angle-invariant enough |
| **Stage 4 (high-level features)** | 🔓 **trainable** | **Where camera-angle composition lives — the actual bottleneck** |
| BN-neck | 🔓 trainable | Normalizes for warehouse distribution statistics |
| ID classifier head | 🔓 trainable | Used only during training; discarded at inference |

**Why not head-only:** A head can only re-weight features the backbone already
extracts. If stage 4 never learned oblique-warehouse representations, no head
fixes it. Head-only typically buys 1-3% on in-distribution data; for
distribution shift it's near-zero.

**Why not full backbone:** With 50-200 unique IDs total (current local 25 +
downloaded scenes) and 1-3 GPU-hours of Colab, full fine-tuning catastrophically
forgets MSMT17 priors. Model gets very good at our warehouse people and bad
at everything else; pipeline numbers might briefly rise then collapse on any
new scene.

**Decision rule for re-evaluating intensity:** see Escalation Paths below.

### Decision 2: Use SOLIDER's own loss recipe

Loss: **ID cross-entropy + triplet (hard mining, margin 0.3) + center loss**.
This is the recipe SOLIDER's own checkpoint was trained with; same recipe means
the existing weights remain a well-conditioned starting point and gradient
directions are familiar to the loss surface.

Alternatives considered: SupCon (more modern, slightly stronger but adds
implementation complexity), ArcFace (overkill for this scale, harder to tune),
plain triplet (loses the regularization benefit of ID + center).

### Decision 3: Leave-scene_044-out validation

Train on all available Warehouse scenes EXCEPT scene_044. Reserve scene_044
entirely for evaluation. This means:

- scene_044 doubles as both the ReID test and the downstream pipeline test —
  no double-dipping concerns since the same scene serves both.
- Cross-view retrieval evaluation on scene_044's 25 IDs / 7 cameras measures
  exactly what the pipeline cares about.
- No leakage from training crops into the cosine-margin / pipeline-HOTA
  numbers we report at the end.

### Decision 4: Match SOLIDER's inference preprocessing

Input pipeline must match what the local `aic24_nvidia/models/solider/__init__.py`
expects at inference (or the fine-tuned weights will be unusable here):

- Image size: **384 × 128** (H × W)
- Normalization: **mean (0.5, 0.5, 0.5), std (0.5, 0.5, 0.5)** — NOT ImageNet
- `semantic_weight = 1.0`
- `NECK_FEAT = 'after'` (BN-centered output)

Training augmentation is layered on top: horizontal flip, color jitter, random
erasing. No resize-jitter beyond the fixed 384×128.

## Data sourcing

**Local state:** Only `scene_044` is downloaded (at
`data/nvidia_mtmc_2024/MTMC_Tracking_2024/val/scene_044/`). It is symlinked
from `../../aic23-nvidia/data/nvidia_mtmc_2024`. **All other scenes need to be
fetched in Colab.**

**Source:** NVIDIA AI City Challenge 2024 Track 1 — MTMC People Tracking
dataset. The release contains multiple Warehouse scenes split across `train/`,
`val/`, `test/`. Get the source URL from wherever scene_044 was originally
obtained (likely the AIC2024 organizers' page or a HuggingFace dataset
mirror).

If the official source is not easily accessible from Colab, alternatives:

1. **Upload local scene_044 to Drive + use that path in Colab.** Just gives us
   what we already have; doesn't expand training data. Falls back to
   single-scene training (acceptable but suboptimal).
2. **Use the related PhysicalAI-SmartSpaces HuggingFace dataset** (`nvidia/
   PhysicalAI-SmartSpaces-MTMC_Tracking_2024` or similar). Verify it contains
   the same scene format before relying on it.
3. **Augment from existing data only.** Heavy synthetic transforms (perspective
   warp, color jitter, simulated camera angles) on scene_044 crops. Lowest
   ceiling; only if (1) and (2) both fail.

**Target dataset size:** ~50,000-200,000 person crops across ~50-200 unique
IDs minimum. SOLIDER's MSMT17 training used 32k IDs, 126k images. Scaling
down by 100-1000x is workable for adapter-style partial fine-tuning.

## Data preparation (Colab notebook cells 1-2)

Per training scene:

1. **Parse `ground_truth.json`** → extract `{annotations: [{camera, frame,
   person_id, world_xy, bbox_2d}]}` rows.
2. **Decode video frames** for each (camera, frame) needed.
3. **Crop person bboxes** with a small padding (5-10 px). Save as JPEG.
4. **Subsample frames** to reduce dataset size: take every Nth frame (start
   with N=5 for ~6 fps at 30 fps source). Each person seen on each camera
   gets ~180 crops per scene per camera at N=5.

Build a single training manifest:

```csv
scene,camera,frame,person_id,crop_path
scene_xxx,camera_0390,12,3,/content/crops/scene_xxx_camera_0390_3_12.jpg
...
```

Split: `scene == "scene_044"` → val. Everything else → train.

## Training (Colab notebook cells 3-4)

### Model assembly

```python
# Pseudo-code; concrete code goes in the notebook.
from aic24_nvidia.models.solider import load_solider_swin_small
backbone = load_solider_swin_small("weights/solider_swin_small.pth")

# Freeze stages 1-3.
for name, p in backbone.named_parameters():
    if "stages.3" in name or "norm" in name or "bn_neck" in name:
        p.requires_grad = True
    else:
        p.requires_grad = False

# Add an ID classifier head used only at training (discarded after).
num_train_ids = len(set(train_manifest["person_id"]))
id_head = nn.Linear(768, num_train_ids)

# Center loss buffer.
center = nn.Parameter(torch.zeros(num_train_ids, 768), requires_grad=False)
```

### Loss

```
loss = λ_id   * CrossEntropy(id_head(feat), label)
     + λ_tri  * TripletLoss(feat, label, margin=0.3, mining="hard")
     + λ_cen  * CenterLoss(feat, label, centers=center)
```

Suggested weights: `λ_id = 1.0, λ_tri = 1.0, λ_cen = 5e-4` (SOLIDER defaults).

### Optimizer / schedule

- Optimizer: **AdamW**, weight_decay = 1e-4.
- LR: **head 1e-4, backbone stage 4 1e-5** (10x smaller for the unfrozen
  backbone block).
- Schedule: **cosine annealing**, 20-30 epochs.
- Batch size: identity-sampled triplet batching — sample P=8 identities × K=4
  instances per batch = 32. T4 should fit this at 384×128.
- Warmup: 5-epoch linear warmup from 1e-6.

### Augmentation

- Random horizontal flip (p=0.5)
- Color jitter (brightness/contrast 0.2)
- Random erasing (p=0.5, area 0.02-0.4)
- No resize jitter (must stay at 384×128 for downstream compatibility)

### Logging

Per epoch: train loss, train identity-batch accuracy, val Rank-1, val mAP,
val cross-view cosine margin (same-id vs diff-id). Save best checkpoint by
val Rank-1.

## Evaluation (Colab notebook cell 5)

### ReID retrieval metrics (on scene_044)

- **Rank-1, Rank-5, mAP** with **cross-camera** queries: for each query crop on
  cam A, retrieve from a gallery containing crops from all cams except A. The
  same-ID match must be on a different camera than the query.
- Computed at the BN-neck output (the same feature the pipeline consumes).

### Per-camera-pair cosine margin

For each pair of cameras (A, B):
- Mean cosine of same-ID crops, one from A and one from B.
- Mean cosine of different-ID crops, one from A and one from B.
- Margin = same - different.

Report the full 7×7 matrix. The bottleneck is the rows/columns for cams
0394, 0395, 0396 against the front cams. Target: those rows have margin ≥ 0.40
(currently expected ≈ 0.10-0.20 based on the linking failure).

### Critical decision metric

**Mean same-ID cosine between back-cam (394/395/396) and front-cam (390-393)
crops.** Currently below the MCPT sim_th = 0.85 effective decision boundary
(otherwise we'd see more linking). Target: > 0.50, ideally > 0.70.

## Local integration (Colab notebook cell 6)

Outputs from the Colab session:

1. `weights/solider_swin_small_warehouse_ft.pth` — the fine-tuned checkpoint.
   Same state-dict shape and key names as the original — drop-in compatible
   with `load_solider_swin_small`.
2. `reid_finetune.ipynb` — the full notebook (commit back to this repo for
   reproducibility).
3. A short results table (Rank-1, mAP, cosine margins, per-camera-pair).

### Code changes required in this repo

Add a single config knob in `aic24_nvidia/config.py`:

```python
@dataclass(frozen=True)
class ReidCfg:
    similarity_thresh: float
    checkpoint_path: str = "weights/solider_swin_small.pth"   # new
```

Parse in `load_config`, threaded through to `aic24_nvidia/models/reid_solider.py`'s
`_embed` so it loads the user-specified weights.

Then `configs/baseline.yaml`:

```yaml
reid:
  similarity_thresh: 0.7
  checkpoint_path: weights/solider_swin_small_warehouse_ft.pth   # was: solider_swin_small.pth
```

### Test before committing

```bash
source .venv/bin/activate
# Rebuild reid + sct + mct + evaluate (~25 min) with the new checkpoint.
rm -rf outputs/baseline/{reid,sct,mct,evaluate}
python pipeline.py all --config configs/baseline.yaml --run-id baseline
```

Compare against v3 baseline (image HOTA 0.7580, world HOTA 0.5055).

## Decision rule

| Outcome | Action |
|---|---|
| Back-cam global IDs ≥ 5 AND world HOTA gain ≥ +0.01 | **Ship** — lock new checkpoint as v4 baseline, update `pipeline-state.md` |
| Back-cam global IDs increase but world HOTA flat or marginal | Check `sim_th` — try lowering 0.85 → 0.75 in a sweep variant |
| Pipeline numbers unchanged | Verify the checkpoint actually loaded (log feature distribution). If load is fine, escalate (see below). |
| Pipeline regresses | Backbone forgot too much. Reduce backbone LR (1e-5 → 1e-6) or freeze stage 4 too and retry head-only. |

## Escalation paths (if partial doesn't work)

In order of cost:

1. **Lower MCPT thresholds:** `sim_th` 0.85 → 0.75, `distance_th` 10 → 15. If
   the new embeddings just need more room to cluster, this is cheap.
2. **Per-camera tracker thresholds:** the harder variant of (1); requires code
   change in `aic24_nvidia/tracking_params.py`.
3. **Unfreeze stage 3 too:** double the trainable capacity, same loss + data
   prep. Adds ~1-2 hours Colab time.
4. **Stage-wise unfreezing:** train head only → unfreeze stage 4 → unfreeze
   stage 3 → unfreeze stage 2. Best results in ReID literature; most complex
   training loop. Full notebook rewrite.
5. **Full backbone fine-tune at very low LR (1e-6).** Risk of MSMT17 forgetting;
   only justified if (3) and (4) fail.

## Out of scope

- **Generic ReID quality improvement** beyond cross-cam linking. Other cosine
  metrics may move, but the target is back-cam linking specifically.
- **Architecture changes** (different Swin variant, different head topology).
- **Training data outside the NVIDIA MTMC dataset.** No MSMT17 mixing, no
  AIC23 data, etc. — fine-tuning aims for domain adaptation, not
  multi-domain learning.
- **End-to-end joint detector+reid training.** YOLO11 stays frozen; that's a
  separate workstream documented in `detector-recall-workstream-paused`.
- **Multi-scene tracker re-tuning after ReID change.** If the fine-tune ships,
  the existing tracker thresholds (epsilon_scpt=0.20-0.25, sim_th=0.85, etc.)
  may no longer be optimal. Re-running the tracker sweeps is a follow-up.

## References

- SOLIDER (the model + checkpoint source):
  https://github.com/tinyvision/SOLIDER-REID
- SOLIDER config we mirror: `configs/msmt17/swin_small.yml` in that repo.
- Our vendored implementation: `aic24_nvidia/models/solider/`.
- ReID loss recipe (ID + triplet + center): Luo et al., "Bag of Tricks and a
  Strong Baseline for Deep Person Re-Identification", CVPR 2019.
- The bottleneck this attacks:
  `docs/superpowers/notes/2026-05-28-world-projection-results.md` →
  "Diagnostic: where the remaining world-HOTA gap lives".

## Hand-off checklist (for the future session)

When you (or a future agent) pick this up:

1. ☐ Confirm scene_044 is still the held-out test scene (check `pipeline-state`
   memory; if v4 was shipped from elsewhere, re-align).
2. ☐ Check `pipeline-state.md` for any new bottleneck info or related work
   committed after 2026-05-28.
3. ☐ Decide data source (Colab Drive upload vs HF mirror vs raw NVIDIA
   release). See Data sourcing section.
4. ☐ Open a fresh Colab notebook. Pull this repo into the Colab VM. Pull the
   base checkpoint (Google Drive ID in
   `aic24_nvidia/models/solider/__init__.py:31`).
5. ☐ Execute cells in order. Save the final `.pth` to Google Drive.
6. ☐ Download `.pth` back to local `weights/solider_swin_small_warehouse_ft.pth`.
7. ☐ Add the `reid.checkpoint_path` config knob (see Local integration).
8. ☐ Re-run baseline pipeline locally; apply the Decision Rule.
9. ☐ If shipping, update `pipeline-state.md` memory and bump the version
   comment in `configs/baseline.yaml`.
