# experiments/

Lightweight harness for running A/B variants of the pipeline against a locked
baseline, **without** copying the repo or re-running the slow stages.

## What is "baseline"?

`configs/baseline.yaml` defines the **v2 reference** — YOLO11 + SOLIDER (native)
+ RTMPose, with the 2026-05-27 sweep-tuned tracking thresholds (image HOTA
0.758, IDF1 0.750, MOTA 0.781 on the 30s Warehouse_001 clip).

⚠️ The model swaps live on `feat/model-upgrades-impl`, NOT on `main`. To
produce `outputs/baseline/`, run from the worktree:

```bash
cd ~/.config/superpowers/worktrees/aic24-nvidia/model-upgrades
python experiments/run.py ensure-baseline           # ≈1 hour
```

After the v2 branch is merged into main, you can run from main directly.

## Quick start

```bash
# 1. Build the baseline once (≈1 hour, full pipeline run on v2 branch).
python experiments/run.py ensure-baseline

# 2. See what experiments are defined.
python experiments/run.py list

# 3. Run one experiment (all its variants).
python experiments/run.py run eps_mcpt_sweep

# 4. Compare results.
python experiments/compare.py
```

## What lives where

```
experiments/
    registry.yaml      ← experiment definitions (edit this to add experiments)
    run.py             ← runner CLI
    compare.py         ← results table CLI
    _lib.py            ← internal helpers (deep-merge, cache symlinks)
    README.md          ← you are here

configs/
    baseline.yaml      ← locked reference config (compare every experiment to this)

outputs/
    baseline/                     ← the locked baseline run (DO NOT DELETE)
    eps_mcpt_sweep__0.30/         ← variant run dir; per-stage subdirs are
                                     symlinks into outputs/baseline/ for
                                     anything before `rerun_from`.
    eps_mcpt_sweep__0.30/_config.yaml      ← merged variant config (read-only)
    eps_mcpt_sweep__0.30/_experiment.json  ← provenance (what experiment, what
                                              overrides, which stages reused)
```

## Worktrees vs this harness — when to use which

| You are changing... | Use |
|---|---|
| A YAML field (threshold, flag, sweep) | **This harness** — config-only, fast |
| A `.py` file (model swap, new algorithm) | **Git worktree** — needs isolated code |
| Both | Branch first, then run experiments from that branch's checkout |

Rule of thumb: if `diff main..your-change` is empty, stay in `main` and use
`experiments/`. If it's non-empty, use a worktree.

## Adding an experiment

Open `registry.yaml`, append a new entry:

```yaml
  - id: my_new_experiment
    description: "One-line what."
    hypothesis: "One-line why it might help."
    base_config: configs/baseline.yaml
    rerun_from: sct        # see table below
    variants:
      - name: "off"
        overrides:
          some_section:
            some_field: false
      - name: "on"
        overrides:
          some_section:
            some_field: true
```

### `rerun_from` — which stage triggers a re-run?

Pick the **earliest** stage whose output depends on the override. The harness
will reuse all earlier stages from `outputs/baseline/`:

| Override section | rerun_from |
|---|---|
| `tracking_params.*` | `sct` |
| `mct.*` (cluster_thresh, min_track_len, hard_world_gate) | `mct` |
| `eval.world_d_max` | `evaluate` |
| `reid.similarity_thresh` | `sct` (consumes embeddings) |
| `pose.keypoint_conf` | `sct` |
| `detect.conf_thresh` / `nms_iou` | `detect` |
| `clip.*`, `scene`, `fps` | `adapt` (full rerun) |

When in doubt, pick earlier. Gating is cheap; wrong-skip is silent.

## How cache reuse actually works

1. The runner creates `outputs/<exp>__<variant>/` if absent.
2. For each stage **before** `rerun_from`, it creates a symlink
   `outputs/<exp>__<variant>/<stage>` → `outputs/baseline/<stage>`.
3. The pipeline's per-stage `gate()` sees a valid `manifest.json` and skips.
4. The first un-cached stage runs with `--force`; subsequent stages run normally
   (they see fresh upstream manifests and won't skip).
5. Outputs from re-run stages are written *into the variant's own run dir* —
   they don't touch baseline.

Disk cost per variant: a few hundred MB (just SCT/MCT/evaluate outputs), not
the 5–10 GB a full re-run would take.

## GPU contention

The pipeline holds the GPU during `detect`, `reid`, `pose`. Two pipeline
processes running concurrently on a 6 GB GPU will OOM. Safe patterns:

- **Sequential**: `python experiments/run.py run eps_mcpt_sweep` runs variants
  one at a time. Default and safe.
- **Parallel SCT-only experiments**: when `rerun_from >= sct`, the variant
  doesn't touch the GPU. You can launch multiple in parallel — they'll race
  only on the upstream's `external/Tracking` symlink, so keep them in the same
  experiment (the harness serializes within an experiment) OR space launches by
  ≥1 second to avoid the symlink race.

## Status & results

```bash
python experiments/run.py status     # which variants have finished
python experiments/compare.py        # results table
python experiments/compare.py --sort-by mct_world.HOTA
python experiments/compare.py --markdown report.md   # also write report.md
```

The table shows mean per-camera image metrics and the scene-level `mct_world`
block, with deltas vs baseline on the chosen sort metric.

## Removing / re-running a variant

```bash
# Force re-run a single variant (deletes its run dir first).
python experiments/run.py run eps_mcpt_sweep --variant 0.30 --force

# Or manually clean up.
rm -rf outputs/eps_mcpt_sweep__0.30
```

⚠️ Do **not** `rm -rf outputs/baseline/` casually — every experiment depends on
it via symlinks. If you really want to rebuild it, use
`python experiments/run.py ensure-baseline --force`.
