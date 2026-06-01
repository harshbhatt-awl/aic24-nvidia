# Results ledger

_Auto-generated from `results/runs.jsonl` by `scripts/results.py`. Don't edit by hand — run `python scripts/results.py scan` to refresh._

- **Baseline** (`baseline`): image HOTA **0.7677**, world HOTA **0.6567** _(world is the project's priority metric)_.
- **Best world HOTA**: `baseline` (0.6567) — (baseline)
- **Runs recorded**: 11  |  **Last updated**: 2026-06-01

Metrics: `img*` = combined per-camera (image space); `w*` = scene 3D-world MCT (priority). `Δw` = world HOTA vs baseline.

## By day

### 2026-06-01

| run_id | experiment / variant | img HOTA | w HOTA | w AssA | w IDF1 | Δw | config | runtime |
|---|---|---|---|---|---|---|---|---|
| `baseline` | (baseline) | 0.7677 | 0.6567 | 0.6597 | 0.8203 | +0.000 | eps_s=0.15 eps_m=0.37 kp=3 stt=120 sim=0.85 ankle_lower | 56m |
| `world_stitch_sweep__g30_d0.5` | world_stitch_sweep/g30_d0.5 | 0.7677 | 0.6567 | 0.6597 | 0.8203 | +0.000 | eps_s=0.15 eps_m=0.37 kp=3 stt=120 sim=0.85 ankle_lower | 56m |
| `world_stitch_sweep__g45_d0.6` | world_stitch_sweep/g45_d0.6 | 0.7677 | 0.6567 | 0.6597 | 0.8203 | +0.000 | eps_s=0.15 eps_m=0.37 kp=3 stt=120 sim=0.85 ankle_lower | 56m |
| `world_stitch_sweep__g60_d0.75` | world_stitch_sweep/g60_d0.75 | 0.7677 | 0.6567 | 0.6597 | 0.8203 | +0.000 | eps_s=0.15 eps_m=0.37 kp=3 stt=120 sim=0.85 ankle_lower | 56m |
| `world_stitch_sweep__g90_d1.0` | world_stitch_sweep/g90_d1.0 | 0.7677 | 0.6567 | 0.6597 | 0.8203 | +0.000 | eps_s=0.15 eps_m=0.37 kp=3 stt=120 sim=0.85 ankle_lower | 56m |
| `world_stitch_sweep__none` | world_stitch_sweep/none | 0.7677 | 0.6479 | 0.6423 | 0.8040 | −0.009 | eps_s=0.15 eps_m=0.37 kp=3 stt=120 sim=0.85 ankle_lower | 56m |

### 2026-05-31

| run_id | experiment / variant | img HOTA | w HOTA | w AssA | w IDF1 | Δw | config | runtime |
|---|---|---|---|---|---|---|---|---|
| `baseline_20260601_024726` | snapshot/baseline | 0.7677 | 0.6479 | 0.6423 | 0.8040 | −0.009 | eps_s=0.15 eps_m=0.37 kp=3 stt=120 sim=0.85 ankle_lower | 53m |
| `reid_finetune__ft` | reid_finetune/ft | 0.7951 | 0.6211 | 0.5881 | 0.7519 | −0.036 | eps_s=0.4 eps_m=0.55 kp=3 stt=120 sim=0.85 ankle_lower | 55m |
| `scene041` | generalization/scene_041 | 0.8282 | 0.4678 | 0.5414 | 0.6325 | −0.189 | eps_s=0.2 eps_m=0.37 kp=1 stt=120 sim=0.85 ankle_lower | 87m |

### 2026-05-30

| run_id | experiment / variant | img HOTA | w HOTA | w AssA | w IDF1 | Δw | config | runtime |
|---|---|---|---|---|---|---|---|---|
| `baseline_20260530_134259` | snapshot/baseline 2026-05-30 | 0.7800 | 0.5282 | 0.6338 | 0.6782 | −0.129 | eps_s=0.2 eps_m=0.37 kp=1 stt=120 sim=0.85 ankle_lower | 55m |

### 2026-05-27

| run_id | experiment / variant | img HOTA | w HOTA | w AssA | w IDF1 | Δw | config | runtime |
|---|---|---|---|---|---|---|---|---|
| `v2_solider` | model-stack/v2 SOLIDER | 0.6821 | 0.3412 | 0.3652 | 0.4796 | −0.316 | eps_s=0.8 eps_m=0.37 kp=1 stt=120 sim=0.85 | 57m |

## By experiment

### (baseline)

| run_id | variant | img HOTA | w HOTA | w DetA | w AssA | w IDF1 | Δw | config | date |
|---|---|---|---|---|---|---|---|---|---|
| `baseline` | — | 0.7677 | 0.6567 | 0.6549 | 0.6597 | 0.8203 | +0.000 | eps_s=0.15 eps_m=0.37 kp=3 stt=120 sim=0.85 ankle_lower | 2026-06-01 |

### generalization

| run_id | variant | img HOTA | w HOTA | w DetA | w AssA | w IDF1 | Δw | config | date |
|---|---|---|---|---|---|---|---|---|---|
| `scene041` | scene_041 | 0.8282 | 0.4678 | 0.4056 | 0.5414 | 0.6325 | −0.189 | eps_s=0.2 eps_m=0.37 kp=1 stt=120 sim=0.85 ankle_lower | 2026-05-31 |

### model-stack

| run_id | variant | img HOTA | w HOTA | w DetA | w AssA | w IDF1 | Δw | config | date |
|---|---|---|---|---|---|---|---|---|---|
| `v2_solider` | v2 SOLIDER | 0.6821 | 0.3412 | 0.3193 | 0.3652 | 0.4796 | −0.316 | eps_s=0.8 eps_m=0.37 kp=1 stt=120 sim=0.85 | 2026-05-27 |

### reid_finetune

| run_id | variant | img HOTA | w HOTA | w DetA | w AssA | w IDF1 | Δw | config | date |
|---|---|---|---|---|---|---|---|---|---|
| `reid_finetune__ft` | ft | 0.7951 | 0.6211 | 0.6571 | 0.5881 | 0.7519 | −0.036 | eps_s=0.4 eps_m=0.55 kp=3 stt=120 sim=0.85 ankle_lower | 2026-05-31 |

### snapshot

| run_id | variant | img HOTA | w HOTA | w DetA | w AssA | w IDF1 | Δw | config | date |
|---|---|---|---|---|---|---|---|---|---|
| `baseline_20260601_024726` | baseline | 0.7677 | 0.6479 | 0.6549 | 0.6423 | 0.8040 | −0.009 | eps_s=0.15 eps_m=0.37 kp=3 stt=120 sim=0.85 ankle_lower | 2026-05-31 |
| `baseline_20260530_134259` | baseline 2026-05-30 | 0.7800 | 0.5282 | 0.4405 | 0.6338 | 0.6782 | −0.129 | eps_s=0.2 eps_m=0.37 kp=1 stt=120 sim=0.85 ankle_lower | 2026-05-30 |

### world_stitch_sweep

| run_id | variant | img HOTA | w HOTA | w DetA | w AssA | w IDF1 | Δw | config | date |
|---|---|---|---|---|---|---|---|---|---|
| `world_stitch_sweep__g30_d0.5` | g30_d0.5 | 0.7677 | 0.6567 | 0.6549 | 0.6597 | 0.8203 | +0.000 | eps_s=0.15 eps_m=0.37 kp=3 stt=120 sim=0.85 ankle_lower | 2026-06-01 |
| `world_stitch_sweep__g45_d0.6` | g45_d0.6 | 0.7677 | 0.6567 | 0.6549 | 0.6597 | 0.8203 | +0.000 | eps_s=0.15 eps_m=0.37 kp=3 stt=120 sim=0.85 ankle_lower | 2026-06-01 |
| `world_stitch_sweep__g60_d0.75` | g60_d0.75 | 0.7677 | 0.6567 | 0.6549 | 0.6597 | 0.8203 | +0.000 | eps_s=0.15 eps_m=0.37 kp=3 stt=120 sim=0.85 ankle_lower | 2026-06-01 |
| `world_stitch_sweep__g90_d1.0` | g90_d1.0 | 0.7677 | 0.6567 | 0.6549 | 0.6597 | 0.8203 | +0.000 | eps_s=0.15 eps_m=0.37 kp=3 stt=120 sim=0.85 ankle_lower | 2026-06-01 |
| `world_stitch_sweep__none` | none | 0.7677 | 0.6479 | 0.6549 | 0.6423 | 0.8040 | −0.009 | eps_s=0.15 eps_m=0.37 kp=3 stt=120 sim=0.85 ankle_lower | 2026-06-01 |
