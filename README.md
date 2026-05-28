# aic24-nvidia

Sibling project to `aic23-nvidia/`. Runs the unmodified
[AIC24_Track1_YACHIYO_RIIPS](https://github.com/riips/AIC24_Track1_YACHIYO_RIIPS) pipeline
on a 30-second sub-clip of one NVIDIA PhysicalAI-SmartSpaces MTMC_Tracking_2024 Warehouse scene.

Per-stage outputs, MOT metrics, and a Streamlit dashboard. See
`../docs/superpowers/specs/2026-05-26-aic24-yachiyo-on-nvidia-mtmc-design.md` for design.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .

# clone upstream
mkdir -p external && cd external
git clone https://github.com/riips/AIC24_Track1_YACHIYO_RIIPS.git
cd ..

# place an NVIDIA Warehouse scene under data/nvidia_mtmc_2024/Warehouse_XXX/
# then:
python pipeline.py all --config configs/baseline.yaml
python pipeline.py dashboard --port 8501
```

## Manual smoke test (before committing GPU hours)

After cloning a real NVIDIA Warehouse scene into `data/nvidia_mtmc_2024/Warehouse_001/`:

```bash
# 1. Adapt only — 5 seconds, no GPU needed
python -c "
import yaml, pathlib
cfg = yaml.safe_load(open('configs/baseline.yaml'))
cfg['clip']['duration_sec'] = 5
pathlib.Path('configs/smoke.yaml').write_text(yaml.safe_dump(cfg))
"
python pipeline.py adapt --config configs/smoke.yaml

# 2. Find the most recent run
ls -t outputs/ | head -1
RUN_ID=$(ls -t outputs/ | head -1)

# 3. Open a frame manually to sanity-check
ffmpeg -y -ss 0 -i "outputs/$RUN_ID/adapted/Original/scene_001/camera_0001/video.mp4" -frames:v 1 /tmp/smoke.jpg
xdg-open /tmp/smoke.jpg  # or open /tmp/smoke.jpg on macOS

# 4. Inspect GT validation
cat outputs/$RUN_ID/adapted/gt_validation.json | jq .match_ratio

# If match_ratio >= 0.95 and the frame looks sane, proceed to a full GPU run.
```
