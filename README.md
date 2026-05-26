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
python pipeline.py all --config configs/warehouse_001_30s.yaml
python pipeline.py dashboard --port 8501
```
