from __future__ import annotations
import json
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT_DEFAULT = REPO_ROOT / "outputs"

STAGE_DIRS = ["adapted", "frames", "detect", "reid", "pose", "sct", "mct", "evaluate"]


def list_runs(outputs_root: Path) -> list[str]:
    if not outputs_root.exists():
        return []
    return sorted((p.name for p in outputs_root.iterdir() if p.is_dir()), reverse=True)


def load_manifest(run_dir: Path, stage: str) -> dict | None:
    p = run_dir / stage / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else None


def find_viz(run_dir: Path, stage: str) -> list[Path]:
    return sorted((run_dir / stage).glob("viz*.mp4"))


def render_stage_tab(run_dir: Path, stage: str) -> None:
    m = load_manifest(run_dir, stage)
    if not m:
        st.info(f"Stage '{stage}' has not run for this run.")
        return
    st.subheader(f"{stage} — status: {m['status']}")
    c1, c2 = st.columns(2)
    c1.metric("Runtime (s)", f"{m['runtime_sec']:.1f}")
    c2.metric("Started", m["started_at"])
    st.json({"params": m["params"], "outputs": m["outputs"]}, expanded=False)
    vizs = find_viz(run_dir, stage)
    if not vizs:
        st.write(f"(no viz mp4 — run `python pipeline.py viz --stage {stage} --run-id {run_dir.name} --config <yaml>`)")
        return
    for v in vizs:
        st.markdown(f"**{v.name}**")
        st.video(str(v))


def main() -> None:
    st.set_page_config(page_title="aic24-nvidia", layout="wide")
    st.title("AIC24 YACHIYO on NVIDIA MTMC")

    outputs_root = Path(st.sidebar.text_input("outputs_root", str(OUTPUTS_ROOT_DEFAULT)))
    runs = list_runs(outputs_root)
    if not runs:
        st.warning(f"No runs found under {outputs_root}."); return
    run = st.sidebar.selectbox("Run", runs, index=0)
    run_dir = outputs_root / run

    tabs = st.tabs(["Overview"] + [s.capitalize() for s in STAGE_DIRS])
    with tabs[0]:
        st.subheader("Per-stage status")
        for s in STAGE_DIRS:
            m = load_manifest(run_dir, s)
            status = "—" if m is None else m["status"]
            runtime = "—" if m is None else f"{m['runtime_sec']:.1f}s"
            st.write(f"- **{s}**: {status} ({runtime})")

    for i, stage in enumerate(STAGE_DIRS, start=1):
        with tabs[i]:
            if stage == "adapted":
                render_stage_tab(run_dir, stage)
                gtv = run_dir / "adapted" / "gt_validation.json"
                if gtv.exists():
                    st.subheader("GT reprojection validation")
                    st.json(json.loads(gtv.read_text()))
            elif stage == "evaluate":
                m = load_manifest(run_dir, "evaluate")
                if not m:
                    st.info("evaluate has not run for this run."); continue
                metrics_path = Path(m["outputs"].get("metrics", ""))
                summary_path = Path(m["outputs"].get("summary", ""))
                if metrics_path.exists():
                    st.subheader("Metrics"); st.json(json.loads(metrics_path.read_text()))
                if summary_path.exists():
                    st.markdown(summary_path.read_text())
            else:
                render_stage_tab(run_dir, stage)


if __name__ == "__main__":
    main()
