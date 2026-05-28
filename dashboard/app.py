"""Streamlit dashboard for per-stage pipeline results — aic24-nvidia.

Launch:
    streamlit run dashboard/app.py [--server.port 8501]
    # Or via CLI:
    python pipeline.py dashboard --port 8501

Tabs:
    Architecture | Overview | Adapt | Frames | Detect | ReID | Pose | SCT | MCT | Metrics

Sibling to the aic23-nvidia dashboard. Read-only: never invokes the pipeline,
only reads files under outputs/<run_id>/<stage>/.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_output_root() -> Path:
    here = Path(__file__).resolve().parent
    # dashboard/app.py → aic24-nvidia/outputs/
    candidate = here.parent / "outputs"
    if candidate.is_dir():
        return candidate
    return here.parent


def _list_runs(output_root: Path) -> list[str]:
    if not output_root.is_dir():
        return []
    dirs = sorted(
        (d for d in output_root.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return [d.name for d in dirs]


def _load_manifest(run_dir: Path, stage: str) -> dict | None:
    p = run_dir / stage / "manifest.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Architecture diagrams
# ---------------------------------------------------------------------------

_DOT_ORIGINAL = r"""
digraph original {
    rankdir=TB;
    bgcolor="transparent";
    node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];
    edge [fontname="Helvetica", fontsize=9];

    data    [label="AIC24 Track 1 dataset\n45 scenes, AI-generated indoor people\n+ official ground-truth", fillcolor="#fff2cc"];
    extract [label="extract_frame.py\n→ Original/<scene>/<cam>/Frame/*.jpg", fillcolor="#fff2cc"];
    det     [label="Detector (in BoT-SORT sibling repo)\nYOLOX-x  (~99M params)\nbytetrack_x_mot17.pth.tar\nresolution HARDCODED to 1920x1080\n→ Detection/<scene>/<cam>.{txt,json}", fillcolor="#cfe2f3"];
    reid    [label="ReID (in deep-person-reid sibling)\nOSNet x1.0 (osnet_ms_m_c.pth.tar)\n512-d feature per detection\n→ EmbedFeature/<scene>/<cam>/feature_*.npy\n(O(N_det) files)", fillcolor="#d9ead3"];
    pose    [label="Pose (in mmpose 0.x sibling)\nHRNet-W48 COCO 256x192\n17 keypoints per detection\n→ Pose/<scene>/<cam>/<cam>_out_keypoint.json", fillcolor="#ead1dc"];
    sct     [label="SCT — YACHIYO tracking/infer.py -scpt\nuses detections + ReID features\nrequires per-cam calibration.json\nwith K[R|t] camera projection matrix\n→ Tracking/<scene>/camera<NNN>_tracking_results.json", fillcolor="#fce5cd"];
    mct     [label="MCT — YACHIYO tracking/infer.py -mcpt\nfeature clustering + pose keypoint\nconditioning + world-coord prior\n→ Tracking/<scene>/whole_tracking_results.json", fillcolor="#f4cccc"];
    submit  [label="tools/generate_submission.py\nfinal global tracks for AIC24 leaderboard", fillcolor="#d0e0e3"];

    data -> extract -> det -> reid -> pose -> sct -> mct -> submit;

    {rank=same; det reid pose}

    label="Original AIC24_Track1_YACHIYO_RIIPS pipeline (as shipped)\nNeeds 3 sibling repos (BoT-SORT, deep-person-reid, mmpose 0.x)\n+ conda envs (botsort_env, torchreid, openmmlab) per stage";
    labelloc="t";
    fontname="Helvetica-Bold";
    fontsize=12;
}
"""

_DOT_OURS = r"""
digraph ours {
    rankdir=TB;
    bgcolor="transparent";
    node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];
    edge [fontname="Helvetica", fontsize=9];

    nv       [label="NVIDIA PhysicalAI-SmartSpaces\nMTMC_Tracking_2024 / Warehouse_001\n7 cameras 0390..0396 × 30s @ 30 fps @ 1080p", fillcolor="#fff2cc"];
    adapter  [label="aic24_nvidia/adapter/\n• discover_cameras (lowercase glob)\n• slice 30s sub-clip per camera\n• materialize YACHIYO layout\n  Original/scene_001/camera_NNNN/video.mp4\n• per-cam calibration.json with K, R, t,\n  homography matrix, camera projection matrix\n• ground_truth.json → per-cam MOT gt.txt\n  + gt_world.txt + gt_validation.json", fillcolor="#fff2cc"];
    frames   [label="frames stage\nshells extract_frame.py from upstream\nsymlink external/Original → outputs/<run>/adapted\n→ 900 JPEGs / camera × 7 = 6,300 frames", fillcolor="#fff2cc"];
    det      [label="Detector\nYOLOX-x  same upstream script\nsymlink external/Detection → outputs/<run>/detect\nupstream writes directly into our run dir\n→ 32,944 detections across 7 cameras", fillcolor="#cfe2f3"];
    reid     [label="ReID\nsame upstream OSNet script\nrun via PYTHONPATH (setup.py broken)\n→ 32,944 × 512-d .npy", fillcolor="#d9ead3"];
    pose     [label="Pose — second venv .venv-pose\nPython 3.10 + torch 1.13.1+cu117\n+ mmcv-full 1.7.0 + mmpose 0.29.0\nstage wrapper invokes .venv-pose/bin/python\n→ HRNet-W48 keypoint JSONs, 7 cameras", fillcolor="#ead1dc"];
    sct      [label="SCT — patched YACHIYO infer.py\n• multiprocessing.set_start_method('fork')\n• patch scene_2_camera_id_file.json\n  with [390..396] before running\n→ Tracking/scene_001/fixed_camera<NNN>_tracking_results.json", fillcolor="#fce5cd"];
    mct      [label="MCT — patched YACHIYO mcpt.py\n• tolerate empty feature_stack\n• tolerate empty unique_local_ids\n→ whole_tracking_results.json\n13 global IDs, 8 spanning 2+ cameras", fillcolor="#f4cccc"];
    eval     [label="Evaluate (TrackEval wrapper)\nYACHIYO JSON → MOT-format CSV\ngenerate seqmap + seqinfo.ini\nrun_mot_challenge.py from external/TrackEval/\n→ per-cam HOTA / IDF1 / MOTA / MOTP / CLR_F1", fillcolor="#d0e0e3"];
    dash     [label="Streamlit dashboard\n(this app)", fillcolor="#cfe2f3"];

    nv -> adapter -> frames -> det -> reid -> pose -> sct -> mct -> eval -> dash;

    label="Our adapted pipeline on NVIDIA Warehouse_001 (6 GB VRAM budget)\nMain venv: Python 3.14 + torch 2.12 (everything except pose)\nPose venv: Python 3.10 + torch 1.13.1 (mmpose 0.x toolchain)";
    labelloc="t";
    fontname="Helvetica-Bold";
    fontsize=12;
}
"""

_COMPARISON_ROWS = [
    ("Data source",
     "AIC24 Track1 (synthetic indoor scenes from NVIDIA Omniverse)",
     "NVIDIA Warehouse_001 (real-style, 7 cams 0390-0396, 30 s sub-clip)"),
    ("Detector",
     "YOLOX-x  (~99M params)",
     "Same: YOLOX-x  (~99M params) — fits 6 GB VRAM"),
    ("Detector script",
     "BoT-SORT/tools/aic24_get_detection.py",
     "Same, invoked from our stage wrapper"),
    ("ReID model",
     "OSNet x1.0 (osnet_ms_m_c)",
     "Same"),
    ("ReID script",
     "deep-person-reid/torchreid/aic24_extract.py",
     "Same; PYTHONPATH override (broken setup.py)"),
    ("Pose backbone",
     "HRNet-W48 COCO 256x192 (mmpose 0.x)",
     "Same — but in a separate .venv-pose (Python 3.10)"),
    ("SCT script",
     "tracking/infer.py (SCT+MCT combined via tracking.sh)",
     "tracking/infer.py -scpt (SCT only) — split via flag"),
    ("MCT script",
     "tracking/infer.py (combined with SCT)",
     "tracking/infer.py -mcpt — separate stage"),
    ("Multi-camera tracking method",
     "anchor-Hungarian on ReID features + pose keypoint condition + world coord",
     "Same algorithm, two upstream crashes patched (feature_stack=None, empty local_ids)"),
    ("Environment",
     "3 conda envs (botsort_env, torchreid, openmmlab) per stage",
     "2 venvs (.venv main + .venv-pose for pose only) — managed by uv"),
    ("Python",
     "3.8/3.9 (mmpose 0.x era)",
     "3.14 (main) + 3.10 (pose) — needed patches in fast_reid (Mapping, torch._six) and infer.py (multiprocessing start method)"),
    ("Stage orchestration",
     "Manual shell scripts (scripts/extract_frame.sh, detection.sh, …)",
     "Single CLI: python pipeline.py {adapt,frames,detect,reid,pose,sct,mct,evaluate,viz,dashboard}"),
    ("Inter-stage contract",
     "Files on disk in fixed Original/Detection/EmbedFeature/Pose/Tracking layout",
     "Same on-disk layout (symlinked from external/) + per-stage manifest.json"),
    ("Atomicity",
     "None — partial outputs left on crash",
     "atomic_stage: write to <stage>.tmp/, rename only on success"),
    ("Resumability",
     "Manual cleanup, re-run from scratch",
     "Manifest-gated; pipeline.py all skips completed stages, --force to re-run"),
    ("Visualisation",
     "None bundled",
     "viz_detect / viz_sct / viz_mct_grid → H.264 mp4s, browser-playable in dashboard"),
    ("Evaluation",
     "Submission file for AIC24 leaderboard",
     "TrackEval HOTA / IDF1 / MOTA per-camera, against converted NVIDIA GT"),
    ("Hyperparameter knobs",
     "Hardcoded inside detector / extractor / tracking scripts",
     "Surfaced in configs/baseline.yaml + recorded in manifests — but NOT propagated to upstream (v2)"),
    ("GPU VRAM budget",
     "12-16 GB (workstation GPU)",
     "6 GB (RTX 3050 Laptop) — verified via torch.cuda.mem_get_info"),
]

_PATCHES = [
    "**Adapter (ours):** `discover_cameras` glob `camera_*.mp4` (lowercase, matches real NVIDIA naming); preserve real camera IDs (e.g. 0390..0396) instead of remapping to 0001..0007; place `calibration.json` and `gt_world.txt` *outside* `Original/scene_NNN/` so YACHIYO's extract_frame.py walks only camera dirs.",
    "**Adapter (ours):** rewrote `calibration.py` and `gt_converter.py` for the real schema `{\"cameras\": {camera_NNNN: {K,R,t}}, \"annotations\": [{camera, frame, person_id, world_xy, bbox_2d}]}`; computed per-camera `camera projection matrix` and `homography matrix` for YACHIYO.",
    "**`bootstrap.py`:** `patch_scene_camera_map` handles the real upstream schema (list of `{scene_name, camera_ids}` dicts).",
    "**fast_reid (in BoT-SORT):** `from collections import Mapping` → `from collections.abc import Mapping` (Python 3.10+).",
    "**fast_reid (in BoT-SORT):** `from torch._six import string_classes` → `string_classes = (str, bytes)` (PyTorch removed `_six`).",
    "**YACHIYO `tracking/infer.py`:** added `multiprocessing.set_start_method('fork', force=True)` — Python 3.14's default `spawn` broke the global-inheritance pattern in `single_tracking`.",
    "**YACHIYO `tracking/src/mcpt.py`:** `assign_global_id` early-returns `tracking_results` when no tracks were assigned (avoids `feature_stack.T` on `None`).",
    "**YACHIYO `tracking/src/mcpt.py`:** `interpolate_tracklet` skips cameras whose `unique_local_ids` is empty (`min([])` crash).",
    "**TrackEval (cloned):** patched `np.float/int/bool` → built-ins (numpy 1.20+ removed those aliases).",
    "**Evaluate stage:** parse TrackEval column `HOTA(0)` (not `HOTA`); generate MOTChallenge seqmap + per-seq `seqinfo.ini`; tolerate stubbed MCT manifest; filter out YACHIYO `OfflineID == -1` placeholder rows; tolerate entries without `GlobalOfflineID`.",
    "**Visualize:** `Coordinate` is a `dict {x1,y1,x2,y2}` in real YACHIYO output (not a list); added `_coord_to_xyxy_local`.",
    "**MCT wrapper:** prefer `fixed_whole_tracking_results.json` only if its GIDs span ≥2 cameras; otherwise fall back to raw `whole_tracking_results.json`.",
    "**Pose stage:** invoke `.venv-pose/bin/python` (Python 3.10 + torch 1.13.1 + mmcv-full 1.7.0 + mmpose 0.29.0) — mmpose 0.x doesn't install on the main Python 3.14 venv.",
]


def _render_architecture(st) -> None:
    st.subheader("Architecture: original AIC24 YACHIYO vs. ours")
    st.markdown(
        "What the upstream `riips/AIC24_Track1_YACHIYO_RIIPS` pipeline expects, "
        "compared with what's actually running here on top of the NVIDIA "
        "PhysicalAI-SmartSpaces dataset on a 6 GB consumer GPU."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("#### Original (AIC24 YACHIYO_RIIPS pipeline)")
        st.graphviz_chart(_DOT_ORIGINAL, use_container_width=True)
    with right:
        st.markdown("#### Ours (on NVIDIA Warehouse_001, dual venv)")
        st.graphviz_chart(_DOT_OURS, use_container_width=True)

    st.divider()
    st.markdown("### Streamlit dashboard architecture")
    st.markdown(
        "The dashboard itself is a thin read-only viewer. It does NOT call into "
        "the pipeline; it only reads files that the stages already wrote to "
        "`outputs/<run_id>/<stage>/`."
    )
    st.graphviz_chart(
        r"""
        digraph dashboard {
            rankdir=LR;
            bgcolor="transparent";
            node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];

            fs       [label="outputs/<run_id>/\n adapted/  frames/  detect/  reid/\n pose/  sct/  mct/  evaluate/\n + manifest.json per stage\n + viz_*.mp4  (H.264)", fillcolor="#fff2cc"];
            reader   [label="dashboard/app.py\n_list_runs() / _load_manifest()", fillcolor="#d9ead3"];
            tabs     [label="Streamlit tabs\nArchitecture | Overview | Adapt | Frames |\nDetect | ReID | Pose | SCT | MCT | Metrics", fillcolor="#cfe2f3"];
            browser  [label="Browser\nhttp://localhost:8501", fillcolor="#fce5cd"];

            fs -> reader -> tabs -> browser;
        }
        """,
        use_container_width=True,
    )

    st.divider()
    st.markdown("### Component-by-component comparison")
    import pandas as pd
    df = pd.DataFrame(
        _COMPARISON_ROWS,
        columns=["Aspect", "Original AIC24 YACHIYO_RIIPS", "Ours (NVIDIA, dual-venv)"],
    )
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### Upstream patches applied")
    st.markdown(
        "The upstream repo + sibling toolchains needed several fixes to run on "
        "modern Python and on real NVIDIA Warehouse data. None changes the algorithm — "
        "they fix import errors, multiprocessing semantics, and crashes on short clips:"
    )
    st.markdown("\n".join(f"{i}. {row}" for i, row in enumerate(_PATCHES, start=1)))


# ---------------------------------------------------------------------------
# Per-stage renderers
# ---------------------------------------------------------------------------

def _render_overview(st, run_dir: Path, selected_run: str) -> None:
    st.subheader(f"Run: `{selected_run}`")
    stages = ["adapted", "frames", "detect", "reid", "pose", "sct", "mct", "evaluate"]
    total_runtime = 0.0
    for stage in stages:
        m = _load_manifest(run_dir, stage)
        if m is None:
            st.error(f"**{stage}**: manifest not found")
            continue
        status = m.get("status", "unknown")
        icon = "green" if status == "ok" else "red"
        rt = m.get("runtime_sec", 0)
        total_runtime += rt
        st.markdown(
            f":{icon}[**{stage}**] — status=`{status}` "
            f"| runtime={rt:.1f}s "
            f"| finished={m.get('finished_at', 'n/a')}"
        )
        with st.expander(f"Manifest JSON — {stage}"):
            st.json(m)
    st.divider()
    st.metric("Total pipeline runtime", f"{total_runtime / 60:.1f} min")


def _render_adapt(st, run_dir: Path) -> None:
    st.subheader("Adapter — NVIDIA scene → YACHIYO layout")
    m = _load_manifest(run_dir, "adapted")
    if m is None:
        st.info("Adapt stage not run yet.")
        return
    cols = st.columns(3)
    cols[0].metric("Status", m.get("status", "?"))
    cols[1].metric("Runtime", f"{m.get('runtime_sec', 0):.1f}s")
    cols[2].metric("Cameras", m.get("outputs", {}).get("cameras", "?"))

    st.markdown("**Source scene**")
    st.code(m.get("inputs", {}).get("scene_dir", "?"))

    gtv_path = run_dir / "adapted" / "gt_validation.json"
    if gtv_path.exists():
        gtv = json.loads(gtv_path.read_text())
        st.subheader("Ground-truth reprojection sanity check")
        cols = st.columns(3)
        cols[0].metric("Total annotations", gtv.get("total", 0))
        cols[1].metric("Matched (within ε px)", gtv.get("matched", 0))
        cols[2].metric("Match ratio", f"{gtv.get('match_ratio', 0) * 100:.1f}%")
        with st.expander("Failure samples (truncated to 20)"):
            st.json(gtv.get("failures_truncated", []))

    scene_json_path = run_dir / "adapted" / "scene.json"
    if scene_json_path.exists():
        st.subheader("Scene mapping (camera names)")
        st.json(json.loads(scene_json_path.read_text()))

    with st.expander("Adapt manifest (raw)"):
        st.json(m)


def _render_frames(st, run_dir: Path) -> None:
    st.subheader("Frame extraction")
    m = _load_manifest(run_dir, "frames")
    if m is None:
        st.info("Frames stage not run yet.")
        return
    cols = st.columns(2)
    cols[0].metric("Status", m.get("status", "?"))
    cols[1].metric("Runtime", f"{m.get('runtime_sec', 0):.1f}s")

    per_cam = m.get("outputs", {}).get("frames_per_camera", {})
    if per_cam:
        st.subheader("Frames per camera")
        rows = [{"camera": k, "frames": v} for k, v in per_cam.items()]
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.metric("Total frames", sum(per_cam.values()))

    with st.expander("Frames manifest (raw)"):
        st.json(m)


def _render_detect(st, run_dir: Path) -> None:
    st.subheader("Detection — YOLOX-X")
    m = _load_manifest(run_dir, "detect")
    if m is None:
        st.info("Detect stage not run yet.")
        return
    cols = st.columns(3)
    cols[0].metric("Status", m.get("status", "?"))
    cols[1].metric("Runtime", f"{m.get('runtime_sec', 0):.1f}s")

    detect_dir = run_dir / "detect"
    # Count detections per camera from the txt files
    counts = {}
    scene_dir = detect_dir / "scene_001"
    if scene_dir.is_dir():
        for txt in sorted(scene_dir.glob("camera_*.txt")):
            n = sum(1 for _ in txt.open())
            counts[txt.stem] = n
    if counts:
        cols[2].metric("Total detections", sum(counts.values()))
        st.subheader("Detections per camera")
        rows = [{"camera": k, "detections": v} for k, v in counts.items()]
        st.dataframe(rows, use_container_width=True, hide_index=True)

    st.subheader("Per-camera viz (bounding boxes + conf)")
    videos = sorted(detect_dir.glob("viz_*.mp4"))
    if not videos:
        st.info(
            "No viz mp4s found. Run: `python pipeline.py viz --config ... --stage detect`"
        )
    else:
        for vp in videos:
            st.write(f"**{vp.name}**  ({vp.stat().st_size / 1024 / 1024:.1f} MB)")
            st.video(str(vp))

    with st.expander("Detect manifest (raw)"):
        st.json(m)


def _render_reid(st, run_dir: Path) -> None:
    st.subheader("ReID — OSNet ×1.0 (512-d features)")
    m = _load_manifest(run_dir, "reid")
    if m is None:
        st.info("ReID stage not run yet.")
        return
    cols = st.columns(3)
    cols[0].metric("Status", m.get("status", "?"))
    cols[1].metric("Runtime", f"{m.get('runtime_sec', 0):.1f}s")

    per_cam = m.get("outputs", {}).get("per_cam_feature_counts", {})
    if per_cam:
        cols[2].metric("Total features", sum(per_cam.values()))
        st.subheader("Features per camera")
        rows = [{"camera": k, "features": v} for k, v in per_cam.items()]
        st.dataframe(rows, use_container_width=True, hide_index=True)

    st.markdown(
        "Each feature is a 512-d vector saved as `feature_<frame>_<id>_<x1>_<x2>_<y1>_<y2>_<conf>.npy` "
        "(one file per detection). No visual output for this stage — features are abstract."
    )
    with st.expander("ReID manifest (raw)"):
        st.json(m)


def _render_pose(st, run_dir: Path) -> None:
    st.subheader("Pose — HRNet-W48 COCO 17-keypoint")
    m = _load_manifest(run_dir, "pose")
    if m is None:
        st.info("Pose stage not run yet.")
        return
    cols = st.columns(3)
    cols[0].metric("Status", m.get("status", "?"))
    cols[1].metric("Runtime", f"{m.get('runtime_sec', 0):.1f}s")
    cols[2].metric("Backbone", m.get("params", {}).get("model", "?"))

    pose_dir = run_dir / "pose" / "scene_001"
    if pose_dir.is_dir():
        st.subheader("Keypoint JSON sizes per camera")
        rows = []
        for cam_dir in sorted(pose_dir.glob("camera_*")):
            kp_files = list(cam_dir.glob("*.json"))
            if kp_files:
                kp = kp_files[0]
                size_mb = kp.stat().st_size / 1024 / 1024
                rows.append({"camera": cam_dir.name, "json": kp.name, "size_MB": f"{size_mb:.2f}"})
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)

    st.markdown(
        "Pose ran in a separate `.venv-pose` (Python 3.10 + torch 1.13.1+cu117 + "
        "mmcv-full 1.7.0 + mmpose 0.29.0) because mmpose 0.x doesn't install on the "
        "main Python 3.14 venv."
    )
    with st.expander("Pose manifest (raw)"):
        st.json(m)


def _render_sct(st, run_dir: Path) -> None:
    st.subheader("Single-Camera Tracking — YACHIYO `infer.py -scpt`")
    m = _load_manifest(run_dir, "sct")
    if m is None:
        st.info("SCT stage not run yet.")
        return
    cols = st.columns(3)
    cols[0].metric("Status", m.get("status", "?"))
    cols[1].metric("Runtime", f"{m.get('runtime_sec', 0):.1f}s")
    cols[2].metric("Cameras tracked", len(m.get("outputs", {})))

    sct_dir = run_dir / "sct"
    st.subheader("Per-camera tracking videos (colored by local OfflineID)")
    videos = sorted(sct_dir.glob("viz_*.mp4"))
    if not videos:
        st.info("No viz mp4s. Run: `python pipeline.py viz --config ... --stage sct`")
    else:
        for vp in videos:
            st.write(f"**{vp.name}**  ({vp.stat().st_size / 1024 / 1024:.1f} MB)")
            st.video(str(vp))

    with st.expander("SCT manifest (raw)"):
        st.json(m)


def _render_mct(st, run_dir: Path) -> None:
    st.subheader("Multi-Camera Tracking — YACHIYO `infer.py -mcpt`")
    m = _load_manifest(run_dir, "mct")
    if m is None:
        st.info("MCT stage not run yet.")
        return
    cols = st.columns(2)
    cols[0].metric("Status", m.get("status", "?"))
    cols[1].metric("Runtime", f"{m.get('runtime_sec', 0):.1f}s")

    # Summarise GlobalOfflineIDs from the json
    gtj_path = m.get("outputs", {}).get("global_tracks_json")
    if gtj_path and Path(gtj_path).exists():
        body = json.loads(Path(gtj_path).read_text())
        cams_per_gid: dict[int, set[str]] = defaultdict(set)
        for cam, entries in body.items():
            if not isinstance(entries, dict):
                continue
            for _sn, e in entries.items():
                if not isinstance(e, dict):
                    continue
                gid = e.get("GlobalOfflineID")
                if gid is None or gid < 0:
                    continue
                cams_per_gid[int(gid)].add(str(cam))
        total_gids = len(cams_per_gid)
        multi_cam = sum(1 for s in cams_per_gid.values() if len(s) >= 2)
        all7 = sum(1 for s in cams_per_gid.values() if len(s) == 7)

        c1, c2, c3 = st.columns(3)
        c1.metric("Unique global IDs", total_gids)
        c2.metric("GIDs spanning ≥2 cameras", multi_cam)
        c3.metric("GIDs spanning all 7", all7)

        st.subheader("Cross-camera ID coverage")
        rows = sorted(
            ({"GID": gid, "cameras": ", ".join(sorted(cams)), "n_cameras": len(cams)}
             for gid, cams in cams_per_gid.items()),
            key=lambda r: (-r["n_cameras"], r["GID"]),
        )
        st.dataframe(rows, use_container_width=True, hide_index=True)

    st.subheader("Multi-camera grid viz")
    grid_vid = run_dir / "mct" / "viz_grid.mp4"
    if grid_vid.exists():
        st.write(
            f"**viz_grid.mp4**  ({grid_vid.stat().st_size / 1024 / 1024:.1f} MB) — "
            f"same global ID = same color across all cameras"
        )
        st.video(str(grid_vid))
    else:
        st.info("No grid viz. Run: `python pipeline.py viz --config ... --stage mct`")

    with st.expander("MCT manifest (raw)"):
        st.json(m)


def _render_metrics(st, run_dir: Path) -> None:
    st.subheader("Evaluation — TrackEval (HOTA / IDF1 / MOTA)")
    eval_dir = run_dir / "evaluate"
    if not eval_dir.is_dir():
        st.info("Evaluate stage not run yet.")
        return

    metrics_path = eval_dir / "metrics.json"
    summary_path = eval_dir / "summary.md"

    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        # Headline metrics from COMBINED row if present
        combined = metrics.get("COMBINED")
        if combined:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Combined HOTA", f"{combined.get('HOTA', 0):.3f}")
            c2.metric("Combined IDF1", f"{combined.get('IDF1', 0):.3f}")
            c3.metric("Combined MOTA", f"{combined.get('MOTA', 0):.3f}")
            c4.metric("Combined MOTP", f"{combined.get('MOTP', 0):.3f}")

        st.subheader("Per-camera metrics")
        rows = []
        for seq, mvals in metrics.items():
            if seq in ("COMBINED", "mct_world"):
                continue
            rows.append({
                "sequence": seq,
                "HOTA": f"{mvals.get('HOTA', 0):.4f}",
                "IDF1": f"{mvals.get('IDF1', 0):.4f}",
                "MOTA": f"{mvals.get('MOTA', 0):.4f}",
                "MOTP": f"{mvals.get('MOTP', 0):.4f}",
                "IDR": f"{mvals.get('IDR', 0):.4f}",
                "IDP": f"{mvals.get('IDP', 0):.4f}",
                "CLR_F1": f"{mvals.get('CLR_F1', 0):.4f}",
            })
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)

        mw = metrics.get("mct_world")
        if isinstance(mw, dict):
            st.subheader("Scene MCT — 3D world")
            if "skipped" in mw:
                st.warning(f"World MCT eval skipped: {mw['skipped']}")
            else:
                cols = st.columns(5)
                for col, key in zip(cols, ("HOTA", "DetA", "AssA", "IDF1", "MOTA")):
                    if key in mw:
                        col.metric(key, f"{mw[key]:.3f}")
                st.caption(f"d_max = {mw.get('d_max_m')} m · "
                           f"{mw.get('frames_evaluated')} frames · "
                           f"{mw.get('dropped_detections', 0)} dets dropped")

    if summary_path.exists():
        with st.expander("summary.md (rendered)"):
            st.markdown(summary_path.read_text())

    if metrics_path.exists():
        with st.expander("metrics.json (raw)"):
            st.json(json.loads(metrics_path.read_text()))


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="AIC24-NVIDIA Pipeline Dashboard",
        page_icon=":movie_camera:",
        layout="wide",
    )
    st.title("AIC24-NVIDIA Pipeline Dashboard")
    st.caption(
        "Runs the unmodified `riips/AIC24_Track1_YACHIYO_RIIPS` pipeline (with a few "
        "compat patches) on NVIDIA PhysicalAI-SmartSpaces MTMC_Tracking_2024 data."
    )

    output_root = _find_output_root()

    with st.sidebar:
        st.header("Run Selection")
        runs = _list_runs(output_root)
        if not runs:
            st.warning(f"No runs found in {output_root}")
            st.stop()
        selected_run = st.selectbox("Pipeline run", runs, index=0)
        st.caption(f"Output root: `{output_root}`")

    run_dir = output_root / selected_run

    tabs = st.tabs([
        "Architecture", "Overview",
        "Adapt", "Frames", "Detect", "ReID", "Pose", "SCT", "MCT",
        "Metrics",
    ])

    with tabs[0]:
        _render_architecture(st)
    with tabs[1]:
        _render_overview(st, run_dir, selected_run)
    with tabs[2]:
        _render_adapt(st, run_dir)
    with tabs[3]:
        _render_frames(st, run_dir)
    with tabs[4]:
        _render_detect(st, run_dir)
    with tabs[5]:
        _render_reid(st, run_dir)
    with tabs[6]:
        _render_pose(st, run_dir)
    with tabs[7]:
        _render_sct(st, run_dir)
    with tabs[8]:
        _render_mct(st, run_dir)
    with tabs[9]:
        _render_metrics(st, run_dir)


if __name__ == "__main__":
    main()
else:
    # When run via `streamlit run`, the module is imported; call main() directly.
    main()
