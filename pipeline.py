from __future__ import annotations
import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from aic24_nvidia.config import load_config, Config
from aic24_nvidia.paths import make_run_id, run_dir as run_dir_for, latest_run_id, stage_dir
from aic24_nvidia.stages import (
    adapt,
    extract_frames,
    detect,
    reid,
    pose,
    sct,
    mct,
    evaluate,
)
from aic24_nvidia import visualize
from aic24_nvidia.manifest import read_manifest, gate

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("pipeline")

STAGE_RUNNERS = {
    "adapt": adapt.run,
    "frames": extract_frames.run,
    "detect": detect.run,
    "reid": reid.run,
    "pose": pose.run,
    "sct": sct.run,
    "mct": mct.run,
    "evaluate": evaluate.run,
}
ORDER = ["adapt", "frames", "detect", "reid", "pose", "sct", "mct", "evaluate"]
STAGE_DIR_NAME = {s: ("adapted" if s == "adapt" else s) for s in ORDER}
UPSTREAM_OF = {
    "adapt": [],
    "frames": ["adapt"],
    "detect": ["frames"],
    "reid": ["detect"],
    "pose": ["reid"],
    "sct": ["pose"],
    "mct": ["sct"],
    "evaluate": ["mct"],
}


def _gate_stage(stage: str, rd: Path, force: bool) -> bool:
    upstream = [stage_dir(rd, STAGE_DIR_NAME[u]) / "manifest.json" for u in UPSTREAM_OF[stage]]
    own_dir = stage_dir(rd, STAGE_DIR_NAME[stage])
    try:
        decision = gate(own_dir, upstream=upstream, force=force)
    except RuntimeError as e:
        log.error("gate failed for stage '%s': %s", stage, e)
        raise
    if decision == "skip":
        log.info("stage '%s' already complete; skip. Use --force to rerun.", stage)
        return False
    return True


def _resolve_run_id(cfg: Config, args) -> str:
    if getattr(args, "run_id", None):
        return args.run_id
    rid = latest_run_id(cfg.outputs_root, cfg.config_filename)
    return rid or make_run_id(cfg.config_filename, at=datetime.now())


def _ensure_run_dir(cfg: Config, run_id: str) -> Path:
    rd = run_dir_for(cfg.outputs_root, run_id)
    rd.mkdir(parents=True, exist_ok=True)
    return rd


def cmd_stage(stage: str, args) -> None:
    cfg = load_config(args.config)
    run_id = _resolve_run_id(cfg, args)
    rd = _ensure_run_dir(cfg, run_id)
    log.info("stage=%s run_id=%s", stage, run_id)
    if not _gate_stage(stage, rd, force=args.force):
        return
    STAGE_RUNNERS[stage](cfg, rd, run_id)


def cmd_all(args) -> None:
    cfg = load_config(args.config)
    run_id = args.run_id or make_run_id(cfg.config_filename, at=datetime.now())
    rd = _ensure_run_dir(cfg, run_id)
    log.info("pipeline ALL run_id=%s", run_id)
    for s in ORDER:
        log.info("=== stage %s ===", s)
        if not _gate_stage(s, rd, force=args.force):
            continue
        STAGE_RUNNERS[s](cfg, rd, run_id)


def cmd_bootstrap(args) -> None:
    subprocess.run(["bash", "scripts/bootstrap_external.sh"], check=True)


def cmd_viz(args) -> None:
    cfg = load_config(args.config)
    run_id = args.run_id or latest_run_id(cfg.outputs_root, cfg.config_filename)
    if not run_id:
        log.error("no run_id and no prior runs"); sys.exit(2)
    rd = run_dir_for(cfg.outputs_root, run_id)
    adapted_root = rd / "adapted"
    scene = "scene_001"

    if args.stage == "detect":
        m = read_manifest(rd / "detect" / "manifest.json")
        for cam, paths in m.outputs.items():
            frame_dir = adapted_root / "Original" / scene / cam / "Frame"
            visualize.viz_detect_from_txt(frame_dir, Path(paths["txt"]),
                                          rd / "detect" / f"viz_{cam}.mp4",
                                          fps=cfg.fps)
    elif args.stage == "sct":
        m = read_manifest(rd / "sct" / "manifest.json")
        for cam_stem, sct_json in m.outputs.items():
            num = int(cam_stem.replace("camera", ""))
            cam_name = f"camera_{num:04d}"
            frame_dir = adapted_root / "Original" / scene / cam_name / "Frame"
            visualize.viz_tracks_from_yachiyo_sct(
                frame_dir, Path(sct_json),
                rd / "sct" / f"viz_{cam_name}.mp4",
                fps=cfg.fps,
            )
    elif args.stage == "mct":
        m = read_manifest(rd / "mct" / "manifest.json")
        mct_json = Path(m.outputs["global_tracks_json"])
        cam_frame_dirs = {p.name: p / "Frame" for p in sorted((adapted_root / "Original" / scene).glob("camera_*"))}
        visualize.viz_mct_grid_from_yachiyo_mct(
            cam_frame_dirs, mct_json,
            rd / "mct" / "viz_grid.mp4",
            fps=cfg.fps,
        )
    else:
        log.error("viz not implemented for stage: %s", args.stage); sys.exit(2)


def cmd_dashboard(args) -> None:
    subprocess.run(["streamlit", "run", "dashboard/app.py",
                    "--server.port", str(args.port)], check=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser("aic24-nvidia")
    sub = p.add_subparsers(dest="cmd", required=True)

    for s in ORDER + ["all"]:
        sp = sub.add_parser(s)
        sp.add_argument("--config", required=True, type=Path)
        sp.add_argument("--run-id", default=None)
        sp.add_argument("--force", action="store_true")
        sp.set_defaults(func=(cmd_all if s == "all"
                              else (lambda a, stage=s: cmd_stage(stage, a))))

    bp = sub.add_parser("bootstrap"); bp.set_defaults(func=cmd_bootstrap)

    vp = sub.add_parser("viz")
    vp.add_argument("--config", required=True, type=Path)
    vp.add_argument("--run-id", default=None)
    vp.add_argument("--stage", required=True, choices=["detect", "sct", "mct"])
    vp.set_defaults(func=cmd_viz)

    dp = sub.add_parser("dashboard")
    dp.add_argument("--port", type=int, default=8501)
    dp.set_defaults(func=cmd_dashboard)

    args = p.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
