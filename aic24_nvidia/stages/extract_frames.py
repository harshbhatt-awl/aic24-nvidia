from __future__ import annotations
import logging
import subprocess
import sys
from pathlib import Path

from ..config import Config
from ..errors import StageError, ValidationError
from ..paths import stage_dir
from .base import atomic_stage

log = logging.getLogger(__name__)


def _yachiyo_scene_int(scene_name: str) -> int:
    return int(scene_name.split("_")[-1])


def _validate_frame_counts(adapted_root: Path, fps: int, duration_sec: float) -> dict[str, int]:
    expected = int(fps * duration_sec)
    tol = max(2, int(expected * 0.02))
    counts: dict[str, int] = {}
    for cam_dir in sorted((adapted_root / "Original").glob("*/camera_*")):
        frame_dir = cam_dir / "Frame"
        if not frame_dir.exists():
            raise ValidationError(f"no Frame/ under {cam_dir}")
        n = len(list(frame_dir.glob("*.jpg")))
        if abs(n - expected) > tol:
            raise ValidationError(
                f"{cam_dir.name}: extracted {n} frames, expected ~{expected} (tol {tol})"
            )
        counts[cam_dir.name] = n
    return counts


def WIRING(run_dir: Path, cfg: Config, output_dir: Path):
    # extract_frame.py runs with CWD=yachiyo and reads Original/ relative to it.
    # output_dir is unused — frames does not expose its own output via symlink.
    return [(cfg.yachiyo_root / "Original", stage_dir(run_dir, "adapted") / "Original")]


def run(cfg: Config, run_dir: Path, run_id: str) -> None:
    adapt_manifest = stage_dir(run_dir, "adapted") / "manifest.json"
    adapted_root = stage_dir(run_dir, "adapted")
    yachiyo = cfg.yachiyo_root
    entry = yachiyo / "tools" / "extract_frame.py"
    if not entry.exists():
        raise FileNotFoundError(f"YACHIYO entry missing: {entry}")

    scene_name = "scene_001"

    with atomic_stage(run_dir, "frames", run_id=run_id, cfg=cfg, wiring=WIRING) as ctx:
        log_path = ctx.work_dir / "log.txt"
        # extract_frame.py runs with CWD=yachiyo and reads yachiyo/Original; WIRING
        # points that at the adapted tree (applied before this body runs).
        with open(log_path, "w") as lf:
            proc = subprocess.run(
                [sys.executable, "tools/extract_frame.py", "-s", scene_name, "./"],
                cwd=yachiyo,
                stdout=lf, stderr=subprocess.STDOUT,
            )
        if proc.returncode != 0:
            raise StageError("frames", proc.returncode, str(log_path))

        counts = _validate_frame_counts(adapted_root, cfg.fps, cfg.clip.duration_sec)

        ctx.set_inputs({"adapted_root": str(adapted_root)})
        ctx.set_outputs({"frames_per_camera": counts})
        ctx.set_params({"fps": cfg.fps, "duration_sec": cfg.clip.duration_sec})
        ctx.set_upstream([str(adapt_manifest)])
