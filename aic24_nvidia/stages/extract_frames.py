from __future__ import annotations
import logging
import subprocess
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


def run(cfg: Config, run_dir: Path, run_id: str) -> None:
    adapt_manifest = stage_dir(run_dir, "adapted") / "manifest.json"
    adapted_root = stage_dir(run_dir, "adapted")
    yachiyo = cfg.yachiyo_root
    entry = yachiyo / "tools" / "extract_frame.py"
    if not entry.exists():
        raise FileNotFoundError(f"YACHIYO entry missing: {entry}")

    scene_name = "scene_001"

    with atomic_stage(run_dir, "frames", run_id=run_id) as ctx:
        log_path = ctx.work_dir / "log.txt"
        # extract_frame.py runs with CWD=yachiyo and root_path="./"
        # which resolves to yachiyo/Original/.. — we need yachiyo/Original to be
        # the adapted tree. The adapt stage already symlinked
        # external/Original -> adapted/Original. But this stage's invocation
        # passes "./" (the upstream repo root) as root_path, so it looks at
        # yachiyo/Original, NOT external/Original. We additionally symlink
        # yachiyo/Original -> adapted/Original for the duration of this stage.
        from ..bootstrap import make_symlink, ensure_dir_clean
        yachiyo_original = yachiyo / "Original"
        ensure_dir_clean(yachiyo_original)
        make_symlink(adapted_root / "Original", yachiyo_original)

        with open(log_path, "w") as lf:
            proc = subprocess.run(
                ["python3", "tools/extract_frame.py", "-s", scene_name, "./"],
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
