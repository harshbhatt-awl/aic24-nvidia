# aic24_nvidia/stages/adapt.py
from __future__ import annotations
import logging
from pathlib import Path

from ..adapter.nvidia_to_yachiyo import adapt_scene
from ..bootstrap import make_symlink
from ..config import Config
from .base import atomic_stage

log = logging.getLogger(__name__)


def run(cfg: Config, run_dir: Path, run_id: str) -> None:
    with atomic_stage(run_dir, "adapted", run_id=run_id) as ctx:
        outputs = adapt_scene(cfg, ctx.work_dir)
        ctx.set_inputs({"scene_dir": str(cfg.scene_dir)})
        ctx.set_outputs(outputs)
        ctx.set_params({
            "clip_start_sec": cfg.clip.start_sec,
            "clip_duration_sec": cfg.clip.duration_sec,
            "fps": cfg.fps,
        })
        ctx.set_upstream([])
    # After atomic promotion, set up the symlink so downstream stages see the data.
    final_adapted = run_dir / "adapted" / "Original"
    make_symlink(final_adapted, cfg.external_root / "Original")
    log.info("symlink: external/Original -> %s", final_adapted)
