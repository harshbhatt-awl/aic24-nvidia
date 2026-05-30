# aic24_nvidia/stages/adapt.py
from __future__ import annotations
import logging
from pathlib import Path

from ..adapter.nvidia_to_yachiyo import adapt_scene
from ..config import Config
from .base import atomic_stage

log = logging.getLogger(__name__)


def WIRING(run_dir: Path, cfg: Config, output_dir: Path):
    # Expose the adapted Original tree so downstream stages and the upstream
    # tooling find frames at external/Original.
    return [(cfg.external_root / "Original", output_dir / "Original")]


def run(cfg: Config, run_dir: Path, run_id: str) -> None:
    with atomic_stage(run_dir, "adapted", run_id=run_id, cfg=cfg, wiring=WIRING) as ctx:
        outputs = adapt_scene(cfg, ctx.work_dir)
        ctx.set_inputs({"scene_dir": str(cfg.scene_dir)})
        ctx.set_outputs(outputs)
        ctx.set_params({
            "clip_start_sec": cfg.clip.start_sec,
            "clip_duration_sec": cfg.clip.duration_sec,
            "fps": cfg.fps,
        })
        ctx.set_upstream([])
