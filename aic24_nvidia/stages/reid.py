from __future__ import annotations
import logging
from pathlib import Path

from ..config import Config
from ..errors import ValidationError
from ..paths import stage_dir
from .base import atomic_stage, assert_vram_free

log = logging.getLogger(__name__)

SCENE = "scene_001"


def WIRING(run_dir: Path, cfg: Config, output_dir: Path):
    # Expose this stage's embeddings at external/EmbedFeature.
    return [(cfg.external_root / "EmbedFeature", output_dir)]


def _per_cam_feature_counts(emb_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    scene_dir = emb_dir / SCENE
    if not scene_dir.exists():
        return counts
    for cam_dir in sorted(scene_dir.glob("camera_*")):
        counts[cam_dir.name] = len(list(cam_dir.glob("feature_*.npy")))
    return counts


def run(cfg: Config, run_dir: Path, run_id: str) -> None:
    assert_vram_free(cfg.vram_min_free_gb)

    detect_manifest = stage_dir(run_dir, "detect") / "manifest.json"

    with atomic_stage(run_dir, "reid", run_id=run_id, cfg=cfg, wiring=WIRING) as ctx:
        # external/EmbedFeature is wired by WIRING (output_dir during run, final
        # after promotion).
        from ..models import reid_solider
        original = cfg.external_root / "Original"
        det_scene = stage_dir(run_dir, "detect") / SCENE
        cams = sorted(p.stem for p in det_scene.glob("camera_*.txt"))
        reid_solider.run_reid(
            det_scene_dir=det_scene,
            original_scene_dir=original / SCENE,
            emb_out_dir=ctx.work_dir,
            scene=SCENE,
            cams=cams,
        )

        counts = _per_cam_feature_counts(ctx.work_dir)
        if not counts:
            raise ValidationError("no per-camera embeddings produced")
        for cam, n in counts.items():
            log.info("reid: %s -> %d features", cam, n)
            if n == 0:
                raise ValidationError(f"{cam}: zero features")

        ctx.set_inputs({"detect_manifest": str(detect_manifest)})
        ctx.set_outputs({"per_cam_feature_counts": counts})
        ctx.set_params({
            "model": "solider_swin_small",
            "similarity_thresh": cfg.reid.similarity_thresh,
        })
        ctx.set_upstream([str(detect_manifest)])
