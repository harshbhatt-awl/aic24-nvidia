from __future__ import annotations
import logging
import subprocess
from pathlib import Path

from ..bootstrap import ensure_dir_clean, make_symlink
from ..config import Config
from ..errors import StageError, ValidationError
from ..paths import stage_dir
from .base import atomic_stage, assert_vram_free

log = logging.getLogger(__name__)

SCENE = "scene_001"


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
    drid = cfg.external_root / "deep-person-reid"
    if not drid.exists():
        raise FileNotFoundError(f"deep-person-reid not found at {drid} — run bootstrap")
    injected = drid / "torchreid" / "aic24_extract.py"
    if not injected.exists():
        raise FileNotFoundError(f"injected file missing: {injected} — re-run bootstrap")

    with atomic_stage(run_dir, "reid", run_id=run_id) as ctx:
        log_path = ctx.work_dir / "log.txt"

        emb_root = cfg.external_root / "EmbedFeature"
        ensure_dir_clean(emb_root)
        make_symlink(ctx.work_dir, emb_root)

        with open(log_path, "w") as lf:
            proc = subprocess.run(
                ["python3", "torchreid/aic24_extract.py", "-s", SCENE, "../"],
                cwd=drid,
                stdout=lf, stderr=subprocess.STDOUT,
            )
        if proc.returncode != 0:
            raise StageError("reid", proc.returncode, str(log_path))

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
            "similarity_thresh": cfg.reid.similarity_thresh,
            "note": "hyperparam recorded but not propagated to upstream",
        })
        ctx.set_upstream([str(detect_manifest)])

    make_symlink(stage_dir(run_dir, "reid"), cfg.external_root / "EmbedFeature")
