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


def _per_cam_detection_files(detect_dir: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    scene_dir = detect_dir / SCENE
    if not scene_dir.exists():
        return out
    for txt in sorted(scene_dir.glob("camera_*.txt")):
        cam = txt.stem  # camera_NNNN
        out[cam] = {
            "txt": str(txt),
            "json": str(scene_dir / f"{cam}.json"),
        }
    return out


def run(cfg: Config, run_dir: Path, run_id: str) -> None:
    assert_vram_free(cfg.vram_min_free_gb)

    frames_manifest = stage_dir(run_dir, "frames") / "manifest.json"
    botsort = cfg.external_root / "BoT-SORT"
    if not botsort.exists():
        raise FileNotFoundError(f"BoT-SORT not found at {botsort} — run scripts/bootstrap_external.sh")
    injected = botsort / "tools" / "aic24_get_detection.py"
    if not injected.exists():
        raise FileNotFoundError(f"injected file missing: {injected} — re-run bootstrap")

    with atomic_stage(run_dir, "detect", run_id=run_id) as ctx:
        log_path = ctx.work_dir / "log.txt"

        # Symlink external/Detection -> outputs/<run_id>/detect.tmp
        # so upstream writes its results directly into our managed run dir.
        # NOTE: ctx.work_dir is detect.tmp; after the stage completes, the
        # context renames it to detect/. The symlink remains valid post-rename
        # because we re-point it to the final dir on success (handled below).
        det_root = cfg.external_root / "Detection"
        ensure_dir_clean(det_root)
        make_symlink(ctx.work_dir, det_root)

        # Run upstream
        with open(log_path, "w") as lf:
            proc = subprocess.run(
                ["python3", "tools/aic24_get_detection.py", "-s", SCENE, "../"],
                cwd=botsort,
                stdout=lf, stderr=subprocess.STDOUT,
            )
        if proc.returncode != 0:
            raise StageError("detect", proc.returncode, str(log_path))

        det_files = _per_cam_detection_files(ctx.work_dir)
        if not det_files:
            raise ValidationError("no per-camera detection files produced")
        for cam, paths in det_files.items():
            n = sum(1 for _ in open(paths["txt"]))
            log.info("detect: %s -> %d detections", cam, n)
            if n == 0:
                raise ValidationError(f"{cam}: zero detections")

        ctx.set_inputs({"frames_manifest": str(frames_manifest)})
        ctx.set_outputs(det_files)
        ctx.set_params({
            "conf_thresh": cfg.detect.conf_thresh,
            "nms_iou": cfg.detect.nms_iou,
            "note": "hyperparams recorded but not propagated to upstream (hardcoded in aic24_get_detection.py)",
        })
        ctx.set_upstream([str(frames_manifest)])

    # After atomic promotion, re-point the symlink to the final dir.
    make_symlink(stage_dir(run_dir, "detect"), cfg.external_root / "Detection")
