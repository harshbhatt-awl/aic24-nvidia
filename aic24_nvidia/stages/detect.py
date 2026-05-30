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
    # Expose this stage's detection output at external/Detection.
    return [(cfg.external_root / "Detection", output_dir)]


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

    with atomic_stage(run_dir, "detect", run_id=run_id, cfg=cfg, wiring=WIRING) as ctx:
        # external/Detection is wired by WIRING (output_dir during run, final
        # after promotion) — upstream writes its results into our managed run dir.
        from ..models import detect_yolo
        original = cfg.external_root / "Original"
        scene_src = original / SCENE
        cams = sorted(p.name for p in scene_src.iterdir() if p.is_dir())
        detect_yolo.run_detection(
            scene_dir=scene_src,
            det_out_dir=ctx.work_dir,
            cams=cams,
            conf_thresh=cfg.detect.conf_thresh,
            nms_iou=cfg.detect.nms_iou,
            weights="yolo11x.pt",
        )

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
            "model": "yolo11x",
            "conf_thresh": cfg.detect.conf_thresh,
            "nms_iou": cfg.detect.nms_iou,
        })
        ctx.set_upstream([str(frames_manifest)])
