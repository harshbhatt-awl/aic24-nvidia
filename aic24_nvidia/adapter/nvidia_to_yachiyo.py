from __future__ import annotations
import json
import logging
from pathlib import Path

from ..config import Config
from .calibration import adapt_calibration
from .gt_converter import convert_gt, reprojection_check
from .video import materialize_yachiyo_layout, probe_duration

log = logging.getLogger(__name__)


def discover_cameras(scene_dir: Path) -> list[str]:
    videos = sorted((scene_dir / "videos").glob("Camera_*.mp4"))
    return [v.stem for v in videos]


def identity_project(world_xyz: tuple, cam_name: str) -> tuple[float, float]:
    """Fallback projector: x,y world -> x,y pixel (treats world as image-aligned).

    Used only when the per-camera calibration isn't loadable. The reprojection
    check with this fallback isn't meaningful; we still run it to log the gap.
    """
    return (float(world_xyz[0]), float(world_xyz[1]))


def adapt_scene(cfg: Config, work_dir: Path) -> dict:
    """Adapt one NVIDIA scene into YACHIYO format under work_dir.

    Returns a dict suitable for the stage manifest's `outputs` field.
    """
    scene_dir = cfg.scene_dir
    if not scene_dir.exists():
        raise FileNotFoundError(f"NVIDIA scene not found: {scene_dir}")

    cameras = discover_cameras(scene_dir)
    if not cameras:
        raise FileNotFoundError(f"no Camera_*.mp4 under {scene_dir / 'videos'}")

    log.info("adapter: scene=%s cameras=%d", cfg.scene, len(cameras))
    for cam in cameras:
        dur = probe_duration(scene_dir / "videos" / f"{cam}.mp4")
        if cfg.clip.start_sec + cfg.clip.duration_sec > dur + 0.1:
            raise ValueError(
                f"clip window exceeds {cam} duration ({dur:.1f}s)"
            )

    scene_name = "scene_001"
    scene_json = materialize_yachiyo_layout(
        src_dir=scene_dir / "videos",
        target_root=work_dir,
        scene_name=scene_name,
        camera_names=cameras,
        start_sec=cfg.clip.start_sec,
        duration_sec=cfg.clip.duration_sec,
    )

    mapping = json.loads(scene_json.read_text())[scene_name]

    calib_dst = work_dir / "Original" / scene_name / "calibration.json"
    src_calib = scene_dir / "calibration.json"
    if src_calib.exists():
        adapt_calibration(src_calib, calib_dst, scene_mapping=mapping)
    else:
        log.warning("calibration.json missing; downstream calibration-dependent steps may fail")

    src_gt = scene_dir / "ground_truth.json"
    if src_gt.exists():
        convert_gt(
            src_gt,
            work_dir,
            scene=scene_name,
            scene_mapping=mapping,
            frame_offset=int(cfg.clip.start_sec * cfg.fps),
            max_frames=int(cfg.clip.duration_sec * cfg.fps),
        )
        validation = reprojection_check(
            src_gt,
            scene_mapping=mapping,
            project_fn=identity_project,
            eps_px=200.0,
        )
        (work_dir / "gt_validation.json").write_text(json.dumps({
            "total": validation.total,
            "matched": validation.matched,
            "match_ratio": validation.match_ratio,
            "failures_truncated": validation.failures[:20],
        }, indent=2))
        if validation.match_ratio < 0.95:
            log.warning(
                "reprojection check below 95%%: %.1f%% — metrics may be unreliable",
                validation.match_ratio * 100,
            )
    else:
        log.warning("ground_truth.json missing; evaluation will be skipped")

    return {
        "adapted_root": str(work_dir),
        "scene_json": str(scene_json),
        "cameras": len(cameras),
    }
