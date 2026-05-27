from __future__ import annotations
import json
import logging
from pathlib import Path

from ..config import Config
from .calibration import adapt_calibration, write_per_camera_calibration
from .gt_converter import convert_gt, reprojection_check
from .video import materialize_yachiyo_layout, probe_duration

log = logging.getLogger(__name__)


def discover_cameras(scene_dir: Path) -> list[str]:
    """Discover camera files in the videos directory.

    Real NVIDIA Warehouse scenes (as pre-processed by the AIC23 sibling project)
    name videos `camera_NNNN.mp4` (lowercase). Camera IDs are NOT necessarily
    sequential (e.g., Warehouse_001 has cameras 0390..0396).
    """
    videos = sorted((scene_dir / "videos").glob("camera_*.mp4"))
    return [v.stem for v in videos]


def _projector_from_calibration(calib_body: dict):
    """Build a project_fn(world_xy_or_xyz, cam_name) -> (u, v) from real
    NVIDIA-format calibration data: {"cameras": {cam_name: {K, R, t}, ...}}.

    Applies the pinhole model: x_img = K [R | t] X_world (X_world in homog).
    Real GT carries 2D world_xy (z=0 floor plane) — we pad z=0 if needed.
    """
    import numpy as np
    cams = calib_body["cameras"]

    def project(world_xy_or_xyz, cam_name: str) -> tuple[float, float]:
        cam = cams[cam_name]
        K = np.array(cam["K"], dtype=float)
        R = np.array(cam["R"], dtype=float)
        t = np.array(cam["t"], dtype=float)
        w = list(world_xy_or_xyz)
        if len(w) == 2:
            w.append(0.0)
        X = np.array(w, dtype=float)
        cam_coords = R @ X + t
        if cam_coords[2] <= 1e-6:
            return (float("inf"), float("inf"))
        img = K @ cam_coords
        return (float(img[0] / img[2]), float(img[1] / img[2]))

    return project


def identity_project(world_xy_or_xyz, cam_name: str) -> tuple[float, float]:
    """Fallback projector when calibration isn't available — treats world x,y
    as image pixel coordinates. Reprojection check with this fallback is
    meaningless; we still run it to record the gap.
    """
    return (float(world_xy_or_xyz[0]), float(world_xy_or_xyz[1]))


def adapt_scene(cfg: Config, work_dir: Path) -> dict:
    """Adapt one NVIDIA scene into YACHIYO format under work_dir.

    Returns a dict suitable for the stage manifest's `outputs` field.
    """
    scene_dir = cfg.scene_dir
    if not scene_dir.exists():
        raise FileNotFoundError(f"NVIDIA scene not found: {scene_dir}")

    cameras = discover_cameras(scene_dir)
    if not cameras:
        raise FileNotFoundError(f"no camera_*.mp4 under {scene_dir / 'videos'}")

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

    # Calibration and gt_world.txt are placed OUTSIDE Original/scene_NNN/ because
    # YACHIYO's upstream extract_frame.py walks every entry in Original/scene_NNN/
    # and tries to create a Frame/ subdirectory inside each — which would fail
    # on regular files. We keep Original/scene_NNN/ containing only camera dirs.
    calib_dst = work_dir / f"{scene_name}_calibration.json"
    src_calib = scene_dir / "calibration.json"
    calib_body: dict | None = None
    if src_calib.exists():
        adapt_calibration(src_calib, calib_dst, scene_mapping=mapping)
        calib_body = json.loads(src_calib.read_text())
        # Write per-camera calibration.json so YACHIYO's SCT stage can populate
        # WorldCoordinate for each detection (required by MCT).
        original_scene_dir = work_dir / "Original" / scene_name
        written = write_per_camera_calibration(calib_body, original_scene_dir, cameras)
        log.info("adapter: wrote %d per-camera calibration files", len(written))
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
        project_fn = _projector_from_calibration(calib_body) if calib_body else identity_project
        eps_px = 50.0 if calib_body else 200.0
        validation = reprojection_check(
            src_gt,
            scene_mapping=mapping,
            project_fn=project_fn,
            eps_px=eps_px,
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
