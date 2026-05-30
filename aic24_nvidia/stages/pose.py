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
    # Expose this stage's keypoints at external/Pose.
    return [(cfg.external_root / "Pose", output_dir)]


def _per_cam_pose_files(pose_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    scene_dir = pose_dir / SCENE
    if not scene_dir.exists():
        return out
    for cam_dir in sorted(scene_dir.glob("camera_*")):
        kp = cam_dir / f"{cam_dir.name}_out_keypoint.json"
        if kp.exists():
            out[cam_dir.name] = str(kp)
    return out


def run(cfg: Config, run_dir: Path, run_id: str) -> None:
    assert_vram_free(cfg.vram_min_free_gb)

    reid_manifest = stage_dir(run_dir, "reid") / "manifest.json"

    detect_dir = stage_dir(run_dir, "detect")
    det_scene_dir = detect_dir / SCENE
    cams = sorted(p.stem for p in det_scene_dir.glob("camera_*.txt"))
    if not cams:
        raise ValidationError(f"no detection .txt files in {det_scene_dir}")

    with atomic_stage(run_dir, "pose", run_id=run_id, cfg=cfg, wiring=WIRING) as ctx:
        # external/Pose is wired by WIRING (output_dir during run, final after
        # promotion).
        from ..models import pose_rtmpose
        original = cfg.external_root / "Original"
        pose_rtmpose.run_pose(
            det_scene_dir=det_scene_dir,
            original_scene_dir=original / SCENE,
            pose_out_dir=ctx.work_dir,
            scene=SCENE,
            cams=cams,
        )

        pose_files = _per_cam_pose_files(ctx.work_dir)
        if not pose_files:
            raise ValidationError("no per-camera pose files produced")
        for cam, p in pose_files.items():
            log.info("pose: %s -> %s", cam, p)

        ctx.set_inputs({"reid_manifest": str(reid_manifest), "detect_dir": str(detect_dir)})
        ctx.set_outputs(pose_files)
        ctx.set_params({
            "keypoint_conf": cfg.pose.keypoint_conf,
            "model": "rtmpose-l",
        })
        ctx.set_upstream([str(reid_manifest)])
