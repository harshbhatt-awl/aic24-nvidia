from __future__ import annotations
import logging
import re
import subprocess
from pathlib import Path

from ..bootstrap import ensure_dir_clean, make_symlink
from ..config import Config
from ..errors import StageError, ValidationError
from ..paths import stage_dir
from .base import atomic_stage, assert_vram_free

log = logging.getLogger(__name__)

SCENE = "scene_001"

HRNET_CONFIG = "configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py"
HRNET_CKPT = (
    "https://download.openmmlab.com/mmpose/top_down/hrnet/"
    "hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth"
)


def _camera_int_from_name(cam_name: str) -> str:
    m = re.fullmatch(r"camera_(\d+)", cam_name)
    if not m:
        raise ValueError(f"unrecognized camera name: {cam_name}")
    return m.group(1)


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
    mmpose = cfg.external_root / "mmpose"
    if not mmpose.exists():
        raise FileNotFoundError(f"mmpose not found at {mmpose} — run bootstrap")
    injected = mmpose / "demo" / "top_down_video_demo_with_track_file.py"
    if not injected.exists():
        raise FileNotFoundError(f"injected file missing: {injected} — re-run bootstrap")

    detect_dir = stage_dir(run_dir, "detect")
    det_scene_dir = detect_dir / SCENE
    cams = sorted(p.stem for p in det_scene_dir.glob("camera_*.txt"))
    if not cams:
        raise ValidationError(f"no detection .txt files in {det_scene_dir}")

    with atomic_stage(run_dir, "pose", run_id=run_id) as ctx:
        log_path = ctx.work_dir / "log.txt"

        pose_root = cfg.external_root / "Pose"
        ensure_dir_clean(pose_root)
        make_symlink(ctx.work_dir, pose_root)

        # mmpose 0.x doesn't install on the main Python 3.14 venv. We use a
        # parallel Python 3.10 venv at .venv-pose with torch 1.13.1+cu117 +
        # mmcv-full 1.7.0 + mmpose 0.29.0.
        pose_python = Path(__file__).resolve().parents[2] / ".venv-pose" / "bin" / "python"
        if not pose_python.exists():
            raise FileNotFoundError(
                f"pose venv missing: {pose_python}. See README for setup."
            )

        with open(log_path, "w") as lf:
            for cam in cams:
                num = _camera_int_from_name(cam)
                det_txt = f"../Detection/{SCENE}/{cam}.txt"
                video = f"../Original/{SCENE}/camera_{num}/video.mp4"
                out_file = f"../Pose/{SCENE}/camera_{num}/camera_{num}_out_keypoint.json"
                cmd = [
                    str(pose_python), "demo/top_down_video_demo_with_track_file.py",
                    det_txt, HRNET_CONFIG, HRNET_CKPT,
                    "--video-path", video,
                    "--out-file", out_file,
                ]
                lf.write(f"\n=== {cam} ===\n{' '.join(cmd)}\n")
                lf.flush()
                proc = subprocess.run(cmd, cwd=mmpose, stdout=lf, stderr=subprocess.STDOUT)
                if proc.returncode != 0:
                    raise StageError("pose", proc.returncode, str(log_path))

        pose_files = _per_cam_pose_files(ctx.work_dir)
        if not pose_files:
            raise ValidationError("no per-camera pose files produced")
        for cam, p in pose_files.items():
            log.info("pose: %s -> %s", cam, p)

        ctx.set_inputs({"reid_manifest": str(reid_manifest), "detect_dir": str(detect_dir)})
        ctx.set_outputs(pose_files)
        ctx.set_params({
            "keypoint_conf": cfg.pose.keypoint_conf,
            "model": "hrnet_w48_coco_256x192",
            "note": "keypoint_conf recorded but not propagated to upstream",
        })
        ctx.set_upstream([str(reid_manifest)])

    make_symlink(stage_dir(run_dir, "pose"), cfg.external_root / "Pose")
