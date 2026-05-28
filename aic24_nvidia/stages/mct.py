from __future__ import annotations
import json
import logging
import shutil
import subprocess
from pathlib import Path

from ..bootstrap import ensure_dir_clean, make_symlink
from ..config import Config
from ..errors import StageError, ValidationError
from ..paths import stage_dir
from ..tracking_params import write_parameters_per_scene, build_tracking_params
from ..world_projection import rewrite_world_coordinates
from .base import atomic_stage, assert_vram_free

log = logging.getLogger(__name__)

SCENE = "scene_001"
SCENE_INT = 1


def _load_camera_map(run_dir: Path) -> dict[int, str]:
    """Read adapted/scene.json -> {numeric_id: nvidia_cam_name}.

    scene.json shape: {scene_name: {yachiyo_cam_name: nvidia_cam_name}}
    where yachiyo_cam_name is "camera_NNNN". Numeric id is
    int(yachiyo_cam_name.split("_")[-1]).
    """
    scene_json = stage_dir(run_dir, "adapted") / "scene.json"
    if not scene_json.exists():
        return {}
    body = json.loads(scene_json.read_text())[SCENE]
    return {int(yk.split("_")[-1]): nvidia for yk, nvidia in body.items()}


def _maybe_rewrite_world_coordinates(
    *,
    cfg_world_projection,
    sct_scene_dir: Path,
    pose_scene_dir: Path,
    calib_root: Path,
    camera_map: dict[int, str],
) -> int:
    if cfg_world_projection.method == "bbox_bottom":
        return 0
    return rewrite_world_coordinates(
        sct_scene_dir=sct_scene_dir,
        pose_scene_dir=pose_scene_dir,
        calib_root=calib_root,
        camera_map=camera_map,
        method=cfg_world_projection.method,
        ankle_min_conf=cfg_world_projection.ankle_min_conf,
    )


def _spans_multiple_cameras(global_json: Path) -> bool:
    body = json.loads(global_json.read_text())
    # whole_tracking_results.json is camera-keyed dict of dicts with GlobalOfflineID.
    cams_per_gid: dict[int, set[str]] = {}
    if not isinstance(body, dict):
        return False
    for cam_key, entries in body.items():
        if not isinstance(entries, dict):
            continue
        for _serial, entry in entries.items():
            gid = entry.get("GlobalOfflineID") if isinstance(entry, dict) else None
            if gid is None:
                continue
            cams_per_gid.setdefault(int(gid), set()).add(str(cam_key))
    return any(len(s) >= 2 for s in cams_per_gid.values())


def run(cfg: Config, run_dir: Path, run_id: str) -> None:
    assert_vram_free(cfg.vram_min_free_gb)

    sct_dir = stage_dir(run_dir, "sct")
    sct_manifest = sct_dir / "manifest.json"
    yachiyo = cfg.yachiyo_root

    with atomic_stage(run_dir, "mct", run_id=run_id) as ctx:
        # Stage SCT outputs into mct.tmp/ so upstream sees them.
        src_scene = sct_dir / SCENE
        dst_scene = ctx.work_dir / SCENE
        if not src_scene.exists():
            raise ValidationError(f"SCT outputs missing at {src_scene}")
        shutil.copytree(src_scene, dst_scene)

        # Optional: override per-detection WorldCoordinate using ankle keypoints.
        # SCPT does not consume WorldCoordinate; this only affects MCT and eval.
        camera_map = _load_camera_map(run_dir)
        rewritten = _maybe_rewrite_world_coordinates(
            cfg_world_projection=cfg.world_projection,
            sct_scene_dir=dst_scene,
            pose_scene_dir=stage_dir(run_dir, "pose") / SCENE,
            calib_root=stage_dir(run_dir, "adapted") / "Original" / SCENE,
            camera_map=camera_map,
        )
        log.info("mct world_projection: method=%s rewrites=%d",
                 cfg.world_projection.method, rewritten)

        log_path = ctx.work_dir / "log.txt"
        for name, target in (
            ("Tracking", ctx.work_dir),
            ("EmbedFeature", stage_dir(run_dir, "reid")),
            ("Detection", stage_dir(run_dir, "detect")),
            ("Pose", stage_dir(run_dir, "pose")),
        ):
            link = yachiyo / name
            ensure_dir_clean(link)
            make_symlink(target, link)

        # Propagate hyperparameters: write parameters_per_scene.py so infer.py
        # reads our config instead of falling back to hardcoded defaults.
        params = build_tracking_params(cfg)
        if not params:
            raise StageError("mct", 1, str(log_path))
        # NOTE: this writes to the shared external/ tree; concurrent runs would race on it.
        write_parameters_per_scene(cfg, yachiyo, SCENE_INT)
        log.info("mct tracking_params: %s", params)

        with open(log_path, "w") as lf:
            proc = subprocess.run(
                ["python", "tracking/infer.py", "-s", str(SCENE_INT), "-mcpt"],
                cwd=yachiyo,
                stdout=lf, stderr=subprocess.STDOUT,
            )
        if proc.returncode != 0:
            raise StageError("mct", proc.returncode, str(log_path))

        # Prefer fixed_whole_tracking_results.json (final, corrected); fall back
        # to the raw whole_tracking_results.json if the correction step was
        # skipped or produced empty global IDs.
        fixed_whole = ctx.work_dir / SCENE / "fixed_whole_tracking_results.json"
        raw_whole = ctx.work_dir / SCENE / "whole_tracking_results.json"
        candidates = []
        if fixed_whole.exists():
            candidates.append(fixed_whole)
        if raw_whole.exists():
            candidates.append(raw_whole)
        whole = None
        for c in candidates:
            if _spans_multiple_cameras(c):
                whole = c
                break
        if whole is None:
            if not candidates:
                raise ValidationError(f"MCT output missing in {ctx.work_dir / SCENE}")
            raise ValidationError("MCT produced no global IDs spanning >=2 cameras")

        ctx.set_inputs({"sct_manifest": str(sct_manifest)})
        ctx.set_outputs({"global_tracks_json": str(whole)})
        ctx.set_params({
            "tracking_params": params,
            "hard_world_gate": cfg.mct.hard_world_gate,
            "world_projection": {
                "method": cfg.world_projection.method,
                "ankle_min_conf": cfg.world_projection.ankle_min_conf,
                "rewrites": rewritten,
            },
            "propagated_via": "parameters_per_scene.py",
        })
        ctx.set_upstream([str(sct_manifest)])

    make_symlink(stage_dir(run_dir, "mct"), cfg.yachiyo_root / "Tracking")
