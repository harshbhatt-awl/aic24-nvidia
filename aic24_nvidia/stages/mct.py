from __future__ import annotations
import logging
import shutil
import subprocess
from pathlib import Path

from ..bootstrap import ensure_dir_clean, make_symlink
from ..config import Config
from ..errors import StageError, ValidationError
from ..paths import stage_dir
from .base import atomic_stage, assert_vram_free

log = logging.getLogger(__name__)

SCENE = "scene_001"
SCENE_INT = 1


def _spans_multiple_cameras(global_json: Path) -> bool:
    import json
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

        log_path = ctx.work_dir / "log.txt"
        for name, target in (
            ("Tracking", ctx.work_dir),
            ("EmbedFeature", stage_dir(run_dir, "reid")),
            ("Detection", stage_dir(run_dir, "detect")),
        ):
            link = yachiyo / name
            ensure_dir_clean(link)
            make_symlink(target, link)

        with open(log_path, "w") as lf:
            proc = subprocess.run(
                ["python", "tracking/infer.py", "-s", str(SCENE_INT), "-mcpt"],
                cwd=yachiyo,
                stdout=lf, stderr=subprocess.STDOUT,
            )
        if proc.returncode != 0:
            raise StageError("mct", proc.returncode, str(log_path))

        whole = ctx.work_dir / SCENE / "fixed_whole_tracking_results.json"
        if not whole.exists():
            # fall back to non-fixed
            whole = ctx.work_dir / SCENE / "whole_tracking_results.json"
        if not whole.exists():
            raise ValidationError(f"MCT output missing in {ctx.work_dir / SCENE}")
        if not _spans_multiple_cameras(whole):
            raise ValidationError("MCT produced no global IDs spanning >=2 cameras")

        ctx.set_inputs({"sct_manifest": str(sct_manifest)})
        ctx.set_outputs({"global_tracks_json": str(whole)})
        ctx.set_params({
            "cluster_thresh": cfg.mct.cluster_thresh,
            "min_track_len": cfg.mct.min_track_len,
            "note": "hyperparams recorded but not propagated to upstream",
        })
        ctx.set_upstream([str(sct_manifest)])

    make_symlink(stage_dir(run_dir, "mct"), cfg.yachiyo_root / "Tracking")
