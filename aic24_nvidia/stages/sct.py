from __future__ import annotations
import json
import logging
import subprocess
from pathlib import Path

from ..bootstrap import patch_scene_camera_map
from ..config import Config
from ..errors import StageError, ValidationError
from ..paths import stage_dir
from ..tracking_params import write_parameters_per_scene, build_tracking_params
from .base import atomic_stage, assert_vram_free

log = logging.getLogger(__name__)

SCENE = "scene_001"
SCENE_INT = 1


def _camera_ids_from_scene_json(scene_json: Path) -> list[int]:
    body = json.loads(scene_json.read_text())[SCENE]
    # body is {yachiyo_cam_name: nvidia_cam_name}; we need integer IDs for YACHIYO.
    # yachiyo_cam_name is "camera_NNNN" → int N.
    return sorted(int(name.split("_")[-1]) for name in body)


def _sct_files(work_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    scene_dir = work_dir / SCENE
    if not scene_dir.exists():
        return out
    # Upstream emits camera<NNN>_tracking_results.json (zero-padded to 3 digits)
    # plus fixed_camera<NNN>_tracking_results.json. We prefer fixed_.
    for f in sorted(scene_dir.glob("fixed_camera*_tracking_results.json")):
        out[f.stem.replace("fixed_", "")] = str(f)
    return out


def WIRING(run_dir: Path, cfg: Config, output_dir: Path):
    # Upstream infer.py (CWD=yachiyo) reads EmbedFeature/ + Detection/ and writes
    # Tracking/. Expose this stage's output as Tracking and wire the inputs.
    y = cfg.yachiyo_root
    return [
        (y / "Tracking", output_dir),
        (y / "EmbedFeature", stage_dir(run_dir, "reid")),
        (y / "Detection", stage_dir(run_dir, "detect")),
    ]


def run(cfg: Config, run_dir: Path, run_id: str) -> None:
    assert_vram_free(cfg.vram_min_free_gb)

    pose_manifest = stage_dir(run_dir, "pose") / "manifest.json"
    yachiyo = cfg.yachiyo_root
    entry = yachiyo / "tracking" / "infer.py"
    if not entry.exists():
        raise FileNotFoundError(f"YACHIYO entry missing: {entry}")

    # Patch scene_2_camera_id_file.json for our scene.
    scene_json_path = stage_dir(run_dir, "adapted") / "scene.json"
    if not scene_json_path.exists():
        raise FileNotFoundError(f"adapter scene.json missing: {scene_json_path}")
    camera_ids = _camera_ids_from_scene_json(scene_json_path)
    patch_scene_camera_map(
        yachiyo / "tracking" / "config" / "scene_2_camera_id_file.json",
        scene=SCENE, camera_ids=camera_ids,
    )

    with atomic_stage(run_dir, "sct", run_id=run_id, cfg=cfg, wiring=WIRING) as ctx:
        log_path = ctx.work_dir / "log.txt"
        # yachiyo/{Tracking,EmbedFeature,Detection} are wired by WIRING before
        # this body runs (Tracking -> output_dir; inputs -> reid/detect finals).
        params = build_tracking_params(cfg)
        if not params:
            raise StageError("sct", 1, str(log_path))
        # NOTE: this writes to the shared external/ tree; concurrent runs would race on it.
        write_parameters_per_scene(cfg, yachiyo, SCENE_INT)
        log.info("sct tracking_params: %s", params)

        with open(log_path, "w") as lf:
            proc = subprocess.run(
                ["python", "tracking/infer.py", "-s", str(SCENE_INT), "-scpt"],
                cwd=yachiyo,
                stdout=lf, stderr=subprocess.STDOUT,
            )
        if proc.returncode != 0:
            raise StageError("sct", proc.returncode, str(log_path))

        outputs = _sct_files(ctx.work_dir)
        if not outputs:
            raise ValidationError("no fixed_camera*_tracking_results.json files produced")
        for cam_stem, p in outputs.items():
            log.info("sct: %s -> %s", cam_stem, p)

        ctx.set_inputs({"pose_manifest": str(pose_manifest)})
        ctx.set_outputs(outputs)
        ctx.set_params({
            "tracking_params": params,
            "track_buffer": cfg.sct.track_buffer,
            "match_thresh": cfg.sct.match_thresh,
            "patched_scene_camera_map": True,
            "propagated_via": "parameters_per_scene.py",
        })
        ctx.set_upstream([str(pose_manifest)])
