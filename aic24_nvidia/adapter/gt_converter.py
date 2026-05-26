from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class GtValidationResult:
    total: int
    matched: int
    failures: list[dict]

    @property
    def match_ratio(self) -> float:
        return self.matched / self.total if self.total else 1.0


def convert_gt(
    src: Path,
    out_dir: Path,
    scene: str,
    scene_mapping: dict[str, str],
    frame_offset: int,
    max_frames: int,
) -> None:
    """Convert NVIDIA ground_truth.json into per-camera MOT gt.txt + world GT.

    - scene_mapping: {yachiyo_cam_name: nvidia_cam_name}
    - frame_offset: only emit frames with frame_id >= frame_offset
    - max_frames: emit up to this many frames (after offset)
    - output MOT frame is 1-indexed from the first emitted frame
    """
    src = Path(src)
    out_dir = Path(out_dir)
    body = json.loads(src.read_text())
    frames = body.get("frames", [])

    inv_mapping = {v: k for k, v in scene_mapping.items()}  # nvidia -> yachiyo

    per_cam: dict[str, list[str]] = {y: [] for y in scene_mapping}
    world_lines: list[str] = []

    emitted = 0
    for frame in frames:
        fid = int(frame["frame_id"])
        if fid < frame_offset:
            continue
        if emitted >= max_frames:
            break
        mot_fid = emitted + 1
        emitted += 1
        for obj in frame.get("objects", []):
            oid = int(obj["id"])
            wxyz = obj.get("world_xyz")
            if wxyz is not None:
                world_lines.append(f"{mot_fid},{oid},{wxyz[0]},{wxyz[1]}")
            for nvidia_cam, payload in obj.get("cameras", {}).items():
                if nvidia_cam not in inv_mapping:
                    continue
                yachiyo_cam = inv_mapping[nvidia_cam]
                x, y, w, h = payload["bbox_xywh"]
                per_cam[yachiyo_cam].append(
                    f"{mot_fid},{oid},{x},{y},{w},{h},1,1,1"
                )

    scene_root = out_dir / "Original" / scene
    for yachiyo_cam, lines in per_cam.items():
        cam_dir = scene_root / yachiyo_cam / "gt"
        cam_dir.mkdir(parents=True, exist_ok=True)
        (cam_dir / "gt.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

    (scene_root / "gt_world.txt").write_text(
        "\n".join(world_lines) + ("\n" if world_lines else "")
    )


def reprojection_check(
    src: Path,
    scene_mapping: dict[str, str],
    project_fn: Callable[[tuple, str], tuple[float, float]],
    eps_px: float = 50.0,
) -> GtValidationResult:
    """Sanity-check NVIDIA GT: project world_xyz via project_fn and compare to
    2D bbox center; count entries within eps_px.

    project_fn(world_xyz, nvidia_cam_name) -> (u, v) pixel coordinates
    """
    src = Path(src)
    body = json.loads(src.read_text())
    inv_mapping = {v: k for k, v in scene_mapping.items()}
    total = 0
    matched = 0
    failures: list[dict] = []
    for frame in body.get("frames", []):
        fid = int(frame["frame_id"])
        for obj in frame.get("objects", []):
            wxyz = obj.get("world_xyz")
            if wxyz is None:
                continue
            for nvidia_cam, payload in obj.get("cameras", {}).items():
                if nvidia_cam not in inv_mapping:
                    continue
                bx, by, bw, bh = payload["bbox_xywh"]
                cx = bx + bw / 2
                cy = by + bh / 2
                pu, pv = project_fn(tuple(wxyz), nvidia_cam)
                d = math.hypot(pu - cx, pv - cy)
                total += 1
                if d <= eps_px:
                    matched += 1
                else:
                    failures.append({"frame": fid, "id": obj["id"],
                                     "camera": nvidia_cam, "dist_px": d})
    return GtValidationResult(total=total, matched=matched, failures=failures)
