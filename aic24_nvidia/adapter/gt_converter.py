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


def _iter_annotations(body: dict):
    """Yield (camera, frame, person_id, world_xy_or_xyz, bbox_xywh) tuples
    from either the real NVIDIA-style schema `{cameras, annotations[]}` or the
    older synthetic schema `{frames[].objects[].cameras[name].bbox_xywh}`.
    """
    if isinstance(body, dict) and "annotations" in body and isinstance(body["annotations"], list):
        for a in body["annotations"]:
            cam = a["camera"]
            frame = int(a["frame"])
            pid = int(a["person_id"])
            wxy = a.get("world_xy") or a.get("world_xyz")
            bbox = a["bbox_2d"]
            yield cam, frame, pid, wxy, bbox
        return
    # Fallback: synthetic schema used by older unit tests.
    for frame in body.get("frames", []):
        fid = int(frame["frame_id"])
        for obj in frame.get("objects", []):
            pid = int(obj["id"])
            wxyz = obj.get("world_xyz")
            for cam, payload in obj.get("cameras", {}).items():
                yield cam, fid, pid, wxyz, payload["bbox_xywh"]


def convert_gt(
    src: Path,
    out_dir: Path,
    scene: str,
    scene_mapping: dict[str, str],
    frame_offset: int,
    max_frames: int,
) -> None:
    """Convert NVIDIA ground_truth.json into per-camera MOT gt.txt + world GT.

    - scene_mapping: {yachiyo_cam_name: source_cam_name}
    - frame_offset: only emit annotations with frame >= frame_offset
    - max_frames: emit up to this many frames (after offset)
    - output MOT frame is 1-indexed: a source frame equal to frame_offset → 1

    Real NVIDIA-style schema:
        {"cameras": {...}, "annotations": [{camera, frame, person_id,
                                            world_xy, bbox_2d}, ...]}
    Older synthetic schema (unit tests):
        {"frames": [{frame_id, objects: [{id, world_xyz, cameras: {name: {bbox_xywh}}}]}]}
    """
    src = Path(src)
    out_dir = Path(out_dir)
    body = json.loads(src.read_text())

    inv_mapping = {v: k for k, v in scene_mapping.items()}

    per_cam: dict[str, list[str]] = {y: [] for y in scene_mapping}
    world_lines: list[str] = []
    seen_world_keys: set[tuple[int, int]] = set()

    frame_max = frame_offset + max_frames  # exclusive upper bound

    for source_cam, frame, pid, wxy, bbox in _iter_annotations(body):
        if frame < frame_offset or frame >= frame_max:
            continue
        if source_cam not in inv_mapping:
            continue
        yachiyo_cam = inv_mapping[source_cam]
        mot_fid = frame - frame_offset + 1
        x, y, w, h = bbox
        per_cam[yachiyo_cam].append(f"{mot_fid},{pid},{x},{y},{w},{h},1,1,1")
        if wxy is not None and (mot_fid, pid) not in seen_world_keys:
            seen_world_keys.add((mot_fid, pid))
            wx, wy = wxy[0], wxy[1]
            world_lines.append(f"{mot_fid},{pid},{wx},{wy}")

    scene_root = out_dir / "Original" / scene
    for yachiyo_cam, lines in per_cam.items():
        cam_dir = scene_root / yachiyo_cam / "gt"
        cam_dir.mkdir(parents=True, exist_ok=True)
        (cam_dir / "gt.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

    # gt_world.txt is placed OUTSIDE Original/scene_NNN/ because YACHIYO's
    # extract_frame.py walks every entry under Original/scene_NNN/ and would
    # treat this file as a camera directory.
    (out_dir / f"{scene}_gt_world.txt").write_text(
        "\n".join(world_lines) + ("\n" if world_lines else "")
    )


def reprojection_check(
    src: Path,
    scene_mapping: dict[str, str],
    project_fn: Callable,
    eps_px: float = 50.0,
) -> GtValidationResult:
    """Sanity-check GT: project world coords via project_fn and compare to
    2D bbox center; count entries within eps_px.

    project_fn(world_xy_or_xyz, source_cam_name) -> (u, v) pixel coordinates
    """
    src = Path(src)
    body = json.loads(src.read_text())
    inv_mapping = {v: k for k, v in scene_mapping.items()}
    total = 0
    matched = 0
    failures: list[dict] = []

    for source_cam, frame, pid, wxy, bbox in _iter_annotations(body):
        if wxy is None:
            continue
        if source_cam not in inv_mapping:
            continue
        bx, by, bw, bh = bbox
        cx = bx + bw / 2
        cy = by + bh / 2
        pu, pv = project_fn(tuple(wxy), source_cam)
        d = math.hypot(pu - cx, pv - cy)
        total += 1
        if d <= eps_px:
            matched += 1
        elif len(failures) < 100:
            failures.append({"frame": frame, "id": pid,
                             "camera": source_cam, "dist_px": d})

    return GtValidationResult(total=total, matched=matched, failures=failures)
