from __future__ import annotations
import csv
import json
import logging
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from ..bootstrap import make_symlink
from ..config import Config
from ..errors import StageError, ValidationError
from ..manifest import read_manifest
from ..paths import stage_dir
from .base import atomic_stage

log = logging.getLogger(__name__)

SCENE = "scene_001"
SCENE_TRACKEVAL = "S001"


def yachiyo_sct_json_to_mot(src_json: Path, dst_txt: Path) -> None:
    """Convert one camera's SCT JSON (serial->{Frame,OfflineID,Coordinate=[x1,y1,x2,y2]})
    into MOTChallenge-format predictions.
    """
    src_json = Path(src_json); dst_txt = Path(dst_txt)
    body = json.loads(src_json.read_text())
    rows: list[tuple[int, int, int, int, int, int]] = []
    for _serial, entry in body.items():
        if not isinstance(entry, dict):
            continue
        f = int(entry["Frame"])
        oid = int(entry["OfflineID"])
        x1, y1, x2, y2 = entry["Coordinate"]
        w = int(round(x2 - x1)); h = int(round(y2 - y1))
        rows.append((f, oid, int(round(x1)), int(round(y1)), w, h))
    rows.sort()
    dst_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_txt, "w") as f:
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]},1,-1,-1,-1\n")


def yachiyo_mct_json_to_mot(src_json: Path, dst_dir: Path) -> None:
    """Split MCT JSON (cam_id->serial->{Frame,GlobalOfflineID,Coordinate}) into
    per-camera MOT-format prediction files, using GlobalOfflineID as the track id.

    Camera IDs in the JSON are stringified ints; we map to camera_NNNN (zero-padded
    to 4 digits) to match the adapter's naming convention.
    """
    src_json = Path(src_json); dst_dir = Path(dst_dir)
    body = json.loads(src_json.read_text())
    dst_dir.mkdir(parents=True, exist_ok=True)
    per_cam: dict[str, list[tuple[int, int, int, int, int, int]]] = defaultdict(list)
    for cam_key, entries in body.items():
        cam_name = f"camera_{int(cam_key):04d}"
        if not isinstance(entries, dict):
            continue
        for _serial, entry in entries.items():
            if not isinstance(entry, dict):
                continue
            f = int(entry["Frame"])
            gid = int(entry["GlobalOfflineID"])
            x1, y1, x2, y2 = entry["Coordinate"]
            w = int(round(x2 - x1)); h = int(round(y2 - y1))
            per_cam[cam_name].append(
                (f, gid, int(round(x1)), int(round(y1)), w, h)
            )
    for cam, rows in per_cam.items():
        rows.sort()
        with open(dst_dir / f"{cam}.txt", "w") as f:
            for r in rows:
                f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]},1,-1,-1,-1\n")


def _build_mot_layout(out_dir: Path, scene: str,
                      cam_gt: dict[str, Path],
                      cam_pred: dict[str, Path]) -> None:
    for cam, gt_path in cam_gt.items():
        dst = out_dir / "gt" / scene / f"{scene}-{cam}" / "gt"
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(gt_path, dst / "gt.txt")
    for cam, pred_path in cam_pred.items():
        dst = out_dir / "trackers" / scene / "yachiyo" / "data"
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(pred_path, dst / f"{scene}-{cam}.txt")


def _summarize_metrics(metrics: dict) -> str:
    lines = ["# Evaluation summary\n"]
    for seq, m in metrics.items():
        lines.append(f"## {seq}")
        for key in ("HOTA", "IDF1", "MOTA"):
            if key in m:
                lines.append(f"- {key}: {m[key]:.4f}")
        lines.append("")
    return "\n".join(lines)


def _run_trackeval(mot_root: Path, scene: str, log_path: Path) -> dict:
    cmd = [
        sys.executable, "-m", "trackeval.scripts.run_mot_challenge",
        "--BENCHMARK", scene,
        "--GT_FOLDER", str(mot_root / "gt"),
        "--TRACKERS_FOLDER", str(mot_root / "trackers"),
        "--TRACKERS_TO_EVAL", "yachiyo",
        "--METRICS", "HOTA", "CLEAR", "Identity",
        "--USE_PARALLEL", "False",
        "--PRINT_RESULTS", "False",
        "--OUTPUT_DETAILED", "True",
    ]
    with open(log_path, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise StageError("evaluate", proc.returncode, str(log_path))

    results: dict = {}
    summary_dir = mot_root / "trackers" / scene / "yachiyo"
    for f in sorted(summary_dir.glob("pedestrian_detailed.csv")):
        with open(f) as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                seq = row.get("seq", "")
                if not seq:
                    continue
                results[seq] = {
                    "HOTA": float(row.get("HOTA", "nan")),
                    "IDF1": float(row.get("IDF1", "nan")),
                    "MOTA": float(row.get("MOTA", "nan")),
                }
    return results


def run(cfg: Config, run_dir: Path, run_id: str) -> None:
    adapted_root = stage_dir(run_dir, "adapted")
    sct_manifest_path = stage_dir(run_dir, "sct") / "manifest.json"
    mct_manifest_path = stage_dir(run_dir, "mct") / "manifest.json"
    sct_manifest = read_manifest(sct_manifest_path)
    mct_manifest = read_manifest(mct_manifest_path)

    cam_gt: dict[str, Path] = {}
    cam_pred_sct: dict[str, Path] = {}

    with atomic_stage(run_dir, "evaluate", run_id=run_id) as ctx:
        mot_root = ctx.work_dir / "mot"

        # Build SCT preds (per-camera MOT)
        sct_pred_dir = ctx.work_dir / "sct_pred"
        for cam_stem, sct_json in sct_manifest.outputs.items():
            # cam_stem is "camera<NNN>" (zero-padded to 3 digits, no underscore).
            # Convert to "camera_NNNN" (zero-padded to 4 digits) for layout consistency.
            num = int(cam_stem.replace("camera", ""))
            cam_name = f"camera_{num:04d}"
            gt = adapted_root / "Original" / SCENE / cam_name / "gt" / "gt.txt"
            if not gt.exists():
                log.warning("no GT for %s; skipping", cam_name)
                continue
            mot_pred = sct_pred_dir / f"{cam_name}.txt"
            yachiyo_sct_json_to_mot(Path(sct_json), mot_pred)
            cam_gt[cam_name] = gt
            cam_pred_sct[cam_name] = mot_pred

        if not cam_gt:
            raise ValidationError("no cameras with GT; cannot evaluate")

        _build_mot_layout(mot_root, SCENE_TRACKEVAL, cam_gt, cam_pred_sct)
        log_path = ctx.work_dir / "log.txt"
        metrics = _run_trackeval(mot_root, SCENE_TRACKEVAL, log_path)

        # Also convert MCT JSON for record-keeping (not currently evaluated separately
        # because MCT HOTA needs a different TrackEval dataset adapter; v2 work).
        mct_pred_dir = ctx.work_dir / "mct_pred"
        yachiyo_mct_json_to_mot(Path(mct_manifest.outputs["global_tracks_json"]), mct_pred_dir)

        (ctx.work_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        (ctx.work_dir / "summary.md").write_text(_summarize_metrics(metrics))

        ctx.set_inputs({
            "sct_manifest": str(sct_manifest_path),
            "mct_manifest": str(mct_manifest_path),
        })
        ctx.set_outputs({
            "metrics": str(ctx.work_dir / "metrics.json"),
            "summary": str(ctx.work_dir / "summary.md"),
            "sct_pred_dir": str(sct_pred_dir),
            "mct_pred_dir": str(mct_pred_dir),
        })
        ctx.set_params({"trackeval": "MOTChallenge", "scope": "per-camera SCT only in v1"})
        ctx.set_upstream([str(sct_manifest_path), str(mct_manifest_path)])
