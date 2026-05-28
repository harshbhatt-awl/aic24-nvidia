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
from ..world_tracks import aggregate_world_tracks, smooth_world_tracks, write_world_pred
from ..world_metrics import run_world_eval, load_world_txt
from .base import atomic_stage

log = logging.getLogger(__name__)

SCENE = "scene_001"
SCENE_TRACKEVAL = "S001"


def _coord_to_xyxy(coord) -> tuple[float, float, float, float]:
    """Accept Coordinate as either dict {x1,y1,x2,y2} (real YACHIYO) or list
    [x1,y1,x2,y2] (older synthetic fixtures)."""
    if isinstance(coord, dict):
        return float(coord["x1"]), float(coord["y1"]), float(coord["x2"]), float(coord["y2"])
    return float(coord[0]), float(coord[1]), float(coord[2]), float(coord[3])


def yachiyo_sct_json_to_mot(src_json: Path, dst_txt: Path) -> None:
    """Convert one camera's SCT JSON (serial->{Frame,OfflineID,Coordinate})
    into MOTChallenge-format predictions.

    Coordinate is the YACHIYO native form: dict {x1,y1,x2,y2}. We emit one MOT
    row per (Frame, OfflineID) pair using the converted (x,y,w,h).
    """
    src_json = Path(src_json); dst_txt = Path(dst_txt)
    body = json.loads(src_json.read_text())
    rows: list[tuple[int, int, int, int, int, int]] = []
    for _serial, entry in body.items():
        if not isinstance(entry, dict):
            continue
        f = int(entry["Frame"])
        oid = int(entry["OfflineID"])
        # YACHIYO emits unassigned detections with OfflineID < 0; drop them so
        # MOT/TrackEval doesn't see duplicate "-1" track ids per frame.
        if oid < 0:
            continue
        x1, y1, x2, y2 = _coord_to_xyxy(entry["Coordinate"])
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
            # Skip entries with no GlobalOfflineID (e.g., when the correction
            # step was skipped and only some tracks got assigned a global ID).
            gid_raw = entry.get("GlobalOfflineID")
            if gid_raw is None:
                continue
            gid = int(gid_raw)
            if gid < 0:
                continue
            f = int(entry["Frame"])
            x1, y1, x2, y2 = _coord_to_xyxy(entry["Coordinate"])
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
                      cam_pred: dict[str, Path],
                      seq_length: int = 900) -> None:
    """Materialize a MOTChallenge-style tree TrackEval can consume:
        out_dir/gt/<scene>-train/<scene>-<cam>/gt/gt.txt
        out_dir/gt/<scene>-train/<scene>-<cam>/seqinfo.ini
        out_dir/gt/seqmaps/<scene>-train.txt
        out_dir/trackers/<scene>-train/yachiyo/data/<scene>-<cam>.txt
    """
    split = "train"
    gt_root = out_dir / "gt" / f"{scene}-{split}"
    trk_root = out_dir / "trackers" / f"{scene}-{split}" / "yachiyo" / "data"
    seqmap_dir = out_dir / "gt" / "seqmaps"

    seq_names = []
    for cam, gt_path in cam_gt.items():
        seq = f"{scene}-{cam}"
        seq_names.append(seq)
        dst_gt = gt_root / seq / "gt"
        dst_gt.mkdir(parents=True, exist_ok=True)
        shutil.copy(gt_path, dst_gt / "gt.txt")
        # MOTChallenge seqinfo.ini stub
        (gt_root / seq / "seqinfo.ini").write_text(
            f"[Sequence]\nname={seq}\nseqLength={seq_length}\nimWidth=1920\nimHeight=1080\nframeRate=30\n"
        )
    for cam, pred_path in cam_pred.items():
        seq = f"{scene}-{cam}"
        trk_root.mkdir(parents=True, exist_ok=True)
        shutil.copy(pred_path, trk_root / f"{seq}.txt")

    seqmap_dir.mkdir(parents=True, exist_ok=True)
    (seqmap_dir / f"{scene}-{split}.txt").write_text("name\n" + "\n".join(seq_names) + "\n")


def _summarize_metrics(metrics: dict) -> str:
    lines = ["# Evaluation summary\n"]
    for seq, m in metrics.items():
        if seq == "mct_world":
            continue
        lines.append(f"## {seq}")
        for key in ("HOTA", "IDF1", "MOTA"):
            if key in m:
                lines.append(f"- {key}: {m[key]:.4f}")
        lines.append("")
    mw = metrics.get("mct_world")
    if isinstance(mw, dict):
        lines.append("## MCT (3D world)")
        if "skipped" in mw:
            lines.append(f"- skipped: {mw['skipped']}")
            if "gt_range" in mw:
                lines.append(f"  - gt_range: {mw['gt_range'][0]}..{mw['gt_range'][1]}")
                lines.append(f"  - pred_range: {mw['pred_range'][0]}..{mw['pred_range'][1]}")
        else:
            for key in ("HOTA", "DetA", "AssA", "IDF1", "MOTA"):
                if key in mw:
                    lines.append(f"- {key}: {mw[key]:.4f}")
            lines.append(f"- d_max_m: {mw.get('d_max_m')}")
            lines.append(f"- frames_evaluated: {mw.get('frames_evaluated')}")
        lines.append("")
    return "\n".join(lines)


def _run_trackeval(mot_root: Path, scene: str, log_path: Path) -> dict:
    # pip-installed trackeval doesn't include the runner scripts; we use the
    # cloned external/TrackEval/scripts/run_mot_challenge.py instead.
    trackeval_script = Path(__file__).resolve().parents[2] / "external" / "TrackEval" / "scripts" / "run_mot_challenge.py"
    cmd = [
        sys.executable, str(trackeval_script),
        "--BENCHMARK", scene,
        "--GT_FOLDER", str(mot_root / "gt"),
        "--TRACKERS_FOLDER", str(mot_root / "trackers"),
        "--TRACKERS_TO_EVAL", "yachiyo",
        "--METRICS", "HOTA", "CLEAR", "Identity",
        "--USE_PARALLEL", "False",
        "--PRINT_RESULTS", "False",
        "--OUTPUT_DETAILED", "True",
        "--PLOT_CURVES", "False",
        "--SPLIT_TO_EVAL", "train",
        "--DO_PREPROC", "False",
    ]
    with open(log_path, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise StageError("evaluate", proc.returncode, str(log_path))

    results: dict = {}
    # TrackEval writes to TRACKERS_FOLDER/<scene>-train/yachiyo/pedestrian_detailed.csv
    summary_csv = (mot_root / "trackers" / f"{scene}-train" / "yachiyo"
                   / "pedestrian_detailed.csv")
    if not summary_csv.exists():
        # Fallback search in case TrackEval used a different layout.
        candidates = list((mot_root / "trackers").rglob("pedestrian_detailed.csv"))
        if not candidates:
            raise StageError("evaluate", 0, str(log_path))
        summary_csv = candidates[0]

    with open(summary_csv) as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            seq = row.get("seq", "")
            if not seq:
                continue
            # TrackEval's "primary" HOTA column is HOTA(0); the value-at-each-
            # threshold columns are HOTA___5, HOTA___10, .... We use HOTA(0).
            def _maybe(key):
                v = row.get(key, "")
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return float("nan")
            results[seq] = {
                "HOTA": _maybe("HOTA(0)"),
                "IDF1": _maybe("IDF1"),
                "MOTA": _maybe("MOTA"),
                "MOTP": _maybe("MOTP"),
                "IDR":  _maybe("IDR"),
                "IDP":  _maybe("IDP"),
                "CLR_F1": _maybe("CLR_F1"),
            }
    return results


def _eval_mct_world(cfg, ctx, mct_global, adapted_root):
    """Build world predictions from MCT JSON and score against gt_world.txt.

    Returns a metrics dict, or {"skipped": reason} when it can't run.
    """
    gt_world = adapted_root / f"{SCENE}_gt_world.txt"
    if not gt_world.exists() or gt_world.stat().st_size == 0:
        return {"skipped": "no gt_world.txt"}
    if not mct_global:
        return {"skipped": "no MCT global tracks"}

    try:
        rows, dropped = aggregate_world_tracks(Path(mct_global))
        if not rows:
            return {"skipped": "MCT produced no valid world points"}
        rows = smooth_world_tracks(
            rows,
            method=cfg.world_smoothing.method,
            ema_alpha=cfg.world_smoothing.ema_alpha,
        )
        pred_txt = ctx.work_dir / "mct_world_pred.txt"
        write_world_pred(rows, pred_txt)

        gt_frames = set(load_world_txt(gt_world).keys())
        pred_frames = {r[0] for r in rows}
        if not (gt_frames & pred_frames):
            log.warning("world eval: GT frames %s..%s do not overlap pred frames %s..%s",
                        min(gt_frames), max(gt_frames), min(pred_frames), max(pred_frames))
            return {"skipped": "GT/pred frame ranges do not overlap",
                    "gt_range": [min(gt_frames), max(gt_frames)],
                    "pred_range": [min(pred_frames), max(pred_frames)]}

        trackeval_root = Path(__file__).resolve().parents[2] / "external" / "TrackEval"
        m = run_world_eval(gt_world, pred_txt, cfg.eval.world_d_max, trackeval_root, seq_name=SCENE)
        m["dropped_detections"] = dropped
        m["frames_evaluated"] = len(gt_frames & pred_frames)
        return m
    except Exception as exc:
        log.exception("world MCT eval failed; reporting skipped")
        return {"skipped": f"exception: {exc}"}


def run(cfg: Config, run_dir: Path, run_id: str) -> None:
    adapted_root = stage_dir(run_dir, "adapted")
    sct_manifest_path = stage_dir(run_dir, "sct") / "manifest.json"
    mct_manifest_path = stage_dir(run_dir, "mct") / "manifest.json"
    sct_manifest = read_manifest(sct_manifest_path)
    mct_manifest = read_manifest(mct_manifest_path)

    import re
    cam_gt: dict[str, Path] = {}
    cam_pred_sct: dict[str, Path] = {}

    with atomic_stage(run_dir, "evaluate", run_id=run_id) as ctx:
        mot_root = ctx.work_dir / "mot"

        # Build SCT preds (per-camera MOT)
        sct_pred_dir = ctx.work_dir / "sct_pred"
        for cam_stem, sct_json in sct_manifest.outputs.items():
            # cam_stem is e.g. "camera390_tracking_results"; extract the digits.
            m = re.search(r"camera(\d+)", cam_stem)
            if not m:
                log.warning("cannot extract camera number from %s; skipping", cam_stem)
                continue
            num = int(m.group(1))
            cam_name = f"camera_{num:04d}"
            gt = adapted_root / "Original" / SCENE / cam_name / "gt" / "gt.txt"
            if not gt.exists() or gt.stat().st_size == 0:
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

        # Also convert MCT JSON for record-keeping if MCT actually ran.
        # If MCT was stubbed (skipped due to missing pose), the manifest's outputs
        # won't have global_tracks_json — we skip the conversion in that case.
        mct_pred_dir = ctx.work_dir / "mct_pred"
        mct_global = mct_manifest.outputs.get("global_tracks_json") if mct_manifest.outputs else None
        if mct_global:
            yachiyo_mct_json_to_mot(Path(mct_global), mct_pred_dir)
        else:
            log.info("MCT outputs not present (likely stubbed); skipping MCT conversion")

        metrics["mct_world"] = _eval_mct_world(cfg, ctx, mct_global, adapted_root)

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
        ctx.set_params({
            "trackeval": "MOTChallenge + NvidiaMTMCWorld",
            "scope": "per-camera SCT + scene 3D-world MCT",
            "world_d_max": cfg.eval.world_d_max,
            "world_smoothing": {
                "method": cfg.world_smoothing.method,
                "ema_alpha": cfg.world_smoothing.ema_alpha,
            },
        })
        ctx.set_upstream([str(sct_manifest_path), str(mct_manifest_path)])
