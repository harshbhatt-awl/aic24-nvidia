from __future__ import annotations
import json
from pathlib import Path

CLAMP_W, CLAMP_H = 1920, 1080


def _write_camera(det_dir, cam, dets_by_frame, img_rel_for_frame):
    """Write camera_NNNN.txt + .json in the exact upstream format.

    dets_by_frame: {frame_id(int, 1-based): [(x1,y1,x2,y2,score(float)), ...]}
    img_rel_for_frame: frame_id -> ImgPath string (relative to run root)
    """
    det_dir = Path(det_dir)
    det_dir.mkdir(parents=True, exist_ok=True)
    txt_path = det_dir / f"{cam}.txt"
    json_path = det_dir / f"{cam}.json"

    u_num = 0
    ret_json: dict[str, dict] = {}
    lines: list[str] = []
    for frame_id in sorted(dets_by_frame):
        for (x1, y1, x2, y2, score) in dets_by_frame[frame_id]:
            x1 = int(max(0, x1)); y1 = int(max(0, y1))
            x2 = int(min(CLAMP_W, x2)); y2 = int(min(CLAMP_H, y2))
            lines.append(f"{cam},{frame_id},1,{x1},{y1},{x2},{y2},{score:.6g}")
            ret_json[str(u_num).zfill(8)] = {
                "Frame": frame_id,
                "ImgPath": img_rel_for_frame(frame_id),
                "NpyPath": "",
                "Coordinate": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "ClusterID": None,
                "OfflineID": None,
            }
            u_num += 1
    txt_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    with open(json_path, "w") as f:
        json.dump(ret_json, f, ensure_ascii=False)


def _make_model(weights):
    """Construct and return a YOLO model for the given weights path."""
    from ultralytics import YOLO
    return YOLO(weights)


def _detect_image(model, img_path, conf_thresh, nms_iou):
    """Run YOLO person detection on a single image.

    Returns a list of (x1, y1, x2, y2, score) float tuples.
    """
    res = model.predict(str(img_path), classes=[0], conf=conf_thresh,
                        iou=nms_iou, imgsz=1920, verbose=False)[0]
    rows = []
    for b in res.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        rows.append((x1, y1, x2, y2, float(b.conf[0])))
    return rows


def run_detection(scene_dir, det_out_dir, cams, conf_thresh, nms_iou,
                  weights="yolo11x.pt", detect=None):
    """Run YOLO11 person detection over Original/<scene>/<cam>/Frame/*.jpg and
    write per-camera detection files via _write_camera.

    detect: optional callable (model, img_path, conf_thresh, nms_iou) -> list[(x1,y1,x2,y2,score)].
            Defaults to _detect_image (module-global). Inject a fake for unit tests.
    """
    if detect is None:
        detect = _detect_image
    model = _make_model(weights)
    scene = Path(scene_dir).name
    for cam in cams:
        frame_dir = Path(scene_dir) / cam / "Frame"
        frame_paths = sorted(frame_dir.glob("*.jpg"))
        dets_by_frame: dict[int, list] = {}
        for fp in frame_paths:
            if not fp.stem.isdigit():
                continue
            frame_id = int(fp.stem)
            rows = detect(model, fp, conf_thresh, nms_iou)
            if rows:
                dets_by_frame[frame_id] = rows
        _write_camera(
            det_dir=Path(det_out_dir) / scene, cam=cam, dets_by_frame=dets_by_frame,
            img_rel_for_frame=lambda f, c=cam, s=scene: f"Original/{s}/{c}/Frame/{f:06d}.jpg",
        )
    del model
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
