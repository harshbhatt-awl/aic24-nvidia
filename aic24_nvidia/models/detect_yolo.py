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


class YoloDetector:
    """Default DetectorBackend: ultralytics YOLO11-x, person class only.

    Mirrors _make_model + _detect_image. conf/nms are captured at load() and
    applied in infer(). imgsz and class id are model-internal defaults.
    """

    IMGSZ = 1920
    PERSON_CLASS = 0

    def __init__(self) -> None:
        self._model = None
        self._conf = None
        self._nms = None

    def load(self, cfg, weights_root) -> None:
        from ultralytics import YOLO
        weights = str(weights_root / cfg.weights) if cfg.weights else "yolo11x.pt"
        self._model = YOLO(weights)
        self._conf = cfg.conf_thresh
        self._nms = cfg.nms_iou

    def infer(self, img_path):
        res = self._model.predict(
            str(img_path), classes=[self.PERSON_CLASS],
            conf=self._conf, iou=self._nms, imgsz=self.IMGSZ, verbose=False,
        )[0]
        rows = []
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            rows.append((x1, y1, x2, y2, float(b.conf[0])))
        return rows

    def teardown(self) -> None:
        self._model = None
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_detection(scene_dir, det_out_dir, cams, cfg, weights_root, backend=None):
    """Run person detection over Original/<scene>/<cam>/Frame/*.jpg and write
    per-camera detection files via _write_camera.

    backend: a DetectorBackend. When None, resolved from cfg.model_name via the
             model registry. Inject a fake in tests.
    """
    if backend is None:
        from .registry import get_detector
        backend = get_detector(cfg.model_name)
    backend.load(cfg, weights_root)
    try:
        scene = Path(scene_dir).name
        for cam in cams:
            frame_dir = Path(scene_dir) / cam / "Frame"
            frame_paths = sorted(frame_dir.glob("*.jpg"))
            dets_by_frame: dict[int, list] = {}
            for fp in frame_paths:
                if not fp.stem.isdigit():
                    continue
                frame_id = int(fp.stem)
                rows = backend.infer(fp)
                if rows:
                    dets_by_frame[frame_id] = rows
            _write_camera(
                det_dir=Path(det_out_dir) / scene, cam=cam, dets_by_frame=dets_by_frame,
                img_rel_for_frame=lambda f, c=cam, s=scene: f"Original/{s}/{c}/Frame/{f:06d}.jpg",
            )
    finally:
        backend.teardown()
