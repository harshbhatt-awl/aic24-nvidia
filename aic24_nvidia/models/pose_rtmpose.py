"""RTMPose-l pose adapter — byte-compatible replacement for the mmpose-based
pose stage.

Byte-compatibility contract:
  • Reads  Detection/<scene>/<cam>.txt (comma-sep: cam,frame,cls,x1,y1,x2,y2,conf)
  • Reads  Original/<scene>/<cam>/Frame/<frame:06d>.jpg
  • Writes Pose/<scene>/<cam>/<cam>_out_keypoint.json

JSON format (must match what tracking/src/pose.py expects):
    {
      "<frame_id_str>": [
        {"bbox": [x1, y1, x2, y2, 1.0],   # x1..y2 = INTEGERS (from detection txt)
         "keypoints": [[x, y, score], ...]  # 17 COCO keypoints
        },
        ...
      ],
      ...
    }

  Top-level keys are frame numbers as STRINGS ("1", "2", ...).
  bbox uses the EXACT integer coords from the detection txt — downstream matches
  pose → detection via f"{frame}_{x1}_{y1}_{x2}_{y2}", so the ints must agree.
  17 COCO keypoints in order: 0=nose … 16=right_ankle.

Part A (format logic) is fully tested and mocked.
Part B (_estimate via rtmlib.RTMPose) is best-effort; verified in GPU smoke run.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Part B: lazy RTMPose-l model (real inference, requires ONNX runtime + GPU)
# ---------------------------------------------------------------------------

# RTMPose-l body7 256×192 ONNX — openmmlab release v1 (best-effort URL).
# rtmlib's BaseTool accepts a .zip URL and caches the extracted .onnx.
# Source: https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose
# See also rtmlib Body.MODE['performance'] for the -x variant (384×288).
# BEST-EFFORT: hash is best-effort, validated at smoke time; if this URL 404s,
# fall back to _RTMPOSE_M_FALLBACK_URL below.
_RTMPOSE_L_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.zip"
)

# Verified rtmlib-registered fallback (RTMPose-m body7 256x192) if the -l URL 404s at smoke time:
_RTMPOSE_M_FALLBACK_URL = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"


class RTMPoseBackend:
    """Default PoseBackend: RTMPose-l body7 256x192 ONNX. Mirrors _estimate's
    lazy load. Default model is the bundled URL; a cfg.weights override is a
    local .onnx path resolved against weights_root."""

    INPUT_SIZE = (192, 256)  # (W, H)

    def __init__(self) -> None:
        self._model = None

    def load(self, cfg, weights_root) -> None:
        import torch
        from rtmlib import RTMPose
        onnx = str(weights_root / cfg.weights) if cfg.weights else _RTMPOSE_L_URL
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = RTMPose(
            onnx_model=onnx, model_input_size=self.INPUT_SIZE,
            backend="onnxruntime", device=dev,
        )

    def estimate(self, img, bboxes):
        keypoints, scores = self._model(img, bboxes=np.array(bboxes, dtype=np.float32))
        out = []
        for kp, sc in zip(keypoints, scores):
            out.append([[float(x), float(y), float(s)] for (x, y), s in zip(kp, sc)])
        return out

    def teardown(self) -> None:
        self._model = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# Part A: format-writing logic
# ---------------------------------------------------------------------------

def run_pose(det_scene_dir, original_scene_dir, pose_out_dir, scene, cams,
             cfg, weights_root, backend=None):
    """Run top-down pose estimation for all cameras in a scene.

    backend: a PoseBackend. When None, resolved from cfg.model_name. Inject a
             fake in tests.
    """
    import cv2  # type: ignore

    if backend is None:
        from .registry import get_pose
        backend = get_pose(cfg.model_name)
    backend.load(cfg, weights_root)
    try:
        det_scene_dir = Path(det_scene_dir)
        for cam in cams:
            det_path = det_scene_dir / f"{cam}.txt"
            dets = np.genfromtxt(det_path, dtype=str, delimiter=",")

            out_dir = Path(pose_out_dir) / scene / cam
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{cam}_out_keypoint.json"

            if dets.ndim == 1 and dets.shape[0] == 0:
                with open(out_path, "w") as f:
                    json.dump({}, f)
                continue

            if dets.ndim == 1:
                dets = dets.reshape(1, -1)

            by_frame: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
            for (_c, frame, _cls, x1, y1, x2, y2, _conf) in dets:
                by_frame[int(frame)].append((int(x1), int(y1), int(x2), int(y2)))

            save: dict[str, list] = {}
            for frame_id in sorted(by_frame):
                img_path = Path(original_scene_dir) / cam / "Frame" / f"{frame_id:06d}.jpg"
                img = cv2.imread(str(img_path))
                bboxes_int = list(by_frame[frame_id])
                kpts = backend.estimate(img, bboxes_int)
                people = []
                for (x1, y1, x2, y2), kp in zip(bboxes_int, kpts):
                    people.append({"bbox": [x1, y1, x2, y2, 1.0], "keypoints": kp})
                save[str(frame_id)] = people

            with open(out_path, "w") as f:
                json.dump(save, f)
    finally:
        backend.teardown()
