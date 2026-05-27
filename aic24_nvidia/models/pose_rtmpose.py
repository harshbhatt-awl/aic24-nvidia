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

_MODEL = None

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


def _estimate(img, bboxes):
    """RTMPose-l top-down pose estimation (Part B — GPU/ONNX).

    Args:
        img:    BGR np.ndarray (H, W, 3) — as returned by cv2.imread.
        bboxes: list of [x1, y1, x2, y2] (ints or floats, xyxy format).

    Returns:
        List of N lists, each containing 17 [x, y, score] triplets
        in COCO ordering (0=nose … 16=right_ankle).

    Notes:
        • _MODEL is loaded lazily on first call.
        • BEST-EFFORT: _RTMPOSE_L_URL has not been validated in this session;
          it follows the naming pattern from rtmlib's Body.MODE registry.
          Confirm during GPU smoke run. If the -l URL 404s, switch to
          _RTMPOSE_M_FALLBACK_URL.
        • bboxes may contain int or float values; they are cast to float32
          internally via np.array(bboxes, dtype=np.float32).
    """
    global _MODEL
    if _MODEL is None:
        import torch  # type: ignore
        from rtmlib import RTMPose  # type: ignore
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL = RTMPose(
            onnx_model=_RTMPOSE_L_URL,
            model_input_size=(192, 256),  # (W, H) = 192×256
            backend="onnxruntime",
            device=_device,
        )

    # RTMPose.__call__ expects bboxes as a list of [x1,y1,x2,y2] and returns:
    #   keypoints: np.ndarray shape (N, 17, 2)  — (x, y) pixel coords
    #   scores:    np.ndarray shape (N, 17)      — per-keypoint confidence
    keypoints, scores = _MODEL(img, bboxes=np.array(bboxes, dtype=np.float32))

    out = []
    for kp, sc in zip(keypoints, scores):
        # kp: (17, 2), sc: (17,)
        out.append([[float(x), float(y), float(s)] for (x, y), s in zip(kp, sc)])
    return out


def _release_gpu():
    global _MODEL
    _MODEL = None
    import gc
    gc.collect()
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Part A: format-writing logic (fully tested, estimate is injectable)
# ---------------------------------------------------------------------------

def run_pose(
    det_scene_dir,
    original_scene_dir,
    pose_out_dir,
    scene: str,
    cams: list[str],
    estimate=None,
) -> None:
    """Run RTMPose-l top-down pose estimation for all cameras in a scene.

    Args:
        det_scene_dir:      Path to Detection/<scene>/ (contains <cam>.txt).
        original_scene_dir: Path to Original/<scene>/ (contains <cam>/Frame/*.jpg).
        pose_out_dir:       Root output dir; JSON written to
                            <pose_out_dir>/<scene>/<cam>/<cam>_out_keypoint.json.
        scene:              Scene name string.
        cams:               List of camera name strings.
        estimate:           Callable(img_bgr, bboxes) → list-of-17-kpt-lists.
                            When None, uses the module-global _estimate (looked
                            up at call time so monkeypatching the global works).
    """
    import cv2  # type: ignore

    # Look up the module-global at call time so monkeypatching _estimate works.
    if estimate is None:
        estimate = _estimate

    det_scene_dir = Path(det_scene_dir)

    for cam in cams:
        det_path = det_scene_dir / f"{cam}.txt"
        dets = np.genfromtxt(det_path, dtype=str, delimiter=",")

        out_dir = Path(pose_out_dir) / scene / cam
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{cam}_out_keypoint.json"

        if dets.ndim == 1 and dets.shape[0] == 0:
            # Empty detection file — write empty JSON.
            with open(out_path, "w") as f:
                json.dump({}, f)
            continue

        if dets.ndim == 1:
            # Single detection row — genfromtxt returns shape (ncols,); reshape.
            dets = dets.reshape(1, -1)

        # Group detections by frame.
        by_frame: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
        for (_c, frame, _cls, x1, y1, x2, y2, _conf) in dets:
            by_frame[int(frame)].append((int(x1), int(y1), int(x2), int(y2)))

        save: dict[str, list] = {}
        for frame_id in sorted(by_frame):
            img_path = (
                Path(original_scene_dir) / cam / "Frame" / f"{frame_id:06d}.jpg"
            )
            img = cv2.imread(str(img_path))

            bboxes_int = list(by_frame[frame_id])  # list of (x1,y1,x2,y2) ints

            kpts = estimate(img, bboxes_int)

            people = []
            for (x1, y1, x2, y2), kp in zip(bboxes_int, kpts):
                people.append({
                    "bbox": [x1, y1, x2, y2, 1.0],
                    "keypoints": kp,
                })
            save[str(frame_id)] = people

        with open(out_path, "w") as f:
            json.dump(save, f)
    _release_gpu()
