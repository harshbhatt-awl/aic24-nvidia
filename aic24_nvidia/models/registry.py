"""Model backend registry — maps a config model_name to a backend class.

Mirrors aic24_nvidia.registry (the StageRegistry): central, explicit, no import
magic. To add a model, implement the relevant protocol in aic24_nvidia/models/
and add one entry to the map below.
"""
from __future__ import annotations

from .backends import DetectorBackend, PoseBackend, ReIDBackend
from .detect_yolo import YoloDetector
from .pose_rtmpose import RTMPoseBackend
from .reid_solider import SoliderReID

DETECTORS: dict[str, type] = {"yolo11x": YoloDetector}
REIDS: dict[str, type] = {"solider_swin_small": SoliderReID}
POSES: dict[str, type] = {"rtmpose-l": RTMPoseBackend}


def _get(table: dict[str, type], name: str, kind: str):
    try:
        return table[name]()
    except KeyError:
        known = ", ".join(sorted(table))
        raise ValueError(f"unknown {kind} model_name {name!r}; known: {known}") from None


def get_detector(name: str) -> DetectorBackend:
    return _get(DETECTORS, name, "detector")


def get_reid(name: str) -> ReIDBackend:
    return _get(REIDS, name, "reid")


def get_pose(name: str) -> PoseBackend:
    return _get(POSES, name, "pose")


def detector_names() -> list[str]:
    return sorted(DETECTORS)


def reid_names() -> list[str]:
    return sorted(REIDS)


def pose_names() -> list[str]:
    return sorted(POSES)
