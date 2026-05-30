import pytest

from aic24_nvidia.models import registry
from aic24_nvidia.models.detect_yolo import YoloDetector
from aic24_nvidia.models.reid_solider import SoliderReID
from aic24_nvidia.models.pose_rtmpose import RTMPoseBackend


def test_default_names_resolve_to_the_current_backends():
    assert isinstance(registry.get_detector("yolo11x"), YoloDetector)
    assert isinstance(registry.get_reid("solider_swin_small"), SoliderReID)
    assert isinstance(registry.get_pose("rtmpose-l"), RTMPoseBackend)


def test_unknown_detector_name_raises_listing_known_names():
    with pytest.raises(ValueError, match="yolo11x"):
        registry.get_detector("nope")


def test_unknown_reid_name_raises():
    with pytest.raises(ValueError, match="solider_swin_small"):
        registry.get_reid("nope")


def test_unknown_pose_name_raises():
    with pytest.raises(ValueError, match="rtmpose-l"):
        registry.get_pose("nope")


def test_known_name_helpers():
    assert "yolo11x" in registry.detector_names()
    assert "solider_swin_small" in registry.reid_names()
    assert "rtmpose-l" in registry.pose_names()
