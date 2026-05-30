"""Per-kind model backend protocols.

A backend owns ONLY inference (load -> infer -> teardown). The byte-compatible
YACHIYO serialization stays in the per-kind orchestrators (run_detection /
run_reid / run_pose). Adding a new model = implement one of these protocols and
register it in aic24_nvidia.models.registry.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np
    from PIL import Image

    from ..config import DetectCfg, PoseCfg, ReidCfg


class DetectorBackend(Protocol):
    def load(self, cfg: "DetectCfg", weights_root: Path) -> None: ...
    def infer(self, img_path: Path) -> list[tuple[float, float, float, float, float]]:
        """Return [(x1, y1, x2, y2, score), ...] for one image."""
        ...
    def teardown(self) -> None: ...


class ReIDBackend(Protocol):
    def load(self, cfg: "ReidCfg", weights_root: Path) -> None: ...
    def embed(self, crop: "Image.Image") -> "np.ndarray":
        """Return a 1-D float32 embedding for one PIL crop."""
        ...
    def teardown(self) -> None: ...


class PoseBackend(Protocol):
    def load(self, cfg: "PoseCfg", weights_root: Path) -> None: ...
    def estimate(self, img: "np.ndarray", bboxes: list) -> list:
        """img: BGR ndarray; bboxes: [[x1,y1,x2,y2], ...].
        Return N lists of 17 [x, y, score] COCO keypoints."""
        ...
    def teardown(self) -> None: ...
