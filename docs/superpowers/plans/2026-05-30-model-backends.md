# Model Backends Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the detector / reid / pose models swappable by config (`model_name`, optional `weights`) behind per-kind backend interfaces, without changing the byte-compatible YACHIYO output.

**Architecture:** Three per-kind `Protocol`s (`DetectorBackend`/`ReIDBackend`/`PoseBackend`) with a `load → infer → teardown` lifecycle. Today's inference primitives become default backend classes; the per-kind orchestrators (`run_detection`/`run_reid`/`run_pose`) keep the byte-compat writers and now resolve a backend via a registry keyed by `model_name`. Mirrors the existing `aic24_nvidia/registry.py` StageRegistry pattern.

**Tech Stack:** Python 3.14, dataclasses, `typing.Protocol`, pytest. Models: ultralytics YOLO, timm-based SOLIDER, rtmlib RTMPose (all loaded lazily inside `load()`).

**Spec:** `docs/superpowers/specs/2026-05-30-model-backends-design.md`

**Conventions:** run `pytest` as `.venv/bin/python -m pytest`. Keep the full unit suite green after every task (`.venv/bin/python -m pytest tests/unit -q`). Run `.venv/bin/ruff check .` before each commit.

---

### Task 1: Backend protocols

**Files:**
- Create: `aic24_nvidia/models/backends.py`

- [ ] **Step 1: Create the protocols module**

```python
# aic24_nvidia/models/backends.py
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
```

- [ ] **Step 2: Verify it imports**

Run: `.venv/bin/python -c "import aic24_nvidia.models.backends as b; print(b.DetectorBackend, b.ReIDBackend, b.PoseBackend)"`
Expected: prints the three protocol classes, no error.

- [ ] **Step 3: Commit**

```bash
git add aic24_nvidia/models/backends.py
git commit -m "feat(models): per-kind backend protocols (load/infer/teardown)"
```

---

### Task 2: YoloDetector backend (extract from primitives)

**Files:**
- Modify: `aic24_nvidia/models/detect_yolo.py`

Behavior-preserving extraction: `_make_model` + `_detect_image` + the gc/empty_cache cleanup become a `YoloDetector` class. `_write_camera` is untouched. `run_detection` is refactored in Task 7 (leave it for now — keep `_make_model`/`_detect_image` until then so nothing breaks).

- [ ] **Step 1: Add the YoloDetector class**

Add to `aic24_nvidia/models/detect_yolo.py` (after the existing `_detect_image`, before `run_detection`). Keep `CLAMP_W`/`CLAMP_H`, `_write_camera`, `_make_model`, `_detect_image` as-is for now.

```python
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
```

- [ ] **Step 2: Verify import + suite still green**

Run: `.venv/bin/python -c "from aic24_nvidia.models.detect_yolo import YoloDetector; YoloDetector()"`
Expected: no error (construct without loading a model).

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: same pass count as before this task (139 passed, 1 skipped) — nothing consumes YoloDetector yet.

- [ ] **Step 3: Commit**

```bash
git add aic24_nvidia/models/detect_yolo.py
git commit -m "feat(models): YoloDetector backend class (extracted from primitives)"
```

---

### Task 3: SoliderReID backend (extract from primitives)

**Files:**
- Modify: `aic24_nvidia/models/reid_solider.py`

- [ ] **Step 1: Add the SoliderReID class**

Add to `aic24_nvidia/models/reid_solider.py` (after `_embed`, before `extract_camera`). Leave `_embed`/`_get_transform`/`_MODEL`/`_TRANSFORM` in place for now (removed in Task 8).

```python
class SoliderReID:
    """Default ReIDBackend: SOLIDER Swin-Small, 768-d embeddings.

    Mirrors _get_transform + _embed's lazy load. Default weights resolve to
    <weights_root>/solider_swin_small.pth (same file the module-relative path
    used when run from the repo root).
    """

    def __init__(self) -> None:
        self._model = None
        self._transform = None

    def load(self, cfg, weights_root) -> None:
        import torch
        import torchvision.transforms as T
        from aic24_nvidia.models.solider import (
            SOLIDER_MEAN, SOLIDER_SIZE, SOLIDER_STD, load_solider_swin_small,
        )
        weights = weights_root / (cfg.weights or "solider_swin_small.pth")
        self._model = load_solider_swin_small(weights)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.eval().to(dev)
        self._transform = T.Compose([
            T.Resize(list(SOLIDER_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=list(SOLIDER_MEAN), std=list(SOLIDER_STD)),
        ])

    def embed(self, crop):
        import torch
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = self._transform(crop.convert("RGB")).unsqueeze(0).to(dev)
        with torch.no_grad():
            feat = self._model(x)
        return feat.cpu().numpy()[0].astype(np.float32)

    def teardown(self) -> None:
        self._model = None
        self._transform = None
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

- [ ] **Step 2: Verify**

Run: `.venv/bin/python -c "from aic24_nvidia.models.reid_solider import SoliderReID; SoliderReID()"`
Expected: no error.

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: 139 passed, 1 skipped.

- [ ] **Step 3: Commit**

```bash
git add aic24_nvidia/models/reid_solider.py
git commit -m "feat(models): SoliderReID backend class (extracted from primitives)"
```

---

### Task 4: RTMPoseBackend backend (extract from primitives)

**Files:**
- Modify: `aic24_nvidia/models/pose_rtmpose.py`

- [ ] **Step 1: Add the RTMPoseBackend class**

Add to `aic24_nvidia/models/pose_rtmpose.py` (after `_estimate`, before `_release_gpu`). Leave `_estimate`/`_MODEL`/`_release_gpu` in place for now (removed in Task 9). `_RTMPOSE_L_URL` already exists in the module.

```python
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
```

- [ ] **Step 2: Verify**

Run: `.venv/bin/python -c "from aic24_nvidia.models.pose_rtmpose import RTMPoseBackend; RTMPoseBackend()"`
Expected: no error.

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: 139 passed, 1 skipped.

- [ ] **Step 3: Commit**

```bash
git add aic24_nvidia/models/pose_rtmpose.py
git commit -m "feat(models): RTMPoseBackend backend class (extracted from primitives)"
```

---

### Task 5: Model registry + factory

**Files:**
- Create: `aic24_nvidia/models/registry.py`
- Test: `tests/unit/test_model_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_model_registry.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_model_registry.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'aic24_nvidia.models.registry'`.

- [ ] **Step 3: Create the registry**

```python
# aic24_nvidia/models/registry.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_model_registry.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add aic24_nvidia/models/registry.py tests/unit/test_model_registry.py
git commit -m "feat(models): registry mapping model_name -> backend"
```

---

### Task 6: Config model_name / weights + validation

**Files:**
- Modify: `aic24_nvidia/config.py`
- Test: `tests/unit/test_model_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_model_config.py
from pathlib import Path

import pytest
import yaml

from aic24_nvidia.config import load_config
from aic24_nvidia.errors import ConfigError

_BASE = {
    "scene": "Warehouse_001",
    "data_root": ".", "weights_root": "./weights", "outputs_root": "./outputs",
    "external_root": "./external",
    "clip": {"start_sec": 0, "duration_sec": 30},
    "detect": {"conf_thresh": 0.5, "nms_iou": 0.5},
    "reid": {"similarity_thresh": 0.7},
    "pose": {"keypoint_conf": 0.3},
    "sct": {"track_buffer": 30, "match_thresh": 0.8},
    "mct": {"cluster_thresh": 0.6, "min_track_len": 10},
    "vram_min_free_gb": 0.0, "fps": 30,
}


def _write(tmp_path, **patch):
    body = {k: dict(v) if isinstance(v, dict) else v for k, v in _BASE.items()}
    for k, v in patch.items():
        body[k] = v
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(body))
    return p


def test_model_names_default_to_current_stack(tmp_path):
    cfg = load_config(_write(tmp_path))
    assert cfg.detect.model_name == "yolo11x"
    assert cfg.reid.model_name == "solider_swin_small"
    assert cfg.pose.model_name == "rtmpose-l"
    assert cfg.detect.weights is None


def test_explicit_model_name_and_weights_are_read(tmp_path):
    cfg = load_config(_write(
        tmp_path,
        detect={"conf_thresh": 0.5, "nms_iou": 0.5, "model_name": "yolo11x",
                "weights": "custom.pt"},
    ))
    assert cfg.detect.weights == "custom.pt"


def test_unknown_detector_model_name_rejected(tmp_path):
    p = _write(tmp_path, detect={"conf_thresh": 0.5, "nms_iou": 0.5, "model_name": "bogus"})
    with pytest.raises(ConfigError, match="model_name"):
        load_config(p)


def test_unknown_reid_model_name_rejected(tmp_path):
    p = _write(tmp_path, reid={"similarity_thresh": 0.7, "model_name": "bogus"})
    with pytest.raises(ConfigError, match="model_name"):
        load_config(p)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_model_config.py -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'model_name'` (or `AttributeError` on `cfg.detect.model_name`).

- [ ] **Step 3: Add the fields to the dataclasses**

In `aic24_nvidia/config.py`, change the three sub-config dataclasses:

```python
@dataclass(frozen=True)
class DetectCfg:
    conf_thresh: float
    nms_iou: float
    model_name: str = "yolo11x"
    weights: str | None = None


@dataclass(frozen=True)
class ReidCfg:
    similarity_thresh: float
    model_name: str = "solider_swin_small"
    weights: str | None = None


@dataclass(frozen=True)
class PoseCfg:
    keypoint_conf: float
    model_name: str = "rtmpose-l"
    weights: str | None = None
```

- [ ] **Step 4: Add validation in load_config**

In `aic24_nvidia/config.py`, find where the sub-configs are built inside `load_config`:

```python
        detect=DetectCfg(**body["detect"]),
        reid=ReidCfg(**body["reid"]),
        pose=PoseCfg(**body["pose"]),
```

Replace that block, and add validation just before the final `return Config(...)`. Add this helper validation immediately before the `return Config(`:

```python
    detect_cfg = DetectCfg(**body["detect"])
    reid_cfg = ReidCfg(**body["reid"])
    pose_cfg = PoseCfg(**body["pose"])

    from .models import registry as _model_registry
    if detect_cfg.model_name not in _model_registry.DETECTORS:
        raise ConfigError(
            f"detect.model_name must be one of {_model_registry.detector_names()}, "
            f"got {detect_cfg.model_name!r}"
        )
    if reid_cfg.model_name not in _model_registry.REIDS:
        raise ConfigError(
            f"reid.model_name must be one of {_model_registry.reid_names()}, "
            f"got {reid_cfg.model_name!r}"
        )
    if pose_cfg.model_name not in _model_registry.POSES:
        raise ConfigError(
            f"pose.model_name must be one of {_model_registry.pose_names()}, "
            f"got {pose_cfg.model_name!r}"
        )
```

Then change the `return Config(...)` to use the locals:

```python
        detect=detect_cfg,
        reid=reid_cfg,
        pose=pose_cfg,
```

(The `from .models import registry` is a local import inside `load_config` to avoid importing the model layer at config-module import time.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_model_config.py tests/unit/test_config.py -q`
Expected: PASS (new model-config tests + existing config tests).

- [ ] **Step 6: Commit**

```bash
git add aic24_nvidia/config.py tests/unit/test_model_config.py
git commit -m "feat(config): model_name/weights on detect/reid/pose + validation"
```

---

### Task 7: Refactor run_detection to drive a backend

**Files:**
- Modify: `aic24_nvidia/models/detect_yolo.py`
- Test: `tests/unit/test_detect_yolo.py`

- [ ] **Step 1: Rewrite the run_detection glue test to inject a fake backend**

Replace `test_run_detection_glue_writes_expected_files` in `tests/unit/test_detect_yolo.py` with:

```python
def test_run_detection_glue_writes_expected_files(tmp_path):
    # Build a tmp scene dir with two tiny jpgs
    frame_dir = tmp_path / "Original" / "scene_001" / "camera_0390" / "Frame"
    frame_dir.mkdir(parents=True)
    for name in ("000001.jpg", "000002.jpg"):
        Image.new("RGB", (64, 64)).save(frame_dir / name)

    # Fake backend: one detection for frame 000001, none for 000002.
    class FakeDetector:
        def load(self, cfg, weights_root):
            self.loaded = True
        def infer(self, img_path):
            if Path(img_path).stem == "000001":
                return [(1.0, 2.0, 3.0, 4.0, 0.9)]
            return []
        def teardown(self):
            self.torn = True

    from aic24_nvidia.config import DetectCfg
    detect_yolo.run_detection(
        scene_dir=tmp_path / "Original" / "scene_001",
        det_out_dir=tmp_path / "out",
        cams=["camera_0390"],
        cfg=DetectCfg(conf_thresh=0.5, nms_iou=0.5),
        weights_root=tmp_path / "weights",
        backend=FakeDetector(),
    )

    txt_path = tmp_path / "out" / "scene_001" / "camera_0390.txt"
    assert txt_path.exists(), "camera_0390.txt not written"
    lines = txt_path.read_text().splitlines()
    assert len(lines) == 1, f"expected 1 line, got {len(lines)}: {lines}"
    assert lines[0] == "camera_0390,1,1,1,2,3,4,0.9"

    j = json.loads((tmp_path / "out" / "scene_001" / "camera_0390.json").read_text())
    assert "00000000" in j
    assert j["00000000"]["Frame"] == 1
    assert len(j) == 1
```

(Leave `test_write_detection_outputs_matches_upstream_format` unchanged — it tests `_write_camera` directly.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_detect_yolo.py::test_run_detection_glue_writes_expected_files -q`
Expected: FAIL — `TypeError: run_detection() got an unexpected keyword argument 'cfg'`.

- [ ] **Step 3: Rewrite run_detection and remove the old primitives**

In `aic24_nvidia/models/detect_yolo.py`, replace the whole `run_detection` function AND delete the now-unused `_make_model` and `_detect_image` functions. Keep `CLAMP_W`/`CLAMP_H`, `_write_camera`, and `YoloDetector`.

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_detect_yolo.py -q`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add aic24_nvidia/models/detect_yolo.py tests/unit/test_detect_yolo.py
git commit -m "refactor(detect): run_detection drives a DetectorBackend"
```

---

### Task 8: Refactor run_reid to drive a backend

**Files:**
- Modify: `aic24_nvidia/models/reid_solider.py`
- Test: `tests/unit/test_reid_solider.py`

- [ ] **Step 1: Update the two reid tests to pass embed explicitly**

In `tests/unit/test_reid_solider.py`, replace the two `monkeypatch.setattr(reid_solider, "_embed", ...)` + `embed=None` calls with an explicit `embed=` callable (the writer seam is unchanged). Change the test signatures from `(tmp_path, monkeypatch)` to `(tmp_path)`.

In `test_npy_filename_and_json_update_match_upstream`, replace:

```python
    monkeypatch.setattr(reid_solider, "_embed", lambda crop: np.ones(768, dtype=np.float32))
    reid_solider.extract_camera(
        det_scene_dir=det_dir, original_scene_dir=tmp_path / "Original" / scene,
        emb_out_dir=emb_dir, scene=scene, cam=cam, embed=None)
```

with:

```python
    reid_solider.extract_camera(
        det_scene_dir=det_dir, original_scene_dir=tmp_path / "Original" / scene,
        emb_out_dir=emb_dir, scene=scene, cam=cam,
        embed=lambda crop: np.ones(768, dtype=np.float32))
```

In `test_empty_detection_file_is_noop`, replace:

```python
    monkeypatch.setattr(reid_solider, "_embed", lambda crop: np.ones(768, dtype=np.float32))
    # Should not raise and should not create any .npy files
    reid_solider.extract_camera(
        det_scene_dir=det_dir, original_scene_dir=tmp_path / "Original" / scene,
        emb_out_dir=emb_dir, scene=scene, cam=cam, embed=None)
```

with:

```python
    # Should not raise and should not create any .npy files
    reid_solider.extract_camera(
        det_scene_dir=det_dir, original_scene_dir=tmp_path / "Original" / scene,
        emb_out_dir=emb_dir, scene=scene, cam=cam,
        embed=lambda crop: np.ones(768, dtype=np.float32))
```

- [ ] **Step 2: Add a run_reid backend-injection test**

Add to `tests/unit/test_reid_solider.py`:

```python
def test_run_reid_drives_injected_backend(tmp_path):
    scene, cam = "scene_001", "camera_0390"
    det_dir = tmp_path / "Detection" / scene
    det_dir.mkdir(parents=True)
    (det_dir / f"{cam}.txt").write_text("camera_0390,5,1,10,20,110,220,0.91\n")
    (det_dir / f"{cam}.json").write_text(json.dumps(
        {"00000000": {"Frame": 5, "ImgPath": "x", "NpyPath": "",
                      "Coordinate": {"x1": 10, "y1": 20, "x2": 110, "y2": 220},
                      "ClusterID": None, "OfflineID": None}}))
    frame_dir = tmp_path / "Original" / scene / cam / "Frame"
    frame_dir.mkdir(parents=True)
    Image.new("RGB", (1920, 1080)).save(frame_dir / "000005.jpg")
    emb_dir = tmp_path / "EmbedFeature"

    events = []

    class FakeReID:
        def load(self, cfg, weights_root):
            events.append("load")
        def embed(self, crop):
            return np.ones(768, dtype=np.float32)
        def teardown(self):
            events.append("teardown")

    from aic24_nvidia.config import ReidCfg
    reid_solider.run_reid(
        det_scene_dir=det_dir, original_scene_dir=tmp_path / "Original" / scene,
        emb_out_dir=emb_dir, scene=scene, cams=[cam],
        cfg=ReidCfg(similarity_thresh=0.7), weights_root=tmp_path / "weights",
        backend=FakeReID())

    assert events == ["load", "teardown"]
    assert list((emb_dir / scene / cam).glob("*.npy"))
```

- [ ] **Step 3: Run tests to verify the run_reid test fails**

Run: `.venv/bin/python -m pytest tests/unit/test_reid_solider.py -q`
Expected: `test_run_reid_drives_injected_backend` FAILS (`TypeError: run_reid() got an unexpected keyword argument 'cfg'`); the two edited `extract_camera` tests PASS (writer logic unchanged).

- [ ] **Step 4: Refactor run_reid and remove the old globals**

In `aic24_nvidia/models/reid_solider.py`: delete `_MODEL`, `_TRANSFORM`, `_get_transform`, `_embed`, and `_release_gpu`. Keep `extract_camera` exactly as-is (it already takes `embed`). Replace `run_reid` with:

```python
def run_reid(det_scene_dir, original_scene_dir, emb_out_dir, scene, cams,
             cfg, weights_root, backend=None):
    """Run ReID embedding extraction for all cameras in a scene.

    backend: a ReIDBackend. When None, resolved from cfg.model_name. Inject a
             fake in tests.
    """
    if backend is None:
        from .registry import get_reid
        backend = get_reid(cfg.model_name)
    backend.load(cfg, weights_root)
    try:
        for cam in cams:
            extract_camera(det_scene_dir, original_scene_dir, emb_out_dir,
                           scene, cam, embed=backend.embed)
    finally:
        backend.teardown()
```

Also change `extract_camera`'s `embed=None` default to a required argument (remove the `if embed is None: embed = _embed` fallback, since `_embed` is gone). Edit the signature and body:

```python
def extract_camera(
    det_scene_dir,
    original_scene_dir,
    emb_out_dir,
    scene: str,
    cam: str,
    embed,
) -> None:
```

and delete these lines from the top of `extract_camera`'s body:

```python
    if embed is None:
        embed = _embed
```

- [ ] **Step 5: Run tests to verify all pass**

Run: `.venv/bin/python -m pytest tests/unit/test_reid_solider.py -q`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add aic24_nvidia/models/reid_solider.py tests/unit/test_reid_solider.py
git commit -m "refactor(reid): run_reid drives a ReIDBackend"
```

---

### Task 9: Refactor run_pose to drive a backend

**Files:**
- Modify: `aic24_nvidia/models/pose_rtmpose.py`
- Test: `tests/unit/test_pose_rtmpose.py`

- [ ] **Step 1: Update the three pose tests to inject a fake backend**

In `tests/unit/test_pose_rtmpose.py`, each test currently does
`monkeypatch.setattr(pose_rtmpose, "_estimate", <lambda>)` then calls `run_pose(...)`.
Replace that pattern in all three tests: drop `monkeypatch` from the signature and
pass a fake backend whose `estimate` is the same lambda. Add this helper at the top
of the file (after the imports):

```python
def _fake_pose(estimate_fn):
    class FakePose:
        def load(self, cfg, weights_root):
            pass
        def estimate(self, img, bboxes):
            return estimate_fn(img, bboxes)
        def teardown(self):
            pass
    return FakePose()
```

Then in `test_pose_json_schema_and_bbox_keys` replace:

```python
    monkeypatch.setattr(pose_rtmpose, "_estimate",
                        lambda img, bboxes: [[[1.0, 2.0, 0.9]] * 17 for _ in bboxes])
    pose_rtmpose.run_pose(det_scene_dir=det_dir,
                          original_scene_dir=tmp_path / "Original" / scene,
                          pose_out_dir=pose_out, scene=scene, cams=[cam])
```

with (also change the test signature `(tmp_path, monkeypatch)` → `(tmp_path)`):

```python
    from aic24_nvidia.config import PoseCfg
    pose_rtmpose.run_pose(
        det_scene_dir=det_dir, original_scene_dir=tmp_path / "Original" / scene,
        pose_out_dir=pose_out, scene=scene, cams=[cam],
        cfg=PoseCfg(keypoint_conf=0.3), weights_root=tmp_path / "weights",
        backend=_fake_pose(lambda img, bboxes: [[[1.0, 2.0, 0.9]] * 17 for _ in bboxes]))
```

Apply the identical transformation to `test_empty_detection_file_writes_empty_json`
(lambda returns `[[[0.0, 0.0, 0.5]] * 17 for _ in bboxes]`) and
`test_single_detection_row` (same `0.0,0.0,0.5` lambda) — drop `monkeypatch`, add the
`cfg=PoseCfg(...)`, `weights_root=...`, `backend=_fake_pose(<same lambda>)` arguments.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_pose_rtmpose.py -q`
Expected: FAIL — `TypeError: run_pose() got an unexpected keyword argument 'cfg'`.

- [ ] **Step 3: Refactor run_pose and remove the old primitives**

In `aic24_nvidia/models/pose_rtmpose.py`: delete `_estimate`, `_MODEL`, and `_release_gpu` (keep `_RTMPOSE_L_URL`, `_RTMPOSE_M_FALLBACK_URL`, and `RTMPoseBackend`). Replace `run_pose` with (the writer body is unchanged; only the signature, the backend resolution, the `estimate` call, and teardown change):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_pose_rtmpose.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add aic24_nvidia/models/pose_rtmpose.py tests/unit/test_pose_rtmpose.py
git commit -m "refactor(pose): run_pose drives a PoseBackend"
```

---

### Task 10: Wire the stages to the new orchestrator signatures

**Files:**
- Modify: `aic24_nvidia/stages/detect.py`
- Modify: `aic24_nvidia/stages/reid.py`
- Modify: `aic24_nvidia/stages/pose.py`

No unit test changes here (stage `run()` bodies are GPU/subprocess and not unit-tested). Verification is: byte-compile, full unit suite green, ruff clean. Each stage already imports its orchestrator module; only the call args + the recorded `model` param change.

- [ ] **Step 1: Update detect.py**

In `aic24_nvidia/stages/detect.py`, replace the `detect_yolo.run_detection(...)` call:

```python
        detect_yolo.run_detection(
            scene_dir=scene_src,
            det_out_dir=ctx.work_dir,
            cams=cams,
            conf_thresh=cfg.detect.conf_thresh,
            nms_iou=cfg.detect.nms_iou,
            weights="yolo11x.pt",
        )
```

with:

```python
        detect_yolo.run_detection(
            scene_dir=scene_src,
            det_out_dir=ctx.work_dir,
            cams=cams,
            cfg=cfg.detect,
            weights_root=cfg.weights_root,
        )
```

And change the recorded params from the hardcoded `"yolo11x"`:

```python
        ctx.set_params({
            "model": "yolo11x",
            "conf_thresh": cfg.detect.conf_thresh,
            "nms_iou": cfg.detect.nms_iou,
        })
```

to read the model name from config:

```python
        ctx.set_params({
            "model": cfg.detect.model_name,
            "conf_thresh": cfg.detect.conf_thresh,
            "nms_iou": cfg.detect.nms_iou,
        })
```

- [ ] **Step 2: Update reid.py**

In `aic24_nvidia/stages/reid.py`, replace the `reid_solider.run_reid(...)` call:

```python
        reid_solider.run_reid(
            det_scene_dir=det_scene,
            original_scene_dir=original / SCENE,
            emb_out_dir=ctx.work_dir,
            scene=SCENE,
            cams=cams,
        )
```

with:

```python
        reid_solider.run_reid(
            det_scene_dir=det_scene,
            original_scene_dir=original / SCENE,
            emb_out_dir=ctx.work_dir,
            scene=SCENE,
            cams=cams,
            cfg=cfg.reid,
            weights_root=cfg.weights_root,
        )
```

And change the recorded params:

```python
        ctx.set_params({
            "model": "solider_swin_small",
            "similarity_thresh": cfg.reid.similarity_thresh,
        })
```

to:

```python
        ctx.set_params({
            "model": cfg.reid.model_name,
            "similarity_thresh": cfg.reid.similarity_thresh,
        })
```

- [ ] **Step 3: Update pose.py**

In `aic24_nvidia/stages/pose.py`, replace the `pose_rtmpose.run_pose(...)` call:

```python
        pose_rtmpose.run_pose(
            det_scene_dir=det_scene_dir,
            original_scene_dir=original / SCENE,
            pose_out_dir=ctx.work_dir,
            scene=SCENE,
            cams=cams,
        )
```

with:

```python
        pose_rtmpose.run_pose(
            det_scene_dir=det_scene_dir,
            original_scene_dir=original / SCENE,
            pose_out_dir=ctx.work_dir,
            scene=SCENE,
            cams=cams,
            cfg=cfg.pose,
            weights_root=cfg.weights_root,
        )
```

And change the recorded params:

```python
        ctx.set_params({
            "keypoint_conf": cfg.pose.keypoint_conf,
            "model": "rtmpose-l",
        })
```

to:

```python
        ctx.set_params({
            "keypoint_conf": cfg.pose.keypoint_conf,
            "model": cfg.pose.model_name,
        })
```

- [ ] **Step 4: Verify byte-compile, full suite, ruff, no leftover primitives**

Run: `.venv/bin/python -m py_compile aic24_nvidia/models/*.py aic24_nvidia/stages/*.py aic24_nvidia/config.py`
Expected: no output (compiles).

Run: `grep -rnE "\b(_make_model|_detect_image|_get_transform|_release_gpu|_embed|_estimate)\b" aic24_nvidia/models/`
Expected: no matches (the extracted primitives are all gone from detect_yolo.py / reid_solider.py / pose_rtmpose.py).

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: PASS — **150 passed, 1 skipped** (139 prior + 6 from test_model_registry + 4 from test_model_config + 1 new reid backend-injection test; the detect/pose/reid byte-format tests are edited in place, not added). Zero failures is the bar.

Run: `.venv/bin/ruff check .`
Expected: `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add aic24_nvidia/stages/detect.py aic24_nvidia/stages/reid.py aic24_nvidia/stages/pose.py
git commit -m "refactor(stages): detect/reid/pose select model backend via config"
```

---

### Task 11: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Document the model registry**

In `CLAUDE.md`, under the "Models (the learned components)" section, add a sentence noting that backends are selected by config:

```markdown
> Backends are swappable by config: `detect.model_name` / `reid.model_name` /
> `pose.model_name` (default to the v3 stack) resolve through
> `aic24_nvidia/models/registry.py` to a `*Backend` class implementing
> `load/infer/teardown`. The per-kind orchestrators (`run_detection`/`run_reid`/
> `run_pose`) keep the byte-compatible YACHIYO writers. `cfg.<stage>.weights`
> overrides the checkpoint (resolved against `weights_root`). `load_config`
> rejects an unknown `model_name`.
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(claude): model backend registry + config-driven model selection"
```

---

## Self-review notes (for the executor)

- **Spec coverage:** protocols (T1), default backends (T2-4), registry+factory (T5), config model_name/weights+validation (T6), orchestrators drive backends + test injection (T7-9), stages pass cfg+weights_root + provenance (T10), docs (T11). All spec sections covered.
- **Behavior-preserving:** `_write_camera` and `extract_camera` byte-format logic is never edited; the pose writer body is copied verbatim into the refactored `run_pose`. The byte-format tests are the safety net.
- **Backward-compat:** `model_name` defaults to the current model; `baseline.yaml` is untouched (pinned by `test_model_names_default_to_current_stack`).
- **Type/name consistency:** factory functions `get_detector/get_reid/get_pose`; maps `DETECTORS/REIDS/POSES`; orchestrator kwargs `cfg`, `weights_root`, `backend`; backend methods `load/infer|embed|estimate/teardown` — used identically across all tasks.
- **Free win:** `load`/`teardown` + `try/finally` in each orchestrator fixes the current mid-stage GPU-leak-on-error gap.
