# Model Backends — Phase 2 (first piece) design

- **Date:** 2026-05-30
- **Status:** approved (brainstorm), pending implementation plan
- **Scope:** the **ModelBackend trio** only — swappable detector / reid / pose
  backends. Builds on the Phase 1 StageRegistry branch. The other P2 pieces
  (run-local `external/`, `DatasetAdapter`, `TrackingBackend`, entry-point
  discovery) are separate specs, gated on this one.

## Problem

Each model stage hardcodes its backend: `detect.py` does `from ..models import
detect_yolo` and calls `detect_yolo.run_detection(...)`; `reid.py`/`pose.py`
likewise. There is no common interface, no config field to choose a model, and
the weights identity is hardcoded (`detect_yolo`: default `weights="yolo11x.pt"`;
`reid_solider`: a fixed `weights/solider_swin_small.pth` path; `pose_rtmpose`: a
fixed ONNX URL). To A/B a different detector, or drop in a fine-tuned SOLIDER
checkpoint, you must edit stage code.

This blocks the paused **detector-recall sweep** and **reid fine-tune** workstreams,
both of which want to swap a model by config.

## Goals

1. Models are selected by config (`model_name`, optional `weights`) and resolved
   through a registry/factory — no stage edits to swap a model.
2. A common per-kind backend interface so a new model implements only inference,
   not the byte-compatible YACHIYO serialization.
3. **Zero behavior change** for the current stack; `configs/baseline.yaml` works
   untouched (defaults = today's models). Guarded by tests.

## Non-goals (separate specs / YAGNI)

New model implementations (we only wrap the 3 existing); entry-point/third-party
discovery (the registry maps can later gain `.from_entry_points()`);
`DatasetAdapter` and `TrackingBackend`; moving model-internal constants
(YOLO `imgsz=1920`/`classes=[0]`, pose input size) into config — they stay as
backend defaults.

## Design

### Per-kind backend protocols (`aic24_nvidia/models/backends.py`)

A small `Protocol` per kind with a **load → infer → teardown** lifecycle
(replacing the lazy module-global + `_release_gpu` pattern). The inference method
mirrors each adapter's already-isolated primitive (`_detect_image` / `_embed` /
`_estimate`):

```python
class DetectorBackend(Protocol):
    def load(self, cfg: DetectCfg, weights_root: Path) -> None: ...
    def infer(self, img_path: Path) -> list[tuple[float, float, float, float, float]]: ...  # (x1,y1,x2,y2,score)
    def teardown(self) -> None: ...

class ReIDBackend(Protocol):
    def load(self, cfg: ReidCfg, weights_root: Path) -> None: ...
    def embed(self, crop) -> "np.ndarray": ...      # PIL.Image -> 1-D float32
    def teardown(self) -> None: ...

class PoseBackend(Protocol):
    def load(self, cfg: PoseCfg, weights_root: Path) -> None: ...
    def estimate(self, img, bboxes) -> list: ...    # BGR ndarray + [[x1,y1,x2,y2],...] -> N x 17 x [x,y,score]
    def teardown(self) -> None: ...
```

### Orchestrators keep serialization; primitives become backends

`run_detection` / `run_reid` / `run_pose` **stay as the per-kind orchestrators**
and keep the byte-compatible YACHIYO writers (`_write_camera`, `extract_camera`'s
writer, `run_pose`'s writer) — the format-critical code, written once per kind.
Today's `_detect_image`/`_embed`/`_estimate` plus their lazy loaders become the
default backend classes `YoloDetector` / `SoliderReID` / `RTMPoseBackend`, wrapping
the identical logic. Each orchestrator becomes:

```
backend = injected backend OR get_<kind>(cfg.<kind>.model_name)
backend.load(cfg.<kind>, weights_root)
for item in ...:                       # cams / frames / crops
    result = backend.infer(...)        # .infer / .embed / .estimate
    <write byte-compatible output>     # unchanged per-kind writer
backend.teardown()
```

So serialization is guaranteed identical across models; a new model implements
only the inference method. The optional `backend=` parameter is the test-injection
seam (formalizing today's monkeypatch of `_detect_image`/`_embed`/`_estimate`).

### Registry / factory (`aic24_nvidia/models/registry.py`)

Mirrors the Phase 1 `StageRegistry` pattern — central, explicit, no import magic.
One map per kind, keyed by the names already recorded in manifests:

```python
DETECTORS: dict[str, type[DetectorBackend]] = {"yolo11x": YoloDetector}
REIDS:     dict[str, type[ReIDBackend]]     = {"solider_swin_small": SoliderReID}
POSES:     dict[str, type[PoseBackend]]     = {"rtmpose-l": RTMPoseBackend}

def get_detector(name: str) -> DetectorBackend: ...   # unknown name -> ValueError listing known names
def get_reid(name: str) -> ReIDBackend: ...
def get_pose(name: str) -> PoseBackend: ...
```

### Config

Each sub-config gains two optional fields, defaulting to today's model so
`configs/baseline.yaml` is unchanged:

```python
@dataclass(frozen=True)
class DetectCfg:
    conf_thresh: float
    nms_iou: float
    model_name: str = "yolo11x"
    weights: str | None = None      # path override, resolved against weights_root

# ReidCfg.model_name = "solider_swin_small"; PoseCfg.model_name = "rtmpose-l"; both gain weights: str | None = None
```

`load_config` validates each `model_name` is a registered backend, so a typo
fails at config load rather than mid-run. `weights` (when set) is resolved against
`cfg.weights_root` and passed to the backend's `load`; when `None` the backend
uses its default (yolo11x.pt / solider_swin_small.pth / RTMPose ONNX URL).

### Stage wiring

`detect.py` → `run_detection(scene_dir, out, cams, cfg.detect, cfg.weights_root)`;
`reid.py`/`pose.py` similarly pass their sub-config + `weights_root`. The
orchestrator resolves the backend via the factory. No stage hardcodes
`from ..models import <impl>` anymore. Manifests keep recording `model_name` (now
from config) for provenance.

## Migration plan (five independently-green steps)

1. Add `backends.py` (protocols) + `registry.py` (maps + factories + validation) +
   a registry test. Nothing consumes them yet.
2. Wrap existing primitives as `YoloDetector`/`SoliderReID`/`RTMPoseBackend` (pure
   move, no behavior change). Characterization: each backend's inference method
   returns what the old primitive did.
3. Add `model_name`/`weights` to `DetectCfg`/`ReidCfg`/`PoseCfg` (defaults) +
   `load_config` validation against the registry.
4. Refactor the three orchestrators to resolve+drive the backend via the
   lifecycle, keeping the byte-compat writers; preserve the `backend=` injection.
5. Update `detect.py`/`reid.py`/`pose.py` to pass `cfg.<stage>` + `weights_root`;
   delete the hardcoded model imports.

## Testing

- **Registry:** known names resolve to the right class; unknown name → `ValueError`
  listing known names; default `model_name`s match the current models.
- **Backends:** the existing `test_detect_yolo` / `test_reid_solider` /
  `test_pose_rtmpose` adapt to inject a fake backend and assert the **same
  byte-compatible output** (serialization is unchanged, so the assertions hold).
- **Config:** `load_config` rejects an unknown `model_name`; absent `model_name`
  defaults to the current model (baseline.yaml unchanged).
- All GPU-free (inference is mocked/injected; lifecycle exercised with a fake).

## Risks & mitigations

- **Byte-format drift when extracting backends (step 2/4).** Mitigated: the
  writers are not touched (only the inference primitive moves into a class), and
  the existing format tests assert exact output.
- **Backward-compat of existing configs.** `model_name` defaults to the current
  model and `load_config` stays tolerant of its absence; a test pins this.
- **Lifecycle vs current lazy-load.** `load()` is called once by the orchestrator
  before the loop and `teardown()` replaces `_release_gpu`; on exception the
  orchestrator's `try/finally` (added in step 4) guarantees teardown, fixing the
  current mid-stage GPU-leak-on-error gap as a free side benefit.
