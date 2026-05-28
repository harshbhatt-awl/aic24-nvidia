# Ankle footpoints + EMA world smoother — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the 0.30 gap between image HOTA (0.758) and 3D-world HOTA (0.458) by (1) overriding the 2D→world projection point from bbox-bottom-center to RTMPose ankle keypoints between SCT and MCT, and (2) applying optional EMA temporal smoothing on per-(frame, gid) world coords in `world_tracks.py`. Ship as config-gated experiment-pipeline variants with byte-identical defaults.

**Architecture:** A new pure module `aic24_nvidia/world_projection.py` rewrites `WorldCoordinate` in each per-camera SCT JSON in place, called from `stages/mct.py` right after the staging copytree and before the upstream MCT subprocess. SCPT does not consume `WorldCoordinate`, so this single injection point captures the full benefit at zero upstream-patch cost. The smoother is a pure post-process added to `world_tracks.py` and invoked from `stages/evaluate.py`. Both improvements are off by default (`world_projection.method = bbox_bottom`, `world_smoothing.method = none`).

**Tech Stack:** Python 3.14, numpy, pytest. No new third-party dependencies.

**Spec:** `docs/superpowers/specs/2026-05-28-ankle-footpoints-and-world-smoother-design.md`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `aic24_nvidia/config.py` | Modify | Add `WorldProjectionCfg`, `WorldSmoothingCfg`; thread into `Config`; parse with defaults. |
| `aic24_nvidia/world_projection.py` | Create | Pure functions: pose lookup, ankle-pixel selection, homography projection, in-place SCT JSON rewrite. |
| `aic24_nvidia/stages/mct.py` | Modify | Invoke `rewrite_world_coordinates` between `shutil.copytree` and the subprocess. |
| `aic24_nvidia/world_tracks.py` | Modify | Add `smooth_world_tracks` (EMA, per-gid). |
| `aic24_nvidia/stages/evaluate.py` | Modify | Plumb `cfg.world_smoothing` through; call smoother before writing pred. |
| `configs/baseline.yaml` | Modify | Add no-op defaults for the new sections. |
| `experiments/registry.yaml` | Modify | Add `ankle_footpoint_sweep` and `world_smoother_sweep`. |
| `tests/unit/test_world_projection.py` | Create | Unit tests for each ankle method + the rewrite. |
| `tests/unit/test_world_smoother.py` | Create | Unit tests for EMA smoothing. |
| `tests/unit/test_config.py` | Modify | Add cases for the new config sections. |

**Camera-naming key:**
- NVIDIA name (in pose/detect/calibration dirs): `camera_0390` (`camera_{:04d}`).
- YACHIYO numeric id (in SCT JSON filenames): `390` → file `camera390_tracking_results.json` (`camera{:03d}` per `external/.../tracking/src/run.py:40`).
- The mapping comes from `run_dir / "adapted" / "scene.json"`, which has shape `{scene: {yachiyo_cam_name: nvidia_cam_name}}`. `yachiyo_cam_name` is `camera_NNNN`; the numeric id is `int(name.split("_")[-1])`.

**Files MCT consumes from SCT:** every `camera{N}_tracking_results.json` and every `fixed_camera{N}_tracking_results.json` in `mct.tmp/scene_001/`. The rewrite must update BOTH (we cannot be sure which one upstream loads internally; rewriting both is idempotent and safe).

**SCT JSON shape (per camera):** `{serial_str: {"Frame": int, "NpyPath": str, "Coordinate": {"x1","y1","x2","y2"}, "WorldCoordinate": {"x","y"}, "OfflineID": int}}`. The `Coordinate` x1..y2 may be int OR float — the pose JSON's join key uses **integers**, so the rewrite must cast to int before lookup.

**Pose JSON shape:** `Pose/scene_001/<nvidia_cam>/<nvidia_cam>_out_keypoint.json` → `{frame_str: [{"bbox": [x1,y1,x2,y2,1.0], "keypoints": [[x,y,score]*17]}, ...]}` (per `aic24_nvidia/models/pose_rtmpose.py:9-23`). COCO 17 ordering: `15 = left_ankle`, `16 = right_ankle`.

**Calibration:** `run_dir / "adapted" / "Original" / "scene_001" / "camera_NNNN" / "calibration.json"`, containing `"homography matrix": [[..3x3..]]` (the world→image homography). World coords come from `inv(H) @ [x_img, y_img, 1]; divide by w`. Same formula as `external/.../tracking/src/utils.py:170`.

---

## Phase A — Config plumbing

### Task A1: Add `WorldProjectionCfg` dataclass + parsing

**Files:**
- Modify: `aic24_nvidia/config.py`
- Test: `tests/unit/test_config.py`

- [ ] **Step 1: Write the failing test**

Add this test to `tests/unit/test_config.py`. The file already has `_minimal(tmp_path) -> dict` and `_write(tmp_path, body) -> Path` helpers — reuse them:

```python
def test_world_projection_defaults(tmp_path):
    """When world_projection block is absent, defaults to bbox_bottom + 0.3."""
    body = _minimal(tmp_path)
    # Do NOT add world_projection
    cfg = load_config(_write(tmp_path, body))

    assert cfg.world_projection.method == "bbox_bottom"
    assert cfg.world_projection.ankle_min_conf == pytest.approx(0.3)


def test_world_projection_explicit(tmp_path):
    """world_projection block is parsed and validated."""
    body = _minimal(tmp_path)
    body["world_projection"] = {"method": "ankle_avg", "ankle_min_conf": 0.5}
    cfg = load_config(_write(tmp_path, body))

    assert cfg.world_projection.method == "ankle_avg"
    assert cfg.world_projection.ankle_min_conf == pytest.approx(0.5)


def test_world_projection_invalid_method(tmp_path):
    """Unknown method is rejected at load time."""
    body = _minimal(tmp_path)
    body["world_projection"] = {"method": "bogus"}
    with pytest.raises(ConfigError, match="world_projection.method"):
        load_config(_write(tmp_path, body))
```

- [ ] **Step 2: Run the test and confirm it fails**

```
.venv/bin/pytest tests/unit/test_config.py::test_world_projection_defaults -v
```

Expected: `FAILED` with `AttributeError: 'Config' object has no attribute 'world_projection'`.

- [ ] **Step 3: Implement `WorldProjectionCfg`**

In `aic24_nvidia/config.py`, after the `EvalCfg` dataclass, add:

```python
@dataclass(frozen=True)
class WorldProjectionCfg:
    method: str = "bbox_bottom"        # bbox_bottom | ankle_avg | ankle_lower | ankle_w_fallback
    ankle_min_conf: float = 0.3
```

Add `world_projection: WorldProjectionCfg` to the `Config` dataclass (between `eval` and `tracking_params`):

```python
    eval: EvalCfg
    world_projection: WorldProjectionCfg
    tracking_params: Mapping[str, object]
```

In `load_config`, after the `eval_cfg` line, add:

```python
    wp_body = body.get("world_projection") or {}
    wp_method = wp_body.get("method", "bbox_bottom")
    if wp_method not in {"bbox_bottom", "ankle_avg", "ankle_lower", "ankle_w_fallback"}:
        raise ConfigError(f"world_projection.method must be one of bbox_bottom|ankle_avg|ankle_lower|ankle_w_fallback, got {wp_method!r}")
    world_projection = WorldProjectionCfg(
        method=wp_method,
        ankle_min_conf=float(wp_body.get("ankle_min_conf", 0.3)),
    )
```

And pass it into the `Config(...)` construction:

```python
        eval=eval_cfg,
        world_projection=world_projection,
        tracking_params=tracking_params,
```

- [ ] **Step 4: Run all three new tests and confirm they pass**

```
.venv/bin/pytest tests/unit/test_config.py::test_world_projection_defaults tests/unit/test_config.py::test_world_projection_explicit tests/unit/test_config.py::test_world_projection_invalid_method -v
```

Expected: 3 passed.

- [ ] **Step 5: Run the full config test file to confirm no regression**

```
.venv/bin/pytest tests/unit/test_config.py -v
```

Expected: all existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add aic24_nvidia/config.py tests/unit/test_config.py
git commit -m "feat(config): add WorldProjectionCfg with method + ankle_min_conf knobs"
```

---

### Task A2: Add `WorldSmoothingCfg` dataclass + parsing

**Files:**
- Modify: `aic24_nvidia/config.py`
- Test: `tests/unit/test_config.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_config.py`:

```python
def test_world_smoothing_defaults(tmp_path):
    """world_smoothing absent -> defaults to none + 0.3."""
    body = _minimal(tmp_path)
    cfg = load_config(_write(tmp_path, body))

    assert cfg.world_smoothing.method == "none"
    assert cfg.world_smoothing.ema_alpha == pytest.approx(0.3)


def test_world_smoothing_explicit(tmp_path):
    body = _minimal(tmp_path)
    body["world_smoothing"] = {"method": "ema", "ema_alpha": 0.5}
    cfg = load_config(_write(tmp_path, body))

    assert cfg.world_smoothing.method == "ema"
    assert cfg.world_smoothing.ema_alpha == pytest.approx(0.5)


def test_world_smoothing_invalid_method(tmp_path):
    body = _minimal(tmp_path)
    body["world_smoothing"] = {"method": "kalman"}  # not implemented in v1
    with pytest.raises(ConfigError, match="world_smoothing.method"):
        load_config(_write(tmp_path, body))


def test_world_smoothing_alpha_range(tmp_path):
    body = _minimal(tmp_path)
    body["world_smoothing"] = {"method": "ema", "ema_alpha": 1.5}
    with pytest.raises(ConfigError, match="ema_alpha"):
        load_config(_write(tmp_path, body))
```

- [ ] **Step 2: Run and confirm failure**

```
.venv/bin/pytest tests/unit/test_config.py::test_world_smoothing_defaults -v
```

Expected: `FAILED` with `AttributeError: ... world_smoothing`.

- [ ] **Step 3: Implement `WorldSmoothingCfg`**

In `aic24_nvidia/config.py`, after `WorldProjectionCfg`:

```python
@dataclass(frozen=True)
class WorldSmoothingCfg:
    method: str = "none"              # none | ema
    ema_alpha: float = 0.3
```

Add to `Config` (after `world_projection`):

```python
    world_projection: WorldProjectionCfg
    world_smoothing: WorldSmoothingCfg
    tracking_params: Mapping[str, object]
```

In `load_config`, after the `world_projection = ...` block, add:

```python
    ws_body = body.get("world_smoothing") or {}
    ws_method = ws_body.get("method", "none")
    if ws_method not in {"none", "ema"}:
        raise ConfigError(f"world_smoothing.method must be one of none|ema, got {ws_method!r}")
    ws_alpha = float(ws_body.get("ema_alpha", 0.3))
    if not (0.0 <= ws_alpha <= 1.0):
        raise ConfigError(f"world_smoothing.ema_alpha must be in [0, 1], got {ws_alpha}")
    world_smoothing = WorldSmoothingCfg(method=ws_method, ema_alpha=ws_alpha)
```

And in the `Config(...)` constructor call:

```python
        world_projection=world_projection,
        world_smoothing=world_smoothing,
        tracking_params=tracking_params,
```

- [ ] **Step 4: Run new tests, confirm pass**

```
.venv/bin/pytest tests/unit/test_config.py -v
```

Expected: all tests pass (4 new + all existing).

- [ ] **Step 5: Commit**

```bash
git add aic24_nvidia/config.py tests/unit/test_config.py
git commit -m "feat(config): add WorldSmoothingCfg with method + ema_alpha knobs"
```

---

### Task A3: Add no-op defaults to `configs/baseline.yaml`

**Files:**
- Modify: `configs/baseline.yaml`

- [ ] **Step 1: Add the two sections**

Open `configs/baseline.yaml` and add (alphabetical placement after `tracking_params:` is fine — but adding before `vram_min_free_gb:` keeps related world-coord knobs grouped):

```yaml
world_projection:
  method: bbox_bottom        # bbox_bottom | ankle_avg | ankle_lower | ankle_w_fallback
  ankle_min_conf: 0.3

world_smoothing:
  method: none               # none | ema
  ema_alpha: 0.3
```

- [ ] **Step 2: Confirm the config loads cleanly**

```
.venv/bin/python -c "from aic24_nvidia.config import load_config; c = load_config('configs/baseline.yaml'); print(c.world_projection); print(c.world_smoothing)"
```

Expected:
```
WorldProjectionCfg(method='bbox_bottom', ankle_min_conf=0.3)
WorldSmoothingCfg(method='none', ema_alpha=0.3)
```

- [ ] **Step 3: Run any existing config-related smoke test**

```
.venv/bin/pytest tests/unit/test_config.py -v
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add configs/baseline.yaml
git commit -m "feat(config): add world_projection + world_smoothing no-op defaults to baseline"
```

---

## Phase B — `world_projection.py` module

### Task B1: `_build_pose_lookup` — index pose JSON by (frame, bbox-ints)

**Files:**
- Create: `aic24_nvidia/world_projection.py`
- Create: `tests/unit/test_world_projection.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_world_projection.py`:

```python
"""Unit tests for aic24_nvidia.world_projection."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_pose_json(path: Path, body: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body))


def test_build_pose_lookup_indexes_by_frame_and_bbox(tmp_path):
    """Pose lookup table is keyed by (frame_int, (x1,y1,x2,y2)) tuple."""
    from aic24_nvidia.world_projection import _build_pose_lookup

    pose = {
        "5": [
            {"bbox": [100, 200, 150, 350, 1.0],
             "keypoints": [[0, 0, 0.0]] * 17},
            {"bbox": [400, 500, 460, 700, 1.0],
             "keypoints": [[0, 0, 0.0]] * 17},
        ],
        "6": [
            {"bbox": [110, 210, 160, 360, 1.0],
             "keypoints": [[0, 0, 0.0]] * 17},
        ],
    }
    pose_path = tmp_path / "camera_0390_out_keypoint.json"
    _write_pose_json(pose_path, pose)

    lookup = _build_pose_lookup(pose_path)

    assert (5, (100, 200, 150, 350)) in lookup
    assert (5, (400, 500, 460, 700)) in lookup
    assert (6, (110, 210, 160, 360)) in lookup
    assert len(lookup) == 3
    # Values are the keypoints list (17 entries).
    assert len(lookup[(5, (100, 200, 150, 350))]) == 17


def test_build_pose_lookup_empty_file(tmp_path):
    from aic24_nvidia.world_projection import _build_pose_lookup
    pose_path = tmp_path / "empty.json"
    _write_pose_json(pose_path, {})
    assert _build_pose_lookup(pose_path) == {}
```

- [ ] **Step 2: Run and confirm failure**

```
.venv/bin/pytest tests/unit/test_world_projection.py::test_build_pose_lookup_indexes_by_frame_and_bbox -v
```

Expected: `FAILED` with `ImportError: cannot import name '_build_pose_lookup'`.

- [ ] **Step 3: Implement the module + helper**

Create `aic24_nvidia/world_projection.py`:

```python
"""Override per-detection WorldCoordinate in SCT JSONs using pose ankle keypoints.

Called from stages/mct.py after the SCT outputs are staged into mct.tmp/ and
before the upstream MCT subprocess runs. Rewrites are in place and idempotent.

Method | What gets projected to world coords
-------|------------------------------------
bbox_bottom        | ((x1+x2)/2, y2)                                — no-op
ankle_avg          | score-weighted mean of left_ankle, right_ankle
ankle_lower        | the ankle with the larger pixel y (planted foot)
ankle_w_fallback   | ankle_avg if both ankle scores >= ankle_min_conf, else bbox_bottom

SCPT does not consume WorldCoordinate (verified via grep on
external/.../tracking/src/scpt.py), so doing this between SCT and MCT does not
change SCT decisions — only MCT clustering and the final eval see the override.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


# COCO-17 ordering used by RTMPose (see aic24_nvidia/models/pose_rtmpose.py:23).
COCO_LEFT_ANKLE = 15
COCO_RIGHT_ANKLE = 16


def _build_pose_lookup(pose_json: Path) -> dict[tuple[int, tuple[int, int, int, int]], list[list[float]]]:
    """Index a per-camera pose JSON by (frame_int, bbox_ints) -> keypoints (17 x [x,y,score])."""
    body = json.loads(Path(pose_json).read_text())
    out: dict[tuple[int, tuple[int, int, int, int]], list[list[float]]] = {}
    for frame_str, entries in body.items():
        frame = int(frame_str)
        if not isinstance(entries, list):
            continue
        for e in entries:
            if not isinstance(e, dict):
                continue
            bbox = e.get("bbox")
            kps = e.get("keypoints")
            if not bbox or kps is None or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            out[(frame, (x1, y1, x2, y2))] = kps
    return out
```

- [ ] **Step 4: Run the tests, confirm pass**

```
.venv/bin/pytest tests/unit/test_world_projection.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add aic24_nvidia/world_projection.py tests/unit/test_world_projection.py
git commit -m "feat(world_projection): pose JSON lookup table (frame, bbox-ints -> keypoints)"
```

---

### Task B2: `_project_to_world` — homography helper

**Files:**
- Modify: `aic24_nvidia/world_projection.py`
- Modify: `tests/unit/test_world_projection.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_world_projection.py`:

```python
def test_project_to_world_identity_homography():
    """Identity homography means world = image coordinates."""
    import numpy as np
    from aic24_nvidia.world_projection import _project_to_world

    H = np.eye(3)
    wx, wy = _project_to_world(123.0, 456.0, H)
    assert wx == pytest.approx(123.0)
    assert wy == pytest.approx(456.0)


def test_project_to_world_translation_homography():
    """Pure translation homography shifts world coords."""
    import numpy as np
    from aic24_nvidia.world_projection import _project_to_world

    # world->image homography that adds (10, 20) when going world->image.
    # So image -> world subtracts (10, 20). H = [[1,0,10],[0,1,20],[0,0,1]].
    H = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]])
    wx, wy = _project_to_world(100.0, 200.0, H)
    assert wx == pytest.approx(90.0)
    assert wy == pytest.approx(180.0)
```

- [ ] **Step 2: Run and confirm failure**

```
.venv/bin/pytest tests/unit/test_world_projection.py::test_project_to_world_identity_homography -v
```

Expected: `FAILED` with `ImportError`.

- [ ] **Step 3: Implement `_project_to_world`**

Add to `aic24_nvidia/world_projection.py` (after the imports / before `_build_pose_lookup`):

```python
def _project_to_world(x_img: float, y_img: float, homography_matrix) -> tuple[float, float]:
    """Project an image-plane point (pixels) to world coordinates (metres).

    Matches the formula used in external/.../tracking/src/utils.py:170 —
    world->image homography H means image->world is inv(H) applied to
    [x, y, 1], with the result divided by its third component.
    """
    import numpy as np  # local import keeps the module import-time cheap

    H_inv = np.linalg.inv(np.asarray(homography_matrix, dtype=np.float64))
    v = H_inv @ np.array([x_img, y_img, 1.0])
    return float(v[0] / v[2]), float(v[1] / v[2])
```

- [ ] **Step 4: Run and confirm pass**

```
.venv/bin/pytest tests/unit/test_world_projection.py -v
```

Expected: all (4) passed.

- [ ] **Step 5: Commit**

```bash
git add aic24_nvidia/world_projection.py tests/unit/test_world_projection.py
git commit -m "feat(world_projection): homography helper _project_to_world"
```

---

### Task B3: `_compute_image_point` — pick the pixel point per method

**Files:**
- Modify: `aic24_nvidia/world_projection.py`
- Modify: `tests/unit/test_world_projection.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_world_projection.py`:

```python
@pytest.fixture
def _kps_two_ankles():
    """17-keypoint list with left_ankle=(100,800,0.9), right_ankle=(110,810,0.8)."""
    kps = [[0.0, 0.0, 0.0]] * 17
    kps = list(kps)
    kps[15] = [100.0, 800.0, 0.9]   # left_ankle
    kps[16] = [110.0, 810.0, 0.8]   # right_ankle
    return kps


def test_compute_image_point_bbox_bottom(_kps_two_ankles):
    from aic24_nvidia.world_projection import _compute_image_point

    bbox = (50, 100, 150, 850)   # bottom-center = (100, 850)
    # bbox_bottom must ignore keypoints and use ((x1+x2)/2, y2).
    x, y = _compute_image_point(bbox, kps=None, method="bbox_bottom", ankle_min_conf=0.3)
    assert (x, y) == pytest.approx((100.0, 850.0))


def test_compute_image_point_ankle_avg_score_weighted(_kps_two_ankles):
    from aic24_nvidia.world_projection import _compute_image_point

    bbox = (50, 100, 150, 850)
    # weighted avg of (100,800)@0.9 and (110,810)@0.8 =
    #   x = (0.9*100 + 0.8*110) / (0.9+0.8) = (90+88)/1.7 ≈ 104.7058...
    #   y = (0.9*800 + 0.8*810) / 1.7        = (720+648)/1.7 ≈ 804.7058...
    x, y = _compute_image_point(bbox, kps=_kps_two_ankles, method="ankle_avg",
                                ankle_min_conf=0.3)
    assert x == pytest.approx((0.9 * 100 + 0.8 * 110) / 1.7)
    assert y == pytest.approx((0.9 * 800 + 0.8 * 810) / 1.7)


def test_compute_image_point_ankle_avg_zero_scores_falls_back(_kps_two_ankles):
    """If both ankle scores are exactly zero, fall back to bbox_bottom."""
    from aic24_nvidia.world_projection import _compute_image_point

    kps = list(_kps_two_ankles)
    kps[15] = [100.0, 800.0, 0.0]
    kps[16] = [110.0, 810.0, 0.0]
    bbox = (50, 100, 150, 850)
    x, y = _compute_image_point(bbox, kps=kps, method="ankle_avg", ankle_min_conf=0.3)
    assert (x, y) == pytest.approx((100.0, 850.0))


def test_compute_image_point_ankle_lower(_kps_two_ankles):
    """ankle_lower picks the ankle with larger pixel y (the planted foot)."""
    from aic24_nvidia.world_projection import _compute_image_point

    bbox = (50, 100, 150, 850)
    # right_ankle at y=810 > left_ankle at y=800, so right_ankle wins.
    x, y = _compute_image_point(bbox, kps=_kps_two_ankles, method="ankle_lower",
                                ankle_min_conf=0.3)
    assert (x, y) == pytest.approx((110.0, 810.0))


def test_compute_image_point_ankle_w_fallback_uses_avg_when_confident(_kps_two_ankles):
    """ankle_w_fallback with both scores >= threshold uses ankle_avg."""
    from aic24_nvidia.world_projection import _compute_image_point

    bbox = (50, 100, 150, 850)
    x, y = _compute_image_point(bbox, kps=_kps_two_ankles, method="ankle_w_fallback",
                                ankle_min_conf=0.3)
    # Same as ankle_avg.
    assert x == pytest.approx((0.9 * 100 + 0.8 * 110) / 1.7)
    assert y == pytest.approx((0.9 * 800 + 0.8 * 810) / 1.7)


def test_compute_image_point_ankle_w_fallback_falls_back_when_low_conf():
    """ankle_w_fallback below threshold reverts to bbox_bottom."""
    from aic24_nvidia.world_projection import _compute_image_point

    kps = [[0.0, 0.0, 0.0]] * 17
    kps[15] = [100.0, 800.0, 0.1]  # below 0.3
    kps[16] = [110.0, 810.0, 0.2]  # below 0.3
    bbox = (50, 100, 150, 850)
    x, y = _compute_image_point(bbox, kps=kps, method="ankle_w_fallback",
                                ankle_min_conf=0.3)
    assert (x, y) == pytest.approx((100.0, 850.0))


def test_compute_image_point_no_pose_falls_back():
    """If pose lookup misses (kps is None) the method always falls back to bbox_bottom."""
    from aic24_nvidia.world_projection import _compute_image_point

    bbox = (50, 100, 150, 850)
    for method in ("ankle_avg", "ankle_lower", "ankle_w_fallback"):
        x, y = _compute_image_point(bbox, kps=None, method=method, ankle_min_conf=0.3)
        assert (x, y) == pytest.approx((100.0, 850.0)), f"method={method}"
```

- [ ] **Step 2: Run and confirm failure**

```
.venv/bin/pytest tests/unit/test_world_projection.py::test_compute_image_point_bbox_bottom -v
```

Expected: `FAILED` with `ImportError`.

- [ ] **Step 3: Implement `_compute_image_point`**

Add to `aic24_nvidia/world_projection.py`:

```python
def _bbox_bottom_point(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, _y1, x2, y2 = bbox
    return (float(x1) + float(x2)) / 2.0, float(y2)


def _compute_image_point(
    bbox: tuple[int, int, int, int],
    kps: list[list[float]] | None,
    method: str,
    ankle_min_conf: float,
) -> tuple[float, float]:
    """Pick the (x_img, y_img) pixel point to project to world coords.

    Falls back to bbox_bottom whenever the chosen method cannot be applied
    (no pose, both ankle scores zero, or low-confidence in ankle_w_fallback).
    """
    if method == "bbox_bottom" or kps is None:
        return _bbox_bottom_point(bbox)

    lx, ly, ls = kps[COCO_LEFT_ANKLE]
    rx, ry, rs = kps[COCO_RIGHT_ANKLE]

    if method == "ankle_avg":
        total = ls + rs
        if total <= 0.0:
            return _bbox_bottom_point(bbox)
        return (ls * lx + rs * rx) / total, (ls * ly + rs * ry) / total

    if method == "ankle_lower":
        if ls <= 0.0 and rs <= 0.0:
            return _bbox_bottom_point(bbox)
        # Pick the ankle with larger pixel y. Skip ankles with score 0.
        candidates = []
        if ls > 0.0:
            candidates.append((ly, lx, ly))
        if rs > 0.0:
            candidates.append((ry, rx, ry))
        # Sort by y descending, then take the first.
        candidates.sort(key=lambda t: -t[0])
        _y_key, cx, cy = candidates[0]
        return float(cx), float(cy)

    if method == "ankle_w_fallback":
        if ls >= ankle_min_conf and rs >= ankle_min_conf:
            total = ls + rs
            return (ls * lx + rs * rx) / total, (ls * ly + rs * ry) / total
        return _bbox_bottom_point(bbox)

    # Unknown method (defensive — config validation prevents this).
    return _bbox_bottom_point(bbox)
```

- [ ] **Step 4: Run all tests in the file**

```
.venv/bin/pytest tests/unit/test_world_projection.py -v
```

Expected: all 11 tests pass.

- [ ] **Step 5: Commit**

```bash
git add aic24_nvidia/world_projection.py tests/unit/test_world_projection.py
git commit -m "feat(world_projection): pixel-point selection per method (bbox_bottom/ankle_avg/ankle_lower/ankle_w_fallback)"
```

---

### Task B4: `rewrite_world_coordinates` — public API rewriting SCT JSONs in place

**Files:**
- Modify: `aic24_nvidia/world_projection.py`
- Modify: `tests/unit/test_world_projection.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_world_projection.py`:

```python
def _write_sct_json(path: Path, body: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body))


def _identity_calib_json(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "camera projection matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
        "homography matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    }))


def test_rewrite_bbox_bottom_is_noop(tmp_path):
    """method=bbox_bottom leaves WorldCoordinate untouched (identity contract)."""
    from aic24_nvidia.world_projection import rewrite_world_coordinates

    sct_scene = tmp_path / "mct.tmp" / "scene_001"
    pose_scene = tmp_path / "Pose" / "scene_001"
    orig_scene = tmp_path / "adapted" / "Original" / "scene_001"

    sct_json = sct_scene / "camera390_tracking_results.json"
    original_body = {
        "00000001": {
            "Frame": 5, "NpyPath": "x",
            "Coordinate": {"x1": 50, "y1": 100, "x2": 150, "y2": 850},
            "WorldCoordinate": {"x": 7.0, "y": 11.0},   # arbitrary; should be preserved
            "OfflineID": 0,
        },
    }
    _write_sct_json(sct_json, original_body)

    pose_json = pose_scene / "camera_0390" / "camera_0390_out_keypoint.json"
    _write_pose_json(pose_json, {
        "5": [{"bbox": [50, 100, 150, 850, 1.0],
               "keypoints": [[0.0, 0.0, 0.0]] * 17}]
    })
    _identity_calib_json(orig_scene / "camera_0390" / "calibration.json")

    rewritten = rewrite_world_coordinates(
        sct_scene_dir=sct_scene,
        pose_scene_dir=pose_scene,
        calib_root=orig_scene,
        camera_map={390: "camera_0390"},
        method="bbox_bottom",
        ankle_min_conf=0.3,
    )

    assert rewritten == 0   # no detections were rewritten
    after = json.loads(sct_json.read_text())
    assert after["00000001"]["WorldCoordinate"] == {"x": 7.0, "y": 11.0}


def test_rewrite_ankle_avg_with_identity_homography(tmp_path):
    """method=ankle_avg with identity homography projects ankles directly."""
    from aic24_nvidia.world_projection import rewrite_world_coordinates

    sct_scene = tmp_path / "mct.tmp" / "scene_001"
    pose_scene = tmp_path / "Pose" / "scene_001"
    orig_scene = tmp_path / "adapted" / "Original" / "scene_001"

    sct_json = sct_scene / "camera390_tracking_results.json"
    _write_sct_json(sct_json, {
        "00000001": {
            "Frame": 5, "NpyPath": "x",
            "Coordinate": {"x1": 50, "y1": 100, "x2": 150, "y2": 850},
            "WorldCoordinate": {"x": -999.0, "y": -999.0},
            "OfflineID": 0,
        },
    })
    pose_json = pose_scene / "camera_0390" / "camera_0390_out_keypoint.json"
    kps = [[0.0, 0.0, 0.0]] * 17
    kps[15] = [100.0, 800.0, 1.0]
    kps[16] = [110.0, 810.0, 1.0]
    _write_pose_json(pose_json, {"5": [{"bbox": [50, 100, 150, 850, 1.0], "keypoints": kps}]})
    _identity_calib_json(orig_scene / "camera_0390" / "calibration.json")

    rewritten = rewrite_world_coordinates(
        sct_scene_dir=sct_scene,
        pose_scene_dir=pose_scene,
        calib_root=orig_scene,
        camera_map={390: "camera_0390"},
        method="ankle_avg",
        ankle_min_conf=0.3,
    )

    assert rewritten == 1
    after = json.loads(sct_json.read_text())
    # ankle_avg with equal weights = midpoint = (105, 805); identity homography -> world same.
    assert after["00000001"]["WorldCoordinate"]["x"] == pytest.approx(105.0)
    assert after["00000001"]["WorldCoordinate"]["y"] == pytest.approx(805.0)


def test_rewrite_pose_miss_falls_back_to_bbox_bottom(tmp_path):
    """Detection without a matching pose entry falls back silently."""
    from aic24_nvidia.world_projection import rewrite_world_coordinates

    sct_scene = tmp_path / "mct.tmp" / "scene_001"
    pose_scene = tmp_path / "Pose" / "scene_001"
    orig_scene = tmp_path / "adapted" / "Original" / "scene_001"

    sct_json = sct_scene / "camera390_tracking_results.json"
    _write_sct_json(sct_json, {
        "00000001": {
            "Frame": 5, "NpyPath": "x",
            "Coordinate": {"x1": 50, "y1": 100, "x2": 150, "y2": 850},
            "WorldCoordinate": {"x": 0.0, "y": 0.0},
            "OfflineID": 0,
        },
    })
    pose_json = pose_scene / "camera_0390" / "camera_0390_out_keypoint.json"
    # Pose JSON has a different bbox — miss on join.
    _write_pose_json(pose_json, {"5": [{"bbox": [999, 999, 9999, 9999, 1.0],
                                        "keypoints": [[0.0, 0.0, 0.0]] * 17}]})
    _identity_calib_json(orig_scene / "camera_0390" / "calibration.json")

    rewritten = rewrite_world_coordinates(
        sct_scene_dir=sct_scene,
        pose_scene_dir=pose_scene,
        calib_root=orig_scene,
        camera_map={390: "camera_0390"},
        method="ankle_avg",
        ankle_min_conf=0.3,
    )

    # Fallback to bbox_bottom = (100, 850); identity homography -> same.
    after = json.loads(sct_json.read_text())
    assert after["00000001"]["WorldCoordinate"]["x"] == pytest.approx(100.0)
    assert after["00000001"]["WorldCoordinate"]["y"] == pytest.approx(850.0)
    assert rewritten == 1   # still counts as "rewritten" (even if it landed on bbox_bottom)


def test_rewrite_processes_both_fixed_and_unfixed(tmp_path):
    """Both camera390_*.json and fixed_camera390_*.json are rewritten."""
    from aic24_nvidia.world_projection import rewrite_world_coordinates

    sct_scene = tmp_path / "mct.tmp" / "scene_001"
    pose_scene = tmp_path / "Pose" / "scene_001"
    orig_scene = tmp_path / "adapted" / "Original" / "scene_001"

    det_body = {
        "00000001": {
            "Frame": 5, "NpyPath": "x",
            "Coordinate": {"x1": 50, "y1": 100, "x2": 150, "y2": 850},
            "WorldCoordinate": {"x": -1.0, "y": -1.0},
            "OfflineID": 0,
        },
    }
    _write_sct_json(sct_scene / "camera390_tracking_results.json", det_body)
    _write_sct_json(sct_scene / "fixed_camera390_tracking_results.json", det_body)

    kps = [[0.0, 0.0, 0.0]] * 17
    kps[15] = [100.0, 800.0, 1.0]
    kps[16] = [110.0, 810.0, 1.0]
    _write_pose_json(
        pose_scene / "camera_0390" / "camera_0390_out_keypoint.json",
        {"5": [{"bbox": [50, 100, 150, 850, 1.0], "keypoints": kps}]},
    )
    _identity_calib_json(orig_scene / "camera_0390" / "calibration.json")

    rewritten = rewrite_world_coordinates(
        sct_scene_dir=sct_scene,
        pose_scene_dir=pose_scene,
        calib_root=orig_scene,
        camera_map={390: "camera_0390"},
        method="ankle_avg",
        ankle_min_conf=0.3,
    )
    assert rewritten == 2  # one detection in each file

    for fname in ("camera390_tracking_results.json", "fixed_camera390_tracking_results.json"):
        body = json.loads((sct_scene / fname).read_text())
        assert body["00000001"]["WorldCoordinate"]["x"] == pytest.approx(105.0)
        assert body["00000001"]["WorldCoordinate"]["y"] == pytest.approx(805.0)
```

- [ ] **Step 2: Run and confirm failure**

```
.venv/bin/pytest tests/unit/test_world_projection.py::test_rewrite_bbox_bottom_is_noop -v
```

Expected: `FAILED` with `ImportError: cannot import name 'rewrite_world_coordinates'`.

- [ ] **Step 3: Implement `rewrite_world_coordinates`**

Add to `aic24_nvidia/world_projection.py`:

```python
def _load_homography(calib_path: Path):
    import numpy as np
    body = json.loads(Path(calib_path).read_text())
    return np.array(body["homography matrix"], dtype=np.float64)


def _rewrite_one_file(
    sct_json: Path,
    pose_lookup: dict,
    homography,
    method: str,
    ankle_min_conf: float,
) -> int:
    """Rewrite `WorldCoordinate` for every detection in one SCT JSON. Returns count."""
    body = json.loads(sct_json.read_text())
    n = 0
    for _serial, entry in body.items():
        if not isinstance(entry, dict):
            continue
        coord = entry.get("Coordinate")
        frame = entry.get("Frame")
        if coord is None or frame is None:
            continue
        try:
            x1 = int(round(float(coord["x1"])))
            y1 = int(round(float(coord["y1"])))
            x2 = int(round(float(coord["x2"])))
            y2 = int(round(float(coord["y2"])))
            frame_i = int(frame)
        except (KeyError, TypeError, ValueError):
            continue

        kps = pose_lookup.get((frame_i, (x1, y1, x2, y2)))
        x_img, y_img = _compute_image_point((x1, y1, x2, y2), kps, method, ankle_min_conf)
        wx, wy = _project_to_world(x_img, y_img, homography)
        # Skip NaN/inf — keep the original WorldCoordinate if projection blew up.
        import math
        if not (math.isfinite(wx) and math.isfinite(wy)):
            continue
        entry["WorldCoordinate"] = {"x": wx, "y": wy}
        n += 1
    sct_json.write_text(json.dumps(body))
    return n


def rewrite_world_coordinates(
    *,
    sct_scene_dir: Path,
    pose_scene_dir: Path,
    calib_root: Path,
    camera_map: dict[int, str],
    method: str,
    ankle_min_conf: float,
) -> int:
    """Rewrite `WorldCoordinate` in every per-camera SCT JSON in place.

    Args:
        sct_scene_dir: e.g. `mct.tmp/scene_001/` — contains
            `camera{N}_tracking_results.json` and
            `fixed_camera{N}_tracking_results.json`.
        pose_scene_dir: e.g. `Pose/scene_001/` — contains
            `<nvidia_cam>/<nvidia_cam>_out_keypoint.json`.
        calib_root: e.g. `adapted/Original/scene_001/` — contains
            `<nvidia_cam>/calibration.json` with `"homography matrix"`.
        camera_map: numeric_id -> nvidia_cam_name, e.g. `{390: "camera_0390"}`.
        method: one of `bbox_bottom | ankle_avg | ankle_lower | ankle_w_fallback`.
        ankle_min_conf: per-keypoint confidence floor used by ankle_w_fallback.

    Returns:
        Total number of detection rewrites across all files. When
        `method == "bbox_bottom"` this is a true no-op and returns 0.
    """
    sct_scene_dir = Path(sct_scene_dir)
    pose_scene_dir = Path(pose_scene_dir)
    calib_root = Path(calib_root)

    if method == "bbox_bottom":
        return 0   # no-op contract; baseline must be byte-identical

    total = 0
    for cam_id, nvidia_name in camera_map.items():
        pose_json = pose_scene_dir / nvidia_name / f"{nvidia_name}_out_keypoint.json"
        calib_json = calib_root / nvidia_name / "calibration.json"
        if not pose_json.exists() or not calib_json.exists():
            # Missing pose or calibration -> skip this camera (no rewrite).
            continue

        pose_lookup = _build_pose_lookup(pose_json)
        homography = _load_homography(calib_json)

        for stem in (f"camera{cam_id:03d}_tracking_results.json",
                     f"fixed_camera{cam_id:03d}_tracking_results.json"):
            sct_json = sct_scene_dir / stem
            if not sct_json.exists():
                continue
            total += _rewrite_one_file(
                sct_json, pose_lookup, homography, method, ankle_min_conf,
            )
    return total
```

- [ ] **Step 4: Run all tests in the file**

```
.venv/bin/pytest tests/unit/test_world_projection.py -v
```

Expected: all 15 tests pass.

- [ ] **Step 5: Commit**

```bash
git add aic24_nvidia/world_projection.py tests/unit/test_world_projection.py
git commit -m "feat(world_projection): rewrite_world_coordinates rewrites SCT JSONs in place"
```

---

## Phase C — Integrate into MCT stage

### Task C1: Call `rewrite_world_coordinates` from `stages/mct.py`

**Files:**
- Modify: `aic24_nvidia/stages/mct.py`
- Test: `tests/unit/test_mct_world_projection_wiring.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_mct_world_projection_wiring.py`:

```python
"""Verify stages/mct.py invokes rewrite_world_coordinates with the right args."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


def test_mct_does_not_call_rewrite_when_method_bbox_bottom(tmp_path, monkeypatch):
    """When world_projection.method is bbox_bottom (default), the rewrite hook is not invoked."""
    # We assert on the call by patching the function name as imported in mct.py.
    from aic24_nvidia.stages import mct as mct_stage

    called = {"count": 0}

    def fake_rewrite(**kwargs):
        called["count"] += 1
        return 0

    monkeypatch.setattr(mct_stage, "rewrite_world_coordinates", fake_rewrite)

    # Drive _maybe_rewrite_world_coordinates if it exists, else call run() with a
    # minimal stub config. To keep this test focused, we'll exercise the helper
    # directly.
    from aic24_nvidia.config import WorldProjectionCfg

    # The mct module exposes a small helper that drives the rewrite based on cfg.
    # The expected helper name is _maybe_rewrite_world_coordinates.
    mct_stage._maybe_rewrite_world_coordinates(
        cfg_world_projection=WorldProjectionCfg(method="bbox_bottom", ankle_min_conf=0.3),
        sct_scene_dir=tmp_path / "mct.tmp" / "scene_001",
        pose_scene_dir=tmp_path / "Pose" / "scene_001",
        calib_root=tmp_path / "adapted" / "Original" / "scene_001",
        camera_map={390: "camera_0390"},
    )

    assert called["count"] == 0   # short-circuit on no-op method


def test_mct_calls_rewrite_when_method_ankle_avg(tmp_path, monkeypatch):
    from aic24_nvidia.stages import mct as mct_stage
    from aic24_nvidia.config import WorldProjectionCfg

    captured = {}

    def fake_rewrite(**kwargs):
        captured.update(kwargs)
        return 7

    monkeypatch.setattr(mct_stage, "rewrite_world_coordinates", fake_rewrite)

    mct_stage._maybe_rewrite_world_coordinates(
        cfg_world_projection=WorldProjectionCfg(method="ankle_avg", ankle_min_conf=0.4),
        sct_scene_dir=tmp_path / "a",
        pose_scene_dir=tmp_path / "b",
        calib_root=tmp_path / "c",
        camera_map={390: "camera_0390"},
    )

    assert captured["method"] == "ankle_avg"
    assert captured["ankle_min_conf"] == pytest.approx(0.4)
    assert captured["camera_map"] == {390: "camera_0390"}
    assert captured["sct_scene_dir"] == tmp_path / "a"
```

- [ ] **Step 2: Run and confirm failure**

```
.venv/bin/pytest tests/unit/test_mct_world_projection_wiring.py -v
```

Expected: FAIL with `AttributeError: module 'aic24_nvidia.stages.mct' has no attribute '_maybe_rewrite_world_coordinates'` (or similar).

- [ ] **Step 3: Implement the helper + wire it into `run()`**

In `aic24_nvidia/stages/mct.py`, add the import near the top:

```python
from ..world_projection import rewrite_world_coordinates
```

And a tiny helper above `run`:

```python
def _maybe_rewrite_world_coordinates(
    *,
    cfg_world_projection,
    sct_scene_dir: Path,
    pose_scene_dir: Path,
    calib_root: Path,
    camera_map: dict[int, str],
) -> int:
    if cfg_world_projection.method == "bbox_bottom":
        return 0
    return rewrite_world_coordinates(
        sct_scene_dir=sct_scene_dir,
        pose_scene_dir=pose_scene_dir,
        calib_root=calib_root,
        camera_map=camera_map,
        method=cfg_world_projection.method,
        ankle_min_conf=cfg_world_projection.ankle_min_conf,
    )
```

Inside `run`, between `shutil.copytree(src_scene, dst_scene)` (line ~51) and the symlink loop, insert:

```python
        # Optional: override per-detection WorldCoordinate using ankle keypoints.
        # SCPT does not consume WorldCoordinate; this only affects MCT and eval.
        camera_map = _load_camera_map(run_dir)
        rewritten = _maybe_rewrite_world_coordinates(
            cfg_world_projection=cfg.world_projection,
            sct_scene_dir=dst_scene,
            pose_scene_dir=stage_dir(run_dir, "pose") / SCENE,
            calib_root=stage_dir(run_dir, "adapted") / "Original" / SCENE,
            camera_map=camera_map,
        )
        log.info("mct world_projection: method=%s rewrites=%d",
                 cfg.world_projection.method, rewritten)
```

Also add this `_load_camera_map` helper near the top of `stages/mct.py` (it mirrors the one already in `sct.py`):

```python
def _load_camera_map(run_dir: Path) -> dict[int, str]:
    """Read adapted/scene.json -> {numeric_id: nvidia_cam_name}.

    scene.json shape: {scene_name: {yachiyo_cam_name: nvidia_cam_name}}
    where yachiyo_cam_name is "camera_NNNN" and nvidia_cam_name is also "camera_NNNN".
    Numeric id is int(yachiyo_cam_name.split("_")[-1]).
    """
    scene_json = stage_dir(run_dir, "adapted") / "scene.json"
    if not scene_json.exists():
        return {}
    body = json.loads(scene_json.read_text())[SCENE]
    return {int(yk.split("_")[-1]): nvidia for yk, nvidia in body.items()}
```

And add `import json` at the top of mct.py if not already present.

Update `ctx.set_params` to record the projection knobs:

```python
        ctx.set_params({
            "tracking_params": params,
            "hard_world_gate": cfg.mct.hard_world_gate,
            "world_projection": {
                "method": cfg.world_projection.method,
                "ankle_min_conf": cfg.world_projection.ankle_min_conf,
                "rewrites": rewritten,
            },
            "propagated_via": "parameters_per_scene.py",
        })
```

- [ ] **Step 4: Run wiring tests**

```
.venv/bin/pytest tests/unit/test_mct_world_projection_wiring.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Run the full unit suite to confirm no regression**

```
.venv/bin/pytest tests/unit/ -v
```

Expected: all existing tests + 17 new tests pass.

- [ ] **Step 6: Commit**

```bash
git add aic24_nvidia/stages/mct.py tests/unit/test_mct_world_projection_wiring.py
git commit -m "feat(mct): invoke rewrite_world_coordinates between SCT staging and MCT subprocess"
```

---

## Phase D — EMA world-coord smoother

### Task D1: `smooth_world_tracks` in `world_tracks.py`

**Files:**
- Modify: `aic24_nvidia/world_tracks.py`
- Create: `tests/unit/test_world_smoother.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_world_smoother.py`:

```python
"""Unit tests for aic24_nvidia.world_tracks.smooth_world_tracks."""
from __future__ import annotations

import pytest


def test_smooth_none_is_identity():
    from aic24_nvidia.world_tracks import smooth_world_tracks

    rows = [
        (1, 100, 10.0, 20.0),
        (2, 100, 12.0, 21.0),
        (3, 100, 14.0, 22.0),
    ]
    out = smooth_world_tracks(rows, method="none", ema_alpha=0.3)
    assert out == rows


def test_smooth_ema_step_function():
    """EMA against a step input; first sample initializes; alpha controls weight."""
    from aic24_nvidia.world_tracks import smooth_world_tracks

    rows = [
        (1, 100, 0.0, 0.0),
        (2, 100, 10.0, 0.0),
        (3, 100, 10.0, 0.0),
    ]
    out = smooth_world_tracks(rows, method="ema", ema_alpha=0.5)
    # s_1 = obs_1 = 0.0
    # s_2 = 0.5*10 + 0.5*0 = 5.0
    # s_3 = 0.5*10 + 0.5*5 = 7.5
    assert out[0] == (1, 100, 0.0, 0.0)
    assert out[1] == (2, 100, 5.0, 0.0)
    assert out[2] == (3, 100, 7.5, 0.0)


def test_smooth_ema_isolates_per_gid():
    """Two gids are smoothed independently."""
    from aic24_nvidia.world_tracks import smooth_world_tracks

    rows = [
        (1, 100, 0.0, 0.0),
        (1, 200, 100.0, 100.0),
        (2, 100, 10.0, 0.0),
        (2, 200, 110.0, 100.0),
    ]
    out = smooth_world_tracks(rows, method="ema", ema_alpha=0.5)
    # Per-gid timeseries (sorted by frame):
    #   gid 100: (0,0) -> (5,0)
    #   gid 200: (100,100) -> (105,100)
    by_key = {(f, g): (x, y) for (f, g, x, y) in out}
    assert by_key[(1, 100)] == (0.0, 0.0)
    assert by_key[(2, 100)] == (5.0, 0.0)
    assert by_key[(1, 200)] == (100.0, 100.0)
    assert by_key[(2, 200)] == (105.0, 100.0)


def test_smooth_ema_single_observation_per_gid():
    """A gid with one point passes through unchanged."""
    from aic24_nvidia.world_tracks import smooth_world_tracks

    rows = [(7, 42, 1.5, 2.5)]
    out = smooth_world_tracks(rows, method="ema", ema_alpha=0.3)
    assert out == rows


def test_smooth_output_is_sorted_by_frame_then_gid():
    """Output order matches the existing aggregate_world_tracks contract."""
    from aic24_nvidia.world_tracks import smooth_world_tracks

    rows = [
        (3, 100, 0.0, 0.0),
        (1, 100, 0.0, 0.0),
        (2, 200, 0.0, 0.0),
        (2, 100, 0.0, 0.0),
    ]
    out = smooth_world_tracks(rows, method="ema", ema_alpha=0.5)
    keys = [(f, g) for (f, g, _x, _y) in out]
    assert keys == sorted(keys)


def test_smooth_invalid_method():
    from aic24_nvidia.world_tracks import smooth_world_tracks
    with pytest.raises(ValueError, match="method"):
        smooth_world_tracks([(1, 1, 0.0, 0.0)], method="kalman", ema_alpha=0.5)
```

- [ ] **Step 2: Run and confirm failure**

```
.venv/bin/pytest tests/unit/test_world_smoother.py::test_smooth_none_is_identity -v
```

Expected: FAIL with `ImportError: cannot import name 'smooth_world_tracks'`.

- [ ] **Step 3: Implement `smooth_world_tracks`**

Append to `aic24_nvidia/world_tracks.py`:

```python
def smooth_world_tracks(
    rows: list[tuple[int, int, float, float]],
    method: str,
    ema_alpha: float,
) -> list[tuple[int, int, float, float]]:
    """Apply temporal smoothing to per-(frame, gid) world coords.

    For method="ema":
        s_t = alpha * obs_t + (1 - alpha) * s_{t-1}
        first observation per gid initializes s.

    Per-gid timeseries are sorted by frame before smoothing; the returned rows
    are sorted by (frame, gid) to match aggregate_world_tracks's output order.

    Args:
        rows: list of (frame, gid, x, y).
        method: "none" (identity) or "ema".
        ema_alpha: smoothing weight in [0, 1]; only used when method=="ema".
    """
    if method == "none":
        return list(rows)
    if method != "ema":
        raise ValueError(f"smooth_world_tracks: unknown method {method!r}")

    # Group by gid and sort by frame within each gid.
    by_gid: dict[int, list[tuple[int, float, float]]] = {}
    for f, g, x, y in rows:
        by_gid.setdefault(g, []).append((f, x, y))
    for g in by_gid:
        by_gid[g].sort(key=lambda t: t[0])

    out: list[tuple[int, int, float, float]] = []
    for g, series in by_gid.items():
        s_x = s_y = 0.0
        for i, (f, x, y) in enumerate(series):
            if i == 0:
                s_x, s_y = x, y
            else:
                s_x = ema_alpha * x + (1.0 - ema_alpha) * s_x
                s_y = ema_alpha * y + (1.0 - ema_alpha) * s_y
            out.append((f, g, s_x, s_y))
    out.sort()
    return out
```

- [ ] **Step 4: Run all smoother tests**

```
.venv/bin/pytest tests/unit/test_world_smoother.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add aic24_nvidia/world_tracks.py tests/unit/test_world_smoother.py
git commit -m "feat(world_tracks): EMA smoother on per-(frame,gid) world coords"
```

---

### Task D2: Plumb the smoother through `stages/evaluate.py`

**Files:**
- Modify: `aic24_nvidia/stages/evaluate.py`
- Modify: `tests/unit/test_evaluate_wrapper.py` (add cases if there's a unit-testable seam; otherwise rely on integration)

- [ ] **Step 1: Inspect the existing test file to see what's already covered**

```
.venv/bin/pytest tests/unit/test_evaluate_wrapper.py --collect-only
```

If the existing tests already exercise `_eval_mct_world` against a synthetic MCT JSON, mirror that style for the new test. Otherwise add the simple wiring test below.

- [ ] **Step 2: Write a wiring test**

Append to (or create) `tests/unit/test_evaluate_world_smoother_wiring.py`:

```python
"""Verify _eval_mct_world calls smooth_world_tracks with the right knobs."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _stub_cfg(smoothing_method="none", ema_alpha=0.3):
    from aic24_nvidia.config import WorldSmoothingCfg, EvalCfg
    cfg = MagicMock()
    cfg.eval = EvalCfg(world_d_max=1.0)
    cfg.world_smoothing = WorldSmoothingCfg(method=smoothing_method, ema_alpha=ema_alpha)
    return cfg


def _write_minimal_mct(tmp_path: Path) -> Path:
    """One detection in one cam, valid for aggregate_world_tracks."""
    mct = tmp_path / "fixed_whole_tracking_results.json"
    mct.write_text(json.dumps({
        "390": {
            "00000001": {
                "Frame": 5,
                "GlobalOfflineID": 1,
                "Coordinate": {"x1": 50, "y1": 100, "x2": 150, "y2": 850},
                "WorldCoordinate": {"x": 1.0, "y": 2.0},
            }
        }
    }))
    return mct


def test_eval_mct_world_calls_smoother(tmp_path, monkeypatch):
    """smooth_world_tracks is invoked with the cfg's method + alpha."""
    from aic24_nvidia.stages import evaluate as evaluate_stage

    captured = {}

    def fake_smooth(rows, method, ema_alpha):
        captured["method"] = method
        captured["ema_alpha"] = ema_alpha
        return rows

    monkeypatch.setattr(evaluate_stage, "smooth_world_tracks", fake_smooth)

    cfg = _stub_cfg(smoothing_method="ema", ema_alpha=0.4)
    ctx = MagicMock()
    ctx.work_dir = tmp_path

    mct_global = _write_minimal_mct(tmp_path)
    adapted_root = tmp_path / "adapted"
    # We do not need a real gt_world.txt for this test — the smoothing call
    # happens before TrackEval; provide one to clear the early-exit guard.
    (adapted_root).mkdir()
    (adapted_root / "scene_001_gt_world.txt").write_text("5,1,1.0,2.0\n")

    # The function may exit early if TrackEval isn't available — that's fine;
    # we only assert the smoother was called.
    try:
        evaluate_stage._eval_mct_world(cfg, ctx, str(mct_global), adapted_root)
    except Exception:
        pass

    assert captured.get("method") == "ema"
    assert captured.get("ema_alpha") == pytest.approx(0.4)
```

- [ ] **Step 3: Run and confirm failure**

```
.venv/bin/pytest tests/unit/test_evaluate_world_smoother_wiring.py -v
```

Expected: FAIL with `AttributeError: module 'aic24_nvidia.stages.evaluate' has no attribute 'smooth_world_tracks'` (or similar — the import won't yet exist as a module attribute).

- [ ] **Step 4: Wire the smoother into `_eval_mct_world`**

At the top of `aic24_nvidia/stages/evaluate.py`, change:

```python
from ..world_tracks import aggregate_world_tracks, write_world_pred
```

to:

```python
from ..world_tracks import aggregate_world_tracks, smooth_world_tracks, write_world_pred
```

In `_eval_mct_world`, change the block:

```python
        rows, dropped = aggregate_world_tracks(Path(mct_global))
        if not rows:
            return {"skipped": "MCT produced no valid world points"}
        pred_txt = ctx.work_dir / "mct_world_pred.txt"
        write_world_pred(rows, pred_txt)
```

to:

```python
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
```

And update the `ctx.set_params(...)` call at the end of `run()` to record the smoother config:

```python
        ctx.set_params({
            "trackeval": "MOTChallenge + NvidiaMTMCWorld",
            "scope": "per-camera SCT + scene 3D-world MCT",
            "world_d_max": cfg.eval.world_d_max,
            "world_smoothing": {
                "method": cfg.world_smoothing.method,
                "ema_alpha": cfg.world_smoothing.ema_alpha,
            },
        })
```

- [ ] **Step 5: Run wiring test**

```
.venv/bin/pytest tests/unit/test_evaluate_world_smoother_wiring.py -v
```

Expected: 1 passed.

- [ ] **Step 6: Run the full unit suite to confirm no regression**

```
.venv/bin/pytest tests/unit/ -v
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add aic24_nvidia/stages/evaluate.py tests/unit/test_evaluate_world_smoother_wiring.py
git commit -m "feat(evaluate): plumb world_smoothing through to smooth_world_tracks"
```

---

## Phase E — Experiment registry

### Task E1: Add the two experiments to `experiments/registry.yaml`

**Files:**
- Modify: `experiments/registry.yaml`

- [ ] **Step 1: Append the experiment blocks**

Append at the bottom of `experiments/registry.yaml` (before any closing comments, after `world_d_max`):

```yaml
  - id: ankle_footpoint_sweep
    description: "Use RTMPose ankle keypoints instead of bbox-bottom-center for world projection."
    hypothesis: "Ankle footpoints reduce 2D->world projection error, raising 3D-world HOTA."
    base_config: configs/baseline.yaml
    rerun_from: mct
    variants:
      - name: "bbox_bottom"          # = baseline (control)
        overrides:
          world_projection:
            method: bbox_bottom
      - name: "ankle_avg"
        overrides:
          world_projection:
            method: ankle_avg
      - name: "ankle_lower"
        overrides:
          world_projection:
            method: ankle_lower
      - name: "ankle_w_fallback"
        overrides:
          world_projection:
            method: ankle_w_fallback
            ankle_min_conf: 0.3      # explicit; matches the baseline default

  - id: world_smoother_sweep
    description: "EMA temporal smoothing on per-(frame, gid) world coords (eval-only)."
    hypothesis: "Smoothing removes per-frame projection noise, raising world HOTA / IDF1."
    base_config: configs/baseline.yaml
    rerun_from: evaluate
    variants:
      - name: "none"                 # = baseline (control)
        overrides:
          world_smoothing:
            method: none
      - name: "ema_0.3"
        overrides:
          world_smoothing:
            method: ema
            ema_alpha: 0.3
      - name: "ema_0.5"
        overrides:
          world_smoothing:
            method: ema
            ema_alpha: 0.5
      - name: "ema_0.7"
        overrides:
          world_smoothing:
            method: ema
            ema_alpha: 0.7
```

- [ ] **Step 2: Also update the comment block at the top of `registry.yaml`**

In the `override changes -> rerun_from` table (lines 13-25 of the file), add two rows so future editors can find the convention:

```
#   world_projection.*              -> mct
#   world_smoothing.*               -> evaluate
```

- [ ] **Step 3: Verify the harness can list both new experiments**

```
.venv/bin/python experiments/run.py list
```

Expected: output table includes `ankle_footpoint_sweep` (4 variants, rerun_from=mct) and `world_smoother_sweep` (4 variants, rerun_from=evaluate).

- [ ] **Step 4: Verify a single variant's config materializes cleanly**

```
.venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from experiments._lib import load_registry, load_yaml, deep_merge
exps = load_registry('experiments/registry.yaml')
exp = next(e for e in exps if e['id'] == 'ankle_footpoint_sweep')
base = load_yaml('configs/baseline.yaml')
v = next(v for v in exp['variants'] if v['name'] == 'ankle_avg')
merged = deep_merge(base, v['overrides'])
print('world_projection:', merged['world_projection'])
"
```

Expected: `world_projection: {'method': 'ankle_avg', 'ankle_min_conf': 0.3}`.

- [ ] **Step 5: Commit**

```bash
git add experiments/registry.yaml
git commit -m "feat(experiments): add ankle_footpoint_sweep + world_smoother_sweep"
```

---

## Phase F — Verification

### Task F1: End-to-end smoke + existing-test regression

**Files:** none modified — verification only.

- [ ] **Step 1: Run the full unit test suite**

```
.venv/bin/pytest tests/unit/ -v
```

Expected: 0 failures. All ~80+ tests pass.

- [ ] **Step 2: Run the synthetic-scene integration test**

```
.venv/bin/pytest tests/integration/test_tiny_scene.py -v
```

Expected: 0 failures (the default config is no-op).

- [ ] **Step 3: Verify the baseline config still loads**

```
.venv/bin/python -c "from aic24_nvidia.config import load_config; c = load_config('configs/baseline.yaml'); print('world_projection:', c.world_projection); print('world_smoothing:', c.world_smoothing)"
```

Expected:
```
world_projection: WorldProjectionCfg(method='bbox_bottom', ankle_min_conf=0.3)
world_smoothing: WorldSmoothingCfg(method='none', ema_alpha=0.3)
```

- [ ] **Step 4: (Pre-baseline; do not run if you don't have the cached pipeline)**

If `outputs/baseline/` does not exist, build it now (long-running, ~45-60 min on a 6 GB RTX 3050):

```
.venv/bin/python experiments/run.py ensure-baseline
```

Expected: terminates with `baseline already present` or completes with `metrics.json` at `outputs/baseline/evaluate/metrics.json`. The HOTA/IDF1/MOTA numbers should approximately match the locked baseline (HOTA ≈ 0.758, world HOTA ≈ 0.458).

- [ ] **Step 5: Run the control variant of each experiment first**

```
.venv/bin/python experiments/run.py run ankle_footpoint_sweep --variant bbox_bottom
.venv/bin/python experiments/run.py run world_smoother_sweep --variant none
```

Expected: each completes in a few minutes (MCT+evaluate for the first; evaluate-only for the second). The resulting metrics should be **byte-identical** (or within floating-point noise) to the baseline — this validates the no-op contract.

```
.venv/bin/python experiments/compare.py --experiment ankle_footpoint_sweep
.venv/bin/python experiments/compare.py --experiment world_smoother_sweep
```

The `bbox_bottom` and `none` rows should match baseline within 1e-6.

- [ ] **Step 6: Run the remaining variants**

```
.venv/bin/python experiments/run.py run ankle_footpoint_sweep
.venv/bin/python experiments/run.py run world_smoother_sweep
```

Expected: full sweeps finish (~22 min for ankle, ~2 min for smoother).

- [ ] **Step 7: Inspect the comparison table**

```
.venv/bin/python experiments/compare.py --sort-by mct_world.HOTA
```

Look for: at least one ankle variant beating bbox_bottom on `mct_world.HOTA`, and at least one EMA variant beating `none`. Record the best of each in a follow-up note.

- [ ] **Step 8: Final regression sweep**

```
.venv/bin/pytest tests/ -v
```

Expected: all green.

- [ ] **Step 9: Commit any incidental fixes**

If steps 1-8 surface a bug (e.g., a path mismatch when running on real data), fix it in a separate commit. Do NOT amend earlier commits.

---

## Self-review notes

- **Spec coverage:** Both improvements (#1 ankle footpoints, #2 EMA smoother) implemented. Both experiments wired into registry. Tests cover all four ankle methods + the no-op default + pose-miss fallback + EMA basics + per-gid isolation + cfg validation.
- **No placeholders:** every step has exact code, exact paths, and exact commands.
- **TDD ordering:** every implementation step is preceded by a failing test and followed by a passing test.
- **Cleanup:** existing baseline behavior is preserved (defaults are bbox_bottom and none). Reverting either improvement is one config edit.
- **Out of scope:** Kalman smoother, per-camera detector tuning, ReID fine-tuning. Cross-product (ankle × EMA) is a follow-up only if both sweeps show positive deltas.
