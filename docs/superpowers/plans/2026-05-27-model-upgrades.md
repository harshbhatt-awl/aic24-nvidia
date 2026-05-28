# Model Upgrades Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the detect/reid/pose models (YOLOX-X → YOLO11-x, OSNet → SOLIDER Swin-Small, HRNet/mmpose → RTMPose-l via rtmlib) with in-package adapters that reproduce the existing on-disk formats byte-for-byte, so SCT/MCT/evaluate stay untouched; then reclaim disk space.

**Architecture:** Three new modules under `aic24_nvidia/models/` each take (scene, camera list, run dir, config) and write exactly the files the upstream injected scripts wrote. The stage modules (`detect.py`/`reid.py`/`pose.py`) call these instead of `subprocess.run([...external...])`, keeping the symlink trick, `atomic_stage`, manifests, and validation unchanged. Pose moves into the main `.venv` (rtmlib is ONNX-only), eliminating `.venv-pose`.

**Tech Stack:** Python 3.14 (`.venv`), `ultralytics` (YOLO11), `rtmlib`+`onnxruntime-gpu` (RTMPose), vendored SOLIDER Swin model + `timm`, numpy, opencv, Pillow, pytest.

**Reference contracts** (must reproduce exactly — verified against upstream source):

- **Detection `.txt`** (`Detection/scene_001/camera_NNNN.txt`), one line per detection, score > 0.1, coords clamped to [0,1920]×[0,1080], cast to int:
  `cam,frame_id,1,x1,y1,x2,y2,score\n` (frame_id 1-indexed; score = raw float).
- **Detection `.json`** (`camera_NNNN.json`): dict keyed `str(u_num).zfill(8)` (u_num = global 0-based counter across the camera), each value `{"Frame": int, "ImgPath": "<path rel to root>", "NpyPath": "", "Coordinate": {"x1":int,"y1":int,"x2":int,"y2":int}, "ClusterID": null, "OfflineID": null}`.
- **ReID `.npy`** (`EmbedFeature/scene_001/camera_NNNN/feature_<frame>_<u_num>_<x1>_<x2>_<y1>_<y2>_<conf>.npy`): single float32 vector. `<frame>` = frame number; `<u_num>` = 1-based per-detection counter (= json-index + 1); coord order is **x1, x2, y1, y2** (ints); `<conf>` = the detection's conf string with `.` removed. Also set `jf[str(idx).zfill(8)]["NpyPath"] = "<scene>/<cam>/feature_...npy"` (idx = 0-based).
- **Pose `.json`** (`Pose/scene_001/camera_NNNN/camera_NNNN_out_keypoint.json`): `{ "<frame_id>": [ {"bbox":[x1,y1,x2,y2,1.0], "keypoints":[[x,y,score], ...17 COCO...]} , ...] }`. bbox ints must equal the detection bbox ints (YACHIYO matches on `f"{frame}_{x1}_{y1}_{x2}_{y2}"`).

Frames for all stages come from `Original/<scene>/<cam>/Frame/<frame:06d>.jpg` (produced by the `frames` stage; `Original` is symlinked).

---

## Task 1: Add dependencies + model-weights scaffolding

**Files:**
- Modify: `pyproject.toml` (dependencies list, lines 10-17)
- Create: `aic24_nvidia/models/__init__.py`
- Create: `weights/.gitignore`

- [ ] **Step 1: Add deps to `pyproject.toml`**

Edit the `dependencies = [...]` array to add:
```toml
    "ultralytics>=8.3",
    "rtmlib>=0.0.13",
    "onnxruntime-gpu>=1.17",
    "timm>=0.9",
```

- [ ] **Step 2: Install into the shared venv**

Run: `.venv/bin/python -m pip install "ultralytics>=8.3" "rtmlib>=0.0.13" "onnxruntime-gpu>=1.17" "timm>=0.9"`
Expected: installs succeed; `.venv/bin/python -c "import ultralytics, rtmlib, timm"` prints nothing and exits 0.

- [ ] **Step 3: Create package + weights dirs**

```bash
mkdir -p aic24_nvidia/models weights
touch aic24_nvidia/models/__init__.py
printf '*\n!.gitignore\n' > weights/.gitignore
```

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml aic24_nvidia/models/__init__.py weights/.gitignore
git commit -m "build: add ultralytics/rtmlib/timm deps + models package scaffold"
```

---

## Task 2: Detection adapter (YOLO11-x)

**Files:**
- Create: `aic24_nvidia/models/detect_yolo.py`
- Test: `tests/unit/test_detect_yolo.py`

- [ ] **Step 1: Write the failing test (format only, model mocked)**

```python
# tests/unit/test_detect_yolo.py
import json
from pathlib import Path
import numpy as np
from aic24_nvidia.models import detect_yolo


def test_write_detection_outputs_matches_upstream_format(tmp_path):
    # two detections in frame 1, one in frame 2, for one camera
    dets_by_frame = {
        1: [(10.4, 20.6, 110.2, 220.9, 0.91), (300.0, 50.0, 360.0, 180.0, 0.55)],
        2: [(12.0, 22.0, 112.0, 222.0, 0.88)],
    }
    det_dir = tmp_path / "detect" / "scene_001"
    detect_yolo._write_camera(
        det_dir=det_dir, cam="camera_0390", dets_by_frame=dets_by_frame,
        img_rel_for_frame=lambda f: f"Original/scene_001/camera_0390/Frame/{f:06d}.jpg",
    )
    txt = (det_dir / "camera_0390.txt").read_text().splitlines()
    assert txt[0] == "camera_0390,1,1,10,20,110,220,0.91"
    assert txt[2] == "camera_0390,2,1,12,22,112,222,0.88"
    j = json.loads((det_dir / "camera_0390.json").read_text())
    assert set(j.keys()) == {"00000000", "00000001", "00000002"}
    assert j["00000000"] == {
        "Frame": 1, "ImgPath": "Original/scene_001/camera_0390/Frame/000001.jpg",
        "NpyPath": "", "Coordinate": {"x1": 10, "y1": 20, "x2": 110, "y2": 220},
        "ClusterID": None, "OfflineID": None,
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_detect_yolo.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute '_write_camera'`.

- [ ] **Step 3: Implement `detect_yolo.py`**

```python
# aic24_nvidia/models/detect_yolo.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Callable

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
            lines.append(f"{cam},{frame_id},1,{x1},{y1},{x2},{y2},{score}")
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_detect_yolo.py -v`
Expected: PASS.

- [ ] **Step 5: Add the model-inference entry point (not unit-tested; GPU)**

Append to `detect_yolo.py`:
```python
def run_detection(scene_dir, det_out_dir, cams, conf_thresh, nms_iou, weights="yolo11x.pt"):
    """Run YOLO11 person detection over Original/<scene>/<cam>/Frame/*.jpg.

    scene_dir: <run>/detect/Original/<scene>  is NOT used; we read frames from
    the Original symlink: <root>/Original/<scene>/<cam>/Frame/<f:06d>.jpg.
    """
    from ultralytics import YOLO
    import cv2  # noqa: F401  (ultralytics handles IO)
    model = YOLO(weights)
    scene = Path(scene_dir).name
    for cam in cams:
        frame_dir = Path(scene_dir) / cam / "Frame"
        frame_paths = sorted(frame_dir.glob("*.jpg"))
        dets_by_frame: dict[int, list] = {}
        for fp in frame_paths:
            frame_id = int(fp.stem)  # 000001 -> 1
            res = model.predict(str(fp), classes=[0], conf=conf_thresh,
                                iou=nms_iou, imgsz=1920, verbose=False)[0]
            rows = []
            for b in res.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                rows.append((x1, y1, x2, y2, float(b.conf[0])))
            if rows:
                dets_by_frame[frame_id] = rows
        _write_camera(
            det_dir=Path(det_out_dir) / scene, cam=cam, dets_by_frame=dets_by_frame,
            img_rel_for_frame=lambda f, c=cam, s=scene: f"Original/{s}/{c}/Frame/{f:06d}.jpg",
        )
```
Note: YOLO COCO person class index is `0`. `frame_id = int(fp.stem)` matches upstream 1-based frames.

- [ ] **Step 6: Commit**

```bash
git add aic24_nvidia/models/detect_yolo.py tests/unit/test_detect_yolo.py
git commit -m "feat(models): YOLO11 detection adapter with byte-compatible output"
```

---

## Task 3: Wire `stages/detect.py` to the adapter

**Files:**
- Modify: `aic24_nvidia/stages/detect.py` (replace subprocess block, lines ~42-83)

- [ ] **Step 1: Replace the upstream subprocess call**

In `run()`, inside `with atomic_stage(...) as ctx:`, replace the `subprocess.run([...aic24_get_detection.py...])` block with:
```python
        from ..models import detect_yolo
        original = cfg.external_root / "Original"
        scene_src = original / SCENE
        cams = sorted(p.name for p in scene_src.iterdir() if p.is_dir())
        detect_yolo.run_detection(
            scene_dir=scene_src,
            det_out_dir=ctx.work_dir,           # writes <work_dir>/scene_001/camera_*.{txt,json}
            cams=cams,
            conf_thresh=cfg.detect.conf_thresh,
            nms_iou=cfg.detect.nms_iou,
            weights="yolo11x.pt",
        )
```
Keep the `Detection` symlink setup, `_per_cam_detection_files`, validation loop, manifest calls. Update `set_params`:
```python
        ctx.set_params({
            "model": "yolo11x",
            "conf_thresh": cfg.detect.conf_thresh,
            "nms_iou": cfg.detect.nms_iou,
        })
```
Remove the `injected = botsort/...; if not injected.exists()` guard and the BoT-SORT existence check (no longer needed). Drop the now-unused `subprocess` import if nothing else uses it.

- [ ] **Step 2: Run existing detect-related unit tests**

Run: `.venv/bin/python -m pytest tests/unit -k detect -v`
Expected: PASS (no GPU paths exercised in unit tests).

- [ ] **Step 3: Commit**

```bash
git add aic24_nvidia/stages/detect.py
git commit -m "feat(detect): use YOLO11 adapter; drop BoT-SORT/YOLOX dependency"
```

---

## Task 4: ReID adapter (SOLIDER Swin-Small) — vendor model + format

**Files:**
- Create: `aic24_nvidia/models/solider/` (vendored SOLIDER Swin model code)
- Create: `aic24_nvidia/models/reid_solider.py`
- Test: `tests/unit/test_reid_solider.py`

- [ ] **Step 1: Vendor the SOLIDER model definition**

Fetch these specific files from `https://github.com/tinyvision/SOLIDER-REID` into `aic24_nvidia/models/solider/`:
- `model/backbones/swin_transformer.py` → `aic24_nvidia/models/solider/swin_transformer.py`
- `model/make_model.py` → `aic24_nvidia/models/solider/make_model.py`
Add `aic24_nvidia/models/solider/__init__.py` (empty). Adjust the two files' relative imports to package-local (`from .swin_transformer import ...`). Download the Swin-Small ReID weights (`swin_small_market.pth` or MSMT variant from the SOLIDER-REID release table) to `weights/solider_swin_small.pth`.

Run: `.venv/bin/python -c "from aic24_nvidia.models.solider.make_model import make_model"` → exits 0.

- [ ] **Step 2: Write the failing format test (embedding mocked)**

```python
# tests/unit/test_reid_solider.py
import json
import numpy as np
from PIL import Image
from aic24_nvidia.models import reid_solider


def test_npy_filename_and_json_update_match_upstream(tmp_path, monkeypatch):
    scene, cam = "scene_001", "camera_0390"
    # detection txt: one det in frame 5
    det_dir = tmp_path / "Detection" / scene
    det_dir.mkdir(parents=True)
    (det_dir / f"{cam}.txt").write_text("camera_0390,5,1,10,20,110,220,0.91\n")
    (det_dir / f"{cam}.json").write_text(json.dumps(
        {"00000000": {"Frame": 5, "ImgPath": "x", "NpyPath": "",
                      "Coordinate": {"x1":10,"y1":20,"x2":110,"y2":220},
                      "ClusterID": None, "OfflineID": None}}))
    # frame image
    frame_dir = tmp_path / "Original" / scene / cam / "Frame"
    frame_dir.mkdir(parents=True)
    Image.new("RGB", (1920, 1080)).save(frame_dir / "000005.jpg")
    emb_dir = tmp_path / "EmbedFeature"

    monkeypatch.setattr(reid_solider, "_embed", lambda crop: np.ones(768, dtype=np.float32))
    reid_solider.extract_camera(
        det_scene_dir=det_dir, original_scene_dir=tmp_path / "Original" / scene,
        emb_out_dir=emb_dir, scene=scene, cam=cam, embed=reid_solider._embed)

    files = list((emb_dir / scene / cam).glob("*.npy"))
    assert len(files) == 1
    # frame=5, u_num=1, x1=10,x2=110,y1=20,y2=220, conf "0.91"->"091"
    assert files[0].name == "feature_5_1_10_110_20_220_091.npy"
    assert np.load(files[0]).shape == (768,)
    j = json.loads((det_dir / f"{cam}.json").read_text())
    assert j["00000000"]["NpyPath"] == f"{scene}/{cam}/feature_5_1_10_110_20_220_091.npy"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_reid_solider.py -v`
Expected: FAIL — `AttributeError: ... extract_camera`.

- [ ] **Step 4: Implement `reid_solider.py`**

```python
# aic24_nvidia/models/reid_solider.py
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image

_MODEL = None  # lazy global


def _embed(crop: "Image.Image") -> np.ndarray:
    """Embed a PIL crop to a 1D float32 vector via SOLIDER Swin-Small."""
    import torch
    import torchvision.transforms as T
    global _MODEL
    if _MODEL is None:
        from .solider.make_model import make_model
        _MODEL = make_model(weights=str(Path("weights/solider_swin_small.pth").resolve()))
        _MODEL.eval().cuda()
    tf = T.Compose([T.Resize([256, 128]), T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    x = tf(crop.convert("RGB")).unsqueeze(0).cuda()
    with torch.no_grad():
        feat = _MODEL(x)
    return feat.cpu().numpy()[0].astype(np.float32)


def extract_camera(det_scene_dir, original_scene_dir, emb_out_dir, scene, cam, embed=_embed):
    """Reproduce aic24_extract.py output for one camera using `embed`."""
    det_scene_dir = Path(det_scene_dir)
    txt = det_scene_dir / f"{cam}.txt"
    json_path = det_scene_dir / f"{cam}.json"
    dets = np.genfromtxt(txt, dtype=str, delimiter=",")
    if dets.ndim == 1:
        dets = dets.reshape(1, -1)
    with open(json_path) as f:
        jf = json.load(f)
    out = Path(emb_out_dir) / scene / cam
    out.mkdir(parents=True, exist_ok=True)
    u_num = 0
    for idx, (_cam, frame, _cls, x1, y1, x2, y2, conf) in enumerate(dets):
        u_num += 1
        cur_frame = int(frame)
        xi1, yi1, xi2, yi2 = int(float(x1)), int(float(y1)), int(float(x2)), int(float(y2))
        fname = "feature_{}_{}_{}_{}_{}_{}_{}.npy".format(
            cur_frame, u_num, xi1, xi2, yi1, yi2, str(conf).replace(".", ""))
        img_path = Path(original_scene_dir) / cam / "Frame" / (frame.zfill(6) + ".jpg")
        crop = Image.open(img_path).crop((float(x1), float(y1), float(x2), float(y2)))
        np.save(out / fname, embed(crop))
        jf[str(idx).zfill(8)]["NpyPath"] = os.path.join(scene, cam, fname)
    with open(json_path, "w") as f:
        json.dump(jf, f, ensure_ascii=False)


def run_reid(det_scene_dir, original_scene_dir, emb_out_dir, scene, cams):
    for cam in cams:
        extract_camera(det_scene_dir, original_scene_dir, emb_out_dir, scene, cam)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_reid_solider.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add aic24_nvidia/models/solider aic24_nvidia/models/reid_solider.py tests/unit/test_reid_solider.py
git commit -m "feat(models): SOLIDER Swin-Small ReID adapter (byte-compatible npy/json)"
```

---

## Task 5: Wire `stages/reid.py` + verify YACHIYO has no hardcoded 512-dim

**Files:**
- Modify: `aic24_nvidia/stages/reid.py` (replace subprocess block)

- [ ] **Step 1: Grep YACHIYO for hardcoded embedding dim**

Run: `grep -rn "512" external/AIC24_Track1_YACHIYO_RIIPS/tracking/src/ | grep -iv "comment"`
Expected: no line that allocates/asserts a 512-length feature buffer. If found (e.g. `np.zeros(512)`), record it; cosine-sim code is dim-agnostic so most hits are unrelated. Document findings in the commit message.

- [ ] **Step 2: Replace the upstream subprocess call**

In `reid.py` `run()`, replace the `subprocess.run([...aic24_extract.py...])` and PYTHONPATH block with:
```python
        from ..models import reid_solider
        original = cfg.external_root / "Original"
        det_scene = stage_dir(run_dir, "detect") / SCENE
        cams = sorted(p.stem for p in det_scene.glob("camera_*.txt"))
        reid_solider.run_reid(
            det_scene_dir=det_scene,
            original_scene_dir=original / SCENE,
            emb_out_dir=ctx.work_dir,    # <work_dir>/scene_001/camera_*/feature_*.npy
            scene=SCENE,
            cams=cams,
        )
```
Keep the `EmbedFeature` symlink, `_per_cam_feature_counts`, validation, manifest. Update `set_params` model name to `"solider_swin_small"`. Remove the `drid`/`injected` existence guards and the `subprocess`/`os` PYTHONPATH code if unused.

- [ ] **Step 3: Run reid unit tests**

Run: `.venv/bin/python -m pytest tests/unit -k reid -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add aic24_nvidia/stages/reid.py
git commit -m "feat(reid): use SOLIDER adapter; note YACHIYO dim-agnostic check"
```

---

## Task 6: Pose adapter (RTMPose-l via rtmlib)

**Files:**
- Create: `aic24_nvidia/models/pose_rtmpose.py`
- Test: `tests/unit/test_pose_rtmpose.py`

- [ ] **Step 1: Write the failing format test (keypoints mocked)**

```python
# tests/unit/test_pose_rtmpose.py
import json
import numpy as np
from PIL import Image
from aic24_nvidia.models import pose_rtmpose


def test_pose_json_schema_and_bbox_keys(tmp_path, monkeypatch):
    scene, cam = "scene_001", "camera_0390"
    det_dir = tmp_path / "Detection" / scene
    det_dir.mkdir(parents=True)
    (det_dir / f"{cam}.txt").write_text(
        "camera_0390,1,1,10,20,110,220,0.91\ncamera_0390,1,1,50,60,90,180,0.7\n")
    frame_dir = tmp_path / "Original" / scene / cam / "Frame"
    frame_dir.mkdir(parents=True)
    Image.new("RGB", (1920, 1080)).save(frame_dir / "000001.jpg")
    pose_out = tmp_path / "Pose"

    # mock: 17 keypoints all (1.0,2.0,0.9)
    monkeypatch.setattr(pose_rtmpose, "_estimate",
                        lambda img, bboxes: [[[1.0, 2.0, 0.9]] * 17 for _ in bboxes])
    pose_rtmpose.run_pose(det_scene_dir=det_dir,
                          original_scene_dir=tmp_path / "Original" / scene,
                          pose_out_dir=pose_out, scene=scene, cams=[cam])

    out = pose_out / scene / cam / f"{cam}_out_keypoint.json"
    j = json.loads(out.read_text())
    assert list(j.keys()) == ["1"]
    assert len(j["1"]) == 2
    assert j["1"][0]["bbox"] == [10, 20, 110, 220, 1.0]
    assert len(j["1"][0]["keypoints"]) == 17
    assert j["1"][0]["keypoints"][0] == [1.0, 2.0, 0.9]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_pose_rtmpose.py -v`
Expected: FAIL — `AttributeError: ... run_pose`.

- [ ] **Step 3: Implement `pose_rtmpose.py`**

```python
# aic24_nvidia/models/pose_rtmpose.py
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

_MODEL = None


def _estimate(img: np.ndarray, bboxes: list[list[float]]) -> list[list[list[float]]]:
    """RTMPose-l top-down. Returns per-bbox list of 17 [x,y,score]."""
    global _MODEL
    if _MODEL is None:
        from rtmlib import RTMPose
        _MODEL = RTMPose(
            onnx_model="https://download.openmmlab.com/mmpose/v1/projects/"
                       "rtmposev1/onnx_sdk/rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.zip",
            model_input_size=(192, 256), backend="onnxruntime", device="cuda")
    keypoints, scores = _MODEL(img, bboxes=np.array(bboxes, dtype=np.float32))
    out = []
    for kp, sc in zip(keypoints, scores):
        out.append([[float(x), float(y), float(s)] for (x, y), s in zip(kp, sc)])
    return out


def run_pose(det_scene_dir, original_scene_dir, pose_out_dir, scene, cams, estimate=_estimate):
    import cv2
    det_scene_dir = Path(det_scene_dir)
    for cam in cams:
        txt = det_scene_dir / f"{cam}.txt"
        dets = np.genfromtxt(txt, dtype=str, delimiter=",")
        if dets.ndim == 1:
            dets = dets.reshape(1, -1)
        by_frame: dict[int, list] = defaultdict(list)
        for (_c, frame, _cls, x1, y1, x2, y2, _conf) in dets:
            by_frame[int(frame)].append((int(x1), int(y1), int(x2), int(y2)))
        save = {}
        for frame_id in sorted(by_frame):
            img_path = Path(original_scene_dir) / cam / "Frame" / f"{frame_id:06d}.jpg"
            img = cv2.imread(str(img_path))
            bboxes = [[x1, y1, x2, y2] for (x1, y1, x2, y2) in by_frame[frame_id]]
            kpts = estimate(img, [[float(v) for v in b] for b in bboxes])
            people = []
            for (x1, y1, x2, y2), kp in zip(bboxes, kpts):
                people.append({"bbox": [x1, y1, x2, y2, 1.0], "keypoints": kp})
            save[str(frame_id)] = people
        out_dir = Path(pose_out_dir) / scene / cam
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"{cam}_out_keypoint.json", "w") as f:
            json.dump(save, f)
```
Note: confirm RTMPose-body7 returns COCO-17 ordering (nose..ankles) matching YACHIYO `tracking/src/pose.py` indices during Task 9 smoke run.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_pose_rtmpose.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aic24_nvidia/models/pose_rtmpose.py tests/unit/test_pose_rtmpose.py
git commit -m "feat(models): RTMPose-l pose adapter via rtmlib (byte-compatible json)"
```

---

## Task 7: Wire `stages/pose.py` to the adapter (drop `.venv-pose`)

**Files:**
- Modify: `aic24_nvidia/stages/pose.py` (replace per-cam subprocess loop + `.venv-pose` invocation)

- [ ] **Step 1: Replace the mmpose subprocess loop**

Replace the `pose_python = .../.venv-pose/...` block and the `for cam in cams: subprocess.run(...)` loop with:
```python
        from ..models import pose_rtmpose
        original = cfg.external_root / "Original"
        det_scene = stage_dir(run_dir, "detect") / SCENE
        pose_rtmpose.run_pose(
            det_scene_dir=det_scene,
            original_scene_dir=original / SCENE,
            pose_out_dir=ctx.work_dir,   # <work_dir>/scene_001/camera_*/camera_*_out_keypoint.json
            scene=SCENE,
            cams=cams,
        )
```
Keep the `Pose` symlink, `_per_cam_pose_files`, validation, manifest. Update `set_params` to `{"keypoint_conf": cfg.pose.keypoint_conf, "model": "rtmpose-l"}`. Remove the `mmpose`/`injected` existence guards, the `HRNET_CONFIG`/`HRNET_CKPT` constants, and the `.venv-pose` existence check.

- [ ] **Step 2: Run pose unit tests**

Run: `.venv/bin/python -m pytest tests/unit -k pose -v`
Expected: PASS.

- [ ] **Step 3: Full unit suite (scoped stages green)**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: the 2 pre-existing evaluate/adapter failures may remain (other session's code); all detect/reid/pose/new-model tests PASS.

- [ ] **Step 4: Commit**

```bash
git add aic24_nvidia/stages/pose.py
git commit -m "feat(pose): RTMPose adapter in main venv; remove mmpose/.venv-pose path"
```

---

## Task 8: rtmlib sanity inference → delete `.venv-pose`

**Files:** none (runtime + cleanup)

- [ ] **Step 1: Sanity inference in main `.venv`**

Run:
```bash
.venv/bin/python - <<'PY'
import numpy as np, cv2
from aic24_nvidia.models import pose_rtmpose
img = np.zeros((1080, 1920, 3), np.uint8)
kp = pose_rtmpose._estimate(img, [[100.0, 100.0, 300.0, 600.0]])
assert len(kp) == 1 and len(kp[0]) == 17, kp
print("rtmpose OK", len(kp[0]), "keypoints")
PY
```
Expected: downloads the RTMPose-l ONNX, prints `rtmpose OK 17 keypoints`.

- [ ] **Step 2: Delete `.venv-pose` (frees ~4.1 G)**

Only after Step 1 prints OK. `.venv-pose` is a symlink in this worktree to the real dir in the main checkout — delete the real target:
```bash
rm -rf "$(readlink -f .venv-pose)" && rm -f .venv-pose
df -h . | tail -1
```
Expected: ~4 G freed.

- [ ] **Step 3: Update docs (CLAUDE.md / README) removing the dual-venv section**

Edit `CLAUDE.md`: remove the "⚠️ Dual-venv setup" section and the pose-venv bullet under Hardware/Running; note pose now runs in `.venv` via rtmlib. (Coordinate with the other session at merge — see Task 10.)

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "chore: remove .venv-pose (RTMPose runs in main venv); reclaim 4.1G"
```

---

## Task 9: End-to-end smoke run (verification gate)

**Files:** none (runtime). Requires GPU + the bootstrapped siblings/data.

- [ ] **Step 1: Run the full pipeline on Warehouse_001_30s with a fresh run id**

Run:
```bash
.venv/bin/python pipeline.py all --config configs/warehouse_001_30s.yaml --run-id upgrade_smoke
```
Expected: all 8 stages report `ok`; no crash in SCT/MCT (proves byte-compatibility).

- [ ] **Step 2: Confirm downstream consumed the new outputs**

Run:
```bash
ls outputs/upgrade_smoke/mct/ && cat outputs/upgrade_smoke/evaluate/metrics*.json 2>/dev/null | head
```
Expected: MCT tracks present; evaluate emits HOTA/IDF1 numbers (values may differ from the YOLOX baseline — that is expected and acceptable for this task).

- [ ] **Step 3: If SCT/MCT errors on format, fix the adapter, re-run**

Likely culprits: frame indexing, npy filename coord order (`x1_x2_y1_y2`), or keypoint ordering. Fix in the relevant `models/*.py`, re-commit, re-run Step 1.

- [ ] **Step 4: Commit any fixes**

```bash
git add -A && git commit -m "fix(models): smoke-run format corrections"
```

---

## Task 10: Delete old weights + merge other session's branch

**Files:** none (cleanup + merge)

- [ ] **Step 1: Delete old weights (only after Task 9 passes)**

```bash
rm -f "$(readlink -f external)/BoT-SORT/bytetrack_x_mot17.pth.tar"
# OSNet/HRNet caches:
rm -f "$(readlink -f data)/deep-person-reid/checkpoints/osnet_ms_m_c.pth.tar"
rm -rf ~/.cache/mim ~/.cache/torch/hub/checkpoints/hrnet_w48* 2>/dev/null
df -h . | tail -1
```
Expected: ~757 M (+ caches) freed.

- [ ] **Step 2: Merge the other session's latest `feat/model-upgrades`**

```bash
git fetch . feat/model-upgrades:feat/model-upgrades 2>/dev/null || true
git merge feat/model-upgrades -m "merge: other session's feat/model-upgrades work"
```
Resolve conflicts (likely in `stages/*.py`, `CLAUDE.md`, evaluate/adapter tests). Re-run `.venv/bin/python -m pytest tests/unit -q` — expect the previously-failing evaluate/adapter tests to now pass if the other session fixed them.

- [ ] **Step 3: Re-run smoke + full unit suite post-merge**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: 0 failures (both sessions' work integrated).

- [ ] **Step 4: Use the finishing-a-development-branch skill** to choose merge/PR/cleanup for `feat/model-upgrades-impl`.

---

## Self-Review notes

- **Spec coverage:** detect/reid/pose adapters (Tasks 2-7), byte-compatible formats (contracts + tests in each), `.venv-pose` removal (Task 8), two-phase deletion (Tasks 8 + 10), verification gates (Tasks 8 Step 1, 9), merge of other session (Task 10) — all covered.
- **Embedding-dim risk:** Task 5 Step 1 explicitly greps YACHIYO before relying on dim-agnosticism.
- **Keypoint-ordering risk:** flagged in Task 6 Step 3, verified in Task 9.
- **Detection distribution shift:** acceptable per spec; `conf_thresh`/`nms_iou` now live via adapter args (Task 3).
