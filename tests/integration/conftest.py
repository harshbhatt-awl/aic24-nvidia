# tests/integration/conftest.py
import json
import subprocess
from pathlib import Path
import pytest


def _make_video(path: Path, dur: int = 5, fps: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", f"testsrc=duration={dur}:size=320x240:rate={fps}",
         "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )


@pytest.fixture
def tiny_scene(tmp_path: Path) -> Path:
    """Create a synthetic Warehouse_T01 scene with 2 cameras × 5s @ 30fps."""
    scene = tmp_path / "data" / "nvidia_mtmc_2024" / "Warehouse_T01"
    (scene / "videos").mkdir(parents=True)
    for cam in ("Camera_0001", "Camera_0002"):
        _make_video(scene / "videos" / f"{cam}.mp4")

    (scene / "calibration.json").write_text(json.dumps({
        "Camera_0001": {"intrinsicMatrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
        "Camera_0002": {"intrinsicMatrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
    }))

    frames = []
    for f in range(30):  # 1 second of GT
        frames.append({
            "frame_id": f,
            "objects": [{
                "id": 1,
                "world_xyz": [100.0, 200.0, 0.0],
                "cameras": {
                    "Camera_0001": {"bbox_xywh": [80 + f, 180, 40, 40]},
                    "Camera_0002": {"bbox_xywh": [400 + f, 200, 40, 40]},
                },
            }],
        })
    (scene / "ground_truth.json").write_text(json.dumps({"frames": frames}))
    return tmp_path
