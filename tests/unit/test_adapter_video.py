import subprocess
from pathlib import Path
import json
import pytest
from aic24_nvidia.adapter.video import slice_video, materialize_yachiyo_layout, probe_duration


def _make_synthetic_video(path: Path, duration_sec: int = 10, fps: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"testsrc=duration={duration_sec}:size=320x240:rate={fps}",
        "-pix_fmt", "yuv420p", str(path),
    ]
    subprocess.run(cmd, check=True)


@pytest.fixture
def src_video(tmp_path: Path) -> Path:
    p = tmp_path / "src" / "Camera_0001.mp4"
    _make_synthetic_video(p, duration_sec=10)
    return p


def test_probe_duration(src_video: Path):
    d = probe_duration(src_video)
    assert 9.0 < d < 11.0


def test_slice_video_trims_correctly(src_video: Path, tmp_path: Path):
    out = tmp_path / "out.mp4"
    slice_video(src_video, out, start_sec=2, duration_sec=5)
    assert out.exists()
    d = probe_duration(out)
    assert 4.5 < d < 5.5


def test_slice_video_rejects_overlong_window(src_video: Path, tmp_path: Path):
    out = tmp_path / "out.mp4"
    with pytest.raises(ValueError, match="duration"):
        slice_video(src_video, out, start_sec=8, duration_sec=10)


def test_materialize_yachiyo_layout(tmp_path: Path):
    src_dir = tmp_path / "src"
    for cam in ("Camera_0001", "Camera_0002"):
        _make_synthetic_video(src_dir / f"{cam}.mp4", duration_sec=10)

    target = tmp_path / "adapted"
    scene_json = materialize_yachiyo_layout(
        src_dir=src_dir,
        target_root=target,
        scene_name="scene_001",
        camera_names=["Camera_0001", "Camera_0002"],
        start_sec=0,
        duration_sec=5,
    )

    assert (target / "Original" / "scene_001" / "camera_0001" / "video.mp4").exists()
    assert (target / "Original" / "scene_001" / "camera_0002" / "video.mp4").exists()
    body = json.loads(scene_json.read_text())
    assert body["scene_001"]["camera_0001"] == "Camera_0001"
