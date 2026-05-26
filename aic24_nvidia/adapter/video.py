from __future__ import annotations
import json
import subprocess
from pathlib import Path


def probe_duration(path: Path) -> float:
    """Return duration in seconds using ffprobe."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return float(out)


def slice_video(src: Path, dst: Path, start_sec: float, duration_sec: float) -> None:
    """Cut [start_sec, start_sec+duration_sec] from src into dst."""
    src = Path(src)
    dst = Path(dst)
    total = probe_duration(src)
    if start_sec + duration_sec > total + 0.1:
        raise ValueError(
            f"requested window [{start_sec}, {start_sec + duration_sec}] "
            f"exceeds source duration {total:.2f}s for {src}"
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Re-encode for accurate keyframe seek; stream-copy gives unpredictable trim.
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", str(start_sec), "-i", str(src),
            "-t", str(duration_sec),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-an", str(dst),
        ],
        check=True,
    )


def materialize_yachiyo_layout(
    src_dir: Path,
    target_root: Path,
    scene_name: str,
    camera_names: list[str],
    start_sec: float,
    duration_sec: float,
) -> Path:
    """Materialize Original/<scene_name>/camera_NNNN/video.mp4 tree.

    Maps NVIDIA Camera_NNNN -> YACHIYO camera_nnnn (lowercase). Writes a
    scene.json beside Original/ that records the source mapping.

    Returns the path to scene.json.
    """
    src_dir = Path(src_dir)
    target_root = Path(target_root)
    scene_dir = target_root / "Original" / scene_name
    scene_dir.mkdir(parents=True, exist_ok=True)

    mapping: dict[str, dict[str, str]] = {scene_name: {}}
    for i, cam in enumerate(camera_names, start=1):
        src = src_dir / f"{cam}.mp4"
        if not src.exists():
            raise FileNotFoundError(f"source video not found: {src}")
        yachiyo_cam = f"camera_{i:04d}"
        dst = scene_dir / yachiyo_cam / "video.mp4"
        slice_video(src, dst, start_sec=start_sec, duration_sec=duration_sec)
        mapping[scene_name][yachiyo_cam] = cam

    scene_json = target_root / "scene.json"
    scene_json.write_text(json.dumps(mapping, indent=2))
    return scene_json
