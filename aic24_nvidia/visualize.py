from __future__ import annotations
import json
import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


def _color_for(id_: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(seed=int(id_) * 9973 + 17)
    c = rng.integers(64, 256, size=3, dtype=np.int32).tolist()
    return (int(c[0]), int(c[1]), int(c[2]))


def _open_writer(out_path: Path, w: int, h: int, fps: int) -> cv2.VideoWriter:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))


def _frame_paths(frame_dir: Path) -> list[Path]:
    return sorted(frame_dir.glob("*.jpg"))


def _reencode_h264(path: Path) -> None:
    """Re-encode an existing mp4 (written by cv2 mp4v) to H.264 in place so
    browsers can play it in <video> tags. No-op if ffmpeg is missing or the
    file is already H.264."""
    import shutil, subprocess
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        return
    path = Path(path)
    if not path.exists():
        return
    try:
        codec = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        return
    if codec in ("h264", "avc1"):
        return
    tmp = path.with_suffix(".h264.mp4")
    proc = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         "-an", str(tmp)],
    )
    if proc.returncode == 0 and tmp.exists():
        tmp.replace(path)


def viz_detect_from_txt(frame_dir: Path, det_txt: Path, out_mp4: Path, fps: int) -> None:
    """Upstream detect emits MOT-like rows: camera_name,frame_id,class,x1,y1,x2,y2,score"""
    dets_by_frame: dict[int, list] = {}
    for line in det_txt.read_text().splitlines():
        parts = line.split(",")
        if len(parts) < 8:
            continue
        fid = int(parts[1])
        x1, y1, x2, y2 = (float(p) for p in parts[3:7])
        conf = float(parts[7])
        dets_by_frame.setdefault(fid, []).append((x1, y1, x2, y2, conf))

    frames = _frame_paths(frame_dir)
    if not frames:
        raise FileNotFoundError(f"no frames in {frame_dir}")
    first = cv2.imread(str(frames[0]))
    h_img, w_img = first.shape[:2]
    writer = _open_writer(out_mp4, w_img, h_img, fps)
    for i, fp in enumerate(frames, start=1):
        img = cv2.imread(str(fp))
        for (x1, y1, x2, y2, conf) in dets_by_frame.get(i, []):
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, f"{conf:.2f}", (int(x1), int(y1) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        writer.write(img)
    writer.release()
    _reencode_h264(out_mp4)


def _coord_to_xyxy_local(coord):
    if isinstance(coord, dict):
        return float(coord["x1"]), float(coord["y1"]), float(coord["x2"]), float(coord["y2"])
    return float(coord[0]), float(coord[1]), float(coord[2]), float(coord[3])


def viz_tracks_from_yachiyo_sct(frame_dir: Path, sct_json: Path, out_mp4: Path, fps: int,
                                color_seed: int = 0) -> None:
    """Render YACHIYO SCT JSON (serial->{Frame,OfflineID,Coordinate}) as per-camera viz."""
    body = json.loads(sct_json.read_text())
    tracks_by_frame: dict[int, list] = {}
    for _serial, e in body.items():
        if not isinstance(e, dict):
            continue
        fid = int(e["Frame"]); tid = int(e["OfflineID"])
        if tid < 0:
            continue
        x1, y1, x2, y2 = _coord_to_xyxy_local(e["Coordinate"])
        tracks_by_frame.setdefault(fid, []).append((tid, x1, y1, x2, y2))

    frames = _frame_paths(frame_dir)
    if not frames:
        raise FileNotFoundError(f"no frames in {frame_dir}")
    first = cv2.imread(str(frames[0]))
    h_img, w_img = first.shape[:2]
    writer = _open_writer(out_mp4, w_img, h_img, fps)
    for i, fp in enumerate(frames, start=1):
        img = cv2.imread(str(fp))
        for (tid, x1, y1, x2, y2) in tracks_by_frame.get(i, []):
            color = _color_for(tid + color_seed)
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(img, f"id={tid}", (int(x1), int(y1) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        writer.write(img)
    writer.release()
    _reencode_h264(out_mp4)


def viz_mct_grid_from_yachiyo_mct(cam_frame_dirs: dict[str, Path], mct_json: Path,
                                  out_mp4: Path, fps: int) -> None:
    """MCT JSON: cam_id_str -> serial -> {Frame, GlobalOfflineID, Coordinate}."""
    body = json.loads(mct_json.read_text())
    # Map cam id (int string) -> "camera_NNNN"
    tracks_by_cam_frame: dict[str, dict[int, list]] = {}
    for cam_key, entries in body.items():
        cam_name = f"camera_{int(cam_key):04d}"
        d: dict[int, list] = {}
        for _serial, e in (entries.items() if isinstance(entries, dict) else []):
            if not isinstance(e, dict):
                continue
            gid_raw = e.get("GlobalOfflineID")
            if gid_raw is None:
                continue
            gid = int(gid_raw)
            if gid < 0:
                continue
            fid = int(e["Frame"])
            x1, y1, x2, y2 = _coord_to_xyxy_local(e["Coordinate"])
            d.setdefault(fid, []).append((gid, x1, y1, x2, y2))
        tracks_by_cam_frame[cam_name] = d

    cams = sorted(cam_frame_dirs)
    cols = int(np.ceil(np.sqrt(len(cams))))
    rows = int(np.ceil(len(cams) / cols))
    sample = cv2.imread(str(_frame_paths(cam_frame_dirs[cams[0]])[0]))
    h_s, w_s = sample.shape[:2]
    tile_w, tile_h = w_s // 2, h_s // 2
    grid_w = tile_w * cols
    grid_h = tile_h * rows

    n_frames = min(len(_frame_paths(d)) for d in cam_frame_dirs.values())
    writer = _open_writer(out_mp4, grid_w, grid_h, fps)
    for i in range(1, n_frames + 1):
        tiles: list[np.ndarray] = []
        for cam in cams:
            fp = cam_frame_dirs[cam] / f"{i:06d}.jpg"
            if not fp.exists():
                tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
            else:
                img = cv2.imread(str(fp))
                for gid, x1, y1, x2, y2 in tracks_by_cam_frame.get(cam, {}).get(i, []):
                    color = _color_for(gid)
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(img, f"g{gid}", (int(x1), int(y1) - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                tile = cv2.resize(img, (tile_w, tile_h))
            tiles.append(tile)
        while len(tiles) < rows * cols:
            tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))
        grid_rows = [np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)]
        writer.write(np.vstack(grid_rows))
    writer.release()
    _reencode_h264(out_mp4)
