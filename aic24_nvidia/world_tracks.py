from __future__ import annotations
import json
import math
from collections import defaultdict
from pathlib import Path


def aggregate_world_tracks(mct_json: Path) -> tuple[list[tuple[int, int, float, float]], int]:
    """Collapse the MCT JSON into one world point per (frame, global_id).

    Multiple cameras seeing the same global id in the same frame are averaged.
    Detections with no GlobalOfflineID, negative id, or non-finite world coords
    are dropped. Returns (sorted rows, dropped_count).
    """
    body = json.loads(Path(mct_json).read_text())
    acc: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)
    dropped = 0
    for _cam_key, entries in body.items():
        if not isinstance(entries, dict):
            continue
        for _serial, e in entries.items():
            if not isinstance(e, dict):
                continue
            gid = e.get("GlobalOfflineID")
            wc = e.get("WorldCoordinate")
            if gid is None or int(gid) < 0 or not isinstance(wc, dict):
                dropped += 1
                continue
            x, y = float(wc.get("x", float("nan"))), float(wc.get("y", float("nan")))
            if not (math.isfinite(x) and math.isfinite(y)):
                dropped += 1
                continue
            acc[(int(e["Frame"]), int(gid))].append((x, y))
    rows: list[tuple[int, int, float, float]] = []
    for (frame, gid), pts in acc.items():
        mx = sum(p[0] for p in pts) / len(pts)
        my = sum(p[1] for p in pts) / len(pts)
        rows.append((frame, gid, mx, my))
    rows.sort()
    return rows, dropped


def write_world_pred(rows: list[tuple[int, int, float, float]], dst: Path) -> None:
    """Write `frame,gid,x,y` rows (matches scene_001_gt_world.txt schema)."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        for frame, gid, x, y in rows:
            f.write(f"{frame},{gid},{x},{y}\n")
