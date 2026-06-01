from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

Row = tuple[int, int, float, float]  # (frame, gid, x, y)


@dataclass(frozen=True)
class TrackSummary:
    gid: int
    first_frame: int
    last_frame: int
    first_xy: tuple[float, float]
    last_xy: tuple[float, float]
    n: int


def summarize_tracks(rows: list[Row]) -> dict[int, TrackSummary]:
    """Per-gid endpoints/counts from (frame, gid, x, y) rows."""
    by_gid: dict[int, list[tuple[int, float, float]]] = {}
    for f, g, x, y in rows:
        by_gid.setdefault(g, []).append((f, x, y))
    out: dict[int, TrackSummary] = {}
    for g, seq in by_gid.items():
        seq.sort(key=lambda t: t[0])
        f0, x0, y0 = seq[0]
        f1, x1, y1 = seq[-1]
        out[g] = TrackSummary(
            gid=g, first_frame=f0, last_frame=f1,
            first_xy=(x0, y0), last_xy=(x1, y1), n=len(seq),
        )
    return out
