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


def find_stitch_edges(
    summaries: dict[int, TrackSummary], *, max_gap_frames: int, max_dist_m: float
) -> list[tuple[float, int, int, int]]:
    """Candidate (dist, gap, gid_a, gid_b) sequential edges passing the tight gate.

    An edge A->B requires strict non-overlap (A.last_frame < B.first_frame), a
    frame gap <= max_gap_frames, and a world endpoint distance <= max_dist_m.
    Sorted by (dist, gap, gid_a, gid_b) ascending.
    """
    edges: list[tuple[float, int, int, int]] = []
    items = list(summaries.values())
    for a in items:
        for b in items:
            if a.gid == b.gid:
                continue
            if not (a.last_frame < b.first_frame):  # strict non-overlap, A before B
                continue
            gap = b.first_frame - a.last_frame
            if gap > max_gap_frames:
                continue
            dist = math.dist(a.last_xy, b.first_xy)
            if dist > max_dist_m:
                continue
            edges.append((dist, gap, a.gid, b.gid))
    edges.sort()
    return edges


def resolve_merges(edges: list[tuple[float, int, int, int]]) -> dict[int, int]:
    """Greedy 1-in/1-out matching + union-find -> {gid: canonical_gid}.

    Edges must be pre-sorted (best first). Each gid is consumed at most once as a
    predecessor end and once as a successor start, so a track continues into at
    most one other and is continued by at most one — preventing fan-in/fan-out
    while still forming chains A->B->C. Canonical id is the minimum gid in a
    component. Returns only gids touched by an accepted merge.
    """
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        hi, lo = (ra, rb) if ra > rb else (rb, ra)
        parent[hi] = lo  # attach larger root under smaller -> canonical = min

    used_end: set[int] = set()
    used_start: set[int] = set()
    for _dist, _gap, a, b in edges:
        if a in used_end or b in used_start:
            continue
        used_end.add(a)
        used_start.add(b)
        union(a, b)

    return {g: find(g) for g in parent}


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
