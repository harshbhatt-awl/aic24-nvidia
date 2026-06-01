"""Unit tests for aic24_nvidia.world_stitch."""
from __future__ import annotations

import pytest


def test_summarize_tracks_endpoints_and_count():
    from aic24_nvidia.world_stitch import summarize_tracks

    rows = [
        (3, 7, 0.0, 0.0),
        (1, 7, 10.0, 20.0),
        (2, 7, 11.0, 21.0),
        (5, 9, 100.0, 100.0),
    ]
    s = summarize_tracks(rows)
    assert set(s) == {7, 9}
    assert s[7].first_frame == 1 and s[7].last_frame == 3
    assert s[7].first_xy == (10.0, 20.0)
    assert s[7].last_xy == (0.0, 0.0)
    assert s[7].n == 3
    assert s[9].first_frame == s[9].last_frame == 5
    assert s[9].n == 1


from aic24_nvidia.world_stitch import find_stitch_edges, summarize_tracks


def _summ(rows):
    return summarize_tracks(rows)


def test_find_edges_accepts_close_sequential_pair():
    # gid 3 ends frame 3 at (0,0); gid 8 starts frame 5 at (0.3,0): gap 2, dist 0.3
    rows = [(1, 3, 0.0, 0.0), (3, 3, 0.0, 0.0), (5, 8, 0.3, 0.0), (7, 8, 0.5, 0.0)]
    edges = find_stitch_edges(_summ(rows), max_gap_frames=45, max_dist_m=0.6)
    assert len(edges) == 1
    dist, gap, a, b = edges[0]
    assert (a, b) == (3, 8)
    assert gap == 2
    assert dist == pytest.approx(0.3)


def test_find_edges_rejects_temporal_overlap():
    # gid 3 spans 1..6, gid 8 spans 5..9 -> overlap, no edge either direction
    rows = [(1, 3, 0.0, 0.0), (6, 3, 0.0, 0.0), (5, 8, 0.1, 0.0), (9, 8, 0.1, 0.0)]
    assert find_stitch_edges(_summ(rows), max_gap_frames=45, max_dist_m=0.6) == []


def test_find_edges_rejects_too_far_in_space():
    rows = [(1, 3, 0.0, 0.0), (3, 3, 0.0, 0.0), (5, 8, 5.0, 0.0), (7, 8, 5.0, 0.0)]
    assert find_stitch_edges(_summ(rows), max_gap_frames=45, max_dist_m=0.6) == []


def test_find_edges_rejects_too_far_in_time():
    rows = [(1, 3, 0.0, 0.0), (3, 3, 0.0, 0.0), (100, 8, 0.1, 0.0), (110, 8, 0.1, 0.0)]
    assert find_stitch_edges(_summ(rows), max_gap_frames=45, max_dist_m=0.6) == []


from aic24_nvidia.world_stitch import resolve_merges


def test_resolve_chain_unions_to_min_gid():
    # edges as (dist, gap, a, b): 3->8 and 8->12 chain into one component, canonical 3
    edges = [(0.1, 2, 3, 8), (0.2, 2, 8, 12)]
    labels = resolve_merges(edges)
    assert labels[3] == 3
    assert labels[8] == 3
    assert labels[12] == 3


def test_resolve_fanin_keeps_closer_edge_only():
    # two predecessors (3, 5) both want successor-start 8; the closer (3) wins,
    # the other start-slot is consumed so 5->8 is skipped.
    edges = [(0.1, 2, 3, 8), (0.4, 2, 5, 8)]
    labels = resolve_merges(edges)
    assert labels[3] == 3 and labels[8] == 3
    assert 5 not in labels  # 5 never merged


def test_resolve_empty():
    assert resolve_merges([]) == {}


def test_resolve_fanout_keeps_first_edge_only():
    # one predecessor-end (3) with two successor-starts (8, 9); the closer (8)
    # wins and consumes 3's end-slot, so 3->9 is skipped.
    edges = [(0.1, 2, 3, 8), (0.4, 2, 3, 9)]
    labels = resolve_merges(edges)
    assert labels[3] == 3 and labels[8] == 3
    assert 9 not in labels  # 9 never merged


def test_resolve_canonical_is_min_regardless_of_union_order():
    # larger gid unioned before the smaller one is introduced: 8->12 then 3->8.
    # canonical must still drag down to the component minimum (3).
    edges = [(0.1, 2, 8, 12), (0.2, 2, 3, 8)]
    labels = resolve_merges(edges)
    assert labels[3] == 3 and labels[8] == 3 and labels[12] == 3
