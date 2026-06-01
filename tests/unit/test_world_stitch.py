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
