"""Unit tests for aic24_nvidia.world_tracks.smooth_world_tracks."""
from __future__ import annotations

import pytest


def test_smooth_none_is_identity():
    from aic24_nvidia.world_tracks import smooth_world_tracks

    rows = [
        (1, 100, 10.0, 20.0),
        (2, 100, 12.0, 21.0),
        (3, 100, 14.0, 22.0),
    ]
    out = smooth_world_tracks(rows, method="none", ema_alpha=0.3)
    assert out == rows


def test_smooth_ema_step_function():
    """EMA against a step input; first sample initializes; alpha controls weight."""
    from aic24_nvidia.world_tracks import smooth_world_tracks

    rows = [
        (1, 100, 0.0, 0.0),
        (2, 100, 10.0, 0.0),
        (3, 100, 10.0, 0.0),
    ]
    out = smooth_world_tracks(rows, method="ema", ema_alpha=0.5)
    # s_1 = obs_1 = 0.0
    # s_2 = 0.5*10 + 0.5*0 = 5.0
    # s_3 = 0.5*10 + 0.5*5 = 7.5
    assert out[0] == (1, 100, 0.0, 0.0)
    assert out[1] == (2, 100, 5.0, 0.0)
    assert out[2] == (3, 100, 7.5, 0.0)


def test_smooth_ema_isolates_per_gid():
    """Two gids are smoothed independently."""
    from aic24_nvidia.world_tracks import smooth_world_tracks

    rows = [
        (1, 100, 0.0, 0.0),
        (1, 200, 100.0, 100.0),
        (2, 100, 10.0, 0.0),
        (2, 200, 110.0, 100.0),
    ]
    out = smooth_world_tracks(rows, method="ema", ema_alpha=0.5)
    # Per-gid timeseries (sorted by frame):
    #   gid 100: (0,0) -> (5,0)
    #   gid 200: (100,100) -> (105,100)
    by_key = {(f, g): (x, y) for (f, g, x, y) in out}
    assert by_key[(1, 100)] == (0.0, 0.0)
    assert by_key[(2, 100)] == (5.0, 0.0)
    assert by_key[(1, 200)] == (100.0, 100.0)
    assert by_key[(2, 200)] == (105.0, 100.0)


def test_smooth_ema_single_observation_per_gid():
    """A gid with one point passes through unchanged."""
    from aic24_nvidia.world_tracks import smooth_world_tracks

    rows = [(7, 42, 1.5, 2.5)]
    out = smooth_world_tracks(rows, method="ema", ema_alpha=0.3)
    assert out == rows


def test_smooth_output_is_sorted_by_frame_then_gid():
    """Output order matches the existing aggregate_world_tracks contract."""
    from aic24_nvidia.world_tracks import smooth_world_tracks

    rows = [
        (3, 100, 0.0, 0.0),
        (1, 100, 0.0, 0.0),
        (2, 200, 0.0, 0.0),
        (2, 100, 0.0, 0.0),
    ]
    out = smooth_world_tracks(rows, method="ema", ema_alpha=0.5)
    keys = [(f, g) for (f, g, _x, _y) in out]
    assert keys == sorted(keys)


def test_smooth_invalid_method():
    from aic24_nvidia.world_tracks import smooth_world_tracks
    with pytest.raises(ValueError, match="method"):
        smooth_world_tracks([(1, 1, 0.0, 0.0)], method="kalman", ema_alpha=0.5)
