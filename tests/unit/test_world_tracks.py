import json
from pathlib import Path
from aic24_nvidia.world_tracks import aggregate_world_tracks, write_world_pred


def _mct_json(tmp_path) -> Path:
    # global id 7 seen by two cameras at frame 1 -> mean of (2,4) and (4,8) = (3,6)
    body = {
        "390": {
            "00000001": {"Frame": 1, "WorldCoordinate": {"x": 2.0, "y": 4.0},
                          "Coordinate": {"x1": 0, "y1": 0, "x2": 1, "y2": 1},
                          "OfflineID": 1, "GlobalOfflineID": 7},
        },
        "391": {
            "00000002": {"Frame": 1, "WorldCoordinate": {"x": 4.0, "y": 8.0},
                          "Coordinate": {"x1": 0, "y1": 0, "x2": 1, "y2": 1},
                          "OfflineID": 3, "GlobalOfflineID": 7},
            "00000003": {"Frame": 2, "WorldCoordinate": {"x": 1.0, "y": 1.0},
                          "Coordinate": {"x1": 0, "y1": 0, "x2": 1, "y2": 1},
                          "OfflineID": 3, "GlobalOfflineID": 7},
            # missing GlobalOfflineID -> dropped
            "00000004": {"Frame": 1, "WorldCoordinate": {"x": 9.0, "y": 9.0},
                          "Coordinate": {"x1": 0, "y1": 0, "x2": 1, "y2": 1},
                          "OfflineID": 5},
            # NaN world coord -> dropped
            "00000005": {"Frame": 3, "WorldCoordinate": {"x": float("nan"), "y": 1.0},
                          "Coordinate": {"x1": 0, "y1": 0, "x2": 1, "y2": 1},
                          "OfflineID": 9, "GlobalOfflineID": 8},
        },
    }
    p = tmp_path / "mct.json"
    p.write_text(json.dumps(body))
    return p


def test_aggregate_means_across_cameras_and_drops_invalid(tmp_path):
    rows, dropped = aggregate_world_tracks(_mct_json(tmp_path))
    assert (1, 7, 3.0, 6.0) in rows
    assert (2, 7, 1.0, 1.0) in rows
    assert all(gid == 7 for (_f, gid, _x, _y) in rows)
    assert dropped == 2  # the None-gid row and the NaN row
    assert len(rows) == 2


def test_write_world_pred_format(tmp_path):
    rows, _ = aggregate_world_tracks(_mct_json(tmp_path))
    out = tmp_path / "pred.txt"
    write_world_pred(rows, out)
    lines = out.read_text().strip().splitlines()
    assert lines[0] == "1,7,3.0,6.0"
    assert len(lines) == 2


def test_aggregate_empty_and_malformed(tmp_path):
    import json
    empty = tmp_path / "empty.json"
    empty.write_text("{}")
    rows, dropped = aggregate_world_tracks(empty)
    assert rows == [] and dropped == 0

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"390": {
        "00000001": {"Frame": 1, "WorldCoordinate": {"x": 1.0, "y": 2.0}, "GlobalOfflineID": "oops"},
        "00000002": {"WorldCoordinate": {"x": 1.0, "y": 2.0}, "GlobalOfflineID": 3},  # missing Frame
        "00000003": {"Frame": 2, "WorldCoordinate": {"x": 0.0, "y": 0.0}, "GlobalOfflineID": 0},  # gid 0 valid
    }}))
    rows, dropped = aggregate_world_tracks(bad)
    assert dropped == 2                       # non-numeric gid + missing Frame
    assert (2, 0, 0.0, 0.0) in rows           # gid 0 is kept
