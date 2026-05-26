from datetime import datetime
from pathlib import Path
from aic24_nvidia.paths import make_run_id, run_dir, stage_dir, latest_run_id


def test_make_run_id_format():
    rid = make_run_id("warehouse_001_30s", at=datetime(2026, 5, 26, 14, 32, 11))
    assert rid == "warehouse_001_30s_20260526_1432"


def test_run_dir_and_stage_dir(tmp_path):
    rd = run_dir(tmp_path, "warehouse_001_30s_20260526_1432")
    assert rd == tmp_path / "warehouse_001_30s_20260526_1432"
    sd = stage_dir(rd, "detect")
    assert sd == rd / "detect"


def test_latest_run_id_returns_most_recent(tmp_path):
    for name in [
        "warehouse_001_30s_20260101_0900",
        "warehouse_001_30s_20260526_1432",
        "warehouse_002_30s_20260301_1000",
    ]:
        (tmp_path / name).mkdir()
    assert latest_run_id(tmp_path, "warehouse_001_30s") == "warehouse_001_30s_20260526_1432"


def test_latest_run_id_returns_none_when_empty(tmp_path):
    assert latest_run_id(tmp_path, "warehouse_001_30s") is None
