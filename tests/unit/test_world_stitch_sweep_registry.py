# tests/unit/test_world_stitch_sweep_registry.py
from pathlib import Path
import yaml

REGISTRY = Path(__file__).resolve().parents[2] / "experiments" / "registry.yaml"


def _experiments():
    data = yaml.safe_load(REGISTRY.read_text())
    return {e["id"]: e for e in data["experiments"]}


def test_world_stitch_sweep_present_and_reruns_from_evaluate():
    exp = _experiments()["world_stitch_sweep"]
    assert exp["base_config"] == "configs/baseline.yaml"
    # world_stitch.* is an eval-only transform -> rerun_from evaluate
    assert exp["rerun_from"] == "evaluate"


def test_world_stitch_sweep_has_none_control_and_endpoint_gap_variants():
    exp = _experiments()["world_stitch_sweep"]
    names = {v["name"] for v in exp["variants"]}
    assert "none" in names
    methods, gaps, dists = set(), set(), set()
    for v in exp["variants"]:
        st = v["overrides"].get("world_stitch", {})
        if "method" in st:
            methods.add(st["method"])
        if "max_gap_frames" in st:
            gaps.add(st["max_gap_frames"])
        if "max_dist_m" in st:
            dists.add(st["max_dist_m"])
    assert "endpoint_gap" in methods
    assert {30, 45, 60, 90} <= gaps
    assert {0.5, 0.6, 0.75, 1.0} <= dists
