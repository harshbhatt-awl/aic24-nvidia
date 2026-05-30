# tests/unit/test_linking_gate_sweep_registry.py
from pathlib import Path
import yaml

REGISTRY = Path(__file__).resolve().parents[2] / "experiments" / "registry.yaml"


def _experiments():
    data = yaml.safe_load(REGISTRY.read_text())
    return {e["id"]: e for e in data["experiments"]}


def test_linking_gate_sweep_present_and_reruns_from_sct():
    exp = _experiments()["linking_gate_sweep"]
    assert exp["base_config"] == "configs/baseline.yaml"
    # tracking_params changes must rerun from sct per registry convention
    assert exp["rerun_from"] == "sct"


def test_linking_gate_sweep_varies_the_two_eligibility_gates():
    exp = _experiments()["linking_gate_sweep"]
    names = {v["name"] for v in exp["variants"]}
    # baseline control + keypoint and short-track variants
    assert "baseline" in names
    kp_vals, st_vals = set(), set()
    for v in exp["variants"]:
        tp = v["overrides"].get("tracking_params", {})
        if "keypoint_condition_th" in tp:
            kp_vals.add(tp["keypoint_condition_th"])
        if "short_track_th" in tp:
            st_vals.add(tp["short_track_th"])
    assert {2, 3} <= kp_vals          # raises the keypoint gate (E2, 74% of drops)
    assert {60, 30} <= st_vals        # lowers the length gate (E1, 26% of drops)
