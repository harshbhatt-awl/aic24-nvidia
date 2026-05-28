"""Unit tests for the experiments/ harness.

These tests do NOT touch the real pipeline — they exercise:
  * deep_merge: overrides patch into a base config correctly
  * registry loading: validation rules
  * setup_cache_symlinks: makes the right symlinks; safe on missing baselines
  * stages_to_reuse / stages_to_rerun: ordering is correct
  * variant_run_id: stable and filesystem-safe

Run with: pytest tests/unit/test_experiment_harness.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments._lib import (
    STAGES,
    STAGE_DIR,
    deep_merge,
    load_registry,
    setup_cache_symlinks,
    stages_to_rerun,
    stages_to_reuse,
    variant_run_id,
)


# --------------------------------------------------------------------------- #
# deep_merge
# --------------------------------------------------------------------------- #

def test_deep_merge_overrides_replace_scalars():
    base = {"a": 1, "b": 2}
    out = deep_merge(base, {"b": 99})
    assert out == {"a": 1, "b": 99}
    assert base == {"a": 1, "b": 2}, "input must not mutate"


def test_deep_merge_recurses_into_dicts():
    base = {"sct": {"track_buffer": 30, "match_thresh": 0.8}}
    out = deep_merge(base, {"sct": {"match_thresh": 0.5}})
    assert out == {"sct": {"track_buffer": 30, "match_thresh": 0.5}}


def test_deep_merge_lists_replace_not_extend():
    base = {"x": [1, 2, 3]}
    out = deep_merge(base, {"x": [9]})
    assert out == {"x": [9]}


def test_deep_merge_tracking_params_real_case():
    """Real overrides shape from registry.yaml."""
    base = {
        "tracking_params": {
            "epsilon_scpt": 0.10,
            "epsilon_mcpt": 0.50,
            "sim_th": 0.85,
        },
        "mct": {"hard_world_gate": True},
    }
    out = deep_merge(base, {"tracking_params": {"epsilon_mcpt": 0.30}})
    assert out["tracking_params"]["epsilon_mcpt"] == 0.30
    assert out["tracking_params"]["epsilon_scpt"] == 0.10  # preserved
    assert out["tracking_params"]["sim_th"] == 0.85         # preserved
    assert out["mct"]["hard_world_gate"] is True            # preserved


# --------------------------------------------------------------------------- #
# stages_to_reuse / stages_to_rerun
# --------------------------------------------------------------------------- #

def test_stages_partition_sct():
    reuse = stages_to_reuse("sct")
    rerun = stages_to_rerun("sct")
    assert reuse == ["adapt", "frames", "detect", "reid", "pose"]
    assert rerun == ["sct", "mct", "evaluate"]
    assert reuse + rerun == list(STAGES)


def test_stages_partition_evaluate():
    """eval-only sweeps reuse everything except the final stage."""
    reuse = stages_to_reuse("evaluate")
    rerun = stages_to_rerun("evaluate")
    assert reuse == ["adapt", "frames", "detect", "reid", "pose", "sct", "mct"]
    assert rerun == ["evaluate"]


def test_stages_partition_adapt():
    """Full rerun — nothing to reuse."""
    reuse = stages_to_reuse("adapt")
    rerun = stages_to_rerun("adapt")
    assert reuse == []
    assert rerun == list(STAGES)


# --------------------------------------------------------------------------- #
# variant_run_id
# --------------------------------------------------------------------------- #

def test_variant_run_id_simple():
    assert variant_run_id("eps_mcpt_sweep", "0.30") == "eps_mcpt_sweep__0.30"


def test_variant_run_id_sanitizes_path_chars():
    """Slashes and spaces would break paths — must be replaced."""
    rid = variant_run_id("e", "a/b c")
    assert "/" not in rid
    assert " " not in rid
    assert rid.startswith("e__")


# --------------------------------------------------------------------------- #
# setup_cache_symlinks
# --------------------------------------------------------------------------- #

def _fake_baseline(root: Path, *stages: str) -> Path:
    """Create a fake baseline run with the given stages each containing a
    manifest.json (status=ok)."""
    base = root / "baseline"
    base.mkdir(parents=True)
    for stage in stages:
        sdir = base / STAGE_DIR[stage]
        sdir.mkdir()
        (sdir / "manifest.json").write_text(json.dumps({
            "stage": stage, "run_id": "baseline", "status": "ok",
            "started_at": "2026-01-01T00:00:00Z",
            "finished_at": "2026-01-01T00:00:01Z",
            "runtime_sec": 1.0,
            "inputs": {}, "outputs": {}, "params": {}, "upstream_manifests": [],
        }))
    return base


def test_setup_cache_symlinks_links_existing_stages(tmp_path: Path):
    baseline = _fake_baseline(tmp_path, "adapt", "frames", "detect")
    target = tmp_path / "exp__v1"
    linked = setup_cache_symlinks(
        baseline_run_dir=baseline,
        target_run_dir=target,
        stages=["adapt", "frames", "detect", "reid"],  # reid not in baseline
    )
    assert sorted(linked) == ["adapt", "detect", "frames"]
    # The links resolve to real baseline dirs.
    assert (target / "adapted").is_symlink()
    assert (target / "adapted" / "manifest.json").exists()
    assert (target / "detect").is_symlink()
    # reid wasn't in baseline — no link created.
    assert not (target / "reid").exists()


def test_setup_cache_symlinks_idempotent(tmp_path: Path):
    """Running twice should not error and should not double-link."""
    baseline = _fake_baseline(tmp_path, "adapt", "frames")
    target = tmp_path / "exp__v1"
    setup_cache_symlinks(baseline_run_dir=baseline, target_run_dir=target,
                         stages=["adapt", "frames"])
    linked2 = setup_cache_symlinks(baseline_run_dir=baseline, target_run_dir=target,
                                   stages=["adapt", "frames"])
    # Second call still reports them as linked (the link exists).
    assert sorted(linked2) == ["adapt", "frames"]


def test_setup_cache_symlinks_does_not_clobber_real_dir(tmp_path: Path):
    """If the target stage dir is already a real (non-symlink) directory, don't replace it."""
    baseline = _fake_baseline(tmp_path, "adapt")
    target = tmp_path / "exp__v1"
    target.mkdir()
    real_dir = target / "adapted"
    real_dir.mkdir()
    (real_dir / "sentinel.txt").write_text("preserve me")

    linked = setup_cache_symlinks(baseline_run_dir=baseline, target_run_dir=target,
                                  stages=["adapt"])
    # Not linked (real dir present), not in returned list.
    assert linked == []
    # Sentinel still there.
    assert (real_dir / "sentinel.txt").read_text() == "preserve me"
    assert not (target / "adapted").is_symlink()


# --------------------------------------------------------------------------- #
# load_registry
# --------------------------------------------------------------------------- #

def test_load_registry_real_file_is_valid():
    """The shipped registry.yaml must parse without errors."""
    p = Path(__file__).resolve().parent.parent.parent / "experiments" / "registry.yaml"
    exps = load_registry(p)
    assert len(exps) >= 1
    for e in exps:
        assert e["id"]
        assert e["rerun_from"] in STAGES
        assert e["variants"]


def test_load_registry_rejects_duplicate_ids(tmp_path: Path):
    p = tmp_path / "reg.yaml"
    p.write_text("""\
experiments:
  - id: dup
    base_config: configs/baseline.yaml
    rerun_from: sct
    variants: [{name: a, overrides: {}}]
  - id: dup
    base_config: configs/baseline.yaml
    rerun_from: sct
    variants: [{name: b, overrides: {}}]
""")
    with pytest.raises(ValueError, match="duplicate experiment id"):
        load_registry(p)


def test_load_registry_rejects_bad_rerun_from(tmp_path: Path):
    p = tmp_path / "reg.yaml"
    p.write_text("""\
experiments:
  - id: bad
    base_config: configs/baseline.yaml
    rerun_from: not_a_real_stage
    variants: [{name: a, overrides: {}}]
""")
    with pytest.raises(ValueError, match="rerun_from"):
        load_registry(p)
