"""Regression guard for the (currently hardcoded) stage registry.

The pipeline's stage graph lives in four parallel dicts in `pipeline.py`
(`STAGE_RUNNERS`, `ORDER`, `UPSTREAM_OF`, `STAGE_DIR_NAME`) and is *duplicated*
in `experiments/_lib.py` (`STAGES`, `STAGE_DIR`). Until a real StageRegistry
replaces them (Phase 1), these tests lock the invariant that all five stay in
sync — so a typo like `'sct'` in one dict and `'sct_'` in another fails loudly
instead of silently mis-gating a stage.
"""
from __future__ import annotations

import sys
from pathlib import Path

# pipeline.py and the experiments package live at the repo root, which is not
# necessarily on sys.path under pytest.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pipeline  # noqa: E402
from experiments import _lib  # noqa: E402


def test_pipeline_dicts_cover_the_same_stages():
    order = set(pipeline.ORDER)
    assert set(pipeline.STAGE_RUNNERS) == order
    assert set(pipeline.STAGE_DIR_NAME) == order
    assert set(pipeline.UPSTREAM_OF) == order


def test_upstream_refs_are_known_stages_and_precede_their_dependents():
    index = {stage: i for i, stage in enumerate(pipeline.ORDER)}
    for stage, upstreams in pipeline.UPSTREAM_OF.items():
        for up in upstreams:
            assert up in index, f"{stage} depends on unknown stage {up!r}"
            assert index[up] < index[stage], (
                f"{stage} depends on {up!r} which comes later in ORDER"
            )


def test_stage_runners_are_callable():
    for stage, runner in pipeline.STAGE_RUNNERS.items():
        assert callable(runner), f"runner for {stage!r} is not callable"


def test_experiment_harness_stage_list_matches_pipeline():
    # The experiment harness keeps its own copy; it must not drift from the
    # orchestrator's source of truth.
    assert tuple(pipeline.ORDER) == _lib.STAGES
    assert pipeline.STAGE_DIR_NAME == _lib.STAGE_DIR
