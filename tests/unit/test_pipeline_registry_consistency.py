"""Guards that pipeline.py and the experiment harness consume the single stage
registry (aic24_nvidia.registry) rather than reintroducing the parallel stage
dicts they used to maintain by hand.

Registry *invariants* (order, upstream topology, unique dir_names) are tested in
test_stage_registry.py; this file guards the *consumers*.
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
from aic24_nvidia import registry  # noqa: E402
from experiments import _lib  # noqa: E402


def test_pipeline_no_longer_defines_parallel_stage_dicts():
    # The four hand-maintained dicts were replaced by the registry; guard against
    # them being reintroduced (which would let a typo silently mis-gate a stage).
    for attr in ("STAGE_RUNNERS", "ORDER", "UPSTREAM_OF", "STAGE_DIR_NAME"):
        assert not hasattr(pipeline, attr), (
            f"pipeline.{attr} should be gone — consume aic24_nvidia.registry instead"
        )


def test_experiment_harness_stage_views_derive_from_registry():
    assert _lib.STAGES == tuple(registry.order())
    assert _lib.STAGE_DIR == {s: registry.dir_name(s) for s in registry.order()}
