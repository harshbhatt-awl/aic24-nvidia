"""Regression guard: stages that shell into the upstream YACHIYO repo must run
the subprocess under the SAME interpreter as the orchestrator (sys.executable),
not a bare 'python'/'python3' resolved from PATH.

A literal 'python' resolves to whatever is first on PATH — the *system*
interpreter when the venv isn't activated — which lacks our deps (e.g. sklearn,
used by YACHIYO's DBSCAN) and silently fails the stage. sys.executable is the
running venv interpreter regardless of activation.
"""
from pathlib import Path

import pytest

_STAGES_DIR = Path(__file__).resolve().parents[2] / "aic24_nvidia" / "stages"
_YACHIYO_SUBPROCESS_STAGES = ["sct.py", "mct.py", "extract_frames.py"]


@pytest.mark.parametrize("fname", _YACHIYO_SUBPROCESS_STAGES)
def test_yachiyo_subprocess_uses_sys_executable(fname):
    src = (_STAGES_DIR / fname).read_text()
    assert "sys.executable" in src, f"{fname}: subprocess should use sys.executable"
    assert '["python"' not in src and '["python3"' not in src, (
        f"{fname}: bare python/python3 subprocess literal — uses the PATH "
        f"interpreter (system python without our deps) instead of the venv"
    )
