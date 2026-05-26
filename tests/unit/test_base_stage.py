from pathlib import Path
import json
import pytest
from aic24_nvidia.stages.base import atomic_stage, vram_guard_disabled
from aic24_nvidia.manifest import Manifest


def test_atomic_stage_promotes_tmp_to_final(tmp_path):
    rd = tmp_path / "run1"
    rd.mkdir()
    with atomic_stage(rd, "detect", run_id="run1") as ctx:
        ctx.work_dir.mkdir(parents=True, exist_ok=True)
        (ctx.work_dir / "out.txt").write_text("hello")
        ctx.set_outputs({"out": str(ctx.work_dir / "out.txt")})
        ctx.set_params({"x": 1})

    assert (rd / "detect" / "out.txt").read_text() == "hello"
    manifest = json.loads((rd / "detect" / "manifest.json").read_text())
    assert manifest["status"] == "ok"
    assert manifest["params"] == {"x": 1}
    assert not (rd / "detect.tmp").exists()


def test_atomic_stage_does_not_promote_on_exception(tmp_path):
    rd = tmp_path / "run1"
    rd.mkdir()
    with pytest.raises(RuntimeError, match="boom"):
        with atomic_stage(rd, "detect", run_id="run1") as ctx:
            ctx.work_dir.mkdir(parents=True, exist_ok=True)
            (ctx.work_dir / "out.txt").write_text("hello")
            raise RuntimeError("boom")

    assert (rd / "detect.tmp").exists()
    assert not (rd / "detect").exists()


def test_vram_guard_disabled_is_a_noop():
    # On systems without CUDA the guard must not crash.
    with vram_guard_disabled():
        pass
