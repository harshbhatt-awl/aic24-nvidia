from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import shutil
import time

from ..manifest import Manifest, write_manifest
from ..paths import stage_dir, stage_tmp_dir


@dataclass
class StageCtx:
    work_dir: Path
    stage: str
    run_id: str
    started_at: str
    _t0: float
    _inputs: dict = field(default_factory=dict)
    _outputs: dict = field(default_factory=dict)
    _params: dict = field(default_factory=dict)
    _upstream: list[str] = field(default_factory=list)

    def set_inputs(self, inputs: dict) -> None:
        self._inputs = dict(inputs)

    def set_outputs(self, outputs: dict) -> None:
        self._outputs = dict(outputs)

    def set_params(self, params: dict) -> None:
        self._params = dict(params)

    def set_upstream(self, upstream: list[str]) -> None:
        self._upstream = list(upstream)


@contextmanager
def atomic_stage(run_dir: Path, stage: str, run_id: str):
    """Run a stage with atomic on-disk semantics.

    - writes outputs to run_dir/<stage>.tmp/
    - on clean exit writes manifest.json (status=ok) then renames .tmp/ -> final/
    - on exception leaves .tmp/ in place for inspection; no final dir created
    """
    tmp = stage_tmp_dir(run_dir, stage)
    final = stage_dir(run_dir, stage)
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.time()
    ctx = StageCtx(work_dir=tmp, stage=stage, run_id=run_id, started_at=started_at, _t0=t0)

    yield ctx

    finished_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    runtime = time.time() - t0

    manifest = Manifest(
        stage=stage,
        run_id=run_id,
        started_at=started_at,
        finished_at=finished_at,
        runtime_sec=runtime,
        inputs=ctx._inputs,
        outputs=ctx._outputs,
        params=ctx._params,
        upstream_manifests=ctx._upstream,
        status="ok",
    )
    write_manifest(manifest, tmp / "manifest.json")

    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)


@contextmanager
def vram_guard_disabled():
    """Stub for tests that don't need a real GPU check."""
    yield


def assert_vram_free(min_free_gb: float) -> None:
    """Refuse to start if free CUDA memory is below threshold.

    On systems without CUDA this is a no-op (we only guard against OOM on the GPU
    we have; absence of CUDA means CPU-only and the upstream will tell us if it
    can't run there).
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return
        free_bytes, _total = torch.cuda.mem_get_info()
        free_gb = free_bytes / (1024 ** 3)
        if free_gb < min_free_gb:
            raise RuntimeError(
                f"insufficient VRAM: {free_gb:.2f} GB free, need {min_free_gb} GB"
            )
    except ImportError:
        return
