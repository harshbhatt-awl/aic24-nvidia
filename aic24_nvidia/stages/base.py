from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import shutil
import time

from ..bootstrap import ensure_dir_clean, make_symlink
from ..manifest import Manifest, write_manifest
from ..paths import stage_dir, stage_tmp_dir


def _apply_links(links) -> None:
    """Materialize a list of (link_path, target_path) symlinks.

    ensure_dir_clean + make_symlink per pair — the exact sequence the stages used
    to run inline. base.py owns this so a stage declares its wiring once (see
    aic24_nvidia.registry) instead of symlinking by hand. NOTE: base.py must NOT
    import registry (registry imports the stage modules, which import base) — the
    wiring callable is passed into atomic_stage instead, so there is no cycle.
    """
    for link, target in links:
        ensure_dir_clean(link)
        make_symlink(target, link)


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
def atomic_stage(run_dir: Path, stage: str, run_id: str, *, cfg=None, wiring=None):
    """Run a stage with atomic on-disk semantics.

    - writes outputs to run_dir/<stage>.tmp/
    - on clean exit writes manifest.json (status=ok) then renames .tmp/ -> final/
    - on exception leaves .tmp/ in place for inspection; no final dir created

    When ``cfg`` and ``wiring`` are supplied, the stage's external symlinks are
    applied centrally here: pre-run with output_dir=<stage>.tmp (so upstream
    tooling reads/writes the in-progress dir) and post-promotion with
    output_dir=<stage> (re-pointed to the final dir). ``wiring`` is a callable
    ``(run_dir, cfg, output_dir) -> list[(link, target)]`` — see registry.StageSpec.
    """
    tmp = stage_tmp_dir(run_dir, stage)
    final = stage_dir(run_dir, stage)
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    if cfg is not None and wiring is not None:
        _apply_links(wiring(run_dir, cfg, tmp))

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

    # Rewrite the manifest so all references to the tmp path become the final path.
    # The manifest was written before rename, so its paths still contain the .tmp
    # directory name. Replace every occurrence of the absolute tmp path string with
    # the absolute final path string so downstream consumers never see stale .tmp refs.
    manifest_path = final / "manifest.json"
    raw = manifest_path.read_text()
    raw = raw.replace(str(tmp.resolve()), str(final.resolve()))
    # Also cover the non-resolved form in case they differ (e.g. symlinks).
    raw = raw.replace(str(tmp), str(final))
    manifest_path.write_text(raw)

    # Re-point the stage's external symlinks at the promoted final dir.
    if cfg is not None and wiring is not None:
        _apply_links(wiring(run_dir, cfg, final))


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
