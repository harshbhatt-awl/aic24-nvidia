"""Single source of truth for the pipeline's stages.

Replaces the four hand-maintained dicts that used to live in ``pipeline.py``
(``STAGE_RUNNERS``/``ORDER``/``UPSTREAM_OF``/``STAGE_DIR_NAME``) and the duplicate
``STAGES``/``STAGE_DIR`` in ``experiments/_lib.py``. Both now derive from
``REGISTRY`` here.

Each ``StageSpec`` also carries a ``wiring`` callable describing the external
symlinks the stage owns (output exposure + input cross-wiring). The ``wiring``
field defaults to a no-op for now; per-stage wiring functions are populated in a
later migration step. ``base.atomic_stage`` consumes ``wiring`` directly (it does
NOT import this module — the callable is passed in — so there is no import cycle).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import Config
from .stages import (
    adapt,
    detect,
    evaluate,
    extract_frames,
    mct,
    pose,
    reid,
    sct,
)

# (symlink_path, target_path)
Link = tuple[Path, Path]


def _no_wiring(run_dir: Path, cfg: Config, output_dir: Path) -> list[Link]:
    return []


@dataclass(frozen=True)
class StageSpec:
    name: str                                   # logical stage name, e.g. "detect"
    dir_name: str                               # output subdir under run_dir ("adapted" for adapt)
    upstream: tuple[str, ...]                   # gating dependencies, by stage name
    run: Callable[[Config, Path, str], None]    # the existing run(cfg, run_dir, run_id)
    wiring: Callable[[Path, Config, Path], list[Link]] = _no_wiring


# Ordered: list position == pipeline execution order.
REGISTRY: tuple[StageSpec, ...] = (
    StageSpec("adapt",    "adapted",  (),           adapt.run),
    StageSpec("frames",   "frames",   ("adapt",),   extract_frames.run),
    StageSpec("detect",   "detect",   ("frames",),  detect.run),
    StageSpec("reid",     "reid",     ("detect",),  reid.run),
    StageSpec("pose",     "pose",     ("reid",),    pose.run),
    StageSpec("sct",      "sct",      ("pose",),    sct.run),
    StageSpec("mct",      "mct",      ("sct",),     mct.run),
    StageSpec("evaluate", "evaluate", ("mct",),     evaluate.run),
)

_BY_NAME = {s.name: s for s in REGISTRY}


def order() -> list[str]:
    return [s.name for s in REGISTRY]


def by_name(name: str) -> StageSpec:
    return _BY_NAME[name]


def upstream_of(name: str) -> tuple[str, ...]:
    return _BY_NAME[name].upstream


def dir_name(name: str) -> str:
    return _BY_NAME[name].dir_name


def validate_registry(reg: tuple[StageSpec, ...] = REGISTRY) -> None:
    """Assert structural invariants. Runs at import for the real REGISTRY; also
    callable on an arbitrary tuple for tests."""
    by_name_map: dict[str, StageSpec] = {}
    seen_dirs: set[str] = set()
    for spec in reg:
        if spec.name in by_name_map:
            raise ValueError(f"duplicate stage name in registry: {spec.name!r}")
        if spec.dir_name in seen_dirs:
            raise ValueError(f"duplicate dir_name in registry: {spec.dir_name!r}")
        by_name_map[spec.name] = spec
        seen_dirs.add(spec.dir_name)

    seen: set[str] = set()
    for spec in reg:
        if not callable(spec.run):
            raise ValueError(f"stage {spec.name!r} has a non-callable run")
        for up in spec.upstream:
            if up not in by_name_map:
                raise ValueError(f"stage {spec.name!r} depends on unknown stage {up!r}")
            if up not in seen:
                raise ValueError(
                    f"stage {spec.name!r} depends on {up!r} which does not precede it"
                )
        seen.add(spec.name)


validate_registry()
