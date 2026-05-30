"""Internal helpers for the experiment harness.

Three responsibilities, each clearly named:

  * deep_merge / load_registry / load_yaml / dump_yaml — config plumbing.
  * stages_to_reuse — given the experiment's `rerun_from`, which stages can we
    inherit from the baseline run?
  * setup_cache_symlinks — physically symlink the cached stage directories
    from outputs/baseline/<stage>/ into outputs/<run_id>/<stage>/ so the
    pipeline's stage gate sees a complete-and-ok manifest and skips.

NOTE on the "external symlinks" gotcha
--------------------------------------
The upstream pipeline relies on shared symlinks under external/ (e.g.
external/Detection -> <run_dir>/detect/) that each stage updates as a side
effect of running. When we *gate-skip* a stage to reuse its outputs, that
symlink is NOT refreshed. To avoid downstream stages reading a stale link, the
runner uses pipeline.py's `--force` ONLY on the *first* stage we re-run
(triggering its own symlink updates from then on), but for the *upstream*
stages whose outputs we inherit, it also re-points the external symlinks at
the new run's stage dirs by importing and calling each stage module's setup
helpers indirectly via a single explicit `prime_external_symlinks` pass.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


# Stage order and dir-name mapping derive from the single source of truth in
# aic24_nvidia.registry. Kept as module-level views so the rest of the harness
# (stages_to_reuse/rerun, compare.py, run.py) is unchanged — but there is no
# longer a hand-maintained duplicate to drift from pipeline.py.
from aic24_nvidia import registry

STAGES = tuple(registry.order())
STAGE_DIR = {s: registry.dir_name(s) for s in STAGES}


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def dump_yaml(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict:
    """Recursively merge `overrides` into a copy of `base`.

    Mappings merge key-by-key. Lists and scalars from `overrides` replace
    whatever was in `base`. Returns a new dict; inputs are not mutated.
    """
    out: dict = dict(base)
    for k, v in overrides.items():
        if k in out and isinstance(out[k], Mapping) and isinstance(v, Mapping):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_registry(registry_path: Path) -> list[dict]:
    body = load_yaml(registry_path)
    exps = body.get("experiments") or []
    seen = set()
    for e in exps:
        for required in ("id", "base_config", "rerun_from", "variants"):
            if required not in e:
                raise ValueError(f"experiment missing '{required}': {e}")
        if e["id"] in seen:
            raise ValueError(f"duplicate experiment id: {e['id']}")
        seen.add(e["id"])
        if e["rerun_from"] not in STAGES:
            raise ValueError(
                f"experiment {e['id']}: rerun_from={e['rerun_from']!r} "
                f"not in {STAGES}"
            )
        for v in e["variants"]:
            if "name" not in v:
                raise ValueError(f"variant missing 'name' in exp {e['id']}: {v}")
            v.setdefault("overrides", {})
    return exps


def stages_to_reuse(rerun_from: str) -> list[str]:
    idx = STAGES.index(rerun_from)
    return list(STAGES[:idx])


def stages_to_rerun(rerun_from: str) -> list[str]:
    idx = STAGES.index(rerun_from)
    return list(STAGES[idx:])


def variant_run_id(exp_id: str, variant_name: str) -> str:
    """Stable run-id for a (experiment, variant) pair.

    `exp__variant` — note the double underscore. The compare tool relies on
    this to split run-ids back into (experiment, variant).
    """
    safe_variant = variant_name.replace("/", "_").replace(" ", "_")
    return f"{exp_id}__{safe_variant}"


def setup_cache_symlinks(
    *,
    baseline_run_dir: Path,
    target_run_dir: Path,
    stages: list[str],
) -> list[str]:
    """For each stage in `stages`, link `target_run_dir/<stage>/` -> baseline's.

    Skips stages whose baseline output doesn't exist yet. Returns the list of
    stages that were actually linked (subset of `stages`).

    We link the *whole stage directory* — manifest, atomic outputs, everything.
    The pipeline gate just checks `manifest.json` exists and status==ok, so
    following a symlink for that check works transparently.
    """
    target_run_dir.mkdir(parents=True, exist_ok=True)
    linked: list[str] = []
    for stage in stages:
        stage_dir_name = STAGE_DIR[stage]
        baseline_stage = baseline_run_dir / stage_dir_name
        target_stage = target_run_dir / stage_dir_name
        if not baseline_stage.exists():
            # Baseline hasn't computed this stage yet; nothing to reuse.
            continue
        if target_stage.exists() or target_stage.is_symlink():
            # Already linked or already a real dir — don't clobber.
            if target_stage.is_symlink():
                linked.append(stage)
            continue
        # Use absolute target so the symlink works regardless of cwd.
        target_stage.symlink_to(baseline_stage.resolve(),
                                target_is_directory=True)
        linked.append(stage)
    return linked


def prime_external_symlinks(
    *,
    run_dir: Path,
    external_root: Path,
    yachiyo_root: Path,
    reused_stages: list[str],
) -> None:
    """Point the upstream's *global* external/ symlinks at this run's stages.

    The upstream YACHIYO repo (and BoT-SORT/mmpose siblings) read inputs via
    symlinks under external/: external/Original, external/Detection,
    external/EmbedFeature, external/Pose, and yachiyo/Tracking. Each stage's
    .run() resets the relevant link when it executes. If a stage is gate-
    skipped (cache-reuse), that update doesn't happen, so a later stage might
    read from the previous experiment's outputs.

    This function pre-stages those symlinks so that EVEN IF we cache-reuse
    upstream stages, the external/ links point at the current run_dir/<stage>/
    (which is itself a symlink into baseline's outputs).

    It replays each reused stage's declared wiring (registry.StageSpec.wiring) —
    the SAME source of truth the live run uses via base.atomic_stage — so the
    cache-reuse and live paths can no longer diverge. (Previously this was a
    hand-maintained mirror that already omitted mct.)
    """
    from types import SimpleNamespace

    from aic24_nvidia.bootstrap import make_symlink  # local import

    run_dir = Path(run_dir)
    cfg = SimpleNamespace(
        external_root=Path(external_root), yachiyo_root=Path(yachiyo_root)
    )
    for stage in reused_stages:
        spec = registry.by_name(stage)
        output_dir = (run_dir / spec.dir_name).resolve()
        if not output_dir.exists():
            continue
        for link, target in spec.wiring(run_dir, cfg, output_dir):
            target = Path(target).resolve()
            if target.exists():
                make_symlink(target, link)
