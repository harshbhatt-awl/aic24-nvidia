#!/usr/bin/env python
"""Experiment runner.

Usage:

    # List all experiments and their variant counts.
    python experiments/run.py list

    # Run every variant of one experiment.
    python experiments/run.py run eps_mcpt_sweep

    # Run a single variant.
    python experiments/run.py run eps_mcpt_sweep --variant 0.30

    # Force re-run even if outputs already exist.
    python experiments/run.py run eps_mcpt_sweep --variant 0.30 --force

    # Show what's been run.
    python experiments/run.py status

Cache-reuse: each variant inherits all upstream stages (those before
`rerun_from`) from outputs/baseline/ via symlinks. The variant's `--run-id` is
`<experiment>__<variant>`.

Before running any variant, outputs/baseline/ must exist and be complete. The
harness will refuse to start otherwise — see `python experiments/run.py
ensure-baseline` to (re)build it.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow `python experiments/run.py` to import the aic24_nvidia package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._lib import (  # noqa: E402
    STAGE_DIR,
    deep_merge,
    dump_yaml,
    load_registry,
    load_yaml,
    prime_external_symlinks,
    repo_root,
    setup_cache_symlinks,
    stages_to_rerun,
    stages_to_reuse,
    variant_run_id,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s exp: %(message)s",
)
log = logging.getLogger("experiments")


BASELINE_RUN_ID = "baseline"
BASELINE_CONFIG_DEFAULT = "configs/baseline.yaml"
REGISTRY_DEFAULT = "experiments/registry.yaml"


# --------------------------------------------------------------------------- #
# Subcommand: list
# --------------------------------------------------------------------------- #

def cmd_list(args: argparse.Namespace) -> int:
    exps = load_registry(Path(args.registry))
    print(f"{'ID':<24} {'#var':>5}  {'rerun_from':<10}  description")
    print("-" * 80)
    for e in exps:
        print(
            f"{e['id']:<24} {len(e['variants']):>5}  "
            f"{e['rerun_from']:<10}  {e.get('description', '')}"
        )
    return 0


# --------------------------------------------------------------------------- #
# Subcommand: status
# --------------------------------------------------------------------------- #

def _has_metrics(run_dir: Path) -> bool:
    return (run_dir / "evaluate" / "metrics.json").exists()


def _run_meta(run_dir: Path) -> dict:
    """Best-effort metadata about a completed run."""
    eval_manifest = run_dir / "evaluate" / "manifest.json"
    if not eval_manifest.exists():
        return {"status": "incomplete"}
    try:
        m = json.loads(eval_manifest.read_text())
    except Exception:
        return {"status": "incomplete"}
    return {
        "status": m.get("status", "?"),
        "finished_at": m.get("finished_at", ""),
        "runtime_sec": m.get("runtime_sec"),
    }


def cmd_status(args: argparse.Namespace) -> int:
    repo = repo_root()
    outputs = repo / "outputs"
    exps = load_registry(Path(args.registry))

    baseline_ok = _has_metrics(outputs / BASELINE_RUN_ID)
    bmeta = _run_meta(outputs / BASELINE_RUN_ID)
    print(f"baseline: {'OK' if baseline_ok else 'MISSING'} "
          f"({bmeta.get('finished_at', '')})")
    print()
    for e in exps:
        print(f"[{e['id']}]")
        for v in e["variants"]:
            rid = variant_run_id(e["id"], v["name"])
            done = _has_metrics(outputs / rid)
            meta = _run_meta(outputs / rid)
            mark = "✓" if done else "·"
            extra = ""
            if done and meta.get("runtime_sec") is not None:
                extra = f"   ({meta['runtime_sec']:.0f}s, {meta.get('finished_at','')})"
            print(f"  {mark} {v['name']:<10}  run_id={rid}{extra}")
        print()
    return 0


# --------------------------------------------------------------------------- #
# Subcommand: run
# --------------------------------------------------------------------------- #

def _materialize_variant_config(
    *,
    base_config_path: Path,
    overrides: dict,
    target_run_dir: Path,
) -> Path:
    """Write the merged YAML into the run dir, return its path."""
    base = load_yaml(base_config_path)
    merged = deep_merge(base, overrides)
    out_path = target_run_dir / "_config.yaml"
    dump_yaml(merged, out_path)
    return out_path


def _invoke_pipeline(
    *,
    config_path: Path,
    run_id: str,
    rerun_from: str,
    force: bool,
) -> int:
    """Run pipeline.py for the stages from `rerun_from` onward.

    We invoke each stage individually so we can pass --force only to the *first*
    rerun stage. Cached upstream stages are gate-skipped automatically.
    """
    repo = repo_root()
    venv_python = repo / ".venv" / "bin" / "python"
    python = str(venv_python) if venv_python.exists() else sys.executable

    stages = stages_to_rerun(rerun_from)
    log.info("run_id=%s  rerun stages=%s", run_id, stages)

    for i, stage in enumerate(stages):
        cmd = [
            python, "pipeline.py", stage,
            "--config", str(config_path),
            "--run-id", run_id,
        ]
        # Force re-run only on the *first* targeted stage so that previously-
        # computed downstream stages from a stale earlier attempt get refreshed
        # too (downstream stages will see upstream re-ran and won't skip).
        if i == 0 or force:
            cmd.append("--force")
        log.info("  $ %s", " ".join(cmd))
        rc = subprocess.run(cmd, cwd=str(repo)).returncode
        if rc != 0:
            log.error("stage %s failed (rc=%d)", stage, rc)
            return rc
    return 0


def _run_one_variant(
    *,
    exp: dict,
    variant: dict,
    force: bool,
    registry_path: Path,
) -> int:
    repo = repo_root()
    rid = variant_run_id(exp["id"], variant["name"])
    target_run_dir = repo / "outputs" / rid
    baseline_run_dir = repo / "outputs" / BASELINE_RUN_ID

    if not _has_metrics(baseline_run_dir):
        log.error(
            "baseline not found at %s — run it first: "
            "python pipeline.py all --config %s --run-id baseline",
            baseline_run_dir, exp["base_config"],
        )
        return 2

    # Idempotency: if the variant's metrics.json already exists and we're not
    # forcing, skip.
    if _has_metrics(target_run_dir) and not force:
        log.info("variant %s already complete; skip. Use --force to rerun.", rid)
        return 0

    # If forcing, blow away the variant's run dir first to start clean.
    if force and target_run_dir.exists():
        log.info("--force: removing %s", target_run_dir)
        _rm_tree_handling_symlinks(target_run_dir)

    # 1. Set up cache symlinks for the upstream stages.
    reuse = stages_to_reuse(exp["rerun_from"])
    linked = setup_cache_symlinks(
        baseline_run_dir=baseline_run_dir,
        target_run_dir=target_run_dir,
        stages=reuse,
    )
    log.info("cache-reuse: linked %d stage(s): %s", len(linked), linked)

    # 2. Write the merged variant config.
    base_config_path = repo / exp["base_config"]
    config_path = _materialize_variant_config(
        base_config_path=base_config_path,
        overrides=variant["overrides"],
        target_run_dir=target_run_dir,
    )

    # 3. Re-point the global external/ symlinks at this variant's run_dir
    #    BEFORE invoking the pipeline, so cache-reused stages don't leave
    #    stale upstream links.
    prime_external_symlinks(
        run_dir=target_run_dir,
        external_root=repo / "external",
        yachiyo_root=repo / "external" / "AIC24_Track1_YACHIYO_RIIPS",
        reused_stages=linked,
    )

    # 4. Record provenance.
    (target_run_dir / "_experiment.json").write_text(json.dumps({
        "experiment_id": exp["id"],
        "variant_name": variant["name"],
        "base_config": str(base_config_path.relative_to(repo)),
        "overrides": variant["overrides"],
        "rerun_from": exp["rerun_from"],
        "reused_stages": linked,
        "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "registry": str(registry_path.relative_to(repo)),
    }, indent=2))

    # 5. Run the pipeline.
    return _invoke_pipeline(
        config_path=config_path,
        run_id=rid,
        rerun_from=exp["rerun_from"],
        force=force,
    )


def _rm_tree_handling_symlinks(path: Path) -> None:
    """rmtree that doesn't follow symlinks into baseline (which would delete it)."""
    import shutil
    if path.is_symlink():
        path.unlink()
        return
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_symlink():
            child.unlink()
        elif child.is_dir():
            _rm_tree_handling_symlinks(child)
        else:
            child.unlink()
    path.rmdir()


def cmd_run(args: argparse.Namespace) -> int:
    registry_path = Path(args.registry).resolve()
    exps = load_registry(registry_path)
    exp = next((e for e in exps if e["id"] == args.experiment), None)
    if exp is None:
        log.error("unknown experiment: %s (use `list`)", args.experiment)
        return 2

    variants = exp["variants"]
    if args.variant:
        variants = [v for v in variants if v["name"] == args.variant]
        if not variants:
            log.error("unknown variant in %s: %s", exp["id"], args.variant)
            return 2

    rc_final = 0
    for v in variants:
        log.info("=" * 70)
        log.info("EXPERIMENT %s  /  VARIANT %s", exp["id"], v["name"])
        log.info("=" * 70)
        rc = _run_one_variant(
            exp=exp, variant=v, force=args.force, registry_path=registry_path,
        )
        if rc != 0:
            log.error("variant %s failed (rc=%d)", v["name"], rc)
            rc_final = rc
            if args.stop_on_failure:
                return rc
    return rc_final


# --------------------------------------------------------------------------- #
# Subcommand: ensure-baseline
# --------------------------------------------------------------------------- #

def cmd_ensure_baseline(args: argparse.Namespace) -> int:
    repo = repo_root()
    baseline_dir = repo / "outputs" / BASELINE_RUN_ID
    if _has_metrics(baseline_dir) and not args.force:
        log.info("baseline already present at %s; nothing to do.", baseline_dir)
        log.info("Use --force to rebuild from scratch.")
        return 0

    config_path = repo / (args.config or BASELINE_CONFIG_DEFAULT)
    venv_python = repo / ".venv" / "bin" / "python"
    python = str(venv_python) if venv_python.exists() else sys.executable
    cmd = [python, "pipeline.py", "all",
           "--config", str(config_path), "--run-id", BASELINE_RUN_ID]
    if args.force:
        cmd.append("--force")
    log.info("$ %s", " ".join(cmd))
    return subprocess.run(cmd, cwd=str(repo)).returncode


# --------------------------------------------------------------------------- #
# Argparse wiring
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser("experiments")
    p.add_argument("--registry", default=REGISTRY_DEFAULT)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("list", help="List all experiments.")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("status", help="Show which variants have completed.")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("run", help="Run one experiment (all variants or one).")
    sp.add_argument("experiment", help="Experiment id (from `list`).")
    sp.add_argument("--variant", default=None,
                    help="Single variant name. Default: all variants.")
    sp.add_argument("--force", action="store_true",
                    help="Rebuild even if outputs already exist.")
    sp.add_argument("--stop-on-failure", action="store_true",
                    help="Stop on first failing variant.")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("ensure-baseline",
                        help="Build outputs/baseline/ if missing.")
    sp.add_argument("--config", default=None,
                    help=f"Override baseline config (default: {BASELINE_CONFIG_DEFAULT}).")
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_ensure_baseline)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
