#!/usr/bin/env python
"""CLI for the durable results ledger.

    # Record every completed run in outputs/ and refresh results/README.md:
    python scripts/results.py scan

    # Record a single run (used by scripts/archive_run.sh before it deletes a run):
    python scripts/results.py add v2_solider
    python scripts/results.py add v2_solider \
        --archived-remote onedrive --archived-path onedrive:aic24/outputs-archive/v2_solider.tar.zst

    # Re-render results/README.md from results/runs.jsonl (no scan):
    python scripts/results.py render

The library doing the work is aic24_nvidia/results.py. This wrapper just wires in
the repo paths, the experiment registry (for run_id -> experiment inference), the
optional results/labels.json, and best-effort git info.

It is tolerant of a bare (non-venv) python: registry/label lookups degrade to
empty and metric extraction still works — handy when archive_run.sh shells out.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from aic24_nvidia import results as R  # noqa: E402

LEDGER = REPO_ROOT / "results" / "runs.jsonl"
README = REPO_ROOT / "results" / "README.md"


def _known_experiments(registry_path: Path) -> frozenset[str]:
    try:
        import yaml

        body = yaml.safe_load(registry_path.read_text()) or {}
        return frozenset(e["id"] for e in (body.get("experiments") or []))
    except Exception:
        return frozenset()


def _labels(path: Path) -> dict:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except (OSError, ValueError):
        return {}


def _git() -> dict:
    def g(*args: str) -> str | None:
        try:
            r = subprocess.run(
                ["git", *args], capture_output=True, text=True, cwd=REPO_ROOT, timeout=5
            )
            return r.stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None

    return {"branch": g("rev-parse", "--abbrev-ref", "HEAD"), "commit": g("rev-parse", "--short", "HEAD")}


def _render_to_readme(records: list[R.RunRecord]) -> None:
    README.parent.mkdir(parents=True, exist_ok=True)
    README.write_text(R.render_markdown(records))


def cmd_scan(args: argparse.Namespace) -> int:
    outputs = REPO_ROOT / "outputs"
    known = _known_experiments(REPO_ROOT / args.registry)
    labels = _labels(REPO_ROOT / args.labels)
    git = _git()

    records = R.load_ledger(LEDGER)
    n = 0
    for ev in sorted(outputs.glob("*/evaluate/metrics.json")):
        rec = R.extract_record(
            ev.parent.parent, known_experiments=known, labels=labels, git=git
        )
        if rec is not None:
            records = R.upsert(records, rec)
            n += 1
    R.save_ledger(LEDGER, records)
    _render_to_readme(records)
    print(
        f"recorded {n} run(s) → {LEDGER.relative_to(REPO_ROOT)}; "
        f"rendered {README.relative_to(REPO_ROOT)} ({len(records)} total)"
    )
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    known = _known_experiments(REPO_ROOT / args.registry)
    labels = _labels(REPO_ROOT / args.labels)
    git = _git()

    archived = None
    if args.archived_remote and args.archived_path:
        archived = {
            "remote": args.archived_remote,
            "remote_path": args.archived_path,
            "archived_at": R.now_iso(),
        }
    rec = R.extract_record(
        REPO_ROOT / "outputs" / args.run_id,
        known_experiments=known,
        labels=labels,
        git=git,
        archived=archived,
    )
    if rec is None:
        print(
            f"no metrics for {args.run_id} (missing evaluate/metrics.json) — nothing recorded",
            file=sys.stderr,
        )
        return 1
    records = R.upsert(R.load_ledger(LEDGER), rec)
    R.save_ledger(LEDGER, records)
    _render_to_readme(records)
    where = " [archived]" if archived else ""
    print(f"recorded {args.run_id}{where} → {LEDGER.relative_to(REPO_ROOT)}")
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    records = R.load_ledger(LEDGER)
    _render_to_readme(records)
    print(f"rendered {README.relative_to(REPO_ROOT)} from {len(records)} run(s)")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser("results", description="Durable results ledger.")
    p.add_argument("--registry", default="experiments/registry.yaml")
    p.add_argument("--labels", default="results/labels.json")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("scan", help="record all outputs/*/evaluate runs + refresh README")
    sub.add_parser("render", help="re-render README from the JSONL")
    a = sub.add_parser("add", help="record a single run")
    a.add_argument("run_id")
    a.add_argument("--archived-remote", default=None)
    a.add_argument("--archived-path", default=None)

    args = p.parse_args(argv)
    return {"scan": cmd_scan, "add": cmd_add, "render": cmd_render}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
