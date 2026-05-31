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

Runs also auto-record themselves after `evaluate` (pipeline.py calls
results.record_run), so `scan` is mainly for back-filling or after manual edits.
All the work lives in aic24_nvidia/results.py; this is a thin wrapper.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from aic24_nvidia import results as R  # noqa: E402


def cmd_scan(args: argparse.Namespace) -> int:
    n = R.scan_outputs(REPO_ROOT)
    ledger, readme = R.ledger_paths(REPO_ROOT)
    print(
        f"recorded {n} run(s) → {ledger.relative_to(REPO_ROOT)}; "
        f"rendered {readme.relative_to(REPO_ROOT)}"
    )
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    archived = None
    if args.archived_remote and args.archived_path:
        archived = {
            "remote": args.archived_remote,
            "remote_path": args.archived_path,
            "archived_at": R.now_iso(),
        }
    rec = R.record_run(args.run_id, repo_root=REPO_ROOT, archived=archived)
    if rec is None:
        print(
            f"no metrics for {args.run_id} (missing evaluate/metrics.json) — nothing recorded",
            file=sys.stderr,
        )
        return 1
    ledger, _ = R.ledger_paths(REPO_ROOT)
    where = " [archived]" if archived else ""
    print(f"recorded {args.run_id}{where} → {ledger.relative_to(REPO_ROOT)}")
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    ledger, readme = R.ledger_paths(REPO_ROOT)
    records = R.load_ledger(ledger)
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text(R.render_markdown(records))
    print(f"rendered {readme.relative_to(REPO_ROOT)} from {len(records)} run(s)")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser("results", description="Durable results ledger.")
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
