#!/usr/bin/env python
"""Compare experiment metrics against the locked baseline.

Usage:

    # Print one table with all completed variants, sorted by HOTA, deltas vs baseline.
    python experiments/compare.py

    # Restrict to one experiment.
    python experiments/compare.py --experiment eps_mcpt_sweep

    # Sort by a different metric.
    python experiments/compare.py --sort-by mct_world.HOTA

    # Also write a markdown report.
    python experiments/compare.py --markdown report.md

    # Include incomplete runs (showing N/A).
    python experiments/compare.py --include-incomplete

The table contains, per run:
  * mean per-camera HOTA / IDF1 / MOTA / MOTP / CLR_F1 (image-space, averaged
    across the 7 cameras)
  * mct_world.HOTA / DetA / AssA / IDF1 / MOTA (scene-level world-space)
  * total runtime (sum of stage manifest runtime_sec; ~0 for cached stages)
  * delta vs baseline on the sort metric

This file has no dependency on rich/tabulate — plain stdlib only — to keep
the harness portable.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._lib import (  # noqa: E402
    STAGES,
    STAGE_DIR,
    load_registry,
    repo_root,
    variant_run_id,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("compare")


# Metrics we surface in the table. "image.*" = averaged per-camera; "mct_world.*"
# = scene-level world MCT. Tuple of (column_label, dotted_path).
METRICS = [
    ("HOTA",         "image.HOTA"),
    ("IDF1",         "image.IDF1"),
    ("MOTA",         "image.MOTA"),
    ("MOTP",         "image.MOTP"),
    ("CLR_F1",       "image.CLR_F1"),
    ("wHOTA",        "mct_world.HOTA"),
    ("wDetA",        "mct_world.DetA"),
    ("wAssA",        "mct_world.AssA"),
    ("wIDF1",        "mct_world.IDF1"),
    ("wMOTA",        "mct_world.MOTA"),
]
DEFAULT_SORT = "image.HOTA"
BASELINE_RUN_ID = "baseline"


@dataclass
class Row:
    run_id: str
    experiment: str | None
    variant: str | None
    metrics: dict[str, float | None]
    runtime_sec: float | None
    complete: bool
    note: str = ""


def _load_metrics(run_dir: Path) -> tuple[dict, bool]:
    p = run_dir / "evaluate" / "metrics.json"
    if not p.exists():
        return {}, False
    try:
        return json.loads(p.read_text()), True
    except Exception:
        return {}, False


def _avg_image_metric(metrics_body: dict, key: str) -> float | None:
    """Average key across all per-camera blocks (S001-camera_*)."""
    vals: list[float] = []
    for k, v in metrics_body.items():
        if not isinstance(v, dict):
            continue
        if k == "mct_world":
            continue
        if key in v and v[key] is not None:
            vals.append(float(v[key]))
    if not vals:
        return None
    return sum(vals) / len(vals)


def _extract(metrics_body: dict, dotted: str) -> float | None:
    space, key = dotted.split(".", 1)
    if space == "image":
        return _avg_image_metric(metrics_body, key)
    if space == "mct_world":
        block = metrics_body.get("mct_world") or {}
        v = block.get(key)
        return float(v) if v is not None else None
    return None


def _total_runtime(run_dir: Path) -> float | None:
    total = 0.0
    any_found = False
    for stage in STAGES:
        mp = run_dir / STAGE_DIR[stage] / "manifest.json"
        if not mp.exists():
            continue
        try:
            m = json.loads(mp.read_text())
        except Exception:
            continue
        rt = m.get("runtime_sec")
        if rt is not None:
            total += float(rt)
            any_found = True
    return total if any_found else None


def _row_for(run_dir: Path, *, experiment: str | None = None,
             variant: str | None = None) -> Row:
    body, complete = _load_metrics(run_dir)
    metrics = {label: _extract(body, path) for label, path in METRICS}
    return Row(
        run_id=run_dir.name,
        experiment=experiment,
        variant=variant,
        metrics=metrics,
        runtime_sec=_total_runtime(run_dir) if complete else None,
        complete=complete,
    )


def _collect_rows(args: argparse.Namespace) -> tuple[Row | None, list[Row]]:
    repo = repo_root()
    outputs = repo / "outputs"

    baseline = _row_for(outputs / BASELINE_RUN_ID,
                        experiment="(baseline)", variant="")
    baseline_row: Row | None = baseline if baseline.complete else None
    if not baseline.complete:
        log.warning(
            "baseline not found / incomplete — deltas won't be shown."
        )

    rows: list[Row] = []
    exps = load_registry(Path(args.registry))
    for e in exps:
        if args.experiment and e["id"] != args.experiment:
            continue
        for v in e["variants"]:
            rid = variant_run_id(e["id"], v["name"])
            r = _row_for(outputs / rid, experiment=e["id"], variant=v["name"])
            if not r.complete and not args.include_incomplete:
                continue
            rows.append(r)
    return baseline_row, rows


def _fmt(v: float | None, *, digits: int = 4) -> str:
    if v is None:
        return "  —   "
    return f"{v:.{digits}f}"


def _fmt_delta(v: float | None, base: float | None) -> str:
    if v is None or base is None:
        return "       "
    d = v - base
    sign = "+" if d >= 0 else "-"
    return f"{sign}{abs(d):.3f}"


def _print_table(baseline: Row | None, rows: list[Row], sort_by: str) -> None:
    # Sort: baseline always first, then by sort_by descending.
    base_metric = baseline.metrics.get(_label_for(sort_by)) if baseline else None
    def sort_key(r: Row) -> float:
        v = r.metrics.get(_label_for(sort_by))
        return -(v if v is not None else float("-inf"))
    rows.sort(key=sort_key)

    # Column widths.
    name_w = max(
        [len("run_id")]
        + [len(r.run_id) for r in rows]
        + ([len(baseline.run_id)] if baseline else [])
    )
    metric_w = 7

    # Header.
    head = f"  {'run_id':<{name_w}} | " + " ".join(
        f"{lbl:>{metric_w}}" for lbl, _ in METRICS
    ) + f"  |  {'Δ' + _label_for(sort_by):>9}  | runtime"
    print(head)
    print("  " + "-" * (len(head) - 2))

    def row_to_line(r: Row, marker: str = " ") -> str:
        cells = " ".join(
            f"{_fmt(r.metrics[lbl]):>{metric_w}}" for lbl, _ in METRICS
        )
        delta = _fmt_delta(r.metrics.get(_label_for(sort_by)), base_metric)
        rt = f"{r.runtime_sec:>6.0f}s" if r.runtime_sec is not None else "    —  "
        return f"{marker} {r.run_id:<{name_w}} | {cells}  |  {delta:>9}  | {rt}"

    if baseline:
        print(row_to_line(baseline, marker="*"))
        print("  " + "-" * (len(head) - 2))
    for r in rows:
        print(row_to_line(r))

    print()
    print(f"  sort: {sort_by}  (descending; * = baseline)")
    if any(not r.complete for r in rows):
        print("  some rows are incomplete (—); use --include-incomplete to see them.")


def _label_for(dotted: str) -> str:
    """Map 'image.HOTA' → 'HOTA', 'mct_world.HOTA' → 'wHOTA' (column label)."""
    for lbl, path in METRICS:
        if path == dotted:
            return lbl
    raise ValueError(f"unknown sort metric: {dotted}")


def _write_markdown(out_path: Path, baseline: Row | None,
                    rows: list[Row], sort_by: str) -> None:
    # Already sorted by _print_table; we resort to be self-contained.
    base_metric = baseline.metrics.get(_label_for(sort_by)) if baseline else None
    rows = list(rows)
    def sort_key(r: Row) -> float:
        v = r.metrics.get(_label_for(sort_by))
        return -(v if v is not None else float("-inf"))
    rows.sort(key=sort_key)

    lines: list[str] = []
    lines.append(f"# Experiment results (sort: `{sort_by}`)")
    lines.append("")
    cols = ["run_id"] + [lbl for lbl, _ in METRICS] + [f"Δ {_label_for(sort_by)}", "runtime_s"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")

    def row_md(r: Row, mark: bool = False) -> str:
        cells = [_fmt(r.metrics[lbl]) for lbl, _ in METRICS]
        delta = _fmt_delta(r.metrics.get(_label_for(sort_by)), base_metric)
        rt = f"{r.runtime_sec:.0f}" if r.runtime_sec is not None else "—"
        run_id = f"**{r.run_id}**" if mark else r.run_id
        return "| " + " | ".join([run_id] + cells + [delta, rt]) + " |"

    if baseline:
        lines.append(row_md(baseline, mark=True))
    for r in rows:
        lines.append(row_md(r))
    out_path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser("compare")
    p.add_argument("--registry", default="experiments/registry.yaml")
    p.add_argument("--experiment", default=None,
                   help="Restrict to one experiment id.")
    p.add_argument("--sort-by", default=DEFAULT_SORT,
                   choices=[m for _, m in METRICS],
                   help=f"Metric to sort by (default {DEFAULT_SORT}).")
    p.add_argument("--include-incomplete", action="store_true",
                   help="Show variants whose metrics.json is missing.")
    p.add_argument("--markdown", default=None,
                   help="Also write a markdown table to this path.")
    args = p.parse_args(argv)

    baseline, rows = _collect_rows(args)
    if not rows and baseline is None:
        print("no completed runs to compare yet.")
        return 0
    _print_table(baseline, rows, args.sort_by)
    if args.markdown:
        _write_markdown(Path(args.markdown), baseline, rows, args.sort_by)
        print(f"\n  wrote markdown: {args.markdown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
