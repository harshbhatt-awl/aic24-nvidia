"""Durable results ledger for pipeline runs.

Each completed run writes metrics to ``outputs/<run_id>/evaluate/metrics.json``
and per-stage ``manifest.json`` files (params + timings). Those live *inside* the
run dir, so when a run is archived to OneDrive and its local dir deleted
(``scripts/archive_run.sh``), the results vanish from disk with it. This module
extracts each run's results into a small, durable, git-tracked ledger
(``results/runs.jsonl``) and renders human-readable views (``results/README.md``)
organised **by day** and **by experiment**.

Stdlib only (no rich/tabulate), mirroring ``experiments/compare.py``. The CLI
lives in ``scripts/results.py``. The library is deliberately free of any
dependency on ``experiments/`` — callers pass the set of known experiment ids in.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

ISO = "%Y-%m-%dT%H:%M:%SZ"
BASELINE_RUN_ID = "baseline"

# Default model stack (v3) — shown in the config fingerprint only when a run
# deviates from it, to keep the rendered tables narrow.
DEFAULT_MODELS = {
    "detect": "yolo11x",
    "reid": "solider_swin_small",
    "pose": "rtmpose-l",
}

# Image metrics read straight from the COMBINED block; world from mct_world.
IMAGE_KEYS = ["HOTA", "IDF1", "MOTA", "MOTP", "CLR_F1"]
WORLD_KEYS = ["HOTA", "DetA", "AssA", "IDF1", "MOTA"]
WORLD_EXTRA = ["dropped_detections", "frames_evaluated", "d_max_m"]

# Tracker knobs surfaced in the config fingerprint (label shown in tables).
KNOB_LABELS = {
    "epsilon_scpt": "eps_s",
    "epsilon_mcpt": "eps_m",
    "keypoint_condition_th": "kp",
    "short_track_th": "stt",
    "sim_th": "sim",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime(ISO)


@dataclass
class RunRecord:
    run_id: str
    date: str | None
    finished_at: str | None
    experiment: str
    variant: str
    image: dict
    world: dict
    runtime_sec: float | None
    config: dict
    note: str = ""
    archived: dict | None = None
    git: dict | None = None
    recorded_at: str | None = None

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> RunRecord:
        names = {f.name for f in fields(cls)}
        return cls(**{k: d.get(k) for k in names})


# --------------------------------------------------------------------------- #
# Reading a run dir
# --------------------------------------------------------------------------- #


def _read_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _manifest(run_dir: Path, stage: str) -> dict:
    return _read_json(run_dir / stage / "manifest.json") or {}


def _all_manifests(run_dir: Path) -> list[dict]:
    out = []
    for p in run_dir.glob("*/manifest.json"):
        m = _read_json(p)
        if m:
            out.append(m)
    return out


def _image_metrics(metrics: dict) -> dict:
    block = metrics.get("COMBINED") or {}
    return {k: block.get(k) for k in IMAGE_KEYS}


def _world_metrics(metrics: dict) -> dict:
    block = metrics.get("mct_world") or {}
    out: dict = {k: block.get(k) for k in WORLD_KEYS}
    for extra in WORLD_EXTRA:
        if extra in block:
            out[extra] = block[extra]
    return out


def _finished_at(run_dir: Path) -> str | None:
    ev = _manifest(run_dir, "evaluate").get("finished_at")
    if ev:
        return ev
    times = [m["finished_at"] for m in _all_manifests(run_dir) if m.get("finished_at")]
    return max(times) if times else None


def _date_of(run_dir: Path) -> str | None:
    fa = _finished_at(run_dir)
    if fa:
        return fa[:10]
    try:
        ts = run_dir.stat().st_mtime
    except OSError:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def _runtime_sec(run_dir: Path) -> float | None:
    total = 0.0
    found = False
    for m in _all_manifests(run_dir):
        rt = m.get("runtime_sec")
        if rt is not None:
            total += float(rt)
            found = True
    return total if found else None


def _config(run_dir: Path) -> dict:
    """Compact fingerprint of *what this run was* — harvested from manifests."""
    det = _manifest(run_dir, "detect").get("params", {})
    rei = _manifest(run_dir, "reid").get("params", {})
    pos = _manifest(run_dir, "pose").get("params", {})
    mct = _manifest(run_dir, "mct").get("params", {})
    sct = _manifest(run_dir, "sct").get("params", {})
    tp = mct.get("tracking_params") or sct.get("tracking_params") or {}

    cfg: dict = {}
    if det.get("model"):
        cfg["detect"] = det["model"]
    if rei.get("model"):
        cfg["reid"] = rei["model"]
    if pos.get("model"):
        cfg["pose"] = pos["model"]
    for knob in KNOB_LABELS:
        if knob in tp:
            cfg[knob] = tp[knob]
    proj = (mct.get("world_projection") or {}).get("method")
    if proj:
        cfg["projection"] = proj
    if "hard_world_gate" in mct:
        cfg["hard_world_gate"] = mct["hard_world_gate"]
    return cfg


def infer_experiment(
    run_id: str,
    known_experiments: frozenset[str] = frozenset(),
    labels: dict | None = None,
    baseline_run_id: str = BASELINE_RUN_ID,
) -> tuple[str, str, str]:
    """Map a run_id to (experiment, variant, note). See spec for the rules."""
    labels = labels or {}
    if run_id in labels:
        lab = labels[run_id]
        return (
            lab.get("experiment", "ad-hoc"),
            lab.get("variant", run_id),
            lab.get("note", ""),
        )
    if run_id == baseline_run_id:
        return "(baseline)", "", ""
    if "__" in run_id:
        exp, variant = run_id.split("__", 1)
        if exp in known_experiments:
            return exp, variant, ""
    m = re.match(r"^(.*)_(\d{8})_(\d{6})$", run_id)
    if m:
        return "snapshot", m.group(1), ""
    return "ad-hoc", run_id, ""


def extract_record(
    run_dir: Path | str,
    *,
    known_experiments: frozenset[str] = frozenset(),
    labels: dict | None = None,
    git: dict | None = None,
    archived: dict | None = None,
    now: str | None = None,
) -> RunRecord | None:
    """Build a RunRecord from a run dir, or None if it has no metrics yet."""
    run_dir = Path(run_dir)
    metrics = _read_json(run_dir / "evaluate" / "metrics.json")
    if metrics is None:
        return None
    exp, variant, note = infer_experiment(run_dir.name, known_experiments, labels, BASELINE_RUN_ID)
    return RunRecord(
        run_id=run_dir.name,
        date=_date_of(run_dir),
        finished_at=_finished_at(run_dir),
        experiment=exp,
        variant=variant,
        image=_image_metrics(metrics),
        world=_world_metrics(metrics),
        runtime_sec=_runtime_sec(run_dir),
        config=_config(run_dir),
        note=note,
        archived=archived,
        git=git,
        recorded_at=now or now_iso(),
    )


# --------------------------------------------------------------------------- #
# Ledger persistence (JSONL)
# --------------------------------------------------------------------------- #


def load_ledger(path: Path | str) -> list[RunRecord]:
    path = Path(path)
    if not path.exists():
        return []
    out: list[RunRecord] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(RunRecord.from_json(json.loads(line)))
    return out


def upsert(records: list[RunRecord], rec: RunRecord) -> list[RunRecord]:
    """Replace any record with the same run_id, else append.

    Preserves a prior record's ``archived`` / ``note`` when the new record does
    not carry one — so a plain ``scan`` never wipes an archived marker or a
    curated note set by an earlier ``add``.
    """
    by_id = {r.run_id: r for r in records}
    prev = by_id.get(rec.run_id)
    if prev is not None:
        if not rec.note and prev.note:
            rec.note = prev.note
        if rec.archived is None and prev.archived is not None:
            rec.archived = prev.archived
    by_id[rec.run_id] = rec
    return list(by_id.values())


def save_ledger(path: Path | str, records: list[RunRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda r: ((r.date or ""), r.run_id))
    path.write_text("\n".join(json.dumps(r.to_json()) for r in ordered) + "\n")


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _fmt(v: object, digits: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def _delta(v: object, base: object) -> str:
    if v is None or base is None:
        return ""
    d = float(v) - float(base)  # type: ignore[arg-type]
    return f"{'+' if d >= 0 else '−'}{abs(d):.3f}"


def _runtime_str(sec: float | None) -> str:
    if sec is None:
        return "—"
    sec = float(sec)
    return f"{sec:.0f}s" if sec < 90 else f"{sec / 60:.0f}m"


def _config_str(cfg: dict) -> str:
    parts: list[str] = []
    for kind in ("detect", "reid", "pose"):
        v = cfg.get(kind)
        if v and v != DEFAULT_MODELS[kind]:
            parts.append(str(v))
    for knob, label in KNOB_LABELS.items():
        if knob in cfg:
            parts.append(f"{label}={cfg[knob]}")
    if cfg.get("projection"):
        parts.append(str(cfg["projection"]))
    return " ".join(parts) if parts else "—"


def _whota(r: RunRecord) -> float:
    v = (r.world or {}).get("HOTA")
    return float(v) if v is not None else float("-inf")


def _exp_variant(r: RunRecord) -> str:
    return f"{r.experiment}/{r.variant}" if r.variant else r.experiment


def render_markdown(
    records: list[RunRecord],
    *,
    baseline_run_id: str = BASELINE_RUN_ID,
    updated: str | None = None,
) -> str:
    recs = list(records)
    base = next((r for r in recs if r.run_id == baseline_run_id), None)
    base_w = (base.world or {}).get("HOTA") if base else None
    updated = updated or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    out: list[str] = []
    out.append("# Results ledger")
    out.append("")
    out.append(
        "_Auto-generated from `results/runs.jsonl` by `scripts/results.py`. "
        "Don't edit by hand — run `python scripts/results.py scan` to refresh._"
    )
    out.append("")
    if base:
        bi, bw = base.image or {}, base.world or {}
        out.append(
            f"- **Baseline** (`{baseline_run_id}`): image HOTA **{_fmt(bi.get('HOTA'))}**, "
            f"world HOTA **{_fmt(bw.get('HOTA'))}** _(world is the project's priority metric)_."
        )
    scored = [r for r in recs if (r.world or {}).get("HOTA") is not None]
    if scored:
        best = max(scored, key=_whota)
        out.append(
            f"- **Best world HOTA**: `{best.run_id}` ({_fmt((best.world or {}).get('HOTA'))}) "
            f"— {_exp_variant(best)}"
        )
    out.append(f"- **Runs recorded**: {len(recs)}  |  **Last updated**: {updated}")
    out.append("")
    out.append(
        "Metrics: `img*` = combined per-camera (image space); `w*` = scene 3D-world MCT "
        "(priority). `Δw` = world HOTA vs baseline."
    )
    out.append("")

    # ---- By day ----------------------------------------------------------- #
    out.append("## By day")
    out.append("")
    by_day: dict[str, list[RunRecord]] = {}
    for r in recs:
        by_day.setdefault(r.date or "unknown", []).append(r)
    day_hdr = (
        "| run_id | experiment / variant | img HOTA | w HOTA | w AssA | w IDF1 | Δw | config | runtime |"
    )
    day_sep = "|---|---|---|---|---|---|---|---|---|"
    for day in sorted(by_day, reverse=True):
        out.append(f"### {day}")
        out.append("")
        out.append(day_hdr)
        out.append(day_sep)
        for r in sorted(by_day[day], key=_whota, reverse=True):
            w = r.world or {}
            out.append(
                f"| `{r.run_id}` | {_exp_variant(r)} | {_fmt((r.image or {}).get('HOTA'))} "
                f"| {_fmt(w.get('HOTA'))} | {_fmt(w.get('AssA'))} | {_fmt(w.get('IDF1'))} "
                f"| {_delta(w.get('HOTA'), base_w)} | {_config_str(r.config or {})} "
                f"| {_runtime_str(r.runtime_sec)} |"
            )
        out.append("")

    # ---- By experiment ---------------------------------------------------- #
    out.append("## By experiment")
    out.append("")
    by_exp: dict[str, list[RunRecord]] = {}
    for r in recs:
        by_exp.setdefault(r.experiment, []).append(r)
    exp_hdr = (
        "| run_id | variant | img HOTA | w HOTA | w DetA | w AssA | w IDF1 | Δw | config | date |"
    )
    exp_sep = "|---|---|---|---|---|---|---|---|---|---|"
    for exp in sorted(by_exp, key=lambda e: (e != "(baseline)", e)):
        out.append(f"### {exp}")
        out.append("")
        out.append(exp_hdr)
        out.append(exp_sep)
        for r in sorted(by_exp[exp], key=lambda r: (r.variant or "", r.run_id)):
            w = r.world or {}
            out.append(
                f"| `{r.run_id}` | {r.variant or '—'} | {_fmt((r.image or {}).get('HOTA'))} "
                f"| {_fmt(w.get('HOTA'))} | {_fmt(w.get('DetA'))} | {_fmt(w.get('AssA'))} "
                f"| {_fmt(w.get('IDF1'))} | {_delta(w.get('HOTA'), base_w)} "
                f"| {_config_str(r.config or {})} | {r.date or '—'} |"
            )
        out.append("")

    return "\n".join(out).rstrip() + "\n"


# --------------------------------------------------------------------------- #
# Orchestration — the IO/wiring layer shared by the CLI and the pipeline hook.
# (The functions above are pure + unit-tested; these read the repo layout.)
# --------------------------------------------------------------------------- #


def ledger_paths(repo_root: Path | str) -> tuple[Path, Path]:
    """``(runs.jsonl, README.md)`` under ``<repo_root>/results/``."""
    base = Path(repo_root) / "results"
    return base / "runs.jsonl", base / "README.md"


def repo_known_experiments(repo_root: Path | str) -> frozenset[str]:
    """Experiment ids from ``experiments/registry.yaml`` (empty if unavailable)."""
    try:
        import yaml

        path = Path(repo_root) / "experiments" / "registry.yaml"
        body = yaml.safe_load(path.read_text()) or {}
        return frozenset(e["id"] for e in (body.get("experiments") or []))
    except Exception:
        return frozenset()


def repo_labels(repo_root: Path | str) -> dict:
    """Curated run labels from ``results/labels.json`` (empty if absent/bad)."""
    p = Path(repo_root) / "results" / "labels.json"
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except (OSError, ValueError):
        return {}


def repo_git(repo_root: Path | str) -> dict:
    """Best-effort current branch/commit; values are None if git is unavailable."""
    import subprocess

    def g(*args: str) -> str | None:
        try:
            r = subprocess.run(
                ["git", *args], capture_output=True, text=True, cwd=str(repo_root), timeout=5
            )
            return r.stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None

    return {"branch": g("rev-parse", "--abbrev-ref", "HEAD"), "commit": g("rev-parse", "--short", "HEAD")}


def _write_views(repo_root: Path | str, records: list[RunRecord]) -> None:
    ledger, readme = ledger_paths(repo_root)
    save_ledger(ledger, records)
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text(render_markdown(records))


def record_run(
    run_id: str,
    *,
    repo_root: Path | str,
    run_dir: Path | str | None = None,
    archived: dict | None = None,
) -> RunRecord | None:
    """Record one run into the ledger and re-render the README.

    Returns the RunRecord, or None (no-op) when the run has no metrics yet — so it
    is safe to call after *any* stage. This is the single wiring point shared by
    the CLI (`results.py add`) and the pipeline's post-evaluate hook.
    """
    repo_root = Path(repo_root)
    run_dir = Path(run_dir) if run_dir is not None else repo_root / "outputs" / run_id
    rec = extract_record(
        run_dir,
        known_experiments=repo_known_experiments(repo_root),
        labels=repo_labels(repo_root),
        git=repo_git(repo_root),
        archived=archived,
    )
    if rec is None:
        return None
    ledger, _ = ledger_paths(repo_root)
    records = upsert(load_ledger(ledger), rec)
    _write_views(repo_root, records)
    return rec


def scan_outputs(repo_root: Path | str) -> int:
    """Record every ``outputs/*/evaluate`` run and refresh the README.

    Returns the number of runs recorded this pass.
    """
    repo_root = Path(repo_root)
    known = repo_known_experiments(repo_root)
    labels = repo_labels(repo_root)
    git = repo_git(repo_root)
    ledger, _ = ledger_paths(repo_root)
    records = load_ledger(ledger)
    n = 0
    for ev in sorted((repo_root / "outputs").glob("*/evaluate/metrics.json")):
        rec = extract_record(ev.parent.parent, known_experiments=known, labels=labels, git=git)
        if rec is not None:
            records = upsert(records, rec)
            n += 1
    _write_views(repo_root, records)
    return n
