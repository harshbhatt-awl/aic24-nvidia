from __future__ import annotations
from datetime import datetime
from pathlib import Path


def make_run_id(config_filename: str, at: datetime | None = None) -> str:
    at = at or datetime.now()
    return f"{config_filename}_{at.strftime('%Y%m%d_%H%M')}"


def run_dir(outputs_root: Path, run_id: str) -> Path:
    return Path(outputs_root) / run_id


def stage_dir(run_dir_: Path, stage: str) -> Path:
    return Path(run_dir_) / stage


def stage_tmp_dir(run_dir_: Path, stage: str) -> Path:
    return Path(run_dir_) / f"{stage}.tmp"


def latest_run_id(outputs_root: Path, config_filename: str) -> str | None:
    root = Path(outputs_root)
    if not root.exists():
        return None
    prefix = f"{config_filename}_"
    candidates = sorted(
        (p.name for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)),
        reverse=True,
    )
    return candidates[0] if candidates else None
