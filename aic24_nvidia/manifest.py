from __future__ import annotations
from dataclasses import dataclass, asdict, field
from pathlib import Path
import json
from typing import Literal


@dataclass
class Manifest:
    stage: str
    run_id: str
    started_at: str
    finished_at: str
    runtime_sec: float
    inputs: dict
    outputs: dict
    params: dict
    upstream_manifests: list[str] = field(default_factory=list)
    status: Literal["ok", "error"] = "ok"


def write_manifest(m: Manifest, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(m), f, indent=2, sort_keys=True)


def read_manifest(path: Path) -> Manifest:
    with open(path) as f:
        body = json.load(f)
    return Manifest(**body)


def gate(stage_dir: Path, upstream: list[Path], force: bool) -> Literal["run", "skip"]:
    """Decide whether to run this stage.

    - If own manifest exists and status==ok and not force -> "skip".
    - If any upstream manifest is missing -> raise.
    - If any upstream status != ok -> raise.
    - Else -> "run".
    """
    for up in upstream:
        up = Path(up)
        if not up.exists():
            raise RuntimeError(f"upstream manifest missing: {up}")
        m = read_manifest(up)
        if m.status != "ok":
            raise RuntimeError(f"upstream status not ok: {up} ({m.status})")

    own = Path(stage_dir) / "manifest.json"
    if own.exists() and not force:
        m = read_manifest(own)
        if m.status == "ok":
            return "skip"
    return "run"
