from __future__ import annotations
import json
import logging
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


def patch_scene_camera_map(cfg_path: Path, scene: str, camera_ids: list[int]) -> None:
    """Insert/replace the scene's camera_ids in upstream's scene_2_camera_id_file.json.

    The upstream schema is a LIST of `{"scene_name": str, "camera_ids": [int, ...]}`
    entries. We replace the matching entry's camera_ids (or append a new entry
    if the scene isn't listed). Also accepts the dict-of-lists schema for
    backward compatibility with older unit tests.
    """
    cfg_path = Path(cfg_path)
    if cfg_path.exists():
        body = json.loads(cfg_path.read_text())
    else:
        body = []

    if isinstance(body, list):
        replaced = False
        for entry in body:
            if isinstance(entry, dict) and entry.get("scene_name") == scene:
                entry["camera_ids"] = list(camera_ids)
                replaced = True
                break
        if not replaced:
            body.append({"scene_name": scene, "camera_ids": list(camera_ids)})
    elif isinstance(body, dict):
        body[scene] = list(camera_ids)
    else:
        raise ValueError(f"unrecognized scene_2_camera_id_file schema: {type(body)}")

    cfg_path.write_text(json.dumps(body, indent=2))


def copy_injected_files(src_dir: Path, dst_dir: Path, filenames: list[str]) -> None:
    """Copy named files from src_dir to dst_dir (creating dst_dir if needed)."""
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        shutil.copy2(src_dir / name, dst_dir / name)


def make_symlink(target: Path, link: Path) -> None:
    """Create or replace a symlink at `link` pointing to `target`."""
    link = Path(link)
    target = Path(target).resolve()
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        if link.is_symlink() or link.is_file():
            link.unlink()
        else:
            shutil.rmtree(link)
    link.symlink_to(target, target_is_directory=target.is_dir())


def ensure_dir_clean(path: Path) -> None:
    """Remove path whether it's a symlink, file, or directory. No-op if absent."""
    path = Path(path)
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def clone_if_missing(repo_url: str, dst: Path, ref: str | None = None) -> None:
    """Git-clone repo_url to dst if dst doesn't exist. Optionally checkout a ref."""
    dst = Path(dst)
    if dst.exists():
        log.info("already present: %s", dst)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", repo_url, str(dst)], check=True)
    if ref:
        subprocess.run(["git", "-C", str(dst), "checkout", ref], check=True)
