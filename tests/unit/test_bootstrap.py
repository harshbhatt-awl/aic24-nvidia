import json
from pathlib import Path
import pytest
from aic24_nvidia.bootstrap import (
    patch_scene_camera_map,
    copy_injected_files,
    make_symlink,
    ensure_dir_clean,
)


def test_patch_scene_camera_map_adds_new_entry(tmp_path):
    cfg = tmp_path / "scene_2_camera_id_file.json"
    cfg.write_text(json.dumps({"scene_061": [1, 2, 3]}))
    patch_scene_camera_map(cfg, scene="scene_001", camera_ids=[101, 102])
    body = json.loads(cfg.read_text())
    assert body["scene_001"] == [101, 102]
    assert body["scene_061"] == [1, 2, 3]


def test_patch_scene_camera_map_replaces_existing(tmp_path):
    cfg = tmp_path / "scene_2_camera_id_file.json"
    cfg.write_text(json.dumps({"scene_001": [1]}))
    patch_scene_camera_map(cfg, scene="scene_001", camera_ids=[101, 102])
    body = json.loads(cfg.read_text())
    assert body["scene_001"] == [101, 102]


def test_copy_injected_files(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "x.py").write_text("hello")
    dst = tmp_path / "dst"
    copy_injected_files(src, dst, ["x.py"])
    assert (dst / "x.py").read_text() == "hello"


def test_make_symlink_replaces_existing(tmp_path):
    target1 = tmp_path / "t1"
    target1.mkdir()
    target2 = tmp_path / "t2"
    target2.mkdir()
    link = tmp_path / "link"
    make_symlink(target1, link)
    assert link.resolve() == target1.resolve()
    make_symlink(target2, link)
    assert link.resolve() == target2.resolve()


def test_ensure_dir_clean_removes_symlink(tmp_path):
    target = tmp_path / "t"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target)
    ensure_dir_clean(link)
    assert not link.exists()


def test_ensure_dir_clean_removes_directory(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    (d / "file.txt").write_text("x")
    ensure_dir_clean(d)
    assert not d.exists()
