import json
from pathlib import Path
import pytest
from aic24_nvidia.adapter.gt_converter import (
    convert_gt,
    reprojection_check,
    GtValidationResult,
)


def _make_nvidia_gt(path: Path) -> None:
    body = {
        "frames": [
            {
                "frame_id": 0,
                "objects": [
                    {
                        "id": 1,
                        "world_xyz": [10.0, 5.0, 0.0],
                        "cameras": {
                            "Camera_0001": {"bbox_xywh": [100, 200, 50, 100]},
                            "Camera_0002": {"bbox_xywh": [400, 250, 60, 110]},
                        },
                    },
                    {
                        "id": 2,
                        "world_xyz": [-3.0, 2.0, 0.0],
                        "cameras": {"Camera_0001": {"bbox_xywh": [600, 300, 55, 105]}},
                    },
                ],
            },
            {
                "frame_id": 1,
                "objects": [
                    {
                        "id": 1,
                        "world_xyz": [10.1, 5.1, 0.0],
                        "cameras": {
                            "Camera_0001": {"bbox_xywh": [101, 201, 50, 100]},
                        },
                    },
                ],
            },
        ]
    }
    path.write_text(json.dumps(body))


def test_convert_gt_per_camera(tmp_path: Path):
    src = tmp_path / "ground_truth.json"
    _make_nvidia_gt(src)
    out_dir = tmp_path / "adapted"
    scene_mapping = {"camera_0001": "Camera_0001", "camera_0002": "Camera_0002"}

    convert_gt(src, out_dir, scene="scene_001", scene_mapping=scene_mapping,
               frame_offset=0, max_frames=2)

    c1 = (out_dir / "Original" / "scene_001" / "camera_0001" / "gt" / "gt.txt").read_text().splitlines()
    assert c1 == [
        "1,1,100,200,50,100,1,1,1",
        "1,2,600,300,55,105,1,1,1",
        "2,1,101,201,50,100,1,1,1",
    ]
    c2 = (out_dir / "Original" / "scene_001" / "camera_0002" / "gt" / "gt.txt").read_text().splitlines()
    assert c2 == [
        "1,1,400,250,60,110,1,1,1",
    ]
    world = (out_dir / "scene_001_gt_world.txt").read_text().splitlines()
    assert world[0].startswith("1,1,10.0,5.0")


def test_convert_gt_respects_frame_offset(tmp_path: Path):
    src = tmp_path / "ground_truth.json"
    _make_nvidia_gt(src)
    out_dir = tmp_path / "adapted"
    scene_mapping = {"camera_0001": "Camera_0001"}

    convert_gt(src, out_dir, scene="scene_001", scene_mapping=scene_mapping,
               frame_offset=1, max_frames=1)

    c1 = (out_dir / "Original" / "scene_001" / "camera_0001" / "gt" / "gt.txt").read_text().splitlines()
    assert c1 == ["1,1,101,201,50,100,1,1,1"]


def test_reprojection_check_synthetic(tmp_path: Path):
    gt = {
        "frames": [{
            "frame_id": 0,
            "objects": [{
                "id": 1,
                "world_xyz": [100.0, 200.0, 0.0],
                "cameras": {"Camera_0001": {"bbox_xywh": [80, 180, 40, 40]}},
            }],
        }],
    }
    src = tmp_path / "gt.json"
    src.write_text(json.dumps(gt))

    def fake_project(world_xyz, cam_name):
        return (world_xyz[0], world_xyz[1])

    result = reprojection_check(src, scene_mapping={"camera_0001": "Camera_0001"},
                                project_fn=fake_project, eps_px=50.0)
    assert isinstance(result, GtValidationResult)
    assert result.total == 1
    assert result.matched == 1
    assert result.match_ratio == 1.0
