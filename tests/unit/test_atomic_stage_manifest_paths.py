import json
from aic24_nvidia.stages.base import atomic_stage


def test_manifest_paths_rewritten_from_tmp_to_final(tmp_path):
    run_dir = tmp_path
    with atomic_stage(run_dir, "detect", run_id="r1") as ctx:
        (ctx.work_dir / "cam.txt").write_text("x")
        ctx.set_outputs({"cam": str(ctx.work_dir / "cam.txt")})
        ctx.set_inputs({"prev": str(ctx.work_dir / "x.json")})

    final = run_dir / "detect"
    tmp = run_dir / "detect.tmp"
    assert final.exists()
    assert not tmp.exists()

    manifest = json.loads((final / "manifest.json").read_text())
    cam = manifest["outputs"]["cam"]
    prev = manifest["inputs"]["prev"]
    # paths point at the final dir, never the stale .tmp dir
    assert "detect.tmp" not in cam and cam.endswith("/detect/cam.txt")
    assert "detect.tmp" not in prev and prev.endswith("/detect/x.json")
