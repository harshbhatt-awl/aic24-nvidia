"""Characterization tests for each stage's external-symlink wiring.

The golden (link -> target) sets below are derived from the inline make_symlink
calls that lived in each stage's run() before the registry refactor. They are the
safety net: the wiring functions must reproduce exactly these links so the
refactor is behavior-preserving.
"""
from pathlib import Path
from types import SimpleNamespace

from aic24_nvidia.stages import adapt, extract_frames, detect, reid, pose, sct, mct
from aic24_nvidia.stages.base import atomic_stage

RUN = Path("/runs/r1")
EXT = Path("/ext")
YAC = EXT / "AIC24_Track1_YACHIYO_RIIPS"
CFG = SimpleNamespace(external_root=EXT, yachiyo_root=YAC)


def _links(mod, output_dir):
    return set(mod.WIRING(RUN, CFG, output_dir))


def test_adapt_exposes_external_original():
    out = RUN / "adapted"
    assert _links(adapt, out) == {(EXT / "Original", out / "Original")}


def test_frames_links_yachiyo_original_to_adapted_tree():
    out = RUN / "frames"  # output_dir is unused by frames (input-only wiring)
    assert _links(extract_frames, out) == {(YAC / "Original", RUN / "adapted" / "Original")}


def test_detect_exposes_external_detection():
    out = RUN / "detect"
    assert _links(detect, out) == {(EXT / "Detection", out)}


def test_reid_exposes_external_embedfeature():
    out = RUN / "reid"
    assert _links(reid, out) == {(EXT / "EmbedFeature", out)}


def test_pose_exposes_external_pose():
    out = RUN / "pose"
    assert _links(pose, out) == {(EXT / "Pose", out)}


def test_sct_wires_tracking_output_and_reads_reid_detect():
    out = RUN / "sct"
    assert _links(sct, out) == {
        (YAC / "Tracking", out),
        (YAC / "EmbedFeature", RUN / "reid"),
        (YAC / "Detection", RUN / "detect"),
    }


def test_mct_wires_tracking_output_and_reads_reid_detect_pose():
    out = RUN / "mct"
    assert _links(mct, out) == {
        (YAC / "Tracking", out),
        (YAC / "EmbedFeature", RUN / "reid"),
        (YAC / "Detection", RUN / "detect"),
        (YAC / "Pose", RUN / "pose"),
    }


def test_atomic_stage_applies_wiring_pre_run_then_post_promotion(tmp_path):
    seen = []

    def fake_wiring(run_dir, cfg, output_dir):
        seen.append(output_dir)
        return []

    cfg = SimpleNamespace(external_root=tmp_path, yachiyo_root=tmp_path)
    with atomic_stage(tmp_path, "detect", run_id="r", cfg=cfg, wiring=fake_wiring) as ctx:
        ctx.set_outputs({})

    assert seen == [tmp_path / "detect.tmp", tmp_path / "detect"]


def test_atomic_stage_wiring_symlink_points_at_final_after_promotion(tmp_path):
    link = tmp_path / "external" / "Detection"

    def wiring(run_dir, cfg, output_dir):
        return [(link, output_dir)]

    cfg = SimpleNamespace(external_root=tmp_path, yachiyo_root=tmp_path)
    with atomic_stage(tmp_path, "detect", run_id="r", cfg=cfg, wiring=wiring) as ctx:
        ctx.set_outputs({})

    assert link.is_symlink()
    assert link.resolve() == (tmp_path / "detect").resolve()
