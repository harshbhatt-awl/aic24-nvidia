"""Tests for the single-source-of-truth stage registry (Phase 1)."""
import pytest

from aic24_nvidia import registry


def test_order_is_the_eight_stages_in_sequence():
    assert registry.order() == [
        "adapt", "frames", "detect", "reid", "pose", "sct", "mct", "evaluate",
    ]


def test_dir_name_maps_adapt_to_adapted_else_identity():
    assert registry.dir_name("adapt") == "adapted"
    for s in ["frames", "detect", "reid", "pose", "sct", "mct", "evaluate"]:
        assert registry.dir_name(s) == s


def test_upstream_chain_is_linear():
    assert registry.upstream_of("adapt") == ()
    assert registry.upstream_of("frames") == ("adapt",)
    assert registry.upstream_of("detect") == ("frames",)
    assert registry.upstream_of("reid") == ("detect",)
    assert registry.upstream_of("pose") == ("reid",)
    assert registry.upstream_of("sct") == ("pose",)
    assert registry.upstream_of("mct") == ("sct",)
    assert registry.upstream_of("evaluate") == ("mct",)


def test_by_name_returns_spec_with_callable_run():
    s = registry.by_name("detect")
    assert s.name == "detect"
    assert callable(s.run)


def test_validate_registry_accepts_the_real_registry():
    registry.validate_registry()  # must not raise


def test_validate_rejects_unknown_upstream():
    bad = (registry.StageSpec("a", "a", ("ghost",), lambda *a: None),)
    with pytest.raises(ValueError, match="unknown stage"):
        registry.validate_registry(bad)


def test_validate_rejects_upstream_that_does_not_precede():
    bad = (
        registry.StageSpec("a", "a", ("b",), lambda *a: None),
        registry.StageSpec("b", "b", (), lambda *a: None),
    )
    with pytest.raises(ValueError, match="precede"):
        registry.validate_registry(bad)


def test_validate_rejects_duplicate_names():
    bad = (
        registry.StageSpec("a", "a", (), lambda *a: None),
        registry.StageSpec("a", "a2", (), lambda *a: None),
    )
    with pytest.raises(ValueError, match="duplicate"):
        registry.validate_registry(bad)


def test_validate_rejects_duplicate_dir_names():
    bad = (
        registry.StageSpec("a", "shared", (), lambda *a: None),
        registry.StageSpec("b", "shared", ("a",), lambda *a: None),
    )
    with pytest.raises(ValueError, match="dir_name"):
        registry.validate_registry(bad)
