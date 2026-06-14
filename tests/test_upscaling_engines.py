"""Registry coverage for the SeedVR2 upscaling engine.

SeedVR2 (ByteDance, Apache-2.0) is a one-step diffusion VSR backend wired into
the upscaling engine registry. These tests pin the registry wiring, the
availability gating, the Real-ESRGAN fallback ordering, and the model-download
catalog entry without requiring the multi-GB weights.
"""

from __future__ import annotations


def test_seedvr2_registered_in_upscaling_domain():
    from opencut.core.engine_registry import get_registry

    reg = get_registry()
    names = {e.name for e in reg.get_engines("upscaling")}
    assert "seedvr2" in names
    # Real-ESRGAN must be retained as the fallback engine.
    assert "realesrgan" in names


def test_seedvr2_preferred_over_realesrgan():
    """SeedVR2 outranks Real-ESRGAN so it auto-selects when its weights exist."""
    from opencut.core.engine_registry import get_registry

    reg = get_registry()
    seedvr2 = reg.get_engine("upscaling", "seedvr2")
    realesrgan = reg.get_engine("upscaling", "realesrgan")
    assert seedvr2 is not None and realesrgan is not None
    assert seedvr2.priority > realesrgan.priority


def test_seedvr2_gated_by_availability_check():
    from opencut.checks import check_seedvr2_available
    from opencut.core.engine_registry import get_registry

    reg = get_registry()
    engine = reg.get_engine("upscaling", "seedvr2")
    assert engine is not None
    # Availability mirrors the diffusion-stack check — absent => unavailable,
    # never a crash.
    assert engine.is_available == check_seedvr2_available()


def test_seedvr2_check_handles_missing_dependency():
    from opencut.checks import check_seedvr2_available

    assert isinstance(check_seedvr2_available(), bool)


def test_seedvr2_has_model_catalog_entry():
    from opencut.core.model_manager import KNOWN_MODELS

    assert "seedvr2-3b" in KNOWN_MODELS
    entry = KNOWN_MODELS["seedvr2-3b"]
    assert entry["category"] == "upscaling"
    assert entry["size_mb"] > 0
    assert entry["url"].startswith("https://")


def test_seedvr2_upscale_raises_without_dependency():
    import pytest

    from opencut.core import upscale_seedvr2

    if upscale_seedvr2.check_seedvr2_available():
        # Installed: full diffusion forward is deferred -> NotImplementedError.
        with pytest.raises(NotImplementedError):
            upscale_seedvr2.upscale(input_path="x.mp4")
    else:
        # Absent: structured RuntimeError so the dispatcher falls back.
        with pytest.raises(RuntimeError):
            upscale_seedvr2.upscale(input_path="x.mp4")


def test_seedvr2_weights_are_apache_licensed():
    """License hygiene: SeedVR2 weights must be MIT-distribution-safe to default."""
    from opencut.core import upscale_seedvr2

    assert upscale_seedvr2.MODEL_LICENSE == "Apache-2.0"
