"""Registry coverage for the LatentSync lip-sync engine.

LatentSync (ByteDance) is the dub pipeline's optional visual lip-sync final
stage. Its *code* is Apache-2.0 but the *checkpoint* licence is unconfirmed
(RESEARCH Open Question 3), so it ships OPT-IN: registered and selectable, but
never auto-selected over the always-available heuristic. These tests pin the
registry wiring, the opt-in ordering, the availability gating, the dub-pipeline
backend plumbing, and the model-download catalog entry.
"""

from __future__ import annotations


def test_latentsync_and_heuristic_registered_in_lip_sync_domain():
    from opencut.core.engine_registry import get_registry

    reg = get_registry()
    names = {e.name for e in reg.get_engines("lip_sync")}
    assert "latentsync" in names
    assert "mediapipe_jaw" in names


def test_latentsync_is_opt_in_not_auto_selected():
    """The heuristic outranks LatentSync so LatentSync is never auto-selected."""
    from opencut.core.engine_registry import get_registry

    reg = get_registry()
    latentsync = reg.get_engine("lip_sync", "latentsync")
    heuristic = reg.get_engine("lip_sync", "mediapipe_jaw")
    assert latentsync is not None and heuristic is not None
    assert heuristic.priority > latentsync.priority
    assert "opt-in" in latentsync.tags


def test_latentsync_gated_by_availability_check():
    from opencut.checks import check_latentsync_available
    from opencut.core.engine_registry import get_registry

    reg = get_registry()
    engine = reg.get_engine("lip_sync", "latentsync")
    assert engine is not None
    assert engine.is_available == check_latentsync_available()


def test_latentsync_check_handles_missing_dependency():
    from opencut.checks import check_latentsync_available

    assert isinstance(check_latentsync_available(), bool)


def test_latentsync_has_model_catalog_entry():
    from opencut.core.model_manager import KNOWN_MODELS

    assert "latentsync-1.6" in KNOWN_MODELS
    entry = KNOWN_MODELS["latentsync-1.6"]
    assert entry["category"] == "lip_sync"
    assert entry["size_mb"] > 0
    assert entry["url"].startswith("https://")


def test_latentsync_apply_raises_without_dependency():
    import pytest

    from opencut.core import lipsync_latentsync

    if lipsync_latentsync.check_latentsync_available():
        with pytest.raises(NotImplementedError):
            lipsync_latentsync.apply_latentsync(video_path="x.mp4")
    else:
        with pytest.raises(RuntimeError):
            lipsync_latentsync.apply_latentsync(video_path="x.mp4")


def test_latentsync_license_is_opt_in():
    """Code is Apache-2.0; checkpoint licence is unconfirmed -> opt-in."""
    from opencut.core import lipsync_latentsync

    assert lipsync_latentsync.MODEL_CODE_LICENSE == "Apache-2.0"
    assert lipsync_latentsync.DEFAULT_OPT_IN is True


def test_dub_config_lip_sync_backend_defaults_to_heuristic():
    from opencut.core.auto_dub_pipeline import DubConfig

    cfg = DubConfig()
    assert cfg.lip_sync_backend == "heuristic"
