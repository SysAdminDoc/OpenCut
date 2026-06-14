"""Registry + default coverage for the diarization engines.

pyannote speaker-diarization-community-1 (CC-BY-4.0, lower DER) is the new
default; legacy 3.1 is retained as the auto-fallback; NVIDIA Sortformer is an
optional NeMo-gated engine. These tests pin the default model, the fallback
constant, the registry wiring, the availability gating, and the model-download
entry without requiring the weights or an HF token.
"""

from __future__ import annotations


def test_default_diarization_model_is_community_1():
    from opencut.core.diarize import DEFAULT_DIARIZATION_MODEL, LEGACY_DIARIZATION_MODEL

    assert DEFAULT_DIARIZATION_MODEL == "pyannote/speaker-diarization-community-1"
    assert LEGACY_DIARIZATION_MODEL == "pyannote/speaker-diarization-3.1"


def test_diarize_config_defaults_to_community_1():
    from opencut.utils.config import DiarizeConfig

    assert DiarizeConfig().model == "pyannote/speaker-diarization-community-1"


def test_diarization_engines_registered():
    from opencut.core.engine_registry import get_registry

    reg = get_registry()
    names = {e.name for e in reg.get_engines("diarization")}
    assert "pyannote_community1" in names
    assert "pyannote_legacy" in names  # legacy retained as fallback
    assert "sortformer" in names       # optional


def test_community1_is_default_over_legacy():
    """community-1 outranks legacy 3.1 so it is the auto-selected default."""
    from opencut.core.engine_registry import get_registry

    reg = get_registry()
    community = reg.get_engine("diarization", "pyannote_community1")
    legacy = reg.get_engine("diarization", "pyannote_legacy")
    assert community is not None and legacy is not None
    assert community.priority > legacy.priority


def test_diarization_gated_by_availability_check():
    from opencut.checks import check_diarization_available
    from opencut.core.engine_registry import get_registry

    reg = get_registry()
    for name in ("pyannote_community1", "pyannote_legacy"):
        engine = reg.get_engine("diarization", name)
        assert engine is not None
        assert engine.is_available == check_diarization_available()


def test_sortformer_gated_by_nemo():
    from opencut.checks import check_sortformer_available
    from opencut.core.engine_registry import get_registry

    reg = get_registry()
    engine = reg.get_engine("diarization", "sortformer")
    assert engine is not None
    assert engine.is_available == check_sortformer_available()
    assert "optional" in engine.tags


def test_diarization_checks_return_bool():
    from opencut.checks import check_diarization_available, check_sortformer_available

    assert isinstance(check_diarization_available(), bool)
    assert isinstance(check_sortformer_available(), bool)


def test_community1_has_model_catalog_entry():
    from opencut.core.model_manager import KNOWN_MODELS

    assert "pyannote-community-1" in KNOWN_MODELS
    entry = KNOWN_MODELS["pyannote-community-1"]
    assert entry["category"] == "diarization"
    assert entry["url"].startswith("https://")
