"""Registry coverage for the NeMo ASR transcription engines (Parakeet / Canary).

Both engines wrap NVIDIA NeMo stubs that activate when ``nemo`` is installed
locally. These tests pin the registry wiring, the availability gating, and the
model-download catalog entries without requiring the multi-GB weights.
"""

from __future__ import annotations


def test_parakeet_and_canary_registered_in_transcription_domain():
    from opencut.core.engine_registry import get_registry

    reg = get_registry()
    names = {e.name for e in reg.get_engines("transcription")}
    assert "parakeet_tdt" in names
    assert "canary_1b_flash" in names
    # Existing Whisper engines must be retained as fallbacks.
    assert "faster_whisper" in names
    assert "crisper_whisper" in names


def test_nemo_engines_gated_by_availability_check():
    from opencut.checks import check_nemo_asr_available
    from opencut.core.engine_registry import get_registry

    reg = get_registry()
    for name in ("parakeet_tdt", "canary_1b_flash"):
        engine = reg.get_engine("transcription", name)
        assert engine is not None
        # The engine's availability mirrors the NeMo check — when NeMo is
        # absent it must report unavailable rather than crash.
        assert engine.is_available == check_nemo_asr_available()


def test_nemo_check_handles_missing_dependency():
    from opencut.checks import check_nemo_asr_available

    # Must return a bool and never raise, regardless of install state.
    assert isinstance(check_nemo_asr_available(), bool)


def test_nemo_engine_modules_accept_the_installed_import_namespace(monkeypatch):
    from opencut.core import asr_canary, asr_parakeet

    def import_probe(name):
        return object() if name == "nemo" else None

    monkeypatch.setattr(asr_canary, "_try_import", import_probe)
    monkeypatch.setattr(asr_parakeet, "_try_import", import_probe)

    assert asr_canary.check_nemo_toolkit_available() is True
    assert asr_parakeet.check_nemo_toolkit_available() is True


def test_parakeet_and_canary_have_model_catalog_entries():
    from opencut.core.model_manager import KNOWN_MODELS

    for key in ("parakeet-tdt-0.6b-v3", "canary-1b-flash"):
        assert key in KNOWN_MODELS, f"{key} missing from KNOWN_MODELS"
        entry = KNOWN_MODELS[key]
        assert entry["category"] == "transcription"
        assert entry["size_mb"] > 0
        assert entry["url"].startswith("https://")


def test_nemo_asr_stubs_raise_without_dependency():
    import pytest

    from opencut.core import asr_canary, asr_parakeet

    if asr_parakeet.check_nemo_toolkit_available():
        pytest.skip("nemo installed; stub raises NotImplementedError instead")
    with pytest.raises(RuntimeError):
        asr_parakeet.transcribe(filepath="x.wav")
    with pytest.raises(RuntimeError):
        asr_canary.transcribe_batch(filepath="x.wav")
