"""Registry coverage for the NeMo ASR transcription engines."""

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


def test_nemo_engines_are_implemented_and_gated_by_runtime():
    from opencut.core.engine_registry import get_registry

    reg = get_registry()
    for name in ("parakeet_tdt", "canary_1b_flash"):
        engine = reg.get_engine("transcription", name)
        assert engine is not None
        assert engine.is_stub is False
        assert engine.is_available == engine.check_fn()


def test_remaining_stub_engines_never_resolve_as_active():
    from opencut.core.engine_registry import get_registry

    reg = get_registry()
    reg.clear_cache()
    for domain, stub_names in (
        ("diarization", {"sortformer"}),
        ("upscaling", {"seedvr2"}),
        ("lip_sync", {"latentsync"}),
    ):
        available = {e.name for e in reg.get_available_engines(domain)}
        assert not (available & stub_names), (domain, available & stub_names)
        resolved = reg.resolve_engine(domain)
        if resolved is not None:
            assert resolved.name not in stub_names, (domain, resolved.name)


def test_nemo_check_handles_missing_dependency():
    from opencut.checks import check_nemo_asr_available

    # Must return a bool and never raise, regardless of install state.
    assert isinstance(check_nemo_asr_available(), bool)


def test_nemo_runtime_is_linux_only():
    from opencut.core.asr_nemo import nemo_runtime_status

    windows = nemo_runtime_status("win32")
    assert windows["available"] is False
    assert windows["platform_supported"] is False
    assert windows["supported_platforms"] == ["linux"]


def test_parakeet_and_canary_have_model_catalog_entries():
    from opencut.core.model_manager import KNOWN_MODELS

    for key in ("parakeet-tdt-0.6b-v3", "canary-1b-flash"):
        assert key in KNOWN_MODELS, f"{key} missing from KNOWN_MODELS"
        entry = KNOWN_MODELS[key]
        assert entry["category"] == "transcription"
        assert entry["size_mb"] > 0
        assert entry["url"].startswith("https://")
        assert "/resolve/main/" not in entry["url"]
        assert len(entry["sha256"]) == 64


def test_nemo_asr_adapters_validate_input_before_runtime():
    import pytest

    from opencut.core import asr_canary, asr_parakeet

    with pytest.raises(FileNotFoundError):
        asr_parakeet.transcribe(filepath="x.wav")
    with pytest.raises(FileNotFoundError):
        asr_canary.transcribe_batch(filepath="x.wav")
