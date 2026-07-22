"""Hybrid ASR routing (opencut.core.asr_router).

Parakeet TDT v3 handles its 25-language European set; Whisper turbo covers the
long tail. Selection is honest — an engine is chosen only when the registry
reports it available, which respects the implementation-state gate, so a
runtime-unavailable Parakeet adapter falls back to Whisper.

The registry is constructed explicitly per test so routing decisions are
deterministic without any real model, GPU, or Adobe host.
"""

from opencut.core import asr_router as ar
from opencut.core.engine_registry import EngineInfo, EngineRegistry


def _registry(*engines: EngineInfo) -> EngineRegistry:
    reg = EngineRegistry()
    for e in engines:
        reg.register(e)
    return reg


def _engine(name, *, available=True, impl_module=None, priority=50):
    return EngineInfo(
        name=name,
        domain=ar.TRANSCRIPTION_DOMAIN,
        display_name=name,
        description=name,
        check_fn=(lambda: available),
        priority=priority,
        impl_module=impl_module,
    )


# --- language matrix ------------------------------------------------------

def test_parakeet_language_matrix():
    assert ar.parakeet_supports_language("de")
    assert ar.parakeet_supports_language("en-US")  # region stripped
    assert ar.parakeet_supports_language("EN")  # case-insensitive
    assert not ar.parakeet_supports_language("ja")  # Japanese not covered
    assert not ar.parakeet_supports_language(None)


def test_canary_language_matrix():
    assert ar.canary_supports_language("fr")
    assert not ar.canary_supports_language("ru")  # Parakeet-only, not Canary


# --- auto routing ---------------------------------------------------------

def test_supported_language_routes_to_parakeet_when_available():
    reg = _registry(_engine("parakeet_tdt", priority=85), _engine("faster_whisper", priority=70))
    route = ar.route_asr("de", registry=reg)
    assert route.engine == "parakeet_tdt"
    assert route.used_fallback is False
    assert route.whisper_model is None


def test_unsupported_language_routes_to_whisper_turbo():
    reg = _registry(_engine("parakeet_tdt", priority=85), _engine("faster_whisper", priority=70))
    route = ar.route_asr("ja", registry=reg)
    assert route.engine == "faster_whisper"
    assert route.whisper_model == ar.WHISPER_TURBO_MODEL
    assert route.used_fallback is False


def test_unknown_language_uses_whisper_broad_coverage():
    reg = _registry(_engine("faster_whisper"))
    route = ar.route_asr(None, registry=reg)
    assert route.engine == "faster_whisper"
    assert "broad coverage" in route.reason


# --- honest availability gate ---------------------------------------------

def test_unavailable_parakeet_falls_back_to_whisper():
    reg = _registry(
        _engine("parakeet_tdt", available=False, impl_module="asr_parakeet", priority=85),
        _engine("faster_whisper", priority=70),
    )
    route = ar.route_asr("de", registry=reg)
    assert route.engine == "faster_whisper"
    assert route.used_fallback is True
    assert "Parakeet unavailable" in route.reason


def test_no_engine_available_returns_none():
    reg = _registry(_engine("parakeet_tdt", available=False, impl_module="asr_parakeet"))
    route = ar.route_asr("de", registry=reg)
    assert route.engine is None
    assert route.used_fallback is True
    assert "no transcription engine" in route.reason


# --- explicit override ----------------------------------------------------

def test_override_honored_when_available():
    reg = _registry(_engine("canary_1b_flash", priority=80), _engine("faster_whisper", priority=70))
    route = ar.route_asr("en", override="canary_1b_flash", registry=reg)
    assert route.engine == "canary_1b_flash"
    assert route.requested == "canary_1b_flash"
    assert route.used_fallback is False


def test_override_falls_back_honestly_when_unavailable():
    reg = _registry(
        _engine("canary_1b_flash", available=False, impl_module="asr_canary", priority=80),
        _engine("faster_whisper", priority=70),
    )
    route = ar.route_asr("en", override="canary_1b_flash", registry=reg)
    assert route.engine == "faster_whisper"
    assert route.requested == "canary_1b_flash"
    assert route.used_fallback is True
    assert "unavailable" in route.reason


def test_override_to_whisper_carries_turbo_model():
    reg = _registry(_engine("faster_whisper"), _engine("parakeet_tdt", priority=85))
    route = ar.route_asr("de", override="faster_whisper", registry=reg)
    assert route.engine == "faster_whisper"
    assert route.whisper_model == ar.WHISPER_TURBO_MODEL


# --- readiness surface ----------------------------------------------------

def test_asr_engine_readiness_separates_implementation_and_runtime(monkeypatch):
    monkeypatch.setattr(
        "opencut.core.asr_nemo.nemo_runtime_status",
        lambda: {
            "reason": "Linux runtime dependency missing",
            "version": "unavailable",
            "supported_platforms": ["linux"],
        },
    )
    reg = _registry(
        _engine("parakeet_tdt", available=False, impl_module="asr_parakeet"),
        _engine("faster_whisper"),
    )
    readiness = {r["name"]: r for r in ar.asr_engine_readiness(registry=reg)}
    assert readiness["parakeet_tdt"]["available"] is False
    assert readiness["parakeet_tdt"]["stub"] is False
    assert readiness["parakeet_tdt"]["parakeet_languages"]  # language list surfaced
    assert readiness["parakeet_tdt"]["runtime_reason"] == "Linux runtime dependency missing"
    assert readiness["parakeet_tdt"]["supported_platforms"] == ["linux"]
    assert readiness["faster_whisper"]["available"] is True


def test_route_to_dict_is_serializable():
    reg = _registry(_engine("faster_whisper"))
    payload = ar.route_asr("en", registry=reg).to_dict()
    assert payload["engine"] == "faster_whisper"
    assert set(payload) >= {"engine", "language", "reason", "used_fallback", "available_engines"}


def test_default_registry_is_honest_without_nemo(monkeypatch):
    from opencut.core.engine_registry import get_registry

    reg = get_registry()
    for name in (ar.PARAKEET_ENGINE, ar.CANARY_ENGINE):
        engine = reg.get_engine(ar.TRANSCRIPTION_DOMAIN, name)
        monkeypatch.setattr(engine, "check_fn", lambda: False)
    reg.clear_cache()
    route = ar.route_asr("de", registry=reg)
    assert route.engine != "parakeet_tdt"
    assert route.engine != "canary_1b_flash"
