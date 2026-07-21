"""Hybrid ASR engine routing.

Routes each caption/transcription request to the best transcription engine for
the request's language: NVIDIA **Parakeet TDT 0.6B v3** for its supported set of
European languages (and **Canary 1B Flash** via an explicit override), and
**Whisper** (``large-v3-turbo`` by default) for the long tail Parakeet does not
cover. Auto-routing improves local throughput on supported languages without
sacrificing Whisper's broader coverage.

Selection is *honest*: an engine is only chosen when the shared engine registry
reports it available, which respects the implementation-state gate (a
terminal-``NotImplementedError`` stub adapter never resolves as active — see
``opencut/core/engine_registry.py`` and commit ``2c746b51``). While the NeMo
adapters remain stubs, every request therefore falls back to Whisper; once the
adapters and models are installed locally the same router transparently starts
sending supported languages to Parakeet.

The router is pure over the registry, so routing decisions and fallbacks are
fully unit-testable without a GPU, model download, or the Adobe host.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional

from opencut.core.captions import normalize_language_code

TRANSCRIPTION_DOMAIN = "transcription"

PARAKEET_ENGINE = "parakeet_tdt"
CANARY_ENGINE = "canary_1b_flash"
DEFAULT_WHISPER_ENGINE = "faster_whisper"
# Whisper's turbo checkpoint — fastest large-quality model, the sensible default
# for the long-tail branch when a Whisper backend is selected.
WHISPER_TURBO_MODEL = "large-v3-turbo"

# Engine names in the transcription domain that are Whisper-family backends and
# therefore accept the turbo model.
WHISPER_FAMILY = frozenset(
    {
        "faster_whisper",
        "crisper_whisper",
        "whisper_cpp",
        "whisperx",
        "openai_whisper",
    }
)

# Parakeet TDT 0.6B v3 — 25 European languages (NVIDIA model card).
PARAKEET_V3_LANGUAGES = frozenset(
    {
        "bg",  # Bulgarian
        "hr",  # Croatian
        "cs",  # Czech
        "da",  # Danish
        "nl",  # Dutch
        "en",  # English
        "et",  # Estonian
        "fi",  # Finnish
        "fr",  # French
        "de",  # German
        "el",  # Greek
        "hu",  # Hungarian
        "it",  # Italian
        "lv",  # Latvian
        "lt",  # Lithuanian
        "mt",  # Maltese
        "pl",  # Polish
        "pt",  # Portuguese
        "ro",  # Romanian
        "sk",  # Slovak
        "sl",  # Slovenian
        "es",  # Spanish
        "sv",  # Swedish
        "ru",  # Russian
        "uk",  # Ukrainian
    }
)

# Canary 1B Flash — transcription + X<->en translation for four languages.
CANARY_LANGUAGES = frozenset({"de", "en", "es", "fr"})


@dataclass
class ASRRoute:
    """The resolved routing decision for one transcription request."""

    engine: Optional[str]          # resolved transcription-domain engine (None = none available)
    language: Optional[str]        # normalized base language subtag, or None if unknown
    reason: str                    # human-readable explanation of the decision
    used_fallback: bool = False    # True when a preferred engine was wanted but unavailable
    requested: Optional[str] = None  # explicit override engine that was asked for, if any
    whisper_model: Optional[str] = None  # turbo model when the engine is Whisper-family
    available_engines: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def base_language(language: Optional[str]) -> str:
    """Return the normalized base language subtag (``en-US`` -> ``en``)."""
    return normalize_language_code(language)


def parakeet_supports_language(language: Optional[str]) -> bool:
    """True when Parakeet TDT v3 covers *language*."""
    code = base_language(language)
    return bool(code) and code in PARAKEET_V3_LANGUAGES


def canary_supports_language(language: Optional[str]) -> bool:
    """True when Canary covers *language*."""
    code = base_language(language)
    return bool(code) and code in CANARY_LANGUAGES


def _whisper_model_for(engine: Optional[str]) -> Optional[str]:
    return WHISPER_TURBO_MODEL if engine in WHISPER_FAMILY else None


def _get_registry(registry):
    if registry is not None:
        return registry
    from opencut.core.engine_registry import get_registry

    return get_registry()


def _available_engine_names(registry) -> List[str]:
    """Available transcription engines, honest about stubs, priority-ordered."""
    return [e.name for e in registry.get_available_engines(TRANSCRIPTION_DOMAIN)]


def _resolve_whisper(available: List[str]) -> Optional[str]:
    """Pick a Whisper-family engine from *available* (priority-ordered)."""
    if DEFAULT_WHISPER_ENGINE in available:
        return DEFAULT_WHISPER_ENGINE
    for name in available:
        if name in WHISPER_FAMILY:
            return name
    return None


def route_asr(
    language: Optional[str] = None,
    override: Optional[str] = None,
    registry=None,
) -> ASRRoute:
    """Resolve which transcription engine should handle *language*.

    Args:
        language: BCP-47/whisper language code or name (``"de"``, ``"en-US"``,
            ``"German"``). ``None``/unknown routes to Whisper for broad coverage.
        override: Explicit engine name to force. Honored only when that engine is
            actually available; otherwise the router falls back honestly and
            records the requested name and the fallback reason.
        registry: Engine registry to consult (defaults to the global singleton).

    Returns:
        An :class:`ASRRoute` describing the decision, fallbacks, and whether a
        Whisper turbo model applies.
    """
    reg = _get_registry(registry)
    available = _available_engine_names(reg)
    code = base_language(language) or None

    # 1. Explicit override wins — but only when it is genuinely runnable.
    if override:
        if override in available:
            return ASRRoute(
                engine=override,
                language=code,
                reason=f"explicit override -> {override}",
                requested=override,
                whisper_model=_whisper_model_for(override),
                available_engines=available,
            )
        # Requested but not runnable (stub/absent dep): fall through to auto,
        # remembering the request so the reason is honest.

    # 2. Auto: prefer Parakeet for its supported languages when it is runnable.
    wanted: Optional[str] = None
    if code and code in PARAKEET_V3_LANGUAGES:
        wanted = PARAKEET_ENGINE
        if wanted in available:
            return ASRRoute(
                engine=wanted,
                language=code,
                reason=f"{code} in Parakeet-v3 set -> {wanted}",
                requested=override,
                used_fallback=bool(override),
                available_engines=available,
            )

    # 3. Long tail (and honest fallback): Whisper turbo.
    whisper = _resolve_whisper(available)
    if whisper:
        if override:
            reason = f"requested {override!r} unavailable; fell back to {whisper}"
        elif wanted:
            reason = f"Parakeet unavailable (stub/deps); fell back to {whisper}"
        elif code:
            reason = f"{code} outside Parakeet set; using {whisper}"
        else:
            reason = f"language unknown; using {whisper} (broad coverage)"
        return ASRRoute(
            engine=whisper,
            language=code,
            reason=reason,
            requested=override,
            used_fallback=bool(override or wanted),
            whisper_model=WHISPER_TURBO_MODEL,
            available_engines=available,
        )

    # 4. Nothing runnable at all.
    return ASRRoute(
        engine=None,
        language=code,
        reason="no transcription engine available",
        requested=override,
        used_fallback=bool(override or wanted),
        available_engines=available,
    )


def asr_engine_readiness(registry=None) -> List[dict]:
    """Honest per-engine readiness for the transcription domain (CEP/UXP display).

    Each entry reports ``available``/``stub`` straight from the registry, so a
    terminal-stub adapter (Parakeet/Canary today) is surfaced as coming-soon,
    never as ready.
    """
    reg = _get_registry(registry)
    out: List[dict] = []
    for e in reg.get_engines(TRANSCRIPTION_DOMAIN):
        out.append(
            {
                "name": e.name,
                "display_name": e.display_name,
                "available": e.is_available,
                "stub": e.is_stub,
                "parakeet_languages": sorted(PARAKEET_V3_LANGUAGES)
                if e.name == PARAKEET_ENGINE
                else None,
            }
        )
    return out


__all__ = [
    "ASRRoute",
    "PARAKEET_V3_LANGUAGES",
    "CANARY_LANGUAGES",
    "WHISPER_TURBO_MODEL",
    "PARAKEET_ENGINE",
    "CANARY_ENGINE",
    "DEFAULT_WHISPER_ENGINE",
    "base_language",
    "parakeet_supports_language",
    "canary_supports_language",
    "route_asr",
    "asr_engine_readiness",
]
