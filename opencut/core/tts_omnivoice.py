"""
OpenCut OmniVoice TTS (Wave H2.4, v1.25.0) — **STUB**

Zero-shot TTS with 600+ languages. New backend alongside F5-TTS /
Chatterbox for long-tail language coverage.
Source: https://github.com/k2-fsa/OmniVoice

Ships stubbed in v1.25.0; route returns 503 ``MISSING_DEPENDENCY`` until
``omnivoice`` is pip-installed and a voice reference file is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from opencut.helpers import _try_import


@dataclass
class OmniVoiceResult:
    output: str = ""
    language: str = ""
    duration: float = 0.0
    characters: int = 0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "language", "duration", "characters", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_omnivoice_available() -> bool:
    return _try_import("omnivoice") is not None and _try_import("torch") is not None


INSTALL_HINT = "pip install omnivoice torch"


SUPPORTED_LANGUAGES: List[str] = [
    # Shortlist — full 600+ list exposed at runtime once the backend loads.
    "en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko",
    "ar", "hi", "bn", "tr", "vi", "th", "pl", "nl", "sv", "fi",
]


def list_models() -> Dict[str, Any]:
    """Return the advertised model catalogue. Stub — returns the shortlist."""
    return {
        "backend": "omnivoice",
        "available": check_omnivoice_available(),
        "languages_advertised": len(SUPPORTED_LANGUAGES),
        "languages_supported": SUPPORTED_LANGUAGES,
        "install_hint": INSTALL_HINT,
        "note": "full 600+ language list will be returned at runtime once installed",
    }


def synthesize(
    text: str,
    reference_audio: str = "",
    language: str = "auto",
    output: Optional[str] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> OmniVoiceResult:
    """Entry-point stub. Raises RuntimeError until backend is installed."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    if not check_omnivoice_available():
        raise RuntimeError(
            "OmniVoice is not installed. " + INSTALL_HINT
        )
    raise NotImplementedError("OmniVoice wiring ships in a future release.")


__all__ = [
    "OmniVoiceResult",
    "SUPPORTED_LANGUAGES",
    "check_omnivoice_available",
    "INSTALL_HINT",
    "list_models",
    "synthesize",
]
