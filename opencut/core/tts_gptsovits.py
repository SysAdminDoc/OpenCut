"""
OpenCut GPT-SoVITS TTS v1.28.0 — STUB

Voice-cloned speech synthesis via GPT-SoVITS server on port 9880.
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional


@dataclass
class GPTSoVITSResult:
    output: str = ""
    voice: str = ""
    text: str = ""
    language: str = "en"
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "voice", "text", "language", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_gptsovits_available() -> bool:
    try:
        resp = urllib.request.urlopen("http://localhost:9880", timeout=2)
        resp.close()
        return True
    except Exception:
        pass
    try:
        import importlib
        importlib.import_module("gpt_sovits")
        return True
    except ImportError:
        return False


INSTALL_HINT = (
    "Install GPT-SoVITS: https://github.com/RVC-Boss/GPT-SoVITS "
    "-- launch server on port 9880 before use"
)


def list_voices() -> List[str]:
    if not check_gptsovits_available():
        return []
    try:
        resp = urllib.request.urlopen("http://localhost:9880/voices", timeout=3)
        data = json.loads(resp.read())
        return list(data.get("voices", []))
    except Exception:
        return []


def synthesize(
    text: str,
    voice: str,
    language: str = "en",
    output: Optional[str] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> GPTSoVITSResult:
    if not check_gptsovits_available():
        raise RuntimeError(f"GPT-SoVITS is not installed or not running. Install with:\n    {INSTALL_HINT}")
    raise NotImplementedError("GPT-SoVITS wiring is not implemented yet. Track the live ROADMAP.md entry.")


__all__ = ["GPTSoVITSResult", "check_gptsovits_available", "INSTALL_HINT", "list_voices", "synthesize"]
