"""
Voice-command grammar for hands-free timeline editing.

Parses a single-phrase utterance (typically from WhisperX / faster-
whisper) into a structured timeline action: cut here, slip 4 frames,
mark, undo, speed up 1.5x, etc.  Designed for the narrow phrasebook
that's actually useful mid-edit — broader natural-language queries
still route through :mod:`opencut.core.nlp_command`.

Design notes
------------
- **Deterministic grammar** (not LLM) so response latency is a
  single-digit ms even on a Raspberry Pi. Predictable outputs and
  zero API cost.
- **Frames + seconds + beats** all parse — the caller picks the
  target sequence fps / bpm and the grammar normalises to seconds.
- **Units are Whitman-style flexible**: "slip three frames", "slip
  3 frames", "slip three f" all map to the same action.
- **Numeric words 0..99** parsed inline — no dependency on ``word2number``.

Grammar
-------
The grammar is intentionally small.  Each action is a verb plus
optional magnitude + unit + direction. Supported verbs:

    cut | trim | split — "cut here", "split at playhead"
    mark | marker      — "mark", "drop a marker", "mark scene start"
    slip | slide       — "slip 4 frames left", "slide 2 seconds"
    nudge              — "nudge 3 frames right"
    speed | rate       — "speed up 1.5x", "half speed"
    undo | redo        — "undo", "redo"
    go | seek          — "go to 00:01:30", "seek to 90 seconds"
    mute | unmute      — "mute audio", "unmute track 2"
    ripple             — "ripple delete"
    zoom               — "zoom in", "zoom out", "zoom to 50%"

Output
------
:func:`parse` returns a :class:`VoiceAction` with ``verb``, ``unit``,
``magnitude_seconds``, ``direction`` (``"left"`` / ``"right"`` / ``None``),
``raw_number`` (the number as spoken), ``confidence`` (0..1), and
``fallback_route`` (set when the grammar can't parse and the caller
should fall through to ``nlp_command.parse_command``).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Verb table
# ---------------------------------------------------------------------------

# verb -> (canonical verb, requires_magnitude)
VERBS: Dict[str, "_VerbSpec"] = {}


@dataclass
class _VerbSpec:
    canonical: str
    aliases: List[str]
    requires_magnitude: bool = False
    accepts_direction: bool = False
    accepts_target_number: bool = False


def _register(spec: _VerbSpec) -> None:
    for alias in spec.aliases:
        VERBS[alias.lower()] = spec


_register(_VerbSpec("cut",       ["cut", "split", "razor", "make cut"]))
_register(_VerbSpec("trim",      ["trim"],                 requires_magnitude=True,
                                  accepts_direction=True))
_register(_VerbSpec("mark",      ["mark", "marker", "drop marker", "add marker"]))
_register(_VerbSpec("slip",      ["slip", "slide"],        requires_magnitude=True,
                                  accepts_direction=True))
_register(_VerbSpec("nudge",     ["nudge", "bump"],        requires_magnitude=True,
                                  accepts_direction=True))
_register(_VerbSpec("speed",     ["speed up", "speed"],    requires_magnitude=True))
_register(_VerbSpec("slow",      ["slow down", "half speed", "slow"], requires_magnitude=False))
_register(_VerbSpec("undo",      ["undo", "revert"]))
_register(_VerbSpec("redo",      ["redo"]))
_register(_VerbSpec("seek",      ["go to", "seek", "seek to", "jump to"],
                                  accepts_target_number=True))
_register(_VerbSpec("mute",      ["mute"]))
_register(_VerbSpec("unmute",    ["unmute"]))
_register(_VerbSpec("ripple",    ["ripple delete", "ripple"]))
_register(_VerbSpec("zoom_in",   ["zoom in"]))
_register(_VerbSpec("zoom_out",  ["zoom out"]))


# Number-words for 0..99 — cheap and predictable vs a NLP library.
_NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "half": 0.5, "quarter": 0.25,
}

_UNIT_ALIASES = {
    "frame": "frames", "frames": "frames", "f": "frames",
    "second": "seconds", "seconds": "seconds", "sec": "seconds", "secs": "seconds",
    "s": "seconds",
    "minute": "minutes", "minutes": "minutes", "min": "minutes", "mins": "minutes",
    "m": "minutes",
    "beat": "beats", "beats": "beats",
    "percent": "percent", "%": "percent", "pct": "percent",
    "x": "ratio", "times": "ratio",
}

_DIR_ALIASES = {
    "left": "left", "back": "left", "backward": "left", "backwards": "left",
    "right": "right", "forward": "right", "forwards": "right", "ahead": "right",
}


@dataclass
class VoiceAction:
    """Structured result of :func:`parse`."""
    verb: str = ""
    unit: Optional[str] = None
    magnitude_seconds: Optional[float] = None
    raw_number: Optional[float] = None
    direction: Optional[str] = None
    confidence: float = 0.0
    matched_text: str = ""
    fallback_route: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, key):
        return getattr(self, key)

    def __contains__(self, key):
        return key in self.__dataclass_fields__

    def keys(self):
        return self.__dataclass_fields__.keys()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TIMECODE_RE = re.compile(
    r"(?P<hms>(?:(?P<h>\d{1,2}):)?(?P<mm>\d{1,2}):(?P<ss>\d{2})(?:\.(?P<ms>\d{1,3}))?)"
)


def _parse_number(text: str) -> Optional[float]:
    """Extract the first number from ``text`` — digit form or word form.

    Handles: ``"3"``, ``"3.5"``, ``"three"``, ``"three and a half"``,
    ``"twenty five"``.  Returns ``None`` when no number is present.
    """
    # Direct float / int
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            pass

    # Word-based: try compound first (twenty one, twenty-five, etc.)
    tokens = re.findall(r"[A-Za-z]+", text.lower())
    total: Optional[float] = None
    for tok in tokens:
        if tok in _NUM_WORDS:
            val = _NUM_WORDS[tok]
            total = val if total is None else total + val
        elif total is not None and tok in ("and", "a"):
            continue
        elif total is not None:
            break
    return total


def _parse_unit(text: str) -> Optional[str]:
    """Find the first known unit in ``text``."""
    tokens = re.findall(r"[A-Za-z%]+", text.lower())
    # Two-pass: prefer explicit units ("percent") over ambiguous ones ("m")
    for unambiguous in ("frames", "seconds", "minutes", "beats",
                        "percent", "%", "x", "times"):
        if unambiguous in tokens or unambiguous in text.lower():
            return _UNIT_ALIASES.get(unambiguous, unambiguous)
    for tok in tokens:
        if tok in _UNIT_ALIASES:
            return _UNIT_ALIASES[tok]
    return None


def _parse_direction(text: str) -> Optional[str]:
    tokens = re.findall(r"[A-Za-z]+", text.lower())
    for tok in tokens:
        if tok in _DIR_ALIASES:
            return _DIR_ALIASES[tok]
    return None


def _parse_timecode(text: str) -> Optional[float]:
    """Match ``"00:01:30"`` / ``"1:30"`` / ``"90.5"`` into seconds."""
    m = _TIMECODE_RE.search(text)
    if not m:
        return None
    mm = int(m.group("mm") or 0)
    ss = int(m.group("ss") or 0)
    h = int(m.group("h") or 0)
    ms = m.group("ms")
    total = h * 3600 + mm * 60 + ss
    if ms:
        # 1 / 10 / 100 / 1000 ms
        total += int(ms) / (10 ** len(ms))
    return float(total)


def _to_seconds(number: Optional[float], unit: Optional[str], fps: float, bpm: float) -> Optional[float]:
    """Normalise a (number, unit) pair to seconds for a given sequence."""
    if number is None:
        return None
    if unit is None:
        # Assume seconds when unit absent
        return float(number)
    if unit == "frames":
        if fps <= 0:
            return None
        return float(number) / fps
    if unit == "seconds":
        return float(number)
    if unit == "minutes":
        return float(number) * 60.0
    if unit == "beats":
        if bpm <= 0:
            return None
        return float(number) * (60.0 / bpm)
    if unit in ("percent", "ratio"):
        # Not convertible to seconds — caller uses raw_number
        return None
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse(
    utterance: str,
    fps: float = 30.0,
    bpm: float = 120.0,
) -> VoiceAction:
    """Parse a single voice utterance into a :class:`VoiceAction`.

    Never raises — returns an empty-verb ``VoiceAction`` with a
    ``fallback_route`` hint when the grammar can't match, so callers
    can route to ``nlp_command.parse_command`` for broader intent
    detection.

    Args:
        utterance: Transcribed text (WhisperX / faster-whisper output).
        fps: Sequence frame rate used to convert ``frames`` → seconds.
        bpm: Musical BPM used to convert ``beats`` → seconds.
    """
    if not utterance or not utterance.strip():
        return VoiceAction(fallback_route="/nlp/command")

    text = utterance.strip().lower().rstrip(".!?,")
    # Collapse consecutive whitespace
    text = re.sub(r"\s+", " ", text)

    # Find the longest matching verb alias in the text
    matched_spec: Optional[_VerbSpec] = None
    matched_alias = ""
    for alias, spec in sorted(VERBS.items(), key=lambda kv: -len(kv[0])):
        if alias == text or text.startswith(alias + " ") or f" {alias} " in f" {text} " or text.endswith(" " + alias):
            matched_spec = spec
            matched_alias = alias
            break

    if matched_spec is None:
        return VoiceAction(
            confidence=0.0,
            matched_text=utterance,
            fallback_route="/nlp/command",
            notes=["no verb matched; route to broader NLP parser"],
        )

    # Strip the verb from the payload so number / unit parsing isn't
    # confused by the verb token itself.
    payload = text.replace(matched_alias, "", 1).strip()

    # Timecode has priority for seek-like verbs
    target_seconds: Optional[float] = None
    if matched_spec.accepts_target_number and _parse_timecode(payload):
        target_seconds = _parse_timecode(payload)

    number = _parse_number(payload) if target_seconds is None else target_seconds
    unit = _parse_unit(payload)
    direction = _parse_direction(payload) if matched_spec.accepts_direction else None

    magnitude_seconds = _to_seconds(number, unit, fps, bpm)

    # Derive confidence:
    # 1.0  — verb + number + unit + direction (everything resolved)
    # 0.8  — verb + number (unit assumed seconds)
    # 0.7  — verb + unit w/o number (ambiguous magnitude)
    # 0.6  — verb only
    confidence = 0.6
    if matched_spec.requires_magnitude and number is None:
        # Required magnitude missing — low confidence, hint to retry
        confidence = 0.3
    elif number is not None and unit is not None:
        confidence = 1.0 if (direction or not matched_spec.accepts_direction) else 0.9
    elif number is not None:
        confidence = 0.8

    return VoiceAction(
        verb=matched_spec.canonical,
        unit=unit,
        magnitude_seconds=(
            round(magnitude_seconds, 4) if magnitude_seconds is not None else None
        ),
        raw_number=number,
        direction=direction,
        confidence=round(confidence, 2),
        matched_text=utterance,
        fallback_route=None if confidence >= 0.6 else "/nlp/command",
        notes=[],
    )


def list_grammar() -> List[Dict]:
    """UI helper — return the full verb catalogue for command-palette display."""
    seen_canonical = set()
    out: List[Dict] = []
    for spec in VERBS.values():
        if spec.canonical in seen_canonical:
            continue
        seen_canonical.add(spec.canonical)
        out.append({
            "verb": spec.canonical,
            "aliases": list(spec.aliases),
            "requires_magnitude": spec.requires_magnitude,
            "accepts_direction": spec.accepts_direction,
            "accepts_target_number": spec.accepts_target_number,
        })
    return sorted(out, key=lambda r: r["verb"])
