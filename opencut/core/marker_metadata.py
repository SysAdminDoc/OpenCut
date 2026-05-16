"""Marker metadata round-trip schema (F103).

Round-tripping markers through OTIO, FCP XML, or EDL routinely loses
two pieces of metadata that editors care about:

* **Colour.** Each NLE has its own naming/spelling — Premiere ships
  with green/red/orange/yellow/purple/blue/cyan/white/rose, DaVinci
  uses an extra "Mint" / "Cream" palette, and OTIO's enum is yet a
  third superset. Without a canonical mapping a Premiere "Rose"
  marker comes back as DaVinci default-blue.
* **Source / comment.** Markers usually carry an "owner" string in
  Premiere ("notes from Joe") that lands in the EDL `* LOC` comment
  but gets dropped by the OTIO importer.

This module is the single source of truth for that mapping. It is
stdlib-only so anyone can call it from any context. The format
inter-conversions are exposed as pure functions; ``MarkerMetadata`` is
the canonical dataclass that callers build and consume.

The OTIO bridge (``opencut.export.otio_export``) and the marker
importer (``opencut.core.marker_import``) both use this module's
``normalise_color`` / ``denormalise_color`` to round-trip cleanly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional


# Canonical palette — the names used throughout OpenCut. Everything else
# normalises into this list. Order matches Premiere's marker palette so
# the most common cases land near the top.
CANONICAL_COLORS: tuple = (
    "green",
    "red",
    "orange",
    "yellow",
    "purple",
    "cyan",
    "blue",
    "white",
    "rose",
    "black",
)


_COLOR_ALIASES: Dict[str, str] = {
    # Premiere
    "premiere:green": "green",
    "premiere:red": "red",
    "premiere:orange": "orange",
    "premiere:yellow": "yellow",
    "premiere:purple": "purple",
    "premiere:cyan": "cyan",
    "premiere:blue": "blue",
    "premiere:white": "white",
    "premiere:rose": "rose",
    "premiere:pink": "rose",
    "premiere:magenta": "purple",
    # DaVinci Resolve
    "davinci:blue": "blue",
    "davinci:cyan": "cyan",
    "davinci:green": "green",
    "davinci:yellow": "yellow",
    "davinci:red": "red",
    "davinci:pink": "rose",
    "davinci:purple": "purple",
    "davinci:fuchsia": "purple",
    "davinci:rose": "rose",
    "davinci:lavender": "purple",
    "davinci:sky": "blue",
    "davinci:mint": "green",
    "davinci:lemon": "yellow",
    "davinci:sand": "orange",
    "davinci:cocoa": "orange",
    "davinci:cream": "white",
    # Avid Media Composer (colour wheel approximation)
    "avid:red": "red",
    "avid:green": "green",
    "avid:blue": "blue",
    "avid:yellow": "yellow",
    "avid:magenta": "purple",
    "avid:cyan": "cyan",
    "avid:black": "black",
    "avid:white": "white",
    # OTIO MarkerColor enum (uppercased in the schema)
    "otio:red": "red",
    "otio:green": "green",
    "otio:blue": "blue",
    "otio:yellow": "yellow",
    "otio:cyan": "cyan",
    "otio:magenta": "purple",
    "otio:pink": "rose",
    "otio:orange": "orange",
    "otio:purple": "purple",
    "otio:white": "white",
    "otio:black": "black",
}


# Reverse mappings — used when exporting to a specific NLE.
_REVERSE: Dict[str, Dict[str, str]] = {
    "premiere": {
        "green": "Green",
        "red": "Red",
        "orange": "Orange",
        "yellow": "Yellow",
        "purple": "Purple",
        "cyan": "Cyan",
        "blue": "Blue",
        "white": "White",
        "rose": "Rose",
        "black": "Black",
    },
    "davinci": {
        "green": "Green",
        "red": "Red",
        "orange": "Sand",
        "yellow": "Yellow",
        "purple": "Purple",
        "cyan": "Cyan",
        "blue": "Blue",
        "white": "Cream",
        "rose": "Pink",
        "black": "Black",
    },
    "avid": {
        "green": "Green",
        "red": "Red",
        "orange": "Yellow",
        "yellow": "Yellow",
        "purple": "Magenta",
        "cyan": "Cyan",
        "blue": "Blue",
        "white": "White",
        "rose": "Magenta",
        "black": "Black",
    },
    "otio": {
        "green": "GREEN",
        "red": "RED",
        "orange": "ORANGE",
        "yellow": "YELLOW",
        "purple": "PURPLE",
        "cyan": "CYAN",
        "blue": "BLUE",
        "white": "WHITE",
        "rose": "PINK",
        "black": "BLACK",
    },
}


def _alias_key(host: str, name: str) -> str:
    return f"{host.strip().lower()}:{(name or '').strip().lower()}"


def normalise_color(value: str, *, host: Optional[str] = None) -> str:
    """Return the canonical colour token for *value*.

    Accepts host-prefixed names (``"davinci:Mint"``) or bare names. Bare
    names fall through every known host's alias map before falling back
    to ``"green"`` so the function is forgiving.
    """
    if not value:
        return "green"
    raw = str(value).strip()
    if not raw:
        return "green"
    lower = raw.lower()
    if ":" in lower:
        return _COLOR_ALIASES.get(lower, "green")
    # If the caller supplied a host hint, prefer that.
    for host_key in (host, "premiere", "davinci", "avid", "otio"):
        if not host_key:
            continue
        canonical = _COLOR_ALIASES.get(_alias_key(host_key, lower))
        if canonical:
            return canonical
    # Direct canonical name.
    if lower in CANONICAL_COLORS:
        return lower
    return "green"


def denormalise_color(canonical: str, host: str) -> str:
    """Return the host-specific spelling for a canonical colour."""
    host_key = host.strip().lower()
    table = _REVERSE.get(host_key)
    if table is None:
        raise ValueError(f"unknown host {host!r}; choose from {sorted(_REVERSE)}")
    return table.get(normalise_color(canonical), table["green"])


def supported_hosts() -> List[str]:
    return sorted(_REVERSE)


@dataclass
class MarkerMetadata:
    """Canonical marker payload used for OTIO/FCP/EDL round-tripping."""

    name: str = "Marker"
    start_seconds: float = 0.0
    duration_seconds: float = 0.0
    color: str = "green"  # canonical
    source: str = ""       # "premiere" / "davinci" / "avid" / "otio" / "csv" / "edl"
    comment: str = ""
    chapter: bool = False
    metadata: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["color"] = normalise_color(self.color, host=self.source or None)
        return payload

    def for_host(self, host: str) -> dict:
        """Render in a host-friendly spelling (e.g. for EDL/FCP exports)."""
        payload = self.as_dict()
        payload["color"] = denormalise_color(payload["color"], host)
        payload["source"] = host
        return payload


def diff_marker_payloads(left: Iterable[dict], right: Iterable[dict]) -> List[str]:
    """Return a list of human-readable diffs between two marker lists.

    Empty when both lists describe the same canonical content. Used by
    the round-trip tests to assert metadata isn't lost in translation.
    """
    diffs: List[str] = []
    left_sorted = sorted(left, key=lambda m: (float(m.get("start_seconds", 0)), m.get("name", "")))
    right_sorted = sorted(right, key=lambda m: (float(m.get("start_seconds", 0)), m.get("name", "")))
    if len(left_sorted) != len(right_sorted):
        diffs.append(f"marker count differs ({len(left_sorted)} vs {len(right_sorted)})")
        return diffs
    for idx, (a, b) in enumerate(zip(left_sorted, right_sorted)):
        for key in ("name", "color", "comment", "chapter"):
            if (a.get(key) or "") != (b.get(key) or ""):
                diffs.append(f"marker {idx}: {key} differs ({a.get(key)!r} vs {b.get(key)!r})")
        if abs(float(a.get("start_seconds", 0)) - float(b.get("start_seconds", 0))) > 0.001:
            diffs.append(f"marker {idx}: start_seconds differs ({a.get('start_seconds')} vs {b.get('start_seconds')})")
    return diffs
