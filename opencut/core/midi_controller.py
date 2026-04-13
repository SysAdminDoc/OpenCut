"""
OpenCut MIDI Controller Mapping Module

Map MIDI CC (continuous controller) messages and note events to
OpenCut parameters and operation triggers.  Mappings are stored as
JSON files that can be loaded by a MIDI listener front-end.

No direct MIDI I/O is performed here — this module manages the
mapping configuration layer.  The front-end (Electron / browser)
uses Web MIDI API or a system MIDI bridge to dispatch events.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("opencut")

_OPENCUT_DIR = os.path.join(os.path.expanduser("~"), ".opencut")
_MIDI_DIR = os.path.join(_OPENCUT_DIR, "midi_maps")


@dataclass
class MidiMapping:
    """A single MIDI CC or note mapping."""
    midi_type: str = "cc"          # "cc" or "note"
    channel: int = 0               # MIDI channel 0-15
    cc_number: int = 0             # CC number or note number
    parameter: str = ""            # OpenCut parameter name
    min_value: float = 0.0         # Parameter minimum
    max_value: float = 1.0         # Parameter maximum
    invert: bool = False           # Invert MIDI value mapping
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MidiMapping":
        return cls(
            midi_type=str(d.get("midi_type", "cc")),
            channel=int(d.get("channel", 0)),
            cc_number=int(d.get("cc_number", 0)),
            parameter=str(d.get("parameter", "")),
            min_value=float(d.get("min_value", 0.0)),
            max_value=float(d.get("max_value", 1.0)),
            invert=bool(d.get("invert", False)),
            description=str(d.get("description", "")),
        )


@dataclass
class MidiMap:
    """A complete MIDI controller mapping set."""
    name: str = ""
    description: str = ""
    device_name: str = ""
    mappings: List[MidiMapping] = field(default_factory=list)
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "device_name": self.device_name,
            "mappings": [m.to_dict() for m in self.mappings],
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MidiMap":
        mappings = [MidiMapping.from_dict(m) for m in d.get("mappings", [])]
        return cls(
            name=d.get("name", ""),
            description=d.get("description", ""),
            device_name=d.get("device_name", ""),
            mappings=mappings,
            version=d.get("version", 1),
        )


# ---------------------------------------------------------------------------
# Default device templates
# ---------------------------------------------------------------------------

_DEFAULT_DEVICES = [
    {"name": "Generic MIDI Controller", "type": "generic"},
    {"name": "Akai APC Mini", "type": "akai_apc_mini"},
    {"name": "Novation Launch Control", "type": "novation_launch_control"},
    {"name": "Korg nanoKONTROL2", "type": "korg_nanokontrol2"},
    {"name": "Behringer X-Touch Mini", "type": "behringer_xtouch_mini"},
    {"name": "Arturia MiniLab", "type": "arturia_minilab"},
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_midi_mapping(
    cc_channel: int,
    parameter: str,
    range_config: Optional[Dict[str, Any]] = None,
    midi_type: str = "cc",
    cc_number: int = 0,
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Create a single MIDI CC/note to parameter mapping.

    Args:
        cc_channel: MIDI channel (0-15).
        parameter: OpenCut parameter name to control.
        range_config: Optional dict with ``min``, ``max``, ``invert`` keys.
        midi_type: ``"cc"`` for continuous controller, ``"note"`` for note trigger.
        cc_number: CC number or MIDI note number.
        on_progress: Optional progress callback.

    Returns:
        Mapping configuration dict.
    """
    if on_progress:
        on_progress(50, "Creating MIDI mapping...")

    range_config = range_config or {}

    mapping = MidiMapping(
        midi_type=midi_type,
        channel=max(0, min(15, cc_channel)),
        cc_number=max(0, min(127, cc_number)),
        parameter=parameter,
        min_value=float(range_config.get("min", 0.0)),
        max_value=float(range_config.get("max", 1.0)),
        invert=bool(range_config.get("invert", False)),
        description=f"{parameter} via MIDI {midi_type.upper()} ch{cc_channel}",
    )

    if on_progress:
        on_progress(100, "MIDI mapping created")

    return mapping.to_dict()


def save_midi_map(
    mappings: List[Dict[str, Any]],
    output_path: str,
    name: str = "",
    device_name: str = "",
    on_progress: Optional[Callable] = None,
) -> str:
    """Save a set of MIDI mappings to a JSON file.

    Args:
        mappings: List of mapping dicts.
        output_path: Destination file path.
        name: Map name.
        device_name: Target MIDI device name.
        on_progress: Optional progress callback.

    Returns:
        Path where the map was saved.
    """
    if on_progress:
        on_progress(30, "Saving MIDI map...")

    midi_map = MidiMap(
        name=name or "Custom MIDI Map",
        device_name=device_name,
        mappings=[MidiMapping.from_dict(m) for m in mappings],
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(midi_map.to_dict(), f, indent=2)

    if on_progress:
        on_progress(100, "MIDI map saved")

    logger.info("Saved MIDI map '%s' (%d mappings) to %s",
                midi_map.name, len(mappings), output_path)
    return output_path


def load_midi_map(
    path: str,
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Load a MIDI mapping file.

    Args:
        path: Path to MIDI map JSON file.
        on_progress: Optional progress callback.

    Returns:
        MIDI map dict with ``name``, ``mappings``, etc.
    """
    if on_progress:
        on_progress(50, "Loading MIDI map...")

    if not os.path.isfile(path):
        raise FileNotFoundError(f"MIDI map not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    midi_map = MidiMap.from_dict(data)

    if on_progress:
        on_progress(100, "MIDI map loaded")

    logger.info("Loaded MIDI map '%s' (%d mappings) from %s",
                midi_map.name, len(midi_map.mappings), path)
    return midi_map.to_dict()


def list_midi_devices(
    on_progress: Optional[Callable] = None,
) -> List[Dict[str, str]]:
    """List supported/known MIDI device templates.

    Returns a static list of known device types.  Actual connected
    device detection is handled by the front-end via Web MIDI API.

    Returns:
        List of device dicts with ``name`` and ``type``.
    """
    if on_progress:
        on_progress(100, f"Listed {len(_DEFAULT_DEVICES)} MIDI devices")

    return list(_DEFAULT_DEVICES)


def list_saved_maps(
    on_progress: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    """List all saved MIDI map files.

    Returns:
        List of map summary dicts.
    """
    if on_progress:
        on_progress(50, "Listing MIDI maps...")

    if not os.path.isdir(_MIDI_DIR):
        return []

    maps = []
    for fname in sorted(os.listdir(_MIDI_DIR)):
        if not fname.endswith(".json"):
            continue
        try:
            fpath = os.path.join(_MIDI_DIR, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            maps.append({
                "name": data.get("name", fname),
                "device_name": data.get("device_name", ""),
                "mapping_count": len(data.get("mappings", [])),
                "path": fpath,
            })
        except (json.JSONDecodeError, OSError):
            continue

    if on_progress:
        on_progress(100, f"Found {len(maps)} MIDI maps")

    return maps
