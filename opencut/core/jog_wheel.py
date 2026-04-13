"""
OpenCut Shuttle / Jog Wheel Integration Module

Map jog wheel rotation to timeline seek, shuttle ring to playback
speed control, and hardware buttons to OpenCut commands.

Supports common editing controller families:
  - Contour Design ShuttlePRO v2 / ShuttleXpress
  - Elgato Stream Deck + (with dial encoders)
  - Generic HID jog/shuttle devices

This module produces mapping configuration JSON consumed by the
front-end HID bridge.  No direct USB/HID I/O is performed.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("opencut")

_OPENCUT_DIR = os.path.join(os.path.expanduser("~"), ".opencut")
_JOG_DIR = os.path.join(_OPENCUT_DIR, "jog_mappings")


@dataclass
class JogAction:
    """A single jog/shuttle/button action mapping."""
    input_type: str = "jog"        # "jog", "shuttle", "button"
    input_id: int = 0              # Button index or axis ID
    action: str = ""               # OpenCut command or parameter
    params: Dict[str, Any] = field(default_factory=dict)
    sensitivity: float = 1.0       # Multiplier for jog/shuttle
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "JogAction":
        return cls(
            input_type=str(d.get("input_type", "jog")),
            input_id=int(d.get("input_id", 0)),
            action=str(d.get("action", "")),
            params=d.get("params", {}),
            sensitivity=float(d.get("sensitivity", 1.0)),
            description=str(d.get("description", "")),
        )


@dataclass
class JogMapping:
    """A complete jog/shuttle controller mapping."""
    name: str = ""
    device_type: str = ""
    description: str = ""
    actions: List[JogAction] = field(default_factory=list)
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "device_type": self.device_type,
            "description": self.description,
            "actions": [a.to_dict() for a in self.actions],
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "JogMapping":
        actions = [JogAction.from_dict(a) for a in d.get("actions", [])]
        return cls(
            name=d.get("name", ""),
            device_type=d.get("device_type", ""),
            description=d.get("description", ""),
            actions=actions,
            version=d.get("version", 1),
        )


# ---------------------------------------------------------------------------
# Supported devices and defaults
# ---------------------------------------------------------------------------

_SUPPORTED_DEVICES = {
    "shuttlepro_v2": {
        "name": "Contour ShuttlePRO v2",
        "type": "shuttlepro_v2",
        "buttons": 15,
        "has_jog": True,
        "has_shuttle": True,
        "shuttle_positions": 7,
    },
    "shuttlexpress": {
        "name": "Contour ShuttleXpress",
        "type": "shuttlexpress",
        "buttons": 5,
        "has_jog": True,
        "has_shuttle": True,
        "shuttle_positions": 7,
    },
    "stream_deck_plus": {
        "name": "Elgato Stream Deck +",
        "type": "stream_deck_plus",
        "buttons": 8,
        "has_jog": True,
        "has_shuttle": False,
        "shuttle_positions": 0,
    },
    "generic_hid": {
        "name": "Generic HID Jog/Shuttle",
        "type": "generic_hid",
        "buttons": 10,
        "has_jog": True,
        "has_shuttle": True,
        "shuttle_positions": 5,
    },
}


_DEFAULT_MAPPINGS = {
    "shuttlepro_v2": JogMapping(
        name="ShuttlePRO v2 Default",
        device_type="shuttlepro_v2",
        description="Default mapping for Contour ShuttlePRO v2",
        actions=[
            JogAction(input_type="jog", input_id=0, action="seek_frames",
                      sensitivity=1.0, description="Jog: frame-by-frame seek"),
            JogAction(input_type="shuttle", input_id=0, action="playback_speed",
                      sensitivity=1.0, description="Shuttle: variable speed playback"),
            JogAction(input_type="button", input_id=0, action="play_pause",
                      description="Play/Pause"),
            JogAction(input_type="button", input_id=1, action="stop",
                      description="Stop"),
            JogAction(input_type="button", input_id=2, action="mark_in",
                      description="Set In point"),
            JogAction(input_type="button", input_id=3, action="mark_out",
                      description="Set Out point"),
            JogAction(input_type="button", input_id=4, action="cut",
                      description="Cut at playhead"),
            JogAction(input_type="button", input_id=5, action="undo",
                      description="Undo"),
            JogAction(input_type="button", input_id=6, action="redo",
                      description="Redo"),
        ],
    ),
    "shuttlexpress": JogMapping(
        name="ShuttleXpress Default",
        device_type="shuttlexpress",
        description="Default mapping for Contour ShuttleXpress",
        actions=[
            JogAction(input_type="jog", input_id=0, action="seek_frames",
                      sensitivity=1.0, description="Jog: frame-by-frame seek"),
            JogAction(input_type="shuttle", input_id=0, action="playback_speed",
                      sensitivity=1.0, description="Shuttle: variable speed"),
            JogAction(input_type="button", input_id=0, action="play_pause",
                      description="Play/Pause"),
            JogAction(input_type="button", input_id=1, action="stop",
                      description="Stop"),
            JogAction(input_type="button", input_id=2, action="mark_in",
                      description="Set In point"),
            JogAction(input_type="button", input_id=3, action="mark_out",
                      description="Set Out point"),
            JogAction(input_type="button", input_id=4, action="cut",
                      description="Cut at playhead"),
        ],
    ),
    "generic_hid": JogMapping(
        name="Generic HID Default",
        device_type="generic_hid",
        description="Basic jog/shuttle mapping for generic HID controllers",
        actions=[
            JogAction(input_type="jog", input_id=0, action="seek_frames",
                      sensitivity=1.0, description="Jog: seek"),
            JogAction(input_type="shuttle", input_id=0, action="playback_speed",
                      sensitivity=1.0, description="Shuttle: speed"),
            JogAction(input_type="button", input_id=0, action="play_pause",
                      description="Play/Pause"),
            JogAction(input_type="button", input_id=1, action="stop",
                      description="Stop"),
        ],
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_jog_mapping(
    device_type: str,
    mappings: List[Dict[str, Any]],
    name: str = "",
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Create a jog/shuttle controller mapping.

    Args:
        device_type: Controller type identifier.
        mappings: List of action mapping dicts, each with ``input_type``,
            ``input_id``, ``action``, and optional ``sensitivity``, ``params``.
        name: Human-readable name.
        on_progress: Optional progress callback.

    Returns:
        Complete JogMapping dict.
    """
    if on_progress:
        on_progress(30, "Creating jog mapping...")

    actions = [JogAction.from_dict(m) for m in mappings]

    mapping = JogMapping(
        name=name or f"{device_type} Custom",
        device_type=device_type,
        actions=actions,
    )

    if on_progress:
        on_progress(100, "Jog mapping created")

    logger.info("Created jog mapping '%s' for %s (%d actions)",
                mapping.name, device_type, len(actions))
    return mapping.to_dict()


def list_supported_devices(
    on_progress: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    """List all supported jog/shuttle devices.

    Returns:
        List of device info dicts.
    """
    if on_progress:
        on_progress(100, f"Listed {len(_SUPPORTED_DEVICES)} devices")

    return list(_SUPPORTED_DEVICES.values())


def get_default_mapping(
    device_type: str,
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Get the default mapping for a device type.

    Args:
        device_type: Controller type identifier.
        on_progress: Optional progress callback.

    Returns:
        Default JogMapping dict.

    Raises:
        ValueError: If device_type has no default mapping.
    """
    if on_progress:
        on_progress(50, f"Loading default for {device_type}...")

    if device_type not in _DEFAULT_MAPPINGS:
        raise ValueError(
            f"No default mapping for device type '{device_type}'. "
            f"Supported: {', '.join(_DEFAULT_MAPPINGS.keys())}"
        )

    mapping = _DEFAULT_MAPPINGS[device_type]

    if on_progress:
        on_progress(100, "Default mapping loaded")

    return mapping.to_dict()


def save_jog_mapping(
    mapping: Dict[str, Any],
    output_path: str = "",
    on_progress: Optional[Callable] = None,
) -> str:
    """Save a jog mapping to a JSON file.

    Args:
        mapping: JogMapping dict.
        output_path: Destination path (auto-generated if empty).
        on_progress: Optional progress callback.

    Returns:
        File path where the mapping was saved.
    """
    if on_progress:
        on_progress(30, "Saving jog mapping...")

    if not output_path:
        os.makedirs(_JOG_DIR, exist_ok=True)
        name = mapping.get("name", "untitled")
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        output_path = os.path.join(_JOG_DIR, f"{safe_name}.json")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    if on_progress:
        on_progress(100, "Jog mapping saved")

    logger.info("Saved jog mapping to %s", output_path)
    return output_path


def load_jog_mapping(
    path: str,
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Load a jog mapping from a JSON file.

    Args:
        path: Path to mapping JSON.
        on_progress: Optional progress callback.

    Returns:
        JogMapping dict.
    """
    if on_progress:
        on_progress(50, "Loading jog mapping...")

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Jog mapping not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if on_progress:
        on_progress(100, "Jog mapping loaded")

    return JogMapping.from_dict(data).to_dict()
