"""
OpenCut Stream Deck Integration Module

Map OpenCut operations to Elgato Stream Deck buttons with dynamic
labels and icons.  Profiles are stored as JSON files under
``~/.opencut/stream_deck_profiles/``.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("opencut")

_OPENCUT_DIR = os.path.join(os.path.expanduser("~"), ".opencut")
_PROFILES_DIR = os.path.join(_OPENCUT_DIR, "stream_deck_profiles")


@dataclass
class ButtonMapping:
    """A single Stream Deck button mapping."""
    button_id: int = 0
    operation: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    label: str = ""
    icon: str = ""
    color: str = "#333333"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ButtonMapping":
        return cls(
            button_id=int(d.get("button_id", 0)),
            operation=str(d.get("operation", "")),
            params=d.get("params", {}),
            label=str(d.get("label", "")),
            icon=str(d.get("icon", "")),
            color=str(d.get("color", "#333333")),
        )


@dataclass
class StreamDeckProfile:
    """A complete Stream Deck profile with button mappings."""
    name: str = ""
    description: str = ""
    device_type: str = "stream_deck_mk2"
    rows: int = 3
    cols: int = 5
    buttons: List[ButtonMapping] = field(default_factory=list)
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "device_type": self.device_type,
            "rows": self.rows,
            "cols": self.cols,
            "buttons": [b.to_dict() for b in self.buttons],
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StreamDeckProfile":
        buttons = [ButtonMapping.from_dict(b) for b in d.get("buttons", [])]
        return cls(
            name=d.get("name", ""),
            description=d.get("description", ""),
            device_type=d.get("device_type", "stream_deck_mk2"),
            rows=d.get("rows", 3),
            cols=d.get("cols", 5),
            buttons=buttons,
            version=d.get("version", 1),
        )


# ---------------------------------------------------------------------------
# Default profiles
# ---------------------------------------------------------------------------

_DEFAULT_PROFILES = {
    "editing": StreamDeckProfile(
        name="editing",
        description="Basic editing operations",
        buttons=[
            ButtonMapping(button_id=0, operation="undo", label="Undo", color="#e74c3c"),
            ButtonMapping(button_id=1, operation="redo", label="Redo", color="#3498db"),
            ButtonMapping(button_id=2, operation="cut", label="Cut", color="#f39c12"),
            ButtonMapping(button_id=3, operation="copy", label="Copy", color="#2ecc71"),
            ButtonMapping(button_id=4, operation="paste", label="Paste", color="#9b59b6"),
            ButtonMapping(button_id=5, operation="play_pause", label="Play", color="#1abc9c"),
            ButtonMapping(button_id=6, operation="stop", label="Stop", color="#e74c3c"),
            ButtonMapping(button_id=7, operation="mark_in", label="In", color="#f1c40f"),
            ButtonMapping(button_id=8, operation="mark_out", label="Out", color="#f1c40f"),
            ButtonMapping(button_id=9, operation="export", label="Export", color="#2ecc71"),
        ],
    ),
    "color_grading": StreamDeckProfile(
        name="color_grading",
        description="Color grading shortcuts",
        buttons=[
            ButtonMapping(button_id=0, operation="auto_color", label="Auto Color", color="#e67e22"),
            ButtonMapping(button_id=1, operation="white_balance", label="WB", color="#ecf0f1"),
            ButtonMapping(button_id=2, operation="exposure_up", label="Exp +", color="#f1c40f"),
            ButtonMapping(button_id=3, operation="exposure_down", label="Exp -", color="#2c3e50"),
            ButtonMapping(button_id=4, operation="reset_grade", label="Reset", color="#e74c3c"),
        ],
    ),
    "audio": StreamDeckProfile(
        name="audio",
        description="Audio editing shortcuts",
        buttons=[
            ButtonMapping(button_id=0, operation="mute_toggle", label="Mute", color="#e74c3c"),
            ButtonMapping(button_id=1, operation="volume_up", label="Vol +", color="#2ecc71"),
            ButtonMapping(button_id=2, operation="volume_down", label="Vol -", color="#e67e22"),
            ButtonMapping(button_id=3, operation="solo_track", label="Solo", color="#f1c40f"),
            ButtonMapping(button_id=4, operation="silence_remove", label="Rm Silence", color="#3498db"),
        ],
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_stream_deck_profile(
    profile_name: str,
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Load a Stream Deck profile by name.

    Checks user-saved profiles first, then falls back to built-in defaults.

    Args:
        profile_name: Profile name.
        on_progress: Optional progress callback.

    Returns:
        Profile dict with button mappings.
    """
    if on_progress:
        on_progress(30, f"Loading profile '{profile_name}'...")

    # Check saved profiles
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in profile_name)
    saved_path = os.path.join(_PROFILES_DIR, f"{safe_name}.json")

    if os.path.isfile(saved_path):
        try:
            with open(saved_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            profile = StreamDeckProfile.from_dict(data)
            if on_progress:
                on_progress(100, "Profile loaded")
            return profile.to_dict()
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load saved profile '%s': %s", profile_name, e)

    # Check defaults
    if profile_name in _DEFAULT_PROFILES:
        if on_progress:
            on_progress(100, "Default profile loaded")
        return _DEFAULT_PROFILES[profile_name].to_dict()

    raise FileNotFoundError(f"Stream Deck profile not found: {profile_name}")


def create_button_mapping(
    button_id: int,
    operation: str,
    params: Optional[Dict[str, Any]] = None,
    label: str = "",
    icon: str = "",
    color: str = "#333333",
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Create a single button mapping configuration.

    Args:
        button_id: Button position index (0-based).
        operation: OpenCut operation name.
        params: Optional parameters for the operation.
        label: Display label for the button.
        icon: Icon identifier or path.
        color: Button background color hex.
        on_progress: Optional progress callback.

    Returns:
        Button mapping dict.
    """
    if on_progress:
        on_progress(50, "Creating button mapping...")

    mapping = ButtonMapping(
        button_id=button_id,
        operation=operation,
        params=params or {},
        label=label or operation.replace("_", " ").title(),
        icon=icon,
        color=color,
    )

    if on_progress:
        on_progress(100, "Button mapping created")

    return mapping.to_dict()


def list_profiles(
    on_progress: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    """List all available Stream Deck profiles (saved + defaults).

    Returns:
        List of profile summary dicts.
    """
    if on_progress:
        on_progress(30, "Listing profiles...")

    profiles = []

    # Defaults
    for name, profile in _DEFAULT_PROFILES.items():
        profiles.append({
            "name": name,
            "description": profile.description,
            "button_count": len(profile.buttons),
            "source": "default",
        })

    # Saved profiles
    if os.path.isdir(_PROFILES_DIR):
        for fname in sorted(os.listdir(_PROFILES_DIR)):
            if not fname.endswith(".json"):
                continue
            try:
                fpath = os.path.join(_PROFILES_DIR, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                name = data.get("name", fname.replace(".json", ""))
                # Skip if it shadows a default
                if name not in _DEFAULT_PROFILES:
                    profiles.append({
                        "name": name,
                        "description": data.get("description", ""),
                        "button_count": len(data.get("buttons", [])),
                        "source": "user",
                    })
            except (json.JSONDecodeError, OSError):
                continue

    if on_progress:
        on_progress(100, f"Found {len(profiles)} profiles")

    return profiles


def save_profile(
    profile: Dict[str, Any],
    on_progress: Optional[Callable] = None,
) -> str:
    """Save a Stream Deck profile to disk.

    Args:
        profile: Profile dict with ``name`` and ``buttons``.
        on_progress: Optional progress callback.

    Returns:
        File path where the profile was saved.
    """
    if on_progress:
        on_progress(30, "Saving profile...")

    os.makedirs(_PROFILES_DIR, exist_ok=True)

    name = profile.get("name", "untitled")
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    save_path = os.path.join(_PROFILES_DIR, f"{safe_name}.json")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    if on_progress:
        on_progress(100, "Profile saved")

    logger.info("Saved Stream Deck profile '%s' to %s", name, save_path)
    return save_path


def export_stream_deck_profile(
    profile: Dict[str, Any],
    output_path: str,
    on_progress: Optional[Callable] = None,
) -> str:
    """Export a Stream Deck profile to a specified path.

    Args:
        profile: Profile dict.
        output_path: Destination file path.
        on_progress: Optional progress callback.

    Returns:
        Output file path.
    """
    if on_progress:
        on_progress(30, "Exporting profile...")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    if on_progress:
        on_progress(100, "Profile exported")

    logger.info("Exported Stream Deck profile to %s", output_path)
    return output_path
