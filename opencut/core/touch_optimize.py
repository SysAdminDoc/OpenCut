"""
OpenCut Touch / Pen Optimization Module

Configuration for touch-optimised interfaces: larger tap targets,
gesture-to-action mappings, pen pressure sensitivity curves, and
layout adaptation for touch-first editing.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("opencut")

_OPENCUT_DIR = os.path.join(os.path.expanduser("~"), ".opencut")
_TOUCH_CONFIG_FILE = os.path.join(_OPENCUT_DIR, "touch_config.json")


@dataclass
class GestureMapping:
    """A gesture-to-action mapping."""
    gesture: str = ""              # e.g. "pinch_zoom", "two_finger_swipe"
    action: str = ""               # OpenCut action name
    params: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GestureMapping":
        return cls(
            gesture=str(d.get("gesture", "")),
            action=str(d.get("action", "")),
            params=d.get("params", {}),
            enabled=bool(d.get("enabled", True)),
            description=str(d.get("description", "")),
        )


@dataclass
class TouchConfig:
    """Complete touch optimization configuration."""
    min_target_size: int = 44          # Minimum tap target in px (Apple HIG)
    button_padding: int = 8            # Extra padding around buttons
    timeline_track_height: int = 60    # Taller tracks for touch
    scrubber_handle_size: int = 32     # Larger scrubber handle
    double_tap_interval_ms: int = 300  # ms window for double-tap
    long_press_ms: int = 500           # ms to trigger long-press
    swipe_threshold_px: int = 30       # Min px for swipe recognition
    pinch_sensitivity: float = 1.0     # Pinch zoom sensitivity
    gestures: List[GestureMapping] = field(default_factory=list)
    touch_enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_target_size": self.min_target_size,
            "button_padding": self.button_padding,
            "timeline_track_height": self.timeline_track_height,
            "scrubber_handle_size": self.scrubber_handle_size,
            "double_tap_interval_ms": self.double_tap_interval_ms,
            "long_press_ms": self.long_press_ms,
            "swipe_threshold_px": self.swipe_threshold_px,
            "pinch_sensitivity": self.pinch_sensitivity,
            "gestures": [g.to_dict() for g in self.gestures],
            "touch_enabled": self.touch_enabled,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TouchConfig":
        gestures = [GestureMapping.from_dict(g) for g in d.get("gestures", [])]
        return cls(
            min_target_size=int(d.get("min_target_size", 44)),
            button_padding=int(d.get("button_padding", 8)),
            timeline_track_height=int(d.get("timeline_track_height", 60)),
            scrubber_handle_size=int(d.get("scrubber_handle_size", 32)),
            double_tap_interval_ms=int(d.get("double_tap_interval_ms", 300)),
            long_press_ms=int(d.get("long_press_ms", 500)),
            swipe_threshold_px=int(d.get("swipe_threshold_px", 30)),
            pinch_sensitivity=float(d.get("pinch_sensitivity", 1.0)),
            gestures=gestures,
            touch_enabled=bool(d.get("touch_enabled", True)),
        )


@dataclass
class PenPressureConfig:
    """Pen / stylus pressure sensitivity configuration."""
    enabled: bool = True
    curve_type: str = "linear"         # "linear", "ease_in", "ease_out", "s_curve"
    min_pressure: float = 0.05         # Ignore below this threshold
    max_pressure: float = 1.0
    brush_size_min: float = 1.0        # Min brush/mask size at min pressure
    brush_size_max: float = 50.0       # Max brush/mask size at max pressure
    opacity_min: float = 0.1
    opacity_max: float = 1.0
    tilt_enabled: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PenPressureConfig":
        return cls(
            enabled=bool(d.get("enabled", True)),
            curve_type=str(d.get("curve_type", "linear")),
            min_pressure=float(d.get("min_pressure", 0.05)),
            max_pressure=float(d.get("max_pressure", 1.0)),
            brush_size_min=float(d.get("brush_size_min", 1.0)),
            brush_size_max=float(d.get("brush_size_max", 50.0)),
            opacity_min=float(d.get("opacity_min", 0.1)),
            opacity_max=float(d.get("opacity_max", 1.0)),
            tilt_enabled=bool(d.get("tilt_enabled", False)),
        )


# ---------------------------------------------------------------------------
# Default gestures
# ---------------------------------------------------------------------------

_DEFAULT_GESTURES = [
    GestureMapping(gesture="pinch_zoom", action="timeline_zoom",
                   description="Pinch to zoom timeline"),
    GestureMapping(gesture="two_finger_swipe_h", action="timeline_scroll",
                   description="Two-finger horizontal swipe to scroll timeline"),
    GestureMapping(gesture="two_finger_swipe_v", action="track_scroll",
                   description="Two-finger vertical swipe to scroll tracks"),
    GestureMapping(gesture="double_tap", action="play_pause",
                   description="Double-tap to play/pause"),
    GestureMapping(gesture="long_press", action="context_menu",
                   description="Long press for context menu"),
    GestureMapping(gesture="three_finger_swipe_left", action="undo",
                   description="Three-finger swipe left to undo"),
    GestureMapping(gesture="three_finger_swipe_right", action="redo",
                   description="Three-finger swipe right to redo"),
    GestureMapping(gesture="tap_hold_drag", action="trim_clip",
                   description="Tap-hold and drag to trim clip edge"),
    GestureMapping(gesture="two_finger_rotate", action="rotate_clip",
                   description="Two-finger rotate for clip rotation"),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_touch_config(
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Get the current touch optimization configuration.

    Loads from disk if a saved config exists, otherwise returns defaults.

    Returns:
        TouchConfig dict with all touch parameters and gesture mappings.
    """
    if on_progress:
        on_progress(30, "Loading touch config...")

    if os.path.isfile(_TOUCH_CONFIG_FILE):
        try:
            with open(_TOUCH_CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            config = TouchConfig.from_dict(data)
            if on_progress:
                on_progress(100, "Touch config loaded")
            return config.to_dict()
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load touch config: %s", e)

    # Return defaults
    config = TouchConfig(gestures=list(_DEFAULT_GESTURES))

    if on_progress:
        on_progress(100, "Default touch config returned")

    return config.to_dict()


def save_touch_config(
    config: Dict[str, Any],
    on_progress: Optional[Callable] = None,
) -> str:
    """Save touch configuration to disk.

    Args:
        config: TouchConfig dict.
        on_progress: Optional progress callback.

    Returns:
        Path where the config was saved.
    """
    if on_progress:
        on_progress(30, "Saving touch config...")

    os.makedirs(_OPENCUT_DIR, exist_ok=True)

    with open(_TOUCH_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    if on_progress:
        on_progress(100, "Touch config saved")

    logger.info("Saved touch config to %s", _TOUCH_CONFIG_FILE)
    return _TOUCH_CONFIG_FILE


def create_gesture_mapping(
    gesture: str,
    action: str,
    params: Optional[Dict[str, Any]] = None,
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Create a gesture-to-action mapping.

    Args:
        gesture: Gesture identifier (e.g. ``pinch_zoom``, ``double_tap``).
        action: OpenCut action to trigger.
        params: Optional action parameters.
        on_progress: Optional progress callback.

    Returns:
        GestureMapping dict.
    """
    if on_progress:
        on_progress(50, "Creating gesture mapping...")

    mapping = GestureMapping(
        gesture=gesture,
        action=action,
        params=params or {},
        description=f"{gesture} -> {action}",
    )

    if on_progress:
        on_progress(100, "Gesture mapping created")

    return mapping.to_dict()


def get_pen_pressure_config(
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Get pen / stylus pressure sensitivity configuration.

    Returns:
        PenPressureConfig dict.
    """
    if on_progress:
        on_progress(50, "Loading pen pressure config...")

    # Load from touch config if it has pen settings
    if os.path.isfile(_TOUCH_CONFIG_FILE):
        try:
            with open(_TOUCH_CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "pen_pressure" in data:
                config = PenPressureConfig.from_dict(data["pen_pressure"])
                if on_progress:
                    on_progress(100, "Pen config loaded")
                return config.to_dict()
        except (json.JSONDecodeError, OSError):
            pass

    # Defaults
    config = PenPressureConfig()

    if on_progress:
        on_progress(100, "Default pen config returned")

    return config.to_dict()


def optimize_layout_for_touch(
    layout_data: Dict[str, Any],
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Optimize a UI layout configuration for touch interaction.

    Adjusts sizing, spacing, and interaction parameters to meet
    touch usability guidelines (minimum 44px targets, adequate spacing).

    Args:
        layout_data: Layout config dict with element dimensions.
        on_progress: Optional progress callback.

    Returns:
        Optimised layout dict with adjustments applied and a summary.
    """
    if on_progress:
        on_progress(20, "Analysing layout for touch...")

    config = get_touch_config()
    min_size = config.get("min_target_size", 44)
    padding = config.get("button_padding", 8)

    adjustments = []
    optimised = dict(layout_data)

    # Check and fix element sizes
    elements = optimised.get("elements", [])
    for elem in elements:
        w = elem.get("width", 0)
        h = elem.get("height", 0)
        changed = False

        if w > 0 and w < min_size:
            elem["width"] = min_size
            changed = True
        if h > 0 and h < min_size:
            elem["height"] = min_size
            changed = True

        # Add padding
        if elem.get("type") in ("button", "toggle", "slider_handle"):
            elem["padding"] = max(elem.get("padding", 0), padding)
            changed = True

        if changed:
            adjustments.append({
                "element": elem.get("id", elem.get("type", "unknown")),
                "adjustment": "resized for touch targets",
            })

    if on_progress:
        on_progress(70, f"Applied {len(adjustments)} touch adjustments...")

    # Global touch optimizations
    optimised["touch_optimized"] = True
    optimised["min_target_size"] = min_size

    # Increase timeline track height
    if "timeline" in optimised:
        tl = optimised["timeline"]
        track_h = tl.get("track_height", 40)
        if track_h < config.get("timeline_track_height", 60):
            tl["track_height"] = config["timeline_track_height"]
            adjustments.append({
                "element": "timeline",
                "adjustment": f"track height {track_h} -> {config['timeline_track_height']}",
            })

    result = {
        "layout": optimised,
        "adjustments": adjustments,
        "total_adjustments": len(adjustments),
        "touch_optimized": True,
    }

    if on_progress:
        on_progress(100, f"Layout optimised ({len(adjustments)} changes)")

    logger.info("Optimised layout for touch: %d adjustments", len(adjustments))
    return result
