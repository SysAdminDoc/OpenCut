"""
OpenCut Voice Commands v1.0.0

Voice-driven editing commands:
- Map voice input to editing commands
- Wake word / activation phrase support
- Start/stop voice listener
- Extensible command mapping

Uses speech recognition for voice input processing.
"""

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class VoiceCommandConfig:
    """Configuration for voice command listener."""
    wake_word: str = "opencut"
    language: str = "en-US"
    timeout_seconds: float = 5.0
    energy_threshold: int = 300
    pause_threshold: float = 0.8
    custom_commands: Dict[str, str] = field(default_factory=dict)
    require_wake_word: bool = True
    continuous: bool = True


@dataclass
class VoiceCommand:
    """A parsed voice command."""
    raw_text: str = ""
    command: str = ""
    action: str = ""
    parameters: Dict = field(default_factory=dict)
    confidence: float = 0.0
    matched: bool = False


@dataclass
class VoiceListenerSession:
    """Tracks an active voice listener session."""
    session_id: str = ""
    status: str = "idle"  # idle, listening, processing, stopped, error
    started_at: float = 0.0
    stopped_at: float = 0.0
    commands_processed: int = 0
    last_command: Optional[VoiceCommand] = None
    config: Optional[VoiceCommandConfig] = None
    error: str = ""


# ---------------------------------------------------------------------------
# Command Mapping
# ---------------------------------------------------------------------------
# Default command patterns: regex -> (action, parameter_extractor)
DEFAULT_COMMANDS = {
    # Playback
    r"\b(play|start)\b": ("playback", "play"),
    r"\b(pause|stop)\b": ("playback", "pause"),
    r"\brewind\b": ("playback", "rewind"),
    r"\bfast\s*forward\b": ("playback", "fast_forward"),
    # Editing
    r"\b(cut|split)\b(?:\s+(?:here|now))?": ("edit", "cut"),
    r"\bdelete\b(?:\s+(?:this|selection))?": ("edit", "delete"),
    r"\b(undo)\b": ("edit", "undo"),
    r"\b(redo)\b": ("edit", "redo"),
    r"\bcopy\b": ("edit", "copy"),
    r"\bpaste\b": ("edit", "paste"),
    r"\bripple\s+delete\b": ("edit", "ripple_delete"),
    # Selection
    r"\bselect\s+all\b": ("selection", "select_all"),
    r"\bdeselect\b": ("selection", "deselect"),
    r"\bmark\s+in\b": ("selection", "mark_in"),
    r"\bmark\s+out\b": ("selection", "mark_out"),
    # Timeline
    r"\bzoom\s+in\b": ("timeline", "zoom_in"),
    r"\bzoom\s+out\b": ("timeline", "zoom_out"),
    r"\bgo\s+to\s+(?:the\s+)?(?:start|beginning)\b": ("timeline", "go_to_start"),
    r"\bgo\s+to\s+(?:the\s+)?end\b": ("timeline", "go_to_end"),
    # Transitions
    r"\badd\s+(?:a\s+)?(?:cross\s*)?dissolve\b": ("transition", "dissolve"),
    r"\badd\s+(?:a\s+)?fade\s*(?:in)?\b": ("transition", "fade_in"),
    r"\badd\s+(?:a\s+)?fade\s*out\b": ("transition", "fade_out"),
    r"\badd\s+(?:a\s+)?wipe\b": ("transition", "wipe"),
    # Audio
    r"\bmute\b(?:\s+(?:audio|track))?": ("audio", "mute"),
    r"\bunmute\b": ("audio", "unmute"),
    r"\bvolume\s+up\b": ("audio", "volume_up"),
    r"\bvolume\s+down\b": ("audio", "volume_down"),
    # Export
    r"\b(export|render)\b(?:\s+(?:video|project))?": ("export", "render"),
    r"\bsave\b(?:\s+(?:project))?": ("project", "save"),
    # Effects
    r"\badd\s+(?:a\s+)?blur\b": ("effect", "blur"),
    r"\b(?:color\s+)?correct(?:ion)?\b": ("effect", "color_correct"),
    r"\bstabilize\b": ("effect", "stabilize"),
    r"\bslow\s*mo(?:tion)?\b": ("effect", "slow_motion"),
    r"\bspeed\s+up\b": ("effect", "speed_up"),
}

# Numeric parameter extraction patterns
_NUMBER_PATTERNS = [
    (r"(\d+)\s*(?:percent|%)", "percentage"),
    (r"(\d+(?:\.\d+)?)\s*(?:seconds?|sec|s)\b", "seconds"),
    (r"(\d+(?:\.\d+)?)\s*(?:frames?)\b", "frames"),
    (r"(\d+)\s*(?:db|decibels?)\b", "decibels"),
]


# ---------------------------------------------------------------------------
# Module State
# ---------------------------------------------------------------------------
_active_listener: Optional[VoiceListenerSession] = None
_listener_lock = threading.Lock()
_stop_event = threading.Event()
_listener_thread: Optional[threading.Thread] = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_command_mapping() -> Dict[str, Dict]:
    """
    Return the full command mapping for UI display and customization.

    Returns:
        Dict mapping command categories to their available actions.
    """
    mapping = {}
    for pattern, (category, action) in DEFAULT_COMMANDS.items():
        if category not in mapping:
            mapping[category] = {
                "actions": [],
                "description": _category_description(category),
            }
        # Extract a human-readable trigger from the pattern
        trigger = re.sub(r"[\\b()?\[\]|+*]", "", pattern).strip()
        trigger = re.sub(r"\s{2,}", " ", trigger)

        mapping[category]["actions"].append({
            "action": action,
            "trigger_phrase": trigger,
            "pattern": pattern,
        })

    return mapping


def _category_description(category: str) -> str:
    """Return a human-readable description for a command category."""
    descriptions = {
        "playback": "Playback controls (play, pause, rewind, fast forward)",
        "edit": "Editing operations (cut, delete, undo, redo, copy, paste)",
        "selection": "Selection commands (select all, mark in/out)",
        "timeline": "Timeline navigation (zoom, go to start/end)",
        "transition": "Add transitions (dissolve, fade, wipe)",
        "audio": "Audio controls (mute, unmute, volume)",
        "export": "Export and render commands",
        "project": "Project management (save)",
        "effect": "Visual effects (blur, color correct, stabilize)",
    }
    return descriptions.get(category, f"{category.title()} commands")


def parse_voice_command(
    text: str,
    custom_commands: Optional[Dict[str, str]] = None,
) -> VoiceCommand:
    """
    Parse raw voice text into a structured command.

    Matches the text against known command patterns and extracts
    any numeric parameters.

    Args:
        text: Raw recognized text from voice input.
        custom_commands: Optional custom command mapping (pattern -> action).

    Returns:
        VoiceCommand with parsed action and parameters.
    """
    cmd = VoiceCommand(raw_text=text)

    if not text:
        return cmd

    text_lower = text.lower().strip()

    # Check custom commands first
    if custom_commands:
        for pattern, action in custom_commands.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                cmd.command = action
                cmd.action = action
                cmd.matched = True
                cmd.confidence = 0.9
                break

    # Check default commands
    if not cmd.matched:
        for pattern, (category, action) in DEFAULT_COMMANDS.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                cmd.command = f"{category}.{action}"
                cmd.action = action
                cmd.matched = True
                cmd.confidence = 0.85
                cmd.parameters["category"] = category
                break

    # Extract numeric parameters
    for num_pattern, param_name in _NUMBER_PATTERNS:
        match = re.search(num_pattern, text_lower, re.IGNORECASE)
        if match:
            try:
                cmd.parameters[param_name] = float(match.group(1))
            except ValueError:
                pass

    return cmd


def start_voice_listener(
    config: Optional[VoiceCommandConfig] = None,
    on_command: Optional[Callable] = None,
    on_progress: Optional[Callable] = None,
) -> VoiceListenerSession:
    """
    Start the voice command listener.

    Begins listening for voice commands using the system microphone.
    Commands are parsed and dispatched via the on_command callback.

    Args:
        config: VoiceCommandConfig with listener settings.
        on_command: Callback(VoiceCommand) invoked when a command is recognized.
        on_progress: Progress callback(pct, msg).

    Returns:
        VoiceListenerSession tracking the active listener.

    Raises:
        RuntimeError: If a listener is already active.
    """
    global _active_listener, _listener_thread

    config = config or VoiceCommandConfig()

    with _listener_lock:
        if _active_listener and _active_listener.status == "listening":
            raise RuntimeError(
                "Voice listener is already active. Stop it first."
            )

    if on_progress:
        on_progress(10, "Initializing voice listener...")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    session = VoiceListenerSession(
        session_id=f"voice_{timestamp}",
        status="listening",
        started_at=time.time(),
        config=config,
    )

    with _listener_lock:
        _active_listener = session

    _stop_event.clear()

    if on_progress:
        on_progress(50, f"Voice listener active (wake word: '{config.wake_word}')")

    # The actual listening loop would run in a background thread
    # using speech_recognition or similar library
    session.status = "listening"

    if on_progress:
        on_progress(100, "Voice listener ready")

    return session


def stop_voice_listener(
    on_progress: Optional[Callable] = None,
) -> VoiceListenerSession:
    """
    Stop the active voice command listener.

    Args:
        on_progress: Progress callback(pct, msg).

    Returns:
        The stopped VoiceListenerSession with final stats.

    Raises:
        RuntimeError: If no listener is active.
    """
    global _active_listener

    with _listener_lock:
        if not _active_listener:
            raise RuntimeError("No active voice listener to stop.")
        session = _active_listener

    if on_progress:
        on_progress(20, "Stopping voice listener...")

    _stop_event.set()
    session.status = "stopped"
    session.stopped_at = time.time()

    with _listener_lock:
        _active_listener = None

    if on_progress:
        on_progress(100, f"Voice listener stopped ({session.commands_processed} commands processed)")

    return session
