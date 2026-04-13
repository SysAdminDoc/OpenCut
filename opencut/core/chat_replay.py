"""
OpenCut Chat Replay Overlay (12.2)

Parse Twitch IRC logs and YouTube chat JSON, render a scrolling chat
overlay synchronized to video timestamps via FFmpeg drawtext.

Supported inputs:
- Twitch IRC log files (.log / .txt) with standard IRC timestamp format
- YouTube chat JSON (yt-dlp --write-chat format)

Overlay rendering via FFmpeg drawtext/ASS subtitle pipeline.
"""

import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from opencut.helpers import (
    FFmpegCmd,
    get_video_info,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class ChatMessage:
    """A single chat message with timestamp."""
    timestamp: float          # seconds into the video
    username: str = ""
    message: str = ""
    color: str = "#FFFFFF"    # hex color for the username
    is_mod: bool = False
    is_sub: bool = False
    badges: List[str] = field(default_factory=list)


@dataclass
class ChatReplayResult:
    """Result of chat replay overlay rendering."""
    output_path: str = ""
    message_count: int = 0
    duration: float = 0.0
    style: str = "scrolling"


# ---------------------------------------------------------------------------
# Username Color Assignment
# ---------------------------------------------------------------------------
_DEFAULT_COLORS = [
    "#FF4500", "#1E90FF", "#00FF7F", "#FFD700", "#FF69B4",
    "#9370DB", "#00CED1", "#FF6347", "#32CD32", "#BA55D3",
    "#FF8C00", "#4169E1", "#2E8B57", "#DC143C", "#7B68EE",
]


def _assign_color(username: str) -> str:
    """Deterministic color assignment from username hash."""
    h = sum(ord(c) for c in username)
    return _DEFAULT_COLORS[h % len(_DEFAULT_COLORS)]


# ---------------------------------------------------------------------------
# Twitch IRC Log Parser
# ---------------------------------------------------------------------------
_TWITCH_IRC_RE = re.compile(
    r"\[(\d{2}:\d{2}:\d{2})\]\s*"           # [HH:MM:SS]
    r"(?:<([^>]+)>|(\w+):)\s*"               # <username> or username:
    r"(.+)$"                                  # message text
)

_TWITCH_IRC_EXTENDED_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s+"                    # timestamp in seconds
    r"(\w+)\s+"                              # username
    r"(.+)$"                                 # message
)


def _parse_hms(hms: str) -> float:
    """Convert HH:MM:SS to seconds."""
    parts = hms.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    return float(parts[0])


def parse_twitch_chat(log_path: str) -> List[ChatMessage]:
    """Parse a Twitch IRC log file into ChatMessage objects.

    Supports formats:
    - ``[HH:MM:SS] <username> message``
    - ``[HH:MM:SS] username: message``
    - ``seconds username message`` (numeric timestamps)

    Args:
        log_path: Path to Twitch IRC log file.

    Returns:
        List of ChatMessage sorted by timestamp.
    """
    if not os.path.isfile(log_path):
        raise FileNotFoundError(f"Chat log not found: {log_path}")

    messages = []
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Try standard IRC format: [HH:MM:SS] <user> msg
            m = _TWITCH_IRC_RE.match(line)
            if m:
                ts = _parse_hms(m.group(1))
                username = m.group(2) or m.group(3) or "unknown"
                text = m.group(4).strip()
                messages.append(ChatMessage(
                    timestamp=ts,
                    username=username,
                    message=text,
                    color=_assign_color(username),
                ))
                continue

            # Try numeric timestamp format: 123.4 user msg
            m2 = _TWITCH_IRC_EXTENDED_RE.match(line)
            if m2:
                ts = float(m2.group(1))
                username = m2.group(2)
                text = m2.group(3).strip()
                messages.append(ChatMessage(
                    timestamp=ts,
                    username=username,
                    message=text,
                    color=_assign_color(username),
                ))

    messages.sort(key=lambda m: m.timestamp)
    logger.info("Parsed %d Twitch chat messages from %s", len(messages), log_path)
    return messages


# ---------------------------------------------------------------------------
# YouTube Chat JSON Parser
# ---------------------------------------------------------------------------
def parse_youtube_chat(json_path: str) -> List[ChatMessage]:
    """Parse a YouTube live chat JSON file into ChatMessage objects.

    Supports yt-dlp ``--write-chat`` format (JSONL with actions) and
    simple array-of-objects format.

    Args:
        json_path: Path to YouTube chat JSON file.

    Returns:
        List of ChatMessage sorted by timestamp.
    """
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Chat JSON not found: {json_path}")

    messages = []

    with open(json_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read().strip()

    # Try JSONL (yt-dlp format) -- one JSON object per line
    if content.startswith("{"):
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                msg = _parse_youtube_chat_object(obj)
                if msg:
                    messages.append(msg)
            except json.JSONDecodeError:
                continue
    elif content.startswith("["):
        # Array format
        try:
            data = json.loads(content)
            for obj in data:
                msg = _parse_youtube_chat_object(obj)
                if msg:
                    messages.append(msg)
        except json.JSONDecodeError:
            logger.warning("Failed to parse YouTube chat JSON: %s", json_path)

    messages.sort(key=lambda m: m.timestamp)
    logger.info("Parsed %d YouTube chat messages from %s", len(messages), json_path)
    return messages


def _parse_youtube_chat_object(obj: dict) -> Optional[ChatMessage]:
    """Extract a ChatMessage from a single YouTube chat JSON object."""
    # yt-dlp live_chat format
    if "replayChatItemAction" in obj:
        action = obj["replayChatItemAction"]
        actions = action.get("actions", [{}])
        if not actions:
            return None
        item_action = actions[0]
        renderer = (
            item_action
            .get("addChatItemAction", {})
            .get("item", {})
            .get("liveChatTextMessageRenderer", {})
        )
        if not renderer:
            return None

        # Timestamp in microseconds from video start
        offset_us = int(action.get("videoOffsetTimeMsec", 0))
        ts = offset_us / 1000.0  # convert ms to seconds

        author = renderer.get("authorName", {}).get("simpleText", "unknown")
        runs = renderer.get("message", {}).get("runs", [])
        text = "".join(r.get("text", "") for r in runs)

        if text:
            return ChatMessage(
                timestamp=ts,
                username=author,
                message=text,
                color=_assign_color(author),
            )
        return None

    # Simple format: {timestamp, username/author, message/text}
    ts = obj.get("timestamp", obj.get("time", obj.get("offset", 0)))
    if isinstance(ts, str):
        ts = _parse_hms(ts) if ":" in ts else float(ts)
    else:
        ts = float(ts)

    username = obj.get("username", obj.get("author", obj.get("user", "unknown")))
    text = obj.get("message", obj.get("text", obj.get("body", "")))

    if text:
        return ChatMessage(
            timestamp=ts,
            username=str(username),
            message=str(text),
            color=_assign_color(str(username)),
        )
    return None


# ---------------------------------------------------------------------------
# ASS Subtitle Generation for Chat Overlay
# ---------------------------------------------------------------------------
def _escape_ass(text: str) -> str:
    """Escape special characters for ASS subtitle format."""
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}").replace("\n", "\\N")


def _hex_to_ass_color(hex_color: str) -> str:
    """Convert #RRGGBB to ASS &HBBGGRR& format."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        hex_color = "FFFFFF"
    r, g, b = hex_color[0:2], hex_color[2:4], hex_color[4:6]
    return f"&H{b}{g}{r}&"


def _generate_ass_subtitle(
    messages: List[ChatMessage],
    video_width: int = 1920,
    video_height: int = 1080,
    position: str = "right",
    font_size: int = 24,
    max_visible: int = 15,
    style: str = "scrolling",
) -> str:
    """Generate an ASS subtitle file for the chat overlay.

    Args:
        messages: Sorted chat messages.
        video_width: Video width in pixels.
        video_height: Video height in pixels.
        position: Overlay position: 'right', 'left', 'bottom'.
        font_size: Font size in pixels.
        max_visible: Maximum simultaneously visible messages.
        style: 'scrolling' for animated scroll, 'static' for pop-in.

    Returns:
        ASS subtitle file content as string.
    """
    line_height = font_size + 4
    panel_height = max_visible * line_height
    margin_v = (video_height - panel_height) // 2

    if position == "left":
        margin_l, margin_r = 20, video_width // 2
        alignment = 7  # top-left
    elif position == "bottom":
        margin_l, margin_r = 100, 100
        margin_v = video_height - panel_height - 20
        alignment = 1  # bottom-left
    else:  # right (default)
        margin_l = video_width // 2
        margin_r = 20
        alignment = 9  # top-right

    header = f"""[Script Info]
Title: OpenCut Chat Replay
ScriptType: v4.00+
PlayResX: {video_width}
PlayResY: {video_height}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: ChatUser,Arial,{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,3,2,1,{alignment},{margin_l},{margin_r},{margin_v},1
Style: ChatMsg,Arial,{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,3,2,1,{alignment},{margin_l},{margin_r},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    display_duration = 5.0  # seconds each message is visible

    for i, msg in enumerate(messages):
        start_sec = msg.timestamp
        end_sec = start_sec + display_duration

        start_ts = _seconds_to_ass_time(start_sec)
        end_ts = _seconds_to_ass_time(end_sec)

        ass_color = _hex_to_ass_color(msg.color)
        username_esc = _escape_ass(msg.username)
        message_esc = _escape_ass(msg.message)

        # Slot position for stacking (cycle through max_visible slots)
        slot = i % max_visible
        y_offset = slot * line_height

        # Combine username (colored) and message
        text = (
            f"{{\\c{ass_color}\\b1}}{username_esc}{{\\c&HFFFFFF&\\b0}}: {message_esc}"
        )

        if style == "scrolling":
            # Fade in effect
            text = f"{{\\fad(200,200)\\pos(0,{y_offset})}}" + text

        events.append(
            f"Dialogue: 0,{start_ts},{end_ts},ChatMsg,,0,0,{y_offset},,{text}"
        )

    return header + "\n".join(events) + "\n"


def _seconds_to_ass_time(seconds: float) -> str:
    """Convert seconds to ASS timestamp H:MM:SS.cc format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


# ---------------------------------------------------------------------------
# Overlay Rendering
# ---------------------------------------------------------------------------
def render_chat_overlay(
    video_path: str,
    messages: List[ChatMessage],
    output_path_str: Optional[str] = None,
    position: str = "right",
    font_size: int = 24,
    opacity: float = 0.85,
    max_visible: int = 15,
    style: str = "scrolling",
    on_progress: Optional[Callable] = None,
) -> ChatReplayResult:
    """Render chat messages as a scrolling overlay on the video.

    Args:
        video_path: Source video file path.
        messages: List of ChatMessage to overlay.
        output_path_str: Output file path. Auto-generated if None.
        position: Overlay position: 'right', 'left', 'bottom'.
        font_size: Font size for chat text.
        opacity: Background panel opacity (0.0-1.0).
        max_visible: Max simultaneous visible messages.
        style: 'scrolling' or 'static'.
        on_progress: Progress callback(pct, msg).

    Returns:
        ChatReplayResult with output path and stats.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not messages:
        raise ValueError("No chat messages to render")

    if on_progress:
        on_progress(5, "Analyzing video...")

    info = get_video_info(video_path)
    width, height = info["width"], info["height"]
    duration = info["duration"]

    if on_progress:
        on_progress(15, f"Generating subtitle file for {len(messages)} messages...")

    # Generate ASS subtitle file
    ass_content = _generate_ass_subtitle(
        messages=messages,
        video_width=width,
        video_height=height,
        position=position,
        font_size=font_size,
        max_visible=max_visible,
        style=style,
    )

    if output_path_str is None:
        output_path_str = output_path(video_path, "chat_overlay")

    tmp_dir = tempfile.mkdtemp(prefix="opencut_chat_")
    ass_path = os.path.join(tmp_dir, "chat_overlay.ass")

    try:
        with open(ass_path, "w", encoding="utf-8") as f:
            f.write(ass_content)

        if on_progress:
            on_progress(30, "Rendering chat overlay...")

        # Build FFmpeg command with ASS subtitle burn-in
        ass_escaped = ass_path.replace("\\", "/").replace(":", "\\\\:")
        vf = f"ass='{ass_escaped}'"

        cmd = (
            FFmpegCmd()
            .input(video_path)
            .video_filter(vf)
            .video_codec("libx264", crf=18, preset="fast")
            .audio_codec("copy")
            .faststart()
            .output(output_path_str)
            .build()
        )
        run_ffmpeg(cmd)

        if on_progress:
            on_progress(100, "Chat overlay complete")

        return ChatReplayResult(
            output_path=output_path_str,
            message_count=len(messages),
            duration=duration,
            style=style,
        )

    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
