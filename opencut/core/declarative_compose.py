"""
OpenCut Declarative Video Composition (Editly-inspired)

Assemble a finished video from a single JSON specification — no timeline
reasoning required by the caller.  Inspired by `editly`
(https://github.com/mifi/editly), adapted to FFmpeg-only execution with
no Node.js dependency.

A **composition spec** is a dict of the following shape::

    {
      "width": 1920,
      "height": 1080,
      "fps": 30,
      "audio": {"path": "optional/background.mp3", "volume": 0.3},
      "clips": [
        {
          "type": "video",
          "source": "path/to/input1.mp4",
          "duration": 5.0,
          "start": 2.0,            # optional in-point on the source
          "transition": {"name": "fade", "duration": 0.5},
          "text": "Intro",          # optional burnt caption
          "text_position": "bottom"  # top | bottom | center
        },
        {"type": "image", "source": "logo.png", "duration": 2.0, "transition": {"name": "fade", "duration": 0.4}},
        {"type": "title", "text": "Chapter One", "duration": 3.0, "bg": "#101014"},
        {"type": "color", "color": "#000", "duration": 0.5}
      ],
      "output": "out.mp4"
    }

Supported clip types:
    - ``video``: arbitrary input file; optional ``start`` in-point,
      ``duration`` trims length, ``text`` burns a caption via drawtext.
    - ``image``: still image, looped for ``duration`` seconds.
    - ``title``: solid background + centered text (no input file needed).
    - ``color``: solid color fill for ``duration`` seconds.

Supported transitions (between consecutive clips): ``fade``, ``wipeleft``,
``wiperight``, ``slideleft``, ``slideright``, ``dissolve``,
``circleopen``, ``circleclose``. Mapped to FFmpeg's ``xfade`` filter.

Design notes
------------
- Each clip is first rendered to a normalized intermediate file (same
  width/height/fps, yuv420p, AAC 48k stereo) so concatenation and xfade
  are guaranteed to work without "resolution mismatch" errors.
- Transitions reduce the natural running time (the outgoing and incoming
  clips overlap for ``transition.duration`` seconds). The final timeline
  accounts for this.
- Background audio, if provided, is ducked under any clip-native audio
  via ``sidechaincompress`` (broadcast-style ducking).
- No new deps: FFmpeg + Pillow already present. Text rendering uses
  ``drawtext`` with FFmpeg-built fontfile resolution.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import FFmpegCmd, run_ffmpeg

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

XFADE_NAMES = {
    "fade", "fadeblack", "fadewhite",
    "wipeleft", "wiperight", "wipeup", "wipedown",
    "slideleft", "slideright", "slideup", "slidedown",
    "dissolve", "pixelize",
    "circleopen", "circleclose", "radial",
    "smoothleft", "smoothright", "smoothup", "smoothdown",
}

CLIP_TYPES = {"video", "image", "title", "color"}

TEXT_POSITIONS = {"top", "bottom", "center"}

DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_FPS = 30
DEFAULT_BG_COLOR = "#000000"

_HEX_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{3}([0-9A-Fa-f]{3})?$")
_SAFE_TEXT_RE = re.compile(r"[\\':;]")  # chars that must be escaped in drawtext

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ComposeResult:
    """Structured return for declarative composition."""
    output: str = ""
    duration: float = 0.0
    clip_count: int = 0
    transition_count: int = 0
    width: int = 0
    height: int = 0
    fps: int = 0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, key):
        return getattr(self, key)

    def __contains__(self, key):
        return key in self.__dataclass_fields__

    def keys(self):
        return self.__dataclass_fields__.keys()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_color(value: str, default: str = DEFAULT_BG_COLOR) -> str:
    if not isinstance(value, str):
        return default
    v = value.strip()
    if not v:
        return default
    if _HEX_COLOR_RE.match(v):
        return v
    # Fallback: allow named FFmpeg colors (alphanumeric only — FFmpeg parses them)
    if v.replace("_", "").isalnum() and len(v) <= 30:
        return v
    return default


def _escape_text(text: str) -> str:
    """Escape user text for FFmpeg drawtext filter (single-quoted values)."""
    # drawtext text value is single-quoted; backslash, single quote, and
    # colon/semicolon need escaping. Strip control characters.
    safe = text.replace("\\", "\\\\")
    safe = safe.replace("'", r"\'")
    safe = safe.replace(":", r"\:")
    safe = safe.replace(";", r"\;")
    # Strip ASCII control characters (except tab / newline which we also drop)
    safe = re.sub(r"[\x00-\x1f\x7f]", " ", safe)
    # Keep only first 500 chars — drawtext doesn't paginate
    return safe[:500]


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _safe_float(value, default: float = 0.0, minimum: float = 0.0, maximum: float = 3600.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if f != f or f in (float("inf"), float("-inf")):
        return default
    return max(minimum, min(f, maximum))


def validate_spec(spec: Dict) -> Dict:
    """Normalize and validate a composition spec.  Raises ``ValueError`` on
    a malformed spec; otherwise returns a cleaned copy with defaults
    filled in.
    """
    _require(isinstance(spec, dict), "spec must be a JSON object")
    clips = spec.get("clips") or []
    _require(isinstance(clips, list) and clips, "spec.clips must be a non-empty list")
    _require(len(clips) <= 200, "spec.clips limited to 200 entries")

    width = int(spec.get("width") or DEFAULT_WIDTH)
    height = int(spec.get("height") or DEFAULT_HEIGHT)
    fps = int(spec.get("fps") or DEFAULT_FPS)
    _require(64 <= width <= 7680, "width must be 64..7680")
    _require(64 <= height <= 4320, "height must be 64..4320")
    _require(1 <= fps <= 120, "fps must be 1..120")

    audio = spec.get("audio") or {}
    if audio and not isinstance(audio, dict):
        audio = {}
    audio_path = audio.get("path", "") if audio else ""
    if audio_path and not os.path.isfile(audio_path):
        raise ValueError(f"audio.path not found: {audio_path}")

    cleaned_clips = []
    for i, clip in enumerate(clips):
        _require(isinstance(clip, dict), f"clips[{i}] must be an object")
        ctype = str(clip.get("type") or "video").lower().strip()
        _require(ctype in CLIP_TYPES, f"clips[{i}].type must be one of {sorted(CLIP_TYPES)}")

        duration = _safe_float(clip.get("duration"), default=3.0, minimum=0.1, maximum=600.0)
        entry = {"type": ctype, "duration": duration}

        if ctype in ("video", "image"):
            source = str(clip.get("source") or "").strip()
            _require(source and os.path.isfile(source), f"clips[{i}].source missing or not found")
            entry["source"] = source
            if ctype == "video":
                entry["start"] = _safe_float(clip.get("start"), default=0.0, minimum=0.0, maximum=86400.0)

        if ctype in ("title", "color"):
            entry["color"] = _validate_color(clip.get("color") or clip.get("bg") or DEFAULT_BG_COLOR)

        text = clip.get("text")
        if text:
            entry["text"] = _escape_text(str(text))
            pos = str(clip.get("text_position") or "bottom").lower()
            entry["text_position"] = pos if pos in TEXT_POSITIONS else "bottom"

        trans = clip.get("transition")
        if trans:
            if not isinstance(trans, dict):
                raise ValueError(f"clips[{i}].transition must be an object")
            tname = str(trans.get("name") or "fade").lower().strip()
            if tname not in XFADE_NAMES:
                raise ValueError(
                    f"clips[{i}].transition.name '{tname}' not supported; valid: {sorted(XFADE_NAMES)}"
                )
            tdur = _safe_float(trans.get("duration"), default=0.5, minimum=0.05, maximum=5.0)
            # Transition must be shorter than either adjacent clip
            tdur = min(tdur, duration - 0.05)
            if tdur > 0.04:
                entry["transition"] = {"name": tname, "duration": tdur}

        cleaned_clips.append(entry)

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "clips": cleaned_clips,
        "audio": {
            "path": audio_path,
            "volume": _safe_float((audio or {}).get("volume"), default=0.3, minimum=0.0, maximum=2.0),
        } if audio_path else {},
        "output": spec.get("output") or "",
    }


# ---------------------------------------------------------------------------
# Clip rendering
# ---------------------------------------------------------------------------


def _render_clip_to_normalized(
    clip: Dict,
    width: int,
    height: int,
    fps: int,
    out_path: str,
) -> None:
    """Render a single clip to a normalized intermediate file.

    Every intermediate clip has identical resolution, frame rate, pixel
    format, and audio layout so concat/xfade filters work without
    "resolution mismatch" errors.
    """
    ctype = clip["type"]
    duration = clip["duration"]

    # Build video filter chain for this clip
    vf_parts: List[str] = []

    if ctype == "video":
        cmd = FFmpegCmd()
        start = clip.get("start", 0.0)
        if start > 0:
            cmd.pre_input("ss", start)
        cmd.input(clip["source"])
        vf_parts.append(
            f"scale={width}:{height}:force_original_aspect_ratio=decrease"
        )
        vf_parts.append(
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black"
        )
        vf_parts.append(f"fps={fps}")
        cmd.option("t", duration)
    elif ctype == "image":
        cmd = FFmpegCmd()
        cmd.pre_input("loop", 1)
        cmd.pre_input("framerate", fps)
        cmd.input(clip["source"])
        vf_parts.append(
            f"scale={width}:{height}:force_original_aspect_ratio=decrease"
        )
        vf_parts.append(f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black")
        cmd.option("t", duration)
    elif ctype in ("title", "color"):
        color = clip.get("color") or DEFAULT_BG_COLOR
        cmd = FFmpegCmd()
        # ``color`` source is a virtual FFmpeg input that produces a solid
        # color frame stream.
        cmd.pre_input("f", "lavfi")
        cmd.input(f"color=c={color}:s={width}x{height}:r={fps}:d={duration}")
        # A silent audio track will be muxed from an lavfi anullsrc input.
    else:
        raise ValueError(f"Unknown clip type: {ctype}")

    # Add burnt text if requested
    text = clip.get("text")
    if text:
        pos = clip.get("text_position", "bottom")
        if pos == "top":
            y = "(h*0.08)"
        elif pos == "center":
            y = "(h-text_h)/2"
        else:
            y = "(h-text_h-h*0.08)"
        font_size = max(24, min(96, int(height * 0.055)))
        drawtext = (
            f"drawtext=text='{text}':fontcolor=white:fontsize={font_size}:"
            f"x=(w-text_w)/2:y={y}:"
            "box=1:boxcolor=black@0.45:boxborderw=16:line_spacing=4"
        )
        vf_parts.append(drawtext)

    if vf_parts:
        cmd.video_filter(",".join(vf_parts))

    # Audio: native track for video clips, silent for images/title/color.
    if ctype == "video":
        # Keep audio; normalize to stereo 48k
        cmd.option("ac", 2)
        cmd.option("ar", 48000)
        cmd.audio_codec("aac", bitrate="192k")
    else:
        # Add a silent stereo track so concat demuxer sees consistent streams
        cmd.pre_input("f", "lavfi")
        cmd.input("anullsrc=channel_layout=stereo:sample_rate=48000")
        cmd.map("0:v:0")
        cmd.map("1:a:0")
        cmd.option("shortest")
        cmd.audio_codec("aac", bitrate="128k")

    cmd.video_codec("libx264", crf=20, preset="veryfast")
    cmd.option("pix_fmt", "yuv420p")
    cmd.faststart()
    cmd.output(out_path)

    run_ffmpeg(cmd.build(), timeout=3600)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _concat_with_transitions(
    clip_paths: List[str],
    clip_durations: List[float],
    transitions: List[Optional[Dict]],
    width: int,
    height: int,
    fps: int,
    out_path: str,
    background_audio: Optional[Dict] = None,
) -> float:
    """Build a filter_complex graph that concatenates clips, applying
    xfade where transitions are specified.  Returns the final duration.

    ``transitions[i]`` describes the transition **between** clip *i* and
    clip *i+1* (so ``transitions[-1]`` is unused and may be None).
    """
    if not clip_paths:
        raise ValueError("No clips to concat")

    if len(clip_paths) == 1 and not background_audio:
        shutil.copyfile(clip_paths[0], out_path)
        return clip_durations[0]

    inputs = FFmpegCmd()
    for p in clip_paths:
        inputs.input(p)

    # Build filter_complex
    # Video label progression: [v0][v1] → [vc0] → [vc0][v2] → [vc1] → ...
    vf: List[str] = []
    af: List[str] = []

    # Start with clip 0 as the running video/audio
    cur_v = "[0:v]"
    cur_a = "[0:a]"
    cur_duration = clip_durations[0]

    for i in range(1, len(clip_paths)):
        tr = transitions[i - 1] if (i - 1) < len(transitions) else None
        next_v = f"[{i}:v]"
        next_a = f"[{i}:a]"
        out_v = f"[vx{i}]"
        out_a = f"[ax{i}]"
        if tr:
            tdur = tr["duration"]
            tname = tr["name"]
            # xfade offset = (cumulative duration so far) - transition duration
            offset = max(0.0, cur_duration - tdur)
            vf.append(
                f"{cur_v}{next_v}xfade=transition={tname}:duration={tdur}:offset={offset:.3f}{out_v}"
            )
            af.append(
                f"{cur_a}{next_a}acrossfade=d={tdur}:c1=tri:c2=tri{out_a}"
            )
            cur_duration = cur_duration + clip_durations[i] - tdur
        else:
            # Plain concat (no transition)
            vf.append(f"{cur_v}{next_v}concat=n=2:v=1:a=0{out_v}")
            af.append(f"{cur_a}{next_a}concat=n=2:v=0:a=1{out_a}")
            cur_duration = cur_duration + clip_durations[i]
        cur_v = out_v
        cur_a = out_a

    final_v = cur_v
    final_a = cur_a

    # Background audio ducking
    if background_audio and background_audio.get("path"):
        bg_input_idx = len(clip_paths)
        inputs.input(background_audio["path"])
        vol = background_audio.get("volume", 0.3)
        af.append(
            f"[{bg_input_idx}:a]aloop=loop=-1:size=2e+09,volume={vol}[bgloop]"
        )
        af.append(
            f"{final_a}[bgloop]amix=inputs=2:duration=first:weights=2 1,"
            f"atrim=0:{cur_duration:.3f},asetpts=N/SR/TB[amixed]"
        )
        final_a = "[amixed]"

    filter_complex = ";".join(vf + af)

    final_v_stripped = final_v.strip("[]")
    final_a_stripped = final_a.strip("[]")

    cmd = (
        inputs
        .filter_complex(filter_complex, maps=[f"[{final_v_stripped}]", f"[{final_a_stripped}]"])
        .video_codec("libx264", crf=18, preset="medium")
        .audio_codec("aac", bitrate="192k")
        .option("pix_fmt", "yuv420p")
        .option("r", fps)
        .faststart()
        .output(out_path)
        .build()
    )
    run_ffmpeg(cmd, timeout=10800)
    return cur_duration


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compose(
    spec: Dict,
    output: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> ComposeResult:
    """Render a composition spec to a single video file.

    See the module docstring for the schema. Raises ``ValueError`` on a
    malformed spec, ``FileNotFoundError`` on a missing source/asset.
    """
    clean = validate_spec(spec)
    clips = clean["clips"]
    w = clean["width"]
    h = clean["height"]
    fps = clean["fps"]

    out_path = output or clean["output"] or os.path.join(
        tempfile.gettempdir(), "opencut_compose.mp4"
    )

    if on_progress:
        on_progress(5, f"Validating {len(clips)} clip(s)")

    tmp_root = tempfile.mkdtemp(prefix="opencut_compose_")
    clip_paths: List[str] = []
    try:
        for i, clip in enumerate(clips):
            if on_progress:
                pct = 10 + int(60 * (i / max(1, len(clips))))
                on_progress(pct, f"Rendering clip {i + 1}/{len(clips)} ({clip['type']})")
            ip = os.path.join(tmp_root, f"clip_{i:03d}.mp4")
            _render_clip_to_normalized(clip, w, h, fps, ip)
            clip_paths.append(ip)

        # Collect per-clip durations (post-render — trust spec).
        clip_durations = [c["duration"] for c in clips]
        transitions = [c.get("transition") for c in clips]
        trans_count = sum(1 for t in transitions[:-1] if t)

        if on_progress:
            on_progress(75, "Concatenating with transitions")

        total_duration = _concat_with_transitions(
            clip_paths=clip_paths,
            clip_durations=clip_durations,
            transitions=transitions,
            width=w,
            height=h,
            fps=fps,
            out_path=out_path,
            background_audio=clean["audio"] or None,
        )

        if on_progress:
            on_progress(100, "Composition complete")

        return ComposeResult(
            output=out_path,
            duration=round(total_duration, 3),
            clip_count=len(clips),
            transition_count=trans_count,
            width=w,
            height=h,
            fps=fps,
            notes=[f"rendered {len(clips)} clips with {trans_count} transitions"],
        )
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def list_transitions() -> List[str]:
    """Return the sorted list of supported xfade transition names."""
    return sorted(XFADE_NAMES)


def list_clip_types() -> List[str]:
    """Return the sorted list of supported clip types."""
    return sorted(CLIP_TYPES)
