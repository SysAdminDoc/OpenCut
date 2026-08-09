"""Deterministic audio-reactive video rendering.

The renderer intentionally uses OpenCut's local PCM/onset analysis instead of
requiring BeatNet or a model download.  BeatNet remains an optional diagnostic
capability, but it is never on the render's critical path.
"""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from opencut.helpers import (
    _try_import,
    get_ffmpeg_path,
    get_ffprobe_path,
    get_video_info,
    output_path,
    run_ffmpeg,
)

PRESETS = {
    "boom": {
        "zoom_pulse": 0.8,
        "chromatic_aberration": 3,
        "color_saturation_boost": 0.6,
        "shake_intensity": 2,
        "strobe_on_beat": False,
    },
    "bass_drop": {
        "zoom_pulse": 1.0,
        "chromatic_aberration": 5,
        "color_saturation_boost": 1.0,
        "shake_intensity": 3,
        "strobe_on_beat": True,
    },
    "snare": {
        "zoom_pulse": 0.4,
        "chromatic_aberration": 1,
        "color_saturation_boost": 0.3,
        "shake_intensity": 1,
        "strobe_on_beat": False,
    },
    "chill": {
        "zoom_pulse": 0.1,
        "chromatic_aberration": 0,
        "color_saturation_boost": 0.1,
        "shake_intensity": 0,
        "strobe_on_beat": False,
    },
}

INSTALL_HINT = (
    "No BeatNet install is required: OpenCut uses deterministic local PCM "
    "onset analysis and the bundled FFmpeg binary."
)
ANALYSIS_WINDOW_MS = 50.0
ANALYSIS_SAMPLE_RATE = 16_000
MAX_KEYFRAMES = 5_000
MAX_FILTER_POINTS = 180
MAX_PARAMETER_VALUES = {
    "zoom_pulse": 2.0,
    "chromatic_aberration": 10.0,
    "color_saturation_boost": 2.0,
    "shake_intensity": 10.0,
}
_PARAMETER_KEYS = frozenset((*MAX_PARAMETER_VALUES, "strobe_on_beat"))


class AudioReactiveCancelled(RuntimeError):
    """Raised when a caller cancels before a render phase begins."""


def check_optional_beatnet_available() -> bool:
    """Return whether the optional BeatNet diagnostic backend is installed."""
    return _try_import("BeatNet") is not None


def check_audio_reactive_available() -> bool:
    """Return whether the deterministic renderer's FFmpeg backend is ready."""
    try:
        return bool(get_ffmpeg_path())
    except (OSError, RuntimeError):
        return False


@dataclass
class AudioReactiveResult:
    output: str = ""
    keyframes: List[Dict[str, Any]] = field(default_factory=list)
    preset: str = ""
    beat_count: int = 0
    notes: List[str] = field(default_factory=list)
    analysis: Dict[str, Any] = field(default_factory=dict)
    backend: str = "pcm_onset"
    capabilities: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def keys(self):
        return (
            "output",
            "keyframes",
            "preset",
            "beat_count",
            "notes",
            "analysis",
            "backend",
            "capabilities",
        )

    def __contains__(self, key: str) -> bool:
        return key in self.keys()


def list_presets() -> List[Dict[str, Any]]:
    """Return the stable, JSON-safe preset catalogue."""
    return [{"name": name, **params} for name, params in PRESETS.items()]


def validate_request_payload(data: Any) -> Optional[str]:
    """Return a synchronous validation error for the route request body."""
    if not isinstance(data, dict):
        return "Request body must be a JSON object."
    preset = str(data.get("preset") or "boom").strip().lower()
    if preset not in PRESETS:
        return f"Unknown preset '{preset}'. Choose from: {', '.join(PRESETS)}"
    custom = data.get("custom_params")
    if custom is not None and not isinstance(custom, dict):
        return "custom_params must be a JSON object."
    if isinstance(custom, dict):
        unknown = sorted(set(custom) - _PARAMETER_KEYS)
        if unknown:
            return f"Unsupported custom parameter(s): {', '.join(unknown)}"
        for name, maximum in MAX_PARAMETER_VALUES.items():
            if name not in custom:
                continue
            value = custom[name]
            if isinstance(value, bool):
                return f"custom_params.{name} must be numeric."
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return f"custom_params.{name} must be numeric."
            if not math.isfinite(numeric) or numeric < 0 or numeric > maximum:
                return f"custom_params.{name} must be between 0 and {maximum}."
        if "strobe_on_beat" in custom and not isinstance(custom["strobe_on_beat"], bool):
            return "custom_params.strobe_on_beat must be boolean."
    return None


def _normalise_parameters(preset: str, custom_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    error = validate_request_payload({"preset": preset, "custom_params": custom_params})
    if error:
        raise ValueError(error)
    params: Dict[str, Any] = dict(PRESETS[preset])
    for name, value in (custom_params or {}).items():
        params[name] = bool(value) if name == "strobe_on_beat" else round(float(value), 6)
    return params


def _raise_if_cancelled(is_cancelled: Optional[Callable[[], bool]]) -> None:
    if is_cancelled and is_cancelled():
        raise AudioReactiveCancelled("Audio-reactive render cancelled.")


def _current_job_id() -> str:
    try:
        from opencut.jobs import get_current_job_id

        return str(get_current_job_id() or "")
    except Exception:  # noqa: BLE001 - direct library calls have no job context
        return ""


def _clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return low
    if not math.isfinite(numeric):
        return low
    return max(low, min(high, numeric))


def _sample_indices(length: int, maximum: int = MAX_KEYFRAMES) -> List[int]:
    if length <= 0:
        return []
    if length <= maximum:
        return list(range(length))
    if maximum == 1:
        return [0]
    return [round(index * (length - 1) / (maximum - 1)) for index in range(maximum)]


def _build_keyframes(features: Dict[str, List[float]], window_ms: float) -> tuple[List[Dict[str, Any]], int]:
    lengths = [len(values) for values in features.values() if isinstance(values, list)]
    frame_count = max(lengths, default=0)
    beats = features.get("beats") or []
    beat_count = sum(1 for value in beats if _clamp(value) > 0)
    indices = _sample_indices(frame_count)
    keyframes = []
    for index in indices:
        amplitude = _clamp((features.get("amplitude") or [])[index] if index < len(features.get("amplitude") or []) else 0)
        rms = _clamp((features.get("rms") or [])[index] if index < len(features.get("rms") or []) else 0)
        beat = bool(index < len(beats) and _clamp(beats[index]) > 0)
        drive = max(amplitude, rms, 1.0 if beat else 0.0)
        keyframes.append(
            {
                "time": round(index * window_ms / 1000.0, 6),
                "amplitude": round(amplitude, 6),
                "rms": round(rms, 6),
                "beat": beat,
                "drive": round(drive, 6),
            }
        )
    return keyframes, beat_count


def _bounded_filter_points(keyframes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(keyframes) <= MAX_FILTER_POINTS:
        return keyframes
    indices = _sample_indices(len(keyframes), MAX_FILTER_POINTS)
    return [keyframes[index] for index in indices]


def _nested_expression(points: List[tuple[float, float]], default: str = "0") -> str:
    expression = default
    for start, value in reversed(points):
        end = start + ANALYSIS_WINDOW_MS / 1000.0
        expression = (
            f"if(between(t,{start:.3f},{end:.3f}),{value:.5f},{expression})"
        )
    return expression


def build_video_filter(
    keyframes: List[Dict[str, Any]],
    params: Dict[str, Any],
    width: int,
    height: int,
) -> str:
    """Build a bounded FFmpeg expression driven by deterministic keyframes."""
    if not keyframes or not any(
        float(params.get(name, 0) or 0) > 0 for name in MAX_PARAMETER_VALUES
    ) and not params.get("strobe_on_beat"):
        return "null"

    points = _bounded_filter_points(keyframes)
    drive = _nested_expression([(float(row["time"]), float(row["drive"])) for row in points])
    beat_points = [
        (float(row["time"]), 1.0)
        for row in points
        if row.get("beat")
    ]
    beat = _nested_expression(beat_points)
    zoom = float(params.get("zoom_pulse") or 0)
    saturation = float(params.get("color_saturation_boost") or 0)
    chromatic = float(params.get("chromatic_aberration") or 0)
    strobe = 0.9 if params.get("strobe_on_beat") else 0.0
    brightness = max(0.0, min(1.0, strobe + zoom * 0.08))
    contrast = max(0.0, min(2.0, 1.0 + zoom * 0.15))
    filters = [
        (
            "eq="
            f"brightness='if(gt({drive},0),{brightness:.5f}*{drive},0)':"
            f"contrast='1+({contrast - 1.0:.5f})*{drive}':"
            f"saturation='1+({saturation:.5f})*{drive}'"
        )
    ]
    if chromatic > 0:
        filters.append(f"hue=h='({chromatic:.5f})*{beat}'")
    shake = float(params.get("shake_intensity") or 0)
    if shake > 0 and width > 2 and height > 2:
        max_offset = max(1, min(int(shake * 2), min(width, height) // 4))
        crop_width = max(2, width - max_offset * 2)
        crop_height = max(2, height - max_offset * 2)
        x_expr = f"{max_offset}+{max_offset}*sin(2*PI*t)*{drive}"
        y_expr = f"{max_offset}+{max_offset}*cos(2*PI*t)*{drive}"
        filters.extend(
            [
                f"crop={crop_width}:{crop_height}:x='{x_expr}':y='{y_expr}'",
                f"scale={width}:{height}:flags=bicubic",
            ]
        )
    return ",".join(filters)


def _output_path_for(video_path: str, requested: Optional[str]) -> str:
    candidate = requested or output_path(video_path, "audio_reactive", "")
    root, extension = os.path.splitext(str(candidate))
    if not extension:
        candidate = root + ".mp4"
    try:
        from opencut.security import validate_output_path

        candidate = validate_output_path(candidate)
    except ImportError:
        candidate = os.path.abspath(candidate)
    if os.path.normcase(os.path.abspath(candidate)) == os.path.normcase(os.path.abspath(video_path)):
        raise ValueError("Output path must differ from video_path.")
    return candidate


def _temporary_output_path(output: str) -> str:
    parent = os.path.dirname(output) or "."
    stem = Path(output).stem
    suffix = Path(output).suffix or ".mp4"
    fd, path = tempfile.mkstemp(prefix=f".{stem}-", suffix=f".part{suffix}", dir=parent)
    os.close(fd)
    os.unlink(path)
    return path


def _audio_stream_available(path: str) -> Optional[bool]:
    """Return whether *path* has an audio stream, or ``None`` if unknown.

    A video without an audio stream is a valid input for this effect: the
    renderer should preserve the video and report that no analysis windows
    were available.  A non-zero probe result is kept distinct from an empty
    successful result so corrupt or unsupported media still produces a useful
    FFmpeg error instead of being silently treated as silent audio.
    """
    try:
        result = subprocess.run(
            [
                get_ffprobe_path(),
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=index",
                "-of",
                "csv=p=0",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError, RuntimeError):
        return None
    if result.returncode != 0:
        return None
    return bool(result.stdout.strip())


def render(
    video_path: str,
    audio_path: str,
    preset: str = "boom",
    custom_params: Optional[Dict[str, Any]] = None,
    output: Optional[str] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> AudioReactiveResult:
    """Analyze audio and render a deterministic, beat-driven video artifact."""
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    audio_source = audio_path or video_path
    if not os.path.isfile(audio_source):
        raise FileNotFoundError(f"Audio file not found: {audio_source}")
    selected_preset = str(preset or "boom").strip().lower()
    params = _normalise_parameters(selected_preset, custom_params)
    output_path_value = _output_path_for(video_path, output)
    _raise_if_cancelled(is_cancelled)

    if on_progress:
        on_progress(5, "Preparing deterministic PCM analysis")
    from opencut.core.rhythm_effects import analyze_audio_features

    audio_stream = _audio_stream_available(audio_source)
    analysis_source = audio_source if audio_stream is not False else None

    def _analysis_progress(percent: float, message: str) -> None:
        _raise_if_cancelled(is_cancelled)
        if on_progress:
            on_progress(10 + int(max(0, min(100, percent)) * 0.45), str(message))

    if analysis_source is None:
        features = {"amplitude": [], "rms": [], "beats": []}
        if on_progress:
            on_progress(55, "No audio stream found; preserving the video")
    else:
        features = analyze_audio_features(
            analysis_source,
            features=["amplitude", "rms", "beats"],
            window_ms=ANALYSIS_WINDOW_MS,
            sample_rate=ANALYSIS_SAMPLE_RATE,
            on_progress=_analysis_progress,
        )
    _raise_if_cancelled(is_cancelled)
    keyframes, beat_count = _build_keyframes(features, ANALYSIS_WINDOW_MS)
    info = get_video_info(video_path) or {}
    width = max(2, int(float(info.get("width") or 1920)))
    height = max(2, int(float(info.get("height") or 1080)))
    duration = max(0.0, float(info.get("duration") or 0.0))
    video_filter = build_video_filter(keyframes, params, width, height)
    if on_progress:
        on_progress(62, f"Built {len(keyframes)} deterministic keyframes")
    _raise_if_cancelled(is_cancelled)

    temporary_output = _temporary_output_path(output_path_value)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        video_path,
    ]
    if analysis_source is not None:
        command.extend(["-i", analysis_source])
    command.extend(["-filter:v", video_filter, "-map", "0:v:0"])
    if analysis_source is not None:
        command.extend(["-map", "1:a:0?"])
    command.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-fps_mode:v",
            "vfr",
        ]
    )
    if analysis_source is not None:
        command.extend(["-c:a", "aac", "-b:a", "192k", "-shortest"])
    command.extend(["-movflags", "+faststart", temporary_output])
    try:
        if on_progress:
            on_progress(70, "Rendering audio-reactive video")
        run_ffmpeg(command, timeout=3600, job_id=_current_job_id())
        _raise_if_cancelled(is_cancelled)
        os.replace(temporary_output, output_path_value)
    except Exception:
        try:
            if os.path.isfile(temporary_output):
                os.unlink(temporary_output)
        except OSError:
            pass
        raise

    if on_progress:
        on_progress(100, "Audio-reactive render complete")
    analysis = {
        "engine": "pcm_onset",
        "window_ms": ANALYSIS_WINDOW_MS,
        "sample_rate": ANALYSIS_SAMPLE_RATE,
        "frame_count": len(features.get("amplitude") or []),
        "duration": duration,
        "audio_stream": audio_stream is not False,
        "optional_beatnet": {
            "available": check_optional_beatnet_available(),
            "used": False,
        },
    }
    notes = [
        "Rendered with deterministic local PCM/onset analysis; no model download was required.",
    ]
    if audio_stream is False:
        notes.append("The input contained no audio stream; visual effects were evaluated with empty analysis.")
    if not keyframes:
        notes.append("The input audio produced no complete analysis windows; the video was preserved.")
    if video_filter == "null" and keyframes:
        notes.append("All visual parameters were zero, so the output uses a deterministic no-op video filter.")
    return AudioReactiveResult(
        output=output_path_value,
        keyframes=keyframes,
        preset=selected_preset,
        beat_count=beat_count,
        notes=notes,
        analysis=analysis,
        capabilities={
            "deterministic_renderer": True,
            "network_required": False,
            "model_download_required": False,
            "vfr_timestamps": True,
        },
    )


__all__ = [
    "ANALYSIS_SAMPLE_RATE",
    "ANALYSIS_WINDOW_MS",
    "AudioReactiveCancelled",
    "AudioReactiveResult",
    "INSTALL_HINT",
    "MAX_PARAMETER_VALUES",
    "PRESETS",
    "build_video_filter",
    "check_audio_reactive_available",
    "check_optional_beatnet_available",
    "list_presets",
    "render",
    "validate_request_payload",
]
