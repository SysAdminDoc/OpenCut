"""
OpenCut Delivery Validation Module (70.3)

Automated delivery validation suite. Check exported files against
delivery specifications for video, audio, container, and subtitle
compliance. Built-in specs for Netflix, YouTube, Broadcast (EBU),
DCP, IMF, Apple TV+, and Amazon.
"""

import json
import logging
import os
import re
import subprocess as _sp
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Any

from opencut.helpers import (
    get_ffmpeg_path,
    get_ffprobe_path,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """Result of a single delivery check."""
    check_name: str = ""
    category: str = ""         # "video", "audio", "container", "subtitle"
    field_name: str = ""
    passed: bool = False
    expected: str = ""
    actual: str = ""
    violation: str = ""
    severity: str = "error"    # "error", "warning", "info"
    required: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DeliverySpec:
    """A delivery specification with check definitions."""
    name: str = ""
    display_name: str = ""
    description: str = ""
    checks: List[Dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ValidationResult:
    """Full delivery validation result."""
    file_path: str = ""
    spec_name: str = ""
    spec_display_name: str = ""
    passed: bool = False
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    warning_checks: int = 0
    errors: List[CheckResult] = field(default_factory=list)
    warnings: List[CheckResult] = field(default_factory=list)
    info: List[CheckResult] = field(default_factory=list)
    all_results: List[CheckResult] = field(default_factory=list)
    file_info: Dict = field(default_factory=dict)
    verdict: str = ""          # "PASS", "FAIL", "WARN"

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "spec_name": self.spec_name,
            "spec_display_name": self.spec_display_name,
            "passed": self.passed,
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "warning_checks": self.warning_checks,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "info": [i.to_dict() for i in self.info],
            "all_results": [r.to_dict() for r in self.all_results],
            "file_info": self.file_info,
            "verdict": self.verdict,
        }


# ---------------------------------------------------------------------------
# Built-In Delivery Specs (validation check definitions)
# ---------------------------------------------------------------------------
# Each check: (category, field_name, operator, value, severity, description)
# operator: "eq", "neq", "gte", "lte", "in", "range", "regex", "exists"

DELIVERY_SPECS: Dict[str, Dict] = {
    "netflix": {
        "display_name": "Netflix",
        "description": "Netflix delivery specifications for original/licensed content",
        "checks": [
            ("video", "codec", "in", ["h264", "h265", "prores", "prores_ks"], "error",
             "Video codec must be H.264, H.265, or ProRes"),
            ("video", "width", "gte", 1920, "error",
             "Minimum 1920px width (Full HD)"),
            ("video", "height", "gte", 1080, "error",
             "Minimum 1080px height (Full HD)"),
            ("video", "frame_rate", "in", [23.976, 24, 25, 29.97, 30, 50, 59.94, 60],
             "error", "Standard frame rate required"),
            ("video", "bitrate", "gte", 10_000_000, "error",
             "Minimum video bitrate 10 Mbps"),
            ("video", "bitrate", "lte", 100_000_000, "error",
             "Maximum video bitrate 100 Mbps"),
            ("video", "color_space", "in", ["bt709", "bt2020", "unknown"], "warning",
             "Color space should be BT.709 or BT.2020"),
            ("audio", "codec", "in", ["aac", "pcm_s24le", "pcm_s16le", "eac3", "ac3"],
             "error", "Audio must be AAC, PCM, or E-AC-3"),
            ("audio", "sample_rate", "eq", 48000, "error",
             "Audio sample rate must be 48kHz"),
            ("audio", "channels", "gte", 2, "error",
             "Minimum stereo audio required"),
            ("audio", "loudness_integrated", "range", [-27.0, -14.0], "warning",
             "Integrated loudness should be -27 to -14 LUFS"),
            ("audio", "loudness_true_peak", "lte", -1.0, "warning",
             "True peak should not exceed -1.0 dBTP"),
            ("container", "format", "in", ["mov", "mxf", "mp4"], "error",
             "Container must be MOV, MXF, or MP4"),
            ("container", "has_timecode", "eq", True, "warning",
             "Timecode track recommended"),
        ],
    },
    "youtube": {
        "display_name": "YouTube",
        "description": "YouTube recommended upload specifications",
        "checks": [
            ("video", "codec", "in", ["h264", "h265", "vp9", "av1", "vp8"],
             "error", "Video codec must be H.264, H.265, VP9, or AV1"),
            ("video", "width", "gte", 1280, "warning",
             "Minimum 720p width recommended"),
            ("video", "frame_rate", "in", [24, 25, 30, 48, 50, 60], "error",
             "Standard frame rate required"),
            ("video", "bitrate", "gte", 5_000_000, "warning",
             "Minimum 5 Mbps recommended for 1080p"),
            ("video", "bitrate", "lte", 200_000_000, "error",
             "Maximum 200 Mbps"),
            ("audio", "codec", "in", ["aac", "opus", "vorbis", "mp3"], "error",
             "Audio codec must be AAC, Opus, Vorbis, or MP3"),
            ("audio", "sample_rate", "in", [44100, 48000], "error",
             "Sample rate 44.1kHz or 48kHz"),
            ("audio", "channels", "gte", 1, "error",
             "At least mono audio required"),
            ("container", "format", "in", ["mp4", "mov", "webm", "mkv", "avi"],
             "error", "Container must be MP4, MOV, WebM, MKV, or AVI"),
        ],
    },
    "broadcast_ebu": {
        "display_name": "Broadcast (EBU R128)",
        "description": "European Broadcasting Union R128 broadcast specifications",
        "checks": [
            ("video", "codec", "in", ["h264", "mpeg2video", "prores", "prores_ks",
                                       "dnxhd", "dnxhr"], "error",
             "H.264, MPEG-2, ProRes, or DNxHD/HR required"),
            ("video", "width", "gte", 1920, "error",
             "Full HD minimum required"),
            ("video", "height", "gte", 1080, "error",
             "Full HD minimum required"),
            ("video", "frame_rate", "in", [25, 50], "error",
             "25 or 50fps for PAL broadcast"),
            ("video", "bitrate", "gte", 35_000_000, "error",
             "Minimum 35 Mbps for broadcast"),
            ("video", "interlacing", "in", ["progressive", "tt", "unknown"],
             "warning", "Progressive or top-field-first preferred"),
            ("audio", "codec", "in", ["pcm_s24le", "pcm_s16le", "pcm_s32le"],
             "error", "PCM audio required for broadcast"),
            ("audio", "sample_rate", "eq", 48000, "error",
             "48kHz required"),
            ("audio", "channels", "gte", 2, "error",
             "Minimum stereo"),
            ("audio", "loudness_integrated", "range", [-24.0, -22.0], "error",
             "Integrated loudness must be -23 LUFS +/- 1 LU (EBU R128)"),
            ("audio", "loudness_true_peak", "lte", -1.0, "error",
             "True peak max -1.0 dBTP"),
            ("container", "format", "in", ["mxf"], "error",
             "MXF container required"),
            ("container", "has_timecode", "eq", True, "error",
             "Timecode track required"),
        ],
    },
    "dcp": {
        "display_name": "DCP (Digital Cinema)",
        "description": "Digital Cinema Package per SMPTE DCI",
        "checks": [
            ("video", "codec", "in", ["jpeg2000", "libopenjpeg"], "error",
             "JPEG2000 required for DCP"),
            ("video", "width", "in", [2048, 1998, 4096, 3996], "error",
             "DCI resolution required (2K or 4K)"),
            ("video", "height", "in", [1080, 858, 2160, 1716], "error",
             "DCI height required"),
            ("video", "frame_rate", "in", [24, 48], "error",
             "24 or 48fps for DCP"),
            ("video", "pix_fmt", "in", ["xyz12le", "xyz12be", "rgb48le"], "warning",
             "XYZ colorspace recommended"),
            ("audio", "codec", "in", ["pcm_s24le"], "error",
             "24-bit PCM required"),
            ("audio", "sample_rate", "eq", 48000, "error",
             "48kHz required"),
            ("audio", "channels", "gte", 6, "error",
             "Minimum 5.1 surround"),
            ("container", "format", "in", ["mxf"], "error",
             "MXF container required"),
        ],
    },
    "imf": {
        "display_name": "IMF (Interoperable Master Format)",
        "description": "SMPTE ST 2067 Interoperable Master Format",
        "checks": [
            ("video", "codec", "in", ["jpeg2000", "libopenjpeg", "h265", "hevc"],
             "error", "JPEG2000 or HEVC required for IMF"),
            ("video", "width", "gte", 1920, "error",
             "Minimum Full HD width"),
            ("video", "height", "gte", 1080, "error",
             "Minimum Full HD height"),
            ("video", "frame_rate", "in", [24, 25, 30, 48, 50, 60], "error",
             "Standard frame rate required"),
            ("audio", "codec", "in", ["pcm_s24le", "pcm_s16le"], "error",
             "PCM audio required"),
            ("audio", "sample_rate", "in", [48000, 96000], "error",
             "48kHz or 96kHz required"),
            ("container", "format", "in", ["mxf"], "error",
             "MXF container required"),
            ("container", "has_timecode", "eq", True, "warning",
             "Timecode track recommended"),
        ],
    },
    "apple_tv_plus": {
        "display_name": "Apple TV+",
        "description": "Apple TV+ delivery specifications",
        "checks": [
            ("video", "codec", "in", ["h265", "hevc", "prores", "prores_ks",
                                       "prores_4444"], "error",
             "H.265 or ProRes required"),
            ("video", "width", "gte", 3840, "error",
             "Minimum 4K UHD width"),
            ("video", "height", "gte", 2160, "error",
             "Minimum 4K UHD height"),
            ("video", "frame_rate", "in", [23.976, 24, 25, 29.97, 50, 59.94],
             "error", "Standard cinema/broadcast frame rate"),
            ("video", "bit_depth", "gte", 10, "error",
             "10-bit minimum for HDR content"),
            ("audio", "codec", "in", ["pcm_s24le", "aac", "eac3", "ac4"],
             "error", "PCM, AAC, E-AC-3, or AC-4"),
            ("audio", "sample_rate", "eq", 48000, "error",
             "48kHz required"),
            ("audio", "channels", "gte", 2, "error",
             "Minimum stereo; 5.1/7.1/Atmos preferred"),
            ("audio", "loudness_integrated", "range", [-27.0, -14.0], "warning",
             "Integrated loudness -27 to -14 LUFS"),
            ("container", "format", "in", ["mov", "mxf"], "error",
             "MOV or MXF container"),
            ("container", "has_timecode", "eq", True, "error",
             "Timecode track required"),
        ],
    },
    "amazon": {
        "display_name": "Amazon Prime Video",
        "description": "Amazon Prime Video delivery specifications",
        "checks": [
            ("video", "codec", "in", ["h264", "h265", "hevc", "prores", "prores_ks"],
             "error", "H.264, H.265, or ProRes"),
            ("video", "width", "gte", 1920, "error",
             "Minimum Full HD width"),
            ("video", "height", "gte", 1080, "error",
             "Minimum Full HD height"),
            ("video", "frame_rate", "in", [23.976, 24, 25, 29.97, 30],
             "error", "Standard frame rate"),
            ("video", "bitrate", "gte", 15_000_000, "error",
             "Minimum 15 Mbps for HD"),
            ("video", "bitrate", "lte", 100_000_000, "error",
             "Maximum 100 Mbps"),
            ("audio", "codec", "in", ["aac", "pcm_s24le", "pcm_s16le", "eac3", "ac3"],
             "error", "AAC, PCM, E-AC-3, or AC-3"),
            ("audio", "sample_rate", "eq", 48000, "error",
             "48kHz required"),
            ("audio", "channels", "gte", 2, "error",
             "Minimum stereo"),
            ("audio", "loudness_integrated", "range", [-27.0, -14.0], "warning",
             "Integrated loudness -27 to -14 LUFS"),
            ("container", "format", "in", ["mov", "mxf", "mp4"], "error",
             "MOV, MXF, or MP4"),
            ("container", "has_timecode", "eq", True, "warning",
             "Timecode recommended"),
        ],
    },
}


# ---------------------------------------------------------------------------
# Media Probe Helpers
# ---------------------------------------------------------------------------

def _probe_video(filepath: str) -> dict:
    """Probe video stream details via ffprobe."""
    cmd = [
        get_ffprobe_path(), "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,bit_rate,pix_fmt,"
        "color_space,color_transfer,color_primaries,field_order,"
        "bits_per_raw_sample,nb_frames,duration,profile",
        "-of", "json", filepath,
    ]
    result = _sp.run(cmd, capture_output=True, timeout=60)
    info = {
        "codec": "unknown", "width": 0, "height": 0, "frame_rate": 0.0,
        "bitrate": 0, "pix_fmt": "unknown", "color_space": "unknown",
        "color_transfer": "unknown", "color_primaries": "unknown",
        "interlacing": "unknown", "bit_depth": 8, "profile": "",
    }
    if result.returncode != 0:
        return info
    try:
        data = json.loads(result.stdout.decode())
        streams = data.get("streams", [])
        if not streams:
            return info
        s = streams[0]
        info["codec"] = s.get("codec_name", "unknown")
        info["width"] = int(s.get("width", 0) or 0)
        info["height"] = int(s.get("height", 0) or 0)

        rfr = s.get("r_frame_rate", "0/1")
        if "/" in str(rfr):
            num, den = rfr.split("/")
            fps_raw = float(num) / float(den) if float(den) else 0.0
            # Round to common frame rates for comparison
            info["frame_rate"] = _snap_frame_rate(fps_raw)
        else:
            info["frame_rate"] = _snap_frame_rate(float(rfr))

        bitrate_str = s.get("bit_rate", "0")
        info["bitrate"] = int(bitrate_str) if bitrate_str and bitrate_str != "N/A" else 0
        info["pix_fmt"] = s.get("pix_fmt", "unknown")
        info["color_space"] = s.get("color_space", "unknown")
        info["color_transfer"] = s.get("color_transfer", "unknown")
        info["color_primaries"] = s.get("color_primaries", "unknown")

        field_order = s.get("field_order", "progressive")
        info["interlacing"] = field_order if field_order else "unknown"

        bps = s.get("bits_per_raw_sample", "")
        info["bit_depth"] = int(bps) if bps and bps != "N/A" else 8
        info["profile"] = s.get("profile", "")
    except (ValueError, KeyError, TypeError) as exc:
        logger.warning("Video probe parse error: %s", exc)
    return info


def _snap_frame_rate(fps: float) -> float:
    """Snap a raw FPS value to nearest standard frame rate."""
    standard_rates = [23.976, 24, 25, 29.97, 30, 48, 50, 59.94, 60]
    if fps <= 0:
        return 0.0
    closest = min(standard_rates, key=lambda r: abs(r - fps))
    if abs(closest - fps) < 0.1:
        return closest
    return round(fps, 3)


def _probe_audio(filepath: str) -> dict:
    """Probe audio stream details via ffprobe."""
    cmd = [
        get_ffprobe_path(), "-v", "quiet",
        "-select_streams", "a:0",
        "-show_entries",
        "stream=codec_name,channels,sample_rate,bit_rate,"
        "bits_per_raw_sample,channel_layout,duration",
        "-of", "json", filepath,
    ]
    result = _sp.run(cmd, capture_output=True, timeout=30)
    info = {
        "codec": "none", "channels": 0, "sample_rate": 0,
        "bitrate": 0, "bit_depth": 0, "channel_layout": "",
        "has_audio": False,
    }
    if result.returncode != 0:
        return info
    try:
        data = json.loads(result.stdout.decode())
        streams = data.get("streams", [])
        if not streams:
            return info
        s = streams[0]
        info["has_audio"] = True
        info["codec"] = s.get("codec_name", "unknown")
        info["channels"] = int(s.get("channels", 0) or 0)
        info["sample_rate"] = int(s.get("sample_rate", 0) or 0)
        br = s.get("bit_rate", "0")
        info["bitrate"] = int(br) if br and br != "N/A" else 0
        bps = s.get("bits_per_raw_sample", "")
        info["bit_depth"] = int(bps) if bps and bps != "N/A" else 0
        info["channel_layout"] = s.get("channel_layout", "")
    except (ValueError, KeyError, TypeError) as exc:
        logger.warning("Audio probe parse error: %s", exc)
    return info


def _probe_container(filepath: str) -> dict:
    """Probe container/format level info via ffprobe."""
    cmd = [
        get_ffprobe_path(), "-v", "quiet",
        "-show_entries",
        "format=format_name,format_long_name,duration,size,bit_rate,"
        "nb_streams,nb_programs",
        "-show_entries", "format_tags=timecode,creation_time",
        "-of", "json", filepath,
    ]
    result = _sp.run(cmd, capture_output=True, timeout=30)
    info = {
        "format": "unknown", "format_long": "", "duration": 0.0,
        "size": 0, "bitrate": 0, "nb_streams": 0,
        "has_timecode": False, "creation_time": "",
    }
    if result.returncode != 0:
        return info
    try:
        data = json.loads(result.stdout.decode())
        fmt = data.get("format", {})
        format_name = fmt.get("format_name", "unknown")
        # Normalize common format names
        info["format"] = _normalize_format(format_name)
        info["format_long"] = fmt.get("format_long_name", "")
        info["duration"] = float(fmt.get("duration", 0) or 0)
        info["size"] = int(fmt.get("size", 0) or 0)
        br = fmt.get("bit_rate", "0")
        info["bitrate"] = int(br) if br and br != "N/A" else 0
        info["nb_streams"] = int(fmt.get("nb_streams", 0) or 0)

        tags = fmt.get("tags", {})
        info["has_timecode"] = "timecode" in tags
        info["creation_time"] = tags.get("creation_time", "")
    except (ValueError, KeyError, TypeError) as exc:
        logger.warning("Container probe parse error: %s", exc)
    return info


def _normalize_format(format_name: str) -> str:
    """Normalize ffprobe format names to common names."""
    # Check simple/exact names first
    if format_name in ("mp4", "mov", "mxf", "mkv", "webm", "avi", "ts"):
        return format_name
    # Multi-format strings from ffprobe
    mapping = {
        "mov,mp4,m4a,3gp,3g2,mj2": "mov",
        "matroska,webm": "mkv",
        "mpegts": "ts",
    }
    for key, val in mapping.items():
        if format_name == key:
            return val
    return format_name


def _measure_loudness(filepath: str) -> dict:
    """Measure audio loudness using FFmpeg loudnorm filter (EBU R128)."""
    cmd = [
        get_ffmpeg_path(),
        "-i", filepath,
        "-af", "loudnorm=print_format=json",
        "-f", "null",
        "-y",
        os.devnull,
    ]
    result = _sp.run(cmd, capture_output=True, timeout=300)
    loudness = {
        "integrated": None,
        "true_peak": None,
        "lra": None,
        "threshold": None,
    }
    if result.returncode != 0:
        return loudness

    stderr = result.stderr.decode(errors="replace")
    # Parse JSON output from loudnorm filter
    json_match = re.search(r'\{[^}]*"input_i"[^}]*\}', stderr, re.DOTALL)
    if not json_match:
        return loudness
    try:
        ldata = json.loads(json_match.group())
        val = ldata.get("input_i", "")
        loudness["integrated"] = float(val) if val and val != "-inf" else None
        val = ldata.get("input_tp", "")
        loudness["true_peak"] = float(val) if val and val != "-inf" else None
        val = ldata.get("input_lra", "")
        loudness["lra"] = float(val) if val and val != "-inf" else None
        val = ldata.get("input_thresh", "")
        loudness["threshold"] = float(val) if val and val != "-inf" else None
    except (ValueError, json.JSONDecodeError):
        pass
    return loudness


# ---------------------------------------------------------------------------
# Check Evaluation Engine
# ---------------------------------------------------------------------------

def _evaluate_check(
    check_def: tuple,
    file_info: dict,
) -> CheckResult:
    """Evaluate a single check against probed file info.

    Args:
        check_def: Tuple of (category, field_name, operator, value, severity, desc).
        file_info: Combined probe results.

    Returns:
        CheckResult.
    """
    category, field_name, operator, expected, severity, description = check_def

    # Resolve actual value from file_info
    actual = _resolve_field(file_info, category, field_name)

    cr = CheckResult(
        check_name=f"{category}.{field_name}",
        category=category,
        field_name=field_name,
        expected=str(expected),
        actual=str(actual),
        severity=severity,
    )

    if actual is None:
        cr.passed = False
        cr.violation = f"Could not determine {category}.{field_name}"
        return cr

    try:
        passed = _evaluate_operator(operator, actual, expected)
    except Exception as exc:
        cr.passed = False
        cr.violation = f"Check evaluation error: {exc}"
        return cr

    cr.passed = passed
    if not passed:
        cr.violation = description
    return cr


def _resolve_field(file_info: dict, category: str, field_name: str) -> Any:
    """Resolve a field value from the combined file_info dict."""
    # Prefix-based lookup
    prefix_map = {
        "video": "video",
        "audio": "audio",
        "container": "container",
        "subtitle": "subtitle",
    }
    prefix = prefix_map.get(category, category)
    info = file_info.get(prefix, {})

    # Direct lookup
    if field_name in info:
        return info[field_name]

    # Handle loudness fields from audio
    if category == "audio":
        loudness = file_info.get("loudness", {})
        if field_name == "loudness_integrated":
            return loudness.get("integrated")
        if field_name == "loudness_true_peak":
            return loudness.get("true_peak")
        if field_name == "loudness_range":
            return loudness.get("lra")

    return None


def _evaluate_operator(operator: str, actual: Any, expected: Any) -> bool:
    """Evaluate a comparison operator."""
    if operator == "eq":
        if isinstance(expected, bool):
            return bool(actual) == expected
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            return abs(float(actual) - float(expected)) < 0.01
        return actual == expected

    if operator == "neq":
        return actual != expected

    if operator == "gte":
        return float(actual) >= float(expected)

    if operator == "lte":
        return float(actual) <= float(expected)

    if operator == "in":
        if isinstance(expected, list):
            # Handle numeric approximate matching
            if isinstance(actual, (int, float)):
                return any(
                    abs(float(actual) - float(e)) < 0.1
                    for e in expected
                    if isinstance(e, (int, float))
                )
            return actual in expected
        return actual in expected

    if operator == "range":
        if isinstance(expected, (list, tuple)) and len(expected) == 2:
            lo, hi = float(expected[0]), float(expected[1])
            return lo <= float(actual) <= hi
        return False

    if operator == "regex":
        return bool(re.match(str(expected), str(actual)))

    if operator == "exists":
        return actual is not None and actual != "" and actual != 0

    logger.warning("Unknown operator: %s", operator)
    return False


# ---------------------------------------------------------------------------
# Subtitle Validation
# ---------------------------------------------------------------------------

def _validate_subtitles(filepath: str) -> dict:
    """Check for subtitle streams in the file."""
    cmd = [
        get_ffprobe_path(), "-v", "quiet",
        "-select_streams", "s",
        "-show_entries", "stream=codec_name,codec_type",
        "-of", "json", filepath,
    ]
    result = _sp.run(cmd, capture_output=True, timeout=30)
    info = {"has_subtitles": False, "subtitle_codec": "", "subtitle_count": 0}
    if result.returncode != 0:
        return info
    try:
        data = json.loads(result.stdout.decode())
        streams = data.get("streams", [])
        info["subtitle_count"] = len(streams)
        info["has_subtitles"] = len(streams) > 0
        if streams:
            info["subtitle_codec"] = streams[0].get("codec_name", "")
    except (ValueError, json.JSONDecodeError):
        pass
    return info


# ---------------------------------------------------------------------------
# Probe video bitrate via format if stream didn't report it
# ---------------------------------------------------------------------------

def _estimate_video_bitrate(filepath: str, container_info: dict,
                            audio_info: dict) -> int:
    """Estimate video bitrate from container total minus audio."""
    total = container_info.get("bitrate", 0)
    audio_br = audio_info.get("bitrate", 0)
    if total > 0 and audio_br >= 0:
        estimate = total - audio_br
        return max(estimate, 0)
    # Fallback: compute from file size and duration
    size = container_info.get("size", 0)
    duration = container_info.get("duration", 0)
    if size > 0 and duration > 0:
        return int((size * 8) / duration)
    return 0


# ---------------------------------------------------------------------------
# Main Validation Function
# ---------------------------------------------------------------------------

def validate_delivery(
    file_path: str,
    spec_name: str = "netflix",
    measure_loudness: bool = True,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Validate a media file against a delivery specification.

    Args:
        file_path: Path to the media file to validate.
        spec_name: Name of delivery spec (e.g. "netflix", "youtube").
        measure_loudness: Whether to measure audio loudness (slower).
        on_progress: Optional callback ``(percent, message)``.

    Returns:
        ValidationResult.to_dict()

    Raises:
        FileNotFoundError: If file_path does not exist.
        ValueError: If spec_name is not recognized.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    spec_key = spec_name.lower().strip()

    # Check built-in specs first, then try delivery_spec module
    spec = DELIVERY_SPECS.get(spec_key)
    if spec is None:
        try:
            from opencut.core.delivery_spec import get_spec as _get_spec
            ext_spec = _get_spec(spec_key)
            if ext_spec and "requirements" in ext_spec:
                # Convert delivery_spec format to our check format
                spec = _convert_ext_spec(ext_spec)
        except ImportError:
            pass

    if spec is None:
        available = sorted(DELIVERY_SPECS.keys())
        raise ValueError(f"Unknown spec '{spec_name}'. Available: {available}")

    if on_progress:
        on_progress(5, "Probing video stream...")

    video_info = _probe_video(file_path)

    if on_progress:
        on_progress(20, "Probing audio stream...")

    audio_info = _probe_audio(file_path)

    if on_progress:
        on_progress(35, "Probing container...")

    container_info = _probe_container(file_path)

    # Fill in missing video bitrate from container
    if video_info["bitrate"] == 0:
        video_info["bitrate"] = _estimate_video_bitrate(
            file_path, container_info, audio_info
        )

    if on_progress:
        on_progress(45, "Checking subtitles...")

    subtitle_info = _validate_subtitles(file_path)

    # Loudness measurement (optional, can be slow)
    loudness_info = {}
    if measure_loudness:
        if on_progress:
            on_progress(50, "Measuring audio loudness (EBU R128)...")
        loudness_info = _measure_loudness(file_path)
    else:
        loudness_info = {
            "integrated": None, "true_peak": None,
            "lra": None, "threshold": None,
        }

    if on_progress:
        on_progress(75, "Evaluating checks...")

    # Combined file info
    file_info = {
        "video": video_info,
        "audio": audio_info,
        "container": container_info,
        "subtitle": subtitle_info,
        "loudness": loudness_info,
    }

    # Run all checks
    all_results = []
    for check_def in spec["checks"]:
        cr = _evaluate_check(check_def, file_info)
        all_results.append(cr)

    # Categorize results
    errors_list = [r for r in all_results if not r.passed and r.severity == "error"]
    warnings_list = [r for r in all_results if not r.passed and r.severity == "warning"]
    info_list = [r for r in all_results if not r.passed and r.severity == "info"]
    passed_list = [r for r in all_results if r.passed]

    has_errors = len(errors_list) > 0
    has_warnings = len(warnings_list) > 0

    if has_errors:
        verdict = "FAIL"
    elif has_warnings:
        verdict = "WARN"
    else:
        verdict = "PASS"

    if on_progress:
        on_progress(100, f"Validation complete: {verdict}")

    result = ValidationResult(
        file_path=file_path,
        spec_name=spec_key,
        spec_display_name=spec.get("display_name", spec_key),
        passed=not has_errors,
        total_checks=len(all_results),
        passed_checks=len(passed_list),
        failed_checks=len(errors_list),
        warning_checks=len(warnings_list),
        errors=errors_list,
        warnings=warnings_list,
        info=info_list,
        all_results=all_results,
        file_info=file_info,
        verdict=verdict,
    )

    logger.info("Delivery validation: %s -> %s (%d/%d checks passed)",
                os.path.basename(file_path), verdict,
                len(passed_list), len(all_results))

    return result.to_dict()


def _convert_ext_spec(ext_spec: dict) -> dict:
    """Convert a delivery_spec.DeliveryProfile dict to internal check format."""
    checks = []
    for req in ext_spec.get("requirements", []):
        cat = req.get("category", "")
        fname = req.get("field_name", "")
        op = req.get("operator", "eq")
        val = req.get("value")
        sev = req.get("severity", "error")
        desc = req.get("description", f"{cat}.{fname} check")
        checks.append((cat, fname, op, val, sev, desc))
    return {
        "display_name": ext_spec.get("display_name", ""),
        "description": ext_spec.get("description", ""),
        "checks": checks,
    }


def list_delivery_specs() -> List[dict]:
    """List all available built-in delivery specs.

    Returns:
        List of dicts with name, display_name, description, check_count.
    """
    return [
        {
            "name": name,
            "display_name": spec["display_name"],
            "description": spec["description"],
            "check_count": len(spec["checks"]),
        }
        for name, spec in sorted(DELIVERY_SPECS.items())
    ]
