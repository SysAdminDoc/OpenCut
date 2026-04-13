"""
OpenCut MXF Container Support Module v1.0.0

Professional MXF (Material Exchange Format) workflow:
- Probe MXF files for metadata, timecode, tracks, OP pattern
- Export to MXF with OP1a or OP-Atom packaging
- Convert to MXF with DNxHR/XDCAM codecs
- Preserve or override embedded timecode

All FFmpeg-based, zero external dependencies.
"""

import json
import logging
import os
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import get_ffmpeg_path, get_ffprobe_path, run_ffmpeg

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# MXF Codec Profiles
# ---------------------------------------------------------------------------

MXF_PROFILES = {
    "dnxhr_lb": {"label": "DNxHR LB (Low Bandwidth)", "codec": "dnxhd", "profile": "dnxhr_lb"},
    "dnxhr_sq": {"label": "DNxHR SQ (Standard Quality)", "codec": "dnxhd", "profile": "dnxhr_sq"},
    "dnxhr_hq": {"label": "DNxHR HQ (High Quality)", "codec": "dnxhd", "profile": "dnxhr_hq"},
    "dnxhr_hqx": {"label": "DNxHR HQX (High Quality 10-bit)", "codec": "dnxhd", "profile": "dnxhr_hqx"},
    "dnxhr_444": {"label": "DNxHR 444 (Uncompressed Quality)", "codec": "dnxhd", "profile": "dnxhr_444"},
    "xdcam_hd422": {"label": "XDCAM HD422", "codec": "mpeg2video",
                     "bitrate": "50000k", "flags": ["-pix_fmt", "yuv422p"]},
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class MXFTrack:
    """Single track within an MXF file."""
    index: int
    codec_type: str      # video, audio, data
    codec_name: str
    duration: float
    extra: Dict = field(default_factory=dict)


@dataclass
class MXFInfo:
    """Probe result for an MXF file."""
    file_path: str
    op_pattern: str      # op1a, opatom, unknown
    timecode: str        # SMPTE timecode string
    tracks: List[Dict] = field(default_factory=list)
    essence_type: str = ""    # dnxhd, mpeg2video, prores, etc.
    duration: float = 0.0
    file_size: int = 0
    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def probe_mxf(input_path: str) -> dict:
    """
    Probe an MXF file for detailed metadata.

    Args:
        input_path: Path to MXF file.

    Returns:
        dict representation of MXFInfo with tracks, timecode, OP pattern, etc.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    # Full probe with format + streams + format_tags
    cmd = [
        get_ffprobe_path(), "-v", "quiet",
        "-show_format", "-show_streams",
        "-of", "json", input_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=30)

    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for: {input_path}")

    data = json.loads(result.stdout.decode())
    fmt = data.get("format", {})
    streams = data.get("streams", [])
    tags = fmt.get("tags", {})

    # Detect OP pattern from format name or metadata
    format_name = fmt.get("format_name", "")
    op_pattern = "unknown"
    if "mxf_opatom" in format_name:
        op_pattern = "opatom"
    elif "mxf" in format_name:
        op_pattern = "op1a"

    # Extract timecode from tags or streams
    timecode = (
        tags.get("timecode", "")
        or tags.get("material_package_timecode", "")
        or tags.get("source_package_timecode", "")
    )
    # Also check data streams for timecode
    if not timecode:
        for s in streams:
            if s.get("codec_type") == "data" and s.get("codec_tag_string") == "tmcd":
                tc_tags = s.get("tags", {})
                timecode = tc_tags.get("timecode", "")
                if timecode:
                    break

    # Build track list
    tracks = []
    essence_type = ""
    for s in streams:
        codec_type = s.get("codec_type", "unknown")
        codec_name = s.get("codec_name", "unknown")
        dur = float(s.get("duration", 0))
        if dur <= 0:
            dur = float(fmt.get("duration", 0))

        track = MXFTrack(
            index=int(s.get("index", 0)),
            codec_type=codec_type,
            codec_name=codec_name,
            duration=dur,
            extra={
                "width": s.get("width"),
                "height": s.get("height"),
                "sample_rate": s.get("sample_rate"),
                "channels": s.get("channels"),
                "pix_fmt": s.get("pix_fmt"),
                "profile": s.get("profile"),
                "bit_rate": s.get("bit_rate"),
            },
        )
        tracks.append(asdict(track))

        if codec_type == "video" and not essence_type:
            essence_type = codec_name

    file_size = int(fmt.get("size", 0))
    if file_size == 0:
        try:
            file_size = os.path.getsize(input_path)
        except OSError:
            pass

    duration = float(fmt.get("duration", 0))

    info = MXFInfo(
        file_path=input_path,
        op_pattern=op_pattern,
        timecode=timecode,
        tracks=tracks,
        essence_type=essence_type,
        duration=duration,
        file_size=file_size,
        metadata={k: v for k, v in tags.items()},
    )
    return asdict(info)


def export_mxf(
    input_path: str,
    output_path_override: Optional[str] = None,
    op_pattern: str = "op1a",
    audio_tracks: Optional[List[int]] = None,
    timecode: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Export/rewrap a video file into MXF container.

    Args:
        input_path: Path to source video.
        output_path_override: Custom output path.
        op_pattern: "op1a" (interleaved, most common) or "opatom" (Avid-compatible).
        audio_tracks: List of audio track indices to include. None = all.
        timecode: Override timecode (e.g., "01:00:00:00"). None = preserve.
        on_progress: Callback(percent, message).

    Returns:
        dict with output_path, op_pattern, timecode.
    """
    if op_pattern not in ("op1a", "opatom"):
        raise ValueError(f"Unknown OP pattern '{op_pattern}'. Use: op1a, opatom")

    out = output_path_override
    if not out:
        base = os.path.splitext(os.path.basename(input_path))[0]
        directory = os.path.dirname(input_path)
        out = os.path.join(directory, f"{base}_{op_pattern}.mxf")

    if on_progress:
        on_progress(10, f"Exporting to MXF ({op_pattern})...")

    # Determine MXF muxer
    mxf_format = "mxf_opatom" if op_pattern == "opatom" else "mxf"

    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
    ]

    # Map streams
    if audio_tracks is not None:
        cmd += ["-map", "0:v"]
        for track_idx in audio_tracks:
            cmd += ["-map", f"0:a:{track_idx}"]
    else:
        cmd += ["-map", "0"]

    # Copy codecs (rewrap)
    cmd += ["-c", "copy"]

    # Timecode
    if timecode:
        cmd += ["-timecode", timecode]

    cmd += ["-f", mxf_format, out]

    if on_progress:
        on_progress(30, "Muxing...")

    run_ffmpeg(cmd, timeout=7200)

    # Read back timecode from output
    actual_tc = timecode or ""
    if not actual_tc:
        try:
            probe = probe_mxf(out)
            actual_tc = probe.get("timecode", "")
        except Exception:
            pass

    if on_progress:
        on_progress(100, f"Exported MXF ({op_pattern})")

    return {
        "output_path": out,
        "op_pattern": op_pattern,
        "timecode": actual_tc,
    }


def convert_to_mxf(
    input_path: str,
    codec: str = "dnxhd",
    profile: str = "dnxhr_hq",
    output_path_override: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Transcode a video to DNxHR or XDCAM in an MXF container.

    Args:
        input_path: Path to source video.
        codec: Target codec ("dnxhd" or "xdcam").
        profile: Codec profile (see MXF_PROFILES).
        output_path_override: Custom output path.
        on_progress: Callback(percent, message).

    Returns:
        dict with output_path, codec, profile.
    """
    # Resolve profile
    profile_key = profile
    if profile_key not in MXF_PROFILES:
        # Try codec-based lookup
        if codec == "xdcam":
            profile_key = "xdcam_hd422"
        elif codec == "dnxhd":
            profile_key = "dnxhr_hq"
        else:
            raise ValueError(
                f"Unknown profile '{profile}'. "
                f"Available: {', '.join(MXF_PROFILES.keys())}"
            )

    prof = MXF_PROFILES[profile_key]

    out = output_path_override
    if not out:
        base = os.path.splitext(os.path.basename(input_path))[0]
        directory = os.path.dirname(input_path)
        out = os.path.join(directory, f"{base}_{profile_key}.mxf")

    if on_progress:
        on_progress(10, f"Transcoding to {prof['label']}...")

    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
    ]

    # Video codec settings
    if prof["codec"] == "dnxhd":
        cmd += ["-c:v", "dnxhd", "-profile:v", prof["profile"]]
        # DNxHR needs specific pixel format
        if "444" in profile_key:
            cmd += ["-pix_fmt", "yuv444p10le"]
        elif "hqx" in profile_key:
            cmd += ["-pix_fmt", "yuv422p10le"]
        else:
            cmd += ["-pix_fmt", "yuv422p"]
    elif prof["codec"] == "mpeg2video":
        cmd += ["-c:v", "mpeg2video"]
        if "bitrate" in prof:
            cmd += ["-b:v", prof["bitrate"]]
        if "flags" in prof:
            cmd += prof["flags"]

    # Audio: PCM for MXF compatibility
    cmd += ["-c:a", "pcm_s16le"]

    # MXF container
    cmd += ["-f", "mxf", out]

    if on_progress:
        on_progress(25, "Encoding...")

    run_ffmpeg(cmd, timeout=14400)  # Longer timeout for transcoding

    if on_progress:
        on_progress(100, f"Converted to {prof['label']} MXF")

    return {
        "output_path": out,
        "codec": prof["codec"],
        "profile": profile_key,
        "profile_label": prof["label"],
    }
