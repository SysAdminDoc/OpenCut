"""
OpenCut Video Podcast to Audio-Only Module (Feature 40.3)

Strip video from podcast recordings, normalize to -16 LUFS mono,
optionally apply noise reduction, and embed ID3 metadata with artwork.

Functions:
    extract_podcast_audio - Full pipeline: strip video, normalize, noise reduce
    add_id3_metadata      - Embed ID3 tags and cover artwork into an audio file
"""

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from opencut.helpers import (
    FFmpegCmd,
    get_ffmpeg_path,
    get_ffprobe_path,
    get_video_info,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")

# Supported output audio formats
SUPPORTED_FORMATS = {"mp3", "aac", "flac", "wav", "ogg", "opus", "m4a"}

# Default ID3 metadata fields
DEFAULT_METADATA = {
    "title": "",
    "artist": "",
    "album": "",
    "genre": "Podcast",
    "year": "",
    "comment": "Extracted by OpenCut",
    "track": "",
    "episode": "",
    "show": "",
    "description": "",
}


@dataclass
class PodcastAudioResult:
    """Result of podcast audio extraction."""
    output_path: str
    format: str
    duration: float
    sample_rate: int
    channels: int
    loudness_lufs: float
    noise_reduced: bool
    file_size_bytes: int


@dataclass
class MetadataResult:
    """Result of ID3 metadata embedding."""
    output_path: str
    metadata_fields: int
    has_artwork: bool


def _get_audio_info(filepath: str) -> dict:
    """Get audio stream details via ffprobe.

    Returns dict with 'codec', 'sample_rate', 'channels', 'bitrate', 'duration'.
    """
    cmd = [
        get_ffprobe_path(),
        "-v", "quiet",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_name,sample_rate,channels,bit_rate",
        "-show_entries", "format=duration,size",
        "-of", "json",
        filepath,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=30, check=False)
    defaults = {
        "codec": "unknown", "sample_rate": 44100, "channels": 2,
        "bitrate": 0, "duration": 0, "size": 0,
    }
    if result.returncode != 0:
        return defaults
    try:
        data = json.loads(result.stdout.decode())
        streams = data.get("streams", [])
        fmt = data.get("format", {})
        if not streams:
            return defaults
        s = streams[0]
        return {
            "codec": s.get("codec_name", "unknown"),
            "sample_rate": int(s.get("sample_rate", 44100)),
            "channels": int(s.get("channels", 2)),
            "bitrate": int(s.get("bit_rate", 0) or 0),
            "duration": float(fmt.get("duration", 0)),
            "size": int(fmt.get("size", 0)),
        }
    except (json.JSONDecodeError, KeyError, ValueError):
        return defaults


def extract_podcast_audio(
    video_path: str,
    output: Optional[str] = None,
    audio_format: str = "mp3",
    bitrate: str = "192k",
    sample_rate: int = 44100,
    mono: bool = True,
    target_lufs: float = -16.0,
    noise_reduce: bool = False,
    noise_reduce_strength: float = 0.3,
    trim_silence: bool = False,
    on_progress: Optional[Callable] = None,
) -> PodcastAudioResult:
    """Extract audio from a video podcast, normalize, and optionally reduce noise.

    Pipeline:
    1. Strip video track
    2. Convert to target format/sample rate/channels
    3. Apply loudness normalization to target LUFS
    4. Optionally apply noise reduction (FFmpeg afftdn)
    5. Optionally trim leading/trailing silence

    Args:
        video_path: Path to the input video file.
        output: Output file path. Auto-generated if None.
        audio_format: Output format: mp3, aac, flac, wav, ogg, opus, m4a.
        bitrate: Audio bitrate (e.g. "128k", "192k", "320k").
        sample_rate: Output sample rate in Hz.
        mono: If True, output mono (recommended for speech podcasts).
        target_lufs: Target loudness in LUFS (default -16, podcast standard).
        noise_reduce: If True, apply FFmpeg afftdn noise reduction.
        noise_reduce_strength: Noise reduction amount 0.0 to 1.0.
        trim_silence: If True, trim leading and trailing silence.
        on_progress: Optional progress callback(pct, msg).

    Returns:
        PodcastAudioResult with output path and audio details.
    """
    if audio_format.lower().lstrip(".") not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format: {audio_format}. "
            f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    fmt = audio_format.lower().lstrip(".")
    channels = 1 if mono else 2

    if output is None:
        output = output_path(video_path, "podcast")
        # Fix extension
        base = os.path.splitext(output)[0]
        ext_map = {"m4a": ".m4a", "aac": ".m4a", "ogg": ".ogg", "opus": ".opus",
                    "flac": ".flac", "wav": ".wav", "mp3": ".mp3"}
        output = base + ext_map.get(fmt, f".{fmt}")

    if on_progress:
        on_progress(5, "Analyzing source audio...")

    info = get_video_info(video_path)
    source_duration = info.get("duration", 0)

    # Step 1: First pass - loudness analysis via loudnorm (dual-pass for accuracy)
    if on_progress:
        on_progress(10, "Analyzing loudness levels...")

    analyze_cmd = [
        get_ffmpeg_path(), "-hide_banner", "-y",
        "-i", video_path,
        "-vn",
        "-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11:print_format=json",
        "-f", "null",
        "-"
    ]

    analyze_result = subprocess.run(
        analyze_cmd, capture_output=True, timeout=600, check=False
    )
    stderr_text = analyze_result.stderr.decode(errors="replace")

    # Parse loudnorm analysis output
    measured_i = target_lufs
    measured_tp = -1.5
    measured_lra = 11.0
    measured_thresh = -26.0
    target_offset = 0.0

    # Extract JSON block from stderr
    json_start = stderr_text.rfind("{")
    json_end = stderr_text.rfind("}") + 1
    if json_start >= 0 and json_end > json_start:
        try:
            loudness_data = json.loads(stderr_text[json_start:json_end])
            measured_i = float(loudness_data.get("input_i", target_lufs))
            measured_tp = float(loudness_data.get("input_tp", -1.5))
            measured_lra = float(loudness_data.get("input_lra", 11.0))
            measured_thresh = float(loudness_data.get("input_thresh", -26.0))
            target_offset = float(loudness_data.get("target_offset", 0.0))
        except (json.JSONDecodeError, ValueError):
            logger.warning("Could not parse loudnorm analysis, using single-pass normalization")

    # Step 2: Build the processing filter chain
    if on_progress:
        on_progress(30, "Building audio processing pipeline...")

    filters = []

    # Loudness normalization (second pass with measured values)
    filters.append(
        f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11"
        f":measured_I={measured_i}:measured_TP={measured_tp}"
        f":measured_LRA={measured_lra}:measured_thresh={measured_thresh}"
        f":offset={target_offset}:linear=true"
    )

    # Noise reduction
    if noise_reduce:
        nr_amount = int(noise_reduce_strength * 40 + 5)  # Map 0-1 to 5-45 dB
        filters.append(f"afftdn=nf=-{nr_amount}")

    # Trim silence
    if trim_silence:
        filters.append("silenceremove=start_periods=1:start_silence=0.5:start_threshold=-50dB")
        filters.append("areverse,silenceremove=start_periods=1:start_silence=0.5:start_threshold=-50dB,areverse")

    af_chain = ",".join(filters)

    # Step 3: Encode final output
    if on_progress:
        on_progress(40, f"Extracting and normalizing to {fmt.upper()}...")

    # Determine codec
    codec_map = {
        "mp3": ("libmp3lame", bitrate),
        "aac": ("aac", bitrate),
        "m4a": ("aac", bitrate),
        "flac": ("flac", None),
        "wav": ("pcm_s16le", None),
        "ogg": ("libvorbis", bitrate),
        "opus": ("libopus", bitrate),
    }
    codec, br = codec_map.get(fmt, ("libmp3lame", bitrate))

    cmd_builder = (
        FFmpegCmd()
        .input(video_path)
        .no_video()
        .audio_filter(af_chain)
        .audio_codec(codec, bitrate=br)
        .option("ar", str(sample_rate))
        .option("ac", str(channels))
    )

    # Format-specific options
    if fmt in ("m4a", "aac"):
        cmd_builder.option("f", "mp4")
        cmd_builder.faststart()

    cmd_builder.output(output)
    cmd = cmd_builder.build()

    run_ffmpeg(cmd, timeout=1800)

    if on_progress:
        on_progress(90, "Finalizing...")

    # Get output file info
    file_size = os.path.getsize(output) if os.path.isfile(output) else 0

    if on_progress:
        on_progress(100, "Podcast audio extracted")

    return PodcastAudioResult(
        output_path=output,
        format=fmt,
        duration=source_duration,
        sample_rate=sample_rate,
        channels=channels,
        loudness_lufs=target_lufs,
        noise_reduced=noise_reduce,
        file_size_bytes=file_size,
    )


def add_id3_metadata(
    audio_path: str,
    metadata: Dict[str, str],
    artwork_path: Optional[str] = None,
    output: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> MetadataResult:
    """Embed ID3 metadata and optional cover artwork into an audio file.

    Uses FFmpeg to write metadata tags. Supported for MP3, M4A/AAC, FLAC, OGG.

    Args:
        audio_path: Path to the audio file.
        metadata: Dict of metadata fields (title, artist, album, genre, year, etc.).
        artwork_path: Optional path to cover art image (JPG/PNG).
        output: Output file path. In-place if None (creates temp + replaces).
        on_progress: Optional progress callback(pct, msg).

    Returns:
        MetadataResult with output path and metadata summary.
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if on_progress:
        on_progress(10, "Preparing metadata...")

    in_place = output is None
    if in_place:
        fd, output = tempfile.mkstemp(suffix=os.path.splitext(audio_path)[1])
        os.close(fd)

    # Build metadata flags
    cmd_builder = FFmpegCmd().input(audio_path)

    has_artwork = False
    if artwork_path and os.path.isfile(artwork_path):
        cmd_builder.input(artwork_path)
        has_artwork = True

    # Map metadata to FFmpeg -metadata flags
    meta_count = 0
    tag_map = {
        "title": "title",
        "artist": "artist",
        "album": "album",
        "genre": "genre",
        "year": "date",
        "comment": "comment",
        "track": "track",
        "episode": "episode_id",
        "show": "show",
        "description": "description",
    }

    for key, value in metadata.items():
        if value and value.strip():
            ffmpeg_key = tag_map.get(key, key)
            cmd_builder.option("metadata", f"{ffmpeg_key}={value.strip()}")
            meta_count += 1

    if on_progress:
        on_progress(30, f"Writing {meta_count} metadata fields...")

    if has_artwork:
        # Embed artwork as attached picture
        cmd_builder.map("0:a", "1:v")
        cmd_builder.option("c:v", "copy")
        cmd_builder.option("disposition:v:0", "attached_pic")
    else:
        cmd_builder.map("0:a")

    cmd_builder.audio_codec("copy")
    cmd_builder.output(output)
    cmd = cmd_builder.build()

    run_ffmpeg(cmd)

    if in_place:
        # Replace original with tagged version
        try:
            os.replace(output, audio_path)
            output = audio_path
        except OSError as e:
            logger.warning("Could not replace original file: %s", e)

    if on_progress:
        on_progress(95, "Metadata embedded")

    return MetadataResult(
        output_path=output,
        metadata_fields=meta_count,
        has_artwork=has_artwork,
    )
