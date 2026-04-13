"""
OpenCut Multi-Track ISO Ingest (31.1)

Import multi-camera ISO (isolated) recordings, auto-sync via
timecode or audio cross-correlation, and create a multitrack
timeline for NLE editing.

Supports:
- Timecode-based sync (reads embedded SMPTE timecode via ffprobe)
- Audio-based sync (cross-correlation fallback)
- Multi-track timeline generation as FCP XML

All via FFmpeg / ffprobe for media analysis.
"""

import json
import logging
import os
import struct
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Callable, List, Optional
from xml.dom import minidom

from opencut.helpers import (
    get_ffmpeg_path,
    get_ffprobe_path,
    get_video_info,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class ISOTrack:
    """A single ISO recording track."""
    file_path: str = ""
    track_index: int = 0
    timecode: str = ""           # SMPTE timecode string (HH:MM:SS:FF)
    timecode_seconds: float = 0.0
    duration: float = 0.0
    width: int = 1920
    height: int = 1080
    fps: float = 29.97
    has_audio: bool = True
    codec: str = ""
    label: str = ""


@dataclass
class SyncedISOTrack:
    """An ISO track with computed sync offset."""
    track: ISOTrack = field(default_factory=ISOTrack)
    offset: float = 0.0         # seconds offset from timeline start
    sync_method: str = ""       # 'timecode' or 'audio'
    confidence: float = 0.0     # 0-1 sync confidence


@dataclass
class ISOIngestResult:
    """Result of multi-track ISO ingest."""
    tracks: List[SyncedISOTrack] = field(default_factory=list)
    timeline_path: str = ""
    timeline_xml: str = ""
    total_duration: float = 0.0
    sync_method: str = ""


# ---------------------------------------------------------------------------
# Timecode Parsing
# ---------------------------------------------------------------------------
def _parse_timecode(tc_str: str, fps: float = 29.97) -> float:
    """Convert SMPTE timecode string to seconds.

    Handles HH:MM:SS:FF and HH:MM:SS;FF (drop-frame) formats.
    """
    if not tc_str:
        return -1.0

    # Normalize separators
    tc = tc_str.replace(";", ":").strip()
    parts = tc.split(":")

    if len(parts) == 4:
        h, m, s, f = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        return h * 3600 + m * 60 + s + f / fps
    elif len(parts) == 3:
        h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
        return h * 3600 + m * 60 + s
    return -1.0


# ---------------------------------------------------------------------------
# Track Detection
# ---------------------------------------------------------------------------
def detect_iso_tracks(file_path: str) -> ISOTrack:
    """Detect ISO recording properties from a video file.

    Reads timecode, codec info, resolution, frame rate, and audio
    presence via ffprobe.

    Args:
        file_path: Path to ISO recording file.

    Returns:
        ISOTrack with detected properties.
    """
    import subprocess as _sp

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"ISO file not found: {file_path}")

    info = get_video_info(file_path)

    # Get detailed stream info including timecode
    cmd = [
        get_ffprobe_path(), "-v", "quiet",
        "-show_entries", "format_tags=timecode:stream_tags=timecode",
        "-show_entries", "stream=codec_name,codec_type",
        "-of", "json", file_path,
    ]
    result = _sp.run(cmd, capture_output=True, timeout=30)

    timecode = ""
    codec = ""
    has_audio = False

    if result.returncode == 0:
        try:
            data = json.loads(result.stdout.decode())

            # Extract timecode from format or stream tags
            fmt_tags = data.get("format", {}).get("tags", {})
            timecode = fmt_tags.get("timecode", "")

            streams = data.get("streams", [])
            for s in streams:
                if s.get("codec_type") == "video" and not codec:
                    codec = s.get("codec_name", "")
                if s.get("codec_type") == "audio":
                    has_audio = True
                # Check stream-level timecode tags
                stags = s.get("tags", {})
                if not timecode and stags.get("timecode"):
                    timecode = stags["timecode"]
        except (json.JSONDecodeError, KeyError):
            pass

    label = os.path.splitext(os.path.basename(file_path))[0]
    tc_seconds = _parse_timecode(timecode, info["fps"]) if timecode else -1.0

    return ISOTrack(
        file_path=file_path,
        track_index=0,
        timecode=timecode,
        timecode_seconds=tc_seconds,
        duration=info["duration"],
        width=info["width"],
        height=info["height"],
        fps=info["fps"],
        has_audio=has_audio,
        codec=codec,
        label=label,
    )


# ---------------------------------------------------------------------------
# Audio-Based Sync
# ---------------------------------------------------------------------------
_SYNC_SAMPLE_RATE = 16000
_SYNC_CHUNK_SAMPLES = int(_SYNC_SAMPLE_RATE * 0.5)


def _extract_audio_pcm(video_path: str) -> bytes:
    """Extract audio as raw 16-bit LE mono PCM."""
    tmp = tempfile.NamedTemporaryFile(suffix=".raw", delete=False)
    tmp.close()
    try:
        cmd = [
            get_ffmpeg_path(), "-hide_banner", "-y",
            "-i", video_path,
            "-ac", "1", "-ar", str(_SYNC_SAMPLE_RATE),
            "-f", "s16le", "-acodec", "pcm_s16le",
            tmp.name,
        ]
        run_ffmpeg(cmd)
        with open(tmp.name, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def _cross_correlate_audio(samples_a: List[int], samples_b: List[int],
                           max_offset_sec: float = 30.0) -> tuple:
    """Cross-correlate two audio signals to find time offset.

    Returns (offset_seconds, confidence).
    """
    chunk_size = _SYNC_CHUNK_SAMPLES
    max_chunks = int(max_offset_sec / 0.5)

    def _energy_hash(samples):
        hashes = []
        for i in range(0, len(samples) - chunk_size, chunk_size):
            chunk = samples[i:i + chunk_size]
            energy = sum(s * s for s in chunk)
            level = min(255, int(energy / (32768 * 32768 * chunk_size / 256)))
            hashes.append(level)
        return hashes

    ha = _energy_hash(samples_a)
    hb = _energy_hash(samples_b)

    if not ha or not hb:
        return 0.0, 0.0

    best_score = -1
    best_offset = 0
    search = min(max_chunks, max(len(ha), len(hb)))

    for offset in range(-search, search + 1):
        score = 0
        count = 0
        for i in range(len(ha)):
            j = i + offset
            if 0 <= j < len(hb):
                score += max(0, 255 - abs(ha[i] - hb[j]))
                count += 1
        if count > 0:
            avg = score / count
            if avg > best_score:
                best_score = avg
                best_offset = offset

    return best_offset * 0.5, min(1.0, best_score / 255.0) if best_score >= 0 else 0.0


# ---------------------------------------------------------------------------
# Sync ISO Recordings
# ---------------------------------------------------------------------------
def sync_iso_recordings(
    file_paths: List[str],
    method: str = "auto",
    reference_index: int = 0,
    max_offset: float = 30.0,
    on_progress: Optional[Callable] = None,
) -> List[SyncedISOTrack]:
    """Synchronize multiple ISO recordings via timecode or audio.

    Args:
        file_paths: List of ISO recording file paths.
        method: Sync method: 'timecode', 'audio', or 'auto' (try timecode first).
        reference_index: Index of the reference recording.
        max_offset: Maximum sync offset search range in seconds.
        on_progress: Progress callback(pct, msg).

    Returns:
        List of SyncedISOTrack with computed offsets.
    """
    if len(file_paths) < 2:
        raise ValueError("Need at least 2 ISO recordings to sync")

    if reference_index >= len(file_paths):
        reference_index = 0

    if on_progress:
        on_progress(5, "Detecting ISO track properties...")

    # Detect all tracks
    tracks = []
    for i, fp in enumerate(file_paths):
        track = detect_iso_tracks(fp)
        track.track_index = i
        tracks.append(track)
        if on_progress:
            pct = 5 + int(20 * (i + 1) / len(file_paths))
            on_progress(pct, f"Detected track {i + 1}/{len(file_paths)}")

    # Determine sync method
    if method == "auto":
        # Use timecode if all tracks have valid timecodes
        all_have_tc = all(t.timecode_seconds >= 0 for t in tracks)
        method = "timecode" if all_have_tc else "audio"
        logger.info("Auto-selected sync method: %s", method)

    synced = []

    if method == "timecode":
        if on_progress:
            on_progress(40, "Syncing via timecode...")

        ref_tc = tracks[reference_index].timecode_seconds
        if ref_tc < 0:
            ref_tc = 0.0

        for i, track in enumerate(tracks):
            tc = track.timecode_seconds if track.timecode_seconds >= 0 else 0.0
            offset = tc - ref_tc
            synced.append(SyncedISOTrack(
                track=track,
                offset=round(offset, 3),
                sync_method="timecode",
                confidence=1.0 if track.timecode_seconds >= 0 else 0.0,
            ))

        if on_progress:
            on_progress(90, "Timecode sync complete")

    else:  # audio sync
        if on_progress:
            on_progress(30, "Extracting audio for sync...")

        audio_data = {}
        for i, track in enumerate(tracks):
            if track.has_audio:
                pcm = _extract_audio_pcm(track.file_path)
                n = len(pcm) // 2
                if n > 0:
                    audio_data[i] = list(struct.unpack(f"<{n}h", pcm[:n * 2]))
            if on_progress:
                pct = 30 + int(30 * (i + 1) / len(tracks))
                on_progress(pct, f"Extracted audio {i + 1}/{len(tracks)}")

        if on_progress:
            on_progress(65, "Cross-correlating audio...")

        ref_samples = audio_data.get(reference_index, [])

        for i, track in enumerate(tracks):
            if i == reference_index:
                synced.append(SyncedISOTrack(
                    track=track,
                    offset=0.0,
                    sync_method="audio",
                    confidence=1.0,
                ))
            elif i in audio_data and ref_samples:
                offset, conf = _cross_correlate_audio(
                    ref_samples, audio_data[i], max_offset
                )
                synced.append(SyncedISOTrack(
                    track=track,
                    offset=round(offset, 3),
                    sync_method="audio",
                    confidence=round(conf, 3),
                ))
            else:
                synced.append(SyncedISOTrack(
                    track=track,
                    offset=0.0,
                    sync_method="audio",
                    confidence=0.0,
                ))

        if on_progress:
            on_progress(90, "Audio sync complete")

    if on_progress:
        on_progress(95, f"Synchronized {len(synced)} ISO tracks via {method}")

    return synced


# ---------------------------------------------------------------------------
# Multicam Timeline Generation
# ---------------------------------------------------------------------------
def _path_to_url(filepath: str) -> str:
    """Convert local path to file:// URL."""
    import urllib.parse
    path = filepath.replace("\\", "/")
    if not path.startswith("/"):
        path = "/" + path
    return "file://localhost" + urllib.parse.quote(path, safe="/:")


def _indent_xml(elem) -> str:
    """Pretty-print XML element."""
    rough = ET.tostring(elem, encoding="unicode")
    parsed = minidom.parseString(rough)
    return parsed.toprettyxml(indent="  ", encoding=None)


def generate_multicam_timeline(
    synced_tracks: List[SyncedISOTrack],
    output_path: Optional[str] = None,
    sequence_name: str = "OpenCut ISO Multicam",
    fps: float = 29.97,
    width: int = 1920,
    height: int = 1080,
    on_progress: Optional[Callable] = None,
) -> ISOIngestResult:
    """Generate a multicam FCP XML timeline from synced ISO tracks.

    Args:
        synced_tracks: List of SyncedISOTrack from sync_iso_recordings().
        output_path: Write XML to this file path. Auto-generates if None.
        sequence_name: Name for the multicam sequence.
        fps: Output timeline frame rate.
        width: Output frame width.
        height: Output frame height.
        on_progress: Progress callback(pct, msg).

    Returns:
        ISOIngestResult with timeline XML and metadata.
    """
    if not synced_tracks:
        raise ValueError("No synced tracks provided")

    if on_progress:
        on_progress(10, "Building multicam timeline...")

    # Calculate total timeline duration
    min_offset = min(st.offset for st in synced_tracks)
    max_end = max(st.offset + st.track.duration for st in synced_tracks)
    total_duration = max_end - min_offset

    timebase = int(round(fps))
    ntsc = abs(fps - timebase) > 0.001
    total_frames = int(round(total_duration * fps))

    # Build XMEML
    xmeml = ET.Element("xmeml", version="5")
    sequence = ET.SubElement(xmeml, "sequence")
    ET.SubElement(sequence, "name").text = sequence_name
    ET.SubElement(sequence, "duration").text = str(total_frames)

    rate_elem = ET.SubElement(sequence, "rate")
    ET.SubElement(rate_elem, "timebase").text = str(timebase)
    ET.SubElement(rate_elem, "ntsc").text = "TRUE" if ntsc else "FALSE"

    media = ET.SubElement(sequence, "media")

    # Video
    video_section = ET.SubElement(media, "video")
    fmt = ET.SubElement(video_section, "format")
    sc = ET.SubElement(fmt, "samplecharacteristics")
    ET.SubElement(sc, "width").text = str(width)
    ET.SubElement(sc, "height").text = str(height)

    for i, st in enumerate(synced_tracks):
        track_elem = ET.SubElement(video_section, "track")
        offset_start = st.offset - min_offset
        start_frame = int(round(offset_start * fps))
        end_frame = int(round((offset_start + st.track.duration) * fps))

        clip_item = ET.SubElement(track_elem, "clipitem")
        label = st.track.label or f"ISO_{i + 1}"
        ET.SubElement(clip_item, "name").text = label
        ET.SubElement(clip_item, "duration").text = str(end_frame - start_frame)

        c_rate = ET.SubElement(clip_item, "rate")
        ET.SubElement(c_rate, "timebase").text = str(timebase)
        ET.SubElement(c_rate, "ntsc").text = "TRUE" if ntsc else "FALSE"

        ET.SubElement(clip_item, "start").text = str(start_frame)
        ET.SubElement(clip_item, "end").text = str(end_frame)
        ET.SubElement(clip_item, "in").text = "0"
        ET.SubElement(clip_item, "out").text = str(end_frame - start_frame)

        file_elem = ET.SubElement(clip_item, "file")
        ET.SubElement(file_elem, "name").text = os.path.basename(st.track.file_path)
        ET.SubElement(file_elem, "pathurl").text = _path_to_url(st.track.file_path)

        if on_progress:
            pct = 10 + int(60 * (i + 1) / len(synced_tracks))
            on_progress(pct, f"Added track {i + 1}/{len(synced_tracks)}")

    # Audio
    audio_section = ET.SubElement(media, "audio")
    a_fmt = ET.SubElement(audio_section, "format")
    a_sc = ET.SubElement(a_fmt, "samplecharacteristics")
    ET.SubElement(a_sc, "depth").text = "16"
    ET.SubElement(a_sc, "samplerate").text = "48000"

    for i, st in enumerate(synced_tracks):
        a_track = ET.SubElement(audio_section, "track")
        offset_start = st.offset - min_offset
        start_frame = int(round(offset_start * fps))
        end_frame = int(round((offset_start + st.track.duration) * fps))

        clip_item = ET.SubElement(a_track, "clipitem")
        label = st.track.label or f"ISO_{i + 1}"
        ET.SubElement(clip_item, "name").text = f"{label}_audio"
        ET.SubElement(clip_item, "duration").text = str(end_frame - start_frame)

        c_rate = ET.SubElement(clip_item, "rate")
        ET.SubElement(c_rate, "timebase").text = str(timebase)
        ET.SubElement(c_rate, "ntsc").text = "TRUE" if ntsc else "FALSE"

        ET.SubElement(clip_item, "start").text = str(start_frame)
        ET.SubElement(clip_item, "end").text = str(end_frame)

        file_elem = ET.SubElement(clip_item, "file")
        ET.SubElement(file_elem, "name").text = os.path.basename(st.track.file_path)
        ET.SubElement(file_elem, "pathurl").text = _path_to_url(st.track.file_path)

    xml_str = _indent_xml(xmeml)

    output_file = None
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(xml_str)
        output_file = output_path

    sync_method = synced_tracks[0].sync_method if synced_tracks else ""

    if on_progress:
        on_progress(100, f"Multicam timeline complete ({len(synced_tracks)} tracks)")

    return ISOIngestResult(
        tracks=synced_tracks,
        timeline_path=output_file or "",
        timeline_xml=xml_str,
        total_duration=round(total_duration, 3),
        sync_method=sync_method,
    )
