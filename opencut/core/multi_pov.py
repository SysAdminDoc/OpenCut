"""
OpenCut Multi-POV Sync (12.4)

Synchronize multiple player/camera recordings via audio
cross-correlation (fingerprinting), then export as a multicam
FCP XML timeline for NLE import.

Pipeline:
1. Extract audio from each recording
2. Cross-correlate audio between all pairs to find time offsets
3. Align recordings to a common timeline
4. Generate multicam FCP XML

Audio extraction and sync via FFmpeg; cross-correlation uses the
audio fingerprinting infrastructure from opencut.core.audio_fingerprint.
"""

import logging
import os
import struct
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
from xml.dom import minidom

from opencut.helpers import (
    get_ffmpeg_path,
    get_video_info,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class SyncedClip:
    """A recording with its computed time offset."""
    file_path: str = ""
    offset: float = 0.0          # seconds offset from reference (0 = earliest)
    duration: float = 0.0
    label: str = ""
    confidence: float = 0.0      # 0-1 sync confidence


@dataclass
class MultiPovResult:
    """Result of multi-POV synchronization."""
    synced_clips: List[SyncedClip] = field(default_factory=list)
    reference_file: str = ""
    xml_path: str = ""
    xml_content: str = ""
    total_duration: float = 0.0


# ---------------------------------------------------------------------------
# Audio Extraction
# ---------------------------------------------------------------------------
_SYNC_SAMPLE_RATE = 16000
_SYNC_CHUNK_SAMPLES = int(_SYNC_SAMPLE_RATE * 0.5)  # 0.5s chunks


def _extract_audio_pcm(video_path: str, sample_rate: int = _SYNC_SAMPLE_RATE) -> bytes:
    """Extract audio from video as raw 16-bit LE mono PCM."""
    tmp = tempfile.NamedTemporaryFile(suffix=".raw", delete=False)
    tmp.close()
    try:
        cmd = [
            get_ffmpeg_path(), "-hide_banner", "-y",
            "-i", video_path,
            "-ac", "1", "-ar", str(sample_rate),
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


def _pcm_to_samples(pcm_data: bytes) -> List[int]:
    """Unpack raw PCM bytes to list of signed 16-bit integers."""
    n = len(pcm_data) // 2
    if n == 0:
        return []
    return list(struct.unpack(f"<{n}h", pcm_data[:n * 2]))


# ---------------------------------------------------------------------------
# Cross-Correlation for Audio Sync
# ---------------------------------------------------------------------------
def _cross_correlate(samples_a: List[int], samples_b: List[int],
                     max_offset_sec: float = 30.0) -> tuple:
    """Find the time offset between two audio signals via cross-correlation.

    Uses chunk-based hashing for efficiency: instead of sample-level
    correlation, we hash 0.5s chunks and find the best overlap.

    Returns (offset_seconds, confidence) where offset is how much B
    is ahead of A (positive = B starts later in real time).
    """
    chunk_size = _SYNC_CHUNK_SAMPLES
    max_offset_chunks = int(max_offset_sec / 0.5)

    # Hash chunks for both signals
    def _hash_chunks(samples):
        hashes = []
        for i in range(0, len(samples) - chunk_size, chunk_size):
            chunk = samples[i:i + chunk_size]
            # Simple energy-based hash
            energy = sum(s * s for s in chunk)
            # Bin into coarse energy level
            level = min(255, int(energy / (32768 * 32768 * chunk_size / 256)))
            hashes.append(level)
        return hashes

    hashes_a = _hash_chunks(samples_a)
    hashes_b = _hash_chunks(samples_b)

    if not hashes_a or not hashes_b:
        return 0.0, 0.0

    best_score = -1
    best_offset = 0

    search_range = min(max_offset_chunks, max(len(hashes_a), len(hashes_b)))

    for offset in range(-search_range, search_range + 1):
        score = 0
        count = 0
        for i in range(len(hashes_a)):
            j = i + offset
            if 0 <= j < len(hashes_b):
                # Similarity: inverse of absolute difference
                diff = abs(hashes_a[i] - hashes_b[j])
                score += max(0, 255 - diff)
                count += 1
        if count > 0:
            avg_score = score / count
            if avg_score > best_score:
                best_score = avg_score
                best_offset = offset

    offset_sec = best_offset * 0.5  # each chunk is 0.5s
    confidence = min(1.0, best_score / 255.0) if best_score >= 0 else 0.0

    return offset_sec, confidence


# ---------------------------------------------------------------------------
# Shared Audio Detection
# ---------------------------------------------------------------------------
def detect_shared_audio(file_paths: List[str]) -> Dict:
    """Detect which recordings share common audio content.

    Extracts audio fingerprints from each file and compares all pairs.

    Args:
        file_paths: List of video file paths.

    Returns:
        Dict with 'pairs' (list of dicts with file_a, file_b, similarity)
        and 'has_shared_audio' bool.
    """
    if len(file_paths) < 2:
        return {"pairs": [], "has_shared_audio": False}

    # Extract short audio samples for quick comparison
    audio_data = {}
    for fp in file_paths:
        if os.path.isfile(fp):
            try:
                pcm = _extract_audio_pcm(fp)
                audio_data[fp] = _pcm_to_samples(pcm)
            except Exception as e:
                logger.warning("Failed to extract audio from %s: %s", fp, e)

    pairs = []
    for i in range(len(file_paths)):
        for j in range(i + 1, len(file_paths)):
            fp_a, fp_b = file_paths[i], file_paths[j]
            if fp_a in audio_data and fp_b in audio_data:
                _, confidence = _cross_correlate(
                    audio_data[fp_a], audio_data[fp_b], max_offset_sec=10.0
                )
                pairs.append({
                    "file_a": fp_a,
                    "file_b": fp_b,
                    "similarity": round(confidence, 3),
                })

    has_shared = any(p["similarity"] > 0.3 for p in pairs)
    return {"pairs": pairs, "has_shared_audio": has_shared}


# ---------------------------------------------------------------------------
# Multi-POV Sync
# ---------------------------------------------------------------------------
def sync_pov_recordings(
    file_paths: List[str],
    reference_index: int = 0,
    max_offset: float = 30.0,
    on_progress: Optional[Callable] = None,
) -> List[SyncedClip]:
    """Synchronize multiple recordings via audio cross-correlation.

    All files are aligned relative to the reference recording
    (index 0 by default, which gets offset=0).

    Args:
        file_paths: List of video file paths to synchronize.
        reference_index: Index of the reference recording (offset=0).
        max_offset: Maximum sync offset to search in seconds.
        on_progress: Progress callback(pct, msg).

    Returns:
        List of SyncedClip with computed offsets.
    """
    if len(file_paths) < 2:
        raise ValueError("Need at least 2 recordings to sync")

    if reference_index >= len(file_paths):
        reference_index = 0

    if on_progress:
        on_progress(5, "Extracting audio from recordings...")

    # Extract audio from all files
    audio_samples = {}
    total = len(file_paths)
    for i, fp in enumerate(file_paths):
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"Recording not found: {fp}")
        pcm = _extract_audio_pcm(fp)
        audio_samples[fp] = _pcm_to_samples(pcm)
        if on_progress:
            pct = 5 + int(40 * (i + 1) / total)
            on_progress(pct, f"Extracted audio {i + 1}/{total}")

    if on_progress:
        on_progress(50, "Cross-correlating audio...")

    ref_path = file_paths[reference_index]
    ref_samples = audio_samples[ref_path]

    synced = []
    for i, fp in enumerate(file_paths):
        info = get_video_info(fp)
        label = os.path.splitext(os.path.basename(fp))[0]

        if i == reference_index:
            synced.append(SyncedClip(
                file_path=fp,
                offset=0.0,
                duration=info["duration"],
                label=label,
                confidence=1.0,
            ))
        else:
            offset, confidence = _cross_correlate(
                ref_samples, audio_samples[fp], max_offset
            )
            synced.append(SyncedClip(
                file_path=fp,
                offset=round(offset, 3),
                duration=info["duration"],
                label=label,
                confidence=round(confidence, 3),
            ))

        if on_progress:
            pct = 50 + int(40 * (i + 1) / total)
            on_progress(pct, f"Synced {i + 1}/{total} recordings")

    if on_progress:
        on_progress(95, f"Synchronized {len(synced)} recordings")

    return synced


# ---------------------------------------------------------------------------
# Multicam XML Generation
# ---------------------------------------------------------------------------
def _path_to_url(filepath: str) -> str:
    """Convert a local file path to a file:// URL for FCP XML."""
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


def generate_multicam_xml(
    synced_clips: List[SyncedClip],
    output_path: Optional[str] = None,
    sequence_name: str = "OpenCut Multi-POV",
    fps: float = 29.97,
    width: int = 1920,
    height: int = 1080,
) -> Dict:
    """Generate FCP XML multicam sequence from synced POV recordings.

    Args:
        synced_clips: List of SyncedClip from sync_pov_recordings().
        output_path: Write XML to this path. Returns XML string if None.
        sequence_name: Name for the multicam sequence.
        fps: Frames per second.
        width: Frame width.
        height: Frame height.

    Returns:
        Dict with xml (str), output (str or None), clip_count, duration.
    """
    if not synced_clips:
        raise ValueError("No synced clips provided")

    # Calculate total duration (from earliest to latest end)
    min_offset = min(c.offset for c in synced_clips)
    max_end = max(c.offset + c.duration for c in synced_clips)
    total_duration = max_end - min_offset

    timebase = int(round(fps))
    ntsc = abs(fps - timebase) > 0.001
    total_frames = int(round(total_duration * fps))

    # Build XMEML structure
    xmeml = ET.Element("xmeml", version="5")
    sequence = ET.SubElement(xmeml, "sequence")
    ET.SubElement(sequence, "name").text = sequence_name
    ET.SubElement(sequence, "duration").text = str(total_frames)

    rate = ET.SubElement(sequence, "rate")
    ET.SubElement(rate, "timebase").text = str(timebase)
    ET.SubElement(rate, "ntsc").text = "TRUE" if ntsc else "FALSE"

    media = ET.SubElement(sequence, "media")

    # Video section
    video_section = ET.SubElement(media, "video")
    fmt = ET.SubElement(video_section, "format")
    sc = ET.SubElement(fmt, "samplecharacteristics")
    ET.SubElement(sc, "width").text = str(width)
    ET.SubElement(sc, "height").text = str(height)

    # One track per POV
    for i, clip in enumerate(synced_clips):
        track = ET.SubElement(video_section, "track")

        # Offset relative to timeline start
        offset_from_start = clip.offset - min_offset
        start_frame = int(round(offset_from_start * fps))
        end_frame = int(round((offset_from_start + clip.duration) * fps))

        clip_item = ET.SubElement(track, "clipitem")
        ET.SubElement(clip_item, "name").text = clip.label or f"POV_{i + 1}"
        ET.SubElement(clip_item, "duration").text = str(end_frame - start_frame)

        c_rate = ET.SubElement(clip_item, "rate")
        ET.SubElement(c_rate, "timebase").text = str(timebase)
        ET.SubElement(c_rate, "ntsc").text = "TRUE" if ntsc else "FALSE"

        ET.SubElement(clip_item, "start").text = str(start_frame)
        ET.SubElement(clip_item, "end").text = str(end_frame)
        ET.SubElement(clip_item, "in").text = "0"
        ET.SubElement(clip_item, "out").text = str(end_frame - start_frame)

        file_elem = ET.SubElement(clip_item, "file")
        ET.SubElement(file_elem, "name").text = os.path.basename(clip.file_path)
        ET.SubElement(file_elem, "pathurl").text = _path_to_url(clip.file_path)

    # Audio section
    audio_section = ET.SubElement(media, "audio")
    a_fmt = ET.SubElement(audio_section, "format")
    a_sc = ET.SubElement(a_fmt, "samplecharacteristics")
    ET.SubElement(a_sc, "depth").text = "16"
    ET.SubElement(a_sc, "samplerate").text = "48000"

    for i, clip in enumerate(synced_clips):
        a_track = ET.SubElement(audio_section, "track")
        offset_from_start = clip.offset - min_offset
        start_frame = int(round(offset_from_start * fps))
        end_frame = int(round((offset_from_start + clip.duration) * fps))

        clip_item = ET.SubElement(a_track, "clipitem")
        ET.SubElement(clip_item, "name").text = f"{clip.label or f'POV_{i + 1}'}_audio"
        ET.SubElement(clip_item, "duration").text = str(end_frame - start_frame)

        c_rate = ET.SubElement(clip_item, "rate")
        ET.SubElement(c_rate, "timebase").text = str(timebase)
        ET.SubElement(c_rate, "ntsc").text = "TRUE" if ntsc else "FALSE"

        ET.SubElement(clip_item, "start").text = str(start_frame)
        ET.SubElement(clip_item, "end").text = str(end_frame)

        file_elem = ET.SubElement(clip_item, "file")
        ET.SubElement(file_elem, "name").text = os.path.basename(clip.file_path)
        ET.SubElement(file_elem, "pathurl").text = _path_to_url(clip.file_path)

    xml_str = _indent_xml(xmeml)

    output_file = None
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(xml_str)
        output_file = output_path
        logger.info("Multi-POV XML written: %s (%d clips)", output_path, len(synced_clips))

    return {
        "xml": xml_str,
        "output": output_file,
        "clip_count": len(synced_clips),
        "duration": round(total_duration, 3),
    }
