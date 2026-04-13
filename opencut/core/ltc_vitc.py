"""
OpenCut LTC/VITC Timecode Module (44.2)

Extract LTC (Linear Timecode) from audio tracks and VITC (Vertical Interval
Timecode) from video scan lines for professional broadcast workflows.
"""

import logging
import os
import struct
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import get_ffmpeg_path, get_video_info, run_ffmpeg

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LTC_FRAME_BITS = 80  # LTC frame is 80 bits
LTC_SYNC_WORD = 0x3FFD  # 16-bit sync word (end of LTC frame)
VITC_LINE_BITS = 90  # VITC line is 90 bits (including CRC)
VITC_SYNC_PATTERN = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]  # VITC sync run-in


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TimecodeExtraction:
    """Result of timecode extraction."""
    timecodes: List[str] = field(default_factory=list)
    frame_numbers: List[int] = field(default_factory=list)
    source_type: str = ""  # "ltc" or "vitc"
    sample_rate: int = 0
    fps: float = 0.0
    start_tc: str = ""
    end_tc: str = ""
    total_frames: int = 0
    gaps: List[Dict] = field(default_factory=list)
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Timecode formatting
# ---------------------------------------------------------------------------

def _frames_to_tc(frame_num: int, fps: float, drop_frame: bool = False) -> str:
    """Convert frame number to SMPTE timecode string HH:MM:SS:FF."""
    if fps <= 0:
        return "00:00:00:00"

    fps_int = round(fps)
    total_frames = frame_num

    if drop_frame and fps_int in (30, 60):
        # Drop-frame timecode calculation
        drop = 2 if fps_int == 30 else 4
        frames_per_10min = fps_int * 60 * 10 - drop * 9
        d = total_frames // frames_per_10min
        m = total_frames % frames_per_10min
        if m < fps_int * 60:
            # First minute of 10 (no drop)
            f = total_frames + drop * 9 * d
        else:
            f = total_frames + drop * 9 * d + drop * ((m - fps_int * 60) // (fps_int * 60 - drop) + 1)
        ff = f % fps_int
        ss = (f // fps_int) % 60
        mm = (f // (fps_int * 60)) % 60
        hh = f // (fps_int * 3600)
        sep = ";"
    else:
        ff = total_frames % fps_int
        ss = (total_frames // fps_int) % 60
        mm = (total_frames // (fps_int * 60)) % 60
        hh = total_frames // (fps_int * 3600)
        sep = ":"

    return f"{hh:02d}:{mm:02d}:{ss:02d}{sep}{ff:02d}"


def _tc_to_frames(tc: str, fps: float) -> int:
    """Convert SMPTE timecode string to frame number."""
    if not tc:
        return 0
    # Handle both : and ; separators
    parts = tc.replace(";", ":").split(":")
    if len(parts) != 4:
        return 0
    try:
        hh, mm, ss, ff = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
    except ValueError:
        return 0
    fps_int = round(fps)
    return hh * fps_int * 3600 + mm * fps_int * 60 + ss * fps_int + ff


# ---------------------------------------------------------------------------
# LTC bit parsing
# ---------------------------------------------------------------------------

def parse_ltc_bits(
    audio_data: bytes,
    sample_rate: int,
    fps: float = 0.0,
) -> List[Dict]:
    """Parse LTC (Linear Timecode) bits from raw audio data.

    LTC is a biphase-encoded signal in the audio stream.
    This function detects zero-crossings to decode bits.

    Args:
        audio_data: Raw 16-bit mono PCM audio data.
        sample_rate: Audio sample rate in Hz.
        fps: Expected frame rate (auto-detected if 0).

    Returns:
        List of dicts with timecode, frame_number for each decoded frame.
    """
    if not audio_data:
        return []

    n_samples = len(audio_data) // 2
    if n_samples == 0:
        return []

    samples = struct.unpack(f"<{n_samples}h", audio_data[:n_samples * 2])

    # Auto-detect fps if not provided
    if fps <= 0:
        fps = 25.0  # default to PAL

    # Find zero crossings for biphase decoding
    crossings = []
    for i in range(1, n_samples):
        if (samples[i] >= 0) != (samples[i - 1] >= 0):
            crossings.append(i)

    if len(crossings) < LTC_FRAME_BITS * 2:
        return []

    # Expected samples per LTC bit
    samples_per_bit = sample_rate / (fps * LTC_FRAME_BITS)
    half_bit = samples_per_bit / 2.0
    tolerance = 0.35  # 35% tolerance

    # Decode biphase bits from zero crossings
    decoded_bits = []
    i = 0
    while i < len(crossings) - 1:
        gap = crossings[i + 1] - crossings[i]
        if abs(gap - samples_per_bit) / max(samples_per_bit, 1) < tolerance:
            # Long gap = 0 bit
            decoded_bits.append(0)
            i += 1
        elif abs(gap - half_bit) / max(half_bit, 1) < tolerance:
            # Short gap = 1 bit (need two short gaps)
            if i + 2 < len(crossings):
                gap2 = crossings[i + 2] - crossings[i + 1]
                if abs(gap2 - half_bit) / max(half_bit, 1) < tolerance:
                    decoded_bits.append(1)
                    i += 2
                else:
                    i += 1
            else:
                i += 1
        else:
            i += 1

    # Find sync words and extract timecodes
    timecodes = []
    for start in range(len(decoded_bits) - LTC_FRAME_BITS):
        # Check for sync word (last 16 bits of 80-bit frame)
        sync_start = start + LTC_FRAME_BITS - 16
        sync_bits = decoded_bits[sync_start:sync_start + 16]
        sync_val = 0
        for b in sync_bits:
            sync_val = (sync_val << 1) | b

        if sync_val == LTC_SYNC_WORD:
            # Extract timecode from LTC frame bits
            frame_bits = decoded_bits[start:start + LTC_FRAME_BITS]

            # BCD decode: frames, seconds, minutes, hours
            # LTC bit layout (simplified):
            # bits 0-3: frame units, bits 8-9: frame tens
            # bits 16-19: sec units, bits 24-26: sec tens
            # bits 32-35: min units, bits 40-42: min tens
            # bits 48-51: hour units, bits 56-57: hour tens
            def _bcd_decode(bits_slice):
                val = 0
                for idx, b in enumerate(bits_slice):
                    val |= (b << idx)
                return val

            if len(frame_bits) >= 64:
                ff_units = _bcd_decode(frame_bits[0:4])
                ff_tens = _bcd_decode(frame_bits[8:10])
                ss_units = _bcd_decode(frame_bits[16:20])
                ss_tens = _bcd_decode(frame_bits[24:27])
                mm_units = _bcd_decode(frame_bits[32:36])
                mm_tens = _bcd_decode(frame_bits[40:43])
                hh_units = _bcd_decode(frame_bits[48:52])
                hh_tens = _bcd_decode(frame_bits[56:58])

                ff = ff_tens * 10 + ff_units
                ss = ss_tens * 10 + ss_units
                mm = mm_tens * 10 + mm_units
                hh = hh_tens * 10 + hh_units

                # Validate ranges
                fps_int = round(fps)
                if 0 <= ff < fps_int and 0 <= ss < 60 and 0 <= mm < 60 and 0 <= hh < 24:
                    tc_str = f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"
                    frame_num = _tc_to_frames(tc_str, fps)
                    timecodes.append({
                        "timecode": tc_str,
                        "frame_number": frame_num,
                        "bit_offset": start,
                    })

    return timecodes


# ---------------------------------------------------------------------------
# LTC extraction
# ---------------------------------------------------------------------------

def extract_ltc(
    audio_path: str,
    fps: float = 0.0,
    channel: int = -1,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Extract LTC timecode from an audio track.

    Args:
        audio_path: Path to audio or video file.
        fps: Expected frame rate (auto-detected if 0).
        channel: Audio channel to analyze (-1 = auto, last channel).
        on_progress: Optional callback(pct, msg).

    Returns:
        TimecodeExtraction-like dict with timecodes and metadata.
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if on_progress:
        on_progress(5, "Extracting audio for LTC analysis")

    # Auto-detect fps from video if available
    if fps <= 0:
        try:
            info = get_video_info(audio_path)
            fps = info.get("fps", 25.0) or 25.0
        except Exception:
            fps = 25.0

    sample_rate = 48000  # LTC needs high sample rate for reliable decoding

    # Extract audio as mono PCM (LTC is often on the last channel)
    tmp_wav = tempfile.mktemp(suffix="_ltc_audio.wav")
    try:
        ffmpeg = get_ffmpeg_path()
        af_filter = "pan=mono|c0=c0" if channel <= 0 else f"pan=mono|c0=c{channel}"
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", audio_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", str(sample_rate), "-ac", "1",
            "-af", af_filter,
            tmp_wav,
        ]
        run_ffmpeg(cmd, timeout=300)

        if on_progress:
            on_progress(30, "Decoding LTC signal")

        # Read PCM data
        with open(tmp_wav, "rb") as f:
            f.read(44)  # skip WAV header
            audio_data = f.read()

        # Parse LTC
        tc_frames = parse_ltc_bits(audio_data, sample_rate, fps)

        if on_progress:
            on_progress(80, f"Found {len(tc_frames)} LTC timecodes")

        # Build result
        timecodes = [tc["timecode"] for tc in tc_frames]
        frame_numbers = [tc["frame_number"] for tc in tc_frames]

        # Detect gaps
        gaps = []
        for i in range(1, len(frame_numbers)):
            expected = frame_numbers[i - 1] + 1
            if frame_numbers[i] != expected:
                gaps.append({
                    "after_frame": frame_numbers[i - 1],
                    "before_frame": frame_numbers[i],
                    "missing_frames": frame_numbers[i] - expected,
                    "after_tc": timecodes[i - 1],
                    "before_tc": timecodes[i],
                })

        start_tc = timecodes[0] if timecodes else ""
        end_tc = timecodes[-1] if timecodes else ""

        if on_progress:
            on_progress(100, "LTC extraction complete")

        return {
            "timecodes": timecodes,
            "frame_numbers": frame_numbers,
            "source_type": "ltc",
            "sample_rate": sample_rate,
            "fps": fps,
            "start_tc": start_tc,
            "end_tc": end_tc,
            "total_frames": len(timecodes),
            "gaps": gaps,
            "confidence": min(1.0, len(timecodes) / max(1, fps * 10)) if timecodes else 0.0,
        }
    finally:
        if os.path.isfile(tmp_wav):
            try:
                os.unlink(tmp_wav)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# VITC extraction
# ---------------------------------------------------------------------------

def extract_vitc(
    video_path: str,
    scan_lines: Optional[List[int]] = None,
    fps: float = 0.0,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Extract VITC (Vertical Interval Timecode) from video scan lines.

    VITC is embedded in the vertical blanking interval of analog video.
    This function analyzes the top scan lines of each frame.

    Args:
        video_path: Path to video file.
        scan_lines: Specific scan line numbers to check (default: lines 14-20).
        fps: Expected frame rate (auto-detected if 0).
        on_progress: Optional callback(pct, msg).

    Returns:
        TimecodeExtraction-like dict.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    info = get_video_info(video_path)
    width = info.get("width", 720) or 720
    height = info.get("height", 480) or 480
    if fps <= 0:
        fps = info.get("fps", 25.0) or 25.0
    info.get("duration", 0.0) or 0.0

    if scan_lines is None:
        scan_lines = list(range(14, 21))  # standard VITC lines

    if on_progress:
        on_progress(5, "Extracting video frames for VITC analysis")

    # Extract top portion of frames as grayscale
    crop_h = max(scan_lines) + 5 if scan_lines else 25
    crop_h = min(crop_h, height)

    tmp_raw = tempfile.mktemp(suffix="_vitc_frames.gray")
    try:
        ffmpeg = get_ffmpeg_path()
        vf = f"crop={width}:{crop_h}:0:0,format=gray"
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", video_path,
            "-vf", vf,
            "-f", "rawvideo", "-pix_fmt", "gray",
            tmp_raw,
        ]
        run_ffmpeg(cmd, timeout=600)

        if not os.path.isfile(tmp_raw):
            return _empty_vitc_result(fps)

        raw_size = os.path.getsize(tmp_raw)
        frame_size = width * crop_h
        if frame_size == 0:
            return _empty_vitc_result(fps)
        n_frames = raw_size // frame_size

        if on_progress:
            on_progress(30, f"Analyzing {n_frames} frames for VITC")

        with open(tmp_raw, "rb") as f:
            raw_data = f.read()

        timecodes = []
        frame_numbers = []

        for fi in range(n_frames):
            offset = fi * frame_size
            frame = raw_data[offset:offset + frame_size]
            if len(frame) < frame_size:
                break

            # Analyze scan lines for VITC pattern
            tc = _decode_vitc_frame(frame, width, crop_h, scan_lines, fps)
            if tc:
                timecodes.append(tc["timecode"])
                frame_numbers.append(tc["frame_number"])

            if on_progress and fi % 200 == 0:
                pct = 30 + int(60 * fi / max(n_frames, 1))
                on_progress(pct, f"Analyzing frame {fi}/{n_frames}")

        # Detect gaps
        gaps = []
        for i in range(1, len(frame_numbers)):
            expected = frame_numbers[i - 1] + 1
            if frame_numbers[i] != expected:
                gaps.append({
                    "after_frame": frame_numbers[i - 1],
                    "before_frame": frame_numbers[i],
                    "missing_frames": frame_numbers[i] - expected,
                })

        start_tc = timecodes[0] if timecodes else ""
        end_tc = timecodes[-1] if timecodes else ""

        if on_progress:
            on_progress(100, "VITC extraction complete")

        return {
            "timecodes": timecodes,
            "frame_numbers": frame_numbers,
            "source_type": "vitc",
            "sample_rate": 0,
            "fps": fps,
            "start_tc": start_tc,
            "end_tc": end_tc,
            "total_frames": len(timecodes),
            "gaps": gaps,
            "confidence": min(1.0, len(timecodes) / max(1, n_frames)) if timecodes else 0.0,
        }
    finally:
        if os.path.isfile(tmp_raw):
            try:
                os.unlink(tmp_raw)
            except OSError:
                pass


def _empty_vitc_result(fps: float) -> dict:
    """Return empty VITC result."""
    return {
        "timecodes": [],
        "frame_numbers": [],
        "source_type": "vitc",
        "sample_rate": 0,
        "fps": fps,
        "start_tc": "",
        "end_tc": "",
        "total_frames": 0,
        "gaps": [],
        "confidence": 0.0,
    }


def _decode_vitc_frame(
    frame_data: bytes,
    width: int,
    height: int,
    scan_lines: List[int],
    fps: float,
) -> Optional[Dict]:
    """Try to decode VITC from specific scan lines in a frame.

    Returns dict with timecode and frame_number, or None.
    """
    for line_num in scan_lines:
        if line_num >= height:
            continue

        line_offset = line_num * width
        line = frame_data[line_offset:line_offset + width]
        if len(line) < width:
            continue

        # Binarize the scan line (threshold at 50% brightness)
        threshold = 128
        binary = [1 if b > threshold else 0 for b in line]

        # Look for VITC sync pattern
        # VITC has a specific run-in pattern at the start
        bits_per_pixel = width / VITC_LINE_BITS
        if bits_per_pixel < 1:
            continue

        # Sample at expected bit positions
        vitc_bits = []
        for bit_idx in range(VITC_LINE_BITS):
            pixel_pos = int(bit_idx * bits_per_pixel + bits_per_pixel / 2)
            if pixel_pos < len(binary):
                vitc_bits.append(binary[pixel_pos])
            else:
                vitc_bits.append(0)

        # Try to decode timecode from VITC bits
        tc = _parse_vitc_bits(vitc_bits, fps)
        if tc is not None:
            return tc

    return None


def _parse_vitc_bits(bits: List[int], fps: float) -> Optional[Dict]:
    """Parse VITC bit pattern to extract timecode.

    VITC layout (90 bits):
    - 2 sync bits, then groups of 10 bits (sync + 8 data + parity)
    - BCD encoded: frames, seconds, minutes, hours

    Returns dict with timecode and frame_number, or None.
    """
    if len(bits) < 90:
        return None

    # Check for sync run-in (first 10 bits should match pattern)
    sync_match = sum(1 for a, b in zip(bits[:10], VITC_SYNC_PATTERN) if a == b)
    if sync_match < 7:  # Allow some tolerance
        return None

    # Extract BCD values from VITC groups
    # Each group is 10 bits: sync(1) + data(8) + parity(1)
    def _extract_group(start):
        if start + 10 > len(bits):
            return 0
        data_bits = bits[start + 1:start + 9]
        val = 0
        for idx, b in enumerate(data_bits):
            val |= (b << idx)
        return val

    try:
        ff_units = _extract_group(10) & 0x0F
        ff_tens = _extract_group(20) & 0x03
        ss_units = _extract_group(30) & 0x0F
        ss_tens = _extract_group(40) & 0x07
        mm_units = _extract_group(50) & 0x0F
        mm_tens = _extract_group(60) & 0x07
        hh_units = _extract_group(70) & 0x0F
        hh_tens = _extract_group(80) & 0x03

        ff = ff_tens * 10 + ff_units
        ss = ss_tens * 10 + ss_units
        mm = mm_tens * 10 + mm_units
        hh = hh_tens * 10 + hh_units

        fps_int = round(fps)
        if not (0 <= ff < fps_int and 0 <= ss < 60 and 0 <= mm < 60 and 0 <= hh < 24):
            return None

        tc_str = f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"
        frame_num = _tc_to_frames(tc_str, fps)

        return {"timecode": tc_str, "frame_number": frame_num}
    except (IndexError, ValueError):
        return None
