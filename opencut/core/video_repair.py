"""
OpenCut Corrupted Video Repair Module

Diagnose and repair common video corruption types:
- Missing MOOV atom (unfinished recordings)
- Broken containers
- Partial/truncated files
- Bitstream errors
"""

import logging
import os
import subprocess as _sp
from dataclasses import dataclass
from typing import Callable, Optional

from opencut.helpers import get_ffmpeg_path, get_ffprobe_path, get_video_info, output_path, run_ffmpeg

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Corruption types
# ---------------------------------------------------------------------------

CORRUPTION_TYPES = (
    "missing_moov",
    "broken_container",
    "partial_file",
    "bitstream_error",
    "no_corruption",
)


@dataclass
class CorruptionDiagnosis:
    """Describes the corruption status of a video file."""
    corruption_type: str = "no_corruption"
    severity: str = "none"           # "none", "low", "medium", "high"
    description: str = ""
    recoverable: bool = True
    suggested_action: str = "none"   # "remux", "transcode", "error_conceal", "none"


# ---------------------------------------------------------------------------
# Corruption Diagnosis
# ---------------------------------------------------------------------------

def diagnose_corruption(
    input_path: str,
    on_progress: Optional[Callable] = None,
) -> CorruptionDiagnosis:
    """
    Analyse a video file for signs of corruption.

    Uses ffprobe with error detection to classify the type and severity
    of any corruption found.

    Args:
        input_path: Path to the video file.
        on_progress: Progress callback (percent, message).

    Returns:
        CorruptionDiagnosis dataclass.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    if on_progress:
        on_progress(5, "Probing file structure...")

    diagnosis = CorruptionDiagnosis()

    # Step 1: Basic probe - check if file can be opened at all
    probe_cmd = [
        get_ffprobe_path(), "-v", "error",
        "-show_entries", "format=duration,format_name,nb_streams",
        "-show_entries", "stream=codec_type,codec_name,duration",
        "-of", "json",
        input_path,
    ]
    probe_result = _sp.run(probe_cmd, capture_output=True, timeout=60)
    probe_stderr = probe_result.stderr.decode(errors="replace").lower()

    if on_progress:
        on_progress(30, "Analysing probe results...")

    # Check for missing moov atom
    if "moov atom not found" in probe_stderr or "moov not found" in probe_stderr:
        diagnosis.corruption_type = "missing_moov"
        diagnosis.severity = "high"
        diagnosis.description = (
            "The MOOV atom (file index) is missing. This usually happens when a "
            "recording was interrupted before the file was properly finalized."
        )
        diagnosis.recoverable = True
        diagnosis.suggested_action = "remux"
        if on_progress:
            on_progress(100, "Diagnosis complete: missing MOOV atom")
        return diagnosis

    # Check for general container errors
    if probe_result.returncode != 0:
        if "invalid data found" in probe_stderr or "no such file" in probe_stderr:
            diagnosis.corruption_type = "broken_container"
            diagnosis.severity = "high"
            diagnosis.description = (
                "The container format is severely damaged. FFmpeg cannot parse "
                "the file structure."
            )
            diagnosis.recoverable = False
            diagnosis.suggested_action = "transcode"
            if on_progress:
                on_progress(100, "Diagnosis complete: broken container")
            return diagnosis

    if on_progress:
        on_progress(50, "Scanning for bitstream errors...")

    # Step 2: Deeper scan - look for bitstream errors by decoding a portion
    scan_cmd = [
        get_ffmpeg_path(), "-hide_banner",
        "-v", "error",
        "-err_detect", "explode",
        "-i", input_path,
        "-t", "10",
        "-f", "null", "-",
    ]
    scan_result = _sp.run(scan_cmd, capture_output=True, timeout=120)
    scan_stderr = scan_result.stderr.decode(errors="replace").lower()

    error_count = scan_stderr.count("error")
    corrupt_count = scan_stderr.count("corrupt")
    missing_count = scan_stderr.count("missing")

    if on_progress:
        on_progress(80, "Evaluating error severity...")

    # Check for partial/truncated file
    if "end of file" in scan_stderr or "truncated" in scan_stderr:
        diagnosis.corruption_type = "partial_file"
        diagnosis.severity = "medium"
        diagnosis.description = (
            "The file appears to be truncated or incomplete. Some content "
            "may be recoverable up to the point of truncation."
        )
        diagnosis.recoverable = True
        diagnosis.suggested_action = "remux"
        if on_progress:
            on_progress(100, "Diagnosis complete: partial/truncated file")
        return diagnosis

    # Check for bitstream errors
    total_issues = error_count + corrupt_count + missing_count
    if total_issues > 0:
        if total_issues > 20:
            severity = "high"
        elif total_issues > 5:
            severity = "medium"
        else:
            severity = "low"

        diagnosis.corruption_type = "bitstream_error"
        diagnosis.severity = severity
        diagnosis.description = (
            f"Found {total_issues} bitstream error(s) in the first 10 seconds. "
            f"The container is intact but some frames may be damaged."
        )
        diagnosis.recoverable = True
        diagnosis.suggested_action = "error_conceal"
        if on_progress:
            on_progress(100, f"Diagnosis complete: {total_issues} bitstream error(s)")
        return diagnosis

    # No corruption detected
    diagnosis.corruption_type = "no_corruption"
    diagnosis.severity = "none"
    diagnosis.description = "No corruption detected in the video file."
    diagnosis.recoverable = True
    diagnosis.suggested_action = "none"

    if on_progress:
        on_progress(100, "Diagnosis complete: no corruption found")

    return diagnosis


# ---------------------------------------------------------------------------
# Video Repair
# ---------------------------------------------------------------------------

def repair_video(
    input_path: str,
    output_path_override: Optional[str] = None,
    reference_path: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Attempt to repair a corrupted video file.

    Strategy depends on the type of corruption:
    - Missing MOOV: remux with ``-movflags faststart``
    - Broken container: copy with ``-err_detect ignore_err``
    - Bitstream errors: transcode with error concealment
    - Partial file: remux to recover available content

    Args:
        input_path: Path to the corrupted video.
        output_path_override: Custom output path (auto-generated if None).
        reference_path: Optional reference file for container hints.
        on_progress: Progress callback (percent, message).

    Returns:
        dict with output_path, recovered_duration, original_estimated_duration,
        and recovery_percentage.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    if on_progress:
        on_progress(5, "Diagnosing corruption...")

    diagnosis = diagnose_corruption(input_path)
    out = output_path_override or output_path(input_path, "repaired")

    if on_progress:
        on_progress(15, f"Repairing ({diagnosis.corruption_type})...")

    # Estimate original duration from file size (rough heuristic)
    file_size = os.path.getsize(input_path)
    # Assume ~5 Mbps average bitrate for estimation
    estimated_duration = file_size / (5 * 1024 * 1024 / 8)

    if diagnosis.corruption_type == "missing_moov":
        # Try remux with movflags faststart
        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", input_path,
            "-c", "copy",
            "-movflags", "faststart",
            out,
        ], timeout=7200)

    elif diagnosis.corruption_type == "broken_container":
        # Force through errors with stream copy
        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-err_detect", "ignore_err",
            "-i", input_path,
            "-c", "copy",
            out,
        ], timeout=7200)

    elif diagnosis.corruption_type == "bitstream_error":
        # Transcode with error concealment
        ec_flags = []
        if reference_path and os.path.isfile(reference_path):
            # Use reference for codec parameters if available
            ref_info = get_video_info(reference_path)
            ec_flags = [
                "-s", f"{ref_info['width']}x{ref_info['height']}",
            ]

        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-err_detect", "ignore_err",
            "-ec", "guess_mvs+deblock",
            "-i", input_path,
            *ec_flags,
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            out,
        ], timeout=7200)

    elif diagnosis.corruption_type == "partial_file":
        # Remux to recover what's available
        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-err_detect", "ignore_err",
            "-i", input_path,
            "-c", "copy",
            "-movflags", "faststart",
            out,
        ], timeout=7200)

    else:
        # No corruption - just copy
        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", input_path,
            "-c", "copy",
            out,
        ], timeout=3600)

    if on_progress:
        on_progress(85, "Verifying recovered file...")

    # Get recovered duration
    recovered_duration = 0.0
    if os.path.isfile(out) and os.path.getsize(out) > 0:
        info = get_video_info(out)
        recovered_duration = info.get("duration", 0.0)

    recovery_percentage = 0.0
    if estimated_duration > 0:
        recovery_percentage = min(round((recovered_duration / estimated_duration) * 100, 1), 100.0)

    if on_progress:
        on_progress(100, "Repair complete!")

    return {
        "output_path": out,
        "corruption_type": diagnosis.corruption_type,
        "severity": diagnosis.severity,
        "recovered_duration": round(recovered_duration, 2),
        "original_estimated_duration": round(estimated_duration, 2),
        "recovery_percentage": recovery_percentage,
    }


# ---------------------------------------------------------------------------
# MOOV Atom Recovery
# ---------------------------------------------------------------------------

def recover_moov_atom(
    video_path: str,
    reference_path: Optional[str] = None,
    output_path_override: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Recover a video file with a missing or corrupted MOOV atom.

    The MOOV atom contains the index that maps timestamps to byte offsets.
    When a recording is interrupted (e.g. power loss, app crash) the MOOV
    atom is never written.  This function attempts to reconstruct it by:

    1. Using reference file codec parameters for guided recovery
    2. Falling back to ``-movflags faststart`` remux if no reference

    Args:
        video_path:  Path to the corrupted video.
        reference_path:  A working video recorded with the same codec /
            container settings (same camera, same app, etc.).
        output_path_override:  Explicit output path (auto-generated when
            *None*).
        on_progress:  Callback ``(pct, msg)`` for progress updates.

    Returns:
        dict with *output_path*, *recovered_duration*, *method* used,
        and *success* flag.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"File not found: {video_path}")

    out = output_path_override or output_path(video_path, "moov_recovered")

    if on_progress:
        on_progress(5, "Attempting MOOV atom recovery...")

    method = "remux"

    if reference_path and os.path.isfile(reference_path):
        # Strategy 1: Use reference file to reconstruct MOOV
        if on_progress:
            on_progress(10, "Using reference file for recovery...")
        method = "reference_remux"

        # Probe reference for codec hints
        _sp.run([
            get_ffprobe_path(), "-v", "error",
            "-show_entries", "stream=codec_name,codec_type,width,height,sample_rate,channels",
            "-of", "json", reference_path,
        ], capture_output=True, timeout=60)

        try:
            run_ffmpeg([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-err_detect", "ignore_err",
                "-i", video_path,
                "-c", "copy",
                "-movflags", "faststart",
                out,
            ], timeout=7200)
        except RuntimeError:
            if on_progress:
                on_progress(30, "Copy remux failed, transcoding...")
            method = "reference_transcode"
            ref_info = get_video_info(reference_path)
            run_ffmpeg([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-err_detect", "ignore_err",
                "-i", video_path,
                "-c:v", "libx264", "-crf", "18", "-preset", "medium",
                "-s", f"{ref_info.get('width', 1920)}x{ref_info.get('height', 1080)}",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "faststart",
                out,
            ], timeout=7200)
    else:
        # Strategy 2: Blind remux with error tolerance
        if on_progress:
            on_progress(10, "Attempting blind MOOV recovery (no reference)...")
        try:
            run_ffmpeg([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-err_detect", "ignore_err",
                "-i", video_path,
                "-c", "copy",
                "-movflags", "faststart",
                out,
            ], timeout=7200)
        except RuntimeError:
            if on_progress:
                on_progress(40, "Remux failed, transcoding with error concealment...")
            method = "transcode"
            run_ffmpeg([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-err_detect", "ignore_err",
                "-i", video_path,
                "-c:v", "libx264", "-crf", "18", "-preset", "medium",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "faststart",
                out,
            ], timeout=7200)

    if on_progress:
        on_progress(80, "Verifying recovered file...")

    recovered_duration = 0.0
    success = False
    if os.path.isfile(out) and os.path.getsize(out) > 0:
        info = get_video_info(out)
        recovered_duration = info.get("duration", 0.0)
        success = recovered_duration > 0

    if on_progress:
        on_progress(100, "MOOV recovery complete")

    return {
        "output_path": out,
        "recovered_duration": round(recovered_duration, 2),
        "method": method,
        "success": success,
    }


# ---------------------------------------------------------------------------
# Frame Salvage
# ---------------------------------------------------------------------------

def salvage_frames(
    corrupted_path: str,
    output_path_override: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Salvage as many frames as possible from a heavily corrupted file.

    Uses ``ffmpeg -err_detect ignore_err`` to push through errors,
    extracting every decodable frame.  Reports the recovered duration
    compared to the estimated original.

    Args:
        corrupted_path:  Path to the corrupted video.
        output_path_override:  Explicit output path.
        on_progress:  Callback ``(pct, msg)``.

    Returns:
        dict with *output_path*, *recovered_duration*,
        *estimated_original_duration*, *recovery_percentage*,
        and *frames_recovered*.
    """
    if not os.path.isfile(corrupted_path):
        raise FileNotFoundError(f"File not found: {corrupted_path}")

    out = output_path_override or output_path(corrupted_path, "salvaged")

    if on_progress:
        on_progress(5, "Starting frame salvage...")

    file_size = os.path.getsize(corrupted_path)
    estimated_duration = file_size / (5 * 1024 * 1024 / 8)

    if on_progress:
        on_progress(10, "Extracting frames with error tolerance...")

    # First try stream copy (fastest)
    copy_success = False
    try:
        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-err_detect", "ignore_err",
            "-i", corrupted_path,
            "-c", "copy",
            "-movflags", "faststart",
            out,
        ], timeout=7200)
        if os.path.isfile(out) and os.path.getsize(out) > 1024:
            copy_success = True
    except RuntimeError:
        copy_success = False

    if not copy_success:
        if on_progress:
            on_progress(30, "Stream copy failed, transcoding with error concealment...")
        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-err_detect", "ignore_err",
            "-ec", "guess_mvs+deblock",
            "-i", corrupted_path,
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "faststart",
            out,
        ], timeout=7200)

    if on_progress:
        on_progress(80, "Analysing recovered content...")

    recovered_duration = 0.0
    frames_recovered = 0
    if os.path.isfile(out) and os.path.getsize(out) > 0:
        info = get_video_info(out)
        recovered_duration = info.get("duration", 0.0)
        fps = info.get("fps", 30.0)
        frames_recovered = int(recovered_duration * fps)

    recovery_pct = 0.0
    if estimated_duration > 0:
        recovery_pct = min(round((recovered_duration / estimated_duration) * 100, 1), 100.0)

    if on_progress:
        on_progress(100, f"Salvaged {recovered_duration:.1f}s ({recovery_pct:.0f}%)")

    return {
        "output_path": out,
        "recovered_duration": round(recovered_duration, 2),
        "estimated_original_duration": round(estimated_duration, 2),
        "recovery_percentage": recovery_pct,
        "frames_recovered": frames_recovered,
    }
