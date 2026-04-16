"""
Loudness Matching

Analyzes and normalizes multiple audio/video clips to the same
integrated loudness (LUFS) for consistent audio across a sequence.
"""

import json
import logging
import os
import re
import subprocess as _sp
from typing import List

try:
    from ..helpers import get_ffmpeg_path
    from ..helpers import run_ffmpeg as _helpers_run_ffmpeg
except ImportError:
    try:
        from opencut.helpers import get_ffmpeg_path
        from opencut.helpers import run_ffmpeg as _helpers_run_ffmpeg
    except ImportError:
        _helpers_run_ffmpeg = None  # type: ignore
        from shutil import which as _which

        def get_ffmpeg_path() -> str:  # type: ignore[misc]
            return _which("ffmpeg") or "ffmpeg"

logger = logging.getLogger("opencut")


def _run_ffmpeg_raw(cmd: List[str], timeout: int = 3600) -> str:
    """Run an FFmpeg command and return stderr text.

    Uses consolidated helper when available (raises RuntimeError on failure,
    returns stderr as str).  Falls back to raw subprocess.
    """
    if _helpers_run_ffmpeg is not None:
        return _helpers_run_ffmpeg(cmd, timeout=timeout)
    try:
        result = _sp.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found. Install FFmpeg: https://ffmpeg.org/download.html")
    except _sp.TimeoutExpired:
        raise RuntimeError(f"FFmpeg timed out running: {' '.join(cmd[:6])}")
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {(result.stderr or '')[-500:]}")
    return result.stderr or ""


# ---------------------------------------------------------------------------
# Loudness measurement
# ---------------------------------------------------------------------------

def measure_loudness(filepath: str) -> dict:
    """
    Measure integrated loudness of an audio/video file using FFmpeg loudnorm.

    Runs FFmpeg in analysis mode (loudnorm=print_format=json) and parses the
    JSON block that is written to stderr.

    Args:
        filepath: Path to the audio or video file.

    Returns:
        Dict with "lufs" (integrated loudness, LUFS), "lra" (loudness range,
        LU), "true_peak" (true peak, dBTP). Also returns "peak" as an alias
        of true_peak for backward compatibility with existing callers that
        read either name — FFmpeg loudnorm does not expose a separate
        sample-peak measurement, so both refer to the input_tp value.

    Raises:
        RuntimeError: If FFmpeg is not installed or analysis fails.
        ValueError: If loudnorm output cannot be parsed.
    """
    cmd = [
        get_ffmpeg_path(), "-hide_banner",
        "-i", filepath,
        "-af", "loudnorm=print_format=json",
        "-vn",
        "-f", "null", "-",
    ]

    # run_ffmpeg raises RuntimeError on non-zero exit; returns stderr as str
    stderr = _run_ffmpeg_raw(cmd)
    json_match = re.search(r"\{[^{}]+\}", stderr, re.DOTALL)
    if not json_match:
        logger.debug("loudnorm stderr:\n%s", stderr[-800:])
        raise ValueError(
            f"Could not parse loudnorm JSON output for '{filepath}'. "
            "Ensure the file has an audio stream."
        )

    try:
        data = json.loads(json_match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(f"loudnorm JSON parse error: {exc}") from exc

    def _f(key: str, default: float = -99.0) -> float:
        try:
            return float(data.get(key, default))
        except (TypeError, ValueError):
            return default

    # FFmpeg ``loudnorm=print_format=json`` only exposes ``input_tp`` (true
    # peak), not a distinct sample peak. Expose it under both names so
    # callers that read ``peak`` and callers that read ``true_peak`` both
    # work, and document the alias in the return docstring above.
    _tp = _f("input_tp")
    return {
        "lufs": _f("input_i"),
        "lra": _f("input_lra"),
        "peak": _tp,
        "true_peak": _tp,
    }


# ---------------------------------------------------------------------------
# Two-pass loudness normalisation
# ---------------------------------------------------------------------------

def normalize_to_lufs(
    input_path: str,
    output_path: str,
    target_lufs: float = -14.0,
    true_peak: float = -1.0,
) -> str:
    """
    Normalize an audio/video file to a target LUFS using FFmpeg two-pass loudnorm.

    Pass 1: Measure actual loudness with loudnorm in analysis mode.
    Pass 2: Apply loudnorm with the measured values for accurate normalisation.

    Args:
        input_path: Path to the source file.
        output_path: Destination path for the normalised file.
        target_lufs: Target integrated loudness (e.g. -14.0 for YouTube, -23.0 for broadcast).
        true_peak: Maximum true peak level in dBTP.

    Returns:
        output_path on success.

    Raises:
        RuntimeError: If FFmpeg is not installed or encoding fails.
        ValueError: If the first-pass loudness measurement cannot be parsed.
    """
    target_lufs = max(-70.0, min(0.0, float(target_lufs)))
    true_peak = max(-10.0, min(0.0, float(true_peak)))

    # --- Pass 1: Measure ---
    logger.debug("Loudness normalisation pass 1 (measure): %s", input_path)
    pass1_cmd = [
        get_ffmpeg_path(), "-hide_banner",
        "-i", input_path,
        "-af", f"loudnorm=I={target_lufs}:TP={true_peak}:LRA=11:print_format=json",
        "-vn",
        "-f", "null", "-",
    ]
    stderr = _run_ffmpeg_raw(pass1_cmd)
    json_match = re.search(r"\{[^{}]+\}", stderr, re.DOTALL)
    if not json_match:
        raise ValueError(
            f"Could not parse loudnorm pass-1 output for '{input_path}'. "
            "Ensure the file has an audio stream."
        )

    try:
        measured = json.loads(json_match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(f"loudnorm pass-1 JSON parse error: {exc}") from exc

    def _mv(key: str, default: str = "0.0") -> str:
        val = str(measured.get(key, default))
        try:
            float(val)
        except (TypeError, ValueError):
            val = default
        return val

    input_i = _mv("input_i", "-70.0")
    input_tp = _mv("input_tp", "-70.0")
    input_lra = _mv("input_lra", "0.0")
    input_thresh = _mv("input_thresh", "-70.0")
    offset = _mv("target_offset", "0.0")

    # --- Pass 2: Apply with measured values ---
    logger.debug("Loudness normalisation pass 2 (apply): %s → %s", input_path, output_path)

    loudnorm_filter = (
        f"loudnorm=I={target_lufs}:TP={true_peak}:LRA=11"
        f":measured_I={input_i}"
        f":measured_TP={input_tp}"
        f":measured_LRA={input_lra}"
        f":measured_thresh={input_thresh}"
        f":offset={offset}"
        f":linear=true:print_format=none"
    )

    ext = os.path.splitext(input_path)[1].lower()
    is_video = ext in (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if is_video:
        pass2_cmd = [
            get_ffmpeg_path(), "-hide_banner", "-y",
            "-i", input_path,
            "-af", loudnorm_filter,
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            output_path,
        ]
    else:
        pass2_cmd = [
            get_ffmpeg_path(), "-hide_banner", "-y",
            "-i", input_path,
            "-af", loudnorm_filter,
            "-c:a", "aac", "-b:a", "192k",
            output_path,
        ]

    # run_ffmpeg raises RuntimeError on non-zero exit
    _run_ffmpeg_raw(pass2_cmd)

    logger.info("Normalised to %.1f LUFS: %s", target_lufs, output_path)
    return output_path


# ---------------------------------------------------------------------------
# Batch loudness matching
# ---------------------------------------------------------------------------

def batch_loudness_match(
    filepaths: List[str],
    output_dir: str,
    target_lufs: float = -14.0,
    on_progress=None,
) -> List[dict]:
    """
    Normalize a list of files to the same integrated loudness target.

    Args:
        filepaths: List of paths to audio/video files.
        output_dir: Directory for normalised output files.
        target_lufs: Target integrated loudness (LUFS).

    Returns:
        List of result dicts, one per input file:
            {"input": str, "output": str, "original_lufs": float, "job_ok": bool}
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for i, filepath in enumerate(filepaths):
        logger.debug("Batch loudness match %d/%d: %s", i + 1, len(filepaths), filepath)

        base = os.path.splitext(os.path.basename(filepath))[0]
        ext = os.path.splitext(filepath)[1] or ".mp4"
        output_path = os.path.join(output_dir, f"{base}_normalized{ext}")

        original_lufs = -99.0
        job_ok = False

        try:
            stats = measure_loudness(filepath)
            original_lufs = stats.get("lufs", -99.0)
            normalize_to_lufs(filepath, output_path, target_lufs=target_lufs)
            job_ok = True
        except Exception as exc:
            logger.error("Loudness match failed for %s: %s", filepath, exc)
            output_path = ""

        results.append({
            "input": filepath,
            "output": output_path,
            "original_lufs": round(original_lufs, 2),
            "job_ok": job_ok,
        })

        if on_progress:
            on_progress(int((i + 1) / len(filepaths) * 100))

    ok_count = sum(1 for r in results if r["job_ok"])
    logger.info("Batch loudness match complete: %d/%d succeeded", ok_count, len(filepaths))
    return results
