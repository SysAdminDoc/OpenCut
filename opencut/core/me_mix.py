"""
OpenCut M&E Mix Export

Generate Music & Effects (M&E) mixes by removing vocals from audio.

Methods:
- "subtract": Center-channel subtraction (vocals are typically center-panned)
- "stems": Combine non-vocal stems if stem separation is available
- Fallback: FFmpeg pan filter for basic center-channel vocal removal

Uses FFmpeg pan filter for center subtraction approach.
"""

import logging
import os
from typing import Callable, Optional

from opencut.helpers import get_ffmpeg_path, run_ffmpeg
from opencut.helpers import output_path as _output_path

logger = logging.getLogger("opencut")


def generate_me_mix(
    input_path: str,
    output_path: Optional[str] = None,
    method: str = "subtract",
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Generate an M&E (Music & Effects) mix by removing vocals.

    Args:
        input_path: Source audio/video file.
        output_path: Output path (auto-generated if None).
        method: Vocal removal method:
            - "subtract": Center-channel subtraction via FFmpeg pan filter.
              Removes center-panned vocals while preserving stereo M&E.
            - "stems": Attempt stem separation and recombine non-vocal stems.
              Falls back to "subtract" if stem separation is unavailable.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path, method_used, notes.
    """
    if output_path is None:
        output_path = _output_path(input_path, "me_mix")

    method = method.lower().strip()
    if method not in ("subtract", "stems"):
        method = "subtract"

    if on_progress:
        on_progress(5, f"Generating M&E mix (method: {method})...")

    method_used = method
    notes = ""

    if method == "stems":
        # Try to use existing stem separation
        result = _try_stems_method(input_path, output_path, on_progress)
        if result is not None:
            return result
        # Fall back to subtract
        method_used = "subtract (stems unavailable)"
        notes = "Stem separation not available, fell back to center-channel subtraction."
        if on_progress:
            on_progress(20, "Stem separation unavailable, falling back to center subtraction...")

    # Center-channel subtraction method
    _center_subtract(input_path, output_path, on_progress)

    if on_progress:
        on_progress(100, "M&E mix generated")

    logger.info("Generated M&E mix (%s): %s -> %s", method_used, input_path, output_path)
    return {
        "output_path": output_path,
        "method_used": method_used,
        "notes": notes or "Center-channel subtraction removes center-panned vocals.",
    }


def _center_subtract(input_path: str, out_path: str, on_progress: Optional[Callable] = None) -> None:
    """
    Remove center-panned vocals using FFmpeg pan filter.

    pan=stereo|c0=c0-c1|c1=c1-c0 extracts the difference between L and R,
    which removes anything panned center (typically lead vocals).
    """
    if on_progress:
        on_progress(30, "Applying center-channel subtraction...")

    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-af", "pan=stereo|c0=c0-c1|c1=c1-c0",
        "-c:v", "copy",
        out_path,
    ]
    run_ffmpeg(cmd)


def _try_stems_method(
    input_path: str,
    out_path: str,
    on_progress: Optional[Callable] = None,
) -> Optional[dict]:
    """
    Try to generate M&E mix using stem separation.

    Attempts to import the existing stem separator, separate vocals,
    and recombine everything except vocals.

    Returns result dict on success, None if stem separation is unavailable.
    """
    try:
        from opencut.core.audio_suite import separate_stems
    except ImportError:
        return None

    if on_progress:
        on_progress(10, "Separating stems...")

    import tempfile
    stems_dir = tempfile.mkdtemp(prefix="opencut_stems_")

    try:
        # separate_stems returns path to output dir with individual stems
        stem_result = separate_stems(input_path, output_dir=stems_dir, on_progress=None)
        if not stem_result or not isinstance(stem_result, str) or not os.path.isdir(stem_result):
            return None

        # Find non-vocal stem files
        stem_files = []
        for f in os.listdir(stem_result):
            lower_f = f.lower()
            if "vocal" not in lower_f and f.endswith((".wav", ".mp3", ".flac")):
                stem_files.append(os.path.join(stem_result, f))

        if not stem_files:
            return None

        if on_progress:
            on_progress(60, f"Mixing {len(stem_files)} non-vocal stems...")

        # Mix all non-vocal stems together
        ffmpeg = get_ffmpeg_path()
        inputs = []
        for sf in stem_files:
            inputs.extend(["-i", sf])

        if len(stem_files) == 1:
            cmd = [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-i", stem_files[0],
                "-c:a", "pcm_s16le",
                out_path,
            ]
        else:
            fc = ""
            for i in range(len(stem_files)):
                fc += f"[{i}:a]"
            fc += f"amix=inputs={len(stem_files)}:duration=longest[out]"
            cmd = [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                *inputs,
                "-filter_complex", fc,
                "-map", "[out]",
                out_path,
            ]

        run_ffmpeg(cmd)

        if on_progress:
            on_progress(100, "M&E mix generated from stems")

        logger.info("Generated M&E mix from %d stems: %s", len(stem_files), out_path)
        return {
            "output_path": out_path,
            "method_used": "stems",
            "notes": f"Combined {len(stem_files)} non-vocal stems.",
        }

    except Exception as exc:
        logger.warning("Stem-based M&E mix failed, will fall back: %s", exc)
        return None
    finally:
        # Clean up temp stems
        import shutil
        shutil.rmtree(stems_dir, ignore_errors=True)
