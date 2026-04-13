"""
OpenCut GIF / Animated WebP / APNG Export Module v1.0.0

High-quality animated image exports with optimization:
- GIF with two-pass palettegen/paletteuse and configurable dithering
- Animated WebP via libwebp_anim
- Animated PNG (APNG) via ffmpeg -f apng

All functions follow the core module pattern: dataclass-style dict results,
optional on_progress callback, and FFmpeg-based processing.
"""

import logging
import os
import tempfile
from typing import Callable, Dict, Optional

from opencut.helpers import get_ffmpeg_path, run_ffmpeg

logger = logging.getLogger("opencut")

# Valid dither algorithms for GIF palettegen
VALID_DITHERS = frozenset({
    "bayer", "heckbert", "floyd_steinberg", "sierra2",
    "sierra2_4a", "sierra3", "burkes", "atkinson", "none",
})


def _file_size_bytes(path: str) -> int:
    """Return file size in bytes, 0 if missing."""
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _get_dimensions(path: str) -> Dict:
    """Get width/height of an image/video file via ffprobe."""
    import json as _json
    import subprocess as _sp

    from opencut.helpers import get_ffprobe_path

    cmd = [
        get_ffprobe_path(), "-v", "quiet", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json", path,
    ]
    try:
        result = _sp.run(cmd, capture_output=True, timeout=15)
        if result.returncode == 0:
            data = _json.loads(result.stdout.decode())
            streams = data.get("streams", [])
            if streams:
                return {
                    "width": int(streams[0].get("width", 0)),
                    "height": int(streams[0].get("height", 0)),
                }
    except Exception:
        pass
    return {"width": 0, "height": 0}


# ---------------------------------------------------------------------------
# GIF Export
# ---------------------------------------------------------------------------

def export_gif(
    input_path: str,
    output_path: Optional[str] = None,
    max_width: int = 480,
    fps: int = 15,
    max_colors: int = 256,
    dither: str = "sierra2_4a",
    loop: int = 0,
    max_file_size_mb: Optional[float] = None,
    on_progress: Optional[Callable] = None,
) -> Dict:
    """
    Export video as optimized GIF using two-pass palette generation.

    Pass 1: palettegen with configurable max_colors and stats_mode=diff
    Pass 2: paletteuse with configurable dithering algorithm

    If max_file_size_mb is specified, iteratively reduces resolution and
    colors until the output fits within the target size.

    Args:
        input_path:       Source video file path.
        output_path:      Destination GIF path (auto-generated if None).
        max_width:        Maximum width in pixels (height scales proportionally).
        fps:              Frame rate for the GIF.
        max_colors:       Maximum palette colors (2-256).
        dither:           Dithering algorithm name.
        loop:             Loop count (0 = infinite).
        max_file_size_mb: Optional file size cap in megabytes.
        on_progress:      Optional callback(percent, message).

    Returns:
        Dict with keys: output_path, file_size, width, height.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Validate parameters
    max_width = max(16, min(max_width, 3840))
    fps = max(1, min(fps, 60))
    max_colors = max(2, min(max_colors, 256))
    if dither not in VALID_DITHERS:
        dither = "sierra2_4a"
    loop = max(0, loop)

    if output_path is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        directory = os.path.dirname(input_path)
        output_path = os.path.join(directory, f"{base}_optimized.gif")

    if on_progress:
        on_progress(5, "Starting GIF export...")

    # Attempt export; if max_file_size_mb set, iterate with reducing quality
    current_width = max_width
    current_colors = max_colors
    attempt = 0
    max_attempts = 6

    while True:
        attempt += 1
        _export_gif_twopass(
            input_path, output_path,
            width=current_width,
            fps=fps,
            max_colors=current_colors,
            dither=dither,
            loop=loop,
            on_progress=on_progress,
            progress_base=5 if attempt == 1 else 50,
        )

        file_size = _file_size_bytes(output_path)

        if max_file_size_mb is None:
            break

        target_bytes = int(max_file_size_mb * 1024 * 1024)
        if file_size <= target_bytes:
            break

        if attempt >= max_attempts:
            logger.warning(
                "GIF export: could not meet %sMB target after %d attempts "
                "(final size: %.1fMB)",
                max_file_size_mb, attempt, file_size / (1024 * 1024),
            )
            break

        # Reduce quality: shrink width by 20% and halve colors (min 16)
        current_width = max(16, int(current_width * 0.8))
        current_colors = max(16, current_colors // 2)

        if on_progress:
            on_progress(
                50, f"File too large ({file_size / (1024*1024):.1f}MB), "
                    f"retrying at {current_width}px / {current_colors} colors..."
            )

    dims = _get_dimensions(output_path)
    if on_progress:
        on_progress(100, "GIF export complete")

    return {
        "output_path": output_path,
        "file_size": file_size,
        "width": dims["width"],
        "height": dims["height"],
    }


def _export_gif_twopass(
    input_path: str,
    output_path: str,
    width: int,
    fps: int,
    max_colors: int,
    dither: str,
    loop: int,
    on_progress: Optional[Callable],
    progress_base: int = 5,
) -> None:
    """Internal two-pass GIF encoding."""
    palette_fd = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    palette_path = palette_fd.name
    palette_fd.close()

    scale_filter = f"fps={fps},scale={width}:-1:flags=lanczos"

    try:
        # Pass 1: Generate color palette
        if on_progress:
            on_progress(progress_base + 10, "Generating color palette...")

        palettegen_filter = (
            f"{scale_filter},palettegen="
            f"max_colors={max_colors}:stats_mode=diff"
        )
        run_ffmpeg([
            get_ffmpeg_path(), "-hide_banner", "-loglevel", "error",
            "-y", "-i", input_path,
            "-vf", palettegen_filter,
            palette_path,
        ], timeout=7200)

        # Pass 2: Apply palette with dithering
        if on_progress:
            on_progress(progress_base + 40, "Encoding GIF with palette...")

        if dither == "none":
            paletteuse_opts = "dither=none:diff_mode=rectangle"
        elif dither == "bayer":
            paletteuse_opts = "dither=bayer:bayer_scale=5:diff_mode=rectangle"
        else:
            paletteuse_opts = f"dither={dither}:diff_mode=rectangle"

        lavfi = (
            f"{scale_filter}[x];[x][1:v]paletteuse={paletteuse_opts}"
        )
        loop_args = ["-loop", str(loop)] if loop >= 0 else []

        run_ffmpeg([
            get_ffmpeg_path(), "-hide_banner", "-loglevel", "error",
            "-y", "-i", input_path, "-i", palette_path,
            "-lavfi", lavfi,
            *loop_args,
            output_path,
        ], timeout=7200)

        if on_progress:
            on_progress(progress_base + 80, "GIF encoding complete")
    finally:
        if os.path.exists(palette_path):
            os.unlink(palette_path)


# ---------------------------------------------------------------------------
# Animated WebP Export
# ---------------------------------------------------------------------------

def export_webp(
    input_path: str,
    output_path: Optional[str] = None,
    max_width: int = 480,
    fps: int = 15,
    quality: int = 75,
    loop: int = 0,
    on_progress: Optional[Callable] = None,
) -> Dict:
    """
    Export video as animated WebP using libwebp_anim.

    Args:
        input_path:  Source video file path.
        output_path: Destination WebP path (auto-generated if None).
        max_width:   Maximum width in pixels (height scales proportionally).
        fps:         Frame rate for the animation.
        quality:     WebP quality (0-100, higher = better).
        loop:        Loop count (0 = infinite).
        on_progress: Optional callback(percent, message).

    Returns:
        Dict with keys: output_path, file_size, width, height.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    max_width = max(16, min(max_width, 3840))
    fps = max(1, min(fps, 60))
    quality = max(0, min(quality, 100))
    loop = max(0, loop)

    if output_path is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        directory = os.path.dirname(input_path)
        output_path = os.path.join(directory, f"{base}_animated.webp")

    if on_progress:
        on_progress(5, "Starting animated WebP export...")

    vf = f"fps={fps},scale={max_width}:-1:flags=lanczos"

    run_ffmpeg([
        get_ffmpeg_path(), "-hide_banner", "-loglevel", "error",
        "-y", "-i", input_path,
        "-vf", vf,
        "-c:v", "libwebp_anim",
        "-quality", str(quality),
        "-loop", str(loop),
        "-an",
        output_path,
    ], timeout=7200)

    if on_progress:
        on_progress(90, "Animated WebP encoding complete")

    file_size = _file_size_bytes(output_path)
    dims = _get_dimensions(output_path)

    if on_progress:
        on_progress(100, "Animated WebP export complete")

    return {
        "output_path": output_path,
        "file_size": file_size,
        "width": dims["width"],
        "height": dims["height"],
    }


# ---------------------------------------------------------------------------
# Animated PNG (APNG) Export
# ---------------------------------------------------------------------------

def export_apng(
    input_path: str,
    output_path: Optional[str] = None,
    max_width: int = 480,
    fps: int = 15,
    on_progress: Optional[Callable] = None,
) -> Dict:
    """
    Export video as animated PNG (APNG).

    Args:
        input_path:  Source video file path.
        output_path: Destination APNG path (auto-generated if None).
        max_width:   Maximum width in pixels (height scales proportionally).
        fps:         Frame rate for the animation.
        on_progress: Optional callback(percent, message).

    Returns:
        Dict with keys: output_path, file_size, width, height.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    max_width = max(16, min(max_width, 3840))
    fps = max(1, min(fps, 60))

    if output_path is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        directory = os.path.dirname(input_path)
        output_path = os.path.join(directory, f"{base}_animated.apng")

    if on_progress:
        on_progress(5, "Starting APNG export...")

    vf = f"fps={fps},scale={max_width}:-1:flags=lanczos"

    run_ffmpeg([
        get_ffmpeg_path(), "-hide_banner", "-loglevel", "error",
        "-y", "-i", input_path,
        "-vf", vf,
        "-f", "apng",
        "-plays", "0",
        "-an",
        output_path,
    ], timeout=7200)

    if on_progress:
        on_progress(90, "APNG encoding complete")

    file_size = _file_size_bytes(output_path)
    dims = _get_dimensions(output_path)

    if on_progress:
        on_progress(100, "APNG export complete")

    return {
        "output_path": output_path,
        "file_size": file_size,
        "width": dims["width"],
        "height": dims["height"],
    }
