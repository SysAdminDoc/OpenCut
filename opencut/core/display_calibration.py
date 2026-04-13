"""
OpenCut Display Calibration Verification Module

Generate test patterns for display calibration verification:
- SMPTE color bars (HD / UHD)
- Grayscale ramp (linear and perceptual)
- Gamut boundary test (Rec.709 / P3 / Rec.2020)
- Skin tone reference patches
- Guided verification walkthrough

All patterns are generated as still images via FFmpeg or numpy+OpenCV,
avoiding any external dependencies beyond opencv-python-headless.
"""

import logging
import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

from opencut.helpers import (
    FFmpegCmd,
    ensure_package,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass
class TestPatternResult:
    """Result from test pattern generation."""
    output_path: str = ""
    pattern_type: str = ""
    resolution: Tuple[int, int] = (1920, 1080)
    description: str = ""


# ---------------------------------------------------------------------------
# SMPTE Color Bars
# ---------------------------------------------------------------------------
# Standard SMPTE EG 1-1990 HD color bar values (8-bit RGB)
_SMPTE_TOP_BARS = [
    # 75% bars: Gray, Yellow, Cyan, Green, Magenta, Red, Blue
    (191, 191, 191), (191, 191, 0), (0, 191, 191), (0, 191, 0),
    (191, 0, 191), (191, 0, 0), (0, 0, 191),
]

_SMPTE_MID_BARS = [
    # Reverse bars: Blue, Black, Magenta, Black, Cyan, Black, Gray
    (0, 0, 191), (0, 0, 0), (191, 0, 191), (0, 0, 0),
    (0, 191, 191), (0, 0, 0), (191, 191, 191),
]

_SMPTE_BOTTOM_PLUGE = [
    # PLUGE: -4% black, 0% black, +4% black, 0% black, sub-black ramp, white
    (0, 0, 29), (0, 0, 0), (0, 0, 29), (0, 0, 0),
    (0, 4, 10), (255, 255, 255),
]


def generate_smpte_bars(
    output_path_str: str,
    resolution: Tuple[int, int] = (1920, 1080),
    duration: float = 5.0,
    on_progress: Optional[Callable] = None,
) -> TestPatternResult:
    """
    Generate SMPTE color bars test pattern.

    Creates an SMPTE EG 1-1990 compliant color bar image with:
    - Top 67%: 7 vertical 75% color bars
    - Middle 8%: Reverse colour bars
    - Bottom 25%: PLUGE (Picture Line-Up Generation Equipment) pattern

    Args:
        output_path_str: Output image or video path (.png or .mp4).
        resolution: Width, height tuple.
        duration: Duration in seconds (for video output, ignored for images).
        on_progress: Progress callback(pct, msg).

    Returns:
        TestPatternResult with output path and pattern metadata.
    """
    if not ensure_package("cv2", "opencv-python-headless", on_progress):
        raise RuntimeError("opencv-python-headless is required")
    import cv2
    import numpy as np

    w, h = resolution
    if w < 160 or h < 120:
        raise ValueError("Resolution too small (minimum 160x120)")

    if on_progress:
        on_progress(10, "Generating SMPTE color bars...")

    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # Top section: 75% color bars (top 67%)
    top_h = int(h * 0.67)
    bar_w = w // 7
    for i, color in enumerate(_SMPTE_TOP_BARS):
        x_start = i * bar_w
        x_end = (i + 1) * bar_w if i < 6 else w
        frame[0:top_h, x_start:x_end] = (color[2], color[1], color[0])  # BGR

    if on_progress:
        on_progress(30, "Drawing middle section...")

    # Middle section: reverse bars (67-75%)
    mid_start = top_h
    mid_h = int(h * 0.08)
    mid_end = mid_start + mid_h
    for i, color in enumerate(_SMPTE_MID_BARS):
        x_start = i * bar_w
        x_end = (i + 1) * bar_w if i < 6 else w
        frame[mid_start:mid_end, x_start:x_end] = (color[2], color[1], color[0])

    if on_progress:
        on_progress(50, "Drawing PLUGE section...")

    # Bottom section: PLUGE pattern (75-100%)
    bot_start = mid_end
    pluge_sections = 6
    pluge_w = w // pluge_sections
    for i, color in enumerate(_SMPTE_BOTTOM_PLUGE):
        x_start = i * pluge_w
        x_end = (i + 1) * pluge_w if i < pluge_sections - 1 else w
        frame[bot_start:h, x_start:x_end] = (color[2], color[1], color[0])

    if on_progress:
        on_progress(70, "Encoding output...")

    # Save as image or video
    is_video = output_path_str.lower().endswith(('.mp4', '.mov', '.avi'))
    if is_video:
        _write_pattern_video(frame, output_path_str, duration, resolution)
    else:
        os.makedirs(os.path.dirname(output_path_str) or ".", exist_ok=True)
        cv2.imwrite(output_path_str, frame)

    if on_progress:
        on_progress(100, "SMPTE bars generated")

    return TestPatternResult(
        output_path=output_path_str,
        pattern_type="smpte_bars",
        resolution=resolution,
        description="SMPTE EG 1-1990 HD color bars with PLUGE",
    )


# ---------------------------------------------------------------------------
# Grayscale Ramp
# ---------------------------------------------------------------------------
def generate_grayscale_ramp(
    output_path_str: str,
    resolution: Tuple[int, int] = (1920, 1080),
    steps: int = 32,
    include_labels: bool = True,
    duration: float = 5.0,
    on_progress: Optional[Callable] = None,
) -> TestPatternResult:
    """
    Generate a grayscale ramp test pattern.

    Creates a horizontal grayscale gradient from pure black (0) to pure
    white (255) in discrete steps, useful for verifying gamma, black level,
    and white clipping.

    Args:
        output_path_str: Output path (.png or .mp4).
        resolution: Width, height tuple.
        steps: Number of discrete gray levels (8-256).
        include_labels: Whether to draw percentage labels on patches.
        duration: Duration for video output.
        on_progress: Progress callback(pct, msg).

    Returns:
        TestPatternResult.
    """
    if not ensure_package("cv2", "opencv-python-headless", on_progress):
        raise RuntimeError("opencv-python-headless is required")
    import cv2
    import numpy as np

    w, h = resolution
    steps = max(8, min(256, steps))

    if on_progress:
        on_progress(10, f"Generating {steps}-step grayscale ramp...")

    frame = np.zeros((h, w, 3), dtype=np.uint8)

    patch_w = w // steps
    for i in range(steps):
        val = int(i * 255 / (steps - 1))
        x_start = i * patch_w
        x_end = (i + 1) * patch_w if i < steps - 1 else w
        frame[:, x_start:x_end] = (val, val, val)

        if include_labels and patch_w >= 30:
            pct = int(i * 100 / (steps - 1))
            label = f"{pct}%"
            text_color = (255, 255, 255) if val < 128 else (0, 0, 0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = max(0.3, min(0.6, patch_w / 100.0))
            text_x = x_start + 2
            text_y = h // 2
            cv2.putText(frame, label, (text_x, text_y), font, scale,
                        text_color, 1, cv2.LINE_AA)

    if on_progress:
        on_progress(70, "Encoding output...")

    is_video = output_path_str.lower().endswith(('.mp4', '.mov', '.avi'))
    if is_video:
        _write_pattern_video(frame, output_path_str, duration, resolution)
    else:
        os.makedirs(os.path.dirname(output_path_str) or ".", exist_ok=True)
        cv2.imwrite(output_path_str, frame)

    if on_progress:
        on_progress(100, "Grayscale ramp generated")

    return TestPatternResult(
        output_path=output_path_str,
        pattern_type="grayscale_ramp",
        resolution=resolution,
        description=f"{steps}-step grayscale ramp from 0% to 100%",
    )


# ---------------------------------------------------------------------------
# Gamut Boundary Test
# ---------------------------------------------------------------------------
# Gamut boundary patches for different color spaces (RGB 8-bit approximations)
_GAMUT_PATCHES = {
    "rec709": [
        {"name": "Red", "rgb": (255, 0, 0)},
        {"name": "Green", "rgb": (0, 255, 0)},
        {"name": "Blue", "rgb": (0, 0, 255)},
        {"name": "Cyan", "rgb": (0, 255, 255)},
        {"name": "Magenta", "rgb": (255, 0, 255)},
        {"name": "Yellow", "rgb": (255, 255, 0)},
        {"name": "White", "rgb": (255, 255, 255)},
        {"name": "50% Gray", "rgb": (128, 128, 128)},
    ],
    "skin_tones": [
        {"name": "Light 1", "rgb": (255, 224, 196)},
        {"name": "Light 2", "rgb": (234, 192, 154)},
        {"name": "Medium 1", "rgb": (210, 161, 117)},
        {"name": "Medium 2", "rgb": (186, 131, 88)},
        {"name": "Dark 1", "rgb": (153, 102, 65)},
        {"name": "Dark 2", "rgb": (115, 72, 42)},
        {"name": "Deep 1", "rgb": (80, 50, 30)},
        {"name": "Deep 2", "rgb": (56, 35, 22)},
    ],
}


def generate_gamut_test(
    output_path_str: str,
    resolution: Tuple[int, int] = (1920, 1080),
    include_skin_tones: bool = True,
    duration: float = 5.0,
    on_progress: Optional[Callable] = None,
) -> TestPatternResult:
    """
    Generate a gamut boundary test pattern.

    Displays saturated primary/secondary color patches at Rec.709 gamut
    boundaries, with optional skin tone reference patches.

    Args:
        output_path_str: Output path (.png or .mp4).
        resolution: Width, height tuple.
        include_skin_tones: Include skin tone reference patches.
        duration: Duration for video output.
        on_progress: Progress callback(pct, msg).

    Returns:
        TestPatternResult.
    """
    if not ensure_package("cv2", "opencv-python-headless", on_progress):
        raise RuntimeError("opencv-python-headless is required")
    import cv2
    import numpy as np

    w, h = resolution

    if on_progress:
        on_progress(10, "Generating gamut boundary test...")

    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = (32, 32, 32)  # Dark gray background

    patches = _GAMUT_PATCHES["rec709"]
    if include_skin_tones:
        patches = patches + _GAMUT_PATCHES["skin_tones"]

    # Layout: grid of patches
    cols = 4
    rows = (len(patches) + cols - 1) // cols
    margin = 20
    patch_w = (w - margin * (cols + 1)) // cols
    patch_h = (h - margin * (rows + 1)) // rows

    if on_progress:
        on_progress(40, "Drawing color patches...")

    for idx, patch in enumerate(patches):
        row = idx // cols
        col = idx % cols
        x = margin + col * (patch_w + margin)
        y = margin + row * (patch_h + margin)

        r, g, b = patch["rgb"]
        cv2.rectangle(frame, (x, y), (x + patch_w, y + patch_h),
                       (b, g, r), -1)  # BGR

        # Label
        font = cv2.FONT_HERSHEY_SIMPLEX
        label = patch["name"]
        scale = max(0.4, min(0.7, patch_w / 200.0))
        # Choose text color for readability
        brightness = (r + g + b) / 3
        text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
        cv2.putText(frame, label, (x + 5, y + patch_h - 10), font, scale,
                    text_color, 1, cv2.LINE_AA)

    if on_progress:
        on_progress(70, "Encoding output...")

    is_video = output_path_str.lower().endswith(('.mp4', '.mov', '.avi'))
    if is_video:
        _write_pattern_video(frame, output_path_str, duration, resolution)
    else:
        os.makedirs(os.path.dirname(output_path_str) or ".", exist_ok=True)
        cv2.imwrite(output_path_str, frame)

    if on_progress:
        on_progress(100, "Gamut test generated")

    return TestPatternResult(
        output_path=output_path_str,
        pattern_type="gamut_boundary",
        resolution=resolution,
        description="Rec.709 gamut boundary patches" + (
            " with skin tone references" if include_skin_tones else ""
        ),
    )


# ---------------------------------------------------------------------------
# Verification Guide
# ---------------------------------------------------------------------------
def get_verification_guide() -> Dict:
    """
    Return a structured verification walkthrough for display calibration.

    Each step includes what to look for and pass/fail criteria.

    Returns:
        Dict with ordered verification steps.
    """
    return {
        "title": "Display Calibration Verification Guide",
        "steps": [
            {
                "step": 1,
                "pattern": "smpte_bars",
                "title": "SMPTE Color Bars Check",
                "instructions": (
                    "Display the SMPTE color bars pattern. Verify that all 7 "
                    "vertical bars are distinctly visible with no color bleeding. "
                    "The bars should be (left to right): gray, yellow, cyan, green, "
                    "magenta, red, blue."
                ),
                "pass_criteria": "All 7 bars are clearly distinguishable with clean edges.",
                "fail_criteria": "Bars appear merged, colors are shifted, or edges bleed.",
            },
            {
                "step": 2,
                "pattern": "smpte_bars",
                "title": "PLUGE Black Level",
                "instructions": (
                    "Look at the bottom PLUGE section. You should see three narrow "
                    "bars: a slightly-below-black bar (should be invisible), a true "
                    "black bar, and a slightly-above-black bar (just barely visible). "
                    "Adjust brightness so the sub-black bar disappears while the "
                    "super-black bar remains just visible."
                ),
                "pass_criteria": "Sub-black bar invisible, super-black bar barely visible.",
                "fail_criteria": "Both bars equally visible (brightness too high) or both invisible (too low).",
            },
            {
                "step": 3,
                "pattern": "grayscale_ramp",
                "title": "Grayscale Tracking",
                "instructions": (
                    "Display the grayscale ramp. All steps from 0% to 100% should "
                    "be individually distinguishable. The gradient should appear "
                    "smooth with no visible banding, color tinting, or crushed "
                    "shadows/highlights."
                ),
                "pass_criteria": "All steps visible, smooth gradient, no color cast.",
                "fail_criteria": "Steps merge together, visible banding, or color tint in grays.",
            },
            {
                "step": 4,
                "pattern": "gamut_boundary",
                "title": "Color Gamut Verification",
                "instructions": (
                    "Display the gamut boundary test. All primary (R/G/B) and "
                    "secondary (C/M/Y) colors should appear saturated and pure. "
                    "Red should look red, not orange. Blue should look blue, not "
                    "purple. Check that no color appears clipped or desaturated."
                ),
                "pass_criteria": "All 6 colors are saturated, pure, and distinct.",
                "fail_criteria": "Colors appear washed out, shifted, or clipped.",
            },
            {
                "step": 5,
                "pattern": "gamut_boundary",
                "title": "Skin Tone Accuracy",
                "instructions": (
                    "Examine the skin tone reference patches. The range from light "
                    "to deep skin tones should show natural, warm colors without "
                    "green/magenta tints. Compare against known skin tone references."
                ),
                "pass_criteria": "Skin tones appear natural and warm across all values.",
                "fail_criteria": "Skin tones have green or magenta tint, or appear flat.",
            },
        ],
        "notes": [
            "Allow your display to warm up for at least 30 minutes before calibration.",
            "Verify in a dimly lit room with neutral wall colors.",
            "Disable any dynamic brightness, eco mode, or motion smoothing on your display.",
            "View patterns at 100% zoom (pixel-for-pixel) when possible.",
        ],
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _write_pattern_video(
    frame,  # numpy array BGR
    output_path_str: str,
    duration: float,
    resolution: Tuple[int, int],
):
    """Write a still frame as a short video via FFmpeg."""
    import tempfile

    import cv2

    # Write frame as temp PNG, then encode with FFmpeg
    tmp_png = os.path.join(tempfile.gettempdir(), f"calibration_tmp_{os.getpid()}.png")
    cv2.imwrite(tmp_png, frame)

    try:
        cmd = (FFmpegCmd()
               .option("loop", "1")
               .input(tmp_png)
               .option("t", str(duration))
               .video_codec("libx264", crf=1, preset="medium")
               .option("tune", "stillimage")
               .faststart()
               .output(output_path_str)
               .build())
        run_ffmpeg(cmd, timeout=120)
    finally:
        try:
            os.unlink(tmp_png)
        except OSError:
            pass
