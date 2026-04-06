"""
OpenCut LUT Library Module v0.8.1

Color grading via Look-Up Tables (LUTs):
- Built-in .cube LUT generator for common cinematic looks
- Apply .cube LUT files via FFmpeg lut3d filter
- Intensity/blend control (mix with original)
- LUT preview (apply to single frame)
- Support for user-provided .cube files

Generates mathematically-defined LUTs as .cube files,
then applies via FFmpeg - zero external dependencies.
"""

import logging
import math
import os
from typing import Callable, Dict, List, Optional

from opencut.helpers import get_ffmpeg_path, run_ffmpeg

logger = logging.getLogger("opencut")

LUTS_DIR = os.path.join(os.path.expanduser("~"), ".opencut", "luts")


# ---------------------------------------------------------------------------
# LUT Generation Helpers
# ---------------------------------------------------------------------------
def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _generate_cube_lut(name: str, transform_fn, size: int = 33) -> str:
    """Generate a .cube LUT file using a color transform function."""
    os.makedirs(LUTS_DIR, exist_ok=True)
    path = os.path.join(LUTS_DIR, f"{name}.cube")

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"TITLE \"{name}\"\n")
        f.write(f"LUT_SIZE {size}\n\n")

        for b_i in range(size):
            for g_i in range(size):
                for r_i in range(size):
                    r = r_i / (size - 1)
                    g = g_i / (size - 1)
                    b = b_i / (size - 1)

                    nr, ng, nb = transform_fn(r, g, b)
                    nr = _clamp(nr)
                    ng = _clamp(ng)
                    nb = _clamp(nb)

                    f.write(f"{nr:.6f} {ng:.6f} {nb:.6f}\n")

    return path


# ---------------------------------------------------------------------------
# Built-in LUT Transforms
# ---------------------------------------------------------------------------
def _teal_orange(r, g, b):
    """Teal & Orange - Hollywood blockbuster look."""
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    # Push shadows toward teal, highlights toward orange
    if lum < 0.5:
        t = lum / 0.5
        nr = r * 0.7 + 0.0 * (1 - t)
        ng = g * 0.85 + 0.3 * (1 - t) * 0.3
        nb = b * 0.9 + 0.5 * (1 - t) * 0.4
    else:
        t = (lum - 0.5) / 0.5
        nr = r * 0.9 + 0.15 * t
        ng = g * 0.85 + 0.05 * t
        nb = b * 0.7 - 0.1 * t
    # Slight contrast boost
    nr = _clamp((nr - 0.5) * 1.15 + 0.5)
    ng = _clamp((ng - 0.5) * 1.1 + 0.5)
    nb = _clamp((nb - 0.5) * 1.1 + 0.5)
    return nr, ng, nb


def _vintage_warm(r, g, b):
    """Warm vintage / retro film look."""
    nr = _clamp(r * 1.1 + 0.05)
    ng = _clamp(g * 0.95 + 0.02)
    nb = _clamp(b * 0.8)
    # Lift blacks slightly
    nr = _clamp(nr * 0.9 + 0.08)
    ng = _clamp(ng * 0.9 + 0.06)
    nb = _clamp(nb * 0.9 + 0.04)
    # Reduce contrast
    nr = _clamp((nr - 0.5) * 0.85 + 0.52)
    ng = _clamp((ng - 0.5) * 0.85 + 0.50)
    nb = _clamp((nb - 0.5) * 0.85 + 0.48)
    return nr, ng, nb


def _cool_desaturate(r, g, b):
    """Cool desaturated look - moody, muted tones."""
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    # Desaturate 40%
    nr = r * 0.6 + lum * 0.4
    ng = g * 0.6 + lum * 0.4
    nb = b * 0.6 + lum * 0.4
    # Cool shift
    nr = _clamp(nr * 0.92)
    ng = _clamp(ng * 0.96)
    nb = _clamp(nb * 1.08)
    return nr, ng, nb


def _high_contrast_bw(r, g, b):
    """High contrast black & white."""
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    # S-curve contrast
    lum = _clamp(1.0 / (1.0 + math.exp(-8 * (lum - 0.5))))
    return lum, lum, lum


def _sepia(r, g, b):
    """Classic sepia tone."""
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    nr = _clamp(lum * 1.2 + 0.1)
    ng = _clamp(lum * 1.0 + 0.05)
    nb = _clamp(lum * 0.75)
    return nr, ng, nb


def _cyberpunk(r, g, b):
    """Cyberpunk neon - high saturation magenta/cyan push."""
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    # Boost saturation
    nr = _clamp(r + (r - lum) * 0.5)
    ng = _clamp(g + (g - lum) * 0.3)
    nb = _clamp(b + (b - lum) * 0.5)
    # Push toward magenta/cyan split
    if lum < 0.4:
        nb = _clamp(nb + 0.15)
        ng = _clamp(ng + 0.08)
    else:
        nr = _clamp(nr + 0.12)
        nb = _clamp(nb + 0.08)
    # Crush blacks, boost highlights
    nr = _clamp((nr - 0.5) * 1.3 + 0.5)
    ng = _clamp((ng - 0.5) * 1.2 + 0.5)
    nb = _clamp((nb - 0.5) * 1.3 + 0.5)
    return nr, ng, nb


def _bleach_bypass(r, g, b):
    """Bleach bypass - desaturated high contrast."""
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    # Mix original with luminance
    nr = (r + lum) / 2
    ng = (g + lum) / 2
    nb = (b + lum) / 2
    # Contrast boost
    nr = _clamp((nr - 0.5) * 1.4 + 0.5)
    ng = _clamp((ng - 0.5) * 1.4 + 0.5)
    nb = _clamp((nb - 0.5) * 1.4 + 0.5)
    return nr, ng, nb


def _golden_hour(r, g, b):
    """Golden hour warm glow."""
    nr = _clamp(r * 1.15 + 0.06)
    ng = _clamp(g * 1.02 + 0.03)
    nb = _clamp(b * 0.82)
    # Soft highlights
    lum = 0.299 * nr + 0.587 * ng + 0.114 * nb
    if lum > 0.6:
        t = (lum - 0.6) / 0.4
        nr = _clamp(nr + 0.1 * t)
        ng = _clamp(ng + 0.06 * t)
    return nr, ng, nb


def _moonlight(r, g, b):
    """Moonlight / night scene - blue shadows, cool midtones."""
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    # Cool everything down
    nr = _clamp(r * 0.75)
    ng = _clamp(g * 0.82)
    nb = _clamp(b * 1.1 + 0.05)
    # Reduce exposure
    nr = _clamp(nr * 0.85)
    ng = _clamp(ng * 0.85)
    nb = _clamp(nb * 0.9)
    # Blue shadows
    if lum < 0.3:
        nb = _clamp(nb + 0.12 * (1 - lum / 0.3))
    return nr, ng, nb


def _cross_process(r, g, b):
    """Cross-processed film look - shifted colors, high saturation."""
    # Channel curve shifts
    nr = _clamp(r * r * 1.2 + r * 0.2)
    ng = _clamp(math.sqrt(g) * 0.9)
    nb = _clamp(b * 0.7 + 0.15 * b * b)
    # Boost saturation
    lum = 0.299 * nr + 0.587 * ng + 0.114 * nb
    nr = _clamp(nr + (nr - lum) * 0.3)
    ng = _clamp(ng + (ng - lum) * 0.3)
    nb = _clamp(nb + (nb - lum) * 0.3)
    return nr, ng, nb


# ---------------------------------------------------------------------------
# LUT Registry
# ---------------------------------------------------------------------------
BUILTIN_LUTS = {
    "teal_orange": {
        "label": "Teal & Orange",
        "description": "Hollywood blockbuster dual-tone look",
        "category": "cinematic",
        "fn": _teal_orange,
    },
    "vintage_warm": {
        "label": "Vintage Warm",
        "description": "Warm retro film with lifted blacks",
        "category": "film",
        "fn": _vintage_warm,
    },
    "cool_desaturate": {
        "label": "Cool Desaturated",
        "description": "Moody, muted cool tones",
        "category": "cinematic",
        "fn": _cool_desaturate,
    },
    "high_contrast_bw": {
        "label": "High Contrast B&W",
        "description": "Punchy black and white with S-curve contrast",
        "category": "bw",
        "fn": _high_contrast_bw,
    },
    "sepia": {
        "label": "Sepia",
        "description": "Classic sepia-toned monochrome",
        "category": "bw",
        "fn": _sepia,
    },
    "cyberpunk": {
        "label": "Cyberpunk",
        "description": "Neon-saturated magenta and cyan split",
        "category": "creative",
        "fn": _cyberpunk,
    },
    "bleach_bypass": {
        "label": "Bleach Bypass",
        "description": "Desaturated high contrast, gritty look",
        "category": "film",
        "fn": _bleach_bypass,
    },
    "golden_hour": {
        "label": "Golden Hour",
        "description": "Warm sunset glow, golden highlights",
        "category": "cinematic",
        "fn": _golden_hour,
    },
    "moonlight": {
        "label": "Moonlight",
        "description": "Cool blue night scene, dark mood",
        "category": "cinematic",
        "fn": _moonlight,
    },
    "cross_process": {
        "label": "Cross Process",
        "description": "Film cross-processing, shifted saturated colors",
        "category": "creative",
        "fn": _cross_process,
    },
}


def get_lut_list() -> List[Dict]:
    """Return available LUTs (built-in + user .cube files)."""
    result = []

    # Built-in
    for name, info in BUILTIN_LUTS.items():
        cube_path = os.path.join(LUTS_DIR, f"{name}.cube")
        result.append({
            "name": name,
            "label": info["label"],
            "description": info["description"],
            "category": info["category"],
            "type": "builtin",
            "generated": os.path.exists(cube_path),
        })

    # User .cube files
    user_dir = os.path.join(LUTS_DIR, "user")
    if os.path.isdir(user_dir):
        for f in sorted(os.listdir(user_dir)):
            if f.lower().endswith(".cube"):
                name = os.path.splitext(f)[0]
                result.append({
                    "name": f"user/{name}",
                    "label": name.replace("_", " ").title(),
                    "description": "User-provided LUT",
                    "category": "user",
                    "type": "user",
                    "generated": True,
                })

    return result


def ensure_lut(name: str) -> str:
    """Ensure a built-in LUT's .cube file exists, generating if needed."""
    if name not in BUILTIN_LUTS:
        # Check user LUTs
        if name.startswith("user/"):
            lut_basename = name.split("/", 1)[1]
            # Prevent path traversal: reject names with directory separators or ..
            if ".." in lut_basename or "/" in lut_basename or "\\" in lut_basename:
                raise ValueError(f"Invalid LUT name: {name}")
            user_dir = os.path.join(LUTS_DIR, "user")
            user_path = os.path.join(user_dir, lut_basename + ".cube")
            # Verify resolved path stays within user LUT directory
            if not os.path.realpath(user_path).startswith(os.path.realpath(user_dir) + os.sep):
                raise ValueError(f"Invalid LUT path: {name}")
            if os.path.exists(user_path):
                return user_path
        raise ValueError(f"Unknown LUT: {name}")

    cube_path = os.path.join(LUTS_DIR, f"{name}.cube")
    if not os.path.exists(cube_path):
        logger.info(f"Generating LUT: {name}")
        _generate_cube_lut(name, BUILTIN_LUTS[name]["fn"])
    return cube_path


# ---------------------------------------------------------------------------
# Apply LUT to Video
# ---------------------------------------------------------------------------
def apply_lut(
    input_path: str,
    lut_name: str,
    intensity: float = 1.0,
    output_path: Optional[str] = None,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Apply a color LUT to video.

    Args:
        lut_name: Built-in LUT name or "user/filename".
        intensity: Blend with original (0.0 = no effect, 1.0 = full LUT).
    """
    cube_path = ensure_lut(lut_name)
    import math
    if not isinstance(intensity, (int, float)) or math.isnan(intensity) or math.isinf(intensity):
        intensity = 1.0
    intensity = max(0.0, min(1.0, intensity))

    if output_path is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        ext = os.path.splitext(input_path)[1] or ".mp4"
        directory = output_dir or os.path.dirname(input_path)
        safe_name = lut_name.replace("/", "_")
        output_path = os.path.join(directory, f"{base}_lut_{safe_name}{ext}")

    if on_progress:
        on_progress(10, f"Applying LUT: {lut_name}...")

    # Escape path for FFmpeg filter syntax (subprocess list invocation, no shell)
    escaped = cube_path.replace("\\", "/")
    escaped = escaped.replace("'", "\\'")
    # Note: do NOT escape colons here -- the path is wrapped in single quotes
    # in the FFmpeg filter expression, so colon escaping is unnecessary and
    # would corrupt Windows drive-letter paths (e.g. C:/... → C\:/...).

    if intensity >= 0.99:
        vf = f"lut3d='{escaped}'"
    else:
        # Blend: split -> apply LUT to one -> merge with original
        vf = (
            f"split[a][b];"
            f"[a]lut3d='{escaped}'[lut];"
            f"[lut][b]blend=all_mode=normal:all_opacity={intensity}"
        )

    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
    ]

    if intensity >= 0.99:
        cmd += ["-vf", vf]
    else:
        cmd += ["-filter_complex", vf]

    cmd += [
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-c:a", "copy",
        output_path,
    ]
    run_ffmpeg(cmd, timeout=7200)

    if on_progress:
        on_progress(100, f"LUT applied: {lut_name}")
    return output_path


# ---------------------------------------------------------------------------
# Generate all built-in LUTs
# ---------------------------------------------------------------------------
def generate_all_luts(on_progress: Optional[Callable] = None) -> int:
    """Pre-generate all built-in .cube LUT files. Returns count generated."""
    count = 0
    total = len(BUILTIN_LUTS)
    for i, (name, info) in enumerate(BUILTIN_LUTS.items()):
        cube_path = os.path.join(LUTS_DIR, f"{name}.cube")
        if not os.path.exists(cube_path):
            if on_progress:
                pct = int((i / total) * 100)
                on_progress(pct, f"Generating LUT: {info['label']}...")
            _generate_cube_lut(name, info["fn"])
            count += 1
    if on_progress:
        on_progress(100, f"Generated {count} LUTs")
    return count


# ---------------------------------------------------------------------------
# AI LUT Generation from Reference Image
# ---------------------------------------------------------------------------
def generate_lut_from_reference(
    reference_path: str,
    lut_name: str = "",
    method: str = "histogram",
    strength: float = 1.0,
    size: int = 33,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Generate a .cube LUT by analyzing a reference image's color palette.

    Computes per-channel CDFs (cumulative distribution functions) from the
    reference image and builds transfer functions that match the color
    distribution. Uses PIL + numpy (both already in the standard dep group).

    Args:
        reference_path: Path to the reference image (jpg, png, etc.).
        lut_name: Name for the generated LUT. Auto-generated from filename if empty.
        method: Analysis method - "histogram" (CDF matching) or "average" (simple color shift).
        strength: Blend strength 0.0-1.0 (1.0 = full match, 0.5 = half).
        size: LUT cube size (17, 33, or 65).
        on_progress: Progress callback(pct, msg).

    Returns:
        Path to the generated .cube LUT file.
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        raise RuntimeError(
            "PIL and numpy are required. Install with: "
            "pip install Pillow numpy"
        )

    if not os.path.isfile(reference_path):
        raise FileNotFoundError(f"Reference image not found: {reference_path}")

    strength = max(0.0, min(1.0, strength))
    size = max(17, min(65, size))

    if not lut_name:
        lut_name = os.path.splitext(os.path.basename(reference_path))[0]
        lut_name = "ref_" + lut_name.replace(" ", "_")[:30]

    if on_progress:
        on_progress(10, "Loading reference image...")

    # Load and analyze reference image
    img = Image.open(reference_path).convert("RGB")
    img_array = np.array(img, dtype=np.float64) / 255.0

    if on_progress:
        on_progress(30, f"Analyzing color distribution ({method})...")

    if method == "histogram":
        # Compute per-channel CDFs from reference
        r_cdf = _compute_cdf(img_array[:, :, 0].flatten())
        g_cdf = _compute_cdf(img_array[:, :, 1].flatten())
        b_cdf = _compute_cdf(img_array[:, :, 2].flatten())

        def transform(r, g, b):
            # Map input through reference CDF (histogram matching)
            nr = _apply_cdf_transfer(r, r_cdf, strength)
            ng = _apply_cdf_transfer(g, g_cdf, strength)
            nb = _apply_cdf_transfer(b, b_cdf, strength)
            return nr, ng, nb

    elif method == "average":
        # Simple color shift toward reference average
        avg_r = float(np.mean(img_array[:, :, 0]))
        avg_g = float(np.mean(img_array[:, :, 1]))
        avg_b = float(np.mean(img_array[:, :, 2]))

        # Also compute standard deviations for contrast matching
        std_r = float(np.std(img_array[:, :, 0])) or 0.2
        std_g = float(np.std(img_array[:, :, 1])) or 0.2
        std_b = float(np.std(img_array[:, :, 2])) or 0.2

        def transform(r, g, b):
            # Shift toward reference average with strength blending
            nr = r + (avg_r - 0.5) * strength * 0.5
            ng = g + (avg_g - 0.5) * strength * 0.5
            nb = b + (avg_b - 0.5) * strength * 0.5
            # Subtle contrast match
            nr = (nr - 0.5) * (1.0 + (std_r / 0.2 - 1.0) * strength * 0.3) + 0.5
            ng = (ng - 0.5) * (1.0 + (std_g / 0.2 - 1.0) * strength * 0.3) + 0.5
            nb = (nb - 0.5) * (1.0 + (std_b / 0.2 - 1.0) * strength * 0.3) + 0.5
            return _clamp(nr), _clamp(ng), _clamp(nb)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'histogram' or 'average'.")

    if on_progress:
        on_progress(50, "Generating .cube LUT file...")

    # Validate lut_name contains no path traversal
    if ".." in lut_name or "/" in lut_name or "\\" in lut_name:
        raise ValueError(f"Invalid LUT name: {lut_name}")

    # Generate to user LUTs directory
    user_dir = os.path.join(LUTS_DIR, "user")
    os.makedirs(user_dir, exist_ok=True)
    cube_path = os.path.join(user_dir, f"{lut_name}.cube")
    if not os.path.realpath(cube_path).startswith(os.path.realpath(user_dir) + os.sep):
        raise ValueError(f"Invalid LUT path: {lut_name}")

    with open(cube_path, "w", encoding="utf-8") as f:
        f.write(f"TITLE \"{lut_name}\"\n")
        f.write(f"# Generated from: {os.path.basename(reference_path)}\n")
        f.write(f"# Method: {method}, Strength: {strength}\n")
        f.write(f"LUT_SIZE {size}\n\n")

        written = 0

        for b_i in range(size):
            for g_i in range(size):
                for r_i in range(size):
                    r = r_i / (size - 1)
                    g = g_i / (size - 1)
                    b = b_i / (size - 1)

                    nr, ng, nb = transform(r, g, b)
                    nr = _clamp(nr)
                    ng = _clamp(ng)
                    nb = _clamp(nb)

                    f.write(f"{nr:.6f} {ng:.6f} {nb:.6f}\n")
                    written += 1

            if on_progress and b_i % 5 == 0:
                pct = 50 + int((b_i / size) * 45)
                on_progress(pct, f"Writing LUT ({b_i + 1}/{size})...")

    if on_progress:
        on_progress(100, f"LUT saved: {lut_name}")

    logger.info("Generated reference LUT: %s -> %s", reference_path, cube_path)
    return cube_path


# ---------------------------------------------------------------------------
# AI Color Grading (Neural LUT from reference — LAB perceptual matching)
# ---------------------------------------------------------------------------
def generate_lut_ai(
    reference_path: str,
    lut_name: str = "",
    size: int = 33,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Generate a .cube LUT using perceptual AI color matching in LAB space.

    Superior to histogram matching: operates in perceptual LAB color space
    with per-channel percentile mapping for natural-looking color grades
    that preserve skin tones and avoid color banding.

    Inspired by Image-Adaptive-3DLUT (CVPR 2022) but uses a lightweight
    statistical approach that needs no GPU or trained models.

    Args:
        reference_path: Path to the reference/look image.
        lut_name: Name for the generated LUT.
        size: LUT cube size (17, 33, or 65).
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        raise RuntimeError("PIL and numpy are required")

    if not os.path.isfile(reference_path):
        raise FileNotFoundError(f"Reference image not found: {reference_path}")

    size = max(17, min(65, size))

    if not lut_name:
        lut_name = "ai_" + os.path.splitext(os.path.basename(reference_path))[0].replace(" ", "_")[:25]

    if on_progress:
        on_progress(10, "Analyzing reference image in LAB space...")

    # Load reference and convert to LAB
    img = Image.open(reference_path).convert("RGB")
    img_np = np.array(img, dtype=np.float32) / 255.0

    # Convert RGB to LAB using simple approximation
    # (avoids cv2 dependency — uses linearized sRGB → XYZ → LAB)
    def _srgb_to_linear(c):
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    def _linear_to_srgb(c):
        return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(np.clip(c, 1e-10, None), 1.0 / 2.4) - 0.055)

    linear = _srgb_to_linear(img_np)
    # sRGB → XYZ (D65)
    x = linear[:, :, 0] * 0.4124564 + linear[:, :, 1] * 0.3575761 + linear[:, :, 2] * 0.1804375
    y = linear[:, :, 0] * 0.2126729 + linear[:, :, 1] * 0.7151522 + linear[:, :, 2] * 0.0721750
    z = linear[:, :, 0] * 0.0193339 + linear[:, :, 1] * 0.1191920 + linear[:, :, 2] * 0.9503041

    # XYZ → LAB
    def _lab_f(t):
        delta = 6.0 / 29.0
        return np.where(t > delta ** 3, np.cbrt(t), t / (3 * delta ** 2) + 4.0 / 29.0)

    fx = _lab_f(x / 0.95047)
    fy = _lab_f(y / 1.00000)
    fz = _lab_f(z / 1.08883)

    ref_L = 116 * fy - 16
    ref_a = 500 * (fx - fy)
    ref_b = 200 * (fy - fz)

    # Compute percentile-based transfer curves in LAB
    n_percentiles = 256
    ref_L_pct = np.percentile(ref_L.flatten(), np.linspace(0, 100, n_percentiles))
    ref_a_pct = np.percentile(ref_a.flatten(), np.linspace(0, 100, n_percentiles))
    ref_b_pct = np.percentile(ref_b.flatten(), np.linspace(0, 100, n_percentiles))

    # Standard sRGB image percentiles (approximate)
    std_L = np.linspace(0, 100, n_percentiles)
    std_a = np.linspace(-128, 127, n_percentiles)
    std_b = np.linspace(-128, 127, n_percentiles)

    if on_progress:
        on_progress(40, "Building neural-inspired LUT...")

    # Build transfer: for each input RGB, convert to LAB, percentile-map, convert back
    if ".." in lut_name or "/" in lut_name or "\\" in lut_name:
        raise ValueError(f"Invalid LUT name: {lut_name}")

    user_dir = os.path.join(LUTS_DIR, "user")
    os.makedirs(user_dir, exist_ok=True)
    cube_path = os.path.join(user_dir, f"{lut_name}.cube")
    if not os.path.realpath(cube_path).startswith(os.path.realpath(user_dir) + os.sep):
        raise ValueError(f"Invalid LUT path: {lut_name}")

    with open(cube_path, "w", encoding="utf-8") as f:
        f.write(f'TITLE "{lut_name}"\n')
        f.write(f"# AI color grade from: {os.path.basename(reference_path)}\n")
        f.write("# Method: LAB perceptual matching\n")
        f.write(f"LUT_SIZE {size}\n\n")

        for b_i in range(size):
            for g_i in range(size):
                for r_i in range(size):
                    r = r_i / (size - 1)
                    g = g_i / (size - 1)
                    b = b_i / (size - 1)

                    # sRGB → linear → XYZ → LAB
                    rl = _srgb_to_linear(np.array([r]))[0]
                    gl = _srgb_to_linear(np.array([g]))[0]
                    bl = _srgb_to_linear(np.array([b]))[0]

                    xi = rl * 0.4124564 + gl * 0.3575761 + bl * 0.1804375
                    yi = rl * 0.2126729 + gl * 0.7151522 + bl * 0.0721750
                    zi = rl * 0.0193339 + gl * 0.1191920 + bl * 0.9503041

                    fxi = _lab_f(np.array([xi / 0.95047]))[0]
                    fyi = _lab_f(np.array([yi / 1.00000]))[0]
                    fzi = _lab_f(np.array([zi / 1.08883]))[0]

                    L_in = 116 * fyi - 16
                    a_in = 500 * (fxi - fyi)
                    b_in = 200 * (fyi - fzi)

                    # Percentile mapping
                    L_pct = np.interp(L_in, std_L, ref_L_pct)
                    a_pct = np.interp(a_in, std_a, ref_a_pct)
                    b_pct = np.interp(b_in, std_b, ref_b_pct)

                    # LAB → XYZ → linear → sRGB
                    fy_o = (L_pct + 16) / 116
                    fx_o = a_pct / 500 + fy_o
                    fz_o = fy_o - b_pct / 200

                    delta = 6.0 / 29.0
                    xo = 0.95047 * (fx_o ** 3 if fx_o > delta else 3 * delta ** 2 * (fx_o - 4.0 / 29.0))
                    yo = 1.00000 * (fy_o ** 3 if fy_o > delta else 3 * delta ** 2 * (fy_o - 4.0 / 29.0))
                    zo = 1.08883 * (fz_o ** 3 if fz_o > delta else 3 * delta ** 2 * (fz_o - 4.0 / 29.0))

                    # XYZ → linear sRGB
                    ro = xo * 3.2404542 + yo * -1.5371385 + zo * -0.4985314
                    go = xo * -0.9692660 + yo * 1.8760108 + zo * 0.0415560
                    bo = xo * 0.0556434 + yo * -0.2040259 + zo * 1.0572252

                    # linear → sRGB
                    ro = float(_linear_to_srgb(np.array([max(0, ro)]))[0])
                    go = float(_linear_to_srgb(np.array([max(0, go)]))[0])
                    bo = float(_linear_to_srgb(np.array([max(0, bo)]))[0])

                    f.write(f"{_clamp(ro):.6f} {_clamp(go):.6f} {_clamp(bo):.6f}\n")

            if on_progress and b_i % 5 == 0:
                pct = 40 + int((b_i / size) * 55)
                on_progress(pct, f"Writing AI LUT ({b_i + 1}/{size})...")

    if on_progress:
        on_progress(100, f"AI color grade LUT saved: {lut_name}")

    logger.info("Generated AI LUT: %s -> %s", reference_path, cube_path)
    return cube_path


def _compute_cdf(channel_data):
    """Compute cumulative distribution function for a color channel (0-1 float array)."""
    import numpy as np
    bins = 256
    hist, bin_edges = np.histogram(channel_data, bins=bins, range=(0.0, 1.0))
    cdf = np.cumsum(hist).astype(np.float64)
    cdf /= cdf[-1] if cdf[-1] > 0 else 1.0
    return cdf


def _apply_cdf_transfer(value, ref_cdf, strength):
    """Apply CDF transfer function to map input value through reference distribution."""
    # Clamp input and map to bin index
    value = max(0.0, min(1.0, value))
    bin_idx = int(value * 255)
    bin_idx = max(0, min(255, bin_idx))

    # Look up reference CDF value
    mapped = float(ref_cdf[bin_idx])

    # Blend with identity based on strength
    result = value * (1.0 - strength) + mapped * strength
    return _clamp(result)


# ---------------------------------------------------------------------------
# LUT Blending (mix any two LUTs with a slider)
# ---------------------------------------------------------------------------
def blend_luts(
    lut_a_name: str,
    lut_b_name: str,
    blend: float = 0.5,
    output_name: str = "",
    size: int = 33,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Blend two .cube LUTs into a new LUT with a single slider.

    Inspired by NILUT (Neural Implicit LUT) continuous style blending.
    Loads both LUTs, linearly interpolates between their color transforms,
    and outputs a new .cube file. Enables smooth transitions between any
    two color grades.

    Args:
        lut_a_name: First LUT name (built-in or "user/filename").
        lut_b_name: Second LUT name.
        blend: Mix ratio (0.0 = fully A, 1.0 = fully B, 0.5 = even mix).
        output_name: Name for blended LUT. Auto-generated if empty.
        size: Output cube size.
    """
    cube_a = ensure_lut(lut_a_name)
    cube_b = ensure_lut(lut_b_name)

    blend = max(0.0, min(1.0, blend))
    size = max(17, min(65, size))

    if not output_name:
        output_name = f"blend_{lut_a_name}_{lut_b_name}_{int(blend * 100)}"
        output_name = output_name.replace("/", "_").replace("\\", "_")[:40]

    if ".." in output_name or "/" in output_name or "\\" in output_name:
        raise ValueError(f"Invalid LUT name: {output_name}")

    if on_progress:
        on_progress(10, "Loading LUTs...")

    # Parse both .cube files
    def _parse_cube(path):
        values = []
        lut_size = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("LUT_SIZE"):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            lut_size = int(parts[-1])
                        except ValueError:
                            pass
                elif line and not line.startswith("#") and not line.startswith("TITLE") and not line.startswith("DOMAIN"):
                    parts = line.split()
                    if len(parts) == 3:
                        try:
                            values.append((float(parts[0]), float(parts[1]), float(parts[2])))
                        except ValueError:
                            pass
        return values, lut_size

    vals_a, size_a = _parse_cube(cube_a)
    vals_b, size_b = _parse_cube(cube_b)

    if on_progress:
        on_progress(30, "Blending LUTs...")

    # If sizes differ, we need to resample — for now require same size or use output size
    # Simple approach: generate output by querying both LUTs at each point
    user_dir = os.path.join(LUTS_DIR, "user")
    os.makedirs(user_dir, exist_ok=True)
    cube_path = os.path.join(user_dir, f"{output_name}.cube")
    if not os.path.realpath(cube_path).startswith(os.path.realpath(user_dir) + os.sep):
        raise ValueError(f"Invalid LUT path: {output_name}")

    # If LUTs are same size and match output, do direct blend
    total_entries = size ** 3
    with open(cube_path, "w", encoding="utf-8") as f:
        f.write(f'TITLE "{output_name}"\n')
        f.write(f"# Blend of {lut_a_name} ({1-blend:.0%}) + {lut_b_name} ({blend:.0%})\n")
        f.write(f"LUT_SIZE {size}\n\n")

        if len(vals_a) == total_entries and len(vals_b) == total_entries:
            # Direct element-wise blend
            for i in range(total_entries):
                ra = vals_a[i][0] * (1 - blend) + vals_b[i][0] * blend
                ga = vals_a[i][1] * (1 - blend) + vals_b[i][1] * blend
                ba = vals_a[i][2] * (1 - blend) + vals_b[i][2] * blend
                f.write(f"{_clamp(ra):.6f} {_clamp(ga):.6f} {_clamp(ba):.6f}\n")
        else:
            raise ValueError(
                f"LUT size mismatch: {lut_a_name} has {len(vals_a)} entries, "
                f"{lut_b_name} has {len(vals_b)} entries (need {total_entries} for size {size}). "
                f"Both LUTs must have the same cube size for blending."
            )
            # Unreachable fallback
            for b_i in range(size):
                for g_i in range(size):
                    for r_i in range(size):
                        r = r_i / (size - 1)
                        g = g_i / (size - 1)
                        b = b_i / (size - 1)
                        f.write(f"{r:.6f} {g:.6f} {b:.6f}\n")

    if on_progress:
        on_progress(100, f"Blended LUT saved: {output_name}")

    logger.info("Blended LUTs: %s + %s -> %s (blend=%.2f)", lut_a_name, lut_b_name, cube_path, blend)
    return cube_path
