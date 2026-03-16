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
import subprocess
import tempfile
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("opencut")

LUTS_DIR = os.path.join(os.path.expanduser("~"), ".opencut", "luts")


def _run_ffmpeg(cmd: List[str], timeout: int = 7200) -> str:
    result = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {result.stderr.decode(errors='replace')[-500:]}")
    return result.stderr.decode(errors="replace")


# ---------------------------------------------------------------------------
# LUT Generation Helpers
# ---------------------------------------------------------------------------
def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _generate_cube_lut(name: str, transform_fn, size: int = 33) -> str:
    """Generate a .cube LUT file using a color transform function."""
    os.makedirs(LUTS_DIR, exist_ok=True)
    path = os.path.join(LUTS_DIR, f"{name}.cube")

    with open(path, "w") as f:
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
            user_path = os.path.join(LUTS_DIR, "user", name.split("/", 1)[1] + ".cube")
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
    intensity = max(0.0, min(1.0, intensity))

    if output_path is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        ext = os.path.splitext(input_path)[1] or ".mp4"
        directory = output_dir or os.path.dirname(input_path)
        safe_name = lut_name.replace("/", "_")
        output_path = os.path.join(directory, f"{base}_lut_{safe_name}{ext}")

    if on_progress:
        on_progress(10, f"Applying LUT: {lut_name}...")

    # Escape path for FFmpeg
    escaped = cube_path.replace("\\", "/").replace(":", "\\:").replace("'", "'\\''")

    if intensity >= 0.99:
        vf = f"lut3d='{escaped}'"
    else:
        # Blend: split -> apply LUT to one -> merge with original
        orig_weight = 1.0 - intensity
        vf = (
            f"split[a][b];"
            f"[a]lut3d='{escaped}'[lut];"
            f"[lut][b]blend=all_mode=normal:all_opacity={intensity}"
        )

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
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
    _run_ffmpeg(cmd)

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
        from PIL import Image
        import numpy as np
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

    # Generate to user LUTs directory
    user_dir = os.path.join(LUTS_DIR, "user")
    os.makedirs(user_dir, exist_ok=True)
    cube_path = os.path.join(user_dir, f"{lut_name}.cube")

    with open(cube_path, "w") as f:
        f.write(f"TITLE \"{lut_name}\"\n")
        f.write(f"# Generated from: {os.path.basename(reference_path)}\n")
        f.write(f"# Method: {method}, Strength: {strength}\n")
        f.write(f"LUT_SIZE {size}\n\n")

        total_entries = size * size * size
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
    # Map input value (0-1) to bin index
    bin_idx = int(value * 255)
    bin_idx = max(0, min(255, bin_idx))

    # Look up reference CDF value
    mapped = float(ref_cdf[bin_idx])

    # Blend with identity based on strength
    result = value * (1.0 - strength) + mapped * strength
    return _clamp(result)
