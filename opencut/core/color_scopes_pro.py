"""
Professional color scopes via ``colour-science``.

Extends ``core/color_scopes.py`` (which renders FFmpeg-native
waveform / vectorscope / histogram filters to PNG) with
mathematically-proper scopes computed from sampled frame pixels:

- **CIE xy chromaticity** — plots the distribution of chromaticities
  on a CIE 1931 xy diagram. More perceptually meaningful than RGB
  histograms because it decouples chromaticity from luminance.
- **CIELUV vectorscope** — equal-distance chromatic error in CIELUV
  space (u*, v*), the space colourists reach for when they care about
  perceptual colour differences.
- **Gamut coverage** — percentage of frame pixels falling inside the
  Rec.709 / DCI-P3 / Rec.2020 gamut triangles, useful for delivering
  HDR masters that claim a wider gamut than the source actually uses.

Backed by the ``colour-science`` pip package
(https://github.com/colour-science/colour, BSD-3) which owns the
reference CIE colourimetry code; our module is a thin wrapper that
samples frames via FFmpeg, passes them through ``colour`` transforms,
and emits structured JSON + PNG plots.

Graceful degradation: when ``colour-science`` is missing, the only
functions available are ``check_colour_science_available()`` and the
existing FFmpeg-based scopes in ``core/color_scopes.py``.
"""

from __future__ import annotations

import logging
import os
import subprocess as _sp
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from opencut.helpers import get_ffmpeg_path

logger = logging.getLogger("opencut")


GAMUTS = ("rec709", "dci-p3", "rec2020")


@dataclass
class ColourScopeResult:
    """Structured scope report."""
    frame_samples: int = 0
    sample_timestamps: List[float] = field(default_factory=list)
    mean_xy: Tuple[float, float] = (0.0, 0.0)
    mean_luv: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    gamut_coverage: Dict[str, float] = field(default_factory=dict)
    # Diagnostic plots written to disk (PNG). May be empty when
    # matplotlib isn't available — stats are still returned.
    chromaticity_plot: str = ""
    vectorscope_plot: str = ""
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, key):
        return getattr(self, key)

    def __contains__(self, key):
        return key in self.__dataclass_fields__

    def keys(self):
        return self.__dataclass_fields__.keys()


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def check_colour_science_available() -> bool:
    try:
        import colour  # noqa: F401
        return True
    except ImportError:
        return False


def check_matplotlib_available() -> bool:
    try:
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------

def _sample_frames(
    input_path: str,
    sample_count: int,
    target_size: int = 256,
) -> Tuple[List[object], List[float]]:
    """Extract ``sample_count`` evenly-spaced frames, ``target_size²``.

    Returns a list of RGB-float ndarrays in [0, 1] and the matching
    timestamps in seconds.  Frames are downsampled aggressively
    (default 256×256) because scopes operate on chromaticity
    distributions — pixel count matters more than resolution.
    """
    import numpy as _np
    from PIL import Image

    # Probe duration
    from opencut.helpers import get_video_info
    info = get_video_info(input_path)
    duration = float(info.get("duration") or 0.0)
    if duration <= 0:
        raise RuntimeError("Could not probe input duration for frame sampling")

    sample_count = max(1, min(sample_count, 120))
    timestamps = [
        (i + 0.5) * duration / sample_count
        for i in range(sample_count)
    ]

    frames: List[_np.ndarray] = []
    tmpdir = tempfile.mkdtemp(prefix="opencut_scopes_")
    try:
        for i, t in enumerate(timestamps):
            fp = os.path.join(tmpdir, f"f_{i:03d}.png")
            cmd = [
                get_ffmpeg_path(), "-hide_banner", "-loglevel", "error",
                "-ss", f"{t:.3f}", "-i", input_path,
                "-vframes", "1",
                "-vf", f"scale={target_size}:{target_size}:flags=area",
                "-q:v", "2",
                fp,
            ]
            proc = _sp.run(
                cmd, capture_output=True, timeout=60, check=False,
            )
            if proc.returncode != 0 or not os.path.isfile(fp):
                continue
            img = Image.open(fp).convert("RGB")
            arr = _np.asarray(img).astype("float32") / 255.0
            frames.append(arr.reshape(-1, 3))  # flatten to (N, 3)
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    return frames, timestamps


# ---------------------------------------------------------------------------
# Scope computation
# ---------------------------------------------------------------------------

def _compute_xy(pixels, colour_mod) -> Tuple[float, float]:
    """Compute the mean CIE xy chromaticity of a (N, 3) RGB array."""
    import numpy as _np
    # sRGB → XYZ → xy
    xyz = colour_mod.sRGB_to_XYZ(pixels)
    xyz = _np.asarray(xyz)
    s = xyz.sum(axis=-1, keepdims=True)
    s[s == 0] = 1e-9
    xyy = xyz / s
    x = float(_np.clip(xyy[..., 0].mean(), 0.0, 1.0))
    y = float(_np.clip(xyy[..., 1].mean(), 0.0, 1.0))
    return x, y


def _compute_luv(pixels, colour_mod) -> Tuple[float, float, float]:
    """Compute mean CIELUV (L*, u*, v*) of (N, 3) RGB pixels."""
    import numpy as _np
    xyz = colour_mod.sRGB_to_XYZ(pixels)
    # D65 whitepoint (sRGB illuminant)
    luv = colour_mod.XYZ_to_Luv(xyz)
    luv = _np.asarray(luv)
    return (
        float(_np.clip(luv[..., 0].mean(), 0.0, 100.0)),
        float(luv[..., 1].mean()),
        float(luv[..., 2].mean()),
    )


# Rec.709 / DCI-P3 / Rec.2020 gamut triangles in CIE xy coordinates.
_GAMUT_VERTICES = {
    "rec709": [(0.640, 0.330), (0.300, 0.600), (0.150, 0.060)],
    "dci-p3": [(0.680, 0.320), (0.265, 0.690), (0.150, 0.060)],
    "rec2020": [(0.708, 0.292), (0.170, 0.797), (0.131, 0.046)],
}


def _point_in_triangle(px: float, py: float, v) -> bool:
    """Barycentric-coordinates inside-triangle test."""
    (ax, ay), (bx, by), (cx, cy) = v
    denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    if abs(denom) < 1e-12:
        return False
    a = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / denom
    b = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / denom
    c = 1.0 - a - b
    return (a >= 0) and (b >= 0) and (c >= 0)


def _compute_gamut_coverage(
    pixels, colour_mod,
) -> Dict[str, float]:
    """Return the fraction (0..1) of pixels inside each known gamut."""
    import numpy as _np
    xyz = colour_mod.sRGB_to_XYZ(pixels)
    s = _np.asarray(xyz).sum(axis=-1, keepdims=True)
    s[s == 0] = 1e-9
    xy = _np.asarray(xyz) / s
    x = xy[..., 0]
    y = xy[..., 1]
    out: Dict[str, float] = {}
    n = len(x)
    if n == 0:
        return {g: 0.0 for g in GAMUTS}
    for name in GAMUTS:
        verts = _GAMUT_VERTICES[name]
        # Vectorised barycentric
        ax, ay = verts[0]
        bx, by = verts[1]
        cx, cy = verts[2]
        denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(denom) < 1e-12:
            out[name] = 0.0
            continue
        a = ((by - cy) * (x - cx) + (cx - bx) * (y - cy)) / denom
        b = ((cy - ay) * (x - cx) + (ax - cx) * (y - cy)) / denom
        c = 1.0 - a - b
        inside = (a >= 0) & (b >= 0) & (c >= 0)
        out[name] = round(float(inside.sum() / n), 4)
    return out


# ---------------------------------------------------------------------------
# Plot rendering (optional, matplotlib)
# ---------------------------------------------------------------------------

def _render_chromaticity_plot(
    x: float, y: float, out_path: str,
) -> Optional[str]:
    """Draw a mean-chromaticity marker on a CIE 1931 xy horseshoe.

    Returns the output path on success; ``None`` when matplotlib
    isn't available (stats-only mode).
    """
    if not check_matplotlib_available():
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 5), dpi=140)
        ax.set_xlim(0, 0.8)
        ax.set_ylim(0, 0.9)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("CIE 1931 xy mean chromaticity")
        # Gamut triangles for reference
        for name, verts in _GAMUT_VERTICES.items():
            xs = [v[0] for v in verts] + [verts[0][0]]
            ys = [v[1] for v in verts] + [verts[0][1]]
            ax.plot(xs, ys, "-", linewidth=1.0, label=name)
        ax.plot([x], [y], "o", markersize=10, label="mean")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        return out_path
    except Exception as exc:  # noqa: BLE001
        logger.warning("Chromaticity plot render failed: %s", exc)
        return None


def _render_vectorscope_plot(
    u: float, v: float, out_path: str,
) -> Optional[str]:
    """Draw a mean CIELUV (u*, v*) marker inside a reference polar grid."""
    if not check_matplotlib_available():
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 5), dpi=140)
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        ax.set_aspect("equal")
        ax.set_title("CIELUV vectorscope (u*, v* mean)")
        ax.axhline(0, color="#555", linewidth=0.8)
        ax.axvline(0, color="#555", linewidth=0.8)
        # Polar reference circles
        for r in (25, 50, 75):
            circle = plt.Circle((0, 0), r, fill=False,
                                color="#888", linestyle=":", linewidth=0.6)
            ax.add_patch(circle)
        ax.plot([u], [v], "o", markersize=10)
        ax.grid(True, linestyle=":", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        return out_path
    except Exception as exc:  # noqa: BLE001
        logger.warning("Vectorscope plot render failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyse_scopes(
    input_path: str,
    sample_count: int = 24,
    output_dir: Optional[str] = None,
    render_plots: bool = True,
    on_progress: Optional[Callable] = None,
) -> ColourScopeResult:
    """Compute mathematically-proper colour scopes on sampled frames.

    Args:
        input_path: Any FFmpeg-decodable video.
        sample_count: Number of evenly-spaced frames to average
            (1..120).  24 is a good trade-off between runtime and
            accuracy on typical short-form content; push to 60+ for
            archival masters.
        output_dir: Directory for the optional PNG plots.  Defaults to
            the system temp dir.
        render_plots: When ``False``, stats are computed but no PNGs
            are written.  Useful for CI / headless checks.
        on_progress: ``(pct, msg)`` callback.

    Returns:
        :class:`ColourScopeResult`.

    Raises:
        RuntimeError: ``colour-science`` not installed.
        FileNotFoundError: ``input_path`` missing.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")
    if not check_colour_science_available():
        raise RuntimeError(
            "colour-science not installed. Install: pip install colour-science"
        )

    import colour  # noqa: F401 — used via module alias below
    import numpy as np

    colour_mod = colour

    if on_progress:
        on_progress(5, f"Sampling {sample_count} frame(s)…")

    frames, timestamps = _sample_frames(input_path, sample_count)
    if not frames:
        raise RuntimeError("No frames could be sampled from input")

    if on_progress:
        on_progress(30, "Computing xy chromaticity + CIELUV…")

    # Concatenate per-frame pixels into a single (N_total, 3) pass
    combined = np.concatenate(frames, axis=0)

    mean_xy = _compute_xy(combined, colour_mod)
    mean_luv = _compute_luv(combined, colour_mod)

    if on_progress:
        on_progress(60, "Computing gamut coverage…")
    gamut = _compute_gamut_coverage(combined, colour_mod)

    # Plots
    chroma_png = ""
    vector_png = ""
    if render_plots:
        out_dir = output_dir or tempfile.gettempdir()
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(input_path))[0]
        chroma_png = _render_chromaticity_plot(
            mean_xy[0], mean_xy[1],
            os.path.join(out_dir, f"{base}_cie_xy.png"),
        ) or ""
        vector_png = _render_vectorscope_plot(
            mean_luv[1], mean_luv[2],
            os.path.join(out_dir, f"{base}_luv_vectorscope.png"),
        ) or ""

    if on_progress:
        on_progress(100, "Colour scope analysis complete")

    return ColourScopeResult(
        frame_samples=len(frames),
        sample_timestamps=[round(t, 3) for t in timestamps[:len(frames)]],
        mean_xy=mean_xy,
        mean_luv=mean_luv,
        gamut_coverage=gamut,
        chromaticity_plot=chroma_png,
        vectorscope_plot=vector_png,
        notes=[f"total_pixels={combined.shape[0]}"],
    )
