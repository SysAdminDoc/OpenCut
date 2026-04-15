"""
OpenCut Real-Time Video Scopes — Scope Data Generation

Extract a video frame and compute scope data for frontend canvas
rendering: waveform, vectorscope, histogram, parade, and false colour.

Each scope returns numerical data as nested lists suitable for
JavaScript canvas rendering.  Scope presets bundle multiple scopes
for common workflows (colorist, exposure, broadcast).
"""

import logging
import os
import subprocess as _sp
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from opencut.helpers import get_ffmpeg_path

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCOPE_TYPES = ("waveform", "vectorscope", "histogram", "parade", "false_color")

SCOPE_PRESETS = {
    "colorist": {
        "name": "Colorist",
        "description": "Waveform + Vectorscope + Parade for colour grading",
        "scopes": ["waveform", "vectorscope", "parade"],
    },
    "exposure": {
        "name": "Exposure",
        "description": "Histogram + False Color for exposure analysis",
        "scopes": ["histogram", "false_color"],
    },
    "broadcast": {
        "name": "Broadcast",
        "description": "Waveform + Vectorscope with broadcast legal range markers",
        "scopes": ["waveform", "vectorscope"],
        "options": {"legal_range": True},
    },
    "full": {
        "name": "Full Analysis",
        "description": "All scopes combined",
        "scopes": ["waveform", "vectorscope", "histogram", "parade", "false_color"],
    },
}

# Broadcast legal range (IRE / 8-bit)
LEGAL_BLACK = 16
LEGAL_WHITE = 235
LEGAL_CHROMA_MIN = 16
LEGAL_CHROMA_MAX = 240

# Histogram / waveform resolution
HISTOGRAM_BINS = 256
WAVEFORM_WIDTH = 256


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ScopeResult:
    """Result from a single scope computation."""
    scope_type: str = ""
    data: dict = field(default_factory=dict)
    legal_range_violations: int = 0
    frame_timestamp: float = 0.0
    processing_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MultiScopeResult:
    """Result from computing multiple scopes at once."""
    scopes: Dict[str, ScopeResult] = field(default_factory=dict)
    preset: str = ""
    frame_timestamp: float = 0.0
    total_time_ms: float = 0.0

    def to_dict(self) -> dict:
        d = {
            "scopes": {k: v.to_dict() for k, v in self.scopes.items()},
            "preset": self.preset,
            "frame_timestamp": self.frame_timestamp,
            "total_time_ms": self.total_time_ms,
        }
        return d


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------
def _extract_frame(video_path: str, timestamp: float,
                   output_path: str) -> str:
    """Extract a single frame as JPEG for scope analysis."""
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-ss", str(max(0.0, timestamp)),
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
        "-y", output_path,
    ]
    result = _sp.run(cmd, capture_output=True, timeout=30)
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")[-300:]
        raise RuntimeError(f"Frame extraction for scopes failed: {stderr}")
    return output_path


def _load_frame_pixels(frame_path: str):
    """Load frame as PIL Image and return (image, pixels, width, height)."""
    try:
        from PIL import Image  # noqa: F811
    except ImportError:
        raise ImportError("PIL (Pillow) is required for scope generation")

    img = Image.open(frame_path).convert("RGB")
    w, h = img.size
    pixels = list(img.getdata())
    return img, pixels, w, h


# ---------------------------------------------------------------------------
# Scope generators
# ---------------------------------------------------------------------------
def _compute_waveform(pixels: list, width: int, height: int,
                      check_legal: bool = False) -> Tuple[dict, int]:
    """Compute waveform (luminance distribution per column).

    Returns (data_dict, legal_violations).
    data_dict keys: r, g, b, luma -- each a list of HISTOGRAM_BINS lists
    representing per-column intensity distribution.
    """
    # Simplified: compute per-row luminance histogram
    # For each column position (sampled), collect luminance values
    sample_cols = min(width, WAVEFORM_WIDTH)
    step = max(1, width // sample_cols)

    luma_columns = [[] for _ in range(sample_cols)]
    r_columns = [[] for _ in range(sample_cols)]
    g_columns = [[] for _ in range(sample_cols)]
    b_columns = [[] for _ in range(sample_cols)]
    violations = 0

    for row in range(height):
        for col_idx in range(sample_cols):
            x = col_idx * step
            pix_idx = row * width + x
            if pix_idx >= len(pixels):
                break
            r, g, b = pixels[pix_idx]
            luma = int(0.299 * r + 0.587 * g + 0.114 * b)
            luma_columns[col_idx].append(luma)
            r_columns[col_idx].append(r)
            g_columns[col_idx].append(g)
            b_columns[col_idx].append(b)

            if check_legal:
                if luma < LEGAL_BLACK or luma > LEGAL_WHITE:
                    violations += 1

    # Reduce to distribution: for each column, histogram of values
    def _col_hist(columns):
        result = []
        for col in columns:
            hist = [0] * HISTOGRAM_BINS
            for v in col:
                hist[min(255, max(0, v))] += 1
            result.append(hist)
        return result

    data = {
        "luma": _col_hist(luma_columns),
        "r": _col_hist(r_columns),
        "g": _col_hist(g_columns),
        "b": _col_hist(b_columns),
        "columns": sample_cols,
        "rows": HISTOGRAM_BINS,
    }
    return data, violations


def _compute_vectorscope(pixels: list, width: int, height: int,
                         check_legal: bool = False) -> Tuple[dict, int]:
    """Compute vectorscope (UV chrominance plot).

    Returns a 2D grid of intensity values representing the UV plane.
    """
    scope_size = 256
    grid = [[0] * scope_size for _ in range(scope_size)]
    violations = 0
    sample_step = max(1, len(pixels) // 50000)  # sub-sample large frames

    for i in range(0, len(pixels), sample_step):
        r, g, b = pixels[i]
        # Convert to YCbCr (only Cb/Cr used for vectorscope)
        cb = int(128 + (-0.169 * r - 0.331 * g + 0.500 * b))
        cr = int(128 + (0.500 * r - 0.419 * g - 0.081 * b))

        cb = max(0, min(255, cb))
        cr = max(0, min(255, cr))

        grid[cr][cb] += 1

        if check_legal:
            if cb < LEGAL_CHROMA_MIN or cb > LEGAL_CHROMA_MAX:
                violations += 1
            if cr < LEGAL_CHROMA_MIN or cr > LEGAL_CHROMA_MAX:
                violations += 1

    # Normalise grid for display (max value -> 255)
    max_val = 1
    for row in grid:
        mv = max(row)
        if mv > max_val:
            max_val = mv

    normalised = []
    for row in grid:
        normalised.append([min(255, int(v / max_val * 255)) for v in row])

    data = {
        "grid": normalised,
        "size": scope_size,
        "center": [128, 128],
        "skin_tone_line": {"start": [128, 148], "end": [128, 58]},
    }
    return data, violations


def _compute_histogram(pixels: list, width: int, height: int,
                       check_legal: bool = False) -> Tuple[dict, int]:
    """Compute RGB + luminance histograms."""
    r_hist = [0] * HISTOGRAM_BINS
    g_hist = [0] * HISTOGRAM_BINS
    b_hist = [0] * HISTOGRAM_BINS
    luma_hist = [0] * HISTOGRAM_BINS
    violations = 0

    for r, g, b in pixels:
        r_hist[min(255, max(0, r))] += 1
        g_hist[min(255, max(0, g))] += 1
        b_hist[min(255, max(0, b))] += 1
        luma = int(0.299 * r + 0.587 * g + 0.114 * b)
        luma = min(255, max(0, luma))
        luma_hist[luma] += 1

        if check_legal and (luma < LEGAL_BLACK or luma > LEGAL_WHITE):
            violations += 1

    # Normalise for display
    max_val = max(max(r_hist), max(g_hist), max(b_hist), max(luma_hist), 1)
    data = {
        "r": [v / max_val for v in r_hist],
        "g": [v / max_val for v in g_hist],
        "b": [v / max_val for v in b_hist],
        "luma": [v / max_val for v in luma_hist],
        "bins": HISTOGRAM_BINS,
        "max_count": max_val,
    }
    return data, violations


def _compute_parade(pixels: list, width: int, height: int,
                    check_legal: bool = False) -> Tuple[dict, int]:
    """Compute RGB parade (separate per-channel waveforms)."""
    sample_cols = min(width, WAVEFORM_WIDTH)
    step = max(1, width // sample_cols)
    violations = 0

    channels = {"r": [], "g": [], "b": []}
    for ch_idx, ch_name in enumerate(("r", "g", "b")):
        columns = [[] for _ in range(sample_cols)]
        for row in range(height):
            for col_idx in range(sample_cols):
                x = col_idx * step
                pix_idx = row * width + x
                if pix_idx >= len(pixels):
                    break
                val = pixels[pix_idx][ch_idx]
                columns[col_idx].append(val)

                if check_legal and (val < LEGAL_BLACK or val > LEGAL_WHITE):
                    violations += 1

        # Histogram per column
        col_hists = []
        for col in columns:
            hist = [0] * HISTOGRAM_BINS
            for v in col:
                hist[min(255, max(0, v))] += 1
            col_hists.append(hist)
        channels[ch_name] = col_hists

    data = {
        "r": channels["r"],
        "g": channels["g"],
        "b": channels["b"],
        "columns": sample_cols,
        "rows": HISTOGRAM_BINS,
    }
    return data, violations


def _compute_false_color(pixels: list, width: int, height: int,
                         check_legal: bool = False) -> Tuple[dict, int]:
    """Compute false colour exposure map.

    Maps luminance zones to colours for easy exposure visualisation:
    - Deep shadows (0-15): Blue
    - Shadows (16-49): Cyan
    - Lower mid (50-99): Green
    - Mid (100-149): Yellow-Green
    - Upper mid (150-199): Yellow
    - Highlights (200-234): Orange
    - Clipped (235-255): Red
    """
    zones = {
        "deep_shadow": {"range": [0, 15], "color": [0, 0, 255], "count": 0},
        "shadow": {"range": [16, 49], "color": [0, 200, 255], "count": 0},
        "lower_mid": {"range": [50, 99], "color": [0, 200, 0], "count": 0},
        "mid": {"range": [100, 149], "color": [180, 230, 0], "count": 0},
        "upper_mid": {"range": [150, 199], "color": [255, 255, 0], "count": 0},
        "highlight": {"range": [200, 234], "color": [255, 140, 0], "count": 0},
        "clipped": {"range": [235, 255], "color": [255, 0, 0], "count": 0},
    }
    violations = 0
    total_pixels = len(pixels)
    luma_map = []  # Flat list of zone indices per pixel

    zone_list = list(zones.keys())
    for r, g, b in pixels:
        luma = int(0.299 * r + 0.587 * g + 0.114 * b)
        luma = min(255, max(0, luma))

        zone_idx = 0
        for zi, zname in enumerate(zone_list):
            lo, hi = zones[zname]["range"]
            if lo <= luma <= hi:
                zones[zname]["count"] += 1
                zone_idx = zi
                break
        luma_map.append(zone_idx)

        if check_legal and (luma < LEGAL_BLACK or luma > LEGAL_WHITE):
            violations += 1

    # Compute percentages
    zone_data = {}
    for zname, zinfo in zones.items():
        pct = (zinfo["count"] / total_pixels * 100) if total_pixels > 0 else 0
        zone_data[zname] = {
            "range": zinfo["range"],
            "color": zinfo["color"],
            "count": zinfo["count"],
            "percentage": round(pct, 2),
        }

    data = {
        "zones": zone_data,
        "width": width,
        "height": height,
        "map": luma_map,  # Flat list of zone indices for pixel-by-pixel rendering
        "zone_names": zone_list,
    }
    return data, violations


_SCOPE_FUNCS = {
    "waveform": _compute_waveform,
    "vectorscope": _compute_vectorscope,
    "histogram": _compute_histogram,
    "parade": _compute_parade,
    "false_color": _compute_false_color,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_scope(
    video_path: str,
    scope_type: str,
    timestamp: float = 0.0,
    check_legal: bool = False,
    on_progress: Optional[Callable] = None,
) -> ScopeResult:
    """Generate scope data for a single frame.

    Args:
        video_path: Source video file.
        scope_type: One of SCOPE_TYPES.
        timestamp: Frame time position in seconds.
        check_legal: Count broadcast legal range violations.
        on_progress: Progress callback (percentage int).

    Returns:
        ScopeResult with data dict and metadata.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if scope_type not in SCOPE_TYPES:
        raise ValueError(f"Unknown scope type '{scope_type}'. Available: {list(SCOPE_TYPES)}")

    t0 = time.time()

    if on_progress:
        on_progress(5)

    # Extract frame to temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", prefix="scope_",
                                      delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        _extract_frame(video_path, timestamp, tmp_path)

        if on_progress:
            on_progress(30)

        # Load pixels
        img, pixels, w, h = _load_frame_pixels(tmp_path)

        if on_progress:
            on_progress(50)

        # Compute scope
        scope_fn = _SCOPE_FUNCS[scope_type]
        data, violations = scope_fn(pixels, w, h, check_legal=check_legal)

        if on_progress:
            on_progress(90)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    elapsed = (time.time() - t0) * 1000

    if on_progress:
        on_progress(100)

    return ScopeResult(
        scope_type=scope_type,
        data=data,
        legal_range_violations=violations,
        frame_timestamp=timestamp,
        processing_time_ms=round(elapsed, 1),
    )


def generate_scopes_preset(
    video_path: str,
    preset: str = "colorist",
    timestamp: float = 0.0,
    on_progress: Optional[Callable] = None,
) -> MultiScopeResult:
    """Generate all scopes in a preset for a single frame.

    Extracts the frame once and computes all requested scopes on the
    same pixel data.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if preset not in SCOPE_PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(SCOPE_PRESETS.keys())}")

    preset_info = SCOPE_PRESETS[preset]
    scope_list = preset_info["scopes"]
    check_legal = preset_info.get("options", {}).get("legal_range", False)

    t0 = time.time()

    if on_progress:
        on_progress(5)

    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", prefix="scope_preset_",
                                      delete=False)
    tmp_path = tmp.name
    tmp.close()

    scopes = {}
    try:
        _extract_frame(video_path, timestamp, tmp_path)

        if on_progress:
            on_progress(20)

        img, pixels, w, h = _load_frame_pixels(tmp_path)

        total = len(scope_list)
        for idx, st in enumerate(scope_list):
            s_t0 = time.time()
            scope_fn = _SCOPE_FUNCS.get(st)
            if scope_fn is None:
                logger.warning("Unknown scope in preset: %s", st)
                continue

            data, violations = scope_fn(pixels, w, h, check_legal=check_legal)
            s_elapsed = (time.time() - s_t0) * 1000

            scopes[st] = ScopeResult(
                scope_type=st,
                data=data,
                legal_range_violations=violations,
                frame_timestamp=timestamp,
                processing_time_ms=round(s_elapsed, 1),
            )

            if on_progress:
                pct = int(20 + (idx + 1) / total * 75)
                on_progress(pct)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    elapsed = (time.time() - t0) * 1000
    if on_progress:
        on_progress(100)

    return MultiScopeResult(
        scopes=scopes,
        preset=preset,
        frame_timestamp=timestamp,
        total_time_ms=round(elapsed, 1),
    )


def generate_scope_from_frame(
    frame_path: str,
    scope_type: str,
    check_legal: bool = False,
) -> ScopeResult:
    """Generate scope data from an already-extracted frame image.

    Useful when the caller has already extracted the frame (e.g. from
    the preview pipeline) and wants scope data without re-extracting.
    """
    if not os.path.isfile(frame_path):
        raise FileNotFoundError(f"Frame not found: {frame_path}")
    if scope_type not in SCOPE_TYPES:
        raise ValueError(f"Unknown scope type: {scope_type}")

    t0 = time.time()
    img, pixels, w, h = _load_frame_pixels(frame_path)
    scope_fn = _SCOPE_FUNCS[scope_type]
    data, violations = scope_fn(pixels, w, h, check_legal=check_legal)
    elapsed = (time.time() - t0) * 1000

    return ScopeResult(
        scope_type=scope_type,
        data=data,
        legal_range_violations=violations,
        frame_timestamp=0.0,
        processing_time_ms=round(elapsed, 1),
    )


def list_scope_types() -> List[dict]:
    """Return available scope types with descriptions."""
    descs = {
        "waveform": "Per-column luminance/RGB distribution (waveform monitor)",
        "vectorscope": "UV chrominance distribution plot",
        "histogram": "RGB and luminance histograms",
        "parade": "Separate RGB waveforms side by side",
        "false_color": "Exposure zones mapped to false colours",
    }
    return [{"id": st, "name": st.replace("_", " ").title(),
             "description": descs.get(st, "")} for st in SCOPE_TYPES]


def list_presets() -> List[dict]:
    """Return available scope presets."""
    return [
        {"id": k, **v} for k, v in SCOPE_PRESETS.items()
    ]
