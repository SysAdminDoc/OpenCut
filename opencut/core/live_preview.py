"""
OpenCut Live Preview — Real-Time AI Effect Preview

Apply visual effects to individual frames at reduced resolution for
instant feedback.  Supported effects include color grading, denoising,
stabilisation-frame sharpening, style transfer placeholders, background
removal, upscale preview, sharpen, blur, vignette, and film grain.

Each effect validates its parameters, applies the transformation via
FFmpeg filters or PIL operations, and returns a PreviewResult with the
output path, processing time, and cache status.
"""

import base64
import hashlib
import json
import logging
import os
import subprocess as _sp
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import get_ffmpeg_path

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Preview resolution defaults
# ---------------------------------------------------------------------------
PREVIEW_WIDTH = 854
PREVIEW_HEIGHT = 480
PREVIEW_QUALITY = 5  # JPEG quality param for FFmpeg (2=best, 31=worst)

# ---------------------------------------------------------------------------
# In-memory preview cache (LRU-ish, bounded)
# ---------------------------------------------------------------------------
_preview_cache: Dict[str, str] = {}  # hash -> file path
_preview_cache_order: list = []
_CACHE_MAX_ENTRIES = 100
_CACHE_MAX_BYTES = 500 * 1024 * 1024  # 500 MB


def _cache_key(input_path: str, timestamp: float, effect: str,
               params: dict) -> str:
    """Compute a deterministic cache key from inputs."""
    mtime = 0.0
    try:
        mtime = os.path.getmtime(input_path)
    except OSError:
        pass
    raw = json.dumps({
        "path": input_path,
        "mtime": mtime,
        "ts": timestamp,
        "effect": effect,
        "params": params,
    }, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _cache_get(key: str) -> Optional[str]:
    """Return cached preview path if it exists on disk, else None."""
    path = _preview_cache.get(key)
    if path and os.path.isfile(path):
        # Move to end (most-recently used)
        if key in _preview_cache_order:
            _preview_cache_order.remove(key)
        _preview_cache_order.append(key)
        return path
    # Stale entry
    _preview_cache.pop(key, None)
    if key in _preview_cache_order:
        _preview_cache_order.remove(key)
    return None


def _cache_put(key: str, path: str) -> None:
    """Store a preview result in the cache, evicting old entries as needed."""
    _preview_cache[key] = path
    _preview_cache_order.append(key)
    _evict_cache()


def _evict_cache() -> None:
    """Evict oldest entries when count or size limits are exceeded."""
    # Count-based eviction
    while len(_preview_cache_order) > _CACHE_MAX_ENTRIES:
        old_key = _preview_cache_order.pop(0)
        old_path = _preview_cache.pop(old_key, None)
        if old_path:
            try:
                os.unlink(old_path)
            except OSError:
                pass

    # Size-based eviction
    total = 0
    for p in _preview_cache.values():
        try:
            total += os.path.getsize(p)
        except OSError:
            pass
    while total > _CACHE_MAX_BYTES and _preview_cache_order:
        old_key = _preview_cache_order.pop(0)
        old_path = _preview_cache.pop(old_key, None)
        if old_path:
            try:
                sz = os.path.getsize(old_path)
                os.unlink(old_path)
                total -= sz
            except OSError:
                pass


def clear_preview_cache() -> int:
    """Remove all cached preview files.  Returns count removed."""
    count = 0
    for path in list(_preview_cache.values()):
        try:
            if os.path.isfile(path):
                os.unlink(path)
                count += 1
        except OSError:
            pass
    _preview_cache.clear()
    _preview_cache_order.clear()
    return count


def preview_cache_stats() -> dict:
    """Return current in-memory preview cache statistics."""
    total_bytes = 0
    for p in _preview_cache.values():
        try:
            total_bytes += os.path.getsize(p)
        except OSError:
            pass
    return {
        "entry_count": len(_preview_cache),
        "total_size_mb": round(total_bytes / (1024 * 1024), 2),
        "max_entries": _CACHE_MAX_ENTRIES,
        "max_size_mb": round(_CACHE_MAX_BYTES / (1024 * 1024), 2),
    }


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class PreviewResult:
    """Result of a live preview operation."""
    preview_path: str = ""
    effect_applied: str = ""
    resolution: str = ""
    processing_time_ms: float = 0.0
    cached: bool = False
    timestamp: float = 0.0
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------
def _extract_frame(video_path: str, timestamp: float,
                   output_path: str, width: int = PREVIEW_WIDTH,
                   height: int = PREVIEW_HEIGHT) -> str:
    """Extract a single frame from video at *timestamp*, scaled to preview res."""
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-ss", str(max(0.0, timestamp)),
        "-i", video_path,
        "-frames:v", "1",
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
               f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
        "-q:v", str(PREVIEW_QUALITY),
        "-y", output_path,
    ]
    result = _sp.run(cmd, capture_output=True, timeout=30)
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")[-300:]
        raise RuntimeError(f"Frame extraction failed: {stderr}")
    return output_path


# ---------------------------------------------------------------------------
# Effect implementations
# ---------------------------------------------------------------------------
def _validate_strength(params: dict, default: float = 0.5,
                       min_v: float = 0.0, max_v: float = 1.0) -> float:
    """Extract and clamp a 'strength' parameter."""
    try:
        val = float(params.get("strength", default))
    except (TypeError, ValueError):
        val = default
    return max(min_v, min(max_v, val))


def _apply_ffmpeg_filter(input_path: str, output_path: str,
                         vf: str) -> str:
    """Apply an FFmpeg video filter to a single image."""
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-vf", vf,
        "-q:v", str(PREVIEW_QUALITY),
        "-y", output_path,
    ]
    result = _sp.run(cmd, capture_output=True, timeout=30)
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")[-300:]
        raise RuntimeError(f"FFmpeg filter failed: {stderr}")
    return output_path


def effect_color_grade(frame_path: str, output_path: str,
                       params: dict) -> str:
    """Apply colour grading via FFmpeg eq/curves filters."""
    brightness = float(params.get("brightness", 0.0))
    contrast = float(params.get("contrast", 1.0))
    saturation = float(params.get("saturation", 1.0))
    gamma = float(params.get("gamma", 1.0))
    temperature = float(params.get("temperature", 0.0))

    # Clamp ranges
    brightness = max(-1.0, min(1.0, brightness))
    contrast = max(0.1, min(3.0, contrast))
    saturation = max(0.0, min(3.0, saturation))
    gamma = max(0.1, min(5.0, gamma))
    temperature = max(-1.0, min(1.0, temperature))

    vf = f"eq=brightness={brightness}:contrast={contrast}:saturation={saturation}:gamma={gamma}"

    # Temperature adjustment via colour balance approximation
    if abs(temperature) > 0.01:
        vf += f",colorbalance=rs={temperature * 0.2}:bs={-temperature * 0.2}"

    return _apply_ffmpeg_filter(frame_path, output_path, vf)


def effect_denoise(frame_path: str, output_path: str,
                   params: dict) -> str:
    """Apply denoising via hqdn3d."""
    strength = _validate_strength(params, 0.5)
    luma = 2.0 + strength * 8.0
    chroma = 1.5 + strength * 6.0
    vf = f"hqdn3d={luma:.1f}:{chroma:.1f}:{luma + 1:.1f}:{chroma + 1:.1f}"
    return _apply_ffmpeg_filter(frame_path, output_path, vf)


def effect_stabilize_frame(frame_path: str, output_path: str,
                           params: dict) -> str:
    """Simulate stabilisation sharpness gain on a single frame."""
    strength = _validate_strength(params, 0.5)
    amount = 0.5 + strength * 1.5
    vf = f"unsharp=5:5:{amount:.2f}:5:5:0.0"
    return _apply_ffmpeg_filter(frame_path, output_path, vf)


def effect_style_transfer(frame_path: str, output_path: str,
                          params: dict) -> str:
    """Style transfer approximation via FFmpeg filter chains.

    Real neural style transfer requires a model; here we approximate
    popular styles with filter combos for instant preview.
    """
    style = params.get("style", "oil_painting")
    styles_map = {
        "oil_painting": "smartblur=lr=1.5:ls=-0.35:lt=-3.5:cr=0.9:cs=0.3:ct=0.3,unsharp=7:7:2.5:7:7:0.5",
        "watercolor": "colorbalance=rs=0.2:gs=-0.1:bs=0.3,smartblur=lr=2:ls=-0.5:lt=-3:cr=1:cs=0.5:ct=0.5",
        "pencil_sketch": "colorchannelmixer=.3:.4:.3:0:.3:.4:.3:0:.3:.4:.3,curves=all='0/0 0.5/0.6 1/1',unsharp=5:5:1.5:5:5:0",
        "vintage": "curves=r='0/0.1 0.5/0.5 1/0.9':g='0/0.05 0.5/0.45 1/0.85':b='0/0 0.5/0.4 1/0.8',eq=saturation=0.7:gamma=1.1",
        "noir": "colorchannelmixer=.3:.4:.3:0:.3:.4:.3:0:.3:.4:.3,curves=all='0/0 0.25/0.15 0.75/0.85 1/1',eq=contrast=1.3",
    }
    vf = styles_map.get(style, styles_map["oil_painting"])
    return _apply_ffmpeg_filter(frame_path, output_path, vf)


def effect_background_remove(frame_path: str, output_path: str,
                             params: dict) -> str:
    """Background removal preview via chroma-key or edge detection.

    Full background removal requires a segmentation model; this preview
    approximates the effect with an edge-highlight overlay.
    """
    vf = "edgedetect=low=0.1:high=0.3:mode=colormix"
    return _apply_ffmpeg_filter(frame_path, output_path, vf)


def effect_upscale_preview(frame_path: str, output_path: str,
                           params: dict) -> str:
    """Upscale with Lanczos interpolation (preview-quality)."""
    factor = max(1, min(4, int(params.get("factor", 2))))
    vf = f"scale=iw*{factor}:ih*{factor}:flags=lanczos"
    return _apply_ffmpeg_filter(frame_path, output_path, vf)


def effect_sharpen(frame_path: str, output_path: str,
                   params: dict) -> str:
    """Sharpen via FFmpeg unsharp mask."""
    strength = _validate_strength(params, 0.5)
    amount = 0.5 + strength * 2.5
    vf = f"unsharp=5:5:{amount:.2f}:5:5:0.0"
    return _apply_ffmpeg_filter(frame_path, output_path, vf)


def effect_blur(frame_path: str, output_path: str,
                params: dict) -> str:
    """Apply Gaussian blur."""
    strength = _validate_strength(params, 0.5)
    sigma = 1.0 + strength * 10.0
    # boxblur takes luma_radius:luma_power
    radius = max(1, int(sigma))
    vf = f"boxblur={radius}:{max(1, int(strength * 3 + 1))}"
    return _apply_ffmpeg_filter(frame_path, output_path, vf)


def effect_vignette(frame_path: str, output_path: str,
                    params: dict) -> str:
    """Apply a vignette darkening effect."""
    strength = _validate_strength(params, 0.5)
    angle = 0.3 + strength * 0.5  # PI/10 to PI/3.6 roughly
    vf = f"vignette=angle={angle:.3f}:mode=forward"
    return _apply_ffmpeg_filter(frame_path, output_path, vf)


def effect_film_grain(frame_path: str, output_path: str,
                      params: dict) -> str:
    """Overlay synthetic film grain noise."""
    strength = _validate_strength(params, 0.3)
    # noise filter: all_seed, all_strength, all_flags=t (temporal)
    noise_amount = max(1, int(strength * 40))
    vf = f"noise=alls={noise_amount}:allf=t+u"
    return _apply_ffmpeg_filter(frame_path, output_path, vf)


# ---------------------------------------------------------------------------
# Effect registry
# ---------------------------------------------------------------------------
EFFECTS: Dict[str, callable] = {
    "color_grade": effect_color_grade,
    "denoise": effect_denoise,
    "stabilize_frame": effect_stabilize_frame,
    "style_transfer": effect_style_transfer,
    "background_remove": effect_background_remove,
    "upscale_preview": effect_upscale_preview,
    "sharpen": effect_sharpen,
    "blur": effect_blur,
    "vignette": effect_vignette,
    "film_grain": effect_film_grain,
}


def list_effects() -> List[dict]:
    """Return metadata for all available preview effects."""
    info = {
        "color_grade": {
            "name": "Color Grade",
            "description": "Adjust brightness, contrast, saturation, gamma, and temperature",
            "params": ["brightness", "contrast", "saturation", "gamma", "temperature"],
        },
        "denoise": {
            "name": "Denoise",
            "description": "Reduce video noise with configurable strength",
            "params": ["strength"],
        },
        "stabilize_frame": {
            "name": "Stabilize (Frame)",
            "description": "Simulate stabilisation sharpness on a single frame",
            "params": ["strength"],
        },
        "style_transfer": {
            "name": "Style Transfer",
            "description": "Apply artistic style filters",
            "params": ["style"],
            "options": {"style": ["oil_painting", "watercolor", "pencil_sketch", "vintage", "noir"]},
        },
        "background_remove": {
            "name": "Background Remove",
            "description": "Edge-highlight preview of background removal",
            "params": [],
        },
        "upscale_preview": {
            "name": "Upscale Preview",
            "description": "Lanczos upscale at 2-4x factor",
            "params": ["factor"],
        },
        "sharpen": {
            "name": "Sharpen",
            "description": "Sharpen via unsharp mask",
            "params": ["strength"],
        },
        "blur": {
            "name": "Blur",
            "description": "Gaussian blur with adjustable radius",
            "params": ["strength"],
        },
        "vignette": {
            "name": "Vignette",
            "description": "Darkening vignette border effect",
            "params": ["strength"],
        },
        "film_grain": {
            "name": "Film Grain",
            "description": "Synthetic film grain overlay",
            "params": ["strength"],
        },
    }
    result = []
    for key in EFFECTS:
        entry = info.get(key, {"name": key, "description": "", "params": []})
        entry["id"] = key
        result.append(entry)
    return result


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------
def generate_live_preview(
    video_path: str,
    effect: str,
    params: Optional[dict] = None,
    timestamp: float = 0.0,
    width: int = PREVIEW_WIDTH,
    height: int = PREVIEW_HEIGHT,
    use_cache: bool = True,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> PreviewResult:
    """Generate a live preview of an effect applied to a video frame.

    Args:
        video_path: Source video file.
        effect: Effect name (must be in EFFECTS).
        params: Effect-specific parameters.
        timestamp: Time position in seconds.
        width: Preview width in pixels.
        height: Preview height in pixels.
        use_cache: Whether to use the in-memory cache.
        output_dir: Directory for output files.
        on_progress: Progress callback (percentage int).

    Returns:
        PreviewResult with preview_path and metadata.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if effect not in EFFECTS:
        raise ValueError(f"Unknown effect '{effect}'. Available: {list(EFFECTS.keys())}")

    params = params or {}
    t0 = time.time()

    if on_progress:
        on_progress(5)

    # Check cache
    ck = _cache_key(video_path, timestamp, effect, params)
    if use_cache:
        cached = _cache_get(ck)
        if cached:
            elapsed = (time.time() - t0) * 1000
            if on_progress:
                on_progress(100)
            return PreviewResult(
                preview_path=cached,
                effect_applied=effect,
                resolution=f"{width}x{height}",
                processing_time_ms=round(elapsed, 1),
                cached=True,
                timestamp=timestamp,
                params=params,
            )

    # Determine output dir
    if not output_dir:
        output_dir = tempfile.mkdtemp(prefix="opencut_preview_")
    os.makedirs(output_dir, exist_ok=True)

    if on_progress:
        on_progress(15)

    # Extract frame
    frame_path = os.path.join(output_dir, f"frame_{ck[:12]}.jpg")
    _extract_frame(video_path, timestamp, frame_path, width, height)

    if on_progress:
        on_progress(40)

    # Apply effect
    preview_path = os.path.join(output_dir, f"preview_{effect}_{ck[:12]}.jpg")
    effect_fn = EFFECTS[effect]
    effect_fn(frame_path, preview_path, params)

    if on_progress:
        on_progress(85)

    # Clean up raw frame if different from output
    if frame_path != preview_path:
        try:
            os.unlink(frame_path)
        except OSError:
            pass

    # Cache the result
    if use_cache:
        _cache_put(ck, preview_path)

    elapsed = (time.time() - t0) * 1000

    if on_progress:
        on_progress(100)

    return PreviewResult(
        preview_path=preview_path,
        effect_applied=effect,
        resolution=f"{width}x{height}",
        processing_time_ms=round(elapsed, 1),
        cached=False,
        timestamp=timestamp,
        params=params,
    )


def generate_preview_base64(
    video_path: str,
    effect: str,
    params: Optional[dict] = None,
    timestamp: float = 0.0,
    width: int = PREVIEW_WIDTH,
    height: int = PREVIEW_HEIGHT,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Generate a preview and return the image as a base64-encoded JPEG.

    Convenience wrapper around generate_live_preview for routes that
    want inline image data rather than file paths.
    """
    result = generate_live_preview(
        video_path=video_path,
        effect=effect,
        params=params,
        timestamp=timestamp,
        width=width,
        height=height,
        on_progress=on_progress,
    )
    img_b64 = ""
    if result.preview_path and os.path.isfile(result.preview_path):
        with open(result.preview_path, "rb") as fh:
            img_b64 = base64.b64encode(fh.read()).decode("ascii")
    d = result.to_dict()
    d["preview_b64"] = img_b64
    return d


def apply_effect_chain(
    video_path: str,
    effects: List[dict],
    timestamp: float = 0.0,
    width: int = PREVIEW_WIDTH,
    height: int = PREVIEW_HEIGHT,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> PreviewResult:
    """Apply a chain of effects sequentially to a single frame.

    Each entry in *effects* is a dict with keys ``effect`` and ``params``.
    Effects are applied in order; the output of one feeds the next.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not effects:
        raise ValueError("No effects specified")

    t0 = time.time()

    if not output_dir:
        output_dir = tempfile.mkdtemp(prefix="opencut_chain_")
    os.makedirs(output_dir, exist_ok=True)

    # Extract frame
    chain_hash = hashlib.sha256(
        json.dumps(effects, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]
    frame_path = os.path.join(output_dir, f"chain_frame_{chain_hash}.jpg")
    _extract_frame(video_path, timestamp, frame_path, width, height)

    current = frame_path
    total = len(effects)
    applied = []

    for idx, fx in enumerate(effects):
        effect_name = fx.get("effect", "")
        effect_params = fx.get("params", {})
        if effect_name not in EFFECTS:
            logger.warning("Skipping unknown effect in chain: %s", effect_name)
            continue

        step_out = os.path.join(output_dir, f"chain_step{idx}_{effect_name}_{chain_hash}.jpg")
        effect_fn = EFFECTS[effect_name]
        effect_fn(current, step_out, effect_params)

        # Remove intermediate file (unless it is the original frame)
        if current != frame_path or idx > 0:
            try:
                os.unlink(current)
            except OSError:
                pass
        current = step_out
        applied.append(effect_name)

        if on_progress:
            pct = int(20 + (idx + 1) / total * 70)
            on_progress(pct)

    # Clean original frame if we have steps
    if applied and os.path.isfile(frame_path) and current != frame_path:
        try:
            os.unlink(frame_path)
        except OSError:
            pass

    elapsed = (time.time() - t0) * 1000
    if on_progress:
        on_progress(100)

    return PreviewResult(
        preview_path=current,
        effect_applied="+".join(applied),
        resolution=f"{width}x{height}",
        processing_time_ms=round(elapsed, 1),
        cached=False,
        timestamp=timestamp,
        params={"chain": effects},
    )
