"""
OpenCut Proxy Generation Pipeline (23.1)

Auto-generate low-resolution proxy files for editing performance:
- Generate proxy at configurable resolution/codec
- Batch proxy generation with progress
- Proxy-to-original relinking for final export
- Proxy storage management

Uses FFmpeg for transcoding to lightweight proxy formats.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional

from opencut.helpers import FFmpegCmd, get_video_info, run_ffmpeg

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROXY_SUFFIX = "_proxy"
PROXY_METADATA_FILE = ".opencut_proxy_map.json"

PROXY_PRESETS = {
    "quarter": {"scale_factor": 0.25, "crf": 28, "codec": "libx264", "preset": "fast"},
    "half": {"scale_factor": 0.5, "crf": 23, "codec": "libx264", "preset": "fast"},
    "720p": {"width": 1280, "height": 720, "crf": 23, "codec": "libx264", "preset": "fast"},
    "540p": {"width": 960, "height": 540, "crf": 28, "codec": "libx264", "preset": "veryfast"},
    "360p": {"width": 640, "height": 360, "crf": 30, "codec": "libx264", "preset": "veryfast"},
    "prores_proxy": {"width": 1280, "height": 720, "codec": "prores_ks", "profile": 0},
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ProxyConfig:
    """Configuration for proxy generation."""
    preset: str = "half"
    width: int = 0
    height: int = 0
    scale_factor: float = 0.0
    crf: int = 23
    codec: str = "libx264"
    encoder_preset: str = "fast"
    audio_codec: str = "aac"
    audio_bitrate: str = "128k"
    proxy_dir: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProxyResult:
    """Result from a single proxy generation."""
    original_path: str = ""
    proxy_path: str = ""
    original_width: int = 0
    original_height: int = 0
    proxy_width: int = 0
    proxy_height: int = 0
    file_size_bytes: int = 0
    compression_ratio: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BatchProxyResult:
    """Result from batch proxy generation."""
    total: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    results: List[ProxyResult] = field(default_factory=list)
    proxy_dir: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _resolve_config(config=None, **kwargs) -> ProxyConfig:
    """Parse ProxyConfig from various input formats."""
    if isinstance(config, ProxyConfig):
        return config
    if isinstance(config, dict):
        return ProxyConfig(
            preset=config.get("preset", "half"),
            width=int(config.get("width", 0)),
            height=int(config.get("height", 0)),
            scale_factor=float(config.get("scale_factor", 0)),
            crf=int(config.get("crf", 23)),
            codec=config.get("codec", "libx264"),
            encoder_preset=config.get("encoder_preset", "fast"),
            audio_codec=config.get("audio_codec", "aac"),
            audio_bitrate=config.get("audio_bitrate", "128k"),
            proxy_dir=config.get("proxy_dir", ""),
        )
    return ProxyConfig(**{k: v for k, v in kwargs.items()
                          if k in ProxyConfig.__dataclass_fields__})


def _compute_proxy_dimensions(
    orig_w: int, orig_h: int, config: ProxyConfig
) -> tuple:
    """Compute proxy width/height from config."""
    if config.width > 0 and config.height > 0:
        return config.width, config.height

    # Check preset for fixed dimensions
    preset_info = PROXY_PRESETS.get(config.preset, {})
    if "width" in preset_info and "height" in preset_info:
        return preset_info["width"], preset_info["height"]

    # Use scale factor
    sf = config.scale_factor or preset_info.get("scale_factor", 0.5)
    pw = int(orig_w * sf)
    ph = int(orig_h * sf)
    # Ensure even dimensions
    pw = pw + (pw % 2)
    ph = ph + (ph % 2)
    return max(2, pw), max(2, ph)


def _get_proxy_path(video_path: str, proxy_dir: str = "") -> str:
    """Compute proxy file path."""
    base = os.path.splitext(os.path.basename(video_path))[0]
    directory = proxy_dir or os.path.join(os.path.dirname(video_path), "proxies")
    return os.path.join(directory, f"{base}{PROXY_SUFFIX}.mp4")


def _save_proxy_map(proxy_dir: str, original: str, proxy: str):
    """Save proxy-to-original mapping for relinking."""
    map_path = os.path.join(proxy_dir, PROXY_METADATA_FILE)
    mapping = {}
    if os.path.isfile(map_path):
        try:
            with open(map_path, "r") as f:
                mapping = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    mapping[os.path.abspath(proxy)] = os.path.abspath(original)
    os.makedirs(proxy_dir, exist_ok=True)
    with open(map_path, "w") as f:
        json.dump(mapping, f, indent=2)


# ---------------------------------------------------------------------------
# Generate Single Proxy
# ---------------------------------------------------------------------------
def generate_proxy(
    video_path: str,
    output_path: Optional[str] = None,
    output_dir: str = "",
    config: Optional[ProxyConfig] = None,
    on_progress: Optional[Callable] = None,
    **kwargs,
) -> ProxyResult:
    """
    Generate a low-resolution proxy from a video file.

    Args:
        video_path: Source video file.
        output_path: Explicit output file path.
        output_dir: Directory for proxy output.
        config: ProxyConfig (or dict).
        on_progress: Callback(percent, message).

    Returns:
        ProxyResult with paths and dimensions.
    """
    cfg = _resolve_config(config, **kwargs)

    if on_progress:
        on_progress(5, "Analyzing source video...")

    info = get_video_info(video_path)
    orig_w = info.get("width", 1920)
    orig_h = info.get("height", 1080)

    pw, ph = _compute_proxy_dimensions(orig_w, orig_h, cfg)

    proxy_dir = output_dir or cfg.proxy_dir or os.path.join(
        os.path.dirname(video_path), "proxies")
    os.makedirs(proxy_dir, exist_ok=True)

    if output_path is None:
        output_path = _get_proxy_path(video_path, proxy_dir)

    if on_progress:
        on_progress(10, f"Generating {pw}x{ph} proxy...")

    # Resolve preset-specific codec settings
    preset_info = PROXY_PRESETS.get(cfg.preset, {})
    codec = cfg.codec or preset_info.get("codec", "libx264")
    crf = cfg.crf or preset_info.get("crf", 23)
    enc_preset = cfg.encoder_preset or preset_info.get("preset", "fast")

    vf = f"scale={pw}:{ph}:flags=bilinear"

    builder = (
        FFmpegCmd()
        .input(video_path)
        .video_filter(vf)
    )

    if codec == "prores_ks":
        profile = preset_info.get("profile", 0)
        builder = (
            builder
            .option("-c:v", "prores_ks")
            .option("-profile:v", str(profile))
        )
    else:
        builder = builder.video_codec(codec, crf=crf, preset=enc_preset)

    builder = (
        builder
        .audio_codec(cfg.audio_codec, bitrate=cfg.audio_bitrate)
        .faststart()
        .output(output_path)
    )

    cmd = builder.build()
    run_ffmpeg(cmd)

    # Save proxy mapping
    _save_proxy_map(proxy_dir, video_path, output_path)

    file_size = os.path.getsize(output_path) if os.path.isfile(output_path) else 0
    orig_size = os.path.getsize(video_path) if os.path.isfile(video_path) else 1
    ratio = orig_size / file_size if file_size > 0 else 0.0

    if on_progress:
        on_progress(100, "Proxy generated.")

    return ProxyResult(
        original_path=video_path,
        proxy_path=output_path,
        original_width=orig_w,
        original_height=orig_h,
        proxy_width=pw,
        proxy_height=ph,
        file_size_bytes=file_size,
        compression_ratio=round(ratio, 2),
    )


# ---------------------------------------------------------------------------
# Batch Generate Proxies
# ---------------------------------------------------------------------------
def batch_generate_proxies(
    file_paths: List[str],
    output_dir: str = "",
    config: Optional[ProxyConfig] = None,
    on_progress: Optional[Callable] = None,
    **kwargs,
) -> BatchProxyResult:
    """
    Generate proxies for multiple video files.

    Args:
        file_paths: List of source video paths.
        output_dir: Shared proxy output directory.
        config: ProxyConfig applied to all files.
        on_progress: Callback(percent, message).

    Returns:
        BatchProxyResult with per-file results.
    """
    cfg = _resolve_config(config, **kwargs)
    total = len(file_paths)
    result = BatchProxyResult(total=total, proxy_dir=output_dir)

    for idx, fp in enumerate(file_paths):
        if on_progress:
            pct = int((idx / max(1, total)) * 90) + 5
            on_progress(pct, f"Generating proxy {idx+1}/{total}...")

        if not os.path.isfile(fp):
            logger.warning("Skipping missing file: %s", fp)
            result.skipped += 1
            continue

        try:
            pr = generate_proxy(
                video_path=fp,
                output_dir=output_dir,
                config=cfg,
            )
            result.results.append(pr)
            result.completed += 1
        except Exception as exc:
            logger.error("Proxy generation failed for %s: %s", fp, exc)
            result.failed += 1
            result.results.append(ProxyResult(original_path=fp))

    if on_progress:
        on_progress(100, f"Batch complete: {result.completed}/{total} proxies.")

    return result


# ---------------------------------------------------------------------------
# Relink Proxy to Original
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 60.1 — Auto Proxy Ingest
# ---------------------------------------------------------------------------
_VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".mkv", ".mxf", ".m4v",
    ".wmv", ".flv", ".webm", ".ts", ".m2ts", ".mpg", ".mpeg",
    ".r3d", ".braw", ".ari", ".dng",
}


def auto_proxy_ingest(
    folder_path: str,
    threshold_resolution: int = 1920,
    proxy_preset: str = "720p",
    output_dir: str = "",
    recursive: bool = True,
    on_progress: Optional[Callable] = None,
) -> BatchProxyResult:
    """
    Automatically detect high-resolution clips in a folder and generate proxies.

    Scans for video files exceeding threshold_resolution width, generates
    proxies for all qualifying clips, and maintains a proxy manifest.

    Args:
        folder_path: Root folder to scan for media.
        threshold_resolution: Minimum width to trigger proxy generation.
        proxy_preset: Proxy preset name (from PROXY_PRESETS).
        output_dir: Directory for proxy output. Defaults to <folder>/proxies.
        recursive: Whether to scan subdirectories.
        on_progress: Callback(pct, msg).

    Returns:
        BatchProxyResult with per-file results.
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    if on_progress:
        on_progress(5, "Scanning for high-resolution clips...")

    # Discover video files
    video_files = []
    if recursive:
        for root, _dirs, files in os.walk(folder_path):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext in _VIDEO_EXTENSIONS:
                    video_files.append(os.path.join(root, fname))
    else:
        for fname in os.listdir(folder_path):
            ext = os.path.splitext(fname)[1].lower()
            if ext in _VIDEO_EXTENSIONS:
                video_files.append(os.path.join(folder_path, fname))

    if not video_files:
        if on_progress:
            on_progress(100, "No video files found")
        return BatchProxyResult(total=0, proxy_dir=output_dir or folder_path)

    if on_progress:
        on_progress(10, f"Found {len(video_files)} video files, checking resolution...")

    # Filter by resolution threshold
    high_res = []
    for i, fp in enumerate(video_files):
        try:
            info = get_video_info(fp)
            w = info.get("width", 0)
            if w >= threshold_resolution:
                high_res.append(fp)
        except Exception as e:
            logger.debug("Skipping %s: %s", fp, e)

        if on_progress and (i + 1) % 10 == 0:
            pct = 10 + int((i / len(video_files)) * 20)
            on_progress(pct, f"Checking {i + 1}/{len(video_files)} files...")

    if not high_res:
        if on_progress:
            on_progress(100, f"No clips exceed {threshold_resolution}px threshold")
        return BatchProxyResult(
            total=len(video_files), skipped=len(video_files),
            proxy_dir=output_dir or folder_path,
        )

    if on_progress:
        on_progress(30, f"{len(high_res)} clips need proxies, generating...")

    # Check existing proxy map to skip already-proxied files
    proxy_dir = output_dir or os.path.join(folder_path, "proxies")
    map_path = os.path.join(proxy_dir, PROXY_METADATA_FILE)
    existing_originals = set()
    if os.path.isfile(map_path):
        try:
            with open(map_path, "r") as f:
                mapping = json.load(f)
            existing_originals = set(mapping.values())
        except (json.JSONDecodeError, OSError):
            pass

    to_generate = [
        fp for fp in high_res
        if os.path.abspath(fp) not in existing_originals
    ]
    skipped = len(high_res) - len(to_generate)

    if not to_generate:
        if on_progress:
            on_progress(100, f"All {len(high_res)} clips already have proxies")
        return BatchProxyResult(
            total=len(high_res), skipped=len(high_res),
            proxy_dir=proxy_dir,
        )

    # Generate proxies
    cfg = ProxyConfig(preset=proxy_preset)
    result = batch_generate_proxies(
        file_paths=to_generate,
        output_dir=proxy_dir,
        config=cfg,
        on_progress=lambda pct, msg: (
            on_progress(30 + int(pct * 0.65), msg) if on_progress else None
        ),
    )

    result.skipped += skipped

    if on_progress:
        on_progress(100, f"Auto ingest complete: {result.completed} proxies generated")

    return result


# ---------------------------------------------------------------------------
# Relink Proxy to Original
# ---------------------------------------------------------------------------
def relink_proxy_to_original(
    proxy_path: str,
    proxy_dir: str = "",
) -> str:
    """
    Resolve the original high-resolution file path from a proxy.

    Reads the proxy metadata map to find the original source.

    Args:
        proxy_path: Path to a proxy file.
        proxy_dir: Directory containing the proxy map file.

    Returns:
        Absolute path to the original file.

    Raises:
        FileNotFoundError: If the proxy map or original file is not found.
    """
    abs_proxy = os.path.abspath(proxy_path)
    search_dirs = []
    if proxy_dir:
        search_dirs.append(proxy_dir)
    search_dirs.append(os.path.dirname(abs_proxy))

    for d in search_dirs:
        map_path = os.path.join(d, PROXY_METADATA_FILE)
        if os.path.isfile(map_path):
            try:
                with open(map_path, "r") as f:
                    mapping = json.load(f)
                if abs_proxy in mapping:
                    original = mapping[abs_proxy]
                    if os.path.isfile(original):
                        return original
                    raise FileNotFoundError(
                        f"Original file not found: {original}")
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to read proxy map %s: %s", map_path, exc)

    raise FileNotFoundError(
        f"No proxy mapping found for: {proxy_path}")
