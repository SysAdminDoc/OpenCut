"""
OpenCut Proxy Generation Pipeline (23.1)

Auto-generate low-resolution proxy files for editing performance:
- Generate proxy at configurable resolution/codec
- Batch proxy generation with progress
- Proxy-to-original relinking for final export
- Proxy storage management

Uses FFmpeg for transcoding to lightweight proxy formats.
"""

import hashlib
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional

from opencut.helpers import FFmpegCmd, get_video_info, run_ffmpeg

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROXY_SUFFIX = "_proxy"
PROXY_METADATA_FILE = ".opencut_proxy_map.json"
PROXY_BATCH_SCHEMA_VERSION = 1
PROXY_BATCH_STATE_DIR = "proxy_batches"
_PROXY_STATE_LOCK = threading.RLock()

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
    cancelled: bool = False
    state_path: str = ""
    items: List[dict] = field(default_factory=list)

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


def _atomic_write_json(path: str, payload: dict) -> None:
    """Durably replace a JSON document without exposing partial contents."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temp_path = os.path.join(
        directory,
        f".{os.path.basename(path)}.{uuid.uuid4().hex}.tmp",
    )
    try:
        with open(temp_path, "x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        try:
            os.remove(temp_path)
        except FileNotFoundError:
            pass


def _load_proxy_map(proxy_dir: str) -> dict:
    map_path = os.path.join(proxy_dir, PROXY_METADATA_FILE)
    if not os.path.isfile(map_path):
        return {}
    try:
        with open(map_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_proxy_map(proxy_dir: str, original: str, proxy: str):
    """Atomically save a proxy-to-original mapping for relinking."""
    map_path = os.path.join(proxy_dir, PROXY_METADATA_FILE)
    with _PROXY_STATE_LOCK:
        mapping = _load_proxy_map(proxy_dir)
        mapping[os.path.abspath(proxy)] = os.path.abspath(original)
        _atomic_write_json(map_path, mapping)


def _proxy_batch_state_root() -> str:
    return os.path.abspath(os.path.expanduser(os.path.join("~", ".opencut", PROXY_BATCH_STATE_DIR)))


def validate_proxy_batch_state_path(path: str) -> str:
    """Confine caller-provided resume state to OpenCut's private state root."""
    raw_path = str(path or "").strip()
    if not raw_path:
        raise ValueError("Proxy batch state path is required")
    candidate = os.path.abspath(os.path.expanduser(raw_path))
    root = _proxy_batch_state_root()
    try:
        confined = os.path.commonpath([candidate, root]) == root
    except ValueError:
        confined = False
    if not confined or not candidate.endswith(".json"):
        raise ValueError("Proxy batch state path must be inside OpenCut's proxy_batches directory")
    return candidate


def proxy_batch_state_path(
    file_paths: List[str],
    output_dir: str = "",
    config: Optional[ProxyConfig] = None,
) -> str:
    """Return a stable private checkpoint path for a batch request."""
    cfg = _resolve_config(config)
    fingerprint = {
        "files": [os.path.abspath(str(path)) for path in file_paths],
        "output_dir": os.path.abspath(output_dir) if output_dir else "",
        "config": cfg.to_dict(),
    }
    encoded = json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode("utf-8")
    batch_id = hashlib.sha256(encoded).hexdigest()[:24]
    return os.path.join(_proxy_batch_state_root(), f"{batch_id}.json")


def load_proxy_batch_state(path: str) -> dict:
    """Load and minimally validate a persisted proxy batch checkpoint."""
    state_path = validate_proxy_batch_state_path(path)
    with _PROXY_STATE_LOCK:
        try:
            with open(state_path, "r", encoding="utf-8") as handle:
                state = json.load(handle)
        except FileNotFoundError as exc:
            raise ValueError("Proxy batch checkpoint is missing; start the batch again") from exc
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"Proxy batch checkpoint is unreadable: {exc}") from exc
    if not isinstance(state, dict) or state.get("schema_version") != PROXY_BATCH_SCHEMA_VERSION:
        raise ValueError("Proxy batch checkpoint has an unsupported schema")
    items = state.get("items")
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise ValueError("Proxy batch checkpoint has invalid items")
    return state


def _source_fingerprint(path: str) -> dict:
    try:
        stat = os.stat(path)
    except OSError:
        return {"source_size": 0, "source_mtime_ns": 0}
    return {"source_size": stat.st_size, "source_mtime_ns": stat.st_mtime_ns}


def _allocate_proxy_paths(file_paths: List[str], output_dir: str) -> List[str]:
    """Allocate stable outputs, disambiguating duplicate source basenames."""
    allocated = []
    claimed = set()
    for index, source in enumerate(file_paths):
        candidate = _get_proxy_path(source, output_dir)
        normalized = os.path.normcase(os.path.abspath(candidate))
        if normalized in claimed:
            base = os.path.splitext(os.path.basename(source))[0]
            identity = f"{os.path.abspath(source)}\0{index}".encode("utf-8")
            digest = hashlib.sha256(identity).hexdigest()[:8]
            directory = output_dir or os.path.join(os.path.dirname(source), "proxies")
            candidate = os.path.join(directory, f"{base}_{digest}{PROXY_SUFFIX}.mp4")
            normalized = os.path.normcase(os.path.abspath(candidate))
        claimed.add(normalized)
        allocated.append(os.path.abspath(candidate))
    return allocated


def ensure_proxy_batch_state(
    file_paths: List[str],
    output_dir: str = "",
    config: Optional[ProxyConfig] = None,
) -> str:
    """Create the full checkpoint before a batch is queued."""
    cfg = _resolve_config(config)
    sources = [os.path.abspath(path) for path in file_paths]
    state_path = proxy_batch_state_path(sources, output_dir, cfg)
    with _PROXY_STATE_LOCK:
        if os.path.isfile(state_path):
            load_proxy_batch_state(state_path)
            return state_path
        effective_output_dir = output_dir or cfg.proxy_dir
        outputs = _allocate_proxy_paths(sources, effective_output_dir)
        now = time.time()
        items = []
        for source, proxy in zip(sources, outputs):
            item = {
                "original_path": source,
                "proxy_path": proxy,
                "status": "pending",
                "error": "",
                "result": None,
            }
            item.update(_source_fingerprint(source))
            items.append(item)
        state = {
            "schema_version": PROXY_BATCH_SCHEMA_VERSION,
            "created_at": now,
            "updated_at": now,
            "output_dir": os.path.abspath(effective_output_dir) if effective_output_dir else "",
            "config": cfg.to_dict(),
            "cancelled": False,
            "items": items,
        }
        _atomic_write_json(state_path, state)
    return state_path


def _save_proxy_batch_state(path: str, state: dict) -> None:
    state["updated_at"] = time.time()
    with _PROXY_STATE_LOCK:
        _atomic_write_json(validate_proxy_batch_state_path(path), state)


# ---------------------------------------------------------------------------
# Generate Single Proxy
# ---------------------------------------------------------------------------
def generate_proxy(
    video_path: str,
    output_path: Optional[str] = None,
    output_dir: str = "",
    config: Optional[ProxyConfig] = None,
    on_progress: Optional[Callable] = None,
    job_id: str = "",
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
        job_id: Optional async job ID used to terminate FFmpeg on cancellation.

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

    output_path = os.path.abspath(output_path)
    output_stem, output_ext = os.path.splitext(output_path)
    temp_output = f"{output_stem}.{uuid.uuid4().hex}.partial{output_ext or '.mp4'}"

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
        .output(temp_output)
    )

    cmd = builder.build()
    try:
        run_ffmpeg(cmd, job_id=job_id)
        if not os.path.isfile(temp_output) or os.path.getsize(temp_output) <= 0:
            raise RuntimeError("FFmpeg did not produce a non-empty proxy")
        proxy_info = get_video_info(temp_output)
        actual_w = int(proxy_info.get("width") or 0)
        actual_h = int(proxy_info.get("height") or 0)
        if actual_w <= 0 or actual_h <= 0:
            raise RuntimeError("Generated proxy has no decodable video stream")
        os.replace(temp_output, output_path)
        _save_proxy_map(proxy_dir, video_path, output_path)
    finally:
        try:
            os.remove(temp_output)
        except FileNotFoundError:
            pass

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
        proxy_width=actual_w,
        proxy_height=actual_h,
        file_size_bytes=file_size,
        compression_ratio=round(ratio, 2),
    )


# ---------------------------------------------------------------------------
# Batch Generate Proxies
# ---------------------------------------------------------------------------
def _proxy_pair_is_valid(item: dict) -> bool:
    """Verify source identity, proxy media, and the relink map as one pair."""
    original = os.path.abspath(str(item.get("original_path") or ""))
    proxy = os.path.abspath(str(item.get("proxy_path") or ""))
    if not os.path.isfile(original) or not os.path.isfile(proxy):
        return False
    try:
        source_stat = os.stat(original)
        if item.get("source_size") not in (None, source_stat.st_size):
            return False
        if item.get("source_mtime_ns") not in (None, source_stat.st_mtime_ns):
            return False
        if os.path.getsize(proxy) <= 0:
            return False
        mapping = _load_proxy_map(os.path.dirname(proxy))
        if mapping.get(proxy) != original:
            return False
        info = get_video_info(proxy)
        width = int(info.get("width") or 0)
        height = int(info.get("height") or 0)
        if width <= 0 or height <= 0:
            return False
        stored_result = item.get("result") or {}
        expected_width = int(stored_result.get("proxy_width") or 0)
        expected_height = int(stored_result.get("proxy_height") or 0)
        if expected_width and width != expected_width:
            return False
        if expected_height and height != expected_height:
            return False
    except Exception:  # noqa: BLE001 - any probe failure means the pair is invalid
        return False
    return True


def _result_from_item(item: dict) -> ProxyResult:
    payload = item.get("result") if isinstance(item.get("result"), dict) else {}
    allowed = ProxyResult.__dataclass_fields__
    return ProxyResult(**{key: value for key, value in payload.items() if key in allowed})


def batch_generate_proxies(
    file_paths: List[str],
    output_dir: str = "",
    config: Optional[ProxyConfig] = None,
    on_progress: Optional[Callable] = None,
    state_path: str = "",
    is_cancelled: Optional[Callable[[], bool]] = None,
    job_id: str = "",
    **kwargs,
) -> BatchProxyResult:
    """
    Generate proxies for multiple video files.

    Args:
        file_paths: List of source video paths.
        output_dir: Shared proxy output directory.
        config: ProxyConfig applied to all files.
        on_progress: Callback(percent, message).
        state_path: Private JSON checkpoint created before queueing.
        is_cancelled: Callback checked before and after each FFmpeg process.
        job_id: Async job ID used to terminate the active FFmpeg child.

    Returns:
        BatchProxyResult with per-file results.
    """
    cfg = _resolve_config(config, **kwargs)
    sources = [os.path.abspath(path) for path in file_paths]
    if not state_path:
        state_path = ensure_proxy_batch_state(sources, output_dir, cfg)
    state_path = validate_proxy_batch_state_path(state_path)
    state = load_proxy_batch_state(state_path)
    state_sources = [os.path.abspath(str(item.get("original_path") or "")) for item in state["items"]]
    if sources and state_sources != sources:
        raise ValueError("Proxy batch checkpoint does not match the requested file list")
    if not sources:
        sources = state_sources
    total = len(state["items"])
    result = BatchProxyResult(
        total=total,
        proxy_dir=state.get("output_dir") or output_dir,
        state_path=state_path,
    )

    # A previous process can stop between marking an item running and starting
    # FFmpeg. Retrying failed/running items is safe because output promotion is
    # atomic and completed pairs are validated below.
    for item in state["items"]:
        if item.get("status") in {"running", "failed"}:
            item["status"] = "pending"
    state["cancelled"] = False
    _save_proxy_batch_state(state_path, state)

    for idx, item in enumerate(state["items"]):
        fp = os.path.abspath(str(item.get("original_path") or ""))
        proxy_path = os.path.abspath(str(item.get("proxy_path") or ""))
        if on_progress:
            pct = int((idx / max(1, total)) * 90) + 5
            on_progress(pct, f"Generating proxy {idx+1}/{total}...")

        if is_cancelled and is_cancelled():
            state["cancelled"] = True
            result.cancelled = True
            _save_proxy_batch_state(state_path, state)
            break

        if _proxy_pair_is_valid(item):
            item["status"] = "completed"
            item["error"] = ""
            result.results.append(_result_from_item(item))
            result.skipped += 1
            _save_proxy_batch_state(state_path, state)
            continue

        if not os.path.isfile(fp):
            logger.warning("Skipping missing file: %s", fp)
            item["status"] = "failed"
            item["error"] = "Source file is missing"
            result.skipped += 1
            _save_proxy_batch_state(state_path, state)
            continue

        item["status"] = "running"
        item["error"] = ""
        _save_proxy_batch_state(state_path, state)
        try:
            pr = generate_proxy(
                video_path=fp,
                output_path=proxy_path,
                output_dir=os.path.dirname(proxy_path),
                config=cfg,
                job_id=job_id,
            )
            result.results.append(pr)
            result.completed += 1
            item["status"] = "completed"
            item["result"] = pr.to_dict()
            item["error"] = ""
            item.update(_source_fingerprint(fp))
            _save_proxy_batch_state(state_path, state)
        except Exception as exc:
            if is_cancelled and is_cancelled():
                item["status"] = "pending"
                item["error"] = "Cancelled; ready to resume"
                state["cancelled"] = True
                result.cancelled = True
                _save_proxy_batch_state(state_path, state)
                break
            logger.error("Proxy generation failed for %s: %s", fp, exc)
            item["status"] = "failed"
            item["error"] = str(exc)
            result.results.append(ProxyResult(original_path=fp))
            _save_proxy_batch_state(state_path, state)

        if is_cancelled and is_cancelled():
            state["cancelled"] = True
            result.cancelled = True
            _save_proxy_batch_state(state_path, state)
            break

    result.failed = sum(1 for item in state["items"] if item.get("status") == "failed")
    result.items = [dict(item) for item in state["items"]]
    if on_progress:
        if result.cancelled:
            on_progress(
                min(99, int(((result.completed + result.skipped) / max(1, total)) * 100)),
                "Proxy batch cancelled; completed items are safe to resume.",
            )
        else:
            on_progress(100, f"Batch complete: {result.completed + result.skipped}/{total} proxies.")

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

    # Let the durable batch validator inspect both media and map entries.
    # Map membership alone is insufficient: the proxy may be missing,
    # truncated, stale relative to its source, or undecodable.
    proxy_dir = output_dir or os.path.join(folder_path, "proxies")

    # Generate proxies
    cfg = ProxyConfig(preset=proxy_preset)
    result = batch_generate_proxies(
        file_paths=high_res,
        output_dir=proxy_dir,
        config=cfg,
        on_progress=lambda pct, msg: (
            on_progress(30 + int(pct * 0.65), msg) if on_progress else None
        ),
    )

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
