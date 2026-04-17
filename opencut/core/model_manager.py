"""
OpenCut Model Download Manager (Feature 10.1)

Background download queue with HTTP Range resume support, progress
tracking, bandwidth throttling, and disk space estimation.
"""

import json
import logging
import os
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.expanduser("~"), ".opencut", "models")
DOWNLOADS_META_DIR = os.path.join(os.path.expanduser("~"), ".opencut", "downloads")
DEFAULT_THROTTLE_KBPS = 0  # 0 = unlimited
CHUNK_SIZE = 64 * 1024  # 64 KB per read


# ---------------------------------------------------------------------------
# Model registry (known models)
# ---------------------------------------------------------------------------
KNOWN_MODELS: Dict[str, Dict] = {
    "whisper-tiny": {
        "url": "https://huggingface.co/openai/whisper-tiny/resolve/main/model.safetensors",
        "size_mb": 151,
        "description": "Whisper Tiny -- smallest, fastest transcription model",
        "category": "transcription",
    },
    "whisper-base": {
        "url": "https://huggingface.co/openai/whisper-base/resolve/main/model.safetensors",
        "size_mb": 290,
        "description": "Whisper Base -- good balance of speed and accuracy",
        "category": "transcription",
    },
    "whisper-small": {
        "url": "https://huggingface.co/openai/whisper-small/resolve/main/model.safetensors",
        "size_mb": 967,
        "description": "Whisper Small -- higher accuracy transcription",
        "category": "transcription",
    },
    "whisper-medium": {
        "url": "https://huggingface.co/openai/whisper-medium/resolve/main/model.safetensors",
        "size_mb": 3060,
        "description": "Whisper Medium -- high accuracy, slower",
        "category": "transcription",
    },
    "whisper-large-v3": {
        "url": "https://huggingface.co/openai/whisper-large-v3/resolve/main/model.safetensors",
        "size_mb": 6170,
        "description": "Whisper Large V3 -- best accuracy, requires significant VRAM",
        "category": "transcription",
    },
    "silero-vad": {
        "url": "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
        "size_mb": 2,
        "description": "Silero VAD -- voice activity detection",
        "category": "audio",
    },
    "realesrgan-x4": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "size_mb": 64,
        "description": "Real-ESRGAN x4 -- video upscaling model",
        "category": "upscaling",
    },
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class DownloadProgress:
    """Progress information for a model download."""
    model_name: str = ""
    url: str = ""
    total_bytes: int = 0
    downloaded_bytes: int = 0
    percent: float = 0.0
    speed_kbps: float = 0.0
    eta_seconds: float = 0.0
    status: str = "pending"  # pending, downloading, paused, completed, failed, cancelled
    error: Optional[str] = None
    output_path: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelEntry:
    """An installed or available model."""
    name: str = ""
    path: str = ""
    size_mb: float = 0.0
    description: str = ""
    category: str = ""
    installed: bool = False
    url: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DiskEstimation:
    """Disk space estimation for a set of models."""
    total_required_mb: float = 0.0
    available_mb: float = 0.0
    sufficient: bool = True
    models: List[Dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Download state
# ---------------------------------------------------------------------------
_downloads: Dict[str, DownloadProgress] = {}
_download_threads: Dict[str, threading.Thread] = {}
_cancel_flags: Dict[str, threading.Event] = {}
_download_lock = threading.Lock()


def _ensure_dirs():
    """Create models and downloads metadata directories."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DOWNLOADS_META_DIR, exist_ok=True)


def _model_output_path(model_name: str, url: str) -> str:
    """Compute the local file path for a model."""
    ext = os.path.splitext(url.split("?")[0])[1] or ".bin"
    safe_name = model_name.replace("/", "_").replace("\\", "_")
    return os.path.join(MODELS_DIR, f"{safe_name}{ext}")


def _save_download_meta(model_name: str, progress: DownloadProgress):
    """Persist download metadata for resume support."""
    _ensure_dirs()
    meta_path = os.path.join(DOWNLOADS_META_DIR, f"{model_name}.json")
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(progress.to_dict(), f)
    except OSError:
        pass


def _load_download_meta(model_name: str) -> Optional[Dict]:
    """Load persisted download metadata."""
    meta_path = os.path.join(DOWNLOADS_META_DIR, f"{model_name}.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    return None


# ---------------------------------------------------------------------------
# Download worker
# ---------------------------------------------------------------------------
def _download_worker(
    model_name: str,
    url: str,
    output_path: str,
    throttle_kbps: int,
    cancel_event: threading.Event,
    on_progress: Optional[Callable],
):
    """Background worker that downloads a model with resume support."""
    progress = _downloads.get(model_name)
    if progress is None:
        return

    # Check for partial download (resume)
    start_byte = 0
    if os.path.isfile(output_path):
        start_byte = os.path.getsize(output_path)
        progress.downloaded_bytes = start_byte

    headers = {
        "User-Agent": "OpenCut-ModelManager/1.0",
    }
    if start_byte > 0:
        headers["Range"] = f"bytes={start_byte}-"

    req = Request(url, headers=headers)
    progress.status = "downloading"

    try:
        with urlopen(req, timeout=30) as resp:
            # Determine total size
            content_length = resp.headers.get("Content-Length")
            if content_length:
                if start_byte > 0 and resp.status == 206:
                    progress.total_bytes = start_byte + int(content_length)
                else:
                    progress.total_bytes = int(content_length)
                    start_byte = 0  # server doesn't support Range

            mode = "ab" if start_byte > 0 and resp.status == 206 else "wb"
            if mode == "wb":
                progress.downloaded_bytes = 0

            last_time = time.time()
            last_bytes = progress.downloaded_bytes

            with open(output_path, mode) as f:
                while True:
                    if cancel_event.is_set():
                        progress.status = "cancelled"
                        _save_download_meta(model_name, progress)
                        return

                    chunk = resp.read(CHUNK_SIZE)
                    if not chunk:
                        break

                    f.write(chunk)
                    progress.downloaded_bytes += len(chunk)

                    # Update speed and ETA
                    now = time.time()
                    elapsed = now - last_time
                    if elapsed >= 0.5:
                        bytes_delta = progress.downloaded_bytes - last_bytes
                        progress.speed_kbps = round((bytes_delta / 1024) / elapsed, 1)
                        remaining = progress.total_bytes - progress.downloaded_bytes
                        if progress.speed_kbps > 0:
                            progress.eta_seconds = round(remaining / (progress.speed_kbps * 1024), 1)
                        last_time = now
                        last_bytes = progress.downloaded_bytes

                    if progress.total_bytes > 0:
                        progress.percent = round(
                            (progress.downloaded_bytes / progress.total_bytes) * 100, 1
                        )

                    if on_progress:
                        on_progress(progress.to_dict())

                    # Bandwidth throttling
                    if throttle_kbps > 0:
                        expected_time = len(chunk) / (throttle_kbps * 1024)
                        actual_time = time.time() - (now if elapsed < 0.5 else last_time)
                        if actual_time < expected_time:
                            time.sleep(expected_time - actual_time)

        progress.status = "completed"
        progress.percent = 100.0
        progress.output_path = output_path
        _save_download_meta(model_name, progress)

        if on_progress:
            on_progress(progress.to_dict())

        logger.info("Model download complete: %s -> %s", model_name, output_path)

    except (URLError, OSError) as exc:
        progress.status = "failed"
        progress.error = str(exc)
        _save_download_meta(model_name, progress)
        logger.error("Model download failed: %s -- %s", model_name, exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def queue_download(
    model_name: str,
    url: Optional[str] = None,
    output_path: Optional[str] = None,
    throttle_kbps: int = DEFAULT_THROTTLE_KBPS,
    on_progress: Optional[Callable] = None,
) -> DownloadProgress:
    """Queue a model for background download with resume support.

    Args:
        model_name: Identifier for the model.
        url: Download URL.  If None, looked up from KNOWN_MODELS.
        output_path: Local save path.  Defaults to ``~/.opencut/models/<name>.<ext>``.
        throttle_kbps: Max download speed in KB/s (0 = unlimited).
        on_progress: Optional progress callback.

    Returns:
        DownloadProgress with the initial state.
    """
    _ensure_dirs()

    # Resolve URL from registry if not provided
    if url is None:
        info = KNOWN_MODELS.get(model_name)
        if info is None:
            prog = DownloadProgress(
                model_name=model_name,
                status="failed",
                error=f"Unknown model: {model_name!r}. Provide a URL or use a known model name.",
            )
            return prog
        url = info["url"]

    if output_path is None:
        output_path = _model_output_path(model_name, url)

    with _download_lock:
        # Don't start duplicate downloads
        existing = _downloads.get(model_name)
        if existing and existing.status == "downloading":
            return existing

    progress = DownloadProgress(
        model_name=model_name,
        url=url,
        status="pending",
        output_path=output_path,
    )

    cancel_event = threading.Event()

    with _download_lock:
        _downloads[model_name] = progress
        _cancel_flags[model_name] = cancel_event

    thread = threading.Thread(
        target=_download_worker,
        args=(model_name, url, output_path, throttle_kbps, cancel_event, on_progress),
        daemon=True,
        name=f"opencut-dl-{model_name}",
    )

    with _download_lock:
        _download_threads[model_name] = thread

    thread.start()
    return progress


def get_download_progress(model_name: str) -> DownloadProgress:
    """Get the current download progress for a model.

    Args:
        model_name: The model identifier.

    Returns:
        DownloadProgress (status='unknown' if no download found).
    """
    with _download_lock:
        progress = _downloads.get(model_name)
    if progress:
        return progress

    # Check persisted metadata for completed/failed downloads
    meta = _load_download_meta(model_name)
    if meta:
        return DownloadProgress(**{
            k: v for k, v in meta.items()
            if k in DownloadProgress.__dataclass_fields__
        })

    return DownloadProgress(model_name=model_name, status="unknown")


def cancel_download(model_name: str) -> bool:
    """Cancel an in-progress download.

    Args:
        model_name: The model identifier.

    Returns:
        True if a download was cancelled, False if none was active.
    """
    with _download_lock:
        cancel_event = _cancel_flags.get(model_name)
        if cancel_event:
            cancel_event.set()
            progress = _downloads.get(model_name)
            if progress:
                progress.status = "cancelled"
            return True
    return False


def list_available_models() -> List[ModelEntry]:
    """List all known models (both installed and available for download).

    Returns:
        List of ModelEntry objects.
    """
    _ensure_dirs()
    entries = []
    for name, info in KNOWN_MODELS.items():
        local_path = _model_output_path(name, info["url"])
        installed = os.path.isfile(local_path)
        size_mb = info.get("size_mb", 0)
        if installed:
            try:
                size_mb = round(os.path.getsize(local_path) / (1024 * 1024), 1)
            except OSError:
                pass
        entries.append(ModelEntry(
            name=name,
            path=local_path if installed else "",
            size_mb=size_mb,
            description=info.get("description", ""),
            category=info.get("category", ""),
            installed=installed,
            url=info["url"],
        ))
    return entries


def list_installed_models() -> List[ModelEntry]:
    """List only models that are installed locally.

    Returns:
        List of ModelEntry objects for installed models.
    """
    return [m for m in list_available_models() if m.installed]


def estimate_disk_usage(model_list: Optional[List[str]] = None) -> DiskEstimation:
    """Estimate disk space required for a set of models.

    Args:
        model_list: List of model names.  If None, estimates for all known models.

    Returns:
        DiskEstimation with space requirements and availability.
    """
    _ensure_dirs()
    if model_list is None:
        model_list = list(KNOWN_MODELS.keys())

    models_info = []
    total_mb = 0.0
    for name in model_list:
        info = KNOWN_MODELS.get(name, {})
        size = info.get("size_mb", 0)
        # Skip already-installed models
        local_path = _model_output_path(name, info.get("url", ""))
        installed = os.path.isfile(local_path)
        if not installed:
            total_mb += size
        models_info.append({
            "name": name,
            "size_mb": size,
            "installed": installed,
            "needs_download": not installed,
        })

    # Check available disk space
    try:
        usage = shutil.disk_usage(MODELS_DIR)
        available_mb = round(usage.free / (1024 * 1024), 1)
    except OSError:
        available_mb = 0.0

    return DiskEstimation(
        total_required_mb=round(total_mb, 1),
        available_mb=available_mb,
        sufficient=available_mb >= total_mb,
        models=models_info,
    )
