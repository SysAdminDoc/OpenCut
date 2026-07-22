"""
OpenCut Model Download Manager (Feature 10.1)

Background download queue with HTTP Range resume support, progress
tracking, bandwidth throttling, and disk space estimation.
"""

import hashlib
import json
import logging
import os
import re
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

from opencut.core.asr_nemo_models import CANARY_SPEC, PARAKEET_SPEC

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
    "parakeet-tdt-0.6b-v3": {
        "url": PARAKEET_SPEC.download_url,
        "sha256": PARAKEET_SPEC.sha256,
        "size_mb": PARAKEET_SPEC.size_mb,
        "description": "Pinned NVIDIA Parakeet TDT 0.6B v3 multilingual ASR (NeMo)",
        "category": "transcription",
    },
    "canary-1b-flash": {
        "url": CANARY_SPEC.download_url,
        "sha256": CANARY_SPEC.sha256,
        "size_mb": CANARY_SPEC.size_mb,
        "description": "Pinned NVIDIA Canary 1B Flash batch ASR and translation (NeMo)",
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
    "seedvr2-3b": {
        "url": "https://huggingface.co/ByteDance-Seed/SeedVR2-3B/resolve/main/seedvr2_3b.safetensors",
        "size_mb": 6000,
        "description": "SeedVR2-3B -- one-step diffusion video super-resolution (Apache-2.0)",
        "category": "upscaling",
    },
    "latentsync-1.6": {
        "url": "https://huggingface.co/ByteDance/LatentSync-1.6/resolve/main/latentsync_unet.pt",
        "size_mb": 5200,
        "description": "LatentSync-1.6 -- audio-conditioned diffusion lip-sync (code Apache-2.0; checkpoint licence opt-in)",
        "category": "lip_sync",
    },
    "pyannote-community-1": {
        "url": "https://huggingface.co/pyannote/speaker-diarization-community-1/resolve/main/config.yaml",
        "size_mb": 90,
        "description": "pyannote speaker-diarization-community-1 -- default diarization pipeline (CC-BY-4.0)",
        "category": "diarization",
    },
    "ic-light-v1": {
        "url": "https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors",
        "size_mb": 1700,
        "description": "IC-Light v1 FC -- text-conditioned per-frame relight (Apache-2.0)",
        "category": "relight",
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
    # Cache validator (ETag / Last-Modified) recorded so a later resume can send
    # If-Range and detect that the remote asset changed, avoiding silent
    # corruption from stitching new bytes onto a stale partial file.
    etag: str = ""

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


_MODEL_KEY_RE = re.compile(r"[^A-Za-z0-9._-]")


def _safe_model_key(model_name: str) -> str:
    """Filesystem-safe token for a model name, used for metadata filenames.

    ``model_name`` is attacker-controllable via the download API, so it must
    never reach a path join verbatim: separators and other special characters
    are collapsed to ``_`` and leading dots are stripped so the result can be
    neither a traversal sequence nor a hidden/dotfile.
    """
    key = _MODEL_KEY_RE.sub("_", model_name or "").lstrip(".")
    return (key or "model")[:200]


def _meta_path(model_name: str) -> str:
    """Path to a model's download-metadata JSON, confined to DOWNLOADS_META_DIR."""
    path = os.path.join(DOWNLOADS_META_DIR, f"{_safe_model_key(model_name)}.json")
    base = os.path.abspath(DOWNLOADS_META_DIR)
    resolved = os.path.abspath(path)
    if resolved != base and not resolved.startswith(base + os.sep):
        raise ValueError(f"Unsafe model name for metadata path: {model_name!r}")
    return resolved


def _save_download_meta(model_name: str, progress: DownloadProgress):
    """Persist download metadata for resume support."""
    _ensure_dirs()
    try:
        meta_path = _meta_path(model_name)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(progress.to_dict(), f)
    except (OSError, ValueError) as exc:
        # Losing resume metadata is non-fatal but must not be silent — a
        # swallowed failure here lets a later resume append onto an unverified
        # partial file. Surface it so the cause (full disk, bad name) is visible.
        logger.warning("Could not persist download metadata for %r: %s", model_name, exc)


def _load_download_meta(model_name: str) -> Optional[Dict]:
    """Load persisted download metadata."""
    try:
        meta_path = _meta_path(model_name)
    except ValueError:
        return None
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
class ModelIntegrityError(RuntimeError):
    """Raised when a freshly downloaded model fails integrity verification."""


def _sha256_of_file(path: str) -> str:
    """Stream *path* through SHA-256 without loading it fully into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_download(model_name: str, output_path: str, expected_sha256: Optional[str]) -> None:
    """Verify a completed download before it is marked available.

    Two independent, defense-in-depth checks:

    - **Checksum** — when the catalogue row carries a ``sha256`` the downloaded
      bytes are hashed and compared; any mismatch fails closed. Most catalogue
      URLs resolve to *mutable* upstream refs (``resolve/main``/``master``), so a
      pinned hash is optional per-row rather than mandatory — a fixed hash there
      would false-alarm on legitimate upstream republishes.
    - **Pickle scan** — pickle-format payloads (``.nemo``/``.pt``/``.pth``/
      ``.ckpt``/``.bin``) can execute code on load, a strictly worse tamper class
      than safetensors/onnx. They are scanned with picklescan (via
      :func:`opencut.core.model_safety.scan_model_file`) and rejected if flagged.

    Raises :class:`ModelIntegrityError` on any failure. The caller quarantines
    the file so a tampered payload is never treated as installed.
    """
    if expected_sha256:
        want = expected_sha256.split(":", 1)[-1].strip().lower()
        got = _sha256_of_file(output_path).lower()
        if got != want:
            raise ModelIntegrityError(
                f"Checksum mismatch for {model_name!r}: expected sha256 {want}, got {got}"
            )

    from opencut.core.model_safety import ModelSecurityError, is_pickle_format, scan_model_file

    if is_pickle_format(output_path):
        try:
            scan_model_file(output_path)
        except ModelSecurityError as exc:
            raise ModelIntegrityError(str(exc)) from exc


def _download_worker(
    model_name: str,
    url: str,
    output_path: str,
    throttle_kbps: int,
    cancel_event: threading.Event,
    on_progress: Optional[Callable],
    expected_sha256: Optional[str] = None,
):
    """Background worker that downloads a model with resume support."""
    progress = _downloads.get(model_name)
    if progress is None:
        return

    # Check for partial download (resume)
    start_byte = 0
    prior_validator = ""
    if os.path.isfile(output_path):
        start_byte = os.path.getsize(output_path)
        progress.downloaded_bytes = start_byte
        prior = _load_download_meta(model_name)
        if prior:
            prior_validator = prior.get("etag") or ""

    headers = {
        "User-Agent": "OpenCut-ModelManager/1.0",
    }
    if start_byte > 0:
        headers["Range"] = f"bytes={start_byte}-"
        # If the remote asset changed since the partial download, If-Range makes
        # the server send a full 200 (handled below as a fresh restart) instead
        # of a 206 range that would corrupt the file.
        if prior_validator:
            headers["If-Range"] = prior_validator

    req = Request(url, headers=headers)
    progress.status = "downloading"

    try:
        with urlopen(req, timeout=30) as resp:
            # Record the cache validator so a future resume can send If-Range.
            progress.etag = (
                resp.headers.get("ETag") or resp.headers.get("Last-Modified") or prior_validator
            )
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

        # Verify integrity before the file is treated as installed. A failure
        # here quarantines (deletes) the payload so a tampered/corrupt download
        # is never advertised as available.
        try:
            _verify_download(model_name, output_path, expected_sha256)
        except ModelIntegrityError as exc:
            progress.status = "failed"
            progress.percent = 0.0
            progress.error = str(exc)
            try:
                os.remove(output_path)
            except OSError:
                pass
            _save_download_meta(model_name, progress)
            if on_progress:
                on_progress(progress.to_dict())
            logger.error("Model integrity check failed: %s -- %s", model_name, exc)
            return

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
    expected_sha256: Optional[str] = None
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
        expected_sha256 = info.get("sha256")
    else:
        # User-supplied URL — validate it blocks file://, private IPs, and
        # other non-public targets before handing it to urllib.
        from opencut.core.url_safety import validate_public_http_url  # local import avoids circular
        try:
            url = validate_public_http_url(url, label="Model download URL")
        except ValueError as exc:
            prog = DownloadProgress(
                model_name=model_name,
                status="failed",
                error=str(exc),
            )
            return prog

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
        args=(model_name, url, output_path, throttle_kbps, cancel_event, on_progress, expected_sha256),
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
