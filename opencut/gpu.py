"""
GPU Memory Context Manager

Provides GPUContext for automatic VRAM cleanup after GPU operations,
plus standalone helpers for device detection and VRAM checks.

All torch imports are optional -- functions degrade gracefully on
CPU-only systems.
"""

import logging
import os
import re
import shutil
import subprocess

logger = logging.getLogger("opencut")


class GPUSelectionError(ValueError):
    """A requested CUDA adapter is not available on this machine."""

    code = "INVALID_GPU_INDEX"
    status_code = 400

    def __init__(self, requested, devices):
        self.requested = requested
        self.available_devices = [dict(device) for device in (devices or [])]
        available = ", ".join(
            f"{device.get('index')}: {device.get('name') or 'CUDA device'}"
            for device in self.available_devices
        ) or "none"
        super().__init__(
            f"GPU index {requested!r} is not available. Available CUDA devices: {available}."
        )

    def to_dict(self) -> dict:
        return {
            "error": str(self),
            "code": self.code,
            "requested_index": self.requested,
            "available_devices": self.available_devices,
            "suggestion": "Choose one of the listed device indexes or select Auto.",
        }


def _normalise_gpu_index(value, *, allow_auto: bool = True):
    """Return a non-negative CUDA index or ``None`` for automatic selection."""
    if value is None and allow_auto:
        return None
    if isinstance(value, bool):
        raise ValueError("GPU index must be a non-negative integer or auto")
    if isinstance(value, str):
        raw = value.strip().lower()
        if allow_auto and raw in {"", "auto", "default", "none", "-1"}:
            return None
        if not raw or not raw.isdigit():
            raise ValueError("GPU index must be a non-negative integer or auto")
        value = int(raw)
    if not isinstance(value, int) or value < 0:
        raise ValueError("GPU index must be a non-negative integer or auto")
    return value


def _configured_gpu_index():
    """Read the saved selection, falling back to the process configuration."""
    try:
        from opencut.user_data import read_user_file

        saved = read_user_file("gpu_settings.json", default=None)
        if isinstance(saved, dict) and "gpu_index" in saved:
            return _normalise_gpu_index(saved.get("gpu_index"))
    except (TypeError, ValueError, OSError) as exc:
        logger.warning("Ignoring invalid saved GPU selection: %s", exc)

    try:
        from flask import current_app, has_app_context

        if has_app_context():
            config = current_app.config.get("OPENCUT")
            if config is not None and hasattr(config, "gpu_index"):
                return _normalise_gpu_index(config.gpu_index)
    except (ImportError, RuntimeError, TypeError, ValueError):
        pass

    raw = os.environ.get("OPENCUT_GPU_INDEX")
    try:
        return _normalise_gpu_index(raw)
    except ValueError:
        logger.warning("Ignoring invalid OPENCUT_GPU_INDEX=%r", raw)
        return None


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def gpu_architecture(device: dict | None) -> str:
    """Return a stable architecture label for a detected CUDA adapter.

    NVIDIA's RTX 50-series is Blackwell and is exposed as compute capability
    12.x by current CUDA runtimes.  Name matching keeps the probe useful when
    ``nvidia-smi`` is available but PyTorch is not; capability matching covers
    renamed workstation and laptop variants.
    """
    device = device if isinstance(device, dict) else {}
    name = str(device.get("name", "")).strip().lower()
    if "blackwell" in name or re.search(r"\brtx\s*(?:50\d{2})\b", name):
        return "blackwell"

    capability = device.get("compute_capability", device.get("compute_cap"))
    if isinstance(capability, (tuple, list)) and capability:
        capability = ".".join(str(part) for part in capability[:2])
    match = re.search(r"(\d+)(?:\.(\d+))?", str(capability or ""))
    if match:
        major = int(match.group(1))
        minor = int(match.group(2) or 0)
        if major >= 12:
            return "blackwell"
        if (major, minor) >= (8, 9):
            return "ada-lovelace"
        if major >= 8:
            return "ampere"
        if (major, minor) >= (7, 5):
            return "turing"
    return "unknown"


def faster_whisper_compute_recommendation(device: dict | None) -> dict:
    """Describe the safe faster-whisper compute type for one GPU.

    CTranslate2's automatic compute-type selection can choose an unsupported
    path on RTX 50-series/Blackwell adapters.  Pinning float16 is the known
    compatible choice while preserving ``auto`` for other hardware.
    """
    architecture = gpu_architecture(device)
    affected = architecture == "blackwell"
    requested_compute_type = "auto"
    compute_type = "float16" if affected else requested_compute_type
    reason = ""
    warning = ""
    if affected:
        reason = (
            "RTX 50-series/Blackwell GPU detected; faster-whisper changed "
            "compute_type from auto to float16 to avoid unsupported cuBLAS execution."
        )
        warning = (
            "RTX 50-series/Blackwell GPU detected; forcing faster-whisper "
            "compute_type=float16 because automatic selection can fail in cuBLAS."
        )
    return {
        "architecture": architecture,
        "affected": affected,
        "requested_compute_type": requested_compute_type,
        "compute_type": compute_type,
        "changed": compute_type != requested_compute_type,
        "reason": reason,
        "warning": warning,
    }


def _annotate_gpu_device(device: dict) -> dict:
    annotated = dict(device)
    recommendation = faster_whisper_compute_recommendation(annotated)
    annotated["architecture"] = recommendation["architecture"]
    annotated["faster_whisper"] = recommendation
    return annotated


def list_gpu_devices() -> list[dict]:
    """Return all visible NVIDIA CUDA adapters with stable integer indexes."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=index,name,memory.total,memory.free,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0:
                devices = []
                for line in (result.stdout or "").splitlines():
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) < 5:
                        continue
                    try:
                        index = int(parts[0])
                    except (TypeError, ValueError):
                        continue
                    # GPU names can contain commas; the final three fields are
                    # always memory total, memory free, and driver version.
                    name = ",".join(parts[1:-3]).strip()
                    devices.append(_annotate_gpu_device({
                        "index": index,
                        "name": name,
                        "memory_total_mb": _safe_float(parts[-3]),
                        "memory_free_mb": _safe_float(parts[-2]),
                        "driver_version": parts[-1],
                    }))
                if devices:
                    return sorted(devices, key=lambda device: device["index"])
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.debug("nvidia-smi GPU query failed: %s", exc)

    if not _HAS_TORCH:
        return []
    try:
        if not torch.cuda.is_available():
            return []
        devices = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            total = float(getattr(props, "total_memory", getattr(props, "total_mem", 0))) / (1024 * 1024)
            capability = None
            try:
                capability = torch.cuda.get_device_capability(index)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                major = getattr(props, "major", None)
                minor = getattr(props, "minor", None)
                if major is not None:
                    capability = (major, minor or 0)
            devices.append(_annotate_gpu_device({
                "index": index,
                "name": str(getattr(props, "name", "CUDA device")),
                "memory_total_mb": round(total, 1),
                "memory_free_mb": 0.0,
                "driver_version": "",
                "compute_capability": capability,
            }))
        return devices
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("PyTorch GPU query failed: %s", exc)
        return []


def gpu_selection_status(devices: list[dict] | None = None) -> dict:
    """Return a JSON-safe description of configured and active GPU state."""
    devices = list_gpu_devices() if devices is None else [_annotate_gpu_device(device) for device in devices]
    configured = _configured_gpu_index()
    available_indexes = {int(device.get("index")) for device in devices if "index" in device}
    error = None
    if configured is not None and configured not in available_indexes:
        error = GPUSelectionError(configured, devices).to_dict()
    selected = (
        configured
        if configured in available_indexes
        else (min(available_indexes) if available_indexes and error is None else None)
    )
    selected_device = next(
        (device for device in devices if device.get("index") == selected),
        None,
    )
    return {
        "configured_index": configured,
        "selected_index": selected,
        "selection_mode": "manual" if configured is not None else "auto",
        "device": f"cuda:{selected}" if selected is not None else "cpu",
        "devices": devices,
        "selection_error": error,
        "faster_whisper": faster_whisper_compute_recommendation(selected_device),
    }


def validate_gpu_index(value, devices: list[dict] | None = None):
    """Validate an API-provided index and return its normalized value."""
    devices = list_gpu_devices() if devices is None else list(devices)
    try:
        index = _normalise_gpu_index(value)
    except ValueError:
        raise GPUSelectionError(value, devices) from None
    if index is not None and index not in {
        int(device.get("index")) for device in devices if "index" in device
    }:
        raise GPUSelectionError(index, devices)
    return index


def save_gpu_selection(value, devices: list[dict] | None = None) -> dict:
    """Validate and persist a user selection, returning the new status."""
    devices = list_gpu_devices() if devices is None else list(devices)
    index = validate_gpu_index(value, devices)
    from opencut.user_data import load_gpu_settings, save_gpu_settings

    settings = load_gpu_settings()
    settings["gpu_index"] = index
    save_gpu_settings(settings)
    # Switch the request thread immediately; future worker threads activate
    # the same persisted selection at their boundary.
    activate_selected_gpu(devices=devices)
    return gpu_selection_status(devices)


def activate_selected_gpu(*, torch_module=None, devices: list[dict] | None = None):
    """Make the configured adapter current for the calling worker thread."""
    module = torch_module if torch_module is not None else torch
    configured = _configured_gpu_index()
    if module is None:
        if configured is not None:
            raise GPUSelectionError(configured, devices or list_gpu_devices())
        return None
    try:
        available = bool(module.cuda.is_available())
    except Exception as exc:
        logger.warning("CUDA availability check failed: %s", exc)
        return None
    if not available:
        if configured is not None:
            raise GPUSelectionError(configured, devices or list_gpu_devices())
        return None

    devices = list_gpu_devices() if devices is None else list(devices)
    indexes = {int(device.get("index")) for device in devices if "index" in device}
    index = configured if configured is not None else 0
    if index not in indexes:
        # A mocked/limited torch runtime may not have nvidia-smi metadata;
        # use its runtime count as the authoritative fallback.
        try:
            count = int(module.cuda.device_count())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            count = 0
        if index < 0 or index >= count:
            raise GPUSelectionError(index, devices)
    try:
        module.cuda.set_device(index)
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise GPUSelectionError(index, devices) from exc
    return index


def get_device_index() -> int | None:
    """Return the effective CUDA index, or ``None`` when CPU is active."""
    status = gpu_selection_status()
    return status.get("selected_index")


def selected_ct2_device_kwargs() -> dict:
    """Return CTranslate2 constructor kwargs for the selected adapter."""
    status = gpu_selection_status()
    if status.get("selection_error"):
        error = status["selection_error"]
        raise GPUSelectionError(error.get("requested_index"), status.get("devices", []))
    index = status.get("selected_index")
    if index is None:
        return {"device": "auto"}
    return {"device": "cuda", "device_index": index}


def selected_onnx_providers() -> list:
    """Return ONNX Runtime providers pinned to the selected CUDA index."""
    status = gpu_selection_status()
    if status.get("selection_error"):
        error = status["selection_error"]
        raise GPUSelectionError(error.get("requested_index"), status.get("devices", []))
    index = status.get("selected_index")
    if index is None:
        # Keep ONNX Runtime's provider probing behaviour when no CUDA device
        # metadata is available (for example, a vendor runtime without torch).
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return [
        ("CUDAExecutionProvider", {"device_id": index}),
        "CPUExecutionProvider",
    ]

# ---------------------------------------------------------------------------
# Optional torch import
# ---------------------------------------------------------------------------
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------

def _cuda_available() -> bool:
    """Wrap ``torch.cuda.is_available()`` so a partially-broken CUDA driver
    (mismatched runtime, kernel mode driver crash, blacklisted device) can't
    raise a hard exception and take down the request handler. We only need
    a boolean answer; treat every error as "no GPU"."""
    if not _HAS_TORCH:
        return False
    try:
        if not torch.cuda.is_available():
            return False
        activate_selected_gpu(torch_module=torch)
        return True
    except GPUSelectionError:
        raise
    except Exception as exc:
        logger.warning("torch.cuda.is_available() raised %s — falling back to CPU", exc)
        return False


def get_device() -> str:
    """Return 'cuda' if a CUDA GPU is available, else 'cpu'."""
    if _cuda_available():
        return "cuda"
    return "cpu"


def check_vram(min_gb: float = 0) -> tuple:
    """
    Check available and total GPU VRAM.

    Args:
        min_gb: Minimum free VRAM in GB to consider sufficient.
                Logged as a warning if not met.

    Returns:
        (available_gb, total_gb) tuple.  Returns (0, 0) when no
        CUDA GPU is detected or torch is unavailable.

    A broken NVIDIA driver can make ``torch.cuda.mem_get_info`` hang
    indefinitely. We run it on a *daemon* thread and abandon it on
    timeout, so the request handler returns in bounded time even if the
    driver is wedged. A ThreadPoolExecutor context manager would block
    on exit waiting for the hung task to finish, which is why we manage
    the thread manually.
    """
    if not _cuda_available():
        return (0.0, 0.0)

    result = {}

    def _query():
        try:
            free, total = torch.cuda.mem_get_info()
            result["free"] = free
            result["total"] = total
        except Exception as exc:  # noqa: BLE001
            result["error"] = exc

    import threading

    t = threading.Thread(target=_query, daemon=True, name="vram-query")
    t.start()
    t.join(timeout=5)

    if t.is_alive():
        # Abandon the thread; it's a daemon so it dies with the process.
        logger.warning("VRAM query timed out (>5s) — NVIDIA driver may be hung")
        return (0.0, 0.0)
    if "error" in result:
        logger.warning("Failed to query VRAM: %s", result["error"])
        return (0.0, 0.0)
    if "free" not in result:
        return (0.0, 0.0)

    available_gb = result["free"] / (1024 ** 3)
    total_gb = result["total"] / (1024 ** 3)

    if min_gb > 0 and available_gb < min_gb:
        logger.warning(
            "Low VRAM: %.2f GB free, %.2f GB required",
            available_gb,
            min_gb,
        )

    return (round(available_gb, 2), round(total_gb, 2))


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

class GPUContext:
    """Context manager for GPU operations.  Auto-cleans VRAM on exit.

    Usage::

        with GPUContext(min_vram_gb=2) as ctx:
            model = load_model().to(ctx.device)
            ctx.register(model)
            result = model(input_tensor)
        # model is deleted and VRAM is freed here

    If *min_vram_gb* is set and insufficient VRAM is available, the
    context still enters successfully but logs a warning -- callers
    can check ``ctx.available_gb`` to decide whether to fall back to
    CPU.
    """

    def __init__(self, min_vram_gb: float = 0, device: str = "cuda"):
        self._requested_device = device
        self._min_vram_gb = min_vram_gb
        self._models = []
        self.available_gb = 0.0
        self.total_gb = 0.0

        # Resolve actual device
        if device == "cuda" and get_device() == "cpu":
            self._device_str = "cpu"
        else:
            self._device_str = device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def device(self):
        """Return a ``torch.device`` if torch is available, else the string."""
        if _HAS_TORCH:
            return torch.device(self._device_str)
        return self._device_str

    def register(self, *models):
        """Track one or more models/tensors for cleanup on exit."""
        self._models.extend(models)

    # ------------------------------------------------------------------
    # Context protocol
    # ------------------------------------------------------------------

    def __enter__(self):
        if self._device_str == "cuda":
            activate_selected_gpu(torch_module=torch)
            self.available_gb, self.total_gb = check_vram(self._min_vram_gb)
            logger.info(
                "GPUContext entered: %.2f / %.2f GB VRAM free",
                self.available_gb,
                self.total_gb,
            )
        else:
            logger.info("GPUContext entered on CPU (no CUDA available)")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Move registered models to CPU and release references
        for obj in self._models:
            try:
                if hasattr(obj, "cpu"):
                    obj.cpu()
            except Exception:
                pass
        self._models.clear()

        # Flush CUDA cache
        if _cuda_available():
            try:
                torch.cuda.empty_cache()
                logger.debug("GPUContext: CUDA cache cleared")
            except Exception as exc:
                logger.warning("GPUContext: failed to clear CUDA cache: %s", exc)

        # Do not suppress exceptions
        return False
