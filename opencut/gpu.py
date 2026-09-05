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

    @property
    def suggestion(self) -> str:
        return "Choose one of the listed device indexes or select Auto."

    def to_dict(self) -> dict:
        return {
            "error": str(self),
            "code": self.code,
            "requested_index": self.requested,
            "available_devices": self.available_devices,
            "suggestion": self.suggestion,
        }


class GPUUnsupportedBuildError(GPUSelectionError):
    """The adapter exists, but the installed runtime cannot execute on it.

    Reported separately because the two failures need opposite responses.
    "Index 0 is not available" told the reporter of issue #7 to pick a
    different device while listing index 0 as available -- the adapter was
    fine, the installed PyTorch build simply carried no kernel image for it.
    """

    code = "GPU_BUILD_UNSUPPORTED"

    def __init__(self, requested, devices, *, reason: str = "", required_build: str = "", cause=None):
        # Deliberately skip GPUSelectionError.__init__: its message asserts an
        # availability claim that is false here.
        ValueError.__init__(self)
        self.requested = requested
        self.available_devices = [dict(device) for device in (devices or [])]
        self.reason = reason or "the installed runtime has no kernel image for this adapter"
        self.required_build = required_build
        self.cause = str(cause) if cause is not None else ""
        device = next(
            (d for d in self.available_devices if d.get("index") == requested),
            None,
        )
        name = (device or {}).get("name") or "the selected CUDA device"
        capability = _capability_label((device or {}).get("compute_capability"))
        detail = f" (compute capability {capability})" if capability else ""
        message = (
            f"GPU index {requested!r} ({name}{detail}) is present but this build cannot run on it: "
            f"{self.reason}."
        )
        if self.required_build:
            message += f" {self.required_build}"
        if self.cause:
            message += f" Underlying error: {self.cause}"
        self.args = (message,)

    @property
    def suggestion(self) -> str:
        if self.required_build:
            return self.required_build
        return "Install a runtime build that supports this adapter, or select CPU."

    def to_dict(self) -> dict:
        body = super().to_dict()
        body.update({
            "reason": self.reason,
            "required_build": self.required_build,
            "underlying_error": self.cause,
        })
        return body


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


def _parse_capability(value) -> tuple[int, int] | None:
    """Return ``(major, minor)`` from a capability in any of its shapes."""
    if isinstance(value, (tuple, list)) and value:
        value = ".".join(str(part) for part in value[:2])
    match = re.search(r"(\d+)(?:\.(\d+))?", str(value or ""))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2) or 0)


def _capability_label(value) -> str:
    parsed = _parse_capability(value)
    return f"{parsed[0]}.{parsed[1]}" if parsed else ""


def torch_supported_capabilities(torch_module=None) -> set[tuple[int, int]]:
    """Return the compute capabilities this PyTorch build carries kernels for.

    ``torch.cuda.get_arch_list()`` reports entries like ``sm_90`` (a compiled
    kernel image) and ``compute_90`` (PTX that can JIT forward). Only the
    ``sm_`` entries are counted: PTX JIT is what fails on Blackwell in the
    field, so treating it as support would restore the bug this replaces.
    """
    module = torch_module if torch_module is not None else torch
    if module is None:
        return set()
    try:
        arches = list(module.cuda.get_arch_list())
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("torch.cuda.get_arch_list() unavailable: %s", exc)
        return set()
    supported: set[tuple[int, int]] = set()
    unparsed: list[str] = []
    for arch in arches:
        # The minor version is always the final digit: sm_86 is 8.6, sm_90 is
        # 9.0, and sm_120 is 12.0 (not 1.20 -- getting this backwards makes a
        # cu128 build look like it cannot run the Blackwell card it was built
        # for). The first group is greedy so it absorbs the extra digit.
        #
        # The optional suffix matters: PyTorch emits architecture-conditional
        # entries like sm_90a and sm_120a for Hopper and Blackwell. Dropping
        # those silently would leave a partially parsed set that still looks
        # authoritative, and a working H100 or RTX 5090 would be graded
        # unsupported -- worse than the bug this whole check exists to fix.
        text = str(arch).strip()
        match = re.fullmatch(r"sm_(\d+)(\d)([a-z]*)", text)
        if match:
            supported.add((int(match.group(1)), int(match.group(2))))
        elif not re.fullmatch(r"compute_\d+[a-z]*", text):
            unparsed.append(text)
    if unparsed:
        # Refuse to grade against a list we only half understand.
        logger.warning(
            "Unrecognised CUDA architecture entries %s; skipping build-support grading.",
            ", ".join(sorted(unparsed)),
        )
        return set()
    return supported


# ---------------------------------------------------------------------------
# The documented GPU install lane
#
# One source for the CUDA wheel index, because it was previously restated in
# README.md, requirements.txt, install.py and upscale_flashvsr.py, all naming
# cu121 -- an index with no sm_120 kernels. Every RTX 50-series user who
# followed the documentation landed in the failure reported as issue #7.
# ---------------------------------------------------------------------------

#: Highest compute capability each PyTorch CUDA wheel index carries kernels for.
CUDA_INDEX_MAX_CAPABILITY = {
    "cu118": (9, 0),
    "cu121": (9, 0),
    "cu124": (9, 0),
    "cu126": (9, 0),
    "cu128": (12, 0),
    "cu129": (12, 0),
}

#: The index OpenCut documents. Must cover NEWEST_CONSUMER_CAPABILITY.
CUDA_WHEEL_INDEX = "cu128"

#: The newest NVIDIA consumer architecture OpenCut claims to support.
#: RTX 50-series (Blackwell) is sm_120.
NEWEST_CONSUMER_CAPABILITY = (12, 0)

CUDA_WHEEL_INDEX_URL = f"https://download.pytorch.org/whl/{CUDA_WHEEL_INDEX}"

#: The exact command the documentation tells users to run.
TORCH_GPU_INSTALL_COMMAND = (
    f'pip install "torch>=2.10" "torchvision>=0.25" torchaudio --index-url {CUDA_WHEEL_INDEX_URL}'
)


def supported_capability_range() -> tuple[tuple[int, int], tuple[int, int]]:
    """Return ``(oldest, newest)`` compute capability the documented lane covers."""
    return (7, 5), CUDA_INDEX_MAX_CAPABILITY[CUDA_WHEEL_INDEX]


#: A build that cannot run an adapter needs a different wheel, not a different
#: device. Keyed by capability major so the advice names a real download.
_REQUIRED_BUILD_HINTS = {
    12: (
        f"RTX 50-series/Blackwell (sm_120) needs a CUDA 12.8 or newer PyTorch build: "
        f"pip install torch --index-url {CUDA_WHEEL_INDEX_URL}"
    ),
    10: (
        f"This adapter needs a CUDA 12.8 or newer PyTorch build: "
        f"pip install torch --index-url {CUDA_WHEEL_INDEX_URL}"
    ),
}

GPU_SUPPORT_USABLE = "usable"
GPU_SUPPORT_UNSUPPORTED = "unsupported-build"
GPU_SUPPORT_UNKNOWN = "unknown"


def gpu_runtime_support(device: dict | None, *, torch_module=None) -> dict:
    """Resolve whether the installed runtime can actually execute on ``device``.

    ``nvidia-smi`` reports every adapter that is physically present, which is a
    different question from whether the installed PyTorch build has kernels for
    it. Conflating the two is why Settings showed an RTX 5070 as healthy while
    every job failed (issue #7).
    """
    device = device if isinstance(device, dict) else {}
    capability = _parse_capability(device.get("compute_capability", device.get("compute_cap")))
    supported = torch_supported_capabilities(torch_module)
    if capability is None or not supported:
        return {
            "state": GPU_SUPPORT_UNKNOWN,
            "compute_capability": _capability_label(capability),
            "supported_capabilities": sorted(f"{a}.{b}" for a, b in supported),
            "reason": (
                "" if supported else
                "the installed runtime does not report a compiled architecture list"
            ),
            "required_build": "",
        }
    if capability in supported:
        return {
            "state": GPU_SUPPORT_USABLE,
            "compute_capability": _capability_label(capability),
            "supported_capabilities": sorted(f"{a}.{b}" for a, b in supported),
            "reason": "",
            "required_build": "",
        }
    label = _capability_label(capability)
    highest = max(supported)
    return {
        "state": GPU_SUPPORT_UNSUPPORTED,
        "compute_capability": label,
        "supported_capabilities": sorted(f"{a}.{b}" for a, b in supported),
        "reason": (
            f"the installed PyTorch build has no kernel image for compute capability {label} "
            f"(it was built for up to {highest[0]}.{highest[1]})"
        ),
        "required_build": _REQUIRED_BUILD_HINTS.get(
            capability[0],
            "Install a PyTorch build compiled for this adapter's compute capability, or select CPU.",
        ),
    }


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
    support = gpu_runtime_support(annotated)
    annotated["runtime_support"] = support
    annotated["usable"] = support["state"] != GPU_SUPPORT_UNSUPPORTED
    return annotated


def list_gpu_devices() -> list[dict]:
    """Return all visible NVIDIA CUDA adapters with stable integer indexes."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        # compute_cap is what tells us whether the installed torch build can
        # execute on an adapter; without it every device looked equally fine
        # (issue #7). Drivers older than 510 do not know the field, so fall
        # back to the original query rather than losing the listing entirely.
        for fields, has_capability in (
            ("index,name,memory.total,memory.free,driver_version,compute_cap", True),
            ("index,name,memory.total,memory.free,driver_version", False),
        ):
            try:
                result = subprocess.run(
                    [nvidia_smi, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                logger.debug("nvidia-smi GPU query failed: %s", exc)
                break
            if result.returncode != 0:
                continue
            trailing = 4 if has_capability else 3
            devices = []
            for line in (result.stdout or "").splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) < trailing + 2:
                    continue
                try:
                    index = int(parts[0])
                except (TypeError, ValueError):
                    continue
                # GPU names can contain commas; the trailing fields are fixed.
                record = {
                    "index": index,
                    "name": ",".join(parts[1:-trailing]).strip(),
                    "memory_total_mb": _safe_float(parts[-trailing]),
                    "memory_free_mb": _safe_float(parts[-trailing + 1]),
                    "driver_version": parts[-trailing + 2],
                }
                if has_capability:
                    record["compute_capability"] = parts[-1]
                devices.append(_annotate_gpu_device(record))
            if devices:
                return sorted(devices, key=lambda device: device["index"])

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
    def _usable(index: int) -> bool:
        device = next((d for d in devices if d.get("index") == index), None)
        return gpu_runtime_support(device)["state"] != GPU_SUPPORT_UNSUPPORTED

    if configured in available_indexes:
        selected = configured
    elif available_indexes and error is None:
        # Automatic selection must skip adapters this build cannot execute on.
        # Picking min() unconditionally meant a machine with an unsupported
        # card at index 0 and a working one at index 1 failed every job.
        runnable = sorted(index for index in available_indexes if _usable(index))
        selected = runnable[0] if runnable else min(available_indexes)
    else:
        selected = None

    selected_device = next(
        (device for device in devices if device.get("index") == selected),
        None,
    )
    # Surface an unrunnable adapter here, where Settings reads it, rather than
    # letting the first job discover it (issue #7).
    support = gpu_runtime_support(selected_device)
    if error is None and support["state"] == GPU_SUPPORT_UNSUPPORTED:
        error = GPUUnsupportedBuildError(
            selected,
            devices,
            reason=support["reason"],
            required_build=support["required_build"],
        ).to_dict()
        # An adapter this build cannot run is not a selection. Leaving it set
        # made get_device_index() hand "cuda:0" to callers that never look at
        # selection_error, so the job failed deep inside a model load instead.
        selected = None
        selected_device = None
    return {
        "configured_index": configured,
        "selected_index": selected,
        "selection_mode": "manual" if configured is not None else "auto",
        "device": f"cuda:{selected}" if selected is not None else "cpu",
        "devices": devices,
        "selection_error": error,
        "runtime_support": support,
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
    else:
        # The adapter is present. Refuse before touching CUDA when the build
        # has no kernels for it, so the message names the real problem rather
        # than denying an index it is simultaneously listing (issue #7).
        device = next((d for d in devices if d.get("index") == index), None)
        support = gpu_runtime_support(device, torch_module=module)
        if support["state"] == GPU_SUPPORT_UNSUPPORTED:
            raise GPUUnsupportedBuildError(
                index,
                devices,
                reason=support["reason"],
                required_build=support["required_build"],
            )
    try:
        module.cuda.set_device(index)
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        # set_device failed on an adapter we listed as present. That is a build
        # or driver problem, never "pick a different index".
        device = next((d for d in devices if d.get("index") == index), None)
        if device is not None:
            support = gpu_runtime_support(device, torch_module=module)
            raise GPUUnsupportedBuildError(
                index,
                devices,
                reason=support["reason"] or "CUDA refused to select this adapter",
                required_build=support["required_build"],
                cause=exc,
            ) from exc
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
    if not onnx_cuda_provider_available():
        # The CPU-only onnxruntime wheel can never satisfy CUDAExecutionProvider.
        # Asking for it anyway produced a warning and a silent CPU fallback that
        # looked like the GPU selection had been ignored (issue #7).
        return ["CPUExecutionProvider"]
    return [
        ("CUDAExecutionProvider", {"device_id": index}),
        "CPUExecutionProvider",
    ]


def onnx_cuda_provider_available() -> bool:
    """Return True when the installed onnxruntime build offers CUDA."""
    try:
        import onnxruntime
    except ImportError:
        return False
    try:
        return "CUDAExecutionProvider" in set(onnxruntime.get_available_providers())
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("onnxruntime provider probe failed: %s", exc)
        return False

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
