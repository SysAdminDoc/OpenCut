"""
OpenCut Apple Silicon / MPS Acceleration (Feature 10.4)

Detects Apple Silicon hardware, routes compatible operations to the MPS
(Metal Performance Shaders) device, and falls back to CPU for unsupported
operations.
"""

import logging
import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Optional

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Operation compatibility table
# ---------------------------------------------------------------------------
# Operations known to work reliably on MPS as of PyTorch 2.x
MPS_COMPATIBLE_OPS = frozenset({
    "inference",
    "transcription",
    "upscaling",
    "style_transfer",
    "face_detection",
    "object_detection",
    "image_classification",
    "embedding",
    "vad",  # voice activity detection
    "caption_generation",
    "scene_detection",
    "depth_estimation",
    "denoising",
})

# Operations that have known issues on MPS and should use CPU
MPS_INCOMPATIBLE_OPS = frozenset({
    "quantization",          # dynamic quantization not supported on MPS
    "int8_inference",        # INT8 ops limited on Metal
    "int4_inference",        # INT4 not supported
    "complex_conv3d",        # Conv3d has MPS gaps
    "spectral_ops",          # FFT operations incomplete
    "sparse_ops",            # Sparse tensor ops unsupported
    "custom_cuda_kernels",   # CUDA-only kernels
    "nms",                   # Non-max suppression (partial support)
    "grid_sample",           # grid_sample backward not supported
    "fractional_max_pool",   # Not implemented on MPS
})


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class AppleSiliconInfo:
    """Information about the Apple Silicon hardware."""
    is_apple_silicon: bool = False
    chip_name: str = ""
    chip_family: str = ""  # M1, M2, M3, M4, etc.
    cpu_cores: int = 0
    gpu_cores: int = 0
    neural_engine_cores: int = 0
    memory_gb: float = 0.0
    macos_version: str = ""
    mps_available: bool = False
    mps_built: bool = False
    torch_version: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DeviceRecommendation:
    """Recommended device for a given operation."""
    operation: str = ""
    recommended_device: str = "cpu"
    reason: str = ""
    mps_compatible: bool = False
    fallback_device: str = "cpu"

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------
def _get_chip_info_sysctl() -> Dict[str, str]:
    """Query Apple chip info via sysctl on macOS."""
    info = {}
    if platform.system() != "Darwin":
        return info
    queries = {
        "chip_name": "machdep.cpu.brand_string",
        "cpu_cores": "hw.ncpu",
        "memory": "hw.memsize",
    }
    for key, sysctl_key in queries.items():
        try:
            result = subprocess.run(
                ["sysctl", "-n", sysctl_key],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                info[key] = result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    return info


def _parse_chip_family(brand_string: str) -> str:
    """Extract chip family (M1, M2, M3, M4) from brand string."""
    brand = brand_string.lower()
    for family in ("m4", "m3", "m2", "m1"):
        if family in brand:
            return family.upper()
    if "apple" in brand:
        return "Apple Silicon (unknown)"
    return ""


def _get_gpu_core_count() -> int:
    """Attempt to detect GPU core count on Apple Silicon."""
    if platform.system() != "Darwin":
        return 0
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-detailLevel", "basic"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                line_lower = line.strip().lower()
                if "total number of cores" in line_lower or "cores" in line_lower:
                    parts = line.split(":")
                    if len(parts) == 2:
                        try:
                            return int(parts[1].strip())
                        except ValueError:
                            pass
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return 0


def _get_neural_engine_cores(chip_family: str) -> int:
    """Return the number of Neural Engine cores for known chip families."""
    ne_cores = {
        "M1": 16,
        "M2": 16,
        "M3": 16,
        "M4": 16,
    }
    return ne_cores.get(chip_family, 0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def detect_apple_silicon(on_progress: Optional[Callable] = None) -> AppleSiliconInfo:
    """Detect Apple Silicon hardware and MPS availability.

    Args:
        on_progress: Optional progress callback.

    Returns:
        AppleSiliconInfo with hardware and software details.
    """
    info = AppleSiliconInfo()

    if on_progress:
        on_progress({"step": "detecting", "message": "Checking hardware..."})

    # Basic platform check
    is_mac = platform.system() == "Darwin"
    is_arm = platform.machine().lower() in ("arm64", "aarch64")
    info.is_apple_silicon = is_mac and is_arm

    if is_mac:
        info.macos_version = platform.mac_ver()[0]

    # Get chip details
    sysctl = _get_chip_info_sysctl()
    chip_brand = sysctl.get("chip_name", "")
    info.chip_name = chip_brand

    if info.is_apple_silicon:
        info.chip_family = _parse_chip_family(chip_brand)
        info.neural_engine_cores = _get_neural_engine_cores(info.chip_family)

    try:
        info.cpu_cores = int(sysctl.get("cpu_cores", "0"))
    except ValueError:
        info.cpu_cores = os.cpu_count() or 0

    try:
        mem_bytes = int(sysctl.get("memory", "0"))
        info.memory_gb = round(mem_bytes / (1024 ** 3), 1)
    except ValueError:
        info.memory_gb = 0.0

    info.gpu_cores = _get_gpu_core_count()

    # Check PyTorch MPS support
    if on_progress:
        on_progress({"step": "checking_mps", "message": "Checking MPS availability..."})

    try:
        import torch
        info.torch_version = torch.__version__
        info.mps_built = hasattr(torch.backends, "mps") and torch.backends.mps.is_built()
        info.mps_available = (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )
    except ImportError:
        info.torch_version = ""
        info.mps_built = False
        info.mps_available = False

    return info


def get_mps_device() -> Optional[object]:
    """Return a ``torch.device('mps')`` if MPS is available, else None.

    Returns:
        ``torch.device('mps')`` or ``None``.
    """
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    except ImportError:
        pass
    return None


def is_op_mps_compatible(operation: str) -> bool:
    """Check whether an operation is compatible with MPS.

    Args:
        operation: Operation name (e.g. ``"inference"``, ``"quantization"``).

    Returns:
        True if the operation is known to work on MPS.
    """
    op = operation.lower().strip()
    if op in MPS_INCOMPATIBLE_OPS:
        return False
    if op in MPS_COMPATIBLE_OPS:
        return True
    # Unknown operations: assume compatible but warn
    logger.debug("Operation %r not in MPS compatibility table; assuming compatible.", op)
    return True


def get_recommended_device(
    operation: str,
    on_progress: Optional[Callable] = None,
) -> DeviceRecommendation:
    """Recommend the best device (MPS vs CPU) for a given operation.

    Args:
        operation: The operation to run.
        on_progress: Optional progress callback.

    Returns:
        DeviceRecommendation with the recommended device and reasoning.
    """
    op = operation.lower().strip()
    rec = DeviceRecommendation(operation=op)

    if on_progress:
        on_progress({"step": "evaluating", "operation": op})

    # Check MPS availability
    mps_device = get_mps_device()
    has_mps = mps_device is not None

    if not has_mps:
        rec.recommended_device = "cpu"
        rec.reason = "MPS not available (not Apple Silicon or PyTorch < 1.12)."
        rec.mps_compatible = False
        rec.fallback_device = "cpu"
        return rec

    # Check CUDA availability (might be running on non-Apple GPU)
    has_cuda = False
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        pass

    if has_cuda:
        rec.recommended_device = "cuda"
        rec.reason = "CUDA GPU available; preferred over MPS for this operation."
        rec.mps_compatible = is_op_mps_compatible(op)
        rec.fallback_device = "mps" if rec.mps_compatible else "cpu"
        return rec

    # MPS path
    compatible = is_op_mps_compatible(op)
    rec.mps_compatible = compatible

    if compatible:
        rec.recommended_device = "mps"
        rec.reason = f"Operation '{op}' is MPS-compatible; using Apple GPU for acceleration."
        rec.fallback_device = "cpu"
    else:
        rec.recommended_device = "cpu"
        rec.reason = f"Operation '{op}' is not MPS-compatible; falling back to CPU."
        rec.fallback_device = "cpu"

    return rec
