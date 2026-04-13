"""
AMD GPU Support.

Detect AMD GPUs, check DirectML and ROCm availability, and provide
GPU capability information for routing compute tasks.
"""

import logging
import os
import platform
import subprocess
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("opencut")


@dataclass
class AMDGPUInfo:
    """Information about a detected AMD GPU."""
    name: str = ""
    vendor_id: str = ""
    device_id: str = ""
    driver_version: str = ""
    vram_mb: int = 0
    supports_directml: bool = False
    supports_rocm: bool = False
    compute_units: int = 0
    architecture: str = ""


def detect_amd_gpu(
    on_progress: Optional[Callable] = None,
) -> List[AMDGPUInfo]:
    """Detect AMD GPUs on the system.

    Returns:
        List of AMDGPUInfo for each detected AMD GPU.
    """
    if on_progress:
        on_progress(10, "Scanning for AMD GPUs")

    gpus: List[AMDGPUInfo] = []

    system = platform.system()
    if system == "Windows":
        gpus = _detect_amd_windows()
    elif system == "Linux":
        gpus = _detect_amd_linux()
    else:
        logger.info("AMD GPU detection not supported on %s", system)

    if on_progress:
        on_progress(60, f"Found {len(gpus)} AMD GPU(s)")

    # Check DirectML/ROCm support for each
    dml_available = _check_directml()
    rocm_available = check_rocm_available()

    for gpu in gpus:
        gpu.supports_directml = dml_available
        gpu.supports_rocm = rocm_available

    if on_progress:
        on_progress(100, "Detection complete")

    return gpus


def _detect_amd_windows() -> List[AMDGPUInfo]:
    """Detect AMD GPUs on Windows via WMI/PowerShell."""
    gpus = []
    try:
        cmd = [
            "powershell", "-NoProfile", "-Command",
            "Get-CimInstance Win32_VideoController | "
            "Where-Object { $_.AdapterCompatibility -like '*AMD*' -or "
            "$_.Name -like '*Radeon*' } | "
            "Select-Object Name, DriverVersion, AdapterRAM, "
            "PNPDeviceID | ConvertTo-Json"
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            import json
            data = json.loads(result.stdout)
            if isinstance(data, dict):
                data = [data]
            for entry in data:
                vram = int(entry.get("AdapterRAM", 0) or 0) // (1024 * 1024)
                pnp = entry.get("PNPDeviceID", "")
                vid, did = "", ""
                if "VEN_" in pnp:
                    vid = pnp.split("VEN_")[1][:4]
                if "DEV_" in pnp:
                    did = pnp.split("DEV_")[1][:4]
                gpus.append(AMDGPUInfo(
                    name=entry.get("Name", "AMD GPU"),
                    driver_version=entry.get("DriverVersion", ""),
                    vram_mb=vram,
                    vendor_id=vid,
                    device_id=did,
                    architecture=_guess_architecture(entry.get("Name", "")),
                ))
    except Exception as exc:
        logger.debug("Windows AMD detection failed: %s", exc)
    return gpus


def _detect_amd_linux() -> List[AMDGPUInfo]:
    """Detect AMD GPUs on Linux via /sys or lspci."""
    gpus = []
    try:
        result = subprocess.run(
            ["lspci", "-d", "1002:", "-v"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for block in result.stdout.split("\n\n"):
                if "VGA" in block or "Display" in block or "3D" in block:
                    name = block.split("\n")[0].split(": ", 1)[-1] if ": " in block else "AMD GPU"
                    gpus.append(AMDGPUInfo(
                        name=name.strip(),
                        vendor_id="1002",
                        architecture=_guess_architecture(name),
                    ))
    except Exception as exc:
        logger.debug("Linux AMD detection failed: %s", exc)
    return gpus


def _guess_architecture(name: str) -> str:
    """Guess AMD GPU architecture from name."""
    name_lower = name.lower()
    if "rdna3" in name_lower or "7900" in name_lower or "7800" in name_lower or "7700" in name_lower:
        return "RDNA3"
    elif "rdna2" in name_lower or "6900" in name_lower or "6800" in name_lower or "6700" in name_lower:
        return "RDNA2"
    elif "rdna" in name_lower or "5700" in name_lower or "5600" in name_lower:
        return "RDNA"
    elif "vega" in name_lower:
        return "Vega"
    elif "polaris" in name_lower or "rx 5" in name_lower or "rx 4" in name_lower:
        return "Polaris"
    return ""


def _check_directml() -> bool:
    """Check if DirectML is available."""
    if platform.system() != "Windows":
        return False
    try:
        import onnxruntime as ort
        return "DmlExecutionProvider" in ort.get_available_providers()
    except ImportError:
        pass
    # Check if torch-directml is available
    try:
        import torch_directml  # noqa: F401
        return True
    except ImportError:
        pass
    return False


def get_directml_device(
    on_progress: Optional[Callable] = None,
) -> Dict:
    """Get the DirectML device for PyTorch/ONNX use.

    Returns:
        Dict with device info and provider details.
    """
    if platform.system() != "Windows":
        return {"available": False, "reason": "DirectML requires Windows"}

    if on_progress:
        on_progress(30, "Checking DirectML")

    result = {
        "available": False,
        "device": None,
        "provider": "DmlExecutionProvider",
        "onnx_available": False,
        "torch_available": False,
    }

    try:
        import onnxruntime as ort
        if "DmlExecutionProvider" in ort.get_available_providers():
            result["onnx_available"] = True
            result["available"] = True
    except ImportError:
        pass

    try:
        import torch_directml
        result["device"] = str(torch_directml.device())
        result["torch_available"] = True
        result["available"] = True
    except ImportError:
        pass

    if on_progress:
        on_progress(100, "DirectML check complete")

    return result


def check_rocm_available(
    on_progress: Optional[Callable] = None,
) -> bool:
    """Check if AMD ROCm is available on the system.

    Returns:
        True if ROCm is available and functional.
    """
    if on_progress:
        on_progress(30, "Checking ROCm")

    # Check via environment
    if os.environ.get("ROCM_PATH") or os.path.isdir("/opt/rocm"):
        try:
            result = subprocess.run(
                ["rocm-smi", "--showid"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                if on_progress:
                    on_progress(100, "ROCm detected")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # Check via PyTorch
    try:
        import torch
        if hasattr(torch, "hip") or (torch.cuda.is_available() and "rocm" in str(torch.version.hip or "")):
            if on_progress:
                on_progress(100, "ROCm via PyTorch")
            return True
    except (ImportError, AttributeError):
        pass

    # Check via ONNX Runtime
    try:
        import onnxruntime as ort
        if "ROCMExecutionProvider" in ort.get_available_providers():
            if on_progress:
                on_progress(100, "ROCm via ONNX Runtime")
            return True
    except ImportError:
        pass

    if on_progress:
        on_progress(100, "ROCm not found")
    return False


def get_amd_capabilities(
    on_progress: Optional[Callable] = None,
) -> Dict:
    """Get comprehensive AMD GPU capabilities summary.

    Returns:
        Dict with GPU info, DirectML status, ROCm status, and recommendations.
    """
    if on_progress:
        on_progress(10, "Gathering AMD capabilities")

    gpus = detect_amd_gpu()
    dml = get_directml_device()
    rocm = check_rocm_available()

    if on_progress:
        on_progress(70, "Compiling results")

    recommendations = []
    if gpus and not dml["available"] and not rocm:
        if platform.system() == "Windows":
            recommendations.append(
                "Install onnxruntime-directml for GPU acceleration: "
                "pip install onnxruntime-directml"
            )
        elif platform.system() == "Linux":
            recommendations.append(
                "Install ROCm for AMD GPU acceleration: "
                "https://rocm.docs.amd.com/en/latest/deploy/linux/install.html"
            )
    elif not gpus:
        recommendations.append("No AMD GPU detected on this system")

    if on_progress:
        on_progress(100, "Capabilities gathered")

    return {
        "gpus": [
            {
                "name": g.name,
                "vram_mb": g.vram_mb,
                "architecture": g.architecture,
                "driver_version": g.driver_version,
            }
            for g in gpus
        ],
        "gpu_count": len(gpus),
        "directml": dml,
        "rocm_available": rocm,
        "recommendations": recommendations,
        "best_provider": _best_provider(dml["available"], rocm),
    }


def _best_provider(dml: bool, rocm: bool) -> str:
    if rocm:
        return "ROCMExecutionProvider"
    if dml:
        return "DmlExecutionProvider"
    return "CPUExecutionProvider"
