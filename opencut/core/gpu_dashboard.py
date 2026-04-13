"""
OpenCut GPU Memory Management Dashboard (Feature 32.4)

Tracks loaded models and their VRAM usage, queries real-time VRAM
allocation, provides manual unload controls, and auto-unload
recommendations when memory is tight.
"""

import logging
import platform
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GPUInfo:
    """Information about a single GPU."""
    index: int = 0
    name: str = "Unknown"
    driver_version: str = ""
    total_vram_mb: float = 0.0
    used_vram_mb: float = 0.0
    free_vram_mb: float = 0.0
    utilization_percent: float = 0.0
    temperature_c: float = 0.0
    power_draw_w: float = 0.0
    gpu_type: str = "unknown"  # nvidia, amd, intel, apple, unknown

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LoadedModel:
    """A model currently loaded in GPU memory."""
    name: str = ""
    size_mb: float = 0.0
    device: str = "cpu"
    loaded_at: float = 0.0
    last_used: float = 0.0
    use_count: int = 0
    pinned: bool = False  # pinned models won't be auto-unloaded

    def to_dict(self) -> dict:
        d = asdict(self)
        d["idle_seconds"] = round(time.time() - self.last_used, 1) if self.last_used > 0 else 0.0
        return d


@dataclass
class VRAMStatus:
    """Current VRAM allocation summary."""
    total_vram_mb: float = 0.0
    used_vram_mb: float = 0.0
    free_vram_mb: float = 0.0
    models_loaded: int = 0
    models_vram_mb: float = 0.0
    system_overhead_mb: float = 0.0
    utilization_percent: float = 0.0
    gpu_type: str = "unknown"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class UnloadRecommendation:
    """Recommendation for which models to unload to free VRAM."""
    required_mb: float = 0.0
    current_free_mb: float = 0.0
    models_to_unload: List[str] = field(default_factory=list)
    freed_mb: float = 0.0
    sufficient: bool = False
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Model registry (in-memory tracking)
# ---------------------------------------------------------------------------
_loaded_models: Dict[str, LoadedModel] = {}
_models_lock = threading.Lock()


def register_model(name: str, size_mb: float, device: str = "cuda", pinned: bool = False) -> LoadedModel:
    """Register a model as loaded in GPU memory.

    Args:
        name: Unique model identifier.
        size_mb: Size of the model in GPU memory (MB).
        device: Device the model is on (``"cuda"``, ``"mps"``, ``"cpu"``).
        pinned: If True, model won't be auto-unloaded.

    Returns:
        The LoadedModel entry.
    """
    now = time.time()
    model = LoadedModel(
        name=name,
        size_mb=size_mb,
        device=device,
        loaded_at=now,
        last_used=now,
        use_count=0,
        pinned=pinned,
    )
    with _models_lock:
        _loaded_models[name] = model
    logger.info("Registered model '%s' (%.1f MB) on %s", name, size_mb, device)
    return model


def touch_model(name: str) -> bool:
    """Update a model's last-used timestamp and increment use count.

    Args:
        name: Model identifier.

    Returns:
        True if model was found and updated.
    """
    with _models_lock:
        model = _loaded_models.get(name)
        if model:
            model.last_used = time.time()
            model.use_count += 1
            return True
    return False


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------
def _query_nvidia_smi() -> List[GPUInfo]:
    """Query NVIDIA GPUs via nvidia-smi."""
    gpus = []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 9:
                    try:
                        gpus.append(GPUInfo(
                            index=int(parts[0]),
                            name=parts[1],
                            driver_version=parts[2],
                            total_vram_mb=float(parts[3]),
                            used_vram_mb=float(parts[4]),
                            free_vram_mb=float(parts[5]),
                            utilization_percent=float(parts[6]),
                            temperature_c=float(parts[7]),
                            power_draw_w=float(parts[8]) if parts[8] != "[N/A]" else 0.0,
                            gpu_type="nvidia",
                        ))
                    except (ValueError, IndexError):
                        pass
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return gpus


def _query_torch_cuda() -> List[GPUInfo]:
    """Query GPU info via PyTorch CUDA."""
    gpus = []
    try:
        import torch
        if not torch.cuda.is_available():
            return gpus
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_mb = props.total_mem / (1024 * 1024)
            allocated = torch.cuda.memory_allocated(i) / (1024 * 1024)
            reserved = torch.cuda.memory_reserved(i) / (1024 * 1024)
            free_mb = total_mb - reserved
            gpus.append(GPUInfo(
                index=i,
                name=props.name,
                total_vram_mb=round(total_mb, 1),
                used_vram_mb=round(allocated, 1),
                free_vram_mb=round(free_mb, 1),
                utilization_percent=round((allocated / total_mb) * 100, 1) if total_mb > 0 else 0.0,
                gpu_type="nvidia" if "nvidia" in props.name.lower() or "geforce" in props.name.lower() else "unknown",
            ))
    except (ImportError, Exception):
        pass
    return gpus


def _query_mps_info() -> List[GPUInfo]:
    """Query Apple MPS GPU info."""
    if platform.system() != "Darwin":
        return []
    try:
        import torch
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            return []
        # Apple Silicon uses unified memory; estimate from system RAM
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                mem_bytes = int(result.stdout.strip())
                total_mb = mem_bytes / (1024 * 1024)
                # Apple allocates ~75% to GPU as needed
                gpu_share = total_mb * 0.75
                allocated = torch.mps.current_allocated_memory() / (1024 * 1024) if hasattr(torch.mps, "current_allocated_memory") else 0
                return [GPUInfo(
                    index=0,
                    name="Apple Silicon (MPS)",
                    total_vram_mb=round(gpu_share, 1),
                    used_vram_mb=round(allocated, 1),
                    free_vram_mb=round(gpu_share - allocated, 1),
                    gpu_type="apple",
                )]
        except (subprocess.TimeoutExpired, ValueError):
            pass
    except ImportError:
        pass
    return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_gpu_info() -> List[GPUInfo]:
    """Detect and return information about all available GPUs.

    Tries NVIDIA (nvidia-smi), then PyTorch CUDA, then Apple MPS.

    Returns:
        List of GPUInfo for each detected GPU.
    """
    gpus = _query_nvidia_smi()
    if gpus:
        return gpus

    gpus = _query_torch_cuda()
    if gpus:
        return gpus

    gpus = _query_mps_info()
    return gpus


def get_vram_status() -> VRAMStatus:
    """Get the current VRAM allocation status.

    Returns:
        VRAMStatus with aggregate memory information.
    """
    gpus = get_gpu_info()

    total = sum(g.total_vram_mb for g in gpus)
    used = sum(g.used_vram_mb for g in gpus)
    free = sum(g.free_vram_mb for g in gpus)
    gpu_type = gpus[0].gpu_type if gpus else "none"

    with _models_lock:
        models_count = len(_loaded_models)
        models_vram = sum(m.size_mb for m in _loaded_models.values() if m.device != "cpu")

    overhead = max(0.0, used - models_vram) if used > 0 else 0.0
    utilization = round((used / total) * 100, 1) if total > 0 else 0.0

    return VRAMStatus(
        total_vram_mb=round(total, 1),
        used_vram_mb=round(used, 1),
        free_vram_mb=round(free, 1),
        models_loaded=models_count,
        models_vram_mb=round(models_vram, 1),
        system_overhead_mb=round(overhead, 1),
        utilization_percent=utilization,
        gpu_type=gpu_type,
    )


def get_loaded_models() -> List[LoadedModel]:
    """Return all models currently tracked as loaded.

    Returns:
        List of LoadedModel entries.
    """
    with _models_lock:
        return list(_loaded_models.values())


def unload_model(
    model_name: str,
    on_progress: Optional[Callable] = None,
) -> bool:
    """Unload a model from GPU memory.

    Removes the model from the tracking registry and attempts to free
    GPU memory via ``torch.cuda.empty_cache()`` or ``torch.mps.empty_cache()``.

    Args:
        model_name: Model identifier to unload.
        on_progress: Optional progress callback.

    Returns:
        True if the model was found and unloaded.
    """
    with _models_lock:
        model = _loaded_models.pop(model_name, None)

    if model is None:
        logger.warning("Cannot unload '%s': not found in loaded models.", model_name)
        return False

    if on_progress:
        on_progress({"step": "unloading", "model": model_name, "size_mb": model.size_mb})

    # Try to free GPU memory
    try:
        import torch
        if model.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif model.device == "mps" and hasattr(torch, "mps"):
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
    except ImportError:
        pass

    logger.info("Unloaded model '%s' (%.1f MB from %s)", model_name, model.size_mb, model.device)

    if on_progress:
        on_progress({"step": "done", "model": model_name, "freed_mb": model.size_mb})

    return True


def recommend_unload(
    required_vram: float,
    on_progress: Optional[Callable] = None,
) -> UnloadRecommendation:
    """Recommend which models to unload to free a given amount of VRAM.

    Uses a priority strategy:
    1. Unpinned models, sorted by idle time (longest-idle first).
    2. Then by use count (least-used first).

    Args:
        required_vram: Amount of VRAM needed (MB).
        on_progress: Optional progress callback.

    Returns:
        UnloadRecommendation with the suggested unload plan.
    """
    status = get_vram_status()
    rec = UnloadRecommendation(
        required_mb=required_vram,
        current_free_mb=status.free_vram_mb,
    )

    if status.free_vram_mb >= required_vram:
        rec.sufficient = True
        rec.reason = f"Sufficient VRAM available ({status.free_vram_mb:.0f} MB free >= {required_vram:.0f} MB needed)."
        return rec

    needed = required_vram - status.free_vram_mb

    if on_progress:
        on_progress({"step": "analyzing", "needed_mb": needed})

    with _models_lock:
        candidates = [
            m for m in _loaded_models.values()
            if not m.pinned and m.device != "cpu"
        ]

    # Sort: longest idle first, then least used
    now = time.time()
    candidates.sort(key=lambda m: (-(now - m.last_used), m.use_count))

    to_unload = []
    freed = 0.0
    for model in candidates:
        to_unload.append(model.name)
        freed += model.size_mb
        if freed >= needed:
            break

    rec.models_to_unload = to_unload
    rec.freed_mb = round(freed, 1)
    rec.sufficient = freed >= needed

    if rec.sufficient:
        rec.reason = (
            f"Unloading {len(to_unload)} model(s) will free {freed:.0f} MB, "
            f"providing the needed {required_vram:.0f} MB."
        )
    else:
        rec.reason = (
            f"Even after unloading all {len(to_unload)} unpinned model(s) "
            f"({freed:.0f} MB), only {status.free_vram_mb + freed:.0f} MB available "
            f"vs {required_vram:.0f} MB needed."
        )

    return rec
