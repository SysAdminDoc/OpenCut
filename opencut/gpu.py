"""
GPU Memory Context Manager

Provides GPUContext for automatic VRAM cleanup after GPU operations,
plus standalone helpers for device detection and VRAM checks.

All torch imports are optional -- functions degrade gracefully on
CPU-only systems.
"""

import logging

logger = logging.getLogger("opencut")

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

def get_device() -> str:
    """Return 'cuda' if a CUDA GPU is available, else 'cpu'."""
    if _HAS_TORCH and torch.cuda.is_available():
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
    """
    if not _HAS_TORCH or not torch.cuda.is_available():
        return (0.0, 0.0)

    try:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(torch.cuda.mem_get_info)
            free, total = future.result(timeout=5)
        available_gb = free / (1024 ** 3)
        total_gb = total / (1024 ** 3)
    except concurrent.futures.TimeoutError:
        logger.warning("VRAM query timed out (>5s) — NVIDIA driver may be hung")
        return (0.0, 0.0)
    except Exception as exc:
        logger.warning("Failed to query VRAM: %s", exc)
        return (0.0, 0.0)

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
        if _HAS_TORCH and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                logger.debug("GPUContext: CUDA cache cleared")
            except Exception as exc:
                logger.warning("GPUContext: failed to clear CUDA cache: %s", exc)

        # Do not suppress exceptions
        return False
