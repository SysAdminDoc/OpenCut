"""
OpenCut Dependency Availability Checks

Centralized check functions so route files don't duplicate definitions.
Each function returns True/False indicating if the dependency is usable.
"""

from opencut.helpers import _try_import


def check_demucs_available():
    """Check if demucs (audio separation) is installed."""
    return _try_import("demucs") is not None


def check_watermark_available():
    """Check if watermark removal deps (lama-cleaner or similar) are installed."""
    return _try_import("simple_lama_inpainting") is not None or _try_import("lama_cleaner") is not None


def check_pedalboard_available():
    """Check if pedalboard (audio effects) is installed."""
    return _try_import("pedalboard") is not None


def check_audiocraft_available():
    """Check if audiocraft (music generation) is installed."""
    return _try_import("audiocraft") is not None


def check_edge_tts_available():
    """Check if edge-tts (text-to-speech) is installed."""
    return _try_import("edge_tts") is not None


def check_rembg_available():
    """Check if rembg (background removal) is installed."""
    return _try_import("rembg") is not None


def check_upscale_available():
    """Check if Real-ESRGAN (upscaling) is installed."""
    return _try_import("realesrgan") is not None


def check_scenedetect_available():
    """Check if PySceneDetect is installed."""
    return _try_import("scenedetect") is not None
