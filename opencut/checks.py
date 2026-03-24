"""
OpenCut Dependency Availability Checks

Centralized check functions so route files don't duplicate definitions.
Each function returns True/False indicating if the dependency is usable.
"""

import threading

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


def check_auto_editor_available():
    """Check if auto-editor (motion-based editing) is installed."""
    import shutil
    return shutil.which("auto-editor") is not None or _try_import("auto_editor") is not None


def check_transnetv2_available():
    """Check if TransNetV2 (ML scene detection) is installed."""
    return _try_import("transnetv2") is not None


def check_resemble_enhance_available():
    """Check if Resemble Enhance (speech super-resolution) is installed."""
    return _try_import("resemble_enhance") is not None


def check_mediapipe_available():
    """Check if MediaPipe (face detection/tracking) is installed."""
    return _try_import("mediapipe") is not None


_ollama_cache = {"result": None, "expires": 0}
_ollama_cache_lock = threading.Lock()


def check_ollama_available():
    """Check if Ollama is running locally (cached for 30s)."""
    import time
    now = time.monotonic()
    with _ollama_cache_lock:
        if _ollama_cache["result"] is not None and now < _ollama_cache["expires"]:
            return _ollama_cache["result"]
    try:
        import urllib.request
        resp = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
        resp.close()
        result = True
    except Exception:
        result = False
    with _ollama_cache_lock:
        _ollama_cache["result"] = result
        _ollama_cache["expires"] = time.monotonic() + 30
    return result


def check_llm_available():
    """Check if any LLM provider is available (Ollama or API key configured)."""
    if check_ollama_available():
        return True
    # API keys would be checked via settings — here we just check Ollama
    return False


def check_color_match_available() -> bool:
    """Check if color matching (OpenCV + NumPy) is available."""
    return _try_import("cv2") is not None and _try_import("numpy") is not None


def check_auto_zoom_available() -> bool:
    """Check if auto zoom face detection (OpenCV) is available."""
    return _try_import("cv2") is not None


def check_loudness_match_available() -> bool:
    """Check if loudness matching (FFmpeg) is available."""
    import shutil
    return shutil.which("ffmpeg") is not None


def check_footage_search_available() -> bool:
    """Check if footage search indexing is available (always True — uses stdlib only)."""
    return True
