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
    """Check if auto-editor (motion-based editing) is available.

    Checks for native Nim binary (v30+) on PATH first, then falls back
    to the legacy pip package (v29.x).
    """
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


def check_silero_vad_available() -> bool:
    """Check if Silero VAD (neural voice activity detection) is available."""
    return _try_import("torch") is not None


def check_crisper_whisper_available() -> bool:
    """Check if CrisperWhisper (verbatim filler detection ASR) is available."""
    # CrisperWhisper is a modified Whisper model loaded via transformers
    return _try_import("transformers") is not None and _try_import("torch") is not None


def check_sam2_available() -> bool:
    """Check if SAM2 (Segment Anything Model 2) is available."""
    return _try_import("sam2") is not None


def check_propainter_available() -> bool:
    """Check if ProPainter (video inpainting) is available."""
    return _try_import("propainter") is not None


def check_otio_available() -> bool:
    """Check if OpenTimelineIO (universal timeline export) is available."""
    return _try_import("opentimelineio") is not None


def check_deepface_available() -> bool:
    """Check if deepface (emotion/face analysis) is available."""
    return _try_import("deepface") is not None


def check_rvm_available() -> bool:
    """Check if Robust Video Matting (temporal bg removal) is available."""
    return _try_import("torch") is not None


def check_depth_available() -> bool:
    """Check if Depth Anything (depth estimation) is available."""
    return _try_import("torch") is not None and _try_import("transformers") is not None


def check_resolve_available() -> bool:
    """Check if DaVinci Resolve scripting API is available."""
    try:
        from opencut.core.resolve_bridge import check_resolve_available as _check
        return _check()
    except Exception:
        return False


def check_multimodal_diarize_available() -> bool:
    """Check if multimodal diarization (face detection + audio) is available."""
    # Needs at least one face detection backend + cv2
    if _try_import("cv2") is None:
        return False
    if _try_import("insightface") is not None:
        return True
    if _try_import("facenet_pytorch") is not None:
        return True
    # Haar cascade fallback is always available with cv2
    return True


def check_broll_generate_available() -> bool:
    """Check if AI B-roll generation (text-to-video) is available."""
    return _try_import("diffusers") is not None and _try_import("torch") is not None


def check_websocket_available() -> bool:
    """Check if WebSocket bridge (websockets package) is available."""
    return _try_import("websockets") is not None


def check_social_post_available() -> bool:
    """Check if social media posting credentials exist."""
    import os
    creds_path = os.path.join(os.path.expanduser("~"), ".opencut", "social_credentials.json")
    return os.path.isfile(creds_path)


def check_neural_interp_available() -> bool:
    """Check if neural frame interpolation is usable.

    Always True — the FFmpeg ``minterpolate`` fallback is always present.
    RIFE-NCNN-Vulkan CLI and torch RIFE are optional accelerators.
    """
    try:
        from opencut.core.neural_interp import check_neural_interp_available as _c
        return _c()
    except Exception:
        return False


def check_rife_cli_available() -> bool:
    """Check if the rife-ncnn-vulkan CLI is on PATH."""
    import shutil
    return shutil.which("rife-ncnn-vulkan") is not None


def check_declarative_compose_available() -> bool:
    """Check if declarative JSON composition can run. Needs FFmpeg only."""
    import shutil
    return shutil.which("ffmpeg") is not None


# --- v1.18.0 Wave A + Wave D availability checks ---

def check_f5_tts_available() -> bool:
    """Check if F5-TTS (flow-matching voice clone) is installed."""
    return _try_import("f5_tts") is not None


def check_beatnet_available() -> bool:
    """Check if BeatNet (beat + downbeat neural detector) is installed."""
    return _try_import("BeatNet") is not None


def check_clip_iqa_available() -> bool:
    """Check if CLIP-IQA+ clip quality scoring is usable."""
    try:
        from opencut.core.clip_quality import check_clip_iqa_available as _c
        return _c()
    except Exception:
        return False


def check_hsemotion_available() -> bool:
    """Check if HSEmotion (face emotion arc) is installed."""
    try:
        from opencut.core.emotion_arc import check_hsemotion_available as _c
        return _c()
    except Exception:
        return False


def check_ab_av1_available() -> bool:
    """Check if ab-av1 (VMAF-target encoder) is on PATH."""
    import shutil
    return shutil.which("ab-av1") is not None


def check_aaf_adapter_available() -> bool:
    """Check if OTIO AAF adapter is installed."""
    try:
        from opencut.export.otio_export import check_aaf_available as _c
        return _c()
    except Exception:
        return False


def check_event_moments_available() -> bool:
    """Event-moment finder uses stdlib + FFmpeg; always available."""
    import shutil
    return shutil.which("ffmpeg") is not None


# --- v1.19.0 Wave A2.3 / A3.1 / A4.2 / D2 availability checks ---

def check_birefnet_available() -> bool:
    try:
        from opencut.core.matte_birefnet import check_birefnet_available as _c
        return _c()
    except Exception:
        return False


def check_pyonfx_available() -> bool:
    return _try_import("pyonfx") is not None


def check_svtav1_psy_available() -> bool:
    try:
        from opencut.core.svtav1_psy import check_svtav1_psy_available as _c
        return _c()
    except Exception:
        return False


def check_ddcolor_available() -> bool:
    try:
        from opencut.core.colorize_ddcolor import check_ddcolor_available as _c
        return _c()
    except Exception:
        return False


def check_vrt_available() -> bool:
    try:
        from opencut.core.restore_vrt import check_vrt_available as _c
        return _c()
    except Exception:
        return False


def check_neural_deflicker_available() -> bool:
    try:
        from opencut.core.deflicker_neural import check_neural_deflicker_available as _c
        return _c()
    except Exception:
        return False


# --- v1.20.0 Wave C availability checks ---

def check_otio_diff_available() -> bool:
    try:
        from opencut.export.otio_diff import check_otio_diff_available as _c
        return _c()
    except Exception:
        return False


def check_quality_metrics_available() -> bool:
    try:
        from opencut.core.quality_metrics import check_quality_metrics_available as _c
        return _c()
    except Exception:
        return False


def check_vmaf_available() -> bool:
    try:
        from opencut.core.quality_metrics import check_vmaf_available as _c
        return _c()
    except Exception:
        return False


def check_sentry_available() -> bool:
    """True when sentry_sdk is importable (DSN need not be set)."""
    return _try_import("sentry_sdk") is not None


# --- v1.21.0 Wave D availability checks ---

def check_vvc_available() -> bool:
    try:
        from opencut.core.vvc_export import check_vvc_available as _c
        return _c()
    except Exception:
        return False


def check_srt_available() -> bool:
    try:
        from opencut.core.srt_streaming import check_srt_available as _c
        return _c()
    except Exception:
        return False


def check_colour_science_available() -> bool:
    return _try_import("colour") is not None


def check_voice_grammar_available() -> bool:
    """Voice-command grammar is pure-Python stdlib — always available."""
    try:
        from opencut.core.voice_command_grammar import parse  # noqa: F401
        return True
    except Exception:
        return False


def check_atheris_available() -> bool:
    """True when Atheris (fuzzing harness dep) is installed."""
    return _try_import("atheris") is not None
