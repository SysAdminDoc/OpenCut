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
    return _try_import("transnetv2") is not None or _try_import("transnetv2_pytorch") is not None


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


# --- v1.22.0 Wave E availability checks ---

def check_shaka_available() -> bool:
    try:
        from opencut.core.shaka_pkg import check_shaka_available as _c
        return _c()
    except Exception:
        return False


def check_obs_bridge_available() -> bool:
    try:
        from opencut.core.obs_bridge import check_obs_bridge_available as _c
        return _c()
    except Exception:
        return False


def check_runpod_available() -> bool:
    """Transport is stdlib; always True. Use check_runpod_api_key for env."""
    return True


def check_runpod_api_key_set() -> bool:
    import os
    return bool((os.environ.get("RUNPOD_API_KEY") or "").strip())


def check_plausible_configured() -> bool:
    try:
        from opencut.core.telemetry_plausible import check_plausible_available as _c
        return _c()
    except Exception:
        return False


def check_aptabase_configured() -> bool:
    try:
        from opencut.core.telemetry_aptabase import check_aptabase_available as _c
        return _c()
    except Exception:
        return False


# --- v1.23.0 Wave F (cross-cutting infrastructure) ---

def check_openapi_available() -> bool:
    """Always True — stdlib-only generator."""
    return True


def check_gpu_semaphore_available() -> bool:
    """Always True — stdlib threading semaphore."""
    return True


def check_rate_limit_categories_available() -> bool:
    """Always True — stdlib threading semaphores."""
    return True


def check_temp_cleanup_available() -> bool:
    try:
        from opencut.core.temp_cleanup import check_temp_cleanup_available as _c
        return _c()
    except Exception:
        return False


# --- v1.24.0 Wave G (second wide-net pass) ---

def check_disk_monitor_available() -> bool:
    try:
        from opencut.core.disk_monitor import check_disk_monitor_available as _c
        return _c()
    except Exception:
        return False


def check_request_correlation_available() -> bool:
    try:
        from opencut.core.request_correlation import check_request_correlation_available as _c
        return _c()
    except Exception:
        return False


def check_deprecation_registry_available() -> bool:
    try:
        from opencut.core.deprecation import check_deprecation_available as _c
        return _c()
    except Exception:
        return False


# --- v1.25.0 Wave H (Commercial Parity & Content-Creator Polish) ---

def check_virality_score_available() -> bool:
    try:
        from opencut.core.virality_score import check_virality_score_available as _c
        return _c()
    except Exception:
        return False


def check_cursor_zoom_available() -> bool:
    try:
        from opencut.core.cursor_zoom import check_cursor_zoom_available as _c
        return _c()
    except Exception:
        return False


def check_changelog_feed_available() -> bool:
    """Always True — stdlib urllib, fails gracefully on network errors."""
    return True


def check_issue_report_available() -> bool:
    """Always True — stdlib only."""
    return True


def check_demo_bundle_available() -> bool:
    try:
        from opencut.core.demo_bundle import check_demo_bundle_available as _c
        return _c()
    except Exception:
        return False


def check_gist_sync_available() -> bool:
    """Always True — stdlib urllib, fails gracefully on network errors."""
    return True


def check_onboarding_available() -> bool:
    """Always True — user_data wrappers."""
    return True


def check_flashvsr_available() -> bool:
    try:
        from opencut.core.upscale_flashvsr import check_flashvsr_available as _c
        return _c()
    except Exception:
        return False


def check_rose_available() -> bool:
    try:
        from opencut.core.inpaint_rose import check_rose_available as _c
        return _c()
    except Exception:
        return False


def check_sammie_available() -> bool:
    try:
        from opencut.core.matte_sammie import check_sammie_available as _c
        return _c()
    except Exception:
        return False


def check_omnivoice_available() -> bool:
    try:
        from opencut.core.tts_omnivoice import check_omnivoice_available as _c
        return _c()
    except Exception:
        return False


def check_reezsynth_available() -> bool:
    try:
        from opencut.core.style_reezsynth import check_reezsynth_available as _c
        return _c()
    except Exception:
        return False


def check_vidmuse_available() -> bool:
    try:
        from opencut.core.music_vidmuse import check_vidmuse_available as _c
        return _c()
    except Exception:
        return False


def check_video_agent_available() -> bool:
    """Always False in v1.25.0 — Tier 3 stub."""
    try:
        from opencut.core.video_agent import check_video_agent_available as _c
        return _c()
    except Exception:
        return False


def check_gen_video_cloud_available() -> bool:
    """Always False in v1.25.0 — Tier 3 stub."""
    try:
        from opencut.core.gen_video_cloud import check_gen_video_cloud_available as _c
        return _c()
    except Exception:
        return False


def check_lipsync_advanced_available() -> bool:
    """Always False in v1.25.0 — Tier 3 stub."""
    try:
        from opencut.core.lipsync_advanced import check_lipsync_advanced_available as _c
        return _c()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Wave K (Completeness Pass) — v1.28.0
# ---------------------------------------------------------------------------

def check_audio_watermark(verbose=False):
    """K1.1 — AudioSeal watermark. Requires audioseal."""
    from opencut.core.audio_watermark import INSTALL_HINT, check_audioseal_available
    ok = check_audioseal_available()
    if verbose:
        status = "OK" if ok else f"MISSING — {INSTALL_HINT}"
        print(f"  audio_watermark (audioseal): {status}")
    return ok


def check_batch_reframe(verbose=False):
    """K1.4 — Batch reframe. Uses FFmpeg only."""
    from opencut.core.batch_reframe import check_batch_reframe_available
    ok = check_batch_reframe_available()
    if verbose:
        print(f"  batch_reframe: {'OK' if ok else 'MISSING — ffmpeg not found'}")
    return ok


def check_clip_rating(verbose=False):
    """K1.5 — Clip rating. Stdlib JSON only."""
    from opencut.core.clip_rating import check_clip_rating_available
    ok = check_clip_rating_available()
    if verbose:
        print(f"  clip_rating: {'OK' if ok else 'MISSING'}")
    return ok


def check_subtitle_qa(verbose=False):
    """K1.6 — Subtitle QA. Stdlib only."""
    from opencut.core.subtitle_qa import check_subtitle_qa_available
    ok = check_subtitle_qa_available()
    if verbose:
        print(f"  subtitle_qa: {'OK' if ok else 'MISSING'}")
    return ok


def check_profanity_censor(verbose=False):
    """K1.7 — Profanity censor. Requires FFmpeg."""
    from opencut.core.profanity_censor import check_profanity_censor_available
    ok = check_profanity_censor_available()
    if verbose:
        status = "OK" if ok else "MISSING — ffmpeg not found"
        print(f"  profanity_censor: {status}")
    return ok


def check_spectral_match(verbose=False):
    """K1.8 — Spectral match. Requires scipy + numpy."""
    from opencut.core.spectral_match import INSTALL_HINT, check_spectral_match_available
    ok = check_spectral_match_available()
    if verbose:
        status = "OK" if ok else f"MISSING — {INSTALL_HINT}"
        print(f"  spectral_match: {status}")
    return ok


def check_lottie_import(verbose=False):
    """K1.9 — Lottie import. Requires lottie pip package."""
    from opencut.core.lottie_import import INSTALL_HINT, check_lottie_available
    ok = check_lottie_available()
    if verbose:
        status = "OK" if ok else f"MISSING — {INSTALL_HINT}"
        print(f"  lottie_import: {status}")
    return ok


def check_semantic_search(verbose=False):
    """K1.10 — Semantic search. Requires torch."""
    from opencut.core.semantic_search import INSTALL_HINT, check_semantic_search_available
    ok = check_semantic_search_available()
    if verbose:
        status = "OK" if ok else f"MISSING — {INSTALL_HINT}"
        print(f"  semantic_search: {status}")
    return ok


def check_gptsovits(verbose=False):
    """K2.1 — GPT-SoVITS TTS stub."""
    from opencut.core.tts_gptsovits import INSTALL_HINT, check_gptsovits_available
    ok = check_gptsovits_available()
    if verbose:
        status = "OK" if ok else f"STUB — {INSTALL_HINT}"
        print(f"  tts_gptsovits: {status}")
    return ok


def check_amphion_tts(verbose=False):
    """K2.2 — Amphion TTS stub."""
    from opencut.core.tts_amphion import INSTALL_HINT, check_amphion_available
    ok = check_amphion_available()
    if verbose:
        status = "OK" if ok else f"STUB — {INSTALL_HINT}"
        print(f"  tts_amphion: {status}")
    return ok


def check_vevo2(verbose=False):
    """K2.3 — Vevo2 singing voice stub."""
    from opencut.core.singing_vevo2 import INSTALL_HINT, check_vevo2_available
    ok = check_vevo2_available()
    if verbose:
        status = "OK" if ok else f"STUB — {INSTALL_HINT}"
        print(f"  singing_vevo2: {status}")
    return ok


def check_cosyvoice2(verbose=False):
    """K2.4 — CosyVoice2 TTS stub."""
    from opencut.core.tts_cosyvoice2 import INSTALL_HINT, check_cosyvoice2_available
    ok = check_cosyvoice2_available()
    if verbose:
        status = "OK" if ok else f"STUB — {INSTALL_HINT}"
        print(f"  tts_cosyvoice2: {status}")
    return ok


def check_echomimic(verbose=False):
    """K2.5 — EchoMimic lipsync stub."""
    from opencut.core.lipsync_echomimic import INSTALL_HINT, check_echomimic_available
    ok = check_echomimic_available()
    if verbose:
        status = "OK" if ok else f"STUB — {INSTALL_HINT}"
        print(f"  lipsync_echomimic: {status}")
    return ok


def check_tokenflow(verbose=False):
    """K2.6 — TokenFlow style transfer stub."""
    from opencut.core.style_tokenflow import INSTALL_HINT, check_tokenflow_available
    ok = check_tokenflow_available()
    if verbose:
        status = "OK" if ok else f"STUB — {INSTALL_HINT}"
        print(f"  style_tokenflow: {status}")
    return ok


def check_track_cutie(verbose=False):
    """K2.7 — Cutie object tracking stub."""
    from opencut.core.track_cutie import INSTALL_HINT, check_cutie_available
    ok = check_cutie_available()
    if verbose:
        status = "OK" if ok else f"STUB — {INSTALL_HINT}"
        print(f"  track_cutie: {status}")
    return ok


def check_track_deva(verbose=False):
    """K2.8 — DEVA tracking stub."""
    from opencut.core.track_deva import INSTALL_HINT, check_deva_available
    ok = check_deva_available()
    if verbose:
        status = "OK" if ok else f"STUB — {INSTALL_HINT}"
        print(f"  track_deva: {status}")
    return ok


def check_searaft(verbose=False):
    """K2.9 — SEA-RAFT optical flow stub."""
    from opencut.core.flow_searaft import INSTALL_HINT, check_searaft_available
    ok = check_searaft_available()
    if verbose:
        status = "OK" if ok else f"STUB — {INSTALL_HINT}"
        print(f"  flow_searaft: {status}")
    return ok


def check_diffbir(verbose=False):
    """K2.10 — DiffBIR restoration stub."""
    from opencut.core.restore_diffbir import INSTALL_HINT, check_diffbir_available
    ok = check_diffbir_available()
    if verbose:
        status = "OK" if ok else f"STUB — {INSTALL_HINT}"
        print(f"  restore_diffbir: {status}")
    return ok


def check_gyroflow(verbose=False):
    """K2.11 — Gyroflow stabilization stub."""
    from opencut.core.stabilize_gyroflow import INSTALL_HINT, check_gyroflow_available
    ok = check_gyroflow_available()
    if verbose:
        status = "OK" if ok else f"STUB — {INSTALL_HINT}"
        print(f"  stabilize_gyroflow: {status}")
    return ok


def check_deblur_motion(verbose=False):
    """K2.12 — Motion deblur stub."""
    from opencut.core.deblur_motion import INSTALL_HINT, check_deblur_motion_available
    ok = check_deblur_motion_available()
    if verbose:
        status = "OK" if ok else f"STUB — {INSTALL_HINT}"
        print(f"  deblur_motion: {status}")
    return ok


def check_depth_depthpro(verbose=False):
    """K2.13 — Depth Pro stub."""
    from opencut.core.depth_depthpro import INSTALL_HINT, check_depthpro_available
    ok = check_depthpro_available()
    if verbose:
        status = "OK" if ok else f"STUB — {INSTALL_HINT}"
        print(f"  depth_depthpro: {status}")
    return ok


def check_depth_flow(verbose=False):
    """K2.14 — DepthFlow stub."""
    from opencut.core.depth_flow import INSTALL_HINT, check_depthflow_available
    ok = check_depthflow_available()
    if verbose:
        status = "OK" if ok else f"STUB — {INSTALL_HINT}"
        print(f"  depth_flow: {status}")
    return ok


def check_sfx_audiogen(verbose=False):
    """K2.15 — AudioGen SFX stub."""
    from opencut.core.sfx_audiogen import INSTALL_HINT, check_audiogen_available
    ok = check_audiogen_available()
    if verbose:
        status = "OK" if ok else f"STUB — {INSTALL_HINT}"
        print(f"  sfx_audiogen: {status}")
    return ok


def check_gen_video_opensora(verbose=False):
    """K2.16 — OpenSora video generation stub."""
    from opencut.core.gen_video_opensora import INSTALL_HINT, check_opensora_available
    ok = check_opensora_available()
    if verbose:
        status = "OK" if ok else f"STUB — {INSTALL_HINT}"
        print(f"  gen_video_opensora: {status}")
    return ok


def check_gen_video_ltx(verbose=False):
    """K2.17 — LTX-Video generation stub."""
    from opencut.core.gen_video_ltx import INSTALL_HINT, check_ltx_available
    ok = check_ltx_available()
    if verbose:
        status = "OK" if ok else f"STUB — {INSTALL_HINT}"
        print(f"  gen_video_ltx: {status}")
    return ok


def check_audio_reactive_fx(verbose=False):
    """K2.18 — Audio Reactive FX stub."""
    from opencut.core.audio_reactive_fx import INSTALL_HINT, check_audio_reactive_available
    ok = check_audio_reactive_available()
    if verbose:
        status = "OK" if ok else f"STUB — {INSTALL_HINT}"
        print(f"  audio_reactive_fx: {status}")
    return ok


def check_cinefocus(verbose=False):
    """K2.19 — CineFocus rack-focus stub."""
    from opencut.core.cinefocus import INSTALL_HINT, check_cinefocus_available
    ok = check_cinefocus_available()
    if verbose:
        status = "OK" if ok else f"STUB — {INSTALL_HINT}"
        print(f"  cinefocus: {status}")
    return ok


# ---------------------------------------------------------------------------
# Wave L — v1.29.0
# ---------------------------------------------------------------------------

def check_elevenlabs_available() -> bool:
    """L1.1 — ElevenLabs cloud TTS (SDK + API key required)."""
    try:
        from opencut.core.tts_elevenlabs import check_elevenlabs_available as _c
        return _c()
    except Exception:
        return False


def check_upscale_hub_available() -> bool:
    """L1.2 — Smart upscaling hub (lanczos always available)."""
    return True


def check_face_reshape_available() -> bool:
    """L1.3 — AI face reshape (mediapipe + cv2 + numpy required)."""
    try:
        from opencut.core.face_reshape import check_face_reshape_available as _c
        return _c()
    except Exception:
        return False


def check_skin_retouch_available() -> bool:
    """L1.4 — AI skin retouch (cv2 + numpy required)."""
    try:
        from opencut.core.skin_retouch import check_skin_retouch_available as _c
        return _c()
    except Exception:
        return False


def check_sparktts_available() -> bool:
    """L2.3 — Spark-TTS CPU-native zero-shot TTS (sparktts package)."""
    return _try_import("sparktts") is not None


def check_moonshine_available() -> bool:
    """L2.6 — Moonshine real-time ASR (CPU-optimized, MIT for English)."""
    return _try_import("moonshine") is not None


def check_wave_l(verbose=False):
    """Run all Wave L availability checks and print a summary."""
    checks = [
        ("elevenlabs_tts", check_elevenlabs_available),
        ("sparktts", check_sparktts_available),
        ("upscale_hub", check_upscale_hub_available),
        ("face_reshape", check_face_reshape_available),
        ("skin_retouch", check_skin_retouch_available),
    ]
    results = []
    for name, fn in checks:
        try:
            ok = fn()
            results.append(ok)
            if verbose:
                print(f"  {name}: {'OK' if ok else 'MISSING'}")
        except Exception as e:
            results.append(False)
            if verbose:
                print(f"  {name}: ERROR — {e}")
    ok_count = sum(1 for r in results if r)
    if verbose:
        print(f"\nWave L: {ok_count}/{len(results)} components available")
    return ok_count, len(results)


def check_wave_k(verbose=False):
    """Run all Wave K availability checks and print a summary."""
    checks = [
        check_audio_watermark, check_batch_reframe, check_clip_rating,
        check_subtitle_qa, check_profanity_censor, check_spectral_match,
        check_lottie_import, check_semantic_search,
        check_gptsovits, check_amphion_tts, check_vevo2, check_cosyvoice2,
        check_echomimic, check_tokenflow, check_track_cutie, check_track_deva,
        check_searaft, check_diffbir, check_gyroflow, check_deblur_motion,
        check_depth_depthpro, check_depth_flow, check_sfx_audiogen,
        check_gen_video_opensora, check_gen_video_ltx,
        check_audio_reactive_fx, check_cinefocus,
    ]
    results = []
    for fn in checks:
        try:
            results.append(fn(verbose=verbose))
        except Exception as e:
            if verbose:
                print(f"  {fn.__name__}: ERROR — {e}")
            results.append(False)
    ok_count = sum(1 for r in results if r)
    if verbose:
        print(f"\nWave K: {ok_count}/{len(results)} components available")
    return ok_count, len(results)
