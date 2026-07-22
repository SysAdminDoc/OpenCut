"""
Multi-Engine Backend — Plugin Architecture

Provides a registry for swappable AI model backends per feature, letting users
choose quality/speed/VRAM tradeoffs.

Each "engine" is a named backend for a specific feature domain:
- silence_detection: "energy", "silero_vad"
- transcription: "faster_whisper", "crisper_whisper", "whisper_cpp"
- background_removal: "rembg", "rvm"
- scene_detection: "threshold", "transnetv2"
- tts: "edge_tts", "kokoro", "chatterbox"
- stem_separation: "demucs", "bs_roformer", "mdx_net"
- face_restoration: "gfpgan", "codeformer"
- upscaling: "realesrgan", "espcn"
- depth_estimation: "depth_anything_v2"
- diarization: "pyannote", "multimodal"
- broll_generation: "cogvideox", "wan", "hunyuan", "svd"
- object_removal: "sam2_propainter", "lama"
- watermark_detection: "florence2", "edge_fallback"

Users configure preferred engines via settings. The registry validates availability
and falls back gracefully.
"""

import importlib.util
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("opencut")

_CORE_DIR = Path(__file__).resolve().parent


@dataclass
class EngineInfo:
    """Metadata about a registered engine."""
    name: str                    # Unique engine ID (e.g., "silero_vad")
    domain: str                  # Feature domain (e.g., "silence_detection")
    display_name: str            # Human-readable name
    description: str             # Short description
    check_fn: Callable[[], bool]  # Returns True if engine is available
    priority: int = 50           # Higher = preferred (when multiple available)
    vram_mb: int = 0             # Approximate VRAM usage
    speed_rating: str = "medium" # "fast", "medium", "slow"
    quality_rating: str = "medium"  # "low", "medium", "high"
    tags: List[str] = field(default_factory=list)
    # Stem of the ``opencut.core`` module that implements this engine's
    # adapter (e.g. "asr_parakeet"). When set and that module is still a
    # terminal NotImplementedError stub (or does not exist yet), the engine
    # stays listed as coming-soon but never reports available / resolves as
    # active — a dependency check alone does not prove the adapter works.
    impl_module: Optional[str] = None

    @property
    def is_stub(self) -> bool:
        """True while the engine's adapter is an unimplemented placeholder."""
        if not self.impl_module:
            return False
        from opencut.core.stub_scan import is_stub_module
        if is_stub_module(self.impl_module):
            return True
        # An adapter module that has not been written yet is equally
        # unimplemented (e.g. sortformer has no core adapter at all).
        return not (_CORE_DIR / f"{self.impl_module}.py").exists()

    @property
    def is_available(self) -> bool:
        if self.is_stub:
            return False
        try:
            return self.check_fn()
        except Exception:
            return False


class EngineRegistry:
    """
    Central registry for AI model backends.

    Thread-safe singleton. Engines register on import, users query
    available engines per domain, and the system auto-selects the best
    available engine based on priority and user preferences.
    """

    def __init__(self):
        self._engines: Dict[str, Dict[str, EngineInfo]] = {}  # domain -> {name -> info}
        self._preferences: Dict[str, str] = {}  # domain -> preferred engine name
        self._lock = threading.Lock()
        self._availability_cache: Dict[str, Tuple[bool, float]] = {}  # name -> (available, timestamp)
        self._cache_ttl = 30.0  # seconds

    def register(self, engine: EngineInfo):
        """Register an engine backend."""
        with self._lock:
            if engine.domain not in self._engines:
                self._engines[engine.domain] = {}
            self._engines[engine.domain][engine.name] = engine
            logger.debug("Registered engine: %s/%s", engine.domain, engine.name)

    def unregister(self, domain: str, name: str):
        """Remove a registered engine."""
        with self._lock:
            if domain in self._engines:
                self._engines[domain].pop(name, None)

    def get_engines(self, domain: str) -> List[EngineInfo]:
        """Get all registered engines for a domain, sorted by priority."""
        with self._lock:
            engines = list(self._engines.get(domain, {}).values())
        return sorted(engines, key=lambda e: e.priority, reverse=True)

    def get_available_engines(self, domain: str) -> List[EngineInfo]:
        """Get only available engines for a domain, sorted by priority."""
        import time
        engines = self.get_engines(domain)
        available = []
        now = time.monotonic()

        for e in engines:
            # Check cache
            with self._lock:
                cached = self._availability_cache.get(e.name)
            if cached and (now - cached[1]) < self._cache_ttl:
                if cached[0]:
                    available.append(e)
                continue

            # Check availability (outside lock — check_fn may be slow)
            is_avail = e.is_available
            with self._lock:
                self._availability_cache[e.name] = (is_avail, now)
            if is_avail:
                available.append(e)

        return available

    def get_engine(self, domain: str, name: str) -> Optional[EngineInfo]:
        """Get a specific engine by domain and name."""
        with self._lock:
            return self._engines.get(domain, {}).get(name)

    def set_preference(self, domain: str, engine_name: str):
        """Set the user's preferred engine for a domain."""
        with self._lock:
            self._preferences[domain] = engine_name
        logger.info("Engine preference set: %s -> %s", domain, engine_name)

    def clear_preference(self, domain: str):
        """Clear any preferred engine for a domain and return to automatic selection."""
        with self._lock:
            self._preferences.pop(domain, None)
        logger.info("Engine preference cleared: %s", domain)

    def get_preference(self, domain: str) -> Optional[str]:
        """Get the user's preferred engine for a domain."""
        with self._lock:
            return self._preferences.get(domain)

    def resolve_engine(self, domain: str, requested: Optional[str] = None) -> Optional[EngineInfo]:
        """
        Resolve the best engine to use for a domain.

        Priority order:
        1. Explicitly requested engine (if available)
        2. User's preferred engine (if available)
        3. Highest-priority available engine

        Returns:
            EngineInfo or None if no engine available.
        """
        available = self.get_available_engines(domain)
        if not available:
            return None

        available_names = {e.name for e in available}

        # 1. Explicitly requested
        if requested and requested in available_names:
            return next(e for e in available if e.name == requested)

        # 2. User preference
        pref = self.get_preference(domain)
        if pref and pref in available_names:
            return next(e for e in available if e.name == pref)

        # 3. Highest priority
        return available[0]

    def get_all_domains(self) -> List[str]:
        """Get all registered domains."""
        with self._lock:
            return list(self._engines.keys())

    def get_status(self) -> Dict[str, Any]:
        """Get full registry status for the /health endpoint."""
        status = {}
        for domain in self.get_all_domains():
            engines = self.get_engines(domain)
            domain_status = []
            for e in engines:
                domain_status.append({
                    "name": e.name,
                    "display_name": e.display_name,
                    "available": e.is_available,
                    "stub": e.is_stub,
                    "priority": e.priority,
                    "vram_mb": e.vram_mb,
                    "speed": e.speed_rating,
                    "quality": e.quality_rating,
                })
            active_engine = self.resolve_engine(domain)
            status[domain] = {
                "engines": domain_status,
                "preferred": self.get_preference(domain),
                "active": active_engine.name if active_engine else None,
            }
        return status

    def load_preferences(self, prefs: Dict[str, str]):
        """Load user preferences from settings."""
        with self._lock:
            self._preferences.update(prefs)
        logger.info("Loaded engine preferences: %s", prefs)

    def export_preferences(self) -> Dict[str, str]:
        """Export current preferences for saving."""
        with self._lock:
            return dict(self._preferences)

    def clear_cache(self):
        """Clear the availability cache."""
        with self._lock:
            self._availability_cache.clear()


# ---------------------------------------------------------------------------
# Global registry singleton
# ---------------------------------------------------------------------------

_registry: Optional[EngineRegistry] = None
_registry_lock = threading.Lock()


def get_registry() -> EngineRegistry:
    """Get the global engine registry (creates on first call)."""
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = EngineRegistry()
            _register_builtin_engines(_registry)
        return _registry


def _register_builtin_engines(reg: EngineRegistry):
    """Register all built-in engine backends."""
    from opencut.checks import (
        check_autoshot_available,
        check_crisper_whisper_available,
        check_deepface_available,
        check_demucs_available,
        check_depth_available,
        check_diarization_available,
        check_edge_tts_available,
        check_iclight_available,
        check_latentsync_available,
        check_nemo_asr_available,
        check_rembg_available,
        check_rvm_available,
        check_sam2_available,
        check_seedvr2_available,
        check_silero_vad_available,
        check_sortformer_available,
        check_transnetv2_available,
        check_upscale_available,
    )

    # --- Silence Detection ---
    reg.register(EngineInfo(
        name="energy",
        domain="silence_detection",
        display_name="Energy Threshold",
        description="Simple energy-based silence detection (no deps required)",
        check_fn=lambda: True,
        priority=30,
        vram_mb=0,
        speed_rating="fast",
        quality_rating="low",
    ))
    reg.register(EngineInfo(
        name="silero_vad",
        domain="silence_detection",
        display_name="Silero VAD",
        description="Neural voice activity detection (87%+ accuracy)",
        check_fn=check_silero_vad_available,
        priority=80,
        vram_mb=50,
        speed_rating="fast",
        quality_rating="high",
    ))

    # --- Transcription ---
    reg.register(EngineInfo(
        name="faster_whisper",
        domain="transcription",
        display_name="Faster Whisper",
        description="CTranslate2-optimized Whisper (fast, accurate)",
        check_fn=lambda: importlib.util.find_spec("faster_whisper") is not None,
        priority=70,
        vram_mb=500,
        speed_rating="fast",
        quality_rating="high",
    ))
    reg.register(EngineInfo(
        name="crisper_whisper",
        domain="transcription",
        display_name="CrisperWhisper",
        description="Verbatim ASR with filler markers (highest accuracy)",
        check_fn=check_crisper_whisper_available,
        priority=90,
        vram_mb=2000,
        speed_rating="slow",
        quality_rating="high",
        tags=["fillers", "verbatim"],
    ))
    reg.register(EngineInfo(
        name="parakeet_tdt",
        domain="transcription",
        display_name="Parakeet TDT 0.6B v3",
        description="Pinned NVIDIA NeMo multilingual ASR with word and segment timestamps",
        check_fn=check_nemo_asr_available,
        priority=85,
        vram_mb=2400,
        speed_rating="fast",
        quality_rating="high",
        tags=["timestamps", "nemo", "multilingual"],
        impl_module="asr_parakeet",
    ))
    reg.register(EngineInfo(
        name="canary_1b_flash",
        domain="transcription",
        display_name="Canary 1B Flash",
        description="Pinned NVIDIA NeMo batch ASR and four-language translation",
        check_fn=check_nemo_asr_available,
        priority=80,
        vram_mb=4200,
        speed_rating="fast",
        quality_rating="high",
        tags=["batch", "nemo", "translation"],
        impl_module="asr_canary",
    ))

    # --- Background Removal ---
    reg.register(EngineInfo(
        name="rembg",
        domain="background_removal",
        display_name="rembg",
        description="Per-frame background removal (U2-Net)",
        check_fn=check_rembg_available,
        priority=50,
        vram_mb=300,
        speed_rating="medium",
        quality_rating="medium",
    ))
    reg.register(EngineInfo(
        name="rvm",
        domain="background_removal",
        display_name="Robust Video Matting",
        description="Temporally consistent video background removal",
        check_fn=check_rvm_available,
        priority=80,
        vram_mb=500,
        speed_rating="medium",
        quality_rating="high",
        tags=["temporal"],
    ))

    # --- Scene Detection ---
    reg.register(EngineInfo(
        name="threshold",
        domain="scene_detection",
        display_name="Threshold",
        description="FFmpeg-based threshold scene detection",
        check_fn=lambda: True,
        priority=30,
        vram_mb=0,
        speed_rating="fast",
        quality_rating="low",
    ))
    reg.register(EngineInfo(
        name="transnetv2",
        domain="scene_detection",
        display_name="TransNetV2",
        description="Neural scene detection (catches fades/dissolves)",
        check_fn=check_transnetv2_available,
        priority=70,
        vram_mb=200,
        speed_rating="medium",
        quality_rating="high",
    ))
    reg.register(EngineInfo(
        name="autoshot",
        domain="scene_detection",
        display_name="AutoShot",
        description="Gradual-transition-aware shot boundary detection (~4% F1 over TransNetV2)",
        check_fn=check_autoshot_available,
        priority=85,
        vram_mb=500,
        speed_rating="medium",
        quality_rating="high",
        tags=["gradual-transitions"],
    ))

    # --- TTS ---
    reg.register(EngineInfo(
        name="edge_tts",
        domain="tts",
        display_name="Edge TTS",
        description="Microsoft Edge text-to-speech (cloud, fast)",
        check_fn=check_edge_tts_available,
        priority=40,
        vram_mb=0,
        speed_rating="fast",
        quality_rating="medium",
    ))
    reg.register(EngineInfo(
        name="kokoro",
        domain="tts",
        display_name="Kokoro",
        description="Local TTS (82M params, <0.3s generation)",
        check_fn=lambda: importlib.util.find_spec("kokoro") is not None,
        priority=70,
        vram_mb=200,
        speed_rating="fast",
        quality_rating="high",
    ))
    reg.register(EngineInfo(
        name="chatterbox",
        domain="tts",
        display_name="Chatterbox",
        description="Voice cloning TTS (15s reference audio)",
        check_fn=lambda: importlib.util.find_spec("chatterbox") is not None,
        priority=60,
        vram_mb=1000,
        speed_rating="slow",
        quality_rating="high",
        tags=["voice_clone"],
    ))

    # --- Stem Separation ---
    reg.register(EngineInfo(
        name="demucs",
        domain="stem_separation",
        display_name="Demucs",
        description="Meta's hybrid stem separator",
        check_fn=check_demucs_available,
        priority=60,
        vram_mb=1000,
        speed_rating="medium",
        quality_rating="high",
    ))
    reg.register(EngineInfo(
        name="bs_roformer",
        domain="stem_separation",
        display_name="BS-RoFormer",
        description="State-of-art Roformer-based separator",
        check_fn=lambda: importlib.util.find_spec("audio_separator") is not None,
        priority=80,
        vram_mb=1500,
        speed_rating="slow",
        quality_rating="high",
    ))

    # --- Depth Estimation ---
    reg.register(EngineInfo(
        name="depth_anything_v2",
        domain="depth_estimation",
        display_name="Depth Anything V2",
        description="Monocular depth estimation for effects",
        check_fn=check_depth_available,
        priority=80,
        vram_mb=800,
        speed_rating="medium",
        quality_rating="high",
    ))

    # --- Upscaling ---
    reg.register(EngineInfo(
        name="seedvr2",
        domain="upscaling",
        display_name="SeedVR2 (one-step diffusion)",
        description="Apache-2.0 one-step diffusion VSR; ~10x faster than multi-step, beats Real-ESRGAN. Falls back to Real-ESRGAN when unavailable.",
        check_fn=check_seedvr2_available,
        priority=90,  # preferred over Real-ESRGAN (80) when its weights are installed
        vram_mb=8000,
        speed_rating="medium",
        quality_rating="high",
        tags=["diffusion", "apache-2.0", "vsr"],
        impl_module="upscale_seedvr2",  # terminal stub — listed but never active
    ))
    reg.register(EngineInfo(
        name="realesrgan",
        domain="upscaling",
        display_name="Real-ESRGAN",
        description="GAN-based video/image upscaling",
        check_fn=check_upscale_available,
        priority=80,
        vram_mb=1500,
        speed_rating="slow",
        quality_rating="high",
    ))

    # --- Lip Sync ---
    # Heuristic MediaPipe jaw-overlay is the always-available default; LatentSync
    # is a higher-quality diffusion engine kept OPT-IN (lower priority, so it is
    # never auto-selected) until its checkpoint licence is confirmed.
    reg.register(EngineInfo(
        name="mediapipe_jaw",
        domain="lip_sync",
        display_name="Heuristic (MediaPipe jaw)",
        description="Audio-driven jaw/mouth overlay; no model download, graceful fallback",
        check_fn=lambda: True,
        priority=50,
        vram_mb=0,
        speed_rating="fast",
        quality_rating="low",
    ))
    reg.register(EngineInfo(
        name="latentsync",
        domain="lip_sync",
        display_name="LatentSync (diffusion)",
        description="Audio-conditioned latent-diffusion lip-sync (opt-in: code Apache-2.0, checkpoint licence unconfirmed). Falls back to heuristic when unavailable.",
        check_fn=check_latentsync_available,
        priority=40,  # below heuristic so it is never auto-selected (opt-in only)
        vram_mb=6000,
        speed_rating="slow",
        quality_rating="high",
        tags=["diffusion", "opt-in", "dubbing"],
        impl_module="lipsync_latentsync",  # terminal stub — listed but never active
    ))

    # --- Speaker Diarization ---
    # community-1 (CC-BY-4.0, lower DER, always freely accessible) is the default;
    # legacy 3.1 is retained as the auto-fallback. Sortformer (NVIDIA NeMo) is an
    # optional engine, available only when NeMo is installed.
    reg.register(EngineInfo(
        name="pyannote_community1",
        domain="diarization",
        display_name="pyannote community-1",
        description="pyannote/speaker-diarization-community-1 (CC-BY-4.0, lower DER) — default",
        check_fn=check_diarization_available,
        priority=80,
        vram_mb=2000,
        speed_rating="medium",
        quality_rating="high",
        tags=["cc-by-4.0", "default"],
    ))
    reg.register(EngineInfo(
        name="pyannote_legacy",
        domain="diarization",
        display_name="pyannote 3.1 (legacy)",
        description="pyannote/speaker-diarization-3.1 — retained fallback",
        check_fn=check_diarization_available,
        priority=60,
        vram_mb=2000,
        speed_rating="medium",
        quality_rating="medium",
    ))
    reg.register(EngineInfo(
        name="sortformer",
        domain="diarization",
        display_name="NVIDIA Sortformer (NeMo)",
        description="NVIDIA Sortformer end-to-end diarization — optional, requires NeMo",
        check_fn=check_sortformer_available,
        priority=50,
        vram_mb=4000,
        speed_rating="medium",
        quality_rating="high",
        tags=["nemo", "optional"],
        impl_module="diarize_sortformer",  # adapter not written yet — never active
    ))

    # --- Relighting ---
    # IC-Light v1 is Apache-2.0 with real public weights (MIT-distribution-safe).
    # IC-Light v2 is intentionally not registered (non-commercial, weights
    # never publicly released).
    reg.register(EngineInfo(
        name="iclight",
        domain="relight",
        display_name="IC-Light v1",
        description="Apache-2.0 per-frame relight (text- or background-conditioned). Gated by availability.",
        check_fn=check_iclight_available,
        priority=70,
        vram_mb=4000,
        speed_rating="slow",
        quality_rating="high",
        tags=["diffusion", "apache-2.0"],
        impl_module="relight_iclight",  # terminal stub — listed but never active
    ))

    # --- Object Removal ---
    reg.register(EngineInfo(
        name="sam2_propainter",
        domain="object_removal",
        display_name="SAM2 + ProPainter",
        description="Click-to-select + temporal inpainting",
        check_fn=check_sam2_available,
        priority=80,
        vram_mb=2000,
        speed_rating="slow",
        quality_rating="high",
    ))

    # --- Emotion Analysis ---
    reg.register(EngineInfo(
        name="deepface",
        domain="emotion_analysis",
        display_name="DeepFace",
        description="Facial emotion recognition for highlights",
        check_fn=check_deepface_available,
        priority=70,
        vram_mb=500,
        speed_rating="medium",
        quality_rating="medium",
    ))

    logger.info(
        "Engine registry initialized: %d engines across %d domains",
        sum(len(engines) for engines in reg._engines.values()),
        len(reg._engines),
    )
