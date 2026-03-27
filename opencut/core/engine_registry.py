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
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("opencut")


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

    @property
    def is_available(self) -> bool:
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
            cached = self._availability_cache.get(e.name)
            if cached and (now - cached[1]) < self._cache_ttl:
                if cached[0]:
                    available.append(e)
                continue

            # Check availability
            is_avail = e.is_available
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
        check_crisper_whisper_available,
        check_deepface_available,
        check_demucs_available,
        check_depth_available,
        check_edge_tts_available,
        check_rembg_available,
        check_rvm_available,
        check_sam2_available,
        check_silero_vad_available,
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
