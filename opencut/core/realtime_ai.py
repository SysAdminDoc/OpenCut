"""
OpenCut Real-Time AI Processing Pipeline (Feature 21.4)

Apply AI effects in real-time during preview (10-15 FPS at reduced
resolution).  Uses ONNX Runtime for fast single-frame inference and
FFmpeg for frame extraction.

Supported models:
  - style_transfer   — neural style transfer
  - background_removal — remove / replace background
  - face_enhance      — face restoration / enhancement
  - color_grade       — automatic colour grading
  - denoise           — AI denoising

Frame extraction via FFmpeg single-frame seeks.
Recent frames cached to avoid reprocessing.
Resolution scaling: run at 1/4 or 1/2 resolution for speed.
"""

import base64
import hashlib
import io
import logging
import os
import subprocess
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from opencut.helpers import get_ffmpeg_path, get_video_info

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.expanduser("~"), ".opencut", "models", "realtime")
DEFAULT_TARGET_FPS = 12
DEFAULT_RESOLUTION_SCALE = 0.5    # half resolution by default
MAX_CACHE_FRAMES = 100
FRAME_EXTRACT_TIMEOUT = 10        # seconds

# Available real-time models with metadata
REALTIME_MODELS: Dict[str, Dict[str, Any]] = {
    "style_transfer": {
        "label": "Style Transfer",
        "description": "Apply neural style transfer in real-time",
        "default_params": {"style": "candy", "intensity": 0.8},
        "supports_params": ["style", "intensity"],
        "min_resolution_scale": 0.25,
        "estimated_fps": 12,
    },
    "background_removal": {
        "label": "Background Removal",
        "description": "Remove or replace video background in real-time",
        "default_params": {"threshold": 0.5, "replace_color": "#00FF00"},
        "supports_params": ["threshold", "replace_color", "blur_radius"],
        "min_resolution_scale": 0.25,
        "estimated_fps": 15,
    },
    "face_enhance": {
        "label": "Face Enhancement",
        "description": "Enhance and restore faces in real-time",
        "default_params": {"strength": 0.7, "upscale": False},
        "supports_params": ["strength", "upscale"],
        "min_resolution_scale": 0.5,
        "estimated_fps": 10,
    },
    "color_grade": {
        "label": "Color Grade",
        "description": "Apply automatic colour grading",
        "default_params": {"preset": "cinematic", "intensity": 0.6},
        "supports_params": ["preset", "intensity"],
        "min_resolution_scale": 0.25,
        "estimated_fps": 20,
    },
    "denoise": {
        "label": "Denoise",
        "description": "AI-powered noise reduction for preview",
        "default_params": {"strength": 0.5},
        "supports_params": ["strength"],
        "min_resolution_scale": 0.5,
        "estimated_fps": 15,
    },
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class RealtimeConfig:
    """Configuration for a real-time preview session."""
    model: str = "style_transfer"
    resolution_scale: float = DEFAULT_RESOLUTION_SCALE
    target_fps: float = DEFAULT_TARGET_FPS
    device: str = "cpu"            # cpu | cuda | dml
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PreviewFrame:
    """A single processed preview frame."""
    frame_data_b64: str = ""       # base64-encoded PNG/JPEG bytes
    timestamp: float = 0.0
    processing_ms: float = 0.0
    width: int = 0
    height: int = 0
    model_name: str = ""
    cached: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RealtimeSession:
    """A live real-time preview session."""
    session_id: str = ""
    model_name: str = ""
    video_path: str = ""
    resolution: Tuple[int, int] = (960, 540)
    fps: float = DEFAULT_TARGET_FPS
    running: bool = False
    config: Optional[RealtimeConfig] = None
    created: float = 0.0
    frame_count: int = 0
    last_frame_time: float = 0.0
    video_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "session_id": self.session_id,
            "model_name": self.model_name,
            "video_path": self.video_path,
            "resolution": list(self.resolution),
            "fps": self.fps,
            "running": self.running,
            "created": self.created,
            "frame_count": self.frame_count,
            "last_frame_time": self.last_frame_time,
            "video_info": self.video_info,
        }
        if self.config:
            d["config"] = self.config.to_dict()
        return d


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_sessions: Dict[str, RealtimeSession] = {}
_sessions_lock = threading.Lock()

# Frame cache: keyed by (session_id, timestamp_rounded, params_hash)
_frame_cache: OrderedDict = OrderedDict()
_cache_lock = threading.Lock()

# ONNX session cache (model_name -> InferenceSession)
_onnx_sessions: Dict[str, Any] = {}
_onnx_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Frame cache helpers
# ---------------------------------------------------------------------------
def _cache_key(session_id: str, timestamp: float, params: dict) -> str:
    """Build a deterministic cache key."""
    ts_rounded = round(timestamp, 2)
    params_str = json.dumps(params, sort_keys=True) if params else ""
    raw = f"{session_id}:{ts_rounded}:{params_str}"
    return hashlib.md5(raw.encode()).hexdigest()


def _cache_get(key: str) -> Optional[PreviewFrame]:
    """Return cached frame or None."""
    with _cache_lock:
        if key in _frame_cache:
            _frame_cache.move_to_end(key)
            return _frame_cache[key]
    return None


def _cache_put(key: str, frame: PreviewFrame):
    """Insert frame into cache, evicting oldest if over limit."""
    with _cache_lock:
        _frame_cache[key] = frame
        _frame_cache.move_to_end(key)
        while len(_frame_cache) > MAX_CACHE_FRAMES:
            _frame_cache.popitem(last=False)


def _cache_clear_session(session_id: str):
    """Remove all cached frames for a session."""
    with _cache_lock:
        [k for k, v in _frame_cache.items()
                          if hasattr(v, "model_name")]
        # Simple approach: clear keys that start with session prefix
        # Since cache keys are hashes, we track session_id in PreviewFrame
        # and compare.  For efficiency we just clear all — sessions are
        # typically short-lived.
        to_del = []
        for k in list(_frame_cache.keys()):
            if k.startswith(session_id[:8]):
                to_del.append(k)
        for k in to_del:
            del _frame_cache[k]


# Need json for cache key generation
import json  # noqa: E402  (after initial imports for readability)


# ---------------------------------------------------------------------------
# Frame extraction via FFmpeg
# ---------------------------------------------------------------------------
def _extract_frame_raw(video_path: str, timestamp: float,
                       width: int, height: int) -> bytes:
    """Extract a single frame from a video at the given timestamp.

    Uses FFmpeg single-frame seek for efficiency.
    Returns raw RGB24 pixel data.
    """
    cmd = [
        get_ffmpeg_path(),
        "-ss", f"{timestamp:.3f}",
        "-i", video_path,
        "-vframes", "1",
        "-s", f"{width}x{height}",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-v", "quiet",
        "pipe:1",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, timeout=FRAME_EXTRACT_TIMEOUT,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg frame extraction failed: {result.stderr.decode(errors='replace')[-200:]}"
            )
        return result.stdout
    except subprocess.TimeoutExpired:
        raise RuntimeError("FFmpeg frame extraction timed out")


def _frame_to_png_b64(raw_rgb: bytes, width: int, height: int) -> str:
    """Convert raw RGB24 bytes to base64-encoded PNG.

    Uses a minimal PNG encoder to avoid a hard dependency on PIL/numpy
    at module level.  Falls back to returning raw base64 if encoding fails.
    """
    try:
        import numpy as np
        from PIL import Image
        arr = np.frombuffer(raw_rgb, dtype=np.uint8).reshape((height, width, 3))
        img = Image.fromarray(arr, "RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=False)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except ImportError:
        # Fallback: return raw RGB as base64 (client can decode knowing dimensions)
        return base64.b64encode(raw_rgb).decode("ascii")


# ---------------------------------------------------------------------------
# ONNX model helpers
# ---------------------------------------------------------------------------
def _get_onnx_session(model_name: str, device: str = "cpu"):
    """Load or return cached ONNX Runtime InferenceSession.

    Returns None if ONNX Runtime is unavailable or model not found.
    """
    with _onnx_lock:
        cache_key = f"{model_name}:{device}"
        if cache_key in _onnx_sessions:
            return _onnx_sessions[cache_key]

    try:
        import onnxruntime as ort
    except ImportError:
        logger.debug("onnxruntime not available — AI preview will use fallback")
        return None

    model_path = os.path.join(MODELS_DIR, f"{model_name}.onnx")
    if not os.path.isfile(model_path):
        logger.debug("ONNX model not found at %s", model_path)
        return None

    providers = []
    if device == "cuda":
        providers.append("CUDAExecutionProvider")
    elif device == "dml":
        providers.append("DmlExecutionProvider")
    providers.append("CPUExecutionProvider")

    try:
        session = ort.InferenceSession(model_path, providers=providers)
        with _onnx_lock:
            _onnx_sessions[f"{model_name}:{device}"] = session
        return session
    except Exception as exc:
        logger.warning("Failed to load ONNX model %s: %s", model_name, exc)
        return None


def _run_onnx_inference(session, frame_rgb: bytes, width: int,
                        height: int, params: dict) -> bytes:
    """Run ONNX inference on a raw RGB frame.

    Returns processed raw RGB bytes.
    """
    try:
        import numpy as np
    except ImportError:
        # Can't process without numpy — return original frame
        return frame_rgb

    arr = np.frombuffer(frame_rgb, dtype=np.uint8).reshape((height, width, 3))
    # Normalise to float32 CHW format (batch=1)
    input_tensor = arr.astype(np.float32).transpose(2, 0, 1)
    input_tensor = np.expand_dims(input_tensor, axis=0)

    input_name = session.get_inputs()[0].name
    try:
        outputs = session.run(None, {input_name: input_tensor})
        result = outputs[0]

        # Convert back to HWC uint8
        if result.ndim == 4:
            result = result[0]
        if result.shape[0] == 3:
            result = result.transpose(1, 2, 0)
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result.tobytes()
    except Exception as exc:
        logger.debug("ONNX inference failed: %s — returning original frame", exc)
        return frame_rgb


def _apply_fallback_effect(frame_rgb: bytes, width: int, height: int,
                           model_name: str, params: dict) -> bytes:
    """Apply a simple fallback effect when ONNX is unavailable.

    Provides basic visual feedback without a real model.
    """
    try:
        import numpy as np
    except ImportError:
        return frame_rgb

    arr = np.frombuffer(frame_rgb, dtype=np.uint8).reshape((height, width, 3)).copy()
    intensity = float(params.get("intensity", 0.7))

    if model_name == "style_transfer":
        # Simple colour tint as placeholder
        tint = np.array([30, -20, 40], dtype=np.float32)
        arr = np.clip(arr.astype(np.float32) + tint * intensity, 0, 255).astype(np.uint8)

    elif model_name == "background_removal":
        # Slight green tint to hint at chroma-key
        arr[:, :, 1] = np.clip(
            arr[:, :, 1].astype(np.float32) + 30 * intensity, 0, 255
        ).astype(np.uint8)

    elif model_name == "face_enhance":
        # Slight sharpening effect (increase contrast around edges)
        strength = float(params.get("strength", 0.7))
        arr = np.clip(
            arr.astype(np.float32) * (1.0 + 0.1 * strength), 0, 255
        ).astype(np.uint8)

    elif model_name == "color_grade":
        # Simple warm/cool shift
        preset = params.get("preset", "cinematic")
        if preset == "cinematic":
            arr[:, :, 0] = np.clip(
                arr[:, :, 0].astype(np.float32) + 10 * intensity, 0, 255
            ).astype(np.uint8)
            arr[:, :, 2] = np.clip(
                arr[:, :, 2].astype(np.float32) - 10 * intensity, 0, 255
            ).astype(np.uint8)

    elif model_name == "denoise":
        # Simple averaging blur as placeholder
        pass  # Blur requires more than trivial code; skip for fallback

    return arr.tobytes()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def create_session(model_name: str, video_path: str,
                   config: Optional[RealtimeConfig] = None,
                   on_progress: Optional[Callable] = None) -> RealtimeSession:
    """Create a new real-time preview session.

    Args:
        model_name: One of the supported model names.
        video_path: Path to the video file.
        config: Optional configuration override.

    Returns:
        ``RealtimeSession`` ready for frame requests.
    """
    if on_progress:
        on_progress(10, "Creating preview session")

    if model_name not in REALTIME_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}.  "
            f"Available: {', '.join(REALTIME_MODELS.keys())}"
        )

    if not os.path.isfile(video_path):
        raise ValueError(f"Video file not found: {video_path}")

    # Get video metadata
    vinfo = get_video_info(video_path)
    src_w = vinfo.get("width", 1920)
    src_h = vinfo.get("height", 1080)

    if config is None:
        model_meta = REALTIME_MODELS[model_name]
        config = RealtimeConfig(
            model=model_name,
            resolution_scale=DEFAULT_RESOLUTION_SCALE,
            target_fps=model_meta.get("estimated_fps", DEFAULT_TARGET_FPS),
            params=dict(model_meta.get("default_params", {})),
        )

    # Clamp resolution scale
    min_scale = REALTIME_MODELS[model_name].get("min_resolution_scale", 0.25)
    scale = max(min_scale, min(1.0, config.resolution_scale))

    # Compute preview resolution (even numbers)
    prev_w = max(2, int(src_w * scale) // 2 * 2)
    prev_h = max(2, int(src_h * scale) // 2 * 2)

    session_id = uuid.uuid4().hex[:12]
    session = RealtimeSession(
        session_id=session_id,
        model_name=model_name,
        video_path=video_path,
        resolution=(prev_w, prev_h),
        fps=config.target_fps,
        running=True,
        config=config,
        created=time.time(),
        video_info=vinfo,
    )

    with _sessions_lock:
        _sessions[session_id] = session

    if on_progress:
        on_progress(100, f"Session {session_id} created ({prev_w}x{prev_h})")

    return session


def get_session(session_id: str) -> Optional[RealtimeSession]:
    """Return a session by ID, or None."""
    with _sessions_lock:
        return _sessions.get(session_id)


def list_sessions() -> List[RealtimeSession]:
    """Return all active sessions."""
    with _sessions_lock:
        return [s for s in _sessions.values() if s.running]


def get_preview_frame(session_id: str, timestamp: float,
                      on_progress: Optional[Callable] = None) -> PreviewFrame:
    """Extract a frame, run the AI model, and return the result.

    Args:
        session_id: Active session ID.
        timestamp: Video timestamp in seconds.

    Returns:
        ``PreviewFrame`` with base64-encoded image data.
    """
    with _sessions_lock:
        session = _sessions.get(session_id)
    if session is None:
        raise ValueError(f"Session not found: {session_id}")
    if not session.running:
        raise ValueError(f"Session {session_id} is stopped")

    params = session.config.params if session.config else {}
    width, height = session.resolution

    # Check cache
    ck = _cache_key(session_id, timestamp, params)
    cached = _cache_get(ck)
    if cached is not None:
        cached.cached = True
        return cached

    if on_progress:
        on_progress(10, "Extracting frame")

    t_start = time.time()

    # Extract raw frame
    raw_rgb = _extract_frame_raw(session.video_path, timestamp, width, height)
    expected_size = width * height * 3
    if len(raw_rgb) != expected_size:
        raise RuntimeError(
            f"Frame size mismatch: got {len(raw_rgb)}, expected {expected_size}"
        )

    if on_progress:
        on_progress(40, "Running AI model")

    # Run model inference
    onnx_sess = _get_onnx_session(session.model_name,
                                   device=session.config.device if session.config else "cpu")
    if onnx_sess is not None:
        processed_rgb = _run_onnx_inference(onnx_sess, raw_rgb, width, height, params)
    else:
        processed_rgb = _apply_fallback_effect(raw_rgb, width, height,
                                                session.model_name, params)

    if on_progress:
        on_progress(80, "Encoding frame")

    # Encode to base64 PNG
    frame_b64 = _frame_to_png_b64(processed_rgb, width, height)

    t_end = time.time()
    processing_ms = round((t_end - t_start) * 1000, 1)

    frame = PreviewFrame(
        frame_data_b64=frame_b64,
        timestamp=timestamp,
        processing_ms=processing_ms,
        width=width,
        height=height,
        model_name=session.model_name,
        cached=False,
    )

    # Cache the result
    _cache_put(ck, frame)

    # Update session stats
    with _sessions_lock:
        if session_id in _sessions:
            _sessions[session_id].frame_count += 1
            _sessions[session_id].last_frame_time = timestamp

    if on_progress:
        on_progress(100, f"Frame ready ({processing_ms:.0f}ms)")

    return frame


def update_params(session_id: str, params: dict,
                  on_progress: Optional[Callable] = None) -> RealtimeSession:
    """Update effect parameters for a running session.

    Invalidates the frame cache for this session so new parameters
    take effect immediately.
    """
    with _sessions_lock:
        session = _sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")
        if not session.running:
            raise ValueError(f"Session {session_id} is stopped")

        if session.config is None:
            model_meta = REALTIME_MODELS.get(session.model_name, {})
            session.config = RealtimeConfig(
                model=session.model_name,
                params=dict(model_meta.get("default_params", {})),
            )

        # Merge new params
        session.config.params.update(params)

    # Invalidate cache for this session
    _cache_clear_session(session_id)

    if on_progress:
        on_progress(100, "Parameters updated")

    return session


def stop_session(session_id: str,
                 on_progress: Optional[Callable] = None) -> bool:
    """Stop a running preview session and clean up resources.

    Returns True if the session existed and was stopped.
    """
    with _sessions_lock:
        session = _sessions.get(session_id)
        if session is None:
            return False
        session.running = False

    # Clean up cached frames
    _cache_clear_session(session_id)

    if on_progress:
        on_progress(100, f"Session {session_id} stopped")

    return True


def remove_session(session_id: str) -> bool:
    """Remove a session entirely from the registry."""
    with _sessions_lock:
        if session_id in _sessions:
            del _sessions[session_id]
            return True
    return False


def list_realtime_models() -> List[Dict[str, Any]]:
    """Return metadata for all available real-time models."""
    result = []
    for name, meta in REALTIME_MODELS.items():
        entry = dict(meta)
        entry["name"] = name
        # Check if ONNX model file exists
        model_path = os.path.join(MODELS_DIR, f"{name}.onnx")
        entry["installed"] = os.path.isfile(model_path)
        result.append(entry)
    return result


def get_cache_stats() -> dict:
    """Return frame cache statistics."""
    with _cache_lock:
        return {
            "size": len(_frame_cache),
            "max_size": MAX_CACHE_FRAMES,
        }


def clear_cache():
    """Clear the entire frame cache."""
    with _cache_lock:
        _frame_cache.clear()
