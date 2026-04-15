"""
OpenCut GPU Preview Pipeline — GPU-Accelerated Preview Rendering

Manage a preview render queue that processes frames using GPU when
available with automatic CPU fallback.  Supports single-frame and
batch preview generation at evenly-spaced timestamps for timeline
scrubbing.

Pipeline stages: extract -> resize_to_preview -> apply_effects -> encode_jpeg.
GPU detection via nvidia-smi or torch.cuda.
"""

import hashlib
import json
import logging
import os
import queue
import subprocess as _sp
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional

from opencut.helpers import get_ffmpeg_path, get_video_info

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_PREVIEW_WIDTH = 854
DEFAULT_PREVIEW_HEIGHT = 480
MAX_QUEUE_SIZE = 50
FRAME_EXTRACT_TIMEOUT = 30
PIPELINE_TIMEOUT = 300  # 5 min total per batch


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------
_gpu_available: Optional[bool] = None
_gpu_info: dict = {}
_gpu_lock = threading.Lock()


def detect_gpu() -> dict:
    """Detect GPU availability via nvidia-smi then torch.cuda.

    Returns dict with keys: available (bool), device (str), name (str),
    memory_mb (int), method (str).
    """
    global _gpu_available, _gpu_info
    with _gpu_lock:
        if _gpu_available is not None:
            return _gpu_info

        info = {
            "available": False,
            "device": "cpu",
            "name": "",
            "memory_mb": 0,
            "method": "none",
        }

        # Try nvidia-smi first (no Python deps needed)
        try:
            result = _sp.run(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                line = result.stdout.strip().split("\n")[0]
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    info["available"] = True
                    info["device"] = "cuda"
                    info["name"] = parts[0]
                    try:
                        info["memory_mb"] = int(float(parts[1]))
                    except (ValueError, IndexError):
                        info["memory_mb"] = 0
                    info["method"] = "nvidia-smi"
        except (FileNotFoundError, _sp.TimeoutExpired, OSError):
            pass

        # Fallback: try torch.cuda
        if not info["available"]:
            try:
                import torch  # noqa: F811
                if torch.cuda.is_available():
                    info["available"] = True
                    info["device"] = "cuda"
                    info["name"] = torch.cuda.get_device_name(0)
                    mem = torch.cuda.get_device_properties(0).total_mem
                    info["memory_mb"] = int(mem / (1024 * 1024))
                    info["method"] = "torch.cuda"
            except Exception:
                pass

        # Check for FFmpeg hardware accel support
        if not info["available"]:
            try:
                ffmpeg = get_ffmpeg_path()
                result = _sp.run(
                    [ffmpeg, "-hide_banner", "-hwaccels"],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0:
                    accels = result.stdout.lower()
                    if "cuda" in accels or "nvdec" in accels:
                        info["available"] = True
                        info["device"] = "cuda"
                        info["method"] = "ffmpeg_hwaccel"
                    elif "d3d11va" in accels or "dxva2" in accels:
                        info["available"] = True
                        info["device"] = "d3d11va"
                        info["method"] = "ffmpeg_hwaccel"
                    elif "videotoolbox" in accels:
                        info["available"] = True
                        info["device"] = "videotoolbox"
                        info["method"] = "ffmpeg_hwaccel"
            except (FileNotFoundError, _sp.TimeoutExpired, OSError):
                pass

        _gpu_available = info["available"]
        _gpu_info = info
        logger.info("GPU detection: %s", json.dumps(info))
        return info


def reset_gpu_detection() -> None:
    """Clear cached GPU detection result (for testing)."""
    global _gpu_available, _gpu_info
    with _gpu_lock:
        _gpu_available = None
        _gpu_info = {}


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class FramePreview:
    """A single preview frame result."""
    timestamp: float = 0.0
    preview_path: str = ""
    width: int = 0
    height: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PipelineResult:
    """Result of a pipeline batch render."""
    frames: List[FramePreview] = field(default_factory=list)
    gpu_used: bool = False
    device: str = "cpu"
    total_time_ms: float = 0.0
    frames_per_second: float = 0.0
    effect_chain: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["frames"] = [f.to_dict() if hasattr(f, "to_dict") else f
                       for f in self.frames]
        return d


# ---------------------------------------------------------------------------
# Frame extraction (GPU-accelerated when possible)
# ---------------------------------------------------------------------------
def _extract_frame_gpu(video_path: str, timestamp: float,
                       output_path: str, width: int, height: int,
                       gpu_info: dict) -> str:
    """Extract a frame using GPU-accelerated decoding when available."""
    ffmpeg = get_ffmpeg_path()
    device = gpu_info.get("device", "cpu")

    # Build command with hardware acceleration if available
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error"]

    if device == "cuda" and gpu_info.get("method") in ("nvidia-smi", "torch.cuda", "ffmpeg_hwaccel"):
        cmd.extend(["-hwaccel", "cuda"])
    elif device == "d3d11va":
        cmd.extend(["-hwaccel", "d3d11va"])
    elif device == "videotoolbox":
        cmd.extend(["-hwaccel", "videotoolbox"])

    cmd.extend([
        "-ss", str(max(0.0, timestamp)),
        "-i", video_path,
        "-frames:v", "1",
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
               f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
        "-q:v", "2",
        "-y", output_path,
    ])

    result = _sp.run(cmd, capture_output=True, timeout=FRAME_EXTRACT_TIMEOUT)
    if result.returncode != 0:
        # Fallback to CPU if GPU extraction failed
        if device != "cpu":
            logger.warning("GPU frame extraction failed, falling back to CPU")
            return _extract_frame_cpu(video_path, timestamp, output_path,
                                      width, height)
        stderr = result.stderr.decode(errors="replace")[-300:]
        raise RuntimeError(f"Frame extraction failed: {stderr}")
    return output_path


def _extract_frame_cpu(video_path: str, timestamp: float,
                       output_path: str, width: int, height: int) -> str:
    """Extract a frame using CPU-only decoding."""
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-ss", str(max(0.0, timestamp)),
        "-i", video_path,
        "-frames:v", "1",
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
               f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
        "-q:v", "2",
        "-y", output_path,
    ]
    result = _sp.run(cmd, capture_output=True, timeout=FRAME_EXTRACT_TIMEOUT)
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")[-300:]
        raise RuntimeError(f"CPU frame extraction failed: {stderr}")
    return output_path


# ---------------------------------------------------------------------------
# Effect application (pipeline stage)
# ---------------------------------------------------------------------------
def _apply_effect_ffmpeg(input_path: str, output_path: str,
                         effect: str, params: dict) -> str:
    """Apply a single effect via FFmpeg filter on an image file."""
    from opencut.core.live_preview import EFFECTS  # noqa: F811

    if effect not in EFFECTS:
        raise ValueError(f"Unknown effect: {effect}")

    fn = EFFECTS[effect]
    fn(input_path, output_path, params)
    return output_path


def _apply_effect_chain_to_frame(frame_path: str, effects: List[dict],
                                 output_dir: str, frame_id: str) -> str:
    """Apply a list of effects sequentially to one frame.

    Each entry in *effects*: {"effect": "name", "params": {...}}.
    """
    current = frame_path
    for idx, fx in enumerate(effects):
        name = fx.get("effect", "")
        params = fx.get("params", {})
        step_out = os.path.join(output_dir, f"pipe_{frame_id}_s{idx}_{name}.jpg")
        _apply_effect_ffmpeg(current, step_out, name, params)
        # Remove intermediate
        if current != frame_path:
            try:
                os.unlink(current)
            except OSError:
                pass
        current = step_out
    return current


# ---------------------------------------------------------------------------
# Pipeline render queue
# ---------------------------------------------------------------------------
class PreviewPipeline:
    """GPU-accelerated preview rendering pipeline with queue management."""

    def __init__(self, max_workers: int = 2, use_gpu: bool = True):
        self._queue: queue.Queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self._use_gpu = use_gpu
        self._max_workers = max_workers
        self._gpu_info = detect_gpu() if use_gpu else {"available": False, "device": "cpu"}
        self._lock = threading.Lock()
        self._active_count = 0

    @property
    def gpu_info(self) -> dict:
        return dict(self._gpu_info)

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def active_count(self) -> int:
        with self._lock:
            return self._active_count

    def render_frame(
        self,
        video_path: str,
        timestamp: float,
        effects: Optional[List[dict]] = None,
        width: int = DEFAULT_PREVIEW_WIDTH,
        height: int = DEFAULT_PREVIEW_HEIGHT,
        output_dir: str = "",
    ) -> FramePreview:
        """Render a single preview frame with optional effect chain."""
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        if not output_dir:
            output_dir = tempfile.mkdtemp(prefix="opencut_pipe_")
        os.makedirs(output_dir, exist_ok=True)

        frame_id = hashlib.sha256(
            f"{video_path}:{timestamp}:{json.dumps(effects or [], sort_keys=True)}".encode()
        ).hexdigest()[:12]

        frame_path = os.path.join(output_dir, f"pipe_raw_{frame_id}.jpg")

        # Extract frame
        use_gpu = self._use_gpu and self._gpu_info.get("available", False)
        if use_gpu:
            _extract_frame_gpu(video_path, timestamp, frame_path,
                               width, height, self._gpu_info)
        else:
            _extract_frame_cpu(video_path, timestamp, frame_path,
                               width, height)

        # Apply effects if any
        final_path = frame_path
        if effects:
            final_path = _apply_effect_chain_to_frame(
                frame_path, effects, output_dir, frame_id,
            )
            # Clean raw frame if different
            if final_path != frame_path:
                try:
                    os.unlink(frame_path)
                except OSError:
                    pass

        return FramePreview(
            timestamp=timestamp,
            preview_path=final_path,
            width=width,
            height=height,
        )

    def render_batch(
        self,
        video_path: str,
        timestamps: Optional[List[float]] = None,
        num_frames: int = 10,
        effects: Optional[List[dict]] = None,
        width: int = DEFAULT_PREVIEW_WIDTH,
        height: int = DEFAULT_PREVIEW_HEIGHT,
        output_dir: str = "",
        on_progress: Optional[Callable] = None,
    ) -> PipelineResult:
        """Render preview frames at multiple timestamps.

        If *timestamps* is not given, generates *num_frames* evenly
        spaced across the video duration.
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        t0 = time.time()
        if not output_dir:
            output_dir = tempfile.mkdtemp(prefix="opencut_batch_")
        os.makedirs(output_dir, exist_ok=True)

        # Determine timestamps
        if timestamps is None:
            info = get_video_info(video_path)
            duration = info.get("duration", 0)
            if duration <= 0:
                duration = 10.0  # fallback
            num_frames = max(1, min(num_frames, 100))
            if num_frames == 1:
                timestamps = [0.0]
            else:
                step = duration / num_frames
                timestamps = [i * step for i in range(num_frames)]

        use_gpu = self._use_gpu and self._gpu_info.get("available", False)
        frames = []
        errors = []
        total = len(timestamps)
        effect_names = [fx.get("effect", "") for fx in (effects or [])]

        with self._lock:
            self._active_count += 1
        try:
            for idx, ts in enumerate(timestamps):
                try:
                    fp = self.render_frame(
                        video_path, ts, effects=effects,
                        width=width, height=height,
                        output_dir=output_dir,
                    )
                    frames.append(fp)
                except Exception as e:
                    logger.warning("Pipeline frame error at ts=%.2f: %s", ts, e)
                    errors.append(f"ts={ts:.2f}: {e}")

                if on_progress and total > 0:
                    pct = int((idx + 1) / total * 100)
                    on_progress(pct)
        finally:
            with self._lock:
                self._active_count -= 1

        elapsed_ms = (time.time() - t0) * 1000
        fps = (len(frames) / (elapsed_ms / 1000)) if elapsed_ms > 0 else 0

        return PipelineResult(
            frames=frames,
            gpu_used=use_gpu,
            device=self._gpu_info.get("device", "cpu"),
            total_time_ms=round(elapsed_ms, 1),
            frames_per_second=round(fps, 2),
            effect_chain=effect_names,
            errors=errors,
        )

    def status(self) -> dict:
        """Return pipeline status info."""
        return {
            "queue_size": self.queue_size,
            "active_renders": self.active_count,
            "gpu": self._gpu_info,
            "max_workers": self._max_workers,
        }


# ---------------------------------------------------------------------------
# Module-level singleton pipeline
# ---------------------------------------------------------------------------
_default_pipeline: Optional[PreviewPipeline] = None
_pipeline_lock = threading.Lock()


def get_pipeline(use_gpu: bool = True) -> PreviewPipeline:
    """Return (or create) the default preview pipeline singleton."""
    global _default_pipeline
    with _pipeline_lock:
        if _default_pipeline is None:
            _default_pipeline = PreviewPipeline(use_gpu=use_gpu)
        return _default_pipeline


def reset_pipeline() -> None:
    """Reset the pipeline singleton (for testing)."""
    global _default_pipeline
    with _pipeline_lock:
        _default_pipeline = None


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------
def render_single_preview(
    video_path: str,
    timestamp: float = 0.0,
    effects: Optional[List[dict]] = None,
    width: int = DEFAULT_PREVIEW_WIDTH,
    height: int = DEFAULT_PREVIEW_HEIGHT,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Render a single preview frame and return result dict."""
    pipeline = get_pipeline()
    fp = pipeline.render_frame(video_path, timestamp, effects=effects,
                               width=width, height=height)
    if on_progress:
        on_progress(100)
    return fp.to_dict()


def render_scrub_previews(
    video_path: str,
    num_frames: int = 10,
    effects: Optional[List[dict]] = None,
    width: int = DEFAULT_PREVIEW_WIDTH,
    height: int = DEFAULT_PREVIEW_HEIGHT,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Generate preview frames at evenly-spaced timestamps for scrubbing."""
    pipeline = get_pipeline()
    result = pipeline.render_batch(
        video_path, num_frames=num_frames, effects=effects,
        width=width, height=height, on_progress=on_progress,
    )
    return result.to_dict()


def pipeline_status() -> dict:
    """Return current pipeline status."""
    pipeline = get_pipeline()
    return pipeline.status()
