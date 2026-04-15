"""
OpenCut Multi-Render Module (70.4)

Simultaneous multi-format render with priority ordering, parallel
execution, per-render progress tracking, GPU/CPU resource allocation,
individual render cancellation, and resume support.
"""

import logging
import os
import subprocess as _sp
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import (
    get_ffmpeg_path,
    get_ffprobe_path,
    get_video_info,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Render Status Enum
# ---------------------------------------------------------------------------

class RenderStatus:
    """Render job status constants."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


# ---------------------------------------------------------------------------
# Codec / Hardware Detection
# ---------------------------------------------------------------------------

_GPU_CODECS = frozenset({
    "h264_nvenc", "hevc_nvenc", "h264_amf", "hevc_amf",
    "h264_qsv", "hevc_qsv", "av1_nvenc", "av1_amf", "av1_qsv",
    "h264_videotoolbox", "hevc_videotoolbox",
})

_CPU_CODECS = frozenset({
    "libx264", "libx265", "libvpx", "libvpx-vp9", "libaom-av1",
    "libsvtav1", "libopenjpeg", "prores_ks", "dnxhd",
    "mpeg2video", "mjpeg",
})


def _is_gpu_codec(codec: str) -> bool:
    """Check if a codec uses GPU acceleration."""
    return codec.lower() in _GPU_CODECS


def _detect_hw_acceleration() -> Dict[str, bool]:
    """Detect available hardware acceleration."""
    hw = {"nvenc": False, "amf": False, "qsv": False, "videotoolbox": False}
    try:
        cmd = [get_ffmpeg_path(), "-hide_banner", "-encoders"]
        result = _sp.run(cmd, capture_output=True, timeout=10)
        output = result.stdout.decode(errors="replace")
        if "h264_nvenc" in output:
            hw["nvenc"] = True
        if "h264_amf" in output:
            hw["amf"] = True
        if "h264_qsv" in output:
            hw["qsv"] = True
        if "h264_videotoolbox" in output:
            hw["videotoolbox"] = True
    except Exception:
        pass
    return hw


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RenderConfig:
    """Configuration for a single render target."""
    name: str = ""
    format: str = "mp4"            # Output container format
    video_codec: str = "libx264"   # FFmpeg video codec
    audio_codec: str = "aac"       # FFmpeg audio codec
    width: int = 0                 # 0 = keep original
    height: int = 0                # 0 = keep original
    frame_rate: float = 0.0        # 0 = keep original
    video_bitrate: str = ""        # e.g. "10M", "5000k", or empty for CRF
    audio_bitrate: str = "192k"
    crf: int = 18
    preset: str = "medium"
    pixel_format: str = "yuv420p"
    audio_sample_rate: int = 0     # 0 = keep original
    audio_channels: int = 0        # 0 = keep original
    priority: int = 0              # Higher = rendered first
    output_suffix: str = ""        # Appended to filename
    output_dir: str = ""           # Override output directory
    extra_ffmpeg_args: List[str] = field(default_factory=list)
    two_pass: bool = False
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def is_gpu(self) -> bool:
        """Check if this config uses GPU-accelerated codec."""
        return _is_gpu_codec(self.video_codec)


@dataclass
class RenderJob:
    """State of a single render within a multi-render batch."""
    render_id: str = ""
    config_name: str = ""
    status: str = RenderStatus.PENDING
    progress: float = 0.0
    message: str = ""
    output_path: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    elapsed_seconds: float = 0.0
    file_size: int = 0
    error: str = ""
    is_gpu: bool = False
    resume_frame: int = 0          # For resume support

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MultiRenderResult:
    """Result of a multi-render operation."""
    total_renders: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
    total_elapsed: float = 0.0
    renders: List[RenderJob] = field(default_factory=list)
    source_file: str = ""
    all_succeeded: bool = False

    def to_dict(self) -> dict:
        return {
            "total_renders": self.total_renders,
            "completed": self.completed,
            "failed": self.failed,
            "cancelled": self.cancelled,
            "total_elapsed": round(self.total_elapsed, 2),
            "renders": [r.to_dict() for r in self.renders],
            "source_file": self.source_file,
            "all_succeeded": self.all_succeeded,
        }


# ---------------------------------------------------------------------------
# Render Cancel Registry
# ---------------------------------------------------------------------------

_cancel_lock = threading.Lock()
_cancelled_renders: Dict[str, bool] = {}
_active_processes: Dict[str, _sp.Popen] = {}


def cancel_render(render_id: str) -> bool:
    """Cancel a specific render by ID.

    Args:
        render_id: The render ID to cancel.

    Returns:
        True if cancel was registered, False if render not found/already done.
    """
    with _cancel_lock:
        _cancelled_renders[render_id] = True
        proc = _active_processes.get(render_id)
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                logger.info("Terminated render process: %s", render_id)
                return True
            except OSError:
                pass
    return render_id in _cancelled_renders


def _is_cancelled(render_id: str) -> bool:
    """Check if a render has been cancelled."""
    with _cancel_lock:
        return _cancelled_renders.get(render_id, False)


def _register_process(render_id: str, proc: _sp.Popen):
    """Register a subprocess for potential cancellation."""
    with _cancel_lock:
        _active_processes[render_id] = proc


def _unregister_process(render_id: str):
    """Unregister a subprocess after completion."""
    with _cancel_lock:
        _active_processes.pop(render_id, None)
        _cancelled_renders.pop(render_id, None)


# ---------------------------------------------------------------------------
# Single Render Execution
# ---------------------------------------------------------------------------

def _build_ffmpeg_cmd(
    video_path: str,
    config: RenderConfig,
    out_path: str,
    source_info: dict,
    pass_num: int = 0,
    resume_frame: int = 0,
) -> list:
    """Build FFmpeg command for a single render config."""
    cmd = [get_ffmpeg_path(), "-y"]

    # Resume support: seek to frame
    if resume_frame > 0:
        fps = source_info.get("fps", 30)
        seek_time = resume_frame / max(fps, 1)
        cmd.extend(["-ss", f"{seek_time:.3f}"])

    cmd.extend(["-i", video_path])

    # Video filters
    vf_parts = []
    if config.width > 0 and config.height > 0:
        vf_parts.append(f"scale={config.width}:{config.height}:flags=lanczos")
    elif config.width > 0:
        vf_parts.append(f"scale={config.width}:-2:flags=lanczos")
    elif config.height > 0:
        vf_parts.append(f"scale=-2:{config.height}:flags=lanczos")

    if config.frame_rate > 0:
        vf_parts.append(f"fps={config.frame_rate}")

    if vf_parts:
        cmd.extend(["-vf", ",".join(vf_parts)])

    # Video codec
    cmd.extend(["-c:v", config.video_codec])

    if config.pixel_format and pass_num != 1:
        cmd.extend(["-pix_fmt", config.pixel_format])

    # Bitrate or CRF
    if config.video_bitrate:
        cmd.extend(["-b:v", config.video_bitrate])
    elif config.crf >= 0 and not _is_gpu_codec(config.video_codec):
        cmd.extend(["-crf", str(config.crf)])

    # Preset
    if config.preset:
        if _is_gpu_codec(config.video_codec):
            cmd.extend(["-preset", config.preset])
        else:
            cmd.extend(["-preset", config.preset])

    # Two-pass handling
    if config.two_pass and pass_num > 0:
        cmd.extend(["-pass", str(pass_num)])
        passlog = out_path + "_passlog"
        cmd.extend(["-passlogfile", passlog])

    # Audio
    if pass_num == 1:
        cmd.extend(["-an"])
    else:
        cmd.extend(["-c:a", config.audio_codec])
        if config.audio_bitrate:
            cmd.extend(["-b:a", config.audio_bitrate])
        if config.audio_sample_rate > 0:
            cmd.extend(["-ar", str(config.audio_sample_rate)])
        if config.audio_channels > 0:
            cmd.extend(["-ac", str(config.audio_channels)])

    # Extra args
    cmd.extend(config.extra_ffmpeg_args)

    # Output
    if pass_num == 1:
        cmd.extend(["-f", "null", os.devnull])
    else:
        cmd.append(out_path)

    return cmd


def _get_duration_frames(video_path: str, source_info: dict) -> int:
    """Get total frames in source video for progress tracking."""
    cmd = [
        get_ffprobe_path(), "-v", "quiet",
        "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames",
        "-of", "csv=p=0", video_path,
    ]
    try:
        result = _sp.run(cmd, capture_output=True, timeout=120)
        frames = int(result.stdout.decode().strip())
        if frames > 0:
            return frames
    except (ValueError, _sp.TimeoutExpired):
        pass

    # Estimate from duration * fps
    duration = source_info.get("duration", 0)
    fps = source_info.get("fps", 30)
    return int(duration * fps) if duration > 0 else 0


def _execute_single_render(
    video_path: str,
    config: RenderConfig,
    source_info: dict,
    render_job: RenderJob,
    on_progress: Optional[Callable] = None,
) -> RenderJob:
    """Execute a single render, updating RenderJob in place."""
    render_job.status = RenderStatus.RUNNING
    render_job.start_time = time.time()

    if on_progress:
        on_progress(render_job.progress, f"Starting: {config.name}")

    try:
        # Build output path
        suffix = config.output_suffix or config.name.replace(" ", "_").lower()
        ext = f".{config.format}" if config.format else ".mp4"
        base = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = config.output_dir or os.path.dirname(video_path)
        out_path = os.path.join(out_dir, f"{base}_{suffix}{ext}")
        render_job.output_path = out_path

        os.makedirs(out_dir, exist_ok=True)

        # Two-pass encoding
        if config.two_pass:
            # Pass 1
            if _is_cancelled(render_job.render_id):
                render_job.status = RenderStatus.CANCELLED
                return render_job

            cmd_p1 = _build_ffmpeg_cmd(
                video_path, config, out_path, source_info,
                pass_num=1, resume_frame=render_job.resume_frame,
            )
            logger.info("Multi-render pass 1: %s", config.name)
            proc = _sp.Popen(cmd_p1, stdout=_sp.PIPE, stderr=_sp.PIPE)
            _register_process(render_job.render_id, proc)
            _, stderr = proc.communicate(timeout=7200)
            _unregister_process(render_job.render_id)

            if _is_cancelled(render_job.render_id):
                render_job.status = RenderStatus.CANCELLED
                return render_job

            if proc.returncode != 0:
                raise RuntimeError(
                    f"Pass 1 failed: {stderr.decode(errors='replace')[-500:]}"
                )
            render_job.progress = 40.0
            render_job.message = "Pass 1 complete"

            # Pass 2
            cmd_p2 = _build_ffmpeg_cmd(
                video_path, config, out_path, source_info,
                pass_num=2, resume_frame=render_job.resume_frame,
            )
            logger.info("Multi-render pass 2: %s", config.name)
            proc = _sp.Popen(cmd_p2, stdout=_sp.PIPE, stderr=_sp.PIPE)
            _register_process(render_job.render_id, proc)
            _, stderr = proc.communicate(timeout=7200)
            _unregister_process(render_job.render_id)

            if _is_cancelled(render_job.render_id):
                render_job.status = RenderStatus.CANCELLED
                return render_job

            if proc.returncode != 0:
                raise RuntimeError(
                    f"Pass 2 failed: {stderr.decode(errors='replace')[-500:]}"
                )

            # Clean up passlog files
            for ext_suffix in ("", "-0.log", "-0.log.mbtree"):
                passlog = out_path + f"_passlog{ext_suffix}"
                if os.path.isfile(passlog):
                    try:
                        os.unlink(passlog)
                    except OSError:
                        pass

        else:
            # Single-pass encoding
            cmd = _build_ffmpeg_cmd(
                video_path, config, out_path, source_info,
                resume_frame=render_job.resume_frame,
            )
            logger.info("Multi-render: %s -> %s", config.name, out_path)

            proc = _sp.Popen(
                cmd, stdout=_sp.PIPE, stderr=_sp.PIPE,
            )
            _register_process(render_job.render_id, proc)

            # Monitor stderr for progress
            total_frames = _get_duration_frames(video_path, source_info)
            stderr_data = b""
            while proc.poll() is None:
                if _is_cancelled(render_job.render_id):
                    proc.terminate()
                    render_job.status = RenderStatus.CANCELLED
                    _unregister_process(render_job.render_id)
                    return render_job

                chunk = proc.stderr.read(4096) if proc.stderr else b""
                stderr_data += chunk

                # Parse frame= for progress
                if total_frames > 0 and chunk:
                    text = chunk.decode(errors="replace")
                    import re
                    m = re.search(r"frame=\s*(\d+)", text)
                    if m:
                        current_frame = int(m.group(1))
                        pct = min(99.0, (current_frame / total_frames) * 100)
                        render_job.progress = pct
                        render_job.message = (
                            f"Frame {current_frame}/{total_frames}"
                        )
                        if on_progress:
                            on_progress(pct, render_job.message)

                time.sleep(0.1)

            # Read remaining stderr
            remaining = proc.stderr.read() if proc.stderr else b""
            stderr_data += remaining
            _unregister_process(render_job.render_id)

            if _is_cancelled(render_job.render_id):
                render_job.status = RenderStatus.CANCELLED
                return render_job

            if proc.returncode != 0:
                raise RuntimeError(
                    f"Encode failed: "
                    f"{stderr_data.decode(errors='replace')[-1000:]}"
                )

        # Success
        render_job.status = RenderStatus.COMPLETE
        render_job.progress = 100.0
        render_job.message = "Complete"
        try:
            render_job.file_size = os.path.getsize(out_path)
        except OSError:
            render_job.file_size = 0

    except Exception as exc:
        render_job.status = RenderStatus.FAILED
        render_job.error = str(exc)
        render_job.message = f"Failed: {exc}"
        logger.error("Render failed (%s): %s", config.name, exc)

    render_job.end_time = time.time()
    render_job.elapsed_seconds = render_job.end_time - render_job.start_time
    return render_job


# ---------------------------------------------------------------------------
# Multi-Render Orchestrator
# ---------------------------------------------------------------------------

def multi_render(
    video_path: str,
    configs: List[RenderConfig],
    parallel: bool = False,
    max_parallel: int = 3,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Execute multiple renders of a video file.

    Args:
        video_path: Source video path.
        configs: List of RenderConfig for each output format.
        parallel: If True, run CPU renders in parallel.
        max_parallel: Max concurrent renders when parallel=True.
        on_progress: Optional callback ``(percent, message)``.

    Returns:
        MultiRenderResult.to_dict()

    Raises:
        FileNotFoundError: If video_path does not exist.
        ValueError: If configs is empty.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Source video not found: {video_path}")
    if not configs:
        raise ValueError("At least one RenderConfig is required")

    if on_progress:
        on_progress(1, "Probing source video...")

    source_info = get_video_info(video_path)
    batch_start = time.time()

    # Sort by priority (higher first)
    sorted_configs = sorted(configs, key=lambda c: c.priority, reverse=True)

    # Create render jobs
    render_jobs: List[RenderJob] = []
    for cfg in sorted_configs:
        rj = RenderJob(
            render_id=str(uuid.uuid4()),
            config_name=cfg.name or f"render_{len(render_jobs)}",
            is_gpu=cfg.is_gpu(),
        )
        render_jobs.append(rj)

    total = len(render_jobs)
    result = MultiRenderResult(
        total_renders=total,
        renders=render_jobs,
        source_file=video_path,
    )

    if on_progress:
        on_progress(5, f"Starting {total} renders...")

    if parallel:
        _run_parallel(
            video_path, sorted_configs, render_jobs,
            source_info, max_parallel, on_progress, total,
        )
    else:
        _run_sequential(
            video_path, sorted_configs, render_jobs,
            source_info, on_progress, total,
        )

    # Compute summary
    result.completed = sum(
        1 for r in render_jobs if r.status == RenderStatus.COMPLETE
    )
    result.failed = sum(
        1 for r in render_jobs if r.status == RenderStatus.FAILED
    )
    result.cancelled = sum(
        1 for r in render_jobs if r.status == RenderStatus.CANCELLED
    )
    result.total_elapsed = time.time() - batch_start
    result.all_succeeded = result.completed == total

    if on_progress:
        on_progress(100, f"Multi-render complete: {result.completed}/{total}")

    logger.info("Multi-render done: %d/%d succeeded in %.1fs",
                result.completed, total, result.total_elapsed)

    return result.to_dict()


def _run_sequential(
    video_path: str,
    configs: List[RenderConfig],
    jobs: List[RenderJob],
    source_info: dict,
    on_progress: Optional[Callable],
    total: int,
):
    """Execute renders sequentially in priority order."""
    for idx, (cfg, rj) in enumerate(zip(configs, jobs)):
        base_pct = 5 + (idx / total) * 90

        def _progress(pct, msg=""):
            if on_progress:
                overall = base_pct + (pct / 100) * (90 / total)
                on_progress(overall, f"[{idx + 1}/{total}] {cfg.name}: {msg}")

        _execute_single_render(video_path, cfg, source_info, rj, _progress)


def _run_parallel(
    video_path: str,
    configs: List[RenderConfig],
    jobs: List[RenderJob],
    source_info: dict,
    max_parallel: int,
    on_progress: Optional[Callable],
    total: int,
):
    """Execute renders in parallel with resource-aware scheduling.

    GPU renders run sequentially (shared GPU resource).
    CPU renders run in parallel up to max_parallel.
    """
    gpu_items = []
    cpu_items = []

    for cfg, rj in zip(configs, jobs):
        if cfg.is_gpu():
            gpu_items.append((cfg, rj))
        else:
            cpu_items.append((cfg, rj))

    completed_count = 0
    completed_lock = threading.Lock()

    def _track_progress(pct, msg=""):
        nonlocal completed_count
        if on_progress:
            with completed_lock:
                overall = 5 + (completed_count / total) * 90
            on_progress(overall, msg)

    # Run GPU renders sequentially first
    for cfg, rj in gpu_items:
        _execute_single_render(video_path, cfg, source_info, rj, _track_progress)
        with completed_lock:
            completed_count += 1

    # Run CPU renders in parallel
    if cpu_items:
        with ThreadPoolExecutor(max_workers=min(max_parallel, len(cpu_items))) as pool:
            futures: Dict[Future, Tuple] = {}
            for cfg, rj in cpu_items:
                future = pool.submit(
                    _execute_single_render,
                    video_path, cfg, source_info, rj, _track_progress,
                )
                futures[future] = (cfg, rj)

            for future in as_completed(futures):
                cfg, rj = futures[future]
                try:
                    future.result(timeout=7200)
                except Exception as exc:
                    rj.status = RenderStatus.FAILED
                    rj.error = str(exc)
                    logger.error("Parallel render failed (%s): %s", cfg.name, exc)
                with completed_lock:
                    completed_count += 1
                    if on_progress:
                        pct = 5 + (completed_count / total) * 90
                        on_progress(pct, f"Completed: {completed_count}/{total}")


# ---------------------------------------------------------------------------
# Utility: List active render IDs
# ---------------------------------------------------------------------------

def get_active_renders() -> List[str]:
    """Return list of currently active render IDs."""
    with _cancel_lock:
        return [
            rid for rid, proc in _active_processes.items()
            if proc.poll() is None
        ]


def cancel_all_renders() -> int:
    """Cancel all active renders. Returns count of cancelled."""
    with _cancel_lock:
        count = 0
        for rid, proc in _active_processes.items():
            _cancelled_renders[rid] = True
            if proc.poll() is None:
                try:
                    proc.terminate()
                    count += 1
                except OSError:
                    pass
        return count


# Type alias for parallel execution tuple unpacking
from typing import Tuple  # noqa: E402 (already imported above but explicit for clarity)
