"""
OpenCut Render Farm Management

Split long render jobs into time-range segments, dispatch segments to
different render nodes, collect completed segments, and concatenate
into final output.  Supports equal_duration, scene_based, and
chapter_based segmentation strategies.  GPU-intensive segments route
to GPU nodes.  Fault tolerance via re-dispatch of failed segments.
"""

import json
import logging
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from opencut.helpers import (
    FFmpegCmd,
    get_ffmpeg_path,
    get_ffprobe_path,
    get_video_info,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_SEGMENT_RETRIES = 2
MIN_SEGMENT_DURATION = 2.0  # seconds
DEFAULT_SEGMENTS = 4


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class TimeRange:
    """A time range within a video."""
    start: float = 0.0
    end: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
        }


@dataclass
class FarmSegment:
    """A single segment of a farm render job."""
    segment_id: str = ""
    time_range: TimeRange = field(default_factory=TimeRange)
    node: str = ""
    status: str = "pending"  # pending, rendering, complete, failed
    output_path: str = ""
    duration_ms: float = 0.0
    retries: int = 0
    error: str = ""
    requires_gpu: bool = False

    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "time_range": self.time_range.to_dict(),
            "node": self.node,
            "status": self.status,
            "output_path": self.output_path,
            "duration_ms": round(self.duration_ms, 1),
            "retries": self.retries,
            "error": self.error,
            "requires_gpu": self.requires_gpu,
        }


@dataclass
class FarmRenderResult:
    """Result from a render farm job."""
    segments: List[FarmSegment] = field(default_factory=list)
    total_render_time: float = 0.0
    speedup_factor: float = 1.0
    output_path: str = ""
    status: str = ""
    total_segments: int = 0
    completed_segments: int = 0
    failed_segments: int = 0
    strategy: str = ""

    def to_dict(self) -> dict:
        return {
            "segments": [s.to_dict() for s in self.segments],
            "total_render_time": round(self.total_render_time, 1),
            "speedup_factor": round(self.speedup_factor, 2),
            "output_path": self.output_path,
            "status": self.status,
            "total_segments": self.total_segments,
            "completed_segments": self.completed_segments,
            "failed_segments": self.failed_segments,
            "strategy": self.strategy,
        }


# ---------------------------------------------------------------------------
# Segmentation strategies
# ---------------------------------------------------------------------------
def segment_equal_duration(duration: float, num_segments: int) -> List[TimeRange]:
    """Split duration into equal time ranges."""
    if duration <= 0 or num_segments <= 0:
        return []
    seg_duration = duration / num_segments
    if seg_duration < MIN_SEGMENT_DURATION and num_segments > 1:
        # Too many segments for this duration
        num_segments = max(1, int(duration / MIN_SEGMENT_DURATION))
        seg_duration = duration / num_segments

    ranges = []
    for i in range(num_segments):
        start = i * seg_duration
        end = min((i + 1) * seg_duration, duration)
        ranges.append(TimeRange(start=start, end=end))
    return ranges


def segment_scene_based(filepath: str, min_duration: float = 5.0) -> List[TimeRange]:
    """
    Split at scene boundaries detected via FFmpeg scene filter.
    Falls back to equal_duration if scene detection fails.
    """
    info = get_video_info(filepath)
    total_dur = info.get("duration", 0)
    if total_dur <= 0:
        return []

    cmd = [
        get_ffmpeg_path(), "-hide_banner",
        "-i", filepath,
        "-vf", "select='gt(scene,0.3)',showinfo",
        "-f", "null",
        "-",
    ]
    try:
        proc = __import__("subprocess").run(
            cmd, capture_output=True, text=True, timeout=120)
        stderr = proc.stderr
    except Exception as e:
        logger.warning("Scene detection failed, falling back to equal: %s", e)
        return segment_equal_duration(total_dur, DEFAULT_SEGMENTS)

    # Parse scene change timestamps from showinfo output
    timestamps = [0.0]
    import re
    for match in re.finditer(r'pts_time:(\d+\.?\d*)', stderr):
        ts = float(match.group(1))
        if ts > timestamps[-1] + min_duration:
            timestamps.append(ts)

    if len(timestamps) < 2:
        return segment_equal_duration(total_dur, DEFAULT_SEGMENTS)

    # Build ranges
    ranges = []
    for i in range(len(timestamps)):
        start = timestamps[i]
        end = timestamps[i + 1] if i + 1 < len(timestamps) else total_dur
        if end - start >= MIN_SEGMENT_DURATION:
            ranges.append(TimeRange(start=start, end=end))

    if not ranges:
        return segment_equal_duration(total_dur, DEFAULT_SEGMENTS)

    return ranges


def segment_chapter_based(filepath: str) -> List[TimeRange]:
    """
    Split at chapter boundaries from the container metadata.
    Falls back to equal_duration if no chapters found.
    """
    info = get_video_info(filepath)
    total_dur = info.get("duration", 0)
    if total_dur <= 0:
        return []

    # Probe for chapters
    cmd = [
        get_ffprobe_path(), "-v", "quiet",
        "-show_chapters", "-of", "json",
        filepath,
    ]
    try:
        proc = __import__("subprocess").run(
            cmd, capture_output=True, text=True, timeout=30)
        if proc.returncode != 0:
            return segment_equal_duration(total_dur, DEFAULT_SEGMENTS)
        data = json.loads(proc.stdout)
        chapters = data.get("chapters", [])
    except Exception as e:
        logger.debug("Chapter detection failed: %s", e)
        return segment_equal_duration(total_dur, DEFAULT_SEGMENTS)

    if not chapters:
        return segment_equal_duration(total_dur, DEFAULT_SEGMENTS)

    ranges = []
    for ch in chapters:
        start = float(ch.get("start_time", 0))
        end = float(ch.get("end_time", 0))
        if end - start >= MIN_SEGMENT_DURATION:
            ranges.append(TimeRange(start=start, end=end))

    if not ranges:
        return segment_equal_duration(total_dur, DEFAULT_SEGMENTS)

    return ranges


def create_segments(filepath: str, strategy: str = "equal_duration",
                    num_segments: int = DEFAULT_SEGMENTS) -> List[TimeRange]:
    """
    Create time range segments using the specified strategy.

    Args:
        filepath: Path to the input video.
        strategy: "equal_duration", "scene_based", or "chapter_based".
        num_segments: Number of segments for equal_duration strategy.

    Returns:
        List of TimeRange segments.
    """
    info = get_video_info(filepath)
    duration = info.get("duration", 0)

    if duration <= 0:
        raise ValueError("Cannot segment file with zero duration")

    if strategy == "scene_based":
        return segment_scene_based(filepath)
    elif strategy == "chapter_based":
        return segment_chapter_based(filepath)
    else:
        return segment_equal_duration(duration, num_segments)


# ---------------------------------------------------------------------------
# Segment rendering
# ---------------------------------------------------------------------------
def _render_segment(input_path: str, time_range: TimeRange,
                    output_dir: str, segment_id: str,
                    render_config: Optional[dict] = None) -> str:
    """Render a single segment using FFmpeg."""
    config = dict(render_config or {})
    codec = config.get("codec", "libx264")
    crf = config.get("crf", 18)
    preset = config.get("preset", "fast")

    ext = ".mp4"
    seg_path = os.path.join(output_dir, f"segment_{segment_id}{ext}")

    builder = FFmpegCmd()
    builder.input(input_path)
    builder.seek(start=time_range.start, end=time_range.end)
    builder.video_codec(codec, crf=crf, preset=preset)
    builder.audio_codec("aac", bitrate="192k")
    builder.faststart()
    builder.output(seg_path)

    cmd = builder.build()
    run_ffmpeg(cmd)

    return seg_path


def _render_segment_remote(node_name: str, input_path: str,
                           time_range: TimeRange,
                           render_config: Optional[dict] = None) -> str:
    """Dispatch a segment render to a remote node."""
    from opencut.core.cloud_render import dispatch_render

    config = dict(render_config or {})
    config["start_time"] = time_range.start
    config["end_time"] = time_range.end

    result = dispatch_render(input_path, config)
    return result.output_path


# ---------------------------------------------------------------------------
# Concatenation
# ---------------------------------------------------------------------------
def _concatenate_segments(segment_paths: List[str], output_file: str) -> str:
    """Concatenate segment files into a single output using FFmpeg concat."""
    if not segment_paths:
        raise ValueError("No segments to concatenate")

    # Create concat list file
    concat_dir = os.path.dirname(segment_paths[0])
    list_path = os.path.join(concat_dir, "concat_list.txt")

    with open(list_path, "w", encoding="utf-8") as f:
        for sp in segment_paths:
            # Escape single quotes in paths
            escaped = sp.replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")

    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-c", "copy",
        "-movflags", "+faststart",
        output_file,
    ]

    run_ffmpeg(cmd)

    # Cleanup concat list
    try:
        os.unlink(list_path)
    except OSError:
        pass

    return output_file


# ---------------------------------------------------------------------------
# Farm render orchestration
# ---------------------------------------------------------------------------
def _dispatch_segment_to_node(segment: FarmSegment, input_path: str,
                              output_dir: str,
                              render_config: Optional[dict] = None,
                              use_remote: bool = False) -> FarmSegment:
    """Render a single segment, locally or remotely."""
    start_time = time.time()
    try:
        if use_remote and segment.node:
            seg_path = _render_segment_remote(
                segment.node, input_path, segment.time_range, render_config)
        else:
            seg_path = _render_segment(
                input_path, segment.time_range, output_dir,
                segment.segment_id, render_config)

        elapsed_ms = (time.time() - start_time) * 1000
        segment.output_path = seg_path
        segment.status = "complete"
        segment.duration_ms = elapsed_ms

    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        segment.status = "failed"
        segment.error = str(e)[:200]
        segment.duration_ms = elapsed_ms
        logger.warning("Segment %s render failed: %s", segment.segment_id, e)

    return segment


def _assign_nodes(segments: List[FarmSegment],
                  available_nodes: Optional[List[dict]] = None):
    """Assign nodes to segments based on capability matching."""
    if not available_nodes:
        return  # All segments render locally

    node_idx = 0
    for seg in segments:
        if seg.requires_gpu:
            # Find a GPU node
            gpu_nodes = [n for n in available_nodes
                         if "gpu" in n.get("capabilities", [])]
            if gpu_nodes:
                seg.node = gpu_nodes[node_idx % len(gpu_nodes)]["name"]
                node_idx += 1
                continue
        if available_nodes:
            seg.node = available_nodes[node_idx % len(available_nodes)]["name"]
            node_idx += 1


def farm_render(input_path: str,
                strategy: str = "equal_duration",
                num_segments: int = DEFAULT_SEGMENTS,
                render_config: Optional[dict] = None,
                output_dir: str = "",
                output_file: str = "",
                use_remote: bool = False,
                on_progress: Optional[Callable] = None) -> FarmRenderResult:
    """
    Execute a farm render: segment, render in parallel, concatenate.

    Args:
        input_path: Path to input video.
        strategy: Segmentation strategy.
        num_segments: Number of segments (for equal_duration).
        render_config: Render settings dict.
        output_dir: Working directory for segments.
        output_file: Final output file path.
        use_remote: If True, dispatch to remote render nodes.
        on_progress: Progress callback (0-100).

    Returns:
        FarmRenderResult with per-segment details and final output.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    start_time = time.time()
    result = FarmRenderResult(strategy=strategy)

    if on_progress:
        on_progress(2)

    # Set up output directory
    if not output_dir:
        output_dir = tempfile.mkdtemp(prefix="opencut_farm_")
    os.makedirs(output_dir, exist_ok=True)

    if not output_file:
        output_file = output_path(input_path, "farm_render", output_dir)

    # Create segments
    if on_progress:
        on_progress(5)

    time_ranges = create_segments(input_path, strategy, num_segments)
    result.total_segments = len(time_ranges)

    # Build segment objects
    segments: List[FarmSegment] = []
    for tr in time_ranges:
        seg = FarmSegment(
            segment_id=uuid.uuid4().hex[:8],
            time_range=tr,
            status="pending",
        )
        segments.append(seg)

    # Assign nodes if using remote rendering
    if use_remote:
        try:
            from opencut.core.cloud_render import check_all_nodes, load_nodes
            check_all_nodes()
            nodes = load_nodes()
            node_dicts = [{"name": n.name, "capabilities": n.capabilities}
                          for n in nodes if n.enabled]
            _assign_nodes(segments, node_dicts)
        except Exception as e:
            logger.warning("Failed to assign remote nodes, using local: %s", e)
            use_remote = False

    if on_progress:
        on_progress(10)

    # Render segments (sequentially for local, could be threaded for remote)
    completed = 0
    for i, seg in enumerate(segments):
        seg.status = "rendering"
        seg = _dispatch_segment_to_node(seg, input_path, output_dir,
                                        render_config, use_remote)

        # Retry failed segments
        retry_count = 0
        while seg.status == "failed" and retry_count < MAX_SEGMENT_RETRIES:
            retry_count += 1
            seg.retries = retry_count
            seg.status = "rendering"
            logger.info("Retrying segment %s (attempt %d)", seg.segment_id, retry_count + 1)
            seg = _dispatch_segment_to_node(seg, input_path, output_dir,
                                            render_config, use_remote=False)

        if seg.status == "complete":
            completed += 1

        segments[i] = seg

        if on_progress:
            pct = 10 + int((completed / max(result.total_segments, 1)) * 75)
            on_progress(min(pct, 85))

    result.segments = segments
    result.completed_segments = completed
    result.failed_segments = sum(1 for s in segments if s.status == "failed")

    if on_progress:
        on_progress(88)

    # Concatenate completed segments
    completed_paths = [s.output_path for s in segments
                       if s.status == "complete" and s.output_path]

    if not completed_paths:
        result.status = "failed"
        result.total_render_time = (time.time() - start_time) * 1000
        raise RuntimeError("All segments failed to render")

    if len(completed_paths) < len(segments):
        logger.warning("%d of %d segments failed, concatenating available segments",
                       result.failed_segments, len(segments))

    if on_progress:
        on_progress(90)

    try:
        final_path = _concatenate_segments(completed_paths, output_file)
        result.output_path = final_path
        result.status = "complete" if result.failed_segments == 0 else "partial"
    except Exception as e:
        result.status = "failed"
        result.total_render_time = (time.time() - start_time) * 1000
        raise RuntimeError(f"Concatenation failed: {e}") from e

    # Calculate timing
    total_time = (time.time() - start_time) * 1000
    result.total_render_time = total_time

    # Speedup factor: sequential render time vs wall clock
    sum_segment_time = sum(s.duration_ms for s in segments if s.status == "complete")
    if total_time > 0:
        result.speedup_factor = sum_segment_time / total_time

    if on_progress:
        on_progress(100)

    return result


# ---------------------------------------------------------------------------
# Segment cleanup
# ---------------------------------------------------------------------------
def cleanup_segments(segments: List[FarmSegment]):
    """Remove rendered segment files from disk."""
    for seg in segments:
        if seg.output_path and os.path.isfile(seg.output_path):
            try:
                os.unlink(seg.output_path)
            except OSError as e:
                logger.debug("Failed to clean segment %s: %s", seg.segment_id, e)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def estimate_segments(filepath: str, strategy: str = "equal_duration",
                      num_segments: int = DEFAULT_SEGMENTS) -> dict:
    """Preview segmentation without rendering."""
    info = get_video_info(filepath)
    ranges = create_segments(filepath, strategy, num_segments)
    return {
        "total_duration": info.get("duration", 0),
        "strategy": strategy,
        "num_segments": len(ranges),
        "segments": [tr.to_dict() for tr in ranges],
    }
