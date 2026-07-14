"""
Smart Render / Partial Re-Encode.

Re-encode only changed segments and stream-copy unchanged GOPs
for dramatically faster exports on incremental edits.
"""

import json
import logging
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from opencut.helpers import (
    FFmpegCmd,
    get_ffprobe_path,
    get_video_info,
    output_path,
    run_ffmpeg,
    write_concat_list,
)

logger = logging.getLogger("opencut")


@dataclass
class ChangedSegment:
    """A segment of video that needs re-encoding."""
    start: float
    end: float
    change_type: str = ""     # modified | inserted | deleted | effect
    description: str = ""


@dataclass
class SmartRenderPlan:
    """Plan for smart rendering a video."""
    total_duration: float
    changed_segments: List[ChangedSegment] = field(default_factory=list)
    copy_segments: List[Tuple[float, float]] = field(default_factory=list)
    encode_duration: float = 0.0
    copy_duration: float = 0.0
    estimated_speedup: float = 1.0
    keyframes: List[float] = field(default_factory=list)


@dataclass
class MediaStreamProfile:
    """Relevant ffprobe fields for one stream."""

    index: int
    codec_type: str
    codec_name: str
    start_time: float = 0.0
    duration: float = 0.0
    time_base: str = ""
    avg_frame_rate: str = ""
    r_frame_rate: str = ""


@dataclass
class MediaProfile:
    """Container and stream invariants used for preflight and validation."""

    duration: float
    start_time: float
    format_name: str
    streams: List[MediaStreamProfile] = field(default_factory=list)


@dataclass
class SmartRenderPreflight:
    """Compatibility decision made before any output artifact is created."""

    eligible: bool
    source_duration: float
    source_start_time: float
    source_format: str
    stream_map: List[Dict[str, Any]]
    keyframe_count: int
    requested_encoder: str
    requested_codec: str
    fallback_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "eligible": self.eligible,
            "source_duration": self.source_duration,
            "source_start_time": self.source_start_time,
            "source_format": self.source_format,
            "stream_map": self.stream_map,
            "keyframe_count": self.keyframe_count,
            "requested_encoder": self.requested_encoder,
            "requested_codec": self.requested_codec,
            "fallback_reasons": list(self.fallback_reasons),
        }


class SmartRenderValidationError(RuntimeError):
    """Raised when a staged render fails media-integrity checks."""


_ENCODER_CODECS = {
    "libx264": "h264",
    "h264_nvenc": "h264",
    "h264_qsv": "h264",
    "h264_amf": "h264",
    "libx265": "hevc",
    "hevc_nvenc": "hevc",
    "hevc_qsv": "hevc",
    "hevc_amf": "hevc",
    "libsvtav1": "av1",
    "libaom-av1": "av1",
    "av1_nvenc": "av1",
    "av1_qsv": "av1",
    "prores_ks": "prores",
}


def _get_keyframes(video_path: str) -> List[float]:
    """Extract keyframe timestamps from a video."""
    cmd = [
        get_ffprobe_path(),
        "-loglevel", "error",
        "-select_streams", "v:0",
        "-show_entries", "packet=pts_time,flags",
        "-of", "csv=p=0",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe keyframe scan failed: {result.stderr[-500:]}")
    keyframes = []
    for line in result.stdout.strip().split("\n"):
        parts = line.split(",")
        if len(parts) >= 2 and "K" in parts[1]:
            try:
                keyframes.append(float(parts[0]))
            except (ValueError, IndexError):
                pass
    return sorted(keyframes)


def _snap_to_keyframe(timestamp: float, keyframes: List[float], direction: str = "before") -> float:
    """Snap a timestamp to the nearest keyframe."""
    if not keyframes:
        return timestamp
    if direction == "before":
        candidates = [k for k in keyframes if k <= timestamp]
        return max(candidates) if candidates else keyframes[0]
    else:
        candidates = [k for k in keyframes if k >= timestamp]
        return min(candidates) if candidates else timestamp


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def detect_changed_segments(
    video_path: str,
    edit_history: List[Dict],
    on_progress: Optional[Callable] = None,
) -> SmartRenderPlan:
    """Analyze edit history to determine which segments need re-encoding.

    Args:
        video_path: Path to the source video.
        edit_history: List of edit operations, each with 'type', 'start', 'end'.

    Returns:
        SmartRenderPlan with changed and copy segments.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not edit_history:
        raise ValueError("Edit history cannot be empty")

    if on_progress:
        on_progress(10, "Analyzing video structure")

    info = get_video_info(video_path)
    duration = info.get("duration", 0)
    if duration <= 0:
        raise ValueError("Could not determine video duration")

    if on_progress:
        on_progress(30, "Extracting keyframes")

    keyframes = _get_keyframes(video_path)

    if on_progress:
        on_progress(50, "Computing changed segments")

    # Parse edit history into changed segments
    changed: List[ChangedSegment] = []
    for edit in edit_history:
        start = float(edit.get("start", 0))
        end = float(edit.get("end", start + 1))
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError("Edit start/end must be finite numbers")
        if start < 0 or end <= start or end > duration:
            raise ValueError(
                f"Invalid edit range {start:.3f}-{end:.3f}; expected 0 <= start < end <= {duration:.3f}"
            )
        change_type = edit.get("type", "modified")
        description = edit.get("description", "")

        # Snap to keyframe boundaries for clean cuts
        snap_start = _snap_to_keyframe(start, keyframes, "before")
        snap_end = _snap_to_keyframe(end, keyframes, "after")

        changed.append(ChangedSegment(
            start=snap_start,
            end=snap_end,
            change_type=change_type,
            description=description,
        ))

    # Merge overlapping segments
    changed.sort(key=lambda s: s.start)
    merged = []
    for seg in changed:
        if merged and seg.start <= merged[-1].end:
            merged[-1].end = max(merged[-1].end, seg.end)
        else:
            merged.append(seg)

    if on_progress:
        on_progress(70, "Computing copy segments")

    # Determine copy segments (gaps between changed segments)
    copy_segments = []
    prev_end = 0.0
    for seg in merged:
        if seg.start > prev_end:
            copy_segments.append((prev_end, seg.start))
        prev_end = seg.end
    if prev_end < duration:
        copy_segments.append((prev_end, duration))

    encode_dur = sum(s.end - s.start for s in merged)
    copy_dur = sum(e - s for s, e in copy_segments)
    speedup = duration / encode_dur if encode_dur > 0 else float("inf")

    if on_progress:
        on_progress(100, "Analysis complete")

    return SmartRenderPlan(
        total_duration=duration,
        changed_segments=merged,
        copy_segments=copy_segments,
        encode_duration=encode_dur,
        copy_duration=copy_dur,
        estimated_speedup=round(speedup, 2),
        keyframes=keyframes,
    )


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else default
    except (TypeError, ValueError):
        return default


def _probe_media(path: str) -> MediaProfile:
    """Probe container/stream facts; unlike get_video_info, fail closed."""
    command = [
        get_ffprobe_path(),
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-show_entries",
        (
            "stream=index,codec_type,codec_name,start_time,duration,time_base,"
            "avg_frame_rate,r_frame_rate:format=start_time,duration,format_name"
        ),
        "-of",
        "json",
        path,
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SmartRenderValidationError(f"ffprobe failed for {path}: {exc}") from exc
    if result.returncode != 0:
        raise SmartRenderValidationError(f"ffprobe rejected {path}: {result.stderr[-500:]}")
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise SmartRenderValidationError(f"ffprobe returned invalid JSON for {path}") from exc

    streams = []
    for item in data.get("streams", []):
        streams.append(
            MediaStreamProfile(
                index=int(item.get("index", len(streams))),
                codec_type=str(item.get("codec_type") or "unknown"),
                codec_name=str(item.get("codec_name") or "unknown"),
                start_time=_float_value(item.get("start_time")),
                duration=_float_value(item.get("duration")),
                time_base=str(item.get("time_base") or ""),
                avg_frame_rate=str(item.get("avg_frame_rate") or ""),
                r_frame_rate=str(item.get("r_frame_rate") or ""),
            )
        )
    fmt = data.get("format", {})
    duration = _float_value(fmt.get("duration"))
    if duration <= 0:
        duration = max((stream.duration for stream in streams), default=0.0)
    if duration <= 0 or not streams:
        raise SmartRenderValidationError(f"No playable streams/duration found in {path}")
    return MediaProfile(
        duration=duration,
        start_time=_float_value(fmt.get("start_time")),
        format_name=str(fmt.get("format_name") or ""),
        streams=streams,
    )


def _frame_rate_is_vfr(stream: MediaStreamProfile) -> bool:
    """Conservatively flag differing declared/average rates as VFR."""
    avg = stream.avg_frame_rate
    nominal = stream.r_frame_rate
    return bool(avg and nominal and avg != "0/0" and nominal != "0/0" and avg != nominal)


def _build_preflight(
    video_path: str,
    plan: SmartRenderPlan,
    encoder: str,
) -> Tuple[SmartRenderPreflight, MediaProfile]:
    source = _probe_media(video_path)
    requested_codec = _ENCODER_CODECS.get(encoder, "")
    if not requested_codec:
        raise ValueError(f"Unsupported smart-render encoder: {encoder!r}")

    videos = [stream for stream in source.streams if stream.codec_type == "video"]
    audios = [stream for stream in source.streams if stream.codec_type == "audio"]
    subtitles = [stream for stream in source.streams if stream.codec_type == "subtitle"]
    other = [
        stream
        for stream in source.streams
        if stream.codec_type not in {"video", "audio", "subtitle"}
    ]
    reasons = []
    if len(videos) != 1:
        reasons.append(f"partial render requires exactly one video stream (found {len(videos)})")
    elif videos[0].codec_name != requested_codec:
        reasons.append(
            f"copied video is {videos[0].codec_name}, requested encoder produces {requested_codec}"
        )
    elif requested_codec not in {"h264", "hevc"}:
        reasons.append(f"{requested_codec} partial segments are not approved for MPEG-TS concat")
    if any(stream.codec_name != "aac" for stream in audios):
        reasons.append("copied audio is not uniformly AAC")
    if subtitles:
        reasons.append("subtitle streams require whole-file preservation")
    if other:
        reasons.append("data/attachment streams require whole-file preservation")
    if videos and _frame_rate_is_vfr(videos[0]):
        reasons.append("variable-frame-rate input requires whole-file timestamp normalization")
    if not plan.keyframes:
        reasons.append("no keyframes were available for safe copied boundaries")

    stream_map = [
        {
            "index": stream.index,
            "type": stream.codec_type,
            "codec": stream.codec_name,
            "time_base": stream.time_base,
            "start_time": stream.start_time,
            "duration": stream.duration,
        }
        for stream in source.streams
    ]
    return (
        SmartRenderPreflight(
            eligible=not reasons,
            source_duration=source.duration,
            source_start_time=source.start_time,
            source_format=source.format_name,
            stream_map=stream_map,
            keyframe_count=len(plan.keyframes),
            requested_encoder=encoder,
            requested_codec=requested_codec,
            fallback_reasons=reasons,
        ),
        source,
    )


def _stream_counts(profile: MediaProfile) -> Dict[str, int]:
    counts = {"video": 0, "audio": 0, "subtitle": 0}
    for stream in profile.streams:
        if stream.codec_type in counts:
            counts[stream.codec_type] += 1
    return counts


def _sample_packet_timestamps(path: str, interval: str) -> Dict[int, List[float]]:
    command = [
        get_ffprobe_path(),
        "-v",
        "error",
        "-read_intervals",
        interval,
        "-show_packets",
        "-show_entries",
        "packet=stream_index,pts_time,dts_time",
        "-of",
        "json",
        path,
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=90)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SmartRenderValidationError(f"timestamp probe failed: {exc}") from exc
    if result.returncode != 0:
        raise SmartRenderValidationError(f"timestamp probe failed: {result.stderr[-500:]}")
    try:
        packets = json.loads(result.stdout).get("packets", [])
    except json.JSONDecodeError as exc:
        raise SmartRenderValidationError("timestamp probe returned invalid JSON") from exc

    timestamps: Dict[int, List[float]] = {}
    for packet in packets:
        value = packet.get("dts_time", packet.get("pts_time"))
        try:
            timestamp = float(value)
            index = int(packet["stream_index"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(timestamp):
            timestamps.setdefault(index, []).append(timestamp)
    return timestamps


def _validate_packet_timestamps(path: str, profile: MediaProfile, tolerance: float) -> int:
    window = min(3.0, profile.duration)
    intervals = [f"%+{window:.3f}"]
    if profile.duration > window:
        intervals.append(f"{max(0.0, profile.duration - window):.3f}%+{window:.3f}")

    sampled = 0
    seen_av = set()
    for interval in intervals:
        by_stream = _sample_packet_timestamps(path, interval)
        for index, timestamps in by_stream.items():
            if not timestamps:
                continue
            sampled += len(timestamps)
            stream = next((item for item in profile.streams if item.index == index), None)
            if stream and stream.codec_type in {"video", "audio"}:
                seen_av.add(index)
            previous = timestamps[0]
            for timestamp in timestamps:
                if timestamp < -tolerance or timestamp > profile.duration + tolerance:
                    raise SmartRenderValidationError(
                        f"stream {index} timestamp {timestamp:.3f}s is outside the output duration"
                    )
                if timestamp + 0.001 < previous:
                    raise SmartRenderValidationError(
                        f"stream {index} timestamps are not monotonic ({previous:.3f} -> {timestamp:.3f})"
                    )
                previous = timestamp

    expected_av = {
        stream.index for stream in profile.streams if stream.codec_type in {"video", "audio"}
    }
    if not expected_av.issubset(seen_av):
        missing = sorted(expected_av - seen_av)
        raise SmartRenderValidationError(f"no timestamp samples for A/V streams {missing}")
    return sampled


def _decode_smoke(path: str, duration: float) -> None:
    window = min(2.0, duration)
    starts = [0.0]
    if duration > window:
        starts.append(max(0.0, duration - window))
    for start in starts:
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-xerror",
            "-ss",
            f"{start:.3f}",
            "-i",
            path,
            "-t",
            f"{window:.3f}",
            "-map",
            "0:v?",
            "-map",
            "0:a?",
            "-f",
            "null",
            os.devnull,
        ]
        run_ffmpeg(command, timeout=120, stderr_cap=2000)


def _validate_render(
    staged_path: str,
    source: MediaProfile,
    requested_codec: str,
) -> Dict[str, Any]:
    output = _probe_media(staged_path)
    source_counts = _stream_counts(source)
    output_counts = _stream_counts(output)
    if output_counts != source_counts:
        raise SmartRenderValidationError(
            f"stream-count mismatch: expected {source_counts}, got {output_counts}"
        )

    tolerance = max(0.25, source.duration * 0.005)
    if abs(output.duration - source.duration) > tolerance:
        raise SmartRenderValidationError(
            f"duration mismatch: expected {source.duration:.3f}s, got {output.duration:.3f}s"
        )
    if abs(output.start_time) > tolerance:
        raise SmartRenderValidationError(
            f"output start time {output.start_time:.3f}s exceeds {tolerance:.3f}s tolerance"
        )

    output_videos = [stream for stream in output.streams if stream.codec_type == "video"]
    if not output_videos or any(stream.codec_name != requested_codec for stream in output_videos):
        codecs = [stream.codec_name for stream in output_videos]
        raise SmartRenderValidationError(
            f"video codec mismatch: expected {requested_codec}, got {codecs}"
        )

    av_streams = [
        stream for stream in output.streams if stream.codec_type in {"video", "audio"}
    ]
    if av_streams:
        starts = [stream.start_time for stream in av_streams]
        if max(starts) - min(starts) > tolerance:
            raise SmartRenderValidationError("A/V stream starts are not aligned")
        ends = [
            stream.start_time + (stream.duration or output.duration) for stream in av_streams
        ]
        if max(ends) - min(ends) > max(tolerance, 0.5):
            raise SmartRenderValidationError("A/V stream ends are not aligned")

    for stream in output.streams:
        if stream.codec_type == "subtitle":
            end = stream.start_time + stream.duration
            if stream.start_time < -tolerance or end > output.duration + tolerance:
                raise SmartRenderValidationError(
                    f"subtitle stream {stream.index} falls outside the output timeline"
                )

    packet_count = _validate_packet_timestamps(staged_path, output, tolerance)
    _decode_smoke(staged_path, output.duration)
    return {
        "stream_counts": output_counts,
        "duration": output.duration,
        "start_time": output.start_time,
        "duration_tolerance": tolerance,
        "sampled_packets": packet_count,
        "decode_smoke": "passed",
    }


def _allocate_staged_output(final_path: str) -> str:
    final = Path(final_path).expanduser().resolve(strict=False)
    parent = final.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {parent}")
    descriptor, staged = tempfile.mkstemp(
        prefix=f".{final.stem}.opencut-",
        suffix=final.suffix or ".mp4",
        dir=str(parent),
    )
    os.close(descriptor)
    os.unlink(staged)
    return staged


def _render_partial(
    video_path: str,
    staged_path: str,
    plan: SmartRenderPlan,
    codec: str,
    crf: int,
    preset: str,
    on_progress: Optional[Callable],
) -> int:
    all_segments = [
        *(('copy', start, end) for start, end in plan.copy_segments),
        *(('encode', segment.start, segment.end) for segment in plan.changed_segments),
    ]
    all_segments.sort(key=lambda segment: segment[1])
    with tempfile.TemporaryDirectory(prefix="opencut_smart_") as temp_dir:
        segment_files = []
        for index, (mode, start, end) in enumerate(all_segments):
            segment_path = os.path.join(temp_dir, f"seg_{index:04d}.ts")
            builder = (
                FFmpegCmd()
                .input(video_path)
                .seek(start=str(start), end=str(end))
                .map("0:v:0", "0:a?")
            )
            if mode == "copy":
                builder.copy_streams()
            else:
                builder.video_codec(codec, crf=crf, preset=preset).audio_codec(
                    "aac", bitrate="192k"
                )
            command = (
                builder.option("avoid_negative_ts", "make_zero")
                .option("reset_timestamps", "1")
                .format("mpegts")
                .output(segment_path)
                .build()
            )
            run_ffmpeg(command, timeout=600)
            segment_files.append(segment_path)
            if on_progress:
                percent = 15 + int(65 * (index + 1) / max(len(all_segments), 1))
                on_progress(percent, f"Segment {index + 1}/{len(all_segments)}")

        concat_path = os.path.join(temp_dir, "concat.txt")
        write_concat_list(segment_files, concat_path)
        concat_builder = (
            FFmpegCmd()
            .pre_input("-f", "concat")
            .pre_input("-safe", "0")
            .input(concat_path)
            .map("0:v?", "0:a?")
            .copy_streams()
        )
        if Path(staged_path).suffix.lower() in {".mp4", ".m4v", ".mov"}:
            concat_builder.faststart()
        command = concat_builder.output(staged_path).build()
        run_ffmpeg(command, timeout=300)
    return len(all_segments)


def _render_full_transcode(
    video_path: str,
    staged_path: str,
    source: MediaProfile,
    codec: str,
    crf: int,
    preset: str,
) -> None:
    builder = (
        FFmpegCmd()
        .input(video_path)
        .map("0:v?", "0:a?", "0:s?")
        .video_codec(codec, crf=crf, preset=preset)
        .audio_codec("aac", bitrate="192k")
        .option("map_metadata", "0")
        .option("map_chapters", "0")
    )
    subtitles = [stream for stream in source.streams if stream.codec_type == "subtitle"]
    if subtitles:
        extension = Path(staged_path).suffix.lower()
        if extension in {".mp4", ".m4v", ".mov"}:
            text_codecs = {"ass", "mov_text", "ssa", "srt", "subrip", "tx3g", "webvtt"}
            if any(stream.codec_name not in text_codecs for stream in subtitles):
                raise SmartRenderValidationError(
                    "Image-based subtitles cannot be preserved in the requested MP4/MOV output"
                )
            builder.option("c:s", "mov_text")
        else:
            builder.option("c:s", "copy")
    if Path(staged_path).suffix.lower() in {".mp4", ".m4v", ".mov"}:
        builder.faststart()
    run_ffmpeg(builder.output(staged_path).build(), timeout=3600)


def smart_render(
    video_path: str,
    changes: List[Dict],
    output_path_str: Optional[str] = None,
    codec: str = "libx264",
    crf: int = 18,
    preset: str = "medium",
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Render transactionally, using partial re-encode only when preflight allows it.

    Args:
        video_path: Path to the source video.
        changes: List of change dicts with 'start', 'end', 'type'.
        output_path_str: Output file path (auto-generated if None).
        codec: Video codec for re-encoded segments.
        crf: Quality factor for re-encoded segments.
        preset: Encoding speed preset.

    Returns:
        Result including preflight, validation, and fallback evidence.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if on_progress:
        on_progress(5, "Planning smart render")

    plan = detect_changed_segments(video_path, changes)
    preflight, source_profile = _build_preflight(video_path, plan, codec)

    if output_path_str is None:
        output_path_str = output_path(video_path, "_smart")
    source_resolved = Path(video_path).expanduser().resolve(strict=True)
    final_resolved = Path(output_path_str).expanduser().resolve(strict=False)
    if source_resolved == final_resolved:
        raise ValueError("Smart-render output cannot replace the source file")

    if on_progress:
        mode = "partial render" if preflight.eligible else "full-transcode fallback"
        on_progress(15, f"Starting {mode}")

    staged_path = _allocate_staged_output(str(final_resolved))
    fallback_used = not preflight.eligible
    fallback_reasons = list(preflight.fallback_reasons)
    total_segments = len(plan.copy_segments) + len(plan.changed_segments)
    validation: Dict[str, Any] = {}
    try:
        if preflight.eligible:
            try:
                total_segments = _render_partial(
                    video_path,
                    staged_path,
                    plan,
                    codec,
                    crf,
                    preset,
                    on_progress,
                )
                if on_progress:
                    on_progress(85, "Validating partial render")
                validation = _validate_render(
                    staged_path, source_profile, preflight.requested_codec
                )
            except Exception as exc:
                fallback_used = True
                fallback_reasons.append(f"partial render failed validation: {exc}")
                logger.warning("Smart-render partial path failed; using fallback: %s", exc)
                try:
                    os.unlink(staged_path)
                except FileNotFoundError:
                    pass

        if fallback_used:
            if on_progress:
                on_progress(35, "Running safe full-transcode fallback")
            _render_full_transcode(
                video_path,
                staged_path,
                source_profile,
                codec,
                crf,
                preset,
            )
            if on_progress:
                on_progress(85, "Validating fallback render")
            validation = _validate_render(
                staged_path, source_profile, preflight.requested_codec
            )

        os.replace(staged_path, final_resolved)
    finally:
        try:
            os.unlink(staged_path)
        except FileNotFoundError:
            pass

    if on_progress:
        on_progress(100, "Smart render complete")

    file_size = final_resolved.stat().st_size

    logger.info(
        "Smart render: %d segments, fallback=%s, %.1fx planned speedup, output=%s",
        total_segments,
        fallback_used,
        plan.estimated_speedup,
        final_resolved,
    )

    return {
        "output_path": str(final_resolved),
        "file_size": file_size,
        "total_segments": total_segments,
        "encoded_segments": len(plan.changed_segments),
        "copied_segments": len(plan.copy_segments),
        "encode_duration": plan.encode_duration,
        "copy_duration": plan.copy_duration,
        "estimated_speedup": plan.estimated_speedup,
        "fallback_used": fallback_used,
        "fallback_reasons": fallback_reasons,
        "preflight": preflight.to_dict(),
        "validation": validation,
    }


def estimate_smart_render_savings(
    video_path: str,
    changes: List[Dict],
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Estimate time and space savings from smart rendering.

    Args:
        video_path: Path to the source video.
        changes: List of change dicts.

    Returns:
        Dict with estimated savings percentages and durations.
    """
    plan = detect_changed_segments(video_path, changes, on_progress=on_progress)

    pct_encode = (plan.encode_duration / plan.total_duration * 100) if plan.total_duration > 0 else 100
    pct_copy = 100 - pct_encode

    return {
        "total_duration": plan.total_duration,
        "encode_duration": plan.encode_duration,
        "copy_duration": plan.copy_duration,
        "percent_reencoded": round(pct_encode, 1),
        "percent_copied": round(pct_copy, 1),
        "estimated_speedup": plan.estimated_speedup,
        "changed_segment_count": len(plan.changed_segments),
        "copy_segment_count": len(plan.copy_segments),
    }
