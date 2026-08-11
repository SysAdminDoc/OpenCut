"""MLT timeline export for Kdenlive and Shotcut.

The two editors consume the same MLT XML graph: source chains/producers feed
an ordered playlist, which is exposed through a tractor.  OpenCut keeps the
graph deliberately small and records source ranges/speed as explicit
properties so an imported project remains inspectable even when an editor
chooses a different UI representation for a timewarp.
"""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..core.silence import TimeSegment
from ..utils.media import MediaInfo, probe
from .premiere import cut_ranges_to_segments

MLT_VERSION = "7.22.0"
DEFAULT_FPS = 25.0
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080


@dataclass(frozen=True)
class MLTClip:
    """A source range plus optional speed and volume automation.

    ``volume_keyframes`` contains ``(position, gain)`` pairs.  Positions are
    seconds relative to the clip when supplied as ``time`` and frame numbers
    when supplied as ``frame``.  Gain is linear (``1.0`` is unity); it is
    converted to dB for the MLT ``volume`` filter.
    """

    source_start: float
    source_end: float
    speed: float = 1.0
    volume_keyframes: tuple[tuple[float, float], ...] = ()
    volume_positions_are_frames: bool = False
    name: str = ""


def _float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _gain_to_db(gain: float) -> str:
    if gain <= 0:
        return "-100"
    return f"{max(-100.0, 20.0 * math.log10(gain)):.6f}".rstrip("0").rstrip(".")


def _coerce_keyframes(raw: Any) -> tuple[tuple[float, float], bool]:
    """Normalize scalar/list keyframes without accepting malformed values."""

    if raw is None:
        return (), False
    if isinstance(raw, Mapping):
        if any(key in raw for key in ("frame", "time", "position", "gain", "value", "volume", "db")):
            raw = [raw]
        else:
            raw = raw.get("keyframes", raw.get("points", raw.get("values", raw)))
    if isinstance(raw, (int, float, str)):
        return ((0.0, _float(raw, 1.0)),), False
    if not isinstance(raw, Sequence) or isinstance(raw, (bytes, bytearray)):
        return (), False

    values: list[tuple[float, float]] = []
    positions_are_frames = False
    for item in raw:
        if isinstance(item, Mapping):
            if "frame" in item:
                position = _float(item.get("frame"), 0.0)
                positions_are_frames = True
            else:
                position = _float(item.get("time", item.get("position", 0.0)), 0.0)
            value = item.get("gain", item.get("value", item.get("volume", 1.0)))
            if "db" in item:
                gain = 10.0 ** (_float(item.get("db"), 0.0) / 20.0)
            else:
                gain = _float(value, 1.0)
        elif isinstance(item, Sequence) and len(item) >= 2:
            position = _float(item[0], 0.0)
            gain = _float(item[1], 1.0)
        else:
            continue
        values.append((max(0.0, position), max(0.0, gain)))
    return tuple(values), positions_are_frames


def _coerce_clip(segment: Any) -> MLTClip | None:
    if isinstance(segment, TimeSegment):
        return MLTClip(float(segment.start), float(segment.end), name=str(segment.label or ""))
    if not isinstance(segment, Mapping):
        return None
    start = _float(segment.get("source_start", segment.get("start", segment.get("in", 0.0))))
    if "source_end" in segment:
        end = _float(segment.get("source_end"), start)
    elif "end" in segment or "out" in segment:
        end = _float(segment.get("end", segment.get("out")), start)
    else:
        end = start + max(0.0, _float(segment.get("duration"), 0.0))
    speed = _float(segment.get("speed", segment.get("rate", 1.0)), 1.0)
    if speed <= 0:
        raise ValueError("MLT clip speed must be greater than zero")
    raw_volume = segment.get("volume_keyframes", segment.get("volume"))
    keyframes, positions_are_frames = _coerce_keyframes(raw_volume)
    return MLTClip(
        source_start=max(0.0, start),
        source_end=max(0.0, end),
        speed=speed,
        volume_keyframes=keyframes,
        volume_positions_are_frames=positions_are_frames,
        name=str(segment.get("name", segment.get("label", "")) or ""),
    )


def _normalize_clips(segments: Iterable[Any]) -> list[MLTClip]:
    clips = []
    for segment in segments:
        clip = _coerce_clip(segment)
        if clip is None or clip.source_end <= clip.source_start:
            continue
        clips.append(clip)
    if not clips:
        raise ValueError("No valid segments to export")
    return clips


def _frame_count(seconds: float, fps: float) -> int:
    return max(0, int(round(max(0.0, seconds) * fps)))


def _timecode(frame: int, fps: float) -> str:
    seconds = max(0.0, frame / fps)
    hours = int(seconds // 3600)
    minutes = int((seconds - hours * 3600) // 60)
    whole = int(seconds - hours * 3600 - minutes * 60)
    millis = int(round((seconds - hours * 3600 - minutes * 60 - whole) * 1000))
    if millis == 1000:
        whole += 1
        millis = 0
    if whole == 60:
        minutes += 1
        whole = 0
    if minutes == 60:
        hours += 1
        minutes = 0
    return f"{hours:02d}:{minutes:02d}:{whole:02d}.{millis:03d}"


def _fps_ratio(fps: float) -> tuple[int, int]:
    """Return a compact rational profile rate without binary-float noise."""

    ratio = Fraction(str(fps)).limit_denominator(1001)
    return ratio.numerator, ratio.denominator


def _file_to_url(filepath: str) -> str:
    try:
        return Path(filepath).resolve().as_uri()
    except (OSError, ValueError):
        return os.path.abspath(filepath).replace("\\", "/")


def _add_property(parent: ET.Element, name: str, value: Any = "") -> ET.Element:
    property_node = ET.SubElement(parent, "property", {"name": name})
    property_node.text = str(value)
    return property_node


def _volume_animation(clip: MLTClip, output_frames: int, fps: float) -> str | None:
    if not clip.volume_keyframes:
        return None
    points: list[tuple[int, float]] = []
    for position, gain in clip.volume_keyframes:
        frame = int(round(position * fps)) if not clip.volume_positions_are_frames else int(round(position))
        frame = max(0, min(output_frames - 1, frame))
        points.append((frame, gain))
    points.sort(key=lambda point: point[0])
    deduped: dict[int, float] = {}
    for frame, gain in points:
        deduped[frame] = gain
    ordered = sorted(deduped.items())
    if len(ordered) == 1:
        return _gain_to_db(ordered[0][1])
    return ";".join(f"{frame}={_gain_to_db(gain)}" for frame, gain in ordered)


def _add_source(mlt: ET.Element, clip: MLTClip, index: int, source_fps: float, source_url: str) -> str:
    source_start = _frame_count(clip.source_start, source_fps)
    source_end = max(source_start + 1, _frame_count(clip.source_end, source_fps))
    source_frames = source_end - source_start
    output_frames = max(1, int(round(source_frames / clip.speed)))
    producer_id = f"opencut_clip_{index}"
    if abs(clip.speed - 1.0) > 1e-6:
        node = ET.SubElement(mlt, "producer", {
            "id": producer_id,
            "in": _timecode(0, source_fps),
            "out": _timecode(max(0, int(math.ceil(source_end / clip.speed)) - 1), source_fps),
        })
        _add_property(node, "length", _timecode(max(1, int(math.ceil(source_end / clip.speed))), source_fps))
        _add_property(node, "eof", "pause")
        _add_property(node, "resource", f"{clip.speed:g}:{source_url}")
        _add_property(node, "warp_speed", f"{clip.speed:g}")
        _add_property(node, "warp_resource", source_url)
        _add_property(node, "warp_pitch", "1")
        _add_property(node, "mlt_service", "timewarp")
        _add_property(node, "shotcut:producer", "avformat")
    else:
        node = ET.SubElement(mlt, "chain", {
            "id": producer_id,
            "in": _timecode(0, source_fps),
            "out": _timecode(source_end - 1, source_fps),
        })
        _add_property(node, "length", _timecode(source_end, source_fps))
        _add_property(node, "eof", "pause")
        _add_property(node, "resource", source_url)
        _add_property(node, "mlt_service", "avformat")
        _add_property(node, "shotcut:caption", clip.name or os.path.basename(source_url))
    _add_property(node, "opencut:source_in", source_start)
    _add_property(node, "opencut:source_out", source_end - 1)
    _add_property(node, "opencut:output_frames", output_frames)
    _add_property(node, "opencut:speed", f"{clip.speed:g}")
    return producer_id


def export_mlt(
    filepath: str | os.PathLike[str],
    segments: Sequence[Any],
    output_path: str | os.PathLike[str],
    sequence_name: str = "OpenCut Edit",
    framerate: float | None = None,
    width: int | None = None,
    height: int | None = None,
    audio_channels: int | None = None,
) -> dict[str, Any]:
    """Write a Kdenlive/Shotcut-compatible MLT project.

    Segment dictionaries use source ``start``/``end`` seconds and may include
    ``speed`` plus ``volume_keyframes``.  Volume keyframes use relative
    seconds unless a point supplies ``frame``; values are linear gain.
    """

    if not output_path:
        raise ValueError("An MLT output path is required")
    filepath = os.fspath(filepath)
    output_path = os.fspath(output_path)
    info: MediaInfo = probe(filepath)
    clips = _normalize_clips(segments)
    video = info.video
    fps = _float(framerate, 0.0) if framerate is not None else 0.0
    fps = fps if fps > 0 else (_float(video.fps, DEFAULT_FPS) if video else DEFAULT_FPS)
    frame_width = int(width or (video.width if video and video.width else DEFAULT_WIDTH))
    frame_height = int(height or (video.height if video and video.height else DEFAULT_HEIGHT))
    channels = int(audio_channels if audio_channels is not None else (info.audio.channels if info.audio else 0))
    channels = max(0, channels)
    fps_num, fps_den = _fps_ratio(fps)

    root = ET.Element("mlt", {
        "LC_NUMERIC": "C",
        "version": MLT_VERSION,
        "title": sequence_name or "OpenCut Edit",
        "producer": "tractor0",
    })
    ET.SubElement(root, "profile", {
        "description": "OpenCut",
        "width": str(max(1, frame_width)),
        "height": str(max(1, frame_height)),
        "progressive": "1",
        "sample_aspect_num": "1",
        "sample_aspect_den": "1",
        "display_aspect_num": str(max(1, frame_width)),
        "display_aspect_den": str(max(1, frame_height)),
        "frame_rate_num": str(fps_num),
        "frame_rate_den": str(fps_den),
        "colorspace": "709",
    })

    source_url = _file_to_url(filepath)
    playlist = ET.SubElement(root, "playlist", {"id": "playlist0"})
    _add_property(playlist, "shotcut:video", "1")
    _add_property(playlist, "shotcut:name", "V1")
    keyframe_count = 0
    output_frames_total = 0
    for index, clip in enumerate(clips):
        source_start = _frame_count(clip.source_start, fps)
        source_end = max(source_start + 1, _frame_count(clip.source_end, fps))
        source_frames = source_end - source_start
        output_frames = max(1, int(round(source_frames / clip.speed)))
        producer_id = _add_source(root, clip, index, fps, source_url)
        if abs(clip.speed - 1.0) > 1e-6:
            entry_in = max(0, int(round(source_start / clip.speed)))
        else:
            entry_in = source_start
        entry = ET.SubElement(playlist, "entry", {
            "producer": producer_id,
            "in": _timecode(entry_in, fps),
            "out": _timecode(entry_in + output_frames - 1, fps),
        })
        _add_property(entry, "opencut:source_in", source_start)
        _add_property(entry, "opencut:source_out", source_end - 1)
        _add_property(entry, "opencut:speed", f"{clip.speed:g}")
        animation = _volume_animation(clip, output_frames, fps)
        if animation is not None:
            filter_node = ET.SubElement(entry, "filter", {
                "in": _timecode(0, fps),
                "out": _timecode(output_frames - 1, fps),
            })
            _add_property(filter_node, "mlt_service", "volume")
            _add_property(filter_node, "level", animation)
            _add_property(filter_node, "shotcut:filter", "audioGain")
            _add_property(filter_node, "kdenlive_id", "volume")
            keyframe_count += len(clip.volume_keyframes)
        output_frames_total += output_frames

    # MLT readers expect source definitions before the playlist that refers
    # to them.  ElementTree appends the playlist before the loop so entries
    # can be filled incrementally; move it into the conventional order once
    # all source nodes exist.
    root.remove(playlist)
    root.insert(1 + len(clips), playlist)

    tractor = ET.SubElement(root, "tractor", {
        "id": "tractor0",
        "in": _timecode(0, fps),
        "out": _timecode(max(0, output_frames_total - 1), fps),
    })
    _add_property(tractor, "shotcut", "1")
    _add_property(tractor, "shotcut:projectAudioChannels", max(2, channels) if channels else 2)
    _add_property(tractor, "kdenlive:clipname", sequence_name or "OpenCut Edit")
    ET.SubElement(tractor, "track", {"producer": "playlist0"})

    main_bin = ET.SubElement(root, "playlist", {"id": "main_bin"})
    _add_property(main_bin, "xml_retain", "1")
    ET.SubElement(main_bin, "entry", {
        "producer": "tractor0",
        "in": _timecode(0, fps),
        "out": _timecode(max(0, output_frames_total - 1), fps),
    })

    ET.indent(root, space="  ")
    tree = ET.ElementTree(root)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return {
        "output_path": output_path,
        "format": "mlt",
        "segments": len(clips),
        "duration_frames": output_frames_total,
        "framerate": fps,
        "speed_changes": sum(abs(clip.speed - 1.0) > 1e-6 for clip in clips),
        "volume_keyframes": keyframe_count,
    }


def export_mlt_from_cuts(
    filepath: str | os.PathLike[str],
    cuts: Sequence[Mapping[str, Any]],
    output_path: str | os.PathLike[str],
    sequence_name: str = "OpenCut Edit",
    total_duration: float = 0.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Invert reviewed remove-ranges and write the kept segments to MLT."""

    info = probe(filepath)
    duration = _float(total_duration, 0.0) or _float(info.duration, 0.0)
    if duration <= 0:
        raise ValueError("Cannot determine source duration for MLT export")
    normalized, kept = cut_ranges_to_segments(list(cuts), duration)
    result = export_mlt(
        filepath,
        kept,
        output_path,
        sequence_name=sequence_name,
        **kwargs,
    )
    result.update({
        "requested_cuts": len(cuts),
        "normalized_cuts": len(normalized),
        "kept_segments": len(kept),
    })
    return result


__all__ = ["MLTClip", "export_mlt", "export_mlt_from_cuts"]
