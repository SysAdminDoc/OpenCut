"""
OpenCut Archival Conform Module (14.4)

Detect mixed frame rates, aspect ratios, interlacing, and colorspace.
Auto-conform clips to project target settings via FFmpeg.
"""

import json
import logging
import os
import subprocess as _sp
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import (
    FFmpegCmd,
    get_ffprobe_path,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


@dataclass
class ConformanceIssue:
    """A detected conformance mismatch."""
    file_path: str
    issue_type: str   # 'frame_rate', 'resolution', 'interlacing', 'colorspace', 'aspect_ratio'
    current_value: str
    target_value: str
    severity: str = "warning"  # 'info', 'warning', 'error'


@dataclass
class ConformanceReport:
    """Full conformance analysis report."""
    issues: List[ConformanceIssue] = field(default_factory=list)
    files_analyzed: int = 0
    files_with_issues: int = 0
    all_conformant: bool = True
    target_settings: Dict = field(default_factory=dict)


@dataclass
class ConformResult:
    """Result of conforming a single clip."""
    output_path: str = ""
    input_path: str = ""
    changes_applied: List[str] = field(default_factory=list)
    duration: float = 0.0


def _probe_detailed(filepath: str) -> dict:
    """Get detailed stream info via ffprobe including interlacing and colorspace."""
    cmd = [
        get_ffprobe_path(), "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,pix_fmt,"
        "color_space,color_transfer,color_primaries,field_order,"
        "sample_aspect_ratio,display_aspect_ratio,codec_name",
        "-show_entries", "format=duration",
        "-of", "json", filepath,
    ]
    result = _sp.run(cmd, capture_output=True, timeout=30)
    defaults = {
        "width": 0, "height": 0, "fps": 0.0, "duration": 0.0,
        "pix_fmt": "unknown", "color_space": "unknown",
        "color_transfer": "unknown", "color_primaries": "unknown",
        "field_order": "progressive", "sar": "1:1", "dar": "unknown",
        "codec": "unknown",
    }
    if result.returncode != 0:
        return defaults

    try:
        data = json.loads(result.stdout.decode())
        streams = data.get("streams", [])
        if not streams:
            return defaults
        s = streams[0]
        fps_p = s.get("r_frame_rate", "30/1").split("/")
        fps = (float(fps_p[0]) / float(fps_p[1])) if len(fps_p) == 2 and float(fps_p[1]) else 30.0
        duration = float(s.get("duration", 0))
        if duration <= 0:
            duration = float(data.get("format", {}).get("duration", 0))

        field_order = s.get("field_order", "progressive")
        if field_order in ("unknown", ""):
            field_order = "progressive"

        return {
            "width": int(s.get("width", 0)),
            "height": int(s.get("height", 0)),
            "fps": round(fps, 3),
            "duration": duration,
            "pix_fmt": s.get("pix_fmt", "unknown"),
            "color_space": s.get("color_space", "unknown"),
            "color_transfer": s.get("color_transfer", "unknown"),
            "color_primaries": s.get("color_primaries", "unknown"),
            "field_order": field_order,
            "sar": s.get("sample_aspect_ratio", "1:1"),
            "dar": s.get("display_aspect_ratio", "unknown"),
            "codec": s.get("codec_name", "unknown"),
        }
    except Exception as e:
        logger.warning("Failed to parse detailed probe for %s: %s", filepath, e)
        return defaults


def analyze_conformance(
    file_paths: List[str],
    target_settings: Dict,
    on_progress: Optional[Callable] = None,
) -> ConformanceReport:
    """Analyze multiple clips for conformance against target settings.

    Args:
        file_paths: List of video file paths to analyze.
        target_settings: Dict with target values:
            - fps (float): Target frame rate (e.g. 23.976, 29.97, 30, 60).
            - width (int): Target width.
            - height (int): Target height.
            - pix_fmt (str): Target pixel format (e.g. 'yuv420p').
            - color_space (str): Target colorspace (e.g. 'bt709').
            - interlaced (bool): Whether target is interlaced (default False).
        on_progress: Progress callback(pct, msg).

    Returns:
        ConformanceReport with detected issues.
    """
    if not file_paths:
        raise ValueError("At least one file path is required")

    target_fps = target_settings.get("fps", 30.0)
    target_w = target_settings.get("width", 1920)
    target_h = target_settings.get("height", 1080)
    target_pix = target_settings.get("pix_fmt", "yuv420p")
    target_cs = target_settings.get("color_space", "bt709")
    target_interlaced = target_settings.get("interlaced", False)

    issues = []
    files_with_issues = set()

    for idx, fp in enumerate(file_paths):
        if not os.path.isfile(fp):
            logger.warning("Skipping missing file: %s", fp)
            continue

        if on_progress:
            pct = int(10 + 80 * idx / len(file_paths))
            on_progress(pct, f"Analyzing {os.path.basename(fp)}...")

        info = _probe_detailed(fp)

        # Check frame rate
        if info["fps"] > 0 and abs(info["fps"] - target_fps) > 0.1:
            issues.append(ConformanceIssue(
                file_path=fp,
                issue_type="frame_rate",
                current_value=f"{info['fps']:.3f} fps",
                target_value=f"{target_fps:.3f} fps",
                severity="warning",
            ))
            files_with_issues.add(fp)

        # Check resolution
        if (info["width"] > 0 and info["height"] > 0 and
                (info["width"] != target_w or info["height"] != target_h)):
            issues.append(ConformanceIssue(
                file_path=fp,
                issue_type="resolution",
                current_value=f"{info['width']}x{info['height']}",
                target_value=f"{target_w}x{target_h}",
                severity="warning",
            ))
            files_with_issues.add(fp)

        # Check interlacing
        is_interlaced = info["field_order"] not in ("progressive", "unknown")
        if is_interlaced != target_interlaced:
            issues.append(ConformanceIssue(
                file_path=fp,
                issue_type="interlacing",
                current_value=info["field_order"],
                target_value="interlaced" if target_interlaced else "progressive",
                severity="warning",
            ))
            files_with_issues.add(fp)

        # Check pixel format
        if info["pix_fmt"] not in ("unknown", target_pix):
            issues.append(ConformanceIssue(
                file_path=fp,
                issue_type="pixel_format",
                current_value=info["pix_fmt"],
                target_value=target_pix,
                severity="info",
            ))
            files_with_issues.add(fp)

        # Check colorspace
        if info["color_space"] not in ("unknown", target_cs):
            issues.append(ConformanceIssue(
                file_path=fp,
                issue_type="colorspace",
                current_value=info["color_space"],
                target_value=target_cs,
                severity="info",
            ))
            files_with_issues.add(fp)

    if on_progress:
        on_progress(100, f"Analysis complete: {len(issues)} issues in {len(files_with_issues)} files")

    return ConformanceReport(
        issues=issues,
        files_analyzed=len(file_paths),
        files_with_issues=len(files_with_issues),
        all_conformant=len(issues) == 0,
        target_settings=target_settings,
    )


def conform_clip(
    video_path: str,
    target_settings: Dict,
    output_path_str: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> ConformResult:
    """Conform a single clip to target settings.

    Args:
        video_path: Input video path.
        target_settings: Same target dict as analyze_conformance().
        output_path_str: Output path. Auto-generated if None.
        on_progress: Progress callback(pct, msg).

    Returns:
        ConformResult with output path and changes applied.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if on_progress:
        on_progress(5, "Probing source clip...")

    info = _probe_detailed(video_path)
    target_fps = target_settings.get("fps", 30.0)
    target_w = target_settings.get("width", 1920)
    target_h = target_settings.get("height", 1080)
    target_pix = target_settings.get("pix_fmt", "yuv420p")
    target_cs = target_settings.get("color_space", "bt709")
    target_interlaced = target_settings.get("interlaced", False)

    if output_path_str is None:
        output_path_str = output_path(video_path, "conformed")

    if on_progress:
        on_progress(15, "Building conform filter chain...")

    changes = []
    vf_parts = []

    # Deinterlace if needed
    is_interlaced = info["field_order"] not in ("progressive", "unknown")
    if is_interlaced and not target_interlaced:
        vf_parts.append("yadif=mode=0:parity=-1:deint=1")
        changes.append(f"deinterlaced ({info['field_order']} -> progressive)")

    # Scale and pad to target resolution
    if info["width"] != target_w or info["height"] != target_h:
        vf_parts.append(
            f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
            f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2"
        )
        changes.append(f"scaled ({info['width']}x{info['height']} -> {target_w}x{target_h})")

    # Set SAR to 1:1
    vf_parts.append("setsar=1")

    # Frame rate conversion
    if abs(info["fps"] - target_fps) > 0.1:
        vf_parts.append(f"fps={target_fps}")
        changes.append(f"fps ({info['fps']:.3f} -> {target_fps:.3f})")

    # Colorspace conversion
    if info["color_space"] not in ("unknown", target_cs):
        vf_parts.append(f"colorspace=all={target_cs}")
        changes.append(f"colorspace ({info['color_space']} -> {target_cs})")

    if on_progress:
        on_progress(30, f"Applying {len(changes)} conform changes...")

    vf_chain = ",".join(vf_parts) if vf_parts else "null"

    cmd = (FFmpegCmd()
           .input(video_path)
           .video_codec("libx264", crf=18, preset="medium", pix_fmt=target_pix)
           .audio_codec("aac", bitrate="192k")
           .video_filter(vf_chain)
           .option("ac", "2")
           .option("ar", "48000")
           .faststart()
           .output(output_path_str)
           .build())

    run_ffmpeg(cmd)

    if not changes:
        changes.append("re-encoded (no significant changes needed)")

    if on_progress:
        on_progress(100, f"Conformed: {', '.join(changes)}")

    return ConformResult(
        output_path=output_path_str,
        input_path=video_path,
        changes_applied=changes,
        duration=info["duration"],
    )


def batch_conform(
    file_paths: List[str],
    target_settings: Dict,
    output_dir: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Batch-conform multiple clips to target settings.

    Args:
        file_paths: List of video file paths.
        target_settings: Target settings dict.
        output_dir: Output directory. Defaults to source directory.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with results list, success_count, error_count.
    """
    if not file_paths:
        raise ValueError("At least one file path is required")

    results = []
    errors = []

    for idx, fp in enumerate(file_paths):
        if on_progress:
            pct = int(5 + 90 * idx / len(file_paths))
            on_progress(pct, f"Conforming {idx + 1}/{len(file_paths)}: "
                             f"{os.path.basename(fp)}...")

        try:
            out = None
            if output_dir:
                base = os.path.splitext(os.path.basename(fp))[0]
                out = os.path.join(output_dir, f"{base}_conformed.mp4")

            result = conform_clip(fp, target_settings, output_path_str=out)
            results.append({
                "input_path": fp,
                "output_path": result.output_path,
                "changes": result.changes_applied,
                "status": "ok",
            })
        except Exception as e:
            logger.warning("Failed to conform %s: %s", fp, e)
            errors.append({"input_path": fp, "error": str(e)})

    if on_progress:
        on_progress(100, f"Batch conform complete: {len(results)} ok, {len(errors)} errors")

    return {
        "results": results,
        "errors": errors,
        "success_count": len(results),
        "error_count": len(errors),
        "target_settings": target_settings,
    }
