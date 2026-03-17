"""
OpenCut Auto-Edit Module

Motion-based editing using auto-editor. Detects static/boring segments
by visual motion analysis and creates edit lists of segments to keep.

Requires: pip install auto-editor
"""

import html
import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class EditSegment:
    """A segment to keep or cut."""
    start: float
    end: float
    action: str = "keep"  # "keep" or "cut"

    @property
    def duration(self):
        return self.end - self.start


@dataclass
class AutoEditResult:
    """Results from auto-editor analysis."""
    segments: List[EditSegment] = field(default_factory=list)
    total_duration: float = 0.0
    kept_duration: float = 0.0
    removed_duration: float = 0.0
    reduction_percent: float = 0.0
    xml_path: Optional[str] = None  # Path to generated Premiere XML


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------
def check_auto_editor_version():
    """
    Check auto-editor availability and version.

    Returns:
        Version string (e.g. "24w51a") or None if not installed.
    """
    try:
        result = subprocess.run(
            ["auto-editor", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: try as Python module
    try:
        result = subprocess.run(
            [shutil.which("python") or "python", "-m", "auto_editor", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return None


def _get_auto_editor_cmd():
    """Get the auto-editor command (standalone or python -m)."""
    if shutil.which("auto-editor"):
        return ["auto-editor"]
    # Try as python module
    return [shutil.which("python") or "python", "-m", "auto_editor"]


def _probe_duration(input_path):
    """Get video duration via ffprobe. Returns float seconds or 0.0."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", input_path],
            capture_output=True, text=True, timeout=30,
        )
        data = json.loads(result.stdout)
        return float(data.get("format", {}).get("duration", 0.0))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------
def _parse_auto_editor_json(json_path, total_duration):
    """
    Parse auto-editor's JSON export into EditSegments.

    auto-editor JSON contains chunks/timeline with start, dur, speed, offset.
    Segments with speed > 0 (and != 99999) are "keep"; speed == 0 or 99999 are "cut".

    Args:
        json_path: Path to auto-editor JSON output.
        total_duration: Total source duration in seconds.

    Returns:
        List of EditSegment objects.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = []

    # auto-editor v2 JSON: {"timeline": {"v": [[ ... ]]}} or {"chunks": [...]}
    # v1 format: list of [start, end, speed] or {"chunks": [[start, end, speed], ...]}
    chunks = None

    if isinstance(data, dict):
        # v2 timeline format
        timeline = data.get("timeline", {})
        if isinstance(timeline, dict):
            v_tracks = timeline.get("v", [])
            if v_tracks and isinstance(v_tracks[0], list):
                for clip in v_tracks[0]:
                    tb = float(clip.get("tb", 30)) or 30.0  # guard against zero
                    start = float(clip.get("offset", clip.get("start", 0))) / tb
                    dur = float(clip.get("dur", 0)) / tb
                    speed = float(clip.get("speed", 1.0))
                    end = start + dur
                    action = "cut" if speed == 0 or speed >= 99999 else "keep"
                    segments.append(EditSegment(start=start, end=end, action=action))

        # v1 chunks format
        chunks = data.get("chunks", None)
    elif isinstance(data, list):
        chunks = data

    if chunks is not None:
        for chunk in chunks:
            if isinstance(chunk, (list, tuple)) and len(chunk) >= 3:
                start = float(chunk[0])
                end = float(chunk[1])
                speed = float(chunk[2])
                action = "cut" if speed == 0 or speed >= 99999 else "keep"
                segments.append(EditSegment(start=start, end=end, action=action))
            elif isinstance(chunk, dict):
                start = float(chunk.get("start", 0))
                end = float(chunk.get("end", start))
                speed = float(chunk.get("speed", 1.0))
                action = "cut" if speed == 0 or speed >= 99999 else "keep"
                segments.append(EditSegment(start=start, end=end, action=action))

    # If parsing produced no segments, treat entire file as kept
    if not segments and total_duration > 0:
        segments.append(EditSegment(start=0.0, end=total_duration, action="keep"))

    return segments


# ---------------------------------------------------------------------------
# Premiere XML export
# ---------------------------------------------------------------------------
def _export_premiere_xml(segments, input_path, output_path):
    """
    Convert edit segments to Premiere Pro compatible FCP 7 XML.

    Generates an XML import file that Premiere can open directly,
    containing a sequence with clips at the keep-segment times.

    Args:
        segments: List of EditSegment (only "keep" segments are placed).
        input_path: Path to the source media file.
        output_path: Path to write the XML file.
    """
    keep_segments = [s for s in segments if s.action == "keep"]
    filename = html.escape(os.path.basename(input_path))
    filepath_url = html.escape(input_path.replace("\\", "/"))

    # Build clip entries
    clip_entries = []
    timeline_pos = 0
    for i, seg in enumerate(keep_segments):
        start_frames = max(0, int(seg.start * 30))
        end_frames = max(start_frames, int(seg.end * 30))
        duration_frames = end_frames - start_frames

        clip_entries.append(f"""
                <clipitem id="clip-{i + 1}">
                    <name>{filename} - Clip {i + 1}</name>
                    <duration>{duration_frames}</duration>
                    <rate><timebase>30</timebase><ntsc>TRUE</ntsc></rate>
                    <start>{timeline_pos}</start>
                    <end>{timeline_pos + duration_frames}</end>
                    <in>{start_frames}</in>
                    <out>{end_frames}</out>
                    <file id="file-1"/>
                </clipitem>""")

        timeline_pos += duration_frames

    total_frames = timeline_pos if timeline_pos > 0 else 1

    xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE xmeml>
<xmeml version="4">
    <sequence>
        <name>OpenCut Auto-Edit</name>
        <duration>{total_frames}</duration>
        <rate>
            <timebase>30</timebase>
            <ntsc>TRUE</ntsc>
        </rate>
        <media>
            <video>
                <track>{"".join(clip_entries)}
                </track>
            </video>
        </media>
    </sequence>
    <bin>
        <children>
            <clip id="master-clip-1">
                <name>{filename}</name>
                <file id="file-1">
                    <name>{filename}</name>
                    <pathurl>file:///{filepath_url}</pathurl>
                    <rate><timebase>30</timebase><ntsc>TRUE</ntsc></rate>
                </file>
            </clip>
        </children>
    </bin>
</xmeml>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml_content)

    logger.info("Exported Premiere XML: %s (%d clips)", output_path, len(keep_segments))


# ---------------------------------------------------------------------------
# Main auto-edit function
# ---------------------------------------------------------------------------
def auto_edit(
    input_path,
    method="motion",
    threshold=0.02,
    margin=0.2,
    min_clip_length=0.5,
    export_xml=True,
    output_dir="",
    on_progress=None,
):
    """
    Run auto-editor on a video file to detect interesting segments.

    Args:
        input_path: Path to the source video file.
        method: Detection method - "motion", "audio", or "both".
        threshold: Motion detection threshold (lower = more sensitive).
        margin: Padding around detected segments in seconds.
        min_clip_length: Minimum segment duration to keep in seconds.
        export_xml: Generate Premiere Pro compatible XML import file.
        output_dir: Directory for output files. Uses temp dir if empty.
        on_progress: Progress callback(pct, msg).

    Returns:
        AutoEditResult with detected segments and stats.
    """
    # Verify auto-editor is available
    version = check_auto_editor_version()
    if version is None:
        raise RuntimeError(
            "auto-editor not found. Install with: pip install auto-editor"
        )

    if on_progress:
        on_progress(5, f"auto-editor {version} found, preparing...")

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Probe duration for timeout scaling and stats
    total_duration = _probe_duration(input_path)

    # Set up output directory
    if output_dir and os.path.isdir(output_dir):
        temp_dir = None
        work_dir = output_dir
    else:
        temp_dir = tempfile.mkdtemp(prefix="opencut_autoedit_")
        work_dir = temp_dir

    json_output = os.path.join(work_dir, "auto_edit_result.json")

    try:
        return _run_auto_edit(
            input_path, method, threshold, margin, min_clip_length,
            export_xml, work_dir, json_output, total_duration, on_progress,
        )
    except Exception:
        if temp_dir and os.path.isdir(temp_dir):
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def _run_auto_edit(input_path, method, threshold, margin, min_clip_length,
                   export_xml, work_dir, json_output, total_duration, on_progress):
    """Inner auto-edit logic (extracted for temp dir cleanup on error)."""
    if on_progress:
        on_progress(10, f"Analyzing video ({method} detection)...")

    # Build auto-editor command
    base_cmd = _get_auto_editor_cmd()
    cmd = base_cmd + [input_path]

    if method == "motion":
        cmd += ["--edit", f"motion:threshold={threshold}"]
    elif method == "audio":
        cmd += ["--edit", "audio"]
    elif method == "both":
        cmd += ["--edit", f"motion:threshold={threshold}", "--edit", "audio"]
    else:
        cmd += ["--edit", f"motion:threshold={threshold}"]

    cmd += [
        "--margin", f"{margin}s",
        "--min-clip-length", f"{min_clip_length}s",
        "--export", "json",
        "--output", json_output,
    ]

    logger.info("Running auto-editor: %s", " ".join(cmd))

    # Scale timeout: base 120s + 5x duration (long videos need time)
    timeout = max(120, int(total_duration * 5) + 120) if total_duration > 0 else 1800

    if on_progress:
        on_progress(20, "Running auto-editor analysis (this may take a while)...")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"auto-editor timed out after {timeout}s processing '{os.path.basename(input_path)}'"
        )

    if result.returncode != 0:
        stderr = result.stderr.strip()[-500:] if result.stderr else "unknown error"
        raise RuntimeError(f"auto-editor failed: {stderr}")

    if on_progress:
        on_progress(60, "Parsing analysis results...")

    # Parse output
    if not os.path.isfile(json_output):
        raise RuntimeError(
            f"auto-editor did not produce expected output at {json_output}"
        )

    segments = _parse_auto_editor_json(json_output, total_duration)

    if on_progress:
        on_progress(80, "Calculating edit statistics...")

    # Calculate stats
    keep_segments = [s for s in segments if s.action == "keep"]
    cut_segments = [s for s in segments if s.action == "cut"]

    kept_duration = sum(s.duration for s in keep_segments)
    removed_duration = sum(s.duration for s in cut_segments)

    # Use probed duration as ground truth if available
    if total_duration <= 0:
        total_duration = kept_duration + removed_duration

    reduction = (removed_duration / total_duration * 100) if total_duration > 0 else 0.0

    # Export Premiere XML if requested
    xml_path = None
    if export_xml and keep_segments:
        if on_progress:
            on_progress(90, "Generating Premiere Pro XML...")

        xml_path = os.path.join(work_dir, "auto_edit_premiere.xml")
        _export_premiere_xml(segments, input_path, xml_path)

    if on_progress:
        on_progress(100, f"Done: keeping {kept_duration:.1f}s of {total_duration:.1f}s ({reduction:.0f}% removed)")

    logger.info(
        "Auto-edit complete: %d segments, %.1fs kept, %.1fs removed (%.0f%%)",
        len(keep_segments), kept_duration, removed_duration, reduction,
    )

    return AutoEditResult(
        segments=segments,
        total_duration=total_duration,
        kept_duration=kept_duration,
        removed_duration=removed_duration,
        reduction_percent=round(reduction, 1),
        xml_path=xml_path,
    )
