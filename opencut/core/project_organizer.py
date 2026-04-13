"""
OpenCut AI Project Organization

Analyzes project media files to classify shots, detect groupings, and
propose an organized bin structure. Uses FFmpeg/ffprobe for media analysis
and heuristic-based classification.

No additional dependencies required.
"""

import json
import logging
import os
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional

from opencut.helpers import (
    get_ffprobe_path,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class MediaFileInfo:
    """Analyzed information about a single media file."""
    path: str
    filename: str = ""
    extension: str = ""
    media_type: str = ""    # "video", "audio", "image"
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    codec: str = ""
    shot_type: str = ""     # "close_up", "wide", "medium", etc.
    date_created: str = ""
    file_size: int = 0
    aspect_ratio: str = ""
    has_audio: bool = False
    scene_group: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class ProjectAnalysis:
    """Complete project media analysis."""
    files: List[MediaFileInfo] = field(default_factory=list)
    total_files: int = 0
    total_duration: float = 0.0
    media_types: Dict[str, int] = field(default_factory=dict)
    shot_types: Dict[str, int] = field(default_factory=dict)
    resolutions: Dict[str, int] = field(default_factory=dict)
    codecs: Dict[str, int] = field(default_factory=dict)
    date_groups: Dict[str, List[str]] = field(default_factory=dict)
    scene_groups: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class BinItem:
    """A single bin in the proposed project structure."""
    name: str
    path: str = ""
    files: List[str] = field(default_factory=list)
    file_count: int = 0
    description: str = ""


@dataclass
class BinStructure:
    """Proposed project bin/folder organization."""
    bins: List[BinItem] = field(default_factory=list)
    total_bins: int = 0
    total_files: int = 0
    strategy: str = ""


# ---------------------------------------------------------------------------
# Media file analysis
# ---------------------------------------------------------------------------
_VIDEO_EXTS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".mxf", ".m4v", ".wmv", ".webm", ".flv", ".ts", ".prores"})
_AUDIO_EXTS = frozenset({".wav", ".mp3", ".aac", ".flac", ".ogg", ".m4a", ".aif", ".aiff", ".wma"})
_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".gif", ".webp", ".exr", ".psd"})


def _classify_media_type(ext: str) -> str:
    """Classify file extension into media type."""
    ext = ext.lower()
    if ext in _VIDEO_EXTS:
        return "video"
    if ext in _AUDIO_EXTS:
        return "audio"
    if ext in _IMAGE_EXTS:
        return "image"
    return "other"


def _probe_media_file(filepath: str) -> dict:
    """Probe a media file with ffprobe and return metadata dict."""
    try:
        cmd = [
            get_ffprobe_path(), "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            filepath,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            return {}
        return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as exc:
        logger.debug("ffprobe failed for %s: %s", filepath, exc)
        return {}


def _get_creation_date(probe_data: dict, filepath: str) -> str:
    """Extract creation date from media metadata or file system."""
    # Try format tags
    fmt = probe_data.get("format", {})
    tags = fmt.get("tags", {})

    # Common metadata date fields
    for key in ("creation_time", "date", "creation_date", "com.apple.quicktime.creationdate"):
        val = tags.get(key, "")
        if val:
            try:
                # Parse ISO format and return date part
                dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
                return dt.strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                pass

    # Fallback to file modification time
    try:
        mtime = os.path.getmtime(filepath)
        return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
    except (OSError, Exception):
        return "unknown"


def _classify_shot_type_basic(width: int, height: int, probe_data: dict) -> str:
    """Basic shot type classification from metadata heuristics."""
    # Aspect ratio hints
    if width <= 0 or height <= 0:
        return "unknown"

    aspect = width / height

    # Check for specific camera metadata
    fmt_tags = probe_data.get("format", {}).get("tags", {})
    model = fmt_tags.get("com.apple.quicktime.model", "").lower()
    make = fmt_tags.get("com.apple.quicktime.make", "").lower()

    # Drone detection
    if any(k in model for k in ("dji", "mavic", "phantom", "drone")):
        return "aerial"
    if any(k in make for k in ("dji",)):
        return "aerial"

    # Screen recording detection (exact standard resolutions)
    if (width == 1920 and height == 1080) or (width == 2560 and height == 1440) or (width == 3840 and height == 2160):
        # Check for screen recording indicators
        encoder = ""
        for stream in probe_data.get("streams", []):
            encoder = stream.get("tags", {}).get("encoder", "").lower()
            if "screen" in encoder or "capture" in encoder:
                return "screen_recording"

    # Vertical video = likely phone selfie/close-up
    if aspect < 0.8:
        return "close_up"

    # Ultra-wide = typically establishing/wide shot
    if aspect > 2.2:
        return "wide"

    return "medium"


def _compute_aspect_ratio_label(width: int, height: int) -> str:
    """Compute human-readable aspect ratio label."""
    if width <= 0 or height <= 0:
        return "unknown"
    ratio = width / height
    if abs(ratio - 16 / 9) < 0.05:
        return "16:9"
    if abs(ratio - 9 / 16) < 0.05:
        return "9:16"
    if abs(ratio - 4 / 3) < 0.05:
        return "4:3"
    if abs(ratio - 1.0) < 0.05:
        return "1:1"
    if abs(ratio - 2.35) < 0.1:
        return "2.35:1"
    if abs(ratio - 2.39) < 0.1:
        return "2.39:1"
    if abs(ratio - 1.85) < 0.05:
        return "1.85:1"
    if abs(ratio - 21 / 9) < 0.1:
        return "21:9"
    return f"{ratio:.2f}:1"


def _infer_scene_group(filename: str) -> str:
    """Infer scene grouping from filename patterns."""
    name = os.path.splitext(filename)[0].lower()

    # Common naming patterns: "Scene01_Take02", "SC01_SH02", "A001C001"
    scene_patterns = [
        r'(?:scene|sc|s)[\s_-]*(\d+)',
        r'(?:shot|sh)[\s_-]*(\d+)',
        r'(?:take|tk)[\s_-]*(\d+)',
        r'^([a-z]\d{3})',  # Camera roll (A001, B002)
        r'(?:clip|c)[\s_-]*(\d+)',
        r'(?:seq|sequence)[\s_-]*(\d+)',
    ]

    for pattern in scene_patterns:
        match = re.search(pattern, name)
        if match:
            label = pattern.split("(")[0].strip("(?:").split("|")[0]
            return f"{label}_{match.group(1)}"

    # Group by common filename prefix (first word or segment before number)
    prefix_match = re.match(r'^([a-zA-Z_-]+)', name)
    if prefix_match:
        prefix = prefix_match.group(1).rstrip("_- ")
        if len(prefix) >= 2:
            return f"group_{prefix}"

    return "ungrouped"


def _analyze_single_file(filepath: str) -> MediaFileInfo:
    """Analyze a single media file and return its metadata."""
    filename = os.path.basename(filepath)
    _, ext = os.path.splitext(filename)
    media_type = _classify_media_type(ext)

    info = MediaFileInfo(
        path=filepath,
        filename=filename,
        extension=ext.lower(),
        media_type=media_type,
    )

    # File size
    try:
        info.file_size = os.path.getsize(filepath)
    except OSError:
        pass

    # Probe with ffprobe
    probe_data = _probe_media_file(filepath)
    if not probe_data:
        info.scene_group = _infer_scene_group(filename)
        return info

    # Extract format info
    fmt = probe_data.get("format", {})
    info.duration = float(fmt.get("duration", 0))

    # Extract stream info
    streams = probe_data.get("streams", [])
    has_video = False
    has_audio = False

    for stream in streams:
        codec_type = stream.get("codec_type", "")
        if codec_type == "video" and not has_video:
            has_video = True
            info.width = int(stream.get("width", 0))
            info.height = int(stream.get("height", 0))
            info.codec = stream.get("codec_name", "")
            fps_str = stream.get("r_frame_rate", "0/1")
            parts = fps_str.split("/")
            if len(parts) == 2 and float(parts[1]) > 0:
                info.fps = round(float(parts[0]) / float(parts[1]), 2)
        elif codec_type == "audio":
            has_audio = True

    info.has_audio = has_audio
    info.aspect_ratio = _compute_aspect_ratio_label(info.width, info.height)
    info.date_created = _get_creation_date(probe_data, filepath)
    info.shot_type = _classify_shot_type_basic(info.width, info.height, probe_data)
    info.scene_group = _infer_scene_group(filename)

    # Build tags
    tags = []
    if media_type != "other":
        tags.append(media_type)
    if info.shot_type and info.shot_type != "unknown":
        tags.append(info.shot_type)
    if info.aspect_ratio and info.aspect_ratio != "unknown":
        tags.append(info.aspect_ratio)
    if info.duration > 60:
        tags.append("long_clip")
    elif info.duration > 0 and info.duration < 5:
        tags.append("short_clip")
    if not has_audio and has_video:
        tags.append("no_audio")
    info.tags = tags

    return info


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def analyze_project_media(
    file_list: List[str],
    on_progress: Optional[Callable] = None,
) -> ProjectAnalysis:
    """
    Analyze a list of media files for project organization.

    Probes each file for metadata, classifies shot types,
    groups by date and scene, and computes statistics.

    Args:
        file_list: List of file paths to analyze.
        on_progress: Progress callback(pct, msg).

    Returns:
        ProjectAnalysis with complete analysis.
    """
    if not file_list:
        raise ValueError("No files provided for analysis")

    if on_progress:
        on_progress(5, f"Analyzing {len(file_list)} media files...")

    # Filter to existing files
    valid_files = [f for f in file_list if os.path.isfile(f)]
    if not valid_files:
        raise ValueError("None of the provided files exist")

    files: List[MediaFileInfo] = []
    total = len(valid_files)

    for i, filepath in enumerate(valid_files):
        if on_progress and (i % max(1, total // 20) == 0):
            pct = 10 + int(70 * i / total)
            on_progress(pct, f"Analyzing file {i + 1}/{total}: {os.path.basename(filepath)}")

        file_info = _analyze_single_file(filepath)
        files.append(file_info)

    if on_progress:
        on_progress(85, "Computing statistics...")

    # Build aggregated stats
    media_types: Dict[str, int] = Counter(f.media_type for f in files)
    shot_types: Dict[str, int] = Counter(f.shot_type for f in files if f.shot_type and f.shot_type != "unknown")
    resolutions: Dict[str, int] = Counter(
        f"{f.width}x{f.height}" for f in files if f.width > 0 and f.height > 0
    )
    codecs: Dict[str, int] = Counter(f.codec for f in files if f.codec)

    total_duration = sum(f.duration for f in files)

    # Date groups
    date_groups: Dict[str, List[str]] = defaultdict(list)
    for f in files:
        if f.date_created and f.date_created != "unknown":
            date_groups[f.date_created].append(f.path)

    # Scene groups
    scene_groups: Dict[str, List[str]] = defaultdict(list)
    for f in files:
        if f.scene_group:
            scene_groups[f.scene_group].append(f.path)

    if on_progress:
        on_progress(100, f"Analyzed {len(files)} files")

    return ProjectAnalysis(
        files=files,
        total_files=len(files),
        total_duration=round(total_duration, 2),
        media_types=dict(media_types),
        shot_types=dict(shot_types),
        resolutions=dict(resolutions),
        codecs=dict(codecs),
        date_groups=dict(date_groups),
        scene_groups=dict(scene_groups),
    )


def generate_bin_structure(
    analysis: ProjectAnalysis,
    strategy: str = "auto",
    on_progress: Optional[Callable] = None,
) -> BinStructure:
    """
    Generate a proposed bin/folder structure from project analysis.

    Strategies:
    - "auto": Choose best strategy based on analysis (default)
    - "by_type": Group by media type (video, audio, image)
    - "by_date": Group by creation date
    - "by_scene": Group by scene/filename pattern
    - "by_shot": Group by shot type classification
    - "by_resolution": Group by resolution/aspect ratio
    - "hybrid": Multi-level organization combining type + scene

    Args:
        analysis: ProjectAnalysis from analyze_project_media().
        strategy: Organization strategy.
        on_progress: Progress callback(pct, msg).

    Returns:
        BinStructure with proposed organization.
    """
    if not analysis.files:
        return BinStructure(strategy=strategy)

    if on_progress:
        on_progress(10, f"Generating bin structure (strategy: {strategy})...")

    # Auto-select strategy
    if strategy == "auto":
        strategy = _auto_select_strategy(analysis)
        if on_progress:
            on_progress(20, f"Selected strategy: {strategy}")

    bins: List[BinItem] = []

    if strategy == "by_type":
        bins = _bins_by_type(analysis)
    elif strategy == "by_date":
        bins = _bins_by_date(analysis)
    elif strategy == "by_scene":
        bins = _bins_by_scene(analysis)
    elif strategy == "by_shot":
        bins = _bins_by_shot(analysis)
    elif strategy == "by_resolution":
        bins = _bins_by_resolution(analysis)
    elif strategy == "hybrid":
        bins = _bins_hybrid(analysis)
    else:
        # Default to type-based
        bins = _bins_by_type(analysis)
        strategy = "by_type"

    # Add an "Unsorted" bin for any unassigned files
    assigned_files = set()
    for b in bins:
        assigned_files.update(b.files)

    unassigned = [f.path for f in analysis.files if f.path not in assigned_files]
    if unassigned:
        bins.append(BinItem(
            name="Unsorted",
            path="/Unsorted",
            files=unassigned,
            file_count=len(unassigned),
            description="Files not matching any organization criteria",
        ))

    total_files = sum(b.file_count for b in bins)

    if on_progress:
        on_progress(100, f"Generated {len(bins)} bins for {total_files} files")

    return BinStructure(
        bins=bins,
        total_bins=len(bins),
        total_files=total_files,
        strategy=strategy,
    )


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------
def _auto_select_strategy(analysis: ProjectAnalysis) -> str:
    """Choose the best organization strategy based on analysis."""
    n_files = analysis.total_files
    n_types = len(analysis.media_types)
    n_dates = len(analysis.date_groups)
    n_scenes = len(analysis.scene_groups)

    # If many different dates, organize by date
    if n_dates >= 3 and n_dates >= n_files * 0.3:
        return "by_date"

    # If clear scene groups exist, use scene grouping
    if n_scenes >= 2 and n_scenes < n_files * 0.8:
        meaningful_groups = sum(
            1 for g in analysis.scene_groups.values() if len(g) >= 2
        )
        if meaningful_groups >= 2:
            return "by_scene"

    # If multiple media types, use type grouping
    if n_types >= 2:
        return "hybrid"

    # Default to shot type for single-type (video-only) projects
    if analysis.shot_types:
        return "by_shot"

    return "by_type"


def _bins_by_type(analysis: ProjectAnalysis) -> List[BinItem]:
    """Group files by media type."""
    groups: Dict[str, List[str]] = defaultdict(list)
    for f in analysis.files:
        label = f.media_type.capitalize() if f.media_type != "other" else "Other"
        groups[label].append(f.path)

    bins = []
    for name, files in sorted(groups.items()):
        bins.append(BinItem(
            name=name,
            path=f"/{name}",
            files=files,
            file_count=len(files),
            description=f"{name} files ({len(files)} items)",
        ))
    return bins


def _bins_by_date(analysis: ProjectAnalysis) -> List[BinItem]:
    """Group files by creation date."""
    bins = []
    for date, files in sorted(analysis.date_groups.items()):
        bins.append(BinItem(
            name=f"Day {date}",
            path=f"/Dates/{date}",
            files=files,
            file_count=len(files),
            description=f"Files from {date} ({len(files)} items)",
        ))
    return bins


def _bins_by_scene(analysis: ProjectAnalysis) -> List[BinItem]:
    """Group files by scene/filename pattern."""
    bins = []
    for group_name, files in sorted(analysis.scene_groups.items()):
        if len(files) < 1:
            continue
        display_name = group_name.replace("_", " ").title()
        bins.append(BinItem(
            name=display_name,
            path=f"/Scenes/{group_name}",
            files=files,
            file_count=len(files),
            description=f"Scene group: {display_name} ({len(files)} items)",
        ))
    return bins


def _bins_by_shot(analysis: ProjectAnalysis) -> List[BinItem]:
    """Group files by shot type classification."""
    groups: Dict[str, List[str]] = defaultdict(list)
    for f in analysis.files:
        shot = f.shot_type if f.shot_type and f.shot_type != "unknown" else "unclassified"
        groups[shot].append(f.path)

    bins = []
    for shot_type, files in sorted(groups.items()):
        display_name = shot_type.replace("_", " ").title()
        bins.append(BinItem(
            name=display_name,
            path=f"/Shot Types/{shot_type}",
            files=files,
            file_count=len(files),
            description=f"{display_name} shots ({len(files)} items)",
        ))
    return bins


def _bins_by_resolution(analysis: ProjectAnalysis) -> List[BinItem]:
    """Group files by resolution."""
    groups: Dict[str, List[str]] = defaultdict(list)
    for f in analysis.files:
        if f.width > 0 and f.height > 0:
            label = f"{f.width}x{f.height}"
        else:
            label = "Unknown Resolution"
        groups[label].append(f.path)

    bins = []
    for res, files in sorted(groups.items()):
        bins.append(BinItem(
            name=res,
            path=f"/Resolutions/{res}",
            files=files,
            file_count=len(files),
            description=f"Resolution: {res} ({len(files)} items)",
        ))
    return bins


def _bins_hybrid(analysis: ProjectAnalysis) -> List[BinItem]:
    """Multi-level hybrid organization: type -> scene/date."""
    bins = []

    # First level: media type
    type_groups: Dict[str, List[MediaFileInfo]] = defaultdict(list)
    for f in analysis.files:
        type_groups[f.media_type].append(f)

    for media_type, type_files in sorted(type_groups.items()):
        type_label = media_type.capitalize()

        # Sub-group videos by scene
        if media_type == "video" and len(type_files) > 3:
            scene_sub: Dict[str, List[str]] = defaultdict(list)
            for f in type_files:
                scene_sub[f.scene_group or "ungrouped"].append(f.path)

            for scene, files in sorted(scene_sub.items()):
                scene_label = scene.replace("_", " ").title()
                bins.append(BinItem(
                    name=f"{type_label} - {scene_label}",
                    path=f"/{type_label}/{scene}",
                    files=files,
                    file_count=len(files),
                    description=f"{type_label}/{scene_label} ({len(files)} items)",
                ))
        else:
            files = [f.path for f in type_files]
            bins.append(BinItem(
                name=type_label,
                path=f"/{type_label}",
                files=files,
                file_count=len(files),
                description=f"{type_label} files ({len(files)} items)",
            ))

    return bins
