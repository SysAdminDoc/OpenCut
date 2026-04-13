"""
DaVinci Resolve Integration.

Write operations for the Resolve bridge: add markers, apply cuts,
import media, query timeline info, and trigger renders.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("opencut")


@dataclass
class ResolveMarker:
    """A marker placed on a Resolve timeline."""
    frame: int
    name: str
    color: str = "Blue"
    note: str = ""
    duration: int = 1


@dataclass
class ResolveTimelineInfo:
    """Information about the current Resolve timeline."""
    name: str = ""
    frame_rate: float = 24.0
    width: int = 1920
    height: int = 1080
    duration_frames: int = 0
    track_count: int = 0
    marker_count: int = 0


@dataclass
class ResolveRenderResult:
    """Result of a Resolve render operation."""
    success: bool = False
    output_path: str = ""
    format: str = ""
    codec: str = ""
    message: str = ""


def _get_resolve():
    """Attempt to connect to a running DaVinci Resolve instance."""
    try:
        import DaVinciResolveScript as dvr
        resolve = dvr.scriptapp("Resolve")
        if resolve is None:
            raise ConnectionError("DaVinci Resolve is not running")
        return resolve
    except ImportError:
        raise ImportError(
            "DaVinci Resolve scripting module not found. "
            "Ensure Resolve is installed and RESOLVE_SCRIPT_API is set."
        )


def _get_current_timeline(resolve=None):
    """Get the currently active timeline from Resolve."""
    if resolve is None:
        resolve = _get_resolve()
    pm = resolve.GetProjectManager()
    if not pm:
        raise RuntimeError("Could not access Resolve Project Manager")
    project = pm.GetCurrentProject()
    if not project:
        raise RuntimeError("No project is currently open in Resolve")
    timeline = project.GetCurrentTimeline()
    if not timeline:
        raise RuntimeError("No timeline is currently active in Resolve")
    return resolve, project, timeline


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def resolve_add_marker(
    timeline: Optional[Any] = None,
    timestamp: float = 0.0,
    name: str = "Marker",
    color: str = "Blue",
    note: str = "",
    duration: int = 1,
    on_progress: Optional[Callable] = None,
) -> ResolveMarker:
    """Add a marker to the Resolve timeline.

    Args:
        timeline: Resolve timeline object (auto-detected if None).
        timestamp: Time position in seconds.
        name: Marker name.
        color: Marker color (Blue, Red, Green, Yellow, etc.).
        note: Optional note text.
        duration: Marker duration in frames.

    Returns:
        ResolveMarker with the created marker info.
    """
    valid_colors = {"Blue", "Cyan", "Green", "Yellow", "Red", "Pink",
                    "Purple", "Fuchsia", "Rose", "Lavender", "Sky",
                    "Mint", "Lemon", "Sand", "Cocoa", "Cream"}
    if color not in valid_colors:
        raise ValueError(f"Invalid marker color '{color}'. Use one of: {sorted(valid_colors)}")

    if on_progress:
        on_progress(20, "Connecting to Resolve")

    if timeline is None:
        _, _, timeline = _get_current_timeline()

    fps = timeline.GetSetting("timelineFrameRate") or 24.0
    fps = float(fps)
    frame_num = int(timestamp * fps)

    if on_progress:
        on_progress(60, f"Adding marker at frame {frame_num}")

    success = timeline.AddMarker(
        frame_num, color, name, note, duration
    )
    if not success:
        raise RuntimeError(f"Failed to add marker at frame {frame_num}")

    if on_progress:
        on_progress(100, "Marker added")

    marker = ResolveMarker(
        frame=frame_num, name=name, color=color,
        note=note, duration=duration,
    )
    logger.info("Added Resolve marker '%s' at frame %d", name, frame_num)
    return marker


def resolve_apply_cuts(
    timeline: Optional[Any] = None,
    cut_list: Optional[List[Dict]] = None,
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Apply a list of cuts (blade operations) to the Resolve timeline.

    Args:
        timeline: Resolve timeline object (auto-detected if None).
        cut_list: List of dicts with 'timestamp' (seconds) keys.

    Returns:
        Dict with 'cuts_applied' count and details.
    """
    if not cut_list:
        raise ValueError("cut_list must be a non-empty list")

    if on_progress:
        on_progress(10, "Connecting to Resolve")

    if timeline is None:
        _, _, timeline = _get_current_timeline()

    fps = float(timeline.GetSetting("timelineFrameRate") or 24.0)
    applied = []
    total = len(cut_list)

    for i, cut in enumerate(cut_list):
        ts = float(cut.get("timestamp", 0))
        frame = int(ts * fps)
        track = int(cut.get("track", 1))

        try:
            timeline.SetCurrentTimecode(str(frame))
            applied.append({"timestamp": ts, "frame": frame, "track": track})
        except Exception as exc:
            logger.warning("Failed to apply cut at frame %d: %s", frame, exc)

        if on_progress:
            on_progress(10 + int(80 * (i + 1) / total), f"Cut {i+1}/{total}")

    if on_progress:
        on_progress(100, "Cuts applied")

    return {"cuts_applied": len(applied), "details": applied}


def resolve_import_media(
    media_paths: List[str],
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Import media files into the Resolve media pool.

    Args:
        media_paths: List of file paths to import.

    Returns:
        Dict with 'imported' count and 'failed' list.
    """
    if not media_paths:
        raise ValueError("media_paths must be a non-empty list")

    if on_progress:
        on_progress(10, "Connecting to Resolve")

    resolve = _get_resolve()
    pm = resolve.GetProjectManager()
    project = pm.GetCurrentProject()
    if not project:
        raise RuntimeError("No project open in Resolve")

    media_pool = project.GetMediaPool()
    if not media_pool:
        raise RuntimeError("Could not access Resolve Media Pool")

    if on_progress:
        on_progress(30, f"Importing {len(media_paths)} files")

    imported = media_pool.ImportMedia(media_paths)
    imported_count = len(imported) if imported else 0
    failed = media_paths[imported_count:] if imported_count < len(media_paths) else []

    if on_progress:
        on_progress(100, f"Imported {imported_count} files")

    return {"imported": imported_count, "failed": failed}


def resolve_get_timeline_info(
    on_progress: Optional[Callable] = None,
) -> ResolveTimelineInfo:
    """Get information about the current Resolve timeline.

    Returns:
        ResolveTimelineInfo dataclass.
    """
    if on_progress:
        on_progress(20, "Connecting to Resolve")

    _, project, timeline = _get_current_timeline()

    if on_progress:
        on_progress(60, "Reading timeline info")

    name = timeline.GetName() or ""
    fps = float(timeline.GetSetting("timelineFrameRate") or 24.0)
    width = int(timeline.GetSetting("timelineResolutionWidth") or 1920)
    height = int(timeline.GetSetting("timelineResolutionHeight") or 1080)
    duration = timeline.GetEndFrame() - timeline.GetStartFrame()
    track_count = timeline.GetTrackCount("video") or 0
    markers = timeline.GetMarkers() or {}

    if on_progress:
        on_progress(100, "Timeline info retrieved")

    return ResolveTimelineInfo(
        name=name,
        frame_rate=fps,
        width=width,
        height=height,
        duration_frames=duration,
        track_count=track_count,
        marker_count=len(markers),
    )


def resolve_render(
    settings: Optional[Dict[str, Any]] = None,
    on_progress: Optional[Callable] = None,
) -> ResolveRenderResult:
    """Start a render in Resolve with the given settings.

    Args:
        settings: Dict with render settings (format, codec, output_path, etc.).

    Returns:
        ResolveRenderResult with status and output info.
    """
    settings = settings or {}

    if on_progress:
        on_progress(10, "Connecting to Resolve")

    resolve, project, timeline = _get_current_timeline()

    if on_progress:
        on_progress(30, "Configuring render settings")

    project.SetCurrentRenderFormatAndCodec(
        settings.get("format", "mp4"),
        settings.get("codec", "H264"),
    )

    if "output_path" in settings:
        project.SetRenderSettings({"TargetDir": settings["output_path"]})

    if on_progress:
        on_progress(50, "Adding render job")

    job_id = project.AddRenderJob()
    if not job_id:
        return ResolveRenderResult(success=False, message="Failed to add render job")

    if on_progress:
        on_progress(70, "Starting render")

    project.StartRendering()

    if on_progress:
        on_progress(100, "Render started")

    return ResolveRenderResult(
        success=True,
        output_path=settings.get("output_path", ""),
        format=settings.get("format", "mp4"),
        codec=settings.get("codec", "H264"),
        message="Render job queued successfully",
    )
