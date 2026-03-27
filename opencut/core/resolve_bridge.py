"""
DaVinci Resolve Python Scripting Bridge

Provides an abstraction layer for interacting with DaVinci Resolve
via its native Python scripting API. Resolve exposes its API through
a Python module at runtime — no extension/plugin installation needed.

Resolve API docs: https://deric.github.io/DaVinciResolve-API-Docs/

Usage:
    from opencut.core.resolve_bridge import ResolveBridge

    bridge = ResolveBridge()
    if bridge.is_connected():
        media = bridge.get_media_pool_clips()
        bridge.add_markers(timeline_name, markers)
        bridge.import_file("/path/to/output.mp4")
"""

import logging
import os
import sys
from typing import Dict, List, Optional

logger = logging.getLogger("opencut")


def check_resolve_available() -> bool:
    """Check if DaVinci Resolve scripting API is available.

    Resolve must be running and the DaVinciResolveScript module must be
    importable. The module is typically found in the Resolve installation's
    scripting directory.
    """
    try:
        _ensure_resolve_path()
        import DaVinciResolveScript  # noqa: F401
        return True
    except (ImportError, Exception):
        return False


def _ensure_resolve_path():
    """Add DaVinci Resolve's scripting module path to sys.path if needed."""
    resolve_paths = []

    if sys.platform == "win32":
        # Windows paths
        program_data = os.environ.get("PROGRAMDATA", "C:\\ProgramData")
        resolve_paths = [
            os.path.join(program_data, "Blackmagic Design", "DaVinci Resolve", "Support", "Developer", "Scripting", "Modules"),
            os.path.join(os.environ.get("APPDATA", ""), "Blackmagic Design", "DaVinci Resolve", "Support", "Developer", "Scripting", "Modules"),
        ]
    elif sys.platform == "darwin":
        # macOS paths
        resolve_paths = [
            "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting/Modules",
            os.path.expanduser("~/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting/Modules"),
        ]
    else:
        # Linux paths
        resolve_paths = [
            "/opt/resolve/Developer/Scripting/Modules",
            "/opt/resolve/libs/Fusion/Modules",
            os.path.expanduser("~/.local/share/DaVinciResolve/Support/Developer/Scripting/Modules"),
        ]

    # Also check RESOLVE_SCRIPT_API env var
    env_path = os.environ.get("RESOLVE_SCRIPT_API")
    if env_path:
        resolve_paths.insert(0, env_path)

    for path in resolve_paths:
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


class ResolveBridge:
    """Bridge to DaVinci Resolve's Python scripting API."""

    def __init__(self):
        self._resolve = None
        self._connected = False
        self._connect()

    def _connect(self):
        """Attempt to connect to a running Resolve instance."""
        try:
            _ensure_resolve_path()
            import DaVinciResolveScript as dvr
            self._resolve = dvr.scriptapp("Resolve")
            self._connected = self._resolve is not None
            if self._connected:
                logger.info("Connected to DaVinci Resolve")
            else:
                logger.warning("DaVinci Resolve not running or not accessible")
        except ImportError:
            logger.debug("DaVinciResolveScript module not found")
            self._connected = False
        except Exception as e:
            logger.warning("Failed to connect to Resolve: %s", e)
            self._connected = False

    def is_connected(self) -> bool:
        """Check if connected to a running Resolve instance."""
        return self._connected and self._resolve is not None

    def reconnect(self) -> bool:
        """Attempt to reconnect if connection has gone stale."""
        self._connected = False
        self._resolve = None
        self._connect()
        return self.is_connected()

    def _ensure_connected(self) -> bool:
        """Verify connection is live; auto-reconnect once if stale."""
        if self.is_connected():
            try:
                # Lightweight liveness check — will throw if Resolve closed
                self._resolve.GetProductName()
                return True
            except Exception:
                logger.info("Resolve connection stale, attempting reconnect...")
                return self.reconnect()
        return self.reconnect()

    def get_project_info(self) -> Optional[Dict]:
        """Get current project information."""
        if not self._ensure_connected():
            return None
        try:
            pm = self._resolve.GetProjectManager()
            project = pm.GetCurrentProject()
            if not project:
                return None
            timeline = project.GetCurrentTimeline()
            return {
                "name": project.GetName(),
                "timeline_count": project.GetTimelineCount(),
                "current_timeline": timeline.GetName() if timeline else None,
                "frame_rate": project.GetSetting("timelineFrameRate"),
                "resolution_x": project.GetSetting("timelineResolutionWidth"),
                "resolution_y": project.GetSetting("timelineResolutionHeight"),
            }
        except Exception as e:
            logger.warning("get_project_info failed: %s", e)
            return None

    def get_media_pool_clips(self) -> List[Dict]:
        """Get all clips from the media pool."""
        if not self._ensure_connected():
            return []
        try:
            pm = self._resolve.GetProjectManager()
            project = pm.GetCurrentProject()
            if not project:
                return []
            media_pool = project.GetMediaPool()
            root_folder = media_pool.GetRootFolder()
            clips = []
            self._walk_folder(root_folder, clips)
            return clips
        except Exception as e:
            logger.warning("get_media_pool_clips failed: %s", e)
            return []

    def _walk_folder(self, folder, clips: list, depth: int = 0):
        """Recursively walk media pool folders."""
        if depth > 20:
            return
        try:
            for clip in folder.GetClipList():
                try:
                    props = clip.GetClipProperty()
                    clips.append({
                        "name": props.get("Clip Name", clip.GetName() if hasattr(clip, "GetName") else ""),
                        "path": props.get("File Path", ""),
                        "duration": props.get("Duration", ""),
                        "fps": props.get("FPS", ""),
                        "resolution": props.get("Resolution", ""),
                        "type": props.get("Type", ""),
                    })
                except Exception:
                    pass
            for subfolder in folder.GetSubFolderList():
                self._walk_folder(subfolder, clips, depth + 1)
        except Exception:
            pass

    def import_file(self, filepath: str, bin_name: str = "OpenCut Output") -> bool:
        """Import a file into the Resolve media pool."""
        if not self._ensure_connected():
            return False
        try:
            pm = self._resolve.GetProjectManager()
            project = pm.GetCurrentProject()
            if not project:
                return False
            media_pool = project.GetMediaPool()

            # Find or create bin
            root = media_pool.GetRootFolder()
            target_folder = None
            for folder in root.GetSubFolderList():
                if folder.GetName() == bin_name:
                    target_folder = folder
                    break
            if not target_folder:
                target_folder = media_pool.AddSubFolder(root, bin_name)

            if target_folder:
                media_pool.SetCurrentFolder(target_folder)

            result = media_pool.ImportMedia([filepath])
            return bool(result)
        except Exception as e:
            logger.warning("import_file failed: %s", e)
            return False

    def add_markers(self, markers: List[Dict]) -> int:
        """Add markers to the current timeline.

        Args:
            markers: List of dicts with 'time' (seconds), 'name', optional 'color'.

        Returns:
            Number of markers successfully added.
        """
        if not self._ensure_connected():
            return 0
        try:
            pm = self._resolve.GetProjectManager()
            project = pm.GetCurrentProject()
            if not project:
                return 0
            timeline = project.GetCurrentTimeline()
            if not timeline:
                return 0

            fps = float(timeline.GetSetting("timelineFrameRate") or 24)
            added = 0
            for m in markers:
                time_sec = float(m.get("time", 0))
                frame_id = int(time_sec * fps)
                name = m.get("name", "Marker")
                color = m.get("color", "Green")

                # Resolve expects color names: Blue, Cyan, Green, Yellow, Red, Pink, Purple, Fuchsia, Rose, Lavender, Sky, Mint, Lemon, Sand, Cocoa, Cream
                if isinstance(color, str) and color.lower() not in ("blue", "cyan", "green", "yellow", "red", "pink", "purple"):
                    color = "Green"
                else:
                    color = color.capitalize()

                success = timeline.AddMarker(
                    frame_id,
                    color,
                    name,
                    "",  # note
                    1,   # duration (frames)
                )
                if success:
                    added += 1

            return added
        except Exception as e:
            logger.warning("add_markers failed: %s", e)
            return 0

    def get_timeline_info(self) -> Optional[Dict]:
        """Get current timeline information."""
        if not self._ensure_connected():
            return None
        try:
            pm = self._resolve.GetProjectManager()
            project = pm.GetCurrentProject()
            if not project:
                return None
            timeline = project.GetCurrentTimeline()
            if not timeline:
                return None

            return {
                "name": timeline.GetName(),
                "frame_rate": timeline.GetSetting("timelineFrameRate"),
                "start_frame": timeline.GetStartFrame(),
                "end_frame": timeline.GetEndFrame(),
                "video_track_count": timeline.GetTrackCount("video"),
                "audio_track_count": timeline.GetTrackCount("audio"),
                "marker_count": len(timeline.GetMarkers()),
            }
        except Exception as e:
            logger.warning("get_timeline_info failed: %s", e)
            return None

    def render_timeline(self, output_path: str, preset: str = "H.264 Master") -> bool:
        """Start a render of the current timeline.

        Args:
            output_path: Output file path.
            preset: Render preset name.

        Returns:
            True if render started successfully.
        """
        if not self._ensure_connected():
            return False
        try:
            pm = self._resolve.GetProjectManager()
            project = pm.GetCurrentProject()
            if not project:
                return False

            project.SetRenderSettings({
                "TargetDir": os.path.dirname(output_path),
                "CustomName": os.path.splitext(os.path.basename(output_path))[0],
            })

            project.AddRenderJob()
            project.StartRendering()
            return True
        except Exception as e:
            logger.warning("render_timeline failed: %s", e)
            return False
