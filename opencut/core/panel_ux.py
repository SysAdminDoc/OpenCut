"""
OpenCut Panel UX Backend Stubs

Backend support for 8 panel-only UX features:
  6.1  Drag-and-Drop handler registration
  6.2  Workspace Layouts (save / load / list / delete)
  6.6  Right-Click Context Menu actions
  6.7  Quick Previews (single-frame operation preview)
  6.8  Theme Toggle (dark / light / system)
  37.1 Guided Walkthroughs (step definitions + completion tracking)
  37.2 Session State persistence
  37.5 Offline Documentation search and retrieval

Layouts, state, and walkthrough completion are stored under
``~/.opencut/panel/``.
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from opencut.helpers import OPENCUT_DIR

logger = logging.getLogger("opencut")

PANEL_DIR = os.path.join(OPENCUT_DIR, "panel")
LAYOUTS_DIR = os.path.join(PANEL_DIR, "layouts")
STATE_FILE = os.path.join(PANEL_DIR, "session_state.json")
WALKTHROUGH_FILE = os.path.join(PANEL_DIR, "walkthroughs_completed.json")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ===================================================================
# 6.1 — Drag-and-Drop Handler
# ===================================================================

@dataclass
class DropAction:
    """Maps a dropped file to an operation."""
    file_path: str
    operation: str
    timestamp: float = field(default_factory=time.time)


# In-memory registry of recent drop actions
_drop_registry: List[DropAction] = []


def register_drop_handler(file_path: str, operation: str) -> DropAction:
    """Register a drag-and-drop action mapping a file to an operation.

    Returns the created DropAction.
    """
    if not file_path:
        raise ValueError("file_path is required")
    if not operation:
        raise ValueError("operation is required")
    action = DropAction(file_path=file_path, operation=operation)
    _drop_registry.append(action)
    logger.info("Drop handler: %s -> %s", file_path, operation)
    return action


def get_drop_registry() -> List[DropAction]:
    """Return all registered drop actions."""
    return list(_drop_registry)


def clear_drop_registry() -> None:
    """Clear the drop action registry (for testing)."""
    _drop_registry.clear()


# ===================================================================
# 6.2 — Workspace Layouts
# ===================================================================

_BUILTIN_LAYOUTS = {
    "Assembly": {
        "name": "Assembly",
        "builtin": True,
        "panels": {
            "source_browser": {"visible": True, "position": "left", "size": "large"},
            "timeline": {"visible": True, "position": "bottom", "size": "medium"},
            "preview": {"visible": True, "position": "center", "size": "large"},
            "inspector": {"visible": False, "position": "right", "size": "small"},
        },
        "description": "Optimized for importing and organizing footage",
    },
    "Audio": {
        "name": "Audio",
        "builtin": True,
        "panels": {
            "waveform": {"visible": True, "position": "center", "size": "large"},
            "mixer": {"visible": True, "position": "right", "size": "medium"},
            "timeline": {"visible": True, "position": "bottom", "size": "large"},
            "effects": {"visible": True, "position": "left", "size": "small"},
        },
        "description": "Focused on audio editing and mixing",
    },
    "Color": {
        "name": "Color",
        "builtin": True,
        "panels": {
            "scopes": {"visible": True, "position": "right", "size": "medium"},
            "color_wheels": {"visible": True, "position": "center", "size": "large"},
            "timeline": {"visible": True, "position": "bottom", "size": "small"},
            "preview": {"visible": True, "position": "left", "size": "medium"},
        },
        "description": "Color grading and correction workspace",
    },
    "Delivery": {
        "name": "Delivery",
        "builtin": True,
        "panels": {
            "render_queue": {"visible": True, "position": "center", "size": "large"},
            "presets": {"visible": True, "position": "left", "size": "medium"},
            "preview": {"visible": True, "position": "right", "size": "medium"},
            "job_monitor": {"visible": True, "position": "bottom", "size": "small"},
        },
        "description": "Export and delivery focused workspace",
    },
}


def save_layout(name: str, state_json: Dict) -> Dict:
    """Save a workspace layout to disk."""
    if not name or not name.strip():
        raise ValueError("Layout name is required")
    _ensure_dir(LAYOUTS_DIR)
    layout = {
        "name": name,
        "builtin": False,
        "state": state_json,
        "saved_at": time.time(),
    }
    path = os.path.join(LAYOUTS_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(layout, fh, indent=2)
    logger.info("Layout saved: %s", name)
    return layout


def load_layout(name: str) -> Optional[Dict]:
    """Load a workspace layout by name.  Checks built-ins first, then disk."""
    if name in _BUILTIN_LAYOUTS:
        return dict(_BUILTIN_LAYOUTS[name])
    path = os.path.join(LAYOUTS_DIR, f"{name}.json")
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load layout %s: %s", name, exc)
    return None


def list_layouts() -> List[Dict]:
    """List all available layouts (built-in + custom)."""
    layouts = []
    for name, data in _BUILTIN_LAYOUTS.items():
        layouts.append({"name": name, "builtin": True, "description": data.get("description", "")})
    _ensure_dir(LAYOUTS_DIR)
    for fname in sorted(os.listdir(LAYOUTS_DIR)):
        if fname.endswith(".json"):
            lname = fname[:-5]
            if lname not in _BUILTIN_LAYOUTS:
                layouts.append({"name": lname, "builtin": False, "description": ""})
    return layouts


def delete_layout(name: str) -> bool:
    """Delete a custom layout.  Built-in layouts cannot be deleted."""
    if name in _BUILTIN_LAYOUTS:
        raise ValueError(f"Cannot delete built-in layout: {name}")
    path = os.path.join(LAYOUTS_DIR, f"{name}.json")
    if os.path.isfile(path):
        os.remove(path)
        logger.info("Layout deleted: %s", name)
        return True
    return False


# ===================================================================
# 6.6 — Right-Click Context Menu Actions
# ===================================================================

_CONTEXT_ACTIONS: Dict[str, List[Dict]] = {
    "video": [
        {"action": "silence_remove", "label": "Remove Silence", "icon": "volume-off"},
        {"action": "scene_detect", "label": "Detect Scenes", "icon": "film"},
        {"action": "stabilize", "label": "Stabilize", "icon": "crosshair"},
        {"action": "speed_ramp", "label": "Speed Ramp", "icon": "zap"},
        {"action": "bg_removal", "label": "Remove Background", "icon": "scissors"},
        {"action": "upscale", "label": "AI Upscale", "icon": "maximize"},
        {"action": "denoise", "label": "Denoise", "icon": "wind"},
        {"action": "color_match", "label": "Color Match", "icon": "droplet"},
        {"action": "export", "label": "Export", "icon": "download"},
        {"action": "captions", "label": "Generate Captions", "icon": "message-square"},
    ],
    "audio": [
        {"action": "silence_remove", "label": "Remove Silence", "icon": "volume-off"},
        {"action": "audio_enhance", "label": "Enhance Audio", "icon": "music"},
        {"action": "denoise", "label": "Denoise Audio", "icon": "wind"},
        {"action": "normalize", "label": "Normalize", "icon": "bar-chart"},
        {"action": "export", "label": "Export Audio", "icon": "download"},
    ],
    "image": [
        {"action": "bg_removal", "label": "Remove Background", "icon": "scissors"},
        {"action": "upscale", "label": "AI Upscale", "icon": "maximize"},
        {"action": "style_transfer", "label": "Style Transfer", "icon": "palette"},
        {"action": "face_enhance", "label": "Face Enhance", "icon": "smile"},
    ],
    "subtitle": [
        {"action": "translate", "label": "Translate", "icon": "globe"},
        {"action": "restyle", "label": "Restyle", "icon": "type"},
        {"action": "retime", "label": "Adjust Timing", "icon": "clock"},
    ],
}


def get_context_menu_actions(clip_type: str) -> List[Dict]:
    """Return applicable right-click actions for the given clip type.

    Falls back to the ``video`` action set for unknown types.
    """
    clip_type = (clip_type or "video").lower().strip()
    return list(_CONTEXT_ACTIONS.get(clip_type, _CONTEXT_ACTIONS["video"]))


# ===================================================================
# 6.7 — Quick Previews (single-frame operation preview)
# ===================================================================

@dataclass
class PreviewResult:
    """Result from generating a single-frame operation preview."""
    operation: str
    source_frame: str
    preview_path: str
    generated_at: float = field(default_factory=time.time)
    status: str = "generated"


def generate_operation_preview(operation: str, frame_path: str) -> PreviewResult:
    """Apply an operation to a single frame for quick preview.

    This is a stub — in production it would call the actual operation
    on the single frame.  Returns metadata about the preview.
    """
    if not operation:
        raise ValueError("operation is required")
    if not frame_path:
        raise ValueError("frame_path is required")
    # In a real implementation, this would run the actual operation on the frame
    preview_path = frame_path.rsplit(".", 1)[0] + f"_preview_{operation}.png"
    return PreviewResult(
        operation=operation,
        source_frame=frame_path,
        preview_path=preview_path,
    )


# ===================================================================
# 6.8 — Theme Toggle
# ===================================================================

_THEMES: Dict[str, Dict[str, str]] = {
    "dark": {
        "name": "dark",
        "--bg-primary": "#1a1a2e",
        "--bg-secondary": "#16213e",
        "--bg-surface": "#0f3460",
        "--text-primary": "#e0e0e0",
        "--text-secondary": "#a0a0b0",
        "--accent": "#e94560",
        "--accent-hover": "#ff6b81",
        "--border": "#2a2a4e",
        "--success": "#2ecc71",
        "--warning": "#f39c12",
        "--error": "#e74c3c",
        "--scrollbar-bg": "#16213e",
        "--scrollbar-thumb": "#0f3460",
    },
    "light": {
        "name": "light",
        "--bg-primary": "#f5f5f5",
        "--bg-secondary": "#ffffff",
        "--bg-surface": "#e8e8e8",
        "--text-primary": "#1a1a2e",
        "--text-secondary": "#555555",
        "--accent": "#e94560",
        "--accent-hover": "#c73a52",
        "--border": "#d0d0d0",
        "--success": "#27ae60",
        "--warning": "#e67e22",
        "--error": "#c0392b",
        "--scrollbar-bg": "#e8e8e8",
        "--scrollbar-thumb": "#c0c0c0",
    },
    "system": {
        "name": "system",
        "--note": "Inherits from OS preference via prefers-color-scheme",
        "--bg-primary": "var(--system-bg-primary)",
        "--bg-secondary": "var(--system-bg-secondary)",
        "--bg-surface": "var(--system-bg-surface)",
        "--text-primary": "var(--system-text-primary)",
        "--text-secondary": "var(--system-text-secondary)",
        "--accent": "#e94560",
        "--accent-hover": "var(--system-accent-hover)",
        "--border": "var(--system-border)",
        "--success": "#2ecc71",
        "--warning": "#f39c12",
        "--error": "#e74c3c",
        "--scrollbar-bg": "var(--system-scrollbar-bg)",
        "--scrollbar-thumb": "var(--system-scrollbar-thumb)",
    },
}


def get_theme(name: str) -> Optional[Dict[str, str]]:
    """Return CSS custom property values for a theme."""
    return dict(_THEMES[name]) if name in _THEMES else None


def list_themes() -> List[Dict[str, str]]:
    """Return summary info for all available themes."""
    return [{"name": t["name"]} for t in _THEMES.values()]


# ===================================================================
# 37.1 — Guided Walkthroughs
# ===================================================================

@dataclass
class WalkthroughStep:
    """A single step in a guided walkthrough."""
    step_number: int
    title: str
    description: str
    target_element: str = ""
    action: str = ""


@dataclass
class Walkthrough:
    """A guided walkthrough for a feature."""
    feature_id: str
    title: str
    description: str
    steps: List[WalkthroughStep] = field(default_factory=list)


_WALKTHROUGHS: Dict[str, Walkthrough] = {
    "first_import": Walkthrough(
        feature_id="first_import",
        title="Importing Your First Video",
        description="Learn how to import media files into OpenCut",
        steps=[
            WalkthroughStep(1, "Open File Browser", "Click the upload area or drag files onto it", "upload-zone", "click"),
            WalkthroughStep(2, "Select Media", "Choose a video, audio, or image file from your computer", "file-input", "select"),
            WalkthroughStep(3, "Confirm Upload", "Your file will appear in the file list once uploaded", "file-list", "observe"),
        ],
    ),
    "first_operation": Walkthrough(
        feature_id="first_operation",
        title="Running Your First Operation",
        description="Apply an AI operation to your media",
        steps=[
            WalkthroughStep(1, "Select Operation", "Browse the operation cards and click one", "ops-grid", "click"),
            WalkthroughStep(2, "Configure Parameters", "Adjust settings in the operation panel", "op-params", "configure"),
            WalkthroughStep(3, "Run", "Click the Run button to start processing", "run-button", "click"),
            WalkthroughStep(4, "Monitor Progress", "Watch the job status bar for progress", "job-status", "observe"),
        ],
    ),
    "workspace_setup": Walkthrough(
        feature_id="workspace_setup",
        title="Customizing Your Workspace",
        description="Arrange panels and save custom layouts",
        steps=[
            WalkthroughStep(1, "Choose a Layout", "Select from Assembly, Audio, Color, or Delivery presets", "layout-selector", "click"),
            WalkthroughStep(2, "Resize Panels", "Drag panel edges to resize them", "panel-divider", "drag"),
            WalkthroughStep(3, "Save Layout", "Save your custom arrangement for later use", "save-layout-btn", "click"),
        ],
    ),
    "caption_workflow": Walkthrough(
        feature_id="caption_workflow",
        title="Generating Captions",
        description="Use AI to automatically generate and edit captions",
        steps=[
            WalkthroughStep(1, "Upload Video", "Import a video with spoken content", "upload-zone", "click"),
            WalkthroughStep(2, "Select Captions", "Choose the Auto Captions operation", "op-captions", "click"),
            WalkthroughStep(3, "Choose Language", "Select the spoken language or use auto-detect", "lang-select", "select"),
            WalkthroughStep(4, "Generate", "Click Run to start transcription", "run-button", "click"),
            WalkthroughStep(5, "Review & Edit", "Review generated captions and make corrections", "caption-editor", "edit"),
        ],
    ),
    "color_grading": Walkthrough(
        feature_id="color_grading",
        title="Color Grading Basics",
        description="Match colors and apply LUTs to your footage",
        steps=[
            WalkthroughStep(1, "Switch to Color Layout", "Select the Color workspace layout", "layout-color", "click"),
            WalkthroughStep(2, "Open Scopes", "View the video scopes panel", "scopes-panel", "observe"),
            WalkthroughStep(3, "Apply Color Match", "Use color match to match grading between clips", "op-color-match", "click"),
            WalkthroughStep(4, "Fine Tune", "Adjust color wheels and curves", "color-wheels", "adjust"),
        ],
    ),
    "export_delivery": Walkthrough(
        feature_id="export_delivery",
        title="Exporting Your Project",
        description="Export video with platform-optimized presets",
        steps=[
            WalkthroughStep(1, "Switch to Delivery", "Select the Delivery workspace layout", "layout-delivery", "click"),
            WalkthroughStep(2, "Choose Preset", "Select an export preset (YouTube, TikTok, etc.)", "preset-selector", "click"),
            WalkthroughStep(3, "Configure Output", "Set output path and any custom settings", "output-config", "configure"),
            WalkthroughStep(4, "Export", "Click Export to render your video", "export-button", "click"),
        ],
    ),
}


def _load_completed_walkthroughs() -> Dict[str, float]:
    """Load walkthrough completion records from disk."""
    if os.path.isfile(WALKTHROUGH_FILE):
        try:
            with open(WALKTHROUGH_FILE, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_completed_walkthroughs(completed: Dict[str, float]) -> None:
    """Persist walkthrough completion records."""
    _ensure_dir(PANEL_DIR)
    with open(WALKTHROUGH_FILE, "w", encoding="utf-8") as fh:
        json.dump(completed, fh, indent=2)


def get_walkthrough(feature_id: str) -> Optional[Dict]:
    """Return walkthrough steps for a feature, with completion status."""
    wt = _WALKTHROUGHS.get(feature_id)
    if wt is None:
        return None
    completed = _load_completed_walkthroughs()
    return {
        "feature_id": wt.feature_id,
        "title": wt.title,
        "description": wt.description,
        "steps": [asdict(s) for s in wt.steps],
        "completed": feature_id in completed,
        "completed_at": completed.get(feature_id),
    }


def list_walkthroughs() -> List[Dict]:
    """List all available walkthroughs with completion status."""
    completed = _load_completed_walkthroughs()
    result = []
    for fid, wt in _WALKTHROUGHS.items():
        result.append({
            "feature_id": fid,
            "title": wt.title,
            "description": wt.description,
            "num_steps": len(wt.steps),
            "completed": fid in completed,
        })
    return result


def mark_walkthrough_completed(feature_id: str) -> bool:
    """Mark a walkthrough as completed.  Returns True if valid feature_id."""
    if feature_id not in _WALKTHROUGHS:
        return False
    completed = _load_completed_walkthroughs()
    completed[feature_id] = time.time()
    _save_completed_walkthroughs(completed)
    return True


# ===================================================================
# 37.2 — Session State Persistence
# ===================================================================

def save_session_state(state_json: Dict) -> Dict:
    """Save panel session state to disk."""
    _ensure_dir(PANEL_DIR)
    state_record = {
        "state": state_json,
        "saved_at": time.time(),
    }
    with open(STATE_FILE, "w", encoding="utf-8") as fh:
        json.dump(state_record, fh, indent=2)
    logger.info("Session state saved")
    return state_record


def restore_session_state() -> Optional[Dict]:
    """Restore the last saved panel session state."""
    if not os.path.isfile(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to restore session state: %s", exc)
        return None


# ===================================================================
# 37.5 — Offline Documentation
# ===================================================================

_DOCUMENTATION: Dict[str, Dict[str, str]] = {
    "getting_started": {
        "topic": "getting_started",
        "title": "Getting Started with OpenCut",
        "content": (
            "OpenCut is an AI-powered video editing backend. "
            "To get started, upload a video file and select an operation from the catalog. "
            "Operations run as background jobs -- you can monitor progress in the status bar. "
            "Results are saved to your configured output directory."
        ),
    },
    "silence_removal": {
        "topic": "silence_removal",
        "title": "Silence Removal",
        "content": (
            "Silence Removal detects and removes silent segments from audio or video. "
            "Parameters: threshold (dB level for silence detection, default -35), "
            "min_duration (minimum silence length in seconds, default 0.5). "
            "The operation preserves non-silent segments and concatenates them."
        ),
    },
    "background_removal": {
        "topic": "background_removal",
        "title": "Background Removal",
        "content": (
            "AI-powered background removal using segmentation models. "
            "Supports transparent background, solid color replacement, or custom background image. "
            "Works best with well-lit subjects against contrasting backgrounds."
        ),
    },
    "upscaling": {
        "topic": "upscaling",
        "title": "AI Video Upscaling",
        "content": (
            "Upscale video resolution using AI super-resolution models. "
            "Supports 2x and 4x scaling. Best results with source footage 720p or higher. "
            "GPU acceleration is recommended for large videos."
        ),
    },
    "captions": {
        "topic": "captions",
        "title": "Auto Captions",
        "content": (
            "Generate captions using Whisper speech recognition. "
            "Supports 90+ languages with automatic language detection. "
            "Output formats include SRT, VTT, and JSON. "
            "Models: tiny, base, small, medium, large-v3, turbo."
        ),
    },
    "color_grading": {
        "topic": "color_grading",
        "title": "Color Grading & Matching",
        "content": (
            "Match color grading between clips or apply LUTs. "
            "Color Match analyzes a reference frame and adjusts the target footage. "
            "Color Scopes provide waveform, vectorscope, histogram, and parade displays. "
            "Supports ACES color management pipeline."
        ),
    },
    "export_presets": {
        "topic": "export_presets",
        "title": "Export Presets",
        "content": (
            "Pre-configured export profiles for YouTube (1080p, 4K, Shorts), "
            "TikTok, Instagram (Feed, Reels, Story), Twitter/X, LinkedIn, "
            "podcast audio, and archive formats (ProRes, DNxHR). "
            "Hardware-accelerated encoding via NVENC, QSV, AMF, VideoToolbox."
        ),
    },
    "keyboard_shortcuts": {
        "topic": "keyboard_shortcuts",
        "title": "Keyboard Shortcuts",
        "content": (
            "Customize keyboard shortcuts in the settings panel. "
            "Default shortcuts: Space (play/pause), J/K/L (shuttle), "
            "I/O (in/out points), C (cut), V (select), B (blade). "
            "All shortcuts can be remapped in Settings > Keyboard."
        ),
    },
    "workspace_layouts": {
        "topic": "workspace_layouts",
        "title": "Workspace Layouts",
        "content": (
            "OpenCut provides four built-in workspace layouts: "
            "Assembly (import/organize), Audio (editing/mixing), "
            "Color (grading/correction), and Delivery (export). "
            "Save custom layouts and switch between them via the layout menu."
        ),
    },
    "ae_integration": {
        "topic": "ae_integration",
        "title": "After Effects Integration",
        "content": (
            "The OpenCut After Effects extension adds an AI processing panel "
            "to After Effects CC 2019+. Supported operations: background removal, "
            "upscale, style transfer, object removal, depth effects, denoise, "
            "and face enhancement. Processes are sent to the OpenCut backend."
        ),
    },
    "api_reference": {
        "topic": "api_reference",
        "title": "API Reference",
        "content": (
            "All OpenCut operations are accessible via REST API. "
            "POST requests require a CSRF token in the X-OpenCut-Token header. "
            "File paths are passed as 'filepath' in the JSON body. "
            "Jobs return a job_id for progress polling via GET /jobs/<id>."
        ),
    },
    "troubleshooting": {
        "topic": "troubleshooting",
        "title": "Troubleshooting",
        "content": (
            "Common issues: FFmpeg not found (install FFmpeg and add to PATH), "
            "GPU out of memory (reduce batch size or use CPU mode), "
            "model download failed (check internet connection, retry). "
            "Logs are stored in ~/.opencut/opencut.log."
        ),
    },
}


def get_documentation(topic: str) -> Optional[Dict[str, str]]:
    """Return documentation for a topic."""
    return dict(_DOCUMENTATION[topic]) if topic in _DOCUMENTATION else None


def search_docs(query: str) -> List[Dict[str, str]]:
    """Search documentation by keyword matching on title and content."""
    if not query or not query.strip():
        return []
    terms = query.lower().split()
    results = []
    for topic, doc in _DOCUMENTATION.items():
        text = (doc["title"] + " " + doc["content"]).lower()
        score = sum(1 for term in terms if term in text)
        if score > 0:
            results.append({
                "topic": doc["topic"],
                "title": doc["title"],
                "score": score,
                "snippet": doc["content"][:120] + "..." if len(doc["content"]) > 120 else doc["content"],
            })
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def list_doc_topics() -> List[Dict[str, str]]:
    """List all available documentation topics."""
    return [
        {"topic": doc["topic"], "title": doc["title"]}
        for doc in _DOCUMENTATION.values()
    ]
