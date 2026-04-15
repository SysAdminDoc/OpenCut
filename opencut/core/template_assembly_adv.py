"""
Advanced Template-Based Video Assembly (Category 74)

Assemble video from JSON templates with typed slots. Each template defines
intro, segments, outro, lower thirds, and music bed positions. Media is
auto-trimmed to fit slot durations, with transitions between segments.

Built-in templates: youtube_video, podcast_video, tutorial, social_clip.
Supports custom template validation and EDL/OTIO-compatible JSON export.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import get_video_info

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SLOT_TYPES = ("video", "image", "text", "audio")
MAX_SLOTS = 100
MAX_SEGMENTS = 50
DEFAULT_TRANSITION = "dissolve"
DEFAULT_TRANSITION_DURATION = 0.5


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class TemplateSlot:
    """A slot in a video template."""
    name: str = ""
    slot_type: str = "video"  # video, image, text, audio
    duration: float = 0.0
    min_duration: float = 0.0
    max_duration: float = 0.0
    position: Dict = field(default_factory=dict)  # x, y, width, height
    required: bool = True
    default_value: str = ""
    layer: int = 0
    transition_in: str = ""
    transition_out: str = ""
    transition_duration: float = DEFAULT_TRANSITION_DURATION

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "slot_type": self.slot_type,
            "duration": self.duration,
            "min_duration": self.min_duration,
            "max_duration": self.max_duration,
            "position": self.position,
            "required": self.required,
            "default_value": self.default_value,
            "layer": self.layer,
            "transition_in": self.transition_in,
            "transition_out": self.transition_out,
            "transition_duration": self.transition_duration,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TemplateSlot":
        return cls(
            name=data.get("name", ""),
            slot_type=data.get("slot_type", data.get("type", "video")),
            duration=float(data.get("duration", 0)),
            min_duration=float(data.get("min_duration", 0)),
            max_duration=float(data.get("max_duration", 0)),
            position=data.get("position", {}),
            required=bool(data.get("required", True)),
            default_value=data.get("default_value", data.get("default", "")),
            layer=int(data.get("layer", 0)),
            transition_in=data.get("transition_in", ""),
            transition_out=data.get("transition_out", ""),
            transition_duration=float(
                data.get("transition_duration", DEFAULT_TRANSITION_DURATION)
            ),
        )


@dataclass
class Template:
    """A video assembly template."""
    name: str = ""
    description: str = ""
    category: str = "general"
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    slots: List[TemplateSlot] = field(default_factory=list)
    default_transition: str = DEFAULT_TRANSITION
    default_transition_duration: float = DEFAULT_TRANSITION_DURATION
    metadata: Dict = field(default_factory=dict)

    @property
    def total_duration(self) -> float:
        return sum(s.duration for s in self.slots if s.slot_type != "audio")

    @property
    def required_slots(self) -> List[TemplateSlot]:
        return [s for s in self.slots if s.required]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "slots": [s.to_dict() for s in self.slots],
            "total_duration": self.total_duration,
            "default_transition": self.default_transition,
            "default_transition_duration": self.default_transition_duration,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Template":
        slots = [TemplateSlot.from_dict(s) for s in data.get("slots", [])]
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            category=data.get("category", "general"),
            width=int(data.get("width", 1920)),
            height=int(data.get("height", 1080)),
            fps=float(data.get("fps", 30.0)),
            slots=slots,
            default_transition=data.get("default_transition", DEFAULT_TRANSITION),
            default_transition_duration=float(
                data.get("default_transition_duration", DEFAULT_TRANSITION_DURATION)
            ),
            metadata=data.get("metadata", {}),
        )


@dataclass
class AssembledClip:
    """A clip placed in the assembled timeline."""
    slot_name: str = ""
    source_file: str = ""
    source_in: float = 0.0
    source_out: float = 0.0
    record_in: float = 0.0
    record_out: float = 0.0
    slot_type: str = "video"
    layer: int = 0
    transition_in: str = ""
    transition_out: str = ""
    transition_duration: float = 0.0
    trimmed: bool = False
    text_content: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.record_out - self.record_in)

    def to_dict(self) -> dict:
        return {
            "slot_name": self.slot_name,
            "source_file": self.source_file,
            "source_in": round(self.source_in, 3),
            "source_out": round(self.source_out, 3),
            "record_in": round(self.record_in, 3),
            "record_out": round(self.record_out, 3),
            "duration": round(self.duration, 3),
            "slot_type": self.slot_type,
            "layer": self.layer,
            "transition_in": self.transition_in,
            "transition_out": self.transition_out,
            "transition_duration": self.transition_duration,
            "trimmed": self.trimmed,
            "text_content": self.text_content,
        }


@dataclass
class ValidationResult:
    """Result of template validation."""
    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
        }


@dataclass
class AssemblyResult:
    """Result of template assembly."""
    clips: List[AssembledClip] = field(default_factory=list)
    total_duration: float = 0.0
    template_name: str = ""
    missing_slots: List[str] = field(default_factory=list)
    edl_text: str = ""
    otio_json: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "clips": [c.to_dict() for c in self.clips],
            "total_duration": round(self.total_duration, 3),
            "template_name": self.template_name,
            "clip_count": len(self.clips),
            "missing_slots": self.missing_slots,
            "edl_text": self.edl_text,
            "has_otio": bool(self.otio_json),
        }


# ---------------------------------------------------------------------------
# Built-in templates
# ---------------------------------------------------------------------------
def _make_youtube_template() -> Template:
    """YouTube video template: intro + segments + outro + end screen."""
    return Template(
        name="youtube_video",
        description="Standard YouTube video with intro, segments, and outro",
        category="youtube",
        slots=[
            TemplateSlot(name="intro", slot_type="video", duration=5.0,
                         min_duration=3.0, max_duration=10.0, required=True,
                         layer=0, transition_out="dissolve"),
            TemplateSlot(name="segment_1", slot_type="video", duration=30.0,
                         min_duration=10.0, max_duration=300.0, required=True,
                         layer=0),
            TemplateSlot(name="segment_2", slot_type="video", duration=30.0,
                         min_duration=10.0, max_duration=300.0, required=False,
                         layer=0),
            TemplateSlot(name="segment_3", slot_type="video", duration=30.0,
                         min_duration=10.0, max_duration=300.0, required=False,
                         layer=0),
            TemplateSlot(name="outro", slot_type="video", duration=5.0,
                         min_duration=3.0, max_duration=15.0, required=True,
                         layer=0, transition_in="dissolve"),
            TemplateSlot(name="end_screen", slot_type="image", duration=20.0,
                         min_duration=5.0, max_duration=20.0, required=False,
                         layer=0),
            TemplateSlot(name="lower_third", slot_type="text", duration=5.0,
                         min_duration=3.0, max_duration=10.0, required=False,
                         layer=1, default_value="Subscribe!"),
            TemplateSlot(name="music_bed", slot_type="audio", duration=0.0,
                         required=False, layer=0),
        ],
    )


def _make_podcast_template() -> Template:
    """Podcast video: static background + waveform position + captions."""
    return Template(
        name="podcast_video",
        description="Podcast video with static background and audio visualization",
        category="podcast",
        slots=[
            TemplateSlot(name="background", slot_type="image", duration=0.0,
                         required=True, layer=0),
            TemplateSlot(name="audio", slot_type="audio", duration=0.0,
                         required=True, layer=0),
            TemplateSlot(name="waveform", slot_type="video", duration=0.0,
                         required=False, layer=1,
                         position={"x": 0, "y": 540, "width": 1920, "height": 200}),
            TemplateSlot(name="captions", slot_type="text", duration=0.0,
                         required=False, layer=2,
                         position={"x": 0, "y": 800, "width": 1920, "height": 100}),
            TemplateSlot(name="intro", slot_type="video", duration=3.0,
                         min_duration=1.0, max_duration=10.0, required=False,
                         layer=0, transition_out="dissolve"),
            TemplateSlot(name="outro", slot_type="video", duration=3.0,
                         min_duration=1.0, max_duration=10.0, required=False,
                         layer=0, transition_in="dissolve"),
        ],
    )


def _make_tutorial_template() -> Template:
    """Tutorial: screen recording + face cam overlay + chapters."""
    return Template(
        name="tutorial",
        description="Tutorial with screen recording, facecam overlay, and chapters",
        category="education",
        slots=[
            TemplateSlot(name="intro", slot_type="video", duration=5.0,
                         min_duration=2.0, max_duration=15.0, required=True,
                         layer=0, transition_out="dissolve"),
            TemplateSlot(name="screen", slot_type="video", duration=60.0,
                         min_duration=10.0, max_duration=3600.0, required=True,
                         layer=0),
            TemplateSlot(name="facecam", slot_type="video", duration=0.0,
                         required=False, layer=1,
                         position={"x": 1560, "y": 720, "width": 320, "height": 240}),
            TemplateSlot(name="chapter_1", slot_type="text", duration=3.0,
                         required=False, layer=2, default_value="Chapter 1"),
            TemplateSlot(name="chapter_2", slot_type="text", duration=3.0,
                         required=False, layer=2, default_value="Chapter 2"),
            TemplateSlot(name="chapter_3", slot_type="text", duration=3.0,
                         required=False, layer=2, default_value="Chapter 3"),
            TemplateSlot(name="outro", slot_type="video", duration=5.0,
                         min_duration=2.0, max_duration=15.0, required=False,
                         layer=0, transition_in="dissolve"),
            TemplateSlot(name="music_bed", slot_type="audio", duration=0.0,
                         required=False, layer=0),
        ],
    )


def _make_social_clip_template() -> Template:
    """Social media clip: hook + content + CTA."""
    return Template(
        name="social_clip",
        description="Short social media clip with hook, content, and call-to-action",
        category="social",
        width=1080,
        height=1920,
        slots=[
            TemplateSlot(name="hook", slot_type="video", duration=3.0,
                         min_duration=1.0, max_duration=5.0, required=True,
                         layer=0),
            TemplateSlot(name="content", slot_type="video", duration=25.0,
                         min_duration=5.0, max_duration=55.0, required=True,
                         layer=0),
            TemplateSlot(name="cta", slot_type="video", duration=2.0,
                         min_duration=1.0, max_duration=5.0, required=True,
                         layer=0, transition_in="dissolve"),
            TemplateSlot(name="caption_overlay", slot_type="text", duration=0.0,
                         required=False, layer=1,
                         position={"x": 0, "y": 1400, "width": 1080, "height": 200}),
            TemplateSlot(name="music_bed", slot_type="audio", duration=0.0,
                         required=False, layer=0),
        ],
    )


BUILTIN_TEMPLATES = {
    "youtube_video": _make_youtube_template,
    "podcast_video": _make_podcast_template,
    "tutorial": _make_tutorial_template,
    "social_clip": _make_social_clip_template,
}


def get_builtin_templates() -> Dict[str, Template]:
    """Return all built-in templates."""
    return {name: factory() for name, factory in BUILTIN_TEMPLATES.items()}


def list_templates() -> List[Dict]:
    """List all available templates with summary info."""
    templates = []
    for name, factory in BUILTIN_TEMPLATES.items():
        tmpl = factory()
        templates.append({
            "name": tmpl.name,
            "description": tmpl.description,
            "category": tmpl.category,
            "slot_count": len(tmpl.slots),
            "required_slots": len(tmpl.required_slots),
            "total_duration": tmpl.total_duration,
            "resolution": f"{tmpl.width}x{tmpl.height}",
        })
    return templates


# ---------------------------------------------------------------------------
# Template validation
# ---------------------------------------------------------------------------
def validate_template(template_data: Dict) -> ValidationResult:
    """Validate a template definition.

    Args:
        template_data: Template definition dict.

    Returns:
        ValidationResult with errors and warnings.
    """
    result = ValidationResult()

    # Check required fields
    if not template_data.get("name"):
        result.errors.append("Template must have a 'name' field")
        result.valid = False

    slots = template_data.get("slots", [])
    if not slots:
        result.errors.append("Template must have at least one slot")
        result.valid = False
        return result

    if len(slots) > MAX_SLOTS:
        result.errors.append(f"Too many slots ({len(slots)}). Max: {MAX_SLOTS}")
        result.valid = False

    seen_names = set()
    has_required_video = False

    for i, slot in enumerate(slots):
        name = slot.get("name", "")
        if not name:
            result.errors.append(f"Slot {i} is missing 'name'")
            result.valid = False
            continue

        if name in seen_names:
            result.errors.append(f"Duplicate slot name: '{name}'")
            result.valid = False
        seen_names.add(name)

        slot_type = slot.get("slot_type", slot.get("type", ""))
        if slot_type not in SLOT_TYPES:
            result.errors.append(
                f"Slot '{name}' has invalid type '{slot_type}'. "
                f"Use: {', '.join(SLOT_TYPES)}"
            )
            result.valid = False

        # Check duration constraints
        duration = float(slot.get("duration", 0))
        min_dur = float(slot.get("min_duration", 0))
        max_dur = float(slot.get("max_duration", 0))

        if min_dur > 0 and max_dur > 0 and min_dur > max_dur:
            result.errors.append(
                f"Slot '{name}': min_duration ({min_dur}) > max_duration ({max_dur})"
            )
            result.valid = False

        if duration > 0 and max_dur > 0 and duration > max_dur:
            result.warnings.append(
                f"Slot '{name}': default duration ({duration}) > max_duration ({max_dur})"
            )

        if slot.get("required", False) and slot_type in ("video", "image"):
            has_required_video = True

    if not has_required_video:
        result.warnings.append("Template has no required video or image slots")

    # Check resolution
    width = int(template_data.get("width", 0))
    height = int(template_data.get("height", 0))
    if width > 0 and height > 0:
        if width < 100 or height < 100:
            result.warnings.append(
                f"Resolution {width}x{height} is unusually small"
            )
        if width > 7680 or height > 4320:
            result.warnings.append(
                f"Resolution {width}x{height} is unusually large"
            )

    return result


# ---------------------------------------------------------------------------
# Media fitting
# ---------------------------------------------------------------------------
def _fit_media_to_slot(
    media_path: str,
    slot: TemplateSlot,
) -> tuple:
    """Determine source in/out to fit media into slot duration.

    Returns (source_in, source_out, trimmed) tuple.
    """
    if not media_path or not os.path.isfile(media_path):
        return 0.0, slot.duration, False

    info = get_video_info(media_path)
    media_duration = info.get("duration", 0.0)

    target_duration = slot.duration

    if target_duration <= 0:
        # No fixed duration: use full media
        return 0.0, media_duration, False

    if media_duration <= 0:
        return 0.0, target_duration, False

    if media_duration <= target_duration:
        # Media shorter than slot: use full media
        return 0.0, media_duration, False

    # Media longer than slot: trim to fit
    # Center the trim
    excess = media_duration - target_duration
    source_in = excess / 2.0
    source_out = source_in + target_duration
    return source_in, source_out, True


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def assemble_from_template(
    template: Template,
    media_map: Dict[str, str],
    on_progress: Optional[Callable] = None,
) -> AssemblyResult:
    """Assemble timeline from template and media map.

    Args:
        template: Template object defining the structure.
        media_map: Dict mapping slot names to file paths or text content.
        on_progress: Callback(pct) for progress.

    Returns:
        AssemblyResult with assembled clip list and EDL.
    """
    if not template.slots:
        raise ValueError("Template has no slots")

    assembled_clips = []
    missing_slots = []
    record_pos = 0.0
    total_slots = len(template.slots)

    for i, slot in enumerate(template.slots):
        if on_progress:
            on_progress(int((i / max(total_slots, 1)) * 85))

        media = media_map.get(slot.name, "")

        if not media and slot.required:
            missing_slots.append(slot.name)
            if slot.default_value:
                media = slot.default_value
            else:
                continue

        if not media:
            continue

        # Handle text slots
        if slot.slot_type == "text":
            text_content = media if isinstance(media, str) else str(media)
            duration = slot.duration if slot.duration > 0 else 5.0
            clip = AssembledClip(
                slot_name=slot.name,
                source_file="",
                source_in=0.0,
                source_out=duration,
                record_in=record_pos,
                record_out=record_pos + duration,
                slot_type="text",
                layer=slot.layer,
                transition_in=slot.transition_in or template.default_transition,
                transition_out=slot.transition_out,
                transition_duration=slot.transition_duration,
                text_content=text_content,
            )
            assembled_clips.append(clip)
            if slot.layer == 0:
                record_pos += duration
            continue

        # Handle audio slots
        if slot.slot_type == "audio":
            media_path = media if isinstance(media, str) else ""
            if media_path and os.path.isfile(media_path):
                info = get_video_info(media_path)
                audio_dur = info.get("duration", 0.0)
            else:
                audio_dur = 0.0

            clip = AssembledClip(
                slot_name=slot.name,
                source_file=media_path,
                source_in=0.0,
                source_out=audio_dur,
                record_in=0.0,
                record_out=audio_dur,
                slot_type="audio",
                layer=slot.layer,
            )
            assembled_clips.append(clip)
            continue

        # Handle video/image slots
        media_path = media if isinstance(media, str) else ""
        source_in, source_out, trimmed = _fit_media_to_slot(media_path, slot)

        clip_duration = source_out - source_in
        if clip_duration <= 0:
            clip_duration = slot.duration if slot.duration > 0 else 5.0
            source_out = source_in + clip_duration

        # Enforce min/max duration
        if slot.min_duration > 0 and clip_duration < slot.min_duration:
            clip_duration = slot.min_duration
            source_out = source_in + clip_duration
        if slot.max_duration > 0 and clip_duration > slot.max_duration:
            clip_duration = slot.max_duration
            source_out = source_in + clip_duration
            trimmed = True

        clip = AssembledClip(
            slot_name=slot.name,
            source_file=media_path,
            source_in=source_in,
            source_out=source_out,
            record_in=record_pos,
            record_out=record_pos + clip_duration,
            slot_type=slot.slot_type,
            layer=slot.layer,
            transition_in=slot.transition_in or "",
            transition_out=slot.transition_out or "",
            transition_duration=slot.transition_duration,
            trimmed=trimmed,
        )
        assembled_clips.append(clip)
        if slot.layer == 0:
            record_pos += clip_duration

    if on_progress:
        on_progress(85)

    # Generate EDL
    edl_text = _generate_assembly_edl(assembled_clips, template.name, template.fps)

    if on_progress:
        on_progress(90)

    # Generate OTIO-compatible JSON
    otio_json = _generate_otio_json(assembled_clips, template)

    if on_progress:
        on_progress(95)

    total_duration = record_pos

    result = AssemblyResult(
        clips=assembled_clips,
        total_duration=total_duration,
        template_name=template.name,
        missing_slots=missing_slots,
        edl_text=edl_text,
        otio_json=otio_json,
    )

    if on_progress:
        on_progress(100)

    return result


# ---------------------------------------------------------------------------
# EDL generation
# ---------------------------------------------------------------------------
def _format_tc(seconds: float, fps: float = 30.0) -> str:
    """Format seconds as SMPTE timecode."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    f = int((seconds % 1.0) * fps)
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"


def _generate_assembly_edl(
    clips: List[AssembledClip],
    title: str = "Template Assembly",
    fps: float = 30.0,
) -> str:
    """Generate EDL from assembled clips."""
    lines = [
        f"TITLE: {title}",
        "FCM: NON-DROP FRAME",
        "",
    ]

    video_clips = [c for c in clips if c.slot_type in ("video", "image")]
    for i, clip in enumerate(video_clips):
        edit_num = f"{i + 1:03d}"
        reel = clip.slot_name[:8].upper() or f"REEL{i + 1:03d}"

        if clip.transition_in == "dissolve":
            trans = f"D    {int(clip.transition_duration * fps):03d}"
        else:
            trans = "C   "

        src_in = _format_tc(clip.source_in, fps)
        src_out = _format_tc(clip.source_out, fps)
        rec_in = _format_tc(clip.record_in, fps)
        rec_out = _format_tc(clip.record_out, fps)

        lines.append(
            f"{edit_num}  {reel:8s}  V     {trans}  {src_in} {src_out} {rec_in} {rec_out}"
        )
        if clip.source_file:
            lines.append(f"* FROM CLIP NAME: {os.path.basename(clip.source_file)}")
        lines.append(f"* SLOT: {clip.slot_name}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# OTIO-compatible JSON export
# ---------------------------------------------------------------------------
def _generate_otio_json(
    clips: List[AssembledClip],
    template: Template,
) -> Dict:
    """Generate OpenTimelineIO-compatible JSON structure."""
    tracks = {}
    for clip in clips:
        layer = clip.layer
        if layer not in tracks:
            tracks[layer] = []

        otio_clip = {
            "OTIO_SCHEMA": "Clip.1",
            "name": clip.slot_name,
            "source_range": {
                "OTIO_SCHEMA": "TimeRange.1",
                "start_time": {
                    "OTIO_SCHEMA": "RationalTime.1",
                    "rate": template.fps,
                    "value": clip.source_in * template.fps,
                },
                "duration": {
                    "OTIO_SCHEMA": "RationalTime.1",
                    "rate": template.fps,
                    "value": clip.duration * template.fps,
                },
            },
            "media_reference": {
                "OTIO_SCHEMA": "ExternalReference.1",
                "target_url": clip.source_file,
            } if clip.source_file else {
                "OTIO_SCHEMA": "GeneratorReference.1",
                "generator_kind": clip.slot_type,
                "parameters": {"text": clip.text_content} if clip.text_content else {},
            },
            "metadata": {
                "slot_name": clip.slot_name,
                "slot_type": clip.slot_type,
                "template": template.name,
            },
        }
        tracks[layer].append(otio_clip)

    otio_tracks = []
    for layer_idx in sorted(tracks.keys()):
        otio_tracks.append({
            "OTIO_SCHEMA": "Track.1",
            "name": f"Layer {layer_idx}",
            "kind": "Video" if layer_idx == 0 else "Video",
            "children": tracks[layer_idx],
        })

    return {
        "OTIO_SCHEMA": "Timeline.1",
        "name": template.name,
        "tracks": {
            "OTIO_SCHEMA": "Stack.1",
            "name": "tracks",
            "children": otio_tracks,
        },
        "metadata": {
            "template": template.name,
            "resolution": f"{template.width}x{template.height}",
            "fps": template.fps,
        },
    }


# ---------------------------------------------------------------------------
# Convenience: load template from JSON file
# ---------------------------------------------------------------------------
def load_template(path: str) -> Template:
    """Load a template from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Template.from_dict(data)


def save_template(template: Template, path: str) -> str:
    """Save a template to a JSON file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(template.to_dict(), f, indent=2)
    return path
