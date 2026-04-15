"""
Batch Timeline Operations (Category 74)

Batch operations on timeline clip lists: color grade, speed change,
audio normalize, transitions, crop/reframe, caption generation.

Supports operation chaining (pipeline), dry-run preview mode, and
per-clip progress callbacks.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BATCH_OPS = (
    "color_grade", "speed", "normalize", "transition",
    "crop", "caption",
)

DEFAULT_LUT_PATH = ""
DEFAULT_SPEED = 1.0
DEFAULT_NORMALIZE_TARGET = -16.0
DEFAULT_TRANSITION = "dissolve"
DEFAULT_TRANSITION_DURATION = 0.5
MAX_BATCH_CLIPS = 500
SUPPORTED_TRANSITIONS = (
    "cut", "dissolve", "fade", "wipe_left", "wipe_right",
    "wipe_up", "wipe_down", "slide_left", "slide_right",
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ClipInfo:
    """Information about a single timeline clip."""
    file_path: str = ""
    start: float = 0.0
    end: float = 0.0
    duration: float = 0.0
    track: int = 1
    index: int = 0
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "track": self.track,
            "index": self.index,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ClipInfo":
        dur = float(data.get("duration", 0))
        start = float(data.get("start", 0))
        end = float(data.get("end", start + dur))
        if dur <= 0 and end > start:
            dur = end - start
        return cls(
            file_path=data.get("file_path", ""),
            start=start,
            end=end,
            duration=dur,
            track=int(data.get("track", 1)),
            index=int(data.get("index", 0)),
            metadata=data.get("metadata", {}),
        )


@dataclass
class OpChange:
    """Description of a change applied to a clip by a batch operation."""
    clip_index: int = 0
    operation: str = ""
    parameter: str = ""
    old_value: str = ""
    new_value: str = ""
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "clip_index": self.clip_index,
            "operation": self.operation,
            "parameter": self.parameter,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "description": self.description,
        }


@dataclass
class BatchOpResult:
    """Result of a batch operation."""
    operation: str = ""
    clips: List[ClipInfo] = field(default_factory=list)
    changes: List[OpChange] = field(default_factory=list)
    clips_affected: int = 0
    dry_run: bool = False
    errors: List[Dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "operation": self.operation,
            "clips": [c.to_dict() for c in self.clips],
            "changes": [ch.to_dict() for ch in self.changes],
            "clips_affected": self.clips_affected,
            "dry_run": self.dry_run,
            "error_count": len(self.errors),
            "errors": self.errors,
        }


@dataclass
class PipelineResult:
    """Result of a chained pipeline of batch operations."""
    operations: List[BatchOpResult] = field(default_factory=list)
    final_clips: List[ClipInfo] = field(default_factory=list)
    total_changes: int = 0
    dry_run: bool = False

    def to_dict(self) -> dict:
        return {
            "operations": [op.to_dict() for op in self.operations],
            "final_clips": [c.to_dict() for c in self.final_clips],
            "total_changes": self.total_changes,
            "dry_run": self.dry_run,
            "operation_count": len(self.operations),
        }


# ---------------------------------------------------------------------------
# Clip list parsing
# ---------------------------------------------------------------------------
def parse_clip_list(clips_data: List[Dict]) -> List[ClipInfo]:
    """Parse a list of clip dicts into ClipInfo objects."""
    clips = []
    for i, data in enumerate(clips_data):
        clip = ClipInfo.from_dict(data)
        clip.index = i
        clips.append(clip)
    return clips


# ---------------------------------------------------------------------------
# Individual batch operations
# ---------------------------------------------------------------------------
def batch_color_grade(
    clips: List[ClipInfo],
    lut_path: str = "",
    grade_params: Optional[Dict] = None,
    dry_run: bool = False,
    on_progress: Optional[Callable] = None,
) -> BatchOpResult:
    """Apply LUT or color grade parameters to all clips.

    Args:
        clips: List of ClipInfo objects.
        lut_path: Path to a .cube or .3dl LUT file.
        grade_params: Dict with keys like brightness, contrast, saturation, gamma.
        dry_run: If True, return preview without applying.
        on_progress: Callback(pct).
    """
    changes = []
    errors = []
    total = len(clips)

    if grade_params is None:
        grade_params = {}

    for i, clip in enumerate(clips):
        if on_progress:
            on_progress(int((i / max(total, 1)) * 100))

        try:
            if lut_path:
                clip.metadata["lut_path"] = lut_path
                changes.append(OpChange(
                    clip_index=clip.index,
                    operation="color_grade",
                    parameter="lut",
                    old_value="",
                    new_value=os.path.basename(lut_path),
                    description=f"Apply LUT: {os.path.basename(lut_path)}",
                ))

            for param, value in grade_params.items():
                old = clip.metadata.get(f"grade_{param}", "default")
                clip.metadata[f"grade_{param}"] = value
                changes.append(OpChange(
                    clip_index=clip.index,
                    operation="color_grade",
                    parameter=param,
                    old_value=str(old),
                    new_value=str(value),
                    description=f"Set {param} to {value}",
                ))
        except Exception as exc:
            errors.append({"clip_index": clip.index, "error": str(exc)})

    return BatchOpResult(
        operation="color_grade",
        clips=clips,
        changes=changes,
        clips_affected=len([c for c in changes if c.clip_index >= 0]),
        dry_run=dry_run,
        errors=errors,
    )


def batch_speed(
    clips: List[ClipInfo],
    speed_factor: float = 1.0,
    maintain_pitch: bool = True,
    dry_run: bool = False,
    on_progress: Optional[Callable] = None,
) -> BatchOpResult:
    """Apply uniform speed change to all clips.

    Args:
        clips: List of ClipInfo objects.
        speed_factor: Speed multiplier (0.25 to 4.0).
        maintain_pitch: Whether to preserve audio pitch.
        dry_run: If True, return preview without applying.
        on_progress: Callback(pct).
    """
    speed_factor = max(0.25, min(4.0, speed_factor))
    changes = []
    errors = []
    total = len(clips)

    for i, clip in enumerate(clips):
        if on_progress:
            on_progress(int((i / max(total, 1)) * 100))

        try:
            old_duration = clip.duration
            new_duration = old_duration / speed_factor if speed_factor > 0 else old_duration

            clip.metadata["speed_factor"] = speed_factor
            clip.metadata["maintain_pitch"] = maintain_pitch
            clip.duration = new_duration
            clip.end = clip.start + new_duration

            changes.append(OpChange(
                clip_index=clip.index,
                operation="speed",
                parameter="speed_factor",
                old_value="1.0",
                new_value=str(speed_factor),
                description=f"Speed {speed_factor}x: {old_duration:.2f}s -> {new_duration:.2f}s",
            ))
        except Exception as exc:
            errors.append({"clip_index": clip.index, "error": str(exc)})

    return BatchOpResult(
        operation="speed",
        clips=clips,
        changes=changes,
        clips_affected=len(changes),
        dry_run=dry_run,
        errors=errors,
    )


def batch_normalize(
    clips: List[ClipInfo],
    target_lufs: float = DEFAULT_NORMALIZE_TARGET,
    dry_run: bool = False,
    on_progress: Optional[Callable] = None,
) -> BatchOpResult:
    """Normalize audio levels across all clips.

    Args:
        clips: List of ClipInfo objects.
        target_lufs: Target loudness in LUFS.
        dry_run: If True, return preview without applying.
        on_progress: Callback(pct).
    """
    target_lufs = max(-70.0, min(0.0, target_lufs))
    changes = []
    errors = []
    total = len(clips)

    for i, clip in enumerate(clips):
        if on_progress:
            on_progress(int((i / max(total, 1)) * 100))

        try:
            current_lufs = clip.metadata.get("loudness_lufs", -23.0)
            adjustment = target_lufs - current_lufs

            clip.metadata["normalize_target_lufs"] = target_lufs
            clip.metadata["normalize_adjustment_db"] = adjustment

            changes.append(OpChange(
                clip_index=clip.index,
                operation="normalize",
                parameter="loudness",
                old_value=f"{current_lufs:.1f} LUFS",
                new_value=f"{target_lufs:.1f} LUFS",
                description=f"Normalize: {current_lufs:.1f} -> {target_lufs:.1f} LUFS "
                            f"({adjustment:+.1f} dB)",
            ))
        except Exception as exc:
            errors.append({"clip_index": clip.index, "error": str(exc)})

    return BatchOpResult(
        operation="normalize",
        clips=clips,
        changes=changes,
        clips_affected=len(changes),
        dry_run=dry_run,
        errors=errors,
    )


def batch_transition(
    clips: List[ClipInfo],
    transition_type: str = DEFAULT_TRANSITION,
    transition_duration: float = DEFAULT_TRANSITION_DURATION,
    dry_run: bool = False,
    on_progress: Optional[Callable] = None,
) -> BatchOpResult:
    """Add transitions between all adjacent clips.

    Args:
        clips: List of ClipInfo objects (must be ordered).
        transition_type: Transition type from SUPPORTED_TRANSITIONS.
        transition_duration: Transition duration in seconds.
        dry_run: If True, return preview without applying.
        on_progress: Callback(pct).
    """
    if transition_type not in SUPPORTED_TRANSITIONS:
        raise ValueError(
            f"Unsupported transition '{transition_type}'. "
            f"Use: {', '.join(SUPPORTED_TRANSITIONS)}"
        )

    transition_duration = max(0.1, min(5.0, transition_duration))
    changes = []
    errors = []
    total = len(clips)

    for i in range(len(clips) - 1):
        if on_progress:
            on_progress(int((i / max(total - 1, 1)) * 100))

        try:
            clip_a = clips[i]
            clip_b = clips[i + 1]

            clip_a.metadata["transition_out"] = transition_type
            clip_a.metadata["transition_out_duration"] = transition_duration
            clip_b.metadata["transition_in"] = transition_type
            clip_b.metadata["transition_in_duration"] = transition_duration

            changes.append(OpChange(
                clip_index=clip_a.index,
                operation="transition",
                parameter="transition_type",
                old_value="cut",
                new_value=f"{transition_type} ({transition_duration}s)",
                description=f"Add {transition_type} ({transition_duration}s) "
                            f"between clips {clip_a.index} and {clip_b.index}",
            ))
        except Exception as exc:
            errors.append({"clip_index": i, "error": str(exc)})

    return BatchOpResult(
        operation="transition",
        clips=clips,
        changes=changes,
        clips_affected=len(changes),
        dry_run=dry_run,
        errors=errors,
    )


def batch_crop(
    clips: List[ClipInfo],
    crop_params: Optional[Dict] = None,
    aspect_ratio: str = "",
    dry_run: bool = False,
    on_progress: Optional[Callable] = None,
) -> BatchOpResult:
    """Apply uniform crop or reframe to all clips.

    Args:
        clips: List of ClipInfo objects.
        crop_params: Dict with x, y, width, height (pixels or percentages).
        aspect_ratio: Target aspect ratio like '16:9', '9:16', '1:1'.
        dry_run: If True, return preview without applying.
        on_progress: Callback(pct).
    """
    if crop_params is None:
        crop_params = {}

    changes = []
    errors = []
    total = len(clips)

    # Parse aspect ratio if provided
    ar_w, ar_h = 0, 0
    if aspect_ratio:
        parts = aspect_ratio.split(":")
        if len(parts) == 2:
            try:
                ar_w, ar_h = int(parts[0]), int(parts[1])
            except ValueError:
                pass

    for i, clip in enumerate(clips):
        if on_progress:
            on_progress(int((i / max(total, 1)) * 100))

        try:
            if ar_w > 0 and ar_h > 0:
                clip.metadata["crop_aspect_ratio"] = aspect_ratio
                changes.append(OpChange(
                    clip_index=clip.index,
                    operation="crop",
                    parameter="aspect_ratio",
                    old_value=clip.metadata.get("crop_aspect_ratio", "original"),
                    new_value=aspect_ratio,
                    description=f"Reframe to {aspect_ratio}",
                ))
            elif crop_params:
                for param, value in crop_params.items():
                    old = clip.metadata.get(f"crop_{param}", "auto")
                    clip.metadata[f"crop_{param}"] = value
                    changes.append(OpChange(
                        clip_index=clip.index,
                        operation="crop",
                        parameter=param,
                        old_value=str(old),
                        new_value=str(value),
                        description=f"Set crop {param} to {value}",
                    ))
        except Exception as exc:
            errors.append({"clip_index": clip.index, "error": str(exc)})

    return BatchOpResult(
        operation="crop",
        clips=clips,
        changes=changes,
        clips_affected=len(set(c.clip_index for c in changes)),
        dry_run=dry_run,
        errors=errors,
    )


def batch_caption(
    clips: List[ClipInfo],
    language: str = "en",
    style: str = "default",
    burn_in: bool = False,
    dry_run: bool = False,
    on_progress: Optional[Callable] = None,
) -> BatchOpResult:
    """Generate captions for each clip.

    Args:
        clips: List of ClipInfo objects.
        language: Caption language code.
        style: Caption style preset.
        burn_in: Whether to burn captions into video.
        dry_run: If True, return preview without applying.
        on_progress: Callback(pct).
    """
    changes = []
    errors = []
    total = len(clips)

    for i, clip in enumerate(clips):
        if on_progress:
            on_progress(int((i / max(total, 1)) * 100))

        try:
            clip.metadata["caption_language"] = language
            clip.metadata["caption_style"] = style
            clip.metadata["caption_burn_in"] = burn_in
            clip.metadata["caption_pending"] = True

            changes.append(OpChange(
                clip_index=clip.index,
                operation="caption",
                parameter="language",
                old_value="",
                new_value=language,
                description=f"Generate {language} captions"
                            f"{' (burn-in)' if burn_in else ' (soft)'}",
            ))
        except Exception as exc:
            errors.append({"clip_index": clip.index, "error": str(exc)})

    return BatchOpResult(
        operation="caption",
        clips=clips,
        changes=changes,
        clips_affected=len(changes),
        dry_run=dry_run,
        errors=errors,
    )


# ---------------------------------------------------------------------------
# Operation dispatcher
# ---------------------------------------------------------------------------
_OP_DISPATCH = {
    "color_grade": batch_color_grade,
    "speed": batch_speed,
    "normalize": batch_normalize,
    "transition": batch_transition,
    "crop": batch_crop,
    "caption": batch_caption,
}


def execute_operation(
    operation: str,
    clips: List[ClipInfo],
    params: Dict,
    dry_run: bool = False,
    on_progress: Optional[Callable] = None,
) -> BatchOpResult:
    """Execute a single batch operation by name.

    Args:
        operation: Operation name from BATCH_OPS.
        clips: List of ClipInfo objects.
        params: Operation-specific parameters.
        dry_run: If True, return preview without applying.
        on_progress: Callback(pct).
    """
    if operation not in _OP_DISPATCH:
        raise ValueError(
            f"Unknown operation '{operation}'. Available: {', '.join(BATCH_OPS)}"
        )

    handler = _OP_DISPATCH[operation]

    # Map params to function arguments
    kwargs = {"clips": clips, "dry_run": dry_run, "on_progress": on_progress}

    if operation == "color_grade":
        kwargs["lut_path"] = params.get("lut_path", "")
        kwargs["grade_params"] = params.get("grade_params", {})
    elif operation == "speed":
        kwargs["speed_factor"] = float(params.get("speed_factor", 1.0))
        kwargs["maintain_pitch"] = bool(params.get("maintain_pitch", True))
    elif operation == "normalize":
        kwargs["target_lufs"] = float(params.get("target_lufs", DEFAULT_NORMALIZE_TARGET))
    elif operation == "transition":
        kwargs["transition_type"] = params.get("transition_type", DEFAULT_TRANSITION)
        kwargs["transition_duration"] = float(
            params.get("transition_duration", DEFAULT_TRANSITION_DURATION)
        )
    elif operation == "crop":
        kwargs["crop_params"] = params.get("crop_params", {})
        kwargs["aspect_ratio"] = params.get("aspect_ratio", "")
    elif operation == "caption":
        kwargs["language"] = params.get("language", "en")
        kwargs["style"] = params.get("style", "default")
        kwargs["burn_in"] = bool(params.get("burn_in", False))

    return handler(**kwargs)


# ---------------------------------------------------------------------------
# Pipeline: chained operations
# ---------------------------------------------------------------------------
def execute_pipeline(
    clips_data: List[Dict],
    operations: List[Dict],
    dry_run: bool = False,
    on_progress: Optional[Callable] = None,
) -> PipelineResult:
    """Execute a chain of batch operations on clips.

    Args:
        clips_data: List of clip dicts with file_path, start, end, etc.
        operations: List of operation dicts with 'operation' and 'params' keys.
        dry_run: If True, preview changes without applying.
        on_progress: Callback(pct) for overall progress.

    Returns:
        PipelineResult with per-operation results and final clip state.
    """
    if not clips_data:
        raise ValueError("No clips provided")

    if len(clips_data) > MAX_BATCH_CLIPS:
        raise ValueError(f"Too many clips ({len(clips_data)}). Max: {MAX_BATCH_CLIPS}")

    if not operations:
        raise ValueError("No operations provided")

    # Validate operations
    for op in operations:
        op_name = op.get("operation", "")
        if op_name not in _OP_DISPATCH:
            raise ValueError(
                f"Unknown operation '{op_name}'. Available: {', '.join(BATCH_OPS)}"
            )

    clips = parse_clip_list(clips_data)
    results = []
    total_ops = len(operations)
    total_changes = 0

    for op_idx, op_spec in enumerate(operations):
        if on_progress:
            base_pct = int((op_idx / max(total_ops, 1)) * 95)
            on_progress(base_pct)

        op_name = op_spec.get("operation", "")
        params = op_spec.get("params", {})

        def _op_progress(pct):
            if on_progress:
                scale = 95 / max(total_ops, 1)
                overall = int(base_pct + (pct / 100.0) * scale)
                on_progress(overall)

        result = execute_operation(
            op_name, clips, params,
            dry_run=dry_run,
            on_progress=_op_progress,
        )
        results.append(result)
        total_changes += len(result.changes)

        # Use modified clips for next operation
        clips = result.clips

    if on_progress:
        on_progress(100)

    return PipelineResult(
        operations=results,
        final_clips=clips,
        total_changes=total_changes,
        dry_run=dry_run,
    )
