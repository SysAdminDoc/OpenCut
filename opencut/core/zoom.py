"""
Auto-zoom keyframe generation.

Analyzes audio energy to determine emphasis points and generates
zoom/scale keyframe data for timeline export.
"""

from dataclasses import dataclass
from typing import List, Optional

from ..utils.config import ZoomConfig
from .audio import find_emphasis_points
from .silence import TimeSegment


@dataclass
class ZoomKeyframe:
    """A zoom keyframe at a specific time."""
    time: float
    scale: float    # 1.0 = 100%, 1.3 = 130%
    anchor_x: float = 0.5  # 0.0 = left, 0.5 = center, 1.0 = right
    anchor_y: float = 0.5  # 0.0 = top,  0.5 = center, 1.0 = bottom


@dataclass
class ZoomEvent:
    """A complete zoom in/out event."""
    start: float
    peak: float       # Time of max zoom
    end: float
    max_scale: float
    anchor_x: float = 0.5
    anchor_y: float = 0.5

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_keyframes(self) -> List[ZoomKeyframe]:
        """Convert to a list of keyframes for export."""
        return [
            ZoomKeyframe(time=self.start, scale=1.0, anchor_x=self.anchor_x, anchor_y=self.anchor_y),
            ZoomKeyframe(time=self.peak, scale=self.max_scale, anchor_x=self.anchor_x, anchor_y=self.anchor_y),
            ZoomKeyframe(time=self.end, scale=1.0, anchor_x=self.anchor_x, anchor_y=self.anchor_y),
        ]


def generate_zoom_events(
    filepath: str,
    config: Optional[ZoomConfig] = None,
    speech_segments: Optional[List[TimeSegment]] = None,
) -> List[ZoomEvent]:
    """
    Generate zoom events based on audio energy analysis.

    Args:
        filepath: Path to the media file.
        config: Zoom configuration. Uses defaults if None.
        speech_segments: If provided, only generate zooms during speech.

    Returns:
        List of ZoomEvent objects.
    """
    if config is None:
        config = ZoomConfig()

    # Find emphasis points in the audio
    emphasis = find_emphasis_points(
        filepath,
        threshold=config.energy_threshold,
        min_interval=config.min_interval,
    )

    # Filter to only speech segments if provided
    if speech_segments:
        emphasis = _filter_to_speech(emphasis, speech_segments)

    # Convert emphasis points to zoom events
    events = []
    for point in emphasis:
        event = ZoomEvent(
            start=max(0.0, point.start - config.zoom_in_duration),
            peak=point.start,
            end=point.start + config.zoom_out_duration,
            max_scale=config.max_zoom,
        )
        events.append(event)

    return events


def _filter_to_speech(
    emphasis: List[TimeSegment],
    speech: List[TimeSegment],
) -> List[TimeSegment]:
    """Keep only emphasis points that fall within speech segments."""
    filtered = []
    for em in emphasis:
        for sp in speech:
            if sp.start <= em.start <= sp.end:
                filtered.append(em)
                break
    return filtered


def zoom_events_to_keyframes(events: List[ZoomEvent]) -> List[ZoomKeyframe]:
    """Flatten all zoom events into a single keyframe list."""
    keyframes = []
    for event in events:
        keyframes.extend(event.to_keyframes())
    return sorted(keyframes, key=lambda k: k.time)
