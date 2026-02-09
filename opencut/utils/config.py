"""
Configuration management for OpenCut.

Handles default settings, user preferences, and preset management.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SilenceConfig:
    """Configuration for silence detection."""
    # Noise threshold in dB (negative). Audio below this is considered silence.
    threshold_db: float = -30.0
    # Minimum silence duration in seconds to trigger a cut.
    min_duration: float = 0.5
    # Padding in seconds to add before/after speech segments.
    padding_before: float = 0.1
    padding_after: float = 0.1
    # How to handle detected silences: "remove", "mute", or "mark"
    silence_action: str = "remove"
    # Minimum speech segment duration (ignore tiny speech blips)
    min_speech_duration: float = 0.25


@dataclass
class CaptionConfig:
    """Configuration for caption/subtitle generation."""
    # Whisper model size: tiny, base, small, medium, large-v3, turbo
    model: str = "base"
    # Language code (None = auto-detect)
    language: Optional[str] = None
    # Enable word-level timestamps for animated captions
    word_timestamps: bool = True
    # Max characters per caption line
    max_line_length: int = 42
    # Max lines per caption
    max_lines: int = 2
    # Output format: "srt", "vtt", "json"
    output_format: str = "srt"
    # Translate to English
    translate: bool = False


@dataclass
class DiarizeConfig:
    """Configuration for speaker diarization (podcast editing)."""
    # Expected number of speakers (None = auto-detect)
    num_speakers: Optional[int] = None
    # Min/max speaker bounds
    min_speakers: int = 1
    max_speakers: int = 10
    # Minimum segment duration before switching cameras (seconds)
    min_segment_duration: float = 1.0
    # HuggingFace auth token for pyannote models
    hf_token: Optional[str] = None


@dataclass
class ZoomConfig:
    """Configuration for auto-zoom effects."""
    # Zoom scale factor (1.0 = no zoom, 1.3 = 30% zoom)
    max_zoom: float = 1.3
    # Duration of zoom-in transition (seconds)
    zoom_in_duration: float = 0.3
    # Duration of zoom-out transition (seconds)
    zoom_out_duration: float = 0.5
    # Minimum time between zooms (seconds)
    min_interval: float = 3.0
    # Audio energy threshold to trigger zoom (0.0-1.0)
    energy_threshold: float = 0.7


@dataclass
class ExportConfig:
    """Configuration for export settings."""
    # Export format: "premiere", "resolve", "fcpxml"
    format: str = "premiere"
    # Sequence/timeline name
    sequence_name: str = "OpenCut Edit"
    # Whether to include video track
    include_video: bool = True
    # Whether to include audio track
    include_audio: bool = True


@dataclass
class OpenCutConfig:
    """Master configuration combining all settings."""
    silence: SilenceConfig = field(default_factory=SilenceConfig)
    captions: CaptionConfig = field(default_factory=CaptionConfig)
    diarize: DiarizeConfig = field(default_factory=DiarizeConfig)
    zoom: ZoomConfig = field(default_factory=ZoomConfig)
    export: ExportConfig = field(default_factory=ExportConfig)


# ----- Presets -----

PRESETS = {
    "default": OpenCutConfig(),

    "aggressive": OpenCutConfig(
        silence=SilenceConfig(
            threshold_db=-25.0,
            min_duration=0.3,
            padding_before=0.05,
            padding_after=0.05,
            min_speech_duration=0.15,
        ),
    ),

    "conservative": OpenCutConfig(
        silence=SilenceConfig(
            threshold_db=-40.0,
            min_duration=1.0,
            padding_before=0.2,
            padding_after=0.2,
            min_speech_duration=0.5,
        ),
    ),

    "podcast": OpenCutConfig(
        silence=SilenceConfig(
            threshold_db=-35.0,
            min_duration=0.75,
            padding_before=0.15,
            padding_after=0.15,
        ),
        diarize=DiarizeConfig(
            num_speakers=2,
            min_segment_duration=1.5,
        ),
    ),

    "youtube": OpenCutConfig(
        silence=SilenceConfig(
            threshold_db=-28.0,
            min_duration=0.4,
            padding_before=0.08,
            padding_after=0.08,
        ),
        captions=CaptionConfig(
            model="small",
            word_timestamps=True,
        ),
        zoom=ZoomConfig(
            max_zoom=1.2,
            min_interval=5.0,
        ),
    ),
}


def get_preset(name: str) -> OpenCutConfig:
    """Get a preset configuration by name."""
    if name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    return PRESETS[name]
