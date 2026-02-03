"""
Speaker diarization for podcast/multicam editing.

Uses pyannote.audio to detect who is speaking when, enabling automatic
camera switching for multi-speaker content.
"""

import os
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..utils.config import DiarizeConfig
from .audio import extract_audio_wav
from .silence import TimeSegment


@dataclass
class SpeakerSegment:
    """A segment of speech attributed to a specific speaker."""
    speaker: str       # e.g., "SPEAKER_00", "SPEAKER_01"
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class DiarizationResult:
    """Complete diarization output."""
    segments: List[SpeakerSegment]
    num_speakers: int = 0
    speakers: List[str] = field(default_factory=list)

    @property
    def total_duration(self) -> float:
        return sum(s.duration for s in self.segments) if self.segments else 0.0

    def get_speaker_segments(self, speaker: str) -> List[SpeakerSegment]:
        """Get all segments for a specific speaker."""
        return [s for s in self.segments if s.speaker == speaker]

    def get_speaker_durations(self) -> Dict[str, float]:
        """Get total speaking duration per speaker."""
        durations: Dict[str, float] = {}
        for seg in self.segments:
            durations[seg.speaker] = durations.get(seg.speaker, 0.0) + seg.duration
        return durations

    def to_camera_switches(
        self,
        speaker_camera_map: Optional[Dict[str, int]] = None,
        min_segment_duration: float = 1.0,
    ) -> List[TimeSegment]:
        """
        Convert diarization to camera switch events for multicam editing.

        Args:
            speaker_camera_map: Mapping of speaker labels to camera/track indices.
                                Auto-assigned if None.
            min_segment_duration: Minimum time before switching cameras.

        Returns:
            List of TimeSegment with label indicating camera index.
        """
        if not self.segments:
            return []

        # Auto-assign cameras if no mapping provided
        if speaker_camera_map is None:
            speaker_camera_map = {
                speaker: i for i, speaker in enumerate(self.speakers)
            }

        # Merge very short segments (keep camera on longer)
        merged = self._merge_short_segments(min_segment_duration)

        switches = []
        for seg in merged:
            camera = speaker_camera_map.get(seg.speaker, 0)
            switches.append(TimeSegment(
                start=seg.start,
                end=seg.end,
                label=f"camera_{camera}",
            ))

        return switches

    def _merge_short_segments(self, min_duration: float) -> List[SpeakerSegment]:
        """Merge segments shorter than min_duration into adjacent segments."""
        if not self.segments:
            return []

        merged = [SpeakerSegment(
            speaker=self.segments[0].speaker,
            start=self.segments[0].start,
            end=self.segments[0].end,
        )]

        for seg in self.segments[1:]:
            if seg.speaker == merged[-1].speaker:
                # Same speaker — extend
                merged[-1].end = seg.end
            elif seg.duration < min_duration:
                # Too short — absorb into previous segment
                merged[-1].end = seg.end
            else:
                merged.append(SpeakerSegment(
                    speaker=seg.speaker,
                    start=seg.start,
                    end=seg.end,
                ))

        return merged


def check_pyannote_available() -> bool:
    """Check if pyannote.audio is installed."""
    try:
        from pyannote.audio import Pipeline  # noqa: F401
        return True
    except ImportError:
        return False


def diarize(
    filepath: str,
    config: Optional[DiarizeConfig] = None,
) -> DiarizationResult:
    """
    Perform speaker diarization on an audio/video file.

    Args:
        filepath: Path to the media file.
        config: Diarization configuration. Uses defaults if None.

    Returns:
        DiarizationResult with speaker segments.

    Raises:
        RuntimeError: If pyannote.audio is not installed.
    """
    if config is None:
        config = DiarizeConfig()

    if not check_pyannote_available():
        raise RuntimeError(
            "pyannote.audio not installed. Install with:\n"
            "  pip install pyannote.audio\n\n"
            "You also need a HuggingFace token:\n"
            "  1. Create account at https://huggingface.co\n"
            "  2. Accept model terms at https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "  3. Create token at https://huggingface.co/settings/tokens\n"
            "  4. Pass token via --hf-token or HUGGINGFACE_TOKEN env var\n"
        )

    from pyannote.audio import Pipeline
    import torch

    # Get HF token
    hf_token = config.hf_token or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HuggingFace token required for pyannote models.\n"
            "Set HUGGINGFACE_TOKEN env var or pass --hf-token."
        )

    # Load pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)

    # Extract audio (pyannote expects WAV)
    wav_path = extract_audio_wav(filepath, sample_rate=16000)

    try:
        # Run diarization
        kwargs = {}
        if config.num_speakers is not None:
            kwargs["num_speakers"] = config.num_speakers
        else:
            if config.min_speakers > 1:
                kwargs["min_speakers"] = config.min_speakers
            if config.max_speakers < 10:
                kwargs["max_speakers"] = config.max_speakers

        diarization = pipeline(wav_path, **kwargs)

        # Parse results
        segments = []
        speakers_set = set()

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(SpeakerSegment(
                speaker=speaker,
                start=turn.start,
                end=turn.end,
            ))
            speakers_set.add(speaker)

        speakers = sorted(speakers_set)

        return DiarizationResult(
            segments=segments,
            num_speakers=len(speakers),
            speakers=speakers,
        )

    finally:
        if os.path.exists(wav_path) and wav_path.startswith(tempfile.gettempdir()):
            os.unlink(wav_path)
