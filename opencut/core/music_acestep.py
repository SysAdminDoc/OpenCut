"""
OpenCut ACE-Step Music Generation Backend

Full-length music generation (up to 4 min) with lyric alignment,
voice cloning, stem separation, lyric editing, and repainting.
3.5B model; generates 1 min of music in 4.7s on RTX 3090.

Licence: Apache-2.0
Repository: https://github.com/ACEStudio/ACE-Step
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from opencut.helpers import _try_import

logger = logging.getLogger("opencut")

INSTALL_HINT = "pip install acestep  # Apache-2.0, full-song music w/ lyrics"

ACESTEP_GENRES = [
    "pop", "rock", "electronic", "hip-hop", "jazz", "classical",
    "folk", "country", "r&b", "metal", "ambient", "cinematic",
]

ACESTEP_MOODS = [
    "happy", "sad", "energetic", "calm", "dramatic", "nostalgic",
    "romantic", "aggressive", "dreamy", "triumphant",
]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ACEStepResult:
    output: str = ""
    duration_seconds: float = 0.0
    genre: str = ""
    mood: str = ""
    has_lyrics: bool = False
    model: str = "ace-step"
    generation_seconds: float = 0.0
    sample_rate: int = 44100
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "duration_seconds", "genre", "mood",
                "has_lyrics", "model", "generation_seconds",
                "sample_rate", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def check_acestep_available() -> bool:
    """Return True when the acestep package is importable."""
    return _try_import("acestep") is not None


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(
    prompt: str = "",
    lyrics: str = "",
    genre: str = "pop",
    mood: str = "happy",
    duration: float = 60.0,
    reference_audio: str = "",
    output_path: str = "",
    on_progress: Optional[Callable] = None,
) -> ACEStepResult:
    """Generate music using ACE-Step.

    Args:
        prompt: Text description of desired music (e.g., "upbeat indie pop").
        lyrics: Optional lyrics to align with generated music.
        genre: Music genre hint.
        mood: Mood/energy hint.
        duration: Target duration in seconds (max 240).
        reference_audio: Path to reference audio for style cloning.
        output_path: Where to write output. Auto-generated if empty.
        on_progress: Optional ``(pct, msg="")`` callback.

    Returns:
        ACEStepResult with output path and metadata.

    Raises:
        RuntimeError: When acestep is not installed or generation fails.
    """
    acestep_mod = _try_import("acestep")
    if acestep_mod is None:
        raise RuntimeError(f"acestep is not installed. {INSTALL_HINT}")

    duration = max(5.0, min(240.0, float(duration)))
    genre = genre.lower().strip() if genre else "pop"
    mood = mood.lower().strip() if mood else "happy"

    if on_progress:
        on_progress(10, "Loading ACE-Step model...")

    notes: List[str] = []

    if not output_path:
        fd, output_path = tempfile.mkstemp(
            suffix=".wav", prefix="opencut_acestep_",
        )
        os.close(fd)

    try:
        from acestep import ACEStep

        model = ACEStep()

        if on_progress:
            on_progress(30, f"Generating {duration:.0f}s {genre} track...")

        kwargs: Dict[str, Any] = {
            "duration": duration,
            "output_path": output_path,
        }

        # Build the prompt from components if no explicit prompt given
        if not prompt:
            parts = []
            if mood:
                parts.append(mood)
            if genre:
                parts.append(genre)
            parts.append("music")
            prompt = " ".join(parts)
        kwargs["prompt"] = prompt
        notes.append(f"Prompt: {prompt[:100]}")

        if lyrics and lyrics.strip():
            kwargs["lyrics"] = lyrics.strip()
            notes.append(f"Lyrics: {len(lyrics.split())} words")

        if reference_audio and os.path.isfile(reference_audio):
            kwargs["reference_audio"] = reference_audio
            notes.append(f"Style ref: {os.path.basename(reference_audio)}")

        start_time = time.monotonic()
        model.generate(**kwargs)
        gen_time = time.monotonic() - start_time

        if on_progress:
            on_progress(90, "Finalizing...")

        notes.append(f"Generated in {gen_time:.1f}s")

        # Check actual output duration
        actual_duration = 0.0
        sample_rate = 44100
        try:
            import wave
            with wave.open(output_path, "rb") as wf:
                sample_rate = wf.getframerate()
                actual_duration = wf.getnframes() / sample_rate if sample_rate > 0 else 0.0
        except Exception:
            actual_duration = duration  # Fallback to requested duration

        if on_progress:
            on_progress(100, "Done")

        return ACEStepResult(
            output=output_path,
            duration_seconds=round(actual_duration, 2),
            genre=genre,
            mood=mood,
            has_lyrics=bool(lyrics and lyrics.strip()),
            model="ace-step",
            generation_seconds=round(gen_time, 2),
            sample_rate=sample_rate,
            notes=notes,
        )

    except ImportError as exc:
        raise RuntimeError(
            f"acestep import failed: {exc}. {INSTALL_HINT}"
        ) from exc
    except Exception as exc:
        if output_path and os.path.isfile(output_path):
            try:
                os.unlink(output_path)
            except OSError:
                pass
        raise RuntimeError(f"ACE-Step generation failed: {exc}") from exc


def edit_lyrics(
    audio_path: str,
    new_lyrics: str,
    output_path: str = "",
    on_progress: Optional[Callable] = None,
) -> ACEStepResult:
    """Edit lyrics in an existing ACE-Step generated track.

    Uses ACE-Step's flow-edit capability to re-synthesize vocals with
    new lyrics while preserving the instrumental arrangement.

    Args:
        audio_path: Path to original ACE-Step output.
        new_lyrics: Replacement lyrics text.
        output_path: Where to write output. Auto-generated if empty.
        on_progress: Optional ``(pct, msg="")`` callback.

    Returns:
        ACEStepResult with the edited output.
    """
    acestep_mod = _try_import("acestep")
    if acestep_mod is None:
        raise RuntimeError(f"acestep is not installed. {INSTALL_HINT}")

    if not audio_path or not os.path.isfile(audio_path):
        raise ValueError(f"Audio file not found: {audio_path}")
    if not new_lyrics or not new_lyrics.strip():
        raise ValueError("new_lyrics must not be empty")

    if on_progress:
        on_progress(10, "Loading ACE-Step model for lyric edit...")

    if not output_path:
        fd, output_path = tempfile.mkstemp(
            suffix=".wav", prefix="opencut_acestep_edit_",
        )
        os.close(fd)

    notes: List[str] = []

    try:
        from acestep import ACEStep

        model = ACEStep()

        if on_progress:
            on_progress(40, "Editing lyrics...")

        start_time = time.monotonic()
        model.edit(
            audio_path=audio_path,
            lyrics=new_lyrics.strip(),
            output_path=output_path,
        )
        gen_time = time.monotonic() - start_time
        notes.append(f"Lyric edit in {gen_time:.1f}s")
        notes.append(f"New lyrics: {len(new_lyrics.split())} words")

        if on_progress:
            on_progress(100, "Done")

        return ACEStepResult(
            output=output_path,
            has_lyrics=True,
            model="ace-step",
            generation_seconds=round(gen_time, 2),
            notes=notes,
        )

    except Exception as exc:
        if output_path and os.path.isfile(output_path):
            try:
                os.unlink(output_path)
            except OSError:
                pass
        raise RuntimeError(f"ACE-Step lyric edit failed: {exc}") from exc
