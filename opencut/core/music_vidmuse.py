"""
OpenCut VidMuse Video-to-Music (L2.4)

Generate background music semantically and rhythmically synchronised to
video content (scene cuts, motion energy, visual tempo).  Unlike MusicGen
(text-only) and ACE-Step (lyric-focused), VidMuse reads the video directly
to compose a matching score.

Licence: MIT
Repository: https://vidmuse.github.io/
Paper: CVPR 2025
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from opencut.helpers import _try_import

logger = logging.getLogger("opencut")

INSTALL_HINT = "pip install vidmuse torch  # MIT, video-conditioned music composition"

VIDMUSE_STYLES = [
    "auto",          # let the model decide based on video content
    "cinematic",     # orchestral / film-score feel
    "upbeat",        # energetic pop / electronic
    "ambient",       # atmospheric / background
    "dramatic",      # tension / suspense
    "cheerful",      # light / happy
]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class VidMuseResult:
    output: str = ""
    bpm: float = 0.0
    duration: float = 0.0
    mood: str = ""
    model: str = "vidmuse"
    generation_seconds: float = 0.0
    sample_rate: int = 44100
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "bpm", "duration", "mood", "model",
                "generation_seconds", "sample_rate", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def check_vidmuse_available() -> bool:
    """Return True when vidmuse + torch are importable."""
    return _try_import("vidmuse") is not None and _try_import("torch") is not None


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(
    video_path: str,
    duration: float = 30.0,
    style_hint: str = "auto",
    output: Optional[str] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> VidMuseResult:
    """Generate music from video content using VidMuse.

    Args:
        video_path: Path to the source video file.
        duration: Target music duration in seconds (max 180).
        style_hint: Style hint — auto, cinematic, upbeat, ambient, dramatic, cheerful.
        output: Output WAV path. Auto-generated if None.
        on_progress: Optional ``(pct, msg="")`` callback.

    Returns:
        VidMuseResult with output path and metadata.

    Raises:
        RuntimeError: When vidmuse/torch is not installed or generation fails.
    """
    if not video_path or not os.path.isfile(video_path):
        raise ValueError(f"Video file not found: {video_path}")

    vidmuse_mod = _try_import("vidmuse")
    if vidmuse_mod is None or _try_import("torch") is None:
        raise RuntimeError(f"vidmuse or torch is not installed. {INSTALL_HINT}")

    duration = max(5.0, min(180.0, float(duration)))
    if style_hint not in VIDMUSE_STYLES:
        style_hint = "auto"

    if on_progress:
        on_progress(5, "Loading VidMuse model...")

    notes: List[str] = []

    if not output:
        fd, output = tempfile.mkstemp(
            suffix=".wav", prefix="opencut_vidmuse_",
        )
        os.close(fd)

    try:
        from vidmuse import VidMuse

        model = VidMuse()

        if on_progress:
            on_progress(20, f"Analyzing video and composing {duration:.0f}s score...")

        start_time = time.monotonic()

        kwargs = {
            "video_path": video_path,
            "output_path": output,
            "duration": duration,
        }
        if style_hint and style_hint != "auto":
            kwargs["style"] = style_hint
            notes.append(f"Style: {style_hint}")

        model.generate(**kwargs)
        gen_time = time.monotonic() - start_time

        if on_progress:
            on_progress(90, "Finalizing...")

        notes.append(f"Generated in {gen_time:.1f}s")

        # Read output metadata
        actual_duration = duration
        sample_rate = 44100
        try:
            import wave as _wave
            with _wave.open(output, "rb") as wf:
                sample_rate = wf.getframerate()
                actual_duration = wf.getnframes() / sample_rate if sample_rate > 0 else duration
        except Exception:
            pass

        if on_progress:
            on_progress(100, "Done")

        return VidMuseResult(
            output=output,
            bpm=0.0,  # VidMuse doesn't always report BPM
            duration=round(actual_duration, 2),
            mood=style_hint,
            model="vidmuse",
            generation_seconds=round(gen_time, 2),
            sample_rate=sample_rate,
            notes=notes,
        )

    except ImportError as exc:
        raise RuntimeError(
            f"vidmuse import failed: {exc}. {INSTALL_HINT}"
        ) from exc
    except Exception as exc:
        if output and os.path.isfile(output):
            try:
                os.unlink(output)
            except OSError:
                pass
        raise RuntimeError(f"VidMuse generation failed: {exc}") from exc


__all__ = [
    "VidMuseResult",
    "check_vidmuse_available",
    "INSTALL_HINT",
    "VIDMUSE_STYLES",
    "generate",
]
