"""
OpenCut DiffRhythm Full-Song Generation (M1.3)

First diffusion-based full-length song generator.  Base model: 1m35s,
full model: up to 4m45s.  Accepts LRC lyrics + optional audio style
reference OR text style prompt.

Licence: Apache-2.0
Repository: https://github.com/ASLP-lab/DiffRhythm
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

INSTALL_HINT = (
    "git clone https://github.com/ASLP-lab/DiffRhythm && "
    "pip install -r DiffRhythm/requirements.txt  # Apache-2.0\n"
    "Also requires espeak-ng for lyrics phonemisation:\n"
    "  Windows: set PHONEMIZER_ESPEAK_LIBRARY and PHONEMIZER_ESPEAK_PATH\n"
    "  macOS:   brew install espeak-ng\n"
    "  Linux:   apt install espeak-ng"
)

DIFFRHYTHM_STYLES = [
    "Jazzy Nightclub Vibe",
    "Pop Emotional Piano",
    "Indie Folk Acoustic",
    "Electronic Dance Energy",
    "Hip-Hop Beat Driven",
    "Cinematic Orchestral",
    "Rock Power Ballad",
    "Lo-Fi Chill Study",
    "R&B Smooth Groove",
    "Metal Aggressive",
]

DIFFRHYTHM_MODELS = {
    "base": "Base model — up to 1m35s, faster generation",
    "full": "Full model — up to 4m45s, higher quality",
}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class DiffRhythmResult:
    output: str = ""
    duration_seconds: float = 0.0
    style: str = ""
    has_lyrics: bool = False
    model_variant: str = "full"
    generation_seconds: float = 0.0
    sample_rate: int = 44100
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "duration_seconds", "style", "has_lyrics",
                "model_variant", "generation_seconds", "sample_rate", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def check_diffrhythm_available() -> bool:
    """Return True when DiffRhythm inference is available."""
    # DiffRhythm is typically a git clone, not a pip package.
    # Check for the inference module or a known marker.
    if _try_import("diffrhythm") is not None:
        return True
    # Fallback: check for env-configured path
    dr_path = os.environ.get("OPENCUT_DIFFRHYTHM_PATH", "")
    if dr_path and os.path.isdir(dr_path):
        return True
    return False


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(
    style_prompt: str = "",
    lyrics_lrc: str = "",
    style_reference: str = "",
    model_variant: str = "full",
    chunked: bool = True,
    output_path: str = "",
    on_progress: Optional[Callable] = None,
) -> DiffRhythmResult:
    """Generate a full song using DiffRhythm.

    Args:
        style_prompt: Text description of desired style (e.g., "Jazzy Nightclub Vibe").
        lyrics_lrc: LRC-format lyrics with timestamps. Optional.
        style_reference: Path to audio style reference file. Optional.
        model_variant: "base" (up to 95s) or "full" (up to 285s).
        chunked: Use chunked mode to reduce VRAM (8 GB minimum).
        output_path: Where to write WAV. Auto-generated if empty.
        on_progress: Optional ``(pct, msg="")`` callback.

    Returns:
        DiffRhythmResult with output path and metadata.

    Raises:
        RuntimeError: When DiffRhythm is not installed or generation fails.
    """
    if not check_diffrhythm_available():
        raise RuntimeError(f"DiffRhythm is not installed. {INSTALL_HINT}")

    if model_variant not in DIFFRHYTHM_MODELS:
        model_variant = "full"

    if not style_prompt and not style_reference:
        style_prompt = "Pop Emotional Piano"

    if on_progress:
        on_progress(5, f"Loading DiffRhythm ({model_variant})...")

    notes: List[str] = []

    if not output_path:
        fd, output_path = tempfile.mkstemp(
            suffix=".wav", prefix="opencut_diffrhythm_",
        )
        os.close(fd)

    try:
        from diffrhythm import DiffRhythm

        model = DiffRhythm(variant=model_variant)

        if on_progress:
            on_progress(20, "Generating song...")

        start_time = time.monotonic()

        kwargs: Dict[str, Any] = {"output_path": output_path}
        if style_prompt:
            kwargs["style_prompt"] = style_prompt
            notes.append(f"Style: {style_prompt[:80]}")
        if lyrics_lrc and lyrics_lrc.strip():
            kwargs["lyrics"] = lyrics_lrc.strip()
            notes.append("Lyrics provided (LRC)")
        if style_reference and os.path.isfile(style_reference):
            kwargs["style_reference"] = style_reference
            notes.append(f"Style ref: {os.path.basename(style_reference)}")
        if chunked:
            kwargs["chunked"] = True

        model.generate(**kwargs)
        gen_time = time.monotonic() - start_time

        if on_progress:
            on_progress(90, "Finalizing...")

        notes.append(f"Model: {model_variant}")
        notes.append(f"Generated in {gen_time:.1f}s")

        # Measure output
        actual_duration = 0.0
        sample_rate = 44100
        try:
            import wave
            with wave.open(output_path, "rb") as wf:
                sample_rate = wf.getframerate()
                actual_duration = wf.getnframes() / sample_rate if sample_rate > 0 else 0.0
        except Exception:
            pass

        if on_progress:
            on_progress(100, "Done")

        return DiffRhythmResult(
            output=output_path,
            duration_seconds=round(actual_duration, 2),
            style=style_prompt,
            has_lyrics=bool(lyrics_lrc and lyrics_lrc.strip()),
            model_variant=model_variant,
            generation_seconds=round(gen_time, 2),
            sample_rate=sample_rate,
            notes=notes,
        )

    except ImportError as exc:
        raise RuntimeError(
            f"DiffRhythm import failed: {exc}. {INSTALL_HINT}"
        ) from exc
    except Exception as exc:
        if output_path and os.path.isfile(output_path):
            try:
                os.unlink(output_path)
            except OSError:
                pass
        raise RuntimeError(f"DiffRhythm generation failed: {exc}") from exc


__all__ = [
    "DiffRhythmResult",
    "check_diffrhythm_available",
    "INSTALL_HINT",
    "DIFFRHYTHM_STYLES",
    "DIFFRHYTHM_MODELS",
    "generate",
]
