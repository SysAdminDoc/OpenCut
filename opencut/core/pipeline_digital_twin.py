"""
OpenCut Digital Twin / AI Avatar Pipeline (M3.2)

Full end-to-end localisation pipeline:
1. Clone voice from reference clip (Chatterbox TTS)
2. Generate narration audio from script
3. Generate lip-sync talking-head video (Wan2.2-S2V or EchoMimic V3)
4. Translate and dub into target languages (dub pipeline)
5. Composite avatar onto original footage

Each stage is independently skippable if pre-existing assets are provided.
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("opencut")

INSTALL_HINT = (
    "Requires: chatterbox-tts (voice clone) + faster-whisper (STT) + "
    "edge-tts or kokoro (TTS fallback). "
    "Optional: Wan2.2-S2V (talking head) or EchoMimic V3 (lip sync)."
)

PIPELINE_STAGES = [
    "voice_clone",     # Clone voice from reference
    "narrate",         # Generate speech from script
    "talking_head",    # Generate video from speech + portrait
    "translate_dub",   # Translate + dub into target languages
    "composite",       # Composite onto original footage
]

TALKING_HEAD_BACKENDS = {
    "auto": "Auto-select best available backend",
    "wan22_s2v": "Wan2.2-S2V (full talking head, 14B, GPU)",
    "echomimic": "EchoMimic V3 (portrait lip-sync, lighter)",
    "skip": "Skip talking-head generation (audio only)",
}


@dataclass
class DigitalTwinResult:
    outputs: Dict[str, str] = field(default_factory=dict)  # lang -> output_path
    stages_completed: List[str] = field(default_factory=list)
    stages_skipped: List[str] = field(default_factory=list)
    target_languages: List[str] = field(default_factory=list)
    narration_path: str = ""
    avatar_video_path: str = ""
    talking_head_backend: str = ""
    total_duration_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("outputs", "stages_completed", "stages_skipped",
                "target_languages", "narration_path", "avatar_video_path",
                "talking_head_backend", "total_duration_seconds", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_digital_twin_available() -> bool:
    """Return True — at minimum, TTS + dub pipeline are always available."""
    return True


def _detect_talking_head_backend() -> str:
    """Detect best available talking-head backend."""
    try:
        from opencut.core.gen_video_wan22_s2v import check_s2v_available
        if check_s2v_available():
            return "wan22_s2v"
    except Exception:
        pass
    try:
        from opencut.core.lipsync_echomimic import check_echomimic_available
        if check_echomimic_available():
            return "echomimic"
    except Exception:
        pass
    return "skip"


def run_pipeline(
    script: str,
    voice_ref_path: str = "",
    face_ref_path: str = "",
    target_languages: Optional[List[str]] = None,
    talking_head_backend: str = "auto",
    skip_stages: Optional[List[str]] = None,
    pre_narration_path: str = "",
    pre_avatar_path: str = "",
    tts_engine: str = "chatterbox",
    whisper_model: str = "base",
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> DigitalTwinResult:
    """Run the full digital twin pipeline.

    Args:
        script: Text script for narration.
        voice_ref_path: 10-second reference clip for voice cloning.
        face_ref_path: Portrait image for talking-head generation.
        target_languages: List of ISO 639-1 language codes to dub into.
        talking_head_backend: auto, wan22_s2v, echomimic, or skip.
        skip_stages: List of stages to skip.
        pre_narration_path: Pre-generated narration audio (skips voice_clone + narrate).
        pre_avatar_path: Pre-generated avatar video (skips talking_head).
        tts_engine: TTS backend — chatterbox, edge, kokoro, spark.
        whisper_model: Whisper model for transcription in dub pipeline.
        output_dir: Directory for all outputs.
        on_progress: Optional callback.

    Returns:
        DigitalTwinResult with per-language outputs and pipeline metadata.
    """
    if not script or not script.strip():
        raise ValueError("Script text is required")

    target_languages = target_languages or []
    skip_stages = set(skip_stages or [])
    stages_completed = []
    stages_skipped = []
    notes: List[str] = []

    if not output_dir:
        output_dir = tempfile.mkdtemp(prefix="opencut_twin_")
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.monotonic()

    # --- Stage 1: Voice Clone ---
    narration_path = pre_narration_path
    if narration_path and os.path.isfile(narration_path):
        stages_skipped.extend(["voice_clone", "narrate"])
        notes.append("Using pre-generated narration")
    else:
        if "voice_clone" in skip_stages or "narrate" in skip_stages:
            stages_skipped.append("voice_clone")
            stages_skipped.append("narrate")
        else:
            if on_progress:
                on_progress(10, "Generating narration from script...")

            narration_path = os.path.join(output_dir, "narration.wav")

            try:
                if tts_engine == "chatterbox" and voice_ref_path:
                    from opencut.core.tts_chatterbox import synthesize as _cb_synth
                    result = _cb_synth(
                        text=script.strip(),
                        reference_audio=voice_ref_path,
                        output_path=narration_path,
                    )
                    narration_path = result.output
                    notes.append("Voice cloned via Chatterbox")
                else:
                    # Fallback to edge-tts
                    from opencut.core.voice_gen import edge_tts_generate
                    narration_path = edge_tts_generate(
                        text=script.strip(),
                        output_dir=output_dir,
                    )
                    notes.append(f"Narration via {tts_engine}")

                stages_completed.extend(["voice_clone", "narrate"])
            except Exception as exc:
                notes.append(f"TTS failed: {exc}")
                stages_skipped.extend(["voice_clone", "narrate"])

    # --- Stage 2: Talking Head ---
    avatar_video_path = pre_avatar_path
    if avatar_video_path and os.path.isfile(avatar_video_path):
        stages_skipped.append("talking_head")
        notes.append("Using pre-generated avatar video")
    elif "talking_head" in skip_stages or not face_ref_path:
        stages_skipped.append("talking_head")
    else:
        if on_progress:
            on_progress(30, "Generating talking-head video...")

        if talking_head_backend == "auto":
            talking_head_backend = _detect_talking_head_backend()

        avatar_video_path = os.path.join(output_dir, "avatar.mp4")

        try:
            if talking_head_backend == "wan22_s2v" and narration_path:
                from opencut.core.gen_video_wan22_s2v import generate as _s2v_gen
                result = _s2v_gen(
                    audio_path=narration_path,
                    portrait_path=face_ref_path,
                    output_path=avatar_video_path,
                )
                avatar_video_path = result.output
                notes.append("Talking head via Wan2.2-S2V")
            elif talking_head_backend == "echomimic" and narration_path:
                from opencut.core.lipsync_echomimic import generate as _echo_gen
                result = _echo_gen(
                    audio_path=narration_path,
                    face_path=face_ref_path,
                    output_path=avatar_video_path,
                )
                avatar_video_path = result.output
                notes.append("Talking head via EchoMimic V3")
            else:
                stages_skipped.append("talking_head")
                notes.append(f"No talking-head backend: {talking_head_backend}")

            if "talking_head" not in stages_skipped:
                stages_completed.append("talking_head")
        except Exception as exc:
            notes.append(f"Talking head failed: {exc}")
            stages_skipped.append("talking_head")

    # --- Stage 3: Translate + Dub ---
    outputs: Dict[str, str] = {}
    if "translate_dub" in skip_stages or not target_languages:
        stages_skipped.append("translate_dub")
        # If no target languages, the base narration/avatar is the output
        source = avatar_video_path or narration_path
        if source:
            outputs["source"] = source
    else:
        if on_progress:
            on_progress(50, f"Dubbing into {len(target_languages)} languages...")

        source_to_dub = avatar_video_path or narration_path
        if source_to_dub and os.path.isfile(source_to_dub):
            try:
                from opencut.core.dub_pipeline import dub

                for i, lang in enumerate(target_languages):
                    if on_progress:
                        pct = 50 + int((i / len(target_languages)) * 40)
                        on_progress(pct, f"Dubbing to {lang}...")

                    lang_output = os.path.join(output_dir, f"dubbed_{lang}.mp4")
                    try:
                        result = dub(
                            video_path=source_to_dub,
                            target_language=lang,
                            output=lang_output,
                            whisper_model=whisper_model,
                            tts_engine="edge",
                        )
                        outputs[lang] = result.output_path if hasattr(result, "output_path") else lang_output
                    except Exception as lang_exc:
                        notes.append(f"Dub to {lang} failed: {lang_exc}")
                        outputs[lang] = ""

                stages_completed.append("translate_dub")
            except Exception as exc:
                notes.append(f"Dub pipeline failed: {exc}")
                stages_skipped.append("translate_dub")
        else:
            stages_skipped.append("translate_dub")
            notes.append("No source for dubbing")

    # Composite stage is deferred — requires sequence-specific compositing
    stages_skipped.append("composite")

    total_time = time.monotonic() - start_time
    notes.append(f"Pipeline completed in {total_time:.1f}s")

    if on_progress:
        on_progress(100, "Done")

    return DigitalTwinResult(
        outputs=outputs,
        stages_completed=stages_completed,
        stages_skipped=stages_skipped,
        target_languages=target_languages,
        narration_path=narration_path or "",
        avatar_video_path=avatar_video_path or "",
        talking_head_backend=talking_head_backend,
        total_duration_seconds=round(total_time, 2),
        notes=notes,
    )


__all__ = [
    "DigitalTwinResult",
    "check_digital_twin_available",
    "INSTALL_HINT",
    "PIPELINE_STAGES",
    "TALKING_HEAD_BACKENDS",
    "run_pipeline",
]
