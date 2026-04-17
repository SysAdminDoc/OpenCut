"""
F5-TTS (Flow-matching zero-shot voice cloning) provider for OpenCut.

F5-TTS is a flow-matching TTS model that produces a voice clone from a
~15 s reference in one shot — notably faster inference than XTTS-v2 at
comparable quality.

Source: https://github.com/SWivid/F5-TTS  (MIT)
Paper:  https://arxiv.org/abs/2410.06885

Follows the same shape as :mod:`opencut.core.voice_gen` Chatterbox
integration so routes can swap providers by name without special cases.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from opencut.helpers import ensure_package

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------

F5_MODELS = {
    "F5-TTS": {
        "description": "Default F5-TTS base model (multilingual, ~1.3 B params).",
        "repo_id": "SWivid/F5-TTS",
        "ckpt_filename": "F5TTS_Base/model_1200000.safetensors",
    },
    "E2-TTS": {
        "description": "E2-TTS companion model (smaller, faster).",
        "repo_id": "SWivid/E2-TTS",
        "ckpt_filename": "E2TTS_Base/model_1200000.safetensors",
    },
}

DEFAULT_MODEL = "F5-TTS"
_MODEL_CACHE: dict = {"model": None, "name": ""}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class F5Result:
    """Structured return for an F5-TTS synthesis."""
    output_path: str = ""
    model: str = ""
    duration_seconds: float = 0.0
    sample_rate: int = 24000
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, key):
        return getattr(self, key)

    def __contains__(self, key):
        return key in self.__dataclass_fields__

    def keys(self):
        return self.__dataclass_fields__.keys()


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def check_f5_available() -> bool:
    """True when `f5_tts` is importable."""
    try:
        import f5_tts  # noqa: F401
        return True
    except ImportError:
        return False


def list_models() -> List[dict]:
    """Return the supported F5/E2 models with their HF repo IDs."""
    return [
        {"name": name, **info}
        for name, info in F5_MODELS.items()
    ]


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

def f5_generate(
    text: str,
    voice_ref: str,
    ref_text: Optional[str] = None,
    output_path: Optional[str] = None,
    output_dir: str = "",
    model: str = DEFAULT_MODEL,
    speed: float = 1.0,
    on_progress: Optional[Callable] = None,
) -> F5Result:
    """Synthesize ``text`` in the style of ``voice_ref`` using F5-TTS.

    Args:
        text:       Text to synthesize. Required.
        voice_ref:  Path to a 5–30 s reference audio clip. Required —
            F5-TTS is zero-shot; without a reference there is no voice
            to clone.
        ref_text:   Optional transcript of ``voice_ref``. F5-TTS produces
            higher quality when the reference transcript is provided.
            Leave ``None`` to let F5-TTS infer via Whisper.
        output_path: Explicit output WAV path.
        output_dir: Directory to place output (used if ``output_path``
            is None).
        model:      Model name from :data:`F5_MODELS`.
        speed:      Playback speed multiplier (0.5–2.0).
        on_progress: ``(pct, msg)`` progress callback.

    Returns:
        :class:`F5Result`.

    Raises:
        RuntimeError: F5-TTS is not installed and could not be installed.
        ValueError:   invalid arguments.
        FileNotFoundError: ``voice_ref`` missing.
    """
    if not text or not str(text).strip():
        raise ValueError("text is required")
    if not voice_ref or not os.path.isfile(voice_ref):
        raise FileNotFoundError(f"voice_ref not found: {voice_ref}")
    if model not in F5_MODELS:
        raise ValueError(
            f"Unknown F5 model '{model}'. Available: {list(F5_MODELS.keys())}"
        )
    speed = max(0.5, min(2.0, float(speed or 1.0)))

    if not check_f5_available():
        ok = ensure_package("f5_tts", "f5-tts", on_progress)
        if not ok:
            raise RuntimeError(
                "f5-tts not installed and automatic install failed. "
                "Run: pip install f5-tts"
            )

    if output_path is None:
        directory = output_dir or tempfile.gettempdir()
        fname = f"f5_{abs(hash(text[:40])) & 0xFFFF:04x}.wav"
        output_path = os.path.join(directory, fname)

    if on_progress:
        on_progress(10, f"Loading {model}…")

    # Lazy import — keeps module-import cost low when F5 isn't used.
    import torch
    from f5_tts.api import F5TTS  # public API since v0.2.0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cached_model = None
    try:
        if _MODEL_CACHE["name"] == model and _MODEL_CACHE["model"] is not None:
            cached_model = _MODEL_CACHE["model"]
        else:
            cached_model = F5TTS(model_type=model, device=device)
            _MODEL_CACHE["model"] = cached_model
            _MODEL_CACHE["name"] = model

        if on_progress:
            on_progress(40, "Synthesising speech…")

        # f5_tts.api returns (wav, sr, spectrogram) tuple on most versions.
        res = cached_model.infer(
            ref_file=voice_ref,
            ref_text=ref_text or "",
            gen_text=text,
            speed=speed,
            file_wave=output_path,
        )

        if on_progress:
            on_progress(90, "Saving audio…")

        # Duration hint: `res` may be an audio tensor or a tuple; we
        # don't rely on it — probe the output after save for accuracy.
        if isinstance(res, tuple) and len(res) >= 2:
            wav, sr = res[0], res[1]
        else:
            wav, sr = None, 24000

        duration = 0.0
        if wav is not None:
            try:
                if hasattr(wav, "shape"):
                    duration = float(wav.shape[-1]) / float(sr or 24000)
            except Exception:  # noqa: BLE001
                duration = 0.0
    finally:
        # Defensive GPU cleanup — the cached model stays alive, but
        # intermediate tensors are freed by the api call itself.
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    if on_progress:
        on_progress(100, "F5 synthesis complete")

    return F5Result(
        output_path=output_path,
        model=model,
        duration_seconds=round(duration, 3),
        sample_rate=int(sr or 24000),
        notes=[f"device={device}"],
    )


def clear_model_cache() -> None:
    """Release the cached F5 model (frees GPU VRAM)."""
    if _MODEL_CACHE["model"] is not None:
        try:
            del _MODEL_CACHE["model"]
        except Exception:  # noqa: BLE001
            pass
        _MODEL_CACHE["model"] = None
        _MODEL_CACHE["name"] = ""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
