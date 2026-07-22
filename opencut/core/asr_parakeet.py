"""Pinned NVIDIA Parakeet TDT batch transcription adapter.

The implementation intentionally imports NeMo only while loading the model.
It emits OpenCut's shared timestamp/confidence/provenance schema and uses the
content-addressed transcript cache before touching the optional runtime.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from opencut.core.asr_nemo import (
    nemo_runtime_available,
    nemo_runtime_status,
    transcribe_many,
)
from opencut.core.asr_nemo_models import PARAKEET_SPEC
from opencut.core.captions import TranscriptionResult, transcription_result_to_dict

INSTALL_HINT = 'python -m pip install -e ".[nemo-asr]"  # Linux only'


@dataclass
class ParakeetResult:
    """Compatibility wrapper around the common transcription result."""

    result: TranscriptionResult
    processing_seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return self.result.text

    @property
    def segments(self) -> list[Any]:
        return self.result.segments

    @property
    def model(self) -> str:
        return PARAKEET_SPEC.model_id

    @property
    def model_revision(self) -> str:
        return PARAKEET_SPEC.revision

    @property
    def cache_hit(self) -> bool:
        return self.result.cache_hit

    @property
    def provenance(self) -> Any:
        return self.result.provenance

    def to_dict(self) -> dict[str, Any]:
        payload = transcription_result_to_dict(self.result)
        payload.update(
            {
                "text": self.text,
                "model": self.model,
                "model_revision": self.model_revision,
                "processing_seconds": self.processing_seconds,
                "cache_hit": self.result.cache_hit,
                "cache_key": self.result.cache_key,
                "cache_path": self.result.cache_path,
                "notes": list(self.notes),
            }
        )
        return payload

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def keys(self):
        return self.to_dict().keys()

    def __contains__(self, key: str) -> bool:
        return key in self.to_dict()


def check_nemo_toolkit_available() -> bool:
    """Return whether the pinned NeMo runtime is usable on this platform."""
    return nemo_runtime_available()


def runtime_info() -> dict[str, Any]:
    """Return runtime and immutable checkpoint metadata for readiness APIs."""
    return {
        **nemo_runtime_status(),
        "engine": PARAKEET_SPEC.engine,
        "model": PARAKEET_SPEC.model_id,
        "model_revision": PARAKEET_SPEC.revision,
        "model_license": "CC-BY-4.0",
        "install_hint": INSTALL_HINT,
    }


def _infer(model: Any, paths: list[str], batch_size: int):
    return model.transcribe(paths, batch_size=batch_size, timestamps=True)


def transcribe(
    audio_path: Optional[str] = None,
    *,
    filepath: Optional[str] = None,
    language: str = "en",
    batch_size: int = 1,
    use_cache: bool = True,
    allow_download: bool = True,
    checkpoint_path: Optional[str] = None,
    device: str = "auto",
    on_progress: Optional[Callable[[int, str], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
    model_instance: Any = None,
    model_loader: Optional[Callable[..., Any]] = None,
    audio_preparer: Optional[Callable[[str], str]] = None,
) -> ParakeetResult:
    """Transcribe one media file using the pinned Parakeet checkpoint."""
    source = str(audio_path or filepath or "").strip()
    if not source:
        raise ValueError("audio_path or filepath is required")
    if not os.path.isfile(source):
        raise FileNotFoundError(source)
    from opencut.core.asr_router import PARAKEET_V3_LANGUAGES

    language_code = str(language or "").strip().lower().split("-", 1)[0]
    if language_code not in PARAKEET_V3_LANGUAGES:
        raise ValueError(
            f"Unsupported Parakeet language '{language_code}'; choose one of: "
            + ", ".join(sorted(PARAKEET_V3_LANGUAGES))
        )

    started = time.monotonic()
    results = transcribe_many(
        PARAKEET_SPEC,
        [source],
        language=language_code,
        target_language=language_code,
        translate=False,
        inference=_infer,
        batch_size=batch_size,
        use_cache=use_cache,
        allow_download=allow_download,
        checkpoint_path=checkpoint_path,
        device=device,
        on_progress=on_progress,
        is_cancelled=is_cancelled,
        model_instance=model_instance,
        model_loader=model_loader,
        audio_preparer=audio_preparer,
    )
    return ParakeetResult(
        result=results[0],
        processing_seconds=round(time.monotonic() - started, 3),
        notes=[
            "Pinned CC-BY-4.0 checkpoint; preserve model attribution with exports."
        ],
    )


__all__ = [
    "INSTALL_HINT",
    "ParakeetResult",
    "check_nemo_toolkit_available",
    "runtime_info",
    "transcribe",
]
