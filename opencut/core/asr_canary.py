"""Pinned NVIDIA Canary-1B-Flash multilingual batch ASR adapter."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from opencut.core.asr_nemo import (
    nemo_runtime_available,
    nemo_runtime_status,
    transcribe_many,
    write_canary_manifest,
)
from opencut.core.asr_nemo_models import CANARY_SPEC
from opencut.core.captions import TranscriptionResult, transcription_result_to_dict

INSTALL_HINT = 'python -m pip install -e ".[nemo-asr]"  # Linux only'
SUPPORTED_LANGUAGES = frozenset({"de", "en", "es", "fr"})


@dataclass
class CanaryResult:
    """Ordered batch result using OpenCut's common transcript schema."""

    source_paths: list[str]
    results: list[TranscriptionResult]
    processing_seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(result.text for result in self.results)

    @property
    def segments(self) -> list[Any]:
        return [segment for result in self.results for segment in result.segments]

    @property
    def model(self) -> str:
        return CANARY_SPEC.model_id

    @property
    def model_revision(self) -> str:
        return CANARY_SPEC.revision

    @property
    def cache_hit(self) -> bool:
        return bool(self.results) and all(result.cache_hit for result in self.results)

    def to_dict(self) -> dict[str, Any]:
        items = []
        for source_path, result in zip(self.source_paths, self.results):
            item = transcription_result_to_dict(result)
            item.update(
                {
                    "source_path": source_path,
                    "text": result.text,
                    "cache_hit": result.cache_hit,
                    "cache_key": result.cache_key,
                    "cache_path": result.cache_path,
                }
            )
            items.append(item)
        return {
            "text": self.text,
            "segments": [segment for item in items for segment in item["segments"]],
            "items": items,
            "model": self.model,
            "model_revision": self.model_revision,
            "processing_seconds": self.processing_seconds,
            "cache_hit": self.cache_hit,
            "notes": list(self.notes),
        }

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
    """Return runtime, languages, and checkpoint metadata for readiness APIs."""
    return {
        **nemo_runtime_status(),
        "engine": CANARY_SPEC.engine,
        "model": CANARY_SPEC.model_id,
        "model_revision": CANARY_SPEC.revision,
        "model_license": "CC-BY-4.0",
        "supported_languages": sorted(SUPPORTED_LANGUAGES),
        "install_hint": INSTALL_HINT,
    }


def _language(value: Optional[str], name: str) -> str:
    selected = str(value or "en").strip().lower().split("-", 1)[0]
    if selected not in SUPPORTED_LANGUAGES:
        supported = ", ".join(sorted(SUPPORTED_LANGUAGES))
        raise ValueError(f"Unsupported Canary {name} '{selected}'; choose one of: {supported}")
    return selected


def transcribe_batch(
    audio_paths: Optional[Iterable[str]] = None,
    *,
    audio_path: Optional[str] = None,
    filepath: Optional[str] = None,
    language: str = "en",
    source_language: Optional[str] = None,
    target_language: Optional[str] = None,
    translate: bool = False,
    batch_size: int = 8,
    use_cache: bool = True,
    allow_download: bool = True,
    checkpoint_path: Optional[str] = None,
    device: str = "auto",
    on_progress: Optional[Callable[[int, str], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
    model_instance: Any = None,
    model_loader: Optional[Callable[..., Any]] = None,
    audio_preparer: Optional[Callable[[str], str]] = None,
) -> CanaryResult:
    """Transcribe or translate an ordered batch using Canary's manifest API."""
    sources = [str(path) for path in (audio_paths or []) if str(path).strip()]
    single = str(audio_path or filepath or "").strip()
    if single:
        sources.insert(0, single)
    if not sources:
        raise ValueError("audio_paths, audio_path, or filepath is required")
    missing = [path for path in sources if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(missing[0])
    if len(sources) > 64:
        raise ValueError("Canary batches are limited to 64 files")

    source = _language(source_language or language, "source language")
    target = _language(target_language or ("en" if translate else source), "target language")

    def infer(model: Any, paths: list[str], selected_batch_size: int):
        manifest_path = write_canary_manifest(
            paths,
            source_language=source,
            target_language=target,
        )
        try:
            return model.transcribe(manifest_path, batch_size=selected_batch_size)
        finally:
            try:
                os.unlink(manifest_path)
            except OSError:
                pass

    started = time.monotonic()
    results = transcribe_many(
        CANARY_SPEC,
        sources,
        language=source,
        target_language=target,
        translate=bool(translate or target != source),
        inference=infer,
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
    return CanaryResult(
        source_paths=[os.path.realpath(path) for path in sources],
        results=results,
        processing_seconds=round(time.monotonic() - started, 3),
        notes=[
            "Pinned CC-BY-4.0 checkpoint; preserve model attribution with exports."
        ],
    )


__all__ = [
    "CanaryResult",
    "INSTALL_HINT",
    "SUPPORTED_LANGUAGES",
    "check_nemo_toolkit_available",
    "runtime_info",
    "transcribe_batch",
]
