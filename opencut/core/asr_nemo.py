"""Shared runtime for pinned NVIDIA NeMo ASR adapters.

This module keeps the heavy NeMo import and multi-gigabyte checkpoint load
behind a lazy boundary.  Transcript-cache reads happen first, so a previously
completed local transcript remains usable while offline or without NeMo.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import sys
import tempfile
import threading
import time
import wave
from contextlib import nullcontext
from importlib import metadata, util
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Optional, Sequence

from opencut.core import transcript_cache
from opencut.core.asr_nemo_models import MIN_NEMO_VERSION, NemoModelSpec
from opencut.core.asr_provenance import apply_language_decision, build_provenance
from opencut.core.audio import extract_audio_wav
from opencut.core.captions import (
    CaptionSegment,
    TranscriptionResult,
    Word,
    transcription_result_from_dict,
    transcription_result_to_dict,
)

logger = logging.getLogger("opencut")

ProgressCallback = Optional[Callable[[int, str], None]]
CancellationCallback = Optional[Callable[[], bool]]
InferenceCallback = Callable[[Any, list[str], int], Sequence[Any]]
AudioPreparer = Callable[[str], str]

_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}
_MODEL_CACHE_LOCK = threading.RLock()
_VERIFIED_CHECKPOINTS: dict[str, tuple[int, int, int]] = {}
_VERIFIED_CHECKPOINTS_LOCK = threading.RLock()


def _version_tuple(value: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", str(value or ""))
    return tuple(int(part) for part in parts[:3])


def nemo_toolkit_version() -> str:
    """Return the installed NeMo distribution version without importing it."""
    try:
        return metadata.version("nemo-toolkit")
    except metadata.PackageNotFoundError:
        return "unavailable"


def nemo_runtime_status(platform_name: Optional[str] = None) -> dict[str, Any]:
    """Describe the conservative, officially supported OpenCut NeMo lane."""
    selected_platform = str(platform_name or sys.platform).lower()
    platform_supported = selected_platform.startswith("linux")
    version = nemo_toolkit_version()
    parsed = _version_tuple(version)
    version_supported = bool(parsed) and (2, 7, 3) <= parsed < (2, 8)
    try:
        import_available = util.find_spec("nemo") is not None
    except (ImportError, ValueError):
        import_available = False

    gpu_available = False
    if import_available:
        try:
            import torch

            gpu_available = bool(torch.cuda.is_available())
        except Exception:  # noqa: BLE001 - readiness must not import-crash
            gpu_available = False

    reason = ""
    if not platform_supported:
        reason = (
            "NeMo 2.7.3 does not publish a supported Windows runtime; "
            "OpenCut enables these adapters on Linux only."
        )
    elif not import_available:
        reason = 'Install the Linux-only source extra: python -m pip install -e ".[nemo-asr]"'
    elif not version_supported:
        reason = f"NeMo {version} is outside OpenCut's supported >=2.7.3,<2.8 range."

    return {
        "available": bool(platform_supported and import_available and version_supported),
        "installed": import_available,
        "version": version,
        "version_supported": version_supported,
        "minimum_version": MIN_NEMO_VERSION,
        "platform": selected_platform,
        "platform_supported": platform_supported,
        "supported_platforms": ["linux"],
        "gpu_available": gpu_available,
        "recommended_hardware": "NVIDIA CUDA GPU",
        "cpu_performance": "unbenchmarked by OpenCut",
        "reason": reason,
    }


def nemo_runtime_available() -> bool:
    return bool(nemo_runtime_status()["available"])


def _report(on_progress: ProgressCallback, percent: int, message: str) -> None:
    if on_progress:
        on_progress(max(0, min(100, int(percent))), str(message))


def _raise_if_cancelled(is_cancelled: CancellationCallback) -> None:
    if is_cancelled and is_cancelled():
        raise InterruptedError("NeMo transcription cancelled")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(path: str, spec: NemoModelSpec) -> str:
    """Fail closed unless *path* is the pinned, scanned NeMo checkpoint."""
    resolved = os.path.realpath(path)
    stat = os.stat(resolved)
    identity = (int(stat.st_size), int(stat.st_mtime_ns), int(stat.st_ctime_ns))
    with _VERIFIED_CHECKPOINTS_LOCK:
        if _VERIFIED_CHECKPOINTS.get(resolved) == identity:
            return resolved

    if stat.st_size != spec.size_bytes:
        raise RuntimeError(
            f"Refusing {spec.filename}: expected {spec.size_bytes} bytes, "
            f"found {stat.st_size}. Re-download the pinned checkpoint."
        )
    actual_sha256 = _sha256(resolved)
    if actual_sha256.lower() != spec.sha256.lower():
        raise RuntimeError(
            f"Refusing {spec.filename}: SHA-256 does not match pinned revision "
            f"{spec.revision}."
        )

    from opencut.core.model_safety import scan_model_file

    scan_model_file(resolved)
    with _VERIFIED_CHECKPOINTS_LOCK:
        _VERIFIED_CHECKPOINTS[resolved] = identity
    return resolved


def _managed_checkpoint_path(spec: NemoModelSpec) -> str:
    from opencut.core.model_manager import MODELS_DIR

    return os.path.join(MODELS_DIR, f"{spec.key}.nemo")


def resolve_checkpoint(
    spec: NemoModelSpec,
    *,
    checkpoint_path: Optional[str] = None,
    allow_download: bool = True,
) -> str:
    """Resolve a model-manager file or the exact pinned Hub snapshot."""
    candidates = [checkpoint_path, _managed_checkpoint_path(spec)]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return verify_checkpoint(candidate, spec)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            'huggingface-hub is required. Install python -m pip install -e ".[nemo-asr]"'
        ) from exc

    try:
        downloaded = hf_hub_download(
            repo_id=spec.model_id,
            filename=spec.filename,
            revision=spec.revision,
            local_files_only=not allow_download,
        )
    except Exception as exc:  # noqa: BLE001 - normalize Hub/cache failures
        mode = "local cache" if not allow_download else "pinned Hugging Face snapshot"
        raise RuntimeError(
            f"Could not resolve {spec.model_id}@{spec.revision} from the {mode}: {exc}"
        ) from exc
    return verify_checkpoint(downloaded, spec)


def choose_device(requested: str = "auto") -> str:
    selected = str(requested or "auto").strip().lower()
    if selected not in {"auto", "cpu", "cuda"}:
        raise ValueError("NeMo device must be one of: auto, cpu, cuda")
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - adapter reports a clear load error later
        cuda_available = False
    if selected == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested for NeMo ASR but no CUDA device is available")
    if selected == "auto":
        return "cuda" if cuda_available else "cpu"
    return selected


def load_model(
    spec: NemoModelSpec,
    *,
    device: str = "auto",
    checkpoint_path: Optional[str] = None,
    allow_download: bool = True,
) -> Any:
    """Load one verified NeMo model once per process/device."""
    status = nemo_runtime_status()
    if not status["available"]:
        raise RuntimeError(status["reason"] or "NeMo ASR runtime is unavailable")
    selected_device = choose_device(device)
    checkpoint = resolve_checkpoint(
        spec,
        checkpoint_path=checkpoint_path,
        allow_download=allow_download,
    )
    key = (spec.key, checkpoint, selected_device)
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached

        if spec.loader == "multitask":
            from nemo.collections.asr.models import EncDecMultiTaskModel

            model_class = EncDecMultiTaskModel
        else:
            import nemo.collections.asr as nemo_asr

            model_class = nemo_asr.models.ASRModel
        model = model_class.restore_from(
            restore_path=checkpoint,
            map_location=selected_device,
        )
        if hasattr(model, "eval"):
            model.eval()
        if hasattr(model, "to"):
            model.to(selected_device)
        _MODEL_CACHE[key] = model
        return model


def clear_model_cache() -> None:
    """Release cached adapter references; primarily useful for tests/shutdown."""
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def _confidence(mapping: Any, fallback: float = 0.0) -> float:
    for name in ("confidence", "probability", "prob"):
        value = mapping.get(name) if isinstance(mapping, dict) else getattr(mapping, name, None)
        if value is not None:
            return max(0.0, min(1.0, _number(value, fallback)))
    return max(0.0, min(1.0, fallback))


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _timestamps(hypothesis: Any, level: str) -> list[Any]:
    timestamp = _field(hypothesis, "timestamp", {}) or {}
    if isinstance(timestamp, dict):
        values = timestamp.get(level, [])
    else:
        values = getattr(timestamp, level, [])
    return list(values or [])


def _stamp_times(stamp: Any) -> tuple[float, float]:
    start = max(0.0, _number(_field(stamp, "start", 0.0)))
    end = max(start, _number(_field(stamp, "end", start), start))
    return round(start, 4), round(end, 4)


def _audio_duration(path: str) -> float:
    try:
        with wave.open(path, "rb") as handle:
            rate = handle.getframerate()
            return round(handle.getnframes() / rate, 4) if rate else 0.0
    except (OSError, wave.Error):
        return 0.0


def normalize_hypothesis(
    hypothesis: Any,
    *,
    language: str,
    duration: float,
    provenance: Any,
) -> TranscriptionResult:
    """Convert NeMo hypothesis/timestamp objects to OpenCut caption types."""
    full_text = str(
        hypothesis if isinstance(hypothesis, str) else _field(hypothesis, "text", "")
    ).strip()
    hypothesis_confidence = _confidence(hypothesis)
    words: list[Word] = []
    for stamp in _timestamps(hypothesis, "word"):
        start, end = _stamp_times(stamp)
        text = str(
            _field(stamp, "word", _field(stamp, "text", _field(stamp, "char", "")))
            or ""
        ).strip()
        if not text:
            continue
        words.append(
            Word(
                text=text,
                start=start,
                end=end,
                confidence=_confidence(stamp, hypothesis_confidence),
                boundary_confidence=_field(stamp, "boundary_confidence"),
            )
        )
    words.sort(key=lambda word: (word.start, word.end))

    segments: list[CaptionSegment] = []
    segment_stamps = _timestamps(hypothesis, "segment")
    for index, stamp in enumerate(segment_stamps):
        start, end = _stamp_times(stamp)
        segment_words = [
            word for word in words
            if word.end > start and word.start < end
        ]
        text = str(
            _field(stamp, "segment", _field(stamp, "text", "")) or ""
        ).strip()
        if not text and segment_words:
            text = " ".join(word.text for word in segment_words)
        if not text and index == 0 and len(segment_stamps) == 1:
            text = full_text
        if not text:
            continue
        segments.append(
            CaptionSegment(
                text=text,
                start=start,
                end=end,
                words=segment_words,
                language=language,
                confidence=_confidence(stamp, hypothesis_confidence),
                boundary_confidence=_field(stamp, "boundary_confidence"),
            )
        )

    if not segments and (full_text or words):
        start = words[0].start if words else 0.0
        end = words[-1].end if words else max(0.0, duration)
        segments.append(
            CaptionSegment(
                text=full_text or " ".join(word.text for word in words),
                start=start,
                end=end,
                words=words,
                language=language,
                confidence=hypothesis_confidence,
            )
        )

    result_duration = max(
        [max(0.0, duration), *(segment.end for segment in segments)],
        default=0.0,
    )
    return TranscriptionResult(
        segments=segments,
        language=language,
        duration=result_duration,
        provenance=provenance,
    )


def _cache_config(
    spec: NemoModelSpec,
    *,
    language: str,
    translate: bool,
) -> Any:
    return SimpleNamespace(
        engine=spec.engine,
        model=spec.key,
        model_revision=spec.revision,
        language=language,
        word_timestamps=True,
        translate=translate,
        diarize=False,
        min_speakers=None,
        max_speakers=None,
    )


def _provenance(
    spec: NemoModelSpec,
    *,
    language: str,
    target_language: str,
    translate: bool,
    device: str,
) -> Any:
    value = build_provenance(
        engine=spec.engine,
        requested_engine=spec.engine,
        model=spec.key,
        model_revision=spec.revision,
        requested_language=language,
        word_timestamps=True,
        translate=translate,
        diarize=False,
        min_speakers=None,
        max_speakers=None,
    )
    value.device = device
    value.compute_type = "NeMo checkpoint default"
    value.deterministic_options.update(
        {
            "checkpoint_filename": spec.filename,
            "checkpoint_sha256": spec.sha256,
            "source_language": language,
            "target_language": target_language,
            "confidence_source": (
                "NeMo timestamp metadata when available; schema fallback is 0.0"
            ),
        }
    )
    apply_language_decision(value, language)
    return value


def _prepare_audio(path: str, preparer: Optional[AudioPreparer]) -> tuple[str, bool]:
    prepared = preparer(path) if preparer else extract_audio_wav(path, sample_rate=16000)
    prepared = os.path.realpath(str(prepared))
    original = os.path.realpath(path)
    cleanup = prepared != original and prepared.startswith(os.path.realpath(tempfile.gettempdir()))
    return prepared, cleanup


def _inference_context():
    try:
        import torch

        return torch.inference_mode()
    except Exception:  # noqa: BLE001 - fake/test models do not require Torch
        return nullcontext()


def transcribe_many(
    spec: NemoModelSpec,
    audio_paths: Iterable[str],
    *,
    language: str,
    target_language: Optional[str],
    translate: bool,
    inference: InferenceCallback,
    batch_size: int = 1,
    use_cache: bool = True,
    allow_download: bool = True,
    checkpoint_path: Optional[str] = None,
    device: str = "auto",
    on_progress: ProgressCallback = None,
    is_cancelled: CancellationCallback = None,
    model_instance: Any = None,
    model_loader: Optional[Callable[..., Any]] = None,
    audio_preparer: Optional[AudioPreparer] = None,
) -> list[TranscriptionResult]:
    """Cache, preprocess, infer, normalize, and persist one ordered batch."""
    sources = [os.path.realpath(str(path)) for path in audio_paths]
    if not sources:
        return []
    source_language = str(language or "en").lower()
    output_language = str(target_language or source_language).lower()
    requested_device = str(device or "auto")
    config = _cache_config(spec, language=source_language, translate=translate)
    provenance = _provenance(
        spec,
        language=source_language,
        target_language=output_language,
        translate=translate,
        device=requested_device,
    )
    extra = {
        "asr_provenance": provenance.cache_identity(),
        "target_language": output_language,
        "checkpoint_sha256": spec.sha256,
    }
    results: list[Optional[TranscriptionResult]] = [None] * len(sources)
    cache_records: dict[int, tuple[str, dict[str, Any]]] = {}
    missing: list[int] = []

    _report(on_progress, 1, "Checking transcript cache")
    for index, source in enumerate(sources):
        _raise_if_cancelled(is_cancelled)
        if use_cache and transcript_cache.cache_enabled():
            try:
                key, cache_metadata = transcript_cache.build_cache_key(
                    source,
                    backend=spec.engine,
                    config=config,
                    extra=extra,
                )
                cache_records[index] = (key, cache_metadata)
                cached = transcript_cache.load_transcript(key)
                if cached:
                    results[index] = transcription_result_from_dict(
                        cached["result"],
                        cache_hit=True,
                        cache_key=key,
                        cache_path=transcript_cache.cache_entry_path(key),
                    )
                    continue
            except Exception as exc:  # noqa: BLE001 - cache failure is non-fatal
                logger.warning("NeMo transcript cache read failed for %s: %s", source, exc)
        missing.append(index)

    if not missing:
        _report(on_progress, 100, "Loaded transcript from local cache")
        return [result for result in results if result is not None]

    _raise_if_cancelled(is_cancelled)
    selected_device = (
        choose_device(device) if model_instance is None else requested_device
    )
    prepared_paths: list[str] = []
    cleanup_paths: list[str] = []
    durations: list[float] = []
    started = time.monotonic()
    try:
        for position, index in enumerate(missing):
            _raise_if_cancelled(is_cancelled)
            prepared, cleanup = _prepare_audio(sources[index], audio_preparer)
            prepared_paths.append(prepared)
            durations.append(_audio_duration(prepared))
            if cleanup:
                cleanup_paths.append(prepared)
            _report(
                on_progress,
                10 + round(20 * (position + 1) / len(missing)),
                "Preparing 16 kHz mono audio",
            )

        _raise_if_cancelled(is_cancelled)
        _report(on_progress, 35, f"Loading {spec.model_id}@{spec.revision[:12]}")
        if model_instance is not None:
            model = model_instance
        else:
            loader = model_loader or load_model
            model = loader(
                spec,
                device=selected_device,
                checkpoint_path=checkpoint_path,
                allow_download=allow_download,
            )
        _raise_if_cancelled(is_cancelled)
        _report(on_progress, 45, "Running local NeMo inference")
        with _inference_context():
            hypotheses = list(inference(model, prepared_paths, max(1, int(batch_size))))
        if len(hypotheses) != len(prepared_paths):
            raise RuntimeError(
                f"NeMo returned {len(hypotheses)} hypotheses for "
                f"{len(prepared_paths)} audio files"
            )

        for position, (index, hypothesis, duration) in enumerate(
            zip(missing, hypotheses, durations)
        ):
            _raise_if_cancelled(is_cancelled)
            result = normalize_hypothesis(
                hypothesis,
                language=output_language,
                duration=duration,
                provenance=_provenance(
                    spec,
                    language=source_language,
                    target_language=output_language,
                    translate=translate,
                    device=selected_device,
                ),
            )
            result.cache_hit = False
            record = cache_records.get(index)
            if record:
                key, cache_metadata = record
                result.cache_key = key
                try:
                    result.cache_path = transcript_cache.store_transcript(
                        key,
                        cache_metadata,
                        transcription_result_to_dict(result),
                    )
                except Exception as exc:  # noqa: BLE001 - inference result survives
                    logger.warning("NeMo transcript cache write failed for %s: %s", sources[index], exc)
            results[index] = result
            _report(
                on_progress,
                90 + round(9 * (position + 1) / len(missing)),
                "Normalizing timestamped transcript",
            )
        logger.info(
            "NeMo %s transcribed %d file(s) in %.2fs",
            spec.key,
            len(missing),
            time.monotonic() - started,
        )
        _report(on_progress, 100, "Transcription complete")
        return [result for result in results if result is not None]
    finally:
        for path in cleanup_paths:
            try:
                os.unlink(path)
            except OSError:
                pass


def write_canary_manifest(
    audio_paths: Sequence[str],
    *,
    source_language: str,
    target_language: str,
) -> str:
    """Write the official Canary JSONL prompt format to a temporary file."""
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".jsonl",
        prefix="opencut-canary-",
        delete=False,
    )
    try:
        for path in audio_paths:
            handle.write(
                json.dumps(
                    {
                        "audio_filepath": os.path.realpath(path),
                        "source_lang": source_language,
                        "target_lang": target_language,
                        "pnc": "yes",
                        "timestamp": "yes",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    finally:
        handle.close()
    return handle.name


__all__ = [
    "clear_model_cache",
    "load_model",
    "nemo_runtime_available",
    "nemo_runtime_status",
    "nemo_toolkit_version",
    "normalize_hypothesis",
    "resolve_checkpoint",
    "transcribe_many",
    "verify_checkpoint",
    "write_canary_manifest",
]
