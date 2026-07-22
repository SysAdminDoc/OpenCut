"""Stable provenance contracts for transcript and caption artifacts."""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Optional

ASR_PROVENANCE_SCHEMA_VERSION = 1

ENGINE_ALIASES = {
    "faster_whisper": "faster-whisper",
    "openai_whisper": "openai-whisper",
    "whisper": "openai-whisper",
}

# Immutable revisions resolved from the model repositories on 2026-07-21.
# Passing these revisions to faster-whisper prevents a mutable Hub ``main``
# branch from silently changing transcript boundaries or cache identity.
FASTER_WHISPER_MODELS = {
    "tiny.en": ("Systran/faster-whisper-tiny.en", "0d3d19a32d3338f10357c0889762bd8d64bbdeba"),
    "tiny": ("Systran/faster-whisper-tiny", "d90ca5fe260221311c53c58e660288d3deb8d356"),
    "base.en": ("Systran/faster-whisper-base.en", "3d3d5dee26484f91867d81cb899cfcf72b96be6c"),
    "base": ("Systran/faster-whisper-base", "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66"),
    "small.en": ("Systran/faster-whisper-small.en", "d1d751a5f8271d482d14ca55d9e2deeebbae577f"),
    "small": ("Systran/faster-whisper-small", "536b0662742c02347bc0e980a01041f333bce120"),
    "medium.en": ("Systran/faster-whisper-medium.en", "a29b04bd15381511a9af671baec01072039215e3"),
    "medium": ("Systran/faster-whisper-medium", "08e178d48790749d25932bbc082711ddcfdfbc4f"),
    "large-v1": ("Systran/faster-whisper-large-v1", "b07c8d4be0be90092aa01a29c975077acb8d15c9"),
    "large-v2": ("Systran/faster-whisper-large-v2", "f0fe81560cb8b68660e564f55dd99207059c092e"),
    "large-v3": ("Systran/faster-whisper-large-v3", "edaa852ec7e145841d8ffdb056a99866b5f0a478"),
    "large": ("Systran/faster-whisper-large-v3", "edaa852ec7e145841d8ffdb056a99866b5f0a478"),
    "distil-large-v2": ("Systran/faster-distil-whisper-large-v2", "fe9b404fc56de3f7c38606ef9ba6fd83526d05e4"),
    "distil-medium.en": ("Systran/faster-distil-whisper-medium.en", "80ddfce281f77766d8943d63109199fc8145dfa5"),
    "distil-small.en": ("Systran/faster-distil-whisper-small.en", "ef77d90526ccd62cde3808ee70626a01e5cf83e4"),
    "distil-large-v3": ("Systran/faster-distil-whisper-large-v3", "c3058b475261292e64a0412df1d2681c06260fab"),
    "distil-large-v3.5": ("distil-whisper/distil-large-v3.5-ct2", "9793ccc07920e0f830e1dba0343efcdf0ef8c903"),
    "large-v3-turbo": ("mobiuslabsgmbh/faster-whisper-large-v3-turbo", "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"),
    "turbo": ("mobiuslabsgmbh/faster-whisper-large-v3-turbo", "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"),
}


def package_version(package: str) -> str:
    """Return an installed package version without importing the package."""
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "unavailable"


def normalize_engine(engine: Optional[str]) -> str:
    value = str(engine or "").strip().lower()
    return ENGINE_ALIASES.get(value, value)


def _local_model_revision(path: str) -> str:
    """Fingerprint a local model manifest without hashing multi-GB weights."""
    root = Path(path).resolve()
    digest = hashlib.sha256()
    for child in sorted(root.rglob("*")):
        if not child.is_file():
            continue
        relative = child.relative_to(root).as_posix()
        stat = child.stat()
        digest.update(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
        if child.name in {"config.json", "preprocessor_config.json"}:
            digest.update(child.read_bytes())
    return f"local-manifest-sha256:{digest.hexdigest()}"


def model_identity(
    engine: str,
    model: str,
    requested_revision: Optional[str] = None,
) -> tuple[str, str]:
    """Return the immutable model identifier and revision used for a run."""
    model_value = str(model or "base")
    if os.path.isdir(model_value):
        return os.path.realpath(model_value), requested_revision or _local_model_revision(model_value)
    if engine in {"faster-whisper", "whisperx"}:
        model_id, pinned = FASTER_WHISPER_MODELS.get(
            model_value,
            (model_value, "operator-supplied"),
        )
        return model_id, str(requested_revision or pinned)
    if engine == "openai-whisper":
        revision = str(requested_revision or "")
        if not revision:
            try:
                import whisper

                url = str(getattr(whisper, "_MODELS", {}).get(model_value, ""))
                for part in url.split("/"):
                    if len(part) == 64 and all(ch in "0123456789abcdef" for ch in part.lower()):
                        revision = part
                        break
            except Exception:  # noqa: BLE001 - identity still records package pin
                revision = ""
        return f"openai/whisper-{model_value}", revision or f"package:{package_version('openai-whisper')}"
    return model_value, str(requested_revision or "unknown")


def _hf_cache_root() -> Path:
    explicit = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")).expanduser()
    return hf_home / "hub"


def cached_hf_revision(model_id: str) -> str:
    """Return the locally resolved Hub commit for an alignment model."""
    if "/" not in model_id:
        return ""
    owner, name = model_id.split("/", 1)
    root = _hf_cache_root() / f"models--{owner.replace('/', '--')}--{name.replace('/', '--')}"
    ref = root / "refs" / "main"
    try:
        value = ref.read_text(encoding="utf-8").strip()
        if value:
            return value
    except OSError:
        pass
    snapshots = root / "snapshots"
    try:
        candidates = sorted(path.name for path in snapshots.iterdir() if path.is_dir())
    except OSError:
        candidates = []
    return candidates[-1] if len(candidates) == 1 else ""


def whisperx_alignment_identity(language: str) -> tuple[str, str]:
    """Resolve WhisperX's language-specific aligner and cached revision."""
    model_id = ""
    try:
        from whisperx import alignment

        code = str(language or "en").lower().split("-", 1)[0]
        model_id = str(
            getattr(alignment, "DEFAULT_ALIGN_MODELS_HF", {}).get(code)
            or getattr(alignment, "DEFAULT_ALIGN_MODELS_TORCH", {}).get(code)
            or ""
        )
    except Exception:  # noqa: BLE001 - optional dependency
        pass
    version = package_version("whisperx")
    if not model_id:
        return "whisperx-language-default", f"package:{version}"
    return model_id, cached_hf_revision(model_id) or f"package:{version}"


@dataclass
class ASRProvenance:
    """Reproducible identity and decision record for one transcript."""

    schema_version: int = ASR_PROVENANCE_SCHEMA_VERSION
    engine: str = "legacy-unknown"
    engine_version: str = "unknown"
    requested_engine: str = "auto"
    model_id: str = "unknown"
    model_revision: str = "unknown"
    alignment_backend: str = "none"
    alignment_version: str = "none"
    alignment_model_id: str = "none"
    alignment_model_revision: str = "none"
    alignment_mode: str = "none"
    requested_language: str = "auto"
    detected_language: str = "unknown"
    language_decision: str = "unknown"
    fallback_reason: str = ""
    device: str = "unknown"
    compute_type: str = "unknown"
    deterministic_options: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def cache_identity(self) -> dict[str, Any]:
        payload = self.to_dict()
        for key in ("detected_language", "language_decision", "fallback_reason", "device", "compute_type"):
            payload.pop(key, None)
        return payload


def build_provenance(
    *,
    engine: str,
    requested_engine: Optional[str],
    model: str,
    model_revision: Optional[str],
    requested_language: Optional[str],
    word_timestamps: bool,
    translate: bool,
    diarize: bool,
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    fallback_reason: str = "",
) -> ASRProvenance:
    model_id, resolved_revision = model_identity(engine, model, model_revision)
    alignment_mode = "none"
    alignment_backend = "none"
    alignment_version = "none"
    alignment_model_id = "none"
    alignment_revision = "none"
    if word_timestamps and engine == "whisperx":
        alignment_mode = "forced-alignment"
        alignment_backend = "whisperx"
        alignment_version = package_version("whisperx")
        alignment_model_id = "language-default-pending-detection"
        alignment_revision = f"package:{alignment_version}"
    elif word_timestamps:
        alignment_mode = "decoder-token-timestamps"
        alignment_backend = engine
        alignment_version = package_version(engine)

    return ASRProvenance(
        engine=engine,
        engine_version=package_version(engine),
        requested_engine=normalize_engine(requested_engine) or "auto",
        model_id=model_id,
        model_revision=resolved_revision,
        alignment_backend=alignment_backend,
        alignment_version=alignment_version,
        alignment_model_id=alignment_model_id,
        alignment_model_revision=alignment_revision,
        alignment_mode=alignment_mode,
        requested_language=str(requested_language or "auto"),
        fallback_reason=str(fallback_reason or ""),
        deterministic_options={
            "word_timestamps": bool(word_timestamps),
            "translate": bool(translate),
            "diarize": bool(diarize),
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
            "vad_filter": engine == "faster-whisper",
        },
    )


def provenance_from_dict(value: Any) -> ASRProvenance:
    """Load current provenance or migrate a pre-provenance artifact."""
    if isinstance(value, ASRProvenance):
        return value
    if not isinstance(value, dict):
        return ASRProvenance(fallback_reason="artifact predates ASR provenance")
    allowed = ASRProvenance.__dataclass_fields__.keys()
    payload = {key: value[key] for key in allowed if key in value}
    if value.get("schema_version") != ASR_PROVENANCE_SCHEMA_VERSION:
        payload["fallback_reason"] = "artifact uses an older ASR provenance schema"
    payload["schema_version"] = ASR_PROVENANCE_SCHEMA_VERSION
    if not isinstance(payload.get("deterministic_options", {}), dict):
        payload["deterministic_options"] = {}
    return ASRProvenance(**payload)


def provenance_to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, ASRProvenance):
        return value.to_dict()
    return provenance_from_dict(value).to_dict()


def apply_language_decision(provenance: ASRProvenance, detected_language: str) -> None:
    provenance.detected_language = str(detected_language or "unknown")
    if provenance.requested_language != "auto":
        provenance.language_decision = "user-override"
    else:
        provenance.language_decision = "auto-detected"
