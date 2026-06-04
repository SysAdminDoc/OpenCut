"""Install-command hints for optional OpenCut dependencies."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Optional


_SEPARATOR_RE = re.compile(r"[^a-z0-9]+")


def _norm(value: str) -> str:
    return _SEPARATOR_RE.sub("", value.lower())


def _hint(
    *,
    extra: str,
    packages: Iterable[str],
    aliases: Iterable[str],
    gpu: bool = False,
    vram_mb: int = 0,
) -> Dict[str, Any]:
    return {
        "extra": extra,
        "packages": tuple(packages),
        "aliases": tuple(aliases),
        "gpu": gpu,
        "vram_mb": vram_mb,
    }


DEPENDENCY_HINTS: Dict[str, Dict[str, Any]] = {
    "silero-vad": _hint(
        extra="depth",
        packages=("torch",),
        aliases=("silero", "vad", "voice activity detection"),
        gpu=False,
    ),
    "captions": _hint(
        extra="captions",
        packages=("faster-whisper",),
        aliases=("caption", "captions", "whisper", "faster-whisper", "openai-whisper"),
    ),
    "whisperx": _hint(
        extra="captions-whisperx",
        packages=("whisperx",),
        aliases=("whisperx", "pyannote"),
        gpu=True,
    ),
    "demucs": _hint(
        extra="audio",
        packages=("demucs",),
        aliases=("demucs", "stem separation", "stems"),
        gpu=True,
    ),
    "deepfilter": _hint(
        extra="audio",
        packages=("deepfilternet",),
        aliases=("deepfilter", "deepfilternet", "df", "noise reduction"),
    ),
    "depth": _hint(
        extra="depth",
        packages=("torch", "torchvision", "transformers"),
        aliases=("depth", "depth anything", "torchvision", "transformers"),
        gpu=True,
        vram_mb=4096,
    ),
    "rvm": _hint(
        extra="depth",
        packages=("torch", "torchvision"),
        aliases=("rvm", "robust video matting", "matting"),
        gpu=True,
        vram_mb=4096,
    ),
    "scene-detect": _hint(
        extra="video",
        packages=("scenedetect[opencv]",),
        aliases=("scene-detect", "scenedetect", "pyscenedetect", "scene detect"),
    ),
    "neural-interp": _hint(
        extra="video",
        packages=("rife-ncnn-vulkan", "practical-rife"),
        aliases=("neural interpolation", "neural-interp", "rife", "frame interpolation"),
        gpu=True,
    ),
    "musicgen": _hint(
        extra="music",
        packages=("audiocraft",),
        aliases=("musicgen", "audiocraft", "audiogen"),
        gpu=True,
    ),
    "f5-tts": _hint(
        extra="tts",
        packages=("f5-tts",),
        aliases=("f5", "f5-tts", "f5_tts", "voice clone"),
        gpu=True,
    ),
    "chatterbox": _hint(
        extra="tts",
        packages=("chatterbox-tts",),
        aliases=("chatterbox", "chatterbox-tts"),
        gpu=True,
    ),
    "gfpgan": _hint(
        extra="ai",
        packages=("gfpgan",),
        aliases=("gfpgan", "face restoration", "skin retouch"),
        gpu=True,
    ),
    "rembg": _hint(
        extra="ai",
        packages=("rembg",),
        aliases=("rembg", "background removal"),
    ),
    "realesrgan": _hint(
        extra="ai",
        packages=("realesrgan",),
        aliases=("realesrgan", "real-esrgan", "upscale"),
        gpu=True,
    ),
    "edge-tts": _hint(
        extra="tts",
        packages=("edge-tts",),
        aliases=("edge tts", "edge-tts"),
    ),
    "kokoro": _hint(
        extra="tts",
        packages=("kokoro",),
        aliases=("kokoro",),
    ),
}


_ALIAS_INDEX = {
    _norm(alias): key
    for key, hint in DEPENDENCY_HINTS.items()
    for alias in (key, *hint["aliases"], *hint["packages"])
}


def lookup_hint(*values: str) -> Optional[Dict[str, Any]]:
    """Return the first install hint matching any supplied text."""
    haystacks = [_norm(value) for value in values if value]
    for haystack in haystacks:
        for alias, key in _ALIAS_INDEX.items():
            if alias and alias in haystack:
                return DEPENDENCY_HINTS[key]
    return None


def build_install_suggestion(
    name: str,
    *,
    extra: Optional[str] = None,
    gpu: bool = False,
    vram_mb: int = 0,
    context: str = "",
    message: str = "",
) -> str:
    """Build an actionable install suggestion for an optional dependency."""
    hint = lookup_hint(name, context, message) or {}
    selected_extra = extra or hint.get("extra")
    packages = tuple(hint.get("packages") or ())
    needs_gpu = bool(gpu or hint.get("gpu"))
    memory_mb = int(vram_mb or hint.get("vram_mb") or 0)

    parts = []
    if selected_extra:
        parts.append(f"Install with: pip install 'opencut[{selected_extra}]'")
    elif packages:
        parts.append(f"Install with: pip install {' '.join(packages)}")
    else:
        parts.append("Install it from the Settings tab under Dependencies.")

    if packages and selected_extra:
        parts.append(f"Missing package hint: {' '.join(packages)}.")
    if needs_gpu:
        parts.append("GPU-recommended; install CUDA-enabled torch first.")
    if memory_mb > 0:
        parts.append(f"Recommended VRAM: {memory_mb // 1024 if memory_mb % 1024 == 0 else round(memory_mb / 1024, 1)} GB.")
    return " ".join(parts)


def suggestion_for_exception(exc: BaseException, *, context: str = "") -> str:
    """Return an install suggestion for a dependency-shaped exception."""
    message = str(exc)
    if not _looks_like_missing_dependency(exc, message):
        return ""
    return build_install_suggestion(message, context=context, message=message)


def _looks_like_missing_dependency(exc: BaseException, message: str) -> bool:
    lower = message.lower()
    return (
        isinstance(exc, ImportError)
        or "no module named" in lower
        or "not installed" in lower
        or "missing package" in lower
        or "dependencies not installed" in lower
    )
