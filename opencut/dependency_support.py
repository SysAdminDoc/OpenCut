"""Canonical Python, platform, and optional-dependency support contract."""

from __future__ import annotations

import platform
import sys
from typing import Iterable, Mapping, Sequence

PYTHON_MIN = (3, 11)
PYTHON_MAX = (3, 14)
PYTHON_REQUIRES = ">=3.11,<3.15"
PYTHON_VERSIONS = ("3.11", "3.12", "3.13", "3.14")
PLATFORMS = ("win32", "linux", "darwin")

# Every PEP 621 extra that OpenCut advertises. The resolver matrix must cover
# exactly this set so adding an extra without a CI lane fails tests.
EXTRA_SUPPORT: Mapping[str, Mapping[str, tuple[str, ...]]] = {
    "standard": {"python": PYTHON_VERSIONS, "platforms": PLATFORMS},
    "captions": {"python": PYTHON_VERSIONS, "platforms": PLATFORMS},
    "audio": {"python": PYTHON_VERSIONS, "platforms": PLATFORMS},
    "video": {"python": PYTHON_VERSIONS, "platforms": PLATFORMS},
    "ai": {"python": PYTHON_VERSIONS, "platforms": PLATFORMS},
    "ai-gpu": {"python": PYTHON_VERSIONS, "platforms": ("win32", "linux")},
    "diarize": {"python": PYTHON_VERSIONS, "platforms": PLATFORMS},
    "nemo-asr": {"python": PYTHON_VERSIONS, "platforms": ("linux",)},
    "auto-edit": {"python": PYTHON_VERSIONS, "platforms": PLATFORMS},
    "scene-ml": {"python": PYTHON_VERSIONS, "platforms": PLATFORMS},
    "reframe": {"python": PYTHON_VERSIONS, "platforms": PLATFORMS},
    "mcp": {"python": PYTHON_VERSIONS, "platforms": PLATFORMS},
    "otio": {"python": PYTHON_VERSIONS, "platforms": PLATFORMS},
    "tts": {"python": PYTHON_VERSIONS, "platforms": PLATFORMS},
    "depth": {"python": PYTHON_VERSIONS, "platforms": PLATFORMS},
    "torch-stack": {"python": PYTHON_VERSIONS, "platforms": PLATFORMS},
    "all": {"python": PYTHON_VERSIONS, "platforms": PLATFORMS},
    "dev": {"python": PYTHON_VERSIONS, "platforms": PLATFORMS},
}

EXTRA_UNSUPPORTED_LANES: Mapping[tuple[str, str, str], str] = {
    **{
        (extra, "3.14", "darwin"): (
            f"opencut-ppro[{extra}] is not supported on macOS with Python 3.14: "
            "faster-whisper requires an onnxruntime cp314 wheel, and the published "
            "macOS wheel set does not cover OpenCut's generic macOS target. Use "
            "Python 3.11-3.13 or install a narrower extra."
        )
        for extra in ("standard", "captions")
    },
    **{
        (extra, python_version, "darwin"): (
            f"opencut-ppro[{extra}] is not supported on OpenCut's generic macOS "
            "target: onnxruntime >=1.26 publishes macOS 14 Apple Silicon wheels "
            "but no wheel matching the generic macOS resolver lane. Install a "
            "narrower extra."
        )
        for extra in ("ai", "all")
        for python_version in PYTHON_VERSIONS
    },
}

UNSUPPORTED_DEPENDENCIES: Mapping[str, str] = {
    "whisperx": (
        "WhisperX 3.8.x requires torchvision <0.24, but OpenCut requires "
        "torchvision >=0.25 with Torch >=2.10 for CVE-2026-24747. No safe "
        "OpenCut install lane is available until WhisperX updates its Torch stack."
    ),
    "audiocraft": (
        "AudioCraft 1.x pins the obsolete Torch 2.1 stack and is not offered "
        "by OpenCut's supported dependency matrix."
    ),
    "resemble-enhance": (
        "Resemble Enhance 0.0.1 pins Torch 2.1.1 and deepspeed 0.12.4 and is "
        "not offered by OpenCut's supported dependency matrix."
    ),
}

DEPENDENCY_EXTRAS: Mapping[str, str] = {
    "faster-whisper": "captions",
    "demucs": "audio",
    "pedalboard": "audio",
    "deepfilternet": "audio",
    "noisereduce": "audio",
    "librosa": "audio",
    "opencv": "video",
    "pillow": "video",
    "numpy": "video",
    "rembg": "ai",
    "realesrgan": "ai",
    "gfpgan": "ai",
    "insightface": "ai",
    "edge-tts": "tts",
    "scenedetect": "video",
    "pyannote.audio": "diarize",
    "nemo-toolkit": "nemo-asr",
    "nemo_toolkit": "nemo-asr",
    "mediapipe": "reframe",
    "torch": "torch-stack",
    "onnxruntime": "ai",
}


def source_extra_install_command(extra: str) -> str:
    """Return the supported install command for a repository checkout.

    OpenCut is not currently published on PyPI.  Keeping this helper beside
    the extra-support contract prevents APIs and the Dependency Dashboard from
    suggesting a registry command that cannot resolve.
    """
    cleaned = extra.strip()
    suffix = f"[{cleaned}]" if cleaned else ""
    return f'python -m pip install -e ".{suffix}"'


def python_version_text(version: Sequence[int] | None = None) -> str:
    selected = sys.version_info if version is None else version
    return f"{int(selected[0])}.{int(selected[1])}"


def python_supported(version: Sequence[int] | None = None) -> bool:
    selected = sys.version_info if version is None else version
    minor = (int(selected[0]), int(selected[1]))
    return PYTHON_MIN <= minor <= PYTHON_MAX


def normalise_platform(value: str | None = None) -> str:
    selected = (value or sys.platform).lower()
    if selected.startswith("win"):
        return "win32"
    if selected.startswith("linux"):
        return "linux"
    if selected.startswith("darwin") or selected.startswith("mac"):
        return "darwin"
    return selected


def extra_support(
    extra: str,
    *,
    version: Sequence[int] | None = None,
    platform_name: str | None = None,
) -> dict:
    contract = EXTRA_SUPPORT.get(extra)
    if contract is None:
        return {"supported": False, "reason": f"Unknown OpenCut extra: {extra}", "extra": extra}
    py_version = python_version_text(version)
    target_platform = normalise_platform(platform_name)
    if py_version not in contract["python"]:
        return {
            "supported": False,
            "reason": f"opencut-ppro[{extra}] supports Python 3.11-3.14; detected {py_version}.",
            "extra": extra,
        }
    if target_platform not in contract["platforms"]:
        names = ", ".join(contract["platforms"])
        return {
            "supported": False,
            "reason": f"opencut-ppro[{extra}] supports {names}; detected {target_platform}.",
            "extra": extra,
        }
    unsupported_reason = EXTRA_UNSUPPORTED_LANES.get(
        (extra, py_version, target_platform)
    )
    if unsupported_reason:
        return {
            "supported": False,
            "reason": unsupported_reason,
            "extra": extra,
        }
    return {
        "supported": True,
        "reason": "",
        "extra": extra,
        "install_hint": source_extra_install_command(extra),
    }


def dependency_support(name: str) -> dict:
    key = name.strip().lower()
    for dependency, unsupported_reason in UNSUPPORTED_DEPENDENCIES.items():
        if dependency in key:
            return {"supported": False, "reason": unsupported_reason, "install_hint": ""}
    extra = DEPENDENCY_EXTRAS.get(key)
    if not extra:
        return {"supported": True, "reason": "", "install_hint": ""}
    return extra_support(extra)


def runtime_contract() -> dict:
    return {
        "python": {
            "detected": platform.python_version(),
            "requires": PYTHON_REQUIRES,
            "supported": python_supported(),
            "versions": list(PYTHON_VERSIONS),
        },
        "platform": normalise_platform(),
        "extras": {
            name: extra_support(name)
            for name in sorted(EXTRA_SUPPORT)
        },
        "unsupported_dependencies": dict(UNSUPPORTED_DEPENDENCIES),
    }


def assert_extra_names(extra_names: Iterable[str]) -> None:
    declared = set(extra_names)
    expected = set(EXTRA_SUPPORT)
    if declared != expected:
        missing = sorted(expected - declared)
        unexpected = sorted(declared - expected)
        raise ValueError(f"dependency support drift: missing={missing}, unexpected={unexpected}")
