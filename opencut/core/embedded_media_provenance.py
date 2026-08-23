"""Provenance and runtime policy for FFmpeg copies embedded in Python wheels.

OpenCut ships an independently attested FFmpeg command-line build, but two
Python dependencies can carry additional decoder copies: OpenCV and PyAV.
Package versions alone are not enough evidence because wheel recipes differ by
platform. This module records the loaded FFmpeg library ABI, hashes native
payloads, and fails closed against the FFmpeg 8.1.2 floor for CVE-2026-8461.

The reviewed temporary OpenCV lane is 4.14.0.94. Its Linux and macOS wheels
must report the FFmpeg 8.1.2 ABI. The Windows wheel still exposes an older
prebuilt videoio plugin ABI, so OpenCut disables that backend before importing
``cv2`` and removes ``opencv_videoio_ffmpeg*.dll`` from frozen artifacts.
Media Foundation remains available for OpenCV video I/O on Windows.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

SECURITY_CVE = "CVE-2026-8461"
FIXED_FFMPEG_VERSION = "8.1.2"
FIXED_LIBRARY_FLOORS: dict[str, tuple[int, int, int]] = {
    "avcodec": (62, 28, 102),
    "avformat": (62, 12, 102),
    "avutil": (60, 26, 102),
}

OPENCV_FFMPEG_PRIORITY_ENV = "OPENCV_VIDEOIO_PRIORITY_FFMPEG"
OPENCV_DISTRIBUTIONS = (
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python",
    "opencv-contrib-python-headless",
)
REVIEWED_OPENCV_VERSIONS = frozenset({"4.14.0.94"})
REVIEWED_PYAV_VERSIONS = frozenset({"18.0.0", "18.1.0"})

EVIDENCE = {
    "advisory": "https://nvd.nist.gov/vuln/detail/CVE-2026-8461",
    "ffmpeg_tag": "https://github.com/FFmpeg/FFmpeg/releases/tag/n8.1.2",
    "opencv_release": "https://github.com/opencv/opencv-python/releases/tag/94",
    "pyav_18_0": "https://github.com/PyAV-Org/PyAV/releases/tag/v18.0.0",
    "pyav_18_1": "https://github.com/PyAV-Org/PyAV/releases/tag/v18.1.0",
}

_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
_OPENCV_LIBRARY_RE = re.compile(
    r"^\s*(avcodec|avformat|avutil):\s+YES\s+\((\d+)\.(\d+)\.(\d+)\)\s*$",
    re.MULTILINE,
)
_OPENCV_FFMPEG_RE = re.compile(
    r"^\s*FFMPEG:\s+(YES|NO)(?:\s+\(([^)]*)\))?\s*$",
    re.MULTILINE,
)
_NATIVE_SUFFIXES = (".dll", ".dylib", ".pyd", ".so")
_AUTO_LOADED_MODULE = object()


class EmbeddedMediaProvenanceError(RuntimeError):
    """Raised when an installed decoder cannot meet the security contract."""


def _version_tuple(value: Sequence[int] | str | None) -> tuple[int, int, int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        match = _VERSION_RE.match(value.strip())
        if not match:
            return None
        return tuple(int(part) for part in match.groups())
    try:
        parts = tuple(int(part) for part in value)
    except (TypeError, ValueError):
        return None
    if len(parts) < 3:
        return None
    return parts[:3]


def grade_library_versions(versions: Mapping[str, Sequence[int] | str]) -> dict:
    """Grade libavcodec/libavformat/libavutil against FFmpeg 8.1.2."""
    normalized: dict[str, tuple[int, int, int]] = {}
    for raw_name, raw_version in versions.items():
        name = str(raw_name).lower().removeprefix("lib")
        parsed = _version_tuple(raw_version)
        if name in FIXED_LIBRARY_FLOORS and parsed is not None:
            normalized[name] = parsed

    missing = sorted(set(FIXED_LIBRARY_FLOORS) - set(normalized))
    below_floor = {
        name: {
            "detected": normalized[name],
            "required": required,
        }
        for name, required in FIXED_LIBRARY_FLOORS.items()
        if name in normalized and normalized[name] < required
    }
    return {
        "ok": not missing and not below_floor,
        "fixed_ffmpeg": FIXED_FFMPEG_VERSION,
        "versions": normalized,
        "missing": missing,
        "below_floor": below_floor,
    }


def parse_opencv_build_information(build_information: str) -> dict:
    """Extract OpenCV's linked FFmpeg ABI without decoding any media."""
    ffmpeg_match = _OPENCV_FFMPEG_RE.search(build_information or "")
    libraries = {
        match.group(1): tuple(int(match.group(index)) for index in range(2, 5))
        for match in _OPENCV_LIBRARY_RE.finditer(build_information or "")
    }
    return {
        "enabled": bool(ffmpeg_match and ffmpeg_match.group(1) == "YES"),
        "build": (ffmpeg_match.group(2) or "").strip() if ffmpeg_match else "",
        "libraries": libraries,
    }


def _installed_decoder_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in (*OPENCV_DISTRIBUTIONS, "av"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def _opencv_installations(versions: Mapping[str, str]) -> list[tuple[str, str]]:
    return [(name, versions[name]) for name in OPENCV_DISTRIBUTIONS if name in versions]


def _loaded_opencv_ffmpeg_enabled(cv2_module) -> bool:
    try:
        backends = cv2_module.videoio_registry.getBackends()
        return cv2_module.CAP_FFMPEG in backends
    except (AttributeError, RuntimeError, TypeError):
        return True


def install_runtime_guard(
    *,
    platform_name: str | None = None,
    installed_versions: Mapping[str, str] | None = None,
    loaded_cv2=_AUTO_LOADED_MODULE,
) -> dict:
    """Disable or reject unreviewed embedded decoders before feature imports.

    OpenCV reads backend priority variables when its video registry first
    initializes. The package root calls this function before importing feature
    modules, preventing ``VideoCapture`` from silently selecting an unverified
    FFmpeg plugin. PyAV has no equivalent disable switch, so an unreviewed PyAV
    distribution is rejected at startup.
    """
    platform_name = platform_name or sys.platform
    versions = dict(installed_versions or _installed_decoder_versions())
    opencv = _opencv_installations(versions)
    pyav_version = versions.get("av")

    if pyav_version and pyav_version not in REVIEWED_PYAV_VERSIONS:
        raise EmbeddedMediaProvenanceError(
            f"OpenCut found unreviewed PyAV {pyav_version}. Install av==18.1.0 "
            f"before processing media ({SECURITY_CVE})."
        )

    should_disable = bool(
        opencv
        and (
            platform_name == "win32"
            or len(opencv) != 1
            or opencv[0][1] not in REVIEWED_OPENCV_VERSIONS
        )
    )
    loaded_backend_enabled = False
    if should_disable:
        os.environ[OPENCV_FFMPEG_PRIORITY_ENV] = "0"
        if loaded_cv2 is _AUTO_LOADED_MODULE:
            loaded_cv2 = sys.modules.get("cv2")
        loaded_backend_enabled = bool(
            loaded_cv2 is not None and _loaded_opencv_ffmpeg_enabled(loaded_cv2)
        )

    return {
        "opencv": [{"distribution": name, "version": version} for name, version in opencv],
        "opencv_ffmpeg": "disabled" if should_disable else ("candidate" if opencv else "absent"),
        "opencv_ffmpeg_was_already_loaded": loaded_backend_enabled,
        "pyav": pyav_version or "",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_native_name(name: str) -> bool:
    lower = name.lower()
    return any(suffix in lower for suffix in _NATIVE_SUFFIXES)


def _distribution_native_files(distribution_name: str, *, include_hashes: bool) -> list[dict]:
    try:
        distribution = metadata.distribution(distribution_name)
    except metadata.PackageNotFoundError:
        return []

    records: list[dict] = []
    for item in distribution.files or ():
        relative = Path(str(item).replace("\\", "/"))
        lower = relative.as_posix().lower()
        filename = relative.name.lower()
        if distribution_name == "av":
            relevant = (
                lower.startswith("av.libs/")
                and any(name in filename for name in ("avcodec", "avformat", "avutil"))
            ) or (lower.startswith("av/_core") and _is_native_name(filename))
        else:
            relevant = "opencv_videoio_ffmpeg" in filename or (
                lower.startswith("cv2/") and filename.startswith("cv2.") and _is_native_name(filename)
            )
        if not relevant:
            continue
        absolute = Path(distribution.locate_file(item))
        if not absolute.is_file():
            continue
        record = {
            "path": relative.as_posix(),
            "size": absolute.stat().st_size,
        }
        if include_hashes:
            record["sha256"] = _sha256(absolute)
        records.append(record)
    return sorted(records, key=lambda item: item["path"])


def _opencv_backend_enabled(cv2_module) -> bool:
    return _loaded_opencv_ffmpeg_enabled(cv2_module)


def _inspect_opencv(*, include_hashes: bool) -> dict:
    versions = _installed_decoder_versions()
    installations = _opencv_installations(versions)
    if not installations:
        return {"installed": False, "ok": True, "status": "absent", "native_files": []}
    if len(installations) != 1:
        return {
            "installed": True,
            "ok": False,
            "status": "blocked",
            "installations": [
                {"distribution": name, "version": version} for name, version in installations
            ],
            "error": "multiple OpenCV distributions export cv2",
            "native_files": [],
        }

    distribution_name, distribution_version = installations[0]
    try:
        cv2 = importlib.import_module("cv2")
        parsed = parse_opencv_build_information(cv2.getBuildInformation())
    except Exception as exc:
        return {
            "installed": True,
            "distribution": distribution_name,
            "version": distribution_version,
            "ok": False,
            "status": "blocked",
            "error": f"could not inspect cv2: {exc}",
            "native_files": _distribution_native_files(
                distribution_name, include_hashes=include_hashes
            ),
        }

    floor = grade_library_versions(parsed["libraries"])
    backend_enabled = _opencv_backend_enabled(cv2) if parsed["enabled"] else False
    if not parsed["enabled"]:
        status = "absent"
        ok = True
        error = ""
    elif not backend_enabled and os.environ.get(OPENCV_FFMPEG_PRIORITY_ENV) == "0":
        status = "disabled"
        ok = True
        error = ""
    elif floor["ok"]:
        status = "verified"
        ok = True
        error = ""
    else:
        status = "blocked"
        ok = False
        error = (
            f"OpenCV FFmpeg does not prove the {FIXED_FFMPEG_VERSION} floor for {SECURITY_CVE}"
        )

    return {
        "installed": True,
        "distribution": distribution_name,
        "version": distribution_version,
        "module_version": str(getattr(cv2, "__version__", "")),
        "reviewed_distribution": distribution_version in REVIEWED_OPENCV_VERSIONS,
        "ffmpeg_enabled_at_build": parsed["enabled"],
        "ffmpeg_backend_enabled": backend_enabled,
        "ffmpeg_build": parsed["build"],
        "libraries": parsed["libraries"],
        "security": floor,
        "status": status,
        "ok": ok,
        "error": error,
        "native_files": _distribution_native_files(
            distribution_name, include_hashes=include_hashes
        ),
    }


def _inspect_pyav(*, include_hashes: bool) -> dict:
    try:
        distribution_version = metadata.version("av")
    except metadata.PackageNotFoundError:
        return {"installed": False, "ok": True, "status": "absent", "native_files": []}

    try:
        av = importlib.import_module("av")
        library_versions = {
            str(name).lower().removeprefix("lib"): tuple(int(part) for part in version[:3])
            for name, version in dict(av.library_versions).items()
            if str(name).lower().removeprefix("lib") in FIXED_LIBRARY_FLOORS
        }
    except Exception as exc:
        return {
            "installed": True,
            "distribution": "av",
            "version": distribution_version,
            "ok": False,
            "status": "blocked",
            "error": f"could not inspect PyAV: {exc}",
            "native_files": _distribution_native_files("av", include_hashes=include_hashes),
        }

    floor = grade_library_versions(library_versions)
    reviewed = distribution_version in REVIEWED_PYAV_VERSIONS
    ok = floor["ok"] and reviewed
    error = ""
    if not reviewed:
        error = f"PyAV {distribution_version} has not been reviewed for {SECURITY_CVE}"
    elif not floor["ok"]:
        error = f"PyAV FFmpeg does not prove the {FIXED_FFMPEG_VERSION} security floor"
    return {
        "installed": True,
        "distribution": "av",
        "version": distribution_version,
        "module_version": str(getattr(av, "__version__", "")),
        "reviewed_distribution": reviewed,
        "libraries": library_versions,
        "security": floor,
        "status": "verified" if ok else "blocked",
        "ok": ok,
        "error": error,
        "native_files": _distribution_native_files("av", include_hashes=include_hashes),
    }


def inspect_runtime(*, required: bool = False, include_hashes: bool = False) -> dict:
    """Inventory installed OpenCV and PyAV decoder copies."""
    opencv = _inspect_opencv(include_hashes=include_hashes)
    pyav = _inspect_pyav(include_hashes=include_hashes)
    errors: list[str] = []
    for label, record in (("OpenCV", opencv), ("PyAV", pyav)):
        if required and not record.get("installed"):
            errors.append(f"{label} is missing from the release environment")
        elif record.get("installed") and not record.get("ok"):
            errors.append(record.get("error") or f"{label} decoder provenance failed")
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "security": {
            "cve": SECURITY_CVE,
            "fixed_ffmpeg": FIXED_FFMPEG_VERSION,
            "library_floors": FIXED_LIBRARY_FLOORS,
            "evidence": EVIDENCE,
        },
        "runtime_policy": {
            "opencv_ffmpeg_priority": os.environ.get(OPENCV_FFMPEG_PRIORITY_ENV, ""),
        },
        "opencv": opencv,
        "pyav": pyav,
        "errors": errors,
        "ok": not errors,
    }


def _artifact_provider(path: Path) -> str | None:
    lower = path.as_posix().lower()
    filename = path.name.lower()
    if "opencv_videoio_ffmpeg" in filename or "opencv_python.libs" in lower:
        return "opencv"
    if any(name in filename for name in ("avcodec", "avformat", "avutil")):
        if "av.libs" in lower:
            return "pyav"
        if "/cv2/" in f"/{lower}" or "opencv" in lower:
            return "opencv"
        return "unknown"
    return None


def scan_artifact_decoder_files(
    artifact_paths: Iterable[Path], *, include_hashes: bool = True
) -> list[dict]:
    """Hash native decoder files found in assembled artifact paths."""
    records: list[dict] = []
    seen: set[Path] = set()
    for raw_root in artifact_paths:
        root = Path(raw_root)
        candidates = [root] if root.is_file() else root.rglob("*") if root.is_dir() else []
        for candidate in candidates:
            if not candidate.is_file():
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            provider = _artifact_provider(candidate)
            if provider is None:
                continue
            seen.add(resolved)
            relative = candidate.name if root.is_file() else (Path(root.name) / candidate.relative_to(root)).as_posix()
            record = {
                "provider": provider,
                "path": str(relative).replace("\\", "/"),
                "size": candidate.stat().st_size,
            }
            if include_hashes:
                record["sha256"] = _sha256(candidate)
            records.append(record)
    return sorted(records, key=lambda item: (item["provider"], item["path"]))


def build_release_inventory(
    *,
    lane: str,
    artifact_paths: Iterable[Path],
    runtime_inventory: dict | None = None,
) -> dict:
    """Combine runtime ABI evidence with the files in an assembled artifact."""
    if lane not in {"windows", "linux", "macos"}:
        raise ValueError(f"unsupported release lane: {lane}")
    runtime = runtime_inventory or inspect_runtime(required=True, include_hashes=True)
    artifact_files = scan_artifact_decoder_files(artifact_paths, include_hashes=True)
    errors = list(runtime.get("errors") or [])
    opencv = runtime.get("opencv") or {}

    if lane == "windows":
        if opencv.get("installed") and opencv.get("status") not in {"disabled", "absent"}:
            errors.append("Windows artifacts require the OpenCV FFmpeg backend to be disabled")
        for record in artifact_files:
            if record["provider"] == "opencv":
                errors.append(
                    f"Windows artifact contains unverified OpenCV FFmpeg file: {record['path']}"
                )
    elif opencv.get("installed") and opencv.get("status") != "verified":
        errors.append(f"{lane} artifacts require a verified OpenCV FFmpeg 8.1.2 ABI")

    for record in artifact_files:
        if record["provider"] == "unknown":
            errors.append(f"artifact contains an unattributed FFmpeg library: {record['path']}")

    deduplicated_errors = list(dict.fromkeys(errors))
    return {
        "schema_version": 1,
        "lane": lane,
        "security": runtime.get("security") or {},
        "runtime_policy": runtime.get("runtime_policy") or {},
        "providers": {
            "opencv": runtime.get("opencv") or {},
            "pyav": runtime.get("pyav") or {},
        },
        "artifact_files": artifact_files,
        "errors": deduplicated_errors,
        "ok": not deduplicated_errors,
    }


def write_manifest(payload: dict, destination: Path) -> Path:
    """Write a stable LF-terminated JSON provenance manifest."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes((json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return destination
