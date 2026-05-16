"""Codec + hardware capability profile (F106).

Probes the local environment so jobs can decide *before* they kick off
whether they will be able to finish. The information surfaces through
``GET /system/capabilities`` and is intentionally cheap to compute —
the probe must not block panel startup.

What we capture today:

* ``ffmpeg`` — presence, version, decoder/encoder lists pruned to a
  curated allow-list, hardware-accelerated encoder availability
  (``nvenc``, ``qsv``, ``amf``, ``videotoolbox``).
* ``gpu`` — derived from :mod:`opencut.gpu` (``cuda``, ``mps``,
  ``directml``, or ``cpu``) plus the existing ``check_vram`` helper.
* ``ffprobe`` — presence + version.
* ``disk`` — free space on the temp + project directories.
* ``python`` — implementation, version, platform.
* ``advisory`` — string hints generated from the probe results
  (``"no NVENC available — falling back to libx264 software encode"``).

The module is deliberately stdlib-only for the probe paths so it works
inside a fresh ``pip install -e .[standard]`` install.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("opencut")

# Encoders worth surfacing — both software and hardware-accelerated. Any
# decoder/encoder name not on this list is hidden so the response stays
# bounded (FFmpeg often ships 200+ codecs that no editor uses).
_ENCODER_ALLOWLIST = (
    # Common video software encoders
    "libx264", "libx265", "libaom-av1", "libsvtav1", "libvpx", "libvpx-vp9", "prores_ks",
    # Hardware-accelerated video encoders
    "h264_nvenc", "hevc_nvenc", "av1_nvenc",
    "h264_qsv", "hevc_qsv", "av1_qsv",
    "h264_amf", "hevc_amf", "av1_amf",
    "h264_videotoolbox", "hevc_videotoolbox", "prores_videotoolbox",
    # Audio encoders worth advertising
    "aac", "libopus", "libmp3lame", "flac", "pcm_s16le", "pcm_s24le",
)


_DECODER_ALLOWLIST = (
    "h264", "hevc", "vp9", "av1", "prores", "mpeg2video", "mjpeg",
    "aac", "mp3", "opus", "flac", "pcm_s16le", "pcm_s24le",
)


@dataclass
class CapabilityFinding:
    """A single recommendation/advisory derived from the probe."""

    severity: str  # "info" | "warning" | "error"
    rule: str
    message: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class CapabilityProfile:
    """Snapshot of what the local install can do today."""

    generated_at: float = field(default_factory=time.time)
    python: dict = field(default_factory=dict)
    ffmpeg: dict = field(default_factory=dict)
    ffprobe: dict = field(default_factory=dict)
    gpu: dict = field(default_factory=dict)
    disk: dict = field(default_factory=dict)
    findings: List[CapabilityFinding] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "version": 1,
            "generated_at": self.generated_at,
            "python": self.python,
            "ffmpeg": self.ffmpeg,
            "ffprobe": self.ffprobe,
            "gpu": self.gpu,
            "disk": self.disk,
            "findings": [f.as_dict() for f in self.findings],
        }


def _run_capturing(cmd: List[str], timeout: float = 8.0) -> Optional[subprocess.CompletedProcess]:
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.debug("capability_profile: failed to run %s: %s", cmd, exc)
        return None


def _resolve_ffmpeg_bin() -> Optional[str]:
    try:
        from opencut.helpers import get_ffmpeg_path

        return get_ffmpeg_path()
    except Exception:
        return shutil.which("ffmpeg")


def _resolve_ffprobe_bin() -> Optional[str]:
    try:
        from opencut.helpers import get_ffmpeg_path

        ff = get_ffmpeg_path()
        if not ff:
            return shutil.which("ffprobe")
        probe = Path(ff).with_name("ffprobe" + (".exe" if os.name == "nt" else ""))
        return str(probe) if probe.exists() else shutil.which("ffprobe")
    except Exception:
        return shutil.which("ffprobe")


def _extract_version(stdout: str) -> str:
    m = re.search(r"version\s+([0-9][^\s]*)", stdout)
    return m.group(1) if m else ""


def _ffmpeg_codecs(ffmpeg_bin: str, codec_kind: str) -> List[str]:
    """Return a sorted list of encoders/decoders for the configured kind."""
    flag = "-encoders" if codec_kind == "encoders" else "-decoders"
    result = _run_capturing([ffmpeg_bin, "-hide_banner", flag])
    if result is None or result.returncode != 0:
        return []

    allow = _ENCODER_ALLOWLIST if codec_kind == "encoders" else _DECODER_ALLOWLIST
    out: List[str] = []
    for raw in result.stdout.splitlines():
        # Lines look like " V..... libx264              H.264 / AVC / MPEG-4 part 10"
        parts = raw.split()
        if len(parts) < 2:
            continue
        name = parts[1]
        if name in allow and name not in out:
            out.append(name)
    return sorted(out)


def _probe_ffmpeg() -> dict:
    ffmpeg_bin = _resolve_ffmpeg_bin()
    if not ffmpeg_bin:
        return {"available": False, "path": "", "version": "", "encoders": [], "decoders": [], "hwaccel": []}

    version_result = _run_capturing([ffmpeg_bin, "-version"])
    version = ""
    if version_result is not None:
        version = _extract_version(version_result.stdout or "")

    hwaccel: List[str] = []
    hw_result = _run_capturing([ffmpeg_bin, "-hide_banner", "-hwaccels"])
    if hw_result is not None and hw_result.returncode == 0:
        for line in hw_result.stdout.splitlines():
            line = line.strip()
            if not line or line.lower().startswith("hardware acceleration methods"):
                continue
            hwaccel.append(line)

    return {
        "available": True,
        "path": ffmpeg_bin,
        "version": version,
        "encoders": _ffmpeg_codecs(ffmpeg_bin, "encoders"),
        "decoders": _ffmpeg_codecs(ffmpeg_bin, "decoders"),
        "hwaccel": sorted(hwaccel),
    }


def _probe_ffprobe() -> dict:
    bin_path = _resolve_ffprobe_bin()
    if not bin_path:
        return {"available": False, "path": "", "version": ""}
    result = _run_capturing([bin_path, "-version"])
    return {
        "available": True,
        "path": bin_path,
        "version": _extract_version(result.stdout) if result else "",
    }


def _probe_gpu() -> dict:
    try:
        from opencut import gpu as _gpu

        device = _gpu.get_device()
        ok, info = _gpu.check_vram(0)
        return {
            "device": device,
            "vram_total_mb": info.get("total_mb", 0),
            "vram_free_mb": info.get("free_mb", 0),
            "driver": info.get("driver", ""),
            "name": info.get("name", ""),
            "ok": bool(ok),
        }
    except Exception as exc:
        logger.debug("capability_profile: gpu probe failed: %s", exc)
        return {"device": "cpu", "vram_total_mb": 0, "vram_free_mb": 0, "ok": False}


def _probe_disk() -> dict:
    paths = {
        "temp": Path(os.environ.get("TMPDIR") or Path.cwd() / "_tmp"),
        "home_opencut": Path.home() / ".opencut",
        "cwd": Path.cwd(),
    }
    out = {}
    for label, p in paths.items():
        try:
            target = p if p.exists() else p.parent
            usage = shutil.disk_usage(target)
            out[label] = {
                "path": str(target),
                "free_bytes": usage.free,
                "total_bytes": usage.total,
                "free_mb": usage.free // (1024 * 1024),
            }
        except OSError:
            out[label] = {"path": str(p), "free_bytes": 0, "total_bytes": 0, "free_mb": 0}
    return out


def _probe_python() -> dict:
    return {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "executable": sys.executable,
    }


def _derive_findings(profile: CapabilityProfile) -> List[CapabilityFinding]:
    findings: List[CapabilityFinding] = []
    if not profile.ffmpeg.get("available"):
        findings.append(
            CapabilityFinding(
                severity="error",
                rule="ffmpeg_missing",
                message="FFmpeg is required but was not found on PATH or bundled.",
            )
        )
    else:
        hw_encoders = [e for e in profile.ffmpeg.get("encoders", []) if "_" in e and not e.startswith("lib")]
        if not hw_encoders:
            findings.append(
                CapabilityFinding(
                    severity="warning",
                    rule="no_hw_encoder",
                    message=(
                        "No hardware-accelerated encoder detected. Long renders will "
                        "fall back to libx264/libx265 (CPU-bound)."
                    ),
                )
            )

    gpu = profile.gpu
    if gpu.get("device") == "cpu":
        findings.append(
            CapabilityFinding(
                severity="info",
                rule="cpu_only",
                message="No CUDA/MPS GPU detected. AI extras requiring a GPU will be unavailable.",
            )
        )
    elif (gpu.get("vram_total_mb") or 0) and gpu["vram_total_mb"] < 6 * 1024:
        findings.append(
            CapabilityFinding(
                severity="warning",
                rule="low_vram",
                message=(
                    f"GPU VRAM is {gpu['vram_total_mb']} MB — many AI backends require >= 8 GB."
                ),
            )
        )

    for label, info in profile.disk.items():
        if info.get("free_mb", 0) and info["free_mb"] < 2048:
            findings.append(
                CapabilityFinding(
                    severity="warning",
                    rule="low_disk",
                    message=f"{label} ({info['path']}) has {info['free_mb']} MB free; renders may fail.",
                )
            )

    if not profile.ffprobe.get("available"):
        findings.append(
            CapabilityFinding(
                severity="warning",
                rule="ffprobe_missing",
                message="ffprobe not detected; metadata extraction will be limited.",
            )
        )

    return findings


def build_profile() -> dict:
    """Run all probes and return the JSON payload."""
    profile = CapabilityProfile()
    profile.python = _probe_python()
    profile.ffmpeg = _probe_ffmpeg()
    profile.ffprobe = _probe_ffprobe()
    profile.gpu = _probe_gpu()
    profile.disk = _probe_disk()
    profile.findings = _derive_findings(profile)
    return profile.as_dict()
