"""
HLS / DASH packaging via Shaka Packager.

Shaka Packager (https://github.com/shaka-project/shaka-packager, MIT)
is Google's industry-standard packager for adaptive streaming: HLS
(.m3u8), DASH (.mpd), CMAF low-latency, and DRM (CENC / Widevine /
PlayReady / FairPlay).  Distinct from
``opencut.core.streaming_package``, which renders FFmpeg-native HLS
and is fine for non-DRM delivery — Shaka is the right tool when
you need:

- ISO BMFF segmentation (.m4s) that passes every major CDN's probe,
- Common Encryption (CENC) with DRM key + IV wiring,
- Low-latency HLS (LL-HLS) partial-segment support,
- Mixed-codec manifests (AV1 + HEVC + H.264 renditions in one MPD).

Design
------
- Shaka is a **standalone binary**, not a pip package.  We resolve
  ``packager`` / ``packager.exe`` from PATH at call time and raise
  ``RuntimeError`` with install guidance when missing.
- Inputs are the per-rendition encoded files — we don't re-encode
  here. Pair with ``core/streaming_package.create_hls_package``
  for a quick render or with ``core/av1_export`` /
  ``core/svtav1_psy`` for quality-first renditions.
- Outputs go under a user-specified directory, with ``manifest.m3u8``
  / ``manifest.mpd`` as the top-level playlist files.  The directory
  is created if missing.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess as _sp
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("opencut")


PACKAGER_BIN_CANDIDATES = ("packager", "packager.exe", "shaka-packager")

# The four major DRM schemes Shaka writes into a CENC init segment.
# Users must supply their own key-server URLs — we don't bundle one.
DRM_SCHEMES = ("widevine", "playready", "fairplay", "common_system")

# Output protocols
OUTPUT_PROTOCOLS = ("hls", "dash", "both")


@dataclass
class PackagerRendition:
    """One encoded input rendition fed to Shaka Packager."""
    input_path: str
    kind: str = "video"                # "video" | "audio" | "text"
    language: str = "und"
    stream_label: str = ""             # e.g. "1080p_h264" — defaults to basename


@dataclass
class PackagerResult:
    """Structured return from a packaging run."""
    output_dir: str = ""
    hls_manifest: Optional[str] = None
    dash_manifest: Optional[str] = None
    protocol: str = "hls"
    rendition_count: int = 0
    drm_scheme: Optional[str] = None
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

_AVAILABILITY_CACHE: Dict[str, Optional[str]] = {"bin": None, "checked": None}


def check_shaka_available() -> bool:
    """Cached lookup — returns True when a packager binary is on PATH."""
    if _AVAILABILITY_CACHE["checked"]:
        return bool(_AVAILABILITY_CACHE["bin"])
    resolved = ""
    for name in PACKAGER_BIN_CANDIDATES:
        path = shutil.which(name)
        if path:
            resolved = path
            break
    _AVAILABILITY_CACHE["bin"] = resolved
    _AVAILABILITY_CACHE["checked"] = True
    return bool(resolved)


def resolve_packager_bin() -> str:
    """Return the resolved Shaka binary path or raise ``RuntimeError``."""
    if not check_shaka_available():
        raise RuntimeError(
            "Shaka Packager not installed. Download the `packager` binary "
            "from https://github.com/shaka-project/shaka-packager/releases "
            "and place it on PATH."
        )
    return _AVAILABILITY_CACHE["bin"] or ""


def version() -> str:
    """Return Shaka's version string or an empty string when missing."""
    if not check_shaka_available():
        return ""
    try:
        out = _sp.run(
            [resolve_packager_bin(), "--version"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        return (out.stdout or out.stderr or "").strip().splitlines()[0][:120]
    except Exception:  # noqa: BLE001
        return ""


# ---------------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------------

def _stream_descriptor(r: PackagerRendition, out_dir: str) -> str:
    """Build a single Shaka ``stream_descriptor`` string for one rendition.

    Format (per Shaka docs): ``input=<path>,stream=<video|audio|text>,
    output=<segment_template_or_mp4>,segment_template=...,language=...,playlist_name=...``
    """
    base = r.stream_label or os.path.splitext(os.path.basename(r.input_path))[0]
    base = "".join(c if c.isalnum() or c in "-_" else "_" for c in base)[:60]

    if r.kind == "video":
        init_seg = os.path.join(out_dir, f"{base}_init.mp4")
        seg_tpl = os.path.join(out_dir, f"{base}_$Number$.m4s")
        playlist = f"{base}.m3u8"
        descriptor = (
            f"input={r.input_path},"
            "stream=video,"
            f"init_segment={init_seg},"
            f"segment_template={seg_tpl},"
            f"playlist_name={playlist}"
        )
    elif r.kind == "audio":
        init_seg = os.path.join(out_dir, f"{base}_init.mp4")
        seg_tpl = os.path.join(out_dir, f"{base}_$Number$.m4s")
        playlist = f"{base}.m3u8"
        descriptor = (
            f"input={r.input_path},"
            "stream=audio,"
            f"init_segment={init_seg},"
            f"segment_template={seg_tpl},"
            f"playlist_name={playlist},"
            f"language={r.language}"
        )
    elif r.kind == "text":
        out_path = os.path.join(out_dir, f"{base}.vtt")
        descriptor = (
            f"input={r.input_path},"
            "stream=text,"
            f"output={out_path},"
            f"language={r.language}"
        )
    else:
        raise ValueError(
            f"Unsupported rendition kind {r.kind!r}; use video/audio/text"
        )
    return descriptor


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def package(
    renditions: List[PackagerRendition],
    output_dir: str,
    protocol: str = "hls",
    segment_duration: float = 4.0,
    low_latency: bool = False,
    drm_scheme: Optional[str] = None,
    drm_key_hex: Optional[str] = None,
    drm_key_id_hex: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
    on_progress: Optional[Callable] = None,
    timeout: int = 7200,
) -> PackagerResult:
    """Package ``renditions`` into an adaptive-streaming manifest.

    Args:
        renditions: One or more :class:`PackagerRendition`.  Must
            include at least one video or audio input.
        output_dir: Directory that will hold the init segments, media
            segments, and manifest files. Created if missing.
        protocol: ``"hls"`` / ``"dash"`` / ``"both"``.
        segment_duration: Target segment length in seconds.
        low_latency: Enable LL-HLS partial segments and CMAF
            ``chunk_duration`` — requires ``protocol="hls"`` or
            ``"both"``.
        drm_scheme: One of :data:`DRM_SCHEMES` for CENC output. When
            set, ``drm_key_hex`` **and** ``drm_key_id_hex`` are
            required (32-char hex each, AES-128).
        drm_key_hex: 32-char hex AES-128 content key.
        drm_key_id_hex: 32-char hex CENC key-ID.
        extra_args: Additional args appended verbatim to the packager
            command.  Power-user escape hatch.
        on_progress: ``(pct, msg)`` progress callback. Shaka doesn't
            stream progress; we fire start + finish only.
        timeout: Subprocess timeout in seconds.

    Returns:
        :class:`PackagerResult` with resolved manifest paths.

    Raises:
        RuntimeError: Shaka missing or returned non-zero.
        ValueError: invalid arguments.
        FileNotFoundError: any rendition input missing.
    """
    if not renditions:
        raise ValueError("At least one rendition required")
    if protocol not in OUTPUT_PROTOCOLS:
        raise ValueError(f"protocol must be one of {OUTPUT_PROTOCOLS}")
    for r in renditions:
        if not os.path.isfile(r.input_path):
            raise FileNotFoundError(f"rendition input not found: {r.input_path}")
        if r.kind not in ("video", "audio", "text"):
            raise ValueError(f"bad rendition kind: {r.kind!r}")
    if drm_scheme is not None:
        if drm_scheme not in DRM_SCHEMES:
            raise ValueError(f"drm_scheme must be one of {DRM_SCHEMES}")
        if not (drm_key_hex and drm_key_id_hex):
            raise ValueError(
                "drm_key_hex AND drm_key_id_hex required when drm_scheme is set"
            )
        if len(drm_key_hex) != 32 or len(drm_key_id_hex) != 32:
            raise ValueError("DRM keys must be 32-char hex (AES-128)")
        try:
            int(drm_key_hex, 16)
            int(drm_key_id_hex, 16)
        except ValueError as exc:
            raise ValueError("DRM keys must be valid hex") from exc
    if low_latency and protocol == "dash":
        raise ValueError("low_latency requires protocol=hls or both")
    segment_duration = max(0.5, min(30.0, float(segment_duration)))

    bin_path = resolve_packager_bin()
    os.makedirs(output_dir, exist_ok=True)

    if on_progress:
        on_progress(5, f"Shaka Packager → {protocol} ({len(renditions)} rendition(s))")

    cmd: List[str] = [bin_path]
    for r in renditions:
        cmd.append(_stream_descriptor(r, output_dir))

    # Global flags
    cmd += [
        "--segment_duration", str(segment_duration),
    ]
    if low_latency:
        cmd += ["--utc_timings", "urn:mpeg:dash:utc:http-head:2014=https://time.akamai.com"]
        cmd += ["--generate_static_live_mpd"]
        cmd += ["--allow_approximate_segment_timeline"]
        # LL-HLS partial segments: 0.33s default
        cmd += ["--hls_base_url", ""]
    # Manifest output paths
    hls_manifest_path: Optional[str] = None
    dash_manifest_path: Optional[str] = None
    if protocol in ("hls", "both"):
        hls_manifest_path = os.path.join(output_dir, "manifest.m3u8")
        cmd += ["--hls_master_playlist_output", hls_manifest_path]
    if protocol in ("dash", "both"):
        dash_manifest_path = os.path.join(output_dir, "manifest.mpd")
        cmd += ["--mpd_output", dash_manifest_path]

    # DRM — CENC with a raw-key
    if drm_scheme:
        cmd += [
            "--enable_raw_key_encryption",
            "--keys", f"label=PRIMARY:key_id={drm_key_id_hex}:key={drm_key_hex}",
            "--protection_scheme", "cenc",
        ]
        if drm_scheme == "common_system":
            cmd += ["--generate_common_pssh"]
        elif drm_scheme == "widevine":
            cmd += ["--generate_widevine_protection_system"]

    if extra_args:
        cmd += [str(a) for a in extra_args]

    logger.debug("Shaka cmd: %s", " ".join(cmd))

    try:
        proc = _sp.run(
            cmd, capture_output=True, text=True,
            timeout=timeout, check=False,
        )
    except _sp.TimeoutExpired as exc:
        raise RuntimeError(
            f"Shaka Packager timed out after {timeout}s"
        ) from exc

    if proc.returncode != 0:
        raise RuntimeError(
            f"Shaka Packager failed (rc={proc.returncode}): "
            f"{(proc.stderr or proc.stdout or '')[-600:]}"
        )

    if on_progress:
        on_progress(100, "Packaging complete")

    return PackagerResult(
        output_dir=output_dir,
        hls_manifest=hls_manifest_path if os.path.isfile(hls_manifest_path or "") else None,
        dash_manifest=dash_manifest_path if os.path.isfile(dash_manifest_path or "") else None,
        protocol=protocol,
        rendition_count=len(renditions),
        drm_scheme=drm_scheme,
        notes=[
            f"segment_duration={segment_duration}",
            f"low_latency={low_latency}",
            f"shaka_version={version()}",
        ],
    )
