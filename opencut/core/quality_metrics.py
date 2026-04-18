"""
Objective video quality metrics via FFmpeg.

Compares a **distorted** video against a **reference** via FFmpeg's
built-in quality metrics — VMAF (Netflix), SSIM, PSNR — producing a
structured report suitable for CI regression gates and subjective
quality benchmarks.

Distinct from ``clip_quality.py`` (which does *subjective* zero-shot
scoring via CLIP-IQA+).  Use this module when you have a ground-truth
reference and want to measure how much re-encoding / processing
degraded it; use ``clip_quality.py`` for single-clip aesthetic scoring.

Requires FFmpeg built with ``--enable-libvmaf`` for VMAF; SSIM and
PSNR are always available.  The bundled OpenCut FFmpeg (Windows
installer path) has libvmaf; distro builds vary.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess as _sp
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import get_ffmpeg_path

logger = logging.getLogger("opencut")


METRICS = ("vmaf", "ssim", "psnr")


@dataclass
class QualityReport:
    """Structured quality report for distorted-vs-reference comparison."""
    distorted: str = ""
    reference: str = ""
    vmaf: Optional[float] = None
    vmaf_min: Optional[float] = None
    vmaf_harmonic: Optional[float] = None
    ssim: Optional[float] = None
    psnr: Optional[float] = None
    passes: Optional[bool] = None         # populated when threshold given
    threshold_vmaf: Optional[float] = None
    frames: Optional[int] = None
    duration: Optional[float] = None
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

def check_quality_metrics_available() -> bool:
    """FFmpeg is required; SSIM/PSNR are always available once FFmpeg is."""
    return bool(get_ffmpeg_path())


def check_vmaf_available() -> bool:
    """Probe FFmpeg's filter list for libvmaf."""
    ff = get_ffmpeg_path()
    if not ff:
        return False
    try:
        proc = _sp.run(
            [ff, "-hide_banner", "-filters"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        return bool(re.search(r"^\s*\S*\s+libvmaf\b", out, flags=re.MULTILINE))
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

_SSIM_RE = re.compile(r"SSIM\s+.*?All:\s*([\d.]+)", re.IGNORECASE)
_PSNR_RE = re.compile(r"PSNR\s+.*?average:\s*([\d.]+)", re.IGNORECASE)


def _parse_vmaf_json(path: str) -> Dict[str, float]:
    """Parse the VMAF JSON log produced by ``libvmaf:log_fmt=json``."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f) or {}

    pooled = data.get("pooled_metrics") or {}
    vmaf_node = pooled.get("vmaf") or pooled.get("vmaf_b_v0.6.3") or {}

    mean = vmaf_node.get("mean")
    min_val = vmaf_node.get("min")
    harmonic = vmaf_node.get("harmonic_mean")

    # Some FFmpeg builds emit the metric under ``metrics``/``aggregate``
    if mean is None:
        agg = data.get("aggregate") or {}
        mean = agg.get("VMAF_score") or agg.get("vmaf")
    if min_val is None:
        agg = data.get("aggregate") or {}
        min_val = agg.get("VMAF_score_min")

    frames = data.get("frames") or []
    return {
        "mean": float(mean) if mean is not None else float("nan"),
        "min": float(min_val) if min_val is not None else float("nan"),
        "harmonic": float(harmonic) if harmonic is not None else float("nan"),
        "frame_count": len(frames),
    }


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def _run_ffmpeg_filter_complex(
    distorted: str,
    reference: str,
    fc: str,
    timeout: int = 3600,
) -> str:
    """Run FFmpeg with a metric filter_complex graph, return stderr text."""
    ff = get_ffmpeg_path()
    cmd = [
        ff, "-hide_banner",
        "-i", distorted,
        "-i", reference,
        "-lavfi", fc,
        "-f", "null", "-",
    ]
    logger.debug("quality_metrics ffmpeg: %s", " ".join(cmd))
    proc = _sp.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg quality-metrics run failed (rc={proc.returncode}): "
            f"{(proc.stderr or '')[-400:]}"
        )
    return proc.stderr or ""


def measure_ssim(distorted: str, reference: str, timeout: int = 3600) -> float:
    """Return the mean SSIM (0..1) of ``distorted`` against ``reference``."""
    stderr = _run_ffmpeg_filter_complex(
        distorted, reference,
        "[0:v][1:v]ssim=stats_file=-",
        timeout=timeout,
    )
    match = _SSIM_RE.search(stderr)
    if not match:
        raise RuntimeError("SSIM result not found in ffmpeg output")
    return float(match.group(1))


def measure_psnr(distorted: str, reference: str, timeout: int = 3600) -> float:
    """Return the mean PSNR (dB) of ``distorted`` against ``reference``."""
    stderr = _run_ffmpeg_filter_complex(
        distorted, reference,
        "[0:v][1:v]psnr=stats_file=-",
        timeout=timeout,
    )
    match = _PSNR_RE.search(stderr)
    if not match:
        raise RuntimeError("PSNR result not found in ffmpeg output")
    return float(match.group(1))


def measure_vmaf(distorted: str, reference: str, timeout: int = 3600) -> Dict[str, float]:
    """Return the VMAF pooled metrics of ``distorted`` vs ``reference``.

    Raises :class:`RuntimeError` when libvmaf isn't compiled into FFmpeg.
    """
    if not check_vmaf_available():
        raise RuntimeError(
            "FFmpeg was not built with --enable-libvmaf. "
            "Use a build from Gyan.dev / BtbN on Windows, or "
            "recompile FFmpeg with libvmaf support."
        )

    fd, json_path = tempfile.mkstemp(suffix=".json", prefix="opencut_vmaf_")
    os.close(fd)
    try:
        # libvmaf escape: colons need \: on Windows paths but FFmpeg's
        # lavfi parser accepts forward slashes everywhere, so prefer
        # those.
        log_path = json_path.replace("\\", "/")
        fc = (
            "[0:v]scale=1920:1080:flags=bicubic,format=yuv420p[dist];"
            "[1:v]scale=1920:1080:flags=bicubic,format=yuv420p[ref];"
            f"[dist][ref]libvmaf=log_fmt=json:log_path={log_path}"
        )
        _run_ffmpeg_filter_complex(distorted, reference, fc, timeout=timeout)

        if not os.path.isfile(json_path):
            raise RuntimeError("VMAF JSON log not written")
        return _parse_vmaf_json(json_path)
    finally:
        try:
            os.unlink(json_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compare_videos(
    distorted: str,
    reference: str,
    metrics: Optional[List[str]] = None,
    threshold_vmaf: Optional[float] = None,
    on_progress: Optional[Callable] = None,
    timeout: int = 3600,
) -> QualityReport:
    """Compute quality metrics for ``distorted`` against ``reference``.

    Args:
        distorted: Path to the lossy / re-encoded file.
        reference: Path to the ground-truth source.
        metrics: Subset of :data:`METRICS`.  Defaults to all three.
        threshold_vmaf: If set, populates ``passes`` on the result
            (``True`` when VMAF >= threshold).
        on_progress: ``(pct, msg)`` callback.
        timeout: Per-metric subprocess timeout in seconds.

    Returns:
        :class:`QualityReport` with per-metric scores, ``passes`` gate,
        and diagnostic ``notes``.

    Raises:
        FileNotFoundError: either input missing.
        RuntimeError: FFmpeg missing or VMAF requested without libvmaf.
        ValueError: invalid metric in ``metrics``.
    """
    if not os.path.isfile(distorted):
        raise FileNotFoundError(f"distorted not found: {distorted}")
    if not os.path.isfile(reference):
        raise FileNotFoundError(f"reference not found: {reference}")
    if not check_quality_metrics_available():
        raise RuntimeError("FFmpeg is not available")

    wanted = list(metrics or METRICS)
    unknown = [m for m in wanted if m not in METRICS]
    if unknown:
        raise ValueError(
            f"Unknown metrics {unknown}. Valid: {list(METRICS)}"
        )

    report = QualityReport(
        distorted=distorted,
        reference=reference,
        threshold_vmaf=threshold_vmaf,
    )
    # Probe reference for duration / frame count (best-effort)
    try:
        from opencut.helpers import get_video_info
        info = get_video_info(reference)
        report.duration = float(info.get("duration") or 0.0)
    except Exception:  # noqa: BLE001
        pass

    n = max(1, len(wanted))
    step = 0
    for name in wanted:
        step += 1
        if on_progress:
            on_progress(
                int(100 * (step - 1) / n),
                f"Measuring {name.upper()}…",
            )
        try:
            if name == "vmaf":
                v = measure_vmaf(distorted, reference, timeout=timeout)
                if v["mean"] == v["mean"]:       # nan filter via self-equality
                    report.vmaf = round(v["mean"], 3)
                if v["min"] == v["min"]:
                    report.vmaf_min = round(v["min"], 3)
                if v["harmonic"] == v["harmonic"]:
                    report.vmaf_harmonic = round(v["harmonic"], 3)
                if v.get("frame_count"):
                    report.frames = int(v["frame_count"])
            elif name == "ssim":
                report.ssim = round(measure_ssim(distorted, reference, timeout=timeout), 4)
            elif name == "psnr":
                report.psnr = round(measure_psnr(distorted, reference, timeout=timeout), 3)
        except RuntimeError as exc:
            report.notes.append(f"{name}: {exc}")
            logger.warning("Quality metric %s failed: %s", name, exc)

    if threshold_vmaf is not None and report.vmaf is not None:
        report.passes = report.vmaf >= float(threshold_vmaf)

    if on_progress:
        on_progress(100, "Quality comparison complete")

    return report


def batch_compare(
    pairs: List[Dict[str, str]],
    metrics: Optional[List[str]] = None,
    threshold_vmaf: Optional[float] = None,
    on_progress: Optional[Callable] = None,
) -> List[QualityReport]:
    """Run :func:`compare_videos` for each ``{distorted, reference}`` pair.

    Useful for CI golden-regression suites: provide a list of pairs,
    get back a list of reports — iterate for pass/fail logic.
    """
    out: List[QualityReport] = []
    n = max(1, len(pairs))
    for i, pair in enumerate(pairs):
        distorted = str(pair.get("distorted") or "")
        reference = str(pair.get("reference") or "")
        if not distorted or not reference:
            out.append(QualityReport(
                distorted=distorted, reference=reference,
                notes=["skipped: missing 'distorted' or 'reference'"],
            ))
            continue
        if on_progress:
            on_progress(int(100 * i / n), f"Pair {i + 1}/{n}")
        try:
            out.append(compare_videos(
                distorted, reference,
                metrics=metrics,
                threshold_vmaf=threshold_vmaf,
            ))
        except Exception as exc:  # noqa: BLE001
            out.append(QualityReport(
                distorted=distorted, reference=reference,
                notes=[f"error: {exc}"],
            ))
    if on_progress:
        on_progress(100, f"Batch complete: {len(out)} pair(s)")
    return out
