"""
VMAF-target encoding via ``ab-av1``.

Wraps ``ab-av1`` (https://github.com/alexheretic/ab-av1, MIT) — a Rust
CLI that auto-searches for the lowest-CRF SVT-AV1 / libx264 / libx265
encode hitting a user-specified target VMAF. Replaces the "guess a CRF
and re-render until it looks right" loop with a one-shot "give me VMAF
95 and figure it out" command.

ab-av1 is an external binary, not a pip package — we shell out and
parse its structured output.  Graceful degradation: when the binary is
missing from PATH, this module raises ``RuntimeError`` and callers
fall back to the existing :mod:`opencut.core.av1_export` preset
encoder.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess as _sp
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import output_path

logger = logging.getLogger("opencut")

AB_AV1_BIN = "ab-av1"

SUPPORTED_ENCODERS = {
    "libsvtav1",
    "libaom-av1",
    "libx264",
    "libx265",
}


@dataclass
class AbAv1Result:
    """Structured return for a ``ab-av1`` encode."""
    output: str = ""
    encoder: str = "libsvtav1"
    target_vmaf: float = 0.0
    achieved_vmaf: float = 0.0
    final_crf: float = 0.0
    duration_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, key):
        return getattr(self, key)

    def __contains__(self, key):
        return key in self.__dataclass_fields__

    def keys(self):
        return self.__dataclass_fields__.keys()


def check_ab_av1_available() -> bool:
    """True when ``ab-av1`` is on PATH."""
    return bool(shutil.which(AB_AV1_BIN))


def version() -> str:
    """Return the installed ab-av1 version string, or ``""``."""
    if not check_ab_av1_available():
        return ""
    try:
        out = _sp.run(
            [AB_AV1_BIN, "--version"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        return (out.stdout or out.stderr or "").strip()
    except Exception:  # noqa: BLE001
        return ""


# ---------------------------------------------------------------------------
# Output parser
# ---------------------------------------------------------------------------

# ab-av1 prints lines like:
#   crf 30 VMAF 94.72
#   encode with crf 30 ...
# and ends with a `{...}` JSON summary on some versions.
_CRF_RE = re.compile(r"crf\s+(\d+(?:\.\d+)?)\s+VMAF\s+(\d+(?:\.\d+)?)", re.IGNORECASE)
_FINAL_CRF_RE = re.compile(
    r"(?:final\s+)?(?:selected|encode\s+with)\s+crf\s+(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _parse_output(stdout: str, stderr: str) -> Dict[str, float]:
    """Extract achieved VMAF and final CRF from ab-av1 output.

    ab-av1's machine output varies across versions.  We probe for:
    1. A trailing JSON line (`{"vmaf": ..., "crf": ...}`).
    2. Regex-match the final `crf N VMAF M` line.
    """
    combined = (stdout or "") + "\n" + (stderr or "")
    # Try JSON lines (most recent versions)
    for line in reversed(combined.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                j = json.loads(line)
                vmaf = float(j.get("vmaf") or j.get("achieved") or 0.0)
                crf = float(j.get("crf") or j.get("final_crf") or 0.0)
                if vmaf > 0 or crf > 0:
                    return {"vmaf": vmaf, "crf": crf}
            except (ValueError, TypeError):
                continue
    # Regex fallback — take the last matching pair
    pairs = _CRF_RE.findall(combined)
    if pairs:
        crf, vmaf = pairs[-1]
        return {"vmaf": float(vmaf), "crf": float(crf)}
    # Last-ditch: separate CRF line
    m = _FINAL_CRF_RE.search(combined)
    if m:
        return {"vmaf": 0.0, "crf": float(m.group(1))}
    return {"vmaf": 0.0, "crf": 0.0}


# ---------------------------------------------------------------------------
# Encode driver
# ---------------------------------------------------------------------------

def encode_to_vmaf(
    input_path: str,
    target_vmaf: float = 95.0,
    encoder: str = "libsvtav1",
    preset: Optional[int] = None,
    output: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
    on_progress: Optional[Callable] = None,
    timeout: int = 10800,
) -> AbAv1Result:
    """Encode ``input_path`` to a VMAF target with ab-av1.

    Args:
        input_path: Any video FFmpeg can read.
        target_vmaf: Desired VMAF (0–100). 95 is a sensible default
            for "visually indistinguishable"; 88–92 for casual web,
            97+ for archival masters.
        encoder: Any of :data:`SUPPORTED_ENCODERS`.
        preset: Encoder preset.  For SVT-AV1 that's 0 (slowest) – 13
            (fastest); ab-av1 defaults to ``8``.  ``None`` = no override.
        output: Explicit output path.  Default: ``<input>_vmaf{N}.mp4``.
        extra_args: Additional args passed verbatim to ``ab-av1 auto-encode``.
        on_progress: ``(pct, msg)`` callback.  ab-av1 streams line-based
            progress; we forward parsed milestones (not a continuous %).
        timeout: Subprocess timeout in seconds. Default 3 hours.

    Returns:
        :class:`AbAv1Result` describing the achieved VMAF / final CRF.

    Raises:
        RuntimeError: ab-av1 missing or returned non-zero.
        ValueError: invalid encoder or target_vmaf.
        FileNotFoundError: ``input_path`` missing.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")
    if encoder not in SUPPORTED_ENCODERS:
        raise ValueError(
            f"Unsupported encoder '{encoder}'. "
            f"Valid: {sorted(SUPPORTED_ENCODERS)}"
        )
    if not (20.0 <= float(target_vmaf) <= 99.5):
        raise ValueError("target_vmaf must be in [20, 99.5]")
    if not check_ab_av1_available():
        raise RuntimeError(
            "ab-av1 not installed. Install from "
            "https://github.com/alexheretic/ab-av1/releases "
            "and ensure the binary is on PATH."
        )

    out_path = output or output_path(input_path, f"vmaf{int(target_vmaf)}")

    cmd: List[str] = [
        AB_AV1_BIN, "auto-encode",
        "--input", input_path,
        "--output", out_path,
        "--encoder", encoder,
        "--min-vmaf", str(float(target_vmaf)),
    ]
    if preset is not None:
        cmd += ["--preset", str(int(preset))]
    if extra_args:
        cmd += list(extra_args)

    if on_progress:
        on_progress(5, f"ab-av1 auto-encode → VMAF {target_vmaf}")

    logger.debug("ab-av1 cmd: %s", " ".join(cmd))

    try:
        proc = _sp.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except _sp.TimeoutExpired as exc:
        raise RuntimeError(
            f"ab-av1 timed out after {timeout}s — try a smaller input "
            "or a faster encoder preset."
        ) from exc

    if proc.returncode != 0:
        raise RuntimeError(
            f"ab-av1 failed (rc={proc.returncode}): "
            f"{proc.stderr[-800:] if proc.stderr else proc.stdout[-800:]}"
        )

    parsed = _parse_output(proc.stdout, proc.stderr)

    duration = 0.0
    if os.path.isfile(out_path):
        try:
            from opencut.helpers import get_video_info
            info = get_video_info(out_path)
            duration = float(info.get("duration") or 0.0)
        except Exception:  # noqa: BLE001
            pass

    if on_progress:
        on_progress(100, f"Encoded (VMAF {parsed['vmaf']:.2f}, CRF {parsed['crf']:.1f})")

    return AbAv1Result(
        output=out_path,
        encoder=encoder,
        target_vmaf=round(float(target_vmaf), 2),
        achieved_vmaf=round(parsed["vmaf"], 2),
        final_crf=round(parsed["crf"], 2),
        duration_seconds=round(duration, 2),
        notes=[f"ab-av1 {version()}"],
    )
