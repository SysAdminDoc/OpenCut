"""
OpenCut Spectral Match v1.28.0

FFT-based EQ matching. Match target audio spectrum to reference.
Requires scipy + numpy.
"""
from __future__ import annotations

import logging
import os
import subprocess
import struct
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import _try_import, get_ffmpeg_path

logger = logging.getLogger("opencut")
INSTALL_HINT = "pip install scipy numpy"


def check_spectral_match_available() -> bool:
    return _try_import("scipy") is not None and _try_import("numpy") is not None


@dataclass
class SpectralMatchResult:
    output: str = ""
    filter_db: List[float] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str):
        return getattr(self, k)

    def keys(self):
        return ("output", "filter_db", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def _decode_to_float(path: str) -> List[float]:
    """Decode up to 60 s of audio to float32 PCM; limits RAM for long files."""
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-y", "-i", path,
        "-t", "60",
        "-f", "f32le", "-ar", "44100", "-ac", "1",
        "pipe:1",
    ]
    try:
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        stderr_msg = e.stderr.decode(errors="replace") if e.stderr else ""
        raise RuntimeError(f"FFmpeg decode failed (exit {e.returncode}): {stderr_msg[:500]}") from e
    n = len(raw) // 4
    return list(struct.unpack(f"{n}f", raw))


def _compute_filter(input_samples, ref_samples, strength: float = 1.0):
    import numpy as np
    n_fft = 65536
    inp = np.array(input_samples, dtype=np.float32)
    ref = np.array(ref_samples, dtype=np.float32)
    inp_spec = np.abs(np.fft.rfft(inp[:n_fft] if len(inp) >= n_fft else np.pad(inp, (0, n_fft - len(inp)))))
    ref_spec = np.abs(np.fft.rfft(ref[:n_fft] if len(ref) >= n_fft else np.pad(ref, (0, n_fft - len(ref)))))
    eps = 1e-9
    inp_spec = np.maximum(inp_spec, eps)
    ref_spec = np.maximum(ref_spec, eps)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / 44100)
    octave_bands = []
    band_edges = []
    f = 20.0
    while f < 20000:
        f_low, f_high = f, f * 2 ** (1 / 3)
        mask = (freqs >= f_low) & (freqs < f_high)
        if mask.any():
            inp_mean = inp_spec[mask].mean()
            ref_mean = ref_spec[mask].mean()
            correction_db = 20 * np.log10(ref_mean / inp_mean) * strength
            correction_db = np.clip(correction_db, -24, 24)
            octave_bands.append(float(correction_db))
            band_edges.append(float((f_low + f_high) / 2))
        f = f_high
    return octave_bands, band_edges


def match(
    input_path: str,
    reference_path: str,
    output: Optional[str] = None,
    strength: float = 1.0,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> SpectralMatchResult:
    if not check_spectral_match_available():
        raise RuntimeError(f"scipy/numpy not installed. Install with:\n    {INSTALL_HINT}")
    if on_progress:
        on_progress(10, "Decoding input audio")
    inp_samples = _decode_to_float(input_path)
    if on_progress:
        on_progress(25, "Decoding reference audio")
    ref_samples = _decode_to_float(reference_path)
    if on_progress:
        on_progress(40, "Computing spectral correction")
    filter_db, band_centers = _compute_filter(inp_samples, ref_samples, strength)
    if output is None:
        base, ext = os.path.splitext(input_path)
        output = f"{base}_matched{ext or '.wav'}"
    eq_filters = []
    for db_val, fc in zip(filter_db, band_centers):
        if abs(db_val) > 0.1:
            bw = fc * 0.4
            eq_filters.append(f"equalizer=f={fc:.1f}:width_type=h:width={bw:.1f}:g={db_val:.2f}")
    if eq_filters:
        cmd = [get_ffmpeg_path(), "-y", "-i", input_path, "-af", ",".join(eq_filters), output]
    else:
        cmd = [get_ffmpeg_path(), "-y", "-i", input_path, "-c:a", "copy", output]
    if on_progress:
        on_progress(70, "Applying spectral correction")
    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        stderr_msg = e.stderr.decode(errors="replace") if e.stderr else ""
        raise RuntimeError(f"FFmpeg EQ apply failed (exit {e.returncode}): {stderr_msg[:500]}") from e
    if on_progress:
        on_progress(100, "Done")
    return SpectralMatchResult(output=output, filter_db=filter_db, notes=[])


def preview(input_path: str, reference_path: str) -> Dict:
    if not check_spectral_match_available():
        raise RuntimeError(f"scipy/numpy not installed. Install with:\n    {INSTALL_HINT}")
    inp_samples = _decode_to_float(input_path)
    ref_samples = _decode_to_float(reference_path)
    filter_db, band_centers = _compute_filter(inp_samples, ref_samples, 1.0)
    return {"filter_db": filter_db, "octave_bands": band_centers, "notes": []}


__all__ = ["check_spectral_match_available", "INSTALL_HINT", "SpectralMatchResult", "match", "preview"]
