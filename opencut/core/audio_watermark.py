"""
OpenCut AudioSeal Watermarking v1.28.0

AI-inaudible audio watermark embed and detect (AudioSeal, Facebook Research).
"""
from __future__ import annotations

import logging
import os
import subprocess
import struct
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from opencut.helpers import _try_import

logger = logging.getLogger("opencut")

INSTALL_HINT = "pip install audioseal"


@dataclass
class WatermarkResult:
    output: str = ""
    method: str = "audioseal"
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str):
        return getattr(self, k)

    def keys(self):
        return ("output", "method", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_audioseal_available() -> bool:
    """True when audioseal pip package is importable."""
    return _try_import("audioseal") is not None


def embed(
    audio_path: str,
    message: str = "opencut",
    output: Optional[str] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> WatermarkResult:
    """Embed an AI-inaudible watermark into audio. Requires audioseal."""
    if not check_audioseal_available():
        raise RuntimeError(f"AudioSeal is not installed. Install with:\n    {INSTALL_HINT}")
    import audioseal  # type: ignore
    import torch

    if on_progress:
        on_progress(5, "Loading AudioSeal generator model")

    generator = audioseal.AudioSeal.load_model("facebook/audioseal-generator-16bits")

    cmd = ["ffmpeg", "-y", "-i", audio_path, "-f", "f32le", "-ar", "16000", "-ac", "1", "pipe:1"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    n = len(raw) // 4
    samples = struct.unpack(f"{n}f", raw)
    wav = torch.tensor(samples).unsqueeze(0).unsqueeze(0)

    if on_progress:
        on_progress(40, "Embedding watermark")

    msg_tensor = torch.zeros(1, 16, dtype=torch.int32)
    for i, ch in enumerate(message[:16]):
        msg_tensor[0, i] = ord(ch) % 128

    watermarked = generator.get_watermark(wav, sample_rate=16000, message=msg_tensor)

    if output is None:
        base, ext = os.path.splitext(audio_path)
        output = f"{base}_watermarked{ext or '.wav'}"

    raw_out = (watermarked.squeeze().numpy() * 32767).astype("int16").tobytes()
    enc_cmd = ["ffmpeg", "-y", "-f", "s16le", "-ar", "16000", "-ac", "1", "-i", "pipe:0", output]
    subprocess.run(enc_cmd, input=raw_out, capture_output=True, check=True)

    if on_progress:
        on_progress(100, "Done")

    return WatermarkResult(output=output, method="audioseal", notes=[])


def detect(audio_path: str) -> dict:
    """Detect AudioSeal watermark. Returns detection result dict."""
    if not check_audioseal_available():
        raise RuntimeError(f"AudioSeal is not installed. Install with:\n    {INSTALL_HINT}")
    import audioseal  # type: ignore
    import torch

    detector = audioseal.AudioSeal.load_model("facebook/audioseal-detector-16bits")
    cmd = ["ffmpeg", "-y", "-i", audio_path, "-f", "f32le", "-ar", "16000", "-ac", "1", "pipe:1"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    n = len(raw) // 4
    samples = struct.unpack(f"{n}f", raw)
    wav = torch.tensor(samples).unsqueeze(0).unsqueeze(0)

    result, message = detector.detect_watermark(wav, sample_rate=16000)
    confidence = float(result.mean().item())
    detected = confidence > 0.5

    decoded_msg = ""
    if detected and message is not None:
        decoded_msg = "".join(
            chr(int(message[0, i].item()) % 128)
            for i in range(min(16, message.shape[1]))
        ).rstrip("\x00")

    return {"detected": detected, "confidence": confidence, "message": decoded_msg, "method": "audioseal"}


__all__ = ["WatermarkResult", "check_audioseal_available", "INSTALL_HINT", "embed", "detect"]
