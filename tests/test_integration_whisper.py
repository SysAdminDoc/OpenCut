"""
Integration tests for Whisper transcription (requires model download).

These tests are marked both ``integration`` and ``slow`` because they need
a Whisper model to be downloaded on first run, which can take significant time.

Run manually with:
    python -m pytest tests/test_integration_whisper.py -v --tb=short
"""

import shutil
import subprocess

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg not installed"),
]


@pytest.fixture
def test_audio(tmp_path):
    """Generate a 5-second 440 Hz sine-wave WAV file."""
    out = tmp_path / "test_audio.wav"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=5",
            "-af", "volume=0.8",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return str(out)


class TestWhisperTranscription:
    def test_transcribe_returns_segments(self, client, csrf_token, test_audio):
        """POST /captions/transcribe with real audio and tiny model."""
        pytest.skip("Requires Whisper model download — run manually with: pytest -m slow")
