"""
Tests for OpenCut core functionality.

Uses a generated test audio file with known silence patterns.
"""

import os
import subprocess
import tempfile
import pytest

# Generate test media files

def generate_test_audio(output_path: str, duration: float = 10.0):
    """
    Generate a test audio file with known silence/speech pattern.

    Pattern (10 seconds total):
      0.0 - 1.0s : silence
      1.0 - 3.5s : speech (sine tone)
      3.5 - 5.0s : silence
      5.0 - 8.0s : speech (sine tone)
      8.0 - 10.0s: silence
    """
    # Build a complex audio filter:
    # - Generate silence and tones at specific intervals
    # - Speech is simulated with a sine wave at conversational level
    filter_str = (
        "aevalsrc=0:d=1[s1];"                          # 0-1s silence
        "sine=frequency=440:duration=2.5[t1];"           # 1-3.5s tone
        "aevalsrc=0:d=1.5[s2];"                         # 3.5-5s silence
        "sine=frequency=330:duration=3[t2];"             # 5-8s tone
        "aevalsrc=0:d=2[s3];"                           # 8-10s silence
        "[s1][t1][s2][t2][s3]concat=n=5:v=0:a=1[out]"
    )

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-filter_complex", filter_str,
        "-map", "[out]",
        "-ar", "48000",
        "-ac", "1",
        "-t", str(duration),
        output_path,
    ]
    subprocess.run(cmd, check=True, timeout=30)


def generate_test_video(output_path: str, duration: float = 10.0):
    """Generate a test video file with the same audio pattern."""
    audio_filter = (
        "aevalsrc=0:d=1[s1];"
        "sine=frequency=440:duration=2.5[t1];"
        "aevalsrc=0:d=1.5[s2];"
        "sine=frequency=330:duration=3[t2];"
        "aevalsrc=0:d=2[s3];"
        "[s1][t1][s2][t2][s3]concat=n=5:v=0:a=1[aout];"
        f"color=c=black:s=320x240:d={duration}:r=30[vout]"
    )

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-filter_complex", audio_filter,
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        output_path,
    ]
    subprocess.run(cmd, check=True, timeout=60)


# ---- Fixtures ----

@pytest.fixture(scope="session")
def test_audio():
    """Create a temporary test audio file."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    generate_test_audio(path)
    yield path
    os.unlink(path)


@pytest.fixture(scope="session")
def test_video():
    """Create a temporary test video file."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        path = f.name
    generate_test_video(path)
    yield path
    os.unlink(path)


# ---- Media Probe Tests ----

class TestMediaProbe:
    def test_probe_audio(self, test_audio):
        from opencut.utils.media import probe
        info = probe(test_audio)
        assert info.has_audio
        assert info.audio.sample_rate == 48000
        assert info.duration > 9.0

    def test_probe_video(self, test_video):
        from opencut.utils.media import probe
        info = probe(test_video)
        assert info.has_video
        assert info.has_audio
        assert info.video.width == 320
        assert info.video.height == 240
        assert info.video.fps == pytest.approx(30.0, abs=1.0)

    def test_probe_nonexistent(self):
        from opencut.utils.media import probe
        with pytest.raises(FileNotFoundError):
            probe("/nonexistent/file.mp4")

    def test_pathurl(self, test_audio):
        from opencut.utils.media import probe
        info = probe(test_audio)
        assert info.pathurl.startswith("file://")
        assert info.filename in info.pathurl or "%20" in info.pathurl

    def test_pathurl_special_characters(self):
        """Regression test: # in filenames must be URL-encoded in pathurl."""
        from opencut.utils.media import MediaInfo
        info = MediaInfo(path="/tmp/Joe Rogan Experience #710 - Gavin McInnes (2).mp4")
        url = info.pathurl
        # Hash must be encoded as %23 (not raw # which truncates URLs)
        assert "%23" in url, f"# not encoded in pathurl: {url}"
        # Spaces must be encoded
        assert "%20" in url, f"spaces not encoded in pathurl: {url}"
        # Parentheses must be encoded
        assert "%28" in url and "%29" in url, f"parens not encoded: {url}"
        # Path should still be a valid file:// URL
        assert url.startswith("file://")


# ---- Silence Detection Tests ----

class TestSilenceDetection:
    def test_detect_silences(self, test_audio):
        from opencut.core.silence import detect_silences
        silences = detect_silences(test_audio, threshold_db=-30, min_duration=0.3)
        # Should find at least 2 silence regions (beginning, middle, end)
        assert len(silences) >= 2

    def test_detect_speech(self, test_audio):
        from opencut.core.silence import detect_speech
        from opencut.utils.config import SilenceConfig
        config = SilenceConfig(threshold_db=-30, min_duration=0.3, padding_before=0.05, padding_after=0.05)
        segments = detect_speech(test_audio, config=config)
        # Should find 2 speech segments
        assert len(segments) == 2
        # First speech segment around 1.0-3.5s
        assert segments[0].start < 1.5
        assert segments[0].end > 3.0
        # Second speech segment around 5.0-8.0s
        assert segments[1].start < 5.5
        assert segments[1].end > 7.5

    def test_edit_summary(self, test_audio):
        from opencut.core.silence import detect_speech, get_edit_summary
        segments = detect_speech(test_audio)
        summary = get_edit_summary(test_audio, segments)
        assert summary["original_duration"] > 0
        assert summary["kept_duration"] > 0
        assert summary["removed_duration"] > 0
        assert 0 < summary["reduction_percent"] < 100


# ---- Premiere XML Export Tests ----

class TestPremiereExport:
    def test_export_xml(self, test_video):
        from opencut.core.silence import detect_speech
        from opencut.export.premiere import export_premiere_xml

        segments = detect_speech(test_video)

        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
            xml_path = f.name

        try:
            export_premiere_xml(test_video, segments, xml_path)
            assert os.path.exists(xml_path)
            assert os.path.getsize(xml_path) > 100

            # Verify it's valid XML
            import xml.etree.ElementTree as ET
            tree = ET.parse(xml_path)
            root = tree.getroot()
            assert root.tag == "xmeml"

            # Verify sequence structure
            seq = root.find("sequence")
            assert seq is not None
            assert seq.find(".//media/video/track") is not None
            assert seq.find(".//media/audio/track") is not None

            # Verify clip items exist
            video_clips = seq.findall(".//media/video/track/clipitem")
            assert len(video_clips) == len(segments)

        finally:
            os.unlink(xml_path)

    def test_export_audio_only(self, test_audio):
        from opencut.core.silence import detect_speech
        from opencut.export.premiere import export_premiere_xml
        from opencut.utils.config import ExportConfig

        segments = detect_speech(test_audio)
        config = ExportConfig(include_video=False)

        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
            xml_path = f.name

        try:
            export_premiere_xml(test_audio, segments, xml_path, config=config)
            assert os.path.exists(xml_path)
        finally:
            os.unlink(xml_path)


# ---- SRT Export Tests ----

class TestSRTExport:
    def test_export_srt(self):
        from opencut.core.captions import CaptionSegment, TranscriptionResult
        from opencut.export.srt import export_srt

        result = TranscriptionResult(
            segments=[
                CaptionSegment(text="Hello world", start=0.0, end=1.5),
                CaptionSegment(text="This is a test", start=2.0, end=3.5),
            ],
            language="en",
        )

        with tempfile.NamedTemporaryFile(suffix=".srt", delete=False, mode="w") as f:
            srt_path = f.name

        try:
            export_srt(result, srt_path)
            content = open(srt_path).read()
            assert "1\n" in content
            assert "Hello world" in content
            assert "-->" in content
            assert "00:00:00,000" in content
        finally:
            os.unlink(srt_path)

    def test_export_vtt(self):
        from opencut.core.captions import CaptionSegment, TranscriptionResult
        from opencut.export.srt import export_vtt

        result = TranscriptionResult(
            segments=[
                CaptionSegment(text="Hello world", start=0.0, end=1.5),
            ],
            language="en",
        )

        with tempfile.NamedTemporaryFile(suffix=".vtt", delete=False, mode="w") as f:
            vtt_path = f.name

        try:
            export_vtt(result, vtt_path)
            content = open(vtt_path).read()
            assert "WEBVTT" in content
            assert "Hello world" in content
        finally:
            os.unlink(vtt_path)


# ---- Config Tests ----

class TestConfig:
    def test_presets(self):
        from opencut.utils.config import get_preset, PRESETS
        for name in PRESETS:
            cfg = get_preset(name)
            assert cfg.silence.threshold_db < 0
            assert cfg.silence.min_duration > 0

    def test_invalid_preset(self):
        from opencut.utils.config import get_preset
        with pytest.raises(ValueError):
            get_preset("nonexistent")
