"""
Tests for caption export preflight and recovery.
"""

from opencut.core.caption_export_preflight import (
    CaptionPreflightResult,
    check_host_compatibility,
    check_segments_valid,
    run_caption_export_preflight,
)


class TestSegmentValidation:

    def test_empty_segments(self):
        diags, ok = check_segments_valid([])
        assert ok is False
        assert diags[0].status == "error"

    def test_valid_segments(self):
        segs = [{"start": 0, "end": 5, "text": "Hello"}]
        diags, ok = check_segments_valid(segs, video_duration=10.0)
        assert ok is True
        assert diags[0].status == "ok"

    def test_out_of_range_segments(self):
        segs = [{"start": 0, "end": 20, "text": "Late"}]
        diags, ok = check_segments_valid(segs, video_duration=10.0)
        assert ok is True
        assert any(d.check == "timecode_range" for d in diags)


class TestHostCompatibility:

    def test_unknown_host(self):
        diags, strategy = check_host_compatibility(None)
        assert strategy == "srt_sidecar"
        assert diags[0].status == "warning"

    def test_risky_host(self):
        diags, strategy = check_host_compatibility("26.0")
        assert strategy == "srt_sidecar"

    def test_safe_host(self):
        diags, strategy = check_host_compatibility("25.6")
        assert strategy == "native"
        assert diags[0].status == "ok"


class TestPreflightIntegration:

    def test_no_input(self):
        result = run_caption_export_preflight()
        assert result.ready is False
        assert result.fallback_strategy == "burnin"

    def test_valid_segments_native(self):
        segs = [
            {"start": 0, "end": 3, "text": "Hello"},
            {"start": 3, "end": 6, "text": "World"},
        ]
        result = run_caption_export_preflight(
            segments=segs,
            host_version="25.6",
            video_duration=10.0,
        )
        assert result.ready is True
        assert result.fallback_strategy == "native"
        assert result.caption_count == 2

    def test_valid_segments_unknown_host(self):
        segs = [{"start": 0, "end": 3, "text": "Test"}]
        result = run_caption_export_preflight(segments=segs)
        assert result.ready is True
        assert result.fallback_strategy == "srt_sidecar"

    def test_srt_text_valid(self):
        srt = "1\n00:00:01,000 --> 00:00:03,000\nHello\n\n2\n00:00:03,000 --> 00:00:05,000\nWorld\n"
        result = run_caption_export_preflight(
            srt_text=srt,
            host_version="25.6",
        )
        assert result.ready is True
        assert result.caption_count == 2

    def test_srt_text_empty(self):
        result = run_caption_export_preflight(srt_text="no cues here")
        assert result.ready is False

    def test_force_strategy(self):
        segs = [{"start": 0, "end": 3, "text": "Test"}]
        result = run_caption_export_preflight(
            segments=segs,
            host_version="25.6",
            force_strategy="burnin",
        )
        assert result.fallback_strategy == "burnin"

    def test_subscriptable(self):
        result = CaptionPreflightResult()
        assert "ready" in result
        assert result["ready"] is True
        assert set(result.keys()) == set(CaptionPreflightResult.__dataclass_fields__.keys())
