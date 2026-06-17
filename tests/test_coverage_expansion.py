"""Coverage expansion tests for previously untested core modules.

Tests pure-logic functions from 10 core modules with mocked external
dependencies.  No FFmpeg, no network, no GPU required.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict
from unittest.mock import MagicMock, patch

import pytest


# ========================================================================
# 1. url_ingest.py
# ========================================================================
class TestUrlIngestCacheKey:
    """Tests for _cache_key deterministic hashing."""

    def test_cache_key_length_is_16(self):
        from opencut.core.url_ingest import _cache_key
        key = _cache_key("https://example.com/video.mp4")
        assert len(key) == 16

    def test_cache_key_is_hex(self):
        from opencut.core.url_ingest import _cache_key
        key = _cache_key("https://example.com/video.mp4")
        int(key, 16)  # raises ValueError if not hex

    def test_cache_key_deterministic(self):
        from opencut.core.url_ingest import _cache_key
        url = "https://example.com/video.mp4"
        assert _cache_key(url) == _cache_key(url)

    def test_cache_key_matches_sha256_prefix(self):
        from opencut.core.url_ingest import _cache_key
        url = "https://example.com/video.mp4"
        expected = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        assert _cache_key(url) == expected

    def test_cache_key_different_urls_differ(self):
        from opencut.core.url_ingest import _cache_key
        k1 = _cache_key("https://a.com/1.mp4")
        k2 = _cache_key("https://b.com/2.mp4")
        assert k1 != k2

    def test_cache_key_unicode_url(self):
        from opencut.core.url_ingest import _cache_key
        key = _cache_key("https://example.com/éè.mp4")
        assert len(key) == 16


class TestIngestResult:
    """Tests for IngestResult dataclass dict-like access."""

    def test_keys_returns_all_field_names(self):
        from opencut.core.url_ingest import IngestResult
        r = IngestResult()
        keys = list(r.keys())
        assert "filepath" in keys
        assert "url" in keys
        assert "title" in keys
        assert "duration" in keys
        assert "filesize_mb" in keys
        assert "source" in keys
        assert "cached" in keys
        assert "notes" in keys

    def test_getitem_by_key(self):
        from opencut.core.url_ingest import IngestResult
        r = IngestResult(filepath="/tmp/test.mp4", title="Test Video")
        assert r["filepath"] == "/tmp/test.mp4"
        assert r["title"] == "Test Video"

    def test_dict_conversion_via_keys(self):
        from opencut.core.url_ingest import IngestResult
        r = IngestResult(filepath="/tmp/x.mp4", url="https://x.com", duration=10.5)
        d = {k: r[k] for k in r.keys()}
        assert d["filepath"] == "/tmp/x.mp4"
        assert d["duration"] == 10.5

    def test_defaults(self):
        from opencut.core.url_ingest import IngestResult
        r = IngestResult()
        assert r.filepath == ""
        assert r.duration == 0.0
        assert r.cached is False
        assert r.notes == []


class TestCheckYtdlpAvailable:
    """Tests for check_ytdlp_available."""

    def test_returns_true_when_found(self):
        with patch("opencut.core.url_ingest.shutil.which", return_value="/usr/bin/yt-dlp"):
            from opencut.core.url_ingest import check_ytdlp_available
            assert check_ytdlp_available() is True

    def test_returns_false_when_missing(self):
        with patch("opencut.core.url_ingest.shutil.which", return_value=None):
            from opencut.core.url_ingest import check_ytdlp_available
            assert check_ytdlp_available() is False


class TestIngestUrlValidation:
    """Tests for ingest_url input validation."""

    def test_rejects_empty_url(self):
        from opencut.core.url_ingest import ingest_url
        with pytest.raises(ValueError, match="url is required"):
            ingest_url("")

    def test_rejects_whitespace_only(self):
        from opencut.core.url_ingest import ingest_url
        with pytest.raises(ValueError, match="url is required"):
            ingest_url("   ")

    def test_rejects_non_http_url(self):
        from opencut.core.url_ingest import ingest_url
        with pytest.raises(ValueError, match="url must be http"):
            ingest_url("ftp://example.com/file.mp4")

    def test_rejects_bare_path(self):
        from opencut.core.url_ingest import ingest_url
        with pytest.raises(ValueError, match="url must be http"):
            ingest_url("/local/path/file.mp4")


# ========================================================================
# 2. multimodal_index.py
# ========================================================================
class TestCheckOcrAvailable:
    """Tests for check_ocr_available."""

    def test_returns_true_with_pytesseract(self):
        fake_mod = MagicMock()
        with patch.dict("sys.modules", {"pytesseract": fake_mod}):
            from opencut.core.multimodal_index import check_ocr_available
            assert check_ocr_available() is True

    def test_returns_true_with_easyocr(self):
        import sys
        # Remove pytesseract if cached, add easyocr
        saved = sys.modules.get("pytesseract")
        sys.modules["pytesseract"] = None  # force ImportError on import
        fake_easy = MagicMock()
        with patch.dict("sys.modules", {"pytesseract": None, "easyocr": fake_easy}):
            # Re-import to get fresh result
            import importlib
            import opencut.core.multimodal_index as mm
            importlib.reload(mm)
            result = mm.check_ocr_available()
        # Restore
        if saved is not None:
            sys.modules["pytesseract"] = saved
        # easyocr present means True
        assert result is True

    def test_returns_false_when_neither(self):
        with patch.dict("sys.modules", {"pytesseract": None, "easyocr": None}):
            import importlib
            import opencut.core.multimodal_index as mm
            importlib.reload(mm)
            result = mm.check_ocr_available()
        assert result is False


class TestCheckAudioClassifyAvailable:
    """Tests for check_audio_classify_available."""

    def test_returns_true_with_librosa(self):
        fake_lib = MagicMock()
        with patch.dict("sys.modules", {"librosa": fake_lib}):
            from opencut.core.multimodal_index import check_audio_classify_available
            assert check_audio_classify_available() is True

    def test_returns_false_without_librosa(self):
        with patch.dict("sys.modules", {"librosa": None}):
            import importlib
            import opencut.core.multimodal_index as mm
            importlib.reload(mm)
            result = mm.check_audio_classify_available()
        assert result is False


class TestGetOcrEngine:
    """Tests for _get_ocr_engine fallback chain."""

    def test_returns_callable(self):
        from opencut.core.multimodal_index import _get_ocr_engine
        engine = _get_ocr_engine()
        assert callable(engine)

    def test_noop_engine_returns_empty_list(self):
        """When neither pytesseract nor easyocr is installed, noop returns []."""
        with patch.dict("sys.modules", {"pytesseract": None, "easyocr": None}):
            import importlib
            import opencut.core.multimodal_index as mm
            importlib.reload(mm)
            engine = mm._get_ocr_engine()
            result = engine("nonexistent.png")
        assert result == []


# ========================================================================
# 3. edl_aaf.py
# ========================================================================
class TestSecondsToTimecode:
    """Tests for _seconds_to_tc timecode conversion."""

    def test_zero(self):
        from opencut.core.edl_aaf import _seconds_to_tc
        assert _seconds_to_tc(0.0) == "00:00:00:00"

    def test_one_second_at_30fps(self):
        from opencut.core.edl_aaf import _seconds_to_tc
        assert _seconds_to_tc(1.0, fps=30.0) == "00:00:01:00"

    def test_one_frame_at_24fps(self):
        from opencut.core.edl_aaf import _seconds_to_tc
        tc = _seconds_to_tc(1.0 / 24.0, fps=24.0)
        assert tc == "00:00:00:01"

    def test_one_hour(self):
        from opencut.core.edl_aaf import _seconds_to_tc
        assert _seconds_to_tc(3600.0, fps=30.0) == "01:00:00:00"

    def test_90_seconds_at_25fps(self):
        from opencut.core.edl_aaf import _seconds_to_tc
        assert _seconds_to_tc(90.0, fps=25.0) == "00:01:30:00"

    def test_negative_clamps_to_zero(self):
        from opencut.core.edl_aaf import _seconds_to_tc
        assert _seconds_to_tc(-5.0) == "00:00:00:00"

    def test_fractional_frames(self):
        from opencut.core.edl_aaf import _seconds_to_tc
        # 2.5 seconds at 30fps = 75 frames = 00:00:02:15
        assert _seconds_to_tc(2.5, fps=30.0) == "00:00:02:15"


class TestTimecodeToSeconds:
    """Tests for _tc_to_seconds timecode parsing."""

    def test_zero_tc(self):
        from opencut.core.edl_aaf import _tc_to_seconds
        assert _tc_to_seconds("00:00:00:00") == 0.0

    def test_one_second(self):
        from opencut.core.edl_aaf import _tc_to_seconds
        assert _tc_to_seconds("00:00:01:00", fps=30.0) == 1.0

    def test_one_frame_at_24fps(self):
        from opencut.core.edl_aaf import _tc_to_seconds
        result = _tc_to_seconds("00:00:00:01", fps=24.0)
        assert abs(result - 1.0 / 24.0) < 0.001

    def test_invalid_tc_raises(self):
        from opencut.core.edl_aaf import _tc_to_seconds
        with pytest.raises(ValueError, match="Invalid timecode"):
            _tc_to_seconds("bad")

    def test_roundtrip_30fps(self):
        from opencut.core.edl_aaf import _seconds_to_tc, _tc_to_seconds
        for secs in [0.0, 1.0, 60.0, 3661.5]:
            tc = _seconds_to_tc(secs, fps=30.0)
            back = _tc_to_seconds(tc, fps=30.0)
            assert abs(back - secs) < 0.05, f"Roundtrip failed for {secs}s"


class TestExportEdl:
    """Tests for export_edl content generation."""

    def test_export_edl_basic(self):
        from opencut.core.edl_aaf import export_edl
        cuts = [
            {"source_in": 0.0, "source_out": 5.0, "record_in": 0.0, "record_out": 5.0,
             "clip_name": "Intro"},
            {"source_in": 10.0, "source_out": 20.0, "record_in": 5.0, "record_out": 15.0},
        ]
        with tempfile.NamedTemporaryFile(suffix=".edl", delete=False, mode="w") as f:
            out_path = f.name
        try:
            result = export_edl(cuts, out_path, title="Test EDL")
            assert result.event_count == 2
            assert os.path.isfile(out_path)
            content = open(out_path, encoding="utf-8").read()
            assert "TITLE: Test EDL" in content
            assert "FROM CLIP NAME: Intro" in content
        finally:
            os.unlink(out_path)

    def test_export_edl_empty_raises(self):
        from opencut.core.edl_aaf import export_edl
        with pytest.raises(ValueError, match="empty"):
            export_edl([], "/tmp/empty.edl")


class TestImportEdl:
    """Tests for import_edl round-trip."""

    def test_import_edl_roundtrip(self):
        from opencut.core.edl_aaf import export_edl, import_edl
        cuts = [
            {"reel": "REEL1", "channel": "V", "transition": "C",
             "source_in": 0.0, "source_out": 5.0,
             "record_in": 0.0, "record_out": 5.0,
             "clip_name": "Shot_A", "source_file": "shot_a.mp4"},
        ]
        with tempfile.NamedTemporaryFile(suffix=".edl", delete=False) as f:
            out_path = f.name
        try:
            export_edl(cuts, out_path, title="Roundtrip Test", fps=30.0)
            result = import_edl(out_path, fps=30.0)
            assert result.title == "Roundtrip Test"
            assert result.event_count == 1
            assert result.cuts[0]["clip_name"] == "Shot_A"
            assert result.cuts[0]["source_file"] == "shot_a.mp4"
        finally:
            os.unlink(out_path)


# ========================================================================
# 4. media_relink.py
# ========================================================================
class TestRelinkCandidateDataclass:
    """Tests for RelinkCandidate and RelinkEntry dataclasses."""

    def test_to_dict(self):
        from opencut.core.media_relink import RelinkCandidate
        c = RelinkCandidate(
            candidate_path="/media/clip.mp4",
            match_type="exact_name",
            confidence=0.95,
            file_size=1024000,
        )
        d = c.to_dict()
        assert d["candidate_path"] == "/media/clip.mp4"
        assert d["confidence"] == 0.95

    def test_relink_entry_to_dict(self):
        from opencut.core.media_relink import RelinkCandidate, RelinkEntry
        e = RelinkEntry(
            offline_path="/old/clip.mp4",
            candidates=[RelinkCandidate(candidate_path="/new/clip.mp4", confidence=0.9)],
            best_match="/new/clip.mp4",
            best_confidence=0.9,
            status="resolved",
        )
        d = e.to_dict()
        assert d["status"] == "resolved"
        assert len(d["candidates"]) == 1


class TestFindCandidates:
    """Tests for find_candidates matching logic."""

    def test_exact_name_match(self):
        from opencut.core.media_relink import find_candidates
        index = {"clip.mp4": ["/media/clip.mp4"]}
        with patch("opencut.core.media_relink._get_file_info", return_value=(1024000, 10.0)):
            candidates = find_candidates("/old/clip.mp4", index)
        assert len(candidates) >= 1
        assert candidates[0].match_type == "exact_name"
        assert candidates[0].confidence >= 0.9

    def test_fuzzy_name_match(self):
        from opencut.core.media_relink import find_candidates
        index = {"clip_v2.mp4": ["/media/clip_v2.mp4"]}
        with patch("opencut.core.media_relink._get_file_info", return_value=(1024000, 10.0)):
            candidates = find_candidates("/old/clip_v1.mp4", index, fuzzy_threshold=0.5)
        # clip_v1.mp4 vs clip_v2.mp4 should fuzzy-match
        assert len(candidates) >= 1
        assert candidates[0].match_type == "fuzzy_name"

    def test_no_match_returns_empty(self):
        from opencut.core.media_relink import find_candidates
        index = {"totally_different.mp4": ["/media/totally_different.mp4"]}
        with patch("opencut.core.media_relink._get_file_info", return_value=(0, 0.0)):
            candidates = find_candidates("/old/xylophone.mov", index, fuzzy_threshold=0.9)
        # Very high threshold, totally different name -> no match
        assert len(candidates) == 0

    def test_size_match_boost(self):
        from opencut.core.media_relink import find_candidates
        index = {"clip.mp4": ["/media/clip.mp4"]}
        # Same name, nearly identical file size -> high confidence
        with patch("opencut.core.media_relink._get_file_info", return_value=(1000000, 10.0)):
            candidates = find_candidates(
                "/old/clip.mp4", index, offline_size=1000000,
            )
        assert candidates[0].confidence >= 0.95

    def test_candidates_sorted_by_confidence(self):
        from opencut.core.media_relink import find_candidates
        index = {
            "clip.mp4": ["/a/clip.mp4"],
            "clip_alt.mp4": ["/b/clip_alt.mp4"],
        }
        with patch("opencut.core.media_relink._get_file_info", return_value=(1000, 5.0)):
            candidates = find_candidates("/old/clip.mp4", index, fuzzy_threshold=0.5)
        # Should be sorted descending by confidence
        for i in range(len(candidates) - 1):
            assert candidates[i].confidence >= candidates[i + 1].confidence


class TestBatchRelink:
    """Tests for batch_relink mapping extraction."""

    def test_filters_by_confidence(self):
        from opencut.core.media_relink import RelinkEntry, RelinkResult, batch_relink
        result = RelinkResult(entries=[
            RelinkEntry(offline_path="/a.mp4", best_match="/new/a.mp4", best_confidence=0.95),
            RelinkEntry(offline_path="/b.mp4", best_match="/new/b.mp4", best_confidence=0.5),
        ])
        mapping = batch_relink(result, min_confidence=0.8)
        assert "/a.mp4" in mapping
        assert "/b.mp4" not in mapping

    def test_empty_result(self):
        from opencut.core.media_relink import RelinkResult, batch_relink
        mapping = batch_relink(RelinkResult())
        assert mapping == {}


# ========================================================================
# 5. auto_update.py
# ========================================================================
class TestVersionParsing:
    """Tests for _parse_version and _version_is_newer."""

    def test_parse_stable(self):
        from opencut.core.auto_update import _parse_version
        result = _parse_version("1.2.3")
        assert result == (1, 2, 3, 99, 0)

    def test_parse_with_v_prefix(self):
        from opencut.core.auto_update import _parse_version
        result = _parse_version("v2.0.1")
        assert result == (2, 0, 1, 99, 0)

    def test_parse_alpha(self):
        from opencut.core.auto_update import _parse_version
        result = _parse_version("1.0.0-alpha.1")
        assert result == (1, 0, 0, 1, 1)

    def test_parse_beta(self):
        from opencut.core.auto_update import _parse_version
        result = _parse_version("1.0.0-beta.2")
        assert result == (1, 0, 0, 2, 2)

    def test_parse_rc(self):
        from opencut.core.auto_update import _parse_version
        result = _parse_version("1.0.0-rc.3")
        assert result == (1, 0, 0, 3, 3)

    def test_parse_invalid_returns_zeros(self):
        from opencut.core.auto_update import _parse_version
        result = _parse_version("not-a-version")
        assert result == (0, 0, 0, 0, 0)


class TestVersionIsNewer:
    """Tests for _version_is_newer comparison."""

    def test_newer_major(self):
        from opencut.core.auto_update import _version_is_newer
        assert _version_is_newer("2.0.0", "1.9.9") is True

    def test_newer_minor(self):
        from opencut.core.auto_update import _version_is_newer
        assert _version_is_newer("1.3.0", "1.2.9") is True

    def test_newer_patch(self):
        from opencut.core.auto_update import _version_is_newer
        assert _version_is_newer("1.2.4", "1.2.3") is True

    def test_same_version(self):
        from opencut.core.auto_update import _version_is_newer
        assert _version_is_newer("1.2.3", "1.2.3") is False

    def test_older_version(self):
        from opencut.core.auto_update import _version_is_newer
        assert _version_is_newer("1.0.0", "2.0.0") is False

    def test_stable_newer_than_rc(self):
        from opencut.core.auto_update import _version_is_newer
        assert _version_is_newer("1.0.0", "1.0.0-rc.1") is True

    def test_rc_newer_than_beta(self):
        from opencut.core.auto_update import _version_is_newer
        assert _version_is_newer("1.0.0-rc.1", "1.0.0-beta.9") is True

    def test_beta_newer_than_alpha(self):
        from opencut.core.auto_update import _version_is_newer
        assert _version_is_newer("1.0.0-beta.1", "1.0.0-alpha.5") is True

    def test_v_prefix_ignored(self):
        from opencut.core.auto_update import _version_is_newer
        assert _version_is_newer("v1.1.0", "v1.0.0") is True
        assert _version_is_newer("v1.0.0", "v1.1.0") is False


class TestParseChangelog:
    """Tests for parse_changelog HTML stripping."""

    def test_strips_html_tags(self):
        from opencut.core.auto_update import ReleaseInfo, parse_changelog
        info = ReleaseInfo(body="<p>New feature</p><br/><b>Bug fix</b>")
        result = parse_changelog(info)
        assert "<p>" not in result
        assert "New feature" in result

    def test_normalizes_blank_lines(self):
        from opencut.core.auto_update import ReleaseInfo, parse_changelog
        info = ReleaseInfo(body="Line 1\n\n\n\n\nLine 2")
        result = parse_changelog(info)
        assert "\n\n\n" not in result

    def test_empty_body(self):
        from opencut.core.auto_update import ReleaseInfo, parse_changelog
        info = ReleaseInfo(body="")
        result = parse_changelog(info)
        assert result == ""


# ========================================================================
# 6. structured_ingest.py
# ========================================================================
class TestRenameByPattern:
    """Tests for rename_by_pattern template expansion."""

    def test_name_and_ext_placeholders(self):
        from opencut.core.structured_ingest import rename_by_pattern
        result = rename_by_pattern("clip.mp4", "{name}_final{ext}")
        assert result == "clip_final.mp4"

    def test_counter_placeholder(self):
        from opencut.core.structured_ingest import rename_by_pattern
        result = rename_by_pattern("clip.mp4", "{counter}_{name}{ext}", {"counter": 42})
        assert result.startswith("0042_clip")
        assert result.endswith(".mp4")

    def test_camera_and_scene(self):
        from opencut.core.structured_ingest import rename_by_pattern
        result = rename_by_pattern(
            "raw.mov", "{camera}_{scene}_{take}{ext}",
            {"camera": "B", "scene": "005", "take": "03"},
        )
        assert result == "B_005_03.mov"

    def test_empty_pattern_returns_original(self):
        from opencut.core.structured_ingest import rename_by_pattern
        result = rename_by_pattern("clip.mp4", "")
        assert result == "clip.mp4"

    def test_sanitizes_invalid_chars(self):
        from opencut.core.structured_ingest import rename_by_pattern
        result = rename_by_pattern("clip.mp4", "{name}:v2{ext}")
        assert ":" not in result

    def test_ensures_extension_present(self):
        from opencut.core.structured_ingest import rename_by_pattern
        result = rename_by_pattern("clip.mp4", "{name}_export")
        assert result.endswith(".mp4")


class TestVerifyChecksum:
    """Tests for verify_checksum computation and matching."""

    def test_sha256_computation(self):
        from opencut.core.structured_ingest import verify_checksum
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            f.write(b"hello world")
            path = f.name
        try:
            result = verify_checksum(path, "sha256")
            expected = hashlib.sha256(b"hello world").hexdigest()
            assert result["hash"] == expected
            assert result["algorithm"] == "sha256"
            assert result["verified"] is None  # no expected provided
        finally:
            os.unlink(path)

    def test_md5_computation(self):
        from opencut.core.structured_ingest import verify_checksum
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            f.write(b"test data")
            path = f.name
        try:
            result = verify_checksum(path, "md5")
            expected = hashlib.md5(b"test data").hexdigest()
            assert result["hash"] == expected
        finally:
            os.unlink(path)

    def test_verify_matching_checksum(self):
        from opencut.core.structured_ingest import verify_checksum
        data = b"verification test"
        expected_hash = hashlib.sha256(data).hexdigest()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            f.write(data)
            path = f.name
        try:
            result = verify_checksum(path, "sha256", expected=expected_hash)
            assert result["verified"] is True
            assert result["match"] is True
        finally:
            os.unlink(path)

    def test_verify_mismatched_checksum(self):
        from opencut.core.structured_ingest import verify_checksum
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            f.write(b"actual content")
            path = f.name
        try:
            result = verify_checksum(path, "sha256", expected="0000dead0000beef")
            assert result["verified"] is True
            assert result["match"] is False
        finally:
            os.unlink(path)

    def test_unsupported_algorithm_raises(self):
        from opencut.core.structured_ingest import verify_checksum
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            f.write(b"x")
            path = f.name
        try:
            with pytest.raises(ValueError, match="Unsupported algorithm"):
                verify_checksum(path, "crc32")
        finally:
            os.unlink(path)

    def test_file_not_found_raises(self):
        from opencut.core.structured_ingest import verify_checksum
        with pytest.raises(FileNotFoundError):
            verify_checksum("/nonexistent/path.bin", "sha256")


# ========================================================================
# 7. batch_conform.py
# ========================================================================
class TestConformSpec:
    """Tests for ConformSpec dataclass and from_dict."""

    def test_defaults(self):
        from opencut.core.batch_conform import ConformSpec
        spec = ConformSpec()
        assert spec.width == 1920
        assert spec.height == 1080
        assert spec.fps == 24.0
        assert spec.video_codec == "libx264"
        assert spec.audio_sample_rate == 48000
        assert spec.pix_fmt == "yuv420p"

    def test_from_dict_basic(self):
        from opencut.core.batch_conform import ConformSpec
        spec = ConformSpec.from_dict({"width": 3840, "height": 2160, "fps": 60.0})
        assert spec.width == 3840
        assert spec.height == 2160
        assert spec.fps == 60.0
        # Defaults preserved for unspecified fields
        assert spec.video_codec == "libx264"

    def test_from_dict_ignores_unknown_keys(self):
        from opencut.core.batch_conform import ConformSpec
        spec = ConformSpec.from_dict({
            "width": 1280, "height": 720,
            "bogus_key": "ignored", "another": 999,
        })
        assert spec.width == 1280
        assert spec.height == 720
        assert not hasattr(spec, "bogus_key")

    def test_from_dict_all_fields(self):
        from opencut.core.batch_conform import ConformSpec
        d = {
            "width": 1280, "height": 720, "fps": 30.0,
            "video_codec": "libx265", "audio_codec": "opus",
            "audio_sample_rate": 44100, "audio_channels": 1,
            "crf": 23, "preset": "fast", "pix_fmt": "yuv422p",
        }
        spec = ConformSpec.from_dict(d)
        assert spec.video_codec == "libx265"
        assert spec.audio_codec == "opus"
        assert spec.audio_sample_rate == 44100
        assert spec.preset == "fast"

    def test_to_dict_roundtrip(self):
        from opencut.core.batch_conform import ConformSpec
        original = ConformSpec(width=1280, height=720, fps=30.0)
        d = original.to_dict()
        restored = ConformSpec.from_dict(d)
        assert restored.width == original.width
        assert restored.height == original.height
        assert restored.fps == original.fps


# ========================================================================
# 8. project_organizer.py
# ========================================================================
class TestClassifyMediaType:
    """Tests for _classify_media_type file extension classification."""

    def test_video_extensions(self):
        from opencut.core.project_organizer import _classify_media_type
        for ext in [".mp4", ".mov", ".avi", ".mkv", ".mxf", ".webm"]:
            assert _classify_media_type(ext) == "video", f"Failed for {ext}"

    def test_audio_extensions(self):
        from opencut.core.project_organizer import _classify_media_type
        for ext in [".wav", ".mp3", ".aac", ".flac", ".ogg"]:
            assert _classify_media_type(ext) == "audio", f"Failed for {ext}"

    def test_image_extensions(self):
        from opencut.core.project_organizer import _classify_media_type
        for ext in [".jpg", ".jpeg", ".png", ".tiff", ".exr"]:
            assert _classify_media_type(ext) == "image", f"Failed for {ext}"

    def test_unknown_extension(self):
        from opencut.core.project_organizer import _classify_media_type
        assert _classify_media_type(".xyz") == "other"

    def test_case_insensitive(self):
        from opencut.core.project_organizer import _classify_media_type
        assert _classify_media_type(".MP4") == "video"
        assert _classify_media_type(".WAV") == "audio"


class TestClassifyShotTypeBasic:
    """Tests for _classify_shot_type_basic heuristics."""

    def test_zero_dimensions_returns_unknown(self):
        from opencut.core.project_organizer import _classify_shot_type_basic
        assert _classify_shot_type_basic(0, 0, {}) == "unknown"

    def test_vertical_video_is_close_up(self):
        from opencut.core.project_organizer import _classify_shot_type_basic
        # 9:16 vertical video (aspect < 0.8)
        assert _classify_shot_type_basic(1080, 1920, {"format": {}, "streams": []}) == "close_up"

    def test_ultrawide_is_wide(self):
        from opencut.core.project_organizer import _classify_shot_type_basic
        # 2.39:1 anamorphic (aspect > 2.2)
        assert _classify_shot_type_basic(2390, 1000, {"format": {}, "streams": []}) == "wide"

    def test_standard_16_9_is_medium(self):
        from opencut.core.project_organizer import _classify_shot_type_basic
        result = _classify_shot_type_basic(
            1920, 1080,
            {"format": {"tags": {}}, "streams": []},
        )
        assert result == "medium"

    def test_dji_drone_is_aerial(self):
        from opencut.core.project_organizer import _classify_shot_type_basic
        probe_data = {
            "format": {"tags": {"com.apple.quicktime.model": "DJI Mavic 3"}},
            "streams": [],
        }
        result = _classify_shot_type_basic(3840, 2160, probe_data)
        assert result == "aerial"


class TestComputeAspectRatioLabel:
    """Tests for _compute_aspect_ratio_label."""

    def test_16_9(self):
        from opencut.core.project_organizer import _compute_aspect_ratio_label
        assert _compute_aspect_ratio_label(1920, 1080) == "16:9"

    def test_9_16(self):
        from opencut.core.project_organizer import _compute_aspect_ratio_label
        assert _compute_aspect_ratio_label(1080, 1920) == "9:16"

    def test_4_3(self):
        from opencut.core.project_organizer import _compute_aspect_ratio_label
        assert _compute_aspect_ratio_label(1440, 1080) == "4:3"

    def test_1_1(self):
        from opencut.core.project_organizer import _compute_aspect_ratio_label
        assert _compute_aspect_ratio_label(1080, 1080) == "1:1"

    def test_zero_returns_unknown(self):
        from opencut.core.project_organizer import _compute_aspect_ratio_label
        assert _compute_aspect_ratio_label(0, 1080) == "unknown"

    def test_custom_ratio(self):
        from opencut.core.project_organizer import _compute_aspect_ratio_label
        result = _compute_aspect_ratio_label(1000, 700)
        assert ":1" in result  # Should be formatted as N.NN:1


class TestInferSceneGroup:
    """Tests for _infer_scene_group filename pattern detection."""

    def test_scene_number_pattern(self):
        from opencut.core.project_organizer import _infer_scene_group
        result = _infer_scene_group("Scene03_Take01.mp4")
        assert "3" in result

    def test_camera_roll_pattern(self):
        from opencut.core.project_organizer import _infer_scene_group
        result = _infer_scene_group("A001C002_210405.mov")
        # Should detect A001-style camera roll
        assert result != "ungrouped"

    def test_ungrouped_for_numeric_name(self):
        from opencut.core.project_organizer import _infer_scene_group
        result = _infer_scene_group("12345.mp4")
        # Purely numeric, no recognized pattern -> ungrouped
        assert result == "ungrouped"

    def test_prefix_grouping(self):
        from opencut.core.project_organizer import _infer_scene_group
        result = _infer_scene_group("interview_john_001.mp4")
        assert "interview" in result.lower()


# ========================================================================
# 9. storage_tiering.py
# ========================================================================
class TestArchiveScanResult:
    """Tests for ArchiveScanResult dataclass."""

    def test_defaults(self):
        from opencut.core.storage_tiering import ArchiveScanResult
        r = ArchiveScanResult()
        assert r.total_scanned == 0
        assert r.eligible_count == 0
        assert r.eligible_size_bytes == 0
        assert r.files == []

    def test_to_dict(self):
        from opencut.core.storage_tiering import ArchiveScanResult
        r = ArchiveScanResult(total_scanned=100, eligible_count=5, eligible_size_bytes=5000000)
        d = r.to_dict()
        assert d["total_scanned"] == 100
        assert d["eligible_count"] == 5


class TestArchiveManifest:
    """Tests for manifest load/save/get."""

    def test_get_archive_manifest_empty_path(self):
        from opencut.core.storage_tiering import get_archive_manifest
        result = get_archive_manifest("")
        assert result == {"version": 1, "entries": {}}

    def test_get_archive_manifest_nonexistent_dir(self):
        from opencut.core.storage_tiering import get_archive_manifest
        result = get_archive_manifest("/nonexistent/archive/path")
        assert result == {"version": 1, "entries": {}}

    def test_manifest_roundtrip(self):
        from opencut.core.storage_tiering import _load_manifest, _save_manifest
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = {
                "version": 1,
                "entries": {
                    "/media/clip.mp4": {
                        "archive_path": "/archive/clip.mp4",
                        "size_bytes": 1000000,
                        "archived_at": "2024-01-01T00:00:00",
                    },
                },
            }
            _save_manifest(tmpdir, manifest)
            loaded = _load_manifest(tmpdir)
            assert loaded["version"] == 1
            assert "/media/clip.mp4" in loaded["entries"]


class TestStubFiles:
    """Tests for stub file creation and reading."""

    def test_create_and_read_stub(self):
        from opencut.core.storage_tiering import _create_stub, _read_stub
        with tempfile.TemporaryDirectory() as tmpdir:
            original = os.path.join(tmpdir, "clip.mp4")
            archive_dest = "/archive/clip.mp4"
            stub_path = _create_stub(original, archive_dest)
            assert os.path.isfile(stub_path)

            data = _read_stub(stub_path)
            assert data["type"] == "opencut_archive_stub"
            assert data["original_path"] == original
            assert data["archive_path"] == archive_dest
            assert "archived_at" in data


class TestScanForArchival:
    """Tests for scan_for_archival threshold logic."""

    def test_scan_nonexistent_dir_raises(self):
        from opencut.core.storage_tiering import scan_for_archival
        with pytest.raises(FileNotFoundError):
            scan_for_archival("/nonexistent/project")

    def test_scan_empty_dir_returns_zero(self):
        from opencut.core.storage_tiering import scan_for_archival
        with tempfile.TemporaryDirectory() as tmpdir:
            result = scan_for_archival(tmpdir, idle_days=0)
        assert result.total_scanned == 0
        assert result.eligible_count == 0

    def test_scan_finds_old_media_files(self):
        from opencut.core.storage_tiering import scan_for_archival
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a .mp4 file and backdate its access/mod times
            fpath = os.path.join(tmpdir, "old_clip.mp4")
            with open(fpath, "wb") as f:
                f.write(b"\x00" * 1024)
            old_time = time.time() - (60 * 86400)  # 60 days ago
            os.utime(fpath, (old_time, old_time))

            result = scan_for_archival(tmpdir, idle_days=30)
        assert result.total_scanned == 1
        assert result.eligible_count == 1
        assert result.files[0]["idle_days"] >= 30

    def test_scan_skips_recent_files(self):
        from opencut.core.storage_tiering import scan_for_archival
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "new_clip.mp4")
            with open(fpath, "wb") as f:
                f.write(b"\x00" * 1024)
            # File just created, so it's recent

            result = scan_for_archival(tmpdir, idle_days=30)
        assert result.total_scanned == 1
        assert result.eligible_count == 0


# ========================================================================
# 10. render_queue.py
# ========================================================================
class TestRenderQueueItem:
    """Tests for RenderQueueItem dataclass."""

    def test_defaults(self):
        from opencut.core.render_queue import RenderQueueItem
        item = RenderQueueItem(id="abc123", input_path="/test.mp4", preset_name="1080p")
        assert item.priority == 3
        assert item.status == "pending"
        assert item.progress == 0

    def test_priority_range(self):
        from opencut.core.render_queue import RenderQueueItem
        item = RenderQueueItem(id="x", input_path="/a.mp4", preset_name="p", priority=5)
        assert item.priority == 5


class TestQueuePriorityOrdering:
    """Tests for get_queue priority sorting."""

    def test_get_queue_returns_sorted_by_priority(self):
        from opencut.core.render_queue import (
            RenderQueueItem,
            _queue,
            _queue_lock,
        )
        # Directly manipulate the internal queue for testing
        with _queue_lock:
            saved = list(_queue)
            _queue.clear()
            _queue.extend([
                RenderQueueItem(id="low", input_path="/a.mp4", preset_name="p", priority=1, created_at=1.0),
                RenderQueueItem(id="high", input_path="/b.mp4", preset_name="p", priority=5, created_at=2.0),
                RenderQueueItem(id="mid", input_path="/c.mp4", preset_name="p", priority=3, created_at=3.0),
            ])

        try:
            from opencut.core.render_queue import get_queue
            items = get_queue()
            assert items[0].id == "high"
            assert items[1].id == "mid"
            assert items[2].id == "low"
        finally:
            with _queue_lock:
                _queue.clear()
                _queue.extend(saved)

    def test_same_priority_ordered_by_created_at(self):
        from opencut.core.render_queue import (
            RenderQueueItem,
            _queue,
            _queue_lock,
            get_queue,
        )
        with _queue_lock:
            saved = list(_queue)
            _queue.clear()
            _queue.extend([
                RenderQueueItem(id="later", input_path="/a.mp4", preset_name="p", priority=3, created_at=200.0),
                RenderQueueItem(id="earlier", input_path="/b.mp4", preset_name="p", priority=3, created_at=100.0),
            ])

        try:
            items = get_queue()
            assert items[0].id == "earlier"
            assert items[1].id == "later"
        finally:
            with _queue_lock:
                _queue.clear()
                _queue.extend(saved)


class TestGetNextPending:
    """Tests for _get_next_pending selection logic."""

    def test_selects_highest_priority_pending(self):
        from opencut.core.render_queue import (
            RenderQueueItem,
            _get_next_pending,
            _queue,
            _queue_lock,
        )
        with _queue_lock:
            saved = list(_queue)
            _queue.clear()
            _queue.extend([
                RenderQueueItem(id="a", input_path="/a.mp4", preset_name="p", priority=2, status="pending"),
                RenderQueueItem(id="b", input_path="/b.mp4", preset_name="p", priority=5, status="pending"),
                RenderQueueItem(id="c", input_path="/c.mp4", preset_name="p", priority=4, status="complete"),
            ])
            result = _get_next_pending()

        try:
            assert result is not None
            assert result.id == "b"
        finally:
            with _queue_lock:
                _queue.clear()
                _queue.extend(saved)

    def test_returns_none_when_no_pending(self):
        from opencut.core.render_queue import (
            RenderQueueItem,
            _get_next_pending,
            _queue,
            _queue_lock,
        )
        with _queue_lock:
            saved = list(_queue)
            _queue.clear()
            _queue.extend([
                RenderQueueItem(id="a", input_path="/a.mp4", preset_name="p", status="complete"),
                RenderQueueItem(id="b", input_path="/b.mp4", preset_name="p", status="error"),
            ])
            result = _get_next_pending()

        try:
            assert result is None
        finally:
            with _queue_lock:
                _queue.clear()
                _queue.extend(saved)


class TestAddToQueueClamping:
    """Tests for add_to_queue priority clamping."""

    def test_priority_clamped_high(self):
        from opencut.core.render_queue import _queue, _queue_lock, add_to_queue
        with _queue_lock:
            saved = list(_queue)

        try:
            item_id = add_to_queue("/test.mp4", "preset", priority=99)
            with _queue_lock:
                item = next(i for i in _queue if i.id == item_id)
            assert item.priority == 5
        finally:
            with _queue_lock:
                _queue.clear()
                _queue.extend(saved)

    def test_priority_clamped_low(self):
        from opencut.core.render_queue import _queue, _queue_lock, add_to_queue
        with _queue_lock:
            saved = list(_queue)

        try:
            item_id = add_to_queue("/test.mp4", "preset", priority=-5)
            with _queue_lock:
                item = next(i for i in _queue if i.id == item_id)
            assert item.priority == 1
        finally:
            with _queue_lock:
                _queue.clear()
                _queue.extend(saved)


class TestQueueStateChecks:
    """Tests for is_queue_paused / is_queue_running."""

    def test_initial_state_not_paused(self):
        from opencut.core.render_queue import is_queue_paused
        # Default state after module load is not paused
        assert is_queue_paused() is False
