"""
Tests for OpenCut Integration Features

Covers all 11 integration modules and their route endpoints:
  30.3 - Adjustment Layers (create, apply, presets)
  30.4 - Nested Sequence Detection (detect, create, replace)
  41.2 - Flight Path Map (parse GPS, render, overlay)
  41.3 - Aerial Hyperlapse (GPS sampling, stabilize, create)
  50.2 - Notion/PM Sync (sync, update, create)
  50.3 - Slack/Discord Notifications (send, format)
  50.4 - Zapier/Make Webhooks (send, register, inbound, list)
  39.1 - Stream Deck Integration (profiles, buttons, export)
  39.2 - MIDI Controller (mapping, save, load, devices)
  39.3 - Shuttle/Jog Wheel (mapping, devices, defaults)
  39.4 - Touch/Pen Optimization (config, gestures, pen, layout)

Uses Flask test client -- no real network, no subprocess, no GPU needed.
External dependencies (FFmpeg, Notion API, Slack/Discord) are mocked.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import csrf_headers

# =====================================================================
# 30.3 — Adjustment Layers Core Tests
# =====================================================================

class TestAdjustmentLayersCore(unittest.TestCase):
    """Tests for opencut.core.adjustment_layers module."""

    def test_build_filter_chain_brightness(self):
        from opencut.core.adjustment_layers import _build_filter_chain
        result = _build_filter_chain([{"type": "brightness", "value": 0.5}])
        assert "eq=brightness=0.500" in result

    def test_build_filter_chain_contrast(self):
        from opencut.core.adjustment_layers import _build_filter_chain
        result = _build_filter_chain([{"type": "contrast", "value": 1.2}])
        assert "eq=contrast=1.200" in result

    def test_build_filter_chain_saturation(self):
        from opencut.core.adjustment_layers import _build_filter_chain
        result = _build_filter_chain([{"type": "saturation", "value": 1.5}])
        assert "eq=saturation=1.500" in result

    def test_build_filter_chain_gamma(self):
        from opencut.core.adjustment_layers import _build_filter_chain
        result = _build_filter_chain([{"type": "gamma", "value": 2.2}])
        assert "eq=gamma=2.200" in result

    def test_build_filter_chain_hue(self):
        from opencut.core.adjustment_layers import _build_filter_chain
        result = _build_filter_chain([{"type": "hue_shift", "value": 45}])
        assert "hue=h=45.0" in result

    def test_build_filter_chain_blur(self):
        from opencut.core.adjustment_layers import _build_filter_chain
        result = _build_filter_chain([{"type": "blur", "value": 5}])
        assert "boxblur=5:5" in result

    def test_build_filter_chain_sharpen(self):
        from opencut.core.adjustment_layers import _build_filter_chain
        result = _build_filter_chain([{"type": "sharpen", "value": 1.5}])
        assert "unsharp=5:5:1.50" in result

    def test_build_filter_chain_temperature(self):
        from opencut.core.adjustment_layers import _build_filter_chain
        result = _build_filter_chain([{"type": "temperature", "value": 0.5}])
        assert "colorbalance" in result

    def test_build_filter_chain_unknown_type(self):
        from opencut.core.adjustment_layers import _build_filter_chain
        result = _build_filter_chain([{"type": "unknown_type", "value": 1}])
        assert result == "null"

    def test_build_filter_chain_empty(self):
        from opencut.core.adjustment_layers import _build_filter_chain
        result = _build_filter_chain([])
        assert result == "null"

    def test_build_filter_chain_multiple(self):
        from opencut.core.adjustment_layers import _build_filter_chain
        result = _build_filter_chain([
            {"type": "brightness", "value": 0.1},
            {"type": "contrast", "value": 1.1},
        ])
        assert "," in result
        assert "brightness" in result
        assert "contrast" in result

    def test_save_adjustment_preset(self):
        from opencut.core.adjustment_layers import save_adjustment_preset
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("opencut.core.adjustment_layers._PRESETS_DIR", tmpdir):
                result = save_adjustment_preset(
                    corrections=[{"type": "brightness", "value": 0.5}],
                    name="test_preset",
                    description="A test preset",
                )
                assert result["name"] == "test_preset"
                assert result["corrections_count"] == 1
                assert os.path.isfile(result["path"])

    def test_load_adjustment_preset(self):
        from opencut.core.adjustment_layers import (
            load_adjustment_preset,
            save_adjustment_preset,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("opencut.core.adjustment_layers._PRESETS_DIR", tmpdir):
                save_adjustment_preset(
                    corrections=[{"type": "blur", "value": 3}],
                    name="load_test",
                )
                loaded = load_adjustment_preset("load_test")
                assert loaded["name"] == "load_test"
                assert len(loaded["corrections"]) == 1

    def test_load_preset_not_found(self):
        from opencut.core.adjustment_layers import load_adjustment_preset
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("opencut.core.adjustment_layers._PRESETS_DIR", tmpdir):
                with pytest.raises(FileNotFoundError):
                    load_adjustment_preset("nonexistent")

    def test_list_adjustment_presets_empty(self):
        from opencut.core.adjustment_layers import list_adjustment_presets
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("opencut.core.adjustment_layers._PRESETS_DIR", tmpdir):
                result = list_adjustment_presets()
                assert result == []

    def test_list_adjustment_presets(self):
        from opencut.core.adjustment_layers import (
            list_adjustment_presets,
            save_adjustment_preset,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("opencut.core.adjustment_layers._PRESETS_DIR", tmpdir):
                save_adjustment_preset([{"type": "brightness", "value": 0.1}], "p1")
                save_adjustment_preset([{"type": "contrast", "value": 1.2}], "p2")
                result = list_adjustment_presets()
                assert len(result) == 2

    @patch("opencut.core.adjustment_layers.run_ffmpeg")
    def test_create_adjustment_layer(self, mock_ff):
        from opencut.core.adjustment_layers import create_adjustment_layer
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "layer.mp4")
            result = create_adjustment_layer(
                corrections=[{"type": "brightness", "value": 0.3}],
                duration=5.0,
                out_path=out,
            )
            assert result.output_path == out
            assert result.corrections_applied == 1
            assert result.duration == 5.0
            mock_ff.assert_called_once()

    @patch("opencut.core.adjustment_layers.run_ffmpeg")
    def test_apply_adjustment_to_range(self, mock_ff):
        from opencut.core.adjustment_layers import apply_adjustment_to_range
        with tempfile.TemporaryDirectory() as tmpdir:
            video = os.path.join(tmpdir, "video.mp4")
            layer = os.path.join(tmpdir, "layer.mp4")
            result = apply_adjustment_to_range(
                video_path=video,
                layer_path=layer,
                start=2.0,
                end=8.0,
            )
            assert result.output_path
            assert result.duration == 6.0
            mock_ff.assert_called_once()


# =====================================================================
# 30.4 — Nested Sequence Detection Core Tests
# =====================================================================

class TestNestedSequenceCore(unittest.TestCase):
    """Tests for opencut.core.nested_sequence module."""

    def _make_items(self, sources):
        """Helper to create timeline items from source name list."""
        return [
            {"source": s, "in": 0.0, "out": 1.0, "clip_id": f"clip_{i}"}
            for i, s in enumerate(sources)
        ]

    def test_detect_no_patterns(self):
        from opencut.core.nested_sequence import detect_repeated_patterns
        items = self._make_items(["a", "b", "c", "d"])
        result = detect_repeated_patterns(items)
        assert result == []

    def test_detect_simple_repeat(self):
        from opencut.core.nested_sequence import detect_repeated_patterns
        items = self._make_items(["a", "b", "a", "b"])
        result = detect_repeated_patterns(items, min_length=2, min_occurrences=2)
        assert len(result) >= 1
        assert result[0].occurrences >= 2

    def test_detect_triple_repeat(self):
        from opencut.core.nested_sequence import detect_repeated_patterns
        items = self._make_items(["x", "y", "x", "y", "x", "y"])
        result = detect_repeated_patterns(items, min_length=2, min_occurrences=2)
        assert len(result) >= 1
        best = result[0]
        assert best.occurrences >= 2

    def test_detect_min_length_filter(self):
        from opencut.core.nested_sequence import detect_repeated_patterns
        items = self._make_items(["a", "b", "a", "b"])
        result = detect_repeated_patterns(items, min_length=3, min_occurrences=2)
        assert result == []

    def test_detect_too_short_timeline(self):
        from opencut.core.nested_sequence import detect_repeated_patterns
        items = self._make_items(["a"])
        result = detect_repeated_patterns(items)
        assert result == []

    def test_create_nested_sequence(self):
        from opencut.core.nested_sequence import PatternMatch, create_nested_sequence
        pattern = PatternMatch(
            pattern_id="abc123",
            items=[{"source": "a", "in": 0, "out": 2.5}, {"source": "b", "in": 0, "out": 3.0}],
            length=2,
            occurrences=3,
            positions=[0, 2, 4],
        )
        seq = create_nested_sequence(pattern, name="My Sequence")
        assert seq.name == "My Sequence"
        assert seq.item_count == 2
        assert seq.duration == 5.5

    def test_create_nested_auto_name(self):
        from opencut.core.nested_sequence import PatternMatch, create_nested_sequence
        pattern = PatternMatch(pattern_id="xyz789", items=[{"source": "a", "in": 0, "out": 1}])
        seq = create_nested_sequence(pattern)
        assert "Nested_" in seq.name

    def test_replace_with_nested(self):
        from opencut.core.nested_sequence import (
            NestedSequence,
            PatternMatch,
            replace_with_nested,
        )
        items = [
            {"source": "a", "in": 0.0, "out": 1.0},
            {"source": "b", "in": 0.0, "out": 1.0},
            {"source": "c", "in": 0.0, "out": 1.0},
            {"source": "a", "in": 0.0, "out": 1.0},
            {"source": "b", "in": 0.0, "out": 1.0},
        ]
        pattern = PatternMatch(
            pattern_id="p1",
            items=[{"source": "a", "in": 0.0, "out": 1.0}, {"source": "b", "in": 0.0, "out": 1.0}],
            length=2, occurrences=2, positions=[0, 3],
        )
        nested = NestedSequence(sequence_id="p1", name="AB Sequence", duration=2.0, item_count=2)
        result = replace_with_nested(items, pattern, nested)
        assert len(result) == 3
        nested_items = [r for r in result if r.get("type") == "nested_sequence"]
        assert len(nested_items) == 2

    def test_replace_no_match(self):
        from opencut.core.nested_sequence import (
            NestedSequence,
            PatternMatch,
            replace_with_nested,
        )
        items = [{"source": "x", "in": 0.0, "out": 1.0}]
        pattern = PatternMatch(
            pattern_id="p1",
            items=[{"source": "z", "in": 0.0, "out": 1.0}],
            length=1, occurrences=1,
        )
        nested = NestedSequence(sequence_id="p1", name="Z")
        result = replace_with_nested(items, pattern, nested)
        assert len(result) == 1
        assert result[0]["source"] == "x"


# =====================================================================
# 41.2 — Flight Path Map Core Tests
# =====================================================================

class TestFlightPathMapCore(unittest.TestCase):
    """Tests for opencut.core.flight_path_map module."""

    def test_parse_dji_srt_gps(self):
        from opencut.core.flight_path_map import _parse_dji_srt_gps
        srt_content = """1
00:00:01,000 --> 00:00:02,000
[latitude: 37.7749] [longitude: -122.4194] [altitude: 100]

2
00:00:02,000 --> 00:00:03,000
[latitude: 37.7750] [longitude: -122.4195] [altitude: 105]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
            f.write(srt_content)
            f.flush()
            points = _parse_dji_srt_gps(f.name)
        os.unlink(f.name)
        assert len(points) == 2
        assert abs(points[0].latitude - 37.7749) < 0.001

    def test_parse_gpx(self):
        from opencut.core.flight_path_map import _parse_gpx
        gpx_content = """<?xml version="1.0"?>
<gpx xmlns="http://www.topografix.com/GPX/1/1">
  <trk><trkseg>
    <trkpt lat="37.7749" lon="-122.4194"><ele>100</ele></trkpt>
    <trkpt lat="37.7750" lon="-122.4195"><ele>105</ele></trkpt>
  </trkseg></trk>
</gpx>"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".gpx", delete=False) as f:
            f.write(gpx_content)
            f.flush()
            points = _parse_gpx(f.name)
        os.unlink(f.name)
        assert len(points) == 2
        assert abs(points[0].altitude - 100) < 0.1

    def test_parse_gps_track_srt(self):
        from opencut.core.flight_path_map import parse_gps_track
        srt_content = "1\n00:00:01,000 --> 00:00:02,000\n[latitude: 10.0] [longitude: 20.0]\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
            f.write(srt_content)
            f.flush()
            points = parse_gps_track(f.name)
        os.unlink(f.name)
        assert len(points) == 1

    def test_parse_gps_file_not_found(self):
        from opencut.core.flight_path_map import parse_gps_track
        with pytest.raises(FileNotFoundError):
            parse_gps_track("/nonexistent/file.srt")

    def test_haversine(self):
        from opencut.core.flight_path_map import GpsPoint, _haversine
        p1 = GpsPoint(latitude=0.0, longitude=0.0)
        p2 = GpsPoint(latitude=0.0, longitude=1.0)
        dist = _haversine(p1, p2)
        # ~111km for 1 degree at equator
        assert 110000 < dist < 112000

    def test_haversine_same_point(self):
        from opencut.core.flight_path_map import GpsPoint, _haversine
        p = GpsPoint(latitude=45.0, longitude=90.0)
        assert _haversine(p, p) == 0.0

    def test_total_distance(self):
        from opencut.core.flight_path_map import GpsPoint, _total_distance
        points = [
            GpsPoint(latitude=0.0, longitude=0.0),
            GpsPoint(latitude=0.0, longitude=0.001),
            GpsPoint(latitude=0.0, longitude=0.002),
        ]
        dist = _total_distance(points)
        assert dist > 0

    def test_total_distance_single_point(self):
        from opencut.core.flight_path_map import GpsPoint, _total_distance
        assert _total_distance([GpsPoint()]) == 0.0

    def test_get_bounds(self):
        from opencut.core.flight_path_map import GpsPoint, _get_bounds
        points = [
            GpsPoint(latitude=10.0, longitude=20.0),
            GpsPoint(latitude=12.0, longitude=22.0),
        ]
        bounds = _get_bounds(points)
        assert bounds["min_lat"] == 10.0
        assert bounds["max_lat"] == 12.0
        assert bounds["min_lon"] == 20.0
        assert bounds["max_lon"] == 22.0

    def test_project_point(self):
        from opencut.core.flight_path_map import GpsPoint, _project_point
        bounds = {"min_lat": 0, "max_lat": 10, "min_lon": 0, "max_lon": 10}
        p = GpsPoint(latitude=5.0, longitude=5.0)
        x, y = _project_point(p, bounds, 640, 480)
        # Should be roughly center
        assert 200 < x < 440
        assert 150 < y < 330


# =====================================================================
# 41.3 — Aerial Hyperlapse Core Tests
# =====================================================================

class TestAerialHyperlapseCore(unittest.TestCase):
    """Tests for opencut.core.aerial_hyperlapse module."""

    def test_haversine(self):
        from opencut.core.aerial_hyperlapse import GpsPoint, _haversine
        p1 = GpsPoint(latitude=0, longitude=0)
        p2 = GpsPoint(latitude=1, longitude=0)
        dist = _haversine(p1, p2)
        assert 110000 < dist < 112000

    def test_sample_by_gps_distance(self):
        from opencut.core.aerial_hyperlapse import sample_by_gps_distance
        # Points along equator, ~111m apart per 0.001 degrees
        points = [
            {"latitude": 0.0, "longitude": i * 0.001}
            for i in range(20)
        ]
        indices = sample_by_gps_distance(points, interval_meters=200)
        # Should select fewer than 20 points
        assert len(indices) < 20
        assert indices[0] == 0
        assert indices[-1] == 19

    def test_sample_empty(self):
        from opencut.core.aerial_hyperlapse import sample_by_gps_distance
        result = sample_by_gps_distance([], interval_meters=10)
        assert result == []

    def test_sample_single_point(self):
        from opencut.core.aerial_hyperlapse import sample_by_gps_distance
        result = sample_by_gps_distance(
            [{"latitude": 0, "longitude": 0}], interval_meters=10,
        )
        assert len(result) == 1

    @patch("opencut.core.aerial_hyperlapse.run_ffmpeg")
    def test_stabilize_aerial(self, mock_ff):
        from opencut.core.aerial_hyperlapse import stabilize_aerial
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "stab.mp4")
            result = stabilize_aerial("/fake/input.mp4", out_path=out)
            assert result == out
            assert mock_ff.call_count == 2  # Two-pass

    @patch("opencut.core.aerial_hyperlapse.run_ffmpeg")
    def test_stabilize_default_output(self, mock_ff):
        from opencut.core.aerial_hyperlapse import stabilize_aerial
        with tempfile.TemporaryDirectory() as tmpdir:
            video = os.path.join(tmpdir, "video.mp4")
            result = stabilize_aerial(video)
            assert "stabilized" in result


# =====================================================================
# 50.2 — Notion Sync Core Tests
# =====================================================================

class TestNotionSyncCore(unittest.TestCase):
    """Tests for opencut.core.notion_sync module."""

    def test_format_properties_string(self):
        from opencut.core.notion_sync import _format_properties
        result = _format_properties({"name": "Test"})
        assert result["name"]["rich_text"][0]["text"]["content"] == "Test"

    def test_format_properties_number(self):
        from opencut.core.notion_sync import _format_properties
        result = _format_properties({"count": 42})
        assert result["count"]["number"] == 42

    def test_format_properties_boolean(self):
        from opencut.core.notion_sync import _format_properties
        result = _format_properties({"done": True})
        assert result["done"]["checkbox"] is True

    def test_format_properties_list(self):
        from opencut.core.notion_sync import _format_properties
        result = _format_properties({"tags": ["a", "b"]})
        assert len(result["tags"]["multi_select"]) == 2

    def test_sync_no_api_key(self):
        from opencut.core.notion_sync import sync_to_notion
        with patch("opencut.core.notion_sync.load_notion_config", return_value={}):
            result = sync_to_notion({}, {})
            assert not result.success
            assert "API key" in result.errors[0]

    def test_sync_no_database_or_page(self):
        from opencut.core.notion_sync import sync_to_notion
        result = sync_to_notion({}, {"api_key": "secret"})
        assert not result.success
        assert any("database_id" in e for e in result.errors)

    @patch("opencut.core.notion_sync._notion_request")
    def test_sync_create_entry(self, mock_req):
        from opencut.core.notion_sync import sync_to_notion
        mock_req.return_value = {"id": "new-page-123"}
        result = sync_to_notion(
            {"name": "Project"},
            {"api_key": "key", "database_id": "db-1"},
        )
        assert result.success
        assert result.pages_created == 1
        assert "new-page-123" in result.page_ids

    @patch("opencut.core.notion_sync._notion_request")
    def test_sync_update_page(self, mock_req):
        from opencut.core.notion_sync import sync_to_notion
        mock_req.return_value = {"id": "existing-page"}
        result = sync_to_notion(
            {"status": "done"},
            {"api_key": "key", "page_id": "existing-page"},
        )
        assert result.success
        assert result.pages_updated == 1

    def test_load_save_config(self):
        from opencut.core.notion_sync import load_notion_config, save_notion_config
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "notion_config.json")
            with patch("opencut.core.notion_sync._CONFIG_FILE", config_path):
                with patch("opencut.core.notion_sync._OPENCUT_DIR", tmpdir):
                    save_notion_config({"api_key": "test123"})
                    loaded = load_notion_config()
                    assert loaded["api_key"] == "test123"


# =====================================================================
# 50.3 — Slack/Discord Notifications Core Tests
# =====================================================================

class TestSlackNotifyCore(unittest.TestCase):
    """Tests for opencut.core.slack_notify module."""

    @patch("opencut.core.slack_notify._post_webhook")
    def test_send_slack(self, mock_post):
        from opencut.core.slack_notify import NotificationResult, send_slack_notification
        mock_post.return_value = NotificationResult(success=True, platform="slack", status_code=200)
        result = send_slack_notification("https://hooks.slack.com/test", "Hello")
        assert result.success
        assert result.platform == "slack"

    @patch("opencut.core.slack_notify._post_webhook")
    def test_send_slack_with_fields(self, mock_post):
        from opencut.core.slack_notify import NotificationResult, send_slack_notification
        mock_post.return_value = NotificationResult(success=True, platform="slack")
        result = send_slack_notification(
            "https://hooks.slack.com/test", "Hi",
            fields=[{"title": "Duration", "value": "5m", "short": True}],
        )
        assert result.success

    @patch("opencut.core.slack_notify._post_webhook")
    def test_send_discord(self, mock_post):
        from opencut.core.slack_notify import NotificationResult, send_discord_notification
        mock_post.return_value = NotificationResult(success=True, platform="discord", status_code=204)
        result = send_discord_notification("https://discord.com/api/webhooks/test", "Hey")
        assert result.success
        assert result.platform == "discord"

    def test_format_job_notification_success_slack(self):
        from opencut.core.slack_notify import format_job_notification
        result = format_job_notification(
            {"job_id": "abc", "job_type": "export", "status": "complete", "duration": 12.5},
            format="slack",
        )
        assert "Complete" in result["title"]
        assert result["color"] == "#36a64f"
        assert result["fields"]

    def test_format_job_notification_error_discord(self):
        from opencut.core.slack_notify import format_job_notification
        result = format_job_notification(
            {"job_id": "xyz", "job_type": "render", "status": "error", "error": "OOM"},
            format="discord",
        )
        assert "Failed" in result["title"]
        assert result["color"] == 0xE74C3C

    def test_format_job_notification_in_progress(self):
        from opencut.core.slack_notify import format_job_notification
        result = format_job_notification(
            {"job_id": "qqq", "job_type": "transcode", "status": "running", "progress": 50},
        )
        assert "Running" in result["title"]


# =====================================================================
# 50.4 — Webhook Integrations Core Tests
# =====================================================================

class TestWebhookIntegrationsCore(unittest.TestCase):
    """Tests for opencut.core.webhook_integrations module."""

    def test_register_and_list(self):
        from opencut.core.webhook_integrations import (
            list_registered_webhooks,
            register_webhook_trigger,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            triggers_file = os.path.join(tmpdir, "triggers.json")
            with patch("opencut.core.webhook_integrations._TRIGGERS_FILE", triggers_file):
                with patch("opencut.core.webhook_integrations._OPENCUT_DIR", tmpdir):
                    register_webhook_trigger("export_complete", "https://example.com/hook")
                    hooks = list_registered_webhooks()
                    assert len(hooks) >= 1
                    assert hooks[-1]["event"] == "export_complete"

    def test_remove_webhook_trigger(self):
        from opencut.core.webhook_integrations import (
            list_registered_webhooks,
            register_webhook_trigger,
            remove_webhook_trigger,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            triggers_file = os.path.join(tmpdir, "triggers.json")
            with patch("opencut.core.webhook_integrations._TRIGGERS_FILE", triggers_file):
                with patch("opencut.core.webhook_integrations._OPENCUT_DIR", tmpdir):
                    register_webhook_trigger("test_event", "https://example.com/x")
                    removed = remove_webhook_trigger("test_event", "https://example.com/x")
                    assert removed
                    hooks = list_registered_webhooks()
                    assert len(hooks) == 0

    def test_remove_nonexistent(self):
        from opencut.core.webhook_integrations import remove_webhook_trigger
        with tempfile.TemporaryDirectory() as tmpdir:
            triggers_file = os.path.join(tmpdir, "triggers.json")
            with patch("opencut.core.webhook_integrations._TRIGGERS_FILE", triggers_file):
                assert not remove_webhook_trigger("nope", "https://nope.com")

    def test_handle_inbound_valid(self):
        from opencut.core.webhook_integrations import handle_inbound_webhook
        result = handle_inbound_webhook({"operation": "export", "params": {"format": "mp4"}})
        assert result["accepted"]
        assert result["operation"] == "export"

    def test_handle_inbound_missing_operation(self):
        from opencut.core.webhook_integrations import handle_inbound_webhook
        result = handle_inbound_webhook({})
        assert not result["accepted"]
        assert "supported_operations" in result

    def test_handle_inbound_unknown_operation(self):
        from opencut.core.webhook_integrations import handle_inbound_webhook
        result = handle_inbound_webhook({"operation": "fly_to_moon"})
        assert not result["accepted"]

    @patch("opencut.core.webhook_integrations.urllib.request.urlopen")
    def test_send_webhook_success(self, mock_urlopen):
        from opencut.core.webhook_integrations import send_webhook
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = b"ok"
        mock_urlopen.return_value = mock_resp
        result = send_webhook("https://example.com", "test", {"key": "val"})
        assert result.success


# =====================================================================
# 39.1 — Stream Deck Core Tests
# =====================================================================

class TestStreamDeckCore(unittest.TestCase):
    """Tests for opencut.core.stream_deck module."""

    def test_get_default_profile_editing(self):
        from opencut.core.stream_deck import get_stream_deck_profile
        profile = get_stream_deck_profile("editing")
        assert profile["name"] == "editing"
        assert len(profile["buttons"]) > 0

    def test_get_default_profile_color(self):
        from opencut.core.stream_deck import get_stream_deck_profile
        profile = get_stream_deck_profile("color_grading")
        assert profile["name"] == "color_grading"

    def test_get_profile_not_found(self):
        from opencut.core.stream_deck import get_stream_deck_profile
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("opencut.core.stream_deck._PROFILES_DIR", tmpdir):
                with pytest.raises(FileNotFoundError):
                    get_stream_deck_profile("nonexistent_xyz_profile")

    def test_create_button_mapping(self):
        from opencut.core.stream_deck import create_button_mapping
        result = create_button_mapping(button_id=5, operation="export", label="Export")
        assert result["button_id"] == 5
        assert result["operation"] == "export"
        assert result["label"] == "Export"

    def test_create_button_auto_label(self):
        from opencut.core.stream_deck import create_button_mapping
        result = create_button_mapping(button_id=0, operation="silence_remove")
        assert result["label"] == "Silence Remove"

    def test_list_profiles(self):
        from opencut.core.stream_deck import list_profiles
        profiles = list_profiles()
        names = [p["name"] for p in profiles]
        assert "editing" in names
        assert "audio" in names

    def test_save_and_load_profile(self):
        from opencut.core.stream_deck import get_stream_deck_profile, save_profile
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("opencut.core.stream_deck._PROFILES_DIR", tmpdir):
                save_profile({"name": "custom_test", "buttons": [
                    {"button_id": 0, "operation": "play_pause"},
                ]})
                profile = get_stream_deck_profile("custom_test")
                assert profile["name"] == "custom_test"

    def test_export_profile(self):
        from opencut.core.stream_deck import export_stream_deck_profile
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "export.json")
            path = export_stream_deck_profile({"name": "test", "buttons": []}, out)
            assert os.path.isfile(path)
            with open(path) as f:
                data = json.load(f)
            assert data["name"] == "test"


# =====================================================================
# 39.2 — MIDI Controller Core Tests
# =====================================================================

class TestMidiControllerCore(unittest.TestCase):
    """Tests for opencut.core.midi_controller module."""

    def test_create_midi_mapping(self):
        from opencut.core.midi_controller import create_midi_mapping
        result = create_midi_mapping(cc_channel=1, parameter="volume", cc_number=7)
        assert result["channel"] == 1
        assert result["parameter"] == "volume"
        assert result["cc_number"] == 7

    def test_create_mapping_with_range(self):
        from opencut.core.midi_controller import create_midi_mapping
        result = create_midi_mapping(
            cc_channel=0, parameter="brightness",
            range_config={"min": 0.0, "max": 2.0, "invert": True},
        )
        assert result["min_value"] == 0.0
        assert result["max_value"] == 2.0
        assert result["invert"] is True

    def test_save_and_load_midi_map(self):
        from opencut.core.midi_controller import load_midi_map, save_midi_map
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "test.json")
            mappings = [
                {"channel": 0, "cc_number": 1, "parameter": "volume"},
                {"channel": 0, "cc_number": 7, "parameter": "pan"},
            ]
            save_midi_map(mappings, out, name="Test Map")
            loaded = load_midi_map(out)
            assert loaded["name"] == "Test Map"
            assert len(loaded["mappings"]) == 2

    def test_load_not_found(self):
        from opencut.core.midi_controller import load_midi_map
        with pytest.raises(FileNotFoundError):
            load_midi_map("/nonexistent/midi.json")

    def test_list_midi_devices(self):
        from opencut.core.midi_controller import list_midi_devices
        devices = list_midi_devices()
        assert len(devices) >= 4
        types = [d["type"] for d in devices]
        assert "generic" in types

    def test_midi_mapping_channel_clamp(self):
        from opencut.core.midi_controller import create_midi_mapping
        result = create_midi_mapping(cc_channel=20, parameter="test")
        assert result["channel"] == 15  # Clamped

    def test_midi_mapping_cc_clamp(self):
        from opencut.core.midi_controller import create_midi_mapping
        result = create_midi_mapping(cc_channel=0, parameter="test", cc_number=200)
        assert result["cc_number"] == 127  # Clamped


# =====================================================================
# 39.3 — Jog Wheel Core Tests
# =====================================================================

class TestJogWheelCore(unittest.TestCase):
    """Tests for opencut.core.jog_wheel module."""

    def test_list_supported_devices(self):
        from opencut.core.jog_wheel import list_supported_devices
        devices = list_supported_devices()
        assert len(devices) >= 3
        types = [d["type"] for d in devices]
        assert "shuttlepro_v2" in types

    def test_get_default_mapping_shuttlepro(self):
        from opencut.core.jog_wheel import get_default_mapping
        mapping = get_default_mapping("shuttlepro_v2")
        assert mapping["device_type"] == "shuttlepro_v2"
        assert len(mapping["actions"]) > 0

    def test_get_default_mapping_invalid(self):
        from opencut.core.jog_wheel import get_default_mapping
        with pytest.raises(ValueError):
            get_default_mapping("nonexistent_device")

    def test_create_jog_mapping(self):
        from opencut.core.jog_wheel import create_jog_mapping
        result = create_jog_mapping("generic_hid", [
            {"input_type": "jog", "input_id": 0, "action": "seek"},
            {"input_type": "button", "input_id": 0, "action": "play"},
        ])
        assert result["device_type"] == "generic_hid"
        assert len(result["actions"]) == 2

    def test_create_jog_mapping_auto_name(self):
        from opencut.core.jog_wheel import create_jog_mapping
        result = create_jog_mapping("generic_hid", [])
        assert "generic_hid" in result["name"]

    def test_save_and_load_jog_mapping(self):
        from opencut.core.jog_wheel import load_jog_mapping, save_jog_mapping
        with tempfile.TemporaryDirectory() as tmpdir:
            mapping = {"name": "test_jog", "device_type": "generic_hid", "actions": []}
            path = save_jog_mapping(mapping, os.path.join(tmpdir, "jog.json"))
            loaded = load_jog_mapping(path)
            assert loaded["name"] == "test_jog"

    def test_load_jog_not_found(self):
        from opencut.core.jog_wheel import load_jog_mapping
        with pytest.raises(FileNotFoundError):
            load_jog_mapping("/nonexistent/jog.json")

    def test_default_mapping_shuttlexpress(self):
        from opencut.core.jog_wheel import get_default_mapping
        mapping = get_default_mapping("shuttlexpress")
        assert mapping["device_type"] == "shuttlexpress"
        actions = mapping["actions"]
        jog_actions = [a for a in actions if a["input_type"] == "jog"]
        assert len(jog_actions) >= 1


# =====================================================================
# 39.4 — Touch Optimization Core Tests
# =====================================================================

class TestTouchOptimizeCore(unittest.TestCase):
    """Tests for opencut.core.touch_optimize module."""

    def test_get_touch_config_defaults(self):
        from opencut.core.touch_optimize import get_touch_config
        with patch("opencut.core.touch_optimize._TOUCH_CONFIG_FILE", "/nonexistent"):
            config = get_touch_config()
            assert config["min_target_size"] == 44
            assert config["touch_enabled"] is True
            assert len(config["gestures"]) > 0

    def test_save_and_load_config(self):
        from opencut.core.touch_optimize import get_touch_config, save_touch_config
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = os.path.join(tmpdir, "touch.json")
            with patch("opencut.core.touch_optimize._TOUCH_CONFIG_FILE", config_file):
                with patch("opencut.core.touch_optimize._OPENCUT_DIR", tmpdir):
                    save_touch_config({"min_target_size": 48, "touch_enabled": True})
                    loaded = get_touch_config()
                    assert loaded["min_target_size"] == 48

    def test_create_gesture_mapping(self):
        from opencut.core.touch_optimize import create_gesture_mapping
        result = create_gesture_mapping("pinch_zoom", "timeline_zoom")
        assert result["gesture"] == "pinch_zoom"
        assert result["action"] == "timeline_zoom"
        assert result["enabled"] is True

    def test_create_gesture_with_params(self):
        from opencut.core.touch_optimize import create_gesture_mapping
        result = create_gesture_mapping("swipe", "undo", params={"direction": "left"})
        assert result["params"]["direction"] == "left"

    def test_get_pen_pressure_config_defaults(self):
        from opencut.core.touch_optimize import get_pen_pressure_config
        with patch("opencut.core.touch_optimize._TOUCH_CONFIG_FILE", "/nonexistent"):
            config = get_pen_pressure_config()
            assert config["enabled"] is True
            assert config["curve_type"] == "linear"
            assert config["brush_size_min"] == 1.0

    def test_optimize_layout_for_touch(self):
        from opencut.core.touch_optimize import optimize_layout_for_touch
        layout = {
            "elements": [
                {"type": "button", "id": "btn1", "width": 30, "height": 30},
                {"type": "slider_handle", "id": "sl1", "width": 20, "height": 20},
            ],
            "timeline": {"track_height": 30},
        }
        with patch("opencut.core.touch_optimize._TOUCH_CONFIG_FILE", "/nonexistent"):
            result = optimize_layout_for_touch(layout)
            assert result["touch_optimized"] is True
            assert result["total_adjustments"] > 0
            # Check elements were resized
            elems = result["layout"]["elements"]
            for elem in elems:
                assert elem["width"] >= 44
                assert elem["height"] >= 44

    def test_optimize_layout_empty(self):
        from opencut.core.touch_optimize import optimize_layout_for_touch
        with patch("opencut.core.touch_optimize._TOUCH_CONFIG_FILE", "/nonexistent"):
            result = optimize_layout_for_touch({})
            assert result["touch_optimized"] is True


# =====================================================================
# Route Smoke Tests — Integration Blueprint
# =====================================================================

class TestIntegrationRoutesSmoke(unittest.TestCase):
    """Route smoke tests for all integration endpoints."""

    @pytest.fixture(autouse=True)
    def _setup_client(self, client, csrf_token):
        self.client = client
        self.token = csrf_token
        self.headers = csrf_headers(csrf_token)

    # --- Adjustment Layers ---

    def test_create_adjustment_layer_missing_corrections(self):
        resp = self.client.post("/api/adjustment-layers/create",
                                headers=self.headers,
                                data=json.dumps({}),
                                content_type="application/json")
        assert resp.status_code == 400

    def test_list_presets(self):
        resp = self.client.get("/api/adjustment-layers/presets")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "presets" in data

    def test_save_preset_missing_name(self):
        resp = self.client.post("/api/adjustment-layers/presets",
                                headers=self.headers,
                                data=json.dumps({"corrections": []}),
                                content_type="application/json")
        assert resp.status_code == 400

    # --- Nested Sequences ---

    def test_detect_patterns_missing_items(self):
        resp = self.client.post("/api/timeline/detect-patterns",
                                headers=self.headers,
                                data=json.dumps({}),
                                content_type="application/json")
        assert resp.status_code == 400

    def test_detect_patterns_valid(self):
        items = [
            {"source": "a", "in": 0, "out": 1},
            {"source": "b", "in": 0, "out": 1},
            {"source": "a", "in": 0, "out": 1},
            {"source": "b", "in": 0, "out": 1},
        ]
        resp = self.client.post("/api/timeline/detect-patterns",
                                headers=self.headers,
                                data=json.dumps({"timeline_items": items}),
                                content_type="application/json")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "patterns" in data

    def test_create_nested_missing_pattern(self):
        resp = self.client.post("/api/timeline/create-nested",
                                headers=self.headers,
                                data=json.dumps({}),
                                content_type="application/json")
        assert resp.status_code == 400

    def test_replace_nested_missing_data(self):
        resp = self.client.post("/api/timeline/replace-nested",
                                headers=self.headers,
                                data=json.dumps({}),
                                content_type="application/json")
        assert resp.status_code == 400

    # --- GPS / Drone ---

    def test_parse_gps_missing_path(self):
        resp = self.client.post("/api/drone/parse-gps",
                                headers=self.headers,
                                data=json.dumps({}),
                                content_type="application/json")
        assert resp.status_code == 400

    def test_gps_sample_missing_points(self):
        resp = self.client.post("/api/drone/gps-sample",
                                headers=self.headers,
                                data=json.dumps({}),
                                content_type="application/json")
        assert resp.status_code == 400

    def test_gps_sample_valid(self):
        points = [
            {"latitude": 0.0, "longitude": i * 0.001}
            for i in range(10)
        ]
        resp = self.client.post("/api/drone/gps-sample",
                                headers=self.headers,
                                data=json.dumps({
                                    "gps_points": points,
                                    "interval_meters": 50,
                                }),
                                content_type="application/json")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "selected_indices" in data

    # --- Notion ---

    def test_notion_sync_route(self):
        with patch("opencut.core.notion_sync.sync_to_notion") as mock_sync:
            from opencut.core.notion_sync import NotionSyncResult
            mock_sync.return_value = NotionSyncResult(success=True, pages_created=1)
            resp = self.client.post("/api/integrations/notion/sync",
                                    headers=self.headers,
                                    data=json.dumps({"project_data": {}, "config": {}}),
                                    content_type="application/json")
            assert resp.status_code == 200

    def test_notion_update_missing_fields(self):
        resp = self.client.patch("/api/integrations/notion/page",
                                 headers=self.headers,
                                 data=json.dumps({}),
                                 content_type="application/json")
        assert resp.status_code == 400

    def test_notion_create_missing_fields(self):
        resp = self.client.post("/api/integrations/notion/entry",
                                headers=self.headers,
                                data=json.dumps({}),
                                content_type="application/json")
        assert resp.status_code == 400

    # --- Slack / Discord ---

    def test_slack_missing_fields(self):
        resp = self.client.post("/api/integrations/slack/send",
                                headers=self.headers,
                                data=json.dumps({}),
                                content_type="application/json")
        assert resp.status_code == 400

    def test_discord_missing_fields(self):
        resp = self.client.post("/api/integrations/discord/send",
                                headers=self.headers,
                                data=json.dumps({}),
                                content_type="application/json")
        assert resp.status_code == 400

    def test_format_notification_route(self):
        resp = self.client.post("/api/integrations/notify/format",
                                headers=self.headers,
                                data=json.dumps({
                                    "job_data": {"job_id": "x", "status": "complete", "job_type": "export"},
                                    "format": "slack",
                                }),
                                content_type="application/json")
        assert resp.status_code == 200
        assert "title" in resp.get_json()

    # --- Webhooks ---

    def test_webhook_send_missing_fields(self):
        resp = self.client.post("/api/integrations/webhooks/send",
                                headers=self.headers,
                                data=json.dumps({}),
                                content_type="application/json")
        assert resp.status_code == 400

    def test_webhook_register_missing_fields(self):
        resp = self.client.post("/api/integrations/webhooks/register",
                                headers=self.headers,
                                data=json.dumps({}),
                                content_type="application/json")
        assert resp.status_code == 400

    def test_webhook_list(self):
        resp = self.client.get("/api/integrations/webhooks")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "webhooks" in data

    def test_webhook_inbound_valid(self):
        resp = self.client.post("/api/integrations/webhooks/inbound",
                                headers=self.headers,
                                data=json.dumps({"operation": "export"}),
                                content_type="application/json")
        assert resp.status_code == 200
        assert resp.get_json()["accepted"]

    def test_webhook_inbound_invalid(self):
        resp = self.client.post("/api/integrations/webhooks/inbound",
                                headers=self.headers,
                                data=json.dumps({"operation": "fly_rocket"}),
                                content_type="application/json")
        assert resp.status_code == 400

    # --- Stream Deck ---

    def test_stream_deck_list_profiles(self):
        resp = self.client.get("/api/hardware/stream-deck/profiles")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "profiles" in data
        assert data["count"] >= 3

    def test_stream_deck_get_profile(self):
        resp = self.client.get("/api/hardware/stream-deck/profile?name=editing")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["name"] == "editing"

    def test_stream_deck_get_unknown(self):
        resp = self.client.get("/api/hardware/stream-deck/profile?name=zzz_nonexistent")
        assert resp.status_code in (400, 404, 500)

    def test_stream_deck_button_missing_op(self):
        resp = self.client.post("/api/hardware/stream-deck/button",
                                headers=self.headers,
                                data=json.dumps({"button_id": 0}),
                                content_type="application/json")
        assert resp.status_code == 400

    def test_stream_deck_button_valid(self):
        resp = self.client.post("/api/hardware/stream-deck/button",
                                headers=self.headers,
                                data=json.dumps({"button_id": 0, "operation": "cut"}),
                                content_type="application/json")
        assert resp.status_code == 200
        assert resp.get_json()["operation"] == "cut"

    def test_stream_deck_export_missing_path(self):
        resp = self.client.post("/api/hardware/stream-deck/export",
                                headers=self.headers,
                                data=json.dumps({"profile": {}}),
                                content_type="application/json")
        assert resp.status_code == 400

    # --- MIDI ---

    def test_midi_devices(self):
        resp = self.client.get("/api/hardware/midi/devices")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] >= 4

    def test_midi_mapping_missing_param(self):
        resp = self.client.post("/api/hardware/midi/mapping",
                                headers=self.headers,
                                data=json.dumps({"channel": 0}),
                                content_type="application/json")
        assert resp.status_code == 400

    def test_midi_mapping_valid(self):
        resp = self.client.post("/api/hardware/midi/mapping",
                                headers=self.headers,
                                data=json.dumps({
                                    "channel": 1,
                                    "parameter": "volume",
                                    "cc_number": 7,
                                }),
                                content_type="application/json")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["parameter"] == "volume"

    def test_midi_save_missing_path(self):
        resp = self.client.post("/api/hardware/midi/save",
                                headers=self.headers,
                                data=json.dumps({"mappings": []}),
                                content_type="application/json")
        assert resp.status_code == 400

    def test_midi_load_missing_path(self):
        resp = self.client.post("/api/hardware/midi/load",
                                headers=self.headers,
                                data=json.dumps({}),
                                content_type="application/json")
        assert resp.status_code == 400

    # --- Jog Wheel ---

    def test_jog_devices(self):
        resp = self.client.get("/api/hardware/jog/devices")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] >= 3

    def test_jog_default(self):
        resp = self.client.get("/api/hardware/jog/default?device_type=shuttlepro_v2")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["device_type"] == "shuttlepro_v2"

    def test_jog_default_invalid(self):
        resp = self.client.get("/api/hardware/jog/default?device_type=nonexistent")
        assert resp.status_code in (400, 500)

    def test_jog_mapping_create(self):
        resp = self.client.post("/api/hardware/jog/mapping",
                                headers=self.headers,
                                data=json.dumps({
                                    "device_type": "generic_hid",
                                    "mappings": [{"input_type": "jog", "input_id": 0, "action": "seek"}],
                                }),
                                content_type="application/json")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["device_type"] == "generic_hid"

    # --- Touch/Pen ---

    def test_touch_config_get(self):
        resp = self.client.get("/api/hardware/touch/config")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "min_target_size" in data
        assert "gestures" in data

    def test_touch_pen_get(self):
        resp = self.client.get("/api/hardware/touch/pen")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "enabled" in data
        assert "curve_type" in data

    def test_touch_gesture_missing_fields(self):
        resp = self.client.post("/api/hardware/touch/gesture",
                                headers=self.headers,
                                data=json.dumps({}),
                                content_type="application/json")
        assert resp.status_code == 400

    def test_touch_gesture_valid(self):
        resp = self.client.post("/api/hardware/touch/gesture",
                                headers=self.headers,
                                data=json.dumps({"gesture": "pinch", "action": "zoom"}),
                                content_type="application/json")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["gesture"] == "pinch"

    def test_touch_optimize_layout(self):
        resp = self.client.post("/api/hardware/touch/optimize-layout",
                                headers=self.headers,
                                data=json.dumps({
                                    "layout": {"elements": [
                                        {"type": "button", "width": 20, "height": 20},
                                    ]},
                                }),
                                content_type="application/json")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["touch_optimized"] is True
