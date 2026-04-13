"""
Unit tests for overlay features:
- Feature 36.1: Platform Safe Zone Overlay
- Feature 44.1: Timecode Burn-In Overlay
- Feature 34.4: Countdown / Elapsed Timer Overlay

Tests pure logic functions with mocked FFmpeg subprocess calls.
"""

from unittest.mock import patch

import pytest


# ========================================================================
# 1. safe_zones.py — Safe Zone Data (pure logic, no mocks needed)
# ========================================================================
class TestSafeZones:
    """Tests for opencut.core.safe_zones — safe zone coordinate calculation."""

    def test_get_safe_zones_youtube_1920x1080(self):
        """YouTube safe zones should return correct pixel coords for 1080p."""
        from opencut.core.safe_zones import get_safe_zones
        zones = get_safe_zones("youtube", 1920, 1080)
        assert len(zones) == 2
        # End screen zone: bottom 20%
        end_screen = zones[0]
        assert end_screen.label == "End Screen Zone"
        assert end_screen.x == 0
        assert end_screen.y == 864  # 0.80 * 1080
        assert end_screen.w == 1920
        assert end_screen.h == 216  # 0.20 * 1080
        # Title bar zone: top 10%
        title = zones[1]
        assert title.label == "Title Bar Zone"
        assert title.y == 0
        assert title.h == 108  # 0.10 * 1080

    def test_get_safe_zones_tiktok(self):
        """TikTok safe zones should include caption, buttons, and top bar."""
        from opencut.core.safe_zones import get_safe_zones
        zones = get_safe_zones("tiktok", 1080, 1920)
        assert len(zones) == 3
        labels = [z.label for z in zones]
        assert "Caption Zone" in labels
        assert "Action Buttons" in labels
        assert "Top Bar Zone" in labels

    def test_get_safe_zones_instagram(self):
        """Instagram safe zones should be defined."""
        from opencut.core.safe_zones import get_safe_zones
        zones = get_safe_zones("instagram", 1080, 1920)
        assert len(zones) == 3
        labels = [z.label for z in zones]
        assert "Caption Zone" in labels
        assert "Action Buttons" in labels

    def test_get_safe_zones_twitter_alias_x(self):
        """'x' should be an alias for 'twitter'."""
        from opencut.core.safe_zones import get_safe_zones
        twitter_zones = get_safe_zones("twitter", 1920, 1080)
        x_zones = get_safe_zones("x", 1920, 1080)
        assert len(twitter_zones) == len(x_zones)
        for tz, xz in zip(twitter_zones, x_zones):
            assert tz.x == xz.x
            assert tz.y == xz.y
            assert tz.w == xz.w
            assert tz.h == xz.h

    def test_get_safe_zones_unsupported_platform(self):
        """Unsupported platform should raise ValueError."""
        from opencut.core.safe_zones import get_safe_zones
        with pytest.raises(ValueError, match="Unsupported platform"):
            get_safe_zones("myspace", 1920, 1080)

    def test_get_safe_zones_small_resolution(self):
        """Safe zones should scale to small resolutions."""
        from opencut.core.safe_zones import get_safe_zones
        zones = get_safe_zones("youtube", 320, 240)
        end_screen = zones[0]
        assert end_screen.y == 192  # 0.80 * 240
        assert end_screen.h == 48   # 0.20 * 240
        assert end_screen.w == 320

    def test_get_safe_zones_clamped_to_frame(self):
        """Zone dimensions should not exceed frame bounds."""
        from opencut.core.safe_zones import get_safe_zones
        zones = get_safe_zones("youtube", 100, 100)
        for z in zones:
            assert z.x + z.w <= 100
            assert z.y + z.h <= 100

    def test_safe_zone_dataclass_fields(self):
        """SafeZone should have all expected fields."""
        from opencut.core.safe_zones import SafeZone
        zone = SafeZone(x=10, y=20, w=100, h=50, label="Test", color="red")
        assert zone.x == 10
        assert zone.y == 20
        assert zone.w == 100
        assert zone.h == 50
        assert zone.label == "Test"
        assert zone.color == "red"

    def test_supported_platforms_list(self):
        """SUPPORTED_PLATFORMS should contain all expected platforms."""
        from opencut.core.safe_zones import SUPPORTED_PLATFORMS
        assert "youtube" in SUPPORTED_PLATFORMS
        assert "tiktok" in SUPPORTED_PLATFORMS
        assert "instagram" in SUPPORTED_PLATFORMS
        assert "twitter" in SUPPORTED_PLATFORMS
        assert "x" in SUPPORTED_PLATFORMS


# ========================================================================
# 2. safe_zones.py — generate_safe_zone_overlay (mocked FFmpeg)
# ========================================================================
class TestSafeZoneOverlay:
    """Tests for generate_safe_zone_overlay with mocked FFmpeg."""

    @patch("opencut.core.safe_zones.run_ffmpeg")
    @patch("opencut.core.safe_zones.get_video_info")
    def test_generate_overlay_youtube(self, mock_info, mock_ffmpeg):
        """Should build correct FFmpeg command for YouTube overlay."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 60.0}

        from opencut.core.safe_zones import generate_safe_zone_overlay
        result = generate_safe_zone_overlay(
            "test.mp4", "youtube", "/tmp/out.mp4", opacity=0.3
        )

        assert result["output_path"] == "/tmp/out.mp4"
        assert result["platform"] == "youtube"
        assert len(result["zones"]) == 2
        mock_ffmpeg.assert_called_once()
        cmd = mock_ffmpeg.call_args[0][0]
        # Should contain drawbox in the -vf filter
        vf_idx = cmd.index("-vf")
        vf_str = cmd[vf_idx + 1]
        assert "drawbox" in vf_str
        assert "drawtext" in vf_str

    @patch("opencut.core.safe_zones.run_ffmpeg")
    @patch("opencut.core.safe_zones.get_video_info")
    def test_generate_overlay_auto_output_path(self, mock_info, mock_ffmpeg):
        """Should auto-generate output path when not provided."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}

        from opencut.core.safe_zones import generate_safe_zone_overlay
        result = generate_safe_zone_overlay("video.mp4", "tiktok")

        assert "safezone_tiktok" in result["output_path"]

    @patch("opencut.core.safe_zones.run_ffmpeg")
    @patch("opencut.core.safe_zones.get_video_info")
    def test_generate_overlay_progress_callback(self, mock_info, mock_ffmpeg):
        """Progress callback should be invoked."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        progress_calls = []

        from opencut.core.safe_zones import generate_safe_zone_overlay
        generate_safe_zone_overlay(
            "test.mp4", "youtube", "/tmp/out.mp4",
            on_progress=lambda p, m: progress_calls.append((p, m)),
        )

        assert len(progress_calls) >= 3  # at least start, middle, end
        assert progress_calls[-1][0] == 100


# ========================================================================
# 3. overlays.py — Timecode Burn-In
# ========================================================================
class TestTimecodeBurnIn:
    """Tests for opencut.core.overlays.burn_timecode."""

    @patch("opencut.core.overlays.run_ffmpeg")
    @patch("opencut.core.overlays.get_video_info")
    def test_burn_timecode_default(self, mock_info, mock_ffmpeg):
        """Default timecode burn should produce valid FFmpeg command."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 60.0}

        from opencut.core.overlays import burn_timecode
        result = burn_timecode("test.mp4", "/tmp/tc.mp4")

        assert result["output_path"] == "/tmp/tc.mp4"
        mock_ffmpeg.assert_called_once()
        cmd = mock_ffmpeg.call_args[0][0]
        vf_idx = cmd.index("-vf")
        vf_str = cmd[vf_idx + 1]
        assert "drawtext" in vf_str
        assert "timecode" in vf_str

    @patch("opencut.core.overlays.run_ffmpeg")
    @patch("opencut.core.overlays.get_video_info")
    def test_burn_timecode_positions(self, mock_info, mock_ffmpeg):
        """Each position preset should produce a valid filter."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        from opencut.core.overlays import burn_timecode

        for pos in ["top-left", "top-right", "bottom-left", "bottom-right", "center"]:
            mock_ffmpeg.reset_mock()
            result = burn_timecode("test.mp4", f"/tmp/tc_{pos}.mp4", position=pos)
            assert result["output_path"] == f"/tmp/tc_{pos}.mp4"
            mock_ffmpeg.assert_called_once()

    @patch("opencut.core.overlays.run_ffmpeg")
    @patch("opencut.core.overlays.get_video_info")
    def test_burn_timecode_custom_start(self, mock_info, mock_ffmpeg):
        """Custom start timecode should appear in filter string."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 24.0, "duration": 10.0}

        from opencut.core.overlays import burn_timecode
        burn_timecode("test.mp4", "/tmp/tc.mp4", start_tc="01:00:00:00")

        cmd = mock_ffmpeg.call_args[0][0]
        vf_idx = cmd.index("-vf")
        vf_str = cmd[vf_idx + 1]
        assert "01\\:00\\:00\\:00" in vf_str

    def test_burn_timecode_invalid_position(self):
        """Invalid position should raise ValueError."""
        with patch("opencut.core.overlays.get_video_info") as mock_info:
            mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
            from opencut.core.overlays import burn_timecode
            with pytest.raises(ValueError, match="Invalid position"):
                burn_timecode("test.mp4", position="middle-left")

    def test_burn_timecode_invalid_tc_format(self):
        """Bad timecode format should raise ValueError."""
        with patch("opencut.core.overlays.get_video_info") as mock_info:
            mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
            from opencut.core.overlays import burn_timecode
            with pytest.raises(ValueError, match="Invalid timecode format"):
                burn_timecode("test.mp4", start_tc="01:00:00")

    @patch("opencut.core.overlays.run_ffmpeg")
    @patch("opencut.core.overlays.get_video_info")
    def test_burn_timecode_auto_output(self, mock_info, mock_ffmpeg):
        """Should auto-generate output path with _timecode suffix."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        from opencut.core.overlays import burn_timecode
        result = burn_timecode("video.mp4")
        assert "timecode" in result["output_path"]


# ========================================================================
# 4. overlays.py — Countdown Timer
# ========================================================================
class TestCountdownTimer:
    """Tests for opencut.core.overlays.burn_countdown."""

    @patch("opencut.core.overlays.run_ffmpeg")
    @patch("opencut.core.overlays.get_video_info")
    def test_burn_countdown_default(self, mock_info, mock_ffmpeg):
        """Countdown with defaults should use video duration."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 120.0}

        from opencut.core.overlays import burn_countdown
        result = burn_countdown("test.mp4", "/tmp/cd.mp4")

        assert result["output_path"] == "/tmp/cd.mp4"
        mock_ffmpeg.assert_called_once()
        cmd = mock_ffmpeg.call_args[0][0]
        vf_idx = cmd.index("-vf")
        vf_str = cmd[vf_idx + 1]
        assert "drawtext" in vf_str
        # Should contain the duration value (120.0) in the expression
        assert "120.0" in vf_str

    @patch("opencut.core.overlays.run_ffmpeg")
    @patch("opencut.core.overlays.get_video_info")
    def test_burn_countdown_custom_duration(self, mock_info, mock_ffmpeg):
        """Custom duration_seconds should override video duration."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 120.0}

        from opencut.core.overlays import burn_countdown
        burn_countdown("test.mp4", "/tmp/cd.mp4", duration_seconds=30)

        cmd = mock_ffmpeg.call_args[0][0]
        vf_idx = cmd.index("-vf")
        vf_str = cmd[vf_idx + 1]
        assert "30.0" in vf_str

    @patch("opencut.core.overlays.run_ffmpeg")
    @patch("opencut.core.overlays.get_video_info")
    def test_burn_countdown_hms_format(self, mock_info, mock_ffmpeg):
        """HH:MM:SS format should include hours in expression."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 3700.0}

        from opencut.core.overlays import burn_countdown
        burn_countdown("test.mp4", "/tmp/cd.mp4", timer_format="HH:MM:SS")

        cmd = mock_ffmpeg.call_args[0][0]
        vf_idx = cmd.index("-vf")
        vf_str = cmd[vf_idx + 1]
        # HH:MM:SS has 3600 divisor for hours
        assert "3600" in vf_str

    @patch("opencut.core.overlays.run_ffmpeg")
    @patch("opencut.core.overlays.get_video_info")
    def test_burn_countdown_auto_output(self, mock_info, mock_ffmpeg):
        """Should auto-generate output path with _countdown suffix."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 60.0}
        from opencut.core.overlays import burn_countdown
        result = burn_countdown("video.mp4")
        assert "countdown" in result["output_path"]

    @patch("opencut.core.overlays.run_ffmpeg")
    @patch("opencut.core.overlays.get_video_info")
    def test_burn_countdown_progress(self, mock_info, mock_ffmpeg):
        """Progress callback should be invoked for countdown."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        calls = []
        from opencut.core.overlays import burn_countdown
        burn_countdown("t.mp4", "/tmp/o.mp4", on_progress=lambda p, m: calls.append(p))
        assert 100 in calls


# ========================================================================
# 5. overlays.py — Elapsed Timer
# ========================================================================
class TestElapsedTimer:
    """Tests for opencut.core.overlays.burn_elapsed_timer."""

    @patch("opencut.core.overlays.run_ffmpeg")
    @patch("opencut.core.overlays.get_video_info")
    def test_burn_elapsed_default(self, mock_info, mock_ffmpeg):
        """Elapsed timer with defaults should count up from 0."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 60.0}

        from opencut.core.overlays import burn_elapsed_timer
        result = burn_elapsed_timer("test.mp4", "/tmp/el.mp4")

        assert result["output_path"] == "/tmp/el.mp4"
        mock_ffmpeg.assert_called_once()
        cmd = mock_ffmpeg.call_args[0][0]
        vf_idx = cmd.index("-vf")
        vf_str = cmd[vf_idx + 1]
        assert "drawtext" in vf_str

    @patch("opencut.core.overlays.run_ffmpeg")
    @patch("opencut.core.overlays.get_video_info")
    def test_burn_elapsed_custom_start(self, mock_info, mock_ffmpeg):
        """Custom start_seconds should appear in expression."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 60.0}

        from opencut.core.overlays import burn_elapsed_timer
        burn_elapsed_timer("test.mp4", "/tmp/el.mp4", start_seconds=90)

        cmd = mock_ffmpeg.call_args[0][0]
        vf_idx = cmd.index("-vf")
        vf_str = cmd[vf_idx + 1]
        assert "90.0" in vf_str

    @patch("opencut.core.overlays.run_ffmpeg")
    @patch("opencut.core.overlays.get_video_info")
    def test_burn_elapsed_auto_output(self, mock_info, mock_ffmpeg):
        """Should auto-generate output path with _elapsed suffix."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        from opencut.core.overlays import burn_elapsed_timer
        result = burn_elapsed_timer("video.mp4")
        assert "elapsed" in result["output_path"]

    @patch("opencut.core.overlays.run_ffmpeg")
    @patch("opencut.core.overlays.get_video_info")
    def test_burn_elapsed_position_bottom_right(self, mock_info, mock_ffmpeg):
        """Default position should be bottom-right."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        from opencut.core.overlays import burn_elapsed_timer
        burn_elapsed_timer("test.mp4", "/tmp/el.mp4")
        cmd = mock_ffmpeg.call_args[0][0]
        vf_idx = cmd.index("-vf")
        vf_str = cmd[vf_idx + 1]
        assert "w-tw-20" in vf_str
        assert "h-th-20" in vf_str

    @patch("opencut.core.overlays.run_ffmpeg")
    @patch("opencut.core.overlays.get_video_info")
    def test_burn_elapsed_progress(self, mock_info, mock_ffmpeg):
        """Progress callback should fire for elapsed timer."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        calls = []
        from opencut.core.overlays import burn_elapsed_timer
        burn_elapsed_timer("t.mp4", "/tmp/o.mp4", on_progress=lambda p, m: calls.append(p))
        assert 100 in calls


# ========================================================================
# 6. overlays.py — Position resolver
# ========================================================================
class TestPositionResolver:
    """Tests for the _resolve_position helper."""

    def test_all_valid_positions(self):
        from opencut.core.overlays import _VALID_POSITIONS, _resolve_position
        for pos in _VALID_POSITIONS:
            x, y = _resolve_position(pos)
            assert isinstance(x, str)
            assert isinstance(y, str)

    def test_invalid_position_raises(self):
        from opencut.core.overlays import _resolve_position
        with pytest.raises(ValueError, match="Invalid position"):
            _resolve_position("nowhere")

    def test_position_case_insensitive(self):
        from opencut.core.overlays import _resolve_position
        x1, y1 = _resolve_position("Top-Left")
        x2, y2 = _resolve_position("top-left")
        assert x1 == x2
        assert y1 == y2


# ========================================================================
# 7. Route smoke tests (Blueprint registration, no FFmpeg)
# ========================================================================
class TestOverlayRoutes:
    """Smoke tests for overlay route registration and validation."""

    @pytest.fixture
    def client(self):
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        app = create_app(config=OpenCutConfig())
        app.config["TESTING"] = True
        return app.test_client()

    @pytest.fixture
    def csrf(self, client):
        resp = client.get("/health")
        data = resp.get_json()
        return data.get("csrf_token", "")

    def _headers(self, token):
        return {"X-OpenCut-Token": token, "Content-Type": "application/json"}

    def test_safe_zones_data_missing_platform(self, client, csrf):
        """Should return 400 when platform is missing."""
        resp = client.post(
            "/overlay/safe-zones/data",
            json={"width": 1920, "height": 1080},
            headers=self._headers(csrf),
        )
        assert resp.status_code == 400
        assert "platform" in resp.get_json()["error"].lower()

    def test_safe_zones_data_missing_resolution(self, client, csrf):
        """Should return 400 when neither filepath nor dimensions given."""
        resp = client.post(
            "/overlay/safe-zones/data",
            json={"platform": "youtube"},
            headers=self._headers(csrf),
        )
        assert resp.status_code == 400

    def test_safe_zones_data_success(self, client, csrf):
        """Should return zones when platform + dimensions provided."""
        resp = client.post(
            "/overlay/safe-zones/data",
            json={"platform": "tiktok", "width": 1080, "height": 1920},
            headers=self._headers(csrf),
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "zones" in data
        assert len(data["zones"]) == 3
        assert data["platform"] == "tiktok"
        assert "supported_platforms" in data

    def test_safe_zones_data_unsupported_platform(self, client, csrf):
        """Should return 400 for unsupported platform."""
        resp = client.post(
            "/overlay/safe-zones/data",
            json={"platform": "myspace", "width": 1920, "height": 1080},
            headers=self._headers(csrf),
        )
        assert resp.status_code == 400
        assert "unsupported" in resp.get_json()["error"].lower()

    def test_timecode_route_no_filepath(self, client, csrf):
        """Timecode route should return 400 with no filepath."""
        resp = client.post(
            "/overlay/timecode",
            json={},
            headers=self._headers(csrf),
        )
        assert resp.status_code == 400

    def test_countdown_route_no_filepath(self, client, csrf):
        """Countdown route should return 400 with no filepath."""
        resp = client.post(
            "/overlay/countdown",
            json={},
            headers=self._headers(csrf),
        )
        assert resp.status_code == 400

    def test_elapsed_route_no_filepath(self, client, csrf):
        """Elapsed timer route should return 400 with no filepath."""
        resp = client.post(
            "/overlay/elapsed-timer",
            json={},
            headers=self._headers(csrf),
        )
        assert resp.status_code == 400

    def test_safe_zones_async_no_filepath(self, client, csrf):
        """Safe zones async route should return 400 with no filepath."""
        resp = client.post(
            "/overlay/safe-zones",
            json={"platform": "youtube"},
            headers=self._headers(csrf),
        )
        assert resp.status_code == 400
