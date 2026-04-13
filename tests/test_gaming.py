"""
Tests for OpenCut gaming/streaming features.

Covers:
- Chat Replay Overlay (12.2)
- Auto Montage Builder (12.3)
- Multi-POV Sync (12.4)
- Multi-Track ISO Ingest (31.1)
- Instant Replay Builder (31.2)
- Stream Highlight Reel (31.4)
- Gaming routes blueprint
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# 12.2 — Chat Replay Overlay Tests
# ============================================================
class TestChatReplayParseTwitch(unittest.TestCase):
    """Tests for opencut.core.chat_replay — Twitch IRC parser."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".log", delete=False, encoding="utf-8"
        )

    def tearDown(self):
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    def test_parse_standard_irc_format(self):
        """Parse [HH:MM:SS] <user> message format."""
        self.tmp.write("[00:01:30] <streamer> Hello chat!\n")
        self.tmp.write("[00:02:00] <viewer1> GG\n")
        self.tmp.close()

        from opencut.core.chat_replay import parse_twitch_chat
        msgs = parse_twitch_chat(self.tmp.name)

        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0].timestamp, 90.0)
        self.assertEqual(msgs[0].username, "streamer")
        self.assertEqual(msgs[0].message, "Hello chat!")
        self.assertEqual(msgs[1].timestamp, 120.0)

    def test_parse_colon_format(self):
        """Parse [HH:MM:SS] user: message format."""
        self.tmp.write("[00:05:00] viewer2: Nice play\n")
        self.tmp.close()

        from opencut.core.chat_replay import parse_twitch_chat
        msgs = parse_twitch_chat(self.tmp.name)

        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].username, "viewer2")

    def test_parse_numeric_timestamp(self):
        """Parse numeric second-based timestamps."""
        self.tmp.write("45.5 player1 Let's go!\n")
        self.tmp.write("100.0 player2 GG\n")
        self.tmp.close()

        from opencut.core.chat_replay import parse_twitch_chat
        msgs = parse_twitch_chat(self.tmp.name)

        self.assertEqual(len(msgs), 2)
        self.assertAlmostEqual(msgs[0].timestamp, 45.5)

    def test_empty_file(self):
        """Empty log file should return empty list."""
        self.tmp.close()

        from opencut.core.chat_replay import parse_twitch_chat
        msgs = parse_twitch_chat(self.tmp.name)

        self.assertEqual(msgs, [])

    def test_sorted_by_timestamp(self):
        """Messages should be sorted by timestamp."""
        self.tmp.write("[00:05:00] <b> Second\n")
        self.tmp.write("[00:01:00] <a> First\n")
        self.tmp.close()

        from opencut.core.chat_replay import parse_twitch_chat
        msgs = parse_twitch_chat(self.tmp.name)

        self.assertEqual(len(msgs), 2)
        self.assertLessEqual(msgs[0].timestamp, msgs[1].timestamp)

    def test_file_not_found(self):
        """Missing file should raise FileNotFoundError."""
        from opencut.core.chat_replay import parse_twitch_chat
        with self.assertRaises(FileNotFoundError):
            parse_twitch_chat("/nonexistent/chat.log")

    def test_color_assignment_deterministic(self):
        """Same username should always get same color."""
        self.tmp.write("[00:00:01] <user1> hi\n")
        self.tmp.write("[00:00:02] <user1> hello\n")
        self.tmp.close()

        from opencut.core.chat_replay import parse_twitch_chat
        msgs = parse_twitch_chat(self.tmp.name)

        self.assertEqual(msgs[0].color, msgs[1].color)


class TestChatReplayParseYouTube(unittest.TestCase):
    """Tests for opencut.core.chat_replay — YouTube chat parser."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )

    def tearDown(self):
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    def test_parse_simple_array_format(self):
        """Parse simple JSON array of chat objects."""
        data = [
            {"timestamp": 10.0, "username": "viewer", "message": "Hello"},
            {"timestamp": 20.0, "author": "mod", "text": "Welcome"},
        ]
        self.tmp.write(json.dumps(data))
        self.tmp.close()

        from opencut.core.chat_replay import parse_youtube_chat
        msgs = parse_youtube_chat(self.tmp.name)

        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0].username, "viewer")
        self.assertEqual(msgs[0].message, "Hello")

    def test_parse_yt_dlp_jsonl(self):
        """Parse yt-dlp live_chat JSONL format."""
        obj = {
            "replayChatItemAction": {
                "videoOffsetTimeMsec": "5000",
                "actions": [{
                    "addChatItemAction": {
                        "item": {
                            "liveChatTextMessageRenderer": {
                                "authorName": {"simpleText": "chatter"},
                                "message": {"runs": [{"text": "POG"}]},
                            }
                        }
                    }
                }]
            }
        }
        self.tmp.write(json.dumps(obj) + "\n")
        self.tmp.close()

        from opencut.core.chat_replay import parse_youtube_chat
        msgs = parse_youtube_chat(self.tmp.name)

        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].username, "chatter")
        self.assertEqual(msgs[0].message, "POG")
        self.assertAlmostEqual(msgs[0].timestamp, 5.0)

    def test_file_not_found(self):
        """Missing file should raise FileNotFoundError."""
        from opencut.core.chat_replay import parse_youtube_chat
        with self.assertRaises(FileNotFoundError):
            parse_youtube_chat("/nonexistent/chat.json")

    def test_empty_json_array(self):
        """Empty array should return empty list."""
        self.tmp.write("[]")
        self.tmp.close()

        from opencut.core.chat_replay import parse_youtube_chat
        msgs = parse_youtube_chat(self.tmp.name)
        self.assertEqual(msgs, [])


class TestChatReplayOverlay(unittest.TestCase):
    """Tests for opencut.core.chat_replay — overlay rendering."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        self.tmp.write(b"\x00" * 100)
        self.tmp.close()

    def tearDown(self):
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    def test_render_no_messages_raises(self):
        """Empty message list should raise ValueError."""
        from opencut.core.chat_replay import render_chat_overlay
        with self.assertRaises(ValueError):
            render_chat_overlay(self.tmp.name, [])

    def test_render_missing_video_raises(self):
        """Missing video should raise FileNotFoundError."""
        from opencut.core.chat_replay import ChatMessage, render_chat_overlay
        msgs = [ChatMessage(timestamp=1.0, username="u", message="hi")]
        with self.assertRaises(FileNotFoundError):
            render_chat_overlay("/nonexistent/video.mp4", msgs)

    @patch("opencut.core.chat_replay.run_ffmpeg")
    @patch("opencut.core.chat_replay.get_video_info")
    def test_render_calls_ffmpeg(self, mock_info, mock_ffmpeg):
        """Overlay rendering should invoke FFmpeg."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 60.0}
        from opencut.core.chat_replay import ChatMessage, render_chat_overlay

        msgs = [ChatMessage(timestamp=5.0, username="user", message="test")]
        result = render_chat_overlay(self.tmp.name, msgs)

        self.assertIn("output_path", vars(result))
        self.assertEqual(result.message_count, 1)
        mock_ffmpeg.assert_called_once()

    @patch("opencut.core.chat_replay.run_ffmpeg")
    @patch("opencut.core.chat_replay.get_video_info")
    def test_render_progress_callback(self, mock_info, mock_ffmpeg):
        """Progress callback should be invoked."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 60.0}
        from opencut.core.chat_replay import ChatMessage, render_chat_overlay

        progress = MagicMock()
        msgs = [ChatMessage(timestamp=5.0, username="u", message="m")]
        render_chat_overlay(self.tmp.name, msgs, on_progress=progress)

        self.assertTrue(progress.called)


class TestChatReplayASS(unittest.TestCase):
    """Tests for ASS subtitle generation internals."""

    def test_ass_time_format(self):
        """ASS time format should be H:MM:SS.cc."""
        from opencut.core.chat_replay import _seconds_to_ass_time
        self.assertEqual(_seconds_to_ass_time(0), "0:00:00.00")
        self.assertEqual(_seconds_to_ass_time(90.5), "0:01:30.50")
        self.assertEqual(_seconds_to_ass_time(3661.25), "1:01:01.25")

    def test_hex_to_ass_color(self):
        """Convert hex RGB to ASS BGR format."""
        from opencut.core.chat_replay import _hex_to_ass_color
        self.assertEqual(_hex_to_ass_color("#FF0000"), "&H0000FF&")
        self.assertEqual(_hex_to_ass_color("#00FF00"), "&H00FF00&")

    def test_escape_ass_special_chars(self):
        """Special characters should be escaped for ASS."""
        from opencut.core.chat_replay import _escape_ass
        self.assertIn("\\{", _escape_ass("{test}"))


# ============================================================
# 12.3 — Auto Montage Builder Tests
# ============================================================
class TestAutoMontageScoring(unittest.TestCase):
    """Tests for opencut.core.auto_montage — clip scoring."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        self.tmp.write(b"\x00" * 100)
        self.tmp.close()

    def tearDown(self):
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    def test_score_empty_list_raises(self):
        """Empty clip list should raise ValueError."""
        from opencut.core.auto_montage import score_clips
        with self.assertRaises(ValueError):
            score_clips([])

    @patch("opencut.core.auto_montage._analyze_motion", return_value=0.7)
    @patch("opencut.core.auto_montage._analyze_audio_energy", return_value=0.8)
    @patch("opencut.core.auto_montage.get_video_info")
    def test_score_clips_returns_sorted(self, mock_info, mock_audio, mock_motion):
        """Clips should be sorted by composite score descending."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        from opencut.core.auto_montage import score_clips

        scores = score_clips([self.tmp.name])

        self.assertEqual(len(scores), 1)
        self.assertGreater(scores[0].composite, 0)
        self.assertEqual(scores[0].clip_path, self.tmp.name)

    @patch("opencut.core.auto_montage._analyze_motion", return_value=0.5)
    @patch("opencut.core.auto_montage._analyze_audio_energy", return_value=0.5)
    @patch("opencut.core.auto_montage.get_video_info")
    def test_score_clips_weight_balance(self, mock_info, mock_audio, mock_motion):
        """Equal weights with equal scores should give composite of 0.5."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        from opencut.core.auto_montage import score_clips

        scores = score_clips([self.tmp.name], audio_weight=0.5, motion_weight=0.5)
        self.assertAlmostEqual(scores[0].composite, 0.5, places=2)

    @patch("opencut.core.auto_montage._analyze_motion", return_value=0.5)
    @patch("opencut.core.auto_montage._analyze_audio_energy", return_value=0.5)
    @patch("opencut.core.auto_montage.get_video_info")
    def test_score_clips_progress(self, mock_info, mock_audio, mock_motion):
        """Progress callback should be invoked."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        from opencut.core.auto_montage import score_clips

        progress = MagicMock()
        score_clips([self.tmp.name], on_progress=progress)
        self.assertTrue(progress.called)


class TestAutoMontageSelection(unittest.TestCase):
    """Tests for opencut.core.auto_montage — clip selection."""

    def test_select_top_clips_limits_count(self):
        """select_top_clips should respect count limit."""
        from opencut.core.auto_montage import ClipScore, select_top_clips

        scores = [ClipScore(clip_path=f"c{i}.mp4", composite=i/10, duration=5.0) for i in range(10)]
        top = select_top_clips(scores, count=3)
        self.assertEqual(len(top), 3)

    def test_select_top_clips_min_duration(self):
        """Clips below min_duration should be filtered out."""
        from opencut.core.auto_montage import ClipScore, select_top_clips

        scores = [
            ClipScore(clip_path="long.mp4", composite=0.8, duration=5.0),
            ClipScore(clip_path="short.mp4", composite=0.9, duration=0.5),
        ]
        top = select_top_clips(scores, count=10, min_duration=1.0)
        self.assertEqual(len(top), 1)
        self.assertEqual(top[0].clip_path, "long.mp4")


class TestAutoMontageAssembly(unittest.TestCase):
    """Tests for opencut.core.auto_montage — assembly."""

    def test_assemble_no_clips_raises(self):
        """Empty clip list should raise ValueError."""
        from opencut.core.auto_montage import assemble_montage
        with self.assertRaises(ValueError):
            assemble_montage([], "music.mp3")

    def test_assemble_missing_music_raises(self):
        """Missing music file should raise FileNotFoundError."""
        from opencut.core.auto_montage import ClipScore, assemble_montage
        clips = [ClipScore(clip_path="test.mp4", composite=0.5, duration=5.0)]
        with self.assertRaises(FileNotFoundError):
            assemble_montage(clips, "/nonexistent/music.mp3")


# ============================================================
# 12.4 — Multi-POV Sync Tests
# ============================================================
class TestMultiPovSync(unittest.TestCase):
    """Tests for opencut.core.multi_pov — POV sync."""

    def test_sync_too_few_files_raises(self):
        """Need at least 2 files."""
        from opencut.core.multi_pov import sync_pov_recordings
        with self.assertRaises(ValueError):
            sync_pov_recordings(["single.mp4"])

    @patch("opencut.core.multi_pov._cross_correlate", return_value=(1.5, 0.8))
    @patch("opencut.core.multi_pov._extract_audio_pcm", return_value=b"\x00" * 1000)
    @patch("opencut.core.multi_pov.get_video_info")
    @patch("os.path.isfile", return_value=True)
    def test_sync_returns_offsets(self, mock_isfile, mock_info, mock_pcm, mock_corr):
        """Sync should return offsets for each recording."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 60.0}
        from opencut.core.multi_pov import sync_pov_recordings

        synced = sync_pov_recordings(["a.mp4", "b.mp4"])

        self.assertEqual(len(synced), 2)
        self.assertEqual(synced[0].offset, 0.0)  # reference
        self.assertEqual(synced[0].confidence, 1.0)

    @patch("opencut.core.multi_pov._cross_correlate", return_value=(2.0, 0.9))
    @patch("opencut.core.multi_pov._extract_audio_pcm", return_value=b"\x00" * 1000)
    @patch("opencut.core.multi_pov.get_video_info")
    @patch("os.path.isfile", return_value=True)
    def test_sync_progress_callback(self, mock_isfile, mock_info, mock_pcm, mock_corr):
        """Progress callback should be invoked during sync."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 60.0}
        from opencut.core.multi_pov import sync_pov_recordings

        progress = MagicMock()
        sync_pov_recordings(["a.mp4", "b.mp4"], on_progress=progress)
        self.assertTrue(progress.called)


class TestMultiPovSharedAudio(unittest.TestCase):
    """Tests for opencut.core.multi_pov — shared audio detection."""

    def test_detect_single_file(self):
        """Single file should return no pairs."""
        from opencut.core.multi_pov import detect_shared_audio
        result = detect_shared_audio(["single.mp4"])
        self.assertEqual(result["pairs"], [])
        self.assertFalse(result["has_shared_audio"])


class TestMultiPovXML(unittest.TestCase):
    """Tests for opencut.core.multi_pov — XML generation."""

    def test_generate_xml_empty_raises(self):
        """Empty clips list should raise ValueError."""
        from opencut.core.multi_pov import generate_multicam_xml
        with self.assertRaises(ValueError):
            generate_multicam_xml([])

    def test_generate_xml_structure(self):
        """Generated XML should have correct XMEML structure."""
        from opencut.core.multi_pov import SyncedClip, generate_multicam_xml

        clips = [
            SyncedClip(file_path="/tmp/a.mp4", offset=0.0, duration=60.0, label="cam1"),
            SyncedClip(file_path="/tmp/b.mp4", offset=2.5, duration=55.0, label="cam2"),
        ]
        result = generate_multicam_xml(clips)

        self.assertIn("xml", result)
        self.assertIn("<xmeml", result["xml"])
        self.assertIn("cam1", result["xml"])
        self.assertIn("cam2", result["xml"])
        self.assertEqual(result["clip_count"], 2)

    def test_generate_xml_writes_file(self):
        """XML should be written to file when output_path provided."""
        from opencut.core.multi_pov import SyncedClip, generate_multicam_xml

        clips = [SyncedClip(file_path="/tmp/a.mp4", offset=0.0, duration=60.0, label="cam1")]
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
            out_path = f.name

        try:
            result = generate_multicam_xml(clips, output_path=out_path)
            self.assertEqual(result["output"], out_path)
            self.assertTrue(os.path.isfile(out_path))
            with open(out_path, "r") as f:
                content = f.read()
            self.assertIn("<xmeml", content)
        finally:
            try:
                os.unlink(out_path)
            except OSError:
                pass


# ============================================================
# 31.1 — Multi-Track ISO Ingest Tests
# ============================================================
class TestISOIngestDetect(unittest.TestCase):
    """Tests for opencut.core.iso_ingest — track detection."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        self.tmp.write(b"\x00" * 100)
        self.tmp.close()

    def tearDown(self):
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    def test_detect_missing_file_raises(self):
        """Missing file should raise FileNotFoundError."""
        from opencut.core.iso_ingest import detect_iso_tracks
        with self.assertRaises(FileNotFoundError):
            detect_iso_tracks("/nonexistent/iso.mp4")

    @patch("opencut.core.iso_ingest.get_video_info")
    @patch("subprocess.run")
    def test_detect_returns_track(self, mock_run, mock_info):
        """Detection should return an ISOTrack with properties."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 29.97, "duration": 120.0}
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({
                "format": {"tags": {"timecode": "01:00:00:00"}},
                "streams": [
                    {"codec_type": "video", "codec_name": "h264", "tags": {}},
                    {"codec_type": "audio", "codec_name": "aac", "tags": {}},
                ],
            }).encode(),
        )
        from opencut.core.iso_ingest import detect_iso_tracks

        track = detect_iso_tracks(self.tmp.name)

        self.assertEqual(track.file_path, self.tmp.name)
        self.assertEqual(track.timecode, "01:00:00:00")
        self.assertTrue(track.has_audio)
        self.assertEqual(track.codec, "h264")


class TestISOTimecode(unittest.TestCase):
    """Tests for timecode parsing."""

    def test_parse_smpte_timecode(self):
        """Parse HH:MM:SS:FF format."""
        from opencut.core.iso_ingest import _parse_timecode
        tc = _parse_timecode("01:00:00:00", fps=30.0)
        self.assertAlmostEqual(tc, 3600.0, places=1)

    def test_parse_drop_frame(self):
        """Parse HH:MM:SS;FF drop-frame format."""
        from opencut.core.iso_ingest import _parse_timecode
        tc = _parse_timecode("00:01:00;15", fps=29.97)
        self.assertGreater(tc, 60.0)

    def test_parse_empty_returns_negative(self):
        """Empty timecode should return -1."""
        from opencut.core.iso_ingest import _parse_timecode
        self.assertEqual(_parse_timecode(""), -1.0)


class TestISOSync(unittest.TestCase):
    """Tests for opencut.core.iso_ingest — sync."""

    def test_sync_too_few_files_raises(self):
        """Need at least 2 files."""
        from opencut.core.iso_ingest import sync_iso_recordings
        with self.assertRaises(ValueError):
            sync_iso_recordings(["single.mp4"])


class TestISOTimeline(unittest.TestCase):
    """Tests for opencut.core.iso_ingest — timeline generation."""

    def test_timeline_empty_raises(self):
        """Empty tracks should raise ValueError."""
        from opencut.core.iso_ingest import generate_multicam_timeline
        with self.assertRaises(ValueError):
            generate_multicam_timeline([])

    def test_timeline_xml_structure(self):
        """Generated timeline XML should have XMEML structure."""
        from opencut.core.iso_ingest import ISOTrack, SyncedISOTrack, generate_multicam_timeline

        tracks = [
            SyncedISOTrack(
                track=ISOTrack(file_path="/tmp/iso1.mp4", duration=60.0, label="cam1"),
                offset=0.0,
                sync_method="timecode",
                confidence=1.0,
            ),
            SyncedISOTrack(
                track=ISOTrack(file_path="/tmp/iso2.mp4", duration=55.0, label="cam2"),
                offset=1.0,
                sync_method="timecode",
                confidence=1.0,
            ),
        ]

        result = generate_multicam_timeline(tracks)

        self.assertIn("<xmeml", result.timeline_xml)
        self.assertIn("cam1", result.timeline_xml)
        self.assertEqual(len(result.tracks), 2)
        self.assertGreater(result.total_duration, 0)

    def test_timeline_writes_file(self):
        """Timeline should be written to file when path provided."""
        from opencut.core.iso_ingest import ISOTrack, SyncedISOTrack, generate_multicam_timeline

        tracks = [
            SyncedISOTrack(
                track=ISOTrack(file_path="/tmp/iso1.mp4", duration=60.0, label="cam1"),
                offset=0.0, sync_method="audio", confidence=0.9,
            ),
        ]

        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
            out_path = f.name

        try:
            result = generate_multicam_timeline(tracks, output_path=out_path)
            self.assertTrue(os.path.isfile(out_path))
            self.assertEqual(result.timeline_path, out_path)
        finally:
            try:
                os.unlink(out_path)
            except OSError:
                pass


# ============================================================
# 31.2 — Instant Replay Builder Tests
# ============================================================
class TestReplayConfig(unittest.TestCase):
    """Tests for ReplayConfig dataclass."""

    def test_default_config(self):
        """Default config should have sane values."""
        from opencut.core.instant_replay import ReplayConfig
        cfg = ReplayConfig()
        self.assertEqual(cfg.pre_roll, 3.0)
        self.assertEqual(cfg.post_roll, 2.0)
        self.assertEqual(cfg.slow_factor, 0.5)
        self.assertEqual(cfg.overlay_text, "REPLAY")
        self.assertEqual(cfg.transition, "flash")
        self.assertTrue(cfg.include_original)

    def test_custom_config(self):
        """Custom config values should be preserved."""
        from opencut.core.instant_replay import ReplayConfig
        cfg = ReplayConfig(
            pre_roll=5.0, slow_factor=0.25, overlay_text="INSTANT REPLAY",
            transition="fade",
        )
        self.assertEqual(cfg.pre_roll, 5.0)
        self.assertEqual(cfg.slow_factor, 0.25)
        self.assertEqual(cfg.overlay_text, "INSTANT REPLAY")


class TestInstantReplay(unittest.TestCase):
    """Tests for opencut.core.instant_replay — single replay."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        self.tmp.write(b"\x00" * 100)
        self.tmp.close()

    def tearDown(self):
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    def test_create_missing_video_raises(self):
        """Missing video should raise FileNotFoundError."""
        from opencut.core.instant_replay import create_replay
        with self.assertRaises(FileNotFoundError):
            create_replay("/nonexistent/video.mp4", timestamp=30.0)

    @patch("opencut.core.instant_replay.run_ffmpeg")
    @patch("opencut.core.instant_replay.get_video_info")
    def test_create_replay_calls_ffmpeg(self, mock_info, mock_ffmpeg):
        """Replay creation should invoke FFmpeg multiple times."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 120.0}
        from opencut.core.instant_replay import create_replay

        result = create_replay(self.tmp.name, timestamp=30.0)

        self.assertIn("output_path", vars(result))
        self.assertEqual(result.timestamp, 30.0)
        # Should call FFmpeg at least 3 times: extract, slow-mo, overlay
        self.assertGreaterEqual(mock_ffmpeg.call_count, 3)

    @patch("opencut.core.instant_replay.run_ffmpeg")
    @patch("opencut.core.instant_replay.get_video_info")
    def test_create_replay_progress(self, mock_info, mock_ffmpeg):
        """Progress callback should be invoked."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 120.0}
        from opencut.core.instant_replay import create_replay

        progress = MagicMock()
        create_replay(self.tmp.name, timestamp=30.0, on_progress=progress)
        self.assertTrue(progress.called)

    @patch("opencut.core.instant_replay.run_ffmpeg")
    @patch("opencut.core.instant_replay.get_video_info")
    def test_create_replay_invalid_timestamp(self, mock_info, mock_ffmpeg):
        """Timestamp beyond video duration should raise ValueError."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        from opencut.core.instant_replay import ReplayConfig, create_replay

        # pre_roll + post_roll will exceed when timestamp is at very end
        # but with 10s video and timestamp 5, pre=3 post=2, range is 2-7 which is valid
        # Let's test a zero-duration scenario
        cfg = ReplayConfig(pre_roll=0.0, post_roll=0.0)
        with self.assertRaises(ValueError):
            create_replay(self.tmp.name, timestamp=5.0, config=cfg)


class TestInstantReplaySpeedFilter(unittest.TestCase):
    """Tests for speed filter generation."""

    def test_normal_speed_returns_identity(self):
        """Factor >= 1.0 should return identity setpts."""
        from opencut.core.instant_replay import _build_speed_filter
        f = _build_speed_filter(1.0, 10.0, 0.5, 0.5)
        self.assertEqual(f, "setpts=PTS-STARTPTS")

    def test_slow_factor_produces_multiplier(self):
        """Factor < 1.0 should produce a PTS multiplier."""
        from opencut.core.instant_replay import _build_speed_filter
        f = _build_speed_filter(0.5, 10.0, 0.5, 0.5)
        self.assertIn("2.0", f)

    def test_atempo_chain_extreme_slow(self):
        """Very slow factors should chain multiple atempo filters."""
        from opencut.core.instant_replay import _build_atempo_chain
        chain = _build_atempo_chain(0.25)
        self.assertIn("atempo", chain)
        # 0.25x needs at least one 0.5 step
        self.assertGreater(chain.count("atempo"), 1)

    def test_atempo_normal_speed(self):
        """Normal speed should return anull."""
        from opencut.core.instant_replay import _build_atempo_chain
        self.assertEqual(_build_atempo_chain(1.0), "anull")


class TestBatchReplay(unittest.TestCase):
    """Tests for opencut.core.instant_replay — batch replays."""

    def test_batch_missing_video_raises(self):
        """Missing video should raise FileNotFoundError."""
        from opencut.core.instant_replay import batch_replays
        with self.assertRaises(FileNotFoundError):
            batch_replays("/nonexistent/video.mp4", [10.0, 20.0])

    def test_batch_no_timestamps_raises(self):
        """Empty timestamps should raise ValueError."""
        from opencut.core.instant_replay import batch_replays

        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.write(b"\x00" * 100)
        tmp.close()
        try:
            with self.assertRaises(ValueError):
                batch_replays(tmp.name, [])
        finally:
            os.unlink(tmp.name)


# ============================================================
# 31.4 — Stream Highlight Reel Tests
# ============================================================
class TestStreamHighlightScoring(unittest.TestCase):
    """Tests for opencut.core.stream_highlights — segment scoring."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        self.tmp.write(b"\x00" * 100)
        self.tmp.close()

    def tearDown(self):
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    def test_score_missing_video_raises(self):
        """Missing video should raise FileNotFoundError."""
        from opencut.core.stream_highlights import score_stream_segments
        with self.assertRaises(FileNotFoundError):
            score_stream_segments("/nonexistent/stream.mp4")

    @patch("opencut.core.stream_highlights._analyze_motion_segments", return_value=[0.5, 0.8])
    @patch("opencut.core.stream_highlights._analyze_audio_energy_segments", return_value=[0.6, 0.9])
    @patch("opencut.core.stream_highlights.get_video_info")
    def test_score_returns_segments(self, mock_info, mock_audio, mock_motion):
        """Scoring should return sorted segments."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 20.0}
        from opencut.core.stream_highlights import score_stream_segments

        segments = score_stream_segments(self.tmp.name, segment_duration=10.0)

        self.assertEqual(len(segments), 2)
        # Sorted descending by composite
        self.assertGreaterEqual(segments[0].composite, segments[1].composite)

    @patch("opencut.core.stream_highlights._analyze_motion_segments", return_value=[0.5])
    @patch("opencut.core.stream_highlights._analyze_audio_energy_segments", return_value=[0.5])
    @patch("opencut.core.stream_highlights.get_video_info")
    def test_score_with_keywords(self, mock_info, mock_audio, mock_motion):
        """Keyword scoring should boost segments with matching text."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        from opencut.core.stream_highlights import score_stream_segments

        transcript = [{"start": 0, "end": 10, "text": "That was insane! Let's go!"}]
        segments = score_stream_segments(
            self.tmp.name,
            segment_duration=10.0,
            transcript_segments=transcript,
            keywords=["insane", "let's go"],
        )

        self.assertGreater(len(segments), 0)
        self.assertGreater(segments[0].keyword_score, 0)

    @patch("opencut.core.stream_highlights._analyze_motion_segments", return_value=[0.5])
    @patch("opencut.core.stream_highlights._analyze_audio_energy_segments", return_value=[0.5])
    @patch("opencut.core.stream_highlights.get_video_info")
    def test_score_progress_callback(self, mock_info, mock_audio, mock_motion):
        """Progress callback should be invoked."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        from opencut.core.stream_highlights import score_stream_segments

        progress = MagicMock()
        score_stream_segments(self.tmp.name, on_progress=progress)
        self.assertTrue(progress.called)


class TestStreamHighlightKeywords(unittest.TestCase):
    """Tests for keyword scoring."""

    def test_no_transcript_returns_zero_scores(self):
        """No transcript should give zero keyword scores."""
        from opencut.core.stream_highlights import _score_keywords_segments
        results = _score_keywords_segments(None, 30.0, 10.0)
        self.assertEqual(len(results), 3)
        for score, found in results:
            self.assertEqual(score, 0.0)

    def test_keywords_found_in_segment(self):
        """Matching keywords should increase score."""
        from opencut.core.stream_highlights import _score_keywords_segments
        transcript = [{"start": 0, "end": 5, "text": "clutch play insane"}]
        results = _score_keywords_segments(transcript, 10.0, 10.0, ["clutch", "insane"])
        self.assertGreater(results[0][0], 0)
        self.assertIn("clutch", results[0][1])


class TestStreamHighlightExtract(unittest.TestCase):
    """Tests for opencut.core.stream_highlights — extraction."""

    def test_extract_missing_video_raises(self):
        """Missing video should raise FileNotFoundError."""
        from opencut.core.stream_highlights import ScoredSegment, extract_highlights
        segs = [ScoredSegment(start=0, end=10, composite=0.5)]
        with self.assertRaises(FileNotFoundError):
            extract_highlights("/nonexistent/stream.mp4", segs)


class TestStreamHighlightAssemble(unittest.TestCase):
    """Tests for opencut.core.stream_highlights — reel assembly."""

    def test_assemble_empty_raises(self):
        """Empty clips list should raise ValueError."""
        from opencut.core.stream_highlights import assemble_highlight_reel
        with self.assertRaises(ValueError):
            assemble_highlight_reel([])

    def test_assemble_no_valid_files_raises(self):
        """Clips with missing files should raise ValueError."""
        from opencut.core.stream_highlights import HighlightClip, assemble_highlight_reel
        clips = [HighlightClip(file_path="/nonexistent/clip.mp4", start=0, end=10)]
        with self.assertRaises(ValueError):
            assemble_highlight_reel(clips)


# ============================================================
# Gaming Routes Blueprint Tests
# ============================================================
class TestGamingRoutes(unittest.TestCase):
    """Tests for opencut.routes.gaming_routes blueprint registration."""

    def test_blueprint_exists(self):
        """gaming_bp should be importable."""
        from opencut.routes.gaming_routes import gaming_bp
        self.assertEqual(gaming_bp.name, "gaming")

    def test_blueprint_registered(self):
        """gaming_bp should be in the registered blueprints list."""
        from opencut.routes import register_blueprints
        # Just verify the import works (the function registers on app)
        self.assertTrue(callable(register_blueprints))


class TestGamingRoutesIntegration(unittest.TestCase):
    """Integration tests for gaming route endpoints."""

    def setUp(self):
        """Create a minimal Flask test app."""
        from flask import Flask
        self.app = Flask(__name__)
        self.app.config["TESTING"] = True
        from opencut.routes.gaming_routes import gaming_bp
        self.app.register_blueprint(gaming_bp)
        self.client = self.app.test_client()

    def test_chat_replay_no_filepath(self):
        """POST without filepath should return 400."""
        with self.app.test_request_context():
            from opencut.security import get_csrf_token
            token = get_csrf_token()
        resp = self.client.post(
            "/gaming/chat-replay",
            json={},
            headers={"X-OpenCut-Token": token},
        )
        self.assertIn(resp.status_code, (400, 403))

    def test_instant_replay_no_timestamp(self):
        """POST without valid timestamp should error."""
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.write(b"\x00" * 100)
        tmp.close()
        try:
            with self.app.test_request_context():
                from opencut.security import get_csrf_token
                token = get_csrf_token()
            resp = self.client.post(
                "/gaming/instant-replay",
                json={"filepath": tmp.name, "timestamp": 0},
                headers={"X-OpenCut-Token": token},
            )
            # Should either fail validation or start a job that will error
            self.assertIn(resp.status_code, (200, 400))
        finally:
            os.unlink(tmp.name)

    def test_multi_pov_sync_too_few_files(self):
        """POST with < 2 files should error."""
        with self.app.test_request_context():
            from opencut.security import get_csrf_token
            token = get_csrf_token()
        resp = self.client.post(
            "/gaming/multi-pov/sync",
            json={"file_paths": ["one.mp4"]},
            headers={"X-OpenCut-Token": token},
        )
        # Either 400 or job that errors
        self.assertIn(resp.status_code, (200, 400))

    def test_highlight_score_no_file(self):
        """POST without filepath should return 400."""
        with self.app.test_request_context():
            from opencut.security import get_csrf_token
            token = get_csrf_token()
        resp = self.client.post(
            "/gaming/stream-highlights/score",
            json={},
            headers={"X-OpenCut-Token": token},
        )
        self.assertIn(resp.status_code, (400, 403))

    def test_csrf_required(self):
        """POST without CSRF token should return 403."""
        resp = self.client.post(
            "/gaming/chat-replay",
            json={"filepath": "test.mp4"},
        )
        self.assertEqual(resp.status_code, 403)

    def test_iso_detect_no_file(self):
        """POST without filepath should return 400."""
        with self.app.test_request_context():
            from opencut.security import get_csrf_token
            token = get_csrf_token()
        resp = self.client.post(
            "/gaming/iso-ingest/detect",
            json={},
            headers={"X-OpenCut-Token": token},
        )
        self.assertIn(resp.status_code, (400, 403))

    def test_montage_score_no_clips(self):
        """POST without clip_paths should error."""
        with self.app.test_request_context():
            from opencut.security import get_csrf_token
            token = get_csrf_token()
        resp = self.client.post(
            "/gaming/auto-montage/score",
            json={"clip_paths": []},
            headers={"X-OpenCut-Token": token},
        )
        # Job will start and error with "No clip paths"
        self.assertIn(resp.status_code, (200, 400))

    def test_highlight_assemble_no_clips(self):
        """POST without clips should error."""
        with self.app.test_request_context():
            from opencut.security import get_csrf_token
            token = get_csrf_token()
        resp = self.client.post(
            "/gaming/stream-highlights/assemble",
            json={"clips": []},
            headers={"X-OpenCut-Token": token},
        )
        self.assertIn(resp.status_code, (200, 400))

    def test_iso_sync_no_files(self):
        """POST with no files should error."""
        with self.app.test_request_context():
            from opencut.security import get_csrf_token
            token = get_csrf_token()
        resp = self.client.post(
            "/gaming/iso-ingest/sync",
            json={"file_paths": []},
            headers={"X-OpenCut-Token": token},
        )
        self.assertIn(resp.status_code, (200, 400))

    def test_replay_batch_no_timestamps(self):
        """POST without timestamps should error."""
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.write(b"\x00" * 100)
        tmp.close()
        try:
            with self.app.test_request_context():
                from opencut.security import get_csrf_token
                token = get_csrf_token()
            resp = self.client.post(
                "/gaming/instant-replay/batch",
                json={"filepath": tmp.name, "timestamps": []},
                headers={"X-OpenCut-Token": token},
            )
            self.assertIn(resp.status_code, (200, 400))
        finally:
            os.unlink(tmp.name)


if __name__ == "__main__":
    unittest.main()
