"""
Tests for the Sequence Index panel backend (RESEARCH_FEATURE_PLAN_2026-05-25 Q7 / F273).

Covers:
  * timecode formatting (HH:MM:SS:FF) for common framerates;
  * sequence-payload coerce + missing-field tolerance;
  * row construction from video + audio tracks + effects;
  * transcript excerpt overlap math;
  * sort + filter (track_type / query / min_rating / has_effects);
  * jsonify protocol;
  * three new routes.
"""
from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


SAMPLE_SEQ = {
    "name": "Sequence 01",
    "duration": 60.0,
    "fps": 24.0,
    "width": 1920,
    "height": 1080,
    "videoTracks": [
        {
            "index": 0,
            "clips": [
                {"name": "B-roll-intro.mp4", "path": "/media/intro.mp4",
                 "start": 0.0, "end": 5.0, "effects": ["Gaussian Blur"]},
                {"name": "interview-take-3.mp4", "path": "/media/take3.mp4",
                 "start": 5.0, "end": 30.0, "effects": []},
            ],
        },
        {"index": 1, "clips": [
            {"name": "lower-third.png", "path": "/media/lt.png",
             "start": 10.0, "end": 13.0, "effects": ["Drop Shadow"]},
        ]},
    ],
    "audioTracks": [
        {"index": 0, "clips": [
            {"name": "voiceover.wav", "path": "/media/voice.wav",
             "start": 0.0, "end": 60.0},
        ]},
    ],
    "markers": [
        {"time": 4.2, "name": "intro", "type": "comment", "color": 0},
        {"time": 30.5, "name": "outro", "type": "comment", "color": 1},
    ],
}

SAMPLE_TRANSCRIPT = [
    {"start": 0.0, "end": 4.5, "text": "Welcome back to the channel."},
    {"start": 5.0, "end": 9.0, "text": "Today we're talking about edge cases."},
    {"start": 10.0, "end": 12.0, "text": "First, a quick housekeeping note."},
    {"start": 30.5, "end": 33.0, "text": "Thanks for watching!"},
]


class TestTimecode(unittest.TestCase):
    def test_zero(self):
        from opencut.core.sequence_index import _seconds_to_timecode
        self.assertEqual(_seconds_to_timecode(0.0, 24.0), "00:00:00:00")

    def test_one_hour(self):
        from opencut.core.sequence_index import _seconds_to_timecode
        self.assertEqual(_seconds_to_timecode(3600.0, 24.0), "01:00:00:00")

    def test_fractional_frames(self):
        from opencut.core.sequence_index import _seconds_to_timecode
        # 1.5s @ 24 fps = frame 36 = 00:00:01:12
        self.assertEqual(_seconds_to_timecode(1.5, 24.0), "00:00:01:12")

    def test_thirty_fps(self):
        from opencut.core.sequence_index import _seconds_to_timecode
        self.assertEqual(_seconds_to_timecode(2.0, 30.0), "00:00:02:00")

    def test_negative_clamped_to_zero(self):
        from opencut.core.sequence_index import _seconds_to_timecode
        self.assertEqual(_seconds_to_timecode(-1.5, 24.0), "00:00:00:00")

    def test_zero_fps_falls_back_to_24(self):
        from opencut.core.sequence_index import _seconds_to_timecode
        self.assertEqual(_seconds_to_timecode(1.0, 0.0), "00:00:01:00")


class TestBuildIndex(unittest.TestCase):
    def test_row_count_matches_video_plus_audio_clips(self):
        from opencut.core.sequence_index import build_index
        result = build_index(SAMPLE_SEQ)
        # 3 video clips + 1 audio clip
        self.assertEqual(result.total_rows, 4)
        self.assertEqual(len(result.rows), 4)
        self.assertEqual(result.marker_count, 2)

    def test_sequence_metadata_preserved(self):
        from opencut.core.sequence_index import build_index
        result = build_index(SAMPLE_SEQ)
        self.assertEqual(result.sequence_name, "Sequence 01")
        self.assertEqual(result.fps, 24.0)
        self.assertEqual(result.width, 1920)
        self.assertEqual(result.height, 1080)

    def test_track_type_assigned(self):
        from opencut.core.sequence_index import build_index
        result = build_index(SAMPLE_SEQ)
        types = [r.track_type for r in result.rows]
        self.assertEqual(types.count("video"), 3)
        self.assertEqual(types.count("audio"), 1)

    def test_effects_propagated_for_video(self):
        from opencut.core.sequence_index import build_index
        result = build_index(SAMPLE_SEQ)
        intro = next(r for r in result.rows if r.name == "B-roll-intro.mp4")
        self.assertEqual(intro.effects, ["Gaussian Blur"])

    def test_transcript_excerpt_overlaps_clip(self):
        from opencut.core.sequence_index import build_index
        result = build_index(SAMPLE_SEQ, transcript_segments=SAMPLE_TRANSCRIPT)
        intro = next(r for r in result.rows if r.name == "B-roll-intro.mp4")
        # 0..5s window overlaps both first transcript segments (0-4.5 and 5-9)
        self.assertIn("Welcome", intro.transcript_excerpt)
        # 'Thanks for watching' is at 30.5s; should NOT appear in 0-5 window.
        self.assertNotIn("Thanks", intro.transcript_excerpt)

    def test_ratings_and_tags_applied(self):
        from opencut.core.sequence_index import build_index
        result = build_index(
            SAMPLE_SEQ,
            ratings={"/media/take3.mp4": 5},
            tags={"/media/take3.mp4": ["hero", "approved"]},
        )
        take3 = next(r for r in result.rows if r.name == "interview-take-3.mp4")
        self.assertEqual(take3.rating, 5)
        self.assertEqual(take3.tags, ["hero", "approved"])

    def test_partial_payload_tolerated(self):
        from opencut.core.sequence_index import build_index
        # Missing fields shouldn't crash.
        result = build_index({"name": "Empty"})
        self.assertEqual(result.sequence_name, "Empty")
        self.assertEqual(result.total_rows, 0)
        self.assertEqual(result.marker_count, 0)

    def test_non_dict_payload_yields_empty_result(self):
        from opencut.core.sequence_index import build_index
        self.assertEqual(build_index("not a dict").total_rows, 0)
        self.assertEqual(build_index(None).total_rows, 0)


class TestSortAndFilter(unittest.TestCase):
    def setUp(self):
        from opencut.core.sequence_index import build_index
        self.rows = build_index(SAMPLE_SEQ).rows

    def test_sort_by_start(self):
        from opencut.core.sequence_index import sort_rows
        sorted_rows = sort_rows(self.rows, "start_s")
        starts = [r.start_s for r in sorted_rows]
        self.assertEqual(starts, sorted(starts))

    def test_sort_by_duration_descending(self):
        from opencut.core.sequence_index import sort_rows
        sorted_rows = sort_rows(self.rows, "duration_s", descending=True)
        durations = [r.duration_s for r in sorted_rows]
        self.assertEqual(durations, sorted(durations, reverse=True))

    def test_sort_with_unknown_key_raises(self):
        from opencut.core.sequence_index import sort_rows
        with self.assertRaises(ValueError):
            sort_rows(self.rows, "definitely_not_a_real_key")

    def test_filter_by_track_type(self):
        from opencut.core.sequence_index import filter_rows
        only_audio = filter_rows(self.rows, track_type="audio")
        self.assertEqual(len(only_audio), 1)
        self.assertEqual(only_audio[0].track_type, "audio")

    def test_filter_by_query_matches_name(self):
        from opencut.core.sequence_index import filter_rows
        hits = filter_rows(self.rows, query="interview")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].name, "interview-take-3.mp4")

    def test_filter_has_effects_true(self):
        from opencut.core.sequence_index import filter_rows
        with_effects = filter_rows(self.rows, has_effects=True)
        # B-roll-intro and lower-third both have effects.
        names = sorted(r.name for r in with_effects)
        self.assertEqual(names, ["B-roll-intro.mp4", "lower-third.png"])

    def test_filter_has_effects_false(self):
        from opencut.core.sequence_index import filter_rows
        no_effects = filter_rows(self.rows, has_effects=False)
        names = sorted(r.name for r in no_effects)
        # interview-take-3 + voiceover.
        self.assertEqual(names, ["interview-take-3.mp4", "voiceover.wav"])


class TestJsonifyProtocol(unittest.TestCase):
    def test_result_subscriptable(self):
        from opencut.core.sequence_index import build_index
        result = build_index(SAMPLE_SEQ)
        self.assertEqual(result["sequence_name"], "Sequence 01")
        self.assertIsInstance(result["rows"], list)
        self.assertEqual(len(result["rows"]), 4)
        self.assertIn("sequence_name", result.keys())


class TestRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from opencut.server import create_app
        cls.app = create_app()
        cls.client = cls.app.test_client()
        cls.token = cls.client.get("/health").get_json().get("csrf_token", "")

    def test_info_endpoint(self):
        resp = self.client.get("/timeline/sequence-index/info")
        self.assertEqual(resp.status_code, 200)
        body = json.loads(resp.data.decode("utf-8"))
        self.assertTrue(body["available"])
        self.assertIn("start_s", body["sort_keys"])

    def test_build_endpoint_returns_rows(self):
        resp = self.client.post(
            "/timeline/sequence-index",
            json={"sequence": SAMPLE_SEQ, "transcript_segments": SAMPLE_TRANSCRIPT},
            headers={"X-OpenCut-Token": self.token},
        )
        self.assertEqual(resp.status_code, 200, resp.data)
        body = json.loads(resp.data.decode("utf-8"))
        self.assertEqual(body["total_rows"], 4)
        self.assertEqual(body["marker_count"], 2)

    def test_build_rejects_missing_sequence(self):
        resp = self.client.post(
            "/timeline/sequence-index",
            json={},
            headers={"X-OpenCut-Token": self.token},
        )
        self.assertEqual(resp.status_code, 400)

    def test_filter_endpoint_round_trips_rows(self):
        # Build, then filter.
        build = self.client.post(
            "/timeline/sequence-index",
            json={"sequence": SAMPLE_SEQ},
            headers={"X-OpenCut-Token": self.token},
        )
        rows = json.loads(build.data.decode("utf-8"))["rows"]
        resp = self.client.post(
            "/timeline/sequence-index/filter",
            json={"rows": rows, "track_type": "video", "sort_key": "duration_s", "descending": True},
            headers={"X-OpenCut-Token": self.token},
        )
        self.assertEqual(resp.status_code, 200, resp.data)
        body = json.loads(resp.data.decode("utf-8"))
        self.assertEqual(body["sort_key"], "duration_s")
        # All returned rows must be video.
        self.assertTrue(all(r["track_type"] == "video" for r in body["rows"]))
        durations = [r["duration_s"] for r in body["rows"]]
        self.assertEqual(durations, sorted(durations, reverse=True))

    def test_filter_rejects_bad_sort_key(self):
        resp = self.client.post(
            "/timeline/sequence-index/filter",
            json={"rows": [], "sort_key": "absolutely_not_a_key"},
            headers={"X-OpenCut-Token": self.token},
        )
        self.assertEqual(resp.status_code, 400)


if __name__ == "__main__":
    unittest.main()
