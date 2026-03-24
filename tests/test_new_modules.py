"""
Tests for new OpenCut core modules.

Covers: repeat_detect, chapter_gen, footage_search, deliverables,
        multicam, loudness_match (interface), auto_zoom (interface),
        color_match (interface), nlp_command.
"""
import csv
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch, mock_open

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# TestRepeatDetect
# ============================================================
class TestRepeatDetect(unittest.TestCase):
    """Tests for repeat_detect.py — pure Python, no external deps."""

    def setUp(self):
        from opencut.core.repeat_detect import detect_repeated_takes, merge_repeat_ranges
        self.detect = detect_repeated_takes
        self.merge = merge_repeat_ranges

    def _seg(self, text, start, end):
        return {"text": text, "start": start, "end": end, "words": []}

    def test_no_repeats_in_distinct_segments(self):
        """Non-similar segments should produce zero repeats."""
        segs = [
            self._seg("Hello world today is great", 0, 2),
            self._seg("Now we talk about Python programming", 2.1, 4),
            self._seg("Final segment with different words", 4.1, 6),
        ]
        result = self.detect(segs, threshold=0.6)
        self.assertEqual(result["repeats"], [])

    def test_exact_duplicate_detected(self):
        """Identical adjacent segments should be flagged as repeat."""
        segs = [
            self._seg("uh so today we are going to talk about cameras", 0, 3),
            self._seg("so today we are going to talk about cameras", 3.2, 6),
            self._seg("and the first thing is the lens", 6.1, 8),
        ]
        result = self.detect(segs, threshold=0.5)
        self.assertGreater(len(result["repeats"]), 0)

    def test_clean_ranges_cover_non_repeat_segments(self):
        """clean_ranges should be a list of dicts with start/end, one per kept segment."""
        segs = [
            self._seg("The weather today is nice", 0, 2),
            self._seg("The weather today is nice again", 2.1, 4),
            self._seg("Moving on to the next topic", 4.2, 6),
        ]
        result = self.detect(segs, threshold=0.5)
        self.assertIsInstance(result["clean_ranges"], list)
        for r in result["clean_ranges"]:
            self.assertIn("start", r)
            self.assertIn("end", r)

    def test_gap_tolerance_prevents_false_positive(self):
        """Segments far apart in time should NOT be flagged even if similar."""
        segs = [
            self._seg("we talk about cameras today", 0, 2),
            self._seg("we talk about cameras today", 60, 62),  # 58s gap
        ]
        result = self.detect(segs, threshold=0.6, gap_tolerance=2.0)
        self.assertEqual(result["repeats"], [])

    def test_merge_repeat_ranges_merges_overlapping(self):
        """Overlapping ranges should be merged into one."""
        repeats = [
            {"start": 1.0, "end": 3.0, "text": "a", "similarity": 0.8, "paired_with": 1},
            {"start": 2.5, "end": 5.0, "text": "b", "similarity": 0.9, "paired_with": 2},
            {"start": 10.0, "end": 12.0, "text": "c", "similarity": 0.7, "paired_with": 3},
        ]
        merged = self.merge(repeats)
        self.assertEqual(len(merged), 2)
        self.assertAlmostEqual(merged[0]["start"], 1.0)
        self.assertAlmostEqual(merged[0]["end"], 5.0)
        self.assertAlmostEqual(merged[1]["start"], 10.0)

    def test_empty_input(self):
        """Empty segment list should return empty results."""
        result = self.detect([], threshold=0.6)
        self.assertEqual(result["repeats"], [])
        self.assertEqual(result["clean_ranges"], [])

    def test_threshold_controls_sensitivity(self):
        """Higher threshold should detect fewer repeats."""
        segs = [
            self._seg("uh so I wanted to mention cameras", 0, 2),
            self._seg("so I wanted to mention cameras today", 2.1, 4),
        ]
        strict = self.detect(segs, threshold=0.95)
        loose = self.detect(segs, threshold=0.3)
        self.assertLessEqual(len(strict["repeats"]), len(loose["repeats"]))

    def test_repeat_entry_has_required_keys(self):
        """Each repeat entry should include start, end, text, similarity, paired_with."""
        segs = [
            self._seg("same words same words same words", 0, 2),
            self._seg("same words same words same words", 2.1, 4),
        ]
        result = self.detect(segs, threshold=0.5)
        self.assertGreater(len(result["repeats"]), 0)
        entry = result["repeats"][0]
        for key in ("start", "end", "text", "similarity", "paired_with"):
            self.assertIn(key, entry)

    def test_merge_empty_repeats(self):
        """merge_repeat_ranges with empty list should return empty list."""
        self.assertEqual(self.merge([]), [])

    def test_single_segment_no_repeats(self):
        """A single segment can never be a repeat."""
        segs = [self._seg("only one segment here", 0, 5)]
        result = self.detect(segs, threshold=0.5)
        self.assertEqual(result["repeats"], [])
        self.assertEqual(len(result["clean_ranges"]), 1)


# ============================================================
# TestChapterGen
# ============================================================
class TestChapterGen(unittest.TestCase):
    """Tests for chapter_gen.py."""

    def setUp(self):
        from opencut.core.chapter_gen import generate_chapters, _time_str_to_seconds
        self.generate = generate_chapters
        self.time_to_secs = _time_str_to_seconds

    def _seg(self, text, start, end):
        return {"text": text, "start": start, "end": end, "words": []}

    def test_time_str_to_seconds_mm_ss(self):
        """M:SS format should parse correctly."""
        self.assertEqual(self.time_to_secs("1:30"), 90.0)
        self.assertEqual(self.time_to_secs("0:45"), 45.0)
        self.assertEqual(self.time_to_secs("10:00"), 600.0)

    def test_time_str_to_seconds_h_mm_ss(self):
        """H:MM:SS format should parse correctly."""
        self.assertEqual(self.time_to_secs("1:00:00"), 3600.0)
        self.assertEqual(self.time_to_secs("1:01:30"), 3690.0)

    def test_time_str_to_seconds_invalid_returns_none(self):
        """Invalid time string should return None."""
        result = self.time_to_secs("not-a-time")
        self.assertIsNone(result)

    def test_heuristic_fallback_no_llm(self):
        """Without a reachable LLM, heuristic should still generate chapters from long pauses."""
        # Create 4 groups of segments with large gaps between them
        segs = []
        for group_start in [0, 120, 300, 500]:
            for i in range(5):
                segs.append(self._seg(
                    f"Word {i} in group", group_start + i * 2, group_start + i * 2 + 1.5
                ))
        # Pass llm_config=None to force heuristic path (LLM will fail and fall through)
        with patch("opencut.core.chapter_gen.query_llm", side_effect=Exception("no LLM")):
            result = self.generate(segs, llm_config=None, max_chapters=10)
        self.assertIn("chapters", result)
        self.assertIn("description_block", result)
        self.assertIsInstance(result["chapters"], list)
        self.assertGreater(len(result["chapters"]), 0)

    def test_description_block_format(self):
        """Description block should have timestamp: title format on each line."""
        segs = [self._seg("intro text", 0, 30), self._seg("main topic text", 35, 90)]
        with patch("opencut.core.chapter_gen.query_llm", side_effect=Exception("no LLM")):
            result = self.generate(segs, llm_config=None)
        block = result.get("description_block", "")
        lines = [l for l in block.strip().split("\n") if l.strip()]
        if lines:
            self.assertRegex(lines[0], r"\d+:\d+")

    def test_max_chapters_respected(self):
        """Should not generate more chapters than max_chapters."""
        segs = []
        for i in range(100):
            segs.append(self._seg(f"segment {i} content text words", i * 10, i * 10 + 8))
        with patch("opencut.core.chapter_gen.query_llm", side_effect=Exception("no LLM")):
            result = self.generate(segs, llm_config=None, max_chapters=5)
        self.assertLessEqual(len(result["chapters"]), 5)

    def test_first_chapter_always_at_zero(self):
        """The first chapter must always start at 0:00."""
        segs = [self._seg("some text here", 0, 10), self._seg("more text here", 15, 30)]
        with patch("opencut.core.chapter_gen.query_llm", side_effect=Exception("no LLM")):
            result = self.generate(segs, llm_config=None)
        self.assertGreater(len(result["chapters"]), 0)
        self.assertAlmostEqual(result["chapters"][0]["seconds"], 0.0)

    def test_source_heuristic_when_llm_fails(self):
        """source field should be 'heuristic' when LLM is unavailable."""
        segs = [self._seg("text here", 0, 5)]
        with patch("opencut.core.chapter_gen.query_llm", side_effect=Exception("no LLM")):
            result = self.generate(segs, llm_config=None)
        self.assertEqual(result["source"], "heuristic")

    @patch("opencut.core.chapter_gen.query_llm")
    def test_llm_chapters_parsed(self, mock_llm):
        """LLM-generated chapters should be parsed and returned correctly."""
        from opencut.core.llm import LLMConfig, LLMResponse
        mock_llm.return_value = LLMResponse(
            text='[{"time": "0:00", "title": "Intro"}, {"time": "1:30", "title": "Main Topic"}]',
            provider="mock", model="mock"
        )
        segs = [self._seg("hello welcome", 0, 5), self._seg("main topic starts", 90, 95)]
        config = LLMConfig(provider="ollama", model="llama3")
        result = self.generate(segs, llm_config=config, max_chapters=10)
        self.assertEqual(len(result["chapters"]), 2)
        self.assertEqual(result["chapters"][0]["title"], "Intro")
        self.assertAlmostEqual(result["chapters"][1]["seconds"], 90.0)

    @patch("opencut.core.chapter_gen.query_llm")
    def test_source_is_llm_when_llm_succeeds(self, mock_llm):
        """source field should be 'llm' when the LLM returns valid chapters."""
        from opencut.core.llm import LLMConfig, LLMResponse
        mock_llm.return_value = LLMResponse(
            text='[{"time": "0:00", "title": "Intro"}]',
            provider="mock", model="mock"
        )
        segs = [self._seg("hello", 0, 5)]
        config = LLMConfig(provider="ollama", model="llama3")
        result = self.generate(segs, llm_config=config)
        self.assertEqual(result["source"], "llm")

    def test_empty_segments_returns_intro(self):
        """Empty segment list should still return at least the Intro chapter."""
        with patch("opencut.core.chapter_gen.query_llm", side_effect=Exception("no LLM")):
            result = self.generate([], llm_config=None)
        self.assertGreaterEqual(len(result["chapters"]), 1)
        self.assertEqual(result["chapters"][0]["time_str"], "0:00")


# ============================================================
# TestFootageSearch
# ============================================================
class TestFootageSearch(unittest.TestCase):
    """Tests for footage_search.py — pure Python."""

    def setUp(self):
        from opencut.core.footage_search import (
            index_file, load_index, save_index, search_footage,
            remove_missing_files, get_index_stats, clear_index
        )
        self.index_file = index_file
        self.load_index = load_index
        self.save_index = save_index
        self.search = search_footage
        self.remove_missing = remove_missing_files
        self.stats = get_index_stats
        self.clear = clear_index

    def test_index_and_search_basic(self):
        """Indexing a file and searching should call save_index."""
        with patch("opencut.core.footage_search.load_index", return_value={}), \
             patch("opencut.core.footage_search.save_index") as mock_save, \
             patch("os.path.getmtime", return_value=12345.0):
            segments = [{"start": 0.0, "end": 5.0, "text": "hello world cameras lenses"}]
            self.index_file("/fake/video.mp4", segments)
            mock_save.assert_called_once()

    def test_search_returns_correct_structure(self):
        """Search results should have path, start, end, text, score keys."""
        fake_index = {
            "/fake/clip.mp4": {
                "mtime": 12345.0,
                "segments": [{"start": 1.0, "end": 3.0, "text": "camera lens focal length test"}],
                "full_text": "camera lens focal length test"
            }
        }
        with patch("opencut.core.footage_search.load_index", return_value=fake_index):
            results = self.search("camera lens", top_k=5)
        self.assertIsInstance(results, list)
        if results:
            r = results[0]
            self.assertIn("path", r)
            self.assertIn("start", r)
            self.assertIn("end", r)
            self.assertIn("text", r)
            self.assertIn("score", r)

    def test_search_empty_index_returns_empty(self):
        """Searching an empty index should return empty list."""
        with patch("opencut.core.footage_search.load_index", return_value={}):
            results = self.search("anything", top_k=5)
        self.assertEqual(results, [])

    def test_search_no_match_returns_empty(self):
        """Query that doesn't match anything should return empty list."""
        fake_index = {
            "/fake/clip.mp4": {
                "mtime": 12345.0,
                "segments": [{"start": 0, "end": 2, "text": "weather outside today sunny"}],
                "full_text": "weather outside today sunny"
            }
        }
        with patch("opencut.core.footage_search.load_index", return_value=fake_index):
            results = self.search("xyzzy quantum flux", top_k=5)
        self.assertEqual(results, [])

    def test_remove_missing_files(self):
        """Files that don't exist should be removed from index."""
        fake_index = {
            "/exists/file.mp4": {"mtime": 0, "segments": [], "full_text": ""},
            "/missing/file.mp4": {"mtime": 0, "segments": [], "full_text": ""},
        }
        with patch("os.path.exists", side_effect=lambda p: "exists" in p):
            cleaned = self.remove_missing(fake_index)
        self.assertIn("/exists/file.mp4", cleaned)
        self.assertNotIn("/missing/file.mp4", cleaned)

    def test_clear_index_returns_success(self):
        """clear_index should return success=True and removed_files count."""
        fake_index = {"/fake/a.mp4": {}, "/fake/b.mp4": {}}
        with patch("opencut.core.footage_search.load_index", return_value=fake_index), \
             patch("opencut.core.footage_search.save_index") as mock_save, \
             patch("os.path.exists", return_value=True):
            result = self.clear()
        mock_save.assert_called_once_with({})
        self.assertEqual(result["removed_files"], 2)
        self.assertTrue(result["success"])

    def test_get_index_stats_structure(self):
        """Stats should include total_files, total_segments, index_size_bytes."""
        with patch("opencut.core.footage_search.load_index", return_value={}), \
             patch("os.path.exists", return_value=False):
            stats = self.stats()
        self.assertIn("total_files", stats)
        self.assertIn("total_segments", stats)
        self.assertIn("index_size_bytes", stats)

    def test_search_top_k_limits_results(self):
        """Search should return at most top_k results."""
        fake_index = {
            f"/fake/clip{i}.mp4": {
                "mtime": float(i),
                "segments": [{"start": 0.0, "end": 5.0, "text": "camera lens focal length"}],
                "full_text": "camera lens focal length"
            }
            for i in range(20)
        }
        with patch("opencut.core.footage_search.load_index", return_value=fake_index):
            results = self.search("camera lens", top_k=3)
        self.assertLessEqual(len(results), 3)

    def test_search_results_sorted_by_score_descending(self):
        """Search results should be sorted by score in descending order."""
        fake_index = {
            "/fake/clip_good.mp4": {
                "mtime": 1.0,
                "segments": [{"start": 0.0, "end": 5.0,
                               "text": "camera lens focal length prime zoom"}],
                "full_text": "camera lens focal length prime zoom"
            },
            "/fake/clip_weak.mp4": {
                "mtime": 2.0,
                "segments": [{"start": 0.0, "end": 5.0, "text": "camera"}],
                "full_text": "camera"
            }
        }
        with patch("opencut.core.footage_search.load_index", return_value=fake_index):
            results = self.search("camera lens", top_k=10)
        if len(results) >= 2:
            self.assertGreaterEqual(results[0]["score"], results[1]["score"])

    def test_get_index_stats_counts_correctly(self):
        """Stats total_files and total_segments should reflect index content."""
        fake_index = {
            "/a.mp4": {"segments": [{"start": 0, "end": 1, "text": "hi"}], "mtime": 0.0, "full_text": "hi"},
            "/b.mp4": {"segments": [{"start": 0, "end": 1, "text": "hello"},
                                    {"start": 1, "end": 2, "text": "world"}],
                       "mtime": 0.0, "full_text": "hello world"},
        }
        with patch("opencut.core.footage_search.load_index", return_value=fake_index), \
             patch("os.path.exists", return_value=False):
            stats = self.stats()
        self.assertEqual(stats["total_files"], 2)
        self.assertEqual(stats["total_segments"], 3)


# ============================================================
# TestDeliverables
# ============================================================
class TestDeliverables(unittest.TestCase):
    """Tests for deliverables.py — CSV generation."""

    def setUp(self):
        from opencut.core.deliverables import (
            generate_vfx_sheet, generate_adr_list,
            generate_music_cue_sheet, generate_asset_list,
            _seconds_to_tc, _seconds_to_readable
        )
        self.vfx = generate_vfx_sheet
        self.adr = generate_adr_list
        self.music = generate_music_cue_sheet
        self.asset = generate_asset_list
        self.tc = _seconds_to_tc
        self.readable = _seconds_to_readable

    def _sample_sequence(self):
        return {
            "name": "TestSeq",
            "duration": 120.0,
            "video_tracks": [
                {"index": 0, "clips": [
                    {"name": "clip_A", "path": "/media/clip_A.mp4",
                     "start": 0, "end": 10, "effects": ["Blur"]},
                    {"name": "clip_B", "path": "/media/clip_B.mp4",
                     "start": 10, "end": 30, "effects": []},
                ]}
            ],
            "audio_tracks": [
                {"index": 0, "clips": [
                    {"name": "voice_over", "path": "/media/voice.wav", "start": 0, "end": 30}
                ]},
                {"index": 1, "clips": [
                    {"name": "ambience", "path": "/media/ambience.wav", "start": 0, "end": 120}
                ]},
                # Music is on track index 2+ per the module logic
                {"index": 2, "clips": [
                    {"name": "music_track", "path": "/media/music.mp3", "start": 0, "end": 120}
                ]},
            ],
            "markers": []
        }

    def test_seconds_to_tc(self):
        """Timecode helper should format correctly."""
        self.assertEqual(self.tc(0, fps=24), "00:00:00:00")
        self.assertEqual(self.tc(3600, fps=24), "01:00:00:00")
        self.assertEqual(self.tc(90.5, fps=24), "00:01:30:12")

    def test_seconds_to_readable(self):
        """Readable time helper should format correctly."""
        self.assertEqual(self.readable(0), "0:00:00")
        self.assertEqual(self.readable(90), "0:01:30")
        self.assertEqual(self.readable(3661), "1:01:01")

    def test_vfx_sheet_creates_csv(self):
        """VFX sheet should create a valid CSV file with data rows."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "vfx_sheet.csv")
            result = self.vfx(self._sample_sequence(), output_path)
            self.assertIsInstance(result, dict)
            self.assertIn("output", result)
            self.assertIn("rows", result)
            self.assertGreater(result["rows"], 0)
            self.assertTrue(os.path.exists(result["output"]))
            with open(result["output"]) as f:
                rows = list(csv.reader(f))
            self.assertGreater(len(rows), 1)  # header + data

    def test_vfx_sheet_row_count_matches_clips(self):
        """VFX sheet row count should match total number of video clips."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "vfx.csv")
            result = self.vfx(self._sample_sequence(), output_path)
        # Sample sequence has 2 video clips
        self.assertEqual(result["rows"], 2)

    def test_adr_list_creates_csv(self):
        """ADR list should create a valid CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "adr_list.csv")
            result = self.adr(self._sample_sequence(), output_path)
            self.assertIsInstance(result, dict)
            self.assertTrue(os.path.exists(result["output"]))

    def test_adr_character_splitting(self):
        """ADR list should split 'Character - line text' clip names correctly."""
        seq = {
            "video_tracks": [],
            "audio_tracks": [
                {"index": 0, "clips": [
                    {"name": "John - Hello there friend", "start": 0, "end": 5}
                ]}
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "adr.csv")
            result = self.adr(seq, output_path)
            with open(result["output"]) as f:
                content = f.read()
        self.assertIn("John", content)
        self.assertIn("Hello there friend", content)

    def test_music_cue_sheet_detects_music_track(self):
        """Music cue sheet should pick up clips on audio track index >= 2."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "music_cues.csv")
            result = self.music(self._sample_sequence(), output_path)
            self.assertIsInstance(result, dict)
            self.assertTrue(os.path.exists(result["output"]))
            with open(result["output"]) as f:
                content = f.read()
        self.assertIn("music_track", content)

    def test_asset_list_all_media(self):
        """Asset list should include all unique media file paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "assets.csv")
            result = self.asset(self._sample_sequence(), output_path)
            self.assertIsInstance(result, dict)
            self.assertTrue(os.path.exists(result["output"]))
            with open(result["output"]) as f:
                content = f.read()
        self.assertIn("clip_A", content)

    def test_asset_list_deduplicates(self):
        """An asset used multiple times should appear only once with correct use_count."""
        seq = {
            "video_tracks": [
                {"index": 0, "clips": [
                    {"name": "clip_A", "path": "/media/clip_A.mp4", "start": 0, "end": 10},
                    {"name": "clip_A", "path": "/media/clip_A.mp4", "start": 20, "end": 30},
                ]}
            ],
            "audio_tracks": [],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "assets.csv")
            result = self.asset(seq, output_path)
            # Only one unique path, so one data row
            self.assertEqual(result["rows"], 1)
            with open(result["output"]) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            self.assertEqual(int(rows[0]["UseCount"]), 2)

    def test_seconds_to_tc_negative_input_clamped(self):
        """Negative seconds should be clamped to 0."""
        self.assertEqual(self.tc(-5.0, fps=24), "00:00:00:00")

    def test_seconds_to_readable_negative_clamped(self):
        """Negative seconds should be clamped to 0."""
        self.assertEqual(self.readable(-10), "0:00:00")


# ============================================================
# TestMulticam
# ============================================================
class TestMulticam(unittest.TestCase):
    """Tests for multicam.py — pure Python."""

    def setUp(self):
        from opencut.core.multicam import (
            generate_multicam_cuts, auto_assign_speakers,
            merge_diarization_segments
        )
        self.gen_cuts = generate_multicam_cuts
        self.auto_assign = auto_assign_speakers
        self.merge = merge_diarization_segments

    def _seg(self, speaker, start, end):
        return {"speaker": speaker, "start": start, "end": end}

    def test_two_speakers_alternating(self):
        """Two alternating speakers should generate cuts between tracks."""
        segs = [
            self._seg("SPEAKER_00", 0, 5),
            self._seg("SPEAKER_01", 6, 11),
            self._seg("SPEAKER_00", 12, 17),
        ]
        speaker_map = {"SPEAKER_00": 0, "SPEAKER_01": 1}
        result = self.gen_cuts(segs, speaker_map, min_cut_duration=0.5)
        self.assertIn("cuts", result)
        cuts = result["cuts"]
        self.assertEqual(len(cuts), 3)
        self.assertEqual(cuts[0]["track"], 0)
        self.assertEqual(cuts[1]["track"], 1)

    def test_auto_assign_speakers_in_order(self):
        """Speakers should be assigned track numbers in order of first appearance."""
        segs = [
            self._seg("SPEAKER_01", 0, 3),
            self._seg("SPEAKER_00", 4, 7),
            self._seg("SPEAKER_02", 8, 11),
        ]
        mapping = self.auto_assign(segs)
        self.assertEqual(mapping["SPEAKER_01"], 0)
        self.assertEqual(mapping["SPEAKER_00"], 1)
        self.assertEqual(mapping["SPEAKER_02"], 2)

    def test_min_cut_duration_filters_short_segments(self):
        """Segments shorter than min_cut_duration should be filtered out."""
        segs = [
            self._seg("SPEAKER_00", 0, 10),
            self._seg("SPEAKER_01", 10.1, 10.5),  # 0.4s — below threshold
            self._seg("SPEAKER_00", 10.6, 20),
        ]
        speaker_map = {"SPEAKER_00": 0, "SPEAKER_01": 1}
        result = self.gen_cuts(segs, speaker_map, min_cut_duration=1.0)
        tracks = [c["track"] for c in result["cuts"]]
        # The 0.4s SPEAKER_01 segment should be filtered out
        self.assertNotIn(1, tracks)

    def test_merge_segments_same_speaker(self):
        """Consecutive same-speaker segments within gap tolerance should merge."""
        segs = [
            self._seg("SPEAKER_00", 0, 5),
            self._seg("SPEAKER_00", 5.3, 10),  # 0.3s gap — should merge
            self._seg("SPEAKER_01", 11, 15),
        ]
        merged = self.merge(segs, gap_tolerance=0.5)
        speaker_00_segs = [s for s in merged if s["speaker"] == "SPEAKER_00"]
        self.assertEqual(len(speaker_00_segs), 1)
        self.assertAlmostEqual(speaker_00_segs[0]["end"], 10)

    def test_merge_does_not_merge_different_speakers(self):
        """Segments from different speakers should never be merged."""
        segs = [
            self._seg("SPEAKER_00", 0, 5),
            self._seg("SPEAKER_01", 5.1, 10),  # tiny gap but different speaker
        ]
        merged = self.merge(segs, gap_tolerance=1.0)
        self.assertEqual(len(merged), 2)

    def test_empty_segments(self):
        """Empty input should return empty cuts."""
        result = self.gen_cuts([], {}, min_cut_duration=1.0)
        self.assertEqual(result["cuts"], [])
        self.assertEqual(result["total_cuts"], 0)

    def test_total_cuts_count(self):
        """total_cuts should match length of cuts array."""
        segs = [self._seg("SPEAKER_00", i * 5, i * 5 + 4) for i in range(5)]
        segs[1]["speaker"] = "SPEAKER_01"
        segs[3]["speaker"] = "SPEAKER_01"
        mapping = self.auto_assign(segs)
        result = self.gen_cuts(segs, mapping, min_cut_duration=0.5)
        self.assertEqual(result["total_cuts"], len(result["cuts"]))

    def test_speaker_to_track_returned_in_result(self):
        """Result should include the speaker_to_track mapping that was used."""
        segs = [self._seg("SPEAKER_00", 0, 5)]
        speaker_map = {"SPEAKER_00": 0}
        result = self.gen_cuts(segs, speaker_map, min_cut_duration=0.5)
        self.assertIn("speaker_to_track", result)
        self.assertEqual(result["speaker_to_track"]["SPEAKER_00"], 0)

    def test_cut_entries_have_required_keys(self):
        """Each cut dict should have time, track, speaker, duration keys."""
        segs = [self._seg("SPEAKER_00", 0, 5), self._seg("SPEAKER_01", 6, 11)]
        mapping = self.auto_assign(segs)
        result = self.gen_cuts(segs, mapping, min_cut_duration=0.5)
        for cut in result["cuts"]:
            for key in ("time", "track", "speaker", "duration"):
                self.assertIn(key, cut)

    def test_auto_assign_empty_returns_empty_map(self):
        """auto_assign_speakers with empty input should return empty dict."""
        mapping = self.auto_assign([])
        self.assertEqual(mapping, {})


# ============================================================
# TestNlpCommand
# ============================================================
class TestNlpCommand(unittest.TestCase):
    """Tests for nlp_command.py — pure Python."""

    def setUp(self):
        from opencut.core.nlp_command import (
            parse_command_keyword, extract_params_from_text, COMMAND_MAP
        )
        self.parse_kw = parse_command_keyword
        self.extract = extract_params_from_text
        self.map = COMMAND_MAP

    def test_silence_keyword_matches(self):
        """'remove silence' should match the silence route."""
        result = self.parse_kw("remove silence from this clip")
        self.assertIsNotNone(result)
        self.assertIn("silence", result["route"])

    def test_filler_keyword_matches(self):
        """'um' and 'uh' should match the filler route."""
        result = self.parse_kw("remove um and uh from the recording")
        self.assertIsNotNone(result)
        self.assertIn("filler", result["route"])

    def test_caption_keyword_matches(self):
        """'transcribe' should match the captions transcribe route."""
        result = self.parse_kw("transcribe this video")
        self.assertIsNotNone(result)
        self.assertIn("caption", result["route"].lower())

    def test_chapter_keyword_matches(self):
        """'chapters' should match the chapter route."""
        result = self.parse_kw("generate chapters for this video")
        self.assertIsNotNone(result)
        self.assertIn("chapter", result["route"].lower())

    def test_no_match_returns_none(self):
        """Unknown command should return None."""
        result = self.parse_kw("xyzzy frobnitz quux")
        self.assertIsNone(result)

    def test_confidence_field_present(self):
        """Result should always have a confidence field greater than 0."""
        result = self.parse_kw("normalize audio loudness")
        self.assertIsNotNone(result)
        self.assertIn("confidence", result)
        self.assertGreater(result["confidence"], 0)

    def test_matched_keyword_field_present(self):
        """Result should include a matched_keyword field."""
        result = self.parse_kw("remove silence")
        self.assertIsNotNone(result)
        self.assertIn("matched_keyword", result)

    def test_extract_number_as_threshold(self):
        """First number in text should be extracted as 'threshold' param."""
        params = self.extract("remove silence longer than 0.5 seconds")
        self.assertIsInstance(params, dict)
        self.assertIn("threshold", params)
        self.assertAlmostEqual(params["threshold"], 0.5)

    def test_extract_language(self):
        """Language names should be recognized and mapped to ISO codes."""
        params = self.extract("transcribe in Spanish")
        self.assertIsInstance(params, dict)
        self.assertIn("language", params)
        self.assertEqual(params["language"], "es")

    def test_extract_lufs_target(self):
        """LUFS value should be extracted as target_lufs param."""
        params = self.extract("normalize to -14 LUFS")
        self.assertIn("target_lufs", params)
        self.assertAlmostEqual(params["target_lufs"], -14.0)

    def test_command_map_has_all_expected_routes(self):
        """COMMAND_MAP should contain entries for all major features."""
        routes = [entry["route"] for entry in self.map]
        route_str = " ".join(routes)
        self.assertIn("silence", route_str)
        self.assertIn("filler", route_str)
        self.assertIn("chapter", route_str)
        self.assertIn("search", route_str)

    def test_case_insensitive_matching(self):
        """Keyword matching should be case insensitive."""
        result_lower = self.parse_kw("remove silence")
        result_upper = self.parse_kw("REMOVE SILENCE")
        self.assertIsNotNone(result_lower)
        self.assertIsNotNone(result_upper)
        self.assertEqual(result_lower["route"], result_upper["route"])

    def test_search_keyword_matches(self):
        """'find footage' should match the footage search route."""
        result = self.parse_kw("find footage of the city")
        self.assertIsNotNone(result)
        self.assertIn("search", result["route"])

    def test_extract_intensity_word(self):
        """Intensity words should map to a float intensity param."""
        params = self.extract("aggressive noise reduction")
        self.assertIn("intensity", params)
        self.assertAlmostEqual(params["intensity"], 0.9)

    def test_extract_zoom_percentage(self):
        """Zoom percentage pattern should be extracted as zoom_amount."""
        params = self.extract("apply 20% zoom")
        self.assertIn("zoom_amount", params)
        self.assertAlmostEqual(params["zoom_amount"], 1.20)

    def test_params_field_is_dict(self):
        """Returned params field should always be a dict."""
        result = self.parse_kw("remove silence")
        self.assertIsNotNone(result)
        self.assertIsInstance(result["params"], dict)


# ============================================================
# TestLoudnessMatchInterface
# ============================================================
class TestLoudnessMatchInterface(unittest.TestCase):
    """Tests for loudness_match.py public interface."""

    def test_measure_loudness_parses_ffmpeg_stderr(self):
        """measure_loudness should call FFmpeg and parse the loudnorm JSON from stderr."""
        from opencut.core.loudness_match import measure_loudness

        fake_json = json.dumps({
            "input_i": "-18.5",
            "input_tp": "-3.2",
            "input_lra": "7.0",
            "input_thresh": "-29.0",
        })
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = fake_json  # loudnorm block is extracted with regex

        with patch("opencut.core.loudness_match._run_ffmpeg", return_value=mock_result):
            result = measure_loudness("/fake/file.mp4")

        self.assertIn("lufs", result)
        self.assertIn("lra", result)
        self.assertIn("peak", result)
        self.assertIn("true_peak", result)
        self.assertAlmostEqual(result["lufs"], -18.5)
        self.assertAlmostEqual(result["lra"], 7.0)

    def test_measure_loudness_raises_on_no_ffmpeg_output(self):
        """measure_loudness should raise ValueError if no JSON block in stderr."""
        from opencut.core.loudness_match import measure_loudness

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = "no JSON here at all"

        with patch("opencut.core.loudness_match._run_ffmpeg", return_value=mock_result):
            with self.assertRaises(ValueError):
                measure_loudness("/fake/file.mp4")

    def test_measure_loudness_raises_runtime_error_no_ffmpeg(self):
        """measure_loudness should raise RuntimeError when FFmpeg binary is missing."""
        from opencut.core.loudness_match import measure_loudness

        with patch("opencut.core.loudness_match._run_ffmpeg", side_effect=RuntimeError("FFmpeg not found")):
            with self.assertRaises(RuntimeError):
                measure_loudness("/fake/file.mp4")

    def test_batch_loudness_match_returns_list(self):
        """batch_loudness_match should return a list of result dicts."""
        from opencut.core.loudness_match import batch_loudness_match

        with patch("opencut.core.loudness_match.normalize_to_lufs",
                   return_value="/fake/out.mp4"), \
             patch("opencut.core.loudness_match.measure_loudness",
                   return_value={"lufs": -18.0, "lra": 7.0, "peak": -3.0, "true_peak": -3.0}):
            with tempfile.TemporaryDirectory() as tmpdir:
                results = batch_loudness_match(
                    ["/fake/a.mp4", "/fake/b.mp4"], tmpdir, target_lufs=-14.0
                )
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)

    def test_batch_loudness_match_result_keys(self):
        """Each batch result dict should have input, output, original_lufs, job_ok keys."""
        from opencut.core.loudness_match import batch_loudness_match

        with patch("opencut.core.loudness_match.normalize_to_lufs",
                   return_value="/fake/out.mp4"), \
             patch("opencut.core.loudness_match.measure_loudness",
                   return_value={"lufs": -18.0, "lra": 7.0, "peak": -3.0, "true_peak": -3.0}):
            with tempfile.TemporaryDirectory() as tmpdir:
                results = batch_loudness_match(["/fake/a.mp4"], tmpdir)
        self.assertEqual(len(results), 1)
        entry = results[0]
        for key in ("input", "output", "original_lufs", "job_ok"):
            self.assertIn(key, entry)

    def test_batch_loudness_match_empty_input(self):
        """batch_loudness_match with empty file list should return empty list."""
        from opencut.core.loudness_match import batch_loudness_match

        with tempfile.TemporaryDirectory() as tmpdir:
            results = batch_loudness_match([], tmpdir)
        self.assertEqual(results, [])

    def test_normalize_to_lufs_calls_ffmpeg_twice(self):
        """normalize_to_lufs should call FFmpeg for pass 1 and pass 2."""
        from opencut.core.loudness_match import normalize_to_lufs

        fake_measured_json = json.dumps({
            "input_i": "-18.0",
            "input_tp": "-3.0",
            "input_lra": "7.0",
            "input_thresh": "-28.0",
            "target_offset": "0.0",
        })
        mock_pass1 = MagicMock()
        mock_pass1.returncode = 0
        mock_pass1.stderr = fake_measured_json

        mock_pass2 = MagicMock()
        mock_pass2.returncode = 0
        mock_pass2.stderr = ""

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "out.mp4")
            with patch("opencut.core.loudness_match._run_ffmpeg", side_effect=[mock_pass1, mock_pass2]) as mock_run:
                normalize_to_lufs("/fake/in.mp4", output_path, target_lufs=-14.0)
        self.assertEqual(mock_run.call_count, 2)


# ============================================================
# TestAutoZoomInterface
# ============================================================
class TestAutoZoomInterface(unittest.TestCase):
    """Tests for auto_zoom.py public interface."""

    def test_ease_function_bounds(self):
        """Easing function output should be in [0, 1] range for all modes."""
        from opencut.core.auto_zoom import _ease
        for mode in ["linear", "ease_in", "ease_out", "ease_in_out"]:
            for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
                val = _ease(t, mode)
                self.assertGreaterEqual(val, 0.0)
                self.assertLessEqual(val, 1.0)

    def test_ease_boundary_values(self):
        """Easing at t=0 should be 0 and t=1 should be 1 for all modes."""
        from opencut.core.auto_zoom import _ease
        for mode in ["linear", "ease_in", "ease_out", "ease_in_out"]:
            self.assertAlmostEqual(_ease(0.0, mode), 0.0, places=5)
            self.assertAlmostEqual(_ease(1.0, mode), 1.0, places=5)

    def test_ease_linear_is_identity(self):
        """linear easing should return t unchanged."""
        from opencut.core.auto_zoom import _ease
        for t in [0.0, 0.1, 0.5, 0.9, 1.0]:
            self.assertAlmostEqual(_ease(t, "linear"), t)

    def test_ease_clamps_out_of_range_input(self):
        """Values outside [0, 1] should be clamped before easing."""
        from opencut.core.auto_zoom import _ease
        self.assertAlmostEqual(_ease(-0.5, "linear"), 0.0)
        self.assertAlmostEqual(_ease(1.5, "linear"), 1.0)

    def test_ease_in_is_slower_at_start(self):
        """ease_in at t=0.5 should be less than linear (slower start)."""
        from opencut.core.auto_zoom import _ease
        self.assertLess(_ease(0.5, "ease_in"), _ease(0.5, "linear"))

    def test_ease_out_is_faster_at_start(self):
        """ease_out at t=0.5 should be greater than linear (faster start)."""
        from opencut.core.auto_zoom import _ease
        self.assertGreater(_ease(0.5, "ease_out"), _ease(0.5, "linear"))

    def test_ease_in_out_symmetric(self):
        """ease_in_out should be symmetric around t=0.5."""
        from opencut.core.auto_zoom import _ease
        # f(0.25) and 1 - f(0.75) should be equal
        self.assertAlmostEqual(_ease(0.25, "ease_in_out"), 1.0 - _ease(0.75, "ease_in_out"), places=5)

    def test_require_cv2_raises_when_unavailable(self):
        """_require_cv2 should raise RuntimeError when cv2 is not installed."""
        import opencut.core.auto_zoom as az
        original = az._CV2_AVAILABLE
        try:
            az._CV2_AVAILABLE = False
            with self.assertRaises(RuntimeError):
                az._require_cv2()
        finally:
            az._CV2_AVAILABLE = original


# ============================================================
# TestColorMatchInterface
# ============================================================
class TestColorMatchInterface(unittest.TestCase):
    """Tests for color_match.py public interface."""

    def test_require_cv2_raises_when_unavailable(self):
        """_require_cv2 should raise RuntimeError when cv2 is not installed."""
        import opencut.core.color_match as cm
        original_cv2 = cm._CV2_AVAILABLE
        original_np = cm._NP_AVAILABLE
        try:
            cm._CV2_AVAILABLE = False
            with self.assertRaises(RuntimeError):
                cm._require_cv2()
        finally:
            cm._CV2_AVAILABLE = original_cv2
            cm._NP_AVAILABLE = original_np

    def test_require_cv2_raises_when_numpy_unavailable(self):
        """_require_cv2 should raise RuntimeError when numpy is not installed."""
        import opencut.core.color_match as cm
        original_cv2 = cm._CV2_AVAILABLE
        original_np = cm._NP_AVAILABLE
        try:
            cm._NP_AVAILABLE = False
            with self.assertRaises(RuntimeError):
                cm._require_cv2()
        finally:
            cm._CV2_AVAILABLE = original_cv2
            cm._NP_AVAILABLE = original_np

    def test_color_match_video_calls_require_cv2(self):
        """color_match_video should fail gracefully if cv2 is unavailable."""
        try:
            from opencut.core.color_match import color_match_video
            import opencut.core.color_match as cm
            original_cv2 = cm._CV2_AVAILABLE
            original_np = cm._NP_AVAILABLE
            try:
                cm._CV2_AVAILABLE = False
                cm._NP_AVAILABLE = False
                with self.assertRaises(RuntimeError):
                    color_match_video("/fake/src.mp4", "/fake/ref.mp4", "/tmp/out.mp4")
            finally:
                cm._CV2_AVAILABLE = original_cv2
                cm._NP_AVAILABLE = original_np
        except ImportError:
            self.skipTest("color_match module not available")

    def test_extract_color_stats_requires_cv2(self):
        """extract_color_stats should raise RuntimeError when cv2 unavailable."""
        try:
            from opencut.core.color_match import extract_color_stats
            import opencut.core.color_match as cm
            original_cv2 = cm._CV2_AVAILABLE
            try:
                cm._CV2_AVAILABLE = False
                with self.assertRaises(RuntimeError):
                    extract_color_stats("/fake/video.mp4")
            finally:
                cm._CV2_AVAILABLE = original_cv2
        except ImportError:
            self.skipTest("color_match module not available")

    def test_build_cumulative_hist_with_numpy(self):
        """_build_cumulative_hist should produce a valid CDF array."""
        try:
            import numpy as np
            from opencut.core.color_match import _build_cumulative_hist

            # Create simple test frames: 4x4x3 arrays with known values
            frame = np.zeros((4, 4, 3), dtype=np.uint8)
            frame[:, :, 0] = 128  # all Y channel = 128

            cdf = _build_cumulative_hist([frame], channel=0)
            # CDF should end at 1.0
            self.assertAlmostEqual(float(cdf[-1]), 1.0, places=5)
            # CDF should be monotonically non-decreasing
            for i in range(len(cdf) - 1):
                self.assertLessEqual(float(cdf[i]), float(cdf[i + 1]))
        except ImportError:
            self.skipTest("numpy not available")

    def test_build_lut_maps_to_uint8_range(self):
        """_build_lut should return a LUT with values in [0, 255]."""
        try:
            import numpy as np
            from opencut.core.color_match import _build_lut

            # Use uniform CDFs — LUT should be identity-like
            src_cdf = np.linspace(0.0, 1.0, 256)
            ref_cdf = np.linspace(0.0, 1.0, 256)
            lut = _build_lut(src_cdf, ref_cdf)
            self.assertEqual(len(lut), 256)
            self.assertGreaterEqual(int(lut.min()), 0)
            self.assertLessEqual(int(lut.max()), 255)
        except ImportError:
            self.skipTest("numpy not available")


if __name__ == "__main__":
    unittest.main()
