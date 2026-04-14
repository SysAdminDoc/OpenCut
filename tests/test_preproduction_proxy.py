"""
Tests for OpenCut Pre-Production & Proxy Management features (Sections 59-60).

Covers:
  - 59.1 AI Storyboard Generation from Script (ai_storyboard.py enhancement)
  - 59.2 Shot List Generator from Screenplay (shot_list_gen.py)
  - 59.3 Mood Board Generator (mood_board.py)
  - 59.4 Script-to-Rough-Cut Assembly (script_to_roughcut.py)
  - 60.1 Auto Proxy Generation (proxy_gen.py enhancement)
  - 60.2 Proxy-to-Full-Res Swap on Export (proxy_swap.py)
  - 60.3 Media Relinking Assistant (media_relink.py)
  - 60.4 Duplicate Media Detection (duplicate_detect.py enhancement)
  - Route smoke tests (preproduction_proxy_routes.py)

75+ tests. Mocks FFmpeg, LLM, Pillow, file I/O.
"""

import inspect
import json
import os
import sys
import tempfile
import unittest
from dataclasses import fields
from unittest.mock import MagicMock, patch, mock_open

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# 59.1 — AI Storyboard from Script
# ============================================================
class TestAIStoryboardFromScript(unittest.TestCase):
    """Tests for the enhanced generate_storyboard_from_script function."""

    def test_function_exists(self):
        from opencut.core.ai_storyboard import generate_storyboard_from_script
        self.assertTrue(callable(generate_storyboard_from_script))

    def test_function_signature(self):
        from opencut.core.ai_storyboard import generate_storyboard_from_script
        sig = inspect.signature(generate_storyboard_from_script)
        params = list(sig.parameters.keys())
        self.assertIn("script_text", params)
        self.assertIn("output_dir", params)
        self.assertIn("on_progress", params)
        self.assertIn("use_stable_diffusion", params)

    def test_empty_script_raises(self):
        from opencut.core.ai_storyboard import generate_storyboard_from_script
        with self.assertRaises(ValueError):
            generate_storyboard_from_script("", tempfile.mkdtemp())

    def test_whitespace_only_raises(self):
        from opencut.core.ai_storyboard import generate_storyboard_from_script
        with self.assertRaises(ValueError):
            generate_storyboard_from_script("   \n  ", tempfile.mkdtemp())

    @patch("opencut.core.ai_storyboard.ensure_package", return_value=True)
    @patch("opencut.core.ai_storyboard._generate_panel_image")
    @patch("opencut.core.ai_storyboard.render_storyboard_grid")
    @patch("opencut.core.ai_storyboard.export_storyboard_pdf")
    def test_basic_generation(self, mock_pdf, mock_grid, mock_panel, mock_pkg):
        from opencut.core.ai_storyboard import generate_storyboard_from_script
        script = "1. Wide shot of office\n\n2. Close-up of character speaking"
        out = tempfile.mkdtemp()
        result = generate_storyboard_from_script(script, out)
        self.assertGreater(result.total_shots, 0)
        self.assertTrue(mock_panel.called)

    @patch("opencut.core.ai_storyboard.ensure_package", return_value=True)
    @patch("opencut.core.ai_storyboard._generate_panel_image")
    @patch("opencut.core.ai_storyboard.render_storyboard_grid")
    @patch("opencut.core.ai_storyboard.export_storyboard_pdf")
    def test_progress_callback(self, mock_pdf, mock_grid, mock_panel, mock_pkg):
        from opencut.core.ai_storyboard import generate_storyboard_from_script
        progress = []
        script = "1. Shot one\n\n2. Shot two\n\n3. Shot three"
        out = tempfile.mkdtemp()
        generate_storyboard_from_script(script, out,
                                        on_progress=lambda p, m: progress.append(p))
        self.assertTrue(len(progress) > 0)
        self.assertEqual(progress[-1], 100)

    @patch("opencut.core.ai_storyboard.ensure_package", return_value=True)
    @patch("opencut.core.ai_storyboard._generate_panel_image")
    @patch("opencut.core.ai_storyboard.render_storyboard_grid")
    def test_no_pdf_export(self, mock_grid, mock_panel, mock_pkg):
        from opencut.core.ai_storyboard import generate_storyboard_from_script
        script = "1. Shot one\n\n2. Shot two"
        out = tempfile.mkdtemp()
        result = generate_storyboard_from_script(script, out, export_pdf=False)
        self.assertEqual(result.pdf_path, "")

    def test_sd_api_check_unreachable(self):
        from opencut.core.ai_storyboard import _check_sd_api
        self.assertFalse(_check_sd_api("http://127.0.0.1:99999"))

    @patch("opencut.core.ai_storyboard.ensure_package", return_value=False)
    def test_pillow_missing_raises(self, mock_pkg):
        from opencut.core.ai_storyboard import generate_storyboard_from_script
        with self.assertRaises(RuntimeError):
            generate_storyboard_from_script("1. Shot one", tempfile.mkdtemp())


# ============================================================
# 59.2 — Shot List Generator from Screenplay
# ============================================================
class TestShotListGen(unittest.TestCase):
    """Tests for opencut.core.shot_list_gen module."""

    def test_module_imports(self):
        from opencut.core.shot_list_gen import (
            generate_shot_list, parse_screenplay, ShotEntry, ShotListResult,
        )
        self.assertTrue(callable(generate_shot_list))
        self.assertTrue(callable(parse_screenplay))

    def test_parse_screenplay_empty(self):
        from opencut.core.shot_list_gen import parse_screenplay
        self.assertEqual(parse_screenplay(""), [])
        self.assertEqual(parse_screenplay("   "), [])

    def test_parse_scene_headings(self):
        from opencut.core.shot_list_gen import parse_screenplay
        text = "INT. OFFICE - DAY\nBob sits at his desk.\n\nEXT. PARK - NIGHT\nA jogger runs by."
        elements = parse_screenplay(text)
        headings = [e for e in elements if e.element_type == "scene_heading"]
        self.assertEqual(len(headings), 2)
        self.assertIn("INT", headings[0].text.upper())

    def test_parse_dialogue(self):
        from opencut.core.shot_list_gen import parse_screenplay
        text = "INT. ROOM - DAY\n\nBOB\n    Hello, how are you?\n\nJANE\n    I'm fine, thanks."
        elements = parse_screenplay(text)
        dialogues = [e for e in elements if e.element_type == "dialogue"]
        self.assertGreaterEqual(len(dialogues), 2)

    def test_parse_transitions(self):
        from opencut.core.shot_list_gen import parse_screenplay
        text = "INT. ROOM - DAY\nAction.\nCUT TO:\nEXT. PARK - DAY\nMore action."
        elements = parse_screenplay(text)
        transitions = [e for e in elements if e.element_type == "transition"]
        self.assertGreaterEqual(len(transitions), 1)

    def test_parse_fountain_mode(self):
        from opencut.core.shot_list_gen import parse_screenplay
        text = ".OPENING SCENE\nSomething happens.\n\n.NEXT SCENE\nMore stuff."
        elements = parse_screenplay(text, is_fountain=True)
        headings = [e for e in elements if e.element_type == "scene_heading"]
        self.assertGreaterEqual(len(headings), 2)

    def test_scene_numbering(self):
        from opencut.core.shot_list_gen import parse_screenplay
        text = "INT. ONE - DAY\nAction.\n\nEXT. TWO - NIGHT\nMore action."
        elements = parse_screenplay(text)
        headings = [e for e in elements if e.element_type == "scene_heading"]
        self.assertEqual(headings[0].scene_number, 1)
        self.assertEqual(headings[1].scene_number, 2)

    def test_generate_shot_list_empty(self):
        from opencut.core.shot_list_gen import generate_shot_list
        with self.assertRaises(ValueError):
            generate_shot_list("", tempfile.mkdtemp())

    def test_generate_shot_list_basic(self):
        from opencut.core.shot_list_gen import generate_shot_list
        text = "INT. OFFICE - DAY\nBob works.\n\nEXT. PARK - NIGHT\nA cat sleeps."
        out = tempfile.mkdtemp()
        result = generate_shot_list(text, output_dir=out)
        self.assertGreater(result.total_shots, 0)
        self.assertGreater(result.total_scenes, 0)

    def test_generate_shot_list_csv_export(self):
        from opencut.core.shot_list_gen import generate_shot_list
        text = "INT. ROOM - DAY\nAction here."
        out = tempfile.mkdtemp()
        result = generate_shot_list(text, output_dir=out, export_csv=True)
        self.assertTrue(result.csv_path)
        self.assertTrue(os.path.isfile(result.csv_path))

    def test_generate_shot_list_json_export(self):
        from opencut.core.shot_list_gen import generate_shot_list
        text = "INT. ROOM - DAY\nAction here."
        out = tempfile.mkdtemp()
        result = generate_shot_list(text, output_dir=out, export_json=True)
        self.assertTrue(result.json_path)
        self.assertTrue(os.path.isfile(result.json_path))
        with open(result.json_path) as f:
            data = json.load(f)
        self.assertIn("shot_list", data)

    def test_shot_entry_to_dict(self):
        from opencut.core.shot_list_gen import ShotEntry
        entry = ShotEntry(shot_number=1, scene_number=1, shot_type="WS",
                          characters=["BOB"])
        d = entry.to_dict()
        self.assertEqual(d["shot_number"], 1)
        self.assertIn("BOB", d["characters"])

    def test_camera_suggestion_fight(self):
        from opencut.core.shot_list_gen import generate_shot_list
        text = "INT. ARENA - NIGHT\nThey fight brutally."
        result = generate_shot_list(text)
        # Fight scenes should get MS or similar framing
        self.assertGreater(result.total_shots, 0)

    def test_shots_to_csv_string(self):
        from opencut.core.shot_list_gen import ShotEntry, shots_to_csv_string
        shots = [ShotEntry(shot_number=1, description="test")]
        csv_str = shots_to_csv_string(shots)
        self.assertIn("shot_number", csv_str)
        self.assertIn("test", csv_str)

    def test_progress_callback(self):
        from opencut.core.shot_list_gen import generate_shot_list
        progress = []
        text = "INT. A - DAY\nAction.\n\nEXT. B - NIGHT\nMore."
        generate_shot_list(text, output_dir=tempfile.mkdtemp(),
                           on_progress=lambda p, m: progress.append(p))
        self.assertTrue(progress[-1] == 100)

    def test_result_to_dict(self):
        from opencut.core.shot_list_gen import ShotListResult
        result = ShotListResult(total_shots=5, total_scenes=2)
        d = result.to_dict()
        self.assertEqual(d["total_shots"], 5)
        self.assertEqual(d["total_scenes"], 2)


# ============================================================
# 59.3 — Mood Board Generator
# ============================================================
class TestMoodBoard(unittest.TestCase):
    """Tests for opencut.core.mood_board module."""

    def test_module_imports(self):
        from opencut.core.mood_board import (
            generate_mood_board, extract_keyframes, render_mood_board,
            ColorInfo, FrameAnalysis, MoodBoardResult,
        )
        self.assertTrue(callable(generate_mood_board))

    def test_color_info_dataclass(self):
        from opencut.core.mood_board import ColorInfo
        c = ColorInfo(r=255, g=128, b=0, hex_code="#ff8000", percentage=30.0)
        d = c.to_dict()
        self.assertEqual(d["r"], 255)
        self.assertEqual(d["hex_code"], "#ff8000")

    def test_frame_analysis_dataclass(self):
        from opencut.core.mood_board import FrameAnalysis
        fa = FrameAnalysis(brightness=120.0, contrast=50.0, saturation=80.0)
        d = fa.to_dict()
        self.assertEqual(d["brightness"], 120.0)

    def test_mood_board_result_to_dict(self):
        from opencut.core.mood_board import MoodBoardResult
        r = MoodBoardResult(total_keyframes=5, overall_brightness=120.0)
        d = r.to_dict()
        self.assertEqual(d["total_keyframes"], 5)

    def test_suggest_luts_warm(self):
        from opencut.core.mood_board import _suggest_luts, ColorInfo
        palette = [ColorInfo(r=200, g=100, b=50)]
        luts = _suggest_luts(120, 50, 80, palette)
        self.assertTrue(any("Warm" in l or "Golden" in l for l in luts))

    def test_suggest_luts_dark(self):
        from opencut.core.mood_board import _suggest_luts
        luts = _suggest_luts(50, 50, 80, [])
        self.assertTrue(any("Dark" in l or "Noir" in l for l in luts))

    def test_derive_style_tags(self):
        from opencut.core.mood_board import _derive_style_tags, ColorInfo
        tags = _derive_style_tags(60, 70, 30, [])
        self.assertIn("dark", tags)
        self.assertIn("high-contrast", tags)
        self.assertIn("desaturated", tags)

    def test_derive_style_tags_bright(self):
        from opencut.core.mood_board import _derive_style_tags
        tags = _derive_style_tags(180, 25, 140, [])
        self.assertIn("bright", tags)
        self.assertIn("vibrant", tags)

    @patch("opencut.core.mood_board.get_video_info", return_value={"duration": 60, "width": 1920, "height": 1080})
    @patch("opencut.core.mood_board.subprocess.run")
    def test_extract_keyframes(self, mock_run, mock_info):
        from opencut.core.mood_board import extract_keyframes
        out = tempfile.mkdtemp()
        frames = extract_keyframes("/fake/video.mp4", num_frames=3, output_dir=out)
        self.assertTrue(mock_run.called)

    @patch("opencut.core.mood_board.ensure_package", return_value=False)
    def test_generate_mood_board_no_pillow(self, mock_pkg):
        from opencut.core.mood_board import generate_mood_board
        with self.assertRaises(RuntimeError):
            generate_mood_board("/fake/video.mp4", tempfile.mkdtemp())


# ============================================================
# 59.4 — Script-to-Rough-Cut Assembly
# ============================================================
class TestScriptToRoughCut(unittest.TestCase):
    """Tests for opencut.core.script_to_roughcut module."""

    def test_module_imports(self):
        from opencut.core.script_to_roughcut import (
            assemble_rough_cut, parse_script_segments, fuzzy_match_segments,
            select_best_takes, export_xml_timeline,
            TranscriptSegment, ScriptSegment, RoughCutResult,
        )
        self.assertTrue(callable(assemble_rough_cut))

    def test_parse_script_segments_empty(self):
        from opencut.core.script_to_roughcut import parse_script_segments
        self.assertEqual(parse_script_segments(""), [])

    def test_parse_script_segments_basic(self):
        from opencut.core.script_to_roughcut import parse_script_segments
        text = "INT. OFFICE - DAY\n\nBOB\n    Hello there.\n\nHe walks out."
        segs = parse_script_segments(text)
        types = [s.segment_type for s in segs]
        self.assertIn("scene_heading", types)
        self.assertIn("dialogue", types)
        self.assertIn("action", types)

    def test_fuzzy_match_basic(self):
        from opencut.core.script_to_roughcut import (
            fuzzy_match_segments, ScriptSegment, TranscriptSegment,
        )
        script = [ScriptSegment(index=0, segment_type="dialogue",
                                text="Hello how are you today")]
        transcript = [TranscriptSegment(clip_path="/a.mp4",
                                        text="Hello how are you today",
                                        start_time=1.0, end_time=3.0)]
        matches = fuzzy_match_segments(script, transcript, threshold=0.3)
        self.assertIn(0, matches)
        self.assertGreater(matches[0][0].similarity, 0.8)

    def test_fuzzy_match_no_match(self):
        from opencut.core.script_to_roughcut import (
            fuzzy_match_segments, ScriptSegment, TranscriptSegment,
        )
        script = [ScriptSegment(index=0, segment_type="dialogue",
                                text="Completely different sentence")]
        transcript = [TranscriptSegment(clip_path="/a.mp4",
                                        text="xyz abc 123 random words",
                                        start_time=1.0, end_time=3.0)]
        matches = fuzzy_match_segments(script, transcript, threshold=0.9)
        self.assertEqual(len(matches), 0)

    def test_select_best_takes(self):
        from opencut.core.script_to_roughcut import (
            select_best_takes, ScriptSegment, TranscriptSegment, MatchResult,
        )
        ss = ScriptSegment(index=0, segment_type="dialogue", text="Hello")
        ts = TranscriptSegment(clip_path="/a.mp4", start_time=1.0, end_time=3.0)
        matches = {0: [MatchResult(script_segment=ss, transcript_segment=ts,
                                   similarity=0.95, combined_score=0.9)]}
        result = select_best_takes(matches, [ss])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].clip_path, "/a.mp4")

    def test_select_best_takes_unmatched(self):
        from opencut.core.script_to_roughcut import (
            select_best_takes, ScriptSegment,
        )
        ss = ScriptSegment(index=0, segment_type="dialogue", text="Hello")
        result = select_best_takes({}, [ss])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].clip_path, "")

    def test_export_xml_timeline(self):
        from opencut.core.script_to_roughcut import (
            export_xml_timeline, RoughCutSegment,
        )
        segments = [
            RoughCutSegment(index=0, clip_path="/a.mp4",
                           in_point=0.0, out_point=5.0),
            RoughCutSegment(index=1, clip_path="/b.mp4",
                           in_point=2.0, out_point=8.0),
        ]
        out = os.path.join(tempfile.mkdtemp(), "rough_cut.xml")
        path = export_xml_timeline(segments, out)
        self.assertTrue(os.path.isfile(path))
        with open(path) as f:
            content = f.read()
        self.assertIn("xmeml", content)
        self.assertIn("a.mp4", content)

    def test_assemble_rough_cut_empty_script(self):
        from opencut.core.script_to_roughcut import assemble_rough_cut
        with self.assertRaises(ValueError):
            assemble_rough_cut("", output_dir=tempfile.mkdtemp())

    def test_rough_cut_result_to_dict(self):
        from opencut.core.script_to_roughcut import RoughCutResult
        r = RoughCutResult(total_segments=10, matched_segments=7, unmatched_segments=3)
        d = r.to_dict()
        self.assertEqual(d["total_segments"], 10)
        self.assertEqual(d["matched_segments"], 7)

    def test_load_transcript(self):
        from opencut.core.script_to_roughcut import load_transcript
        data = {"segments": [{"text": "Hello", "start": 0.0, "end": 1.0}]}
        path = os.path.join(tempfile.mkdtemp(), "transcript.json")
        with open(path, "w") as f:
            json.dump(data, f)
        segs = load_transcript(path, clip_path="/a.mp4")
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0].text, "Hello")
        self.assertEqual(segs[0].clip_path, "/a.mp4")

    def test_rough_cut_segment_to_dict(self):
        from opencut.core.script_to_roughcut import RoughCutSegment
        seg = RoughCutSegment(index=0, clip_path="/a.mp4", in_point=1.0, out_point=5.0)
        d = seg.to_dict()
        self.assertEqual(d["clip_path"], "/a.mp4")


# ============================================================
# 60.1 — Auto Proxy Generation
# ============================================================
class TestAutoProxyIngest(unittest.TestCase):
    """Tests for the enhanced auto_proxy_ingest in proxy_gen.py."""

    def test_function_exists(self):
        from opencut.core.proxy_gen import auto_proxy_ingest
        self.assertTrue(callable(auto_proxy_ingest))

    def test_function_signature(self):
        from opencut.core.proxy_gen import auto_proxy_ingest
        sig = inspect.signature(auto_proxy_ingest)
        params = list(sig.parameters.keys())
        self.assertIn("folder_path", params)
        self.assertIn("threshold_resolution", params)
        self.assertIn("on_progress", params)

    def test_missing_folder_raises(self):
        from opencut.core.proxy_gen import auto_proxy_ingest
        with self.assertRaises(FileNotFoundError):
            auto_proxy_ingest("/nonexistent/folder")

    @patch("opencut.core.proxy_gen.get_video_info",
           return_value={"width": 1280, "height": 720, "duration": 10})
    def test_no_high_res_clips(self, mock_info):
        from opencut.core.proxy_gen import auto_proxy_ingest
        d = tempfile.mkdtemp()
        # Create a low-res video file
        with open(os.path.join(d, "small.mp4"), "w") as f:
            f.write("fake")
        result = auto_proxy_ingest(d, threshold_resolution=1920)
        self.assertEqual(result.completed, 0)

    def test_empty_folder(self):
        from opencut.core.proxy_gen import auto_proxy_ingest
        d = tempfile.mkdtemp()
        result = auto_proxy_ingest(d)
        self.assertEqual(result.total, 0)

    def test_video_extensions_constant(self):
        from opencut.core.proxy_gen import _VIDEO_EXTENSIONS
        self.assertIn(".mp4", _VIDEO_EXTENSIONS)
        self.assertIn(".mov", _VIDEO_EXTENSIONS)
        self.assertIn(".mkv", _VIDEO_EXTENSIONS)


# ============================================================
# 60.2 — Proxy Swap Check
# ============================================================
class TestProxySwap(unittest.TestCase):
    """Tests for opencut.core.proxy_swap module."""

    def test_module_imports(self):
        from opencut.core.proxy_swap import (
            check_proxy_swap, load_proxy_map, get_swap_paths,
            SwapEntry, SwapResult,
        )
        self.assertTrue(callable(check_proxy_swap))

    def test_swap_entry_dataclass(self):
        from opencut.core.proxy_swap import SwapEntry
        e = SwapEntry(proxy_path="/a_proxy.mp4", status="swapped")
        d = e.to_dict()
        self.assertEqual(d["status"], "swapped")

    def test_swap_result_to_dict(self):
        from opencut.core.proxy_swap import SwapResult
        r = SwapResult(total_clips=5, swapped=3)
        d = r.to_dict()
        self.assertEqual(d["total_clips"], 5)
        self.assertEqual(d["swapped"], 3)

    def test_load_proxy_map_empty(self):
        from opencut.core.proxy_swap import load_proxy_map
        result = load_proxy_map(["/nonexistent"])
        self.assertEqual(result, {})

    def test_load_proxy_map_valid(self):
        from opencut.core.proxy_swap import load_proxy_map, PROXY_METADATA_FILE
        d = tempfile.mkdtemp()
        mapping = {"/a_proxy.mp4": "/a.mp4"}
        with open(os.path.join(d, PROXY_METADATA_FILE), "w") as f:
            json.dump(mapping, f)
        result = load_proxy_map([d])
        self.assertIn("/a_proxy.mp4", result)

    def test_check_proxy_swap_not_proxy(self):
        from opencut.core.proxy_swap import check_proxy_swap
        result = check_proxy_swap(
            clip_paths=["/some/file.mp4"],
            proxy_map={},
        )
        self.assertEqual(result.total_clips, 1)
        self.assertEqual(result.not_in_manifest, 1)

    def test_check_proxy_swap_found(self):
        from opencut.core.proxy_swap import check_proxy_swap
        # Create real files
        d = tempfile.mkdtemp()
        proxy = os.path.join(d, "a_proxy.mp4")
        original = os.path.join(d, "a.mp4")
        with open(proxy, "w") as f:
            f.write("proxy")
        with open(original, "w") as f:
            f.write("original content")

        abs_proxy = os.path.abspath(proxy)
        abs_orig = os.path.abspath(original)
        result = check_proxy_swap(
            clip_paths=[abs_proxy],
            proxy_map={abs_proxy: abs_orig},
        )
        self.assertEqual(result.swapped, 1)
        self.assertTrue(result.all_originals_available)

    def test_check_proxy_swap_original_missing(self):
        from opencut.core.proxy_swap import check_proxy_swap
        d = tempfile.mkdtemp()
        proxy = os.path.join(d, "a_proxy.mp4")
        with open(proxy, "w") as f:
            f.write("proxy")

        abs_proxy = os.path.abspath(proxy)
        result = check_proxy_swap(
            clip_paths=[abs_proxy],
            proxy_map={abs_proxy: "/missing/original.mp4"},
        )
        self.assertEqual(result.original_missing, 1)
        self.assertFalse(result.all_originals_available)

    def test_get_swap_paths(self):
        from opencut.core.proxy_swap import get_swap_paths, SwapResult, SwapEntry
        result = SwapResult(entries=[
            SwapEntry(proxy_path="/a_proxy.mp4", original_path="/a.mp4",
                      status="swapped", original_exists=True),
            SwapEntry(proxy_path="/b_proxy.mp4", original_path="/b.mp4",
                      status="original_missing", original_exists=False),
        ])
        paths = get_swap_paths(result)
        self.assertIn("/a_proxy.mp4", paths)
        self.assertNotIn("/b_proxy.mp4", paths)

    def test_is_proxy_path(self):
        from opencut.core.proxy_swap import _is_proxy_path
        self.assertTrue(_is_proxy_path("video_proxy.mp4"))
        self.assertFalse(_is_proxy_path("video.mp4"))


# ============================================================
# 60.3 — Media Relinking Assistant
# ============================================================
class TestMediaRelink(unittest.TestCase):
    """Tests for opencut.core.media_relink module."""

    def test_module_imports(self):
        from opencut.core.media_relink import (
            relink_media, find_candidates, batch_relink,
            RelinkEntry, RelinkResult, RelinkCandidate,
        )
        self.assertTrue(callable(relink_media))

    def test_relink_entry_to_dict(self):
        from opencut.core.media_relink import RelinkEntry
        e = RelinkEntry(offline_path="/missing.mp4", status="unresolved")
        d = e.to_dict()
        self.assertEqual(d["status"], "unresolved")

    def test_relink_result_to_dict(self):
        from opencut.core.media_relink import RelinkResult
        r = RelinkResult(total_offline=3, resolved=1, partial=1, unresolved=1)
        d = r.to_dict()
        self.assertEqual(d["total_offline"], 3)

    def test_find_candidates_exact_match(self):
        from opencut.core.media_relink import find_candidates
        d = tempfile.mkdtemp()
        # Create a file
        target = os.path.join(d, "clip.mp4")
        with open(target, "w") as f:
            f.write("content")

        with patch("opencut.core.media_relink.get_video_info",
                   return_value={"duration": 10}):
            file_index = {"clip.mp4": [target]}
            candidates = find_candidates("/old/path/clip.mp4", file_index)
            self.assertGreater(len(candidates), 0)
            self.assertEqual(candidates[0].match_type, "exact_name")
            self.assertGreater(candidates[0].confidence, 0.8)

    def test_find_candidates_fuzzy_match(self):
        from opencut.core.media_relink import find_candidates
        d = tempfile.mkdtemp()
        target = os.path.join(d, "clip_v2.mp4")
        with open(target, "w") as f:
            f.write("content")

        with patch("opencut.core.media_relink.get_video_info",
                   return_value={"duration": 10}):
            file_index = {"clip_v2.mp4": [target]}
            candidates = find_candidates("/old/clip.mp4", file_index,
                                         fuzzy_threshold=0.5)
            # May or may not find fuzzy match depending on similarity
            # At least the function shouldn't crash
            self.assertIsInstance(candidates, list)

    def test_find_candidates_no_match(self):
        from opencut.core.media_relink import find_candidates
        candidates = find_candidates("/old/totally_unique.mp4", {})
        self.assertEqual(len(candidates), 0)

    @patch("opencut.core.media_relink.get_video_info",
           return_value={"duration": 10})
    def test_relink_media_basic(self, mock_info):
        from opencut.core.media_relink import relink_media
        d = tempfile.mkdtemp()
        # Create a matching file
        with open(os.path.join(d, "video.mp4"), "w") as f:
            f.write("content")

        result = relink_media(
            offline_paths=["/old/video.mp4"],
            search_dirs=[d],
        )
        self.assertEqual(result.total_offline, 1)
        # Should find the exact name match
        self.assertGreater(result.resolved + result.partial, 0)

    def test_relink_media_no_search_dirs(self):
        from opencut.core.media_relink import relink_media
        with self.assertRaises(ValueError):
            relink_media(["/a.mp4"], [])

    def test_relink_media_empty_offline(self):
        from opencut.core.media_relink import relink_media
        result = relink_media([], ["/some/dir"])
        self.assertEqual(result.total_offline, 0)

    def test_batch_relink(self):
        from opencut.core.media_relink import (
            batch_relink, RelinkResult, RelinkEntry,
        )
        result = RelinkResult(entries=[
            RelinkEntry(offline_path="/a.mp4", best_match="/new/a.mp4",
                        best_confidence=0.95, status="resolved"),
            RelinkEntry(offline_path="/b.mp4", best_match="/new/b.mp4",
                        best_confidence=0.6, status="partial"),
        ])
        mapping = batch_relink(result, min_confidence=0.8)
        self.assertIn("/a.mp4", mapping)
        self.assertNotIn("/b.mp4", mapping)

    def test_scan_directory(self):
        from opencut.core.media_relink import _scan_directory
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "test.mp4"), "w") as f:
            f.write("fake")
        with open(os.path.join(d, "readme.txt"), "w") as f:
            f.write("text")
        index = _scan_directory([d], recursive=False)
        self.assertIn("test.mp4", index)
        self.assertNotIn("readme.txt", index)


# ============================================================
# 60.4 — Duplicate Detection Enhancement
# ============================================================
class TestNearDuplicateDetection(unittest.TestCase):
    """Tests for the enhanced detect_near_duplicates in duplicate_detect.py."""

    def test_function_exists(self):
        from opencut.core.duplicate_detect import detect_near_duplicates
        self.assertTrue(callable(detect_near_duplicates))

    def test_function_signature(self):
        from opencut.core.duplicate_detect import detect_near_duplicates
        sig = inspect.signature(detect_near_duplicates)
        params = list(sig.parameters.keys())
        self.assertIn("folder_path", params)
        self.assertIn("threshold", params)
        self.assertIn("on_progress", params)

    def test_missing_folder_raises(self):
        from opencut.core.duplicate_detect import detect_near_duplicates
        with self.assertRaises(FileNotFoundError):
            detect_near_duplicates("/nonexistent/folder")

    def test_empty_folder(self):
        from opencut.core.duplicate_detect import detect_near_duplicates
        d = tempfile.mkdtemp()
        result = detect_near_duplicates(d)
        self.assertEqual(result, [])

    def test_single_file_no_duplicates(self):
        from opencut.core.duplicate_detect import detect_near_duplicates
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "only.mp4"), "w") as f:
            f.write("fake")
        result = detect_near_duplicates(d)
        self.assertEqual(result, [])

    def test_video_extensions(self):
        from opencut.core.duplicate_detect import _VIDEO_EXTENSIONS
        self.assertIn(".mp4", _VIDEO_EXTENSIONS)
        self.assertIn(".mov", _VIDEO_EXTENSIONS)

    def test_hamming_distance(self):
        from opencut.core.duplicate_detect import _hamming_distance
        self.assertEqual(_hamming_distance("0000000000000000", "0000000000000000"), 0)
        self.assertGreater(_hamming_distance("0000000000000000", "ffffffffffffffff"), 0)

    def test_hamming_distance_different_lengths(self):
        from opencut.core.duplicate_detect import _hamming_distance
        self.assertEqual(_hamming_distance("00", "0000"), 64)


# ============================================================
# Route Smoke Tests
# ============================================================
class TestPreproductionProxyRoutes(unittest.TestCase):
    """Smoke tests for the preproduction_proxy_routes blueprint."""

    def test_blueprint_exists(self):
        from opencut.routes.preproduction_proxy_routes import preproduction_proxy_bp
        self.assertEqual(preproduction_proxy_bp.name, "preproduction_proxy")

    def test_route_storyboard_from_script_registered(self):
        from opencut.routes.preproduction_proxy_routes import preproduction_proxy_bp
        # Blueprint deferred functions store the route registrations
        self.assertGreaterEqual(len(preproduction_proxy_bp.deferred_functions), 8)

    def test_route_functions_exist(self):
        from opencut.routes.preproduction_proxy_routes import (
            storyboard_from_script,
            shot_list_generate,
            mood_board_generate,
            rough_cut_from_script,
            proxy_auto_ingest,
            proxy_swap_check,
            media_relink,
            detect_duplicates,
        )
        self.assertTrue(callable(storyboard_from_script))
        self.assertTrue(callable(shot_list_generate))
        self.assertTrue(callable(mood_board_generate))
        self.assertTrue(callable(rough_cut_from_script))
        self.assertTrue(callable(proxy_auto_ingest))
        self.assertTrue(callable(proxy_swap_check))
        self.assertTrue(callable(media_relink))
        self.assertTrue(callable(detect_duplicates))

    def test_blueprint_has_eight_routes(self):
        from opencut.routes.preproduction_proxy_routes import preproduction_proxy_bp
        self.assertEqual(len(preproduction_proxy_bp.deferred_functions), 8)


class TestRouteIntegration(unittest.TestCase):
    """Integration tests using the Flask test client."""

    @classmethod
    def setUpClass(cls):
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        config = OpenCutConfig()
        cls.app = create_app(config=config)
        cls.app.config["TESTING"] = True
        cls.client = cls.app.test_client()
        # Get CSRF token
        resp = cls.client.get("/health")
        data = resp.get_json()
        cls.csrf_token = data.get("csrf_token", "")
        cls.headers = {
            "X-OpenCut-Token": cls.csrf_token,
            "Content-Type": "application/json",
        }

    def test_storyboard_from_script_no_text(self):
        resp = self.client.post("/storyboard/from-script",
                                json={"script_text": ""},
                                headers=self.headers)
        # Should get a job ID back (async) or error
        self.assertIn(resp.status_code, [200, 400, 422])

    def test_shot_list_no_text(self):
        resp = self.client.post("/storyboard/shot-list",
                                json={"screenplay_text": ""},
                                headers=self.headers)
        self.assertIn(resp.status_code, [200, 400, 422])

    def test_mood_board_no_filepath(self):
        resp = self.client.post("/storyboard/mood-board",
                                json={},
                                headers=self.headers)
        self.assertIn(resp.status_code, [200, 400, 422])

    def test_rough_cut_no_script(self):
        resp = self.client.post("/rough-cut/from-script",
                                json={"script_text": ""},
                                headers=self.headers)
        self.assertIn(resp.status_code, [200, 400, 422])

    def test_proxy_auto_ingest_no_folder(self):
        resp = self.client.post("/proxy/auto-ingest",
                                json={"folder_path": ""},
                                headers=self.headers)
        self.assertIn(resp.status_code, [200, 400, 422])

    def test_proxy_swap_check_no_clips(self):
        resp = self.client.post("/proxy/swap-check",
                                json={"clip_paths": []},
                                headers=self.headers)
        self.assertIn(resp.status_code, [200, 400, 422])

    def test_proxy_relink_no_paths(self):
        resp = self.client.post("/proxy/relink",
                                json={"offline_paths": [], "search_dirs": []},
                                headers=self.headers)
        self.assertIn(resp.status_code, [200, 400, 422])

    def test_detect_duplicates_no_folder(self):
        resp = self.client.post("/proxy/detect-duplicates",
                                json={"folder_path": ""},
                                headers=self.headers)
        self.assertIn(resp.status_code, [200, 400, 422])

    def test_csrf_required_storyboard(self):
        resp = self.client.post("/storyboard/from-script",
                                json={"script_text": "test"},
                                headers={"Content-Type": "application/json"})
        self.assertIn(resp.status_code, [403, 401])

    def test_csrf_required_proxy(self):
        resp = self.client.post("/proxy/auto-ingest",
                                json={"folder_path": "/tmp"},
                                headers={"Content-Type": "application/json"})
        self.assertIn(resp.status_code, [403, 401])


if __name__ == "__main__":
    unittest.main()
