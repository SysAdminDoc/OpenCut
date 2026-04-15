"""
Tests for OpenCut Audio Post-Production features (Category 82).

Covers:
  - ADR Cue System (session CRUD, cue CRUD, cue sheet CSV/JSON, guide audio, replacement)
  - M&E Mix (stem separation mock, track mute, spectral, loudness matching, residual)
  - Dialogue Premix (per-speaker pipeline, EQ presets, compressor, LUFS targeting)
  - Surround Upmix (all modes, pan coefficients, channel routing, downmix validation)
  - Foley Cueing (event detection, cue sheet export, SFX placement, category matching)
  - Audio Post Routes (smoke tests for all 13 endpoints)
"""

import json
import os
import sys
import tempfile
import unittest
from dataclasses import fields
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# ADR Cue System Tests
# ============================================================
class TestADRCueSystem(unittest.TestCase):
    """Tests for opencut.core.adr_cue_system module."""

    def test_adr_cue_dataclass_fields(self):
        from opencut.core.adr_cue_system import ADRCue
        f_names = {f.name for f in fields(ADRCue)}
        self.assertIn("cue_id", f_names)
        self.assertIn("character_name", f_names)
        self.assertIn("original_line", f_names)
        self.assertIn("timecode_in", f_names)
        self.assertIn("timecode_out", f_names)
        self.assertIn("scene_context", f_names)
        self.assertIn("reason", f_names)
        self.assertIn("priority", f_names)
        self.assertIn("status", f_names)

    def test_adr_session_dataclass_fields(self):
        from opencut.core.adr_cue_system import ADRSession
        f_names = {f.name for f in fields(ADRSession)}
        self.assertIn("session_id", f_names)
        self.assertIn("project_name", f_names)
        self.assertIn("source_path", f_names)
        self.assertIn("cues", f_names)
        self.assertIn("fps", f_names)
        self.assertIn("reel", f_names)

    def test_adr_cue_duration(self):
        from opencut.core.adr_cue_system import ADRCue
        cue = ADRCue(timecode_in=10.0, timecode_out=15.5)
        self.assertAlmostEqual(cue.duration(), 5.5)

    def test_adr_cue_duration_negative(self):
        from opencut.core.adr_cue_system import ADRCue
        cue = ADRCue(timecode_in=20.0, timecode_out=10.0)
        self.assertEqual(cue.duration(), 0.0)

    def test_adr_cue_to_dict(self):
        from opencut.core.adr_cue_system import ADRCue
        cue = ADRCue(cue_id="abc", character_name="Alice",
                     timecode_in=1.0, timecode_out=3.0)
        d = cue.to_dict()
        self.assertEqual(d["cue_id"], "abc")
        self.assertEqual(d["character_name"], "Alice")
        self.assertIn("duration", d)
        self.assertAlmostEqual(d["duration"], 2.0)

    def test_adr_session_to_dict(self):
        from opencut.core.adr_cue_system import ADRCue, ADRSession
        session = ADRSession(session_id="s1", project_name="Test")
        session.cues.append(ADRCue(timecode_in=0, timecode_out=2.0))
        d = session.to_dict()
        self.assertEqual(d["session_id"], "s1")
        self.assertEqual(d["cue_count"], 1)
        self.assertAlmostEqual(d["total_duration"], 2.0)

    def test_seconds_to_timecode(self):
        from opencut.core.adr_cue_system import _seconds_to_timecode
        tc = _seconds_to_timecode(3661.5, fps=24.0)
        # 1 hour, 1 min, 1 sec, 12 frames
        self.assertEqual(tc, "01:01:01:12")

    def test_seconds_to_timecode_zero(self):
        from opencut.core.adr_cue_system import _seconds_to_timecode
        tc = _seconds_to_timecode(0.0, fps=24.0)
        self.assertEqual(tc, "00:00:00:00")

    def test_timecode_to_seconds(self):
        from opencut.core.adr_cue_system import _timecode_to_seconds
        s = _timecode_to_seconds("01:00:00:00", fps=24.0)
        self.assertAlmostEqual(s, 3600.0, places=1)

    def test_timecode_to_seconds_invalid(self):
        from opencut.core.adr_cue_system import _timecode_to_seconds
        with self.assertRaises(ValueError):
            _timecode_to_seconds("bad_timecode")

    @patch("opencut.core.adr_cue_system._save_session")
    @patch("opencut.core.adr_cue_system._load_session")
    def test_add_cue(self, mock_load, mock_save):
        from opencut.core.adr_cue_system import ADRSession, add_cue
        session = ADRSession(session_id="s1", project_name="Test")
        mock_load.return_value = session

        cue = add_cue("s1", "Bob", "Hello world", 10.0, 12.0, reason="noise")
        self.assertEqual(cue.character_name, "Bob")
        self.assertEqual(cue.original_line, "Hello world")
        self.assertEqual(cue.reason, "noise")
        self.assertEqual(cue.status, "pending")
        mock_save.assert_called_once()

    @patch("opencut.core.adr_cue_system._save_session")
    @patch("opencut.core.adr_cue_system._load_session")
    def test_add_cue_invalid_session(self, mock_load, mock_save):
        from opencut.core.adr_cue_system import add_cue
        mock_load.return_value = None
        with self.assertRaises(ValueError):
            add_cue("bad_id", "Bob", "Hello", 1.0, 2.0)

    @patch("opencut.core.adr_cue_system._save_session")
    @patch("opencut.core.adr_cue_system._load_session")
    def test_add_cue_clamps_priority(self, mock_load, mock_save):
        from opencut.core.adr_cue_system import ADRSession, add_cue
        session = ADRSession(session_id="s1", project_name="Test")
        mock_load.return_value = session

        cue = add_cue("s1", "X", "Y", 0, 1, priority=99)
        self.assertEqual(cue.priority, 5)

    @patch("opencut.core.adr_cue_system._save_session")
    @patch("opencut.core.adr_cue_system._load_session")
    def test_add_cue_invalid_reason_defaults(self, mock_load, mock_save):
        from opencut.core.adr_cue_system import ADRSession, add_cue
        session = ADRSession(session_id="s1", project_name="Test")
        mock_load.return_value = session

        cue = add_cue("s1", "X", "Y", 0, 1, reason="nonexistent_reason")
        self.assertEqual(cue.reason, "other")

    @patch("opencut.core.adr_cue_system._save_session")
    @patch("opencut.core.adr_cue_system._load_session")
    def test_update_cue(self, mock_load, mock_save):
        from opencut.core.adr_cue_system import ADRCue, ADRSession, update_cue
        session = ADRSession(session_id="s1", project_name="Test")
        session.cues.append(ADRCue(cue_id="c1", character_name="Old"))
        mock_load.return_value = session

        cue = update_cue("s1", "c1", character_name="New")
        self.assertEqual(cue.character_name, "New")

    @patch("opencut.core.adr_cue_system._save_session")
    @patch("opencut.core.adr_cue_system._load_session")
    def test_update_cue_not_found(self, mock_load, mock_save):
        from opencut.core.adr_cue_system import ADRSession, update_cue
        session = ADRSession(session_id="s1")
        mock_load.return_value = session
        with self.assertRaises(ValueError):
            update_cue("s1", "no_such_cue", character_name="X")

    @patch("opencut.core.adr_cue_system._save_session")
    @patch("opencut.core.adr_cue_system._load_session")
    def test_remove_cue(self, mock_load, mock_save):
        from opencut.core.adr_cue_system import ADRCue, ADRSession, remove_cue
        session = ADRSession(session_id="s1")
        session.cues.append(ADRCue(cue_id="c1"))
        session.cues.append(ADRCue(cue_id="c2"))
        mock_load.return_value = session

        result = remove_cue("s1", "c1")
        self.assertTrue(result)
        self.assertEqual(len(session.cues), 1)

    @patch("opencut.core.adr_cue_system._save_session")
    @patch("opencut.core.adr_cue_system._load_session")
    def test_remove_cue_not_found(self, mock_load, mock_save):
        from opencut.core.adr_cue_system import ADRSession, remove_cue
        session = ADRSession(session_id="s1")
        mock_load.return_value = session

        result = remove_cue("s1", "no_such")
        self.assertFalse(result)

    @patch("opencut.core.adr_cue_system._load_session")
    def test_list_cues_sorted_by_timecode(self, mock_load):
        from opencut.core.adr_cue_system import ADRCue, ADRSession, list_cues
        session = ADRSession(session_id="s1")
        session.cues = [
            ADRCue(cue_id="c1", timecode_in=5.0, timecode_out=6.0),
            ADRCue(cue_id="c2", timecode_in=1.0, timecode_out=2.0),
        ]
        mock_load.return_value = session

        cues = list_cues("s1", sort_by="timecode")
        self.assertEqual(cues[0]["cue_id"], "c2")
        self.assertEqual(cues[1]["cue_id"], "c1")

    @patch("opencut.core.adr_cue_system._load_session")
    def test_list_cues_filter_by_status(self, mock_load):
        from opencut.core.adr_cue_system import ADRCue, ADRSession, list_cues
        session = ADRSession(session_id="s1")
        session.cues = [
            ADRCue(cue_id="c1", status="pending"),
            ADRCue(cue_id="c2", status="approved"),
        ]
        mock_load.return_value = session

        cues = list_cues("s1", status_filter="approved")
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0]["cue_id"], "c2")

    @patch("opencut.core.adr_cue_system._load_session")
    def test_list_cues_filter_by_character(self, mock_load):
        from opencut.core.adr_cue_system import ADRCue, ADRSession, list_cues
        session = ADRSession(session_id="s1")
        session.cues = [
            ADRCue(cue_id="c1", character_name="Alice"),
            ADRCue(cue_id="c2", character_name="Bob"),
        ]
        mock_load.return_value = session

        cues = list_cues("s1", character_filter="alice")
        self.assertEqual(len(cues), 1)

    def test_export_cue_sheet_csv(self):
        from opencut.core.adr_cue_system import ADRCue, ADRSession
        with patch("opencut.core.adr_cue_system._load_session") as mock_load:
            session = ADRSession(session_id="s1", project_name="Test",
                                 reel="R2", fps=24.0)
            session.cues = [
                ADRCue(cue_id="c1", character_name="Alice",
                       original_line="Hello", timecode_in=1.0, timecode_out=3.0,
                       reason="noise"),
            ]
            mock_load.return_value = session

            from opencut.core.adr_cue_system import export_cue_sheet_csv
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
                out_path = f.name
            try:
                result = export_cue_sheet_csv("s1", out_path)
                self.assertEqual(result, out_path)
                with open(out_path, encoding="utf-8") as f:
                    content = f.read()
                self.assertIn("Cue#", content)
                self.assertIn("Character", content)
                self.assertIn("Alice", content)
                self.assertIn("R2", content)
            finally:
                os.unlink(out_path)

    def test_export_cue_sheet_json(self):
        from opencut.core.adr_cue_system import ADRCue, ADRSession
        with patch("opencut.core.adr_cue_system._load_session") as mock_load:
            session = ADRSession(session_id="s1", project_name="Test", fps=24.0)
            session.cues = [
                ADRCue(cue_id="c1", character_name="Bob",
                       original_line="World", timecode_in=5.0, timecode_out=8.0,
                       reason="script_change", priority=1, status="pending"),
            ]
            mock_load.return_value = session

            from opencut.core.adr_cue_system import export_cue_sheet_json
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                out_path = f.name
            try:
                result = export_cue_sheet_json("s1", out_path)
                self.assertEqual(result, out_path)
                with open(out_path, encoding="utf-8") as f:
                    data = json.load(f)
                self.assertEqual(data["project_name"], "Test")
                self.assertEqual(data["total_cues"], 1)
                self.assertIn("cues", data)
                self.assertEqual(data["cues"][0]["character"], "Bob")
                self.assertIn("tc_in", data["cues"][0])
                self.assertIn("tc_out", data["cues"][0])
                self.assertIn("status_summary", data)
            finally:
                os.unlink(out_path)

    @patch("opencut.core.adr_cue_system.run_ffmpeg")
    @patch("opencut.core.adr_cue_system.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    @patch("opencut.core.adr_cue_system._load_session")
    def test_extract_guide_audio(self, mock_load, mock_ffmpeg, mock_run):
        from opencut.core.adr_cue_system import ADRCue, ADRSession, extract_guide_audio
        session = ADRSession(session_id="s1", source_path="/fake/video.mp4")
        session.cues = [ADRCue(cue_id="c1", timecode_in=10.0, timecode_out=15.0)]
        mock_load.return_value = session

        with patch("os.path.isfile", return_value=True):
            result = extract_guide_audio("s1", "c1", preroll=2.0, postroll=1.0,
                                         output_dir=tempfile.gettempdir())
        self.assertEqual(result["cue_id"], "c1")
        self.assertAlmostEqual(result["start"], 8.0)
        self.assertAlmostEqual(result["end"], 16.0)
        self.assertAlmostEqual(result["duration"], 8.0)
        mock_run.assert_called_once()

    @patch("opencut.core.adr_cue_system.run_ffmpeg")
    @patch("opencut.core.adr_cue_system.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    @patch("opencut.core.adr_cue_system._load_session")
    def test_extract_guide_no_source(self, mock_load, mock_ffmpeg, mock_run):
        from opencut.core.adr_cue_system import ADRCue, ADRSession, extract_guide_audio
        session = ADRSession(session_id="s1", source_path="")
        session.cues = [ADRCue(cue_id="c1")]
        mock_load.return_value = session

        with self.assertRaises(ValueError):
            extract_guide_audio("s1", "c1")

    def test_create_session(self):
        from opencut.core.adr_cue_system import create_session
        with patch("opencut.core.adr_cue_system._save_session"):
            session = create_session("My Project", reel="R3", fps=30.0)
        self.assertEqual(session.project_name, "My Project")
        self.assertEqual(session.reel, "R3")
        self.assertAlmostEqual(session.fps, 30.0)
        self.assertTrue(len(session.session_id) > 0)

    def test_adr_constants(self):
        from opencut.core.adr_cue_system import ADR_PRIORITIES, ADR_REASONS, ADR_STATUSES, CSV_COLUMNS
        self.assertIn("noise", ADR_REASONS)
        self.assertIn("script_change", ADR_REASONS)
        self.assertIn("performance", ADR_REASONS)
        self.assertIn(1, ADR_PRIORITIES)
        self.assertIn(5, ADR_PRIORITIES)
        self.assertIn("pending", ADR_STATUSES)
        self.assertIn("approved", ADR_STATUSES)
        self.assertEqual(CSV_COLUMNS[0], "Cue#")
        self.assertIn("Character", CSV_COLUMNS)
        self.assertIn("TC In", CSV_COLUMNS)


# ============================================================
# M&E Mix Tests
# ============================================================
class TestMEMix(unittest.TestCase):
    """Tests for opencut.core.me_mix module."""

    def test_me_mix_result_fields(self):
        from opencut.core.me_mix import MEMixResult
        f_names = {f.name for f in fields(MEMixResult)}
        self.assertIn("output_path", f_names)
        self.assertIn("method_used", f_names)
        self.assertIn("residual_dialogue_score", f_names)
        self.assertIn("duration", f_names)
        self.assertIn("loudness_lufs", f_names)

    def test_me_mix_result_to_dict(self):
        from opencut.core.me_mix import MEMixResult
        r = MEMixResult(output_path="/out.wav", method_used="spectral",
                        residual_dialogue_score=0.3, duration=120.0)
        d = r.to_dict()
        self.assertEqual(d["method_used"], "spectral")
        self.assertAlmostEqual(d["residual_dialogue_score"], 0.3)

    def test_list_methods(self):
        from opencut.core.me_mix import list_methods
        methods = list_methods()
        ids = {m["id"] for m in methods}
        self.assertIn("auto", ids)
        self.assertIn("stem_separation", ids)
        self.assertIn("track_mute", ids)
        self.assertIn("spectral", ids)

    @patch("opencut.core.me_mix._measure_residual_dialogue", return_value=0.2)
    @patch("opencut.core.me_mix._get_duration", return_value=60.0)
    @patch("opencut.core.me_mix._level_match")
    @patch("opencut.core.me_mix.run_ffmpeg")
    @patch("opencut.core.me_mix.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_generate_me_spectral(self, mock_ffp, mock_run, mock_lvl, mock_dur, mock_res):
        from opencut.core.me_mix import generate_me_mix
        result = generate_me_mix(
            "/fake/input.wav", method="spectral", output_format="wav",
        )
        self.assertEqual(result.method_used, "spectral")
        self.assertAlmostEqual(result.residual_dialogue_score, 0.2)
        self.assertAlmostEqual(result.duration, 60.0)
        mock_run.assert_called_once()

    @patch("opencut.core.me_mix._measure_residual_dialogue", return_value=0.1)
    @patch("opencut.core.me_mix._get_duration", return_value=30.0)
    @patch("opencut.core.me_mix._level_match")
    @patch("opencut.core.me_mix._count_audio_streams", return_value=4)
    @patch("opencut.core.me_mix.run_ffmpeg")
    @patch("opencut.core.me_mix.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_generate_me_track_mute(self, mock_ffp, mock_run, mock_count,
                                     mock_lvl, mock_dur, mock_res):
        from opencut.core.me_mix import generate_me_mix
        result = generate_me_mix(
            "/fake/input.wav", method="track_mute",
            dialogue_tracks=[0, 2],
        )
        self.assertEqual(result.method_used, "track_mute")
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(str(c) for c in cmd)
        self.assertIn("amerge", cmd_str)

    @patch("opencut.core.me_mix._count_audio_streams", return_value=2)
    @patch("opencut.core.me_mix.run_ffmpeg")
    @patch("opencut.core.me_mix.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_track_mute_all_muted_raises(self, mock_ffp, mock_run, mock_count):
        from opencut.core.me_mix import _me_via_track_mute
        with self.assertRaises(ValueError):
            _me_via_track_mute("/fake.wav", "/out.wav", [0, 1], "wav", None)

    @patch("opencut.core.me_mix.subprocess.run")
    def test_measure_band_energy(self, mock_sub):
        from opencut.core.me_mix import _measure_band_energy
        mock_sub.return_value = MagicMock(
            stderr="mean_volume: -20.0 dB\n",
            returncode=0,
        )
        energy = _measure_band_energy("/fake.wav", 300, 4000)
        self.assertGreater(energy, 0)

    def test_me_mix_supported_formats(self):
        from opencut.core.me_mix import SUPPORTED_FORMATS
        self.assertIn("wav", SUPPORTED_FORMATS)
        self.assertIn("mp3", SUPPORTED_FORMATS)

    @patch("opencut.core.me_mix.run_ffmpeg")
    @patch("opencut.core.me_mix.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_level_match_calls_loudnorm(self, mock_ffp, mock_run):
        from opencut.core.me_mix import _level_match
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
            f.write(b"fake")
        try:
            # Mock run_ffmpeg and os.replace
            with patch("os.replace"):
                _level_match(tmp, -23.0)
            cmd = mock_run.call_args[0][0]
            cmd_str = " ".join(str(c) for c in cmd)
            self.assertIn("loudnorm", cmd_str)
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    @patch("shutil.which", return_value=None)
    @patch("opencut.core.me_mix.subprocess.run")
    def test_stem_separation_unavailable_returns_none(self, mock_sub, mock_which):
        from opencut.core.me_mix import _me_via_stem_separation
        mock_sub.return_value = MagicMock(returncode=1)
        result = _me_via_stem_separation("/fake.wav", "/out.wav", "wav")
        self.assertIsNone(result)


# ============================================================
# Dialogue Premix Tests
# ============================================================
class TestDialoguePremix(unittest.TestCase):
    """Tests for opencut.core.dialogue_premix module."""

    def test_premix_result_fields(self):
        from opencut.core.dialogue_premix import PremixResult
        f_names = {f.name for f in fields(PremixResult)}
        self.assertIn("output_path", f_names)
        self.assertIn("speakers_processed", f_names)
        self.assertIn("per_speaker_stats", f_names)
        self.assertIn("target_lufs", f_names)

    def test_speaker_stats_fields(self):
        from opencut.core.dialogue_premix import SpeakerStats
        f_names = {f.name for f in fields(SpeakerStats)}
        self.assertIn("speaker_id", f_names)
        self.assertIn("sibilance_detected", f_names)
        self.assertIn("deess_applied", f_names)
        self.assertIn("eq_preset", f_names)
        self.assertIn("dynamic_range_db", f_names)

    def test_eq_presets_exist(self):
        from opencut.core.dialogue_premix import EQ_PRESETS
        self.assertIn("interview", EQ_PRESETS)
        self.assertIn("podcast", EQ_PRESETS)
        self.assertIn("broadcast", EQ_PRESETS)
        self.assertIn("film", EQ_PRESETS)
        self.assertIn("voiceover", EQ_PRESETS)

    def test_eq_presets_have_required_keys(self):
        from opencut.core.dialogue_premix import EQ_PRESETS
        for name, preset in EQ_PRESETS.items():
            self.assertIn("label", preset, f"Missing label in {name}")
            self.assertIn("description", preset, f"Missing description in {name}")
            self.assertIn("filters", preset, f"Missing filters in {name}")
            self.assertIn("target_lufs", preset, f"Missing target_lufs in {name}")
            self.assertIsInstance(preset["filters"], list)

    def test_compressor_presets_exist(self):
        from opencut.core.dialogue_premix import COMPRESSOR_PRESETS
        for ct in ("interview", "podcast", "broadcast", "film", "voiceover"):
            self.assertIn(ct, COMPRESSOR_PRESETS)
            self.assertIn("acompressor", COMPRESSOR_PRESETS[ct])

    def test_deess_strengths(self):
        from opencut.core.dialogue_premix import DEESS_FREQS
        self.assertIn("gentle", DEESS_FREQS)
        self.assertIn("moderate", DEESS_FREQS)
        self.assertIn("aggressive", DEESS_FREQS)
        for strength, params in DEESS_FREQS.items():
            self.assertIn("freq", params)
            self.assertIn("gain", params)
            self.assertLess(params["gain"], 0)

    def test_list_presets(self):
        from opencut.core.dialogue_premix import list_presets
        presets = list_presets()
        self.assertGreaterEqual(len(presets), 5)
        ids = {p["id"] for p in presets}
        self.assertIn("podcast", ids)
        self.assertIn("interview", ids)

    def test_list_deess_strengths(self):
        from opencut.core.dialogue_premix import list_deess_strengths
        strengths = list_deess_strengths()
        self.assertEqual(len(strengths), 3)

    def test_build_processing_chain(self):
        from opencut.core.dialogue_premix import _build_processing_chain
        chain, desc = _build_processing_chain("podcast", "moderate", {"sibilance": True})
        self.assertTrue(len(chain) >= 4)
        self.assertTrue(any("equalizer" in f for f in chain))
        self.assertTrue(any("acompressor" in f for f in chain))
        self.assertTrue(any("dynaudnorm" in f for f in chain))

    def test_build_processing_chain_no_deess(self):
        from opencut.core.dialogue_premix import _build_processing_chain
        chain, desc = _build_processing_chain("broadcast", "moderate", {"sibilance": False})
        # No de-ess filter when sibilance not detected
        deess_filters = [f for f in chain if "7000" in f or "6000" in f or "5500" in f]
        self.assertEqual(len(deess_filters), 0)

    @patch("opencut.core.dialogue_premix._analyze_audio", return_value={
        "lufs": -20, "sibilance": True, "dynamic_range": 15, "duration": 60, "peak_db": -1})
    @patch("opencut.core.dialogue_premix.run_ffmpeg")
    @patch("opencut.core.dialogue_premix.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_premix_single_speaker(self, mock_ffp, mock_run, mock_analyze):
        from opencut.core.dialogue_premix import premix_dialogue
        result = premix_dialogue(
            "/fake/input.wav", content_type="podcast", target_lufs=-16.0,
        )
        self.assertEqual(result.speakers_processed, 1)
        self.assertEqual(result.content_type, "podcast")
        self.assertAlmostEqual(result.target_lufs, -16.0)
        self.assertEqual(len(result.per_speaker_stats), 1)
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(str(c) for c in cmd)
        self.assertIn("loudnorm", cmd_str)

    @patch("opencut.core.dialogue_premix._analyze_audio", return_value={
        "lufs": -22, "sibilance": False, "dynamic_range": 10, "duration": 30, "peak_db": -2})
    @patch("opencut.core.dialogue_premix.run_ffmpeg")
    @patch("opencut.core.dialogue_premix.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_premix_interview_preset(self, mock_ffp, mock_run, mock_analyze):
        from opencut.core.dialogue_premix import premix_dialogue
        result = premix_dialogue(
            "/fake/input.wav", content_type="interview",
        )
        self.assertEqual(result.content_type, "interview")
        cmd_str = " ".join(str(c) for c in mock_run.call_args[0][0])
        self.assertIn("highpass=f=80", cmd_str)

    @patch("opencut.core.dialogue_premix.run_ffmpeg")
    @patch("opencut.core.dialogue_premix.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_premix_simple_legacy(self, mock_ffp, mock_run):
        from opencut.core.dialogue_premix import premix_simple
        result = premix_simple("/fake/input.wav", target_lufs=-23.0)
        self.assertIn("output_path", result)
        self.assertIn("processing_chain", result)
        self.assertEqual(result["target_lufs"], -23.0)

    def test_premix_result_to_dict(self):
        from opencut.core.dialogue_premix import PremixResult, SpeakerStats
        stats = SpeakerStats(speaker_id="s1", speaker_label="Alice")
        r = PremixResult(output_path="/out.wav", speakers_processed=1,
                         per_speaker_stats=[stats])
        d = r.to_dict()
        self.assertEqual(d["speakers_processed"], 1)
        self.assertEqual(len(d["per_speaker_stats"]), 1)

    def test_premix_clamps_target_lufs(self):
        from opencut.core.dialogue_premix import _premix_single_speaker
        with patch("opencut.core.dialogue_premix._analyze_audio", return_value={
            "lufs": -20, "sibilance": False, "dynamic_range": 10, "duration": 10, "peak_db": 0}):
            with patch("opencut.core.dialogue_premix.run_ffmpeg"):
                with patch("opencut.core.dialogue_premix.get_ffmpeg_path", return_value="ffmpeg"):
                    # Won't crash with extreme LUFS
                    result = _premix_single_speaker(
                        "/fake.wav", "/out.wav", "podcast", -50.0, "moderate",
                    )
                    # Internally the function should clamp (we check the pipeline ran)
                    self.assertEqual(result.speakers_processed, 1)


# ============================================================
# Surround Upmix Tests
# ============================================================
class TestSurroundUpmix(unittest.TestCase):
    """Tests for opencut.core.surround_upmix module."""

    def test_surround_result_fields(self):
        from opencut.core.surround_upmix import SurroundResult
        f_names = {f.name for f in fields(SurroundResult)}
        self.assertIn("output_path", f_names)
        self.assertIn("layout", f_names)
        self.assertIn("channels", f_names)
        self.assertIn("duration", f_names)
        self.assertIn("downmix_correlation", f_names)

    def test_surround_layouts(self):
        from opencut.core.surround_upmix import SURROUND_LAYOUTS
        self.assertIn("5.1", SURROUND_LAYOUTS)
        self.assertIn("7.1", SURROUND_LAYOUTS)
        self.assertEqual(SURROUND_LAYOUTS["5.1"]["channels"], 6)
        self.assertEqual(SURROUND_LAYOUTS["7.1"]["channels"], 8)

    def test_surround_51_channel_names(self):
        from opencut.core.surround_upmix import SURROUND_LAYOUTS
        names = SURROUND_LAYOUTS["5.1"]["channel_names"]
        self.assertEqual(names, ["FL", "FR", "FC", "LFE", "BL", "BR"])

    def test_surround_71_channel_names(self):
        from opencut.core.surround_upmix import SURROUND_LAYOUTS
        names = SURROUND_LAYOUTS["7.1"]["channel_names"]
        self.assertEqual(names, ["FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR"])

    def test_upmix_modes(self):
        from opencut.core.surround_upmix import UPMIX_MODES
        self.assertIn("simple_5_1", UPMIX_MODES)
        self.assertIn("music_5_1", UPMIX_MODES)
        self.assertIn("dialogue_5_1", UPMIX_MODES)
        self.assertIn("simple_7_1", UPMIX_MODES)

    def test_export_formats(self):
        from opencut.core.surround_upmix import EXPORT_FORMATS
        self.assertIn("wav", EXPORT_FORMATS)
        self.assertIn("ac3", EXPORT_FORMATS)
        self.assertIn("eac3", EXPORT_FORMATS)

    def test_list_layouts(self):
        from opencut.core.surround_upmix import list_layouts
        layouts = list_layouts()
        ids = {lay["id"] for lay in layouts}
        self.assertIn("5.1", ids)
        self.assertIn("7.1", ids)

    def test_list_upmix_modes(self):
        from opencut.core.surround_upmix import list_upmix_modes
        modes = list_upmix_modes()
        self.assertGreaterEqual(len(modes), 4)

    def test_list_export_formats(self):
        from opencut.core.surround_upmix import list_export_formats
        fmts = list_export_formats()
        ids = {f["id"] for f in fmts}
        self.assertIn("wav", ids)
        self.assertIn("ac3", ids)

    def test_build_upmix_filter_simple_51(self):
        from opencut.core.surround_upmix import _build_upmix_filter
        f = _build_upmix_filter("simple_5_1", "5.1")
        self.assertIn("pan=5.1", f)
        self.assertIn("FL=", f)
        self.assertIn("FC=", f)
        self.assertIn("LFE=", f)
        self.assertIn("BL=", f)
        self.assertIn("BR=", f)

    def test_build_upmix_filter_music_51(self):
        from opencut.core.surround_upmix import _build_upmix_filter
        f = _build_upmix_filter("music_5_1", "5.1")
        self.assertIn("pan=5.1", f)
        # Music mode should have wider front stereo
        self.assertIn("FL=0.8", f)

    def test_build_upmix_filter_dialogue_51(self):
        from opencut.core.surround_upmix import _build_upmix_filter
        f = _build_upmix_filter("dialogue_5_1", "5.1")
        self.assertIn("pan=5.1", f)
        # Dialogue mode should have heavier center
        self.assertIn("FC=0.7", f)

    def test_build_upmix_filter_71(self):
        from opencut.core.surround_upmix import _build_upmix_filter
        f = _build_upmix_filter("simple_7_1", "7.1")
        self.assertIn("pan=7.1", f)
        self.assertIn("SL=", f)
        self.assertIn("SR=", f)

    def test_pan_coefficients_front_center(self):
        from opencut.core.surround_upmix import _calculate_pan_coefficients
        gains = _calculate_pan_coefficients(0, 1.0, "5.1")
        # At 0 degrees (front center), FC should be dominant
        self.assertIn("FC", gains)
        self.assertIn("FL", gains)
        self.assertIn("FR", gains)
        self.assertIn("LFE", gains)
        self.assertIn("BL", gains)
        self.assertIn("BR", gains)
        # All gains should be non-negative
        for ch, g in gains.items():
            self.assertGreaterEqual(g, 0.0, f"{ch} has negative gain")

    def test_pan_coefficients_71(self):
        from opencut.core.surround_upmix import _calculate_pan_coefficients
        gains = _calculate_pan_coefficients(90, 1.0, "7.1")
        self.assertIn("SL", gains)
        self.assertIn("SR", gains)
        # At 90 degrees, SR should be strong
        self.assertGreater(gains["SR"], 0.0)

    def test_pan_coefficients_normalized(self):
        from opencut.core.surround_upmix import _calculate_pan_coefficients
        gains = _calculate_pan_coefficients(45, 0.8, "5.1")
        non_lfe = sum(v for k, v in gains.items() if k != "LFE")
        # Non-LFE gains should sum to approximately 1.0
        self.assertAlmostEqual(non_lfe, 1.0, places=2)

    def test_pan_coefficients_center_distance(self):
        from opencut.core.surround_upmix import _calculate_pan_coefficients
        gains = _calculate_pan_coefficients(0, 0.0, "5.1")
        # At distance 0, all non-LFE gains should be roughly equal
        non_lfe = {k: v for k, v in gains.items() if k != "LFE"}
        values = list(non_lfe.values())
        if values:
            spread = max(values) - min(values)
            self.assertLess(spread, 0.3, "Gains should be roughly equal at center")

    def test_build_pan_filter_51(self):
        from opencut.core.surround_upmix import _build_pan_filter
        gains = {"FL": 0.5, "FR": 0.3, "FC": 0.1, "LFE": 0.05, "BL": 0.05, "BR": 0.0}
        f = _build_pan_filter(gains, "5.1")
        self.assertIn("pan=5.1", f)
        self.assertIn("FL=0.5000", f)
        self.assertIn("FR=0.3000", f)

    @patch("opencut.core.surround_upmix._measure_channel_levels", return_value={})
    @patch("opencut.core.surround_upmix._measure_downmix_correlation", return_value=0.95)
    @patch("opencut.core.surround_upmix._get_duration", return_value=120.0)
    @patch("opencut.core.surround_upmix.run_ffmpeg")
    @patch("opencut.core.surround_upmix.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_upmix_simple_51(self, mock_ffp, mock_run, mock_dur, mock_corr, mock_lvl):
        from opencut.core.surround_upmix import upmix_surround
        result = upmix_surround("/fake/input.wav", mode="simple_5_1")
        self.assertEqual(result.layout, "5.1")
        self.assertEqual(result.channels, 6)
        self.assertEqual(result.mode, "simple_5_1")
        self.assertAlmostEqual(result.downmix_correlation, 0.95)
        mock_run.assert_called_once()
        cmd_str = " ".join(str(c) for c in mock_run.call_args[0][0])
        self.assertIn("pan=5.1", cmd_str)

    @patch("opencut.core.surround_upmix._get_duration", return_value=60.0)
    @patch("opencut.core.surround_upmix.run_ffmpeg")
    @patch("opencut.core.surround_upmix.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_pan_surround(self, mock_ffp, mock_run, mock_dur):
        from opencut.core.surround_upmix import pan_surround
        result = pan_surround("/fake/input.wav", angle=90.0, distance=0.8)
        self.assertEqual(result.layout, "5.1")
        self.assertIn("90", result.mode)
        mock_run.assert_called_once()

    @patch("opencut.core.surround_upmix._measure_rms_level", return_value=-20.0)
    @patch("opencut.core.surround_upmix.run_ffmpeg")
    @patch("opencut.core.surround_upmix.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_validate_downmix(self, mock_ffp, mock_run, mock_rms):
        from opencut.core.surround_upmix import validate_downmix
        result = validate_downmix("/fake/surround.wav", "/fake/original.wav",
                                   output_path="/fake/downmix.wav")
        self.assertIn("correlation", result)
        self.assertIn("phase_issues", result)
        self.assertIn("level_difference_db", result)
        self.assertFalse(result["phase_issues"])  # Same levels -> no issues

    @patch("opencut.core.surround_upmix._get_duration", return_value=60.0)
    @patch("opencut.core.surround_upmix.run_ffmpeg")
    @patch("opencut.core.surround_upmix.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_export_surround_ac3(self, mock_ffp, mock_run, mock_dur):
        from opencut.core.surround_upmix import export_surround
        result = export_surround("/fake/surround.wav", export_format="ac3")
        self.assertEqual(result["format"], "ac3")
        self.assertEqual(result["codec"], "ac3")
        cmd_str = " ".join(str(c) for c in mock_run.call_args[0][0])
        self.assertIn("ac3", cmd_str)

    @patch("opencut.core.surround_upmix.run_ffmpeg")
    @patch("opencut.core.surround_upmix.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_extract_lfe(self, mock_ffp, mock_run):
        from opencut.core.surround_upmix import extract_lfe
        extract_lfe("/fake/input.wav", cutoff_hz=120.0)
        cmd_str = " ".join(str(c) for c in mock_run.call_args[0][0])
        self.assertIn("lowpass=f=120.0", cmd_str)

    def test_speaker_angles(self):
        from opencut.core.surround_upmix import SPEAKER_ANGLES
        self.assertEqual(SPEAKER_ANGLES["FC"], 0)
        self.assertIn("FL", SPEAKER_ANGLES)
        self.assertIn("FR", SPEAKER_ANGLES)
        self.assertIn("BL", SPEAKER_ANGLES)
        self.assertIn("BR", SPEAKER_ANGLES)


# ============================================================
# Foley Cueing Tests
# ============================================================
class TestFoleyCue(unittest.TestCase):
    """Tests for opencut.core.foley_cue module."""

    def test_foley_cue_fields(self):
        from opencut.core.foley_cue import FoleyCue
        f_names = {f.name for f in fields(FoleyCue)}
        self.assertIn("cue_id", f_names)
        self.assertIn("event_type", f_names)
        self.assertIn("timecode", f_names)
        self.assertIn("duration", f_names)
        self.assertIn("intensity", f_names)
        self.assertIn("suggested_sound", f_names)

    def test_foley_session_fields(self):
        from opencut.core.foley_cue import FoleySession
        f_names = {f.name for f in fields(FoleySession)}
        self.assertIn("session_id", f_names)
        self.assertIn("source_path", f_names)
        self.assertIn("cues", f_names)
        self.assertIn("categories_detected", f_names)

    def test_foley_categories_count(self):
        from opencut.core.foley_cue import FOLEY_CATEGORIES
        self.assertEqual(len(FOLEY_CATEGORIES), 8)

    def test_foley_categories_keys(self):
        from opencut.core.foley_cue import FOLEY_CATEGORIES
        expected = {"footstep", "door", "impact", "cloth", "glass", "water",
                    "mechanical", "ambient"}
        self.assertEqual(set(FOLEY_CATEGORIES.keys()), expected)

    def test_foley_categories_have_required_fields(self):
        from opencut.core.foley_cue import FOLEY_CATEGORIES
        for cat_id, cat in FOLEY_CATEGORIES.items():
            self.assertIn("label", cat, f"Missing label in {cat_id}")
            self.assertIn("description", cat, f"Missing description in {cat_id}")
            self.assertIn("keywords", cat, f"Missing keywords in {cat_id}")
            self.assertIn("freq_range", cat, f"Missing freq_range in {cat_id}")
            self.assertIn("min_interval", cat, f"Missing min_interval in {cat_id}")

    def test_list_categories(self):
        from opencut.core.foley_cue import list_categories
        cats = list_categories()
        self.assertEqual(len(cats), 8)
        ids = {c["id"] for c in cats}
        self.assertIn("footstep", ids)
        self.assertIn("door", ids)

    def test_foley_cue_to_dict(self):
        from opencut.core.foley_cue import FoleyCue
        cue = FoleyCue(cue_id="F1", event_type="impact", timecode=5.5,
                       duration=0.2, intensity=0.8)
        d = cue.to_dict()
        self.assertEqual(d["event_type"], "impact")
        self.assertAlmostEqual(d["timecode"], 5.5)

    def test_foley_session_to_dict(self):
        from opencut.core.foley_cue import FoleyCue, FoleySession
        session = FoleySession(session_id="fs1", source_path="/fake.mp4")
        session.cues = [FoleyCue(event_type="footstep"), FoleyCue(event_type="door")]
        session.categories_detected = ["footstep", "door"]
        d = session.to_dict()
        self.assertEqual(d["cue_count"], 2)
        self.assertEqual(len(d["categories_detected"]), 2)

    def test_classify_events_scene_changes(self):
        from opencut.core.foley_cue import _classify_events
        scene_changes = [{"time": 5.0, "score": 0.9}]
        cues = _classify_events(scene_changes, [], [], ["door", "impact"], 0.5)
        event_types = {c.event_type for c in cues}
        self.assertTrue(event_types.intersection({"door", "impact"}))

    def test_classify_events_empty(self):
        from opencut.core.foley_cue import FOLEY_CATEGORIES, _classify_events
        cues = _classify_events([], [], [], list(FOLEY_CATEGORIES.keys()), 0.5)
        self.assertEqual(len(cues), 0)

    def test_deduplicate_cues(self):
        from opencut.core.foley_cue import FoleyCue, _deduplicate_cues
        cues = [
            FoleyCue(event_type="impact", timecode=5.0, confidence=0.8),
            FoleyCue(event_type="impact", timecode=5.1, confidence=0.6),  # Too close
            FoleyCue(event_type="impact", timecode=10.0, confidence=0.7),
        ]
        result = _deduplicate_cues(cues, ["impact"])
        self.assertEqual(len(result), 2)

    def test_suggest_sound(self):
        from opencut.core.foley_cue import _suggest_sound
        self.assertEqual(_suggest_sound("footstep"), "hard_surface_step")
        self.assertEqual(_suggest_sound("door"), "wooden_door_close")
        self.assertEqual(_suggest_sound("impact"), "body_impact_medium")
        self.assertEqual(_suggest_sound("glass"), "glass_clink")

    def test_export_cue_sheet_csv(self):
        from opencut.core.foley_cue import FoleyCue, FoleySession, export_cue_sheet
        session = FoleySession(session_id="fs1")
        session.cues = [
            FoleyCue(cue_id="F01", event_type="footstep", timecode=1.0,
                     duration=0.1, intensity=0.5, suggested_sound="step"),
        ]

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            out_path = f.name
        try:
            export_cue_sheet(session, out_path, "csv")
            with open(out_path, encoding="utf-8") as f:
                content = f.read()
            self.assertIn("Cue#", content)
            self.assertIn("footstep", content)
        finally:
            os.unlink(out_path)

    def test_export_cue_sheet_json(self):
        from opencut.core.foley_cue import FoleyCue, FoleySession, export_cue_sheet
        session = FoleySession(session_id="fs1")
        session.cues = [
            FoleyCue(cue_id="F01", event_type="door", timecode=2.0),
        ]

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out_path = f.name
        try:
            export_cue_sheet(session, out_path, "json")
            with open(out_path, encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["cue_count"], 1)
            self.assertEqual(data["cues"][0]["event_type"], "door")
        finally:
            os.unlink(out_path)

    def test_catalog_sfx_library(self):
        from opencut.core.foley_cue import _catalog_sfx_library
        with tempfile.TemporaryDirectory() as d:
            # Create test files
            os.makedirs(os.path.join(d, "footsteps"))
            open(os.path.join(d, "footsteps", "step_concrete.wav"), "w").close()
            open(os.path.join(d, "door_slam.wav"), "w").close()
            open(os.path.join(d, "impact_heavy.mp3"), "w").close()
            open(os.path.join(d, "not_audio.txt"), "w").close()

            catalog = _catalog_sfx_library(d)
            self.assertGreater(len(catalog["footstep"]), 0)
            self.assertGreater(len(catalog["door"]), 0)
            self.assertGreater(len(catalog["impact"]), 0)

    def test_match_cues_to_sfx(self):
        from opencut.core.foley_cue import FoleyCue, _match_cues_to_sfx
        cues = [
            FoleyCue(cue_id="F01", event_type="footstep"),
            FoleyCue(cue_id="F02", event_type="glass"),  # No match in catalog
        ]
        catalog = {
            "footstep": ["/sfx/step.wav"],
            "door": [],
            "impact": [],
            "cloth": [],
            "glass": [],
            "water": [],
            "mechanical": [],
            "ambient": [],
        }
        matches = _match_cues_to_sfx(cues, catalog)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["category"], "footstep")

    @patch("opencut.core.foley_cue.get_video_info", return_value={"duration": 120})
    @patch("opencut.core.foley_cue._detect_audio_transients", return_value=[])
    @patch("opencut.core.foley_cue._detect_motion_peaks", return_value=[])
    @patch("opencut.core.foley_cue._detect_scene_changes", return_value=[
        {"time": 5.0, "score": 0.9},
        {"time": 15.0, "score": 0.8},
    ])
    def test_detect_foley_cues_integration(self, mock_sc, mock_mp, mock_at, mock_vi):
        from opencut.core.foley_cue import detect_foley_cues
        session = detect_foley_cues("/fake/video.mp4", sensitivity=0.8)
        self.assertGreater(len(session.cues), 0)
        self.assertEqual(session.source_path, "/fake/video.mp4")
        self.assertAlmostEqual(session.total_duration, 120)

    @patch("opencut.core.foley_cue.run_ffmpeg")
    @patch("opencut.core.foley_cue.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_place_sfx_no_matches(self, mock_ffp, mock_run):
        from opencut.core.foley_cue import FoleySession, place_sfx
        session = FoleySession(session_id="fs1", total_duration=60)
        with tempfile.TemporaryDirectory() as d:
            result = place_sfx(session, d, "/fake/audio.wav",
                               output_path="/fake/out.wav")
        self.assertEqual(result["sfx_placed"], 0)

    def test_place_sfx_bad_library_dir(self):
        from opencut.core.foley_cue import FoleySession, place_sfx
        session = FoleySession(session_id="fs1")
        with self.assertRaises(ValueError):
            place_sfx(session, "/nonexistent/dir", "/fake/audio.wav")


# ============================================================
# Audio Post Routes Tests
# ============================================================
class TestAudioPostRoutes(unittest.TestCase):
    """Smoke tests for audio_post_routes blueprint."""

    def test_blueprint_exists(self):
        from opencut.routes.audio_post_routes import audio_post_bp
        self.assertEqual(audio_post_bp.name, "audio_post")

    def test_blueprint_has_routes(self):
        from opencut.routes.audio_post_routes import audio_post_bp
        rules = list(audio_post_bp.deferred_functions)
        self.assertGreater(len(rules), 0)

    def test_route_functions_exist(self):
        from opencut.routes import audio_post_routes as mod
        self.assertTrue(hasattr(mod, "route_adr_create"))
        self.assertTrue(hasattr(mod, "route_adr_cue"))
        self.assertTrue(hasattr(mod, "route_adr_cues"))
        self.assertTrue(hasattr(mod, "route_adr_cuesheet"))
        self.assertTrue(hasattr(mod, "route_adr_record"))
        self.assertTrue(hasattr(mod, "route_me_mix"))
        self.assertTrue(hasattr(mod, "route_dialogue_premix"))
        self.assertTrue(hasattr(mod, "route_premix_presets"))
        self.assertTrue(hasattr(mod, "route_surround_upmix"))
        self.assertTrue(hasattr(mod, "route_surround_layouts"))
        self.assertTrue(hasattr(mod, "route_foley_analyze"))
        self.assertTrue(hasattr(mod, "route_foley_place"))
        self.assertTrue(hasattr(mod, "route_foley_categories"))

    def test_route_count(self):
        """Verify all 13 routes are registered."""
        from opencut.routes.audio_post_routes import audio_post_bp
        from flask import Flask
        app = Flask(__name__)
        app.register_blueprint(audio_post_bp, url_prefix="/api")
        route_count = sum(1 for rule in app.url_map.iter_rules()
                          if rule.endpoint.startswith("audio_post."))
        self.assertEqual(route_count, 13)

    def test_foley_categories_endpoint(self):
        from opencut.routes.audio_post_routes import audio_post_bp
        from flask import Flask
        app = Flask(__name__)
        app.register_blueprint(audio_post_bp, url_prefix="/api")
        with app.test_client() as client:
            resp = client.get("/api/audio/foley/categories")
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertIn("categories", data)
            self.assertEqual(data["count"], 8)

    def test_premix_presets_endpoint(self):
        from opencut.routes.audio_post_routes import audio_post_bp
        from flask import Flask
        app = Flask(__name__)
        app.register_blueprint(audio_post_bp, url_prefix="/api")
        with app.test_client() as client:
            resp = client.get("/api/audio/premix-presets")
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertIn("presets", data)
            self.assertIn("deess_strengths", data)
            self.assertGreaterEqual(len(data["presets"]), 5)

    def test_surround_layouts_endpoint(self):
        from opencut.routes.audio_post_routes import audio_post_bp
        from flask import Flask
        app = Flask(__name__)
        app.register_blueprint(audio_post_bp, url_prefix="/api")
        with app.test_client() as client:
            resp = client.get("/api/audio/surround-layouts")
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertIn("layouts", data)
            self.assertIn("modes", data)
            self.assertIn("export_formats", data)

    def test_adr_create_no_csrf(self):
        """POST without CSRF token should return 403."""
        from opencut.routes.audio_post_routes import audio_post_bp
        from flask import Flask
        app = Flask(__name__)
        app.register_blueprint(audio_post_bp, url_prefix="/api")
        with app.test_client() as client:
            resp = client.post("/api/audio/adr/create",
                               json={"project_name": "Test"})
            self.assertEqual(resp.status_code, 403)

    def test_adr_cues_no_session_id(self):
        """GET cues without session_id should return 400."""
        from opencut.routes.audio_post_routes import audio_post_bp
        from flask import Flask
        app = Flask(__name__)
        app.register_blueprint(audio_post_bp, url_prefix="/api")
        with app.test_client() as client:
            resp = client.get("/api/audio/adr/cues")
            self.assertEqual(resp.status_code, 400)

    def test_adr_cue_post_no_csrf(self):
        """POST cue without CSRF should return 403."""
        from opencut.routes.audio_post_routes import audio_post_bp
        from flask import Flask
        app = Flask(__name__)
        app.register_blueprint(audio_post_bp, url_prefix="/api")
        with app.test_client() as client:
            resp = client.post("/api/audio/adr/cue",
                               json={"session_id": "s1", "character_name": "X"})
            self.assertEqual(resp.status_code, 403)


# ============================================================
# Cross-Module Integration Tests
# ============================================================
class TestCrossModule(unittest.TestCase):
    """Integration tests across audio post modules."""

    def test_all_modules_importable(self):
        """All five core modules should import without error."""
        import importlib
        for mod_name in (
            "opencut.core.adr_cue_system",
            "opencut.core.me_mix",
            "opencut.core.dialogue_premix",
            "opencut.core.surround_upmix",
            "opencut.core.foley_cue",
        ):
            importlib.import_module(mod_name)

    def test_routes_importable(self):
        import importlib
        importlib.import_module("opencut.routes.audio_post_routes")

    def test_adr_reasons_constant(self):
        from opencut.core.adr_cue_system import ADR_REASONS
        self.assertIsInstance(ADR_REASONS, tuple)
        self.assertGreater(len(ADR_REASONS), 3)

    def test_adr_statuses_constant(self):
        from opencut.core.adr_cue_system import ADR_STATUSES
        self.assertEqual(ADR_STATUSES, ("pending", "recorded", "approved", "rejected"))

    def test_me_methods_match_route_expectations(self):
        from opencut.core.me_mix import list_methods
        methods = list_methods()
        self.assertTrue(any(m["id"] == "auto" for m in methods))

    def test_premix_presets_match_route(self):
        from opencut.core.dialogue_premix import list_presets
        presets = list_presets()
        self.assertTrue(all("id" in p and "label" in p for p in presets))

    def test_surround_layouts_match_route(self):
        from opencut.core.surround_upmix import list_layouts
        layouts = list_layouts()
        self.assertTrue(all("id" in lay and "channels" in lay for lay in layouts))

    def test_foley_categories_match_route(self):
        from opencut.core.foley_cue import list_categories
        cats = list_categories()
        self.assertTrue(all("id" in c and "label" in c for c in cats))


if __name__ == "__main__":
    unittest.main()
