"""
Tests for OpenCut Music & Safety features.

Covers:
  - Lip Sync Verification (16.2)
  - Rhythm-Driven Effects (16.3)
  - C2PA Content Credentials (27.1)
  - Invisible AI Watermarking (27.2)
  - Evidence Chain-of-Custody (35.3)
  - LTC/VITC Timecode (44.2)
  - Timecode-Based Sync (44.3)
  - Music Safety Routes (smoke tests)
"""

import inspect
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# 16.2 — Lip Sync Verification Tests
# ============================================================
class TestLipSyncVerify(unittest.TestCase):
    """Tests for opencut.core.lip_sync_verify module."""

    def test_verify_lip_sync_signature(self):
        from opencut.core.lip_sync_verify import verify_lip_sync
        sig = inspect.signature(verify_lip_sync)
        self.assertIn("video_path", sig.parameters)
        self.assertIn("threshold", sig.parameters)
        self.assertIn("window_ms", sig.parameters)
        self.assertIn("on_progress", sig.parameters)
        self.assertIsNone(sig.parameters["on_progress"].default)

    def test_compute_mouth_energy_signature(self):
        from opencut.core.lip_sync_verify import compute_mouth_energy
        sig = inspect.signature(compute_mouth_energy)
        self.assertIn("video_path", sig.parameters)
        self.assertIn("window_ms", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    def test_compute_audio_energy_signature(self):
        from opencut.core.lip_sync_verify import compute_audio_energy
        sig = inspect.signature(compute_audio_energy)
        self.assertIn("audio_path", sig.parameters)
        self.assertIn("window_ms", sig.parameters)
        self.assertIn("sample_rate", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    def test_find_drift_segments_signature(self):
        from opencut.core.lip_sync_verify import find_drift_segments
        sig = inspect.signature(find_drift_segments)
        self.assertIn("correlation", sig.parameters)
        self.assertIn("threshold", sig.parameters)
        self.assertIn("min_duration_windows", sig.parameters)

    def test_find_drift_segments_empty(self):
        from opencut.core.lip_sync_verify import find_drift_segments
        result = find_drift_segments([], threshold=0.3)
        self.assertEqual(result, [])

    def test_find_drift_segments_no_drift(self):
        from opencut.core.lip_sync_verify import find_drift_segments
        corr = [0.8, 0.9, 0.7, 0.85, 0.9]
        result = find_drift_segments(corr, threshold=0.3)
        self.assertEqual(result, [])

    def test_find_drift_segments_detects_drift(self):
        from opencut.core.lip_sync_verify import find_drift_segments
        # Low correlation in the middle
        corr = [0.8, 0.9] + [0.1, 0.1, 0.1, 0.1, 0.1, 0.1] + [0.8, 0.9]
        result = find_drift_segments(corr, threshold=0.3, min_duration_windows=5)
        self.assertEqual(len(result), 1)
        self.assertIn("start_ms", result[0])
        self.assertIn("end_ms", result[0])
        self.assertIn("avg_correlation", result[0])

    def test_find_drift_segments_min_duration_filter(self):
        from opencut.core.lip_sync_verify import find_drift_segments
        # Too short drift segment
        corr = [0.8, 0.1, 0.1, 0.8, 0.9]
        result = find_drift_segments(corr, threshold=0.3, min_duration_windows=5)
        self.assertEqual(len(result), 0)

    def test_lip_sync_result_dataclass(self):
        from opencut.core.lip_sync_verify import LipSyncResult
        r = LipSyncResult()
        self.assertEqual(r.overall_correlation, 0.0)
        self.assertEqual(r.drift_segments, [])
        self.assertTrue(r.in_sync)

    def test_verify_lip_sync_file_not_found(self):
        from opencut.core.lip_sync_verify import verify_lip_sync
        with self.assertRaises(FileNotFoundError):
            verify_lip_sync("/nonexistent/video.mp4")

    def test_compute_audio_energy_file_not_found(self):
        from opencut.core.lip_sync_verify import compute_audio_energy
        with self.assertRaises(FileNotFoundError):
            compute_audio_energy("/nonexistent/audio.wav")


# ============================================================
# 16.3 — Rhythm-Driven Effects Tests
# ============================================================
class TestRhythmEffects(unittest.TestCase):
    """Tests for opencut.core.rhythm_effects module."""

    def test_apply_rhythm_effects_signature(self):
        from opencut.core.rhythm_effects import apply_rhythm_effects
        sig = inspect.signature(apply_rhythm_effects)
        self.assertIn("video_path", sig.parameters)
        self.assertIn("audio_path", sig.parameters)
        self.assertIn("effect_map", sig.parameters)
        self.assertIn("output", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    def test_create_effect_map_signature(self):
        from opencut.core.rhythm_effects import create_effect_map
        sig = inspect.signature(create_effect_map)
        self.assertIn("audio_feature", sig.parameters)
        self.assertIn("visual_effect", sig.parameters)
        self.assertIn("intensity", sig.parameters)

    def test_analyze_audio_features_signature(self):
        from opencut.core.rhythm_effects import analyze_audio_features
        sig = inspect.signature(analyze_audio_features)
        self.assertIn("audio_path", sig.parameters)
        self.assertIn("features", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    def test_create_effect_map_valid(self):
        from opencut.core.rhythm_effects import create_effect_map
        m = create_effect_map("amplitude", "zoom", 1.5)
        self.assertEqual(m["audio_feature"], "amplitude")
        self.assertEqual(m["visual_effect"], "zoom")
        self.assertEqual(m["intensity"], 1.5)

    def test_create_effect_map_invalid_feature(self):
        from opencut.core.rhythm_effects import create_effect_map
        with self.assertRaises(ValueError):
            create_effect_map("invalid_feature", "zoom")

    def test_create_effect_map_invalid_effect(self):
        from opencut.core.rhythm_effects import create_effect_map
        with self.assertRaises(ValueError):
            create_effect_map("amplitude", "invalid_effect")

    def test_create_effect_map_clamps_intensity(self):
        from opencut.core.rhythm_effects import create_effect_map
        m = create_effect_map("amplitude", "zoom", 100.0)
        self.assertEqual(m["intensity"], 5.0)

    def test_audio_features_constant(self):
        from opencut.core.rhythm_effects import AUDIO_FEATURES
        self.assertIn("beats", AUDIO_FEATURES)
        self.assertIn("amplitude", AUDIO_FEATURES)
        self.assertIn("rms", AUDIO_FEATURES)

    def test_visual_effects_constant(self):
        from opencut.core.rhythm_effects import VISUAL_EFFECTS
        self.assertIn("zoom", VISUAL_EFFECTS)
        self.assertIn("brightness", VISUAL_EFFECTS)
        self.assertIn("shake", VISUAL_EFFECTS)

    def test_apply_rhythm_effects_file_not_found(self):
        from opencut.core.rhythm_effects import apply_rhythm_effects
        with self.assertRaises(FileNotFoundError):
            apply_rhythm_effects("/nonexistent/video.mp4")


# ============================================================
# 27.1 — C2PA Content Credentials Tests
# ============================================================
class TestC2PA(unittest.TestCase):
    """Tests for opencut.core.c2pa_embed module."""

    def test_create_c2pa_manifest_signature(self):
        from opencut.core.c2pa_embed import create_c2pa_manifest
        sig = inspect.signature(create_c2pa_manifest)
        self.assertIn("operations", sig.parameters)
        self.assertIn("source_hash", sig.parameters)
        self.assertIn("title", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    def test_embed_c2pa_signature(self):
        from opencut.core.c2pa_embed import embed_c2pa
        sig = inspect.signature(embed_c2pa)
        self.assertIn("video_path", sig.parameters)
        self.assertIn("manifest", sig.parameters)
        self.assertIn("output", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    def test_read_c2pa_signature(self):
        from opencut.core.c2pa_embed import read_c2pa
        sig = inspect.signature(read_c2pa)
        self.assertIn("video_path", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    def test_create_manifest_basic(self):
        from opencut.core.c2pa_embed import create_c2pa_manifest
        manifest = create_c2pa_manifest(
            operations=[{"action": "ai_enhance"}],
            source_hash="abc123",
            title="test.mp4",
        )
        self.assertIn("c2pa_version", manifest)
        self.assertIn("claim_generator", manifest)
        self.assertIn("instance_id", manifest)
        self.assertEqual(manifest["source_hash"], "abc123")
        self.assertEqual(len(manifest["operations"]), 1)

    def test_create_manifest_empty_operations(self):
        from opencut.core.c2pa_embed import create_c2pa_manifest
        manifest = create_c2pa_manifest()
        self.assertEqual(manifest["operations"], [])
        self.assertIn("assertions", manifest)

    def test_create_manifest_ai_disclosure(self):
        from opencut.core.c2pa_embed import create_c2pa_manifest
        manifest = create_c2pa_manifest(
            operations=[{"action": "ai_upscale"}, {"action": "trim"}],
        )
        # Should have AI disclosure assertion
        labels = [a["label"] for a in manifest["assertions"]]
        self.assertIn("c2pa.ai_disclosure", labels)

    def test_create_manifest_no_ai_disclosure(self):
        from opencut.core.c2pa_embed import create_c2pa_manifest
        manifest = create_c2pa_manifest(
            operations=[{"action": "trim"}, {"action": "export"}],
        )
        labels = [a["label"] for a in manifest["assertions"]]
        self.assertNotIn("c2pa.ai_disclosure", labels)

    def test_manifest_has_uuid_instance_id(self):
        from opencut.core.c2pa_embed import create_c2pa_manifest
        manifest = create_c2pa_manifest()
        self.assertTrue(manifest["instance_id"].startswith("urn:uuid:"))

    def test_embed_c2pa_file_not_found(self):
        from opencut.core.c2pa_embed import embed_c2pa
        with self.assertRaises(FileNotFoundError):
            embed_c2pa("/nonexistent/video.mp4", {})

    def test_read_c2pa_file_not_found(self):
        from opencut.core.c2pa_embed import read_c2pa
        with self.assertRaises(FileNotFoundError):
            read_c2pa("/nonexistent/video.mp4")

    def test_hash_file_helper(self):
        from opencut.core.c2pa_embed import _hash_file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            f.write(b"test content")
            tmp_path = f.name
        try:
            h = _hash_file(tmp_path)
            self.assertEqual(len(h), 64)  # SHA-256 hex
        finally:
            os.unlink(tmp_path)


# ============================================================
# 27.2 — Invisible AI Watermarking Tests
# ============================================================
class TestInvisibleWatermark(unittest.TestCase):
    """Tests for opencut.core.invisible_watermark module."""

    def test_embed_watermark_signature(self):
        from opencut.core.invisible_watermark import embed_watermark
        sig = inspect.signature(embed_watermark)
        self.assertIn("video_path", sig.parameters)
        self.assertIn("message", sig.parameters)
        self.assertIn("output", sig.parameters)
        self.assertIn("strength", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    def test_extract_watermark_signature(self):
        from opencut.core.invisible_watermark import extract_watermark
        sig = inspect.signature(extract_watermark)
        self.assertIn("video_path", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    def test_verify_watermark_signature(self):
        from opencut.core.invisible_watermark import verify_watermark
        sig = inspect.signature(verify_watermark)
        self.assertIn("video_path", sig.parameters)
        self.assertIn("expected_message", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    def test_message_to_bits_roundtrip(self):
        from opencut.core.invisible_watermark import _bits_to_message, _message_to_bits
        original = "Hello OpenCut!"
        bits = _message_to_bits(original)
        self.assertGreater(len(bits), 0)
        decoded, valid = _bits_to_message(bits)
        self.assertTrue(valid)
        self.assertEqual(decoded, original)

    def test_message_to_bits_empty_roundtrip(self):
        from opencut.core.invisible_watermark import _bits_to_message, _message_to_bits
        bits = _message_to_bits("")
        decoded, valid = _bits_to_message(bits)
        self.assertTrue(valid)
        self.assertEqual(decoded, "")

    def test_bits_to_message_invalid_header(self):
        from opencut.core.invisible_watermark import _bits_to_message
        bad_bits = [0] * 200
        msg, valid = _bits_to_message(bad_bits)
        self.assertFalse(valid)

    def test_bits_to_message_too_short(self):
        from opencut.core.invisible_watermark import _bits_to_message
        msg, valid = _bits_to_message([1, 0, 1])
        self.assertFalse(valid)

    def test_embed_watermark_file_not_found(self):
        from opencut.core.invisible_watermark import embed_watermark
        with self.assertRaises(FileNotFoundError):
            embed_watermark("/nonexistent/video.mp4", "test")

    def test_embed_watermark_empty_message(self):
        from opencut.core.invisible_watermark import embed_watermark
        with self.assertRaises(ValueError):
            embed_watermark("/some/video.mp4", "")

    def test_embed_watermark_message_too_long(self):
        from opencut.core.invisible_watermark import embed_watermark
        with self.assertRaises(ValueError):
            embed_watermark("/some/video.mp4", "x" * 3000)

    def test_embed_bits_in_frame(self):
        from opencut.core.invisible_watermark import _embed_bits_in_frame
        # 16x16 grayscale frame
        frame = bytearray([128] * 256)
        bits = [1, 0, 1, 0]
        modified, count = _embed_bits_in_frame(frame, 16, 16, bits, strength=25)
        self.assertGreater(count, 0)
        self.assertEqual(len(modified), 256)

    def test_watermark_constants(self):
        from opencut.core.invisible_watermark import WATERMARK_MAGIC, WATERMARK_VERSION
        self.assertEqual(WATERMARK_MAGIC, b"OCWM")
        self.assertEqual(WATERMARK_VERSION, 1)


# ============================================================
# 35.3 — Evidence Chain-of-Custody Tests
# ============================================================
class TestEvidenceChain(unittest.TestCase):
    """Tests for opencut.core.evidence_chain module."""

    def test_create_custody_chain_signature(self):
        from opencut.core.evidence_chain import create_custody_chain
        sig = inspect.signature(create_custody_chain)
        self.assertIn("video_path", sig.parameters)
        self.assertIn("operator", sig.parameters)
        self.assertIn("case_id", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    def test_log_operation_signature(self):
        from opencut.core.evidence_chain import log_operation
        sig = inspect.signature(log_operation)
        self.assertIn("chain", sig.parameters)
        self.assertIn("operation_data", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    def test_finalize_chain_signature(self):
        from opencut.core.evidence_chain import finalize_chain
        sig = inspect.signature(finalize_chain)
        self.assertIn("chain", sig.parameters)
        self.assertIn("output_hash", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    def test_export_custody_report_signature(self):
        from opencut.core.evidence_chain import export_custody_report
        sig = inspect.signature(export_custody_report)
        self.assertIn("chain", sig.parameters)
        self.assertIn("output_path", sig.parameters)
        self.assertIn("format", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    @patch("opencut.core.evidence_chain.get_video_info", return_value={"width": 1920, "height": 1080, "fps": 30, "duration": 60})
    def test_create_custody_chain_basic(self, mock_info):
        from opencut.core.evidence_chain import create_custody_chain
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            f.write(b"fake video data")
            tmp = f.name
        try:
            chain = create_custody_chain(tmp, operator="TestOp", case_id="CASE-001")
            self.assertIn("chain_id", chain)
            self.assertTrue(chain["chain_id"].startswith("CoC-"))
            self.assertEqual(chain["operator"], "TestOp")
            self.assertEqual(chain["case_id"], "CASE-001")
            self.assertFalse(chain["finalized"])
            self.assertEqual(len(chain["entries"]), 1)
            self.assertEqual(chain["entries"][0]["operation"], "chain_created")
        finally:
            os.unlink(tmp)

    def test_create_custody_chain_file_not_found(self):
        from opencut.core.evidence_chain import create_custody_chain
        with self.assertRaises(FileNotFoundError):
            create_custody_chain("/nonexistent/video.mp4")

    @patch("opencut.core.evidence_chain.get_video_info", return_value={})
    def test_log_operation_adds_entry(self, mock_info):
        from opencut.core.evidence_chain import create_custody_chain, log_operation
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            f.write(b"fake")
            tmp = f.name
        try:
            chain = create_custody_chain(tmp)
            chain = log_operation(chain, {
                "operation": "trim",
                "description": "Trimmed first 5 seconds",
            })
            self.assertEqual(len(chain["entries"]), 2)
            self.assertEqual(chain["entries"][1]["operation"], "trim")
        finally:
            os.unlink(tmp)

    @patch("opencut.core.evidence_chain.get_video_info", return_value={})
    def test_finalize_chain_locks(self, mock_info):
        from opencut.core.evidence_chain import create_custody_chain, finalize_chain
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            f.write(b"fake")
            tmp = f.name
        try:
            chain = create_custody_chain(tmp)
            chain = finalize_chain(chain, output_hash="abc123")
            self.assertTrue(chain["finalized"])
            self.assertEqual(chain["final_hash"], "abc123")
            self.assertIn("chain_integrity_hash", chain)
        finally:
            os.unlink(tmp)

    @patch("opencut.core.evidence_chain.get_video_info", return_value={})
    def test_finalize_chain_double_finalize_raises(self, mock_info):
        from opencut.core.evidence_chain import create_custody_chain, finalize_chain
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            f.write(b"fake")
            tmp = f.name
        try:
            chain = create_custody_chain(tmp)
            chain = finalize_chain(chain)
            with self.assertRaises(ValueError):
                finalize_chain(chain)
        finally:
            os.unlink(tmp)

    @patch("opencut.core.evidence_chain.get_video_info", return_value={})
    def test_log_operation_on_finalized_raises(self, mock_info):
        from opencut.core.evidence_chain import create_custody_chain, finalize_chain, log_operation
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            f.write(b"fake")
            tmp = f.name
        try:
            chain = create_custody_chain(tmp)
            chain = finalize_chain(chain)
            with self.assertRaises(ValueError):
                log_operation(chain, {"operation": "trim"})
        finally:
            os.unlink(tmp)

    @patch("opencut.core.evidence_chain.get_video_info", return_value={})
    def test_export_json_report(self, mock_info):
        from opencut.core.evidence_chain import create_custody_chain, export_custody_report
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            f.write(b"fake")
            tmp = f.name
        try:
            chain = create_custody_chain(tmp, operator="Tester")
            report_path = tmp + ".report.json"
            result = export_custody_report(chain, report_path, format="json")
            self.assertEqual(result["format"], "json")
            self.assertTrue(os.path.isfile(report_path))
            with open(report_path) as rp:
                report = json.load(rp)
            self.assertEqual(report["report_type"], "chain_of_custody")
            self.assertEqual(report["operator"], "Tester")
            os.unlink(report_path)
        finally:
            os.unlink(tmp)

    @patch("opencut.core.evidence_chain.get_video_info", return_value={})
    def test_export_pdf_report(self, mock_info):
        from opencut.core.evidence_chain import create_custody_chain, export_custody_report
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            f.write(b"fake")
            tmp = f.name
        try:
            chain = create_custody_chain(tmp)
            report_path = tmp + ".report.pdf"
            result = export_custody_report(chain, report_path, format="pdf")
            self.assertEqual(result["format"], "pdf")
            self.assertTrue(os.path.isfile(report_path))
            self.assertGreater(result["size"], 0)
            os.unlink(report_path)
        finally:
            os.unlink(tmp)

    def test_export_invalid_format(self):
        from opencut.core.evidence_chain import export_custody_report
        with self.assertRaises(ValueError):
            export_custody_report({}, "/tmp/out.txt", format="xml")


# ============================================================
# 44.2 — LTC/VITC Timecode Tests
# ============================================================
class TestLtcVitc(unittest.TestCase):
    """Tests for opencut.core.ltc_vitc module."""

    def test_extract_ltc_signature(self):
        from opencut.core.ltc_vitc import extract_ltc
        sig = inspect.signature(extract_ltc)
        self.assertIn("audio_path", sig.parameters)
        self.assertIn("fps", sig.parameters)
        self.assertIn("channel", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    def test_extract_vitc_signature(self):
        from opencut.core.ltc_vitc import extract_vitc
        sig = inspect.signature(extract_vitc)
        self.assertIn("video_path", sig.parameters)
        self.assertIn("scan_lines", sig.parameters)
        self.assertIn("fps", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    def test_parse_ltc_bits_signature(self):
        from opencut.core.ltc_vitc import parse_ltc_bits
        sig = inspect.signature(parse_ltc_bits)
        self.assertIn("audio_data", sig.parameters)
        self.assertIn("sample_rate", sig.parameters)
        self.assertIn("fps", sig.parameters)

    def test_timecode_extraction_dataclass(self):
        from opencut.core.ltc_vitc import TimecodeExtraction
        tc = TimecodeExtraction()
        self.assertEqual(tc.timecodes, [])
        self.assertEqual(tc.frame_numbers, [])
        self.assertEqual(tc.source_type, "")
        self.assertEqual(tc.confidence, 0.0)

    def test_frames_to_tc_basic(self):
        from opencut.core.ltc_vitc import _frames_to_tc
        tc = _frames_to_tc(0, 25.0)
        self.assertEqual(tc, "00:00:00:00")

    def test_frames_to_tc_one_second(self):
        from opencut.core.ltc_vitc import _frames_to_tc
        tc = _frames_to_tc(25, 25.0)
        self.assertEqual(tc, "00:00:01:00")

    def test_frames_to_tc_one_hour(self):
        from opencut.core.ltc_vitc import _frames_to_tc
        tc = _frames_to_tc(25 * 3600, 25.0)
        self.assertEqual(tc, "01:00:00:00")

    def test_tc_to_frames_basic(self):
        from opencut.core.ltc_vitc import _tc_to_frames
        f = _tc_to_frames("00:00:00:00", 25.0)
        self.assertEqual(f, 0)

    def test_tc_to_frames_roundtrip(self):
        from opencut.core.ltc_vitc import _frames_to_tc, _tc_to_frames
        for frame in [0, 1, 25, 750, 90000]:
            tc = _frames_to_tc(frame, 25.0)
            back = _tc_to_frames(tc, 25.0)
            self.assertEqual(back, frame)

    def test_tc_to_frames_invalid(self):
        from opencut.core.ltc_vitc import _tc_to_frames
        self.assertEqual(_tc_to_frames("", 25.0), 0)
        self.assertEqual(_tc_to_frames("invalid", 25.0), 0)

    def test_parse_ltc_bits_empty(self):
        from opencut.core.ltc_vitc import parse_ltc_bits
        result = parse_ltc_bits(b"", 48000)
        self.assertEqual(result, [])

    def test_extract_ltc_file_not_found(self):
        from opencut.core.ltc_vitc import extract_ltc
        with self.assertRaises(FileNotFoundError):
            extract_ltc("/nonexistent/audio.wav")

    def test_extract_vitc_file_not_found(self):
        from opencut.core.ltc_vitc import extract_vitc
        with self.assertRaises(FileNotFoundError):
            extract_vitc("/nonexistent/video.mp4")

    def test_ltc_frame_bits_constant(self):
        from opencut.core.ltc_vitc import LTC_FRAME_BITS
        self.assertEqual(LTC_FRAME_BITS, 80)


# ============================================================
# 44.3 — Timecode-Based Sync Tests
# ============================================================
class TestTcSync(unittest.TestCase):
    """Tests for opencut.core.tc_sync module."""

    def test_sync_by_timecode_signature(self):
        from opencut.core.tc_sync import sync_by_timecode
        sig = inspect.signature(sync_by_timecode)
        self.assertIn("sources", sig.parameters)
        self.assertIn("fps", sig.parameters)
        self.assertIn("output", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    def test_find_common_timecode_range_signature(self):
        from opencut.core.tc_sync import find_common_timecode_range
        sig = inspect.signature(find_common_timecode_range)
        self.assertIn("timecodes", sig.parameters)

    def test_compute_tc_offsets_signature(self):
        from opencut.core.tc_sync import compute_tc_offsets
        sig = inspect.signature(compute_tc_offsets)
        self.assertIn("sources", sig.parameters)

    def test_generate_synced_timeline_signature(self):
        from opencut.core.tc_sync import generate_synced_timeline
        sig = inspect.signature(generate_synced_timeline)
        self.assertIn("sources", sig.parameters)
        self.assertIn("offsets", sig.parameters)
        self.assertIn("output_path", sig.parameters)

    def test_find_common_range_empty(self):
        from opencut.core.tc_sync import find_common_timecode_range
        result = find_common_timecode_range([])
        self.assertFalse(result["valid"])
        self.assertEqual(result["duration_frames"], 0)

    def test_find_common_range_overlap(self):
        from opencut.core.tc_sync import find_common_timecode_range
        tcs = [
            {"start_frame": 0, "end_frame": 100, "fps": 25.0},
            {"start_frame": 50, "end_frame": 150, "fps": 25.0},
        ]
        result = find_common_timecode_range(tcs)
        self.assertTrue(result["valid"])
        self.assertEqual(result["start_frame"], 50)
        self.assertEqual(result["end_frame"], 100)
        self.assertEqual(result["duration_frames"], 50)

    def test_find_common_range_no_overlap(self):
        from opencut.core.tc_sync import find_common_timecode_range
        tcs = [
            {"start_frame": 0, "end_frame": 50, "fps": 25.0},
            {"start_frame": 100, "end_frame": 200, "fps": 25.0},
        ]
        result = find_common_timecode_range(tcs)
        self.assertFalse(result["valid"])

    def test_compute_offsets_basic(self):
        from opencut.core.tc_sync import compute_tc_offsets
        sources = [
            {"filepath": "a.mp4", "start_frame": 0, "fps": 25.0},
            {"filepath": "b.mp4", "start_frame": 50, "fps": 25.0},
        ]
        offsets = compute_tc_offsets(sources)
        self.assertIn("a.mp4", offsets)
        self.assertIn("b.mp4", offsets)
        self.assertEqual(offsets["a.mp4"]["offset_frames"], 0)
        self.assertEqual(offsets["b.mp4"]["offset_frames"], 50)

    def test_compute_offsets_empty(self):
        from opencut.core.tc_sync import compute_tc_offsets
        offsets = compute_tc_offsets([])
        self.assertEqual(offsets, {})

    def test_sync_by_timecode_no_sources(self):
        from opencut.core.tc_sync import sync_by_timecode
        with self.assertRaises(ValueError):
            sync_by_timecode([])

    def test_sync_by_timecode_file_not_found(self):
        from opencut.core.tc_sync import sync_by_timecode
        with self.assertRaises(FileNotFoundError):
            sync_by_timecode(["/nonexistent/a.mp4"])

    def test_generate_json_timeline(self):
        from opencut.core.tc_sync import generate_synced_timeline
        sources = [
            {"filepath": "cam1.mp4", "label": "cam1", "start_tc": "01:00:00:00",
             "end_tc": "01:01:00:00", "fps": 25.0, "duration": 60.0, "tc_source": "embedded",
             "start_frame": 90000, "end_frame": 91500},
        ]
        offsets = {"cam1.mp4": {"offset_frames": 0, "offset_seconds": 0.0}}
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
            out = f.name
        try:
            result = generate_synced_timeline(sources, offsets, out, format="json")
            self.assertEqual(result["format"], "json")
            self.assertTrue(os.path.isfile(out))
            with open(out) as fp:
                tl = json.load(fp)
            self.assertEqual(tl["total_sources"], 1)
            self.assertEqual(len(tl["tracks"]), 1)
        finally:
            os.unlink(out)

    def test_generate_edl_timeline(self):
        from opencut.core.tc_sync import generate_synced_timeline
        sources = [
            {"filepath": "cam1.mp4", "label": "cam1", "start_tc": "01:00:00:00",
             "end_tc": "01:01:00:00", "fps": 25.0, "duration": 60.0,
             "start_frame": 90000, "end_frame": 91500},
        ]
        offsets = {"cam1.mp4": {"offset_frames": 0, "offset_seconds": 0.0}}
        with tempfile.NamedTemporaryFile(delete=False, suffix=".edl") as f:
            out = f.name
        try:
            result = generate_synced_timeline(sources, offsets, out, format="edl")
            self.assertEqual(result["format"], "edl")
            with open(out) as fp:
                content = fp.read()
            self.assertIn("TITLE:", content)
            self.assertIn("cam1.mp4", content)
        finally:
            os.unlink(out)

    def test_tc_conversion_helpers(self):
        from opencut.core.tc_sync import _frames_to_tc, _tc_to_frames
        self.assertEqual(_tc_to_frames("00:00:00:00", 25.0), 0)
        self.assertEqual(_frames_to_tc(0, 25.0), "00:00:00:00")
        self.assertEqual(_frames_to_tc(25, 25.0), "00:00:01:00")


# ============================================================
# Route smoke tests
# ============================================================
class TestMusicSafetyRoutes(unittest.TestCase):
    """Smoke tests for music_safety_routes Blueprint."""

    def test_blueprint_exists(self):
        from opencut.routes.music_safety_routes import music_safety_bp
        self.assertEqual(music_safety_bp.name, "music_safety")

    def test_rhythm_presets_endpoint(self):
        """GET /video/rhythm-effects/presets should return features and effects."""
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        app = create_app(config=OpenCutConfig())
        app.config["TESTING"] = True
        client = app.test_client()
        resp = client.get("/video/rhythm-effects/presets")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("audio_features", data)
        self.assertIn("visual_effects", data)
        self.assertIn("amplitude", data["audio_features"])
        self.assertIn("zoom", data["visual_effects"])

    def test_lip_sync_route_no_filepath(self):
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        app = create_app(config=OpenCutConfig())
        app.config["TESTING"] = True
        client = app.test_client()
        # Get CSRF token
        health = client.get("/health")
        token = health.get_json().get("csrf_token", "")
        resp = client.post("/video/lip-sync-verify",
                           json={},
                           headers={"X-OpenCut-Token": token})
        self.assertEqual(resp.status_code, 400)

    def test_watermark_embed_route_no_message(self):
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        app = create_app(config=OpenCutConfig())
        app.config["TESTING"] = True
        client = app.test_client()
        health = client.get("/health")
        token = health.get_json().get("csrf_token", "")
        resp = client.post("/video/watermark/embed",
                           json={"filepath": "/nonexistent.mp4"},
                           headers={"X-OpenCut-Token": token})
        # Should get 400 (invalid filepath) since file doesn't exist
        self.assertIn(resp.status_code, [400, 202])

    def test_blueprint_registered(self):
        """Verify music_safety_bp is registered in the Flask app."""
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        app = create_app(config=OpenCutConfig())
        bp_names = [bp.name for bp in app.blueprints.values()]
        self.assertIn("music_safety", bp_names)


if __name__ == "__main__":
    unittest.main()
