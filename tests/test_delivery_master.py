"""
Tests for OpenCut Delivery & Mastering (Category 70).

Covers:
  - DCPConfig / DCPAsset / DCPResult dataclasses and to_dict
  - DCP format definitions and config validation
  - DCP export with mocked FFmpeg (video MXF, audio MXF, CPL, PKL, ASSETMAP, VOLINDEX)
  - IMFConfig / IMFAudioTrack / IMFResult dataclasses and to_dict
  - IMF profile definitions and config validation
  - IMF export with mocked FFmpeg
  - IMF timecode parsing and conversion
  - DeliveryProfile / SpecRequirement / SpecCompareResult dataclasses
  - Built-in delivery spec definitions (Netflix, YouTube, etc.)
  - Spec CRUD: get_spec, list_specs, create_custom_spec, compare_specs
  - Spec suggestion via suggest_spec
  - DeliverySpec / CheckResult / ValidationResult dataclasses
  - DELIVERY_SPECS built-in validation specs
  - validate_delivery with mocked ffprobe/ffmpeg
  - Check evaluation engine (_evaluate_operator, _snap_frame_rate)
  - RenderConfig / RenderJob / MultiRenderResult dataclasses
  - Multi-render with mocked subprocess (sequential, parallel)
  - Render cancellation
  - GPU codec detection
  - Route smoke tests for all delivery_master_bp endpoints
"""

import os
import sys
import json
import tempfile
import shutil
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ===================================================================
# Feature 70.1 — DCP Export
# ===================================================================


class TestDCPConfig(unittest.TestCase):
    """Tests for DCPConfig dataclass."""

    def test_defaults(self):
        from opencut.core.dcp_export import DCPConfig
        c = DCPConfig()
        self.assertEqual(c.title, "Untitled")
        self.assertEqual(c.format_key, "2K_flat")
        self.assertEqual(c.frame_rate, 24)
        self.assertEqual(c.content_kind, "feature")
        self.assertEqual(c.audio_channels, 6)
        self.assertEqual(c.audio_sample_rate, 48000)
        self.assertEqual(c.audio_bit_depth, 24)
        self.assertFalse(c.encrypt)
        self.assertFalse(c.three_d)

    def test_to_dict(self):
        from opencut.core.dcp_export import DCPConfig
        c = DCPConfig(title="Test Film", format_key="4K_scope")
        d = c.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["title"], "Test Film")
        self.assertEqual(d["format_key"], "4K_scope")
        self.assertIn("frame_rate", d)
        self.assertIn("audio_channels", d)

    def test_validate_valid(self):
        from opencut.core.dcp_export import DCPConfig
        c = DCPConfig()
        errors = c.validate()
        self.assertEqual(errors, [])

    def test_validate_bad_format(self):
        from opencut.core.dcp_export import DCPConfig
        c = DCPConfig(format_key="8K_ultra")
        errors = c.validate()
        self.assertTrue(any("format_key" in e for e in errors))

    def test_validate_bad_frame_rate(self):
        from opencut.core.dcp_export import DCPConfig
        c = DCPConfig(frame_rate=30)
        errors = c.validate()
        self.assertTrue(any("Frame rate" in e for e in errors))

    def test_validate_bad_channels(self):
        from opencut.core.dcp_export import DCPConfig
        c = DCPConfig(audio_channels=0)
        errors = c.validate()
        self.assertTrue(any("audio_channels" in e for e in errors))

    def test_validate_bad_sample_rate(self):
        from opencut.core.dcp_export import DCPConfig
        c = DCPConfig(audio_sample_rate=44100)
        errors = c.validate()
        self.assertTrue(any("audio_sample_rate" in e for e in errors))

    def test_validate_bad_content_kind(self):
        from opencut.core.dcp_export import DCPConfig
        c = DCPConfig(content_kind="music_video")
        errors = c.validate()
        self.assertTrue(any("content_kind" in e for e in errors))


class TestDCPAsset(unittest.TestCase):
    """Tests for DCPAsset dataclass."""

    def test_defaults(self):
        from opencut.core.dcp_export import DCPAsset
        a = DCPAsset()
        self.assertEqual(a.uuid, "")
        self.assertEqual(a.filename, "")
        self.assertEqual(a.asset_type, "")
        self.assertEqual(a.hash_algorithm, "SHA-1")

    def test_to_dict(self):
        from opencut.core.dcp_export import DCPAsset
        a = DCPAsset(uuid="urn:uuid:123", filename="video.mxf",
                     asset_type="video_mxf", file_size=1000)
        d = a.to_dict()
        self.assertEqual(d["uuid"], "urn:uuid:123")
        self.assertEqual(d["file_size"], 1000)


class TestDCPResult(unittest.TestCase):
    """Tests for DCPResult dataclass."""

    def test_defaults(self):
        from opencut.core.dcp_export import DCPResult
        r = DCPResult()
        self.assertEqual(r.output_dir, "")
        self.assertEqual(r.assets, [])
        self.assertEqual(r.frame_rate, 24)
        self.assertEqual(r.total_size_bytes, 0)

    def test_to_dict_with_assets(self):
        from opencut.core.dcp_export import DCPAsset, DCPResult
        asset = DCPAsset(uuid="urn:uuid:abc", filename="test.mxf")
        r = DCPResult(title="Film", assets=[asset], duration_seconds=120.5)
        d = r.to_dict()
        self.assertEqual(d["title"], "Film")
        self.assertEqual(d["duration_seconds"], 120.5)
        self.assertEqual(len(d["assets"]), 1)
        self.assertEqual(d["assets"][0]["uuid"], "urn:uuid:abc")


class TestDCPFormats(unittest.TestCase):
    """Tests for DCP_FORMATS dict."""

    def test_all_formats_present(self):
        from opencut.core.dcp_export import DCP_FORMATS
        self.assertIn("2K_flat", DCP_FORMATS)
        self.assertIn("2K_scope", DCP_FORMATS)
        self.assertIn("4K_flat", DCP_FORMATS)
        self.assertIn("4K_scope", DCP_FORMATS)

    def test_2k_flat_dimensions(self):
        from opencut.core.dcp_export import DCP_FORMATS
        f = DCP_FORMATS["2K_flat"]
        self.assertEqual(f["width"], 1998)
        self.assertEqual(f["height"], 1080)

    def test_4k_scope_dimensions(self):
        from opencut.core.dcp_export import DCP_FORMATS
        f = DCP_FORMATS["4K_scope"]
        self.assertEqual(f["width"], 4096)
        self.assertEqual(f["height"], 1716)

    def test_all_formats_have_required_keys(self):
        from opencut.core.dcp_export import DCP_FORMATS
        for key, fmt in DCP_FORMATS.items():
            self.assertIn("label", fmt, f"{key} missing 'label'")
            self.assertIn("width", fmt, f"{key} missing 'width'")
            self.assertIn("height", fmt, f"{key} missing 'height'")
            self.assertIn("aspect", fmt, f"{key} missing 'aspect'")
            self.assertIn("resolution", fmt, f"{key} missing 'resolution'")


class TestDCPExport(unittest.TestCase):
    """Tests for export_dcp with mocked FFmpeg."""

    def test_file_not_found(self):
        from opencut.core.dcp_export import export_dcp
        with self.assertRaises(FileNotFoundError):
            export_dcp("/nonexistent/video.mp4", "/tmp/out")

    def test_invalid_config(self):
        from opencut.core.dcp_export import DCPConfig, export_dcp
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"\x00" * 100)
            tmp = f.name
        try:
            config = DCPConfig(format_key="invalid")
            with self.assertRaises(ValueError):
                export_dcp(tmp, "/tmp/out", config)
        finally:
            os.unlink(tmp)

    @patch("opencut.core.dcp_export._sha1_file", return_value="fakehash==")
    @patch("opencut.core.dcp_export._file_size", return_value=1024)
    @patch("opencut.core.dcp_export._sp.run")
    def test_export_dcp_success(self, mock_run, mock_fsize, mock_sha):
        from opencut.core.dcp_export import DCPConfig, export_dcp

        # Mock ffprobe/ffmpeg responses
        probe_response = json.dumps({
            "streams": [{
                "width": 1920, "height": 1080, "r_frame_rate": "24/1",
                "codec_name": "h264", "pix_fmt": "yuv420p", "nb_frames": "2400",
                "duration": "100.0",
            }],
            "format": {"duration": "100.0", "nb_streams": "2"},
        }).encode()

        mock_run.return_value = MagicMock(
            returncode=0, stdout=probe_response,
            stderr=b"", poll=MagicMock(return_value=0),
        )

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"\x00" * 100)
            tmp = f.name

        out_dir = tempfile.mkdtemp()
        try:
            config = DCPConfig(title="TestFilm", format_key="2K_flat")
            progress_calls = []
            result = export_dcp(tmp, out_dir, config,
                                on_progress=lambda p, m: progress_calls.append((p, m)))

            self.assertIn("output_dir", result)
            self.assertEqual(result["title"], "TestFilm")
            self.assertEqual(result["format_label"], "2K Flat (1.85:1)")
            self.assertEqual(result["frame_rate"], 24)
            self.assertIsInstance(result["assets"], list)
            self.assertTrue(len(progress_calls) > 0)
        finally:
            os.unlink(tmp)
            shutil.rmtree(out_dir, ignore_errors=True)


# ===================================================================
# Feature 70.2 — IMF Package
# ===================================================================


class TestIMFConfig(unittest.TestCase):
    """Tests for IMFConfig dataclass."""

    def test_defaults(self):
        from opencut.core.imf_package import IMFConfig
        c = IMFConfig()
        self.assertEqual(c.title, "Untitled")
        self.assertEqual(c.profile, "application_2")
        self.assertEqual(c.frame_rate, 24)
        self.assertEqual(c.width, 1920)
        self.assertEqual(c.height, 1080)
        self.assertFalse(c.supplemental)

    def test_to_dict(self):
        from opencut.core.imf_package import IMFConfig
        c = IMFConfig(title="IMF Test", profile="application_2e")
        d = c.to_dict()
        self.assertEqual(d["title"], "IMF Test")
        self.assertEqual(d["profile"], "application_2e")
        self.assertIn("audio_tracks", d)

    def test_validate_valid(self):
        from opencut.core.imf_package import IMFConfig
        c = IMFConfig()
        self.assertEqual(c.validate(), [])

    def test_validate_bad_profile(self):
        from opencut.core.imf_package import IMFConfig
        c = IMFConfig(profile="nonexistent")
        errors = c.validate()
        self.assertTrue(len(errors) > 0)

    def test_validate_width_exceeds_profile(self):
        from opencut.core.imf_package import IMFConfig
        c = IMFConfig(profile="application_2", width=4096)
        errors = c.validate()
        self.assertTrue(any("Width" in e for e in errors))

    def test_validate_supplemental_without_cpl(self):
        from opencut.core.imf_package import IMFConfig
        c = IMFConfig(supplemental=True, original_cpl_id="")
        errors = c.validate()
        self.assertTrue(any("supplemental" in e.lower() for e in errors))

    def test_validate_bad_codec(self):
        from opencut.core.imf_package import IMFConfig
        c = IMFConfig(video_codec="h264")
        errors = c.validate()
        self.assertTrue(any("Codec" in e for e in errors))


class TestIMFAudioTrack(unittest.TestCase):
    """Tests for IMFAudioTrack dataclass."""

    def test_defaults(self):
        from opencut.core.imf_package import IMFAudioTrack
        t = IMFAudioTrack()
        self.assertEqual(t.language, "en")
        self.assertEqual(t.label, "English")
        self.assertEqual(t.channels, 6)

    def test_to_dict(self):
        from opencut.core.imf_package import IMFAudioTrack
        t = IMFAudioTrack(language="es", label="Spanish")
        d = t.to_dict()
        self.assertEqual(d["language"], "es")
        self.assertEqual(d["label"], "Spanish")


class TestIMFResult(unittest.TestCase):
    """Tests for IMFResult dataclass."""

    def test_defaults(self):
        from opencut.core.imf_package import IMFResult
        r = IMFResult()
        self.assertEqual(r.assets, [])
        self.assertEqual(r.audio_mxf_paths, [])
        self.assertFalse(r.supplemental)

    def test_to_dict(self):
        from opencut.core.imf_package import IMFResult
        r = IMFResult(title="Test", profile="application_2",
                      profile_label="App 2", duration_seconds=60.123)
        d = r.to_dict()
        self.assertEqual(d["duration_seconds"], 60.123)
        self.assertFalse(d["supplemental"])


class TestIMFProfiles(unittest.TestCase):
    """Tests for IMF_PROFILES dict."""

    def test_all_profiles_present(self):
        from opencut.core.imf_package import IMF_PROFILES
        self.assertIn("application_2", IMF_PROFILES)
        self.assertIn("application_2e", IMF_PROFILES)
        self.assertIn("application_4", IMF_PROFILES)
        self.assertIn("application_5", IMF_PROFILES)

    def test_profile_keys(self):
        from opencut.core.imf_package import IMF_PROFILES
        for name, p in IMF_PROFILES.items():
            self.assertIn("label", p, f"{name} missing 'label'")
            self.assertIn("max_width", p, f"{name} missing 'max_width'")
            self.assertIn("video_codecs", p, f"{name} missing 'video_codecs'")
            self.assertIn("frame_rates", p, f"{name} missing 'frame_rates'")


class TestIMFTimecode(unittest.TestCase):
    """Tests for IMF timecode parsing/conversion."""

    def test_parse_timecode(self):
        from opencut.core.imf_package import _parse_timecode
        frames = _parse_timecode("01:00:00:00", 24)
        self.assertEqual(frames, 86400)

    def test_parse_timecode_with_frames(self):
        from opencut.core.imf_package import _parse_timecode
        frames = _parse_timecode("00:01:00:12", 24)
        self.assertEqual(frames, 1452)

    def test_parse_invalid_timecode(self):
        from opencut.core.imf_package import _parse_timecode
        self.assertEqual(_parse_timecode("invalid", 24), 0)

    def test_frames_to_timecode(self):
        from opencut.core.imf_package import _frames_to_timecode
        tc = _frames_to_timecode(86400, 24)
        self.assertEqual(tc, "01:00:00:00")

    def test_frames_to_timecode_roundtrip(self):
        from opencut.core.imf_package import _parse_timecode, _frames_to_timecode
        original = "00:02:30:12"
        frames = _parse_timecode(original, 24)
        result = _frames_to_timecode(frames, 24)
        self.assertEqual(result, original)


class TestIMFExport(unittest.TestCase):
    """Tests for export_imf with mocked FFmpeg."""

    def test_file_not_found(self):
        from opencut.core.imf_package import export_imf
        with self.assertRaises(FileNotFoundError):
            export_imf("/nonexistent/video.mp4", "/tmp/out")

    def test_invalid_config(self):
        from opencut.core.imf_package import IMFConfig, export_imf
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"\x00" * 100)
            tmp = f.name
        try:
            config = IMFConfig(profile="bad_profile")
            with self.assertRaises(ValueError):
                export_imf(tmp, "/tmp/out", config)
        finally:
            os.unlink(tmp)


# ===================================================================
# Feature 70.5 — Delivery Spec Manager
# ===================================================================


class TestSpecRequirement(unittest.TestCase):
    """Tests for SpecRequirement dataclass."""

    def test_defaults(self):
        from opencut.core.delivery_spec import SpecRequirement
        r = SpecRequirement()
        self.assertEqual(r.category, "")
        self.assertEqual(r.operator, "eq")
        self.assertEqual(r.severity, "error")
        self.assertTrue(r.required)

    def test_to_dict(self):
        from opencut.core.delivery_spec import SpecRequirement
        r = SpecRequirement(category="video", field_name="codec",
                            operator="in", value=["h264", "h265"])
        d = r.to_dict()
        self.assertEqual(d["category"], "video")
        self.assertEqual(d["value"], ["h264", "h265"])

    def test_describe_custom(self):
        from opencut.core.delivery_spec import SpecRequirement
        r = SpecRequirement(description="Custom description")
        self.assertEqual(r.describe(), "Custom description")

    def test_describe_auto(self):
        from opencut.core.delivery_spec import SpecRequirement
        r = SpecRequirement(category="audio", field_name="sample_rate",
                            operator="eq", value=48000, unit="Hz")
        desc = r.describe()
        self.assertIn("audio.sample_rate", desc)
        self.assertIn("48000", desc)


class TestDeliveryProfile(unittest.TestCase):
    """Tests for DeliveryProfile dataclass."""

    def test_defaults(self):
        from opencut.core.delivery_spec import DeliveryProfile
        p = DeliveryProfile()
        self.assertEqual(p.requirements, [])
        self.assertFalse(p.built_in)

    def test_to_dict(self):
        from opencut.core.delivery_spec import DeliveryProfile, SpecRequirement
        req = SpecRequirement(category="video", field_name="codec")
        p = DeliveryProfile(name="test", requirements=[req])
        d = p.to_dict()
        self.assertEqual(d["name"], "test")
        self.assertEqual(len(d["requirements"]), 1)

    def test_requirements_by_category(self):
        from opencut.core.delivery_spec import DeliveryProfile, SpecRequirement
        reqs = [
            SpecRequirement(category="video", field_name="codec"),
            SpecRequirement(category="audio", field_name="sample_rate"),
            SpecRequirement(category="video", field_name="width"),
        ]
        p = DeliveryProfile(requirements=reqs)
        video_reqs = p.get_requirements_by_category("video")
        self.assertEqual(len(video_reqs), 2)

    def test_requirement_count(self):
        from opencut.core.delivery_spec import DeliveryProfile, SpecRequirement
        reqs = [
            SpecRequirement(category="video"),
            SpecRequirement(category="video"),
            SpecRequirement(category="audio"),
        ]
        p = DeliveryProfile(requirements=reqs)
        counts = p.requirement_count()
        self.assertEqual(counts["video"], 2)
        self.assertEqual(counts["audio"], 1)


class TestBuiltInSpecs(unittest.TestCase):
    """Tests for BUILT_IN_SPECS dict."""

    def test_all_specs_present(self):
        from opencut.core.delivery_spec import BUILT_IN_SPECS
        expected = ["netflix", "youtube", "apple_tv_plus", "amazon",
                    "broadcast_ebu", "dcp", "theatrical"]
        for name in expected:
            self.assertIn(name, BUILT_IN_SPECS, f"Missing spec: {name}")

    def test_all_specs_are_built_in(self):
        from opencut.core.delivery_spec import BUILT_IN_SPECS
        for name, profile in BUILT_IN_SPECS.items():
            self.assertTrue(profile.built_in, f"{name} not marked built_in")

    def test_all_specs_have_requirements(self):
        from opencut.core.delivery_spec import BUILT_IN_SPECS
        for name, profile in BUILT_IN_SPECS.items():
            self.assertTrue(len(profile.requirements) > 0,
                            f"{name} has no requirements")

    def test_netflix_has_video_checks(self):
        from opencut.core.delivery_spec import BUILT_IN_SPECS
        netflix = BUILT_IN_SPECS["netflix"]
        video_reqs = netflix.get_requirements_by_category("video")
        self.assertTrue(len(video_reqs) >= 5)

    def test_dcp_requires_jpeg2000(self):
        from opencut.core.delivery_spec import BUILT_IN_SPECS
        dcp = BUILT_IN_SPECS["dcp"]
        codec_req = [r for r in dcp.requirements
                     if r.category == "video" and r.field_name == "codec"]
        self.assertTrue(len(codec_req) > 0)
        self.assertIn("jpeg2000", str(codec_req[0].value))


class TestGetSpec(unittest.TestCase):
    """Tests for get_spec function."""

    def test_get_builtin(self):
        from opencut.core.delivery_spec import get_spec
        spec = get_spec("netflix")
        self.assertIsNotNone(spec)
        self.assertEqual(spec["name"], "netflix")

    def test_get_nonexistent(self):
        from opencut.core.delivery_spec import get_spec
        spec = get_spec("nonexistent_spec_xyz")
        self.assertIsNone(spec)

    def test_get_case_insensitive(self):
        from opencut.core.delivery_spec import get_spec
        spec = get_spec("YouTube")
        self.assertIsNotNone(spec)
        self.assertEqual(spec["name"], "youtube")


class TestListSpecs(unittest.TestCase):
    """Tests for list_specs function."""

    def test_list_returns_all_builtin(self):
        from opencut.core.delivery_spec import list_specs
        specs = list_specs()
        names = [s["name"] for s in specs]
        self.assertIn("netflix", names)
        self.assertIn("youtube", names)

    def test_list_entries_have_required_keys(self):
        from opencut.core.delivery_spec import list_specs
        specs = list_specs()
        for s in specs:
            self.assertIn("name", s)
            self.assertIn("display_name", s)
            self.assertIn("platform", s)
            self.assertIn("built_in", s)


class TestCreateCustomSpec(unittest.TestCase):
    """Tests for create_custom_spec function."""

    def test_create_and_retrieve(self):
        from opencut.core.delivery_spec import (
            create_custom_spec, get_spec, delete_custom_spec,
        )
        try:
            result = create_custom_spec(
                name="test_custom_spec_70",
                requirements=[{
                    "category": "video", "field_name": "codec",
                    "operator": "eq", "value": "h264",
                }],
                display_name="Test Custom",
            )
            self.assertEqual(result["name"], "test_custom_spec_70")
            fetched = get_spec("test_custom_spec_70")
            self.assertIsNotNone(fetched)
        finally:
            delete_custom_spec("test_custom_spec_70")

    def test_create_empty_name(self):
        from opencut.core.delivery_spec import create_custom_spec
        with self.assertRaises(ValueError):
            create_custom_spec(name="")

    def test_create_overwrite_builtin(self):
        from opencut.core.delivery_spec import create_custom_spec
        with self.assertRaises(ValueError):
            create_custom_spec(name="netflix")


class TestCompareSpecs(unittest.TestCase):
    """Tests for compare_specs function."""

    def test_compare_same_spec(self):
        from opencut.core.delivery_spec import compare_specs
        result = compare_specs("netflix", "netflix")
        self.assertEqual(result["total_differences"], 0)
        self.assertTrue(len(result["common"]) > 0)

    def test_compare_different_specs(self):
        from opencut.core.delivery_spec import compare_specs
        result = compare_specs("netflix", "dcp")
        self.assertTrue(result["total_differences"] > 0)
        self.assertIn("spec_a_name", result)
        self.assertIn("spec_b_name", result)

    def test_compare_nonexistent(self):
        from opencut.core.delivery_spec import compare_specs
        with self.assertRaises(ValueError):
            compare_specs("netflix", "nonexistent_xyz")


class TestSuggestSpec(unittest.TestCase):
    """Tests for suggest_spec function."""

    def test_suggest_dcp(self):
        from opencut.core.delivery_spec import suggest_spec
        result = suggest_spec({"codec": "jpeg2000", "container": "mxf"})
        self.assertEqual(result, "dcp")

    def test_suggest_youtube(self):
        from opencut.core.delivery_spec import suggest_spec
        result = suggest_spec({"codec": "h264", "container": "mp4",
                               "width": 1280})
        self.assertEqual(result, "youtube")

    def test_suggest_none(self):
        from opencut.core.delivery_spec import suggest_spec
        result = suggest_spec({})
        self.assertIsNone(result)

    def test_suggest_broadcast(self):
        from opencut.core.delivery_spec import suggest_spec
        result = suggest_spec({
            "codec": "h264", "container": "mxf",
            "audio_codec": "pcm_s24le", "fps": 25,
        })
        self.assertEqual(result, "broadcast_ebu")


# ===================================================================
# Feature 70.3 — Delivery Validation
# ===================================================================


class TestCheckResult(unittest.TestCase):
    """Tests for CheckResult dataclass."""

    def test_defaults(self):
        from opencut.core.delivery_validate import CheckResult
        cr = CheckResult()
        self.assertEqual(cr.check_name, "")
        self.assertFalse(cr.passed)
        self.assertEqual(cr.severity, "error")

    def test_to_dict(self):
        from opencut.core.delivery_validate import CheckResult
        cr = CheckResult(check_name="video.codec", passed=True,
                         expected="h264", actual="h264")
        d = cr.to_dict()
        self.assertTrue(d["passed"])
        self.assertEqual(d["check_name"], "video.codec")


class TestValidationResult(unittest.TestCase):
    """Tests for ValidationResult dataclass."""

    def test_defaults(self):
        from opencut.core.delivery_validate import ValidationResult
        vr = ValidationResult()
        self.assertFalse(vr.passed)
        self.assertEqual(vr.verdict, "")
        self.assertEqual(vr.errors, [])

    def test_to_dict(self):
        from opencut.core.delivery_validate import CheckResult, ValidationResult
        err = CheckResult(check_name="v.c", passed=False, violation="bad codec")
        vr = ValidationResult(
            file_path="/test.mp4", spec_name="netflix",
            passed=False, total_checks=10, failed_checks=1,
            errors=[err], verdict="FAIL",
        )
        d = vr.to_dict()
        self.assertEqual(d["verdict"], "FAIL")
        self.assertEqual(len(d["errors"]), 1)


class TestDeliverySpecs(unittest.TestCase):
    """Tests for DELIVERY_SPECS built-in validation specs."""

    def test_all_specs_present(self):
        from opencut.core.delivery_validate import DELIVERY_SPECS
        expected = ["netflix", "youtube", "broadcast_ebu", "dcp",
                    "imf", "apple_tv_plus", "amazon"]
        for name in expected:
            self.assertIn(name, DELIVERY_SPECS, f"Missing: {name}")

    def test_specs_have_checks(self):
        from opencut.core.delivery_validate import DELIVERY_SPECS
        for name, spec in DELIVERY_SPECS.items():
            self.assertIn("checks", spec, f"{name} missing 'checks'")
            self.assertTrue(len(spec["checks"]) > 0,
                            f"{name} has no checks")

    def test_check_tuple_format(self):
        from opencut.core.delivery_validate import DELIVERY_SPECS
        for name, spec in DELIVERY_SPECS.items():
            for idx, check in enumerate(spec["checks"]):
                self.assertEqual(len(check), 6,
                                 f"{name} check {idx} has wrong length")
                cat, field, op, val, sev, desc = check
                self.assertIn(cat, ("video", "audio", "container", "subtitle"),
                              f"{name} check {idx} bad category")


class TestSnapFrameRate(unittest.TestCase):
    """Tests for _snap_frame_rate helper."""

    def test_snap_23976(self):
        from opencut.core.delivery_validate import _snap_frame_rate
        self.assertEqual(_snap_frame_rate(23.98), 23.976)

    def test_snap_29970(self):
        from opencut.core.delivery_validate import _snap_frame_rate
        self.assertEqual(_snap_frame_rate(29.97), 29.97)

    def test_snap_exact_24(self):
        from opencut.core.delivery_validate import _snap_frame_rate
        self.assertEqual(_snap_frame_rate(24.0), 24)

    def test_snap_zero(self):
        from opencut.core.delivery_validate import _snap_frame_rate
        self.assertEqual(_snap_frame_rate(0.0), 0.0)

    def test_snap_non_standard(self):
        from opencut.core.delivery_validate import _snap_frame_rate
        # Far from any standard rate, should round
        result = _snap_frame_rate(15.5)
        self.assertIsInstance(result, float)


class TestEvaluateOperator(unittest.TestCase):
    """Tests for _evaluate_operator helper."""

    def test_eq_numbers(self):
        from opencut.core.delivery_validate import _evaluate_operator
        self.assertTrue(_evaluate_operator("eq", 48000, 48000))
        self.assertFalse(_evaluate_operator("eq", 44100, 48000))

    def test_eq_bool(self):
        from opencut.core.delivery_validate import _evaluate_operator
        self.assertTrue(_evaluate_operator("eq", True, True))
        self.assertFalse(_evaluate_operator("eq", False, True))

    def test_neq(self):
        from opencut.core.delivery_validate import _evaluate_operator
        self.assertTrue(_evaluate_operator("neq", "h264", "h265"))
        self.assertFalse(_evaluate_operator("neq", "h264", "h264"))

    def test_gte(self):
        from opencut.core.delivery_validate import _evaluate_operator
        self.assertTrue(_evaluate_operator("gte", 1920, 1920))
        self.assertTrue(_evaluate_operator("gte", 3840, 1920))
        self.assertFalse(_evaluate_operator("gte", 1280, 1920))

    def test_lte(self):
        from opencut.core.delivery_validate import _evaluate_operator
        self.assertTrue(_evaluate_operator("lte", 48000, 48000))
        self.assertFalse(_evaluate_operator("lte", 96000, 48000))

    def test_in_list(self):
        from opencut.core.delivery_validate import _evaluate_operator
        self.assertTrue(_evaluate_operator("in", "h264", ["h264", "h265"]))
        self.assertFalse(_evaluate_operator("in", "vp9", ["h264", "h265"]))

    def test_in_numeric_approximate(self):
        from opencut.core.delivery_validate import _evaluate_operator
        self.assertTrue(_evaluate_operator("in", 23.976, [23.976, 24, 25]))
        self.assertTrue(_evaluate_operator("in", 23.98, [23.976, 24, 25]))

    def test_range(self):
        from opencut.core.delivery_validate import _evaluate_operator
        self.assertTrue(_evaluate_operator("range", -20.0, [-27, -14]))
        self.assertFalse(_evaluate_operator("range", -30.0, [-27, -14]))
        self.assertFalse(_evaluate_operator("range", -10.0, [-27, -14]))

    def test_regex(self):
        from opencut.core.delivery_validate import _evaluate_operator
        self.assertTrue(_evaluate_operator("regex", "h264", r"h26[45]"))
        self.assertFalse(_evaluate_operator("regex", "vp9", r"h26[45]"))


class TestValidateDelivery(unittest.TestCase):
    """Tests for validate_delivery with mocked ffprobe/ffmpeg."""

    def test_file_not_found(self):
        from opencut.core.delivery_validate import validate_delivery
        with self.assertRaises(FileNotFoundError):
            validate_delivery("/nonexistent.mp4")

    def test_bad_spec_name(self):
        from opencut.core.delivery_validate import validate_delivery
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"\x00" * 100)
            tmp = f.name
        try:
            with self.assertRaises(ValueError):
                validate_delivery(tmp, spec_name="nonexistent_xyz_spec")
        finally:
            os.unlink(tmp)

    @patch("opencut.core.delivery_validate._sp.run")
    def test_validate_netflix_pass(self, mock_run):
        from opencut.core.delivery_validate import validate_delivery

        # Video probe
        video_probe = json.dumps({
            "streams": [{
                "codec_name": "h264", "width": 1920, "height": 1080,
                "r_frame_rate": "24/1", "bit_rate": "20000000",
                "pix_fmt": "yuv420p", "color_space": "bt709",
                "field_order": "progressive", "bits_per_raw_sample": "8",
                "profile": "High",
            }]
        }).encode()

        # Audio probe
        audio_probe = json.dumps({
            "streams": [{
                "codec_name": "aac", "channels": 2, "sample_rate": "48000",
                "bit_rate": "192000", "bits_per_raw_sample": "16",
                "channel_layout": "stereo",
            }]
        }).encode()

        # Container probe
        container_probe = json.dumps({
            "format": {
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "duration": "100.0", "size": "250000000",
                "bit_rate": "20192000", "nb_streams": "2",
                "tags": {"timecode": "01:00:00:00"},
            }
        }).encode()

        # Loudness measurement
        loudness_stderr = b'{"input_i": "-23.0", "input_tp": "-2.5", "input_lra": "10.0", "input_thresh": "-33.0"}'

        call_count = [0]
        def _side_effect(*args, **kwargs):
            call_count[0] += 1
            cmd = args[0] if args else kwargs.get("args", [])
            cmd_str = " ".join(str(c) for c in cmd)
            if "select_streams" in cmd_str and "v:0" in cmd_str:
                return MagicMock(returncode=0, stdout=video_probe, stderr=b"")
            elif "select_streams" in cmd_str and "a:0" in cmd_str:
                return MagicMock(returncode=0, stdout=audio_probe, stderr=b"")
            elif "select_streams" in cmd_str and " s" in cmd_str:
                return MagicMock(returncode=0, stdout=b'{"streams":[]}', stderr=b"")
            elif "format_name" in cmd_str or "format=" in cmd_str:
                return MagicMock(returncode=0, stdout=container_probe, stderr=b"")
            elif "loudnorm" in cmd_str:
                return MagicMock(returncode=0, stdout=b"", stderr=loudness_stderr)
            return MagicMock(returncode=0, stdout=b'{}', stderr=b"")

        mock_run.side_effect = _side_effect

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"\x00" * 100)
            tmp = f.name
        try:
            result = validate_delivery(tmp, spec_name="netflix",
                                       measure_loudness=True)
            self.assertIn("verdict", result)
            self.assertIn("total_checks", result)
            self.assertIn("all_results", result)
            self.assertIsInstance(result["all_results"], list)
        finally:
            os.unlink(tmp)

    @patch("opencut.core.delivery_validate._sp.run")
    def test_validate_youtube_no_loudness(self, mock_run):
        from opencut.core.delivery_validate import validate_delivery

        probe = json.dumps({
            "streams": [{
                "codec_name": "h264", "width": 1920, "height": 1080,
                "r_frame_rate": "30/1", "bit_rate": "10000000",
                "pix_fmt": "yuv420p",
            }]
        }).encode()

        mock_run.return_value = MagicMock(returncode=0, stdout=probe, stderr=b"")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"\x00" * 100)
            tmp = f.name
        try:
            result = validate_delivery(tmp, spec_name="youtube",
                                       measure_loudness=False)
            self.assertIn("verdict", result)
        finally:
            os.unlink(tmp)


class TestListDeliverySpecs(unittest.TestCase):
    """Tests for list_delivery_specs function."""

    def test_returns_list(self):
        from opencut.core.delivery_validate import list_delivery_specs
        specs = list_delivery_specs()
        self.assertIsInstance(specs, list)
        self.assertTrue(len(specs) >= 7)

    def test_entries_have_keys(self):
        from opencut.core.delivery_validate import list_delivery_specs
        specs = list_delivery_specs()
        for s in specs:
            self.assertIn("name", s)
            self.assertIn("display_name", s)
            self.assertIn("check_count", s)


# ===================================================================
# Feature 70.4 — Multi-Render
# ===================================================================


class TestRenderConfig(unittest.TestCase):
    """Tests for RenderConfig dataclass."""

    def test_defaults(self):
        from opencut.core.multi_render import RenderConfig
        rc = RenderConfig()
        self.assertEqual(rc.format, "mp4")
        self.assertEqual(rc.video_codec, "libx264")
        self.assertEqual(rc.audio_codec, "aac")
        self.assertEqual(rc.crf, 18)
        self.assertEqual(rc.preset, "medium")
        self.assertFalse(rc.two_pass)

    def test_to_dict(self):
        from opencut.core.multi_render import RenderConfig
        rc = RenderConfig(name="HD", width=1920, height=1080)
        d = rc.to_dict()
        self.assertEqual(d["name"], "HD")
        self.assertEqual(d["width"], 1920)

    def test_is_gpu_cpu_codec(self):
        from opencut.core.multi_render import RenderConfig
        rc = RenderConfig(video_codec="libx264")
        self.assertFalse(rc.is_gpu())

    def test_is_gpu_nvenc(self):
        from opencut.core.multi_render import RenderConfig
        rc = RenderConfig(video_codec="h264_nvenc")
        self.assertTrue(rc.is_gpu())


class TestRenderJob(unittest.TestCase):
    """Tests for RenderJob dataclass."""

    def test_defaults(self):
        from opencut.core.multi_render import RenderJob, RenderStatus
        rj = RenderJob()
        self.assertEqual(rj.status, RenderStatus.PENDING)
        self.assertEqual(rj.progress, 0.0)
        self.assertEqual(rj.resume_frame, 0)

    def test_to_dict(self):
        from opencut.core.multi_render import RenderJob
        rj = RenderJob(render_id="abc", config_name="test", progress=50.0)
        d = rj.to_dict()
        self.assertEqual(d["render_id"], "abc")
        self.assertEqual(d["progress"], 50.0)


class TestMultiRenderResult(unittest.TestCase):
    """Tests for MultiRenderResult dataclass."""

    def test_defaults(self):
        from opencut.core.multi_render import MultiRenderResult
        mr = MultiRenderResult()
        self.assertEqual(mr.renders, [])
        self.assertFalse(mr.all_succeeded)

    def test_to_dict(self):
        from opencut.core.multi_render import MultiRenderResult, RenderJob
        rj = RenderJob(render_id="x", config_name="test")
        mr = MultiRenderResult(total_renders=1, renders=[rj],
                               total_elapsed=5.678)
        d = mr.to_dict()
        self.assertEqual(d["total_elapsed"], 5.68)
        self.assertEqual(len(d["renders"]), 1)


class TestGPUCodecDetection(unittest.TestCase):
    """Tests for GPU codec detection."""

    def test_gpu_codecs(self):
        from opencut.core.multi_render import _is_gpu_codec
        self.assertTrue(_is_gpu_codec("h264_nvenc"))
        self.assertTrue(_is_gpu_codec("hevc_nvenc"))
        self.assertTrue(_is_gpu_codec("h264_amf"))
        self.assertTrue(_is_gpu_codec("av1_nvenc"))

    def test_cpu_codecs(self):
        from opencut.core.multi_render import _is_gpu_codec
        self.assertFalse(_is_gpu_codec("libx264"))
        self.assertFalse(_is_gpu_codec("libx265"))
        self.assertFalse(_is_gpu_codec("libaom-av1"))


class TestRenderCancel(unittest.TestCase):
    """Tests for render cancellation."""

    def test_cancel_nonexistent(self):
        from opencut.core.multi_render import cancel_render
        result = cancel_render("nonexistent-render-id")
        self.assertTrue(result)  # registers cancellation

    def test_cancel_registered(self):
        from opencut.core.multi_render import (
            _cancelled_renders, _cancel_lock, cancel_render, _is_cancelled,
        )
        rid = "test-cancel-123"
        cancel_render(rid)
        self.assertTrue(_is_cancelled(rid))
        # Cleanup
        with _cancel_lock:
            _cancelled_renders.pop(rid, None)


class TestMultiRender(unittest.TestCase):
    """Tests for multi_render with mocked subprocess."""

    def test_file_not_found(self):
        from opencut.core.multi_render import RenderConfig, multi_render
        with self.assertRaises(FileNotFoundError):
            multi_render("/nonexistent.mp4", [RenderConfig()])

    def test_empty_configs(self):
        from opencut.core.multi_render import multi_render
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"\x00" * 100)
            tmp = f.name
        try:
            with self.assertRaises(ValueError):
                multi_render(tmp, [])
        finally:
            os.unlink(tmp)

    @patch("opencut.core.multi_render._sp.run")
    @patch("opencut.core.multi_render._sp.Popen")
    def test_multi_render_sequential(self, mock_popen, mock_run):
        from opencut.core.multi_render import RenderConfig, multi_render

        # Mock ffprobe for get_video_info
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({
                "streams": [{"width": 1920, "height": 1080,
                             "r_frame_rate": "30/1", "duration": "10.0"}],
                "format": {"duration": "10.0"},
            }).encode(),
            stderr=b"",
        )

        # Mock Popen for FFmpeg encoding
        mock_proc = MagicMock()
        mock_proc.poll.side_effect = [None, 0]
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read.return_value = b"frame=  300 fps=30\n"
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.read.return_value = b""
        mock_proc.communicate.return_value = (b"", b"")
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"\x00" * 100)
            tmp = f.name

        out_dir = tempfile.mkdtemp()
        try:
            configs = [
                RenderConfig(name="HD", format="mp4", video_codec="libx264",
                             width=1920, height=1080, output_dir=out_dir),
                RenderConfig(name="SD", format="mp4", video_codec="libx264",
                             width=1280, height=720, output_dir=out_dir),
            ]
            result = multi_render(tmp, configs, parallel=False)
            self.assertEqual(result["total_renders"], 2)
            self.assertIn("renders", result)
            self.assertIn("all_succeeded", result)
        finally:
            os.unlink(tmp)
            shutil.rmtree(out_dir, ignore_errors=True)


class TestBuildFFmpegCmd(unittest.TestCase):
    """Tests for _build_ffmpeg_cmd helper."""

    def test_basic_cmd(self):
        from opencut.core.multi_render import RenderConfig, _build_ffmpeg_cmd
        rc = RenderConfig(
            video_codec="libx264", audio_codec="aac",
            crf=20, preset="fast",
        )
        source = {"fps": 30, "width": 1920, "height": 1080}
        cmd = _build_ffmpeg_cmd("/input.mp4", rc, "/output.mp4", source)
        self.assertIn("-c:v", cmd)
        self.assertIn("libx264", cmd)
        self.assertIn("-crf", cmd)
        self.assertIn("20", cmd)

    def test_two_pass_cmd(self):
        from opencut.core.multi_render import RenderConfig, _build_ffmpeg_cmd
        rc = RenderConfig(
            video_codec="libx264", two_pass=True,
            video_bitrate="10M",
        )
        source = {"fps": 30, "width": 1920, "height": 1080}
        cmd_p1 = _build_ffmpeg_cmd("/in.mp4", rc, "/out.mp4", source, pass_num=1)
        self.assertIn("-pass", cmd_p1)
        self.assertIn("1", cmd_p1)
        self.assertIn("-an", cmd_p1)

    def test_scale_filter(self):
        from opencut.core.multi_render import RenderConfig, _build_ffmpeg_cmd
        rc = RenderConfig(width=1280, height=720)
        source = {"fps": 30, "width": 1920, "height": 1080}
        cmd = _build_ffmpeg_cmd("/in.mp4", rc, "/out.mp4", source)
        vf_idx = cmd.index("-vf") + 1
        self.assertIn("scale=1280:720", cmd[vf_idx])

    def test_resume_seek(self):
        from opencut.core.multi_render import RenderConfig, _build_ffmpeg_cmd
        rc = RenderConfig()
        source = {"fps": 30, "width": 1920, "height": 1080}
        cmd = _build_ffmpeg_cmd("/in.mp4", rc, "/out.mp4", source,
                                resume_frame=900)
        self.assertIn("-ss", cmd)


class TestGetActiveRenders(unittest.TestCase):
    """Tests for utility functions."""

    def setUp(self):
        """Clear global state before each test."""
        from opencut.core.multi_render import (
            _active_processes, _cancelled_renders, _cancel_lock,
        )
        with _cancel_lock:
            _active_processes.clear()
            _cancelled_renders.clear()

    def test_get_active_returns_list(self):
        from opencut.core.multi_render import get_active_renders
        result = get_active_renders()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)

    def test_cancel_all_returns_count(self):
        from opencut.core.multi_render import cancel_all_renders
        count = cancel_all_renders()
        self.assertIsInstance(count, int)
        self.assertEqual(count, 0)


# ===================================================================
# Route Smoke Tests
# ===================================================================


class TestDeliveryMasterRoutes(unittest.TestCase):
    """Smoke tests for delivery_master_bp routes."""

    @classmethod
    def setUpClass(cls):
        """Create a minimal Flask app with the delivery_master blueprint."""
        from flask import Flask
        from opencut.routes.delivery_master_routes import delivery_master_bp
        cls.app = Flask(__name__)
        cls.app.config["TESTING"] = True
        cls.app.register_blueprint(delivery_master_bp)
        cls.client = cls.app.test_client()

    def test_specs_list(self):
        """GET /delivery/specs should return spec list."""
        resp = self.client.get("/delivery/specs")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("specs", data)
        self.assertIn("count", data)
        self.assertTrue(data["count"] > 0)

    def test_specs_get_netflix(self):
        """GET /delivery/specs/netflix should return Netflix spec."""
        resp = self.client.get("/delivery/specs/netflix")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("spec", data)
        self.assertEqual(data["spec"]["name"], "netflix")

    def test_specs_get_nonexistent(self):
        """GET /delivery/specs/xyz should return 404."""
        resp = self.client.get("/delivery/specs/xyz_does_not_exist")
        self.assertEqual(resp.status_code, 404)

    def test_specs_create_no_csrf(self):
        """POST /delivery/specs without CSRF should fail."""
        resp = self.client.post("/delivery/specs",
                                json={"name": "test_route_spec"})
        # Expect 403 from require_csrf
        self.assertIn(resp.status_code, [400, 403])

    def test_render_cancel_no_csrf(self):
        """POST /render/multi/cancel without CSRF should fail."""
        resp = self.client.post("/render/multi/cancel",
                                json={"render_id": "test"})
        self.assertIn(resp.status_code, [400, 403])

    def test_validate_no_csrf(self):
        """POST /delivery/validate without CSRF should fail."""
        resp = self.client.post("/delivery/validate",
                                json={"filepath": "/test.mp4"})
        self.assertIn(resp.status_code, [400, 403])

    def test_export_dcp_no_csrf(self):
        """POST /export/dcp without CSRF should fail."""
        resp = self.client.post("/export/dcp",
                                json={"filepath": "/test.mp4"})
        self.assertIn(resp.status_code, [400, 403])

    def test_export_imf_no_csrf(self):
        """POST /export/imf without CSRF should fail."""
        resp = self.client.post("/export/imf",
                                json={"filepath": "/test.mp4"})
        self.assertIn(resp.status_code, [400, 403])

    def test_render_multi_no_csrf(self):
        """POST /render/multi without CSRF should fail."""
        resp = self.client.post("/render/multi",
                                json={"filepath": "/test.mp4"})
        self.assertIn(resp.status_code, [400, 403])

    def test_specs_compare_no_csrf(self):
        """POST /delivery/specs/compare without CSRF should fail."""
        resp = self.client.post("/delivery/specs/compare",
                                json={"spec_a": "netflix", "spec_b": "youtube"})
        self.assertIn(resp.status_code, [400, 403])


class TestSpecDiffDataclass(unittest.TestCase):
    """Tests for SpecDiff and SpecCompareResult dataclasses."""

    def test_spec_diff_defaults(self):
        from opencut.core.delivery_spec import SpecDiff
        sd = SpecDiff()
        self.assertEqual(sd.field_path, "")
        self.assertEqual(sd.diff_type, "")

    def test_spec_diff_to_dict(self):
        from opencut.core.delivery_spec import SpecDiff
        sd = SpecDiff(field_path="video.codec", diff_type="changed",
                      spec_a_value="h264", spec_b_value="h265")
        d = sd.to_dict()
        self.assertEqual(d["field_path"], "video.codec")
        self.assertEqual(d["diff_type"], "changed")

    def test_compare_result_defaults(self):
        from opencut.core.delivery_spec import SpecCompareResult
        cr = SpecCompareResult()
        self.assertEqual(cr.total_differences, 0)
        self.assertEqual(cr.added, [])
        self.assertEqual(cr.removed, [])

    def test_compare_result_to_dict(self):
        from opencut.core.delivery_spec import SpecCompareResult, SpecDiff
        diff = SpecDiff(field_path="test", diff_type="added")
        cr = SpecCompareResult(spec_a_name="a", spec_b_name="b",
                               added=[diff], total_differences=1)
        d = cr.to_dict()
        self.assertEqual(d["total_differences"], 1)
        self.assertEqual(len(d["added"]), 1)


class TestDeliverySpecDataclass(unittest.TestCase):
    """Tests for DeliverySpec in delivery_validate module."""

    def test_defaults(self):
        from opencut.core.delivery_validate import DeliverySpec
        ds = DeliverySpec()
        self.assertEqual(ds.name, "")
        self.assertEqual(ds.checks, [])

    def test_to_dict(self):
        from opencut.core.delivery_validate import DeliverySpec
        ds = DeliverySpec(name="test", display_name="Test Spec")
        d = ds.to_dict()
        self.assertEqual(d["name"], "test")
        self.assertEqual(d["display_name"], "Test Spec")


class TestNormalizeFormat(unittest.TestCase):
    """Tests for _normalize_format helper."""

    def test_mov_container(self):
        from opencut.core.delivery_validate import _normalize_format
        self.assertEqual(_normalize_format("mov,mp4,m4a,3gp,3g2,mj2"), "mov")

    def test_mxf(self):
        from opencut.core.delivery_validate import _normalize_format
        self.assertEqual(_normalize_format("mxf"), "mxf")

    def test_matroska(self):
        from opencut.core.delivery_validate import _normalize_format
        self.assertEqual(_normalize_format("matroska,webm"), "mkv")

    def test_simple_mp4(self):
        from opencut.core.delivery_validate import _normalize_format
        self.assertEqual(_normalize_format("mp4"), "mp4")


class TestRenderStatusConstants(unittest.TestCase):
    """Tests for RenderStatus constants."""

    def test_constants(self):
        from opencut.core.multi_render import RenderStatus
        self.assertEqual(RenderStatus.PENDING, "pending")
        self.assertEqual(RenderStatus.RUNNING, "running")
        self.assertEqual(RenderStatus.COMPLETE, "complete")
        self.assertEqual(RenderStatus.FAILED, "failed")
        self.assertEqual(RenderStatus.CANCELLED, "cancelled")


class TestIMFAsset(unittest.TestCase):
    """Tests for IMFAsset dataclass."""

    def test_defaults(self):
        from opencut.core.imf_package import IMFAsset
        a = IMFAsset()
        self.assertEqual(a.uuid, "")
        self.assertEqual(a.original_filename, "")

    def test_to_dict(self):
        from opencut.core.imf_package import IMFAsset
        a = IMFAsset(uuid="urn:uuid:abc", filename="video.mxf",
                     asset_type="video_mxf", file_size=5000)
        d = a.to_dict()
        self.assertEqual(d["file_size"], 5000)


class TestIMFValidateConstraints(unittest.TestCase):
    """Tests for _validate_imf_constraints."""

    def test_no_warnings_when_within_bounds(self):
        from opencut.core.imf_package import IMFConfig, _validate_imf_constraints
        config = IMFConfig(profile="application_2", width=1920, height=1080)
        source = {"width": 1920, "height": 1080, "fps": 24.0}
        warnings = _validate_imf_constraints(config, source)
        self.assertEqual(warnings, [])

    def test_warns_on_oversized_source(self):
        from opencut.core.imf_package import IMFConfig, _validate_imf_constraints
        config = IMFConfig(profile="application_2")
        source = {"width": 4096, "height": 2160, "fps": 24.0}
        warnings = _validate_imf_constraints(config, source)
        self.assertTrue(len(warnings) > 0)


class TestHardwareDetection(unittest.TestCase):
    """Tests for _detect_hw_acceleration."""

    @patch("opencut.core.multi_render._sp.run")
    def test_detect_nvenc(self, mock_run):
        from opencut.core.multi_render import _detect_hw_acceleration
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=b"V..... h264_nvenc NVIDIA encoder\n",
            stderr=b"",
        )
        hw = _detect_hw_acceleration()
        self.assertTrue(hw["nvenc"])
        self.assertFalse(hw["amf"])

    @patch("opencut.core.multi_render._sp.run")
    def test_detect_nothing(self, mock_run):
        from opencut.core.multi_render import _detect_hw_acceleration
        mock_run.return_value = MagicMock(
            returncode=0, stdout=b"V..... libx264\n", stderr=b"",
        )
        hw = _detect_hw_acceleration()
        self.assertFalse(hw["nvenc"])
        self.assertFalse(hw["amf"])


if __name__ == "__main__":
    unittest.main()
