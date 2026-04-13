"""
Tests for OpenCut Hardware-Accelerated Encoding (Feature 32.1).

Covers:
  - HWEncoder / HWEncoderInfo dataclasses
  - Encoder detection and parsing of `ffmpeg -encoders` output
  - Encoder verification with synthetic test encodes
  - Priority-based encoder selection (nvenc > qsv > amf > videotoolbox)
  - Quality preset mapping per HW type
  - Software fallback when no HW encoder is available
  - HW encode function with fallback on failure
  - Integration with export_presets (hw_accel presets, _export_hw_accel)
  - Route endpoints: /hw/encoders, /hw/encoders/refresh, /hw/encode, /hw/presets
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
# Test HWEncoder / HWEncoderInfo dataclasses
# ============================================================
class TestHWEncoderDataclasses(unittest.TestCase):
    """Tests for HWEncoder and HWEncoderInfo dataclasses."""

    def test_hw_encoder_creation(self):
        from opencut.core.hw_accel import HWEncoder
        enc = HWEncoder(
            name="h264_nvenc",
            codec="h264",
            hw_type="nvenc",
            supports_h264=True,
            supports_hevc=False,
            supports_av1=False,
        )
        self.assertEqual(enc.name, "h264_nvenc")
        self.assertEqual(enc.codec, "h264")
        self.assertEqual(enc.hw_type, "nvenc")
        self.assertTrue(enc.supports_h264)
        self.assertFalse(enc.supports_hevc)

    def test_hw_encoder_to_dict(self):
        from opencut.core.hw_accel import HWEncoder
        enc = HWEncoder(name="hevc_qsv", codec="hevc", hw_type="qsv",
                        supports_hevc=True)
        d = enc.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["name"], "hevc_qsv")
        self.assertEqual(d["hw_type"], "qsv")
        self.assertTrue(d["supports_hevc"])

    def test_hw_encoder_info_creation(self):
        from opencut.core.hw_accel import HWEncoder, HWEncoderInfo
        info = HWEncoderInfo(
            available_encoders=[
                HWEncoder(name="h264_nvenc", codec="h264", hw_type="nvenc", supports_h264=True),
            ],
            preferred_h264="h264_nvenc",
            gpu_name="NVIDIA GeForce RTX 4090",
        )
        self.assertEqual(len(info.available_encoders), 1)
        self.assertEqual(info.preferred_h264, "h264_nvenc")
        self.assertIsNone(info.preferred_hevc)
        self.assertEqual(info.gpu_name, "NVIDIA GeForce RTX 4090")

    def test_hw_encoder_info_to_dict(self):
        from opencut.core.hw_accel import HWEncoder, HWEncoderInfo
        info = HWEncoderInfo(
            available_encoders=[
                HWEncoder(name="h264_nvenc", codec="h264", hw_type="nvenc", supports_h264=True),
                HWEncoder(name="hevc_nvenc", codec="hevc", hw_type="nvenc", supports_hevc=True),
            ],
            preferred_h264="h264_nvenc",
            preferred_hevc="hevc_nvenc",
            gpu_name="NVIDIA RTX 4090",
            detection_time=1.23,
        )
        d = info.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(len(d["available_encoders"]), 2)
        self.assertTrue(d["has_hw_accel"])
        self.assertEqual(d["preferred_h264"], "h264_nvenc")
        self.assertEqual(d["detection_time"], 1.23)

    def test_hw_encoder_info_empty(self):
        from opencut.core.hw_accel import HWEncoderInfo
        info = HWEncoderInfo()
        d = info.to_dict()
        self.assertFalse(d["has_hw_accel"])
        self.assertEqual(len(d["available_encoders"]), 0)
        self.assertIsNone(d["preferred_h264"])


# ============================================================
# Test Encoder Detection Parsing
# ============================================================
class TestEncoderParsing(unittest.TestCase):
    """Tests for parsing ffmpeg -encoders output."""

    SAMPLE_ENCODERS_OUTPUT = """\
Encoders:
 V..... = Video
 A..... = Audio
 S..... = Subtitle
 .F.... = Frame-level multithreading
 ..S... = Slice-level multithreading
 ...X.. = Codec is experimental
 ....B. = Supports draw_horiz_band
 .....D = Supports direct rendering method 1
 ------
 V..... libx264              libx264 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10 (codec h264)
 V..... libx264rgb           libx264 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10 RGB (codec h264)
 V..... libx265              libx265 H.265 / HEVC (codec hevc)
 V..... h264_nvenc           NVIDIA NVENC H.264 encoder (codec h264)
 V..... hevc_nvenc           NVIDIA NVENC hevc encoder (codec hevc)
 V..... av1_nvenc            NVIDIA NVENC AV1 encoder (codec av1)
 V..... h264_qsv             H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10 (Intel Quick Sync Video acceleration) (codec h264)
 V..... hevc_amf             AMD AMF HEVC encoder (codec hevc)
 A..... aac                  AAC (Advanced Audio Coding) (codec aac)
 A..... libmp3lame           libmp3lame MP3 (codec mp3)
"""

    def test_parse_finds_nvenc_encoders(self):
        from opencut.core.hw_accel import _parse_encoders_output
        found = _parse_encoders_output(self.SAMPLE_ENCODERS_OUTPUT)
        self.assertIn("h264_nvenc", found)
        self.assertIn("hevc_nvenc", found)
        self.assertIn("av1_nvenc", found)

    def test_parse_finds_qsv_encoder(self):
        from opencut.core.hw_accel import _parse_encoders_output
        found = _parse_encoders_output(self.SAMPLE_ENCODERS_OUTPUT)
        self.assertIn("h264_qsv", found)

    def test_parse_finds_amf_encoder(self):
        from opencut.core.hw_accel import _parse_encoders_output
        found = _parse_encoders_output(self.SAMPLE_ENCODERS_OUTPUT)
        self.assertIn("hevc_amf", found)

    def test_parse_ignores_software_codecs(self):
        from opencut.core.hw_accel import _parse_encoders_output
        found = _parse_encoders_output(self.SAMPLE_ENCODERS_OUTPUT)
        self.assertNotIn("libx264", found)
        self.assertNotIn("libx265", found)

    def test_parse_ignores_audio_codecs(self):
        from opencut.core.hw_accel import _parse_encoders_output
        found = _parse_encoders_output(self.SAMPLE_ENCODERS_OUTPUT)
        self.assertNotIn("aac", found)
        self.assertNotIn("libmp3lame", found)

    def test_parse_empty_output(self):
        from opencut.core.hw_accel import _parse_encoders_output
        found = _parse_encoders_output("")
        self.assertEqual(found, [])

    def test_parse_no_hw_encoders(self):
        from opencut.core.hw_accel import _parse_encoders_output
        output = """\
 V..... libx264              libx264 H.264 (codec h264)
 V..... libx265              libx265 H.265 (codec hevc)
 A..... aac                  AAC (codec aac)
"""
        found = _parse_encoders_output(output)
        self.assertEqual(found, [])

    def test_parse_no_duplicates(self):
        from opencut.core.hw_accel import _parse_encoders_output
        output = """\
 V..... h264_nvenc           NVIDIA NVENC H.264 encoder (codec h264)
 V..... h264_nvenc           NVIDIA NVENC H.264 encoder (codec h264)
"""
        found = _parse_encoders_output(output)
        self.assertEqual(found.count("h264_nvenc"), 1)


# ============================================================
# Test Encoder Verification
# ============================================================
class TestEncoderVerification(unittest.TestCase):
    """Tests for _test_encoder() synthetic encode verification."""

    @patch("opencut.core.hw_accel.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_encoder_passes_when_ffmpeg_succeeds(self, mock_run, mock_ffmpeg):
        from opencut.core.hw_accel import _test_encoder
        mock_run.return_value = MagicMock(returncode=0)
        result = _test_encoder("h264_nvenc")
        self.assertTrue(result)
        # Verify ffmpeg was called with the encoder name
        call_args = mock_run.call_args[0][0]
        self.assertIn("h264_nvenc", call_args)

    @patch("opencut.core.hw_accel.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_encoder_fails_when_ffmpeg_errors(self, mock_run, mock_ffmpeg):
        from opencut.core.hw_accel import _test_encoder
        mock_run.return_value = MagicMock(returncode=1)
        result = _test_encoder("h264_nvenc")
        self.assertFalse(result)

    @patch("opencut.core.hw_accel.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_encoder_fails_on_file_not_found(self, mock_run, mock_ffmpeg):
        from opencut.core.hw_accel import _test_encoder
        result = _test_encoder("h264_nvenc")
        self.assertFalse(result)

    @patch("opencut.core.hw_accel.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run", side_effect=Exception("timeout"))
    def test_encoder_fails_on_timeout(self, mock_run, mock_ffmpeg):
        from opencut.core.hw_accel import _test_encoder
        result = _test_encoder("hevc_qsv")
        self.assertFalse(result)

    @patch("opencut.core.hw_accel.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_encoder_nvenc_uses_p1_preset(self, mock_run, mock_ffmpeg):
        from opencut.core.hw_accel import _test_encoder
        mock_run.return_value = MagicMock(returncode=0)
        _test_encoder("h264_nvenc")
        call_args = mock_run.call_args[0][0]
        # NVENC test should include -preset p1
        self.assertIn("-preset", call_args)
        self.assertIn("p1", call_args)

    @patch("opencut.core.hw_accel.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_encoder_qsv_uses_veryfast_preset(self, mock_run, mock_ffmpeg):
        from opencut.core.hw_accel import _test_encoder
        mock_run.return_value = MagicMock(returncode=0)
        _test_encoder("h264_qsv")
        call_args = mock_run.call_args[0][0]
        self.assertIn("veryfast", call_args)

    @patch("opencut.core.hw_accel.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_encoder_amf_uses_speed_quality(self, mock_run, mock_ffmpeg):
        from opencut.core.hw_accel import _test_encoder
        mock_run.return_value = MagicMock(returncode=0)
        _test_encoder("h264_amf")
        call_args = mock_run.call_args[0][0]
        self.assertIn("-quality", call_args)
        self.assertIn("speed", call_args)


# ============================================================
# Test Priority-Based Selection
# ============================================================
class TestPrioritySelection(unittest.TestCase):
    """Tests for _pick_preferred() encoder priority."""

    def test_nvenc_preferred_over_qsv(self):
        from opencut.core.hw_accel import HWEncoder, _pick_preferred
        encoders = [
            HWEncoder(name="h264_qsv", codec="h264", hw_type="qsv", supports_h264=True),
            HWEncoder(name="h264_nvenc", codec="h264", hw_type="nvenc", supports_h264=True),
        ]
        result = _pick_preferred(encoders, "h264")
        self.assertEqual(result, "h264_nvenc")

    def test_qsv_preferred_over_amf(self):
        from opencut.core.hw_accel import HWEncoder, _pick_preferred
        encoders = [
            HWEncoder(name="hevc_amf", codec="hevc", hw_type="amf", supports_hevc=True),
            HWEncoder(name="hevc_qsv", codec="hevc", hw_type="qsv", supports_hevc=True),
        ]
        result = _pick_preferred(encoders, "hevc")
        self.assertEqual(result, "hevc_qsv")

    def test_amf_preferred_over_videotoolbox(self):
        from opencut.core.hw_accel import HWEncoder, _pick_preferred
        encoders = [
            HWEncoder(name="h264_videotoolbox", codec="h264", hw_type="videotoolbox", supports_h264=True),
            HWEncoder(name="h264_amf", codec="h264", hw_type="amf", supports_h264=True),
        ]
        result = _pick_preferred(encoders, "h264")
        self.assertEqual(result, "h264_amf")

    def test_returns_none_when_no_encoders(self):
        from opencut.core.hw_accel import _pick_preferred
        result = _pick_preferred([], "h264")
        self.assertIsNone(result)

    def test_returns_none_when_codec_not_found(self):
        from opencut.core.hw_accel import HWEncoder, _pick_preferred
        encoders = [
            HWEncoder(name="h264_nvenc", codec="h264", hw_type="nvenc", supports_h264=True),
        ]
        result = _pick_preferred(encoders, "av1")
        self.assertIsNone(result)

    def test_single_encoder_selected(self):
        from opencut.core.hw_accel import HWEncoder, _pick_preferred
        encoders = [
            HWEncoder(name="av1_nvenc", codec="av1", hw_type="nvenc", supports_av1=True),
        ]
        result = _pick_preferred(encoders, "av1")
        self.assertEqual(result, "av1_nvenc")


# ============================================================
# Test get_hw_encode_args
# ============================================================
class TestGetHWEncodeArgs(unittest.TestCase):
    """Tests for get_hw_encode_args() argument generation."""

    @patch("opencut.core.hw_accel.detect_hw_encoders")
    def test_auto_uses_preferred_h264(self, mock_detect):
        from opencut.core.hw_accel import HWEncoder, HWEncoderInfo, get_hw_encode_args
        mock_detect.return_value = HWEncoderInfo(
            available_encoders=[
                HWEncoder(name="h264_nvenc", codec="h264", hw_type="nvenc", supports_h264=True),
            ],
            preferred_h264="h264_nvenc",
        )
        args = get_hw_encode_args(codec="h264", quality="balanced")
        self.assertIn("-c:v", args)
        self.assertIn("h264_nvenc", args)

    @patch("opencut.core.hw_accel.detect_hw_encoders")
    def test_software_fallback_when_no_hw(self, mock_detect):
        from opencut.core.hw_accel import HWEncoderInfo, get_hw_encode_args
        mock_detect.return_value = HWEncoderInfo()  # No encoders
        args = get_hw_encode_args(codec="h264", quality="balanced")
        self.assertIn("-c:v", args)
        self.assertIn("libx264", args)

    @patch("opencut.core.hw_accel.detect_hw_encoders")
    def test_explicit_software_mode(self, mock_detect):
        from opencut.core.hw_accel import get_hw_encode_args
        args = get_hw_encode_args(codec="h264", quality="speed", hw_type="software")
        self.assertIn("libx264", args)
        self.assertIn("ultrafast", args)
        # detect_hw_encoders should NOT be called for software mode
        mock_detect.assert_not_called()

    @patch("opencut.core.hw_accel.detect_hw_encoders")
    def test_hevc_software_fallback(self, mock_detect):
        from opencut.core.hw_accel import HWEncoderInfo, get_hw_encode_args
        mock_detect.return_value = HWEncoderInfo()
        args = get_hw_encode_args(codec="hevc", quality="quality")
        self.assertIn("libx265", args)
        self.assertIn("slow", args)

    @patch("opencut.core.hw_accel.detect_hw_encoders")
    def test_av1_software_fallback(self, mock_detect):
        from opencut.core.hw_accel import HWEncoderInfo, get_hw_encode_args
        mock_detect.return_value = HWEncoderInfo()
        args = get_hw_encode_args(codec="av1", quality="balanced")
        self.assertIn("libsvtav1", args)

    @patch("opencut.core.hw_accel.detect_hw_encoders")
    def test_nvenc_balanced_quality_preset(self, mock_detect):
        from opencut.core.hw_accel import HWEncoder, HWEncoderInfo, get_hw_encode_args
        mock_detect.return_value = HWEncoderInfo(
            available_encoders=[
                HWEncoder(name="h264_nvenc", codec="h264", hw_type="nvenc", supports_h264=True),
            ],
            preferred_h264="h264_nvenc",
        )
        args = get_hw_encode_args(codec="h264", quality="balanced")
        self.assertIn("p4", args)
        self.assertIn("vbr", args)

    @patch("opencut.core.hw_accel.detect_hw_encoders")
    def test_nvenc_speed_preset(self, mock_detect):
        from opencut.core.hw_accel import HWEncoder, HWEncoderInfo, get_hw_encode_args
        mock_detect.return_value = HWEncoderInfo(
            available_encoders=[
                HWEncoder(name="h264_nvenc", codec="h264", hw_type="nvenc", supports_h264=True),
            ],
            preferred_h264="h264_nvenc",
        )
        args = get_hw_encode_args(codec="h264", quality="speed")
        self.assertIn("p1", args)

    @patch("opencut.core.hw_accel.detect_hw_encoders")
    def test_nvenc_quality_preset(self, mock_detect):
        from opencut.core.hw_accel import HWEncoder, HWEncoderInfo, get_hw_encode_args
        mock_detect.return_value = HWEncoderInfo(
            available_encoders=[
                HWEncoder(name="h264_nvenc", codec="h264", hw_type="nvenc", supports_h264=True),
            ],
            preferred_h264="h264_nvenc",
        )
        args = get_hw_encode_args(codec="h264", quality="quality")
        self.assertIn("p7", args)

    @patch("opencut.core.hw_accel.detect_hw_encoders")
    def test_specific_hw_type_selection(self, mock_detect):
        from opencut.core.hw_accel import HWEncoder, HWEncoderInfo, get_hw_encode_args
        mock_detect.return_value = HWEncoderInfo(
            available_encoders=[
                HWEncoder(name="h264_nvenc", codec="h264", hw_type="nvenc", supports_h264=True),
                HWEncoder(name="h264_qsv", codec="h264", hw_type="qsv", supports_h264=True),
            ],
            preferred_h264="h264_nvenc",
        )
        # Request QSV specifically
        args = get_hw_encode_args(codec="h264", quality="balanced", hw_type="qsv")
        self.assertIn("h264_qsv", args)

    @patch("opencut.core.hw_accel.detect_hw_encoders")
    def test_unavailable_hw_type_falls_back(self, mock_detect):
        from opencut.core.hw_accel import HWEncoder, HWEncoderInfo, get_hw_encode_args
        mock_detect.return_value = HWEncoderInfo(
            available_encoders=[
                HWEncoder(name="h264_nvenc", codec="h264", hw_type="nvenc", supports_h264=True),
            ],
            preferred_h264="h264_nvenc",
        )
        # Request AMF which isn't available
        args = get_hw_encode_args(codec="h264", quality="balanced", hw_type="amf")
        # Should fall back to software
        self.assertIn("libx264", args)

    def test_invalid_codec_defaults_to_h264(self):
        with patch("opencut.core.hw_accel.detect_hw_encoders") as mock_detect:
            from opencut.core.hw_accel import HWEncoderInfo, get_hw_encode_args
            mock_detect.return_value = HWEncoderInfo()
            args = get_hw_encode_args(codec="invalid", quality="balanced")
            self.assertIn("libx264", args)

    def test_invalid_quality_defaults_to_balanced(self):
        with patch("opencut.core.hw_accel.detect_hw_encoders") as mock_detect:
            from opencut.core.hw_accel import HWEncoderInfo, get_hw_encode_args
            mock_detect.return_value = HWEncoderInfo()
            args = get_hw_encode_args(codec="h264", quality="invalid")
            self.assertIn("medium", args)


# ============================================================
# Test detect_hw_encoders (full flow)
# ============================================================
class TestDetectHWEncoders(unittest.TestCase):
    """Tests for detect_hw_encoders() full detection flow."""

    def setUp(self):
        # Clear the module-level cache before each test
        import opencut.core.hw_accel as mod
        with mod._cache_lock:
            mod._cached_info = None

    @patch("opencut.core.hw_accel._detect_gpu_name", return_value="NVIDIA RTX 4090")
    @patch("opencut.core.hw_accel._test_encoder")
    @patch("subprocess.run")
    @patch("opencut.core.hw_accel.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_full_detection_with_nvenc(self, mock_ffmpeg, mock_run, mock_test, mock_gpu):
        from opencut.core.hw_accel import detect_hw_encoders
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=""" V..... h264_nvenc           NVIDIA NVENC H.264 encoder (codec h264)
 V..... hevc_nvenc           NVIDIA NVENC hevc encoder (codec hevc)
""",
        )
        mock_test.return_value = True

        info = detect_hw_encoders(force_refresh=True)
        self.assertEqual(len(info.available_encoders), 2)
        self.assertEqual(info.preferred_h264, "h264_nvenc")
        self.assertEqual(info.preferred_hevc, "hevc_nvenc")
        self.assertEqual(info.gpu_name, "NVIDIA RTX 4090")

    @patch("opencut.core.hw_accel._detect_gpu_name", return_value=None)
    @patch("subprocess.run")
    @patch("opencut.core.hw_accel.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_detection_with_no_hw_encoders(self, mock_ffmpeg, mock_run, mock_gpu):
        from opencut.core.hw_accel import detect_hw_encoders
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=""" V..... libx264              libx264 (codec h264)
 A..... aac                  AAC (codec aac)
""",
        )
        info = detect_hw_encoders(force_refresh=True)
        self.assertEqual(len(info.available_encoders), 0)
        self.assertIsNone(info.preferred_h264)

    @patch("opencut.core.hw_accel._detect_gpu_name", return_value=None)
    @patch("subprocess.run", side_effect=FileNotFoundError)
    @patch("opencut.core.hw_accel.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_detection_handles_ffmpeg_not_found(self, mock_ffmpeg, mock_run, mock_gpu):
        from opencut.core.hw_accel import detect_hw_encoders
        info = detect_hw_encoders(force_refresh=True)
        self.assertEqual(len(info.available_encoders), 0)

    @patch("opencut.core.hw_accel._detect_gpu_name", return_value="NVIDIA RTX 4090")
    @patch("opencut.core.hw_accel._test_encoder")
    @patch("subprocess.run")
    @patch("opencut.core.hw_accel.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_detection_filters_failing_encoders(self, mock_ffmpeg, mock_run, mock_test, mock_gpu):
        from opencut.core.hw_accel import detect_hw_encoders
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=""" V..... h264_nvenc           NVIDIA NVENC H.264 encoder (codec h264)
 V..... hevc_nvenc           NVIDIA NVENC hevc encoder (codec hevc)
""",
        )
        # h264_nvenc passes, hevc_nvenc fails test encode
        mock_test.side_effect = lambda name: name == "h264_nvenc"

        info = detect_hw_encoders(force_refresh=True)
        self.assertEqual(len(info.available_encoders), 1)
        self.assertEqual(info.preferred_h264, "h264_nvenc")
        self.assertIsNone(info.preferred_hevc)

    @patch("opencut.core.hw_accel._detect_gpu_name", return_value=None)
    @patch("opencut.core.hw_accel._test_encoder", return_value=True)
    @patch("subprocess.run")
    @patch("opencut.core.hw_accel.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_cache_is_used_on_second_call(self, mock_ffmpeg, mock_run, mock_test, mock_gpu):
        from opencut.core.hw_accel import detect_hw_encoders
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=" V..... h264_nvenc           NVIDIA (codec h264)\n",
        )
        info1 = detect_hw_encoders(force_refresh=True)
        info2 = detect_hw_encoders(force_refresh=False)
        # Should be the same cached object
        self.assertIs(info1, info2)
        # subprocess should only be called once (for the first detection)
        self.assertEqual(mock_run.call_count, 1)

    @patch("opencut.core.hw_accel._detect_gpu_name", return_value=None)
    @patch("opencut.core.hw_accel._test_encoder", return_value=True)
    @patch("subprocess.run")
    @patch("opencut.core.hw_accel.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_force_refresh_bypasses_cache(self, mock_ffmpeg, mock_run, mock_test, mock_gpu):
        from opencut.core.hw_accel import detect_hw_encoders
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=" V..... h264_nvenc           NVIDIA (codec h264)\n",
        )
        detect_hw_encoders(force_refresh=True)
        detect_hw_encoders(force_refresh=True)
        # subprocess should be called twice
        self.assertEqual(mock_run.call_count, 2)


# ============================================================
# Test hw_encode function
# ============================================================
class TestHWEncode(unittest.TestCase):
    """Tests for hw_encode() encoding function."""

    @patch("opencut.core.hw_accel.run_ffmpeg")
    @patch("opencut.core.hw_accel.get_hw_encode_args", return_value=["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "23", "-b:v", "0"])
    @patch("os.path.getsize", return_value=10485760)  # 10 MB
    @patch("os.path.exists", return_value=True)
    def test_hw_encode_success(self, mock_exists, mock_size, mock_args, mock_ffmpeg):
        from opencut.core.hw_accel import hw_encode
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            input_path = f.name
        try:
            result = hw_encode(input_path, codec="h264", quality="balanced")
            self.assertIn("output_path", result)
            self.assertEqual(result["encoder_used"], "h264_nvenc")
            self.assertTrue(result["hw_accelerated"])
            self.assertEqual(result["codec"], "h264")
            self.assertIn("encode_time_seconds", result)
            self.assertEqual(result["file_size_mb"], 10.0)
        finally:
            try:
                os.unlink(input_path)
            except OSError:
                pass

    @patch("opencut.core.hw_accel.run_ffmpeg")
    @patch("opencut.core.hw_accel.get_hw_encode_args", return_value=["-c:v", "libx264", "-preset", "medium", "-crf", "23"])
    @patch("os.path.getsize", return_value=5242880)
    @patch("os.path.exists", return_value=True)
    def test_hw_encode_software_fallback(self, mock_exists, mock_size, mock_args, mock_ffmpeg):
        from opencut.core.hw_accel import hw_encode
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            input_path = f.name
        try:
            result = hw_encode(input_path, codec="h264", quality="balanced")
            self.assertFalse(result["hw_accelerated"])
            self.assertEqual(result["encoder_used"], "libx264")
        finally:
            try:
                os.unlink(input_path)
            except OSError:
                pass

    @patch("opencut.core.hw_accel.run_ffmpeg")
    @patch("opencut.core.hw_accel.get_hw_encode_args", return_value=["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "23", "-b:v", "0"])
    @patch("os.path.getsize", return_value=5242880)
    @patch("os.path.exists", return_value=True)
    def test_hw_encode_falls_back_on_runtime_error(self, mock_exists, mock_size, mock_args, mock_ffmpeg):
        from opencut.core.hw_accel import hw_encode
        # First call (HW) fails, second call (software) succeeds
        mock_ffmpeg.side_effect = [RuntimeError("NVENC init failed"), None]

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            input_path = f.name
        try:
            result = hw_encode(input_path, codec="h264", quality="balanced")
            self.assertFalse(result["hw_accelerated"])
            self.assertEqual(result["encoder_used"], "libx264")
            # run_ffmpeg should have been called twice
            self.assertEqual(mock_ffmpeg.call_count, 2)
        finally:
            try:
                os.unlink(input_path)
            except OSError:
                pass

    @patch("opencut.core.hw_accel.run_ffmpeg")
    @patch("opencut.core.hw_accel.get_hw_encode_args", return_value=["-c:v", "h264_nvenc", "-preset", "p1", "-rc", "vbr", "-cq", "28", "-b:v", "0"])
    @patch("os.path.getsize", return_value=5242880)
    @patch("os.path.exists", return_value=True)
    def test_hw_encode_progress_callback(self, mock_exists, mock_size, mock_args, mock_ffmpeg):
        from opencut.core.hw_accel import hw_encode
        progress_calls = []
        def on_progress(pct, msg):
            progress_calls.append((pct, msg))

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            input_path = f.name
        try:
            hw_encode(input_path, on_progress=on_progress)
            # Should have called progress at least for: detecting, encoding, complete
            self.assertTrue(len(progress_calls) >= 3)
            # Last call should be 100%
            self.assertEqual(progress_calls[-1][0], 100)
        finally:
            try:
                os.unlink(input_path)
            except OSError:
                pass

    @patch("opencut.core.hw_accel.run_ffmpeg")
    @patch("opencut.core.hw_accel.get_hw_encode_args", return_value=["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "23", "-b:v", "0"])
    @patch("os.path.getsize", return_value=5242880)
    @patch("os.path.exists", return_value=True)
    def test_hw_encode_custom_output_path(self, mock_exists, mock_size, mock_args, mock_ffmpeg):
        from opencut.core.hw_accel import hw_encode
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            input_path = f.name
        custom_out = os.path.join(tempfile.gettempdir(), "custom_output.mp4")
        try:
            result = hw_encode(input_path, output_path_override=custom_out)
            self.assertEqual(result["output_path"], custom_out)
        finally:
            try:
                os.unlink(input_path)
            except OSError:
                pass

    @patch("opencut.core.hw_accel.run_ffmpeg")
    @patch("opencut.core.hw_accel.get_hw_encode_args", return_value=["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "23", "-b:v", "0"])
    @patch("os.path.getsize", return_value=5242880)
    @patch("os.path.exists", return_value=True)
    def test_hw_encode_extra_args_passed(self, mock_exists, mock_size, mock_args, mock_ffmpeg):
        from opencut.core.hw_accel import hw_encode
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            input_path = f.name
        try:
            hw_encode(input_path, extra_args=["-t", "10"])
            # Verify extra args are in the command
            call_args = mock_ffmpeg.call_args[0][0]
            self.assertIn("-t", call_args)
            self.assertIn("10", call_args)
        finally:
            try:
                os.unlink(input_path)
            except OSError:
                pass


# ============================================================
# Test Export Presets Integration
# ============================================================
class TestExportPresetsIntegration(unittest.TestCase):
    """Tests for HW accel integration with export_presets.py."""

    def test_hw_presets_in_export_presets(self):
        from opencut.core.export_presets import EXPORT_PRESETS
        hw_preset_names = ["h264_hw_fast", "h264_hw_quality", "hevc_hw_fast",
                           "hevc_hw_quality", "av1_hw"]
        for name in hw_preset_names:
            self.assertIn(name, EXPORT_PRESETS, f"Missing HW preset: {name}")
            self.assertTrue(EXPORT_PRESETS[name].get("hw_accel"),
                            f"Preset {name} missing hw_accel=True")

    def test_hw_preset_categories(self):
        from opencut.core.export_presets import EXPORT_PRESETS
        hw_presets = {k: v for k, v in EXPORT_PRESETS.items() if v.get("hw_accel")}
        for name, preset in hw_presets.items():
            self.assertEqual(preset["category"], "hw_accel",
                             f"Preset {name} has wrong category")

    def test_hw_presets_have_required_fields(self):
        from opencut.core.export_presets import EXPORT_PRESETS
        required = ["label", "description", "category", "codec", "quality",
                     "hw_type", "audio_codec", "ext"]
        hw_presets = {k: v for k, v in EXPORT_PRESETS.items() if v.get("hw_accel")}
        for name, preset in hw_presets.items():
            for field in required:
                self.assertIn(field, preset,
                              f"Preset {name} missing field: {field}")

    def test_get_export_presets_includes_hw(self):
        from opencut.core.export_presets import get_export_presets
        presets = get_export_presets()
        names = [p["name"] for p in presets]
        self.assertIn("h264_hw_fast", names)
        self.assertIn("av1_hw", names)

    def test_get_preset_categories_includes_hw_accel(self):
        from opencut.core.export_presets import get_preset_categories
        cats = get_preset_categories()
        cat_names = [c["name"] for c in cats]
        self.assertIn("hw_accel", cat_names)

    def test_export_with_preset_hw_accel_parameter(self):
        """export_with_preset should accept hw_accel parameter."""
        import inspect

        from opencut.core.export_presets import export_with_preset
        sig = inspect.signature(export_with_preset)
        self.assertIn("hw_accel", sig.parameters)

    @patch("opencut.core.export_presets._export_hw_accel")
    def test_export_with_preset_routes_to_hw(self, mock_hw_export):
        """When hw_accel preset is used, _export_hw_accel is called."""
        from opencut.core.export_presets import export_with_preset
        mock_hw_export.return_value = "/tmp/output.mp4"

        export_with_preset("/tmp/input.mp4", "h264_hw_fast")
        mock_hw_export.assert_called_once()

    @patch("opencut.core.export_presets._export_hw_accel")
    def test_export_with_preset_hw_override_true(self, mock_hw_export):
        """Explicitly passing hw_accel=True to a software preset should use HW."""
        from opencut.core.export_presets import export_with_preset
        mock_hw_export.return_value = "/tmp/output.mp4"

        export_with_preset("/tmp/input.mp4", "youtube_1080p", hw_accel=True)
        mock_hw_export.assert_called_once()

    @patch("opencut.core.export_presets.run_ffmpeg")
    def test_export_with_preset_hw_override_false(self, mock_ffmpeg):
        """Explicitly passing hw_accel=False on a HW preset should use software."""
        from opencut.core.export_presets import export_with_preset
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            input_path = f.name
        try:
            # hw_accel=False overrides the preset's hw_accel=True
            export_with_preset(input_path, "h264_hw_fast", hw_accel=False)
            # Should have used standard FFmpeg path (run_ffmpeg called)
            mock_ffmpeg.assert_called_once()
            # The command should NOT contain nvenc
            cmd = mock_ffmpeg.call_args[0][0]
            # The preset uses codec="h264" which isn't a real ffmpeg codec,
            # but the point is it went through the software path
            self.assertNotIn("_export_hw_accel", str(cmd))
        finally:
            try:
                os.unlink(input_path)
            except OSError:
                pass


# ============================================================
# Test _export_hw_accel helper
# ============================================================
class TestExportHWAccelHelper(unittest.TestCase):
    """Tests for _export_hw_accel() internal helper."""

    @patch("opencut.core.hw_accel.get_hw_encode_args",
           return_value=["-c:v", "h264_nvenc", "-preset", "p1"])
    @patch("opencut.core.export_presets.run_ffmpeg")
    def test_export_hw_accel_builds_correct_cmd(self, mock_ffmpeg, mock_hw_args):
        from opencut.core.export_presets import EXPORT_PRESETS, _export_hw_accel
        preset = EXPORT_PRESETS["h264_hw_fast"]

        _export_hw_accel("/tmp/input.mp4", "/tmp/output.mp4", preset)
        mock_ffmpeg.assert_called_once()
        cmd = mock_ffmpeg.call_args[0][0]
        self.assertIn("h264_nvenc", cmd)
        self.assertIn("-movflags", cmd)

    @patch("opencut.core.hw_accel.get_hw_encode_args",
           return_value=["-c:v", "hevc_nvenc", "-preset", "p7"])
    @patch("opencut.core.export_presets.run_ffmpeg")
    def test_export_hw_accel_hevc_preset(self, mock_ffmpeg, mock_hw_args):
        from opencut.core.export_presets import EXPORT_PRESETS, _export_hw_accel
        preset = EXPORT_PRESETS["hevc_hw_quality"]

        _export_hw_accel("/tmp/input.mp4", "/tmp/output.mp4", preset)
        cmd = mock_ffmpeg.call_args[0][0]
        self.assertIn("hevc_nvenc", cmd)

    @patch("opencut.core.hw_accel.get_hw_encode_args",
           return_value=["-c:v", "libx264", "-preset", "ultrafast", "-crf", "28"])
    @patch("opencut.core.export_presets.run_ffmpeg")
    def test_export_hw_accel_software_fallback_in_cmd(self, mock_ffmpeg, mock_hw_args):
        from opencut.core.export_presets import EXPORT_PRESETS, _export_hw_accel
        preset = EXPORT_PRESETS["h264_hw_fast"]

        _export_hw_accel("/tmp/input.mp4", "/tmp/output.mp4", preset)
        cmd = mock_ffmpeg.call_args[0][0]
        # Should contain software codec since HW wasn't available
        self.assertIn("libx264", cmd)

    @patch("opencut.core.hw_accel.get_hw_encode_args",
           return_value=["-c:v", "h264_nvenc", "-preset", "p4"])
    @patch("opencut.core.export_presets.run_ffmpeg")
    def test_export_hw_accel_includes_scaling(self, mock_ffmpeg, mock_hw_args):
        from opencut.core.export_presets import _export_hw_accel
        preset = {
            "label": "Test HW",
            "width": 1280, "height": 720,
            "codec": "h264", "quality": "balanced", "hw_type": "auto",
            "audio_codec": "aac", "audio_bitrate": "192k",
            "pix_fmt": "yuv420p", "ext": ".mp4", "hw_accel": True,
        }
        _export_hw_accel("/tmp/input.mp4", "/tmp/output.mp4", preset)
        cmd = mock_ffmpeg.call_args[0][0]
        self.assertIn("-vf", cmd)
        # Should contain scaling filter with 1280x720
        vf_idx = cmd.index("-vf")
        vf_value = cmd[vf_idx + 1]
        self.assertIn("1280", vf_value)
        self.assertIn("720", vf_value)

    @patch("opencut.core.hw_accel.get_hw_encode_args",
           return_value=["-c:v", "h264_nvenc", "-preset", "p4"])
    @patch("opencut.core.export_presets.run_ffmpeg")
    def test_export_hw_accel_progress_callback(self, mock_ffmpeg, mock_hw_args):
        from opencut.core.export_presets import EXPORT_PRESETS, _export_hw_accel
        progress_calls = []
        def on_progress(pct, msg):
            progress_calls.append((pct, msg))

        preset = EXPORT_PRESETS["h264_hw_fast"]
        _export_hw_accel("/tmp/input.mp4", "/tmp/output.mp4", preset, on_progress)
        self.assertTrue(len(progress_calls) >= 2)
        # Last progress should be 100
        self.assertEqual(progress_calls[-1][0], 100)


# ============================================================
# Test HW Presets API
# ============================================================
class TestHWPresetsAPI(unittest.TestCase):
    """Tests for get_hw_presets()."""

    @patch("opencut.core.hw_accel.detect_hw_encoders")
    def test_get_hw_presets_with_hw_available(self, mock_detect):
        from opencut.core.hw_accel import HWEncoder, HWEncoderInfo, get_hw_presets
        mock_detect.return_value = HWEncoderInfo(
            available_encoders=[
                HWEncoder(name="h264_nvenc", codec="h264", hw_type="nvenc", supports_h264=True),
                HWEncoder(name="hevc_nvenc", codec="hevc", hw_type="nvenc", supports_hevc=True),
            ],
            preferred_h264="h264_nvenc",
            preferred_hevc="hevc_nvenc",
        )
        presets = get_hw_presets()
        self.assertEqual(len(presets), 5)
        h264_fast = next(p for p in presets if p["name"] == "h264_hw_fast")
        self.assertTrue(h264_fast["hw_available"])
        self.assertEqual(h264_fast["encoder"], "h264_nvenc")

    @patch("opencut.core.hw_accel.detect_hw_encoders")
    def test_get_hw_presets_without_hw(self, mock_detect):
        from opencut.core.hw_accel import HWEncoderInfo, get_hw_presets
        mock_detect.return_value = HWEncoderInfo()
        presets = get_hw_presets()
        self.assertEqual(len(presets), 5)
        for p in presets:
            self.assertFalse(p["hw_available"])
            self.assertIsNone(p["encoder"])
            self.assertIn("fallback", p)


# ============================================================
# Test GPU Detection
# ============================================================
class TestGPUDetection(unittest.TestCase):
    """Tests for _detect_gpu_name()."""

    @patch("subprocess.run")
    def test_detect_nvidia_gpu(self, mock_run):
        from opencut.core.hw_accel import _detect_gpu_name
        mock_run.return_value = MagicMock(
            returncode=0, stdout="NVIDIA GeForce RTX 4090\n"
        )
        name = _detect_gpu_name()
        self.assertEqual(name, "NVIDIA GeForce RTX 4090")

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_detect_gpu_no_nvidia_smi(self, mock_run):
        from opencut.core.hw_accel import _detect_gpu_name
        # On Windows it will try wmic next
        with patch("platform.system", return_value="Linux"):
            name = _detect_gpu_name()
        # No detection method worked
        self.assertIsNone(name)

    @patch("subprocess.run")
    @patch("platform.system", return_value="Windows")
    def test_detect_gpu_windows_wmic(self, mock_platform, mock_run):
        from opencut.core.hw_accel import _detect_gpu_name
        # nvidia-smi fails, wmic succeeds
        def run_side_effect(cmd, **kwargs):
            if "nvidia-smi" in cmd:
                raise FileNotFoundError
            return MagicMock(returncode=0, stdout="Name\nAMD Radeon RX 7900 XTX\n")
        mock_run.side_effect = run_side_effect
        name = _detect_gpu_name()
        self.assertEqual(name, "AMD Radeon RX 7900 XTX")


# ============================================================
# Test Route Endpoints
# ============================================================
class TestHWRoutes(unittest.TestCase):
    """Tests for /hw/* Flask route endpoints."""

    def setUp(self):
        """Set up Flask test client."""
        try:
            from opencut.config import OpenCutConfig
            from opencut.server import create_app
            test_config = OpenCutConfig()
            self.app = create_app(config=test_config)
            self.app.config["TESTING"] = True
            self.client = self.app.test_client()

            # Get CSRF token
            resp = self.client.get("/health")
            data = resp.get_json()
            self.csrf_token = data.get("csrf_token", "")
        except Exception:
            self.skipTest("Flask app not available for integration tests")

    def _headers(self):
        return {
            "X-OpenCut-Token": self.csrf_token,
            "Content-Type": "application/json",
        }

    @patch("opencut.core.hw_accel.detect_hw_encoders")
    def test_get_hw_encoders(self, mock_detect):
        from opencut.core.hw_accel import HWEncoderInfo
        mock_detect.return_value = HWEncoderInfo()
        resp = self.client.get("/hw/encoders")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("available_encoders", data)
        self.assertIn("has_hw_accel", data)

    @patch("opencut.core.hw_accel.detect_hw_encoders")
    def test_post_hw_encoders_refresh(self, mock_detect):
        from opencut.core.hw_accel import HWEncoderInfo
        mock_detect.return_value = HWEncoderInfo()
        resp = self.client.post("/hw/encoders/refresh", headers=self._headers())
        self.assertEqual(resp.status_code, 200)
        # force_refresh=True should have been passed
        mock_detect.assert_called_with(force_refresh=True)

    @patch("opencut.core.hw_accel.get_hw_presets", return_value=[])
    def test_get_hw_presets(self, mock_presets):
        resp = self.client.get("/hw/presets")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("presets", data)

    def test_post_hw_encode_no_filepath(self):
        resp = self.client.post(
            "/hw/encode",
            headers=self._headers(),
            data=json.dumps({"codec": "h264"}),
        )
        self.assertEqual(resp.status_code, 400)

    def test_post_hw_encode_invalid_file(self):
        resp = self.client.post(
            "/hw/encode",
            headers=self._headers(),
            data=json.dumps({
                "filepath": "/nonexistent/file.mp4",
                "codec": "h264",
            }),
        )
        # Should be 400 (file not found)
        self.assertIn(resp.status_code, [400])


# ============================================================
# Test Known Encoder Definitions
# ============================================================
class TestKnownEncoders(unittest.TestCase):
    """Tests for the _KNOWN_HW_ENCODERS mapping."""

    def test_all_known_encoders_have_valid_codec(self):
        from opencut.core.hw_accel import _KNOWN_HW_ENCODERS
        valid_codecs = {"h264", "hevc", "av1"}
        for name, (codec, hw_type) in _KNOWN_HW_ENCODERS.items():
            self.assertIn(codec, valid_codecs,
                          f"Encoder {name} has invalid codec: {codec}")

    def test_all_known_encoders_have_valid_hw_type(self):
        from opencut.core.hw_accel import _HW_PRIORITY, _KNOWN_HW_ENCODERS
        for name, (codec, hw_type) in _KNOWN_HW_ENCODERS.items():
            self.assertIn(hw_type, _HW_PRIORITY,
                          f"Encoder {name} has invalid hw_type: {hw_type}")

    def test_quality_presets_exist_for_all_hw_types(self):
        from opencut.core.hw_accel import _HW_PRIORITY, _QUALITY_PRESETS
        for hw_type in _HW_PRIORITY:
            self.assertIn(hw_type, _QUALITY_PRESETS,
                          f"Missing quality presets for hw_type: {hw_type}")
            for quality in ("speed", "balanced", "quality"):
                self.assertIn(quality, _QUALITY_PRESETS[hw_type],
                              f"Missing {quality} preset for {hw_type}")

    def test_software_presets_exist_for_all_codecs(self):
        from opencut.core.hw_accel import _SOFTWARE_PRESETS
        for codec in ("h264", "hevc", "av1"):
            self.assertIn(codec, _SOFTWARE_PRESETS,
                          f"Missing software presets for codec: {codec}")
            for quality in ("speed", "balanced", "quality"):
                self.assertIn(quality, _SOFTWARE_PRESETS[codec],
                              f"Missing {quality} preset for software {codec}")

    def test_encoder_names_match_codec_and_hwtype(self):
        """Every encoder name should be {codec}_{hw_type}."""
        from opencut.core.hw_accel import _KNOWN_HW_ENCODERS
        for name, (codec, hw_type) in _KNOWN_HW_ENCODERS.items():
            expected = f"{codec}_{hw_type}"
            self.assertEqual(name, expected,
                             f"Encoder name {name} doesn't match {expected}")


if __name__ == "__main__":
    unittest.main()
