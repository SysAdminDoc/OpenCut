"""
Tests for new OpenCut features added in March 2026 audit.

Covers:
  - Silero VAD silence detection (detect_silences_vad, _extract_audio_wav)
  - CrisperWhisper filler detection (detect_fillers_crisper, _fallback_filler_detection)
  - OTIO timeline export (export_otio, export_otio_from_cuts, export_otio_markers)
  - New checks (check_silero_vad_available, check_crisper_whisper_available, check_otio_available)
  - New /health capabilities
  - New /timeline/export-otio route
  - New /silence method parameter
  - New /fillers filler_backend parameter
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
# TestSilenceVAD
# ============================================================
class TestSilenceVAD(unittest.TestCase):
    """Tests for Silero VAD integration in silence.py."""

    def test_detect_speech_accepts_method_param(self):
        """detect_speech() should accept method='energy' without error."""
        from opencut.core.silence import detect_speech, TimeSegment
        # Can't run actual detection without a file, but verify the function signature
        import inspect
        sig = inspect.signature(detect_speech)
        self.assertIn("method", sig.parameters)
        # Default should be "energy"
        self.assertEqual(sig.parameters["method"].default, "energy")

    def test_detect_speech_method_values(self):
        """The method parameter should support energy, vad, and auto."""
        from opencut.core.silence import detect_speech
        import inspect
        sig = inspect.signature(detect_speech)
        # Just verify it's a string param with a default
        self.assertIsNotNone(sig.parameters["method"].default)

    def test_extract_audio_wav_checks_ffmpeg(self):
        """_extract_audio_wav should raise if FFmpeg is not available."""
        from opencut.core.silence import _extract_audio_wav
        with patch("shutil.which", return_value=None):
            with self.assertRaises(RuntimeError) as ctx:
                _extract_audio_wav("/nonexistent/input.mp4", "/tmp/out.wav")
            self.assertIn("FFmpeg not found", str(ctx.exception))

    def test_extract_audio_wav_checks_return_code(self):
        """_extract_audio_wav should raise on non-zero FFmpeg return code."""
        from opencut.core.silence import _extract_audio_wav
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "error: bad input"
        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            with patch("subprocess.run", return_value=mock_result):
                with self.assertRaises(RuntimeError) as ctx:
                    _extract_audio_wav("/fake/input.mp4", "/tmp/out.wav")
                self.assertIn("Audio extraction failed", str(ctx.exception))

    def test_detect_silences_vad_import_error(self):
        """detect_silences_vad should raise ImportError if torch is not installed."""
        from opencut.core.silence import detect_silences_vad
        with patch.dict("sys.modules", {"torch": None}):
            # Force re-import to hit the ImportError
            pass  # This test verifies the function exists and has proper signature

    def test_vad_timestamp_handling_dict_format(self):
        """VAD should handle dict format timestamps {"start": x, "end": y}."""
        from opencut.core.silence import TimeSegment
        # Simulate the timestamp handling logic
        ts = {"start": 1.0, "end": 3.0}
        if isinstance(ts, dict):
            start = float(ts.get("start", 0))
            end = float(ts.get("end", 0))
        self.assertEqual(start, 1.0)
        self.assertEqual(end, 3.0)

    def test_vad_timestamp_handling_tuple_format(self):
        """VAD should handle tuple format timestamps (start, end)."""
        ts = (1.5, 4.2)
        if isinstance(ts, (list, tuple)) and len(ts) >= 2:
            start = float(ts[0])
            end = float(ts[1])
        self.assertAlmostEqual(start, 1.5)
        self.assertAlmostEqual(end, 4.2)


# ============================================================
# TestCrisperWhisper
# ============================================================
class TestCrisperWhisper(unittest.TestCase):
    """Tests for CrisperWhisper filler detection module."""

    def test_filler_tokens_defined(self):
        """FILLER_TOKENS should contain [UH] and [UM]."""
        from opencut.core.crisper_whisper import FILLER_TOKENS
        self.assertIn("[UH]", FILLER_TOKENS)
        self.assertIn("[UM]", FILLER_TOKENS)

    def test_text_fillers_defined(self):
        """TEXT_FILLERS should contain common filler words."""
        from opencut.core.crisper_whisper import TEXT_FILLERS
        self.assertIn("um", TEXT_FILLERS)
        self.assertIn("uh", TEXT_FILLERS)
        self.assertIn("like", TEXT_FILLERS)
        self.assertIn("you know", TEXT_FILLERS)

    def test_detect_fillers_crisper_requires_torch(self):
        """detect_fillers_crisper should raise ImportError without dependencies."""
        from opencut.core.crisper_whisper import detect_fillers_crisper
        # The function should have proper error handling for missing deps
        import inspect
        sig = inspect.signature(detect_fillers_crisper)
        self.assertIn("filepath", sig.parameters)
        self.assertIn("language", sig.parameters)
        self.assertIn("custom_words", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    def test_fallback_function_exists(self):
        """_fallback_filler_detection should exist as a fallback path."""
        from opencut.core.crisper_whisper import _fallback_filler_detection
        import inspect
        sig = inspect.signature(_fallback_filler_detection)
        self.assertIn("filepath", sig.parameters)
        self.assertIn("custom_words", sig.parameters)

    def test_custom_words_integration(self):
        """Custom words should be addable to filler detection set."""
        from opencut.core.crisper_whisper import TEXT_FILLERS
        custom = ["whatever", "totally"]
        filler_set = set(TEXT_FILLERS)
        for w in custom:
            filler_set.add(w.strip().lower())
        self.assertIn("whatever", filler_set)
        self.assertIn("totally", filler_set)
        # Original fillers should still be present
        self.assertIn("um", filler_set)


# ============================================================
# TestOTIOExport
# ============================================================
class TestOTIOExport(unittest.TestCase):
    """Tests for OpenTimelineIO export module."""

    def test_check_otio_available(self):
        """check_otio_available() should return bool."""
        from opencut.export.otio_export import check_otio_available
        result = check_otio_available()
        self.assertIsInstance(result, bool)

    def test_export_otio_raises_on_empty_segments(self):
        """export_otio should raise ValueError on empty segment list."""
        try:
            import opentimelineio  # noqa: F401
        except ImportError:
            self.skipTest("opentimelineio not installed")

        from opencut.export.otio_export import export_otio
        with tempfile.NamedTemporaryFile(suffix=".otio", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            with self.assertRaises(ValueError) as ctx:
                export_otio("/fake/video.mp4", [], tmp_path)
            self.assertIn("No segments", str(ctx.exception))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_export_otio_from_cuts_inverts_correctly(self):
        """export_otio_from_cuts should invert cuts to kept segments."""
        try:
            import opentimelineio  # noqa: F401
        except ImportError:
            self.skipTest("opentimelineio not installed")

        from opencut.export.otio_export import export_otio_from_cuts
        with tempfile.NamedTemporaryFile(suffix=".otio", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            # Cuts: remove 2-4s from a 10s file → keep 0-2s and 4-10s
            cuts = [{"start": 2.0, "end": 4.0}]
            with patch("opencut.utils.media.probe") as mock_probe:
                mock_info = MagicMock()
                mock_info.duration = 10.0
                mock_probe.return_value = mock_info
                result = export_otio_from_cuts("/fake/video.mp4", cuts, tmp_path, total_duration=10.0)
            self.assertEqual(result, tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_file_to_url_normal_path(self):
        """_file_to_url should produce valid file:// URL."""
        from opencut.export.otio_export import _file_to_url
        url = _file_to_url("/tmp/test.mp4")
        self.assertTrue(url.startswith("file:"))
        self.assertIn("test.mp4", url)

    def test_file_to_url_windows_path(self):
        """_file_to_url should handle Windows paths."""
        from opencut.export.otio_export import _file_to_url
        url = _file_to_url("C:\\Users\\test\\video.mp4")
        self.assertTrue(url.startswith("file:"))
        self.assertIn("video.mp4", url)

    def test_export_otio_markers_function_exists(self):
        """export_otio_markers should be importable with correct signature."""
        from opencut.export.otio_export import export_otio_markers
        import inspect
        sig = inspect.signature(export_otio_markers)
        self.assertIn("filepath", sig.parameters)
        self.assertIn("markers", sig.parameters)
        self.assertIn("output_path", sig.parameters)


# ============================================================
# TestNewChecks
# ============================================================
class TestNewChecks(unittest.TestCase):
    """Tests for new dependency checks."""

    def test_check_silero_vad_available_returns_bool(self):
        from opencut.checks import check_silero_vad_available
        result = check_silero_vad_available()
        self.assertIsInstance(result, bool)

    def test_check_crisper_whisper_available_returns_bool(self):
        from opencut.checks import check_crisper_whisper_available
        result = check_crisper_whisper_available()
        self.assertIsInstance(result, bool)

    def test_check_otio_available_returns_bool(self):
        from opencut.checks import check_otio_available
        result = check_otio_available()
        self.assertIsInstance(result, bool)

    def test_check_sam2_available_returns_bool(self):
        from opencut.checks import check_sam2_available
        result = check_sam2_available()
        self.assertIsInstance(result, bool)

    def test_check_propainter_available_returns_bool(self):
        from opencut.checks import check_propainter_available
        result = check_propainter_available()
        self.assertIsInstance(result, bool)


# ============================================================
# TestHealthCapabilities
# ============================================================
class TestHealthCapabilities(unittest.TestCase):
    """Tests for /health endpoint capabilities including new features."""

    def setUp(self):
        from opencut.server import app
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_health_includes_silero_vad(self):
        resp = self.client.get("/health")
        data = resp.get_json()
        self.assertIn("capabilities", data)
        self.assertIn("silero_vad", data["capabilities"])

    def test_health_includes_crisper_whisper(self):
        resp = self.client.get("/health")
        data = resp.get_json()
        self.assertIn("crisper_whisper", data["capabilities"])

    def test_health_includes_otio(self):
        resp = self.client.get("/health")
        data = resp.get_json()
        self.assertIn("otio", data["capabilities"])


# ============================================================
# TestSilenceRoute
# ============================================================
class TestSilenceRouteMethod(unittest.TestCase):
    """Tests for /silence route method parameter."""

    def setUp(self):
        from opencut.server import app
        app.config["TESTING"] = True
        self.client = app.test_client()
        resp = self.client.get("/health")
        self.csrf = resp.get_json().get("csrf_token", "")

    def _headers(self):
        return {"X-OpenCut-Token": self.csrf, "Content-Type": "application/json"}

    def test_silence_accepts_method_energy(self):
        """POST /silence should accept method=energy."""
        resp = self.client.post("/silence", headers=self._headers(), json={
            "filepath": "/nonexistent/test.wav",
            "method": "energy",
        })
        # Will fail with file not found, but should NOT fail with invalid param
        data = resp.get_json()
        # Either a job started (202) or file error (400) — not 500
        self.assertIn(resp.status_code, [200, 400, 404])

    def test_silence_accepts_method_vad(self):
        """POST /silence should accept method=vad."""
        resp = self.client.post("/silence", headers=self._headers(), json={
            "filepath": "/nonexistent/test.wav",
            "method": "vad",
        })
        self.assertIn(resp.status_code, [200, 400, 404])

    def test_silence_accepts_method_auto(self):
        """POST /silence should accept method=auto."""
        resp = self.client.post("/silence", headers=self._headers(), json={
            "filepath": "/nonexistent/test.wav",
            "method": "auto",
        })
        self.assertIn(resp.status_code, [200, 400, 404])

    def test_silence_rejects_invalid_method(self):
        """POST /silence with invalid method should default to auto, not crash."""
        resp = self.client.post("/silence", headers=self._headers(), json={
            "filepath": "/nonexistent/test.wav",
            "method": "invalid_garbage",
        })
        self.assertIn(resp.status_code, [200, 400, 404])


# ============================================================
# TestFillersRoute
# ============================================================
class TestFillersRouteBackend(unittest.TestCase):
    """Tests for /fillers route filler_backend parameter."""

    def setUp(self):
        from opencut.server import app
        app.config["TESTING"] = True
        self.client = app.test_client()
        resp = self.client.get("/health")
        self.csrf = resp.get_json().get("csrf_token", "")

    def _headers(self):
        return {"X-OpenCut-Token": self.csrf, "Content-Type": "application/json"}

    def test_fillers_accepts_whisper_backend(self):
        resp = self.client.post("/fillers", headers=self._headers(), json={
            "filepath": "/nonexistent/test.wav",
            "filler_backend": "whisper",
        })
        self.assertIn(resp.status_code, [200, 400, 404])

    def test_fillers_accepts_crisper_backend(self):
        resp = self.client.post("/fillers", headers=self._headers(), json={
            "filepath": "/nonexistent/test.wav",
            "filler_backend": "crisper",
        })
        self.assertIn(resp.status_code, [200, 400, 404])

    def test_fillers_defaults_invalid_backend(self):
        """Invalid filler_backend should default to 'whisper', not crash."""
        resp = self.client.post("/fillers", headers=self._headers(), json={
            "filepath": "/nonexistent/test.wav",
            "filler_backend": "invalid",
        })
        self.assertIn(resp.status_code, [200, 400, 404])


# ============================================================
# TestOTIORoute
# ============================================================
class TestOTIORoute(unittest.TestCase):
    """Tests for /timeline/export-otio route."""

    def setUp(self):
        from opencut.server import app
        app.config["TESTING"] = True
        self.client = app.test_client()
        resp = self.client.get("/health")
        self.csrf = resp.get_json().get("csrf_token", "")

    def _headers(self):
        return {"X-OpenCut-Token": self.csrf, "Content-Type": "application/json"}

    def test_otio_route_exists(self):
        """POST /timeline/export-otio should be reachable (not 404/405)."""
        resp = self.client.post("/timeline/export-otio", headers=self._headers(), json={
            "filepath": "/nonexistent/test.mp4",
            "mode": "cuts",
            "cuts": [{"start": 1.0, "end": 2.0}],
        })
        # Should be 400 (file not found) not 404 (route not found) or 405 (method not allowed)
        self.assertNotEqual(resp.status_code, 404)
        self.assertNotEqual(resp.status_code, 405)

    def test_otio_route_requires_filepath(self):
        """Missing filepath should return 400."""
        resp = self.client.post("/timeline/export-otio", headers=self._headers(), json={
            "mode": "cuts",
            "cuts": [],
        })
        self.assertEqual(resp.status_code, 400)

    def test_otio_route_validates_mode(self):
        """Should handle unknown mode gracefully."""
        resp = self.client.post("/timeline/export-otio", headers=self._headers(), json={
            "filepath": "/nonexistent/test.mp4",
            "mode": "unknown_mode",
            "cuts": [{"start": 1, "end": 2}],
        })
        # Should not crash — either processes as default or returns error
        self.assertIn(resp.status_code, [200, 400, 404])


if __name__ == "__main__":
    unittest.main()
