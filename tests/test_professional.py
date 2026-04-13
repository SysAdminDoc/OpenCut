"""
Tests for OpenCut Professional / Broadcast features.

Covers:
  - Film Stock Emulation (film_emulation.py)
  - Glitch Effects Engine (glitch_effects.py)
  - Duplicate / Near-Duplicate Detection (duplicate_detect.py)
  - Smart Fit-to-Fill (fit_to_fill.py)
  - Wide Gamut Workflow (wide_gamut.py)
  - MXF Container Support (mxf_support.py)
  - Professional Routes (professional_routes.py)

Uses mocking for FFmpeg/FFprobe -- no real subprocess or GPU needed.
"""

import json
import os
import sys
import tempfile
import unittest
from dataclasses import asdict
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.conftest import csrf_headers

# ============================================================
# Film Stock Emulation Tests
# ============================================================

class TestFilmEmulation(unittest.TestCase):
    """Tests for opencut/core/film_emulation.py."""

    def test_list_film_stocks_returns_all(self):
        from opencut.core.film_emulation import FILM_STOCKS, list_film_stocks
        stocks = list_film_stocks()
        self.assertEqual(len(stocks), len(FILM_STOCKS))
        names = {s["name"] for s in stocks}
        self.assertIn("kodak_portra_400", names)
        self.assertIn("kodak_ektar_100", names)
        self.assertIn("fuji_pro_400h", names)
        self.assertIn("kodak_vision3_500t", names)
        self.assertIn("fuji_velvia_50", names)
        self.assertIn("ilford_hp5", names)

    def test_list_film_stocks_structure(self):
        from opencut.core.film_emulation import list_film_stocks
        stocks = list_film_stocks()
        for stock in stocks:
            self.assertIn("name", stock)
            self.assertIn("label", stock)
            self.assertIn("description", stock)
            self.assertIn("is_bw", stock)
            self.assertIn("grain_intensity", stock)

    def test_ilford_hp5_is_bw(self):
        from opencut.core.film_emulation import list_film_stocks
        stocks = {s["name"]: s for s in list_film_stocks()}
        self.assertTrue(stocks["ilford_hp5"]["is_bw"])
        self.assertFalse(stocks["kodak_portra_400"]["is_bw"])

    def test_apply_film_stock_invalid_stock(self):
        from opencut.core.film_emulation import apply_film_stock
        with self.assertRaises(ValueError) as ctx:
            apply_film_stock("/fake/input.mp4", stock="nonexistent")
        self.assertIn("Unknown film stock", str(ctx.exception))

    @patch("opencut.core.film_emulation.run_ffmpeg")
    @patch("opencut.core.film_emulation.get_video_info", return_value={
        "width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0,
    })
    def test_apply_film_stock_calls_ffmpeg(self, mock_info, mock_ff):
        from opencut.core.film_emulation import apply_film_stock
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            tmp = f.name
        try:
            progress_calls = []
            result = apply_film_stock(
                tmp, stock="kodak_portra_400", grain_amount=0.5, halation=0.3,
                on_progress=lambda p, m: progress_calls.append((p, m)),
            )
            self.assertIn("output_path", result)
            self.assertEqual(result["stock"], "kodak_portra_400")
            self.assertTrue(mock_ff.called)
            self.assertTrue(len(progress_calls) > 0)
        finally:
            os.unlink(tmp)

    @patch("opencut.core.film_emulation.run_ffmpeg")
    @patch("opencut.core.film_emulation.get_video_info", return_value={
        "width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0,
    })
    def test_apply_film_stock_bw_no_halation(self, mock_info, mock_ff):
        from opencut.core.film_emulation import apply_film_stock
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            tmp = f.name
        try:
            result = apply_film_stock(tmp, stock="ilford_hp5", halation=0.5)
            # B&W stock should not use halation even when requested
            self.assertEqual(result["stock"], "ilford_hp5")
            # Should use -vf not -filter_complex for B&W (no halation)
            cmd = mock_ff.call_args[0][0]
            self.assertIn("-vf", cmd)
        finally:
            os.unlink(tmp)

    def test_build_color_filters(self):
        from opencut.core.film_emulation import FILM_STOCKS, _build_color_filters
        preset = FILM_STOCKS["kodak_portra_400"]
        filters = _build_color_filters(preset, grain_amount=0.5)
        self.assertIsInstance(filters, list)
        self.assertTrue(len(filters) > 0)
        # Should have eq, colorbalance, curves, and noise
        filter_str = ",".join(filters)
        self.assertIn("eq=", filter_str)
        self.assertIn("colorbalance=", filter_str)
        self.assertIn("curves=", filter_str)

    def test_build_color_filters_bw(self):
        from opencut.core.film_emulation import FILM_STOCKS, _build_color_filters
        preset = FILM_STOCKS["ilford_hp5"]
        filters = _build_color_filters(preset, grain_amount=0.5)
        filter_str = ",".join(filters)
        self.assertIn("format=gray", filter_str)

    def test_grain_amount_clamped(self):
        from opencut.core.film_emulation import FILM_STOCKS, _build_color_filters
        preset = FILM_STOCKS["kodak_portra_400"]
        # Zero grain should produce no noise filter
        filters_no_grain = _build_color_filters(preset, grain_amount=0.0)
        filter_str = ",".join(filters_no_grain)
        self.assertNotIn("noise=", filter_str)


# ============================================================
# Glitch Effects Tests
# ============================================================

class TestGlitchEffects(unittest.TestCase):
    """Tests for opencut/core/glitch_effects.py."""

    def test_glitch_effects_registry(self):
        from opencut.core.glitch_effects import GLITCH_EFFECTS
        self.assertIn("rgb_split", GLITCH_EFFECTS)
        self.assertIn("scan_displacement", GLITCH_EFFECTS)
        self.assertIn("block_corruption", GLITCH_EFFECTS)
        self.assertIn("noise_burst", GLITCH_EFFECTS)
        self.assertIn("chromatic_aberration", GLITCH_EFFECTS)

    def test_apply_glitch_invalid_effect(self):
        from opencut.core.glitch_effects import apply_glitch
        with self.assertRaises(ValueError) as ctx:
            apply_glitch("/fake/input.mp4", effect="nonexistent")
        self.assertIn("Unknown glitch effect", str(ctx.exception))

    @patch("opencut.core.glitch_effects.run_ffmpeg")
    @patch("opencut.core.glitch_effects.get_video_info", return_value={
        "width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0,
    })
    def test_apply_rgb_split(self, mock_info, mock_ff):
        from opencut.core.glitch_effects import apply_glitch
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            tmp = f.name
        try:
            result = apply_glitch(tmp, effect="rgb_split", intensity=0.7)
            self.assertEqual(result["effect"], "rgb_split")
            self.assertIn("output_path", result)
            cmd = mock_ff.call_args[0][0]
            cmd_str = " ".join(cmd)
            self.assertIn("rgbashift", cmd_str)
        finally:
            os.unlink(tmp)

    @patch("opencut.core.glitch_effects.run_ffmpeg")
    @patch("opencut.core.glitch_effects.get_video_info", return_value={
        "width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0,
    })
    def test_apply_chromatic_aberration_uses_filter_complex(self, mock_info, mock_ff):
        from opencut.core.glitch_effects import apply_glitch
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            tmp = f.name
        try:
            result = apply_glitch(tmp, effect="chromatic_aberration", intensity=0.5)
            self.assertEqual(result["effect"], "chromatic_aberration")
            cmd = mock_ff.call_args[0][0]
            self.assertIn("-filter_complex", cmd)
        finally:
            os.unlink(tmp)

    @patch("opencut.core.glitch_effects.run_ffmpeg")
    @patch("opencut.core.glitch_effects.get_video_info", return_value={
        "width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0,
    })
    def test_apply_noise_burst(self, mock_info, mock_ff):
        from opencut.core.glitch_effects import apply_glitch
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            tmp = f.name
        try:
            apply_glitch(tmp, effect="noise_burst", intensity=0.8)
            cmd = mock_ff.call_args[0][0]
            cmd_str = " ".join(cmd)
            self.assertIn("noise=", cmd_str)
        finally:
            os.unlink(tmp)

    def test_apply_glitch_sequence_empty_timeline(self):
        from opencut.core.glitch_effects import apply_glitch_sequence
        with self.assertRaises(ValueError):
            apply_glitch_sequence("/fake/input.mp4", effects_timeline=[])

    def test_apply_glitch_sequence_missing_keys(self):
        from opencut.core.glitch_effects import apply_glitch_sequence
        with self.assertRaises(ValueError) as ctx:
            apply_glitch_sequence("/fake/input.mp4", effects_timeline=[
                {"effect": "rgb_split"}  # missing start_time, end_time
            ])
        self.assertIn("missing", str(ctx.exception))

    def test_build_glitch_filter_all_effects(self):
        from opencut.core.glitch_effects import GLITCH_EFFECTS, _build_glitch_filter
        info = {"width": 1920, "height": 1080}
        for effect_name in GLITCH_EFFECTS:
            vf, fc = _build_glitch_filter(effect_name, 0.5, info)
            self.assertTrue(vf or fc, f"No filter output for {effect_name}")

    def test_intensity_scales_rgb_shift(self):
        from opencut.core.glitch_effects import _build_glitch_filter
        info = {"width": 1920, "height": 1080}
        vf_low, _ = _build_glitch_filter("rgb_split", 0.1, info)
        vf_high, _ = _build_glitch_filter("rgb_split", 1.0, info)
        # Higher intensity should have larger shift values
        self.assertNotEqual(vf_low, vf_high)


# ============================================================
# Duplicate Detection Tests
# ============================================================

class TestDuplicateDetect(unittest.TestCase):
    """Tests for opencut/core/duplicate_detect.py."""

    def test_video_hash_dataclass(self):
        from opencut.core.duplicate_detect import VideoHash
        vh = VideoHash(
            file_path="/test.mp4", phash_first="abcd1234", phash_mid="5678ef01",
            phash_last="deadbeef", duration=60.0, file_size=1024000,
        )
        self.assertEqual(vh.file_path, "/test.mp4")
        self.assertEqual(vh.phash_first, "abcd1234")

    def test_compute_phash_nonexistent_file(self):
        from opencut.core.duplicate_detect import compute_phash
        result = compute_phash("/nonexistent/image.pgm")
        self.assertEqual(result, "0" * 16)  # fallback zero hash

    def test_compute_phash_with_pgm(self):
        from opencut.core.duplicate_detect import compute_phash
        # Create a minimal PGM file (32x32 grayscale)
        with tempfile.NamedTemporaryFile(suffix=".pgm", delete=False, mode="wb") as f:
            f.write(b"P5\n32 32\n255\n")
            # Gradient pattern
            for y in range(32):
                for x in range(32):
                    f.write(bytes([int((x + y) / 62 * 255)]))
            tmp = f.name
        try:
            result = compute_phash(tmp)
            self.assertEqual(len(result), 16)  # 64-bit hash as hex
            self.assertNotEqual(result, "0" * 16)
        finally:
            os.unlink(tmp)

    def test_hamming_distance(self):
        from opencut.core.duplicate_detect import _hamming_distance
        self.assertEqual(_hamming_distance("0000000000000000", "0000000000000000"), 0)
        self.assertEqual(_hamming_distance("ffffffffffffffff", "0000000000000000"), 64)
        self.assertEqual(_hamming_distance("0000000000000001", "0000000000000000"), 1)

    def test_hamming_distance_mismatched_lengths(self):
        from opencut.core.duplicate_detect import _hamming_distance
        self.assertEqual(_hamming_distance("abc", "abcdef"), 64)

    def test_hash_similarity_identical(self):
        from opencut.core.duplicate_detect import VideoHash, _hash_similarity
        h1 = VideoHash("/a.mp4", "abcdef0123456789", "abcdef0123456789",
                        "abcdef0123456789", 60.0, 1024)
        h2 = VideoHash("/b.mp4", "abcdef0123456789", "abcdef0123456789",
                        "abcdef0123456789", 60.0, 2048)
        sim = _hash_similarity(h1, h2)
        self.assertGreater(sim, 0.95)

    def test_hash_similarity_different(self):
        from opencut.core.duplicate_detect import VideoHash, _hash_similarity
        h1 = VideoHash("/a.mp4", "0000000000000000", "0000000000000000",
                        "0000000000000000", 60.0, 1024)
        h2 = VideoHash("/b.mp4", "ffffffffffffffff", "ffffffffffffffff",
                        "ffffffffffffffff", 60.0, 1024)
        sim = _hash_similarity(h1, h2)
        self.assertLess(sim, 0.3)

    def test_find_duplicates_too_few_files(self):
        from opencut.core.duplicate_detect import find_duplicates
        result = find_duplicates(["/a.mp4"])
        self.assertEqual(result, [])

    @patch("opencut.core.duplicate_detect.compute_video_hash")
    def test_find_duplicates_groups(self, mock_hash):
        from opencut.core.duplicate_detect import VideoHash, find_duplicates
        # Make two files that hash identically
        mock_hash.side_effect = [
            VideoHash("/a.mp4", "abcdef0123456789", "abcdef0123456789",
                      "abcdef0123456789", 60.0, 2048),
            VideoHash("/b.mp4", "abcdef0123456789", "abcdef0123456789",
                      "abcdef0123456789", 60.0, 1024),
            VideoHash("/c.mp4", "0000000000000000", "0000000000000000",
                      "0000000000000000", 30.0, 512),
        ]
        groups = find_duplicates(["/a.mp4", "/b.mp4", "/c.mp4"], threshold=0.85)
        self.assertEqual(len(groups), 1)
        self.assertIn("/a.mp4", groups[0]["files"])
        self.assertIn("/b.mp4", groups[0]["files"])
        # /a.mp4 has larger file_size, should be recommended_keep
        self.assertEqual(groups[0]["recommended_keep"], "/a.mp4")

    def test_read_pgm_p2_format(self):
        from opencut.core.duplicate_detect import _read_pgm_pixels
        with tempfile.NamedTemporaryFile(suffix=".pgm", delete=False, mode="wb") as f:
            content = "P2\n4 4\n255\n" + " ".join(str(i * 17) for i in range(16))
            f.write(content.encode())
            tmp = f.name
        try:
            pixels = _read_pgm_pixels(tmp)
            self.assertIsNotNone(pixels)
            self.assertEqual(len(pixels), 4)
            self.assertEqual(len(pixels[0]), 4)
        finally:
            os.unlink(tmp)


# ============================================================
# Fit-to-Fill Tests
# ============================================================

class TestFitToFill(unittest.TestCase):
    """Tests for opencut/core/fit_to_fill.py."""

    def test_invalid_target_duration(self):
        from opencut.core.fit_to_fill import fit_to_fill
        with self.assertRaises(ValueError):
            fit_to_fill("/fake/input.mp4", target_duration=0)
        with self.assertRaises(ValueError):
            fit_to_fill("/fake/input.mp4", target_duration=-5)

    def test_invalid_method(self):
        from opencut.core.fit_to_fill import fit_to_fill
        with self.assertRaises(ValueError) as ctx:
            fit_to_fill("/fake/input.mp4", target_duration=10, method="invalid")
        self.assertIn("Unknown method", str(ctx.exception))

    @patch("opencut.core.fit_to_fill.run_ffmpeg")
    @patch("opencut.core.fit_to_fill.get_video_info")
    def test_fit_uniform_speedup(self, mock_info, mock_ff):
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 20.0}
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            tmp = f.name
        try:
            from opencut.core.fit_to_fill import fit_to_fill
            result = fit_to_fill(tmp, target_duration=10, method="uniform")
            self.assertEqual(result["method"], "uniform")
            self.assertAlmostEqual(result["speed_factor"], 2.0, places=2)
        finally:
            os.unlink(tmp)

    @patch("opencut.core.fit_to_fill.run_ffmpeg")
    @patch("opencut.core.fit_to_fill.get_video_info")
    def test_fit_uniform_slowdown(self, mock_info, mock_ff):
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            tmp = f.name
        try:
            from opencut.core.fit_to_fill import fit_to_fill
            result = fit_to_fill(tmp, target_duration=20, method="uniform")
            self.assertAlmostEqual(result["speed_factor"], 0.5, places=2)
        finally:
            os.unlink(tmp)

    @patch("opencut.core.fit_to_fill.run_ffmpeg")
    @patch("opencut.core.fit_to_fill.get_video_info")
    def test_fit_auto_uses_uniform_for_small_change(self, mock_info, mock_ff):
        # 10% change should use uniform
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            tmp = f.name
        try:
            from opencut.core.fit_to_fill import fit_to_fill
            result = fit_to_fill(tmp, target_duration=9.0, method="auto")
            self.assertEqual(result["method"], "uniform")
        finally:
            os.unlink(tmp)

    @patch("opencut.core.fit_to_fill.run_ffmpeg")
    @patch("opencut.core.fit_to_fill.get_video_info")
    def test_fit_auto_uses_eased_for_large_change(self, mock_info, mock_ff):
        # 50% change should use eased
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            tmp = f.name
        try:
            from opencut.core.fit_to_fill import fit_to_fill
            result = fit_to_fill(tmp, target_duration=20.0, method="auto")
            self.assertEqual(result["method"], "eased")
        finally:
            os.unlink(tmp)

    def test_atempo_chain_normal(self):
        from opencut.core.fit_to_fill import _build_atempo_chain
        self.assertEqual(_build_atempo_chain(1.0), "")
        chain = _build_atempo_chain(1.5)
        self.assertIn("atempo=", chain)

    def test_atempo_chain_extreme_fast(self):
        from opencut.core.fit_to_fill import _build_atempo_chain
        chain = _build_atempo_chain(4.0)
        # Should chain multiple atempo filters for >2x
        self.assertIn("atempo=2.0", chain)

    def test_atempo_chain_extreme_slow(self):
        from opencut.core.fit_to_fill import _build_atempo_chain
        chain = _build_atempo_chain(0.3)
        # Should chain atempo=0.5 filters for <0.5x
        self.assertIn("atempo=0.5", chain)

    @patch("opencut.core.fit_to_fill.run_ffmpeg")
    @patch("opencut.core.fit_to_fill.get_video_info")
    def test_progress_callback(self, mock_info, mock_ff):
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            tmp = f.name
        try:
            from opencut.core.fit_to_fill import fit_to_fill
            calls = []
            fit_to_fill(tmp, target_duration=8, method="uniform",
                        on_progress=lambda p, m: calls.append(p))
            self.assertTrue(len(calls) > 0)
            self.assertEqual(calls[-1], 100)
        finally:
            os.unlink(tmp)


# ============================================================
# Wide Gamut Tests
# ============================================================

class TestWideGamut(unittest.TestCase):
    """Tests for opencut/core/wide_gamut.py."""

    def test_gamut_map_contains_expected(self):
        from opencut.core.wide_gamut import GAMUT_MAP
        self.assertIn("bt709", GAMUT_MAP)
        self.assertIn("bt2020", GAMUT_MAP)
        self.assertIn("dci_p3", GAMUT_MAP)
        self.assertIn("srgb", GAMUT_MAP)
        self.assertFalse(GAMUT_MAP["bt709"]["is_wide"])
        self.assertTrue(GAMUT_MAP["bt2020"]["is_wide"])

    @patch("opencut.core.wide_gamut.subprocess.run")
    def test_detect_gamut_bt709(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({
                "streams": [{
                    "color_primaries": "bt709",
                    "color_transfer": "bt709",
                    "color_space": "bt709",
                    "bits_per_raw_sample": "8",
                    "pix_fmt": "yuv420p",
                }]
            }).encode(),
        )
        from opencut.core.wide_gamut import detect_gamut
        result = detect_gamut("/fake/video.mp4")
        self.assertEqual(result["gamut_name"], "bt709")
        self.assertFalse(result["is_wide_gamut"])
        self.assertEqual(result["bit_depth"], 8)

    @patch("opencut.core.wide_gamut.subprocess.run")
    def test_detect_gamut_bt2020(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({
                "streams": [{
                    "color_primaries": "bt2020",
                    "color_transfer": "bt2020-10",
                    "color_space": "bt2020nc",
                    "bits_per_raw_sample": "10",
                    "pix_fmt": "yuv420p10le",
                }]
            }).encode(),
        )
        from opencut.core.wide_gamut import detect_gamut
        result = detect_gamut("/fake/video.mp4")
        self.assertEqual(result["gamut_name"], "bt2020")
        self.assertTrue(result["is_wide_gamut"])
        self.assertEqual(result["bit_depth"], 10)

    @patch("opencut.core.wide_gamut.subprocess.run")
    def test_detect_gamut_fallback_on_probe_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout=b"")
        from opencut.core.wide_gamut import detect_gamut
        result = detect_gamut("/fake/video.mp4")
        # Should default to bt709
        self.assertEqual(result["gamut_name"], "bt709")

    def test_convert_gamut_invalid_target(self):
        from opencut.core.wide_gamut import convert_gamut
        with self.assertRaises(ValueError) as ctx:
            convert_gamut("/fake/input.mp4", target_gamut="xyz")
        self.assertIn("Unknown target gamut", str(ctx.exception))

    def test_convert_gamut_invalid_intent(self):
        from opencut.core.wide_gamut import convert_gamut
        with self.assertRaises(ValueError) as ctx:
            convert_gamut("/fake/input.mp4", intent="bogus")
        self.assertIn("Unknown intent", str(ctx.exception))

    @patch("opencut.core.wide_gamut.run_ffmpeg")
    @patch("opencut.core.wide_gamut.subprocess.run")
    def test_convert_gamut_calls_ffmpeg(self, mock_probe, mock_ff):
        mock_probe.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"streams": [{"color_primaries": "bt2020",
                                            "color_transfer": "bt2020-10",
                                            "color_space": "bt2020nc",
                                            "pix_fmt": "yuv420p10le"}]}).encode(),
        )
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            tmp = f.name
        try:
            from opencut.core.wide_gamut import convert_gamut
            result = convert_gamut(tmp, target_gamut="bt709", intent="perceptual")
            self.assertIn("output_path", result)
            self.assertEqual(result["target_gamut"], "bt709")
            self.assertTrue(mock_ff.called)
            cmd = mock_ff.call_args[0][0]
            cmd_str = " ".join(cmd)
            self.assertIn("zscale", cmd_str)
        finally:
            os.unlink(tmp)

    def test_gamut_info_dataclass(self):
        from opencut.core.wide_gamut import GamutInfo
        info = GamutInfo(
            gamut_name="bt709", gamut_label="BT.709", primaries="bt709",
            transfer="bt709", matrix="bt709", is_wide_gamut=False, bit_depth=8,
        )
        d = asdict(info)
        self.assertEqual(d["gamut_name"], "bt709")


# ============================================================
# MXF Support Tests
# ============================================================

class TestMXFSupport(unittest.TestCase):
    """Tests for opencut/core/mxf_support.py."""

    def test_mxf_profiles_registry(self):
        from opencut.core.mxf_support import MXF_PROFILES
        self.assertIn("dnxhr_hq", MXF_PROFILES)
        self.assertIn("dnxhr_lb", MXF_PROFILES)
        self.assertIn("xdcam_hd422", MXF_PROFILES)

    def test_mxf_info_dataclass(self):
        from opencut.core.mxf_support import MXFInfo
        info = MXFInfo(
            file_path="/test.mxf", op_pattern="op1a",
            timecode="01:00:00:00",
        )
        d = asdict(info)
        self.assertEqual(d["op_pattern"], "op1a")
        self.assertEqual(d["timecode"], "01:00:00:00")

    @patch("opencut.core.mxf_support.subprocess.run")
    def test_probe_mxf(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({
                "format": {
                    "format_name": "mxf",
                    "duration": "60.0",
                    "size": "1048576",
                    "tags": {"timecode": "01:00:00:00"},
                },
                "streams": [
                    {"index": 0, "codec_type": "video", "codec_name": "dnxhd",
                     "width": 1920, "height": 1080, "duration": "60.0",
                     "pix_fmt": "yuv422p", "profile": "DNxHR HQ",
                     "tags": {}},
                    {"index": 1, "codec_type": "audio", "codec_name": "pcm_s16le",
                     "sample_rate": "48000", "channels": 2, "duration": "60.0",
                     "tags": {}},
                ],
            }).encode(),
        )
        with tempfile.NamedTemporaryFile(suffix=".mxf", delete=False) as f:
            f.write(b"fake")
            tmp = f.name
        try:
            from opencut.core.mxf_support import probe_mxf
            result = probe_mxf(tmp)
            self.assertEqual(result["op_pattern"], "op1a")
            self.assertEqual(result["timecode"], "01:00:00:00")
            self.assertEqual(result["essence_type"], "dnxhd")
            self.assertEqual(len(result["tracks"]), 2)
        finally:
            os.unlink(tmp)

    def test_probe_mxf_file_not_found(self):
        from opencut.core.mxf_support import probe_mxf
        with self.assertRaises(FileNotFoundError):
            probe_mxf("/nonexistent/file.mxf")

    @patch("opencut.core.mxf_support.subprocess.run")
    def test_probe_mxf_opatom(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({
                "format": {"format_name": "mxf_opatom", "duration": "30.0",
                           "size": "500000", "tags": {}},
                "streams": [
                    {"index": 0, "codec_type": "video", "codec_name": "dnxhd",
                     "duration": "30.0", "tags": {}},
                ],
            }).encode(),
        )
        with tempfile.NamedTemporaryFile(suffix=".mxf", delete=False) as f:
            f.write(b"fake")
            tmp = f.name
        try:
            from opencut.core.mxf_support import probe_mxf
            result = probe_mxf(tmp)
            self.assertEqual(result["op_pattern"], "opatom")
        finally:
            os.unlink(tmp)

    def test_export_mxf_invalid_op(self):
        from opencut.core.mxf_support import export_mxf
        with self.assertRaises(ValueError) as ctx:
            export_mxf("/fake/input.mp4", op_pattern="invalid")
        self.assertIn("Unknown OP pattern", str(ctx.exception))

    @patch("opencut.core.mxf_support.run_ffmpeg")
    @patch("opencut.core.mxf_support.subprocess.run")
    def test_export_mxf_op1a(self, mock_probe, mock_ff):
        # Patch probe for timecode readback
        mock_probe.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({
                "format": {"format_name": "mxf", "duration": "10", "size": "100",
                           "tags": {"timecode": "01:00:00:00"}},
                "streams": [],
            }).encode(),
        )
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            tmp = f.name
        try:
            from opencut.core.mxf_support import export_mxf
            result = export_mxf(tmp, op_pattern="op1a", timecode="01:00:00:00")
            self.assertIn("output_path", result)
            self.assertEqual(result["op_pattern"], "op1a")
            cmd = mock_ff.call_args[0][0]
            self.assertIn("-f", cmd)
            idx = cmd.index("-f")
            self.assertEqual(cmd[idx + 1], "mxf")
        finally:
            os.unlink(tmp)

    @patch("opencut.core.mxf_support.run_ffmpeg")
    def test_convert_to_mxf_dnxhr(self, mock_ff):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            tmp = f.name
        try:
            from opencut.core.mxf_support import convert_to_mxf
            result = convert_to_mxf(tmp, codec="dnxhd", profile="dnxhr_hq")
            self.assertEqual(result["profile"], "dnxhr_hq")
            cmd = mock_ff.call_args[0][0]
            cmd_str = " ".join(cmd)
            self.assertIn("dnxhd", cmd_str)
            self.assertIn("dnxhr_hq", cmd_str)
            self.assertIn("pcm_s16le", cmd_str)
        finally:
            os.unlink(tmp)

    def test_convert_to_mxf_invalid_profile(self):
        from opencut.core.mxf_support import convert_to_mxf
        with self.assertRaises(ValueError) as ctx:
            convert_to_mxf("/fake/input.mp4", codec="invalid", profile="invalid")
        self.assertIn("Unknown profile", str(ctx.exception))


# ============================================================
# Route Smoke Tests
# ============================================================

class TestProfessionalRoutes:
    """Smoke tests for professional_routes.py endpoints."""

    # --- Film Stock ---

    def test_list_film_stocks_route(self, client):
        resp = client.get("/effects/film-stocks")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "stocks" in data
        assert len(data["stocks"]) >= 6

    def test_apply_film_stock_requires_csrf(self, client):
        resp = client.post("/effects/film-stock",
                           data=json.dumps({"filepath": "/fake.mp4"}),
                           content_type="application/json")
        assert resp.status_code == 403

    def test_apply_film_stock_no_file(self, client, csrf_token):
        resp = client.post("/effects/film-stock",
                           data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    # --- Glitch ---

    def test_apply_glitch_requires_csrf(self, client):
        resp = client.post("/effects/glitch",
                           data=json.dumps({"filepath": "/fake.mp4"}),
                           content_type="application/json")
        assert resp.status_code == 403

    def test_apply_glitch_no_file(self, client, csrf_token):
        resp = client.post("/effects/glitch",
                           data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_apply_glitch_sequence_requires_csrf(self, client):
        resp = client.post("/effects/glitch-sequence",
                           data=json.dumps({"filepath": "/fake.mp4"}),
                           content_type="application/json")
        assert resp.status_code == 403

    # --- Duplicate Detection ---

    def test_find_duplicates_requires_csrf(self, client):
        resp = client.post("/media/find-duplicates",
                           data=json.dumps({}),
                           content_type="application/json")
        assert resp.status_code == 403

    # --- Fit-to-Fill ---

    def test_fit_to_fill_requires_csrf(self, client):
        resp = client.post("/video/fit-to-fill",
                           data=json.dumps({"filepath": "/fake.mp4"}),
                           content_type="application/json")
        assert resp.status_code == 403

    def test_fit_to_fill_no_file(self, client, csrf_token):
        resp = client.post("/video/fit-to-fill",
                           data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    # --- Gamut ---

    def test_detect_gamut_requires_csrf(self, client):
        resp = client.post("/video/gamut/detect",
                           data=json.dumps({"filepath": "/fake.mp4"}),
                           content_type="application/json")
        assert resp.status_code == 403

    def test_detect_gamut_no_file(self, client, csrf_token):
        resp = client.post("/video/gamut/detect",
                           data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_convert_gamut_requires_csrf(self, client):
        resp = client.post("/video/gamut/convert",
                           data=json.dumps({"filepath": "/fake.mp4"}),
                           content_type="application/json")
        assert resp.status_code == 403

    def test_check_clipping_requires_csrf(self, client):
        resp = client.post("/video/gamut/check-clipping",
                           data=json.dumps({"filepath": "/fake.mp4"}),
                           content_type="application/json")
        assert resp.status_code == 403

    # --- MXF ---

    def test_probe_mxf_requires_csrf(self, client):
        resp = client.post("/mxf/probe",
                           data=json.dumps({"filepath": "/fake.mxf"}),
                           content_type="application/json")
        assert resp.status_code == 403

    def test_probe_mxf_no_file(self, client, csrf_token):
        resp = client.post("/mxf/probe",
                           data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_export_mxf_requires_csrf(self, client):
        resp = client.post("/mxf/export",
                           data=json.dumps({"filepath": "/fake.mp4"}),
                           content_type="application/json")
        assert resp.status_code == 403

    def test_convert_mxf_requires_csrf(self, client):
        resp = client.post("/mxf/convert",
                           data=json.dumps({"filepath": "/fake.mp4"}),
                           content_type="application/json")
        assert resp.status_code == 403


if __name__ == "__main__":
    unittest.main()
