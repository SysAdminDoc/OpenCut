"""
Unit tests for Color & MAM features:
- Color Scopes (13.1)
- Three-Way Color Wheels (13.2)
- HSL Qualifier (13.3)
- Power Windows (13.6)
- ACES Color Pipeline (43.1)
- Proxy Generation (23.1)
- AI Metadata Enrichment (23.2)
- Kinetic Typography (26.1)
- Data-Driven Animation (26.2)
- Shape Layer Animation (26.3)
- Route smoke tests for all endpoints
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest


# ========================================================================
# 1. Color Scopes (13.1)
# ========================================================================
class TestColorScopes:
    """Tests for opencut.core.color_scopes."""

    def test_scope_result_dataclass(self):
        from opencut.core.color_scopes import ScopeResult
        r = ScopeResult(scope_type="waveform", output_path="/tmp/w.png",
                        width=720, height=480, timestamp=1.5)
        assert r.scope_type == "waveform"
        assert r.width == 720
        assert r.height == 480
        assert r.timestamp == 1.5
        d = r.to_dict()
        assert d["scope_type"] == "waveform"

    def test_all_scopes_result_dataclass(self):
        from opencut.core.color_scopes import AllScopesResult, ScopeResult
        r = AllScopesResult(timestamp=2.0, output_dir="/tmp")
        r.scopes["waveform"] = ScopeResult(scope_type="waveform")
        assert "waveform" in r.scopes
        d = r.to_dict()
        assert "scopes" in d

    def test_generate_waveform_calls_ffmpeg(self):
        from opencut.core.color_scopes import generate_waveform
        with patch("opencut.core.color_scopes.run_ffmpeg") as mock_ff:
            result = generate_waveform("/tmp/test.mp4", timestamp=1.0,
                                        output_path="/tmp/wf.png")
            assert mock_ff.called
            assert result.scope_type == "waveform"
            assert result.output_path == "/tmp/wf.png"

    def test_generate_waveform_default_output(self):
        from opencut.core.color_scopes import generate_waveform
        with patch("opencut.core.color_scopes.run_ffmpeg"):
            result = generate_waveform("/tmp/clip.mp4")
            assert "waveform" in result.output_path

    def test_generate_waveform_progress(self):
        from opencut.core.color_scopes import generate_waveform
        progress_calls = []
        with patch("opencut.core.color_scopes.run_ffmpeg"):
            generate_waveform("/tmp/test.mp4", output_path="/tmp/wf.png",
                              on_progress=lambda p, m: progress_calls.append(p))
        assert 10 in progress_calls
        assert 100 in progress_calls

    def test_generate_vectorscope_calls_ffmpeg(self):
        from opencut.core.color_scopes import generate_vectorscope
        with patch("opencut.core.color_scopes.run_ffmpeg") as mock_ff:
            result = generate_vectorscope("/tmp/test.mp4",
                                           output_path="/tmp/vs.png")
            assert mock_ff.called
            assert result.scope_type == "vectorscope"

    def test_generate_rgb_parade_calls_ffmpeg(self):
        from opencut.core.color_scopes import generate_rgb_parade
        with patch("opencut.core.color_scopes.run_ffmpeg") as mock_ff:
            result = generate_rgb_parade("/tmp/test.mp4",
                                          output_path="/tmp/rp.png")
            assert mock_ff.called
            assert result.scope_type == "rgb_parade"

    def test_generate_histogram_calls_ffmpeg(self):
        from opencut.core.color_scopes import generate_histogram
        with patch("opencut.core.color_scopes.run_ffmpeg") as mock_ff:
            result = generate_histogram("/tmp/test.mp4",
                                         output_path="/tmp/h.png")
            assert mock_ff.called
            assert result.scope_type == "histogram"

    def test_generate_histogram_display_modes(self):
        from opencut.core.color_scopes import generate_histogram
        for mode in ("stack", "parade", "overlay"):
            with patch("opencut.core.color_scopes.run_ffmpeg"):
                result = generate_histogram("/tmp/test.mp4",
                                             output_path="/tmp/h.png",
                                             display_mode=mode)
                assert result.scope_type == "histogram"

    def test_generate_all_scopes(self):
        from opencut.core.color_scopes import generate_all_scopes
        with patch("opencut.core.color_scopes.run_ffmpeg"), \
             tempfile.TemporaryDirectory() as tmpdir:
            result = generate_all_scopes("/tmp/test.mp4",
                                          output_dir=tmpdir)
            assert len(result.scopes) == 4
            assert "waveform" in result.scopes
            assert "vectorscope" in result.scopes
            assert "rgb_parade" in result.scopes
            assert "histogram" in result.scopes


# ========================================================================
# 2. Three-Way Color Wheels (13.2)
# ========================================================================
class TestColorWheels:
    """Tests for opencut.core.color_wheels."""

    def test_color_wheel_settings_defaults(self):
        from opencut.core.color_wheels import ColorWheelSettings
        s = ColorWheelSettings()
        assert s.lift == (0.0, 0.0, 0.0)
        assert s.saturation == 1.0
        assert s.is_neutral()

    def test_color_wheel_settings_not_neutral(self):
        from opencut.core.color_wheels import ColorWheelSettings
        s = ColorWheelSettings(lift=(0.1, 0.0, 0.0))
        assert not s.is_neutral()

    def test_color_wheel_result_dataclass(self):
        from opencut.core.color_wheels import ColorWheelResult
        r = ColorWheelResult(output_path="/tmp/out.mp4", preview=False)
        d = r.to_dict()
        assert d["output_path"] == "/tmp/out.mp4"
        assert d["preview"] is False

    def test_apply_color_wheels_calls_ffmpeg(self):
        from opencut.core.color_wheels import apply_color_wheels
        with patch("opencut.core.color_wheels.run_ffmpeg") as mock_ff:
            result = apply_color_wheels(
                "/tmp/test.mp4",
                lift=(0.1, 0.0, -0.1),
                output_path="/tmp/graded.mp4",
            )
            assert mock_ff.called
            assert result.output_path == "/tmp/graded.mp4"

    def test_apply_color_wheels_with_settings_dict(self):
        from opencut.core.color_wheels import apply_color_wheels
        with patch("opencut.core.color_wheels.run_ffmpeg"):
            result = apply_color_wheels(
                "/tmp/test.mp4",
                settings={"lift": [0.1, 0, 0], "gamma": [0, 0.1, 0]},
                output_path="/tmp/graded.mp4",
            )
            assert result.settings["lift"] == (0.1, 0.0, 0.0)

    def test_preview_color_wheels(self):
        from opencut.core.color_wheels import preview_color_wheels
        with patch("opencut.core.color_wheels.run_ffmpeg"):
            result = preview_color_wheels(
                "/tmp/test.mp4", timestamp=2.0,
                output_path="/tmp/preview.png",
            )
            assert result.preview is True
            assert result.output_path == "/tmp/preview.png"

    def test_build_colorbalance_filter(self):
        from opencut.core.color_wheels import (
            ColorWheelSettings,
            _build_colorbalance_filter,
        )
        s = ColorWheelSettings(lift=(0.5, -0.3, 0.0))
        f = _build_colorbalance_filter(s)
        assert "colorbalance=" in f
        assert "rs=0.5" in f

    def test_build_eq_filter_neutral(self):
        from opencut.core.color_wheels import ColorWheelSettings, _build_eq_filter
        s = ColorWheelSettings()
        assert _build_eq_filter(s) is None

    def test_build_eq_filter_with_saturation(self):
        from opencut.core.color_wheels import ColorWheelSettings, _build_eq_filter
        s = ColorWheelSettings(saturation=1.5)
        f = _build_eq_filter(s)
        assert f is not None
        assert "saturation=1.5" in f


# ========================================================================
# 3. HSL Qualifier (13.3)
# ========================================================================
class TestHSLQualifier:
    """Tests for opencut.core.hsl_qualifier."""

    def test_hsl_range_defaults(self):
        from opencut.core.hsl_qualifier import HSLRange
        r = HSLRange()
        assert r.hue_center == 120.0
        assert r.sat_min == 0.2
        d = r.to_dict()
        assert "hue_center" in d

    def test_secondary_correction_defaults(self):
        from opencut.core.hsl_qualifier import SecondaryCorrection
        c = SecondaryCorrection()
        assert c.hue_shift == 0.0
        assert c.saturation == 1.0

    def test_qualify_hsl_calls_ffmpeg(self):
        from opencut.core.hsl_qualifier import qualify_hsl
        with patch("opencut.core.hsl_qualifier.run_ffmpeg") as mock_ff:
            result = qualify_hsl(
                "/tmp/test.mp4",
                hsl_range={"hue_center": 90, "hue_width": 40},
                output_path="/tmp/hsl.mp4",
            )
            assert mock_ff.called
            assert result.output_path == "/tmp/hsl.mp4"
            assert result.hsl_range is not None

    def test_preview_matte_calls_ffmpeg(self):
        from opencut.core.hsl_qualifier import preview_matte
        with patch("opencut.core.hsl_qualifier.run_ffmpeg"):
            result = preview_matte(
                "/tmp/test.mp4", timestamp=1.0,
                output_path="/tmp/matte.png",
            )
            assert result.matte_preview is True

    def test_apply_secondary_correction(self):
        from opencut.core.hsl_qualifier import apply_secondary_correction
        with patch("opencut.core.hsl_qualifier.run_ffmpeg"):
            result = apply_secondary_correction(
                "/tmp/test.mp4",
                qualification={"hue_center": 200, "hue_width": 30},
                correction={"hue_shift": 10, "saturation": 1.3},
                output_path="/tmp/sec.mp4",
            )
            assert result.correction is not None
            assert result.correction["hue_shift"] == 10

    def test_parse_hsl_from_tuples(self):
        from opencut.core.hsl_qualifier import _parse_hsl_range
        r = _parse_hsl_range(hue_range=(90, 45), sat_range=(0.3, 0.9))
        assert r.hue_center == 90
        assert r.hue_width == 45
        assert r.sat_min == 0.3


# ========================================================================
# 4. Power Windows (13.6)
# ========================================================================
class TestPowerWindows:
    """Tests for opencut.core.power_windows."""

    def test_power_window_dataclass(self):
        from opencut.core.power_windows import PowerWindow
        w = PowerWindow(shape="circle", x=0.5, y=0.5, width=0.4, height=0.4)
        assert w.shape == "circle"
        d = w.to_dict()
        assert d["shape"] == "circle"

    def test_create_power_window(self):
        from opencut.core.power_windows import create_power_window
        w = create_power_window(
            shape="rectangle", position=(0.3, 0.7),
            feather=0.1, width=0.5, height=0.3,
        )
        assert w.shape == "rectangle"
        assert w.x == 0.3
        assert w.y == 0.7

    def test_create_power_window_invalid_shape(self):
        from opencut.core.power_windows import create_power_window
        w = create_power_window(shape="invalid")
        assert w.shape == "circle"

    def test_track_window(self):
        from opencut.core.power_windows import track_window
        with patch("opencut.core.power_windows.get_video_info") as mock_info, \
             tempfile.TemporaryDirectory() as tmpdir:
            mock_info.return_value = {"width": 1920, "height": 1080,
                                       "fps": 30, "duration": 10}
            out = os.path.join(tmpdir, "tracking.json")
            result = track_window(
                "/tmp/test.mp4",
                window={"shape": "circle", "x": 0.5, "y": 0.5},
                output_path=out,
            )
            assert result.tracking is not None
            assert result.tracking["fps"] == 30

    def test_apply_windowed_correction(self):
        from opencut.core.power_windows import apply_windowed_correction
        with patch("opencut.core.power_windows.get_video_info") as mock_info, \
             patch("opencut.core.power_windows.run_ffmpeg"):
            mock_info.return_value = {"width": 1920, "height": 1080}
            result = apply_windowed_correction(
                "/tmp/test.mp4",
                window_data={"shape": "circle", "x": 0.5, "y": 0.5},
                correction={"brightness": 0.2, "contrast": 1.2},
                output_path="/tmp/windowed.mp4",
            )
            assert result.correction_applied is True

    def test_tracking_data_dataclass(self):
        from opencut.core.power_windows import TrackingData
        t = TrackingData(fps=30, duration=5.0, total_frames=150)
        d = t.to_dict()
        assert d["fps"] == 30
        assert d["total_frames"] == 150


# ========================================================================
# 5. ACES Color Pipeline (43.1)
# ========================================================================
class TestACESPipeline:
    """Tests for opencut.core.aces_pipeline."""

    def test_aces_config_defaults(self):
        from opencut.core.aces_pipeline import ACESConfig
        c = ACESConfig()
        assert c.idt == "srgb"
        assert c.odt == "rec709"

    def test_aces_result_dataclass(self):
        from opencut.core.aces_pipeline import ACESResult
        r = ACESResult(output_path="/tmp/aces.mp4", idt_name="sRGB")
        d = r.to_dict()
        assert d["idt_name"] == "sRGB"

    def test_list_available_idts(self):
        from opencut.core.aces_pipeline import list_available_idts
        idts = list_available_idts()
        assert len(idts) > 5
        keys = [i["key"] for i in idts]
        assert "srgb" in keys
        assert "slog3" in keys

    def test_list_available_odts(self):
        from opencut.core.aces_pipeline import list_available_odts
        odts = list_available_odts()
        assert len(odts) > 3
        keys = [o["key"] for o in odts]
        assert "rec709" in keys

    def test_detect_camera_idt_default(self):
        from opencut.core.aces_pipeline import detect_camera_idt
        with patch("opencut.core.aces_pipeline.get_video_info") as mock_info:
            mock_info.return_value = {"color_space": "", "color_transfer": ""}
            idt = detect_camera_idt("/tmp/test.mp4")
            assert idt == "srgb"

    def test_detect_camera_idt_slog3(self):
        from opencut.core.aces_pipeline import detect_camera_idt
        with patch("opencut.core.aces_pipeline.get_video_info") as mock_info:
            mock_info.return_value = {"color_space": "", "color_transfer": "slog3"}
            idt = detect_camera_idt("/tmp/test.mp4")
            assert idt == "slog3"

    def test_detect_camera_idt_rec2020(self):
        from opencut.core.aces_pipeline import detect_camera_idt
        with patch("opencut.core.aces_pipeline.get_video_info") as mock_info:
            mock_info.return_value = {"color_space": "bt2020nc",
                                       "color_transfer": ""}
            idt = detect_camera_idt("/tmp/test.mp4")
            assert idt == "rec2020"

    def test_apply_aces_pipeline(self):
        from opencut.core.aces_pipeline import apply_aces_pipeline
        with patch("opencut.core.aces_pipeline.run_ffmpeg"):
            result = apply_aces_pipeline(
                "/tmp/test.mp4", idt="srgb", odt="rec709",
                output_path="/tmp/aces.mp4",
            )
            assert result.output_path == "/tmp/aces.mp4"
            assert result.idt_name == "sRGB / Rec.709"

    def test_apply_aces_pipeline_with_config(self):
        from opencut.core.aces_pipeline import ACESConfig, apply_aces_pipeline
        with patch("opencut.core.aces_pipeline.run_ffmpeg"):
            cfg = ACESConfig(idt="slog3", odt="rec2020_pq", exposure=1.5)
            result = apply_aces_pipeline(
                "/tmp/test.mp4", config=cfg,
                output_path="/tmp/aces_hdr.mp4",
            )
            assert result.config["exposure"] == 1.5


# ========================================================================
# 6. Proxy Generation (23.1)
# ========================================================================
class TestProxyGen:
    """Tests for opencut.core.proxy_gen."""

    def test_proxy_config_defaults(self):
        from opencut.core.proxy_gen import ProxyConfig
        c = ProxyConfig()
        assert c.preset == "half"
        assert c.crf == 23

    def test_proxy_result_dataclass(self):
        from opencut.core.proxy_gen import ProxyResult
        r = ProxyResult(original_path="/tmp/orig.mp4", proxy_path="/tmp/proxy.mp4")
        d = r.to_dict()
        assert d["original_path"] == "/tmp/orig.mp4"

    def test_proxy_presets_exist(self):
        from opencut.core.proxy_gen import PROXY_PRESETS
        assert "half" in PROXY_PRESETS
        assert "quarter" in PROXY_PRESETS
        assert "720p" in PROXY_PRESETS

    def test_generate_proxy(self):
        from opencut.core.proxy_gen import generate_proxy
        with patch("opencut.core.proxy_gen.get_video_info") as mock_info, \
             patch("opencut.core.proxy_gen.run_ffmpeg"), \
             patch("opencut.core.proxy_gen.os.path.getsize", return_value=1000), \
             patch("opencut.core.proxy_gen.os.path.isfile", return_value=True), \
             tempfile.TemporaryDirectory() as tmpdir:
            mock_info.return_value = {"width": 3840, "height": 2160}
            out = os.path.join(tmpdir, "proxy.mp4")
            result = generate_proxy(
                "/tmp/4k.mp4", output_path=out,
                output_dir=tmpdir,
            )
            assert result.proxy_path == out
            assert result.original_width == 3840

    def test_batch_generate_proxies(self):
        from opencut.core.proxy_gen import batch_generate_proxies
        with patch("opencut.core.proxy_gen.generate_proxy") as mock_gen, \
             patch("opencut.core.proxy_gen.os.path.isfile", return_value=True):
            from opencut.core.proxy_gen import ProxyResult
            mock_gen.return_value = ProxyResult(proxy_path="/tmp/p.mp4")
            result = batch_generate_proxies(
                ["/tmp/a.mp4", "/tmp/b.mp4"],
                output_dir="/tmp/proxies",
            )
            assert result.total == 2
            assert result.completed == 2

    def test_relink_proxy_to_original(self):
        from opencut.core.proxy_gen import relink_proxy_to_original
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a proxy map
            map_path = os.path.join(tmpdir, ".opencut_proxy_map.json")
            proxy_abs = os.path.abspath(os.path.join(tmpdir, "proxy.mp4"))
            orig_abs = os.path.abspath(os.path.join(tmpdir, "original.mp4"))
            with open(map_path, "w") as f:
                json.dump({proxy_abs: orig_abs}, f)
            # Create the original file
            with open(orig_abs, "w") as f:
                f.write("test")
            result = relink_proxy_to_original(proxy_abs, proxy_dir=tmpdir)
            assert result == orig_abs

    def test_relink_proxy_not_found(self):
        from opencut.core.proxy_gen import relink_proxy_to_original
        with pytest.raises(FileNotFoundError):
            relink_proxy_to_original("/tmp/nonexistent_proxy.mp4")


# ========================================================================
# 7. AI Metadata Enrichment (23.2)
# ========================================================================
class TestAIMetadata:
    """Tests for opencut.core.ai_metadata."""

    def test_enriched_metadata_defaults(self):
        from opencut.core.ai_metadata import EnrichedMetadata
        m = EnrichedMetadata()
        assert m.file_path == ""
        assert m.tags == []
        d = m.to_dict()
        assert "shot_type" in d

    def test_batch_enrich_result_dataclass(self):
        from opencut.core.ai_metadata import BatchEnrichResult
        r = BatchEnrichResult(total=5)
        assert r.total == 5
        assert r.completed == 0

    def test_enrich_metadata(self):
        from opencut.core.ai_metadata import enrich_metadata
        with patch("opencut.core.ai_metadata.get_video_info") as mock_info, \
             patch("opencut.core.ai_metadata._get_file_metadata", return_value={}), \
             patch("opencut.core.ai_metadata._analyze_brightness",
                   return_value={"avg_brightness": 150}), \
             patch("opencut.core.ai_metadata._detect_dominant_colors",
                   return_value=["warm"]):
            mock_info.return_value = {
                "width": 1920, "height": 1080, "fps": 30,
                "duration": 60, "codec": "h264",
            }
            meta = enrich_metadata("/tmp/test.mp4")
            assert meta.resolution == "1920x1080"
            assert meta.fps == 30
            assert len(meta.tags) > 0

    def test_detect_objects(self):
        from opencut.core.ai_metadata import detect_objects
        with patch("opencut.core.ai_metadata.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stderr="yavg=128")
            result = detect_objects("/tmp/frame.jpg")
            assert isinstance(result, list)

    def test_classify_scene(self):
        from opencut.core.ai_metadata import classify_scene
        with patch("opencut.core.ai_metadata._analyze_brightness",
                   return_value={"avg_brightness": 200}):
            result = classify_scene("/tmp/frame.jpg")
            assert "category" in result
            assert "lighting" in result

    def test_batch_enrich(self):
        from opencut.core.ai_metadata import EnrichedMetadata, batch_enrich
        with patch("opencut.core.ai_metadata.enrich_metadata") as mock_enrich, \
             patch("opencut.core.ai_metadata.os.path.isfile", return_value=True):
            mock_enrich.return_value = EnrichedMetadata(file_path="/tmp/a.mp4")
            result = batch_enrich(["/tmp/a.mp4", "/tmp/b.mp4"])
            assert result.total == 2
            assert result.completed == 2

    def test_shot_types_constant(self):
        from opencut.core.ai_metadata import SHOT_TYPES
        assert "close_up" in SHOT_TYPES
        assert "wide" in SHOT_TYPES
        assert len(SHOT_TYPES) == 10


# ========================================================================
# 8. Kinetic Typography (26.1)
# ========================================================================
class TestKineticType:
    """Tests for opencut.core.kinetic_type."""

    def test_kinetic_preset_dataclass(self):
        from opencut.core.kinetic_type import KineticPreset
        p = KineticPreset(name="Fade In", easing="ease_out")
        assert p.name == "Fade In"

    def test_kinetic_result_dataclass(self):
        from opencut.core.kinetic_type import KineticResult
        r = KineticResult(output_path="/tmp/kt.mp4", text="Hello")
        d = r.to_dict()
        assert d["text"] == "Hello"

    def test_list_animation_presets(self):
        from opencut.core.kinetic_type import list_animation_presets
        presets = list_animation_presets()
        assert len(presets) >= 10
        keys = [p["key"] for p in presets]
        assert "typewriter" in keys
        assert "fade_in" in keys
        assert "wave" in keys

    def test_animate_text_fade_in(self):
        from opencut.core.kinetic_type import animate_text
        with patch("opencut.core.kinetic_type.run_ffmpeg"):
            result = animate_text(
                "Hello World", animation_preset="fade_in",
                duration=3.0, output_path="/tmp/kt.mp4",
            )
            assert result.text == "Hello World"
            assert result.preset == "fade_in"

    def test_animate_text_typewriter(self):
        from opencut.core.kinetic_type import animate_text
        with patch("opencut.core.kinetic_type.run_ffmpeg"):
            result = animate_text(
                "Type this", animation_preset="typewriter",
                output_path="/tmp/tw.mp4",
            )
            assert result.preset == "typewriter"

    def test_animate_text_invalid_preset(self):
        from opencut.core.kinetic_type import animate_text
        with patch("opencut.core.kinetic_type.run_ffmpeg"):
            result = animate_text(
                "Test", animation_preset="nonexistent",
                output_path="/tmp/kt.mp4",
            )
            assert result.preset == "fade_in"

    def test_create_custom_animation(self):
        from opencut.core.kinetic_type import create_custom_animation
        with patch("opencut.core.kinetic_type.run_ffmpeg"):
            result = create_custom_animation(
                keyframes=[
                    {"time": 0, "opacity": 0, "x": 0.5, "y": 0.5},
                    {"time": 3, "opacity": 1, "x": 0.5, "y": 0.3},
                ],
                text="Custom", output_path="/tmp/custom.mp4",
            )
            assert result.preset == "custom"
            assert result.text == "Custom"

    def test_render_kinetic_text_preset(self):
        from opencut.core.kinetic_type import render_kinetic_text
        with patch("opencut.core.kinetic_type.run_ffmpeg"):
            result = render_kinetic_text(
                animation_data={"text": "Test", "preset": "wave", "duration": 2},
                output_path="/tmp/kt.mp4",
            )
            assert result.text == "Test"

    def test_render_kinetic_text_keyframes(self):
        from opencut.core.kinetic_type import render_kinetic_text
        with patch("opencut.core.kinetic_type.run_ffmpeg"):
            result = render_kinetic_text(
                animation_data={
                    "text": "KF", "duration": 2,
                    "keyframes": [
                        {"time": 0, "opacity": 0},
                        {"time": 2, "opacity": 1},
                    ],
                },
                output_path="/tmp/kf.mp4",
            )
            assert result.preset == "custom"

    def test_easing_functions(self):
        from opencut.core.kinetic_type import (
            _ease_bounce,
            _ease_in,
            _ease_in_out,
            _ease_linear,
            _ease_out,
        )
        assert _ease_linear(0.5) == 0.5
        assert _ease_in(0.0) == 0.0
        assert _ease_in(1.0) == 1.0
        assert _ease_out(1.0) == 1.0
        assert 0 <= _ease_in_out(0.5) <= 1
        assert _ease_bounce(1.0) == pytest.approx(1.0, abs=0.01)


# ========================================================================
# 9. Data-Driven Animation (26.2)
# ========================================================================
class TestDataAnimation:
    """Tests for opencut.core.data_animation."""

    def test_data_template_defaults(self):
        from opencut.core.data_animation import DataTemplate
        t = DataTemplate()
        assert t.chart_type == "bar"
        assert t.width == 1920

    def test_data_animation_result(self):
        from opencut.core.data_animation import DataAnimationResult
        r = DataAnimationResult(chart_type="bar", data_points=5)
        d = r.to_dict()
        assert d["chart_type"] == "bar"

    def test_load_data_source_list(self):
        from opencut.core.data_animation import _load_data_source
        data = [{"label": "A", "value": 10}]
        result = _load_data_source(data)
        assert len(result) == 1

    def test_load_data_source_json_string(self):
        from opencut.core.data_animation import _load_data_source
        data = '[{"label": "A", "value": 10}]'
        result = _load_data_source(data)
        assert len(result) == 1

    def test_extract_labels_values(self):
        from opencut.core.data_animation import _extract_labels_values
        data = [{"name": "Apple", "count": "30"},
                {"name": "Banana", "count": "20"}]
        labels, values = _extract_labels_values(data)
        assert labels == ["Apple", "Banana"]
        assert values == [30.0, 20.0]

    def test_create_data_animation_bar(self):
        from opencut.core.data_animation import create_data_animation
        with patch("opencut.core.data_animation.run_ffmpeg"):
            result = create_data_animation(
                template={"chart_type": "bar", "title": "Sales"},
                data_source=[{"item": "A", "value": 10},
                             {"item": "B", "value": 20}],
                output_path="/tmp/bar.mp4",
            )
            assert result.chart_type == "bar"
            assert result.data_points == 2

    def test_render_bar_chart(self):
        from opencut.core.data_animation import render_bar_chart
        with patch("opencut.core.data_animation.run_ffmpeg"):
            result = render_bar_chart(
                data=[{"name": "X", "val": "50"}],
                output_path="/tmp/bc.mp4",
            )
            assert result.chart_type == "bar"

    def test_render_counter(self):
        from opencut.core.data_animation import render_counter
        with patch("opencut.core.data_animation.run_ffmpeg"):
            result = render_counter(
                start=0, end=100, duration=3,
                output_path="/tmp/counter.mp4",
                title="Revenue",
                prefix="$",
            )
            assert result.chart_type == "counter"
            assert result.duration == 3

    def test_create_data_animation_counter(self):
        from opencut.core.data_animation import create_data_animation
        with patch("opencut.core.data_animation.run_ffmpeg"):
            result = create_data_animation(
                template={"chart_type": "counter"},
                data_source=[{"val": 0}, {"val": 100}],
                output_path="/tmp/counter.mp4",
            )
            assert result.chart_type == "counter"


# ========================================================================
# 10. Shape Layer Animation (26.3)
# ========================================================================
class TestShapeAnimation:
    """Tests for opencut.core.shape_animation."""

    def test_shape_definition_defaults(self):
        from opencut.core.shape_animation import ShapeDefinition
        s = ShapeDefinition()
        assert s.shape_type == "circle"
        assert s.color == "white"

    def test_shape_animation_result(self):
        from opencut.core.shape_animation import ShapeAnimationResult
        r = ShapeAnimationResult(animation_type="morph", duration=3.0)
        d = r.to_dict()
        assert d["animation_type"] == "morph"

    def test_parse_shape_dict(self):
        from opencut.core.shape_animation import _parse_shape
        s = _parse_shape({"shape_type": "rectangle", "width": 0.5})
        assert s.shape_type == "rectangle"
        assert s.width == 0.5

    def test_parse_shape_string(self):
        from opencut.core.shape_animation import _parse_shape
        s = _parse_shape("triangle")
        assert s.shape_type == "triangle"

    def test_animate_shape_morph(self):
        from opencut.core.shape_animation import animate_shape_morph
        with patch("opencut.core.shape_animation.run_ffmpeg"):
            result = animate_shape_morph(
                shape_a={"shape_type": "circle", "width": 0.2},
                shape_b={"shape_type": "rectangle", "width": 0.4},
                output_path="/tmp/morph.mp4",
            )
            assert result.animation_type == "morph"

    def test_animate_stroke_draw(self):
        from opencut.core.shape_animation import animate_stroke_draw
        with patch("opencut.core.shape_animation.run_ffmpeg"):
            result = animate_stroke_draw(
                "/tmp/shape.svg", duration=3.0,
                output_path="/tmp/draw.mp4",
            )
            assert result.animation_type == "stroke_draw"

    def test_animate_fill_transition(self):
        from opencut.core.shape_animation import animate_fill_transition
        with patch("opencut.core.shape_animation.run_ffmpeg"):
            result = animate_fill_transition(
                color_a="#4285F4", color_b="#EA4335",
                duration=2.0, output_path="/tmp/fill.mp4",
            )
            assert result.animation_type == "fill_transition"
            assert result.duration == 2.0

    def test_animate_fill_transition_hex_parsing(self):
        from opencut.core.shape_animation import animate_fill_transition
        with patch("opencut.core.shape_animation.run_ffmpeg"):
            result = animate_fill_transition(
                color_a="0xFF0000", color_b="0x00FF00",
                output_path="/tmp/fill2.mp4",
            )
            assert result.animation_type == "fill_transition"


# ========================================================================
# Route smoke tests
# ========================================================================
class TestColorMAMRoutes:
    """Smoke tests for all Color & MAM routes."""

    # -- Color Scopes --
    def test_waveform_route_missing_path(self, client, csrf_token):
        from tests.conftest import csrf_headers
        resp = client.post("/video/color-scopes/waveform",
                           headers=csrf_headers(csrf_token),
                           json={})
        assert resp.status_code == 400

    def test_vectorscope_route_missing_path(self, client, csrf_token):
        from tests.conftest import csrf_headers
        resp = client.post("/video/color-scopes/vectorscope",
                           headers=csrf_headers(csrf_token),
                           json={})
        assert resp.status_code == 400

    def test_rgb_parade_route_missing_path(self, client, csrf_token):
        from tests.conftest import csrf_headers
        resp = client.post("/video/color-scopes/rgb-parade",
                           headers=csrf_headers(csrf_token),
                           json={})
        assert resp.status_code == 400

    def test_histogram_route_missing_path(self, client, csrf_token):
        from tests.conftest import csrf_headers
        resp = client.post("/video/color-scopes/histogram",
                           headers=csrf_headers(csrf_token),
                           json={})
        assert resp.status_code == 400

    def test_all_scopes_route_missing_path(self, client, csrf_token):
        from tests.conftest import csrf_headers
        resp = client.post("/video/color-scopes/all",
                           headers=csrf_headers(csrf_token),
                           json={})
        assert resp.status_code == 400

    # -- Color Wheels --
    def test_color_wheels_preview_missing_path(self, client, csrf_token):
        from tests.conftest import csrf_headers
        resp = client.post("/video/color-wheels/preview",
                           headers=csrf_headers(csrf_token),
                           json={})
        assert resp.status_code == 400

    # -- HSL Qualifier --
    def test_hsl_matte_preview_missing_path(self, client, csrf_token):
        from tests.conftest import csrf_headers
        resp = client.post("/video/hsl-qualifier/matte-preview",
                           headers=csrf_headers(csrf_token),
                           json={})
        assert resp.status_code == 400

    # -- ACES --
    def test_aces_detect_idt_missing_path(self, client, csrf_token):
        from tests.conftest import csrf_headers
        resp = client.post("/video/aces/detect-idt",
                           headers=csrf_headers(csrf_token),
                           json={})
        assert resp.status_code == 400

    def test_aces_list_idts(self, client):
        resp = client.get("/video/aces/idts")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "idts" in data
        assert len(data["idts"]) > 5

    def test_aces_list_odts(self, client):
        resp = client.get("/video/aces/odts")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "odts" in data
        assert len(data["odts"]) > 3

    # -- Power Windows --
    def test_power_window_create_route(self, client, csrf_token):
        from tests.conftest import csrf_headers
        resp = client.post("/video/power-windows/create",
                           headers=csrf_headers(csrf_token),
                           json={"shape": "circle", "position": [0.5, 0.5]})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["shape"] == "circle"

    # -- Proxy --
    def test_proxy_relink_missing_path(self, client, csrf_token):
        from tests.conftest import csrf_headers
        resp = client.post("/video/proxy/relink",
                           headers=csrf_headers(csrf_token),
                           json={})
        assert resp.status_code == 400

    # -- AI Metadata --
    def test_ai_metadata_detect_objects_missing(self, client, csrf_token):
        from tests.conftest import csrf_headers
        resp = client.post("/video/ai-metadata/detect-objects",
                           headers=csrf_headers(csrf_token),
                           json={})
        assert resp.status_code == 400

    def test_ai_metadata_classify_scene_missing(self, client, csrf_token):
        from tests.conftest import csrf_headers
        resp = client.post("/video/ai-metadata/classify-scene",
                           headers=csrf_headers(csrf_token),
                           json={})
        assert resp.status_code == 400

    # -- Kinetic Text --
    def test_kinetic_text_presets_route(self, client):
        resp = client.get("/video/kinetic-text/presets")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "presets" in data
        assert len(data["presets"]) >= 10
