"""
Tests for OpenCut Motion Design & Animation (Category 79).

Covers:
  - Kinetic Typography (all 12 presets, easing, segmentation modes)
  - Data Animation (CSV/JSON loading, chart types, interpolation)
  - Shape Animation (morphing, stroke drawing, point resampling)
  - Expression Engine (sandbox functions, safety, timeout)
  - Particle System (emitter types, presets, physics, lifetime)
  - Motion Design Routes (smoke tests)
"""

import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


# ============================================================
# Kinetic Typography Tests
# ============================================================
class TestKineticTypography:
    """Tests for opencut.core.kinetic_typography module."""

    def test_list_presets_returns_12(self):
        from opencut.core.kinetic_typography import list_presets
        presets = list_presets()
        assert len(presets) == 12
        names = [p["name"] for p in presets]
        assert "bounce" in names
        assert "typewriter" in names

    def test_all_preset_names(self):
        from opencut.core.kinetic_typography import ANIMATION_PRESETS
        expected = {
            "bounce", "elastic", "typewriter", "wave", "cascade",
            "spiral", "explode", "assemble", "fade_in", "slide_up",
            "slide_left", "scale_pop",
        }
        assert set(ANIMATION_PRESETS.keys()) == expected

    def test_preset_functions_return_keyframes(self):
        from opencut.core.kinetic_typography import ANIMATION_PRESETS
        for name, fn in ANIMATION_PRESETS.items():
            kf = fn(0, 5, 3.0)
            assert "easing" in kf, f"{name} missing easing"
            assert "delay" in kf, f"{name} missing delay"
            assert "start" in kf, f"{name} missing start"
            assert "end" in kf, f"{name} missing end"
            assert "opacity" in kf["start"]
            assert "opacity" in kf["end"]

    def test_segment_text_char(self):
        from opencut.core.kinetic_typography import _segment_text
        result = _segment_text("Hi", "char")
        assert result == ["H", "i"]

    def test_segment_text_word(self):
        from opencut.core.kinetic_typography import _segment_text
        result = _segment_text("Hello World", "word")
        assert result == ["Hello", "World"]

    def test_segment_text_line(self):
        from opencut.core.kinetic_typography import _segment_text
        result = _segment_text("Line1\nLine2", "line")
        assert result == ["Line1", "Line2"]

    def test_segment_text_single_line(self):
        from opencut.core.kinetic_typography import _segment_text
        result = _segment_text("No newlines", "line")
        assert result == ["No newlines"]

    def test_easing_linear(self):
        from opencut.core.kinetic_typography import _ease_linear
        assert _ease_linear(0.0) == 0.0
        assert _ease_linear(0.5) == 0.5
        assert _ease_linear(1.0) == 1.0

    def test_easing_ease_in(self):
        from opencut.core.kinetic_typography import _ease_in
        assert _ease_in(0.0) == 0.0
        assert _ease_in(1.0) == 1.0
        assert _ease_in(0.5) < 0.5  # ease_in is slow at start

    def test_easing_ease_out(self):
        from opencut.core.kinetic_typography import _ease_out
        assert _ease_out(0.0) == 0.0
        assert _ease_out(1.0) == 1.0
        assert _ease_out(0.5) > 0.5  # ease_out is fast at start

    def test_easing_ease_in_out(self):
        from opencut.core.kinetic_typography import _ease_in_out
        assert _ease_in_out(0.0) == 0.0
        assert _ease_in_out(1.0) == 1.0

    def test_easing_bounce(self):
        from opencut.core.kinetic_typography import _ease_bounce
        assert _ease_bounce(0.0) == 0.0
        assert abs(_ease_bounce(1.0) - 1.0) < 0.01

    def test_easing_elastic(self):
        from opencut.core.kinetic_typography import _ease_elastic
        assert _ease_elastic(0.0) == 0.0
        assert _ease_elastic(1.0) == 1.0

    def test_easing_cubic_bezier(self):
        from opencut.core.kinetic_typography import _ease_cubic_bezier
        assert abs(_ease_cubic_bezier(0.0)) < 0.1
        assert abs(_ease_cubic_bezier(1.0) - 1.0) < 0.1

    def test_get_easing_known(self):
        from opencut.core.kinetic_typography import get_easing
        fn = get_easing("linear")
        assert fn(0.5) == 0.5

    def test_get_easing_unknown_defaults(self):
        from opencut.core.kinetic_typography import get_easing
        fn = get_easing("nonexistent")
        # Should default to ease_out
        assert fn(0.0) == 0.0

    def test_interpolate(self):
        from opencut.core.kinetic_typography import _interpolate
        assert _interpolate(0, 100, 0.5) == 50
        assert _interpolate(10, 20, 0.0) == 10
        assert _interpolate(10, 20, 1.0) == 20

    def test_get_element_transform_before_delay(self):
        from opencut.core.kinetic_typography import _get_element_transform
        kf = {
            "delay": 1.0,
            "anim_duration": 0.5,
            "start": {"opacity": 0.0, "y_offset": 100},
            "end": {"opacity": 1.0, "y_offset": 0},
            "easing": "linear",
        }
        tf = _get_element_transform(kf, 0.5, lambda t: t)
        assert tf["opacity"] == 0.0
        assert tf["y_offset"] == 100

    def test_get_element_transform_after_animation(self):
        from opencut.core.kinetic_typography import _get_element_transform
        kf = {
            "delay": 0.0,
            "anim_duration": 0.5,
            "start": {"opacity": 0.0},
            "end": {"opacity": 1.0},
        }
        tf = _get_element_transform(kf, 1.0, lambda t: t)
        assert tf["opacity"] == 1.0

    def test_parse_color(self):
        from opencut.core.kinetic_typography import _parse_color
        assert _parse_color("#FF0000") == (255, 0, 0)
        assert _parse_color("#00FF00") == (0, 255, 0)
        assert _parse_color("#0000FF") == (0, 0, 255)
        assert _parse_color("#FFF") == (255, 255, 255)

    def test_kinetic_result_to_dict(self):
        from opencut.core.kinetic_typography import KineticResult
        r = KineticResult(
            output_path="/tmp/test.mp4",
            frames_rendered=90,
            duration=3.0,
            preset_used="bounce",
        )
        d = r.to_dict()
        assert d["output_path"] == "/tmp/test.mp4"
        assert d["frames_rendered"] == 90
        assert d["preset_used"] == "bounce"

    def test_render_kinetic_empty_text_raises(self):
        from opencut.core.kinetic_typography import render_kinetic_text
        with pytest.raises(ValueError, match="empty"):
            render_kinetic_text(text="", preset="bounce")

    def test_render_kinetic_bad_preset_raises(self):
        from opencut.core.kinetic_typography import render_kinetic_text
        with pytest.raises(ValueError, match="Unknown preset"):
            render_kinetic_text(text="Hi", preset="nonexistent")

    def test_preview_empty_text_raises(self):
        from opencut.core.kinetic_typography import preview_frame
        with pytest.raises(ValueError, match="empty"):
            preview_frame(text="", preset="bounce")


# ============================================================
# Data Animation Tests
# ============================================================
class TestDataAnimation:
    """Tests for opencut.core.data_animation module."""

    def test_list_chart_types(self):
        from opencut.core.data_animation import list_chart_types
        types_list = list_chart_types()
        type_names = [t["type"] for t in types_list]
        assert "bar_chart" in type_names
        assert "line_chart" in type_names
        assert "counter" in type_names
        assert "label" in type_names
        assert "pie_chart" in type_names
        assert "progress_bar" in type_names

    def test_load_csv_data(self):
        from opencut.core.data_animation import load_data
        csv_str = "name,value\nAlpha,100\nBeta,200"
        rows = load_data(csv_str, "csv")
        assert len(rows) == 2
        assert rows[0]["name"] == "Alpha"
        assert rows[0]["value"] == 100.0

    def test_load_json_data_list(self):
        from opencut.core.data_animation import load_data
        json_str = '[{"name": "A", "value": 10}]'
        rows = load_data(json_str, "json")
        assert len(rows) == 1
        assert rows[0]["name"] == "A"

    def test_load_json_data_with_rows_key(self):
        from opencut.core.data_animation import load_data
        json_str = '{"rows": [{"x": 1}, {"x": 2}]}'
        rows = load_data(json_str, "json")
        assert len(rows) == 2

    def test_load_json_data_with_data_key(self):
        from opencut.core.data_animation import load_data
        json_str = '{"data": [{"x": 1}]}'
        rows = load_data(json_str, "json")
        assert len(rows) == 1

    def test_load_data_auto_detect_json(self):
        from opencut.core.data_animation import load_data
        json_str = '[{"a": 1}]'
        rows = load_data(json_str, "auto")
        assert len(rows) == 1

    def test_load_data_auto_detect_csv(self):
        from opencut.core.data_animation import load_data
        csv_str = "col1,col2\nval1,42"
        rows = load_data(csv_str, "auto")
        assert len(rows) == 1

    def test_load_data_empty_raises(self):
        from opencut.core.data_animation import load_data
        with pytest.raises(ValueError, match="empty"):
            load_data("", "auto")

    def test_load_data_from_file(self):
        from opencut.core.data_animation import load_data_from_file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                         delete=False) as f:
            f.write("name,val\nX,10\nY,20")
            path = f.name
        try:
            rows = load_data_from_file(path)
            assert len(rows) == 2
        finally:
            os.unlink(path)

    def test_load_data_from_missing_file(self):
        from opencut.core.data_animation import load_data_from_file
        with pytest.raises(FileNotFoundError):
            load_data_from_file("/nonexistent/file.csv")

    def test_resolve_binding(self):
        from opencut.core.data_animation import _resolve_binding
        row = {"revenue": 1000, "name": "Q1"}
        assert _resolve_binding("${data.revenue}", row) == 1000
        assert _resolve_binding("${revenue}", row) == 1000
        assert _resolve_binding("plain text", row) == "plain text"
        assert _resolve_binding("${data.missing}", row) == 0.0

    def test_resolve_element_bindings(self):
        from opencut.core.data_animation import _resolve_element_bindings
        elem = {"type": "counter", "value": "${data.count}", "label": "Items"}
        row = {"count": 42}
        resolved = _resolve_element_bindings(elem, row)
        assert resolved["value"] == 42
        assert resolved["label"] == "Items"
        assert resolved["type"] == "counter"

    def test_validate_template_valid(self):
        from opencut.core.data_animation import validate_template
        template = {
            "elements": [
                {"type": "bar_chart", "id": "bar1", "height": "${data.val}"}
            ]
        }
        data = [{"val": 100}]
        result = validate_template(template, data)
        assert result["valid"] is True

    def test_validate_template_unknown_type(self):
        from opencut.core.data_animation import validate_template
        template = {
            "elements": [{"type": "unknown_chart", "id": "x"}]
        }
        result = validate_template(template, [{"a": 1}])
        assert result["valid"] is False
        assert any("unknown type" in e for e in result["errors"])

    def test_validate_template_no_elements(self):
        from opencut.core.data_animation import validate_template
        result = validate_template({"elements": []}, [])
        assert result["valid"] is False

    def test_validate_template_missing_binding_warning(self):
        from opencut.core.data_animation import validate_template
        template = {
            "elements": [
                {"type": "counter", "id": "c", "value": "${data.missing_field}"}
            ]
        }
        data = [{"other_field": 1}]
        result = validate_template(template, data)
        assert result["valid"] is True
        assert len(result["warnings"]) > 0

    def test_interpolate_value_numeric(self):
        from opencut.core.data_animation import _interpolate_value
        assert _interpolate_value(0, 100, 0.5) == 50
        assert _interpolate_value(10, 20, 0.0) == 10

    def test_interpolate_value_string_crossfade(self):
        from opencut.core.data_animation import _interpolate_value
        assert _interpolate_value("A", "B", 0.3) == "A"
        assert _interpolate_value("A", "B", 0.7) == "B"

    def test_interpolate_elements(self):
        from opencut.core.data_animation import _interpolate_elements
        e1 = {"value": 0.0, "label": "Start"}
        e2 = {"value": 100.0, "label": "End"}
        result = _interpolate_elements(e1, e2, 0.5)
        assert result["value"] == 50.0

    def test_data_anim_result_to_dict(self):
        from opencut.core.data_animation import DataAnimResult
        r = DataAnimResult(
            output_path="/tmp/data.mp4",
            data_rows_rendered=5,
            elements_count=3,
            duration=10.0,
        )
        d = r.to_dict()
        assert d["data_rows_rendered"] == 5
        assert d["elements_count"] == 3

    def test_parse_color(self):
        from opencut.core.data_animation import _parse_color
        assert _parse_color("#FF0000") == (255, 0, 0)
        assert _parse_color(123) == (200, 200, 200)

    def test_auto_scale(self):
        from opencut.core.data_animation import _auto_scale
        lo, hi = _auto_scale([10, 20, 30])
        assert lo < 10
        assert hi > 30

    def test_auto_scale_empty(self):
        from opencut.core.data_animation import _auto_scale
        lo, hi = _auto_scale([])
        assert lo == 0.0
        assert hi == 100.0

    def test_render_data_animation_no_data_raises(self):
        from opencut.core.data_animation import render_data_animation
        with pytest.raises(ValueError, match="No data"):
            render_data_animation(template={"elements": [{"type": "counter"}]})

    def test_render_data_animation_no_elements_raises(self):
        from opencut.core.data_animation import render_data_animation
        with pytest.raises(ValueError, match="no elements"):
            render_data_animation(
                template={"elements": []},
                data=[{"a": 1}],
            )


# ============================================================
# Shape Animation Tests
# ============================================================
class TestShapeAnimation:
    """Tests for opencut.core.shape_animation module."""

    def test_list_shape_types(self):
        from opencut.core.shape_animation import list_shape_types
        types_list = list_shape_types()
        names = [t["type"] for t in types_list]
        assert "circle" in names
        assert "star" in names
        assert "custom_path" in names

    def test_list_animation_types(self):
        from opencut.core.shape_animation import list_animation_types
        types_list = list_animation_types()
        names = [t["type"] for t in types_list]
        assert "morph" in names
        assert "draw_stroke" in names
        assert "fill_fade" in names
        assert "scale_rotate" in names

    def test_generate_circle(self):
        from opencut.core.shape_animation import generate_circle
        pts = generate_circle(100, 100, 50, 32)
        assert len(pts) == 32
        for x, y in pts:
            dist = math.sqrt((x - 100) ** 2 + (y - 100) ** 2)
            assert abs(dist - 50) < 1.0

    def test_generate_rectangle(self):
        from opencut.core.shape_animation import generate_rectangle
        pts = generate_rectangle(0, 0, 200, 100, 40)
        assert len(pts) == 40

    def test_generate_rounded_rect(self):
        from opencut.core.shape_animation import generate_rounded_rect
        pts = generate_rounded_rect(0, 0, 200, 100, 10, 40)
        assert len(pts) == 40

    def test_generate_star(self):
        from opencut.core.shape_animation import generate_star
        pts = generate_star(200, 200, 100, 40, 5, 60)
        assert len(pts) >= 50

    def test_generate_polygon(self):
        from opencut.core.shape_animation import generate_polygon
        pts = generate_polygon(100, 100, 50, 6, 36)
        assert len(pts) == 36

    def test_generate_line(self):
        from opencut.core.shape_animation import generate_line
        pts = generate_line(0, 0, 100, 100, 20)
        assert len(pts) == 20
        assert pts[0] == (0, 0)

    def test_generate_arc(self):
        from opencut.core.shape_animation import generate_arc
        pts = generate_arc(100, 100, 50, 0, 180, 30)
        assert len(pts) == 30

    def test_resample_points(self):
        from opencut.core.shape_animation import _resample_points
        pts = [(0, 0), (100, 0), (100, 100)]
        resampled = _resample_points(pts, 10)
        assert len(resampled) == 10
        assert resampled[0] == (0, 0)

    def test_resample_single_point(self):
        from opencut.core.shape_animation import _resample_points
        resampled = _resample_points([(50, 50)], 5)
        assert len(resampled) == 5
        assert all(p == (50, 50) for p in resampled)

    def test_resample_empty(self):
        from opencut.core.shape_animation import _resample_points
        resampled = _resample_points([], 5)
        assert len(resampled) == 5

    def test_morph_shapes(self):
        from opencut.core.shape_animation import generate_circle, morph_shapes
        c1 = generate_circle(100, 100, 50, 16)
        c2 = generate_circle(200, 200, 80, 16)
        result = morph_shapes(c1, c2, 0.5, 16)
        assert len(result) == 16
        # Midpoint should be roughly between centers
        avg_x = sum(p[0] for p in result) / len(result)
        assert 120 < avg_x < 180

    def test_morph_shapes_t0(self):
        from opencut.core.shape_animation import morph_shapes
        s1 = [(0, 0), (10, 0), (10, 10)]
        s2 = [(100, 100), (110, 100), (110, 110)]
        result = morph_shapes(s1, s2, 0.0, 3)
        for (rx, ry), (sx, sy) in zip(result, [(0, 0), (10, 0), (10, 10)]):
            assert abs(rx - sx) < 2
            assert abs(ry - sy) < 2

    def test_morph_shapes_t1(self):
        from opencut.core.shape_animation import morph_shapes
        s1 = [(0, 0), (10, 0)]
        s2 = [(100, 100), (110, 100)]
        result = morph_shapes(s1, s2, 1.0, 2)
        assert abs(result[0][0] - 100) < 2
        assert abs(result[0][1] - 100) < 2

    def test_build_shape_circle(self):
        from opencut.core.shape_animation import build_shape
        pts = build_shape({"type": "circle", "cx": 100, "cy": 100, "radius": 50})
        assert len(pts) == 64

    def test_build_shape_star(self):
        from opencut.core.shape_animation import build_shape
        pts = build_shape({
            "type": "star", "cx": 200, "cy": 200,
            "outer_radius": 100, "inner_radius": 40, "spikes": 5,
        })
        assert len(pts) > 0

    def test_build_shape_polygon(self):
        from opencut.core.shape_animation import build_shape
        pts = build_shape({
            "type": "polygon", "cx": 100, "cy": 100,
            "radius": 50, "sides": 8,
        })
        assert len(pts) > 0

    def test_build_shape_unknown_defaults_circle(self):
        from opencut.core.shape_animation import build_shape
        pts = build_shape({"type": "unknown_type"})
        assert len(pts) == 64

    def test_parse_svg_path_simple(self):
        from opencut.core.shape_animation import parse_svg_path
        pts = parse_svg_path("M 0 0 L 100 0 L 100 100 Z", 20)
        assert len(pts) == 20

    def test_parse_color(self):
        from opencut.core.shape_animation import _parse_color
        assert _parse_color("#FF0000") == (255, 0, 0)

    def test_lerp_color(self):
        from opencut.core.shape_animation import _lerp_color
        c = _lerp_color((0, 0, 0), (255, 255, 255), 0.5)
        assert 120 < c[0] < 135

    def test_apply_transform_identity(self):
        from opencut.core.shape_animation import _apply_transform
        pts = [(10, 10), (20, 20)]
        result = _apply_transform(pts, scale=1.0, rotation=0.0)
        for (rx, ry), (ox, oy) in zip(result, pts):
            assert abs(rx - ox) < 0.01
            assert abs(ry - oy) < 0.01

    def test_apply_transform_scale(self):
        from opencut.core.shape_animation import _apply_transform
        pts = [(0, 0), (10, 0)]
        result = _apply_transform(pts, scale=2.0, center=(0, 0))
        assert abs(result[1][0] - 20) < 0.01

    def test_shape_anim_result_to_dict(self):
        from opencut.core.shape_animation import ShapeAnimResult
        r = ShapeAnimResult(
            output_path="/tmp/shape.mp4",
            frames=90,
            shapes_count=2,
        )
        d = r.to_dict()
        assert d["frames"] == 90
        assert d["shapes_count"] == 2

    def test_render_shape_no_shapes_raises(self):
        from opencut.core.shape_animation import render_shape_animation
        with pytest.raises(ValueError, match="At least one"):
            render_shape_animation(shapes=[])

    def test_render_shape_bad_animation_raises(self):
        from opencut.core.shape_animation import render_shape_animation
        with pytest.raises(ValueError, match="Unknown animation"):
            render_shape_animation(
                shapes=[{"type": "circle"}],
                animation="nonexistent",
            )

    def test_render_shape_morph_needs_two(self):
        from opencut.core.shape_animation import render_shape_animation
        with pytest.raises(ValueError, match="at least 2"):
            render_shape_animation(
                shapes=[{"type": "circle"}],
                animation="morph",
            )

    def test_path_length(self):
        from opencut.core.shape_animation import _path_length
        pts = [(0, 0), (3, 0), (3, 4)]
        length = _path_length(pts)
        assert abs(length - 7.0) < 0.01

    def test_easing_functions(self):
        from opencut.core.shape_animation import EASING
        for name, fn in EASING.items():
            assert fn(0.0) == 0.0, f"{name}(0) != 0"
            assert fn(1.0) == 1.0, f"{name}(1) != 1"


# ============================================================
# Expression Engine Tests
# ============================================================
class TestExpressionEngine:
    """Tests for opencut.core.expression_engine module."""

    def test_evaluate_simple_math(self):
        from opencut.core.expression_engine import evaluate_expression
        assert evaluate_expression("2 + 3") == 5.0

    def test_evaluate_with_time(self):
        from opencut.core.expression_engine import (
            ExpressionContext,
            evaluate_expression,
        )
        ctx = ExpressionContext(time=1.5)
        result = evaluate_expression("time * 2", ctx)
        assert result == 3.0

    def test_evaluate_with_frame(self):
        from opencut.core.expression_engine import (
            ExpressionContext,
            evaluate_expression,
        )
        ctx = ExpressionContext(frame=10)
        result = evaluate_expression("frame + 5", ctx)
        assert result == 15.0

    def test_evaluate_sin(self):
        from opencut.core.expression_engine import evaluate_expression
        result = evaluate_expression("sin(0)")
        assert abs(result) < 0.001

    def test_evaluate_cos(self):
        from opencut.core.expression_engine import evaluate_expression
        result = evaluate_expression("cos(0)")
        assert abs(result - 1.0) < 0.001

    def test_evaluate_lerp(self):
        from opencut.core.expression_engine import evaluate_expression
        result = evaluate_expression("lerp(0, 100, 0.5)")
        assert result == 50.0

    def test_evaluate_clamp(self):
        from opencut.core.expression_engine import evaluate_expression
        assert evaluate_expression("clamp(1.5, 0, 1)") == 1.0
        assert evaluate_expression("clamp(-0.5, 0, 1)") == 0.0
        assert evaluate_expression("clamp(0.5, 0, 1)") == 0.5

    def test_evaluate_noise(self):
        from opencut.core.expression_engine import evaluate_expression
        result = evaluate_expression("noise(1.0)")
        assert -1.0 <= result <= 1.0

    def test_evaluate_pi_constant(self):
        from opencut.core.expression_engine import evaluate_expression
        result = evaluate_expression("pi")
        assert abs(result - math.pi) < 0.001

    def test_evaluate_random_seeded(self):
        from opencut.core.expression_engine import (
            ExpressionContext,
            evaluate_expression,
        )
        ctx1 = ExpressionContext(frame=5, seed=42)
        ctx2 = ExpressionContext(frame=5, seed=42)
        r1 = evaluate_expression("random()", ctx1)
        r2 = evaluate_expression("random()", ctx2)
        assert r1 == r2

    def test_evaluate_audio_amplitude(self):
        from opencut.core.expression_engine import (
            ExpressionContext,
            evaluate_expression,
        )
        ctx = ExpressionContext(audio_amplitude=0.8)
        result = evaluate_expression("audio_amplitude", ctx)
        assert result == 0.8

    def test_evaluate_beat(self):
        from opencut.core.expression_engine import (
            ExpressionContext,
            evaluate_expression,
        )
        ctx = ExpressionContext(beat=True)
        result = evaluate_expression("1 if beat else 0", ctx)
        assert result == 1.0

    def test_evaluate_custom_vars(self):
        from opencut.core.expression_engine import (
            ExpressionContext,
            evaluate_expression,
        )
        ctx = ExpressionContext(custom_vars={"my_val": 42})
        result = evaluate_expression("my_val + 8", ctx)
        assert result == 50.0

    def test_safety_blocks_import(self):
        from opencut.core.expression_engine import evaluate_expression
        with pytest.raises(ValueError, match="Unsafe|Forbidden"):
            evaluate_expression("__import__('os')")

    def test_safety_blocks_attribute_access(self):
        from opencut.core.expression_engine import _check_ast_safety
        error = _check_ast_safety("x.__class__")
        assert error is not None

    def test_safety_blocks_builtins(self):
        from opencut.core.expression_engine import _check_ast_safety
        error = _check_ast_safety("x.__builtins__")
        assert error is not None

    def test_safety_blocks_eval(self):
        from opencut.core.expression_engine import _check_ast_safety
        error = _check_ast_safety("eval('1+1')")
        assert error is not None

    def test_safety_blocks_exec(self):
        from opencut.core.expression_engine import _check_ast_safety
        error = _check_ast_safety("exec('x=1')")
        assert error is not None

    def test_safety_blocks_open(self):
        from opencut.core.expression_engine import _check_ast_safety
        error = _check_ast_safety("open('/etc/passwd')")
        assert error is not None

    def test_safety_allows_math(self):
        from opencut.core.expression_engine import _check_ast_safety
        assert _check_ast_safety("sin(time * 3) + cos(frame)") is None

    def test_safety_allows_conditionals(self):
        from opencut.core.expression_engine import _check_ast_safety
        assert _check_ast_safety("1.0 if beat else 0.0") is None

    def test_empty_expression_raises(self):
        from opencut.core.expression_engine import evaluate_expression
        with pytest.raises(ValueError, match="empty"):
            evaluate_expression("")

    def test_syntax_error_raises(self):
        from opencut.core.expression_engine import evaluate_expression
        with pytest.raises(ValueError, match="Unsafe|Syntax"):
            evaluate_expression("def foo(): pass")

    def test_noise_rejects_excessive_octaves(self):
        from opencut.core.expression_engine import evaluate_expression
        with pytest.raises(ValueError, match="octaves"):
            evaluate_expression("noise(1.0, octaves=10**8)", timeout_ms=50)

    def test_evaluate_timeline(self):
        from opencut.core.expression_engine import evaluate_timeline
        result = evaluate_timeline("time * 10", fps=10, duration=1.0)
        assert len(result.values) == 10
        assert result.values[0] == 0.0
        assert abs(result.values[-1] - 9.0) < 0.01

    def test_evaluate_timeline_min_max(self):
        from opencut.core.expression_engine import evaluate_timeline
        result = evaluate_timeline("frame", fps=10, duration=1.0)
        assert result.min_value == 0.0
        assert result.max_value == 9.0

    def test_evaluate_timeline_with_errors(self):
        from opencut.core.expression_engine import evaluate_timeline
        # division by zero on frame 0
        result = evaluate_timeline("1 / frame", fps=5, duration=1.0)
        assert len(result.errors) > 0

    def test_expression_result_to_dict(self):
        from opencut.core.expression_engine import ExpressionResult
        r = ExpressionResult(
            values=[1.0, 2.0, 3.0],
            min_value=1.0,
            max_value=3.0,
        )
        d = r.to_dict()
        assert d["min"] == 1.0
        assert d["max"] == 3.0
        assert len(d["values"]) == 3

    def test_create_context(self):
        from opencut.core.expression_engine import create_context
        ctx = create_context(time=1.0, frame=30, fps=30.0, my_var=99)
        assert ctx.time == 1.0
        assert ctx.frame == 30
        assert ctx.custom_vars["my_var"] == 99

    def test_evaluate_multi(self):
        from opencut.core.expression_engine import (
            ExpressionContext,
            evaluate_multi,
        )
        ctx = ExpressionContext(time=1.0)
        results = evaluate_multi({
            "scale": "1 + sin(time)",
            "opacity": "clamp(time, 0, 1)",
        }, ctx)
        assert "scale" in results
        assert "opacity" in results
        assert results["opacity"] == 1.0

    def test_smoothstep(self):
        from opencut.core.expression_engine import _smoothstep
        assert _smoothstep(0, 1, 0.0) == 0.0
        assert _smoothstep(0, 1, 1.0) == 1.0
        assert 0.4 < _smoothstep(0, 1, 0.5) < 0.6

    def test_remap(self):
        from opencut.core.expression_engine import _remap
        assert _remap(5, 0, 10, 0, 100) == 50.0

    def test_ping_pong(self):
        from opencut.core.expression_engine import _ping_pong
        assert _ping_pong(0.0, 1.0) == 0.0
        assert _ping_pong(1.0, 1.0) == 1.0
        assert _ping_pong(1.5, 1.0) == 0.5

    def test_step(self):
        from opencut.core.expression_engine import _step
        assert _step(0.5, 0.3) == 0.0
        assert _step(0.5, 0.7) == 1.0

    def test_pulse(self):
        from opencut.core.expression_engine import _pulse
        assert _pulse(0.0, 1.0, 0.5) == 1.0
        assert _pulse(0.6, 1.0, 0.5) == 0.0

    def test_noise_1d(self):
        from opencut.core.expression_engine import _noise_1d
        val = _noise_1d(0.5, 42)
        assert -1.0 <= val <= 1.0

    def test_noise_2d(self):
        from opencut.core.expression_engine import _noise_2d
        val = _noise_2d(0.5, 0.5, 42)
        assert -1.0 <= val <= 1.0

    def test_noise_octaves(self):
        from opencut.core.expression_engine import noise
        val = noise(1.0, 0.0, octaves=3, seed=42)
        assert -1.0 <= val <= 1.0

    def test_list_functions(self):
        from opencut.core.expression_engine import list_functions
        funcs = list_functions()
        names = [f["name"] for f in funcs]
        assert "sin" in names
        assert "lerp" in names
        assert "noise" in names

    def test_context_to_globals_has_no_real_builtins(self):
        from opencut.core.expression_engine import ExpressionContext
        ctx = ExpressionContext()
        globs = ctx.to_globals()
        assert globs["__builtins__"] == {}


# ============================================================
# Particle System Tests
# ============================================================
class TestParticleSystem:
    """Tests for opencut.core.particle_system module."""

    def test_list_presets(self):
        from opencut.core.particle_system import list_presets
        presets = list_presets()
        names = [p["name"] for p in presets]
        assert "snow" in names
        assert "rain" in names
        assert "confetti" in names
        assert "fire_sparks" in names
        assert "dust" in names
        assert "bubbles" in names
        assert "magic_sparkle" in names
        assert "smoke_rising" in names

    def test_list_emitter_types(self):
        from opencut.core.particle_system import list_emitter_types
        types_list = list_emitter_types()
        names = [t["type"] for t in types_list]
        assert "point" in names
        assert "line" in names
        assert "circle" in names
        assert "rectangle" in names
        assert "burst" in names

    def test_list_sprite_types(self):
        from opencut.core.particle_system import list_sprite_types
        types_list = list_sprite_types()
        names = [t["type"] for t in types_list]
        assert "circle" in names
        assert "star" in names
        assert "snowflake" in names
        assert "smoke" in names

    def test_particle_init(self):
        from opencut.core.particle_system import Particle
        p = Particle()
        assert p.alive is True
        assert p.age == 0.0
        assert p.opacity == 1.0

    def test_particle_update_ages(self):
        from opencut.core.particle_system import Particle
        p = Particle()
        p.lifetime = 1.0
        p.update(0.5, 0, 0, 0, 0, 42, None)
        assert p.alive is True
        assert abs(p.age - 0.5) < 0.01

    def test_particle_dies_after_lifetime(self):
        from opencut.core.particle_system import Particle
        p = Particle()
        p.lifetime = 1.0
        p.update(1.5, 0, 0, 0, 0, 42, None)
        assert p.alive is False

    def test_particle_gravity(self):
        from opencut.core.particle_system import Particle
        p = Particle()
        p.lifetime = 10.0
        p.vy = 0.0
        p.update(1.0, 100.0, 0, 0, 0, 42, None)
        assert p.vy > 0  # Gravity pulls down (positive Y)

    def test_particle_wind(self):
        from opencut.core.particle_system import Particle
        p = Particle()
        p.lifetime = 10.0
        p.vx = 0.0
        p.update(1.0, 0, 50.0, 0, 0, 42, None)
        assert p.vx > 0  # Wind pushes right

    def test_particle_bounce(self):
        from opencut.core.particle_system import Particle
        p = Particle()
        p.lifetime = 10.0
        p.x = -10
        p.vx = -50
        bounds = (0, 0, 1920, 1080)
        p.update(0.01, 0, 0, 0, 0, 42, bounds)
        assert p.x >= 0
        assert p.vx > 0  # Reversed

    def test_particle_opacity_interpolation(self):
        from opencut.core.particle_system import Particle
        p = Particle()
        p.lifetime = 2.0
        p.opacity_start = 1.0
        p.opacity_end = 0.0
        p.update(1.0, 0, 0, 0, 0, 42, None)
        assert p.opacity < 1.0
        assert p.opacity > 0.0

    def test_particle_size_interpolation(self):
        from opencut.core.particle_system import Particle
        p = Particle()
        p.lifetime = 2.0
        p.size_start = 10.0
        p.size_end = 0.0
        p.update(1.0, 0, 0, 0, 0, 42, None)
        assert p.size < 10.0
        assert p.size > 0.0

    def test_emitter_point_spawn(self):
        from opencut.core.particle_system import ParticleEmitter
        em = ParticleEmitter(
            emitter_type="point",
            position=(500, 500),
            emit_rate=100,
        )
        em.update(0.1, (1920, 1080))
        assert len(em.particles) > 0
        assert em.total_spawned > 0

    def test_emitter_line_spawn(self):
        from opencut.core.particle_system import ParticleEmitter
        em = ParticleEmitter(
            emitter_type="line",
            position=(0, 0),
            line_end=(100, 0),
            emit_rate=100,
        )
        em.update(0.1, (1920, 1080))
        assert len(em.particles) > 0
        # All particles should be between x=0 and x=100
        for p in em.particles:
            assert -10 <= p.x <= 110  # Allow small drift

    def test_emitter_circle_spawn(self):
        from opencut.core.particle_system import ParticleEmitter
        em = ParticleEmitter(
            emitter_type="circle",
            position=(500, 500),
            circle_radius=50,
            emit_rate=100,
        )
        em.update(0.1, (1920, 1080))
        assert len(em.particles) > 0

    def test_emitter_rectangle_spawn(self):
        from opencut.core.particle_system import ParticleEmitter
        em = ParticleEmitter(
            emitter_type="rectangle",
            position=(500, 500),
            rect_size=(200, 100),
            emit_rate=100,
        )
        em.update(0.1, (1920, 1080))
        assert len(em.particles) > 0

    def test_emitter_burst(self):
        from opencut.core.particle_system import ParticleEmitter
        em = ParticleEmitter(
            emitter_type="burst",
            position=(500, 500),
            burst_count=30,
        )
        em.update(0.1, (1920, 1080))
        assert em.total_spawned == 30
        # Second update should NOT spawn more
        em.update(0.1, (1920, 1080))
        assert em.total_spawned == 30

    def test_emitter_max_particles(self):
        from opencut.core.particle_system import ParticleEmitter
        em = ParticleEmitter(
            emitter_type="point",
            emit_rate=10000,
            max_particles=50,
        )
        em.update(1.0, (1920, 1080))
        assert len(em.particles) <= 50

    def test_emitter_peak_active(self):
        from opencut.core.particle_system import ParticleEmitter
        em = ParticleEmitter(
            emitter_type="burst",
            burst_count=20,
        )
        em.update(0.1, (1920, 1080))
        assert em.peak_active == 20

    def test_build_emitter_from_preset_snow(self):
        from opencut.core.particle_system import _build_emitter_from_preset
        em = _build_emitter_from_preset("snow", (1920, 1080))
        assert em.sprite_type == "snowflake"
        assert em.gravity > 0

    def test_build_emitter_from_preset_unknown(self):
        from opencut.core.particle_system import _build_emitter_from_preset
        with pytest.raises(ValueError, match="Unknown preset"):
            _build_emitter_from_preset("nonexistent", (1920, 1080))

    def test_build_emitter_with_overrides(self):
        from opencut.core.particle_system import _build_emitter_from_preset
        em = _build_emitter_from_preset("snow", (1920, 1080),
                                         emit_rate=999)
        assert em.emit_rate == 999

    def test_particle_result_to_dict(self):
        from opencut.core.particle_system import ParticleResult
        r = ParticleResult(
            output_path="/tmp/particles.mp4",
            frames_rendered=150,
            total_particles_spawned=500,
            peak_active_particles=120,
        )
        d = r.to_dict()
        assert d["total_particles_spawned"] == 500
        assert d["peak_active_particles"] == 120

    def test_render_particles_no_preset_no_config_raises(self):
        from opencut.core.particle_system import render_particles
        with pytest.raises(ValueError, match="Either preset"):
            render_particles()

    def test_preview_no_preset_no_config_raises(self):
        from opencut.core.particle_system import preview_frame
        with pytest.raises(ValueError, match="Either preset"):
            preview_frame()

    def test_noise_2d_range(self):
        from opencut.core.particle_system import _noise_2d
        for x in range(10):
            val = _noise_2d(x * 0.1, 0.5, 42)
            assert -1.0 <= val <= 1.0


# ============================================================
# Motion Design Routes Smoke Tests
# ============================================================
class TestMotionDesignRoutes:
    """Smoke tests for the motion_design_bp Flask blueprint."""

    @pytest.fixture
    def app(self):
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        cfg = OpenCutConfig()
        flask_app = create_app(config=cfg)
        flask_app.config["TESTING"] = True
        return flask_app

    @pytest.fixture
    def client(self, app):
        return app.test_client()

    @pytest.fixture
    def csrf_token(self, client):
        resp = client.get("/health")
        data = resp.get_json()
        return data.get("csrf_token", "")

    def _headers(self, token):
        return {
            "X-OpenCut-Token": token,
            "Content-Type": "application/json",
        }

    def test_kinetic_text_presets_get(self, client):
        resp = client.get("/api/motion/kinetic-text/presets")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "presets" in data
        assert len(data["presets"]) == 12

    def test_kinetic_text_presets_primary_path_get(self, client):
        resp = client.get("/motion/kinetic-text/presets")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "presets" in data

    def test_shape_animate_types_get(self, client):
        resp = client.get("/api/motion/shape-animate/types")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "shape_types" in data
        assert "animation_types" in data

    def test_particles_presets_get(self, client):
        resp = client.get("/api/motion/particles/presets")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "presets" in data

    def test_kinetic_text_no_csrf(self, client):
        resp = client.post(
            "/api/motion/kinetic-text",
            json={"text": "Hello"},
        )
        assert resp.status_code == 403

    def test_kinetic_text_no_text(self, client, csrf_token):
        resp = client.post(
            "/api/motion/kinetic-text",
            json={"text": ""},
            headers=self._headers(csrf_token),
        )
        # Async job will be created, error will be in job status
        # or it may reject synchronously
        assert resp.status_code in (200, 400)

    def test_data_animation_validate_no_template(self, client, csrf_token):
        resp = client.post(
            "/api/motion/data-animation/validate",
            json={},
            headers=self._headers(csrf_token),
        )
        assert resp.status_code == 400

    def test_data_animation_validate_valid(self, client, csrf_token):
        resp = client.post(
            "/api/motion/data-animation/validate",
            json={
                "template": {
                    "elements": [
                        {"type": "counter", "id": "c1", "value": "${data.v}"}
                    ]
                },
                "data": [{"v": 100}],
            },
            headers=self._headers(csrf_token),
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["valid"] is True

    def test_expression_evaluate_no_csrf(self, client):
        resp = client.post(
            "/api/motion/expression/evaluate",
            json={"expression": "1+1"},
        )
        assert resp.status_code == 403

    def test_particles_no_csrf(self, client):
        resp = client.post(
            "/api/motion/particles",
            json={"preset": "snow"},
        )
        assert resp.status_code == 403
