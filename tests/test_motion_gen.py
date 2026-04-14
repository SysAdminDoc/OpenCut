"""
Tests for OpenCut Motion & Generation features.

Covers:
  - Generative Extend (all directions, fallback methods, edge cases)
  - Green-Screen-Free Background Replacement (all background types, methods)
  - Consistent Character Generation (profile CRUD, generation)
  - Motion Brush (directions, mask modes, still vs video)
  - Motion/Gen routes (smoke tests)
"""

import json
import os
import sys
import tempfile
from unittest.mock import patch

import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ===================================================================
# Generative Extend - Dataclasses
# ===================================================================
class TestExtendResult:
    def test_default_result(self):
        from opencut.core.generative_extend import ExtendResult
        r = ExtendResult()
        assert r.output_path == ""
        assert r.original_duration == 0.0
        assert r.extended_duration == 0.0
        assert r.frames_generated == 0
        assert r.model_used == ""

    def test_populated_result(self):
        from opencut.core.generative_extend import ExtendResult
        r = ExtendResult(
            output_path="/out.mp4",
            original_duration=10.0,
            extended_duration=12.0,
            frames_generated=60,
            model_used="freeze",
        )
        assert r.extended_duration == 12.0
        assert r.frames_generated == 60

    def test_to_dict(self):
        from opencut.core.generative_extend import ExtendResult
        r = ExtendResult(output_path="/out.mp4", model_used="flow")
        d = r.to_dict()
        assert isinstance(d, dict)
        assert d["output_path"] == "/out.mp4"
        assert d["model_used"] == "flow"


# ===================================================================
# Generative Extend - Validation
# ===================================================================
class TestExtendClipValidation:
    def test_missing_file_raises(self):
        from opencut.core.generative_extend import extend_clip
        with pytest.raises(FileNotFoundError):
            extend_clip("/nonexistent/video.mp4")

    def test_invalid_direction_raises(self):
        from opencut.core.generative_extend import extend_clip
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            with pytest.raises(ValueError, match="Invalid direction"):
                extend_clip(path, direction="sideways")
        finally:
            os.unlink(path)

    def test_invalid_model_raises(self):
        from opencut.core.generative_extend import extend_clip
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            with pytest.raises(ValueError, match="Invalid model"):
                extend_clip(path, model="nonexistent_model")
        finally:
            os.unlink(path)


# ===================================================================
# Generative Extend - Constants
# ===================================================================
class TestExtendConstants:
    def test_valid_directions(self):
        from opencut.core.generative_extend import VALID_DIRECTIONS
        assert "forward" in VALID_DIRECTIONS
        assert "backward" in VALID_DIRECTIONS
        assert "both" in VALID_DIRECTIONS

    def test_valid_models(self):
        from opencut.core.generative_extend import VALID_MODELS
        assert "auto" in VALID_MODELS
        assert "freeze" in VALID_MODELS
        assert "flow" in VALID_MODELS


# ===================================================================
# Generative Extend - Model Detection
# ===================================================================
class TestModelDetection:
    @patch("importlib.import_module", side_effect=ImportError)
    def test_fallback_to_freeze(self, _mock):
        from opencut.core.generative_extend import _detect_available_model
        result = _detect_available_model()
        assert result == "freeze"

    def test_detect_returns_valid_model(self):
        from opencut.core.generative_extend import VALID_MODELS, _detect_available_model
        result = _detect_available_model()
        assert result in VALID_MODELS or result in ("wan2.1", "flow", "freeze")


# ===================================================================
# Generative Extend - Freeze Frame
# ===================================================================
class TestFreezeFrame:
    @patch("opencut.core.generative_extend.run_ffmpeg")
    @patch("opencut.core.generative_extend.get_video_info")
    def test_freeze_forward(self, mock_info, mock_ffmpeg):
        from opencut.core.generative_extend import _generate_via_freeze
        mock_info.return_value = {"duration": 10.0, "width": 1920, "height": 1080, "fps": 30.0}
        mock_ffmpeg.return_value = ""

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        fd, out = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        try:
            _generate_via_freeze(path, 2.0, "forward", 30.0, out)
            assert mock_ffmpeg.call_count == 2  # extract frame + zoompan
        finally:
            os.unlink(path)
            if os.path.exists(out):
                os.unlink(out)

    @patch("opencut.core.generative_extend.run_ffmpeg")
    @patch("opencut.core.generative_extend.get_video_info")
    def test_freeze_backward(self, mock_info, mock_ffmpeg):
        from opencut.core.generative_extend import _generate_via_freeze
        mock_info.return_value = {"duration": 10.0, "width": 1920, "height": 1080, "fps": 30.0}
        mock_ffmpeg.return_value = ""

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        fd, out = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        try:
            _generate_via_freeze(path, 1.5, "backward", 30.0, out)
            assert mock_ffmpeg.call_count == 2
        finally:
            os.unlink(path)
            if os.path.exists(out):
                os.unlink(out)


# ===================================================================
# Generative Extend - Full Pipeline
# ===================================================================
class TestExtendClipPipeline:
    @patch("opencut.core.generative_extend.run_ffmpeg")
    @patch("opencut.core.generative_extend.get_video_info")
    def test_extend_forward_freeze(self, mock_info, mock_ffmpeg):
        from opencut.core.generative_extend import extend_clip
        mock_info.return_value = {"duration": 5.0, "width": 1920, "height": 1080, "fps": 30.0}
        mock_ffmpeg.return_value = ""

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            result = extend_clip(path, extend_seconds=2.0, direction="forward", model="freeze")
            assert result.model_used == "freeze"
            assert result.original_duration == 5.0
            assert result.extended_duration == 7.0
        finally:
            os.unlink(path)

    @patch("opencut.core.generative_extend.run_ffmpeg")
    @patch("opencut.core.generative_extend.get_video_info")
    def test_extend_both_freeze(self, mock_info, mock_ffmpeg):
        from opencut.core.generative_extend import extend_clip
        mock_info.return_value = {"duration": 5.0, "width": 1920, "height": 1080, "fps": 30.0}
        mock_ffmpeg.return_value = ""

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            result = extend_clip(path, extend_seconds=1.0, direction="both", model="freeze")
            assert result.model_used == "freeze"
            assert result.extended_duration == 7.0  # 5 + 1 + 1
        finally:
            os.unlink(path)

    @patch("opencut.core.generative_extend.run_ffmpeg")
    @patch("opencut.core.generative_extend.get_video_info")
    def test_extend_clamps_duration(self, mock_info, mock_ffmpeg):
        from opencut.core.generative_extend import extend_clip
        mock_info.return_value = {"duration": 5.0, "width": 1920, "height": 1080, "fps": 30.0}
        mock_ffmpeg.return_value = ""

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            # Requesting 100s should be clamped to 30s
            result = extend_clip(path, extend_seconds=100.0, model="freeze")
            assert result.extended_duration == 35.0  # 5 + 30
        finally:
            os.unlink(path)

    @patch("opencut.core.generative_extend.run_ffmpeg")
    @patch("opencut.core.generative_extend.get_video_info")
    def test_progress_callback_called(self, mock_info, mock_ffmpeg):
        from opencut.core.generative_extend import extend_clip
        mock_info.return_value = {"duration": 5.0, "width": 1920, "height": 1080, "fps": 30.0}
        mock_ffmpeg.return_value = ""

        progress_calls = []

        def on_prog(pct, msg=""):
            progress_calls.append((pct, msg))

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            extend_clip(path, model="freeze", on_progress=on_prog)
            assert len(progress_calls) >= 3
        finally:
            os.unlink(path)


# ===================================================================
# Green-Screen-Free - Dataclasses
# ===================================================================
class TestBGReplaceResult:
    def test_default_result(self):
        from opencut.core.greenscreen_free import BGReplaceResult
        r = BGReplaceResult()
        assert r.output_path == ""
        assert r.method_used == ""
        assert r.frames_processed == 0
        assert r.avg_confidence == 0.0

    def test_populated_result(self):
        from opencut.core.greenscreen_free import BGReplaceResult
        r = BGReplaceResult(
            output_path="/out.mp4",
            method_used="rembg",
            frames_processed=300,
            avg_confidence=0.92,
        )
        assert r.frames_processed == 300

    def test_to_dict(self):
        from opencut.core.greenscreen_free import BGReplaceResult
        r = BGReplaceResult(method_used="mediapipe")
        d = r.to_dict()
        assert isinstance(d, dict)
        assert d["method_used"] == "mediapipe"


# ===================================================================
# Green-Screen-Free - Validation
# ===================================================================
class TestBGReplaceValidation:
    def test_missing_file_raises(self):
        from opencut.core.greenscreen_free import replace_background
        with pytest.raises(FileNotFoundError):
            replace_background("/nonexistent/video.mp4")

    def test_invalid_method_raises(self):
        from opencut.core.greenscreen_free import replace_background
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            with pytest.raises(ValueError, match="Invalid method"):
                replace_background(path, method="deepfake")
        finally:
            os.unlink(path)


# ===================================================================
# Green-Screen-Free - Helpers
# ===================================================================
class TestBGHelpers:
    def test_parse_hex_color_6digit(self):
        from opencut.core.greenscreen_free import _parse_hex_color
        assert _parse_hex_color("#FF0000") == (0, 0, 255)  # BGR

    def test_parse_hex_color_3digit(self):
        from opencut.core.greenscreen_free import _parse_hex_color
        assert _parse_hex_color("#F00") == (0, 0, 255)

    def test_parse_hex_color_no_hash(self):
        from opencut.core.greenscreen_free import _parse_hex_color
        assert _parse_hex_color("00FF00") == (0, 255, 0)

    def test_parse_hex_color_invalid_raises(self):
        from opencut.core.greenscreen_free import _parse_hex_color
        with pytest.raises(ValueError):
            _parse_hex_color("GGGGGG")

    def test_is_hex_color_true(self):
        from opencut.core.greenscreen_free import _is_hex_color
        assert _is_hex_color("#FF0000") is True
        assert _is_hex_color("AABBCC") is True
        assert _is_hex_color("#ABC") is True

    def test_is_hex_color_false(self):
        from opencut.core.greenscreen_free import _is_hex_color
        assert _is_hex_color("blur") is False
        assert _is_hex_color("none") is False
        assert _is_hex_color("/path/to/file.png") is False
        assert _is_hex_color("") is False

    def test_detect_method_returns_valid(self):
        from opencut.core.greenscreen_free import VALID_METHODS, _detect_available_method
        result = _detect_available_method()
        assert result in VALID_METHODS or result in ("rembg", "mediapipe", "sam2")

    def test_valid_methods_constant(self):
        from opencut.core.greenscreen_free import VALID_METHODS
        assert "auto" in VALID_METHODS
        assert "rembg" in VALID_METHODS
        assert "mediapipe" in VALID_METHODS
        assert "sam2" in VALID_METHODS

    def test_valid_bg_keywords(self):
        from opencut.core.greenscreen_free import VALID_BG_KEYWORDS
        assert "blur" in VALID_BG_KEYWORDS
        assert "none" in VALID_BG_KEYWORDS


# ===================================================================
# Green-Screen-Free - Compositing
# ===================================================================
class TestCompositing:
    def test_refine_matte(self):
        from opencut.core.greenscreen_free import _refine_matte
        import numpy as np
        alpha = np.full((100, 100), 255, dtype=np.uint8)
        refined = _refine_matte(alpha, edge_blur=2)
        assert refined.shape == (100, 100)

    def test_temporal_smooth_single(self):
        from opencut.core.greenscreen_free import _temporal_smooth_matte
        import numpy as np
        buf = []
        matte = np.full((50, 50), 200, dtype=np.uint8)
        result = _temporal_smooth_matte(buf, matte)
        assert result.shape == (50, 50)
        assert len(buf) == 1

    def test_temporal_smooth_multiple(self):
        from opencut.core.greenscreen_free import _temporal_smooth_matte
        import numpy as np
        buf = []
        for i in range(6):
            matte = np.full((50, 50), 100 + i * 20, dtype=np.uint8)
            _temporal_smooth_matte(buf, matte)
        # Buffer should be capped at _TEMPORAL_WINDOW
        assert len(buf) <= 5

    def test_composite_frame(self):
        from opencut.core.greenscreen_free import _composite_frame
        import numpy as np
        fg = np.full((100, 100, 3), 255, dtype=np.uint8)  # White FG
        bg = np.zeros((100, 100, 3), dtype=np.uint8)       # Black BG
        alpha = np.full((100, 100), 128, dtype=np.uint8)   # 50% blend
        result = _composite_frame(fg, alpha, bg)
        assert result.shape == (100, 100, 3)
        # Should be roughly half brightness
        assert 100 < result[50, 50, 0] < 160


# ===================================================================
# Character Consistency - Dataclasses
# ===================================================================
class TestCharacterProfile:
    def test_default_profile(self):
        from opencut.core.character_consistency import CharacterProfile
        p = CharacterProfile()
        assert p.profile_id == ""
        assert p.name == ""
        assert p.num_references == 0

    def test_populated_profile(self):
        from opencut.core.character_consistency import CharacterProfile
        p = CharacterProfile(
            profile_id="abc123",
            name="hero",
            num_references=5,
            created_at="2025-01-01T00:00:00Z",
        )
        assert p.profile_id == "abc123"
        assert p.name == "hero"

    def test_profile_to_dict(self):
        from opencut.core.character_consistency import CharacterProfile
        p = CharacterProfile(profile_id="x", name="test")
        d = p.to_dict()
        assert isinstance(d, dict)
        assert d["profile_id"] == "x"


class TestGenerationResult:
    def test_default_result(self):
        from opencut.core.character_consistency import GenerationResult
        r = GenerationResult()
        assert r.output_path == ""
        assert r.prompt == ""
        assert r.duration == 0.0

    def test_populated_result(self):
        from opencut.core.character_consistency import GenerationResult
        r = GenerationResult(
            output_path="/out.mp4",
            prompt="hero walks through city",
            character_id="abc",
            duration=4.0,
            model_used="placeholder",
        )
        assert r.model_used == "placeholder"

    def test_generation_to_dict(self):
        from opencut.core.character_consistency import GenerationResult
        r = GenerationResult(prompt="test scene")
        d = r.to_dict()
        assert d["prompt"] == "test scene"


# ===================================================================
# Character Consistency - Validation
# ===================================================================
class TestCharacterValidation:
    def test_create_empty_images_raises(self):
        from opencut.core.character_consistency import create_character_profile
        with pytest.raises(ValueError, match="At least one"):
            create_character_profile([])

    def test_create_no_valid_images_raises(self):
        from opencut.core.character_consistency import create_character_profile
        with pytest.raises(ValueError, match="No valid"):
            create_character_profile(["/nonexistent/img1.png", "/nonexistent/img2.png"])

    def test_generate_empty_prompt_raises(self):
        from opencut.core.character_consistency import (
            CharacterProfile,
            generate_consistent_scene,
        )
        profile = CharacterProfile(profile_id="test", name="test")
        with pytest.raises(ValueError, match="prompt is required"):
            generate_consistent_scene("", profile)

    def test_load_nonexistent_profile_raises(self):
        from opencut.core.character_consistency import load_character_profile
        with pytest.raises(FileNotFoundError):
            load_character_profile("nonexistent_profile_id_12345")


# ===================================================================
# Character Consistency - Profile CRUD
# ===================================================================
class TestCharacterProfileCRUD:
    @patch("opencut.core.character_consistency._extract_embeddings_insightface", return_value={})
    @patch("opencut.core.character_consistency._extract_embeddings_clip", return_value={})
    def test_create_profile_saves_metadata(self, _clip, _face):
        from opencut.core.character_consistency import (
            _profile_meta_path,
            create_character_profile,
            delete_character_profile,
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"fake image data")
            img_path = f.name
        try:
            profile = create_character_profile([img_path], name="test_hero")
            assert profile.name == "test_hero"
            assert profile.profile_id != ""
            assert os.path.isfile(_profile_meta_path(profile.profile_id))

            # Read metadata
            with open(_profile_meta_path(profile.profile_id)) as mf:
                meta = json.load(mf)
            assert meta["name"] == "test_hero"
        finally:
            os.unlink(img_path)
            if profile:
                delete_character_profile(profile.profile_id)

    def test_list_profiles(self):
        from opencut.core.character_consistency import list_character_profiles
        profiles = list_character_profiles()
        assert isinstance(profiles, list)

    @patch("opencut.core.character_consistency._extract_embeddings_insightface", return_value={})
    @patch("opencut.core.character_consistency._extract_embeddings_clip", return_value={})
    def test_delete_profile(self, _clip, _face):
        from opencut.core.character_consistency import (
            _profile_dir,
            create_character_profile,
            delete_character_profile,
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"fake image")
            img_path = f.name
        try:
            profile = create_character_profile([img_path], name="delete_me")
            pid = profile.profile_id
            assert os.path.isdir(_profile_dir(pid))

            result = delete_character_profile(pid)
            assert result is True
            assert not os.path.isdir(_profile_dir(pid))
        finally:
            os.unlink(img_path)

    def test_delete_nonexistent_returns_false(self):
        from opencut.core.character_consistency import delete_character_profile
        assert delete_character_profile("nonexistent_999") is False


# ===================================================================
# Character Consistency - Generation
# ===================================================================
class TestCharacterGeneration:
    @patch("opencut.core.character_consistency.run_ffmpeg")
    def test_generate_placeholder(self, mock_ffmpeg):
        from opencut.core.character_consistency import (
            CharacterProfile,
            generate_consistent_scene,
        )
        mock_ffmpeg.return_value = ""
        profile = CharacterProfile(profile_id="test", name="hero")

        result = generate_consistent_scene(
            prompt="hero walks through city",
            character_profile=profile,
            duration=2.0,
        )
        assert result.model_used == "placeholder"
        assert result.prompt == "hero walks through city"
        assert result.duration == 2.0

    @patch("opencut.core.character_consistency.run_ffmpeg")
    def test_generate_clamps_duration(self, mock_ffmpeg):
        from opencut.core.character_consistency import (
            CharacterProfile,
            generate_consistent_scene,
        )
        mock_ffmpeg.return_value = ""
        profile = CharacterProfile(profile_id="test", name="hero")

        result = generate_consistent_scene("test", profile, duration=100.0)
        assert result.duration == 30.0  # Clamped to max

    @patch("opencut.core.character_consistency.run_ffmpeg")
    def test_generate_progress_callback(self, mock_ffmpeg):
        from opencut.core.character_consistency import (
            CharacterProfile,
            generate_consistent_scene,
        )
        mock_ffmpeg.return_value = ""
        progress_calls = []

        def on_prog(pct, msg=""):
            progress_calls.append((pct, msg))

        profile = CharacterProfile(profile_id="test", name="hero")
        generate_consistent_scene("scene prompt", profile, on_progress=on_prog)
        assert len(progress_calls) >= 2


# ===================================================================
# Motion Brush - Dataclasses
# ===================================================================
class TestMotionBrushResult:
    def test_default_result(self):
        from opencut.core.motion_brush import MotionBrushResult
        r = MotionBrushResult()
        assert r.output_path == ""
        assert r.duration == 0.0
        assert r.regions_animated == 0
        assert r.method_used == ""

    def test_populated_result(self):
        from opencut.core.motion_brush import MotionBrushResult
        r = MotionBrushResult(
            output_path="/out.mp4",
            duration=4.0,
            regions_animated=3,
            method_used="ffmpeg_regions",
        )
        assert r.regions_animated == 3

    def test_to_dict(self):
        from opencut.core.motion_brush import MotionBrushResult
        r = MotionBrushResult(method_used="ffmpeg_mask")
        d = r.to_dict()
        assert isinstance(d, dict)
        assert d["method_used"] == "ffmpeg_mask"


class TestMotionRegion:
    def test_default_region(self):
        from opencut.core.motion_brush import MotionRegion
        r = MotionRegion()
        assert r.x == 0
        assert r.y == 0
        assert r.direction == "right"
        assert r.strength == 0.5

    def test_validate_clamps(self):
        from opencut.core.motion_brush import MotionRegion
        r = MotionRegion(x=-10, y=2000, w=5000, h=5000, strength=2.0, direction="invalid")
        r.validate(1920, 1080)
        assert r.x == 0
        assert r.y == 1079
        assert r.strength == 1.0
        assert r.direction == "right"  # Reset to default


# ===================================================================
# Motion Brush - Constants & Helpers
# ===================================================================
class TestMotionBrushHelpers:
    def test_valid_directions(self):
        from opencut.core.motion_brush import VALID_DIRECTIONS
        assert "left" in VALID_DIRECTIONS
        assert "right" in VALID_DIRECTIONS
        assert "zoom_in" in VALID_DIRECTIONS
        assert "rotate_cw" in VALID_DIRECTIONS

    def test_is_image(self):
        from opencut.core.motion_brush import _is_image
        assert _is_image("photo.png") is True
        assert _is_image("photo.jpg") is True
        assert _is_image("video.mp4") is False

    def test_is_video(self):
        from opencut.core.motion_brush import _is_video
        assert _is_video("clip.mp4") is True
        assert _is_video("clip.mov") is True
        assert _is_video("photo.png") is False

    def test_parse_regions_dict(self):
        from opencut.core.motion_brush import _parse_regions
        result = _parse_regions({"x": 10, "y": 20, "w": 100, "h": 100, "direction": "up"})
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].direction == "up"

    def test_parse_regions_list(self):
        from opencut.core.motion_brush import _parse_regions
        result = _parse_regions([
            {"x": 0, "y": 0, "w": 50, "h": 50, "direction": "left"},
            {"x": 100, "y": 100, "w": 50, "h": 50, "direction": "right"},
        ])
        assert len(result) == 2

    def test_parse_regions_string(self):
        from opencut.core.motion_brush import _parse_regions
        result = _parse_regions("/path/to/mask.png")
        assert result == "/path/to/mask.png"

    def test_parse_regions_limits(self):
        from opencut.core.motion_brush import _MAX_REGIONS, _parse_regions
        regions = [{"x": i, "y": 0, "w": 10, "h": 10} for i in range(50)]
        result = _parse_regions(regions)
        assert len(result) == _MAX_REGIONS


# ===================================================================
# Motion Brush - Validation
# ===================================================================
class TestMotionBrushValidation:
    def test_missing_file_raises(self):
        from opencut.core.motion_brush import apply_motion_brush
        with pytest.raises(FileNotFoundError):
            apply_motion_brush("/nonexistent/image.png", {"x": 0, "y": 0, "w": 10, "h": 10})

    def test_empty_mask_raises(self):
        from opencut.core.motion_brush import apply_motion_brush
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            with pytest.raises(ValueError):
                apply_motion_brush(path, None)
        finally:
            os.unlink(path)


# ===================================================================
# Motion Brush - Mask Generation
# ===================================================================
class TestMaskGeneration:
    def test_create_motion_mask(self):
        from opencut.core.motion_brush import create_motion_mask
        import numpy as np
        # Create a real small test image
        import cv2
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(f.name, img)
            img_path = f.name

        try:
            mask_path = create_motion_mask(img_path, [
                {"x": 10, "y": 10, "w": 30, "h": 30, "strength": 0.8},
            ])
            assert os.path.isfile(mask_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            assert mask is not None
            assert mask.shape == (100, 100)
        finally:
            os.unlink(img_path)
            if os.path.exists(mask_path):
                os.unlink(mask_path)


# ===================================================================
# Motion Brush - FFmpeg Motion
# ===================================================================
class TestMotionBrushApply:
    @patch("opencut.core.motion_brush.run_ffmpeg")
    @patch("opencut.core.motion_brush.ensure_package", return_value=True)
    def test_still_image_motion(self, _mock_pkg, mock_ffmpeg):
        from opencut.core.motion_brush import apply_motion_brush
        mock_ffmpeg.return_value = ""

        import cv2
        import numpy as np
        # Create a real tiny image so cv2.imread returns a valid array
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = np.zeros((1080, 1920, 3), dtype=np.uint8)
            cv2.imwrite(f.name, img)
            path = f.name
        try:
            result = apply_motion_brush(
                path,
                [{"x": 100, "y": 100, "w": 200, "h": 200, "direction": "right", "strength": 0.5}],
            )
            assert result.method_used == "ffmpeg_regions"
            assert result.regions_animated == 1
        finally:
            os.unlink(path)

    @patch("opencut.core.motion_brush.run_ffmpeg")
    @patch("opencut.core.motion_brush.get_video_info")
    def test_video_motion(self, mock_info, mock_ffmpeg):
        from opencut.core.motion_brush import apply_motion_brush
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 5.0}
        mock_ffmpeg.return_value = ""

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake video")
            path = f.name
        try:
            result = apply_motion_brush(
                path,
                [
                    {"x": 0, "y": 0, "w": 100, "h": 100, "direction": "up"},
                    {"x": 500, "y": 500, "w": 200, "h": 200, "direction": "zoom_in"},
                ],
                {"duration": 3.0},
            )
            assert result.regions_animated == 2
            assert result.method_used == "ffmpeg_regions"
        finally:
            os.unlink(path)


# ===================================================================
# Motion Brush - Direction Filter Building
# ===================================================================
class TestMotionFilterBuilding:
    def test_build_filter_left(self):
        from opencut.core.motion_brush import MotionRegion, _build_region_filter
        r = MotionRegion(x=100, y=100, w=200, h=200, direction="left", strength=0.5)
        f = _build_region_filter(r, 1920, 1080, 4.0, 30.0, 0)
        assert "scroll" in f
        assert "-1" in f

    def test_build_filter_zoom_in(self):
        from opencut.core.motion_brush import MotionRegion, _build_region_filter
        r = MotionRegion(x=100, y=100, w=200, h=200, direction="zoom_in", strength=0.5)
        f = _build_region_filter(r, 1920, 1080, 4.0, 30.0, 0)
        assert "crop" in f
        assert "scale" in f

    def test_build_filter_rotate_cw(self):
        from opencut.core.motion_brush import MotionRegion, _build_region_filter
        r = MotionRegion(x=100, y=100, w=200, h=200, direction="rotate_cw", strength=0.5)
        f = _build_region_filter(r, 1920, 1080, 4.0, 30.0, 0)
        assert "rotate" in f

    def test_build_filter_rotate_ccw(self):
        from opencut.core.motion_brush import MotionRegion, _build_region_filter
        r = MotionRegion(x=100, y=100, w=200, h=200, direction="rotate_ccw", strength=0.5)
        f = _build_region_filter(r, 1920, 1080, 4.0, 30.0, 0)
        assert "rotate" in f
        assert "-" in f

    def test_build_filter_zoom_out(self):
        from opencut.core.motion_brush import MotionRegion, _build_region_filter
        r = MotionRegion(x=100, y=100, w=200, h=200, direction="zoom_out", strength=0.5)
        f = _build_region_filter(r, 1920, 1080, 4.0, 30.0, 0)
        assert "crop" in f

    def test_build_filter_up(self):
        from opencut.core.motion_brush import MotionRegion, _build_region_filter
        r = MotionRegion(x=100, y=100, w=200, h=200, direction="up", strength=0.5)
        f = _build_region_filter(r, 1920, 1080, 4.0, 30.0, 0)
        assert "scroll" in f


# ===================================================================
# Route Blueprint Smoke Tests
# ===================================================================
class TestMotionGenRoutesRegistration:
    def test_blueprint_exists(self):
        from opencut.routes.motion_gen_routes import motion_gen_bp
        assert motion_gen_bp.name == "motion_gen"

    def test_routes_defined(self):
        import opencut.routes.motion_gen_routes as mod
        assert hasattr(mod, "generative_extend")
        assert hasattr(mod, "replace_background")
        assert hasattr(mod, "replace_background_preview")
        assert hasattr(mod, "character_create")
        assert hasattr(mod, "character_list")
        assert hasattr(mod, "character_generate")
        assert hasattr(mod, "motion_brush")
        assert hasattr(mod, "motion_brush_preview")

    def test_character_list_route(self):
        from flask import Flask
        from opencut.routes.motion_gen_routes import motion_gen_bp
        app = Flask(__name__)
        app.register_blueprint(motion_gen_bp)
        with app.test_client() as client:
            resp = client.get("/ai/character/list")
            assert resp.status_code == 200
            data = resp.get_json()
            assert "profiles" in data
            assert "count" in data

    def test_generative_extend_route_no_csrf(self):
        from flask import Flask
        from opencut.routes.motion_gen_routes import motion_gen_bp
        app = Flask(__name__)
        app.config["TESTING"] = True
        app.register_blueprint(motion_gen_bp)
        with app.test_client() as client:
            resp = client.post("/video/generative-extend", json={"filepath": "test.mp4"})
            # Should fail with CSRF error (403 or 400)
            assert resp.status_code in (400, 403)

    def test_replace_background_route_no_csrf(self):
        from flask import Flask
        from opencut.routes.motion_gen_routes import motion_gen_bp
        app = Flask(__name__)
        app.config["TESTING"] = True
        app.register_blueprint(motion_gen_bp)
        with app.test_client() as client:
            resp = client.post("/video/replace-background", json={"filepath": "test.mp4"})
            assert resp.status_code in (400, 403)

    def test_motion_brush_route_no_csrf(self):
        from flask import Flask
        from opencut.routes.motion_gen_routes import motion_gen_bp
        app = Flask(__name__)
        app.config["TESTING"] = True
        app.register_blueprint(motion_gen_bp)
        with app.test_client() as client:
            resp = client.post("/video/motion-brush", json={"filepath": "test.mp4"})
            assert resp.status_code in (400, 403)

    def test_character_create_route_no_csrf(self):
        from flask import Flask
        from opencut.routes.motion_gen_routes import motion_gen_bp
        app = Flask(__name__)
        app.config["TESTING"] = True
        app.register_blueprint(motion_gen_bp)
        with app.test_client() as client:
            resp = client.post("/ai/character/create", json={})
            assert resp.status_code in (400, 403)

    def test_character_generate_route_no_csrf(self):
        from flask import Flask
        from opencut.routes.motion_gen_routes import motion_gen_bp
        app = Flask(__name__)
        app.config["TESTING"] = True
        app.register_blueprint(motion_gen_bp)
        with app.test_client() as client:
            resp = client.post("/ai/character/generate", json={})
            assert resp.status_code in (400, 403)


# ===================================================================
# Route Integration Tests with CSRF
# ===================================================================
class TestMotionGenRoutesCsrf:
    def test_generative_extend_missing_filepath(self):
        from flask import Flask
        from opencut.routes.motion_gen_routes import motion_gen_bp
        from opencut.security import get_csrf_token
        app = Flask(__name__)
        app.config["TESTING"] = True
        app.register_blueprint(motion_gen_bp)
        with app.test_client() as client:
            token = get_csrf_token()
            resp = client.post(
                "/video/generative-extend",
                json={},
                headers={"X-OpenCut-Token": token},
            )
            assert resp.status_code == 400

    def test_character_create_starts_job(self):
        """Character create with filepath_required=False returns a job_id (200)."""
        from flask import Flask
        from opencut.routes.motion_gen_routes import motion_gen_bp
        from opencut.security import get_csrf_token
        app = Flask(__name__)
        app.config["TESTING"] = True
        app.register_blueprint(motion_gen_bp)
        with app.test_client() as client:
            token = get_csrf_token()
            resp = client.post(
                "/ai/character/create",
                json={},
                headers={"X-OpenCut-Token": token},
            )
            # filepath_required=False means validation happens in the worker thread,
            # so the HTTP response is 200 with a job_id
            assert resp.status_code == 200
            data = resp.get_json()
            assert "job_id" in data

    def test_character_generate_starts_job(self):
        """Character generate with filepath_required=False returns a job_id (200)."""
        from flask import Flask
        from opencut.routes.motion_gen_routes import motion_gen_bp
        from opencut.security import get_csrf_token
        app = Flask(__name__)
        app.config["TESTING"] = True
        app.register_blueprint(motion_gen_bp)
        with app.test_client() as client:
            token = get_csrf_token()
            resp = client.post(
                "/ai/character/generate",
                json={"prompt": "test"},
                headers={"X-OpenCut-Token": token},
            )
            assert resp.status_code == 200
            data = resp.get_json()
            assert "job_id" in data

    def test_motion_brush_missing_mask(self):
        from flask import Flask
        from opencut.routes.motion_gen_routes import motion_gen_bp
        from opencut.security import get_csrf_token
        app = Flask(__name__)
        app.config["TESTING"] = True
        app.register_blueprint(motion_gen_bp)
        with app.test_client() as client:
            token = get_csrf_token()
            resp = client.post(
                "/video/motion-brush",
                json={"filepath": "test.mp4"},
                headers={"X-OpenCut-Token": token},
            )
            assert resp.status_code == 400

    def test_replace_background_preview_missing_file(self):
        from flask import Flask
        from opencut.routes.motion_gen_routes import motion_gen_bp
        from opencut.security import get_csrf_token
        app = Flask(__name__)
        app.config["TESTING"] = True
        app.register_blueprint(motion_gen_bp)
        with app.test_client() as client:
            token = get_csrf_token()
            resp = client.post(
                "/video/replace-background/preview",
                json={},
                headers={"X-OpenCut-Token": token},
            )
            assert resp.status_code == 400
