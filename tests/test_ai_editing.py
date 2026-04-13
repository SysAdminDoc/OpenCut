"""
Tests for OpenCut AI Editing features.

Covers:
- Eye contact correction
- AI overdub / voice correction
- Lip sync generation
- Voice conversion (RVC)
- B-roll suggestion
- Morph cut / smooth jump cut
- Frame extension / outpainting
- AI storyboard generator
- AI editing routes
"""

import os
import tempfile
from unittest.mock import patch

import pytest


# ===================================================================
# Eye Contact
# ===================================================================
class TestEyeContactConfig:
    def test_default_config(self):
        from opencut.core.eye_contact import EyeContactConfig
        cfg = EyeContactConfig()
        assert cfg.strength == 0.7
        assert cfg.smoothing_window == 5
        assert cfg.max_yaw_correction == 25.0
        assert cfg.max_pitch_correction == 15.0
        assert cfg.face_confidence == 0.5
        assert cfg.only_center_face is True

    def test_custom_config(self):
        from opencut.core.eye_contact import EyeContactConfig
        cfg = EyeContactConfig(strength=0.5, smoothing_window=10)
        assert cfg.strength == 0.5
        assert cfg.smoothing_window == 10


class TestEyeContactResult:
    def test_default_result(self):
        from opencut.core.eye_contact import EyeContactResult
        r = EyeContactResult()
        assert r.output_path == ""
        assert r.frames_processed == 0
        assert r.frames_corrected == 0
        assert r.avg_gaze_offset == 0.0

    def test_populated_result(self):
        from opencut.core.eye_contact import EyeContactResult
        r = EyeContactResult(output_path="/out.mp4", frames_processed=100,
                             frames_corrected=80, avg_gaze_offset=5.2)
        assert r.frames_corrected == 80


class TestDetectGazeDirection:
    def test_no_face_mesh_returns_not_detected(self):
        import numpy as np

        from opencut.core.eye_contact import detect_gaze_direction
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_gaze_direction(frame, face_mesh=None)
        assert result["detected"] is False

    @patch("opencut.core.eye_contact.ensure_package", return_value=True)
    def test_returns_dict_keys(self, _mock):
        import numpy as np

        from opencut.core.eye_contact import detect_gaze_direction
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_gaze_direction(frame, face_mesh=None)
        assert "detected" in result
        assert "yaw" in result
        assert "pitch" in result
        assert "left_eye_center" in result
        assert "right_eye_center" in result
        assert "face_center" in result


# ===================================================================
# Overdub
# ===================================================================
class TestOverdubConfig:
    def test_default_config(self):
        from opencut.core.overdub import OverdubConfig
        cfg = OverdubConfig()
        assert cfg.crossfade_ms == 150
        assert cfg.tts_backend == "edge"
        assert cfg.language == "en"
        assert cfg.speed == 1.0

    def test_custom_config(self):
        from opencut.core.overdub import OverdubConfig
        cfg = OverdubConfig(crossfade_ms=200, language="es")
        assert cfg.crossfade_ms == 200
        assert cfg.language == "es"


class TestOverdubResult:
    def test_default_result(self):
        from opencut.core.overdub import OverdubResult
        r = OverdubResult()
        assert r.output_path == ""
        assert r.segment_start == 0.0
        assert r.new_text == ""
        assert r.tts_backend_used == ""


class TestOverdubTTSFallback:
    @patch("opencut.core.overdub.run_ffmpeg")
    def test_fallback_generates_audio(self, mock_ffmpeg):
        from opencut.core.overdub import _generate_tts_fallback
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            _generate_tts_fallback("Hello world test text", path)
            assert mock_ffmpeg.called
            cmd = mock_ffmpeg.call_args[0][0]
            assert "anullsrc" in " ".join(cmd)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    @patch("opencut.core.overdub.run_ffmpeg")
    def test_fallback_speed(self, mock_ffmpeg):
        from opencut.core.overdub import _generate_tts_fallback
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            _generate_tts_fallback("test", path, speed=2.0)
            assert mock_ffmpeg.called
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


class TestOverdubValidation:
    @patch("opencut.core.overdub.ensure_package", return_value=True)
    @patch("opencut.core.overdub.run_ffmpeg")
    @patch("opencut.core.overdub.get_video_info", return_value={"duration": 60})
    def test_empty_text_raises(self, _info, _ffmpeg, _pkg):
        from opencut.core.overdub import overdub_segment
        with pytest.raises(ValueError, match="empty"):
            overdub_segment("/video.mp4", 5.0, 10.0, "")

    @patch("opencut.core.overdub.ensure_package", return_value=True)
    @patch("opencut.core.overdub.run_ffmpeg")
    @patch("opencut.core.overdub.get_video_info", return_value={"duration": 60})
    def test_start_gte_end_raises(self, _info, _ffmpeg, _pkg):
        from opencut.core.overdub import overdub_segment
        with pytest.raises(ValueError, match="less than"):
            overdub_segment("/video.mp4", 10.0, 5.0, "hello")


# ===================================================================
# Lip Sync
# ===================================================================
class TestLipSyncConfig:
    def test_default_config(self):
        from opencut.core.lip_sync_gen import LipSyncConfig
        cfg = LipSyncConfig()
        assert cfg.face_confidence == 0.5
        assert cfg.blend_radius == 15
        assert cfg.jaw_sensitivity == 1.0
        assert cfg.smooth_frames == 3

    def test_custom_config(self):
        from opencut.core.lip_sync_gen import LipSyncConfig
        cfg = LipSyncConfig(jaw_sensitivity=2.0, smooth_frames=5)
        assert cfg.jaw_sensitivity == 2.0


class TestLipSyncResult:
    def test_default_result(self):
        from opencut.core.lip_sync_gen import LipSyncResult
        r = LipSyncResult()
        assert r.frames_synced == 0
        assert r.audio_duration == 0.0


class TestDetectMouthRegion:
    def test_no_face_mesh_returns_not_detected(self):
        import numpy as np

        from opencut.core.lip_sync_gen import detect_mouth_region
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_mouth_region(frame, face_mesh=None)
        assert result["detected"] is False
        assert result["mouth_center"] is None

    def test_returns_expected_keys(self):
        import numpy as np

        from opencut.core.lip_sync_gen import detect_mouth_region
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_mouth_region(frame)
        assert "detected" in result
        assert "mouth_bbox" in result
        assert "mouth_open_ratio" in result
        assert "jaw_landmarks" in result


class TestExtractAudioEnergy:
    @patch("opencut.core.lip_sync_gen.ensure_package", return_value=True)
    def test_returns_list(self, _pkg):
        from opencut.core.lip_sync_gen import _extract_audio_energy
        # Non-existent file returns empty list
        result = _extract_audio_energy("/nonexistent.wav", 30.0)
        assert isinstance(result, list)


# ===================================================================
# Voice Conversion
# ===================================================================
class TestVoiceConversionConfig:
    def test_default_config(self):
        from opencut.core.voice_conversion import VoiceConversionConfig
        cfg = VoiceConversionConfig()
        assert cfg.pitch_shift == 0
        assert cfg.index_rate == 0.75
        assert cfg.f0_method == "harvest"
        assert cfg.sample_rate == 44100

    def test_custom_config(self):
        from opencut.core.voice_conversion import VoiceConversionConfig
        cfg = VoiceConversionConfig(pitch_shift=5, f0_method="crepe")
        assert cfg.pitch_shift == 5
        assert cfg.f0_method == "crepe"


class TestVoiceConversionResult:
    def test_default_result(self):
        from opencut.core.voice_conversion import VoiceConversionResult
        r = VoiceConversionResult()
        assert r.output_path == ""
        assert r.method == ""


class TestListVoiceModels:
    def test_returns_list(self):
        from opencut.core.voice_conversion import list_voice_models
        models = list_voice_models()
        assert isinstance(models, list)

    def test_models_have_expected_keys(self):
        from opencut.core.voice_conversion import RVC_MODELS_DIR, list_voice_models
        # Create a fake model file
        os.makedirs(RVC_MODELS_DIR, exist_ok=True)
        fake_model = os.path.join(RVC_MODELS_DIR, "_test_model.pth")
        try:
            with open(fake_model, "wb") as f:
                f.write(b"fake")
            models = list_voice_models()
            test_models = [m for m in models if m["name"] == "_test_model"]
            assert len(test_models) == 1
            assert "path" in test_models[0]
            assert "size_mb" in test_models[0]
        finally:
            try:
                os.unlink(fake_model)
            except OSError:
                pass


class TestFindModel:
    def test_missing_model_raises(self):
        from opencut.core.voice_conversion import _find_model
        with pytest.raises(FileNotFoundError):
            _find_model("nonexistent_model_xyz")


class TestVoiceConversionFfmpeg:
    @patch("opencut.core.voice_conversion.run_ffmpeg")
    def test_ffmpeg_fallback_runs(self, mock_ffmpeg):
        from opencut.core.voice_conversion import VoiceConversionConfig, _convert_voice_ffmpeg
        cfg = VoiceConversionConfig(pitch_shift=3)
        _convert_voice_ffmpeg("/input.wav", "/output.wav", cfg)
        assert mock_ffmpeg.called


# ===================================================================
# B-Roll Suggestion
# ===================================================================
class TestBRollCue:
    def test_default_cue(self):
        from opencut.core.broll_suggest import BRollCue
        cue = BRollCue(start=1.0, end=3.0, cue_type="visual_phrase")
        assert cue.score == 0.0
        assert cue.keywords == []

    def test_cue_with_keywords(self):
        from opencut.core.broll_suggest import BRollCue
        cue = BRollCue(start=0, end=5, cue_type="keyword",
                       keywords=["mountain", "sunset"], category="nature")
        assert len(cue.keywords) == 2


class TestDetectBrollCues:
    def test_empty_transcript(self):
        from opencut.core.broll_suggest import detect_broll_cues
        result = detect_broll_cues([])
        assert result == []

    def test_visual_phrase_detection(self):
        from opencut.core.broll_suggest import detect_broll_cues
        transcript = [
            {"start": 0, "end": 5, "text": "Let me show you something amazing"},
        ]
        cues = detect_broll_cues(transcript)
        assert len(cues) >= 1
        assert cues[0].cue_type == "visual_phrase"

    def test_topic_shift_detection(self):
        from opencut.core.broll_suggest import detect_broll_cues
        transcript = [
            {"start": 0, "end": 5, "text": "Moving on to the next topic"},
        ]
        cues = detect_broll_cues(transcript)
        assert any(c.cue_type == "topic_shift" for c in cues)

    def test_keyword_detection(self):
        from opencut.core.broll_suggest import detect_broll_cues
        transcript = [
            {"start": 0, "end": 5, "text": "The mountain landscape was beautiful"},
        ]
        cues = detect_broll_cues(transcript)
        assert any(c.cue_type == "keyword" for c in cues)

    def test_pause_detection(self):
        from opencut.core.broll_suggest import detect_broll_cues
        transcript = [
            {"start": 0, "end": 5, "text": "First segment"},
            {"start": 8, "end": 12, "text": "Second segment"},
        ]
        cues = detect_broll_cues(transcript, min_gap=1.0)
        assert any(c.cue_type == "pause" for c in cues)

    def test_min_score_filter(self):
        from opencut.core.broll_suggest import detect_broll_cues
        transcript = [
            {"start": 0, "end": 5, "text": "Let me show you the mountain view"},
        ]
        cues = detect_broll_cues(transcript, min_score=0.95)
        # Only very high score cues should remain
        assert all(c.score >= 0.95 for c in cues)


class TestSuggestBroll:
    def test_no_footage_index(self):
        from opencut.core.broll_suggest import suggest_broll
        transcript = [
            {"start": 0, "end": 5, "text": "Take a look at this ocean scene"},
        ]
        result = suggest_broll(transcript, footage_index=None)
        assert len(result.cues) >= 1
        assert result.matches == []

    def test_with_footage_matching(self):
        from opencut.core.broll_suggest import suggest_broll
        transcript = [
            {"start": 0, "end": 5, "text": "Take a look at the ocean waves"},
        ]
        footage = [
            {"name": "ocean_clip.mp4", "path": "/clips/ocean.mp4",
             "keywords": ["ocean", "waves", "water"], "tags": ["nature"],
             "duration": 10.0, "category": "nature", "description": "Ocean waves crashing"},
        ]
        result = suggest_broll(transcript, footage_index=footage)
        assert len(result.matches) >= 1
        assert result.coverage_ratio > 0

    def test_cutlist_generation(self):
        from opencut.core.broll_suggest import BRollCue, BRollMatch, generate_broll_cutlist
        cues = [BRollCue(start=0, end=5, cue_type="keyword", keywords=["test"])]
        matches = [BRollMatch(cue_index=0, clip_name="clip.mp4", relevance=0.8)]
        cutlist = generate_broll_cutlist(cues, matches)
        assert cutlist.unmatched_cues == 0
        assert cutlist.coverage_ratio == 1.0


# ===================================================================
# Morph Cut
# ===================================================================
class TestMorphCutConfig:
    def test_default_config(self):
        from opencut.core.morph_cut import MorphCutConfig
        cfg = MorphCutConfig()
        assert cfg.transition_frames == 8
        assert cfg.blend_mode == "optical_flow"
        assert cfg.face_weight == 0.7


class TestMorphCutResult:
    def test_default_result(self):
        from opencut.core.morph_cut import MorphCutResult
        r = MorphCutResult()
        assert r.frames_interpolated == 0
        assert r.face_detected is False


class TestDetectFaceRegion:
    @patch("opencut.core.morph_cut.ensure_package", return_value=True)
    def test_no_face_returns_none(self, _pkg):
        import numpy as np

        from opencut.core.morph_cut import detect_face_region
        # Black frame has no faces
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_face_region(frame)
        assert result is None


class TestInterpolateFrames:
    def test_crossfade_mode(self):
        import numpy as np

        from opencut.core.morph_cut import MorphCutConfig, interpolate_frames
        a = np.full((100, 100, 3), 0, dtype=np.uint8)
        b = np.full((100, 100, 3), 255, dtype=np.uint8)
        cfg = MorphCutConfig(blend_mode="crossfade")
        frames = interpolate_frames(a, b, count=4, config=cfg)
        assert len(frames) == 4
        # Middle frame should be roughly 128 (50% blend)
        mid_val = frames[1].mean()
        assert 80 < mid_val < 200

    def test_zero_count(self):
        import numpy as np

        from opencut.core.morph_cut import interpolate_frames
        a = np.zeros((100, 100, 3), dtype=np.uint8)
        b = np.ones((100, 100, 3), dtype=np.uint8) * 255
        frames = interpolate_frames(a, b, count=0)
        assert len(frames) == 0

    def test_optical_flow_mode(self):
        import numpy as np

        from opencut.core.morph_cut import MorphCutConfig, interpolate_frames
        a = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        b = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cfg = MorphCutConfig(blend_mode="optical_flow", transition_frames=4)
        frames = interpolate_frames(a, b, count=4, config=cfg)
        assert len(frames) == 4


# ===================================================================
# Frame Extension
# ===================================================================
class TestFrameExtensionConfig:
    def test_default_config(self):
        from opencut.core.frame_extension import FrameExtensionConfig
        cfg = FrameExtensionConfig()
        assert cfg.fill_method == "reflect"
        assert cfg.temporal_method == "hold"

    def test_custom_config(self):
        from opencut.core.frame_extension import FrameExtensionConfig
        cfg = FrameExtensionConfig(fill_method="blur", blur_strength=99)
        assert cfg.fill_method == "blur"
        assert cfg.blur_strength == 99


class TestSpatialExtensionResult:
    def test_default_result(self):
        from opencut.core.frame_extension import SpatialExtensionResult
        r = SpatialExtensionResult()
        assert r.output_path == ""
        assert r.original_size == (0, 0)


class TestDetectEdgeRegions:
    def test_returns_expected_keys(self):
        import numpy as np

        from opencut.core.frame_extension import detect_edge_regions
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = detect_edge_regions(frame)
        assert "edge_complexity" in result
        assert "dominant_colors" in result
        assert "recommended_fill" in result
        assert result["recommended_fill"] in ("reflect", "blur", "inpaint")

    def test_simple_frame_recommends_reflect(self):
        import numpy as np

        from opencut.core.frame_extension import detect_edge_regions
        # Uniform frame should have low complexity
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        result = detect_edge_regions(frame)
        assert result["recommended_fill"] == "reflect"


class TestParseAspectRatio:
    def test_standard_ratios(self):
        from opencut.core.frame_extension import _parse_aspect_ratio
        assert _parse_aspect_ratio("16:9") == (16, 9)
        assert _parse_aspect_ratio("4:3") == (4, 3)
        assert _parse_aspect_ratio("21:9") == (21, 9)

    def test_slash_format(self):
        from opencut.core.frame_extension import _parse_aspect_ratio
        assert _parse_aspect_ratio("16/9") == (16, 9)

    def test_invalid_raises(self):
        from opencut.core.frame_extension import _parse_aspect_ratio
        with pytest.raises(ValueError):
            _parse_aspect_ratio("invalid")


class TestComputeTargetSize:
    def test_wider_target(self):
        from opencut.core.frame_extension import _compute_target_size
        w, h = _compute_target_size(640, 480, 16, 9)
        assert w > 640
        assert h == 480

    def test_taller_target(self):
        from opencut.core.frame_extension import _compute_target_size
        w, h = _compute_target_size(1920, 1080, 4, 3)
        assert w == 1920
        assert h > 1080


class TestExtendFrame:
    def test_reflect_fill(self):
        import numpy as np

        from opencut.core.frame_extension import FrameExtensionConfig, _extend_frame
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        cfg = FrameExtensionConfig(fill_method="reflect")
        result = _extend_frame(frame, 854, 480, cfg)
        assert result.shape == (480, 854, 3)

    def test_blur_fill(self):
        import numpy as np

        from opencut.core.frame_extension import FrameExtensionConfig, _extend_frame
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        cfg = FrameExtensionConfig(fill_method="blur")
        result = _extend_frame(frame, 854, 480, cfg)
        assert result.shape == (480, 854, 3)

    def test_replicate_fill(self):
        import numpy as np

        from opencut.core.frame_extension import FrameExtensionConfig, _extend_frame
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        cfg = FrameExtensionConfig(fill_method="replicate")
        result = _extend_frame(frame, 854, 480, cfg)
        assert result.shape == (480, 854, 3)

    def test_same_size_noop(self):
        import numpy as np

        from opencut.core.frame_extension import FrameExtensionConfig, _extend_frame
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        cfg = FrameExtensionConfig()
        result = _extend_frame(frame, 640, 480, cfg)
        assert result.shape == (480, 640, 3)


# ===================================================================
# AI Storyboard
# ===================================================================
class TestShotDescription:
    def test_default_shot(self):
        from opencut.core.ai_storyboard import ShotDescription
        s = ShotDescription(shot_number=1)
        assert s.shot_type == ""
        assert s.camera_direction == ""
        assert s.dialogue == ""


class TestParseScriptShots:
    def test_empty_script(self):
        from opencut.core.ai_storyboard import parse_shot_descriptions
        result = parse_shot_descriptions("")
        assert result == []

    def test_numbered_shots(self):
        from opencut.core.ai_storyboard import parse_shot_descriptions
        script = """1. Wide shot of the office building exterior
2. Medium shot of John entering the lobby
3. Close-up of the elevator buttons"""
        shots = parse_shot_descriptions(script)
        assert len(shots) == 3
        assert shots[0].shot_number == 1

    def test_paragraph_splitting(self):
        from opencut.core.ai_storyboard import parse_shot_descriptions
        script = """A wide aerial view of the city at dawn.

Close-up of a hand reaching for a coffee cup.

Medium shot of Sarah looking out the window."""
        shots = parse_shot_descriptions(script)
        assert len(shots) == 3

    def test_shot_type_detection(self):
        from opencut.core.ai_storyboard import parse_shot_descriptions
        script = "1. Wide shot of mountains\n2. Close-up of flowers"
        shots = parse_shot_descriptions(script)
        assert shots[0].shot_type == "WS"
        assert shots[1].shot_type == "CU"

    def test_camera_direction_detection(self):
        from opencut.core.ai_storyboard import parse_shot_descriptions
        script = "1. Camera pan left across the landscape\n2. Dolly in on the subject"
        shots = parse_shot_descriptions(script)
        assert shots[0].camera_direction == "PAN LEFT"
        assert shots[1].camera_direction == "DOLLY IN"

    def test_dialogue_extraction(self):
        from opencut.core.ai_storyboard import parse_shot_descriptions
        script = '1. John says "Hello there" to the audience'
        shots = parse_shot_descriptions(script)
        assert shots[0].dialogue == "Hello there"

    def test_scene_heading_parsing(self):
        from opencut.core.ai_storyboard import parse_shot_descriptions
        script = """INT. OFFICE - DAY
John sits at his desk, typing.

EXT. PARKING LOT - NIGHT
Sarah walks to her car."""
        shots = parse_shot_descriptions(script)
        assert len(shots) == 2


class TestStoryboardGeneration:
    @patch("opencut.core.ai_storyboard.ensure_package", return_value=True)
    def test_generate_storyboard(self, _pkg):
        from opencut.core.ai_storyboard import generate_storyboard
        with tempfile.TemporaryDirectory() as tmpdir:
            script = """1. Wide shot of mountains
2. Close-up of hiker
3. Medium shot of campfire"""
            result = generate_storyboard(
                script, tmpdir, export_pdf=False,
            )
            assert result.total_shots == 3
            assert len(result.panels) == 3
            assert result.grid_path != ""
            assert os.path.isfile(result.grid_path)

    @patch("opencut.core.ai_storyboard.ensure_package", return_value=True)
    def test_empty_script_raises(self, _pkg):
        from opencut.core.ai_storyboard import generate_storyboard
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="empty"):
                generate_storyboard("", tmpdir)

    @patch("opencut.core.ai_storyboard.ensure_package", return_value=True)
    def test_storyboard_with_pdf(self, _pkg):
        from opencut.core.ai_storyboard import generate_storyboard
        with tempfile.TemporaryDirectory() as tmpdir:
            script = "1. Wide shot\n2. Medium shot"
            result = generate_storyboard(script, tmpdir, export_pdf=True)
            assert result.pdf_path != ""
            assert os.path.isfile(result.pdf_path)

    @patch("opencut.core.ai_storyboard.ensure_package", return_value=True)
    def test_progress_callback(self, _pkg):
        from opencut.core.ai_storyboard import generate_storyboard
        progress_calls = []
        def on_progress(pct, msg):
            progress_calls.append((pct, msg))

        with tempfile.TemporaryDirectory() as tmpdir:
            script = "1. Shot one\n2. Shot two"
            generate_storyboard(script, tmpdir, export_pdf=False,
                                on_progress=on_progress)
            assert len(progress_calls) > 0
            assert progress_calls[-1][0] == 100


class TestRenderStoryboardGrid:
    @patch("opencut.core.ai_storyboard.ensure_package", return_value=True)
    def test_empty_images_raises(self, _pkg):
        from opencut.core.ai_storyboard import render_storyboard_grid
        with pytest.raises(ValueError, match="No images"):
            render_storyboard_grid([], [], "/dev/null")


class TestExportStoryboardPdf:
    @patch("opencut.core.ai_storyboard.ensure_package", return_value=True)
    def test_empty_storyboard_raises(self, _pkg):
        from opencut.core.ai_storyboard import Storyboard, export_storyboard_pdf
        sb = Storyboard()
        with pytest.raises(ValueError, match="No panels"):
            export_storyboard_pdf(sb, "/dev/null")


# ===================================================================
# Route Blueprint Registration
# ===================================================================
class TestAiEditingBlueprintRegistration:
    def test_blueprint_importable(self):
        from opencut.routes.ai_editing_routes import ai_editing_bp
        assert ai_editing_bp.name == "ai_editing"

    def test_blueprint_in_register_blueprints(self):
        """Verify ai_editing_bp is registered in routes/__init__.py."""
        import inspect

        from opencut.routes import register_blueprints
        source = inspect.getsource(register_blueprints)
        assert "ai_editing_bp" in source

    def test_routes_defined(self):
        # Blueprint has deferred registrations; check the module has route functions
        import opencut.routes.ai_editing_routes as mod
        assert hasattr(mod, "ai_eye_contact")
        assert hasattr(mod, "ai_overdub")
        assert hasattr(mod, "ai_lip_sync")
        assert hasattr(mod, "ai_voice_convert")
        assert hasattr(mod, "ai_voice_models")
        assert hasattr(mod, "ai_broll_suggest")
        assert hasattr(mod, "ai_morph_cut")
        assert hasattr(mod, "ai_extend_spatial")
        assert hasattr(mod, "ai_extend_temporal")
        assert hasattr(mod, "ai_storyboard")
        assert hasattr(mod, "ai_detect_edges")
        assert hasattr(mod, "ai_parse_script")


# ===================================================================
# Route Integration Smoke Tests
# ===================================================================
class TestAiEditingRoutes:
    """Smoke tests verifying route handler signatures and basic validation."""

    def test_voice_models_route(self):
        """GET /ai/voice-models should return JSON without filepath."""
        from flask import Flask

        from opencut.routes.ai_editing_routes import ai_editing_bp
        app = Flask(__name__)
        app.register_blueprint(ai_editing_bp)
        with app.test_client() as client:
            resp = client.get("/ai/voice-models")
            assert resp.status_code == 200
            data = resp.get_json()
            assert "models" in data
            assert "count" in data

    def test_parse_script_route_missing_text(self):
        """POST /ai/parse-script should fail without script_text."""
        from flask import Flask

        from opencut.routes.ai_editing_routes import ai_editing_bp
        from opencut.security import get_csrf_token
        app = Flask(__name__)
        app.config["TESTING"] = True
        app.register_blueprint(ai_editing_bp)
        with app.test_client() as client:
            token = get_csrf_token()
            resp = client.post(
                "/ai/parse-script",
                json={},
                headers={"X-OpenCut-Token": token},
            )
            assert resp.status_code == 400

    def test_parse_script_route_with_text(self):
        """POST /ai/parse-script should parse valid script."""
        from flask import Flask

        from opencut.routes.ai_editing_routes import ai_editing_bp
        from opencut.security import get_csrf_token
        app = Flask(__name__)
        app.config["TESTING"] = True
        app.register_blueprint(ai_editing_bp)
        with app.test_client() as client:
            token = get_csrf_token()
            resp = client.post(
                "/ai/parse-script",
                json={"script_text": "1. Wide shot of city\n2. Close-up of face"},
                headers={"X-OpenCut-Token": token},
            )
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["count"] == 2
            assert len(data["shots"]) == 2
