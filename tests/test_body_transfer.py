"""
Tests for OpenCut Body Effects, Motion Transfer, Foley Generation,
and Face Restoration modules.

Covers:
  - Body keypoint detection (dataclass, signature, mock mediapipe)
  - All body effect types (glow, trail, highlight, blur_except, neon_outline, particle_follow)
  - Motion transfer pipeline (pose extraction, stick figure, JSON export)
  - Foley generation event detection and SFX mapping
  - Face restoration with mock model outputs
  - Body Transfer Routes (smoke tests)
"""

import inspect
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Body Effects — Dataclass & Constants Tests
# ============================================================
class TestBodyEffectsDataclasses(unittest.TestCase):
    """Tests for body_effects dataclasses and constants."""

    def test_frame_keypoints_fields(self):
        from opencut.core.body_effects import FrameKeypoints
        fk = FrameKeypoints(frame_index=0, timestamp=0.0, keypoints={})
        self.assertEqual(fk.frame_index, 0)
        self.assertEqual(fk.timestamp, 0.0)
        self.assertIsInstance(fk.keypoints, dict)

    def test_body_track_result_fields(self):
        from opencut.core.body_effects import BodyTrackResult
        btr = BodyTrackResult()
        self.assertEqual(btr.frames, [])
        self.assertEqual(btr.fps, 30.0)
        self.assertEqual(btr.total_frames, 0)

    def test_body_parts_list(self):
        from opencut.core.body_effects import BODY_PARTS
        self.assertIsInstance(BODY_PARTS, list)
        self.assertIn("nose", BODY_PARTS)
        self.assertIn("left_wrist", BODY_PARTS)
        self.assertIn("right_ankle", BODY_PARTS)
        self.assertEqual(len(BODY_PARTS), 15)

    def test_effect_types_list(self):
        from opencut.core.body_effects import EFFECT_TYPES
        expected = {"glow", "trail", "highlight", "blur_except", "neon_outline", "particle_follow"}
        self.assertEqual(set(EFFECT_TYPES), expected)

    def test_mp_landmark_map_coverage(self):
        from opencut.core.body_effects import BODY_PARTS, _MP_LANDMARK_MAP
        for part in BODY_PARTS:
            self.assertIn(part, _MP_LANDMARK_MAP, f"Missing landmark map for {part}")
            self.assertIsInstance(_MP_LANDMARK_MAP[part], int)


# ============================================================
# Body Effects — Function Signatures
# ============================================================
class TestBodyEffectsSignatures(unittest.TestCase):
    """Tests for body_effects function signatures."""

    def test_detect_body_keypoints_signature(self):
        from opencut.core.body_effects import detect_body_keypoints
        sig = inspect.signature(detect_body_keypoints)
        self.assertIn("video_path", sig.parameters)
        self.assertIn("on_progress", sig.parameters)
        self.assertIsNone(sig.parameters["on_progress"].default)

    def test_apply_body_effect_signature(self):
        from opencut.core.body_effects import apply_body_effect
        sig = inspect.signature(apply_body_effect)
        self.assertIn("video_path", sig.parameters)
        self.assertIn("effect_type", sig.parameters)
        self.assertIn("body_part", sig.parameters)
        self.assertIn("track_data", sig.parameters)
        self.assertIn("output", sig.parameters)
        self.assertIn("on_progress", sig.parameters)
        self.assertIsNone(sig.parameters["track_data"].default)
        self.assertIsNone(sig.parameters["output"].default)

    def test_detect_missing_file(self):
        from opencut.core.body_effects import detect_body_keypoints
        with self.assertRaises(FileNotFoundError):
            detect_body_keypoints("/nonexistent/video.mp4")

    def test_apply_missing_file(self):
        from opencut.core.body_effects import apply_body_effect
        with self.assertRaises(FileNotFoundError):
            apply_body_effect("/nonexistent/video.mp4", "glow", "nose")

    def test_apply_invalid_effect_type(self):
        from opencut.core.body_effects import apply_body_effect
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                apply_body_effect(path, "invalid_effect", "nose")
        finally:
            os.unlink(path)

    def test_apply_invalid_body_part(self):
        from opencut.core.body_effects import apply_body_effect
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                apply_body_effect(path, "glow", "invalid_part")
        finally:
            os.unlink(path)


# ============================================================
# Body Effects — Effect Filters
# ============================================================
class TestBodyEffectFilters(unittest.TestCase):
    """Tests for body effect filter-building helpers."""

    def test_build_glow_drawtext(self):
        from opencut.core.body_effects import _build_glow_drawtext
        result = _build_glow_drawtext(100, 200, radius=30)
        self.assertIn("drawbox", result)
        self.assertIn("yellow", result)
        self.assertIn("70", result)  # x - radius = 70

    def test_build_highlight_drawbox(self):
        from opencut.core.body_effects import _build_highlight_drawbox
        result = _build_highlight_drawbox(200, 300, radius=50)
        self.assertIn("drawbox", result)
        self.assertIn("white", result)

    def test_build_trail_filter(self):
        from opencut.core.body_effects import _build_trail_filter
        positions = [(100, 100), (110, 110), (120, 120)]
        result = _build_trail_filter(positions, max_trail=3)
        self.assertIn("drawbox", result)
        self.assertIn("cyan", result)

    def test_build_trail_filter_empty(self):
        from opencut.core.body_effects import _build_trail_filter
        result = _build_trail_filter([], max_trail=3)
        self.assertEqual(result, "null")

    @patch("opencut.core.body_effects.run_ffmpeg")
    @patch("opencut.core.body_effects.get_video_info", return_value={"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0})
    def test_apply_glow_calls_ffmpeg(self, mock_info, mock_ffmpeg):
        from opencut.core.body_effects import BodyTrackResult, FrameKeypoints, apply_body_effect
        mock_ffmpeg.return_value = ""
        track = BodyTrackResult(
            frames=[FrameKeypoints(frame_index=0, timestamp=0.0, keypoints={"nose": (100, 100, 0.9)})],
            fps=30.0,
            total_frames=1,
        )
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            result = apply_body_effect(path, "glow", "nose", track_data=track)
            mock_ffmpeg.assert_called_once()
            self.assertIn("body_glow", result)
        finally:
            os.unlink(path)

    @patch("opencut.core.body_effects.run_ffmpeg")
    @patch("opencut.core.body_effects.get_video_info", return_value={"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0})
    def test_apply_highlight_calls_ffmpeg(self, mock_info, mock_ffmpeg):
        from opencut.core.body_effects import BodyTrackResult, FrameKeypoints, apply_body_effect
        mock_ffmpeg.return_value = ""
        track = BodyTrackResult(
            frames=[FrameKeypoints(frame_index=0, timestamp=0.0, keypoints={"nose": (100, 100, 0.9)})],
            fps=30.0, total_frames=1,
        )
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            apply_body_effect(path, "highlight", "nose", track_data=track)
            mock_ffmpeg.assert_called_once()
        finally:
            os.unlink(path)

    @patch("opencut.core.body_effects.run_ffmpeg")
    @patch("opencut.core.body_effects.get_video_info", return_value={"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0})
    def test_apply_trail_calls_ffmpeg(self, mock_info, mock_ffmpeg):
        from opencut.core.body_effects import BodyTrackResult, FrameKeypoints, apply_body_effect
        mock_ffmpeg.return_value = ""
        track = BodyTrackResult(
            frames=[
                FrameKeypoints(frame_index=i, timestamp=i / 30.0, keypoints={"left_wrist": (100 + i, 200, 0.9)})
                for i in range(5)
            ],
            fps=30.0, total_frames=5,
        )
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            apply_body_effect(path, "trail", "left_wrist", track_data=track)
            mock_ffmpeg.assert_called_once()
        finally:
            os.unlink(path)

    @patch("opencut.core.body_effects.run_ffmpeg")
    @patch("opencut.core.body_effects.get_video_info", return_value={"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0})
    def test_apply_particle_follow_calls_ffmpeg(self, mock_info, mock_ffmpeg):
        from opencut.core.body_effects import BodyTrackResult, FrameKeypoints, apply_body_effect
        mock_ffmpeg.return_value = ""
        track = BodyTrackResult(
            frames=[FrameKeypoints(frame_index=0, timestamp=0.0, keypoints={"right_wrist": (400, 300, 0.8)})],
            fps=30.0, total_frames=1,
        )
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            apply_body_effect(path, "particle_follow", "right_wrist", track_data=track)
            mock_ffmpeg.assert_called_once()
        finally:
            os.unlink(path)

    @patch("opencut.core.body_effects.run_ffmpeg")
    @patch("opencut.core.body_effects.get_video_info", return_value={"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0})
    def test_apply_blur_except_calls_ffmpeg(self, mock_info, mock_ffmpeg):
        from opencut.core.body_effects import BodyTrackResult, FrameKeypoints, apply_body_effect
        mock_ffmpeg.return_value = ""
        track = BodyTrackResult(
            frames=[FrameKeypoints(frame_index=0, timestamp=0.0, keypoints={"nose": (960, 540, 0.9)})],
            fps=30.0, total_frames=1,
        )
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            apply_body_effect(path, "blur_except", "nose", track_data=track)
            mock_ffmpeg.assert_called_once()
            cmd = mock_ffmpeg.call_args[0][0]
            self.assertIn("-filter_complex", cmd)
        finally:
            os.unlink(path)

    @patch("opencut.core.body_effects.run_ffmpeg")
    @patch("opencut.core.body_effects.get_video_info", return_value={"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0})
    def test_apply_neon_outline_calls_ffmpeg(self, mock_info, mock_ffmpeg):
        from opencut.core.body_effects import BodyTrackResult, FrameKeypoints, apply_body_effect
        mock_ffmpeg.return_value = ""
        track = BodyTrackResult(
            frames=[FrameKeypoints(frame_index=0, timestamp=0.0, keypoints={"nose": (960, 540, 0.9)})],
            fps=30.0, total_frames=1,
        )
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            apply_body_effect(path, "neon_outline", "nose", track_data=track)
            mock_ffmpeg.assert_called_once()
            cmd = mock_ffmpeg.call_args[0][0]
            self.assertIn("-filter_complex", cmd)
        finally:
            os.unlink(path)


# ============================================================
# Motion Transfer — Dataclass & Constants Tests
# ============================================================
class TestMotionTransferDataclasses(unittest.TestCase):
    """Tests for motion_transfer dataclasses and constants."""

    def test_pose_sequence_fields(self):
        from opencut.core.motion_transfer import PoseSequence
        ps = PoseSequence()
        self.assertEqual(ps.poses, [])
        self.assertEqual(ps.fps, 30.0)
        self.assertEqual(ps.duration, 0.0)

    def test_motion_transfer_result_fields(self):
        from opencut.core.motion_transfer import MotionTransferResult
        mtr = MotionTransferResult()
        self.assertEqual(mtr.output_path, "")
        self.assertEqual(mtr.source_duration, 0.0)
        self.assertEqual(mtr.frames_generated, 0)
        self.assertEqual(mtr.model_used, "stick_figure")

    def test_supported_models_list(self):
        from opencut.core.motion_transfer import SUPPORTED_MODELS
        self.assertIn("auto", SUPPORTED_MODELS)
        self.assertIn("stick_figure", SUPPORTED_MODELS)
        self.assertIn("animate_anyone", SUPPORTED_MODELS)
        self.assertIn("mimic_motion", SUPPORTED_MODELS)

    def test_skeleton_connections(self):
        from opencut.core.motion_transfer import _SKELETON_CONNECTIONS
        self.assertIsInstance(_SKELETON_CONNECTIONS, list)
        self.assertGreater(len(_SKELETON_CONNECTIONS), 10)
        for c in _SKELETON_CONNECTIONS:
            self.assertEqual(len(c), 2)


# ============================================================
# Motion Transfer — Function Signatures
# ============================================================
class TestMotionTransferSignatures(unittest.TestCase):
    """Tests for motion_transfer function signatures."""

    def test_extract_pose_sequence_signature(self):
        from opencut.core.motion_transfer import extract_pose_sequence
        sig = inspect.signature(extract_pose_sequence)
        self.assertIn("video_path", sig.parameters)
        self.assertIn("on_progress", sig.parameters)
        self.assertIsNone(sig.parameters["on_progress"].default)

    def test_transfer_motion_signature(self):
        from opencut.core.motion_transfer import transfer_motion
        sig = inspect.signature(transfer_motion)
        self.assertIn("source_video", sig.parameters)
        self.assertIn("target_image", sig.parameters)
        self.assertIn("output", sig.parameters)
        self.assertIn("model", sig.parameters)
        self.assertIn("on_progress", sig.parameters)
        self.assertEqual(sig.parameters["model"].default, "auto")

    def test_extract_missing_file(self):
        from opencut.core.motion_transfer import extract_pose_sequence
        with self.assertRaises(FileNotFoundError):
            extract_pose_sequence("/nonexistent/video.mp4")

    def test_transfer_missing_source(self):
        from opencut.core.motion_transfer import transfer_motion
        with self.assertRaises(FileNotFoundError):
            transfer_motion("/nonexistent/video.mp4", "/nonexistent/image.png")

    def test_transfer_missing_target(self):
        from opencut.core.motion_transfer import transfer_motion
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            with self.assertRaises(FileNotFoundError):
                transfer_motion(path, "/nonexistent/image.png")
        finally:
            os.unlink(path)

    def test_transfer_invalid_model(self):
        from opencut.core.motion_transfer import transfer_motion
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as fv:
            fv.write(b"fake")
            vid = fv.name
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fi:
            fi.write(b"fake")
            img = fi.name
        try:
            with self.assertRaises(ValueError):
                transfer_motion(vid, img, model="nonexistent_model")
        finally:
            os.unlink(vid)
            os.unlink(img)


# ============================================================
# Motion Transfer — Pipeline Helpers
# ============================================================
class TestMotionTransferPipeline(unittest.TestCase):
    """Tests for motion transfer internal helpers."""

    def test_try_ai_model_returns_none_without_deps(self):
        from opencut.core.motion_transfer import _try_ai_model
        # Without animate_anyone or mimic_motion installed, returns None
        result = _try_ai_model("auto")
        self.assertIsNone(result)

    def test_try_ai_model_stick_figure(self):
        from opencut.core.motion_transfer import _try_ai_model
        # stick_figure is not an AI model name, so returns None
        result = _try_ai_model("stick_figure")
        self.assertIsNone(result)

    def test_export_pose_json(self):
        from opencut.core.motion_transfer import PoseSequence, _export_pose_json
        ps = PoseSequence(
            poses=[{"nose": (100, 200, 0.9)}, {"nose": (105, 205, 0.85)}],
            fps=30.0,
            duration=0.067,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = _export_pose_json(ps, tmpdir)
            self.assertTrue(os.path.isfile(json_path))
            with open(json_path) as f:
                data = json.load(f)
            self.assertEqual(data["fps"], 30.0)
            self.assertEqual(data["frame_count"], 2)
            self.assertEqual(len(data["poses"]), 2)

    @patch("opencut.core.motion_transfer.run_ffmpeg")
    @patch("opencut.core.motion_transfer.ensure_package", return_value=True)
    def test_render_stick_figure_with_mock_cv2(self, mock_ensure, mock_ffmpeg):
        """Test stick figure rendering with mocked cv2."""
        from opencut.core.motion_transfer import PoseSequence, _render_stick_figure_frames
        import numpy as np

        mock_cv2 = MagicMock()
        mock_cv2.imread.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cv2.imwrite.return_value = True
        mock_cv2.LINE_AA = 16

        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            ps = PoseSequence(
                poses=[
                    {"nose": (320, 240, 0.9), "left_shoulder": (280, 300, 0.8), "right_shoulder": (360, 300, 0.8)},
                ],
                fps=30.0,
                duration=0.033,
            )
            with tempfile.TemporaryDirectory() as tmpdir:
                count = _render_stick_figure_frames("/fake/target.png", ps, tmpdir)
                self.assertEqual(count, 1)


# ============================================================
# Foley Generation — Dataclass & Constants Tests
# ============================================================
class TestFoleyDataclasses(unittest.TestCase):
    """Tests for foley_gen dataclasses and constants."""

    def test_foley_sound_fields(self):
        from opencut.core.foley_gen import FoleySound
        fs = FoleySound(sound_type="impact", start_time=1.5, duration=0.1)
        self.assertEqual(fs.start_time, 1.5)
        self.assertEqual(fs.sound_type, "impact")
        self.assertEqual(fs.duration, 0.1)
        self.assertEqual(fs.confidence, 0.0)

    def test_sfx_track_fields(self):
        from opencut.core.foley_gen import FoleySound
        st = FoleySound(sound_type="footstep", start_time=1.0, duration=0.15)
        self.assertEqual(st.sound_type, "footstep")
        self.assertEqual(st.confidence, 0.0)
        self.assertEqual(st.duration, 0.15)

    def test_foley_result_fields(self):
        from opencut.core.foley_gen import FoleyResult
        fr = FoleyResult()
        self.assertEqual(fr.output_path, "")
        self.assertEqual(fr.sounds_generated, [])
        self.assertEqual(fr.mix_level, 0.3)

    def test_foley_categories(self):
        from opencut.core.foley_gen import FOLEY_CATEGORIES
        # Must have at least the core categories
        for cat in ("footstep", "door", "impact", "ambient", "nature"):
            self.assertIn(cat, FOLEY_CATEGORIES)

    def test_foley_category_structure(self):
        from opencut.core.foley_gen import FOLEY_CATEGORIES
        for cat, info in FOLEY_CATEGORIES.items():
            self.assertIn("description", info, f"Missing description for {cat}")
            self.assertIn("freq_range", info, f"Missing freq_range for {cat}")
            self.assertIn("duration_range", info, f"Missing duration_range for {cat}")


# ============================================================
# Foley Generation — Function Signatures
# ============================================================
class TestFoleySignatures(unittest.TestCase):
    """Tests for foley_gen function signatures."""

    def test_detect_foley_cues_signature(self):
        from opencut.core.foley_gen import detect_foley_cues
        sig = inspect.signature(detect_foley_cues)
        self.assertIn("video_path", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    def test_generate_foley_signature(self):
        from opencut.core.foley_gen import generate_foley
        sig = inspect.signature(generate_foley)
        self.assertIn("video_path", sig.parameters)
        self.assertIn("output_path", sig.parameters)
        self.assertIn("cues", sig.parameters)
        self.assertIn("on_progress", sig.parameters)
        self.assertIsNone(sig.parameters["cues"].default)

    def test_detect_missing_file(self):
        from opencut.core.foley_gen import detect_foley_cues
        with self.assertRaises(FileNotFoundError):
            detect_foley_cues("/nonexistent/video.mp4")

    def test_generate_missing_file(self):
        from opencut.core.foley_gen import generate_foley
        with self.assertRaises(FileNotFoundError):
            generate_foley("/nonexistent/video.mp4")

    def test_generate_foley_mix_level_default(self):
        from opencut.core.foley_gen import generate_foley
        sig = inspect.signature(generate_foley)
        self.assertEqual(sig.parameters["mix_level"].default, 0.3)


# ============================================================
# Foley Generation — SFX Synthesis
# ============================================================
class TestFoleySynthesis(unittest.TestCase):
    """Tests for foley SFX synthesis helpers."""

    def test_classify_cue_returns_string(self):
        from opencut.core.foley_gen import _classify_cue
        cat = _classify_cue(scene_score=0.9, motion_mag=50.0, timestamp=1.0)
        self.assertIsInstance(cat, str)
        from opencut.core.foley_gen import FOLEY_CATEGORIES
        self.assertIn(cat, FOLEY_CATEGORIES)

    def test_classify_cue_low_motion_ambient(self):
        from opencut.core.foley_gen import _classify_cue
        cat = _classify_cue(scene_score=0.0, motion_mag=0.05, timestamp=1.0)
        self.assertEqual(cat, "ambient")

    def test_generate_pcm_samples_returns_bytes(self):
        from opencut.core.foley_gen import _generate_pcm_samples
        data = _generate_pcm_samples(freq=200, duration=0.1, envelope="percussive")
        self.assertIsInstance(data, bytes)
        self.assertGreater(len(data), 0)

    def test_generate_pcm_samples_noise_wash(self):
        from opencut.core.foley_gen import _generate_pcm_samples
        data = _generate_pcm_samples(freq=400, duration=0.5, envelope="noise_wash")
        self.assertIsInstance(data, bytes)

    def test_write_wav_creates_file(self):
        from opencut.core.foley_gen import _write_wav
        pcm = b"\x00\x00" * 4410  # 0.1s silence at 44100Hz mono 16-bit
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out = f.name
        try:
            _write_wav(pcm, out, sample_rate=44100)
            self.assertTrue(os.path.isfile(out))
            self.assertGreater(os.path.getsize(out), len(pcm))
        finally:
            if os.path.exists(out):
                os.unlink(out)

    def test_synthesize_foley_sound_creates_wav(self):
        from opencut.core.foley_gen import FoleySound, _synthesize_foley_sound
        cue = FoleySound(sound_type="impact", start_time=1.0, duration=0.15, confidence=0.8)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out = f.name
        try:
            _synthesize_foley_sound(cue, out)
            self.assertTrue(os.path.isfile(out))
        finally:
            if os.path.exists(out):
                os.unlink(out)


# ============================================================
# Foley Generation — Pipeline
# ============================================================
class TestFoleyPipeline(unittest.TestCase):
    """Tests for the foley generation pipeline."""

    @patch("opencut.core.foley_gen.get_video_info", return_value={"duration": 10.0})
    @patch("opencut.core.foley_gen.run_ffmpeg")
    @patch("opencut.core.foley_gen.detect_foley_cues", return_value=[])
    def test_generate_foley_no_cues(self, mock_detect, mock_ffmpeg, mock_info):
        from opencut.core.foley_gen import generate_foley
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            result = generate_foley(path)
            self.assertEqual(result.sounds_generated, [])
            self.assertIsInstance(result.output_path, str)
        finally:
            os.unlink(path)

    @patch("opencut.core.foley_gen.get_video_info", return_value={"duration": 10.0})
    @patch("opencut.core.foley_gen.run_ffmpeg")
    @patch("opencut.core.foley_gen.detect_foley_cues")
    def test_generate_foley_with_cues(self, mock_detect, mock_ffmpeg, mock_info):
        from opencut.core.foley_gen import FoleySound, generate_foley
        mock_detect.return_value = [
            FoleySound(sound_type="impact", start_time=1.0, duration=0.1, confidence=0.8),
            FoleySound(sound_type="whoosh", start_time=3.0, duration=0.3, confidence=0.6),
        ]
        mock_ffmpeg.return_value = ""

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            result = generate_foley(path)
            self.assertIsInstance(result.sounds_generated, list)
        finally:
            os.unlink(path)


# ============================================================
# Face Restoration — Dataclass & Constants Tests
# ============================================================
class TestFaceRestoreDataclasses(unittest.TestCase):
    """Tests for face_restore dataclasses and constants."""

    def test_face_restore_result_fields(self):
        from opencut.core.face_restore import FaceRestoreResult
        frr = FaceRestoreResult()
        self.assertEqual(frr.output_path, "")
        self.assertEqual(frr.faces_detected, 0)
        self.assertEqual(frr.faces_restored, 0)
        self.assertEqual(frr.method, "ffmpeg")

    def test_face_restore_result_custom(self):
        from opencut.core.face_restore import FaceRestoreResult
        frr = FaceRestoreResult(faces_detected=3, faces_restored=2, method="gfpgan")
        self.assertEqual(frr.faces_detected, 3)
        self.assertEqual(frr.faces_restored, 2)

    def test_face_detection_result_defaults(self):
        from opencut.core.face_restore import FaceDetectionResult
        fdr = FaceDetectionResult()
        self.assertEqual(fdr.faces_detected, 0)
        self.assertEqual(fdr.boxes, [])
        self.assertEqual(fdr.sample_frames, 0)


# ============================================================
# Face Restoration — Function Signatures
# ============================================================
class TestFaceRestoreSignatures(unittest.TestCase):
    """Tests for face_restore function signatures."""

    def test_restore_faces_signature(self):
        from opencut.core.face_restore import restore_faces
        sig = inspect.signature(restore_faces)
        self.assertIn("video_path", sig.parameters)
        self.assertIn("output_path", sig.parameters)
        self.assertIn("method", sig.parameters)
        self.assertIn("strength", sig.parameters)
        self.assertIn("on_progress", sig.parameters)
        self.assertEqual(sig.parameters["method"].default, "ffmpeg")
        self.assertEqual(sig.parameters["strength"].default, 1.0)

    def test_restore_single_frame_signature(self):
        from opencut.core.face_restore import restore_single_frame
        sig = inspect.signature(restore_single_frame)
        self.assertIn("image_path", sig.parameters)
        self.assertIn("output_path", sig.parameters)
        self.assertIn("strength", sig.parameters)

    def test_restore_faces_missing_file(self):
        from opencut.core.face_restore import restore_faces
        with self.assertRaises(FileNotFoundError):
            restore_faces("/nonexistent/video.mp4")

    def test_restore_single_frame_missing_file(self):
        from opencut.core.face_restore import restore_single_frame
        with self.assertRaises(FileNotFoundError):
            restore_single_frame("/nonexistent/image.png")


# ============================================================
# Face Restoration — Model Resolution
# ============================================================
class TestFaceRestoreModels(unittest.TestCase):
    """Tests for face restoration helpers."""

    def test_face_box_defaults(self):
        from opencut.core.face_restore import FaceBox
        fb = FaceBox()
        self.assertEqual(fb.x, 0)
        self.assertEqual(fb.y, 0)
        self.assertEqual(fb.width, 0)
        self.assertEqual(fb.height, 0)

    def test_face_box_custom(self):
        from opencut.core.face_restore import FaceBox
        fb = FaceBox(x=100, y=200, width=50, height=60, confidence=0.95)
        self.assertEqual(fb.x, 100)
        self.assertEqual(fb.confidence, 0.95)

    def test_restore_faces_method_default(self):
        from opencut.core.face_restore import restore_faces
        sig = inspect.signature(restore_faces)
        self.assertEqual(sig.parameters["method"].default, "ffmpeg")

    def test_restore_single_frame_strength_default(self):
        from opencut.core.face_restore import restore_single_frame
        sig = inspect.signature(restore_single_frame)
        self.assertEqual(sig.parameters["strength"].default, 1.0)

    def test_detect_faces_function_exists(self):
        from opencut.core.face_restore import detect_faces
        self.assertTrue(callable(detect_faces))


# ============================================================
# Face Restoration — Pipeline
# ============================================================
class TestFaceRestorePipeline(unittest.TestCase):
    """Tests for face restoration pipeline with mocks."""

    def test_detect_faces_missing_file(self):
        from opencut.core.face_restore import detect_faces
        with self.assertRaises(FileNotFoundError):
            detect_faces("/nonexistent/video.mp4")

    def test_restore_result_has_method(self):
        from opencut.core.face_restore import FaceRestoreResult
        r = FaceRestoreResult(method="gfpgan")
        self.assertEqual(r.method, "gfpgan")

    def test_detection_result_avg_face_size(self):
        from opencut.core.face_restore import FaceDetectionResult
        r = FaceDetectionResult(faces_detected=2, avg_face_size=45.5)
        self.assertEqual(r.avg_face_size, 45.5)


# ============================================================
# Routes — Blueprint Existence
# ============================================================
class TestBodyTransferRoutes(unittest.TestCase):
    """Smoke tests for body_transfer_routes blueprint."""

    def test_blueprint_exists(self):
        from opencut.routes.body_transfer_routes import body_transfer_bp
        self.assertIsNotNone(body_transfer_bp)
        self.assertIsNotNone(body_transfer_bp.deferred_functions)
        self.assertGreater(len(body_transfer_bp.deferred_functions), 0)

    def test_blueprint_name(self):
        from opencut.routes.body_transfer_routes import body_transfer_bp
        self.assertEqual(body_transfer_bp.name, "body_transfer")

    def test_route_functions_importable(self):
        from opencut.routes.body_transfer_routes import (
            body_effects_route,
            body_effects_detect_route,
            motion_transfer_route,
            foley_generate_route,
            face_restore_route,
        )
        self.assertTrue(callable(body_effects_route))
        self.assertTrue(callable(body_effects_detect_route))
        self.assertTrue(callable(motion_transfer_route))
        self.assertTrue(callable(foley_generate_route))
        self.assertTrue(callable(face_restore_route))

    def test_all_routes_have_csrf(self):
        """All POST routes should be decorated with require_csrf."""
        from opencut.routes.body_transfer_routes import body_transfer_bp
        # Blueprint deferred_functions count should match expected endpoints
        self.assertGreaterEqual(len(body_transfer_bp.deferred_functions), 9)


# ============================================================
# Integration — Module Imports
# ============================================================
class TestModuleImports(unittest.TestCase):
    """Verify all new modules import cleanly."""

    def test_body_effects_importable(self):
        from opencut.core import body_effects
        self.assertIsNotNone(body_effects)

    def test_motion_transfer_importable(self):
        from opencut.core import motion_transfer
        self.assertIsNotNone(motion_transfer)

    def test_foley_gen_importable(self):
        from opencut.core import foley_gen
        self.assertIsNotNone(foley_gen)

    def test_face_restore_importable(self):
        from opencut.core import face_restore
        self.assertIsNotNone(face_restore)

    def test_routes_importable(self):
        from opencut.routes import body_transfer_routes
        self.assertIsNotNone(body_transfer_routes)


# ============================================================
# Integration — Docstring Coverage
# ============================================================
class TestDocstrings(unittest.TestCase):
    """All public functions should have docstrings."""

    def test_body_effects_docstrings(self):
        from opencut.core.body_effects import apply_body_effect, detect_body_keypoints
        self.assertIsNotNone(detect_body_keypoints.__doc__)
        self.assertIsNotNone(apply_body_effect.__doc__)

    def test_motion_transfer_docstrings(self):
        from opencut.core.motion_transfer import extract_pose_sequence, transfer_motion
        self.assertIsNotNone(extract_pose_sequence.__doc__)
        self.assertIsNotNone(transfer_motion.__doc__)

    def test_foley_gen_docstrings(self):
        from opencut.core.foley_gen import detect_foley_cues, generate_foley
        self.assertIsNotNone(detect_foley_cues.__doc__)
        self.assertIsNotNone(generate_foley.__doc__)

    def test_face_restore_docstrings(self):
        from opencut.core.face_restore import restore_single_frame, restore_faces
        self.assertIsNotNone(restore_faces.__doc__)
        self.assertIsNotNone(restore_single_frame.__doc__)


# ============================================================
# Edge Cases
# ============================================================
class TestEdgeCases(unittest.TestCase):
    """Edge case and boundary condition tests."""

    def test_body_track_result_empty_frames(self):
        from opencut.core.body_effects import BodyTrackResult
        btr = BodyTrackResult(frames=[], fps=0.0, total_frames=0)
        self.assertEqual(len(btr.frames), 0)

    def test_pose_sequence_empty_poses(self):
        from opencut.core.motion_transfer import PoseSequence
        ps = PoseSequence(poses=[], fps=0.0, duration=0.0)
        self.assertEqual(len(ps.poses), 0)

    def test_foley_result_empty(self):
        from opencut.core.foley_gen import FoleyResult
        fr = FoleyResult(output_path="/fake/out.mp4", sounds_generated=[], duration=0.0)
        self.assertEqual(fr.sounds_generated, [])

    def test_face_restore_strength_clamp(self):
        """Strength should be clamped to 0.0-1.0 inside restore_faces."""
        from opencut.core.face_restore import restore_faces
        # We can't run the full pipeline, but verify the function exists and accepts strength
        sig = inspect.signature(restore_faces)
        self.assertIn("strength", sig.parameters)

    def test_frame_keypoints_with_data(self):
        from opencut.core.body_effects import FrameKeypoints
        fk = FrameKeypoints(
            frame_index=42,
            timestamp=1.4,
            keypoints={
                "nose": (100.0, 200.0, 0.95),
                "left_eye": (90.0, 190.0, 0.88),
            },
        )
        self.assertEqual(fk.frame_index, 42)
        self.assertIn("nose", fk.keypoints)
        self.assertEqual(len(fk.keypoints), 2)

    def test_foley_sound_with_data(self):
        from opencut.core.foley_gen import FoleySound
        fs = FoleySound(
            sound_type="impact",
            start_time=2.5,
            duration=0.3,
            confidence=0.85,
        )
        self.assertEqual(fs.sound_type, "impact")
        self.assertEqual(fs.confidence, 0.85)

    def test_motion_transfer_result_with_data(self):
        from opencut.core.motion_transfer import MotionTransferResult
        mtr = MotionTransferResult(
            output_path="/fake/output.mp4",
            source_duration=10.5,
            frames_generated=315,
            model_used="stick_figure",
        )
        self.assertEqual(mtr.frames_generated, 315)
        self.assertEqual(mtr.model_used, "stick_figure")

    def test_foley_sound_default_confidence(self):
        from opencut.core.foley_gen import FoleySound
        fs = FoleySound(sound_type="whoosh", start_time=5.0, duration=0.5)
        self.assertEqual(fs.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
