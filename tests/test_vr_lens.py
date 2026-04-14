"""
Unit tests for VR/360 and Lens Correction features (Sections 51-52):

51.1 - 360 Video Stabilization (vr_stabilize)
51.2 - Equirectangular to Flat / Keyframed Reframe (video_360 enhancements)
51.3 - FOV Region Extraction from 360 (video_360 enhancements)
51.4 - Spatial Audio Alignment for VR (spatial_audio_vr)
52.1 - Lens Distortion Correction with Camera Profiles (lens_correction enhancements)
52.2 - Rolling Shutter Enhanced Correction (rolling_shutter enhancements)
52.3 - Chromatic Aberration Removal (chromatic_aberration)
52.4 - Lens Profile Auto-Detection (lens_profile)
+ Route smoke tests for vr_lens_routes
"""

import json
import math
import os
import struct
from unittest.mock import MagicMock, patch, mock_open

import pytest


# ========================================================================
# 51.1 VR Stabilize
# ========================================================================
class TestVRStabilize:
    """Tests for opencut.core.vr_stabilize."""

    def test_gyro_sample_dataclass(self):
        from opencut.core.vr_stabilize import GyroSample
        s = GyroSample(timestamp=1.0, yaw=0.5, pitch=-0.3, roll=0.1)
        assert s.timestamp == 1.0
        assert s.yaw == 0.5
        assert s.pitch == -0.3
        assert s.roll == 0.1

    def test_vr_stabilize_result_dataclass(self):
        from opencut.core.vr_stabilize import VRStabilizeResult
        r = VRStabilizeResult()
        assert r.output_path == ""
        assert r.method == "visual"
        assert r.gyro_samples == 0
        assert r.smoothing == 10
        assert r.frames_processed == 0
        assert r.camera_model == ""

    def test_stabilize_vr_file_not_found(self):
        from opencut.core.vr_stabilize import stabilize_vr
        with pytest.raises(FileNotFoundError):
            stabilize_vr("/nonexistent/video.mp4")

    def test_detect_gyro_source_file_not_found(self):
        from opencut.core.vr_stabilize import detect_gyro_source
        with pytest.raises(FileNotFoundError):
            detect_gyro_source("/nonexistent/video.mp4")

    def test_smooth_gyro_empty(self):
        from opencut.core.vr_stabilize import _smooth_gyro
        assert _smooth_gyro([], 10) == []

    def test_smooth_gyro_low_smoothing(self):
        from opencut.core.vr_stabilize import GyroSample, _smooth_gyro
        samples = [GyroSample(0.1, 1.0, 0.0, 0.0)]
        result = _smooth_gyro(samples, 1)
        assert result == samples

    def test_smooth_gyro_averaging(self):
        from opencut.core.vr_stabilize import GyroSample, _smooth_gyro
        samples = [
            GyroSample(0.0, 1.0, 0.0, 0.0),
            GyroSample(0.1, 3.0, 0.0, 0.0),
            GyroSample(0.2, 5.0, 0.0, 0.0),
        ]
        result = _smooth_gyro(samples, 3)
        assert len(result) == 3
        # Middle sample should be averaged
        assert abs(result[1].yaw - 3.0) < 0.01

    def test_gyro_to_v360_script_empty(self):
        from opencut.core.vr_stabilize import _gyro_to_v360_script
        expr = _gyro_to_v360_script([], 30.0, 10.0)
        assert expr == "v360=e:e"

    def test_gyro_to_v360_script_with_samples(self):
        from opencut.core.vr_stabilize import GyroSample, _gyro_to_v360_script
        samples = [
            GyroSample(0.0, 1.0, 0.5, -0.2),
            GyroSample(0.033, 1.5, 0.3, 0.1),
        ]
        expr = _gyro_to_v360_script(samples, 30.0, 0.1)
        assert "v360=e:e" in expr
        assert "yaw=" in expr
        assert "pitch=" in expr
        assert "roll=" in expr

    @patch("opencut.core.vr_stabilize.os.path.isfile", return_value=True)
    @patch("opencut.core.vr_stabilize.get_video_info")
    @patch("opencut.core.vr_stabilize.detect_gyro_source")
    @patch("opencut.core.vr_stabilize.run_ffmpeg")
    def test_stabilize_vr_visual_fallback(self, mock_ffmpeg, mock_gyro, mock_info, mock_isfile):
        from opencut.core.vr_stabilize import stabilize_vr
        mock_info.return_value = {"fps": 30.0, "duration": 10.0, "width": 3840, "height": 1920}
        mock_gyro.return_value = ("none", [])

        result = stabilize_vr("/test/video.mp4")
        assert result.method == "visual"
        assert result.gyro_samples == 0

    @patch("opencut.core.vr_stabilize.os.path.isfile", return_value=True)
    @patch("opencut.core.vr_stabilize.get_video_info")
    @patch("opencut.core.vr_stabilize.run_ffmpeg")
    def test_stabilize_vr_force_visual(self, mock_ffmpeg, mock_info, mock_isfile):
        from opencut.core.vr_stabilize import stabilize_vr
        mock_info.return_value = {"fps": 30.0, "duration": 10.0, "width": 3840, "height": 1920}

        result = stabilize_vr("/test/video.mp4", force_visual=True)
        assert result.method == "visual"

    @patch("opencut.core.vr_stabilize.os.path.isfile", return_value=True)
    @patch("opencut.core.vr_stabilize.get_video_info")
    @patch("opencut.core.vr_stabilize.detect_gyro_source")
    @patch("opencut.core.vr_stabilize.run_ffmpeg")
    def test_stabilize_vr_with_gyro(self, mock_ffmpeg, mock_gyro, mock_info, mock_isfile):
        from opencut.core.vr_stabilize import GyroSample, stabilize_vr
        mock_info.return_value = {"fps": 30.0, "duration": 10.0, "width": 3840, "height": 1920}
        mock_gyro.return_value = ("gpmf", [
            GyroSample(0.0, 0.5, 0.1, 0.0),
            GyroSample(0.1, 0.3, -0.1, 0.0),
        ])

        result = stabilize_vr("/test/video.mp4")
        assert result.method == "gyro_gpmf"
        assert result.gyro_samples == 2

    def test_stabilize_vr_smoothing_clamped(self):
        from opencut.core.vr_stabilize import stabilize_vr
        with pytest.raises(FileNotFoundError):
            stabilize_vr("/missing.mp4", smoothing=999)

    def test_parse_gpmf_gyro_no_file(self):
        from opencut.core.vr_stabilize import _parse_gpmf_gyro
        result = _parse_gpmf_gyro("/nonexistent/video.mp4")
        assert result == []

    def test_parse_insta360_gyro_no_file(self):
        from opencut.core.vr_stabilize import _parse_insta360_gyro
        result = _parse_insta360_gyro("/nonexistent/video.mp4")
        assert result == []


# ========================================================================
# 51.2 Equirectangular to Flat / Keyframed Reframe
# ========================================================================
class TestEquirectToFlat:
    """Tests for equirect_to_flat and keyframed_reframe."""

    def test_reframe_result_dataclass(self):
        from opencut.core.video_360 import ReframeResult
        r = ReframeResult()
        assert r.output_path == ""
        assert r.width == 1920
        assert r.height == 1080
        assert r.keyframes_used == 0
        assert r.method == "static"

    def test_reframe_keyframe_dataclass(self):
        from opencut.core.video_360 import ReframeKeyframe
        kf = ReframeKeyframe(time=1.0, yaw=45.0, pitch=-10.0, roll=5.0)
        assert kf.time == 1.0
        assert kf.yaw == 45.0
        assert kf.h_fov == 90.0

    def test_equirect_to_flat_file_not_found(self):
        from opencut.core.video_360 import equirect_to_flat
        with pytest.raises(FileNotFoundError):
            equirect_to_flat("/nonexistent/video.mp4")

    @patch("opencut.core.video_360.os.path.isfile", return_value=True)
    @patch("opencut.core.video_360.get_video_info")
    @patch("opencut.core.video_360.run_ffmpeg")
    def test_equirect_to_flat_basic(self, mock_ffmpeg, mock_info, mock_isfile):
        from opencut.core.video_360 import equirect_to_flat
        mock_info.return_value = {"width": 3840, "height": 1920, "duration": 10.0}

        result = equirect_to_flat("/test/360.mp4", yaw=45.0, pitch=10.0)
        assert result.output_path.endswith(".mp4")
        assert result.method == "static"
        assert result.width == 1920
        assert result.height == 1080

    @patch("opencut.core.video_360.os.path.isfile", return_value=True)
    @patch("opencut.core.video_360.get_video_info")
    @patch("opencut.core.video_360.run_ffmpeg")
    def test_equirect_to_flat_with_roll(self, mock_ffmpeg, mock_info, mock_isfile):
        from opencut.core.video_360 import equirect_to_flat
        mock_info.return_value = {"width": 3840, "height": 1920, "duration": 10.0}

        result = equirect_to_flat("/test/360.mp4", roll=15.0)
        assert result.output_path != ""

    @patch("opencut.core.video_360.os.path.isfile", return_value=True)
    @patch("opencut.core.video_360.get_video_info")
    @patch("opencut.core.video_360.run_ffmpeg")
    def test_equirect_to_flat_custom_fov(self, mock_ffmpeg, mock_info, mock_isfile):
        from opencut.core.video_360 import equirect_to_flat
        mock_info.return_value = {"width": 3840, "height": 1920, "duration": 10.0}

        result = equirect_to_flat("/test/360.mp4", h_fov=120.0, w_fov=70.0)
        assert result.output_path != ""

    def test_equirect_to_flat_clamps_values(self):
        from opencut.core.video_360 import equirect_to_flat
        with pytest.raises(FileNotFoundError):
            equirect_to_flat("/missing.mp4", yaw=999, pitch=999, h_fov=999)

    def test_interpolate_keyframes_empty(self):
        from opencut.core.video_360 import _interpolate_keyframes
        yaw, pitch, roll, h_fov, w_fov = _interpolate_keyframes([], 1.0)
        assert yaw == 0.0
        assert h_fov == 90.0

    def test_interpolate_keyframes_single(self):
        from opencut.core.video_360 import ReframeKeyframe, _interpolate_keyframes
        kfs = [ReframeKeyframe(time=0.0, yaw=45.0, pitch=10.0)]
        yaw, pitch, roll, h_fov, w_fov = _interpolate_keyframes(kfs, 5.0)
        assert yaw == 45.0
        assert pitch == 10.0

    def test_interpolate_keyframes_two(self):
        from opencut.core.video_360 import ReframeKeyframe, _interpolate_keyframes
        kfs = [
            ReframeKeyframe(time=0.0, yaw=0.0, pitch=0.0),
            ReframeKeyframe(time=10.0, yaw=90.0, pitch=30.0),
        ]
        # At midpoint, cosine interpolation should give ~45.0 yaw
        yaw, pitch, roll, h_fov, w_fov = _interpolate_keyframes(kfs, 5.0)
        assert 40.0 < yaw < 50.0
        assert 10.0 < pitch < 20.0

    def test_interpolate_keyframes_before_first(self):
        from opencut.core.video_360 import ReframeKeyframe, _interpolate_keyframes
        kfs = [ReframeKeyframe(time=5.0, yaw=45.0)]
        yaw, pitch, roll, h_fov, w_fov = _interpolate_keyframes(kfs, 0.0)
        assert yaw == 45.0

    def test_interpolate_keyframes_after_last(self):
        from opencut.core.video_360 import ReframeKeyframe, _interpolate_keyframes
        kfs = [ReframeKeyframe(time=0.0, yaw=45.0)]
        yaw, pitch, roll, h_fov, w_fov = _interpolate_keyframes(kfs, 100.0)
        assert yaw == 45.0

    def test_keyframed_reframe_file_not_found(self):
        from opencut.core.video_360 import keyframed_reframe
        with pytest.raises(FileNotFoundError):
            keyframed_reframe("/nonexistent/video.mp4", [{"time": 0, "yaw": 0}])

    @patch("opencut.core.video_360.os.path.isfile", return_value=True)
    def test_keyframed_reframe_no_keyframes(self, mock_isfile):
        from opencut.core.video_360 import keyframed_reframe
        with pytest.raises(ValueError, match="At least one keyframe"):
            keyframed_reframe("/test/video.mp4", [])

    @patch("opencut.core.video_360.os.path.isfile", return_value=True)
    @patch("opencut.core.video_360.get_video_info")
    @patch("opencut.core.video_360.run_ffmpeg")
    def test_keyframed_reframe_single_keyframe(self, mock_ffmpeg, mock_info, mock_isfile):
        from opencut.core.video_360 import keyframed_reframe
        mock_info.return_value = {"width": 3840, "height": 1920, "duration": 10.0, "fps": 30.0}

        result = keyframed_reframe(
            "/test/360.mp4",
            [{"time": 0, "yaw": 45, "pitch": 0}],
        )
        assert result.output_path != ""

    @patch("opencut.core.video_360.shutil")
    @patch("opencut.core.video_360.os.path.isfile", return_value=True)
    @patch("opencut.core.video_360.get_video_info")
    @patch("opencut.core.video_360.run_ffmpeg")
    def test_keyframed_reframe_multiple_keyframes(self, mock_ffmpeg, mock_info, mock_isfile, mock_shutil):
        from opencut.core.video_360 import keyframed_reframe
        mock_info.return_value = {"width": 3840, "height": 1920, "duration": 3.0, "fps": 30.0}

        result = keyframed_reframe(
            "/test/360.mp4",
            [
                {"time": 0, "yaw": 0, "pitch": 0},
                {"time": 1.5, "yaw": 90, "pitch": 20},
                {"time": 3.0, "yaw": 180, "pitch": 0},
            ],
        )
        assert result.method == "keyframed"
        assert result.keyframes_used == 3


# ========================================================================
# 51.3 FOV Region Extraction
# ========================================================================
class TestFOVExtraction:
    """Tests for extract_fov_regions."""

    def test_fov_region_dataclass(self):
        from opencut.core.video_360 import FOVRegion
        r = FOVRegion(label="test", yaw=45.0, pitch=0.0, h_fov=90.0, confidence=0.8)
        assert r.label == "test"
        assert r.yaw == 45.0

    def test_fov_extraction_result_dataclass(self):
        from opencut.core.video_360 import FOVExtractionResult
        r = FOVExtractionResult()
        assert r.regions == []
        assert r.total_regions == 0

    def test_extract_fov_regions_file_not_found(self):
        from opencut.core.video_360 import extract_fov_regions
        with pytest.raises(FileNotFoundError):
            extract_fov_regions("/nonexistent/video.mp4")

    @patch("opencut.core.video_360.os.makedirs")
    @patch("opencut.core.video_360.os.path.isfile", return_value=True)
    @patch("opencut.core.video_360.get_video_info")
    @patch("opencut.core.video_360._detect_motion_regions")
    @patch("opencut.core.video_360.run_ffmpeg")
    def test_extract_fov_regions_with_defaults(self, mock_ffmpeg, mock_motion, mock_info, mock_isfile, mock_makedirs):
        from opencut.core.video_360 import extract_fov_regions
        mock_info.return_value = {"width": 3840, "height": 1920, "duration": 10.0, "fps": 30.0}
        mock_motion.return_value = [(0.0, 0.0, 0.8), (90.0, 10.0, 0.6)]

        # Disable XML generation to avoid file I/O on mocked paths
        result = extract_fov_regions("/test/360.mp4", num_regions=4, generate_xml=False)
        assert result.total_regions >= 2

    def test_generate_multicam_xml(self):
        from opencut.core.video_360 import FOVRegion, _generate_multicam_xml
        import tempfile

        regions = [
            FOVRegion(label="front", yaw=0, pitch=0, output_path="/test/front.mp4"),
            FOVRegion(label="right", yaw=90, pitch=0, output_path="/test/right.mp4"),
        ]
        tmp = os.path.join(tempfile.gettempdir(), "test_multicam.fcpxml")

        try:
            path = _generate_multicam_xml(regions, 10.0, 30.0, tmp)
            assert os.path.isfile(path)
            with open(path) as f:
                content = f.read()
            assert "fcpxml" in content
            assert "front" in content
            assert "right" in content
            assert "multicam-clip" in content
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass


# ========================================================================
# 51.4 Spatial Audio VR
# ========================================================================
class TestSpatialAudioVR:
    """Tests for opencut.core.spatial_audio_vr."""

    def test_speaker_position_dataclass(self):
        from opencut.core.spatial_audio_vr import SpeakerPosition
        sp = SpeakerPosition(label="front", azimuth=0.0, elevation=0.0)
        assert sp.label == "front"
        assert sp.distance == 1.0

    def test_spatial_audio_result_dataclass(self):
        from opencut.core.spatial_audio_vr import SpatialAudioResult
        r = SpatialAudioResult()
        assert r.channels == 4
        assert r.format == "ambix"
        assert r.sample_rate == 48000

    def test_pixel_to_spherical_center(self):
        from opencut.core.spatial_audio_vr import pixel_to_spherical
        az, el = pixel_to_spherical(1920, 960, 3840, 1920)
        assert abs(az - 0.0) < 0.1
        assert abs(el - 0.0) < 0.1

    def test_pixel_to_spherical_edges(self):
        from opencut.core.spatial_audio_vr import pixel_to_spherical
        az, el = pixel_to_spherical(0, 0, 3840, 1920)
        assert abs(az - (-180.0)) < 0.1
        assert abs(el - 90.0) < 0.1

    def test_spherical_to_pixel_roundtrip(self):
        from opencut.core.spatial_audio_vr import pixel_to_spherical, spherical_to_pixel
        w, h = 3840, 1920
        x, y = 960, 480
        az, el = pixel_to_spherical(x, y, w, h)
        rx, ry = spherical_to_pixel(az, el, w, h)
        assert abs(rx - x) <= 1
        assert abs(ry - y) <= 1

    def test_encode_ambisonics_front(self):
        from opencut.core.spatial_audio_vr import _encode_ambisonics_coefficients
        w, y, z, x = _encode_ambisonics_coefficients(0.0, 0.0)
        assert abs(w - 1.0) < 0.01
        assert abs(y - 0.0) < 0.01
        assert abs(z - 0.0) < 0.01
        assert abs(x - 1.0) < 0.01

    def test_encode_ambisonics_left(self):
        from opencut.core.spatial_audio_vr import _encode_ambisonics_coefficients
        w, y, z, x = _encode_ambisonics_coefficients(90.0, 0.0)
        assert abs(w - 1.0) < 0.01
        assert abs(y - 1.0) < 0.01  # left channel
        assert abs(z - 0.0) < 0.01
        assert abs(x - 0.0) < 0.1   # ~0 for 90 degrees

    def test_encode_ambisonics_above(self):
        from opencut.core.spatial_audio_vr import _encode_ambisonics_coefficients
        w, y, z, x = _encode_ambisonics_coefficients(0.0, 90.0)
        assert abs(w - 1.0) < 0.01
        assert abs(z - 1.0) < 0.01  # up channel

    def test_encode_ambisonics_distance_attenuation(self):
        from opencut.core.spatial_audio_vr import _encode_ambisonics_coefficients
        w_near, _, _, _ = _encode_ambisonics_coefficients(0.0, 0.0, 0.5)
        w_far, _, _, _ = _encode_ambisonics_coefficients(0.0, 0.0, 2.0)
        assert w_near > w_far

    def test_build_ambisonics_filter_empty(self):
        from opencut.core.spatial_audio_vr import _build_ambisonics_filter
        assert _build_ambisonics_filter([], 2) == ""

    def test_build_ambisonics_filter_single(self):
        from opencut.core.spatial_audio_vr import SpeakerPosition, _build_ambisonics_filter
        speakers = [SpeakerPosition(label="center", azimuth=0.0, elevation=0.0)]
        filt = _build_ambisonics_filter(speakers, 2)
        assert "pan=mono" in filt
        assert "pan=4c" in filt
        assert "ambiout" in filt

    def test_build_ambisonics_filter_multi(self):
        from opencut.core.spatial_audio_vr import SpeakerPosition, _build_ambisonics_filter
        speakers = [
            SpeakerPosition(label="left", azimuth=-45.0),
            SpeakerPosition(label="right", azimuth=45.0, audio_track=1),
        ]
        filt = _build_ambisonics_filter(speakers, 2)
        assert "amix" in filt
        assert "inputs=2" in filt

    def test_spatialize_audio_file_not_found(self):
        from opencut.core.spatial_audio_vr import spatialize_audio
        with pytest.raises(FileNotFoundError):
            spatialize_audio("/nonexistent/video.mp4")

    @patch("opencut.core.spatial_audio_vr.subprocess.run")
    @patch("opencut.core.spatial_audio_vr.os.path.isfile", return_value=True)
    @patch("opencut.core.spatial_audio_vr.get_video_info")
    @patch("opencut.core.spatial_audio_vr.run_ffmpeg")
    def test_spatialize_audio_manual_speakers(self, mock_ffmpeg, mock_info, mock_isfile, mock_sub):
        from opencut.core.spatial_audio_vr import spatialize_audio
        mock_info.return_value = {"width": 3840, "height": 1920, "duration": 10.0, "fps": 30.0}
        mock_sub.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"streams": [{"channels": 2}]}),
        )

        result = spatialize_audio(
            "/test/360.mp4",
            speakers=[{"label": "center", "azimuth": 0, "elevation": 0}],
            auto_detect=False,
        )
        assert result.speakers == 1
        assert result.channels == 4
        assert result.format == "ambix"


# ========================================================================
# 52.1 Camera Profile Database + Auto-Detection
# ========================================================================
class TestCameraProfiles:
    """Tests for camera profile database and auto-detection."""

    def test_camera_profile_dataclass(self):
        from opencut.core.lens_correction import CameraProfile
        p = CameraProfile(camera_model="GoPro", k1=-0.3)
        assert p.camera_model == "GoPro"
        assert p.k1 == -0.3

    def test_camera_detection_result_dataclass(self):
        from opencut.core.lens_correction import CameraDetectionResult
        r = CameraDetectionResult()
        assert not r.detected
        assert r.confidence == 0.0

    def test_camera_profiles_database_not_empty(self):
        from opencut.core.lens_correction import CAMERA_PROFILES
        assert len(CAMERA_PROFILES) > 10

    def test_camera_profiles_have_required_fields(self):
        from opencut.core.lens_correction import CAMERA_PROFILES
        for key, profile in CAMERA_PROFILES.items():
            assert profile.camera_model != "", f"Profile {key} missing camera_model"
            assert profile.category != "", f"Profile {key} missing category"

    def test_camera_model_map_not_empty(self):
        from opencut.core.lens_correction import _CAMERA_MODEL_MAP
        assert len(_CAMERA_MODEL_MAP) > 5

    def test_list_camera_profiles_all(self):
        from opencut.core.lens_correction import list_camera_profiles
        profiles = list_camera_profiles()
        assert len(profiles) > 10
        assert all("id" in p for p in profiles)
        assert all("camera_model" in p for p in profiles)

    def test_list_camera_profiles_filtered(self):
        from opencut.core.lens_correction import list_camera_profiles
        drones = list_camera_profiles(category="drone")
        assert all(p["category"] == "drone" for p in drones)
        assert len(drones) > 0

    def test_auto_detect_camera_file_not_found(self):
        from opencut.core.lens_correction import auto_detect_camera
        with pytest.raises(FileNotFoundError):
            auto_detect_camera("/nonexistent/video.mp4")

    @patch("opencut.core.lens_correction.subprocess.run")
    @patch("opencut.core.lens_correction.os.path.isfile", return_value=True)
    def test_auto_detect_camera_gopro(self, mock_isfile, mock_sub):
        from opencut.core.lens_correction import auto_detect_camera
        mock_sub.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({
                "format": {
                    "tags": {"com.apple.quicktime.model": "GoPro HERO12 Black"}
                },
                "streams": [{"tags": {"handler_name": "GoPro MET"}}],
            }),
        )

        result = auto_detect_camera("/test/gopro.mp4")
        assert result.detected
        assert "gopro" in result.camera_model.lower() or "hero12" in result.camera_model.lower()
        assert result.profile is not None
        assert result.suggested_k1 != 0.0

    @patch("opencut.core.lens_correction.subprocess.run")
    @patch("opencut.core.lens_correction.os.path.isfile", return_value=True)
    def test_auto_detect_camera_iphone(self, mock_isfile, mock_sub):
        from opencut.core.lens_correction import auto_detect_camera
        mock_sub.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({
                "format": {
                    "tags": {"com.apple.quicktime.model": "iPhone 15 Pro"}
                },
                "streams": [],
            }),
        )

        result = auto_detect_camera("/test/iphone.mp4")
        assert result.detected
        assert result.profile is not None

    @patch("opencut.core.lens_correction.subprocess.run")
    @patch("opencut.core.lens_correction.os.path.isfile", return_value=True)
    def test_auto_detect_camera_unknown(self, mock_isfile, mock_sub):
        from opencut.core.lens_correction import auto_detect_camera
        mock_sub.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"format": {"tags": {}}, "streams": []}),
        )

        result = auto_detect_camera("/test/unknown.mp4")
        assert not result.detected

    @patch("opencut.core.lens_correction.subprocess.run")
    @patch("opencut.core.lens_correction.os.path.isfile", return_value=True)
    def test_auto_detect_camera_dji(self, mock_isfile, mock_sub):
        from opencut.core.lens_correction import auto_detect_camera
        mock_sub.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({
                "format": {
                    "tags": {"com.apple.quicktime.model": "FC3582"}
                },
                "streams": [],
            }),
        )

        result = auto_detect_camera("/test/dji.mp4")
        assert result.detected

    @patch("opencut.core.lens_correction.os.path.isfile", return_value=True)
    @patch("opencut.core.lens_correction.auto_detect_camera")
    @patch("opencut.core.lens_correction.correct_lens_distortion")
    def test_correct_with_profile_manual(self, mock_correct, mock_detect, mock_isfile):
        from opencut.core.lens_correction import correct_with_profile
        mock_correct.return_value = {"output_path": "/test/output.mp4"}

        result = correct_with_profile("/test/video.mp4", profile_id="gopro_hero12_wide")
        assert result["camera_model"] == "GoPro HERO12"
        assert result["k1"] == -0.30

    def test_correct_with_profile_unknown_id(self):
        from opencut.core.lens_correction import correct_with_profile
        with pytest.raises(ValueError, match="Unknown profile"):
            correct_with_profile("/test/video.mp4", profile_id="nonexistent_camera")

    def test_lens_presets_still_work(self):
        from opencut.core.lens_correction import list_lens_presets
        presets = list_lens_presets()
        assert len(presets) > 0
        assert all("id" in p for p in presets)


# ========================================================================
# 52.2 Rolling Shutter Enhanced
# ========================================================================
class TestRollingShutterEnhanced:
    """Tests for enhanced rolling shutter correction."""

    def test_rs_correction_result_dataclass(self):
        from opencut.core.rolling_shutter import RSCorrectionResult
        r = RSCorrectionResult()
        assert r.output_path == ""
        assert r.method == "per_row"
        assert r.readout_time_ms == 0.0

    @patch("opencut.core.rolling_shutter.os.path.isfile", return_value=True)
    @patch("opencut.core.rolling_shutter.get_video_info")
    def test_estimate_readout_time_30fps(self, mock_info, mock_isfile):
        from opencut.core.rolling_shutter import estimate_readout_time
        mock_info.return_value = {"fps": 30.0, "height": 1080}
        rt = estimate_readout_time("/test/video.mp4")
        assert 20.0 < rt < 30.0  # ~25ms for 30fps 1080p

    @patch("opencut.core.rolling_shutter.os.path.isfile", return_value=True)
    @patch("opencut.core.rolling_shutter.get_video_info")
    def test_estimate_readout_time_60fps(self, mock_info, mock_isfile):
        from opencut.core.rolling_shutter import estimate_readout_time
        mock_info.return_value = {"fps": 60.0, "height": 1080}
        rt = estimate_readout_time("/test/video.mp4")
        assert 10.0 < rt < 15.0  # ~12.5ms for 60fps 1080p

    @patch("opencut.core.rolling_shutter.os.path.isfile", return_value=True)
    @patch("opencut.core.rolling_shutter.get_video_info")
    def test_estimate_readout_4k(self, mock_info, mock_isfile):
        from opencut.core.rolling_shutter import estimate_readout_time
        mock_info.return_value = {"fps": 30.0, "height": 2160}
        rt = estimate_readout_time("/test/video.mp4")
        # 4K has higher readout ratio
        assert rt > 25.0

    def test_enhanced_rs_correction_file_not_found(self):
        from opencut.core.rolling_shutter import correct_rolling_shutter_enhanced
        with pytest.raises(FileNotFoundError):
            correct_rolling_shutter_enhanced("/nonexistent/video.mp4")

    @patch("opencut.core.rolling_shutter.os.path.isfile", return_value=True)
    @patch("opencut.core.rolling_shutter.get_video_info")
    @patch("opencut.core.rolling_shutter.run_ffmpeg")
    def test_enhanced_rs_correction_basic(self, mock_ffmpeg, mock_info, mock_isfile):
        from opencut.core.rolling_shutter import correct_rolling_shutter_enhanced
        mock_info.return_value = {"fps": 30.0, "height": 1080, "duration": 10.0}

        result = correct_rolling_shutter_enhanced("/test/video.mp4", strength=0.5)
        assert result.method == "per_row"
        assert result.readout_time_ms > 0
        assert result.rows_analyzed == 1080

    @patch("opencut.core.rolling_shutter.os.path.isfile", return_value=True)
    @patch("opencut.core.rolling_shutter.get_video_info")
    @patch("opencut.core.rolling_shutter.run_ffmpeg")
    def test_enhanced_rs_correction_high_strength(self, mock_ffmpeg, mock_info, mock_isfile):
        from opencut.core.rolling_shutter import correct_rolling_shutter_enhanced
        mock_info.return_value = {"fps": 30.0, "height": 1080, "duration": 10.0}

        result = correct_rolling_shutter_enhanced("/test/video.mp4", strength=0.9)
        assert result.strength == 0.9

    @patch("opencut.core.rolling_shutter.os.path.isfile", return_value=True)
    @patch("opencut.core.rolling_shutter.get_video_info")
    @patch("opencut.core.rolling_shutter.run_ffmpeg")
    def test_enhanced_rs_custom_readout(self, mock_ffmpeg, mock_info, mock_isfile):
        from opencut.core.rolling_shutter import correct_rolling_shutter_enhanced
        mock_info.return_value = {"fps": 30.0, "height": 1080, "duration": 10.0}

        result = correct_rolling_shutter_enhanced(
            "/test/video.mp4", readout_time_ms=20.0
        )
        assert result.readout_time_ms == 20.0

    def test_original_rs_still_works(self):
        from opencut.core.rolling_shutter import correct_rolling_shutter
        with pytest.raises(FileNotFoundError):
            correct_rolling_shutter("/nonexistent/video.mp4")


# ========================================================================
# 52.3 Chromatic Aberration
# ========================================================================
class TestChromaticAberration:
    """Tests for opencut.core.chromatic_aberration."""

    def test_ca_detection_result_dataclass(self):
        from opencut.core.chromatic_aberration import CADetectionResult
        r = CADetectionResult()
        assert not r.detected
        assert r.severity == "none"
        assert r.red_shift_x == 0

    def test_ca_correction_result_dataclass(self):
        from opencut.core.chromatic_aberration import CACorrectionResult
        r = CACorrectionResult()
        assert r.output_path == ""
        assert r.method == "auto"

    def test_detect_ca_file_not_found(self):
        from opencut.core.chromatic_aberration import detect_chromatic_aberration
        with pytest.raises(FileNotFoundError):
            detect_chromatic_aberration("/nonexistent/video.mp4")

    def test_correct_ca_file_not_found(self):
        from opencut.core.chromatic_aberration import correct_chromatic_aberration
        with pytest.raises(FileNotFoundError):
            correct_chromatic_aberration("/nonexistent/video.mp4")

    @patch("opencut.core.chromatic_aberration.os.path.isfile", return_value=True)
    @patch("opencut.core.chromatic_aberration.get_video_info")
    @patch("opencut.core.chromatic_aberration.detect_chromatic_aberration")
    @patch("opencut.core.chromatic_aberration.run_ffmpeg")
    def test_correct_ca_auto_detect(self, mock_ffmpeg, mock_detect, mock_info, mock_isfile):
        from opencut.core.chromatic_aberration import (
            CADetectionResult,
            correct_chromatic_aberration,
        )
        mock_info.return_value = {"width": 1920, "height": 1080, "duration": 10.0}
        mock_detect.return_value = CADetectionResult(
            detected=True, severity="moderate",
            red_shift_x=-1, blue_shift_x=1, confidence=0.7,
        )

        result = correct_chromatic_aberration("/test/video.mp4")
        assert result.method == "auto"
        assert result.red_shift_x == -1
        assert result.blue_shift_x == 1

    @patch("opencut.core.chromatic_aberration.os.path.isfile", return_value=True)
    @patch("opencut.core.chromatic_aberration.get_video_info")
    @patch("opencut.core.chromatic_aberration.run_ffmpeg")
    def test_correct_ca_manual(self, mock_ffmpeg, mock_info, mock_isfile):
        from opencut.core.chromatic_aberration import correct_chromatic_aberration
        mock_info.return_value = {"width": 1920, "height": 1080}

        result = correct_chromatic_aberration(
            "/test/video.mp4",
            red_shift_x=-2, blue_shift_x=1,
            auto_detect=False,
        )
        assert result.method == "manual"
        assert result.red_shift_x == -2
        assert result.blue_shift_x == 1

    def test_correct_ca_clamps_shifts(self):
        from opencut.core.chromatic_aberration import correct_chromatic_aberration
        with pytest.raises(FileNotFoundError):
            correct_chromatic_aberration(
                "/missing.mp4",
                red_shift_x=-50, auto_detect=False,
            )

    @patch("opencut.core.chromatic_aberration.os.path.isfile", return_value=True)
    @patch("opencut.core.chromatic_aberration.get_video_info")
    def test_detect_ca_zero_duration(self, mock_info, mock_isfile):
        from opencut.core.chromatic_aberration import detect_chromatic_aberration
        mock_info.return_value = {"width": 1920, "height": 1080, "duration": 0.0}
        result = detect_chromatic_aberration("/test/video.mp4")
        assert not result.detected


# ========================================================================
# 52.4 Lens Profile Auto-Detection
# ========================================================================
class TestLensProfile:
    """Tests for opencut.core.lens_profile."""

    def test_lens_info_dataclass(self):
        from opencut.core.lens_profile import LensInfo
        info = LensInfo()
        assert info.camera_make == ""
        assert info.frame_rate == 0.0

    def test_auto_correction_result_dataclass(self):
        from opencut.core.lens_profile import AutoCorrectionResult
        r = AutoCorrectionResult()
        assert r.output_path == ""
        assert r.corrections_applied == []

    def test_get_lens_info_file_not_found(self):
        from opencut.core.lens_profile import get_lens_info
        with pytest.raises(FileNotFoundError):
            get_lens_info("/nonexistent/video.mp4")

    @patch("opencut.core.lens_correction.auto_detect_camera")
    @patch("opencut.core.lens_profile.subprocess.run")
    @patch("opencut.core.lens_profile.os.path.isfile", return_value=True)
    @patch("opencut.core.lens_profile.get_video_info")
    def test_get_lens_info_basic(self, mock_info, mock_isfile, mock_sub, mock_detect):
        from opencut.core.lens_profile import get_lens_info
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "codec": "h264"}
        mock_sub.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({
                "format": {
                    "tags": {
                        "com.apple.quicktime.make": "Apple",
                        "com.apple.quicktime.model": "iPhone 15 Pro",
                    }
                },
                "streams": [{"tags": {}, "codec_name": "h264", "pix_fmt": "yuv420p"}],
            }),
        )
        mock_detect.return_value = MagicMock(profile=None, camera_model="", confidence=0.0)

        info = get_lens_info("/test/video.mp4")
        assert info.camera_make == "Apple"
        assert info.camera_model == "iPhone 15 Pro"
        assert info.resolution == "1920x1080"

    def test_auto_correct_lens_file_not_found(self):
        from opencut.core.lens_profile import auto_correct_lens
        with pytest.raises(FileNotFoundError):
            auto_correct_lens("/nonexistent/video.mp4")

    @patch("opencut.core.lens_correction.correct_lens_distortion")
    @patch("opencut.core.lens_correction.auto_detect_camera")
    @patch("opencut.core.lens_profile.get_lens_info")
    @patch("opencut.core.lens_profile.os.path.isfile", return_value=True)
    def test_auto_correct_with_detected_profile(self, mock_isfile, mock_lens_info, mock_detect, mock_correct):
        from opencut.core.lens_correction import CameraDetectionResult, CameraProfile
        from opencut.core.lens_profile import LensInfo, auto_correct_lens

        mock_lens_info.return_value = LensInfo(camera_model="GoPro HERO12")
        mock_detect.return_value = CameraDetectionResult(
            detected=True,
            camera_model="GoPro HERO12",
            profile=CameraProfile(camera_model="GoPro HERO12", k1=-0.3, k2=0.05),
            confidence=0.9,
            suggested_k1=-0.3,
            suggested_k2=0.05,
        )
        mock_correct.return_value = {"output_path": "/test/output.mp4"}

        result = auto_correct_lens("/test/gopro.mp4")
        assert len(result.corrections_applied) > 0
        assert result.k1 == -0.3


# ========================================================================
# Route Smoke Tests
# ========================================================================
class TestVRLensRoutes:
    """Smoke tests for the VR & lens routes."""

    @pytest.fixture
    def app(self):
        from flask import Flask
        from opencut.routes.vr_lens_routes import vr_lens_bp
        app = Flask(__name__)
        app.register_blueprint(vr_lens_bp)
        app.config["TESTING"] = True
        return app

    @pytest.fixture
    def client(self, app):
        return app.test_client()

    def _csrf_headers(self):
        from opencut.security import get_csrf_token
        return {"X-OpenCut-Token": get_csrf_token(), "Content-Type": "application/json"}

    def test_vr_stabilize_no_filepath(self, client):
        resp = client.post("/api/vr/stabilize",
                           json={},
                           headers=self._csrf_headers())
        assert resp.status_code == 400

    def test_vr_reframe_no_filepath(self, client):
        resp = client.post("/api/vr/reframe",
                           json={},
                           headers=self._csrf_headers())
        assert resp.status_code == 400

    def test_vr_extract_fov_no_filepath(self, client):
        resp = client.post("/api/vr/extract-fov",
                           json={},
                           headers=self._csrf_headers())
        assert resp.status_code == 400

    def test_vr_spatial_audio_no_filepath(self, client):
        resp = client.post("/api/vr/spatial-audio",
                           json={},
                           headers=self._csrf_headers())
        assert resp.status_code == 400

    def test_lens_auto_detect_no_filepath(self, client):
        resp = client.post("/api/lens/auto-detect",
                           json={},
                           headers=self._csrf_headers())
        assert resp.status_code == 400

    def test_lens_correct_distortion_no_filepath(self, client):
        resp = client.post("/api/lens/correct-distortion",
                           json={},
                           headers=self._csrf_headers())
        assert resp.status_code == 400

    def test_lens_chromatic_aberration_no_filepath(self, client):
        resp = client.post("/api/lens/chromatic-aberration",
                           json={},
                           headers=self._csrf_headers())
        assert resp.status_code == 400

    def test_lens_profiles_get(self, client):
        resp = client.get("/api/lens/profiles")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "profiles" in data
        assert "total" in data
        assert "categories" in data
        assert data["total"] > 0

    def test_lens_profiles_with_category(self, client):
        resp = client.get("/api/lens/profiles?category=drone")
        assert resp.status_code == 200
        data = resp.get_json()
        assert all(p["category"] == "drone" for p in data["profiles"])

    def test_lens_profiles_with_presets(self, client):
        resp = client.get("/api/lens/profiles?include_presets=true")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "presets" in data

    def test_lens_profiles_without_presets(self, client):
        resp = client.get("/api/lens/profiles?include_presets=false")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "presets" not in data

    def test_csrf_required_on_post(self, client):
        resp = client.post("/api/vr/stabilize", json={"filepath": "/test.mp4"})
        assert resp.status_code == 403

    def test_vr_stabilize_bad_filepath(self, client):
        resp = client.post("/api/vr/stabilize",
                           json={"filepath": "/nonexistent/video.mp4"},
                           headers=self._csrf_headers())
        assert resp.status_code == 400

    def test_lens_correct_bad_filepath(self, client):
        resp = client.post("/api/lens/correct-distortion",
                           json={"filepath": "/nonexistent/video.mp4"},
                           headers=self._csrf_headers())
        assert resp.status_code == 400
