"""F305 — face detection must survive the declared opencv-python 5 wheel.

OpenCV 5 moved the Haar and HOG detectors into ``opencv_contrib``, which the
declared ``opencv-python>=5,<6`` dependency does not ship. Thirteen modules
called ``cv2.CascadeClassifier`` directly, so a manifest-faithful install
raised ``AttributeError`` in auto-zoom, face blur, reframe, redaction, and
thumbnail selection. These tests pin the compatibility contract, including a
simulated OpenCV 5 environment.
"""

from __future__ import annotations

import os
import pathlib
import re
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opencut.core import face_detect_compat as fc  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CORE_DIR = REPO_ROOT / "opencut" / "core"


@pytest.fixture(autouse=True)
def _clear_cache():
    fc.clear_detector_cache()
    yield
    fc.clear_detector_cache()


class TestNoDirectCascadeUse:
    def test_no_module_constructs_a_cascade_classifier_directly(self):
        """The regression that broke every OpenCV 5 install must not return."""
        offenders = []
        for path in CORE_DIR.rglob("*.py"):
            if path.name == "face_detect_compat.py":
                continue  # documents the symbol in prose and probes it safely
            text = path.read_text(encoding="utf-8", errors="replace")
            # Alias-agnostic on purpose: `import cv2 as _c` then
            # `_c.CascadeClassifier(...)` is the same defect and a
            # `cv2.`-anchored pattern would wave it through.
            if re.search(r"\.CascadeClassifier\s*\(", text):
                offenders.append(path.relative_to(REPO_ROOT).as_posix())
        assert offenders == [], (
            "these modules still construct cv2.CascadeClassifier directly and "
            f"will raise AttributeError on opencv-python 5: {offenders}"
        )

    def test_compat_module_never_calls_the_symbol_unguarded(self):
        text = (CORE_DIR / "face_detect_compat.py").read_text(encoding="utf-8")
        assert 'hasattr(cv2mod, "CascadeClassifier")' in text


class TestDetectorResolution:
    def test_returns_a_detector_with_the_cv2_interface(self):
        detector = fc.create_face_detector()
        assert hasattr(detector, "detectMultiScale")
        assert detector.backend in {"yunet", "haar", "unavailable"}

    def test_detects_nothing_on_a_blank_frame_without_raising(self):
        detector = fc.create_face_detector()
        gray = np.zeros((240, 320), dtype=np.uint8)
        assert list(detector.detectMultiScale(gray, 1.1, 5)) == []

    def test_shared_detector_is_cached(self):
        assert fc.get_shared_face_detector() is fc.get_shared_face_detector()

    def test_clear_cache_rebuilds(self):
        first = fc.get_shared_face_detector()
        fc.clear_detector_cache()
        assert fc.get_shared_face_detector() is not first


class TestOpenCv5Simulation:
    """The scenario the declared dependency actually produces."""

    def test_missing_cascade_classifier_degrades_instead_of_raising(self):
        import cv2

        saved = cv2.CascadeClassifier
        try:
            del cv2.CascadeClassifier
            fc.clear_detector_cache()
            detector = fc.create_face_detector()
            gray = np.zeros((120, 160), dtype=np.uint8)
            # The old code raised AttributeError here.
            assert list(detector.detectMultiScale(gray, 1.1, 5)) == []
            assert detector.empty() is True
        finally:
            cv2.CascadeClassifier = saved
            fc.clear_detector_cache()

    def test_every_migrated_module_imports_without_cascade_classifier(self):
        import importlib

        import cv2

        modules = [
            "ai_reframe_multi", "auto_zoom", "deepfake_detect", "face_tagging",
            "face_tools", "morph_cut", "multimodal_diarize", "redaction",
            "screenshot_video", "skin_retouch", "smart_reframe", "talking_head",
            "thumbnail",
        ]
        saved = cv2.CascadeClassifier
        try:
            del cv2.CascadeClassifier
            for name in modules:
                module = importlib.import_module(f"opencut.core.{name}")
                importlib.reload(module)
        finally:
            cv2.CascadeClassifier = saved

    def test_missing_cv2_entirely_is_survivable(self):
        with patch.object(fc, "_cv2", lambda: None):
            detector = fc.create_face_detector()
            assert detector.backend == "unavailable"
            assert detector.detectMultiScale(None) == []


class TestYuNetBackend:
    def test_yunet_is_preferred_when_a_model_is_configured(self, tmp_path, monkeypatch):
        model = tmp_path / "face_detection_yunet.onnx"
        model.write_bytes(b"not-a-real-model")
        monkeypatch.setenv(fc.YUNET_MODEL_ENV, str(model))

        fake_detector = MagicMock()
        fake_detector.detect.return_value = (1, np.array([[10.0, 20.0, 30.0, 40.0]]))
        fake_cv2 = MagicMock()
        fake_cv2.FaceDetectorYN.create.return_value = fake_detector

        with patch.object(fc, "_cv2", lambda: fake_cv2):
            fc.clear_detector_cache()
            detector = fc.create_face_detector()
            assert detector.backend == "yunet"
            boxes = detector.detectMultiScale(np.zeros((60, 80, 3), dtype=np.uint8))
        assert boxes == [(10, 20, 30, 40)]

    def test_yunet_load_failure_falls_back_to_haar(self, tmp_path, monkeypatch):
        model = tmp_path / "face_detection_yunet.onnx"
        model.write_bytes(b"corrupt")
        monkeypatch.setenv(fc.YUNET_MODEL_ENV, str(model))
        fc.clear_detector_cache()
        # The real cv2 will reject the corrupt model; Haar must take over.
        detector = fc.create_face_detector()
        assert detector.backend in {"haar", "unavailable"}

    def test_yunet_filters_boxes_below_min_size(self, tmp_path, monkeypatch):
        model = tmp_path / "face_detection_yunet.onnx"
        model.write_bytes(b"x")
        monkeypatch.setenv(fc.YUNET_MODEL_ENV, str(model))

        fake_detector = MagicMock()
        fake_detector.detect.return_value = (
            1,
            np.array([[0.0, 0.0, 5.0, 5.0], [10.0, 10.0, 50.0, 50.0]]),
        )
        fake_cv2 = MagicMock()
        fake_cv2.FaceDetectorYN.create.return_value = fake_detector

        with patch.object(fc, "_cv2", lambda: fake_cv2):
            fc.clear_detector_cache()
            detector = fc.create_face_detector()
            boxes = detector.detectMultiScale(
                np.zeros((60, 80, 3), dtype=np.uint8), minSize=(30, 30)
            )
        assert boxes == [(10, 10, 50, 50)]
