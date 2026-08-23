"""One face detector for every call site, across OpenCV 4 and OpenCV 5.

OpenCV 5 moved the Haar and HOG detectors out of the core module into
``opencv_contrib`` (see the OpenCV 5 release notes: "Haar-based and HOG-based
detectors have been moved to opencv_contrib"). ``pyproject.toml`` declares
``opencv-python==4.14.0.94``. OpenCV 5 also remains supported for source
installs, but neither standard wheel ships contrib, so every direct
``cv2.CascadeClassifier(...)`` call raises ``AttributeError`` for anyone who
installs the project as declared. That silently broke face-tracked auto-zoom,
face blur, reframe, redaction, and thumbnail selection.

This module resolves a detector once, preferring the modern DNN detector and
falling back to Haar where it still exists:

1. **YuNet** (``cv2.FaceDetectorYN``) when an ONNX model is resolvable. Present
   in OpenCV 4.5.4+ and 5.x, so this is the forward path.
2. **Haar** (``cv2.CascadeClassifier``) when the symbol and cascade data are
   available — OpenCV 4.x, or 5.x with ``opencv-contrib-python`` installed.
3. A null detector that reports no faces, so callers keep their existing
   centre-crop / no-face fallback instead of crashing.

The returned object exposes ``detectMultiScale(gray, ...)`` with the same
signature and ``(x, y, w, h)`` return shape the call sites already use, so
migrating a call site is a one-line change.

Set ``OPENCUT_YUNET_ONNX`` to a YuNet model path to force the DNN backend.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Optional

logger = logging.getLogger("opencut")

#: Environment override for the YuNet ONNX model.
YUNET_MODEL_ENV = "OPENCUT_YUNET_ONNX"

#: Default cascade every legacy call site used.
DEFAULT_CASCADE = "haarcascade_frontalface_default.xml"

INSTALL_HINT = (
    "Install a face-detection backend: either set OPENCUT_YUNET_ONNX to a YuNet "
    "ONNX model, or install opencv-contrib-python for the legacy Haar cascades."
)

_lock = threading.Lock()
_cached: dict[str, Any] = {}


def _cv2():
    try:
        import cv2  # noqa: PLC0415 - optional dependency, resolved lazily
    except ImportError:
        return None
    return cv2


def _yunet_model_path(cv2mod) -> Optional[str]:
    """Locate a YuNet ONNX model, or return ``None``."""
    configured = (os.environ.get(YUNET_MODEL_ENV) or "").strip()
    if configured and os.path.isfile(configured):
        return configured

    candidates = [
        os.path.join(os.path.expanduser("~"), ".opencut", "models", "face_detection_yunet.onnx"),
        os.path.join(os.path.dirname(__file__), "..", "data", "face_detection_yunet.onnx"),
    ]
    data_dir = getattr(cv2mod, "data", None)
    if data_dir is not None:
        base = getattr(data_dir, "haarcascades", "")
        if base:
            candidates.append(os.path.join(os.path.dirname(base.rstrip("/\\")), "face_detection_yunet.onnx"))
    for candidate in candidates:
        resolved = os.path.abspath(candidate)
        if os.path.isfile(resolved):
            return resolved
    return None


def cascade_scale_image_flag() -> int:
    """``cv2.CASCADE_SCALE_IMAGE`` for call sites, with a literal fallback.

    OpenCV 5 dropped the ``CASCADE_*`` flag constants along with the cascade
    API, so reading the attribute raises AttributeError on the declared wheel
    exactly like ``cv2.CascadeClassifier`` did. The value is a stable public
    constant, so falling back to it keeps the Haar path bit-identical on
    opencv 4 rather than quietly switching the detector to ``flags=0``.
    """
    cv2mod = _cv2()
    if cv2mod is None:
        return 2
    return int(getattr(cv2mod, "CASCADE_SCALE_IMAGE", 2))


def _haar_cascade_path(cv2mod, cascade_name: str) -> Optional[str]:
    data_dir = getattr(cv2mod, "data", None)
    if data_dir is not None:
        base = getattr(data_dir, "haarcascades", "")
        if base:
            builtin = os.path.join(base, cascade_name)
            if os.path.isfile(builtin):
                return builtin
    for candidate in (
        os.path.join(os.path.dirname(__file__), cascade_name),
        os.path.join(os.path.expanduser("~"), ".opencut", cascade_name),
    ):
        if os.path.isfile(candidate):
            return candidate
    return None


class _YuNetDetector:
    """YuNet wrapped in the ``detectMultiScale`` shape the call sites expect."""

    backend = "yunet"

    def __init__(self, cv2mod, model_path: str):
        self._cv2 = cv2mod
        self._model_path = model_path
        self._detector = cv2mod.FaceDetectorYN.create(model_path, "", (320, 320))
        self._detector_lock = threading.Lock()

    def detectMultiScale(  # noqa: N802 - mirrors the cv2 API the call sites use
        self,
        image,
        scaleFactor: float = 1.1,  # noqa: N803 - cv2 signature compatibility
        minNeighbors: int = 5,  # noqa: N803
        minSize: Optional[tuple] = None,  # noqa: N803
        **_kwargs,
    ):
        cv2mod = self._cv2
        frame = image
        # YuNet needs 3-channel BGR; the call sites hand us grayscale.
        if getattr(frame, "ndim", 3) == 2:
            frame = cv2mod.cvtColor(frame, cv2mod.COLOR_GRAY2BGR)
        height, width = frame.shape[:2]
        if not height or not width:
            return []

        with self._detector_lock:
            self._detector.setInputSize((width, height))
            _retval, faces = self._detector.detect(frame)

        if faces is None:
            return []
        min_w, min_h = (minSize or (0, 0))
        boxes = []
        for face in faces:
            x, y, w, h = (int(round(float(v))) for v in face[:4])
            if w <= 0 or h <= 0 or w < min_w or h < min_h:
                continue
            boxes.append((max(0, x), max(0, y), w, h))
        return boxes


class _HaarDetector:
    """Thin pass-through so the Haar path keeps its exact legacy behaviour."""

    backend = "haar"

    def __init__(self, cascade):
        self._cascade = cascade

    def detectMultiScale(self, image, *args, **kwargs):  # noqa: N802 - cv2 API
        return self._cascade.detectMultiScale(image, *args, **kwargs)

    def empty(self) -> bool:
        return bool(self._cascade.empty())


class _NullDetector:
    """Reports no faces so callers use their existing no-face fallback."""

    backend = "unavailable"

    def detectMultiScale(self, image, *args, **kwargs):  # noqa: N802 - cv2 API
        return []

    def empty(self) -> bool:
        return True


def create_face_detector(cascade_name: str = DEFAULT_CASCADE):
    """Return a detector exposing ``detectMultiScale``.

    Never raises: an environment with no usable backend gets a null detector,
    matching the pre-existing "empty cascade -> centre crop" behaviour.
    """
    cv2mod = _cv2()
    if cv2mod is None:
        logger.warning("face detection unavailable: cv2 is not installed. %s", INSTALL_HINT)
        return _NullDetector()

    model_path = _yunet_model_path(cv2mod)
    if model_path and hasattr(cv2mod, "FaceDetectorYN"):
        try:
            return _YuNetDetector(cv2mod, model_path)
        except Exception as exc:  # noqa: BLE001 - fall through to Haar
            logger.warning("YuNet model %s failed to load (%s); trying Haar.", model_path, exc)

    if hasattr(cv2mod, "CascadeClassifier"):
        cascade_path = _haar_cascade_path(cv2mod, cascade_name)
        if cascade_path:
            cascade = cv2mod.CascadeClassifier(cascade_path)
            if not cascade.empty():
                return _HaarDetector(cascade)
            logger.warning("Haar cascade at %s loaded empty.", cascade_path)
        else:
            logger.warning("Haar cascade %s not found on disk.", cascade_name)

    logger.warning(
        "No face-detection backend is available; detection will report no faces. %s",
        INSTALL_HINT,
    )
    return _NullDetector()


def get_shared_face_detector(cascade_name: str = DEFAULT_CASCADE):
    """Process-wide cached detector, built under a lock (Flask is threaded)."""
    cached = _cached.get(cascade_name)
    if cached is not None:
        return cached
    with _lock:
        cached = _cached.get(cascade_name)
        if cached is None:
            cached = create_face_detector(cascade_name)
            _cached[cascade_name] = cached
    return cached


def clear_detector_cache() -> None:
    """Drop cached detectors. Used by tests and after installing a backend."""
    with _lock:
        _cached.clear()


def active_backend(cascade_name: str = DEFAULT_CASCADE) -> str:
    """Return ``yunet``/``haar``/``unavailable`` for diagnostics."""
    return getattr(get_shared_face_detector(cascade_name), "backend", "unavailable")


def check_face_detection_available() -> bool:
    """True when a real detector (not the null fallback) is resolvable."""
    return active_backend() != "unavailable"
