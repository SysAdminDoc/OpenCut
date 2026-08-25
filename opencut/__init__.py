"""
OpenCut - Open Source Video Editing Automation for Premiere Pro

Automatically remove silences, generate captions, switch podcast cameras,
and more. Exports Premiere Pro / DaVinci Resolve / FCP XML.
"""

import os
import sys

MIN_PYTHON = (3, 11)
MAX_PYTHON = (3, 14)


def _require_supported_python(version_info=None):
    """Reject unsupported source launches before importing feature modules."""
    version_info = sys.version_info if version_info is None else version_info
    detected = tuple(version_info[:3])
    if MIN_PYTHON <= detected[:2] <= MAX_PYTHON:
        return
    detected_text = ".".join(str(part) for part in detected)
    required_text = f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}-{MAX_PYTHON[0]}.{MAX_PYTHON[1]}"
    raise RuntimeError(
        f"OpenCut requires Python {required_text}; detected Python "
        f"{detected_text}. Install a supported Python from "
        "https://www.python.org/downloads/ and retry."
    )


_require_supported_python()

__version__ = "1.55.1"
__author__ = "OpenCut Contributors"
__license__ = "MIT"

# The Windows wheel's FFmpeg plugin must be disabled before importing
# ``opencut.core`` because that package eagerly imports modules which load cv2.
# Setting this after cv2's video registry initializes does not disable FFmpeg.
if sys.platform == "win32":
    os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "0"

# Lock down embedded media backends before any feature module can import cv2
# or PyAV. Windows artifacts use Media Foundation and omit OpenCV's unverified
# FFmpeg plugin; other platforms must prove the reviewed wheel/ABI at release.
from opencut.core.embedded_media_provenance import (  # noqa: E402
    install_runtime_guard as _install_runtime_decoder_guard,
)

_install_runtime_decoder_guard()

# Install before importing feature modules so local-only mode cannot be bypassed
# by a direct HTTP client, third-party SDK, or network-capable subprocess.
from opencut.network_policy import install_egress_guard as _install_egress_guard  # noqa: E402

_install_egress_guard()
