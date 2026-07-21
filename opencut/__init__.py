"""
OpenCut - Open Source Video Editing Automation for Premiere Pro

Automatically remove silences, generate captions, switch podcast cameras,
and more. Exports Premiere Pro / DaVinci Resolve / FCP XML.
"""

import sys

MIN_PYTHON = (3, 11)


def _require_supported_python(version_info=None):
    """Reject unsupported source launches before importing feature modules."""
    version_info = sys.version_info if version_info is None else version_info
    detected = tuple(version_info[:3])
    if detected[:2] >= MIN_PYTHON:
        return
    detected_text = ".".join(str(part) for part in detected)
    required_text = ".".join(str(part) for part in MIN_PYTHON)
    raise RuntimeError(
        f"OpenCut requires Python {required_text} or newer; detected Python "
        f"{detected_text}. Install a supported Python from "
        "https://www.python.org/downloads/ and retry."
    )


_require_supported_python()

__version__ = "1.35.0"
__author__ = "OpenCut Contributors"
__license__ = "MIT"

# Install before importing feature modules so local-only mode cannot be bypassed
# by a direct HTTP client, third-party SDK, or network-capable subprocess.
from opencut.network_policy import install_egress_guard as _install_egress_guard  # noqa: E402

_install_egress_guard()
