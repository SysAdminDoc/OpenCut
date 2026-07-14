"""
OpenCut - Open Source Video Editing Automation for Premiere Pro

Automatically remove silences, generate captions, switch podcast cameras,
and more. Exports Premiere Pro / DaVinci Resolve / FCP XML.
"""

__version__ = "1.33.1"
__author__ = "OpenCut Contributors"
__license__ = "MIT"

# Install before importing feature modules so local-only mode cannot be bypassed
# by a direct HTTP client, third-party SDK, or network-capable subprocess.
from opencut.network_policy import install_egress_guard as _install_egress_guard

_install_egress_guard()
