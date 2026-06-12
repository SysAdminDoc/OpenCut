"""
OpenCut Gyroflow Stabilization v1.28.0 — STUB

Gyroscope-assisted video stabilization via Gyroflow binary.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional


@dataclass
class GyroflowResult:
    output: str = ""
    stabilization_method: str = "gyro"
    duration: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "stabilization_method", "duration", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_gyroflow_available() -> bool:
    return shutil.which("gyroflow") is not None


INSTALL_HINT = (
    "Download gyroflow binary from https://github.com/gyroflow/gyroflow/releases "
    "and add to PATH"
)


def list_lens_profiles() -> List[str]:
    return []


def stabilize(
    video_path: str,
    gyro_data_path: Optional[str] = None,
    lens_profile: Optional[str] = None,
    horizon_lock: bool = False,
    output: Optional[str] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> GyroflowResult:
    if not check_gyroflow_available():
        raise RuntimeError(f"Gyroflow is not installed. Install with:\n    {INSTALL_HINT}")
    raise NotImplementedError("Gyroflow wiring is not implemented yet. Track the live ROADMAP.md entry.")


__all__ = ["GyroflowResult", "check_gyroflow_available", "INSTALL_HINT",
           "list_lens_profiles", "stabilize"]
