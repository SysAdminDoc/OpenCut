"""
OpenCut Configuration

Centralized config dataclass. All environment variable reads happen here
so the rest of the codebase accesses config values, not raw env vars.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OpenCutConfig:
    """Application configuration, populated from environment or overrides."""

    bundled_mode: bool = False
    whisper_models_dir: Optional[str] = None
    torch_home: Optional[str] = None
    florence_model_dir: Optional[str] = None
    lama_model_dir: Optional[str] = None
    max_content_length: int = 100 * 1024 * 1024  # 100 MB
    cors_origins: list = field(default_factory=lambda: ["null", "file://"])

    @classmethod
    def from_env(cls) -> "OpenCutConfig":
        """Create config from environment variables (single source of truth)."""
        whisper_dir = os.environ.get("WHISPER_MODELS_DIR", None)
        bundled = (
            os.environ.get("OPENCUT_BUNDLED", "").lower() == "true"
            or whisper_dir is not None
        )
        return cls(
            bundled_mode=bundled,
            whisper_models_dir=whisper_dir,
            torch_home=os.environ.get("TORCH_HOME", None),
            florence_model_dir=os.environ.get("OPENCUT_FLORENCE_DIR", None),
            lama_model_dir=os.environ.get("OPENCUT_LAMA_DIR", None),
        )
