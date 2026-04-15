"""
OpenCut Configuration

Centralized config dataclass. All environment variable reads happen here
so the rest of the codebase accesses config values, not raw env vars.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _env_int(name: str, default: int, *, min_val: int | None = None) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    if min_val is not None and value < min_val:
        return min_val
    return value


def _env_csv(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if raw is None:
        return list(default)
    items = [item.strip() for item in raw.split(",")]
    cleaned = [item for item in items if item]
    return cleaned or list(default)


@dataclass
class OpenCutConfig:
    """Application configuration, populated from environment or overrides."""

    bundled_mode: bool = False
    whisper_models_dir: Optional[str] = None
    torch_home: Optional[str] = None
    florence_model_dir: Optional[str] = None
    lama_model_dir: Optional[str] = None
    max_content_length: int = 100 * 1024 * 1024  # 100 MB
    cors_origins: list[str] = field(default_factory=lambda: ["null", "file://"])

    # Job system defaults (single source of truth; mirrored as module-level
    # constants in opencut/jobs.py for use outside Flask app context)
    job_max_age: int = 3600              # Auto-clean jobs older than 1 hour
    max_concurrent_jobs: int = 10        # Prevent job spam / GPU OOM
    max_batch_files: int = 100           # Max files per batch request
    job_stuck_timeout: int = 7200        # Mark running jobs as error after 2 hours

    @classmethod
    def from_env(cls) -> "OpenCutConfig":
        """Create config from environment variables (single source of truth)."""
        whisper_dir = os.environ.get("WHISPER_MODELS_DIR", None)
        bundled = (
            _env_bool("OPENCUT_BUNDLED", default=False)
            or whisper_dir is not None
        )
        return cls(
            bundled_mode=bundled,
            whisper_models_dir=whisper_dir,
            torch_home=os.environ.get("TORCH_HOME", None),
            florence_model_dir=os.environ.get("OPENCUT_FLORENCE_DIR", None),
            lama_model_dir=os.environ.get("OPENCUT_LAMA_DIR", None),
            max_content_length=_env_int(
                "OPENCUT_MAX_CONTENT_LENGTH",
                100 * 1024 * 1024,
                min_val=1024 * 1024,
            ),
            cors_origins=_env_csv("OPENCUT_CORS_ORIGINS", ["null", "file://"]),
            job_max_age=_env_int("OPENCUT_JOB_MAX_AGE", 3600, min_val=60),
            max_concurrent_jobs=_env_int("OPENCUT_MAX_CONCURRENT_JOBS", 10, min_val=1),
            max_batch_files=_env_int("OPENCUT_MAX_BATCH_FILES", 100, min_val=1),
            job_stuck_timeout=_env_int("OPENCUT_JOB_STUCK_TIMEOUT", 7200, min_val=60),
        )
