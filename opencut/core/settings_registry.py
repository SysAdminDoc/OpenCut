"""What every persisted setting is for, and what honours it.

OpenCut stores a dozen settings files under ``~/.opencut``. A survey keyed on
the stored file rather than the accessor found that seven of them reached no
consumer at all: the panel could save a chapter or multicam default, the
settings API would round-trip it happily, and the route that ran the job
ignored it in favour of a literal. Two of them (``color_profiles.json``,
``auto_zoom_presets.json``) had no reader on either side, backend or panel.

The registry below states, for each stored file, which of three things it is:

``DEFAULT``
    A value the backend must apply when a request omits the field. It names the
    module that has to reference it, and ``tests/test_settings_consumers.py``
    fails if that reference disappears.
``COLLECTION``
    User-managed data the panel reads through the settings API and passes back
    explicitly. The panel is the consumer; the backend is storage.
``REMOVED``
    Kept only so :func:`migrate_removed_settings` can clean it off disk.

Anything stored by ``user_data`` and absent from this table fails the same
test, so a new setting cannot be added without saying which kind it is.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger("opencut")

SettingKind = Literal["default", "collection", "removed"]


@dataclass(frozen=True)
class PersistedSetting:
    """One file under ``~/.opencut`` and the reason it exists."""

    filename: str
    loader: str
    kind: SettingKind
    #: For ``default``: the module that must apply it, as a package-relative
    #: path. For the others, empty.
    consumer: str = ""
    note: str = ""


PERSISTED_SETTINGS: tuple[PersistedSetting, ...] = (
    PersistedSetting(
        "whisper_settings.json", "load_whisper_settings", "default",
        consumer="core/captions.py",
        note="Transcription model and language reach the caption engine.",
    ),
    PersistedSetting(
        "local_only.json", "load_local_only_setting", "default",
        consumer="config.py",
        note="Privacy mode. config.is_local_only() reads the file directly.",
    ),
    PersistedSetting(
        "workflows.json", "load_workflows", "default",
        consumer="routes/workflow.py",
        note="Saved workflow definitions, executed by the workflow route.",
    ),
    PersistedSetting(
        "assistant_dismissed.json", "load_assistant_dismissed", "default",
        consumer="routes/system_workspace_routes.py",
        note="Which assistant prompts the operator has dismissed.",
    ),
    PersistedSetting(
        "chapter_defaults.json", "load_chapter_defaults", "default",
        consumer="routes/caption_analysis_routes.py",
        note="Applied when a chapter request omits max_chapters or naming style.",
    ),
    PersistedSetting(
        "multicam_config.json", "load_multicam_config", "default",
        consumer="routes/video_editing.py",
        note="Applied when a multicam request omits min_cut_duration.",
    ),
    PersistedSetting(
        "footage_index_config.json", "load_footage_index_config", "default",
        consumer="routes/search.py",
        note="Index location and transcription model for footage indexing.",
    ),
    PersistedSetting(
        "loudness_settings.json", "load_loudness_target", "default",
        consumer="routes/audio.py",
        note="Applied when a normalise request omits the target LUFS.",
    ),
    PersistedSetting(
        "user_presets.json", "load_presets", "collection",
        note="Export presets the panel lists and submits explicitly.",
    ),
    PersistedSetting(
        "favorites.json", "load_favorites", "collection",
        note="Panel favourites; the backend only stores them.",
    ),
    PersistedSetting(
        "color_profiles.json", "load_color_profiles", "removed",
        note=(
            "Colour-match reference profiles. Never read by any backend module "
            "or either panel since it was added; there is no colour-match "
            "surface that consumes a saved profile."
        ),
    ),
    PersistedSetting(
        "auto_zoom_presets.json", "load_auto_zoom_presets", "removed",
        note=(
            "Auto-zoom presets. Stored and served, but no backend module and "
            "neither panel ever read them back."
        ),
    ),
)

BY_FILENAME: dict[str, PersistedSetting] = {entry.filename: entry for entry in PERSISTED_SETTINGS}

#: Files whose only remaining purpose is to be cleaned up.
REMOVED_SETTING_FILES: tuple[str, ...] = tuple(
    entry.filename for entry in PERSISTED_SETTINGS if entry.kind == "removed"
)


def settings_requiring_a_consumer() -> tuple[PersistedSetting, ...]:
    return tuple(entry for entry in PERSISTED_SETTINGS if entry.kind == "default")


def migrate_removed_settings(directory: str | os.PathLike | None = None) -> list[str]:
    """Delete settings files that no longer have a purpose.

    Returns the names actually removed. Safe to call repeatedly, and safe when
    the directory does not exist: a user who never saved one of these has
    nothing to clean up.
    """
    from opencut.helpers import OPENCUT_DIR

    base = str(directory) if directory is not None else OPENCUT_DIR
    removed: list[str] = []
    for name in REMOVED_SETTING_FILES:
        path = os.path.join(base, name)
        if not os.path.isfile(path):
            continue
        try:
            os.remove(path)
        except OSError as exc:  # pragma: no cover - permissions
            logger.warning("Could not remove obsolete setting %s: %s", name, exc)
            continue
        removed.append(name)
        logger.info("Removed obsolete setting file %s (nothing read it)", name)
    return removed
