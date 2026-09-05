"""A persisted setting nobody reads is a lie told to the operator.

A survey keyed on the stored file rather than the accessor found seven settings
under ``~/.opencut`` that reached no consumer at all. The panel could save a
chapter or multicam default, the settings API round-tripped it, and the route
that ran the job used a literal instead. Two of them, ``color_profiles.json``
and ``auto_zoom_presets.json``, had no reader on either side.

These tests keep the registry honest in both directions: every file
``user_data`` stores has to be classified, and every classified default has to
have a consumer that still references it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from opencut.core.settings_registry import (
    BY_FILENAME,
    PERSISTED_SETTINGS,
    REMOVED_SETTING_FILES,
    migrate_removed_settings,
    settings_requiring_a_consumer,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = REPO_ROOT / "opencut"
USER_DATA = PACKAGE / "user_data.py"


def _stored_filenames() -> set[str]:
    """Every JSON file ``user_data`` reads or writes."""
    text = USER_DATA.read_text(encoding="utf-8")
    return set(re.findall(r'(?:read_user_file|write_user_file)\(\s*"([^"]+\.json)"', text))


# ---------------------------------------------------------------------------
# The registry covers what is actually stored
# ---------------------------------------------------------------------------

def test_every_stored_setting_is_classified():
    """A new setting cannot be added without saying what reads it."""
    stored = _stored_filenames()
    unclassified = sorted(stored - set(BY_FILENAME))
    assert not unclassified, (
        "these settings are written to disk but not in PERSISTED_SETTINGS, so "
        "nothing checks whether anything reads them: " + ", ".join(unclassified)
    )


def test_the_registry_does_not_describe_settings_that_no_longer_exist():
    stored = _stored_filenames()
    # A removed setting is deliberately absent from user_data once its loader
    # goes; until then it must still be stored.
    stale = sorted(
        entry.filename
        for entry in PERSISTED_SETTINGS
        if entry.filename not in stored and entry.kind != "removed"
    )
    assert not stale, f"registry describes settings user_data no longer stores: {stale}"


def test_registry_entries_are_well_formed():
    for entry in PERSISTED_SETTINGS:
        assert entry.filename.endswith(".json")
        assert entry.note, f"{entry.filename} needs a note saying why it exists"
        if entry.kind == "default":
            assert entry.consumer, f"{entry.filename} is a default with no named consumer"
        else:
            assert not entry.consumer, f"{entry.filename} is {entry.kind}; it should name no consumer"


# ---------------------------------------------------------------------------
# Every declared default is actually applied somewhere
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "entry", settings_requiring_a_consumer(), ids=lambda entry: entry.filename
)
def test_each_default_is_read_by_its_declared_consumer(entry):
    """The regression: stored, served, and then ignored by the job."""
    consumer = PACKAGE / entry.consumer
    assert consumer.is_file(), f"{entry.filename} names a consumer that does not exist: {entry.consumer}"
    source = consumer.read_text(encoding="utf-8", errors="replace")
    assert entry.loader in source or entry.filename in source, (
        f"{entry.consumer} no longer reads {entry.filename}; the setting is "
        "stored and served but nothing applies it"
    )


def test_the_chapter_route_passes_the_saved_minimum_duration_through():
    """Reading the default is not enough; it has to reach the generator."""
    source = (PACKAGE / "routes" / "caption_analysis_routes.py").read_text(encoding="utf-8")
    call = source.split("generate_chapters(", 1)[1][:400]
    assert "min_chapter_duration=" in call, (
        "the saved minimum chapter duration is read but never handed to the generator"
    )


def test_no_route_still_hardcodes_a_default_it_now_reads():
    """Guard the specific literals these defaults replaced."""
    multicam = (PACKAGE / "routes" / "video_editing.py").read_text(encoding="utf-8")
    assert 'data.get("min_cut_duration", 1.0)' not in multicam

    chapters = (PACKAGE / "routes" / "caption_analysis_routes.py").read_text(encoding="utf-8")
    assert 'data.get("max_chapters", 15)' not in chapters

    search = (PACKAGE / "routes" / "search.py").read_text(encoding="utf-8")
    assert 'data.get("model", "base")' not in search


# ---------------------------------------------------------------------------
# Removal migration
# ---------------------------------------------------------------------------

def test_removed_settings_are_deleted_from_disk(tmp_path):
    for name in REMOVED_SETTING_FILES:
        (tmp_path / name).write_text("[]", encoding="utf-8")

    removed = migrate_removed_settings(tmp_path)

    assert sorted(removed) == sorted(REMOVED_SETTING_FILES)
    for name in REMOVED_SETTING_FILES:
        assert not (tmp_path / name).exists()


def test_the_migration_is_safe_to_run_twice(tmp_path):
    (tmp_path / REMOVED_SETTING_FILES[0]).write_text("[]", encoding="utf-8")
    assert migrate_removed_settings(tmp_path) == [REMOVED_SETTING_FILES[0]]
    assert migrate_removed_settings(tmp_path) == []


def test_the_migration_leaves_live_settings_alone(tmp_path):
    keep = tmp_path / "whisper_settings.json"
    keep.write_text(json.dumps({"model": "small"}), encoding="utf-8")
    for name in REMOVED_SETTING_FILES:
        (tmp_path / name).write_text("[]", encoding="utf-8")

    migrate_removed_settings(tmp_path)

    assert keep.exists(), "the migration deleted a setting that is still in use"
    assert json.loads(keep.read_text(encoding="utf-8"))["model"] == "small"


def test_the_migration_tolerates_a_missing_directory(tmp_path):
    assert migrate_removed_settings(tmp_path / "nope") == []


def test_a_removed_setting_names_why_it_went():
    for entry in PERSISTED_SETTINGS:
        if entry.kind == "removed":
            assert len(entry.note) > 40, (
                f"{entry.filename} was removed without recording why; the next "
                "person will just add it back"
            )
