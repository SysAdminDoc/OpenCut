"""Upgrade conformance: a v1.25.1 user-data tree must survive the current build.

v1.25.1 is the newest public installer; source is many schema versions past
it. Every store here is exercised against the *real* v1.25.1 on-disk shape
(see :mod:`tests.legacy_user_data_fixture`) rather than a synthesized "old"
database, so a migration that only works on data this build wrote fails here.

The upgrade must also be offline: opening a local store may not reach the
network, so the whole migration phase runs with sockets disabled.
"""
from __future__ import annotations

import json
import socket
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest

from tests.legacy_user_data_fixture import (
    LEGACY_API_KEY,
    LEGACY_FOOTAGE_PATH,
    LEGACY_JOB_IDS,
    LEGACY_JOURNAL_LABELS,
    LEGACY_PLUGIN_NAME,
    LEGACY_REVIEW_ID,
    LEGACY_VERSION,
    build_v1_25_1_user_data,
    snapshot_tree,
)


def _user_version(path: Path) -> int:
    conn = sqlite3.connect(path)
    try:
        return int(conn.execute("PRAGMA user_version").fetchone()[0])
    finally:
        conn.close()


def _columns(path: Path, table: str) -> set[str]:
    conn = sqlite3.connect(path)
    try:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    finally:
        conn.close()


@contextmanager
def _no_network():
    """Fail closed on any socket use inside the ``with`` block.

    Owns a private MonkeyPatch so undoing the socket patch cannot also undo
    the store repointing done by the ``legacy_user_data`` fixture.
    """
    def _blocked(*_args, **_kwargs):
        raise AssertionError("upgrade must not touch the network")

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(socket, "socket", _blocked)
        patcher.setattr(socket, "create_connection", _blocked)
        yield


@pytest.fixture
def legacy_root(tmp_path) -> Path:
    return build_v1_25_1_user_data(tmp_path / "dot-opencut")


@pytest.fixture
def legacy_user_data(legacy_root, monkeypatch):
    """Point every user-data store at the legacy tree and reset module state."""
    import opencut.core.footage_index_db as footage_index
    import opencut.core.plugin_installation as plugin_installation
    import opencut.core.review_links as review_links
    import opencut.job_store as job_store
    import opencut.user_data as user_data
    from opencut import journal as journal_module

    monkeypatch.setattr(user_data, "OPENCUT_DIR", str(legacy_root))

    monkeypatch.setattr(job_store, "_DB_PATH", str(legacy_root / "jobs.db"))
    job_store.close_all_connections()
    job_store._INITIALIZED = False
    job_store._INITIALIZED_PATH = None
    job_store._LOCAL = threading.local()
    job_store._ALL_CONNECTIONS = {}

    monkeypatch.setattr(journal_module, "_DB_PATH", str(legacy_root / "journal.db"))
    journal_module.close_all_connections()
    journal_module._thread_local = threading.local()

    monkeypatch.setattr(footage_index, "_DB_PATH", str(legacy_root / "footage_index.db"))
    footage_index.close_all_connections()
    footage_index._thread_local = threading.local()

    monkeypatch.setattr(review_links, "REVIEWS_DIR", str(legacy_root / "reviews"))
    monkeypatch.setattr(
        plugin_installation, "TRUST_STORE_PATH",
        str(legacy_root / "trusted-plugin-publishers.json"),
    )

    yield legacy_root

    job_store.close_all_connections()
    journal_module.close_all_connections()
    footage_index.close_all_connections()
    job_store._INITIALIZED = False
    job_store._INITIALIZED_PATH = None


# ---------------------------------------------------------------------------
# SQLite stores
# ---------------------------------------------------------------------------
def test_job_history_survives_the_upgrade(legacy_user_data):
    import opencut.job_store as job_store

    with _no_network():
        job_store.init_db()

    db_path = legacy_user_data / "jobs.db"
    assert _user_version(db_path) == job_store.SCHEMA_VERSION
    # v2 columns landed without dropping the v1.25.1 ones.
    columns = _columns(db_path, "jobs")
    assert {"resumable", "partial_output_path", "exit_reason"} <= columns
    assert {"result_json", "payload_json", "completed_at"} <= columns

    listed = {job["id"] for job in job_store.list_jobs(limit=50)}
    assert set(LEGACY_JOB_IDS) <= listed

    finished = job_store.get_job(LEGACY_JOB_IDS[0])
    assert finished["status"] == "complete"
    assert finished["result"]["output"] == "/media/legacy/a-cut.mp4"


def test_interrupted_jobs_are_recovered_not_lost(legacy_user_data):
    import opencut.job_store as job_store

    with _no_network():
        job_store.init_db()
        job_store.mark_interrupted()

    recovered = {job["id"] for job in job_store.get_interrupted_jobs()}
    assert LEGACY_JOB_IDS[1] in recovered
    # The finished job must not be swept up by recovery.
    assert LEGACY_JOB_IDS[0] not in recovered


def test_journal_migrates_past_the_out_of_band_forward_json_column(legacy_user_data):
    from opencut import journal as journal_module

    # v1.25.1 added `forward_json` with a bare ALTER while user_version stayed
    # 0, so the v2 migration must be a no-op instead of a duplicate-column
    # failure.
    with _no_network():
        journal_module.init_db()

    db_path = legacy_user_data / "journal.db"
    assert _user_version(db_path) == journal_module.SCHEMA_VERSION
    columns = _columns(db_path, "journal")
    assert {"forward_json", "transaction_id", "status", "updated_at"} <= columns

    entries = journal_module.list_entries(limit=10)
    labels = {entry["label"] for entry in entries}
    assert set(LEGACY_JOURNAL_LABELS) <= labels


def test_journal_v3_backfills_timestamps_for_legacy_rows(legacy_user_data):
    from opencut import journal as journal_module

    with _no_network():
        journal_module.init_db()

    conn = sqlite3.connect(legacy_user_data / "journal.db")
    try:
        rows = conn.execute(
            "SELECT created_at, started_at, completed_at, updated_at FROM journal"
        ).fetchall()
    finally:
        conn.close()
    assert rows
    for created_at, started_at, completed_at, updated_at in rows:
        assert started_at == created_at
        assert completed_at == created_at
        assert updated_at == created_at


def test_footage_index_rebuilds_fts_without_losing_rows(legacy_user_data):
    import opencut.core.footage_index_db as footage_index

    with _no_network():
        footage_index.init_db()

    db_path = legacy_user_data / "footage_index.db"
    assert _user_version(db_path) == footage_index.SCHEMA_VERSION
    assert {"ocr_text", "audio_tags"} <= _columns(db_path, "footage")

    stats = footage_index.get_stats()
    assert stats["total_files"] == 1

    # The v2 migration drops and recreates the FTS table; search must still
    # find the pre-upgrade transcript rather than an empty index.
    hits = footage_index.search("edge cases", limit=5)
    assert [hit["file_path"] for hit in hits] == [LEGACY_FOOTAGE_PATH]


# ---------------------------------------------------------------------------
# JSON stores
# ---------------------------------------------------------------------------
def test_legacy_queue_list_migrates_to_a_versioned_document(legacy_user_data):
    from opencut import queue_store

    with _no_network():
        entries, migrated = queue_store.load_queue()

    assert migrated is True
    assert [entry["endpoint"] for entry in entries] == ["/silence"]

    queue_store.save_queue(entries)
    saved = json.loads((legacy_user_data / "job_queue.json").read_text(encoding="utf-8"))
    assert saved["schema_version"] == queue_store.QUEUE_SCHEMA_VERSION
    assert saved["entries"][0]["id"] == "queue-legacy-001"


def test_unversioned_settings_still_load(legacy_user_data):
    import opencut.user_data as user_data

    with _no_network():
        presets = user_data.load_presets()
        favorites = user_data.load_favorites()
        workflows = user_data.load_workflows()
        whisper = user_data.load_whisper_settings()
        loudness = user_data.load_loudness_target()
        chapters = user_data.load_chapter_defaults()

    assert "Quiet room" in presets
    assert favorites == ["silence", "captions"]
    assert "Legacy podcast" in workflows
    assert whisper["model"] == "base"
    assert whisper["cpu_mode"] is False
    persisted_whisper = json.loads(
        (legacy_user_data / "whisper_settings.json").read_text(encoding="utf-8")
    )
    assert persisted_whisper["_schema_version"] == user_data.WHISPER_SETTINGS_SCHEMA_VERSION
    assert loudness["target_lufs"] == -14.0
    # A key the legacy file omits must fall back to the current default
    # instead of coming back missing.
    assert loudness["true_peak"] == -1.0
    assert chapters["max_chapters"] == 8


def test_plaintext_llm_key_is_migrated_or_disabled_never_silently_kept(legacy_user_data):
    import opencut.user_data as user_data

    with _no_network():
        settings = user_data.load_llm_settings()

    assert settings["provider"] == "openai"
    on_disk = json.loads((legacy_user_data / "llm_settings.json").read_text(encoding="utf-8"))

    if settings["api_key"] == LEGACY_API_KEY:
        # Only acceptable when the key moved into a vault and the file was
        # sanitized, or when plaintext storage was explicitly opted into.
        assert on_disk.get("_credential_storage") in ("os_vault", "plaintext-opt-in")
        if on_disk.get("_credential_storage") == "os_vault":
            assert "api_key" not in on_disk
            assert on_disk.get("api_key_set") is True
    else:
        # No vault and no opt-in: the key is disabled rather than used.
        assert settings["api_key"] == ""


def test_review_records_migrate_with_a_backup(legacy_user_data):
    import opencut.core.review_links as review_links

    with _no_network():
        reviews = review_links._load_reviews()

    record = reviews[LEGACY_REVIEW_ID]
    assert record["schema_version"] == review_links.REVIEW_SCHEMA_VERSION
    assert record["versions"], "a pre-versioning record must gain a version list"
    assert record["current_version_id"]
    # Comments are re-anchored to the version they belong to.
    assert record["comments"][0]["version_id"] == record["current_version_id"]
    # The pre-migration file is preserved so the upgrade stays reversible.
    backup = legacy_user_data / "reviews" / "reviews.pre-versioning.json"
    assert backup.is_file()
    original = json.loads(backup.read_text(encoding="utf-8"))
    assert "versions" not in original[LEGACY_REVIEW_ID]


def test_missing_publisher_trust_store_is_not_treated_as_tampering(legacy_user_data):
    import opencut.core.plugin_installation as plugin_installation

    # v1.25.1 predates the trust store entirely.
    assert not (legacy_user_data / "trusted-plugin-publishers.json").exists()
    with _no_network():
        store = plugin_installation._load_trust_store()
    assert store == {"version": plugin_installation.TRUST_STORE_VERSION, "publishers": {}}


def test_installed_plugin_manifest_still_validates(legacy_user_data):
    from opencut.core import plugin_manifest

    manifest_path = legacy_user_data / "plugins" / LEGACY_PLUGIN_NAME / "plugin.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    with _no_network():
        result = plugin_manifest.validate_manifest_schema(payload)
    assert result.valid, result.errors


# ---------------------------------------------------------------------------
# Whole-tree invariants
# ---------------------------------------------------------------------------
def test_upgrade_adds_and_rewrites_but_never_deletes_user_files(legacy_user_data):
    import opencut.core.footage_index_db as footage_index
    import opencut.core.review_links as review_links
    import opencut.job_store as job_store
    import opencut.user_data as user_data
    from opencut import journal as journal_module
    from opencut import queue_store

    before = snapshot_tree(legacy_user_data)

    with _no_network():
        job_store.init_db()
        journal_module.init_db()
        footage_index.init_db()
        queue_store.load_queue()
        review_links._load_reviews()
        user_data.load_presets()
        user_data.load_llm_settings()

    job_store.close_all_connections()
    journal_module.close_all_connections()
    footage_index.close_all_connections()

    after = snapshot_tree(legacy_user_data)
    missing = sorted(set(before) - set(after))
    assert not missing, f"upgrade removed user files: {missing}"

    # The installer stamp is the only record of the previous version, so a
    # migration must not clobber it.
    installer = json.loads((legacy_user_data / "installer.json").read_text(encoding="utf-8"))
    assert installer["app_version"] == LEGACY_VERSION


def test_upgrade_is_idempotent(legacy_user_data):
    import opencut.core.footage_index_db as footage_index
    import opencut.job_store as job_store
    from opencut import journal as journal_module

    for _ in range(2):
        with _no_network():
            job_store.init_db()
            journal_module.init_db()
            footage_index.init_db()
        job_store.close_all_connections()
        journal_module.close_all_connections()
        footage_index.close_all_connections()
        job_store._INITIALIZED = False
        job_store._INITIALIZED_PATH = None

    assert _user_version(legacy_user_data / "jobs.db") == job_store.SCHEMA_VERSION
    assert _user_version(legacy_user_data / "journal.db") == journal_module.SCHEMA_VERSION
    assert _user_version(legacy_user_data / "footage_index.db") == footage_index.SCHEMA_VERSION
    assert footage_index.get_stats()["total_files"] == 1


# ---------------------------------------------------------------------------
# Rollback: uninstall must leave the upgraded tree recoverable
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALLER_SRC = REPO_ROOT / "installer" / "src" / "OpenCut.Installer"


def _read_cs(relative: str) -> str:
    return (INSTALLER_SRC / relative).read_text(encoding="utf-8", errors="replace")


def test_uninstall_preserves_user_data_by_default():
    config = _read_cs("Models/InstallConfig.cs")
    # An auto-defaulted bool is false, i.e. "keep my data" unless asked.
    assert "public bool RemoveUserData { get; set; }" in config
    assert "RemoveUserData { get; set; } = true" not in config

    uninstall = _read_cs("Services/UninstallEngine.cs")
    assert "_config.RemoveUserData" in uninstall
    assert "Preserving OpenCut user data" in uninstall


def test_requested_user_data_removal_is_backed_up_and_verified_first():
    removal = _read_cs("Services/UserDataRemovalService.cs")
    for guarantee in ("CreateBackup", "ValidateBackup", "EnsureSafeSource"):
        assert guarantee in removal, f"user-data removal must still {guarantee}"
    # The backup is validated before the source is touched.
    assert removal.index("ValidateBackup") < removal.index("RemoveSourceAtomically")
