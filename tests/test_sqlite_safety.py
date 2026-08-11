"""F309 — FTS5 memory-safety floor for locally-opened SQLite databases.

CVE-2026-11822 corrupts memory when an FTS5 ``MATCH`` runs against a crafted
database file (fixed in SQLite 3.53.2). These tests pin the narrow policy: a
foreign index is refused on an unpatched runtime, while an index this install
owns keeps working everywhere — adding the guard must not brick existing users.
"""

from __future__ import annotations

import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opencut.core import sqlite_safety as ss  # noqa: E402


def _make_db(path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE IF NOT EXISTS t (x TEXT)")
    return conn


class TestVersionPolicy:
    def test_floor_is_the_published_fix_version(self):
        assert ss.FTS5_SAFE_VERSION == (3, 53, 2)
        assert ss.CVE_REFERENCE == "CVE-2026-11822"

    def test_runtime_version_is_reported(self):
        assert ss.runtime_version() == sqlite3.sqlite_version
        assert len(ss.runtime_version_tuple()) == 3

    def test_patched_flag_tracks_the_floor(self):
        assert ss.fts5_runtime_is_patched() == (
            ss.runtime_version_tuple() >= ss.FTS5_SAFE_VERSION
        )

    def test_report_names_the_active_policy(self):
        report = ss.fts5_safety_report()
        assert report["fts5_floor"] == "3.53.2"
        assert report["cve"] == "CVE-2026-11822"
        assert isinstance(report["fts5_runtime_patched"], bool)

    def test_version_parsing_tolerates_junk(self):
        assert ss.resolve_optional_version("3.53") == (3, 53, 0)
        assert ss.resolve_optional_version(None) == (0, 0, 0)
        assert ss.resolve_optional_version("3.x.2") == (3, 0, 2)


class TestProvenanceStamp:
    def test_stamp_round_trips(self, tmp_path):
        conn = _make_db(tmp_path / "a.db")
        assert ss.has_local_provenance(conn) is False
        ss.stamp_local_provenance(conn)
        assert ss.has_local_provenance(conn) is True
        conn.close()

    def test_stamp_survives_reopen(self, tmp_path):
        path = tmp_path / "b.db"
        conn = _make_db(path)
        ss.stamp_local_provenance(conn)
        conn.commit()
        conn.close()
        reopened = sqlite3.connect(str(path))
        assert ss.has_local_provenance(reopened) is True
        reopened.close()


class TestTrustDecision:
    def test_freshly_created_database_is_trusted_and_stamped(self, tmp_path):
        conn = _make_db(tmp_path / "new.db")
        ss.ensure_fts5_database_trusted(conn, str(tmp_path / "new.db"), created_here=True)
        assert ss.has_local_provenance(conn) is True
        conn.close()

    def test_foreign_database_is_refused_on_an_unpatched_runtime(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ss, "fts5_runtime_is_patched", lambda: False)
        monkeypatch.setattr(ss, "is_inside_user_data_dir", lambda _p: False)
        path = tmp_path / "downloaded.db"
        conn = _make_db(path)
        with pytest.raises(ss.UntrustedFts5DatabaseError) as excinfo:
            ss.ensure_fts5_database_trusted(conn, str(path))
        conn.close()
        assert "CVE-2026-11822" in str(excinfo.value)
        assert excinfo.value.code == "UNTRUSTED_FTS5_DATABASE"

    def test_foreign_database_is_allowed_on_a_patched_runtime(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ss, "fts5_runtime_is_patched", lambda: True)
        monkeypatch.setattr(ss, "is_inside_user_data_dir", lambda _p: False)
        path = tmp_path / "downloaded.db"
        conn = _make_db(path)
        ss.ensure_fts5_database_trusted(conn, str(path))  # must not raise
        conn.close()

    def test_previously_stamped_database_is_trusted_when_unpatched(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ss, "fts5_runtime_is_patched", lambda: False)
        monkeypatch.setattr(ss, "is_inside_user_data_dir", lambda _p: False)
        path = tmp_path / "ours.db"
        conn = _make_db(path)
        ss.stamp_local_provenance(conn)
        ss.ensure_fts5_database_trusted(conn, str(path))  # must not raise
        conn.close()

    def test_existing_index_in_the_user_data_dir_is_adopted(self, tmp_path, monkeypatch):
        """Adding the guard must not refuse an index users already have."""
        monkeypatch.setattr(ss, "fts5_runtime_is_patched", lambda: False)
        monkeypatch.setattr(ss, "is_inside_user_data_dir", lambda _p: True)
        path = tmp_path / "legacy_unstamped.db"
        conn = _make_db(path)
        assert ss.has_local_provenance(conn) is False
        ss.ensure_fts5_database_trusted(conn, str(path))  # must not raise
        assert ss.has_local_provenance(conn) is True, "adopted indexes get stamped"
        conn.close()


class TestUserDataOwnership:
    def test_user_data_paths_are_recognised(self):
        from opencut.user_data import OPENCUT_DIR

        assert ss.is_inside_user_data_dir(os.path.join(OPENCUT_DIR, "footage_index.db"))

    def test_outside_paths_are_not(self, tmp_path):
        assert ss.is_inside_user_data_dir(str(tmp_path / "elsewhere.db")) is False


class TestLiveIndexes:
    def test_footage_index_still_opens(self, tmp_path, monkeypatch):
        from opencut.core import footage_index_db as db

        monkeypatch.setattr(db, "_DB_PATH", str(tmp_path / "footage.db"))
        db.close_all_connections()
        db.init_db()
        db.index_file(str(tmp_path / "clip.mp4"), "hello world", duration=1.0)
        assert db.search("hello")
        db.close_all_connections()
