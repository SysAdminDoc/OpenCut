"""Tests for SQLite-backed footage index."""

import pytest


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Use a temporary database for all tests."""
    import opencut.core.footage_index_db as mod
    db_path = str(tmp_path / "test_footage.db")
    monkeypatch.setattr(mod, "_DB_PATH", db_path)
    # Reset thread-local connection
    mod._thread_local.conn = None
    mod.init_db()
    yield db_path


class TestIndexFile:
    def test_index_and_search(self, tmp_path):
        from opencut.core.footage_index_db import index_file, search

        # Create a dummy file
        f = tmp_path / "test.mp4"
        f.write_bytes(b"fake video")

        index_file(str(f), "Hello world this is a test transcript", duration=60)
        results = search("test transcript")
        assert len(results) >= 1
        assert results[0]["file_path"] == str(f)

    def test_search_returns_snippet(self, tmp_path):
        from opencut.core.footage_index_db import index_file, search

        f = tmp_path / "clip.mp4"
        f.write_bytes(b"x")
        index_file(str(f), "The quick brown fox jumps over the lazy dog")

        results = search("brown fox")
        assert len(results) >= 1
        assert "snippet" in results[0]

    def test_search_empty_query(self):
        from opencut.core.footage_index_db import search
        assert search("") == []
        assert search("   ") == []

    def test_search_coerces_invalid_limit(self, tmp_path):
        from opencut.core.footage_index_db import index_file, search

        f1 = tmp_path / "one.mp4"
        f2 = tmp_path / "two.mp4"
        f1.write_bytes(b"x")
        f2.write_bytes(b"y")

        index_file(str(f1), "shared transcript")
        index_file(str(f2), "shared transcript")

        results = search("shared transcript", limit=0)

        assert len(results) == 1

    def test_upsert(self, tmp_path):
        from opencut.core.footage_index_db import index_file, search

        f = tmp_path / "vid.mp4"
        f.write_bytes(b"x")

        index_file(str(f), "first transcript")
        index_file(str(f), "updated transcript with new content")

        results = search("updated transcript")
        assert len(results) == 1

        # Old transcript should not match
        old_results = search("first transcript")
        assert len(old_results) == 0


class TestNeedsReindex:
    def test_new_file_needs_index(self, tmp_path):
        from opencut.core.footage_index_db import needs_reindex

        f = tmp_path / "new.mp4"
        f.write_bytes(b"x")
        assert needs_reindex(str(f)) is True

    def test_indexed_file_no_reindex(self, tmp_path):
        from opencut.core.footage_index_db import index_file, needs_reindex

        f = tmp_path / "done.mp4"
        f.write_bytes(b"x")
        index_file(str(f), "transcript here")
        assert needs_reindex(str(f)) is False


class TestStats:
    def test_empty_stats(self):
        from opencut.core.footage_index_db import get_stats
        stats = get_stats()
        assert stats["total_files"] == 0

    def test_stats_after_index(self, tmp_path):
        from opencut.core.footage_index_db import get_stats, index_file

        f = tmp_path / "a.mp4"
        f.write_bytes(b"data" * 100)
        index_file(str(f), "some text", duration=30, file_size=400)

        stats = get_stats()
        assert stats["total_files"] == 1
        assert stats["total_size"] == 400


class TestClearAndCleanup:
    def test_clear(self, tmp_path):
        from opencut.core.footage_index_db import clear_index, get_stats, index_file

        f = tmp_path / "z.mp4"
        f.write_bytes(b"x")
        index_file(str(f), "content")
        clear_index()
        assert get_stats()["total_files"] == 0

    def test_remove_missing(self, tmp_path):
        from opencut.core.footage_index_db import get_stats, index_file, remove_missing_files

        # Index a file that exists
        f1 = tmp_path / "exists.mp4"
        f1.write_bytes(b"x")
        index_file(str(f1), "exists")

        # Index a file path that doesn't exist
        index_file("/nonexistent/fake.mp4", "gone")

        assert get_stats()["total_files"] == 2
        removed = remove_missing_files()
        assert removed == 1
        assert get_stats()["total_files"] == 1

    def test_get_conn_prunes_dead_thread_connections(self):
        import opencut.core.footage_index_db as mod

        class DummyConn:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        dummy = DummyConn()
        mod._ALL_CONNECTIONS[999999] = dummy

        import threading
        from unittest.mock import patch

        with patch.object(mod.threading, "enumerate", return_value=[threading.current_thread()]):
            mod._get_conn()

        assert dummy.closed is True
        assert 999999 not in mod._ALL_CONNECTIONS
