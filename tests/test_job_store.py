"""
Tests for OpenCut Job Store (SQLite persistence).
"""

import time
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def isolate_db(tmp_path):
    """Each test gets its own SQLite database."""
    db_path = str(tmp_path / "test_jobs.db")
    with patch("opencut.job_store._DB_PATH", db_path):
        import opencut.job_store as store
        store._INITIALIZED = False
        store._LOCAL = type(store._LOCAL)()  # fresh thread-local
        store._ALL_CONNECTIONS = {}
        yield store


class TestJobStore:

    def test_init_creates_table(self, isolate_db):
        isolate_db.init_db()
        # Should not raise
        jobs = isolate_db.list_jobs()
        assert isinstance(jobs, list)
        assert len(jobs) == 0

    def test_save_and_get_job(self, isolate_db):
        job = {
            "id": "abc123",
            "type": "silence",
            "filepath": "/tmp/test.wav",
            "status": "complete",
            "progress": 100,
            "message": "Done",
            "result": {"segments": 5, "output": "/tmp/out.wav"},
            "error": None,
            "created": time.time(),
        }
        isolate_db.save_job(job)
        retrieved = isolate_db.get_job("abc123")
        assert retrieved is not None
        assert retrieved["id"] == "abc123"
        assert retrieved["type"] == "silence"
        assert retrieved["status"] == "complete"
        assert retrieved["result"]["segments"] == 5

    def test_get_nonexistent_job(self, isolate_db):
        assert isolate_db.get_job("nonexistent") is None

    def test_list_jobs_filtered(self, isolate_db):
        for i, status in enumerate(["complete", "error", "complete", "cancelled"]):
            isolate_db.save_job({
                "id": f"job{i}",
                "type": "test",
                "status": status,
                "created": time.time() - i,
            })
        all_jobs = isolate_db.list_jobs()
        assert len(all_jobs) == 4

        completed = isolate_db.list_jobs(status="complete")
        assert len(completed) == 2

        errors = isolate_db.list_jobs(status="error")
        assert len(errors) == 1

    def test_update_existing_job(self, isolate_db):
        isolate_db.save_job({
            "id": "upd1",
            "type": "test",
            "status": "running",
            "progress": 50,
            "created": time.time(),
        })
        # Update status
        isolate_db.save_job({
            "id": "upd1",
            "type": "test",
            "status": "complete",
            "progress": 100,
            "message": "Done",
            "result": {"ok": True},
            "created": time.time(),
        })
        job = isolate_db.get_job("upd1")
        assert job["status"] == "complete"
        assert job["progress"] == 100

    def test_mark_interrupted(self, isolate_db):
        isolate_db.save_job({
            "id": "running1",
            "type": "test",
            "status": "running",
            "created": time.time(),
        })
        isolate_db.save_job({
            "id": "done1",
            "type": "test",
            "status": "complete",
            "created": time.time(),
        })
        count = isolate_db.mark_interrupted()
        assert count == 1
        job = isolate_db.get_job("running1")
        assert job["status"] == "interrupted"
        # Complete job should be unchanged
        done = isolate_db.get_job("done1")
        assert done["status"] == "complete"

    def test_cleanup_old_jobs(self, isolate_db):
        old_time = time.time() - (8 * 24 * 3600)  # 8 days ago
        isolate_db.save_job({
            "id": "old1",
            "type": "test",
            "status": "complete",
            "created": old_time,
            "completed_at": old_time,
        })
        isolate_db.save_job({
            "id": "new1",
            "type": "test",
            "status": "complete",
            "created": time.time(),
        })
        count = isolate_db.cleanup_old_jobs()
        assert count == 1
        assert isolate_db.get_job("old1") is None
        assert isolate_db.get_job("new1") is not None

    def test_cleanup_old_jobs_uses_completed_at_when_present(self, isolate_db):
        old_created = time.time() - (8 * 24 * 3600)
        isolate_db.save_job({
            "id": "recently-finished",
            "type": "test",
            "status": "complete",
            "created": old_created,
            "completed_at": time.time(),
        })

        count = isolate_db.cleanup_old_jobs()

        assert count == 0
        assert isolate_db.get_job("recently-finished") is not None

    def test_update_existing_job_persists_endpoint_and_payload(self, isolate_db):
        created = time.time()
        isolate_db.save_job({
            "id": "prov1",
            "type": "test",
            "status": "running",
            "created": created,
            "_endpoint": "/silence",
            "_payload": {"threshold": -28, "mode": "fast"},
        })
        isolate_db.save_job({
            "id": "prov1",
            "type": "test",
            "status": "complete",
            "progress": 100,
            "message": "Done",
            "result": {"ok": True},
            "created": created,
            "started_at": created + 1,
        })

        job = isolate_db.get_job("prov1")

        assert job["status"] == "complete"
        assert job["endpoint"] == "/silence"
        assert job["payload"] == {"threshold": -28, "mode": "fast"}
        assert job["started_at"] == pytest.approx(created + 1)
        assert job["completed_at"] is not None

    def test_get_job_stats(self, isolate_db):
        for status in ["complete", "complete", "error", "cancelled"]:
            isolate_db.save_job({
                "id": f"stat_{status}_{time.time()}",
                "type": "test",
                "status": status,
                "created": time.time(),
            })
        stats = isolate_db.get_job_stats()
        assert stats["complete"] == 2
        assert stats["error"] == 1
        assert stats["cancelled"] == 1
        assert stats["total"] == 4

    def test_list_with_limit_and_offset(self, isolate_db):
        for i in range(10):
            isolate_db.save_job({
                "id": f"page{i}",
                "type": "test",
                "status": "complete",
                "created": time.time() + i,
            })
        page1 = isolate_db.list_jobs(limit=3, offset=0)
        page2 = isolate_db.list_jobs(limit=3, offset=3)
        assert len(page1) == 3
        assert len(page2) == 3
        assert page1[0]["id"] != page2[0]["id"]
