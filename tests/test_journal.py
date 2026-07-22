"""Tests for the Operation Journal (v1.9.28)."""

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch


def _isolate_journal():
    """Point journal.py at a throwaway DB path + reset any thread-local
    connection so each test starts clean.
    """
    from opencut import journal as journal_module

    tmpdir = tempfile.mkdtemp(prefix="opencut_journal_test_")
    db_path = os.path.join(tmpdir, "journal.db")

    # Close any pre-existing thread-local conn
    try:
        if getattr(journal_module._thread_local, "conn", None):
            journal_module._thread_local.conn.close()
            journal_module._thread_local.conn = None
    except Exception:
        pass
    journal_module._ALL_CONNECTIONS.clear()
    journal_module._DB_PATH = db_path
    return tmpdir, db_path


class TestJournalStore(unittest.TestCase):
    def setUp(self):
        self._tmp, self._db = _isolate_journal()

    def tearDown(self):
        from opencut import journal as jm
        try:
            jm.close_all_connections()
        except Exception:
            pass

    def test_record_and_list(self):
        from opencut import journal as jm
        entry = jm.record(
            "add_markers", "3 beat markers", {"markers": [{"time": 1.0}]},
            clip_path="/tmp/x.mp4",
        )
        self.assertTrue(entry["id"])
        self.assertEqual(entry["action"], "add_markers")
        self.assertFalse(entry["reverted"])
        self.assertTrue(entry["revertible"])
        self.assertEqual(entry["inverse"]["markers"][0]["time"], 1.0)

        rows = jm.list_entries(limit=5)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], entry["id"])

    def test_unknown_action_rejected(self):
        from opencut import journal as jm
        with self.assertRaises(ValueError):
            jm.record("bogus", "x", {})

    def test_mark_reverted_is_one_shot(self):
        from opencut import journal as jm
        e = jm.record("batch_rename", "x", {"renames": []})
        self.assertTrue(jm.mark_reverted(e["id"]))
        # Second attempt returns False (idempotent)
        self.assertFalse(jm.mark_reverted(e["id"]))

    def test_include_reverted_filter(self):
        from opencut import journal as jm
        a = jm.record("add_markers", "a", {})
        b = jm.record("add_markers", "b", {})
        jm.mark_reverted(a["id"])

        active = jm.list_entries(include_reverted=False)
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["id"], b["id"])

    def test_clear_all(self):
        from opencut import journal as jm
        jm.record("add_markers", "a", {})
        jm.record("batch_rename", "b", {})
        self.assertEqual(jm.clear_all(), 2)
        self.assertEqual(jm.list_entries(), [])

    def test_revertible_flag_maps_action(self):
        from opencut import journal as jm
        revertible = jm.record("add_markers", "x", {})
        self.assertTrue(revertible["revertible"])
        # import_captions is recorded but has no auto-revert
        info_only = jm.record("import_captions", "y", {})
        self.assertFalse(info_only["revertible"])

    def test_list_entries_coerces_invalid_limit(self):
        from opencut import journal as jm
        jm.record("add_markers", "a", {})
        jm.record("add_markers", "b", {})

        rows = jm.list_entries(limit=0)

        self.assertEqual(len(rows), 1)

    def test_get_conn_prunes_dead_thread_connections(self):
        from opencut import journal as jm

        class DummyConn:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        dummy = DummyConn()
        jm._ALL_CONNECTIONS[999999] = dummy

        with patch.object(jm.threading, "enumerate", return_value=[threading.current_thread()]):
            jm._get_conn()

        self.assertTrue(dummy.closed)
        self.assertNotIn(999999, jm._ALL_CONNECTIONS)

    def test_large_payloads_spill_to_content_files(self):
        from opencut import journal as jm

        original_limit = jm.MAX_JOURNAL_PAYLOAD_JSON_BYTES
        jm.MAX_JOURNAL_PAYLOAD_JSON_BYTES = 128
        try:
            entry = jm.record(
                "add_markers",
                "large",
                {"markers": [{"comment": "x" * 2048}]},
                forward_payload={
                    "endpoint": "/timeline/markers",
                    "payload": {"comment": "y" * 2048},
                },
            )
        finally:
            jm.MAX_JOURNAL_PAYLOAD_JSON_BYTES = original_limit

        self.assertTrue(entry["inverse"]["_opencut_payload_spill"])
        self.assertEqual(entry["inverse"]["field"], "inverse_json")
        self.assertTrue(Path(entry["inverse"]["path"]).is_file())
        self.assertEqual(
            os.path.commonpath([os.path.dirname(jm._DB_PATH), entry["inverse"]["path"]]),
            os.path.dirname(jm._DB_PATH),
        )
        with open(entry["inverse"]["path"], encoding="utf-8") as fh:
            self.assertEqual(json.load(fh), {"markers": [{"comment": "x" * 2048}]})

        self.assertTrue(entry["forward"]["_opencut_payload_spill"])
        self.assertEqual(entry["forward"]["field"], "forward_json")
        self.assertTrue(Path(entry["forward"]["path"]).is_file())

    def test_pending_checkpoint_survives_restart_and_completes_atomically(self):
        from opencut import journal as jm

        checkpoint = jm.begin_checkpoint(
            "add_markers",
            "Add two chapter markers",
            inverse_payload={"markers": [{"time": 2.0, "comment": "Chapter"}]},
            preview_payload={"clips": ["Interview"], "settings": {"count": 2}},
        )
        self.assertEqual(checkpoint["status"], jm.CHECKPOINT_PENDING)
        self.assertTrue(checkpoint["automatic_recovery_available"])

        jm.close_all_connections()
        recovered = jm.list_incomplete_checkpoints()
        self.assertEqual([row["transaction_id"] for row in recovered], [checkpoint["transaction_id"]])
        self.assertEqual(recovered[0]["preview"]["clips"], ["Interview"])

        completed = jm.complete_checkpoint(
            checkpoint["transaction_id"],
            diagnostics_payload={"host_result": {"added": 2}},
        )
        self.assertEqual(completed["status"], jm.CHECKPOINT_COMPLETED)
        self.assertIsNotNone(completed["completed_at"])
        self.assertEqual(jm.list_incomplete_checkpoints(), [])

    def test_checkpoint_survives_abrupt_process_exit(self):
        from opencut import journal as jm

        jm.close_all_connections()
        script = """
import os
import sys
from opencut import journal
journal._DB_PATH = sys.argv[1]
journal.begin_checkpoint(
    "batch_rename",
    "Crash injection",
    inverse_payload={"renames": [{"nodeId": "7", "newName": "Before"}]},
    preview_payload={"items": [{"before": "Before", "after": "After"}]},
)
os._exit(73)
"""
        crashed = subprocess.run(
            [sys.executable, "-c", script, self._db],
            cwd=Path(__file__).resolve().parents[1],
            check=False,
            timeout=30,
        )
        self.assertEqual(crashed.returncode, 73)
        recovered = jm.list_incomplete_checkpoints()
        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0]["label"], "Crash injection")
        self.assertTrue(recovered[0]["automatic_recovery_available"])

    def test_v2_database_migrates_completed_rows_losslessly(self):
        from opencut import journal as jm

        jm.close_all_connections()
        conn = sqlite3.connect(self._db)
        conn.execute("""
            CREATE TABLE journal (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at REAL NOT NULL,
                action TEXT NOT NULL,
                clip_path TEXT NOT NULL DEFAULT '',
                label TEXT NOT NULL DEFAULT '',
                inverse_json TEXT NOT NULL DEFAULT '{}',
                reverted INTEGER NOT NULL DEFAULT 0,
                reverted_at REAL,
                forward_json TEXT
            )
        """)
        conn.execute(
            """INSERT INTO journal (
                   created_at, action, label, inverse_json, forward_json
               ) VALUES (?, ?, ?, ?, ?)""",
            (10.0, "add_markers", "legacy", '{"markers": [{"time": 1}]}', None),
        )
        conn.execute("PRAGMA user_version = 2")
        conn.commit()
        conn.close()

        jm.init_db()
        rows = jm.list_entries()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label"], "legacy")
        self.assertEqual(rows[0]["status"], jm.CHECKPOINT_COMPLETED)
        self.assertEqual(rows[0]["completed_at"], 10.0)
        self.assertIsNone(rows[0]["transaction_id"])

    def test_failed_recovery_keeps_diagnostics_and_can_be_retried(self):
        from opencut import journal as jm

        checkpoint = jm.begin_checkpoint(
            "batch_rename",
            "Rename two items",
            inverse_payload={"renames": [{"nodeId": "1", "newName": "Before"}]},
            preview_payload={"items": [{"before": "Before", "after": "After"}]},
        )
        failed = jm.mark_recovery_failed(
            checkpoint["transaction_id"],
            "Premiere project is not open",
            diagnostics_payload={"host": "offline"},
        )
        self.assertEqual(failed["status"], jm.CHECKPOINT_RECOVERY_FAILED)
        self.assertEqual(failed["diagnostics"], {"host": "offline"})
        self.assertTrue(failed["automatic_recovery_available"])

        restored = jm.mark_recovered(checkpoint["transaction_id"])
        self.assertEqual(restored["status"], jm.CHECKPOINT_RESTORED)
        self.assertTrue(restored["reverted"])
        self.assertEqual(jm.list_incomplete_checkpoints(), [])

    def test_spilled_inverse_is_verified_before_recovery(self):
        from opencut import journal as jm

        original_limit = jm.MAX_JOURNAL_PAYLOAD_JSON_BYTES
        jm.MAX_JOURNAL_PAYLOAD_JSON_BYTES = 64
        try:
            checkpoint = jm.begin_checkpoint(
                "add_markers",
                "Large marker plan",
                inverse_payload={"markers": [{"comment": "x" * 512}]},
            )
        finally:
            jm.MAX_JOURNAL_PAYLOAD_JSON_BYTES = original_limit

        self.assertTrue(checkpoint["automatic_recovery_available"])
        resolved = jm.get_checkpoint(checkpoint["transaction_id"], resolve_payloads=True)
        self.assertEqual(resolved["inverse"]["markers"][0]["comment"], "x" * 512)

        Path(checkpoint["inverse"]["path"]).write_text("{}", encoding="utf-8")
        corrupted = jm.get_checkpoint(checkpoint["transaction_id"])
        self.assertFalse(corrupted["automatic_recovery_available"])
        self.assertEqual(
            corrupted["artifact_integrity"]["inverse"]["state"],
            "hash_mismatch",
        )

    def test_retention_never_prunes_incomplete_checkpoint(self):
        from opencut import journal as jm

        original_limit = jm.MAX_JOURNAL_ENTRIES
        jm.MAX_JOURNAL_ENTRIES = 2
        try:
            pending = jm.begin_checkpoint("apply_cuts", "Interrupted cut", preview_payload={"cuts": 3})
            for index in range(4):
                jm.record("add_markers", f"done-{index}", {"markers": []})
        finally:
            jm.MAX_JOURNAL_ENTRIES = original_limit

        rows = jm.list_entries(limit=20)
        self.assertEqual(len(rows), 3)
        self.assertIn(pending["transaction_id"], {row["transaction_id"] for row in rows})

    def test_pruning_removes_orphan_payload_spills(self):
        from opencut import journal as jm

        original_entries = jm.MAX_JOURNAL_ENTRIES
        original_bytes = jm.MAX_JOURNAL_PAYLOAD_JSON_BYTES
        jm.MAX_JOURNAL_ENTRIES = 1
        jm.MAX_JOURNAL_PAYLOAD_JSON_BYTES = 64
        try:
            old = jm.record("add_markers", "old", {"markers": [{"comment": "x" * 512}]})
            spill_path = Path(old["inverse"]["path"])
            self.assertTrue(spill_path.is_file())
            jm.record("add_markers", "new", {"markers": []})
        finally:
            jm.MAX_JOURNAL_ENTRIES = original_entries
            jm.MAX_JOURNAL_PAYLOAD_JSON_BYTES = original_bytes
        self.assertFalse(spill_path.exists())

    def test_clear_preserves_incomplete_recovery_evidence(self):
        from opencut import journal as jm

        pending = jm.begin_checkpoint("apply_cuts", "Interrupted cut")
        jm.record("add_markers", "completed", {"markers": []})
        self.assertEqual(jm.clear_all(), 1)
        rows = jm.list_entries()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["transaction_id"], pending["transaction_id"])


class TestJournalRoutes(unittest.TestCase):
    def setUp(self):
        self._tmp, self._db = _isolate_journal()
        from opencut.server import app
        app.config["TESTING"] = True
        self.client = app.test_client()
        resp = self.client.get("/health")
        self.csrf = resp.get_json().get("csrf_token", "")

    def tearDown(self):
        from opencut import journal as jm
        try:
            jm.close_all_connections()
        except Exception:
            pass

    def _h(self):
        return {"Content-Type": "application/json", "X-OpenCut-Token": self.csrf}

    def test_record_requires_csrf(self):
        r = self.client.post("/journal/record", json={"action": "add_markers"})
        self.assertEqual(r.status_code, 403)

    def test_record_round_trip(self):
        body = {
            "action": "add_markers",
            "label": "3 beats",
            "clip_path": "/tmp/clip.mp4",
            "inverse": {"markers": [{"time": 0.5, "comment": "Beat"}]},
        }
        r = self.client.post("/journal/record",
                             data=json.dumps(body),
                             headers=self._h())
        self.assertEqual(r.status_code, 201)
        entry = r.get_json()
        self.assertEqual(entry["action"], "add_markers")

        lst = self.client.get("/journal/list").get_json()
        self.assertEqual(len(lst), 1)
        self.assertEqual(lst[0]["id"], entry["id"])

    def test_record_rejects_unknown_action(self):
        r = self.client.post("/journal/record",
                             data=json.dumps({"action": "bogus"}),
                             headers=self._h())
        self.assertEqual(r.status_code, 400)
        self.assertIn("valid", r.get_json())

    def test_record_rejects_non_dict_inverse(self):
        r = self.client.post("/journal/record",
                             data=json.dumps({
                                 "action": "add_markers",
                                 "inverse": "not a dict",
                             }),
                             headers=self._h())
        self.assertEqual(r.status_code, 400)

    def test_mark_reverted_flow(self):
        # Record
        e = self.client.post("/journal/record",
                             data=json.dumps({
                                 "action": "add_markers",
                                 "inverse": {"markers": []},
                             }),
                             headers=self._h()).get_json()
        eid = e["id"]

        # Mark reverted
        r = self.client.post(f"/journal/mark-reverted/{eid}",
                             data=json.dumps({}),
                             headers=self._h())
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.get_json()["reverted"])

        # Second attempt => 409
        r2 = self.client.post(f"/journal/mark-reverted/{eid}",
                              data=json.dumps({}),
                              headers=self._h())
        self.assertEqual(r2.status_code, 409)

    def test_mark_reverted_404_for_unknown(self):
        r = self.client.post("/journal/mark-reverted/99999",
                             data=json.dumps({}),
                             headers=self._h())
        self.assertEqual(r.status_code, 404)

    def test_mark_reverted_rejects_non_revertible(self):
        e = self.client.post("/journal/record",
                             data=json.dumps({
                                 "action": "import_captions",
                                 "inverse": {},
                             }),
                             headers=self._h()).get_json()
        r = self.client.post(f"/journal/mark-reverted/{e['id']}",
                             data=json.dumps({}),
                             headers=self._h())
        self.assertEqual(r.status_code, 400)

    def test_delete_and_clear(self):
        e = self.client.post("/journal/record",
                             data=json.dumps({"action": "add_markers"}),
                             headers=self._h()).get_json()
        r = self.client.delete(f"/journal/{e['id']}", headers=self._h())
        self.assertEqual(r.status_code, 200)

        # clear on empty returns removed=0
        c = self.client.post("/journal/clear",
                             data=json.dumps({}),
                             headers=self._h())
        self.assertEqual(c.status_code, 200)
        self.assertEqual(c.get_json()["removed"], 0)

    def test_checkpoint_route_crash_recovery_flow(self):
        body = {
            "action": "add_markers",
            "label": "Add chapter markers",
            "inverse": {"markers": [{"time": 1.5, "comment": "Chapter"}]},
            "preview": {"clips": ["Interview"], "settings": {"count": 1}},
        }
        created = self.client.post(
            "/journal/checkpoints", data=json.dumps(body), headers=self._h()
        )
        self.assertEqual(created.status_code, 201)
        checkpoint = created.get_json()
        txid = checkpoint["transaction_id"]

        pending = self.client.get("/journal/recovery").get_json()
        self.assertEqual(pending["count"], 1)
        self.assertEqual(pending["checkpoints"][0]["transaction_id"], txid)
        self.assertTrue(pending["checkpoints"][0]["automatic_recovery_available"])

        resolved = self.client.get(f"/journal/checkpoints/{txid}?resolve=1")
        self.assertEqual(resolved.status_code, 200)
        self.assertEqual(resolved.get_json()["inverse"]["markers"][0]["time"], 1.5)

        failure = self.client.post(
            f"/journal/checkpoints/{txid}/recovery-failed",
            data=json.dumps({"error": "host unavailable", "diagnostics": {"host": "offline"}}),
            headers=self._h(),
        )
        self.assertEqual(failure.status_code, 200)
        self.assertEqual(failure.get_json()["status"], "recovery_failed")

        diagnostics = self.client.get(f"/journal/checkpoints/{txid}/diagnostics")
        self.assertEqual(diagnostics.status_code, 200)
        self.assertIn("attachment", diagnostics.headers["Content-Disposition"])
        self.assertEqual(diagnostics.get_json()["checkpoint"]["transaction_id"], txid)

        restored = self.client.post(
            f"/journal/checkpoints/{txid}/recovered",
            data=json.dumps({}),
            headers=self._h(),
        )
        self.assertEqual(restored.status_code, 200)
        self.assertEqual(restored.get_json()["status"], "restored")
        self.assertEqual(self.client.get("/journal/recovery").get_json()["count"], 0)

    def test_checkpoint_completion_and_validation_routes(self):
        invalid = self.client.post(
            "/journal/checkpoints",
            data=json.dumps({"action": "bogus", "preview": []}),
            headers=self._h(),
        )
        self.assertEqual(invalid.status_code, 400)

        created = self.client.post(
            "/journal/checkpoints",
            data=json.dumps({"action": "import_sequence", "label": "Import XML"}),
            headers=self._h(),
        ).get_json()
        completed = self.client.post(
            f"/journal/checkpoints/{created['transaction_id']}/complete",
            data=json.dumps({
                "inverse": {"name": "OpenCut Sequence"},
                "diagnostics": {"host_result": {"sequence_name": "OpenCut Sequence"}},
            }),
            headers=self._h(),
        )
        self.assertEqual(completed.status_code, 200)
        self.assertEqual(completed.get_json()["status"], "completed")
        self.assertEqual(self.client.get("/journal/recovery").get_json()["count"], 0)

    def test_cleanup_routes_preserve_incomplete_checkpoint(self):
        created = self.client.post(
            "/journal/checkpoints",
            data=json.dumps({
                "action": "add_markers",
                "inverse": {"markers": [{"time": 1, "comment": "Beat"}]},
            }),
            headers=self._h(),
        ).get_json()
        entry_id = created["id"]

        deleted = self.client.delete(f"/journal/{entry_id}", headers=self._h())
        self.assertEqual(deleted.status_code, 409)
        legacy_revert = self.client.post(
            f"/journal/mark-reverted/{entry_id}",
            data=json.dumps({}),
            headers=self._h(),
        )
        self.assertEqual(legacy_revert.status_code, 409)

        cleared = self.client.post(
            "/journal/clear", data=json.dumps({}), headers=self._h()
        )
        self.assertEqual(cleared.status_code, 200)
        self.assertEqual(cleared.get_json()["preserved_incomplete"], 1)
        self.assertEqual(self.client.get("/journal/recovery").get_json()["count"], 1)


if __name__ == "__main__":
    unittest.main()
