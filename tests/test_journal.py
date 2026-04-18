"""Tests for the Operation Journal (v1.9.28)."""

import json
import os
import tempfile
import threading
import unittest
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


if __name__ == "__main__":
    unittest.main()
