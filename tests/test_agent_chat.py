"""
Tests for the F143 chat-conductor scaffold (RESEARCH_FEATURE_PLAN_2026-05-25 Q2).

Covers:
  * keyword fallback planner produces multi-step plans for intents that
    use conjunctions / plurals;
  * LLM JSON-extraction parser tolerates markdown fences + leading prose;
  * session persistence + mark_step_status round-trip;
  * heuristic self-review surfaces failed / rejected steps as drift;
  * five new routes (plan, step-status, review, session, info).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
import uuid
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestLLMPlanParser(unittest.TestCase):
    def test_parse_strict_json_array(self):
        from opencut.core.agent_chat import _parse_llm_plan
        text = '[{"label":"x","endpoint":"/silence","payload":{}}]'
        parsed = _parse_llm_plan(text)
        self.assertIsInstance(parsed, list)
        self.assertEqual(parsed[0]["endpoint"], "/silence")

    def test_parse_markdown_fenced_json(self):
        from opencut.core.agent_chat import _parse_llm_plan
        text = (
            "```json\n"
            '[{"label":"x","endpoint":"/silence","payload":{}}]\n'
            "```"
        )
        parsed = _parse_llm_plan(text)
        self.assertEqual(parsed[0]["endpoint"], "/silence")

    def test_parse_with_leading_prose(self):
        from opencut.core.agent_chat import _parse_llm_plan
        text = (
            "Sure, here's the plan:\n\n"
            '[{"label":"x","endpoint":"/silence","payload":{}}]\n\n'
            "Hope that helps!"
        )
        parsed = _parse_llm_plan(text)
        self.assertEqual(parsed[0]["endpoint"], "/silence")

    def test_parse_returns_none_on_nonsense(self):
        from opencut.core.agent_chat import _parse_llm_plan
        self.assertIsNone(_parse_llm_plan("not json at all"))
        self.assertIsNone(_parse_llm_plan(""))


class TestKeywordPlanner(unittest.TestCase):
    def test_two_step_intent(self):
        from opencut.core import agent_chat
        r = agent_chat.plan("Cut silences then add captions")
        self.assertEqual(r.source, "keyword")
        endpoints = [s.endpoint for s in r.plan]
        self.assertIn("/captions/silence", endpoints)
        self.assertIn("/captions/transcribe", endpoints)

    def test_plural_stripping_matches(self):
        from opencut.core import agent_chat
        # 'silences' should still match the 'silence' keyword.
        r = agent_chat.plan("Cut the silences")
        self.assertGreaterEqual(len(r.plan), 1)
        self.assertEqual(r.plan[0].endpoint, "/captions/silence")

    def test_unknown_intent_produces_empty_plan_with_note(self):
        from opencut.core import agent_chat
        r = agent_chat.plan("draw me a unicorn flying over Saturn")
        self.assertEqual(r.plan, [])
        self.assertTrue(any("no steps" in n.lower() or "specific" in n.lower() for n in r.notes))

    def test_empty_intent_raises(self):
        from opencut.core import agent_chat
        with self.assertRaises(ValueError):
            agent_chat.plan("")


class TestSessionPersistence(unittest.TestCase):
    def setUp(self):
        # Redirect the sessions dir to a tempdir so we don't leak across runs.
        self.tmp = tempfile.mkdtemp(prefix="agent_chat_test_")
        from opencut.core import agent_chat
        self.mod = agent_chat
        self._orig_session_dir = agent_chat._session_dir
        agent_chat._session_dir = lambda: self.tmp

    def tearDown(self):
        self.mod._session_dir = self._orig_session_dir
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_plan_round_trips_through_load_session(self):
        r = self.mod.plan("Cut silences", session_id="abc123")
        sess = self.mod.load_session("abc123")
        self.assertIsNotNone(sess)
        self.assertEqual(sess["intent"], "Cut silences")
        self.assertEqual(len(sess["plan"]), len(r.plan))

    def test_load_session_unknown_returns_none(self):
        self.assertIsNone(self.mod.load_session("never_existed"))

    def test_mark_step_status_updates_session(self):
        r = self.mod.plan("Cut silences", session_id="markstep")
        sid = r.plan[0].step_id
        sess = self.mod.mark_step_status("markstep", sid, "ok", job_id="job-1")
        statuses = {s["step_id"]: s["status"] for s in sess["plan"]}
        self.assertEqual(statuses[sid], "ok")
        # executed_steps appends a log entry.
        self.assertEqual(sess["executed_steps"][-1]["step_id"], sid)
        self.assertEqual(sess["executed_steps"][-1]["job_id"], "job-1")

    def test_mark_step_status_rejects_bad_status(self):
        r = self.mod.plan("Cut silences", session_id="badstatus")
        with self.assertRaises(ValueError):
            self.mod.mark_step_status("badstatus", r.plan[0].step_id, "not_a_status")

    def test_mark_step_status_unknown_session_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.mod.mark_step_status("nope", "x", "ok")

    def test_session_id_rejects_path_traversal(self):
        with self.assertRaises(ValueError):
            self.mod._session_path("../../etc/passwd")


class TestSelfReview(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="agent_chat_review_")
        from opencut.core import agent_chat
        self.mod = agent_chat
        self._orig = agent_chat._session_dir
        agent_chat._session_dir = lambda: self.tmp

    def tearDown(self):
        self.mod._session_dir = self._orig
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_heuristic_review_clean_plan_matches(self):
        r = self.mod.plan("Cut silences", session_id="rev1")
        sid = r.plan[0].step_id
        self.mod.mark_step_status("rev1", sid, "ok")
        review = self.mod.review("rev1")
        self.assertTrue(review.matched)
        self.assertEqual(review.source, "heuristic")

    def test_heuristic_review_failed_step_drifts(self):
        r = self.mod.plan("Cut silences then add captions", session_id="rev2")
        sid = r.plan[0].step_id
        self.mod.mark_step_status("rev2", sid, "failed", reason="missing dep")
        review = self.mod.review("rev2")
        self.assertFalse(review.matched)
        self.assertTrue(any("failed" in note.lower() for note in review.drift_notes))

    def test_review_unknown_session_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.mod.review("never_existed")


class TestAgentChatRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from opencut.server import create_app
        cls.app = create_app()
        cls.client = cls.app.test_client()
        cls.token = cls.client.get("/health").get_json().get("csrf_token", "")
        # Sandbox sessions to a tempdir to avoid leaking into ~/.opencut.
        from opencut.core import agent_chat
        cls.mod = agent_chat
        cls.tmp = tempfile.mkdtemp(prefix="agent_chat_route_")
        cls._orig = agent_chat._session_dir
        agent_chat._session_dir = lambda: cls.tmp

    @classmethod
    def tearDownClass(cls):
        cls.mod._session_dir = cls._orig
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_info_endpoint(self):
        resp = self.client.get("/agent/chat/info")
        self.assertEqual(resp.status_code, 200)
        body = json.loads(resp.data.decode("utf-8"))
        self.assertTrue(body["available"])
        self.assertIn("plan", body["modes"])

    def test_plan_endpoint(self):
        resp = self.client.post(
            "/agent/chat/plan",
            json={"intent": "Cut silences then add captions"},
            headers={"X-OpenCut-Token": self.token},
        )
        self.assertEqual(resp.status_code, 200, resp.data)
        body = json.loads(resp.data.decode("utf-8"))
        self.assertEqual(body["source"], "keyword")
        self.assertTrue(body["session_id"])
        self.assertGreaterEqual(len(body["plan"]), 1)

    def test_plan_rejects_empty_intent(self):
        resp = self.client.post(
            "/agent/chat/plan",
            json={"intent": "   "},
            headers={"X-OpenCut-Token": self.token},
        )
        self.assertEqual(resp.status_code, 400)

    def test_step_status_round_trip(self):
        plan_resp = self.client.post(
            "/agent/chat/plan",
            json={"intent": "Cut silences", "session_id": "route-sess-1"},
            headers={"X-OpenCut-Token": self.token},
        ).get_json()
        step_id = plan_resp["plan"][0]["step_id"]
        resp = self.client.post(
            "/agent/chat/step-status",
            json={"session_id": "route-sess-1", "step_id": step_id, "status": "ok"},
            headers={"X-OpenCut-Token": self.token},
        )
        self.assertEqual(resp.status_code, 200)
        body = json.loads(resp.data.decode("utf-8"))
        self.assertTrue(body["ok"])

    def test_step_status_unknown_session_404(self):
        resp = self.client.post(
            "/agent/chat/step-status",
            json={"session_id": "ghost", "step_id": "abc", "status": "ok"},
            headers={"X-OpenCut-Token": self.token},
        )
        self.assertEqual(resp.status_code, 404)

    def test_review_endpoint(self):
        self.client.post(
            "/agent/chat/plan",
            json={"intent": "Cut silences", "session_id": "rev-route"},
            headers={"X-OpenCut-Token": self.token},
        )
        resp = self.client.post(
            "/agent/chat/review",
            json={"session_id": "rev-route"},
            headers={"X-OpenCut-Token": self.token},
        )
        self.assertEqual(resp.status_code, 200)
        body = json.loads(resp.data.decode("utf-8"))
        self.assertIn("matched", body)
        self.assertIn("drift_notes", body)

    def test_session_get(self):
        self.client.post(
            "/agent/chat/plan",
            json={"intent": "Cut silences", "session_id": "sess-get"},
            headers={"X-OpenCut-Token": self.token},
        )
        resp = self.client.get("/agent/chat/session?session_id=sess-get")
        self.assertEqual(resp.status_code, 200)
        body = json.loads(resp.data.decode("utf-8"))
        self.assertEqual(body["session_id"], "sess-get")

    def test_session_get_unknown_404(self):
        resp = self.client.get("/agent/chat/session?session_id=never_existed")
        self.assertEqual(resp.status_code, 404)


if __name__ == "__main__":
    unittest.main()
