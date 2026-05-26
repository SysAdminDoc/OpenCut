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

    def test_drift_score_decreases_per_failure(self):
        """Heuristic score must drop 25 per failed step, 15 per rejected."""
        r = self.mod.plan("Cut silences then add captions", session_id="dscore1")
        # Two-step plan; mark both as failed.
        for s in r.plan:
            self.mod.mark_step_status("dscore1", s.step_id, "failed")
        review = self.mod.review("dscore1")
        self.assertFalse(review.matched)
        # 100 - 25 * 2 = 50.
        self.assertEqual(review.drift_score, 50)

    def test_drift_score_clean_plan_is_100(self):
        r = self.mod.plan("Cut silences", session_id="dscore_clean")
        for s in r.plan:
            self.mod.mark_step_status("dscore_clean", s.step_id, "ok")
        review = self.mod.review("dscore_clean")
        self.assertEqual(review.drift_score, 100)
        self.assertTrue(review.matched)

    def test_drift_score_floored_at_zero(self):
        """Many failures must clamp the score at 0, not go negative."""
        # 5 fragments → 5 keyword candidates, but plan dedupes by route.
        r = self.mod.plan(
            "Cut silences and denoise audio and add captions and zoom in and color match",
            session_id="dscore_floor",
        )
        for s in r.plan:
            self.mod.mark_step_status("dscore_floor", s.step_id, "failed")
        review = self.mod.review("dscore_floor")
        self.assertGreaterEqual(review.drift_score, 0)
        self.assertEqual(review.drift_score, max(0, 100 - 25 * len(r.plan)))

    def test_review_result_to_dict_contains_new_fields(self):
        r = self.mod.plan("Cut silences", session_id="newfields")
        review = self.mod.review("newfields")
        d = review.to_dict()
        self.assertIn("drift_score", d)
        self.assertIn("suggested_retry", d)
        # heuristic path → suggested_retry is None.
        self.assertIsNone(d["suggested_retry"])


class TestStructuredReviewParser(unittest.TestCase):
    """The structured-JSON parser must tolerate the same cruft as the
    plan parser (markdown fences, leading prose, etc.)."""

    def setUp(self):
        from opencut.core import agent_chat
        self.mod = agent_chat

    def test_parse_strict_json_object(self):
        text = (
            '{"matched": false, "drift_score": 45, '
            '"drift_notes": ["caption step skipped"], "suggested_retry": null}'
        )
        parsed = self.mod._parse_structured_review(text)
        self.assertIsInstance(parsed, dict)
        self.assertFalse(parsed["matched"])
        self.assertEqual(parsed["drift_score"], 45)

    def test_parse_markdown_fenced_object(self):
        text = (
            "```json\n"
            '{"matched": true, "drift_score": 100, '
            '"drift_notes": [], "suggested_retry": null}\n'
            "```"
        )
        parsed = self.mod._parse_structured_review(text)
        self.assertTrue(parsed["matched"])

    def test_parse_with_leading_prose(self):
        text = (
            "Here's the review:\n\n"
            '{"matched": false, "drift_score": 60, '
            '"drift_notes": ["x"], "suggested_retry": null}'
        )
        parsed = self.mod._parse_structured_review(text)
        self.assertEqual(parsed["drift_score"], 60)

    def test_parse_returns_none_on_nonsense(self):
        self.assertIsNone(self.mod._parse_structured_review("not json"))
        self.assertIsNone(self.mod._parse_structured_review(""))


class TestSuggestedRetryCoercion(unittest.TestCase):
    def setUp(self):
        from opencut.core import agent_chat
        self.mod = agent_chat

    def test_coerce_well_formed_retry(self):
        retry = self.mod._coerce_suggested_retry({
            "label": "Add captions",
            "endpoint": "/captions/transcribe",
            "payload": {"model": "base"},
            "rationale": "user asked for captions, none were added",
        })
        self.assertIsNotNone(retry)
        self.assertEqual(retry["endpoint"], "/captions/transcribe")

    def test_coerce_endpoint_missing_leading_slash(self):
        retry = self.mod._coerce_suggested_retry({
            "label": "x",
            "endpoint": "captions/transcribe",
            "payload": {},
        })
        self.assertEqual(retry["endpoint"], "/captions/transcribe")

    def test_coerce_rejects_non_dict(self):
        self.assertIsNone(self.mod._coerce_suggested_retry(None))
        self.assertIsNone(self.mod._coerce_suggested_retry("not a dict"))
        self.assertIsNone(self.mod._coerce_suggested_retry([]))

    def test_coerce_rejects_missing_endpoint(self):
        self.assertIsNone(self.mod._coerce_suggested_retry({"label": "x"}))
        self.assertIsNone(self.mod._coerce_suggested_retry({"endpoint": ""}))

    def test_coerce_rationale_truncated(self):
        long = "x" * 500
        retry = self.mod._coerce_suggested_retry({
            "label": "x",
            "endpoint": "/foo",
            "rationale": long,
        })
        self.assertEqual(len(retry["rationale"]), 240)


class TestStructuredReviewLLMPath(unittest.TestCase):
    """End-to-end: mock query_llm so we can drive the structured branch
    deterministically without a real LLM."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="agent_chat_struct_")
        from opencut.core import agent_chat
        self.mod = agent_chat
        self._orig = agent_chat._session_dir
        agent_chat._session_dir = lambda: self.tmp

    def tearDown(self):
        self.mod._session_dir = self._orig
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _fake_llm_resp(self, payload_json: str):
        class _Resp:
            def __init__(self, text):
                self.text = text
        return _Resp(payload_json)

    def test_structured_response_sets_source_and_score(self):
        r = self.mod.plan("Cut silences and add captions", session_id="struct1")
        for s in r.plan:
            self.mod.mark_step_status("struct1", s.step_id, "ok")
        llm_config = {"provider": "ollama", "model": "fake"}  # value not used
        fake_text = json.dumps({
            "matched": False,
            "drift_score": 70,
            "drift_notes": ["captions were transcribed in English but user wanted Spanish"],
            "suggested_retry": {
                "label": "Translate captions to Spanish",
                "endpoint": "/captions/translate",
                "payload": {"target_lang": "es"},
                "rationale": "intent specified Spanish translation",
            },
        })
        with patch("opencut.core.llm.query_llm", return_value=self._fake_llm_resp(fake_text)):
            review = self.mod.review("struct1", llm_config=llm_config)
        self.assertEqual(review.source, "llm_structured")
        self.assertEqual(review.drift_score, 70)
        self.assertFalse(review.matched)
        self.assertIsNotNone(review.suggested_retry)
        self.assertEqual(review.suggested_retry["endpoint"], "/captions/translate")

    def test_appended_retry_added_to_session_plan(self):
        r = self.mod.plan("Cut silences", session_id="struct_append")
        sid = r.plan[0].step_id
        self.mod.mark_step_status("struct_append", sid, "failed")
        llm_config = {"provider": "ollama", "model": "fake"}
        fake_text = json.dumps({
            "matched": False,
            "drift_score": 40,
            "drift_notes": ["retry silence detection with stricter threshold"],
            "suggested_retry": {
                "label": "Retry silence detection",
                "endpoint": "/silence",
                "payload": {"threshold_db": -35},
                "rationale": "previous threshold was too lenient",
            },
        })
        with patch("opencut.core.llm.query_llm", return_value=self._fake_llm_resp(fake_text)):
            self.mod.review("struct_append", llm_config=llm_config, append_retry=True)
        sess = self.mod.load_session("struct_append")
        # Plan grew from 1 to 2 steps.
        self.assertEqual(len(sess["plan"]), 2)
        new_step = sess["plan"][-1]
        self.assertEqual(new_step["endpoint"], "/silence")
        self.assertEqual(new_step["status"], "planned")
        self.assertIn("F144 self-review", new_step["reason"])

    def test_append_retry_false_does_not_grow_plan(self):
        r = self.mod.plan("Cut silences", session_id="struct_noappend")
        sid = r.plan[0].step_id
        self.mod.mark_step_status("struct_noappend", sid, "failed")
        llm_config = {"provider": "ollama", "model": "fake"}
        fake_text = json.dumps({
            "matched": False, "drift_score": 30,
            "drift_notes": ["x"],
            "suggested_retry": {"label": "y", "endpoint": "/silence", "payload": {}, "rationale": ""},
        })
        with patch("opencut.core.llm.query_llm", return_value=self._fake_llm_resp(fake_text)):
            self.mod.review("struct_noappend", llm_config=llm_config, append_retry=False)
        sess = self.mod.load_session("struct_noappend")
        self.assertEqual(len(sess["plan"]), 1)

    def test_structured_parse_failure_falls_back_to_free_text(self):
        r = self.mod.plan("Cut silences", session_id="struct_fallback")
        for s in r.plan:
            self.mod.mark_step_status("struct_fallback", s.step_id, "ok")
        llm_config = {"provider": "ollama", "model": "fake"}
        with patch("opencut.core.llm.query_llm", return_value=self._fake_llm_resp("not parseable JSON")):
            review = self.mod.review("struct_fallback", llm_config=llm_config)
        # Source falls through structured -> llm (free text).
        self.assertIn(review.source, ("llm", "heuristic"))


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
