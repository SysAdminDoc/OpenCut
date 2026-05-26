"""
Tests for the structured check-failure registry + /system/check-failures
(RESEARCH_FEATURE_PLAN_2026-05-25 E5).

``opencut.checks`` historically swallowed every ``except Exception:``
in its 40+ availability probes, masking install failures as silent
503s downstream. The new registry captures exception type + message +
timestamp under the calling probe's ``__name__``; ``/system/check-
failures`` exposes the registry for support diagnostics.

These tests cover:
  - ``record_check_failure`` / ``_record_caller_failure`` shape.
  - ``get_check_failures`` is a thread-safe copy.
  - ``clear_check_failures`` empties the registry.
  - The new ``/system/check-failures`` route returns the registry.
"""
from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCheckFailureRegistry(unittest.TestCase):
    def setUp(self):
        from opencut import checks
        self.checks = checks
        checks.clear_check_failures()

    def test_record_failure_captures_type_and_message(self):
        try:
            raise RuntimeError("synthetic install failure")
        except RuntimeError as exc:
            self.checks.record_check_failure("check_foo", exc)
        failures = self.checks.get_check_failures()
        self.assertIn("check_foo", failures)
        self.assertEqual(failures["check_foo"]["exception"], "RuntimeError")
        self.assertEqual(failures["check_foo"]["message"], "synthetic install failure")
        self.assertIsInstance(failures["check_foo"]["ts"], float)

    def test_message_is_capped_to_500_chars(self):
        long_msg = "x" * 1000
        try:
            raise ValueError(long_msg)
        except ValueError as exc:
            self.checks.record_check_failure("check_long", exc)
        msg = self.checks.get_check_failures()["check_long"]["message"]
        self.assertEqual(len(msg), 500)

    def test_get_check_failures_returns_copy_not_live(self):
        try:
            raise OSError("boom")
        except OSError as exc:
            self.checks.record_check_failure("check_a", exc)
        snap = self.checks.get_check_failures()
        snap["check_a"]["exception"] = "MUTATED"
        # Subsequent get should not be affected.
        again = self.checks.get_check_failures()
        self.assertEqual(again["check_a"]["exception"], "OSError")

    def test_clear_check_failures_empties_registry(self):
        try:
            raise RuntimeError("z")
        except RuntimeError as exc:
            self.checks.record_check_failure("check_z", exc)
        self.assertTrue(self.checks.get_check_failures())
        self.checks.clear_check_failures()
        self.assertEqual(self.checks.get_check_failures(), {})

    def test_record_caller_failure_uses_calling_frame_name(self):
        def some_check_xyz():
            try:
                raise ImportError("missing-foo")
            except ImportError as exc:
                self.checks._record_caller_failure(exc)
        some_check_xyz()
        failures = self.checks.get_check_failures()
        self.assertIn("some_check_xyz", failures)
        self.assertEqual(failures["some_check_xyz"]["exception"], "ImportError")


class TestCheckFailuresRoute(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from opencut.server import create_app
        from opencut import checks
        cls.app = create_app()
        cls.client = cls.app.test_client()
        cls.checks = checks

    def setUp(self):
        self.checks.clear_check_failures()
        # Acquire CSRF token from /health.
        h = self.client.get("/health").get_json()
        self.token = h.get("csrf_token", "")

    def test_get_empty_registry(self):
        resp = self.client.get(
            "/system/check-failures",
            headers={"X-OpenCut-Token": self.token},
        )
        self.assertEqual(resp.status_code, 200)
        payload = json.loads(resp.data.decode("utf-8"))
        self.assertEqual(payload, {"failures": {}})

    def test_get_returns_recorded_failures(self):
        try:
            raise FileNotFoundError("the binary is missing")
        except FileNotFoundError as exc:
            self.checks.record_check_failure("check_thing", exc)
        resp = self.client.get(
            "/system/check-failures",
            headers={"X-OpenCut-Token": self.token},
        )
        payload = json.loads(resp.data.decode("utf-8"))
        self.assertIn("check_thing", payload["failures"])
        self.assertEqual(payload["failures"]["check_thing"]["exception"], "FileNotFoundError")

    def test_delete_clears_registry(self):
        try:
            raise OSError("blow")
        except OSError as exc:
            self.checks.record_check_failure("check_x", exc)
        resp = self.client.delete(
            "/system/check-failures",
            headers={"X-OpenCut-Token": self.token},
        )
        self.assertEqual(resp.status_code, 200)
        # Registry should now be empty.
        self.assertEqual(self.checks.get_check_failures(), {})


if __name__ == "__main__":
    unittest.main()
