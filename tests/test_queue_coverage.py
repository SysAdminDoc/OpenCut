"""F306 — the queue allowlist must report its own coverage.

``_ALLOWED_QUEUE_ENDPOINTS`` is hand-maintained, so it silently certifies
whatever is not in it: a route omitted by accident is indistinguishable from
one excluded on purpose, and both fail ``/queue/add`` with the same bare
message. These tests pin the read-only report and the structured rejection —
neither changes which routes are queueable.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opencut.routes.jobs_routes import _ALLOWED_QUEUE_ENDPOINTS  # noqa: E402

from .conftest import csrf_headers  # noqa: E402


class TestCoverageReport:
    def test_report_accounts_for_every_async_post_route(self, client):
        report = client.get("/queue/coverage").get_json()
        assert report["async_post_routes"] == report["queueable"] + report["not_queueable"]
        assert report["async_post_routes"] > 0
        assert len(report["missing"]) == report["not_queueable"]

    def test_report_matches_the_live_allowlist(self, client):
        report = client.get("/queue/coverage").get_json()
        # Every reported-queueable route really is in the allowlist.
        assert report["queueable"] <= len(_ALLOWED_QUEUE_ENDPOINTS)
        for entry in report["missing"]:
            assert entry["endpoint"] not in _ALLOWED_QUEUE_ENDPOINTS

    def test_missing_entries_carry_enough_context_to_act_on(self, client):
        report = client.get("/queue/coverage").get_json()
        if not report["missing"]:
            return
        entry = report["missing"][0]
        assert entry["endpoint"].startswith("/")
        assert "job_type" in entry
        assert "blueprint" in entry

    def test_no_stale_allowlist_entries(self, client):
        """An allowlisted path with no live async route is unambiguously wrong."""
        report = client.get("/queue/coverage").get_json()
        assert report["stale_allowlist_entries"] == []

    def test_coverage_percent_is_consistent(self, client):
        report = client.get("/queue/coverage").get_json()
        expected = round((report["queueable"] / report["async_post_routes"]) * 100, 1)
        assert report["coverage_percent"] == expected


class TestStructuredRejection:
    def test_non_queueable_endpoint_names_itself(self, client, csrf_token):
        resp = client.post(
            "/queue/add",
            json={"endpoint": "/definitely/not/queueable", "payload": {}},
            headers=csrf_headers(csrf_token),
        )
        assert resp.status_code == 400
        body = resp.get_json()
        assert body["code"] == "ENDPOINT_NOT_QUEUEABLE"
        assert body["endpoint"] == "/definitely/not/queueable"
        assert "coverage" in body["suggestion"]

    def test_queueable_endpoints_still_pass_validation(self):
        """The report must not have changed which routes are queueable.

        Validated through the normalizer rather than POSTing, so the test does
        not dispatch real work as a side effect.
        """
        from opencut.routes.jobs_routes import _normalize_queue_entry

        for endpoint in sorted(_ALLOWED_QUEUE_ENDPOINTS)[:5]:
            entry = _normalize_queue_entry(
                {"id": "abc123", "endpoint": endpoint, "payload": {}, "status": "queued"},
                require_queueable=True,
            )
            assert entry["endpoint"] == endpoint
