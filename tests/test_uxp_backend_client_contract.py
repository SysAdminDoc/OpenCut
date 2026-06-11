"""F217 — UXP BackendClient HTTP-shape contract tests.

The UXP panel's `BackendClient` (in `extension/com.opencut.uxp/main.js`)
talks to the Python backend over HTTP. The panel's job-running flow
depends on a specific response shape:

* CSRF token comes from ``GET /health`` under the ``csrf_token`` field.
* Mutating requests carry ``X-OpenCut-Token`` and re-fetch CSRF on 403
  exactly once before surfacing the error.
* Async job submissions return ``{job_id: "..."}`` (or ``{id: ...}``).
* Job status polling is at ``GET /status/<job_id>`` and returns
  ``{status: "complete"|"error"|"cancelled"|"running", ...}``.
* The wrapper times out at 120s and surfaces the timeout as a normal
  ``{ok: false}`` result, not an exception.
* Refreshed CSRF tokens in response headers (``X-OpenCut-Token``)
  must be honoured so the next mutating call doesn't 403.

These tests pin the **JS-side** contract by static-analysing the
panel source AND pin the **server-side** contract by exercising
the corresponding Flask routes. If either side drifts, the test
fires before a release.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_JS = REPO_ROOT / "extension" / "com.opencut.uxp" / "main.js"


def _read_main_js() -> str:
    return MAIN_JS.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# JS-side contract — static analysis of main.js
# ---------------------------------------------------------------------------


def test_backend_client_module_present():
    text = _read_main_js()
    assert "const BackendClient = " in text, (
        "UXP main.js must expose a BackendClient module"
    )


def test_backend_client_exports_canonical_verbs():
    text = _read_main_js()
    # Find the BackendClient IIFE body and pull its closing `return { ... }`.
    iife_start = text.find("const BackendClient")
    assert iife_start >= 0, "BackendClient declaration missing"
    body = text[iife_start: iife_start + 6000]
    match = re.search(
        r"return\s+\{\s*([^}]+)\s*\}\s*;\s*\}\s*\)\s*\(\s*\)\s*;",
        body,
    )
    assert match, "BackendClient closure must end with a `return { ... }`"
    exported = match.group(1)
    for verb in ("call", "get", "post", "del", "checkHealth", "fetchCsrf"):
        assert verb in exported, (
            f"BackendClient must export {verb!r}; current export block: {exported.strip()}"
        )


def test_backend_client_uses_x_opencut_token_header():
    text = _read_main_js()
    assert '"X-OpenCut-Token"' in text or "'X-OpenCut-Token'" in text, (
        "BackendClient must send the X-OpenCut-Token CSRF header"
    )
    # On 403 the wrapper must refresh CSRF and retry exactly once.
    assert re.search(
        r"resp\.status\s*===\s*403", text,
    ), "BackendClient must detect 403 status to trigger CSRF refresh"


def test_backend_client_carries_120s_timeout():
    text = _read_main_js()
    # 120s = 120000ms — must be present so a hung backend can't pin the
    # button forever.
    assert "120000" in text, (
        "BackendClient must carry a 120-second fetch timeout (120000 ms)"
    )


def test_backend_client_refreshes_csrf_token_from_response_header():
    text = _read_main_js()
    assert 'resp.headers.get("X-OpenCut-Token")' in text \
        or "resp.headers.get('X-OpenCut-Token')" in text, (
        "BackendClient must refresh csrfToken from response headers"
    )


def test_backend_client_returns_ok_failure_shape():
    """The wrapper must return ``{ok, data, error, status}`` objects so
    callers can branch on `r.ok` without try/catch nesting."""
    text = _read_main_js()
    # Search for "return { ok: false" — the failure path must surface
    # this shape.
    assert re.search(r"return\s*\{\s*ok:\s*false", text), (
        "BackendClient must return {ok: false, ...} on failure"
    )
    assert re.search(r"return\s*\{\s*ok:\s*true", text), (
        "BackendClient must return {ok: true, ...} on success"
    )


def test_backend_client_surfaces_timeout_as_normal_result():
    """A timeout must produce ``ok: false`` with an actionable error
    message, not an unhandled rejection."""
    text = _read_main_js()
    assert "timed out" in text.lower(), (
        "BackendClient must include 'timed out' guidance in its error string"
    )
    assert "OpenCut Server is still running" in text, (
        "BackendClient timeout message must point users at the server status"
    )


def test_job_poller_uses_status_endpoint():
    """JobPoller polls ``/status/<job_id>``; if the route changes this test
    fires before the panel breaks."""
    text = _read_main_js()
    assert re.search(r"`/status/\$\{[a-zA-Z_]+\}`", text), (
        "JobPoller must poll /status/<job_id>"
    )


def test_job_id_is_pulled_from_canonical_field():
    """Async job submissions return ``{job_id: "..."}`` (canonical) with
    a legacy ``{id: ...}`` fallback. The poller must accept either."""
    text = _read_main_js()
    assert "job_id" in text, "JobPoller must read r.data.job_id from the response"
    assert re.search(r"r\.data\?\.job_id\s*\?\?\s*r\.data\?\.id", text), (
        "JobPoller must accept either {job_id} or legacy {id} as the job identifier"
    )


def test_terminal_job_statuses_are_complete_error_cancelled():
    """JobPoller branches on the canonical terminal statuses. If the
    backend renames any of these, the poller will spin forever."""
    text = _read_main_js()
    for status in ("complete", "error", "cancelled"):
        assert f'"{status}"' in text or f"'{status}'" in text, (
            f"JobPoller must recognise terminal job status {status!r}"
        )


def test_job_cancel_clears_button_loading_state():
    text = _read_main_js()
    assert "function clearButtonLoadingStates()" in text
    assert 'document.querySelectorAll("button.loading")' in text
    assert "UIController.clearButtonLoadingStates();" in text
    assert re.search(r"async function cancel\(\).*?return true;", text, re.S), (
        "JobPoller.cancel() must report whether it actually cancelled a job"
    )


# ---------------------------------------------------------------------------
# Server-side contract — exercise the routes the UXP panel calls
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    from opencut.server import _get_app

    app = _get_app()
    app.config["TESTING"] = True
    return app.test_client()


def _csrf(client) -> str:
    return client.get("/health").get_json().get("csrf_token", "")


def test_health_route_returns_csrf_token(client):
    """The UXP BackendClient bootstraps CSRF by GETting /health and
    reading `csrf_token` out of the JSON body."""
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert "csrf_token" in body, (
        "GET /health must return a `csrf_token` field for the UXP panel"
    )
    assert isinstance(body["csrf_token"], str)
    assert body["csrf_token"], "csrf_token must be non-empty"


def test_health_route_does_not_cors_allow_null_origin_by_default(client):
    resp = client.get("/health", headers={"Origin": "null"})
    body = resp.get_json()

    assert resp.status_code == 200
    assert resp.headers.get("Access-Control-Allow-Origin") != "null"
    assert "csrf_token" not in body


def test_health_route_does_not_emit_csrf_to_file_or_null_origins():
    from opencut.config import OpenCutConfig
    from opencut.server import create_app

    app = create_app(OpenCutConfig(cors_origins=["null", "file://", "http://localhost:3000"]))
    app.config["TESTING"] = True
    test_client = app.test_client()

    for origin in ("null", "file://"):
        resp = test_client.get("/health", headers={"Origin": origin})
        body = resp.get_json()
        assert resp.status_code == 200
        assert "csrf_token" not in body

    resp = test_client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert resp.status_code == 200
    assert resp.get_json().get("csrf_token")


def test_status_route_returns_canonical_status_field(client):
    """The UXP JobPoller branches on `status in (complete, error, cancelled, running)`.
    A 404 for an unknown job is acceptable as long as the shape is JSON.
    """
    resp = client.get("/status/does-not-exist-9999")
    # Backend might 200 with status="not_found" or 404 — either is fine
    # as long as the response is JSON.
    body = resp.get_json()
    assert body is not None, "/status/<id> must always return JSON"


def test_mutating_route_requires_csrf(client):
    """The UXP wrapper auto-attaches X-OpenCut-Token on mutating calls.
    The backend must reject a missing token with 403 (so the wrapper's
    auto-refresh-on-403 path fires)."""
    resp = client.post("/cancel-all", data=json.dumps({}),
                       headers={"Content-Type": "application/json"})
    # 403 (missing token) or 401 (auth required) — both trigger the
    # UXP refresh path. Any 2xx would be a CSRF bypass.
    assert resp.status_code in (401, 403), (
        f"Mutating routes must require CSRF; got {resp.status_code}"
    )


def test_health_route_does_not_require_csrf(client):
    """GET /health must work without a token so the panel can bootstrap."""
    resp = client.get("/health")
    assert resp.status_code == 200


def test_capabilities_field_is_dict_when_present(client):
    """UXP's `_updateCapabilityHints` does `_capabilities[key]` lookups,
    so the field must be an object (not a list or string)."""
    body = client.get("/health").get_json()
    if "capabilities" in body:
        assert isinstance(body["capabilities"], dict), (
            "/health capabilities field must be a JSON object so UXP can "
            "index it by feature name"
        )
