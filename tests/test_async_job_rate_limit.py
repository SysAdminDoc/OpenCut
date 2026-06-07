from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path
import re

from flask import Flask


ROUTE_RATE_LIMIT_CALL_RE = re.compile(r"\brate_limit(?:_release)?\s*\(")


def _inline_pool():
    class InlinePool:
        def submit(self, job_id, fn):
            future = Future()
            try:
                fn()
            except BaseException as exc:  # noqa: BLE001
                future.set_exception(exc)
            else:
                future.set_result(None)
            return future

    return InlinePool()


def _make_rate_limited_app(rate_limit_key):
    from opencut.jobs import async_job

    app = Flask(__name__)

    @app.route("/rate-limited-job", methods=["POST"])
    @async_job("rate-limited-job", filepath_required=False, rate_limit_key=rate_limit_key)
    def rate_limited_job(job_id, filepath, data):
        return {"ok": True}

    return app


def test_async_job_rate_limit_rejects_before_job_creation(monkeypatch):
    import opencut.jobs as jobs_mod
    from opencut.security import rate_limit, rate_limit_release

    key = "test_async_job_rejects_before_creation"
    app = _make_rate_limited_app(key)

    def fail_new_job(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("_new_job should not run when the rate limit is saturated")

    monkeypatch.setattr(jobs_mod, "_new_job", fail_new_job)
    assert rate_limit(key) is True
    try:
        response = app.test_client().post("/rate-limited-job", json={"mode": "locked"})
    finally:
        rate_limit_release(key)

    assert response.status_code == 429
    assert response.get_json()["code"] == "RATE_LIMITED"


def test_async_job_rate_limit_releases_after_worker_completion(monkeypatch):
    from opencut.security import rate_limit, rate_limit_release

    key = "test_async_job_releases_after_worker"
    app = _make_rate_limited_app(key)
    monkeypatch.setattr("opencut.workers.get_pool", lambda: _inline_pool())

    response = app.test_client().post("/rate-limited-job", json={"mode": "complete"})

    assert response.status_code == 200
    assert response.get_json()["job_id"]
    assert rate_limit(key) is True
    rate_limit_release(key)


def test_async_job_rate_limit_releases_when_job_creation_fails(monkeypatch):
    import opencut.jobs as jobs_mod
    from opencut.jobs import TooManyJobsError
    from opencut.security import rate_limit, rate_limit_release

    key = "test_async_job_releases_after_too_many"
    app = _make_rate_limited_app(key)

    def too_many_jobs(*args, **kwargs):  # noqa: ANN002, ANN003
        raise TooManyJobsError("too many jobs")

    monkeypatch.setattr(jobs_mod, "_new_job", too_many_jobs)

    response = app.test_client().post("/rate-limited-job", json={"mode": "full"})

    assert response.status_code == 429
    assert response.get_json()["code"] == "TOO_MANY_JOBS"
    assert rate_limit(key) is True
    rate_limit_release(key)


def test_async_job_rate_limit_callable_can_skip_limit(monkeypatch):
    from opencut.security import rate_limit, rate_limit_release

    key = "test_async_job_conditional_key"
    app = _make_rate_limited_app(lambda data: key if data.get("locked") else None)
    monkeypatch.setattr("opencut.workers.get_pool", lambda: _inline_pool())

    assert rate_limit(key) is True
    try:
        response = app.test_client().post("/rate-limited-job", json={"locked": False})
    finally:
        rate_limit_release(key)

    assert response.status_code == 200
    assert response.get_json()["job_id"]


def test_route_modules_do_not_call_rate_limit_primitives_directly():
    repo_root = Path(__file__).resolve().parents[1]
    route_dir = repo_root / "opencut" / "routes"
    offenders = []
    for path in sorted(route_dir.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for match in ROUTE_RATE_LIMIT_CALL_RE.finditer(text):
            line_number = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{path.relative_to(repo_root)}:{line_number}: {match.group(0)}")

    assert not offenders, (
        "Route modules should use async_job(rate_limit_key=...), "
        "require_rate_limit(...), or rate_limit_slot(...) instead of "
        f"manual rate-limit primitive calls: {offenders}"
    )
