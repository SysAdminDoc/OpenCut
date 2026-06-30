import ast
import threading
import time
from concurrent.futures import Future
from pathlib import Path

import pytest
from flask import Flask


@pytest.fixture
def isolated_job_store(tmp_path, monkeypatch):
    import opencut.jobs as jobs_mod
    import opencut.job_store as store

    store.close_all_connections()
    monkeypatch.setattr(store, "_DB_PATH", str(tmp_path / "jobs.db"))
    store._INITIALIZED = False
    store._INITIALIZED_PATH = None
    store._LOCAL = threading.local()
    store._ALL_CONNECTIONS = {}

    def sync_persist(job_dict, *, sync=False):
        store.save_job(job_dict)

    monkeypatch.setattr(jobs_mod, "_persist_job", sync_persist)
    yield store
    store.close_all_connections()
    store._INITIALIZED = False
    store._INITIALIZED_PATH = None


def _headers():
    from opencut.security import get_csrf_token

    return {
        "Content-Type": "application/json",
        "X-OpenCut-Token": get_csrf_token(),
    }


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


def _deferred_pool():
    class DeferredPool:
        def submit(self, job_id, fn):
            return Future()

    return DeferredPool()


def _make_resume_app():
    from opencut.jobs import async_job
    from opencut.routes.jobs_routes import jobs_bp
    from opencut.security import require_csrf

    app = Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(jobs_bp)

    @app.route("/resume-fixture", methods=["POST"])
    @require_csrf
    @async_job("resume-fixture", filepath_required=False, resumable=True)
    def resume_fixture(job_id, filepath, data):
        return {"payload": data, "filepath": filepath}

    @app.route("/non-resumable-fixture", methods=["POST"])
    @require_csrf
    @async_job("non-resume-fixture", filepath_required=False)
    def non_resume_fixture(job_id, filepath, data):
        return {"payload": data, "filepath": filepath}

    return app


def _async_job_decorators(source):
    decorators_by_type = {}
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            if not isinstance(decorator.func, ast.Name) or decorator.func.id != "async_job":
                continue
            if not decorator.args:
                continue
            job_type = decorator.args[0]
            if not isinstance(job_type, ast.Constant) or not isinstance(job_type.value, str):
                continue

            kwargs = {}
            for keyword in decorator.keywords:
                if keyword.arg is None:
                    continue
                try:
                    kwargs[keyword.arg] = ast.literal_eval(keyword.value)
                except ValueError:
                    kwargs[keyword.arg] = None
            decorators_by_type.setdefault(job_type.value, []).append(kwargs)

    return decorators_by_type


def test_job_store_migrates_and_round_trips_resume_metadata(isolated_job_store):
    isolated_job_store.init_db()
    columns = {
        row[1]
        for row in isolated_job_store._get_conn().execute("PRAGMA table_info(jobs)").fetchall()
    }

    assert "resumable" in columns
    assert "partial_output_path" in columns
    assert "resume_source_job_id" in columns
    assert "resume_attempt" in columns

    isolated_job_store.save_job({
        "id": "resume-meta",
        "type": "captions",
        "filepath": "/tmp/in.mov",
        "status": "interrupted",
        "progress": 42,
        "created": time.time(),
        "_endpoint": "/captions",
        "_payload": {"filepath": "/tmp/in.mov"},
        "resumable": True,
        "partial_output_path": "/tmp/in.partial",
        "resume_source_job_id": "old-job",
        "resume_attempt": 2,
    })

    job = isolated_job_store.get_job("resume-meta")

    assert job["resumable"] is True
    assert job["partial_output_path"] == "/tmp/in.partial"
    assert job["resume_source_job_id"] == "old-job"
    assert job["resume_attempt"] == 2
    assert job["endpoint"] == "/captions"
    assert job["payload"] == {"filepath": "/tmp/in.mov"}


def test_async_job_persists_running_resume_metadata(isolated_job_store, monkeypatch, tmp_path):
    from opencut.jobs import jobs, job_lock

    monkeypatch.setattr("opencut.workers.get_pool", lambda: _deferred_pool())
    with job_lock:
        jobs.clear()

    partial = str(tmp_path / "partial.json")
    app = _make_resume_app()
    response = app.test_client().post(
        "/resume-fixture",
        json={
            "partial_output_path": partial,
            "resume_source_job_id": "interrupted-1",
            "resume_attempt": "2",
        },
        headers=_headers(),
    )

    body = response.get_json()
    assert response.status_code == 200
    job = isolated_job_store.get_job(body["job_id"])
    assert job["status"] == "running"
    assert job["resumable"] is True
    assert job["partial_output_path"] == partial
    assert job["resume_source_job_id"] == "interrupted-1"
    assert job["resume_attempt"] == 2
    assert job["endpoint"] == "/resume-fixture"


def test_resume_interrupted_job_dispatches_new_resumable_job(isolated_job_store, monkeypatch, tmp_path):
    from opencut.jobs import jobs, job_lock

    monkeypatch.setattr("opencut.workers.get_pool", lambda: _inline_pool())
    with job_lock:
        jobs.clear()

    partial = str(tmp_path / "captions.partial.json")
    isolated_job_store.save_job({
        "id": "interrupted-1",
        "type": "resume-fixture",
        "status": "interrupted",
        "created": time.time(),
        "_endpoint": "/resume-fixture",
        "_payload": {"custom": "value"},
        "resumable": True,
        "partial_output_path": partial,
    })

    app = _make_resume_app()
    response = app.test_client().post("/jobs/interrupted-1/resume", headers=_headers())
    body = response.get_json()

    assert response.status_code == 202
    assert body["success"] is True
    assert body["source_job_id"] == "interrupted-1"
    assert body["partial_output_path"] == partial
    assert body["resume_attempt"] == 1
    assert body["job_id"] != "interrupted-1"

    with job_lock:
        resumed = jobs[body["job_id"]]
    assert resumed["status"] == "complete"
    assert resumed["resumable"] is True
    assert resumed["resume_source_job_id"] == "interrupted-1"
    assert resumed["resume_attempt"] == 1
    assert resumed["partial_output_path"] == partial
    assert resumed["result"]["payload"]["custom"] == "value"
    assert resumed["result"]["payload"]["resume_from_job_id"] == "interrupted-1"


def test_resume_rejects_non_resumable_interrupted_job(isolated_job_store):
    isolated_job_store.save_job({
        "id": "not-resumable",
        "type": "resume-fixture",
        "status": "interrupted",
        "created": time.time(),
        "_endpoint": "/resume-fixture",
        "_payload": {"custom": "value"},
        "resumable": False,
    })

    app = _make_resume_app()
    response = app.test_client().post("/jobs/not-resumable/resume", headers=_headers())
    body = response.get_json()

    assert response.status_code == 409
    assert body["code"] == "JOB_NOT_RESUMABLE"


def test_resume_requires_current_route_to_be_marked_resumable(isolated_job_store):
    isolated_job_store.save_job({
        "id": "stale-resumable",
        "type": "non-resume-fixture",
        "status": "interrupted",
        "created": time.time(),
        "_endpoint": "/non-resumable-fixture",
        "_payload": {"custom": "value"},
        "resumable": True,
    })

    app = _make_resume_app()
    response = app.test_client().post("/jobs/stale-resumable/resume", headers=_headers())
    body = response.get_json()

    assert response.status_code == 409
    assert body["code"] == "JOB_RESUME_UNAVAILABLE"
    assert "not marked resumable" in body["error"]


def test_checkpointable_routes_are_marked_resumable():
    expectations = {
        "opencut/routes/captions.py": {
            "captions": {"disk_operation": "transcribe", "resumable": True},
            "transcript": {"disk_operation": "transcribe", "resumable": True},
            "whisperx": {"disk_operation": "transcribe", "resumable": True},
        },
        "opencut/routes/audio.py": {
            "separate": {"disk_operation": "demucs", "resumable": True},
        },
        "opencut/routes/video_core.py": {
            "export": {"disk_operation": "video_export", "resumable": True},
            "export_preset": {"disk_operation": "video_export", "resumable": True},
        },
        "opencut/routes/video_specialty.py": {
            "shorts_pipeline": {"resumable": True},
        },
        "opencut/routes/wave_l_routes.py": {
            "depth_estimate_v2": {"resumable": True},
        },
    }

    root = Path(__file__).resolve().parents[1]
    for rel_path, route_expectations in expectations.items():
        source = (root / rel_path).read_text(encoding="utf-8")
        decorators = _async_job_decorators(source)
        for job_type, expected_kwargs in route_expectations.items():
            matches = decorators.get(job_type, [])
            assert matches, f"{rel_path} is missing @async_job({job_type!r}, ...)"
            assert any(
                all(match.get(name) == value for name, value in expected_kwargs.items())
                for match in matches
            ), f"{rel_path} @async_job({job_type!r}) is missing {expected_kwargs!r}"
