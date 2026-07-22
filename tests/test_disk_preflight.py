from concurrent.futures import Future
from pathlib import Path

from flask import Flask


def test_estimate_required_mb_uses_operation_ratio(tmp_path):
    from opencut.core.preflight import estimate_required_mb

    source = tmp_path / "clip.mov"
    source.write_bytes(b"x" * 2 * 1024 * 1024)

    assert estimate_required_mb("demucs", str(source), minimum_mb=1) == 8
    assert estimate_required_mb("video_export", str(source), minimum_mb=1) == 3


def test_ensure_disk_for_returns_output_dir_and_probe_result(tmp_path, monkeypatch):
    from opencut.core.preflight import ensure_disk_for

    source = tmp_path / "clip.mov"
    output_dir = tmp_path / "renders"
    source.write_bytes(b"x" * 1024)
    output_dir.mkdir()

    def fake_preflight(path, required_mb):
        return {
            "ok": False,
            "free_mb": 7,
            "required_mb": required_mb,
            "note": "Only 7 MB free.",
        }

    monkeypatch.setattr("opencut.core.disk_monitor.preflight", fake_preflight)

    result = ensure_disk_for(
        "video_export",
        str(source),
        {"output_dir": str(output_dir)},
        required_mb=12,
    )

    assert result["ok"] is False
    assert result["free_mb"] == 7
    assert result["required_mb"] == 12
    assert result["output_dir"] == str(output_dir)
    assert result["operation"] == "video_export"


def _make_disk_test_app():
    from opencut.jobs import async_job

    app = Flask(__name__)

    @app.route("/disk-test", methods=["POST"])
    @async_job("disk-test", disk_operation="video_export")
    def disk_test(job_id, filepath, data):
        return {"ok": True, "filepath": filepath}

    return app


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


def test_async_job_disk_preflight_failure_returns_507(tmp_path, monkeypatch):
    source = tmp_path / "clip.mov"
    output_dir = tmp_path / "renders"
    source.write_bytes(b"x")
    output_dir.mkdir()

    def fake_ensure(operation, source_path, data, required_mb=None):
        return {
            "ok": False,
            "operation": operation,
            "required_mb": 2048,
            "free_mb": 128,
            "output_dir": str(output_dir),
            "note": "Only 128 MB free.",
        }

    monkeypatch.setattr("opencut.core.preflight.ensure_disk_for", fake_ensure)
    app = _make_disk_test_app()
    response = app.test_client().post(
        "/disk-test",
        json={"filepath": str(source), "output_dir": str(output_dir)},
    )

    body = response.get_json()
    assert response.status_code == 507
    assert body["code"] == "INSUFFICIENT_STORAGE"
    assert body["required_mb"] == 2048
    assert body["free_mb"] == 128
    assert body["output_dir"] == str(output_dir)
    assert body["operation"] == "video_export"
    assert "job_id" not in body


def test_async_job_disk_preflight_ok_still_creates_job(tmp_path, monkeypatch):
    from opencut.jobs import jobs, job_lock

    source = tmp_path / "clip.mov"
    output_dir = tmp_path / "renders"
    source.write_bytes(b"x")
    output_dir.mkdir()

    def fake_ensure(operation, source_path, data, required_mb=None):
        return {
            "ok": True,
            "operation": operation,
            "required_mb": 1,
            "free_mb": 999,
            "output_dir": str(output_dir),
            "note": "",
        }

    monkeypatch.setattr("opencut.core.preflight.ensure_disk_for", fake_ensure)
    monkeypatch.setattr("opencut.workers.get_pool", lambda: _inline_pool())

    with job_lock:
        jobs.clear()
    app = _make_disk_test_app()
    response = app.test_client().post(
        "/disk-test",
        json={"filepath": str(source), "output_dir": str(output_dir)},
    )

    body = response.get_json()
    assert response.status_code == 200
    assert body["job_id"]
    with job_lock:
        job = jobs[body["job_id"]]
        assert job["status"] == "complete"
        assert job["result"]["ok"] is True


def test_high_impact_routes_enable_disk_preflight():
    expectations = {
        "opencut/routes/caption_generation_routes.py": [
            '@async_job("captions", disk_operation="transcribe", resumable=True)',
        ],
        "opencut/routes/caption_transcript_routes.py": [
            '@async_job("transcript", disk_operation="transcribe", resumable=True)',
        ],
        "opencut/routes/caption_pipeline_routes.py": [
            '@async_job("full", disk_operation="full_pipeline")',
        ],
        "opencut/routes/caption_enhancement_routes.py": [
            '@async_job("whisperx", disk_operation="transcribe", resumable=True)',
        ],
        "opencut/routes/caption_render_routes.py": [
            '@async_job("burnin", disk_operation="video_export")',
        ],
        "opencut/routes/audio.py": [
            '@async_job("separate", disk_operation="demucs", resumable=True)',
            '@async_job("deepfilter", disk_operation="deepfilter")',
        ],
        "opencut/routes/video_core.py": [
            '@async_job("export", disk_operation="video_export", resumable=True)',
            '@async_job("export_preset", disk_operation="video_export", resumable=True)',
        ],
        "opencut/routes/video_ai.py": [
            '@async_job("upscale", disk_operation="video_ai_heavy", rate_limit_key="ai_gpu")',
            '@async_job("interpolate", disk_operation="video_ai_heavy", rate_limit_key="ai_gpu")',
            '@async_job("denoise", disk_operation="video_ai_heavy", rate_limit_key=_denoise_rate_limit_key)',
        ],
    }

    root = Path(__file__).resolve().parents[1]
    for rel_path, snippets in expectations.items():
        source = (root / rel_path).read_text(encoding="utf-8")
        for snippet in snippets:
            assert snippet in source
