import importlib
import os
import threading
import time


def _reload_gpu(monkeypatch, request, *, max_jobs="1", timeout=None):
    original_env = {
        "OPENCUT_MAX_GPU_JOBS": os.environ.get("OPENCUT_MAX_GPU_JOBS"),
        "OPENCUT_GPU_ACQUIRE_TIMEOUT": os.environ.get("OPENCUT_GPU_ACQUIRE_TIMEOUT"),
    }
    monkeypatch.setenv("OPENCUT_MAX_GPU_JOBS", max_jobs)
    if timeout is None:
        monkeypatch.delenv("OPENCUT_GPU_ACQUIRE_TIMEOUT", raising=False)
    else:
        monkeypatch.setenv("OPENCUT_GPU_ACQUIRE_TIMEOUT", str(timeout))
    import opencut.core.gpu_semaphore as gpu_semaphore

    def restore_module():
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        importlib.reload(gpu_semaphore)

    request.addfinalizer(restore_module)

    return importlib.reload(gpu_semaphore)


def test_default_gpu_acquire_timeout_is_30_seconds(monkeypatch, request):
    gpu = _reload_gpu(monkeypatch, request)

    assert gpu.ACQUIRE_TIMEOUT == 30.0
    assert gpu.status().to_dict()["acquire_timeout_seconds"] == 30.0


def test_env_override_preserves_nonblocking_behavior(monkeypatch, request):
    gpu = _reload_gpu(monkeypatch, request, timeout=0)

    assert gpu.acquire() is True
    try:
        start = time.monotonic()
        assert gpu.acquire() is False
        assert time.monotonic() - start < 0.5
        assert gpu.status().to_dict()["rejected_total"] == 1
    finally:
        gpu.release()


def test_default_acquire_waits_for_released_slot(monkeypatch, request):
    gpu = _reload_gpu(monkeypatch, request)

    assert gpu.acquire() is True

    def release_later():
        time.sleep(0.05)
        gpu.release()

    thread = threading.Thread(target=release_later)
    thread.start()
    start = time.monotonic()
    try:
        assert gpu.acquire() is True
        assert time.monotonic() - start >= 0.03
    finally:
        gpu.release()
        thread.join(timeout=2)


def test_gpu_exclusive_timeout_returns_retry_metadata(monkeypatch, request):
    gpu = _reload_gpu(monkeypatch, request, timeout=0)

    assert gpu.acquire() is True
    try:
        @gpu.gpu_exclusive(fail_fast_raises=False)
        def run_gpu_job():
            return {"ok": True}

        result = run_gpu_job()

        assert result["error"] == "GPU_BUSY"
        assert result["retry_after"] == 1
        assert result["queue_depth"] == 1
    finally:
        gpu.release()


def test_gpu_busy_error_maps_to_retry_after_response(monkeypatch, request):
    from flask import Flask
    from opencut.errors import safe_error

    gpu = _reload_gpu(monkeypatch, request, timeout=1)
    err = gpu.GpuBusyError("GPU_BUSY: busy", retry_after=7, queue_depth=2)

    app = Flask(__name__)
    with app.app_context():
        response, status = safe_error(err, context="gpu")

    payload = response.get_json()
    assert status == 429
    assert payload["code"] == "GPU_BUSY"
    assert response.headers["Retry-After"] == "7"
