"""Tests for enriched async job metadata (N9)."""

from __future__ import annotations

import time
from types import SimpleNamespace

from flask import Blueprint

from opencut.core import job_diagnostics as jd


def _isolate_job_store(monkeypatch, tmp_path):
    import opencut.job_store as store

    store.close_all_connections()
    monkeypatch.setattr(store, "_DB_PATH", str(tmp_path / "jobs.db"))
    store._INITIALIZED = False
    store._INITIALIZED_PATH = None
    store._LOCAL = type(store._LOCAL)()
    store._ALL_CONNECTIONS = {}
    return store


def test_classify_exit_reason_enum_values():
    assert jd.classify_exit_reason("complete") == "complete"
    assert jd.classify_exit_reason("cancelled") == "cancelled"
    assert jd.classify_exit_reason("interrupted") == "interrupted"
    assert jd.classify_exit_reason("error", error="CUDA out of memory") == "oom"
    assert jd.classify_exit_reason("error", exc=MemoryError("allocation failed")) == "oom"
    assert jd.classify_exit_reason("error", exc=TimeoutError("timed out")) == "timeout"
    assert jd.classify_exit_reason("error", code="INSUFFICIENT_STORAGE") == "preflight_failed"
    assert jd.classify_exit_reason("error", error="unknown failure") == "error"


def test_job_resource_sampler_keeps_peak_values():
    samples = iter([
        {"peak_vram_mb": 512, "peak_cpu_pct": 10, "peak_rss_mb": 128},
        {"peak_vram_mb": 256, "peak_cpu_pct": 90, "peak_rss_mb": 64},
    ])

    sampler = jd.JobResourceSampler(
        pid=123,
        interval_seconds=60,
        sampler=lambda _pid: next(samples),
    )

    sampler.start()
    snapshot = sampler.stop()

    assert snapshot == {
        "peak_vram_mb": 512,
        "peak_cpu_pct": 90,
        "peak_rss_mb": 128,
    }


def test_sample_process_resources_uses_process_tree_and_nvml():
    class FakeMemory:
        def __init__(self, rss):
            self.rss = rss

    class FakeProcess:
        def __init__(self, pid, cpu, rss, children=None):
            self.pid = pid
            self._cpu = cpu
            self._rss = rss
            self._children = children or []

        def cpu_percent(self, interval=None):
            return self._cpu

        def memory_info(self):
            return FakeMemory(self._rss)

        def children(self, recursive=True):
            return list(self._children)

    child = FakeProcess(202, 25.5, 512 * 1024 * 1024)
    root = FakeProcess(101, 50.0, 1024 * 1024 * 1024, [child])
    fake_psutil = SimpleNamespace(Process=lambda pid: root)
    fake_proc = SimpleNamespace(pid=202, usedGpuMemory=768 * 1024 * 1024)
    fake_nvml = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlDeviceGetCount=lambda: 1,
        nvmlDeviceGetHandleByIndex=lambda idx: "gpu0",
        nvmlDeviceGetComputeRunningProcesses=lambda handle: [fake_proc],
        nvmlDeviceGetGraphicsRunningProcesses=lambda handle: [],
    )

    snapshot = jd.sample_process_resources(
        101,
        psutil_module=fake_psutil,
        pynvml_module=fake_nvml,
    )

    assert snapshot == {
        "peak_cpu_pct": 76,
        "peak_rss_mb": 1536,
        "peak_vram_mb": 768,
    }


def test_async_job_persists_resource_metadata(app, monkeypatch, tmp_path):
    store = _isolate_job_store(monkeypatch, tmp_path)
    import opencut.jobs as jobs_mod

    original_persist = jobs_mod._persist_job

    def sync_persist(job_dict, *, sync=False):
        original_persist(job_dict, sync=True)

    class FakeSampler:
        def start(self):
            return self

        def stop(self):
            return {
                "peak_vram_mb": 2048,
                "peak_cpu_pct": 77,
                "peak_rss_mb": 512,
            }

    monkeypatch.setattr(jobs_mod, "_persist_job", sync_persist)
    monkeypatch.setattr(jd, "JobResourceSampler", FakeSampler)

    bp = Blueprint("n9_job_metadata", __name__)

    @bp.route("/n9/job-metadata", methods=["POST"])
    @jobs_mod.async_job("n9_metadata", filepath_required=False)
    def n9_metadata(job_id, filepath, data):
        return {"ok": True}

    app.register_blueprint(bp)
    client = app.test_client()
    response = client.post("/n9/job-metadata", json={"mode": "test"})
    assert response.status_code == 200
    job_id = response.get_json()["job_id"]

    with jobs_mod.job_lock:
        future = jobs_mod.jobs[job_id]["_future"]
    future.result(timeout=5)

    live = jobs_mod._get_job_copy(job_id)
    persisted = store.get_job(job_id)

    for record in (live, persisted):
        assert record["status"] == "complete"
        assert record["exit_reason"] == "complete"
        assert record["peak_vram_mb"] == 2048
        assert record["peak_cpu_pct"] == 77
        assert record["peak_rss_mb"] == 512


def test_jobs_detail_falls_back_to_persisted_metadata(app, monkeypatch, tmp_path):
    store = _isolate_job_store(monkeypatch, tmp_path)
    created = time.time()
    store.save_job({
        "id": "persisted-n9",
        "type": "metadata",
        "status": "error",
        "progress": 10,
        "error": "CUDA out of memory",
        "exit_reason": "oom",
        "peak_vram_mb": 8192,
        "peak_cpu_pct": 100,
        "peak_rss_mb": 4096,
        "created": created,
    })

    response = app.test_client().get("/jobs/persisted-n9")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["id"] == "persisted-n9"
    assert payload["exit_reason"] == "oom"
    assert payload["peak_vram_mb"] == 8192
    assert payload["peak_cpu_pct"] == 100
    assert payload["peak_rss_mb"] == 4096
