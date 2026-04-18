import asyncio
import concurrent.futures
import threading


def test_run_async_sync_uses_single_worker_fallback(monkeypatch):
    import opencut.core.voice_gen as voice_gen

    real_asyncio_run = asyncio.run
    real_executor = concurrent.futures.ThreadPoolExecutor
    main_thread_id = threading.get_ident()
    recorded = {}

    class RecordingExecutor:
        def __init__(self, *args, **kwargs):
            recorded.update(kwargs)
            self._inner = real_executor(*args, **kwargs)

        def __enter__(self):
            self._inner.__enter__()
            return self

        def __exit__(self, exc_type, exc, tb):
            return self._inner.__exit__(exc_type, exc, tb)

        def submit(self, *args, **kwargs):
            return self._inner.submit(*args, **kwargs)

    def fake_run(coro):
        if threading.get_ident() == main_thread_id:
            raise RuntimeError("loop already running")
        return real_asyncio_run(coro)

    monkeypatch.setattr(voice_gen.asyncio, "run", fake_run)
    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", RecordingExecutor)

    async def sample():
        return "ok"

    result = voice_gen._run_async_sync(sample(), timeout=5)

    assert result == "ok"
    assert recorded["max_workers"] == 1
    assert recorded["thread_name_prefix"] == "oc-tts"
