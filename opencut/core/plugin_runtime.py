"""Supervised out-of-process runtime for third-party Python plugins.

This boundary is process isolation for availability, not an operating-system
security sandbox. Workers receive a narrow authenticated JSON-lines protocol,
a sanitized environment, time/size limits, and crash-loop quarantine.
"""

from __future__ import annotations

import atexit
import json
import os
import queue
import secrets
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import psutil

STARTUP_TIMEOUT_SECONDS = 8.0
REQUEST_TIMEOUT_SECONDS = 30.0
MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_WORKER_MEMORY_MB = 512
MAX_CRASHES = 3
CRASH_WINDOW_SECONDS = 60.0


class PluginWorkerError(RuntimeError):
    """Safe parent-side worker failure with no plugin exception payload."""

    code = "PLUGIN_WORKER_UNAVAILABLE"
    status = 503
    suggestion = "Review plugin worker health, then restart or quarantine the plugin."


class PluginWorkerQuarantined(PluginWorkerError):
    code = "PLUGIN_WORKER_QUARANTINED"
    suggestion = "Inspect the plugin and explicitly restart its quarantined worker."


def _sanitized_worker_environment() -> dict[str, str]:
    allowed = {
        "APPDATA",
        "COMSPEC",
        "HOME",
        "LANG",
        "LC_ALL",
        "LOCALAPPDATA",
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "WINDIR",
    }
    env = {key: value for key, value in os.environ.items() if key.upper() in allowed}
    env.update(
        {
            "OPENCUT_PLUGIN_WORKER": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


class PluginWorkerSupervisor:
    def __init__(self, plugin_info: dict[str, Any]):
        self.plugin_info = dict(plugin_info)
        self.name = str(plugin_info["name"])
        self.plugin_dir = str(Path(plugin_info["path"]).resolve())
        self.capabilities = [
            str(capability)
            for capability in (plugin_info.get("capabilities") or [])
            if isinstance(capability, str)
        ]
        self.data_dir = str(
            Path(plugin_info.get("_data_dir") or (Path(self.plugin_dir) / "data")).resolve()
        )
        self._trust_required = bool(plugin_info.get("_trust_required", True))
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._responses: queue.Queue[str | None] = queue.Queue()
        self._reader: threading.Thread | None = None
        self._watchdog: threading.Thread | None = None
        self._token = ""
        self._catalog: dict[str, Any] | None = None
        self._crashes: deque[float] = deque()
        self._quarantined = False
        self._last_error = ""
        self._started_at = 0.0

    def _prune_crashes(self) -> None:
        cutoff = time.monotonic() - CRASH_WINDOW_SECONDS
        while self._crashes and self._crashes[0] < cutoff:
            self._crashes.popleft()

    def _record_failure(self, code: str) -> None:
        self._last_error = code
        self._crashes.append(time.monotonic())
        self._prune_crashes()
        if len(self._crashes) >= MAX_CRASHES:
            self._quarantined = True
        self._terminate_locked()

    def _reader_loop(
        self,
        process: subprocess.Popen[str],
        responses: queue.Queue[str | None],
    ) -> None:
        assert process.stdout is not None
        try:
            for line in process.stdout:
                responses.put(line)
        finally:
            responses.put(None)

    def _watchdog_loop(self, process: subprocess.Popen[str]) -> None:
        while process.poll() is None:
            time.sleep(0.2)
            with self._lock:
                if self._process is not process:
                    return
                if self._memory_bytes_locked() > MAX_WORKER_MEMORY_MB * 1024 * 1024:
                    self._record_failure("memory_limit")
                    return

    def _write_locked(self, payload: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None or process.poll() is not None:
            raise PluginWorkerError("Plugin worker is not running.")
        encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        if len(encoded.encode("utf-8")) > MAX_REQUEST_BYTES:
            raise PluginWorkerError("Plugin worker request exceeds the IPC size limit.")
        try:
            process.stdin.write(encoded + "\n")
            process.stdin.flush()
        except OSError as exc:
            self._record_failure("protocol_write")
            raise PluginWorkerError("Plugin worker IPC write failed.") from exc

    def _memory_bytes_locked(self) -> int:
        process = self._process
        if process is None or process.poll() is not None:
            return 0
        try:
            parent = psutil.Process(process.pid)
            total = parent.memory_info().rss
            for child in parent.children(recursive=True):
                total += child.memory_info().rss
            return total
        except (psutil.Error, OSError):
            return 0

    def _await_locked(self, request_id: str, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._memory_bytes_locked() > MAX_WORKER_MEMORY_MB * 1024 * 1024:
                self._record_failure("memory_limit")
                raise PluginWorkerError("Plugin worker exceeded its memory limit.")
            process = self._process
            if process is None or process.poll() is not None:
                self._record_failure("process_exit")
                raise PluginWorkerError("Plugin worker exited unexpectedly.")
            try:
                line = self._responses.get(timeout=min(0.1, max(0.01, deadline - time.monotonic())))
            except queue.Empty:
                continue
            if line is None:
                self._record_failure("protocol_closed")
                raise PluginWorkerError("Plugin worker closed its IPC channel.")
            if len(line.encode("utf-8", errors="replace")) > MAX_RESPONSE_BYTES:
                self._record_failure("response_limit")
                raise PluginWorkerError("Plugin worker response exceeds the IPC size limit.")
            try:
                response = json.loads(line)
            except json.JSONDecodeError as exc:
                self._record_failure("invalid_protocol")
                raise PluginWorkerError("Plugin worker returned invalid IPC data.") from exc
            if not isinstance(response, dict) or response.get("id") != request_id:
                self._record_failure("protocol_mismatch")
                raise PluginWorkerError("Plugin worker response did not match the request.")
            return response
        self._record_failure("request_timeout")
        raise PluginWorkerError("Plugin worker request timed out.")

    def _start_locked(self) -> None:
        if self._quarantined:
            raise PluginWorkerQuarantined("Plugin worker is in crash-loop quarantine.")
        if self._process is not None and self._process.poll() is None:
            return

        from opencut.core.plugin_manifest import validate_plugin_manifest

        if self._trust_required:
            validation = validate_plugin_manifest(self.plugin_dir)
            if not validation.valid:
                self._last_error = "trust_validation"
                raise PluginWorkerError("Plugin trust validation failed before worker start.")

        responses: queue.Queue[str | None] = queue.Queue()
        self._responses = responses
        self._token = secrets.token_urlsafe(32)
        command = [
            sys.executable,
            "-m",
            "opencut.core.plugin_worker",
            "--plugin-dir",
            self.plugin_dir,
            "--plugin-name",
            self.name,
            "--memory-mb",
            str(MAX_WORKER_MEMORY_MB),
        ]
        kwargs: dict[str, Any] = {
            "cwd": str(Path(__file__).resolve().parents[2]),
            "env": _sanitized_worker_environment(),
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.DEVNULL,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "bufsize": 1,
            "close_fds": True,
        }
        if os.name == "nt":
            kwargs["creationflags"] = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
        else:
            kwargs["start_new_session"] = True
        try:
            self._process = subprocess.Popen(command, **kwargs)
            self._reader = threading.Thread(
                target=self._reader_loop,
                args=(self._process, responses),
                name=f"opencut-plugin-{self.name}-ipc",
                daemon=True,
            )
            self._reader.start()
            self._watchdog = threading.Thread(
                target=self._watchdog_loop,
                args=(self._process,),
                name=f"opencut-plugin-{self.name}-watchdog",
                daemon=True,
            )
            self._watchdog.start()
            request_id = secrets.token_hex(8)
            self._write_locked(
                {
                    "id": request_id,
                    "action": "hello",
                    "token": self._token,
                    "context": {
                        "capabilities": self.capabilities,
                        "data_dir": self.data_dir,
                    },
                }
            )
            ready = self._await_locked(request_id, STARTUP_TIMEOUT_SECONDS)
            if not ready.get("ok") or not isinstance(ready.get("catalog"), dict):
                self._record_failure("startup_failed")
                raise PluginWorkerError("Plugin worker failed during startup.")
            catalog = ready["catalog"]
            if self._catalog is not None and catalog != self._catalog:
                self._record_failure("catalog_drift")
                self._quarantined = True
                raise PluginWorkerQuarantined(
                    "Plugin route catalog changed after registration; restart OpenCut."
                )
            self._catalog = catalog
            self._started_at = time.time()
            self._last_error = ""
        except PluginWorkerError:
            raise
        except Exception as exc:
            self._record_failure("spawn_failed")
            raise PluginWorkerError("Plugin worker could not be started.") from exc

    def _terminate_locked(self) -> None:
        process = self._process
        self._process = None
        self._token = ""
        if process is None:
            return
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1.0)
        except (OSError, subprocess.SubprocessError):
            pass
        for stream in (process.stdin, process.stdout):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass

    def probe_catalog(self) -> dict[str, Any]:
        """Validate/import in a disposable worker, then remain stopped."""
        with self._lock:
            self._start_locked()
            assert self._catalog is not None
            catalog = json.loads(json.dumps(self._catalog))
            self._terminate_locked()
            return catalog

    def call(self, action: str, **payload: Any) -> dict[str, Any]:
        with self._lock:
            self._start_locked()
            request_id = secrets.token_hex(8)
            self._write_locked(
                {
                    "id": request_id,
                    "action": action,
                    "token": self._token,
                    **payload,
                }
            )
            response = self._await_locked(request_id, REQUEST_TIMEOUT_SECONDS)
            if not response.get("ok"):
                self._last_error = str(response.get("code") or "plugin_error")[:80]
                raise PluginWorkerError("Plugin worker rejected the request.")
            return response

    def stop(self) -> None:
        with self._lock:
            self._terminate_locked()

    def restart(self) -> dict[str, Any]:
        acquired = self._lock.acquire(timeout=0.1)
        if not acquired:
            # An in-flight request owns the protocol lock. Killing the worker
            # makes that request fail promptly so an operator restart does not
            # wait for the full request timeout.
            process = self._process
            try:
                if process is not None and process.poll() is None:
                    process.kill()
            except OSError:
                pass
            acquired = self._lock.acquire(timeout=2.0)
        if not acquired:
            raise PluginWorkerError("Plugin worker could not be interrupted for restart.")
        try:
            self._terminate_locked()
            self._quarantined = False
            self._crashes.clear()
            self._last_error = ""
            self._start_locked()
            return self.status()
        finally:
            self._lock.release()

    def status(self) -> dict[str, Any]:
        acquired = self._lock.acquire(timeout=0.05)
        if acquired:
            try:
                self._prune_crashes()
                return self._status_payload(busy=False)
            finally:
                self._lock.release()
        return self._status_payload(busy=True)

    def _status_payload(self, *, busy: bool) -> dict[str, Any]:
        process = self._process
        running = process is not None and process.poll() is None
        state = "quarantined" if self._quarantined else ("running" if running else "stopped")
        if busy and running:
            state = "busy"
        return {
            "name": self.name,
            "state": state,
            "isolation": "supervised_process",
            "security_boundary": "availability isolation; not an OS sandbox",
            "crash_count": len(self._crashes),
            "last_error": self._last_error,
            "started_at": self._started_at if running else None,
            "limits": {
                "startup_timeout_seconds": STARTUP_TIMEOUT_SECONDS,
                "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
                "memory_mb": MAX_WORKER_MEMORY_MB,
                "request_bytes": MAX_REQUEST_BYTES,
                "response_bytes": MAX_RESPONSE_BYTES,
            },
        }


_supervisors: dict[str, PluginWorkerSupervisor] = {}
_supervisors_lock = threading.Lock()


def register_supervisor(supervisor: PluginWorkerSupervisor) -> None:
    with _supervisors_lock:
        previous = _supervisors.get(supervisor.name)
        if previous is not None and previous is not supervisor:
            previous.stop()
        _supervisors[supervisor.name] = supervisor


def get_supervisor(name: str) -> PluginWorkerSupervisor | None:
    with _supervisors_lock:
        return _supervisors.get(name)


def worker_statuses() -> list[dict[str, Any]]:
    with _supervisors_lock:
        supervisors = list(_supervisors.values())
    return [supervisor.status() for supervisor in supervisors]


def restart_worker(name: str) -> dict[str, Any]:
    supervisor = get_supervisor(name)
    if supervisor is None:
        raise KeyError(name)
    return supervisor.restart()


def unregister_supervisor(name: str) -> None:
    with _supervisors_lock:
        supervisor = _supervisors.pop(name, None)
    if supervisor is not None:
        supervisor.stop()


def _shutdown_workers() -> None:
    with _supervisors_lock:
        supervisors = list(_supervisors.values())
        _supervisors.clear()
    for supervisor in supervisors:
        supervisor.stop()


atexit.register(_shutdown_workers)
