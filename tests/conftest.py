"""
Shared pytest fixtures for OpenCut integration tests.
"""

import builtins
import io
import logging
import os
import sys
from pathlib import Path

import pytest

_fixture_log = logging.getLogger("opencut.tests.conftest")
_REAL_USER_DATA_ROOT = os.path.normcase(
    os.path.realpath(os.path.join(os.path.expanduser("~"), ".opencut"))
)


def _is_real_user_data_path(path) -> bool:
    """Return whether *path* resolves inside the developer's real profile."""
    if isinstance(path, int):
        return False
    try:
        candidate = os.path.normcase(
            os.path.realpath(os.path.expanduser(os.fsdecode(os.fspath(path))))
        )
    except (OSError, TypeError, ValueError):
        return False
    return candidate == _REAL_USER_DATA_ROOT or candidate.startswith(
        _REAL_USER_DATA_ROOT + os.sep
    )


def _reject_real_user_data_write(path, operation: str) -> None:
    if _is_real_user_data_path(path):
        raise AssertionError(
            f"test attempted to {operation} real OpenCut user data: {path!s}"
        )


def _replace_real_user_data_path(value, isolated_root):
    """Map a loaded OpenCut path from the real profile into ``isolated_root``."""
    if isinstance(value, Path):
        candidate = os.path.normcase(os.path.realpath(os.fspath(value)))
    elif isinstance(value, str):
        candidate = os.path.normcase(os.path.realpath(value))
    else:
        return None
    if not (
        candidate == _REAL_USER_DATA_ROOT
        or candidate.startswith(_REAL_USER_DATA_ROOT + os.sep)
    ):
        return None
    relative = os.path.relpath(candidate, _REAL_USER_DATA_ROOT)
    if relative == ".":
        return Path(isolated_root) if isinstance(value, Path) else str(isolated_root)
    replacement = Path(isolated_root) / relative
    return replacement if isinstance(value, Path) else str(replacement)


def _redirect_loaded_opencut_paths(monkeypatch, isolated_root) -> None:
    """Redirect already-imported OpenCut path constants for this test."""
    for module_name, module in list(sys.modules.items()):
        if module is None or not (
            module_name == "opencut" or module_name.startswith("opencut.")
        ):
            continue
        for name, value in list(vars(module).items()):
            replacement = _replace_real_user_data_path(value, isolated_root)
            if replacement is not None:
                monkeypatch.setattr(module, name, replacement)


@pytest.fixture
def app():
    """Create a Flask app instance configured for testing."""
    from opencut.config import OpenCutConfig
    from opencut.server import create_app
    test_config = OpenCutConfig()
    flask_app = create_app(config=test_config, testing=True)
    flask_app.config["TESTING"] = True
    return flask_app


@pytest.fixture
def client(app):
    """Flask test client -- no real network, no subprocess needed."""
    return app.test_client()


@pytest.fixture
def csrf_token(client):
    """Fetch a valid CSRF token from the /health endpoint."""
    resp = client.get("/health")
    data = resp.get_json()
    return data.get("csrf_token", "")


def csrf_headers(token):
    """Build headers dict with CSRF token and JSON content type."""
    return {
        "X-OpenCut-Token": token,
        "Content-Type": "application/json",
    }


@pytest.fixture(autouse=True, scope="session")
def _shutdown_worker_pool():
    """Shut down the WorkerPool after all tests to prevent pytest hang on exit."""
    yield
    from opencut.workers import shutdown_pool
    shutdown_pool(wait=False)


@pytest.fixture(autouse=True)
def _isolate_os_credential_vault(monkeypatch):
    """Never let tests read or write the developer's real OS credential vault."""
    from opencut import credential_store

    backend = credential_store.MemoryCredentialBackend()
    monkeypatch.setattr(credential_store, "_backend_override", backend)
    monkeypatch.setattr(credential_store, "_last_error", "")
    monkeypatch.delenv(credential_store.INSECURE_OPT_IN_ENV, raising=False)
    return backend


@pytest.fixture(autouse=True, scope="session")
def _isolate_process_user_data_home(tmp_path_factory):
    """Keep dynamically computed OpenCut paths inside one test-only home."""
    isolated_home = str(tmp_path_factory.mktemp("opencut-home"))
    patch = pytest.MonkeyPatch()
    patch.setenv("HOME", isolated_home)
    real_expanduser = os.path.expanduser

    def isolated_expanduser(path):
        if path == "~":
            return isolated_home
        if isinstance(path, str) and (path.startswith("~/") or path.startswith("~\\")):
            return os.path.join(isolated_home, path[2:])
        return real_expanduser(path)

    patch.setattr(os.path, "expanduser", isolated_expanduser)
    _redirect_loaded_opencut_paths(patch, Path(isolated_home) / ".opencut")
    try:
        yield
    finally:
        patch.undo()


@pytest.fixture(autouse=True)
def _isolate_persistent_module_writers(
    monkeypatch, tmp_path, _isolate_process_user_data_home
):
    """Redirect module-level persistence constants to a per-test directory."""
    from opencut.core import multilang_subtitle, render_queue, review_comments

    monkeypatch.setattr(
        render_queue, "_QUEUE_PATH", str(tmp_path / "render_queue.json")
    )
    monkeypatch.setattr(render_queue, "_ensure_opencut_dir", lambda: None)
    monkeypatch.setattr(review_comments, "_REVIEWS_DIR", str(tmp_path / "reviews"))
    monkeypatch.setattr(
        multilang_subtitle, "SUBTITLE_DIR", str(tmp_path / "subtitles")
    )

    with render_queue._queue_lock:
        saved_queue = list(render_queue._queue)
        render_queue._queue.clear()
    try:
        yield
    finally:
        with render_queue._queue_lock:
            render_queue._queue.clear()
            render_queue._queue.extend(saved_queue)


@pytest.fixture(autouse=True)
def _reject_real_user_data_writes(monkeypatch):
    """Fail immediately if a test tries to mutate the real ``~/.opencut``."""
    real_open = builtins.open
    real_io_open = io.open
    real_os_open = os.open

    def guarded_open(file, mode="r", *args, **kwargs):
        if isinstance(mode, str) and any(
            flag in mode for flag in ("w", "a", "x", "+")
        ):
            _reject_real_user_data_write(file, f"open for {mode!r}")
        return real_open(file, mode, *args, **kwargs)

    def guarded_io_open(file, mode="r", *args, **kwargs):
        if isinstance(mode, str) and any(
            flag in mode for flag in ("w", "a", "x", "+")
        ):
            _reject_real_user_data_write(file, f"io.open for {mode!r}")
        return real_io_open(file, mode, *args, **kwargs)

    def guarded_os_open(path, flags, *args, **kwargs):
        write_flags = (
            os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        )
        if flags & write_flags:
            _reject_real_user_data_write(path, f"os.open with flags {flags}")
        return real_os_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(io, "open", guarded_io_open)
    monkeypatch.setattr(os, "open", guarded_os_open)

    real_makedirs = os.makedirs

    def guarded_makedirs(path, mode=0o777, exist_ok=False):
        if _is_real_user_data_path(path) and not os.path.isdir(path):
            _reject_real_user_data_write(path, "create a directory at")
        return real_makedirs(path, mode=mode, exist_ok=exist_ok)

    monkeypatch.setattr(os, "makedirs", guarded_makedirs)

    real_mkdir = os.mkdir

    def guarded_mkdir(path, mode=0o777, *, dir_fd=None):
        if _is_real_user_data_path(path) and not os.path.isdir(path):
            _reject_real_user_data_write(path, "create a directory at")
        return real_mkdir(path, mode=mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "mkdir", guarded_mkdir)

    for name in (
        "replace",
        "rename",
        "renames",
        "remove",
        "unlink",
        "rmdir",
        "removedirs",
        "truncate",
        "utime",
        "chmod",
        "link",
        "symlink",
    ):
        real_operation = getattr(os, name)

        def guarded_operation(
            *args, _name=name, _operation=real_operation, **kwargs
        ):
            path_args = (
                args[:2]
                if _name in {"replace", "rename", "renames", "link", "symlink"}
                else args[:1]
            )
            for path in path_args:
                _reject_real_user_data_write(path, f"os.{_name}")
            return _operation(*args, **kwargs)

        monkeypatch.setattr(os, name, guarded_operation)


@pytest.fixture(autouse=True)
def _neutralize_machine_local_only(monkeypatch):
    """Keep tests hermetic against this machine's ``~/.opencut/local_only.json``.

    Network-gated features call ``config.is_local_only()``, which reads a
    user-global settings file. On a developer machine that has privacy /
    local-only mode enabled, every test that exercises a cloud-capable feature
    without mocking the gate would spuriously fail (stock search, external TTS,
    etc.). Simulate a clean machine — no setting file — by default; tests still
    opt into local-only mode via the ``OPENCUT_LOCAL_ONLY`` env var (checked
    first, before the file) or by patching ``is_local_only`` directly.
    """
    from opencut import user_data

    real_read = user_data.read_user_file

    def _patched_read(filename, default=None):
        if filename == "local_only.json":
            return {} if default is None else default
        return real_read(filename, default=default)

    monkeypatch.setattr(user_data, "read_user_file", _patched_read)


@pytest.fixture(autouse=True)
def _isolate_global_state():
    """Reset module-level mutable state between tests.

    The hot-path job dict, queue, and process registry in ``opencut.jobs``
    survive across tests because they're module globals. Without this
    fixture, a test that posts to ``/silence`` leaves a "running" entry
    that the next test sees when it queries ``/jobs`` or ``/jobs/stats``.
    The same applies to the in-memory queue in ``routes.jobs_routes`` and
    the per-app caches in ``routes.system``. Cleanup runs *after* each
    test so individual test bodies still see their own writes.
    """
    yield
    try:
        from opencut import jobs as _jobs_mod
        with _jobs_mod.job_lock:
            _jobs_mod.jobs.clear()
            _jobs_mod._job_processes.clear()
    except Exception as _e:
        _fixture_log.warning("jobs teardown failed: %s", _e)
    try:
        from opencut.routes import jobs_routes as _jr
        with _jr.job_queue_lock:
            _jr.job_queue.clear()
            _jr._queue_state["running"] = False
            _jr._queue_persistence_enabled = False
            _jr._queue_app = None
            _jr._queue_storage_error = None
    except Exception as _e:
        _fixture_log.warning("job_queue teardown failed: %s", _e)
    try:
        from opencut.routes.system import invalidate_caps_cache
        invalidate_caps_cache()
    except Exception as _e:
        _fixture_log.warning("caps_cache teardown failed: %s", _e)
