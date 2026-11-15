"""F461: diagnostics read the log the server writes.

The server logs to ``~/.opencut/server.log``. Issue reports, crash packets,
job diagnostics and ``/logs/export`` used to look for ``opencut.log``, a file
nothing writes, so every bundle shipped without logs.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import uuid
import zipfile

import pytest

from opencut import server_log


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".opencut").mkdir(parents=True)
    real_expanduser = os.path.expanduser
    monkeypatch.setattr(
        "os.path.expanduser",
        lambda p: str(home) if p == "~" else real_expanduser(p),
    )
    import opencut.core.issue_report as ir

    monkeypatch.setattr(ir, "_HOME", str(home))
    return home


@pytest.fixture
def server_logger(fake_home):
    """The ``opencut`` logger writing where the server's file handler would."""
    from opencut import server

    handler = logging.handlers.RotatingFileHandler(
        server_log.server_log_path(),
        maxBytes=server._file_handler.maxBytes,
        backupCount=server._file_handler.backupCount,
        encoding="utf-8",
    )
    handler.setFormatter(server._file_handler.formatter)
    logger = logging.getLogger("opencut")
    logger.addHandler(handler)
    try:
        yield logger
    finally:
        logger.removeHandler(handler)
        handler.close()


def test_server_writes_the_shared_path():
    from opencut import server

    assert server.LOG_FILE == server_log.server_log_path()
    assert os.path.basename(server.LOG_FILE) == "server.log"
    assert server._file_handler.baseFilename == os.path.abspath(server.LOG_FILE)
    assert server._file_handler.backupCount == server_log.SERVER_LOG_BACKUP_COUNT
    assert server._file_handler.maxBytes == server_log.SERVER_LOG_MAX_BYTES


def test_marker_logged_by_the_server_reaches_every_bundle(server_logger, fake_home, tmp_path, monkeypatch, client):
    marker = f"f461-marker-{uuid.uuid4().hex}"
    job_id = f"job-{uuid.uuid4().hex[:8]}"
    server_logger.warning("diagnostics marker %s for %s", marker, job_id)
    for handler in server_logger.handlers:
        handler.flush()
    assert marker in (fake_home / ".opencut" / "server.log").read_text(encoding="utf-8")

    from opencut.core import crash_packet, issue_report, job_diagnostics

    report = issue_report.bundle(title="t", description="d")
    assert marker in report["body"]

    packet = crash_packet.build_packet(output_path=tmp_path / "packet.zip", include_jobs=False)
    with zipfile.ZipFile(packet.output_path) as zf:
        assert marker in zf.read("server.log").decode("utf-8")

    monkeypatch.setattr(job_diagnostics, "_resolve_job", lambda _job_id: {"status": "done"})
    diag = job_diagnostics.build_diagnostic(job_id)
    assert diag.log_slice is not None
    assert marker in diag.log_slice.body
    assert diag.log_slice.source == server_log.server_log_path()

    resp = client.get("/logs/export")
    assert resp.status_code == 200
    assert "opencut_logs.txt" in resp.headers["Content-Disposition"]
    assert marker in resp.get_data(as_text=True)


def test_tail_reaches_into_rotated_files_when_the_live_log_is_short(fake_home):
    data_dir = fake_home / ".opencut"
    (data_dir / "server.log.2").write_text("oldest\n", encoding="utf-8")
    (data_dir / "server.log.1").write_text("older-1\nolder-2\n", encoding="utf-8")
    (data_dir / "server.log").write_text("newest\n", encoding="utf-8")

    assert server_log.server_log_tail(max_lines=3) == ["older-1", "older-2", "newest"]
    assert server_log.server_log_tail(max_lines=1) == ["newest"]
    assert server_log.server_log_tail(max_lines=50) == ["oldest", "older-1", "older-2", "newest"]
    assert server_log.server_log_tail() == ["oldest", "older-1", "older-2", "newest"]


def test_tail_byte_budget_spans_files_and_drops_the_partial_line(fake_home):
    data_dir = fake_home / ".opencut"
    # Exact bytes: text mode would write CRLF on Windows and shift the budget.
    (data_dir / "server.log.1").write_bytes(b"aaaa\nbbbb\n")
    (data_dir / "server.log").write_bytes(b"cccc\n")

    # 5 bytes from server.log, then 7 of server.log.1's 10: "a\nbbbb\n",
    # whose leading fragment "a" is dropped.
    assert server_log.server_log_tail(max_bytes=12) == ["bbbb", "cccc"]
    assert server_log.server_log_tail(max_bytes=5) == ["cccc"]


def test_tail_with_no_log_is_empty(fake_home):
    assert server_log.server_log_tail(max_lines=10) == []
    assert server_log.server_log_tail(max_lines=0) == []


def test_export_without_any_log_is_404(fake_home, client, monkeypatch):
    import opencut.routes.settings as settings_routes

    monkeypatch.setattr(settings_routes, "OPENCUT_DIR", str(fake_home / ".opencut"))
    resp = client.get("/logs/export")
    assert resp.status_code == 404


def test_export_includes_rotated_lines_and_the_crash_log(fake_home, client, monkeypatch):
    import opencut.routes.settings as settings_routes

    data_dir = fake_home / ".opencut"
    monkeypatch.setattr(settings_routes, "OPENCUT_DIR", str(data_dir))
    (data_dir / "server.log.1").write_text("before rotation\n", encoding="utf-8")
    (data_dir / "server.log").write_text("after rotation\n", encoding="utf-8")
    (data_dir / "crash.log").write_text("Traceback marker\n", encoding="utf-8")

    resp = client.get("/logs/export")
    assert resp.status_code == 200
    assert "opencut_logs.txt" in resp.headers["Content-Disposition"]
    text = resp.get_data(as_text=True)
    assert text.index("before rotation") < text.index("after rotation")
    assert "Traceback marker" in text
