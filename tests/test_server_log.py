"""F461: diagnostics read the log the server writes.

The server logs to ``~/.opencut/server.log``. Issue reports, crash packets,
job diagnostics and ``/logs/export`` used to look for ``opencut.log``, a file
nothing writes, so every bundle shipped without logs.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import time
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


def test_secrets_the_server_logged_never_reach_a_bundle(server_logger, fake_home, tmp_path, monkeypatch, client):
    import opencut.routes.settings as settings_routes

    monkeypatch.setattr(settings_routes, "OPENCUT_DIR", str(fake_home / ".opencut"))
    job_id = f"job-{uuid.uuid4().hex[:8]}"
    secrets = {
        "bearer": "abcDEF1234567890tokenvalue",
        "openai": "sk-proj-AAAAbbbbCCCCdddd1234",
        "userinfo": "hunter2pass",
        "srt": "MySrtPass12345",
        "shaka": "fedcba9876543210fedcba9876543210",
    }
    server_logger.warning("%s request Authorization: Bearer %s", job_id, secrets["bearer"])
    server_logger.warning("%s calling api_key=%s", job_id, secrets["openai"])
    server_logger.warning("%s source https://alice:%s@example.com/feed", job_id, secrets["userinfo"])
    server_logger.warning("%s output srt://host:9000?mode=caller&passphrase=%s", job_id, secrets["srt"])
    server_logger.warning(
        "%s packager --keys label=PRIMARY:key_id=00112233445566778899aabbccddeeff:key=%s",
        job_id,
        secrets["shaka"],
    )
    for handler in server_logger.handlers:
        handler.flush()
    raw = (fake_home / ".opencut" / "server.log").read_text(encoding="utf-8")
    assert all(value in raw for value in secrets.values())

    from opencut.core import crash_packet, issue_report, job_diagnostics

    report = issue_report.bundle(title="t", description="d")
    packet = crash_packet.build_packet(output_path=tmp_path / "packet.zip", include_jobs=False)
    with zipfile.ZipFile(packet.output_path) as zf:
        packet_log = zf.read("server.log").decode("utf-8")
    monkeypatch.setattr(
        job_diagnostics,
        "_resolve_job",
        lambda _job_id: {
            "status": "error",
            "error": f"upload failed for srt://host:9000?passphrase={secrets['srt']}",
            "params": {"url": f"https://alice:{secrets['userinfo']}@example.com/x", "api_key": secrets["openai"]},
        },
    )
    diag = job_diagnostics.build_diagnostic(job_id)
    resp = client.get("/logs/export")
    assert resp.status_code == 200

    outputs = {
        "issue report": report["body"],
        "crash packet": packet_log,
        "job log slice": diag.log_slice.body if diag.log_slice else "",
        "job error": diag.error,
        "job metadata": str(diag.metadata),
        "log export": resp.get_data(as_text=True),
    }
    assert job_id in outputs["job log slice"]
    for where, text in outputs.items():
        for name, value in secrets.items():
            assert value not in text, f"{name} secret leaked into the {where}"
    assert "[REDACTED]" in outputs["log export"]


def test_credential_forms_in_the_json_log_never_reach_a_bundle(
    server_logger, fake_home, tmp_path, monkeypatch, client
):
    """server.log lines are JSON objects, so a header is never at the start of
    a line and quotes and backslashes arrive escaped. Each form below got
    through every bundle before the log lines were decoded first."""
    import opencut.routes.settings as settings_routes
    import opencut.server as server_module

    data_dir = fake_home / ".opencut"
    monkeypatch.setattr(settings_routes, "OPENCUT_DIR", str(data_dir))
    monkeypatch.setattr(server_module, "LOG_FILE", str(data_dir / "server.log"))
    job_id = f"job-{uuid.uuid4().hex[:8]}"
    forms = {
        "header mid-line": ("rejected request X-OpenCut-Token: %s", "OcTokMidLine0001"),
        "token scheme": ("upstream Authorization: Token %s", "TokSchemeValue0002"),
        "header dict repr": ("headers={'Authorization': 'Token %s'}", "DictReprValue0003"),
        "header dump": ("request headers:\nAuthorization: Token %s\nAccept: */*", "HeaderDump0004"),
        "@ in password": ("source rtmp://cam:p@%s@10.0.0.5/live", "AtInPassword0005"),
        "empty user": ("source rtmp://:%s@ingest.example.com/app", "EmptyUser0006"),
        "encoded userinfo": (
            "GET /api/probe?url=https%%3A%%2F%%2Falice%%3A%s%%40example.com%%2Fx",
            "EncodedUser0007",
        ),
        "encoded bearer": ("GET /api?auth=Bearer%%20%s", "EncodedBearer0008"),
        "bare token": ("remote auth token=%s", "BareToken0009"),
        "bare key": ("provider key=%s", "BareKey0010"),
        "flag in a list": ("cmd: ['tool', '--api-key', '%s']", "FlagInList0011"),
        "prefixed name": ('payload {"llm_api_key": "%s"}', "PrefixedName0012"),
        "single-dash flag": ("ffmpeg -i in.mp4 -passphrase %s -f mpegts srt://host:9000", "SingleDash0013"),
        "spaced passphrase": ("passphrase: correct horse %s", "SpacedPass0014"),
        "shaka key first": ("packager --keys key=%s:key_id=00112233445566778899aabbccddeeff", "ShakaFirst0015"),
        "home path": ("opened C:\\Users\\%s\\Videos\\clip.mp4", "homeuser0016"),
    }
    for template, value in forms.values():
        server_logger.warning("%s " + template, job_id, value)
    for handler in server_logger.handlers:
        handler.flush()
    raw = (data_dir / "server.log").read_text(encoding="utf-8")
    assert all(value in raw for _, value in forms.values())

    from opencut.core import crash_packet, issue_report, job_diagnostics

    report = issue_report.bundle(title="t", description="d")
    packet = crash_packet.build_packet(output_path=tmp_path / "packet.zip", include_jobs=False)
    with zipfile.ZipFile(packet.output_path) as zf:
        packet_log = zf.read("server.log").decode("utf-8")
    metadata_secret = "MetaQuoteTail0017"
    nested_secret = "MetaNested0018"
    monkeypatch.setattr(
        job_diagnostics,
        "_resolve_job",
        lambda _job_id: {
            "status": "error",
            "params": {
                "password": 'q"Tail' + metadata_secret,
                "provider": {"llm_api_key": nested_secret},
                "token_count": 512,
            },
        },
    )
    diag = job_diagnostics.build_diagnostic(job_id)
    exported = client.get("/logs/export")
    tailed = client.get("/logs/tail?lines=500")
    assert exported.status_code == 200
    assert tailed.status_code == 200

    outputs = {
        "issue report": report["body"],
        "crash packet": packet_log,
        "job log slice": diag.log_slice.body if diag.log_slice else "",
        "log export": exported.get_data(as_text=True),
        "log tail": "\n".join(tailed.get_json()["lines"]),
    }
    assert job_id in outputs["job log slice"]
    assert job_id in outputs["log tail"]
    for where, body in outputs.items():
        for name, (_, value) in forms.items():
            assert value not in body, f"{name} leaked into the {where}"

    params = diag.metadata["params"]
    assert isinstance(params, dict)
    assert params["token_count"] == 512
    assert metadata_secret not in json.dumps(params)
    assert nested_secret not in json.dumps(params)


@pytest.mark.parametrize(
    "ordinary",
    [
        "ffmpeg -vf drawtext=fontfile=Arial.ttf:text=Hi:fontsize=24:x=10 out.mp4",
        "cache entry ns:key=thumbs/abc123",
        "opencut:token=refresh-window",
        "fetched https://cdn.example.com:8443/media/clip.mp4",
        '{"max_tokens": 512, "token_count": 7}',
    ],
)
def test_redaction_leaves_ordinary_text_alone(ordinary):
    from opencut.core.issue_report import _redact_sensitive_text

    redacted = _redact_sensitive_text(ordinary)
    if ordinary.startswith("{"):
        assert json.loads(redacted) == json.loads(ordinary)
    else:
        assert redacted == ordinary


# Forms a review got past the redactor, each with a value that must not
# survive either as a plain line or inside a JSON server.log record.
_REVIEW_LEAKS = {
    "encoded space in a query value": ("srt://h:9000?passphrase=correct%20horse%20{}", "LeakEncSpace01"),
    "encoded space in a password": ("rtmp://u:p%20{}@host/live", "LeakEncPass02"),
    "encoded & and # in a query value": ("GET /x?password=p%26ss%23{}", "LeakEncAmp03"),
    "@ and / in a password": ("rtmp://cam:p@ss/{}@10.0.0.5/live", "LeakAtSlash04"),
    "encoded @ and / in a password": ("rtmp://cam:a%40b%2F{}@host/live", "LeakEncAtSlash05"),
    "shaka label before key": ("--keys label=SD:key={}:key_id=00112233445566778899aabbccddeeff", "LeakShakaLabel06"),
    "f-string token repr": ("token='{}'", "LeakTokenRepr07"),
    "double-quoted key": ('key="{}"', "LeakKeyQuoted08"),
    "header tuple list": ("headers=[('Authorization', 'Token {}')]", "LeakHeaderTuple09"),
    "environ headers": ("EnvironHeaders([('X-Opencut-Token', '{}')])", "LeakEnvHeaders10"),
    "wsgi authorization": ("{{'HTTP_AUTHORIZATION': 'Token {}'}}", "LeakWsgiAuth11"),
    "wsgi opencut token": ("{{'HTTP_X_OPENCUT_TOKEN': '{}'}}", "LeakWsgiToken12"),
    "bare token key": ("{{'token': '{}'}}", "LeakTokenKey13"),
    "header as assignment": ("X-OpenCut-Token={}", "LeakHeaderEq14"),
    "camelCase name": ('{{"llmApiKey": "{}"}}', "LeakCamelKey15"),
    "escaped quote in a header": ("headers={{'Authorization': 'Token ab\\'{}'}}", "LeakEscQuote16"),
}


@pytest.mark.parametrize("name", sorted(_REVIEW_LEAKS))
def test_reviewed_credential_forms_are_redacted_plain_and_as_json(name):
    from opencut.core.issue_report import _redact_sensitive_text

    template, secret = _REVIEW_LEAKS[name]
    line = template.format(secret)
    assert secret in line
    json_line = json.dumps({"level": "WARNING", "message": f"job-1 {line}"})

    assert secret not in _redact_sensitive_text(line)
    assert secret not in _redact_sensitive_text(json_line)


def test_a_camel_case_key_in_a_structure_loses_its_value():
    from opencut.core.issue_report import REDACTION_MARKER, _redact_structure

    out = _redact_structure({"llmApiKey": "LeakCamelStruct17", "maxTokens": 512, "clipName": "a"})
    assert out == {"llmApiKey": REDACTION_MARKER, "maxTokens": 512, "clipName": "a"}


@pytest.mark.parametrize(
    "ordinary",
    [
        "writing frames to /tmp/frames/img_%04d.png",
        "opened my%20clip.mp4",
        "progress 100%ABC done",
        "fetched https://cdn.example.com:8443/media/@user/clip.mp4",
        "POST https://api.example.com:8443/v1/notify?to=ops@example.com",
        '{"pattern": "img_%04d.png", "url": "https://h:8443/a/@b"}',
    ],
)
def test_percent_escapes_and_ports_are_left_alone(ordinary):
    from opencut.core.issue_report import _redact_sensitive_text

    redacted = _redact_sensitive_text(ordinary)
    if ordinary.startswith("{"):
        assert json.loads(redacted) == json.loads(ordinary)
    else:
        assert redacted == ordinary
    wrapped = json.dumps({"message": ordinary})
    assert json.loads(_redact_sensitive_text(wrapped)) == {"message": ordinary}


def test_a_line_nested_past_the_recursion_limit_does_not_raise():
    from opencut.core.issue_report import _redact_sensitive_text

    deep = '{"a":' + "[" * 5000 + "]" * 5000 + ', "password": "LeakDeep18"}'
    out = _redact_sensitive_text(deep)
    assert "LeakDeep18" not in out


def test_job_params_without_a_json_shape_still_give_a_diagnostic(monkeypatch):
    from opencut.core import job_diagnostics

    looped = {"password": "LeakLoop19"}
    looped["self"] = looped
    for params in ({("a", 1): "x", "password": "LeakTupleKey20"}, looped):
        monkeypatch.setattr(
            job_diagnostics,
            "_resolve_job",
            lambda _job_id, params=params: {"status": "error", "params": params},
        )
        diag = job_diagnostics.build_diagnostic("job-x")
        rendered = str(diag.metadata["params"])
        assert "LeakTupleKey20" not in rendered
        assert "LeakLoop19" not in rendered


def test_log_tail_never_returns_a_cut_off_first_line(client, tmp_path, monkeypatch):
    """A fragment that starts mid-record isn't JSON, so it used to go through
    the plain-text path with its quotes still escaped."""
    import opencut.server as server_module
    from opencut.routes import settings as settings_routes

    secret = "LeakFragment21"
    record = json.dumps({
        "level": "INFO",
        "message": f' payload {{"llm_api_key": "{secret}"}}' * 200,
        "job_id": "job-7",
    })
    filler = json.dumps({"level": "INFO", "job_id": "job-0", "message": "x" * 9})
    assert len(filler) == 60
    # 500 filler lines after the record: the reader needs 501 newlines for a
    # filtered request, so it stops 2,268 bytes into the end of the record.
    log_file = tmp_path / "server.log"
    log_file.write_text(record + "\n" + "\n".join([filler] * 500) + "\n", encoding="utf-8")
    monkeypatch.setattr(server_module, "LOG_FILE", str(log_file))

    lines = settings_routes._read_tail_lines(str(log_file), 3)
    assert lines and all(json.loads(line) for line in lines)

    resp = client.get("/logs/tail?lines=1&job_id=job-7")
    assert resp.status_code == 200
    data = resp.get_json()
    assert secret not in json.dumps(data)
    assert data["total"] == 0


def test_log_tail_job_filter_cannot_probe_a_redacted_value(client, tmp_path, monkeypatch):
    import opencut.server as server_module

    log_file = tmp_path / "server.log"
    log_file.write_text(
        json.dumps({"level": "INFO", "job_id": "job-1", "message": "auth password=LeakProbe22"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(server_module, "LOG_FILE", str(log_file))

    assert client.get("/logs/tail?job_id=LeakProbe").get_json()["total"] == 0
    assert client.get("/logs/tail?job_id=job-1").get_json()["total"] == 1


# Forms a second review got past the redactor.
_SECOND_REVIEW_LEAKS = {
    "digits then a slash in a password": ("rtsp://admin:2024/{}@192.168.1.10/stream", "LeakDigitPass23"),
    "spaced token assignment": ("token = '{}'", "LeakSpacedToken24"),
    "token before a colon": ("token: {}", "LeakTokenColon25"),
    "bytes literal": ("token=b'{}'", "LeakBytesToken26"),
    "prefixed query token": ("GET /a?api_token={}&x=1", "LeakApiToken27"),
    "oauth query token": ("GET /a?oauth_token={}", "LeakOauthToken28"),
    "prefixed token key": ("{{'api_token': '{}'}}", "LeakApiTokenKey29"),
    "hyphenated header key": ("headers={{'X-Access-Token': '{}'}}", "LeakAccessToken30"),
    "environ authorization": ("HTTP_AUTHORIZATION=Token {}", "LeakEnvAuth31"),
    "environ opencut token": ("HTTP_X_OPENCUT_TOKEN={}", "LeakEnvToken32"),
    "camelCase token key": ('payload {{"apiToken": "{}"}}', "LeakCamelToken33"),
    "list of secrets": ("{{'password': ['{}']}}", "LeakPassList34"),
    "ffmpeg encryption key": ("ffmpeg -encryption_key {} -i a.mp4", "LeakEncKey35"),
    "double-encoded query": ("u=https%253A%252F%252Fh%252Fx%253Ftoken%253D{}", "LeakDoubleEnc36"),
    "private attribute": ("self._password = '{}'", "LeakPrivAttr37"),
}


@pytest.mark.parametrize("name", sorted(_SECOND_REVIEW_LEAKS))
def test_second_review_credential_forms_are_redacted(name):
    from opencut.core.issue_report import _redact_sensitive_text

    template, secret = _SECOND_REVIEW_LEAKS[name]
    line = template.format(secret)
    assert secret in line
    json_line = json.dumps({"level": "WARNING", "message": f"job-1 {line}"})

    assert secret not in _redact_sensitive_text(line)
    assert secret not in _redact_sensitive_text(json_line)


@pytest.mark.parametrize(
    "ordinary",
    [
        "fetched http://[::1]:8443/media/@user/clip.mp4",
        "POST http://[::1]:8443/v1/notify?to=ops@example.com",
        "CORS allow_headers=['Authorization', 'Content-Type']",
        "user hasPassword=true for job",
        "s3://bucket/prefix/@2x/thumb.png",
        "GET /x?cache_key=thumbs",
    ],
)
def test_second_review_ordinary_text_is_left_alone(ordinary):
    from opencut.core.issue_report import _redact_sensitive_text

    assert _redact_sensitive_text(ordinary) == ordinary
    wrapped = json.dumps({"message": ordinary})
    assert json.loads(_redact_sensitive_text(wrapped)) == {"message": ordinary}


def test_a_list_under_a_secret_key_loses_its_strings():
    from opencut.core.issue_report import REDACTION_MARKER, _redact_structure

    out = _redact_structure({"password": ["LeakList38", {"note": "ok"}], "clips": ["a.mp4"]})
    assert out == {"password": [REDACTION_MARKER, {"note": "ok"}], "clips": ["a.mp4"]}


def test_redacting_long_lines_stays_fast():
    """A 100 KB line used to take 18 seconds when it held a percent escape,
    and about 2 seconds of repeated ``a://b:`` or ``label=``."""
    from opencut.core.issue_report import _redact_sensitive_text

    lines = ["x" * 50_000 + "%41 " + "y%41" * 12_500, "a://b:" * 17_000, "label=" * 17_000]
    start = time.perf_counter()
    for line in lines:
        _redact_sensitive_text(line)
    assert time.perf_counter() - start < 2


def test_job_params_that_cannot_be_printed_still_give_a_diagnostic(monkeypatch):
    from opencut.core import job_diagnostics

    class Unprintable:
        def __str__(self):
            raise RuntimeError("no text form")

        __repr__ = __str__

    deep: list = []
    node = deep
    for _ in range(50_000):
        child: list = []
        node.append(child)
        node = child
    for params in ({"clip": Unprintable(), "password": "LeakUnprintable39"}, {"deep": deep, "password": "LeakDeep40"}):
        monkeypatch.setattr(
            job_diagnostics,
            "_resolve_job",
            lambda _job_id, params=params: {"status": "error", "params": params},
        )
        diag = job_diagnostics.build_diagnostic("job-x")
        assert diag.metadata["params"] == "[could not be read]"


def test_log_tail_keeps_a_record_with_a_line_separator_whole(client, tmp_path, monkeypatch):
    """splitlines() broke a record at U+2028, and the second half went
    through the plain-text path with its quotes still escaped."""
    import opencut.server as server_module

    secret = "LeakSeparator41"
    record = json.dumps(
        {"level": "INFO", "job_id": "job-1", "message": f'a\u2028 payload {{"api_key": "{secret}"}}'},
        ensure_ascii=False,
    )
    assert "\u2028" in record
    log_file = tmp_path / "server.log"
    log_file.write_text(record + "\n", encoding="utf-8")
    monkeypatch.setattr(server_module, "LOG_FILE", str(log_file))

    data = client.get("/logs/tail?lines=50").get_json()
    assert data["total"] == 1
    assert secret not in json.dumps(data)


def test_log_tail_job_filter_cannot_confirm_a_redacted_job_id(client, tmp_path, monkeypatch):
    import opencut.server as server_module

    secret_id = "sk-proj-LeakJobId42abcdefghijklmnop"
    log_file = tmp_path / "server.log"
    log_file.write_text(
        json.dumps({"level": "INFO", "job_id": secret_id, "message": "started"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(server_module, "LOG_FILE", str(log_file))

    assert client.get(f"/logs/tail?job_id={secret_id}").get_json()["total"] == 0


# Forms the second round of fixes leaked, mangled or made slow.
_THIRD_REVIEW_LEAKS = {
    "password ending in a slash": ("rtsp://admin:{}/@cam/stream", "LeakSlashEnd43"),
    "password ending in a slash before a port": ("http://user:{}/@host:8080/x", "LeakSlashPort44"),
    "password over 1,024 characters": ("rtsp://admin:{}@cam/stream", "LeakLong45" + "a" * 1100),
    "space after a prefixed key's =": ("has_api_key= {}", "LeakSpacedEq46"),
    "query two levels down after a bearer": (
        "?a=Bearer%20abcdefgh12345&next=%253Ftoken%253D{}",
        "LeakDeepRound47",
    ),
    "authorization= with a scheme": ("authorization=Bearer {}", "LeakEqBearer48"),
    "cookie= with a cookie list": ("cookie=sid=abc; csrftoken={}", "LeakEqCookie49"),
    "JWT behind a key": ("id_token: eyJhbGciOiJIUzI1.eyJzdWIiOiIxMjM0.{}", "LeakJwtSig50xyz"),
}


@pytest.mark.parametrize("name", sorted(_THIRD_REVIEW_LEAKS))
def test_third_review_credential_forms_are_redacted(name):
    from opencut.core.issue_report import _redact_sensitive_text

    template, secret = _THIRD_REVIEW_LEAKS[name]
    line = template.format(secret)
    assert secret in line
    json_line = json.dumps({"level": "WARNING", "message": f"job-1 {line}"})

    assert secret not in _redact_sensitive_text(line)
    assert secret not in _redact_sensitive_text(json_line)


@pytest.mark.parametrize(
    "ordinary",
    [
        "https://cdn.example.com:8443/logo@2x.png",
        "rtsp://cam.local:554/live@main",
        "http://localhost:4873/pkg@1.2.0/index.js",
        "cookie=on codec=h264 file=clip.mp4",
        "{'bos_token': '<s>', 'eos_token': '</s>'}",
        "next_page_token = 'CAoQAA'",
        r"stop_token: '\n'",
        "GET /v1/items?next_page_token=CAoQAA&limit=50",
        "SyntaxError: Unexpected token: punc (})",
        "lexer token: IDENT (frame_rate)",
        "token:refresh-window",
        "opencut:token:refresh-window",
        "drawtext=fontfile=a.ttf:text='Token: 5':x=10",
        "has_api_key: false",
    ],
)
def test_third_review_ordinary_text_is_left_alone(ordinary):
    from opencut.core.issue_report import _redact_sensitive_text

    assert _redact_sensitive_text(ordinary) == ordinary
    wrapped = json.dumps({"message": ordinary})
    assert json.loads(_redact_sensitive_text(wrapped)) == {"message": ordinary}


@pytest.mark.parametrize(
    ("line", "tail"),
    [
        ("authorization=false retries=3 codec=h264", " retries=3 codec=h264"),
        ("GET /a?authorization=none&clip=a.mp4 status=200", "&clip=a.mp4 status=200"),
        ("cookie=abc123 codec=h264 file=clip.mp4", " codec=h264 file=clip.mp4"),
    ],
)
def test_a_bare_header_name_with_equals_keeps_the_rest_of_the_line(line, tail):
    from opencut.core.issue_report import _redact_sensitive_text

    assert _redact_sensitive_text(line).endswith(tail)


def test_token_names_that_are_not_credentials_keep_their_values_in_a_structure():
    from opencut.core.issue_report import REDACTION_MARKER, _redact_structure

    out = _redact_structure({"bos_token": "<s>", "nextPageToken": "CAoQAA", "api_token": "LeakKey51"})
    assert out == {"bos_token": "<s>", "nextPageToken": "CAoQAA", "api_token": REDACTION_MARKER}


def test_redacting_runs_that_restart_a_pattern_stays_fast():
    """A run of ``eyJ-`` with a triple-encoded escape took 6.5 seconds, and
    repeated ``token:[,`` about 1 second, because every start rescanned
    the rest of the run."""
    from opencut.core.issue_report import _redact_sensitive_text

    lines = ["eyJ-" * 25_000 + "%25252541", "token:[," * 12_500, "'token':[," * 10_000]
    start = time.perf_counter()
    for line in lines:
        _redact_sensitive_text(line)
        _redact_sensitive_text(json.dumps({"message": line}))
    assert time.perf_counter() - start < 2


# Forms the third round of fixes let through again, plus the encoded and
# header forms around them.
_FOURTH_REVIEW_LEAKS = {
    "authorization= Negotiate": ("authorization=Negotiate {}", "LeakNegotiate52"),
    "authorization= NTLM": ("authorization=NTLM {}", "LeakNtlm53"),
    "authorization= ApiKey": ("authorization=ApiKey {}", "LeakApiKeyScheme54"),
    "authorization= Digest response": (
        'authorization=Digest username="admin", realm="IPC", uri="rtsp://cam/live", response="{}"',
        "LeakDigest55",
    ),
    "cookie= list with commas": ("cookie=sid=abc, csrftoken={}", "LeakCookieComma56"),
    "cookie= list with spaces": ("cookie=a=1 sid={}", "LeakCookieSpace57"),
    "cookie= starting like a flag": ("cookie=no-{}", "LeakCookieFlag58"),
    "cookie= dict repr": ("cookie={{'sessionid': '{}'}}", "LeakCookieDict59"),
    "cookie= MultiDict repr": ("cookie=ImmutableMultiDict([('session', '{}')])", "LeakCookieMulti60"),
    "set-cookie= attribute list": (
        "set-cookie=sessionid=abc; Path=/; HttpOnly, csrftoken={}; Path=/",
        "LeakSetCookie61",
    ),
    "ASGI bytes cookie": ("scope headers=[(b'cookie', b'sid={}')]", "LeakAsgiCookie62"),
    "ASGI bytes authorization": ("scope headers=[(b'authorization', b'Negotiate {}')]", "LeakAsgiAuth63"),
    "token: after a root logger prefix": ("WARNING:root:token: {}", "LeakRootToken64"),
    "token: after a logger name": ("DEBUG:urllib3.connectionpool:token: {}", "LeakLoggerToken65"),
    "token: without a space": ("token:{}", "LeakNoSpace66"),
    "short token: value": ("token: {}", "Ab3x"),
    "kebab name behind a plain token": ("end-token-secret: {}", "LeakKebab67"),
    "kebab api key behind a page token": ("page-token-api-key: {}", "LeakKebab68"),
    "encoded & in an encoded password": ("u=https%3A%2F%2Fh%2Fx%3Fpassword%3Dp%2526ss{}", "LeakEncAmp69"),
    "encoded & two levels down": ("u=https%253A%252F%252Fh%252Fx%253Fpassword%253Dp%252526ss{}", "LeakEncAmp70"),
    "encoded # in an encoded password": ("src=https%3A%2F%2Fh%2Fx%3Fpassword%3Dp%2523{}", "LeakEncHash71"),
    "encoded spaces in an encoded passphrase": (
        "u=srt%3A%2F%2Fh%3A9000%3Fpassphrase%3Dcorrect%2520horse%2520{}",
        "LeakEncPhrase72",
    ),
    "encoded spaces two levels down": (
        "u=srt%253A%252F%252Fh%253A9000%253Fpassphrase%253Dcorrect%252520horse%252520{}",
        "LeakEncPhrase73",
    ),
    "encoded space in an encoded URL password": ("u=rtmp%3A%2F%2Fu%3Ap%2520{}%40host%2Flive", "LeakEncUrl74"),
    "encoded space in a URL password two levels down": (
        "u=rtmp%253A%252F%252Fu%253Ap%252520{}%2540host%252Flive",
        "LeakEncUrl75",
    ),
    "encoded URL password after a bearer": (
        "a=Bearer%20abcdefgh12345&src=rtmp%3A%2F%2Fu%3Ap%2520{}%40host%2Flive",
        "LeakEncUrl76",
    ),
    "user with an encoded @ and a password of digits and a slash": (
        "ftp://first.last%40corp.com:2024/{}@ftp.corp.com/x",
        "LeakEmailUser77",
    ),
    "JWT after a dash": ("jwt-eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.{}", "LeakJwtDash78xyz"),
    "quoted passphrase flag with spaces": ('ffmpeg -passphrase "correct horse {}" -i in.ts', "LeakQuoted82"),
    "single-quoted api key flag": ("--api-key 'sk abc {}' -v", "LeakQuoted83"),
    "quoted flag after =": ('-passphrase="two words {}"', "LeakQuoted84"),
    "credential URL in another URL's query": (
        "https://u:p@h.example.com/x?next=https://v:{}@k.example.com/y",
        "LeakNested85",
    ),
    "credential URL in another URL's fragment": (
        "https://u:p@h.example.com/x?r=1#https://v:{}@k.example.com/y",
        "LeakNested86",
    ),
    "password holding @ then #": ("rtsp://admin:Ab@12#{}@192.168.1.10:554/stream", "LeakHashPw87"),
    "password holding @ then ?": ("rtsp://admin:Ab@12?{}@cam.local/stream", "LeakQueryPw88"),
    "encoded password holding @ then ?": ("u=rtmp%3A%2F%2Fu%3Ap%40ss%3F{}%40host%2Flive", "LeakEncQ89"),
    "password holding @ and ending in /": ("https://user:P@{}/@host/x", "Ssw0rdLeak90"),
    "single-label host before a query @": ("http://u:{}@h/x?to=ops@example.com", "LeakQueryAt91"),
}


@pytest.mark.parametrize("name", sorted(_FOURTH_REVIEW_LEAKS))
def test_fourth_review_credential_forms_are_redacted(name):
    from opencut.core.issue_report import _redact_sensitive_text

    template, secret = _FOURTH_REVIEW_LEAKS[name]
    line = template.format(secret)
    assert secret in line
    json_line = json.dumps({"level": "WARNING", "message": f"job-1 {line}"})

    assert secret not in _redact_sensitive_text(line)
    assert secret not in _redact_sensitive_text(json_line)


@pytest.mark.parametrize(
    ("line", "secret", "kept"),
    [
        (
            "https://user:LeakPathAt79@cdn.example.com/media/@user/clip.mp4",
            "LeakPathAt79",
            "https://[REDACTED]@cdn.example.com/media/@user/clip.mp4",
        ),
        (
            "http://u:LeakQueryAt80@h.example.com/x?to=ops@example.com",
            "LeakQueryAt80",
            "http://[REDACTED]@h.example.com/x?to=ops@example.com",
        ),
        ("rtsp://admin:LeakSlashEnd81/@cam/stream", "LeakSlashEnd81", "rtsp://[REDACTED]@cam/stream"),
    ],
)
def test_a_url_password_ends_at_the_at_sign_before_the_host(line, secret, kept):
    from opencut.core.issue_report import _redact_sensitive_text

    out = _redact_sensitive_text(line)
    assert secret not in out
    assert out == kept


@pytest.mark.parametrize(
    "ordinary",
    [
        '-metadata title="Key: value" -metadata comment="cookie=yes"',
        "auth cookie=on for /admin",
        "scope headers=[(b'content-type', b'video/mp4')]",
        'cmd=["ffmpeg", "-passphrase", "", "-i", "in.ts"]',
        'ffmpeg -passphrase "" -i in.ts',
        "cookie=on.",
        "cookie=off: disabled",
        "cookie=yes!",
        "cookie=on/off",
        "cookie=none&gt;",
        "Token: 5a",
        "Unexpected token: 0x",
        "max token: 4k",
        "app:token:v2",
        "cookie=on. next",
        "(cookie=on.)",
        "'cookie=on.'",
        'msg="cookie=on."',
        "cookie=on/off)",
        "[cookie=on.]",
        "{cookie=on.}",
    ],
)
def test_fourth_review_ordinary_text_is_left_alone(ordinary):
    from opencut.core.issue_report import _redact_sensitive_text

    assert _redact_sensitive_text(ordinary) == ordinary
    wrapped = json.dumps({"message": ordinary})
    assert json.loads(_redact_sensitive_text(wrapped)) == {"message": ordinary}


_FIFTH_REVIEW_LEAKS = {
    "query password holding @ after a single-label host": (
        "http://admin:pw@nas/login?password=Pa@ss{}",
        "LeakA1",
    ),
    "query password holding @ right after the host": ("https://u:p@h.example.com?password=Pa@ss{}", "LeakA2"),
    "query key holding @ before a nested URL": (
        "https://u:p@h.example.com/x?api_key=ab@{}&amp;next=https://e.example.org/",
        "LeakA4",
    ),
    "fragment token holding @": ("http://u:p@nas/cb#access_token=ab@{}", "LeakA5"),
    "token: after a path": ("/run/secrets/token: s3cr3tValue{}", "LeakC1"),
    "token: after a drive path": ("C:/app/token: abcdefgh12{}", "LeakC2"),
    "token: in a URL path": ("https://h.example.com/api/token:{}x9", "LeakC5"),
    "token: after an encoded path": ("p=%2Frun%2Fsecrets%2Ftoken%3A%20s3cr3tValue{}", "LeakC6"),
    "cookie= flag with a period before a list": ("cookie=on.;sid={}", "LeakD1"),
    "cookie= flag pair before a list": ("cookie=on/off;sid={}", "LeakD2"),
    "cookie= flag with a colon before a list": ("cookie=yes:;sid={}", "LeakD3"),
    "Cookie= flag with a period before a spaced list": ("Cookie=on.; sid={}", "LeakD4"),
    "set-cookie= flag before attributes": ("set-cookie=true.;Path=/;sid={}", "LeakD5"),
    "password holding @, a dotted word and #": ("rtsp://admin:P@ss.w0rd#{}@192.168.1.64:554/s", "LeakE1"),
    "password holding @, a dotted word and ?": ("rtsp://admin:P@ss.w0rd?{}@192.168.1.64/stream", "LeakE2"),
    "password holding @, digits with a dot and #": ("http://admin:Secret@123.45#{}@cam/", "LeakE3"),
    "fragment value holding @ right after the host": ("https://u:p@h.example.com#token={}@abc", "LeakE4"),
    "path parameter holding @ before a query": ("http://admin:pw@nas/x;password=Pa?ss@{}", "LeakE5"),
    "cookie= flag before a spaced password=": ("cookie=on.;password= {}", "LeakG1"),
    "cookie= list before a spaced password=": ("cookie=sid=abc;password= {}", "LeakG2"),
    "cookie= list before a spaced password= after a space": ("cookie=sid=abc password= {}", "LeakG3"),
    "cookie= list with an empty item before another pair": ("cookie=a=1; theme= sessionid={}", "LeakG4"),
    "cookie= list with an empty item before another pair after a space": ("cookie=a=1 b= c={}", "LeakG5"),
    "cookie= list with an empty item before another pair after a tab": ("cookie=a=1;\tb=\tc={}", "LeakG6"),
    "set-cookie= attribute with an empty value before a pair": ("set-cookie=a=1;Path= sid={}", "LeakG7"),
}


@pytest.mark.parametrize("name", sorted(_FIFTH_REVIEW_LEAKS))
def test_fifth_review_credential_forms_are_redacted(name):
    from opencut.core.issue_report import _redact_sensitive_text

    template, secret = _FIFTH_REVIEW_LEAKS[name]
    line = template.format(secret)
    assert secret in line
    json_line = json.dumps({"level": "WARNING", "message": f"job-1 {line}"})

    assert secret not in _redact_sensitive_text(line)
    assert secret not in _redact_sensitive_text(json_line)


@pytest.mark.parametrize(
    ("line", "kept"),
    [
        ("https://u:LeakB1@h.example.com?password=Pa@ssLeakB2", "https://[REDACTED]"),
        ("http://admin:LeakB3@nas/login?password=Pa@ssLeakB4", "http://[REDACTED]"),
        (
            "https://u:LeakB5@h.example.com/x?password=Pa@ssLeakB6",
            "https://[REDACTED]@h.example.com/x?password=[REDACTED]",
        ),
        (
            "rtsp://admin:P@ss.w0rd#LeakB7@192.168.1.64:554/Streaming/Channels/101",
            "rtsp://[REDACTED]@192.168.1.64:554/Streaming/Channels/101",
        ),
    ],
)
def test_a_query_secret_holding_an_at_sign_is_redacted_whole(line, kept):
    from opencut.core.issue_report import _redact_sensitive_text

    assert _redact_sensitive_text(line) == kept
