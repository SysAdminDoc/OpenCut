"""CLI route escape-hatch tests (E13)."""

from __future__ import annotations

import json

import click
import pytest
from click.testing import CliRunner

from opencut import cli as cli_module


class _FakeResponse:
    def __init__(self, payload: dict, *, status: int = 200):
        self.payload = payload
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_match_manifest_route_accepts_parameterized_paths():
    manifest = {
        "routes": [
            {
                "rule": "/jobs/<job_id>",
                "methods": ["GET"],
                "endpoint": "jobs.job_detail",
                "blueprint": "jobs",
            }
        ]
    }

    route = cli_module._match_manifest_route("GET", "/jobs/abc123", manifest=manifest)

    assert route["rule"] == "/jobs/<job_id>"


def test_match_manifest_route_rejects_missing_route_without_network():
    with pytest.raises(click.ClickException, match="Route not found"):
        cli_module._match_manifest_route(
            "GET",
            "/not-real",
            manifest={"routes": [{"rule": "/health", "methods": ["GET"]}]},
        )


def test_cli_route_get_adds_query_without_csrf_probe(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout):
        calls.append((req, timeout))
        assert req.full_url == "http://127.0.0.1:5679/health?compact=1"
        assert req.get_method() == "GET"
        return _FakeResponse({"ok": True})

    monkeypatch.setattr(cli_module.urllib.request, "urlopen", fake_urlopen)

    result = CliRunner().invoke(
        cli_module.cli,
        ["route", "GET", "/health", "--query", "compact=1"],
    )

    assert result.exit_code == 0, result.output
    assert '"ok": true' in result.output
    assert len(calls) == 1


def test_cli_route_get_rejects_body_options_before_http(monkeypatch):
    def fake_urlopen(req, timeout):
        raise AssertionError("network should not be used for invalid GET bodies")

    monkeypatch.setattr(cli_module.urllib.request, "urlopen", fake_urlopen)

    result = CliRunner().invoke(
        cli_module.cli,
        ["route", "GET", "/health", "--data", '{"ignored":true}'],
    )

    assert result.exit_code != 0
    assert "GET routes do not send a JSON body" in result.output


def test_cli_route_post_fetches_csrf_and_sends_json(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout):
        calls.append((req, timeout))
        if isinstance(req, str):
            assert req == "http://127.0.0.1:5679/health"
            return _FakeResponse({"csrf_token": "csrf-123", "version": "1.32.0"})
        assert req.full_url == "http://127.0.0.1:5679/queue/add"
        assert req.get_method() == "POST"
        headers = {key.lower(): value for key, value in req.headers.items()}
        assert headers["x-opencut-token"] == "csrf-123"
        assert json.loads(req.data.decode("utf-8")) == {
            "endpoint": "/captions",
            "payload": {"filepath": "C:/clip.mp4"},
            "priority": 5,
        }
        return _FakeResponse({"queue_id": "q1"})

    monkeypatch.setattr(cli_module.urllib.request, "urlopen", fake_urlopen)

    result = CliRunner().invoke(
        cli_module.cli,
        [
            "route",
            "POST",
            "/queue/add",
            "--data",
            '{"endpoint":"/captions","payload":{"filepath":"C:/clip.mp4"}}',
            "--field",
            "priority=5",
        ],
    )

    assert result.exit_code == 0, result.output
    assert '"queue_id": "q1"' in result.output
    assert len(calls) == 2


def test_cli_route_rejects_unknown_manifest_route_before_http(monkeypatch):
    def fake_urlopen(req, timeout):
        raise AssertionError("network should not be used for invalid routes")

    monkeypatch.setattr(cli_module.urllib.request, "urlopen", fake_urlopen)

    result = CliRunner().invoke(cli_module.cli, ["route", "GET", "/definitely-not-a-route"])

    assert result.exit_code != 0
    assert "Route not found in manifest" in result.output


def test_cli_local_db_diagnostics_json(monkeypatch):
    monkeypatch.setattr(
        cli_module,
        "_local_db_diagnostics_payload",
        lambda: {
            "count": 1,
            "stores": [
                {
                    "store": "jobs",
                    "exists": True,
                    "files": {
                        "database": {"bytes": 4096},
                        "wal": {"bytes": 0},
                    },
                    "page_count": 1,
                    "freelist_count": 0,
                    "recommended_action": "ok",
                }
            ],
        },
    )

    result = CliRunner().invoke(cli_module.cli, ["local-db-diagnostics", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["count"] == 1
    assert payload["stores"][0]["store"] == "jobs"
    assert payload["stores"][0]["recommended_action"] == "ok"
