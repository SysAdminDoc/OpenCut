"""Host-header trust policy (DNS-rebinding defence).

OpenCut trusts loopback peers, so a rebound DNS name reaches the API from a
loopback socket while keeping the attacker's origin. These tests pin the
policy in ``opencut/trusted_hosts.py`` and the gate wired into ``create_app``.
"""

from __future__ import annotations

import pytest

from opencut.config import OpenCutConfig
from opencut.server import create_app
from opencut.trusted_hosts import (
    build_trusted_hosts,
    is_loopback_hostname,
    is_trusted_host,
    normalize_hostname,
    split_host_port,
)

ATTACKER_HOST = "attacker.invalid:5680"


def _app(**config_kwargs):
    app = create_app(config=OpenCutConfig(**config_kwargs), testing=True)
    app.config["TESTING"] = True
    return app


# --------------------------------------------------------------------------
# Parsing / classification
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("localhost", ("localhost", None)),
        ("localhost:5679", ("localhost", "5679")),
        ("127.0.0.1:5679", ("127.0.0.1", "5679")),
        ("[::1]", ("::1", None)),
        ("[::1]:5679", ("::1", "5679")),
        ("::1", ("::1", None)),
        ("", ("", None)),
        ("[::1", ("", None)),
    ],
)
def test_split_host_port(raw, expected):
    assert split_host_port(raw) == expected


@pytest.mark.parametrize(
    "hostname",
    ["localhost", "LOCALHOST", "localhost.", "127.0.0.1", "127.5.4.3", "::1", "[::1]"],
)
def test_loopback_forms_are_recognized(hostname):
    assert is_loopback_hostname(hostname) is True


@pytest.mark.parametrize("hostname", ["attacker.invalid", "192.168.1.5", "example.com", ""])
def test_non_loopback_forms_are_not_loopback(hostname):
    assert is_loopback_hostname(hostname) is False


def test_normalize_hostname_strips_root_dot_zone_and_brackets():
    assert normalize_hostname("Example.COM.") == "example.com"
    assert normalize_hostname("[fe80::1%eth0]") == "fe80::1"


# --------------------------------------------------------------------------
# Allowlist construction
# --------------------------------------------------------------------------


def test_loopback_names_are_implicit():
    assert "localhost" in build_trusted_hosts()


def test_named_bind_host_is_trusted_but_wildcards_are_not():
    assert "studio.lan" in build_trusted_hosts(bind_host="studio.lan")
    assert build_trusted_hosts(bind_host="0.0.0.0") == build_trusted_hosts()
    assert build_trusted_hosts(bind_host="::") == build_trusted_hosts()


def test_configured_entries_drop_ports_and_case():
    allowed = build_trusted_hosts(configured=["Studio.LAN:5679", " ", "renderbox"])
    assert "studio.lan" in allowed
    assert "renderbox" in allowed


# --------------------------------------------------------------------------
# Trust decisions
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw_host",
    [
        "localhost",
        "localhost:5679",
        "127.0.0.1:5679",
        "127.0.0.1",
        "[::1]:5679",
        "::1",
    ],
)
def test_loopback_hosts_and_ports_are_trusted(raw_host):
    assert is_trusted_host(raw_host, build_trusted_hosts()) is True


@pytest.mark.parametrize(
    "raw_host",
    [ATTACKER_HOST, "attacker.invalid", "opencut.attacker.invalid", "", "localhost:notaport"],
)
def test_unconfigured_hosts_are_rejected(raw_host):
    assert is_trusted_host(raw_host, build_trusted_hosts()) is False


def test_configured_host_is_trusted_without_widening_defaults():
    allowed = build_trusted_hosts(configured=["studio.lan"])
    assert is_trusted_host("studio.lan:5679", allowed) is True
    assert is_trusted_host("attacker.invalid:5679", allowed) is False


def test_subtree_entry_matches_only_that_subtree():
    allowed = build_trusted_hosts(configured=[".studio.lan"])
    assert is_trusted_host("render.studio.lan", allowed) is True
    assert is_trusted_host("studio.lan", allowed) is True
    assert is_trusted_host("notstudio.lan", allowed) is False


def test_ip_literals_need_the_remote_bind_opt_in():
    allowed = build_trusted_hosts()
    assert is_trusted_host("192.168.1.5:5679", allowed) is False
    assert is_trusted_host("192.168.1.5:5679", allowed, allow_ip_literals=True) is True


# --------------------------------------------------------------------------
# Middleware behaviour
# --------------------------------------------------------------------------


def test_loopback_health_still_issues_a_csrf_token():
    client = _app().test_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json().get("csrf_token")


def test_attacker_host_and_matching_origin_is_rejected_without_a_token():
    """The live DNS-rebinding shape: loopback peer, attacker Host + Origin."""
    client = _app().test_client()
    resp = client.get(
        "/health",
        headers={"Host": ATTACKER_HOST, "Origin": f"http://{ATTACKER_HOST}"},
    )
    assert resp.status_code == 400
    body = resp.get_json()
    assert body["code"] == "UNTRUSTED_HOST"
    assert "csrf_token" not in body
    assert "attacker.invalid" not in resp.get_data(as_text=True)


def test_rejection_happens_before_csrf_processing_on_mutations():
    client = _app().test_client()
    resp = client.post(
        "/shutdown",
        headers={"Host": ATTACKER_HOST},
        json={},
    )
    assert resp.status_code == 400
    assert resp.get_json()["code"] == "UNTRUSTED_HOST"


def test_configured_trusted_host_reaches_health():
    client = _app(trusted_hosts=["studio.lan"]).test_client()
    resp = client.get("/health", headers={"Host": "studio.lan:5679"})
    assert resp.status_code == 200
    assert resp.get_json().get("csrf_token")


def test_named_bind_host_reaches_health():
    client = _app(bind_host="studio.lan").test_client()
    resp = client.get("/health", headers={"Host": "studio.lan:5679"})
    assert resp.status_code == 200


def test_remote_bind_opt_in_allows_ip_literal_hosts(monkeypatch):
    monkeypatch.setenv("OPENCUT_ALLOW_REMOTE", "1")
    client = _app(bind_host="0.0.0.0").test_client()
    ip_resp = client.get("/health", headers={"Host": "192.168.1.5:5679"})
    assert ip_resp.status_code == 200
    # Names still require explicit configuration — rebinding uses names.
    name_resp = client.get("/health", headers={"Host": ATTACKER_HOST})
    assert name_resp.status_code == 400
