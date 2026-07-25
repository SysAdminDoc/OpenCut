"""CEP external URL launch boundaries."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CEP_MAIN = REPO_ROOT / "extension" / "com.opencut.panel" / "client" / "main.js"
CEP_UTILS = REPO_ROOT / "extension" / "com.opencut.panel" / "client" / "panel-utils.js"


def test_cep_oauth_launch_is_shell_free_and_uses_csinterface():
    source = CEP_MAIN.read_text(encoding="utf-8")

    assert 'normalizeOAuthUrl(r.auth_url)' in source
    assert "cs.openURLInDefaultBrowser(authUrl)" in source
    assert 'require("child_process")' not in source
    assert 'execFile("cmd"' not in source


def test_cep_oauth_url_policy_is_https_with_loopback_http_only():
    source = CEP_UTILS.read_text(encoding="utf-8")

    assert 'protocol === "https:"' in source
    assert 'protocol === "http:"' in source
    assert 'hostname === "localhost"' in source
    assert 'hostname === "127.0.0.1"' in source
    assert 'hostname === "[::1]"' in source
    assert "return raw;" in source
