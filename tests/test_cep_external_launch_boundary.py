"""CEP external launch and Node-privilege boundaries."""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PANEL_ROOT = REPO_ROOT / "extension" / "com.opencut.panel"
CEP_MAIN = PANEL_ROOT / "client" / "main.js"
CEP_UTILS = PANEL_ROOT / "client" / "panel-utils.js"
CEP_MANIFEST = PANEL_ROOT / "CSXS" / "manifest.xml"

#: Production panel sources. ``dist/`` is a build artifact and ``tests/`` may
#: legitimately reference the banned names while asserting their absence.
PRODUCTION_PANEL_SOURCES = sorted(
    path
    for path in (PANEL_ROOT / "client").rglob("*.js")
    if "dist" not in path.parts
)

#: Node/process/filesystem entry points that must not appear in the panel's
#: browser context. Matching on the *names* catches the aliased form
#: (``var localRequire = window.cep_node.require; localRequire("child_process")``)
#: that a literal ``require("child_process")`` check walks straight past.
BANNED_NODE_TOKENS = (
    "cep_node",
    "child_process",
    "require(",
    "process.env",
    "__dirname",
    "os.homedir",
)


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


@pytest.mark.parametrize("source_path", PRODUCTION_PANEL_SOURCES, ids=lambda p: p.name)
def test_production_panel_imports_no_node_modules(source_path):
    """No production panel source may reach Node, process, or the filesystem.

    Both the direct call and the aliased indirection are rejected: the panel
    previously held ``--enable-nodejs`` solely to open a log file, and the old
    check only looked for a literal ``require("child_process")``.
    """
    source = source_path.read_text(encoding="utf-8")
    for token in BANNED_NODE_TOKENS:
        assert token not in source, (
            f"{source_path.name} references {token!r}; the CEP panel runs "
            "without Node privileges. Route OS actions through a server-owned "
            "endpoint such as POST /system/open-path."
        )


def test_log_actions_use_the_server_owned_fixed_target():
    source = CEP_MAIN.read_text(encoding="utf-8")

    assert '"/system/open-path"' in source
    assert 'target: "server_log"' in source
    assert 'target: "log_dir"' in source


@pytest.mark.parametrize(
    "manifest_path",
    sorted(PANEL_ROOT.rglob("manifest.xml")),
    ids=lambda p: str(p.relative_to(PANEL_ROOT)),
)
def test_packaged_manifests_grant_no_node_privileges(manifest_path):
    manifest = manifest_path.read_text(encoding="utf-8")

    assert "--enable-nodejs" not in manifest
    assert "--mixed-context" not in manifest
    assert "CEFCommandLine" not in manifest


def test_shipped_cep_manifest_exists_and_is_privilege_free():
    """Guard against the parametrized sweep silently matching nothing."""
    assert CEP_MANIFEST.is_file()
    manifest = CEP_MANIFEST.read_text(encoding="utf-8")
    assert re.search(r"<Extension\s+Id=", manifest)
    assert "--enable-nodejs" not in manifest
