"""F147 — opencut-mcp-server registry manifest tests.

The manifest is a small JSON file that mirrors the entry the upstream
`modelcontextprotocol/servers` directory will list. These tests pin:

1. The committed manifest matches the live MCP tool catalogue so
   the upstream listing cannot silently drift.
2. The manifest shape is suitable for registry consumption (name,
   transports, install command, tool list, license).
3. The generator script's CLI contract: --check exits non-zero on
   drift, exits 0 in sync.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from opencut import __version__
from opencut.tools import dump_mcp_registry_manifest as tool

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "opencut" / "_generated" / "mcp_server_registry.json"
EXTENDED_MANIFEST = REPO_ROOT / "opencut" / "_generated" / "mcp_extended_tools.json"
DOC = REPO_ROOT / "docs" / "MCP_SERVER.md"


def test_manifest_file_exists():
    assert MANIFEST.is_file(), f"F147 manifest must exist at {MANIFEST}"


def test_doc_exists():
    assert DOC.is_file(), "docs/MCP_SERVER.md is required for F147"


def test_doc_documents_upstream_registration_steps():
    text = DOC.read_text(encoding="utf-8")
    for keyword in (
        "modelcontextprotocol/servers",
        "F147",
        "opencut-mcp-server",
        "dump_mcp_registry_manifest",
    ):
        assert keyword in text, f"docs/MCP_SERVER.md must mention {keyword!r}"


def test_doc_tool_counts_match_generated_manifests():
    text = DOC.read_text(encoding="utf-8")
    curated = json.loads(MANIFEST.read_text(encoding="utf-8"))["tool_count"]
    extended = json.loads(EXTENDED_MANIFEST.read_text(encoding="utf-8"))["tool_count"]

    assert f"{curated} curated tools" in text
    assert f"Tool catalogue ({curated} tools)" in text
    assert f"{extended:,} generated route-level tools" in text
    assert f"generated {extended:,} route-level set" in text


def test_committed_manifest_shape():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert data["name"] == "opencut-mcp-server"
    assert data["license"] == "MIT"
    assert data["manifest_version"] == tool.MANIFEST_VERSION
    assert isinstance(data["tools"], list) and data["tools"]
    assert data["tool_count"] == len(data["tools"])
    assert "stdio" in data["transports"]
    assert "http" in data["transports"]
    assert data["install"]["command"].startswith("pip install")
    assert data["install"]["python_min"] == "3.11"
    assert data["run"]["stdio"] == "opencut-mcp-server"


def test_committed_manifest_matches_live_catalogue():
    """No drift between the live MCP tool catalogue and the committed file."""
    live = tool.build_manifest()
    committed = tool.load_committed()
    diff = tool.diff_manifests(committed, live)
    assert diff == {"changed": False}, (
        f"Manifest is stale. Regenerate with "
        f"`python -m opencut.tools.dump_mcp_registry_manifest`. "
        f"Diff: {diff}"
    )


def test_manifest_version_matches_package_version():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert data["version"] == __version__


def test_manifest_diff_detects_version_drift():
    live = tool.build_manifest()
    committed = dict(live)
    committed["version"] = "0.0.0"

    diff = tool.diff_manifests(committed, live)

    assert diff["changed"] is True
    assert "version" in diff["fields"]


def test_manifest_tool_names_are_unique_and_namespaced():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    names = [tool_entry["name"] for tool_entry in data["tools"]]
    assert len(names) == len(set(names)), "Duplicate tool name in manifest"
    for name in names:
        assert name.startswith("opencut_"), (
            f"All MCP tool names must use the opencut_ namespace; got {name}"
        )


def test_manifest_at_least_expected_minimum_tools():
    """The MCP curated surface is currently broad enough for common workflows.
    Keep a floor so regressions in mcp_server.py become obvious."""
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert data["tool_count"] >= 86, (
        f"Expected at least 86 curated MCP tools; got {data['tool_count']}"
    )


# ---------------------------------------------------------------------------
# CLI contract
# ---------------------------------------------------------------------------


def _run_cli(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "opencut.tools.dump_mcp_registry_manifest",
            *args,
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=60,
    )


def test_cli_check_passes_in_sync():
    result = _run_cli("--check")
    assert result.returncode == 0, (
        f"--check should exit 0 when in sync; stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )
    assert "in sync" in result.stdout


def test_cli_check_emits_json():
    result = _run_cli("--check", "--json")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["diff"]["changed"] is False
    assert payload["live"]["name"] == "opencut-mcp-server"


def test_cli_writes_to_custom_path(tmp_path):
    target = tmp_path / "manifest.json"
    result = _run_cli("--output", str(target))
    assert result.returncode == 0
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["name"] == "opencut-mcp-server"
    assert data["tool_count"] >= 86
