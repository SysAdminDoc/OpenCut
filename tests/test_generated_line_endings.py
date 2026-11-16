"""Generators must write tracked files with LF endings on every OS.

``Path.write_text`` opens in text mode, so on Windows each ``\\n`` becomes
``\\r\\n`` unless ``newline="\\n"`` is passed. Every committed JSON and doc these
tools produce is LF (``.gitattributes`` sets ``eol=lf``), so a regeneration on
Windows rewrote whole files with CRLF and made byte-level drift checks fail.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = {"alpha": [1, 2], "beta": {"gamma": "delta"}}


def _script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_eol_under_test", REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MANIFEST_WRITERS = [
    ("opencut.tools.adobe_premierepro_versions", "write_snapshot"),
    ("opencut.tools.adobe_uxp_compatibility", "write_manifest"),
    ("opencut.tools.dump_api_aliases", "write_alias_manifest"),
    ("opencut.tools.dump_cep_uxp_parity", "write_manifest"),
    ("opencut.tools.dump_mcp_agent_skill", "write_manifest"),
    ("opencut.tools.dump_mcp_extended_tools", "write_manifest"),
    ("opencut.tools.dump_mcp_registry_manifest", "write_manifest"),
    ("opencut.tools.dump_openapi_contract", "write_manifest"),
    ("opencut.tools.dump_panel_feature_parity", "write_manifest"),
    ("opencut.tools.dump_route_manifest", "write_manifest"),
    ("opencut.tools.dump_uxp_migration_dashboard", "write_manifest"),
]


def _assert_lf_only(path: Path):
    data = path.read_bytes()
    assert b"\n" in data, f"{path.name} was written as a single line, nothing to check"
    assert b"\r\n" not in data, f"{path.name} was written with CRLF endings"


@pytest.mark.parametrize("module_name,func_name", MANIFEST_WRITERS, ids=[m.rsplit(".", 1)[1] for m, _ in MANIFEST_WRITERS])
def test_manifest_writer_uses_lf(tmp_path, module_name, func_name):
    writer = getattr(importlib.import_module(module_name), func_name)
    out = tmp_path / "nested" / "manifest.json"
    writer(PAYLOAD, out)
    _assert_lf_only(out)


@pytest.mark.parametrize("script_name", ["sync_shared_locales", "release_gate"])
def test_script_json_writer_uses_lf(tmp_path, script_name):
    out = tmp_path / "nested" / "payload.json"
    _script(script_name)._write_json(out, PAYLOAD)
    _assert_lf_only(out)


def test_mcp_skill_doc_writer_uses_lf(tmp_path, monkeypatch):
    module = importlib.import_module("opencut.tools.dump_mcp_agent_skill")
    monkeypatch.setattr(module, "render_doc", lambda skill, path: "# MCP\n\nskill block\n")
    out = tmp_path / "MCP_SERVER.md"
    module.write_doc({}, out)
    _assert_lf_only(out)


def test_model_cards_writer_uses_lf(tmp_path, monkeypatch):
    module = importlib.import_module("opencut.tools.dump_model_cards")
    json_path = tmp_path / "gen" / "model_cards.json"
    docs_path = tmp_path / "docs" / "MODELS.md"
    monkeypatch.setattr(module, "JSON_PATH", json_path)
    monkeypatch.setattr(module, "DOCS_PATH", docs_path)
    monkeypatch.setattr(module, "_render_markdown", lambda manifest: "# Models\n\nrow\n")
    module.write_artifacts(PAYLOAD)
    _assert_lf_only(json_path)
    _assert_lf_only(docs_path)
