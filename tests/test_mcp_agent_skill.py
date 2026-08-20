"""F345 — the MCP server ships a packaged agent skill.

An agent meeting 88 tools for the first time otherwise rediscovers the same
conventions from raw schemas every session. The skill is generated from the
registry, so these tests mostly guard that it cannot drift away from the tools
it claims to describe.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from opencut.core.mcp_agent_skill import (
    CONVENTIONS,
    TOOL_FAMILIES,
    WORKED_FLOWS,
    build_skill,
    render_markdown,
)
from opencut.tools.dump_mcp_agent_skill import (
    DOC_END,
    DOC_PATH,
    DOC_START,
    MANIFEST_PATH,
    diff_skill,
    load_committed,
    render_doc,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO_ROOT / "opencut" / "_generated" / "mcp_server_registry.json"


def _registry_tool_names() -> set[str]:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return {str(tool["name"]) for tool in registry["tools"]}


def test_committed_skill_matches_the_live_generator():
    assert diff_skill(load_committed(), build_skill()) == []


def test_every_registry_tool_appears_exactly_once():
    """A tool missing from the skill is a tool the agent never learns about."""
    skill = build_skill()
    listed = [tool["name"] for family in skill["families"] for tool in family["tools"]]

    assert len(listed) == len(set(listed)), "a tool was filed under two families"
    assert set(listed) == _registry_tool_names()
    assert skill["tool_count"] == len(_registry_tool_names())


def test_no_tool_falls_through_to_the_catch_all():
    """The catch-all exists so nothing is dropped, not as the normal path."""
    skill = build_skill()
    other = [family for family in skill["families"] if family["id"] == "other"]

    assert other == [], (
        "these tools matched no family and would be listed without context: "
        f"{[tool['name'] for tool in other[0]['tools']] if other else []}"
    )


def test_family_ids_are_unique():
    ids = [key for key, _label, _matchers in TOOL_FAMILIES]

    assert len(ids) == len(set(ids))


def test_worked_flow_only_names_tools_that_exist():
    """A skill that teaches a tool name the server does not expose is worse than none."""
    available = _registry_tool_names()
    for flow in WORKED_FLOWS:
        for step in flow["steps"]:
            assert step["tool"] in available, f"{flow['id']} names a missing tool: {step['tool']}"


def test_the_coldstart_flow_covers_transcribe_review_export():
    flow = next(f for f in WORKED_FLOWS if f["id"] == "transcribe-review-export")
    tools = [step["tool"] for step in flow["steps"]]

    assert tools[0] == "opencut_transcribe"
    assert "opencut_review_bundle" in tools
    assert tools[-1] == "opencut_export_video"


def test_every_convention_states_the_consequence_of_ignoring_it():
    for convention in CONVENTIONS:
        assert convention["rule"].strip()
        assert convention["ignoring_it"].strip()


def test_conventions_cover_jobs_and_review_before_mutate():
    ids = {convention["id"] for convention in CONVENTIONS}

    assert {"durable-jobs", "review-before-mutate"} <= ids


def test_skill_records_the_registry_version_it_came_from():
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))

    assert build_skill()["source_manifest_version"] == registry["manifest_version"]


def test_rendered_markdown_names_the_job_polling_tool():
    body = render_markdown(build_skill())

    assert "opencut_job_status" in body
    assert "Tool families" in body


def test_docs_carry_the_generated_block_between_markers():
    body = DOC_PATH.read_text(encoding="utf-8")

    assert DOC_START in body and DOC_END in body
    assert body.index(DOC_START) < body.index(DOC_END)


def test_doc_block_is_in_sync():
    assert DOC_PATH.read_text(encoding="utf-8") == render_doc(build_skill())


def test_regenerating_the_doc_is_idempotent(tmp_path):
    """A second run must not append a second block."""
    scratch = tmp_path / "MCP_SERVER.md"
    scratch.write_text(DOC_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    skill = build_skill()

    once = render_doc(skill, scratch)
    scratch.write_text(once, encoding="utf-8")
    twice = render_doc(skill, scratch)

    assert once == twice
    assert twice.count(DOC_START) == 1


def test_cli_check_passes_in_sync():
    result = subprocess.run(
        [sys.executable, "-m", "opencut.tools.dump_mcp_agent_skill", "--check"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_cli_check_fails_on_a_stale_manifest(tmp_path):
    """The gate has to actually detect drift, not just pass."""
    stale = tmp_path / "mcp_agent_skill.json"
    payload = build_skill()
    payload["tool_count"] = payload["tool_count"] + 1
    stale.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "opencut.tools.dump_mcp_agent_skill", "--check",
         "--output", str(stale)],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120,
    )

    assert result.returncode == 1
    assert "stale" in result.stdout.lower()


def test_manifest_ships_in_the_generated_directory():
    assert MANIFEST_PATH.is_file()
    assert MANIFEST_PATH.parent.name == "_generated"
