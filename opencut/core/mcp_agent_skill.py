"""Generate a packaged agent skill from the MCP tool registry.

An agent meeting OpenCut's 98-tool MCP server for the first time has to
rediscover the same conventions from raw schemas every session: that a tool may
return a job rather than a result, that destructive timeline writes go through a
review pass, and that the curated catalogue deliberately omits configuration
routes. This module renders those conventions and a family map of the tools into
one versioned artifact, derived from the registry so it cannot drift away from
the tools it describes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "opencut" / "_generated" / "mcp_server_registry.json"

SKILL_VERSION = 1
SKILL_NAME = "opencut-ppro"

#: Families keyed by the token after the ``opencut_`` prefix. Order is the order
#: they appear in the rendered skill. ``other`` is the deliberate catch-all so a
#: newly added tool is listed rather than silently dropped.
TOOL_FAMILIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "surfaces",
        "Route-family entry points",
        ("surface",),
    ),
    (
        "cut",
        "Cut and clean",
        ("silence", "filler", "repeat", "auto", "scene", "cut", "trim", "speed"),
    ),
    (
        "captions",
        "Captions and transcript",
        ("transcribe", "caption", "chapters", "subtitle", "translate", "adr"),
    ),
    (
        "audio",
        "Audio",
        ("audio", "denoise", "music", "tts", "beat", "voice", "spectral", "loudness"),
    ),
    (
        "video",
        "Video and render",
        (
            "export", "video", "color", "chromakey", "blend", "concat", "multicam",
            "highlights", "zoom", "lut", "upscale", "frame", "preview", "stabilize",
            "merge", "letterbox", "transitions", "interpolate", "depth", "style",
            "shorts", "vfx", "social",
        ),
    ),
    (
        "retouch",
        "Face and retouch",
        ("face", "skin", "lipsync", "echomimic"),
    ),
    (
        "assets",
        "Footage and ingest",
        ("index", "ingest", "footage", "timeline", "bins", "url"),
    ),
    (
        "review",
        "Review and provenance",
        ("review", "c2pa", "qc", "brand", "marker"),
    ),
    (
        "system",
        "Capability and jobs",
        (
            "job", "capability", "gpu", "workflow", "chat", "search", "batch",
            "dependencies", "pip", "system", "info", "feature", "state", "nlp", "command",
        ),
    ),
)

#: Conventions an agent cannot infer from a tool schema. Each one states the
#: rule and the observable consequence of ignoring it.
CONVENTIONS: tuple[Dict[str, str], ...] = (
    {
        "id": "durable-jobs",
        "title": "A tool may hand back a job instead of a result",
        "rule": (
            "Every tool returns either a synchronous `result` object or a `job_id`. "
            "Poll `opencut_job_status` until the job reports `complete`, `error`, "
            "`interrupted`, or `cancelled`. Clients that declare the "
            "`io.modelcontextprotocol/tasks` extension get a task handle instead and "
            "should use `tasks/get`."
        ),
        "ignoring_it": (
            "Treating the job acknowledgement as the finished result reports success "
            "for work that has not run yet, and the output file will not exist."
        ),
    },
    {
        "id": "review-before-mutate",
        "title": "Propose edits, then apply them",
        "rule": (
            "Detection tools return ranges; they do not touch the timeline. Build a "
            "`opencut_review_bundle`, let a human accept or reject, and apply the "
            "outcome with `opencut_review_action`. Cut application offers a "
            "non-destructive mode that disables clips rather than deleting them."
        ),
        "ignoring_it": (
            "Applying detected ranges straight to a sequence deletes a user's media "
            "on the strength of a heuristic, which is the failure editors distrust "
            "these tools for."
        ),
    },
    {
        "id": "curated-surface",
        "title": "The catalogue is curated on purpose",
        "rule": (
            "Install, settings, and housekeeping routes are deliberately absent so an "
            "MCP client cannot reconfigure the backend. The route-shaped extended "
            "catalogue is off unless `OPENCUT_MCP_EXTENDED_TOOLS=1` is set."
        ),
        "ignoring_it": (
            "Reaching for a missing capability through the REST surface bypasses the "
            "boundary that keeps an agent from changing the user's install."
        ),
    },
    {
        "id": "local-paths",
        "title": "Everything runs locally against real paths",
        "rule": (
            "Tools take filesystem paths on the machine running the backend and write "
            "output beside the input unless told otherwise. There is no upload step "
            "and no cloud key. Confirm a path exists before starting long work."
        ),
        "ignoring_it": (
            "A path that only exists on the client produces a job that fails minutes "
            "in, after the user has waited for it."
        ),
    },
)

#: A coldstart agent should be able to run this end to end from the skill alone.
WORKED_FLOWS: tuple[Dict[str, Any], ...] = (
    {
        "id": "transcribe-review-export",
        "title": "Transcribe, review the cuts, export",
        "steps": (
            {
                "tool": "opencut_transcribe",
                "note": "Returns a job. Poll opencut_job_status for the transcript.",
            },
            {
                "tool": "opencut_silence_remove",
                "note": "Detects removable ranges. Nothing is applied yet.",
            },
            {
                "tool": "opencut_review_bundle",
                "note": "Packages the proposed ranges for a human decision.",
            },
            {
                "tool": "opencut_review_action",
                "note": "Applies the accepted ranges, or disables rather than deletes.",
            },
            {
                "tool": "opencut_export_video",
                "note": "Renders the result. Returns a job; poll it to completion.",
            },
        ),
    },
)


def _load_registry(path: Path = REGISTRY_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _family_for(tool_name: str) -> str:
    stem = tool_name.removeprefix("opencut_")
    tokens = stem.split("_")
    for key, _label, matchers in TOOL_FAMILIES:
        if any(token in matchers for token in tokens):
            return key
    return "other"


def build_skill(path: Path = REGISTRY_PATH) -> dict:
    """Render the agent skill from the committed MCP registry."""
    registry = _load_registry(path)
    tools = registry.get("tools") or []

    grouped: Dict[str, List[Dict[str, str]]] = {key: [] for key, _l, _m in TOOL_FAMILIES}
    grouped["other"] = []
    for tool in sorted(tools, key=lambda item: str(item.get("name", ""))):
        grouped[_family_for(str(tool.get("name", "")))].append({
            "name": str(tool.get("name", "")),
            "description": str(tool.get("description", "")),
        })

    labels = {key: label for key, label, _m in TOOL_FAMILIES}
    labels["other"] = "Other"
    families = [
        {"id": key, "label": labels[key], "tools": grouped[key]}
        for key, _label, _matchers in (*TOOL_FAMILIES, ("other", "Other", ()))
        if grouped[key]
    ]

    return {
        "skill_version": SKILL_VERSION,
        "name": SKILL_NAME,
        "source_manifest_version": registry.get("manifest_version"),
        "tool_count": len(tools),
        "conventions": [dict(entry) for entry in CONVENTIONS],
        "families": families,
        "flows": [
            {"id": flow["id"], "title": flow["title"], "steps": [dict(s) for s in flow["steps"]]}
            for flow in WORKED_FLOWS
        ],
    }


def render_markdown(skill: dict) -> str:
    """Render the skill as the prose block embedded in docs/MCP_SERVER.md."""
    lines: List[str] = []
    lines.append(
        f"Generated from `opencut/_generated/mcp_server_registry.json` "
        f"({skill['tool_count']} tools). Regenerate with "
        f"`python -m opencut.tools.dump_mcp_agent_skill`."
    )
    lines.append("")
    lines.append("### Conventions")
    lines.append("")
    for convention in skill["conventions"]:
        lines.append(f"**{convention['title']}**")
        lines.append("")
        lines.append(convention["rule"])
        lines.append("")
        lines.append(f"Ignoring it: {convention['ignoring_it']}")
        lines.append("")
    for flow in skill["flows"]:
        lines.append(f"### {flow['title']}")
        lines.append("")
        for index, step in enumerate(flow["steps"], start=1):
            lines.append(f"{index}. `{step['tool']}` — {step['note']}")
        lines.append("")
    lines.append("### Tool families")
    lines.append("")
    for family in skill["families"]:
        names = ", ".join(f"`{tool['name']}`" for tool in family["tools"])
        lines.append(f"**{family['label']}** ({len(family['tools'])}): {names}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
