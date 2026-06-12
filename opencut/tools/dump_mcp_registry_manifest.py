"""F147 — Generate the MCP registry manifest for `opencut-mcp-server`.

The upstream `modelcontextprotocol/servers` directory accepts a small
JSON entry per server with name, install command, transports, and the
curated tool catalogue. We generate that entry from the live
``opencut/mcp_server.py`` so the upstream listing cannot drift away
from the actual tool surface.

Use it three ways:

* ``python -m opencut.tools.dump_mcp_registry_manifest`` rewrites
  ``opencut/_generated/mcp_server_registry.json``.
* ``python -m opencut.tools.dump_mcp_registry_manifest --check`` exits
  non-zero when the committed manifest disagrees with the live tool
  catalogue; release smoke calls this.
* ``build_manifest()`` returns the manifest dict for tests / scripts.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

from opencut import __version__
from opencut.mcp_server import MCP_TOOLS

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = (
    REPO_ROOT / "opencut" / "_generated" / "mcp_server_registry.json"
)
MANIFEST_VERSION = 1

HOMEPAGE = "https://github.com/SysAdminDoc/OpenCut"
LICENSE = "MIT"
SUMMARY = (
    "OpenCut exposes its 1,300+ video-editing routes (silence/filler "
    "removal, captions, AI dubbing, scene detection, exports, review "
    "bundles, brand kit, etc.) over the Model Context Protocol so AI "
    "clients can drive Adobe Premiere Pro pipelines locally and "
    "without cloud APIs."
)
INSTALL_COMMAND = "pip install \"opencut[mcp]\""
RUN_STDIO = "opencut-mcp-server"
RUN_HTTP = "opencut-mcp-server --http"


def build_manifest() -> dict:
    """Return the structured manifest dict (Python objects, JSON-safe)."""
    tools: List[dict] = []
    for tool in MCP_TOOLS:
        name = tool.get("name", "")
        description = tool.get("description", "")
        if not name:
            continue
        tools.append(
            {
                "name": name,
                "description": description,
            }
        )
    tools.sort(key=lambda t: t["name"])

    manifest = {
        "name": "opencut-mcp-server",
        "version": __version__,
        "manifest_version": MANIFEST_VERSION,
        "category": "video-editing",
        "summary": SUMMARY,
        "homepage": HOMEPAGE,
        "repository": HOMEPAGE,
        "license": LICENSE,
        "tags": [
            "video",
            "editor",
            "premiere-pro",
            "ffmpeg",
            "whisper",
            "captions",
            "ai",
        ],
        "transports": ["stdio", "http"],
        "install": {
            "language": "python",
            "command": INSTALL_COMMAND,
            "python_min": "3.11",
            "executable": "opencut-mcp-server",
        },
        "run": {
            "stdio": RUN_STDIO,
            "http": RUN_HTTP,
            "list_tools": "opencut-mcp-server --list-tools",
        },
        "client_examples": {
            "claude_code": {
                "mcpServers": {
                    "opencut": {
                        "command": "opencut-mcp-server",
                        "args": [],
                    }
                }
            },
            "cursor": {
                "mcp.servers": {
                    "opencut": {
                        "command": "opencut-mcp-server",
                        "args": [],
                    }
                }
            },
        },
        "tools": tools,
        "tool_count": len(tools),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return manifest


def write_manifest(manifest: dict, path: Path = MANIFEST_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialised = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    path.write_text(serialised, encoding="utf-8")
    return path


def load_committed(path: Path = MANIFEST_PATH) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _comparable(manifest: dict) -> dict:
    """Strip only the timestamp so public registry metadata cannot drift."""
    compare = dict(manifest)
    compare.pop("generated_at", None)
    return compare


def diff_manifests(committed: Optional[dict], live: dict) -> dict:
    if committed is None:
        return {
            "changed": True,
            "reason": "committed manifest is absent",
        }
    cmp_committed = _comparable(committed)
    cmp_live = _comparable(live)
    if cmp_committed == cmp_live:
        return {"changed": False}

    diff = {"changed": True, "fields": []}
    keys = sorted(set(cmp_committed) | set(cmp_live))
    for key in keys:
        if cmp_committed.get(key) != cmp_live.get(key):
            diff["fields"].append(key)
    return diff


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the opencut-mcp-server registry manifest from the "
            "live MCP tool catalogue, or check the committed file for drift."
        )
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero on drift between committed manifest and live catalogue.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON to stdout.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=MANIFEST_PATH,
        help="Path to read/write the manifest (default: %(default)s).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    live = build_manifest()

    if args.check:
        committed = load_committed(args.output)
        diff = diff_manifests(committed, live)
        if args.json:
            print(json.dumps({"diff": diff, "live": live}, indent=2, sort_keys=True))
        else:
            if diff.get("changed"):
                print(
                    "opencut-mcp-server registry manifest is stale. "
                    "Regenerate with `python -m opencut.tools."
                    "dump_mcp_registry_manifest`."
                )
                fields = diff.get("fields") or []
                if fields:
                    print(f"  changed fields: {', '.join(fields)}")
                else:
                    print(f"  reason: {diff.get('reason', 'unknown')}")
            else:
                print(
                    f"opencut-mcp-server manifest in sync "
                    f"({live['tool_count']} tools)."
                )
        return 1 if diff.get("changed") else 0

    write_manifest(live, args.output)
    if args.json:
        print(
            json.dumps(
                {"path": str(args.output), "manifest": live},
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(
            f"Wrote {args.output} ({live['tool_count']} tools, "
            f"version={live['version']})."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
