"""Generate the F345 packaged agent skill for the MCP server."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

from opencut.core.mcp_agent_skill import build_skill, render_markdown

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "opencut" / "_generated" / "mcp_agent_skill.json"
DOC_PATH = REPO_ROOT / "docs" / "MCP_SERVER.md"
DOC_START = "<!-- agent-skill:start -->"
DOC_END = "<!-- agent-skill:end -->"


def write_manifest(skill: dict, path: Path = MANIFEST_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(skill, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def render_doc(skill: dict, path: Path = DOC_PATH) -> str:
    """Return docs/MCP_SERVER.md with the skill block replaced.

    The skill lives inside the existing MCP document rather than a new file so
    there is one place to look and no second document to keep in step.
    """
    body = path.read_text(encoding="utf-8")
    block = f"{DOC_START}\n\n{render_markdown(skill)}\n{DOC_END}"
    if DOC_START in body and DOC_END in body:
        head, _, rest = body.partition(DOC_START)
        _, _, tail = rest.partition(DOC_END)
        return f"{head}{block}{tail}"
    separator = "" if body.endswith("\n\n") else ("\n" if body.endswith("\n") else "\n\n")
    return f"{body}{separator}## Agent skill\n\n{block}\n"


def write_doc(skill: dict, path: Path = DOC_PATH) -> Path:
    path.write_text(render_doc(skill, path), encoding="utf-8")
    return path


def load_committed(path: Path = MANIFEST_PATH) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def diff_skill(committed: Optional[dict], live: dict, doc_path: Path = DOC_PATH) -> list[str]:
    diff: list[str] = []
    if committed is None:
        diff.append("agent skill manifest is missing")
    elif committed != live:
        for key in sorted(set(committed) | set(live)):
            if committed.get(key) != live.get(key):
                diff.append(f"changed field: {key}")
    if doc_path.is_file() and doc_path.read_text(encoding="utf-8") != render_doc(live, doc_path):
        diff.append(f"{doc_path.name} skill block is stale")
    return diff


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero when the committed skill artifacts are stale.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the live skill as JSON.")
    parser.add_argument("--output", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args(list(argv) if argv is not None else None)

    live = build_skill()
    if args.check:
        diff = diff_skill(load_committed(args.output), live)
        if args.json:
            print(json.dumps({"diff": diff, "live": live}, indent=2, sort_keys=True))
        elif diff:
            print(
                "MCP agent skill is stale. Regenerate with "
                "`python -m opencut.tools.dump_mcp_agent_skill`."
            )
            for line in diff:
                print(f"  {line}")
        else:
            print(f"MCP agent skill in sync ({live['tool_count']} tools).")
        return 1 if diff else 0

    write_manifest(live, args.output)
    write_doc(live)
    if args.json:
        print(json.dumps(live, indent=2, sort_keys=True))
    else:
        print(f"Wrote {args.output} and {DOC_PATH.name} ({live['tool_count']} tools).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
