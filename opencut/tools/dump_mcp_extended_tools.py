"""Generate the opt-in extended MCP route-tool catalogue (F194)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

from opencut import mcp_extended_tools as extended
from opencut import mcp_server

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "opencut" / "_generated" / "mcp_extended_tools.json"

_SPECIAL_CURATED_ROUTES = {
    ("DELETE", "/settings/brand-kit"),
    ("POST", "/settings/brand-kit/preview"),
    ("POST", "/search/ai/index"),
    ("GET", "/search/ai/index/status"),
}

# Queue and host-write recovery stay REST/UI-only. These routes expose complete
# persisted payloads (including local media paths and host diagnostics), while
# their mutations alter durable recovery state without the MCP registry's
# destructive confirmation contract. Agents can still use established curated
# queue and journal tools where their confirmation contract is explicit.
_REST_ONLY_ROUTES = {
    # Diagnostic cache identity is intentionally not an agent action. Keeping
    # this REST/CLI-only also avoids making local cache keys discoverable via
    # the broad generated route catalogue.
    ("GET", "/captions/cache/provenance/<cache_key>"),
    ("GET", "/queue/export"),
    ("POST", "/queue/import"),
    ("POST", "/queue/replay/<queue_id>"),
    ("GET", "/journal/recovery"),
    ("POST", "/journal/checkpoints"),
    ("GET", "/journal/checkpoints/<transaction_id>"),
    ("POST", "/journal/checkpoints/<transaction_id>/complete"),
    ("POST", "/journal/checkpoints/<transaction_id>/recovery-failed"),
    ("POST", "/journal/checkpoints/<transaction_id>/recovered"),
    ("GET", "/journal/checkpoints/<transaction_id>/diagnostics"),
}


def curated_route_keys() -> set[tuple[str, str]]:
    return {
        (method.upper(), path)
        for method, path in mcp_server._TOOL_ROUTES.values()
    } | set(_SPECIAL_CURATED_ROUTES)


def excluded_route_keys() -> set[tuple[str, str]]:
    """Return curated and intentionally REST-only route keys."""
    return curated_route_keys() | _REST_ONLY_ROUTES


def build_manifest() -> dict:
    return extended.build_manifest(excluded_route_keys=excluded_route_keys())


def write_manifest(manifest: dict, path: Path = MANIFEST_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_committed(path: Path = MANIFEST_PATH) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def diff_manifests(committed: Optional[dict], live: dict) -> list[str]:
    if committed is None:
        return ["committed extended MCP manifest is absent"]
    if committed == live:
        return []
    changed = sorted(
        field for field in set(committed) | set(live)
        if committed.get(field) != live.get(field)
    )
    return [f"changed fields: {', '.join(changed)}"]


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero when the committed generated catalogue is stale.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON.",
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
        diff = diff_manifests(load_committed(args.output), live)
        if args.json:
            print(json.dumps({"diff": diff, "live": live}, indent=2, sort_keys=True))
        elif diff:
            print(
                "Extended MCP route-tool manifest is stale. Regenerate with "
                "`python -m opencut.tools.dump_mcp_extended_tools`."
            )
            for line in diff:
                print(f"  {line}")
        else:
            print(
                f"Extended MCP route-tool manifest in sync "
                f"({live['tool_count']} tools)."
            )
        return 1 if diff else 0

    write_manifest(live, args.output)
    if args.json:
        print(json.dumps(live, indent=2, sort_keys=True))
    else:
        print(f"Wrote {args.output} ({live['tool_count']} extended MCP tools).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
