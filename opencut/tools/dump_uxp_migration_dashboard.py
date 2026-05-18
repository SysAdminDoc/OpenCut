"""Generate the F260 UXP migration risk dashboard artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

from opencut.core.cep_uxp_parity import build_dashboard_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "opencut" / "_generated" / "uxp_migration_dashboard.json"
PANEL_PATH = REPO_ROOT / "extension" / "com.opencut.uxp" / "uxp-migration-dashboard.json"


def write_manifest(manifest: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_committed(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def diff_manifests(committed: Optional[dict], live: dict, label: str) -> list[str]:
    if committed is None:
        return [f"{label} is absent"]
    if committed == live:
        return []
    changed = [field for field in sorted(set(committed) | set(live)) if committed.get(field) != live.get(field)]
    return [f"{label} changed fields: {', '.join(changed)}"]


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Exit non-zero if committed dashboard artifacts are stale.")
    parser.add_argument("--json", action="store_true", help="Emit the live manifest or diff as JSON.")
    parser.add_argument("--output", type=Path, default=MANIFEST_PATH, help="Repository generated artifact path.")
    parser.add_argument("--panel-output", type=Path, default=PANEL_PATH, help="Bundled UXP panel dashboard JSON path.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    live = build_dashboard_manifest()
    if args.check:
        diff = []
        diff.extend(diff_manifests(load_committed(args.output), live, "generated dashboard"))
        diff.extend(diff_manifests(load_committed(args.panel_output), live, "panel dashboard"))
        if args.json:
            print(json.dumps({"diff": diff, "live": live}, indent=2, sort_keys=True))
        elif diff:
            print(
                "UXP migration dashboard artifacts are stale. Regenerate with "
                "`python -m opencut.tools.dump_uxp_migration_dashboard`."
            )
            for line in diff:
                print(f"  {line}")
        else:
            print(
                "UXP migration dashboard artifacts in sync "
                f"({live['summary']['function_count']} host actions, "
                f"{live['summary']['cep_only']} CEP-only)."
            )
        return 1 if diff else 0

    write_manifest(live, args.output)
    write_manifest(live, args.panel_output)
    if args.json:
        print(json.dumps(live, indent=2, sort_keys=True))
    else:
        print(f"Wrote {args.output} and {args.panel_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
