"""Generate the F198 CEP-to-UXP parity catalogue artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

from opencut.core.cep_uxp_parity import build_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "opencut" / "_generated" / "cep_uxp_parity.json"


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
        return ["committed CEP/UXP parity manifest is absent"]
    if committed == live:
        return []
    fields = sorted(set(committed) | set(live))
    changed = [field for field in fields if committed.get(field) != live.get(field)]
    return [f"changed fields: {', '.join(changed)}"]


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero when the committed JSON artifact is stale.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the live manifest or diff as JSON.",
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
                "CEP/UXP parity manifest is stale. Regenerate with "
                "`python -m opencut.tools.dump_cep_uxp_parity`."
            )
            for line in diff:
                print(f"  {line}")
        else:
            print(
                f"CEP/UXP parity manifest in sync "
                f"({live['function_count']} functions, {live['cep_only_count']} CEP-only)."
            )
        return 1 if diff else 0

    write_manifest(live, args.output)
    if args.json:
        print(json.dumps(live, indent=2, sort_keys=True))
    else:
        print(
            f"Wrote {args.output} "
            f"({live['function_count']} functions, {live['cep_only_count']} CEP-only)."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
