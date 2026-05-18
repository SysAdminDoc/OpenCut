"""F251 — `@adobe/premierepro` npm version tracker.

OpenCut's CEP -> UXP migration depends on the `@adobe/premierepro` npm
package shipping new APIs (`createCaptionTrack`, `startDrag`, etc.) that
close current CEP-only gaps. This tool catches the moment those APIs
land by snapshotting Adobe's published `latest` / `beta` dist-tag pair
and diffing it against a committed reference file.

Use it three ways:

* ``python -m opencut.tools.adobe_premierepro_versions`` rewrites
  ``opencut/_generated/adobe_premierepro_versions.json`` from the live
  registry. Use this when you intentionally ship an updated reference.
* ``python -m opencut.tools.adobe_premierepro_versions --check`` returns
  exit code ``0`` when the committed file matches the live registry, and
  exit code ``2`` when a drift is detected. **CI deliberately does NOT
  fail closed on drift** — the wrapping step (release_smoke + the weekly
  scheduled workflow) parses the JSON output and surfaces the diff as a
  notification rather than a release blocker.
* ``build_versions()`` returns the structured dict for tests / scripts.

The tool fetches the npm registry through stdlib ``urllib.request`` so
it does not require a Node toolchain. Network failure produces a stable
JSON shape with ``status="network_error"`` instead of raising.

Reference URL: ``https://registry.npmjs.org/@adobe/premierepro``.

Source of truth for the F-number: see
``.ai/research/2026-05-17/FEATURE_BACKLOG_ADDENDUM.md`` (UXP subagent §3).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

PACKAGE = "@adobe/premierepro"
REGISTRY_URL = f"https://registry.npmjs.org/{PACKAGE.replace('/', '%2F')}"
REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_PATH = REPO_ROOT / "opencut" / "_generated" / "adobe_premierepro_versions.json"
SNAPSHOT_VERSION = 1

# Track the dist-tags that matter for OpenCut's UXP migration. Anything
# else Adobe publishes ("next", "experimental", etc.) is captured in the
# "all" list but does not gate or notify.
TRACKED_TAGS = ("latest", "beta")


@dataclass
class VersionSnapshot:
    """One serialisable snapshot of `@adobe/premierepro` registry state."""

    package: str = PACKAGE
    status: str = "ok"  # "ok" | "network_error" | "parse_error"
    error: Optional[str] = None
    snapshot_version: int = SNAPSHOT_VERSION
    recorded_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    )
    dist_tags: Dict[str, str] = field(default_factory=dict)
    tracked_versions: List[str] = field(default_factory=list)
    latest_release_versions: List[str] = field(default_factory=list)
    release_count: int = 0
    notes: List[str] = field(default_factory=list)


def _http_get(url: str, *, timeout: float = 15.0) -> dict:
    """GET a JSON URL via stdlib urllib. Raises ``urllib.error.URLError``
    on connection failure and ``ValueError`` on non-JSON body."""
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": "opencut-tools"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def _build_offline_snapshot(reason: str) -> VersionSnapshot:
    """Return a snapshot used when the registry can't be reached."""
    snap = VersionSnapshot(
        status="network_error",
        error=reason,
        notes=[
            "Offline placeholder — populate by re-running "
            "`python -m opencut.tools.adobe_premierepro_versions` "
            "from a host with npm registry access."
        ],
    )
    return snap


def fetch_registry(*, timeout: float = 15.0) -> VersionSnapshot:
    """Fetch live `@adobe/premierepro` metadata from the npm registry."""
    try:
        payload = _http_get(REGISTRY_URL, timeout=timeout)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        return _build_offline_snapshot(f"{type(exc).__name__}: {exc}")
    except (ValueError, json.JSONDecodeError) as exc:
        snap = VersionSnapshot(status="parse_error", error=str(exc))
        return snap

    dist_tags_raw = payload.get("dist-tags") or {}
    versions_raw = payload.get("versions") or {}
    tracked = {
        tag: dist_tags_raw[tag] for tag in TRACKED_TAGS if tag in dist_tags_raw
    }
    all_versions = sorted(versions_raw.keys(), key=_semver_key, reverse=True)
    return VersionSnapshot(
        status="ok",
        dist_tags=dict(sorted(dist_tags_raw.items())),
        tracked_versions=[tracked[tag] for tag in TRACKED_TAGS if tag in tracked],
        latest_release_versions=all_versions[:10],
        release_count=len(versions_raw),
    )


def _semver_key(version: str) -> tuple:
    """Best-effort semver sort key — never raises on weird suffixes."""
    base, _, suffix = version.partition("-")
    parts: List[int] = []
    for chunk in base.split("."):
        try:
            parts.append(int(chunk))
        except ValueError:
            parts.append(0)
    # Pad to 3 parts so 26.3 sorts after 26.2.99.
    while len(parts) < 3:
        parts.append(0)
    # Pre-release suffix sorts before its base version (npm semver rules).
    return tuple(parts) + (0 if suffix else 1, suffix)


def build_versions(*, offline: bool = False, timeout: float = 15.0) -> dict:
    """Return the structured version snapshot dict."""
    snap = (
        _build_offline_snapshot("offline=True override")
        if offline
        else fetch_registry(timeout=timeout)
    )
    return asdict(snap)


def load_committed_snapshot(
    path: Path = SNAPSHOT_PATH,
) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_snapshot(snapshot: dict, path: Path = SNAPSHOT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialised = json.dumps(snapshot, indent=2, sort_keys=True) + "\n"
    path.write_text(serialised, encoding="utf-8")
    return path


def _comparable(snapshot: dict) -> dict:
    """Strip timestamp / error fields before equality check."""
    compare = dict(snapshot)
    compare.pop("recorded_at", None)
    compare.pop("error", None)
    return compare


def diff_snapshots(committed: Optional[dict], live: dict) -> dict:
    """Compute a human-readable diff between committed and live."""
    diff: Dict[str, dict] = {}
    if committed is None:
        diff["status"] = {"from": "<absent>", "to": live.get("status")}
        diff["dist_tags"] = {"from": {}, "to": live.get("dist_tags") or {}}
        return {"changed": True, "fields": diff}

    cmp_committed = _comparable(committed)
    cmp_live = _comparable(live)

    fields_changed: Dict[str, dict] = {}

    if cmp_committed.get("status") != cmp_live.get("status"):
        fields_changed["status"] = {
            "from": cmp_committed.get("status"),
            "to": cmp_live.get("status"),
        }

    committed_tags = cmp_committed.get("dist_tags") or {}
    live_tags = cmp_live.get("dist_tags") or {}
    tag_keys = sorted(set(committed_tags) | set(live_tags))
    tag_diff: Dict[str, dict] = {}
    for tag in tag_keys:
        if committed_tags.get(tag) != live_tags.get(tag):
            tag_diff[tag] = {
                "from": committed_tags.get(tag),
                "to": live_tags.get(tag),
            }
    if tag_diff:
        fields_changed["dist_tags"] = tag_diff

    committed_releases = list(cmp_committed.get("latest_release_versions") or [])
    live_releases = list(cmp_live.get("latest_release_versions") or [])
    if committed_releases != live_releases:
        added = [v for v in live_releases if v not in committed_releases]
        removed = [v for v in committed_releases if v not in live_releases]
        fields_changed["latest_release_versions"] = {
            "added": added,
            "removed": removed,
        }

    if cmp_committed.get("release_count") != cmp_live.get("release_count"):
        fields_changed["release_count"] = {
            "from": cmp_committed.get("release_count"),
            "to": cmp_live.get("release_count"),
        }

    return {"changed": bool(fields_changed), "fields": fields_changed}


def _format_diff_text(diff: dict, *, package: str = PACKAGE) -> str:
    if not diff.get("changed"):
        return f"{package}: no drift between committed snapshot and live npm registry."
    lines = [f"{package}: drift detected — review and either:"]
    lines.append(
        "  - regenerate the snapshot: "
        "`python -m opencut.tools.adobe_premierepro_versions`"
    )
    lines.append(
        "  - or file an F-number for the new API surface, if Adobe shipped one."
    )
    fields = diff.get("fields") or {}
    for key, change in fields.items():
        lines.append(f"  [{key}]")
        if key == "dist_tags":
            for tag, swing in (change or {}).items():
                lines.append(
                    f"    {tag}: {swing.get('from')!r} -> {swing.get('to')!r}"
                )
        elif key == "latest_release_versions":
            added = change.get("added") or []
            removed = change.get("removed") or []
            if added:
                lines.append(f"    added: {', '.join(added)}")
            if removed:
                lines.append(f"    removed: {', '.join(removed)}")
        else:
            lines.append(f"    {change}")
    return "\n".join(lines)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch and snapshot `@adobe/premierepro` registry metadata, "
            "or check for drift against the committed snapshot."
        )
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Compare live registry against the committed snapshot. "
            "Exit code 2 signals drift; CI wrappers treat it as a "
            "notification, not a release blocker."
        ),
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip the network call (emits an offline placeholder snapshot).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON to stdout.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SNAPSHOT_PATH,
        help="Path to read/write the snapshot (default: %(default)s).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    live = build_versions(offline=args.offline)

    if args.check:
        committed = load_committed_snapshot(args.output)
        diff = diff_snapshots(committed, live)
        payload = {
            "package": PACKAGE,
            "status": live.get("status"),
            "live": live,
            "committed_present": committed is not None,
            "drift": diff,
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(_format_diff_text(diff))
        return 2 if diff.get("changed") else 0

    write_snapshot(live, args.output)
    if args.json:
        print(
            json.dumps(
                {"package": PACKAGE, "snapshot_path": str(args.output), "snapshot": live},
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(
            f"Wrote {args.output} (status={live.get('status')}, "
            f"dist_tags={live.get('dist_tags') or {}}, "
            f"release_count={live.get('release_count')})."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
