#!/usr/bin/env python3
"""Seed source-linked GitHub issues from ``.github/issue-seeds.yml``.

The repository roadmap (``ROADMAP.md``) carries dozens of source-linked
backlog rows but ``gh issue list SysAdminDoc/OpenCut`` currently returns
nothing — public contributors have no entrypoint into the work. This
script bridges that gap by turning the curated YAML manifest into actual
tracker entries.

Usage::

    python scripts/seed_github_issues.py --dry-run
    python scripts/seed_github_issues.py --labels --dry-run
    python scripts/seed_github_issues.py --apply
    python scripts/seed_github_issues.py --good-first --apply

Authentication and write access to ``SysAdminDoc/OpenCut`` are required
for ``--apply``. The script is idempotent: it skips any seed whose
``roadmap_id`` already appears in an existing open issue title.

Why YAML instead of bespoke JSON: the seed manifest is hand-edited; YAML
keeps the multi-line ``body`` fields readable. PyYAML is an optional
runtime dep — the loader falls back to a tiny, focused parser when PyYAML
is missing so that contributors without it can still run ``--dry-run``.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

# Force UTF-8 stdout so → arrows and similar glyphs in seed bodies don't blow
# up the Windows console (cp1252 default).
for _stream_attr in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_attr, None)
    if _stream is not None and getattr(_stream, "encoding", "").lower() != "utf-8":
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, io.UnsupportedOperation):
            pass

REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS_PATH = REPO_ROOT / ".github" / "issue-seeds.yml"
LABELS_PATH = REPO_ROOT / ".github" / "labels.yml"
DEFAULT_REPO = "SysAdminDoc/OpenCut"


@dataclass
class Seed:
    roadmap_id: str
    title: str
    body: str
    labels: List[str] = field(default_factory=list)
    good_first: bool = False


@dataclass
class Label:
    name: str
    color: str
    description: str = ""


class SeedParseError(RuntimeError):
    """Raised when the YAML manifest cannot be parsed."""


# ---------------------------------------------------------------------------
# Minimal YAML loader
# ---------------------------------------------------------------------------
#
# The seed manifest only uses a tiny subset of YAML (top-level mapping,
# list-of-mappings, scalars, JSON-style inline lists, and ``|`` literal
# block scalars). We keep an embedded parser to avoid a hard PyYAML
# dependency for ``--dry-run`` and the focused tests, but prefer PyYAML
# when it is available because the embedded parser is intentionally
# strict.


def _load_yaml(text: str) -> dict:
    try:
        import yaml  # type: ignore
    except Exception:
        return _load_yaml_minimal(text)
    return yaml.safe_load(text) or {}


def _load_yaml_minimal(text: str) -> object:
    lines = text.splitlines()

    # Detect whether the top level is a mapping or a list by looking at the
    # first non-blank, non-comment line.
    for sample in lines:
        stripped_sample = sample.strip()
        if not stripped_sample or stripped_sample.startswith("#"):
            continue
        if stripped_sample.startswith("- "):
            return _parse_list(lines)
        break

    idx = 0
    result: dict = {}

    while idx < len(lines):
        raw = lines[idx]
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            idx += 1
            continue

        if raw.startswith(" ") or raw.startswith("\t"):
            raise SeedParseError(f"unexpected indented line at top level: {raw!r}")

        key, sep, after = stripped.partition(":")
        if not sep:
            raise SeedParseError(f"expected 'key:' at top level, got {raw!r}")
        key = key.strip()
        after = after.strip()

        if after:
            result[key] = _coerce_scalar(after)
            idx += 1
            continue

        idx += 1
        block_lines: List[str] = []
        while idx < len(lines):
            nxt = lines[idx]
            if nxt.strip() == "" or nxt.startswith(" ") or nxt.startswith("\t"):
                block_lines.append(nxt)
                idx += 1
            else:
                break

        result[key] = _parse_block(block_lines)

    return result


def _parse_block(block: List[str]) -> object:
    # Determine the base indent.
    indents = [len(line) - len(line.lstrip(" ")) for line in block if line.strip()]
    if not indents:
        return []
    base = min(indents)
    cleaned = [line[base:] if line.strip() else "" for line in block]

    # List vs mapping detection.
    if any(line.startswith("- ") or line.strip() == "-" for line in cleaned):
        return _parse_list(cleaned)
    return _parse_mapping(cleaned)


def _parse_list(lines: List[str]) -> List[object]:
    items: List[object] = []
    chunk: List[str] = []
    for line in lines:
        if line.startswith("- "):
            if chunk:
                items.append(_parse_mapping(chunk))
                chunk = []
            chunk.append(line[2:])
        elif line.strip() == "-":
            if chunk:
                items.append(_parse_mapping(chunk))
                chunk = []
        else:
            # Continuation belongs to the current item; preserve indentation
            # relative to the "- " column.
            chunk.append(line[2:] if line.startswith("  ") else line)
    if chunk:
        items.append(_parse_mapping(chunk))
    return items


def _parse_mapping(lines: List[str]) -> dict:
    result: dict = {}
    idx = 0
    while idx < len(lines):
        raw = lines[idx]
        if not raw.strip() or raw.lstrip().startswith("#"):
            idx += 1
            continue

        key, sep, after = raw.partition(":")
        if not sep:
            raise SeedParseError(f"expected 'key:' inside mapping, got {raw!r}")
        key = key.strip()
        after = after.strip()

        if after == "|":
            idx += 1
            block: List[str] = []
            base = None
            while idx < len(lines):
                nxt = lines[idx]
                if nxt.strip() == "":
                    block.append("")
                    idx += 1
                    continue
                spaces = len(nxt) - len(nxt.lstrip(" "))
                if base is None:
                    base = spaces
                if spaces < base:
                    break
                block.append(nxt[base:])
                idx += 1
            # Trim trailing blanks like the YAML spec for ``|``.
            while block and block[-1] == "":
                block.pop()
            result[key] = "\n".join(block) + "\n"
            continue

        if after:
            result[key] = _coerce_scalar(after)
            idx += 1
            continue

        # Nested block: gather and recurse.
        idx += 1
        nested: List[str] = []
        while idx < len(lines):
            nxt = lines[idx]
            if nxt.strip() == "" or nxt.startswith(" ") or nxt.startswith("\t"):
                nested.append(nxt)
                idx += 1
            else:
                break
        result[key] = _parse_block(nested) if nested else {}

    return result


def _coerce_scalar(token: str) -> object:
    if token.startswith("[") and token.endswith("]"):
        return [
            item.strip().strip('"').strip("'")
            for item in token[1:-1].split(",")
            if item.strip()
        ]
    if token.lower() in {"true", "false"}:
        return token.lower() == "true"
    if (token.startswith('"') and token.endswith('"')) or (
        token.startswith("'") and token.endswith("'")
    ):
        return token[1:-1]
    return token


# ---------------------------------------------------------------------------
# Manifest loaders
# ---------------------------------------------------------------------------


def load_seeds(path: Path = SEEDS_PATH) -> List[Seed]:
    if not path.exists():
        raise FileNotFoundError(f"seed manifest missing: {path}")
    data = _load_yaml(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SeedParseError("issue-seeds.yml must have a top-level mapping with a 'seeds:' key")
    seeds_raw = data.get("seeds")
    if not isinstance(seeds_raw, list):
        raise SeedParseError("issue-seeds.yml must define a top-level 'seeds:' list")

    seeds: List[Seed] = []
    for entry in seeds_raw:
        if not isinstance(entry, dict):
            raise SeedParseError(f"each seed entry must be a mapping, got {entry!r}")
        roadmap_id = str(entry.get("roadmap_id") or "").strip()
        title = str(entry.get("title") or "").strip()
        body = str(entry.get("body") or "").strip()
        labels = entry.get("labels") or []
        if not isinstance(labels, list):
            raise SeedParseError(f"labels must be a list for {roadmap_id!r}")
        if not roadmap_id or not title:
            raise SeedParseError(f"seed entry needs roadmap_id and title: {entry!r}")
        if roadmap_id not in title:
            raise SeedParseError(
                f"seed title for {roadmap_id} must contain the roadmap id for dedup ({title!r})"
            )
        seeds.append(
            Seed(
                roadmap_id=roadmap_id,
                title=title,
                body=body,
                labels=[str(lbl) for lbl in labels],
                good_first=bool(entry.get("good_first") or "good first issue" in labels),
            )
        )
    return seeds


def load_labels(path: Path = LABELS_PATH) -> List[Label]:
    if not path.exists():
        raise FileNotFoundError(f"labels manifest missing: {path}")
    data = _load_yaml(path.read_text(encoding="utf-8"))
    # ``labels.yml`` is a top-level list, not a mapping with a "labels" key.
    if isinstance(data, dict):
        entries = data.get("labels") or list(data.values())
    elif isinstance(data, list):
        entries = data
    else:
        raise SeedParseError("labels.yml must contain a top-level list of label objects")
    if not isinstance(entries, list):
        raise SeedParseError("labels.yml must contain a top-level list of label objects")

    labels: List[Label] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise SeedParseError(f"label entries must be mappings, got {entry!r}")
        name = str(entry.get("name") or "").strip()
        color = str(entry.get("color") or "").strip().lstrip("#")
        description = str(entry.get("description") or "").strip()
        if not name:
            raise SeedParseError(f"label entry missing name: {entry!r}")
        if not re.fullmatch(r"[0-9a-fA-F]{6}", color):
            raise SeedParseError(f"label {name!r} has invalid hex color: {color!r}")
        labels.append(Label(name=name, color=color, description=description))
    return labels


# ---------------------------------------------------------------------------
# gh CLI shims
# ---------------------------------------------------------------------------


def _run_gh(args: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _gh_available() -> bool:
    return shutil.which("gh") is not None


def existing_issue_titles(repo: str) -> List[str]:
    """Return open + closed issue titles for dedup, best-effort."""
    if not _gh_available():
        return []
    result = _run_gh(
        [
            "issue",
            "list",
            "--repo",
            repo,
            "--state",
            "all",
            "--limit",
            "500",
            "--json",
            "title",
        ]
    )
    if result.returncode != 0:
        sys.stderr.write(
            f"[seed-issues] WARN: gh issue list failed ({result.returncode}): {result.stderr.strip()}\n"
        )
        return []
    try:
        payload = json.loads(result.stdout or "[]")
    except json.JSONDecodeError:
        return []
    return [str(item.get("title", "")) for item in payload]


def issue_already_seeded(seed: Seed, existing_titles: Iterable[str]) -> bool:
    needle = seed.roadmap_id.upper()
    return any(needle in (title or "").upper() for title in existing_titles)


# ---------------------------------------------------------------------------
# Apply paths
# ---------------------------------------------------------------------------


def apply_labels(repo: str, labels: List[Label], *, dry_run: bool) -> List[str]:
    if not _gh_available():
        raise RuntimeError("gh CLI not found on PATH — install GitHub CLI or run --dry-run only")

    applied: List[str] = []
    for label in labels:
        cmd = [
            "label",
            "create",
            label.name,
            "--repo",
            repo,
            "--color",
            label.color,
            "--force",
        ]
        if label.description:
            cmd += ["--description", label.description]
        if dry_run:
            applied.append(f"DRY: gh {' '.join(cmd)}")
            continue
        result = _run_gh(cmd)
        if result.returncode != 0:
            raise RuntimeError(
                f"label create failed for {label.name!r}: {result.stderr.strip() or result.stdout.strip()}"
            )
        applied.append(label.name)
    return applied


def apply_seeds(
    repo: str,
    seeds: List[Seed],
    *,
    dry_run: bool,
    good_first_only: bool,
    once: bool,
) -> List[str]:
    selected = [s for s in seeds if (s.good_first if good_first_only else True)]

    if not _gh_available() and not dry_run:
        raise RuntimeError("gh CLI not found on PATH — install GitHub CLI or run --dry-run only")

    existing = [] if dry_run else existing_issue_titles(repo)
    actions: List[str] = []

    for seed in selected:
        if not dry_run and issue_already_seeded(seed, existing):
            actions.append(f"SKIP {seed.roadmap_id}: existing issue matches title prefix")
            continue

        cmd = [
            "issue",
            "create",
            "--repo",
            repo,
            "--title",
            seed.title,
            "--body",
            seed.body,
        ]
        for lbl in seed.labels:
            cmd += ["--label", lbl]

        if dry_run:
            actions.append(f"DRY {seed.roadmap_id}: gh {' '.join(cmd)}")
        else:
            result = _run_gh(cmd)
            if result.returncode != 0:
                raise RuntimeError(
                    f"issue create failed for {seed.roadmap_id}: "
                    f"{result.stderr.strip() or result.stdout.strip()}"
                )
            actions.append(f"OK {seed.roadmap_id}: {result.stdout.strip()}")
            if once:
                break

    return actions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="owner/repo to seed (default: %(default)s)")
    parser.add_argument("--labels", action="store_true", help="ensure labels from .github/labels.yml exist")
    parser.add_argument("--good-first", action="store_true", help="only seed entries flagged good_first / labeled 'good first issue'")
    parser.add_argument("--once", action="store_true", help="apply only the first seed (used for smoke tests)")
    parser.add_argument("--dry-run", action="store_true", default=True, help="print intended actions without calling gh (default)")
    parser.add_argument("--apply", dest="dry_run", action="store_false", help="actually call gh issue/label create")
    parser.add_argument("--json", action="store_true", help="emit a machine-readable summary on stdout")
    args = parser.parse_args(argv)

    summary: dict = {"repo": args.repo, "dry_run": args.dry_run}

    if args.labels:
        labels = load_labels()
        summary["labels"] = apply_labels(args.repo, labels, dry_run=args.dry_run)

    seeds = load_seeds()
    summary["seeds"] = apply_seeds(
        args.repo,
        seeds,
        dry_run=args.dry_run,
        good_first_only=args.good_first,
        once=args.once,
    )

    if args.json:
        json.dump(summary, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        for line in summary.get("labels", []):
            print(line)
        for line in summary["seeds"]:
            print(line)
        if args.dry_run:
            print("\n(dry-run — re-run with --apply to push to GitHub)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
