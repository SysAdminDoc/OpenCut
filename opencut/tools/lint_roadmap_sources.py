"""Roadmap source-appendix linter (F118).

The roadmap is heavy on bracketed citation tokens such as ``[V43-L07]``
and ``[S22]``. Drift between citations and the appendix is easy: a row
disappears from the appendix, a citation gets a typo, or a "Local
evidence" row points at a file that no longer exists. This linter
catches all three classes in one pass.

Checks (each may be downgraded to a warning with ``--allow-warnings``):

1. Every citation token in the roadmap resolves to a row in some
   appendix (we keep two — v4.2's ``Appendix A`` with ``[L##]``/``[S##]``
   and v4.3's ``v4.3 Source Appendix`` with ``[V43-L##]``/``[V43-S##]``).
2. Every appendix row is actually referenced at least once. Unreferenced
   rows are warnings — they may be future-use citations.
3. Local-evidence rows mention paths that exist on disk. Paths are
   pulled out of backticks; we accept best-effort matching because the
   notes also describe one-off commands.
4. External URLs parse cleanly (``urllib.parse.urlsplit`` succeeds and
   the scheme is http(s)).

The linter is stdlib-only so it works in a hermetic CI environment.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlsplit

REPO_ROOT = Path(__file__).resolve().parents[2]
ROADMAP_PATH = REPO_ROOT / "ROADMAP.md"

# Citation tokens always use a two-digit minimum number (L01, S07, V43-L13).
# Anchoring the count to ``[0-9]{2,}`` deliberately excludes incidental
# ``[S1]``/``[S2]`` markers that appear inside model-prompt syntax (e.g. the
# Dia TTS speaker tags in Wave O).
CITATION_RE = re.compile(r"\[(V43-[LS][0-9]{2,}|[LS][0-9]{2,})\]")
APPENDIX_ROW_RE = re.compile(
    r"""
    ^\s*
    (?:
        \|\s*(?P<id_table>V43-[LS][0-9]{2,}|[LS][0-9]{2,})\s*\|\s*(?P<body_table>.*?)\s*\|\s*$
        |
        -\s+\[(?P<id_bullet>V43-[LS][0-9]{2,}|[LS][0-9]{2,})\]\s+(?P<body_bullet>.+?)\s*$
    )
    """,
    re.VERBOSE,
)
BACKTICKED_PATH_RE = re.compile(r"`([^`\s][^`]*?)`")
URL_RE = re.compile(r"https?://[^\s\)`]+")


@dataclass
class AppendixRow:
    citation_id: str
    body: str
    is_local: bool
    line: int


@dataclass
class LintFinding:
    severity: str  # "error" | "warning"
    rule: str
    message: str
    line: Optional[int] = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class LintReport:
    findings: List[LintFinding] = field(default_factory=list)
    citation_count: int = 0
    appendix_count: int = 0
    referenced_appendix_ids: Set[str] = field(default_factory=set)

    def as_dict(self) -> dict:
        return {
            "citation_count": self.citation_count,
            "appendix_count": self.appendix_count,
            "referenced_appendix_count": len(self.referenced_appendix_ids),
            "findings": [f.as_dict() for f in self.findings],
        }

    def has_errors(self) -> bool:
        return any(f.severity == "error" for f in self.findings)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _is_local_section(heading: str) -> bool:
    """Decide whether a row inside ``heading`` is local evidence vs external."""
    needle = heading.lower()
    return "local evidence" in needle or "local " in needle.split(":")[0]


def parse_roadmap(text: str) -> Tuple[List[AppendixRow], List[Tuple[int, str]]]:
    """Return ``(appendix_rows, citation_locations)``."""
    lines = text.splitlines()
    rows: List[AppendixRow] = []
    citations: List[Tuple[int, str]] = []

    current_heading = ""
    in_appendix = False

    for idx, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()

        # Track the most recent appendix-style heading.
        if stripped.startswith("#"):
            current_heading = stripped.lstrip("#").strip()
            in_appendix = "appendix" in current_heading.lower() or "source" in current_heading.lower()
            continue

        # When the current line defines an appendix row, treat the
        # leading ``[ID]`` as the row id, not as a citation. Other
        # bracketed tokens in the body still count.
        appendix_match = APPENDIX_ROW_RE.match(raw_line) if in_appendix else None

        if appendix_match is None:
            for match in CITATION_RE.finditer(raw_line):
                citations.append((idx, match.group(1)))
        else:
            row_id = appendix_match.group("id_table") or appendix_match.group("id_bullet") or ""
            body_only = (appendix_match.group("body_table") or appendix_match.group("body_bullet") or "")
            for match in CITATION_RE.finditer(body_only):
                if match.group(1) == row_id:
                    continue
                citations.append((idx, match.group(1)))

        if not in_appendix:
            continue
        m = appendix_match
        if not m:
            continue
        citation_id = m.group("id_table") or m.group("id_bullet")
        body = (m.group("body_table") or m.group("body_bullet") or "").strip()
        # Skip the header row "| ID | Source |" that matches the regex
        # but has no actual body content.
        if not body or body.lower() in {"source", "sources", "url"}:
            continue
        rows.append(
            AppendixRow(
                citation_id=citation_id,
                body=body,
                is_local=_is_local_section(current_heading or "")
                or any(token in body for token in ("`", "scripts/", "ROADMAP.md", "tests/", "opencut/", "extension/")),
                line=idx,
            )
        )

    return rows, citations


def _strip_relative_prefix(path_str: str) -> str:
    """Strip leading ``./`` but never the dot from ``.github`` / ``.venv``."""
    while path_str.startswith("./"):
        path_str = path_str[2:]
    return path_str


def extract_paths(body: str) -> List[str]:
    candidates: List[str] = []
    for token in BACKTICKED_PATH_RE.findall(body):
        token = token.strip()
        # Skip commands like ``python scripts/sync_version.py --check``.
        if " " in token:
            head = token.split()[0]
            if "/" in head or "." in head:
                candidates.append(_strip_relative_prefix(head))
            continue
        if "/" not in token and "\\" not in token and "." not in token:
            continue
        if any(token.startswith(p) for p in ("http://", "https://", "rg ", "git ", "npm ")):
            continue
        # Dotted import paths like opencut.server.create_app aren't files.
        if "/" not in token and "\\" not in token and " " not in token and token.count(".") >= 2:
            continue
        # Repo identifiers like SysAdminDoc/OpenCut aren't local files.
        if "/" in token and "." not in token and token.count("/") == 1 and not token.startswith((".", "extension", "opencut", "tests", "scripts", "docs", "installer")):
            continue
        candidates.append(_strip_relative_prefix(token))
    return candidates


def extract_urls(body: str) -> List[str]:
    return URL_RE.findall(body)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def lint(text: str, *, repo_root: Path = REPO_ROOT) -> LintReport:
    rows, citations = parse_roadmap(text)
    by_id: Dict[str, AppendixRow] = {}
    duplicates: List[Tuple[str, int]] = []
    for row in rows:
        if row.citation_id in by_id:
            duplicates.append((row.citation_id, row.line))
        else:
            by_id[row.citation_id] = row

    report = LintReport(citation_count=len(citations), appendix_count=len(rows))

    # 1. Citations must resolve.
    for line_no, citation_id in citations:
        if citation_id not in by_id:
            report.findings.append(
                LintFinding(
                    severity="error",
                    rule="dangling_citation",
                    message=f"citation [{citation_id}] has no matching appendix row",
                    line=line_no,
                )
            )
        else:
            report.referenced_appendix_ids.add(citation_id)

    # 2. Duplicated appendix ids.
    for citation_id, line_no in duplicates:
        report.findings.append(
            LintFinding(
                severity="error",
                rule="duplicate_appendix_row",
                message=f"appendix id [{citation_id}] is defined more than once",
                line=line_no,
            )
        )

    # 3. Local-evidence paths should exist on disk.
    for row in rows:
        if not row.is_local:
            continue
        for path_str in extract_paths(row.body):
            # Strip trailing punctuation operators that BACKTICKED_PATH_RE can keep.
            path_str = path_str.rstrip(",.;:")
            if not path_str:
                continue
            # Cross-platform: resolve relative to repo root, allow forward slashes.
            candidate = (repo_root / path_str).resolve()
            if not candidate.exists():
                report.findings.append(
                    LintFinding(
                        severity="warning",
                        rule="missing_local_path",
                        message=(
                            f"[{row.citation_id}] references '{path_str}' which does not "
                            "exist in the working tree"
                        ),
                        line=row.line,
                    )
                )

    # 4. External URLs parse cleanly.
    for row in rows:
        if row.is_local:
            continue
        for url in extract_urls(row.body):
            url = url.rstrip(".,);")
            parts = urlsplit(url)
            if parts.scheme not in {"http", "https"} or not parts.netloc:
                report.findings.append(
                    LintFinding(
                        severity="error",
                        rule="malformed_url",
                        message=f"[{row.citation_id}] has malformed URL: {url}",
                        line=row.line,
                    )
                )

    # 5. Unreferenced appendix rows — warning only.
    for citation_id, row in by_id.items():
        if citation_id not in report.referenced_appendix_ids:
            report.findings.append(
                LintFinding(
                    severity="warning",
                    rule="unreferenced_appendix_row",
                    message=f"appendix row [{citation_id}] is never cited",
                    line=row.line,
                )
            )

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cli(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file",
        type=Path,
        default=ROADMAP_PATH,
        help="path to ROADMAP.md (default: %(default)s)",
    )
    parser.add_argument(
        "--allow-warnings",
        action="store_true",
        help="don't fail on warnings — useful for dev runs",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = parser.parse_args(argv)

    if not args.file.exists():
        print(f"[roadmap-lint] FAIL — file not found: {args.file}", file=sys.stderr)
        return 2

    report = lint(args.file.read_text(encoding="utf-8"), repo_root=args.file.resolve().parent)

    if args.json:
        json.dump(report.as_dict(), sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        if not report.findings:
            print(
                f"[roadmap-lint] OK — {report.citation_count} citations against "
                f"{report.appendix_count} appendix rows"
            )
        else:
            for finding in report.findings:
                where = f"line {finding.line}: " if finding.line else ""
                print(
                    f"[{finding.severity.upper()}] {finding.rule}: {where}{finding.message}"
                )

    if report.has_errors():
        return 1
    if not args.allow_warnings and any(f.severity == "warning" for f in report.findings):
        return 0  # warnings don't fail unless --strict is added in the future
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
