"""Compare the public provenance docs against the code that enforces them.

``docs/RELEASE_PROVENANCE.md`` and ``docs/PYTHON_ADVISORIES.md`` describe policy
that lives in executable form elsewhere, and the two drifted apart without
anything noticing. At the time this was written the document told a reader that
a tagged FFmpeg release ``>= 8.1.1`` was acceptable on an open release lane,
while ``opencut/core/ffmpeg_provenance.py`` had closed the release lane
entirely, raised the floor to 8.1.3, and moved the snapshot floor forward by
almost a month. The bundled pin named in the document was two builds behind the
one the installers actually fetch.

The tests over those documents asserted the stale strings literally, so they
kept passing while the statements became false.

Run with ``--check`` to fail on the first divergence, naming the field, the
value the code holds, and what the document says instead. No network access.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_PROVENANCE_DOC = REPO_ROOT / "docs" / "RELEASE_PROVENANCE.md"
PYTHON_ADVISORIES_DOC = REPO_ROOT / "docs" / "PYTHON_ADVISORIES.md"
APP_CONSTANTS = REPO_ROOT / "installer" / "src" / "OpenCut.Installer" / "Models" / "AppConstants.cs"


@dataclass(frozen=True)
class DocFact:
    """One executable fact that a document has to state."""

    field: str
    value: str
    document: Path
    #: How the fact should read in prose, when the raw value is not searchable.
    rendered: Optional[str] = None
    #: Statements that must NOT survive alongside this fact. Presence alone
    #: cannot catch a document that states the current value and a superseded
    #: one in the next sentence, which is exactly how this drifted.
    forbidden: tuple[str, ...] = ()

    @property
    def needle(self) -> str:
        return self.rendered if self.rendered is not None else self.value


def _states(text: str, needle: str) -> bool:
    """True when ``text`` states ``needle`` and not merely a longer token.

    Plain ``in`` let a document claiming ">= 8.1.30" satisfy a floor of
    ">= 8.1.3".
    """
    for match in re.finditer(re.escape(needle), text):
        tail = text[match.end():match.end() + 2]
        if needle[-1].isdigit():
            # A digit straight after extends the number (8.1.3 -> 8.1.30), and
            # so does a dot followed by one (8.1.3 -> 8.1.3.1). A dot that ends
            # a sentence does not.
            if tail[:1].isdigit() or (tail[:1] == "." and tail[1:2].isdigit()):
                continue
        return True
    return False


def _display_path(path: Path) -> str:
    """Repo-relative when possible, absolute otherwise (tests point elsewhere)."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _csharp_const(source: str, name: str) -> str:
    match = re.search(rf'{name}\s*=\s*"([^"]+)"', source)
    return match.group(1) if match else ""


def collect_release_provenance_facts() -> list[DocFact]:
    """Facts the release-provenance document must state, read from the code."""
    from opencut.core import embedded_media_provenance as embedded
    from opencut.core import ffmpeg_provenance as provenance

    facts: list[DocFact] = [
        DocFact(
            "ffmpeg_provenance.SNAPSHOT_FLOOR_DATE",
            provenance.SNAPSHOT_FLOOR_DATE,
            RELEASE_PROVENANCE_DOC,
        ),
        DocFact(
            "embedded_media_provenance.FIXED_FFMPEG_VERSION",
            embedded.FIXED_FFMPEG_VERSION,
            RELEASE_PROVENANCE_DOC,
            rendered=f"FFmpeg {embedded.FIXED_FFMPEG_VERSION} library floor",
        ),
    ]

    release_floor = ".".join(str(part) for part in provenance.RELEASE_FLOOR)
    if provenance.RELEASE_LANE_OPEN:
        facts.append(
            DocFact(
                "ffmpeg_provenance.RELEASE_FLOOR",
                release_floor,
                RELEASE_PROVENANCE_DOC,
                rendered=f">= {release_floor}",
            )
        )
    else:
        # The document must not advertise a lane the code refuses. Requiring the
        # word keeps a future reopening from silently leaving stale prose.
        facts.append(
            DocFact(
                "ffmpeg_provenance.RELEASE_LANE_OPEN=False",
                "closed",
                RELEASE_PROVENANCE_DOC,
                rendered="Release lane is closed",
                # The wording that offered the lane the code refuses. Stating
                # the new fact while leaving the old sentence in place is the
                # failure a presence-only check cannot see.
                forbidden=(
                    "acceptable on **either** lane",
                    "acceptable on either lane",
                ),
            )
        )

    if APP_CONSTANTS.is_file():
        source = APP_CONSTANTS.read_text(encoding="utf-8", errors="replace")
        for const, label in (
            ("BundledFfmpegVersion", "AppConstants.BundledFfmpegVersion"),
            ("BundledFfmpegPackageSha256", "AppConstants.BundledFfmpegPackageSha256"),
        ):
            value = _csharp_const(source, const)
            if value:
                facts.append(DocFact(label, value, RELEASE_PROVENANCE_DOC))

    return facts


def collect_advisory_facts() -> list[DocFact]:
    """Every waived advisory must appear in the public allow-list table."""
    from opencut.tools.pip_audit_extras import ALLOWED_ADVISORIES

    facts = []
    for advisory_id, entry in sorted(ALLOWED_ADVISORIES.items()):
        facts.append(DocFact(f"ALLOWED_ADVISORIES[{advisory_id}]", advisory_id, PYTHON_ADVISORIES_DOC))
        facts.append(
            DocFact(
                f"ALLOWED_ADVISORIES[{advisory_id}].package",
                entry.package,
                PYTHON_ADVISORIES_DOC,
                rendered=f"`{entry.package}`",
            )
        )
        for alias in entry.aliases:
            facts.append(
                DocFact(f"ALLOWED_ADVISORIES[{advisory_id}].alias", alias, PYTHON_ADVISORIES_DOC)
            )
    return facts


#: Rows in the "Floor raises" table whose new floor must match the real pin.
#: Only single-package rows are checked; rows naming several packages or an
#: extras-only lane state their scope in prose and are left to review.
_BACKTICKED = re.compile(r"`([^`]+)`")
_PACKAGE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")
_SPECIFIER = re.compile(r"^[<>=!~]")


def documented_floor_raises(doc_text: str) -> list[tuple[str, str]]:
    """Return ``(package, declared new floor)`` from the floor-raise table."""
    if "## Floor raises" not in doc_text:
        return []
    table = doc_text.split("## Floor raises", 1)[1].split("\n## ", 1)[0]
    rows: list[tuple[str, str]] = []
    for line in table.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or set(stripped) <= set("|-: "):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) < 3:
            continue
        # Column 1 names one or more packages; column 3 is the new floor. Cells
        # carry prose around the backticks ("`>=8.3.3,<9` (lock `8.4.1`)"), so
        # take the backticked tokens and keep the ones that look like what they
        # should be. A single regex over the whole row could not do both this
        # and the multi-package rows.
        packages = [
            value for value in _BACKTICKED.findall(cells[0]) if _PACKAGE_NAME.match(value)
        ]
        floors = [
            value for value in _BACKTICKED.findall(cells[2]) if _SPECIFIER.match(value)
        ]
        if not packages or not floors:
            continue
        if len(packages) == len(floors):
            # "`torch` / `torchvision` ... `>=2.10.0` / `>=0.25.0`" pairs up.
            rows.extend(zip(packages, floors))
        else:
            # One floor covering several packages, as onnxruntime does. Where
            # the counts disagree some other way, apply the first floor to each
            # package rather than skipping the row unchecked.
            rows.extend((package, floors[0]) for package in packages)
    return rows


def floor_divergences(doc_text: str) -> list[dict]:
    """Compare each documented floor raise against the declared dependency.

    A floor raise recorded in the document but never applied to
    ``pyproject.toml`` reads as a completed hardening step that did not happen.
    """
    pyproject = REPO_ROOT / "pyproject.toml"
    if not pyproject.is_file():
        return []
    declared = pyproject.read_text(encoding="utf-8", errors="replace")

    problems = []
    for package, new_floor in documented_floor_raises(doc_text):
        expected = f"{package}{new_floor}"
        if expected not in declared:
            problems.append({
                "field": f"docs/PYTHON_ADVISORIES.md floor raise for {package}",
                "expected": f'"{expected}" in pyproject.toml',
                "document": "pyproject.toml",
                "problem": "documented floor raise is not the declared dependency",
            })
    return problems


#: Any gyan.dev build token, release or snapshot flavoured.
_BUNDLED_PIN_RE = re.compile(
    r"\b(?:\d+\.\d+\.\d+|\d{4}-\d{2}-\d{2}-git-[0-9a-f]+)-[A-Za-z_]+_build-www\.gyan\.dev\b"
)


def contradicting_bundled_pins(doc_text: str) -> list[dict]:
    """Report any bundled FFmpeg pin other than the one the installers fetch.

    Presence checks cannot see a document that states the current pin and, two
    paragraphs later, an older one. This is how the release-provenance document
    ended up naming an 8.1.2 release while the installers fetched a 2026-08-03
    snapshot: both statements were in the file.
    """
    if not APP_CONSTANTS.is_file():
        return []
    current = _csharp_const(APP_CONSTANTS.read_text(encoding="utf-8", errors="replace"), "BundledFfmpegVersion")
    if not current:
        return []
    others = sorted(set(_BUNDLED_PIN_RE.findall(doc_text)) - {current})
    return [
        {
            "field": "docs/RELEASE_PROVENANCE.md bundled FFmpeg pin",
            "expected": f"{stale!r} removed; the installers fetch {current!r}",
            "document": "docs/RELEASE_PROVENANCE.md",
            "problem": "document names a bundled build the installers do not fetch",
        }
        for stale in others
    ]


def undocumented_waivers(doc_text: str) -> list[str]:
    """Return allow-list rows in the document that the code does not waive.

    A waiver the code dropped but the document still advertises is the more
    dangerous direction: it tells a reader a vulnerability was reviewed and
    accepted when nothing enforces that any more.
    """
    from opencut.tools.pip_audit_extras import ALLOWED_ADVISORIES

    known = set(ALLOWED_ADVISORIES)
    for entry in ALLOWED_ADVISORIES.values():
        known.update(entry.aliases)

    documented = set(re.findall(r"\b(?:CVE-\d{4}-\d{4,7}|GHSA-[0-9a-z]{4}-[0-9a-z]{4}-[0-9a-z]{4})\b", doc_text))
    # Only rows inside the allow-list table matter; advisories mentioned in
    # rationale prose are not claims that OpenCut waived them.
    table = doc_text.split("## Floor raises", 1)[0]
    documented &= set(re.findall(r"\b(?:CVE-\d{4}-\d{4,7}|GHSA-[0-9a-z]{4}-[0-9a-z]{4}-[0-9a-z]{4})\b", table))
    return sorted(documented - known)


DOC_FACT_SOURCES: tuple[Callable[[], Iterable[DocFact]], ...] = (
    collect_release_provenance_facts,
    collect_advisory_facts,
)


def find_divergences() -> list[dict]:
    """Return every executable fact the documents fail to state."""
    cache: dict[Path, str] = {}
    problems: list[dict] = []

    for source in DOC_FACT_SOURCES:
        for fact in source():
            if fact.document not in cache:
                cache[fact.document] = (
                    fact.document.read_text(encoding="utf-8", errors="replace")
                    if fact.document.is_file()
                    else ""
                )
            text = cache[fact.document]
            if not text:
                problems.append({
                    "field": fact.field,
                    "expected": fact.needle,
                    "document": _display_path(fact.document),
                    "problem": "document is missing",
                })
                continue
            if not _states(text, fact.needle):
                problems.append({
                    "field": fact.field,
                    "expected": fact.needle,
                    "document": _display_path(fact.document),
                    "problem": "not stated in the document",
                })
            for stale in fact.forbidden:
                if stale in text:
                    problems.append({
                        "field": fact.field,
                        "expected": f"{stale!r} removed",
                        "document": _display_path(fact.document),
                        "problem": "document still states a superseded value",
                    })

    advisories_text = cache.get(PYTHON_ADVISORIES_DOC) or (
        PYTHON_ADVISORIES_DOC.read_text(encoding="utf-8", errors="replace")
        if PYTHON_ADVISORIES_DOC.is_file()
        else ""
    )
    provenance_text = cache.get(RELEASE_PROVENANCE_DOC) or (
        RELEASE_PROVENANCE_DOC.read_text(encoding="utf-8", errors="replace")
        if RELEASE_PROVENANCE_DOC.is_file()
        else ""
    )
    problems.extend(contradicting_bundled_pins(provenance_text))
    problems.extend(floor_divergences(advisories_text))

    for stale in undocumented_waivers(advisories_text):
        problems.append({
            "field": "docs/PYTHON_ADVISORIES.md allow-list",
            "expected": f"{stale} removed, or restored to ALLOWED_ADVISORIES",
            "document": "docs/PYTHON_ADVISORIES.md",
            "problem": "document waives an advisory the code does not",
        })

    return problems


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify the provenance documents state what the code enforces."
    )
    parser.add_argument("--check", action="store_true", help="Exit non-zero on divergence.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args(argv)

    problems = find_divergences()

    if args.json:
        print(json.dumps({"ok": not problems, "divergences": problems}, indent=2, sort_keys=True))
    elif problems:
        print(f"{len(problems)} provenance documentation divergence(s):")
        for problem in problems:
            print(f"  {problem['document']}: {problem['field']}")
            print(f"    {problem['problem']}: expected to find {problem['expected']!r}")
    else:
        print("Provenance documents match the executable policy.")

    return 1 if (problems and args.check) else 0


if __name__ == "__main__":
    sys.exit(main())
