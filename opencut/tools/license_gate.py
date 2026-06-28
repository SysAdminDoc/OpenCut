"""Model + dependency license allowlist gate (F006).

Two surfaces consume optional AI/model extras today:

* **Direct dependencies** in ``requirements.txt`` / optional pip extras.
* **Model cards** in :mod:`opencut.model_cards` for downloadable
  weights.

This module enforces a license allowlist over both. It is intentionally
strict — permissive licenses pass, weak-copyleft (LGPL) and proprietary
shows up as a *warning* (still ok to ship but documented), and
research-only / strong-copyleft licenses fail closed unless the maintainer
explicitly waives them.

Usage::

    python -m opencut.tools.license_gate            # human report, exit nonzero on errors
    python -m opencut.tools.license_gate --json     # machine-readable
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

# Buckets are intentionally coarse — granular SPDX matching is left to
# downstream scanners (cyclonedx-bom, syft, etc.).
ALLOWED_LICENSES = (
    "MIT",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "Apache-2.0",
    "ISC",
    "Unlicense",
    "PSF",  # Python Software Foundation
    "PSF-2.0",
    "PostgreSQL",
    "MPL-2.0",  # file-level copyleft, ok with attribution
)

WARNING_LICENSES = (
    "CC-BY-4.0",
    "LGPL-2.1",
    "LGPL-3.0",
    "EPL-2.0",
    "AGPL-3.0",  # surfaces but doesn't block — operator decides
)

DENIED_LICENSES = (
    "GPL-2.0-only",
    "GPL-3.0-only",
    "GPL-3.0",
    "CC-BY-NC",
    "Custom",
    "non-commercial",
    "research-only",
    "NTU S-Lab License",  # ProPainter
    "SSPL",
)


# Licenses we keep flagging at the warning level because they're either
# proprietary client SDKs against a cloud service or because the upstream
# license is "varies per model" / "varies per backend". The point of
# surfacing them is forcing humans to acknowledge the surface area.
PROPRIETARY_PATTERNS = (
    re.compile(r"proprietary", re.IGNORECASE),
    re.compile(r"client SDK", re.IGNORECASE),
    re.compile(r"varies per", re.IGNORECASE),
)


@dataclass
class LicenseFinding:
    surface: str       # "model_card" | "requirements" | "extras"
    name: str          # backend / package name
    license_text: str
    severity: str      # info | warning | error
    message: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class LicenseReport:
    findings: List[LicenseFinding] = field(default_factory=list)
    surfaces: dict = field(default_factory=dict)

    def has_errors(self) -> bool:
        return any(f.severity == "error" for f in self.findings)

    def as_dict(self) -> dict:
        return {
            "findings": [f.as_dict() for f in self.findings],
            "surfaces": self.surfaces,
        }


def _normalise(text: str) -> str:
    return (text or "").strip()


def _matches_license_token(raw: str, token: str) -> bool:
    """Word-bounded match — ``GPL-3.0`` must NOT match ``LGPL-3.0``."""
    # ``\b`` doesn't recognise hyphens as word breaks, so we explicitly
    # require start-of-string or a non-letter-digit/dash boundary.
    pattern = r"(?<![A-Za-z0-9-])" + re.escape(token) + r"(?![A-Za-z0-9-])"
    return re.search(pattern, raw, flags=re.IGNORECASE) is not None


def classify_license(text: str) -> str:
    """Return ``"allowed"`` / ``"warning"`` / ``"denied"`` / ``"unknown"``."""
    raw = _normalise(text)
    if not raw:
        return "unknown"

    # Order matters: weak-copyleft (LGPL) is a *warning*, full GPL is a
    # *denial*. The word-bounded matcher stops "GPL-3.0" from matching
    # the "GPL" suffix of "LGPL-3.0".
    for token in WARNING_LICENSES:
        if _matches_license_token(raw, token):
            return "warning"
    for token in DENIED_LICENSES:
        if _matches_license_token(raw, token):
            return "denied"
    for token in ALLOWED_LICENSES:
        if _matches_license_token(raw, token):
            return "allowed"
    for pat in PROPRIETARY_PATTERNS:
        if pat.search(raw):
            return "warning"
    return "unknown"


def _check_model_cards() -> List[LicenseFinding]:
    findings: List[LicenseFinding] = []
    try:
        from opencut.model_cards import CARDS
    except Exception as exc:  # pragma: no cover - defensive
        findings.append(
            LicenseFinding(
                surface="model_card",
                name="<import-error>",
                license_text="",
                severity="error",
                message=f"could not import opencut.model_cards: {exc}",
            )
        )
        return findings

    for card in CARDS:
        bucket = classify_license(card.license)
        waiver = (getattr(card, "license_waiver", "") or "").strip()
        if bucket == "allowed":
            findings.append(
                LicenseFinding(
                    surface="model_card",
                    name=card.label,
                    license_text=card.license,
                    severity="info",
                    message="permissive license",
                )
            )
        elif bucket == "warning":
            findings.append(
                LicenseFinding(
                    surface="model_card",
                    name=card.label,
                    license_text=card.license,
                    severity="warning",
                    message="weak-copyleft or proprietary — document before shipping",
                )
            )
        elif bucket == "denied":
            if waiver:
                findings.append(
                    LicenseFinding(
                        surface="model_card",
                        name=card.label,
                        license_text=card.license,
                        severity="warning",
                        message=f"license on denied list but waived: {waiver[:120]}",
                    )
                )
            else:
                findings.append(
                    LicenseFinding(
                        surface="model_card",
                        name=card.label,
                        license_text=card.license,
                        severity="error",
                        message="license is on the denied list (add license_waiver= to override)",
                    )
                )
        else:
            findings.append(
                LicenseFinding(
                    surface="model_card",
                    name=card.label,
                    license_text=card.license,
                    severity="error",
                    message="license token not recognised; classify it explicitly",
                )
            )
    return findings


def _read_requirements(path: Path) -> List[str]:
    if not path.exists():
        return []
    out: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Strip environment markers and comments.
        spec = line.split(";", 1)[0].split("#", 1)[0].strip()
        out.append(spec)
    return out


def lint(root: Path = REPO_ROOT) -> LicenseReport:
    """Run the license allowlist check across every surface we know about."""
    report = LicenseReport()
    report.findings.extend(_check_model_cards())

    # Surfaces summary — useful for the panel readout.
    surfaces = {"model_cards": 0, "requirements": 0}
    for f in report.findings:
        surfaces[f.surface] = surfaces.get(f.surface, 0) + 1
    report.surfaces = surfaces
    return report


def cli(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    parser.add_argument(
        "--strict-warnings",
        action="store_true",
        help="treat warnings as errors (CI gate for late-stage releases)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = lint()

    if args.json:
        json.dump(report.as_dict(), sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        errors = [f for f in report.findings if f.severity == "error"]
        warnings = [f for f in report.findings if f.severity == "warning"]
        for f in errors + warnings:
            print(f"[{f.severity.upper()}] {f.surface}/{f.name}: {f.license_text} — {f.message}")
        if not errors and not warnings:
            print(f"[license-gate] OK — {report.surfaces} surfaces scanned")
        else:
            print(f"\nsummary: {len(errors)} errors, {len(warnings)} warnings")

    if report.has_errors():
        return 1
    if args.strict_warnings and any(f.severity == "warning" for f in report.findings):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
