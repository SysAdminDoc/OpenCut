"""Static WCAG contrast audit for committed panel design tokens.

The CEP and UXP panels are not browser-rendered during PR-fast, but their
foreground/background design tokens are committed CSS and can be checked
deterministically. This gate audits curated token pairs that represent normal
panel text, secondary text, muted UI chrome, and accent-button foregrounds.

Usage::

    python -m opencut.tools.contrast_audit --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
AA_NORMAL_TEXT = 4.5
AA_LARGE_OR_NON_TEXT = 3.0


@dataclass(frozen=True)
class ContrastPair:
    foreground: str
    background: str
    minimum_ratio: float = AA_NORMAL_TEXT
    usage: str = "normal text"


@dataclass(frozen=True)
class ContrastTarget:
    path: Path
    pairs: Tuple[ContrastPair, ...]


@dataclass(frozen=True)
class TokenBlock:
    path: Path
    selector: str
    line: int
    tokens: Dict[str, str]


@dataclass(frozen=True)
class ContrastFinding:
    path: str
    selector: str
    line: int
    foreground: str
    foreground_value: str
    background: str
    background_value: str
    ratio: float
    minimum_ratio: float
    usage: str
    status: str

    def as_dict(self) -> dict:
        return asdict(self)


PANEL_PAIRS = (
    ContrastPair("text-primary", "bg-void", usage="CEP body text"),
    ContrastPair("text-primary", "bg-card", usage="CEP card text"),
    ContrastPair("text-secondary", "bg-card", usage="CEP secondary card text"),
    ContrastPair("text-white", "neon-cyan-dark", usage="CEP primary button text"),
    ContrastPair("bg-void", "neon-cyan", usage="CEP accent button text"),
    ContrastPair("bg-void", "neon-green", usage="CEP success badge text"),
    ContrastPair("bg-void", "neon-orange", usage="CEP warning badge text"),
    ContrastPair(
        "text-muted",
        "bg-elevated",
        minimum_ratio=AA_LARGE_OR_NON_TEXT,
        usage="CEP muted metadata and non-primary UI chrome",
    ),
)

UXP_PAIRS = (
    ContrastPair("text-primary", "bg-primary", usage="UXP body text"),
    ContrastPair("text-primary", "bg-card", usage="UXP card text"),
    ContrastPair("text-secondary", "bg-card", usage="UXP secondary card text"),
    ContrastPair("text-primary", "bg-hover", usage="UXP hover-state text"),
    ContrastPair("bg-primary", "accent", usage="UXP accent button text"),
    ContrastPair("bg-primary", "success", usage="UXP success badge text"),
    ContrastPair("bg-primary", "warning", usage="UXP warning badge text"),
)

DEFAULT_TARGETS = (
    ContrastTarget(
        Path("extension/com.opencut.panel/client/style.css"),
        PANEL_PAIRS,
    ),
    ContrastTarget(
        Path("extension/com.opencut.uxp/style.css"),
        UXP_PAIRS,
    ),
)

ROOT_BLOCK_RE = re.compile(r"(?P<selector>:root)\s*\{", re.MULTILINE)
TOKEN_RE = re.compile(r"--(?P<name>[\w-]+)\s*:\s*(?P<value>[^;]+);")
HEX_RE = re.compile(r"#(?P<hex>[0-9a-fA-F]{3}|[0-9a-fA-F]{6})\b")
RGB_RE = re.compile(
    r"rgba?\(\s*(?P<r>\d{1,3})\s*,\s*(?P<g>\d{1,3})\s*,\s*(?P<b>\d{1,3})"
)
VAR_RE = re.compile(r"var\(\s*--(?P<name>[\w-]+)")


def _find_matching_brace(text: str, open_index: int) -> int:
    depth = 0
    for index in range(open_index, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("unclosed CSS block")


def parse_root_token_blocks(path: Path) -> List[TokenBlock]:
    """Parse ``:root`` CSS custom-property blocks from a stylesheet."""
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks: List[TokenBlock] = []
    for index, match in enumerate(ROOT_BLOCK_RE.finditer(text), start=1):
        close = _find_matching_brace(text, match.end() - 1)
        body = text[match.end():close]
        tokens: Dict[str, str] = {}
        for token_match in TOKEN_RE.finditer(body):
            tokens[token_match.group("name")] = token_match.group("value").strip()
        if not tokens:
            continue
        line = text.count("\n", 0, match.start()) + 1
        blocks.append(
            TokenBlock(
                path=path,
                selector=f":root[{index}]",
                line=line,
                tokens=tokens,
            )
        )
    return blocks


def _expand_short_hex(value: str) -> str:
    if len(value) == 3:
        return "".join(ch * 2 for ch in value)
    return value


def _rgb_from_css_value(
    value: str,
    tokens: Dict[str, str],
    *,
    seen: Optional[set[str]] = None,
) -> Optional[Tuple[int, int, int]]:
    """Resolve a token value to an opaque RGB triplet when possible."""
    seen = seen or set()
    hex_match = HEX_RE.search(value)
    if hex_match:
        hex_value = _expand_short_hex(hex_match.group("hex"))
        return (
            int(hex_value[0:2], 16),
            int(hex_value[2:4], 16),
            int(hex_value[4:6], 16),
        )

    rgb_match = RGB_RE.search(value)
    if rgb_match:
        rgb = tuple(int(rgb_match.group(name)) for name in ("r", "g", "b"))
        if all(0 <= channel <= 255 for channel in rgb):
            return rgb
        return None

    var_match = VAR_RE.search(value)
    if var_match:
        name = var_match.group("name")
        if name in seen or name not in tokens:
            return None
        return _rgb_from_css_value(tokens[name], tokens, seen={*seen, name})

    return None


def _linearize(channel: float) -> float:
    channel = channel / 255.0
    if channel <= 0.04045:
        return channel / 12.92
    return ((channel + 0.055) / 1.055) ** 2.4


def relative_luminance(rgb: Tuple[int, int, int]) -> float:
    red, green, blue = (_linearize(channel) for channel in rgb)
    return (0.2126 * red) + (0.7152 * green) + (0.0722 * blue)


def contrast_ratio(foreground: Tuple[int, int, int], background: Tuple[int, int, int]) -> float:
    fg_lum = relative_luminance(foreground)
    bg_lum = relative_luminance(background)
    lighter = max(fg_lum, bg_lum)
    darker = min(fg_lum, bg_lum)
    return (lighter + 0.05) / (darker + 0.05)


def audit_block(
    block: TokenBlock,
    pairs: Iterable[ContrastPair],
    *,
    repo_root: Path = REPO_ROOT,
) -> List[ContrastFinding]:
    findings: List[ContrastFinding] = []
    for pair in pairs:
        if pair.foreground not in block.tokens or pair.background not in block.tokens:
            continue
        foreground = _rgb_from_css_value(block.tokens[pair.foreground], block.tokens)
        background = _rgb_from_css_value(block.tokens[pair.background], block.tokens)
        if foreground is None or background is None:
            continue
        ratio = contrast_ratio(foreground, background)
        try:
            rel_path = block.path.relative_to(repo_root)
        except ValueError:
            rel_path = block.path
        findings.append(
            ContrastFinding(
                path=rel_path.as_posix(),
                selector=block.selector,
                line=block.line,
                foreground=pair.foreground,
                foreground_value=block.tokens[pair.foreground],
                background=pair.background,
                background_value=block.tokens[pair.background],
                ratio=round(ratio, 2),
                minimum_ratio=pair.minimum_ratio,
                usage=pair.usage,
                status="ok" if ratio >= pair.minimum_ratio else "fail",
            )
        )
    return findings


def build_contrast_report(
    targets: Sequence[ContrastTarget] = DEFAULT_TARGETS,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict:
    findings: List[ContrastFinding] = []
    missing_files: List[str] = []
    unaudited_targets: List[str] = []

    for target in targets:
        path = target.path if target.path.is_absolute() else repo_root / target.path
        if not path.is_file():
            missing_files.append(target.path.as_posix())
            continue
        target_findings: List[ContrastFinding] = []
        for block in parse_root_token_blocks(path):
            target_findings.extend(audit_block(block, target.pairs, repo_root=repo_root))
        if not target_findings:
            unaudited_targets.append(target.path.as_posix())
        findings.extend(target_findings)

    failures = [finding for finding in findings if finding.status == "fail"]
    status = "fail" if failures or missing_files or unaudited_targets else "ok"
    return {
        "status": status,
        "summary": {
            "targets": len(targets),
            "audited_pairs": len(findings),
            "failures": len(failures),
            "missing_files": len(missing_files),
            "unaudited_targets": len(unaudited_targets),
        },
        "thresholds": {
            "aa_normal_text": AA_NORMAL_TEXT,
            "aa_large_or_non_text": AA_LARGE_OR_NON_TEXT,
        },
        "missing_files": missing_files,
        "unaudited_targets": unaudited_targets,
        "findings": [finding.as_dict() for finding in findings],
    }


def _print_text_report(report: dict) -> None:
    summary = report["summary"]
    print(
        "contrast-audit: "
        f"{report['status']} "
        f"({summary['audited_pairs']} pairs, {summary['failures']} failures)"
    )
    for path in report["missing_files"]:
        print(f"missing stylesheet: {path}")
    for path in report["unaudited_targets"]:
        print(f"no auditable token pairs: {path}")
    for finding in report["findings"]:
        if finding["status"] != "fail":
            continue
        print(
            f"{finding['path']}:{finding['line']} {finding['selector']} "
            f"{finding['foreground']} on {finding['background']} "
            f"ratio {finding['ratio']} < {finding['minimum_ratio']} "
            f"({finding['usage']})"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args(argv)

    report = build_contrast_report()
    if args.json:
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        _print_text_report(report)
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
