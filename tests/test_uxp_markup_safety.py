"""Static guard for UXP markup rendering sinks."""

from __future__ import annotations

from collections import Counter
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
UXP_MAIN = REPO_ROOT / "extension" / "com.opencut.uxp" / "main.js"
UXP_UTILS = REPO_ROOT / "extension" / "com.opencut.uxp" / "uxp-utils.js"

INNER_HTML_ASSIGNMENT_RE = re.compile(
    r"(?P<target>[A-Za-z0-9_.$?]+)\.innerHTML\s*=\s*(?P<rhs>.*?);",
    re.DOTALL,
)
HTML_ACCUMULATOR_RE = re.compile(r"\bhtml\s*\+=\s*(?P<rhs>.*?);", re.DOTALL)
INSERT_ADJACENT_HTML_RE = re.compile(
    r"\.insertAdjacentHTML\s*\(\s*(?:\"[^\"]*\"|'[^']*')\s*,\s*(?P<rhs>.*?)\)\s*;",
    re.DOTALL,
)
OUTER_HTML_ASSIGNMENT_RE = re.compile(
    r"(?P<target>[A-Za-z0-9_.$?]+)\.outerHTML\s*=\s*(?P<rhs>.*?);",
    re.DOTALL,
)
TEMPLATE_EXPR_RE = re.compile(r"\$\{(?P<expr>[^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
FUNCTION_RE = re.compile(r"(?:async\s+)?function\s+(?P<name>[A-Za-z0-9_]+)\s*\(")

EXPECTED_INNER_HTML_SINKS = Counter(
    {
        ("showToast", "toast"): 1,
        ("resetSearchResults", "list"): 1,
        ("scanProjectClips", "datalist"): 1,
        ("scanProjectClips", "clipSelect"): 1,
        ("showCutResult", "body"): 1,
        ("runFootageSearch", "list"): 2,
        ("runFootageSearch", "el"): 1,
        ("loadSequenceInfo", "grid"): 2,
        ("uxpLoadEngines", "grid"): 3,
        ("uxpLoadMigrationRisk", "grid"): 4,
        ("renderShortsBundleSummaryUxp", "list"): 1,
        ("renderPlan", "list"): 1,
        ("renderReview", "notes"): 1,
        ("populateSelects", "sel"): 1,
        # Plugin-trust dashboard sinks: empty-state/loading/error branches
        # interpolate only escaped static i18n strings; the populated grid
        # assigns the pre-escaped `html` built by the plugin row helpers.
        ("renderPluginTrustDashboard", "grid"): 2,
        ("uxpLoadPluginTrust", "grid"): 2,
    }
)

EXPECTED_HTML_ACCUMULATORS = Counter(
    {
        ("uxpLoadEngines", "html"): 12,
        # Plugin-trust dashboard: every dynamic value (plugin/quarantine names,
        # versions, status/summary text, action routes) is wrapped in
        # UIController.escapeHtml by the pluginTrustRowHtml/pluginQuarantineRowHtml/
        # pluginMarketplaceRowsHtml helpers before being accumulated here.
        ("renderPluginTrustDashboard", "html"): 3,
    }
)

REQUIRED_ESCAPED_FIELD_PATTERNS = {
    "clip names and paths": (
        "UIController.escapeHtml(c.path)",
        "UIController.escapeHtml(c.name)",
    ),
    "search result cards": (
        "UIController.escapeHtml(card.label)",
        "UIController.escapeHtml(card.kindLabel)",
        "UIController.escapeHtml(card.timeLabel)",
        "UIController.escapeHtml(card.path",
        "UIController.escapeHtml(card.preview)",
        "UIController.escapeHtml(card.scoreLabel)",
    ),
    "engine routing rows": (
        "UIController.escapeHtml(domainId)",
        "UIController.escapeHtml(domain)",
        "UIController.escapeHtml(modeClass)",
        "UIController.escapeHtml(modeLabel)",
        "UIController.escapeHtml(summary)",
        "UIController.escapeHtml(eng.name)",
        "UIController.escapeHtml(label)",
    ),
    "migration dashboard rows": (
        "UIController.escapeHtml(row.name",
        "UIController.escapeHtml(stateClass)",
        "UIController.escapeHtml(statusLabel)",
        "UIController.escapeHtml(summaryText)",
        "UIController.escapeHtml(tags)",
    ),
}

RAW_TEMPLATE_FIELD_PATTERNS = (
    "${c.path}",
    "${c.name}",
    "${card.label}",
    "${card.kindLabel}",
    "${card.timeLabel}",
    "${card.path}",
    "${card.preview}",
    "${card.scoreLabel}",
    "${eng.name}",
    "${eng.display_name}",
    "${label}",
    "${row.name}",
    "${row.role}",
    "${row.replacement_plan}",
    "${summaryText}",
    "${tags}",
)

SAFE_TEMPLATE_EXPR_PATTERNS = (
    "UIController.escapeHtml(",
    "escapeHtml(",
    "formatTimecode(",
    "formatCompactDuration(",
)
SAFE_EXACT_TEMPLATE_EXPRS = {
    "icons[tone] ?? icons.info",
    "sel",
}

SAFE_LITERAL_EXPR_RE = re.compile(
    r"""
    \A\s*
    (?:
      "" |
      '' |
      ``
    )
    \s*\Z
    """,
    re.VERBOSE,
)


def _source() -> str:
    return UXP_MAIN.read_text(encoding="utf-8")


def _line_for(source: str, offset: int) -> int:
    return source[:offset].count("\n") + 1


def _function_for(source: str, offset: int) -> str:
    current = "<module>"
    for match in FUNCTION_RE.finditer(source[:offset]):
        current = match.group("name")
    return current


def _is_empty_assignment(rhs: str) -> bool:
    return SAFE_LITERAL_EXPR_RE.match(rhs.strip()) is not None


def _template_expressions(snippet: str) -> list[str]:
    return [match.group("expr").strip() for match in TEMPLATE_EXPR_RE.finditer(snippet)]


def _unsafe_expressions(snippet: str) -> list[str]:
    unsafe: list[str] = []
    for expr in _template_expressions(snippet):
        if expr in SAFE_EXACT_TEMPLATE_EXPRS:
            continue
        if any(pattern in expr for pattern in SAFE_TEMPLATE_EXPR_PATTERNS):
            continue
        unsafe.append(expr)
    return unsafe


def test_uxp_escape_helper_escapes_attribute_and_markup_delimiters():
    body = UXP_UTILS.read_text(encoding="utf-8")

    for needle in ('replace(/&/g, "&amp;")', 'replace(/</g, "&lt;")', 'replace(/>/g, "&gt;")'):
        assert needle in body
    assert 'replace(/"/g, "&quot;")' in body
    assert "replace(/'/g, \"&#39;\")" in body


def test_uxp_inner_html_assignments_are_reviewed_and_escaped():
    source = _source()
    assignments = list(INNER_HTML_ASSIGNMENT_RE.finditer(source))

    assert assignments, "UXP panel should have reviewed innerHTML sinks"

    actual_sinks = Counter(
        (
            _function_for(source, match.start()),
            match.group("target").split(".")[-1].replace("?", ""),
        )
        for match in assignments
    )
    assert actual_sinks == EXPECTED_INNER_HTML_SINKS

    failures: list[str] = []
    for match in assignments:
        line = _line_for(source, match.start())
        target = match.group("target").split(".")[-1].replace("?", "")
        rhs = match.group("rhs")
        if _is_empty_assignment(rhs):
            continue
        unsafe = _unsafe_expressions(rhs)
        if unsafe:
            failures.append(
                f"line {line}: unescaped template expression(s): {', '.join(unsafe)}"
            )

    assert not failures, "\n".join(failures)


def test_uxp_html_accumulators_escape_dynamic_template_values():
    source = _source()
    snippets = list(HTML_ACCUMULATOR_RE.finditer(source))

    assert snippets, "UXP panel should have reviewed HTML accumulator snippets"

    actual_accumulators = Counter(
        (_function_for(source, match.start()), "html")
        for match in snippets
    )
    assert actual_accumulators == EXPECTED_HTML_ACCUMULATORS

    failures: list[str] = []
    for match in snippets:
        line = _line_for(source, match.start())
        unsafe = _unsafe_expressions(match.group("rhs"))
        if unsafe:
            failures.append(
                f"line {line}: unescaped accumulator expression(s): {', '.join(unsafe)}"
            )

    assert not failures, "\n".join(failures)


def test_uxp_insert_adjacent_and_outer_html_sinks_are_escaped():
    """Fail-closed guard for the non-innerHTML markup sinks.

    The UXP panel currently has zero insertAdjacentHTML/outerHTML sinks; any
    future addition is scanned here and must escape every dynamic template
    expression (same safelist as the innerHTML guard).
    """
    source = _source()
    sinks = list(INSERT_ADJACENT_HTML_RE.finditer(source)) + list(
        OUTER_HTML_ASSIGNMENT_RE.finditer(source)
    )

    failures: list[str] = []
    for match in sinks:
        line = _line_for(source, match.start())
        rhs = match.group("rhs")
        if _is_empty_assignment(rhs):
            continue
        unsafe = _unsafe_expressions(rhs)
        if unsafe:
            failures.append(
                f"line {line}: unescaped markup sink expression(s): {', '.join(unsafe)}"
            )

    assert not failures, "\n".join(failures)


def test_uxp_user_and_backend_fixture_fields_are_escaped_before_markup():
    source = _source()

    missing: list[str] = []
    for fixture_name, patterns in REQUIRED_ESCAPED_FIELD_PATTERNS.items():
        for pattern in patterns:
            if pattern not in source:
                missing.append(f"{fixture_name}: missing {pattern}")

    raw = [pattern for pattern in RAW_TEMPLATE_FIELD_PATTERNS if pattern in source]

    assert not missing, "\n".join(missing)
    assert not raw, "raw template field(s) in markup path: " + ", ".join(raw)
