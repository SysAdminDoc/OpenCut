#!/usr/bin/env python3
"""
CEP locale drift gate (RESEARCH_FEATURE_PLAN_2026-05-25 Q6).

The CEP panel ships ``client/locales/en.json`` with hundreds of keys.
Historic drift left ~142 of 426 keys without a consumer in
``index.html`` or ``main.js``, and several user-visible strings in
``main.js`` (lines 2100, 2125, 2264 per the research pass) were
hardcoded English instead of going through ``t("…", fallback)``.

This linter:
  1. Extracts every key in ``en.json``.
  2. Walks ``index.html`` for
     ``data-i18n[-title|-label|-alt|-placeholder|-aria-label]=…``
     attributes.
  3. Walks ``main.js`` for ``t("…")`` / ``t('…')`` calls and structured
     locale metadata fields such as ``labelKey: "…"``.
  4. Reports:
       - dead keys (in en.json, no consumer)
       - missing keys (consumed but not in en.json)

Dead keys and missing keys both fail at zero tolerance. The historic
dead-key cleanup is complete, so new unused locale keys should be
removed or wired before they land.

Usage:
    python scripts/i18n_lint.py            # report
    python scripts/i18n_lint.py --check    # exit 1 if drift exceeds floor
    python scripts/i18n_lint.py --json     # JSON for CI
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCALES = ROOT / "extension" / "com.opencut.panel" / "client" / "locales"
INDEX_HTML = ROOT / "extension" / "com.opencut.panel" / "client" / "index.html"
MAIN_JS = ROOT / "extension" / "com.opencut.panel" / "client" / "main.js"

# The historic dead-key cleanup is complete. Keep the floor at zero so new
# unused locale keys fail the gate immediately.
DEAD_KEY_BASELINE = 0

# data-i18n="…" / data-i18n-title="…" / data-i18n-label="…" /
# data-i18n-alt="…" /
# data-i18n-placeholder="…" / data-i18n-aria-label="…"
HTML_I18N_RE = re.compile(
    r'data-i18n(?:-(?:title|label|alt|placeholder|aria-label))?="([^"]+)"'
)

# t("key") / t('key') — key must look like an i18n key (dotted lowercase + underscore).
JS_I18N_RE = re.compile(
    r"""\bt\s*\(\s*['"]([a-z][a-zA-Z0-9_]*(?:\.[a-zA-Z][a-zA-Z0-9_]*)+)['"]"""
)

# Structured JS metadata such as labelKey/titleKey/placeholderKey. Keep this
# intentionally narrow so unrelated fields like api_key are not counted.
JS_KEY_FIELD_RE = re.compile(
    r"""(?:\b|['"])(?:label|title|description|desc|aria|ariaLabel|placeholder|tooltip|section|group|meta|status|copy|hint)Key(?:\b|['"])\s*:\s*['"]([a-z][a-zA-Z0-9_]*(?:\.[a-zA-Z][a-zA-Z0-9_]*)+)['"]"""
)

# Count-label helpers take singular and plural locale keys as arguments, then
# call t(key, fallback) dynamically. Treat those literal key arguments as live
# consumers so plural labels cannot ship with fallback-only text.
JS_PLUGIN_COUNT_LABEL_RE = re.compile(
    r"""\bpluginCountLabel\s*\(\s*[^,]+,\s*['"]([a-z][a-zA-Z0-9_]*(?:\.[a-zA-Z][a-zA-Z0-9_]*)+)['"]\s*,\s*['"][^'"]*['"]\s*,\s*['"]([a-z][a-zA-Z0-9_]*(?:\.[a-zA-Z][a-zA-Z0-9_]*)+)['"]""",
    re.DOTALL,
)


def _load_en_keys() -> set[str]:
    en = json.loads((LOCALES / "en.json").read_text(encoding="utf-8"))
    return set(en.keys())


def _scan_html_consumers() -> set[str]:
    if not INDEX_HTML.exists():
        return set()
    return set(HTML_I18N_RE.findall(INDEX_HTML.read_text(encoding="utf-8")))


def _read_main_js() -> str:
    if not MAIN_JS.exists():
        return ""
    return MAIN_JS.read_text(encoding="utf-8")


def _scan_js_call_consumers(source: str | None = None) -> set[str]:
    if source is None:
        source = _read_main_js()
    return set(JS_I18N_RE.findall(source))


def _scan_js_metadata_consumers(source: str | None = None) -> set[str]:
    if source is None:
        source = _read_main_js()
    keys = set(JS_KEY_FIELD_RE.findall(source))
    for one_key, many_key in JS_PLUGIN_COUNT_LABEL_RE.findall(source):
        keys.add(one_key)
        keys.add(many_key)
    return keys


def _scan_js_consumers_from_source(source: str) -> set[str]:
    return _scan_js_call_consumers(source) | _scan_js_metadata_consumers(source)


def _scan_js_consumers() -> set[str]:
    return _scan_js_consumers_from_source(_read_main_js())


def evaluate() -> dict:
    keys = _load_en_keys()
    html_consumers = _scan_html_consumers()
    js_source = _read_main_js()
    js_call_consumers = _scan_js_call_consumers(js_source)
    js_metadata_consumers = _scan_js_metadata_consumers(js_source)
    js_consumers = js_call_consumers | js_metadata_consumers
    consumers = html_consumers | js_consumers
    dead = sorted(keys - consumers)
    missing = sorted(consumers - keys)
    return {
        "total_keys": len(keys),
        "html_consumers": len(html_consumers),
        "js_consumers": len(js_consumers),
        "js_call_consumers": len(js_call_consumers),
        "js_metadata_consumers": len(js_metadata_consumers),
        "unique_consumers": len(consumers),
        "dead_count": len(dead),
        "dead_keys": dead,
        "missing_count": len(missing),
        "missing_keys": missing,
        "baseline": DEAD_KEY_BASELINE,
        "dead_over_baseline": max(0, len(dead) - DEAD_KEY_BASELINE),
    }


def cmd_report(check: bool) -> int:
    e = evaluate()
    print(
        f"i18n keys: {e['total_keys']} total | "
        f"{e['unique_consumers']} consumers "
        f"({e['html_consumers']} HTML + {e['js_consumers']} JS; "
        f"{e['js_call_consumers']} t-calls + {e['js_metadata_consumers']} metadata keys)"
    )
    print(f"  dead keys: {e['dead_count']} (baseline allowed: {e['baseline']})")
    print(f"  missing keys: {e['missing_count']}")
    if e["missing_keys"]:
        print("\n  Missing keys (consumed but not in en.json):")
        for k in e["missing_keys"][:20]:
            print(f"    {k}")
        if e["missing_count"] > 20:
            print(f"    ... +{e['missing_count'] - 20} more")
    if check:
        fail = False
        if e["missing_count"] > 0:
            print(
                f"\nFAIL: {e['missing_count']} key(s) consumed but not in en.json — "
                "add them or fix the typo.",
                file=sys.stderr,
            )
            fail = True
        if e["dead_over_baseline"] > 0:
            print(
                f"\nFAIL: {e['dead_count']} dead keys exceeds baseline "
                f"({e['baseline']}). Either remove the new dead key or "
                "wire it to a `data-i18n` attribute, `t(...)` call, or supported JS key field.",
                file=sys.stderr,
            )
            fail = True
        if fail:
            return 1
        print("\ni18n drift within baseline.")
    return 0


def cmd_json() -> int:
    e = evaluate()
    json.dump(e, sys.stdout, indent=2)
    sys.stdout.write("\n")
    if e["missing_count"] > 0 or e["dead_over_baseline"] > 0:
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="CEP locale drift gate.")
    parser.add_argument("--check", action="store_true", help="Exit 1 on missing keys or dead-key growth.")
    parser.add_argument("--json", action="store_true", help="JSON for CI.")
    args = parser.parse_args()
    if args.json:
        sys.exit(cmd_json())
    sys.exit(cmd_report(args.check))


if __name__ == "__main__":
    main()
