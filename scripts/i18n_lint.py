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
  3. Walks the CEP runtime owners for ``t("…")`` / ``translate("…")`` calls
     and structured locale metadata fields such as ``labelKey: "…"``.
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
import html
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCALES = ROOT / "extension" / "com.opencut.panel" / "client" / "locales"
INDEX_HTML = ROOT / "extension" / "com.opencut.panel" / "client" / "index.html"
MAIN_JS = ROOT / "extension" / "com.opencut.panel" / "client" / "main.js"
BACKEND_CLIENT_JS = ROOT / "extension" / "com.opencut.panel" / "client" / "backend-client.js"
TRANSCRIPT_CORRECTION_JS = ROOT / "extension" / "com.opencut.panel" / "client" / "transcript-correction-controller.js"
GPU_SELECTION_JS = ROOT / "extension" / "com.opencut.panel" / "client" / "gpu-selection-controller.js"
HOST_WRITE_VERIFICATION_JS = ROOT / "extension" / "com.opencut.panel" / "client" / "host-write-verification.js"
CLIENT_DIR = ROOT / "extension" / "com.opencut.panel" / "client"
UXP_DIR = ROOT / "extension" / "com.opencut.uxp"

#: Both panels paint the inline `data-i18n` text before i18n init finishes and
#: keep it as the fallback when a key is missing, so it has to say the same
#: thing the locale does. (label, index.html, en.json)
FALLBACK_PANELS = (
    ("cep", INDEX_HTML, LOCALES / "en.json"),
    ("uxp", UXP_DIR / "index.html", UXP_DIR / "locales" / "en.json"),
)

# Vendored or non-runtime scripts that never consume locale keys.
_NON_RUNTIME_JS = frozenset({"CSInterface.js"})


def _runtime_js_sources() -> tuple[Path, ...]:
    """Every panel runtime script, discovered rather than enumerated.

    This list used to be hand-maintained, so extracting a controller out of
    main.js silently removed its keys from the scan and the gate then reported
    them as dead — which is exactly how `results-controller.js` and
    `update-controller.js` put 35 live keys on the dead list. Globbing keeps a
    new extraction covered on the commit that creates it.
    """
    return tuple(
        sorted(
            path
            for path in CLIENT_DIR.glob("*.js")
            if path.name not in _NON_RUNTIME_JS
        )
    )


RUNTIME_JS_SOURCES = _runtime_js_sources()

# The historic dead-key cleanup is complete. Keep the floor at zero so new
# unused locale keys fail the gate immediately.
DEAD_KEY_BASELINE = 0

# data-i18n="…" / data-i18n-title="…" / data-i18n-label="…" /
# data-i18n-alt="…" /
# data-i18n-placeholder="…" / data-i18n-aria-label="…"
HTML_I18N_RE = re.compile(
    r'data-i18n(?:-(?:title|label|alt|placeholder|aria-label))?="([^"]+)"'
)

# t("key") / translate("key") — key must look like an i18n key.
JS_I18N_RE = re.compile(
    r"""\b(?:t|translate)\s*\(\s*['"]([a-z][a-zA-Z0-9_]*(?:\.[a-zA-Z][a-zA-Z0-9_]*)+)['"]"""
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


def _read_runtime_js() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in RUNTIME_JS_SOURCES
        if path.exists()
    )


def _scan_js_call_consumers(source: str | None = None) -> set[str]:
    if source is None:
        source = _read_runtime_js()
    return set(JS_I18N_RE.findall(source))


def _scan_js_metadata_consumers(source: str | None = None) -> set[str]:
    if source is None:
        source = _read_runtime_js()
    keys = set(JS_KEY_FIELD_RE.findall(source))
    for one_key, many_key in JS_PLUGIN_COUNT_LABEL_RE.findall(source):
        keys.add(one_key)
        keys.add(many_key)
    return keys


def _scan_js_consumers_from_source(source: str) -> set[str]:
    return _scan_js_call_consumers(source) | _scan_js_metadata_consumers(source)


def _scan_js_consumers() -> set[str]:
    return _scan_js_consumers_from_source(_read_runtime_js())


# ----------------------------------------------------------------------
# data-i18n fallback drift
# ----------------------------------------------------------------------
# The dead/missing-key checks above prove a key is *used*. They say nothing
# about the English text sitting next to it in the HTML, which is what a user
# reads on first paint and whenever i18n init fails. Those copies had drifted
# far enough to rename features (a "Studio Workspace" heading where the locale
# said "Cut Pass"), so the fallback text is now pinned to the locale value.

_VOID_TAGS = frozenset({
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
})

#: data-i18n-<x> attribute -> the attribute it translates.
_FALLBACK_ATTRS = {
    "data-i18n-title": "title",
    "data-i18n-label": "label",
    "data-i18n-alt": "alt",
    "data-i18n-placeholder": "placeholder",
    "data-i18n-aria-label": "aria-label",
}

#: Both panels hand the translated string to a nested label span when there is
#: one, so that span's text is the fallback rather than the whole element's.
_LABEL_CLASSES = ("btn-label", "i18n-text")


class _FallbackParser(HTMLParser):
    """Collect the text each `data-i18n` element and attribute paints."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.records: list[dict] = []
        self._stack: list[dict] = []

    def handle_starttag(self, tag, attrs):
        attributes = {name: (value or "") for name, value in attrs}
        line = self.getpos()[0]
        for data_attr, target in _FALLBACK_ATTRS.items():
            key = attributes.get(data_attr)
            if key and target in attributes:
                self.records.append({
                    "key": key,
                    "attr": target,
                    "line": line,
                    "text": attributes[target],
                })
        if tag in _VOID_TAGS:
            return
        self._stack.append({
            "tag": tag,
            "key": attributes.get("data-i18n"),
            "line": line,
            "text": [],
            "label": None,
            "is_label": any(
                cls in attributes.get("class", "").split() for cls in _LABEL_CLASSES
            ),
        })

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)

    def handle_data(self, data):
        for frame in self._stack:
            frame["text"].append(data)

    def handle_endtag(self, tag):
        for index in range(len(self._stack) - 1, -1, -1):
            if self._stack[index]["tag"] != tag:
                continue
            frame = self._stack.pop(index)
            # Unclosed inner tags leave stale frames above this one.
            del self._stack[index:]
            text = "".join(frame["text"])
            if frame["is_label"]:
                for parent in self._stack:
                    if parent["label"] is None:
                        parent["label"] = text
            if frame["key"]:
                self.records.append({
                    "key": frame["key"],
                    "attr": None,
                    "line": frame["line"],
                    "text": frame["label"] if frame["label"] is not None else text,
                })
            return


def _collapse(value: str) -> str:
    """HTML collapses runs of whitespace, so the comparison has to as well."""
    return " ".join(html.unescape(value).split())


def fallback_drifts(html_source: str, locale: dict) -> list[dict]:
    """Every `data-i18n` fallback whose text disagrees with its locale value."""
    parser = _FallbackParser()
    parser.feed(html_source)
    parser.close()
    drifts = []
    for record in parser.records:
        expected = locale.get(record["key"])
        if not isinstance(expected, str):
            continue
        actual = _collapse(record["text"])
        # An empty element is filled entirely from the locale at runtime; there
        # is no first-paint copy to keep honest.
        if not actual or actual == _collapse(expected):
            continue
        drifts.append({
            "key": record["key"],
            "attr": record["attr"],
            "line": record["line"],
            "html": actual,
            "locale": expected,
        })
    return sorted(drifts, key=lambda drift: drift["line"])


def evaluate_fallbacks() -> list[dict]:
    """Fallback drift for both panels, newest-panel-last for stable output."""
    out = []
    for panel, html_path, locale_path in FALLBACK_PANELS:
        if not html_path.is_file() or not locale_path.is_file():
            continue
        locale = json.loads(locale_path.read_text(encoding="utf-8"))
        for drift in fallback_drifts(html_path.read_text(encoding="utf-8"), locale):
            out.append({"panel": panel, **drift})
    return out

def evaluate() -> dict:
    keys = _load_en_keys()
    html_consumers = _scan_html_consumers()
    js_source = _read_runtime_js()
    js_call_consumers = _scan_js_call_consumers(js_source)
    js_metadata_consumers = _scan_js_metadata_consumers(js_source)
    js_consumers = js_call_consumers | js_metadata_consumers
    consumers = html_consumers | js_consumers
    dead = sorted(keys - consumers)
    missing = sorted(consumers - keys)
    fallbacks = evaluate_fallbacks()
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
        "fallback_drift_count": len(fallbacks),
        "fallback_drifts": fallbacks,
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
    print(f"  fallback drift: {e['fallback_drift_count']}")
    if e["fallback_drifts"]:
        print("\n  Fallback text that disagrees with en.json:")
        for drift in e["fallback_drifts"][:20]:
            where = f"{drift['panel']} index.html:{drift['line']}"
            attr = f" [@{drift['attr']}]" if drift["attr"] else ""
            print(f"    {where} {drift['key']}{attr}")
            print(f"      html   : {drift['html']}")
            print(f"      locale : {drift['locale']}")
        if e["fallback_drift_count"] > 20:
            print(f"    ... +{e['fallback_drift_count'] - 20} more")
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
        if e["fallback_drift_count"] > 0:
            print(
                f"\nFAIL: {e['fallback_drift_count']} data-i18n fallback(s) no longer "
                "match en.json. The locale is the source of truth: copy the locale "
                "value into the HTML, or change both together.",
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
    if e["missing_count"] > 0 or e["dead_over_baseline"] > 0 or e["fallback_drift_count"] > 0:
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
