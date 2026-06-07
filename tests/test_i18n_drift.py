"""
Tests for scripts/i18n_lint.py (RESEARCH_FEATURE_PLAN_2026-05-25 Q6/E6).

The CEP locale ledger ``client/locales/en.json`` historically drifted
~142 keys away from any consumer in ``index.html`` / ``main.js``. The
linter exposes both directions of drift:
  - dead keys: in en.json but never referenced
  - missing keys: referenced but not in en.json

This test:
  - confirms the live tree's missing-key count is zero (hard gate);
  - confirms the dead-key baseline and live dead-key count stay at zero;
  - exercises the linter's HTML, JS call, and JS key-field extraction
    regexes against synthetic inputs.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "i18n_lint.py"
LOCALE = REPO_ROOT / "extension" / "com.opencut.panel" / "client" / "locales" / "en.json"


def _load_module():
    name = "scripts_i18n_lint"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _duplicate_json_keys(raw: str) -> list[str]:
    duplicates: list[str] = []

    def collect_pairs(pairs):
        counts = Counter(key for key, _value in pairs)
        duplicates.extend(key for key, count in counts.items() if count > 1)
        return dict(pairs)

    json.loads(raw, object_pairs_hook=collect_pairs)
    return sorted(set(duplicates))


class TestI18nDrift(unittest.TestCase):
    def setUp(self):
        self.mod = _load_module()

    def test_no_missing_keys_in_live_tree(self):
        e = self.mod.evaluate()
        self.assertEqual(
            e["missing_keys"], [],
            f"Missing i18n keys (consumed but not in en.json): {e['missing_keys']}",
        )

    def test_dead_key_baseline_is_zero(self):
        e = self.mod.evaluate()
        self.assertEqual(
            e["baseline"],
            0,
            "DEAD_KEY_BASELINE should remain zero now that the live locale file has no dead keys.",
        )

    def test_no_dead_keys_in_live_tree(self):
        e = self.mod.evaluate()
        self.assertEqual(
            e["dead_keys"],
            [],
            f"Dead i18n keys (in en.json but never consumed): {e['dead_keys']}",
        )

    def test_live_locale_file_has_unique_keys(self):
        duplicate_keys = _duplicate_json_keys(LOCALE.read_text(encoding="utf-8"))
        self.assertEqual(
            duplicate_keys,
            [],
            f"Duplicate locale keys in en.json: {duplicate_keys}",
        )

    def test_html_regex_captures_all_data_i18n_variants(self):
        html = (
            '<button data-i18n="audio.duck">Duck</button>'
            '<input data-i18n-placeholder="audio.placeholder">'
            '<div data-i18n-title="video.title_tip"></div>'
            '<optgroup data-i18n-label="video.option_group"></optgroup>'
            '<img data-i18n-alt="preview.frame_alt">'
            '<a data-i18n-aria-label="video.aria_close"></a>'
        )
        found = set(self.mod.HTML_I18N_RE.findall(html))
        self.assertEqual(found, {
            "audio.duck",
            "audio.placeholder",
            "video.title_tip",
            "video.option_group",
            "preview.frame_alt",
            "video.aria_close",
        })

    def test_js_regex_only_captures_dotted_i18n_keys(self):
        js = """
        var x = t("audio.duck");
        var y = t('video.title_tip');
        var z = t("not_an_i18n_key");  // single segment — not matched
        var w = t(variable);             // dynamic — not matched
        var u = subscribe(\"chan.foo\"); // wrong function name — not matched
        """
        found = set(self.mod.JS_I18N_RE.findall(js))
        # Single-segment keys aren't matched; dotted keys are.
        self.assertIn("audio.duck", found)
        self.assertIn("video.title_tip", found)
        self.assertNotIn("not_an_i18n_key", found)

    def test_js_key_field_regex_captures_supported_locale_metadata(self):
        js = """
        var shortcut = { labelKey: "shortcuts.command_palette" };
        var quoted = { "placeholderKey": "forms.search_placeholder" };
        var title = { titleKey: "workspace.choose_clip_title" };
        var external = { apiKey: "provider.secret" };
        var loose = { key: "not.locale.metadata" };
        """
        found = self.mod._scan_js_metadata_consumers(js)
        self.assertEqual(found, {
            "shortcuts.command_palette",
            "forms.search_placeholder",
            "workspace.choose_clip_title",
        })
        self.assertNotIn("provider.secret", found)
        self.assertNotIn("not.locale.metadata", found)

    def test_js_consumer_union_includes_key_field_metadata(self):
        js = """
        var direct = t("audio.duck");
        var shortcut = { labelKey: "shortcuts.command_palette" };
        """
        self.assertEqual(self.mod._scan_js_consumers_from_source(js), {
            "audio.duck",
            "shortcuts.command_palette",
        })


if __name__ == "__main__":
    unittest.main()
