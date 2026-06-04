"""
Tests for scripts/i18n_lint.py (RESEARCH_FEATURE_PLAN_2026-05-25 Q6/E6).

The CEP locale ledger ``client/locales/en.json`` historically drifted
~142 keys away from any consumer in ``index.html`` / ``main.js``. The
linter exposes both directions of drift:
  - dead keys: in en.json but never referenced
  - missing keys: referenced but not in en.json

This test:
  - confirms the live tree's missing-key count is zero (hard gate);
  - confirms the live tree's dead-key count stays at or below
    ``DEAD_KEY_BASELINE`` (soft gate; reduce over time);
  - exercises the linter's HTML and JS extraction regexes against
    synthetic inputs.
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "i18n_lint.py"


def _load_module():
    name = "scripts_i18n_lint"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestI18nDrift(unittest.TestCase):
    def setUp(self):
        self.mod = _load_module()

    def test_no_missing_keys_in_live_tree(self):
        e = self.mod.evaluate()
        self.assertEqual(
            e["missing_keys"], [],
            f"Missing i18n keys (consumed but not in en.json): {e['missing_keys']}",
        )

    def test_dead_keys_under_baseline_in_live_tree(self):
        e = self.mod.evaluate()
        self.assertLessEqual(
            e["dead_count"], e["baseline"],
            (
                f"Dead-key count {e['dead_count']} exceeds baseline "
                f"{e['baseline']}. Either reduce dead keys or update "
                "DEAD_KEY_BASELINE in scripts/i18n_lint.py."
            ),
        )

    def test_html_regex_captures_all_data_i18n_variants(self):
        html = (
            '<button data-i18n="audio.duck">Duck</button>'
            '<input data-i18n-placeholder="audio.placeholder">'
            '<div data-i18n-title="video.title_tip"></div>'
            '<optgroup data-i18n-label="video.option_group"></optgroup>'
            '<a data-i18n-aria-label="video.aria_close"></a>'
        )
        found = set(self.mod.HTML_I18N_RE.findall(html))
        self.assertEqual(found, {
            "audio.duck",
            "audio.placeholder",
            "video.title_tip",
            "video.option_group",
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


if __name__ == "__main__":
    unittest.main()
