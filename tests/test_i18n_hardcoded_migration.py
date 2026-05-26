"""
Regression test for the E6 hardcoded-English migration pass.

`RESEARCH_FEATURE_PLAN_2026-05-25.md` E6 identified four user-visible
``showToast(...)`` strings that bypassed the i18n layer:

  - main.js:2100  "Server reconnected"
  - main.js:2125  "OpenCut v{X} available — visit GitHub to update"
  - main.js:2264  "Refreshing Premiere project media…"
  - main.js:2554  "Tip: Save your project before processing"

The follow-on commit replaced each with ``t("key", fallback)``. This
test asserts:

  * the four ``toast.*`` keys are present in ``en.json``;
  * the four sites in ``main.js`` route through ``t("toast.…", …)``;
  * the bare-English form is *no longer* used as the toast message
    argument (catches an accidental revert).
"""
from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EN_JSON = REPO_ROOT / "extension" / "com.opencut.panel" / "client" / "locales" / "en.json"
MAIN_JS = REPO_ROOT / "extension" / "com.opencut.panel" / "client" / "main.js"


MIGRATED_KEYS = (
    "toast.server_reconnected",
    "toast.update_available",
    "toast.refreshing_media",
    "toast.save_first",
)


# Each tuple: (i18n_key, regex of expected `t("key", ...)` invocation,
# regex of the now-banned bare-English `showToast("...")` form that the
# original code shipped).
EXPECTED_CALLS = (
    (
        "toast.server_reconnected",
        re.compile(r't\(\s*"toast\.server_reconnected"'),
        re.compile(r'showToast\(\s*"Server reconnected"'),
    ),
    (
        "toast.refreshing_media",
        re.compile(r't\(\s*"toast\.refreshing_media"'),
        re.compile(r'showToast\(\s*"Refreshing Premiere project media'),
    ),
    (
        "toast.save_first",
        re.compile(r't\(\s*"toast\.save_first"'),
        re.compile(r'showToast\(\s*"Tip: Save your project before processing"'),
    ),
    (
        # The update-available site interpolates {version}; we only assert
        # the i18n lookup happens — the bare form check is loose because
        # the fallback path still uses the English literal as a safety
        # net when the locale lacks the templated form.
        "toast.update_available",
        re.compile(r't\(\s*"toast\.update_available"'),
        None,
    ),
)


class TestI18nHardcodedMigration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.en = json.loads(EN_JSON.read_text(encoding="utf-8"))
        cls.js = MAIN_JS.read_text(encoding="utf-8")

    def test_all_migrated_keys_in_en_json(self):
        for key in MIGRATED_KEYS:
            with self.subTest(key=key):
                self.assertIn(key, self.en,
                              f"en.json missing migrated key {key!r}")
                self.assertTrue(self.en[key].strip(),
                                f"en.json {key!r} has empty value")

    def test_main_js_routes_through_t_for_each_key(self):
        for key, present_re, banned_re in EXPECTED_CALLS:
            with self.subTest(key=key):
                self.assertRegex(
                    self.js, present_re,
                    f"main.js does not invoke t({key!r}, …) — migration may have reverted",
                )
                if banned_re is None:
                    continue
                # Bare-English form must not appear as a showToast argument.
                self.assertNotRegex(
                    self.js, banned_re,
                    f"main.js still contains bare-English showToast for {key!r}",
                )


if __name__ == "__main__":
    unittest.main()
