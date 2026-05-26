"""
CEP panel a11y guard tests (RESEARCH_FEATURE_PLAN_2026-05-25 E7).

The original research file flagged "toasts lack aria-live; modal focus
trap incomplete" — investigation showed both are *actually* implemented
in main.js (showToast at ~line 9902 sets role + aria-live + aria-atomic;
initOverlayFocusManagement at ~line 808 handles Tab + Escape + return-
focus). What was missing was a guard against silent regression.

These tests assert the a11y attributes the panel relies on are present
in main.js. Any refactor that drops them will fail this gate.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MAIN_JS = REPO_ROOT / "extension" / "com.opencut.panel" / "client" / "main.js"


def _extract_function_body(js: str, fn_name: str) -> str:
    """Return the body of ``function fn_name(...) {...}`` by brace-counting.

    Returns ``""`` if the function is not found.
    """
    m = re.search(rf"function\s+{re.escape(fn_name)}\s*\([^)]*\)\s*\{{", js)
    if not m:
        return ""
    i = m.end()  # right after the opening brace
    depth = 1
    start = i
    while i < len(js) and depth > 0:
        ch = js[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return js[start:i]
        i += 1
    return ""


class TestPanelA11yInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.js = MAIN_JS.read_text(encoding="utf-8")

    def test_showtoast_sets_aria_live(self):
        body = _extract_function_body(self.js, "showToast")
        self.assertTrue(body, "showToast function not found in main.js")
        self.assertIn('"aria-live"', body, "showToast must set aria-live")
        self.assertIn('"role"', body, "showToast must set role")

    def test_showtoast_sets_aria_atomic(self):
        body = _extract_function_body(self.js, "showToast")
        self.assertIn(
            '"aria-atomic"', body,
            "showToast should set aria-atomic so screen readers announce the "
            "whole message even when re-rendered",
        )

    def test_initoverlayfocusmanagement_handles_escape_and_tab(self):
        body = _extract_function_body(self.js, "initOverlayFocusManagement")
        self.assertTrue(body, "initOverlayFocusManagement function not found")
        self.assertIn('e.key === "Escape"', body, "Overlay handler must trap Escape")
        self.assertIn('e.key !== "Tab"', body, "Overlay handler must trap Tab")

    def test_overlay_activation_records_return_focus(self):
        body = _extract_function_body(self.js, "activateOverlay")
        self.assertTrue(body, "activateOverlay function not found")
        self.assertIn(
            "document.activeElement", body,
            "activateOverlay must record document.activeElement as the "
            "return-focus target for deactivateOverlay",
        )

    def test_syncoverlay_uses_inert_and_aria_hidden(self):
        body = _extract_function_body(self.js, "syncOverlayBackgroundState")
        self.assertTrue(body, "syncOverlayBackgroundState function not found")
        self.assertIn(
            '"aria-hidden"', body,
            "Overlay background must set aria-hidden on app shell",
        )
        self.assertIn(
            "inert", body,
            "Overlay background must set inert on app shell",
        )


if __name__ == "__main__":
    unittest.main()
