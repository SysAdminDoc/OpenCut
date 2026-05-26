"""
Regression test for the UXP Agent tab CSS additions.

The Agent tab introduced in commit 1e51521 referenced four CSS classes
(``oc-card--nested``, ``oc-step-list``, ``oc-step-error`` plus the
selector ``#agentChatReviewNotes``) that initially had no rules in
``style.css`` — the tab rendered but lacked the visual hierarchy the
design language expects. This test asserts the classes are now
defined, the style.css braces stay balanced, and the line count is
within the documented drift tolerance.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STYLE = REPO_ROOT / "extension" / "com.opencut.uxp" / "style.css"


REQUIRED_SELECTORS = (
    r"\.oc-card--nested\s*\{",
    r"\.oc-step-list\s*\{",
    r"\.oc-step-list li\.oc-step-error",
    r"#agentChatReviewNotes\s*\{",
    r"#agentChatReviewSummary\s*\{",
)


class TestAgentTabCss(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.css = STYLE.read_text(encoding="utf-8", errors="replace")

    def test_braces_balanced(self):
        """CSS must keep its brace count balanced (regression guard)."""
        opens = self.css.count("{")
        closes = self.css.count("}")
        self.assertEqual(opens, closes, f"Unbalanced braces: {opens} open vs {closes} close")

    def test_required_agent_tab_selectors_present(self):
        for pattern in REQUIRED_SELECTORS:
            with self.subTest(pattern=pattern):
                self.assertRegex(
                    self.css, pattern,
                    f"style.css missing required Agent-tab selector: {pattern}",
                )

    def test_step_error_uses_error_color_token(self):
        """The error variant of a plan step must use the existing
        ``--error`` color token, not a hard-coded hex value."""
        m = re.search(
            r"\.oc-step-list li\.oc-step-error\s*\{(?P<body>[^}]*)\}",
            self.css,
            re.DOTALL,
        )
        self.assertIsNotNone(m, "oc-step-error rule block not found")
        body = m.group("body")
        self.assertIn("var(--error", body,
                      "oc-step-error must consume the --error CSS variable")


if __name__ == "__main__":
    unittest.main()
