"""Forced-colors coverage must exist in both panels and stay there.

Windows High Contrast replaces every author colour with a small system
palette. Anything a panel expressed *only* as a background tint — the active
tab, a disabled control, a status severity, a progress fill — collapses into
the same surface and stops being distinguishable.

The behavioural proof lives in the Playwright suite
(`panel-regression.spec.mjs`, the `forced-colors` describes), which needs a
browser. This module is the always-on guard: it pins that the rules exist, that
they cover each state the acceptance calls out, and that severity is not
carried by hue alone.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CEP_DIR = REPO_ROOT / "extension" / "com.opencut.panel" / "client"
UXP_CSS = REPO_ROOT / "extension" / "com.opencut.uxp" / "style.css"
RENDERED_SPEC = (
    REPO_ROOT / "extension" / "com.opencut.panel" / "tests" / "rendered"
    / "panel-regression.spec.mjs"
)

FORCED_COLORS_RE = re.compile(r"@media\s*\(\s*forced-colors:\s*active\s*\)\s*\{")

#: System colour keywords are the only values that survive the mode.
SYSTEM_COLOURS = ("CanvasText", "Canvas", "Highlight", "GrayText")


def _cep_css() -> str:
    """The CEP panel loads four stylesheets; the mode may be handled in any."""
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in sorted(CEP_DIR.glob("*.css"))
    )


def _forced_colors_blocks(css: str) -> str:
    """Concatenate the bodies of every forced-colors block."""
    bodies = []
    for match in FORCED_COLORS_RE.finditer(css):
        depth = 1
        index = match.end()
        while index < len(css) and depth:
            if css[index] == "{":
                depth += 1
            elif css[index] == "}":
                depth -= 1
            index += 1
        bodies.append(css[match.end():index])
    return "\n".join(bodies)


class TestForcedColorsRulesExist(unittest.TestCase):
    def setUp(self):
        self.panels = {
            "cep": _forced_colors_blocks(_cep_css()),
            "uxp": _forced_colors_blocks(UXP_CSS.read_text(encoding="utf-8")),
        }

    def test_both_panels_declare_forced_colors_rules(self):
        for name, block in self.panels.items():
            with self.subTest(panel=name):
                self.assertTrue(block.strip(), f"{name} has no forced-colors rules")

    def test_rules_use_system_colour_keywords(self):
        for name, block in self.panels.items():
            with self.subTest(panel=name):
                used = [colour for colour in SYSTEM_COLOURS if colour in block]
                self.assertGreaterEqual(
                    len(used), 3,
                    f"{name} forced-colors rules barely use the system palette: {used}",
                )

    def test_focus_is_an_outline_not_a_background(self):
        for name, block in self.panels.items():
            with self.subTest(panel=name):
                self.assertIn(":focus-visible", block)
                self.assertIn("outline", block)
                self.assertIn("Highlight", block)

    def test_disabled_state_uses_graytext(self):
        for name, block in self.panels.items():
            with self.subTest(panel=name):
                self.assertIn(":disabled", block)
                self.assertIn("GrayText", block)

    def test_selected_navigation_is_restated(self):
        selectors = {"cep": ".nav-tab.active", "uxp": ".oc-tab.active"}
        for name, block in self.panels.items():
            with self.subTest(panel=name):
                self.assertIn(selectors[name], block)

    def test_severity_is_not_carried_by_colour_alone(self):
        """Error and success need a glyph once their hue is taken away."""
        for name, block in self.panels.items():
            with self.subTest(panel=name):
                self.assertIn("::before", block)
                self.assertIn("content:", block)
                self.assertIn('"! "', block)

    def test_surfaces_gain_an_edge(self):
        for name, block in self.panels.items():
            with self.subTest(panel=name):
                self.assertIn("border: 1px solid CanvasText", block)

    def test_forced_color_adjust_is_only_used_where_a_fill_is_meaningful(self):
        """`forced-color-adjust: none` opts out of the palette; keep it narrow."""
        for name, block in self.panels.items():
            with self.subTest(panel=name):
                opt_outs = block.count("forced-color-adjust: none")
                self.assertLessEqual(
                    opt_outs, 4,
                    f"{name} opts out of the forced palette {opt_outs} times",
                )


class TestRenderedCoverage(unittest.TestCase):
    """The browser-backed proof has to stay wired up."""

    @classmethod
    def setUpClass(cls):
        cls.spec = RENDERED_SPEC.read_text(encoding="utf-8")

    def test_spec_emulates_forced_colors(self):
        self.assertIn('forcedColors: "active"', self.spec)

    def test_spec_covers_both_surfaces(self):
        self.assertIn("forced-colors", self.spec)
        self.assertIn("keeps the shell navigable", self.spec)

    def test_spec_asserts_focus_and_selection(self):
        self.assertIn("keeps the active tab distinguishable", self.spec)
        self.assertIn("shows a focus indicator that is not a tint", self.spec)


if __name__ == "__main__":
    unittest.main()
