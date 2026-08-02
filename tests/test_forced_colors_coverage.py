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
UXP_DIR = REPO_ROOT / "extension" / "com.opencut.uxp"
RENDERED_SPEC = (
    REPO_ROOT / "extension" / "com.opencut.panel" / "tests" / "rendered"
    / "panel-regression.spec.mjs"
)

FORCED_COLORS_RE = re.compile(r"@media\s*\(\s*forced-colors:\s*active\s*\)\s*\{")

#: System colour keywords are the only values that survive the mode.
SYSTEM_COLOURS = ("CanvasText", "Canvas", "Highlight", "GrayText")
CLASS_SELECTOR_RE = re.compile(r"(?<![\w-])\.([A-Za-z_][\w-]*)")
CLASS_ASSIGNMENT_RE = re.compile(
    r"\bclass(?:Name)?\s*=\s*([\"'`])(?P<value>.*?)(?:\1)",
    re.DOTALL,
)
CLASS_LIST_RE = re.compile(
    r"\bclassList\.(?:add|remove|toggle)\s*\(\s*([\"'`])(?P<value>[^\"'`]+)\1"
)


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


def _forced_color_class_selectors(block: str) -> set[str]:
    """Return class names referenced by selectors in a forced-colors block."""
    block = re.sub(r"/\*.*?\*/", "", block, flags=re.DOTALL)
    selectors = set()
    for match in re.finditer(r"([^{}]+)\{", block):
        selectors.update(CLASS_SELECTOR_RE.findall(match.group(1)))
    return selectors


def _panel_markup_classes(panel: str) -> set[str]:
    """Collect static classes from panel HTML and JavaScript markup builders."""
    if panel == "cep":
        sources = [*CEP_DIR.glob("*.html"), *CEP_DIR.glob("*.js")]
    else:
        sources = [UXP_DIR / "index.html", *UXP_DIR.glob("*.js")]

    classes = set()
    for path in sources:
        markup = path.read_text(encoding="utf-8", errors="replace")
        values = [
            *(
                match.group("value")
                for match in CLASS_ASSIGNMENT_RE.finditer(markup)
            ),
            *(
                match.group("value")
                for match in CLASS_LIST_RE.finditer(markup)
            ),
        ]
        for value in values:
            classes.update(re.findall(r"[A-Za-z_][\w-]*", value))
    return classes


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

    def test_every_forced_color_class_selector_is_live(self):
        """A palette rule must target markup that the panel can actually render."""
        for name, block in self.panels.items():
            with self.subTest(panel=name):
                selectors = _forced_color_class_selectors(block)
                live_classes = _panel_markup_classes(name)
                missing = sorted(selectors - live_classes)
                self.assertEqual(
                    missing,
                    [],
                    f"{name} forced-colors selectors have no live markup: {missing}",
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
