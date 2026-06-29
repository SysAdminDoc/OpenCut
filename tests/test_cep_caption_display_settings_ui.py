"""CEP F236 caption display-settings parity tests."""

from __future__ import annotations

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CEP_HTML = REPO_ROOT / "extension" / "com.opencut.panel" / "client" / "index.html"
CEP_JS = REPO_ROOT / "extension" / "com.opencut.panel" / "client" / "main.js"
CEP_CSS = REPO_ROOT / "extension" / "com.opencut.panel" / "client" / "style.css"

REQUIRED_SELECT_IDS = (
    "capDispFont",
    "capDispSize",
    "capDispTextColor",
    "capDispTextOpacity",
    "capDispBgColor",
    "capDispBgOpacity",
    "capDispEdge",
)


class TestCepCaptionDisplaySettingsHtml(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = CEP_HTML.read_text(encoding="utf-8", errors="replace")

    def test_card_present_in_captions_panel(self):
        self.assertIn('id="panel-captions"', self.html)
        self.assertIn('id="captionDisplaySettingsCard"', self.html)
        self.assertIn('id="captionDisplaySettingsTitle"', self.html)

    def test_all_selects_declared(self):
        for select_id in REQUIRED_SELECT_IDS:
            with self.subTest(select=select_id):
                self.assertIn(f'id="{select_id}"', self.html)

    def test_buttons_status_and_preview_declared(self):
        for element_id in (
            "capDispPreviewBtn",
            "capDispResetBtn",
            "capDispStatus",
            "capDispPreviewBox",
            "capDispPreviewArea",
            "capDispPreviewSample",
        ):
            with self.subTest(element=element_id):
                self.assertIn(f'id="{element_id}"', self.html)

    def test_fcc_compliance_hint_present(self):
        self.assertIn("FCC", self.html)
        self.assertIn("2026-08-17", self.html)
        self.assertIn("https://www.ecfr.gov/current/title-47/section-79.103", self.html)


class TestCepCaptionDisplaySettingsJs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.js = CEP_JS.read_text(encoding="utf-8", errors="replace")

    def test_init_function_defined_and_called(self):
        self.assertIn("function initCaptionDisplaySettingsCard()", self.js)
        self.assertIn("initCaptionDisplaySettingsCard();", self.js)

    def test_backend_endpoints_referenced(self):
        self.assertIn("/captions/display-settings/tokens", self.js)
        self.assertIn("/captions/display-settings/preview", self.js)

    def test_required_selects_are_cached_and_configured(self):
        for select_id in REQUIRED_SELECT_IDS:
            with self.subTest(select=select_id):
                self.assertIn(f'el.{select_id} = $("{select_id}")', self.js)
                self.assertIn(f'id: "{select_id}"', self.js)

    def test_font_options_surface_resolution_status(self):
        self.assertIn("font_resolution", self.js)
        self.assertIn('source !== "preferred_file" ? "fallback" : "resolved"', self.js)
        self.assertIn("option.title = opt.font_resolution.warning", self.js)

    def test_preview_and_reset_handlers_wired(self):
        self.assertIn('addEventListener("click", refreshCaptionDisplayPreview)', self.js)
        self.assertIn('addEventListener("click", resetCaptionDisplayDefaults)', self.js)


class TestCepCaptionDisplaySettingsCss(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.css = CEP_CSS.read_text(encoding="utf-8", errors="replace")

    def test_braces_balanced(self):
        self.assertEqual(self.css.count("{"), self.css.count("}"))

    def test_preview_selectors_present(self):
        for selector in (
            ".caption-display-card",
            ".caption-display-grid",
            ".caption-display-preview-area",
            ".caption-display-preview-sample",
            "#fccComplianceNotice",
        ):
            with self.subTest(selector=selector):
                self.assertIn(selector, self.css)


if __name__ == "__main__":
    unittest.main()
