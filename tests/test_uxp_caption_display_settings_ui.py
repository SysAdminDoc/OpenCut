"""
Tests for the F236 FCC caption display-settings UI surfacing.

Backend contract is already pinned by ``tests/test_caption_display_*``
(repository-side, in place since Pass 17). This test pins the UI side:

  * the new Captions-tab card exists in ``com.opencut.uxp/index.html``
    with the seven select dropdowns the JS module expects;
  * the FCC compliance-date date string is present in the hint;
  * ``initCaptionDisplaySettingsCard()`` is defined in ``main.js`` and
    is called from ``initApp()``;
  * the CSS gained the new preview-area styles and braces stay balanced.

The actual interactive behaviour is tested via the backend routes
(``/captions/display-settings/tokens`` + ``/.../preview``) which the
existing test suite already covers.
"""
from __future__ import annotations

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
UXP_HTML = REPO_ROOT / "extension" / "com.opencut.uxp" / "index.html"
UXP_JS = REPO_ROOT / "extension" / "com.opencut.uxp" / "main.js"
UXP_CSS = REPO_ROOT / "extension" / "com.opencut.uxp" / "style.css"


REQUIRED_SELECT_IDS = (
    "capDispFont",
    "capDispSize",
    "capDispTextColor",
    "capDispTextOpacity",
    "capDispBgColor",
    "capDispBgOpacity",
    "capDispEdge",
)


class TestCaptionDisplaySettingsHtml(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = UXP_HTML.read_text(encoding="utf-8", errors="replace")

    def test_card_present_in_captions_tab(self):
        # The card sits inside #tab-captions and has the F236 id.
        self.assertIn('id="captionDisplaySettingsCard"', self.html)

    def test_all_seven_selects_declared(self):
        for sel_id in REQUIRED_SELECT_IDS:
            with self.subTest(select=sel_id):
                self.assertIn(f'id="{sel_id}"', self.html,
                              f"Captions display-settings card missing select id={sel_id!r}")

    def test_preview_and_reset_buttons_declared(self):
        self.assertIn('id="capDispPreviewBtn"', self.html)
        self.assertIn('id="capDispResetBtn"', self.html)

    def test_status_line_and_preview_box_declared(self):
        self.assertIn('id="capDispStatus"', self.html)
        self.assertIn('id="capDispPreviewBox"', self.html)
        self.assertIn('id="capDispPreviewArea"', self.html)
        self.assertIn('id="capDispPreviewSample"', self.html)

    def test_fcc_compliance_date_in_hint(self):
        # The compliance date must be discoverable. Hard-coded string
        # match here; the JS module will overlay the date from the
        # backend's compliance_date field at runtime if it changes.
        self.assertIn("2026-08-17", self.html)
        self.assertIn("FCC", self.html)


class TestCaptionDisplaySettingsJs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.js = UXP_JS.read_text(encoding="utf-8", errors="replace")

    def test_init_function_defined(self):
        self.assertIn("function initCaptionDisplaySettingsCard()", self.js)

    def test_init_called_from_initapp(self):
        self.assertIn("initCaptionDisplaySettingsCard();", self.js)

    def test_endpoints_referenced(self):
        self.assertIn("/captions/display-settings/tokens", self.js)
        self.assertIn("/captions/display-settings/preview", self.js)

    def test_preview_button_handler_wired(self):
        # Look for the addEventListener pair on capDispPreviewBtn /
        # capDispResetBtn within 200 chars.
        import re
        self.assertRegex(
            self.js,
            r'"capDispPreviewBtn"\)\?\.addEventListener',
            "Preview button is not wired to refreshPreview",
        )
        self.assertRegex(
            self.js,
            r'"capDispResetBtn"\)\?\.addEventListener',
            "Reset button is not wired to resetDefaults",
        )


class TestCaptionDisplaySettingsCss(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.css = UXP_CSS.read_text(encoding="utf-8", errors="replace")

    def test_braces_balanced(self):
        opens = cls_open = self.css.count("{")
        closes = self.css.count("}")
        self.assertEqual(opens, closes, f"Unbalanced braces in style.css: {opens} vs {closes}")

    def test_preview_area_selectors_present(self):
        self.assertIn(".oc-caption-preview-area", self.css)
        self.assertIn(".oc-caption-preview-sample", self.css)
        self.assertIn("#fccComplianceNotice", self.css)


class TestBackendContractStable(unittest.TestCase):
    """The UI assumes the token-schema shape; pin it so a backend
    regression breaks the UI test (early-warning gate)."""

    @classmethod
    def setUpClass(cls):
        from opencut.server import create_app
        cls.app = create_app()
        cls.client = cls.app.test_client()

    def test_tokens_endpoint_returns_required_shape(self):
        resp = self.client.get("/captions/display-settings/tokens")
        self.assertEqual(resp.status_code, 200)
        payload = resp.get_json()
        # Compliance date is the user-facing anchor.
        self.assertIn("compliance_date", payload)
        # Defaults dict must include every setting the UI form binds to.
        for key in (
            "font", "size", "text_color", "text_opacity",
            "background_color", "background_opacity", "edge_style",
        ):
            with self.subTest(key=key):
                self.assertIn(key, payload.get("defaults", {}),
                              f"Token schema 'defaults' missing key={key!r}")
        # Token categories must include the 5 the UI populates dropdowns from.
        for cat in ("font", "size", "color", "opacity", "edge_style"):
            with self.subTest(category=cat):
                self.assertIn(cat, payload.get("tokens", {}),
                              f"Token schema 'tokens' missing category={cat!r}")


if __name__ == "__main__":
    unittest.main()
