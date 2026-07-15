"""Static UI contract tests for plugin trust dashboards in Settings."""

from __future__ import annotations

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CEP_ROOT = REPO_ROOT / "extension" / "com.opencut.panel" / "client"
UXP_ROOT = REPO_ROOT / "extension" / "com.opencut.uxp"


class TestCepPluginTrustSettingsUi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (CEP_ROOT / "index.html").read_text(encoding="utf-8", errors="replace")
        cls.js = (CEP_ROOT / "main.js").read_text(encoding="utf-8", errors="replace")
        cls.css = (CEP_ROOT / "style.css").read_text(encoding="utf-8", errors="replace")

    def test_settings_card_markup_present(self):
        for element_id in (
            "pluginTrustCard",
            "pluginTrustList",
            "refreshPluginTrustBtn",
            "pluginTrustLoadedValue",
            "pluginTrustFailedValue",
            "pluginTrustQuarantineValue",
            "pluginTrustMarketplaceValue",
            "pluginTrustStatusLine",
        ):
            with self.subTest(element=element_id):
                self.assertIn(f'id="{element_id}"', self.html)

    def test_js_uses_backend_trust_contract(self):
        for needle in (
            'api("GET", "/plugins/trust"',
            "function loadPluginTrustDashboard()",
            "renderPluginTrustDashboard(data)",
            "capability_badges",
            "quarantine.entries",
            "marketplace.plugins",
            "confirm_name and confirm_token",
            "/plugins/quarantine/delete",
            "plugin-install-approval-checkbox",
            'api("POST", "/plugins/marketplace/install"',
            "approve_publisher_fingerprint",
            "function bindPluginWorkerActions(actions)",
            "/plugins/workers/restart",
            "not an OS security sandbox",
        ):
            with self.subTest(needle=needle):
                self.assertIn(needle, self.js)

    def test_language_capability_dropdown_normalizes_route_payload(self):
        self.assertIn("function normalizeLanguageOptions(languages)", self.js)
        self.assertIn("Array.isArray(languages)", self.js)
        self.assertIn("languageOptionLabel(langMap[a], a).localeCompare", self.js)
        self.assertNotIn("langMap[a].localeCompare(langMap[b])", self.js)

    def test_css_selectors_present(self):
        for selector in (
            ".plugin-trust-list",
            ".plugin-trust-row",
            ".plugin-capability-badge",
            ".plugin-action-contract",
            ".plugin-trust-subhead",
            ".plugin-install-approval",
        ):
            with self.subTest(selector=selector):
                self.assertIn(selector, self.css)


class TestUxpPluginTrustSettingsUi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (UXP_ROOT / "index.html").read_text(encoding="utf-8", errors="replace")
        cls.js = (UXP_ROOT / "main.js").read_text(encoding="utf-8", errors="replace")
        cls.css = (UXP_ROOT / "style.css").read_text(encoding="utf-8", errors="replace")

    def test_settings_card_markup_present(self):
        for element_id in (
            "uxpPluginTrustCard",
            "uxpPluginTrustGrid",
            "uxpRefreshPluginTrustBtn",
            "settingsPluginLoadedValue",
            "settingsPluginFailedValue",
            "settingsPluginQuarantineValue",
            "settingsPluginMarketplaceValue",
            "settingsPluginTrustStatus",
        ):
            with self.subTest(element=element_id):
                self.assertIn(f'id="{element_id}"', self.html)

    def test_js_uses_backend_trust_contract(self):
        for needle in (
            'BackendClient.get("/plugins/trust")',
            "async function uxpLoadPluginTrust()",
            "renderPluginTrustDashboard(response.data)",
            "capability_badges",
            "quarantine?.entries",
            "marketplace?.plugins",
            "confirm_name and confirm_token",
            "/plugins/quarantine/delete",
            "oc-plugin-install-approval-checkbox",
            'BackendClient.post("/plugins/marketplace/install"',
            "approve_publisher_fingerprint",
            "function bindPluginWorkerActions(actions)",
            "/plugins/workers/restart",
            "not an OS security sandbox",
        ):
            with self.subTest(needle=needle):
                self.assertIn(needle, self.js)

    def test_css_selectors_present(self):
        for selector in (
            ".oc-plugin-trust-row",
            ".oc-plugin-capability",
            ".oc-plugin-action-contract",
            ".oc-plugin-trust-subhead",
            ".oc-plugin-install-approval",
        ):
            with self.subTest(selector=selector):
                self.assertIn(selector, self.css)


if __name__ == "__main__":
    unittest.main()
