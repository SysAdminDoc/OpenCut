"""Static guardrails for the UXP panel i18n foundation."""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UXP_ROOT = REPO_ROOT / "extension" / "com.opencut.uxp"
UXP_HTML = UXP_ROOT / "index.html"
UXP_JS = UXP_ROOT / "main.js"
UXP_LOCALE = UXP_ROOT / "locales" / "en.json"

I18N_ATTRIBUTES = (
    "data-i18n",
    "data-i18n-title",
    "data-i18n-label",
    "data-i18n-alt",
    "data-i18n-placeholder",
    "data-i18n-aria-label",
)


def _html() -> str:
    return UXP_HTML.read_text(encoding="utf-8")


def _js() -> str:
    return UXP_JS.read_text(encoding="utf-8")


def _locale() -> dict[str, str]:
    return json.loads(UXP_LOCALE.read_text(encoding="utf-8"))


def _html_i18n_keys() -> set[str]:
    html = _html()
    keys: set[str] = set()
    for attribute in I18N_ATTRIBUTES:
        keys.update(re.findall(rf"\s{re.escape(attribute)}=\"([^\"]+)\"", html))
    return keys


def _js_i18n_keys() -> set[str]:
    js = _js()
    keys = set(re.findall(r'(?<![A-Za-z0-9_$])t\(\s*"([^"]+)"', js))
    keys.update(
        re.findall(
            r'(?<![A-Za-z0-9_$])setStatus\(\s*"([a-z0-9_.-]+\.[a-z0-9_.-]+)"',
            js,
        )
    )
    keys.update(
        re.findall(
            r"(?:titleKey|subtitleKey|kickerKey|textKey|actionLabelKey):\s*\"([^\"]+)\"",
            js,
        )
    )
    return keys


def test_uxp_locale_file_is_valid_and_local_to_panel():
    locale = _locale()

    assert UXP_LOCALE.exists()
    assert "UXP_LOCALE_PATH  = \"locales/en.json\";" in _js()
    assert locale["uxp.document_title"] == "OpenCut UXP"


def test_uxp_i18n_loader_supports_dom_text_and_attributes():
    js = _js()

    assert "function t(key, fallback)" in js
    assert "function applyI18nToDOM(root = document)" in js
    assert "async function loadLocale(lang = \"en\")" in js
    for data_attribute, dom_attribute in (
        ("data-i18n-title", "title"),
        ("data-i18n-label", "label"),
        ("data-i18n-alt", "alt"),
        ("data-i18n-placeholder", "placeholder"),
        ("data-i18n-aria-label", "aria-label"),
    ):
        assert f'["{data_attribute}", "{dom_attribute}"]' in js

    assert js.index("await loadLocale();") < js.index("bindEvents();")


def test_uxp_shell_i18n_attributes_are_present_and_covered():
    html_keys = _html_i18n_keys()
    locale = _locale()

    assert len(re.findall(r"\sdata-i18n(?:-[a-z-]+)?=", _html())) >= 220
    assert {
        "common.skip_to_main",
        "conn.backend_status",
        "nav.feature_tabs",
        "processing.progress",
        "uxp.audio.clip_path",
        "uxp.audio.ai_noise_reduction",
        "uxp.audio.select_clip_placeholder",
        "uxp.audio.denoise_afftdn",
        "uxp.audio.limit_true_peak",
        "uxp.audio.reference_audio_placeholder",
        "uxp.audio.detect_beats_add_markers",
        "uxp.captions.transcription",
        "uxp.captions.workflow_readiness",
        "uxp.captions.select_clip_placeholder",
        "uxp.captions.model_turbo",
        "uxp.captions.language_auto",
        "uxp.captions.current_plan",
        "uxp.captions.result_details",
        "uxp.captions.result_placeholder",
        "uxp.cut.clip_input",
        "uxp.cut.clip_path_placeholder",
        "uxp.cut.detect_auto",
        "uxp.cut.filler_detection_backend",
        "uxp.cut.detect_remove_fillers",
        "uxp.cut.apply_cuts_to_timeline",
        "uxp.fcc.caption_display_settings",
        "uxp.fcc.compliance_notice_prefix",
        "uxp.fcc.source_link",
        "uxp.fcc.text_color",
        "uxp.fcc.preview",
        "uxp.fcc.loading_tokens",
        "uxp.fcc.caption_preview_sample",
        "uxp.tabs.cut",
        "uxp.tabs.deliverables",
        "uxp.workspace.current_context",
        "uxp.workspace.choose_media",
        "uxp.guide.choose_media_title",
    }.issubset(html_keys)

    missing = sorted(key for key in html_keys if key not in locale)
    assert missing == []


def test_uxp_dynamic_i18n_keys_are_covered_by_locale():
    js_keys = _js_i18n_keys()
    locale = _locale()

    assert {
        "conn.online",
        "conn.connecting",
        "conn.offline",
        "uxp.status.backend_connected",
        "uxp.status.backend_offline",
        "uxp.guide.backend_offline_title",
        "uxp.guide.writeback_ready_title",
        "uxp.fcc.rendering_preview",
        "uxp.fcc.preview_failed",
        "uxp.fcc.defaults_loaded",
        "uxp.fcc.token_schema_failed",
        "uxp.workspace.library_clip_count_many",
    }.issubset(js_keys)

    missing = sorted(key for key in js_keys if key not in locale)
    assert missing == []


def test_uxp_connection_state_does_not_depend_on_visible_english_label():
    js = _js()

    assert "function isBackendConnected()" in js
    assert 'textContent?.trim() === "Online"' not in js
    assert 'dataset.state === "connected"' in js
