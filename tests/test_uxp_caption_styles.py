"""UXP caption-style catalog parity with the backend style library."""

from __future__ import annotations

import re
from pathlib import Path

from opencut.core.caption_styles import BUILTIN_STYLES

REPO_ROOT = Path(__file__).resolve().parents[1]
UXP_INDEX = REPO_ROOT / "extension" / "com.opencut.uxp" / "index.html"
UXP_MAIN = REPO_ROOT / "extension" / "com.opencut.uxp" / "main.js"


def _caption_style_select_html() -> str:
    html = UXP_INDEX.read_text(encoding="utf-8")
    match = re.search(
        r'<select[^>]+id="captionStyle"[^>]*>(?P<body>.*?)</select>',
        html,
        flags=re.DOTALL,
    )
    assert match, "UXP captionStyle select not found"
    return match.group("body")


def _option_values(select_html: str) -> list[str]:
    return re.findall(r'<option[^>]+value="([^"]+)"', select_html)


def test_uxp_caption_style_bootstrap_uses_backend_style_ids():
    values = _option_values(_caption_style_select_html())

    assert values == ["minimal_clean"]
    assert set(values).issubset(BUILTIN_STYLES)
    assert not {
        "youtube_bold",
        "neon_pop",
        "cinematic",
        "netflix",
        "sports",
        "minimal",
    }.intersection(values)


def test_uxp_loads_caption_styles_from_backend_catalog():
    js = UXP_MAIN.read_text(encoding="utf-8")

    assert 'const CAPTION_STYLES_ENDPOINT = "/captions/styles";' in js
    assert "BackendClient.get(CAPTION_STYLES_ENDPOINT)" in js
    assert 'populateCaptionStyleSelect(styles, "backend")' in js
    assert "normalizeCaptionStyleCatalog(styles)" in js
    assert "caption_style: style" in js


def test_backend_caption_style_catalog_exceeds_uxp_bootstrap_fallback():
    values = _option_values(_caption_style_select_html())

    assert len(BUILTIN_STYLES) >= 50
    assert len(values) < len(BUILTIN_STYLES)
