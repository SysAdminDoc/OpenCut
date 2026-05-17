"""User-overridable closed-caption display setting tokens (F236).

The token set mirrors the 47 CFR 79.103 display-setting surface at an
implementation level: font, size, text color/opacity, caption background,
window color, and edge style. It does not claim legal compliance by itself;
it gives OpenCut routes a single normalized contract for preview and burn-in.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Optional, Union

FCC_COMPLIANCE_DATE = "2026-08-17"
FCC_ECFR_URL = "https://www.ecfr.gov/current/title-47/section-79.103"
FCC_COMPLIANCE_NOTICE_URL = "https://www.federalregister.gov/d/2025-02816"

FONT_TOKENS = {
    "default": "Arial",
    "monospace_serif": "Courier New",
    "proportional_serif": "Georgia",
    "monospace_sans": "Consolas",
    "proportional_sans": "Arial",
    "casual": "Comic Sans MS",
    "cursive": "Segoe Script",
    "small_caps": "Arial Small Caps",
}

SIZE_TOKENS = {
    "small": 36,
    "standard": 48,
    "large": 60,
    "extra_large": 72,
}

COLOR_TOKENS = {
    "white": "#FFFFFF",
    "black": "#000000",
    "red": "#FF0000",
    "green": "#00AA00",
    "blue": "#0066FF",
    "yellow": "#FFFF00",
    "magenta": "#FF00FF",
    "cyan": "#00FFFF",
}

OPACITY_TOKENS = {
    "solid": 1.0,
    "translucent": 0.65,
    "transparent": 0.0,
}

EDGE_TOKENS = {
    "none": {"outline": 0, "shadow": 0, "css": "none"},
    "raised": {"outline": 2, "shadow": 1, "css": "1px 1px 0 #FFFFFF, -1px -1px 0 #000000"},
    "depressed": {"outline": 2, "shadow": 1, "css": "-1px -1px 0 #FFFFFF, 1px 1px 0 #000000"},
    "uniform": {"outline": 3, "shadow": 0, "css": "0 0 2px #000000, 0 0 4px #000000"},
    "drop_shadow": {"outline": 1, "shadow": 3, "css": "3px 3px 3px rgba(0,0,0,0.85)"},
}


@dataclass(frozen=True)
class CaptionDisplaySettings:
    font: str = "proportional_sans"
    size: str = "standard"
    text_color: str = "white"
    text_opacity: str = "solid"
    background_color: str = "black"
    background_opacity: str = "translucent"
    edge_style: str = "uniform"
    window_color: str = "black"
    window_opacity: str = "transparent"

    def as_dict(self) -> dict:
        return asdict(self)


DEFAULT_DISPLAY_SETTINGS = CaptionDisplaySettings()

READILY_ACCESSIBLE_FACTORS = [
    {
        "id": "proximity",
        "label": "Proximity",
        "implementation_hint": "Expose all caption display controls from one caption settings surface.",
    },
    {
        "id": "discoverability",
        "label": "Discoverability",
        "implementation_hint": "Make the caption settings entry point visible and name it consistently.",
    },
    {
        "id": "previewability",
        "label": "Previewability",
        "implementation_hint": "Show a live sample while a user changes font, size, color, opacity, or edge tokens.",
    },
    {
        "id": "consistency_persistence",
        "label": "Consistency and persistence",
        "implementation_hint": "Persist the selected tokens and reuse them for caption preview/export surfaces.",
    },
]


def token_schema() -> dict:
    """Return the canonical caption display setting token schema."""
    return {
        "compliance_date": FCC_COMPLIANCE_DATE,
        "sources": {
            "ecfr_47_cfr_79_103": FCC_ECFR_URL,
            "federal_register_compliance_date": FCC_COMPLIANCE_NOTICE_URL,
        },
        "readily_accessible_factors": READILY_ACCESSIBLE_FACTORS,
        "defaults": DEFAULT_DISPLAY_SETTINGS.as_dict(),
        "tokens": {
            "font": [{"id": key, "font_family": value} for key, value in FONT_TOKENS.items()],
            "size": [{"id": key, "font_size": value} for key, value in SIZE_TOKENS.items()],
            "color": [{"id": key, "hex": value} for key, value in COLOR_TOKENS.items()],
            "opacity": [{"id": key, "alpha": value} for key, value in OPACITY_TOKENS.items()],
            "edge_style": [
                {
                    "id": key,
                    "outline": value["outline"],
                    "shadow": value["shadow"],
                }
                for key, value in EDGE_TOKENS.items()
            ],
        },
    }


def _coerce_token(raw: Mapping[str, object], key: str, allowed: Mapping[str, object], default: str) -> str:
    value = str(raw.get(key, default)).strip().lower().replace("-", "_")
    return value if value in allowed else default


def normalise_display_settings(raw: Optional[Mapping[str, object]] = None) -> CaptionDisplaySettings:
    """Normalize caller-provided caption display settings onto known tokens."""
    src: Mapping[str, object] = raw if isinstance(raw, Mapping) else {}
    return CaptionDisplaySettings(
        font=_coerce_token(src, "font", FONT_TOKENS, DEFAULT_DISPLAY_SETTINGS.font),
        size=_coerce_token(src, "size", SIZE_TOKENS, DEFAULT_DISPLAY_SETTINGS.size),
        text_color=_coerce_token(src, "text_color", COLOR_TOKENS, DEFAULT_DISPLAY_SETTINGS.text_color),
        text_opacity=_coerce_token(src, "text_opacity", OPACITY_TOKENS, DEFAULT_DISPLAY_SETTINGS.text_opacity),
        background_color=_coerce_token(
            src, "background_color", COLOR_TOKENS, DEFAULT_DISPLAY_SETTINGS.background_color
        ),
        background_opacity=_coerce_token(
            src, "background_opacity", OPACITY_TOKENS, DEFAULT_DISPLAY_SETTINGS.background_opacity
        ),
        edge_style=_coerce_token(src, "edge_style", EDGE_TOKENS, DEFAULT_DISPLAY_SETTINGS.edge_style),
        window_color=_coerce_token(src, "window_color", COLOR_TOKENS, DEFAULT_DISPLAY_SETTINGS.window_color),
        window_opacity=_coerce_token(src, "window_opacity", OPACITY_TOKENS, DEFAULT_DISPLAY_SETTINGS.window_opacity),
    )


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    clean = hex_color.strip().lstrip("#")
    return int(clean[0:2], 16), int(clean[2:4], 16), int(clean[4:6], 16)


def _css_rgba(color_token: str, opacity_token: str) -> str:
    r, g, b = _hex_to_rgb(COLOR_TOKENS[color_token])
    alpha = OPACITY_TOKENS[opacity_token]
    return f"rgba({r},{g},{b},{alpha:.2f})"


def _ass_color(color_token: str, opacity_token: str) -> str:
    r, g, b = _hex_to_rgb(COLOR_TOKENS[color_token])
    alpha = OPACITY_TOKENS[opacity_token]
    ass_alpha = int(round((1.0 - alpha) * 255))
    return f"&H{ass_alpha:02X}{b:02X}{g:02X}{r:02X}"


DisplaySettingsInput = Optional[Union[CaptionDisplaySettings, Mapping[str, object]]]


def settings_to_ass_force_style(settings: DisplaySettingsInput = None) -> str:
    """Convert settings into an FFmpeg/libass force_style string."""
    normalized = (
        settings
        if isinstance(settings, CaptionDisplaySettings)
        else normalise_display_settings(settings)
    )
    edge = EDGE_TOKENS[normalized.edge_style]
    parts = {
        "FontName": FONT_TOKENS[normalized.font],
        "FontSize": str(SIZE_TOKENS[normalized.size]),
        "PrimaryColour": _ass_color(normalized.text_color, normalized.text_opacity),
        "BackColour": _ass_color(normalized.background_color, normalized.background_opacity),
        "OutlineColour": _ass_color("black", "solid"),
        "Outline": str(edge["outline"]),
        "Shadow": str(edge["shadow"]),
        "BorderStyle": "3" if OPACITY_TOKENS[normalized.background_opacity] > 0 else "1",
    }
    return ",".join(f"{key}={value}" for key, value in parts.items())


def settings_to_preview_css(settings: DisplaySettingsInput = None) -> Dict[str, str]:
    """Convert settings into CSS-like preview values for panel clients."""
    normalized = (
        settings
        if isinstance(settings, CaptionDisplaySettings)
        else normalise_display_settings(settings)
    )
    return {
        "fontFamily": FONT_TOKENS[normalized.font],
        "fontSize": f"{SIZE_TOKENS[normalized.size]}px",
        "color": _css_rgba(normalized.text_color, normalized.text_opacity),
        "backgroundColor": _css_rgba(normalized.background_color, normalized.background_opacity),
        "textShadow": EDGE_TOKENS[normalized.edge_style]["css"],
        "windowColor": _css_rgba(normalized.window_color, normalized.window_opacity),
    }


def build_preview_payload(
    settings: DisplaySettingsInput = None,
    sample_text: str = "Caption preview",
) -> dict:
    """Return normalized settings plus preview/rendering hints."""
    normalized = (
        settings
        if isinstance(settings, CaptionDisplaySettings)
        else normalise_display_settings(settings)
    )
    return {
        "sample_text": sample_text[:200],
        "settings": normalized.as_dict(),
        "preview_css": settings_to_preview_css(normalized),
        "ass_force_style": settings_to_ass_force_style(normalized),
        "token_schema": token_schema(),
    }
