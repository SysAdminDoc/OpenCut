from opencut.core import caption_display_settings as cds


def test_f236_token_schema_covers_fcc_display_surface():
    schema = cds.token_schema()

    assert schema["compliance_date"] == "2026-08-17"
    assert "ecfr_47_cfr_79_103" in schema["sources"]
    assert {factor["id"] for factor in schema["readily_accessible_factors"]} == {
        "proximity",
        "discoverability",
        "previewability",
        "consistency_persistence",
    }
    assert {item["id"] for item in schema["tokens"]["font"]} >= {
        "default",
        "monospace_serif",
        "proportional_serif",
        "monospace_sans",
        "proportional_sans",
        "casual",
        "cursive",
        "small_caps",
    }
    for item in schema["tokens"]["font"]:
        assert "font_resolution" in item
        assert {"requested_family", "source", "font_path"} <= set(item["font_resolution"])
    assert {item["id"] for item in schema["tokens"]["opacity"]} == {
        "solid",
        "translucent",
        "transparent",
    }
    assert {item["id"] for item in schema["tokens"]["edge_style"]} == {
        "none",
        "raised",
        "depressed",
        "uniform",
        "drop_shadow",
    }


def test_f236_display_settings_are_normalized_and_renderable():
    settings = cds.normalise_display_settings(
        {
            "font": "monospace-serif",
            "size": "large",
            "text_color": "yellow",
            "text_opacity": "translucent",
            "background_color": "blue",
            "background_opacity": "solid",
            "edge_style": "drop-shadow",
            "window_color": "magenta",
            "window_opacity": "transparent",
        }
    )

    assert settings.font == "monospace_serif"
    assert settings.edge_style == "drop_shadow"

    css = cds.settings_to_preview_css(settings)
    assert css["fontFamily"] == "Courier New"
    assert css["fontSize"] == "60px"
    assert css["color"] == "rgba(255,255,0,0.65)"
    assert css["backgroundColor"] == "rgba(0,102,255,1.00)"

    force_style = cds.settings_to_ass_force_style(settings)
    assert "FontName=Courier New" in force_style
    assert "FontSize=60" in force_style
    assert "PrimaryColour=&H59" in force_style
    assert "BackColour=&H00FF6600" in force_style
    assert "Shadow=3" in force_style


def test_f236_display_settings_fallback_to_defaults_for_unknown_tokens():
    settings = cds.normalise_display_settings(
        {
            "font": "not-a-font",
            "size": "massive",
            "text_color": "orange",
            "edge_style": "embossed",
        }
    )

    assert settings == cds.DEFAULT_DISPLAY_SETTINGS


def test_f236_display_setting_routes_expose_tokens_and_preview(client, csrf_token):
    from tests.conftest import csrf_headers

    tokens = client.get("/captions/display-settings/tokens")
    assert tokens.status_code == 200
    token_payload = tokens.get_json()
    assert token_payload["compliance_date"] == "2026-08-17"

    preview = client.post(
        "/captions/display-settings/preview",
        json={
            "sample_text": "Readable captions",
            "settings": {
                "font": "small_caps",
                "size": "extra_large",
                "text_color": "cyan",
                "edge_style": "raised",
            },
        },
        headers=csrf_headers(csrf_token),
    )

    assert preview.status_code == 200
    payload = preview.get_json()
    assert payload["sample_text"] == "Readable captions"
    assert payload["settings"]["font"] == "small_caps"
    assert payload["settings"]["size"] == "extra_large"
    assert payload["preview_css"]["fontSize"] == "72px"
    assert payload["ass_force_style"].startswith("FontName=Arial Small Caps")
