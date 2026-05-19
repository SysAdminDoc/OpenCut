import pytest


def test_delivery_standard_inventory_covers_open_backlog_items():
    from opencut.core.delivery_standards import (
        count_required_tools,
        delivery_standard_f_numbers,
        delivery_standard_ids,
        list_delivery_standard_presets,
    )

    assert delivery_standard_ids() == (
        "adm_bwf_atmos_master",
        "dolby_vision_profile_5_8_1",
        "dpp_imf_broadcast",
        "netflix_imf_dolby_vision",
    )
    assert delivery_standard_f_numbers() == ("F245", "F246", "F247", "F248")

    presets = list_delivery_standard_presets()
    assert len(presets) == 4
    assert all(preset["commercial_boundary"] for preset in presets)

    tool_counts = count_required_tools()
    assert tool_counts["required"] >= 6
    assert tool_counts["optional"] >= 4


def test_netflix_imf_plan_keeps_certification_boundary_explicit():
    from opencut.core.delivery_standards import build_delivery_standard_plan

    plan = build_delivery_standard_plan(
        "netflix_imf_dolby_vision",
        source_path="C:/media/source.mov",
        output_dir="C:/masters",
        title="Episode 101",
    )

    assert plan["preset_id"] == "netflix_imf_dolby_vision"
    assert plan["f_numbers"] == ["F245"]
    assert plan["runs_external_tools"] is False
    assert "Netflix" in plan["commercial_boundary"]

    commands = plan["commands"]
    flattened = [item for command in commands for item in command["argv"]]
    assert "dovi_tool" in flattened
    assert "mp4dash" in flattened
    assert any(command["label"] == "Validate IMF XML" for command in commands)
    assert any(command["required"] is False for command in commands)


def test_dpp_plan_includes_metadata_sidecar_and_receiver_warnings():
    from opencut.core.delivery_standards import build_delivery_standard_plan

    plan = build_delivery_standard_plan("dpp_imf_broadcast", title="Broadcast")

    assert plan["f_numbers"] == ["F246"]
    assert plan["target_profile"].startswith("IMF Application")
    assert any("DPP" in note for note in plan["constraints"])
    assert any("dpp_metadata.json" in " ".join(command["argv"]) for command in plan["commands"])


def test_dolby_vision_profile_plan_validates_supported_profiles():
    from opencut.core.delivery_standards import build_delivery_standard_plan

    profile_5 = build_delivery_standard_plan(
        "dolby_vision_profile_5_8_1",
        dolby_profile="5",
        title="DV",
    )
    assert profile_5["f_numbers"] == ["F247"]
    assert any("Profile 5" in command["label"] for command in profile_5["commands"])
    assert any("Profile 7" in constraint for constraint in profile_5["constraints"])

    with pytest.raises(ValueError, match="dolby_profile"):
        build_delivery_standard_plan("dolby_vision_profile_5_8_1", dolby_profile="7")


def test_adm_bwf_plan_exposes_axml_chna_and_dolby_encode_boundary():
    from opencut.core.delivery_standards import build_delivery_standard_plan

    plan = build_delivery_standard_plan("adm_bwf_atmos_master", adm_render_layout="0+5+0")

    assert plan["f_numbers"] == ["F248"]
    assert "ADM BW64" in plan["target_profile"]
    commands = plan["commands"]
    assert any(command["argv"][:2] == ["ear-utils", "dump_axml"] for command in commands)
    assert any(command["argv"][:2] == ["ear-utils", "dump_chna"] for command in commands)
    assert any(command["argv"][0] == "dee" and command["required"] is False for command in commands)
    assert ".ec3" in plan["commercial_boundary"]


def test_delivery_mastering_routes_return_presets_and_plans():
    from flask import Flask

    from opencut.routes.delivery_master_routes import delivery_master_bp

    app = Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(delivery_master_bp)
    client = app.test_client()

    presets_resp = client.get("/delivery/mastering-presets")
    assert presets_resp.status_code == 200
    presets = presets_resp.get_json()
    assert presets["count"] == 4
    assert presets["f_numbers"] == ["F245", "F246", "F247", "F248"]
    assert presets["runs_external_tools"] is False

    plan_resp = client.get("/delivery/mastering-plan?preset=adm_bwf_atmos_master")
    assert plan_resp.status_code == 200
    plan = plan_resp.get_json()["plan"]
    assert plan["preset_id"] == "adm_bwf_atmos_master"

    bad_resp = client.get("/delivery/mastering-plan?preset=unknown")
    assert bad_resp.status_code == 400

    post_resp = client.post("/delivery/mastering-plan", json={"preset": "adm_bwf_atmos_master"})
    assert post_resp.status_code == 403
