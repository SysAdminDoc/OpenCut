import pytest
from flask import Flask

from opencut.config import OpenCutConfig
from opencut.routes import (
    RouteCollisionError,
    assert_no_route_collisions,
    find_route_collisions,
    get_core_blueprints,
)
from opencut.server import create_app

EXPECTED_CORE_BLUEPRINT_ORDER = (
    "ai_content",
    "ai_editing",
    "ai_intel",
    "analysis",
    "architecture",
    "audio",
    "audio_adv",
    "audio_expand",
    "audio_post",
    "audio_prod",
    "batch_data",
    "body_transfer",
    "captions",
    "cloud_distrib",
    "collab_review",
    "color_mam",
    "composition_dubbing",
    "content_gen",
    "content",
    "context",
    "creative",
    "deliverables",
    "delivery_master",
    "delivery",
    "dev_scripting",
    "documentary",
    "editing_wf",
    "education",
    "encoding",
    "engagement_content",
    "enhanced_media",
    "enhancement",
    "format",
    "gaming",
    "generative",
    "hw",
    "infra",
    "integration",
    "jobs",
    "journal",
    "motion_design",
    "motion_gen",
    "multiview_repurpose",
    "music_safety",
    "next_gen_ai",
    "nlp",
    "object_intel",
    "overlay",
    "pipeline_intel",
    "platform_infra",
    "platform_ux",
    "plugins",
    "preproduction_proxy",
    "preview_realtime",
    "privacy_spectral",
    "processing",
    "production",
    "professional",
    "qc",
    "remote_realtime",
    "repair_gen",
    "search",
    "settings",
    "solver_agent",
    "sound_music",
    "subtitle_pro",
    "subtitle",
    "system",
    "timeline",
    "timeline_auto",
    "timeline_intel",
    "tools",
    "transcript_edit",
    "utility",
    "ux_intel",
    "vfx_advanced",
    "video_ai",
    "video_core",
    "video_editing",
    "video_effects",
    "video_fx",
    "video_proc",
    "video_specialty",
    "video_vfx",
    "voice_speech",
    "vr_lens",
    "wave_a",
    "wave_b",
    "wave_c",
    "wave_d",
    "wave_e",
    "wave_f",
    "wave_g",
    "wave_h",
    "wave_k",
    "wave_l",
    "workflow",
    "workflow_dev",
    "workflow_auto",
)


def test_no_duplicate_route_method_pairs(app):
    assert find_route_collisions(app) == {}


def test_core_blueprint_registry_has_unique_names():
    names = [bp.name for bp in get_core_blueprints()]

    assert names
    assert len(names) == len(set(names))
    assert "motion_design" in names
    assert "system" in names


def test_core_blueprint_registration_order_is_stable(app):
    core_names = tuple(bp.name for bp in get_core_blueprints())
    registered_names = tuple(app.blueprints)

    assert core_names == EXPECTED_CORE_BLUEPRINT_ORDER
    assert registered_names == EXPECTED_CORE_BLUEPRINT_ORDER + ("motion_design_api",)
    assert app.blueprints["motion_design_api"] is app.blueprints["motion_design"]


def test_collision_prone_routes_bind_to_expected_endpoints(app):
    adapter = app.url_map.bind("")

    assert adapter.match("/api/scripting/execute", method="POST")[0] == "dev_scripting.scripting_execute"
    assert adapter.match("/api/workflow/scripting/execute", method="POST")[0] == "workflow_dev.scripting_execute"
    assert adapter.match("/audio/me-mix", method="POST")[0] == "audio_post.route_me_mix"
    assert adapter.match("/audio/me-mix/basic", method="POST")[0] == "audio_prod.route_me_mix"
    assert adapter.match("/audio/dialogue-premix", method="POST")[0] == "audio_post.route_dialogue_premix"
    assert adapter.match("/audio/dialogue-premix/basic", method="POST")[0] == "audio_prod.route_dialogue_premix"
    assert adapter.match("/timeline/diff", method="POST")[0] == "platform_infra.timeline_diff_route"
    assert adapter.match("/timeline/otio-diff", method="POST")[0] == "wave_c.route_timeline_diff"


def test_find_route_collisions_reports_duplicates():
    app = Flask(__name__)

    app.add_url_rule("/dup", endpoint="first", view_func=lambda: "first", methods=["GET"])
    app.add_url_rule("/dup", endpoint="second", view_func=lambda: "second", methods=["GET"])

    duplicates = find_route_collisions(app)

    assert duplicates == {
        ("/dup", ("GET",)): ["first", "second"],
    }


def test_assert_no_route_collisions_raises_for_duplicates():
    app = Flask(__name__)

    app.add_url_rule("/dup", endpoint="first", view_func=lambda: "first", methods=["POST"])
    app.add_url_rule("/dup", endpoint="second", view_func=lambda: "second", methods=["POST"])

    with pytest.raises(RouteCollisionError, match="Duplicate route registrations detected"):
        assert_no_route_collisions(app)


def test_create_app_invokes_runtime_route_guard(monkeypatch):
    def _boom(_app):
        raise RouteCollisionError("synthetic duplicate")

    monkeypatch.setattr("opencut.routes.assert_no_route_collisions", _boom)
    monkeypatch.setattr("opencut.core.plugins.load_all_plugins", lambda _app: {"loaded": []})

    with pytest.raises(RouteCollisionError, match="synthetic duplicate"):
        create_app(config=OpenCutConfig())
