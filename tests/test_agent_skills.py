"""Built-in agent skill catalogue tests."""

from __future__ import annotations

from opencut.core.agent_skills import (
    get_builtin_skill,
    list_builtin_skills,
    validate_skill_plan,
)


def test_wedding_skill_manifest_loads():
    skills = list_builtin_skills()
    ids = {skill.skill_id for skill in skills}

    assert "wedding-cinematic-reel" in ids

    skill = get_builtin_skill("wedding-cinematic-reel")
    assert skill is not None
    summary = skill.summary()
    assert summary["name"] == "Wedding Cinematic Reel"
    assert summary["category"] == "event"
    assert summary["default_target_duration_seconds"] == 240
    assert summary["step_count"] == 5
    assert "editing.color-match" in summary["required_features"]
    assert "audio.beat-markers" in summary["required_features"]


def test_wedding_skill_plan_routes_exist_in_live_app():
    from opencut.tools.dump_route_manifest import build_manifest

    skill = get_builtin_skill("wedding-cinematic-reel")
    assert skill is not None
    assert validate_skill_plan(skill.skill_id, skill.plan) == []

    endpoints = [step["endpoint"] for step in skill.plan["steps"]]
    assert endpoints == [
        "/video/color-match",
        "/audio/beat-markers",
        "/video/highlights",
        "/video/merge",
        "/export-video",
    ]

    live_routes = {route["rule"] for route in build_manifest()["routes"]}
    assert set(endpoints).issubset(live_routes)


def test_agent_skills_routes_expose_wedding_skill(client):
    response = client.get("/agent/skills")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["count"] >= 1
    assert any(skill["id"] == "wedding-cinematic-reel" for skill in payload["skills"])

    detail = client.get("/agent/skills/wedding-cinematic-reel")
    assert detail.status_code == 200
    skill = detail.get_json()
    assert skill["id"] == "wedding-cinematic-reel"
    assert skill["plan"]["default_target_duration_seconds"] == 240
    assert skill["plan"]["steps"][1]["endpoint"] == "/audio/beat-markers"
    assert "reference_clip_path" in {
        item["name"] for item in skill["plan"]["inputs"]
    }


def test_agent_skill_route_returns_404_for_unknown_skill(client):
    response = client.get("/agent/skills/not-a-real-skill")
    assert response.status_code == 404
    assert response.get_json()["skill_id"] == "not-a-real-skill"
