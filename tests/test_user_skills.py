"""User-installed agent skill catalogue tests."""

from __future__ import annotations

import json
import textwrap


def _write_skill(
    root,
    skill_id: str,
    *,
    endpoint: str = "/agent/chat/info",
    method: str = "GET",
    category: str = "automation",
):
    skill_dir = root / skill_id
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        textwrap.dedent(
            f"""\
            ---
            id: {skill_id}
            name: {skill_id}
            title: User Skill {skill_id}
            description: Test user skill for catalogue loading.
            version: 1.0.0
            author: tests
            license: MIT
            category: {category}
            applicable_to: ["project"]
            required_features: []
            estimated_seconds: 30
            ---

            # User Skill {skill_id}

            Use this skill to exercise the third-party skill loader.
            """
        ),
        encoding="utf-8",
    )
    (skill_dir / "plan.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "skill_id": skill_id,
                "default_target_duration_seconds": 30,
                "inputs": [],
                "steps": [
                    {
                        "id": "read_host_capability",
                        "label": "Read host capability",
                        "method": method,
                        "endpoint": endpoint,
                        "params": {},
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return skill_dir


def test_user_skill_directory_loads_into_combined_catalogue(tmp_path, monkeypatch):
    from opencut.core import agent_skills

    user_root = tmp_path / "skills"
    _write_skill(user_root, "daily-social-package")
    monkeypatch.setattr(agent_skills, "USER_SKILLS_DIR", user_root)

    user_skills = agent_skills.list_user_skills()
    assert [skill.skill_id for skill in user_skills] == ["daily-social-package"]
    assert user_skills[0].summary()["source"] == "user"

    combined = {skill.skill_id: skill for skill in agent_skills.list_skills()}
    assert "wedding-cinematic-reel" in combined
    assert combined["daily-social-package"].source == "user"

    detail = agent_skills.get_skill("daily-social-package")
    assert detail is not None
    assert detail.to_dict()["plan"]["steps"][0]["endpoint"] == "/agent/chat/info"


def test_user_skill_route_exposes_validated_user_skill(tmp_path, monkeypatch, client):
    from opencut.core import agent_skills

    user_root = tmp_path / "skills"
    _write_skill(user_root, "timeline-qc-kit", category="review")
    monkeypatch.setattr(agent_skills, "USER_SKILLS_DIR", user_root)

    response = client.get("/agent/skills?category=review")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["count"] == 1
    assert payload["skills"][0]["id"] == "timeline-qc-kit"
    assert payload["skills"][0]["source"] == "user"

    detail = client.get("/agent/skills/timeline-qc-kit")
    assert detail.status_code == 200
    body = detail.get_json()
    assert body["source"] == "user"
    assert body["metadata"]["title"] == "User Skill timeline-qc-kit"


def test_user_skill_rejects_unavailable_plan_endpoint(tmp_path, monkeypatch):
    from opencut.core import agent_skills

    user_root = tmp_path / "skills"
    _write_skill(user_root, "bad-endpoint-skill", endpoint="/not-a-real-opencut-route")
    monkeypatch.setattr(agent_skills, "USER_SKILLS_DIR", user_root)

    assert agent_skills.list_user_skills() == []
    assert agent_skills.get_user_skill("bad-endpoint-skill") is None


def test_user_skill_cannot_shadow_builtin_skill(tmp_path, monkeypatch):
    from opencut.core import agent_skills

    user_root = tmp_path / "skills"
    _write_skill(user_root, "wedding-cinematic-reel")
    monkeypatch.setattr(agent_skills, "USER_SKILLS_DIR", user_root)

    skill = agent_skills.get_skill("wedding-cinematic-reel")
    assert skill is not None
    assert skill.source == "builtin"

    summaries = [
        skill.summary()
        for skill in agent_skills.list_skills()
        if skill.skill_id == "wedding-cinematic-reel"
    ]
    assert len(summaries) == 1
    assert summaries[0]["source"] == "builtin"
