"""Built-in agent skill catalogue.

Skills are lightweight, inspectable orchestration packages.  Each built-in
skill lives under ``opencut/data/builtin_skills/<skill-id>/`` with a
``SKILL.md`` front matter block and a structured ``plan.json``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BUILTIN_SKILLS_DIR = Path(__file__).resolve().parents[1] / "data" / "builtin_skills"
SUPPORTED_SCHEMA_MAJOR = 1
_SKILL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


class SkillLoadError(ValueError):
    """Raised when a skill package is malformed."""


@dataclass(frozen=True)
class AgentSkill:
    """A parsed built-in agent skill."""

    skill_id: str
    metadata: Dict[str, Any]
    instructions: str
    plan: Dict[str, Any]
    manifest_path: Path
    plan_path: Path

    def summary(self) -> Dict[str, Any]:
        """Return the compact skill shape used by list endpoints."""
        return {
            "id": self.skill_id,
            "name": str(self.metadata.get("title") or self.metadata.get("name") or self.skill_id),
            "description": str(self.metadata.get("description") or ""),
            "version": str(self.metadata.get("version") or ""),
            "category": str(self.metadata.get("category") or "general"),
            "license": str(self.metadata.get("license") or ""),
            "applicable_to": list(self.metadata.get("applicable_to") or []),
            "required_features": list(self.metadata.get("required_features") or []),
            "estimated_seconds": int(self.metadata.get("estimated_seconds") or 0),
            "default_target_duration_seconds": int(self.plan.get("default_target_duration_seconds") or 0),
            "step_count": len(self.plan.get("steps") or []),
            "inputs": list(self.plan.get("inputs") or []),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Return the full skill payload."""
        payload = self.summary()
        payload.update({
            "metadata": dict(self.metadata),
            "instructions": self.instructions,
            "plan": self.plan,
            "manifest_path": str(self.manifest_path),
            "plan_path": str(self.plan_path),
        })
        return payload


def _parse_scalar(value: str) -> Any:
    raw = value.strip()
    if raw == "":
        return ""
    if raw[0] in "[{\"" or raw in {"true", "false", "null"}:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw.strip("\"")
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def parse_skill_markdown(text: str) -> Tuple[Dict[str, Any], str]:
    """Parse the minimal SKILL.md front matter shape used by built-ins."""
    if not text.startswith("---"):
        raise SkillLoadError("SKILL.md is missing front matter")
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise SkillLoadError("SKILL.md front matter is not closed")

    metadata: Dict[str, Any] = {}
    for line in parts[1].splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise SkillLoadError(f"Invalid front matter line: {line!r}")
        key, value = stripped.split(":", 1)
        key = key.strip()
        if not key:
            raise SkillLoadError(f"Invalid front matter key: {line!r}")
        metadata[key] = _parse_scalar(value)

    instructions = parts[2].strip()
    return metadata, instructions


def validate_skill_plan(skill_id: str, plan: Dict[str, Any]) -> List[str]:
    """Return human-readable validation errors for a skill plan."""
    errors: List[str] = []
    schema_version = plan.get("schema_version")
    if not isinstance(schema_version, int):
        errors.append("plan.schema_version must be an integer")
    elif schema_version != SUPPORTED_SCHEMA_MAJOR:
        errors.append(f"unsupported plan schema_version {schema_version!r}")

    if plan.get("skill_id") != skill_id:
        errors.append("plan.skill_id must match the skill directory/front matter id")

    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        errors.append("plan.steps must be a non-empty list")
        return errors

    seen_ids = set()
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            errors.append(f"step {index} must be an object")
            continue
        step_id = step.get("id")
        if not isinstance(step_id, str) or not step_id:
            errors.append(f"step {index} is missing id")
        elif step_id in seen_ids:
            errors.append(f"duplicate step id: {step_id}")
        else:
            seen_ids.add(step_id)
        endpoint = step.get("endpoint")
        if not isinstance(endpoint, str) or not endpoint.startswith("/"):
            errors.append(f"step {index} is missing a route endpoint")
        method = step.get("method", "POST")
        if method not in {"GET", "POST", "DELETE", "PUT", "PATCH"}:
            errors.append(f"step {index} has unsupported method {method!r}")
        if not isinstance(step.get("params", {}), dict):
            errors.append(f"step {index} params must be an object")
    return errors


def _load_skill_dir(skill_dir: Path) -> AgentSkill:
    manifest_path = skill_dir / "SKILL.md"
    plan_path = skill_dir / "plan.json"
    if not manifest_path.is_file():
        raise SkillLoadError(f"{skill_dir} is missing SKILL.md")
    if not plan_path.is_file():
        raise SkillLoadError(f"{skill_dir} is missing plan.json")

    metadata, instructions = parse_skill_markdown(manifest_path.read_text(encoding="utf-8"))
    skill_id = str(metadata.get("id") or metadata.get("name") or skill_dir.name).strip()
    if not _SKILL_ID_RE.match(skill_id):
        raise SkillLoadError(f"Invalid skill id: {skill_id!r}")

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    errors = validate_skill_plan(skill_id, plan)
    if errors:
        raise SkillLoadError("; ".join(errors))

    return AgentSkill(
        skill_id=skill_id,
        metadata=metadata,
        instructions=instructions,
        plan=plan,
        manifest_path=manifest_path,
        plan_path=plan_path,
    )


def list_builtin_skills(category: str = "") -> List[AgentSkill]:
    """Load all built-in skills, optionally filtering by category."""
    if not BUILTIN_SKILLS_DIR.is_dir():
        return []
    category = category.strip().lower()
    skills: List[AgentSkill] = []
    for child in sorted(BUILTIN_SKILLS_DIR.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        skill = _load_skill_dir(child)
        if category and str(skill.metadata.get("category", "")).lower() != category:
            continue
        skills.append(skill)
    return skills


def get_builtin_skill(skill_id: str) -> Optional[AgentSkill]:
    """Return a built-in skill by id, or ``None`` when it does not exist."""
    skill_id = skill_id.strip()
    if not _SKILL_ID_RE.match(skill_id):
        return None
    skill_dir = BUILTIN_SKILLS_DIR / skill_id
    if not skill_dir.is_dir():
        return None
    return _load_skill_dir(skill_dir)
