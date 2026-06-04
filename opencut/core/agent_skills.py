"""Agent skill catalogue.

Skills are lightweight, inspectable orchestration packages.  Built-in skills
live under ``opencut/data/builtin_skills/<skill-id>/`` and user-installed
skills live under ``~/.opencut/skills/<skill-id>/``.  Each package contains a
``SKILL.md`` front matter block and a structured ``plan.json``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from opencut.helpers import OPENCUT_DIR

BUILTIN_SKILLS_DIR = Path(__file__).resolve().parents[1] / "data" / "builtin_skills"
USER_SKILLS_DIR = Path(OPENCUT_DIR) / "skills"
ROUTE_MANIFEST_PATH = Path(__file__).resolve().parents[1] / "_generated" / "route_manifest.json"
SUPPORTED_SCHEMA_MAJOR = 1
_SKILL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
logger = logging.getLogger("opencut")


class SkillLoadError(ValueError):
    """Raised when a skill package is malformed."""


@dataclass(frozen=True)
class AgentSkill:
    """A parsed agent skill."""

    skill_id: str
    metadata: Dict[str, Any]
    instructions: str
    plan: Dict[str, Any]
    manifest_path: Path
    plan_path: Path
    source: str = "builtin"

    def summary(self) -> Dict[str, Any]:
        """Return the compact skill shape used by list endpoints."""
        return {
            "id": self.skill_id,
            "source": self.source,
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


def _load_route_method_pairs() -> Set[Tuple[str, str]]:
    """Return the generated host route/method set used for skill validation."""
    try:
        data = json.loads(ROUTE_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Cannot load route manifest for skill validation: %s", exc)
        return set()

    pairs: Set[Tuple[str, str]] = set()
    for route in data.get("routes", []):
        if not isinstance(route, dict):
            continue
        rule = str(route.get("rule") or "")
        if not rule:
            continue
        for method in route.get("methods") or []:
            pairs.add((str(method).upper(), rule))
    return pairs


def validate_skill_plan(
    skill_id: str,
    plan: Dict[str, Any],
    available_routes: Optional[Set[Tuple[str, str]]] = None,
) -> List[str]:
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
        elif available_routes is not None and isinstance(endpoint, str):
            if (method, endpoint) not in available_routes:
                errors.append(f"step {index} references unavailable route {method} {endpoint}")
        if not isinstance(step.get("params", {}), dict):
            errors.append(f"step {index} params must be an object")
    return errors


def _load_skill_dir(
    skill_dir: Path,
    *,
    source: str,
    validate_routes: bool = False,
    available_routes: Optional[Set[Tuple[str, str]]] = None,
) -> AgentSkill:
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
    if skill_id != skill_dir.name:
        raise SkillLoadError("Skill id must match the skill directory name")

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    route_pairs = available_routes if available_routes is not None else (
        _load_route_method_pairs() if validate_routes else None
    )
    errors = validate_skill_plan(skill_id, plan, available_routes=route_pairs)
    if errors:
        raise SkillLoadError("; ".join(errors))

    return AgentSkill(
        skill_id=skill_id,
        metadata=metadata,
        instructions=instructions,
        plan=plan,
        manifest_path=manifest_path,
        plan_path=plan_path,
        source=source,
    )


def _list_skill_dir(
    base_dir: Path,
    *,
    source: str,
    category: str = "",
    validate_routes: bool = False,
) -> List[AgentSkill]:
    """Load all skills under a base directory, optionally filtering by category."""
    if not base_dir.is_dir():
        return []
    category = category.strip().lower()
    route_pairs = _load_route_method_pairs() if validate_routes else None
    skills: List[AgentSkill] = []
    for child in sorted(base_dir.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        try:
            skill = _load_skill_dir(
                child,
                source=source,
                validate_routes=validate_routes,
                available_routes=route_pairs,
            )
        except (SkillLoadError, json.JSONDecodeError, OSError) as exc:
            if source == "user":
                logger.warning("Skipping invalid user skill %s: %s", child.name, exc)
                continue
            raise
        if category and str(skill.metadata.get("category", "")).lower() != category:
            continue
        skills.append(skill)
    return skills


def list_builtin_skills(category: str = "") -> List[AgentSkill]:
    """Load all built-in skills, optionally filtering by category."""
    return _list_skill_dir(BUILTIN_SKILLS_DIR, source="builtin", category=category)


def list_user_skills(category: str = "") -> List[AgentSkill]:
    """Load validated user-installed skills from ``~/.opencut/skills``."""
    return _list_skill_dir(
        USER_SKILLS_DIR,
        source="user",
        category=category,
        validate_routes=True,
    )


def list_skills(category: str = "") -> List[AgentSkill]:
    """Load the combined built-in + user skill catalogue.

    Built-in skill IDs win on collision so user packages cannot shadow a
    bundled workflow.
    """
    skills = list_builtin_skills(category=category)
    seen = {skill.skill_id for skill in skills}
    for skill in list_user_skills(category=category):
        if skill.skill_id in seen:
            logger.warning(
                "Skipping user skill %s because a built-in skill uses that id",
                skill.skill_id,
            )
            continue
        skills.append(skill)
        seen.add(skill.skill_id)
    return skills


def get_builtin_skill(skill_id: str) -> Optional[AgentSkill]:
    """Return a built-in skill by id, or ``None`` when it does not exist."""
    skill_id = skill_id.strip()
    if not _SKILL_ID_RE.match(skill_id):
        return None
    skill_dir = BUILTIN_SKILLS_DIR / skill_id
    if not skill_dir.is_dir():
        return None
    return _load_skill_dir(skill_dir, source="builtin")


def get_user_skill(skill_id: str) -> Optional[AgentSkill]:
    """Return a validated user skill by id, or ``None`` when unavailable."""
    skill_id = skill_id.strip()
    if not _SKILL_ID_RE.match(skill_id):
        return None
    skill_dir = USER_SKILLS_DIR / skill_id
    if not skill_dir.is_dir():
        return None
    try:
        return _load_skill_dir(skill_dir, source="user", validate_routes=True)
    except (SkillLoadError, json.JSONDecodeError, OSError) as exc:
        logger.warning("Skipping invalid user skill %s: %s", skill_id, exc)
        return None


def get_skill(skill_id: str) -> Optional[AgentSkill]:
    """Return a built-in or user skill by id.

    Built-ins are resolved first to avoid user packages shadowing bundled
    workflows.
    """
    return get_builtin_skill(skill_id) or get_user_skill(skill_id)
