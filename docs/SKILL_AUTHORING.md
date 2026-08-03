# Skill Authoring

Agent skills are inspectable workflow packages for the OpenCut agent catalogue.
Built-in skills live in `opencut/data/builtin_skills/`. User-installed skills
live in `~/.opencut/skills/<skill-id>/`.

Each skill directory must contain:

- `SKILL.md` with YAML-like front matter and human-readable instructions.
- `plan.json` with the structured route plan the agent can execute.

## Directory Layout

```text
~/.opencut/skills/my-review-skill/
  SKILL.md
  plan.json
```

The directory name, `SKILL.md` `id`, and `plan.json` `skill_id` must match.
Skill IDs must use lowercase letters, numbers, dots, underscores, or dashes.

## SKILL.md

```markdown
---
id: my-review-skill
name: my-review-skill
title: My Review Skill
description: Run a short review workflow against the current project.
version: 1.0.0
author: your-name
license: MIT
category: review
applicable_to: ["project"]
required_features: []
estimated_seconds: 30
---

# My Review Skill

Use this skill when the user asks for a quick project review.
```

The front matter parser supports simple scalars and JSON-style lists. Keep
metadata small and put operating instructions in the Markdown body.

## plan.json

```json
{
  "schema_version": 1,
  "skill_id": "my-review-skill",
  "default_target_duration_seconds": 30,
  "inputs": [],
  "steps": [
    {
      "id": "read_agent_info",
      "label": "Read agent capability",
      "method": "GET",
      "endpoint": "/agent/chat/info",
      "params": {}
    }
  ]
}
```

Validation rules:

- `schema_version` must be `1`.
- `skill_id` must match the containing directory and `SKILL.md` ID.
- `steps` must be a non-empty list with unique step IDs.
- `method` must be one of `GET`, `POST`, `DELETE`, `PUT`, or `PATCH`.
- `endpoint` must be an OpenCut route in `opencut/_generated/route_manifest.json`.
- `params` must be an object.

User skills that fail validation are skipped and logged. Built-in skill IDs take
precedence, so a user skill cannot shadow a bundled workflow.

## Catalogue API

`GET /agent/skills` lists built-in and valid user skills. Each entry includes a
`source` field with `builtin` or `user`.

`GET /agent/skills/<skill-id>` returns the full instructions and plan for either
source.
