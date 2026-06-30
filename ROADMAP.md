# Roadmap

Single task tracker for known issues and planned work. Items below come from
the June 2026 engineering/product audit (verified findings, with file
locations); fixes already shipped are recorded in CHANGELOG.md and git
history, not here.

Blocked items (credential/license/hardware-gated) live in
[`Roadmap_Blocked.md`](Roadmap_Blocked.md).

## Research-Driven Additions

### P0

- [ ] P0 — Restore CEP i18n catalog completeness
  Why: `scripts/i18n_lint.py --check` fails with 51 consumed-but-missing keys from recent palette and plugin trust UI work.
  Evidence: `scripts/i18n_lint.py --check`; `tests/test_i18n_drift.py`; `extension/com.opencut.panel/i18n/en.json`; `extension/com.opencut.panel/i18n/es.json`
  Touches: `extension/com.opencut.panel/i18n/en.json`, `extension/com.opencut.panel/i18n/es.json`, `tests/test_i18n_drift.py`
  Acceptance: `py -3.12 scripts/i18n_lint.py --check` and `py -3.12 -m pytest tests/test_i18n_drift.py tests/test_uxp_i18n.py tests/test_i18n_hardcoded_migration.py -q` pass.
  Complexity: S

- [ ] P0 — Fix the resumable-job invariant test to recognize decorator kwargs
  Why: `tests/test_job_resume.py` fails even though `shorts_pipeline` is marked `resumable=True`; the guard searches for an exact decorator string and misses the added `rate_limit_key`.
  Evidence: `tests/test_job_resume.py:235`; `opencut/routes/video_specialty.py:256`
  Touches: `tests/test_job_resume.py`, possibly `opencut/jobs.py`
  Acceptance: `py -3.12 -m pytest tests/test_job_resume.py -q` passes, and the test still fails if `resumable=True` is removed from checkpointable routes.
  Complexity: S

### P1

- [ ] P1 — Redact remote render-node secrets and validate node URLs
  Why: remote-node APIs currently return raw `api_key` values and accept arbitrary URLs before calling `<url>/health`, which weakens the local trust boundary.
  Evidence: `opencut/core/remote_process.py`; `opencut/routes/remote_realtime_routes.py`; OWASP SSRF Prevention Cheat Sheet
  Touches: `opencut/core/remote_process.py`, `opencut/routes/remote_realtime_routes.py`, `opencut/security.py`, `tests/test_remote_realtime.py`
  Acceptance: register/list responses never include raw node API keys; unsupported schemes, embedded credentials, malformed hosts, and disallowed private-network targets are rejected with tests; persisted local config still works for approved nodes.
  Complexity: M

- [ ] P1 — Unify social publishing routes around real upload versus dry-run export prep
  Why: `/social/upload` uses real platform upload code, but `/publish/upload` returns the `multi_publish` stub's `pending_upload` result while README and command palette language imply direct OAuth publishing.
  Evidence: `opencut/routes/system.py:2676`; `opencut/routes/editing_workflow_routes.py:245`; `opencut/core/multi_publish.py:249`; YouTube/TikTok/Instagram publishing API docs
  Touches: `opencut/routes/editing_workflow_routes.py`, `opencut/core/multi_publish.py`, `opencut/core/social_post.py`, `opencut/core/command_palette.py`, README copy/tests
  Acceptance: users entering any publish/upload surface either perform a real platform API upload through `social_post.py` or receive an explicit dry-run/export-prep response; no route reports publish completion for the stub path.
  Complexity: M

- [ ] P1 — Refresh MCP server docs and bridge counts from generated manifests
  Why: MCP docs and bridge comments still cite 39 curated tools and 1,325 extended tools while generated state reports newer counts.
  Evidence: `docs/MCP_SERVER.md`; `opencut/routes/mcp_bridge_routes.py`; `opencut/_generated/mcp_extended_tools.json`; `CHANGELOG.md`
  Touches: `docs/MCP_SERVER.md`, `opencut/routes/mcp_bridge_routes.py`, `tests/test_mcp_registry_manifest.py`, `tests/test_mcp_extended_tools.py`
  Acceptance: curated and extended tool counts in docs/comments are generated or checked, and `py -3.12 -m opencut.tools.dump_mcp_extended_tools --check` plus the MCP manifest tests pass.
  Complexity: S

- [ ] P1 — Remove dead roadmap anchors from model-card and feature-readiness install hints
  Why: generated model docs still mention `roadmap H2.x`/`H3.x` anchors that no longer exist in the live root roadmap.
  Evidence: `opencut/model_cards.py`; `docs/MODELS.md`; `opencut/_generated/model_cards.json`; `opencut/_generated/feature_readiness.json`
  Touches: `opencut/model_cards.py`, `opencut/tools/dump_model_cards.py`, `opencut/tools/dump_feature_readiness.py`, `docs/MODELS.md`, generated manifests, `tests/test_model_cards.py`
  Acceptance: generated model and feature-readiness outputs contain no dead `roadmap H*` references, replacement wording points to stable readiness/model docs, and model-card/readiness generator checks pass.
  Complexity: S

### P2

- [ ] P2 — Wire or retire the FastAPI `/ws/jobs` WebSocket stub
  Why: the migration layer exposes "WebSocket job updates coming soon" while the repository already has a real WebSocket bridge for job/live-update behavior.
  Evidence: `opencut/core/fastapi_app.py`; `opencut/core/ws_bridge.py`; `tests/test_architecture.py`
  Touches: `opencut/core/fastapi_app.py`, `opencut/core/ws_bridge.py`, `tests/test_architecture.py`, WebSocket documentation
  Acceptance: `/ws/jobs` either emits real job updates through the existing bridge contract or is removed/hidden from supported migration docs and tests; no shipped endpoint returns a "coming soon" echo.
  Complexity: M

- [ ] P2 — Make UXP locale lint warnings fail after fixing Spanish diacritics
  Why: `scripts/lint_locales.py` currently exits cleanly while warning on Spanish UXP copy such as missing accents in `línea`, `acción`, `ejecución`, and `información`.
  Evidence: `scripts/lint_locales.py`; `extension/com.opencut.uxp/i18n/es.json`
  Touches: `extension/com.opencut.uxp/i18n/es.json`, `scripts/lint_locales.py`, locale lint tests
  Acceptance: locale lint reports zero Spanish diacritic warnings and CI/local release checks fail on future warning-level locale regressions.
  Complexity: S

- [ ] P2 — Add a generated-doc drift gate for README/MCP/model capability facts
  Why: recent fixes moved several facts to generated checks, but stale MCP counts and dead model-card roadmap anchors show that docs can still drift from manifests.
  Evidence: `scripts/sync_badges.py`; `opencut/tools/dump_mcp_extended_tools.py`; `opencut/tools/dump_model_cards.py`; `docs/MCP_SERVER.md`; `docs/MODELS.md`
  Touches: `scripts/release_smoke.py`, doc generator/check scripts, README/doc tests
  Acceptance: one local smoke step detects stale README facts, MCP counts, and model-card/generated-readiness anchors before push; the step passes after regenerated docs.
  Complexity: M
