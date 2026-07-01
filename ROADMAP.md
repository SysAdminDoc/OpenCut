# Roadmap

Single task tracker for known issues and planned work. Items below come from
the June 2026 engineering/product audit (verified findings, with file
locations); fixes already shipped are recorded in CHANGELOG.md and git
history, not here.

Blocked items (credential/license/hardware-gated) live in
[`Roadmap_Blocked.md`](Roadmap_Blocked.md).

## Research-Driven Additions

### P1

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
