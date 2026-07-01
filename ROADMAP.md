# Roadmap

Single task tracker for known issues and planned work. Items below come from
the June 2026 engineering/product audit (verified findings, with file
locations); fixes already shipped are recorded in CHANGELOG.md and git
history, not here.

Blocked items (credential/license/hardware-gated) live in
[`Roadmap_Blocked.md`](Roadmap_Blocked.md).

## Research-Driven Additions

### P1

### P2

- [ ] P2 — Add a generated-doc drift gate for README/MCP/model capability facts
  Why: recent fixes moved several facts to generated checks, but stale MCP counts and dead model-card roadmap anchors show that docs can still drift from manifests.
  Evidence: `scripts/sync_badges.py`; `opencut/tools/dump_mcp_extended_tools.py`; `opencut/tools/dump_model_cards.py`; `docs/MCP_SERVER.md`; `docs/MODELS.md`
  Touches: `scripts/release_smoke.py`, doc generator/check scripts, README/doc tests
  Acceptance: one local smoke step detects stale README facts, MCP counts, and model-card/generated-readiness anchors before push; the step passes after regenerated docs.
  Complexity: M
