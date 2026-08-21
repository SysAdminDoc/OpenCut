# Roadmap — OpenCut

Actionable work only. Historical and completed roadmap material is archived in CHANGELOG.md; blocked work is kept in Roadmap_Blocked.md.

## Research-Driven Additions

Added 2026-08-10, extended 2026-08-11 and 2026-08-20 from the research passes recorded in `RESEARCH.md`.
IDs continue the existing F-number scheme (highest prior allocation before 2026-08-11: F318; before
2026-08-20: F334).

F359-F365 came from a different source: an audit run on 2026-08-20 over the F335-F358 drain, checking
each closed item against its own acceptance text rather than against the research. Highest prior
allocation before that audit: F358. F359 needed a maintainer's voice on a public issue and moved to
Roadmap_Blocked.md; the rest landed.

### P1

### P2

### P3

- [ ] P3 — F385 — MCP extended tool catalogue is checked but never exercised
  Category: quality
  Where: `opencut/tools/dump_mcp_extended_tools.py`, `tests/test_mcp_extended_tools.py`, the `--check` step in `scripts/release_smoke.py:574`
  Problem: The gate compares the committed manifest to the generated one. Nothing calls the tools it lists, so a tool can be catalogued, shipped, and broken at the same time without any gate noticing.
  Repro: `python -m opencut.tools.dump_mcp_extended_tools --check` passes on the current tree; there is no counterpart that starts the MCP server and invokes a tool.
  Fix: Add a smoke that starts the MCP surface against the test backend and calls a representative tool from each family, asserting the response shape the manifest claims.
  Acceptance: A tool whose handler is deleted fails the new smoke, not just the manifest diff.
  Confidence: Verified
  Effort: M

## Audit Findings — 2026-08-20 (deep audit pass, v1.50.0 at 5588bd4a)

Full-repo engineering/UX/visual audit. Baseline this pass: `python -m pytest tests/ -q` = 11,354 passed,
45 skipped, 0 failures; `ruff check opencut/` clean; panel vitest 225 passed. IDs continue from F365.
Scanner sweeps dismissed as false positives after tracing: all 10 gitleaks hits (dummy fixture tokens in
tests), bandit B307 eval (AST-whitelisted expression sandbox, `ast.Attribute` forbidden), all B608 SQL
hits (parameterised or literal WHERE fragments), B603/B607 subprocess (list-form argv throughout). The
one open GitHub issue (#5) is already tracked as F359 in Roadmap_Blocked.md.

### P1

### P2

### P3
