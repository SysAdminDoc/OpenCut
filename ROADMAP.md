# Roadmap

Single task tracker for known issues and planned work. Items below come from
verified engineering/product audits through 2026-07-14 (with file locations);
fixes already shipped are recorded in CHANGELOG.md and git
history, not here.

Blocked items (credential/license/hardware-gated) live in
[`Roadmap_Blocked.md`](Roadmap_Blocked.md).

## Research-Driven Additions

### P0

### P3

- [ ] P3 — Add explicit `__all__` to decomposed route submodules
  Why: The captions/system/Wave L facades chain `from .x import *` without `__all__`; a future same-name helper in two submodules would shadow silently in the facade namespace with no error.
  Where: `opencut/routes/wave_l_routes.py`, `opencut/routes/captions.py`, `opencut/routes/system.py` and their submodules.
- [ ] P3 — Decide the version-skew behavior for checkpoint-gated host writes
  Why: `journalCheckpointedHostWrite` (CEP) and `runCheckpointedUxpHostWrite` (UXP) fail closed when `POST /journal/checkpoints` is unavailable; a panel that port-scans onto a ≤1.41.0 backend loses all host writes with "Could not create a recovery checkpoint". Either detect backend version and degrade gracefully, or surface a "backend too old" message.
  Where: `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.uxp/main.js`.
- [ ] P3 — Full-hash review artifact verification on demand
  Why: v1.42.1 added a size fast-path integrity check on portal media serving; a deliberate same-size tamper still passes. A background or on-demand SHA-256 re-verification (per version, surfaced in review metadata) would complete the immutability story.
  Where: `opencut/core/review_portal.py`, `opencut/core/review_links.py`.
