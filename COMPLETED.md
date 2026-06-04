# OpenCut Completed Work

Root summary of shipped roadmap work. Detailed historical ledgers remain in
[ROADMAP-COMPLETED.md](ROADMAP-COMPLETED.md) and [CHANGELOG.md](CHANGELOG.md).

Last consolidated: 2026-06-04.

## Current Delivered Baseline

- Live release line: v1.32.0.
- Local-first Premiere automation backend with CEP and UXP panels, Flask routes,
  async job orchestration, MCP bridge, DaVinci Resolve bridge, and installer /
  packaging surfaces.
- Current generated-count sources are the manifest files, not hand-written
  marketing counts.

## Shipped Work By Area

| Area | Completed work |
|---|---|
| Core automation | Silence/filler removal, captions/transcription, audio cleanup, video effects, export, timeline actions, deliverables, search, workflow presets, NLP commands, and plugin loading. |
| Job platform | `@async_job`, SQLite job persistence, priority workers, cancellation, interrupted-job marking, queue allowlists, structured errors, and log correlation. |
| Panel UX | CEP panel, UXP panel, command palette, status surfaces, quick actions, i18n groundwork, keyboard/a11y gates, UXP Agent tab, caption display settings, and panel parity checks. |
| Release trust | Route manifest, OpenAPI checks, version sync, release smoke, pip/npm advisory gates, SBOM, installer policy, signing/notarization wiring, Linux packaging, and generated README badges. |
| Review and delivery | Review bundles, marker sidecars, SVG annotations, threaded comments, LAN review portal, notifications/webhooks, transfer bundles, delivery standards, and audio-description draft planning. |
| AI/model governance | Model cards, feature readiness registry, AI eval harness, C2PA sidecars, deepfake detection, AI feature reconciliation, optional telemetry policy, and local-first defaults. |
| Research closures | The May 25 plan closed or advanced route-surface, agent, UXP MCP, sequence-index, shorts-variant, enhance macro, test-breadth, i18n, a11y, security, and CI governance items. |
| Performance and recovery | The May 26 plan now has N1 closed: transcripts are cached by source SHA-256 plus backend/settings, with shared core integration and cache stats/clear routes. |
| Dependency guidance | The May 26 plan now has N2 closed: optional-dependency failures surface OpenCut extra commands, package hints, and GPU/VRAM notes through structured errors and async job status. |

## Historical Detail

- [ROADMAP-COMPLETED.md](ROADMAP-COMPLETED.md) keeps the original completed
  phase table and remaining original-roadmap leftovers.
- [CHANGELOG.md](CHANGELOG.md) is the authoritative chronological release log.
- [ROADMAP.md](ROADMAP.md) remains the detailed F-number and wave-letter
  implementation ledger because release-smoke and roadmap-lint tests rely on it.
