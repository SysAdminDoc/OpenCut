# OpenCut Completed Work

Root summary of shipped roadmap work. Detailed historical ledgers remain in
[ROADMAP-COMPLETED.md](ROADMAP-COMPLETED.md) and [CHANGELOG.md](CHANGELOG.md).

Last consolidated: 2026-06-05.

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
| Panel UX | CEP panel, UXP panel, command palette, status surfaces, quick actions, E15 i18n migration through batch 150, keyboard/a11y gates, UXP Agent tab, caption display settings, and panel parity checks. |
| Release trust | Route manifest, OpenAPI checks, version sync, release smoke, pip/npm advisory gates, SBOM, installer policy, signing/notarization wiring, Linux packaging, and generated README badges. |
| Review and delivery | Review bundles, marker sidecars, SVG annotations, threaded comments, LAN review portal, notifications/webhooks, transfer bundles, delivery standards, and audio-description draft planning. |
| AI/model governance | Model cards, feature readiness registry, AI eval harness, C2PA sidecars, deepfake detection, AI feature reconciliation, optional telemetry policy, and local-first defaults. |
| Research closures | The May 25 plan closed or advanced route-surface, agent, UXP MCP, sequence-index, shorts-variant, enhance macro, test-breadth, i18n, a11y, security, and CI governance items. |
| Performance and recovery | The May 26 plan now has N1 closed: transcripts are cached by source SHA-256 plus backend/settings, with shared core integration and cache stats/clear routes. |
| Dependency guidance | The May 26 plan now has N2 closed: optional-dependency failures surface OpenCut extra commands, package hints, and GPU/VRAM notes through structured errors and async job status. |
| Resource contention | The May 26 plan now has N3 closed: GPU semaphore contention waits up to 30 seconds by default and returns retry metadata when the wait budget is exhausted. |
| Webhook discoverability | The May 26 plan now has N6 closed: `/webhooks/event-types`, `/api/webhooks/event-types`, and `/mcp/info` expose webhook event names, canonical replacements, schema pointers, and legacy aliases. |
| Webhook trust | The May 26 plan now has E11 closed: new HTTP webhook registrations require an HMAC secret unless the caller explicitly opts into unsigned local testing. |
| Disk preflight | The May 26 plan now has N4 closed: heavyweight async jobs run output-volume disk checks before job creation and return 507 with required/free/output-dir metadata when space is insufficient. |
| Interrupted-job resume | The May 26 plan now has N5 closed: checkpointable async jobs persist resume metadata and can be re-enqueued from interrupted history through `POST /jobs/<job_id>/resume`. |
| Plugin job registration | The May 26 plan now has N7 closed: plugin manifests can declare background jobs, and plugin routes can enqueue namespaced jobs through the core async-job tracker. |
| Third-party agent skills | The May 26 plan now has N8 closed: validated user skills load from `~/.opencut/skills/<id>/`, share the `/agent/skills` catalogue, and reject unavailable route references at load time. |
| CEP caption display settings | The May 26 plan now has E14 closed: the CEP Captions tab exposes the F236 display-settings card with token loading, live preview, and parity tests. |
| Enriched job metadata | The May 26 plan now has N9 closed: live status, persisted history, diagnostics, and webhooks expose peak resource fields plus explicit terminal exit reasons. |
| Request correlation | The May 26 plan now has N10 closed: async workers and FFmpeg subprocesses carry request IDs through `OPENCUT_REQUEST_ID` and prefixed stderr logs. |
| Workflow safety | The May 26 plan now has E12 closed: workflow validation is derived from route-manifest `workflow.label` metadata, with 53 explicit workflow-safe route opt-ins and metadata drift checks. |
| CLI API escape hatch | The May 26 plan now has E13 closed: `opencut route METHOD PATH` validates against the generated route manifest, handles JSON/query request shaping, fetches CSRF tokens for mutating calls, and prints backend responses for scripts. |
| Lockfile advisory coverage | RA-34 is closed: `requirements-lock.txt` is part of the default pip-audit wrapper/release-smoke target set, the vulnerable `idna` lock pin is refreshed to 3.16, and the fully pinned lockfile target audits with `pip-audit --no-deps`. |

## Historical Detail

- [ROADMAP-COMPLETED.md](ROADMAP-COMPLETED.md) keeps the original completed
  phase table and remaining original-roadmap leftovers.
- [CHANGELOG.md](CHANGELOG.md) is the authoritative chronological release log.
- [ROADMAP.md](ROADMAP.md) remains the detailed F-number and wave-letter
  implementation ledger because release-smoke and roadmap-lint tests rely on it.
