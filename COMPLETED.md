# OpenCut Completed Work

Root summary of shipped roadmap work. Detailed historical ledgers remain in
[ROADMAP-COMPLETED.md](ROADMAP-COMPLETED.md) and [CHANGELOG.md](CHANGELOG.md).

Last consolidated: 2026-06-07.

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
| Core automation | Silence/filler removal, captions/transcription, caption round-trip sidecars, diff/apply revisions, CEP/hybrid caption write contracts, caption metadata-loss regression fixtures, sequence-index host locators, Magic Clips dry-run plan graph, approved-candidate render handoff, explainable scoring, preset-driven multi-platform rendering, review-board parity, checkpointed resume manifests, output bundle handoff manifests, and downstream timeline/social handoff payloads, audio cleanup, video effects, export, timeline actions, deliverables, search, workflow presets, NLP commands, and plugin loading. |
| Job platform | `@async_job`, worker-lifetime async rate-limit slots, SQLite job persistence, explicit local SQLite schema versions, priority workers, cancellation, interrupted-job marking, queue allowlists, structured errors with request IDs in JSON bodies and typed-error log context, and log correlation. |
| Local data stores | Local SQLite schemas use explicit `user_version` migrations, oversized job-result plus journal inverse/forward JSON payloads spill to content-addressed local files with API-visible metadata, local DB diagnostics report page/freelist/WAL/file-size maintenance posture through CLI and feature routes, destructive local SQLite maintenance paths expose dry-run counts, optional backups, and audit metadata, and user-data deletes/replacements create capped tombstone snapshots with restore routes. |
| Panel UX | CEP panel, UXP panel, command palette, status surfaces, structured CEP empty states, quick actions, E15 i18n migration through batch 173 with a zero-dead-key drift gate, JS metadata-key scanner coverage, `data-i18n-alt` scanner coverage, and UXP i18n foundation/Cut/Captions/FCC display/Audio/Video/Timeline/Search shell guards, keyboard/a11y gates, UXP Agent tab, caption display settings, Magic Clips plan/review/render/bundle boards, UXP caption-track snapshot reads, concrete CEP caption placement handoffs, and panel parity checks. |
| Release trust | Route manifest, OpenAPI checks, version sync, release smoke, WCAG AA panel contrast token audit, route-level rate-limit primitive migration guard, pip/npm advisory gates, audited `opencut[all]` convenience extra policy, Torch deserialization hardening and Torch-stack advisory floors, `open-path` allowlist hardening, CLIP cache safe-deserialization hardening, scripting-console source-size limits, Gaussian splat preview send-file confinement, expression-engine timeline thread-churn reduction, security rejection audit logging for CSRF/path/rate-limit/auth denials, cleanup-thread lazy initialization, Python 3.13 classifier retraction guard, UXP manifest schema-version guard, UXP Hybrid package validator, UXP deprecated-API sentinel, UXP clipboard permission/fallback guard, UXP picker-scoped filesystem permission guard, UXP external-launch permission guard, UXP WebView dev/release permission profiles, UXP inline confirmation guard, declared-only SBOM metadata/naming, installer policy, signing/notarization wiring, Linux packaging, generated README badges and non-badge count gates, Adobe tracker hardening for release-channel tags, deterministic probe exit codes, tracker labels, no-`gh` label dry-runs, GitHub Actions full-SHA pins, Release Full Node 22 panel runtime pin, Release Full read-only build plus write-scoped release-upload job, release artifact provenance attestations, CEP panel shared-folder Node gate aliases, release-smoke Ruff import-order cleanup, Docker distribution-doc guards, Docker HTTP-only default runtime parity, Docker dependency/fail-closed/build-context hygiene guards, test-environment bootstrap checks, and shared destructive-operation dry-run/confirmation helpers for queue/log/cache/model/render-cache/temp/plugin/user-data/chat/undo/search/worker-pool deletes and clears. |
| Review and delivery | Review bundles, marker sidecars, SVG annotations, threaded comments, LAN review portal, notifications/webhooks, transfer bundles, delivery standards, and audio-description draft planning. |
| AI/model governance | Model cards, feature readiness registry, AI eval harness, C2PA sidecars, deepfake detection, AI feature reconciliation, optional telemetry policy, and local-first defaults. |
| Research closures | The May 25 plan closed or advanced route-surface, agent, UXP MCP, sequence-index, shorts-variant, enhance macro, test-breadth, i18n, a11y, security, and CI governance items. |
| Performance and recovery | The May 26 plan now has N1 closed: transcripts are cached by source SHA-256 plus backend/settings, with shared core integration and cache stats/clear routes; render-cache cleanup and invalidation also reject forged index output paths before unlinking, model/cache deletion paths expose preview plans with per-path errors, expression timeline evaluation no longer creates one worker thread per frame, and importing shared helpers no longer starts the deferred temp-cleanup worker until cleanup is scheduled. |
| Dependency guidance | The May 26 plan now has N2 closed: optional-dependency failures surface OpenCut extra commands, package hints, and GPU/VRAM notes through structured errors and async job status. RA-01/RA-02 are closed: Ruff targets the Python 3.11 package floor, and `requirements.txt` mirrors the `pyproject.toml` core/standard dependency bounds. |
| Resource contention | The May 26 plan now has N3 closed: GPU semaphore contention waits up to 30 seconds by default and returns retry metadata when the wait budget is exhausted. |
| Webhook discoverability | The May 26 plan now has N6 closed: `/webhooks/event-types`, `/api/webhooks/event-types`, and `/mcp/info` expose webhook event names, canonical replacements, schema pointers, and legacy aliases. |
| Webhook trust | The May 26 plan now has E11 closed: new HTTP webhook registrations require an HMAC secret unless the caller explicitly opts into unsigned local testing. |
| Disk preflight | The May 26 plan now has N4 closed: heavyweight async jobs run output-volume disk checks before job creation and return 507 with required/free/output-dir metadata when space is insufficient. |
| Interrupted-job resume | The May 26 plan now has N5 closed: checkpointable async jobs persist resume metadata and can be re-enqueued from interrupted history through `POST /jobs/<job_id>/resume`. |
| Plugin job registration | The May 26 plan now has N7 closed: plugin manifests can declare background jobs, plugin routes can enqueue namespaced jobs through the core async-job tracker, and plugin uninstall now moves through quarantine/restore/permanent-delete states. |
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
