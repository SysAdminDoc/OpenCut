# OpenCut Research Report

Root synthesis of current research and planning inputs. Detailed research plans
are archived under [docs/archive/research](docs/archive/research/).

Last consolidated: 2026-06-04. Research-driven additions refreshed: 2026-06-03;
freshness refresh: 2026-06-04.

2026-06-04 freshness refresh: the N8 third-party skill loader, E14 CEP
caption display-settings parity work, N9 enriched job metadata, N10 request-ID
subprocess propagation, E12 manifest-derived workflow allowlist, and E13 CLI
route escape hatch are now represented as shipped in the live v4.126 docs; E15
also has its fourth through twenty-ninth rolling i18n batches recorded there. No new duplicate
extensibility/accessibility/observability/workflow/scripting rows were promoted.
Focused
verification passed for the N8 skill tests, E14 CEP/UXP caption display-setting
UI gates, N9 job metadata gates, N10 request-correlation subprocess gates, and
E12 workflow/route-manifest gates, the E13 CLI route tests, and the E15 i18n
migration/drift gates; the route manifest now reports 1,523 routes / 107
blueprints, and `py -3.12 scripts/sync_version.py --check` kept v1.32.0 in
sync. Current
external anchors still support the existing backlog shape: Adobe documents UXP
as the Premiere v25.6+ extensibility path (`https://developer.adobe.com/premiere-pro/uxp/`),
Adobe's UXP API guidance warns that newer UXP APIs fail on older host versions
(`https://developer.adobe.com/premiere-pro/uxp/resources/fundamentals/apis/`),
Adobe's Firefly AI Assistant announcement reinforces agentic multi-step creative
workflows (`https://news.adobe.com/en/gb/news/2026/04/adobe-new-creative-agent`),
Generative Extend remains a current Premiere feature
(`https://helpx.adobe.com/premiere/desktop/edit-projects/edit-with-generative-ai/generative-extend-overview.html`),
FFmpeg 8.1 is current upstream (`https://ffmpeg.org/`), and active OSS
comparators include MLT v7.38.0 and LosslessCut v3.68.0. The open queue remains
E15 plus external F202/F252 and the RA-01..RA-10 research items below.

## Executive Summary

OpenCut is a local-first automation backend for Adobe Premiere Pro: a Flask app
(1,523 routes / 107 blueprints / ~599 core modules, 8,800+ tests) that exposes
silence/filler removal, transcription and captions, audio cleanup, video
effects, export, review bundles, CLI route scripting, an MCP bridge, and CEP + UXP panels. It is
already extremely broad. The May 26 performance/recovery research pass
(N1-N10, E11, E12, E13, E14) is now shipped through v4.100, and E15 is actively
rolling in v4.126; the strongest remaining direction is **not** another wave of
model surfaces but making the existing surface easier to run, debug, resume,
extend, and trust.

This 2026-06-03 pass read the actual persistence, error, dependency, plugin, and
correlation code and scanned the 2026 competitive market. The highest-value
opportunities it surfaced — all net-new versus the open continuation queue:

1. **Surface `request_id` in the JSON error body** (RA-04) — the correlation ID
   is already in the `X-Request-ID` header and job metadata but missing from the
   error envelope users actually copy into bug reports. [Verified]
2. **`PRAGMA user_version` schema versioning** for the two SQLite stores
   (RA-05) — migrations are currently ad-hoc `ALTER TABLE` in try/except; N5
   added resume columns and N9 has now added more metadata. [Verified]
3. **Guard + back up destructive wipes** (RA-06) — `journal.clear_all()`, plugin
   `uninstall`, and the cache-clear route delete user state with no dry-run,
   confirm token, or recoverable backup. [Verified]
4. **Cap persisted job `result_json` size** (RA-07) — `save_job()` serializes the
   full result into SQLite with no ceiling. [Verified]
5. **Job-store/journal compaction + size diagnostic** (RA-08) — cleanup never
   `VACUUM`s and the journal has no retention. [Verified]
6. **Log direct typed errors** (RA-03) — only `safe_error()` logs; typed
   `OpenCutError` / `error_response` returns leave no log line. [Verified]
7. **Align Ruff `target-version` with `requires-python`** (RA-01) — py39 vs
   >=3.11 skew. [Verified]
8. **Reconcile `requirements.txt` floors with `pyproject.toml`** (RA-02) — pin
   drift weakens the advisory story. [Verified]
9. **Timeline-native caption round-trip parity** (RA-09) — the AutoCut
   differentiator OpenCut does not yet match. [Likely]
10. **"Magic clips" long-form-to-shorts macro** (RA-10) — 2026 table-stakes for
    the shorts persona; composable from existing primitives. [Likely]

## Evidence Reviewed

- **Git range:** `git log -30 --oneline`; 39 commits since 2026-05-20 at the
  start of this pass. The N1-N10/E11/E12/E13/E14 continuation queue is now closed
  through v4.100, E15 has v4.101-v4.126 rolling batches, and the earlier checkpoints include `b228e42`, `ae25c96`,
  `ead2a3d`, `40e43cb`, `9c13b9a`, and `58d0781`.
- **Persistence:** `opencut/job_store.py` (SQLite jobs, WAL, no `user_version`,
  unbounded `result_json`, no `VACUUM`), `opencut/journal.py` (rollback ledger,
  bare `ALTER TABLE` migration, `clear_all()` with no backup),
  `opencut/core/transcript_cache.py` (N1 content-addressed cache).
- **Errors/diagnostics:** `opencut/errors.py` (only `safe_error` logs;
  `error_response`/`to_response` omit `request_id` and emit no log),
  `opencut/core/request_correlation.py` (header echo + job stamping exist;
  body field does not).
- **Dependencies:** `pyproject.toml` (`requires-python>=3.11`, ruff `py39`),
  `requirements.txt` (looser/unbounded floors vs extras).
- **Extensibility:** `opencut/core/agent_skills.py` (N8 user loader now scans
  validated `~/.opencut/skills/<id>/` packages), `opencut/routes/plugins.py` (install requires
  restart to load routes; uninstall `shutil.rmtree` with no backup).
- **Security/bind:** `opencut/server.py` loopback default + `OPENCUT_ALLOW_REMOTE`
  gate + F112 token requirement on remote bind — confirmed sound.
- **External sources (2026):** Adobe Firefly AI Assistant launch + Premiere
  Generative Extend ([news.adobe.com](https://news.adobe.com/news/2026/04/adobe-new-creative-agent),
  [helpx.adobe.com generative-extend](https://helpx.adobe.com/premiere/desktop/edit-projects/edit-with-generative-ai/generative-extend-overview.html));
  AutoCut vs Submagic caption comparison ([autocut.com](https://www.autocut.com/en/blogs/AutoCut-vs-Submagic/));
  Submagic 2026 review / Magic Clips ([max-productive.ai](https://max-productive.ai/ai-tools/submagic/)).
- **Unverifiable here:** live coverage floor (F205 runs time out on this VM),
  live Premiere round-trip behavior (no Premiere on this machine), live Apple
  notarization (needs secrets + macOS runner).

## Current Product Map

| Layer | Where | Notes |
|---|---|---|
| Entry points | `opencut.cli:main`, `opencut.server:main`, `opencut.mcp_server:main` | console scripts in `pyproject.toml`; CLI includes a manifest-validated `route` escape hatch. |
| Routes | `opencut/routes/*.py` (~90 blueprints) | captions, audio, editing, delivery, review, plugins, jobs, MCP bridge. |
| Core | `opencut/core/*.py` (~599 modules) | per-feature processing; FFmpeg subprocess heavy. |
| Job platform | `opencut/jobs.py`, `job_store.py`, `workers.py` | `@async_job`, SQLite persistence, priority workers, cancellation, disk preflight, interrupted-job resume. |
| Persistence | `~/.opencut/jobs.db`, `journal.db`, `transcript_cache/` | WAL SQLite + content-addressed cache. |
| Panels | `extension/com.opencut.panel` (CEP), `com.opencut.uxp` (UXP) | UXP WebView cutover blocked on live UDT capture (F252). |
| Persona | solo podcasters / YouTubers on Premiere 2019+ (CEP) / 25.6+ (UXP) | local-first, no cloud, no API keys for core features. |

## Feature Inventory

| Feature area | Access | Code | Maturity | Coverage |
|---|---|---|---|---|
| Silence/filler removal, auto-edit | routes + skills | `core/auto_edit.py`, `auto_montage.py` | mature | tested |
| Transcription + captions | `/captions*`, `/transcript*` | `core/captions.py`, `routes/captions.py` | mature; N1 cache added | strong |
| Caption display settings (FCC) | `/captions/display-settings/*` | `core/caption_display_settings.py` | shipped (F236); surfaced in UXP and CEP | tested |
| Audio cleanup / pro chain | `/audio*` | `core/audio_*` | mature | tested |
| Review bundles + markers | `/review*`, `/collab*` | `core/review*`, `annotations.py` | mature (F225–F229) | tested |
| Shorts A/B variants | route/skill | `core/ab_variant.py`, `best_take.py` | shipped | tested |
| API scripting | `opencut route METHOD PATH` | `opencut/cli.py` | manifest-validated CLI escape hatch (E13) | tested |
| MCP bridge | `/mcp/*`, `opencut-mcp-server` | `mcp_server.py`, `mcp_extended_tools.py` | 39 curated + 1,466 opt-in | tested |
| Plugins | `/plugins/*` | `routes/plugins.py`, `core/plugins.py` | install needs restart; background jobs now use the core async-job tracker; no hot-reload, no backup on uninstall | partial |
| Agent skills | built-in + validated user packages | `core/agent_skills.py` | user loader shipped (N8); no marketplace UI | tested |
| Webhooks | `/webhooks/*` | `core/webhook_system.py` | discovery + signed-by-default (N6/E11) | tested |

## Competitive Landscape

- **Adobe Firefly AI Assistant (public beta, Apr 2026)** — agentic orchestration
  across Premiere/Photoshop/etc. from one natural-language prompt; "first cut
  from raw footage"; Generative Extend. *Lesson:* OpenCut's MCP/agent-skill
  framing is the right shape; the gap is a polished "raw footage → first cut"
  macro. *Avoid:* cloud-only, paid-plan gating, API-key dependence — OpenCut's
  local-first stance is the differentiator, keep it.
- **AutoCut (Premiere extension)** — direct analogue: AutoCaptions, AutoCut
  Silences, AutoZoom, AutoB-Rolls; **timeline-native caption editing** is its
  cited advantage. *Lesson:* RA-09 round-trip parity closes the one place a
  reviewer picks AutoCut. *Avoid:* paid-pack upsell model.
- **Submagic** — shorts-first: animated captions + silence/filler removal +
  contextual B-roll + auto-zoom + Magic Clips (auto-extract shorts). *Lesson:*
  RA-10 (magic clips) is now table-stakes for the shorts persona. *Avoid:*
  online-only, Storyblocks lock-in.
- **Descript** — transcript-first long-form editing + collaboration. *Lesson:*
  OpenCut's local transcript editing already overlaps; keep round-trip fidelity
  high. *Avoid:* heavy account/cloud coupling.

## Quality & Friction Findings

- **Major — no `request_id` in error bodies.** Header-only correlation forces
  users to read response headers to file a useful bug report. → RA-04.
- **Major — ad-hoc SQLite migrations, no `user_version`.** N5 and N9 added
  columns onto a versionless schema; no downgrade
  detection. → RA-05.
- **Major — destructive wipes without guard/backup.** `journal.clear_all()`
  (the rollback ledger), plugin uninstall, cache clear. → RA-06.
- **Minor — typed errors are invisible in logs.** Only `safe_error()` logs;
  typed `OpenCutError`/`error_response` paths emit nothing. → RA-03.
- **Minor — unbounded `result_json`** in `jobs.db`. → RA-07.
- **Minor — no compaction/retention/size diagnostic** for the SQLite stores. →
  RA-08.
- **Cosmetic — Ruff/Python target skew** and **requirements/pyproject pin
  drift.** → RA-01, RA-02.

## Architecture & Technical Findings

- **Module boundaries** are clean: routes → core → FFmpeg subprocess; job
  platform is isolated behind `@async_job`.
- **Persistence** is two WAL SQLite DBs plus a content-addressed cache. The
  weak spot is migration discipline (RA-05) and unbounded growth (RA-07/RA-08).
- **Concurrency** uses thread-local SQLite connections with dead-thread pruning
  and a GPU semaphore with a 30s acquire wait (N3) — solid.
- **Error handling** is a well-structured taxonomy; the gap is the logging +
  request_id seam at the typed-error path (RA-03/RA-04).
- **Dependency health** is actively gated (pip-audit, npm advisory, SBOM, model
  cards). The residual risk is the requirements/pyproject drift (RA-02) and the
  lint target skew (RA-01), not vulnerable pins.
- **Release automation** is extensive (route manifest, OpenAPI, version sync,
  release smoke with 16 chained gates, signing/notarization wiring).

## Security / Privacy / Data Safety

- **Bind model is sound** [Verified]: loopback default, `OPENCUT_ALLOW_REMOTE`
  opt-in, F112 per-install token required before a remote bind is announced.
- **Webhooks are signed-by-default** (E11) and SSRF numeric-IP bypass is blocked
  (`4647e0e`).
- **Data-safety gap** [Verified]: destructive local-state wipes lack the
  dry-run/confirm/backup posture the project applies elsewhere (RA-06). This is
  the top safety item in this pass.

## UX & Accessibility

- Caption display settings follow the FCC token set (F236); both UXP and CEP now
  expose the discoverable display-settings card, and CEP a11y invariant gates
  remain in place.
- The competitive UX gap is timeline-native caption round-trip (RA-09), where a
  reviewer would otherwise choose AutoCut.

## Explicit Non-Goals

- **Cloud rendering / hosted SaaS** — rejected: contradicts the local-first,
  no-upload value proposition that distinguishes OpenCut from Firefly/Submagic.
- **Bundling new heavy diffusion/TTS models this pass** — rejected: the repo is
  already model-saturated; the marginal value is in reliability/observability.
- **Replacing SQLite with a server DB** — rejected: overkill for a single-user
  local backend; the fix is migration discipline (RA-05), not a new datastore.
- **Trusting client-supplied `X-Request-ID`** — already correctly rejected in
  `request_correlation.py` (regenerated to prevent log injection); keep it.

## Open Questions

- **RA-05 after N5/N9:** N5 and N9 shipped with idempotent ad-hoc migrations to
  avoid blocking crash-recovery and observability work, but `PRAGMA
  user_version` remains the right follow-up now that `jobs.db` has more
  additive columns.
- **RA-09 round-trip fidelity** cannot be verified without a live Premiere
  install. [Needs validation] — needs a real export→edit→import loop before
  committing to a styling-metadata schema.
- **Coverage floor** (F205) remains unmeasured on this VM (runs time out); not a
  blocker for these doc-only additions.

## Research Inputs (archived)

- [docs/archive/research/RESEARCH_FEATURE_PLAN_2026-05-25.md](docs/archive/research/RESEARCH_FEATURE_PLAN_2026-05-25.md) — governance, route-surface, agent, UXP, i18n, a11y, CI, supply-chain loop.
- [docs/archive/research/RESEARCH_FEATURE_PLAN_2026-05-26.md](docs/archive/research/RESEARCH_FEATURE_PLAN_2026-05-26.md) — performance, observability, crash-recovery, plugin extensibility, resource-preflight, trust-signals pass (N1–N10/E11–E15).
- [docs/RESEARCH.md](docs/RESEARCH.md) — earlier tracked research summary.
- [ROADMAP.md](ROADMAP.md) — canonical detailed F-number and wave-letter ledger; "Active Continuation Queue (May 26 Plan)" tracks the shipped and remaining continuation items, and the "Research-Driven Additions" section holds this pass's RA-01..RA-10 items.
- [ROADMAP-NEXT.md](ROADMAP-NEXT.md) — older active-wave worksheet.

## Archive Notes

Root `research.md` is ignored by policy and was not tracked at the start of this
pass. It remains untouched as a local artifact. The tracked May 25 and May 26
research plans were moved into `docs/archive/research/`.
