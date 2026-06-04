# OpenCut Research Report

Root synthesis of current research and planning inputs. Detailed research plans
are archived under [docs/archive/research](docs/archive/research/).

Last consolidated: 2026-06-04. Research-driven additions refreshed: 2026-06-04.

2026-06-04 freshness refresh: the N8 third-party skill loader, E14 CEP
caption display-settings parity work, N9 enriched job metadata, N10 request-ID
subprocess propagation, E12 manifest-derived workflow allowlist, and E13 CLI
route escape hatch are now represented as shipped in the live v4.217 docs; E15
also has its fourth through one-hundred-tenth rolling i18n batches recorded there, and `TODO.md`
is now the compact active execution queue. No new duplicate
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
comparators include MLT v7.38.0 and LosslessCut v3.68.0. The compact open queue
in `TODO.md` remains E15 plus external F202/F252 and the RA-01..RA-27 research items below. Cycles 2
through 4 added UXP packaging-trust guardrails from Adobe's current manifest,
filesystem, API-reference, changelog, Hybrid Plugin, external-process, and
WebView docs. Cycle 5 then re-ran the optional-extra Python advisory gate and
found `pyproject[all]` failing on five unwaived Torch/Transformers advisories
while `requirements.txt` remained clean; RA-15 captures the build-lane
follow-up. Cycle 6 re-ran the Adobe Premiere Pro npm package drift check and
found a stable `release-26.2` dist-tag newer than `latest`; RA-16 captures the
F251 release-channel tracker follow-up. Cycle 7 rechecked Adobe UXP manifest
schema and deprecated API notes; RA-17/RA-18 capture the manifest-version and
deprecation-sentinel follow-ups. Cycle 8 checked UXP clipboard permission and
beta alert/confirmation behavior; RA-19/RA-20 capture those runtime permission
follow-ups. Cycle 9 repeated those two UXP findings and promoted no new rows.
Cycle 10 checked the advertised Python 3.13 classifier against committed CI
coverage; RA-21 captures the release-metadata proof/retraction follow-up. Cycle
11 checked Release Full CEP panel Node runtime determinism; RA-22 captures the
Node pin follow-up. Cycle 12 checked GitHub Actions immutable-reference posture;
RA-23 captures the workflow action SHA-pin follow-up. Cycle 13 checked Release
Full `GITHUB_TOKEN` least-privilege posture; RA-24 captures the job-permission
scoping follow-up. Cycle 14 checked Docker dependency install-surface drift;
RA-25 captures the container dependency guard follow-up. Cycle 15 checked Docker
runtime docs, non-root volume paths, and WebSocket port posture; RA-26 captures
the runtime parity follow-up. Cycle 16 checked Docker GPU compose launch
commands against tracked files and profile comments; RA-27 captures the missing
compose-file command drift follow-up.

## Executive Summary

OpenCut is a local-first automation backend for Adobe Premiere Pro: a Flask app
(1,523 routes / 107 blueprints / ~599 core modules, 8,800+ tests) that exposes
silence/filler removal, transcription and captions, audio cleanup, video
effects, export, review bundles, CLI route scripting, an MCP bridge, and CEP + UXP panels. It is
already extremely broad. The May 26 performance/recovery research pass
(N1-N10, E11, E12, E13, E14) is now shipped through v4.100, and E15 is actively
rolling in v4.217; the strongest remaining direction is **not** another wave of
model surfaces but making the existing surface easier to run, debug, resume,
extend, and trust.

This 2026-06-03 pass read the actual persistence, error, dependency, plugin, and
correlation code and scanned the 2026 competitive market. The highest-value
opportunities it surfaced — all net-new versus the open continuation queue:

1. **Burn down new `pyproject[all]` Torch/Transformers advisory failures**
   (RA-15) — the optional-extra `pip-audit` gate now fails on five unwaived
   advisories even though `requirements.txt` is clean. [Verified]
2. **Track Adobe release-channel dist-tags in F251** (RA-16) — Adobe now
   publishes a stable `release-26.2` dist-tag newer than `latest`, but the
   tracker policy only treats `latest` and `beta` as first-class inputs.
   [Verified]
3. **Surface `request_id` in the JSON error body** (RA-04) — the correlation ID
   is already in the `X-Request-ID` header and job metadata but missing from the
   error envelope users actually copy into bug reports. [Verified]
4. **`PRAGMA user_version` schema versioning** for the two SQLite stores
   (RA-05) — migrations are currently ad-hoc `ALTER TABLE` in try/except; N5
   added resume columns and N9 has now added more metadata. [Verified]
5. **Guard + back up destructive wipes** (RA-06) — `journal.clear_all()`, plugin
   `uninstall`, and the cache-clear route delete user state with no dry-run,
   confirm token, or recoverable backup. [Verified]
6. **Cap persisted job `result_json` size** (RA-07) — `save_job()` serializes the
   full result into SQLite with no ceiling. [Verified]
7. **Job-store/journal compaction + size diagnostic** (RA-08) — cleanup never
   `VACUUM`s and the journal has no retention. [Verified]
8. **Log direct typed errors** (RA-03) — only `safe_error()` logs; typed
   `OpenCutError` / `error_response` returns leave no log line. [Verified]
9. **Align Ruff `target-version` with `requires-python`** (RA-01) — py39 vs
   >=3.11 skew. [Verified]
10. **Reconcile `requirements.txt` floors with `pyproject.toml`** (RA-02) — pin
    drift weakens the advisory story. [Verified]
11. **Timeline-native caption round-trip parity** (RA-09) — the AutoCut
    differentiator OpenCut does not yet match. [Likely]
12. **"Magic clips" long-form-to-shorts macro** (RA-10) — 2026 table-stakes for
    the shorts persona; composable from existing primitives. [Likely]
13. **Minimize UXP filesystem permissions** (RA-11) — the base UXP manifest still
    asks for `fullAccess` even though picker-based `request` access should cover
    normal import/export browsing. [Verified]
14. **Add F253 Hybrid Plugin package validation** (RA-12) — Adobe now documents
    Premiere 26.2 `.uxpaddon` support; OpenCut should validate package layout and
    platform binaries before treating native add-ons as distribution-ready.
    [Verified]
15. **Declare and harden UXP external-launch permissions** (RA-13) — the UXP
    OAuth browser handoff uses `shell.openExternal` without a manifest
    `launchProcess` scheme allowlist, consent-dialog developer text, or denial
    result handling. [Verified]
16. **Split UXP WebView dev and release bridge permissions** (RA-14) — the
    dormant Bolt/WebView scaffold currently requires `localAndRemote` message
    bridging and dev localhost domains even though release cutover should use
    packaged local WebView content. [Verified]
17. **Add UXP manifest schema drift guard and explicit `manifestVersion`**
    (RA-17) — the live UXP manifest omits Adobe's required manifestVersion
    while the dormant WebView scaffold declares v6 despite Premiere docs
    naming v5 support. [Verified]
18. **Add a UXP API deprecation sentinel before F252 cutover** (RA-18) — current
    source avoids deprecated Clipboard and legacy `uxpvideo*` APIs, but the
    cutover path needs a static regression guard. [Verified]
19. **Declare UXP clipboard permission and centralize copy fallback** (RA-19) —
    current UXP clipboard calls use the supported API, but the live and
    generated manifests omit the required clipboard permission. [Verified]
20. **Replace or explicitly gate UXP `window.confirm` usage** (RA-20) — Adobe
    moved `alert`, `prompt`, and `confirm` back behind the beta `enableAlerts`
    feature flag, but OpenCut still has a raw UXP `window.confirm` call.
    [Verified]
21. **Prove or retract the advertised Python 3.13 classifier** (RA-21) —
    `pyproject.toml` advertises Python 3.13 while the committed build,
    PR-fast, and Adobe-version workflows still run only 3.12. [Verified]
22. **Pin Node explicitly in Release Full panel gates** (RA-22) — PR Fast pins
    Node 22 before panel npm install/test work, while Release Full runs the
    panel npm gate without any `setup-node` step. [Verified]
23. **Pin GitHub Actions workflow actions to full-length SHAs** (RA-23) —
    release/signing workflows still reference `actions/*` by mutable tags even
    though GitHub documents full-length commit SHA pins as the immutable Action
    release option. [Verified]
24. **Scope Release Full `GITHUB_TOKEN` permissions by job** (RA-24) — Release
    Full grants workflow-wide `contents: write`, while PR Fast and the Adobe
    tracker use narrower permissions. Only release upload paths need write
    access. [Verified]
25. **Align Docker dependency installs with tracked Python install surfaces**
    (RA-25) — the Dockerfile still installs retired `deep-translator` and
    `pydub` packages even though the audited source install surfaces removed
    them. [Verified]
26. **Align Docker runtime docs, volume home, and WebSocket exposure** (RA-26)
    — Docker quick-start examples still reference the old root home path, while
    compose omits the documented/exposed WebSocket port and the bridge defaults
    to a container-local bind. [Verified]
27. **Fix Docker GPU compose launch command drift** (RA-27) — the README GPU
    launch command references an untracked `docker-compose.gpu.yml` file even
    though the tracked compose file already exposes GPU mode through the `gpu`
    profile. [Verified]

## Evidence Reviewed

- **Git range:** `git log -30 --oneline`; 39 commits since 2026-05-20 at the
  start of this pass. The N1-N10/E11/E12/E13/E14 continuation queue is now closed
  through v4.100, E15 has v4.101-v4.192 rolling batches, and the earlier checkpoints include `b228e42`, `ae25c96`,
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
  `requirements.txt` (looser/unbounded floors vs extras), and
  `py -3.12 -m opencut.tools.pip_audit_extras --json` (2026-06-04:
  `requirements.txt` clean; `pyproject[all]` failed with unwaived
  `torch==2.8.0` PYSEC-2025-203/204/206/PYSEC-2026-139 and
  `transformers==4.57.6` PYSEC-2025-217).
- **Python support metadata:** `pyproject.toml` advertises
  `Programming Language :: Python :: 3.13`, while
  `.github/workflows/build.yml`, `.github/workflows/pr-fast.yml`, and
  `.github/workflows/adobe-premierepro-versions.yml` still set
  `python-version` to 3.12. GitHub's current Python Actions guide includes
  3.13 in matrix examples, and Python's current 3.13 documentation confirms
  the active release line.
- **Node CI runtime:** `.github/workflows/pr-fast.yml` uses
  `actions/setup-node@v4` with `node-version: '22'` before panel `npm ci`, while
  `.github/workflows/build.yml` runs the Release Full panel `npm ci`,
  `npm run audit:check`, `npm test`, `npm run build:verify`, and
  `npm run build` steps without a prior `setup-node` step. The CEP panel lockfile
  records Vitest's supported Node engines as `^20.0.0 || ^22.0.0 || >=24.0.0`;
  GitHub's current Node Actions docs recommend `setup-node` for consistent
  behavior across runners and Node versions, and runner images update weekly.
- **Extensibility:** `opencut/core/agent_skills.py` (N8 user loader now scans
  validated `~/.opencut/skills/<id>/` packages), `opencut/routes/plugins.py` (install requires
  restart to load routes; uninstall `shutil.rmtree` with no backup).
- **UXP packaging:** `extension/com.opencut.uxp/manifest.json`
  (`localFileSystem: "fullAccess"`), `extension/com.opencut.uxp/main.js`
  picker helpers and `shell.openExternal` OAuth handoff,
  `docs/UXP_MIGRATION.md` F253 residual plan, and Adobe's UXP
  manifest/filesystem/external-process/Hybrid Plugin docs (`https://developer.adobe.com/premiere-pro/uxp/plugins/concepts/manifest/`,
  `https://developer.adobe.com/premiere-pro/uxp/resources/recipes/filesystem-operations/`,
  `https://developer.adobe.com/premiere-pro/uxp/resources/recipes/external-process/`,
  `https://developer.adobe.com/premiere-pro/uxp/changelog/`,
  `https://developer.adobe.com/premiere-pro/uxp/plugins/hybrid-plugins/build/`).
  Cycle 8 additionally checked UXP Clipboard docs and the UXP API changelog
  against `navigator.clipboard.writeText(...)` and `window.confirm(...)` usage.
- **Security/bind:** `opencut/server.py` loopback default + `OPENCUT_ALLOW_REMOTE`
  gate + F112 token requirement on remote bind — confirmed sound.
- **External sources (2026):** Adobe Firefly AI Assistant launch + Premiere
  Generative Extend ([news.adobe.com](https://news.adobe.com/news/2026/04/adobe-new-creative-agent),
  [helpx.adobe.com generative-extend](https://helpx.adobe.com/premiere/desktop/edit-projects/edit-with-generative-ai/generative-extend-overview.html));
  AutoCut vs Submagic caption comparison ([autocut.com](https://www.autocut.com/en/blogs/AutoCut-vs-Submagic/));
  Submagic 2026 review / Magic Clips ([max-productive.ai](https://max-productive.ai/ai-tools/submagic/));
  Adobe UXP manifest/filesystem guidance and Premiere 26.2 Hybrid Plugin docs
  (links in UXP packaging bullet above).
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
| UXP packaging trust | manifest + release gates | `extension/com.opencut.uxp/manifest.json`, `bolt-webview/uxp.config.ts`, `docs/UXP_MIGRATION.md` | new RA-11/RA-12/RA-13 guardrail work | planned |

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
- **Major — optional `[all]` dependency audit now fails.** `requirements.txt`
  is clean, but `pyproject[all]` resolves vulnerable Torch/Transformers
  versions with five unwaived advisories. → RA-15.
- **Release-signal gap — Adobe stable release-channel tags are not first-class.**
  F251 currently tracks only `latest` and `beta`, but the live npm registry now
  exposes `release-26.2: 26.2.1` newer than `latest: 26.2.0`. → RA-16.
- **Packaging trust — UXP manifest schema version is implicit or inconsistent.**
  The live manifest omits `manifestVersion`, while the WebView scaffold emits
  version 6 despite current Premiere docs naming version 5 support. → RA-17.
- **Packaging trust — deprecated UXP APIs have no sentinel.** Current UXP source
  does not use deprecated Clipboard APIs or legacy `uxpvideo*` events, but
  F252/WebView work can regress without a static guard. → RA-18.
- **Packaging trust — base UXP requests broad filesystem access.** Adobe
  recommends `request` for user-selected files and warns users may deny
  `fullAccess`; OpenCut should reserve broad/direct path access for justified
  hybrid or direct-path cases. → RA-11.
- **Packaging trust — F253 lacks a pre-native package validator.** Hybrid
  `.uxpaddon` work now has official Premiere 26.2 documentation; CI should catch
  broken addon folder/architecture layouts before distribution. → RA-12.
- **Packaging trust — UXP external launches lack a declared permission/consent
  guard.** Adobe's Premiere UXP docs require `launchProcess.schemes` for
  `shell.openExternal`, clear developer text in the consent dialog, and denial
  handling; the OAuth handoff currently relies on a bare `openExternal` call. →
  RA-13.

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
  cards). The residual risk is now the requirements/pyproject drift (RA-02), the
  lint target skew (RA-01), the optional `[all]` advisory failure (RA-15), and
  Adobe package release-channel drift (RA-16).
- **Release automation** is extensive (route manifest, OpenAPI, version sync,
  release smoke with 16 chained gates, signing/notarization wiring).
- **UXP distribution posture** should split base-package trust from optional
  Hybrid Plugin power: the base panel can likely use picker-scoped filesystem
  access and explicit external-launch permissions, while F253 native add-ons need
  a stricter package validator and host version/capability gate.

## Security / Privacy / Data Safety

- **Bind model is sound** [Verified]: loopback default, `OPENCUT_ALLOW_REMOTE`
  opt-in, F112 per-install token required before a remote bind is announced.
- **Webhooks are signed-by-default** (E11) and SSRF numeric-IP bypass is blocked
  (`4647e0e`).
- **Data-safety gap** [Verified]: destructive local-state wipes lack the
  dry-run/confirm/backup posture the project applies elsewhere (RA-06). This is
  the top safety item in this pass.
- **Install-trust gap** [Verified]: broad UXP `fullAccess` is avoidable if
  picker-scoped access covers current import/export browse flows (RA-11).
- **Consent-trust gap** [Verified]: OAuth browser launches should be constrained
  to declared URL schemes and expose a graceful denial/manual-copy path (RA-13).
- **Supply-chain gap** [Verified]: optional `pyproject[all]` installs now fail
  the Python advisory gate on unwaived Torch/Transformers issues, so release
  smoke needs an upgrade/split/waiver decision before the next all-extra build
  claim (RA-15).

## UX & Accessibility

- Caption display settings follow the FCC token set (F236); both UXP and CEP now
  expose the discoverable display-settings card, and CEP a11y invariant gates
  remain in place.
- The competitive UX gap is timeline-native caption round-trip (RA-09), where a
  reviewer would otherwise choose AutoCut.
- The distribution UX gap is trust at install time: a least-privilege UXP
  manifest and validated optional Hybrid Plugin package make the post-CEP path
  easier to accept in Adobe Marketplace or enterprise installs (RA-11/RA-12).
- The external-launch UX gap is consent clarity: OAuth handoff should explain why
  the browser opens and give editors a manual path when they deny the UXP consent
  dialog (RA-13).

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
- **RA-11 live permission behavior** needs a UDT smoke after switching to
  `request`, because picker token persistence and project-folder export behavior
  must be validated in Premiere rather than inferred from docs alone.
- **RA-13 OAuth launch behavior** needs UDT capture for both consent approval and
  denial, because Adobe documents user consent as part of the `openExternal`
  contract and static tests can only prove manifest/source alignment.
- **RA-15 resolver choice** needs a build-lane decision: upgrade Torch-compatible
  extras to 2.9+ where possible, split heavy GPU/depth extras out of `[all]`, or
  document narrowly scoped no-fix waivers with tests and `docs/PYTHON_ADVISORIES.md`.
- **RA-16 release-channel policy** needs the F251 tracker to decide how many
  Adobe `release-*` streams to retain in generated snapshots when npm publishes
  multiple stable branches at once.
- **RA-17 manifest-version policy** needs a documented UDT/Premiere smoke before
  any package claims a schema version newer than the Premiere-supported v5
  listed in current docs.
- **RA-19 clipboard permission** needs the base and generated UXP manifests to
  declare the narrow supported permission if current clipboard-copy behavior is
  retained.
- **RA-20 confirmation behavior** needs either an in-panel confirmation modal or
  an explicit beta-alert manifest decision backed by UDT evidence.
- **RA-21 Python 3.13 support proof** needs either a CI matrix lane covering
  dependency install plus fast manifest/smoke checks under 3.13, or removal of
  the classifier until that lane passes.
- **RA-22 Release Full Node determinism** needs a Release Full `setup-node`
  step before panel npm commands and a workflow-shape test that keeps PR Fast
  and Release Full on the same supported Node major.
- **RA-23 GitHub Actions SHA pins** needs full-length commit SHA workflow action
  references, version/update comments, and a workflow-security test that rejects
  mutable action tags or branches.
- **RA-24 Release Full token permissions** needs read-only default permissions
  for build/test/package jobs, narrow `contents: write` only around release
  uploads, and a workflow guard test against broad workflow-level write scope.
- **RA-25 Docker dependency surface** needs Docker installs to consume the
  canonical requirements/extras or a guarded minimal list that excludes retired
  `deep-translator` and `pydub` names.
- **RA-26 Docker runtime parity** needs a deliberate HTTP-only or
  WebSocket-capable Docker stance, aligned `/home/opencut/.opencut` volume
  examples, compose/exposed port consistency, and a container config drift test.
- **RA-27 Docker GPU compose command** needs README and compose comments to
  agree on one tracked GPU launch command, plus a docs/config guard against
  missing compose-file references.

## Research Inputs (archived)

- [docs/archive/research/RESEARCH_FEATURE_PLAN_2026-05-25.md](docs/archive/research/RESEARCH_FEATURE_PLAN_2026-05-25.md) — governance, route-surface, agent, UXP, i18n, a11y, CI, supply-chain loop.
- [docs/archive/research/RESEARCH_FEATURE_PLAN_2026-05-26.md](docs/archive/research/RESEARCH_FEATURE_PLAN_2026-05-26.md) — performance, observability, crash-recovery, plugin extensibility, resource-preflight, trust-signals pass (N1–N10/E11–E15).
- [docs/archive/research/RESEARCH_FEATURE_PLAN_2026-06-04_CYCLE7.md](docs/archive/research/RESEARCH_FEATURE_PLAN_2026-06-04_CYCLE7.md) — UXP manifest schema drift and deprecated API sentinel follow-up (RA-17/RA-18).
- [docs/archive/research/RESEARCH_FEATURE_PLAN_2026-06-04_CYCLE8.md](docs/archive/research/RESEARCH_FEATURE_PLAN_2026-06-04_CYCLE8.md) — UXP clipboard permission and beta alert/confirmation follow-ups (RA-19/RA-20).
- [docs/archive/research/RESEARCH_FEATURE_PLAN_2026-06-04_CYCLE9.md](docs/archive/research/RESEARCH_FEATURE_PLAN_2026-06-04_CYCLE9.md) — duplicate UXP clipboard/confirmation recheck; no new RA row promoted.
- [docs/archive/research/RESEARCH_FEATURE_PLAN_2026-06-04_CYCLE10.md](docs/archive/research/RESEARCH_FEATURE_PLAN_2026-06-04_CYCLE10.md) — Python 3.13 classifier-vs-CI support proof follow-up (RA-21).
- [docs/archive/research/RESEARCH_FEATURE_PLAN_2026-06-04_CYCLE11.md](docs/archive/research/RESEARCH_FEATURE_PLAN_2026-06-04_CYCLE11.md) — Release Full CEP panel Node runtime pin follow-up (RA-22).
- [docs/archive/research/RESEARCH_FEATURE_PLAN_2026-06-04_CYCLE12.md](docs/archive/research/RESEARCH_FEATURE_PLAN_2026-06-04_CYCLE12.md) — GitHub Actions full-SHA pinning follow-up (RA-23).
- [docs/archive/research/RESEARCH_FEATURE_PLAN_2026-06-04_CYCLE13.md](docs/archive/research/RESEARCH_FEATURE_PLAN_2026-06-04_CYCLE13.md) — Release Full `GITHUB_TOKEN` job-permission scoping follow-up (RA-24).
- [docs/archive/research/RESEARCH_FEATURE_PLAN_2026-06-04_CYCLE14.md](docs/archive/research/RESEARCH_FEATURE_PLAN_2026-06-04_CYCLE14.md) — Docker dependency install-surface drift follow-up (RA-25).
- [docs/archive/research/RESEARCH_FEATURE_PLAN_2026-06-04_CYCLE15.md](docs/archive/research/RESEARCH_FEATURE_PLAN_2026-06-04_CYCLE15.md) — Docker runtime docs, volume-home, and WebSocket exposure follow-up (RA-26).
- [docs/archive/research/RESEARCH_FEATURE_PLAN_2026-06-04_CYCLE16.md](docs/archive/research/RESEARCH_FEATURE_PLAN_2026-06-04_CYCLE16.md) — Docker GPU compose launch command follow-up (RA-27).
- [docs/RESEARCH.md](docs/RESEARCH.md) — earlier tracked research summary.
- [ROADMAP.md](ROADMAP.md) — canonical detailed F-number and wave-letter ledger; "Active Continuation Queue (May 26 Plan)" tracks the shipped and remaining continuation items, and the "Research-Driven Additions" section holds this pass's RA-01..RA-27 items.
- [ROADMAP-NEXT.md](ROADMAP-NEXT.md) — older active-wave worksheet.

## Archive Notes

Root `research.md` is ignored by policy and was not tracked at the start of this
pass. It remains untouched as a local artifact. The tracked May 25 and May 26
research plans were moved into `docs/archive/research/`.
