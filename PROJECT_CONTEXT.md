# OpenCut — Project Context

**Canonical, cross-tool source of truth for project memory, architecture, shipping cadence, and entry points.**
**Last consolidated:** 2026-06-05 (237 autonomous research/verification/implementation/wrap-up passes, with Passes 1-34 on 2026-05-17 — see `.ai/research/2026-05-17/`). Pass 3 verified the live state, walked `host/index.jsx`, drafted the F143-F145 agent-conductor RFC, and quantified the market-fit story. Pass 4 ran the full release-smoke gate, fixed release-gate lint drift, and prepared the local research + hardening commit. Passes 5-75 are recorded in ROADMAP.md and the pass update notes below. Pass 76 closed F220-F222 by adding external RVC backend execution/fallback handling, natural-language color-intent grading on `/ai/auto-grade`, cut-point pacing analysis on `/ai/pacing-analysis`, and route/catalogue tests. Pass 77 closed F224 by adding `/ai/deepfake-detect`, deepfake evidence metadata, and authenticity-report review guidance. Pass 78 closed F228-F230 by adding review-bundle voice-note attachments and optional HLS browser-scrubbing renditions. Pass 79 closed F232 by adding optional Headscale/Tailscale command-plan descriptors for cross-site review portal sharing. Pass 80 closed F235 by adding WCAG 3 draft descriptive transcript and extended AD timing hooks. Pass 81 closed F245-F248 by adding deterministic Netflix IMF/Dolby Vision, DPP IMF, Dolby Vision Profile 5/8.1, and ADM BW64 delivery-standard planning routes. Pass 82 closed F205 by completing the CI-style coverage run and raising Release Full to `--cov-fail-under=54`. Pass 83 advanced F252.3 with strict UDT result-capture validation for the future WebView manifest switch. Pass 84 hardened async-job persistence and cleanup by redacting persisted request secrets, terminating timed-out child processes, preserving timing samples during executor shutdown, and tightening model-delete input validation. Pass 85 hardened standalone Web UI uploads with streamed multipart writes, an explicit local upload-size boundary, partial-file cleanup, and structured JSON/base64 validation errors. Pass 86 fixed webhook job-event compatibility so dotted async-job events match both new dotted subscriptions and legacy `job_complete` / `job_failed` subscriptions. Pass 87 closed N1 from the May 26 continuation queue with persistent SHA-256 transcript caching, cache stats/clear routes, and generated surface refreshes; route manifest now reports 1,519 routes / 107 blueprints and the opt-in extended MCP catalogue reports 1,463 tools. Pass 88 closed N2 with a shared optional-dependency install-hint registry that feeds `missing_dependency()`, generic structured errors, and async job status suggestions. Pass 89 closed N3 by changing GPU semaphore contention from instant default rejection to a bounded 30-second wait with retry metadata. Pass 90 closed N6 by exposing webhook event-type discovery through `/webhooks/event-types`, `/api/webhooks/event-types`, and `/mcp/info`. Pass 91 closed E11 by requiring signed webhook registration by default with explicit unsigned opt-in and warning-file creation. Pass 92 closed N4 by adding decorator-level disk preflight for heavyweight async routes. Pass 93 closed N5 by adding resumability metadata, durable running-job persistence, and `POST /jobs/<job_id>/resume` for interrupted checkpointable jobs. Pass 94 closed N7 by adding manifest-declared plugin jobs, the `plugin_job(...)` decorator, and the bundled `long-job-demo` example. Pass 95 closed N8 by loading validated user agent skills from `~/.opencut/skills/<id>/`, tagging catalogue entries by source, and rejecting user plans that reference routes absent from the generated route manifest. Pass 96 closed E14 by porting the F236 caption display-settings card into the CEP Captions tab with token loading and preview parity against UXP. Pass 97 closed N9 by persisting peak job resource metadata and explicit exit reasons, adding a live-or-persisted `GET /jobs/<job_id>` detail route, and refreshing generated route/MCP surfaces. Pass 98 closed N10 by restoring request IDs inside async worker threads and tagging FFmpeg subprocess environments/logged stderr with `OPENCUT_REQUEST_ID`. Pass 99 closed E12 by deriving workflow validation from route-manifest `workflow.label` metadata, adding explicit workflow-safe route opt-ins, and tightening route-manifest drift checks for metadata changes. Pass 100 closed E13 by adding a manifest-validated CLI route command with JSON/query request shaping and automatic CSRF handling. Pass 101 advanced E15 with an eight-string CEP i18n install/status feedback batch. Pass 102 advanced E15 with a nine-string OAuth/live-updates feedback batch. Pass 103 advanced E15 with a ten-string clip/workflow input-prompt batch. Pass 104 advanced E15 with a nine-string media/caption chain-prompt batch. Pass 105 advanced E15 with a ten-string Whisper/settings status-feedback batch. Pass 106 advanced E15 with a ten-string import/export result-feedback batch. Pass 107 advanced E15 with a ten-string session replay/journal apply feedback batch. Pass 108 advanced E15 with a seven-string settings/live utility feedback batch. Pass 109 advanced E15 with a nine-string path/journal revert feedback batch. Pass 110 advanced E15 with a ten-string transcript/preview/polish feedback batch. Pass 111 advanced E15 with a seven-string polish finish/journal/cut-review feedback batch. Pass 112 advanced E15 with a ten-string preset management feedback batch. Pass 113 advanced E15 with a seven-string preset file import/export feedback batch. Pass 114 advanced E15 with a ten-site project template feedback/label batch. Pass 115 advanced E15 with a 19-key transcript editor/summarize feedback batch. Pass 116 advanced E15 with a nine-key caption display-settings feedback batch. Pass 117 advanced E15 with a 24-key model/GPU/queue feedback batch. Pass 118 advanced E15 with a 14-key recent-output browser/batch-picker feedback batch. Pass 119 advanced E15 with a 17-key custom workflow builder feedback batch. Pass 120 advanced E15 with an 11-key workflow preset loading/run/completion feedback batch. Pass 121 advanced E15 with a 23-key workflow preset summary/status feedback batch. Pass 122 advanced E15 with a 14-key batch-run status/progress feedback batch. Pass 123 advanced E15 with a 15-key batch summary/default-status feedback batch. Pass 124 advanced E15 with a 15-key workspace shell status feedback batch. Pass 125 advanced E15 with a 15-key connection/project-media scan feedback batch. Pass 126 advanced E15 with a 49-key workspace header/stage metadata batch. Pass 127 advanced E15 with a 10-key central job lifecycle feedback batch. Pass 128 advanced E15 with a 13-key central cleaned error-guidance batch. Pass 129 advanced E15 with a 15-key install/action helper feedback batch. Pass 130 advanced E15 with an 18-key helper/status microcopy batch. Pass 131 advanced E15 with a 13-key shortcut/numeric validation batch. Pass 132 advanced E15 with a 10-key status/language fallback batch. Pass 133 advanced E15 with a 16-key onboarding wizard batch. Pass 134 advanced E15 with a 12-key Wave H share/report feedback batch. Pass 135 advanced E15 with a 25-key live-updates bridge status batch. Pass 136 advanced E15 with a 13-key engine availability batch. Pass 137 advanced E15 with a 25-key engine summary/preference batch. Pass 138 advanced E15 with a 25-key settings backend/speech availability batch. Pass 139 advanced E15 with a 13-key Whisper installed/ready availability batch. Pass 140 advanced E15 with a 14-key Settings GPU/restart and Whisper CPU-toggle batch. Pass 141 advanced E15 with a 17-key Search empty/result availability batch. Pass 142 advanced E15 with a 54-key Search indexing/progress batch. Pass 143 advanced E15 with a 20-key Settings diagnostics/dependency feedback batch. Pass 144 advanced E15 with a 37-key Command Palette shell feedback batch. Pass 145 advanced E15 with a 32-key Favorites/Recent/Preview shell feedback batch. Pass 146 advanced E15 with a 9-key optional-engine install feedback batch. Pass 147 advanced E15 with a 6-key Merge/Trim utility feedback batch. Pass 148 advanced E15 with a 7-key LLM/defaults utility feedback batch. Pass 149 advanced E15 with an 8-key Timeline write-back feedback batch. Pass 150 advanced E15 with an 8-key Timeline marker/project-item feedback batch. Pass 151 advanced E15 with a 10-key Timeline rename/smart-bin feedback batch. Pass 152 advanced E15 with a 10-key Captions chapter/SRT feedback batch. Pass 153 advanced E15 with a 7-key Audio loudness feedback batch. Pass 154 advanced E15 with an 11-key Deliverables sequence-summary feedback batch. Pass 155 advanced E15 with a 17-key Deliverables load/generate feedback batch. Pass 156 advanced E15 with a 10-key Deliverables label/summary feedback batch. Pass 157 advanced E15 with a 12-key Timeline multicam/repeat feedback batch. Pass 158 advanced E15 with a 17-key Custom Workflow summary/status feedback batch. Pass 159 advanced E15 with a 13-key Activity/job-history feedback batch. Pass 160 advanced E15 with a 4-key Media/Waveform feedback batch. Pass 161 advanced E15 with a 10-key NLP/OTIO feedback batch. Pass 162 advanced E15 with a 5-key clipboard/generated-output feedback batch. Pass 163 advanced E15 with a 10-key journal summary/revert-control batch. Pass 164 advanced E15 with a 15-key journal loading/status/clear batch. Pass 165 advanced E15 with an 18-key interview-polish result/control batch. Pass 166 advanced E15 with an 11-key preflight modal feedback batch and shared Cancel reuse. Pass 167 advanced E15 with a 12-key assistant/preview/cancel feedback batch and shared Apply reuse. Pass 168 refreshed optional-extra advisory research and recorded RA-15 for the Torch/Transformers build-lane decision. Pass 169 advanced E15 with an 11-key status/media-preview/quick-workflow feedback batch. Pass 170 advanced E15 with a 15-key central job result-summary batch and added TODO.md as the compact active execution queue. Pass 171 advanced E15 with a 9-key helper/status tone and loading feedback batch. Pass 172 advanced E15 with a 9-key journal/media metadata label batch. Pass 173 advanced E15 with a 20-key notification/history status label batch. Pass 174 advanced E15 with a 15-key output-browser/review-summary batch. Pass 175 advanced E15 with an 8-key LLM settings status batch. Pass 176 advanced E15 with a 4-key workflow/processing-estimate status batch. Pass 177 advanced E15 with a 12-key smart-bin rule editor batch. Pass 178 advanced E15 with a 2-key engine grammar batch. Pass 179 advanced E15 with an 8-key shared grammar batch. Pass 180 advanced E15 with a 4-key fallback-label batch. Pass 181 advanced E15 with a 5-key preview/provider/marker fallback batch. Pass 182 advanced E15 with a 6-key journal action-label batch. Pass 183 advanced E15 with a 6-key workflow step-group batch. Pass 184 advanced E15 with a 10-key Command Palette tool-name batch. Pass 185 advanced E15 with a second 10-key Command Palette tool-name batch. Pass 186 advanced E15 with a third 10-key Command Palette video/export/caption tool-name batch. Pass 187 advanced E15 with a fourth 10-key Command Palette workflow/search tool-name batch. Pass 188 advanced E15 with the final 2-key Command Palette tool-name batch. Pass 189 advanced E15 with a 2-key Settings preset-selector placeholder batch. Pass 190 advanced E15 with a 3-key generic API request-error batch. Pass 191 advanced E15 with a 2-key CEP bridge/output fallback batch. Pass 192 advanced E15 by tightening the existing update-available toast fallback guard. Pass 193 consolidated Cycle 6 research by adding RA-16 for Adobe `release-*` dist-tag tracking in F251. Pass 194 advanced E15 with an 18-key first-viewport static shell HTML migration. Pass 195 consolidated Cycle 7 research by adding RA-17/RA-18 for UXP manifest schema and deprecation sentinels. Pass 196 advanced E15 by making DOM i18n apply translated title, placeholder, and aria-label attributes at runtime. Pass 197 advanced E15 with a 24-key workspace-stage/media-source static HTML migration, consolidated Cycle 8 RA-19/RA-20 for UXP clipboard permission and beta alert/confirmation handling, and recorded Cycle 9 as a duplicate no-new-row recheck. Pass 198 advanced E15 with a 24-key Cut quick-action/interview/assistant static HTML migration. Pass 199 consolidated Cycle 10 research by adding RA-21 for Python 3.13 classifier-vs-CI proof. Pass 200 advanced E15 with a 2-key main-navigation title/ARIA static HTML migration. Pass 201 advanced E15 with a 4-key media action title/ARIA static HTML migration. Pass 202 advanced E15 with a 19-key Silence form static HTML migration. Pass 203 consolidated Cycle 11 research by adding RA-22 for Release Full CEP panel Node runtime pinning. Pass 204 advanced E15 with a 12-key Fillers form static HTML migration. Pass 205 consolidated Cycle 12 research by adding RA-23 for GitHub Actions full-SHA workflow action pinning. Pass 206 advanced E15 with a 5-key Full Pipeline preset static HTML migration. Pass 207 advanced E15 with a 9-key Trim form static HTML migration. Pass 208 consolidated Cycle 13 research by adding RA-24 for Release Full GITHUB_TOKEN job-permission scoping. Pass 209 advanced E15 with a 9-key Auto-Edit form static HTML migration. Pass 210 advanced E15 with an 8-key Highlights form static HTML migration. Pass 211 consolidated Cycle 14 research by adding RA-25 for Docker dependency install-surface drift around retired `deep-translator` and `pydub` packages. Pass 212 advanced E15 with a 5-key Captions quick-action static HTML migration. Pass 213 advanced E15 with a 12-key FCC Caption Display Settings static HTML migration. Pass 214 consolidated Cycle 15 research by adding RA-26 for Docker runtime docs, non-root volume paths, and WebSocket port posture. Pass 215 advanced E15 with a 19-key Styled Captions model/language static HTML migration. Pass 216 consolidated Cycle 16 research by adding RA-27 for Docker GPU compose launch command drift. Pass 217 advanced E15 with a 23-key Styled Captions style selector static HTML migration plus DOM/data-lint support for translatable label attributes. Pass 218 advanced E15 with a 14-key Styled Captions preview/action-word/install-hint static HTML migration. Pass 219 advanced E15 with a 4-key Subtitle Export form static HTML migration that reused existing model/language keys. Pass 220 consolidated Cycle 17 research by adding RA-28 for README non-badge generated-count drift. Pass 221 advanced E15 with a 23-key Transcript editor static HTML migration. Pass 222 advanced E15 with a 13-key Caption Translation form static HTML migration. Pass 223 advanced E15 with a 4-key Karaoke Captions form static HTML migration. Pass 224 advanced E15 with an 8-key Caption Burn-in form static HTML migration. Pass 225 advanced E15 with a 7-key Animated Captions form static HTML migration. Pass 226 consolidated Cycle 19 research by adding RA-29 for Docker dependency install fail-closed/specifier guardrails. Pass 227 advanced E15 with a 3-key Repeat Take Detection form static HTML migration. Pass 228 consolidated Cycle 20 research by adding RA-30 for Docker build-context secret/log hygiene. Pass 229 advanced E15 with a 4-key YouTube Chapters form static HTML migration. Pass 230 advanced E15 with a 23-key Audio quick-action and stem-separation static HTML migration. Pass 231 advanced E15 with a 5-key Audio Denoise static HTML migration. Pass 232 advanced E15 with a 14-key Audio Studio FX and DeepFilterNet static HTML migration. Pass 233 advanced E15 with an 11-key Audio Effects static HTML migration. Pass 234 advanced E15 with an 18-key Audio TTS static HTML migration. Pass 235 advanced E15 with a 17-key Audio SFX and Tone static HTML migration. Pass 236 advanced E15 with a 3-key Audio Ducking static HTML migration. Pass 237 consolidated the June 5 comprehensive research archive into RA-31 through RA-36 covering Adobe tracker exit-code capture, tracker label contracts, no-`gh` label dry-runs, lockfile advisory coverage, SBOM fidelity labeling, and UNC/HGFS-safe CEP panel Node commands.
**Live version:** v1.32.0.

> This file is the place to land first. It is intentionally **smaller** than `CLAUDE.md` and `ROADMAP.md` and **does not duplicate** their granular content. It tells you what each other file is for and where to look next.

---

## 1. Identity (one paragraph)

OpenCut is a **local-first, MIT-licensed automation backend for Adobe Premiere Pro**, with a DaVinci Resolve scripting bridge and an MCP server sidecar. The backend is a Python/Flask server bound to `127.0.0.1:5679` (HTTP) + `:5680` (WebSocket) + `:5681` (MCP JSON-RPC). Two Premiere panels ship with it: a **CEP** panel (`com.opencut.panel`, ~15,263-line `main.js` as of 2026-05-25) for Premiere 2019–25.5, and a **UXP** panel (`com.opencut.uxp`, ~5,568-line ES module as of 2026-05-25) for Premiere 25.6+. Both panels talk to the same backend. No subscriptions, no cloud, no API keys required for core features. (Panel-size drift is enforced by `scripts/check_doc_sizes.py`.)

---

## 2. Numbers you should trust (live as of 2026-06-04)

| Surface | Count | Source of truth |
|---|---|---|
| API routes | **1,523** | `opencut/_generated/route_manifest.json` (F099) |
| Blueprints | **107** | same |
| Core processing modules (`opencut/core/`) | **599** Python files | `ls opencut/core` |
| Route files (`opencut/routes/`) | **105** (excluding `__init__.py`) | `ls opencut/routes` |
| Tests | **221 test_*.py files** + **2 Vitest panel test files** (8,800+ tests claimed) | `ls tests/`, `extension/com.opencut.panel/tests/` |
| CI coverage floor | **54%** | `.github/workflows/build.yml` + `.ai/research/2026-05-17/F205_COVERAGE_FLOOR_SUCCESS.md` (F205) |
| Optional AI/model cards | **47** | `opencut/_generated/model_cards.json` + `docs/MODELS.md` (F115) |
| `/api/*` routes | **234** total; **17** true aliases; **217** canonical `/api` routes | `opencut/_generated/api_aliases.json` (F199) |
| Feature readiness records | **108** total; **66** route-derived records / **90** route bindings | `opencut/registry.py` + `opencut/_generated/feature_readiness.json` + `opencut.catalog_contract` (F100/F191/F196/F197) |
| OpenAPI typed response endpoints | **110** | `opencut.openapi_registry` + `opencut/openapi.py` (F192/F193) |
| MCP curated tools | **39** | `opencut/mcp_server.py` (F195) |
| MCP extended route tools | **1,466 opt-in** | `opencut/_generated/mcp_extended_tools.json` (F194) |
| CEP JSX host functions | **18 total; 2 CEP-only** | `opencut/_generated/cep_uxp_parity.json` (F198) |
| CEP locale keys (English) | **1,933** | `extension/com.opencut.panel/client/locales/en.json` |
| Current version | **1.32.0** | `pyproject.toml`, `python scripts/sync_version.py --check` |

The README route badge is regenerated from the route manifest. **The manifest is the source of truth.** Never quote a hand-edited number in CI or docs that bypasses the manifest.

---

## 3. The two planning paradigms in this repo

Two parallel planning ledgers coexist. They are not contradictory — they describe different things:

| Ledger | What it is | Owner doc | Status entry |
|---|---|---|---|
| **Wave letters** (A → T) | AI-model feature *surfaces*. Each wave bundles 5–20 new core modules + routes + checks under a wave-letter blueprint (`wave_a_routes.py` … `wave_l_routes.py`). | `ROADMAP-NEXT.md` + `CHANGELOG.md` + wave sections of `ROADMAP.md` | A-M shipped; N-T planned and F180-tiered. |
| **F-numbers** (F001 → F272) | Governance, infrastructure, security, release-trust, packaging, accessibility, and standards items. Started by the v4.3 audit on 2026-05-16. | `ROADMAP.md` v4.3+ tier table | F001-F120 shipped or actively in flight; F121-F190 came from Pass 1, F191-F260 from Pass 2, and F261-F272 from Pass 3 of the 2026-05-17 research run. |

Hold both in your head. When a new piece of work arrives, ask: "is this a model surface (wave letter) or a governance/infra item (F-number)?" Most agent-driven work will be one or the other; chat-conductor work (F143-F145) is on the F-number side because it's a governance/UX item more than a model integration. For Waves N-T, use `.ai/research/2026-05-18/WAVE_N_T_F_NUMBER_LEDGER.md` (F180) to decide whether a row reuses an existing F-number or stays wave-only.

---

## 4. Architecture in one diagram

```
Premiere CEP panel ─┐
Premiere UXP panel ─┼─► HTTP localhost:5679  ─► Flask app (create_app() factory)
DaVinci Resolve ────┤   WebSocket :5680           ├─ routes/* (107 blueprints) ─► core/* (599 modules) ─► FFmpeg / Whisper / Demucs / Torch / ONNX
MCP client     ─────┘   MCP HTTP :5681            ├─ jobs.py (@async_job decorator, SQLite persistence, GPU rate limit)
                                                  ├─ security.py (CSRF, path-traversal, SSRF guards, safe_pip_install)
                                                  ├─ auth.py (loopback bypass + 256-bit token for non-loopback)
                                                  ├─ registry.py + _generated/feature_readiness.json (feature readiness states for UI gating)
                                                  ├─ model_cards.py + opencut/_generated/* (license/model truth)
                                                  └─ checks.py (118 public check_* probes; 86 check_*_available gates)
```

Key invariants:
- **`@async_job("type")` is the only sanctioned way** to write long-running routes. It handles thread spawn, job_id correlation, cancellation, SQLite persistence (`~/.opencut/jobs.db`), structured error mapping, and rate-limit acquire/release pairing.
- **All blueprints** are registered through a deterministic ordered tuple in `routes/__init__.py`. A runtime route-collision guard refuses startup on duplicate `(method, path)` pairs.
- **User data** lives at `~/.opencut/` — jobs.db, footage_index.db, transcript_cache, social_credentials.json, llm_settings.json, brand_kit.json, onboarding.json, plugin/lock files, model caches, auth.json (F112). All access goes through `user_data.py` wrappers with per-file locks and atomic `os.replace` writes.
- **Plugin manifest v1** (F116) validates `plugin.json` + `plugin.lock.json` hash, capability allowlist, and `OPENCUT_PLUGIN_ALLOW_UNSIGNED` opt-in before mounting any blueprint.

For module-level patterns and the deep gotcha list (~270 entries), see **[`CLAUDE.md`](CLAUDE.md)** — it is the authoritative developer + agent reference.

---

## 5. Documentation map (what to read for what)

| You want to know… | Read this |
|---|---|
| What features ship today, with examples | `README.md` |
| Module-level patterns, every async-job rule, every safe_bool / UXP / CEP convention | `CLAUDE.md` |
| Compact active execution queue | `TODO.md` |
| What's planned, in what tier, with sources | `ROADMAP.md` (v4.3 sections — F001-F120 + Wave 1-7 + Wave N-T) |
| Shipped roadmap summary | `COMPLETED.md` + `ROADMAP-COMPLETED.md` |
| What shipped in each release | `CHANGELOG.md` (v1.0 → v1.32.0) |
| Current research synthesis | `RESEARCH_REPORT.md` |
| Wave-letter detail (Apr 2026 plan) | `ROADMAP-NEXT.md` |
| Per-dependency / per-model upgrade ledger | `MODERNIZATION.md` + `docs/MODELS.md` (auto-generated, F115) |
| Threat model + responsible disclosure | `SECURITY.md` |
| Dev setup | `DEVELOPMENT.md` + `CONTRIBUTING.md` |
| Plugin authoring | `docs/PLUGIN_AUTHORING.md` |
| Skill authoring | `docs/SKILL_AUTHORING.md` |
| UXP migration plan | `docs/UXP_MIGRATION.md` |
| macOS notarization release path | `docs/MACOS_NOTARIZATION.md` |
| Windows ARM64 packaging | `docs/WINDOWS_ARM64_PACKAGING.md` (F101) |
| Linux Flatpak/AppImage distribution | `docs/LINUX_DISTRIBUTION.md` (F249) |
| Optional Aptabase telemetry | `docs/TELEMETRY.md` (F250) |
| Node advisories disposition | `docs/NODE_ADVISORIES.md` (F095) |
| 2026-04 competitive analysis | `AUDIT.md` (v1.11) + `docs/RESEARCH.md` — both predate ROADMAP v4.3 |
| 2026-05-25/26 feature research plans | `docs/archive/research/` |
| 402-feature aspirational catalogue | `features.md` — *aspirational; not a ship promise* |
| 2026-05-17 research run | `.ai/research/2026-05-17/` (20 research artefacts + implementation handoff updates) |
| 2026-05-18 F180 governance bridge | `.ai/research/2026-05-18/WAVE_N_T_F_NUMBER_LEDGER.md` (Wave N-T rows mapped to F-number disposition) |
| Codex-era handoff snapshot | `CODEX-CHANGELOG.md` (already-merged work, kept for historical reference) |
| Future-Claude session handoff | `CLAUDE-HANDOFF-PROMPT.md` |
| LAN review portal contract | `docs/REVIEW_PORTAL.md` |
| Review notification feeds and signed webhooks | `docs/REVIEW_NOTIFICATIONS.md` |
| Audio-description draft/review workflow | `docs/AUDIO_DESCRIPTIONS.md` |
| IMF, Dolby Vision, DPP, and ADM delivery-standard planning | `docs/DELIVERY_STANDARDS.md` |

When two roadmap files disagree, **ROADMAP.md v4.3 wins**. When ROADMAP.md and the live code disagree, **the code wins** (and ROADMAP.md drifted — open an F-number).

---

## 6. Where to look for each kind of question

| Question | Look here |
|---|---|
| Why is route X failing with 503? | `opencut/checks.py` → `check_*` probe; `opencut/registry.py` + `_generated/feature_readiness.json` for readiness state; `docs/MODELS.md` for install hint. |
| How do I write an async route? | `CLAUDE.md` → `@async_job` decorator section. |
| What's the SQLite job store schema? | `opencut/job_store.py` (~200 lines). |
| How do I add a new optional AI extra? | `opencut/checks.py` (add `check_X_available()`), `opencut/model_cards.py` (add card), `opencut/registry.py` (add `FeatureRecord`), then a route in the appropriate wave file. |
| What's planned for the next release? | `ROADMAP.md` → Now / Next tier tables, plus `RESEARCH_REPORT.md` for the current synthesis; `gh issue list` once F182 (issue seeder run) is executed. |
| What shipped in the last release? | `CHANGELOG.md`. |
| Why was X rejected? | `ROADMAP.md` → Rejected tier (F043, F078-F086) + this file § 9 for newly explicit rejects. |
| How does the CEP panel call ExtendScript? | `extension/com.opencut.panel/host/index.jsx` + `PremiereBridge` abstraction in `main.js`. |
| How does the UXP panel access Premiere? | `extension/com.opencut.uxp/main.js` → `PProBridge` class; `premierepro` UXP module lazy-imported. |
| What does the MCP server expose? | `opencut/mcp_server.py` (~1,160 lines, 39 curated tools). |
| What's the threat model? | `SECURITY.md` + `opencut/auth.py` + `opencut/security.py`. |
| How are version surfaces synced? | `scripts/sync_version.py --check` (covers 19 surfaces). |
| What does the release smoke matrix check? | `scripts/release_smoke.py` (F098) chains bootstrap, version-sync, route/api/feature/model generated-manifest drift checks, roadmap lint, the F241 text-shaping gate, ruff, focused pytest, pip-audit, npm advisory allow-list, and panel-source verification. |
| Why is the README route count different from the manifest? | The README marketing badge has drifted. The manifest is the truth (F099). |

---

## 7. Hard constraints (do not violate)

1. **License posture**: MIT repo. Optional models/codecs/plugins each need a model card with an explicit licence. CC-BY-NC, research-only, and non-commercial licences are tracked but not shipped enabled by default. AGPL / GPL is rejected for code reuse (reference patterns OK).
2. **Network posture**: core editing must work entirely offline. Cloud APIs may be optional connectors only, never mandatory for core editing. Telemetry off by default.
3. **Runtime**: Python ≥3.11 for source installs. The F121/F133/F135 security dependency refresh forced the floor above Python 3.9 because Pillow 12.2 and WhisperX 3.8.5 require ≥3.10 and onnxruntime 1.25+ requires ≥3.11.
4. **One new pip dep per feature, max.** Prefer extending existing deps (FFmpeg, OpenCV, Pillow, transformers).
5. **Graceful degradation**: every optional dependency gated by a `check_X_available()` function returning 503 `MISSING_DEPENDENCY` with an install hint.
6. **Security**: CSRF on all POST/DELETE; loopback bypass + 256-bit auth token for non-loopback (F112); plugin sandbox (F116) refuses unsigned plugins unless explicitly enabled; path validation rejects `..`, null bytes, UNC/network paths (including post-realpath); SSRF protection on outbound URLs.
7. **C2PA provenance** on generated/exported media (F110 → upgrading to 2.3 in F140).

---

## 8. Pass 4 hardening + release-gate cleanup (2026-05-17)

The seven-file security hardening batch that Pass 1 found dirty was validated in Pass 4 and included in the local checkpoint commit. Pass 4 also applied Ruff's safe unused-import / import-order fixes to the release-smoke lint scope so `python scripts/release_smoke.py --json` is green. Branch was **25 commits ahead** of `origin/main` before this checkpoint; pushing to `SysAdminDoc/OpenCut` still depends on local GitHub auth.

| File | What | Why |
|---|---|---|
| `opencut/auth.py` | `ipaddress.is_loopback()` replaces literal `{127.0.0.1, ::1}` set; strips IPv6 zone + brackets | Closes `127.0.0.2` bypass of F112 auth gate |
| `opencut/security.py` | Rejects realpath starting with `\\` or `//` | Closes symlink-to-UNC bypass |
| `opencut/helpers.py` | `_run_ffmpeg_with_progress` re-architected with `finally` that always unregisters job process + joins stderr drain; returns `-1` sentinel on double-timeout | Fixes process-registry leak + `None.returncode` foot-gun |
| `opencut/user_data.py` | `write_user_file` mkdirs nested parent + works around `mkstemp` Windows path-sep refusal | Future-proofs nested user-data |
| `opencut/routes/captions.py` | `force` flag uses `safe_bool` | Catches v1.9.22 audit miss |
| `opencut/routes/system.py` | `include_jobs` uses `safe_bool` | Same |
| `opencut/routes/timeline.py` | `include_media` uses `safe_bool` | Same |

**Validation:** targeted hardening slice passed (`119 passed`), release-smoke passed end-to-end (`232 passed` in pytest-fast, route/model/version/license gates clean, pip-audit clean, npm advisory allow-list clean).

---

## 9. Shipping cadence — Now / Next / Later (2026-05-17)

Highlights only. Full ledger in `ROADMAP.md` + `.ai/research/2026-05-17/PRIORITIZATION_MATRIX.md`.

**Now (v1.33 — v1.34, ~3–4 weeks):**
- F138 commit dirty hardening batch
- [x] F121 Pillow 12.2 (CVE-2026-40192 / 25990)
- [x] F122 flask-cors 6.x (5 CVEs)
- F123 pydub replacement / audioop-lts shim (Python 3.13)
- F126 OpenTimelineIO-Plugins migration (AAF adapter)
- [x] F127a Python runtime floor decision (resolved to Python 3.11+ because onnxruntime 1.25+ requires it)
- F128 FFmpeg filter regression suite
- [x] F130 OpenCV 4.13 wheel refresh
- [x] F133 onnxruntime ≥1.25
- F139 caption translation endpoint (NLLB-200, low-effort high-value)
- F140 C2PA 2.3 sidecar bump
- [x] F135 whisperx 3.8.5 bump
- F149 fill K3.5 AI Slate ID (Florence-2 already installed)
- F162 SAM 2 → SAM 3.1
- F163 Depth Anything V2 → Depth Anything 3
- F167 OmniVoice fill of Wave H2.4
- F176-F178 eval dataset bundle, model cards sweep, eval harness v2
- F181-F185 bootstrap/cleanup fixes
- [x] F191 generated route/check readiness registry
- [x] F197 registry-owned `NON_AI_CHECKS` allowlist
- [x] F199 generated `/api` alias policy manifest
- [x] F202 macOS notarization release wiring (repository-side)
- [x] F204 release SBOM attachment
- [x] F207 bundled FFmpeg version installer manifest
- [x] F208 OpenAPI spec validity gate
- [x] F209 MCP tool/route consistency gate
- [x] F218 blueprint import-order stability gate
- [x] F219 SBOM completeness gate
- [x] F236 FCC caption display-settings token gate
- [x] F237 EBU R 128 v5.0 / ITU BS.1770-5 loudness registry correction
- [x] F261 cross-platform source launchers (`OpenCut-Server.command` + `OpenCut-Server.sh`)
- [x] F262 UXP sample-repo URL fix
- [x] F270 README "$1,400/year" positioning lead
- [x] F264 CI npm-audit machine-parseable assertion
- [x] F266 two-function CEP residual + drop-QE documentation
- [x] F259 UXP HTTP-on-macOS workaround documentation + auto-HTTPS sidecar plan
- [x] F251 `@adobe/premierepro` weekly npm registry tracker (CI + drift JSON)
- [x] F147 `opencut-mcp-server` upstream registry manifest + install doc
- [x] F131 esbuild ≥0.25 resolved-tree CI assertion (`npm run audit:esbuild`)
- [x] F137 `mcp` SDK pin `>=1.26,<2` (block pre-alpha 2.x rewrite)
- [x] F139 `POST /captions/translate` SRT-in / SRT-out round-trip path
- [x] F126 OTIO AAF adapter pin (`otio-aaf-adapter>=2.0,<3` with `opentimelineio>=0.17,<1` in `[otio]` and `[all]`)
- [x] F181 bootstrap fallback when `sys.executable` is a broken UV trampoline
- [x] F185 features.md aspirational-catalogue banner + precedence rule
- [x] F140 C2PA 2.3 alignment (manifest spec version, action vocabulary, cloud-trust-list slot)
- [x] F123 audioop / pydub Python 3.13 compat (audioop_shim module + pydub pin retirement)
- [x] F128 FFmpeg filter regression suite (24 required filters + 13 graph parses + silencedetect/loudnorm sanity)
- [x] F184 docs/ROADMAP.md + ROADMAP-COMPLETED.md collapsed to pointer stubs; `roadmap-mirror` release-smoke gate
- [x] F178 Eval harness v2 (VRAM peak, reference score, backend telemetry, `compare-backends` aggregator + route)
- [x] F177 Model card 2026-Q2 sweep gates (category floor, license/privacy/hardware prefix allowlists, feature_id uniqueness)
- [x] F176 Public eval-dataset catalogue (13 datasets, 6 modalities, opt-in download gate + `commercial_use_ok` flag, 2 routes)
- [x] F176 follow-up: opt-in download runner with dry-run default, triple-gate safety, and `file://` test fixture
- [x] F200 Windows installer policy doc + lockstep tests (WPF recommended, Inno deprecated-but-supported, milestone-gated retirement)
- [x] F211 Cross-platform launcher smoke tests (5 entry points: .bat, 2x .vbs, .command, .sh)
- [x] F217 UXP BackendClient HTTP-shape contract test (JS-side static gates + server-side runtime gates)
- [x] F223 RTL/CJK/bidi caption Unicode validation gate (SRT, ASS, and burn-in ASS text preservation)
- [x] F242 ICU4X/UAX14-compatible CJK caption line breaking (SRT/VTT, overlays, shot-aware wrapping)

**Next (v1.35 — v1.42, ~6 months):**
- **F143–F145** `/agent/chat` conductor + post-turn self-review + Skills SDK + MCP packaging (flagship)
- **F146** UXP-native MCP transport (survives Sept 2026 CEP EOL)
- **F158** StreamDiffusionV2 real-time preview (biggest UX leap)
- **F127b** Transformers v5 implementation + cascades (F124 basicsr replace, F125 audiocraft isolate, F134 pyannote 4, F136 scenedetect 0.7)
- **F148–F156** DaVinci 21 / Descript parity fills (face age, IntelliScript, CineFocus, eye contact, Overdub, trailer, VidMuse, OpusClip variants)
- **F164–F174** model upgrades (LTX-2.3, Wan 2.7, daVinci-MagiHuman, IndexTTS2, VoxCPM2, etc.)
- **F129, F132, F141, F142** infra (FFmpeg 8.1, Vite 8, IMSC 1.3, OCIO 2.5/ACES 2.0)
- **F160a** WebView UI UXP spike

**Later:**
- F157 Motion Brush, F160b WebView impl, F173 Mimi codec, F175 MagiCompiler, F179 features.md sweep, F184 docs/ROADMAP mirror resolution.

**Newly explicit rejects (in addition to ROADMAP.md v4.3 rejects):**
- Mistral Voxtral TTS (CC-BY-NC)
- MatAnyone 2 in production (NTU S-Lab 1.0 non-commercial; research eval only)
- HunyuanVideo / 1.5 default-on (Tencent territory carve-outs)

**Adobe gap reports (file, not implement):**
- F186-F190 — `createCaptionTrack`, `createSubsequence`, `exportAsFinalCutProXML/exportAsProject`, `startDrag`, QE-DOM replacements.

---

## 9.4 Live verification of the governance gates (Pass 3, 2026-05-17)

Pass 3 executed the F-numbered governance gates against the live repo:

| Check | Result |
|---|---|
| `python -m opencut.tools.dump_route_manifest --check` | ✅ PASS — 1,361 routes / 101 blueprints, no drift |
| `python scripts/sync_version.py --check` | ✅ PASS — 19 surfaces at v1.32.0 |
| `python scripts/bootstrap_check.py` | ✅ PASS — all 6 sub-checks (Python 3.12.10, 25-dep auditable lock, server import) |
| `python -m pip_audit -r requirements-lock.txt` | ✅ PASS — "No known vulnerabilities found" |
| `npm audit` in `extension/com.opencut.panel` | ✅ EXPECTED — 1 moderate Vite path-traversal matches the F095 waiver |
| Cross-platform launchers (Wave I I1.4) | **PASS after Pass 5** — `OpenCut-Server.command` (macOS) and `OpenCut-Server.sh` (Linux) now exist and mirror the Windows launcher's bundled Python, FFmpeg, and model-cache environment handling. Pass 3 originally identified this as the F261 gap. |

**Side finding from Pass 3 deep walk:** OpenCut's CEP-only JSX surface is **2 of 18 functions (~11%)**, not the 5 features Pass 2's UXP subagent listed. Only `ocAddNativeCaptionTrack` (no UXP `createCaptionTrack()`) and `ocQeReflect` (QE DOM CEP-only) lack UXP equivalents in `@adobe/premierepro@26.3.0-beta.67`. F252 (UXP migration) revised from XL to L; F253 (Hybrid Plugin) revised from XL to L if scoped to caption-track + drag-out only.

**Pass 4 release-smoke result:** `python scripts/release_smoke.py --json` now exits `0`. It covers bootstrap, version sync, route manifest, model cards, license gate, roadmap lint, Ruff (`E,F,I`), pytest-fast (`232 passed`), pip-audit, npm advisory allow-list, and panel-source verification. Ruff initially failed on unused imports/import ordering; Pass 4 applied safe Ruff fixes in `opencut/` and `scripts/`.

---

## 9.5 Two regulatory deadlines on the *Now* tier (Pass 2)

Pass 2 surfaced two deadlines that force scope changes regardless of feature backlog:

1. **F202 — Apple notarisation mandatory for Homebrew Cask 2026-09-01.** Pass 10 added Developer ID signing + `xcrun notarytool` release wiring and `docs/MACOS_NOTARIZATION.md`. **Remaining operational action:** configure the GitHub secrets and validate the first tagged macOS release against Apple's service.
2. **F236 — FCC "readily-accessible" caption display-settings rule effective 2026-08-17.** Repository-side token schema, preview route, and burn-in integration are now closed locally in Pass 17. Remaining product work is UI persistence/discoverability polish rather than the core token contract.

Both deadlines fall before the next Wave (~v1.35) is expected to ship. F202's repository-side tooling is now in place; F236's core token contract is also in place, with live acceptance still dependent on downstream app/UI packaging choices.

---

## 10. The biggest non-obvious gaps

These deserve explicit attention because they are not currently owned by any single document:

1. **Chat-conductor agent (F143)** — Descript Underlord and FireRed-OpenStoryline have proven sidebar-chat + timeline-diff + post-turn self-review is the converging UX. OpenCut has every building block (1,523 routes, MCP sidecar, LLM abstraction) but no conductor. Highest-leverage gap.
2. **UXP MCP transport (F146)** — every competing PPro MCP server today is CEP-bound and will break Sept 2026. The first UXP-MCP wins post-EOL.
3. **Real-time editor-loop preview (F158)** — StreamDiffusionV2 + Diffusion Templates (Apr 2026, MIT) unlock real-time on existing LTX-2.3 / Wan backends. CapCut / Runway / Captions charge for this.
4. **Caption translation standalone (F139)** — every commercial editor ships it; OpenCut has full dubbing but no SRT-in-SRT-out path.
5. **C2PA 2.3 + IMSC 1.3 (F140 + F141)** — Adobe and the broadcast industry are moving here; OpenCut's local-first provenance + accessibility story is otherwise strong.
6. **The CEP→UXP 15% parity gap** — workflow builder, full settings panel, plugin UI. Adobe CEP EOL is ~Sept 2026 (≈4 months away).
7. **The audiocraft `torch==2.1.0` pin** is the single biggest blocker on torch upgrades. Cascades to Transformers v5, pyannote 4.x, scenedetect 0.7, and 20+ model surfaces that increasingly require torch ≥2.6.
8. **`features.md` (402 features) ↔ F-number reconciliation** is overdue (F179). Currently ~250 of the 402 items live in implicit limbo. Pass 2 sample-walk (40 entries) found ~60% SHIPPED, ~27% UNCLEAR → 5 new F-numbers graduated (F220-F224).
9. **OpenTimelineIO Marker schema is now the F105 review-bundle interchange anchor** (Pass 48, F225; extended in Passes 49-54 with F226/F227/F229/F231/F233/F234, Pass 78 with F228/F230, and Pass 79 with F232). Review bundles include `markers.otio`, deterministic SVG drawing overlays, `annotations/index.json`, `review_threads.json`, `premiere_markers.csv`, `review_markers.edl`, `voice_notes/index.json`, optional copied voice-note audio, and optional `hls/master.m3u8` browser-scrubbing renditions while preserving `markers.json`; LAN share links use HMAC-signed review portal URLs with Caddy/mDNS descriptors and optional Headscale/Tailscale command-plan descriptors, review activity emits Atom feed entries plus optional HMAC-signed webhooks, and delivery transfer bundles provide croc/rclone handoff plans.
10. **WebView UI in Bolt UXP (March 2026, MIT) is the correct CEP→UXP migration target for the ~15,263-line vanilla JS main.js** (Pass 2, F252). Rewriting to Spectrum widgets is months of work for negligible end-user benefit. Pass 59 added the dormant F252.1 Bolt/WebView scaffold at `extension/com.opencut.uxp/bolt-webview/`; Pass 60 added `PProBridge.executeHostAction()` plus `window.OpenCutUXPHost` for the 14 direct-UXP actions; Pass 61 closed F254 by adding `createSubsequenceFromRange()` for UXP range exports; Pass 62 closed F255 with `exportSubsequenceWithEncoder()`; Pass 63 closed F256 with Transcript API helpers for caption-QC context; Pass 64 closed F257 with Object Mask state helpers; Pass 65 closed F258 with UXP AAF export helpers; Pass 66 closed F260 with generated migration-risk dashboard artifacts in Settings; Pass 67 closed F267 with generated UDT smoke-harness artifacts plus a bundled `window.OpenCutUXPUdtHarness` runner for the 14 direct-UXP actions; Pass 83 added the F252.3 result-capture validator for that harness. Live WebView cutover, captured in-Premiere UDT results that pass strict validation, and UI migration remain open. Bolt UXP 1.3 also ships a `public-hybrid/` template for the C++ `.uxpaddon` path (F253) that covers the 5 truly CEP-blocked features (file drag-out, QE DOM, FCPXML/OTIO **import**, `createCaptionTrack`, `exportAsProject` sub-selection save).
11. **The route-readiness and MCP surfaces are better but still not complete** (Pass 43, updated Pass 97): F191 now adds `opencut/_generated/feature_readiness.json` with **66** route-derived readiness records across **90** direct route/check bindings, F196 adds registry-primary model-card/check cross-validation, `/system/feature-state` exposes **108** records, F192/F193 raise legacy `/openapi.json` typed response schemas from **30** to **110** dataclass-discovered route entries, F195 raises the curated MCP tool surface from **27** to **39** tools, F194 adds **1,466** opt-in generated extended MCP route tools with **101** response-schema annotations, F208 pins `/openapi.json` route coverage + path-parameter validity, and F209 pins every curated MCP tool/special action route against the live Flask app. Remaining visibility work is route-level coverage, not model-card/check drift.
12. **`/agent/chat` conductor design is converged** (Pass 3, [AGENT_UX_RFC.md](.ai/research/2026-05-17/AGENT_UX_RFC.md)) — Copilot Workspace editable-plan + Cursor checkpoint+rollback + Underlord post-turn self-review + Aider snapshot-discipline + Claude Code Skills format. **Adopt**, don't invent. F143 (L) + F144 (S) + F145 (M) = ~6-8 weeks at 1 maintainer, ships v1.36 inside the F252 UXP shell. Three patterns deliberately **NOT** copied: Cursor's "accept all" button (render-cost dominates attention), Aider's auto-commit-before-preview (user must see the render first), Claude Code's atomic multi-file apply (even Claude users file per-hunk-accept requests).
13. **OpenCut has a quantified market-positioning story** (Pass 3, [MARKET_POSITIONING.md](.ai/research/2026-05-17/MARKET_POSITIONING.md); README lead shipped in Pass 5) — replaces ~**$1,400/yr** of competitor subscriptions: ~$720/yr (AutoCut + AutoPod + Submagic bundle) + ~$288/yr (Descript Creator) + ~$299-699/yr (Topaz Video AI new subscription, perpetual killed Oct 3 2025). **Mister Horse Animation Composer ~900k installs proves free-shell + paid-packs is the scale-without-VC model for the Premiere ecosystem.** Three categories to deprioritise (weak WTP): avatar generation, OpusClip-virality-as-pillar, sports-highlights-as-headline.

---

## 11. How to contribute / onboard a new agent

1. Read this file.
2. Read `CLAUDE.md` (skim the *Gotchas* section in particular).
3. Skim `ROADMAP.md` § Tier Plan and the v4.3 Phase 2 ledger.
4. `gh issue list` (after F182 ships) for seeded starter work.
5. Use `@async_job` for any new long-running route.
6. New optional AI extra? Always pair: `check_X_available()` in `checks.py` → `FeatureRecord` in `registry.py` → model card in `model_cards.py` → route in the right wave file.
7. Run `python scripts/release_smoke.py --json` before opening a PR.
8. Keep CEP and UXP additions in lockstep (or document why one ships first).

---

## 12. Where this consolidation lives

This file is the **canonical project context**. It is intentionally small. The supporting artefacts from the 2026-05-17 research runs and implementation passes live in **`.ai/research/2026-05-17/`**. The F180 Wave N-T governance bridge lives in **`.ai/research/2026-05-18/`**:

**Pass 41 update (no standalone research file):**
- F192 OpenAPI response-schema expansion lives in `opencut/schemas.py`, `opencut/openapi.py`, and `tests/test_openapi_contract.py`; F193 later moved the surface to dataclass discovery and raised it to 110 typed endpoints

**Pass 71 update (no standalone research file):**
- F193 OpenAPI schema introspection lives in `opencut/openapi_registry.py`, `opencut/openapi.py`, registered schema/core dataclasses, generated `opencut/_generated/mcp_extended_tools.json`, and `tests/test_openapi_contract.py` / `tests/test_mcp_extended_tools.py`

**Pass 72 update (no standalone research file):**
- F196 registry catalogue enforcement lives in `opencut/catalog_contract.py`, expanded curated rows in `opencut/registry.py`, `tests/test_catalog_contract.py`, and the `pytest-fast` release-smoke test list

**Pass 73 update (no standalone research file):**
- F206 CI split lives in `.github/workflows/pr-fast.yml`, the renamed `.github/workflows/build.yml` Release Full workflow, `tests/test_ci_workflow_split.py`, and the `pytest-fast` release-smoke test list

**Pass 74 update (no standalone research file):**
- F210 CEP/UXP utility coverage lives in `extension/com.opencut.panel/client/panel-utils.js`, `extension/com.opencut.panel/tests/`, `extension/com.opencut.panel/vitest.config.mjs`, `extension/com.opencut.uxp/uxp-utils.js`, `tests/test_panel_vitest_gate.py`, and the release-smoke `panel-unit` step

**Pass 75 update (no standalone research file):**
- F212 WPF installer coverage lives in `installer/src/OpenCut.Installer/CommandLine/`, `installer/tests/OpenCut.Installer.Tests/`, `scripts/test_wpf_installer.ps1`, `scripts/smoke_wpf_installer.ps1`, `.github/workflows/build.yml`, and `tests/test_wpf_installer_test_suite.py`

**Pass 76 update (no standalone research file):**
- F220-F222 AI feature reconciliation lives in `opencut/core/voice_conversion.py`, `opencut/core/auto_color.py`, `opencut/core/pacing_analysis.py`, `opencut/routes/ai_content_routes.py`, generated `opencut/_generated/route_manifest.json`, and focused AI editing/content tests. The primary AI routes now cover RVC backend execution, natural-language color intents, and cut-point pacing templates.

**Pass 77 update (no standalone research file):**
- F224 deepfake detection reconciliation lives in `opencut/core/deepfake_detect.py`, `opencut/routes/video_vfx_routes.py`, generated route/MCP artifacts, and `tests/test_video_vfx.py`. The AI route alias, segment evidence tags, and authenticity-report review metadata are now pinned.

**Pass 78 update (no standalone research file):**
- F228-F230 review-bundle reconciliation lives in `opencut/core/review_bundle.py`, `opencut/routes/timeline.py`, `docs/REVIEW_BUNDLES.md`, generated route/MCP artifacts, and `tests/test_review_bundle.py`. Bundle voice notes, HLS rendition packaging, route response fields, and docs are now pinned.

**Pass 79 update (no standalone research file):**
- F232 review-portal Headscale planning lives in `opencut/core/review_portal.py`, `opencut/routes/platform_infra_routes.py`, `docs/REVIEW_PORTAL.md`, and `tests/test_review_portal.py`. The route now returns sanitized operator-run Headscale/Tailscale command plans without creating keys or enabling networking in request handling.

**Pass 80 update (no standalone research file):**
- F235 WCAG 3 draft AD hooks live in `opencut/core/audio_description.py`, `opencut/routes/audio_advanced_routes.py`, `docs/AUDIO_DESCRIPTIONS.md`, generated route manifest, and `tests/test_audio_advanced.py`. Draft output can now include descriptive transcript events, extended-audio-description timing metadata, and non-normative W3C draft compatibility metadata.

**Pass 81 update (no standalone research file):**
- F245-F248 delivery-standard planning lives in `opencut/core/delivery_standards.py`, `opencut/routes/delivery_master_routes.py`, `docs/DELIVERY_STANDARDS.md`, generated route/MCP manifests, and `tests/test_delivery_standards.py`. OpenCut now exposes read-only plans for Netflix IMF/Dolby Vision, DPP/broadcaster IMF, Dolby Vision Profile 5/8.1 OSS review packaging, and ADM BW64 Atmos-master preparation, while keeping external tools, streamer acceptance, broadcaster QC, and Dolby commercial encoding outside request handling.

**Pass 82 update (F205 evidence note):**
- F205 coverage-floor uplift lives in `.github/workflows/build.yml` and `.ai/research/2026-05-17/F205_COVERAGE_FLOOR_SUCCESS.md`. The completed CI-style run reported 8,540 passed, 16 skipped, and 54.095% line coverage, so Release Full now enforces `--cov-fail-under=54`; Pass 12/23 partial attempts are historical only.

**Pass 83 update (no standalone research file):**
- F252.3 UDT result capture validation lives in `opencut/core/uxp_udt_results.py`, `opencut/tools/validate_uxp_udt_results.py`, `tests/test_uxp_udt_results.py`, `scripts/release_smoke.py`, and `docs/UXP_MIGRATION.md`. It validates pasted UDT harness JSON before any live WebView manifest cutover; the actual Premiere UDT run remains external/device-locked.

**Pass 84 update (no standalone research file):**
- Job runtime hardening lives in `opencut/jobs.py`, `opencut/routes/system.py`, `tests/test_hardening.py`, and `tests/test_job_cancellation_race.py`. Persisted async-job payloads now recursively redact credential-shaped fields before SQLite storage, stuck-job cleanup terminates registered subprocesses instead of only marking jobs failed, job-time recording has the same executor-shutdown fallback as job persistence, and `/models/delete` rejects non-string paths with a structured client error.

**Pass 85 update (no standalone research file):**
- Standalone Web UI upload hardening lives in `opencut/core/web_ui.py`, `opencut/routes/platform_ux_routes.py`, and `tests/test_platform_ux.py`. Multipart uploads now stream to disk in bounded chunks instead of buffering the full file in memory, `OPENCUT_WEB_UPLOAD_MAX_BYTES` controls the local upload limit with 413 responses and partial-file cleanup, and the JSON/base64 fallback returns structured client errors for malformed bodies instead of falling through to generic server errors.

**Pass 86 update (no standalone research file):**
- Webhook job-event compatibility lives in `opencut/core/webhook_system.py` and `tests/test_dev_scripting.py`. The webhook event catalogue now accepts `job.complete`, `job.error`, and `job.cancelled`, while `fire_event()` maps dotted async-job deliveries to legacy `job_complete`, `job_failed`, and `error` subscriptions so existing webhook configs continue receiving terminal job events.

**Pass 87 update (no standalone research file):**
- N1 transcript-cache work lives in `opencut/core/transcript_cache.py`, `opencut/core/captions.py`, `opencut/routes/captions.py`, generated route/API/feature-readiness/MCP manifests, and `tests/test_transcript_cache.py`. Cached transcripts are keyed by source SHA-256 plus backend/settings, written atomically under `~/.opencut/transcript_cache/`, surfaced through `TranscriptionResult` cache metadata, and managed with `/captions/cache/stats` plus CSRF-protected `/captions/cache/clear`.

**Pass 88 update (no standalone research file):**
- N2 install-hint work lives in `opencut/core/install_hints.py`, `opencut/errors.py`, `opencut/jobs.py`, and `tests/test_missing_dependency_hints.py`. Missing-dependency responses now include OpenCut extra commands, package hints, and GPU/VRAM notes for the May 26 top-12 dependency failures, including async job status errors.

**Pass 89 update (no standalone research file):**
- N3 GPU semaphore wait work lives in `opencut/core/gpu_semaphore.py`, `opencut/errors.py`, `opencut/jobs.py`, and `tests/test_gpu_semaphore_wait.py`. The default acquire wait is now 30 seconds, `OPENCUT_GPU_ACQUIRE_TIMEOUT=0` preserves fail-fast behavior, and timed-out contention carries `GPU_BUSY` retry metadata into direct errors and async job status.

**Pass 90 update (no standalone research file):**
- N6 webhook event discovery lives in `opencut/core/webhook_system.py`, `opencut/routes/dev_scripting_routes.py`, `opencut/routes/mcp_bridge_routes.py`, `tests/test_dev_scripting.py`, and `tests/test_mcp_bridge.py`. `/webhooks/event-types` and `/api/webhooks/event-types` expose canonical webhook event metadata plus legacy aliases, and `/mcp/info` now advertises the discovery endpoint for tooling clients.

**Pass 91 update (no standalone research file):**
- E11 mandatory webhook signature work lives in `opencut/core/webhook_system.py`, `opencut/routes/dev_scripting_routes.py`, `docs/REVIEW_NOTIFICATIONS.md`, and `tests/test_dev_scripting.py`. New HTTP webhook registrations require a non-empty HMAC secret unless `?allow_unsigned=true` is explicitly provided, and unsigned configs create a one-time `webhooks_unsigned.txt` operator warning.

**Pass 92 update (no standalone research file):**
- N4 disk preflight work lives in `opencut/core/preflight.py`, `opencut/jobs.py`, heavyweight caption/audio/video route decorators, and `tests/test_disk_preflight.py`. `@async_job` can now run a synchronous disk budget check before creating jobs and return HTTP 507 with required/free/output-dir metadata when the output volume is below budget.

**Pass 93 update (no standalone research file):**
- N5 interrupted-job resume work lives in `opencut/jobs.py`, `opencut/job_store.py`, `opencut/routes/jobs_routes.py`, checkpointable route decorators, and `tests/test_job_resume.py`. `@async_job(..., resumable=True)` persists resume lineage and partial-output metadata, running jobs are saved before worker dispatch, and `POST /jobs/<job_id>/resume` replays persisted endpoint/payload data only for interrupted jobs whose current route remains resumable.

**Pass 94 update (no standalone research file):**
- N7 plugin job-registration work lives in `opencut/core/plugins.py`, `opencut/core/plugin_manifest.py`, `opencut/data/example_plugins/long-job-demo/`, `tests/test_plugins.py`, and `tests/test_plugin_manifest.py`. Plugin manifests can declare `jobs` behind `jobs.register`, plugin routes can use `plugin_job(...)` to enqueue namespaced core async jobs, and path-like payload fields stay confined to the plugin `data/` directory unless the plugin declares `host.filesystem`.

**Pass 97 update (no standalone research file):**
- N9 enriched-job-metadata work lives in `opencut/jobs.py`, `opencut/job_store.py`, `opencut/core/job_diagnostics.py`, `opencut/routes/jobs_routes.py`, generated route/MCP manifests, and `tests/test_job_metadata.py`. Async jobs now sample peak CPU/RSS and optional NVML VRAM, persist explicit `exit_reason` values, expose the fields through live status/history/diagnostics plus `GET /jobs/<job_id>`, and preserve additive SQLite compatibility for older `jobs.db` files.

**Pass 98 update (no standalone research file):**
- N10 request-ID subprocess propagation lives in `opencut/core/request_correlation.py`, `opencut/jobs.py`, `opencut/helpers.py`, and `tests/test_request_correlation_subprocess.py`. Async jobs stamp and restore request IDs across worker threads, FFmpeg subprocesses receive `OPENCUT_REQUEST_ID`, and logged FFmpeg stderr is request-prefixed without changing returned stderr or progress parsing.

**Pass 99 update (no standalone research file):**
- E12 manifest-derived workflow allowlist work lives in `opencut/core/workflow.py`, `opencut/tools/dump_route_manifest.py`, generated `opencut/_generated/route_manifest.json`, route workflow-step annotations, `tests/test_route_manifest.py`, and `tests/test_workflow.py`. Workflow-safe async POST routes now opt in explicitly with `workflow_step(...)`, and validation loads labels from route-manifest metadata.

**Pass 100 update (no standalone research file):**
- E13 CLI surface parity work lives in `opencut/cli.py` and `tests/test_cli_route.py`. `opencut route METHOD PATH` validates route/method pairs against the generated route manifest, supports query parameters plus JSON bodies from literals/files/stdin/fields, automatically sends the local CSRF token for mutating requests, and prints formatted or raw backend responses.

**Pass 101 update (no standalone research file):**
- E15 i18n batch 4 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Eight dependency/install feedback messages now route through `t(...)`; the guarded migration ledger covers 21 keys across four rounds, and the drift gate reports 447 keys, 305 consumers, 142 dead keys, and 0 missing keys.

**Pass 102 update (no standalone research file):**
- E15 i18n batch 5 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Nine OAuth/social-auth and live-updates bridge messages now route through `t(...)`; the guarded migration ledger covers 30 keys across five rounds, and the drift gate reports 456 keys, 314 consumers, 142 dead keys, and 0 missing keys.

**Pass 103 update (no standalone research file):**
- E15 i18n batch 6 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Ten clip/workflow input prompts now route through `t(...)`; the guarded migration ledger covers 40 keys across six rounds, and the drift gate reports 466 keys, 324 consumers, 142 dead keys, and 0 missing keys.

**Pass 104 update (no standalone research file):**
- E15 i18n batch 7 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Nine media/caption chain prompts now route through `t(...)`; the guarded migration ledger covers 49 keys across seven rounds, and the drift gate reports 475 keys, 333 consumers, 142 dead keys, and 0 missing keys.

**Pass 105 update (no standalone research file):**
- E15 i18n batch 8 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Ten Whisper/settings status feedback messages now route through `t(...)`; the guarded migration ledger covers 59 keys across eight rounds, and the drift gate reports 485 keys, 343 consumers, 142 dead keys, and 0 missing keys.

**Pass 106 update (no standalone research file):**
- E15 i18n batch 9 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Ten import/export result feedback messages now route through `t(...)`; the guarded migration ledger covers 69 keys across nine rounds, and the drift gate reports 495 keys, 353 consumers, 142 dead keys, and 0 missing keys.

**Pass 107 update (no standalone research file):**
- E15 i18n batch 10 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Ten session replay/journal apply feedback messages now route through `t(...)`; the guarded migration ledger covers 79 keys across ten rounds, and the drift gate reports 505 keys, 363 consumers, 142 dead keys, and 0 missing keys.

**Pass 108 update (no standalone research file):**
- E15 i18n batch 11 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Seven settings/live utility feedback messages now route through `t(...)`; the guarded migration ledger covers 86 keys across eleven rounds, and the drift gate reports 512 keys, 370 consumers, 142 dead keys, and 0 missing keys.

**Pass 109 update (no standalone research file):**
- E15 i18n batch 12 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Nine path open/reveal and journal revert feedback messages now route through `t(...)`; the guarded migration ledger covers 95 keys across twelve rounds, and the drift gate reports 521 keys, 379 consumers, 142 dead keys, and 0 missing keys.

**Pass 110 update (no standalone research file):**
- E15 i18n batch 13 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Ten transcript/preview/polish feedback messages now route through `t(...)`; the guarded migration ledger covers 105 keys across thirteen rounds, and the drift gate reports 531 keys, 389 consumers, 142 dead keys, and 0 missing keys.

**Pass 111 update (no standalone research file):**
- E15 i18n batch 14 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Seven polish finish/journal/cut-review feedback messages now route through `t(...)`; the guarded migration ledger covers 112 keys across fourteen rounds, and the drift gate reports 538 keys, 396 consumers, 142 dead keys, and 0 missing keys.

**Pass 112 update (no standalone research file):**
- E15 i18n batch 15 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Ten preset management feedback messages now route through `t(...)`; the guarded migration ledger covers 122 keys across fifteen rounds, and the drift gate reports 548 keys, 406 consumers, 142 dead keys, and 0 missing keys.

**Pass 113 update (no standalone research file):**
- E15 i18n batch 16 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Seven preset file import/export feedback messages now route through `t(...)`; the guarded migration ledger covers 129 keys across sixteen rounds, and the drift gate reports 555 keys, 413 consumers, 142 dead keys, and 0 missing keys.

**Pass 114 update (no standalone research file):**
- E15 i18n batch 17 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Ten project template feedback and label sites now route through `t(...)`; the guarded migration ledger covers 139 keys across seventeen rounds, and the drift gate reports 564 keys, 423 consumers, 141 dead keys, and 0 missing keys.

**Pass 115 update (no standalone research file):**
- E15 i18n batch 18 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Transcript timeline/editor copy, summarize-panel feedback, and playhead-sync failure feedback now route through `t(...)`; the guarded migration ledger covers 158 keys across eighteen rounds, and the drift gate reports 583 keys, 442 consumers, 141 dead keys, and 0 missing keys.

**Pass 116 update (no standalone research file):**
- E15 i18n batch 19 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Caption display-settings preview/loading/reset/error status copy and preview sample text now route through `t(...)`; the guarded migration ledger covers 167 keys across nineteen rounds, and the drift gate reports 592 keys, 451 consumers, 141 dead keys, and 0 missing keys.

**Pass 117 update (no standalone research file):**
- E15 i18n batch 20 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Model inventory/delete feedback, GPU recommendation feedback, and queue-clear/status messages now route through `t(...)`; the guarded migration ledger covers 191 keys across twenty rounds, and the drift gate reports 616 keys, 475 consumers, 141 dead keys, and 0 missing keys.

**Pass 118 update (no standalone research file):**
- E15 i18n batch 21 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Recent-output browser labels/actions and batch-picker empty/remove feedback now route through `t(...)`; the guarded migration ledger covers 205 keys across twenty-one rounds, and the drift gate reports 630 keys, 489 consumers, 141 dead keys, and 0 missing keys.

**Pass 119 update (no standalone research file):**
- E15 i18n batch 22 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Custom workflow builder validation, save/load/delete summaries, run status, empty states, per-step remove labels, and saved-workflow select fallbacks now route through `t(...)`; the guarded migration ledger covers 222 keys across twenty-two rounds, and the drift gate reports 647 keys, 506 consumers, 141 dead keys, and 0 missing keys.

**Pass 120 update (no standalone research file):**
- E15 i18n batch 23 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Workflow preset library loading, preset optgroups/options, preset run status, and workflow terminal completion/error/cancel feedback now route through `t(...)`; the guarded migration ledger covers 233 keys across twenty-three rounds, and the drift gate reports 658 keys, 517 consumers, 141 dead keys, and 0 missing keys.

**Pass 121 update (no standalone research file):**
- E15 i18n batch 24 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Workflow preset descriptions, summary pills, preset-count labels, readiness titles, reconnect guidance, empty states, and selected-clip status copy now route through `t(...)`; the guarded migration ledger covers 256 keys across twenty-four rounds, and the drift gate reports 681 keys, 540 consumers, 141 dead keys, and 0 missing keys.

**Pass 122 update (no standalone research file):**
- E15 i18n batch 25 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Batch-run start, failure, running, poll-failure, progress, finished, and completion feedback now route through `t(...)`; the guarded migration ledger covers 270 keys across twenty-five rounds, and the drift gate reports 695 keys, 554 consumers, 141 dead keys, and 0 missing keys.

**Pass 123 update (no standalone research file):**
- E15 i18n batch 26 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Batch queue labels, operation summaries, reconnect guidance, add-more prompts, and ready-state status copy now route through `t(...)`; the guarded migration ledger covers 285 keys across twenty-six rounds, and the drift gate reports 710 keys, 569 consumers, 141 dead keys, and 0 missing keys.

**Pass 124 update (no standalone research file):**
- E15 i18n batch 27 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Workspace shell disconnected copy, source fallbacks, suite fallback, status badge labels/titles, and clip-selection prompts now route through `t(...)`; the guarded migration ledger covers 300 keys across twenty-seven rounds, and the drift gate reports 725 keys, 584 consumers, 141 dead keys, and 0 missing keys.

**Pass 125 update (no standalone research file):**
- E15 i18n batch 28 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Connection badge/port labels, reconnect banner, GPU badge, project-media scan placeholders, clip-select fallbacks, and Premiere media read failure alerts now route through `t(...)`; the guarded migration ledger covers 315 keys across twenty-eight rounds, and the drift gate reports 738 keys, 599 consumers, 139 dead keys, and 0 missing keys.

**Pass 126 update (no standalone research file):**
- E15 i18n batch 29 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Workspace header tab subtitles and per-tab stage kicker/title/copy metadata now route through `t(...)` at render time; the guarded migration ledger covers 364 keys across twenty-nine rounds, and the drift gate reports 787 keys, 648 consumers, 139 dead keys, and 0 missing keys.

**Pass 127 update (no standalone research file):**
- E15 i18n batch 30 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Central job lifecycle busy, chained-step prefix, preparing, start-failure, default processing, run-failed, unknown-error, finished, and success-summary feedback now route through `t(...)`; the guarded migration ledger covers 374 keys across thirty rounds, and the drift gate reports 797 keys, 658 consumers, 139 dead keys, and 0 missing keys.

**Pass 128 update (no standalone research file):**
- E15 i18n batch 31 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Central cleaned error guidance for route/backend/request failures, source/file permission problems, timeouts, install advice, memory retries, and server-running hints now route through `t(...)`; the guarded migration ledger covers 387 keys across thirty-one rounds, and the drift gate reports 810 keys, 671 consumers, 139 dead keys, and 0 missing keys.

**Pass 129 update (no standalone research file):**
- E15 i18n batch 32 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Inline install helper defaults/failures, manual Demucs/watermark install feedback, install start messages for depth/emotion/CrisperWhisper/B-roll/multimodal diarization, structured error-code guidance, spinner working text, and alert action links now route through `t(...)`; the guarded migration ledger covers 402 keys across thirty-two rounds, and the drift gate reports 825 keys, 686 consumers, 139 dead keys, and 0 missing keys.

**Pass 130 update (no standalone research file):**
- E15 i18n batch 33 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Interview-polish state hints, video-AI dependency hints, loudness measuring copy, watermark detection feedback, social result labels, social upload feedback, issue-bundle fallback, demo-footage feedback, and BridgeTalk readiness toasts now route through `t(...)`; the guarded migration ledger covers 420 keys across thirty-three rounds, and the drift gate reports 843 keys, 704 consumers, 139 dead keys, and 0 missing keys.

**Pass 131 update (no standalone research file):**
- E15 i18n batch 34 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Shortcut registry labels and numeric input validation toasts now route through `t(...)`; the guarded migration ledger covers 433 keys across thirty-four rounds, and the drift gate reports 848 keys, 717 consumers, 131 dead keys, and 0 missing keys.

**Pass 132 update (no standalone research file):**
- E15 i18n batch 35 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Status-bar uptime, CPU/RAM, GPU-unavailable, job-count labels, job summary fallback, and language-unavailable feedback now route through `t(...)`; the guarded migration ledger covers 443 keys across thirty-five rounds, and the drift gate reports 857 keys, 726 consumers, 131 dead keys, and 0 missing keys.

**Pass 133 update (no standalone research file):**
- E15 i18n batch 36 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Wave H onboarding wizard titles, body copy, step counter, navigation labels, and completion toast now route through `t(...)`; the guarded migration ledger covers 459 keys across thirty-six rounds, and the drift gate reports 873 keys, 742 consumers, 131 dead keys, and 0 missing keys.

**Pass 134 update (no standalone research file):**
- E15 i18n batch 37 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Wave H changelog, issue-report, and gist share/import prompts and feedback now route through `t(...)`; the guarded migration ledger covers 471 keys across thirty-seven rounds, and the drift gate reports 885 keys, 754 consumers, 131 dead keys, and 0 missing keys.

**Pass 135 update (no standalone research file):**
- E15 i18n batch 38 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Live-updates bridge status labels, listener counts, connect buttons, availability hints, and settings-summary warning copy now route through `t(...)`; the guarded migration ledger covers 496 keys across thirty-eight rounds, and the drift gate reports 910 keys, 779 consumers, 131 dead keys, and 0 missing keys.

**Pass 136 update (no standalone research file):**
- E15 i18n batch 39 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Engine registry loading, unavailable, refresh, and settings-summary availability copy now route through `t(...)`; the guarded migration ledger covers 509 keys across thirty-nine rounds, and the drift gate reports 923 keys, 792 consumers, 131 dead keys, and 0 missing keys.

**Pass 137 update (no standalone research file):**
- E15 i18n batch 40 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Engine registry dynamic state labels, per-domain summaries, option suffixes, aggregate routing summaries, and preference-success toasts now route through `t(...)`; the guarded migration ledger covers 534 keys across forty rounds, and the drift gate reports 948 keys, 817 consumers, 131 dead keys, and 0 missing keys.

**Pass 138 update (no standalone research file):**
- E15 i18n batch 41 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Settings backend/speech summary defaults, backend reconnect states, and Whisper offline/not-installed status copy now route through `t(...)`; the guarded migration ledger covers 559 keys across forty-one rounds, and the drift gate reports 973 keys, 842 consumers, 131 dead keys, and 0 missing keys.

**Pass 139 update (no standalone research file):**
- E15 i18n batch 42 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Whisper installed/ready backend fallback, CPU/GPU device labels, installed titles, ready status lines, and settings speech summaries now route through `t(...)`; the guarded migration ledger covers 572 keys across forty-two rounds, and the drift gate reports 986 keys, 855 consumers, 131 dead keys, and 0 missing keys.

**Pass 140 update (no standalone research file):**
- E15 i18n batch 43 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Settings GPU status summaries, backend restart labels, and Whisper CPU-mode toggle status copy now route through `t(...)`; the guarded migration ledger covers 586 keys across forty-three rounds, and the drift gate reports 1,000 keys, 869 consumers, 131 dead keys, and 0 missing keys.

**Pass 141 update (no standalone research file):**
- E15 i18n batch 44 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Search-tab empty, cleared, unavailable, no-match, and clip-result accessibility states now route through `t(...)`; the guarded migration ledger covers 603 keys across forty-four rounds, and the drift gate reports 1,017 keys, 886 consumers, 131 dead keys, and 0 missing keys.

**Pass 142 update (no standalone research file):**
- E15 i18n batch 45 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Search-tab index counts, ready/empty pills, refresh failures, clear-index prompts/errors, indexing job progress, search-in-progress states, result scores, and selection status now route through `t(...)`; the guarded migration ledger covers 657 keys across forty-five rounds, and the drift gate reports 1,071 keys, 940 consumers, 131 dead keys, and 0 missing keys.

**Pass 143 update (no standalone research file):**
- E15 i18n batch 46 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Settings dependency-health check states, dependency status summaries, settings import/export feedback, and crash-log clear feedback now route through `t(...)`; the guarded migration ledger covers 677 keys across forty-six rounds, and the drift gate reports 1,091 keys, 960 consumers, 131 dead keys, and 0 missing keys.

**Pass 144 update (no standalone research file):**
- E15 i18n batch 47 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Command Palette descriptions, section labels, badges, empty states, status text, item aria labels, and open affordances now route through `t(...)`; the guarded migration ledger covers 714 keys across forty-seven rounds, and the drift gate reports 1,128 keys, 997 consumers, 131 dead keys, and 0 missing keys.

**Pass 145 update (no standalone research file):**
- E15 i18n batch 48 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Favorite operation labels/toasts/aria, recent-clip dropdown copy, and side-by-side preview modal/refresh feedback now route through `t(...)`; the guarded migration ledger covers 746 keys across forty-eight rounds, and the drift gate reports 1,160 keys, 1,029 consumers, 131 dead keys, and 0 missing keys.

**Pass 146 update (no standalone research file):**
- E15 i18n batch 49 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Optional-engine install progress hints and OpenTimelineIO install success feedback now route through `t(...)`; the guarded migration ledger covers 755 keys across forty-nine rounds, and the drift gate reports 1,169 keys, 1,038 consumers, 131 dead keys, and 0 missing keys.

**Pass 147 update (no standalone research file):**
- E15 i18n batch 50 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Audio waveform preview button copy, trim validation feedback, and Merge Clips empty/remove/min-count feedback now route through `t(...)`; the guarded migration ledger covers 761 keys across fifty rounds, and the drift gate reports 1,175 keys, 1,044 consumers, 131 dead keys, and 0 missing keys.

**Pass 148 update (no standalone research file):**
- E15 i18n batch 51 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. LLM settings save/test toasts, defaults-save feedback, transcript-summary pending copy, and LUT reference-image validation now route through `t(...)`; the guarded migration ledger covers 768 keys across fifty-one rounds, and the drift gate reports 1,182 keys, 1,051 consumers, 131 dead keys, and 0 missing keys.

**Pass 149 update (no standalone research file):**
- E15 i18n batch 52 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Timeline write-back Premiere-required alerts, cut-apply feedback, beat-marker summaries, and marker-add feedback now route through `t(...)`; the guarded migration ledger covers 776 keys across fifty-two rounds, and the drift gate reports 1,190 keys, 1,059 consumers, 131 dead keys, and 0 missing keys.

**Pass 150 update (no standalone research file):**
- E15 i18n batch 53 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Sequence-marker empty/error states, marker-export prerequisites/results, project-item load failures, and shared unexpected Premiere fallbacks now route through `t(...)`; the guarded migration ledger covers 784 keys across fifty-three rounds, and the drift gate reports 1,198 keys, 1,067 consumers, 131 dead keys, and 0 missing keys.

**Pass 151 update (no standalone research file):**
- E15 i18n batch 54 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Timeline rename and smart-bin validation, no-op, offline-validation, result, and shared action-failure feedback now route through `t(...)`; the guarded migration ledger covers 794 keys across fifty-four rounds, and the drift gate reports 1,208 keys, 1,077 consumers, 131 dead keys, and 0 missing keys.

**Pass 152 update (no standalone research file):**
- E15 i18n batch 55 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Chapter copy/add-marker feedback and SRT import validation, offline parse, import result, and caption-segment status now route through `t(...)`; the guarded migration ledger covers 804 keys across fifty-five rounds, and the drift gate reports 1,218 keys, 1,087 consumers, 131 dead keys, and 0 missing keys.

**Pass 153 update (no standalone research file):**
- E15 i18n batch 56 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Loudness Match no-media validation, result-table headers, OK/failed labels, and empty-result hints now route through `t(...)`; the guarded migration ledger covers 811 keys across fifty-six rounds, and the drift gate reports 1,225 keys, 1,094 consumers, 131 dead keys, and 0 missing keys.

**Pass 154 update (no standalone research file):**
- E15 i18n batch 57 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Deliverables sequence-summary fallbacks, clip-count text, loaded/not-loaded pills, readiness titles, and status prompts now route through `t(...)`; the guarded migration ledger covers 822 keys across fifty-seven rounds, and the drift gate reports 1,236 keys, 1,105 consumers, 131 dead keys, and 0 missing keys.

**Pass 155 update (no standalone research file):**
- E15 i18n batch 58 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Deliverables load-sequence connection/cache/result feedback, load-failure hints/alerts, generation progress/failure/success feedback, network fallback, and output fallback now route through `t(...)`; the guarded migration ledger covers 839 keys across fifty-eight rounds, and the drift gate reports 1,253 keys, 1,122 consumers, 131 dead keys, and 0 missing keys.

**Pass 156 update (no standalone research file):**
- E15 i18n batch 59 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Deliverables document labels, project output summaries, last-export activity text, and no-export empty-state text now route through `t(...)`; the guarded migration ledger covers 849 keys across fifty-nine rounds, and the drift gate reports 1,263 keys, 1,132 consumers, 131 dead keys, and 0 missing keys.

**Pass 157 update (no standalone research file):**
- E15 i18n batch 60 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Timeline multicam result/apply feedback, generated speaker/track labels, smart-bin empty-state text, repeat summaries/badges/guards, and shared no-cuts guards now route through `t(...)`; the guarded migration ledger covers 861 keys across sixty rounds, and the drift gate reports 1,275 keys, 1,144 consumers, 131 dead keys, and 0 missing keys.

**Pass 158 update (no standalone research file):**
- E15 i18n batch 61 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Custom Workflow saved-library labels/titles, draft summaries, empty-summary titles, loading/reconnect/empty/name prompts, and save/run readiness prompts now route through `t(...)`; the guarded migration ledger covers 878 keys across sixty-one rounds, and the drift gate reports 1,292 keys, 1,161 consumers, 131 dead keys, and 0 missing keys.

**Pass 159 update (no standalone research file):**
- E15 i18n batch 62 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Activity/job-history toggle labels, empty states, interrupted-job notices, session-context open/re-run/apply actions, job-history re-run/apply titles, and journal replay apply titles now route through `t(...)`; the guarded migration ledger covers 891 keys across sixty-two rounds, and the drift gate reports 1,305 keys, 1,174 consumers, 131 dead keys, and 0 missing keys.

**Pass 160 update (no standalone research file):**
- E15 i18n batch 63 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Media drag/drop file-loaded and missing-path feedback plus waveform load-failure toast variants now route through `t(...)`; the guarded migration ledger covers 895 keys across sixty-three rounds, and the drift gate reports 1,309 keys, 1,178 consumers, 131 dead keys, and 0 missing keys.

**Pass 161 update (no standalone research file):**
- E15 i18n batch 64 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. NLP command validation/results and OTIO export marker guards, progress, error, exported, and saved feedback now route through `t(...)`; the guarded migration ledger covers 905 keys across sixty-four rounds, and the drift gate reports 1,319 keys, 1,188 consumers, 131 dead keys, and 0 missing keys.

**Pass 162 update (no standalone research file):**
- E15 i18n batch 65 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. B-roll generated feedback, YouTube chapter clipboard alerts, transcript-summary copy feedback, and shared copy-failure feedback now route through `t(...)`; the guarded migration ledger covers 910 keys across sixty-five rounds, and the drift gate reports 1,324 keys, 1,193 consumers, 131 dead keys, and 0 missing keys.

**Pass 163 update (no standalone research file):**
- E15 i18n batch 66 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Journal recent-action summaries, latest-entry labels, undo-ready/context-only labels, reverted badges, and revert/reverting controls now route through `t(...)`; the guarded migration ledger covers 920 keys across sixty-six rounds, and the drift gate reports 1,334 keys, 1,203 consumers, 131 dead keys, and 0 missing keys.

**Pass 164 update (no standalone research file):**
- E15 i18n batch 67 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Journal loading, unavailable, empty, rollback-status, undo-ready-status, clear-confirmation, and clearing feedback now route through `t(...)`; the guarded migration ledger covers 935 keys across sixty-seven rounds, and the drift gate reports 1,349 keys, 1,218 consumers, 131 dead keys, and 0 missing keys.

**Pass 165 update (no standalone research file):**
- E15 i18n batch 68 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Interview-polish step labels, result detail counts, compressed summaries, action buttons, polish/batch button states, and failed/unknown fallbacks now route through `t(...)`; the guarded migration ledger covers 953 keys across sixty-eight rounds, and the drift gate reports 1,367 keys, 1,236 consumers, 131 dead keys, and 0 missing keys.

**Pass 166 update (no standalone research file):**
- E15 i18n batch 69 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Preflight modal accessibility labels, titles, readiness/blocking copy, file fallbacks, section headers, run-anyway labels, and the shared cancel button now route through `t(...)`; the guarded migration ledger covers 965 keys across sixty-nine rounds, and the drift gate reports 1,378 keys, 1,248 consumers, 130 dead keys, and 0 missing keys.

**Pass 167 update (no standalone research file):**
- E15 i18n batch 70 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Processing cancel states, audio-preview rendering feedback, sequence-assistant empty/loading/error/detail/action copy, and the shared Apply button now route through `t(...)`; the guarded migration ledger covers 978 keys across seventy rounds, and the drift gate reports 1,390 keys, 1,261 consumers, 129 dead keys, and 0 missing keys.

**Pass 168 update (no standalone research file):**
- Optional-extra advisory research lives in `RESEARCH_REPORT.md` and `ROADMAP.md`. A fresh `py -3.12 -m opencut.tools.pip_audit_extras --json` kept `requirements.txt` clean but found five unwaived Torch/Transformers advisories through `pyproject[all]`; RA-15 tracks whether to upgrade, split optional GPU/depth extras, or document scoped waivers.

**Pass 169 update (no standalone research file):**
- E15 i18n batch 71 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Loudness error fallback, recent-files labels, watermark detection progress/errors, reframe output dimensions, journal context-only tooltips, disconnected status metrics, clip-preview fallback copy, and quick-workflow preset-missing alerts now route through `t(...)`; the guarded migration ledger covers 989 keys across seventy-one rounds, and the drift gate reports 1,397 keys, 1,271 consumers, 126 dead keys, and 0 missing keys.

- E15 i18n batch 72 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Central job result summaries now route segments, filler removal, captions/words, audio processing details, beat detection, stem outputs, scene counts, indexed-file counts, error totals, and exported-file fallbacks through `t(...)`; `TODO.md` now carries the compact active execution queue. The guarded migration ledger covers 1,004 keys across seventy-two rounds, and the drift gate reports 1,412 keys, 1,286 consumers, 126 dead keys, and 0 missing keys.

- E15 i18n batch 73 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Helper hint tone labels, toolbar overflow hints, filler-backend missing guidance, waveform-loading feedback, and NLP command processing-button copy now route through `t(...)`; the guarded migration ledger covers 1,013 keys across seventy-three rounds, and the drift gate reports 1,421 keys, 1,295 consumers, 126 dead keys, and 0 missing keys.

- E15 i18n batch 74 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Journal marker/rename summaries, clip metadata resolution/video/audio/file-size/transcript labels, and the utility-job start fallback now route through `t(...)`; the guarded migration ledger covers 1,022 keys across seventy-four rounds, and the drift gate reports 1,430 keys, 1,304 consumers, 126 dead keys, and 0 missing keys.

- E15 i18n batch 75 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Notification headings, session-context relative-time/loading/no-clip labels, and job-history status labels now route through `t(...)`; the guarded migration ledger covers 1,042 keys across seventy-five rounds, and the drift gate reports 1,450 keys, 1,324 consumers, 126 dead keys, and 0 missing keys.

- E15 i18n batch 76 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Output browser loading/error/empty states, output toggle labels, untitled-output fallback, job-history source/output details, and cut-review summaries now route through `t(...)`; the guarded migration ledger covers 1,057 keys across seventy-six rounds, and the drift gate reports 1,465 keys, 1,339 consumers, 126 dead keys, and 0 missing keys.

- E15 i18n batch 77 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. LLM settings status guidance, saved-settings load fallback, provider test progress, connection failure, unreachable-provider, and connected-state feedback now route through `t(...)`; the guarded migration ledger covers 1,065 keys across seventy-seven rounds, and the drift gate reports 1,473 keys, 1,347 consumers, 126 dead keys, and 0 missing keys.

- E15 i18n batch 78 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Saved workflow option step counts, processing ETA labels, and silence-cuts-ready timeline status now route through `t(...)`; the guarded migration ledger covers 1,069 keys across seventy-eight rounds, and the drift gate reports 1,477 keys, 1,351 consumers, 126 dead keys, and 0 missing keys.

- E15 i18n batch 79 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Smart-bin rule placeholders, remove controls, rule-type labels, and field labels now route through `t(...)`; the guarded migration ledger covers 1,081 keys across seventy-nine rounds, and the drift gate reports 1,489 keys, 1,363 consumers, 126 dead keys, and 0 missing keys.

- E15 i18n batch 80 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Engine-summary grammar fragments and the engine preference Auto fallback now route through `t(...)`; the guarded migration ledger covers 1,083 keys across eighty rounds, and the drift gate reports 1,491 keys, 1,365 consumers, 126 dead keys, and 0 missing keys.

- E15 i18n batch 81 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Shared grammar fragments for batch-queue, saved-workflow, journal rollback, and dependency-health summaries now route through `t(...)`; the guarded migration ledger covers 1,091 keys across eighty-one rounds, and the drift gate reports 1,499 keys, 1,373 consumers, 126 dead keys, and 0 missing keys.

- E15 i18n batch 82 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Premiere import bin fallbacks, GPU recommendation unavailable labels, and the generic GPU fallback now route through `t(...)`; the guarded migration ledger covers 1,095 keys across eighty-two rounds, and the drift gate reports 1,503 keys, 1,377 consumers, 126 dead keys, and 0 missing keys.

- E15 i18n batch 83 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Preview frame alt text, hosted LLM provider fallback labels, and beat/chapter marker defaults now route through `t(...)`; the guarded migration ledger covers 1,100 keys across eighty-three rounds, and the drift gate reports 1,508 keys, 1,382 consumers, 126 dead keys, and 0 missing keys.

- E15 i18n batch 84 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Journal action labels for marker, rename, import, and smart-bin entries now route through `t(...)`; the guarded migration ledger covers 1,106 keys across eighty-four rounds, and the drift gate reports 1,514 keys, 1,388 consumers, 126 dead keys, and 0 missing keys.

- E15 i18n batch 85 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Workflow preset step-group labels now route through `t(...)`; the guarded migration ledger covers 1,112 keys across eighty-five rounds, and the drift gate reports 1,520 keys, 1,394 consumers, 126 dead keys, and 0 missing keys.

- E15 i18n batch 86 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Command Palette display names for the first ten core tools now route through `t(...)` while preserving stable palette identifiers; the guarded migration ledger covers 1,122 keys across eighty-six rounds, and the drift gate reports 1,530 keys, 1,404 consumers, 126 dead keys, and 0 missing keys.

- E15 i18n batch 87 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Command Palette display names for the next ten audio/video tools now route through `t(...)` while preserving stable palette identifiers; the guarded migration ledger covers 1,132 keys across eighty-seven rounds, and the drift gate reports 1,540 keys, 1,414 consumers, 126 dead keys, and 0 missing keys.

- E15 i18n batch 88 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Command Palette display names for the next ten video/export/caption tools now route through `t(...)` while preserving stable palette identifiers; the guarded migration ledger covers 1,142 keys across eighty-eight rounds, and the drift gate reports 1,550 keys, 1,424 consumers, 126 dead keys, and 0 missing keys.

- E15 i18n batch 89 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Command Palette display names for the next ten workflow/search/export/settings tools now route through `t(...)` while preserving stable palette identifiers; the guarded migration ledger covers 1,152 keys across eighty-nine rounds, and the drift gate reports 1,560 keys, 1,434 consumers, 126 dead keys, and 0 missing keys.

- E15 i18n batch 90 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. The remaining Command Palette display names now route through `t(...)` while preserving stable palette identifiers; the guarded migration ledger covers 1,154 keys across ninety rounds, and the drift gate reports 1,562 keys, 1,436 consumers, 126 dead keys, and 0 missing keys.

- E15 i18n batch 91 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Settings preset selector placeholders now route through `t(...)` with escaped translated option labels; the guarded migration ledger covers 1,156 keys across ninety-one rounds, and the drift gate reports 1,564 keys, 1,438 consumers, 126 dead keys, and 0 missing keys.

- E15 i18n batch 92 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Generic API request errors now route through `t(...)` before they surface through alert/toast flows; the guarded migration ledger covers 1,159 keys across ninety-two rounds, and the drift gate reports 1,567 keys, 1,441 consumers, 126 dead keys, and 0 missing keys.

- E15 i18n batch 93 lives in `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. CEP bridge and output-browser type fallbacks now route through `t(...)`; the guarded migration ledger covers 1,161 keys across ninety-three rounds, and the drift gate reports 1,569 keys, 1,443 consumers, 126 dead keys, and 0 missing keys.

- E15 i18n batch 94 lives in `extension/com.opencut.panel/client/main.js` and `tests/test_i18n_hardcoded_migration.py`. The update-available toast now reuses the existing `toast.update_available` template fallback directly and the guard bans the previous concatenated English fallback; the guarded migration ledger remains at 1,161 keys across ninety-four rounds, and the drift gate reports 1,569 keys, 1,443 consumers, 126 dead keys, and 0 missing keys.

- Cycle 6 research consolidation lives in `ROADMAP.md`, `TODO.md`, `RESEARCH_REPORT.md`, and `docs/archive/research/RESEARCHER_QUEUE_CYCLE_6_2026-06-04.md`. RA-16 tracks Adobe `release-*` dist-tags in F251 so stable Premiere UXP release-channel package movement is captured beside `latest` and `beta`.

- E15 i18n batch 95 lives in `extension/com.opencut.panel/client/index.html`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. First-viewport static shell HTML now exposes data-i18n hooks for skip-link, brand metadata, workspace chip/title/subtitle, processing/alert/session controls, server reconnect text, and initial workspace-stage copy; the guarded migration ledger covers 1,179 keys across ninety-five rounds, and the drift gate reports 1,587 keys, 1,461 consumers, 126 dead keys, and 0 missing keys.

- Cycle 7 research consolidation lives in `ROADMAP.md`, `TODO.md`, `RESEARCH_REPORT.md`, and `docs/archive/research/RESEARCH_FEATURE_PLAN_2026-06-04_CYCLE7.md`. RA-17 tracks UXP manifest schema drift and explicit Premiere-supported `manifestVersion` validation; RA-18 tracks a deprecated UXP Clipboard/legacy `uxpvideo*` sentinel before F252 WebView cutover work.

- E15 i18n batch 96 lives in `extension/com.opencut.panel/client/main.js` and `tests/test_i18n_hardcoded_migration.py`. `applyI18nToDOM()` now applies translated `title`, `placeholder`, and `aria-label` values from `data-i18n-title`, `data-i18n-placeholder`, and `data-i18n-aria-label`, so static HTML attribute hooks are runtime behavior rather than lint-only metadata; the guarded migration ledger remains at 1,179 keys across ninety-six rounds.

- E15 i18n batch 97 lives in `extension/com.opencut.panel/client/index.html`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Workspace-stage action labels/cards and media-source intake shell HTML now expose static `data-i18n` / translated attribute hooks while preserving button icon structure; the guarded migration ledger covers 1,203 keys across ninety-seven rounds, and the drift gate reports 1,611 keys, 1,487 consumers, 124 dead keys, and 0 missing keys.

- E15 i18n batch 98 lives in `extension/com.opencut.panel/client/index.html`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Cut quick-action titles/meta/tags, the Interview Polish static shell, and the Sequence Assistant static shell now expose `data-i18n` / translated attribute hooks while preserving button icon/count structure; the guarded migration ledger covers 1,227 keys across ninety-eight rounds, and the drift gate reports 1,635 keys, 1,512 consumers, 123 dead keys, and 0 missing keys.

- Cycle 8/9 research consolidation lives in `ROADMAP.md`, `TODO.md`, `RESEARCH_REPORT.md`, `docs/archive/research/RESEARCH_FEATURE_PLAN_2026-06-04_CYCLE8.md`, and `docs/archive/research/RESEARCH_FEATURE_PLAN_2026-06-04_CYCLE9.md`. RA-19 tracks UXP clipboard permission/fallback handling for the current Clipboard API; RA-20 tracks replacing raw UXP `window.confirm` usage or explicitly gating beta alert APIs with documented evidence. Cycle 9 repeated those same findings and promoted no new row.

- Cycle 10 research consolidation lives in `ROADMAP.md`, `TODO.md`, `RESEARCH_REPORT.md`, and `docs/archive/research/RESEARCH_FEATURE_PLAN_2026-06-04_CYCLE10.md`. RA-21 tracks proving the advertised Python 3.13 classifier with CI coverage or retracting the classifier until a 3.13 lane passes.

- E15 i18n batch 99 lives in `extension/com.opencut.panel/client/index.html`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Main navigation tablist and top-level tab title/ARIA attributes now expose translated hooks while preserving icon and visible-label structure; the guarded migration ledger covers 1,229 keys across ninety-nine rounds, and the drift gate reports 1,637 keys, 1,521 consumers, 116 dead keys, and 0 missing keys.

- E15 i18n batch 100 lives in `extension/com.opencut.panel/client/index.html`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Media intake action buttons now expose translated title/ARIA attributes and explicit child-label hooks while preserving icon structure; the guarded migration ledger covers 1,233 keys across one hundred rounds, and the drift gate reports 1,641 keys, 1,525 consumers, 116 dead keys, and 0 missing keys.

- E15 i18n batch 101 lives in `extension/com.opencut.panel/client/index.html`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Silence form preset/options, slider titles, padding ARIA labels, detection-method options, the VAD hint, waveform label, and Preview 10s action now expose static locale hooks; the guarded migration ledger covers 1,254 keys across one hundred one rounds, and the drift gate reports 1,660 keys, 1,546 consumers, 114 dead keys, and 0 missing keys.

- Cycle 11 research consolidation lives in `ROADMAP.md`, `TODO.md`, `RESEARCH_REPORT.md`, and `docs/archive/research/RESEARCH_FEATURE_PLAN_2026-06-04_CYCLE11.md`. RA-22 tracks pinning Release Full's CEP panel Node runtime to match PR Fast before treating panel npm gates as deterministic release evidence.

- E15 i18n batch 102 lives in `extension/com.opencut.panel/client/index.html`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Fillers form backend/model options, the custom-filler helper and placeholder, the missing-backend hint, and the CrisperWhisper install action now expose static locale hooks; the guarded migration ledger covers 1,266 keys across one hundred two rounds, and the drift gate reports 1,672 keys, 1,557 consumers, 115 dead keys, and 0 missing keys.

- Cycle 12 research consolidation lives in `ROADMAP.md`, `TODO.md`, `RESEARCH_REPORT.md`, and `docs/archive/research/RESEARCH_FEATURE_PLAN_2026-06-04_CYCLE12.md`. RA-23 tracks full-length SHA pins and guard tests for GitHub Actions workflow `uses:` references before treating release/signing workflows as immutable supply-chain evidence.

- E15 i18n batch 103 lives in `extension/com.opencut.panel/client/index.html`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Full Pipeline YouTube/Default/Aggressive/Conservative/Podcast preset options now expose static locale hooks; the guarded migration ledger covers 1,271 keys across one hundred three rounds, and the drift gate reports 1,677 keys, 1,562 consumers, 115 dead keys, and 0 missing keys.

- E15 i18n batch 104 lives in `extension/com.opencut.panel/client/index.html`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Trim form start/end labels, helper hints, mode options, and quality options now expose static locale hooks while reusing shared Mode/Quality form labels; the guarded migration ledger covers 1,280 keys across one hundred four rounds, and the drift gate reports 1,686 keys, 1,573 consumers, 113 dead keys, and 0 missing keys.

- Cycle 13 research consolidation lives in `ROADMAP.md`, `TODO.md`, `RESEARCH_REPORT.md`, and `docs/archive/research/RESEARCH_FEATURE_PLAN_2026-06-04_CYCLE13.md`. RA-24 tracks scoping Release Full `GITHUB_TOKEN` permissions by job so build/test/package legs stay read-only and only release-upload paths receive `contents: write`.

- Cycle 14 research consolidation lives in `ROADMAP.md`, `TODO.md`, `RESEARCH_REPORT.md`, and `docs/archive/research/RESEARCH_FEATURE_PLAN_2026-06-04_CYCLE14.md`. RA-25 tracks aligning Docker dependency installs with the audited Python install surfaces so retired `deep-translator` and `pydub` packages cannot return through the container path.

- Cycle 15 research consolidation lives in `ROADMAP.md`, `TODO.md`, `RESEARCH_REPORT.md`, and `docs/archive/research/RESEARCH_FEATURE_PLAN_2026-06-04_CYCLE15.md`. RA-26 tracks aligning Docker runtime docs, non-root `/home/opencut/.opencut` volume examples, compose/exposed ports, and explicit HTTP-only vs WebSocket-capable container posture.

- Cycle 16 research consolidation lives in `ROADMAP.md`, `TODO.md`, `RESEARCH_REPORT.md`, and `docs/archive/research/RESEARCH_FEATURE_PLAN_2026-06-04_CYCLE16.md`. RA-27 tracks aligning Docker GPU launch docs and compose comments so onboarding cannot reference a missing `docker-compose.gpu.yml` file.

- Cycle 17 research consolidation lives in `ROADMAP.md`, `TODO.md`, `RESEARCH_REPORT.md`, and `docs/archive/research/RESEARCH_FEATURE_PLAN_2026-06-04_CYCLE17.md`. RA-28 tracks extending generated-count checks beyond README badges so stale route/module/blueprint claims in prose, diagrams, and project-structure comments cannot drift from route-manifest and project-context truth.

- Cycle 18 research duplicate-check lives in `ROADMAP.md`, `RESEARCH_REPORT.md`, and `docs/archive/research/RESEARCH_FEATURE_PLAN_2026-06-04_CYCLE18.md`. It promoted no new RA row because Docker CI/release-smoke coverage evidence mapped back to RA-25 through RA-27.

- Cycle 19 research consolidation lives in `ROADMAP.md`, `TODO.md`, `RESEARCH_REPORT.md`, and `docs/archive/research/RESEARCH_FEATURE_PLAN_2026-06-04_CYCLE19.md`. RA-29 tracks making Docker dependency installs fail closed and parse requirement specifiers literally, separate from RA-25's retired-package Docker dependency-surface guard.

- Cycle 20 research consolidation lives in `ROADMAP.md`, `TODO.md`, `RESEARCH_REPORT.md`, and `docs/archive/research/RESEARCH_FEATURE_PLAN_2026-06-04_CYCLE20.md`. RA-30 tracks aligning Docker build-context secret/log hygiene with Git-ignored `.env*` and `*.log` patterns before `COPY . /app` can include local artifacts.

- Cycle 21 through Cycle 28 research consolidation lives in `ROADMAP.md`, `TODO.md`, `RESEARCH_REPORT.md`, `docs/archive/research/RESEARCH_FEATURE_PLAN_2026-06-05_COMPREHENSIVE.md`, and the Cycle 21 through Cycle 28 archive notes. RA-31 through RA-36 track Adobe tracker exit-code capture, tracker label contracts, label dry-runs without `gh`, lockfile advisory coverage, release SBOM fidelity, and UNC/HGFS-safe CEP panel Node commands. Cycles 24 and 25 promoted no new rows.

- E15 i18n batch 105 lives in `extension/com.opencut.panel/client/index.html`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Auto-Edit method options, Motion Threshold/Margin/Min Clip Length labels, and their slider title tooltips now expose static locale hooks while reusing the shared Detection Method label; the guarded migration ledger covers 1,289 keys across one hundred five rounds, and the drift gate reports 1,695 keys, 1,582 consumers, 113 dead keys, and 0 missing keys.

- E15 i18n batch 106 lives in `extension/com.opencut.panel/client/index.html`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Highlights form shared labels, max-count title, min/max duration ARIA labels, emotion-highlight ARIA, missing-dependency hint, Install Now action, and Requires label now expose static locale hooks while preserving literal package commands; the guarded migration ledger covers 1,297 keys across one hundred six rounds, and the drift gate reports 1,703 keys, 1,592 consumers, 111 dead keys, and 0 missing keys.

- E15 i18n batch 107 lives in `extension/com.opencut.panel/client/index.html`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Captions Auto Subtitle/Translate quick-action titles, labels, meta copy, and preset tags now expose static locale hooks while preserving nested button icon/copy structure; the guarded migration ledger covers 1,302 keys across one hundred seven rounds, and the drift gate reports 1,708 keys, 1,597 consumers, 111 dead keys, and 0 missing keys.

- E15 i18n batch 108 lives in `extension/com.opencut.panel/client/index.html`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. The FCC Caption Display Settings card title, compliance notice, FCC source link, token labels, Preview/Reset actions, loading status, live-preview label, and sample placeholder now expose static locale hooks; the guarded migration ledger covers 1,314 keys across one hundred eight rounds, and the drift gate reports 1,720 keys, 1,610 consumers, 110 dead keys, and 0 missing keys.

- E15 i18n batch 109 lives in `extension/com.opencut.panel/client/index.html`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Styled Captions Model/Language labels, eight Whisper model options, and eleven language options now expose static locale hooks while leaving style-preset and preview-legend text for a dedicated follow-up; the guarded migration ledger covers 1,333 keys across one hundred nine rounds, and the drift gate reports 1,739 keys, 1,631 consumers, 108 dead keys, and 0 missing keys.

- E15 i18n batch 110 lives in `extension/com.opencut.panel/client/index.html`, `extension/com.opencut.panel/client/locales/en.json`, `extension/com.opencut.panel/client/main.js`, `scripts/i18n_lint.py`, `tests/test_i18n_hardcoded_migration.py`, and `tests/test_i18n_drift.py`. Styled Captions Style/Font Size labels, four style optgroup labels, and nineteen style preset options now expose static locale hooks; DOM/runtime and lint support now cover `data-i18n-label`; the guarded migration ledger covers 1,356 keys across one hundred ten rounds, and the drift gate reports 1,762 keys, 1,656 consumers, 106 dead keys, and 0 missing keys.

- E15 i18n batch 111 lives in `extension/com.opencut.panel/client/index.html`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Styled Captions preview sample words, Current/Action word legend labels, Custom Action Words controls, and the missing-Whisper install hint now expose static locale hooks while preserving the literal `pip install faster-whisper` command; the guarded migration ledger covers 1,370 keys across one hundred eleven rounds, and the drift gate reports 1,776 keys, 1,670 consumers, 106 dead keys, and 0 missing keys.

- E15 i18n batch 112 lives in `extension/com.opencut.panel/client/index.html`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Subtitle Export Model/Language/Format labels, eight Whisper model options, nine language options, and four output-format options now expose static locale hooks; the guarded migration ledger covers 1,374 keys across one hundred twelve rounds, and the drift gate reports 1,780 keys, 1,675 consumers, 105 dead keys, and 0 missing keys.

- E15 i18n batch 113 lives in `extension/com.opencut.panel/client/index.html`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Transcript editor Model/options, Undo/Redo ARIA, export/search/timeline/summary controls, short format options, and timeline hint copy now expose static locale hooks while preserving runtime transcript timeline/status i18n behavior; the guarded migration ledger covers 1,397 keys across one hundred thirteen rounds, and the drift gate reports 1,803 keys, 1,699 consumers, 104 dead keys, and 0 missing keys.

- E15 i18n batch 114 lives in `extension/com.opencut.panel/client/index.html`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Caption Translation Step/Model, Source Language, Target Language, Output Format, and NLLB install-hint shell text now expose static locale hooks while reusing shared model/language/format keys where possible; the guarded migration ledger covers 1,410 keys across one hundred fourteen rounds, and the drift gate reports 1,816 keys, 1,715 consumers, 101 dead keys, and 0 missing keys.

- E15 i18n batch 115 lives in `extension/com.opencut.panel/client/index.html`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Karaoke Captions Model, Font, Font Size, speaker-label checkbox, and WhisperX install-hint shell text now expose static locale hooks while preserving literal font family names; the guarded migration ledger covers 1,414 keys across one hundred fifteen rounds, and the drift gate reports 1,820 keys, 1,719 consumers, 101 dead keys, and 0 missing keys.

- E15 i18n batch 116 lives in `extension/com.opencut.panel/client/index.html`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Caption Burn-in Style/options, Model/options, and Auto-import result checkbox shell text now expose static locale hooks while preserving the existing title, description, and run-action hooks; the guarded migration ledger covers 1,422 keys across one hundred sixteen rounds, and the drift gate reports 1,828 keys, 1,728 consumers, 100 dead keys, and 0 missing keys.

- E15 i18n batch 117 lives in `extension/com.opencut.panel/client/index.html`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Animated Captions Animation/options, Font Size, Words Per Line, and Model/options shell text now expose static locale hooks while preserving the existing title, description, and run-action hooks; the guarded migration ledger covers 1,429 keys across one hundred seventeen rounds, and the drift gate reports 1,835 keys, 1,737 consumers, 98 dead keys, and 0 missing keys.

- E15 i18n batch 118 lives in `extension/com.opencut.panel/client/index.html`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Repeat Take Detection title, Model/options, and Similarity Threshold shell text now expose static locale hooks while preserving the existing description, run-action, and apply-action hooks; the guarded migration ledger covers 1,432 keys across one hundred eighteen rounds, and the drift gate reports 1,838 keys, 1,740 consumers, 98 dead keys, and 0 missing keys.

- E15 i18n batch 119 lives in `extension/com.opencut.panel/client/index.html`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. YouTube Chapters LLM Provider/options, Model, API Key, Max Chapters, Chapter List, Copy, and marker-action shell text now expose static locale hooks while preserving the existing description and Generate Chapters action hooks; the guarded migration ledger covers 1,436 keys across one hundred nineteen rounds, and the drift gate reports 1,842 keys, 1,749 consumers, 93 dead keys, and 0 missing keys.

- E15 i18n batch 120 lives in `extension/com.opencut.panel/client/index.html`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Audio quick-action titles/labels/meta text plus the AI Stem Separation title, model/output selectors, stem checkbox labels, auto-import checkbox, run action, and Demucs install hint now expose static locale hooks while preserving nested icons, inputs, model names, and literal package commands; the guarded migration ledger covers 1,459 keys across one hundred twenty rounds, and the drift gate reports 1,865 keys, 1,772 consumers, 93 dead keys, and 0 missing keys.

- E15 i18n batch 121 lives in `extension/com.opencut.panel/client/index.html`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Audio Denoise method controls, strength label/guidance, and Preview 10s title/text now expose static locale hooks while preserving the existing run action and audio-preview player; the guarded migration ledger covers 1,464 keys across one hundred twenty-one rounds, and the drift gate reports 1,870 keys, 1,779 consumers, 91 dead keys, and 0 missing keys.

- E15 i18n batch 122 lives in `extension/com.opencut.panel/client/index.html`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Audio Studio FX and DeepFilterNet title/description/control labels, category options, auto-import rows, action label, and install hints now expose static locale hooks while preserving dynamic effect-parameter rendering and literal package commands; the guarded migration ledger covers 1,478 keys across one hundred twenty-two rounds, and the drift gate reports 1,884 keys, 1,795 consumers, 89 dead keys, and 0 missing keys.

- E15 i18n batch 123 lives in `extension/com.opencut.panel/client/index.html`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Audio Effects selector options and Apply Effect action now expose static locale hooks while preserving backend effect values; the guarded migration ledger covers 1,489 keys across one hundred twenty-three rounds, and the drift gate reports 1,895 keys, 1,806 consumers, 89 dead keys, and 0 missing keys.

- E15 i18n batch 124 lives in `extension/com.opencut.panel/client/index.html`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Audio TTS engine/voice/speed/text labels, engine and voice options, text placeholder, auto-import copy, and Edge TTS install hints now expose static locale hooks while preserving backend engine and voice identifiers; the guarded migration ledger covers 1,507 keys across one hundred twenty-four rounds, and the drift gate reports 1,913 keys, 1,828 consumers, 85 dead keys, and 0 missing keys.

- E15 i18n batch 125 lives in `extension/com.opencut.panel/client/index.html`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Audio SFX/Tone Generator title, generator controls, preset selector/options, waveform controls, frequency/duration labels, auto-import copy, and Generate Sound action now expose static locale hooks while preserving backend generator, preset, waveform, frequency, and duration values; the guarded migration ledger covers 1,524 keys across one hundred twenty-five rounds, and the drift gate reports 1,930 keys, 1,848 consumers, 82 dead keys, and 0 missing keys.

- E15 i18n batch 126 lives in `extension/com.opencut.panel/client/index.html`, `extension/com.opencut.panel/client/locales/en.json`, and `tests/test_i18n_hardcoded_migration.py`. Audio Ducking title, music path label, Browse action, music-volume and duck-amount labels, auto-import copy, and run action now expose static locale hooks while preserving backend input values; the guarded migration ledger covers 1,527 keys across one hundred twenty-six rounds, and the drift gate reports 1,933 keys, 1,853 consumers, 80 dead keys, and 0 missing keys.

**Pass 42 update (no standalone research file):**
- F198 CEP/UXP parity catalogue lives in `opencut/core/cep_uxp_parity.py`, generated `opencut/_generated/cep_uxp_parity.json`, and `tests/test_cep_uxp_parity_catalogue.py`; the pinned CEP-only surface remains `ocAddNativeCaptionTrack` and `ocQeReflect`

**Pass 43 update (no standalone research file):**
- F194 extended MCP route-tool generation lives in `opencut/mcp_extended_tools.py`, `opencut/tools/dump_mcp_extended_tools.py`, generated `opencut/_generated/mcp_extended_tools.json`, and `tests/test_mcp_extended_tools.py`; default MCP remains 39 curated tools unless `OPENCUT_MCP_EXTENDED_TOOLS=1` or `--extended-tools` is set

**Pass 44 update (no standalone research file):**
- F201 WPF installer CI lives in `.github/workflows/build.yml`, `scripts/build_wpf_installer_ci.ps1`, and `tests/test_wpf_installer_ci.py`; Windows tag/manual builds now archive the recommended WPF installer separately before the Inno fallback build

**Pass 45 update (no standalone research file):**
- F203 Windows Authenticode signing tooling lives in `.github/workflows/build.yml`, `scripts/sign_windows_artifacts.ps1`, `docs/WINDOWS_CODESIGNING.md`, and `tests/test_windows_codesigning.py`; first live signed release still requires configured `WINDOWS_CODESIGN_*` repository secrets

**Pass 46 update (no standalone research file):**
- F223 caption Unicode validation lives in `opencut/core/caption_unicode_validation.py`, `opencut/tools/caption_unicode_validation.py`, `tests/test_caption_unicode_validation.py`, and the release-smoke `caption-unicode` step. It validates RTL, mixed bidi, Indic, Japanese, and Chinese fixtures through SRT, ASS, and burn-in ASS text export. Pass 47 added F242 no-space CJK wrapping.

**Pass 47 update (no standalone research file):**
- F242 caption line breaking lives in `opencut/core/caption_line_breaks.py`, `opencut/export/srt.py`, `opencut/core/styled_captions.py`, `opencut/core/subtitle_shot_aware.py`, `docs/CAPTION_LINE_BREAKING.md`, and `tests/test_caption_line_breaks.py`. It removes whitespace-only wrapping for no-space CJK captions while keeping ICU4X/UAX14 as the documented reference model and avoiding a mandatory binary ICU dependency.

**Pass 40 update (no standalone research file):**
- F214 performance benchmark coverage lives in `opencut/core/performance_benchmarks.py` and `tests/test_performance_benchmark_registry.py`; release-smoke wiring is in `scripts/release_smoke.py`

**Pass 39 update (no standalone research file):**
- F216 job-cancellation race coverage lives in `tests/test_job_cancellation_race.py`; the production fix is in `opencut/jobs.py`, and release-smoke wiring is in `scripts/release_smoke.py`

**Pass 38 update (no standalone research file):**
- F215 fuzz harness expansion lives in `tests/fuzz/test_parser_fuzz.py` and `tests/test_fuzz_harness_targets.py`; parser hardening landed in `opencut/core/c2pa_sidecar.py`, `opencut/core/plugin_manifest.py`, `opencut/core/lut_library.py`, `opencut/core/webhook_signature.py`, and `opencut/security.py`

**Pass 36 artefact (1 file):**
- `WAVE_N_T_F_NUMBER_LEDGER.md` — F180 bridge from every Wave N, O, P, Q, R, S, and T row to either existing F-number ownership or explicit wave-only disposition; pinned by `tests/test_wave_f_number_ledger.py`

**Pass 1 artefacts (10 files):**
- `STATE_OF_REPO.md` — live repo reconnaissance
- `MEMORY_CONSOLIDATION.md` — inventory + reconciliation of every roadmap/changelog/instruction file
- `COMPETITOR_MATRIX.md` — Premiere extension competitors, OSS NLEs, commercial AI tools, agentic editor systems
- `DATASET_MODEL_INTEGRATION_REVIEW.md` — 2026-Q2 model surfaces with licences + integration recs
- `SECURITY_AND_DEPENDENCY_REVIEW.md` — CVEs, deprecation pressures, torch cascade, action items
- `FEATURE_BACKLOG.md` — F121-F190 raw harvest (70 items)
- `PRIORITIZATION_MATRIX.md` — Now / Next / Later / Under Consideration / Rejected tier placement (Pass-1 sections; Pass-2 added §6.5)
- `SOURCE_REGISTER.md` — every local + external source cited (Pass-1 R-prefixed IDs; Pass-2 appended)
- `RESEARCH_LOG.md` — search strategy + saturation + bias notes (Pass-1 sections; Pass-2 appended)
- `CHANGESET_SUMMARY.md` — every file written / edited by this run (Pass-1 sections; Pass-2 appended)

**Pass 2 artefacts (5 new files + 6 updates):**
- `ROUTE_READINESS_AUDIT.md` — route manifest deep-dive: F100/F115/MCP coverage gaps; CEP-bound routes; /api/* alias surface
- `INSTALLER_AUDIT.md` — WPF .NET 9 installer + Inno Setup + PyInstaller + Docker + Windows ARM64; CI pipeline; signing/notarisation deadlines
- `TEST_COVERAGE_GAPS.md` — 12 specific gaps (OpenAPI validity, MCP consistency, JS unit tests, launcher smoke, WPF installer tests, ML perf benchmarks, fuzz extensions, race conditions, UXP contract, SBOM completeness)
- `FEATURES_RECONCILIATION.md` — features.md sample walk; 5 UNCLEAR → F-numbers graduated
- `FEATURE_BACKLOG_ADDENDUM.md` — F191-F260 (+70 items); two regulatory deadlines on Now tier (F202 Apple notarisation, F236 FCC caption tokens)

**Pass 3 artefacts (4 new files + 4 updates):**
- `LIVE_VERIFICATION.md` — F099/F096/F093/F094/npm-audit live results; cross-platform launcher gap confirmed (F261); side-channel discoveries
- `CEP_UXP_PARITY_MATRIX.md` — completes F198: all 18 `ocXxx` JSX functions mapped against `@adobe/premierepro@26.3.0-beta.67`; only 2 truly CEP-only; F252/F253 effort revised XL → L
- `AGENT_UX_RFC.md` — F143-F145 design RFC: adopts Copilot Workspace plan + Cursor checkpoint + Underlord self-review + Aider snapshot + Claude Code Skills; rejects accept-all, auto-commit-before-preview, atomic multi-file apply
- `MARKET_POSITIONING.md` — OpenCut replaces ~$1,400/yr subscriptions; 3 pitches + 3 deprioritise + Mister Horse free-shell+paid-packs model (F268-F272)

**Pass 4 updates (no new standalone file):**
- `LIVE_VERIFICATION.md` §8 — full release-smoke first-run failure, Ruff cleanup, final PASS matrix, and targeted `119 passed` validation
- `SOURCE_REGISTER.md` Pass 4 section — local command evidence and refreshed official web sources
- `RESEARCH_LOG.md` Pass 4 section — phases, limitations, saturation note
- `CHANGESET_SUMMARY.md` §9 — Pass 4 file/code/validation summary
- `CONTINUE_FROM_HERE.md` §10 — Pass 5 entry point

**Pass 5 updates (no new standalone file):**
- `.gitattributes` — LF line-ending rules for POSIX launchers
- `OpenCut-Server.command` + `OpenCut-Server.sh` — F261 launchers
- `README.md` — F270 market-positioning lead + macOS/Linux launch instructions
- `extension/com.opencut.uxp/uxp-api-notes.md` — F262 sample URL fix
- `ROADMAP.md`, `PROJECT_CONTEXT.md`, `CHANGESET_SUMMARY.md`, `CONTINUE_FROM_HERE.md`, `LIVE_VERIFICATION.md`, `INSTALLER_AUDIT.md` — status updates marking the batch closed
- Validation: `git diff --check` PASS; `python scripts/release_smoke.py --json` PASS (`232 passed` in pytest-fast)

**Pass 6 updates (no new standalone file):**
- `extension/com.opencut.panel/scripts/check-advisories.mjs` — `--json` output for machine-readable advisory status
- `scripts/release_smoke.py` — `npm-advisory` step parses advisory JSON and fails closed if it is missing, malformed, non-`ok`, or has unwaived findings
- `docs/NODE_ADVISORIES.md` — documents the `--json` release-smoke contract
- `docs/UXP_MIGRATION.md` — F266 CEP-residual table and drop-QE rules
- `tests/test_release_smoke.py`, `tests/test_node_advisories.py`, `tests/test_uxp_migration_docs.py` — regression coverage for F264/F266
- Validation: targeted F264/F266 tests PASS (`20 passed`); full `python scripts/release_smoke.py --json` PASS (`232 passed` in pytest-fast)

**Pass 7 updates (no new standalone file):**
- `opencut/tools/dump_api_aliases.py` + `opencut/_generated/api_aliases.json` — F199 alias policy manifest
- `scripts/release_smoke.py` — added `api-aliases` drift check
- `tests/test_api_aliases.py` — committed-vs-live alias manifest guard
- `ROADMAP.md`, `PROJECT_CONTEXT.md`, `CHANGESET_SUMMARY.md`, `CONTINUE_FROM_HERE.md` — status updates and correction of the earlier 233-alias-pairs wording

**Pass 8 updates (no new standalone file):**
- `opencut/tools/dump_feature_readiness.py` + `opencut/_generated/feature_readiness.json` — F191 generated readiness manifest from route functions that call public `checks.py` probes
- `opencut/registry.py` — loads generated readiness records, merges generated routes into curated records, and owns `NON_AI_CHECKS` (F197)
- `opencut/model_cards.py` — consumes the registry-owned `NON_AI_CHECKS` allowlist
- `scripts/release_smoke.py` — added `feature-readiness` drift check and release-gate tests
- `tests/test_feature_readiness_generator.py`, `tests/test_feature_registry.py` — committed-vs-live generator checks and manifest merge coverage
- Validation: focused F191/F197 tests PASS (`35 passed`); full `python scripts/release_smoke.py --json` PASS (`241 passed` in pytest-fast)

**Pass 9-13 implementation updates (no new standalone file):**
- Pass 9 closed F195 with 39 curated MCP tools and MCP route/dispatch/path-validation tests; full release smoke PASS (`246 passed` in pytest-fast).
- Pass 10 closed F202 repository-side macOS notarization wiring; first live Apple acceptance still needs configured secrets and a macOS release runner.
- Pass 11 closed F204 release SBOM attachment for tagged/manual Linux release builds; Pass 16 later closed the deeper F219 completeness gate.
- Pass 12 closed F207 installer FFmpeg manifests and left F205 open after a 20-minute local coverage timeout; full release smoke PASS (`254 passed` in pytest-fast).
- Pass 13 closed F208 by normalizing `/openapi.json` path parameters, making operation IDs unique, adding mutating-method 400/403 responses, and including `tests/test_openapi_contract.py` in release smoke; full release smoke PASS (`258 passed` in pytest-fast).
- Pass 14 closed F209 by fixing `opencut_chat_edit` to shipped `POST /chat` and adding a live Flask route-consistency test for all MCP default and special action routes; full release smoke PASS (`259 passed` in pytest-fast).
- Pass 15 closed F218 by pinning the 99-blueprint core registration order plus `motion_design_api` alias ordering, and by adding `tests/test_route_collisions.py` to release smoke; full release smoke PASS (`266 passed` in pytest-fast).
- Pass 16 closed F219 by extending `scripts/sbom.py` with 40 unique declared Python dependency components, 47 model-card components, and 88 CycloneDX dependency graph entries; full release smoke PASS (`269 passed` in pytest-fast).
- Pass 17 closed F236 by adding FCC-style caption display setting tokens, schema/preview routes, and burn-in `display_settings` integration; full release smoke PASS (`273 passed` in pytest-fast).
- Pass 18 closed F237 by adding `opencut/core/loudness_standards.py`, sharing source-backed preset metadata across audio normalization/analysis/broadcast QC, and documenting that ITU-R BS.1770-5 is current while BS.1770-4 is superseded; full release smoke PASS (`278 passed` in pytest-fast).
- Pass 19 closed F240 by adding `opencut/core/caption_reading_profiles.py`, `GET /captions/qc/reading-profiles`, and `reading_profile` overlays for `POST /captions/qc`; full release smoke PASS (`284 passed` in pytest-fast).
- Pass 20 closed F241 by adding `opencut/tools/text_shaping_gate.py`, a `text-shaping` release-smoke step, CI workflow wiring, and tests that enforce FFmpeg/libass HarfBuzz/FriBidi support while reporting Pillow RAQM and optional Skia shaping capability.
- Pass 21 closed F243 by making `opencut/export/srt.py` default to UTF-8 without BOM, adding a legacy BOM opt-in through routes/CLI, and adding `tests/test_srt_encoding.py` to release smoke.
- Pass 22 closed F244 by adding segment ASR confidence, language confidence, Hindi/Arabic review flags, and low-confidence review reason codes to Whisper transcription outputs, transcript cache/state, JSON export, caption/transcript routes, and CLI output; full release smoke PASS (`300 passed` in pytest-fast).
- Pass 23 reattempted F205 coverage measurement and wrote `.ai/research/2026-05-17/F205_INTERRUPTED_COVERAGE_NOTE.md`; pytest was interrupted after 2,206.6 seconds, the partial ignored JSON reported 52.12% coverage, and no CI floor change was made.
- Pass 24 closed **F259** (UXP macOS HTTP workaround doc + manifest port-allowlist test), **F251** (Adobe `@adobe/premierepro` npm registry tracker + weekly GitHub Action + release-smoke `warn`-tier drift gate), **F147** (committed `opencut-mcp-server` registry manifest + `docs/MCP_SERVER.md` + drift-check release-smoke step), and **F131** (`check-esbuild-pin.mjs` + `npm run audit:esbuild` + release-smoke `esbuild-pin` step). 41 new tests across four files added to `pytest-fast`. Release smoke green excluding pytest-fast/pip-audit/npm-advisory/panel-source. `StepResult.status` taxonomy now includes a `warn` tier for informational drift signals that should not block a release.
- Pass 25 closed **F137** (pinned `mcp>=1.26,<2` in `pyproject.toml` so the pre-alpha MCP 2.x rewrite cannot be pulled in by transitive resolution) and the SRT-in / SRT-out completion of **F139** (`POST /captions/translate` accepts `srt_path` / `srt_content` and emits translated SRT when requested, honouring the F243 legacy-BOM toggle). 19 new tests across `tests/test_mcp_sdk_pin.py` (3) and `tests/test_captions_translate_srt.py` (16); both wired into `pytest-fast`. Full release smoke green (15 chained gates, 50 gate test files, 341 individual tests).
- Pass 26 closed three cleanup items in one batch: **F126** (`otio-aaf-adapter` pinned into `[otio]` and `[all]` extras so the AAF export path keeps working after the OTIO adapter split; Pass 68 refreshed this to `>=2.0,<3` with `opentimelineio>=0.17,<1`), **F181** (`scripts/bootstrap_check.py` now has a `_resolve_python_for_subprocess()` helper that detects broken UV-style trampolines and falls back to a working system Python on PATH; `check_version_sync` translates spawn failures into actionable remediation hints), and **F185** (`features.md` now opens with an aspirational-catalogue banner and codifies the "ROADMAP.md wins / the code wins" precedence rule). 13 new tests across `tests/test_otio_aaf_adapter_pin.py` (5), the extended `tests/test_bootstrap_check.py` (+4), and `tests/test_features_md_banner.py` (4). Full release smoke green (15 chained gates, 51 gate test files).
- Pass 27 closed **F140** (C2PA 2.3 alignment in `opencut/core/c2pa_sidecar.py` — manifest records `c2pa_spec_version="2.3"`, action vocabulary tuple covers the C2PA 2.3 documented set, unknown actions tolerate-but-warn, optional `cloud_trust_list`/`live`/`software_agent` fields, claim-generator string advertises the spec) and **F123** (`opencut.core.audioop_shim.install_audioop_shim()` aliases `audioop_lts` into `sys.modules["audioop"]` on Python 3.13+; pydub dropped from `[standard]`/`[audio]`/`[all]` extras — the OpenCut tree has zero `import pydub` calls). 17 new tests across `tests/test_c2pa_sidecar.py` (+8) and `tests/test_audioop_shim.py` (9). Full release smoke green (15 chained gates, 52 gate test files).
- Pass 28 closed **F128** (FFmpeg filter regression suite). `tests/test_ffmpeg_filter_regression.py` (41 tests) covers 24 required filter names and 13 representative filter-graph parses, plus specialised sanity for `silencedetect`/`loudnorm` and an FFmpeg version floor. Test auto-discovers the bundled FFmpeg via `OPENCUT_FFMPEG` / `FFMPEG_BINARY` / PATH / bundled `ffmpeg/` dir; skips cleanly when no FFmpeg is present. When F129 (FFmpeg 8.1 bump) lands, this is the first gate to flip if any filter is renamed or removed. Full release smoke green (15 chained gates, 53 gate test files).
- Pass 29 closed **F184** (`docs/ROADMAP.md` and `docs/ROADMAP-COMPLETED.md` collapsed to ~25-line pointer stubs; release smoke gains a `roadmap-mirror` step that fails closed when either stub grows past 60 lines or loses the pointer language; `tests/test_roadmap_mirror.py` pins the contract) and **F178** (`EvalResult` gains `vram_peak_mb` / `reference_score` / `backend` / `backend_choice_reason`; `run_evaluation` resets the torch CUDA peak counter on entry; new `compare_backends()` aggregator + `GET /system/ai-eval/<feature_id>/compare-backends` route emit latency/quality/VRAM stats grouped by backend and "best for latency / best for quality" hints without picking a winner). Route manifest bumped to 1,363 routes / 101 blueprints. 13 new tests across the two files. Full release smoke green (16 chained gates including the new `roadmap-mirror`, 54 gate test files).
- Pass 30 closed **F177** (model_cards.py 2026-Q2 sweep). Existing 47-card coverage was already complete, so F177 closed by adding 6 forward-looking sweep gates to `tests/test_model_cards.py`: per-category coverage floor (audio/captions/editing/generation/lipsync/llm/video each need ≥1 card), license-prefix allowlist (SPDX-friendly + in-house markers), privacy-prefix allowlist (local-only/local+cloud/cloud), hardware-prefix allowlist (cpu/gpu/cpu-gpu/cloud), 40-card baseline floor, feature_id uniqueness gate. Allowlists are deliberately prefix-based with free-text suffix allowed for nuance. 13 model_cards tests pass; release smoke green (16 chained gates).
- Pass 31 closed **F176** (public eval-dataset catalogue). New `opencut/core/eval_datasets.py` registers 13 datasets across 6 modalities (video / speech / music / audio / captions / provenance) with license, citation, size, `commercial_use_ok` flag, and `auto`/`manual` acquisition mode. Auto-download is gated by **two** conditions: operator opt-in (`OPENCUT_DOWNLOAD_EVAL=1`) AND `commercial_use_ok=True`. Two new routes: `GET /system/eval-datasets` (with `modality`/`target`/`commercial_only`/`compact` filters) and `GET /system/eval-datasets/<dataset_id>` (404 with `EVAL_DATASET_NOT_FOUND` on unknown). Route manifest bumped to 1,365 routes / 101 blueprints. 26 new tests in `tests/test_eval_datasets.py`. Release smoke green (16 chained gates, lint clean).
- Pass 32 added the **F176 follow-up download runner** at `opencut/tools/download_eval_dataset.py`. Dry-run by default; refuses execution unless three gates all hold (registry-known + `OPENCUT_DOWNLOAD_EVAL=1` or `--force` + `commercial_use_ok=True` or `--accept-noncommercial-license`). CLI exit codes 0/2/3 = ok/blocked/unknown. Stdlib urllib transport; `file://` URL test fixture covers the full execute path without network. 19 new tests in `tests/test_download_eval_dataset.py`. Release smoke green (16 chained gates, lint clean).
- Pass 33 closed **F200** (Windows installer policy doc + lockstep tests) and **F211** (cross-platform launcher smoke tests). F200 ships `docs/INSTALLER_POLICY.md` designating the WPF installer as the recommended path and the Inno script as a deprecated fallback with a milestone-gated retirement plan; `tests/test_installer_policy.py` (7 tests) extracts the WPF C# constants + Inno `#define` directives and asserts they match on 4 lockstep invariants. F211 ships `tests/test_launcher_scripts.py` (16 tests) covering all 5 launcher entry points — existence, shebang, LF endings on POSIX, `python -m opencut(.server)` entry, path-quoting that survives Program Files, the OPENCUT_HOME + bundled-FFmpeg env contract, and the 100755 git-index executable bit (with VMware shared-folder fallback). 23 new tests; release smoke green (16 chained gates, lint clean).
- Pass 34 closed **F217** (UXP BackendClient HTTP-shape contract test). New `tests/test_uxp_backend_client_contract.py` (15 tests) pins both sides of the contract: JS-side static gates on `extension/com.opencut.uxp/main.js` (exported verbs, CSRF header, 403-refresh-retry, 120s timeout, response-header CSRF refresh, `{ok,data,error,status}` return shape, timeout surfacing as result not rejection, `/status/<job_id>` polling, `job_id`/`id` field acceptance, terminal status set) + server-side runtime gates (`/health` returns `csrf_token`, `/status/<unknown>` returns JSON, mutating routes require CSRF, `/health` does not, `capabilities` is a dict). Release smoke green (16 chained gates, lint clean).

Future research runs should land under `.ai/research/<YYYY-MM-DD>/` and update this file's *§ 9 Shipping cadence* + *§ 9.5 regulatory deadlines* + *§ 10 The biggest non-obvious gaps*. The next planned run is documented in `.ai/research/2026-05-17/CONTINUE_FROM_HERE.md`.
