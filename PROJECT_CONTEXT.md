# OpenCut — Project Context

**Canonical, cross-tool source of truth for project memory, architecture, shipping cadence, and entry points.**
**Last consolidated:** 2026-05-19 (eighty-four autonomous research/verification/implementation/wrap-up passes, with Passes 1-34 on 2026-05-17 — see `.ai/research/2026-05-17/`). Pass 3 verified the live state, walked `host/index.jsx`, drafted the F143-F145 agent-conductor RFC, and quantified the market-fit story. Pass 4 ran the full release-smoke gate, fixed release-gate lint drift, and prepared the local research + hardening commit. Passes 5-75 are recorded in ROADMAP.md and the pass update notes below. Pass 76 closed F220-F222 by adding external RVC backend execution/fallback handling, natural-language color-intent grading on `/ai/auto-grade`, cut-point pacing analysis on `/ai/pacing-analysis`, and route/catalogue tests. Pass 77 closed F224 by adding `/ai/deepfake-detect`, deepfake evidence metadata, and authenticity-report review guidance. Pass 78 closed F228-F230 by adding review-bundle voice-note attachments and optional HLS browser-scrubbing renditions. Pass 79 closed F232 by adding optional Headscale/Tailscale command-plan descriptors for cross-site review portal sharing. Pass 80 closed F235 by adding WCAG 3 draft descriptive transcript and extended AD timing hooks. Pass 81 closed F245-F248 by adding deterministic Netflix IMF/Dolby Vision, DPP IMF, Dolby Vision Profile 5/8.1, and ADM BW64 delivery-standard planning routes; route manifest now reports 1,381 routes and the opt-in extended MCP catalogue reports 1,325 tools. Pass 82 closed F205 by completing the CI-style coverage run and raising Release Full to `--cov-fail-under=54`. Pass 83 advanced F252.3 with strict UDT result-capture validation for the future WebView manifest switch. Pass 84 hardened async-job persistence and cleanup by redacting persisted request secrets, terminating timed-out child processes, preserving timing samples during executor shutdown, and tightening model-delete input validation.
**Live version:** v1.32.0.

> This file is the place to land first. It is intentionally **smaller** than `CLAUDE.md` and `ROADMAP.md` and **does not duplicate** their granular content. It tells you what each other file is for and where to look next.

---

## 1. Identity (one paragraph)

OpenCut is a **local-first, MIT-licensed automation backend for Adobe Premiere Pro**, with a DaVinci Resolve scripting bridge and an MCP server sidecar. The backend is a Python/Flask server bound to `127.0.0.1:5679` (HTTP) + `:5680` (WebSocket) + `:5681` (MCP JSON-RPC). Two Premiere panels ship with it: a **CEP** panel (`com.opencut.panel`, ~7,730-line `main.js`) for Premiere 2019–25.5, and a **UXP** panel (`com.opencut.uxp`, ~1,500-line ES module) for Premiere 25.6+. Both panels talk to the same backend. No subscriptions, no cloud, no API keys required for core features.

---

## 2. Numbers you should trust (live as of 2026-05-18)

| Surface | Count | Source of truth |
|---|---|---|
| API routes | **1,381** | `opencut/_generated/route_manifest.json` (F099) |
| Blueprints | **101** | same |
| Core processing modules (`opencut/core/`) | **538** Python files | `ls opencut/core` |
| Route files (`opencut/routes/`) | **101** | `ls opencut/routes` |
| Tests | **190 test_*.py files** + **2 Vitest panel test files** (≥7,600 tests claimed) | `ls tests/`, `extension/com.opencut.panel/tests/` |
| CI coverage floor | **54%** | `.github/workflows/build.yml` + `.ai/research/2026-05-17/F205_COVERAGE_FLOOR_SUCCESS.md` (F205) |
| Optional AI/model cards | **47** | `opencut/_generated/model_cards.json` + `docs/MODELS.md` (F115) |
| `/api/*` routes | **233** total; **15** true aliases; **218** canonical `/api` routes | `opencut/_generated/api_aliases.json` (F199) |
| Feature readiness records | **100** total; **58** route-derived records / **67** route bindings | `opencut/registry.py` + `opencut/_generated/feature_readiness.json` + `opencut.catalog_contract` (F100/F191/F196/F197) |
| OpenAPI typed response endpoints | **110** | `opencut.openapi_registry` + `opencut/openapi.py` (F192/F193) |
| MCP curated tools | **39** | `opencut/mcp_server.py` (F195) |
| MCP extended route tools | **1,325 opt-in** | `opencut/_generated/mcp_extended_tools.json` (F194) |
| CEP JSX host functions | **18 total; 2 CEP-only** | `opencut/_generated/cep_uxp_parity.json` (F198) |
| CEP locale keys (English) | 417 | `extension/com.opencut.panel/client/locales/en.json` |
| Current version | **1.32.0** | `pyproject.toml`, `python scripts/sync_version.py --check` |

The README narrative cites "1,344 routes" — that's a stale marketing badge. **The manifest is the source of truth.** Never quote a hand-edited number in CI or docs that bypasses the manifest.

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
DaVinci Resolve ────┤   WebSocket :5680           ├─ routes/* (101 blueprints) ─► core/* (538 modules) ─► FFmpeg / Whisper / Demucs / Torch / ONNX
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
- **User data** lives at `~/.opencut/` — jobs.db, footage_index.db, social_credentials.json, llm_settings.json, brand_kit.json, onboarding.json, plugin/lock files, model caches, auth.json (F112). All access goes through `user_data.py` wrappers with per-file locks and atomic `os.replace` writes.
- **Plugin manifest v1** (F116) validates `plugin.json` + `plugin.lock.json` hash, capability allowlist, and `OPENCUT_PLUGIN_ALLOW_UNSIGNED` opt-in before mounting any blueprint.

For module-level patterns and the deep gotcha list (~270 entries), see **[`CLAUDE.md`](CLAUDE.md)** — it is the authoritative developer + agent reference.

---

## 5. Documentation map (what to read for what)

| You want to know… | Read this |
|---|---|
| What features ship today, with examples | `README.md` |
| Module-level patterns, every async-job rule, every safe_bool / UXP / CEP convention | `CLAUDE.md` |
| What's planned, in what tier, with sources | `ROADMAP.md` (v4.3 sections — F001-F120 + Wave 1-7 + Wave N-T) |
| What shipped in each release | `CHANGELOG.md` (v1.0 → v1.32.0) |
| Wave-letter detail (Apr 2026 plan) | `ROADMAP-NEXT.md` |
| Per-dependency / per-model upgrade ledger | `MODERNIZATION.md` + `docs/MODELS.md` (auto-generated, F115) |
| Threat model + responsible disclosure | `SECURITY.md` |
| Dev setup | `DEVELOPMENT.md` + `CONTRIBUTING.md` |
| UXP migration plan | `docs/UXP_MIGRATION.md` |
| macOS notarization release path | `docs/MACOS_NOTARIZATION.md` |
| Windows ARM64 packaging | `docs/WINDOWS_ARM64_PACKAGING.md` (F101) |
| Linux Flatpak/AppImage distribution | `docs/LINUX_DISTRIBUTION.md` (F249) |
| Optional Aptabase telemetry | `docs/TELEMETRY.md` (F250) |
| Node advisories disposition | `docs/NODE_ADVISORIES.md` (F095) |
| 2026-04 competitive analysis | `AUDIT.md` (v1.11) + `research.md` (v1.28.2) — both predate ROADMAP v4.3 |
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
| What's planned for the next release? | `ROADMAP.md` → Now / Next tier tables; `gh issue list` once F182 (issue seeder run) is executed. |
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

1. **Chat-conductor agent (F143)** — Descript Underlord and FireRed-OpenStoryline have proven sidebar-chat + timeline-diff + post-turn self-review is the converging UX. OpenCut has every building block (1,381 routes, MCP sidecar, LLM abstraction) but no conductor. Highest-leverage gap.
2. **UXP MCP transport (F146)** — every competing PPro MCP server today is CEP-bound and will break Sept 2026. The first UXP-MCP wins post-EOL.
3. **Real-time editor-loop preview (F158)** — StreamDiffusionV2 + Diffusion Templates (Apr 2026, MIT) unlock real-time on existing LTX-2.3 / Wan backends. CapCut / Runway / Captions charge for this.
4. **Caption translation standalone (F139)** — every commercial editor ships it; OpenCut has full dubbing but no SRT-in-SRT-out path.
5. **C2PA 2.3 + IMSC 1.3 (F140 + F141)** — Adobe and the broadcast industry are moving here; OpenCut's local-first provenance + accessibility story is otherwise strong.
6. **The CEP→UXP 15% parity gap** — workflow builder, full settings panel, plugin UI. Adobe CEP EOL is ~Sept 2026 (≈4 months away).
7. **The audiocraft `torch==2.1.0` pin** is the single biggest blocker on torch upgrades. Cascades to Transformers v5, pyannote 4.x, scenedetect 0.7, and 20+ model surfaces that increasingly require torch ≥2.6.
8. **`features.md` (402 features) ↔ F-number reconciliation** is overdue (F179). Currently ~250 of the 402 items live in implicit limbo. Pass 2 sample-walk (40 entries) found ~60% SHIPPED, ~27% UNCLEAR → 5 new F-numbers graduated (F220-F224).
9. **OpenTimelineIO Marker schema is now the F105 review-bundle interchange anchor** (Pass 48, F225; extended in Passes 49-54 with F226/F227/F229/F231/F233/F234, Pass 78 with F228/F230, and Pass 79 with F232). Review bundles include `markers.otio`, deterministic SVG drawing overlays, `annotations/index.json`, `review_threads.json`, `premiere_markers.csv`, `review_markers.edl`, `voice_notes/index.json`, optional copied voice-note audio, and optional `hls/master.m3u8` browser-scrubbing renditions while preserving `markers.json`; LAN share links use HMAC-signed review portal URLs with Caddy/mDNS descriptors and optional Headscale/Tailscale command-plan descriptors, review activity emits Atom feed entries plus optional HMAC-signed webhooks, and delivery transfer bundles provide croc/rclone handoff plans.
10. **WebView UI in Bolt UXP (March 2026, MIT) is the correct CEP→UXP migration target for the 7,730-line vanilla JS main.js** (Pass 2, F252). Rewriting to Spectrum widgets is months of work for negligible end-user benefit. Pass 59 added the dormant F252.1 Bolt/WebView scaffold at `extension/com.opencut.uxp/bolt-webview/`; Pass 60 added `PProBridge.executeHostAction()` plus `window.OpenCutUXPHost` for the 14 direct-UXP actions; Pass 61 closed F254 by adding `createSubsequenceFromRange()` for UXP range exports; Pass 62 closed F255 with `exportSubsequenceWithEncoder()`; Pass 63 closed F256 with Transcript API helpers for caption-QC context; Pass 64 closed F257 with Object Mask state helpers; Pass 65 closed F258 with UXP AAF export helpers; Pass 66 closed F260 with generated migration-risk dashboard artifacts in Settings; Pass 67 closed F267 with generated UDT smoke-harness artifacts plus a bundled `window.OpenCutUXPUdtHarness` runner for the 14 direct-UXP actions; Pass 83 added the F252.3 result-capture validator for that harness. Live WebView cutover, captured in-Premiere UDT results that pass strict validation, and UI migration remain open. Bolt UXP 1.3 also ships a `public-hybrid/` template for the C++ `.uxpaddon` path (F253) that covers the 5 truly CEP-blocked features (file drag-out, QE DOM, FCPXML/OTIO **import**, `createCaptionTrack`, `exportAsProject` sub-selection save).
11. **The route-readiness and MCP surfaces are better but still not complete** (Pass 43, updated Pass 81): F191 now adds `opencut/_generated/feature_readiness.json` with **58** route-derived readiness records across **67** direct route/check bindings, F196 adds registry-primary model-card/check cross-validation, `/system/feature-state` exposes **100** records, F192/F193 raise legacy `/openapi.json` typed response schemas from **30** to **110** dataclass-discovered route entries, F195 raises the curated MCP tool surface from **27** to **39** tools, F194 adds **1,325** opt-in generated extended MCP route tools with **100** response-schema annotations, F208 pins `/openapi.json` route coverage + path-parameter validity, and F209 pins every curated MCP tool/special action route against the live Flask app. Remaining visibility work is route-level coverage, not model-card/check drift.
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
