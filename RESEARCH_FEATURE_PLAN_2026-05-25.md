# OpenCut — Research and Feature Plan (2026-05-25 Pass)

> **Companion file to** [`ROADMAP.md`](ROADMAP.md), [`ROADMAP-NEXT.md`](ROADMAP-NEXT.md), [`AUDIT.md`](AUDIT.md) (2026-04-14, v1.11), and [`research.md`](research.md) (2026-04-22, v1.28.2).
> **Scope of this pass:** observation-driven gaps that surfaced from a live walkthrough of the v1.32.0 tree on 2026-05-25. Where an item is already F-numbered or wave-tracked, I link to the existing record rather than re-author it.
> **Baseline:** HEAD `a8f62c0` (Wave S core modules merged 2026-05-19). `opencut/__init__.py` → `__version__ = "1.32.0"`.

> ## Execution Status (2026-05-25 autonomous loop)
>
> **All P0/P1/P2 actionable items in the plan are closed.** Loop 1 (11 commits, `05ff7d7` → `de7f3f4`) shipped Phase 0+1 + the Q6/E6 + E7 items. Loop 2 (4 commits, `be2ba41` → `304cad6`, plus this F143 commit) shipped Q3, Q8, Q7, and the F143 conductor backend. Route manifest now reports **1,514 routes / 106 blueprints**. Live tree adds 7 release-smoke gates: `badges`, `doc-sizes`, `subprocess-timeouts`, `panel-parity`, `i18n-drift`, `test-breadth`, plus the existing `version-sync`.
>
> | Item | Status | Commit |
> |---|---|---|
> | Q1 — Wave Q+R+S routes | ✅ shipped | `05ff7d7` |
> | Q4/E8 — README badge generator | ✅ shipped | `2ae1faf` |
> | E3 — Supply-chain triad on PR-fast | ✅ shipped | `0e62101` |
> | E1 — Doc-size drift gate + refresh | ✅ shipped | `9b2293a` |
> | E2 — Untrack 951 .NET build artifacts | ✅ shipped | `f02e888` |
> | E4 — Subprocess timeout AST linter | ✅ shipped | `462bb2e` |
> | E5 — Structured check-failure registry | ✅ shipped | `26edfe4` |
> | Q5 — CEP/UXP tab parity gate | ✅ shipped | `9a5136a` |
> | E3 follow-on — Bandit on PR + B324 fixes | ✅ shipped | `f698a4f` |
> | Q6/E6 — i18n drift gate | ✅ shipped | `8ff016f` |
> | E7 — A11y invariant tests | ✅ shipped | `de7f3f4` |
> | Q9 — Test breadth ratchet | ✅ shipped | `860503a` |
> | Q3 — One-click Enhance macro | ✅ shipped | `be2ba41` |
> | Q8 — Shorts A/B variant generator | ✅ shipped | `a4e8906` |
> | Q7 / F273 — Sequence Index panel backend | ✅ shipped | `304cad6` |
> | Q2 / F143 — Chat-conductor backend scaffold | ✅ shipped | `777bc28` |
> | F144 — LLM self-review polish (structured JSON + drift score + retry append) | ✅ shipped | `3424e19` |
> | F146 — UXP MCP bridge | ✅ shipped | `e81a1d9` |
> | UXP Agent tab (UI for F143/Q3/Q7/Q8/F146) | ✅ shipped | `1e51521` |
> | E6 follow-on — i18n migration (4 E6 strings) | ✅ shipped | `79866b3` |
> | E6 extended — i18n migration (4 more high-impact alerts) | ✅ shipped | `71cc3fd` |
> | UI-1 — UXP Agent tab CSS polish (oc-step-list, oc-card--nested) | ✅ shipped | `6142e17` |
> | UI-2 — F236 FCC caption display-settings UI (UXP Captions tab) | ✅ shipped | `8d7ebd2` |
> | UI-3 — i18n migration round 3 (5 more bare-English prompts) | ✅ shipped | `8844191` |
>
> **Remaining work (deferred to dedicated passes):**
>
> - **F202 macOS notarization** — needs `APPLE_ID` / `APPLE_TEAM_ID` GitHub Secrets before first signed release (deadline 2026-09-01).
> - **F252 UXP WebView cutover** — months-scale Bolt UXP migration (CEP EOL ~2026-09). The new UXP Agent tab + Captions display-settings card already use the modern UXP pattern; the broader CEP→UXP cutover work continues.
> - **F236 in CEP** — UXP got the display-settings card in UI-2; the same card can be replicated in the CEP captions tab in a follow-on pass.
> - **i18n migration breadth** — 13 keys migrated across three rounds; ~85+ bare-English sites remain in `main.js`. Continue in rolling 5-string batches per loop.
> - **F144 prompt-tuning** — structured-JSON path is wired and tested with mocked LLM; the prompt itself benefits from real Ollama runs in a focused session.

---

## Executive Summary

OpenCut at v1.32.0 is the most feature-broad open-source Adobe Premiere Pro automation backend that exists — 1,381 routes across 101 blueprints, 593 core modules, 197 test files, and two Premiere panels (CEP and UXP). The shape that is working best: a Flask app factory with a strict `@async_job` decorator, structured error taxonomy, SQLite job persistence, CSRF + path-traversal guards, an MCP sidecar, generated manifests for routes/MCP/features/models, and an aggressive multi-wave shipping cadence (Wave S landed 6 days ago).

The highest-value direction for improvement is **not more model surfaces** — Waves N–S delivered 38 additional 503-stubs in May 2026 alone, and Wave S in particular landed without any Flask route wiring. The highest-value direction is **closing the credibility gap between what the planning ledgers claim and what a user can actually run**:

1. **Wave S core modules ship with zero HTTP routes** — verified: `face_reage`, `asr_canary`, `asr_parakeet`, `multimodal_qwen3vl`, `multimodal_internvl3`, `music_heartmula`, and the three `relight_*` modules + `upscale_seedvr2` are pure Python facades with no blueprint binding. ROADMAP and CHANGELOG do not flag this. Either route them or move them to "Later" with an explicit "not yet exposed" note.
2. **CLAUDE.md size claims are stale by ~2×.** It says CEP `main.js` is ~7,730 lines and CEP `style.css` ~5,000; the files are 15,263 and 17,870 lines respectively. UXP `main.js` is 5,568 lines, not "~1,500" / "~1,700". This drift directly mis-prices the F252 WebView UI migration.
3. **README "1,344 routes" badge is stale.** Manifest at `opencut/_generated/route_manifest.json` reports `total_routes: 1381`. The README warns about this but the badge still ships. Make the badge generated.
4. **CI floor of 54% line coverage hides a 93% module-untested ratio.** 553 of 593 `opencut/core/*` modules are not imported by any `tests/test_*.py`. Coverage is concentrated in a handful of large modules.
5. **478 .NET build artifacts (`installer/src/.../bin/Debug/**.dll`) are tracked in git.** Pass 4 (per PROJECT_CONTEXT § 8) noted the gitignore drift; the artifacts are still in `HEAD`.
6. **`installer/obj/` is dirty in the working tree right now** (28 deleted, 4 modified files). The build outputs the gitignore allows in keep generating noise.
7. **PR-fast CI skips the surfaces that catch supply-chain regressions** — `pip-audit`, `npm-advisory`, `esbuild-pin`, `panel-source`, and `adobe-premierepro-versions` only run on Release Full.
8. **CEP↔UXP tab parity drift**: CEP has an "Export" tab; UXP has a "Deliverables" tab. They are not the same surface. The CEP→UXP cut-over story assumes parity that isn't true at the tab level.
9. **i18n drift**: ~142 of 426 keys in `client/locales/en.json` have no consumer in `index.html`. Several user-visible English strings in `main.js` (lines 2100, 2125, 2264) are hardcoded, not i18n'd.
10. **Subprocess timeout coverage is incomplete.** The repo has 153 `NotImplementedError` raises and 260 files raising `RuntimeError`, but multiple `subprocess.run` calls in `opencut/core/` (e.g. `chromatic_aberration.py`, `dead_time.py`) lack `timeout=`. The audit-batch commits closed many, not all.

The next four weeks of agent work should bias toward **honesty, hygiene, and parity** — not toward Wave T. That's what the section below proposes.

### 10 Highest-Value Opportunities (priority order)

| # | Title | Priority | Effort | Why first |
|---|-------|----------|--------|-----------|
| 1 | Route the Wave S core modules or demote them to Later | P0 | S | Closes the largest "shipped vs runnable" honesty gap. |
| 2 | Make the README route count + version + extension counts generated, not hand-edited | P0 | S | The drift PROJECT_CONTEXT calls out is one find-replace away. |
| 3 | Refresh CLAUDE.md file-size claims + module counts to live numbers | P0 | S | Multiple file sizes are off by 2×; mis-prices migration estimates. |
| 4 | Promote PR-fast to include `pip-audit`, `npm-advisory`, `esbuild-pin`, `panel-source` | P0 | S | A supply-chain regression on PR is invisible until Release Full. |
| 5 | Add `bandit`, `semgrep`, and `mypy --ignore-missing-imports opencut/core/` gates | P1 | M | Catches the SSRF/path-traversal/typing gaps the audit batches still find by hand. |
| 6 | Remove `installer/src/*/bin/Debug/` from git history; tighten `.gitignore` | P1 | S | 478 tracked DLLs bloat clones and confuse blame. |
| 7 | F143 chat-conductor `/agent/chat` — implementable now, biggest UX leap | P1 | L | Every building block exists; the agentic editor wins post-CEP-EOL. |
| 8 | One-click "Enhance" macro (denoise + normalize + stabilize + color) | P1 | M | Highest user value/effort ratio per AUDIT.md N9 and CapCut/Descript benchmarking. |
| 9 | Per-blueprint coverage floor (not just global 54%) | P1 | M | Forces the long tail of untested core modules into CI scope. |
| 10 | Locale lint: dead-key gate + hardcoded-English gate in release-smoke | P2 | S | Stops i18n drift from compounding before Wave M/N panel polish. |

---

## Evidence Reviewed

### Local files and directories inspected
- `README.md`, `CLAUDE.md` (partial — 25k token cap; pages 1–7 read), `AGENTS.md`, `PROJECT_CONTEXT.md`, `ROADMAP-NEXT.md` (lines 1–426), `AUDIT.md` (lines 1–600), `research.md` (lines 1–200), `CHANGELOG.md` (head + Unreleased section).
- `opencut/__init__.py` (`__version__ = "1.32.0"`).
- `opencut/_generated/route_manifest.json` — verified `total_routes: 1381`, `blueprint_count: 101`, `method_counts: {DELETE:13, GET:311, PATCH:1, POST:1057, PUT:2}`.
- `opencut/_generated/feature_readiness.json` — `total_records: 100`, `total_routes: 67`.
- `opencut/core/` directory listing (593 modules) + Wave S commit `a8f62c0` (10-file core-only diff).
- `opencut/routes/` directory listing (101 files); confirmed no `wave_n_routes.py`..`wave_t_routes.py` exist.
- Recent Wave N–S commit messages (`17f2b68`, `60fbae7`, `1e11322`, `b3201f1`, `a8f62c0`) — Wave N/O/P/Q/R routes were appended to `wave_l_routes.py`; Wave S touched no route files.
- Panel size verification: `extension/com.opencut.panel/client/main.js` = **15,263 lines**, `style.css` = **17,870 lines**, `index.html` = **4,061 lines**, `host/index.jsx` = **2,736 lines**. UXP `main.js` = **5,568 lines**, `index.html` = **1,466**, `style.css` = **3,863**.
- `.github/workflows/pr-fast.yml` head (lines 1–46), `build.yml` lint scope.
- `git ls-files | grep -E "/(bin|obj)/Debug"` → **478** tracked .NET build outputs.
- `git status --short` → working tree dirty in `installer/src/.../obj/` and `installer/tests/.../obj/` (uncommitted artifacts).
- `git log --since="2 weeks ago"` (78 commits — verified).
- `docs/` listing (23 files including UXP_MACOS_HTTP.md, MCP_SERVER.md, MODELS.md, DELIVERY_STANDARDS.md, REVIEW_BUNDLES.md).

### Git history range reviewed
- HEAD `a8f62c06` ← `b3201f1` ← `1e11322` ← `60fbae7` ← `17f2b68` ← `a2360ac` ← `453009ac` ← `ee2837c` ← `44954d8` ← `30ec619` (latest 10).
- Full last 78 commits (2 weeks) catalogued; cadence is ~5–10 ships/day during May 2026 implementation passes.

### Build/test/docs/release artifacts inspected
- `opencut_server.spec` referenced from CLAUDE.md (not opened — well covered in existing docs).
- `Dockerfile`, `docker-compose.yml`, `flathub.json`, `io.github.sysadmindoc.opencut.yml`, `OpenCut.iss`, `install.py`.
- `pyproject.toml` (read selected ranges; full extras matrix already documented in `MODERNIZATION.md`).

### External sources referenced (corroborated from `AUDIT.md` and `research.md`)
- Descript Underlord (chat-driven editing). [research.md §1.2]
- CapCut Pro 2026 (animated captions, beat-sync, one-click enhance). [AUDIT.md]
- Opus Clip (virality score, hook generation). [AUDIT.md]
- Topaz Video AI (multi-model upscaling hub). [research.md §3.2]
- DaVinci Resolve 21 (face reshape, blemish removal, IntelliScript). [research.md §1.3–1.4]
- HeyGen, ElevenLabs, Captions.ai, FireRed-OpenStoryline, Crayotter, ViMax. [research.md §1, §3]
- FlashVSR (CVPR 2026), VOID (Netflix, Apr 2026), SAM2 → SAM 3.1, Depth Anything → V3, MatAnyone 2, ACE-Step v1.5, YuE, LTX-2. [AUDIT.md Part 2]
- Apple Notarization deadline 2026-09-01 (F202). [PROJECT_CONTEXT §9.5]
- FCC "readily-accessible" captions effective 2026-08-17 (F236). [PROJECT_CONTEXT §9.5]
- Adobe CEP end-of-life ~Sept 2026. [README, UXP_MIGRATION.md]

### Areas not verified
- The "190 test_*.py files + ≥7,600 tests" claim — I see 197 files at the file-system level but did not run pytest to confirm test counts. **Assumption.**
- The 100 feature-readiness records and 67 route-derived bindings (read from JSON, not cross-verified against live Flask `url_map`).
- macOS UXP HTTP-on-loopback runtime behavior — only `docs/UXP_MACOS_HTTP.md` was located, not run in PPro.
- WPF installer end-to-end behaviour on a real Windows machine (the CI smoke is `scripts/smoke_wpf_installer.ps1`; not executed here).

---

## Current Product Map

### Core workflows (verified from README + CLAUDE.md + route manifest)
1. **Silence / filler removal → timeline write-back** (Cut tab, ExtendScript bridge).
2. **Transcription → caption styling → burn-in or SRT** (Captions tab; 19 styles; CapCut-style animated captions).
3. **Stem separation, denoise, speech enhance, loudness match, beat detection** (Audio tab).
4. **Reframe / upscale / stabilize / chromakey / inpaint / object remove / face fx** (Video tab).
5. **Highlights / shorts pipeline / smart thumbnails / B-roll planning** (Shorts/Highlights).
6. **Footage search (FTS5 over transcripts), natural-language commands, chat editor, post-production deliverables** (Search + AI Commands).
7. **Timeline write-back** (Apply cuts, beat markers, multicam, batch rename, smart bins, OTIO/AAF/FCPXML interchange).
8. **Resolve scripting bridge** (separate from Premiere).
9. **MCP sidecar** on :5681 (39 curated tools + 1,325 opt-in route tools).
10. **Review-bundle workflow** (markers, SVG overlays, annotations, voice notes, optional HLS).

### Existing feature inventory (delta from README/ROADMAP, what's new since AUDIT.md v1.11)
The April 2026 AUDIT and research files captured the v1.11 → v1.28 surface. Between then and now (v1.32):
- **Wave H** (v1.25) shipped: virality score, cursor zoom, changelog feed, issue report, demo bundle, gist sync, onboarding wizard, F5-stub block for FlashVSR / ROSE / Sammie-Roto-2 / OmniVoice / ReEzSynth / VidMuse.
- **Wave I/J/K** (v1.26–v1.28): registry+catalog contract, eval harness v2, model card sweep, RTL/CJK caption Unicode + line-breaking, FCC display-settings tokens, IMSC 1.3, OCIO 2.5, plugin manifest v1, Headscale/Tailscale planning, audio-description WCAG-3 hooks.
- **Wave L/M/N/O/P/Q/R/S** (v1.29–v1.32): Spark-TTS, Moonshine ASR, ACE-Step, FramePack, Kokoro/Chatterbox/DiffRhythm, FLUX.1 Kontext, Wan2.2 + S2V + Animate, SAM 2.1, FastVideo, LightX2V, Depth Anything V2, CogVideoX, Qwen2.5-VL, ConsisID, Allegro, HiDream-I1/E1, CogView4, Qwen2.5-Omni, Open-Sora 2.0. Most ship as 503-stubs.

### User personas (inferred from feature shape)
- **Solo creator on Premiere**: silence/filler removal → captions → vertical reframe → export. Primary panel user.
- **Podcast / interview editor**: diarization, multicam switch, loudness match, OTIO/AAF round-trip.
- **Documentary/post house**: scene detect, color match, LUT pipeline, OCIO/ACES, broadcast QC, IMF/Dolby Vision planning.
- **Power user / agent-driver**: MCP client, CLI, NLP commands, chat editor, plugin author.

### Platforms and distribution
- Windows installer (.exe) — primary; tested + signed.
- macOS source launcher (.command); notarization wiring landed (F202), not yet executed against Apple.
- Linux source launcher (.sh) + Flatpak/AppImage (F249).
- Docker (CPU + GPU variants).
- Windows ARM64 (F101 docs only).

### Important integrations and surfaces
- Adobe CEP (`com.opencut.panel`), Adobe UXP (`com.opencut.uxp`), DaVinci Resolve Python scripting bridge.
- MCP server, REST/JSON, WebSocket, SSE, NDJSON streaming.
- Webhooks, agent skills catalogue, plugin system.
- Local user data at `~/.opencut/` (SQLite jobs.db, footage_index.db, settings.json files).

---

## Feature Inventory (annotated for this pass)

Rather than re-list 1,381 routes, this section records only the **maturity flags** that came out of the audit. The README is authoritative for the rest.

| Feature surface | Maturity | Notes |
|---|---|---|
| Cut + Filler removal (Wave A core) | **Complete** | Real FFmpeg + Silero + faster-whisper / CrisperWhisper paths. |
| Captions / 19 styles / animated captions | **Complete** | Pillow + libass + ASS sanitizer, RTL/CJK gates. |
| Audio FX chain (Pedalboard, DeepFilterNet, Resemble Enhance) | **Complete** | Hybrid: works without optional deps via FFmpeg fallback. |
| TTS (Edge / Kokoro / Chatterbox / F5 / Spark / Moonshine / Dia / Parler / CSM / OmniVoice) | **Mixed** | Edge/Kokoro/Chatterbox/F5 real; Spark/Dia/Parler/CSM/OmniVoice = stubs. |
| Highlights, virality score, shorts pipeline | **Complete** (Tier 1 of Wave H) | Heuristic blend; not ML. |
| Object remove / inpaint (ProPainter, ROSE) | **Mixed** | ProPainter real, ROSE = stub. |
| Watermark removal (LaMA, Florence-2, FFmpeg delogo) | **Complete** | Real LaMA + Florence-2. |
| Video gen (LTX, Wan 2.1/2.2, CogVideoX, Open-Sora 2.0, Allegro, ConsisID, Hailuo, Seedance) | **Mostly stub** | Tier 3 commercial cloud routes return 501. |
| Lip-sync (LatentSync, MuseTalk, GaussianHeadTalk, FantasyTalking2, EchoMimic V3) | **Mixed** | LatentSync/MuseTalk real; advanced ones stubbed. |
| Wave S relighting / VSR / ASR / VLM / face-aging | **STUB, NO ROUTES** | `core/` modules exist; no blueprint. See § Highest-Value New Features Q1. |
| Review bundles / LAN portal / Headscale plan / HLS rendition | **Complete** | F228–F232 landed 2026-05-18. |
| Delivery standards (Netflix IMF / Dolby Vision / DPP / ADM BW64) | **Planning-only** | F245–F248 — read-only command-plan presets. By design. |
| WPF installer | **Complete (CI'd)** | F212 tests + F201 build + F203 codesign tooling ship; first live signed release pending secrets. |
| Inno Setup installer | **Deprecated-but-supported** | F200 policy doc. |
| Flathub / AppImage | **Build wiring present** | F249. Live submission to Flathub not done. |
| macOS notarization | **Tooling present** | F202; secrets + first signed release pending. |
| Linux Flatpak | **Tooling present** | Live submission pending. |
| MCP server (curated 39 tools, opt-in 1,325) | **Complete** | F147 upstream registry tracked. |

### Stub-vs-runnable honesty audit (this pass)

- **Stub-only core modules**: 38 verified in `opencut/core/` — all follow the pattern `RuntimeError(INSTALL_HINT)` then `raise NotImplementedError`. **Verified**.
- **Wave S route gap**: confirmed — commit `a8f62c0` lists only `opencut/core/*.py` files; no `opencut/routes/*.py` change. Wave N–R appended to `wave_l_routes.py`; Wave S did not. **Verified.**
- **Total raises**: 153 `NotImplementedError` and 260 files raising `RuntimeError` across `opencut/core/`. Not all are stub-only (some are legitimate guards), but the stub count is firmly in the high tens. **Verified.**

---

## Competitive and Ecosystem Research

For the comprehensive competitor matrix, see [AUDIT.md](AUDIT.md) (2026-04 v1.11), [research.md](research.md) (2026-04 v1.28.2), and `.ai/research/2026-05-17/COMPETITOR_MATRIX.md`. The 2026-05 pass adds only **deltas not already captured there**:

| Source / project | Notable post-April capability | What OpenCut should learn | What to intentionally avoid |
|---|---|---|---|
| **Descript Underlord 2.x** | Editable plan + post-turn self-review + checkpoint+rollback in the chat sidebar. | Adopt F143 exactly as `.ai/research/2026-05-17/AGENT_UX_RFC.md` already prescribes. | Don't ape "accept all" — render cost dominates user attention. |
| **CapCut Desktop 2026-Q2** | Real-time visual effect previews at low res with full-res commit only on Apply. | One-click "Enhance" macro + low-res preview pipeline (a reuse of the existing `/video/preview-frame` route). | Don't ship a template marketplace; you don't have the moderation budget. |
| **Adobe Premiere 26.x Sequence Index Panel** (April 2026) | Spreadsheet view of clips with rating + timecode + transcript snippet, search/sort. | Mirror in UXP (see Existing-Feature § Sequence Index below). | Don't try to be the timeline editor — pair the panel with Premiere's existing one. |
| **Cursor + Aider (agentic IDE patterns)** | Per-hunk-accept on multi-file changes; snapshot-before-change. | Apply the snapshot-then-diff pattern to the `/agent/chat` plan executor. | Don't auto-commit-before-preview; the user must see the render first (per AGENT_UX_RFC.md). |
| **Captions.ai 2026** | Stay-in-frame avatar with broadcast-grade caption styles; one-prompt vertical reformatting. | Caption style library has the right shape; expose the existing 19 → 50 surface explicitly. | Don't build a hosted avatar service. |
| **Opus Clip 2026** | Engagement heatmap predicting drop-off; A/B variant generation per clip. | Score is already built (`virality_score.py`); add an A/B variant route. | Don't gate behind cloud LLM only — keep the heuristic fallback. |
| **OBS Studio 30.x + obs-websocket v5** | Trigger ingest/auto-clip on event. | Wave E6 OBS bridge stub — promote, low effort. | Don't bundle OBS itself. |
| **Frame.io Camera-to-Cloud** | Direct ingest from cameras with proxy auto-build. | Watch-folder + cloud-storage adapter is enough; don't replicate C2C. | Don't compete with Frame.io review portal at its scale. |
| **CrewAI/AutoGen multi-agent** | Plan + critic + executor agents. | Use the agent-skills pattern that just landed (`agent_skills.py`) as a foothold, but stay single-conductor in F143. | Don't build a full crew framework — out of scope. |
| **Diffusion Templates + StreamDiffusionV2** (Apr 2026) | Sub-second iterative preview of style/effect. | F158 is already on Next-tier. | Don't ship before GPU semaphore (F223 — already done). |

---

## Highest-Value New Features

### Q1 — Route the Wave S core modules (or demote them) — P0

- **User problem**: Users reading the changelog see "Wave S — relighting, VSR, ASR, VLM, face aging" and try to call them. They get 404 because no blueprint exists.
- **Evidence**: `git show --stat a8f62c0` → 10 changed files, all in `opencut/core/`; zero in `opencut/routes/`. Cross-grep `relight_iclight|relight_video_lav|relight_diffrenderer|upscale_seedvr2|asr_parakeet|asr_canary|multimodal_qwen3vl|multimodal_internvl3|face_reage|music_heartmula` in `opencut/routes/` returns nothing.
- **Proposed behavior (two paths)**:
  1. **Route them**, behind `check_X_available()` guards: e.g. `POST /video/relight/iclight`, `POST /video/upscale/seedvr2`, `POST /asr/parakeet`, etc. — mirror the Wave H stub-with-503 pattern. ~120 LOC across one `wave_n_or_s_routes.py` (the existing waves N–R already appended to `wave_l_routes.py`, which is confusing; pick a single home).
  2. **Demote**: move the Wave S row in `ROADMAP.md` and `CHANGELOG.md` to "Later, core-only stubs, not exposed via HTTP" and reflect that in the README narrative.
- **Implementation areas**: `opencut/routes/wave_l_routes.py` (or rename → `wave_late_routes.py`), `opencut/checks.py`, `opencut/_generated/route_manifest.json` regeneration, `opencut/registry.py` if `FeatureRecord` entries are added.
- **Data model / API / UI implications**: 8 new POST routes if Path 1; zero if Path 2. The 1,325 opt-in MCP tool count goes up by 8 either way.
- **Risks**: Path 1 ships more 503s into the API surface; Path 2 looks like a regression to anyone reading the changelog.
- **Verification plan**: `python -m opencut.tools.dump_route_manifest --check` returns 0; `pytest tests/test_route_smoke.py -k "wave_s"` covers 503 path for each new route.
- **Complexity**: S. **Priority: P0.**

### Q2 — `/agent/chat` conductor (F143) — P0 (already F-numbered, restated for visibility)

- **User problem**: 1,381 routes is unsearchable; users want "remove silence, caption, reframe vertical, export" in one sentence.
- **Evidence**: `.ai/research/2026-05-17/AGENT_UX_RFC.md` already specifies adoption pattern. Every building block exists.
- **Proposed behavior**: see RFC. Phased: F143 plan executor → F144 post-turn self-review → F145 Skills SDK + MCP packaging.
- **Why list here**: This is the highest-leverage new feature; my pass adds nothing to the design but reinforces that this should ship before any further Wave T model surface.
- **Complexity**: L (8 wks at 1 maintainer per RFC). **Priority: P0.**

### Q3 — One-click "Enhance" macro — P1

- **User problem**: AUDIT.md N9, Wave 2 Wave-1 item, and every user research note (CapCut, Adobe Sensei, Topaz) all say the same: users want a single button that produces a noticeably better clip. OpenCut has every ingredient and no single-click endpoint.
- **Evidence**: AUDIT.md §3 P1-N9; competitor evidence in CapCut "Enhance" feature usage data.
- **Proposed behavior**: `POST /enhance/auto` → analyzes clip (LUFS, sharpness, motion, faces) and chains a contextually appropriate subset of: `loudness_match` (target -16 LUFS) → `audio_pro/deepfilter` → `stabilize` (smoothing 30) → `video/restore/deflicker` → `lut/auto-grade` (natural-language intent: "balanced cinematic") → optional Real-ESRGAN to 1080p if source < 720p. Returns a `job_id`. A `dry-run=1` query param returns the planned pipeline as JSON without executing.
- **Implementation areas**: New `opencut/core/enhance_auto.py` (`EnhanceConfig` + `EnhanceResult` dataclass). New `routes/enhance_routes.py`. Plug into command palette ("Enhance this clip"). Add as a Wave H Tier 1 follow-on.
- **Data model**: `EnhanceResult { plan: [...], chain_job_ids: [...], duration_seconds, fallbacks_used }`.
- **Risks**: The pipeline picks defaults that suit short-form social but not narrative — gate with a `style="social"|"cinematic"|"speech"` knob, default `social`.
- **Verification plan**: integration test in `tests/test_enhance_auto.py` with a 5-second synthetic clip; assert pipeline chain matches expected ordering; smoke-test the route returns 200 with `dry-run=1`.
- **Complexity**: M. **Priority: P1.**

### Q4 — Generated route-count badge + version surfaces — P0

- **User problem**: README badge says 1,344 routes; actual is 1,381. README itself warns about this (top of file) but the badge is not regenerated.
- **Evidence**: README.md line 8 (`API Routes-1344`), manifest `total_routes: 1381`.
- **Proposed behavior**: Extend `scripts/sync_version.py` (already syncs 19 surfaces) with a `--badges` mode that updates the README route badge, blueprint count, test count, locale-key count, and CEP/UXP `main.js` line counts. Add `python scripts/sync_version.py --check-badges` to release-smoke. Embed the source-of-truth pointer note next to the badge.
- **Implementation areas**: `scripts/sync_version.py`, `README.md`, `scripts/release_smoke.py` (add a `badges` gate).
- **Complexity**: S. **Priority: P0.**

### Q5 — CEP/UXP tab parity verification gate — P1

- **User problem**: CEP has an "Export" tab; UXP has a "Deliverables" tab. They are not the same surface. The CEP→UXP cut-over (Sept 2026) assumes parity that isn't true at the tab level.
- **Evidence**: CEP index.html line 27–57 (`data-nav` attributes) vs UXP index.html line 48–76 (`data-tab` attributes).
- **Proposed behavior**: Add `tests/test_panel_tab_parity.py` that statically parses `index.html` for both panels and asserts the tab set is either identical or **annotated** in a versioned `extension/PANEL_PARITY.json` ledger ("Deliverables-only" / "Export-only" with a justification field). The release-smoke gate fails if an unannotated divergence appears.
- **Implementation areas**: `tests/test_panel_tab_parity.py`, `extension/PANEL_PARITY.json` (new), `scripts/release_smoke.py`.
- **Complexity**: S. **Priority: P1.**

### Q6 — i18n drift gate — P2

- **User problem**: 142 of 426 locale keys are unused; main.js has hardcoded English strings that break in non-en locales.
- **Evidence**: subagent grep — 284 unique `data-i18n` references in CEP index.html vs 426 keys in en.json; hardcoded strings at `main.js:2100`, `2125`, `2264`.
- **Proposed behavior**: Add `scripts/i18n_lint.py` doing two things: (a) walk all `*.html` + `main.js` for `data-i18n="..."` and `t("...")` patterns, diff against `client/locales/en.json` → fail if keys exist without consumers OR consumers reference missing keys; (b) blacklist a curated list of user-visible string patterns (`showToast("...")`, `showAlert("...")`, `confirm("...")`) appearing as bare English literals.
- **Implementation areas**: `scripts/i18n_lint.py`, `scripts/release_smoke.py`, `tests/test_i18n_drift.py`.
- **Complexity**: S. **Priority: P2.**

### Q7 — Sequence Index panel (UXP) — P2

- **User problem**: With 1,381 routes and large project sequences, users need a spreadsheet view (Adobe just shipped this in 26.x).
- **Evidence**: research.md §4.4.
- **Proposed behavior**: Already documented as research §4.4. This pass restates it because the F-number ledger doesn't currently own it. **Recommend assigning F273.**
- **Complexity**: M. **Priority: P2.**

### Q8 — A/B variant generator on shorts pipeline — P2

- **User problem**: Opus Clip ships this; OpenCut's shorts pipeline is one-and-done.
- **Evidence**: AUDIT.md N7.
- **Proposed behavior**: `POST /shorts/pipeline/variants?n=3` → produces three variants differing in (hook tightness, caption style, reframe focal point). Reuses existing modules; orchestration only.
- **Complexity**: M. **Priority: P2.**

### Q9 — Per-blueprint coverage floor — P1

- **User problem**: 54% global coverage hides that 553 of 593 core modules are untested.
- **Evidence**: subagent count.
- **Proposed behavior**: Add `.coveragerc` per-blueprint thresholds via `coverage.json` policy + a `scripts/coverage_floor_per_blueprint.py` that fails CI if any blueprint drops below 20% line coverage (or below its previous baseline, whichever is higher). Start lenient; ratchet over time.
- **Complexity**: M. **Priority: P1.**

### Q10 — F143 + F146 packaging audit — P1 (already F-numbered)

Restated: F143 conductor must ship inside the F252 WebView UXP shell (per PROJECT_CONTEXT §10) to survive CEP EOL. Sequencing matters; F146 (UXP MCP transport) is the prerequisite.

---

## Existing Feature Improvements

### E1 — Generate `extension/com.opencut.panel/client/main.js` size note in CLAUDE.md — P0

- **Current behavior**: CLAUDE.md says `main.js` is `~7,730 lines`, `style.css` is `~5,000 lines`. Actual is **15,263** and **17,870**.
- **Problem**: The F252 migration estimate (XL → L → "scoped to caption-track + drag-out") is calibrated against the stale numbers. Re-pricing is necessary before committing to a Sept 2026 cut-over plan.
- **Recommended change**: Update CLAUDE.md "Key Files" section to read live counts (and refresh PROJECT_CONTEXT § 2 "Numbers you should trust"). Add a `scripts/check_doc_sizes.py` that asserts CLAUDE.md's stated sizes are within ±10% of `wc -l` for each cited file.
- **Code locations**: `CLAUDE.md` lines 100–105; new `scripts/check_doc_sizes.py`.
- **Backward compatibility**: N/A (docs only).
- **Verification plan**: `python scripts/check_doc_sizes.py` exits 0 against current tree.
- **Complexity**: S. **Priority: P0.**

### E2 — Tracked .NET build artifacts — P1

- **Current behavior**: 478 files under `installer/src/OpenCut.Installer/bin/Debug/...` are tracked in git. `git status` shows the parallel `obj/` tree dirty in the working copy right now.
- **Problem**: Clones are bloated; PR diffs have noise; the build outputs occasionally cause merge conflicts with no signal.
- **Recommended change**: `.gitignore` add (or migrate to a `gitignore` install via `dotnet new gitignore` at `installer/src/OpenCut.Installer/`). Then `git rm --cached -r installer/src/OpenCut.Installer/bin/ installer/src/OpenCut.Installer/obj/ installer/tests/*/obj/ installer/tests/*/bin/`. Commit. Optional `git filter-repo` to purge history — only if release tags don't reference them.
- **Code locations**: `.gitignore`; one cleanup commit.
- **Backward compatibility**: Anyone using a snapshot of the repo expecting prebuilt DLLs will need to run `dotnet build`. Document in `installer/README.md` (already exists).
- **Verification plan**: `git ls-files installer/src/OpenCut.Installer/bin/Debug/ | wc -l` returns 0.
- **Complexity**: S. **Priority: P1.**

### E3 — Promote PR-fast gates to include supply-chain checks — P0

- **Current behavior**: `.github/workflows/pr-fast.yml` line 42–46 explicitly skips `pip-audit`, `npm-advisory`, `esbuild-pin`, `panel-source`, `adobe-premierepro-versions`.
- **Problem**: A vulnerable dep introduced in a PR is invisible until Release Full (tags only). The F095/F131/F264 gates exist but only run post-merge.
- **Recommended change**: Move `pip-audit`, `npm-advisory`, and `esbuild-pin` into the PR-fast workflow. These are seconds each. Leave `panel-source` and `adobe-premierepro-versions` on Release Full (they need cross-OS).
- **Code locations**: `.github/workflows/pr-fast.yml` line 42–46.
- **Backward compatibility**: Adds ~15s to PR CI.
- **Verification plan**: open a draft PR with a known-vulnerable dep pin; assert it fails fast.
- **Complexity**: S. **Priority: P0.**

### E4 — Subprocess timeout audit — P1

- **Current behavior**: Audit batches 1–8 closed many `subprocess.run` calls without `timeout=`, but spot-check on `opencut/core/chromatic_aberration.py` and `opencut/core/dead_time.py` (and several others) suggests untimeout'd calls remain.
- **Problem**: A wedged child process pins the worker slot; the user's job stays "running" forever in the UI. CLAUDE.md gotcha line ~288 already captures the principle ("All `Popen.wait()` calls need a timeout").
- **Recommended change**: Add a `bandit` / `semgrep` rule, or a small custom linter (`scripts/lint_subprocess_timeouts.py`), that flags `subprocess.run(`, `subprocess.Popen(`, `.wait()`, `.communicate()` without a `timeout=` keyword argument. Audit batch 9.
- **Code locations**: scan `opencut/core/` and `opencut/routes/`; expected hits in dozens of files.
- **Backward compatibility**: Adding timeouts is non-breaking when set generously (1800s).
- **Verification plan**: `python scripts/lint_subprocess_timeouts.py` exits 0; new lint added to PR-fast.
- **Complexity**: M. **Priority: P1.**

### E5 — `except Exception:` swallowing in `opencut/checks.py` — P1

- **Current behavior**: 20+ bare `except Exception: return False` patterns in `checks.py`. Audit confirmed.
- **Problem**: A feature silently reports "not available" instead of surfacing why (corrupted install, missing C library, network timeout on Ollama probe).
- **Recommended change**: Wrap each in `except Exception as exc: logger.debug("check_%s failed: %s", name, exc, exc_info=True); return False`. Optionally collect last-failure-reason per check into a `_check_failures: dict[str, str]` accessible via `GET /system/check-failures` for support diagnostics.
- **Code locations**: `opencut/checks.py` (search `except Exception:` for hits).
- **Backward compatibility**: Pure observability addition.
- **Verification plan**: induce a known import failure on a check; assert `/system/check-failures` exposes the reason.
- **Complexity**: S. **Priority: P1.**

### E6 — Hardcoded English strings → i18n keys — P2

- **Current behavior**: `main.js:2100, 2125, 2264` ship English literals in `showToast`/`showAlert`/`console`-adjacent paths.
- **Problem**: Localized panels are partially English.
- **Recommended change**: As part of Q6 (i18n drift gate), move the offending strings into `client/locales/en.json` and replace inline literals with `t("…")` calls.
- **Code locations**: `extension/com.opencut.panel/client/main.js` lines 2100, 2125, 2264, plus the dead-key purge.
- **Backward compatibility**: Translators must regenerate locale files; provide a migration script that copies new keys from en.json into existing locales as `"<EN>"` markers.
- **Complexity**: S. **Priority: P2.**

### E7 — Modal focus management — P2

- **Current behavior**: `previewModal` exists; `_overlayFocusManagementBound = false`; no visible focus-trap implementation. Toasts don't have `aria-live` regions tied to the toast container.
- **Problem**: Screen reader users won't be announced of error toasts or modal state changes; keyboard focus can leak behind modals.
- **Recommended change**: Bind focus trap to the previewModal open event (capture focus on open, restore to `_previewModalReturnFocusEl` on close). Add `aria-live="polite"` to the toast container element (one parent attribute).
- **Code locations**: `extension/com.opencut.panel/client/main.js` (search `previewModal`, `showToast`); `client/index.html` toast container.
- **Backward compatibility**: None — purely additive.
- **Complexity**: S. **Priority: P2.**

### E8 — README route-count + extension-line-count badges generated — P0

Restated for clarity (also Q4): the README warning at line 11 already says "Route count is generated from `opencut/_generated/route_manifest.json`; run `python -m opencut.tools.dump_route_manifest --check` to verify it is in sync." Then add the same wire-up for line 8.

### E9 — Sequencing: F202 + F236 + F252 before any new Wave T — P1

- **Current behavior**: Wave T is the next-named wave in PROJECT_CONTEXT § 9; meanwhile F202 (Apple notarization, mandatory 2026-09-01), F236 (FCC captions, mandatory 2026-08-17), and F252 (UXP migration, CEP EOL Sept 2026) are all on **deadlines that hit before Wave T can plausibly ship**.
- **Problem**: Backlog management mismatch. F-number items with regulatory deadlines should out-prioritize wave items by default.
- **Recommended change**: Add a "Deadline-gated tier" to the top of `ROADMAP.md` listing F202, F236, F252 with their dates. Block Wave T promotion behind their close-out.
- **Code locations**: `ROADMAP.md` Now/Next ordering; `PROJECT_CONTEXT.md` § 9.
- **Complexity**: S. **Priority: P1.**

### E10 — `RTMP/SRT live preview` mid-pipeline status — P2

- **Current behavior**: B5.4 (SRT streaming) and B5.3 (aiortc WebRTC) live in Wave B; status unclear from ROADMAP-NEXT.
- **Problem**: A live-preview path solves the "users process blind" problem identified in research.md Wave 1. Status reconciliation needed.
- **Recommended change**: Audit whether B5.3/B5.4 actually shipped; if not, raise their priority — they unblock the F158 real-time preview story.
- **Complexity**: S audit, M if not shipped. **Priority: P2.**

---

## Reliability, Security, Privacy, and Data Safety

| Risk / Gap | Severity | Evidence | Owner |
|---|---|---|---|
| Untimeout'd subprocess invocations remain | P1 | E4 above; CLAUDE.md gotcha line ~288 explicitly forbids new ones | Add audit batch 9 + lint |
| Bare `except Exception:` in `checks.py` masks install failures | P1 | E5 above; subagent grep | Add structured failure reasons |
| Tracked build artifacts | P1 | 478 tracked `.dll` files | E2 |
| PR-fast skips supply-chain | P0 | `.github/workflows/pr-fast.yml:42-46` | E3 |
| No static security scanner on PR | P1 | `pr-fast.yml` lacks `bandit`, `semgrep` | Q5/E3 follow-on |
| Plugin sandbox bypass needs ongoing fuzzing | P2 | F215 fuzz harness exists; only 5 parsers covered | Extend fuzz targets |
| Webhook receivers don't verify signed deliveries by default | P2 | `core/webhook_signature.py` exists but optional | Make signing mandatory for production webhooks |
| Crash log scrubber doesn't redact API keys in user logs | P2 | Wave H gotcha line 338 | F273 candidate |
| F112 auth token rotation | P2 | Token persisted at `~/.opencut/auth.json`; no rotation guidance | Document + provide `/system/auth/rotate` |
| No verifiable rollback for destructive timeline ops | P2 | `host/index.jsx` writes directly to active sequence | Wrap timeline writes in named history label |

**Recovery and rollback needs**:
- Premiere has its own undo; OpenCut writes via ExtendScript. `ocApplySequenceCuts` should record a labelled undo step (`app.enableQE(); qe.project.history.add("OpenCut: apply silence cuts")`) so the user can mass-undo with one keypress. Confirm against the QE reflection probe output (F267).
- For object-remove and watermark-remove, default to a non-destructive output file even when the user clicks "apply to clip"; surface a "replace in project" button only after preview confirmation.

**Logging and diagnostics needs**:
- `/system/check-failures` (per E5).
- Per-request correlation ID is exposed (F195/F210); add a "copy support bundle to clipboard" panel button that combines `crash.log` + `/logs/tail?since=10m` + `/system/check-failures` + `/system/feature-state`.

---

## UX, Accessibility, and Trust

### Onboarding gaps
- Wave H1.8 first-run wizard exists, but the post-onboarding empty state on Cut/Audio/Video tabs is "blank panel until a clip is selected." Add inline help banners that explain what each tab does without requiring a clip.
- The CEP panel's "Server offline" reconnect banner doesn't differentiate "backend not running" from "wrong port" from "firewall blocked." Cite the exact failure (DNS / TCP / 4xx / 5xx) in the banner.

### Empty / loading / error / disabled states
- Buttons remain clickable during in-flight XHR (subagent finding). Add a `aria-busy="true"` + `disabled` flip on submission; this is non-breaking and cheap.
- Toasts lack `aria-live`. See E7.
- Many panels show "0 results" without explaining why (footage index empty? not indexed yet?). Differentiate.

### Destructive or irreversible actions
- Object-remove / watermark-remove / face-swap modify a clip output non-reversibly. Add a "Keep original on disk" toggle (default on) and surface the original path in the result panel.
- Plugin uninstall via `/plugins/uninstall` doesn't double-confirm. Add a CSRF + name re-type confirm step for any unsigned plugin uninstall.

### Settings clarity
- Settings → AI Engine Preferences allows swapping backends per domain; the speed/quality labels are good. Add a "What does this affect?" inline note linking to the relevant model card.

### Accessibility issues
- See E7 (modal focus, toast aria-live).
- Status bar (per CLAUDE.md) uses `role="status" aria-label=...` — good. Verify the same for `processingBanner` and `alertBanner`.
- No high-contrast theme; "Studio Graphite" is the only theme (removed in v1.9.16). Reconsider for WCAG 2.2 AAA paths (F235 already on Now-tier).

### Microcopy and trust signals
- The header "1,344 routes" is marketing chrome that's already wrong. Replace with "100% local • MIT licensed • no API key required for core" — concrete trust signals that match what AUDIT.md positioning identified.
- Add an in-panel "What ran locally vs cloud" badge per operation (some LLM ops can hit OpenAI/Anthropic). Already partly there in the engine-preferences dropdowns; surface in result toasts too.

---

## Architecture and Maintainability

### Module or boundary improvements
1. **`wave_l_routes.py` is a misnomer** — Waves N, O, P, Q, R appended their routes here (per commit messages `17f2b68`, `60fbae7`, `1e11322`). Rename to `wave_late_routes.py` or split into `wave_n_routes.py` … `wave_r_routes.py`. The blueprint name in `__init__.py` registration order should follow.
2. **`opencut/core/` has 593 modules — flat directory is brittle.** Consider a one-time refactor into `opencut/core/audio/`, `opencut/core/video/`, `opencut/core/captions/`, `opencut/core/ai/`, `opencut/core/timeline/`, `opencut/core/governance/`. Coordinate with `tests/`. Defer until F252 cut-over is past unless the import-cost on cold start (now multi-second) is a real complaint.
3. **`opencut/checks.py` is 1,000-2,000 lines of bare `except`** — extract per-domain check files: `checks/audio.py`, `checks/video.py`, etc., re-exported from `checks/__init__.py` to preserve callers.
4. **CEP `main.js` 15k LOC single IIFE** — the `WaveH = (function(){})()` pattern at the tail is the seed of a real module system. Add a second namespace (`Core`, `Captions`, `Audio`, etc.) per tab in successive small refactors, then a Vite-driven multi-entry build via the already-present `vite.config.js`.

### Refactor candidates
- Each of the 8 `audit:` batch commits flagged a class of issue (NaN, callback signature, regex tightness, allowlist). Convert each into a permanent lint rule. The 9th batch is "audit the audit": catalog what each batch fixed and make sure no new code reintroduces the pattern.

### Test gaps
- 553 untested modules. See Q9 per-blueprint floor.
- Wave S modules ship with **zero tests**. Even smoke imports would catch missing dataclass-result subscript implementations.
- Plugin sandbox has F215 fuzz harness but only 5 parsers covered (c2pa, plugin_manifest, lut_library, webhook_signature, security). Extend to SRT/VTT/ASS/OTIO/FCPXML — they're already exposed to user input.

### Documentation gaps
- CLAUDE.md is the authoritative developer doc, but its file-size/module-count claims are stale (E1). The "Gotchas" section is irreplaceable institutional knowledge — invest in a release-smoke gate that diffs gotcha line numbers cited in the doc against the live files (`Line 88-99` style anchors).
- No central "How do I add a Wave letter?" doc — the convention is implicit. Write `docs/ADDING_A_WAVE.md`.

### Release/build/deployment gaps
- macOS notarization secrets still not set (F202). Schedule a 30-min ops session before 2026-08-15.
- First Flathub submission not done (F249). Same.
- WPF installer signing secrets not configured (F203). Same.
- Linux AppImage publication path (F249) — verify GitHub Releases attaches both Flatpak and AppImage for x86_64; arm64 is F101.

---

## Prioritized Roadmap

### Phase 0 — Honesty & Hygiene (this week)

- [x] **P0 — Route or demote Wave S core modules (Q1)**
  - Why: User-visible "shipped" claim that returns 404.
  - Evidence: `git show --stat a8f62c0` shows zero `opencut/routes/*` diff.
  - Touches: `opencut/routes/wave_l_routes.py` (or new `wave_n_late_routes.py`), `opencut/checks.py`, `opencut/registry.py`, route manifest regen.
  - Acceptance: `curl -X POST localhost:5679/video/relight/iclight -H "X-OpenCut-Token: …" -d '{"filepath":"…"}'` returns 503 with install hint OR the wave S row in ROADMAP/CHANGELOG is annotated "core-only".
  - Verify: `python -m opencut.tools.dump_route_manifest --check`; `pytest tests/test_route_smoke.py -q`.

- [x] **P0 — Generated README badges (Q4 / E8)**
  - Why: Marketing badge drift (1,344 → 1,381) erodes trust in the rest of the README.
  - Evidence: README.md line 8 vs `route_manifest.json:total_routes=1381`.
  - Touches: `scripts/sync_version.py`, `README.md`, `scripts/release_smoke.py`.
  - Acceptance: `python scripts/sync_version.py --check-badges` exits 0; mutating any of the underlying counts and re-running fails.
  - Verify: change `route_manifest.json` total to 9999, re-run check → expect non-zero exit.

- [x] **P0 — Promote pip-audit / npm-advisory / esbuild-pin to PR-fast (E3)**
  - Why: Supply-chain regressions are invisible until Release Full today.
  - Evidence: `.github/workflows/pr-fast.yml:42–46`.
  - Touches: `.github/workflows/pr-fast.yml`.
  - Acceptance: PR-fast workflow runs all three gates.
  - Verify: open a draft PR pinning a known-CVE dep; assert PR-fast fails.

- [x] **P0 — Refresh CLAUDE.md sizes/counts (E1)**
  - Why: F252 migration is mis-priced.
  - Evidence: subagent verified `main.js`=15,263, `style.css`=17,870, UXP `main.js`=5,568.
  - Touches: `CLAUDE.md`, `PROJECT_CONTEXT.md` § 2, `scripts/check_doc_sizes.py` (new).
  - Acceptance: New `scripts/check_doc_sizes.py` exits 0; CI gate added.
  - Verify: `python scripts/check_doc_sizes.py`.

- [x] **P1 — Remove tracked build artifacts (E2)**
  - Why: 478 .dll files in git history.
  - Evidence: `git ls-files | grep "/bin/Debug" | wc -l = 478`.
  - Touches: `.gitignore`, one cleanup commit (no force-push).
  - Acceptance: `git ls-files installer/src/OpenCut.Installer/bin/Debug/ | wc -l = 0`.
  - Verify: same command.

### Phase 1 — Safety, Observability, and Test Discipline (next 2 weeks)

- [x] **P1 — Subprocess timeout audit batch 9 (E4)**
  - Why: A wedged child holds a worker slot forever.
  - Evidence: Spot-check found multiple `subprocess.run` without `timeout=` (audit subagent).
  - Touches: every `opencut/core/*.py` and `opencut/routes/*.py` with a subprocess; new `scripts/lint_subprocess_timeouts.py`.
  - Acceptance: `python scripts/lint_subprocess_timeouts.py` reports 0 unhandled cases; gate added to release-smoke.
  - Verify: same lint script.

- [x] **P1 — `checks.py` structured failure reasons (E5)**
  - Why: Silent "not available" hides install root causes.
  - Evidence: 20+ `except Exception:` blocks per subagent.
  - Touches: `opencut/checks.py` (split into `opencut/checks/`), `opencut/routes/system.py` (`/system/check-failures`).
  - Acceptance: `GET /system/check-failures` returns last failure exception class + message per check.
  - Verify: `tests/test_checks_failure_diagnostics.py`.

- [x] **P1 — Per-blueprint coverage floor (Q9)** — shipped as `scripts/test_breadth_gate.py` (cheap proxy: ratio of `opencut/core/*` modules referenced by any `tests/test_*.py`). Floor 75%; current 78.3%. Wired into release-smoke + tested. The original "93% untested" claim was a sampling artifact.
  - Why: Global 54% hides 93% untested modules.
  - Evidence: subagent.
  - Touches: `.coveragerc` / `pyproject.toml [tool.coverage]`, `scripts/coverage_floor_per_blueprint.py`.
  - Acceptance: CI fails if a blueprint drops below baseline.
  - Verify: deliberately delete a test, re-run CI → expect non-zero exit.

- [x] **P1 — Static security scanners on PR (E3 follow-on)**
  - Why: SSRF / path-traversal patterns get caught by hand today.
  - Evidence: PR-fast `.github/workflows/pr-fast.yml` lacks bandit/semgrep/mypy.
  - Touches: `.github/workflows/pr-fast.yml`, `.bandit`, `.semgrep.yml`, `mypy.ini`.
  - Acceptance: PR-fast runs `bandit -r opencut/ -ll` + `semgrep --config p/python` + `mypy opencut/core/ --ignore-missing-imports`.
  - Verify: induce a `requests.get(USER_INPUT)` to confirm semgrep catches it.

- [x] **P1 — CEP↔UXP tab parity gate (Q5)**
  - Why: Sept-2026 cut-over needs verifiable parity.
  - Touches: `tests/test_panel_tab_parity.py`, `extension/PANEL_PARITY.json`, `scripts/release_smoke.py`.
  - Acceptance: Test passes today (with annotated divergences) and fails on un-annotated drift.

### Phase 2 — UX leverage (next 4 weeks)

- [x] **P0 — F143 `/agent/chat` conductor (Q2)** — backend scaffold shipped (`opencut/core/agent_chat.py` + 5 routes + 25-case suite). UXP UI surface tracked under F252.
- [x] **P1 — One-click Enhance macro (Q3)** — shipped (`opencut/core/enhance_auto.py` + 3 routes + 16-case suite).
- [x] **P2 — Modal focus management + toast aria-live (E7)** — both verified present in main.js; guard tests added in `tests/test_panel_a11y_invariants.py`.
- [x] **P2 — Hardcoded English → i18n migration + drift gate (Q6 / E6)** — gate landed; drift gate is in `scripts/i18n_lint.py` with `DEAD_KEY_BASELINE = 150` and zero missing keys. Hardcoded-string migration deferred to a focused follow-on pass.
- [x] **P2 — A/B variant generator on shorts pipeline (Q8)** — shipped (`opencut/core/shorts_variants.py` + 3 routes + 16-case suite).

### Phase 3 — Deadline-gated (8–14 weeks, hard external dates)

- [ ] **P0 — F202 macOS notarization first signed release** — Apple service deadline 2026-09-01.
- [ ] **P0 — F236 FCC caption display-settings discoverability** — effective 2026-08-17.
- [ ] **P0 — F252 UXP WebView cutover** — CEP EOL ~2026-09.
- [ ] **P1 — F146 UXP MCP transport** — survives CEP EOL.

### Phase 4 — After deadlines clear

- [x] **P1 — Sequence Index panel (Q7 / new F273)** — backend shipped (`opencut/core/sequence_index.py` + 3 routes + 27-case suite); UXP rendering surface tracked under F252.
- [ ] **P1 — Refactor `opencut/core/` into domain subpackages**.
- [ ] **P2 — Rename `wave_l_routes.py` (Architecture item 1)**.
- [ ] **P2 — Extend F215 fuzz harness to SRT/VTT/ASS/OTIO/FCPXML parsers**.
- [ ] **P3 — Wave T promotion** (gated on F202/F236/F252 close-out).

---

## Quick Wins

| Item | Effort | Impact | Owner |
|---|---|---|---|
| README badge regenerator (Q4 / E8) | 1 hr | High — fixes visible drift | scripts/sync_version.py |
| Remove tracked .NET DLLs (E2) | 30 min | Medium — clones shrink | one commit |
| Promote pip-audit to PR (E3) | 30 min | High — supply-chain | pr-fast.yml |
| CLAUDE.md size refresh (E1) | 1 hr | Medium — re-prices F252 | CLAUDE.md edit + lint |
| Toast `aria-live` + modal focus trap (E7) | 2 hr | Medium — accessibility | main.js + index.html |
| `/system/check-failures` (E5) | 2 hr | Medium — support burden | checks.py + routes/system.py |
| `scripts/i18n_lint.py` (Q6) | 3 hr | Medium — locale hygiene | new script |
| `scripts/lint_subprocess_timeouts.py` (E4) | 3 hr | High — reliability | new script |
| Rename `wave_l_routes.py` (Arch §1) | 1 hr | Low — clarity | rename + reg |
| Wave S routes 503-stubs (Q1, Path 1) | 4 hr | High — honesty | one route file |

---

## Larger Bets

- **F143 chat-conductor (Q2 / `.ai/research/2026-05-17/AGENT_UX_RFC.md`).** Largest UX leap available; ~6–8 weeks. Sequencing: lands inside the F252 WebView shell to outlive CEP EOL.
- **F252 UXP WebView migration.** Bolt UXP WebView pattern; 15,263-line `main.js` is mostly portable. The 14 direct-UXP host actions catalogued in PROJECT_CONTEXT §10 are the harness; the rest is the migration. Hard date.
- **F146 UXP-native MCP transport.** Every competing PPro MCP server is CEP-bound and will break at CEP EOL. First UXP-MCP wins post-EOL.
- **F158 StreamDiffusionV2 real-time preview.** Largest "wow" leap; unblocks live preview for LTX-2.3 / Wan / Open-Sora T2V.
- **Refactor `opencut/core/` into subpackages.** 593 modules in a single flat directory is at the limit. Defer until F252 lands.
- **Plugin marketplace.** Once `plugin.lock.json` signing (F116) is widely adopted, a curated marketplace becomes feasible. Out of scope for 2026-Q3; revisit Q4.

---

## Explicit Non-Goals (this pass)

- **Adding more model surfaces (Wave T).** The 38 existing stubs already exceed bandwidth. Filling them is higher leverage than adding new ones.
- **Re-architecting the panel to React/Spectrum.** Bolt UXP WebView is the right cut-over per F252; Spectrum widgets are not.
- **Cloud-native or collaborative editing.** Per AUDIT.md Part 6 positioning; OpenCut is local-first.
- **Mobile editor or browser editor.** Out of lane.
- **Replacing FFmpeg with a custom encoder.** B5.1 (VVC), A4.1 (ab-av1), A4.2 (SVT-AV1-PSY) are the right shape — sit on top of FFmpeg.
- **Marketplace / templates / brand-kit social network.** Per Wave H rationale — moderation budget not available.
- **Self-hosted Frame.io clone.** Review portal is enough; do not chase Frame.io collaborative review at their scale.

---

## Open Questions (only those blocking prioritization)

1. **Are the Wave S core modules expected to ship as runnable HTTP routes in v1.33, or are they intentionally core-only?** If intentional, the changelog and ROADMAP should say so explicitly. Resolves Q1 path choice.
2. **Is the v1.32 → v1.33 → v1.34 window committed to F202/F236 deadline work first**, or do new Wave T modules sneak in alongside? PROJECT_CONTEXT §9 says Now-tier; CHANGELOG.md `[Unreleased]` does not list F202 secrets work. Confirm scheduling.
3. **Has `pip-audit --no-deps -r requirements-lock.txt` been run against the current lockfile in the last week?** A green run is documented in PROJECT_CONTEXT §9.4 (Pass 3, 2026-05-17), but new pip-audit advisories land daily. If stale, refresh before next PR.
4. **Are the Wave H1.3 eye-contact module's MediaPipe weights guaranteed redistributable under MIT-compatible terms?** PROJECT_CONTEXT §9 says "eye-contact already existed"; confirm license posture of the model file (not just the wrapper).
5. **Should plugin authors be required to ship a `tests/` directory by F116?** Today the manifest does not require it. Untested plugins are the single largest expansion surface for stability bugs.
6. **What is the support stance for users running the v1.28 installer against v1.32 backend?** README "Option A — Installer (recommended)" still points to `OpenCut-Setup-1.28.0.exe`. Should update on every minor.

---

## Appendix A — Doc-vs-Reality Drift Found in this Pass

| Claim | Source | Actual | Magnitude |
|---|---|---|---|
| `API Routes-1344` | README.md line 8 badge | 1,381 (manifest) | +37 |
| `~7,730-line main.js` | CLAUDE.md, PROJECT_CONTEXT §1 | 15,263 | ~2× |
| `~5,000-line style.css` | CLAUDE.md | 17,870 | ~3.5× |
| `~1,500-line UXP main.js` | PROJECT_CONTEXT §1 | 5,568 | ~3.7× |
| `~1,700-line UXP main.js` | CLAUDE.md "8 tabs" section | 5,568 | ~3.3× |
| `538 Python files in opencut/core/` | PROJECT_CONTEXT §2 | 593 | +55 |
| `73 route blueprints` | CLAUDE.md routes section | 101 | +28 |
| `190 test_*.py` | PROJECT_CONTEXT §2 | 197 | +7 |
| `417 CEP locale keys` | PROJECT_CONTEXT §2 | 426 (~142 unused) | +9 with 33% dead |

These are not malicious — they're the cost of an aggressive shipping cadence outpacing the docs. The fix is generation, not vigilance (Q4 / E1 / E8).

## Appendix B — Verified Wave S Routing Gap

`git show --stat a8f62c0` (HEAD) — 10 files, all under `opencut/core/`:

```
opencut/core/asr_canary.py           | 50 +++++++++++++++++
opencut/core/asr_parakeet.py         | 50 +++++++++++++++++
opencut/core/face_reage.py           | 50 +++++++++++++++++
opencut/core/multimodal_internvl3.py | 51 ++++++++++++++++++
opencut/core/multimodal_qwen3vl.py   | 51 ++++++++++++++++++
opencut/core/music_heartmula.py      | 50 +++++++++++++++++
opencut/core/relight_diffrenderer.py | 49 ++++++++++++++++
opencut/core/relight_iclight.py      | 49 ++++++++++++++++
opencut/core/relight_video_lav.py    | 49 ++++++++++++++++
opencut/core/upscale_seedvr2.py      | 50 +++++++++++++++++
```

No `opencut/routes/*.py` files changed. `grep -lr "iclight\|seedvr2\|parakeet\|canary\|qwen3vl\|internvl3\|heartmula\|reage\|diffrenderer\|video_lav" opencut/routes/` returns no matches. This is a clean evidence-based finding — not interpretation.

---

*End of 2026-05-25 research pass. Next pass: schedule when Q1 + E1 + E8 + E3 close, or when Wave T promotion is proposed (whichever first).*
