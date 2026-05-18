# OpenCut Research — CONTINUE FROM HERE (for Pass 27)

**This file's purpose:** if a future autonomous research session starts up, **read this first** before re-doing any of the work already on disk.

**Last update:** 2026-05-17 (after Pass 26; Passes 1-26 all ran on the same calendar day)
**Session state:** all mandated artefacts exist, Pass 4 ran full release-smoke successfully, Pass 5 closed F261/F262/F270, Pass 6 closed F264/F266, Pass 7 closed F199, Pass 8 closed F191/F197, Pass 9 closed F195, Pass 10 closed the repository-side F202 notarization tooling, Pass 11 closed F204 release SBOM upload, Pass 12 closed F207 installer FFmpeg manifest after an F205 coverage-measurement timeout, Pass 13 closed F208 OpenAPI contract validation, Pass 14 closed F209 MCP route consistency, Pass 15 closed F218 blueprint import-order stability, Pass 16 closed F219 SBOM completeness, Pass 17 closed F236 FCC caption display-settings tokens, Pass 18 closed F237 loudness standards metadata, Pass 19 closed F240 caption reading-speed profiles, Pass 20 closed F241 text-shaping CI/release gating, Pass 21 closed F243 UTF-8 no-BOM SRT policy, Pass 22 closed F244 Whisper confidence + human-review flags, Pass 23 wrapped up an interrupted F205 coverage reattempt without changing the coverage floor, Pass 24 closed F259/F251/F147/F131 in a single governance + migration quick-win batch, Pass 25 closed F137 MCP SDK pin + F139 caption translation SRT round-trip, and Pass 26 closed F126 OTIO AAF adapter pin + F181 UV trampoline bootstrap fallback + F185 features.md banner. This file documents deferred research/product work for a future Pass 27+, not a broken or incomplete research run.

---

## 1. State at hand-off

- **Repo branch:** `main`, expected 46 commits ahead of `origin/main` after the Pass-23 wrap-up commit. Push to `SysAdminDoc/OpenCut` is blocked by local GitHub auth (`MavenImaging` lacks permission for `SysAdminDoc/OpenCut`).
- **Last shipped version:** v1.32.0 (light theme + appearance toggle, 2026-05-09).
- **Live counts:** 1,362 routes / 101 blueprints / 525 core modules / 143 test files / 47 model cards / 117 public `check_*` probes (86 `check_*_available`) / 84 `FeatureRecord` entries / 39 MCP tools / 30 OpenAPI-typed endpoints.
- **F-numbers in ledger:** F001-F272 (Pass 1 added F121-F190, Pass 2 added F191-F260, Pass 3 added F261-F272).
- **Wave letters in ledger:** A-M shipped; N-T planned in ROADMAP.md but not yet F-number-tiered (covered by F180).

### Pass 27 entry point

1. **Push checkpoint commits** once GitHub auth is available on this machine. Pushing remains blocked by the `SysAdminDoc/OpenCut` vs `MavenImaging` credential mismatch.
2. **Continue the remaining Now queue.** F126, F137, F139, F147, F181, F185, F251, and F259 are now closed (Passes 24-26). F182 (gh issue seeder run) is blocked on GitHub auth. F183 (log-file cleanup) is structurally closed — `.gitignore` already covers `*.log` and no log files are git-tracked. F205 still needs a runner where coverage can finish. Remaining Pass-1 Now items that are local-effort-feasible: **F123** (pydub replacement / audioop-lts shim, M), **F128** (FFmpeg filter regression suite, M), **F140** (C2PA 2.3 sidecar bump, M). Network-required: F121 (Pillow 12.2), F122 (flask-cors 6.x), F133 (onnxruntime ≥1.25), F135 (whisperx 3.8.5). Larger Pass-1 Now items: F149/F162/F163/F167/F169 (each requires a model install + integration work).
3. **Complete F179** full `features.md` reconciliation; this remains the largest knowledge debt.
4. **Run a Python 3.10/3.11/3.13 install matrix** for `[all]`; this cannot be fully proven from this VM's single Python 3.12 runtime.

### Pass 26 checkpoint

| Item | Status |
|---|---|
| F126 | **DONE** — `pyproject.toml` `[otio]` and `[all]` extras now pin `otio-aaf-adapter>=0.6,<1`. Tests in `tests/test_otio_aaf_adapter_pin.py` (5) pin the extras shape, the version range, and the existing `check_aaf_available` two-tier probe. |
| F181 | **DONE** — `_resolve_python_for_subprocess()` in `scripts/bootstrap_check.py` probes `sys.executable` then falls back to `shutil.which("python")`/`python3`/`py`. `check_version_sync` translates `FileNotFoundError`/`OSError`/`TimeoutExpired` into actionable remediation hints. 4 new tests in `tests/test_bootstrap_check.py` cover happy path, fallback, trampoline hint, timeout. |
| F185 | **DONE** — `features.md` opens with the aspirational-catalogue banner + cross-links + precedence rule. `tests/test_features_md_banner.py` (4) pins banner text, links, status line, and precedence. |
| F183 | **STRUCTURALLY CLOSED** — `.gitignore` already has `*.log`; `git ls-files | grep '\.log$'` returns empty. Working-tree `pt.log` / `build*.log` / `pytest-*.log` are untracked. No git-tracked log file remediation needed. |
| Focused validation | PASS — `17 passed` across F126/F181/F185 tests. |
| Release smoke | PASS — all 15 chained gates green; `pytest-fast` reports `51 gate tests passed`. Ruff `opencut/` scope clean. |
| Files to review | `pyproject.toml`, `scripts/bootstrap_check.py`, `scripts/release_smoke.py`, `features.md`, `tests/test_otio_aaf_adapter_pin.py`, `tests/test_bootstrap_check.py`, `tests/test_features_md_banner.py`, ROADMAP.md v4.29 section, PROJECT_CONTEXT.md, and this file. |

### Pass 25 checkpoint

| Item | Status |
|---|---|
| F137 | **DONE** — `pyproject.toml` now pins `mcp>=1.26,<2`. MCP 2.x is a pre-alpha FastMCP → McpServer rewrite that would break `opencut.mcp_server.MCP_TOOLS`. `tests/test_mcp_sdk_pin.py` (3 tests) blocks any future loosening below 1.26 or above the 1.x line. |
| F139 (SRT round-trip) | **DONE** — `POST /captions/translate` now accepts `srt_path` (file on disk) or `srt_content` (raw SRT text) in addition to the legacy `segments` list. SRT input is parsed with the existing `_parse_srt_content` helper, translated through the same auto / NLLB / SeamlessM4T backends, and written back as SRT when the caller passes `srt_path`, `output_srt=true`, or `srt_output_path`. The F243 `srt_legacy_bom` toggle is honoured on the written file. `tests/test_captions_translate_srt.py` (16 tests) covers validator, helper round-trip, segments path, SRT-content path, SRT-path read with legacy BOM round-trip, and the empty-input 400. |
| Focused validation | PASS — `19 passed` across `tests/test_mcp_sdk_pin.py` + `tests/test_captions_translate_srt.py`. |
| Release smoke | PASS — all 15 chained gates green; `pytest-fast` reports `50 gate tests passed`. Ruff clean. |
| Files to review | `pyproject.toml`, `opencut/routes/captions.py`, `scripts/release_smoke.py`, `tests/test_mcp_sdk_pin.py`, `tests/test_captions_translate_srt.py`, plus roadmap/research docs. |

### Pass 24 checkpoint

| Item | Status |
|---|---|
| F259 | **DONE** — `docs/UXP_MACOS_HTTP.md` documents the four shipped workarounds (port autodiscovery, fetchWithTimeout, exponential backoff, WS retry) and the deferred auto-HTTPS sidecar plan (sequenced with F146/F252). `extension/com.opencut.uxp/uxp-api-notes.md` cross-links the new doc. `tests/test_uxp_macos_http.py` pins manifest port allowlist + doc invariants. |
| F251 | **DONE** — `opencut/tools/adobe_premierepro_versions.py` fetches `@adobe/premierepro` from `registry.npmjs.org` via stdlib urllib, snapshots dist-tags + last 10 releases. `opencut/_generated/adobe_premierepro_versions.json` committed as the drift reference (current: `latest=26.2.0`, `beta=26.3.0-beta.67`). `.github/workflows/adobe-premierepro-versions.yml` runs Mondays at 06:00 UTC and opens/updates a labelled tracking issue on drift. Release smoke `adobe-premierepro-versions` step surfaces drift as a `warn`, not a `fail`. |
| F147 | **DONE** locally — `opencut/_generated/mcp_server_registry.json` is generated from the live 39-tool `MCP_TOOLS` catalogue via `opencut.tools.dump_mcp_registry_manifest`. `docs/MCP_SERVER.md` documents transports, install snippets, and the upstream `modelcontextprotocol/servers` PR procedure. Release smoke `mcp-registry` step asserts no drift. The actual upstream PR remains as a credentialed external action. |
| F251/F147 release-smoke wiring | `StepResult.status` now supports `warn` (info-only); the new `adobe-premierepro-versions` step uses it so registry drift is visible without blocking releases. |
| F131 | **DONE** — `extension/com.opencut.panel/scripts/check-esbuild-pin.mjs` parses `npm ls esbuild --all --json`, asserts every resolved instance is `>=0.25.0`, exits non-zero on violation, exposes a `--json` mode. `npm run audit:esbuild` is the panel entry point. Release smoke `esbuild-pin` gracefully skips when Node or `node_modules` is absent. |
| Focused validation | PASS — `41 passed` across `tests/test_uxp_macos_http.py` (9), `tests/test_adobe_premierepro_versions.py` (14), `tests/test_mcp_registry_manifest.py` (11), and `tests/test_esbuild_pin.py` (7). |
| Release smoke (without pytest-fast, pip-audit, npm-advisory, panel-source) | PASS — all 13 chained gates green. |
| Ruff | PASS — `python -m ruff check opencut/ --select E,F,I --ignore E501,E402` clean. |
| Files to review | `docs/UXP_MACOS_HTTP.md`, `docs/MCP_SERVER.md`, `opencut/tools/adobe_premierepro_versions.py`, `opencut/tools/dump_mcp_registry_manifest.py`, `opencut/_generated/adobe_premierepro_versions.json`, `opencut/_generated/mcp_server_registry.json`, `.github/workflows/adobe-premierepro-versions.yml`, `extension/com.opencut.panel/scripts/check-esbuild-pin.mjs`, `extension/com.opencut.panel/package.json`, `extension/com.opencut.uxp/uxp-api-notes.md`, `scripts/release_smoke.py` (3 new steps + warn tier), `tests/test_uxp_macos_http.py`, `tests/test_adobe_premierepro_versions.py`, `tests/test_mcp_registry_manifest.py`, `tests/test_esbuild_pin.py`, `ROADMAP.md` (v4.27 section), `PROJECT_CONTEXT.md`, and this file. |

### Pass 23 checkpoint

| Item | Status |
|---|---|
| F205 | **OPEN** — second local CI-style coverage attempt was interrupted after 2,206.6 seconds (36m46s). |
| Partial evidence | `dist\coverage-f205.json` parsed successfully but is ignored and incomplete: coverage.py 7.14.0, 126,421 statements, 65,890 covered, 60,531 missing, 52.1195% reported coverage across 670 files; SHA256 `63DD45BF6C617BB05A7944911DEFF735A528F37F96CAD4CCC10F6E93CF59A6F9`. |
| Cleanup | Stopped one leftover `python.exe -m pytest tests sidecar/tests -q` process, then removed stale `.coverage` and `dist\coverage-f205.json` after recording their evidence. |
| Decision | Do not update `.github/workflows/build.yml`; `--cov-fail-under=50` remains the only valid CI floor until a full coverage command completes. |
| Evidence file | `.ai/research/2026-05-17/F205_INTERRUPTED_COVERAGE_NOTE.md`. |

### Pass 22 checkpoint

| Item | Status |
|---|---|
| F244 | **DONE** — Whisper transcript segments now carry ASR confidence, language confidence, Hindi/Arabic review flags, and stable low-confidence review reason codes across route, cache, state, JSON export, edited export, and CLI surfaces. |
| Focused validation | PASS — `12 passed` for `tests/test_caption_language_confidence.py tests/test_captions_regressions.py tests/test_srt_encoding.py`. |
| Release smoke | PASS — all 14 steps green; pytest-fast `300 passed`. |
| Files to review | `opencut/core/captions.py`, `opencut/routes/captions.py`, `opencut/export/srt.py`, `opencut/polish_state.py`, `opencut/cli.py`, `tests/test_caption_language_confidence.py`, `scripts/release_smoke.py`, plus roadmap/research docs. |

---

## 2. Artefacts on disk after Pass 4 (`.ai/research/2026-05-17/`)

| File | Status | Lines (approx) |
|---|---|---|
| `STATE_OF_REPO.md` | Pass 1 | 300 |
| `MEMORY_CONSOLIDATION.md` | Pass 1 | 270 |
| `COMPETITOR_MATRIX.md` | Pass 1 | 260 |
| `DATASET_MODEL_INTEGRATION_REVIEW.md` | Pass 1 | 330 |
| `SECURITY_AND_DEPENDENCY_REVIEW.md` | Pass 1 | 300 |
| `FEATURE_BACKLOG.md` | Pass 1 (F121-F190) | 190 |
| `PRIORITIZATION_MATRIX.md` | Pass 1 + Pass 2 §6.5 | 230 |
| `SOURCE_REGISTER.md` | Pass 1 + Pass 2 section | 340 |
| `RESEARCH_LOG.md` | Pass 1 + Pass 2 section | 320 |
| `CHANGESET_SUMMARY.md` | Pass 1 + Pass 2 + Pass 3 + Pass 4 sections | 260+ |
| **`ROUTE_READINESS_AUDIT.md`** | **Pass 2 (new)** | 220 |
| **`INSTALLER_AUDIT.md`** | **Pass 2 (new)** | 200 |
| **`TEST_COVERAGE_GAPS.md`** | **Pass 2 (new)** | 230 |
| **`FEATURES_RECONCILIATION.md`** | **Pass 2 (new)** | 220 |
| **`FEATURE_BACKLOG_ADDENDUM.md`** | **Pass 2 (new, F191-F260)** | 230 |
| **`LIVE_VERIFICATION.md`** | **Pass 3 + Pass 4 release-smoke validation** | 300+ |
| **`CEP_UXP_PARITY_MATRIX.md`** | **Pass 3** | 220 |
| **`AGENT_UX_RFC.md`** | **Pass 3** | 280 |
| **`MARKET_POSITIONING.md`** | **Pass 3** | 200 |
| **`CONTINUE_FROM_HERE.md`** | **this file** | ~350 |

**At repo root:** `PROJECT_CONTEXT.md` (canonical context, Pass 1-9 updates), `ROADMAP.md` (v4.3-v4.12 sections), `AGENTS.md` (pointer added), `CLAUDE.md` (pointer added, gitignored).

---

## 3. Historical deferred work for Pass 3 (mostly closed)

This section is preserved because Pass 3/4 used it as the checklist. See §7 and §10 for what is now closed and what remains.

### 3.1 Quick wins (S effort)

1. **Run `python -m opencut.tools.dump_route_manifest --check`** to verify the cached 1,362-route figure against live `url_map`. Possible drift after future route commits.
2. **Run `python scripts/release_smoke.py --json`** end-to-end and capture which steps actually fail. The v4.3 audit listed several failing steps that may have been fixed by the F099/F098/F112 commits.
3. **Verify cross-platform launcher files exist**: `OpenCut-Server.command` (macOS), `OpenCut-Server.sh` (Linux). Pass 2 inferred from Wave I but did not list them. Run `ls Z:/repos/OpenCut | grep Server` and `chmod +x` if needed.
4. **Read `extension/com.opencut.uxp/uxp-api-notes.md`** — internal CEP-vs-UXP comparison the team maintains. Cross-reference against Pass 2 UXP subagent §1-§10 findings.
5. **Walk `tests/fuzz/test_parser_fuzz.py`** — audit which validators have fuzz coverage vs F215's 8 proposed targets.
6. **Inspect `opencut/preflight.py`** (180 lines) and **`opencut/workers.py`** — Pass 2 didn't read these. Likely covers the GPU semaphore + WorkerPool architecture.
7. **Inspect `opencut/journal.py`** — Operation Journal feature referenced in CLAUDE.md, not deeply understood. Has its own blueprint (`journal` — 5 routes per manifest).
8. **Inspect `tests/conftest.py`** to understand the Flask test fixture pattern + CSRF fixture.
9. **Run a live `pip-audit`** against `requirements-lock.txt` to verify the F094 burn-down stayed clean.
10. **Run `npm audit --json` in `extension/com.opencut.panel`** to verify F095 mitigations still hold.

### 3.2 Medium effort

1. **Complete the F179 features.md reconciliation** — Pass 2 sample-walked 40 entries; ~370 remain. Estimated 1-2 days. Output: `features_status.md` companion with `[shipped]` / `[planned-F###]` / `[planned-W###]` / `[rejected]` / `[unclear]` markers per entry. Likely surfaces another 10-20 F-numbers.
2. **Walk CLAUDE.md lines 500-1509** — Pass 1 sampled 1-300, Pass 2 sampled 300-500. Lines 500-1509 are mostly v1.18.0 and earlier change history. Lower-priority for forward planning but useful for confirming the SHIPPED status of older features.
3. **Catalogue CEP-only routes** (F198 *Next* tier) — Pass 2 estimated ~30 routes are ExtendScript-bound. A precise list requires walking `extension/com.opencut.panel/host/index.jsx` + every route that calls into it. Per-route UXP-replacement plan.
4. **Generate `opencut/_generated/api_aliases.json`** (F199 *Now* tier) — **DONE in Pass 7**; live result is 15 true aliases + 218 canonical `/api` routes.
5. **Live PyPI / npm install-matrix check** — `pip install -e ".[all]"` on Python 3.10 / 3.11 / 3.12 / 3.13; `npm ci` in `extension/com.opencut.panel`. Catch silent-dead packages (e.g. `realesrgan>=0.3,<1` may no longer install on Py3.13).

### 3.3 Larger investigations

1. **Audit Codex / Cursor / Copilot patterns for video editing agents** — Pass 2 covered Underlord, FireRed, vibeframe. Cursor's IDE-agent pattern for video editing (if it exists) wasn't surveyed; could be a model for OpenCut's F143 conductor.
2. **Survey commercial NLE plugin pricing as a market-fit signal** — the Pass-1 competitor matrix listed pricing but didn't analyse it. Which Premiere extension prices correlate with which feature sets? Useful for the F088 review-bundle and F143 chat-conductor positioning.
3. **Adobe Premiere 26.3+ beta release notes weekly** — pin a CI step (F251) that diffs `@adobe/premierepro@beta` typings week-over-week. Once shipped, the same script could be run by Pass 3 to capture what's changed since 2026-05-07 (when 26.3.0-beta.67 was published).
4. **Survey Frame.io V4 webhook payload shape against Frame.io OSS competitor schemas** — Pass 2's Frame.io subagent §3 gave a rough consensus comment data model. A more rigorous spec would let F225 (OTIO Marker anchor) carry a translation table.
5. **Profile the F176 eval dataset bundle download size + per-dataset license verification** — Pass 2 listed 17 datasets; Pass 3 should check sizes (some are 10s of GB) and licence boilerplate per dataset.

### 3.4 Strategic decisions awaiting RFC

These are *Now*-tier items per the v4.3 audit that Pass 1 + Pass 2 reaffirmed but the maintainer has not yet decided:

1. **F127 RFC — Python 3.10 floor + Transformers v5 cascade.** Decision required before F124 (basicsr replacement), F125 (audiocraft isolation), F134 (pyannote 4.x), F136 (scenedetect 0.7) can be sequenced.
2. **F161 Under-Consideration — UXP Hybrid Plugin sidecar RFC.** Decision required before F253 (drag-out + QE-equivalent ops .uxpaddon) can be sequenced.
3. **F143 design space RFC — `/agent/chat` conductor.** Decision required on: UI surface (CEP+UXP), LLM provider routing, timeline diff representation, post-turn self-review semantics, F145 Skills SDK packaging shape.
4. **F200 — WPF-vs-Inno installer policy.** Two installers ship today. Decide which is recommended; document; consider retiring the other.
5. **F252 — Bolt UXP + WebView UI commitment vs Spectrum widget rewrite.** Pass 2 strongly recommends WebView UI but the maintainer may have other reasons.

---

## 4. Known limitations of the 2026-05-17 run

1. **No live `pytest` / `release_smoke` / `pip-audit` runs.** All findings are static analysis + manifest reads + external research. The dirty working tree's `helpers.py` finally-block change in particular should be exercised by F216 (concurrent job-cancellation race test).
2. **No commit of the 7-file dirty hardening batch.** User authorisation was not given. Recommendation stands as F138; the diff in `STATE_OF_REPO.md §6` documents the change set.
3. **No push to `SysAdminDoc/OpenCut`.** Auth blocked from this VM. 25 commits await another push origin.
4. **Closed-source benchmarks** (Hailuo, Seedance, Kling, Veo, Sora, Wan 2.5/2.6) cannot be verified; treated as opaque comparisons.
5. **Wan 2.7 weights publication status** — Apache 2.0 announced but a definitive "weights live on HF" confirmation wasn't found. F165 remains gated.
6. **`createCaptionTrack` UXP API** — Pass 2 confirmed it is **not** in `@adobe/premierepro@26.3.0-beta.67` typings. Pass 1's F186 Adobe gap report is correct as filed.
7. **`ProjectConverter.importFromFinalCutProXML` and `importFromOpenTimelineIO`** — Pass 2 discovered these were **removed** in the beta typings (versus shipping in 26.2.0). Need a new F-number (F261) tracking Adobe re-landing these as round-trip surfaces, or a Hybrid Plugin replacement (F253).

---

## 5. Suggested Pass-3 entry point

If the next research session opens fresh with the same prompt:

1. Read `PROJECT_CONTEXT.md` (canonical, ~280 lines) + this file (`CONTINUE_FROM_HERE.md`, ~250 lines). That's enough to understand state.
2. Skim the Pass-2 §3.1 quick-wins above. If any of them look high-value, knock them out first — they're 5-15 min each.
3. Pick **one** of §3.2's medium items or §3.3's larger investigations as the Pass-3 deep dive. The full features.md F179 reconciliation is probably the highest-value single item.
4. If nothing in §3 fits the session window, do a **third research wave** focused on whatever wave letters (N-T) are still un-implemented when you read this. They were ROADMAP.md-named pre-v4.3 and are explicitly waiting on F180 to be re-tiered through the F-number lens.

---

## 6. Decision authority

These artefacts are advisory. The OpenCut maintainer (`SysAdminDoc`) makes final tier-placement and shipping-cadence decisions. Future research passes should not assume that v4.5's Pass-2 F-numbers F191-F260 are committed to the live roadmap until they appear in `CHANGELOG.md` for a shipped version.

The ROADMAP.md v4.5 section is **proposed**, not enacted. Same applies to all Pass-1 / Pass-2 artefacts.

---

## 7. Pass 3 update (same day, third autonomous research run)

Pass 3 closed several items from §3.1 and §3.4 above, and surfaced one real shipped-vs-actual ledger discrepancy.

### What Pass 3 closed

| Pass 2 deferred item | Pass 3 status |
|---|---|
| §3.1.1 Run `dump_route_manifest --check` | ✅ Ran; 1,359 routes / 101 blueprints, no drift |
| §3.1.3 Verify cross-platform launcher files exist | ❌ Confirmed **missing**; promoted to F261 (Now) |
| §3.1.4 Read `extension/com.opencut.uxp/uxp-api-notes.md` | ✅ Read; 76-line file; minor URL typo (F262) |
| §3.1.5 Walk `tests/fuzz/test_parser_fuzz.py` | ✅ First 100 lines; confirmed 5 documented fuzz targets |
| §3.1.6 Inspect `preflight.py` + `workers.py` | ✅ preflight full read (180 lines); workers first 100 lines |
| §3.1.7 Inspect `journal.py` | ✅ First 80 lines; 6 valid actions, 4 revertible |
| §3.1.8 Inspect `tests/conftest.py` | ✅ Full read (84 lines); clean Flask test fixture |
| §3.1.9 Run live pip-audit | ✅ Ran; "No known vulnerabilities found" — F094 burn-down current |
| §3.1.10 Run live `npm audit --json` | ✅ Ran; 1 moderate Vite path-traversal matches F095 waiver |
| §3.2.3 Catalogue CEP-only routes (F198) | ✅ `CEP_UXP_PARITY_MATRIX.md` — 18 JSX functions, only 2 truly CEP-only |
| §3.3.1 Cursor/Copilot patterns for video editing | ✅ Pass-3 IDE-agent subagent returned; `AGENT_UX_RFC.md` adopts converged pattern |
| §3.3.2 Commercial NLE pricing market-fit signal | ✅ Pass-3 NLE-pricing subagent returned; `MARKET_POSITIONING.md` quantifies "$1,400/yr replaced" |
| §3.4.3 F143 design space RFC | ✅ `AGENT_UX_RFC.md` is the deliverable |

### What Pass 3 deferred to Pass 4

| Item | Why |
|---|---|
| §3.1.2 Run `python scripts/release_smoke.py --json` end-to-end | F098 runner not executed in Pass 3 (too long for the session window; would have run pip-audit + ruff + pytest + npm-audit serially). Run in Pass 4. |
| §3.2.1 **Complete F179 features.md reconciliation** (370 remaining entries) | Largest single deferred item across all 3 passes. Still 1-2 days. Pass 4 should pick this up if no shipping-blocker arises. |
| §3.2.2 Walk CLAUDE.md lines 500-1509 | Pass 3 read lines 300-500. Lines 500-1509 are mostly v1.18.0 and earlier change history; lower priority. |
| §3.2.4 Generate `opencut/_generated/api_aliases.json` | DONE in Pass 7. |
| §3.2.5 Live PyPI / npm install-matrix check | Needs Python 3.10/3.11/3.13 environments; only Python 3.12 available on this VM. Defer until a CI-runner can do the matrix. |
| §3.3.3 Adobe Premiere 26.3+ beta release notes weekly diff | F251 — proposed as a CI step, not a one-shot research pass. Schedule the CI step rather than a Pass-4 research run. |
| §3.4.1 F127 RFC — Python 3.10 floor + Transformers v5 cascade | Strategic decision waiting on maintainer input. Not a research task. |
| §3.4.2 F161 RFC — UXP Hybrid Plugin sidecar | Likewise. |
| §3.4.4 F200 — WPF-vs-Inno installer policy | Likewise. |
| §3.4.5 F252 — Bolt UXP + WebView UI commitment | Pass 3 narrowed the scope (XL→L) and provided sub-phases; maintainer decision remains. |

### New deferrals Pass 3 surfaced

| Item | Why |
|---|---|
| **Live UDT verification of the 14 low-risk JSX→UXP ports (F267)** | Requires running Premiere with UDT — no automation. Pass 4 could draft the test plan; actual runs need a human in front of Premiere. |
| **Implement F261 (ship the missing macOS `.command` + Linux `.sh` launchers)** | Trivial S-effort code change, not research. ~10 lines of shell script + `chmod +x` + add to release packaging. Maintainer can ship in next dep-bump release. |
| **Implement F270 (README marketing copy refresh with "$1,400/yr" lead)** | Trivial S-effort doc change, not research. Recommended copy in `MARKET_POSITIONING.md` §7. |
| **Adobe Exchange storefront listing (F268)** | Requires Adobe developer account + storefront submission process. Out of scope for research; maintainer action. |

---

## 8. Pass-3 entry-point synthesis for Pass 4

If a Pass 4 opens:

1. Read `PROJECT_CONTEXT.md` (~310 lines after Pass 3) + this file (~340 lines after Pass 3). That's enough state.
2. Three trivial maintainer wins available **before** any research:
   - **F261** — write `OpenCut-Server.command` + `OpenCut-Server.sh` (15 minutes)
   - **F262** — fix the uxp-api-notes URL typo (1 minute)
   - **F270** — paste the proposed README lead from `MARKET_POSITIONING.md` §7 (5 minutes)
3. If Pass 4 has time for research, the highest-value deferred items are:
   - **§3.2.1 F179 features.md reconciliation** (1-2 days, largest knowledge debt) — emits `features_status.md` companion
   - **§3.2.4 F199 api_aliases.json** (DONE in Pass 7)
   - **Walk CLAUDE.md lines 500-1509** (still unread; ~45 min) — extract any remaining patterns
4. **Strategic decisions awaiting maintainer**: F127 (Py 3.10 floor), F161 (Hybrid Plugin), F200 (WPF vs Inno), F252 (UXP migration commit). All four have full RFC text in Pass-2 + Pass-3 artefacts; ready for maintainer review.

---

## 9. State at hand-off (historical end-of-Pass-3 snapshot)

- **Historical repo branch state:** `main`, 25 commits ahead of `origin/main`, dirty working tree (7 modified files, uncommitted). Pass 4 superseded this by validating the hardening batch and preparing the local checkpoint commit.
- **Last shipped version:** v1.32.0.
- **Live verification results:** F099/F096/F093/F094 all PASS; npm audit at expected waived-Vite level; cross-platform launchers gap CONFIRMED (real shipping-vs-actual discrepancy).
- **F-numbers in ledger:** F001-F272 (Pass 1 added F121-F190, Pass 2 added F191-F260, Pass 3 added F261-F272).
- **CEP-EOL exposure**: 2 of 18 JSX functions truly CEP-only. F252 + F253 effort revised XL→L; comfortably inside the Sept 2026 window.
- **`/agent/chat` conductor**: design RFC complete (`AGENT_UX_RFC.md`); ready to ship v1.36 if F252 lands v1.34-v1.35.
- **Market positioning**: "$1,400/yr replaced" quantified; Mister Horse distribution model recommended.

---

## 10. Pass 4 update (same day, release-smoke and commit-prep)

Pass 4 closed the biggest remaining verification gap: the full release-smoke runner now passes.

### What Pass 4 closed

| Item | Status |
|---|---|
| Run `python scripts/release_smoke.py --json` end-to-end | **PASS** after safe Ruff cleanup |
| Targeted hardening test slice | **PASS** — `119 passed` |
| Release-smoke pytest-fast | **PASS** — `232 passed` |
| Release-smoke Ruff gate (`E,F,I`) | **PASS** after safe unused-import/import-order fixes in `opencut/` and `scripts/` |
| `pip-audit -r requirements-lock.txt` | **PASS** — no known vulnerabilities |
| npm advisory state | **PASS** in release-smoke allow-list step; raw `npm audit --json` still shows the known moderate Vite `.map` advisory that F095 documents |
| `npm view @adobe/premierepro version dist-tags --json` | Confirmed `latest=26.2.0`, `beta=26.3.0-beta.67` |

### Pass 13 entry point

1. **Push checkpoint commits** once GitHub auth is available on this machine.
2. **Continue the F191-F260 Now queue.** F205 remains open but needs a reliable long-running coverage measurement. F236, F237, F240, F241, F243, and F244 are closed; the remaining Now items are F205, F251, and F259. F251 and F259 likely need refreshed Adobe/UXP verification before implementation.
3. **Complete F179** full `features.md` reconciliation; this remains the largest knowledge debt.
4. **Run a Python 3.10/3.11/3.13 install matrix** for `[all]`; this cannot be fully proven from this VM's single Python 3.12 runtime.

### Current limitations

- No full cross-version Python install matrix.
- No Premiere UDT runtime verification of the 14 low-risk JSX to UXP ports.
- Push was attempted and failed with `remote: Permission to SysAdminDoc/OpenCut.git denied to MavenImaging.` Local auth still needs to be fixed outside the repo.
- The raw `npm audit --json` output still reports the moderate Vite advisory because the repo intentionally allows it below the release-smoke threshold; keep F095/`docs/NODE_ADVISORIES.md` as the disposition.

---

## 11. Pass 5 update (same day, launcher/docs implementation)

Pass 5 closed the three smallest Pass-3 Now items and left the larger research/development queue intact.

### What Pass 5 closed

| Item | Status |
|---|---|
| F261 | **DONE** — added `OpenCut-Server.command` and `OpenCut-Server.sh`; the shell launcher sets `OPENCUT_HOME`, handles bundled/system Python 3.9+, bundled FFmpeg, bundled model env vars, and starts `python -m opencut.server`. |
| F262 | **DONE** — fixed `extension/com.opencut.uxp/uxp-api-notes.md` sample repo URL to `AdobeDocs/uxp-premiere-pro-samples`. |
| F270 | **DONE** — README lead now uses the "$1,400/year" subscription-replacement story and Quick Start names the macOS/Linux launchers. |

### Validation after Pass 5

| Command | Result |
|---|---|
| `git diff --check` | PASS |
| `python scripts/release_smoke.py --json` | PASS — bootstrap, version-sync, route-manifest, model-cards, license-gate, roadmap-lint, Ruff, pytest-fast (`232 passed`), pip-audit, npm-advisory, and panel-source all green |

### Remaining immediate work

- Pass-3 Now items F261, F262, F264, F266, and F270 are closed locally.
- F179 remains the largest knowledge debt.
- Cross-platform launcher runtime verification still needs macOS/Linux CI or local runtime coverage (related to F211).
- Push is blocked by GitHub auth: `git push origin main` failed with `remote: Permission to SysAdminDoc/OpenCut.git denied to MavenImaging.` / HTTP 403. The local commits are valid; pushing needs credentials with write access to `SysAdminDoc/OpenCut`.

---

## 12. Pass 6 update (same day, F264/F266 implementation)

Pass 6 closed the remaining Pass-3 Now items.

### What Pass 6 closed

| Item | Status |
|---|---|
| F264 | **DONE** — `check-advisories.mjs --json` emits a stable machine-readable report; `scripts/release_smoke.py` parses it and fails closed on malformed JSON, non-`ok` status, or unwaived advisories. |
| F266 | **DONE** — `docs/UXP_MIGRATION.md` now documents `ocAddNativeCaptionTrack` and `ocQeReflect` as the two CEP residuals, keeps native caption track creation as the Hybrid Plugin target, and marks QE reflection as retire/replace-by-use-case. |

### Validation after Pass 6

| Command | Result |
|---|---|
| `node scripts/check-advisories.mjs --json` | PASS — one Vite advisory allowed, zero unwaived |
| `python scripts/release_smoke.py --only npm-advisory --json` | PASS — machine-readable advisory path reports `1 allowed` |
| `python -m pytest tests/test_release_smoke.py tests/test_node_advisories.py tests/test_uxp_migration_docs.py -q` | PASS — `20 passed` |
| `node --check extension/com.opencut.panel/scripts/check-advisories.mjs` | PASS |
| `python scripts/release_smoke.py --json` | PASS — all 11 steps green; pytest-fast `232 passed` |

### Remaining immediate work

- F199 is closed locally. Live correction: 233 `/api/*` routes exist, but only 15 are true aliases; 218 are canonical `/api` routes.
- F179 remains the largest knowledge debt.
- Push remains blocked by the `SysAdminDoc/OpenCut` vs `MavenImaging` credential mismatch.

---

## 13. Pass 7 update (same day, F199 API alias manifest)

Pass 7 closed F199 and corrected the earlier Pass-2 wording.

### What Pass 7 closed

| Item | Status |
|---|---|
| F199 | **DONE** — added `opencut/tools/dump_api_aliases.py`, generated `opencut/_generated/api_aliases.json`, wired `api-aliases` into release smoke, and added `tests/test_api_aliases.py`. |

### Validation after Pass 7

| Command | Result |
|---|---|
| `python -m opencut.tools.dump_api_aliases --check` | PASS — 15 aliases, 218 canonical `/api` routes |
| `python -m pytest tests/test_api_aliases.py tests/test_release_smoke.py -q` | PASS — `16 passed` |
| `python -m py_compile opencut/tools/dump_api_aliases.py scripts/release_smoke.py` | PASS |
| `ruff check opencut/tools/dump_api_aliases.py scripts/release_smoke.py --select E,F,I --ignore E501,E402` | PASS |
| `python scripts/release_smoke.py --json` | PASS — all 12 release-smoke steps green; pytest-fast `236 passed` |

### Live correction

The app has 233 `/api/*` routes. Only 15 are true aliases with equivalent bare routes; 218 are canonical `/api` routes. Future docs should not describe the surface as "233 alias pairs."

### Remaining immediate work

- F179 full `features.md` reconciliation remains the largest knowledge debt.
- Python 3.10/3.11/3.13 `[all]` install matrix remains unproven on this VM.
- Push remains blocked by the `SysAdminDoc/OpenCut` vs `MavenImaging` credential mismatch.

---

## 14. Pass 8 update (same day, F191/F197 feature readiness)

Pass 8 closed the route/check readiness generation item and the registry-owned allowlist item.

### What Pass 8 closed

| Item | Status |
|---|---|
| F191 | **DONE** — added `opencut/tools/dump_feature_readiness.py`, generated `opencut/_generated/feature_readiness.json`, loaded/merged generated records in `opencut/registry.py`, wired `feature-readiness` into release smoke, and added generator/registry tests. |
| F197 | **DONE** — moved `NON_AI_CHECKS` into `opencut/registry.py`; `opencut/model_cards.py` now imports the registry-owned tuple so model-card validation and readiness derivation share one allowlist. |

### Validation after Pass 8

| Command | Result |
|---|---|
| `python -m opencut.tools.dump_feature_readiness` | PASS — generated 58 records / 67 route bindings |
| `python -m py_compile opencut/registry.py opencut/model_cards.py opencut/tools/dump_feature_readiness.py scripts/release_smoke.py` | PASS |
| `python -m pytest tests/test_feature_registry.py tests/test_feature_readiness_generator.py tests/test_model_cards.py tests/test_release_smoke.py -q` | PASS — `35 passed` |
| `python scripts/release_smoke.py --only feature-readiness --json` | PASS |
| `python scripts/release_smoke.py --json` | PASS — all 13 release-smoke steps green; pytest-fast `241 passed` |

### Live correction

`/system/feature-state` now exposes 84 feature records. The generated F191 manifest covers direct route functions that visibly call public `checks.py` probes. It is not a full per-route readiness matrix for all 1,362 routes; deeper core-only gates still belong to F196/F209.

### Remaining immediate work

- F195 is closed locally; the next route/tooling item in the same family is F209 (MCP tool/route consistency gate).
- F179 full `features.md` reconciliation remains the largest knowledge debt.
- Python 3.10/3.11/3.13 `[all]` install matrix remains unproven on this VM.
- Push remains blocked by the `SysAdminDoc/OpenCut` vs `MavenImaging` credential mismatch.

---

## 15. Pass 9 update (same day, F195 MCP curated tool expansion)

Pass 9 closed the missing curated MCP tools item.

### What Pass 9 closed

| Item | Status |
|---|---|
| F195 | **DONE** — expanded `MCP_TOOLS` from 27 to 39 entries, added route mappings for 12 shipped post-Wave-M routes, handled Brand Kit and semantic-search multi-action dispatch, and expanded MCP path validation to cover the new scalar/array path keys. |

### Validation after Pass 9

| Command | Result |
|---|---|
| `python -m py_compile opencut/mcp_server.py scripts/release_smoke.py tests/test_mcp_server.py` | PASS |
| `python -m pytest tests/test_mcp_server.py tests/test_release_smoke.py -q` | PASS — `17 passed` |
| `ruff check opencut/mcp_server.py scripts/release_smoke.py tests/test_mcp_server.py --select E,F,I --ignore E501,E402` | PASS |
| `python scripts/release_smoke.py --json` | PASS — all 13 release-smoke steps green; pytest-fast `246 passed` |

### Remaining immediate work

- F202 repository-side tooling is closed; the first live Apple acceptance still needs repository secrets and a macOS release runner.
- Local-verifiable next candidates: F204 (release SBOM attach plumbing), F208 (OpenAPI validity test), F209 (MCP tool/route consistency test), F218 (blueprint import-order stability), or F219 (SBOM completeness test).
- Push remains blocked by the `SysAdminDoc/OpenCut` vs `MavenImaging` credential mismatch.

---

## 16. Pass 10 update (same day, F202 macOS notarization tooling)

Pass 10 closed the repository-side macOS notarization release path.

### What Pass 10 closed

| Item | Status |
|---|---|
| F202 | **DONE LOCALLY** — added `scripts/notarize_macos.sh`, wired tagged/manual macOS release builds to sign the PyInstaller bundle with Developer ID hardened runtime, submit `OpenCut-Server-macOS.zip` through `xcrun notarytool`, and upload the notarized ZIP on tag releases. |

### Validation after Pass 10

| Command | Result |
|---|---|
| `python -m pytest tests/test_macos_notarization.py tests/test_release_smoke.py -q` | PASS — `15 passed` |
| `ruff check tests/test_macos_notarization.py scripts/release_smoke.py --select E,F,I --ignore E501,E402` | PASS |
| `C:\Program Files\Git\bin\bash.exe -n scripts/notarize_macos.sh` | PASS |
| `python scripts/release_smoke.py --json` | PASS — all 13 release-smoke steps green; pytest-fast `249 passed` |

### Limitation

Apple notarization itself was not executed from this Windows VM. GitHub secrets still need to be configured: `MACOS_CERTIFICATE_P12_BASE64`, `MACOS_CERTIFICATE_PASSWORD`, `APPLE_API_KEY_ID`, `APPLE_API_ISSUER_ID`, and `APPLE_API_PRIVATE_KEY`.

---

## 17. Pass 11 update (same day, F204 release SBOM attachment)

Pass 11 closed the SBOM release attachment item.

### What Pass 11 closed

| Item | Status |
|---|---|
| F204 | **DONE** — Linux tagged/manual release builds generate `dist/opencut-sbom.cyclonedx.json`, archive it as `OpenCut-SBOM-CycloneDX`, and upload it to GitHub Releases on tags. |

### Validation after Pass 11

| Command | Result |
|---|---|
| `python -m pytest tests/test_release_sbom.py tests/test_release_smoke.py -q` | PASS — `14 passed` |
| `ruff check tests/test_release_sbom.py scripts/release_smoke.py --select E,F,I --ignore E501,E402` | PASS |
| `python -c "import pathlib, yaml; yaml.safe_load(pathlib.Path('.github/workflows/build.yml').read_text())"` | PASS |
| `python scripts/sbom.py --format json --output dist/opencut-sbom.cyclonedx.json` | PASS — generated CycloneDX 1.5 JSON |
| `python scripts/release_smoke.py --json` | PASS — all 13 release-smoke steps green; pytest-fast `251 passed` |

### Remaining immediate work

- F205 coverage-floor uplift is next in the Now list; it needs an actual coverage measurement before changing `--cov-fail-under`.
- F219 is now closed in Pass 16; F204 attaches the SBOM, and F219 asserts completeness against declared deps, model cards, and dependency graph entries.

---

## 18. Pass 12 update (same day, F205 attempt + F207 installer FFmpeg manifest)

Pass 12 attempted F205 and closed F207.

### F205 status

The local environment initially lacked `pytest-cov` and `pytest-xdist`; those were installed with `python -m pip install pytest-cov pytest-xdist`. The full CI-style measurement command then timed out after 20 minutes:

```powershell
python -m pytest tests/ -q --tb=short --cov=opencut --cov-report=term --cov-report=json:dist\coverage-f205.json --cov-fail-under=0 -n auto --dist worksteal
```

No `dist\coverage-f205.json` was produced. F205 remains open; do not raise `--cov-fail-under=50` until a complete measurement exists.

### What Pass 12 closed

| Item | Status |
|---|---|
| F207 | **DONE** — current bundled FFmpeg/ffprobe version is pinned as `8.0.1-essentials_build-www.gyan.dev`; WPF and Inno installers write it to `~/.opencut/installer.json`. |

### Validation after Pass 12

| Command | Result |
|---|---|
| `.\ffmpeg\ffmpeg.exe -version` | PASS — first line reports `8.0.1-essentials_build-www.gyan.dev` |
| `python -m pytest tests/test_ffmpeg_installer_manifest.py tests/test_release_smoke.py -q` | PASS — `15 passed` |
| `ruff check tests/test_ffmpeg_installer_manifest.py scripts/release_smoke.py --select E,F,I --ignore E501,E402` | PASS |
| `python -m py_compile scripts/release_smoke.py` | PASS |
| `python scripts/release_smoke.py --json` | PASS — all 13 release-smoke steps green; pytest-fast `254 passed` |
| `dotnet build installer/src/OpenCut.Installer/OpenCut.Installer.csproj --no-restore` | BLOCKED — no .NET SDK installed on this VM |

---

## 19. Pass 13 update (same day, F208 OpenAPI contract gate)

Pass 13 closed the OpenAPI validity test item.

### What Pass 13 closed

| Item | Status |
|---|---|
| F208 | **DONE** — `/openapi.json` now converts Flask `<param>` path syntax to OpenAPI `{param}` syntax, emits path parameter objects, uses unique path-qualified operation IDs, documents mutating-method 400/403 responses, and has a release-smoke contract test. |

### Validation after Pass 13

| Command | Result |
|---|---|
| `python -m pytest tests/test_openapi_contract.py tests/test_release_smoke.py -q` | PASS — `16 passed` |
| `ruff check opencut/openapi.py tests/test_openapi_contract.py scripts/release_smoke.py --select E,F,I --ignore E501,E402` | PASS |
| `python -m py_compile opencut/openapi.py scripts/release_smoke.py tests/test_openapi_contract.py` | PASS |
| `python scripts/release_smoke.py --json` | PASS — all 13 release-smoke steps green; pytest-fast `258 passed` |

### Remaining immediate work

- F205 remains open and should resume only where a full coverage command can finish.
- The next local-verifiable Now item is F219 (SBOM completeness).

---

## 21. Pass 15 update (same day, F218 blueprint import-order stability)

Pass 15 closed the blueprint import-order item.

### What Pass 15 closed

| Item | Status |
|---|---|
| F218 | **DONE** — `tests/test_route_collisions.py` now pins the exact 99-item `get_core_blueprints()` order, asserts `motion_design_api` is appended last for legacy `/api/motion/*` routes, and runs in release smoke. |

### Validation after Pass 15

| Command | Result |
|---|---|
| `python -m pytest tests/test_route_collisions.py tests/test_release_smoke.py -q` | PASS — `19 passed` |
| `ruff check tests/test_route_collisions.py scripts/release_smoke.py --select E,F,I --ignore E501,E402` | PASS |
| `python -m py_compile tests/test_route_collisions.py scripts/release_smoke.py` | PASS |
| `python scripts\release_smoke.py --json` | PASS — all 13 release-smoke steps green; pytest-fast `266 passed` |

### Remaining immediate work

- F205 remains open and should resume only where a full coverage command can finish.
- The next local-verifiable Now item is F219 (SBOM completeness).

---

## 22. Pass 16 update (same day, F219 SBOM completeness)

Pass 16 closed the SBOM completeness item.

### What Pass 16 closed

| Item | Status |
|---|---|
| F219 | **DONE** — `scripts/sbom.py` now emits unique declared Python dependency components, 47 model-card components, and a non-empty CycloneDX dependency graph; `tests/test_sbom_completeness.py` runs in release smoke. |

### Validation after Pass 16

| Command | Result |
|---|---|
| `python -m pytest tests/test_sbom_completeness.py tests/test_release_sbom.py tests/test_release_smoke.py -q` | PASS — `17 passed` |
| `ruff check scripts/sbom.py tests/test_sbom_completeness.py scripts/release_smoke.py --select E,F,I --ignore E501,E402` | PASS |
| `python -m py_compile scripts/sbom.py tests/test_sbom_completeness.py scripts/release_smoke.py` | PASS |
| JSON/XML SBOM generation | PASS — 14 required components / 73 optional components / 47 model-card components |
| `python scripts\release_smoke.py --json` | PASS — all 13 release-smoke steps green; pytest-fast `269 passed` |

### Remaining immediate work

- F205 remains open and should resume only where a full coverage command can finish.
- The next local-verifiable Now items are F241, F243, and F244.
- F236 was closed in Pass 17 after fresh regulatory verification, and F237 was closed in Pass 18 after fresh ITU/EBU/FFmpeg verification. F251 and F259 likely need fresh Adobe/UXP verification before implementation because their source facts can drift.

---

## 23. Pass 17 update (same day, F236 FCC caption display-settings tokens)

Pass 17 closed the FCC caption display-settings token item.

### What Pass 17 closed

| Item | Status |
|---|---|
| F236 | **DONE** — `opencut/core/caption_display_settings.py` now defines FCC-style user-overridable caption display setting tokens, `/captions/display-settings/tokens` and `/captions/display-settings/preview` expose the schema/preview contract, and `/captions/burnin/file` accepts `display_settings`. |

### Validation after Pass 17

| Command | Result |
|---|---|
| `python -m pytest tests/test_caption_display_settings.py tests/test_route_manifest.py tests/test_release_smoke.py -q` | PASS — `21 passed` |
| `ruff check opencut/core/caption_display_settings.py opencut/routes/captions.py tests/test_caption_display_settings.py scripts/release_smoke.py --select E,F,I --ignore E501,E402` | PASS |
| `python -m py_compile opencut/core/caption_display_settings.py opencut/routes/captions.py tests/test_caption_display_settings.py scripts/release_smoke.py` | PASS |
| `python -m opencut.tools.dump_route_manifest` | PASS — regenerated 1,361 routes / 101 blueprints |
| `python -m opencut.tools.dump_api_aliases --check` | PASS — 15 aliases / 218 canonical `/api` routes |
| `python -m opencut.tools.dump_feature_readiness --check` | PASS — 58 generated records / 67 route bindings |
| `python scripts\release_smoke.py --json` | PASS — all 13 release-smoke steps green; pytest-fast `273 passed` |

### Remaining immediate work

- F205 remains open and should resume only where a full coverage command can finish.
- The next local-verifiable Now items are F241, F243, and F244.
- F251 and F259 likely need fresh Adobe/UXP verification before implementation.

---

## 26. Pass 20 update (same day, F241 text-shaping gate)

Pass 20 closed the text-shaping CI/release gate item.

### What Pass 20 closed

| Item | Status |
|---|---|
| F241 | **DONE** — `opencut/tools/text_shaping_gate.py` now hard-fails if FFmpeg/libass lacks HarfBuzz, FriBidi, ASS, or subtitles support; it also reports Pillow RAQM and optional Skia shaping capability with strict flags for packaging environments. Release smoke and GitHub Actions run the gate. |

### Validation after Pass 20

| Command | Result |
|---|---|
| `python -m opencut.tools.text_shaping_gate --json` | PASS — FFmpeg/libass hard gates OK; Pillow RAQM warning; Skia skipped |
| `python -m pytest tests/test_text_shaping_gate.py tests/test_release_smoke.py -q --tb=short` | PASS — `17 passed` |
| `ruff check opencut/tools/text_shaping_gate.py scripts/release_smoke.py tests/test_text_shaping_gate.py --select E,F,I --ignore E501,E402` | PASS |
| `python -m py_compile opencut/tools/text_shaping_gate.py scripts/release_smoke.py tests/test_text_shaping_gate.py` | PASS |
| `python scripts\release_smoke.py --only text-shaping --json` | PASS — one advisory Pillow warning |
| `python scripts\release_smoke.py --json` | PASS — all 14 release-smoke steps green; pytest-fast `289 passed` |

### Remaining immediate work

- F205 remains open and should resume only where a full coverage command can finish.
- The next local-verifiable Now items are F243 and F244.
- F251 and F259 likely need fresh Adobe/UXP verification before implementation.

---

## 27. Pass 21 update (same day, F243 UTF-8 no-BOM SRT policy)

Pass 21 closed the SRT writer encoding policy item.

### What Pass 21 closed

| Item | Status |
|---|---|
| F243 | **DONE** — `opencut/export/srt.py` now writes UTF-8 without BOM by default and exposes `legacy_windows_bom=True` for old Windows players. `/captions`, `/transcript/export`, `/full`, `/interview-polish`, `opencut captions`, `opencut full`, and shot-aware file export all have an opt-in path. |

### Validation after Pass 21

| Command | Result |
|---|---|
| `python -m pytest tests/test_srt_encoding.py tests/test_captions_regressions.py tests/test_core.py::TestSRTExport tests/test_subtitle_pro.py::TestShotAwareExport -q --tb=short` | PASS — `13 passed` |
| `ruff check opencut/export/srt.py opencut/routes/captions.py opencut/cli.py opencut/core/subtitle_shot_aware.py scripts/release_smoke.py tests/test_srt_encoding.py --select E,F,I --ignore E501,E402` | PASS |
| `python -m py_compile opencut/export/srt.py opencut/routes/captions.py opencut/cli.py opencut/core/subtitle_shot_aware.py scripts/release_smoke.py tests/test_srt_encoding.py` | PASS |
| `python scripts\release_smoke.py --json` | PASS — all 14 release-smoke steps green; pytest-fast `294 passed` |

### Remaining immediate work

- F205 remains open and should resume only where a full coverage command can finish.
- F244 is now closed in Pass 22; remaining Now items are F205, F251, and F259.
- F251 and F259 likely need fresh Adobe/UXP verification before implementation.

---

## 25. Pass 19 update (same day, F240 caption reading-speed profiles)

Pass 19 closed the caption reading-speed profile item.

### What Pass 19 closed

| Item | Status |
|---|---|
| F240 | **DONE** — `opencut/core/caption_reading_profiles.py` now defines source-backed reading-speed profiles, `/captions/qc/reading-profiles` exposes them, and `/captions/qc` accepts a `reading_profile` overlay while preserving the base caption standard. Source verification corrected the shorthand: Netflix adult is 20 CPS, Netflix children is 17 CPS, BBC is 160-180 WPM, DCMP upper-level is 160 WPM, FCC is qualitative, and YouTube 220 WPM is OpenCut advisory rather than an official YouTube rule. |

### Validation after Pass 19

| Command | Result |
|---|---|
| `python -m pytest tests/test_caption_reading_profiles.py tests/test_caption_qc.py tests/test_analysis.py::TestCaptionCompliance -q --tb=short` | PASS — `31 passed` |
| `ruff check opencut/core/caption_reading_profiles.py opencut/core/caption_compliance.py opencut/core/caption_qc.py opencut/routes/captions.py scripts/release_smoke.py tests/test_caption_reading_profiles.py --select E,F,I --ignore E501,E402` | PASS |
| `python -m py_compile opencut/core/caption_reading_profiles.py opencut/core/caption_compliance.py opencut/core/caption_qc.py opencut/routes/captions.py scripts/release_smoke.py tests/test_caption_reading_profiles.py` | PASS |
| `python -m opencut.tools.dump_route_manifest --check --quiet` | PASS — 1,362 routes / 101 blueprints |
| `python -m opencut.tools.dump_api_aliases --check` | PASS — 15 aliases / 218 canonical `/api` routes |
| `python -m opencut.tools.dump_feature_readiness --check` | PASS — 58 generated records / 67 route bindings |
| `python scripts\release_smoke.py --json` | PASS — all 13 release-smoke steps green; pytest-fast `284 passed` |

### Remaining immediate work

- F205 remains open and should resume only where a full coverage command can finish.
- The next local-verifiable Now items are F241, F243, and F244.
- F251 and F259 likely need fresh Adobe/UXP verification before implementation.

---

## 24. Pass 18 update (same day, F237 loudness standards registry)

Pass 18 closed the loudness standards correction item.

### What Pass 18 closed

| Item | Status |
|---|---|
| F237 | **DONE** — `opencut/core/loudness_standards.py` now centralizes ITU BS.1770-5, EBU R128 v5.0, FFmpeg loudnorm, and platform/profile loudness targets. The stale "drop speculative -5" premise was corrected: ITU-R BS.1770-5 is in force and BS.1770-4 is superseded. |

### Validation after Pass 18

| Command | Result |
|---|---|
| `python -m pytest tests/test_loudness_standards.py tests/test_release_smoke.py -q` | PASS — `17 passed` |
| focused compatibility/route pytest slice | PASS — `9 passed` |
| `ruff check opencut/core/loudness_standards.py opencut/core/audio_suite.py opencut/core/audio_analysis.py opencut/core/broadcast_qc.py opencut/routes/audio.py tests/test_loudness_standards.py scripts/release_smoke.py --select E,F,I --ignore E501,E402` | PASS |
| `python -m py_compile opencut/core/loudness_standards.py opencut/core/audio_suite.py opencut/core/audio_analysis.py opencut/core/broadcast_qc.py opencut/routes/audio.py tests/test_loudness_standards.py scripts/release_smoke.py` | PASS |
| `python scripts/release_smoke.py --json` | PASS — all 13 steps green; pytest-fast `278 passed` |

### Remaining immediate work

- F205 remains open and should resume only where a full coverage command can finish.
- The next local-verifiable Now items are F241, F243, and F244.
- F251 and F259 likely need fresh Adobe/UXP verification before implementation.

---

## 20. Pass 14 update (same day, F209 MCP route consistency)

Pass 14 closed the MCP tool/route consistency item.

### What Pass 14 closed

| Item | Status |
|---|---|
| F209 | **DONE** — fixed `opencut_chat_edit` from planned `/agent/chat` to shipped `POST /chat`, then added a live Flask route-consistency test for all 39 MCP tool route mappings plus special action dispatch paths. |

### Validation after Pass 14

| Command | Result |
|---|---|
| `python -m pytest tests/test_mcp_server.py tests/test_release_smoke.py -q` | PASS — `18 passed` |
| `ruff check opencut/mcp_server.py tests/test_mcp_server.py --select E,F,I --ignore E501,E402` | PASS |
| `python -m py_compile opencut/mcp_server.py tests/test_mcp_server.py` | PASS |
| live route-table probe | PASS — 39 MCP tools / 39 route mappings / 0 missing backend routes |
| `python scripts/release_smoke.py --json` | PASS — all 13 release-smoke steps green; pytest-fast `259 passed` |

### Remaining immediate work

- F205 remains open and should resume only where a full coverage command can finish.
- The next local-verifiable Now items are F218 (blueprint import-order stability) and F219 (SBOM completeness).

---

## 28. Pass 23 update (same day, F205 interrupted coverage reattempt wrap-up)

Pass 23 did not close an F-number. It was a wrap-up pass after the autonomous loop started a second F205 coverage measurement and the session was interrupted before pytest completed.

### F205 status after Pass 23

| Item | Status |
|---|---|
| Attempted command | `python -m pytest tests/ -q --tb=short --cov=opencut --cov-report=term-missing --cov-report=json:dist\coverage-f205.json --cov-fail-under=0 -n auto --dist worksteal` |
| Runtime | Interrupted after 2,206.6 seconds (36m46s). |
| Partial JSON | Valid JSON, but incomplete because pytest did not complete: 126,421 statements, 65,890 covered, 60,531 missing, 52.1195% reported coverage across 670 files. |
| Cleanup | Stopped one leftover pytest process (`python.exe -m pytest tests sidecar/tests -q`), then removed stale `.coverage` and `dist\coverage-f205.json` after recording their evidence. |
| Decision | F205 remains open. Do not raise `--cov-fail-under=50` from this partial run. |

### Remaining immediate work

- F205 should resume only on a local or CI runner where the full coverage command can finish cleanly.
- F251 and F259 remain the other Now items and need fresh Adobe/UXP verification before implementation.
- Push remains blocked by the `SysAdminDoc/OpenCut` vs local GitHub-account mismatch.
