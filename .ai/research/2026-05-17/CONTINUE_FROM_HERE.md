# OpenCut Research — CONTINUE FROM HERE (for Pass 7)

**This file's purpose:** if a future autonomous research session starts up, **read this first** before re-doing any of the work already on disk.

**Last update:** 2026-05-17 (during Pass 6; Passes 1-6 all ran on the same calendar day)
**Session state:** all mandated artefacts exist, Pass 4 ran full release-smoke successfully, Pass 5 closed F261/F262/F270, and Pass 6 closed F264/F266. This file documents deferred research/product work for a future Pass 7+, not a broken or incomplete research run.

---

## 1. State at hand-off

- **Repo branch:** `main`, 25 commits ahead of `origin/main` before the Pass-4 checkpoint commit. Push to `SysAdminDoc/OpenCut` still depends on local GitHub auth.
- **Last shipped version:** v1.32.0 (light theme + appearance toggle, 2026-05-09).
- **Live counts:** 1,359 routes / 101 blueprints / 523 core modules / 131 test files / 47 model cards / 105 `check_X_available()` functions / 29 `FeatureRecord` entries / 27 MCP tools / 30 OpenAPI-typed endpoints.
- **F-numbers in ledger:** F001-F272 (Pass 1 added F121-F190, Pass 2 added F191-F260, Pass 3 added F261-F272).
- **Wave letters in ledger:** A-M shipped; N-T planned in ROADMAP.md but not yet F-number-tiered (covered by F180).

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

**At repo root:** `PROJECT_CONTEXT.md` (canonical context, Pass 1-6 updates), `ROADMAP.md` (v4.3-v4.9 sections), `AGENTS.md` (pointer added), `CLAUDE.md` (pointer added, gitignored).

---

## 3. Historical deferred work for Pass 3 (mostly closed)

This section is preserved because Pass 3/4 used it as the checklist. See §7 and §10 for what is now closed and what remains.

### 3.1 Quick wins (S effort)

1. **Run `python -m opencut.tools.dump_route_manifest --check`** to verify the cached 1,359-route figure against live `url_map`. Possible drift after recent commits.
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
4. **Generate `opencut/_generated/api_aliases.json`** (F199 *Now* tier) — map `/api/*` aliases to canonical routes (233 pairs).
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
| §3.2.4 Generate `opencut/_generated/api_aliases.json` | F199 deliverable; needs a small Python script that walks the manifest. Easy Pass-4 quick win. |
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
   - **§3.2.4 F199 api_aliases.json** (30 minutes, small script)
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

### Pass 7 entry point

1. **Push checkpoint commits** once GitHub auth is available on this machine.
2. **Implement F199** — generate `opencut/_generated/api_aliases.json` mapping `/api/*` aliases to canonical routes, plus a consistency test.
3. **Complete F179** full `features.md` reconciliation; this remains the largest knowledge debt.
4. **Run a Python 3.10/3.11/3.13 install matrix** for `[all]`; this cannot be fully proven from this VM's single Python 3.12 runtime.

### Current limitations

- No full cross-version Python install matrix.
- No Premiere UDT runtime verification of the 14 low-risk JSX to UXP ports.
- No push attempted; local auth still needs to be fixed outside the repo.
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

- F199 is the smallest remaining Now-tier implementation candidate from the Pass-2 deferred list.
- F179 remains the largest knowledge debt.
- Push remains blocked by the `SysAdminDoc/OpenCut` vs `MavenImaging` credential mismatch.
