# OpenCut — Changeset Summary (2026-05-17 research run)

**Run scope:** Autonomous deep research, memory consolidation, and roadmap planning. Read-only on existing code (no functional changes); created / appended documentation artefacts only.

---

## 1. Files created

### Repo root (1 file)

| Path | Purpose |
|---|---|
| `PROJECT_CONTEXT.md` | Canonical, cross-tool source of truth. Identity + current numbers + architecture + documentation map + question-routing table + hard constraints + active in-flight changes + Now/Next/Later cadence summary + biggest gaps + onboarding guide. ~280 lines. Replaces no existing file; supplements `CLAUDE.md` and `ROADMAP.md`. |

### `.ai/research/2026-05-17/` (10 files)

| Path | Purpose | Approx. size |
|---|---|---|
| `.ai/research/2026-05-17/STATE_OF_REPO.md` | Local repo reconnaissance: identity, live counts (1,359 routes / 101 blueprints / 523 core modules / 131 test files / 47 model cards), architecture diagram, wave + F-number ledger snapshot, dirty working tree breakdown, CI/build/release infrastructure, known infrastructure debt, security posture snapshot, what this audit changed | ~300 lines |
| `.ai/research/2026-05-17/MEMORY_CONSOLIDATION.md` | Inventory + reconciliation of every AI/agent instruction file, roadmap doc, changelog. Reconciliation logic (where files agree / disagree / duplicate / conflict). Open conflicts. Recommended changes to AGENTS.md + CLAUDE.md. What gets extracted to PROJECT_CONTEXT.md. Question-routing table. Gaps no doc owns. Action items | ~270 lines |
| `.ai/research/2026-05-17/SECURITY_AND_DEPENDENCY_REVIEW.md` | TL;DR action table. Per-dependency tables (core web / scientific / audio / ASR / generative / ML framework / tooling / frontend / FFmpeg). The torch ≥2.6 cascade analysis. Action items mapped to F-numbers. Repo-side hardening in the dirty working tree. CVE summary table | ~300 lines |
| `.ai/research/2026-05-17/COMPETITOR_MATRIX.md` | Direct Premiere extension competitors. OSS NLEs. Adjacent OSS automation. Commercial closed-source AI tools. **Agentic / chat-driven editing systems** (Underlord, FireRed, vibeframe, VideoAgent, ViMax, etc.). Mind-the-gap summary. Notable paywalls that exist because local-AI alternative isn't ready | ~260 lines |
| `.ai/research/2026-05-17/DATASET_MODEL_INTEGRATION_REVIEW.md` | Models the team has not yet considered (post-2026-05-16): video gen, TTS/voice/dub, restoration/upscaling/depth/tracking, standards/spec, agentic/MCP/tooling. Datasets relevant to evaluation harness. External APIs/integrations. Model card / licence hygiene. AI eval harness v2 extensions. Priority for AI/integration items | ~330 lines |
| `.ai/research/2026-05-17/FEATURE_BACKLOG.md` | Raw harvest of 70 new F-numbers F121-F190. Grouped: A) security/deps, B) caption/accessibility/standards, C) agentic/chat/MCP, D) AI capability gaps, E) real-time/streaming, F) model surface refreshes, G) eval/governance/docs, H) Premiere UXP gap reports for Adobe. Explicit declined items | ~190 lines |
| `.ai/research/2026-05-17/PRIORITIZATION_MATRIX.md` | Now / Next / Later / Under Consideration / Rejected tier placement with Impact/Effort/Risk scores. Sequencing summary v1.33-v1.43+. Capacity assumptions | ~210 lines |
| `.ai/research/2026-05-17/SOURCE_REGISTER.md` | Every local + external source cited. Local R-L01-41 (41 files/commands). External: R-P01-18 Premiere (18), R-M01-15 MCP/agents (15), R-C01-19 commercial products (19), R-A01-27 AI models (27), R-D01-60 dependency advisories (60), R-S01-08 community signal (8) | ~250 lines |
| `.ai/research/2026-05-17/RESEARCH_LOG.md` | Phases executed. Subagent prompts. Search strategies. Saturation tests per category. Failed searches / dead ends. Bias / assumptions to flag. Things this audit did NOT do. Continuation hints | ~230 lines |
| `.ai/research/2026-05-17/CHANGESET_SUMMARY.md` | This file | ~150 lines |

**Total new content:** ~2,770 lines of structured markdown across 11 files.

---

## 2. Files edited (minimal, additive)

| Path | Change | Lines added |
|---|---|---|
| `AGENTS.md` | Appended "Canonical Project Context" block (pointer to `PROJECT_CONTEXT.md` + `ROADMAP.md` + `CHANGELOG.md` + `.ai/research/2026-05-17/`). Pre-existing 9-line content preserved verbatim. | ~12 |
| `CLAUDE.md` | Inserted "Canonical Project Context" block immediately after the `# OpenCut - CLAUDE.md` title and before `## Tech Stack`. Pre-existing 1,509-line content preserved verbatim. Markdown-lint warnings on the heading are pre-existing throughout the file (tight heading format). **NOTE**: `CLAUDE.md`, `CODEX-CHANGELOG.md`, `CLAUDE-HANDOFF-PROMPT.md`, and `research.md` are listed in `.gitignore` (team convention: agent memory is local-only). The edit persists on disk for future agent sessions on this machine but will not appear in `git diff` or commits. The corresponding pointer in `AGENTS.md` is the *tracked* analogue. | ~12 |
| `ROADMAP.md` | Appended a new "**2026-05-17 v4.4 Autonomous Research Audit (delta)**" section between the v4.3 status note and the existing v4.3 Phase 0 heading. Includes: Phase 0 delta (what v4.3 missed or what changed since 2026-05-16), Phase 1 research coverage delta, Phase 2 F121-F190 tier summary, Phase 3 top three strategic moves the v4.3 audit understated, Phase 4 self-audit. References the 10 supporting artefacts in `.ai/research/2026-05-17/`. Existing v4.3 content (sections from "## 2026-05-16 v4.3 Autonomous Research Audit" through the v4.3 source appendix) preserved verbatim. | ~70 |

---

## 3. Files NOT changed (deliberately)

- `README.md` — marketing-soft route count (1,344) is stale vs manifest (1,359). Did not edit; the F099 manifest is the truth and is documented in `PROJECT_CONTEXT.md`. The README badge regeneration is a separate cleanup.
- `ROADMAP-NEXT.md` — predecessor to ROADMAP.md v4.3 / v4.4. Kept as archive with its source URLs and per-wave gotchas.
- `ROADMAP-COMPLETED.md` — high-level summary, stale relative to CHANGELOG.md. Kept; banner is on the F185 backlog.
- `CHANGELOG.md` — authoritative for ship dates. Not edited (no release shipped today).
- `MODERNIZATION.md` — module-by-module dep audit, baseline v1.9.18. Regeneration covered by F177 model cards sweep.
- `AUDIT.md`, `research.md` — v1.11.0 / v1.28.2 competitive audits. Kept as predecessor to ROADMAP.md v4.3 / v4.4.
- `features.md` — 402-feature aspirational catalogue. Banner pending F185 (Now tier).
- `CODEX-CHANGELOG.md` — Codex handoff snapshot. Kept.
- `CLAUDE-HANDOFF-PROMPT.md` — boilerplate prompt. Kept.
- `docs/*` — `MODELS.md`, `NODE_ADVISORIES.md`, `RESEARCH.md`, `ROADMAP.md`, `ROADMAP-COMPLETED.md`, `UXP_MIGRATION.md`, `WINDOWS_ARM64_PACKAGING.md`. Auto-generated or live artefacts; not edited.
- `.github/copilot-instructions.md` — should get the same canonical-context pointer eventually, but deferred. Out of scope for this run.
- **All Python source files** (`opencut/**/*.py`) — read-only research; no functional changes.
- **All test files** (`tests/**/*.py`) — read-only; no test changes.
- **The 7 dirty modified files in the working tree** (auth.py, security.py, helpers.py, user_data.py, routes/captions.py, routes/system.py, routes/timeline.py) — left exactly as found. Documented as **F138 commit-pending** in PRIORITIZATION_MATRIX.md and PROJECT_CONTEXT.md §8.

---

## 4. Files NOT created (and why)

- **`CONTINUE_FROM_HERE.md`** — not needed. This audit completed within session budget. Continuation hints are captured in `RESEARCH_LOG.md` §8.

---

## 5. Total impact

| Metric | Value |
|---|---|
| Files created | 11 |
| Files edited | 3 |
| Lines added across edited files | ~94 |
| Lines created in new files | ~2,770 |
| Source citations (this run alone) | 138 unique R-prefixed IDs |
| New F-numbers proposed | 70 (F121-F190) |
| Existing F-numbers status-updated | 0 (v4.3 ledger preserved verbatim) |
| Wave letters touched | 0 (wave-letter plan preserved; F180 calls for future re-tier) |
| Python / JSX / JS / CSS / config changes | 0 |
| Tests added or modified | 0 |
| Dependency pins changed | 0 (recommendations only — see SECURITY_AND_DEPENDENCY_REVIEW.md for action items) |
| Commits made | 0 (read-only research run; F138 recommends committing the dirty tree as a separate PR) |
| Pushes attempted | 0 (push to `SysAdminDoc/OpenCut` is blocked from this VM per persistent memory) |

---

## 6. Verification checklist (post-run)

A future session can verify this audit by running:

```bash
# All artefacts in place
ls Z:/repos/OpenCut/.ai/research/2026-05-17/
ls Z:/repos/OpenCut/PROJECT_CONTEXT.md

# Cross-tool pointers
grep -l "PROJECT_CONTEXT.md" Z:/repos/OpenCut/AGENTS.md Z:/repos/OpenCut/CLAUDE.md

# v4.4 section in ROADMAP.md
grep -n "v4.4 Autonomous Research Audit" Z:/repos/OpenCut/ROADMAP.md

# No accidental code touches
cd Z:/repos/OpenCut && git status --short
# Should show only the 7 pre-existing modified files (auth.py, security.py, helpers.py, user_data.py, captions.py, system.py, timeline.py)
# AND the new files / edits this run added:
#   AGENTS.md (modified)
#   CLAUDE.md (modified)
#   PROJECT_CONTEXT.md (untracked)
#   ROADMAP.md (modified)
#   .ai/research/2026-05-17/* (untracked)
```

---

## 7. What is now true that was not before

1. There is a single `PROJECT_CONTEXT.md` at the repo root that an agent or contributor can open first and route from.
2. The wave-letter vs F-number distinction is named.
3. The dirty working tree is documented as a coherent security batch (F138), not random in-flight noise.
4. Post-2026-05-16 model surfaces (daVinci-MagiHuman, LTX-2.3, etc.), standards (C2PA 2.3, IMSC 1.3, OCIO 2.5), and dependency exposures (Pillow CVEs, audiocraft cascade) are tracked.
5. The Now-tier security debt is explicit: Pillow 12.2, flask-cors 6.x, pydub-on-3.13, OpenTimelineIO-Plugins migration.
6. The Next-tier flagship is named: `/agent/chat` conductor + UXP MCP transport + StreamDiffusionV2 real-time preview.
7. ROADMAP.md self-cites the v4.4 audit; future v4.5 audits know to read this run.

---

## 7.5 Pass 2 additions (same day, second autonomous research run)

Pass 2 added 5 new artefacts + extended 6 existing ones + appended ROADMAP.md v4.5 section + updated PROJECT_CONTEXT.md + wrote CONTINUE_FROM_HERE.md.

### Files created in Pass 2 (`.ai/research/2026-05-17/`)

| Path | Purpose | Approx lines |
|---|---|---|
| `ROUTE_READINESS_AUDIT.md` | F100 / F115 / MCP / OpenAPI coverage gaps; CEP-bound route catalogue; /api/* alias surface; proposes F191-F199 | 220 |
| `INSTALLER_AUDIT.md` | WPF .NET 9 + Inno + PyInstaller + Docker + Windows ARM64 (F101); CI walk; signing/notarisation deadlines; proposes F200-F207 | 200 |
| `TEST_COVERAGE_GAPS.md` | 12 specific gaps (OpenAPI validity, MCP consistency, JS unit tests, launcher smoke, WPF installer tests, ML perf benchmarks, fuzz extensions, race conditions, UXP contract, SBOM completeness); proposes F208-F219 | 230 |
| `FEATURES_RECONCILIATION.md` | features.md (402 entries) sample walk on 40 entries; status methodology; ~60% SHIPPED, ~27% UNCLEAR; proposes F220-F224 | 220 |
| `FEATURE_BACKLOG_ADDENDUM.md` | F191-F260 (+70 items) consolidated from the four new artefacts + 3 Pass-2 subagent briefs; tier summary with regulatory-deadline priority bumps (F202, F236) | 230 |
| `CONTINUE_FROM_HERE.md` | Pass-3 hand-off: quick wins / medium / large items; known limitations; suggested entry point | 250 |

### Files extended in Pass 2

| Path | What was added |
|---|---|
| `PRIORITIZATION_MATRIX.md` | §6.5 — Pass-2 tier deltas with two regulatory-deadline bumps to *Now* (F202, F236) |
| `SOURCE_REGISTER.md` | Pass-2 section — 14 new local IDs (R-L42-55) + 15 Premiere/UXP IDs (R-P19-33) + 30 Frame.io IDs (R-F01-30) + 27 niche AI / accessibility / standards IDs (R-N01-27) — ~85 new R-prefixed sources |
| `RESEARCH_LOG.md` | Pass-2 section — phases, subagent prompts, saturation tests, failed searches, bias, things-not-done, continuation hints |
| `CHANGESET_SUMMARY.md` | This addition (§7.5) |
| `ROADMAP.md` (root) | Appended v4.5 audit section: Phase 0 (what Pass 1 missed) + Phase 1 (research delta) + Phase 2 (F191-F260 tier summary) + Phase 3 (top 3 strategic moves Pass 1 understated: OTIO Marker review-bundle anchor, Bolt UXP WebView migration, Hybrid Plugins for 5 CEP-blocked features) + Phase 4 (self-audit) |
| `PROJECT_CONTEXT.md` (root) | §1 last-consolidated note ("two passes that day"); §3 wave/F-number table extended to F260; new §9.5 regulatory deadlines block (F202, F236); §10 expanded with 4 new biggest-gaps entries (features.md reconciliation 60% shipped, OTIO Marker anchor, WebView UI migration target, route-readiness coverage gap); §12 artefact list extended with Pass-2 files |

### Files NOT changed in Pass 2

- All Python source files — read-only research; no functional changes.
- All test files — no test additions.
- All build / installer configs — no changes.
- `AGENTS.md` / `CLAUDE.md` — already received the canonical-context pointer in Pass 1; no further edits in Pass 2.
- The 7 dirty modified files in working tree — promoted into the Pass-4 validated local checkpoint as F138.

### Cumulative impact (Pass 1 + Pass 2)

| Metric | Pass 1 | Pass 2 | Total |
|---|---:|---:|---:|
| Files created | 11 | 6 | 17 |
| Files edited | 3 | 6 (incl. 4 existing artefacts + ROADMAP + PROJECT_CONTEXT) | 9 unique files touched |
| Lines created in new files | ~2,770 | ~1,350 | ~4,120 |
| Lines added to edited files | ~94 | ~480 | ~574 |
| F-numbers newly proposed | F121-F190 (70) | F191-F260 (70) | F121-F260 (**140 new**) |
| Wave letters touched | 0 | 0 | 0 (F180 retire-wave-letters-into-F-numbers still deferred) |
| Source citations | 138 R-prefixed | +85 R-prefixed | ~225 unique R-prefixed |
| Subagents launched | 3 | 3 | 6 total |
| Python / JS / config changes | 0 | 0 | 0 |
| Tests added or modified | 0 | 0 | 0 |
| Dependency pins changed | 0 | 0 | 0 (recommendations only) |
| Commits made | 0 | 0 | 0 |
| Pushes attempted | 0 | 0 | 0 |

---

## 7.6 Pass 3 additions (same day, third autonomous research run)

Pass 3 added 4 new artefacts + updated PROJECT_CONTEXT.md (§9.4 live verification block + §10 expansions 12-13) + appended ROADMAP.md v4.6 + extended CHANGESET (this section) + CONTINUE_FROM_HERE.md.

### Files created in Pass 3 (`.ai/research/2026-05-17/`)

| Path | Purpose | Approx lines |
|---|---|---|
| `LIVE_VERIFICATION.md` | Executed CONTINUE_FROM_HERE §3.1 quick-wins. F099/F096/F093/F094/npm-audit live results. **Cross-platform launcher gap confirmed (F261)**. Side-channel: bootstrap warnings, FFmpeg path detection. JSX inventory (18 `ocXxx` functions). Pass-3 corrections to Pass 1/2. Proposes F261-F265 | 240 |
| `CEP_UXP_PARITY_MATRIX.md` | **Completes F198** (CEP-only route catalogue). All 18 JSX functions × `@adobe/premierepro@26.3.0-beta.67` typings; risk classification (14 low, 1 med, 1 different-mechanism, 2 truly CEP-only). Deep-dive on `ocAddNativeCaptionTrack` and `ocQeReflect`. Revised F252 (XL→L) + F253 (XL→L) effort estimates with sub-phases. Proposes F266-F267 | 220 |
| `AGENT_UX_RFC.md` | F143-F145 design space RFC. Adopts: Copilot Workspace editable-plan, Cursor checkpoint+rollback, Underlord post-turn self-review, Aider snapshot discipline, per-region accept/reject, Claude Code Skills format. Rejects: accept-all, auto-commit-before-preview, atomic multi-file apply. Endpoint shape, self-review sketch, Skills SDK layout. Proposes phasing within v1.36. Open questions for maintainer | 280 |
| `MARKET_POSITIONING.md` | OpenCut replaces ~$1,400/yr subscriptions ($720 AutoCut+AutoPod+Submagic + $288 Descript + $299-699 Topaz). Mister Horse Animation Composer ~900k installs proves free-shell + paid-packs model. Per-creator-type pain-point map. README copy proposal. Proposes F268-F272 | 200 |

### Files extended in Pass 3

| Path | What was added |
|---|---|
| `PROJECT_CONTEXT.md` (root) | §9.4 — live verification of governance gates + 2-function CEP residual; §10 expanded with 2 more biggest-gaps entries (#12 agent conductor convergence, #13 market positioning); §12 artefact list extended with Pass-3 files |
| `ROADMAP.md` (root) | Appended v4.6 section: Phase 0 live verification (4 gates pass, 1 ledger gap), Phase 1 CEP↔UXP parity (2 of 18 CEP-only), Phase 2 agent UX RFC, Phase 3 market positioning, Phase 4 F261-F272 deltas, Phase 5 top 3 strategic moves Pass 2 understated, Phase 6 self-audit |
| `CHANGESET_SUMMARY.md` | This section (§7.6) |
| `CONTINUE_FROM_HERE.md` | Pass-4 hand-off updated with what Pass 3 closed + remaining deferrals |

### Pass-3 subagents launched

1. **NLE plugin pricing market-fit signal** — Descript ARR + 78% transcription-only stat; OpusClip + Submagic + HeyGen pricing structure; **Topaz killed perpetual Oct 3, 2025** (strongest market signal of period); Aescripts top-sellers reconstruction; Mister Horse Animation Composer 900k-install model; SaaS↔OSS migration drivers; per-creator-type pain points
2. **Cursor/Copilot/IDE-agent patterns** — Cursor 2.0 Composer + checkpoints; Copilot Workspace editable-plan; Claude Code Skills open standard (agentskills.io); Aider git-commit-per-edit discipline; Cody 2-stage agentic chat; MCP Inspector; mapping IDE concepts to video editor; 3 patterns to copy + 3 to deliberately NOT copy

### F-numbers added in Pass 3

| F# | Title | Tier |
|---|---|---|
| F261 | Ship missing `OpenCut-Server.command` + `.sh` launchers (closes Wave I I1.4 ledger discrepancy) | **Now** |
| F262 | Fix uxp-api-notes.md sample-repo URL typo | Now |
| F263 | Re-run pip-audit on full `[all]` extras | Next |
| F264 | Add `npm audit --json` machine-parseable CI assertion | Now |
| F265 | UDT test harness for all 18 `ocXxx` JSX functions | Later |
| F266 | Document 2-function CEP residual + drop-QE plan | Now |
| F267 | UDT test harness for 14 low-risk JSX→UXP ports | Next |
| F268 | Adobe Exchange storefront listing | Next |
| F269 | Premium model-pack bundling format | Later |
| F270 | README "$1,400/yr" marketing copy refresh | Now |
| F271 | Per-feature VRAM requirement UI surface | Next |
| F272 | Wedding-specific Skill (color match + beat sync + 4-min reel) | Next |

### Cumulative impact (Pass 1 + Pass 2 + Pass 3)

| Metric | Pass 1 | Pass 2 | Pass 3 | Total |
|---|---:|---:|---:|---:|
| Files created in `.ai/research/2026-05-17/` | 10 | 6 | 4 | 20 |
| Root files created | 1 (PROJECT_CONTEXT) | 0 | 0 | 1 |
| Root files edited | 2 (AGENTS, ROADMAP) + 1 (CLAUDE gitignored) | 1 (ROADMAP) + 1 (PROJECT_CONTEXT) | 1 (ROADMAP) + 1 (PROJECT_CONTEXT) | 4 unique tracked + 1 gitignored |
| Lines created in new files | ~2,770 | ~1,350 | ~940 | ~5,060 |
| Lines added to edited root files | ~94 | ~480 | ~190 | ~764 |
| F-numbers newly proposed | F121-F190 (70) | F191-F260 (70) | F261-F272 (12) | **152 new** |
| Source citations (R-prefixed) | 138 | +85 | +20 | ~243 unique |
| Subagents launched | 3 | 3 | 2 | **8 total** |
| Live commands executed | 0 | 0 | 6 | 6 |
| Python / JS / config code changes | 0 | 0 | 0 | 0 |
| Tests added or modified | 0 | 0 | 0 | 0 |
| Dependency pins changed | 0 | 0 | 0 | 0 (recommendations only) |
| Commits made | 0 | 0 | 0 | 0 |
| Pushes attempted | 0 | 0 | 0 | 0 (auth blocked from this VM) |

---

## 8. Recommended next-session actions (in priority order)

1. **Commit the 7-file dirty hardening batch** as F138 (single PR: *"Harden auth loopback classification, UNC realpath, helper lifecycle, and safe_bool follow-up"*).
2. **Push the 25-commit backlog** to `SysAdminDoc/OpenCut` (auth needs to come from a machine with push permission — this VM is blocked).
3. **Run the `gh issue` seeder** (F182) — `python scripts/seed_github_issues.py` — to populate the contributor channel from `.github/issue-seeds.yml`.
4. **Take F121 + F122 + F126 + F130 + F133 + F135 + F137** as a single dependency-bump release (v1.33.0) — they are independent low-risk security fixes with no API impact.
5. **Schedule F123 (pydub) + F128 (FFmpeg filter regression suite)** as a paired pre-release task. These unblock F129 bundled FFmpeg bump and full Python 3.13 support.
6. **Open an RFC issue for F127 (Python 3.10 floor + Transformers v5)** — this is the highest-impact strategic decision in the audit and deserves a dedicated discussion thread before implementation.
7. **Open an RFC issue for F143 (`/agent/chat` conductor)** — the flagship Next-tier item; design space includes UI surface (CEP+UXP), LLM provider routing, timeline diff representation, and post-turn self-review semantics.

---

## 9. Pass 4 additions (same day, release-smoke validation)

Pass 4 converted the research checkpoint from "well documented" to "release-gate clean".

### Files extended in Pass 4

| Path | What was added |
|---|---|
| `PROJECT_CONTEXT.md` | Updated "last consolidated" to Pass 4, changed the F-number range to F001-F272, replaced the stale "uncommitted hardening" section with Pass-4 validation status, and added release-smoke PASS details. |
| `ROADMAP.md` | Added v4.7 status note and a release-smoke validation addendum. |
| `.ai/research/2026-05-17/LIVE_VERIFICATION.md` | Added §8 with first failed release-smoke run, Ruff cleanup actions, final PASS matrix, and targeted `119 passed` validation. |
| `.ai/research/2026-05-17/SOURCE_REGISTER.md` | Added Pass-4 local command evidence R-P4-L01 through R-P4-L09 and external source refresh R-P4-E01 through R-P4-E05. |
| `.ai/research/2026-05-17/RESEARCH_LOG.md` | Added Pass-4 phases, validation results, and saturation note. |
| `.ai/research/2026-05-17/CONTINUE_FROM_HERE.md` | Updated header from Pass 3 to Pass 5 and added Pass-4 handoff state. |

### Code cleanup included in Pass 4

| Scope | Change |
|---|---|
| F138 hardening batch | Kept the security changes from the dirty working tree: loopback classification, UNC realpath guard, FFmpeg helper lifecycle, nested user-data writes, and safe_bool follow-up. |
| Release-smoke lint scope | Applied Ruff safe fixes for unused imports and import ordering in `opencut/` and `scripts/` so `ruff check opencut --select E,F,I --ignore E501,E402` passes. |

### Validation after Pass 4

| Command | Result |
|---|---|
| `python -m opencut.tools.dump_route_manifest --check` | PASS |
| `python scripts/sync_version.py --check` | PASS |
| `python scripts/bootstrap_check.py` | PASS |
| `python -m pip_audit -r requirements-lock.txt` | PASS |
| `ruff check opencut --select E,F,I --ignore E501,E402` | PASS |
| Targeted pytest slice | PASS — `119 passed` |
| `python scripts/release_smoke.py --json` | PASS — release-smoke pytest-fast `232 passed` |

### Cumulative impact after Pass 4

| Metric | Total |
|---|---:|
| Research files in `.ai/research/2026-05-17/` | 20 |
| Root files created | 1 (`PROJECT_CONTEXT.md`) |
| F-numbers newly proposed this research date | 152 (F121-F272) |
| Local validation gates passed in final state | route manifest, version sync, bootstrap, model cards, license gate, roadmap lint, Ruff, pytest-fast, pip-audit, npm advisory allow-list, panel-source |
| Commits made by Pass 4 | 1 local checkpoint commit |

---

## 10. Pass 5 additions (same day, launcher/docs implementation)

Pass 5 converted three Pass-3 Now items from research findings into repository changes.

### Files added or edited in Pass 5

| Path | Change |
|---|---|
| `.gitattributes` | Added LF line-ending rules for POSIX launchers so shell scripts remain runnable from release/source checkouts. |
| `OpenCut-Server.command` | Added macOS double-click launcher that delegates to the POSIX shell launcher. |
| `OpenCut-Server.sh` | Added Linux/POSIX launcher with `OPENCUT_HOME`, bundled/system Python 3.9+ detection, bundled FFmpeg path support, bundled model environment variables, and `python -m opencut.server` startup. |
| `README.md` | Replaced the generic local-first lead with the quantified "$1,400/year" subscription-replacement lead and added macOS/Linux launch instructions. |
| `extension/com.opencut.uxp/uxp-api-notes.md` | Corrected the UXP sample repository URL to `AdobeDocs/uxp-premiere-pro-samples`. |
| `ROADMAP.md` | Added v4.8 status/addendum and marked F261/F262/F270 closed. |
| `PROJECT_CONTEXT.md` | Updated canonical context for Pass 5 and marked cross-platform launchers/README positioning current. |
| `.ai/research/2026-05-17/LIVE_VERIFICATION.md` | Updated the launcher gap to closed after Pass 5 and added a Pass-5 implementation note. |
| `.ai/research/2026-05-17/INSTALLER_AUDIT.md` | Updated launcher inventory to show the macOS/Linux files now exist. |
| `.ai/research/2026-05-17/CONTINUE_FROM_HERE.md` | Moved the handoff to Pass 6 and listed remaining immediate work. |

### Items closed in Pass 5

| F# | Result |
|---|---|
| F261 | Closed — macOS/Linux source launchers exist. |
| F262 | Closed — UXP docs URL typo fixed. |
| F270 | Closed — README lead and launcher instructions updated. |

### Validation after Pass 5

| Command | Result |
|---|---|
| `git diff --check` | PASS |
| `python scripts/release_smoke.py --json` | PASS — all 11 release-smoke steps green; pytest-fast reported `232 passed` |

---

## 12. Pass 7 additions (same day, F199 implementation)

Pass 7 closed F199 and corrected the earlier alias-count assumption.

### Files added or edited in Pass 7

| Path | Change |
|---|---|
| `opencut/tools/dump_api_aliases.py` | Added generator/checker for `/api` alias policy derived from the live route manifest. |
| `opencut/_generated/api_aliases.json` | New generated manifest: 233 total `/api/*` routes, 15 true aliases, 218 canonical `/api` routes. |
| `scripts/release_smoke.py` | Added `api-aliases` drift check to the release-smoke matrix. |
| `tests/test_api_aliases.py` | Added committed-vs-live manifest guard plus shape/policy checks. |
| `ROADMAP.md`, `PROJECT_CONTEXT.md`, `CONTINUE_FROM_HERE.md` | Marked F199 closed and corrected "233 alias pairs" wording. |

### Item closed in Pass 7

| F# | Result |
|---|---|
| F199 | Closed — generated `/api` alias policy manifest and drift checks. |

### Validation after Pass 7

| Command | Result |
|---|---|
| `python -m opencut.tools.dump_api_aliases --check` | PASS — 15 aliases, 218 canonical `/api` routes |
| `python -m pytest tests/test_api_aliases.py tests/test_release_smoke.py -q` | PASS — `16 passed` |
| `python -m py_compile opencut/tools/dump_api_aliases.py scripts/release_smoke.py` | PASS |
| `ruff check opencut/tools/dump_api_aliases.py scripts/release_smoke.py --select E,F,I --ignore E501,E402` | PASS |
| `python scripts/release_smoke.py --json` | PASS — all 12 release-smoke steps green; pytest-fast reported `236 passed` |

### Push status after Pass 5

`git push origin main` failed with HTTP 403:

```
remote: Permission to SysAdminDoc/OpenCut.git denied to MavenImaging.
fatal: unable to access 'https://github.com/SysAdminDoc/OpenCut.git/': The requested URL returned error: 403
```

The local commits are intact; pushing requires GitHub credentials with write access to `SysAdminDoc/OpenCut`.

---

## 11. Pass 6 additions (same day, F264/F266 implementation)

Pass 6 closed the remaining Pass-3 Now items.

### Files added or edited in Pass 6

| Path | Change |
|---|---|
| `extension/com.opencut.panel/scripts/check-advisories.mjs` | Added `--json` mode with stable `status`, `summary`, `allowed`, and `unwaived` fields. |
| `scripts/release_smoke.py` | Changed `npm-advisory` to invoke `check-advisories.mjs --json`, parse the machine-readable report, and fail on malformed output or unwaived advisories. |
| `docs/NODE_ADVISORIES.md` | Documented the JSON advisory-check command and release-smoke contract. |
| `docs/UXP_MIGRATION.md` | Added F266 section naming `ocAddNativeCaptionTrack` and `ocQeReflect`, plus the drop-QE rules. |
| `tests/test_release_smoke.py` | Added coverage for the JSON advisory contract and malformed-output failure. |
| `tests/test_node_advisories.py` | Added text guard for JSON output support in the Node checker. |
| `tests/test_uxp_migration_docs.py` | Added F266 documentation guard. |
| `ROADMAP.md`, `PROJECT_CONTEXT.md`, `CONTINUE_FROM_HERE.md` | Updated status to mark F264/F266 closed and identify F199/F179 as the next work. |

### Items closed in Pass 6

| F# | Result |
|---|---|
| F264 | Closed — release smoke now asserts machine-readable npm advisory JSON. |
| F266 | Closed — CEP residual and drop-QE plan documented. |

### Validation after Pass 6

| Command | Result |
|---|---|
| `node scripts/check-advisories.mjs --json` | PASS — one Vite advisory reported as allowed, zero unwaived |
| `python scripts/release_smoke.py --only npm-advisory --json` | PASS — `npm-advisory` reports `advisories on allow-list (1 allowed)` |
| `python -m pytest tests/test_release_smoke.py tests/test_node_advisories.py tests/test_uxp_migration_docs.py -q` | PASS — `20 passed` |
| `node --check extension/com.opencut.panel/scripts/check-advisories.mjs` | PASS |
| `python scripts/release_smoke.py --json` | PASS — all 11 release-smoke steps green; pytest-fast reported `232 passed` |

---

## 13. Pass 8 additions (same day, F191/F197 implementation)

Pass 8 closed the generated feature-readiness registry and registry-owned allowlist items.

### Files added or edited in Pass 8

| Path | Change |
|---|---|
| `opencut/tools/dump_feature_readiness.py` | Added generator/checker that scans route functions for public `checks.py` probes, joins endpoints to the live route manifest, and emits a deterministic readiness manifest. |
| `opencut/_generated/feature_readiness.json` | New generated manifest: 58 route-derived records across 67 direct route/check bindings. |
| `opencut/registry.py` | Loads generated readiness records, merges generated routes into curated records, exposes generated manifest metadata, and owns `NON_AI_CHECKS`. |
| `opencut/model_cards.py` | Imports `NON_AI_CHECKS` from `registry.py` so F115 and F191 share one allowlist. |
| `scripts/release_smoke.py` | Added `feature-readiness` drift check and included the generator tests in the release-gate pytest slice. |
| `tests/test_feature_readiness_generator.py` | Added committed-vs-live generator drift tests and route/probe binding assertions. |
| `tests/test_feature_registry.py` | Added generated-record merge coverage and shared allowlist assertions. |
| `ROADMAP.md`, `PROJECT_CONTEXT.md`, `ROUTE_READINESS_AUDIT.md`, `CONTINUE_FROM_HERE.md` | Marked F191/F197 closed and updated live counts/remaining gaps. |

### Items closed in Pass 8

| F# | Result |
|---|---|
| F191 | Closed — generated route/check readiness manifest loaded into `/system/feature-state`. |
| F197 | Closed — `NON_AI_CHECKS` now belongs to `registry.py`. |

### Validation after Pass 8

| Command | Result |
|---|---|
| `python -m opencut.tools.dump_feature_readiness` | PASS — 58 generated records / 67 route bindings |
| `python -m py_compile opencut/registry.py opencut/model_cards.py opencut/tools/dump_feature_readiness.py scripts/release_smoke.py` | PASS |
| `python -m pytest tests/test_feature_registry.py tests/test_feature_readiness_generator.py tests/test_model_cards.py tests/test_release_smoke.py -q` | PASS — `35 passed` |
| `python scripts/release_smoke.py --only feature-readiness --json` | PASS |
| `python scripts/release_smoke.py --json` | PASS — all 13 release-smoke steps green; pytest-fast reported `241 passed` |

---

## 14. Pass 9 additions (same day, F195 MCP curated tool expansion)

Pass 9 closed the missing post-Wave-M curated MCP tools item.

### Files added or edited in Pass 9

| Path | Change |
|---|---|
| `opencut/mcp_server.py` | Expanded `MCP_TOOLS` from 27 to 39 entries, added route mappings for 12 shipped routes, added Brand Kit and semantic-search action dispatch, and expanded MCP path validation for new scalar/array path keys. |
| `tests/test_mcp_server.py` | Added registration, route mapping, dispatch, special-action, and MCP path-validation coverage. |
| `scripts/release_smoke.py` | Added `tests/test_mcp_server.py` to the release-gate pytest slice. |
| `ROADMAP.md`, `PROJECT_CONTEXT.md`, `ROUTE_READINESS_AUDIT.md`, `FEATURE_BACKLOG_ADDENDUM.md`, `PRIORITIZATION_MATRIX.md`, `CONTINUE_FROM_HERE.md` | Marked F195 closed, updated MCP tool counts from 27 to 39, and recorded the remaining route/tooling gaps. |

### Items closed in Pass 9

| F# | Result |
|---|---|
| F195 | Closed — curated MCP coverage now includes face reshape, skin retouch, smart upscale, ElevenLabs TTS, caption QC, review bundles, C2PA provenance, marker import, capability probe, Brand Kit, semantic search, and spectral match. |

### Validation after Pass 9

| Command | Result |
|---|---|
| `python -m py_compile opencut/mcp_server.py scripts/release_smoke.py tests/test_mcp_server.py` | PASS |
| `python -m pytest tests/test_mcp_server.py tests/test_release_smoke.py -q` | PASS — `17 passed` |
| `ruff check opencut/mcp_server.py scripts/release_smoke.py tests/test_mcp_server.py --select E,F,I --ignore E501,E402` | PASS |
| `python scripts/release_smoke.py --json` | PASS — all 13 release-smoke steps green; pytest-fast reported `246 passed` |

---

## 15. Pass 10 additions (same day, F202 macOS notarization tooling)

Pass 10 closed the repository-side macOS signing/notarization release path.

### Files added or edited in Pass 10

| Path | Change |
|---|---|
| `scripts/notarize_macos.sh` | Added macOS-only Developer ID certificate import, hardened-runtime Mach-O signing, `xcrun notarytool submit --wait`, and notarized ZIP creation. |
| `.github/workflows/build.yml` | Tagged/manual macOS release builds now call the notarization script and upload `OpenCut-Server-macOS.zip` to GitHub Releases. |
| `docs/MACOS_NOTARIZATION.md` | Documents required GitHub secrets, local commands, and Apple references. |
| `tests/test_macos_notarization.py` | Static release-gate coverage for notarytool usage, hardened runtime signing, workflow wiring, and documentation. |
| `scripts/release_smoke.py` | Added `tests/test_macos_notarization.py` to the release-gate pytest slice. |
| `ROADMAP.md`, `PROJECT_CONTEXT.md`, `INSTALLER_AUDIT.md`, `FEATURE_BACKLOG_ADDENDUM.md`, `PRIORITIZATION_MATRIX.md`, `CONTINUE_FROM_HERE.md` | Marked F202 repository-side tooling closed and documented the remaining secret/live-service limitation. |

### Items closed in Pass 10

| F# | Result |
|---|---|
| F202 | Closed locally — macOS release builds have Developer ID signing + notarytool submission wiring; live Apple acceptance requires configured secrets. |

### Validation after Pass 10

| Command | Result |
|---|---|
| `python -m pytest tests/test_macos_notarization.py tests/test_release_smoke.py -q` | PASS — `15 passed` |
| `ruff check tests/test_macos_notarization.py scripts/release_smoke.py --select E,F,I --ignore E501,E402` | PASS |
| `C:\Program Files\Git\bin\bash.exe -n scripts/notarize_macos.sh` | PASS |
| `python scripts/release_smoke.py --json` | PASS — all 13 release-smoke steps green; pytest-fast reported `249 passed` |

---

## 16. Pass 11 additions (same day, F204 release SBOM attachment)

Pass 11 closed automatic SBOM attachment for release builds.

### Files added or edited in Pass 11

| Path | Change |
|---|---|
| `.github/workflows/build.yml` | Linux tagged/manual release builds now generate `dist/opencut-sbom.cyclonedx.json`, archive it as `OpenCut-SBOM-CycloneDX`, and upload it to GitHub Releases on tags. |
| `tests/test_release_sbom.py` | Added generator smoke coverage and workflow wiring assertions for SBOM generation/archive/upload. |
| `scripts/release_smoke.py` | Added `tests/test_release_sbom.py` to the release-gate pytest slice. |
| `ROADMAP.md`, `PROJECT_CONTEXT.md`, `INSTALLER_AUDIT.md`, `FEATURE_BACKLOG_ADDENDUM.md`, `PRIORITIZATION_MATRIX.md`, `CONTINUE_FROM_HERE.md` | Marked F204 closed and kept F219 as the deeper SBOM completeness gap. |

### Items closed in Pass 11

| F# | Result |
|---|---|
| F204 | Closed — release builds now attach the generated CycloneDX SBOM. |

### Validation after Pass 11

| Command | Result |
|---|---|
| `python -m pytest tests/test_release_sbom.py tests/test_release_smoke.py -q` | PASS — `14 passed` |
| `ruff check tests/test_release_sbom.py scripts/release_smoke.py --select E,F,I --ignore E501,E402` | PASS |
| `python -c "import pathlib, yaml; yaml.safe_load(pathlib.Path('.github/workflows/build.yml').read_text())"` | PASS |
| `python scripts/sbom.py --format json --output dist/opencut-sbom.cyclonedx.json` | PASS |
| `python scripts/release_smoke.py --json` | PASS — all 13 release-smoke steps green; pytest-fast reported `251 passed` |

---

## 17. Pass 12 additions (same day, F205 attempt + F207 installer FFmpeg manifest)

Pass 12 attempted F205 and closed F207.

### Files added or edited in Pass 12

| Path | Change |
|---|---|
| `installer/src/OpenCut.Installer/Models/AppConstants.cs` | Added bundled FFmpeg/ffprobe version constants and the installer manifest filename. |
| `installer/src/OpenCut.Installer/Models/InstallConfig.cs` | Added `InstallerManifestPath` for `~/.opencut/installer.json`. |
| `installer/src/OpenCut.Installer/Services/InstallEngine.cs` | Added a manifest-writing step that emits app/install paths, installer kind, FFmpeg versions, and install timestamp. |
| `installer/src/OpenCut.Installer/GlobalUsings.cs` | Added `System.Text.Json` for manifest serialization. |
| `OpenCut.iss` | Added bundled FFmpeg/ffprobe version defines and an Inno post-install manifest writer. |
| `tests/test_ffmpeg_installer_manifest.py` | Added static coverage for WPF/Inno FFmpeg version manifest contracts. |
| `scripts/release_smoke.py` | Added `tests/test_ffmpeg_installer_manifest.py` to the release-gate pytest slice. |
| `ROADMAP.md`, `PROJECT_CONTEXT.md`, `INSTALLER_AUDIT.md`, `FEATURE_BACKLOG_ADDENDUM.md`, `PRIORITIZATION_MATRIX.md`, `CONTINUE_FROM_HERE.md` | Marked F207 closed and recorded the F205 coverage timeout blocker. |

### Items closed in Pass 12

| F# | Result |
|---|---|
| F207 | Closed — bundled FFmpeg version is now in `AppConstants.cs` and the installed `~/.opencut/installer.json` manifest for both WPF and Inno paths. |

### Validation after Pass 12

| Command | Result |
|---|---|
| `python -m pytest tests/test_ffmpeg_installer_manifest.py tests/test_release_smoke.py -q` | PASS — `15 passed` |
| `ruff check tests/test_ffmpeg_installer_manifest.py scripts/release_smoke.py --select E,F,I --ignore E501,E402` | PASS |
| `python -m py_compile scripts/release_smoke.py` | PASS |
| `.\ffmpeg\ffmpeg.exe -version` | PASS — reports `8.0.1-essentials_build-www.gyan.dev` |
| `python scripts/release_smoke.py --json` | PASS — all 13 release-smoke steps green; pytest-fast reported `254 passed` |
| `dotnet build installer/src/OpenCut.Installer/OpenCut.Installer.csproj --no-restore` | BLOCKED — no .NET SDK installed on this VM |

### F205 attempt

Installed missing local plugins with `python -m pip install pytest-cov pytest-xdist`, then attempted the CI-style coverage measurement with the floor disabled. The command timed out after 20 minutes and produced no `dist\coverage-f205.json`, so F205 remains open.

---

## 18. Pass 13 additions (same day, F208 OpenAPI contract gate)

Pass 13 closed F208.

### Files added or edited in Pass 13

| Path | Change |
|---|---|
| `opencut/openapi.py` | Converts Flask path parameters to OpenAPI `{param}` syntax, emits path parameter metadata, generates unique path-qualified operation IDs for aliased endpoints, and adds 400/403 response shapes for every mutating method. |
| `tests/test_openapi_contract.py` | New contract tests for `/openapi.json` route parity, path parameters, operation IDs, response schemas, and `/api/openapi.json` path-parameter syntax. |
| `scripts/release_smoke.py` | Added `tests/test_openapi_contract.py` to the `pytest-fast` release gate. |
| `ROADMAP.md`, `PROJECT_CONTEXT.md`, `TEST_COVERAGE_GAPS.md`, `FEATURE_BACKLOG_ADDENDUM.md`, `PRIORITIZATION_MATRIX.md`, `SOURCE_REGISTER.md`, `RESEARCH_LOG.md`, `CONTINUE_FROM_HERE.md` | Marked F208 closed and updated the next-pass handoff. |

### Items closed in Pass 13

| F# | Result |
|---|---|
| F208 | Closed — OpenAPI structural validity and route coverage are pinned in release smoke. |

### Validation after Pass 13

| Command | Result |
|---|---|
| `python -m pytest tests/test_openapi_contract.py tests/test_release_smoke.py -q` | PASS — `16 passed` |
| `ruff check opencut/openapi.py tests/test_openapi_contract.py scripts/release_smoke.py --select E,F,I --ignore E501,E402` | PASS |
| `python -m py_compile opencut/openapi.py scripts/release_smoke.py tests/test_openapi_contract.py` | PASS |
| `python scripts/release_smoke.py --json` | PASS — all 13 release-smoke steps green; pytest-fast `258 passed` |

### Remaining immediate work

F205 remains open after the Pass 12 coverage timeout. The next local-verifiable Now items are F209, F218, and F219.

---

## 19. Pass 14 additions (same day, F209 MCP route consistency)

Pass 14 closed F209.

### Files added or edited in Pass 14

| Path | Change |
|---|---|
| `opencut/mcp_server.py` | Corrected `opencut_chat_edit` from planned `/agent/chat` to the shipped `POST /chat` route. |
| `tests/test_mcp_server.py` | Added a live Flask route-consistency test covering all 39 MCP tool routes plus dynamic action routes for music, style transfer, Brand Kit, semantic search, and job status. |
| `ROADMAP.md`, `PROJECT_CONTEXT.md`, `TEST_COVERAGE_GAPS.md`, `FEATURE_BACKLOG_ADDENDUM.md`, `PRIORITIZATION_MATRIX.md`, `SOURCE_REGISTER.md`, `RESEARCH_LOG.md`, `CONTINUE_FROM_HERE.md` | Marked F209 closed and updated the next-pass handoff. |

### Items closed in Pass 14

| F# | Result |
|---|---|
| F209 | Closed — curated MCP route drift now fails release smoke. |

### Validation after Pass 14

| Command | Result |
|---|---|
| `python -m pytest tests/test_mcp_server.py tests/test_release_smoke.py -q` | PASS — `18 passed` |
| `ruff check opencut/mcp_server.py tests/test_mcp_server.py --select E,F,I --ignore E501,E402` | PASS |
| `python -m py_compile opencut/mcp_server.py tests/test_mcp_server.py` | PASS |
| live route-table probe | PASS — 39 MCP tools / 39 route mappings / 0 missing backend routes |
| `python scripts/release_smoke.py --json` | PASS — all 13 release-smoke steps green; pytest-fast `259 passed` |

### Remaining immediate work

F205 remains open after the Pass 12 coverage timeout. The next local-verifiable Now items are F218 and F219.
