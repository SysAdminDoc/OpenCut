# OpenCut — Memory Consolidation

**Audit date:** 2026-05-17
**Scope:** Reconcile every AI/agent instruction file, roadmap document, and historical changelog into a single coherent canonical layer (`PROJECT_CONTEXT.md`) without losing tool-specific or wave-specific context.

---

## 1. Inventory

### Instruction files (LLM/agent)

| File | Audience | Status | Action |
|---|---|---|---|
| `AGENTS.md` | Codex, OpenAI-flavoured agents | Pointer only ("see ./CLAUDE.md"). 9 lines. | **Preserve** — Codex consults this filename specifically. Add a "Canonical Project Context" pointer to `PROJECT_CONTEXT.md`. |
| `CLAUDE.md` | Claude Code, Claude Desktop, any Anthropic agent | The **canonical** developer memory. 1,509 lines. Module-by-module deep gotchas, every async-job rule, every safe_bool pitfall, every UXP-vs-CEP convention. | **Preserve** — this is the actual working memory. Add the same pointer header. Do not summarise away — the value is in the granular gotchas. |
| `CLAUDE-HANDOFF-PROMPT.md` | Future Claude sessions taking over | 139-line boilerplate prompt with handoff checklist. | **Preserve as-is** — narrow purpose. |
| `.github/copilot-instructions.md` | GitHub Copilot | Present per ROADMAP citation `[V43-L04]`. | Inspect; preserve. Add canonical pointer if it doesn't already point at `PROJECT_CONTEXT.md`. |

No `GEMINI.md`, `.cursor/rules/`, `.cursorrules`, `.windsurfrules`, `.claude/`, `.claude-instructions` were found in this repo. The agent-memory surface is currently {AGENTS.md → CLAUDE.md, CLAUDE-HANDOFF-PROMPT.md, copilot-instructions.md}.

### Roadmap documents

| File | Lines | Vintage | Reconciliation |
|---|---|---|---|
| `ROADMAP.md` | 2,703 | v4.3 — 2026-05-16 | **Live decision layer.** Contains the full F001-F120 ledger, all wave-letter (A-T) tier tables, source appendix. Authoritative when it disagrees with anything else. |
| `ROADMAP-NEXT.md` | 843 | 2026-04-17 | Earlier Wave A-K plan. Mostly shipped through v1.28.0. **Keep as archive** — has source URLs and per-wave gotchas not duplicated elsewhere. |
| `ROADMAP-COMPLETED.md` | 42 | ~v1.9.26 era | High-level table of Phases 0-6. **Stale** — last meaningful update was many waves ago. Mark as archive; do not delete; note in `PROJECT_CONTEXT.md` that ROADMAP.md tier table is the live truth. |
| `docs/ROADMAP.md` | — | mirror | Same file as repo root — kept for legacy doc link compatibility. |
| `docs/ROADMAP-COMPLETED.md` | — | mirror | Same. |
| `MODERNIZATION.md` | 214 | 2026-04-06 baseline v1.9.18 | Per-dependency, per-model upgrade ledger. Most Tier-1 items show DONE. Still useful as the **canonical dep/model audit table** until a successor is written. |
| `AUDIT.md` | 624 | v1.11.0, 2026-04-14 | Competitive analysis vs Descript / CapCut / Runway / AutoCut / OpusClip / Topaz / Adobe Firefly. **Largely superseded** by the v4.3 audit + `research.md` but unique commentary on AutoCut / AutoPod direct competitors. |
| `research.md` | 329 | v1.28.2, 2026-04-22 | Competitive gap analysis vs Captions.ai / DaVinci 21 / HeyGen / Premiere 26 / Topaz / Crayotter / Submagic / FireRed-OpenStoryline. Section §1.1 MCP server proposal — **already implemented** in v1.30.0 (`opencut/mcp_server.py`). |
| `features.md` | 2,932 | v1.10.4 baseline, 2026-04-12 (last updated 2026-04-14) | 402 features across 82 categories. **Oldest aspirational plan.** Mostly absorbed into ROADMAP.md tiers but not all 402 items have explicit ship status. |
| `docs/RESEARCH.md` | — | — | Older OSS feature research summary. Still cited by README. |
| `docs/UXP_MIGRATION.md` | — | — | Live CEP→UXP migration plan. |
| `docs/WINDOWS_ARM64_PACKAGING.md` | — | F101 deliverable | Live. |
| `docs/NODE_ADVISORIES.md` | — | F095 deliverable | Live. |
| `docs/MODELS.md` | — | F115 deliverable, auto-generated | Live. |

### Changelogs

| File | Role | Action |
|---|---|---|
| `CHANGELOG.md` | Per-release human changelog through v1.32.0 (1,197 lines). | **Authoritative.** Keep. |
| `CODEX-CHANGELOG.md` | One-time Codex handoff snapshot (2026-04-15). 220 lines. All described work is merged. | **Preserve as historical artefact.** Add a top-of-file banner noting it is a snapshot and `CHANGELOG.md` is the live record. |
| `pt.log`, `build104.log`, `build35.log`, `pytest-audit.log`, `pytest-first-failure.log`, `pytest-full.log` | Build / test log files in repo root. | **Investigate** — these look like accidental commits; check `.gitignore` and consider removing in a future cleanup pass. Out of scope for this consolidation. |

### Memory directories

None present (no `.ai/`, `.claude/`, etc.). This research run **creates `.ai/research/2026-05-17/`**.

---

## 2. Reconciliation logic

### Where files agree

| Topic | Sources | Verdict |
|---|---|---|
| Local-first philosophy, no mandatory cloud, MIT license, FFmpeg-first architecture | README, CLAUDE.md, all roadmap files | Universally consistent. |
| Backend = Flask on `127.0.0.1:5679`, WebSocket on `5680`, async-job pattern via `@async_job`, SQLite job persistence | CLAUDE.md, README, CHANGELOG.md | Consistent. |
| Wave-letter ship history (A → M shipped, N → T planned) | ROADMAP.md, ROADMAP-NEXT.md, CHANGELOG.md | Consistent. |
| F-number ship history (F099, F100, F102-F106, F109-F112, F115-F118, F120, F006, F010, F011, F066, F095, F097 shipped) | ROADMAP.md v4.3 implementation checklist, git log | Consistent. |
| CEP EOL ~Sept 2026 | README FAQ, ROADMAP.md, ROADMAP-NEXT.md, docs/UXP_MIGRATION.md | Consistent. |

### Where files disagree

| Topic | Disagreement | Resolution (evidence) |
|---|---|---|
| **Route count** | README badge "1344"; README narrative "1,334"; ROADMAP.md baseline "1,344"; `routes/__init__.py` docstring "1,152" (legacy); v4.3 baseline "1,275"; live manifest 1,359. | **The manifest wins.** F099's `opencut/_generated/route_manifest.json` is the single source of truth. The README badge has drifted across multiple releases; the F099/F098 pipeline is supposed to regenerate it but the docstring at the top of the README still notes the badge is generated. Practical reading: route count is ≥1,300; do not cite a specific number outside the manifest. |
| **Blueprint count** | CLAUDE.md says "88 blueprints (17 original + 71 expansion)" *and* "98 blueprints" *and* "99 blueprints" in different sections; ROADMAP.md says 101; manifest says 101. | **Manifest wins** (101). CLAUDE.md narrative needs a regeneration sweep next time the file is touched. |
| **Test count** | README "5,742+ tests across 77 test files"; CHANGELOG.md "7,551 tests"; ROADMAP.md "7,600+ tests"; `ls tests/` = 131 files. | Files and test count both grew; CLAUDE.md "5,742+" is stale, README is stale. Live count is in the pytest output; ≥7,600 is the current marketing number. |
| **Number of v1.32.0 features** | README "1,344 routes, 8 panel tabs, 50+ sub-tabs"; ROADMAP.md "302 features across 62 categories"; features.md "402 features across 82 categories"; F-number ledger "F001-F120". | All three count *different things*. **Routes** are HTTP endpoints (1,359 today). **Features** in `features.md` are workflow concepts (402 aspirational). **F-numbers** are governance/infra items from the v4.3 audit (currently F001-F120, ~22 shipped in the *Now* tier of the past 30 days). All three coexist; `PROJECT_CONTEXT.md` will name the distinction. |
| **GPU isolation** | ROADMAP-NEXT.md "P0, not implemented as of v1.17.0"; ROADMAP.md "GPU semaphore Wave 3A MVP shipped v1.23.0"; v4.3 audit "still not full process isolation". | **Both true.** The simple `MAX_CONCURRENT_GPU_JOBS = 3` semaphore is in place. **Full out-of-process worker pool is not.** Note this nuance in `PROJECT_CONTEXT.md`. |
| **UXP parity %** | README "near-complete feature parity"; CLAUDE.md "UXP panel at ~85% parity, CEP end-of-life ~Sept 2026"; ROADMAP-NEXT.md "Tier 3 H3.5 WebView UI UXP migration research spike". | **CLAUDE.md is the realistic number** (~85%). README is marketing-soft. The remaining 15% is workflow builder, full settings, plugin UI per CLAUDE.md. |
| **MCP server status** | research.md §1.1 (Apr 2026) "HIGH priority — not implemented"; CHANGELOG.md v1.30.0 (2026-05-04) "MCP sidecar with 27 tools shipped"; `opencut/mcp_server.py` exists, 930 lines. | **research.md is stale on this one item.** MCP server shipped 2026-05-04. Note in `PROJECT_CONTEXT.md` that research.md predates v1.30.0 ship date. |
| **Caption translation** | research.md §1.5 "MEDIUM-HIGH gap"; no clear ship signal. | Not yet shipped per `git log` grep. Still a real gap. Confirmed for backlog. |

### Where files duplicate

| Topic | Duplicated in | Action |
|---|---|---|
| FFmpeg-first principle, "one dep per feature", graceful degradation | README, CLAUDE.md, ROADMAP.md, ROADMAP-NEXT.md, MODERNIZATION.md | One canonical statement in `PROJECT_CONTEXT.md`. Other files can keep their context-local restatement (they're not wrong). |
| Wave-letter shipping cadence | ROADMAP.md, ROADMAP-NEXT.md, CHANGELOG.md | Cadence table in `PROJECT_CONTEXT.md` references the **CHANGELOG.md** as authoritative ship dates and **ROADMAP.md** as authoritative for what's planned next. |
| Per-module gotchas | CLAUDE.md (canonical) | Do **not** duplicate. `PROJECT_CONTEXT.md` says "see CLAUDE.md for module-level patterns/gotchas." |
| Per-dependency upgrade plans | MODERNIZATION.md | Likewise — `PROJECT_CONTEXT.md` links out, doesn't restate the table. |
| Per-feature competitive analysis | AUDIT.md (v1.11), research.md (v1.28.2), ROADMAP.md v4.3 audit | Three eras of the same exercise. v4.3 (in ROADMAP.md) is the most recent and the live decision layer. AUDIT.md + research.md are useful for **historical reasoning** ("why did we reject mobile editor?", "why did we adopt MCP?"). Keep both as archive. |

### Where files conflict in a way that requires a decision

| Conflict | Files | Resolution + open question |
|---|---|---|
| **Should ROADMAP-COMPLETED.md continue to exist?** | ROADMAP-COMPLETED.md is 42 lines, last meaningfully updated mid-Phase-6 era. ROADMAP.md and CHANGELOG.md now carry the live shipped state. | **Keep, but mark archive.** Future "what shipped" reads should go to CHANGELOG.md. ROADMAP-COMPLETED.md keeps a top-of-file note. Removing it is a separate cleanup PR. |
| **Should features.md continue to exist?** | features.md is 2,932 lines, 402 features, 82 categories, oldest plan. Not every feature has a ship/reject status. | **Keep as long-range catalogue.** Treat it the way a product catalogue treats moonshot ideas — they live somewhere visible but are not promised. Cross-reference where ROADMAP.md F-numbers or wave letters absorbed an items. Out of scope for this consolidation pass to do that reconciliation row-by-row. |
| **Should AUDIT.md / research.md collapse into one "competitive research" doc?** | Three separate competitive audits exist (AUDIT.md April, research.md April, ROADMAP.md v4.3 May). | **Yes, eventually.** Out of scope for this pass. For now, ROADMAP.md v4.3 is the live audit; AUDIT.md + research.md are predecessors. Add a top-of-file note on each older audit pointing to ROADMAP.md v4.3. |
| **Does CLAUDE.md actively conflict with itself on counts?** | CLAUDE.md says "88 Blueprints (17 original + 71 expansion)" in one place and "1,152 total routes" in another, even though the manifest says 101 blueprints / 1,359 routes. | **CLAUDE.md narrative is partially stale.** Module-level gotchas are still correct (those don't change with new blueprints). The opening "Architecture" + "Route Blueprints" subsections need a refresh sweep when CLAUDE.md is next touched. Not blocking. Note in `MEMORY_CONSOLIDATION.md`. |

---

## 3. Open conflicts (not resolved by source-code evidence)

1. **`features.md` aspirational plan vs ROADMAP.md tier table**. Many `features.md` items live in implicit limbo — neither explicitly rejected nor explicitly absorbed. Resolution requires a 1-2 day reconciliation pass that walks all 402 features and marks each as `[shipped]` / `[planned]` / `[rejected]` / `[absorbed-into-F###]` / `[absorbed-into-W###.##]`. Not done in this audit.

2. **MODERNIZATION.md baseline is v1.9.18 (2026-04-06)**. Items like "auto-editor v30 Nim binary" are marked DONE; items like "AV1 export preset" are marked DONE. But the file has no update for the **post-v1.18 wave** that introduced 27 new model surfaces (SEA-RAFT, Cutie, DEVA, EchoMimic V3, etc.) — each of which deserves its own modernization entry. Resolution: regenerate MODERNIZATION.md from the live model_cards (F115). Not done in this audit.

3. **Plugin marketplace** is touched in three places: features.md (32.x plugin marketplace), ROADMAP.md F033 "community template marketplace" (Later), F079 "in-app feature marketplace with RCE" (Rejected). Are these the same thing or different? The F033/F079 split is intentional (sandboxed vs not). features.md's vague 32.x should be marked absorbed into F033.

4. **`docs/` mirror copies of `ROADMAP.md` and `ROADMAP-COMPLETED.md`** drift. Decision: pick a canonical location (repo root vs `docs/`), then make the other a symlink or a generation step. Out of scope for this audit.

---

## 4. Recommended changes to AGENTS.md and CLAUDE.md

Add a single block at the top of each:

```markdown
## Canonical Project Context

For consolidated project memory, current architecture, shipping cadence,
known gaps, and the live roadmap entry points, see `PROJECT_CONTEXT.md`
at the repo root. That file is the cross-tool source of truth; this file
remains the {Codex / Claude / Copilot}-specific instruction file.

Live roadmap: `ROADMAP.md` (v4.3, 2026-05-16+).
Live changelog: `CHANGELOG.md` (through v1.32.0).
```

**Do not merge `CLAUDE.md` into `PROJECT_CONTEXT.md`.** CLAUDE.md is the deep gotcha file (1,509 lines of module-by-module hazards) — it is genuinely Claude-specific in that it's structured for an LLM agent rather than a human contributor. Leaving it intact is the right call.

**Do not merge `AGENTS.md` into anything else either** — Codex / other-agent tools look up `AGENTS.md` by filename. The 9-line pointer is correct as-is plus the new canonical-context block.

---

## 5. What gets extracted to PROJECT_CONTEXT.md

| Section in PROJECT_CONTEXT.md | Sourced from |
|---|---|
| Identity (one paragraph) | README + CLAUDE.md |
| Current code surface area numbers | live manifest (route_manifest.json, model_cards.json, registry.py) |
| Architecture diagram + key modules | CLAUDE.md "Architecture" section, condensed; routes list condensed |
| Shipping cadence table (waves + F-numbers) | CHANGELOG.md + git log + ROADMAP.md |
| Active vs planned vs rejected backlog | ROADMAP.md v4.3 tier table |
| Documentation map | this file (§ 1) |
| Hard constraints (license, runtime, network, security) | ROADMAP.md v4.3 phase 0 + SECURITY.md |
| Where to look for each kind of question | this file (§ 4 below) |
| Known gaps that no document currently owns | this file (§ 7 below) |
| Pointer to research run artefacts | this file + ROADMAP.md |

---

## 6. Where to look for each kind of question

| Question | Look here |
|---|---|
| "Why is route X failing with 503?" | `opencut/checks.py` → `check_X_available()`; `opencut/registry.py` for readiness state; `docs/MODELS.md` for install hint. |
| "How do I write an async route?" | `CLAUDE.md` → `@async_job` decorator section and gotcha list. |
| "What's the SQLite job store schema?" | `opencut/job_store.py` (~200 lines). |
| "How do I add a new optional AI extra?" | `opencut/checks.py` (add `check_X_available()`), `opencut/model_cards.py` (add card), `opencut/registry.py` (add `FeatureRecord`), then a route in the appropriate wave file. |
| "What's planned for the next release?" | `ROADMAP.md` → Now / Next tier tables; `gh issue list` (currently empty until F097 seeds run). |
| "What shipped in the last release?" | `CHANGELOG.md`. |
| "Why was X rejected?" | `ROADMAP.md` → Rejected tier (F043, F078–F086) with one-line rationale each. |
| "How does the CEP panel call ExtendScript?" | `extension/com.opencut.panel/host/index.jsx` (~2230 lines); panel side uses `evalScript()` via `PremiereBridge` abstraction in `main.js`. |
| "How does the UXP panel access Premiere?" | `extension/com.opencut.uxp/main.js` → `PProBridge` class; `premierepro` UXP module lazy-imported. |
| "What does the MCP server expose?" | `opencut/mcp_server.py` (930 lines, 27 tools). |
| "Where are dependency upgrade notes?" | `MODERNIZATION.md` (baseline old, but the table format is the right place to extend). |
| "What's the threat model?" | `SECURITY.md` + `opencut/auth.py` + `opencut/security.py`. |
| "How are version surfaces synced?" | `scripts/sync_version.py --check`. |
| "Why does README say 1,344 routes when the manifest says 1,359?" | Marketing badge has drifted; F099 manifest is the truth. |

---

## 7. Gaps that no current document owns

1. **A single "what is the relationship between a wave letter and an F number?" explainer.** They both appear in ROADMAP.md without a top-level distinction. `PROJECT_CONTEXT.md` will own this.
2. **An explicit "what is in the dirty working tree right now and why" note**, to prevent the next session from re-discovering the in-flight `safe_bool` + UNC + loopback hardening from scratch. `PROJECT_CONTEXT.md` § "Active in-flight changes" will own this.
3. **A "core/ taxonomy"** — there are 523 files in `opencut/core/`; no document classifies them. CLAUDE.md groups some, MODERNIZATION.md grouped 68 of them in April. The remaining ~455 are uncategorised. Resolution: too much work to do row-by-row in this audit; instead, lean on the **alphabetical grouping** that already lives in the wave files and the registry. `PROJECT_CONTEXT.md` will point at the routes/core directories and let `Grep` do the rest.
4. **A "what is each `.bat` / `.vbs` / `.ps1` / `.iss` for?"** — installer surface is wide (`Install.bat`, `Install.ps1`, `InstallerBuilder.ps1`, `OpenCut.iss`, `OpenCut-Launcher.vbs`, `OpenCut-Server.bat`, `OpenCut-Server.vbs`, `Uninstall.bat`). README mentions some. `DEVELOPMENT.md` is 149 lines and doesn't fully enumerate. Documenting this is a future cleanup, not this run.
5. **Test failure history / flaky-test ledger** — no document tracks which tests are known-flaky or under what conditions they fail. Out of scope.
6. **`pt.log`, `build104.log`, `build35.log`, `pytest-*.log` in repo root** — these look accidentally committed. Cleanup recommendation noted but not executed.

---

## 8. Action items from this consolidation

1. **Write `PROJECT_CONTEXT.md`** — done in this research run.
2. **Add canonical-context pointer block to `AGENTS.md`** and `CLAUDE.md` — done in this research run (minimal edit, preserves all existing content).
3. **Add `.github/copilot-instructions.md` pointer** — TODO (deferred; out of scope unless file is touched by another task).
4. **Banner CODEX-CHANGELOG.md as a snapshot** — deferred (file is correct; a 3-line banner edit could go in next cleanup PR).
5. **Banner AUDIT.md and research.md as "predecessor to ROADMAP.md v4.3"** — deferred.
6. **Reconciliation of features.md against F-numbers** — deferred to a dedicated 1-2 day pass.
7. **Regenerate MODERNIZATION.md from `model_cards.py`** — deferred; the data is in `docs/MODELS.md` already.
8. **Decide canonical home for ROADMAP duplicates** (repo root vs `docs/`) — deferred to repo maintainer.
9. **Commit the 7 in-flight hardening changes** before resuming feature work — recommendation, not done in this audit because user did not authorise commits.
