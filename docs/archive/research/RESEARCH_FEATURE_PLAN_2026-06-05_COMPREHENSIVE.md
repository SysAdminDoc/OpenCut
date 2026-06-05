# Project Research and Feature Plan

Prepared: 2026-06-05

Repository baseline: `main` at `07b70c0` (`docs: add release sbom fidelity research note`)

Output type: planning-only companion report. No product code, tests, package metadata, roadmap ledger, or active TODO queue was changed by this report.

## Executive Summary

OpenCut is no longer a small Premiere helper. The live repository presents a local-first editing automation platform with a Python/Flask loopback server, a CEP panel, a UXP migration path, 1,523 generated API routes across 107 blueprints, 599 core modules, 1,466 opt-in MCP tools, 55 generated model cards, and release checks around route manifests, model cards, feature readiness, Node advisories, SBOM generation, and dependency audits.

The highest-value work is not another broad feature wave. It is converting existing reach into trusted, deterministic, user-visible workflows:

1. Fix the current security evidence gaps: `requirements-lock.txt` still pins vulnerable `idna==3.11`, the optional `[all]` audit currently has five unwaived Torch/Transformers findings, and the release SBOM is declared-dependency-only while the artifact name can be read as a resolved vulnerability inventory.
2. Finish the UXP/WebView migration on least-privilege terms: Adobe's current UXP docs make manifest versioning, filesystem scope, WebView domain scope, external-launch permissions, clipboard API behavior, and UDT/Premiere evidence first-class release requirements.
3. Close release and CI trust gaps: Release Full needs deterministic Node setup, narrower token permissions, action SHA pins, resolved-SBOM or explicit declared-only labeling, and Adobe tracker label/exit-code hardening.
4. Make core local stores safer: error bodies should include request IDs, typed-error paths need structured logs, SQLite stores need explicit `PRAGMA user_version`, destructive wipes need dry-run/backup confirmation, and persisted job results need caps.
5. Compete where editor workflows are moving: Descript, Submagic, AutoCut, Adobe, and LosslessCut show that creators value transcript editing, styled captions, long-form-to-short extraction, silence removal, fast rough cuts, and directly editable timeline outputs. OpenCut should prioritize timeline-native caption round trips and a Magic Clips macro that composes existing local primitives instead of adding more heavy models.

The repository already has a good planning shape: `ROADMAP.md` and `TODO.md` remain the active source of truth, while detailed research lives under `docs/archive/research/`. This file extends that research archive and incorporates Cycle 21 through Cycle 28 findings as RA-31 through RA-36 candidates, without rewriting the active queue.

## Evidence Reviewed

Local repository evidence:

| Evidence | Finding |
| --- | --- |
| `rtk git log -10` | Current `main` head is `07b70c0`; recent commits are docs/research and i18n migration work, not feature-code changes. |
| `git status --short --branch` | Worktree was clean except untracked `docs/archive/research/RESEARCH_FEATURE_PLAN_2026-06-04_CYCLE28.md`. |
| `CLAUDE.md`, `PROJECT_CONTEXT.md`, `ROADMAP.md`, `TODO.md`, `RESEARCH_REPORT.md` | Canonical planning source is `ROADMAP.md`/`TODO.md`; detailed research belongs under `docs/archive/research/`. |
| `py -3.12 scripts\sync_version.py --check` | Version references are synchronized at `v1.32.0`. |
| `py -3.12 -m opencut.tools.dump_route_manifest --check` | Route manifest is current: 1,523 routes across 107 blueprints. |
| `py -3.12 -m opencut.tools.dump_feature_readiness --check` | Feature readiness is current: 66 generated records and 90 route bindings. |
| `py -3.12 -m opencut.tools.dump_mcp_extended_tools --check` | MCP extended tools are current: 1,466 tools. |
| `py -3.12 -m opencut.tools.dump_model_cards --check` | Model cards are current: 55 cards across 7 categories. |
| `npm run build:verify` in `extension/com.opencut.panel` | CEP panel source/dist integrity check passes from the normal checkout path. |
| `npm run audit:check -- --json` in `extension/com.opencut.panel` | Node advisory gate passes with one allowed Vite advisory and zero unwaived advisories. |
| `py -3.12 -m pip_audit -r requirements-lock.txt --format json --progress-spinner off` | Fails on `idna==3.11`, affected by `GHSA-65pc-fj4g-8rjx` / `CVE-2026-45409`, patched at 3.15. |
| `py -3.12 -m opencut.tools.pip_audit_extras --json --extra all` | `requirements.txt` resolves clean, but `pyproject[all]` fails with five unwaived Torch/Transformers advisories plus two allowed waivers. |
| `py -3.12 scripts\sbom.py --format json --output dist\research-plan-sbom-check.json` | The repository SBOM generator emits declared runtime/optional/model-card components, not a resolved installed-environment inventory. |
| `README.md` | Badges reflect 1,523 routes, but prose and diagrams still contain stale route/module/blueprint counts and a missing GPU compose file command. |
| `.github/workflows/build.yml`, `.github/workflows/pr-fast.yml`, `.github/workflows/adobe-premierepro-versions.yml` | PR Fast pins Node 22; Release Full lacks matching setup-node, has workflow-level `contents: write`, and tracker issue creation depends on mutable labels and shell exit-code capture behavior. |
| `extension/com.opencut.uxp/manifest.json`, `extension/com.opencut.uxp/main.js`, `extension/com.opencut.uxp/bolt-webview/uxp.config.ts` | UXP migration has scaffolding, but live permission/schema/deprecation evidence remains incomplete before final F252 cutover. |
| `Dockerfile`, `docker-compose.yml`, `.dockerignore` | Docker support has dependency-surface, documentation, install-fail-open, and build-context hygiene drift. |
| `docs/archive/research/RESEARCH_FEATURE_PLAN_2026-06-04_CYCLE21.md` through `CYCLE28.md` | Adds RA-31 through RA-36 candidates around Adobe tracker exit codes, label contracts, dry-run labeling, lockfile audit coverage, SBOM fidelity, and UNC/HGFS-safe panel commands. |

External ecosystem evidence reviewed:

| Source | Why it matters |
| --- | --- |
| [Adobe Premiere Pro UXP manifest documentation](https://developer.adobe.com/premiere-pro/uxp/plugins/concepts/manifest/) | Premiere UXP plugins require manifest metadata, permissions, host compatibility, WebView settings, launch permissions, IPC declarations, and manifest schema behavior. |
| [Adobe UXP filesystem operations](https://developer.adobe.com/premiere-pro/uxp/resources/recipes/filesystem-operations/) | Adobe explicitly recommends accurate filesystem permissions and warns that `fullAccess` can make users uncomfortable or block installs. |
| [Adobe UXP external-process recipe](https://developer.adobe.com/premiere-pro/uxp/resources/recipes/external-process/) | External launches require declared schemes/extensions and explicit user consent context. |
| [Adobe UXP changelog](https://developer.adobe.com/premiere-pro/uxp/changelog/) and [UXP API changelog](https://developer.adobe.com/premiere-pro/uxp/uxp-api/changelog3-p) | Premiere Pro v25.6 made UXP official; v26.2 adds hybrid plugins; UXP v8 changes WebView, clipboard, alert, and CSS behavior. |
| [Adobe Premiere Pro AI feature announcement](https://news.adobe.com/news/2025/04/new-ai-innovation-in-industry) | Native Premiere now includes Generative Extend, Media Intelligence, and Caption Translation, raising the bar for third-party workflow assistants. |
| [Descript editor interface](https://help.descript.com/hc/en-us/articles/37585546799757-The-editor-interface) and [Descript AI tools overview](https://help.descript.com/hc/en-us/articles/27252457732237-AI-Tools-Overview) | Competitor workflow centers on transcript editing, timeline refinement, AI cleanup, filler-word removal, and long-form repurposing. |
| [Submagic API introduction](https://docs.submagic.co/introduction) | Submagic exposes styled AI captions, Magic Clips, templates, multilingual transcription, webhooks, and export/publish flows. |
| [AutoCut download/features page](https://www.autocut.com/en/download/) | AutoCut markets Premiere/DaVinci plugin features around silence removal, captions, auto zoom, B-roll, chapters, profanity filtering, and resize. |
| [LosslessCut GitHub repository](https://github.com/mifi/lossless-cut) | Open-source competitor shows durable demand for fast, local, FFmpeg-backed rough cuts, smart cut, stream editing, and lossless workflows. |
| [OpenTimelineIO documentation](https://opentimelineio.readthedocs.io/en/latest/index.html) and [OTIO adapters](https://opentimelineio.readthedocs.io/en/latest/tutorials/adapters.html) | OTIO is the relevant open interchange layer for editorial cut data and cross-NLE adapters. |
| [CycloneDX SBOM capability](https://cyclonedx.org/capabilities/sbom/) and [CycloneDX VDR capability](https://cyclonedx.org/capabilities/vdr/) | SBOMs are for component inventories; VDR/VEX-style artifacts communicate known vulnerability coverage and completeness. |
| [GitHub `GITHUB_TOKEN` docs](https://docs.github.com/actions/concepts/security/github_token), [automatic token permissions](https://docs.github.com/en/actions/security-guides/automatic-token-authentication), and [artifact attestations](https://docs.github.com/actions/security-for-github-actions/using-artifact-attestations/using-artifact-attestations-to-establish-provenance-for-builds) | Release workflows should grant least required token access and can improve provenance through attestations. |
| [GitHub advisory `GHSA-65pc-fj4g-8rjx`](https://github.com/advisories/GHSA-65pc-fj4g-8rjx) | Confirms `idna < 3.15` is affected by a 2026 denial-of-service advisory and 3.15 is patched. |

## Current Product Map

| Area | Current state | Important constraint |
| --- | --- | --- |
| Local backend | Python 3.11+ Flask server on loopback, large route catalog, generated route manifest, feature readiness manifest, model cards, and MCP tool export. | Local-first trust is a product feature. Avoid cloud dependency as the default path. |
| Premiere integration | CEP panel is the shipped panel; UXP panel and Bolt WebView scaffold exist; ExtendScript/host dispatch coverage remains part of the migration path. | F252 cannot be called complete until live Premiere UDT evidence is captured. |
| API and automation surface | 1,523 routes, 107 blueprints, 1,466 opt-in MCP tools, and many media/audio/caption/model workflows. | The next risk is operability and trust, not lack of routes. |
| AI/media features | Whisper/transcription, captions, translation, denoise, stem separation, beat/auto-edit, best-take, montage, variants, model cards, and readiness manifests. | Optional model dependency surfaces must stay auditable and explainable. |
| Storage | SQLite job store and journal with WAL pragmas; caches and generated artifacts under local app data paths. | Schema evolution is ad-hoc; destructive wipes and result growth need bounds. |
| Release packaging | PyInstaller/WPF/macOS/Docker/SBOM/release smoke workflows exist; route/model/card sync gates exist. | Release evidence still has Node determinism, token scope, SBOM fidelity, action pinning, Docker, and lockfile audit gaps. |
| Documentation/planning | `ROADMAP.md` is the detailed ledger; `TODO.md` is the compact active queue; `RESEARCH_REPORT.md` covers through Cycle 20; archive research notes cover later cycles. | Keep active queue edits separate from planning-only research unless implementation work actually lands. |

Core user personas inferred from the repo:

- Premiere power editor: wants time-saving in-panel controls, transcript/caption automation, timeline-safe outputs, and no accidental destructive operations.
- Creator/shorts editor: wants long-form-to-short extraction, styled captions, silence cleanup, auto zoom/B-roll, and exportable variants.
- Local privacy-sensitive user: wants no mandatory upload path, clear dependency/model provenance, and transparent local storage behavior.
- Maintainer/operator: wants generated truth checks, release smoke, SBOM/advisory evidence, and issue/roadmap automation that does not depend on a fragile workstation state.

## Feature Inventory

| Feature family | Current inventory | Strength | Gap to close |
| --- | --- | --- | --- |
| Captions and transcripts | Caption generation, QC, translation, burn/export surfaces, model cards, readiness records. | Strong local caption pipeline and regulatory orientation. | RA-09 timeline-native caption round trip with editable Premiere track parity. |
| Short-form and auto-edit | Auto-edit primitives, best-take, montage, variants, beat/audio helpers. | Many composable building blocks already exist. | RA-10 one-shot "give me N short candidates" macro with ranked ranges and rationale. |
| Audio AI | Denoise, SFX, TTS, effects, audio pro/i18n migration batches. | Active migration work and broad feature set. | Optional dependency advisories make the full convenience extra less trustworthy until RA-15 is resolved. |
| Model provenance | 55 generated model cards and generator check. | Better than a generic dependency list; clear domain grouping. | Tie model-card evidence into release trust pack and SBOM/VDR language. |
| API/MCP automation | 1,523 routes and 1,466 opt-in MCP tools. | Very large surface with generated consistency checks. | Error/request correlation and destructive action safety need to match route scale. |
| CEP panel | Existing production UI, Node advisory and dist verification checks. | Stable shipped path with working local checks. | RA-36 UNC/HGFS-safe package-script entry points for Windows shared-folder workflows. |
| UXP/WebView | Base UXP panel, Bolt WebView scaffold, host-action dispatch work. | Correct migration direction for a large vanilla JS panel. | RA-11/13/14/17/18/19/20 plus live UDT proof before F252 claims. |
| Release trust | Route/model/readiness sync, pip audit extras, SBOM script, release smoke, macOS notarization wiring. | Good foundation for public release claims. | RA-22/23/24/31/32/34/35 are needed before release evidence is robust. |
| Docker | Dockerfile and compose with GPU profile. | Useful supported launch path. | RA-25/26/27/29/30 align dependency install, docs, runtime posture, and build context hygiene. |
| GitHub issue/tracker automation | Issue seeder, label manifest, Adobe version tracker workflow. | Useful contributor and Adobe drift automation. | RA-31/32/33 make tracker/dry-run behavior reliable. |

## Competitive and Ecosystem Research

Adobe's UXP direction changes OpenCut's migration risk from optional modernization to release eligibility. Premiere UXP is official for v25.6 and beyond, and Premiere v26.2 adds Hybrid Plugin support for native `.uxpaddon` workloads. Adobe's manifest and recipes emphasize explicit manifest versioning, accurate permissions, WebView domain scoping, filesystem least privilege, external launch declarations, and user consent context. OpenCut's F252/F253 plan is directionally right, but the release should not claim a final cutover until manifest, permission, deprecation, WebView bridge, and UDT evidence are captured.

Adobe's native Premiere feature set also changed the competitive bar. Generative Extend, Media Intelligence, and Caption Translation mean OpenCut should not compete by promising generic generation inside Premiere. Its defensible lane is local-first automation around editorial glue: captions that round-trip into the real timeline, local transcript intelligence, deterministic media operations, routeable workflows, clear model provenance, and user-controllable exports.

Descript demonstrates that transcript-first editing is mainstream, not niche. Its editor centers the script, timeline, media layers, AI cleanup, filler-word removal, and repurposing. OpenCut should not build a separate cloud editor, but RA-09 should make transcript/caption edits survive a Premiere timeline round trip with schema-backed styling metadata.

Submagic and AutoCut show the creator market's expected bundle: styled captions, Magic Clips or auto-viral extraction, silence removal, auto zoom, B-roll suggestions, chaptering, profanity filtering, and platform resizing. OpenCut already owns many primitives, so RA-10 should be an orchestration feature rather than a new model wave: given a long source and target count, return ranked clip ranges, reasons, transcript snippets, caption style, and optional timeline/export instructions.

LosslessCut shows persistent demand for fast local rough cutting and stream operations. OpenCut can use this lesson without becoming a general-purpose standalone editor: expose fast rough-cut helpers and timeline/export paths that keep the editor in Premiere and preserve local media control.

OpenTimelineIO is the relevant interchange ecosystem for editorial cut information. For OpenCut, OTIO is more valuable as an adapter/test-fixture and interchange target than as a full replacement for Premiere-specific host actions. It can help RA-09 and RA-10 produce testable round-trip artifacts.

CycloneDX and GitHub supply-chain guidance affect how OpenCut should label release artifacts. A declared dependency SBOM is useful, but it should not be presented as a resolved installed-environment vulnerability inventory. CycloneDX's VDR/VEX concepts give OpenCut a better path: publish an SBOM for component inventory and a separate vulnerability/advisory report, or explicitly label current SBOMs as declared-only.

## Highest-Value New Features

| Feature | Why now | Implementation-ready shape | Acceptance evidence |
| --- | --- | --- | --- |
| RA-09 timeline-native caption round trip | Competitors win on captions users can edit on the actual timeline; OpenCut already has caption generation/export. | Add a caption track export/import schema that preserves OpenCut style metadata through SRT/ASS or marker fixtures and host round-trip paths. | Fixture test exports captions, simulates a Premiere edit, imports back, and preserves timing/style metadata. |
| RA-10 Magic Clips macro | Submagic/AutoCut/Descript-style long-form repurposing is a creator expectation; OpenCut already has scoring/editing primitives. | Add one orchestration route/skill that takes source media, target count/duration/platform, and returns ranked clip ranges with transcript snippets and rationale. | Synthetic multi-scene integration test returns stable ranked ranges without adding new heavy dependencies. |
| UXP/WebView trust-ready cutover package | Adobe's current UXP requirements make permission and manifest correctness part of install trust. | Treat F252 as a release package: manifest schema guard, filesystem downscope, WebView bridge split, launch/clipboard declarations, deprecation sentinel, confirmation replacement, and UDT capture. | Static tests plus recorded UDT/Premiere smoke for panel load, backend connection, OAuth/manual-copy fallback, file picker paths, and WebView bridge behavior. |
| Release trust pack | Public users need to know whether dependency, model, SBOM, and advisory claims are current. | Publish declared SBOM plus resolved SBOM or VDR/VEX report; include pip-audit targets, model-card count, route manifest, and release artifact provenance. | Release smoke fails if SBOM label and audit target coverage disagree. |
| Adobe compatibility tracker hardening | F251/F202/F252 depend on accurate Adobe drift signals. | Capture exit codes safely, seed required labels, allow dry-run label validation without `gh`, and track stable release-channel dist-tags. | Workflow tests and no-`gh` dry-run tests; weekly workflow opens/updates issues only when drift is real. |

## Existing Feature Improvements

| ID | Improvement | Evidence | Acceptance |
| --- | --- | --- | --- |
| RA-01 | Align Ruff target-version with Python floor. | `pyproject.toml` requires Python `>=3.11` but Ruff target remains `py39`. | Lint target reflects supported runtime and a test/metadata check prevents drift. |
| RA-02 | Reconcile `requirements.txt` and `pyproject.toml`. | `requirements.txt` uses a looser `faster-whisper>=1.0`; pyproject standard extra uses `>=1.1,<2`. | Dependency metadata check fails when floors diverge. |
| RA-03 | Log direct typed-error paths with structured context. | `safe_error()` logs, but direct `OpenCutError`/`error_response` paths can return without equivalent logging. | Focused tests prove typed errors log code/status/request context. |
| RA-04 | Include `request_id` in JSON error bodies. | Middleware already echoes `X-Request-ID`; error body omits it. | Route-smoke asserts body `request_id` equals response header. |
| RA-05 | Add `PRAGMA user_version` migrations. | Job store and journal use ad-hoc table/column creation. | Migration tests advance v0 DBs and reject too-new DBs with actionable errors. |
| RA-06 | Add dry-run, backup, and confirmation for destructive wipes. | `journal.clear_all()`, plugin uninstall, and cache clear can destroy user state. | Default returns counts only; confirmed delete writes backup where appropriate. |
| RA-07 | Cap persisted job `result_json`. | `save_job()` serializes full job results into SQLite rows. | Oversized result stores bounded summary/pointer while live status keeps full result in memory. |
| RA-08 | Add compaction and retention diagnostics. | Cleanup deletes rows but has no surfaced size/VACUUM/retention posture. | Diagnostic reports row counts and bytes; compaction test proves size reduction after bulk delete. |
| RA-09 | Add timeline-native caption round-trip parity. | Captions exist, but timeline-editable round-trip is the competitive gap. | Export/import fixture preserves timing and style metadata. |
| RA-10 | Add Magic Clips long-form-to-shorts macro. | Auto-edit primitives exist; no single ranked short-extraction workflow exists. | One route returns ranked candidate ranges and rationales on synthetic fixtures. |
| RA-11 | Tighten UXP filesystem permissions. | Base UXP manifest uses broad `fullAccess` while code also uses picker APIs. | Manifest and code prove picker/request-scoped access covers current workflows. |
| RA-12 | Add hybrid plugin package validator. | Adobe v26.2 Hybrid Plugins make native add-ons viable but strict. | Validator catches missing `.uxpaddon` layout/architecture before distribution. |
| RA-13 | Declare and harden external-launch permissions. | `shell.openExternal()` is used; Adobe requires declared schemes and consent context. | Manifest includes narrow schemes and tests cover approval/denial/manual-copy fallback. |
| RA-14 | Split WebView permissions between dev and release. | Generated config enables `localAndRemote`; release posture needs narrower domains. | Dev/release manifests are generated separately and statically checked. |
| RA-15 | Decide optional `[all]` advisory strategy. | `pyproject[all]` audit currently fails on Torch/Transformers advisories. | Either upgrade/split/document the extra and make the audit result deliberate. |
| RA-16 | Track Adobe release-channel dist-tags. | Existing tracker follows `latest`/`beta`; release-channel tags can move independently. | Tracker records configured stable release tags and reports meaningful drift. |
| RA-17 | Add UXP manifest schema guard. | Live UXP manifest omits explicit supported `manifestVersion`; generated config uses version 6 while Premiere docs name version 5 support. | Static guard enforces documented manifest policy with UDT evidence. |
| RA-18 | Add UXP API deprecation sentinel. | Current UXP API changelog deprecates older clipboard/video APIs; cutover can regress silently. | Static sentinel blocks deprecated clipboard/video API usage in UXP/WebView path. |
| RA-19 | Declare UXP clipboard permission and centralize copy fallback. | UXP code writes to clipboard; manifest/behavior needs explicit coverage. | Copy path handles permission denial and tests cover fallback. |
| RA-20 | Replace or gate raw UXP confirmation behavior. | `window.confirm` is used while UXP alert APIs have beta/flag caveats. | In-panel confirmation component or documented feature flag with tests. |
| RA-21 | Prove or retract Python 3.13 classifier. | Classifier advertises 3.13 support without matching CI evidence. | CI matrix covers 3.13 or classifier is removed until proven. |
| RA-22 | Pin Release Full Node runtime. | PR Fast uses `actions/setup-node@v4` with Node 22; Release Full lacks equivalent setup. | Release Full Node version is explicit and panel gates run under that version. |
| RA-23 | Pin GitHub Actions to full-length SHAs. | Workflows use mutable action tags. | Static workflow guard blocks unpinned third-party actions. |
| RA-24 | Scope Release Full token permissions by job. | Release Full has workflow-level `contents: write`. | Default permissions are read-only; only release upload job receives write. |
| RA-25 | Align Docker dependency surface. | Dockerfile installs packages outside canonical tracked surfaces. | Docker install consumes canonical requirements/extras or static guard proves parity. |
| RA-26 | Align Docker runtime docs and posture. | Docker comments/runtime paths and HTTP/WebSocket posture drift. | README/compose/Dockerfile agree on non-root home and exposed services. |
| RA-27 | Fix Docker GPU compose command drift. | README references a missing `docker-compose.gpu.yml`; compose has GPU profile instead. | README command matches tracked compose files/profiles. |
| RA-28 | Add README non-badge count gate. | README prose/diagram retain stale 1,334/980/360/73 style counts. | Generated-count test fails on stale exact count claims outside generated badges. |
| RA-29 | Make Docker dependency install fail closed. | Shell-form specifiers and `|| echo` can mask broken installs. | Dependency installs use quoted/canonical specs and fail on errors. |
| RA-30 | Align Docker build-context secret/log hygiene. | `.dockerignore` omits some `.gitignore` secret/log patterns before `COPY . /app`. | Static guard ensures `.env`, `.env.*`, and `*.log` are ignored by Docker context. |
| RA-31 | Harden Adobe tracker exit-code capture. | Workflow redirects `--check --json` then writes `$?` after command under `continue-on-error`. | Probe explicitly captures rc and output; downstream notification uses deterministic output. |
| RA-32 | Seed Adobe tracker issue labels. | Workflow searches/creates issues with `f251`, `uxp`, `tracking`; live/default labels do not define them. | Labels manifest and seeder include tracker labels; workflow can find or create issues. |
| RA-33 | Allow label dry-runs without `gh`. | `apply_labels()` checks for `gh` before respecting `dry_run`. | `--labels --dry-run` works on machines without `gh`; real apply still requires `gh`. |
| RA-34 | Restore lockfile advisory coverage and refresh vulnerable lock pins. | `requirements-lock.txt` pins `idna==3.11`; pip-audit reports `CVE-2026-45409`. | Release smoke/audit includes lockfile and lock pin is updated to a patched version. |
| RA-35 | Publish resolved SBOM or label current artifact declared-only. | `scripts/sbom.py` builds a declared-dependency SBOM; generated SBOM omitted vulnerable lockfile `idna`. | Release artifact name/metadata accurately states declared-only or includes resolved installed packages plus vulnerability evidence. |
| RA-36 | Make CEP panel Node command entry points UNC/HGFS-safe. | Cycle 28 found `npm run` resolves scripts from `C:\Windows` on UNC/HGFS paths, while direct `node scripts/*.mjs` succeeds. | Documented local commands work from UNC/HGFS or route through a validated wrapper, with tests covering the command contract. |

## Reliability/Security/Privacy/Data Safety

Security and privacy priorities:

- RA-34 is immediate because a tracked lockfile contains a current 2026 vulnerability in `idna`. Even if production resolution differs, the lockfile should not carry a known vulnerable pin without a waiver and audit target.
- RA-15 is immediate because `pyproject[all]` is advertised as a convenience surface and currently fails with five unwaived advisories. The fix is a policy decision: keep all, split build-lane extras, upgrade, or document known exposure with explicit waivers.
- RA-35 should prevent false confidence. A declared-dependency SBOM is useful, but it should not be named or described like a resolved install inventory.
- RA-23 and RA-24 bring release workflows closer to least-privilege supply-chain practice.
- RA-11, RA-13, RA-14, RA-17, RA-18, RA-19, and RA-20 are the UXP trust bundle. Adobe's current docs make broad filesystem permissions, WebView domain breadth, external launches, clipboard APIs, and beta dialogs visible install/runtime risks.
- RA-06 should be treated as data-safety work, not polish. Local-first software has to make destructive maintenance predictable, reversible where possible, and explicit.
- RA-07 and RA-08 reduce local availability risk from unbounded SQLite growth.
- RA-29 and RA-30 prevent Docker from silently building partially provisioned or locally contaminated images.

Recommended near-term security gate:

1. Update `requirements-lock.txt` past `idna==3.11`, add lockfile auditing to release smoke, and assert the lockfile target is not skipped.
2. Decide the `[all]` extra posture and make `pip_audit_extras` fail only on unwaived, policy-relevant findings.
3. Rename or replace SBOM artifacts so component inventory and vulnerability evidence are separate and accurate.
4. Add release workflow static checks for action pinning and token permission boundaries.

## UX/Accessibility/Trust

OpenCut's UX trust issue is not only visual. It is whether users can understand what will happen, why permission prompts appear, what data is stored locally, and whether generated media remains editable in Premiere.

Priorities:

- Finish E15 i18n migration batches because a 1,500-route product and large panel need consistent localized strings before the UXP cutover is judged by real users.
- Use UXP permissions as trust copy: picker-scoped filesystem access and narrow launch schemes are easier to explain than broad `fullAccess`.
- Replace raw confirmation flows with panel-native confirmations that can show row counts, paths, and backup locations.
- Add `request_id` to error bodies so bug reports include an identifier without asking users to inspect headers.
- Make README generated counts and Docker commands match live truth. Stale public docs undermine trust in generated badges and release smoke.
- Treat RA-09 as an accessibility and creator workflow feature: captions are not only an export artifact; they need editable timeline control and style preservation.
- Treat RA-10 as an operator workflow feature: ranked short candidates should show rationale and transcript evidence, not just output files.

## Architecture/Maintainability

The architecture is strong because it already has generated manifests, route/readiness/model-card checks, and a local-first boundary. The maintainability risk is that each major surface has a different kind of drift:

- API scale drift: route and MCP manifests are guarded; keep extending that generated-truth model to README prose and issue seeds.
- Dependency drift: Python runtime, requirements, pyproject extras, lockfile, Docker, and SBOM targets need one shared policy.
- UXP migration drift: base manifest, generated Bolt config, UDT docs, and actual panel behavior need one validation path before F252 closes.
- SQLite drift: ad-hoc migrations work for small changes but become fragile across long-lived local installs.
- Release workflow drift: PR Fast and Release Full differ in Node setup and permission posture.
- Windows path drift: Cycle 28 shows that direct Node scripts can pass while documented `npm run` commands fail on UNC/HGFS paths.

Recommended maintainability shape:

- Add small static guards rather than broad rewrites: README count guard, workflow action pin guard, Docker context guard, Docker install guard, lockfile audit guard, UXP deprecation sentinel, manifest schema guard.
- Keep route/model/readiness generators authoritative and avoid hand-maintained exact counts outside generated sections.
- Add a shared dependency-surface manifest or policy doc that names each audit target: runtime requirements, pyproject standard, pyproject all, lockfile, Docker install, panel package, SBOM declared inventory, and resolved SBOM/VDR.
- For SQLite, implement versioned migrations once and reuse the pattern in both job store and journal.

## Prioritized Roadmap

- [ ] **P0 - RA-34 Restore lockfile advisory coverage and refresh vulnerable lock pins** - Update `idna` past the patched version, add `requirements-lock.txt` to release/audit coverage, and prove the lockfile audit fails on future vulnerable pins. Verify: `py -3.12 -m pip_audit -r requirements-lock.txt --format json --progress-spinner off`, focused release-smoke/audit tests, `git diff --check`.
- [ ] **P0 - RA-15 Resolve optional `[all]` advisory policy** - Decide whether `[all]` remains a convenience extra, splits into build-lane extras, upgrades Torch/Transformers, or documents waivers. Verify: `py -3.12 -m opencut.tools.pip_audit_extras --json --extra all` returns only policy-accepted findings.
- [ ] **P0 - F252 plus RA-11/13/14/17/18/19/20 UXP cutover readiness bundle** - Complete manifest version policy, filesystem downscope, WebView dev/release split, launch/clipboard permissions, deprecation sentinel, confirmation replacement, and live UDT evidence. Verify: static UXP tests plus captured Premiere/UDT panel smoke.
- [ ] **P1 - RA-35 Publish resolved SBOM or label current artifact declared-only** - Keep declared SBOM if useful, but make release artifact naming and vulnerability evidence accurate. Verify: generated SBOM includes expected resolved packages or artifact metadata explicitly says declared-only.
- [ ] **P1 - RA-22/23/24 Harden Release Full trust gates** - Pin Node, pin workflow actions to full SHAs, and scope token permissions by job. Verify: workflow static tests and a Release Full dry run where possible.
- [ ] **P1 - RA-31/32 Harden Adobe tracker notification contract** - Capture tracker exit codes deterministically and seed labels used by workflow issue search/create. Verify: workflow unit/static tests and label dry-run.
- [ ] **P1 - RA-04/03 Add request ID and structured typed-error logging** - Make errors easier to correlate from user reports and logs. Verify: route-smoke error test and typed-error log test.
- [ ] **P1 - RA-05/07/08 Add SQLite migration, result cap, and maintenance diagnostics** - Give local stores versioning, bounded rows, compaction, and surfaced health. Verify: migration, oversized-result, and compaction diagnostic tests.
- [ ] **P1 - RA-06 Guard destructive wipes with dry-run/backup/confirmation** - Protect journal, plugin uninstall, and cache clear operations. Verify: delete paths return counts first and write backups where applicable.
- [ ] **P2 - RA-25/26/27/29/30 Docker parity and hygiene bundle** - Align Docker dependency installs, docs, GPU command, runtime posture, fail-closed installs, and `.dockerignore`. Verify: focused Docker guard tests and real Docker build evidence when available.
- [ ] **P2 - RA-28 README non-badge generated-count gate** - Remove stale exact prose/diagram counts or generate them from source-of-truth data. Verify: README count drift test and badge sync check.
- [ ] **P2 - RA-36 Make CEP panel Node command entry points UNC/HGFS-safe** - Ensure documented local panel checks work from Windows shared-folder paths or route through a validated wrapper. Verify: documented commands from UNC/HGFS and focused command-contract tests.
- [ ] **P2 - RA-09 Timeline-native caption round trip** - Preserve OpenCut caption timing/style metadata through Premiere timeline edits. Verify: round-trip fixture tests.
- [ ] **P2 - RA-16 Track Adobe release-channel dist-tags** - Extend F251 to stable release-channel tags beyond `latest`/`beta`. Verify: generated Adobe version snapshot and tracker tests.
- [ ] **P2 - RA-21 Prove or retract Python 3.13 classifier** - Add CI evidence or remove the classifier. Verify: CI matrix or metadata test.
- [ ] **P3 - RA-10 Magic Clips macro** - Compose existing local auto-edit primitives into a ranked long-form-to-shorts route. Verify: synthetic multi-scene integration test.
- [ ] **P3 - RA-33 Let issue-label dry-runs run without `gh`** - Make local no-credential validation work. Verify: `--labels --dry-run` test without `gh`.
- [ ] **P3 - RA-01/02 Metadata alignment cleanup** - Align Ruff target and requirements/pyproject floors. Verify: metadata drift tests.

## Quick Wins

- RA-01: change Ruff target to match the Python support floor and add a simple metadata assertion.
- RA-02: synchronize `faster-whisper` floor between `requirements.txt` and `pyproject.toml`.
- RA-28: remove stale README exact route/module/blueprint counts or move them behind generated checks.
- RA-31: capture Adobe tracker shell exit code before any command can overwrite it.
- RA-32: add `f251`, `uxp`, and `tracking` labels to the label manifest or workflow contract.
- RA-33: move `_gh_available()` checks behind the `dry_run` branch in `apply_labels()`.
- RA-36: document and test a direct-node or PowerShell wrapper path for panel checks from UNC/HGFS workspaces.
- RA-27: update README GPU Docker command to the tracked compose profile command.

## Larger Bets

- F252 UXP/WebView final cutover: largest migration risk because it needs code, manifests, docs, static checks, and live Premiere UDT evidence.
- RA-09 timeline-native captions: high user value but requires careful schema design and round-trip fixtures.
- RA-10 Magic Clips macro: product-visible and competitor-relevant, but should be built by composing existing modules with transparent scoring.
- RA-35 resolved release SBOM/VDR: important trust work that may need new tooling such as `cyclonedx-py`, `syft`, or an explicit resolved virtualenv build step.
- RA-05/07/08 SQLite maintenance bundle: not flashy, but important for long-lived local installs.
- Docker parity bundle: requires careful validation across developer machines, CI, and optional GPU paths.

## Explicit Non-Goals

- Do not add a new broad model/dependency wave before RA-15, RA-34, and RA-35 are resolved.
- Do not weaken local-first defaults or make cloud upload mandatory for core caption, transcript, or editing workflows.
- Do not remove the shipped CEP path until UXP live evidence proves the migrated path covers required host actions.
- Do not claim Apple notarization, UXP install acceptance, Premiere UDT behavior, or release-channel tracker behavior without captured live evidence.
- Do not describe declared dependency SBOMs as resolved installed-environment vulnerability inventories.
- Do not rewrite `ROADMAP.md` or `TODO.md` from this planning-only report; promote items there only when implementation work is selected.
- Do not replace Premiere as the primary editing surface. OpenCut's advantage is workflow automation around the editor, not becoming a standalone NLE.

## Open Questions

- Which policy should govern `pyproject[all]`: keep as a convenience install, split into narrower extras, or require separate build-lane environments for GPU/model-heavy packages?
- Should the release SBOM become a resolved installed-environment SBOM, or should the current declared-dependency SBOM stay with explicit declared-only labeling plus a separate VDR/VEX artifact?
- What exact Premiere/UDT version should be the minimum proof point for F252, given Adobe's manifest v5 docs and generated config references to v6?
- Should UXP release WebView permissions allow only loopback/backend domains, or should any remote domain remain in a dev-only manifest?
- Which Apple credentials and GitHub environments will be used for the first live F202 notarization acceptance run?
- Should Docker remain a first-class supported path for GPU features if optional model packages keep driving advisory churn?
- What is the preferred output shape for RA-10: route-only, panel action, MCP tool, or all three after one shared orchestration core?
- Should RA-31 through RA-36 be appended to the active `TODO.md` queue immediately, or wait until the current E15/F202/F252/RA-01..RA-30 order advances?
