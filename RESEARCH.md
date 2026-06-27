# Research - OpenCut

## Executive Summary
OpenCut is a local-first Python/Flask automation server with Adobe Premiere CEP/UXP panels for editing automation, captions, audio cleanup, media intelligence, delivery prep, and optional model-backed workflows. Its strongest current shape is not a standalone NLE clone; it is a broad, privacy-preserving Premiere automation layer with generated truth gates for routes, feature readiness, i18n, model cards, SBOMs, and release smoke. The highest-value direction is to restore trust and release hygiene before expanding feature breadth: update the bundled FFmpeg/provenance floor, regenerate stale model-card artifacts, restore the CEP Node advisory gate, replace CEP native dialogs with panel-local flows, reconcile public docs with the repo's local-build-only policy, and stop stale roadmap issue seeds from creating obsolete GitHub issues.

Top opportunities in priority order:
1. Verified - refresh the bundled FFmpeg binary and provenance floor; `ffmpeg\ffmpeg.exe -version` reports 8.0.1 while `scripts\verify_ffmpeg_provenance.py` requires release >= 8.1.1 or snapshot >= 2026-06-10 and currently fails.
2. Verified - restore generated model-card freshness; `py -3.12 -m opencut.tools.dump_model_cards --check` fails against `opencut/model_cards.py`.
3. Verified - restore the CEP Node advisory gate; `npm run audit:check -- --json` fails on `js-yaml` GHSA-h67p-54hq-rp68 and Vite GHSA-4w7w-66w2-5vf9/GHSA-v6wh-96g9-6wx3/GHSA-fx2h-pf6j-xcff.
4. Verified - replace remaining CEP native `confirm()`/`prompt()` flows and add a CEP guard mirroring `tests/test_uxp_confirmation_guard.py`.
5. Verified - reconcile active docs with the removed GitHub Actions/Dependabot path; `592ec577` removed workflows, but active docs still promise CI, workflows, and Dependabot monitoring.
6. Verified - guard `.github/issue-seeds.yml` against stale/shipped roadmap IDs; dry-run still emits obsolete F097-F116 issues even though active `ROADMAP.md` only carries current research additions.
7. Verified - keep generated route, feature-readiness, and locale gates as the product inventory; `dump_route_manifest --check`, `dump_feature_readiness --check`, `scripts/i18n_lint.py --check`, and `scripts/lint_locales.py --check` pass.

## Product Map
- Core workflows: Premiere panel control, REST/CLI/MCP automation, silence/filler/repeat-take cutting, transcription/caption translation/export, audio cleanup/stems/TTS/dubbing, footage search, delivery documents, and model-gated video/audio/image tools.
- User personas: Premiere editors who want local automation; privacy-sensitive creators; technical operators scripting routes; maintainers packaging a Windows-first installable tool.
- Platforms and distribution: Python 3.11+ server, Windows installer/spec files, Adobe CEP panel, Adobe UXP panel, optional Docker/Flatpak/AppImage metadata, and local HTTP/WebSocket/MCP ports.
- Key integrations and data flows: CEP/UXP panels call localhost Flask routes; generated manifests feed docs/panel truth; SQLite/job stores track work; optional model/provider integrations are gated by capability checks and model/license cards.

## Competitive Landscape
- Adobe Premiere Pro and UXP - Adobe is moving core editing toward native AI-assisted media search, captions, object masking, and UXP panels. Learn from the platform shift and keep UXP parity strong; avoid rebuilding host-native editing primitives when OpenCut can orchestrate them.
- OpenCut-app/OpenCut and browser CapCut alternatives - their issue queue highlights headless APIs, import/export portability, memory pressure, CJK/RTL captions, fonts, and offline expectations. Learn from user demand for portability and text/caption completeness; avoid pivoting this repo into a separate browser editor.
- Descript, CapCut, OpusClip, FireCut - commercial tools package silence removal, transcript editing, captions, B-roll, virality scoring, and social variants as polished one-click workflows. Learn from finished workflow UX; avoid cloud-first or metered-credit assumptions that conflict with OpenCut's local-first value.
- Kdenlive, Shotcut, Flowblade, OpenShot - OSS NLE competitors are judged on stability, export fidelity, keyframe retention, fonts, proxies, crash recovery, and platform packaging. Learn from their reliability pain points; avoid duplicating full timeline-editing surface area.
- LosslessCut, auto-editor, WhisperX, OpenTimelineIO, FFmpeg - adjacent tools win by being narrow, scriptable, and reliable. Learn from small composable commands, stable interchange, and binary provenance; avoid optional dependency bloat without generated availability/license truth.
- Topaz Video AI and Runway - users expect high-end restoration, segmentation, generation, and enhancement, but these are compute/license-heavy surfaces. Learn the expectation set; keep defaults license-clean and local, with restricted engines opt-in only.

## Security, Privacy, and Reliability
- Verified - bundled FFmpeg is below the repo's documented security floor: `ffmpeg\ffmpeg.exe` reports `8.0.1-essentials_build-www.gyan.dev`, while `docs/RELEASE_PROVENANCE.md:38` requires release `>= 8.1.1` or a June 2026+ git-master snapshot and `scripts\verify_ffmpeg_provenance.py` exits `RESULT: BELOW FLOOR`.
- Verified - model/license truth is stale: `opencut/tools/dump_model_cards.py` reports committed `docs/MODELS.md` and `opencut/_generated/model_cards.json` out of sync with `opencut/model_cards.py`.
- Verified - CEP Node advisory gate is failing: `extension/com.opencut.panel/scripts/check-advisories.mjs` only waives GHSA-4w7w-66w2-5vf9, while npm audit now reports `js-yaml` GHSA-h67p-54hq-rp68 plus additional Vite advisories GHSA-v6wh-96g9-6wx3 and GHSA-fx2h-pf6j-xcff.
- Verified - CEP still uses native dialogs for destructive or credential-like flows in `extension/com.opencut.panel/client/main.js:9632`, `15119`, `15399`, `16565`, `16635`, and `16667`. UXP already has a no-alert/confirm/prompt guard in `tests/test_uxp_confirmation_guard.py` and an inline "Confirm Clear" pattern in `extension/com.opencut.uxp/main.js:3578`.
- Verified - public active docs now overstate automation that no longer exists: `.github/workflows` and `.github/dependabot.yml` are absent after `592ec577`, while `SECURITY.md:47`, `CONTRIBUTING.md:11`, `CONTRIBUTING.md:95`, `CONTRIBUTING.md:108`, `docs/MACOS_NOTARIZATION.md:3`, `docs/WINDOWS_CODESIGNING.md:4`, `docs/WINDOWS_CODESIGNING.md:61`, and `docs/INSTALLER_POLICY.md` still reference CI, GitHub Actions, workflow builds, or Dependabot.
- Verified - issue seeding can publish stale/shipped work: `.github/issue-seeds.yml` contains F097-F116 entries sourced from `ROADMAP.md v4.3`, and `py -3.12 scripts\seed_github_issues.py --dry-run --once` emits shipped or obsolete items such as F098, F099, F100, F111, F112, and F115.
- Verified - route and feature truth gates are healthy today: `dump_route_manifest --check` passes with 1551 routes across 107 blueprints; `dump_feature_readiness --check` passes with 71 records and 97 route bindings.
- Verified - CEP and UXP localization are not active gaps: `scripts/i18n_lint.py --check` reports 2571 keys with 0 dead/missing keys; `scripts/lint_locales.py --check` reports matching English/Spanish UXP locale catalogs.
- Missing guardrails: CEP native-dialog guard; current-binary FFmpeg provenance check in the release path; advisory policy that can stay green while Vite 5 remains pinned; seed validation that refuses roadmap IDs absent from active `ROADMAP.md`.
- Recovery/rollback needs: if Vite cannot be upgraded because of the documented VMware HGFS build regression, advisory waivers must stay explicit, documented, and limited to dev-server-only risk; production CEP output must remain built and smoke-tested.

## Architecture Assessment
- Generated truth is a core boundary. Keep route, feature-readiness, model-card, SBOM, FFmpeg provenance, and advisory checks as generated contracts rather than hand-maintained docs.
- CEP and UXP have diverged on destructive-action UX. Port UXP's second-click inline confirmation/toast/status pattern into CEP instead of adding more native dialogs.
- Release documentation needs a single local-build path. Active docs should name local commands and release artifact responsibilities instead of removed GitHub Actions workflows or Dependabot.
- `.github/issue-seeds.yml` needs to be treated as generated-or-validated operational data, not a permanent backlog. The seeder should parse active `ROADMAP.md`, skip archived/shipped seeds, and fail before creating issues with missing roadmap IDs.
- Large CEP files remain candidates for future modularization, but the active reliability issue is behavior and tests, not a broad refactor.
- Testing gaps: no CEP equivalent of `tests/test_uxp_confirmation_guard.py`; current Node advisory policy does not encode the new Vite/js-yaml state; generated model-card drift and FFmpeg binary provenance both fail current checks; issue-seed tests assert old seed volume instead of active-roadmap alignment.
- Category coverage: security/reliability/distribution produce active additions; accessibility and i18n have passing local gates; observability/logging and offline/local-only are already present; plugin ecosystem remains documented rather than a top immediate risk; mobile and multi-user SaaS conflict with the local Premiere architecture; migration/upgrade risk is represented by UXP parity and Vite/FFmpeg provenance work.

## Rejected Ideas
- Full standalone browser editor - rejected; OpenCut-app/OpenCut already targets that market, while this repo's architecture is a Premiere automation server/panel.
- Mobile app or multi-user hosted SaaS - rejected; conflicts with local-first localhost, panel, and optional model-gated architecture.
- Duplicate i18n cleanup item - rejected; CEP and UXP locale checks pass today.
- Duplicate capability-manifest item - rejected; route and feature-readiness generators pass today and prior work already classifies implemented/gated/stubbed surfaces.
- Immediate Vite major upgrade as a standalone active item - rejected for this pass; `docs/NODE_ADVISORIES.md` and `Roadmap_Blocked.md` show the Vite 6+ path is blocked by HGFS validation, so the actionable item is to restore the current advisory gate with explicit policy.
- New broad AI-model integrations such as Depth Anything 3, SeedVR2, AutoShot, Parakeet/Canary, LatentSync, IC-Light, and multimodal search - rejected for this pass; recent history and generated manifests show these have either shipped, are dependency-gated, or are intentionally stubbed/blocked, and none should be re-added without a failing current check.
- GitHub Actions or Dependabot automation - rejected by repository policy and current repo state; local build/test/advisory gates are the supported path.
- Manual rewrite of legacy root markdown files - rejected; this pass is limited to active research deliverables, and stale legacy files should only be moved or removed by a dedicated hygiene item if they affect shipped docs.

## Sources
OSS competitors and issue signal:
- https://github.com/OpenCut-app/OpenCut/issues/827
- https://github.com/OpenCut-app/OpenCut/issues/812
- https://github.com/OpenCut-app/OpenCut/issues/817
- https://github.com/OpenCut-app/OpenCut/issues/826
- https://github.com/OpenShot/openshot-qt/issues/6063
- https://github.com/OpenShot/openshot-qt/issues/6056
- https://github.com/mltframework/shotcut/issues/1827
- https://github.com/mltframework/shotcut/issues/1696
- https://github.com/jliljebl/flowblade/issues/1237
- https://github.com/mifi/lossless-cut/issues/2943
- https://github.com/mifi/lossless-cut/issues/2851
- https://github.com/magic-peach/reframe/issues/1575
- https://github.com/magic-peach/reframe/issues/1501

Commercial and platform:
- https://www.adobe.com/products/premiere/ai-video-editor.html
- https://developer.adobe.com/premiere-pro/uxp/
- https://www.descript.com
- https://www.capcut.com/tools/ai-video-editor
- https://www.opus.pro
- https://firecut.ai
- https://www.topazlabs.com/topaz-video-ai

Adjacent tools, standards, and advisories:
- https://github.com/WyattBlue/auto-editor
- https://github.com/m-bain/whisperX
- https://github.com/AcademySoftwareFoundation/OpenTimelineIO
- https://ffmpeg.org/security.html
- https://www.w3.org/TR/WCAG22/
- https://github.com/advisories/GHSA-h67p-54hq-rp68
- https://github.com/advisories/GHSA-4w7w-66w2-5vf9
- https://github.com/advisories/GHSA-v6wh-96g9-6wx3
- https://github.com/advisories/GHSA-fx2h-pf6j-xcff

## Open Questions
None that block prioritization. The active decisions are implementation choices inside the roadmap items below.
