# Research - OpenCut

## Executive Summary
OpenCut is a local-first Python/Flask automation server plus Adobe Premiere CEP/UXP panels for editing, captions, audio cleanup, media intelligence, delivery prep, and optional model-backed workflows. Its strongest current shape is not a standalone NLE clone; it is a broad, privacy-preserving Premiere automation layer with strong generated manifests, i18n gates, and release-smoke checks. The highest-value direction is to restore the failing trust checks before adding more model surface: regenerate stale model-card artifacts, make the CEP Node advisory gate green again while the Vite major upgrade remains hardware-blocked, and port UXP's inline-confirmation pattern back to CEP so destructive actions do not rely on native `confirm()`/`prompt()` dialogs.

Top opportunities in priority order:
1. Verified - restore generated model-card freshness; `py -3.12 -m opencut.tools.dump_model_cards --check` fails today.
2. Verified - restore the CEP Node advisory gate; `npm run audit:check -- --json` fails on `js-yaml` GHSA-h67p-54hq-rp68 and Vite GHSA-4w7w-66w2-5vf9/GHSA-v6wh-96g9-6wx3/GHSA-fx2h-pf6j-xcff.
3. Verified - replace remaining CEP native `confirm()`/`prompt()` flows and add a CEP guard mirroring `tests/test_uxp_confirmation_guard.py`.
4. Verified - keep using generated capability/readiness manifests instead of adding broad feature-inventory work; `dump_route_manifest --check` and `dump_feature_readiness --check` pass.
5. Likely - commercial short-form tools show value in polished assembly UX, but OpenCut already has many of those primitives; implementation should focus on release trust and panel parity first.
6. Likely - open-source editors' issue queues emphasize export fidelity, keyframe retention, font/localization handling, and crash recovery, which supports hardening existing workflows over starting a separate editor surface.

## Product Map
- Core workflows: Premiere panel control, REST/CLI/MCP automation, silence/filler/repeat-take cutting, transcription/caption translation/export, audio cleanup/stems/TTS/dubbing, footage search, delivery documents, model-gated video/audio/image tools.
- User personas: Premiere editors who want local automation; privacy-sensitive creators; technical users scripting routes; packagers maintaining a Windows-first installable tool.
- Platforms and distribution: Python 3.11+ server, Windows installer/spec files, Adobe CEP panel, Adobe UXP panel, optional Docker/Flatpak/AppImage metadata, local ports for HTTP/WebSocket/MCP.
- Key integrations and data flows: CEP/UXP panels call localhost Flask routes; generated manifests feed docs/panel truth; SQLite/job stores track work; optional model/provider integrations are gated by capability checks and model/license cards.

## Competitive Landscape
- Adobe Premiere Pro and UXP - Adobe is moving core editing toward native AI-assisted media search, captions, object masking, and UXP panels. Learn from the platform shift and keep UXP parity strong; avoid rebuilding host-native editing primitives when OpenCut can orchestrate them.
- OpenCut-app/OpenCut and browser CapCut alternatives - their issue queue highlights import/export projects, CJK captions/fonts, SRT import, offline PWA, and headless APIs. Learn from user demand for portability and text/caption completeness; avoid pivoting this repo into a separate browser editor.
- Descript, CapCut, OpusClip, FireCut - commercial tools package silence removal, transcript editing, captions, B-roll, virality scoring, and social variants as polished one-click workflows. Learn from the finished workflow UX; avoid cloud-first or metered-credit assumptions that conflict with OpenCut's local-first value.
- Kdenlive, Shotcut, Flowblade, OpenShot - OSS NLE competitors are judged on stability, export fidelity, keyframes, fonts, proxies, and platform packaging. Learn from their recovery and export pain points; avoid duplicating full timeline-editing surface area.
- auto-editor, WhisperX, OpenTimelineIO, FFmpeg - adjacent tools win by being narrow, scriptable, and reliable. Learn from small composable commands and stable interchange; avoid adding optional dependency bloat without generated availability/license truth.
- Topaz Video AI and Runway - users expect high-end restoration, segmentation, generation, and enhancement, but these are compute/license-heavy surfaces. Learn the expectation set; keep defaults license-clean and local, with restricted engines opt-in only.

## Security, Privacy, and Reliability
- Verified - model/license truth is currently stale: `opencut/tools/dump_model_cards.py` reports committed `docs/MODELS.md` and `opencut/_generated/model_cards.json` out of sync with `opencut/model_cards.py`.
- Verified - CEP Node advisory gate is failing: `extension/com.opencut.panel/scripts/check-advisories.mjs` only waives GHSA-4w7w-66w2-5vf9, while npm audit now reports `js-yaml` GHSA-h67p-54hq-rp68 plus additional Vite advisories GHSA-v6wh-96g9-6wx3 and GHSA-fx2h-pf6j-xcff. `docs/NODE_ADVISORIES.md` documents the Vite 5 HGFS constraint, so the fix must either update the lock/waiver policy or safely prove a supported Vite upgrade.
- Verified - CEP still uses native dialogs for destructive or credential-like flows in `extension/com.opencut.panel/client/main.js:9632`, `15119`, `15399`, `16565`, `16635`, and `16667`. UXP already has a no-alert/confirm/prompt guard in `tests/test_uxp_confirmation_guard.py` and an inline "Confirm Clear" pattern in `extension/com.opencut.uxp/main.js:3578`.
- Verified - route and feature truth gates are healthy today: `dump_route_manifest --check` passes with 1516 implemented, 29 dependency-gated, and 6 stub routes; `dump_feature_readiness --check` passes with 71 records and 97 route bindings.
- Verified - CEP and UXP localization are not active gaps: `scripts/i18n_lint.py --check` reports 2571 keys, 0 dead keys, 0 missing keys; `scripts/lint_locales.py --check` reports matching English/Spanish UXP locale catalogs.
- Missing guardrails: a CEP-specific native-dialog test; a documented/fixed handling path for new Node advisories while Vite 5 is pinned; a check or release habit that regenerates model cards immediately when `opencut/model_cards.py` changes.
- Recovery/rollback needs: if Vite cannot be upgraded because of the documented VMware HGFS build regression, advisory waivers must stay explicit, documented, and limited to dev-server-only risk; production CEP output must remain built and smoke-tested.

## Architecture Assessment
- Generated truth is a core boundary. Keep route, feature-readiness, model-card, SBOM, and advisory checks as generated contracts rather than hand-maintained docs.
- CEP and UXP have diverged on destructive-action UX. Port UXP's second-click inline confirmation/toast/status pattern into CEP instead of adding more native dialogs.
- The panel advisory gate is the right choke point, but it currently fails before release. Fixing the gate should include `tests/test_node_advisories.py` coverage, `docs/NODE_ADVISORIES.md`, and `package-lock.json` policy rather than a silent waiver.
- Large CEP files remain candidates for future modularization, but the active reliability issue is behavior and tests, not a broad refactor.
- Testing gaps: no CEP equivalent of `tests/test_uxp_confirmation_guard.py`; current Node audit policy does not encode the new Vite/js-yaml state; generated model-card drift must be fixed before release.
- Documentation gaps: `docs/MODELS.md` is stale because the generator check fails; `docs/NODE_ADVISORIES.md` is stale relative to the current npm audit result.
- Category coverage: security and reliability produce the active additions; accessibility and i18n have passing local gates; observability/logging and offline/local-only are already present; distribution/packaging and UXP WebView live validation are blocked in `Roadmap_Blocked.md`; plugin ecosystem, mobile, multi-user SaaS, and full standalone editor work do not align with this repo's Premiere-local architecture.

## Rejected Ideas
- Full standalone browser editor - rejected; OpenCut-app/OpenCut already targets that market, while this repo's architecture is a Premiere automation server/panel.
- Mobile app or multi-user hosted SaaS - rejected; conflicts with local-first localhost, panel, and optional model-gated architecture.
- Duplicate i18n cleanup item - rejected; CEP and UXP locale checks pass today.
- Duplicate capability-manifest item - rejected; route and feature-readiness generators pass today and prior work already classifies implemented/gated/stubbed surfaces.
- Immediate Vite major upgrade as a standalone active item - rejected for this pass; `docs/NODE_ADVISORIES.md` and `Roadmap_Blocked.md` show the Vite 6+ path is blocked by HGFS validation, so the actionable item is to restore the current advisory gate with explicit policy.
- New broad AI-model integrations such as Depth Anything 3, SeedVR2, AutoShot, Parakeet/Canary, LatentSync, IC-Light, and multimodal search - rejected for this pass; recent history and generated manifests show these have either shipped, are dependency-gated, or are intentionally stubbed/blocked, and none should be re-added without a failing current check.
- GitHub Actions or Dependabot automation - rejected by repository policy; local build/test/advisory gates are the supported path.

## Sources
OSS competitors and issue signal:
- https://github.com/OpenCut-app/OpenCut/issues/827
- https://github.com/OpenCut-app/OpenCut/issues/817
- https://github.com/OpenCut-app/OpenCut/issues/749
- https://github.com/OpenCut-app/OpenCut/issues/714
- https://github.com/mltframework/shotcut/issues/1827
- https://github.com/mltframework/shotcut/issues/1758
- https://github.com/mltframework/shotcut/issues/1696
- https://github.com/jliljebl/flowblade/issues/1237
- https://github.com/jliljebl/flowblade/issues/1234
- https://github.com/jliljebl/flowblade/issues/1203

Commercial and platform:
- https://www.adobe.com/products/premiere/ai-video-editor.html
- https://developer.adobe.com/premiere-pro/uxp/
- https://www.descript.com
- https://www.capcut.com/tools/ai-video-editor
- https://www.opus.pro
- https://www.topazlabs.com/topaz-video-ai
- https://runwayml.com
- https://firecut.ai

Adjacent tools and standards:
- https://github.com/WyattBlue/auto-editor
- https://github.com/m-bain/whisperX
- https://github.com/AcademySoftwareFoundation/OpenTimelineIO
- https://ffmpeg.org/security.html
- https://www.w3.org/TR/WCAG22/
- https://www.ecfr.gov/current/title-47/chapter-I/subchapter-C/part-79/subpart-B/section-79.103

Security advisories:
- https://github.com/advisories/GHSA-h67p-54hq-rp68
- https://github.com/advisories/GHSA-4w7w-66w2-5vf9
- https://github.com/advisories/GHSA-v6wh-96g9-6wx3
- https://github.com/advisories/GHSA-fx2h-pf6j-xcff

## Open Questions
None that block prioritization. The active decisions are implementation choices inside the roadmap items below.
