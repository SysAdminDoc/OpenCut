# Research — OpenCut

## Executive Summary
Verified: OpenCut v1.33.1 is a local-first Adobe Premiere Pro automation system, not a general browser editor: Python/Flask backend, CEP and UXP panels, generated route/readiness/model manifests, plugin loading, MCP, local installers, and FFmpeg-backed media workflows. The strongest current direction is trustable local automation that exposes readiness/provenance clearly before an editor runs a workflow. Highest-value opportunities: 1) move the CEP panel off unsupported Vite 5, 2) make README/product facts generated enough that badges, route counts, test counts, and caption-style claims cannot drift, 3) make `/ux/feature-index` readiness-aware so command/search surfaces do not expose dead or speculative routes as runnable, 4) resolve real caption fonts and CJK fallback for styled/burned captions, 5) finish the existing roadmap item for plugin trust/quarantine visibility in Settings, and 6) keep UXP WebView cutover gated on real Premiere UDT capture instead of static confidence.

## Product Map
- Core workflows: local Premiere panel automation for cuts, captions, audio cleanup, VFX, export presets, timeline write-back, plugin routes, model readiness, and MCP/API automation.
- User personas: Premiere editors who want local automation, social/video creators avoiding cloud usage caps, post/operations users who need predictable batch media tooling, and technical users writing plugins or scripts against REST/MCP.
- Platforms and distribution: Python 3.11+ on Windows/macOS/Linux, Adobe Premiere CEP compatibility, Premiere UXP migration path, WPF/installer packaging, Docker/local server, MIT package `opencut-ppro`.
- Key integrations and data flows: CEP/UXP panel -> Flask routes -> async job/history -> FFmpeg/model backends -> generated route/readiness/model-card artifacts; plugins load from `~/.opencut/plugins`; optional C2PA/content-credential work is local and signer-dependent.

## Competitive Landscape
- Adobe Premiere Pro / UXP / Content Credentials: strong native AI search, transcript/caption, UXP distribution, and provenance expectations. Learn: UXP and content authenticity are table-stakes for trust. Avoid: relying on CEP-only or static migration claims without live host evidence.
- FireCut, AutoPod, AutoCut, TimeBolt: paid Premiere automation succeeds by making repetitive editor actions fast and obvious. Learn: command/search surfaces must route to real, ready actions. Avoid: a huge action catalog that mixes shipped and planned routes without clear state.
- Descript, OpusClip, Submagic: cloud-first tools lead on transcript editing, clips, captions, templates, and collaboration. Learn: caption/style/font polish and social deliverables matter. Avoid: cloud-default collaboration that conflicts with OpenCut's local-first privacy value.
- LosslessCut and auto-editor: OSS competitors win by being local, focused, FFmpeg-backed, and reliable. Learn: simple, deterministic operations beat broad but unverified feature promises. Avoid: exposing speculative AI routes as if installed.
- Kdenlive and Shotcut: full NLEs show the breadth of local video editing, effects, and timeline UX. Learn: clear status, export robustness, and media handling. Avoid: trying to become a full NLE inside Premiere.
- OpenTimelineIO, Remotion, Editly: adjacent tools prove the value of interchange, declarative video, and generated outputs. Learn: schema/versioned metadata and reproducible exports are worth continuing. Avoid: license-incompatible or cloud-bound dependency expansion.
- OpenCut-app/OpenCut and pyvideotrans: public issue signal highlights API/headless demand, auto-import, CJK fonts, translation, and install reliability. Learn: multilingual/local automation is valued. Avoid: duplicating the browser editor's full product surface.

## Security, Privacy, and Reliability
- Verified risk: `extension/com.opencut.panel/package.json` still uses Vite `^5.4.21`; Vite's release page marks Vite 5 unsupported, and dev-server advisories keep this dependency class security-sensitive even if OpenCut uses local panel builds.
- Verified risk: `scripts/sync_badges.py --check` currently fails because README has `10300+` tests while the script expects `10800+`; README also advertises `19 styles` while `opencut.core.caption_styles.BUILTIN_STYLES` has 55.
- Verified risk: `opencut/core/command_palette.py` exposes 215 feature-index entries and 183 route targets absent from `opencut/_generated/route_manifest.json`; the C2PA action points to `/platform/c2pa` while live routes include `/provenance/c2pa` and `/video/c2pa/embed`.
- Verified risk: styled caption definitions carry `font_family`, but `_build_drawtext_filter()` emits `fontfile=''` and does not resolve a font path or family; CJK line breaking is tested, but CJK glyph rendering can still fail at FFmpeg drawtext time.
- Existing roadmap trust work remains valid: plugin lock/quarantine routes exist, but panel-visible plugin trust state is still the active user-facing gap.
- Missing guardrails: supported-line dependency gate for CEP panel tooling, generated README fact checks beyond badges, readiness-aware UX feature index tests, and caption font-resolution/CJK render tests.
- Recovery needs: generated manifests should distinguish implemented, dependency-gated, planned, and blocked actions so panels can disable or explain unsafe/unavailable actions instead of surfacing generic route failures.

## Architecture Assessment
- `opencut/core/command_palette.py` is a hand-curated catalog that has drifted from the generated route/readiness registry. Refactor toward a generated or enriched index that carries readiness and only treats live routes as runnable.
- README facts are partly generated but still partly hand-written. Extend the existing badge sync approach to route prose, test prose, and caption style counts instead of adding more manual edits.
- Caption rendering has good parsing/wrapping tests but not enough font/backend evidence. Add a font resolver around `caption_styles.py` and test generated filter strings plus at least one CJK smoke render where FFmpeg is available.
- CEP/UXP migration docs show strong static guardrails, but live WebView cutover still depends on Premiere UDT capture. Keep that as an open validation question until a real host run exists.
- Active working notes still contain stale CI/GitHub Actions references even though this repo's current policy and `.github` state are local-build only. That is a dev-experience reliability issue for future agents.
- Category coverage: security and upgrade strategy map to the Vite/plugin work; accessibility and i18n map to caption CJK/font rendering; observability and testing map to readiness/fact-drift guards; docs/distribution map to README and local-build cleanup; plugin ecosystem is the existing roadmap item; mobile and cloud multi-user are rejected as product misfits; offline resilience and UXP migration stay local-first and evidence-gated.

## Rejected Ideas
- Cloud-first multi-user collaboration from Descript/OpusClip: rejected because OpenCut's differentiator is local-first Premiere automation; keep collaboration optional/review-export oriented.
- Mobile app: rejected because the product is a desktop Premiere extension/server stack.
- Full NLE replacement: rejected because Premiere is the host; Kdenlive/Shotcut/Olive are useful references, not a target product shape.
- Implement every `checks.py` Tier 3 stub: rejected because broad model chasing would increase maintenance/security load without proving user value.
- GitHub Actions as the release/build answer: rejected because current repo policy is local builds and releases from this workstation; use local smoke gates instead.
- New caption-template expansion: rejected because backend has 55 styles; the gap is rendering fidelity, font fallback, and documentation drift.

## Sources
Repository and OSS:
- https://github.com/SysAdminDoc/OpenCut
- https://github.com/OpenCut-app/OpenCut
- https://github.com/mifi/lossless-cut
- https://github.com/WyattBlue/auto-editor
- https://github.com/KDE/kdenlive
- https://github.com/mltframework/shotcut
- https://github.com/AcademySoftwareFoundation/OpenTimelineIO
- https://github.com/remotion-dev/remotion
- https://github.com/mifi/editly
- https://github.com/jianchang512/pyvideotrans

Commercial and platform:
- https://www.adobe.com/products/premiere/ai-video-editor.html
- https://helpx.adobe.com/premiere-pro/using/media-intelligence-and-search-panel.html
- https://developer.adobe.com/premiere-pro/uxp/
- https://developer.adobe.com/premiere-pro/uxp/guides/distribution/
- https://helpx.adobe.com/creative-cloud/help/content-credentials.html
- https://firecut.ai/
- https://www.autopod.fm/
- https://www.timebolt.io/
- https://www.descript.com/pricing
- https://www.opus.pro/pricing
- https://www.submagic.co/

Standards, dependencies, and security:
- https://vite.dev/releases
- https://github.com/advisories/GHSA-p9ff-h696-f583
- https://github.com/advisories/GHSA-vg6x-rcgg-rjx6
- https://spec.c2pa.org/specifications/specifications/2.4/index.html
- https://opensource.contentauthenticity.org/docs/c2patool/
- https://github.com/advisories/GHSA-v2wj-q39q-566r

Community and discovery:
- https://github.com/wentianli/awesome-video-editing
- https://github.com/OpenCut-app/OpenCut/issues/817
- https://github.com/OpenCut-app/OpenCut/issues/827

## Open Questions
- Which signer/key workflow should be canonical for release-grade C2PA credentials? This requires a human-controlled certificate/key decision before implementation can be fully trusted.
- When can a real Premiere 26.x UDT capture be produced for `window.OpenCutUXPUdtHarness.run({ includeMutating: true })`? Static UXP tests are not a substitute for host evidence.
