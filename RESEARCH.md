# Research - OpenCut

## Executive Summary

Verified: OpenCut v1.33.1 is no longer an early feature backlog; it is a broad local-first Premiere automation system with CEP and UXP panels, a Flask backend, generated manifests, model/license cards, release smoke gates, plugin routes, and local distribution docs. Recent commits already closed the previous highest-risk findings around bundled FFmpeg provenance, CEP native dialogs, stale model cards, CEP Node advisories, UXP quick actions, and local-release policy drift.

The next useful work should tighten trust and UXP migration parity instead of adding unrelated features. The highest-value additions are: add a static UXP DOM-sink safety gate; reclassify caption translation engines by license before promoting NLLB/Seamless as default commercial-safe local translation; generate UXP caption style choices from the shared 55-style backend catalog; promote C2PA from unsigned sidecars to signed embedded/export-verifiable credentials; and surface plugin trust, lock, and quarantine status in panel Settings.

Assumption: the product should remain local-first, Premiere-centered, and user-installable without relying on GitHub Actions or cloud services. That matches the repo README, SECURITY, CONTRIBUTING, and current release policy.

## Product Map

- Verified: the backend is Python 3.11+/Flask with route, MCP, model-card, feature-readiness, and release-smoke generators. `opencut/_generated/route_manifest.json` still records a small set of explicit stub routes, while most research items from earlier files have shipped.
- Verified: Premiere integration is split between CEP (`extension/com.opencut.panel`) and UXP (`extension/com.opencut.uxp`). UXP is the strategic surface because Adobe is moving Premiere extensibility to UXP/WebView, while CEP remains the compatibility panel.
- Verified: the release posture is local-first. README, SECURITY, Docker docs, and recent commits emphasize local builds, local backend binding, CSRF, path validation, optional cloud features, and no default public sidecar services.
- Verified: model-card governance is real. `scripts/dump_model_cards.py --check` reports 66 cards across eight categories, but translation code still directly advertises and downloads NLLB/Seamless paths that are not represented as distinct model-card/gating decisions.
- Verified: caption/style breadth exists in backend code. `opencut.core.caption_styles.BUILTIN_STYLES` currently reports 55 styles, while the UXP caption style select exposes only six fixed options.

## Competitive Landscape

- Adobe Premiere Pro is the baseline competitor, not just the host. Media Intelligence, AI search, transcription, captioning, text-based editing, and Content Credentials raise the minimum expected experience for any Premiere automation plugin.
- FireCut, AutoPod, AutoCut, and Timebolt compete as paid editing automation tools. Their wedge is editor-time savings, especially shorts, multicam, captions, silence removal, and assistant workflows.
- Descript, OpusClip, Submagic, and similar cloud-first tools compete on social editing speed, caption templates, repurposing, and collaboration. Their weakness is subscription/cloud friction; OpenCut's local-first position remains useful if the UXP panel feels equally polished.
- LosslessCut and auto-editor are strong OSS examples of focused local tooling. They win by being simple and reliable, not by having every AI feature.
- Kdenlive, Shotcut, Remotion, and OpenTimelineIO are adjacent open ecosystems. They matter for interchange, timeline semantics, and user expectations around transparent local media processing.
- The separate `OpenCut-app/OpenCut` project is a public open-source editor with active user issues. It is not a direct Premiere plugin, but its issue tracker is a useful signal for UX expectations: reliability, upload/media handling, and export robustness matter more than feature count.

## Security, Privacy, and Reliability

- Verified: the current repo has meaningful security gates: CSRF, path validation, advisory waivers, model cards, fuzz targets, route manifests, structured errors, and FFmpeg provenance verification. `ffmpeg.exe` verifies as Gyan 8.1.2, at/after the documented 8.1.1 security floor.
- Verified: CEP native dialogs are guarded by tests, and UXP imports an `escapeHtml` helper. Likely gap: UXP still uses many `innerHTML` assignments for backend/user-facing result markup. Most current uses escape interpolated values, but there is no dedicated static allowlist test for UXP HTML sinks comparable to the native-dialog guard.
- Verified: C2PA exists in two forms: a tested sidecar route and a `c2pa_embed` helper that writes metadata/sidecars. Likely gap: `opencut/core/c2pa_sidecar.py` explicitly says real C2PA verifiers will not accept the unsigned sidecar as a trust credential.
- Verified: plugin install/list/uninstall/quarantine routes exist, and install requires manifest validation. Likely gap: the panel does not clearly expose loaded, skipped, unsigned, locked, quarantined, or failed plugin state as a user trust surface.
- Verified: translation code uses SeamlessM4T first and falls back to NLLB. Likely gap: those engines are not separately surfaced in the generated model-card list or panel copy as license-sensitive optional engines.

## Architecture Assessment

Strengths:

- Verified: generated artifacts make drift visible. Route manifests, model cards, feature readiness, advisory docs, and release-smoke checks are the right direction for a repo with a very large route surface.
- Verified: the project keeps optional heavyweight engines behind extras and availability checks instead of making every install enormous.
- Verified: CEP and UXP coexistence lets the project migrate without breaking existing users.
- Verified: local-first distribution docs, Docker hardening, and installer work make OpenCut more credible than a script collection.

Risks:

- Verified: UXP parity is now a product risk. UXP is the future panel, but current UXP controls lag backend/CEP capability in visible places such as caption styles.
- Verified: trust features are split across backend routes and docs. Model cards, C2PA, plugin locks, quarantine, and advisories need panel-level affordances so users can see trust state before running actions.
- Likely: translation licensing can become a release blocker if NLLB/Seamless are marketed as normal defaults without model-card entries, license copy, and opt-in behavior.
- Likely: broad route count increases regression risk. The repo should keep favoring static gates, manifest checks, and focused parity tests over manual review.

## Rejected Ideas

- Rejected: "build paper edit" as a new item. Transcript/paper-edit routes and tests already exist, and this pass should not duplicate shipped work.
- Rejected: "add more backend caption templates" as the next item. The backend already reports 55 built-in styles; the research gap is UXP/CEP/backend parity and drift prevention.
- Rejected: "bump Vite major immediately." Vite 5 advisories are documented and waived, and the repo already records a VMware HGFS regression blocker for the major upgrade.
- Rejected: "make cloud AI the default competitive answer." Competitors already sell cloud workflows; OpenCut's differentiated position is local-first Premiere automation with explicit opt-ins.
- Rejected: "replace the whole plugin system." Routes, locks, validation, marketplace, and quarantine already exist. The more useful next step is exposing trust state and operations in the panel.

## Sources

Repository and OSS:

- https://github.com/SysAdminDoc/OpenCut
- https://github.com/OpenCut-app/OpenCut
- https://github.com/mifi/lossless-cut
- https://github.com/WyattBlue/auto-editor
- https://github.com/AcademySoftwareFoundation/OpenTimelineIO
- https://github.com/KDE/kdenlive
- https://github.com/mltframework/shotcut
- https://github.com/remotion-dev/remotion

Commercial and platform:

- https://www.adobe.com/products/premiere/ai-video-editor.html
- https://helpx.adobe.com/premiere-pro/using/media-intelligence-and-search-panel.html
- https://developer.adobe.com/premiere-pro/uxp/
- https://developer.adobe.com/premiere-pro/uxp/guides/distribution/
- https://firecut.ai/
- https://www.autopod.fm/
- https://www.autocut.com/
- https://www.timebolt.io/
- https://www.descript.com/pricing
- https://www.opus.pro/pricing
- https://www.submagic.co/

Standards, security, and distribution:

- https://vite.dev/releases
- https://ffmpeg.org/security.html
- https://github.com/advisories/GHSA-fx2h-pf6j-xcff
- https://github.com/advisories/GHSA-v6wh-96g9-6wx3
- https://github.com/facebookresearch/seamless_communication
- https://huggingface.co/facebook/nllb-200-distilled-600M
- https://spec.c2pa.org/specifications/specifications/2.4/index.html
- https://opensource.contentauthenticity.org/docs/c2patool/
- https://helpx.adobe.com/creative-cloud/help/content-credentials.html

Community signal:

- https://www.trustpilot.com/review/opus.pro
- https://www.trustpilot.com/review/submagic.co

## Open Questions

- Should NLLB and SeamlessM4T remain installable for personal/non-commercial workflows, or should the default translation path switch to a clearly commercial-safe local engine first?
- Which C2PA implementation should be canonical for release exports: `c2patool`, `c2pa-python`, or a thin adapter that supports both?
- Should UXP load caption styles from a backend endpoint at runtime, a generated JSON asset, or a shared checked-in style manifest?
- Should plugin trust state live in the Settings tab only, or should risky plugin actions also surface inline warnings in the command palette and job history?
- Which blocked UXP live-test items need real Premiere 26.x evidence before the WebView manifest can become the default entrypoint?
