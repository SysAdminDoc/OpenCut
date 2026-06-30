# Research — OpenCut

## Executive Summary
Verified: OpenCut v1.33.1 is a local-first Adobe Premiere Pro automation stack: Python/Flask on localhost, CEP plus UXP panels, generated route/readiness/model/MCP manifests, FFmpeg/model-backed media workflows, plugin trust controls, and local installers. Its strongest current shape is not "another video editor"; it is trustable local automation that keeps capability, privacy, provenance, and recovery state explicit before an editor runs a workflow. Highest-value opportunities: repair the current i18n and job-resume verification failures; stop leaking remote-render-node secrets; reconcile real social upload routes with stub publish routes and README claims; keep MCP/model generated docs in lockstep with manifests; retire WebSocket/FastAPI "coming soon" surfaces or wire them to the real bridge; and keep UXP/live Premiere migration evidence in the blocked queue until a host run exists.

## Product Map
- Core workflows: Premiere panel automation for cuts, captions, audio cleanup, VFX, Magic Clips/shorts, social/export presets, timeline write-back, plugin actions, model readiness, and REST/MCP automation.
- User personas: Premiere editors who want local repetitive-work automation; creators avoiding cloud usage caps; post/ops users who need batch media tooling and recoverable jobs; technical users scripting the REST/MCP/plugin surfaces.
- Platforms and distribution: Python 3.11+ server on Windows/macOS/Linux, CEP panel, UXP migration path, WPF Windows installer, Docker/local server, MIT `opencut-ppro` package metadata.
- Key integrations and data flows: CEP/UXP -> Flask routes -> async jobs/history -> FFmpeg/model backends -> generated route/readiness/model-card/MCP artifacts; plugins load from `~/.opencut/plugins`; optional social, cloud, telemetry, and C2PA paths are explicit opt-ins.

## Competitive Landscape
- Adobe Premiere Pro / UXP / Content Credentials: strong native AI search, transcripts, captions, UXP distribution, and provenance expectations. Learn: UXP and content authenticity are trust basics. Avoid: claiming migration completion without live Premiere UDT evidence.
- FireCut, AutoPod, TimeBolt, AutoCut: paid Premiere automation wins by making common edits obvious and reliable. Learn: command surfaces must lead to ready actions, not stubs. Avoid: mixing export-prep stubs with real upload flows under the same label.
- Descript, OpusClip, Submagic, Gling: cloud creator tools lead on transcript editing, caption polish, short-form handoff, templates, and publishing. Learn: social handoff and caption readiness must be honest and polished. Avoid: cloud-default collaboration that weakens OpenCut's local-first privacy value.
- LosslessCut and auto-editor: OSS tools succeed through focused local FFmpeg-backed behavior and deterministic CLI flows. Learn: local reliability beats broad unverified capability lists. Avoid: large catalogs with stale counts or dead roadmap anchors.
- Kdenlive and Shotcut: local NLEs show mature export/status/media handling. Learn: errors, export state, and media interchange need careful UX. Avoid: trying to become a full NLE inside Premiere.
- OpenTimelineIO, Remotion, Editly: adjacent tools prove the value of schema/versioned interchange and reproducible generated video. Learn: generated manifests and stable contracts are worth preserving. Avoid: hand-maintained docs where generated data already exists.
- OpenCut-app/OpenCut and pyvideotrans: public issue signal emphasizes API/headless workflows, auto-import, multilingual handling, and install reliability. Learn: local multilingual automation is a real demand. Avoid: duplicating a browser editor product shape.

## Security, Privacy, and Reliability
- Verified risk: `py -3.12 scripts/i18n_lint.py --check` fails with 51 consumed-but-missing CEP keys, including `palette.*` and `settings.plugin_*`; focused `tests/test_i18n_drift.py` also fails, so recent plugin/palette UI strings are not release-ready.
- Verified risk: `tests/test_job_resume.py::test_checkpointable_routes_are_marked_resumable` fails because the test searches for an exact decorator string, while `opencut/routes/video_specialty.py` correctly has `@async_job("shorts_pipeline", resumable=True, rate_limit_key="ai_gpu")`.
- Verified risk: `opencut/core/remote_process.py` persists remote-node `api_key` values in `~/.opencut/remote_nodes.json`, and `opencut/routes/remote_realtime_routes.py` returns `node.to_dict()` from register/list endpoints, exposing raw node secrets to any authorized local caller.
- Verified risk: remote render-node registration accepts arbitrary URL strings and immediately calls `<url>/health`; OWASP SSRF guidance recommends strict scheme/host validation, no embedded credentials, explicit private-network policy, and tests for resolver edge cases.
- Verified risk: `/social/upload` delegates to real OAuth upload logic in `opencut/core/social_post.py`, but `/publish/upload` uses `opencut/core/multi_publish.py::publish_to_platform()`, which is documented as a stub returning `pending_upload`. README and command palette language can overstate direct publishing depending on which surface a user enters.
- Verified drift: `docs/MCP_SERVER.md` and `opencut/routes/mcp_bridge_routes.py` still cite 39 curated tools and 1,325 extended tools while the changelog says 86 curated tools and `opencut/_generated/mcp_extended_tools.json` reports 1,450 tools.
- Verified drift: `opencut/model_cards.py`, `docs/MODELS.md`, and generated feature/model manifests still contain install hints such as `stub — roadmap H2.1` even though the live root roadmap no longer has those anchors.
- Verified risk: `opencut/core/fastapi_app.py` exposes a `/ws/jobs` WebSocket stub that replies "WebSocket job updates coming soon" while `opencut/core/ws_bridge.py` contains a real WebSocket bridge used by the app.
- Verified i18n quality gap: `py -3.12 scripts/lint_locales.py` returns success but warns on Spanish UXP missing diacritics, so locale quality can regress without breaking release checks.
- Missing guardrails: release smoke should cover the i18n drift and generated-doc count failures that currently escape narrower manifest checks.

## Architecture Assessment
- `opencut/core/remote_process.py` needs a public/private serialization boundary: persisted secrets may remain local, but route responses and logs should use redacted dictionaries and URL validation helpers.
- The publish stack has two competing meanings of "upload": real OAuth upload in `social_post.py` and dry-run/export-prep in `multi_publish.py`. Unify route naming, responses, and panel/command copy so users know when a platform API call actually happened.
- Generated data already exists for MCP tools, model cards, feature readiness, and route readiness; docs should read from those generators or have failing tests for stale counts and dead anchors.
- `tests/test_job_resume.py` should parse decorator metadata or route registration state instead of brittle exact source strings; the current source contains the correct resumable marker.
- FastAPI migration code should either bridge to the real job WebSocket mechanism or keep the stub unavailable from supported docs/tests until it is truthful.
- Category coverage: security maps to remote-node SSRF/secret redaction; accessibility and i18n map to CEP missing keys and UXP locale warnings; observability maps to job WebSockets and release smoke; testing maps to drift gates; docs and distribution map to generated MCP/model counts and blocked package-manager releases; plugin ecosystem is shipped enough to need locale completion; mobile and cloud multi-user are product misfits; migration remains evidence-gated in `Roadmap_Blocked.md`.

## Rejected Ideas
- Cloud-first collaboration from Descript/OpusClip/Submagic: rejected because OpenCut's differentiator is local-first Premiere automation; keep exports/review handoffs optional.
- Mobile app: rejected because the product is a desktop Premiere/server workflow.
- Full NLE replacement: rejected because Premiere is the host; Kdenlive/Shotcut are references for reliability, not target product shape.
- More speculative AI/model stubs: rejected until existing stub/model-card readiness docs stop pointing at dead roadmap anchors.
- GitHub Actions as the release/test answer: rejected because repo policy and current `.github` state are local-build only.
- Directly implementing PyPI/Homebrew/winget in this pass: rejected because those are already credential/signing-gated in `Roadmap_Blocked.md`.
- Moving UXP WebView cutover back to active work: rejected until live Premiere UDT evidence exists.

## Sources
Repository and OSS:
- https://github.com/SysAdminDoc/OpenCut
- https://github.com/WyattBlue/auto-editor
- https://github.com/mifi/lossless-cut
- https://github.com/AcademySoftwareFoundation/OpenTimelineIO
- https://github.com/KDE/kdenlive
- https://github.com/mltframework/shotcut
- https://github.com/remotion-dev/remotion
- https://github.com/mifi/editly
- https://github.com/OpenCut-app/OpenCut
- https://github.com/jianchang512/pyvideotrans
- https://github.com/wentianli/awesome-video-editing

Commercial and platform:
- https://www.adobe.com/products/premiere/ai-video-editor.html
- https://helpx.adobe.com/premiere-pro/using/media-intelligence-and-search-panel.html
- https://developer.adobe.com/premiere-pro/uxp/
- https://developer.adobe.com/premiere-pro/uxp/guides/distribution/
- https://helpx.adobe.com/creative-cloud/help/content-credentials.html
- https://firecut.ai/
- https://www.autopod.fm/
- https://www.timebolt.io/
- https://www.descript.com/
- https://www.opus.pro/
- https://www.submagic.co/

Standards, APIs, dependencies, and security:
- https://cheatsheetseries.owasp.org/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html
- https://developers.google.com/youtube/v3/guides/uploading_a_video
- https://developers.tiktok.com/doc/content-posting-api-reference-direct-post/
- https://developers.facebook.com/docs/instagram-platform/content-publishing/
- https://spec.c2pa.org/specifications/specifications/2.4/index.html
- https://opensource.contentauthenticity.org/docs/c2patool/
- https://www.w3.org/TR/webvtt1/
- https://www.w3.org/TR/WCAG22/

## Open Questions
- Which human-controlled signer/key workflow should be canonical for release-grade C2PA credentials?
- When can a real Premiere 26.x UDT capture be produced for `window.OpenCutUXPUdtHarness.run({ includeMutating: true })`?
