# Roadmap

Single task tracker for known issues and planned work. Items below come from
the June 2026 engineering/product audit (verified findings, with file
locations); fixes already shipped are recorded in CHANGELOG.md and git
history, not here.

Blocked items (credential/license/hardware-gated) live in
[`Roadmap_Blocked.md`](Roadmap_Blocked.md).

## Research-Driven Additions

### P0

- [ ] P0 — Patch the Click command-injection advisory in every install lane
  Why: The audited lock pins Click 8.3.1, which is affected by CVE-2026-7246; the fix is small, released, and should precede feature work.
  Evidence: `requirements-lock.txt`; `pyproject.toml`; live `pip-audit` finding `PYSEC-2026-2132`; NVD CVE-2026-7246; Click 8.3.3 changelog.
  Touches: `requirements-lock.txt`, `requirements.txt`, `pyproject.toml`, `docs/PYTHON_ADVISORIES.md`, dependency/audit tests.
  Acceptance: all declared and locked Click floors are >=8.3.3; `py -3.12 -m pip_audit -r requirements-lock.txt --format json --progress-spinner off` and the release advisory gate pass with no Click waiver.
  Complexity: S

- [ ] P0 — Close the unguarded SSRF and unbounded download in URL ingest
  Why: `ingest_url` fetches an attacker-supplied URL with only a scheme check — it never calls the project's own `validate_public_http_url`, never honors local-only mode, follows redirects, and streams the body with no size cap, so `POST /search/ingest` can reach `127.0.0.1`/`169.254.169.254` or fill the disk.
  Evidence: `opencut/core/url_ingest.py::ingest_url()` and `_fetch_direct()` (lines 135-231); the guard already exists at `opencut/core/url_safety.py::validate_public_http_url`; route `opencut/routes/search.py`; webhook path uses the guard, this one does not.
  Touches: `opencut/core/url_ingest.py`, `opencut/core/url_safety.py`, `opencut/routes/search.py`, network-allowed/local-only helper, SSRF and size-limit tests.
  Acceptance: URL ingest rejects non-public/loopback/link-local hosts before connecting and re-validates every redirect hop; respects local-only mode via `require_network_allowed`; enforces a configurable byte ceiling in the read loop; verifies downloaded bytes are media (ffprobe) before returning; regression tests cover private-IP, redirect-to-private, oversized, non-media, and local-only-blocked cases.
  Complexity: S

### P1

- [ ] P1 — Make feature readiness prove implementation as well as dependency presence
  Why: Import probes can currently report unfinished adapters as available after their dependency is installed, so generated readiness can direct users to deterministic `NotImplementedError` failures.
  Evidence: `opencut/registry.py::FeatureRecord.resolved_state()`; 52 `opencut/core/*.py` files with terminal `NotImplementedError`, including `asr_canary.py`, `asr_parakeet.py`, and `ezaudio_service.py`; FireCut/AutoPod focus on ready actions.
  Touches: `opencut/registry.py`, `opencut/core/feature_readiness.py`, `opencut/tools/dump_feature_readiness.py`, `opencut/_generated/feature_readiness.json`, readiness/model-card/MCP consumers, readiness tests.
  Acceptance: each feature has an explicit implementation state independent of dependency/hardware state; unfinished entrypoints remain `stub` when imports succeed; generation/release checks fail if an `available` feature's canonical valid-request smoke ends in `NotImplementedError`; panels and MCP expose the same state and reason.
  Complexity: M

- [ ] P1 — Unify bounded, transactional archive ingestion
  Why: Project restore and Lottie inspection accept unbounded expanded ZIP data, and project restore can leave a partial destination; the plugin installer already demonstrates most required safeguards.
  Evidence: `opencut/core/project_archive.py::restore_archive()`; `opencut/core/lottie_import.py::info()`; bounded extraction in `opencut/core/plugin_marketplace.py`; Kdenlive 26.04.1 malicious-project security fix.
  Touches: shared security/archive helper, `opencut/core/project_archive.py`, `opencut/core/lottie_import.py`, `opencut/core/plugin_marketplace.py`, archive and fuzz tests.
  Acceptance: every ZIP consumer enforces path/special-entry checks plus configurable member and expanded-byte limits before extraction/read; restores stage outside the destination and promote atomically; rejection or failure leaves no partial destination; regression tests cover traversal, absolute paths, oversized members, compression bombs, malformed manifests, and cleanup.
  Complexity: M

- [ ] P1 — Wire UXP batch rename and smart bins to their existing host actions
  Why: The UXP bridge already implements both operations, but the visible controls are always disabled and their handlers only claim CEP handoff, creating avoidable parity and trust drift.
  Evidence: `extension/com.opencut.uxp/main.js::batchRenameProjectItems()`, `createSmartBins()`, `runBatchRename()`, `runSmartBins()`, and `updateTimelineReadiness()`; Adobe Premiere UXP API documentation.
  Touches: `extension/com.opencut.uxp/main.js`, `extension/com.opencut.uxp/index.html`, UXP locales, UXP bridge/timeline contract tests.
  Acceptance: buttons enable only when the host action and project state are ready; users preview affected items/rules before mutation; rename records and exercises an inverse rename; smart-bin results report created items; unsupported hosts retain an honest disabled reason; mocked UXP tests cover ready, unsupported, cancel, success, partial failure, and undo paths.
  Complexity: M

- [ ] P1 — Add a semantic project-facts documentation gate
  Why: Existing generators pass while user-facing facts contradict source and policy, so counts alone do not keep the README, code comments, and blocked ledger truthful.
  Evidence: `README.md` says eight UXP tabs while `extension/com.opencut.uxp/index.html` has nine; `Roadmap_Blocked.md` still cites Vite 5.4.21 and removed `.github/workflows/build.yml`; the CSRF comment at `extension/com.opencut.uxp/main.js:1868` contradicts its `/health` implementation.
  Touches: `README.md`, `Roadmap_Blocked.md`, affected release/UXP docs and comments, `scripts/release_smoke.py::step_generated_docs`, generated-facts helpers, release-smoke and documentation tests.
  Acceptance: one checked project-facts source drives or validates tab names/count, package/tool versions, CSRF endpoint/header, release-policy/workflow references, and active/blocked state; each seeded contradiction makes the documentation/release gate fail; current contradictions are removed.
  Complexity: M

- [ ] P1 — Add rendered CEP/UXP state, keyboard, and responsive regression coverage
  Why: The panels' largest risk surface is presently checked through 11 Node utility tests and source invariants, not a rendered DOM across the themes and states users operate.
  Evidence: `extension/com.opencut.panel/tests/`; `extension/com.opencut.panel/vitest.config.mjs`; 18,495-line CEP `style.css`; 17,460-line CEP `main.js`; WCAG 2.2; Vitest 4.1 browser/trace support.
  Touches: panel dev dependencies/config, CEP and UXP browser fixtures/tests, accessibility assertions, screenshot baselines, `scripts/release_smoke.py`.
  Acceptance: deterministic headless tests render every top-level tab at 480px, preferred width, and 1200px in dark/light/auto themes; exercise keyboard tab/roving-tab/focus-trap/Escape behavior plus empty/loading/error/offline/permission/destructive-confirmation states; fail on uncaught errors, horizontal page overflow, inaccessible names, focus loss, or approved screenshot drift; run in release smoke without Premiere.
  Complexity: L

- [ ] P1 — Establish a model-weight supply-chain boundary
  Why: All `torch.load` sites rely on `weights_only=True` as the safety mechanism, but CVE-2026-24747 (GHSA-63cw-57p8-fm3p, CVSS 9.8, Torch <=2.9.1) bypasses exactly that path, and `pyproject.toml` floors `torch>=2.6` with no vulnerable-version ceiling or scanner gate on downloaded checkpoints.
  Evidence: `opencut/core/face_swap.py:105`, `opencut/core/model_quantization.py:373`, `opencut/core/video_ai.py:881`; `pyproject.toml` `depth`/`torch-stack` extras (`torch>=2.6`); GHSA-63cw-57p8-fm3p; picklescan CVE-2026-53875 (require >=1.0.3).
  Touches: `pyproject.toml`, `requirements-lock.txt`, a model-cache verification helper, model-manager/download paths, advisory docs, dependency/security tests.
  Acceptance: Torch floors sit above the confirmed CVE-2026-24747 fixed release across every extra; downloaded weights are scanned (picklescan >=1.0.3) or hash-pinned before load and untrusted files are rejected; a test asserts the floor and the scan gate; confirm the exact GHSA range first (see RESEARCH Open Questions).
  Complexity: S

- [ ] P1 — Route all outbound fetches through one connect-time SSRF guard
  Why: `validate_public_http_url` deliberately does not resolve DNS and its callers re-resolve and follow redirects, leaving a DNS-rebinding / redirect-into-private-range window; URL ingest, webhooks, and model downloads each make their own (or no) decision.
  Evidence: `opencut/core/url_safety.py:17` ("accept as-is"); `opencut/core/webhook_system.py::_send_payload` (lines 487-527); `opencut/core/url_ingest.py`; model-manager fetch path.
  Touches: shared network-fetch helper, `opencut/core/url_safety.py`, `webhook_system.py`, `url_ingest.py`, model download callers, SSRF/rebind/redirect tests.
  Acceptance: one helper resolves the host, rejects private/loopback/link-local IPs, re-validates each redirect hop's resolved IP, and enforces timeouts/byte limits; all three outbound fetch surfaces use it; tests cover rebinding (public->private on second resolve) and redirect-to-private.
  Complexity: M

### P2

- [ ] P2 — Consolidate caption XML export and validate IMSC 1.3 conformance
  Why: OpenCut already exports EBU-TT/TTML/IMSC through two implementations, but tests assert XML shape rather than conformance to the May 2026 IMSC 1.3 Recommendation and multilingual round trips.
  Evidence: `opencut/core/broadcast_cc.py`; `opencut/core/broadcast_caption.py`; `tests/test_delivery.py`; `tests/test_subtitle_pro.py`; W3C IMSC 1.3 and WebVTT specifications.
  Touches: canonical caption model/exporter, both broadcast modules and routes, caption compliance/preflight, delivery docs, conformance and multilingual/RTL/vertical-text fixtures.
  Acceptance: one canonical exporter supports explicit legacy and IMSC 1.3 profiles; generated documents pass a checked conformance corpus; import/export round trips preserve UTF-8 text, language, timing, regions, styles, and writing direction; existing EBU-TT/TTML callers remain compatible or receive a documented migration error.
  Complexity: L

- [ ] P2 — Decompose the CEP and UXP panel monoliths behind contract tests
  Why: Controllers/styles have grown far beyond the repository's own decomposition guidance, making parity, review, and safe UI changes harder; the rendered regression item must land first to reduce refactor risk.
  Evidence: `CONTRIBUTING.md`; `extension/com.opencut.panel/client/main.js` (17,460 lines); `style.css` (18,495 lines); `extension/com.opencut.uxp/main.js` (8,191 lines); UXP `style.css` (4,592 lines).
  Touches: CEP/UXP state, backend client, i18n, job, timeline, component, token, layout, and bootstrap modules; Vite build; parity/source/release tests.
  Acceptance: shared responsibility boundaries are extracted without changing public IDs, host-action names, route payloads, or visual baselines; entrypoints contain bootstrap/orchestration rather than feature implementations; state/API/i18n/job/timeline modules have focused tests; panel build, parity, browser, i18n, and release-smoke gates pass.
  Complexity: XL

- [ ] P2 — Harden two low-cost reliability defaults
  Why: `/health` returns the CSRF token to any Origin outside a two-value denylist, and `check_disk_space` fails open on probe error so callers proceed on a genuinely full local volume, turning a clean 507 into a half-written mid-render failure.
  Evidence: `opencut/routes/system.py:374-401`; `opencut/helpers.py:187-193`; `opencut/jobs.py` admits `OPENCUT_MAX_CONCURRENT_JOBS` (max 100) onto a fixed 10-thread pool.
  Touches: `opencut/routes/system.py`, `opencut/helpers.py`, `opencut/jobs.py`, corresponding tests.
  Acceptance: CSRF token is exposed only to same-origin/no-Origin requests (allowlist); disk preflight fails open only for un-probeable network drives and closed for local-path probe failures; effective job concurrency is clamped to the worker pool's `max_workers`; tests cover each case.
  Complexity: S

- [ ] P2 — Add a local semantic media-search index
  Why: Premiere 26 Media Intelligence made natural-language search over project media table stakes; OpenCut already indexes footage but has no embedding search, and a local CLIP/embedding index is achievable without a cloud dependency and reinforces the privacy story.
  Evidence: `opencut/core/*` footage index modules; Premiere 26 Media Intelligence (Adobe blog 2026-01-20); RESEARCH.md Competitive Landscape.
  Touches: footage-index core, a new embedding/index module, search route(s), CEP/UXP search UI, model-readiness registry, tests.
  Acceptance: a user can query project media by natural-language description and get ranked local results; embeddings are computed and stored locally with no network egress; the feature reports honest readiness when its optional model dependency is absent; search route and ranking are tested.
  Complexity: L

- [ ] P2 — Add script/transcript-to-timeline assembly with alt-takes
  Why: Resolve 20 IntelliScript proved script->timeline assembly (matching transcribed takes, alt-takes on extra tracks) and it sits squarely in OpenCut's transcript-editing wheelhouse as a marquee local write-back feature.
  Evidence: Resolve 20 IntelliScript (Engadget 2026); existing transcript/alignment and timeline write-back in `opencut/core/` + journal; RESEARCH.md Competitive Landscape.
  Touches: transcript/alignment core, a take-matching module, timeline write-back + journal, review/preview UI, CEP/UXP handoff, tests.
  Acceptance: given a script and clip transcripts, OpenCut produces an ordered timeline plan mapping script lines to best-matching takes with alternates on separate tracks; the plan is previewable and reversible via the journal before write-back; matching and plan generation are tested against fixtures.
  Complexity: L

- [ ] P2 — Emit C2PA 2.4 AI-disclosure and durable soft-binding credentials on export
  Why: The repo already embeds signed C2PA provenance but should target C2PA 2.4 (Apr 2026), which added the machine-readable `c2pa.ai-disclosure` assertion and durable/soft-binding credentials that survive Premiere re-encode — honest local AI provenance no cloud competitor emphasizes.
  Evidence: existing C2PA provenance embedding (commit `338e3f67`); C2PA 2.4 spec (spec.c2pa.org/.../2.4); RESEARCH.md Architecture Assessment.
  Touches: C2PA provenance module, export pipeline, AI-action tracking (which steps were AI-assisted), assertion/soft-binding tests.
  Acceptance: exports embed a C2PA 2.4 manifest declaring AI-assisted actions via `c2pa.ai-disclosure` and a soft-binding assertion; a conformance test validates the manifest against the 2.4 profile; disclosure reflects which pipeline steps actually ran.
  Complexity: M

### P3

- [ ] P3 — Target the MCP 2026-07-28 revision for the MCP server
  Why: The 2026-07-28 MCP revision is the largest since launch (stateless core, elicitation replaced by Multi-Round-Trip Requests, Tasks extension for long-running work, cache metadata on resources); long renders/transcodes map directly onto the Tasks extension and few tools will be spec-current.
  Evidence: `opencut/mcp_server.py`, `opencut/mcp_extended_tools.py` (pinned `mcp>=1.26,<2`); MCP 2026-07-28 RC (blog.modelcontextprotocol.io); RESEARCH.md Architecture Assessment.
  Touches: `opencut/mcp_server.py`, `mcp_extended_tools.py`, MCP catalogue/generated docs, MCP dependency pin, MCP conformance tests.
  Acceptance: the server negotiates the 2026-07-28 protocol; long-running tools use the Tasks extension rather than blocking calls; resource/list reads emit cache metadata (`ttlMs`/`cacheScope`); user prompts use MRTR (`InputRequiredResult`) not deprecated elicitation; a conformance test covers the handshake and a Tasks round trip.
  Complexity: M

- [ ] P3 — Add a Parakeet-v3 + Whisper-turbo hybrid ASR router
  Why: Parakeet TDT v3 beats Whisper large-v3 on the Open ASR Leaderboard at ~3,000x realtime for 25 EU languages while whisper-large-v3-turbo covers 99 languages; auto-routing by detected language gives best local speed without sacrificing coverage, and the Parakeet adapter is currently a stub.
  Evidence: `opencut/core/asr_parakeet.py`, `opencut/core/asr_canary.py` (terminal `NotImplementedError`); faster-whisper engine already present; Open ASR Leaderboard 2026 (Northflank); RESEARCH.md Competitive Landscape.
  Touches: ASR engine registry, `asr_parakeet.py`, a language-detection/router module, caption pipeline, readiness registry, tests.
  Acceptance: caption generation detects language and routes supported languages to Parakeet and the rest to Whisper turbo, with an explicit override; each engine reports honest readiness when its model/dependency is absent; routing decisions and fallbacks are tested. Depends on the feature-readiness implementation-state item landing first.
  Complexity: M

- [ ] P3 — Localize the Python/CLI backend and add panel locales beyond en/es
  Why: The CEP and UXP panels ship English + Spanish only while the Python/CLI backend has no i18n framework (English-only error strings) and no RTL support anywhere, despite DE/FR/JA/PT labels already appearing untranslated in `en.json`.
  Evidence: `extension/com.opencut.panel/client/locales/`, `extension/com.opencut.uxp/locales/` (only `en`/`es`); no gettext/babel in `opencut/`; no `dir="rtl"` in panel HTML.
  Touches: a backend i18n layer (gettext/babel), CLI/route/core user-facing strings, new panel locale files, RTL layout handling, locale-lint/release tests.
  Acceptance: backend user-facing strings are translatable with an English fallback; at least one additional panel locale beyond en/es ships and passes the existing locale-lint gate; an RTL locale renders without layout breakage; a test guards locale-key parity across languages.
  Complexity: L
