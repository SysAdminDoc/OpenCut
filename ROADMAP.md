# Roadmap

Single task tracker for known issues and planned work. Items below come from
the June 2026 engineering/product audit (verified findings, with file
locations); fixes already shipped are recorded in CHANGELOG.md and git
history, not here.

Blocked items (credential/license/hardware-gated) live in
[`Roadmap_Blocked.md`](Roadmap_Blocked.md).

## Research-Driven Additions

### P1

- [ ] P1 — Add rendered CEP/UXP state, keyboard, and responsive regression coverage
  Why: The panels' largest risk surface is presently checked through 11 Node utility tests and source invariants, not a rendered DOM across the themes and states users operate.
  Evidence: `extension/com.opencut.panel/tests/`; `extension/com.opencut.panel/vitest.config.mjs`; 18,495-line CEP `style.css`; 17,460-line CEP `main.js`; WCAG 2.2; Vitest 4.1 browser/trace support.
  Touches: panel dev dependencies/config, CEP and UXP browser fixtures/tests, accessibility assertions, screenshot baselines, `scripts/release_smoke.py`.
  Acceptance: deterministic headless tests render every top-level tab at 480px, preferred width, and 1200px in dark/light/auto themes; exercise keyboard tab/roving-tab/focus-trap/Escape behavior plus empty/loading/error/offline/permission/destructive-confirmation states; fail on uncaught errors, horizontal page overflow, inaccessible names, focus loss, or approved screenshot drift; run in release smoke without Premiere.
  Complexity: L

### P2

- [ ] P2 — Migrate remaining hardcoded CEP panel colors to theme tokens
  Why: The CEP `style.css` light theme is driven entirely by `html.theme-light`
  token overrides, but several component literals bypass the tokens and keep
  dark-theme values on a light background, hurting contrast. The status dots were
  fixed; the tinted-white text-on-translucent cases need rendered contrast checks
  before swapping (they sit on colored/alpha surfaces, not a flat background).
  Evidence: `extension/com.opencut.panel/client/style.css` text hex at ~5119,
  5152, 5163, 6242, 6313, 6341, 6467, 6965, 11405, 11417; `accent-color: #d4b17a`
  at 9901/10615 (rest of the file uses `accent-color: var(--neon-cyan)`).
  Touches: `style.css` component rules, the `--neon-*`/`--success`/`--error`/
  `--text-*` token set, light-theme overrides.
  Acceptance: each flagged literal is either replaced with the correct semantic
  token or documented as an intentional on-color value verified to meet WCAG AA
  contrast in both themes; no component keeps a dark-only color on the light
  surface. Best landed with the rendered CEP/UXP coverage item so contrast is
  checked automatically.
  Complexity: M

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
