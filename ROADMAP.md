# Roadmap — OpenCut

Actionable work only. Historical and completed roadmap material is archived in CHANGELOG.md; blocked work is kept in Roadmap_Blocked.md.

## Research-Driven Additions

Added 2026-08-10, extended 2026-08-11 and 2026-08-20 from the research passes recorded in `RESEARCH.md`.
IDs continue the existing F-number scheme (highest prior allocation before 2026-08-11: F318; before
2026-08-20: F334).

F359-F365 came from a different source: an audit run on 2026-08-20 over the F335-F358 drain, checking
each closed item against its own acceptance text rather than against the research. Highest prior
allocation before that audit: F358. F359 needed a maintainer's voice on a public issue and moved to
Roadmap_Blocked.md; the rest landed.

### P1

### P2

### P3

## Audit Findings — 2026-08-20 (deep audit pass, v1.50.0 at 5588bd4a)

Full-repo engineering/UX/visual audit. Baseline this pass: `python -m pytest tests/ -q` = 11,354 passed,
45 skipped, 0 failures; `ruff check opencut/` clean; panel vitest 225 passed. IDs continue from F365.
Scanner sweeps dismissed as false positives after tracing: all 10 gitleaks hits (dummy fixture tokens in
tests), bandit B307 eval (AST-whitelisted expression sandbox, `ast.Attribute` forbidden), all B608 SQL
hits (parameterised or literal WHERE fragments), B603/B607 subprocess (list-form argv throughout). The
one open GitHub issue (#5) is already tracked as F359 in Roadmap_Blocked.md.

### P1

### P2

- [ ] P2 — F368 — UXP panel light theme is systematically broken at the base layer
  Category: visual
  Where: `extension/com.opencut.uxp/style.css` (zero `html.theme-light` rules in 5,253 lines; 21 winning dark-only declarations), `extension/com.opencut.uxp/uxp-command-center.css:1000-1008, 1092-1108, 134, 1337/2155, 1690, 1735`, `extension/com.opencut.uxp/uxp-command-center-layout.css:91`
  Problem: In light theme the UXP panel's primary CTA keeps the dark accent (`.oc-btn-primary` re-hardcodes `#86a8f7`/`#10141b` at uxp-command-center.css:1000-1008, defeating the tokenized rule at :92; no `theme-light` counterpart exists in any UXP stylesheet — verified by grep). Seven near-white inks are invisible on white: `.oc-result-copy strong` (:1668), `.oc-result-list-item-title` (:1757, verified `#f7f9fc`), `.oc-deliv-title` (:1912), `.oc-result-insight-value` (:3755), `#seqInfoGrid .oc-info-val` (:3946), `.oc-deliv-btn:hover` (:1823), `.oc-chip:hover` (:1872). The light status-line overrides at :1410-1424 are dead code — the ID-scoped dark rules at :1095-1108 (specificity 1,2,0) beat `html.theme-light …` (0,3,1), so Settings status lines keep dark tints in light theme. Toast `[data-state]` gradients (style.css:4216-4226) beat the tokenized `.oc-toast` and stay dark-green/brown/red-black on light. The focus-visible block at style.css:4824-4826 discards the `var(--border-focus)` token used by the earlier correct rule at :4606 and hardcodes the dark-theme blue.
  Evidence: Cascade-resolved sweep over all UXP stylesheets (only winning declarations reported); the three highest-impact claims re-verified directly in source this pass (`sed`/`grep` on :1000-1008, :1757, :1095-1108, and `grep -c theme-light style.css` = 0). Note: the rendered Playwright goldens pass because they pin the current (broken) light renderings — fixing this requires regenerating light-theme goldens, and the existing 1% threshold discipline from CLAUDE.md applies.
  Fix: Route every listed declaration through the existing `--cc-*`/`--border-focus` token system (the light values already exist: accent `#466fd3`, `--cc-shadow-float`); add the missing `html.theme-light .oc-btn-primary` pair or better, drop the re-hardcode at :1000-1008 entirely so :92's tokens win; de-ID the `#tab-settings .oc-status-line` rules (or raise the light overrides' specificity) so :1410-1424 stops being dead; tokenize the toast gradients and the :4824 focus block. Then re-capture the light-theme goldens deliberately.
  Acceptance: Grep for the listed literals in UXP CSS returns only token-definition lines; the light-theme rendered suite shows a `#466fd3`-family primary button, readable result-list titles, and light toasts; the dead-rule pair at :1410-1424 either applies or is removed.
  Confidence: Verified
  Effort: L

- [ ] P2 — F369 — CEP light theme gaps rooted in a mislabeled dark block
  Category: visual
  Where: `extension/com.opencut.panel/client/style.css:17537-17915` (structural cause), plus winning hits at :17765-17766, :18282-18302, :18594, :18599, :18733-18740, :18814; `extension/com.opencut.panel/client/command-center.css:574, 639, 669, 1190, 1553, 1597`; `extension/com.opencut.panel/client/main.js:12213, 12222`
  Problem: An unprefixed dark-theme block (~30 rules, 49 color declarations) begins at style.css:17537 *inside* the section labeled `LIGHT THEME (html.theme-light)` (banner at :16988), so both themes receive dark styling and only accidental earlier-specificity light rules neutralize most of it. The two that escape: `.quick-action-icon`/`.workspace-stage-card-icon` ink `#bcd4ff` on white (~1.6:1, used 3x in index.html) and its `rgba(119,167,255,0.12)` background. Also winning in light theme: `.oc-feature-gated::after` badges (`#ffbd63` on white ~1.9:1, applied by feature-state.js), the progress track (`rgba(255,255,255,0.075)` — invisible on light) and fill (dark-accent gradient `#77a7ff→#4fd6a4`), wizard checkbox-row hover/focus (white-wash hover, 12%-alpha focus ring), dropdown/context-menu/recent-clips black shadows (command-center.css, `--cc-shadow-float` exists and is unused), `.oc-status-line` white wash, and the waveform canvas painting `rgba(0,0,0,0.3)` + fixed HSL bars regardless of theme (main.js:12213/12222).
  Evidence: Cascade-resolved sweep (winning declarations only); the structural mislabel at :17537 is the root cause the individual hits fall out of. Same golden-pinning caveat as F368.
  Fix: First move/prefix the :17537-17915 block correctly (it belongs above the light section or under explicit dark scoping) so the section banner stops lying; then tokenize the listed survivors (progress track/fill and feature-gated badges need real light values, shadows use `--cc-shadow-float`); have `drawWaveform` read its two colors from CSS custom properties via getComputedStyle like cep-theme.js already does for other canvas work.
  Acceptance: Light-theme CEP shows readable stage-card icons, visible progress track with the `#466fd3`-family fill, readable feature-gated badges, soft light shadows on dropdown/context menus, and a theme-aware waveform; the LIGHT THEME section contains only `html.theme-light` rules.
  Confidence: Verified
  Effort: L

- [ ] P2 — F370 — Studio workbench clips don't flip with their timeline in light theme
  Category: visual
  Where: `extension/com.opencut.panel/client/studio-workbench-v2.css` and the byte-identical `extension/com.opencut.uxp/studio-workbench-v2.css` — :841-852 (`.studio-clip`), :1586-1599 (`.studio-sequence-clip`), :1676-1677 (`.studio-result-thumb`), :1475/:1485 (`.studio-subject`), :1494 (`.studio-before-after`), :744-749 (`.studio-action` focus ring), :887 (dead `.studio-wave--slate`)
  Problem: `html.theme-light .studio-timeline` flips the track surface to `#e9eef5` (:799-800) and `.studio-sequence-grid` likewise (:1582), but the clips sitting on them keep dark-navy chips (`#172a3c` fills, `#8fb3ff`/`#9ebcff` ink, dark repeating gradients) — half-flipped components in both panels. `.studio-result-thumb` and `.studio-subject` blocks are dark-only; `.studio-before-after` is a hard `#fff` that blows out in dark theme. Separately, `.studio-action`'s focus indicator is `outline: none` replaced by a 3px ring of `--studio-accent-soft`, which is 10-12% alpha in both themes — effectively invisible focus for keyboard users on those controls. `.studio-wave--slate` (:887) matches nothing in HTML or JS.
  Evidence: Sweep resolved parent-flips-child-doesn't asymmetry per selector; media mocks deliberately excluded (caption-over-video previews at :1318-1324 etc. are intentionally dark). The file ships twice verbatim, so `tests/test_shared_panel_assets.py` treats it as shared — fix once, byte-copy applies to both.
  Fix: Add `html.theme-light` counterparts for `.studio-clip`/`.studio-sequence-clip`/`.studio-result-thumb`/`.studio-subject` using the existing light studio palette (the `#e9eef5` family already chosen at :799); give `.studio-before-after` a token or dark variant; raise `--studio-accent-soft` ring alpha to a visible level or use `var(--border-focus)`; delete the dead :887 rule. Keep the two copies byte-identical so the shared-asset gate passes.
  Acceptance: Light-theme workbench shows light clips on the light timeline in both panels; `.studio-action` focus ring is visible in both themes; `.studio-wave--slate` is gone; `tests/test_shared_panel_assets.py` still passes.
  Confidence: Verified
  Effort: M

- [ ] P2 — F371 — Shipped HTML fallback strings have drifted from the locales
  Category: ux
  Where: `extension/com.opencut.panel/client/index.html` (57 fallback strings no longer matching `client/locales/en.json`), `extension/com.opencut.uxp/index.html` (32 vs `locales/en.json`)
  Problem: The inline `data-i18n` fallback text is what users see on first paint and whenever i18n init fails, and it has drifted — some are full rewrites, not near-misses: index.html:152 shows "Studio Workspace" where `workspace.cut_kicker` = "Cut Pass" (verified); index.html:1112/1176/1202/1214 show "Whisper Model" where `forms.model` = "Model"; index.html:3920 "Refresh Availability" vs locale "Refresh availability"; index.html:146 "Server disconnected…" vs locale "Backend disconnected…"; index.html:3932 "Live Updates Bridge" vs `settings.ws_bridge` = "WebSocket Bridge". Related: four different strings describe the one update-check retry (`en.json:1392` "…Click Check again to retry.", `en.json:2168` "…Click Refresh to try again.", `uxp:144`, `uxp:1150`) and only a "Check Again" button exists (`en.json:1397`) — two of the four name a button that isn't there.
  Evidence: Sweep enumerated the drifts; the load-bearing example (`workspace.cut_kicker` = "Cut Pass" vs HTML "Studio Workspace" at index.html:152) re-verified directly this pass. The existing i18n gates (`scripts/i18n_lint.py`, `tests/test_i18n_hardcoded_migration.py`) check key usage and hardcoded-string migration but nothing compares inline fallbacks to locale values, which is exactly the gap.
  Fix: Sync every drifted fallback to its locale value (the locale is the source of truth — several drifts are stale pre-rename text), collapse the four retry strings onto the real button name, then extend `scripts/i18n_lint.py` (or a new check in the same gate) to assert `data-i18n` fallback text equals the en.json value so the drift cannot reopen.
  Acceptance: The new lint check passes and fails when a fallback is edited without the locale; first-paint text matches post-i18n text on both panels.
  Confidence: Verified
  Effort: M

- [ ] P2 — F372 — Internal identifiers leak into user-facing strings
  Category: ux
  Where: `extension/com.opencut.uxp/locales/en.json:387-388` ("Regenerate the F260 dashboard artifact"), six strings naming `ocAddNativeCaptionTrack` (uxp:914, 1001, 1255, 1314, 1316, 1394), `en.json:622`/`uxp:1615` (exposes `/video/shorts-pipeline`), `en.json:1503`/`uxp:358` and `en.json:1536`/`uxp:380` (leak `confirm_name`/`confirm_token` API params), `uxp:772` ("Check result JSON."), `en.json:1484`/`uxp:345` (worker-isolation design-doc prose), bare dead-ends `en.json:651`+`uxp:393` ("HTTP {status}"), nine bare "Error: {error}" keys (en.json:261, 1109, 1888, 1974, 2047; uxp:40, 1591, 1597, 1602), five "Unknown error" strings (en.json:237, 1331, 2166, 2818; uxp:1141)
  Problem: Users are told to regenerate an internal F-numbered build artifact, validate files "for the CEP ocAddNativeCaptionTrack bridge", or check a JSON payload they cannot see. These read as debugger notes, give no action a user can take, and leak internal route/parameter names into the UI. The bare "Error: {error}"/"HTTP {status}"/"Unknown error" family gives no recovery path where the panel already owns good recovery copy (`cleanUiMessage` and `ERROR_CODE_ACTIONS` in main.js:4390-4470 demonstrate the house style).
  Evidence: Every key spot-checkable by exact reference; F260 strings and confirm_token leak re-verified verbatim this pass via json load.
  Fix: Rewrite each to state what happened and the user's next step in the existing calm style (e.g. migration dashboard: "Migration data isn't available in this build. Reinstall or update OpenCut."); route generic failures through `cleanUiMessage`-equivalent phrasing; keep internal detail in logs, not labels. Do not touch the two panels' log lines — only user-visible strings.
  Acceptance: Grep for `F260`, `ocAddNativeCaptionTrack`, `confirm_token`, `shorts-pipeline` in both en.json files returns zero user-visible hits; the rewritten strings pass the human-voice rules; es.json gets the same keys re-translated.
  Confidence: Verified
  Effort: M

### P3

- [ ] P2 — F377 — The CEP panel lint gate fails at HEAD
  Category: maintainability
  Where: `extension/com.opencut.panel/package.json` `lint` script (`eslint --max-warnings 24 …`); the 25 warnings are all `no-unused-vars` in `extension/com.opencut.panel/client/main.js`
  Problem: `npm run lint` exits 1 on a clean checkout because main.js carries 25 unused-variable warnings against a ceiling of 24. The gate has been red long enough that nobody reads it, so it cannot catch a real regression. Not caught by the 2026-08-20 audit, which ran ruff, pytest and vitest but not the panel lint script.
  Evidence: measured this pass on clean HEAD (stashed working tree): `npm run lint` -> exit 1, `25 problems (0 errors, 25 warnings)`, `ESLint found too many warnings (maximum: 24)`. Identical count with F367's changes applied, so F367 did not cause it. Sample offenders: main.js:17953 `_wsAutoConnected`, :17954 `_origOnHealth`, :18714/:18717 `evt`, :18727 `res`.
  Fix: Delete the genuinely unused bindings (most are assigned-never-read leftovers) rather than raising the ceiling; where a parameter must stay for signature reasons, prefix it `_` to match the config's ignore pattern. Then lower `--max-warnings` to the new count so the gate has teeth again.
  Acceptance: `npm run lint` exits 0 on a clean tree, and `--max-warnings` equals the remaining warning count.
  Confidence: Verified
  Effort: S

- [ ] P3 — F373 — One concept, many names: terminology drift across both panels
  Category: ux
  Where: exact keys per cluster — filler ops (`en.json:465, 510, 712, 934, 959`; `uxp:1043`), Auto Shorts/Magic Clips/Shorts Pipeline (`en.json:1216, 1822, 2154`; `uxp:1517, 1627-1628`), backend vs server (`uxp:46, 1146`; `index.html:146`), engine section names (`en.json:1401`; index.html:3915; `uxp:259`; uxp/index.html:1894), Clear Index vs Clear Library inside one confirm flow (`en.json:1639, 1641, 1649, 1656`), Burn-In vs Burn-in (`en.json:243` vs `600, 735, 1788, 2801`), Whisper model label casing CEP vs UXP (`en.json:309-319` vs `uxp:869-877`), RoFormer spellings (`uxp:1896-1897` vs uxp/index.html:644-645 vs `en.json:121, 125`), Real-ESRGAN vs RealESRGAN within `uxp:1500-1502`, Gist vs gist (`en.json:819-830, 2584-2585`), "Premiere connection required." odd one out (`en.json:2110` vs 196/1978/2111), min-clip-length label vs its own segment tooltip (`en.json:257-258`), faster-whisper install flow failing as "Whisper" (`uxp:1835-1864`)
  Problem: The same object or action carries two to four names, sometimes inside a single dialog (Clear Index button → "Clear footage search library?" title → "Clear Library" confirm → "Footage index cleared" toast). Terminology drift is the strongest "several tools glued together" signal a user gets.
  Evidence: Sweep with exact keys; each hit is one json lookup to confirm.
  Fix: Pick one term per cluster (suggested: "Remove Filler Words", "Magic Clips" as the product name with "Auto Shorts" dropped, "backend", "Engine Routing", "footage index", "Burn-In Captions", upstream spellings "Mel-Band RoFormer"/"BS-RoFormer"/"Real-ESRGAN", "Gist" capitalized as proper noun) and sweep every listed key plus matching es.json entries in one change.
  Acceptance: Each cluster resolves to a single term across en.json (both panels), the HTML fallbacks, and es.json; the Clear Index flow uses one noun end to end.
  Confidence: Likely
  Effort: M

- [ ] P3 — F374 — Copy mechanics: typos, mixed ellipses, split-sentence i18n
  Category: ux
  Where: `en.json:2523` + `uxp:1527` ("Pointilism" → "Pointillism"), `uxp/locales/es.json:15` ("Pestanas" → "Pestañas"), 26 ASCII "..." strings in CEP en.json where the house style is "…" (collisions listed: 1552, 1633, 2843 vs 831, 1380, 1601; UXP mirrors the inverse with "…" leaking at uxp:135, 465), `en.json:2355-2356` ("->" where `en.json:961` uses "→"), `en.json:1329` trailing space in `progress.step_prefix`, `en.json:2034` the locale's only curly apostrophe, `uxp:1071-1072` FCC sentence split across two concatenated keys (untranslatable, unpaired paren; CEP's `en.json:284` is one sentence), "-- Select a clip --" placeholders (`en.json:1059`, `uxp:1665`) and "GPU: --"/"Jobs: --" (`en.json:1771, 1774`) vs "N/A" (`en.json:409`), status phrasing "faster-whisper is not installed until requested." (`uxp:1835-1836, 1846`)
  Problem: Individually trivial, collectively the difference between finished and near-finished copy. The FCC split is also an i18n correctness bug — no translator can reorder around a hardcoded concatenation point.
  Evidence: Sweep with exact keys; Pointilism and es.json hits re-verified this pass.
  Fix: One copy pass over the listed keys: fix the two spellings, normalize each panel to its own ellipsis convention (or both to "…"), replace "->" with "→", strip the trailing space (the step prefix already gets its spacing from the join), pick straight apostrophes everywhere, merge the FCC string into one key with a {date} placeholder like CEP's, replace "--" placeholders with the select's proper placeholder pattern and "N/A", reword the install-status lines to "Not installed — installs on first use."
  Acceptance: All listed keys updated in en.json (both panels) + es.json; no "..."/"…" collision remains within either panel's locale; the FCC notice is a single translatable string in UXP.
  Confidence: Likely
  Effort: S

- [ ] P3 — F375 — The UXP-pending route map can rot silently
  Category: maintainability
  Where: `opencut/core/cep_uxp_parity.py` — `_route_gate_errors` (route-classification gate) and `ROUTE_UXP_PENDING`
  Problem: F362 added the pending classification with a floor ratchet, but nothing flags a `ROUTE_UXP_PENDING` entry whose route has since gained a UXP path (rows classify it "covered" and the map entry lingers) or has been deleted from the CEP panel entirely (`live_pending = deferred & cep` silently drops it from the count while the dead entry remains). Both rots leave the map overstating the backlog and make the floor mushy — the "porting a route lowers the count" contract only holds if someone remembers to edit the map.
  Evidence: Read of the shipped gate logic (this pass's own code, re-read with fresh intent): no error or manifest field surfaces `pending ∩ uxp` or `pending − cep`.
  Fix: In `_route_gate_errors`, emit gate errors (or at minimum a `stale_pending` manifest field the dashboard shows) for pending entries that are now covered and for entries no longer present in the CEP inventory, so the map must be pruned in the same change that ports or removes a route. Extend `TestF362PendingRoutesAreTrackedRatherThanExcluded` in `tests/test_uxp_migration_dashboard.py` with both cases.
  Acceptance: A pending entry for a route that is covered (or gone from CEP) fails the gate with a named route; the two new tests pass; regenerating the dashboard after a simulated port forces the map edit.
  Confidence: Verified
  Effort: S

- [ ] P3 — F376 — Third-party XML parsed with stdlib parsers at import boundaries
  Category: security
  Where: import paths that accept files commonly obtained from third parties: `opencut/core/caption_interchange.py:703` (`fromstring`), `opencut/core/screenplay_parser.py:79` (`ElementTree.parse`), `opencut/core/iso_ingest.py:387`, `opencut/core/multicam_xml.py:24`, `opencut/core/multi_pov.py:303` (`minidom.parseString`), `opencut/core/standards_validators.py:172`, `opencut/core/flight_path_map.py:132`; generation-only sites (podcast_rss, fcpxml_export, premiere.py) are lower concern
  Problem: `xml.etree`/`minidom` do not resolve external entities (XXE-safe by default) but remain exposed to entity-expansion memory amplification on crafted DTDs. Caption/timeline/screenplay files are exactly the kind of artifact users download from strangers, so a malicious file can spike the local server's memory. Loopback-only deployment caps the blast radius at self-DoS, hence P3, and no SECURITY.md claim is contradicted (checked — the file makes no XML-hardening claim).
  Evidence: bandit B314/B318 hits triaged one by one this pass; `defusedxml` absent from the dependency tree (grep of opencut/, requirements, pyproject = 0 hits). Not reproduced as an actual memory spike — hence the confidence level.
  Fix: Either adopt `defusedxml` for the seven import-boundary sites (actively maintained, PSF-license-compatible; keep stdlib for generation), or pre-reject DTDs cheaply (refuse input whose prolog contains `<!DOCTYPE` — caption/FCPXML/screenplay interchange formats never legitimately carry one) at the shared read helper before parsing. The DTD-reject route avoids a new dependency and matches the repo's stdlib-first posture.
  Acceptance: A fixture with a billion-laughs DTD is rejected with a clean error at every listed import path (parameterized test); legitimate fixture files still parse; whichever mechanism lands is noted in SECURITY.md's input-handling section.
  Confidence: Needs-repro
  Effort: M
