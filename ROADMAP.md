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

- [ ] P2 — F371 — Shipped HTML fallback strings have drifted from the locales
  Category: ux
  Where: `extension/com.opencut.panel/client/index.html` (57 fallback strings no longer matching `client/locales/en.json`), `extension/com.opencut.uxp/index.html` (32 vs `locales/en.json`)
  Problem: The inline `data-i18n` fallback text is what users see on first paint and whenever i18n init fails, and it has drifted — some are full rewrites, not near-misses: index.html:152 shows "Studio Workspace" where `workspace.cut_kicker` = "Cut Pass" (verified); index.html:1112/1176/1202/1214 show "Whisper Model" where `forms.model` = "Model"; index.html:3920 "Refresh Availability" vs locale "Refresh availability"; index.html:146 "Server disconnected…" vs locale "Backend disconnected…"; index.html:3932 "Live Updates Bridge" vs `settings.ws_bridge` = "WebSocket Bridge". Related: four different strings describe the one update-check retry (`en.json:1392` "…Click Check again to retry.", `en.json:2168` "…Click Refresh to try again.", `uxp:144`, `uxp:1150`) and only a "Check Again" button exists (`en.json:1397`) — two of the four name a button that isn't there.
  Evidence: Sweep enumerated the drifts; the load-bearing example (`workspace.cut_kicker` = "Cut Pass" vs HTML "Studio Workspace" at index.html:152) re-verified directly this pass. The existing i18n gates (`scripts/i18n_lint.py`, `tests/test_i18n_hardcoded_migration.py`) check key usage and hardcoded-string migration but nothing compares inline fallbacks to locale values, which is exactly the gap.
  Fix: Sync every drifted fallback to its locale value (the locale is the source of truth — several drifts are stale pre-rename text), collapse the four retry strings onto the real button name, then extend `scripts/i18n_lint.py` (or a new check in the same gate) to assert `data-i18n` fallback text equals the en.json value so the drift cannot reopen.
  Acceptance: The new lint check passes and fails when a fallback is edited without the locale; first-paint text matches post-i18n text on both panels.
  Confidence: Verified
  Effort: M

- [ ] P2 — F378 — CEP collapsible headers are mouse-only
  Category: a11y
  Where: `extension/com.opencut.panel/client/main.js` around the collapsible-header click wiring (~13378). UXP already does this in `uxp-ui-controller.js:363-386`.
  Problem: CEP section headers toggle on click only. They have no `role="button"`, `tabindex`, `aria-expanded`, or Enter/Space handling, so keyboard and screen-reader users cannot open Settings subsections the UXP panel already exposes.
  Why this pass skipped it: CEP `main.js` sits on a 18815-line budget (18800 after this audit). The UXP controller already owns the correct pattern; copying it inline would blow the budget. Extract a shared helper or delete dead CEP lines first.
  Acceptance: Every collapsible header is reachable by Tab, announces expanded/collapsed, and toggles on Enter and Space, matching UXP.
  Confidence: Verified
  Effort: S

- [ ] P2 — F379 — Restart Backend and Clear Whisper cache have no confirm
  Category: ux
  Where: `restartBackend()` in `extension/com.opencut.panel/client/main.js:8855`; Whisper cache clear in the same settings card. UXP already uses panel-local confirm for Clear Index.
  Problem: Restart Backend immediately POSTs `/shutdown`. There is no `showPanelConfirm`, so a misclick drops every in-flight job. Clear Whisper cache is similarly one-click destructive.
  Why this pass skipped it: wrapping `restartBackend` in `showPanelConfirm` is ~8 lines and the CEP budget cannot absorb it without an extraction. Add keys next to `settings.restart_backend` in CEP `en.json` when it lands.
  Acceptance: Both actions ask for confirmation with a calm title, what will stop, and a labelled confirm button. Esc and the overlay dismiss without restarting.
  Confidence: Verified
  Effort: S

### P3

- [ ] P3 — F373 — One concept, many names: terminology drift across both panels
  Category: ux
  Where: exact keys per cluster — filler ops (`en.json:465, 510, 712, 934, 959`; `uxp:1043`), Auto Shorts/Magic Clips/Shorts Pipeline (`en.json:1216, 1822, 2154`; `uxp:1517, 1627-1628`), backend vs server (`uxp:46, 1146`; `index.html:146`), engine section names (`en.json:1401`; index.html:3915; `uxp:259`; uxp/index.html:1894), Clear Index vs Clear Library inside one confirm flow (`en.json:1639, 1641, 1649, 1656`), Burn-In vs Burn-in (`en.json:243` vs `600, 735, 1788, 2801`), Whisper model label casing CEP vs UXP (`en.json:309-319` vs `uxp:869-877`), RoFormer spellings (`uxp:1896-1897` vs uxp/index.html:644-645 vs `en.json:121, 125`), Real-ESRGAN vs RealESRGAN within `uxp:1500-1502`, Gist vs gist (`en.json:819-830, 2584-2585`), "Premiere connection required." odd one out (`en.json:2110` vs 196/1978/2111), min-clip-length label vs its own segment tooltip (`en.json:257-258`), faster-whisper install flow failing as "Whisper" (`uxp:1835-1864`)
  Problem: The same object or action carries two to four names, sometimes inside a single dialog (Clear Index button → "Clear footage search library?" title → "Clear Library" confirm → "Footage index cleared" toast). Terminology drift is the strongest "several tools glued together" signal a user gets.
  Evidence: Sweep with exact keys; each hit is one json lookup to confirm.
  Fix: Pick one term per cluster (suggested: "Remove Filler Words", "Magic Clips" as the product name with "Auto Shorts" dropped, "backend", "Engine Routing", "footage index", "Burn-In Captions", upstream spellings "Mel-Band RoFormer"/"BS-RoFormer"/"Real-ESRGAN", "Gist" capitalized as proper noun) and sweep every listed key plus matching es.json entries in one change.
  Acceptance: Each cluster resolves to a single term across en.json (both panels), the HTML fallbacks, and es.json; the Clear Index flow uses one noun end to end.
  Confidence: Likely
  Effort: M

- [ ] P3 — F380 — Generic "Error: {error}" / "Unknown error" copy still has no recovery path
  Category: ux
  Where: nine bare "Error: {error}" keys (en.json:261, 1109, 1888, 1974, 2047; uxp:40, 1591, 1597, 1602), five "Unknown error" strings (en.json:237, 1331, 2166, 2818; uxp:1141). The named-ID leaks from F372 are done.
  Problem: These still dump the raw error with no next step, unlike `cleanUiMessage` / `ERROR_CODE_ACTIONS` in CEP `main.js`.
  Fix: Route them through the existing recovery phrasing. Keep internal detail in logs.
  Acceptance: Grep for `^Error: \{error\}$` and `Unknown error` in both en.json files returns zero user-visible hits, or each remaining hit names a recovery action.
  Confidence: Verified
  Effort: S

- [ ] P3 — F382 — Spaced hyphens stand in for dashes across 52 locale strings
  Category: ux
  Where: 9 CEP keys (`settings.whisper_model_*`, `settings.plugin_publisher_identity`) and 43 UXP keys (`uxp.cut.runtime.filler_done_status`, `uxp.agent.runtime.job_queued`, `uxp.search.runtime.search_ready_*`, `uxp.deliverables.runtime.*`, `uxp.captions.runtime.chapter_line`, …)
  Problem: The house rule bans " - " as a dash substitute in anything a user reads, and these are user-visible status lines and option labels. Some are genuinely prose ("Job queued - watch the progress bar above.", "No steps matched - try a more specific intent."); others are compact separators inside a status line ("{time} - {title}") where a period would read worse.
  Evidence: Enumerated 2026-08-21 while closing F374; the two classes need different treatment, which is why F374 did not sweep them.
  Fix: Split the list into prose and separator uses. Rewrite the prose ones as two sentences; leave or re-punctuate the separators deliberately (a middot is already used elsewhere: `search.files_with_segments` = "{files} • {segments}").
  Acceptance: No user-visible locale string uses " - " as sentence punctuation in either panel; separator uses are a short, documented list.
  Confidence: Verified
  Effort: S

- [ ] P3 — F381 — Areas this audit did not exercise
  Category: quality
  Where: live Premiere CEP/UXP host, WPF installer, Bolt WebView scaffold, MCP extended tool catalogue, `npm run test:rendered` screenshot goldens, CEP disabled Timeline/Deliverables/Rename `title` why-disabled copy
  Why: No Premiere host on this machine; installer and Bolt need their own runtime; the rendered suite is a known 2-4% geometry flake (CLAUDE.md 2026-08-20) and is not theme proof. Disabled-button titles need CEP `main.js` budget the same way F378/F379 do.
  Acceptance: Each surface has a named follow-up with a repro that does not require guessing.
