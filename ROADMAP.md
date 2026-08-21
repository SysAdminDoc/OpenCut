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

### P3

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
