# OpenCut Researcher Queue - Cycle 6 - 2026-06-04

This companion note records the Cycle 6 research finding while the build lane is
actively editing `ROADMAP.md`, `TODO.md`, `RESEARCH_REPORT.md`, and
`PROJECT_CONTEXT.md` for E15 batches. Reconcile this item into the canonical
roadmap once those shared docs are clean.

## Evidence Reviewed

- `py -3.12 -m opencut.tools.adobe_premierepro_versions --check --json`
  returned drift for `@adobe/premierepro`: `beta` moved from
  `26.3.0-beta.67` to `26.3.0-beta.85`, and the live registry now includes
  `release-26.2: 26.2.1`.
- `npm view @adobe/premierepro dist-tags --json` returned
  `latest: 26.2.0`, `beta: 26.3.0-beta.85`, and `release-26.2: 26.2.1`.
- `npm view @adobe/premierepro versions --json` confirmed `26.2.1` is a
  published version alongside the 26.3 beta train.
- `opencut/tools/adobe_premierepro_versions.py` still defines
  `TRACKED_TAGS = ("latest", "beta")`, and `tracked_versions` omits
  `release-26.2`.
- `tests/test_adobe_premierepro_versions.py` documents the F251 contract as a
  `latest` / `beta` pair, not all release-channel tags.
- npm's official dist-tag docs say tags can provide named release streams and
  that npm gives special install behavior only to `latest`, so project-specific
  stable stream tags such as `release-26.2` need explicit tracker policy.

## Promoted Item

- [ ] **P2 - RA-16 Track Adobe release-channel dist-tags in F251**
  - Why: OpenCut's CEP-to-UXP migration depends on Adobe UXP package movement,
    but F251 currently treats only `latest` and `beta` as first-class tracked
    versions even though Adobe now publishes a stable `release-26.2` tag that is
    newer than `latest`.
  - Evidence: live npm registry dist-tags for `@adobe/premierepro`
    (`https://registry.npmjs.org/@adobe%2Fpremierepro`), npm dist-tag docs
    (`https://docs.npmjs.com/cli/dist-tag/`), local
    `opencut/tools/adobe_premierepro_versions.py`, and
    `tests/test_adobe_premierepro_versions.py`.
  - Touches: `opencut/tools/adobe_premierepro_versions.py`,
    `opencut/_generated/adobe_premierepro_versions.json`,
    `.github/workflows/adobe-premierepro-versions.yml`,
    `scripts/release_smoke.py`, `tests/test_adobe_premierepro_versions.py`,
    and `docs/UXP_MIGRATION.md`.
  - Acceptance: F251 treats `release-*` tags as explicit stable release-channel
    inputs, includes them in tracked snapshot/report metadata, keeps drift issue
    copy from implying beta-only review, and preserves fail-open release-smoke
    behavior.
  - Verify: `py -3.12 -m pytest tests/test_adobe_premierepro_versions.py -q`
    and `py -3.12 -m opencut.tools.adobe_premierepro_versions --check --json`.
  - Complexity: S.

## Non-Duplicates Checked

- F251 already polls the package and compares all `dist_tags`, so this is not a
  replacement for F251. The gap is first-class policy/test/report handling for
  stable release-channel tags that are not `latest`.
- F252/F253 remain the implementation lanes for UXP WebView cutover and hybrid
  native add-ons. RA-16 only hardens the registry signal they consume.
- Existing Node advisory work remains unchanged: `node scripts/check-advisories.mjs
  --json` still reports one allowed Vite advisory and zero unwaived advisories.
