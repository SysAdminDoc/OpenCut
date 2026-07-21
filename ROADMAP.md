# Roadmap

Single task tracker for known issues and planned work. Items below come from
verified engineering/product audits through 2026-07-14 (with file locations);
fixes already shipped are recorded in CHANGELOG.md and git
history, not here.

Blocked items (credential/license/hardware-gated) live in
[`Roadmap_Blocked.md`](Roadmap_Blocked.md).

## Research-Driven Additions

### P2

- [ ] P2 — Decompose the CEP and UXP panel monoliths behind contract tests
  Why: Controllers/styles have grown far beyond the repository's own decomposition guidance, making parity, review, and safe UI changes harder; the rendered regression gate now provides the contract needed to extract them safely.
  Evidence: `CONTRIBUTING.md`; `extension/com.opencut.panel/client/main.js` (16,497 lines); CEP `style.css` (16,278 lines); `extension/com.opencut.uxp/main.js` (8,107 lines); UXP `style.css` (4,247 lines); rendered gate in commit `6a44b951`.
  Touches: CEP/UXP state, backend client, i18n, job, timeline, component, token, layout, and bootstrap modules; Vite build; parity/source/release tests.
  Acceptance: shared responsibility boundaries are extracted without changing public IDs, host-action names, route payloads, or visual baselines; entrypoints contain bootstrap/orchestration rather than feature implementations; state/API/i18n/job/timeline modules have focused tests; panel build, parity, browser, i18n, and release-smoke gates pass.
  Complexity: XL

- [ ] P2 — Complete local semantic media search with portable sidecars
  Why: The local CLIP index and `/search/semantic` routes already work, but Premiere's portable `.prmi` workflow highlights OpenCut's remaining relink/invalidation and panel-discovery gap.
  Evidence: `opencut/core/semantic_video_search.py`; `opencut/routes/object_intel_routes.py`; `tests/test_object_intel.py`; Premiere Media Intelligence metadata docs; RESEARCH.md Competitive Landscape.
  Touches: semantic-search cache/sidecar persistence, project relink/invalidation, CEP/UXP search UI, model-readiness copy, ranking fixtures.
  Acceptance: project embeddings persist in a portable versioned sidecar; moved/relinked or changed media is deterministically reused or invalidated; CEP and UXP expose ranked natural-language results; all computation remains local; readiness, ranking, relink, and invalidation paths are tested.
  Complexity: L

### P3

- [ ] P3 — Run installer user-area file operations as the invoking user
  Why: With `PrivilegesRequired=admin`, a standard user elevating with a separate admin account gets the CEP panel, installer manifest, and ffmpeg PATH written into the admin's profile while PlayerDebugMode (correctly written via `runasoriginaluser`) targets the invoking user; uninstall then backs up the wrong (usually empty) `.opencut`. Data preservation errs safe (wrong profile → NOT_FOUND → real data untouched), so P3.
  Where: `OpenCut.iss:170` (`WriteInstallerManifest`), `:283` (`InstallCEPExtension`), `:311` (`AddToPath`), `:494-495` (uninstall `ConfigDir`).
  Acceptance: user-area copies run via a `runasoriginaluser` helper (as reg.exe already does), so per-user artifacts land in the invoking user's profile regardless of the elevation account.

- [ ] P3 — Flatten the CEP command-center stylesheet into a single layer
  Why: `command-center.css` is three stacked authoring passes; the second `:root` block fully redefines the first and ~500 lines (sidebar width, radius, title sizing, duplicated media queries) are overridden wholesale later in file order. Future edits must reason through the dead cascade, and the `html.theme-light` token block only stays coherent by luck.
  Where: `extension/com.opencut.panel/client/command-center.css`.
  Acceptance: one token layer and one rule per selector; rendered CEP/UXP visual baselines and geometry/contrast tests still pass unchanged.

- [ ] P3 — Isolate Depth Anything 3 behind a compatible worker
  Why: The official DA3 package requires NumPy 1.x and regular `opencv-python`, which cannot resolve in OpenCut's supported NumPy 2 / `opencv-python-headless>=4.13` process; an isolated runtime is required before DA3 can be advertised safely.
  Evidence: `pyproject.toml`; `https://pypi.org/pypi/depth-anything-3/0.1.1/json` (dependencies and official model-license table); `https://pypi.org/pypi/opencv-python-headless/4.13.0.92/json`; `opencut/core/cinefocus.py`.
  Touches: a DA3 worker/venv boundary, lifecycle and IPC, depth engine registry/model cards, CineFocus depth conversion, optional-runtime setup, license/dependency/regression tests.
  Acceptance: OpenCut and the DA3 runtime resolve independently with no duplicate `cv2` wheels; only Apache-2.0 Small/Base checkpoints are selectable; DA3 depth is converted to the same near/far convention consumed from DA2 disparity; cancellation, worker failure, and DA2 fallback are deterministic; generated readiness/model artifacts and an end-to-end isolated fixture pass before DA3 can become a default.
  Complexity: L

- [ ] P3 — Localize the Python/CLI backend and add panel locales beyond en/es
  Why: The CEP and UXP panels ship English + Spanish only while the Python/CLI backend has no i18n framework (English-only error strings) and no RTL support anywhere, despite DE/FR/JA/PT labels already appearing untranslated in `en.json`.
  Evidence: `extension/com.opencut.panel/client/locales/`, `extension/com.opencut.uxp/locales/` (only `en`/`es`); no gettext/babel in `opencut/`; no `dir="rtl"` in panel HTML.
  Touches: a backend i18n layer (gettext/babel), CLI/route/core user-facing strings, new panel locale files, RTL layout handling, locale-lint/release tests.
  Acceptance: backend user-facing strings are translatable with an English fallback; at least one additional panel locale beyond en/es ships and passes the existing locale-lint gate; an RTL locale renders without layout breakage; a test guards locale-key parity across languages.
  Complexity: L
