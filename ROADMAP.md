# Roadmap — OpenCut

Actionable work only. Historical and completed roadmap material is archived in CHANGELOG.md; blocked work is kept in Roadmap_Blocked.md.

## Research-Driven Additions

Added 2026-08-10, extended 2026-08-11 and 2026-08-20 from the research passes recorded in `RESEARCH.md`.
IDs continue the existing F-number scheme (highest prior allocation before 2026-08-11: F318; before
2026-08-20: F334).

### P1

### P2

- [ ] P2 — F351 — Express a boundary-crossing cut through the typed UXP API
  Why: F349 landed the typed cut path on 2026-08-20, but `createRemoveItemsAction` removes whole track items and the typed 26.3 API exposes no razor, so the most common case — silence inside a single long clip — still falls back to `sequence.rippleDelete()`, which premiere-pro-mcp #21 measured returning success while changing nothing on 26.3. The fallback is now reported rather than silent, so the gap is visible, not fixed.
  Evidence: `extension/com.opencut.uxp/uxp-cut-planner.js` (`planCutRemoval` returns `removable: false` with a `straddling` list); `extension/com.opencut.uxp/main.js` (`_applyOneCut`, `_rippleDeleteFallback`); `tests/test_uxp_cut_planner.py::test_an_item_spanning_the_whole_range_blocks_the_typed_path`; `@adobe/premierepro` 26.3.0 typings expose `createCloneTrackItemAction`, `createSetStartAction`, `createSetEndAction`, `createSetInPointAction`, `createSetOutPointAction` but no razor/split action
  Touches: `extension/com.opencut.uxp/uxp-cut-planner.js`, `extension/com.opencut.uxp/main.js`, `opencut/tools/adobe_uxp_compatibility.py` (new capabilities), `tests/test_uxp_cut_planner.py`
  Acceptance: A cut whose range crosses a track-item boundary is applied by cloning the item and trimming the two halves, then ripple-removing what lies between, all inside one `executeTransaction` so a partial failure cannot leave a half-cut timeline; the ripple accounting is verified against the read-back contract; `planCutRemoval` reports the split plan rather than refusing; the `rippleDelete` fallback is removed only once the live-host lane confirms the replacement.
  Note: the ripple arithmetic across mixed trims and removals is the risky part and cannot be proven headless — F252's live-Premiere lane gates dropping the fallback, not writing the code.
  Complexity: M

- [ ] P2 — F328 — Ratchet the direct-surface ratio so new routes cannot ship unreachable
  Why: The repo's own manifest reports 280 of 1,568 shipped routes reachable from any first-party surface (17.9%), 1,288 integration-only, and zero routes whose primary surface is the CLI — so every wave adds API faster than it adds product, and the ratio is measured but nothing stops it falling.
  Evidence: `opencut/_generated/route_manifest.json` → `surface_coverage.summary` (`direct_surface_routes: 280`, `integration_only_routes: 1288`, `coverage_percent: 17.9`, `primary_counts.cli: 0`); 19 CLI commands in `opencut/cli.py`; 88 MCP tools in `opencut/_generated/mcp_server_registry.json`; the gate at `surface_coverage.gate` only asserts every route is classified, never that the ratio holds
  Touches: `opencut/tools/dump_route_manifest.py`, `scripts/release_smoke.py`, `opencut/cli.py`, `opencut/core/mcp_tools.py`, `opencut/core/command_palette.py`, `tests/`
  Acceptance: The release gate fails when `coverage_percent` falls below the value recorded at the time the ratchet lands; a new route must either declare a surface or carry an explicit `integration-only` justification that the gate records; the report names the largest integration-only route families so a triage or deprecation decision has data behind it.
  Complexity: M

### P3

- [ ] P3 — F350 — Rename the co-named per-panel assets so one filename means one file
  Why: `backend-client.js`, `command-center.css`, `command-center-layout.css`, and `command-center-tokens.css` each exist under the same name in both panels while being genuinely different implementations (for example `command-center.css` is 2,416 lines in CEP and 2,627 in UXP) — one name for two files at the same cascade position invites edits landing in the wrong panel, which is the failure the shared-asset drift gate cannot catch because these were never copies.
  Evidence: `tests/test_shared_panel_assets.py` records all four in `panel_specific` with byte comparisons; RESEARCH.md 2026-08-11 named `command-center.css` shipping "as two unrelated files under one name and cascade position"
  Touches: `extension/com.opencut.panel/client/`, `extension/com.opencut.uxp/`, both `index.html` link/script tags, `scripts/i18n_lint.py` and any build/verification file lists, `tests/test_shared_panel_assets.py`
  Acceptance: Each per-panel asset carries a name that identifies its panel (or lives under a panel-scoped directory), every referencing tag and tool list is updated, and the entries are removed from `panel_specific` because the collision no longer exists.
  Complexity: S

- [ ] P3 — F352 — Replace the hand-maintained lockfiles with a generated PEP 751 `pylock.toml`
  Why: F317 landed the PEP 639 half on 2026-08-20 (SPDX expression, `license-files`, deprecated classifier removed, verified in built wheel metadata). The lockfile half is untouched: four hand-maintained `requirements-*-lock.txt` files, one of them 126 KB, still carry a version-sync surface that has accumulated dozens of `fix:` commits, and pip 26.1+ installs a standard `pylock.toml` directly.
  Evidence: `requirements-lock.txt`, `requirements-build-lock.txt`, `requirements-release-lock.txt`, `requirements.txt`; `python -m pip lock --help` on pip 26.2.1 works but is marked EXPERIMENTAL and needs a network resolve; https://peps.python.org/pep-0751/
  Touches: `pyproject.toml`, `requirements*.txt`, `scripts/sync_version.py`, `scripts/check_dependency_matrix.py`, `scripts/release_smoke.py`, `Dockerfile`, `docs/`
  Acceptance: A generated `pylock.toml` reproduces the release environment and is verified in release smoke; the bespoke lockfiles are either removed or generated from it; the version-sync target count is updated to match.
  Note: `pip lock` is EXPERIMENTAL and resolves over the network, so pinning a release lane on it needs a deliberate call about reproducibility and offline builds — decide that before wiring it into the release gate. The torch/faster-whisper extras make a full resolve slow.
  Complexity: M

- [ ] P3 — F344 — Rank repeat clusters with a best-take recommendation
  Why: `repeat_detect` finds repeated sentences but ranks nothing, so review shows "these repeat" without "keep this one" — while AutoCut Repeat and Gling sell exactly the keep-best-take selection, and a heuristic (filler count, WPM stability, completion) with an optional LLM verdict layers cleanly on the existing detection output.
  Evidence: `opencut/core/repeat_detect.py` (detection and range merging only); https://github.com/rafcopy/auto-cut-agent (LLM-based take dedup in a UXP+local-server design, 2026-08-13); https://opentools.ai/tools/gling-ai (bad-take marking)
  Touches: `opencut/core/repeat_detect.py`, `opencut/core/llm.py` consumers, `opencut/routes/captions.py` (`/captions/repeat-detect`), both panels' cut-review surfaces, `tests/test_new_modules.py`
  Acceptance: Each repeat cluster carries a ranked keep-candidate with per-take signals (filler count, speech-rate stability, sentence completion, optional LLM verdict with recorded fallback); the review UI preselects the keep and cuts the rest; the heuristic path works with no LLM configured; existing detect-only output remains available.
  Complexity: M

- [ ] P3 — F346 — Activate the /analyze/video/qwen3vl lane through local Ollama vision models
  Why: Content-aware editing ("cut the boring parts" from semantic video understanding, not just audio) is where the closest CLI competitor and the agentic wave are heading, and OpenCut already has the route stubbed (`/analyze/video/qwen3vl`, 501) plus an LLM layer that fronts Ollama — which serves Qwen-VL-class models locally — so one stub activation delivers per-segment semantic relevance scoring with no cloud key.
  Evidence: `opencut/_generated/route_manifest.json` (qwen3vl/internvl3 stubs); https://github.com/WyattBlue/auto-editor/issues/1273 (content-aware edit method, 2026-06-25); `opencut/core/llm.py` (Ollama support); the text-first economy pattern from browser-use/video-use recorded in the 2026-08-11 pass
  Touches: `opencut/core/multimodal_qwen3vl.py` (remove terminal NotImplementedError per readiness rules), `opencut/core/llm.py`, the wave_qrs route, highlights integration, `opencut/model_cards.py`, `tests/`
  Acceptance: The route leaves stub state through the established readiness flow (stub_scan reclassifies it once the terminal raise is gone and `check_X_available()` gates it); frame-sampled scoring returns per-segment relevance keeping transcript as the primary signal and pixels at decision points; it runs against a local Ollama vision model with no API key; the manifest and README counts regenerate.
  Complexity: L
