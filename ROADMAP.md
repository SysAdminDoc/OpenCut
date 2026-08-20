# Roadmap — OpenCut

Actionable work only. Historical and completed roadmap material is archived in CHANGELOG.md; blocked work is kept in Roadmap_Blocked.md.

## Research-Driven Additions

Added 2026-08-10, extended 2026-08-11 and 2026-08-20 from the research passes recorded in `RESEARCH.md`.
IDs continue the existing F-number scheme (highest prior allocation before 2026-08-11: F318; before
2026-08-20: F334).

### P1

### P2

- [ ] P2 — F358 — Diagnose the panel-rendered failure blocking the release gate
  Why: `panel-rendered` is the fourth step failing the release gate and the only one not yet characterised. The Playwright goldens have drifted 2 to 3 percent on this machine before without any CSS change, so this may be environmental, but it has not been confirmed and a release cannot be cut until it is one or the other.
  Evidence: `python scripts/release_gate.py verify` reports `release smoke failed: media-conformance, pip-audit, panel-rendered, npm-advisory`; media-conformance is fixed, the other two are F356 and F357
  Touches: `extension/com.opencut.panel/tests/rendered/`, `extension/com.opencut.panel/playwright.config.mjs`
  Acceptance: The failure is identified as either a real regression, which is fixed, or as environment-driven rendering drift, which is recorded with the measured delta and handled so the gate stops failing on it. Do not update goldens to hide a real change — inspect the expected/actual/diff artifacts first.
  Note: the suite needs more than three minutes for its 63 single-worker cases, so budget for that.
  Complexity: M

### P3

- [ ] P3 — F354 — Give the largest integration-only route families a surface or a deprecation date
  Why: F328 landed the ratchet on 2026-08-20, so the ratio can no longer fall — but it locked in at 17.9%, with 1,313 routes reachable from nothing a user can click. The gate names where the mass sits (`wave_l` 80, `platform_infra` 55, `wave_k` 48, `wave_qrs` 42, `integration` 40), which is the data the decision needed and did not have. Holding the line is not the same as fixing it.
  Evidence: `python -m opencut.tools.dump_surface_ratchet --check` prints the ten largest families; `opencut/_generated/surface_ratchet.json` records all 104; `opencut/_generated/route_manifest.json` → `surface_coverage.summary.coverage_percent: 17.9`
  Touches: `opencut/cli.py`, `opencut/core/command_palette.py`, `opencut/core/mcp_tools.py`, `opencut/_generated/surface_ratchet.json`, `tests/`
  Acceptance: Each of the largest families is either given a first-party surface (a CLI command, a palette entry, or a curated MCP tool) or recorded as deprecated with a removal target; the ratchet baseline is re-recorded upward with the commit stating which families moved and why; `primary_counts.cli` is no longer zero.
  Note: work one family at a time and re-record the floor after each — a single sweeping pass would make the ratchet meaningless for the next change.
  Complexity: L

- [ ] P3 — F352 — Replace the hand-maintained lockfiles with a generated PEP 751 `pylock.toml`
  Why: F317 landed the PEP 639 half on 2026-08-20 (SPDX expression, `license-files`, deprecated classifier removed, verified in built wheel metadata). The lockfile half is untouched: four hand-maintained `requirements-*-lock.txt` files, one of them 126 KB, still carry a version-sync surface that has accumulated dozens of `fix:` commits, and pip 26.1+ installs a standard `pylock.toml` directly.
  Evidence: `requirements-lock.txt`, `requirements-build-lock.txt`, `requirements-release-lock.txt`, `requirements.txt`; `python -m pip lock --help` on pip 26.2.1 works but is marked EXPERIMENTAL and needs a network resolve; https://peps.python.org/pep-0751/
  Touches: `pyproject.toml`, `requirements*.txt`, `scripts/sync_version.py`, `scripts/check_dependency_matrix.py`, `scripts/release_smoke.py`, `Dockerfile`, `docs/`
  Acceptance: A generated `pylock.toml` reproduces the release environment and is verified in release smoke; the bespoke lockfiles are either removed or generated from it; the version-sync target count is updated to match.
  Note: `pip lock` is EXPERIMENTAL and resolves over the network, so pinning a release lane on it needs a deliberate call about reproducibility and offline builds — decide that before wiring it into the release gate. The torch/faster-whisper extras make a full resolve slow. Confirmed available 2026-08-20 on pip 26.2.1 (`python -m pip lock --help` accepts a local project path or requirement specifiers), so the tool is not the blocker; the decision and the release-lane verification are.
  Update 2026-08-20: the release-gate blocker is cleared — the ten in-flight files landed, so `release_gate.py verify` runs again and a release-lane change can now be verified end to end.
  Probed 2026-08-20, so the next pass starts from facts rather than guesses. `python -m pip lock .` resolves this project's base dependencies in roughly a minute and writes a valid `lock-version = "1.0"` file of about 9.4 KB. Two traps found: the output path must sit inside the project tree, because PEP 751 records package paths relative to the output's directory and pip raises `ValueError: ... is not in the subpath of ...` otherwise; and the filename must be `pylock.toml` or `pylock.<name>.toml` or pip warns it is not a valid lock file name. The extras were not probed — the torch/faster-whisper resolve is the slow one, and that is the number that decides whether this can sit in a gate at all.
  Remaining judgement call, unchanged: `pip lock` is EXPERIMENTAL and resolves over the network. Recommended shape when this is picked up — commit the generated `pylock.toml` as an artifact and have release smoke verify it offline against the existing lockfiles, rather than having the gate itself resolve. That keeps releases reproducible and network-independent. Removing the four bespoke lockfiles is the risky half and should be a separate change from adding the artifact.
  Complexity: M

- [ ] P3 — F346 — Activate the /analyze/video/qwen3vl lane through local Ollama vision models
  Why: Content-aware editing ("cut the boring parts" from semantic video understanding, not just audio) is where the closest CLI competitor and the agentic wave are heading, and OpenCut already has the route stubbed (`/analyze/video/qwen3vl`, 501) plus an LLM layer that fronts Ollama — which serves Qwen-VL-class models locally — so one stub activation delivers per-segment semantic relevance scoring with no cloud key.
  Evidence: `opencut/_generated/route_manifest.json` (qwen3vl/internvl3 stubs); https://github.com/WyattBlue/auto-editor/issues/1273 (content-aware edit method, 2026-06-25); `opencut/core/llm.py` (Ollama support); the text-first economy pattern from browser-use/video-use recorded in the 2026-08-11 pass
  Touches: `opencut/core/multimodal_qwen3vl.py` (remove terminal NotImplementedError per readiness rules), `opencut/core/llm.py`, the wave_qrs route, highlights integration, `opencut/model_cards.py`, `tests/`
  Acceptance: The route leaves stub state through the established readiness flow (stub_scan reclassifies it once the terminal raise is gone and `check_X_available()` gates it); frame-sampled scoring returns per-segment relevance keeping transcript as the primary signal and pixels at decision points; it runs against a local Ollama vision model with no API key; the manifest and README counts regenerate.
  Complexity: L
