# Roadmap

Single task tracker for known issues and planned work. Items below come from
verified engineering/product audits through 2026-07-29 (with file locations);
fixes already shipped are recorded in CHANGELOG.md and git
history, not here.

Blocked items (credential/license/hardware-gated) live in
[`Roadmap_Blocked.md`](Roadmap_Blocked.md).

## Research-Driven Additions

### P0 — 2026-07-29

### P1 — 2026-07-25

- [ ] P1 — Enforce fresh-clone packaging and documentation integrity
  Why: Tracked entry points and docs reference ignored, retired, or stale surfaces, so a clean clone cannot reproduce the documented build topology.
  Evidence: `BUILD.bat`, root `InstallerBuilder.ps1` hardcoded `0.6.5`, `opencut_server.spec` retired hidden imports, `README.md:465`, `docs/INSTALLER_POLICY.md`; tracked `ROADMAP.md` links to ignored `Roadmap_Blocked.md`.
  Touches: `BUILD.bat`, `InstallerBuilder.ps1`, `opencut_server.spec`, `README.md`, tracked `docs/`, generated-doc/release checks.
  Acceptance: A clean detached clone follows every tracked relative link, required guidance does not depend on ignored local files, `BUILD.bat` invokes the maintained WPF builder and produces the current artifact name, PyInstaller imports are source/dependency-derived, and docs no longer present signing or obsolete .NET/output paths as release gates.
  Complexity: M

- [ ] P1 — Add public-installer-to-current upgrade conformance
  Why: The latest public installer is v1.25.1 while source is v1.43.0, but no automated fixture proves that accumulated local state survives that upgrade.
  Evidence: `README.md:71`, GitHub v1.25.1 release, installer migration/store code, Descript version history, OpenShot recovery documentation.
  Touches: `installer/`, user-data migration helpers, queue/journal/index/review/plugin/settings stores, installer smoke tests.
  Acceptance: An isolated v1.25.1-shaped user-data fixture upgrades to the current build without network access; settings, queue, journal, indexes, review versions, plugin trust, and secrets references remain valid or migrate with explicit backups; rollback/uninstall leaves user data recoverable.
  Complexity: L

- [ ] P1 — Add a release-gated media conformance corpus
  Why: Real FFmpeg integration tests cover basic 24 fps H.264/AAC media but not the timing, channel, color-depth, and path combinations that commonly break automated edits.
  Evidence: `tests/test_integration_ffmpeg.py`; ASR boundary fixture; LosslessCut metadata/timestamp issues; WhisperX alignment issues; Stack Overflow VFR/concat sync cases; Meta FFmpeg engineering; Netflix VMAF model guidance.
  Touches: `tests/fixtures/`, FFmpeg integration tests, trim/merge/caption/interchange/export modules, release smoke.
  Acceptance: Deterministic synthetic or permissively licensed fixtures cover CFR/VFR and non-zero/delayed PTS, 23.976/29.97 drop-frame/59.94, corrupt/decode-error continuation, mono/stereo/multichannel/no-audio, subtitle/data/attachment streams, 8/10-bit HDR/color metadata, rotation, Unicode/space paths, and proxy/original parity; operations assert duration/timecode, A/V sync, stream/channel/pixel metadata, error accounting, and VMAF model identity within declared tolerances.
  Complexity: L

- [ ] P1 — Turn OpenAPI into a typed executable contract
  Why: The generated spec inventories 1,570 operations but omits most request bodies and reduces most responses to an unbounded object, forcing clients to re-read source.
  Evidence: `opencut/openapi.py` emits 3.0.3 at `/openapi.json`; `opencut/core/openapi_spec.py` emits 3.1.0 at `/api/openapi.json`; `opencut/core/fastapi_app.py:558` has another generation adapter; OpenAPI 3.1.1 and MCP 2026-07-28 require JSON Schema 2020-12 semantics.
  Touches: all OpenAPI generation adapters/endpoints, route validation/schema registries, MCP registry, generated spec/tests, client fixtures.
  Acceptance: One OpenAPI 3.1.x schema source feeds the canonical endpoint and explicit compatibility adapters; panel- and MCP-exposed operations have typed request, success, and stable error schemas sourced from shared validators; contract tests exercise generated valid/invalid payloads; a coverage ratchet prevents typed-operation regression.
  Complexity: XL

### P1 — 2026-07-29

- [ ] P1 — Make generated feature readiness fail closed
  Why: Generated feature records drop their implementation-module identity, so installed dependencies can turn terminal `NotImplementedError` adapters into false “available” capabilities.
  Evidence: Live probe resolved `auto.deblur-motion`, `auto.searaft`, and `auto.track-cutie` as available with empty `impl_module`; `opencut/registry.py:145-166,1052-1069`, `opencut/tools/dump_feature_readiness.py:263-316`, corresponding adapters and `tests/test_feature_impl_readiness.py`.
  Touches: feature/readiness generation and loading, stub scanner, route/MCP/panel manifests and counts, readiness tests.
  Acceptance: Generated records preserve or derive implementation modules; every terminal stub remains `stub` regardless of installed imports; all surfaces report the same state/reason/count; the three regressions and a future generated-stub fixture fail if advertised available.
  Complexity: M

- [ ] P1 — Make contract and readiness introspection side-effect free
  Why: A manifest check can execute production app startup, touching user data and starting cleanup, migrations, workers, or plugins during an ostensibly read-only operation.
  Evidence: `opencut/tools/dump_route_manifest.py:289-302`, `opencut/server.py:335-375,463-471`; observed stale-job cleanup and worker startup while building a manifest.
  Touches: Flask app factory/blueprint registration, route/readiness/OpenAPI generators, generator tests, release checks.
  Acceptance: Contract generators use an explicit introspection app or registration graph; tests sentinel filesystem, user-data, subprocess, thread, network, migration, cleanup, and plugin boundaries; repeated generation is deterministic and leaves no runtime state changes.
  Complexity: M

- [ ] P1 — Finish UXP Sequence Index as an accessible working table
  Why: The UI promises a spreadsheet view but discards returned rows and shows only a summary, leaving backend filter/sort and host locators unusable.
  Evidence: `extension/com.opencut.uxp/index.html:1502-1512`, `extension/com.opencut.uxp/main.js:8460-8504`, `opencut/core/sequence_index.py:343-525`, `opencut/routes/sequence_index_routes.py:51-179`; Adobe Premiere 26.3 Sequence Index.
  Touches: UXP Sequence Index markup/controller/styles/locales and host payload, `opencut/core/sequence_index.py`, filter routes/client, host jump action, CSV export, rendered/unit tests.
  Acceptance: Rows render with correct table/grid semantics and keyboard navigation; search, sortable columns, offline/flash/effects filters, result count, loading/empty/error states, column persistence, host jump, and filtered CSV export work without re-walking the timeline; large indexes remain responsive.
  Complexity: L

- [ ] P1 — Migrate MCP across the 2026-07-28 protocol boundary
  Why: OpenCut hardcodes `2024-11-05` and excludes SDK 2 while the final 2026-07-28 specification changes discovery, stateless metadata, result envelopes, subscriptions, caching, schemas, tracing, and transport security.
  Evidence: `pyproject.toml:147`, `opencut/mcp_server.py:1641,1787`, generated MCP registries; MCP 2026-07-28 specification/changelog and Python SDK 2.0.0 supersede the expired 2026-07-20 date gate in the ignored blocker ledger.
  Touches: MCP dependency and server/transports, REST/OpenAPI schema registry, generated tool registry, auth/origin tests, CEP/UXP MCP status/counts, conformance fixtures, `Roadmap_Blocked.md`.
  Acceptance: The obsolete local blocker row is retired; one conformance matrix proves supported legacy handshakes and the 2026-07-28 `server/discover`/per-request era; modern results carry required `resultType`, identity/capability metadata, cache hints, JSON Schema 2020-12, tracing, and appropriate subscription behavior; legacy clients remain functional; UI counts are generated, not hardcoded; optional extensions are not claimed without tests.
  Complexity: L

- [ ] P1 — Gate standards-labelled outputs with reference validators
  Why: OpenCut labels IMF and IMSC 1.3 output as conformant and presents loudness/VMAF thresholds without independent reference proof, so invalid edge cases can leave with authoritative-looking claims.
  Evidence: `opencut/core/imf_package.py:1-82,421-688`, `opencut/core/caption_interchange.py:361-759`, `README.md:289`; Netflix Photon, W3C IMSC 1.3 test/HRM suites, `imschrm`, EBU Loudness Test Set, ITU-R BS.1770, Netflix VMAF model guidance.
  Touches: IMF/caption/loudness/VMAF modules, isolated validator adapters, conformance fixtures/reports, release smoke, README capability wording.
  Acceptance: Representative IMPs pass Photon `IMPAnalyzer`; IMSC output passes the W3C `imsc1_3` corpus and `imschrm` against the W3C HRM pass/fail suite, including Japanese/style cases; loudness passes applicable EBU reference signals; VMAF receipts identify model/version/scaling/mode; every validator emits a machine-readable report, and unsupported profiles are downgraded from “validated” rather than silently shipped.
  Complexity: L

### P2 — 2026-07-25

- [ ] P2 — Validate command-palette destinations
  Why: Several commands promise a specific destination but target nonexistent sub-tab IDs and silently land on a broad default page.
  Evidence: `extension/com.opencut.panel/client/main.js:4425,13695-13698`; `extension/com.opencut.panel/client/index.html`.
  Touches: CEP command registry, tab/sub-tab activation, HTML IDs, navigation unit/rendered tests.
  Acceptance: Every registered command resolves to exactly one visible, focusable destination; missing targets fail the test suite; Workflow Presets, Project Templates, Keyboard Shortcuts, Job History, and dependency recovery open the promised control.
  Complexity: S

- [ ] P2 — Complete UXP first-run and settings portability
  Why: UXP lacks the CEP panel’s recoverable onboarding, full settings import/export, support-bundle export, and issue-report path.
  Evidence: CEP onboarding/settings implementation and rendered tests; `extension/com.opencut.uxp/index.html` nine-pane Settings surface.
  Touches: UXP onboarding/settings UI, shared settings/support endpoints, locale files, rendered state and keyboard tests.
  Acceptance: A new user can connect, choose media, understand unavailable capabilities, and reach a first successful operation; settings round-trip with schema/version validation and redacted support export; malformed imports are non-destructive and actionable.
  Complexity: L

- [ ] P2 — Make update notices persistent and actionable
  Why: Both panels reduce update availability to a short-lived “visit GitHub” toast with no release notes, retry, or durable destination.
  Evidence: `extension/com.opencut.uxp/main.js:8099-8110`; CEP update-check implementation; Descript changelog; Frame.io version-update UX.
  Touches: CEP/UXP settings and status surfaces, update endpoint/client, locale strings, rendered tests.
  Acceptance: Available updates persist in Settings/About with current/available versions, release notes and a validated browser action; dismissed state is version-scoped; offline/error states retain retry guidance; no update launches automatically.
  Complexity: M

- [ ] P2 — Add forced-colors and high-contrast regression coverage
  Why: Existing light/dark contrast tests do not cover Windows forced-colors behavior, where custom surfaces and focus indicators can disappear.
  Evidence: CEP/UXP CSS lacks `forced-colors` rules; WCAG 2.2 non-text contrast and focus criteria; Playwright forced-colors emulation.
  Touches: shared panel tokens/styles, icon/status semantics, Playwright rendered matrix.
  Acceptance: Both panels remain navigable with `forced-colors: active`; text, focus, selected, disabled, error, and success states remain distinguishable without color alone; screenshots and existing semantic/keyboard checks cover the mode.
  Complexity: M

- [ ] P2 — Version the plugin compatibility contract
  Why: Plugin trust and isolation are strong, but `api_version == 1` is a hard equality check with no host-version range or migration diagnostic for future ecosystem growth.
  Evidence: `opencut/core/plugin_manifest.py:150`, example plugins, OpenTimelineIO schema-versioning guidance, OpenCut-app plugin-first architecture.
  Touches: plugin manifest schema/loader/marketplace, plugin diagnostics CLI/API, examples, compatibility tests.
  Acceptance: Manifests declare schema and supported OpenCut plugin-API ranges; install/start rejects incompatibility before activation with an actionable report; a doctor command validates installed plugins; v1 fixtures remain readable and a documented migration path is test-backed.
  Complexity: M

- [ ] P2 — Split the remaining panel controller hotspots
  Why: CEP and UXP controllers still centralize lifecycle, bridge state, navigation, settings, and result rendering, and were among the highest-churn files in the last 200 commits.
  Evidence: `extension/com.opencut.panel/client/main.js` (~18,200 lines), `extension/com.opencut.uxp/main.js` (~8,700 lines), recent decomposition commits.
  Touches: CEP/UXP controller modules, build/source-safety checks, unit and rendered tests.
  Acceptance: Navigation, update lifecycle, settings/diagnostics, and result-state controllers have explicit imports and teardown contracts; no duplicate global ownership remains; controller size/churn budgets are machine-checked; behavior and rendered snapshots remain unchanged.
  Complexity: L

### P2 — 2026-07-29

- [ ] P2 — Compile workflows into preflighted resumable plans
  Why: Saved workflows validate endpoint names but can run for hours before discovering invalid parameters, unavailable dependencies, incompatible media, output collisions, or a failed later step.
  Evidence: `opencut/core/workflow.py:163-180,186-345,378-388`; OpenCut queue/journal/checkpoint primitives; Descript recovery/version history; auto-editor’s inspectable automation model.
  Touches: workflow schema/compiler/executor, typed OpenAPI/readiness registry, media probe, queue/journal/checkpoints, CEP/UXP plan review and tests.
  Acceptance: Save and Run compile the same immutable plan; preflight validates typed parameters, capabilities, media/streams, space, output policy, network use, and side-effect class; users can preview and explicitly approve destructive/cloud steps; completed idempotent steps persist with artifacts/checksums and a failed or restarted workflow resumes safely without repeating them.
  Complexity: L

- [ ] P2 — Test production UI states at real breakpoint boundaries
  Why: Current rendered coverage can pass synthetic state markup, treat placeholder/value as an accessible name, and miss the exact widths where panel layouts change.
  Evidence: `extension/com.opencut.panel/tests/rendered/panel-regression.spec.mjs:12-20,602-626,1692-1701`; CEP/UXP media queries at 620, 700, 820/821, 980, 1020, and 1050.
  Touches: production state renderers, shared rendered fixtures/helpers, CEP/UXP viewport matrix, accessibility and overflow assertions.
  Acceptance: Loading/empty/offline/permission/error/destructive/success states are produced through production renderers; accessible names follow the platform computation and never pass from placeholder/value alone; each actual breakpoint is exercised at boundary minus one, boundary, and boundary plus one in both themes with overflow, focus, keyboard, and semantic assertions.
  Complexity: M

- [ ] P2 — Turn the benchmark registry into a reproducible runner
  Why: The repository defines benchmark IDs and advisory budgets but cannot execute or compare them with enough provenance to guide releases or backend choices.
  Evidence: `opencut/core/performance_benchmarks.py`, `tests/test_performance_benchmark_registry.py`, `opencut/core/eval_datasets.py`; VEBench and Netflix VMAF methodology.
  Touches: benchmark CLI/runner, pinned opt-in fixtures, backend adapters, JSON receipts/baselines, diagnostics and release-smoke integration.
  Acceptance: A documented opt-in command runs selected registered backends and records fixture hash/license, model/dependency versions, hardware, seed, warm-up, repeats, timing/memory, and quality metrics; JSON receipts compare only compatible environments with declared tolerances; unavailable backends skip truthfully; release checks may consume a same-host baseline without penalizing different hardware.
  Complexity: M
