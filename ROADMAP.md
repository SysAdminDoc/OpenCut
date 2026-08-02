# Roadmap

Single task tracker for known issues and planned work. Items below come from
verified engineering/product audits through 2026-07-29 (with file locations);
fixes already shipped are recorded in CHANGELOG.md and git
history, not here.

Blocked items (credential/license/hardware-gated) are tracked separately in a
maintainer-local `Roadmap_Blocked.md`, which is deliberately untracked — it is
a working file, not part of a clone. This file is the tracked queue.

## Research-Driven Additions

### P0 — 2026-07-29

### P1 — 2026-07-25

### P1 — 2026-07-29

### P2 — 2026-07-25

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

## Audit Findings — 2026-08-02

Baseline recorded before this audit (all green, so every item below is a new
finding, not a pre-existing failure): `py -3.12 -m pytest tests/ -q` →
**10726 passed, 21 skipped, 4656 subtests, 0 failed** (686 s);
`ruff check opencut/ --select E,F,I --ignore E501` → clean;
`scripts/sync_version.py --check` → all files in sync at v1.46.0; all five
generated manifests report in-sync; `npx playwright test` (panel rendered
suite) → 57 passed, 1 skipped; `npm run lint` → 0 errors, 24 warnings.

### P1 — 2026-08-02


### P2 — 2026-08-02


- [ ] P2 — Escape colons (and backslashes) in every drawtext text value
  Category: correctness
  Where: `opencut/core/instant_replay.py:210` (`text_esc = config.overlay_text.replace("'", "\\'")`); `opencut/core/motion_graphics.py:98-99,231-232`. The correct helper already exists: `opencut/helpers.py:447` (`escape_drawtext`).
  Problem: These sites escape only `'` (and sometimes `\`), never `:`. In an FFmpeg filtergraph the parser strips quotes before the option parser splits on `:`, so single-quoting does **not** protect a colon — which is precisely why `helpers.escape_drawtext` escapes it even inside quotes. Colons are routine in titles and overlays, so the user gets an opaque job failure for ordinary input. None of these sites sets `expansion=none`, so literal `%{...}` in user text is also evaluated as a drawtext expression.
  Evidence: Probed the bundled `ffmpeg/ffmpeg.exe` directly. `-vf "drawtext=text='Round 2: FIGHT'"` fails with `No option name near ' FIGHT'` / `Error parsing filterchain`. The same command with the colon escaped (`'Round 2\: FIGHT'`) succeeds (only fontconfig warnings). Reachable via `POST /gaming/instant-replay` (`opencut/routes/gaming_routes.py:442`) with `overlay_text: "REPLAY: Goal #2"`, and via `POST /video/title-card` (`opencut/routes/video_specialty.py:124`) with `text: "Episode 2: The Return"`.
  Fix: Route both modules through `helpers.escape_drawtext` and pair it with `expansion=none`, as that helper's docstring already mandates. A wider sweep of modules with their own local escapers is filed separately below.
  Acceptance: A test renders drawtext with `Round 2: FIGHT`, `C:\media\clip`, and `100%{x}` through each affected entry point and asserts FFmpeg exits 0 and the rendered text is literal.
  Confidence: Verified
  Effort: S

- [ ] P2 — Make the queue allowlist reject sync routes (queued work runs, then reports failure)
  Category: correctness
  Where: `opencut/routes/jobs_routes.py:205,211,221` (entries in `_ALLOWED_QUEUE_ENDPOINTS`) and `:989-997` (`_dispatch_queue_entry`).
  Problem: `/video/lut/blend`, `/video/multicam-cuts`, and `/video/multicam-xml` are queueable but are not `@async_job` routes. `_dispatch_queue_entry` requires a `job_id` in the response; a sync route returns its full result with none, so the entry is marked `error: "Route did not return a job ID"` **after the work has already executed and written its output**. The user sees a failed queue entry for work that succeeded. If a sync handler exceeds `QUEUE_DISPATCH_TIMEOUT` (60 s), the timeout fires while the dispatch thread keeps executing it untracked.
  Evidence: Enumerated against the live `url_map`: of 216 allowlist entries, exactly these 3 resolve to view functions whose `_opencut_async_job` attribute is `False`; all three exist and accept POST.
  Fix: Either promote the three routes to `@async_job`, or teach `_dispatch_queue_entry` to treat a 2xx response without a `job_id` as `complete` — the workflow engine already handles exactly this shape at `opencut/core/workflow.py:290-291`. Then add a release-gate test that diffs the allowlist against the live route table and fails on any entry that is not an async job.
  Acceptance: The new test fails while any allowlist entry is a sync route and passes afterwards; queuing `/video/multicam-cuts` completes as `complete`, not `error`.
  Confidence: Verified
  Effort: S

- [ ] P2 — Propagate workflow cancellation to the running sub-job
  Category: reliability
  Where: `opencut/core/workflow.py:378-416` (`_wait_for_job`) and `:216-228` (the between-steps cancel check).
  Problem: The parent-cancel check runs only *between* steps. `_wait_for_job` polls a sub-job for up to 3600 s without ever checking the parent, and cancelling the parent kills only the process registered under the parent's job id — the sub-job has its own id and its own registered FFmpeg child, which is untouched. Cancelling a workflow during a long step returns control to the user immediately while the heavy work keeps running, holding a worker slot and a concurrency slot to completion. The same applies on step timeout: the workflow errors out and abandons a still-running sub-job.
  Evidence: `_wait_for_job`'s loop has no reference to `parent_job_id` (the parameter is not consulted inside the polling loop); `_cancel_job(parent)` → `_kill_job_process(parent)` looks up only the parent's registered process.
  Fix: Check `_is_cancelled(parent_job_id)` inside `_wait_for_job`'s poll loop, and call `_cancel_job(sub_job_id)` on parent-cancel and on step timeout.
  Acceptance: A test starts a two-step workflow, cancels the parent during step 1, and asserts the sub-job reaches `cancelled` and its process is no longer registered.
  Confidence: Verified (code trace)
  Effort: S

- [ ] P2 — Catch job-poll rejections in the UXP social upload flow
  Category: correctness
  Where: `extension/com.opencut.uxp/main.js:5836-5861` (`runSocialUpload`); `extension/com.opencut.uxp/job-controller.js:178-192` (`poll`).
  Problem: `JobPoller.poll()` rejects on job error (`onError = (message) => reject(new Error(message))`) and also when another job is already active (`markJobStarting()` returns false). `runSocialUpload` is the only `JobPoller.poll` call site without a surrounding try/catch, so on upload failure — unconfigured OAuth, a mid-upload network drop — the async function throws past the trailing `UIController.hideProcessing()` and `UIController.setButtonLoading("socialUploadBtnUxp", false)`. The processing banner stays up with `aria-busy="true"`, the Upload button stays disabled in its loading state, no error toast appears, and the rejection is unhandled. Cancel cannot recover it (`cancel()` returns false with no active job), so only a panel reload clears it.
  Evidence: Of the 12 `JobPoller.poll(` call sites in `main.js`, a `try {` appears within the preceding 14 lines for all of them except line 5849 (`runSocialUpload`). `runUpscaleUxp` at `:7814-7830` shows the correct pattern.
  Fix: Wrap the poll in try/catch/finally mirroring `runUpscaleUxp`, showing an error toast and clearing the processing/loading state in `finally`.
  Acceptance: A test rejects the poll for the upload flow and asserts the processing banner is hidden, the button is re-enabled, and an error toast is shown.
  Confidence: Verified
  Effort: S

- [ ] P2 — Re-check the active job id after an awaited status fetch (cancel race)
  Category: correctness
  Where: `extension/com.opencut.uxp/job-controller.js:127-163` (`pollJob`), `:165-176` (`cancel`).
  Problem: `schedulePoll` checks `state.activeJobId === jobId` before starting a poll, but `pollJob` never re-checks after its `await client.get(...)` resolves. If the user cancels while a status request is in flight, the response still drives `finishSuccess`/`finishError`, so the feature's `onComplete` runs after "Job cancelled." — producing a success toast that contradicts the cancel — and `fireCompletionHooks()` fires a second time. This is the same race the CEP panel patched in v1.9.20 ("close SSE/poll before nulling `currentJob`"); the extracted UXP controller lost the guard.
  Evidence: `pollJob` dispatches terminal handlers immediately after the awaited fetch with no interleaving `state.activeJobId` comparison.
  Fix: Add `if (state.activeJobId !== jobId) return;` immediately after the awaited status fetch, before dispatching any terminal handler.
  Acceptance: A test cancels a job while a status response is in flight and asserts no completion hook fires and no success toast is shown.
  Confidence: Verified
  Effort: S

- [ ] P2 — Stop tests from writing into the developer's real `~/.opencut`
  Category: testing
  Where: `tests/test_coverage_expansion.py:1064-1097` (`TestAddToQueueClamping`) against `opencut/core/render_queue.py:22,96-97,148-150`; `tests/test_collab_review.py:88-93` (`_session`, used by ~30 tests) against `opencut/core/review_comments.py:30,127-135`; `tests/test_subtitle_pro.py` (`TestMultiLang*`) against `opencut/core/multilang_subtitle.py:28`.
  Problem: These tests exercise real persistence helpers whose paths are module-level constants rooted at `~/.opencut`, and nothing redirects them. `add_to_queue()` appends and immediately `_save_queue()`s to `~/.opencut/render_queue.json`; the test's `finally` restores only the in-memory `_queue` and never re-saves, so the on-disk entry survives. Because `render_queue.py` runs `_load_queue()` at import, a real OpenCut session then loads the test entry — and would attempt to render `/test.mp4` if the queue were started. Review sessions use uuid-unique names specifically to avoid cross-test collisions, which means every full run deposits a fresh batch of orphan JSON files that nothing ever removes.
  Evidence: On this machine, `~/.opencut/render_queue.json` contains **35** accumulated entries, all `{"input_path": "/test.mp4", "preset_name": "preset", "error": "Unknown preset: preset"}`, with `created_at` timestamps spanning multiple past runs. `~/.opencut/reviews/` holds **1344** orphan JSON files totalling ~909 KB. The autouse `clear_session_cache()` fixture (`test_collab_review.py:34-39`) clears memory only.
  Fix: Monkeypatch `render_queue._QUEUE_PATH`, `review_comments._REVIEWS_DIR`, and `multilang_subtitle.SUBTITLE_DIR` to `tmp_path` (the file-level autouse pattern in `tests/test_user_data_tombstones.py` is the model). Then add a conftest-level guard that fails any test which creates or modifies a path under the real `~/.opencut`, so this class of leak cannot return.
  Acceptance: The guard fixture fails on the current tests and passes after they are isolated; a full suite run leaves `~/.opencut/render_queue.json` and `~/.opencut/reviews/` byte-identical. Note for the implementer: the 35 queue entries and 1344 review files already on this machine are pre-existing residue and can be deleted once the leak is closed.
  Confidence: Verified
  Effort: S

- [ ] P2 — Report update-check failures honestly instead of claiming the current version is latest
  Category: reliability
  Where: `opencut/routes/system_model_routes.py:339-411` (`check_for_update`), cache at `:318-320`.
  Problem: `result` is seeded with `latest_version = current`, and on any failure — offline, DNS, GitHub 5xx, rate limit — the `except` branch adds `error: "offline"` but leaves that seeded value in place and returns HTTP 200. So the endpoint asserts the latest release equals the installed version when it has no idea, and any client rendering "You're up to date (latest 1.46.0)" is stating something unverified. Worse, the failed result is then written into `_update_cache` with a full `_UPDATE_CACHE_TTL` of 3600 s, so one transient blip suppresses update checks for an hour with no retry. Both panels read only `update_available` (`extension/com.opencut.uxp/main.js:8302-8303`; `extension/com.opencut.panel/client/main.js:2471-2472`) and drop the error entirely, so the user is never told the check failed.
  Evidence: Observed live against v1.46.0: `GET /system/update-check` returned `{"current_version":"1.46.0","error":"offline","latest_version":"1.46.0","update_available":false}` with HTTP 200.
  Fix: On failure set `latest_version` to `null` (not `current`) and keep `update_available` false; cache failures for a much shorter TTL (or not at all) so a retry is possible; have both panels surface the error state with retry guidance rather than silently ignoring it. This refines — and should be implemented together with — the existing "Make update notices persistent and actionable" P2 item above, whose acceptance already calls for offline/error states retaining retry guidance.
  Acceptance: With the network blocked, the endpoint returns `latest_version: null` and an error field, a subsequent call retries rather than serving an hour-old failure, and both panels show a "couldn't check for updates" state with a retry affordance.
  Confidence: Verified
  Effort: S

- [ ] P2 — Repair the four dead or mis-wired CEP controls
  Category: ux
  Where: (a) `extension/com.opencut.panel/client/main.js:16792` with markup at `client/index.html:3213-3220`; (b) `client/main.js:10192-10204` vs controls at `client/index.html:442,449`; (c) `client/main.js:17800-17822` (`tryDemo`) vs `:3101-3126` (`selectFile`); (d) `client/main.js:15272-15286` (`applyMulticamCuts`) vs `:15151-15156` (`applySequenceCuts`).
  Problem: Four shipped controls do nothing, or the wrong thing, with no feedback.
  (a) Deliverables "Open Folder" calls `cs.evalScript('openFolderInFinder(...)')`, but no such ExtendScript function exists anywhere in the repo — the host defines 57 functions and this is not one. ExtendScript returns "EvalScript error." into a no-op callback, so the button is silently dead on a shipped result surface.
  (b) Silence "Preview 10s" reads `el.silenceThreshold` / `el.silenceMinDur`, neither of which exists, so it always falls back to hardcoded `-30 dB / 0.4 s`. The real sliders are `#threshold` and `#minDuration` (the ones `runSilence` uses at `:4931-4932`), so even an untouched panel previews different parameters than it will run — while `index.html:492` promises "hear what gets cut with current threshold".
  (c) `tryDemo` assigns `selectedPath` directly and writes to a `#selectedClipLabel` element that does not exist, instead of calling `selectFile()`. `body.has-clip` is never added, so the "select a clip" empty-state overlay stays up on every workspace panel while the toast says "Loaded demo footage — try any tab" — the first-run demo appears broken.
  (d) `applyMulticamCuts` never checks `r.error`, so a host failure like `{"error":"No active sequence"}` parses fine and shows the **success** toast "Multicam cuts applied: 0". It is also the only sequence-mutating write that bypasses `journalCheckpointedHostWrite`, violating the panel's own stated invariant at `:2345-2347`, so a crash mid-apply leaves no recovery row.
  Evidence: Repo-wide grep finds `openFolderInFinder` only at the call site; `el.silenceThreshold`/`el.silenceMinDur` and `#selectedClipLabel` resolve to nothing while `#threshold`, `#minDuration` exist in `index.html`; `applySequenceCuts` checks `r.error` and `applyMulticamCuts` does not.
  Fix: (a) add `openFolderInFinder(path)` to `host/index.jsx` (the `new File(path).parent.execute()` pattern with project guards and a JSON error return), check the result, and hide the button outside Premiere; (b) read `el.threshold`/`el.minDuration`; (c) replace the body of `tryDemo`'s try-block with `selectFile(data.path)`; (d) route `applyMulticamCuts` through `journalCheckpointedHostWrite` and check `r.error` before the success toast.
  Acceptance: Each control is covered by a test asserting it invokes the intended function with the current UI values and surfaces host errors as errors. Manually: loading demo footage clears the empty state on every tab.
  Confidence: Verified
  Effort: M

- [ ] P2 — Stop the Photon adapter failing clean IMF packages
  Category: correctness
  Where: `opencut/core/standards_validators.py:257-267` (`validate_imf_package` output scan).
  Problem: The scan classifies any output line containing the substring `error` as an error: `if "error" in lowered or "fatal" in lowered: report.errors.append(stripped)`, then `report.passed = completed.returncode == 0 and not report.errors`. Netflix Photon's `IMPAnalyzer` prints a per-asset summary of the form `CPL_<uuid>.xml has no errors or warnings` for **clean** assets — which contains "errors", lands in `report.errors`, and flips `passed` to `False`. The validator therefore reports failure precisely when the package is clean. Any file path containing "error" trips it too.
  Evidence: The substring test is unanchored, and the only Photon test coverage is the missing-jar case (`tests/test_standards_validators.py:176`), so the parsing branch is untested. Latent on this machine (no `OPENCUT_PHOTON_JAR` configured) but wired into the release gate at `scripts/release_smoke.py:1339`.
  Fix: Anchor on Photon's actual severity tokens (lines beginning `ERROR`/`FATAL`, e.g. `^\s*(ERROR|FATAL)\b`) and explicitly exclude the "has no errors or warnings" summary line.
  Acceptance: A test feeds recorded Photon output for a clean package and asserts `passed is True` with zero errors, and feeds output containing a real `ERROR` record and asserts `passed is False`.
  Confidence: Verified (static; requires the Photon jar to execute end-to-end)
  Effort: S

- [ ] P2 — Claim the port before marking previous jobs interrupted
  Category: correctness
  Where: `opencut/server.py:585-591` (`mark_interrupted()` / `cleanup_old_jobs()`) versus `:609-630` (port check, `_nuke_old_servers`, `_write_pid`).
  Problem: `run_server` marks every `running` row in the shared `~/.opencut/jobs.db` as `interrupted` *before* checking the port and before the kill sequence. If the kill fails, the second instance deliberately falls back to `port+1..port+10`, so both instances run. Instance B has then marked instance A's actively-running jobs as `interrupted`, making them resume candidates via `/jobs/interrupted` and `/jobs/<id>/resume` — duplicate execution of work still running on A. B's `initialize_job_queue` also rewrites `job_queue.json`, marking A's `running`/`started` entries `SERVER_RESTARTED` while A's queue runner keeps mutating the same file (last-writer-wins).
  Evidence: Statement order in `run_server`; the alternate-port fallback is an intentional, documented path, which is what makes the two-instance state reachable.
  Fix: Move the `mark_interrupted()` / `cleanup_old_jobs()` block to after `_write_pid(effective_port)`, i.e. after this instance has won a port.
  Acceptance: A test starting a second instance while the first holds the port asserts the first instance's `running` rows are untouched.
  Confidence: Verified (code order)
  Effort: S

- [ ] P2 — Write the tc-sync timeline where the user can find it
  Category: correctness
  Where: `opencut/core/tc_sync.py:512-514`; helper signature `opencut/helpers.py:119-124`.
  Problem: The same `output_path()` argument misuse as the beat-cuts item, but with a worse outcome because it succeeds silently. `_output_path(sources[0], "_tc_sync", ext)` passes the extension as `output_dir`, so the timeline is written to a CWD-relative hidden directory with the wrong extension. `generate_synced_timeline` does `os.makedirs` (`:352-354`), so the directory is created and the write succeeds — the user simply never finds the file, and it is named as if it were video. This is the default path for `POST /video/tc-sync` when `output_path` is omitted (`opencut/routes/music_safety_routes.py:114`).
  Evidence: `output_path('C:/media/camA.mp4', '_tc_sync', '.json')` returns `'.json\\camA__tc_sync.mp4'` — hidden `.json/` directory relative to the server's CWD, doubled underscore, and a `.mp4` extension on JSON content.
  Fix: Build the path as `os.path.splitext(_output_path(sources[0], "tc_sync"))[0] + ext`.
  Acceptance: A test calls the entry point without `output_path` and asserts the result sits beside the first source file with the correct `.json`/`.edl` extension and a single underscore.
  Confidence: Verified
  Effort: S

### P3 — 2026-08-02

- [ ] P3 — Delete the test that cannot fail
  Category: testing
  Where: `tests/test_voice_speech.py:859-888` (`TestLipSync.test_extract_audio_features`).
  Problem: The call and all three `assertIn`s are wrapped in `try: ... except Exception: pass  # FFmpeg mock may not capture all paths`. `AssertionError` is an `Exception`, so the test passes regardless of what `_extract_audio_features` returns — including if it raises or returns garbage. It inflates the green count while verifying nothing.
  Evidence: Read at the cited lines; the `except Exception: pass` encloses every assertion.
  Fix: Remove the try/except and let the deterministic mock (which already writes fake PCM) drive real assertions; if a specific environment-dependent exception is genuinely expected, `pytest.skip` on that exact type only.
  Acceptance: Mutating `_extract_audio_features` to return `{}` makes the test fail.
  Confidence: Verified
  Effort: S

- [ ] P3 — Compare the README route claim against the number it actually claims
  Category: testing
  Where: `scripts/check_doc_sizes.py:80-81` (`_route_count` → `_route_manifest_value("total_routes")`) used by the "README routes badge", "README feature overview API routes", and "README architecture API routes" targets at `:103-122`; data in `opencut/_generated/route_manifest.json`.
  Problem: The README states **1,544 shipped API routes** and explicitly defines that as excluding 25 strategic 501 stubs (`README.md:257`), but the checker compares that claim against `total_routes` (1569), a different quantity. The manifest already publishes the right field — `shipped_route_count: 1544` — and it is not used. The check reports −1.6% drift and passes only because it is inside the ±15% tolerance, so it neither validates the claim nor would catch real drift in the shipped count until it exceeded ~230 routes. This is a check wired to the wrong data source: it looks like it ran, and authorises anything.
  Evidence: `python scripts/check_doc_sizes.py` prints `README feature overview API routes  README.md  1544  1569  -1.6%` and concludes "All documented sizes within tolerance". `route_manifest.json` contains `{"total_routes": 1569, "shipped_route_count": 1544, "blueprint_count": 107}`.
  Fix: Point the three shipped-route targets at `shipped_route_count`, and tighten their tolerance to 0 (an exact-match claim deserves an exact check). Decide separately whether the badge, which reads only "API Routes", should show the total or be relabelled "Shipped Routes".
  Acceptance: Changing `shipped_route_count` in the manifest by one fails the check; the current tree passes with an exact match.
  Confidence: Verified
  Effort: S

- [ ] P3 — Neutralise spreadsheet formulas in exported CSV cells
  Category: security
  Where: `opencut/core/sequence_index.py:553-564` (`_csv_cell` / `rows_to_csv`), BOM written at `:600`.
  Problem: `rows_to_csv` writes `name`, `path`, `tags`, and the ASR-derived `transcript_excerpt` verbatim. A clip named `=HYPERLINK(...)` — or transcript text beginning `=`, `+`, `-`, or `@` — becomes a live formula when the export is opened. The file is written with a `utf-8-sig` BOM specifically so Excel opens it cleanly, so Excel is the intended consumer. Transcript content is attacker-influenceable (it is whatever the video says), which is what lifts this above a pure theoretical.
  Evidence: `_csv_cell` applies type formatting (bools, floats, list joins) but no formula-injection guard; the export route is `POST /timeline/sequence-index/export-csv`.
  Fix: Prefix a single quote on any cell whose first character is `=`, `+`, `-`, `@`, tab, or CR, following the OWASP CSV-injection guidance.
  Acceptance: A test exports a row whose clip name is `=cmd|'/c calc'!A1` and asserts the emitted cell is prefixed and inert.
  Confidence: Verified
  Effort: S

- [ ] P3 — Keep `Infinity` out of job-result JSON
  Category: correctness
  Where: `opencut/core/quality_metrics.py:103,224-234,361`; consumed by `opencut/routes/wave_c_routes.py:113-119`.
  Problem: Parsing `average:inf` for a lossless match (fixed earlier today) is correct, but `measure_psnr` then returns `float("inf")`, `round(inf, 3)` is still `inf`, and the value flows into the async-job result dict. Python's `json.dumps` and Flask's default provider serialise that as the bare token `Infinity`, which is not valid JSON — `JSON.parse` in both panels throws. Comparing a stream copy or lossless transcode against its source is a realistic workflow, so a perfect result is the one the client cannot read.
  Evidence: The NaN self-equality filter already applied to VMAF at `:346-351` shows the intended pattern; PSNR has no equivalent.
  Fix: Clamp to a documented sentinel — either a `99.0` dB cap or `null` plus a `notes` entry such as "identical content" — mirroring the VMAF handling.
  Acceptance: A test compares a file with itself and asserts `json.dumps(report)` produces valid JSON that `json.loads` round-trips.
  Confidence: Likely (serialisation path traced; panel parse failure not executed)
  Effort: S

- [ ] P3 — Return 403, not 500, for non-ASCII auth tokens
  Category: reliability
  Where: `opencut/security.py:116` (`is_csrf_token_valid`); same pattern at `opencut/auth.py:395` (`is_token_valid`) and `opencut/core/review_links.py:760` (`get_review`).
  Problem: WSGI decodes headers as latin-1, so `X-OpenCut-Token` can legitimately contain non-ASCII characters. `hmac.compare_digest` raises `TypeError: comparing strings with non-ASCII characters is not supported` for non-ASCII `str` operands, and the exception escapes the CSRF middleware. An unauthenticated caller converts a should-be-403 into a 500, and every 500 appends a full traceback to `~/.opencut/crash.log` (`opencut/server.py:456-468`) with no rotation — a trivial disk-fill and log-flood primitive. It also blinds the security audit trail: `record_csrf_rejection` is never reached, so these attempts never appear in `security_audit.jsonl`. The failure is state-dependent: with an empty token pool the comprehension is empty and returns `False`; once any token has been issued (i.e. after the panel calls `/health`) it raises.
  Evidence: `POST /settings/llm` with header `X-OpenCut-Token: "\xff"` returned 500 `{"code":"INTERNAL_ERROR"}` with a traceback through `security.py:116`.
  Fix: Reject candidates failing `str.isascii()` before comparing (or encode both sides to bytes with `surrogateescape`). Apply to all three sites.
  Acceptance: A test posting a non-ASCII token gets 403, a `csrf_rejection` audit record is written, and `crash.log` is unchanged.
  Confidence: Verified
  Effort: S

- [ ] P3 — Coerce non-string JSON values before calling string methods
  Category: reliability
  Where: `opencut/routes/settings.py:761` (`data.get("api_key", "").startswith("***")`); `opencut/routes/system_model_routes.py:285-290` (`.strip()` on `provider`, `model`, `api_key`, `base_url`).
  Problem: `get_json_dict()` guarantees the body is a JSON object but says nothing about value types; these sites assume `str`. A non-string value yields a 500 `AttributeError` instead of a 400 `INVALID_INPUT`, plus a crash-log append per request (compounding the item above). The surrounding normalisation block at `settings.py:765-771` already uses `str(...)` coercion for exactly this reason — these two sites were missed.
  Evidence: `POST /settings/llm {"api_key": 123}` → 500 `AttributeError: 'int' object has no attribute 'startswith'`; `POST /llm/test {"provider": 5}` → 500 `AttributeError: 'int' object has no attribute 'strip'`.
  Fix: Coerce with `str(data.get(...) or "")` before string methods, matching the adjacent block.
  Acceptance: Both requests return 400 with a structured `INVALID_INPUT` error and no crash-log entry.
  Confidence: Verified
  Effort: S

- [ ] P3 — Enable `B018` and triage the discarded expressions it finds
  Category: maintainability
  Where: Lint config in `pyproject.toml` (`ruff` `--select E,F,I` per `scripts/release_smoke.py`). Sites: `opencut/core/auto_dub_pipeline.py:577-578`, `noise_classify.py:243-244`, `broll_suggest.py:234`, `smart_defaults.py:169`, `subtitle_timing.py:174`, `clean_plate.py:159`, `glitch_effects.py:220-221`, `nd_filter_sim.py:61`, `plate_blur.py:112,297`, `power_windows.py:254-255`, `shot_classify.py:498`, `spatial_audio_vr.py:213-214`, `video_360.py:151,565`, `video_compare.py:247-248`, `routes/audio_expansion_routes.py:203`.
  Problem: ~21 statements compute a value and discard it. Three tiers: (1) a wasted `ffprobe` subprocess per request in `auto_dub_pipeline._composite_dubbed_audio:577` and `noise_classify._remove_noise_ffmpeg:243`, where the `get_video_info(...)` result is entirely unused (these are the only 2 truly dead call sites out of 215); (2) two probable dropped assignments — `broll_suggest.py:234` discards `clip.get("name","").lower()`, so clip filenames never enter `all_clip_terms` and B-roll cues cannot match on filename; `smart_defaults.py:169` computes the screen-recording codec check and throws it away, so the classifier silently uses only resolution and motion; (3) plain dead lines. The release lint gate selects only `E,F,I`, so `B018` (flake8-bugbear useless-expression) never runs and this whole class is invisible.
  Evidence: AST scan across `opencut/` for expression statements whose value is unused.
  Fix: Add `B018` to the ruff selection, then triage each hit individually — delete the dead ones, and decide whether `broll_suggest` and `smart_defaults` need the assignment restored (these two change behaviour, so they need a test either way).
  Acceptance: `ruff check opencut/ --select E,F,I,B018` is clean; B-roll matching on clip filename is covered by a test if that assignment is restored.
  Confidence: Verified (deadness); the two behaviour gaps are Likely-unintended and need a judgement call
  Effort: M

- [ ] P3 — Route the remaining drawtext call sites through the shared escaper
  Category: correctness
  Where: `opencut/core/watermark.py:150`, `kinetic_type.py:234,465`, `template_assembly.py:577`, `thumbnail_ab.py:357`, `adr_cueing.py:249-254`, `multicam_grid.py:156`, `camera_solver.py:698`, `character_consistency.py:691`, `data_animation.py:981`. Related `%`-escaping: `hook_generator.py:416-421`, `ab_variant.py:149-154`, `programmatic_video.py:94-97`. Helper: `opencut/helpers.py:447` (`escape_drawtext`).
  Problem: Each module carries its own partial escaper handling only `'` and sometimes `:`, never `\`, and none sets `expansion=none`. A Windows path or a backslash in user text breaks the filtergraph, and literal `%{...}` in user text is evaluated as a drawtext expression rather than printed. `adr_cueing.py` additionally embeds an unescaped `cue.cue_id`. The `%`→`%%` escaping in the last three modules is likely wrong under `expansion=normal` and would render a literal double `%%`.
  Evidence: Probed the bundled FFmpeg: `drawtext=text='C:\media'` fails with `Invalid argument`, and `%{eif:...}` is evaluated (exit 0) rather than printed. Reachable via user text on the owning routes (e.g. `/video/watermark`, kinetic-type routes at `opencut/routes/color_mam_routes.py:804-880`).
  Fix: Replace every local escaper with `helpers.escape_drawtext` paired with `expansion=none`, as that helper's docstring mandates. Then add a guard test that fails if any `drawtext=` construction in `opencut/core/` does not go through the helper.
  Acceptance: A parametrised test renders `C:\media\clip`, `Title: Part 2`, and `100%{x}` through each affected entry point, asserting exit 0 and literal output. The guard test fails if a new local escaper is introduced.
  Confidence: Verified (the breaking inputs are confirmed against the bundled binary; per-site reachability varies)
  Effort: M

- [ ] P3 — Use SMPTE drop-frame math in tc-sync
  Category: correctness
  Where: `opencut/core/tc_sync.py:57-69` (`_tc_to_frames`), `:251` (`compute_tc_offsets`), `:302` (`find_common_timecode_range`). Correct implementation already exists at `opencut/core/timecode_utils.py:260-286`.
  Problem: `_tc_to_frames` strips the `;` drop-frame separator and computes `hh*fps_int*3600 + …`, which overcounts 29.97 DF timecode by 2 frames per non-tenth minute (e.g. `01:00:00;02` → 108,002 rather than the correct 107,894). Two cameras striped with DF timecode starting minutes apart get relative offsets wrong by roughly 2 frames per minute of timecode delta — defeating the module's stated frame-accurate purpose. Separately, `compute_tc_offsets` and `find_common_timecode_range` apply `sources[0]`'s fps to every source's frame counts, so mixed 25/50 fps sets produce wrong `offset_seconds`.
  Evidence: The `;` separator is discarded before the arithmetic, and the arithmetic contains no drop-frame correction; `timecode_utils` has the correct algorithm and is not imported here.
  Fix: Delegate to `timecode_utils.timecode_to_frames` (honouring `;`) and convert each source with its own fps before comparing.
  Acceptance: A test with two 29.97 DF sources one hour apart asserts the computed offset is exact, and a mixed 25/50 fps pair asserts correct `offset_seconds`.
  Confidence: Verified
  Effort: M

- [ ] P3 — Consolidate the duplicate filter-path escaper and catch metric timeouts
  Category: correctness
  Where: `opencut/core/quality_metrics.py:106-115` (`_escape_filter_path`) versus `opencut/helpers.py:443` (`escape_filter_path`); and `opencut/core/quality_metrics.py:343-364` (`compare_videos` per-metric loop).
  Problem: Two defects in one module. (1) The local escaper handles `\`→`/` and `:` but not apostrophes, while the shared helper handles `'` with the close/reopen idiom — so VMAF breaks for any user whose profile path contains an apostrophe (the log path comes from `tempfile.mkstemp`, e.g. `C:\Users\O'Brien\AppData\Local\Temp\...`). It also duplicates a consolidated helper, against the repo's own convention. (2) The per-metric loop catches only `RuntimeError`, but `_run_ffmpeg_filter_complex` calls `_sp.run(..., timeout=timeout)`, which raises `subprocess.TimeoutExpired` — so one hung metric (VMAF on long media being the obvious case) aborts the entire report including metrics already measured, instead of degrading into `notes` like every other failure. The docstring promises per-metric isolation.
  Evidence: Both are direct reads of the cited lines; the helper's apostrophe handling is present and the local copy's is absent.
  Fix: Use `helpers.escape_filter_path` and delete the local copy; add `_sp.TimeoutExpired` to the per-metric `except` clause.
  Acceptance: A test with an apostrophe in the temp path measures VMAF successfully; a test where one metric times out still returns the other metrics with a note.
  Confidence: Verified
  Effort: S

- [ ] P3 — Scope the IMSC validator's log capture to ttconv
  Category: correctness
  Where: `opencut/core/standards_validators.py:160-206` (`_CollectingHandler` attachment in `validate_imsc`).
  Problem: The handler is added to both `logging.getLogger("ttconv")` and the root logger, but ttconv propagates to root by default — so every ttconv warning/error is captured twice and appears duplicated in `report.errors`/`report.warnings`. More seriously, any logger in the process that propagates to root (including `"opencut"`, which has handlers but default `propagate=True`, `opencut/server.py:79-84`) contributes records during the validation window, so an unrelated ERROR lands in `report.errors` and `report.passed = not report.errors` becomes a false failure. Currently only test and release-gate callers exercise it (mostly single-threaded), which is what keeps this at P3 — but it becomes a live flake the moment the validator is exposed as a route.
  Evidence: Both `addHandler` calls are present at the cited lines with no filter on `record.name`.
  Fix: Attach only to the `ttconv` logger, or keep the root attachment behind a filter on `record.name.startswith("ttconv")`, and de-duplicate findings before returning.
  Acceptance: A test that logs an unrelated ERROR to the `opencut` logger during validation asserts `report.passed` is unaffected and no duplicate findings are recorded.
  Confidence: Verified
  Effort: S

- [ ] P3 — Return 400, not 500, for malformed sequence-index payloads
  Category: reliability
  Where: `opencut/core/sequence_index.py:513-514` (`filter_rows`) and `:143` (frame conversion); route handler `opencut/routes/sequence_index_routes.py:200`, row rebuild at `:47-49`.
  Problem: Two crash shapes. (1) `filter_rows` calls `t.lower()` / `e.lower()` on `tags` and `effects` elements; `_dict_to_row` preserves non-string list elements, so a round-tripped row with `tags: [1]` plus a `query` raises `AttributeError`, which the route does not catch — a 500 where a 400 belongs. (2) Python's `json.loads` accepts `Infinity`, so an infinite `start`/`end`/`fps` reaches `int(round(seconds * fps))` and raises `OverflowError` → 500. (`NaN` yields a 400, but with the cryptic message "cannot convert float NaN to integer".)
  Evidence: The `.lower()` calls are unguarded and `_dict_to_row` performs no element coercion; `_safe_float` does not reject non-finite values.
  Fix: Coerce list elements to `str` in `_dict_to_row`/`build_index`, and make `_safe_float` reject non-finite values with a clear validation message.
  Acceptance: Both payloads return 400 with an actionable message; a test covers `tags: [1]` with a query and `start: Infinity`.
  Confidence: Verified
  Effort: S

- [ ] P3 — Purge terminal jobs by completion time, not creation time
  Category: correctness
  Where: `opencut/jobs.py:664-667` (`_cleanup_old_jobs`); compare the correct SQLite TTL at `opencut/job_store.py:453-455`.
  Problem: Terminal jobs are deleted from memory when `now - created > JOB_MAX_AGE` (1 h default), so a job that *ran* longer than an hour becomes eligible for purge on the first 5-minute tick after it completes. `/status/<job_id>` — the endpoint the CEP panel polls — then 404s ("Job not found") for a job that finished seconds ago. `/jobs/<job_id>` falls back to SQLite, but `/status` does not. The SQLite TTL already does this correctly with `COALESCE(completed_at, created_at)`.
  Evidence: The in-memory branch keys on `created` only.
  Fix: Use `now - (completed_at or created)` for the terminal-purge branch; leave stuck-job detection keyed on `created`.
  Acceptance: A test with a terminal job created 90 minutes ago but completed 1 minute ago asserts it survives the purge and `/status` still returns it.
  Confidence: Verified
  Effort: S

- [ ] P3 — Verify the port holder is OpenCut before force-killing it
  Category: reliability
  Where: `opencut/pid.py:197-231,238-274` (`_kill_via_netstat`, strategy 3); `_is_opencut_on_port` is defined and re-exported at `opencut/server.py:537` but never called in the kill path.
  Problem: Strategies 1 and 2 are correctly OpenCut-specific (its own endpoint, its own PID file). Strategy 3 runs `taskkill /F /T` against whatever PID is listening on the port — including an unrelated user application and its entire process tree. It fires exactly when the first two fail, which is precisely the case where the holder is *not* OpenCut. The aggressive startup behaviour appears intentional, but killing a foreign process tree is a real hazard on a workstation.
  Evidence: Call-graph check — `_is_opencut_on_port` has no callers in the kill path.
  Fix: Gate strategy 3 on `_is_opencut_on_port(host, port)`; if the holder is not OpenCut, skip straight to the alternate-port search that already exists.
  Acceptance: A test with a non-OpenCut listener on the port asserts no kill is attempted and the server binds an alternate port.
  Confidence: Verified
  Effort: S

- [ ] P3 — Hold the per-file lock across read-modify-write sequences
  Category: correctness
  Where: `opencut/user_data.py:373-379` (`create_user_tombstone`), `:517-528` (`save_assistant_dismissed`), plus route-level load→mutate→save such as `opencut/routes/workflow.py:243-265`.
  Problem: The per-file `RLock` makes each individual `read_user_file`/`write_user_file` atomic, but these helpers release it between the read and the write. Flask is threaded, so two concurrent requests interleave and one update is lost — e.g. two concurrent `/workflows/delete` calls each create a tombstone and one silently vanishes, breaking the reversibility guarantee that the destructive-confirmation flow advertises.
  Evidence: Each helper calls the read wrapper and the write wrapper as separate lock acquisitions.
  Fix: Expose a `with user_file_lock(filename):` context manager (the lock is already an `RLock`, so nesting is safe) and wrap the read-modify-write sequences.
  Acceptance: A concurrency test issuing two simultaneous tombstone-creating deletes asserts both tombstones exist.
  Confidence: Verified (code trace; requires concurrency to observe)
  Effort: M

- [ ] P3 — Let the queue runner wait as long as the job is allowed to run
  Category: correctness
  Where: `opencut/routes/jobs_routes.py:1043-1058` (the `_run` poll loop); related terminal-state check in `opencut/core/workflow.py` `_wait_for_job`.
  Problem: The runner polls a dispatched job for 1800 s then marks the entry `QUEUE_JOB_TIMEOUT`, but the job itself may run for 7200 s (`job_stuck_timeout`) and workflows wait 3600 s per step. The runner then starts the **next** entry while the "timed-out" job is still executing, so two heavy jobs run concurrently despite the queue's one-at-a-time design, and the user sees an error for a job that later completes successfully. Separately, `_wait_for_job` treats only `complete`/`error`/`cancelled` as terminal, so an `interrupted` sub-job spins the full timeout.
  Evidence: The 1800 s constant is independent of, and shorter than, both `_JOB_STUCK_TIMEOUT` and the workflow step budget.
  Fix: Poll until the job reaches a terminal state or `_JOB_STUCK_TIMEOUT` (which already guarantees termination) rather than an independent shorter deadline; add `interrupted` to the terminal set.
  Acceptance: A test with a job running longer than 1800 s asserts the queue does not start the next entry and does not report a timeout error.
  Confidence: Verified
  Effort: S

- [ ] P3 — Make the shutdown WAL checkpoint actually run
  Category: reliability
  Where: `opencut/job_store.py:159-182` (`close_all_connections`) and `:145-155`; same pattern in `opencut/journal.py:95-104,158-172`.
  Problem: Connections are created with sqlite3's default `check_same_thread=True`. `close_all_connections()` runs at exit on the main thread and calls `execute("PRAGMA wal_checkpoint(TRUNCATE)")` plus `close()` on connections created by `_io_pool`/worker threads — both raise `ProgrammingError` ("SQLite objects created in a thread can only be used in that same thread") and both are swallowed. Since nearly all `save_job` writes happen on `_io_pool` threads, the documented "checkpoints WAL before closing to avoid orphaned -wal/-shm files" never happens for the connections that matter. The same swallow hides failed closes in the dead-thread pruning paths. Impact is limited (process exit releases the handles) but the hygiene the code claims is not occurring — another check that always appears to pass.
  Evidence: Reproduced the `ProgrammingError` for a cross-thread `close()`/`PRAGMA` against these modules.
  Fix: Open connections with `check_same_thread=False` (each is already thread-confined by design), or have each pool thread close its own connection via an executor-shutdown hook.
  Acceptance: A test asserts that after `close_all_connections()` the `-wal` file is truncated and no exception was swallowed.
  Confidence: Verified
  Effort: M

- [ ] P3 — Wire or remove the versioned-config migration framework
  Category: maintainability
  Where: `opencut/user_data.py:161-257` (`CONFIG_SCHEMAS`, `register_config_schema`, `read_user_file_versioned`, `_MIGRATION_BACKUP_SUFFIX`).
  Problem: `CONFIG_SCHEMAS` has zero production registrations and `read_user_file_versioned` has zero production callers — the only caller is `tests/test_config_and_userdata.py:187-221`. All real reads go through plain `read_user_file`, so a schema migration registered tomorrow would never run in production while appearing to be supported. ~100 lines of framework verified only against itself.
  Evidence: Repo-wide call-site search returns only the test module.
  Fix: Either wire `read_user_file_versioned` into the `load_X()` wrappers (its evident purpose) or delete it with its tests. Given the repo already ships JSON schema migrations elsewhere, wiring is probably correct — but pick one.
  Acceptance: Either a production `load_X()` path is covered by a migration test, or the framework and its tests are gone and the suite still passes.
  Confidence: Verified
  Effort: S (delete) / M (wire in)

- [ ] P3 — Drop `noisereduce` from the declared dependencies
  Category: maintainability
  Where: `requirements.txt` (STANDARD section); `pyproject.toml` `standard` and `audio` extras; install hints at `opencut/core/dependency_support.py:83` and `opencut/routes/system_runtime_routes.py:371,397`.
  Problem: `noisereduce` is never imported anywhere in the repo — no static import, no `import_module`, no `reduce_noise` usage; it appears only in install-hint strings. Everyone installing `[standard]` or `[audio]` pulls the package plus its scipy chain for code that cannot use it. Other spot-checked dependencies (rich, waitress, psutil, keyring, python-json-logger, scenedetect) are all genuinely imported.
  Evidence: Repo-wide import scan finds zero usages.
  Fix: Remove it from the extras and `requirements.txt`, or keep it documented as plugin-only; update the two hint tables either way.
  Acceptance: A fresh `pip install -e ".[audio]"` does not pull `noisereduce`, and the dependency-support table no longer advertises it as a supported backend.
  Confidence: Verified
  Effort: S

- [ ] P3 — Deselect network/integration tests from the default run
  Category: testing
  Where: `tests/test_integration_whisper.py:16-20`; `[tool.pytest.ini_options]` in `pyproject.toml`.
  Problem: The `integration` and `slow` markers are declared but there is no `addopts` filter, so a plain `pytest` downloads a Whisper model over the network and runs real FFmpeg renders. The docstring says "Run manually" but nothing enforces it, and the only guard is a `skipif` on FFmpeg availability.
  Evidence: No `addopts` entry in the pytest config; the marker declarations exist without a default filter.
  Fix: Set `addopts = -m "not integration and not slow"` and document opting in with `-m integration`.
  Acceptance: A plain `pytest` run performs no network I/O; `pytest -m integration` still runs the suite.
  Confidence: Verified
  Effort: S

- [ ] P3 — Cover the workflow between-steps cancellation branch
  Category: testing
  Where: `opencut/core/workflow.py:218-227` (the `_is_cancelled(parent_job_id)` early exit).
  Problem: The branch that returns "Workflow cancelled by user" with partial `step_results` is exercised by no test — the only `parent_job_id` reference in the test suite is against a mocked `run_workflow` (`tests/test_workflow.py:146`), and the repo-root `.coverage` shows these lines unexecuted. Job-level cancellation is well covered (`tests/test_job_cancellation_race.py` is solid), but the workflow-chain contract — partial results and the `steps_completed` count — is unverified. This matters more once the propagation fix above lands.
  Evidence: Coverage data plus the absence of a non-mocked caller.
  Fix: Add a test that flips `_is_cancelled` after the first step and asserts the partial-result shape and `steps_completed`.
  Acceptance: The new test fails if the early-exit branch is removed.
  Confidence: Verified
  Effort: S

- [ ] P3 — Consolidate the duplicated route helpers
  Category: maintainability
  Where: `_json_object_or_400` defined five times — `opencut/routes/dev_scripting_routes.py:26`, `plugins.py:42`, `workflow.py:34`, `workflow_dev_routes.py:34`, `workflow_routes.py:20`; `_stub_503` defined three times — `wave_h_routes.py:73`, `wave_k_routes.py:26`, `wave_l_contract.py:10`.
  Problem: Copies have already drifted: `wave_h_routes._stub_503` has no default for `hint` while the others do, so behaviour depends on which module a route happens to live in. The repo's own "consolidated helpers" convention (CLAUDE.md) exists for exactly this.
  Evidence: Definition counts from a repo-wide search.
  Fix: Move both into `opencut/helpers.py` or a new `opencut/routes/_common.py` and import them; reconcile the `hint` default deliberately.
  Acceptance: Each helper is defined once; all callers import it; the suite passes.
  Confidence: Verified
  Effort: S

- [ ] P3 — Fix the UXP controls that silently do nothing
  Category: correctness
  Where: (a) `extension/com.opencut.uxp/main.js:4693,4701` (Auto Zoom aspect); (b) `:4589-4616` (Loudness Match); (c) `:6704-6708` (chat actions); (d) `:6386-6388` (OTIO export path fallback).
  Problem: Four controls mislead the user about what they do. (a) `zoomAspect` is read into `aspect` and never included in the request payload (`{ filepath, zoom_amount, easing }`), so the user's 9:16 / 1:1 choice has no effect on the output. (b) "Loudness Match" posts `{files: [clipPath, refPath], target_lufs: -14.0}`, so the backend batch-normalises *both* files to a fixed −14 LUFS: the reference's loudness is never measured and a pointless normalised copy of the reference is produced, while the UI ("Matching loudness to reference…", a required reference picker) promises reference-matching. (c) The chat flow toasts "Executing {count} action(s)…" but only counts `r.data.actions` — no dispatch follows. (d) `document.getElementById("clipPathCut")?.value?.trim() ?? document.getElementById("clipPathVideo")?.value?.trim()` uses `??`, but an empty Cut input yields `""`, which is not nullish, so the Video-tab fallback is dead code; meanwhile `updateTimelineReadiness` (`:2999-3001`) uses `||`, so the Export OTIO button *enables* when only `clipPathVideo` is set and then dead-ends on "Select a clip first."
  Evidence: Each is a direct read of the cited lines; the payload objects visibly omit the read values.
  Fix: (a) include `aspect` in the payload (and confirm the backend honours it, else remove the control); (b) either measure the reference first and use its LUFS as the target, or relabel to "Normalize to −14 LUFS" and drop the reference input; (c) wire the actions through the existing NLP apply path or change the copy to "N suggested action(s) — review in the result panel"; (d) change `??` to `||`.
  Acceptance: Each control is covered by a test asserting the request payload or dispatched action reflects the UI state; the OTIO button and its handler agree on which inputs count.
  Confidence: Verified
  Effort: M

- [ ] P3 — Order-guard the Sequence Index filter requests
  Category: correctness
  Where: `extension/com.opencut.uxp/main.js:8954-8981` (debounced filter), `:9061-9065` (facet/sort change handlers).
  Problem: The debounced search (200 ms) and the un-debounced facet/sort handlers each POST `/timeline/sequence-index/filter` with no in-flight cancellation or sequence token, so a slow earlier response landing after a fast later one overwrites `visibleRows` and re-renders headers and sort indicators with stale results. The payload also re-ships the full `rows` array on every debounced keystroke — acceptable at the 250-row page size, heavy on large sequences.
  Evidence: Neither handler tracks a request generation nor holds an `AbortController`.
  Fix: Keep a monotonically increasing request id and apply a response only if it is still the latest, or abort the previous request via `AbortController`.
  Acceptance: A test that resolves two filter responses out of order asserts only the later request's results are rendered.
  Confidence: Likely (race is structural; not reproduced against a live host)
  Effort: S

- [ ] P3 — Tidy the panel copy and the dead onboarding markup
  Category: ux
  Where: `extension/com.opencut.panel/client/locales/en.json`; dead markup at `extension/com.opencut.panel/client/index.html:4145-4207`; count semantics at `client/main.js:15157` versus `host/index.jsx:1850-1870`.
  Problem: Four small quality issues. (1) `audio.effects_desc`, `cut.full_desc`, and `video.style_desc` use a double space plus ASCII `--` where the rest of the file consistently uses an em dash. (2) Terminology is split between "backend" (78 strings) and "server" (13), sometimes within one flow — `conn.dot_disconnected` says "Server disconnected" while `conn.start_hint` says "Start the backend with Start-OpenCut.bat…". (3) The static first-run wizard body — three steps, a Quick Tip, a "Don't show again" checkbox, and an "Open Workspace" button, ~60 lines with 10 live i18n keys — is unreachable: `wizardCloseBtn` and `wizardDontShow` appear in no JS file, and the only consumer of `#wizardOverlay` is the server-backed onboarding, which wipes `card.innerHTML` first (`main.js:18038,18159`). (4) The "Applied {count} cuts" toast reports `r.applied`, which `ocApplySequenceCuts` increments per clip removed *per track*, so a 3-cut apply on a 1V/2A sequence reports "Applied 9 cuts".
  Evidence: String counts from `en.json`; repo-wide search finds no references to the two wizard control ids; the JSX increments inside the per-track loop.
  Fix: Normalise to the em dash; pick one user-facing term (the rest of the UI favours "backend") and apply it consistently; delete the static wizard body (keeping the overlay shell) or wire it as the offline fallback tour; report cuts applied rather than clip-removals, or relabel the toast.
  Acceptance: A lint/test pass asserts no `--` in `en.json` descriptions and a single term for the backend concept; the toast count matches the number of cuts requested.
  Confidence: Verified
  Effort: S

- [ ] P3 — Use function replacement for interpolated error text
  Category: correctness
  Where: pervasive in `extension/com.opencut.panel/client/main.js` — e.g. `:15153`, `:15236`, `:16027`.
  Problem: i18n interpolation uses `String.prototype.replace("{error}", text)` with a string replacement, so backend or FFmpeg error text containing `$&`, `$'`, `` $` ``, or `$$` is mangled by JavaScript's replacement-pattern expansion. Low probability individually, but stderr passthrough makes it reachable and the pattern is repeated widely.
  Evidence: The call sites pass a raw string as the replacement argument.
  Fix: Add one shared interpolation helper that uses a function replacement (`.replace("{error}", function () { return text; })`) and route these call sites through it — closing the whole class rather than the three cited lines.
  Acceptance: A test interpolating an error string containing `$&` and `` $` `` asserts the output is literal.
  Confidence: Likely
  Effort: M

- [ ] P3 — Gate the panel lint warnings
  Category: maintainability
  Where: `npm run lint` in `extension/com.opencut.panel/package.json`; warnings concentrated in `client/main.js`.
  Problem: The lint script exits 0 with 24 warnings (14 `no-redeclare`, 10 `no-unused-vars`), so the count can drift upward unnoticed. Most `no-redeclare` hits are idiomatic ES5 `var i` loop counters re-declared in the same function scope and are harmless — including `editDebounceTimer` at `:128` and `:8030`, which share one binding, so `cleanupTimers()` does clear it. The value here is preventing drift, not fixing the current hits.
  Evidence: `npx eslint client/main.js` lists the 14 `no-redeclare` sites; the two `editDebounceTimer` declarations are both at IIFE top level, so the second is a redundant no-op rather than a bug.
  Fix: Set a warning ceiling (`--max-warnings 24`) so the count cannot grow, then reduce it over time — the unused-vars hits are the ones worth clearing first.
  Acceptance: `npm run lint` fails if a new warning is introduced.
  Confidence: Verified
  Effort: S

- [ ] P3 — Close the UXP teardown asymmetry
  Category: maintainability
  Where: `extension/com.opencut.uxp/main.js:8237-8241` (`beforeunload`) versus `:6722-6852` (`uxpWsDisconnect`).
  Problem: The unload handler closes the SSE stream, theme sync, and the media-scan interval, but never calls `uxpWsDisconnect()`, so `_uxpWs` and `_uxpWsReconnectTimer` survive teardown. UXP host teardown usually reaps them, which keeps this low severity, but the cleanup is inconsistent with the SSE handling in the very same handler and will leak if the panel is reloaded rather than closed.
  Evidence: `uxpWsDisconnect` has no caller in the teardown path.
  Fix: Call `uxpWsDisconnect()` from the `beforeunload` handler alongside the existing cleanup.
  Acceptance: A test asserts the socket and reconnect timer are cleared on teardown.
  Confidence: Verified
  Effort: S

- [ ] P3 — Bring the UXP full-report flow under the single-job contract
  Category: maintainability
  Where: `extension/com.opencut.uxp/main.js:5671-5701` (`runFullReport`).
  Problem: It drives the global processing banner and progress bar through direct `BackendClient.post` loops without calling `markJobStarting`, so a real `JobPoller` job started concurrently (a Settings quick action, or a WebSocket progress event) contends for `progressFill` and `processingMsg`. Impact is low because the deliverables POSTs are fast, but this is the one flow that bypasses the single-job contract every other flow honours.
  Evidence: No `markJobStarting`/`state` interaction in the function.
  Fix: Acquire the job lock via the same controller path the other flows use, or use a scoped progress surface that does not share the global banner.
  Acceptance: A test starting a poller job during a full-report run asserts the banner reflects one owner.
  Confidence: Verified
  Effort: S

- [ ] P3 — Reconcile the queue allowlist with the documented invariant
  Category: docs
  Where: `opencut/routes/jobs_routes.py:179-398` (`_ALLOWED_QUEUE_ENDPOINTS`); the invariant is stated in `CLAUDE.md` Gotchas ("New async routes MUST be added to `_ALLOWED_QUEUE_ENDPOINTS`, or queue operations silently fail").
  Problem: Measured against the live app, 760 parameterless async POST routes exist and 547 of them are not queueable — whole families (`/qc/*`, `/export/gif|prores|dcp`, `/repair/*`, `/privacy/*`, `/rough-cut/*`, `/spectral/*`). Either the invariant is stale and the allowlist is deliberate curation (in which case the gotcha should say so), or this is accumulated omission and hundreds of routes silently return "Endpoint not queueable". Given entries were added wave-by-wave, omission looks more likely — but the intent needs an owner's decision, and right now the documentation and the code disagree.
  Evidence: Route-table enumeration against `_ALLOWED_QUEUE_ENDPOINTS` (the same script used for the sync-route finding above).
  Fix: Decide the intent, then encode it: if curation, rewrite the CLAUDE.md gotcha to say the allowlist is opt-in and explain the criteria; if omission, triage the 547 routes. Either way add the release-gate test that diffs async routes against the allowlist so the two cannot drift silently again.
  Acceptance: The documentation matches the code, and the drift test encodes whichever rule was chosen.
  Confidence: Verified (the numbers); the intent is a judgement call for the maintainer
  Effort: S (decide + test) / M (triage 547 routes)

- [ ] P3 — Unaudited areas needing their own pass
  Category: docs
  Where: repo-wide.
  Problem: This audit did not cover, and no finding above should be read as clearing: the installer (`installer/`, `OpenCut.iss`, `Install.ps1`) and its .NET build; Docker and the Linux packaging lane (`Dockerfile`, `packaging/linux/`, `io.github.sysadmindoc.opencut.yml`); the CLI surface (`opencut/cli.py`, ~1,781 lines) beyond the two commands touched incidentally; the ~130 core modules not sampled (the media pass prioritised FFmpeg-command builders, parsers, and timecode math); the plugin examples under `opencut/data/example_plugins/`; localisation completeness for `es.json` (only key *presence* is machine-checked, not translation quality); and any behaviour requiring a live Adobe Premiere host — every panel finding here was verified by code trace plus the headless rendered suite, never against Premiere itself.
  Evidence: Scope of this pass, recorded honestly.
  Fix: Schedule a pass per area, starting with the installer and Docker lanes since they gate distribution.
  Acceptance: Each listed area has had a recorded audit pass.
  Confidence: Verified
  Effort: L
