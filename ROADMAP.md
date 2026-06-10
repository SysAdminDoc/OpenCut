# Roadmap

Single task tracker for known issues and planned work. Items below come from
the June 2026 engineering/product audit (verified findings, with file
locations); fixes already shipped are recorded in CHANGELOG.md and git
history, not here.

## P1 — Release blocking

- [ ] P1 — Docker container crash-loops on startup
  Why: the image binds 0.0.0.0 but never sets OPENCUT_ALLOW_REMOTE=1, so server.main() refuses the bind and exits 2; restart:unless-stopped loops it forever and the healthcheck never passes.
  Where: Dockerfile (ENV/CMD), docker-compose.yml, opencut/server.py:841

- [ ] P1 — UXP: cancelling a job permanently disables the button that started it
  Why: JobPoller.cancel() closes SSE, nulls activeJobId, and never fires the job's onError/onComplete, so setButtonLoading(btn, false) never runs and every reconnect path skips .loading buttons — the Run button stays stuck until panel reload.
  Where: extension/com.opencut.uxp/main.js:1950-1957 (cancel), 1932-1936, 5732-5736, 5556-5560

- [ ] P1 — UXP: FCC caption display-settings card is non-functional
  Why: code reads schema.tokens / resp.preview_css off the BackendClient wrapper ({ok,data,status}) instead of resp.data — selects populate empty and preview never applies, while status reports success. Card exists for 47 CFR § 79.103 compliance (effective 2026-08-17).
  Where: extension/com.opencut.uxp/main.js:7649-7666, 7617-7625 (use responseData() like main.js:7184)

## P2 — Security / hardening

- [ ] P2 — CORS default allows null/file:// origins while /health hands out the CSRF token unauthenticated
  Why: a local HTML file or sandboxed null-origin context can read /health, obtain the CSRF token, and drive every mutating loopback route. Drop "null" from defaults or gate token issuance.
  Where: opencut/config.py:69,98, opencut/server.py:357, opencut/routes/system.py (/health)

- [ ] P2 — Werkzeug dev server used for the Docker/remote path
  Why: app.run() is not a production server; the Docker image's whole purpose is non-localhost exposure. Ship waitress (Windows) / gunicorn (Linux) for that path.
  Where: opencut/server.py:645, Dockerfile

- [ ] P2 — MCP HTTP transport has no authentication when bound non-loopback
  Why: --http-bind 0.0.0.0 exposes every tool (transcribe, pip-install, arbitrary routes) with only a printed warning; should require the opencut.auth token or refuse non-loopback binds.
  Where: opencut/mcp_server.py:1005-1096, 1124-1132

- [ ] P2 — ~/.opencut/packages is inserted at sys.path[0]
  Why: anything dropped there shadows stdlib modules and executes in the server process; frozen builds also trust whatever python is on PATH for site-packages. Append instead of insert-at-0; validate or gate the frozen-build path.
  Where: opencut/server.py:207-213, 157-202; opencut/security.py:439; opencut/helpers.py:24-26

- [ ] P2 — MCP extended route tools bypass the MCP-layer path validation
  Why: handle_tool_call dispatches opencut_route_* tools before the _validate_mcp_filepath block, so the documented fail-fast path checks never run for the opt-in extended surface (backend still validates).
  Where: opencut/mcp_server.py:767-778 vs 790-829, opencut/mcp_extended_tools.py:322-362

- [ ] P2 — No structural guarantee that every mutating route carries @require_csrf
  Why: CSRF is per-route across ~107 blueprints; today coverage is 100% but nothing enforces it for new routes. Add a startup assertion or test that every POST/PUT/PATCH/DELETE view is wrapped, or move to a before_request gate with an exempt list.
  Where: opencut/security.py:131-146, opencut/routes/__init__.py

- [ ] P2 — Remaining unbounded numeric inputs can OOM the render path
  Why: kinetic-text and delivery-master accept width/height/font_size/resolution with no max (e.g. width=999999999 reaches PIL/FFmpeg allocation); sibling routes in the same files clamp correctly.
  Where: opencut/routes/color_mam_routes.py:741-743, 759-761, 842-843, 879-880, 905-906, 937-938; opencut/routes/delivery_master_routes.py:139-140

- [ ] P2 — Auth middleware install failure is swallowed (fail-open)
  Why: if opencut.auth raises at app-build time the before_request token gate is silently skipped; should be fatal when OPENCUT_ALLOW_REMOTE=1.
  Where: opencut/server.py:372-400

## P2 — Correctness / reliability

- [ ] P2 — UXP: second concurrent job clobbers the single global job tracker
  Why: one global activeJobId/_activeSSE; starting job B halts job A's poll loop silently, A's promise never settles, its button stays loading, and the shared banner races. Disable all run buttons while a job is active, or track jobs per-id.
  Where: extension/com.opencut.uxp/main.js:1833-1840, 1848, 1963-1975

- [ ] P2 — UXP: backend port scan runs once at load; Refresh never rescans
  Why: if the backend starts later or restarts on a fallback port (5680+), checkConnection pings the stale port forever and the panel shows Offline with no recovery short of reopening it.
  Where: extension/com.opencut.uxp/main.js:7051, 5723, 481-491

- [ ] P2 — UXP: WebSocket reconnect loop cannot be stopped (including by its own Stop button)
  Why: onclose unconditionally schedules reconnect in 5s; disconnect() clears the timer before close fires, which schedules a fresh one. Needs a manual-disconnect flag plus capped backoff. Also ws://127.0.0.1:5680 is hardcoded, absent from manifest network domains, and collides with the HTTP fallback port range.
  Where: extension/com.opencut.uxp/main.js:6136, 6168-6188, 7086; extension/com.opencut.uxp/manifest.json:30-36

- [ ] P2 — UXP: one transient /status failure abandons a healthy running job
  Why: pollJob's error path surfaces "Polling error" and stops on the first dropped request; retry 2-3 consecutive failures before giving up.
  Where: extension/com.opencut.uxp/main.js:1893-1898

- [ ] P2 — UXP: background health check repaints the panel "offline" every 8 seconds
  Why: each cycle sets state "connecting" first, flipping the workspace card to the red offline variant and back — constant flicker on healthy connections. Keep prior state during background checks.
  Where: extension/com.opencut.uxp/main.js:5501, 2090-2107, 7130-7147

- [ ] P2 — CEP style.css is eight stacked re-skins, including two conflicting light themes
  Why: six full :root token redefinitions (12, 4462, 5386, 13215, 15466, 17214), two divergent html.theme-light blocks (16701 vs 17628), three different :focus-visible rules, and a triplicated prefers-reduced-motion block — the effective theme is "whatever the last pass overrode" and ~⅓ of 18k lines is dead weight. Consolidate to one token block per theme; then retokenize the ~340 stray hex literals.
  Where: extension/com.opencut.panel/client/style.css

- [ ] P2 — styled_captions frame-pipe encoder can deadlock on stderr
  Why: stderr is read only after all frames are written to stdin; if FFmpeg fills the stderr pipe buffer with warnings it stops reading stdin and both processes block. Drain stderr in a background thread like helpers.py does.
  Where: opencut/core/styled_captions.py:1199-1251

- [ ] P2 — Real-ESRGAN "balanced" upscale tier is dead on arrival
  Why: RealESRGANer(model_path=None) crashes in __init__ on every released package, and no pretrained weights are ever resolved; model_name is ignored. Resolve the weight file/URL and pass it.
  Where: opencut/core/upscale_pro.py:122-126

- [ ] P2 — helpers progress-runner pipes decode with the platform codec
  Why: helpers.py's _run_ffmpeg_with_progress Popen uses text=True without encoding=; UTF-8 filenames in FFmpeg output raise UnicodeDecodeError in the drain on cp1252 and can stall the loop (video_core's copy of this pattern is already fixed — port the same encoding="utf-8", errors="replace" here).
  Where: opencut/helpers.py:~733 (also audit other text=True Popen sites in core/)

- [ ] P2 — speed_up_silences treats MP3s with cover art as video
  Why: probe() falls back to the attached_pic stream so has_video is true for podcast MP3s; the video trim/concat chain then fails on a single-frame stream. Expose attached_pic / has_real_video and branch on it.
  Where: opencut/core/silence.py:591-593, opencut/utils/media.py:235-248

- [ ] P2 — Concat lists in ~17 feature modules still use unescaped, platform-codec writes
  Why: write_concat_list() now exists in helpers.py (UTF-8 + correct quote escaping) and the merge path uses it, but beat_cuts, ai_intro_gen, auto_dub_pipeline, photo_montage, instant_replay, cursor_zoom, event_recap, stream_highlights, video_360, generative_extend, auto_montage, glitch_effects, hook_generator, fit_to_fill, paper_edit, guest_compilation, stringout_reel still hand-roll open(path,"w") lists that break on apostrophes/non-ASCII names.
  Where: opencut/core/* (grep: "file '" writes); replace with helpers.write_concat_list

- [ ] P2 — Drawtext escaping is duplicated and wrong about apostrophes in ~10 modules
  Why: the local _escape_drawtext copies use \' inside single quotes (apostrophes silently dropped, graph can be mangled); helpers.escape_drawtext (ffmpeg-verified, expansion=none contract) should replace them.
  Where: opencut/core/caption_styles.py:365-372, click_overlay.py, news_ticker.py, quiz_overlay.py, telemetry_overlay.py, callout_gen.py, audiogram.py, brand_kit.py, end_screen.py, guest_compilation.py

## P3 — Lower-severity correctness, UX, packaging

- [ ] P3 — Inno installer writes HKCU/PlayerDebugMode in the elevated user's hive
  Why: with PrivilegesRequired=admin, HKCU and {%USERPROFILE} resolve to the elevating admin account, so the CEP panel-enable key can land for the wrong user and Premiere never loads the panel. Use runasoriginaluser or HKU of the invoking SID.
  Where: OpenCut.iss:30, 88-99, 154-173, 259-271

- [ ] P3 — OpenCut-Server.vbs builds cmd /c by unquoted concatenation
  Why: install paths containing &/^ break or inject; quote values or set env via WshShell.Environment like OpenCut-Launcher.vbs does.
  Where: OpenCut-Server.vbs:8-27

- [ ] P3 — auth.json gets no ACL hardening on Windows
  Why: chmod runs only under POSIX; the 256-bit bearer token is readable by other profile-access users on shared machines. Apply a restrictive DACL or document the limitation.
  Where: opencut/auth.py:80-85

- [ ] P3 — Installer fallbacks install unpinned dependencies
  Why: Install.ps1 and install.py pip-install loose specs while requirements.txt/requirements-lock.txt are properly bounded — supply-chain drift between install paths. Point installers at the lock/requirements file.
  Where: Install.ps1:384, install.py:78-83

- [ ] P3 — Placeholder GitHub URLs in user-facing output
  Why: cli.py and InstallerBuilder.ps1 reference github.com/opencut instead of github.com/SysAdminDoc/OpenCut.
  Where: opencut/cli.py:48, InstallerBuilder.ps1:582

- [ ] P3 — CLI: loudness-match writes to CWD for bare filenames; auto-zoom raises raw KeyError
  Why: os.path.dirname("clip.mp4") is "" → outputs land in the process CWD silently; auto-zoom indexes info['width'/'height'] directly while fps uses .get.
  Where: opencut/cli.py:1049, 1112-1117

- [ ] P3 — Export cancel kills FFmpeg without reaping before deleting the partial file
  Why: on Windows the output handle is briefly held after TerminateProcess, so the unlink silently fails and partials accumulate; kill(); wait(timeout) then unlink.
  Where: opencut/routes/video_core.py (cancel path), same pattern in trim/merge

- [ ] P3 — preview_frame returns success with an empty image on failure
  Why: ffmpeg return code is unchecked and mkstemp pre-creates the file the existence check then trusts; check returncode and size>0.
  Where: opencut/routes/video_core.py:~1470-1484

- [ ] P3 — Multiview/repurpose routes skip input validation on secondary paths
  Why: video_paths / content_path / reaction_path forwarded to core without validate_filepath, inconsistent with sibling modules.
  Where: opencut/routes/multiview_repurpose_routes.py:80-89, 131-142

- [ ] P3 — CEP: notification tone/heading inferred via English-only regexes on translated text
  Why: in any non-English locale every toast degrades to "info" (errors lose role=alert and assertive announcement). Pass explicit type from call sites.
  Where: extension/com.opencut.panel/client/main.js:8207-8246

- [ ] P3 — CEP: command-palette section labels and fallback descriptions bypass i18n
  Why: "Matching Tools" / "Recent" / "Favorites" etc. are hardcoded English in panel-utils while tool names localize — inconsistent.
  Where: extension/com.opencut.panel/client/panel-utils.js:63-83, 230-267

- [ ] P3 — CEP: language dropdown offers 10 locales but only en.json ships
  Why: every non-English choice falls back with a toast; filter options to locale files that exist, or ship the locales.
  Where: extension/com.opencut.panel/client/index.html:3615-3626, client/locales/

- [ ] P3 — CEP: job-history dedupe collapses distinct runs
  Why: server-history merge treats (type,status) as identity, dropping legitimate entries; compare job id or (type, createdAt).
  Where: extension/com.opencut.panel/client/main.js:~10160-10166

- [ ] P3 — CEP: time estimate parses duration by regexing rendered DOM text
  Why: fragile to format/locale changes; read numeric duration from clip state instead.
  Where: extension/com.opencut.panel/client/main.js:~12186-12194

- [ ] P3 — CEP: wizard and audio-preview dialogs bypass the overlay stack
  Why: aria-modal dialogs without focus trap or inert background (palette/preview modal do it right); route all four dialogs through activateOverlay/deactivateOverlay, give the context menu arrow-key support and focus restoration, and make one Escape close only the topmost surface.
  Where: extension/com.opencut.panel/client/main.js:11381-11527 (wizard/audioPreview/escape chain), index.html:4039, 4064

- [ ] P3 — CEP: body.job-active lock is pointer-events-only
  Why: keyboard activation still reaches "locked" buttons and AT announces them enabled; set disabled/aria-disabled or inert during jobs.
  Where: extension/com.opencut.panel/client/style.css:619-647, main.js startJob

- [ ] P3 — CEP: NLP command auto-executes the server-chosen route at confidence > 0.6
  Why: a misrouted natural-language command immediately starts a processing job; show parsed route/params with a confirm step or raise the bar.
  Where: extension/com.opencut.panel/client/main.js:~15321-15323

- [ ] P3 — CEP: --text-faint at 10-11px fails AA contrast; no prefers-contrast support
  Why: ~3.5:1 against card surfaces; bump small text to --text-muted as part of the style.css consolidation.
  Where: extension/com.opencut.panel/client/style.css (e.g. command-palette-hint ~3950)

- [ ] P3 — UXP: i18n gaps and locale hygiene from the recent es push
  Why: es.json has zero Spanish diacritics across 1,381 keys and a {plural} hack that breaks agreement; en.json has 113 duplicated keys (uxp.agent.runtime.* / uxp.captions.runtime.* pasted twice); toast headings/dismiss label/status-tone regexes/shortcut labels are hardcoded English; formatI18n's unescaped string replace corrupts values containing $& or $'. Add a locale lint (key uniqueness, placeholder parity) to the workflow.
  Where: extension/com.opencut.uxp/locales/en.json, es.json; main.js:238-244, 2061-2117, 5541-5548, 7551-7557

- [ ] P3 — UXP: small correctness fixes
  Why: project-item duration reads .seconds off a Promise (await precedence, main.js:849); addMarkers doesn't await getFirstMarkerAtTime so names/colors are silently dropped (682); Enter bypasses loading guards → duplicate submissions (5872-5914); ok-without-job_id responses are reported as failures in ~9 handlers; reconnect blanket-enables primary buttons regardless of per-tab prerequisites (5520-5527).
  Where: extension/com.opencut.uxp/main.js

- [ ] P3 — UXP: ship hygiene
  Why: udt-smoke.js (mutating test harness) loads in every production session — gate behind a debug flag; style.css stacks four :root themes with ~82 stray hex (gold-era gradients clash with the final blue accent); bolt-webview shim posts to targetOrigin "*" and reserves an exec passthrough.
  Where: extension/com.opencut.uxp/index.html:1653, style.css, csinterface-shim.js:51-54, 160-167

- [ ] P3 — Tests: function-scoped app fixture rebuilds the full app per test
  Why: ~100 blueprints + plugin load + background sweep threads per test, never torn down — slow suite and flake risk; use session/module scope or disable sweeps under TESTING. Also: coverage gate is 54%; pr-fast CI has no Windows runner despite Windows being the primary platform.
  Where: tests/conftest.py:12-34, .github/workflows/pr-fast.yml, build.yml

- [ ] P3 — captions timeout cleanup: deleting the temp WAV while a timed-out worker still holds it
  Why: after the timeout fix the finally-block unlink can hit PermissionError on Windows while the orphaned Whisper thread finishes; schedule deferred cleanup (helpers._schedule_temp_cleanup) for the timeout path.
  Where: opencut/core/captions.py (finally block)

- [ ] P3 — waveform_timeline materializes all PCM samples as a Python list
  Why: ~1 GB transient for an hour of audio; sibling audio.analyze_energy uses array for this reason.
  Where: opencut/core/waveform_timeline.py (generate_waveform_data)

- [ ] P3 — smart_trim swallows analysis timeouts as "no speech detected"
  Why: the 120s subprocess timeouts are caught by except Exception and return [], yielding confident wrong results on long files; surface them.
  Where: opencut/core/smart_trim.py:156, 222
