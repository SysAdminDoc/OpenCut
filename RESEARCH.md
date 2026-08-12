# Research — OpenCut

Date: 2026-08-11 — replaces all prior research (previous pass: 2026-08-10, v1.48.0).

## Executive Summary

OpenCut v1.48.0 is a local-first Premiere Pro automation backend: a loopback Flask service (1,593 routes,
1,568 shipped, 107 blueprints), a 19-command CLI, an 88-tool MCP server, CEP + UXP panels, a durable job
engine, SQLite/FTS5 + federated media index, FFmpeg 8.x pipelines, optional local AI adapters, and
OTIO/AAF/MLT interchange. The 2026-08-10 pass drained to four items; F303–F313 shipped in the fourteen
commits since. This pass therefore went at ground that pass did not touch: the two panels as a *system*,
the gap between the route surface and the user surface, the AI-clipping and agentic-editing markets, and
the correctness of the host write-back that the whole product depends on.

The strongest finding is that **the product's headline capability may be silently failing on current
Premiere and nothing in the tree would notice.** OpenCut writes cuts to the timeline through
`sequence.rippleDelete()` (UXP) and track-item deletion (CEP ExtendScript) and reports success from its
own loop counter — it never re-reads the sequence to confirm the edit landed. An independent Premiere
bridge with 184 stars measured, on Premiere 26.3, that `ripple_delete`, `razor`, and
`ComponentParam.setValue()` return success and mutate nothing, while additive operations still commit.
If that reproduces here, "cuts straight to your timeline" is currently a false claim on 26.x, and the
repo's own honest-claims discipline — a documented defect class across ~19 `fix:` commits — has no
instrument pointed at its most important assertion.

The second theme is that **the successor panel is 40% of the shipped one, and the deadline moved.**
The CEP panel calls 202 backend paths (124 real routes the UXP panel never calls); UXP calls 80.
`extension/PANEL_PARITY.json` records `$adobe_cep_eol: "approximately 2026-09"` (last updated
2026-05-25) and uses it to justify freezing two CEP tabs. Adobe's own current statement puts removal at
roughly **November 2026**, a year after Premiere 25.6. The parity test guards tab-level divergence only;
the 124-route feature gap inside shared tabs is unguarded.

Top opportunities, priority order:

1. **P0 — Prove the timeline write-back actually mutated the sequence.** Read state back through a
   different call than the one that wrote it and fail loudly when nothing changed.
   `extension/com.opencut.uxp/main.js:1005`, `extension/com.opencut.panel/host/index.jsx:1827`; zero
   read-back verification exists anywhere in the tree (Verified in-tree; the 26.3 no-op needs live
   validation).
2. **P1 — Measure and gate CEP→UXP feature parity, not just tabs.** 124 CEP-only real routes vs 19
   UXP-only; `tests/test_panel_tab_parity.py` asserts only that tab divergence is annotated.
3. **P1 — Correct the CEP end-of-life date and re-derive the investment policy from it.** Adobe:
   "support both CEP and UXP for a calendar year, after which we will remove support for CEP
   extensibilty" (Nov 2025, PPro 25.6) → ~2026-11, not the ledger's 2026-09.
4. **P1 — Stop classifying five hardcoded-501 handlers as `dependency-gated`.**
   `opencut/routes/wave_h_routes.py:491,522,540,558,582` return 503 when the dependency is missing and
   501 when it is present; `opencut/tools/dump_route_manifest.py:62-65` lets the `_stub_503(` marker win,
   so installing the dependency never makes them work and they inflate the 1,568 shipped count.
5. **P1 — Feed the project glossary into the ASR decoder instead of correcting after the fact.**
   `opencut/core/captions.py:930,1227,1241,1353` pass no `initial_prompt`/`hotwords`;
   `_apply_project_glossary` (`:740`) is post-hoc find/replace. Every commercial competitor sells this as
   "custom vocabulary".
6. **P1 — Unify the two panels' i18n namespaces.** CEP `en.json` has 2,868 keys, UXP 1,927, and **26 are
   shared**. Spanish exists only for the UXP panel — the panel no installer ships.
7. **P2 — Scope silence detection to in/out points and add a tighten-don't-delete mode.**
   `detect_silences()` takes no range; the most-voted open Premiere idea in this area asks for exactly
   this and Adobe's answer was "use Text-Based Editing", which the requester rejected.
8. **P2 — Guard long-file ASR repetition loops.** No `compression_ratio_threshold`, `no_speech_threshold`,
   or repetition detection anywhere; the failure is well documented in the FFmpeg-Whisper community.
9. **P2 — Record upstream maintenance status per engine and stop auto-installing abandoned packages.**
   `opencut/core/audio_pro.py:529` pip-installs `deepfilternet`, whose newest PyPI release is 2023-08-31
   and whose repository has not been pushed since 2024-10-17.
10. **P2 — Stop the direct-surface ratio falling further.** The repo's own manifest reports 280 of 1,568
    shipped routes reachable from any first-party surface (17.9%), 1,288 integration-only, and
    `primary_counts.cli = 0`.

Two dependency findings arrived after the above was drafted and rank alongside items 4–6: the per-CVE
FFmpeg matrix grades 4 of ~16 advisories from the same 2026-07 disclosure batch, and `urllib3>=2.6.3` — a
floor the project chose for security reasons — sits below two High-severity advisories fixed in 2.7.0.
Both are detailed under Security below and queued as F332 and F334.

Confidence labels: **Verified** = confirmed in this tree or against a primary source during this pass;
**Likely** = strong multi-signal inference still needing implementation validation;
**Needs live validation** = cannot be closed headlessly.

## Product Map

### Core workflows
- **Media-to-cut** — analyse (silence/filler/beat/scene/transcript/OCR), review the proposal, write cuts,
  markers, captions, or media changes back into Premiere through CEP or UXP.
- **Search-to-edit** — index transcript, OCR, face, and visual metadata across configured roots; turn a
  hit into a reviewable proposal or a timeline operation.
- **Caption and delivery** — transcribe, correct, style, synchronise, standards-check (TTML/IMSC 1.3/
  EBU-TT-D), render, and export to subtitle, caption, or interchange formats.
- **Review and approval** — long operations create durable jobs and versioned review artifacts with
  progress, cancellation, redaction, and explicit approval before host-facing or destructive actions.
- **Automation** — one loopback service reached from CLI, REST, MCP (Tasks extension), panels, plugins.

### Personas
Solo editors and small post teams wanting fast local rough cuts without uploading footage; podcast /
education / social producers needing transcript-driven edits and captions; caption and delivery operators
needing deterministic exports and audit trails; technical users and agents orchestrating scriptable,
cancellable operations without arbitrary code execution.

### Platforms and distribution
Python 3.11–3.14. Windows: PyInstaller + WPF installer + Inno Setup, bundled pinned FFmpeg. Linux:
AppImage/Flatpak metadata. macOS: source lane. Premiere integration is CEP (primary — shipped by every
installer, Adobe removal ~2026-11) plus UXP (strategic, PPRO minVersion 25.6). Newest published artifact
is v1.25.1 (2026-04-20) against a 1.48.0 tree; that gap is tracked in `Roadmap_Blocked.md` and gates every
downstream channel.

### Key integrations and data flows
Panels/CLI/MCP submit validated commands to Flask routes; long work runs on bounded workers that persist
job state and write artifacts under `~/.opencut/`. CEP and UXP adapters translate approved operations into
Premiere calls — Python never calls ExtendScript directly. `network_policy.py` installs an
`sys.addaudithook` egress guard with an AST module inventory.

### Scope boundaries
Not a mobile editor, cloud collaboration service, multi-user MAM, credit-metered service, or a replacement
for Premiere's timeline UI.

## Competitive Landscape

### Opus Clip / Vizard / Submagic / Klap (AI clipping SaaS — not previously surveyed)
Opus decomposes its virality score into Hook/Flow/Value/Trend, 0–99, and sorts results by it *by default*
— then paywalls the score itself. Its 2026 changelog adds non-speech clip triggers (visual object, sound,
emotion), transcript-placed generative B-roll, one-click SFX detection+generation, bad-take removal shown
inline before it removes anything, and an MCP connector (Business tier). Submagic meters videos, clip
length, *and* credits simultaneously and sells custom ASR vocabulary at Business tier.
**Learn:** rank-and-explain, show-before-you-remove, and named style catalogues are cheap UX patterns
OpenCut can adopt with compute it already spends — its CEP panel already does the ranked virality view
with sub-scores and weights, which is ahead of the field and unmarketed.
**Avoid:** every one of these products meters something. A local tool that invents an internal cost unit
imports the worst artifact of the SaaS model for zero benefit; show wall-clock, VRAM, and queue depth.
Also avoid their generative-marketplace lane — Descript's own changelog records an avatar model being
discontinued out from under users.

### Descript
The only competitor that owns **seam repair**: `Regenerate Speech` / `Video Regenerate` re-synthesise
across an edit boundary so a jump cut disappears. Also shipped a public API (2026-04) and an MCP server
(2026-06), and publicly swapped its default ASR vendor twice in 2026.
**Learn:** the end of a cutting pipeline is repairing what the cut broke — OpenCut has `core/morph_cut.py`
but does not present it as the completion of the silence/filler/take-removal flow.
**Avoid:** dual metering (media minutes *and* AI credits).

### premiere-pro-mcp (184★) and the Premiere-MCP cohort
Its closed-issue list is a free defect catalogue for anyone bridging Premiere, and issue #21 (2026-07-21)
is the most important external finding of this pass: on 26.3, restructuring operations on already-
materialised timeline state return success and do nothing, while additive operations commit. Open issue
#77 is an unanswered request for transcript-based editing — the thing OpenCut already ships, in front of
an audience 5× this repo's.
**Learn:** the bug is a class, not an instance; verify by reading back through a different call.
**Avoid:** competing on advertised tool count (279 / 282 / 1,027 across the cohort).

### auto-editor (4.8k★) and the 2026 agentic-editing wave
`browser-use/video-use` (20.5k★), `FireRed-OpenStoryline` (3.2k★), `HKUDS/VideoAgent` (1.7k★),
`OpenChatCut`, `veedstudio/open-edit`, `kinocut` — all pushed within weeks, all local-or-agent-first.
**Learn:** `video-use`'s text-first strategy (a ~12 KB transcript instead of frames, pixels sampled only
at decision points) is the cheapest known way to make long footage tractable for an LLM.
**Avoid:** their output model. Nearly all of them render a finished MP4 and discard the timeline.
OpenCut's round-trippable, editable Premiere output is the one axis none of them contest.

### LosslessCut / Subtitle Edit / OpenTimelineIO
LosslessCut #126 ("smart cut", 157 reactions, open since 2018) remains the most-demanded unbuilt OSS
video feature — and OpenCut already has it in `core/smart_render.py`, undermarketed.
**Learn:** OTIO is the load-bearing dependency of the differentiator and it is fragile: 0.18.1 (2025-11-09)
is still flagged prerelease with no release in 9 months, `otio-aaf-adapter` is a 23★ out-of-core plugin,
and OTIO's own help-wanted backlog is exactly what breaks Premiere handoffs — integer/integer rates
(#190), A/V clip linking (#343), markers in CMX3600 (#169), and no schema for keyframe curves (#331).
**Avoid:** building a second standalone trimmer or subtitle editor.

### Adobe
Premiere absorbed generic media search and scene detection; the durable community pain is elsewhere and
repeats across years: filler-word detection returning nothing on obvious ums (32-reply thread, Adobe's own
answer was that the model cannot run on macOS 15), 26.x transcription emitting nonsense repeats severe
enough that users downgraded, a caption change forcing a full timeline re-render, silence detection that
ignores in/out points (Adobe's staff answer: use Text-Based Editing; the requester rejected it), and a
2023 idea still open whose Adobe answer was literally "use an FFmpeg script".
**Learn:** those five are OpenCut's pitch, verbatim, and four of the five are already built.
**Avoid:** chasing first-party parity on generative features.

## Security, Privacy, and Reliability

### New findings this pass

- **Timeline write-back is unverified — Verified in-tree, needs live validation, critical impact.**
  `extension/com.opencut.uxp/main.js:1005` awaits `seq.rippleDelete(startTick, endTick)` and treats a
  non-throwing call as success. `extension/com.opencut.panel/host/index.jsx:1827` (`ocApplySequenceCuts`)
  returns `{applied, errors}` where `applied` is incremented by its own loop, not re-read from the
  sequence. A repo-wide grep for read-back verification (`verifyWrite|readBack|confirmApplied|
  assertApplied`) returns nothing. premiere-pro-mcp #21 measured this exact API returning success and
  mutating nothing on 26.3. Data-safety consequence: a user who sees "42 cuts applied" and saves has no
  signal that the sequence is unchanged.
- **Five routes advertise a dependency gate they cannot pass — Verified.**
  `/video/upscale/flashvsr`, `/video/inpaint/rose`, `/video/matte/sammie`, `/audio/tts/omnivoice`,
  `/video/style/reezsynth` return `_stub_503` when `check_X_available()` is False and a hardcoded
  `error_response("NOT_IMPLEMENTED", …, 501)` when it is True. `_DEPENDENCY_MARKERS` in
  `opencut/tools/dump_route_manifest.py:62-65` matches `_stub_503(` first, so the manifest labels them
  `dependency-gated` — a class its own comment defines as "fully implemented but require an optional
  dependency" — and counts them as shipped. The same file's comments already state that handlers
  delegating to a terminal `NotImplementedError` adapter *are* stubs; the classifier does not implement
  that rule for inline 501s.
- **The CEP deprecation date driving investment policy is wrong and stale — Verified.**
  `extension/PANEL_PARITY.json` (`$updated: 2026-05-25`) asserts `$adobe_cep_eol: "approximately
  2026-09"` and uses it to justify "do not invest further" on the CEP `export` and `nlp` tabs. Adobe's
  PProPanel ReadMe (Last Updated November 2025, Premiere Pro 25.6) states: "As of Premiere Pro 25.6, CEP
  extensions to Premiere Pro have been superseded by UXP Extensibility" and "the plan is to support both
  CEP and UXP for a calendar year, after which we will remove support for CEP extensibilty" — roughly
  2026-11. No firmer date exists publicly; a direct developer question on Adobe's forum (2026-01-08) is
  still unanswered.
- **A runtime pip path installs an abandoned package — Verified.** `opencut/core/audio_pro.py:529` calls
  `ensure_package("df", "deepfilternet")`. DeepFilterNet's newest PyPI release is 0.5.6 (2023-08-31) and
  its repository has not been pushed since 2024-10-17. `core/engine_registry.py` carries no
  upstream-maintenance field and `model_cards.py` has no DeepFilterNet card, so nothing in the product
  can tell a user they are installing dead software.
- **The default ASR wrapper is stalled — Verified, medium risk.** `faster-whisper>=1.1,<2` is a required
  dependency in four places in `pyproject.toml`; upstream's newest release is 1.2.1 (2025-10-31) with 314
  open issues and no commit since 2025-11-19. The CTranslate2 engine underneath is healthy (4.8.1,
  2026-08-05), and `asr_router.py` already fronts Parakeet/Canary/Moonshine/NeMo adapters — so this is a
  wrapper bus-factor risk, not an engine risk, and the mitigation is routing policy plus the blocked
  FFmpeg-whisper lane (F307), not a rewrite.
- **ASR has no long-file repetition guard — Verified.** No `compression_ratio_threshold`,
  `no_speech_threshold`, `condition_on_previous_text`, or post-hoc repetition detection appears in
  `opencut/core/captions.py`. Whisper's degradation on hour-plus audio (looping a single phrase for the
  remainder of the file) is a well-documented failure that produces a plausible-looking, silently wrong
  transcript — the worst shape for a tool whose next step is deleting footage based on it.
- **The per-CVE FFmpeg matrix grades a quarter of its own batch — Verified.** F304 replaced the snapshot
  date heuristic with fix-commit ancestry grading, but `opencut/core/ffmpeg_provenance.py` names only
  CVE-2026-64832, -64833, -64835, and -66041 from a disclosure batch of roughly sixteen High-severity
  FFmpeg advisories published 2026-07-22→24 against "FFmpeg through 8.1.2". Twelve appear nowhere in the
  repository, including CVE-2026-64831 (Vulkan HEVC hardware decoder, stack overflow), CVE-2026-64830
  (VobSub demuxer heap overflow), CVE-2026-66040 (native PNG/APNG encoder heap OOB write), and
  CVE-2026-66036 (`vf_hqdn3d` OOB write) — all in codepaths this product drives directly. The pinned
  snapshot may well contain every fix; the defect is that the gate reports a clear per-CVE verdict from
  partial coverage, which is the same honesty failure the readiness system exists to prevent.
- **The FFmpeg release lane is closed against a version that never shipped — Verified.**
  `ffmpeg_provenance.py:56-57` sets `RELEASE_FLOOR = (8, 1, 3)` and comments that the lane "remains closed
  until upstream publishes 8.1.3". Upstream released **9.0 "Lei" on 2026-08-04** instead; 8.1.2
  (2026-06-17) closed the 8.1 line. Every source install is therefore steered to a dated git-master
  snapshot. Note for whoever opens the lane: the 9.0 branch was cut from master on 2026-06-26, *before*
  the July fix commits landed, so a 2026-08-04 release date is not evidence the fixes are present —
  it needs the same ancestry check the matrix applies to the snapshot.
- **The urllib3 floor is itself vulnerable — Verified.** `pyproject.toml:86` pins `urllib3>=2.6.3` with an
  inline rationale citing CVE-2026-21441. Two further High-severity advisories published 2026-05-11 are
  fixed only in 2.7.0: CVE-2026-44431 (sensitive headers forwarded across origins on proxied low-level
  redirects, `>=1.23, <2.7.0`) and CVE-2026-44432 (decompression-bomb bypass in parts of the streaming
  API, `>=2.6.0, <2.7.0`). Confirmed against the GitHub Advisory API this pass. Pillow (`>=12.3.0,<13`)
  and Werkzeug (`>=3.1.6`) floors were checked in the same pass and are correct for their 2026 advisories.
- **Panel error swallowing persists in the CEP monolith — Verified.** `backend-client.js` is now clean,
  but `extension/com.opencut.panel/client/main.js` still holds 41 empty `catch (e) {}` blocks, and
  `opencut/` holds 226 `except Exception: pass` sites. This is why the single open external bug arrived
  with an empty logs section.

### Positive controls to preserve
CSRF on all mutating routes with an opaque-origin bootstrap channel, trusted-host/DNS-rebinding gate,
loopback-only default, SSRF and path validation, `addaudithook` egress guard with AST module inventory,
plugin trust/isolation, redacted job payloads, bounded workers, durable job journals, ZIP-slip defences,
C2PA provenance, per-CVE FFmpeg acceptance grading, an FTS5 memory-safety floor, and rendered WCAG 2.2 AA
scans across tabs and breakpoints with no suppressions. Zero TODO/FIXME/HACK markers in any of the three
source trees — incomplete work is expressed through a formal readiness registry and generated manifests
instead, which is why finding #2 above matters: the manifest is the product's memory.

### Known external blockers excluded here
macOS notarization, the live Premiere host lane, release publication, the OpenCV/Transformers
dependency-stack decision, the FFmpeg-whisper model acquisition (F307), the Flathub policy decision
(F310), localization human review, PyPI/Homebrew/winget publication, and the queue-allowlist intent
decision all remain in `Roadmap_Blocked.md`.

## Architecture Assessment

### Strengths
The readiness system, generated manifests that fail closed, `core/stub_scan.py`, and the surface-coverage
gate give this repo unusually honest self-reporting for 575k lines. `smart_render.py`, `morph_cut.py`,
`brand_kit.py`, `seo_optimizer.py`, `split_screen.py`, `auto_dub_pipeline.py`, `asr_parakeet.py`, and
`semantic_video_search.py` mean most of what the commercial field paywalls is already built — the deficit
is surfacing and marketing, not capability.

### Main seams
1. **Host-truth seam.** Every host mutation is fire-and-forget. There is no read-back, no post-condition,
   and no differential check, so the product cannot distinguish "applied" from "silently ignored". This is
   the highest-severity gap in the tree and it sits under the headline feature.
2. **Two-panel seam — now the dominant cost centre.** Panel churn leads all code: `panel/client/main.js`
   77 commits and `uxp/main.js` 70 in the last 400, `index.html` 59/50, `en.json` 39/39 — a lockstep
   pattern that only duplicated logic produces. The two panels share 26 of 4,795 i18n keys, ship
   `command-center.css` as two unrelated files under the same name and cascade position, and diverge by
   124 backend routes. `studio-workbench-v2.css`/`.js` are byte-identical copies with no generator or
   drift gate. Extraction has reached controllers but not a shared core.
3. **Surface seam.** 1,568 shipped routes; 280 reachable from a panel, palette, CLI, or MCP tool (17.9%);
   1,288 integration-only; 19 CLI commands; 88 MCP tools; **zero** routes whose primary surface is the
   CLI. Every wave adds routes faster than surfaces, and the ratio is measured but not ratcheted.
4. **Interchange seam.** The differentiator rests on OTIO (`>=0.17,<1`, currently 0.18.1 and still
   prerelease after 9 months) and on `otio-aaf-adapter` (23★). The unbounded `<1` ceiling will admit 0.19,
   which moves `otioz`/`otiod` from Python into the C++ core — a behaviour change to a shipped export
   path. `export/otio_compat.py` reports the runtime version but nothing pins or contract-tests it.
5. **Delivery seam.** Unchanged and still the largest gap between what the project does and what anyone
   can install: newest published artifact v1.25.1 (2026-04-20) against 1.48.0.

### Test and documentation gaps
- No test asserts that a host mutation changed the sequence; `tests/test_panel_tab_parity.py` asserts only
  that *tab*-level divergence is annotated, leaving the 124-route feature gap unguarded.
- `pyproject.toml:267-269` declares `[tool.ruff]` with no `lint` section, so an editor-integrated ruff
  applies its default rule set while the gate applies `--select E,F,I --ignore E501,E402` from
  `.pre-commit-config.yaml` and `scripts/release_smoke.py`. `[tool.pytest.ini_options]` sets no
  `testpaths` and no `--strict-markers` despite 10,751 tests across 346 files.
- The pre-push hook runs 2 of 346 test modules; everything else depends on a developer remembering
  `release_smoke.py`. There is deliberately no CI, which makes the local gate's breadth the only control.
- `feature_readiness.json` was generated 2026-08-03 while `route_manifest.json` regenerates on every route
  change (2026-08-11) — the two manifests that jointly define "what works" drift on different clocks.
- Neither `README.md` nor `docs/` mentions that Premiere 2026 lists CEP panels under **Extensions
  (Legacy)**; that one sentence pre-empts the most predictable support question, which has already burned
  two comparable projects publicly.
- `CLAUDE.md` states `installer/bin|obj|publish` are tracked; `git ls-files` shows zero tracked files
  under them.

### Operating constraints
The single-user loopback model, optional-dependency policy, no-code-signing rule, no-telemetry default,
and the absence of hosted CI are coherent and should not be traded away. Any recommendation requiring
signing, cloud inference, metered credits, or multi-user state contradicts them and is rejected below.

## Rejected Ideas

- **Build GOP-aware smart cut (LosslessCut #126, 157 reactions).** Already present:
  `opencut/core/smart_render.py` with keyframe probing and snap-to-keyframe. Source: OSS issue mining.
- **Add Parakeet / Canary / Moonshine / NeMo ASR.** Already present: `core/asr_parakeet.py`,
  `asr_canary.py`, `asr_moonshine.py`, `asr_nemo.py`, `asr_nemo_models.py`, routed by `asr_router.py`.
  Source: Open ASR Leaderboard research.
- **Add seam repair / jump-cut smoothing (Descript's differentiator).** Already present:
  `core/morph_cut.py`. The gap is presentation, folded into the surface work, not a new module.
- **Add a brand kit, SEO/hashtag/title generation, or split-screen layout reframe.** Already present:
  `core/brand_kit.py`, `core/seo_optimizer.py`, `core/social_captions.py`, `core/platform_publish.py`,
  `core/split_screen.py`, `core/ai_reframe_multi.py`. Source: AI-clipping SaaS survey.
- **Add voice-cloned dubbing.** Already present: `core/auto_dub_pipeline.py`, `dub_pipeline.py`,
  `ai_dubbing.py` with translate + voice-reference + TTS staging. Source: Opus/Descript changelogs.
- **Add face/object signals to the media index (edit-mind's axis).** Already present:
  `core/face_tagging.py`, `ocr_extract.py`, `semantic_video_search.py` (CLIP), `federated_media_index.py`.
- **Migrate off faster-whisper.** The stall is in the Python wrapper, not the CTranslate2 engine, and
  `asr_router.py` already fronts four alternative adapters. Recorded above as a supply-chain risk; a
  rewrite would trade a known-good pinned wrapper for churn.
- **Add hosted CI to run the 10,751 tests.** The repo states hosted workflow files are intentionally
  absent and verification runs locally. The actionable form is widening the local gate, not adding CI.
- **Build a `.prproj` / autosave corruption salvage tool.** Real, repeated, and entirely unclaimed
  community pain — but OpenCut never reads Premiere project files; the panel is its only host channel, and
  reverse-engineering a proprietary container would add a permanent maintenance surface outside the
  established trust and data model. Source: Adobe forum bug 1331881.
- **Adopt an agent-native JSON timeline convention (FableCut / OpenChatCut / kinocut).** Every
  implementation is per-tool and non-interoperable; OTIO plus MCP already covers the same ground with
  round-trip fidelity, which is the moat. Source: 2026 agentic-editing survey.
- **Introduce credits, points, cloud sync, hosted media libraries, or a generative-model marketplace.**
  Each contradicts the local-first, no-metering model; the marketplace lane additionally fails whenever a
  vendor deprecates a model, which Descript's own changelog documents.
- **Generic accessibility or i18n overhaul.** Rendered WCAG 2.2 AA scans already run with no suppressions;
  the specific, actionable defect is the 26-key panel namespace overlap, raised as its own item.
- **Rewrite the CEP panel monolith or force a UXP cutover now.** Adobe's removal is ~2026-11 and the
  cutover is gated on the blocked live-host lane. Measure and close parity along the existing seam first.
- **Extend the plugin ecosystem.** Loader, manifest validation, trust isolation, authoring docs,
  marketplace client, and two example plugins all ship; no unmet plugin-author need surfaced in this pass
  either, so there is still nothing to build against. Source: 2026 agentic-editing and OSS survey.
- **Add mobile, multi-user, or upgrade/migration mechanisms.** Mobile and multi-user contradict the
  local-first threat and storage model; installer upgrade recovery with snapshot-and-restore, uninstall
  cleanup, and local DB migrations already ship, and the only real upgrade gap is that no current build is
  obtainable — which is the blocked release-publication item, not a missing mechanism.
- **Add an offline mode.** Local-by-default is the existing posture with `OPENCUT_LOCAL_ONLY` and an
  audithook egress guard; the optional-model install paths are the only network dependency and they are
  already explicit. Source: repository evidence.
- **Migrate the MCP server to the 2026-07-28 specification.** Already done: `opencut/mcp_server.py:1785`
  declares the stateless 2026-07-28 revision, implements `server/discover`, stamps `resultType` on every
  result, and serves `tasks/get|update|cancel` — ahead of the deprecation clock on Roots, Sampling, and
  Logging. Source: MCP specification changelog.
- **Add a Python 3.14 free-threaded build.** OpenCut's parallelism is already in FFmpeg subprocesses and
  GIL-releasing C extensions (CTranslate2, ONNX Runtime, torch), so free-threading buys almost nothing
  while costing 5–10% single-thread throughput and requiring every native wheel to be re-qualified; torch
  ships 3.14t on Linux only and PyInstaller makes no free-threaded claim. Source: PEP 779, torch 2.13
  release notes, PyInstaller CHANGES.
- **Adopt WebGPU in the CEP panel.** CEP 12 runs Chromium 99 and WebGPU shipped at milestone 113, so it is
  unavailable there; the UXP panel is the only surface where it could land, and that work is gated behind
  the blocked live-host lane. WebCodecs (milestone 94) *is* reachable in CEP 12 and is noted as a future
  option for in-panel scrub decode, not proposed as work. Source: Chrome platform status.

## Sources

### Repository evidence
- `extension/com.opencut.uxp/main.js:1005`; `extension/com.opencut.panel/host/index.jsx:1827`
- `extension/PANEL_PARITY.json`; `tests/test_panel_tab_parity.py`; `opencut/_generated/cep_uxp_parity.json`
- `opencut/routes/wave_h_routes.py:491,522,540,558,582`; `opencut/tools/dump_route_manifest.py:55-75`
- `opencut/_generated/route_manifest.json` (`surface_coverage`, `readiness_counts`)
- `opencut/core/captions.py:740,930,1227,1241,1353`; `opencut/core/silence.py:37,428`
- `opencut/core/audio_pro.py:529`; `opencut/core/engine_registry.py`; `opencut/export/otio_compat.py`
- `pyproject.toml:69,170,230,267-269`; `.pre-commit-config.yaml`

### Host and platform
- https://github.com/Adobe-CEP/Samples/blob/master/PProPanel/ReadMe.md
- https://community.adobe.com/questions-606/cep-uxp-roadmap-should-developers-stop-building-cep-plugins-and-what-happens-to-existing-ones-1614807
- https://hyperbrew.co/blog/uxp-plugins-in-premiere-2026/
- https://github.com/leancoderkavy/premiere-pro-mcp/issues/21
- https://github.com/leancoderkavy/premiere-pro-mcp/issues/77
- https://github.com/tmoroney/auto-subs/issues/571

### Competitors
- https://help.opus.pro/docs/article/virality-score · https://opusclip.canny.io/changelog
- https://www.opus.pro/pricing · https://www.submagic.co/pricing · https://www.vizard.ai/pricing
- https://www.descript.com/features · https://feedback.descript.com/changelog
- https://riverside.com/magic-clips · https://www.captions.ai/pricing · https://klap.app/pricing
- https://github.com/browser-use/video-use · https://github.com/HKUDS/VideoAgent
- https://github.com/veedstudio/open-edit · https://github.com/KyaniteLabs/kinocut
- https://github.com/mifi/lossless-cut/issues/126 · https://github.com/WyattBlue/auto-editor

### Community signal
- https://community.adobe.com/t5/premiere-pro-ideas/feature-request-apply-silence-detection-and-removal-only-between-in-out-points/idi-p/15448602
- https://community.adobe.com/t5/premiere-pro-discussions/p-filler-words-not-found/td-p/15073046
- https://community.adobe.com/t5/premiere-pro-ideas/automatically-remove-silence-from-a-video/idi-p/13577633
- https://community.adobe.com/feature-requests-730/overhaul-captioning-workflow-1555697
- https://community.adobe.com/bug-reports-733/transcription-has-stopped-working-properly-after-latest-update-1627635
- https://news.ycombinator.com/item?id=44886647 · https://lobste.rs/s/ddssxd/captioning_all_my_youtube_videos_with_ai
- https://github.com/SysAdminDoc/OpenCut/issues/5

### Dependencies and standards
- https://pypi.org/pypi/faster-whisper/json · https://github.com/SYSTRAN/faster-whisper/commits/master
- https://pypi.org/pypi/deepfilternet/json · https://github.com/Rikorose/DeepFilterNet
- https://github.com/ggml-org/whisper.cpp/releases · https://github.com/k2-fsa/sherpa-onnx/releases
- https://github.com/AcademySoftwareFoundation/OpenTimelineIO/releases
- https://github.com/AcademySoftwareFoundation/OpenTimelineIO/issues/190
- https://github.com/AcademySoftwareFoundation/OpenTimelineIO/issues/343
- https://github.com/OpenTimelineIO/otio-aaf-adapter/releases
- https://github.com/ZFTurbo/Music-Source-Separation-Training · https://huggingface.co/spaces/hf-audio/open_asr_leaderboard
- https://ffmpeg.org/download.html · https://nvd.nist.gov/vuln/search (cpe:2.3:a:ffmpeg:ffmpeg, 2026-07)
- https://github.com/advisories/GHSA-qccp-gfcp-xxvc · https://github.com/advisories/GHSA-mf9v-mfxr-j63j
- https://modelcontextprotocol.io/specification/2026-07-28/changelog · https://peps.python.org/pep-0779/
- https://developer.adobe.com/premiere-pro/uxp/changelog/ · https://chromestatus.com/feature/5669293909868544

## Open Questions

- Does `rippleDelete` (UXP) and ExtendScript track-item deletion (CEP) actually mutate a Premiere 26.3
  sequence, or does it return success and no-op as premiere-pro-mcp #21 measured? This decides whether
  F319 is a verification harness or an emergency rewrite of the write-back path. Only a live 26.x host
  answers it; the read-back instrument in F319 is designed so the *user* answers it from a bug report.
- Is the 124-route CEP/UXP divergence deliberate sequencing (agent/search/deliverables intentionally
  UXP-first, export/nlp intentionally CEP-terminal) or accumulated omission? The ledger annotates tabs but
  never features, so the gate in F320 must report the gap before anyone can change behaviour.
- Should the direct-surface ratio be raised by adding surfaces or by retiring routes? 1,288
  integration-only routes are either an API product or dead weight, and the answer changes whether F328 is
  a ratchet or a deprecation programme.
