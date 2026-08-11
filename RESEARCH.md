# Research — OpenCut

Date: 2026-08-10 — replaces all prior research (previous pass: 2026-08-08, v1.47.0).

## Executive Summary

OpenCut v1.48.0 is a local-first Premiere Pro automation bridge: a Flask loopback service (1,590 routes,
1,565 shipped), CLI, MCP server, CEP + UXP panels, durable job engine, SQLite/FTS5 + federated media
index, FFmpeg processing, optional local AI adapters, OTIO/AAF/MLT interchange, and Windows/Linux
distribution lanes. Every P1/P2 opportunity from the 2026-08-08 pass shipped in v1.48.0 — auto-index
durability, the MCP Tasks adapter, the UXP compatibility drift gate, the federated index, the
deterministic audio-reactive renderer, and the MCP Apps review surface are all verified present in the
tree. The roadmap was genuinely drained, so this pass went after new ground: a live external bug report,
the blocked ledger's accuracy, in-hand-but-unused capabilities of the bundled binary, and 2026 platform
policy changes.

The strongest finding is that **the project's engineering is far ahead of its delivery**. 92 of the last
200 commits are `fix:`, releases cut every 2–4 days, and the last *published* GitHub Release is v1.25.1
(2026-04-20) against a source tree at 1.48.0 — 112 days and 23 versions of shipped work no user can
install. The one open external issue is a total blocker for a macOS CEP user on 1.48.0, and it has a
concrete root cause in this tree.

Top opportunities, priority order:

1. **P0 — Fix the CEP CSRF bootstrap for opaque origins.** GitHub issue #5 (2026-08-10, macOS, v1.48.0):
   the panel connects, then every action fails "Invalid or missing CSRF token".
   `opencut/routes/system.py:411-437` withholds `csrf_token` from `/health` whenever `Origin` is `null`
   or `file://`, and the CEP panel is a `file://` document (`CSXS/manifest.xml:29`, no `CEFCommandLine`
   overrides). Verified code path; platform behaviour needs live validation.
2. **P0 — Close the FFmpeg security floor: it is no longer blocked.** `Roadmap_Blocked.md:9-19` is stale
   by ~12 days. The bundled binary is the pinned post-fix snapshot and `require_security_floor()`
   **passes** — measured this pass. What remains is the item's own acceptance criterion (a per-CVE
   fix-commit/capability matrix instead of the current date heuristic), which is now unblocked work.
3. **P1 — Migrate the 13 `cv2.CascadeClassifier` call sites to YuNet.** `pyproject.toml:72` declares
   `opencv-python>=5,<6`; OpenCV 5 removed `cv2.CascadeClassifier`, so face-tracked auto-zoom and Haar
   face blur raise `AttributeError` for anyone who installs per the manifest. `FaceDetectorYN` works on
   both 4.x and 5.x, which dissolves the dependency standoff rather than waiting on it.
4. **P1 — Use the `whisper` audio filter already compiled into the bundled FFmpeg.** The shipped build
   carries `--enable-whisper`; a whisper.cpp ASR lane needs no torch, no Python model stack, and no
   download beyond a GGML model — directly attacking the "21 of 73 features are dependency-gated" problem.
5. **P1 — Make queue non-queueability observable.** 223 endpoints are allowlisted against ~760
   parameterless async POST routes; the rest fail with a generic "Endpoint not queueable". A read-only
   coverage report and a structured error name the gap without pre-empting the maintainer's curation
   decision (`Roadmap_Blocked.md:125`).
6. **P1 — Stop the panel transport from swallowing errors.**
   `extension/com.opencut.panel/client/backend-client.js:30,67` discard JSON-parse failures and every
   callback exception; 149 empty `catch (e) {}` blocks sit behind them. This is why issue #5 arrived with
   an empty logs section.
7. **P2 — Decide the Flathub lane.** Flathub's 2026 requirements now state that applications containing
   "AI-generated or AI-assisted code, documentation, or any other content are not allowed" and that
   submission PRs "must not be generated, opened, or automated using AI tools or agents", with permanent
   ban for repeat violations; console software is separately rejected. The repo ships a Flathub manifest
   and documents AI-assisted development. Verified verbatim on docs.flathub.org.
8. **P2 — Ship APV (RFC 9924).** The bundled FFmpeg already exposes `liboapv` encode plus `apv` and
   `apv_vulkan` decode. No OpenCut route touches it. Mirrors the existing `vvc_export.py` shape.

Confidence labels: **Verified** = confirmed in this tree or against a primary source during this pass;
**Likely** = strong multi-signal inference still needing implementation validation;
**Needs live validation** = cannot be closed headlessly.

## Product Map

### Core workflows
- **Media-to-cut** — analyse (silence/filler/beat/scene/transcript/OCR), review the proposal, write cuts,
  markers, captions, or media changes back through CEP or UXP.
- **Search-to-edit** — index transcript, OCR, audio-tag and visual metadata across configured roots; turn
  a search hit into a reviewable proposal or a timeline operation.
- **Caption and delivery** — transcribe, correct, style, synchronise, standards-check (TTML/IMSC 1.3/
  EBU-TT-D), render, and export to subtitle, caption, or interchange formats.
- **Review and approval** — long operations create durable jobs and versioned review artifacts with
  progress, cancellation, redaction, and explicit approval before host-facing or destructive actions.
- **Automation** — one loopback service reached from CLI, REST, MCP (with the Tasks extension), panels,
  and plugins.

### Personas
Solo editors and small post teams wanting fast local rough cuts without uploading footage; podcast/
education/social producers needing transcript-driven edits and captions; caption and delivery operators
needing deterministic exports and audit trails; technical users orchestrating scriptable, cancellable
operations without arbitrary code execution.

### Platforms and distribution
Python 3.11–3.14 core. Windows: PyInstaller + WPF installer + Inno Setup, bundled FFmpeg. Linux:
AppImage/Flatpak metadata. macOS: source/package lane, notarization blocked and in tension with the
project's no-signing policy. Premiere integration is CEP (primary, shipped by every installer) plus UXP
(strategic, PPRO minVersion 25.6). Optional model stacks are lazy-loaded; `[all]` is audited separately
from torch-backed extras.

### Key integrations and data flows
Panels/CLI/MCP submit validated commands to Flask routes; long work runs on bounded workers that persist
job state and write artifacts under `~/.opencut/`. CEP and UXP adapters translate approved operations
into Premiere calls — Python never calls ExtendScript directly. `network_policy.py` installs an
`sys.addaudithook` egress guard and an AST inventory so any new outbound module must be classified.

### Scope boundaries
Not a mobile editor, cloud collaboration service, multi-user MAM, or a replacement for Premiere's
timeline UI. Those were considered and rejected against the local-first threat and storage model.

## Competitive Landscape

### Adobe (Premiere 26.x, Media Intelligence, UXP)
Premiere 26.0 shipped AI Object Mask and an upgraded Media Intelligence that searches visuals,
transcripts, and metadata; Speech-to-Text, Enhance Speech, Scene Edit Detection, Auto Reframe, and Color
Match are free in-box, while Generative Extend and Firefly B-roll burn generative credits.
**Learn:** Adobe has absorbed generic "media search" and "scene detection" — those are no longer
differentiators. **Avoid:** chasing first-party parity. UXP 26.3 landed `ProjectConverter.exportAAF`,
`ObjectMaskUtils`, `EncoderManager.startBatchEncode`, transcript `hasTranscript`/`querySupportedLanguages`,
`Marker.guid`, `createSubClipAction`, and made `Sequence.setSelection` synchronous with actions requiring
`project.lockedAccess()`. OpenCut's UXP panel already feature-detects and uses all of these except
`createSubClipAction` (Verified by grep of `extension/com.opencut.uxp/main.js`).

### FireCut, AutoCut, AutoPod
FireCut ships roughly every 10 days ($19–44/mo; Magic Cut "raw footage to first draft" landed 2026-07-22).
AutoCut's Basic tier is silence removal *only* — captions, podcast, smart zoom, viral clips, B-roll,
repetition, profanity, and chapters all sit behind the AI tier. AutoPod is $29/mo flat, machine-locked.
**Learn:** what they paywall is precisely OpenCut's already-shipped surface — that is the marketing story,
and it argues for coherent task-shaped presets over a 1,565-route catalogue. **Avoid:** credit
accounting, machine locking, and opaque one-number scores.

### auto-editor
31.4.2 (2026-07-31), zero open issues, and now a Tauri + Nim-core desktop app with signed auto-updates.
Ships past OpenCut on multi-layer compositing, 255 edit labels, linked A/V dissolve transitions, and
NVIDIA Parakeet ASR alongside Whisper. **Learn:** a CLI-first project shipping a signed, auto-updating
desktop binary is the delivery discipline OpenCut lacks. **Avoid:** its zero-issue posture — there is no
public wishlist to mine.

### LosslessCut
42.8k stars; the single most-demanded unmet OSS feature is "smart cut" (re-encode only boundary GOPs),
issue #126 with 157 reactions. OpenCut already has `opencut/core/smart_render.py`. **Learn:** the demand
is real and OpenCut's implementation is undermarketed. **Avoid:** rebuilding a standalone trimmer.

### Subtitle Edit 5.x
Full Avalonia rewrite (5.0.0, 2026-06-24) giving one codebase native on Windows/macOS/Linux; SE4 is
bug-fix only. **Learn:** a single cross-platform UI codebase is what lets a small project support three
OSes — relevant to the CEP/UXP dual-panel cost, which is this repo's largest pain centre.
**Avoid:** becoming a second standalone subtitle editor.

### Premiere MCP projects (leancoderkavy/premiere-pro-mcp, ayushozha/AdobePremiereProMCP)
178★ and 82★, advertising 279 and 1,027 tools with CEP bridges and capability-aware UXP. **Learn:** the
demand for agent-driven Premiere control is now contested, and OpenCut's differentiator must be the
allowlisted catalogue, approvals, durable jobs, and the standardised Tasks lifecycle — not tool count.
**Avoid:** an arbitrary script-execution tool to match tool counts.

### OpenCut-app/OpenCut (name collision)
82,135 stars, MIT, actively pushed. This repo (36★) is effectively unfindable by name. The maintainer has
already decided to keep the product name and distribute as `opencut-ppro` (README, `pyproject.toml`);
this pass does not re-litigate that, but it is the reason discovery work must lean on channels
(winget/PyPI) rather than search.

### Adjacent model ecosystem
`audio-separator` 0.44.x carries a 100+ model catalogue, and Mel-Band RoFormer reports higher separation
quality than the BS-RoFormer entry currently registered in `core/engine_registry.py:466`. Because the
dependency is already pinned, this is a registry entry rather than a new stack. **Learn:** the separator
model zoo now moves faster than Demucs, whose repository is archived even though the PyPI package still
ships. **Avoid:** vendoring model weights or making any single separator mandatory.

### Community signal
The loudest 2026 signal is anti-AI-bloat and pro-stability, not more AI. Adobe's own silence detection is
reported misaligned from the waveform (bug 1554235), and captioning is the most-begged-for fix —
"a slight change requires completely re-rendering the timeline". Both are direct wedges for
`opencut/core/silence.py` and the caption burn-in path.

## Security, Privacy, and Reliability

### New findings this pass

- **CEP CSRF bootstrap withheld for opaque origins — Verified code path, needs live validation, critical
  impact.** `opencut/routes/system.py:411-437` returns `/health` without `csrf_token` when the request
  carries `Origin: null` or `file://` (`_CSRF_BOOTSTRAP_BLOCKED_ORIGINS`, line 230). The CEP panel is
  loaded from `./client/dist/index.html` over `file://` with no `<CEFCommandLine>` overrides
  (`extension/com.opencut.panel/CSXS/manifest.xml:29`). `/health` still returns 200 with capabilities, so
  the panel reports "connected" while every mutating call 403s. The retry in
  `extension/com.opencut.panel/client/backend-client.js:78-90` re-fetches `/health` and gets no token
  again, so it cannot recover. This exactly reproduces GitHub issue #5. The blocklist is legitimate — it
  stops a hostile local page harvesting the mutation token — so the fix must be a bootstrap channel a
  browser page cannot reach (ExtendScript-mediated read of a 0600 local token file is the natural one,
  since CEP Node privileges were deliberately removed in v1.44.0), not removal of the blocklist.
- **CSRF rejections are unobservable — Verified, medium impact.** `record_csrf_rejection()`
  (`opencut/security_audit.py:145-150`) stores only `token_present` and is exposed by no route; the
  bootstrap-withholding decision is not recorded at all. Issue #5 arrived with an empty logs block because
  there was nothing for the reporter to attach.
- **The FFmpeg P0 blocker is resolved and the ledger is stale — Verified.** `Roadmap_Blocked.md:9-19`
  states no compliant binary exists. Measured this pass: the bundled
  `ffmpeg/ffmpeg.exe` reports `2026-08-03-git-01a25f74cc-full_build-www.gyan.dev`, which equals
  `PINNED_INSTALLER_VERSION` in `opencut/core/ffmpeg_provenance.py:87`, and
  `require_security_floor()` returns `{"ok": true, "lane": "snapshot"}` covering all four July-2026 CVEs.
  Independently confirmed against the GitHub compare API: commit `01a25f74cc` is a descendant of every
  recorded fix commit — `4c62174` (CVE-2026-64832) ahead 493, `6f80e27` (CVE-2026-64833) ahead 532,
  `1836ef9` (CVE-2026-64835) ahead 523, `4da9812` (CVE-2026-66041) ahead 490, each with `behind_by: 0`.
  Residual weakness: the snapshot lane is a *date* comparison (`SNAPSHOT_FLOOR_DATE = "2026-07-06"`), so a
  differently-branched build dated after the floor would pass without containing the fixes — the item's
  own acceptance criterion (per-CVE fix-commit and capability matrix) is still unmet, and is now unblocked.
- **`opencv-python>=5,<6` breaks 13 modules on a manifest-faithful install — Verified.**
  `cv2.CascadeClassifier` is called by `auto_zoom`, `face_tools`, `ai_reframe_multi`, `deepfake_detect`,
  `face_tagging`, `morph_cut`, `multimodal_diarize`, `redaction`, `screenshot_video`, `skin_retouch`,
  `smart_reframe`, `talking_head`, and `thumbnail`. OpenCV 5 removed the classifier, so these raise
  `AttributeError` for fresh installs. `FaceDetectorYN` (YuNet) is present in both 4.5.4+ and 5.x.
- **SQLite FTS5 memory corruption (CVE-2026-11822, fixed 3.53.2) — Verified upstream, low-to-medium
  local exposure.** Out-of-bounds read in `fts5LeafSeek()` and heap overflow in `fts5ChunkIterate()`,
  triggered by a MATCH query against a *malicious database file*. OpenCut runs FTS5 in
  `core/footage_index_db.py` and the federated index; the realistic vector is a restored or imported
  index, not query text. There is no runtime `sqlite3.sqlite_version` floor anywhere in the tree.
- **`transformers>=5.3` permits CVE-2026-9856 — Verified upstream, low local exposure.** Path traversal
  via `chat_template` keys in `save_pretrained()`, affecting through 5.9.x, fixed 5.10.0. OpenCut calls
  `save_pretrained` nowhere (Verified by grep), so direct exposure is nil; raising the floor is hygiene
  for transitive callers, and it interacts with the blocked `huggingface-hub<1` lane.
- **Panel transport discards errors — Verified.** `backend-client.js:30` swallows every JSON-parse
  failure and `:67` swallows every user callback exception; 149 empty `catch (e) {}` blocks exist across
  the two panels. Python-side, `security_audit.py:43,93,103` silently discards three classes of failure
  inside the security-audit module itself — the worst-placed of the 239 `except Exception: pass` sites.

### Positive controls to preserve
CSRF on all mutating routes, trusted-host/DNS-rebinding gate, loopback-only default with an explicit
remote-auth token lane, SSRF and path validation, an `addaudithook` egress guard with an AST-enforced
module inventory, plugin trust/isolation, redacted job payloads, bounded workers, durable job journals,
ZIP-slip defences, C2PA embed + sidecar provenance, and rendered WCAG 2.2 AA scans across tabs, themes,
and breakpoints with no suppressions. Zero `shell=True`, zero bare `except:`, zero `pickle.load`, and
`eval`/`exec` only inside two audited sandboxes with dedicated AST-safety tests.

### Known external blockers excluded here
macOS notarization, the live Premiere host lane (UXP WebView cutover, `document.theme`, semantic-search
panel UI, 26.x smoke), release publication, the OpenCV/Transformers dependency-stack decision, broader
localization and Spanish review, PyPI/Homebrew/winget publication, and the queue-allowlist intent
decision all remain in `Roadmap_Blocked.md`. Two of them — winget and macOS notarization — are gated on
code signing, which the project's standing policy forbids; they should be closed as won't-do rather than
tracked as blocked.

## Architecture Assessment

### Strengths
The job engine already persists before returning, runs bounded workers, records terminal state, and
supports progress and cooperative cancellation — which is why the MCP Tasks adapter was a thin mapping.
Optional-import discipline, generated manifests that fail closed (`route_manifest.json`,
`feature_readiness.json`, `adobe_uxp_compatibility.json`), and `core/stub_scan.py` keeping stub modules
classified as stubs even when their dependency installs, together give this repo unusually honest
self-reporting for its size.

### Main seams
1. **Bootstrap/trust seam.** The panel's only credential path is `/health`, and it is origin-gated. There
   is no second, non-browser-reachable channel, so a single origin-header behaviour change bricks the
   product. This is the highest-leverage architectural gap and the cause of issue #5.
2. **Panel monolith seam.** `client/main.js` (~15.3k lines) and `client/style.css` (~17.9k lines) carry 68
   and 36 `fix:` commits respectively across the last 400 — the top pain centre by a wide margin, ahead of
   any backend module. Extraction should follow the transport boundary that `backend-client.js` already
   established rather than a speculative rewrite.
3. **Capability seam.** The bundled FFmpeg is compiled with `--enable-whisper`, `--enable-liboapv`,
   `--enable-libvvenc`, `--enable-libsvtav1`, `--enable-vulkan`, and `--enable-libplacebo`. OpenCut
   exposes VVC and SVT-AV1 but not the whisper filter or APV. Capabilities already paid for in the
   shipped payload should be preferred over new Python model stacks — 21 of 73 features are currently
   `missing_dependency`.
4. **Queue seam.** `_ALLOWED_QUEUE_ENDPOINTS` (223 entries, `routes/jobs_routes.py:180`) is a hand-written
   list guarding a route surface that grows by wave. A list-scoped gate certifies whatever is not in it;
   the coverage must be computed and reported even before the curation question is answered.
5. **Distribution seam.** Development cadence is ~2 days per release; delivery has been frozen 112 days.
   Every downstream channel (winget, Homebrew, awesome-lists, Flathub) is transitively blocked on the
   single untaken operator action, and `scripts/sync_version.py` rewriting 33 patterns across 23 files on
   each bump is why versions ship without tags.

### Test and documentation gaps
- No test asserts that a `file://` or `Origin: null` request to `/health` yields a usable panel bootstrap;
  the only coverage of `_health_should_expose_csrf_token` is `tests/test_reliability_hardening.py:20`.
- No drift test diffs async POST routes against `_ALLOWED_QUEUE_ENDPOINTS`.
- `tests/test_adobe_premierepro_versions.py:58,70,77` skip three dist-tag contract assertions because the
  snapshot was captured offline; the contracts are never verified.
- No runtime assertion on `sqlite3.sqlite_version`, and no test proves the OpenCV floor is importable —
  the declared-floor gate reports the violation but nothing fails on the call sites.
- `CLAUDE.md` states `ROADMAP.md` and `RESEARCH.md` are gitignored. They are not: `.gitignore:57-58`
  un-ignores both and `git ls-files` confirms they are tracked. Only `Roadmap_Blocked.md` is ignored.

### Operating constraints
The single-user loopback model, optional-dependency policy, no-code-signing rule, and local-only option
are coherent and should not be traded away. Any recommendation that requires signing, cloud inference, or
multi-user state contradicts them and is rejected below.

## Rejected Ideas

- **Adopt Premiere 26.3 UXP APIs (`exportAAF`, `startBatchEncode`, `ObjectMaskUtils`, `hasTranscript`,
  `querySupportedLanguages`, `lockedAccess`).** Already implemented and feature-detected in
  `extension/com.opencut.uxp/main.js` (lines 829, 1305, 1318, 1338, 1790, 1860). Only
  `createSubClipAction` is absent, and it needs a live host to validate. Source: competitor research.
- **Ship IMSC 1.3 support.** Already shipped with the correct REC date — `core/broadcast_caption.py:57`,
  `core/caption_interchange.py:28`. Source: standards research.
- **Add C2PA / Content Credentials.** Already present: `core/c2pa_embed.py`, `core/c2pa_sidecar.py`.
  Source: standards research.
- **Add ACES/OCIO colour management.** Already present: `core/aces_pipeline.py`, `core/ocio_validate.py`.
  Source: standards research.
- **Implement GOP-aware smart cut.** Already present: `core/smart_render.py`. Source: LosslessCut #126.
- **Swap diarization to pyannote community-1.** Already the default —
  `core/diarize.py:129`. Source: pyannote 4.x research.
- **Add Parakeet/NeMo ASR.** Already present: `core/asr_parakeet.py`, `core/asr_nemo_models.py`, and the
  `nemo-asr` extra. Source: Open ASR Leaderboard research.
- **Re-audit "unverified" CVE citations in `pyproject.toml`.** CVE-2026-7246 (click ≤8.3.2 `click.edit()`
  command injection, fixed 8.3.3) and CVE-2026-25645 (requests <2.33.0 predictable temp filename) were
  both confirmed real on NVD this pass. The citations are accurate; no action. Source: dependency research
  flagged them as unverifiable — that flag was wrong.
- **Add a `numpy` `python_version` marker.** `numpy>=1.24` with `requires-python>=3.11` resolves correctly;
  3.11 users get 2.4.x rather than 2.5.x. Divergence, not a defect. Source: dependency research.
- **Migrate to VVC as the delivery codec.** No browser support as of 2026 and no consumer hardware decode
  expected before 2028; `vvc_export.py` already exists and is ahead of demand. SVT-AV1 and APV are the
  higher-ROI lanes.
- **Rewrite the CEP panel monolith or force a CEP→UXP cutover now.** Adobe has announced no CEP removal
  date for Premiere; CEP 12 is security-fix-only but shipping. A cutover abandons pre-25.6 users and is
  gated on the blocked live-host lane. Extract along the existing transport boundary instead.
- **Pursue macOS notarization or winget.** Both require code signing, which standing policy forbids.
  Close as won't-do rather than track as blocked.
- **Mobile, cloud collaboration, multi-user MAM.** Contradict the local-first threat, storage, and
  distribution model.
- **Add a generic accessibility or i18n overhaul.** Rendered WCAG 2.2 AA scans with no suppressions
  already run across tabs, themes, and breakpoints; the remaining locale work is human-translation gated
  and correctly sits in `Roadmap_Blocked.md:84,135`.
- **Extend the plugin ecosystem.** The loader, manifest validation, authoring docs, marketplace client,
  and two example plugins exist, and v1.48.0 fixed the shipped manifests' API declarations. No evidence
  surfaced this pass of an unmet plugin-author need, so no work is proposed.
- **Add a new upgrade/migration mechanism.** WPF installer upgrade recovery with snapshot-and-restore,
  source-installer uninstall cleanup, and local DB migrations all shipped in v1.46.0–v1.48.0. The only
  remaining upgrade gap is that users cannot obtain a build at all, which is the blocked release-publication
  item — not a missing mechanism.
- **Reopen the project-name decision.** Settled in README and `pyproject.toml` (`opencut-ppro`); the 82k-star
  collision is a discovery problem to solve through channels, not a rename.

## Sources

### Repository evidence
- `opencut/routes/system.py:230,411-437`; `opencut/security.py:106-150`; `opencut/security_audit.py:145`
- `extension/com.opencut.panel/client/backend-client.js:24-90`; `extension/com.opencut.panel/CSXS/manifest.xml:29`
- `opencut/core/ffmpeg_provenance.py:40-105`; `opencut/routes/jobs_routes.py:180`; `pyproject.toml:72,151,185`
- `Roadmap_Blocked.md`; `opencut/_generated/route_manifest.json`; `opencut/_generated/feature_readiness.json`
- GitHub issue https://github.com/SysAdminDoc/OpenCut/issues/5

### Host and platform
- https://developer.adobe.com/premiere-pro/uxp/changelog/
- https://github.com/adobe/premierepro-types
- https://medium.com/adobetech/updates-for-creative-cloud-desktop-extensibility-0dd5c663563e
- https://hyperbrew.co/blog/uxp-plugins-in-premiere-2026/
- https://helpx.adobe.com/premiere/desktop/edit-projects/edit-with-generative-ai/generative-extend-faq.html

### Competitors
- https://firecut.ai/pricing/premiere-pro/ · https://firecut.ai/changelog/
- https://www.autocut.com/en/pricing/ · https://www.autopod.fm/pricing
- https://github.com/WyattBlue/auto-editor/releases · https://github.com/mifi/lossless-cut/issues/126
- https://github.com/SubtitleEdit/subtitleedit/discussions/11744
- https://github.com/leancoderkavy/premiere-pro-mcp · https://github.com/OpenCut-app/OpenCut

### Community signal
- https://community.adobe.com/feature-requests-730/premiere-pro-incorrectly-detects-silences-in-transcript-and-text-based-editing-1554235
- https://community.adobe.com/feature-requests-730/overhaul-captioning-workflow-1555697
- https://community.adobe.com/t5/premiere-pro-discussions/premiere-pro-2025-is-a-mess-stop-pushing-broken-features/m-p/15306568

### Security and dependencies
- https://www.ffmpeg.org/security.html · https://nvd.nist.gov/vuln/detail/CVE-2026-64832
- https://nvd.nist.gov/vuln/detail/CVE-2026-11822 · https://nvd.nist.gov/vuln/detail/CVE-2026-9856
- https://nvd.nist.gov/vuln/detail/CVE-2026-7246 · https://nvd.nist.gov/vuln/detail/CVE-2026-25645
- https://pypi.org/pypi/mediapipe/json · https://pypi.org/pypi/transformers/json

### Standards and distribution
- https://www.w3.org/TR/ttml-imsc1.3/ · https://peps.python.org/pep-0751/
- https://docs.flathub.org/docs/for-app-authors/requirements
- https://learn.microsoft.com/en-us/windows/apps/package-and-deploy/smartscreen-reputation
- https://developer.adobe.com/developer-distribution/creative-cloud/docs/guides/submission/overview
- https://modelcontextprotocol.io/specification/2026-07-28/changelog

## Open Questions

- Does Premiere's CEP runtime on macOS actually send `Origin: null` on the `/health` XHR, and does it
  differ from Windows? This determines whether F303's fix is the origin path alone or also a transport
  change. Only a live macOS Premiere host can answer it; the diagnostic in F303 is designed to capture it
  from the reporter without one.
- Is the queue allowlist deliberate curation or accumulated omission? Until the maintainer answers, the
  coverage report in F306 must describe the gap without changing behaviour.
- Should the Flathub manifest be retired outright or maintained against an attestation the project cannot
  currently make? This is a policy choice with a permanent-ban downside, not an engineering one.
- Which whisper.cpp GGML model tier should the FFmpeg-filter ASR lane default to, and should it ship a
  model or download on first use? The answer changes whether F307 is genuinely offline-first.
