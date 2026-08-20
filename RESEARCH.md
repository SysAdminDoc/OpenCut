# Research — OpenCut

Date: 2026-08-20 — replaces all prior research (previous pass: 2026-08-11, v1.48.0).

## Executive Summary

OpenCut v1.48.0 is a local-first Premiere Pro automation backend: a loopback Flask service (1,593 routes,
1,563 shipped, 107 blueprints), a 19-command CLI, a 98-tool MCP server, CEP + UXP panels, a durable job
engine, SQLite/FTS5 + federated media index, FFmpeg 8.x pipelines, optional local AI adapters, and
OTIO/AAF/MLT interchange. Since the 2026-08-11 pass, F319 (host-write read-back verification), F320
(feature-level panel parity gate), F321 (CEP horizon correction), and F322 (terminal-501 reclassification)
all shipped (commits `1fc60a34`, `fa16d53c`, `f05e6363`, 2026-08-12). This pass therefore went outward:
the June 2026 competitive teardown self-declared a ~3-month shelf life and expired, so its A-series gap
list was re-graded against the current tree, the Premiere-plugin competitor tier and Adobe's native 26.x
surface were re-surveyed, the dependency/platform ecosystem was refreshed, and community complaint mining
was re-run. Roughly 65 external sources were consulted across seven parallel research streams.

The headline finding is that **the June 2026 gap list is now almost entirely built and almost entirely
invisible.** A2 offline caption translation (`captions_enhanced.py`, NLLB license-gated with waiver), A3
paper-edit (`core/paper_edit.py`, `transcript_timeline_edit.py`), A4 animated caption templates (55
styles), A5 B-roll from the user's own library (`core/broll_suggest.py` — cue detection matched against
the footage index), A6 beat-synced assembly (beat-cut lane), A9 explainable virality
(`core/virality_score.py` with per-signal decomposition), A10 script-based assembly
(`core/script_to_roughcut.py`, `screenplay_parser.py`, `auto_rough_cut.py`), and a local review bundle
with comment normalization (`core/review_bundle.py` F105, `routes/collab_review_routes.py`) all exist.
Meanwhile the field charges $14–44/month for subsets of this (FireCut Max prices script-based editing at
$44/mo; Resolve gives IntelliScript away free). The deficit is not capability. The 2026-08-11 pass named
17.9% of shipped routes reachable from any first-party surface (F328), a published installer 23 versions
stale (blocked release item), and the one open external bug being a first-contact CSRF failure (issue #5).
F354 has since raised direct-surface coverage to 19.8% and given the ten largest route families named CLI,
palette, and curated MCP entry points. Nothing found this pass changes the remaining ordering; most of what
it found sharpens it.

Second, **the CEP/UXP platform risk hardened.** Two independent 2026 sources report CEP panels no longer
loading normally on Premiere 2026 (Hyper Brew's March analysis; auto-subs #571 field reports), while
Adobe's own statement still implies coexistence until ~Nov 2026. These conflict, and only the blocked
live-host smoke pass resolves them — but Adobe's 26.2/26.3 UXP releases added Hybrid Plugins (C++),
encoder/batch-encode control, `ProjectConverter.exportAAF`, `ObjectMaskUtils`, marker `guid`s, and
Transcript APIs, which plausibly unblock the two high-risk CEP-only host functions
(`ocAddNativeCaptionTrack`, `ocQeReflect`) that anchor the 119-route `uxp-pending` backlog. That audit is
headless work available now (F337).

Top opportunities this pass, priority order (new IDs F335–F346; details in ROADMAP.md):

1. **P1 — Map the two high-risk CEP-only host functions onto the Premiere 26.3 UXP API surface** (F337).
   The typings are already pinned at 26.3; the migration dashboard's route gate is red on exactly this.
2. **P1 — Prove the CSRF bootstrap fix closes issue #5's first-contact failure** (F338). A 39-star
   project pays more for one first-run failure than for any missing feature.
3. **P1 — Micro audio fades at silence/filler cut boundaries** (F335). Hard razor joins click; auto-editor
   queued the same fix in June. No `afade`/crossfade exists in `core/silence.py`.
4. **P1 — Batched faster-whisper inference for long files** (F336). Upstream's `BatchedInferencePipeline`
   is ~4x on long files; `captions.py` is sequential-only. Competitor slowness is a top complaint.
5. **P2 — Cut-review "disable instead of delete" write-back** (F339). Reviewer-tested FireCut gap;
   non-destructive staging fits OpenCut's review-first identity; no disable path exists in either host layer.
6. **P2 — Timeline-performance guardrail for large cut batches** (F340). Thousands of razor cuts make
   sequences unusable; warn, merge, or consolidate before write-back.
7. **P2 — Multicam cutting grammar** (F341). `multicam.py` exposes one knob (`min_cut_duration`);
   AutoPod's distrusted wide-shot handling and per-speaker-track requirement are its two weaknesses, and
   OpenCut's mixed-track diarization path is an unmarketed differentiator.
8. **P2 — De-reverb/denoise checkpoints via the pinned audio-separator** (F342). Registry-only addition,
   same shape as F316.
9. **P3 — F343–F346**: naming-section refresh against the OpenCut-app relaunch (85K stars, rewrite merged
   2026-07-14), best-take selection on repeat clusters, a packaged agent skill for the MCP server, and
   activating the `/analyze/video/qwen3vl` stub through local Ollama vision models.

Market events worth recording (they change positioning, not code): CapCut Pro doubled to $19.99/mo with
formerly-free captions/filler-removal/vocal-isolation paywalled (Feb 2026 renewal cliff); Play.ht was
acquired by Meta and shut down 2025-12-31, deleting customer voice clones — the strongest possible
argument for local voice cloning; Descript-refugee sentiment is loud on HN (2026-07-27 thread). All are
blocked on release publication before they can be exploited.

Dependency posture verified this pass: Flask floor `>=3.1.3` already covers CVE-2026-27205; Werkzeug
`>=3.1.6` exceeds the 3.1.5 advisory fix; pyannote is already pinned `>=4.0,<5` with community-1/exclusive
mode in `core/diarize.py`; NLLB is properly license-gated with an explicit waiver; SAM 3 is already carded
and checked; C2PA AI assertions (IPTC `digitalSourceType`) already ship in `core/c2pa_embed.py`. FFmpeg
9.0.1 (2026-08-12) gives F333 a tagged release to grade. Azure Trusted Signing is now ~$9.99/mo for
individual developers (US/CA), which softens the winget blocker from "budget" to "policy decision".

Confidence labels: **Verified** = confirmed in this tree or against a primary source during this pass;
**Likely** = strong multi-signal inference still needing implementation validation;
**Needs live validation** = cannot be closed headlessly.

## Product Map

### Core workflows
- **Media-to-cut** — analyse (silence/filler/beat/scene/transcript/OCR), review the proposal, write cuts,
  markers, captions, or media changes back into Premiere through CEP or UXP, now with read-back
  verification (F319).
- **Search-to-edit** — index transcript, OCR, face, and visual metadata across configured roots; turn a
  hit into a reviewable proposal or a timeline operation.
- **Caption and delivery** — transcribe, correct, style, synchronise, standards-check (TTML/IMSC 1.3/
  EBU-TT-D), render, and export to subtitle, caption, or interchange formats.
- **Review and approval** — durable jobs, versioned review artifacts, redaction, explicit approval before
  host-facing or destructive actions; portable local review bundles (F105).
- **Automation** — one loopback service reached from CLI, REST, MCP (Tasks extension), panels, plugins.

### Personas
Solo editors and small post teams wanting fast local rough cuts without uploading footage; podcast /
education / social producers needing transcript-driven edits and captions; caption and delivery operators
needing deterministic exports and audit trails; technical users and agents orchestrating scriptable,
cancellable operations without arbitrary code execution.

### Platforms and distribution
Python 3.11–3.14. Windows: PyInstaller + WPF installer + Inno Setup, bundled pinned FFmpeg. Linux:
AppImage/Flatpak metadata. macOS: source lane. Premiere integration is CEP (primary — shipped by every
installer; Adobe removal ~2026-11, with conflicting 2026 field reports of earlier breakage on Premiere
2026 — Needs live validation) plus UXP (strategic, PPRO minVersion 25.6). Newest published artifact is
v1.25.1 (2026-04-20) against a 1.48.0 tree; that gap gates every downstream channel and every marketing
wedge this pass found.

### Key integrations and data flows
Panels/CLI/MCP submit validated commands to Flask routes; long work runs on bounded workers that persist
job state and write artifacts under `~/.opencut/`. CEP and UXP adapters translate approved operations into
Premiere calls — Python never calls ExtendScript directly. `network_policy.py` installs a
`sys.addaudithook` egress guard with an AST module inventory.

### Scope boundaries
Not a mobile editor, cloud collaboration service, multi-user MAM, credit-metered service, or a replacement
for Premiere's timeline UI.

## Competitive Landscape

### Adobe Premiere 26.x (first-party)
Rebranded to "Adobe Premiere" with 26.0 (Jan 2026). Native now: Media Intelligence semantic search
(26.0 adds sound-description search and similar-take finding), Object Mask (20x faster tracking in 26.3),
Generative Extend, Single-Word Captions (26.3), one-click filler/pause deletion, Auto-Match Loudness,
Firefly Boards. Beta for late 2026: Color Mode, Firefly AI Assistant agent, AI Translation + Generative
Dubbing with lip-sync. Everything generative is credit-metered Firefly cloud.
**Learn:** plain silence/filler removal and unstyled word captions are no longer differentiators — keep
them as the on-ramp, differentiate on review depth, multilingual/styled output, headless access, and
"no meter". Adobe's meter is OpenCut's moat, and it grows as Adobe adds AI.
**Avoid:** competing with Object Mask, Generative Extend, or Firefly dubbing head-on.

### AutoCut / FireCut / TimeBolt / AutoPod / Excalibur (Premiere plugin tier)
AutoCut: $6.60–19.90/seat/mo; silences, captions, zoom, podcast multicam, resize, profanity, repeat
removal, B-roll; supports Premiere 2023–2026 and Resolve. FireCut: $10–44/mo; transcription hours metered
on every tier; script-based editing and workflow automations locked to the $44 Max tier; reviewer-tested
weaknesses: silence detection fails on background noise, breaks on a 2-hour 3-track sequence, cannot
disable-instead-of-delete, no sensitivity control. TimeBolt: $97/yr or ~$247–347 lifetime; local
processing; users call the UI clunky; forum reports of sequences becoming unusably laggy after
thousands of plugin cuts. AutoPod: $29/mo; multicam by active speaker but useless on mixed/shared audio
tracks, no filler removal, leaves silence padding around speech. Excalibur: $75 one-time command palette.
**Learn:** every reviewer complaint here maps to an OpenCut feature or a small item queued this pass
(F339 disable mode, F340 cut-count guardrail, F341 multicam grammar + mixed-track benchmark). Nobody in
this tier is free; none appears in a 2026 "best plugins" roundup as free, and neither does OpenCut — a
pure awareness gap gated on the release item.
**Avoid:** metering anything; shipping automation grammar without review.

### Descript / Riverside
Descript: $16–65/mo, dual-metered (media minutes and AI credits); Underlord became an agentic multi-step
editor; Rooms added remote recording; still owns seam repair (Regenerate Speech). Its Premiere XML
round-trip is broken in both directions per Adobe forum bug reports and Descript's own open feature
request — while OpenCut edits the real timeline in place, which eliminates the round-trip entirely.
Riverside: $24–79/mo; Magic Clips, eye-contact correction, chat editing.
**Learn:** "transcript editing with zero export" is a marketable sentence OpenCut can say and Descript
cannot. HN's Descript-refugee thread (2026-07-27) demands local, unmetered, audio-only-capable editing.
**Avoid:** dual metering; cloud-held voice models (see Play.ht below).

### Opus Clip / Klap / Vizard (clipping SaaS, brief update)
Opus Pro ($29/mo) paywalls its virality score; Klap claims trend-cycle analysis; Vizard's score is widely
called decorative. OpenCut's `virality_score.py` already decomposes signals per-clip and is ahead of the
field on explainability — unmarketed, unreachable from most surfaces.
**Learn:** rank-and-explain is won; surface it (F328 direction). **Avoid:** black-box scoring.

### CapCut (price event)
Pro went ~$9.99 → $19.99/mo with a Feb 2026 renewal cliff; formerly-free dynamic captions, filler-word
removal, and vocal isolation moved behind the paywall; June 2025 ToS grants ByteDance broad rights over
uploads; Trustpilot carries billing-after-cancellation complaints. OpenCut ships exactly the paywalled
trio locally for $0. **Learn:** a comparison page targeting this cohort writes itself — after a release
exists to point at.

### DaVinci Resolve 20/21 (free benchmark)
Resolve 20 free tier ships IntelliScript (script → timeline assembly), AI Audio Assistant, Multicam
SmartSwitch, AI Animated Subtitles; Studio ($295 one-time) adds Magic Mask 2, depth, de-aging (21.0,
June 2026). **Learn:** Resolve proves script-based assembly is a free-tier feature, validating
`script_to_roughcut.py` — the gap is surface and proof, not capability. OpenCut's Resolve bridge keeps it
relevant to editors who straddle both.

### auto-editor (5.0K stars, active)
31.5.0 (2026-08-13) bundles FFmpeg 9.0.1, adds pitch-shift and overlay actions, stacked fcp7-XML effects.
Closed issues queue micro-fades at cut edges (#1272), SmartCut compare (#1285), Parakeet linkage (#1284),
and content-aware editing via video-LLM (#1273).
**Learn:** micro-fades (F335) is the cheapest quality win it identified; content-aware scoring maps to
OpenCut's stubbed `/analyze/video/qwen3vl` lane (F346). **Avoid:** its render-and-discard output model.

### OpenCut-app (the unrelated web editor)
85,234 stars (2026-08-20), full Rust-core rewrite merged ~2026-07-14, preview at new.opencut.app; roadmap:
plugin store, headless HTTP rendering, scripting, MCP server, desktop/mobile Q4 2026. Its top-commented
issues: Chinese localization, native mobile, OOM on large media; an active "project name infringement"
thread confirms the naming territory is contested.
**Learn:** it independently validated the headless+MCP+plugin direction OpenCut already ships. The README
naming section is stale (48K stars, "when it relaunches") — F343. **Avoid:** the name fight; win the
qualifier ("OpenCut for Premiere", `opencut-ppro`).

### auto-subs (4.1K stars)
Now targets Resolve, Premiere, and After Effects; v3.8.0 (2026-07-21) added optional MMS forced alignment
to refine word timestamps post-ASR. Its issue #571 is a field report of CEP extensions not loading on
Premiere 2026. **Learn:** alignment-refinement is a caption-timing quality lever OpenCut's whisperx lane
already covers — verify parity rather than add. Its CEP breakage report is one of the two sources behind
the live-validation flag on the CEP horizon.

### ElevenLabs / Play.ht / HeyGen / Captions.ai (voice + talking-head commodity check)
ElevenLabs meters characters ($6–990/mo tiers); HeyGen credit-meters dubbing/lipsync; Captions.ai
(Mirage) gates eye-contact correction behind subscription. Play.ht was acquired by Meta (2025-07) and
terminated 2025-12-31 with customer voice clones deleted. Eye-contact correction is now table-stakes
across four commercial products and still has no viable permissively-licensed open model (re-checked this
pass; unchanged from the June 2026 disqualification).
**Learn:** "your cloud voice clone got deleted; a local one can't be" is a durable message for the
Chatterbox lane. **Avoid:** eye-contact until an open model actually exists.

### Frame.io (review category)
V4 forced-migration completed 2026-06-01; free tier is 2 users/2GB; Pro $15/user/mo. OpenCut already
ships the local-first alternative (`core/review_bundle.py` — zip with HTML summary, markers, captions,
deterministic hashing; comment normalization back into canonical marker shape). No free competitor exists
in this category. **Learn:** this is a surfacing/marketing story (F328), not new code.

### Subtitle Edit / Kdenlive (caption OSS)
Subtitle Edit 4.0.15 (2026-02-06): local Whisper, 300+ formats, built-in translation passes. Kdenlive
25.08: refined Whisper/Vosk speech editor. **Learn:** both validate transcript-first editing UX inside a
full editor; OpenCut's equivalent surfaces exist. Nothing new to build here.

## Security, Privacy, and Reliability

### Still open from the 2026-08-11 pass
- **FFmpeg per-CVE matrix grades 4 of ~16 July advisories** (F332, Verified, unchanged). New data point:
  FFmpeg 9.0.1 shipped 2026-08-12, giving F333 a tagged release to grade — the 9.0 branch-point caveat
  (cut 2026-06-26, before the July fixes landed on master) still applies to 9.0.x until ancestry-checked.
- **urllib3 floor below two High advisories fixed in 2.7.0** (F334, Verified, unchanged).
- **Error swallowing**: 41 empty `catch (e) {}` blocks in the CEP monolith, 226 `except Exception: pass`
  sites in `opencut/` (unchanged; why issue #5 arrived with an empty logs section).

### New findings this pass
- **The only open external bug is a first-contact failure — Verified.** Issue #5 (2026-08-10), "Invalid
  or missing CSRF Token", blocks a real user at the panel's first mutation. The Unreleased CHANGELOG
  records CSRF-bootstrap work, but nothing ties it to the reported scenario, no regression test names it,
  and the README troubleshooting section does not mention the error string. Queued as F338. The deeper
  problem: the fix helps nobody until a release ships (blocked item), because the reporter is necessarily
  on v1.25.1 or a source checkout.
- **Conflicting evidence on CEP loading in Premiere 2026 — Needs live validation.** Hyper Brew
  (2026-03-31) states CEP panels are no longer natively loaded in Premiere 2026 with no documented
  re-enable flag; auto-subs #571 reports the same from the field. Adobe's PProPanel ReadMe (Nov 2025)
  still implies coexistence to ~2026-11, and F329 documents the "Extensions (Legacy)" menu relocation
  that explains at least some reports. If the breakage reading is right, the CEP-primary installer is
  already broken for 26.x users and UXP parity (119 `uxp-pending` routes) becomes an emergency rather
  than a deadline. The blocked Premiere 26.x smoke item is the only resolver; its priority rises.
- **Dependency floors verified clean this pass — Verified.** Flask `>=3.1.3` covers CVE-2026-27205
  (session `Vary: Cookie` disclosure); Werkzeug `>=3.1.6` exceeds the CVE-2026-21860 fix (Windows
  device-name path traversal in `safe_join` — squarely this product's threat model, already covered);
  PyTorch `>=2.10` remains exactly right for CVE-2026-24747 (`weights_only=True` bypass). Transformers v5
  minors ship breaking changes routinely; the hash-pinned release locks already mitigate this, but any
  floor-only install lane inherits the churn.
- **Signing economics changed — Verified.** Azure Trusted Signing (being renamed Artifact Signing) is
  ~$9.99/mo and open to individual developers in US/CA; SignPath Foundation still offers free OV signing
  to qualifying OSS. The winget blocker in `Roadmap_Blocked.md` was recorded as "code-signing budget";
  it is now a policy decision, not a cost barrier. The no-signing stance (F318) remains coherent, but the
  premise changed and the decision deserves re-recording either way.

### Positive controls to preserve
Everything from the prior pass, plus the newly shipped host-write read-back verification (F319 —
`client/host-write-verification.js`, `uxp-host-write-verification.js`) and the feature-level parity gate
(F320). CSRF with opaque-origin bootstrap, trusted-host/DNS-rebinding gate, loopback-only default, SSRF
and path validation, `addaudithook` egress guard, plugin trust/isolation, redacted job payloads, bounded
workers, durable job journals, ZIP-slip defences, C2PA provenance with IPTC digitalSourceType assertions,
per-CVE FFmpeg acceptance grading, FTS5 memory-safety floor, rendered WCAG 2.2 AA scans, license-gated
restricted models (NLLB waiver pattern), and zero inline debt markers with readiness expressed through
generated, test-enforced manifests.

### Known external blockers excluded here
macOS notarization, the live Premiere host lane, release publication, the OpenCV/Transformers
dependency-stack decision, the FFmpeg-whisper model acquisition (F307), the Flathub policy decision
(F310), localization human review, PyPI/Homebrew/winget publication, and the queue-allowlist intent
decision all remain in `Roadmap_Blocked.md`.

## Architecture Assessment

### Strengths
The readiness system, generated manifests that fail closed, `core/stub_scan.py`, and the surface-coverage
gate give this repo unusually honest self-reporting for 575K lines. The June 2026 A-series regrade shows
the capability inventory is effectively complete against the commercial field; `smart_render.py`,
`morph_cut.py`, `paper_edit.py`, `script_to_roughcut.py`, `broll_suggest.py`, `virality_score.py`,
`review_bundle.py`, and the dubbing/ASR adapter fleet mean most of what competitors paywall is already
built. The deficit is reach: surfaces, releases, and proof. F352 now adds a
generated PEP 751 lock for the release lane and verifies it offline against the
existing hashed inputs, so lock generation does not become a network dependency
of the gate.

### Main seams
1. **Host-truth seam — instrumented, not yet answered.** F319 shipped read-back verification contracts
   for both panels; whether `rippleDelete` and ExtendScript deletion actually mutate a 26.3 sequence
   remains open until a live host runs it. The instrument now exists so a user's bug report answers it.
2. **Two-panel seam — dominant cost centre, now with a sharper edge.** 119 `uxp-pending` routes, 2
   high-risk host functions, and the conflicting CEP-2026 breakage reports. F337 (26.3 API audit) is the
   headless work that shrinks the unknown; the parity gate was deliberately relaxed from route-level to
   feature-level to keep CI green — the failing route-level gate object still sits in
   `uxp_migration_dashboard.json` and reads as red to a cold reader.
3. **Surface seam.** 310 of 1,563 shipped routes direct-reachable (19.8%), `primary_counts.cli = 10`.
   Every high-value June-gap feature that turned out to be already built is on the wrong side of this
   ratio. F328 (ratchet) is the structural fix; F335/F339/F341 add surface value where users already are,
   and F354 gives the ten largest integration-only families named CLI, palette, and curated MCP entry points.
4. **Interchange seam.** OTIO `>=0.17,<1` still admits the 0.19 C++ bundle rewrite unpinned (F331,
   unchanged; 0.18.1 remains newest, still prerelease-flagged, 9+ months quiet).
5. **Delivery seam — still the worst.** v1.25.1 (2026-04-20) vs 1.48.0 source. Every marketing wedge this
   pass surfaced (CapCut price cliff, Play.ht shutdown, Descript refugees, plugin-roundup absence) is
   unusable until this closes, and the CSRF fix for issue #5 is invisible to its own reporter.

### Refactor candidates
- `core/silence.py` export path: no boundary fades (F335); range scoping and tighten mode already queued
  (F325).
- `core/multicam.py`: single-knob grammar (F341).
- `core/captions.py`: sequential-only ASR (F336); decoder-level glossary biasing already queued (F323).
- Host layers: no clip-disable primitive (F339); no cut-count guardrail (F340).

### Test and documentation gaps
- Everything from the prior pass still stands (ruff/pytest config drift F330, pre-push breadth, manifest
  clock drift, `installer/bin` CLAUDE.md claim contradicted by `git ls-files`).
- No regression test names the issue-#5 CSRF failure shape (F338).
- No fixture stresses the FireCut failure cases OpenCut can beat: 2-hour 3-track sequences and noisy-floor
  silence detection (fold into F341's benchmark fixture and existing silence tests).
- README naming section carries stale facts (F343).

### Operating constraints
The single-user loopback model, optional-dependency policy, no-code-signing rule (premise changed —
re-record the decision, see Security), no-telemetry default, and the absence of hosted CI are coherent
and should not be traded away. Any recommendation requiring cloud inference, metered credits, or
multi-user state contradicts them and is rejected below.

## Rejected Ideas

Carried forward from 2026-08-11 (all still hold; re-verified where marked):

- **GOP-aware smart cut** — already present (`core/smart_render.py`). Source: LosslessCut #126.
- **Parakeet/Canary/Moonshine/NeMo ASR** — already present (`asr_router.py` + adapters).
- **Seam repair / jump-cut smoothing** — already present (`core/morph_cut.py`); surfacing work, not new code.
- **Brand kit / SEO / split-screen reframe** — already present (`brand_kit.py`, `seo_optimizer.py`,
  `split_screen.py`, `ai_reframe_multi.py`).
- **Voice-cloned dubbing** — already present (`auto_dub_pipeline.py`, `dub_pipeline.py`, `ai_dubbing.py`).
- **Face/OCR/visual media-index signals** — already present (`face_tagging.py`, `ocr_extract.py`,
  `semantic_video_search.py`, `federated_media_index.py`).
- **Migrate off faster-whisper** — wrapper-stall risk is recorded (F327 territory); engine is healthy;
  adapters exist. A rewrite trades a known-good pin for churn.
- **Hosted CI** — deliberately absent; widen the local gate instead.
- **`.prproj` corruption salvage** — outside the panel-only host channel and trust model.
- **Agent-native JSON timeline convention** — OTIO + MCP already cover it interoperably.
- **Credits/cloud/marketplace mechanics** — contradict local-first, no-metering.
- **Generic a11y or i18n overhaul** — WCAG scans ship; the actionable defect is F324's namespace split.
- **CEP monolith rewrite / forced UXP cutover now** — measure and close parity along the seam (F320/F337);
  the cutover is gated on the blocked live-host lane.
- **Plugin-ecosystem expansion** — loader, trust, docs, marketplace client, examples all ship; no unmet
  author need surfaced again this pass.
- **Mobile / multi-user / migration mechanisms** — contradict the model; upgrade machinery ships.
- **Offline mode** — local-by-default plus `OPENCUT_LOCAL_ONLY` already is one.
- **MCP 2026-07-28 migration** — already done (`mcp_server.py` declares the stateless revision).
- **Python 3.14 free-threaded build** — parallelism already lives in subprocesses and GIL-releasing
  extensions; wheels not there.
- **WebGPU in CEP** — Chromium 99 ceiling; UXP-only, gated on live host.

New rejections from this pass:

- **Eye-contact / gaze correction** — demanded across Descript, Riverside, Captions.ai, HeyGen, and Adobe
  forum threads; re-checked 2026-08-20: still no viable permissively-licensed open model (NVIDIA Maxine is
  proprietary SDK). Re-check on a future pass; do not build speculatively. Source: commercial sweep.
- **Cloud ASR fallback (Groq/OpenAI/Gemini)** — auto-editor added it (#1278); for OpenCut it dilutes the
  local wedge for a user segment (GPU-less) that CPU Whisper tiny/base and the Parakeet lane already
  serve. Cloud LLMs remain the only sanctioned cloud inference. Source: auto-editor issues.
- **MatAnyone 2 matting backend** — CVPR 2026 quality leader, but NTU S-Lab license is non-commercial;
  RVM ships and SAM 3 is carded. Revisit only if relicensed. Source: OSS sweep.
- **SeedVR2 / FlashVSR upscalers** — already present as `/video/upscale/seedvr2` (dependency-gated) and
  `/video/upscale/flashvsr` (stub); the routes exist, activation is tracked by the readiness system, not
  research. Source: ecosystem sweep, `route_manifest.json`.
- **Kokoro TTS** — already the shipped "balanced" TTS engine. Source: OpenCut-app draft PR #525 (they are
  catching up to it).
- **Auto B-roll from the user's own library** — already present (`core/broll_suggest.py` matches cues
  against the footage index). The #1 cross-competitor paywalled feature is built; it needs surface, not
  code. Source: commercial paywall ranking; grep verification 2026-08-20.
- **Review-comments round-trip** — already present (`review_bundle.py` comment normalization +
  `collab_review_routes.py`). Source: Frame.io category scan; grep verification 2026-08-20.
- **Script-based editing / paper-edit** — already present (`script_to_roughcut.py`, `paper_edit.py`,
  `screenplay_parser.py`). FireCut charges $44/mo for this. Source: June A-series regrade.
- **WhisperLive-style live captioning while recording** — OpenCut does not own a recording surface;
  Premiere does. Out of the product's data path. Source: OSS sweep.
- **Shotcut-style stock elements/soundboard panel** — asset licensing burden and panel scope creep; SFX
  generation already ships procedurally. Source: Shotcut 26.8 release.
- **stable-ts migration** — not a dependency (grep verification 2026-08-20); its 2026-05-30 archival is
  moot here. Source: OSS sweep.
- **pyannote 4.0 upgrade** — already pinned `>=4.0,<5`; community-1/exclusive-mode handling present in
  `core/diarize.py`. Source: ecosystem sweep; grep verification 2026-08-20.
- **C2PA AI-edit assertions** — already present (`c2pa_embed.py` builds IPTC digitalSourceType actions).
  YouTube's Content Credentials pilot makes this newly marketable — a positioning note, not code. Source:
  ecosystem sweep; grep verification 2026-08-20.

## Sources

### Repository evidence
- `opencut/_generated/route_manifest.json`, `uxp_migration_dashboard.json`, `panel_feature_parity.json`
- `opencut/core/silence.py`, `multicam.py:88`, `captions.py`, `repeat_detect.py`, `broll_suggest.py`,
  `paper_edit.py`, `script_to_roughcut.py`, `review_bundle.py`, `virality_score.py`, `c2pa_embed.py:36-176`,
  `diarize.py`, `model_cards.py:197-216` (NLLB waiver)
- `extension/com.opencut.panel/host/index.jsx` (no `.disabled`, no consolidation — greps 2026-08-20)
- `pyproject.toml:47` (flask>=3.1.3), `:127,201` (pyannote>=4.0)
- `docs/RESEARCH_COMPETITIVE_TEARDOWN_2026-06-10.md` (A-series baseline)
- git log 2026-08-12: `1fc60a34` (F319), `fa16d53c` (F320), `f05e6363` (F321)

### Host and platform
- https://developer.adobe.com/premiere-pro/uxp/changelog (26.2 Hybrid Plugins; 26.3 encoder/AAF/Transcript/ObjectMaskUtils)
- https://hyperbrew.co/blog/uxp-plugins-in-premiere-2026/ · https://github.com/tmoroney/auto-subs/issues/571
- https://github.com/Adobe-CEP/Samples/blob/master/PProPanel/ReadMe.md
- https://medium.com/adobetech/updates-for-creative-cloud-desktop-extensibility-0dd5c663563e

### Competitors
- https://community.adobe.com/announcements-727/what-s-new-in-adobe-premiere-26-3-june-2026-1628369
- https://blog.adobe.com/en/publish/2026/04/15/adobe-extends-leadership-video-unleashing-new-ai-powered-creation-firefly-reinventing-color-editors-in-premiere
- https://firecut.ai/pricing/all/ · https://www.autocut.com/en/ · https://www.timebolt.io/pricing · https://www.saasworthy.com/product/autopod-fm
- https://cutback.video/blog/the-best-auto-silence-removal-plugin-for-premiere-pro · https://www.freevisuals.net/post/firecut-ai-review
- https://fluxnote.io/guides/descript-pricing-2026 · https://www.descript.com/blog/article/descript-season-6-meet-underlord
- https://www.newsweek.com/app-used-millions-nearly-doubles-subscription-price-overnight-11535999 (CapCut)
- https://bigvu.tv/blog/capcut-free-vs-pro-what-2026s-restructure-actually-gives-you/
- https://www.starkinsider.com/2025/05/davinci-resolve-20-drops-with-ai-powered-upgrades.html · https://www.cgchannel.com/2026/06/blackmagic-design-releases-davinci-resolve-21-0/
- https://github.com/WyattBlue/auto-editor/releases (31.5.0) · https://github.com/WyattBlue/auto-editor/issues/1272
- https://github.com/OpenCut-app/OpenCut (85K stars) · https://explainx.ai/blog/opencut-rewrite-plugins-headless-mcp-2026
- https://texttolab.com/blog/play-ht-shutdown-alternatives · https://www.eesel.ai/blog/captions-ai
- https://help.frame.io/en/articles/9859849-adobe-premiere-frame-io-v4-comments-panel-overview
- https://getrecut.com/ · https://klap.app/alternatives/vizard-ai · https://quso.ai/blog/opus-clip-pricing

### Community signal
- https://github.com/SysAdminDoc/OpenCut/issues/5
- https://news.ycombinator.com/item?id=49065779 (Descript refugees, 2026-07-27)
- https://news.ycombinator.com/item?id=45980760 (Mosaic agentic editing; determinism + local demand)
- https://community.adobe.com/questions-729/using-a-plugin-to-cut-out-silence-and-the-sequence-becomes-unusably-laggy-1411061
- https://community.adobe.com/t5/premiere-pro-ideas/the-ultimate-premiere-pro-feature-request-2025/idi-p/15634461
- https://diyai.io/ai-tools/video-generation/reviews/autopod-review/ · https://fivetaco.com/products/timebolt/reviews
- https://aescripts.com/learn/post/26-premiere-pro-plugins-to-streamline-your-workflow

### Dependencies and standards
- https://ffmpeg.org/download.html (9.0.1, 2026-08-12) · https://ffmpeg.org/security.html
- https://github.com/SYSTRAN/faster-whisper/releases (BatchedInferencePipeline)
- https://www.pyannote.ai/blog/community-1 · https://github.com/nyrahealth/CrisperWhisper
- https://github.com/nomadkaraoke/python-audio-separator (de-reverb checkpoints)
- https://www.sentinelone.com/vulnerability-database/cve-2026-27205/ (Flask) · https://github.com/advisories/GHSA-63cw-57p8-fm3p (torch 2.10)
- https://github.com/facebookresearch/sam3 · https://github.com/pq-yang/MatAnyone2 (license check)
- https://azure.microsoft.com/en-in/pricing/details/trusted-signing/ · https://comparecheapssl.com/free-code-signing-certificate-and-how-to-get-it/
- https://editorsweblog.org/2026/04/12/c2pa-adoption-tracker-platforms-content-credentials-2026
- https://peps.python.org/pep-0751/ · https://www.infoq.com/news/2026/05/pip-261-dependency-cooldowns/
- https://www.remotion.dev/blog (Agent Skills)

## Open Questions

- Does the shipped read-back instrumentation (F319) confirm or refute the premiere-pro-mcp #21 no-op on a
  live Premiere 26.3 host — and separately, does the CEP panel load at all on Premiere 2026 outside the
  "Extensions (Legacy)" menu? Both need the blocked live-host lane; the second decides whether the
  ~2026-11 horizon is real or already passed.
- Is the 124-route CEP/UXP divergence deliberate sequencing or accumulated omission? F320's gate now
  reports it at feature level; the intent decision is unmade.
- Should the direct-surface ratio rise by adding surfaces or retiring routes? The June A-series regrade
  sharpened this: the highest-value already-built features (paper-edit, broll_suggest, review bundles,
  virality explainability) are precisely the ones a surface-first answer would promote (F328).
- Does the maintainer accept a ~$120/yr signing subscription now that Azure Trusted Signing admits
  individuals, or does the no-signing policy (F318) stand on principle? Either answer should be recorded
  with its date and premise.
