# OpenCut -- Roadmap

**Version**: 5.0
**Updated**: 2026-06-07
**Baseline**: v1.32.0 -- 1,539 routes, 107 blueprints, 602 core modules, 9,900+ tests, CEP + UXP panels, DaVinci Resolve bridge, MCP server
**License**: MIT
**Replaces**: ROADMAP.md v4.x (implementation ledger archived in git history)

> Route and blueprint counts are generated from `opencut/_generated/route_manifest.json`.
> Regenerate with `python -m opencut.tools.dump_route_manifest` before each release.

---

## How to read this document

| Document | Purpose |
|---|---|
| **TODO.md** | Compact active execution queue for the next implementation pass |
| **This file** | Strategic roadmap: what to build, in what order, and why |
| **ROADMAP-NEXT.md** | Older Wave A-K detail (mostly shipped through v1.28.x) |
| **COMPLETED.md** | Summarized shipped work |
| **CHANGELOG.md** | Chronological release log |
| **features.md** | Aspirational 402-feature brainstorm catalogue -- not a ship promise |

When this file and features.md disagree, **this file wins**.
When this file and the live code disagree, **the code wins**.

---

## Design principles

1. **Local-first.** Core editing works entirely offline. No cloud, no API keys, no subscriptions for core features. Telemetry disabled by default.
2. **One new dependency per feature, max.** Prefer optional pip extras with graceful degradation via `check_X_available()`.
3. **Permissive licenses only.** MIT, Apache-2, BSD, ISC, LGPL are fine. CC-BY-NC, research-only, GPL code reuse, and unclear licenses are rejected or deferred.
4. **Never break what works.** Every wave ships independently. Backward-compatible API evolution.
5. **Match existing patterns.** Subscriptable dataclass results, `@async_job` decorator, `on_progress` callbacks, queue allowlist entries, model card per AI feature.
6. **CEP is legacy; UXP is the future.** All new panel features target UXP first. CEP gets maintenance only until Adobe enforces EOL (~September 2026).

---

## Hard constraints

| Constraint | Detail |
|---|---|
| Python runtime | >=3.11 (onnxruntime 1.25+ requires it; 3.10/3.11 EOL October 2026 -- plan 3.12+ baseline) |
| Network posture | Loopback-only by default; non-loopback requires `OPENCUT_ALLOW_REMOTE=1` + auth token |
| CSRF | `@require_csrf` on every POST/PUT/PATCH/DELETE route |
| Path validation | Rejects `..`, null bytes, symlinks outside allowlist, UNC/network paths |
| C2PA provenance | On all generated/exported media (upgrading to spec 2.3) |
| Plugin sandbox | Refuses unsigned plugins unless `OPENCUT_PLUGIN_ALLOW_UNSIGNED=1` |
| Request size | 100 MB `MAX_CONTENT_LENGTH` |

---

## Tier definitions

| Tier | Meaning | Timeline |
|---|---|---|
| **Now** | Actively queued or in-flight. Blocks next release or has a regulatory/security deadline. | v1.33-v1.34, ~4 weeks |
| **Next** | High-value, well-scoped, dependencies resolved. | v1.35-v1.42, ~6 months |
| **Later** | Valuable but lower urgency, larger effort, or waiting on upstream. | 6-12+ months |
| **Under Consideration** | Needs more research, community signal, or a champion. | No timeline |
| **Rejected** | Explicitly declined with reasoning. | -- |

---

## Now (v1.33-v1.34)

### Ongoing work

| ID | Item | Status | Detail |
|---|---|---|---|
| E15 | CEP i18n migration / UXP i18n expansion | Rolling batches (CEP 173/~160+; UXP foundation + Cut/Captions/FCC display/Audio/Video-through-Style-Transfer slices) | Removing bare-English strings from the CEP panel and expanding scanner coverage; UXP now has shell/Cut/Captions/FCC display/Audio/Video-through-Style-Transfer locale guards while full parity remains open. |
| F202 | macOS notarization live acceptance | Blocked: needs GitHub secrets | Repository wiring exists. Deadline: **2026-09-01**. |
| F252 | UXP WebView cutover | Blocked: needs Premiere UDT evidence | Bolt UXP scaffold exists. |

### Security (immediate)

| Item | Severity | Detail |
|---|---|---|
| PyTorch CVE-2025-32434 | CVSS 9.3 | Closed 2026-06-07: model quantization now uses `weights_only=True`, unsafe pickle checkpoints return a clear error, and Torch-backed extras require `torch>=2.6` / `torchvision>=0.21`. |
| CLIP cache deserialization | High | Closed 2026-06-07: semantic video search caches now use compressed `.npz` files with JSON metadata and `numpy.load(..., allow_pickle=False)` instead of raw pickle caches. |
| Flask CVE-2026-27205 | Info Disclosure | Session Vary: Cookie header missing in <=3.1.2 |
| Pillow CVE-2026-25990 | High | Out-of-bounds write |
| OpenEXR CVE-2026-39886/40244/40250 | High | HTJ2K/DWA decoder overflow |
| Python 3.10/3.11 EOL | Planning | Both EOL October 31, 2026 |

### Governance and hardening (RA-01 through RA-36)

| ID | Item | Effort | Why |
|---|---|---|---|
| RA-01 | Ruff target-version alignment | S | Closed 2026-06-06: Ruff now targets the declared Python 3.11 floor |
| RA-02 | requirements/pyproject alignment | S | Closed 2026-06-06: core/standard requirements now mirror `pyproject.toml` bounds |
| RA-03 | Direct typed error logging | S | Closed 2026-06-06: direct typed errors now log structured context |
| RA-04 | Request ID in error bodies | S | Closed 2026-06-06: structured error bodies now include the generated request ID |
| RA-05 | SQLite PRAGMA user_version | M | Closed 2026-06-06: local SQLite stores now use explicit `user_version` migrations |
| RA-06 | Destructive wipe backup | M | Closed 2026-06-06: local SQLite destructive maintenance paths now expose dry-run counts, optional backups, and audit metadata |
| RA-07 | Job result_json cap | S | Closed 2026-06-06: oversized job results spill to content-addressed local files |
| RA-08 | DB compaction diagnostic | S | Closed 2026-06-06: local SQLite diagnostics report page, freelist, WAL, and file-size posture |
| RA-09 | Timeline-native captions | L | Closed 2026-06-06: RA-46 sidecars, RA-47 diff/apply, RA-48 UXP snapshot reads, RA-49 CEP/hybrid write contracts, and RA-50 metadata-loss fixtures shipped |
| RA-10 | Magic clips macro | L | Closed 2026-06-06: RA-51 through RA-56 shipped dry-run plan graphs, approved-candidate render handoff, explainable scoring, platform preset rendering, CEP/UXP review-board parity, checkpointed resumable runs, and output bundle handoff manifests |
| RA-11 | UXP least-privilege filesystem | M | Closed 2026-06-06: live and WebView manifests use picker-scoped `localFileSystem: "request"` with static guards against direct file APIs |
| RA-12 | Hybrid plugin validator | M | Closed 2026-06-06: static validator checks UXP Hybrid `.uxpaddon` manifest opt-in, safe addon names, host shape, and mac arm64/mac x64/win x64 package layout |
| RA-13 | UXP external launch perms | M | Closed 2026-06-06: live and WebView manifests declare HTTPS-only `launchProcess`, OAuth launches validate HTTPS URLs, and static tests block file-launch APIs |
| RA-14 | WebView permission split | M | Closed 2026-06-06: dormant WebView config exports development and release manifest profiles with dev-only hot reload domains and release-local message bridge |
| RA-15 | [all] advisory decision | M | Closed 2026-06-06: `opencut[all]` is the release-audited convenience lane; Torch/Transformers-backed packages are explicit via `torch-stack` and named feature extras |
| RA-16 | Adobe dist-tag tracking | S | release-* tags untracked |
| RA-17 | UXP manifest schema guard | M | Closed 2026-06-06: live UXP manifest declares Premiere-supported `manifestVersion: 5` and tests guard the dormant WebView v6 template separately |
| RA-18 | UXP deprecation sentinel | M | Closed 2026-06-06: static guard blocks deprecated Clipboard APIs, object-form clipboard writes, and legacy `uxpvideo*` events in UXP/WebView sources |
| RA-19 | UXP clipboard permission | S | Closed 2026-06-06: live and WebView manifests declare clipboard `readAndWrite`, and copy actions use a shared fallback helper |
| RA-20 | UXP confirmation guard | S | Closed 2026-06-06: raw UXP browser dialogs are blocked and search-index clear uses inline second-click confirmation |
| RA-21 | Python 3.13 classifier proof | M | Closed 2026-06-06: untested Python 3.13 classifier retracted until a CI lane proves it |
| RA-22 | Release Full Node pin | S | Closed 2026-06-06: Release Full now sets up Node 22 before Linux CEP panel npm gates, matching PR Fast |
| RA-23 | GitHub Actions SHA pins | M | Closed 2026-06-06: workflow action refs are pinned to full-length SHAs with adjacent version comments |
| RA-24 | Release Full token perms | M | Closed 2026-06-06: Release Full build/test/package legs are read-only, with release uploads isolated in a write-scoped tag-only job |
| RA-25 | Docker dependency surface | M | Closed 2026-06-06: Docker installs from tracked `requirements.txt` and no longer reintroduces retired packages |
| RA-26 | Docker runtime parity | M | Closed 2026-06-06: Docker defaults publish HTTP 5679 only, with WebSocket/MCP sidecars documented as opt-in |
| RA-27 | Docker GPU compose | S | Closed 2026-06-06: README and compose docs now use the committed GPU profile command |
| RA-28 | README count gate | S | Closed 2026-06-06: README non-badge count claims are checked against generated/live counts |
| RA-29 | Docker fail-closed | M | Closed 2026-06-06: Docker dependency installs use the requirements file and fail on pip errors |
| RA-30 | Docker build-context hygiene | S | Closed 2026-06-06: `.dockerignore` mirrors secret/log ignores and excludes local runtime/cache DB state |
| RA-31 | Adobe tracker exit-code | S | Exit codes lost |
| RA-32 | Adobe tracker labels | S | Unseeded labels |
| RA-33 | Label dry-run without gh | S | Requires CLI |
| RA-35 | Release SBOM fidelity | M | Closed 2026-06-06: declared-SBOM artifact naming and CycloneDX fidelity metadata |
| RA-36 | CEP UNC/HGFS-safe Node | M | Closed 2026-06-06: Windows-safe panel aliases route shared-folder npm gates through a script-root anchored wrapper |

### Drop-in model upgrades

| Item | Effort | Source |
|---|---|---|
| Silero VAD v5 -> v6.2 | S | https://github.com/snakers4/silero-vad |
| Mel-RoFormer (upgrade BS-RoFormer) | S | https://github.com/lucidrains/BS-RoFormer |
| Whisper large-v3-turbo evaluation | S | https://huggingface.co/openai/whisper-large-v3-turbo |

---

## Next (v1.35-v1.42, ~6 months)

### Chat-Conductor Agent (F143-F146) -- flagship

Descript Underlord ($24-65/mo), Captions.ai agent ($13-30/mo), Adobe Firefly AI Assistant, Riverside Co-Creator, and HeyGen MCP all prove sidebar-chat + timeline-diff is the converging UX. OpenCut has every building block but no conductor.

| ID | Item | Effort |
|---|---|---|
| F143 | /agent/chat conductor -- plan, execute, checkpoint, rollback | L |
| F144 | Post-turn self-review -- drift score, suggested-retry | S |
| F145 | Skills SDK -- declarative manifest, route-validated plan | M |
| F146 | UXP-native MCP transport (survives CEP EOL) | M |

### AI model upgrades

| Item | Effort | License |
|---|---|---|
| SAM 2 -> SAM 3 (open-vocabulary concept segmentation) | M | Meta license |
| Depth Anything V2 -> V3 (SOTA metric depth) | M | Apache-2 |
| LTX-Video 0.9.5 -> 13B (30fps, 30x faster) | M | Custom (free <$10M) |
| Wan 2.2 -> 2.7 (15-sec, instruction editing) | M | Apache-2 |
| HunyuanVideo 1.5 (2x inference speed) | M | Tencent license |
| FFmpeg 7.x -> 8.0 (Whisper filter, Vulkan codecs) | M | LGPL |
| SeedVR2 (one-step video restore+upscale) | M | Apache-2/MIT |
| FlashVSR (real-time streaming super-res) | M | Check repo |
| Qwen3-TTS (10 languages) | M | Apache-2 |
| CosyVoice 3.0 (150ms latency, 9 languages) | M | Check repo |
| Dia 1.6B (multi-speaker dialogue) | M | Apache-2 |
| pyannote.audio 4.0 (pure PyTorch) | S | MIT |
| Sapiens2 (unified human vision) | M | Meta license |
| VideoLLaMA 3 (video understanding) | M | Apache-2 |
| YuE (full-song generation) | M | Apache-2 |
| Stable Audio 3.0 (6-min tracks, open weights) | M | Stability license |

### Competitor parity (paywalled at $6-99/mo by competitors)

| Item | Effort | Who charges |
|---|---|---|
| AI eye contact correction | L | Descript, Captions.ai, VEED ($24-30/mo) |
| AI Overdub / voice correction | L | Descript, ElevenLabs ($6-99/mo) |
| Morph Cut / smooth jump cut | L | Premiere Pro (bundled $23/mo) |
| AI auto-color grading by intent | M | Colourlab.ai ($99/mo) |
| Best take selection scoring | M | AutoCut, Gling ($10-50/mo) |
| Watch folder / hot folder | M | Adobe Media Encoder (bundled) |
| Render queue / multi-format batch | M | DaVinci Resolve (free) |
| A/B variant generation for shorts | M | Opus Clip ($15-29/mo) |
| Engagement / virality scoring | M | Opus Clip, Vizard ($15-49/mo) |
| AI caption styling (animated, viral) | M | Submagic, CapCut Pro ($12-40/mo) |
| AI dubbing with lip sync | L | Descript, Captions.ai ($24-99/mo) |
| Bad take detection | M | Gling ($10-50/mo) |

### Infrastructure and migration

| Item | Effort |
|---|---|
| Transformers v5 + torch cascade (F127b) | L |
| Vite 8 panel build (F132) | S |
| IMSC 1.3 timed text (F141) | M |
| OpenColorIO 2.5 / ACES 2.0 (F142) | L |
| features.md reconciliation (F179) | M |
| Hybrid Plugin .uxpaddon (F253) | L |
| OpenTimelineIO 0.18 | S |

### UX and panel

| Item | Effort |
|---|---|
| Drag-and-drop file handling | S |
| Side-by-side before/after preview | M |
| Interactive waveform timeline | L |
| Quick Mode / focused UI | M |
| Workspace layout presets | M |
| Premiere right-click context menu (UXP) | M |

### Audio expansion

| Item | Effort |
|---|---|
| Podcast production suite (one-click polish) | L |
| Audio restoration (declip, dehum, decrackle, dewind, dereverb) | M |
| Audio fingerprinting / copyright detection | M |
| Soft subtitle embedding in containers | S |

### Video expansion

| Item | Effort |
|---|---|
| HDR/SDR tone mapping | M |
| Multi-camera audio sync | M |
| Corrupted video repair | M |
| Rolling shutter correction | L |
| Video comparison / diff tool | S |
| Motion tracking with overlays | L |

### Workflow and automation

| Item | Effort |
|---|---|
| Conditional workflow steps | M |
| Long-form to multi-short pipeline | L |
| Podcast episode bundle | M |
| Beat-synced auto-edit | L |

### Distribution expansion

LosslessCut ships on 7+ platforms (41K stars). Distribution is a competitive weakness.

| Item | Effort |
|---|---|
| pip install to PyPI | S |
| Homebrew Cask (macOS) | S |
| winget package (Windows) | S |
| Snap package (Linux) | M |

---

## Later (6-12+ months)

### Advanced AI

| Item | Effort |
|---|---|
| AI lip sync (MuseTalk 1.5 / LatentSync) | L |
| AI talking head / avatar (SadTalker / LivePortrait) | XL |
| AI foley / SFX from video (AudioLDM2) | M |
| AI scene description / alt-text | M |
| AI video summarization | M |
| AI content moderation | M |
| AI relighting (IC-Light) | L |
| Specialized upscaling models (face, low-light) | L |
| OpenVoice V2 (instant voice cloning, MIT) | M |

### Platform and ecosystem

| Item | Effort |
|---|---|
| DaVinci Resolve deep integration | L |
| FCPXML import/export | M |
| OBS Studio bridge | M |
| Web-based standalone editor | XL |
| Stream Deck integration | S |

### Professional workflows

| Item | Effort |
|---|---|
| Multi-user collaboration | L |
| Edit decision snapshots | M |
| Branching edit workflows | L |
| Script/storyboard integration | L |
| Broadcast QC / standards checker | M |
| Batch transcode with preset matrix | M |

### Advanced audio and video

| Item | Effort |
|---|---|
| Real-time voice conversion (RVC, Applio) | L |
| Stem remix / per-stem effects | M |
| Spectral audio editing | L |
| 360 video support | L |
| Spatial audio / binaural rendering | M |
| Photo + video montage builder | M |

---

## Under Consideration

| Item | Category | Why deferred |
|---|---|---|
| FastAPI migration | Architecture | Flask works at current scale |
| TypeScript panel migration | Dev-experience | Incremental; no dedicated sprint |
| Template marketplace | Plugin ecosystem | Needs community scale |
| Mobile companion app | Mobile | Low ROI vs Premiere focus |
| Cloud render offloading | Performance | Contradicts local-first |
| Real-time collaborative editing | Multi-user | OT/CRDT complexity massive |
| WebGPU ML inference | Performance | 82.7% browser support but Python more capable |
| VapourSynth bridge | Integration | GPL subprocess risk |
| OpenFX filter support | Integration | Unclear value for extension |
| Text-based editing as primary UI | UX | Descript moat; OpenCut enhances NLEs |

---

## Rejected

| Item | Reason |
|---|---|
| Mistral Voxtral TTS | CC-BY-NC incompatible with MIT |
| MatAnyone 2 production | NTU S-Lab non-commercial license |
| HunyuanVideo default-on | Tencent territory carve-outs; remains optional |
| IndexTTS-2 default integration | Non-commercial license; evaluate only |
| AGPL/GPL code reuse | License contamination; subprocess wrappers OK |
| Electron wrapper | 200+ MB overhead; no benefit |
| Built-in video player | Mission creep |
| Crypto/NFT features | No demand; reputation risk |
| Telemetry default-on | Contradicts local-first |

---

## Category coverage audit

| Category | Status | Notes |
|---|---|---|
| UX / Panel | Strong | E15 i18n, UXP WebView, drag-drop, workspace layouts |
| Performance | Good | StreamDiffusion, FlashVSR, GPU semaphore, disk preflight |
| Security | **Action needed** | PyTorch deserialization hardening closed; Flask, Pillow, and OpenEXR advisories still need follow-up; 36 RA-items |
| Reliability | Good | Job persistence, crash logging, resume, structured errors |
| Integrations | Strong | OTIO, DaVinci, MCP, OBS planned |
| Data | Good | SQLite WAL, transcript cache, FTS5, atomic writes |
| Platform/OS | Strong | Windows, macOS, Linux, Docker |
| Dev-experience | Good | CLI route, plugin SDK, skill authoring, Vitest |
| Accessibility | Good | FCC captions, WCAG 3 AD, flash detection, RTL/CJK |
| i18n/l10n | Active | 2,009 locale keys; 50+ languages via NLLB |
| Observability | Good | JSON logging, Sentry/GlitchTip, Aptabase |
| Testing | Strong | 9,000+ tests, 54% coverage, fuzz harness |
| Docs | Good | CLAUDE.md, CONTRIBUTING, SECURITY, DEVELOPMENT |
| Distribution | **Weak** | Windows-only installer. Homebrew/winget/Snap/pip planned. |
| Plugin ecosystem | Good | Manifest v1, sandbox, examples, skills |
| Mobile | Thin | No mobile story |
| Offline/Resilience | Strong | Core design principle |
| Multi-user | Thin | Review bundles + LAN portal only |
| Migration | Good | CEP->UXP dashboard, parity catalogue |

---

## Competitive landscape

| Competitor | Pricing | OpenCut advantage | OpenCut gap |
|---|---|---|---|
| Descript ($24-65/mo) | Sub | 1,534 routes; MIT; local-first | Chat conductor, Overdub |
| CapCut (free-$20/mo) | Freemium | Pro AI; Premiere; privacy | Templates, mobile |
| AutoCut ($29/mo) | Sub | 10x breadth; MIT | Timeline speed |
| Gling ($10-50/mo) | Sub | Full pipeline; local | Bad take detection |
| Opus Clip ($15/mo) | Sub | Local; full pipeline | Virality scoring |
| Runway ($12-76/mo) | Credits | Local; MIT | Gen-4.5, Aleph |
| Topaz ($299-699/yr) | Sub | Broader; MIT; extensible | 19 upscale models |
| ElevenLabs ($6-99/mo) | Sub | Local clone; MIT | Voice library scale |
| iZotope RX ($99-1,399) | Perp | MIT; extensible | Spectral editing |
| DaVinci Resolve Free | Free | Premiere integration; AI | Color; Fusion; free NLE |
| Adobe built-in ($23/mo) | Bundled | MIT; extensible | Generative Extend |
| LosslessCut (free) | Free | AI; captions; effects | 7+ distribution platforms |

### What competitors paywall (high-value signals)

1. **Voice cloning** ($6-99/mo) -- ElevenLabs, Descript, Kapwing, VEED
2. **AI dubbing with lip sync** ($24-99/mo) -- Descript, Captions.ai, Kapwing, HeyGen
3. **Virality scoring** ($15-49/mo) -- Opus Clip ($50M raised), Vizard (10M+ users)
4. **Animated captions** ($12-40/mo) -- Submagic (4M+ users), CapCut Pro
5. **Eye contact correction** ($24-30/mo) -- Descript, Captions.ai, Riverside
6. **Audio restoration** ($99-1,399) -- iZotope RX
7. **Specialized upscaling** ($299-699/yr) -- Topaz (19 models)
8. **Chat editing agent** ($24-65/mo) -- Descript Underlord, Riverside Co-Creator
9. **Generative video** ($10-95/mo) -- Runway Gen-4.5, Pika 2.5
10. **AI avatars** ($29-89/mo) -- HeyGen, Synthesia ($4B valuation)

---

## Key deadlines

| Date | Item |
|---|---|
| **2026-08-17** | FCC caption display-settings rule (F236 shipped; polish remaining) |
| **~2026-09-01** | Adobe CEP end-of-life (F252 must be ready) |
| **2026-09-01** | Apple notarization for Homebrew (F202 needs live acceptance) |
| **2026-10-31** | Python 3.10/3.11 EOL (plan 3.12+ baseline) |

---

## Autonomous Research Addendum - 2026-06-06

This addendum continues the nonstop roadmap loop from the compact v5.0
roadmap, `TODO.md`, `PROJECT_CONTEXT.md`, and archived June research notes. It
does not replace the current v5.0 restructuring already present in the working
tree.

### Cycle 1: Repository comprehension

| Area | Evidence reviewed | Finding | Roadmap impact |
|---|---|---|---|
| Roadmap source of truth | `ROADMAP.md`, `TODO.md`, `PROJECT_CONTEXT.md`, `.ai/research/2026-05-17/CONTINUE_FROM_HERE.md` | `ROADMAP.md` is now a compact strategic document, while `TODO.md` carries the active execution queue and `PROJECT_CONTEXT.md` carries pass history through Pass 264. | Keep new research as compact addenda with implementation-ready specs instead of rebuilding the removed v4 ledger. |
| Current worktree | `git status --short` | `ROADMAP.md` was already modified before this run, replacing the prior long v4 ledger with v5.0. | Append only; do not revert or overwrite the active uncommitted roadmap restructuring. |
| Recent commits | `git log -10 --oneline` | The latest ten commits are E15 i18n migration batches for export/video controls. | Continue E15 as active product debt, but roadmap research should now prioritize the RA queue that protects release trust and migration readiness. |
| Session tooling | `rtk git log -10` | `rtk` is available on this shell path and was used for the required recent-history check. | Continuation runs should keep using `rtk` for git history when available. |

### Cycle 2: Current feature inventory delta

| Area | Existing feature | Evidence | Maturity | Notes |
|---|---|---|---|---|
| UXP migration | UXP panel, WebView scaffold, UDT harness, migration dashboard | `extension/com.opencut.uxp/manifest.json`, `extension/com.opencut.uxp/bolt-webview/README.md`, `extension/com.opencut.uxp/uxp-udt-harness.json`, `PROJECT_CONTEXT.md` | High but externally gated | F252 remains gated by live Premiere UDT evidence and manifest cutover. |
| Adobe API drift tracker | Weekly `@adobe/premierepro` workflow and Python registry probe | `.github/workflows/adobe-premierepro-versions.yml`, `opencut/tools/adobe_premierepro_versions.py` | Useful but fragile | RA-31/RA-32 are confirmed by local workflow evidence. |
| GitHub issue seeding | YAML issue/label seeder with dry-run mode | `scripts/seed_github_issues.py`, `.github/labels.yml`, `.github/issue-seeds.yml` | Useful but fragile | RA-33 is confirmed: label dry-run still requires `gh` before checking `dry_run`. |
| Release CI | PR Fast and Release Full workflows | `.github/workflows/pr-fast.yml`, `.github/workflows/build.yml` | Strong tests, narrower token posture | RA-24 is closed: Release Full defaults to `contents: read`, the build matrix is read-only, and only the tag-only release-upload job receives `contents: write`. |
| Docker distribution | Multi-stage image, `.dockerignore`, compose | `Dockerfile`, `.dockerignore`, `docker-compose.yml`, README Docker section | Medium | RA-25/RA-26/RA-29/RA-30 are closed with dependency-surface, fail-closed install, HTTP-only default runtime, and build-context hygiene guards. |

### Cycle 3: Governance and release-trust specs

#### Feature: RA-31 Adobe tracker exit-code capture

**Problem:** The Adobe tracker workflow intends to notify on drift, but the
`Probe registry` step runs the drift command before writing `$GITHUB_OUTPUT`.
If GitHub's bash shell exits on the non-zero drift code before the `echo`, the
`Notify on drift` step can miss the signal even though `continue-on-error` lets
the step continue overall.

**Proposed solution:** Capture the command exit code without relying on a
post-failure statement. Use `set +e` around the probe, always write
`exit_code`, and preserve the JSON artifact even when drift is detected.

**Affected files:** `.github/workflows/adobe-premierepro-versions.yml`,
`tests/test_adobe_premierepro_versions_workflow.py` or a new workflow static
test.

**Acceptance criteria:**

- [ ] A simulated drift exit code still writes `steps.probe.outputs.exit_code`.
- [ ] The notification step condition reads the output and opens or updates one issue.
- [ ] The drift JSON artifact uploads on success, drift, parse error, and network error.

**Priority:** P1. **Effort:** S. **Confidence:** High.

#### Feature: RA-32 Adobe tracker label contract

**Problem:** The tracker workflow searches with `labels: 'f251'` and creates
issues with `['f251', 'uxp', 'tracking']`, but `.github/labels.yml` defines
`area:uxp-plugin`, `roadmap:*`, and `priority:*` labels, not those exact three.
Issue creation can fail or search can silently miss existing tracker issues.

**Proposed solution:** Either add `f251`, `uxp`, and `tracking` to
`.github/labels.yml`, or migrate the workflow to existing seeded labels such as
`area:uxp-plugin`, `roadmap:now`, and `type:chore`. Add a static test that every
workflow-created label exists in the label manifest.

**Affected files:** `.github/workflows/adobe-premierepro-versions.yml`,
`.github/labels.yml`, `tests/test_github_label_contracts.py`.

**Acceptance criteria:**

- [ ] All labels used by workflow-created issues exist in `.github/labels.yml`.
- [ ] The tracker issue search uses the same labels it creates.
- [ ] A dry-run label seed command can show the exact labels before a live apply.

**Priority:** P1. **Effort:** S. **Confidence:** High.

#### Feature: RA-33 no-`gh` label dry-run

**Problem:** `scripts/seed_github_issues.py --labels --dry-run` is documented as
the safe inspection path, but `apply_labels()` checks `_gh_available()` before
checking `dry_run`, so contributors without GitHub CLI cannot inspect intended
label commands.

**Proposed solution:** Move the `_gh_available()` check behind the dry-run
branch in `apply_labels()`. Keep `--apply` strict.

**Affected files:** `scripts/seed_github_issues.py`,
`tests/test_seed_github_issues.py`.

**Acceptance criteria:**

- [ ] `--labels --dry-run --json` succeeds when `_gh_available()` is false.
- [ ] `--labels --apply` still fails with a clear message when `gh` is absent.
- [ ] Seed dry-run behavior remains unchanged.

**Priority:** P2. **Effort:** S. **Confidence:** High.

#### Feature: RA-24 Release Full token least privilege

**Problem:** `.github/workflows/build.yml` grants `contents: write` at workflow
scope, so every matrix job and third-party action receives write-capable token
permissions even though only tag release upload steps require write access.
GitHub's own Actions docs recommend least required `GITHUB_TOKEN` access.

**Proposed solution:** Set workflow-level `permissions: contents: read`, then
move `contents: write` only to the release-upload job or split upload into a
dedicated job that depends on build artifacts.

**Affected files:** `.github/workflows/build.yml`,
`tests/test_workflow_permissions.py`.

**Status:** Closed 2026-06-06 with a read-only build matrix plus a tag-only
write-scoped `release-upload` job.

**Acceptance criteria:**

- [x] Non-upload jobs run with `contents: read`.
- [x] Release upload steps still have enough permission to publish tag assets.
- [x] Static tests fail if workflow-level `contents: write` returns.

**Priority:** P1. **Effort:** M. **Confidence:** High.

### Cycle 4: UXP/WebView migration audit

Adobe's Premiere UXP docs now position UXP as the current Premiere extensibility
platform for Premiere 25.6+, and Adobe's manifest docs say Premiere supports
manifest version `5`. The live OpenCut manifest has `id`, `name`, `version`,
`main`, host `minVersion: 25.6`, loopback network allowlist, Premiere-supported
`manifestVersion: 5`, and picker-scoped `localFileSystem: "request"`.

| Candidate | Evidence | Recommendation | Priority |
|---|---|---|---|
| RA-17 manifest schema guard | Closed 2026-06-06: live `extension/com.opencut.uxp/manifest.json` now declares `manifestVersion: 5`; `tests/test_uxp_manifest_schema.py` guards required live keys and the dormant WebView scaffold's separate v6 template. | Keep the live-vs-scaffold schema split documented until F252 WebView cutover changes the active entrypoint. | Done |
| RA-11 least-privilege filesystem | Closed 2026-06-06: live and WebView manifests declare picker-scoped `localFileSystem: "request"` and `tests/test_uxp_filesystem_permission.py` guards the open-file/open-folder boundary. | Keep direct arbitrary file APIs out of UXP until a new workflow earns a separate permission review. | Done |
| RA-19 clipboard permission | Closed 2026-06-06: live and WebView manifests declare `clipboard: "readAndWrite"`, and UXP output copy routes through `copyTextToClipboard()`. | Keep the shared helper and manifest permission in sync while copy actions remain in the UXP surface. | Done |
| RA-13 launchProcess permission | Closed 2026-06-06: live and WebView manifests declare HTTPS-only `launchProcess` schemes with no file extensions, and `tests/test_uxp_external_launch_permission.py` guards the OAuth launch helper plus no-`openPath()` contract. | Keep external launch limited to OAuth browser handoff unless a new workflow earns a separate permission review. | Done |
| RA-14 WebView permission split | Closed 2026-06-06: dormant WebView config now separates development and release manifest profiles, keeping Vite/hot-reload domains out of the release profile. | Keep final WebView packaging on the release profile once F252 cutover evidence is captured. | Done |
| F252 WebView cutover | Adobe WebView UI guidance; `bolt-webview` scaffold exists | Keep cutover blocked until live UDT capture validates the 14 direct-UXP host actions and the manifest entrypoint switch. | P0 external |

**External sources:** Adobe Premiere UXP API docs
`https://developer.adobe.com/premiere-pro/uxp/`; Adobe UXP plugin tutorial
`https://developer.adobe.com/premiere-pro/uxp/plugins/`; Adobe manifest docs
`https://developer.adobe.com/premiere-pro/uxp/plugins/concepts/manifest/`;
Adobe Bolt UXP WebView article
`https://blog.developer.adobe.com/en/publish/2026/03/introducing-webview-ui-in-bolt-uxp-build-richer-adobe-plugins-faster`.

### Cycle 5: Docker and distribution hardening

Docker's own build docs emphasize that the build context is what the builder
can access and that `.dockerignore` is the mechanism for excluding files from
that context. Docker's build-secret docs also warn against passing secrets via
build args or environment variables because they can persist.

| Candidate | Evidence | Recommendation | Priority |
|---|---|---|---|
| RA-25 Docker dependency surface | Closed 2026-06-06: `Dockerfile` installs from tracked `requirements.txt`; static tests block retired `pydub` and `deep-translator` from the Docker path. | Reopen only if Docker adds a separate optional dependency surface outside the audited requirements/pyproject policy. | Done |
| RA-29 Docker fail-closed installs | Closed 2026-06-06: Docker dependency installation no longer uses shell-form version specifiers or `|| echo` masking. | Keep dependency failures fatal unless a documented intentionally-omitted feature path is added with its own guard. | Done |
| RA-30 build-context hygiene | Closed 2026-06-06: `.dockerignore` now excludes `.env*`, key/cert/credential/log files, coverage output, `.opencut/`, and local SQLite/cache DB artifacts. | Keep Docker context hygiene tests synchronized with sensitive `.gitignore` patterns and local runtime-state conventions. | Done |
| Docker runtime parity | Closed 2026-06-06: Docker docs, Compose, and Dockerfile now agree that default containers publish the HTTP API on 5679 only; WebSocket/MCP sidecars require explicit custom profiles/services. | Keep README, compose files, and exposed runtime ports synchronized through Docker distribution tests. | Done |

**External sources:** Docker build context docs
`https://docs.docker.com/build/building/context/`; Docker build secrets docs
`https://docs.docker.com/build/building/secrets/`; Dockerfile best practices
`https://docs.docker.com/engine/userguide/eng-image/dockerfile_best-practices/`.

### Cycle 6: Competitive and market signal refresh

| Source | What it reveals | OpenCut opportunity | Priority |
|---|---|---|---|
| Descript pricing and Underlord pages | AI co-editor access is a pricing lever, not a fringe feature. | Promote F143/F144 conductor UX: natural-language plan, timeline diff, checkpoint, self-review, rollback. | P1 |
| CapCut Desktop AI page | Script-to-video, auto-reframe, and auto-captions are table-stakes for creator tools. | Bundle existing local primitives into a Magic Clips macro instead of adding another single-purpose model. | P1 |
| DaVinci Resolve 21 coverage and feature guide | Resolve is moving into hybrid photo/video, AI search, CineFocus, and local AI workflows. | Keep OpenCut differentiated through Premiere automation, local-first privacy, and export/interchange breadth; add CineFocus-style depth rack focus only after current RA queue. | P2 |
| LosslessCut GitHub | Durable demand exists for fast, local, FFmpeg-backed cutting and broad distribution. | Improve OpenCut distribution routes (pip, winget, Homebrew, Snap) and expose fast rough-cut workflows that feed Premiere timelines. | P1 |
| CapCut user complaints on Reddit | Users complain about feature availability, paywalls, and watermark/region uncertainty. | Emphasize local-first, no subscription, no watermark, and predictable feature availability in onboarding and docs. | P2 |

**External sources:** Descript pricing `https://www.descript.com/pricing`;
Descript Underlord `https://www.descript.com/underlord`; CapCut Desktop AI
`https://www.capcut.com/tools/desktop-ai-power/`; LosslessCut GitHub
`https://github.com/mifi/lossless-cut`; DaVinci Resolve 21 New Features Guide
`https://documents.blackmagicdesign.com/SupportNotes/DaVinci_Resolve_21_New_Features_Guide.pdf`.

### Cycle 7: E15 i18n migration audit

`python scripts/i18n_lint.py --json` passed with 2,273 locale keys, 1,077 HTML
consumers, 1,161 JS consumers, 2,218 unique consumers, 55 dead keys, 0 missing
keys, and 0 dead keys over the 150-key baseline. Cycle 76 later closed the
dead-key cleanup and moved the baseline to zero; Cycle 77 added JS metadata-key
consumer coverage. Remaining E15 work should target still-bare user-visible
strings and dynamic-rendering scanner gaps.

| Candidate | Evidence | Recommendation | Priority |
|---|---|---|---|
| E15 dead-key cleanup | `scripts/i18n_lint.py --json` now reports 0 dead keys and a 0-key baseline | Keep the baseline at zero and remove or wire any future dead key in the same batch that introduces it. | P2 |
| E15 scanner coverage | `scripts/i18n_lint.py` scans `data-i18n*`, direct `t(...)` calls, and supported JS key-field metadata | Continue targeted scanner coverage for dynamic `innerHTML`, option-label builders, tooltip/title helpers, and generated command-palette labels where false negatives are likely. | P2 |
| E15 roadmap status | `TODO.md` now tracks batch 172 | Keep `ROADMAP.md` status tied to linter facts, not older dead-key counts. | P1 |

### Cycle 8: UXP/WebView cutover audit

The UXP migration has more repository-side guardrails than the compact roadmap
currently shows. `docs/UXP_MIGRATION.md` records F252.1 WebView scaffold,
F252.2 host-action dispatcher, F254-F258 API helper work, F260 dashboard,
F267 UDT harness, and F252.3 strict result-capture validation. Static files show
14 direct-UXP actions in the harness, 5 safe default scenarios, 8 mutating
scenarios, and 1 file-write scenario. The remaining blocker is not another
static manifest; it is a live Premiere UDT capture with
`includeMutating: true`, followed by strict validation.

Focused pytest validation was attempted with both system Python 3.13 and 3.12,
but both lack `pytest`; the repo `.venv` points at a placeholder
`C:\Users\--\...Python311\python.exe`. No dependencies were installed during this
research-only pass.

| Candidate | Evidence | Recommendation | Priority |
|---|---|---|---|
| F252 live UDT capture | `tests/test_uxp_udt_results.py`, `extension/com.opencut.uxp/udt-smoke.js`, `docs/UXP_MIGRATION.md` | Treat WebView cutover as blocked until `window.OpenCutUXPUdtHarness.run({ includeMutating: true })` is captured in Premiere and passes `validate_uxp_udt_results`. | P0 external |
| RA-17 manifest-version guard | Closed 2026-06-06: live `manifest.json` declares `manifestVersion: 5`, while dormant `bolt-webview/uxp.config.ts` remains a separate v6 cutover template with explicit docs and tests. | Reopen only if Adobe changes Premiere's supported manifest schema or the F252 WebView entrypoint becomes live. | Done |
| RA-11 filesystem permission split | Closed 2026-06-06: live and scaffold configs use `localFileSystem: "request"` because current UXP file access is picker-scoped through open-file and open-folder dialogs. | Reopen only if OpenCut adds direct arbitrary-path UXP file reads/writes. | Done |
| RA-19 clipboard permission | Closed 2026-06-06: the live manifest and WebView scaffold declare `clipboard: "readAndWrite"`, and `tests/test_uxp_clipboard_permission.py` guards the helper/manifest contract. | Revisit only if copy flows are removed or Adobe changes the Premiere UXP clipboard permission contract. | Done |
| RA-13 launchProcess permission | Closed 2026-06-06: live and scaffold configs declare only HTTPS external-launch schemes, social OAuth launch uses a normalizing helper, and static tests reject file-launch APIs until a dedicated extension review exists. | Revisit only if OpenCut adds a non-OAuth external launch workflow. | Done |
| RA-14 WebView permission split | Closed 2026-06-06: dormant `uxp.config.ts` exports `developmentManifest` and `releaseManifest`; dev keeps Vite/hot-reload domains and `localAndRemote`, while release removes remote WebView domains and uses `localOnly`. | Keep WebView release packaging pointed at `releaseManifest` when F252 cutover moves from scaffold to active entrypoint. | Done |
| RA-20 confirmation guard | Closed 2026-06-06: UXP source no longer calls raw browser dialogs; `tests/test_uxp_confirmation_guard.py` blocks `window.alert`, `window.prompt`, `window.confirm`, and bare dialog calls. | Keep destructive UXP actions on panel-native confirmation flows unless the manifest explicitly opts into beta alerts with live evidence. | Done |

### Cycle 9: Docker/runtime parity audit

The repo has `docker-compose.yml` with both CPU and GPU services. The GPU path
is selected through `docker compose --profile gpu up opencut-server-gpu` so the
profiled service is targeted directly and the default CPU service does not
collide on port 5679. README and Dockerfile copy-paste commands now match the
committed compose file and the non-root `/home/opencut/.opencut` data path.

| Candidate | Evidence | Recommendation | Priority |
|---|---|---|---|
| RA-27 Docker GPU compose command | README previously referenced a missing `docker-compose.gpu.yml`; `docker-compose.yml` uses `profiles: [gpu]` | Closed by updating README/compose comments to `docker compose --profile gpu up opencut-server-gpu` and adding a release-smoke docs guard. | Done |
| RA-26 non-root volume docs and runtime posture | Closed 2026-06-06: Dockerfile run examples and Compose volumes use `/home/opencut/.opencut`, the image exposes only HTTP 5679 by default, and README documents WebSocket/MCP sidecars as opt-in container profiles. | Reopen only if the Docker image starts or publishes WebSocket/MCP sidecars by default. | Done |
| RA-25 Docker dependency surface | Closed 2026-06-06: Docker installs the tracked requirements file and no longer lists retired `pydub` or `deep-translator` packages. | Revisit only if Docker needs a separate feature-extra install profile with explicit advisory evidence. | Done |
| RA-29 fail-closed install | Closed 2026-06-06: the Docker dependency layer no longer masks pip failures or relies on shell-parsed requirement specifiers. | Keep the Dockerfile dependency path on requirements-file installs or quoted explicit requirements. | Done |
| RA-30 build-context hygiene | Closed 2026-06-06: `.dockerignore` mirrors `.env*`, key, cert, credential, and log ignores and adds local runtime/cache DB exclusions. | Keep `tests/test_docker_distribution_docs.py` covering secret/log and runtime-state patterns. | Done |
| Docker port posture | Closed 2026-06-06: default Docker runtime publishes HTTP 5679 only and leaves optional WebSocket 5680 / MCP 5681 sidecars to explicit custom services. | Add dedicated profiles before exposing sidecar ports from containerized runs. | Done |

### Cycle 10: GitHub Actions supply-chain audit

OpenCut's workflows use tag-pinned third-party Actions:
`actions/checkout@v4`, `actions/setup-python@v5`, `actions/setup-node@v4`,
`actions/upload-artifact@v4`, and `actions/github-script@v7`. GitHub's Actions
settings docs describe `OWNER/REPOSITORY@TAG-OR-SHA` allowlisting and explicitly
support requiring full-length SHA pins. GitHub's artifact attestation docs also
support signed build-provenance claims for binaries and container images.

| Candidate | Evidence | Recommendation | Priority |
|---|---|---|---|
| RA-23 full-SHA action pins | Closed 2026-06-06: non-local workflow `uses:` references now point at full-length SHAs with adjacent version comments, and `tests/test_workflow_action_pins.py` rejects mutable refs. | Keep workflow action updates explicit by changing both the SHA and nearby version comment through the static guard. | Done |
| RA-24 token least privilege | Closed 2026-06-06: Release Full defaults to `contents: read`, the build matrix is read-only, and the tag-only `release-upload` job is the only `contents: write` boundary. | Keep release upload authority isolated from build/test/package jobs and guard against workflow-level write-token regressions. | Done |
| RA-22 Release Full Node pin | Closed 2026-06-06: Release Full uses `actions/setup-node@v4` with Node 22 before Linux CEP panel npm gates, and PR Fast uses the same runtime. | Keep Release Full and PR Fast panel runtimes in lockstep before treating npm advisory/build evidence as deterministic release proof. | Done |
| Release provenance attestation | Closed 2026-06-06: Release Full now packages server release assets before upload, generates GitHub artifact attestations for uploaded server/Linux/Windows/SBOM subjects, and documents `gh attestation verify` commands. | Keep release uploads and attested subject paths in lockstep through `tests/test_release_provenance_attestation.py`. | Done |

**External sources:** GitHub artifact attestations
`https://docs.github.com/actions/concepts/security/artifact-attestations`;
GitHub build provenance attestation guide
`https://docs.github.com/en/actions/how-tos/security-for-github-actions/using-artifact-attestations/using-artifact-attestations-to-establish-provenance-for-builds`;
GitHub Actions repository settings and SHA allowlisting
`https://docs.github.com/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/managing-github-actions-settings-for-a-repository`;
GitHub workflow permissions syntax
`https://docs.github.com/actions/using-workflows/workflow-syntax-for-github-actions`.

### Cycle 11: SQLite/local-store safety audit

OpenCut keeps several user-local SQLite databases under `.opencut`: jobs,
operation journal, footage index, and pipeline-health metrics. The scanned
stores already use WAL mode and `synchronous=NORMAL`, but migration and
maintenance policy is inconsistent. `job_store.py` performs column-existence
based `ALTER TABLE` migrations; `journal.py` adds `forward_json` the same way;
`footage_index_db.py` and `pipeline_health.py` initialize schema directly. None
of the scanned stores use `PRAGMA user_version`, and none expose a common local
DB diagnostic/maintenance surface.

SQLite's own docs make this a good time to add explicit policy: `user_version`
is application-owned metadata, WAL databases require checkpoint thinking, large
deletes can leave free pages behind, and `VACUUM INTO` can create compact backup
copies before destructive maintenance.

| Candidate | Evidence | Recommendation | Priority |
|---|---|---|---|
| RA-37 SQLite schema versions | `opencut/job_store.py`, `opencut/journal.py`, `opencut/core/footage_index_db.py`, `opencut/core/pipeline_health.py` | Closed 2026-06-06: added a local-DB migration helper that records `user_version`, runs ordered idempotent migrations per store, and rejects newer unknown schemas with downgrade-safe errors. | P1 |
| RA-38 payload-size quotas | `jobs.result_json`, `journal.inverse_json`, and `journal.forward_json` | Closed 2026-06-06: added per-field JSON byte caps, content-addressed local spill files under `.opencut/payload_spills`, and structured spill metadata in job and journal list/detail payloads. | P1 |
| RA-39 local DB maintenance diagnostics | `cleanup_old_jobs`, `journal.clear_all`, `clear_index`, and metric purges delete rows | Closed 2026-06-06: added `opencut local-db-diagnostics`, feature-area diagnostics routes, and a shared diagnostic helper with `page_count`, `freelist_count`, WAL checkpoint status, file sizes, and recommended maintenance action. | P1 |
| RA-40 backup-before-wipe policy | `journal.clear_all`, journal DELETE route, footage-index clear/rebuild, and health reset paths perform destructive local deletes without a common backup/dry-run contract | Closed 2026-06-06: added a shared local DB maintenance helper with affected-row counts, `dry_run` results, optional `VACUUM INTO` backups, and JSONL audit entries for local SQLite destructive operations. | P1 |

**External sources:** SQLite PRAGMA docs
`https://www.sqlite.org/pragma.html`; SQLite WAL docs
`https://www.sqlite.org/wal.html`; SQLite VACUUM docs
`https://www.sqlite.org/lang_vacuum.html`.

### Cycle 12: Destructive operation and cache-wipe safety audit

OpenCut already has useful path and CSRF guardrails around many destructive
operations: `validate_path()` blocks traversal, UNC paths, alternate data
streams, and reserved device names; `is_path_within()` uses `realpath()` plus
`commonpath()` for component-aware containment; model deletion is limited to
known cache roots. The remaining gap is operator recoverability. User-visible
delete/clear/prune routes often mutate immediately, and route smoke tests mostly
assert status codes rather than preview, confirmation, backup, audit, or
containment invariants.

Comparable tool patterns support a stronger contract: Docker prune commands
prompt unless `--force` is supplied, `kubectl delete` exposes client/server
dry-run modes and warns about forced deletion consistency risks, and `git clean`
has a documented dry-run mode before removing untracked files.

| Candidate | Evidence | Recommendation | Priority |
|---|---|---|---|
| RA-41 destructive-operation plan contract | `/models/delete`, `/whisper/clear-cache`, `/plugins/uninstall`, `/cache/cleanup`, `/cache/invalidate`, `/captions/cache/clear`, `/system/temp-cleanup/sweep`, `/logs/clear`, `/presets/delete`, `/workflows/delete`, `/workflow/delete`, `/queue/clear`; tests mainly cover status codes | Closed 2026-06-06: the original named route list plus adjacent assistant/chat/undo/search/worker-pool clears now have shared dry-run plans and confirm-token enforcement, with plugin deletes also requiring typed `confirm_name`; journal clear remains covered by the local DB dry-run/backup contract. | P1 |
| RA-42 render-cache delete containment | `opencut/core/render_cache.py` writes cached outputs under `CACHE_DIR`, but `cleanup_cache()` and `invalidate_downstream()` trust persisted `index.json` `output_path` values when calling `os.unlink()` | Closed 2026-06-06: render-cache reads, cleanup, and downstream invalidation now reject corrupted index paths unless they resolve under `CACHE_DIR` and match the expected cache-key basename, preserving outside files while dropping bad index entries. | P0 |
| RA-43 plugin uninstall quarantine | `opencut/routes/plugins.py` validates plugin names and confines paths under `PLUGINS_DIR`, then `shutil.rmtree(plugin_dir)` removes the plugin immediately; archive notes already called out missing re-type confirmation | Closed 2026-06-06: uninstall now requires typed `confirm_name`, moves plugins into timestamped quarantine before unloading, and exposes quarantine list, restore, and permanent-delete endpoints. | P1 |
| RA-44 model/cache clear preview | `whisper_clear_cache()` removes matching Hugging Face entries with `ignore_errors=True` and deletes whole OpenCut/Whisper cache directories; `/models/delete` removes cache files/dirs immediately after allowed-root checks | Closed 2026-06-06: Whisper cache clear and model delete now support `dry_run`/`preview` plans with exact paths, byte counts, categories, and per-path delete errors instead of silent `ignore_errors=True` wipes. | P1 |
| RA-45 user-data delete snapshots | Preset/workflow deletes rewrite JSON files atomically, and settings export exists, but delete endpoints do not create per-item tombstones or expose undo metadata | Closed 2026-06-06: user-data mutations now write capped tombstone snapshots for presets, workflows, favorites, and assistant dismissals, with list/restore routes and audit metadata. | P2 |

**External sources:** Docker prune docs
`https://docs.docker.com/engine/manage-resources/pruning/`; Docker system prune
reference `https://docs.docker.com/reference/cli/docker/system/prune/`;
Kubernetes `kubectl delete` docs
`https://kubernetes.io/docs/reference/kubectl/generated/kubectl_delete/`;
Git `git clean` docs `https://git-scm.com/docs/git-clean/2.23.0.html`.

### Cycle 13: Caption timeline bridge and round-trip feasibility audit

OpenCut has a strong caption generation/export foundation, but the current
timeline-native bridge is split across three incomplete paths. The backend can
generate SRT, VTT, JSON, and ASS from Whisper/WhisperX-style transcripts;
`/transcript` returns editable word-level transcript segments; and
`/transcript/export` reconstructs edited segments into a `TranscriptionResult`,
runs the SRT/VTT QC gate, and writes an edited sidecar. That is a solid
outbound caption path, especially because JSON export preserves speaker,
language, confidence, word, and human-review metadata.

The missing piece is a durable caption identity and write-back model. The
current `/timeline/srt-to-captions` route is validation/normalization only: it
parses a selected SRT into `start`, `end`, and `text`, or passes supplied
segments through after cleaning the same three fields. `_parse_srt()` has useful
safety limits, including a 16 MB cap and HTML-tag stripping, but it necessarily
drops speaker labels, review flags, word IDs, style/display tokens, source
segment IDs, and the `transcript_cache_key` returned by `/captions`. The UXP
panel mirrors that limitation: `runSrtImport()` posts to
`/timeline/srt-to-captions`, then directs the user to the CEP
`ocAddNativeCaptionTrack` handoff to place the parsed cues. UXP still does not
call a host action that creates a caption track.

Host integration confirms why this needs to stay hybrid for now. The CEP host
still has `ocAddNativeCaptionTrack()`, which writes a temp SRT, imports it into
an `OpenCut Captions` bin, and has JSX mock coverage for valid, invalid, empty,
and no-project payloads. The UXP parity catalogue marks
`ocAddNativeCaptionTrack` as `cep_only`, and `extension/com.opencut.uxp/main.js`
returns `{ok:false, cepFallback:true}` with the reason "No UXP caption-track
creation API is available in the pinned parity catalogue." Current Adobe UXP
docs reviewed on 2026-06-06 expose read-oriented caption APIs on `Sequence`
(`getCaptionTrack`, `getCaptionTrackCount`) and `CaptionTrack`
(`getTrackItems`), but the scanned official reference did not surface a
documented create/import caption-track write API. That makes the near-term
architecture: UXP read/snapshot/diff, CEP or hybrid write, and backend
canonical sidecars for lossless metadata.

#### RA-46 canonical caption sidecar schema

**Priority:** P1. **Effort:** M. **Confidence:** High.

**Status:** Closed 2026-06-06. `opencut/core/caption_roundtrip.py` now writes
versioned `.opencut-captions.json` sidecars for caption exports, caption routes
return `sidecar_path`/`metadata_preserved`, and `/timeline/srt-to-captions`
uses matching sidecars to restore metadata while labeling SRT-only parses as
lossy.

**Evidence:** `/captions` returns `transcript_cache_key`; `/transcript` emits
editable segment IDs and word timings; `export_json()` preserves full transcript
metadata; `/timeline/srt-to-captions` drops everything except `start`, `end`,
and `text`.

**Recommended implementation:** Add
`opencut/core/caption_roundtrip.py` with a versioned sidecar writer/reader,
then write `<media>.opencut-captions.json` beside generated SRT/VTT/ASS/JSON
exports. Each cue should carry `caption_id`, `source_segment_id`,
`transcript_cache_key`, `source_file_hash`, `start`, `end`, `text`, `word_ids`,
`speaker`, `language`, confidence/review fields, display-setting token IDs,
export format, and optional host locators such as `sequence_guid`,
`caption_track_index`, and host track-item IDs when UXP can read them.

**Acceptance criteria:**

- [x] Generating captions writes a versioned sidecar for SRT/VTT/ASS/JSON
      exports without changing existing response fields.
- [x] The sidecar round-trips all metadata currently preserved by
      `export_json()`, including words, speaker, language, confidence, and
      review fields.
- [x] Missing or stale sidecars produce explicit warnings, not silent metadata
      loss.
- [x] Tests prove SRT-only parse remains lossy while sidecar-backed parse is
      metadata-preserving.

**Risks:** Host track-item identifiers may not be stable across Premiere
sessions; the sidecar must treat host locators as hints and fall back to
time/text matching.

#### RA-47 caption diff and apply endpoints

**Priority:** P1. **Effort:** L. **Confidence:** High.

**Status:** Closed 2026-06-06. `POST /captions/round-trip/diff` now compares
sidecar-backed or lossy caption edits, classifies text/timing/style/split/merge
and inserted/deleted changes, and `POST /captions/round-trip/apply` persists a
confirmed content-addressed revision without overwriting transcript cache state.

**Evidence:** `/transcript/export` exports edited segments but does not mutate
transcript cache/state; `/timeline/srt-to-captions` validates SRTs but does not
compare them against original transcripts or prior exports.

**Recommended implementation:** Add `POST /captions/round-trip/diff` and
`POST /captions/round-trip/apply`. Diff should accept edited SRT text/path,
edited segment JSON, or a UXP caption-track snapshot plus the canonical sidecar.
It should classify `text_changed`, `timing_changed`, `split`, `merge`,
`deleted`, `inserted`, and `style_changed` cues. Apply should require an
explicit confirmation token and persist a new transcript-edit state file or
cache revision rather than overwriting the original transcript entry.

**Acceptance criteria:**

- [x] Diff returns counts, per-cue changes, confidence, warnings, and an
      unchanged/changed summary suitable for a review UI.
- [x] Apply stores a new revision linked to the original `transcript_cache_key`
      and source file hash.
- [x] Apply is idempotent for unchanged SRT plus matching sidecar.
- [x] A no-sidecar request still works as a lossy timing/text diff and labels
      metadata preservation as unavailable.

**Risks:** SRT editors can reorder, split, or merge cues in ways that require
fuzzy matching; start with deterministic `caption_id` matching and only then
fallback to timing/text similarity.

#### RA-48 UXP caption-track read bridge

**Priority:** P1. **Effort:** M. **Confidence:** Medium-high.

**Status:** Closed 2026-06-06. `ocGetCaptionTrackSnapshot` is now a read-only
UXP direct host action with a safe-by-default UDT fixture, distinct failure
reasons for project/sequence/caption API states, and snapshot segments shaped
for `/captions/round-trip/diff`.

**Evidence:** Current Adobe UXP reference pages list
`Sequence.getCaptionTrack()`, `Sequence.getCaptionTrackCount()`, and
`CaptionTrack.getTrackItems()`; OpenCut's UXP bridge already has read actions
for sequence info and markers, but `ocAddNativeCaptionTrack` is pinned as
CEP-only.

**Recommended implementation:** Add a read-only UXP host action such as
`ocGetCaptionTrackSnapshot` that uses the active sequence, enumerates caption
tracks, calls `getTrackItems(trackItemType, includeEmptyTrackItems)`, and maps
available text/timing fields into the RA-46 sidecar/diff schema. If required
track-item text APIs are absent or undocumented at runtime, return an explicit
capability failure with the API names and Premiere version.

**Acceptance criteria:**

- [x] The action appears in the UXP direct-action manifest only when fixture
      coverage proves a non-mutating read path.
- [x] Empty projects, no active sequence, no caption tracks, and API-missing
      states return distinct, test-covered reasons.
- [x] Snapshot output can be passed directly to `/captions/round-trip/diff`.
- [x] The implementation does not claim UXP caption creation/import support
      until an official Adobe write API is documented and live-tested.

**Risks:** Adobe forum evidence indicates `getTrackItems()` parameter behavior
has been confusing; UDT must capture the exact parameter contract before this is
treated as reliable.

#### RA-49 CEP/hybrid caption write contract

**Priority:** P1. **Effort:** M. **Confidence:** High.

**Status:** Closed 2026-06-06. `ocAddNativeCaptionTrack()` now accepts legacy
segment arrays plus RA-46 sidecar/cue and caption-snapshot payloads, returns a
normalized import/placement contract, and records CEP host version details plus
placement fallback mode. UXP SRT Prep copy now points to the concrete CEP
`ocAddNativeCaptionTrack` handoff until UXP caption writes have documented API
support.

**Evidence:** CEP `importCaptions()` tries `seq.addCaptionTrack()` and
`captionTrack.insertClip()` before falling back to video-track/project-panel
import; before this item, `ocAddNativeCaptionTrack()` returned only `success`
and `captions_added`.

**Recommended implementation:** Normalize CEP/hybrid caption write results into
a richer payload: `success`, `captions_added`, `imported`,
`added_to_timeline`, `bin_name`, `caption_track_index`, `fallback_path`,
`sidecar_path`, warnings, and host version details. The UXP panel should
surface this as a deliberate "Place captions using CEP/hybrid" handoff until
UXP has a documented write API.

**Acceptance criteria:**

- [x] `ocAddNativeCaptionTrack()` accepts RA-46 sidecar-aware segment payloads
      while remaining compatible with the existing `start/end/text` array.
- [x] The result payload distinguishes project import, native caption-track
      placement, video-track fallback, and manual-drag fallback.
- [x] JSX mock tests assert the richer result contract.
- [x] UXP UI copy points to a concrete bridge action instead of a generic
      caption-placement instruction.

**Risks:** CEP caption behavior can differ by Premiere version; the contract
must record fallback mode rather than treating all successful imports as native
timeline placement.

#### RA-50 caption metadata-loss regression tests

**Priority:** P1. **Effort:** S. **Confidence:** High.

**Status:** Closed 2026-06-06. Caption metadata tests now lock SRT-only
metadata loss, sidecar-backed import/diff preservation, split/merge/insert/delete
classifications, stale sidecar export-path warnings, and no-sidecar degraded
mode.

**Evidence:** Existing tests cover SRT text/timing round trips and CEP
`ocAddNativeCaptionTrack` basics, but no scanned test asserts that speaker,
review, style, and transcript-cache metadata survive a full caption round-trip.

**Recommended implementation:** Add focused tests for RA-46 through RA-49 before
expanding UI work: route tests for sidecar creation/diff/apply, unit tests for
fuzzy cue matching, UXP static tests for snapshot action registration, and JSX
mock tests for richer CEP write results.

**Acceptance criteria:**

- [x] Tests demonstrate SRT-only parse drops non-SRT metadata.
- [x] Tests demonstrate sidecar-backed diff preserves metadata across
      edit/export/import.
- [x] Tests cover split/merge/deleted/inserted cue classifications.
- [x] Tests cover stale sidecar warnings and no-sidecar degraded mode.

**External source anchors:** Adobe Premiere UXP overview
`https://developer.adobe.com/premiere-pro/uxp/`; Premiere UXP API reference
`https://developer.adobe.com/premiere-pro/uxp/ppro_reference/`; `Sequence`
reference
`https://developer.adobe.com/premiere-pro/uxp/ppro_reference/classes/sequence/`;
`CaptionTrack` reference
`https://developer.adobe.com/premiere-pro/uxp/ppro_reference/classes/captiontrack/`;
Adobe Community caption-track parameter thread
`https://community.adobe.com/questions-729/issue-accessing-caption-items-via-captiontrack-api-in-premiere-pro-uxp-scripting-1419037`.

### Cycle 14: Shorts pipeline and Magic Clips macro composition audit

**Local code evidence:**

- `opencut/core/shorts_pipeline.py` already implements a real execution
  pipeline: transcribe, normalize transcript segments, extract highlights, trim
  clips, face-track or fixed-crop reframe, optionally burn captions, and copy
  final MP4s. It is execution-first: there is no dry-run plan, cached transcript
  gate, candidate review model, thumbnail output, platform preset identity,
  manifest, or checkpoint/resume contract.
- The live route is `opencut/routes/video_specialty.py`, not the stale
  `opencut/routes/video.py` path. `/video/shorts-pipeline` is CSRF-protected,
  rate-limited under `ai_gpu`, decorated with `@workflow_step("Running shorts
  pipeline")`, and returns an async job whose result contains `clips` and
  `total_clips`.
- `opencut/core/workflow.py` can validate sequential route workflows, call
  async endpoints through Flask's test client, poll child jobs, and check parent
  cancellation between steps. It still chains one `current_input` file path and
  `_extract_output_path()` only finds one output, so it cannot currently model a
  one-to-many Magic Clips bundle with candidates, thumbnails, captions, variants,
  and platform exports.
- `opencut/core/shorts_variants.py` and
  `opencut/routes/shorts_variants_routes.py` are the strongest local precedent:
  `plan_variants()` powers `/shorts/variants/dry-run` with descriptors and no
  rendering, while `/shorts/variants` renders the same plan shape.
- `opencut/core/virality_score.py` already exposes `rank(candidates, ...)` with
  audio-energy, transcript-hook, and visual-salience signals, but
  `generate_shorts()` does not call it before selecting/rendering clips.
- `opencut/core/export_presets.py` already defines `youtube_shorts`, `tiktok`,
  and `instagram_reels` dimensions/duration constraints. The CEP panel maps
  platform dimensions manually, while the UXP panel currently sends only max
  clips, face tracking, and burn-in caption flags to `/video/shorts-pipeline`.
- `opencut/core/thumbnail.py` and `opencut/core/thumbnail_ab.py` can score frames
  and generate thumbnail variants, but shorts pipeline results do not include
  thumbnail candidates or thumbnail-grid artifacts.
- `opencut/core/long_to_shorts.py` writes `shorts_metadata.csv`; the newer
  `shorts_pipeline` returns richer in-memory `ShortClip` data but does not
  persist an output manifest.
- The job store has resume metadata fields, but the shorts pipeline uses a temp
  directory and deletes it after the loop; completed clips are durable, while
  transcript/highlight/reframe/caption intermediates are not resumable.

**External product anchors:** Riverside Magic Clips identifies highlights, sets
pace, adds animated captions/layouts, and lets users preview/customize/export:
`https://support.riverside.com/hc/en-us/articles/12124048765981-Generate-AI-Magic-Clips`
and `https://riverside.com/magic-clips`. OpusClip positions ClipAnything as
multimodal clipping with visual/audio/sentiment cues, brand templates, and
social-account setup:
`https://help.opus.pro/docs/article/introduction-to-opusclip`,
`https://www.opus.pro/brand-templates`, and
`https://help.opus.pro/docs/article/things-you-should-set-up-first`. Descript's
Create Clips/Underlord pages emphasize AI-selected moments, user choice,
captions/layouts, aspect-ratio changes, B-roll, and editor control:
`https://www.descript.com/clips` and
`https://help.descript.com/hc/en-us/articles/36803785502221-Underlord-beta-Your-AI-co-editor-in-Descript`.
CapCut's official long-video-to-shorts pages emphasize caption templates,
duration ranges, generated titles/descriptions/hashtags, editing after
generation, and direct social export:
`https://www.capcut.com/tools/auto-video-editor`,
`https://www.capcut.com/resource/create-shorter-videos`, and
`https://www.capcut.com/resource/ai-shorts-maker`.

**Implementation direction:** Treat `shorts_pipeline.generate_shorts()` as the
rendering primitive, not the Magic Clips product boundary. Add a small
plan-first conductor that composes transcript cache, highlight extraction,
virality ranking, reframe planning, caption style selection, A/B variants,
thumbnail candidates, export presets, and job checkpoints into a single
reviewable graph. RA-10 remains the umbrella; RA-51 through RA-56 below split it
into implementable work.

#### RA-51 Magic Clips plan graph contract

Closed 2026-06-06: `opencut/core/magic_clips.py` and
`POST /video/magic-clips/plan` now emit a deterministic dry-run plan graph, and
`/video/shorts-pipeline` accepts approved plan/candidate handoffs for rendering
only the reviewed subset.

**Priority:** P1. **Effort:** M. **Confidence:** High.

**Recommended implementation:** Add `opencut/core/magic_clips.py` with
dataclasses for `MagicClipsConfig`, `MagicClipsPlan`, `MagicClipCandidate`, and
`MagicClipStep`. Add `/video/magic-clips/plan` or
`/video/shorts-pipeline/dry-run` as a sync route that returns the plan without
rendering. If transcript/highlight data is not already cached, the plan route
should return `requires_analysis` steps instead of silently starting ASR or
FFmpeg work; a separate async "analyze and plan" job can populate the cache.

**Acceptance criteria:**

- [x] Plan output contains stable IDs, source path hash, config hash, candidate
      windows, step dependencies, estimated outputs, and reasons.
- [x] Dry-run never writes rendered media and never runs expensive ASR/FFmpeg
      analysis unless an explicit `analyze=1`/async mode is requested.
- [x] Existing `/video/shorts-pipeline` can accept a plan ID or candidate IDs and
      render exactly that approved subset.
- [x] Unit tests compare deterministic JSON snapshots for cached-transcript,
      no-cache, and invalid-config scenarios.

#### RA-52 Candidate scoring and explainable selection

Closed 2026-06-06: Magic Clips plans now score candidates with deterministic
highlight, transcript-hook, duration-fit, and speaker-continuity factors while
recording `selection_reason`, `score_breakdown`, `fallback_mode`, and rejected
candidate diagnostics.

**Priority:** P1. **Effort:** M. **Confidence:** High.

**Recommended implementation:** Feed highlight windows from
`opencut.core.highlights.extract_highlights` into a ranking layer that uses
`opencut.core.virality_score.rank()` plus transcript hook text, duration fit,
speaker continuity, and optional domain detectors such as sports/highlight
modules. Persist `selection_reason`, `score_breakdown`, `fallback_mode`, and
`rejected_candidates` in the plan.

**Acceptance criteria:**

- [x] Candidate ordering is deterministic for identical transcript/media inputs.
- [x] The plan includes per-candidate reasons that can be shown in UXP/CEP
      without re-running model calls.
- [x] When LLM scoring fails or is disabled, heuristic scoring still returns
      usable candidates and clearly marks the fallback.
- [x] Tests cover tie-breaking, too-short windows, overlapping highlights, and
      malformed transcript segments.

#### RA-53 Platform preset and multi-ratio export contract

Closed 2026-06-06: Magic Clips plans and approved renders now carry platform
preset IDs from `export_presets.py`, expose social target constraints, and
render one preset-clamped, dimension-conformed output per approved platform target.

**Priority:** P1. **Effort:** M. **Confidence:** High.

**Recommended implementation:** Replace panel-side dimension maps with preset
IDs from `opencut/core/export_presets.py` and a shared option endpoint. Magic
Clips plans should support one or more platform targets per approved candidate,
with generated render steps for `youtube_shorts`, `tiktok`, `instagram_reels`,
square/social feed, or custom dimensions. Reuse `batch_reframe.py` and
`smart_reframe.py` when a multi-ratio batch is requested.

**Acceptance criteria:**

- [x] UXP and CEP send preset IDs, not ad-hoc width/height maps.
- [x] Plans expose platform constraints, target dimensions, max duration, and
      output filename templates before rendering.
- [x] Rendering enforces preset duration/dimension constraints consistently with
      `export_presets.py`.
- [x] Tests verify at least YouTube Shorts, TikTok, Reels, and square outputs.

#### RA-54 Review-board UI parity for UXP and CEP

**Priority:** P1. **Effort:** M. **Confidence:** Medium.

**Recommended implementation:** Add a Magic Clips review board in both UXP and
CEP surfaces: candidate cards with score, reason, transcript excerpt, platform
targets, caption style, thumbnail candidates, and approve/reject toggles. UXP
should reach parity with CEP's duration/platform/LLM controls, while CEP should
adopt the new dry-run plan endpoint instead of only posting the render job.

**Acceptance criteria:**

- [x] Both panels can preview a plan, approve/reject candidates, and render only
      approved candidates.
- [x] The UI clearly separates "Plan", "Analyze", and "Render" states.
- [x] Candidate controls do not require copying raw file paths between the
      Magic Clips and A/B variants panels.
- [x] Static tests cover route wiring and payload parity for UXP and CEP.

#### RA-55 Checkpointed and resumable Magic Clips jobs

**Priority:** P1. **Effort:** L. **Confidence:** Medium.

**Recommended implementation:** Persist the plan and intermediate state under a
run directory before rendering. Record transcript cache IDs, highlight IDs,
trimmed clip paths, reframe outputs, caption files, thumbnail paths, and final
exports in a manifest that can be resumed. Wire job resume metadata to resume
from the first missing/invalid step rather than restarting the whole pipeline.

**Acceptance criteria:**

- [x] Cancelled jobs preserve completed intermediates and report the next
      resumable step.
- [x] Resume skips completed steps when config/source hashes still match.
- [x] Temp cleanup does not delete files referenced by a resumable manifest.
- [x] Tests simulate cancel-after-transcribe, cancel-after-first-render, and
      source/config mismatch.

#### RA-56 Output bundle manifest and downstream handoff

**Priority:** P2. **Effort:** M. **Confidence:** High.

**Recommended implementation:** Emit `magic_clips_manifest.json` and an optional
CSV alongside final exports. Include source fingerprint, plan ID, candidate IDs,
platform preset, start/end/duration, transcript excerpts, score breakdown,
caption paths, thumbnail paths, variant paths, export paths, and any social
metadata suggestions. Let downstream timeline import/social upload features
consume this manifest instead of re-discovering files.

**Acceptance criteria:**

- [x] Every rendered clip has a manifest record with enough metadata to audit
      why it was selected and how it was rendered.
- [x] Manifest schema version is explicit and covered by fixture tests.
- [x] The manifest can represent multi-platform variants without duplicate
      candidate records.
- [x] UXP/CEP can open the output folder and display completed bundle contents
      from the manifest.

### New implementation-ready product specs

#### Feature: Timeline-native caption round trip

**Problem:** OpenCut can generate, style, export, and burn captions, but users
still need timeline-native editing and round-trip preservation when they adjust
caption timing or copy in Premiere.

**Proposed solution:** Add a caption round-trip schema that exports OpenCut
caption segments with timing, style token IDs, source transcript IDs, and review
flags, then imports edited timeline captions back into the local transcript and
caption state.

**User stories:**

- As an editor, I want to fix captions directly on the Premiere timeline, so
  OpenCut's transcript and style metadata stay synchronized.
- As a producer, I want proof that FCC/display-token choices survived export and
  reimport, so compliance review does not restart from scratch.

**UX requirements:** Surface "Export to native caption track", "Import edited
caption track", and "Compare caption changes" in the Captions/Timeline bridge.
Show changed timing, changed text, deleted captions, and style-token loss.

**Technical requirements:** Extend the existing caption export/import route set,
UXP host bridge, and transcript cache. Use deterministic fixture files for
Premiere-independent tests and UDT scenarios for live host coverage.

**Acceptance criteria:**

- [ ] A fixture export/import preserves timing, speaker label, style token, and
      review flags.
- [ ] Imported edits produce a diff before mutating transcript state.
- [ ] UXP and CEP paths either both work or the unsupported host returns a
      structured fallback reason.

**Dependencies:** F236 caption display tokens, F252/F267 UXP UDT harness,
caption cache/state.

**Risks:** Premiere native caption APIs can drift; style metadata may require an
OpenCut sidecar if the host cannot carry all tokens.

**Priority:** P1. **Effort:** L. **Confidence:** Medium.

#### Feature: Magic Clips macro

**Problem:** Competitors win creator mindshare by packaging transcript,
highlight, reframe, captions, and export into one understandable outcome. OpenCut
has the primitives but exposes many of them as separate operations.

**Proposed solution:** Add a dry-run-first macro that chains transcript,
highlight scoring, repeated-take cleanup, face reframe, styled captions,
thumbnail selection, and platform export presets into a reviewable plan.

**User stories:**

- As a creator, I want one command that turns a long recording into candidate
  Shorts/Reels/TikToks, so I can review edits instead of configuring every tool.
- As a privacy-sensitive editor, I want that workflow to run locally, so client
  footage never leaves the workstation.

**UX requirements:** Add a single "Magic Clips" quick action with plan preview,
clip count target, platform preset, caption style, and "apply to timeline" or
"export files" destinations. Include reasons for each selected moment.

**Technical requirements:** Build on existing `shorts_pipeline`, transcript
cache, LLM highlight extraction, face reframe, caption burn-in, export presets,
job queue, and checkpoint/resume metadata. Dry-run returns the planned job graph.
Cycle 14 decomposes this into RA-51 through RA-56.

**Acceptance criteria:**

- [ ] `dry_run=1` returns a deterministic graph with selected moments and
      reasons.
- [ ] The macro can run without cloud LLMs by using heuristic scoring.
- [ ] Each sub-job emits progress and can be canceled or resumed through the
      existing job system.

**Priority:** P1. **Effort:** L. **Confidence:** High.

### Research log

| Date | Cycle | Research area | Sources / files reviewed | Key findings | Roadmap changes |
|---|---|---|---|---|---|
| 2026-06-06 | Cycle 1 | Repository comprehension | `CLAUDE.md`, `PROJECT_CONTEXT.md`, `ROADMAP.md`, `TODO.md`, `.ai/research/2026-05-17/CONTINUE_FROM_HERE.md`, `git log -10` | Roadmap was already compacted to v5.0; append-only update is safest. | Added addendum and continuation state. |
| 2026-06-06 | Cycle 2 | Current feature inventory | README, TODO, UXP manifest, workflow files, Docker files | Active debt clusters are E15, UXP cutover, Adobe tracker, CI permissions, Docker hygiene. | Added feature inventory delta. |
| 2026-06-06 | Cycle 3 | Governance and CI | Adobe tracker workflow, tracker tool, label seeder, GitHub Actions docs | RA-31/32/33/24 are all confirmed with local evidence. | Added detailed specs and acceptance criteria. |
| 2026-06-06 | Cycle 4 | UXP/WebView | Adobe UXP docs, manifest docs, WebView article, OpenCut UXP manifest | Manifest and permission hardening should stay in Now/Next until WebView cutover is proven. | Added UXP migration audit. |
| 2026-06-06 | Cycle 5 | Docker/distribution | Dockerfile, `.dockerignore`, Docker docs | Container path can reintroduce retired deps and secret build-context risk. | Added Docker hardening plan. |
| 2026-06-06 | Cycle 6 | Competitive signals | Descript, CapCut, DaVinci, LosslessCut, Reddit complaints | AI co-editor, Magic Clips, timeline-native captions, and distribution remain high-value differentiators. | Added two product specs and market signal table. |
| 2026-06-06 | Cycle 7 | E15 i18n audit | `scripts/i18n_lint.py`, `tests/test_i18n_hardcoded_migration.py`, CEP panel files | Live linter passes: 2,273 keys, 2,218 consumers, 55 dead keys, 0 missing keys. | Updated E15 status and cleanup recommendations. |
| 2026-06-06 | Cycle 8 | UXP/WebView cutover | `docs/UXP_MIGRATION.md`, UXP scaffold, UDT harness/result tests, UXP manifest | Static gates are mature; remaining blocker is live UDT capture plus manifest-version/permission hardening. | Added UXP cutover audit and validation blocker. |
| 2026-06-06 | Cycle 9 | Docker/runtime parity | Dockerfile, docker-compose.yml, `.dockerignore`, `.gitignore`, README | README GPU command and Dockerfile comments drift from the actual compose/runtime contract; Docker install and build-context hygiene remain active RA items. | Added Docker/runtime parity audit. |
| 2026-06-06 | Cycle 10 | GitHub Actions supply chain | Workflows, archived Cycle 12/13 research, GitHub docs | Mutable action tags, broad Release Full write token, missing Node pin, and missing artifact attestations remain release-trust work. | Added workflow supply-chain audit. |
| 2026-06-06 | Cycle 11 | SQLite/local-store safety | `job_store.py`, `journal.py`, `footage_index_db.py`, `pipeline_health.py`, journal/job/index tests, SQLite docs | Local DBs use WAL but lack explicit `user_version`, shared maintenance diagnostics, payload caps, and backup-before-wipe policy. | Added RA-37 through RA-40. |
| 2026-06-06 | Cycle 12 | Destructive operations and cache wipes | `security.py`, `plugins.py`, `system.py`, `render_cache.py`, `transcript_cache.py`, `temp_cleanup.py`, settings/workflow/jobs/search routes, route smoke tests, Docker/Kubernetes/Git docs | Path and CSRF guards exist, but destructive routes lack a shared dry-run, confirm, backup/quarantine, and forged-index containment contract. | Added RA-41 through RA-45. |
| 2026-06-06 | Cycle 13 | Caption timeline bridge and round-trip feasibility | `captions.py`, `timeline.py`, `export/srt.py`, transcript cache/edit modules, CEP/UXP host bridges, caption tests, Adobe UXP docs | Caption generation/export is mature, but native timeline placement is CEP/hybrid-only and SRT validation drops metadata without a canonical sidecar/diff model. | Added RA-46 through RA-50 and refreshed RA-09 implementation direction. |
| 2026-06-06 | Cycle 14 | Shorts pipeline and Magic Clips macro composition | `shorts_pipeline.py`, `video_specialty.py`, workflow core/routes, `shorts_variants.py`, `virality_score.py`, `export_presets.py`, thumbnail modules, UXP/CEP panel paths, Riverside/OpusClip/Descript/CapCut docs | OpenCut has the rendering primitives and a variant dry-run precedent, but no Magic Clips plan graph, explainable candidate board, preset-driven multi-platform contract, resumable intermediates, or output manifest. | Added RA-51 through RA-56 and refined RA-10 as the parent macro. |
| 2026-06-06 | Cycle 15 | Adobe/NPM tracker hardening | `opencut/tools/adobe_premierepro_versions.py`, `.github/workflows/adobe-premierepro-versions.yml`, `.github/labels.yml`, `scripts/seed_github_issues.py`, npm registry live dist-tags, GitHub Actions workflow docs | Live Adobe npm tags now include `beta=26.3.0-beta.85` and `release-26.2=26.2.1`; GitHub bash steps run with `-e`, so drift exit codes must be captured before a success exit; tracker labels need one shared search/create contract. | Closed RA-16, RA-31, RA-32, and RA-33 with schema v2 tracking, workflow and label tests, and release-smoke coverage. |
| 2026-06-06 | Cycle 16 | Distribution packaging and Docker docs | README, Dockerfile, docker-compose.yml, LosslessCut downloads, Homebrew Cask docs, Microsoft WinGet docs, Snapcraft docs, PyPI trusted publishing docs | Broad distribution requires stable release artifacts, checksums, silent-install metadata, and package-manager-specific manifests; the immediate repo bug was a missing Docker GPU compose override and root-home Docker run examples. | Closed RA-27 with committed GPU profile commands, non-root Docker run examples, Compose config cleanup, and release-smoke docs coverage. |
| 2026-06-06 | Cycle 17 | Test environment repair guard | `.venv`, `py -3.12`, `scripts/bootstrap_check.py`, `tests/test_bootstrap_check.py`, README Testing | The repo `.venv` can pass metadata bootstrap while lacking pytest/dev tooling, so test runs need an explicit dev-import check and repair command. | Added `bootstrap_check.py --dev`, README `.venv` repair commands, and focused bootstrap tests. |
| 2026-06-06 | Cycle 18 | README generated-count drift gate | README, `scripts/check_doc_sizes.py`, route manifest, live panel/source files, root test files | README badges were already described as generated, but prose, architecture diagrams, and project-structure comments could still drift from route/module/test truth. | Closed RA-28 by extending doc-size checks to README non-badge route, module, blueprint, panel line-count, and root test-file claims. |
| 2026-06-06 | Cycle 19 | Release SBOM fidelity | `scripts/sbom.py`, `.github/workflows/build.yml`, `tests/test_release_sbom.py`, `tests/test_sbom_completeness.py`, CycloneDX and GitHub artifact-attestation docs | The release SBOM is useful as a declared inventory, but it should not look like a resolved installed-environment vulnerability inventory. | Closed RA-35 by renaming the release SBOM path/artifact and adding declared-only CycloneDX metadata plus lockfile audit-target evidence. |
| 2026-06-06 | Cycle 20 | Local SQLite schema versioning | `opencut/local_db_migrations.py`, `job_store.py`, `journal.py`, `footage_index_db.py`, `pipeline_health.py`, local DB migration tests | The local stores had idempotent schema creation but no durable SQLite `user_version` boundary or future-version rejection. | Closed RA-37/RA-05 with explicit per-store schema versions, ordered migrations, and downgrade-safe unknown-schema errors. |
| 2026-06-06 | Cycle 21 | Local SQLite payload spillover | `opencut/local_db_payloads.py`, `job_store.py`, `journal.py`, job/journal tests | Job result and journal inverse/forward payloads could grow without a SQLite row-size boundary. | Closed RA-38/RA-07 with per-field caps, content-addressed spill files, and API-visible spill metadata. |
| 2026-06-06 | Cycle 22 | Local SQLite maintenance diagnostics | `opencut/local_db_diagnostics.py`, `opencut/cli.py`, local DB and CLI tests | The stores had cleanup paths but no shared operator-visible page/freelist/WAL/file-size diagnostic. | Closed RA-39/RA-08 with read-only CLI and route diagnostics plus a reusable SQLite diagnostic helper. |
| 2026-06-06 | Cycle 23 | Local SQLite destructive maintenance safety | `opencut/local_db_maintenance.py`, `job_store.py`, `journal.py`, `footage_index_db.py`, `pipeline_health.py`, destructive route tests | Journal, job cleanup, footage-index clear, and health reset operations could delete local SQLite rows without a shared preview, backup, or audit contract. | Closed RA-40/RA-06 with dry-run affected-row counts, optional compact backups, JSONL audit records, and route-visible metadata. |
| 2026-06-06 | Cycle 24 | Render-cache delete containment | `opencut/core/render_cache.py`, `tests/test_render_cache_safety.py`, platform cache tests | A forged render-cache `index.json` could point cleanup or downstream invalidation at files outside `CACHE_DIR`. | Closed RA-42 by validating resolved cache output paths against the cache root and cache-key basename before any unlink. |
| 2026-06-06 | Cycle 25 | Plugin uninstall quarantine | `opencut/routes/plugins.py`, generated route manifests, plugin quarantine tests | Plugin uninstall removed plugin directories immediately after path validation, with no restore window or typed confirmation. | Closed RA-43 with quarantine move/unload ordering, restore/permanent-delete endpoints, and route-visible metadata. |
| 2026-06-06 | Cycle 26 | Model/cache clear preview | `opencut/routes/system.py`, model cache preview tests | Whisper cache clear used broad best-effort deletes and model delete mutated immediately without an exact preview plan. | Closed RA-44 with dry-run/preview plans for Whisper cache and model-cache deletion plus per-path error reporting. |
| 2026-06-06 | Cycle 27 | User-data tombstone snapshots | `opencut/user_data.py`, settings/system/workflow routes, tombstone tests | Presets, workflows, favorites, and assistant dismissals could be removed or replaced without restore metadata. | Closed RA-45 with capped tombstone snapshots, restore metadata, and user-data restore routes. |
| 2026-06-06 | Cycle 28 | Destructive clear confirmation plans | `opencut/security.py`, `jobs_routes.py`, `settings.py`, `system.py`, `captions.py`, destructive-operation/cache tests | User-visible clear/delete routes still mutated immediately with no shared plan/confirm-token contract. | Advanced RA-41 with shared destructive-plan helpers plus dry-run and confirmation-token enforcement for queue, log, caption-cache, Whisper-cache, and model-cache clears. |
| 2026-06-06 | Cycle 29 | Render-cache and temp cleanup confirmation plans | `opencut/core/render_cache.py`, `opencut/core/temp_cleanup.py`, `platform_infra_routes.py`, `wave_f_routes.py`, destructive-operation tests | Render-cache cleanup/invalidation and manual temp cleanup sweeps could mutate files from route calls without the shared confirmation-token contract. | Advanced RA-41 with non-mutating render-cache and temp-cleanup plans plus confirmation-token enforcement for `/cache/cleanup`, `/cache/invalidate`, and `/system/temp-cleanup/sweep`. |
| 2026-06-06 | Cycle 30 | Plugin and user-data delete confirmation plans | `opencut/routes/plugins.py`, `settings.py`, `workflow.py`, `user_data.py`, plugin/user-data/workflow tests | Plugin uninstall/quarantine delete and tombstone-backed preset/workflow deletes still relied on typed names or restore snapshots without the shared dry-run token contract. | Advanced RA-41 with shared dry-run plans and confirmation-token enforcement for `/plugins/uninstall`, `/plugins/quarantine/delete`, `/presets/delete`, `/workflows/delete`, and `/workflow/delete`; closure scan found adjacent clear/cleanup routes for the next pass. |
| 2026-06-06 | Cycle 31 | Adjacent state-clear confirmation plans | `opencut/routes/system.py`, `workflow_dev_routes.py`, `search.py`, `footage_index_db.py`, destructive-operation/user-data/workflow tests | Assistant dismissal clears, chat clears, undo-history clears, and search cleanup could mutate in-memory or local-index state without the shared review/confirm-token contract. | Advanced RA-41 with dry-run plans and confirmation-token enforcement for `/assistant/dismiss-clear`, `/chat/clear`, `/api/undo/clear`, and `/search/cleanup`; worker-pool cleanup remains the next process-lifecycle audit target. |
| 2026-06-06 | Cycle 32 | Worker-pool cleanup confirmation plan | `opencut/routes/architecture_routes.py`, `tests/test_architecture.py` | Worker-pool cleanup can terminate active worker processes without a dry-run target list or confirm-token review. | Closed RA-41 by adding active-worker dry-run plans and confirmation-token enforcement to `/architecture/worker-pool/cleanup`; final scan leaves journal clear under the existing local DB dry-run/backup contract. |
| 2026-06-06 | Cycle 33 | Optional `[all]` advisory policy | `pyproject.toml`, `opencut/tools/pip_audit_extras.py`, `docs/PYTHON_ADVISORIES.md`, dependency/release-smoke tests, README | `pyproject[all]` pulled Torch/Transformers through WhisperX, Demucs, RealESRGAN/GFPGAN, pyannote.audio, and TransNetV2, then failed pip-audit with five unallowed findings. | Closed RA-15 by keeping `opencut[all]` as the release-audited convenience lane, moving Torch/Transformers-backed packages to explicit `opencut[torch-stack]` or narrower feature extras, and verifying `pyproject[all]` has zero advisories. |
| 2026-06-06 | Cycle 34 | UXP manifest schema guard | Adobe Premiere UXP manifest docs, `extension/com.opencut.uxp/manifest.json`, Bolt/WebView scaffold, UXP migration docs/tests | Adobe docs list `manifestVersion` as required and Premiere-supported version 5, while the live manifest omitted it and the dormant WebView scaffold declares version 6. | Closed RA-17 by declaring `manifestVersion: 5` in the live UXP manifest, documenting the live-vs-scaffold schema split, and adding `tests/test_uxp_manifest_schema.py`. |
| 2026-06-06 | Cycle 35 | UXP deprecated API sentinel | Adobe Premiere UXP API changelog, `extension/com.opencut.uxp/main.js`, `extension/com.opencut.uxp/bolt-webview/`, `tests/test_uxp_deprecation_sentinel.py` | Adobe deprecates older Clipboard APIs, object-form clipboard writes, and legacy `uxpvideo*` video events; OpenCut currently avoids them, but no static guard protected the UXP/WebView cutover path. | Closed RA-18 by adding a UXP/WebView source sentinel that fails on deprecated Clipboard APIs, object-form `writeText`, or legacy `uxpvideo*` event names while preserving the supported string clipboard write path. |
| 2026-06-06 | Cycle 36 | UXP clipboard permission and fallback | Adobe Premiere UXP clipboard and manifest docs, `extension/com.opencut.uxp/manifest.json`, `extension/com.opencut.uxp/main.js`, Bolt/WebView scaffold | Adobe defaults clipboard access to unavailable unless `requiredPermissions.clipboard` is declared. OpenCut writes copied output text but the manifests lacked the permission and the copy path handled async denial inline. | Closed RA-19 by declaring clipboard `readAndWrite` in both manifest surfaces, centralizing output copy through `copyTextToClipboard()`, and adding manifest/helper regression tests. |
| 2026-06-06 | Cycle 37 | UXP confirmation guard | Adobe Premiere UXP API changelog, `extension/com.opencut.uxp/main.js`, `extension/com.opencut.uxp/manifest.json`, UXP confirmation tests | Adobe keeps browser alert/prompt/confirm APIs behind a beta `enableAlerts` flag. OpenCut used raw `window.confirm` for search-index clearing without a manifest feature-flag decision. | Closed RA-20 by replacing the raw dialog with an inline second-click panel confirmation, keeping `enableAlerts` disabled, and adding a static raw-dialog guard. |
| 2026-06-06 | Cycle 38 | Docker dependency and context hardening | Docker build/context docs, pip requirement-specifier docs, `Dockerfile`, `.dockerignore`, `requirements.txt`, Docker guard tests | The Docker image still carried a hand-written optional dependency install list that could reintroduce retired packages, shell-parse unquoted specifiers, and mask pip failures, while `.dockerignore` did not explicitly block local secrets/logs/runtime DB state before `COPY . /app`. | Closed RA-25/RA-29/RA-30 by installing from tracked `requirements.txt`, keeping pip failures fatal, mirroring sensitive ignore patterns, excluding local runtime/cache DB artifacts, and extending Docker distribution guard tests. |
| 2026-06-06 | Cycle 39 | Docker runtime parity | Docker port-publishing docs, Dockerfile `EXPOSE`, `docker-compose.yml`, README Docker quick start, WebSocket/MCP sidecar evidence | Dockerfile and product docs mentioned WebSocket/MCP sidecar ports, but the image does not install/start/publish those sidecars by default. Publishing 5680/5681 would imply support the default container does not provide. | Closed RA-26 by documenting Docker as HTTP 5679 only by default, keeping sidecars opt-in, aligning Dockerfile/Compose/README, and extending Docker distribution tests to guard the port posture. |
| 2026-06-06 | Cycle 40 | Release Full Node runtime pin | GitHub Actions setup-node docs, `.github/workflows/build.yml`, `.github/workflows/pr-fast.yml`, panel CI gate tests | Release Full ran Linux CEP panel npm gates without the explicit Node 22 setup that PR Fast already used, so release evidence could drift with runner defaults. | Closed RA-22 by adding a Linux-only Node 22 setup step before the Release Full CEP panel gates and a regression test that compares the PR Fast and Release Full runtime pins. |
| 2026-06-06 | Cycle 41 | Release smoke Ruff import-order cleanup | Release-smoke Ruff gate, `opencut/routes/__init__.py`, package import blocks, route manifest and collision tests | The broader release-smoke Ruff gate failed on 17 existing `I001` import-order findings, including the blueprint import block. | Restored the Ruff gate with mechanical import ordering and rechecked route-manifest plus route-collision invariants. |
| 2026-06-06 | Cycle 42 | Release Full token permissions | `.github/workflows/build.yml`, release upload steps, workflow permission tests, SBOM workflow tests | Release Full still granted `contents: write` at workflow scope, so build/test/package matrix jobs and third-party actions received write-capable tokens even though only tag release uploads needed them. | Closed RA-24 by defaulting the workflow and build matrix to `contents: read`, moving all `gh release upload` calls into a tag-only `release-upload` job with `contents: write`, and adding static permission guards. |
| 2026-06-06 | Cycle 43 | Python 3.13 classifier retraction | `pyproject.toml`, CI workflow Python versions, dependency-surface tests, release-smoke pytest-fast list | Package metadata advertised Python 3.13, but every committed GitHub Actions lane still installs Python 3.12. | Closed RA-21 by removing the untested classifier until a CI lane proves it and adding a metadata guard that blocks the classifier without a matching workflow lane. |
| 2026-06-06 | Cycle 44 | GitHub Actions SHA pins | `.github/workflows/*.yml`, workflow action tag SHAs, panel/workflow permission tests | Workflow `uses:` references still pointed at mutable major tags such as `actions/checkout@v4`, leaving release/signing workflows dependent on tag movement. | Closed RA-23 by pinning every non-local action ref to a full SHA, preserving adjacent version comments, and adding a release-smoke guard against mutable action refs. |
| 2026-06-06 | Cycle 45 | Release artifact provenance attestations | GitHub artifact attestation docs, `actions/attest` README, `.github/workflows/build.yml`, release provenance tests | Release Full uploaded packaged artifacts and the declared SBOM without a signed provenance claim for the exact uploaded files. | Closed the provenance follow-up by adding pinned `actions/attest@v4`, least-extra attestation permissions, pre-upload server packaging, verification docs, and release-smoke static guards. |
| 2026-06-06 | Cycle 46 | CEP UNC/HGFS-safe Node commands | `extension/com.opencut.panel/package.json`, `panel-node-gate.ps1`, panel advisory/build docs, release-smoke tests | Documented panel npm gate commands could be launched from Windows shared-folder paths where `cmd.exe` falls back to `C:\Windows`, causing relative `scripts/*.mjs` paths to resolve incorrectly. | Closed RA-36 by adding Windows-safe `:win` aliases that locate the wrapper from `%INIT_CWD%`; the wrapper then executes the Node scripts from `$PSScriptRoot`, with docs and release-smoke coverage for the shared-folder entry points. |
| 2026-06-06 | Cycle 47 | Request IDs in typed error bodies | `opencut/errors.py`, `opencut/server.py`, request-correlation middleware, hardening tests | Structured JSON error bodies echoed codes and suggestions but omitted the generated server request ID, forcing operators to correlate from response headers alone. | Closed RA-04 by enriching centralized typed error bodies with the generated request ID, routing direct server typed errors through the shared helper, and adding release-smoke coverage for `error_response`, `OpenCutError`, `safe_error`, and built-in error handlers. |
| 2026-06-06 | Cycle 48 | Direct typed error logging | `opencut/errors.py`, request ID tests, typed-error logging tests, release-smoke list | `OpenCutError` and direct `error_response` paths returned useful JSON but did not reliably emit a structured log record with the code, status, request ID, method, path, and caller context. | Closed RA-03 by centralizing typed-error log records, preserving single exception logs for `safe_error`, and adding release-smoke coverage for raised `OpenCutError`, direct `error_response`, and `safe_error(OpenCutError)` paths. |
| 2026-06-06 | Cycle 49 | Python dependency and lint alignment | `pyproject.toml`, `requirements.txt`, dependency-surface tests, Ruff import ordering | Ruff still targeted Python 3.9 despite the package floor being Python 3.11+, and the installable `requirements.txt` surface had looser bounds than the audited `pyproject.toml` core/standard declarations. | Closed RA-01/RA-02 by setting Ruff to `py311`, syncing core/standard requirement bounds, and adding drift guards that derive the lint target from `requires-python` and ensure `requirements.txt` contains the `pyproject.toml` core plus standard dependency surface. |
| 2026-06-06 | Cycle 50 | CEP i18n workflow preset shell | CEP `index.html`, `en.json`, i18n hardcoded-migration tests | The Export Workflow Presets card still had bare static shell strings for preset/library status, custom workflow controls, and step selector options despite dynamic workflow text being localized. | Advanced E15 to batch 154 by wiring those static strings through `data-i18n*` attributes and locale keys; the drift gate now reports 2,295 keys, 2,242 consumers, 53 dead keys, and 0 missing keys. |
| 2026-06-06 | Cycle 51 | Caption round-trip sidecars | `opencut/core/caption_roundtrip.py`, caption export routes, timeline SRT parser, caption metadata tests | Native UXP caption-track writes are still not documented, and SRT-only parsing drops speaker, word, language, review, cache, and style metadata needed for editable Premiere timeline round trips. | Closed RA-46 under RA-09 by writing versioned caption sidecars, returning sidecar metadata from caption exports, and enriching `/timeline/srt-to-captions` output from matching sidecars while explicitly warning when SRT-only metadata is unavailable. |
| 2026-06-06 | Cycle 52 | Caption round-trip diff/apply | `opencut/core/caption_roundtrip.py`, `/captions/round-trip/*`, route manifest, caption metadata tests | Sidecars preserved metadata, but there was still no API for reviewing timeline edits or storing a confirmed transcript revision after an SRT/UXP caption-track round trip. | Closed RA-47 by adding sidecar-backed and lossy diff support, confirmation-token guarded apply, content-addressed revision storage, and route/manifest/test coverage for changed, unchanged, no-sidecar, and idempotent apply flows. |
| 2026-06-06 | Cycle 53 | UXP caption-track snapshot read bridge | `extension/com.opencut.uxp/main.js`, UXP UDT harness manifests, UXP host-action tests | RA-47 could accept UXP caption-track snapshots, but the UXP bridge had no read-only caption-track action and still treated native captions as CEP-only write work. | Closed RA-48 by adding `ocGetCaptionTrackSnapshot`, distinct read failure reasons, diff-compatible snapshot segment payloads, and a safe-by-default UDT scenario while keeping caption creation/import unsupported in UXP. |
| 2026-06-06 | Cycle 54 | CEP/hybrid caption write contract | `extension/com.opencut.panel/host/index.jsx`, `tests/jsx_mock.js`, UXP SRT Prep copy | Caption sidecars and UXP snapshots were ready, but the CEP caption writer returned only a thin success/count payload and the UXP handoff still described a generic caption flow. | Closed RA-49 by normalizing CEP caption import/write placement results, accepting sidecar-aware payloads, covering native/video/project/manual modes in the JSX mock, and naming the CEP `ocAddNativeCaptionTrack` handoff in UXP. |
| 2026-06-06 | Cycle 55 | Caption metadata-loss regression fixtures | `tests/test_caption_language_confidence.py`, caption round-trip routes/core | RA-46 through RA-49 shipped the sidecar/diff/snapshot/write pieces, but the regression suite still lacked a consolidated proof that metadata loss and preservation boundaries stay explicit. | Closed RA-50 and RA-09 by adding fixtures for SRT-only metadata loss, sidecar-backed import/diff preservation, split/merge/insert/delete classifications, stale sidecar warnings, and no-sidecar degraded diff mode. |
| 2026-06-06 | Cycle 56 | Sequence-index host locators | `opencut/core/sequence_index.py`, `opencut/routes/sequence_index_routes.py`, `tests/test_sequence_index.py`, Adobe UXP marker/sequence docs | Sequence Index ratings/tags were keyed by clip path only, CEP sequence info used snake-case track keys, and marker payloads were counted but not returned with reusable locator metadata. | Added stable `locator_id` and `host_locators` fields to Sequence Index rows, preserved them through filter route round-trips, made locator-keyed ratings/tags override path fallbacks, propagated sequence GUIDs, returned normalized marker locator payloads, and accepted CEP `video_tracks`/`audio_tracks`. |
| 2026-06-06 | Cycle 57 | Magic Clips plan graph | `opencut/core/magic_clips.py`, `opencut/core/shorts_pipeline.py`, `opencut/routes/video_specialty.py`, generated route/MCP manifests, Magic Clips tests | The shorts pipeline rendered directly from selected highlights, but RA-51 needed a reviewable dry-run graph and a way to render only approved candidates. | Closed RA-51 with stable Magic Clips plan/candidate/step IDs, source/config hashes, estimated platform outputs, analysis-required fallback steps, and approved-candidate render handoff support. |
| 2026-06-06 | Cycle 58 | Magic Clips candidate scoring | `opencut/core/magic_clips.py`, `tests/test_magic_clips.py` | RA-51 exposed candidates, but the plan lacked explainable ordering, score breakdowns, fallback labels, and rejected-candidate evidence for short, overlapping, or malformed inputs. | Closed RA-52 with deterministic heuristic scoring, per-candidate selection reasons, fallback mode metadata, score breakdowns, rejected-candidate diagnostics, and focused regression tests. |
| 2026-06-06 | Cycle 59 | Magic Clips platform presets | `opencut/core/shorts_pipeline.py`, `opencut/routes/video_specialty.py`, `tests/test_magic_clips.py` | Magic Clips plans exposed preset IDs, but approved rendering still used ad-hoc width/height settings and produced only one output per candidate. | Closed RA-53 by deriving render targets from `export_presets.py`, passing platform IDs from approved plans into `/video/shorts-pipeline`, clamping output durations, conforming preset dimensions, emitting target dimensions, and testing YouTube Shorts, TikTok, Reels, and square feed outputs. |
| 2026-06-06 | Cycle 60 | Magic Clips review-board parity | CEP and UXP panel HTML/JS/CSS, `tests/test_magic_clips_panel_ui.py` | CEP still went straight to render and UXP lacked duration/platform/caption controls plus approved-candidate review, so users could not preview the dry-run plan or render only approved clips from both panel surfaces. | Closed RA-54 by adding CEP and UXP Magic Clips review boards, dry-run plan buttons, approved-only render actions, preset/caption/LLM payload parity, Plan/Analyze/Render status text, and static route/payload tests. |
| 2026-06-06 | Cycle 61 | Magic Clips checkpointed resume | `opencut/core/shorts_pipeline.py`, `opencut/routes/video_specialty.py`, `tests/test_magic_clips.py`, `tests/test_job_resume.py` | Reviewed Magic Clips renders still relied on temp intermediates and generic job resume metadata, so an interrupted run could not preserve transcript/highlight/render state or skip already-completed clips. | Closed RA-55 by writing a versioned run manifest, keeping reviewed intermediates under a run directory, marking the shorts route resumable, storing manifest paths in job metadata and responses, resuming only when source/config hashes match, and testing cancel-after-transcribe, cancel-after-first-render, and config mismatch paths. |
| 2026-06-06 | Cycle 62 | Magic Clips output bundle handoff | `opencut/core/shorts_pipeline.py`, `opencut/routes/video_specialty.py`, CEP/UXP Magic Clips panel code, `tests/test_magic_clips.py`, `tests/test_magic_clips_panel_ui.py` | Magic Clips reviewed renders produced files and run checkpoints, but downstream tools still had to rediscover exports and could not consume a grouped candidate/variant bundle. | Closed RA-56 by writing `magic_clips_manifest.json` plus CSV handoff files, grouping multi-platform variants under one candidate, surfacing bundle paths/payloads through the route and clip results, and rendering completed bundle contents in CEP and UXP review boards. |
| 2026-06-06 | Cycle 63 | UXP external launch permission | `extension/com.opencut.uxp/manifest.json`, `extension/com.opencut.uxp/main.js`, `extension/com.opencut.uxp/bolt-webview/`, `docs/UXP_MIGRATION.md`, `tests/test_uxp_external_launch_permission.py` | The UXP social OAuth flow called `shell.openExternal()` but the manifests did not declare `launchProcess`, and the WebView scaffold allowed generic http(s) URL launches without an explicit no-file-launch contract. | Closed RA-13 by declaring HTTPS-only launch schemes with an empty extension allowlist, routing OAuth browser handoff through HTTPS normalization and manual fallback, aligning the WebView wrapper, and adding static guards against broad schemes or `openPath()` usage. |
| 2026-06-06 | Cycle 64 | UXP filesystem permission | `extension/com.opencut.uxp/manifest.json`, `extension/com.opencut.uxp/main.js`, `extension/com.opencut.uxp/bolt-webview/uxp.config.ts`, `docs/UXP_MIGRATION.md`, `docs/UXP_MACOS_HTTP.md`, `tests/test_uxp_filesystem_permission.py` | UXP file access already went through `getFileForOpening()` and `getFolder()` pickers, but both manifest surfaces still requested broad `localFileSystem: "fullAccess"`. | Closed RA-11 by narrowing both manifest surfaces to `localFileSystem: "request"`, documenting the picker-scoped boundary, and adding static guards that reject direct filesystem APIs until a separate permission review exists. |
| 2026-06-06 | Cycle 65 | UXP WebView permission profiles | `extension/com.opencut.uxp/bolt-webview/uxp.config.ts`, `extension/com.opencut.uxp/bolt-webview/README.md`, `docs/UXP_MIGRATION.md`, `tests/test_uxp_webview_permission_split.py`, `tests/test_uxp_webview_scaffold.py` | The dormant WebView scaffold carried one dev-shaped permission profile with Vite domains, hot-reload WebSocket domains, and `localAndRemote` messaging, so release packaging had no static boundary for local-only WebView content. | Closed RA-14 by exporting development and release manifest profiles, keeping hot reload/Vite domains dev-only, using `localOnly` release messaging with no remote WebView domains, and adding static guards for the split. |
| 2026-06-06 | Cycle 66 | CEP i18n deliverables and settings shell | CEP `index.html`, `en.json`, `tests/test_i18n_hardcoded_migration.py` | E15 had the Export Workflow Presets shell localized, but Export Deliverables, LLM settings, preset diagnostics, and related controls still carried static English shell copy. | Advanced E15 through batches 155 and 156 by wiring those controls through `data-i18n*` hooks and locale keys; the drift gate now reports 2,315 keys, 2,267 consumers, 48 dead keys, and 0 missing keys. |
| 2026-06-06 | Cycle 67 | UXP Hybrid package validator | Adobe UXP Hybrid packaging docs, `opencut/core/uxp_hybrid_package.py`, `opencut/tools/validate_uxp_hybrid_package.py`, `tests/test_uxp_hybrid_package.py` | Adobe's Hybrid plugin rules require manifest v6+, `addon.name`, `requiredPermissions.enableAddon`, and strict `.uxpaddon` placement for mac arm64, mac x64, and win x64; OpenCut had no repository-side validator before adding native addons. | Closed RA-12 by adding a static validator and CLI for unpacked UXP bundles, keeping the live UXP manifest valid as non-hybrid, allowing independent partial-architecture warnings, failing Marketplace layout gaps, and wiring the guard into release-smoke pytest-fast. |
| 2026-06-06 | Cycle 68 | CEP i18n templates and model inventory shell | CEP `index.html`, `en.json`, `tests/test_i18n_hardcoded_migration.py` | The Settings Project Templates and AI Models cards still had static English descriptions, labels, placeholders, ARIA/title attributes, and idle model-inventory status copy despite dynamic template/model feedback already being localized. | Advanced E15 to batch 157 by wiring those static strings through `data-i18n*` hooks, reusing existing template/form/refresh keys where possible; the drift gate now reports 2,324 keys, 2,279 consumers, 45 dead keys, and 0 missing keys. |
| 2026-06-06 | Cycle 69 | Magic Clips downstream bundle reuse | `opencut/core/shorts_pipeline.py`, `opencut/routes/video_specialty.py`, `opencut/routes/timeline.py`, `opencut/routes/system.py`, `tests/test_magic_clips.py` | Magic Clips bundle manifests existed, but timeline and social consumers still needed a canonical way to consume grouped candidate/output metadata without rediscovering files. | Added a bundle-root-checked downstream handoff with timeline import records and social upload payloads, returned it from `/video/shorts-pipeline`, exposed `/timeline/magic-clips-import-plan`, and added `/social/upload` dry-run bundle planning. |
| 2026-06-06 | Cycle 70 | CEP i18n engine routing/live updates shell | CEP `index.html`, `en.json`, `tests/test_i18n_hardcoded_migration.py` | Settings Engine Routing and Live Updates Bridge cards still carried static English descriptions, refresh labels, idle hints, status labels, listener counts, and bridge action buttons despite their dynamic runtime states already using locale keys. | Advanced E15 to batch 158 by wiring those shell strings through `data-i18n*` hooks, reusing existing engine/WebSocket keys where possible; the drift gate now reports 2,333 keys, 2,288 consumers, 45 dead keys, and 0 missing keys. |
| 2026-06-06 | Cycle 71 | CEP i18n audio/zoom and GPU recommendation shell | CEP `index.html`, `en.json`, `tests/test_i18n_hardcoded_migration.py` | Settings Audio & Zoom Defaults and GPU Recommendation controls still had bare-English labels, easing options, and action buttons while the surrounding settings shell was localized. | Advanced E15 to batch 159 by wiring those labels/options/buttons through `data-i18n` hooks and locale keys; the drift gate now reports 2,345 keys, 2,300 consumers, 45 dead keys, and 0 missing keys. |
| 2026-06-06 | Cycle 72 | CEP i18n shortcut/About shell | CEP `index.html`, `en.json`, `tests/test_i18n_hardcoded_migration.py` | The Settings keyboard shortcut reference and About links still rendered static English labels, and three shortcut keys were already in the locale file but unused by the static shell. | Advanced E15 to batch 160 by wiring the shortcut reference and About labels through locale hooks, reusing existing shortcut keys and reducing dead locale keys to 42; the drift gate now reports 2,349 keys, 2,307 consumers, 42 dead keys, and 0 missing keys. |
| 2026-06-06 | Cycle 73 | CEP i18n duplicate dead-key cleanup | CEP `en.json`, `tests/test_i18n_drift.py` | The locale file still carried unused legacy keys for already-localized audio, captions, silence, and shortcut controls, plus repeated duplicate entries for GPU and settings labels. | Advanced E15 to cleanup batch 161 by removing six unused locale keys, pruning duplicate entries, and adding a locale-file duplicate-key guard; the drift gate now reports 2,343 keys, 2,307 consumers, 36 dead keys, and 0 missing keys. |
| 2026-06-06 | Cycle 74 | CEP i18n settings/form cleanup | CEP `index.html`, `en.json`, `tests/test_i18n_hardcoded_migration.py` | Settings Preferences and Whisper CPU-mode labels still had bare-English shell copy, and the locale file still carried unused generic form labels after related controls moved to more specific keys. | Advanced E15 to batch 162 by wiring six settings labels through existing locale keys and removing nine unused generic form locale keys; the drift gate now reports 2,334 keys, 2,313 consumers, 21 dead keys, and 0 missing keys. |
| 2026-06-06 | Cycle 75 | CEP i18n audio/shorts/timeline shell | CEP `index.html`, `tests/test_i18n_hardcoded_migration.py` | Audio enhancement, loudness match, Shorts options, and timeline marker export controls still had bare-English labels even though matching locale keys already existed. | Advanced E15 to batch 163 by wiring those controls through existing locale keys; the drift gate now reports 2,334 keys, 2,320 consumers, 14 dead keys, and 0 missing keys. |
| 2026-06-06 | Cycle 76 | CEP i18n final dead-key cleanup | CEP `en.json`, `scripts/i18n_lint.py`, `tests/test_i18n_drift.py` | The locale file still carried 14 unused keys after the latest shell migration, even though every static consumer had a matching locale entry. | Advanced E15 to batch 164 by removing the final unused locale keys and tightening the dead-key baseline to zero; the drift gate now reports 2,320 keys, 2,320 consumers, 0 dead keys, and 0 missing keys. |
| 2026-06-06 | Cycle 77 | CEP i18n JS metadata scanner coverage | `scripts/i18n_lint.py`, `tests/test_i18n_drift.py` | The drift scanner still only counted HTML locale attributes and direct `t(...)` calls, leaving structured JS locale-key metadata invisible to the consumer ledger. | Advanced E15 to batch 165 by counting supported JS key-field metadata such as `labelKey`; the drift gate now reports 2,320 keys, 2,320 consumers, 16 JS metadata consumers, 0 dead keys, and 0 missing keys. |
| 2026-06-06 | Cycle 78 | CEP i18n Auto Shorts and Settings shell | CEP `index.html`, `main.js`, `en.json`, `tests/test_i18n_hardcoded_migration.py` | Auto Shorts still had bare form labels/options/buttons, Magic Clips review-board status/detail strings, and the approved-render alert outside locale hooks, and Settings studio-readiness still had bare overview copy. | Advanced E15 to batch 166 by wiring those shells through locale keys; the drift gate now reports 2,360 keys, 2,360 consumers, 16 JS metadata consumers, 0 dead keys, and 0 missing keys. |
| 2026-06-07 | Cycle 79 | CEP i18n tab panels and Audio Normalize shell | CEP `index.html`, `en.json`, `tests/test_i18n_hardcoded_migration.py` | Captions, Audio, Video, Export, Timeline, and Search/NLP tab panels still had bare-English `aria-label` region names, and Audio Normalize still had bare preset options, meter labels, and preview control copy. | Advanced E15 to batch 167 by wiring those surfaces through locale hooks; the drift gate now reports 2,372 keys, 2,372 consumers, 16 JS metadata consumers, 0 dead keys, and 0 missing keys. |
| 2026-06-07 | Cycle 80 | CEP i18n Footage Search shell and PyTorch hardening | CEP `index.html`, `en.json`, `tests/test_i18n_hardcoded_migration.py`, `opencut/core/model_quantization.py`, `pyproject.toml`, dependency tests | Footage Search still had bare-English shell strings, model quantization forced unsafe PyTorch pickle loading, and Torch-backed optional extras admitted vulnerable pre-2.6 Torch versions. | Advanced E15 to batch 168 and closed the PyTorch hardening P0 by wiring Footage Search through locale hooks, switching quantization loads to `weights_only=True`, raising the Torch/TorchVision floor, and adding regression coverage. |
| 2026-06-07 | Cycle 81 | `open-path` allowlist hardening | `opencut/routes/system.py`, `tests/test_hardening.py`, `tests/test_new_features.py` | `/system/open-path` direct-open mode used an executable-extension blocklist that missed Windows control/shell payloads such as `.msc`, `.cpl`, `.settingcontent-ms`, and `.url`. | Replaced the blocklist with a safe media/document extension allowlist for direct open mode while preserving reveal mode for validated files; regression coverage rejects the missed dangerous extensions. |
| 2026-06-07 | Cycle 82 | CLIP cache safe deserialization | `opencut/core/semantic_video_search.py`, `tests/test_object_intel.py` | Semantic video search loaded predictable `~/.opencut/clip_cache/clip_*.pkl` files with raw `pickle.load()`, creating a cache-poisoning execution path if an attacker could write to the cache directory. | Replaced raw pickle caches with compressed `.npz` files that store JSON metadata and load arrays via `numpy.load(..., allow_pickle=False)`; regression coverage verifies both the safe-load option and a no-pickle cache round trip. |
| 2026-06-07 | Cycle 83 | Scripting-console resource limit | `opencut/core/scripting_console.py`, `opencut/routes/dev_scripting_routes.py`, `opencut/routes/workflow_dev_routes.py`, `tests/test_dev_scripting.py`, `tests/test_workflow_dev.py` | The scripting console had output and timeout caps but no source-size cap, allowing oversized code payloads to consume avoidable compile/exec resources. | Added a 100 KiB (102,400-byte) `MAX_CODE_LENGTH_BYTES` cap, enforced it in the core sandbox and both scripting HTTP routes before compile/exec, and covered direct core rejection plus HTTP 400 `CODE_TOO_LARGE` responses for 200 KiB submitted scripts. |
| 2026-06-07 | Cycle 84 | CEP i18n Timeline and Settings shell | CEP `index.html`, `en.json`, `tests/test_i18n_hardcoded_migration.py` | Timeline write-back, OTIO, beat-marker, multicam, marker-export, rename/smart-bin controls plus Settings system, dependency-health, and Whisper readiness copy still had bare-English shell strings after the prior Footage Search batch. | Advanced E15 to batch 169 by wiring those surfaces through locale hooks; the drift gate now reports 2,431 keys, 2,431 consumers, 16 JS metadata consumers, 0 dead keys, and 0 missing keys. |
| 2026-06-07 | Cycle 85 | CEP i18n Journal/Whisper shell and splat preview confinement | CEP `index.html`, `en.json`, `tests/test_i18n_hardcoded_migration.py`, `opencut/routes/generative_routes.py`, `tests/test_generative_routes_security.py` | Settings Operation Journal and Whisper readiness/default-model shell copy still had bare-English strings, and `/gaussian-splat/preview-frame` trusted the renderer-returned `frame_path` before calling `send_file()`. | Advanced E15 to batch 170 by wiring those CEP shells through locale hooks, and added a fail-closed preview-frame path validator that only serves existing renderer outputs under system temp or `~/.opencut`; the drift gate now reports 2,457 keys, 2,457 consumers, 16 JS metadata consumers, 0 dead keys, and 0 missing keys. |
| 2026-06-07 | Cycle 86 | Expression-engine thread churn and CEP i18n footer/wizard shell | `opencut/core/expression_engine.py`, `tests/test_motion_design.py`, CEP `index.html`, `main.js`, `en.json`, `scripts/i18n_lint.py`, `tests/test_i18n_drift.py`, `tests/test_i18n_hardcoded_migration.py` | `evaluate_expression()` spawned a fresh daemon `threading.Thread` for every eval as the timeout mechanism, so `evaluate_timeline()` created one worker per frame; the CEP progress/results/footer, command palette, preview modals, context menu, and first-run wizard still had bare-English shell attributes and copy outside locale hooks. | Replaced per-eval worker spawning with inline trace-deadline evaluation that restores any prior trace hook, added a 30-second timeline regression proving `evaluate_timeline()` creates no raw worker threads, and advanced E15 to batch 171 by localizing those CEP shells plus adding `data-i18n-alt` scanner support. The timeline benchmark held the thread count at 2 before and after while evaluating 900 frames with no errors; the drift gate now reports 2,515 keys, 2,515 consumers, 16 JS metadata consumers, 0 dead keys, and 0 missing keys. |
| 2026-06-07 | Cycle 87 | Security rejection audit logging | `opencut/security_audit.py`, `opencut/security.py`, `opencut/server.py`, `opencut/routes/system.py`, `tests/test_security_audit.py` | CSRF failures, path-validation rejections, remote auth denials, and rate-limit rejections returned 4xx responses without a structured trail for incident review, and token/path evidence needed to avoid leaking secrets. | Added a best-effort schema-tagged `security_audit.jsonl` writer, hooked CSRF rejection, `validate_path()` rejection branches, rate-limit denials, and remote auth-token denials into it, preserved request context and request IDs when available, redacted CSRF/auth token values, recorded rejected path evidence as a short preview plus SHA-256 hash, and exposed capped recent reads via `/system/audit-log`. Test apps disable the default sink unless `OPENCUT_SECURITY_AUDIT_LOG` is set. |
| 2026-06-07 | Cycle 88 | Cleanup-thread lazy initialization | `opencut/helpers.py`, `tests/test_helpers_cleanup.py`, `scripts/release_smoke.py` | Importing `opencut.helpers` started the `opencut-temp-cleanup` daemon as a utility-module side effect, affecting CLI tools and tests that only needed helper constants or path utilities. | Deferred the cleanup daemon behind `_ensure_cleanup_thread_started()` so it starts only after `_schedule_temp_cleanup()` queues the first file; fresh-interpreter tests prove import stays thread-clean and the worker appears on first scheduled cleanup, and pytest-fast now carries the guard. |
| 2026-06-07 | Cycle 89 | CEP i18n captions/audio/NLP utility shell | CEP `index.html`, `en.json`, `tests/test_i18n_hardcoded_migration.py`, `tests/test_i18n_drift.py` | Captions quick-action labels, SRT import controls, beat-marker stats, audio form placeholders, MusicGen controls, LUT path placeholders, NLP command shell, and LLM settings placeholders still had nested or attribute-level bare English outside explicit locale hooks. | Advanced E15 to batch 172 by wiring those shells through locale keys and adding focused guard coverage; the drift gate now reports 2,543 keys, 2,543 consumers, 16 JS metadata consumers, 0 dead keys, and 0 missing keys. |
| 2026-06-07 | Cycle 90 | WCAG contrast release gate | `opencut/tools/contrast_audit.py`, `scripts/release_smoke.py`, `tests/test_contrast_audit.py`, CEP/UXP panel CSS | The June 6 research plan identified no automated WCAG/contrast audit in CI, so panel token regressions could ship despite extensive static ARIA coverage. | Added a stdlib static contrast audit over committed CEP/UXP `:root` design-token blocks, wired it into release smoke and pytest-fast as `contrast-audit`, proved a deliberately low-contrast fixture fails, and raised CEP `--text-muted` to `#707090` so muted chrome clears 3.15:1 on `--bg-elevated`. The gate audits 72 token pairs with 0 failures. |
| 2026-06-07 | Cycle 91 | Async rate-limit migration | `opencut/jobs.py`, `opencut/security.py`, async route modules, `tests/test_async_job_rate_limit.py`, `scripts/release_smoke.py` | The June 6 research plan found 25+ manual route-level `rate_limit()` / `rate_limit_release()` pairs, which made async worker-lifetime locks easy to leak and inconsistent with the decorator pattern. | Added `async_job(rate_limit_key=...)` with synchronous 429 rejection before job creation and release paths for worker completion/setup failures, migrated model-install and GPU-heavy async routes to that wrapper, preserved conditional BasicVSR denoise locking, converted MCP bridge per-tool throttling to `rate_limit_slot()`, and added a release-smoke guard proving no route module calls the primitives directly. |
| 2026-06-07 | Cycle 92 | CEP structured empty states | CEP `index.html`, `main.js`, `style.css`, `locales/en.json`, `tests/test_i18n_hardcoded_migration.py` | The June 6 UX audit found CEP still used plain hints or hidden empty containers where UXP had structured `oc-empty-state` components, especially for zero-result lists and Favorites. | Promoted `buildEmptyHintMarkup()` to emit shared `oc-empty-state` classes while preserving hint/tone semantics, rendered a localized Favorites empty state instead of hiding the bar, and added static coverage for job history, batch files, workflow steps, footage search, and favorites empty-state surfaces. |
| 2026-06-07 | Cycle 93 | CEP Settings preferences i18n shell | CEP `index.html`, `locales/en.json`, `tests/test_i18n_hardcoded_migration.py` | The Settings preferences card still had hardcoded shell copy for the preferences description, output-location choices, appearance/theme choices, UI language choices, and the GPU/backend log control labels. | Advanced E15 to batch 173 by wiring those labels and options through locale hooks while preserving the current native UI-language labels; the drift gate now reports 2,564 keys, 2,564 consumers, 16 JS metadata consumers, 0 dead keys, and 0 missing keys. |
| 2026-06-07 | Cycle 94 | UXP i18n foundation shell | UXP `index.html`, `main.js`, `locales/en.json`, `tests/test_uxp_i18n.py`, `scripts/release_smoke.py` | The UXP panel had no locale loader or guarded first-viewport i18n surface, and dynamic connection/workspace labels still depended on visible English state text. | Added UXP-local locale loading with `t()` and `applyI18nToDOM()`, wired the shell/tab/workspace/processing surfaces through locale hooks, changed connection checks to use state instead of the visible label, and added a release-smoke static guard for UXP HTML and JS locale coverage. |
| 2026-06-07 | Cycle 95 | UXP Cut tab i18n shell | UXP `index.html`, `locales/en.json`, `tests/test_uxp_i18n.py` | After the foundation slice, the Cut & Clean tab still had bare-English labels, placeholders, select options, result labels, and accessible names immediately below the localized workspace header. | Localized Cut-tab clip input, silence detection, filler cleanup, result-summary labels, placeholders, select options, and ARIA labels, then raised the UXP i18n static guard to a 90+ attribute floor with representative Cut-tab keys. |
| 2026-06-07 | Cycle 96 | UXP Captions tab i18n shell | UXP `index.html`, `locales/en.json`, `tests/test_uxp_i18n.py` | The Captions tab still had bare-English shell copy for transcription setup, model/language/style options, chapter generation, repeat detection, result metadata, placeholders, and accessible names. | Localized the Captions tab static shell across transcription, chapters, repeat detection, and result review, then raised the UXP i18n static guard to a 170+ attribute floor with representative Captions-tab keys. |
| 2026-06-07 | Cycle 97 | UXP FCC display i18n shell | UXP `index.html`, `main.js`, `locales/en.json`, `tests/test_uxp_i18n.py`, `tests/test_uxp_caption_display_settings_ui.py` | The F236 caption display-settings card still had hardcoded field labels, compliance notice text, preview controls, live preview labels, and dynamic token/preview status messages. | Localized the FCC card's static and dynamic shell, preserved the FCC source link during runtime compliance-date updates, and raised the UXP i18n static guard to a 190+ attribute floor with representative FCC card keys. |
| 2026-06-07 | Cycle 98 | UXP Audio tab i18n shell | UXP `index.html`, `locales/en.json`, `tests/test_uxp_i18n.py` | The Audio tab still had bare-English clip input, denoise, normalization, loudness-match, and beat-marker labels, placeholders, option labels, checkbox text, and action buttons. | Localized the Audio tab static shell across denoise, normalize, loudness match, and beat markers, then raised the UXP i18n static guard to a 220+ attribute floor with representative Audio-tab keys. |
| 2026-06-07 | Cycle 99 | UXP Video tab core i18n shell | UXP `index.html`, `locales/en.json`, `tests/test_uxp_i18n.py` | The top/core Video tab still had bare-English clip input, Color Match, Auto Zoom, Multicam Switch, Multimodal Diarization, B-roll Generation, and Depth Effects labels, placeholders, select options, ARIA labels, hints, and action buttons. | Localized the Video shell through Depth Effects, then raised the UXP i18n static guard to a 280+ attribute floor with representative Video-tab keys. |
| 2026-06-07 | Cycle 100 | UXP Video effects i18n shell | UXP `index.html`, `locales/en.json`, `tests/test_uxp_i18n.py` | The next Video-tab run still had bare-English Emotion Highlights, B-Roll Analysis, Chat Editor, AI Upscale, Scene Detection, and Style Transfer labels, placeholders, select options, ARIA labels, hints, and action buttons. | Localized the Video effects shell through Style Transfer, then raised the UXP i18n static guard to a 325+ attribute floor with representative Video-tab keys and verified the rendered UXP Video tab in the in-app browser. |

### Research queries to run later

- `site:developer.adobe.com/premiere-pro/uxp manifestVersion Premiere supports version 5 clipboard permission`
- `site:forums.adobe.com Premiere UXP WebView plugin local file system fullAccess clipboard permission`
- `Descript Underlord timeline diff checkpoint rollback user review`
- `CapCut desktop auto captions auto reframe script to video complaints region watermark`
- `DaVinci Resolve 21 CineFocus IntelliSearch AI local workflow user reviews`
- `LosslessCut distribution appimage snap homebrew winget packaging video editor`
- `GitHub Actions pin actions full SHA artifact attestations release workflow`
- `Docker .dockerignore secrets build context Python project best practices`
- `site:developer.adobe.com/premiere-pro/uxp createCaptionTrack addCaptionTrack CaptionTrack UXP`
- `Premiere Pro UXP getTrackItems CaptionTrack text timing parameters`
- `Riverside Magic Clips focus speaker keywords layouts animated captions official`
- `OpusClip ClipAnything visual audio sentiment cues brand templates social accounts official`
- `Descript Create Clips Underlord aspect ratio B-roll captions official`
- `CapCut Long video to shorts caption template duration hashtags official`

### Next research cycles

1. Cycle 101: Continue UXP Video-tab i18n coverage into the remaining Shorts Pipeline and Social Media Upload shell or resume CEP E15 hardcoded-shell cleanup.
2. Cycle 102: Continue E15 scanner coverage or another release-trust gap from the June 6 plan.
3. Cycle 103: Audit caption UX again only if Adobe publishes a documented UXP caption write API.
4. Cycle 104: Revisit UXP cutover only after live UDT evidence is available.
5. Cycle 105: Re-scan Adobe UXP Hybrid packaging docs after the next Premiere UXP SDK release.

### Continuation State

#### Last completed cycle

Cycle 100: UXP Video effects i18n shell through Style Transfer.

#### Current focus

Continue from active release-trust, migration hardening, Docker hardening, and
product workflow specs. RA-05/RA-37, RA-06/RA-40, RA-07/RA-38, RA-08/RA-39,
RA-01, RA-02, RA-03, RA-04, RA-11, RA-12, RA-13, RA-14, RA-15, RA-16, RA-17, RA-18, RA-19, RA-20, RA-21, RA-22, RA-23, RA-24, RA-25, RA-26, RA-27, RA-28, RA-29, RA-30, RA-31, RA-32, RA-33, RA-35, RA-36, RA-42, RA-43, RA-44, and
RA-45, RA-54, RA-55, and RA-56 are closed, and the bootstrap dev-check guard is in place. RA-41 is
closed: shared dry-run/confirm-token helpers cover the original named
endpoint list plus adjacent assistant/chat/undo/search/worker-pool clears, and
journal clear is covered by the local DB dry-run/backup contract. RA-15 keeps
`opencut[all]` as the audited convenience lane while Torch/Transformers-backed
packages stay explicit through named feature extras and `torch-stack`. RA-17
keeps the shipped UXP manifest on Premiere-supported schema version 5 while the
WebView scaffold's version 6 template remains dormant. RA-18 keeps deprecated
Clipboard APIs, object-form clipboard writes, and legacy `uxpvideo*` events out
of UXP/WebView sources. RA-19 declares the clipboard permission and routes copy
behavior through a shared fallback helper. RA-11 keeps UXP filesystem access
picker-scoped through `localFileSystem: "request"`. RA-13 keeps external launch
limited to HTTPS OAuth browser handoff with no file-extension launches. RA-14
keeps the WebView scaffold split between a dev hot-reload profile and a
release local-only message bridge profile. RA-20 keeps UXP destructive
confirmation on a panel-native second-click flow rather than beta browser
dialogs. RA-25/RA-29/RA-30 keep Docker installs on the tracked requirements
surface, fail closed on dependency install errors, and exclude local
secret/log/runtime DB state from the Docker build context. RA-26 keeps the
default Docker runtime explicitly HTTP-only on 5679 and leaves WebSocket/MCP
sidecars to custom opt-in services. Continue with the remaining release-trust
and product workflow specs. RA-22 keeps Release Full's Linux panel npm gates on
the same explicit Node 22 runtime as PR Fast before treating panel advisory,
unit, and build evidence as release proof.
Cycle 86 closed expression-engine per-frame thread churn with inline
trace-deadline evaluation and a constant-thread-count timeline regression, and
advanced E15 to batch 171 with progress/results/footer, command palette,
preview, context-menu, first-run wizard, and `data-i18n-alt` scanner coverage.
Cycle 87 closed the security audit logging gap with a best-effort
`security_audit.jsonl` writer for CSRF, path-validation, rate-limit, and remote
auth-token rejections, including request IDs when available, token redaction,
hashed path evidence, and capped recent reads through `/system/audit-log`.
Cycle 88 closed cleanup-thread lazy initialization by deferring
`opencut-temp-cleanup` until `_schedule_temp_cleanup()` first queues work, with
subprocess tests covering import and first schedule behavior.
Cycle 89 advanced E15 to batch 172 by localizing Captions quick-action labels,
SRT import controls, beat-marker stats, audio form placeholders and MusicGen
controls, LUT path placeholders, NLP command shell, and LLM settings
placeholders.
Cycle 90 closed the WCAG contrast audit finding with a stdlib
`opencut.tools.contrast_audit` gate over CEP/UXP panel design tokens, release
smoke wiring, pytest-fast coverage, and a deliberate low-contrast fixture. The
primary CEP `--text-muted` token now clears the 3.0:1 muted-chrome floor on
`--bg-elevated`, and the gate audits 72 token pairs with 0 failures.
Cycle 91 closed the rate-limit migration finding by moving async model-install
and GPU-heavy route locks into worker-lifetime `async_job(rate_limit_key=...)`
handling, keeping conditional BasicVSR denoise locking, using
`rate_limit_slot()` for MCP bridge per-tool keys, and adding release-smoke
coverage that blocks direct route-level rate-limit primitive calls.
Cycle 92 closed the CEP empty-state parity finding by routing the shared CEP
empty hint helper through `oc-empty-state` classes, adding localized Favorites
empty-state copy, and guarding job history, batch files, workflow steps,
footage search, and favorites with static migration coverage. Browser
validation loaded the CEP shell from a local static server and verified the
footage-search empty-state DOM/classes rendered on a nonblank page; runtime
Favorites interaction was covered statically because the raw CEP classic-script
shell is not the intended normal-browser runtime. Panel Vitest, ESLint, and
build verification covered the local frontend surface.
Cycle 93 advanced E15 to batch 173 by localizing the remaining CEP Settings
preferences shell labels, options, and GPU/backend log controls, preserving
the native UI-language labels and keeping the CEP drift gate at 0 dead and 0
missing keys.
Cycle 94 opened the UXP i18n parity track with a local `locales/en.json`,
`loadLocale()`, `t()`, and DOM attribute application in the UXP panel. The
first-viewport shell, tab bar, processing banner, connection label, workspace
overview, and dynamic workspace guide now use locale attribute hooks, and
`tests/test_uxp_i18n.py` is wired into release smoke to guard UXP locale
coverage and state-based connection checks.
Cycle 95 extended that UXP i18n coverage into the Cut & Clean tab: clip input,
silence detection, filler cleanup, result-summary labels, placeholders, select
options, and accessible names now use locale keys, and the static guard requires
at least 90 UXP i18n attributes.
Cycle 96 extended that UXP i18n coverage into the Captions tab: transcription
setup, chapter generation, repeat detection, result metadata, placeholders,
select options, checkbox labels, and accessible names now use locale keys, and
the static guard requires at least 170 UXP i18n attributes.
Cycle 97 extended that UXP i18n coverage into the F236 caption display-settings
card: field labels, compliance notice copy, FCC source link, preview controls,
live preview labels, dynamic status feedback, and preview sample fallback now use
locale keys. The compliance-date replacement now updates only the date span so
the source link remains intact, and the static guard requires at least 190 UXP
i18n attributes.
Cycle 98 extended that UXP i18n coverage into the Audio tab: clip input,
denoise method controls, normalization controls, loudness-match fields,
beat-marker inputs, placeholders, option labels, checkbox text, and action
buttons now use locale keys, and the static guard requires at least 220 UXP i18n
attributes.
Cycle 99 extended that UXP i18n coverage into the Video tab through Depth
Effects: clip input, Color Match, Auto Zoom, Multicam Switch, Multimodal
Diarization, B-roll Generation, and Depth Effects controls, placeholders,
select options, ARIA labels, hints, and action buttons now use locale keys, and
the static guard requires at least 280 UXP i18n attributes.
Cycle 100 extended the same UXP i18n coverage through the Video effects shell:
Emotion Highlights, B-Roll Analysis, Chat Editor, AI Upscale, Scene Detection,
and Style Transfer controls, placeholders, select options, ARIA labels, hints,
and action buttons now use locale keys, and the static guard requires at least
325 UXP i18n attributes.
The package Ruff release-smoke gate is clean again after mechanical import
ordering, with route-manifest and route-collision checks re-run after the
blueprint import-block cleanup.
RA-24 keeps Release Full build/test/package jobs on read-only contents
permission while the tag-only release-upload job owns the write-capable release
token.
RA-12 keeps future UXP Hybrid addon packaging behind a static validator that
checks manifest v6+/`enableAddon`, safe `.uxpaddon` names, production host
shape, and the mac arm64/mac x64/win x64 layout before release claims.
RA-21 keeps package metadata to the tested Python classifier set until a 3.13
workflow lane proves runtime support.
RA-23 keeps non-local workflow actions pinned to full-length SHAs with adjacent
version comments so workflow action updates stay explicit. Release provenance is
closed for the current Release Full shape: uploaded server archives, Linux
packages, Windows installer, and declared SBOM paths are attested before release
upload, and operator verification commands live in `docs/RELEASE_PROVENANCE.md`.
RA-36 keeps Windows shared-folder panel npm gates on wrapper aliases that
resolve the wrapper from npm's original working directory and execute Node
scripts from the wrapper's script directory.
RA-04 keeps structured JSON error envelopes tied to the generated server
request ID so client-visible errors can be correlated directly with logs.
RA-03 keeps direct typed error responses logged with structured code, status,
request ID, method, path, and typed-error context fields.
The Gaussian splat preview `send_file()` confinement item is closed: renderer
outputs for `/gaussian-splat/preview-frame` must resolve to an existing file
under the system temp directory or `~/.opencut`, and unconfined renderer paths
return 403 before Flask can serve them.
RA-01/RA-02 keep Ruff's Python parser target aligned with the package floor and
keep `requirements.txt` core/standard dependency bounds synchronized with
`pyproject.toml`.
E15 is advanced through batch 173: progress/results/footer chrome, command
palette shell, preview/audio preview modals, clip context menu, first-run wizard,
`data-i18n-alt` scanner coverage, Settings Operation Journal, Whisper
readiness/default-model shell, Timeline write-back, OTIO, beat-marker,
multicam, marker-export, rename/smart-bin controls, Settings system,
dependency-health, Footage Search shell copy, tab panel
region labels, Audio Normalize shell controls, Auto Shorts form labels/options/buttons,
and Settings preferences shell controls,
Magic Clips review-board status/detail copy, the approved-render alert, and the
Settings studio-readiness overview shell now use locale hooks, the final unused
CEP locale keys have been removed, the dead-key baseline is zero, the scanner counts supported JS metadata locale keys
such as `labelKey`, and audio
enhancement, loudness match, Shorts options, timeline marker export, Settings
Preferences, Whisper CPU mode,
Settings shortcut/About, Audio & Zoom
Defaults, GPU Recommendation, Settings Engine Routing, Live Updates Bridge,
Settings Project Templates, AI Models, Export Deliverables, LLM settings, preset
diagnostics, Workflow Presets static shell strings, Captions quick-action labels,
SRT import controls, beat-marker stats, audio form placeholders and MusicGen
controls, LUT path placeholders, NLP command shell, and LLM settings
placeholders now use locale hooks, and the CEP drift gate reports
2,564 keys, 2,564 consumers, 16 JS metadata consumers, 0 dead
keys, and 0 missing keys. UXP i18n has a first shell plus Cut/Captions/FCC
display/Audio/Video-through-Style-Transfer foundation and a 325+ static
locale-coverage guard, but full UXP parity remains open.
RA-46 is closed under RA-09: caption exports now write versioned sidecars and
timeline SRT parsing can preserve metadata when a sidecar is available.
RA-47 is closed under RA-09: caption round-trip diff/apply APIs now support
sidecar-backed metadata-preserving reviews, lossy no-sidecar diffs, and
confirmation-token guarded revision storage.
RA-48 is closed under RA-09: UXP can now read caption-track snapshots into the
round-trip diff schema when the host exposes caption read APIs. RA-49 is closed
under RA-09: CEP caption writes now return a sidecar-aware placement contract
with explicit native, video-track, project-import, and manual-drag modes. RA-50
is closed under RA-09: metadata-loss fixtures now cover SRT-only loss,
sidecar-backed preservation, split/merge/insert/delete classifications, stale
sidecar warnings, and no-sidecar degraded mode. RA-09 is closed.

#### Important findings so far

- Direct repo write access to `ROADMAP.md` on the UNC project path succeeded in
  this turn.
- The shared memory path specified in `AGENTS.md` did not exist under
  `C:\Users\Matt\.claude`.
- `rtk` was available on PATH and was used for the required recent-history
  check.
- `ROADMAP.md` had uncommitted user/pre-existing changes before this turn;
  append-only edits preserve that work.
- The Adobe tracker, label seeder, Release Full permissions, UXP manifest, and
  Dockerfile all have concrete local evidence for the active RA queue.
- E15's live i18n linter passes with 0 dead keys and 0 missing keys; remaining
  work should target new hardcoded shell strings and additional scanner coverage,
  not broad missing-key repair.
- The repo `.venv` can launch but lacks pytest/dev tooling; `bootstrap_check.py
  --metadata-only --dev` now catches that state, while `py -3.12` passes the
  same dev check in this workspace.
- README non-badge route, module, blueprint, panel line-count, and root test-file
  claims now run through `scripts/check_doc_sizes.py` beside the older
  documentation size targets.
- Release SBOM workflow artifacts now use declared-SBOM naming, and generated
  CycloneDX metadata marks the inventory as `declared-only` with source,
  exclusion, and lockfile advisory-audit evidence.
- Docker GPU docs now point at the committed `gpu` profile service command, and
  `tests/test_docker_distribution_docs.py` guards against missing compose
  override references.
- Docker dependency installs now consume `requirements.txt` directly, avoid
  retired `pydub`/`deep-translator` packages, fail on pip errors, and guard
  `.dockerignore` secret/log/runtime-state exclusions before `COPY . /app`.
- Docker runtime docs, Compose, and Dockerfile now agree that default containers
  publish HTTP 5679 only; WebSocket 5680 and MCP 5681 sidecars require explicit
  custom services/profiles.
- Release Full's Linux panel npm gates now set up Node 22 before `npm ci`,
  matching PR Fast's panel runtime pin.
- The package Ruff release-smoke gate is clean after mechanical import-order
  cleanup across existing package files.
- Release Full now keeps build/test/package jobs on `contents: read`; the
  tag-only release-upload job packages server assets, generates GitHub artifact
  attestations, and uploads only the attested release paths.
- The Python 3.13 classifier is retracted until a committed workflow lane tests
  that runtime.
- Non-local GitHub Actions workflow references are full-SHA pinned with adjacent
  version comments and a release-smoke guard rejects mutable refs.
- CEP panel npm advisory, esbuild-pin, and build verification gates now have
  Windows-safe `:win` aliases for UNC/HGFS checkouts.
- Structured JSON error bodies now include the generated server request ID that
  matches the `X-Request-ID` response header.
- Direct typed errors now emit structured log records for error code, status,
  request ID, method, path, and typed-error context.
- Ruff now treats `tomllib` as Python 3.11 stdlib, so import-order checks can
  surface package files that were clean under the older Python 3.9 parser
  target.
- E15 batch 162 wired Settings Preferences and Whisper CPU-mode labels through
  existing locale keys, removed unused generic form locale keys, and reduced
  dead locale keys from 36 to 21 while keeping missing keys at 0.
- E15 batch 163 wired audio enhancement, loudness match, Shorts option, and
  timeline marker export labels through existing locale keys, reducing dead
  locale keys from 21 to 14 while keeping missing keys at 0.
- E15 batch 164 removed the final 14 dead CEP locale keys after the scanner
  confirmed they had no static consumers, then set the dead-key baseline to
  zero so regressions fail immediately.
- E15 batch 165 expanded the drift scanner to count supported JS locale-key
  metadata fields and now reports 16 metadata consumers without changing the
  zero-dead/zero-missing live state.
- E15 batch 166 localized Auto Shorts form labels/options/buttons, Magic Clips
  review-board status/detail copy, the approved-render alert, and the Settings
  studio-readiness overview shell while preserving the zero-dead/zero-missing
  drift posture.
- E15 batch 167 localized the remaining CEP tab panel `aria-label` region names
  plus Audio Normalize preset options, loudness meter labels, and preview control
  copy.
- E15 batch 168 localized the Footage Search card, index summary, empty state,
  query controls, search action, and results region label.
- E15 batch 169 localized Timeline write-back, OTIO, beat-marker, multicam,
  marker-export, rename/smart-bin controls plus Settings system,
  dependency-health, and Whisper readiness shell copy.
- E15 batch 170 localized Settings Operation Journal and Whisper
  readiness/default-model shell copy while preserving the zero-dead/zero-missing
  drift posture.
- E15 batch 171 localized progress/results/footer chrome, command palette shell,
  preview/audio preview modals, clip context menu, and first-run wizard copy,
  while adding `data-i18n-alt` scanner/runtime coverage.
- E15 batch 172 localized Captions quick-action labels, SRT import controls,
  beat-marker stats, audio form placeholders and MusicGen controls, LUT path
  placeholders, NLP command shell, and LLM settings placeholders.
- E15 batch 173 localized the remaining Settings preferences shell labels,
  output-location options, theme options, GPU checking label, backend log button
  label, and UI language choices while keeping the dead/missing-key baseline at
  zero.
- PyTorch deserialization hardening is closed: quantization loads now use
  `weights_only=True`, unsafe pickle checkpoints produce a clear error, and
  Torch-backed optional extras require `torch>=2.6` / `torchvision>=0.21`.
- `open-path` allowlist hardening is closed: direct open mode now accepts only
  safe media/document extensions, while reveal mode still selects validated
  files in the OS file manager.
- CLIP cache deserialization hardening is closed: semantic video search now
  writes compressed `.npz` caches with JSON metadata and loads embedding arrays
  with `allow_pickle=False` instead of raw pickle caches.
- Scripting-console resource-limit hardening is closed: oversized source payloads
  are rejected before compile/exec in both the core sandbox and HTTP route.
- Gaussian splat preview send-file confinement is closed: preview frames are
  served only when the renderer output resolves under system temp or
  `~/.opencut`, and unconfined renderer paths fail with 403.
- RA-12 is closed as a static packaging guard; actual native addon loading
  still needs UDT/native-platform evidence when a `.uxpaddon` is introduced.
- SRT remains a lossy text/timing carrier; the new sidecar path is the metadata
  preservation contract for caption timeline round trips until native UXP writes
  or hybrid caption writes are live-tested.
- Caption revision apply is content-addressed and idempotent for unchanged edits,
  but it intentionally stores a new revision file instead of mutating the
  original transcript cache entry.
- UXP caption-track handling is still read-only: `ocGetCaptionTrackSnapshot`
  advertises explicit `reason_code` values for no project, no active sequence,
  no caption tracks, and missing caption APIs, while `ocAddNativeCaptionTrack`
  remains CEP/hybrid-only until Adobe documents a write API.
- The scanned SQLite stores now stamp explicit SQLite `user_version` values via
  ordered idempotent local migrations and reject newer unknown schemas.
- `jobs.result_json`, `journal.inverse_json`, and `journal.forward_json` now
  spill oversized JSON into content-addressed `.opencut/payload_spills` files
  and return structured spill metadata instead of large inline rows.
- Local SQLite stores now have shared page/freelist/WAL/file-size diagnostics
  through CLI and feature routes, and destructive local SQLite clears now expose
  dry-run affected-row counts, optional `VACUUM INTO` backups, and JSONL audit
  entries before mutation.
- Render-cache reads, cleanup, and downstream invalidation now fail closed on
  forged `index.json` output paths by requiring files to resolve under
  `CACHE_DIR` and match the cache-key basename before any unlink.
- Plugin uninstall now moves plugin directories into timestamped quarantine
  entries before unloading, requires typed `confirm_name`, and exposes list,
  restore, and permanent-delete quarantine actions.
- Whisper cache clear and model delete now expose dry-run/preview plans with
  exact paths, byte counts, categories, and per-path deletion errors.
- Preset, workflow, favorites, and assistant dismissal mutations now write
  capped tombstone snapshots with restore metadata and restore routes.
- Destructive routes generally have CSRF and input validation, but no shared
  dry-run/confirmation token contract.
- Caption generation/export is mature enough to support SRT/VTT/ASS/JSON and
  edited transcript export with QC, but `/timeline/srt-to-captions` is currently
  a validation/normalization route that returns only `start`, `end`, and `text`.
- `/transcript/export` writes edited caption files but does not persist a new
  transcript cache/state revision or diff edited captions against the original
  transcript.
- CEP still has the practical caption write path via `ocAddNativeCaptionTrack()`
  and `importCaptions()`, while UXP returns an explicit CEP fallback for native
  caption-track creation.
- Current Adobe UXP docs reviewed on 2026-06-06 expose caption-track read APIs
  (`Sequence.getCaptionTrack*`, `CaptionTrack.getTrackItems`) but no scanned
  documented create/import caption-track write API.
- `shorts_pipeline.generate_shorts()` is a working render primitive, but it is
  execution-first and lacks a dry-run graph, candidate review, preset identity,
  thumbnail output, output manifest, or checkpoint/resume boundary.
- The live shorts route is `/video/shorts-pipeline` in
  `opencut/routes/video_specialty.py`; the older `opencut/routes/video.py`
  pointer is stale.
- The workflow engine can chain one async route output into the next and polls
  child jobs, but its one-file `current_input` model does not fit a one-to-many
  Magic Clips bundle.
- `shorts_variants.plan_variants()` and `/shorts/variants/dry-run` already show
  the right local dry-run pattern for a Magic Clips plan contract.
- `virality_score.rank()` can score/rank candidate clips, but
  `generate_shorts()` currently relies on highlight extraction scores and does
  not integrate that ranking layer.
- `export_presets.py` owns Shorts/TikTok/Reels presets, but CEP maps dimensions
  manually and UXP exposes only max clips, face tracking, and burn-in captions.
- Thumbnail scoring/variant modules and `long_to_shorts.py` CSV metadata are
  available primitives, but the main shorts pipeline does not emit thumbnails or
  a durable manifest.
- Current Riverside, OpusClip, Descript, and CapCut docs all reinforce the same
  product shape: AI-selected moments plus review/customize controls, captions,
  layouts/aspect ratios, and export/share handoff.
- Current npm registry evidence captured on 2026-06-06 shows
  `@adobe/premierepro` `beta=26.3.0-beta.85`, `latest=26.2.0`, and
  `release-26.2=26.2.1`; schema v2 now records release-channel tags in
  `tracked_dist_tags`.
- GitHub Actions runs explicit `shell: bash` steps with `-e -o pipefail`, so
  drift commands that intentionally return 2 must use `set +e`, capture `$?`,
  then return success after writing `GITHUB_OUTPUT`.
- The Adobe tracker workflow now uses one `trackerLabels` array for issue search
  and creation, and `.github/labels.yml` declares `f251`, `uxp`, and `tracking`.
- `scripts/seed_github_issues.py --labels --dry-run` no longer requires the
  GitHub CLI; real apply still requires it.

#### Next best actions

1. Continue E15 rolling CEP i18n migration with another hardcoded-shell audit or scanner-coverage pass; dead-key cleanup should remain at zero.
2. Audit UXP i18n parity or another remaining UX gap from the June 6 plan.
3. Revisit UXP cutover only after live UDT evidence is available.

#### Unprocessed leads

- Confirm future package-manager artifacts reuse the Release Full attestation
  path when they are added to GitHub Releases.
- Whether future Docker profiles should publish optional WebSocket 5680 or MCP
  5681 sidecars now that the default container posture is HTTP-only.
- Whether future Adobe caption-write APIs should reopen RA-09 or create a new
  focused UXP caption-write item.
- Whether Adobe ships a documented UXP caption write API after the 2026-06-06
  reference scan.
- Whether RA-40 should cover every user-data wipe path or stay scoped to local
  SQLite stores plus cache rebuilds.
- Whether RA-41 should become the shared contract that consolidates the closed
  destructive-route hardening patterns.

#### Files still to inspect

- `tests/test_i18n_hardcoded_migration.py`
- `scripts/i18n_lint.py`
- `extension/com.opencut.panel/client/index.html`
- `extension/com.opencut.panel/client/main.js`
- `extension/com.opencut.panel/client/locales/en.json`
- `tests/test_uxp_*`
- `docs/UXP_MIGRATION.md`
- Adobe tracker docs/tests/generated outputs
- `tests/*workflow*` for route-manifest and async polling fixtures

#### Searches still to run

- `rg -n "data-i18n|t\\(|hardcoded|dead key|missing key" extension scripts tests`
- `rg -n "manifestVersion|clipboard|localFileSystem|webview|fullAccess" extension docs tests`
- `rg -n "docker-compose.gpu|OPENCUT_HOST|5680|5681|\\.env|\\.log" README.md Dockerfile docker-compose*.yml .dockerignore .gitignore`
- `rg -n "workflow_step|async_job|checkpoint|resume|cancel|macro|plan" opencut tests`

---

## Appendix: Sources

Full source register with 200+ URLs lives in `.ai/research/2026-05-17/SOURCE_REGISTER.md`.
Key external sources referenced in this roadmap:

- [S01] https://developer.adobe.com/premiere-pro/uxp/
- [S03] https://blog.developer.adobe.com/en/publish/2026/04/uxp-hybrid-plugins-now-available-for-premiere
- [S04] https://blog.developer.adobe.com/en/publish/2026/03/introducing-webview-ui-in-bolt-uxp-build-richer-adobe-plugins-faster
- [S05] https://github.com/hyperbrew/bolt-uxp
- [S06] https://medium.com/adobetech/updates-for-creative-cloud-desktop-extensibility-0dd5c663563e
- [S07] https://news.adobe.com/en/gb/news/2026/04/adobe-new-creative-agent
- [S10] https://github.com/mifi/lossless-cut (41K stars)
- [S14] https://github.com/SYSTRAN/faster-whisper (23K stars)
- [S21] https://github.com/facebookresearch/sam2 (19K stars)
- [S25] https://github.com/Lightricks/LTX-Video
- [S38] https://github.com/AcademySoftwareFoundation/OpenColorIO
- [S51] https://github.com/leancoderkavy/premiere-pro-mcp (MCP, 269 tools)
- [S60] https://www.descript.com
- [S61] https://www.capcut.com
- [S62] https://www.autocut.com
- [S63] https://www.opus.pro ($50M raised)
- [S64] https://vizard.ai (10M+ users)
- [S68] https://elevenlabs.io
- [S69] https://www.captions.ai
- [S70] https://www.submagic.co (4M+ users)
- [S73] https://riverside.fm
- [S74] https://www.heygen.com (MCP integration)
- [S76] https://www.topazlabs.com/topaz-video-ai
- [S77] https://www.izotope.com/en/products/rx.html
- [S90] https://ffmpeg.org
- [S91] https://www.w3.org/TR/ttml-imsc1.3/
- [S92] https://c2pa.org/specifications/specifications/2.3/specs/C2PA_Specification.html
- [S93] https://losslesscut.net/
- [S94] https://docs.brew.sh/Cask-Cookbook
- [S95] https://formulae.brew.sh/cask/losslesscut
- [S96] https://learn.microsoft.com/en-us/windows/package-manager/package/manifest
- [S97] https://snapcraft.io/docs/snapcraft-yaml-schema/
- [S98] https://docs.pypi.org/trusted-publishers/

---

## Implementation ledger

The v4.0-v4.248 implementation log (5,535 lines) is archived in git history. Equivalent information:

- **CHANGELOG.md** -- chronological release log
- **COMPLETED.md** -- shipped work by area
- **ROADMAP-COMPLETED.md** -- original completed phase tables
- **ROADMAP-NEXT.md** -- Wave A-K detail (shipped through v1.28.x)
- **TODO.md** -- compact active execution queue
- **`.ai/research/2026-05-17/`** -- 20 research artifacts
- **`docs/archive/research/`** -- archived research plans
