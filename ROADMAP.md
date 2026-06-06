# OpenCut -- Roadmap

**Version**: 5.0
**Updated**: 2026-06-06
**Baseline**: v1.32.0 -- 1,534 routes, 107 blueprints, 599 core modules, 9,200+ tests, CEP + UXP panels, DaVinci Resolve bridge, MCP server
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
| E15 | CEP i18n migration | Rolling batches (153/~160) | Removing bare-English strings from 15,263-line main.js; `TODO.md` last synced this at v4.265 / batch 153. |
| F202 | macOS notarization live acceptance | Blocked: needs GitHub secrets | Repository wiring exists. Deadline: **2026-09-01**. |
| F252 | UXP WebView cutover | Blocked: needs Premiere UDT evidence | Bolt UXP scaffold exists. |

### Security (immediate)

| Item | Severity | Detail |
|---|---|---|
| PyTorch CVE-2025-32434 | CVSS 9.3 | torch.load(weights_only=True) can still execute code |
| Flask CVE-2026-27205 | Info Disclosure | Session Vary: Cookie header missing in <=3.1.2 |
| Pillow CVE-2026-25990 | High | Out-of-bounds write |
| OpenEXR CVE-2026-39886/40244/40250 | High | HTJ2K/DWA decoder overflow |
| Python 3.10/3.11 EOL | Planning | Both EOL October 31, 2026 |

### Governance and hardening (RA-01 through RA-36)

| ID | Item | Effort | Why |
|---|---|---|---|
| RA-01 | Ruff target-version alignment | S | py39 vs >=3.11 skew |
| RA-02 | requirements/pyproject alignment | S | Pin drift weakens advisories |
| RA-03 | Direct typed error logging | S | OpenCutError leaves no log |
| RA-04 | Request ID in error bodies | S | Missing from JSON envelope |
| RA-05 | SQLite PRAGMA user_version | M | Closed 2026-06-06: local SQLite stores now use explicit `user_version` migrations |
| RA-06 | Destructive wipe backup | M | Closed 2026-06-06: local SQLite destructive maintenance paths now expose dry-run counts, optional backups, and audit metadata |
| RA-07 | Job result_json cap | S | Closed 2026-06-06: oversized job results spill to content-addressed local files |
| RA-08 | DB compaction diagnostic | S | Closed 2026-06-06: local SQLite diagnostics report page, freelist, WAL, and file-size posture |
| RA-09 | Timeline-native captions | L | UXP createCaptionTrack() |
| RA-10 | Magic clips macro | L | Long-to-shorts table-stakes |
| RA-11 | UXP least-privilege filesystem | M | fullAccess too broad |
| RA-12 | Hybrid plugin validator | M | .uxpaddon packaging |
| RA-13 | UXP external launch perms | M | Missing launchProcess allowlist |
| RA-14 | WebView permission split | M | Dev vs release permissions |
| RA-15 | [all] advisory decision | M | 5 unwaived advisories |
| RA-16 | Adobe dist-tag tracking | S | release-* tags untracked |
| RA-17 | UXP manifest schema guard | M | Missing manifestVersion |
| RA-18 | UXP deprecation sentinel | M | Block deprecated APIs |
| RA-19 | UXP clipboard permission | S | Missing declaration |
| RA-20 | UXP confirmation guard | S | Raw window.confirm |
| RA-21 | Python 3.13 classifier proof | M | Advertised but untested |
| RA-22 | Release Full Node pin | S | Unmatched Node versions |
| RA-23 | GitHub Actions SHA pins | M | Mutable tag references |
| RA-24 | Release Full token perms | M | Over-broad contents: write |
| RA-25 | Docker dependency surface | M | Retired packages installed |
| RA-26 | Docker runtime parity | M | Old paths, missing ports |
| RA-27 | Docker GPU compose | S | Closed 2026-06-06: README and compose docs now use the committed GPU profile command |
| RA-28 | README count gate | S | Closed 2026-06-06: README non-badge count claims are checked against generated/live counts |
| RA-29 | Docker fail-closed | M | Unquoted specifiers |
| RA-30 | Docker build-context hygiene | S | Missing .env*/.log ignores |
| RA-31 | Adobe tracker exit-code | S | Exit codes lost |
| RA-32 | Adobe tracker labels | S | Unseeded labels |
| RA-33 | Label dry-run without gh | S | Requires CLI |
| RA-35 | Release SBOM fidelity | M | Closed 2026-06-06: declared-SBOM artifact naming and CycloneDX fidelity metadata |
| RA-36 | CEP UNC/HGFS-safe Node | M | Shared-folder paths |

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
| Security | **Action needed** | 4 CVEs (PyTorch CVSS 9.3, Flask, Pillow, OpenEXR); 36 RA-items |
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
| Release CI | PR Fast and Release Full workflows | `.github/workflows/pr-fast.yml`, `.github/workflows/build.yml` | Strong tests, weak token posture | RA-24 remains concrete: Release Full has workflow-level `contents: write`; PR Fast is already read-only. |
| Docker distribution | Multi-stage image, `.dockerignore`, compose | `Dockerfile`, `.dockerignore`, `docker-compose.yml`, README Docker section | Medium | RA-25/RA-29/RA-30 remain concrete: optional dependency install masks failures and `.dockerignore` lacks explicit secret/log exclusions. |

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

**Acceptance criteria:**

- [ ] Non-upload jobs run with `contents: read`.
- [ ] Release upload steps still have enough permission to publish tag assets.
- [ ] Static tests fail if workflow-level `contents: write` returns.

**Priority:** P1. **Effort:** M. **Confidence:** High.

### Cycle 4: UXP/WebView migration audit

Adobe's Premiere UXP docs now position UXP as the current Premiere extensibility
platform for Premiere 25.6+, and Adobe's manifest docs say Premiere supports
manifest version `5`. The live OpenCut manifest has `id`, `name`, `version`,
`main`, host `minVersion: 25.6`, and loopback network allowlist, but it does not
declare `manifestVersion`; it also still grants `localFileSystem: "fullAccess"`.

| Candidate | Evidence | Recommendation | Priority |
|---|---|---|---|
| RA-17 manifest schema guard | `extension/com.opencut.uxp/manifest.json`; Adobe UXP manifest docs | Add explicit manifest-version validation and schema drift tests before claiming packaged UXP readiness. | P1 |
| RA-11 least-privilege filesystem | `localFileSystem: "fullAccess"` in UXP manifest | Split development/full-access needs from release needs; document any required full-access paths and test the narrowed manifest once WebView cutover is validated. | P1 |
| RA-19 clipboard permission | UXP permission queue in `TODO.md`; manifest lacks clipboard declaration | Add a central clipboard helper and manifest permission only when copy actions need it; test fallback copy messages. | P2 |
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
| RA-25 Docker dependency surface | `Dockerfile` installs `pydub` and `deep-translator` even though Python extras moved away from those packages elsewhere. | Align Docker installs with tracked `requirements.txt`/`pyproject.toml` extras so retired packages cannot return through the container path. | P1 |
| RA-29 Docker fail-closed installs | `RUN pip install ... || echo "Some optional deps failed -- continuing"` masks dependency failures. | Replace masked installs with explicit optional groups or a generated constraints file; fail closed unless the feature is intentionally omitted. | P1 |
| RA-30 build-context hygiene | `.dockerignore` excludes broad dirs but lacks explicit `.env*`, `*.log`, credentials, cache DB, and local state patterns. | Add a secret/log/state section and a test that compares `.gitignore` sensitive patterns with `.dockerignore`. | P1 |
| Docker runtime parity | README mentions GPU compose file while only `docker-compose.yml` was surfaced in the initial file scan. | Keep README, compose files, and exposed HTTP/WebSocket ports synchronized through a docs test. | P2 |

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
keys, and 0 dead keys over the 150-key baseline. This means the E15 queue is no
longer primarily about broad missing-key drift; it is now a finish-line cleanup
around the remaining dead keys and any still-bare user-visible strings not
covered by the static scanner.

| Candidate | Evidence | Recommendation | Priority |
|---|---|---|---|
| E15 dead-key cleanup | `scripts/i18n_lint.py --json` reports 55 dead keys | Group dead keys by UI domain, then either rewire genuinely reusable strings or delete obsolete keys with focused migrated-key tests. | P2 |
| E15 scanner coverage | `scripts/i18n_lint.py` only scans `data-i18n*` and `t(...)` calls | Add targeted scanner coverage for dynamic `innerHTML`, option-label builders, tooltip/title helpers, and generated command-palette labels where false negatives are likely. | P2 |
| E15 roadmap status | `TODO.md` says batch 153; `PROJECT_CONTEXT.md` logs through pass 264 | Keep `ROADMAP.md` status tied to linter facts, not older dead-key counts. | P1 |

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
| RA-17 manifest-version guard | Live `manifest.json` lacks `manifestVersion`; dormant `bolt-webview/uxp.config.ts` uses `manifestVersion: 6`; Adobe docs currently describe Premiere manifest version `5`. | Add a manifest compatibility test that resolves the supported version from current Adobe docs or package schema, then fails if live and scaffold manifests diverge. | P1 |
| RA-11 filesystem permission split | Live and scaffold configs use `localFileSystem: "fullAccess"` | Make a release manifest profile that requests only required file operations after the UDT capture proves which host paths need direct access. | P1 |
| RA-19 clipboard permission | TODO tracks clipboard permission; manifest does not declare clipboard-specific permissions | Centralize copy behavior, then declare the narrow permission only if UXP requires it for actual copy flows. | P2 |
| RA-20 confirmation guard | TODO tracks raw `window.confirm` / beta alert posture | Search UXP/CEP code for confirmation APIs and replace with a panel-native dialog or explicit beta-gated helper. | P2 |

### Cycle 9: Docker/runtime parity audit

The repo has `docker-compose.yml` with both CPU and GPU services. The GPU path
is selected through `docker compose --profile gpu up opencut-server-gpu` so the
profiled service is targeted directly and the default CPU service does not
collide on port 5679. README and Dockerfile copy-paste commands now match the
committed compose file and the non-root `/home/opencut/.opencut` data path.

| Candidate | Evidence | Recommendation | Priority |
|---|---|---|---|
| RA-27 Docker GPU compose command | README previously referenced a missing `docker-compose.gpu.yml`; `docker-compose.yml` uses `profiles: [gpu]` | Closed by updating README/compose comments to `docker compose --profile gpu up opencut-server-gpu` and adding a release-smoke docs guard. | Done |
| RA-26 non-root volume docs | `docker-compose.yml` uses `/home/opencut/.opencut`; Dockerfile comments previously used `/root/.opencut` | Dockerfile comments now match the non-root home path; broader RA-26 port/runtime posture remains open. | Partial |
| RA-25 Docker dependency surface | Dockerfile still installs `pydub` and `deep-translator`; tests elsewhere forbid `pydub` pins and project extras removed it | Generate Docker optional dependency install lists from tracked package metadata or remove retired packages from the container path. | P1 |
| RA-29 fail-closed install | Dockerfile masks optional dependency failures with `|| echo "Some optional deps failed -- continuing"` | Replace with explicit optional-stage selection or a known-fail waiver file; release builds should fail when declared Docker deps fail to install. | P1 |
| RA-30 build-context hygiene | `.gitignore` excludes `.env`, `.env.*`, `*.key`, `*.pem`, `credentials*.json`, and `*.log`; `.dockerignore` does not | Mirror the sensitive `.gitignore` patterns into `.dockerignore` and add a test for secret/log pattern parity. | P1 |
| Docker port posture | Dockerfile exposes 5679/5680; README and UXP docs mention backend HTTP 5679, WebSocket 5680, MCP 5681 | Decide whether containerized MCP is supported; if yes, expose/map 5681 and document auth posture. If no, state HTTP/WebSocket-only container boundary. | P2 |

### Cycle 10: GitHub Actions supply-chain audit

OpenCut's workflows use tag-pinned third-party Actions:
`actions/checkout@v4`, `actions/setup-python@v5`, `actions/setup-node@v4`,
`actions/upload-artifact@v4`, and `actions/github-script@v7`. GitHub's Actions
settings docs describe `OWNER/REPOSITORY@TAG-OR-SHA` allowlisting and explicitly
support requiring full-length SHA pins. GitHub's artifact attestation docs also
support signed build-provenance claims for binaries and container images.

| Candidate | Evidence | Recommendation | Priority |
|---|---|---|---|
| RA-23 full-SHA action pins | `.github/workflows/*.yml`; archived Cycle 12 research; GitHub action allowlist docs | Pin non-local workflow `uses:` references to full-length SHAs, keep adjacent version comments, and add a static test that rejects mutable tags/branches. | P1 |
| RA-24 token least privilege | `.github/workflows/build.yml` workflow-level `contents: write`; PR Fast and Adobe tracker already scoped narrower | Default Release Full to `contents: read`; isolate release-upload permissions to the smallest tag/manual upload boundary. | P1 |
| RA-22 Release Full Node pin | PR Fast uses `actions/setup-node@v4` with Node 22; Release Full runs panel npm gates on Linux without a matching setup-node step | Add explicit Node 22 setup before Release Full panel gates so npm advisory/build evidence matches PR Fast. | P1 |
| Release provenance attestation | Release Full uploads binaries, installers, Linux packages, and SBOM but no `attest-build-provenance` step appears in workflow scan | Add GitHub artifact attestations for release artifacts and SBOM after RA-24 narrows permissions; document verification commands. | P2 |

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
| RA-41 destructive-operation plan contract | `/models/delete`, `/whisper/clear-cache`, `/plugins/uninstall`, `/cache/cleanup`, `/cache/invalidate`, `/captions/cache/clear`, `/system/temp-cleanup/sweep`, `/logs/clear`, `/presets/delete`, `/workflows/delete`, `/workflow/delete`, `/queue/clear`; tests mainly cover status codes | Advanced 2026-06-06: shared destructive dry-run plan and confirm-token helpers now protect `/queue/clear`, `/logs/clear`, `/captions/cache/clear`, `/whisper/clear-cache`, and `/models/delete`; continue extending the contract to remaining cache cleanup/invalidation, temp-cleanup, plugin cleanup, and other destructive routes. | P1 |
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
`/timeline/srt-to-captions`, then tells the user to use CEP or native captions
flow to place the parsed cues. It does not call a host action that creates a
caption track.

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

- [ ] Generating captions writes a versioned sidecar for SRT/VTT/ASS/JSON
      exports without changing existing response fields.
- [ ] The sidecar round-trips all metadata currently preserved by
      `export_json()`, including words, speaker, language, confidence, and
      review fields.
- [ ] Missing or stale sidecars produce explicit warnings, not silent metadata
      loss.
- [ ] Tests prove SRT-only parse remains lossy while sidecar-backed parse is
      metadata-preserving.

**Risks:** Host track-item identifiers may not be stable across Premiere
sessions; the sidecar must treat host locators as hints and fall back to
time/text matching.

#### RA-47 caption diff and apply endpoints

**Priority:** P1. **Effort:** L. **Confidence:** High.

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

- [ ] Diff returns counts, per-cue changes, confidence, warnings, and an
      unchanged/changed summary suitable for a review UI.
- [ ] Apply stores a new revision linked to the original `transcript_cache_key`
      and source file hash.
- [ ] Apply is idempotent for unchanged SRT plus matching sidecar.
- [ ] A no-sidecar request still works as a lossy timing/text diff and labels
      metadata preservation as unavailable.

**Risks:** SRT editors can reorder, split, or merge cues in ways that require
fuzzy matching; start with deterministic `caption_id` matching and only then
fallback to timing/text similarity.

#### RA-48 UXP caption-track read bridge

**Priority:** P1. **Effort:** M. **Confidence:** Medium-high.

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

- [ ] The action appears in the UXP direct-action manifest only when fixture
      coverage proves a non-mutating read path.
- [ ] Empty projects, no active sequence, no caption tracks, and API-missing
      states return distinct, test-covered reasons.
- [ ] Snapshot output can be passed directly to `/captions/round-trip/diff`.
- [ ] The implementation does not claim UXP caption creation/import support
      until an official Adobe write API is documented and live-tested.

**Risks:** Adobe forum evidence indicates `getTrackItems()` parameter behavior
has been confusing; UDT must capture the exact parameter contract before this is
treated as reliable.

#### RA-49 CEP/hybrid caption write contract

**Priority:** P1. **Effort:** M. **Confidence:** High.

**Evidence:** CEP `importCaptions()` tries `seq.addCaptionTrack()` and
`captionTrack.insertClip()` before falling back to video-track/project-panel
import; `ocAddNativeCaptionTrack()` currently returns only `success` and
`captions_added`.

**Recommended implementation:** Normalize CEP/hybrid caption write results into
a richer payload: `success`, `captions_added`, `imported`,
`added_to_timeline`, `bin_name`, `caption_track_index`, `fallback_path`,
`sidecar_path`, warnings, and host version details. The UXP panel should
surface this as a deliberate "Place captions using CEP/hybrid" handoff until
UXP has a documented write API.

**Acceptance criteria:**

- [ ] `ocAddNativeCaptionTrack()` accepts RA-46 sidecar-aware segment payloads
      while remaining compatible with the existing `start/end/text` array.
- [ ] The result payload distinguishes project import, native caption-track
      placement, video-track fallback, and manual-drag fallback.
- [ ] JSX mock tests assert the richer result contract.
- [ ] UXP UI copy points to a concrete bridge action instead of a generic
      "CEP or native captions flow" instruction.

**Risks:** CEP caption behavior can differ by Premiere version; the contract
must record fallback mode rather than treating all successful imports as native
timeline placement.

#### RA-50 caption metadata-loss regression tests

**Priority:** P1. **Effort:** S. **Confidence:** High.

**Evidence:** Existing tests cover SRT text/timing round trips and CEP
`ocAddNativeCaptionTrack` basics, but no scanned test asserts that speaker,
review, style, and transcript-cache metadata survive a full caption round-trip.

**Recommended implementation:** Add focused tests for RA-46 through RA-49 before
expanding UI work: route tests for sidecar creation/diff/apply, unit tests for
fuzzy cue matching, UXP static tests for snapshot action registration, and JSX
mock tests for richer CEP write results.

**Acceptance criteria:**

- [ ] Tests demonstrate SRT-only parse drops non-SRT metadata.
- [ ] Tests demonstrate sidecar-backed diff preserves metadata across
      edit/export/import.
- [ ] Tests cover split/merge/deleted/inserted cue classifications.
- [ ] Tests cover stale sidecar warnings and no-sidecar degraded mode.

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

**Priority:** P1. **Effort:** M. **Confidence:** High.

**Recommended implementation:** Add `opencut/core/magic_clips.py` with
dataclasses for `MagicClipsConfig`, `MagicClipsPlan`, `MagicClipCandidate`, and
`MagicClipStep`. Add `/video/magic-clips/plan` or
`/video/shorts-pipeline/dry-run` as a sync route that returns the plan without
rendering. If transcript/highlight data is not already cached, the plan route
should return `requires_analysis` steps instead of silently starting ASR or
FFmpeg work; a separate async "analyze and plan" job can populate the cache.

**Acceptance criteria:**

- [ ] Plan output contains stable IDs, source path hash, config hash, candidate
      windows, step dependencies, estimated outputs, and reasons.
- [ ] Dry-run never writes rendered media and never runs expensive ASR/FFmpeg
      analysis unless an explicit `analyze=1`/async mode is requested.
- [ ] Existing `/video/shorts-pipeline` can accept a plan ID or candidate IDs and
      render exactly that approved subset.
- [ ] Unit tests compare deterministic JSON snapshots for cached-transcript,
      no-cache, and invalid-config scenarios.

#### RA-52 Candidate scoring and explainable selection

**Priority:** P1. **Effort:** M. **Confidence:** High.

**Recommended implementation:** Feed highlight windows from
`opencut.core.highlights.extract_highlights` into a ranking layer that uses
`opencut.core.virality_score.rank()` plus transcript hook text, duration fit,
speaker continuity, and optional domain detectors such as sports/highlight
modules. Persist `selection_reason`, `score_breakdown`, `fallback_mode`, and
`rejected_candidates` in the plan.

**Acceptance criteria:**

- [ ] Candidate ordering is deterministic for identical transcript/media inputs.
- [ ] The plan includes per-candidate reasons that can be shown in UXP/CEP
      without re-running model calls.
- [ ] When LLM scoring fails or is disabled, heuristic scoring still returns
      usable candidates and clearly marks the fallback.
- [ ] Tests cover tie-breaking, too-short windows, overlapping highlights, and
      malformed transcript segments.

#### RA-53 Platform preset and multi-ratio export contract

**Priority:** P1. **Effort:** M. **Confidence:** High.

**Recommended implementation:** Replace panel-side dimension maps with preset
IDs from `opencut/core/export_presets.py` and a shared option endpoint. Magic
Clips plans should support one or more platform targets per approved candidate,
with generated render steps for `youtube_shorts`, `tiktok`, `instagram_reels`,
square/social feed, or custom dimensions. Reuse `batch_reframe.py` and
`smart_reframe.py` when a multi-ratio batch is requested.

**Acceptance criteria:**

- [ ] UXP and CEP send preset IDs, not ad-hoc width/height maps.
- [ ] Plans expose platform constraints, target dimensions, max duration, and
      output filename templates before rendering.
- [ ] Rendering enforces preset duration/dimension constraints consistently with
      `export_presets.py`.
- [ ] Tests verify at least YouTube Shorts, TikTok, Reels, and square outputs.

#### RA-54 Review-board UI parity for UXP and CEP

**Priority:** P1. **Effort:** M. **Confidence:** Medium.

**Recommended implementation:** Add a Magic Clips review board in both UXP and
CEP surfaces: candidate cards with score, reason, transcript excerpt, platform
targets, caption style, thumbnail candidates, and approve/reject toggles. UXP
should reach parity with CEP's duration/platform/LLM controls, while CEP should
adopt the new dry-run plan endpoint instead of only posting the render job.

**Acceptance criteria:**

- [ ] Both panels can preview a plan, approve/reject candidates, and render only
      approved candidates.
- [ ] The UI clearly separates "Plan", "Analyze", and "Render" states.
- [ ] Candidate controls do not require copying raw file paths between the
      Magic Clips and A/B variants panels.
- [ ] Static tests cover route wiring and payload parity for UXP and CEP.

#### RA-55 Checkpointed and resumable Magic Clips jobs

**Priority:** P1. **Effort:** L. **Confidence:** Medium.

**Recommended implementation:** Persist the plan and intermediate state under a
run directory before rendering. Record transcript cache IDs, highlight IDs,
trimmed clip paths, reframe outputs, caption files, thumbnail paths, and final
exports in a manifest that can be resumed. Wire job resume metadata to resume
from the first missing/invalid step rather than restarting the whole pipeline.

**Acceptance criteria:**

- [ ] Cancelled jobs preserve completed intermediates and report the next
      resumable step.
- [ ] Resume skips completed steps when config/source hashes still match.
- [ ] Temp cleanup does not delete files referenced by a resumable manifest.
- [ ] Tests simulate cancel-after-transcribe, cancel-after-first-render, and
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

- [ ] Every rendered clip has a manifest record with enough metadata to audit
      why it was selected and how it was rendered.
- [ ] Manifest schema version is explicit and covered by fixture tests.
- [ ] The manifest can represent multi-platform variants without duplicate
      candidate records.
- [ ] UXP/CEP can open the output folder and display completed bundle contents
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

1. Cycle 29: Continue RA-41 on render-cache cleanup/invalidation, temp-cleanup sweep, plugin quarantine cleanup, and any remaining destructive routes without the shared confirmation contract.
2. Cycle 30: Inspect caption round-trip implementation fixtures for RA-46 through RA-50.
3. Cycle 31: Inspect sequence-index and marker metadata workflows for reusable host locator patterns.
4. Cycle 32: Inspect Magic Clips implementation fixtures for RA-51 through RA-56.
5. Cycle 33: Revisit Adobe tracker drift after the next scheduled npm publish window.

### Continuation State

#### Last completed cycle

Cycle 28: Destructive clear confirmation plans.

#### Current focus

Continue from active release-trust, migration hardening, Docker hardening, and
product workflow specs. RA-05/RA-37, RA-06/RA-40, RA-07/RA-38, RA-08/RA-39,
RA-16, RA-27, RA-28, RA-31, RA-32, RA-33, RA-35, RA-42, RA-43, RA-44, and
RA-45 are closed, and the bootstrap dev-check guard is in place; RA-41 now has
shared dry-run/confirm-token helpers for queue, log, caption-cache,
Whisper-cache, and model-cache clears and should continue across remaining
cache cleanup/invalidation, temp-cleanup, plugin cleanup, and other destructive
routes without the shared confirmation contract.

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
- E15's live i18n linter passes with 55 dead keys and 0 missing keys; remaining
  work should target dead-key cleanup and scanner coverage, not broad missing-key
  repair.
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
- Release Full still has workflow-level `contents: write`, mutable action tags,
  and no artifact attestation step in the scanned workflows.
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

1. Inspect local DB migration implementation shape and test fixture needs for RA-37 through RA-40.
2. Inspect destructive-operation implementation shape and test fixture needs for RA-41 through RA-45.
3. Continue release-trust hardening on RA-15/RA-17+ or Docker RA-25/RA-26/RA-29/RA-30.

#### Unprocessed leads

- GitHub Actions artifact attestations for release provenance.
- UXP `manifestVersion` and WebView permission split specifics.
- Whether RA-26 should map/expose MCP port 5681 in containerized runs or keep
  Docker documented as HTTP/WebSocket-only.
- Whether the Magic Clips plan endpoint should be `/video/magic-clips/plan`,
  `/video/shorts-pipeline/dry-run`, or both with one canonical core planner.
- Whether RA-51 through RA-56 should be added as separate active TODO rows or
  nested under the existing RA-10 Magic Clips macro row.
- Whether RA-46 through RA-50 should be added as separate active TODO rows or
  nested under the existing RA-09 timeline-native captions row.
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
