# OpenCut — Market-Fit & Positioning (Pass 3)

**Audit date:** 2026-05-17 (Pass 3)
**Source:** NLE pricing market-fit subagent + Pass-1 COMPETITOR_MATRIX + Pass-3 IDE-agent subagent.
**Status:** advisory — informs README/marketing copy and prioritisation, not code.

---

## 1. Three "you'd otherwise pay $X for" pitches (the README/marketing line)

| OpenCut feature | What you'd pay for it elsewhere | Local-first / privacy advantage |
|---|---|---|
| **Silence-cut + multicam podcast + styled captions (existing routes)** | **AutoCut $14.99/mo + AutoPod $29/mo + Submagic $16/mo = $60/mo combined ($720/yr)** | Wedding/corporate/legal client-privacy is the #1 SaaS-to-OSS migration driver per Pass-3 subagent; OpenCut already ships every piece |
| **Whisper-local transcript editing + Premiere round-trip** | **Descript Creator $24/mo ($288/yr)** | Descript itself migrated its transcript backend to Whisper (per GH discussion #1515) — OpenCut removes the SaaS middleman. 78% of Descript users only use transcription per Pass-3 subagent |
| **Real-ESRGAN / Video2X upscale via panel** | **Topaz Video AI $299/yr Personal or $699/yr Pro (perpetual killed Oct 3, 2025)** | Topaz's perpetual EOL is the most exploitable competitor move of 2025-2026 per Pass-3 subagent |

**Total combined comparable spend OpenCut replaces: ~$1,400/yr per creator.**

---

## 2. Three categories to deprioritise (weak willingness-to-pay signal)

| Category | Why deprioritise |
|---|---|
| **Avatar generation (HeyGen-class)** | Capex-heavy (model hosting, voice cloning datasets), enterprise-sales motion, watermark-trial funnel doesn't fit MIT free story. OpenCut Wave M `lipsync_echomimic.py` is enough for the dub use case; full avatar pipeline (F148 face-age, F153 Overdub) is a *Next*-tier nice-to-have, not a pillar |
| **OpusClip-style virality scoring as a pillar feature** | Algorithmic moat (training data on what actually went viral), not feature moat; OSS won't catch up; CapCut now ships highlight detection free. OpenCut's Wave H virality score (F### shipped) is "checkbox, not pillar." |
| **Sports highlight detection as a headline** | Referenced as differentiator in pass-1 backlog (F143-class) but no public WTP data; ship F156 OpusClip A/B variants and F### sports-highlights as **checkboxes** under "Shorts pipeline", not as standalone headlines |

---

## 3. One pricing/distribution lesson — Mister Horse Animation Composer

**Animation Composer = ~900,000 installs**, free shell + paid packs model. **The only Premiere extension ecosystem that produced 6-figure install footprint without VC.**

**Translation to OpenCut:**
- **Free MIT core panel** — every existing route, every existing feature, no payment surface (current state, keep)
- **Optional paid model packs** distributed through **Adobe Exchange** (the canonical Premiere extension storefront):
  - Premium voice pack (50 cloned voices, MIT-acceptable training data)
  - Premium upscaler pack (Video2X / ESRGAN model bundle pre-downloaded)
  - Premium B-roll footage library (Pexels/Pixabay curated)
  - Premium LUT pack (cinematic film stocks)
- **Avoid Topaz's mistake** of selling a one-time licence on AI-compute features — those operating costs only get worse with model size

**Distribution channels:**
1. GitHub Releases (current — primary)
2. Adobe Exchange (paid packs, currently unused — F268)
3. Homebrew Cask once F202 ships (macOS)
4. Flathub once F249 ships (Linux)
5. winget/Chocolatey (Windows secondary)

---

## 4. SaaS-to-OSS migration drivers — OpenCut's tailwinds

Pass-3 subagent identified the top three drivers pushing creators FROM SaaS TO OSS:

1. **Price** — zero monthly fees
2. **Privacy** — sensitive client footage cannot leave the editor's machine
3. **Watermark/moderation removal** — SaaS imposes brand limits and content rules

OpenCut already wins on all three. The README copy should lead with this directly rather than burying it in "local-first" jargon.

**Counter-drivers (OSS-to-SaaS migration) — OpenCut's gaps:**

1. **Install complexity** — "ComfyUI is gold standard for privacy but punishing for non-technical users." → OpenCut's installer (F202 Apple notarisation + Flatpak F249 + Inno Setup) addresses this; F261 missing macOS/Linux launchers must close
2. **VRAM requirements** — Wan 2.2 needs 16GB+; many creators don't have it → OpenCut needs F063 cost estimator UI (shows VRAM requirement before triggering); F176 eval dataset bundle to publish actual hardware tiers per feature
3. **Lack of UI** — pure CLI tools lose to SaaS UIs → OpenCut's CEP+UXP panels are the moat here; F252 WebView UI keeps the UI as good as SaaS competitors

---

## 5. Per-creator-type pain-point map (where each feature lands)

| Creator type | #1 pain point | OpenCut feature that addresses | Tier |
|---|---|---|---|
| Vlog / short-form | Caption styling + B-roll insertion + silence removal | Wave H styled captions + broll_insert + silence — **shipped** | parity |
| Podcast | Multicam camera switching + silence | Wave M sports-highlights + multicam_xml + silence — **shipped** | parity |
| Docs / interview | Transcript-based editing | core/transcript_timeline_edit.py — **shipped** | parity |
| Tutorial / screen-recording | Cursor-zoom + chapter markers | Wave H cursor-zoom + Wave A chapters — **shipped** | parity |
| Wedding | Color grading + music sync + long-form assembly | color_match + beat_sync + declarative_compose — **shipped** but UX scattered | needs **F143 conductor** |
| Sports | Highlight detection | Wave M sports-highlights — **shipped** (Tier 1) | parity |
| Corporate | Brand kit + template assembly | K1.2 brand_kit + project_templates — **shipped** | parity |
| Legal / forensic | Privacy + provenance + zero-cloud | F110 C2PA + F112 auth + F119 brand kit + local-only — **shipped** | **moat** |

**The wedding case is the most under-served** — every individual piece is shipped, but the orchestration (color-match this clip to that reference clip, sync cuts to music beats, assemble a 4-minute highlight reel) requires the user to drive 5+ separate routes. **F143 chat-conductor + F145 wedding-specific skill = unique advantage no other tool offers.**

---

## 6. Subscription vs perpetual vs free — what 2026 says

| Model | 2026 trajectory | Winner / loser |
|---|---|---|
| Subscription SaaS (Descript, Captions, Submagic, OpusClip, HeyGen, ElevenLabs) | ARR economics dominate; even Topaz capitulated Oct 2025 | **Winner** for AI-compute features |
| Perpetual one-time (Topaz pre-2025, Neat Video, Beauty Box, FilmConvert, Mocha Pro) | Holds for creative-asset value (LUTs, plugins, effects); fails for compute-heavy AI | **Loser** for AI-compute, **winner** for creative assets |
| Free + paid packs (Mister Horse Animation Composer) | ~900k installs, only successful non-VC scale model | **Winner** for distribution scale |
| Free / MIT / OSS (Kdenlive, OpenShot, Shotcut, DaVinci Resolve free) | Beginner/Linux niche **except** DaVinci which dominates "Premiere alternative" segment; >85% of indie editors never activate DaVinci Studio | **Winner** for free-tier user base when paired with a real UI |
| Free MIT + optional paid packs (proposed for OpenCut) | Untested at OpenCut scale; Animation Composer proves it works in Premiere ecosystem | **Recommended** |

---

## 7. README/marketing copy recommendations (concrete)

**Current README opening (paraphrased from Pass 1 STATE_OF_REPO):**
> A free, open-source Premiere Pro extension that brings AI-powered video editing automation … all running locally on your machine. No subscriptions, no cloud, no API keys required.

**Proposed Pass-3 lead (incorporates market-fit findings):**
> **OpenCut replaces ~$1,400/year of video-editing subscriptions** with a free, MIT-licensed Premiere Pro extension. Silence-cut + podcast multicam + styled captions (vs $60/mo AutoCut + AutoPod + Submagic). Whisper-local transcript editing (vs $24/mo Descript). Real-ESRGAN upscale (vs $299/yr Topaz). All running locally — your footage never leaves your machine, so legal, corporate, and wedding clients can never be the breach.

**Anchor numbers in the badge bar:**
- "Replaces ~$1,400/yr"
- "1,359 routes" (already)
- "27 MCP tools" (new)
- "47 AI models, all opt-in" (new)
- "100% local, 0 API keys required for core" (already, sharper)

---

## 8. New F-numbers from Pass-3 market positioning

| F# | Title | Priority | Effort | Notes |
|---|---|---|---|---|
| F268 | Adobe Exchange storefront listing for free OpenCut core panel | Next | M | Mister Horse Animation Composer's distribution channel; ~900k installs proof |
| F269 | Premium model-pack bundling format + skeleton (`~/.opencut/packs/`) | Later | M | Free shell + paid packs, Mister Horse model |
| F270 | README marketing copy refresh ("replaces $1,400/yr") | Now | S | Lead with concrete dollar comparisons |
| F271 | Per-feature VRAM requirement UI surface | Next | S | Closes the #2 OSS-to-SaaS migration driver |
| F272 | Wedding-specific Skill (F145) — color match + beat sync + 4-min reel | Next | M | Most under-served creator type per §5 |
