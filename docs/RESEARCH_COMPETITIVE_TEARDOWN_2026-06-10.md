# Competitive Teardown — AI Video Editing for Premiere Pro Workflows

**Prepared for:** OpenCut v1.32.0 roadmap planning
**Date:** 2026-06-10 (all prices/features access-dated; this market ships quarterly — sherlock gradings carry a ~3-month shelf life)
**Method:** 5-angle adversarially-verified web research (107 claims extracted, 25 verified by 3-vote panels, 19 confirmed / 6 killed) + 6 parallel census workstreams. Confidence labels: **OBSERVED** (official/primary page), **REPORTED** (user/review/secondary), **INFERRED** (deduction), **UNVERIFIED** (could not confirm — never silently guessed).

---

## The headline findings

1. **"Local processing" alone is NOT the wedge.** Adobe's Media Intelligence search runs fully on-device ("footage, analysis data, and searches never leave your computer" — OBSERVED, Adobe helpx), and DaVinci Resolve's and Final Cut's entire AI suites are local and unmetered. OpenCut's durable wedge is the *combination*: free + MIT + unlimited + cross-NLE + API/MCP scriptable + depth (review UI, filler/repeated-take detection) + zero per-minute fees. Argue locality per-feature (Adobe's caption translation, Firefly editor, and Generative Extend ARE cloud/credit-gated).
2. **The Transcriptive precedent is the existential risk template.** Digital Anarchy killed Transcriptive Rough Cutter, PowerSearch, and its web services effective **May 1, 2026**, citing Adobe's native transcription plus the CEP→UXP transition (OBSERVED, digitalanarchy.com EOL post). Every CEP transcript/caption panel — including OpenCut's CEP path — faces the same two vectors. UXP migration is a P0 platform item, not housekeeping.
3. **Credit/subscription rage is the single richest complaint vein in the market** (Opus, Captions, Filmora, CapCut, Descript, Submagic — 15+ verbatim quotes below). "No credits, no upload, no expiry, no watermark" is a positioning line every competitor's own Trustpilot page proves.
4. **Even the category leader's multicam is distrusted.** AutoPod users report spending "just as much time fixing the edit as editing from scratch." Multicam auto-switching that editors can *tune and review* is an open leapfrog, not a solved problem.
5. **Edge-TTS is a cloud service (Microsoft).** OpenCut's "100% local" tagline has one real exception in the current inventory. Fix it (local TTS swap) before competitors or reviewers find it. (INFERRED risk; the dependency is factual.)

---

# Part 1 — Competitor census

35+ products profiled across five categories. Pricing OBSERVED from official pricing pages on 2026-06-10 unless marked.

## 1A. Premiere Pro plugins/panels (direct competitors)

### AutoPod — autopod.fm
- **Price:** $29/mo single tier, annual = 1 month free, 30-day trial (OBSERVED). No enterprise tier.
- **Locus:** local audio-activity analysis of separate mic tracks (REPORTED/INFERRED — no official privacy statement exists; mechanism confirmed by failure mode: shared audio track breaks speaker detection). Online license activation.
- **Integration:** CEP panels; full timeline write-back (finished multicam sequences, social resizes, jump cuts).
- **Depth:** Multi-Camera Editor up to 10 cams + 10 mics, auto-switch by active mic, configurable shot frequency (3 wide-shot settings only — users want finer control), presets. **No vision/face analysis — audio-only switching** (INFERRED from mechanism + failure reports). Jump Cut Editor (per-mic dB thresholds). Social Clip Creator (3 aspect ratios, watermarks, batch export).
- **JTBD:** video-podcast multicam producers.
- **Momentum:** no public changelog (UNVERIFIED). Traction: strong SEO/affiliate ecosystem; piracy-seeking threads signal price friction.
- **Complaints:** edit quality ("edits it makes are often bad… I end up changing a lot anyway"), wide-shot overuse with no fine-tuning, shared-audio failure, subscription fatigue driving churn to one-time-price competitors (quotes C1–C4, Appendix).

### FireCut — firecut.ai
- **Price:** Starter $19/mo ($10 annual), Pro $34/mo ($20 annual) — captions, podcast multitrack, filler removal, chapters, zooms, B-roll/music; AI voiceover +$10/mo; Team $34/user; lifetime "upon request." Usage caps measured in processed timeline duration (OBSERVED).
- **Locus:** hybrid, cloud-dependent (online license validation mandatory; usage metered server-side; caption rendering local) (OBSERVED/INFERRED).
- **Integration:** CEP panel, deep write-back (cuts, zoom keyframes, caption clips, markers, music/B-roll insertion).
- **Depth:** silence cutting w/ J-cuts, filler words, repetition detection, profanity filter, animated captions 50+ languages w/ emoji animations, AI chapters, Storyblocks B-roll, multitrack podcast editing, shorts highlights.
- **Momentum: HIGH** — 5 releases Mar–Jun 2026 (v1.2.14→v1.2.20), captions-heavy investment (OBSERVED changelog). Ali Abdaal association in reviewer titles (REPORTED).
- **Complaints:** GPU/OpenCL failures across r/IntelArc + r/premiere; caption animations "too quick"; third complaint UNVERIFIED.

### AutoCut — autocut.com
- **Price:** Basic $9.9/mo or $79/yr (silence only); AI $19.8/mo or $178.8/yr; Enterprise $19.9/mo (OBSERVED). 14-day trial.
- **Locus:** hybrid — audio uploaded for analysis per privacy pages (REPORTED, pages 404'd on direct fetch).
- **Integration:** CEP panel (Premiere + Resolve), full write-back.
- **Depth:** Silences, Captions, Podcast multicam, Zoom, Viral shorts, B-roll (Storyblocks), Repeat (bad takes), Profanity, Chapters, Resize.
- **Traction:** Trustpilot 4.7/152 (paid Trustpilot subscription; doesn't reply to negatives — OBSERVED).
- **Complaints:** "total disaster… support is no help"; "spent more time fixing the errors this plugin created than just doing it myself"; J/L-cut style forced, "randomly switch to a person sitting silently" (quotes C5–C7).

### Excalibur — knightsoftheeditingtable.com
- **Price:** $120 one-time perpetual, 2 machines (OBSERVED; left aescripts entirely).
- **Locus:** local; pure scripting automation, no ML.
- **Depth:** keyboard-driven command palette (search-and-apply effects/presets), chained user commands, batch export. Sibling tools ($25–$75): Quiver, Watchtower, Spell Book, etc.
- **Relevance:** workflow-automation expectations benchmark, not an AI competitor. Complaints: none surfaced.

### BeatEdit 2 — Mamoworld via aescripts
- **Price:** $99.99 one-time (OBSERVED). Trial available.
- **Locus:** local (IBT beat tracker + MARSYAS, in-memory processing — OBSERVED).
- **Integration:** CEP panel writing sequence/clip markers; pairs with Premiere's Automate to Sequence for auto beat-cut assembly.
- **Depth:** state-of-the-art-claimed beat detection + "rhythmically relevant peaks," subdivision, quantization, beat-info panel. **Limitations (OBSERVED):** wav/mp3 files only (can't read audio embedded in video; must export first), memory-bound on long files.
- **Momentum:** v2.2.006 May 19, 2026 — maintained, mature. Demand-side: users push back on $100 for beat markers ("What makes this worth $100?" — quote C42).

### Simon Says — simonsaysai.com
- **Price:** PAYG $0.25/min; Starter $15/mo (2 hr); Pro $33/mo (4 hr, $6.50/extra hr); Pro+ $125/mo (30 hr); **on-prem air-gapped tier (Mac standard / Linux-VM enhanced, custom pricing)** (OBSERVED).
- **Locus:** cloud; the air-gapped on-prem tier is itself proof the privacy-constrained segment pays (OBSERVED).
- **Integration:** CEP panel; **"Assemble" text-based paper-cut editor that reconnects selects to source media as a Premiere sequence** (OBSERVED/REPORTED) — the closest commercial analog to white-space #1 below.
- **JTBD:** documentary/broadcast/enterprise post; 100-language transcription/translation.
- **Complaints:** timecode drift requiring manual fixes, "incredibly bad customer service," thin UI (REPORTED, Capterra).

### Transcriptive / Rough Cutter (Digital Anarchy) — **DEAD**
- **EOL May 1, 2026** — all web services and products terminated; reasons stated: Adobe native transcription + CEP→UXP transition + refocus on VFX (OBSERVED). Historical pricing: $199 license + $0.04–0.15/min transcription.
- **Strategic meaning:** first-party sherlocking + platform migration killed a real business in this exact niche. OpenCut's free/local model is immune to the unit-economics part but NOT the CEP part.

### PremiereCopilot — premierecopilot.com (new entrant)
- **Price:** free tier (all 12 tools, daily limits, no watermark); Pro+ $76/yr; lifetime bundles (e.g., Podcast $59) (OBSERVED). Positions as "3× cheaper than the cheapest competitor."
- **Locus:** **cloud** — "All AI processing runs on our secure servers"; claims only waveforms/transcripts/frame-metadata upload (OBSERVED).
- **Depth:** AI timeline assistant, smart captions, podcast multicam (10 cams), silences, bad takes, virals, 99-language subtitles, chapters, diarization-to-tracks, emotion-driven auto zoom, GenAI hub (Kling/Runway/Flux).
- **Relevance:** validates demand for an LLM copilot inside Premiere; attacks on price. Traction UNVERIFIED (too new). It is the closest *feature-shape* competitor to OpenCut — but cloud.

### Premiere Assistant (Cutback) — cutback.video
- **Price:** Free (3 transcriptions/mo); Lite $8/mo; Basic $13/mo (multicam 3 cams); Pro $30/mo (OBSERVED).
- **Locus:** cloud STT (INFERRED from hour quotas). Panel with multicam switching, silence cuts, 29-language translation, animated caption presets, background removal. Young; runs anti-AutoPod SEO. Traction UNVERIFIED.

### Wraith Multi-Cam Editor (Phantom Editor) — phantomeditor.video
- New 2026 AutoPod challenger, **one-time price** (data thin, REPORTED via own blog). Notable only because AutoPod churners name it: "switching to phantom editor since their multicam editor is just a one time price… I'm so done with subscriptions" (quote C3).

## 1B. Adobe first-party AI (the sherlock baseline)

All OBSERVED from helpx/adobe blogs, accessed 2026-06-10; verified 3-0 by adversarial panels unless noted.

| Feature | Status | Locus | Notes |
|---|---|---|---|
| Text-based editing + **pause detection/deletion** (sensitivity slider, delete-all) | Stable, GA | Local | Basic transcript silence-cutting is first-party |
| **Bulk bleep/mute of words & sensitive info** ("Censor Transcript") | v25.6, Nov 2025 | Local | One-pass profanity/PII redaction shipped |
| **Media Intelligence** semantic search (visual + transcript + metadata; v25.6 added audio-by-description + similar-shot/alternate-take) | GA Apr 2025→ | **Local** ("models installed locally, don't require the internet") | Searches *open project*, not cross-library; users report runaway indexing (quote C30) |
| **Caption Translation, 27 languages** | GA Apr 2025 | **Cloud** (third-party translation services per docs) | Offline translation remains open |
| **Generative Extend** (video+audio, 4K, vertical) | GA Apr 2025 | Cloud, Firefly credits | ~2s video limits; no local equivalent exists |
| **AI Object Masking** (one-click select+track; Sharp/Smooth modes in 26.2) | v26.0 Jan 2026 / v26.2 Apr 2026 | Local | Magic-Mask equivalent, quarterly-improved |
| Enhance Speech / Auto Reframe / Scene Edit Detection / Remix | Stable | Mixed | Enhance Speech inserts artifacts in silences (quote C29) |
| **Color Mode** (grading workspace overhaul, 32-bit, operations layers) | Public beta Apr 15, 2026; GA "later 2026" | Local | UX overhaul, NOT auto-grading — color match/LUT-gen not sherlocked |
| **Firefly Video Editor** (browser): multi-track timeline, text-based editing, **Quick Cut** auto-rough-cut, **Prompt-to-Edit (Runway Aleph)** generative pixel edits | Public beta Dec 16, 2025 / Apr 15, 2026 | Cloud, credit-gated | Parallel surface, no Premiere write-back; "Firefly AI Assistant" is coming-soon vaporware (claim refuted 1-2) |

**Cadence:** stable releases v26.0 (Jan) → v26.2 (Apr 2026), roughly quarterly. **Platform signal:** CEP 12 is the final CEP major; ExtendScript supported "through September 2026" for Premiere; UXP graduated from beta in 25.6 with marketplace submissions open (OBSERVED, Adobe dev blog).

## 1C. Standalone editors with Premiere round-trip

### Gling — gling.ai
- **Price:** Free 1 hr/mo (watermark, **no XML export**); Plus $20/mo ($10 annual) 10 hr; Pro $40/mo ($20) 30 hr; Elite $100/mo ($50) 100 hr (OBSERVED). XML export is paywalled.
- **Locus:** hybrid/ambiguous — desktop app + web; marketing references "secure upload systems" while one review claims footage stays local with desktop app (CONFLICTING, UNVERIFIED).
- **Integration:** one-way XML to Premiere/Resolve, FCPXML; no re-import.
- **Depth — the bad-take benchmark:** transcribes, **detects repeated sentences across takes, auto-selects the best delivery**. Independent measurement: 96.7% content-preservation, 85.4% F1, but only 77.5% filler cleanliness (REPORTED, max-productive.ai, Apr 2026). 2026 additions: intentional-pause detection, custom stoplists, noise-floor detection, Aggressive/Natural toggle. Also: captions, auto-framing, multicam, AI B-roll, noise removal.
- **Traction:** 50K+ creators; used by 1.5–2.2M-sub YouTubers (REPORTED).
- **Complaints:** exported MP4 jitter/framerate loss, comedic-timing blindness ("always ruins jokes"), feature bloat regressing core silence editing (quotes C6–C9).

### Timebolt — timebolt.io
- **Price:** Free (watermark); Pro $17/mo or $97/yr; **Lifetime $347**; Teams $97/seat/yr; **Vault enterprise: on-prem, HIPAA/FINRA/FedRAMP, "zero cloud dependency"** (OBSERVED). **UMCHECK filler/bad-take detection is pay-per-use $0.03/min on top of Pro.**
- **Locus:** local desktop for core waveform silence cutting ("Your footage never leaves your machine") — **but UMCHECK sends audio to AWS** (hybrid; contradicts the blanket claim) (OBSERVED/REPORTED).
- **Integration:** Premiere XML **plus a Premiere extension plugin**; FCPXML incl. multicam; Resolve; Camtasia; SRT that re-syncs after edits. Deepest export menu of the silence-cutters.
- **Depth:** 1-hr video jump-cut in ~13s at 0.01s precision; **Punch & FastForward** (punch-in zoom; speed-up instead of cut — continuity-preserving for screencasts); Turbo Mode; retake detection for scripted video (v7.0.7, Nov 2025).
- **Momentum:** active — Vault launch Jan 2026, bad-take detection Nov 2025.
- **Complaints:** slow rendering/hangs on big files; dated UI (REPORTED, G2 summaries).

### Recut — getrecut.com
- **Price:** **$129 one-time** (or $15/mo); trial = 3 free exports (OBSERVED).
- **Locus:** fully local desktop (Mac/Win) (OBSERVED architecture; explicit no-upload claim UNVERIFIED).
- **Integration:** XML to Premiere/FCP/Resolve/ScreenFlow/CapCut + direct MP4. Cuts only — no markers/transcripts.
- **Depth:** deliberately narrow: silence detection/removal, frame-accurate, multi-track/multicam-aware sync-preserving cuts.
- **Momentum:** maintenance cadence (v4.4.1–4.4.4 Mar–Apr 2026). Indie single-dev. Complaints: minimal volume; narrow scope vs Timebolt (INFERRED).
- **Strategic note:** proof that "local + one-time price" sells in this category.

### Descript — descript.com
- **Price:** Free (1 hr transcription, watermark); Hobbyist $24/mo ($16 annual); Creator $35/mo ($24); Business $65/mo ($50). **AI features burn metered credits on top of subscription** (OBSERVED).
- **Locus:** cloud (Descript Drive; ElevenLabs Scribe v2 default transcription since May 2026). **Offline editing was stripped away** (quote C15).
- **Integration:** one-way timeline export to Premiere/FCP/Reaper/Pro Tools preserving source timecode. **Documented losses:** Studio Sound exports only 0%/100%; multicam flattens to stacked layers (open Canny request); no XML re-import (OBSERVED, help docs).
- **Depth — the text-editing UX benchmark:** word-level delete-text→cut-video propagation; one-click filler removal; Overdub voice clone (type corrections into your own voice); Studio Sound intensity slider; Eye Contact; Rooms multiplayer recording; **Underlord agentic co-editor** (NL commands, multi-step plans, model picker: Claude Opus/Sonnet 4.6, Gemini 3.1 Pro, GPT 5.2). What makes the UX better than a transcript panel: edits are *bidirectional and immediate* (the doc IS the timeline — no apply step), corrections synthesize seamlessly via Overdub, and agent edits arrive as reviewable proposals against a project brief.
- **Momentum: category-highest.** Dec 2025–May 2026: Underlord v2 (self-correction, economy mode), Opus 4.6, Media Library, color correction, transitions, **public API + MCP support (Apr 2026)**, automatic edit-review pass. Anthropic case study: agent users export 8.6% more.
- **Traction:** G2 4.6/865; $50M Series C led by OpenAI Startup Fund (2022), ~$550M valuation, no new round found.
- **Complaints:** instability ("froze twice trying to make a 60-second clip"), credit burn ("a month's worth of credits lasts about a day"), Underlord accuracy ("cuts words off mid-word… twice the work"), pricing hike with "almost no notice," perpetual UI churn (quotes C10–C15, C39).

### Wisecut — wisecut.ai
- **Price:** Free 30 min/mo (no downloads on free); Starter $15.75/mo (480 min); Professional $75.67/mo (1,800 min, 4K); Autopilot from $49/mo (OBSERVED).
- **Locus:** cloud. **Premiere integration: NONE** (MP4 + SRT only — weakest round-trip story in the set).
- **Depth:** auto-cut + **smart background music with auto-ducking** (its differentiator), translated subtitles, punch-in zooms, storyboard text editing, channel-to-shorts Autopilot.
- **Complaints:** uploads going missing, slow processing, no mid-sentence micro-cut control (REPORTED, AppSumo).

### Eddie AI — heyeddie.ai
- **Price:** PAYG **$15/export credit**; Pro $167/mo annual (120 exports/yr); Pro+ $333/mo; Ultra $1,250/mo (OBSERVED). Unique per-export monetization, priced for pro teams.
- **Locus:** cloud analysis, desktop front-end; enterprise "on your cloud" (INFERRED).
- **Integration:** deepest NLE menu in the census — Premiere XML, FCPXML, Resolve XML, **Avid Media Composer/EDL**, ProRes+captions, SRT, **Word-doc paper edits**; XML relinks to source footage (OBSERVED help docs).
- **Depth:** multicam ingest→sync→active-speaker switch→multicam sequence export; rough-cut engine proposing narrative structure; prompt-driven string-outs; Scripted Mode (2026). "ChatGPT for video editing" positioning.
- **Traction:** bootstrapped, ~$550K ARR 2025 (REPORTED, GetLatka).
- **Complaints:** B-roll placement imperfect; "not a one-click final edit"; falls apart on vérité/observational footage — "better at surfacing moments than assembling them meaningfully" (quotes C36–C37).

## 1D. Shorts clippers & caption SaaS (set caption/clip expectations)

Condensed; all cloud; none has a native Premiere panel. Full pricing in agent data, key facts here.

| Product | Price anchor (2026-06-10) | Premiere path | Signature depth | Traction/momentum | Sharpest complaint |
|---|---|---|---|---|---|
| **Opus Clip** | Free 60 cr/mo (watermark, clips expire 3 days); Starter $15; Pro $29/mo (~300 cr/mo); credits ≈ 1/min | **XML + SRT export, Adobe Video Partner** — best handoff in cohort | ClipAnything multimodal clipping, virality score, AI B-roll, 3.0 (Feb 2026): 15-min mid-form, 4K, Agent Opus beta | $20M SoftBank @ $215M val (Mar 2025); G2 4.7/127; Trustpilot 4.0/~302 **with 22% 1-star** | "deducted 570 credits for a project that glitched and ruined the video" (C16); removed daily credits Jun 2026 (C17) |
| **Vizard** | Free 60 cr; Creator ~$19–29/mo (UNVERIFIED exact) | None | Speaker reframe, timeline editor, scheduler | G2 4.7/340, Capterra 4.9/432 | credit forfeiture on cancel; reframing misses off-face entries |
| **Klap** | Starter $14/mo annual; Pro $39; Pro+ $94 (no free tier) | None | Dubbing 29 langs (Pro+) | bootstrapped ~$440K ARR | support/refund black hole; "5+ minutes to export a one minute short" |
| **Submagic** | Free 3 videos; Starter $19/mo ($12 ann.); Pro $39 ($23); Business $69 ($41); **Magic Clips +$19/mo add-on** | None (SRT only) | **The caption depth bar:** dozens of template families (Trending/Emoji/Premium/Speakers) + 10 custom, per-word karaoke, auto-emoji + GIFs, **sound-effect insertion**, auto-zooms/transitions, AI B-roll, hooks | **$8M ARR bootstrapped, 13 people** (Apr 2025); Trustpilot ~809 | export quality degradation ("rasterized layer… distorts the color grade"), terms changed mid-contract, no batch export (C21–C24) |
| **Captions.ai → Mirage** | Pro $9.99/mo; Max $24.99 (500 cr); Scale $69.99+ | None | AI Edit, Mirage Studio generative avatars, eye-contact, lipdub 29+ langs | $60M Series C @ $500M (Adobe Ventures!); **$75M growth Mar 2026**, >$175M total | "you pay for a subscription but are still limited by credits… charged twice" (C25) |
| **Quso (ex-vidyo)** | Free 75 cr; Lite $24/mo; Growth $40/mo | None | clip→schedule suite pivot | 4M+ users claimed; G2 4.8 | bugs forcing redos; aggressive marketing |
| **Munch** | **pivoted/exited clipping** → munchstudio.com done-for-you $38–60/mo | None | — | category-churn datapoint | — |
| **Veed** | Lite $19/mo ($12 ann.); Pro $49 ($24); credits don't roll over | None (SRT) | full editor, 125-lang subtitles, dubbing, eye-contact, Fabric 1.0 gen-video + Sora 2/Veo 3.1/Kling 3.0 integrations | G2 4.6/1,755; Trustpilot ~3,528 | cancellation friction, hidden credit costs |
| **Kapwing** | Pro $16/mo annual; Business $50 (voice clone, lip sync) | None (SRT) | collaborative browser editor | G2 4.1/40; Trustpilot mixed ~1,323 | refund refusal ("£191… refunded $23") |
| **CapCut / Pro** | Standard ~$9.99/mo; **Pro $19.99/mo / $179.99/yr** (annual jumped ~$77→$179.99, >100%, renewal cliff Feb 20, 2026 — Newsweek covered it); 1,200 AI credits/mo | None (closed ecosystem) | full editor, speaker-ID captions, templates, avatars | hundreds of millions MAU; **Trustpilot 1.2/5**; ToS grants ByteDance "perpetual, worldwide, irrevocable license" incl. biometric features (Jun 2025 terms) | stealth price doubling; "Pro export" wall; bot-only support |

**Category read (INFERRED):** pure clipping is commoditizing (Munch exited, Quso pivoted, Captions pivoted to generative ads) while caption-styling (Submagic, bootstrapped $8M ARR) is the healthiest single-feature business. Watermark removal is the #1 upgrade trigger across all free tiers.

## 1E. Competing NLE AI suites (expectation-setters)

### DaVinci Resolve 20/21 (Blackmagic)
- **Price:** free version; **Studio $295 one-time**, major versions free for owners (OBSERVED/REPORTED).
- **Locus:** **Neural Engine is 100% local, unmetered** ("All image processing… is done locally"); ~4–8 GB VRAM for most features.
- **Suite (Resolve 20, Apr 2025):** AI **IntelliScript** (builds timeline from a provided script — best takes on track 1, alternates above), **Multicam SmartSwitch** (active speaker via **audio + lip movement** — vision-assisted, unlike AutoPod), IntelliCut (silence removal + dialogue checkerboarding), Animated Subtitles, Magic Mask v2, Dialogue Matcher, Audio Assistant, Music Editor (retime music to length), Detect Music Beats, SuperScale, VoiceConvert (user-trainable), Set Extender. Studio-gated.
- **Resolve 21 (stable Jun 4, 2026):** AI Blemish Removal, Face Age Transformer, Face Reshaper, CineFocus, Motion Deblur, UltraSharpen, **IntelliSearch (find clips by objects/people)**, SlateID (reads slates), **Speech Generator (script→VO + custom voice training)** (OBSERVED, cgchannel).
- **Complaints:** Magic Mask "Completely Broken in Resolve 20" thread (freezes/crashes, revert-to-v19 workaround); Voice Isolation "flawed and glitchy"; AI features tank timeline performance (REPORTED).

### Final Cut Pro 11→12 (Apple)
- **Price:** $299.99 one-time; **Apple Creator Studio sub Jan 2026: $12.99/mo / $129/yr** bundling FCP+Logic+Motion+Compressor+Pixelmator (REPORTED). Feature-parity anxiety for one-time owners is live (quote in census data).
- **FCP 12 (Jan 28, 2026):** **Transcript Search** (NL search of spoken words), **Visual Search** (objects/actions by natural language), **Beat Detection** for music cutting, improved Magnetic Mask — all on-device Apple Silicon (REPORTED).
- **Complaint:** transcription English-only at launch.

### Filmora (Wondershare) — the cautionary tale
- **Price:** Basic $49.99/yr (0 AI credits!); Advanced $59.99/yr (1,000 credits); credit packs $9.99/300. AI background removal on a 10-min video ≈ 200–300 credits; active users exhaust the annual allowance in 4–8 weeks; realistic annual cost $270–340 (OBSERVED/REPORTED).
- **Complaints:** "The AI generation nearly always fails; they say the credits will be refunded, but never are" (C27); credits charged on failed jobs (C28); Trustpilot trial-billing trap reports.
- **Lesson (INFERRED):** metered AI inside a paid editor generates the loudest sustained anger in the entire market. OpenCut's "unlimited" is the antithesis — say so.

## 1F. Open-source / local
- **auto-editor** (OpenCut's upstream): 4.4k★, **Unlicense/public domain — SETTLED by LICENSE-file fetch** ("This is free and unencumbered software released into the public domain"), **very active: v30.5.0 Jun 9, 2026, now 99.7% Nim** (rewritten from Python — watch integration compatibility). Premiere via FCP7 XML export only, no panel — OpenCut's in-panel write-back is a concrete depth advantage over its own upstream (OBSERVED).
- **jumpcutter** (carykh): 3.2k★, MIT, **abandoned** ("I'm not going to be actively updating").
- **whisperX:** 22.4k★, BSD-2-Clause (relicensed 2024), v3.8.6 May 25, 2026. ~70× realtime batched large-v2 (A40-benchmarked; consumer 3080/3090-class REPORTED 60–70×), <8 GB VRAM. Diarization via pyannote community-1 (CC-BY-4.0, offline after one gated HF download).
- **Kdenlive STT:** GPL-3.0; VOSK + Whisper engines, fully local; setup friction is the recurring pain.
- **FunClip** (Alibaba): 5.8k★ MIT, active May 2026 — local ASR text-select-to-clip. **MoneyPrinterTurbo:** 85.1k★ MIT, v1.3.0 Jun 10, 2026 — faceless-shorts demand signal. **ClipsAI:** 501★ MIT (WhisperX+pyannote reframe pipeline). **ShortGPT:** 7.4k★ MIT, slowing, cloud-dependent. **AI-Youtube-Shorts-Generator:** 3.8k★ MIT but highlight-ranking always needs a cloud LLM.

---

# Part 2 — Feature matrix

✅ full · 🟡 partial · ❌ none. Columns: **OC**=OpenCut v1.32.0, **AD**=Adobe first-party, **AP**=AutoPod, **FC**=FireCut, **AC**=AutoCut, **GL**=Gling, **TB**=Timebolt, **DS**=Descript, **ED**=Eddie AI, **OP**=Opus Clip, **SM**=Submagic, **CA**=Captions.ai, **DR**=Resolve Studio.

| Capability | OC | AD | AP | FC | AC | GL | TB | DS | ED | OP | SM | CA | DR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Silence/jump-cut automation | ✅ | 🟡 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 | ❌ | 🟡 | 🟡 | ✅ |
| Filler-word removal | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | 🟡 | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| Repeated/bad-take detection | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ❌ | 🟡 | ❌ | 🟡 |
| Cut-review UI before apply | ✅ | 🟡 | ❌ | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | 🟡 |
| Transcription | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | 🟡 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Animated caption templates (karaoke/emoji/SFX) | ❌ | 🟡 | ❌ | ✅ | ✅ | 🟡 | ❌ | 🟡 | ❌ | ✅ | ✅ | ✅ | ✅ |
| Caption translation | ❌ | ✅☁ | ❌ | ✅☁ | ❌ | ❌ | ❌ | ✅☁ | ❌ | ✅☁ | ✅☁ | ✅☁ | ❌ |
| Text-based editing UX | 🟡 | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | 🟡 | ❌ | ✅ |
| LLM chapters/highlights | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| Multicam auto-switch | ✅ | ❌ | ✅ | ✅ | ✅ | 🟡 | ❌ | 🟡 | ✅ | ❌ | ❌ | ❌ | ✅ |
| Script-based assembly (IntelliScript-style) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 🟡 | ❌ | ❌ | ❌ | ✅ |
| Audio cleanup (denoise/speech enhance) | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ | 🟡 | ❌ | ✅ |
| Loudness normalize / batch match | ✅ | 🟡 | ❌ | ❌ | ❌ | ❌ | ❌ | 🟡 | ❌ | ❌ | ❌ | ❌ | ✅ |
| Stem separation | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 🟡 |
| Beat detection → markers | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Beat-synced auto-cut assembly | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 🟡 |
| TTS / voice clone | 🟡☁ | ❌ | ❌ | ✅☁ | ❌ | ❌ | ❌ | ✅☁ | ❌ | ❌ | ❌ | ✅☁ | ✅ |
| Dubbing / lip sync | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 🟡☁ | ❌ | ❌ | ❌ | ✅☁ | ❌ |
| Auto reframe 9:16 + shorts pipeline | ✅ | 🟡 | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | 🟡 | ✅ | ✅ | ✅ | 🟡 |
| Virality scoring / clip ranking | ❌ | ❌ | ❌ | 🟡 | 🟡 | ❌ | ❌ | ❌ | 🟡 | ✅ | 🟡 | ❌ | ❌ |
| AI B-roll (stock insert) | ❌ | ❌ | ❌ | ✅☁ | ✅☁ | ✅☁ | ❌ | 🟡 | ✅ | ✅☁ | ✅☁ | ✅☁ | ❌ |
| B-roll matching from OWN library | ❌ | 🟡 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Footage search — transcript | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ |
| Footage search — visual/semantic | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ |
| Color match / AI LUT gen | ✅ | 🟡 | ❌ | ❌ | ❌ | ❌ | ❌ | 🟡 | ❌ | ❌ | ❌ | ❌ | ✅ |
| Background removal / masking | ✅ | ✅ | ❌ | ❌ | 🟡 | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ |
| Upscale / frame interpolation | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| NL command interface / agent | ✅ | ❌* | ❌ | ❌ | ❌ | ❌ | ❌ | ✅☁ | ✅☁ | 🟡 | ❌ | 🟡 | ❌ |
| Premiere timeline write-back | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 🟡 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Cross-NLE export (FCP/Resolve/Avid) | 🟡 | ❌ | ❌ | ❌ | ❌ | 🟡 | ✅ | 🟡 | ✅ | ❌ | ❌ | ❌ | — |
| REST API / MCP / plugin SDK | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | 🟡 | 🟡 | ❌ | 🟡 |
| Fully local AI processing | 🟡† | 🟡 | ✅ | ❌ | ❌ | ❌ | 🟡 | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Price for full AI suite | **$0** | $22.99/mo CC + credits | $29/mo | $34/mo | $19.8/mo | $40/mo | $17/mo+$0.03/min | $35/mo+credits | $167/mo | $29/mo | $39/mo | $24.99/mo | $295 once |

Row notes: ☁ = cloud-processed. *Adobe's Firefly AI Assistant is announced/coming-soon, not shipped (refuted as a current sherlock 1-2). †OpenCut is local except **Edge-TTS (Microsoft cloud)** — see P0-5. Adobe silence-cutting = pause-deletion in text-based editing (no motion/audio auto-edit). Resolve's multicam SmartSwitch uses audio **+ lip movement** — the only vision-assisted switcher shipping. Descript/Eddie "review UI" = proposal-style edits. Submagic caption row is the depth ceiling: karaoke + emoji + GIFs + SFX + auto-zoom + hooks. OpenCut text-based editing = transcript editing + cut application, not Descript-grade bidirectional doc-editing (hence 🟡).

**Honest OpenCut grade:** broadest feature surface in the matrix (only Resolve Studio rivals it), unique in API/MCP + write-back + $0, but ❌/🟡 on five monetized capabilities: animated caption templates, translation, visual search, B-roll automation, virality ranking — and one tagline-breaking cloud dependency (Edge-TTS).

---

# Part 3 — Gap analysis

## 3A. Missing features (OpenCut lacks entirely)

| # | Gap | Who monetizes it | Their price | Demand evidence | Local feasibility (→Part 5) |
|---|---|---|---|---|---|
| A1 | **Visual/semantic footage search + face/OCR tagging** (cross-library, not just transcripts) | Adobe Media Intelligence (CC sub), Resolve 21 IntelliSearch (Studio $295), FCP 12 Visual Search, Eddie | bundled in $23–295 | "by far the biggest improvement would be with OCR and facial tagging" (C53); "scrubbing 8 seasons of footage… and I hate it" (C54); 30-yr editor built his own on-prem engine because pro options were "enterprise-level expensive… or opaque regarding data privacy" (C48) | ✅ SigLIP (Apache-2.0, ~1.8 GB) / OpenCLIP (MIT); face-rec model licensing needs vetting (InsightFace weights non-commercial — UNVERIFIED alternative required) |
| A2 | **Offline caption translation (batch)** | Adobe (27 langs, **cloud**), FireCut, Descript, Veed — all cloud | bundled / metered | "Where I've found a gap is batch transcription + translation… the NLEs don't really handle that" (C55) | ✅ OPUS-MT (CC-BY-4.0) / M2M-100 (MIT), <1 GB, CPU-viable. **NLLB-200 is CC-BY-NC — do NOT use** |
| A3 | **Paper edit: transcript text-selects → assembled Premiere sequence** | Simon Says Assemble ($15–125/mo cloud), Eddie ($15/export), Transcriptive (dead) | $15+/mo or per-export | The strongest white-space signal: editors hand-roll WhisperX+Gemini and name the round-trip as the missing piece — "Works well aside from having to find it all back in premiere after laying it out in text" (C57); semantic interview-selects confirmed as the proven value of AI assembly (C56) | ✅ pure orchestration of existing OpenCut pieces (whisper + LLM + write-back). No new model |
| A4 | **Animated caption template system** (per-word karaoke, emoji, SFX triggers, template families) | Submagic ($19–69/mo), Captions, Opus, FireCut, Resolve Animated Subtitles | $12–41/mo | Submagic = $8M ARR on caption styling alone; category's healthiest business | ✅ no model needed — MOGRT/rendering engineering |
| A5 | **AI B-roll auto-placement from user's own library** | Eddie (logs + places B-roll over A-roll spine); Opus/Submagic/FireCut only do *stock* B-roll | $15/export – $39/mo | "No one wants to log broll" (C52); Broll Selector + Eddie both fail at shot-quality judgment (C51) | ✅ SigLIP/OpenCLIP embeddings + transcript alignment |
| A6 | **Beat-synced auto-cut assembly** (cut/montage to music, not just markers) | BeatEdit ($99.99) + manual Automate-to-Sequence; Resolve Detect Beats | $99.99 once | "any built-in way to analyse music and automatically place markers on beats, so cuts can be snapped?" (C61); "$100?" pushback (C42) | ✅ librosa (ISC) / BeatNet (CC-BY-4.0). **madmom models CC-BY-NC-SA — avoid** |
| A7 | **Voice cloning** (typed corrections in speaker's voice; custom VO) | Descript Overdub, Resolve 21 Speech Generator (custom voice training, local!), Captions | bundled $24–35/mo / $295 | Descript's flagship retention feature; Resolve just made local voice training table-stakes | 🟡 OpenF5-TTS (Apache-2.0 retrain) — quality below original F5-TTS; official F5-TTS weights CC-BY-NC (user-fetched only) |
| A8 | **Lip-synced dubbing** | Captions/Mirage lipdub 29+ langs (cloud) | $24.99+/mo | dubbing demand rising w/ Klap/Veed/Kapwing all shipping it | 🟡 MuseTalk (code MIT, weights commercial-OK but OpenRAIL-class → separate download); LatentSync 1.6 needs 18 GB VRAM |
| A9 | **Virality scoring / clip ranking** for the Shorts pipeline | Opus, Vizard, Quso | $15–40/mo | Opus's core pitch; but also its top complaint ("cuts random and irrelevant clips") — opportunity is *explainable* ranking | ✅ local LLM scoring rubric (Ollama) — engineering, not new models |
| A10 | **Script-based assembly** (match takes to a provided script) | Resolve 20 IntelliScript (Studio) | $295 | scripted-content creators; Timebolt added retake-for-scripted Nov 2025 | ✅ alignment of whisper output against script text — no new model |
| A11 | **Silent-track/dead-channel cleanup** (multicam, no re-cutting) | nobody | — | direct unserved ask (C59) | ✅ trivial DSP |
| A12 | **Eye-contact correction** | Descript, Captions, Veed (all cloud/proprietary — NVIDIA Maxine lineage) | bundled | demand exists in talking-head segment | ❌ **DISQUALIFIED — no production-viable maintained OSS model** (only 2018-era research dumps; would be R&D, not integration) |
| A13 | **Generative extend / gap-fill video** | Adobe (GA, Firefly credits) | credits | — | ❌ **DISQUALIFIED — no comparably mature MIT-shippable local video-generation model** |

## 3B. Depth deltas (OpenCut has it; competitor does it meaningfully better)

1. **Multicam switching — Resolve SmartSwitch & AutoPod vs OpenCut diarization.** Resolve uses audio **+ lip movement** (vision); AutoPod offers per-mic dB thresholds, shot-frequency presets, and one-person/two-person/wide shot types. But the evidence says even those are distrusted: wide-shot overuse with only 3 coarse settings (C2), wrong-camera favoritism (C4, C5), forced J/L-cut style (C5). **The quality bar nobody clears: tunable switch frequency, jump-cut avoidance windows, look-at-speaker weighting, and a reviewable cut list before commit.** OpenCut already has the review-UI pattern from Cut & Clean — extending it to multicam is the leapfrog.
2. **Bad-take detection — Gling.** Gling clusters repeated *sentences* across takes and auto-picks the best delivery, with published accuracy figures (96.7% content-safety / 85.4% F1). What earns trust: conservative defaults (content preservation over aggressiveness), Aggressive/Natural toggle, intentional-pause detection, custom stoplists. OpenCut's transcript-similarity detection should publish the same metrics and adopt sensitivity presets + stoplists.
3. **Captions — Submagic vs static burn-in.** The delta is specific: per-word karaoke highlight timing, auto-emoji/GIF insertion keyed to keywords, **sound-effect triggers**, auto-zoom/transition pulses synchronized to caption beats, template *families* with brand kits, and hook-title generation. None requires cloud AI — it's a rendering/template system. OpenCut's burn-in is the floor of this market, not the middle.
4. **Text-based editing — Descript vs a transcript panel.** Descript's UX wins because the document *is* the timeline: word-level deletes propagate instantly and bidirectionally, no apply step; Overdub typing fixes flubs without re-recording; agent edits arrive as reviewable proposals. A transcript panel that batches edits then applies is a fundamentally laggier loop. Counter-evidence worth exploiting: Underlord's NL prompting is *disliked* when it replaces deterministic controls ("Pressing an on/off toggle… is much easier than typing instructions that don't work" — C12). OpenCut's NL interface should stay alongside, never instead of, deterministic buttons.
5. **Search — Adobe Media Intelligence vs FTS5.** Adobe searches visuals + audio descriptions + similar-shots, locally. OpenCut's transcript-only FTS5 is shallower *within a project* — but Adobe only indexes the open project, ships no API, and users report runaway indexing that "COMPLETELY bogs down my system" with the off-switch ignored (C30). OpenCut wins on *scope* (whole library, cross-project), control, and scriptability — once A1 adds the visual modality.
6. **Transcription quality — already an OpenCut win.** "I transcript a weekly show with WhisperX (which is miles better than the transcription in Premiere)" (C58). Publish a benchmark and say it out loud.
7. **Cross-NLE export — Eddie/Timebolt.** Eddie exports Premiere XML, FCPXML, Resolve XML, **Avid Media Composer/EDL**, ProRes+captions, Word paper-edits; Timebolt adds Camtasia. OpenCut has FCP XML; Resolve-flavored XML and EDL/Avid would complete the story auto-editor's export list sketches.

## 3C. White space (nobody serves well) — evidence-ranked

1. **Transcript paper-edit → real Premiere timeline round-trip** (C56, C57, C58) — editors hand-roll it; Simon Says is cloud + metered; Transcriptive is dead.
2. **"Find the best take/shots" visual quality judgment** — both named tools fail (C51: "Broll Selector never worked… Eddie didn't identify the strongest takes").
3. **Local cross-library AI indexing w/ face + OCR tagging, manual-tag respect, NLE round-trip** (C48, C53, C54, C63).
4. **Local transcription + diarization + word-scrub GUI** for journalism/NDA work — users enumerate exactly these two gaps vs Otter/Trint (C44–C47).
5. **Trustable, tunable multicam auto-switching** (C1, C2, C4, C5).
6. **One-time-price / non-subscription AI editing** (C3, C38, C40, C42) — OpenCut's $0 over-serves this; the gap is *awareness*.
7. **Reviewable/undoable AI cut passes** (C59, C60, C11) — OpenCut largely has this; market it as the differentiator vs "twice the work" agent edits.
8. **Batch local translation/dubbing/subtitling integrated with the NLE** (C55).
9. **Auto multicam/two-camera color matching that works** (C62 — built-in Color Match failing is treated as normal; advice defaults to scopes or paid CineMatch). OpenCut's color-match-to-reference extended to batch multicam grouping.
10. **Beat-synced cutting affordably** (C61, C42).
11. **Silent-channel/dead-track cleanup without re-cutting** (C59).
12. **Local PII/profanity redaction at org scale** — corporate DLP can't police "paste into textbox" AI usage (C49); Adobe shipped cloud-adjacent censorship but the vendor-vetting burden remains (see 3D).

## 3D. The local/free wedge — where it demonstrably converts

- **Legal (documented institutional duty, not vibes):** ABA-published GPSolo eReport (Sept 2025): cloud AI transcription "gains access to otherwise confidential attorney-client communications… introducing an outsider into a privileged session risks inadvertent waiver"; Model Rule 1.6 imposes "reasonable efforts" duties; **North Carolina State Bar 2024 Formal Ethics Opinion 1 (adopted Nov 1, 2024)** requires vetting AI vendors' data security before use. The article recommends deploying AI "locally or within secure firm infrastructure." (OBSERVED; cite the article's recommendation, NOT "the ABA endorses" — that framing was refuted in verification.) Corroborated by Duane Morris client advisory on AI transcription privilege pitfalls (Feb 2026).
- **Journalism (verbatim refusals):** "If the source is sensitive enough that I'm not comfortable putting it in the cloud, I'm transcribing by hand" (C44); investigative journalists already standardize on local Whisper tools and name the UX gaps — diarization + word-scrub GUI (C45, C46); child-welfare reporters exclude Otter outright (C47).
- **Competitors monetize this wedge — proof it pays:** Simon Says sells an **air-gapped on-prem tier**; Timebolt launched **Vault** (Jan 2026): on-prem, "HIPAA/FINRA/FedRAMP, zero cloud dependency." OpenCut gives that away.
- **Healthcare:** HHS guidance prohibits media/film crew access to PHI without authorization (OBSERVED, hhs.gov PDF) — any cloud upload of clinical footage is a compliance event; this burden is structural.
- **Per-minute/credit pain (high-volume users):** silence-remover author: "Looked at AutoCut and TimeBolt, didn't want to pay a monthly fee" (C38); Descript canceler: "price almost doubled with no notice… I won't be tricked into paying double" (C39); Opus credit-burn rage (C16–C19); Filmora failed-job credit charges (C27, C28); subscription-sprawl purges (C40, C41).
- **Low-bandwidth/field production:** no fresh verbatim evidence found this pass — honest gap; the locality argument stands on inference only here (INFERRED).

---

# Part 4 — Adobe sherlock threat analysis

Graded against shipped Premiere (Beta + stable), Adobe MAX/NAB 2025–26 announcements, and the Firefly roadmap. Adobe's cadence: quarterly stable releases; Color Mode GA expected later 2026; Firefly Quick Cut may migrate into Premiere (open question — would upgrade cut-automation risk).

| Opportunity | Grade | Reasoning |
|---|---|---|
| Visual semantic search (A1) | **ALREADY SHERLOCKED** (in-project, local) — but **SAFE as cross-library + face/OCR + API** | Media Intelligence is local and good, but project-scoped, no API, no face tagging (C53), and indexing misbehaves (C30). Differentiate on scope + control, not existence |
| Offline caption translation (A2) | **SAFE** | Adobe's 27-language translation routes through cloud third-party services (OBSERVED); a fully-offline pipeline serves exactly the vertical Adobe's architecture can't |
| Paper edit → timeline (A3) | **AT RISK (12–24 mo)** | Firefly editor has text-based editing + Quick Cut, but cloud-only, credit-gated, no Premiere write-back; Premiere's own text-based editing does deletion, not select-assembly. The local + in-Premiere + LLM-selects combination is unclaimed |
| Animated caption templates (A4) | **AT RISK** | Premiere has caption styling and the ecosystem (FireCut/Submagic) is crowded; differentiation = free + local + Premiere-native captions track output. Lower durability, high demand |
| B-roll from own library (A5) | **SAFE-ish** | Media Intelligence *finds* clips; nobody auto-*places* from the user's own library locally. Adobe's incentive points at Firefly stock/gen B-roll (credits), not your archive |
| Beat-synced assembly (A6) | **SAFE** | No Adobe signal; only BeatEdit ($99.99) and Resolve serve it |
| Voice clone / local TTS (A7) | **AT RISK from Resolve, not Adobe** | Resolve 21 shipped local Speech Generator w/ custom voice training; Adobe TTS would be cloud/credit. Local free clone remains differentiated |
| Lip-sync dubbing (A8) | **SAFE (12 mo)** | Adobe dubbing exists in Firefly/Express territory, cloud + credits; local lip-sync is heavy but unclaimed |
| Virality ranking (A9) | **SAFE** | Adobe has nothing; Firefly clipping would be cloud. Opus's quality complaints are the bar |
| Script-based assembly (A10) | **AT RISK** | Resolve shipped IntelliScript; Adobe matching it in Premiere is plausible; still absent today |
| Multicam depth (3B-1) | **SAFE** | Adobe has no multicam auto-switching at all (and stable multicam has open UX bugs — C33, C34); Resolve's SmartSwitch sets the bar, AutoPod the price |
| PII/profanity redaction (3C-12) | **ALREADY SHERLOCKED (basic)** — wedge survives | v25.6 bulk bleep/mute ships; but the legal vertical's duty is *vendor-vetting* — "auditable, open-source, fully local" beats "Adobe cloud account" for Rule 1.6 diligence |
| Background removal / masking (existing rembg) | **ALREADY SHERLOCKED** | AI Object Masking v26.0 + 26.2 modes. Maintain, don't invest |
| Generative extend | **ALREADY SHERLOCKED + locally infeasible** | Disqualified both ways |
| **Platform: CEP panel survival** | **THE structural threat** | CEP 12 = final major; ExtendScript "through September 2026"; UXP GA'd in 25.6 w/ marketplace open; Transcriptive cited exactly this in its death note. Not a feature risk — an existence risk |

---

# Part 5 — Local feasibility mapping

Existing toolbox (faster-whisper, Demucs, Real-ESRGAN, RIFE, MediaPipe, TransNetV2, rembg, MusicGen, Resemble Enhance, Ollama) extended. Weights licensing verified by LICENSE/model-card fetch 2026-06-10 (Part 4 of agent data).

| Feature | Model/technique | Code license | **Weights** license | MIT-shippable? | VRAM | Maturity |
|---|---|---|---|---|---|---|
| Transcription+align+diarization (deepen) | **WhisperX** + pyannote community-1 | BSD-2-Clause | model CC-BY-4.0 (gated HF download, runs offline after) | ✅ | <8 GB (large-v2) | v3.8.6 May 2026, 22.4k★ |
| Offline translation (A2) | **OPUS-MT/MarianMT**; **M2M-100** | — | CC-BY-4.0 (spot-check per language pair); MIT | ✅ / ✅ | <1 GB, CPU-viable | mature |
| | ~~NLLB-200~~ | — | **CC-BY-NC-4.0** | ❌ **do not use** | — | — |
| Visual search / B-roll matching (A1, A5) | **SigLIP** (so400m); **OpenCLIP** | Apache-2.0 / MIT | Apache-2.0 / per-checkpoint (LAION OK, verify each) | ✅ | ~1–2 GB | stable / v3.3.0 Feb 2026 |
| Face tagging (A1) | candidate models need vetting — InsightFace pretrained weights are research-only | — | — | **UNVERIFIED — vet before committing** | — | — |
| Voice clone (A7) | **OpenF5-TTS-Base** (Apache-2.0 retrain on CC-BY Emilia-YODAS) | MIT (F5-TTS framework) | **Apache-2.0** | ✅ (quality below original — REPORTED) | ~6–8 GB | WIP-labeled |
| | F5-TTS official ckpts | MIT | **CC-BY-NC** (Emilia data) | ❌ bundle / 🟡 user-fetched w/ disclosure | ~6 GB | v1.1.20 Apr 2026, 14.7k★ |
| | ~~XTTS-v2~~ | MPL-2.0 | **CPML non-commercial; Coqui dead — no commercial path exists** | ❌ | ~6 GB | orphaned |
| Lip sync (A8) | **MuseTalk 1.5** | MIT | README: "available for any purpose, even commercially"; HF tag CreativeML OpenRAIL-M → commercial-OK **with use-restriction rider; not pure MIT** | 🟡 ship weights as separate download w/ their notice; keep app MIT | 4 GB min (slow); 30fps+ inference-loop on V100 (preprocessing excluded — end-to-end is slower; fine for offline) | v1.5 Mar 2025, 6k★, active |
| | LatentSync 1.6 | Apache-2.0 | OpenRAIL++ | 🟡 same pattern | **8 GB (v1.5) / 18 GB (v1.6)** | active, 5.8k★ |
| Beat-aware cutting (A6) | **librosa** (DSP) + **BeatNet** | ISC / CC-BY-4.0 | n/a / CC-BY-4.0 | ✅ | CPU | librosa 0.11.0 Mar 2025 |
| | ~~madmom~~ | BSD code | **models CC-BY-NC-SA** | ❌ models | — | stale (2018) |
| Eye contact (A12) | none viable (Maxine proprietary; OSS = 2018-era research dumps) | — | — | ❌ **DISQUALIFIED** | — | — |
| Generative extend (A13) | no mature local video-gen at Premiere quality | — | — | ❌ **DISQUALIFIED** | — | — |
| Paper edit, script assembly, virality ranking, silent-track cleanup, caption templates (A3, A9–A11, A4) | existing stack (whisper + Ollama + FFmpeg + write-back) | — | — | ✅ no new models | — | — |
| **Local TTS to replace Edge-TTS (P0-5)** | OpenF5-TTS / Piper-class | Apache/MIT | Apache-2.0 (verify per voice) | ✅ | CPU–8 GB | active |

---

# Part 6 — Synthesis & roadmap

## 6.1 Executive summary — the 10 highest-leverage moves

1. **Migrate to UXP before it's forced (P0).** Transcriptive died citing CEP→UXP + Adobe-native features; CEP 12 is the final major; ExtendScript support runs "through September 2026"; UXP marketplace submissions are open as of Premiere 25.6. Every quarter on CEP-only is borrowed time. This is the only item on this list that can kill the product.
2. **Ship the paper edit (P0).** Select transcript passages (or let the local LLM select by theme: "find every moment about belonging"), assemble a real Premiere sequence. Strongest white-space evidence in the dataset; commercial analogs are cloud+metered (Simon Says) or $15/export (Eddie); the dead Transcriptive vacated the niche. OpenCut already owns every ingredient.
3. **Add visual search + make library search cross-project (P0).** SigLIP embeddings alongside FTS5; face/OCR tagging as fast-follow (after license vetting). Adobe sherlocked in-project local search — the durable position is *whole-archive*, API-scriptable, with manual-tag respect (C63's stated requirements).
4. **Leapfrog multicam trust (P0).** Lip-activity weighting (MediaPipe — already shipped for reframe), tunable wide-shot frequency (continuous, not 3 presets), jump-cut-avoidance window, and the existing cut-review UI extended to multicam. Every competitor complaint (C1–C5) is a spec line.
5. **Replace Edge-TTS with local TTS (P0, small).** It's the one cloud call in a "100% local" product. OpenF5-TTS/Piper-class. Do it before a reviewer does the traffic capture.
6. **Offline batch caption translation (P1).** OPUS-MT/M2M-100. Adobe's translation is cloud — this is the cleanest "local where Adobe isn't" feature, and it compounds the legal/journalism wedge.
7. **Animated caption template system (P1).** Karaoke per-word, emoji, SFX triggers, template families. Pure engineering, no models; Submagic's $8M ARR proves the demand; AT RISK vs Adobe but the free+local+native-caption-track combination holds.
8. **Beat-synced assembly + B-roll-from-own-library (P1).** librosa/BeatNet auto-cut closes the $99.99 BeatEdit gap (with its wav-only limitation as the spec to beat); SigLIP matching closes "no one wants to log broll."
9. **Publish the wedge collateral (P1, marketing).** Privacy one-pager citing NC State Bar 2024 FEO 1 + the ABA-published article (carefully attributed); cost calculator (AutoPod $348/yr + Descript $288 + Opus $174 + Submagic $228 ≈ $1,038–1,400/yr — substantiates the existing tagline); WhisperX-vs-Premiere transcription accuracy comparison (C58 says editors already believe it).
10. **Adobe Exchange + reviewer distribution (P1).** Marketplace listing (10-business-day review; companion apps explicitly allowed; tri-arch + macOS notarization required — Linux backend stays direct-download); aescripts (70/30); Premiere Gal and Javier Mercedes first.

**Market momentum read:** money is flowing to cloud clip tools (Opus $20M @ $215M SoftBank Mar 2025; Captions/Mirage $75M Mar 2026, >$175M total, Adobe Ventures on the cap table) while pure clipping commoditizes (Munch exited, Quso pivoted, Captions pivoted to generative ads) and the most-loved business is bootstrapped caption styling (Submagic $8M ARR). OpenAI shut down Sora (Mar–Apr 2026); Runway ($5.3B) now ships *inside* Firefly — Adobe integrates rather than gatekeeps external AI. Resolve and FCP both shipped local NL visual search within the last 6 months (Resolve 21 Jun 2026, FCP 12 Jan 2026) — local, unmetered AI is becoming the *NLE baseline*, which validates OpenCut's architecture and raises the bar for its depth. CC ≈ 41M subscribers, Premiere ≈ 30M users (REPORTED, moderate confidence): the TAM for a free Premiere extension is enormous and the incumbent plugins charge $20–35/mo into visible subscription fatigue.

## 6.2 Prioritized backlog

Scoring: Demand (strong/med/weak, from Part 3 evidence) · Durability vs Adobe (high/med/low, from Part 4) · Feasibility (✅ now / 🟡 caveats) · Effort (S/M/L/XL).

### P0
| Item | Demand | Durability | Feas. | Effort | Acceptance criterion |
|---|---|---|---|---|---|
| UXP panel migration (CEP kept during transition) | structural | — | ✅ | XL | Panel passes Adobe marketplace review on UXP with feature parity for cut-apply + captions + markers |
| Paper edit: transcript selects → sequence | strong (C56–58) | med | ✅ | M | Select N transcript ranges (manual or LLM-prompt) → new Premiere sequence with correctly trimmed, source-linked clips in chosen order, plus FCP XML/EDL fallback |
| Visual semantic search (SigLIP) + cross-project index | strong (C48, 53–54) | med (scope-differentiated) | ✅ | L | NL query over whole indexed library returns ranked clips w/ thumbnails; respects existing manual tags; exposed via REST + MCP |
| Multicam v2: lip-activity weighting + continuous shot-frequency tuning + review UI | strong (C1–C5) | high | ✅ | L | On a 3-cam shared-audio test shoot, active-speaker accuracy beats audio-only diarization measurably; user can adjust wide-shot % as a slider and preview/veto every switch before apply |
| Local TTS replacing Edge-TTS | med (tagline integrity) | high | ✅ | S | Zero network calls during TTS; "100% local" claim survives a packet capture |
| Silent-track/dead-channel cleanup | med (C59) | high | ✅ | S | Multitrack clip in → blank channels removed, no cuts introduced, sync preserved |

### P1
| Item | Demand | Durability | Feas. | Effort | Acceptance criterion |
|---|---|---|---|---|---|
| Offline batch caption translation (OPUS-MT/M2M-100) | med-strong (C55) | high (Adobe = cloud) | ✅ | M | Batch-translate SRT/caption tracks for N files × M languages fully offline; round-trips to native Premiere caption tracks |
| Animated caption templates (karaoke/emoji/SFX, families) | strong ($8M ARR comp) | low-med | ✅ | L | ≥6 template families incl. per-word karaoke + emoji + SFX trigger; output as burn-in AND native caption track + MOGRT |
| Beat-synced auto-assembly | med (C42, C61) | high | ✅ | M | Reads audio embedded in video (beats BeatEdit's wav-only limit); assembles selected clips cut-to-beat with subdivision control |
| B-roll auto-placement from own library | med-strong (C51–52) | med-high | ✅ | L | Given an A-roll transcript, proposes time-aligned B-roll from indexed library with confidence scores; review UI before insert |
| Bad-take detection v2: published accuracy + sensitivity presets + stoplists | med (Gling benchmark) | high | ✅ | M | Self-published content-preservation/F1 benchmark ≥ Gling's reported 96.7%/85.4% on a public test set; Aggressive/Natural presets |
| Privacy one-pager + cost calculator + transcription benchmark | med (sales collateral) | high | ✅ | S | Three artifacts live in README/site, each claim footnoted to a primary source |
| Adobe Exchange + aescripts listings; outreach to Premiere Gal / Javier Mercedes / comparison-video channels | — | — | ✅ | M | Listed on both marketplaces; ≥2 independent review videos live |
| Virality/clip ranking with explainable scores | med (Opus complaints) | high | ✅ | M | Shorts pipeline outputs ranked candidates with per-criterion rationale; user-editable rubric |
| Cross-NLE export completion (Resolve XML flavor, EDL/Avid) | med (Eddie parity) | high | ✅ | M | Cut lists round-trip into Resolve and Media Composer without relink errors |

### P2
| Item | Demand | Durability | Feas. | Effort | Acceptance criterion |
|---|---|---|---|---|---|
| Voice cloning (OpenF5-TTS; optional user-fetched F5-TTS weights w/ NC disclosure) | med | med (Resolve shipped local) | 🟡 quality | L | Clone from ≥3s reference; typed correction patches a flubbed word in-place |
| Lip-synced dubbing (MuseTalk; LatentSync for high-VRAM) | med | high (12 mo) | 🟡 weights rider + GPU | XL | Translated audio + lip-synced video preview on a 12 GB GPU; weights fetched separately under their own license notice |
| Script-based assembly (IntelliScript parity) | med | med | ✅ | L | Provided script → timeline with best takes on V1, alternates stacked above |
| Multicam batch color matching | med (C62) | med | ✅ | M | N cameras matched to reference within stated ΔE on a test shoot |
| Face/OCR tagging in library index | strong (C53) | med | 🟡 license vetting first | L | Search by named person/on-screen text locally; model weights verified redistributable |

**Explicitly disqualified:** eye-contact correction (no viable OSS model), generative extend (no local model + sherlocked). **Deprioritized:** anything masking/background-removal beyond maintenance (sherlocked by AI Object Masking, quarterly-improved).

## 6.3 Positioning & messaging

Attack lines, each backed by quotable evidence:

1. **"No credits. No upload. No expiry."** — vs Opus ("deducted 570 credits for a project that glitched"), Captions ("limited by credits… charged twice"), Filmora ("credits… never refunded"), Descript ("a month's worth of credits lasts about a day"). Every line links to the competitor's own Trustpilot/Reddit page.
2. **The cost calculator** — substantiate the existing "$1,400/yr" tagline with named SKUs: AutoPod $348/yr + Descript Creator $288/yr + Opus Pro $174/yr + Submagic Pro $276/yr + Filmora credits ~$270–340/yr. Date-stamp it; prices verified 2026-06-10.
3. **The privacy one-pager (legal/journalism/healthcare)** — NC State Bar 2024 Formal Ethics Opinion 1 (vendor-vetting duty) + ABA-published GPSolo analysis (privilege-waiver risk of cloud transcription; recommends local deployment — attribute to the article, not "the ABA") + HHS PHI guidance. Close with: competitors charge enterprise prices for this exact property (Simon Says air-gapped tier, Timebolt Vault "HIPAA/FINRA/FedRAMP, zero cloud dependency").
4. **"Better transcription than Premiere's — measured."** — editors already say it (C58); publish the WER benchmark.
5. **"Review every cut before it touches your timeline."** — vs Underlord ("cuts words off mid-word… twice the work") and Opus (no edit preview at all). OpenCut's review-UI is the trust feature the agentic wave is missing.
6. **"One-time price" beats us — we're $0.** — the AutoPod→Phantom churn quote (C3) shows the market actively shopping on pricing model; free + open-source ends that conversation.
7. **What NOT to claim:** don't say "Adobe's AI is cloud" generically (Media Intelligence and Object Masking are local — the claim would be falsified in one reviewer test). Don't claim "100% local" until Edge-TTS is replaced. Don't state "ABA endorses local AI."

## 6.4 Distribution

**Adobe Exchange / CC Marketplace (OBSERVED, developer.adobe.com):** submit via Developer Distribution portal; UXP = .ccx, CEP = signed .zxp; review targets **10 business days** across 10 dimensions; **companion/local-backend apps are explicitly permitted** with rules — must install cleanly, not cause "abnormal CPU or RAM usage," and hybrid plugins **must ship macOS arm64 + macOS x64 + Windows x64 (missing one = automatic rejection)**; macOS requires Developer ID signing + notarization. Paid listings via FastSpring, **developer keeps 90%** — moot for free OpenCut, which also sidesteps the EU DSA trader-info requirement friction. Practical notes: provide reviewer test setup for the Flask backend; no launching external apps from the panel (browser links excepted); Linux users remain direct-download (marketplace doesn't cover them).

**aescripts:** ~70/30 author split (REPORTED — their pages 403 automated fetch); they handle VAT/chargebacks/support-ticketing/marketing; Premiere is a first-class category; exclusivity terms not public — ask. Their free **ZXP/UXP Installer** is the de-facto standard for self-hosted distribution regardless of listing.

**Also:** Adobe Video Partner Program (FireCut, AutoPod, Cutback, Opus are members — apply); FxFactory (Mac-centric); Product Hunt (FireCut precedent).

**Top-10 reviewer/community targets (evidence-based):**
1. **Premiere Gal** (~568K) — dedicated FireCut review exists; the single best-fit channel.
2. **Javier Mercedes** (200K+) — AutoPod tutorials for both Premiere and Resolve; exact podcast-editor ICP.
3. **Kevin Stratvert** (~3.98M) — maintains an "AI video editing" tag incl. Premiere tutorials; mass reach.
4. **Justin Odisho** (~1M) — Premiere institution; pitch tutorial-integration, not review.
5. **Cinecom.net / Jordy Vandeput** (~2.9M, UNVERIFIED count) — sponsored-integration likely required.
6. **Think Media** (3.38M) — "edit faster" angle for beginner creators.
7. **Editing Empire** — Premiere/AE-exclusive; adjacent to "AI Premiere Plugin changes EVERYTHING" genre videos.
8. **Eleven Percent** — publishes AI-Premiere-plugin tutorials AND builds competing plugins (AutoEdit, B-Roll Selectr); part channel, part competitor — engage carefully.
9. **The comparison-video mid-tail** — an active genre through Dec 2025 ("AutoCut vs AutoPod," "FireCut Review 2026," "AI Podcast Editing Comparison"); resolve channel handles from the video URLs and pitch the matrix in Part 2 as their script.
10. **Trade press:** ProVideoCoalition (covers every Adobe AI beat), CineD (demonstrably covers Premiere AI plugins), Premiere Bro (active Plugins tag), Frame.io Insider, No Film School (UNVERIFIED coverage). Communities: r/podcasting (~150K; live Descript-churn threads), r/premiere, Adobe Video Discord (25,963 members, Adobe-endorsed). r/editors bans most self-promo — participate, don't post ads.

---

# Appendix — User-complaint corpus (48 verbatim quotes)

All REPORTED; verbatim (ellipses = trims); accessed 2026-06-10/11.

**AutoPod / multicam panels**
- C1 "I'm about to drop it, the edits it makes are often bad. I end up watching through and changing a lot anyway" — u/Wu-Tang_Killa_Bees, 2026-05-20 — reddit.com/r/premiere/comments/1ti8iq8
- C2 "it has 3 settings for the frequency of wide shots… Medium uses the wide like 65% of the time which is just ridiculous… I probably spend just as time fixing the edit as I would editing from scratch" — same thread, 2026-05-26
- C3 "switching to phantom editor since their multicam editor is just a one time price and not a subscription. I'm so done with subscriptions." — u/mistahfree, 2026-05-20, same thread
- C4 "it hasn't been able to determine which guest is speaking either, and heavily favors one even when the other is talking." — u/Sophska, 2026-05-26 — reddit.com/r/premiere/comments/1tof0bf
- C5 "it loves to randomly switch to a person that is just sitting silently… way too excited about wide angles… I've timed the time saved by these tools and found it barely saves any" — u/enewwave (AutoCut), 2026-05-21 — reddit.com/r/premiere/comments/1ti8iq8

**Gling**
- C6 "the pacing feature either makes my voice go too fast or doesn't get rid of the long silences… Plus my voice sounds off." — r/NewTubers OP, 2025-03-22 — reddit.com/r/NewTubers/comments/1jhcrdx
- C7 "the export loses framerate and get jittery and I kinda hate the quality of it now" — u/Habaneropapi, 2025-03-26, same thread
- C8 "I like the interface of Gling but its got major performance issues" — u/Habaneropapi, same thread
- C9 "The edits are so choppy, And It always ruins jokes, because it doesn't understand timing… they kept updating it and adding features. But It's basic functions (editing the silences) have suffered." — u/Accurate-Whereas5051, 2025-07-06, same thread

**Descript**
- C10 "since last pricing update some features are eating my credits so much" — r/Descript OP, 2026-03-21 — reddit.com/r/Descript/comments/1rzkbs4
- C11 "I've stopped using Underlord period… It just cuts words off mid-word… it was twice the work to use the AI tools that are supposed to cut the work in half." — u/r1ckey24, 2026-03-21, same thread
- C12 "Underlord has never worked well… Pressing an on/off toggle button and manual editing is much easier than typing instructions that don't work." — u/thefakekiwi, 2026-03-22, same thread
- C13 "don't you think applying it from next month is a bit severe? i.e. almost no notice." — r/podcasting OP, 2025-10-26 — reddit.com/r/podcasting/comments/1oghwan
- C14 "Descript is like renting a house from the owner who also happens to live there and constantly moves the furniture around" — u/FloresPodcastCo, 2025-10-26, same thread
- C15 "Descript is getting worse and worse… the app is becoming more bloated and painful to use, and key features like offline editing have been stripped away." — u/adamschoales, 2026-02-12 — reddit.com/r/editors/comments/1r2yi48
- C16–C19 (Opus Clip, Trustpilot, trustpilot.com/review/opus.pro): "deducted 570 credits for a project that glitched and ruined the video… deceptively stole an additional 180 credits" (2026-06-10); "BOT RUN TRASH PRODUCT… They recently (06/01/26) took the Daily credits which was THEIR ONLY EDGE" (2026-06-08); "the cuts began focusing on me instead of showing what I was talking about… annoying AI voices are sometimes dubbed into the video" (2026-06-08); "Its been 12 weeks and no one has responded… NO REFUNDS OF CREDITS" (2026-06-05)
- C20 "I switched from Opus to Klap. The difference is Klap actually tries to edit well, not just clip random high-energy words." — u/Radiiaant, 2026-05-20 — reddit.com/r/editors/comments/1r2yi48
- C21–C24 (Submagic, Trustpilot, trustpilot.com/review/submagic.co): "they compress each video with a rasterized layer… distorts the quality of the color grade… additional paywalls after subscribing" (2026-06-11); "The company later changed those terms unilaterally, without proper notice" (2026-06-07); "support is non-existent and the product is constantly crashing… cost me $800 and hours of my life downloading each video one by one" (2026-03-22); "Their API does not allow you to edit captions, or to move b-roll… They do not have a refund policy" (2026-03-12)
- C25–C26 (Captions.ai, Trustpilot, trustpilot.com/review/captions.ai): "They charge you for subscription then they give you credits then the ai glitches then the credits get over…" (2026-05-25); "Went to try free trial. Tried to charge me." (2026-05-27)
- C27 "The AI generation nearly always fails; they say the credits will be refunded, but never are." — r/Filmora OP, 2025-08-10 — reddit.com/r/Filmora/comments/1mm84tv
- C28 "tried text to speech feature twice, failed both times but it still takes my credits." — u/Tomatillo-Weak, 2025-08-19, same thread

**Adobe first-party**
- C29 "when using Enhance Speech… there are artifacts inserted during silence… distorts the very beginning of [dialogue]." — Adobe Community, 2024-11-02 — community.adobe.com/questions-729/...-1414372
- C30 "the new project just starts building an Adobe Media Intelligence Project Index File which COMPLETLEY bogs down my system and makes it crash… everything scanny or 'intelligent' is turned off" — u/i_sell_you_lies, 2025-12-11 — reddit.com/r/premiere/comments/1pjjlgb
- C31 "text-based editing hardly works… Premiere will not select it unless I attempt it 2 more times." — u/JeanGabinsChin, 2026-04-07 — reddit.com/r/premiere/comments/1sf1hkn
- C32 "as I make more edits to my timeline, responsiveness really slows… frustrating to the point of non-functional." — same user/thread
- C33 "the previews in the multicam view are incorrectly scaled and you just see a crop in the middle of the shot which effective renders it unusable." — u/smushkan, 2026-03-03 — reddit.com/r/premiere/comments/1rjxzrm
- C34 "Usually when I make a multicam sequence, the sound is just blank segments in my timeline" — u/Antique-Kitchen9027, same thread
- C35 (Resolve gating) "How hard is it to get a clean mask without the AI tool in the paid version?" + "Magic Mask is not silver bullet… not precise and stable over time" — reddit.com/r/davinciresolve/comments/1sqlds7, 2026-04-20
- C36 "the rough cut quality seems questionable for anything that isn't a clean talking head interview… completely out of its depth [for vérité]." — r/editors OP, 2026-05-08 — reddit.com/r/editors/comments/1t7f6gs
- C37 "better at surfacing moments than assembling them meaningfully… for anything with complex action, b-roll, or vérité coverage it falls apart quickly." — u/ravet007, same thread

**Fee pain**
- C38 "Looked at AutoCut and TimeBolt, didn't want to pay a monthly fee for it, so I just built it myself" — r/premiere OP, 2026-03-12 — reddit.com/r/premiere/comments/1rrbb2n
- C39 "The price for my plan has almost doubled with no notice… I won't be tricked into paying double for a service that I don't use." — u/Woman_Of_Words, 2025-10-27 — reddit.com/r/podcasting/comments/1oghwan
- C40 "my workflow turned into 15 browser tabs, random $15/mo charges i kept forgetting to cancel" — r/ContentCreators OP, 2026-05-28 — reddit.com/r/ContentCreators/comments/1tpsjg0
- C41 "canceled other subscriptions… reduced my costs by a good 30%." — u/suaveSavior, same thread
- C42 "What advantage does this have over just hitting M on beat? What makes this worth $100?" — u/VulGerrity, 2025-12-29 — reddit.com/r/premiere/comments/1pwtve4
- C43 "Try Trint." / "Agreed. Expensive but powerful." — r/editors, 2026-02-12 — reddit.com/r/editors/comments/1r2yi48

**Privacy/NDA**
- C44 "If the source is sensitive enough that I'm not comfortable putting it in the cloud, I'm transcribing by hand directly from my audio recorder" — u/Morpheus636_, 2026-04-10 — reddit.com/r/Journalism/comments/1shof6u
- C45 "i use macwhisper… nothing ever goes to the cloud. the scrubbing isn't as easy as a SaaS product like Trint" — u/eurydicey, same thread
- C46 "what it lacks vs Otter is 1) diarization and 2) a convenient GUI that'll let you scrub through the track based on the words" — u/Pop-X-, same thread
- C47 "Journalists who cover child welfare… don't want their interviews out there like otter makes available." — u/Friction_in_the_air, 2026-04-14, same thread
- C48 "The 'pro' options gave me sticker shock… enterprise-level expensive, required uploading footage to the cloud, or were opaque regarding data privacy. So, I decided to build the tool I actually needed" — u/ddnyc2021 (30-yr editor), 2026-01-20 — reddit.com/r/editors/comments/1qi2fnd
- C49 "Users are pasting sensitive data into browser-based tools… DLP doesn't fully cover modern AI usage patterns." — r/sysadmin OP, 2026-04-14 — reddit.com/r/sysadmin/comments/1sl5mde
- C50 = C15 (Descript offline editing stripped — cross-listed)

**White-space wishes**
- C51 "What I really need is visual/B-roll-based selection: identifying which shots are actually useful… Broll Selector – unfortunately it never worked properly… Eddie AI… didn't really remove the bad moments or identify the strongest takes." — 2026-06-09 — reddit.com/r/AIToolsAndTips/comments/1u1d7dd
- C52 "No one wants to log broll." — u/shamirallibhai (Eddie AI founder), 2026-05-09 — reddit.com/r/editors/comments/1t7f6gs
- C53 "by far the biggest improvement would be with OCR and facial tagging." — u/what_a_pickle, 2026-01-20 — reddit.com/r/editors/comments/1qi2fnd
- C54 "I've been scrubbing 8 seasons of footage, multiple times, for multiple projects, and I hate it." — u/Goat_Wizard_Doom_666, same thread
- C55 "Where I've found a gap is batch transcription + translation… the NLEs don't really handle that." — u/SeaworthinessAway519, 2026-05-09 — reddit.com/r/editors/comments/1t7f6gs
- C56 "If I want a subject to talk about the 'feeling of belonging', I can ask Eddie instead of keyword searching a transcript" — u/thekempest, 2026-05-08, same thread
- C57 "Works well aside from having to find it all back in premiere after laying it out in text" — u/jacintosalz, 2026-05-29, same thread
- C58 "I transcript a weekly show with WhisperX (which is miles better than the transcription in Premiere), then upload that transcript into Gemini… gives me the timecodes for sections" — u/ivanjimenezchg, 2026-05-08, same thread
- C59 "if I have a clip that have say 8 channels and 3 of them are completely blank I would love the ability to auto remove this only" — u/ralphdeonori, 2026-03-12 — reddit.com/r/premiere/comments/1rrbb2n
- C60 "can you by any chance make an easy way to undo if the pass was too aggressive?" — u/Audiophile-Heaven, 2026-05-05, same thread
- C61 "I'm wondering if Premiere has any built-in way to analyse music and automatically place markers on beats" — r/premiere OP, 2025-12-27 — reddit.com/r/premiere/comments/1pwtve4
- C62 "I've tried the color match wheel but the shots below just don't look alike/synced color-wise… do i need to find a colorist" — r/premiere OP, 2026-03-10 — reddit.com/r/premiere/comments/1rq5whb
- C63 "Will Blaze be able to read existing keywords/tags that are already embedded in my files…? Does Blaze support manual tagging alongside the AI indexing?" — u/regularivar, 2026-01-21 — reddit.com/r/editors/comments/1qi2fnd

**Known evidence gaps (honest):** FireCut third complaint, Excalibur/BeatEdit/Recut verbatim complaints, fresh Timebolt-specific UX threads, "air-gapped edit bay" verbatims, low-bandwidth field-production verbatims, Vizard funding, aescripts exclusivity terms, FastSpring fee treatment, Gling desktop-app upload behavior — all marked UNVERIFIED above rather than guessed.

**Refuted claims excluded from this report (failed adversarial verification):** "Firefly AI Assistant sherlocks NL commands" (coming-soon vaporware, 1-2); "MuseTalk weights are pure-MIT commercial-cleared" (1-2 — settled as OpenRAIL-class, commercial-OK with rider); "MuseTalk 4GB RTX 3050 Ti benchmark" (0-3); the original "auto-editor license refutation" was itself overturned by direct LICENSE-file fetch (Unlicense confirmed); "ABA endorses local-first AI" attribution overreach (1-2).
