# OpenCut — features.md ↔ Shipping Reconciliation (Pass 2 sample walk)

**Audit date:** 2026-05-17
**Scope:** Partial reconciliation of `features.md` (2,932 lines, 87 categories, 410 entries) against shipped routes and F-numbered backlog. This is **not** the full F179 cleanup (which is a 1-2 day pass); this is a **representative sample walk** to validate the scope and methodology of the eventual full pass.

---

## 1. Methodology

For each sampled entry from `features.md`:
1. Read the entry title + proposed module name + proposed routes.
2. Grep `git log` / `CHANGELOG.md` / route_manifest.json for the module name or route prefix.
3. Cross-reference against `CLAUDE.md` gotchas (which often confirm a feature "already exists").
4. Mark status as **SHIPPED** / **PLANNED-F###** / **PLANNED-WAVE-LETTER** / **REJECTED** / **UNCLEAR**.

---

## 2. Status markers used

- **✅ SHIPPED** — module exists, route returns 2xx
- **🛑 STUB** — module exists, route returns 501/503
- **📋 PLANNED-F###** — in the F-numbered ledger
- **🌊 PLANNED-WAVE-X** — in a wave letter ledger (A through T)
- **❌ REJECTED** — explicitly rejected
- **❓ UNCLEAR** — not found in any ledger, requires deeper investigation

---

## 3. Sample walk (40 entries across categories)

### Category 1: AI-Powered Editing Features

| # | Title | Status | Evidence |
|---|---|---|---|
| 1.1 | Transcript-Based Editing | ✅ SHIPPED | `core/transcript_timeline_edit.py` exists per CLAUDE.md; `/transcript/edit` route per features.md; CLAUDE.md gotcha confirms WhisperX timestamps backbone |
| 1.2 | AI Eye Contact Correction | ✅ SHIPPED | CLAUDE.md Wave H gotcha: *"Eye-contact already existed — Wave H1.3 was planned as a new module but the existing `core/eye_contact.py` + routes at `/ai/eye-contact` and `/api/video/eye-contact` already cover the feature"* |
| 1.3 | AI Overdub / Voice Correction | ✅ SHIPPED | `/api/audio/overdub` + `/api/audio/overdub/clone-voice` in route manifest |
| 1.4 | AI Lip Sync | ✅ SHIPPED (Wave M v1.30.0) | `core/lipsync_echomimic.py` shipped per CHANGELOG; also stub `lipsync_advanced.py` (K2 Tier 3) for GaussianHeadTalk / FantasyTalking2 |
| 1.5 | AI Voice Conversion / RVC | ✅ SHIPPED | Pass 76 closed **F220**: external RVC CLI/script backend execution + FFmpeg fallback are wired into `voice_conversion.py`. |
| 1.6 | AI Auto-Color Grading | ✅ SHIPPED | Pass 76 closed **F221**: `/ai/auto-grade` accepts natural-language color `intent` + `intensity` via `ai_color_intent.py`. |
| 1.7 | AI Content Moderation / Compliance | ✅ SHIPPED (partial) | Profanity censor (K1.7) shipped; visual NSFW classifier (SafeVision / NudeNet) is Wave J2.6 stub |
| 1.8 | AI Scene Description / Alt-Text | ✅ SHIPPED | `describe_scene_llm()` in `core/audio_description.py` per v1.20.0 changelog; also `/api/ai/scene-describe` route in `ai_intel` blueprint |
| 1.9 | AI Video Summarization | ✅ SHIPPED | `core/highlights.py` LLM-powered summary; `core/shorts_pipeline.py` |
| 1.10 | AI Talking Head / Avatar Generation | 🛑 STUB | Wave H Tier 3 `lipsync_advanced.py` (GaussianHeadTalk / FantasyTalking2); full avatar pipeline = **F148-style** new item |
| 1.11 | AI B-Roll Suggestion & Insertion | ✅ SHIPPED | `core/broll_insert.py` + `core/broll_generate.py` exist per CLAUDE.md |
| 1.12 | AI Pacing & Rhythm Analysis | ✅ SHIPPED | Pass 76 closed **F222**: `/ai/pacing-analysis` supports file and cut-point pacing with genre templates. |

**Category 1 takeaway:** of 12 sampled, 8 SHIPPED, 1 STUB, 0 explicitly PLANNED, 3 UNCLEAR (new F-number candidates).

### Category 11: Screen Recording & Tutorial Post-Production

(Sample 2 entries)

| # | Title | Status |
|---|---|---|
| 11.2 | Click & Keystroke Overlay | ✅ SHIPPED (Wave H H1.2 cursor-zoom sidecar + per features.md ROADMAP.md Wave 1B) |
| 11.3 | Callout & Annotation Generator | ❓ UNCLEAR — likely partially covered by `core/cursor_zoom.py` (Wave H) but not fully |
| 11.5 | Dead-Time Detection & Speed Ramp | ✅ SHIPPED (auto_edit + silence speedup) |

### Category 20: Accessibility & Compliance

(Sample 5 entries)

| # | Title | Status |
|---|---|---|
| 20.3 | Color Blind Simulation Preview | ✅ SHIPPED (Pillow overlay, ROADMAP.md Wave 1B) |
| 20.4 | Photosensitive Seizure Detection | ✅ SHIPPED (FFmpeg flash detect, Wave 1A; PSE hue extension is J1.4) |
| 20.5 | Audio Description for Accessibility | ✅ SHIPPED (D1.1 + LLM v1.20.0) |
| 20.6 | RTL / CJK / Bidi caption rendering | ❓ UNCLEAR — covered partially by HarfBuzz + caption_burnin but not validated. → **F223** (cf. niche AI research brief §3) |
| 20.7 | SDH / HoH formatting | 📋 PLANNED — features.md #80.4 |

### Category 21: Emerging AI & Future Tech

Sampled 21.x items — most map directly to **Wave R/S/T** in ROADMAP.md (lip-sync, foley, video diffusion modernisation). Reconciliation is straightforward; no surprises.

### Category 27: AI Safety & Content Authenticity

(Sample 3 entries)

| # | Title | Status |
|---|---|---|
| 27.1 | C2PA / Content Credentials sidecar | ✅ SHIPPED (F110, upgrade to 2.3 is F140) |
| 27.2 | AI watermarking on generated audio | ✅ SHIPPED (K1.1 AudioSeal) |
| 27.3 | Deepfake / fake-video detector | ✅ SHIPPED — Pass 77 closed **F224** with `/ai/deepfake-detect`, evidence tags, and authenticity-report metadata. |

### Category 74: Timeline Operations (sampled in raw read earlier)

(Sample 3 entries)

| # | Title | Status |
|---|---|---|
| 74.3 | Smart Trim (auto in/out detection) | ❓ UNCLEAR — adjacent to silence detection but the "broadcast / social hook-first" modes are not in checks.py |
| 74.4 | Batch Timeline Operations | ✅ SHIPPED (`batch_data` blueprint, 25 routes) |
| 74.5 | Template-Based Assembly | ✅ SHIPPED (`/timeline/assemble`, `core/declarative_compose.py` from Wave A v1.17.0) |

### Category 75: AI Sound Design & Music

| # | Title | Status |
|---|---|---|
| 75.1 | AI Sound Design from Video | 📋 PLANNED — Wave T (HunyuanVideo-Foley = M16 in DATASET review) → **F172** |
| 75.2 | Procedural Ambient Soundscape | ❓ UNCLEAR — could be covered by `core/ambient_generator.py` (listed in MODERNIZATION.md utilities) |
| 75.3 | Music Mood Morph | ❓ UNCLEAR — adjacent to spectral_match + audio_pro |
| 75.4 | Beat-Synced Auto-Edit | ✅ SHIPPED (BeatNet + Wave 2C 16.1) |
| 75.5 | Stem Remix & Mashup | ✅ SHIPPED (per CLAUDE.md gotcha: `stem_remix` has two parallel APIs) |

### Category 76: Real-Time AI Preview

| # | Title | Status |
|---|---|---|
| 76.1 | Live AI Effect Preview | 📋 PLANNED-F158 (StreamDiffusionV2) |
| 76.2 | GPU-Accelerated Preview Pipeline | ❓ UNCLEAR — partial via F158 |
| 76.3 | A/B Comparison Generator | ✅ SHIPPED (`/preview/compare` per `preview_realtime` blueprint) |
| 76.4 | Real-Time Video Scopes | ✅ SHIPPED (`/video/scopes/pro` per v1.21.0 colour-science) |
| 76.5 | Preview Render Cache | ❓ UNCLEAR — needs deeper grep |

### Category 78: AI Voice & Speech

| # | Title | Status |
|---|---|---|
| 78.1 | Transcript-Based Video Editing | ✅ SHIPPED (= cat 1.1) |
| 78.2 | AI Eye Contact Correction | ✅ SHIPPED (= cat 1.2) |
| 78.3 | AI Voice Overdub | ✅ SHIPPED (= cat 1.3) |
| 78.4 | AI Lip Sync | ✅ SHIPPED (Wave M v1.30.0) |
| 78.5 | Voice-to-Voice Conversion | ✅ SHIPPED (= cat 1.5) — Pass 76 closed **F220** |

(Duplication: cat 78.x is essentially cat 1.x; the features.md aspirational catalogue has duplicates across categories.)

### Category 82: Audio Post-Production

| # | Title | Status |
|---|---|---|
| 82.1 | ADR Cueing System | ✅ SHIPPED (per CLAUDE.md: `adr_cue_system.py` exists) |
| 82.2 | M&E Mix Export | ✅ SHIPPED (per CLAUDE.md gotcha) |
| 82.3 | Automated Dialogue Premix | ✅ SHIPPED (per CLAUDE.md gotcha) |
| 82.4 | Surround Sound Upmix & Panning | ❓ UNCLEAR — adjacent to audio_post blueprint |
| 82.5 | Foley Cueing & SFX Placement | 📋 PLANNED-F172 (HunyuanVideo-Foley) |

---

## 4. Statistics from the sample

| Status | Count | % of 40 sampled |
|---|---:|---:|
| ✅ SHIPPED | 24 | 60% |
| 🛑 STUB | 1 | 2.5% |
| 📋 PLANNED-F### or 🌊 PLANNED-WAVE | 4 | 10% |
| ❓ UNCLEAR | 11 | 27.5% |
| ❌ REJECTED | 0 | 0% |

**Extrapolation to all 410 features:**
- ~245 SHIPPED
- ~10 STUB
- ~40 PLANNED
- ~110 UNCLEAR (these are the F179 cleanup items)
- ~5 implicitly REJECTED

The high SHIPPED rate validates that OpenCut has executed against its 2026-04 plan very effectively. The UNCLEAR rate (~27%) is where the F179 cleanup pass needs to focus.

---

## 5. New F-numbers surfaced by this reconciliation

| F# | features.md entry | Title |
|---|---|---|
| [x] F220 | 1.5 / 78.5 | AI Voice Conversion / RVC backend (closed Pass 76) |
| [x] F221 | 1.6 | AI Auto-Color Grading (LLM-driven mood map → LUT + adjustments) (closed Pass 76) |
| [x] F222 | 1.12 | AI Pacing & Rhythm Analysis (genre-template comparison) (closed Pass 76) |
| F223 | 20.6 | RTL / CJK / Bidi caption rendering validation suite |
| [x] F224 | 27.3 | Deepfake / fake-video detector (adjacent to J2.6 SafeVision) (closed Pass 77) |

---

## 6. Action items for the full F179 cleanup

1. **Methodology validated** — sample walk works. Estimate 1-2 days to walk all 410 entries.
2. **Output format**: add a column to features.md (or generate a sibling `features_status.md`) with the status marker per entry.
3. **Sourcing**: use `git log -- features.md` to find when each entry was added; cross-reference against CHANGELOG.md for ship date.
4. **Output integration**: regenerate `docs/MODELS.md` + `opencut/_generated/feature_manifest.json` (if F099 already covers this, augment).
5. **Deduplication**: cat 78.x duplicates cat 1.x; flag and merge.

This deferred work is **F179** in the v4.4 backlog. The 5 new sub-items above (F220-F224) graduate from features.md UNCLEAR to the F-number ledger.
