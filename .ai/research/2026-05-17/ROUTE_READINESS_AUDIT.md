# OpenCut — Route Readiness Audit

**Audit date:** 2026-05-17 (Pass 2)
**Source of truth:** `opencut/_generated/route_manifest.json` (generated 2026-05-16T20:36:05Z), `opencut/registry.py` (F100 feature catalogue, 514 lines), `opencut/checks.py` (105 functions), `opencut/_generated/model_cards.json` (47 cards), `opencut/openapi.py` (mapping table).

---

## 1. Totals

| Metric | Count |
|---|---|
| Total HTTP routes | **1,359** |
| GET routes | 295 |
| POST routes | 1,049 |
| DELETE routes | 13 |
| PUT routes | 2 |
| PATCH routes | 1 |
| Blueprints | **101** |
| Routes with explicit `FeatureRecord` in `registry.py` (F100) | **29** |
| Routes with explicit response-schema in `openapi.py` | **30** |
| Optional-dependency `check_X_available()` functions | **105** (in `opencut/checks.py`) |
| Model cards in `model_cards.json` (F115) | **47** |
| MCP tools in `mcp_server.py` MCP_TOOLS array | **27** |

Coverage gap: 1,359 routes vs 29 in registry vs 30 in OpenAPI schemas vs 27 in MCP. **The vast majority of routes have no machine-readable readiness state and no typed response schema.** This is a structural visibility gap.

---

## 2. Top blueprints by route count

| Blueprint | Routes | Likely role |
|---|---:|---|
| `system` | 60 | Health, info, dependencies, GPU, models, capability, telemetry, deprecation |
| `wave_k` | 59 | v1.28.0 completeness pass — AudioSeal, Brand Kit, Podcast Suite, Batch Reframe, Clip Rating, Subtitle QA, Spectral Match, Lottie, Semantic Search + 27 Tier 2/3 stubs |
| `platform_infra` | 53 | Cross-cutting infra (jobs, queue, throttle, OS hooks) |
| `integration` | 40 | Adjustment layers, Resolve bridge, Premiere XMP, OAuth, etc. |
| `audio` | 38 | Original audio bp (silence, separate, normalize, etc.) |
| `color_mam` | 34 | Color + Media Asset Management |
| `wave_h` | 34 | v1.25.0 commercial-parity Tier 1 + Tier 2/3 stubs |
| `settings` | 30 | Presets / favorites / workflows / LLM / brand kit / onboarding |
| `batch_data` | 25 | Batch ops, CSV/JSON imports |
| `captions` | 24 | Whisper / WhisperX / styled / animated / karaoke / QC |
| `editing_wf` | 24 | Editing workflow surface |
| `video_fx` | 24 | FX, chromakey, LUT, particles, style transfer |
| `workflow_dev` | 24 | Developer scripting / macros / webhooks |
| `audio_adv` | 22 | Advanced audio processing |
| `music_safety` | 21 | Music content ID / safety / clearance |
| `pipeline_intel` | 21 | Pipeline + intelligence surfaces |
| `platform_ux` | 21 | Panel UX, command palette |
| `documentary` | 20 | Documentary-specific workflow |
| `utility` | 19 | Utility endpoints |
| `video_vfx` | 19 | VFX surfaces |
| ... | ... | (full list in `route_manifest.json`) |
| `nlp` | 1 | NLP / chat (the very thin blueprint that hosts the future agent-chat conductor) |
| `<app>` | 1 | Flask static |

**Observation:** the `nlp` blueprint has **1 route** — `/nlp/command` (natural-language → API route mapping with 19-entry keyword + LLM dispatch). The proposed `/agent/chat` conductor (F143) would land here or in a sibling `agent` blueprint. The blueprint is currently underweight relative to its strategic importance.

---

## 3. Route readiness breakdown (best inferable)

| Readiness state | Routes (estimate) | Source |
|---|---|---|
| **Live** (tested + working with no optional dep) | ~700-900 | Default state; manifest entries without a known stub marker |
| **Live with optional dep** (gracefully 503 when missing) | ~250-300 | Inferred from 105 check_X_available + wave_a..wave_l blueprints |
| **Stub 503 `MISSING_DEPENDENCY`** | ~30-50 | Wave K Tier 2 (19) + Wave H Tier 2 (6) + scattered |
| **Stub 501 `ROUTE_STUBBED`** | ~12-18 | Wave K Tier 3 (8) + Wave H Tier 3 (3) + scattered |
| **F100-registered with explicit state** | 29 | `opencut/registry.py` |
| **OpenAPI-schema-typed** | 30 | `opencut/openapi.py` `_ENDPOINT_SCHEMAS` |

**Critical gap:** there is no single endpoint that returns "for every route, what readiness state does it have?". The closest is `GET /system/feature-state` (F100) which only covers 29 records, and `GET /system/dependencies` which only covers ~30 deps via the dashboard.

**Recommended fix (new F-number):**
- **F191** — Auto-derive `FeatureRecord` for every route that maps to a known `check_X_available()`. Today only 29 of ~250-300 dep-gated routes are in the F100 registry. The dispatch should be:
  1. parse route_manifest.json
  2. for each route, look up its endpoint's check function via decorator introspection (e.g. `@async_job` could carry a `check=check_demucs_available` kwarg)
  3. auto-generate `FeatureRecord(state=AVAILABLE, probe=check_X)` entries
  4. surface in `GET /system/feature-state` so the panel can grey out gated routes even when they're not hand-added to the registry

---

## 4. OpenAPI coverage gap

`opencut/openapi.py` `_ENDPOINT_SCHEMAS` maps **30 endpoints** to typed dataclasses:
- `/health`, `/system/update-check`
- 4 deliverables, 1 export-from-markers
- 1 silence, 2 audio (loudness-match, beat-markers)
- 3 video core (color-match, auto-zoom, multicam-cuts)
- 2 captions (chapters, repeat-detect)
- 1 context, 1 workflow
- 4 video AI (upscale, rembg, interpolate, denoise)
- 1 shorts pipeline, 3 depth, 1 broll
- 1 plugins list

`_JOB_ENDPOINTS` set (35 endpoints) returns `JobResponse` schema for async POSTs.

**The other ~1,300 routes get `{type: "object"}` (essentially untyped) in the OpenAPI spec.** Documentation tools (Swagger UI, Insomnia, Postman) will show the bare endpoint with no field info.

**Recommended fix:**
- **F192** — Bulk-add typed response dataclasses for the top 50 most-used routes (the 280-route video group should at least share a `VideoResult` base). Schemas could be auto-generated from runtime sample responses via a small CI script.
- **F193** — Replace `_ENDPOINT_SCHEMAS` hand-table with introspection over `core/*Result` dataclasses (most modules already return subscriptable dataclasses per CLAUDE.md convention).

---

## 5. MCP tool coverage gap

`mcp_server.py` exposes **27 MCP tools**:
1. `opencut_transcribe`
2. `opencut_silence_remove`
3. `opencut_export_video`
4. `opencut_highlights`
5. `opencut_separate_audio`
6. `opencut_tts`
7. `opencut_style_transfer`
8. `opencut_face_enhance`
9. `opencut_generate_music`
10. `opencut_job_status`
11. `opencut_repeat_detect`
12. `opencut_chapters`
13. `opencut_footage_search`
14. `opencut_index_footage`
15. `opencut_color_match`
16. `opencut_loudness_match`
17. `opencut_auto_zoom`
18. `opencut_multicam_cuts`
19. `opencut_denoise_audio`
20. `opencut_upscale`
21. `opencut_scene_detect`
22. `opencut_depth_map`
23. `opencut_shorts_pipeline`
24. `opencut_dub_video` (Wave M)
25. `opencut_sports_highlights` (Wave M)
26. `opencut_lipsync_echomimic` (Wave M)
27. `opencut_chat_edit` (Wave M)

**Coverage: 27 of 1,359 routes (~2%).** The competing AdobePremiereProMCP server exposes 1,060 tools (mostly auto-generated boilerplate over Premiere CEP calls). OpenCut's choice to hand-curate 27 ergonomic tools is the right one **for now** — but the maintainer should at least consider auto-generation for tier-2 surfaces.

**Recommended fix:**
- **F194** — Auto-generate a `tools/extended/` set of MCP tools from the route manifest + OpenAPI schema. Tag them clearly as "auto-generated, lower priority" so MCP clients (Claude Code, Cursor) can opt in/out. This pairs with F192 (typed schemas).
- **F195** — Extend the MCP_TOOLS with the post-Wave-M shipped routes: `opencut_face_reshape`, `opencut_skin_retouch`, `opencut_smart_upscale`, `opencut_elevenlabs_tts`, `opencut_caption_qc`, `opencut_review_bundle`, `opencut_c2pa_provenance`, `opencut_marker_import`, `opencut_capability_probe`, `opencut_brand_kit`, `opencut_semantic_search`, `opencut_spectral_match`. (~12 new entries.)

---

## 6. The model card vs check function vs registry mismatch

There are three overlapping catalogues:
- **`checks.py`**: 105 `check_X_available()` functions (covers everything from `demucs` to `cinefocus`)
- **`model_cards.py`**: 47 cards with licence + hardware + privacy + install hint
- **`registry.py`**: 29 `FeatureRecord` entries with readiness state + route list

The deltas:
- Functions in `checks.py` without a model card: ~58 (system / orchestration checks like `check_neural_interp_available()` and stdlib-only checks like `check_voice_grammar_available()`).
- Functions in `checks.py` without a `FeatureRecord`: ~76. Most are wave_a..wave_k optional surfaces.
- Model cards without a `FeatureRecord`: ~30. The F115 cards are richer than the F100 registry.

**Recommended fix:**
- **F196** — Make `registry.py` the **primary** catalogue and have `model_cards.py` + `checks.py` derived from it (or at least cross-validated in CI via `release_smoke.py`).
- **F197** — Add a `NON_AI_CHECKS` allowlist to `registry.py` (mirror of F115's allowlist) for stdlib-only or orchestration-only checks that intentionally have no model card.

---

## 7. Routes that look CEP/ExtendScript-bound (Sept 2026 EOL risk)

A quick grep through the manifest for endpoints that obviously depend on ExtendScript:

| Route prefix | Routes | EOL risk |
|---|---:|---|
| `/resolve/*` | 10 | None — Resolve has its own Python scripting API |
| `/api/adjustment-layers/*` | 4 | **HIGH** — likely calls ExtendScript via panel |
| `/api/markers/*`, `/api/sequence/*`, `/timeline/*` (most) | ~30 | **HIGH** — sequence/marker writes are the canonical ExtendScript bridge |
| `/api/scripting/execute`, `/api/macro/play` | 2 | **HIGH** — direct ExtendScript dispatch |
| `/system/qe-reflect` (F H2.8) | 1 | **HIGH** — QE DOM is CEP-only |

**Action:** F146 (UXP-native MCP transport) and F160 (WebView UXP migration) need to cover these surfaces. Recommend an explicit **"CEP-only route catalogue"** doc — new F:
- **F198** — Catalogue every route that requires ExtendScript/CEP and provide a UXP-replacement plan (or explicit "blocked by Adobe UXP API gap" note pointing to F186-F190).

---

## 8. Auto-routes via `/api/*` aliases

The manifest contains **233 routes under `/api/*`**. Pass 7 corrected the original assumption that these were mostly aliases: `opencut/_generated/api_aliases.json` shows **15 true aliases** and **218 canonical `/api` routes**. A route is now counted as an alias only when an `/api/*` rule has the same methods as an equivalent bare rule after stripping `/api`.

**Action:**
- **F199** — **DONE in Pass 7.** `opencut.tools.dump_api_aliases` generates `opencut/_generated/api_aliases.json`, and release smoke checks it with `python -m opencut.tools.dump_api_aliases --check`.

---

## 9. Recommended new F-numbers (from this Pass-2 route audit)

| F# | Title | Priority | Effort |
|---|---|---|---|
| F191 | Auto-derive `FeatureRecord` from check functions + route manifest | Now | M |
| F192 | Bulk add OpenAPI response schemas for top 50 routes | Next | M |
| F193 | Replace `_ENDPOINT_SCHEMAS` hand-table with dataclass introspection | Later | M |
| F194 | Auto-generate "extended" MCP tools from route manifest | Next | L |
| F195 | Add 12 missing MCP tools for post-Wave-M shipped routes | Now | S |
| F196 | Make `registry.py` primary; derive `model_cards` / `checks` | Later | L |
| F197 | Add `NON_AI_CHECKS` allowlist to `registry.py` | Now | S |
| F198 | CEP-only route catalogue + UXP replacement plan | Next | M |
| F199 | Document `/api/*` alias policy + generate alias map | Done in Pass 7 | S |

All flow into Pass-2 updates to `FEATURE_BACKLOG.md`, `PRIORITIZATION_MATRIX.md`, and ROADMAP v4.5.
