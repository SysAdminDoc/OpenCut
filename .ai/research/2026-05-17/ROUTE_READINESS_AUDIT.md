# OpenCut — Route Readiness Audit

**Audit date:** 2026-05-17 (Pass 2)
**Source of truth:** `opencut/_generated/route_manifest.json`, `opencut/registry.py` (F100/F191 feature catalogue), `opencut/_generated/feature_readiness.json` (58 generated records / 67 route bindings), `opencut/checks.py` (118 public `check_*` probes, 86 `check_*_available` gates), `opencut/_generated/model_cards.json` (47 cards), `opencut/openapi_registry.py` + `opencut/openapi.py` (110 discovered response-schema bindings after F193), `opencut/mcp_server.py` (39 curated tools after F195).

---

## 1. Totals

| Metric | Count |
|---|---|
| Total HTTP routes | **1,376** |
| GET routes | 300 |
| POST routes | 1,050 |
| DELETE routes | 13 |
| PUT routes | 2 |
| PATCH routes | 1 |
| Blueprints | **101** |
| Feature readiness records exposed by `registry.py` (F100/F191) | **84** |
| Generated readiness records / route bindings | **58** / **67** |
| Routes with explicit response-schema in `openapi.py` | **110** |
| Public `check_*` probes / `check_*_available` gates | **118** / **86** (in `opencut/checks.py`) |
| Model cards in `model_cards.json` (F115) | **47** |
| MCP tools in `mcp_server.py` MCP_TOOLS array | **39** |

Coverage gap after Pass 71: 1,376 routes vs 84 registry records / 67 generated route bindings vs 110 OpenAPI schemas vs 39 curated MCP tools. F191 improved the readiness surface for direct route/check bindings, F192 expanded the typed response-schema table, F193 made it dataclass-discovered, and F195 added 12 shipped post-Wave-M MCP tools, but most routes still have no typed response schema and no MCP surface. This remains a structural visibility gap.

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
| **F100/F191 registered with explicit state** | 84 records / 67 generated route bindings | `opencut/registry.py`, `opencut/_generated/feature_readiness.json` |
| **OpenAPI-schema-typed** | 110 | `opencut/openapi_registry.py` dataclass discovery feeding `opencut/openapi.py` |

**Pass 8 update:** `GET /system/feature-state` now includes generated F191 records from direct route/check bindings. It still is not a per-route readiness matrix for all 1,365 routes; it is a feature-readiness manifest with generated route lists for probes the scanner can see.

**Recommended fix (new F-number):**
- **F191** — **DONE in Pass 8.** `opencut.tools.dump_feature_readiness` statically scans route functions for public `checks.py` probes, joins endpoints to the live route manifest, writes `opencut/_generated/feature_readiness.json`, and `registry.py` loads/merges those generated rows into `GET /system/feature-state`.

---

## 4. OpenAPI coverage gap

`opencut.openapi_registry` now discovers registered response dataclasses and feeds `opencut/openapi.py` `_ENDPOINT_SCHEMAS`, which maps **110 endpoints** to typed dataclasses.

**Pass 41 update:** F192 closed the first bulk expansion by adding reusable response envelopes in `opencut/schemas.py` and mapping the next high-traffic jobs/system/model/GPU/MCP/analytics/annotations/captions/audio/TTS/settings/AI-catalogue route batch. `tests/test_openapi_contract.py` pins a floor of 80 typed paths and checks representative generated response properties.

**Pass 71 update:** F193 replaced the legacy endpoint hand-table with dataclass-discovered route metadata and added selected `core/*Result` payloads such as audio-description drafts, transfer bundles, marker imports, eval dataset details, crash packets, project health, OCIO validation, review bundles, and C2PA sidecars. The OpenAPI contract now pins a floor of 105 typed paths plus nested dataclass properties.

The original 30-endpoint seed covered:
- `/health`, `/system/update-check`
- 4 deliverables, 1 export-from-markers
- 1 silence, 2 audio (loudness-match, beat-markers)
- 3 video core (color-match, auto-zoom, multicam-cuts)
- 2 captions (chapters, repeat-detect)
- 1 context, 1 workflow
- 4 video AI (upscale, rembg, interpolate, denoise)
- 1 shorts pipeline, 3 depth, 1 broll
- 1 plugins list

`_JOB_ENDPOINTS` set (35 endpoints) returns `JobResponse` schema for async POSTs that are not explicitly mapped.

**The other ~1,250 routes get `{type: "object"}` (essentially untyped) in the OpenAPI spec.** Documentation tools (Swagger UI, Insomnia, Postman) will show the bare endpoint with no field info.

**Recommended fix:**
- **F192** — **DONE in Pass 41.** Bulk-added typed response dataclasses for the first high-traffic route batch; legacy `/openapi.json` now has 100 typed endpoints.
- **F193** — **DONE in Pass 71.** Replaced the `_ENDPOINT_SCHEMAS` hand-table with dataclass discovery over registered schema/core result classes.

---

## 5. MCP tool coverage gap

`mcp_server.py` exposes **39 MCP tools** after F195:
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
28. `opencut_face_reshape`
29. `opencut_skin_retouch`
30. `opencut_smart_upscale`
31. `opencut_elevenlabs_tts`
32. `opencut_caption_qc`
33. `opencut_review_bundle`
34. `opencut_c2pa_provenance`
35. `opencut_marker_import`
36. `opencut_capability_probe`
37. `opencut_brand_kit`
38. `opencut_semantic_search`
39. `opencut_spectral_match`

**Coverage: 39 of 1,359 routes (~3%).** The competing AdobePremiereProMCP server exposes 1,060 tools (mostly auto-generated boilerplate over Premiere CEP calls). OpenCut's choice to hand-curate ergonomic tools is the right one **for now** — but the maintainer should at least consider auto-generation for tier-2 surfaces.

**Recommended fix:**
- **F194** — **DONE in Pass 43.** `opencut/_generated/mcp_extended_tools.json` now contains 1,307 opt-in `opencut_route_*` tools generated from the route manifest + OpenAPI schema map; default MCP remains the 39 curated tools unless `OPENCUT_MCP_EXTENDED_TOOLS=1` or `--extended-tools` is set.
- **F195** — **DONE in Pass 9.** `opencut/mcp_server.py` now exposes `opencut_face_reshape`, `opencut_skin_retouch`, `opencut_smart_upscale`, `opencut_elevenlabs_tts`, `opencut_caption_qc`, `opencut_review_bundle`, `opencut_c2pa_provenance`, `opencut_marker_import`, `opencut_capability_probe`, `opencut_brand_kit`, `opencut_semantic_search`, and `opencut_spectral_match`; `tests/test_mcp_server.py` pins registration, dispatch, special actions, and path validation.

---

## 6. The model card vs check function vs registry mismatch

There are three overlapping catalogues:
- **`checks.py`**: 117 public `check_*` probes, including 86 `check_X_available()` gates (covers everything from `demucs` to `cinefocus`)
- **`model_cards.py`**: 47 cards with licence + hardware + privacy + install hint
- **`registry.py` + `_generated/feature_readiness.json`**: 84 `FeatureRecord` entries with readiness state + route list

The deltas:
- Functions in `checks.py` without a model card still exist for system / orchestration checks and stdlib-only guards.
- Functions in `checks.py` without a `FeatureRecord` are reduced but not eliminated; F191 only covers route functions that visibly call the public probe or a core helper aliased by checks.py.
- Model cards without a curated manual `FeatureRecord` now often surface via generated records, but F196 is still needed if registry becomes the primary catalogue.

**Recommended fix:**
- **F196** — Make `registry.py` the **primary** catalogue and have `model_cards.py` + `checks.py` derived from it (or at least cross-validated in CI via `release_smoke.py`).
- **F197** — **DONE in Pass 8.** `NON_AI_CHECKS` now lives in `registry.py`; `model_cards.py` imports the registry-owned tuple so F115 and F191 share one allowlist.

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
- **F198** — **DONE in Pass 42.** `opencut/core/cep_uxp_parity.py` and `opencut/_generated/cep_uxp_parity.json` catalogue all 18 `ocXxx` CEP host functions and pin the two true CEP-only calls (`ocAddNativeCaptionTrack`, `ocQeReflect`) with replacement plans.

---

## 8. Auto-routes via `/api/*` aliases

The manifest contains **233 routes under `/api/*`**. Pass 7 corrected the original assumption that these were mostly aliases: `opencut/_generated/api_aliases.json` shows **15 true aliases** and **218 canonical `/api` routes**. A route is now counted as an alias only when an `/api/*` rule has the same methods as an equivalent bare rule after stripping `/api`.

**Action:**
- **F199** — **DONE in Pass 7.** `opencut.tools.dump_api_aliases` generates `opencut/_generated/api_aliases.json`, and release smoke checks it with `python -m opencut.tools.dump_api_aliases --check`.

---

## 9. Recommended new F-numbers (from this Pass-2 route audit)

| F# | Title | Priority | Effort |
|---|---|---|---|
| F191 | Auto-derive `FeatureRecord` from check functions + route manifest | Done in Pass 8 | M |
| F192 | Bulk add OpenAPI response schemas for top 50 routes | Done in Pass 41 | M |
| F193 | Replace `_ENDPOINT_SCHEMAS` hand-table with dataclass introspection | Done in Pass 71 | M |
| F194 | Auto-generate "extended" MCP tools from route manifest | Done in Pass 43 | L |
| F195 | Add 12 missing MCP tools for post-Wave-M shipped routes | Done in Pass 9 | S |
| F196 | Make `registry.py` primary; derive `model_cards` / `checks` | Later | L |
| F197 | Add `NON_AI_CHECKS` allowlist to `registry.py` | Done in Pass 8 | S |
| F198 | CEP-only route catalogue + UXP replacement plan | Done in Pass 42 | M |
| F199 | Document `/api/*` alias policy + generate alias map | Done in Pass 7 | S |

All flow into Pass-2 updates to `FEATURE_BACKLOG.md`, `PRIORITIZATION_MATRIX.md`, and ROADMAP v4.5.
