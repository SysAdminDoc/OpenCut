# Roadmap

Single task tracker for known issues and planned work. Items below come from
the June 2026 engineering/product audit (verified findings, with file
locations); fixes already shipped are recorded in CHANGELOG.md and git
history, not here.

Blocked items (credential/license/hardware-gated) live in
[`Roadmap_Blocked.md`](Roadmap_Blocked.md).

## P3

- [ ] P3 — Surface model privacy/license posture in feature-state and engine controls
  Why: OpenCut's strongest competitive wedge is local/private execution with explicit model licensing, but that posture is mostly in model cards and docs instead of the action point where users choose engines or cloud-backed features.
  Evidence: `opencut/model_cards.py`, `docs/MODELS.md`, `/system/feature-state`, `extension/com.opencut.panel/client/feature-state.js`, local-only mode, competitor cloud/credit positioning from Adobe, Descript, Opus, HeyGen/Rask, and Premiere plugin tools
  Touches: model-card registry, feature-state response schema, CEP/UXP engine controls, local-only gating UI, locale strings, schema/contract tests, panel tests
  Acceptance: model-backed feature-state entries expose local/cloud, license, install, and hardware posture; CEP and UXP show concise badges/tooltips before execution; local-only mode visibly blocks cloud-backed features before click; tests cover at least one local, dependency-gated, restricted-license, and cloud-backed feature.
  Complexity: M

- [ ] P3 — Local multimodal footage search (object + on-screen-text + sound-event index)
  Why: Premiere 26 Media Intelligence does NL search across visuals, transcripts, and described sounds on-device; OpenCut indexes only speech + CLIP frames. Object-detection/OCR/audio-event indexes would match it locally — a defensible OSS differentiator.
  Evidence: opencut/core/semantic_video_search.py (speech + CLIP only); helpx.adobe.com Media Intelligence; twelvelabs.io/product/video-search
  Touches: semantic_video_search.py, FTS5 index schema, footage indexing pipeline, search route + panel Search tab
  Acceptance: footage search returns hits for on-screen objects, on-screen text, and sound events (not just spoken words), with the index built locally and incrementally.
  Complexity: L

- [ ] P3 — Add scene-detection AutoShot engine (beats TransNetV2 on gradual transitions)
  Why: AutoShot beats TransNetV2 by ~4.2% F1 and handles gradual transitions PySceneDetect misses; scene_detect.py registry is only threshold + TransNetV2.
  Evidence: opencut/core/scene_detect.py (threshold + transnetv2); engine_registry.py scene_detection domain; github.com/wentaozhu/AutoShot
  Touches: opencut/core/scene_detect.py, engine_registry.py, checks.py, scene-detection route/preset
  Acceptance: AutoShot is a selectable scene-detection engine with availability check and registry test; TransNetV2/threshold retained.
  Complexity: M

- [ ] P3 — Import-by-link footage ingest connectors
  Why: Opus Clip ingests directly from Zoom Clips / Apple Podcasts / Medal links with no re-upload; OpenCut requires a local file path for every operation.
  Evidence: coldiq.com/tools/opus-clip (May 2026 import-by-link); OpenCut routes require local filepath
  Touches: new ingest module (URL → local cache via yt-dlp-class fetch), watch_folder/scheduled_jobs integration, panel input affordance
  Acceptance: a supported URL is fetched to the local media cache and becomes usable by any existing route exactly like a local file; failures return structured errors.
  Complexity: M
