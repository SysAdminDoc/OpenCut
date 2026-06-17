# Roadmap

Single task tracker for known issues and planned work. Items below come from
the June 2026 engineering/product audit (verified findings, with file
locations); fixes already shipped are recorded in CHANGELOG.md and git
history, not here.

Blocked items (credential/license/hardware-gated) live in
[`Roadmap_Blocked.md`](Roadmap_Blocked.md).

## P3

- [ ] P3 — Local multimodal footage search (object + on-screen-text + sound-event index)
  Why: Premiere 26 Media Intelligence does NL search across visuals, transcripts, and described sounds on-device; OpenCut indexes only speech + CLIP frames. Object-detection/OCR/audio-event indexes would match it locally — a defensible OSS differentiator.
  Evidence: opencut/core/semantic_video_search.py (speech + CLIP only); helpx.adobe.com Media Intelligence; twelvelabs.io/product/video-search
  Touches: semantic_video_search.py, FTS5 index schema, footage indexing pipeline, search route + panel Search tab
  Acceptance: footage search returns hits for on-screen objects, on-screen text, and sound events (not just spoken words), with the index built locally and incrementally.
  Complexity: L

