# Roadmap — OpenCut

Actionable work only. Historical and completed roadmap material is archived in CHANGELOG.md; blocked work is kept in Roadmap_Blocked.md.

## Research-Driven Additions

Added 2026-08-10, extended 2026-08-11 and 2026-08-20 from the research passes recorded in `RESEARCH.md`.
IDs continue the existing F-number scheme (highest prior allocation before 2026-08-11: F318; before
2026-08-20: F334).

### P1

### P2

### P3

- [ ] P3 — F346 — Activate the /analyze/video/qwen3vl lane through local Ollama vision models
  Why: Content-aware editing ("cut the boring parts" from semantic video understanding, not just audio) is where the closest CLI competitor and the agentic wave are heading, and OpenCut already has the route stubbed (`/analyze/video/qwen3vl`, 501) plus an LLM layer that fronts Ollama — which serves Qwen-VL-class models locally — so one stub activation delivers per-segment semantic relevance scoring with no cloud key.
  Evidence: `opencut/_generated/route_manifest.json` (qwen3vl/internvl3 stubs); https://github.com/WyattBlue/auto-editor/issues/1273 (content-aware edit method, 2026-06-25); `opencut/core/llm.py` (Ollama support); the text-first economy pattern from browser-use/video-use recorded in the 2026-08-11 pass
  Touches: `opencut/core/multimodal_qwen3vl.py` (remove terminal NotImplementedError per readiness rules), `opencut/core/llm.py`, the wave_qrs route, highlights integration, `opencut/model_cards.py`, `tests/`
  Acceptance: The route leaves stub state through the established readiness flow (stub_scan reclassifies it once the terminal raise is gone and `check_X_available()` gates it); frame-sampled scoring returns per-segment relevance keeping transcript as the primary signal and pixels at decision points; it runs against a local Ollama vision model with no API key; the manifest and README counts regenerate.
  Complexity: L
