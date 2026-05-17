# OpenCut — `/agent/chat` Conductor UX RFC (Pass 3)

**Audit date:** 2026-05-17 (Pass 3)
**Status:** RFC — describes the design space, names the chosen patterns, identifies the rejected ones. Not a commitment. Replaces the Pass-1+Pass-2 "F143 design space RFC" deferred item from CONTINUE_FROM_HERE.md §3.4.

**Concerns:** F143 (the conductor itself), F144 (post-turn self-review), F145 (Skills SDK + MCP packaging), with implications for F252 (UXP migration) and F225 (OTIO Marker review-bundle anchor).

---

## 1. Problem statement

OpenCut has 1,359 routes + 27 MCP tools + an `LLMConfig`-backed `core/llm.py` abstraction over Ollama/OpenAI/Anthropic/Gemini. It does **not** have a conductor that maps "intent → sequence of API calls → preview → commit." Competing systems (Descript Underlord, Captions.ai Mirage, FireRed-OpenStoryline, vibeframe, ViMax, CrePal, Odysser) all converge on the same UX pattern; OpenCut has every building block but no conductor.

**Goal:** ship `/agent/chat` with a UX that matches or beats Descript Underlord while respecting OpenCut's local-first, no-mandatory-cloud constraints.

---

## 2. Source patterns

| Pattern | Source | Adopt? | Rationale |
|---|---|---|---|
| Sidebar chat with timeline-diff before commit | Descript Underlord, Captions.ai Mirage | **Yes** | Industry-converging UX |
| Post-turn self-review (agent diffs its own work vs intent, re-invokes if drift) | Descript Underlord (2026) | **Yes (F144)** | Single highest trust-builder; cheap second-pass model |
| Editable plan **before** execution | GitHub Copilot Workspace | **Yes** | Cheapest intervention point; user rewrites bullets rather than rejecting after the fact |
| Checkpoint + rollback (don't pile new edits on a broken state) | Cursor Composer 2.0 | **Yes** | Maps cleanly to OTIO project snapshots (F225 anchor) |
| Per-file accept/reject | Cursor Composer; Claude Code GH issues #31395/#33932 (requested) | **Yes (extended to per-region)** | Per-region accept/reject is the video analog; required for trust |
| Atomic multi-file apply | Claude Code (current) | **No** | Pain point even in code; in video it's worse because per-region preview render cost is high |
| "Accept all edits" button | Cursor Composer | **No** | Render cost makes "oops, revert" expensive in attention even if data revert is instant |
| Auto-commit-before-preview | Aider | **No** | User must see rendered before/after first; commit-then-revert means judging from data structure |
| Snapshot-per-step (Aider's discipline, not its timing) | Aider | **Yes (F225 OTIO snapshot per accepted step)** | Cheap undo |
| Reusable skill manifest (`SKILL.md`) | Claude Code Skills, agentskills.io | **Yes (F145)** | Open standard since Dec 2025; works across Claude.ai/Code/SDK |
| MCP Inspector for debugging | Anthropic MCP Inspector | **Yes (dev-only)** | Invaluable for 27-tool surface during F143 dev |
| 2-stage reflection (Haiku decides what to fetch, Sonnet does the work) | Cody agentic chat | **Conditional** | Useful but adds latency; gate on per-skill flag |
| Editable bullet plan rendering | Copilot Workspace | **Yes** | Already match OpenCut's workflow-engine v2.0 step list |

---

## 3. Mapping IDE-agent concepts → video editor

| IDE concept | OpenCut equivalent |
|---|---|
| **File** | **Timeline region** = `(track_index, in_seconds, out_seconds)`. *Not* clip — clips are too granular, regions are the natural review unit. |
| **Multi-file plan** | Workflow chain (already exists as `core/workflow.py` v2.0); surface as editable bullet list before execution |
| **Diff** | Three-layer:<br>(a) **Plan pane** (top) — editable bullets<br>(b) **Thumbnail strip** (middle) — before/after frames at each affected cut, 5 frames per side, scrubbable<br>(c) **OTIO XML diff** (bottom, collapsible) — power users + agent self-review |
| **Run tests** | `preflight.py` checks + 480p proxy render of changed regions only + automated checks (gap detection, audio clipping ≥-1 dBTP, caption overlap, broken media refs) |
| **Git commit** | OTIO snapshot per accepted turn at `~/.opencut/snapshots/<session>/<step>.otio`. Premiere auto-save does its own thing; OpenCut emits a sibling OTIO file for round-tripping. |
| **Tab autocomplete (next edit site)** | "Next marker" jump: after agent inserts at marker N, suggest action at marker N+1. Also: silence/gap auto-fill, caption suggestion |
| **Re-prompt loop after reject** | Roll back to pre-step snapshot, surface "narrow the scope?" prompt with the rejected step's intent diff |

---

## 4. Proposed UI surface

**Location:** UXP panel (per F252 Bolt UXP + WebView UI migration). CEP panel will follow only if F252 is delayed; new chat-conductor work should not land in CEP-only.

**Layout:**

```
+----------------------------------------------------------+
| Tab: Chat | Cut | Captions | Audio | Video | Timeline | …|
+----------------------------------------------------------+
| > "Cut the silences, add captions in Spanish, export 9:16"|
+----------------------------------------------------------+
| PLAN (editable bullets)                              [Run]|
| □ 1. Detect silences (silero VAD, threshold -30 dB)       |
| □ 2. Apply cuts via ocApplySequenceCuts                   |
| □ 3. Transcribe (faster-whisper, large-v3, lang=auto)     |
| □ 4. Translate captions to es-ES (NLLB-200)               |
| □ 5. Burn in captions (style: YouTube Bold)               |
| □ 6. Reframe 9:16 with face tracking                      |
| □ 7. Export                                               |
+----------------------------------------------------------+
| DIFF (after run)                                          |
| [Thumb strip: before / after at each cut boundary]        |
| [▶ OTIO XML diff — 21 markers added, 14 cuts applied]     |
+----------------------------------------------------------+
| [Accept step 1] [Reject step 1] [Reject all → re-prompt]  |
| Self-review: ✓ silences cut as planned, ✓ captions match  |
|              intent, ⚠ reframe missed face at 0:42        |
+----------------------------------------------------------+
```

**Why this layout:** Copilot Workspace's plan-then-execute is the cheapest intervention. Per-step accept/reject (not "accept all") forces user to look at each preview. Post-turn self-review surfaces drift before user has to catch it. OTIO XML diff at the bottom is for the agent's own use and for power users.

---

## 5. Endpoint shape

**Request:**
```json
POST /agent/chat
{
  "session_id": "uuid",
  "intent": "Cut the silences, add Spanish captions, export 9:16",
  "context": {
    "active_sequence": "<sequence_id>",
    "selected_clip": "<clip_id?>",
    "filepath": "<resolved input?>"
  },
  "llm": { "provider": "ollama", "model": "llama3.2:latest" },
  "mode": "plan"  // or "execute" or "self-review"
}
```

**Response (mode=plan):**
```json
{
  "session_id": "uuid",
  "step_id": "uuid",
  "plan": [
    {"label": "...", "endpoint": "/silence", "payload": {...}, "skill": "polish-interview"},
    ...
  ],
  "snapshot_before": "<otio path>",
  "estimated_total_seconds": 47
}
```

**Response (mode=execute, per accepted step):**
```json
{
  "step_index": 2,
  "status": "complete",
  "job_id": "...",
  "diff": {
    "otio_diff_path": "<otio>",
    "thumbnails_before": ["<path>", ...],
    "thumbnails_after": ["<path>", ...]
  },
  "self_review": {
    "verdict": "match",  // or "drift" or "fail"
    "notes": "21 markers added matching plan",
    "recommended_reprompt": ""
  }
}
```

**Response (mode=self-review):**
```json
{
  "session_id": "uuid",
  "step_results": [
    {"step_index": 0, "verdict": "match", ...},
    {"step_index": 1, "verdict": "drift", "drift_summary": "reframe missed face at 0:42",
     "suggested_fix": {"endpoint": "/video/reframe", "payload": {"manual_anchor": ...}}}
  ]
}
```

Per the CLAUDE.md async-job pattern, `execute` mode launches `@async_job` steps with SSE/WebSocket progress to the panel.

---

## 6. Self-review (F144) implementation sketch

Pattern: after each accepted execute step, kick off a **cheap second-pass model call** (Haiku/Llama-tiny):

```python
def self_review(step_intent: str, step_result: dict, otio_diff: dict) -> dict:
    """Run a cheap second-pass to verify the step matched the user's intent."""
    prompt = f"""
    The user wanted: {step_intent}
    The agent did: {step_result['endpoint']} with payload {step_result['payload']}
    Resulting OTIO diff: {otio_diff_summary(otio_diff)}

    Answer in JSON: {{ "verdict": "match|drift|fail",
                       "drift_summary": "...",
                       "suggested_fix": {{...}} or null }}
    """
    return query_llm(prompt, model="ollama/llama3.2:1b").parsed_json
```

**Cost:** ~0.5-2s per step on a local Llama-3.2-1B; doubles total agent runtime but pays back the trust premium.

**Trigger:** automatic after every `execute` step. User can disable via `OPENCUT_AGENT_SELF_REVIEW=0`.

---

## 7. Skills SDK (F145) shape

Adopt the **Claude Code Skills** open standard (agentskills.io, Dec 2025).

**Layout** (`~/.opencut/skills/polish-interview/`):
```
SKILL.md           # front-matter + agent instructions
plan.json          # default workflow chain
preview_thumbs/    # optional UI hints
scripts/           # optional helper Python (must be MIT-friendly)
```

**SKILL.md front-matter:**
```yaml
---
name: polish-interview
description: Clean up an interview — silence cut, filler removal, transcript edit, normalize loudness, color match.
version: 1.0
author: opencut-community
license: MIT
applicable_to: ["sequence"]
required_features: ["silence", "captions.whisperx", "audio.loudness-match", "editing.color-match"]
estimated_seconds: 120
---

# Polish Interview

When the user asks for "clean up this interview" or "polish the talking head", run:
1. Silence detection (`POST /silence`) with threshold -30 dB
...
```

**Distribution:** ship 6 built-in skills (`polish-interview`, `podcast-cleanup`, `social-cuts`, `youtube-upload`, `documentary-rough`, `studio-audio`) matching the existing 6 workflow presets. User can drop new `SKILL.md` files into `~/.opencut/skills/` and they auto-register via the existing plugin loader pattern (F116).

**Discovery for MCP:** annotate every MCP tool with a `skill_compatible: ["polish-interview", "podcast-cleanup"]` tag so MCP clients can filter.

---

## 8. Three patterns deliberately NOT copied

1. **"Accept all edits" button (Cursor)** — render cost makes attention cost dominate. Force per-step approval until trust is established. If a user really wants accept-all, gate it behind a `--trust-agent` flag with prominent warning.
2. **Auto-commit-before-preview (Aider)** — user must see rendered before/after first. We adopt Aider's snapshot **discipline** (one OTIO per turn) but not its commit **timing** (preview-then-commit, not commit-then-revert).
3. **Atomic multi-file apply (Claude Code current behaviour)** — even Claude users are filing per-hunk-accept requests (issues #31395, #33932). In video, a 4-region change where one breaks should not force rejecting all four. Per-region accept/reject from day one.

---

## 9. Phasing (revised F143 / F144 / F145 effort)

Pass-2 estimates: F143 L, F144 M, F145 L. Pass-3 reclassification:

| F# | Title | Effort | Sub-phases |
|---|---|---|---|
| F143 | `/agent/chat` conductor (plan / execute / self-review modes) | **L** | F143.1 plan-mode (S), F143.2 execute-mode + per-step diff (M), F143.3 OTIO snapshot per step (S), F143.4 sidebar UI in UXP panel (M) |
| F144 | Post-turn self-review | **S** | One LLM call per step + drift JSON parser; reuse `core/llm.py` |
| F145 | Skills SDK + MCP packaging | **M** | F145.1 `~/.opencut/skills/` loader (S), F145.2 6 built-in skills (M), F145.3 MCP `skill_compatible` annotation (S) |
| **F143-F145 total** | **flagship** | **~6-8 weeks** at 1 maintainer | Ships as a single v1.36 feature |

---

## 10. Open questions for maintainer review

1. **Default LLM provider:** Ollama by default (local-first), OpenAI/Anthropic opt-in via existing `core/llm.py`. Confirmed.
2. **Self-review LLM:** smaller of the same provider (Llama-3.2:1b for Ollama, GPT-5-mini for OpenAI, Haiku for Anthropic). Reasonable?
3. **Snapshot directory:** `~/.opencut/snapshots/<session_id>/<step_index>.otio`. Retention: 30 days then auto-prune. Reasonable?
4. **UXP vs CEP:** F143 ships UXP-only. CEP fallback only if F252 slips. Confirmed.
5. **MCP exposure:** each accepted step is also callable directly as an MCP tool (skill-aware filtering). The 27 existing tools stay; agent-chat is a 28th. Confirmed.
6. **Skill manifest format:** Claude Code Skills (agentskills.io). Open standard, MIT. Confirmed.
7. **Streaming:** SSE for step-by-step events + WebSocket on `:5680` for OTIO-diff live updates. Reuse existing infrastructure.
8. **Cancellation:** every `execute` step is `@async_job`; the existing cancel infrastructure handles it.

---

## 11. Definition of done for F143-F145

- `POST /agent/chat` accepts intent + returns editable plan (Copilot Workspace pattern)
- User accepts/rejects per step (Cursor pattern, never accept-all)
- Each accepted step writes an OTIO snapshot (Aider pattern, snapshot not auto-commit)
- Post-turn self-review fires automatically and surfaces drift (Underlord 2026 pattern)
- 6 built-in skills ship via `~/.opencut/skills/` matching existing workflow presets
- Skills are MCP-discoverable via `skill_compatible` tag
- UXP panel surfaces the three-layer diff (plan / thumbnails / OTIO XML)
- `tests/test_agent_chat.py` covers: plan-mode parse, execute-mode dispatch, self-review drift, skill loading, snapshot creation
- Doc page: `docs/AGENT_CHAT.md` with skills authoring guide + privacy notes (LLM provider choice opt-in)

---

## 12. Why this is the right time

Pass-1 + Pass-2 + Pass-3 evidence converges:

- **Market signal** (Pass 3 NLE pricing subagent): 22.4% creator skill investment is in video; AI tool adoption +342% YoY; the biggest white-space is "conversational/agentic timeline editing — no shipping product owns it." Descript's transcription-first user base (78%) shows the entry hook; voice/agent features are the upsell.
- **Building blocks ready** (Pass 1+2 audit): 1,359 routes + 27 MCP tools + `core/llm.py` + `core/workflow.py` v2.0 + `journal.py` Operation Journal (one-click rollback) + 6 workflow presets ready to graduate to skills.
- **Standards converging** (Pass 3 IDE-agent subagent): Claude Code Skills, Copilot Workspace plans, Cursor checkpoint+rollback, Underlord self-review — all 2025-2026 patterns. OpenCut can adopt the converged shape rather than invent a new one.
- **UXP migration timing** (Pass 3 CEP/UXP parity): F252 ships UXP scaffold in v1.34-v1.35; F143-F145 lands in v1.36 inside the UXP shell. CEP fallback only as belt-and-suspenders.

**Recommended sequence:** F202 + F236 (regulatory, v1.33) → F252.1+F252.2 (UXP scaffold + JSX port, v1.34) → F121-F128 (dep bumps, v1.34) → F252.3 (HTML/CSS via WebView, v1.35) → **F143+F144+F145 (agent conductor + self-review + skills, v1.36)** → F252.4+F252.5 (CEP fallback + cutover, v1.37).
