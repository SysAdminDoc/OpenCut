# Audio Description Drafts

OpenCut supports two audio-description stages:

- `/audio/description/microsoft-draft` creates a reviewable draft script from per-scene descriptions, dialogue transcript timing, and silence gaps.
- `/audio/description/generate` renders approved description lines into an output video/audio track.

The draft route follows the authoring shape documented by Microsoft's `ai-audio-descriptions` project: generate scene descriptions, pair them with a dialogue transcript, place rewritten descriptions in silent gaps, send the draft to a human AD editor, then insert the final script with TTS.

Source: https://github.com/microsoft/ai-audio-descriptions

## Draft Route

`POST /audio/description/microsoft-draft`

Request fields:

- `filepath` or `video_path`: optional source video. Required only when `scene_descriptions` are not supplied.
- `scene_descriptions`: optional list of `{timestamp, description, importance}` objects. This can be the output of `/api/ai/scene-describe`.
- `scene_timestamps` or `timestamps`: optional timestamps used when OpenCut must derive scene descriptions from the source video.
- `transcript`: optional transcript list or Whisper-style `{segments: [...]}` / `{words: [...]}` payload with `start`, `end`, `text`, and optional `speaker`.
- `gaps` or `silence_gaps`: optional precomputed dialogue gaps. If omitted, the route derives gaps from transcript timing or FFmpeg silence detection.
- `min_gap_seconds`, `max_gap_seconds`, `context_seconds`, `words_per_second`: placement and rewrite controls.
- `tts_backend_hint`: defaults to `indextts2`, so F168 can render approved lines without changing the draft schema.

Response fields:

- `draft_format`: `opencut.microsoft-ad-draft.v1`.
- `source` and `source_url`: identify the Microsoft-compatible workflow being targeted.
- `cues[*]`: reviewable AD cues with scene timing, target gap, rewritten `script`, original `source_description`, dialogue context, word budget, estimated duration, fit status, review reason, and TTS backend hint.
- `transcript_segments` and `gaps`: normalized timing inputs used for the plan.
- `workflow`: the expected review chain from scene descriptions through TTS insertion.

## Review Boundary

The draft route does not claim to produce a final compliant AD track. It deliberately marks cues as `needs_review` because the Microsoft workflow keeps a human AD editor in the loop before TTS insertion. Use `/audio/description/generate` only after the cue text and placement have been approved or adjusted.
