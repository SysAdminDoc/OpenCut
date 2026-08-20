# OpenCut MCP Server

> **What:** The `opencut-mcp-server` console script speaks the [Model
> Context Protocol](https://modelcontextprotocol.io) so AI clients
> (Claude Code, Cursor, Continue, Aider, etc.) can drive OpenCut's
> 88 curated MCP tools and 1,474 generated route-level tools without
> writing one-off HTTP shims.
>
> **Status:** Shipped since v1.30.0. 88 curated tools cover the most
> common silence/filler/captions/highlights/export pipelines plus
> the Pass-2 expansion (Brand Kit, semantic search, marker import,
> review bundles, C2PA provenance, ElevenLabs TTS, caption QC,
> spectral match, capability probe, face reshape, skin retouch,
> smart upscale). F194 adds an opt-in generated extended catalogue for
> clients that deliberately want route-level access beyond the curated set.
>
> **Tracking F-number:** **F147** — registration in the upstream
> `modelcontextprotocol/servers` directory.

## Positioning after Premiere 26.x

Premiere now provides first-party active-sequence indexing, Single-Word
Captions, transcript pause/filler deletion, Media Intelligence search,
loudness matching, and bulk bleep/mute. OpenCut's MCP and REST surfaces are
not marketed as replacements for those host controls. The differentiated
value is whole-library and cross-project scope, reviewable proposal ranges,
exportable artifacts, unlimited local runs, and headless automation from a
script or MCP client.

---

## 1. Quick start

```bash
# 1. Install OpenCut with MCP extras.
pip install "opencut[mcp]"

# 2. Start the local backend.
opencut-server &

# 3. Start the MCP server (stdio JSON-RPC by default).
opencut-mcp-server
```

The MCP server proxies every tool call through the local Flask
backend on `127.0.0.1:5679`. It is **single-user, loopback-only by
design** — no network exposure, no cloud, no auth tokens required
for local use. (For non-loopback binds use `OPENCUT_ALLOW_REMOTE=1`
plus the F112 auth token; the MCP server reads the same token.)

## 2. Transports

| Mode | Command | Use case |
|---|---|---|
| stdio JSON-RPC | `opencut-mcp-server` (default) | Claude Code, Cursor, Continue, Aider — every MCP client that spawns a subprocess. |
| HTTP JSON-RPC | `opencut-mcp-server --http` | Remote MCP clients that connect by URL. Binds `127.0.0.1:5681` only. |
| Discovery | `opencut-mcp-server --list-tools` | Dump the curated `MCP_TOOLS` array as JSON — useful for client install screens. |
| Extended discovery | `OPENCUT_MCP_EXTENDED_TOOLS=1 opencut-mcp-server --list-tools` or `opencut-mcp-server --extended-tools --list-tools` | Include generated lower-priority `opencut_route_*` tools. |

## 2.1 Protocol revisions

OpenCut serves both sides of the 2026-07-28 boundary from one dispatcher, so
the stdio and HTTP transports cannot drift apart.

| Revision | How a client opens | What results look like |
|---|---|---|
| `2026-07-28` | Call `server/discover`, or send nothing and state the version per request in `_meta` under `io.modelcontextprotocol/protocolVersion`. No `initialize`, no session id. | Every result carries `resultType: "complete"` and `_meta` with `io.modelcontextprotocol/serverInfo`. `tools/list`, `prompts/list`, `resources/list`, and `resources/templates/list` also carry `ttlMs` and `cacheScope`. |
| `2025-11-25` … `2024-11-05` | The `initialize` / `notifications/initialized` handshake, still accepted. | Unchanged — the newer required fields are deliberately omitted so an older client's schema still validates the response. |

- `server/discover` returns `protocolVersions`, `capabilities`, `serverInfo`,
  and `schemaDialect` (JSON Schema 2020-12).
- A request naming an unsupported version is rejected with
  `-32022` (`UnsupportedProtocolVersion`) and the supported list, rather than
  answered on a guess.
- OpenTelemetry `traceparent` / `tracestate` / `baggage` values supplied in a
  request's `_meta` are echoed back in the result's `_meta`.
- The `io.modelcontextprotocol/tasks` extension is backed by the durable
  backend job store. A client that declares the extension can receive a
  `resultType: "task"` handle from a long-running `tools/call`; the handle's
  `taskId` is the existing OpenCut `job_id` and remains pollable after the MCP
  sidecar reconnects. `tasks/get`, `tasks/update`, and `tasks/cancel` map to
  persisted job state and cooperative backend cancellation.
- **Not claimed, because not implemented:** `subscriptions/listen`, sampling,
  and roots. Calling them returns `-32601` rather than a half-working stub.
- The `mcp` SDK is not imported by OpenCut — the server speaks JSON-RPC
  directly — so both the 1.x and 2.x SDK lines work as client-side tooling.

## 3. Tool catalogue (88 tools)

The full schema for each tool lives in
[`opencut/mcp_server.py`](../opencut/mcp_server.py)
under `MCP_TOOLS`. Categories:

- **Cut & clean** — silence detect/remove, filler removal, repeat
  detection, auto-edit, scenes.
- **Captions** — Whisper transcribe, caption QC, chapters, edited
  transcript export, translation.
- **Audio** — denoise, separation, normalize, music gen, TTS,
  spectral match.
- **Video** — export, trim, merge, color match, auto zoom, multicam,
  highlights, semantic search.
- **Production** — review bundles, Brand Kit, marker import,
  capability probe, face reshape, skin retouch, smart upscale, C2PA
  provenance, ElevenLabs TTS.

Every curated tool returns either a synchronous `result` dict or a `job_id`.
Clients without Tasks support can poll that ID via `opencut_job_status`. Clients
that declare `io.modelcontextprotocol/tasks` receive a standard task handle for
eligible long-running calls and should use `tasks/get` instead. The list is curated:
core editing routes are exposed; install / settings / housekeeping
routes are deliberately left to the HTTP REST surface so MCP clients
can't accidentally reconfigure the backend.

### MCP Tasks compatibility

Tasks are created only after the backend confirms the returned job is durable.
The adapter maps `running` to `working`, `complete` to `completed`, `error` or
`interrupted` to `failed`, and `cancelled` to `cancelled`. OpenCut does not
currently create `input_required` jobs, so `tasks/update` is an empty,
read-validated acknowledgement. Clients that do not declare the extension keep
the pre-existing text-wrapped job response and `opencut_job_status` workflow.

### MCP Apps review/progress surface

Clients that advertise the `io.modelcontextprotocol/ui` extension with the
`text/html;profile=mcp-app` MIME type receive a versioned
`ui://opencut/review-progress/v1/index.html` resource. `tools/list` links the
resource only for the job-status, review-bundle, federated-search, and review
action tools; clients without that capability continue to receive the normal
text result and an empty `resources/list` response.

The bundled view has no network, frame, or browser permissions. Tool results
are copied into `structuredContent` after local-path redaction, and its
buttons can call only `opencut_review_action` for refresh, cancellation, or
approval decisions. `resources/read` rejects the URI unless the client
advertised the Apps capability, and unknown resource URIs are rejected.

### Extended route catalogue (F194)

`opencut/_generated/mcp_extended_tools.json` is generated from
`opencut/_generated/route_manifest.json` plus the OpenAPI response-schema
map. It exposes lower-priority `opencut_route_*` tools for route-level
coverage that the curated catalogue does not attempt to hand-design.

The extended catalogue is disabled by default. Enable it only for clients
that can handle a large, route-shaped tool surface:

```bash
OPENCUT_MCP_EXTENDED_TOOLS=1 opencut-mcp-server
# or
opencut-mcp-server --extended-tools
```

Generated tools are tagged with `metadata.generated=true` and
`metadata.priority="extended"`. Path parameters are top-level arguments;
GET routes accept an optional `query` object; mutating routes accept an
optional `body` object. The curated tools remain the preferred interface
for common workflows.

## 4. Registry-friendly manifest

`opencut/_generated/mcp_server_registry.json` is the
machine-readable manifest the MCP upstream registry pulls in. It is
**generated from the live tool catalogue** so it cannot drift:

```bash
python -m opencut.tools.dump_mcp_registry_manifest
python -m opencut.tools.dump_mcp_extended_tools
```

The same tools run in release smoke (`mcp-registry` plus the F194
extended-tool test) and fail closed if committed manifests disagree with
the live catalogues.

Fields the manifest captures:

| Field | Source |
|---|---|
| `name` | Always `opencut-mcp-server`. |
| `version` | `opencut.__version__`. |
| `description` | First paragraph of `opencut/mcp_server.py` docstring. |
| `homepage` | `https://github.com/SysAdminDoc/OpenCut`. |
| `repository` | Same, for the upstream registry's `Source` link. |
| `install` | `pip install "opencut[mcp]"` + run-command stanza. |
| `transport` | `["stdio", "http"]`. |
| `tools` | One entry per curated tool with name + description. |
| `license` | `MIT` from `pyproject.toml`. |

The extended manifest is separate on purpose: upstream registries and
most users should see the 88 curated tools by default, while local
power users can opt into the generated 1,474 route-level set.

## 5. Registering with `modelcontextprotocol/servers`

The upstream registry lives at
<https://github.com/modelcontextprotocol/servers>. The maintainer
process for F147 is:

1. Run `python -m opencut.tools.dump_mcp_registry_manifest` to make
   sure the committed manifest is fresh.
2. Fork `modelcontextprotocol/servers` and open a PR adding an entry
   to the `Community Servers` table (Adobe Premiere / video editing
   category).
3. Use the language from this file's §1 (Quick start) as the upstream
   description so the install snippet stays in sync between the two
   repos.
4. Reference `opencut/_generated/mcp_server_registry.json` in the
   PR description so reviewers can verify the tool catalogue without
   cloning OpenCut.

The upstream PR is the only step that requires GitHub credentials
for `modelcontextprotocol/servers`; everything inside OpenCut is
automated by the dump tool + release-smoke step.

## 6. Client configuration snippets

### Claude Code

```json
{
  "mcpServers": {
    "opencut": {
      "command": "opencut-mcp-server",
      "args": []
    }
  }
}
```

### Cursor

```json
{
  "mcp.servers": {
    "opencut": {
      "command": "opencut-mcp-server",
      "args": []
    }
  }
}
```

### Custom HTTP client

```json
{
  "url": "http://127.0.0.1:5681",
  "transport": "http"
}
```

Both stdio clients should set `cwd` to the user's project folder
so file paths in tool calls resolve correctly.

## 7. Acceptance criteria for closing F147

F147 is closed when:

1. `docs/MCP_SERVER.md` exists. ✅
2. `python -m opencut.tools.dump_mcp_registry_manifest` regenerates a
   manifest under `opencut/_generated/`. ✅
3. Release smoke verifies the committed manifest matches the live
   tool catalogue (`mcp-registry` step). ✅
4. The upstream PR against `modelcontextprotocol/servers` is filed
   and merged. ☐ — pending GitHub credentials.

Item 4 requires a credentialed push to a third-party repo and is
tracked as the only remaining external action for F147.

## 8. References

- `opencut/mcp_server.py` — full implementation, 1,160+ lines, 88
  curated tools, JSON-RPC 2.0 over stdio + HTTP.
- `opencut/_generated/mcp_server_registry.json` — registry manifest
  this doc points at.
- `opencut/_generated/mcp_extended_tools.json` — opt-in generated
  route-level MCP catalogue.
- `opencut/tools/dump_mcp_registry_manifest.py` — generator + check
  runner.
- `opencut/tools/dump_mcp_extended_tools.py` — F194 extended-catalogue
  generator + check runner.
- `tests/test_mcp_registry_manifest.py` — committed-vs-live guard.
- `tests/test_mcp_extended_tools.py` — generated extended-catalogue
  guard and opt-in dispatch coverage.

## Agent skill

<!-- agent-skill:start -->

Generated from `opencut/_generated/mcp_server_registry.json` (88 tools). Regenerate with `python -m opencut.tools.dump_mcp_agent_skill`.

### Conventions

**A tool may hand back a job instead of a result**

Every tool returns either a synchronous `result` object or a `job_id`. Poll `opencut_job_status` until the job reports `complete`, `error`, `interrupted`, or `cancelled`. Clients that declare the `io.modelcontextprotocol/tasks` extension get a task handle instead and should use `tasks/get`.

Ignoring it: Treating the job acknowledgement as the finished result reports success for work that has not run yet, and the output file will not exist.

**Propose edits, then apply them**

Detection tools return ranges; they do not touch the timeline. Build a `opencut_review_bundle`, let a human accept or reject, and apply the outcome with `opencut_review_action`. Cut application offers a non-destructive mode that disables clips rather than deleting them.

Ignoring it: Applying detected ranges straight to a sequence deletes a user's media on the strength of a heuristic, which is the failure editors distrust these tools for.

**The catalogue is curated on purpose**

Install, settings, and housekeeping routes are deliberately absent so an MCP client cannot reconfigure the backend. The route-shaped extended catalogue is off unless `OPENCUT_MCP_EXTENDED_TOOLS=1` is set.

Ignoring it: Reaching for a missing capability through the REST surface bypasses the boundary that keeps an agent from changing the user's install.

**Everything runs locally against real paths**

Tools take filesystem paths on the machine running the backend and write output beside the input unless told otherwise. There is no upload step and no cloud key. Confirm a path exists before starting long work.

Ignoring it: A path that only exists on the client produces a job that fails minutes in, after the user has waited for it.

### Transcribe, review the cuts, export

1. `opencut_transcribe` — Returns a job. Poll opencut_job_status for the transcript.
2. `opencut_silence_remove` — Detects removable ranges. Nothing is applied yet.
3. `opencut_review_bundle` — Packages the proposed ranges for a human decision.
4. `opencut_review_action` — Applies the accepted ranges, or disables rather than deletes.
5. `opencut_export_video` — Renders the result. Returns a job; poll it to completion.

### Tool families

**Cut and clean** (9): `opencut_auto_zoom`, `opencut_filler_remove`, `opencut_repeat_detect`, `opencut_scene_detect`, `opencut_silence_remove`, `opencut_speed_change`, `opencut_speed_ramp`, `opencut_timeline_beat_cut`, `opencut_trim_video`

**Captions and transcript** (10): `opencut_adr_list`, `opencut_caption_animated`, `opencut_caption_burnin`, `opencut_caption_karaoke`, `opencut_caption_qc`, `opencut_caption_srt_import`, `opencut_caption_styled`, `opencut_caption_translate`, `opencut_chapters`, `opencut_transcribe`

**Audio** (15): `opencut_audio_duck`, `opencut_audio_effects`, `opencut_audio_enhance`, `opencut_audio_isolate`, `opencut_audio_normalize`, `opencut_beat_markers`, `opencut_denoise_audio`, `opencut_denoise_video`, `opencut_elevenlabs_tts`, `opencut_generate_music`, `opencut_loudness_match`, `opencut_music_cue_sheet`, `opencut_separate_audio`, `opencut_spectral_match`, `opencut_tts`

**Video and render** (27): `opencut_batch_export`, `opencut_blend_videos`, `opencut_chromakey`, `opencut_color_match`, `opencut_concat_videos`, `opencut_depth_map`, `opencut_dub_video`, `opencut_export_video`, `opencut_highlights`, `opencut_interpolate`, `opencut_letterbox`, `opencut_lut_apply`, `opencut_lut_generate`, `opencut_merge_videos`, `opencut_multicam_cuts`, `opencut_preview_frame`, `opencut_reframe_video`, `opencut_shorts_pipeline`, `opencut_smart_upscale`, `opencut_social_upload`, `opencut_sports_highlights`, `opencut_stabilize_video`, `opencut_style_transfer`, `opencut_transitions`, `opencut_upscale`, `opencut_vfx_sheet`, `opencut_video_fx`

**Face and retouch** (4): `opencut_face_enhance`, `opencut_face_reshape`, `opencut_lipsync_echomimic`, `opencut_skin_retouch`

**Footage and ingest** (5): `opencut_footage_search`, `opencut_index_footage`, `opencut_ingest_url`, `opencut_timeline_batch_rename`, `opencut_timeline_smart_bins`

**Review and provenance** (5): `opencut_brand_kit`, `opencut_c2pa_provenance`, `opencut_marker_import`, `opencut_review_action`, `opencut_review_bundle`

**Capability and jobs** (13): `opencut_capability_probe`, `opencut_chat_edit`, `opencut_dependencies`, `opencut_feature_state`, `opencut_federated_search`, `opencut_gpu_status`, `opencut_job_status`, `opencut_nlp_command`, `opencut_pip`, `opencut_semantic_search`, `opencut_system_info`, `opencut_workflow_presets`, `opencut_workflow_run`

<!-- agent-skill:end -->
