# OpenCut MCP Server

> **What:** The `opencut-mcp-server` console script speaks the [Model
> Context Protocol](https://modelcontextprotocol.io) so AI clients
> (Claude Code, Cursor, Continue, Aider, etc.) can drive OpenCut's
> 86 curated MCP tools and 1,450 generated route-level tools without
> writing one-off HTTP shims.
>
> **Status:** Shipped since v1.30.0. 86 curated tools cover the most
> common silence/filler/captions/highlights/export pipelines plus
> the Pass-2 expansion (Brand Kit, semantic search, marker import,
> review bundles, C2PA provenance, ElevenLabs TTS, caption QC,
> spectral match, capability probe, face reshape, skin retouch,
> smart upscale). F194 adds an opt-in generated extended catalogue for
> clients that deliberately want route-level access beyond the curated set.
>
> **Tracking F-number:** **F147** — registration in the upstream
> `modelcontextprotocol/servers` directory.

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

## 3. Tool catalogue (86 tools)

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

Every curated tool returns either a synchronous `result` dict or a `job_id`
the client can poll via `opencut_job_status`. The list is curated:
core editing routes are exposed; install / settings / housekeeping
routes are deliberately left to the HTTP REST surface so MCP clients
can't accidentally reconfigure the backend.

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
most users should see the 86 curated tools by default, while local
power users can opt into the generated 1,450 route-level set.

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

- `opencut/mcp_server.py` — full implementation, 1,160+ lines, 86
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
- `.ai/research/2026-05-17/FEATURE_BACKLOG.md` — F147 entry (C.
  Agentic / chat / MCP).
