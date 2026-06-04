# Plugin Authoring

OpenCut plugins live under `~/.opencut/plugins/<plugin-name>/` and are loaded at
server startup. A plugin can ship Flask routes, declare capabilities in
`plugin.json`, and register long-running background work through the standard
OpenCut async job tracker.

## Minimal Manifest

```json
{
  "name": "my-plugin",
  "version": "1.0.0",
  "description": "Adds a custom OpenCut workflow",
  "api_version": 1,
  "capabilities": ["http.routes"],
  "routes": [
    {"method": "GET", "path": "/status", "description": "Read plugin status"}
  ]
}
```

Installed plugins need a matching `plugin.lock.json` unless the operator starts
OpenCut with `OPENCUT_PLUGIN_ALLOW_UNSIGNED=1`.

## Background Jobs

Declare the `jobs.register` capability and list each job ID in `plugin.json`:

```json
{
  "name": "long-job-demo",
  "version": "1.0.0",
  "description": "Demonstrates plugin background jobs",
  "api_version": 1,
  "capabilities": ["http.routes", "jobs.register"],
  "routes": [
    {"method": "POST", "path": "/start", "description": "Start the job"}
  ],
  "jobs": [
    {"id": "render_preview", "label": "Render Preview"}
  ]
}
```

Then wrap a plugin route with `plugin_job(...)`:

```python
from flask import Blueprint
from opencut.core.plugins import plugin_job

plugin_bp = Blueprint("my_plugin", __name__)


@plugin_bp.route("/start", methods=["POST"])
@plugin_job("my-plugin", "render_preview", filepath_required=False)
def start(job_id, filepath, data):
    return {"ok": True, "payload": data}
```

Plugin jobs are normal OpenCut jobs. They use the global concurrency cap, show
up in `/status/<job_id>`, persist to job history, and can opt into resume with
`resumable=True`.

## Filesystem Scope

Plugins without `host.filesystem` can only use path-like job payload fields
inside their own `data/` directory. Declare `host.filesystem` only when the
plugin genuinely needs to read or write host project files.

See `opencut/data/example_plugins/long-job-demo/` for a complete example.
