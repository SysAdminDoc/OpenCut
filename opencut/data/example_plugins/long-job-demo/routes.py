"""Example plugin that registers a real OpenCut async job."""

from flask import Blueprint

from opencut.core.plugins import plugin_job
from opencut.jobs import _update_job

plugin_bp = Blueprint("long_job_demo_plugin", __name__)


@plugin_bp.route("/start", methods=["POST"])
@plugin_job(
    "long-job-demo",
    "render_preview",
    label="Render Preview",
    description="Runs a short multi-step demo task through the standard async job tracker.",
    filepath_required=False,
    resumable=True,
)
def start(job_id, filepath, data):
    steps = data.get("steps", 3)
    try:
        steps = int(steps)
    except (TypeError, ValueError):
        steps = 3
    steps = max(1, min(5, steps))

    for step in range(steps):
        _update_job(
            job_id,
            progress=int(((step + 1) / steps) * 90),
            message=f"Demo step {step + 1}/{steps}",
        )

    return {
        "plugin": "long-job-demo",
        "job": "render_preview",
        "steps": steps,
        "message": "Demo plugin job completed.",
    }
