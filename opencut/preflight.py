"""Pipeline preflight checks (v1.9.33 — feature G).

Before committing to a long-running pipeline, probe the machine for the
conditions that would cause it to fail mid-run: missing deps, missing
file, insufficient disk space, unreadable media. Returns a structured
report the panel can render as a "are you sure?" modal.

Keep preflight *cheap* — no Whisper load, no probe with ffmpeg, just
quick filesystem + module-import checks. 100ms budget.
"""

from __future__ import annotations

import os
import shutil
from typing import List


PIPELINES = {
    "interview-polish": {
        "label": "Interview Polish",
        "checks": [
            ("whisper",        True),
            ("ffmpeg",         True),
            ("disk_space_mb",  500),  # needs room for XML + SRT + chapter + temp
        ],
        "soft_checks": [
            ("repeat_detect", "needs Whisper"),
            ("chapters",      "uses Whisper transcript + optional LLM"),
        ],
    },
    "full": {
        "label": "Full Pipeline",
        "checks": [
            ("ffmpeg",        True),
            ("disk_space_mb", 400),
        ],
        "soft_checks": [
            ("whisper",       "required for captions + filler cut"),
        ],
    },
    "shorts-pipeline": {
        "label": "Shorts Pipeline",
        "checks": [
            ("whisper",        True),
            ("ffmpeg",         True),
            ("disk_space_mb",  1500),  # outputs multiple clip files
        ],
        "soft_checks": [
            ("llm",           "used to rank highlights"),
            ("mediapipe",     "used for face-tracked reframe"),
        ],
    },
}


def _check_name(key: str) -> str:
    return {
        "whisper":    "Whisper (transcription)",
        "ffmpeg":     "FFmpeg",
        "repeat_detect": "Repeated-take detection",
        "chapters":   "Chapter generation",
        "llm":        "LLM provider (Ollama / API key)",
        "mediapipe":  "MediaPipe (face tracking)",
    }.get(key, key)


def _probe_check(key: str, required) -> dict:
    """Run one check. Returns ``{ok, reason?, fix?}``."""
    from opencut import checks as c
    from opencut.helpers import _try_import

    if key == "whisper":
        ok = _try_import("faster_whisper") is not None or _try_import("whisper") is not None
        return {"ok": ok, "label": _check_name(key),
                "fix": "Install with 'pip install faster-whisper' or from the Settings → Dependencies tab."
                       if not ok else ""}
    if key == "ffmpeg":
        ok = shutil.which("ffmpeg") is not None or bool(os.environ.get("OPENCUT_FFMPEG_PATH"))
        return {"ok": ok, "label": _check_name(key),
                "fix": "Install FFmpeg or rely on the bundled binary in the installer."
                       if not ok else ""}
    if key == "repeat_detect":
        ok = _try_import("faster_whisper") is not None or _try_import("whisper") is not None
        return {"ok": ok, "label": _check_name(key),
                "fix": "Whisper must be installed — repeat-detect uses the transcript."
                       if not ok else ""}
    if key == "chapters":
        ok = _try_import("faster_whisper") is not None or _try_import("whisper") is not None
        return {"ok": ok, "label": _check_name(key),
                "fix": "Needs Whisper for the transcript. LLM optional — heuristic fallback works."
                       if not ok else ""}
    if key == "llm":
        ok = c.check_llm_available()
        return {"ok": ok, "label": _check_name(key),
                "fix": "Start Ollama locally or set an API key in Settings → LLM."
                       if not ok else ""}
    if key == "mediapipe":
        ok = c.check_mediapipe_available()
        return {"ok": ok, "label": _check_name(key),
                "fix": "Install with 'pip install mediapipe'." if not ok else ""}
    # Fallback for unknown keys — never block on unrecognised checks.
    return {"ok": True, "label": key, "fix": ""}


def _disk_space_ok(output_dir: str, required_mb: int) -> dict:
    """Check that *output_dir*'s filesystem has *required_mb* free."""
    try:
        probe_dir = output_dir or os.path.expanduser("~")
        if not os.path.isdir(probe_dir):
            probe_dir = os.path.dirname(probe_dir) or os.path.expanduser("~")
        total, used, free = shutil.disk_usage(probe_dir)
        free_mb = int(free / (1024 * 1024))
        return {
            "ok": free_mb >= required_mb,
            "label": "Disk space at output",
            "detail": f"{free_mb} MB free (need ≥{required_mb} MB)",
            "fix": f"Free up disk space at {probe_dir} before running."
                   if free_mb < required_mb else "",
        }
    except Exception as e:
        return {"ok": True, "label": "Disk space at output",
                "detail": f"Could not probe: {e}", "fix": ""}


def run_preflight(pipeline: str, filepath: str = "", output_dir: str = "") -> dict:
    """Run the preflight checklist for *pipeline*. Returns a dict of
    ``{pipeline, file_ok, blocking: [...], warnings: [...], pass: bool}``.
    """
    if pipeline not in PIPELINES:
        return {"error": f"Unknown pipeline: {pipeline}",
                "valid": sorted(PIPELINES)}

    spec = PIPELINES[pipeline]
    blocking: List[dict] = []
    warnings: List[dict] = []

    # File check — not in the declarative list because it's argument-driven.
    file_ok = True
    file_detail = ""
    if filepath:
        if not os.path.isfile(filepath):
            file_ok = False
            file_detail = "File not found."
            blocking.append({"label": "Input file",
                             "ok": False,
                             "detail": file_detail,
                             "fix": "Verify the path and try again."})
        else:
            try:
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                file_detail = f"{size_mb:.1f} MB"
            except OSError:
                file_detail = "readable"

    for key, required in spec["checks"]:
        if key == "disk_space_mb":
            result = _disk_space_ok(output_dir or os.path.dirname(filepath), required)
        else:
            result = _probe_check(key, required)
        if not result["ok"]:
            blocking.append(result)

    for key, note in spec["soft_checks"]:
        result = _probe_check(key, True)
        if not result["ok"]:
            warnings.append({
                "label": result["label"],
                "ok": False,
                "detail": note,
                "fix": result.get("fix", ""),
            })

    return {
        "pipeline": pipeline,
        "pipeline_label": spec["label"],
        "file": {"path": filepath, "ok": file_ok, "detail": file_detail},
        "blocking": blocking,
        "warnings": warnings,
        "pass": (file_ok and not blocking),
    }
