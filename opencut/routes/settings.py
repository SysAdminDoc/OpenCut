"""
OpenCut Settings Routes

Presets, favorites, workflows, settings import/export, job time estimation,
log export, job retry.
"""

import json
import os
import time

from flask import Blueprint, jsonify, request, send_file

from opencut import __version__
from opencut.errors import safe_error
from opencut.helpers import OPENCUT_DIR, compute_estimate
from opencut.security import get_json_dict, require_csrf, safe_float, safe_int
from opencut.user_data import (
    load_favorites,
    load_presets,
    load_workflows,
    save_favorites,
    save_presets,
    save_workflows,
)

settings_bp = Blueprint("settings", __name__)


def _invalid_input_response(message: str):
    return jsonify({
        "error": message,
        "code": "INVALID_INPUT",
        "suggestion": "Send a top-level JSON object in the request body.",
    }), 400


def _require_json_object():
    try:
        return get_json_dict(), None
    except ValueError as e:
        return None, _invalid_input_response(str(e))


def _parse_log_line(line: str) -> dict:
    """Parse plain-text and JSON log lines into a filterable record."""
    raw = line.rstrip()
    level = ""
    job_id = ""
    if not raw:
        return {"raw": raw, "level": level, "job_id": job_id}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            level = str(parsed.get("level", "")).upper()
            job_id = str(parsed.get("job_id", ""))
            return {"raw": raw, "level": level, "job_id": job_id}
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    for candidate in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        if f"[{candidate}]" in raw:
            level = candidate
            break
    return {"raw": raw, "level": level, "job_id": job_id}


# ---------------------------------------------------------------------------
# User Presets
# ---------------------------------------------------------------------------
@settings_bp.route("/presets", methods=["GET"])
def list_presets():
    """List all saved user presets."""
    return jsonify(load_presets())


@settings_bp.route("/presets/save", methods=["POST"])
@require_csrf
def save_preset():
    """Save a named preset."""
    data, error = _require_json_object()
    if error:
        return error
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Preset name required"}), 400
    if len(name) > 100:
        return jsonify({"error": "Preset name too long"}), 400
    settings = data.get("settings", {})
    if not isinstance(settings, dict):
        return jsonify({"error": "Settings must be an object"}), 400
    presets = load_presets()
    if name not in presets and len(presets) >= 500:
        return jsonify({"error": "Too many presets (max 500)"}), 400
    presets[name] = {"settings": settings, "saved": time.time()}
    save_presets(presets)
    return jsonify({"success": True, "name": name})


@settings_bp.route("/presets/delete", methods=["POST"])
@require_csrf
def delete_preset():
    """Delete a named preset."""
    data, error = _require_json_object()
    if error:
        return error
    name = data.get("name", "").strip()
    presets = load_presets()
    if name in presets:
        del presets[name]
        save_presets(presets)
    return jsonify({"success": True})


# ---------------------------------------------------------------------------
# Favorites / Pinned Operations
# ---------------------------------------------------------------------------
@settings_bp.route("/favorites", methods=["GET"])
def get_favorites():
    """Get user's favorite/pinned operations."""
    return jsonify(load_favorites())


@settings_bp.route("/favorites/save", methods=["POST"])
@require_csrf
def save_favorites_route():
    """Save user's favorite operations list."""
    data, error = _require_json_object()
    if error:
        return error
    favorites = data.get("favorites", [])
    if not isinstance(favorites, list):
        return jsonify({"error": "favorites must be a list"}), 400
    if len(favorites) > 200:
        return jsonify({"error": "Too many favorites (max 200)"}), 400
    save_favorites(favorites)
    return jsonify({"success": True})


# ---------------------------------------------------------------------------
# Custom Workflow Templates
# ---------------------------------------------------------------------------
@settings_bp.route("/workflows/list", methods=["GET"])
def list_workflows():
    """List saved custom workflow templates."""
    return jsonify(load_workflows())


@settings_bp.route("/workflows/save", methods=["POST"])
@require_csrf
def save_workflow():
    """Save a custom workflow template."""
    data, error = _require_json_object()
    if error:
        return error
    name = data.get("name", "").strip()
    steps = data.get("steps", [])
    if not name:
        return jsonify({"error": "Name required"}), 400
    if len(name) > 100:
        return jsonify({"error": "Workflow name too long"}), 400
    if not steps or not isinstance(steps, list):
        return jsonify({"error": "Steps must be a non-empty list"}), 400
    if len(steps) > 50:
        return jsonify({"error": "Too many workflow steps (max 50)"}), 400
    workflows = load_workflows()
    # Update or add
    found = False
    for wf in workflows:
        if wf.get("name") == name:
            wf["steps"] = steps
            wf["updated"] = time.time()
            found = True
            break
    if not found:
        if len(workflows) >= 100:
            return jsonify({"error": "Too many workflows (max 100)"}), 400
        workflows.append({"name": name, "steps": steps, "created": time.time()})
    save_workflows(workflows)
    return jsonify({"success": True})


@settings_bp.route("/workflows/delete", methods=["POST"])
@require_csrf
def delete_workflow():
    """Delete a custom workflow template."""
    data, error = _require_json_object()
    if error:
        return error
    name = data.get("name", "").strip()
    workflows = load_workflows()
    workflows = [wf for wf in workflows if wf.get("name") != name]
    save_workflows(workflows)
    return jsonify({"success": True})


# ---------------------------------------------------------------------------
# Settings Import/Export
# ---------------------------------------------------------------------------
@settings_bp.route("/settings/export", methods=["GET"])
def export_settings():
    """Export all OpenCut settings (presets, favorites, workflows) as a single JSON bundle."""
    bundle = {"version": __version__, "exported": time.time()}
    try:
        bundle["presets"] = load_presets()
    except Exception:
        bundle["presets"] = {}
    try:
        bundle["favorites"] = load_favorites()
    except Exception:
        bundle["favorites"] = []
    try:
        bundle["workflows"] = load_workflows()
    except Exception:
        bundle["workflows"] = []
    return jsonify(bundle)


@settings_bp.route("/settings/import", methods=["POST"])
@require_csrf
def import_settings():
    """Import settings bundle (presets, favorites, workflows)."""
    data, error = _require_json_object()
    if error:
        return error
    imported = []
    if "presets" in data and isinstance(data["presets"], dict):
        valid_presets = {}
        for name, preset in data["presets"].items():
            if not isinstance(name, str):
                continue
            clean_name = name.strip()
            if not clean_name or len(clean_name) > 100:
                continue
            if not isinstance(preset, dict):
                continue
            valid_presets[clean_name] = preset
        existing = load_presets()
        existing.update(valid_presets)
        if len(existing) > 500:
            return jsonify({"error": "Too many presets (max 500)"}), 400
        save_presets(existing)
        imported.append("presets")
    if "favorites" in data and isinstance(data["favorites"], list):
        favs = data["favorites"][:200]  # Cap at 200 favorites
        save_favorites(favs)
        imported.append("favorites")
    if "workflows" in data and isinstance(data["workflows"], list):
        wfs = data["workflows"][:100]  # Cap at 100 workflows
        # Validate workflow steps before importing
        valid_wfs = []
        for wf in wfs:
            if not isinstance(wf, dict):
                continue
            if not wf.get("name"):
                continue
            steps = wf.get("steps", [])
            if isinstance(steps, list) and all(
                isinstance(s, dict) and s.get("endpoint") and s.get("label")
                for s in steps
            ):
                valid_wfs.append(wf)
        # Merge with existing workflows (update by name, append new ones)
        existing_wfs = load_workflows()
        existing_names = {wf.get("name"): i for i, wf in enumerate(existing_wfs)}
        for wf in valid_wfs:
            name = wf.get("name")
            if name in existing_names:
                existing_wfs[existing_names[name]] = wf
            else:
                existing_wfs.append(wf)
        if len(existing_wfs) > 100:
            existing_wfs = existing_wfs[:100]
        save_workflows(existing_wfs)
        imported.append("workflows")
    return jsonify({"success": True, "imported": imported})


# ---------------------------------------------------------------------------
# Job Time Estimation
# ---------------------------------------------------------------------------
@settings_bp.route("/system/estimate-time", methods=["POST"])
@require_csrf
def estimate_job_time():
    """Estimate processing time based on historical data."""
    data, error = _require_json_object()
    if error:
        return error
    job_type = data.get("type", "")
    file_duration = safe_float(data.get("file_duration", 0))
    return jsonify(compute_estimate(job_type, file_duration))


# ---------------------------------------------------------------------------
# Log Export
# ---------------------------------------------------------------------------
@settings_bp.route("/logs/export", methods=["GET"])
def export_logs():
    """Export the crash log file for debugging."""
    crash_log = os.path.join(OPENCUT_DIR, "crash.log")
    if not os.path.isfile(crash_log):
        return jsonify({"error": "No crash log found"}), 404
    return send_file(crash_log, mimetype="text/plain", as_attachment=True,
                     download_name="opencut_crash.log")


@settings_bp.route("/logs/tail", methods=["GET"])
def tail_logs():
    """Return the last N lines from the server log, optionally filtered.

    Query params:
        lines (int): Number of lines to return (default 100, max 500)
        level (str): Filter by log level (DEBUG, INFO, WARNING, ERROR)
        job_id (str): Filter by job ID
    """
    from opencut.server import LOG_FILE
    lines = min(safe_int(request.args.get("lines", 100), default=100, min_val=1, max_val=500), 500)
    level_filter = request.args.get("level", "").upper()
    job_filter = request.args.get("job_id", "")
    if not os.path.isfile(LOG_FILE):
        return jsonify({"lines": [], "total": 0})
    try:
        # Read only the tail of the file to avoid loading multi-MB logs
        # into memory. When filters are active, read more lines to ensure
        # enough matches survive filtering.
        read_limit = lines * 10 if (level_filter or job_filter) else lines * 2
        read_limit = max(read_limit, 500)
        with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
        # Only process the tail portion to bound memory/CPU
        candidate_lines = all_lines[-read_limit:]
    except OSError:
        return jsonify({"lines": [], "total": 0})
    # Filter
    result = []
    for line in candidate_lines:
        record = _parse_log_line(line)
        if level_filter and record["level"] != level_filter:
            continue
        if job_filter and job_filter not in record["raw"] and job_filter != record["job_id"]:
            continue
        result.append(record["raw"])
    # Tail
    tail = result[-lines:]
    return jsonify({"lines": tail, "total": len(result)})


@settings_bp.route("/logs/clear", methods=["POST"])
@require_csrf
def clear_logs():
    """Clear the crash log and active server log."""
    crash_log = os.path.join(OPENCUT_DIR, "crash.log")
    try:
        from opencut.server import LOG_FILE

        cleared = []
        for path in (crash_log, LOG_FILE):
            if not os.path.isfile(path):
                continue
            with open(path, "w", encoding="utf-8"):
                pass
            cleared.append(os.path.basename(path))
        return jsonify({"success": True, "cleared": cleared})
    except OSError as e:
        return safe_error(e, "clear_logs")


# ---------------------------------------------------------------------------
# Job Retry
# ---------------------------------------------------------------------------
@settings_bp.route("/jobs/retry/<job_id>", methods=["POST"])
@require_csrf
def retry_job(job_id):
    """Retry a failed/cancelled job by re-submitting its original parameters."""
    from opencut.jobs import _get_job_copy

    job = _get_job_copy(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job.get("status") not in ("error", "cancelled"):
        return jsonify({"error": "Only failed or cancelled jobs can be retried"}), 400

    job_type = job.get("type", "")
    filepath = job.get("filepath", "")
    if not job_type or not filepath:
        return jsonify({"error": "Cannot retry: missing job type or filepath"}), 400

    return jsonify({
        "retry_available": True,
        "original_type": job_type,
        "original_filepath": filepath,
        "message": "Use the original endpoint to re-run this operation",
    })


# ---------------------------------------------------------------------------
# LLM Settings
# ---------------------------------------------------------------------------

@settings_bp.route("/settings/llm", methods=["GET"])
def get_llm_settings():
    """Get LLM provider configuration."""
    try:
        from ..user_data import load_llm_settings
    except ImportError:
        from opencut.user_data import load_llm_settings
    settings = load_llm_settings()
    # Never return the API key in full — mask it
    if settings.get("api_key"):
        settings = dict(settings)
        settings["api_key"] = "***" + settings["api_key"][-4:] if len(settings["api_key"]) > 4 else "****"
    return jsonify(settings)


@settings_bp.route("/settings/llm", methods=["POST"])
@require_csrf
def save_llm_settings_route():
    """Save LLM provider configuration."""
    try:
        from ..user_data import load_llm_settings, save_llm_settings
    except ImportError:
        from opencut.user_data import load_llm_settings, save_llm_settings
    data, error = _require_json_object()
    if error:
        return error
    current = load_llm_settings()
    # Don't overwrite key if masked value sent back
    if data.get("api_key", "").startswith("***"):
        data["api_key"] = current.get("api_key", "")
    normalized = {k: v for k, v in data.items() if k in current}
    if "provider" in normalized:
        normalized["provider"] = str(normalized["provider"]).strip()[:50] or current.get("provider", "ollama")
    if "model" in normalized:
        normalized["model"] = str(normalized["model"]).strip()[:100] or current.get("model", "llama3")
    if "base_url" in normalized:
        normalized["base_url"] = str(normalized["base_url"]).strip()[:500]
    if "api_key" in normalized:
        normalized["api_key"] = str(normalized["api_key"]).strip()
    if "max_tokens" in normalized:
        normalized["max_tokens"] = safe_int(
            normalized["max_tokens"],
            default=current.get("max_tokens", 2000),
            min_val=1,
            max_val=32768,
        )
    if "temperature" in normalized:
        normalized["temperature"] = safe_float(
            normalized["temperature"],
            default=current.get("temperature", 0.3),
            min_val=0.0,
            max_val=2.0,
        )
    current.update(normalized)
    save_llm_settings(current)
    return jsonify({"success": True})


# ---------------------------------------------------------------------------
# Footage Index Settings
# ---------------------------------------------------------------------------

@settings_bp.route("/settings/footage-index", methods=["GET"])
def get_footage_index_config():
    try:
        from ..user_data import load_footage_index_config
    except ImportError:
        from opencut.user_data import load_footage_index_config
    return jsonify(load_footage_index_config())


@settings_bp.route("/settings/footage-index", methods=["POST"])
@require_csrf
def save_footage_index_config_route():
    try:
        from ..user_data import load_footage_index_config, save_footage_index_config
    except ImportError:
        from opencut.user_data import load_footage_index_config, save_footage_index_config
    data, error = _require_json_object()
    if error:
        return error
    config = load_footage_index_config()
    config.update({k: v for k, v in data.items() if k in config})
    save_footage_index_config(config)
    return jsonify({"success": True})


# ---------------------------------------------------------------------------
# Loudness Target
# ---------------------------------------------------------------------------

@settings_bp.route("/settings/loudness-target", methods=["GET"])
def get_loudness_target():
    try:
        from ..user_data import load_loudness_target
    except ImportError:
        from opencut.user_data import load_loudness_target
    return jsonify(load_loudness_target())


@settings_bp.route("/settings/loudness-target", methods=["POST"])
@require_csrf
def save_loudness_target_route():
    try:
        from ..user_data import load_loudness_target, save_loudness_target
    except ImportError:
        from opencut.user_data import load_loudness_target, save_loudness_target
    data, error = _require_json_object()
    if error:
        return error
    settings = load_loudness_target()
    settings.update({k: v for k, v in data.items() if k in settings})
    save_loudness_target(settings)
    return jsonify({"success": True})


# ---------------------------------------------------------------------------
# Multicam Defaults
# ---------------------------------------------------------------------------

@settings_bp.route("/settings/multicam", methods=["GET"])
def get_multicam_config():
    try:
        from ..user_data import load_multicam_config
    except ImportError:
        from opencut.user_data import load_multicam_config
    return jsonify(load_multicam_config())


@settings_bp.route("/settings/multicam", methods=["POST"])
@require_csrf
def save_multicam_config_route():
    try:
        from ..user_data import load_multicam_config, save_multicam_config
    except ImportError:
        from opencut.user_data import load_multicam_config, save_multicam_config
    data, error = _require_json_object()
    if error:
        return error
    config = load_multicam_config()
    config.update({k: v for k, v in data.items() if k in config})
    save_multicam_config(config)
    return jsonify({"success": True})


# ---------------------------------------------------------------------------
# Auto Zoom Presets
# ---------------------------------------------------------------------------

@settings_bp.route("/settings/auto-zoom", methods=["GET"])
def get_auto_zoom_presets():
    try:
        from ..user_data import load_auto_zoom_presets
    except ImportError:
        from opencut.user_data import load_auto_zoom_presets
    return jsonify(load_auto_zoom_presets())


@settings_bp.route("/settings/auto-zoom", methods=["POST"])
@require_csrf
def save_auto_zoom_presets_route():
    try:
        from ..user_data import load_auto_zoom_presets, save_auto_zoom_presets
    except ImportError:
        from opencut.user_data import load_auto_zoom_presets, save_auto_zoom_presets
    data, error = _require_json_object()
    if error:
        return error
    presets = load_auto_zoom_presets()
    presets.update({k: v for k, v in data.items() if k in presets})
    save_auto_zoom_presets(presets)
    return jsonify({"success": True})


# ---------------------------------------------------------------------------
# Chapter Generation Defaults
# ---------------------------------------------------------------------------

@settings_bp.route("/settings/chapters", methods=["GET"])
def get_chapter_defaults():
    try:
        from ..user_data import load_chapter_defaults
    except ImportError:
        from opencut.user_data import load_chapter_defaults
    return jsonify(load_chapter_defaults())


@settings_bp.route("/settings/chapters", methods=["POST"])
@require_csrf
def save_chapter_defaults_route():
    try:
        from ..user_data import load_chapter_defaults, save_chapter_defaults
    except ImportError:
        from opencut.user_data import load_chapter_defaults, save_chapter_defaults
    data, error = _require_json_object()
    if error:
        return error
    defaults = load_chapter_defaults()
    defaults.update({k: v for k, v in data.items() if k in defaults})
    save_chapter_defaults(defaults)
    return jsonify({"success": True})


# ---------------------------------------------------------------------------
# Project Templates
# ---------------------------------------------------------------------------

def _load_builtin_templates():
    """Load built-in project templates from the data directory."""
    templates_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "project_templates.json"
    )
    try:
        with open(templates_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Failed to load built-in templates: %s", exc)
        return []


def _load_user_templates():
    """Load user-saved templates from ~/.opencut/templates/."""
    templates_dir = os.path.join(OPENCUT_DIR, "templates")
    templates = []
    if not os.path.isdir(templates_dir):
        return templates
    for fname in os.listdir(templates_dir):
        if not fname.endswith(".json"):
            continue
        try:
            fpath = os.path.join(templates_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                tpl = json.load(f)
            if isinstance(tpl, dict) and tpl.get("id"):
                tpl["user_template"] = True
                templates.append(tpl)
        except Exception:
            continue
    return templates


@settings_bp.route("/templates/list", methods=["GET"])
def list_templates():
    """Return all project templates (built-in + user-saved)."""
    builtin = _load_builtin_templates()
    user = _load_user_templates()
    return jsonify({"builtin": builtin, "user": user})


@settings_bp.route("/templates/save", methods=["POST"])
@require_csrf
def save_template():
    """Save a custom project template."""
    data, error = _require_json_object()
    if error:
        return error
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Template name required"}), 400
    if len(name) > 100:
        return jsonify({"error": "Template name too long"}), 400
    # Build a safe filename
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in name.lower())
    if not safe_id:
        return jsonify({"error": "Invalid template name"}), 400
    templates_dir = os.path.join(OPENCUT_DIR, "templates")
    os.makedirs(templates_dir, exist_ok=True)
    # Limit user templates
    existing = [f for f in os.listdir(templates_dir) if f.endswith(".json")]
    if len(existing) >= 50:
        return jsonify({"error": "Too many user templates (max 50)"}), 400
    tpl = {
        "id": "user_" + safe_id,
        "name": name,
        "description": data.get("description", "Custom template"),
        "export": data.get("export", {}),
        "audio": data.get("audio", {}),
        "captions": data.get("captions", {}),
        "aspect": data.get("aspect", "16:9"),
        "saved": time.time(),
    }
    fpath = os.path.join(templates_dir, safe_id + ".json")
    # Atomic write via temp file + rename to prevent corruption
    import tempfile
    fd, tmp_path = tempfile.mkstemp(dir=templates_dir, suffix=".tmp", prefix=safe_id + ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(tpl, f, indent=2)
        os.replace(tmp_path, fpath)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return jsonify({"success": True, "template": tpl})


@settings_bp.route("/templates/apply", methods=["POST"])
@require_csrf
def apply_template():
    """Apply a template's settings. Returns the settings to apply on the frontend."""
    data, error = _require_json_object()
    if error:
        return error
    template_id = data.get("id", "").strip()
    if not template_id:
        return jsonify({"error": "Template ID required"}), 400
    # Search built-in first, then user templates
    all_templates = _load_builtin_templates() + _load_user_templates()
    tpl = None
    for t in all_templates:
        if t.get("id") == template_id:
            tpl = t
            break
    if not tpl:
        return jsonify({"error": "Template not found"}), 404
    return jsonify({"success": True, "template": tpl})
