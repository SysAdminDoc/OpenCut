"""
OpenCut Wave E Routes (v1.22.0)

Delivery packaging + live-production + cloud-dispatch + telemetry:
- ``POST /delivery/shaka/package``       — Shaka Packager HLS/DASH/CENC
- ``GET  /delivery/shaka/info``          — backend availability + version
- ``POST /integration/obs/status``       — OBS WebSocket status snapshot
- ``POST /integration/obs/switch-scene`` — switch OBS program scene
- ``POST /integration/obs/recording``    — start/stop/toggle recording
- ``POST /integration/obs/screenshot``   — save a scene screenshot
- ``POST /cloud/runpod/submit``          — submit a RunPod serverless job
- ``GET  /cloud/runpod/status/<endpoint>/<job_id>`` — poll a job
- ``POST /cloud/runpod/cancel``          — cancel a pending job
- ``GET  /cloud/runpod/info``            — availability + SDK state
- ``POST /telemetry/plausible/track``    — emit a Plausible event
- ``GET  /telemetry/plausible/info``     — telemetry wiring status
"""

from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request

from opencut.errors import safe_error
from opencut.jobs import _update_job, async_job
from opencut.security import require_csrf, validate_filepath, validate_path

logger = logging.getLogger("opencut")

wave_e_bp = Blueprint("wave_e", __name__)


# ---------------------------------------------------------------------------
# Shaka Packager
# ---------------------------------------------------------------------------

@wave_e_bp.route("/delivery/shaka/info", methods=["GET"])
def route_shaka_info():
    from opencut.core import shaka_pkg
    return jsonify({
        "installed": shaka_pkg.check_shaka_available(),
        "version": shaka_pkg.version(),
        "protocols": list(shaka_pkg.OUTPUT_PROTOCOLS),
        "drm_schemes": list(shaka_pkg.DRM_SCHEMES),
        "hint": (
            "Download the `packager` binary from "
            "https://github.com/shaka-project/shaka-packager/releases "
            "and ensure it's on PATH."
        ),
    })


def _parse_shaka_renditions(raw):
    from opencut.core.shaka_pkg import PackagerRendition
    out = []
    if not isinstance(raw, list):
        return out
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        ip = str(entry.get("input_path") or "").strip()
        kind = str(entry.get("kind") or "video").lower()
        lang = str(entry.get("language") or "und")[:6]
        label = str(entry.get("stream_label") or "")[:60]
        if not ip:
            continue
        try:
            ip = validate_filepath(ip)
        except ValueError:
            continue
        out.append(PackagerRendition(
            input_path=ip, kind=kind, language=lang, stream_label=label,
        ))
    return out


@wave_e_bp.route("/delivery/shaka/package", methods=["POST"])
@require_csrf
@async_job("shaka_package", filepath_required=False)
def route_shaka_package(job_id, filepath, data):
    from opencut.core import shaka_pkg
    from opencut.security import safe_bool, safe_float

    renditions = _parse_shaka_renditions(data.get("renditions"))
    if not renditions:
        raise ValueError("'renditions' must be a non-empty list of {input_path, kind[, language, stream_label]}")

    output_dir = str(data.get("output_dir") or "").strip()
    if not output_dir:
        raise ValueError("'output_dir' is required")
    output_dir = validate_path(output_dir)

    protocol = str(data.get("protocol") or "hls").lower()
    segment_duration = safe_float(
        data.get("segment_duration", 4.0), 4.0, min_val=0.5, max_val=30.0,
    )
    low_latency = safe_bool(data.get("low_latency", False), False)
    drm_scheme = data.get("drm_scheme")
    if drm_scheme is not None and not isinstance(drm_scheme, str):
        drm_scheme = None
    drm_key_hex = data.get("drm_key_hex") if isinstance(data.get("drm_key_hex"), str) else None
    drm_key_id_hex = data.get("drm_key_id_hex") if isinstance(data.get("drm_key_id_hex"), str) else None
    extra_args = data.get("extra_args")
    if not isinstance(extra_args, list):
        extra_args = None
    else:
        extra_args = [str(a)[:80] for a in extra_args[:20] if isinstance(a, (str, int, float))]

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=int(pct), message=str(msg))

    result = shaka_pkg.package(
        renditions=renditions,
        output_dir=output_dir,
        protocol=protocol,
        segment_duration=segment_duration,
        low_latency=low_latency,
        drm_scheme=drm_scheme,
        drm_key_hex=drm_key_hex,
        drm_key_id_hex=drm_key_id_hex,
        extra_args=extra_args,
        on_progress=_on_progress,
    )
    return dict(result)


# ---------------------------------------------------------------------------
# OBS WebSocket bridge
# ---------------------------------------------------------------------------

def _obs_connect_params(data):
    """Normalise OBS connection args from the request JSON."""
    from opencut.security import safe_float, safe_int
    host = str(data.get("host") or "127.0.0.1")[:120]
    port = safe_int(data.get("port", 4455), 4455, min_val=1, max_val=65535)
    password = data.get("password") if isinstance(data.get("password"), str) else None
    timeout = safe_float(data.get("timeout", 5.0), 5.0, min_val=0.5, max_val=30.0)
    return host, port, password, timeout


@wave_e_bp.route("/integration/obs/status", methods=["POST"])
@require_csrf
def route_obs_status():
    try:
        from opencut.core import obs_bridge
        data = request.get_json(silent=True) or {}
        host, port, password, timeout = _obs_connect_params(data)
        status = obs_bridge.status(host, port, password, timeout)
        return jsonify(dict(status))
    except Exception as exc:  # noqa: BLE001
        return safe_error(exc, "obs_status")


@wave_e_bp.route("/integration/obs/switch-scene", methods=["POST"])
@require_csrf
def route_obs_switch_scene():
    try:
        from opencut.core import obs_bridge
        data = request.get_json(force=True) or {}
        scene = str(data.get("scene_name") or "").strip()
        if not scene:
            return jsonify({
                "error": "'scene_name' is required",
                "code": "INVALID_INPUT",
            }), 400
        host, port, password, timeout = _obs_connect_params(data)
        return jsonify(obs_bridge.switch_scene(scene, host, port, password, timeout))
    except ValueError as exc:
        return jsonify({"error": str(exc), "code": "INVALID_INPUT"}), 400
    except Exception as exc:  # noqa: BLE001
        return safe_error(exc, "obs_switch_scene")


@wave_e_bp.route("/integration/obs/recording", methods=["POST"])
@require_csrf
def route_obs_recording():
    try:
        from opencut.core import obs_bridge
        data = request.get_json(force=True) or {}
        action = str(data.get("action") or "status").lower()
        host, port, password, timeout = _obs_connect_params(data)
        return jsonify(obs_bridge.recording(action, host, port, password, timeout))
    except ValueError as exc:
        return jsonify({"error": str(exc), "code": "INVALID_INPUT"}), 400
    except Exception as exc:  # noqa: BLE001
        return safe_error(exc, "obs_recording")


@wave_e_bp.route("/integration/obs/screenshot", methods=["POST"])
@require_csrf
def route_obs_screenshot():
    try:
        from opencut.core import obs_bridge
        from opencut.security import safe_int
        data = request.get_json(force=True) or {}
        scene = data.get("scene_name")
        if scene is not None and not isinstance(scene, str):
            scene = None
        output_path = str(data.get("output_path") or "").strip()
        if not output_path:
            return jsonify({
                "error": "'output_path' is required",
                "code": "INVALID_INPUT",
            }), 400
        output_path = validate_path(output_path)
        host, port, password, timeout = _obs_connect_params(data)
        width = safe_int(data.get("width", 1280), 1280, min_val=64, max_val=7680)
        height = safe_int(data.get("height", 720), 720, min_val=64, max_val=4320)
        saved = obs_bridge.take_screenshot(
            scene_name=scene, output_path=output_path,
            host=host, port=port, password=password,
            width=width, height=height, timeout=timeout,
        )
        return jsonify({"output": saved, "scene": scene or ""})
    except ValueError as exc:
        return jsonify({"error": str(exc), "code": "INVALID_INPUT"}), 400
    except Exception as exc:  # noqa: BLE001
        return safe_error(exc, "obs_screenshot")


# ---------------------------------------------------------------------------
# RunPod
# ---------------------------------------------------------------------------

@wave_e_bp.route("/cloud/runpod/info", methods=["GET"])
def route_runpod_info():
    from opencut.core import runpod_render
    return jsonify({
        "transport_available": runpod_render.check_runpod_available(),
        "sdk_installed": runpod_render.check_runpod_sdk_installed(),
        "env_key_set": bool((__import__("os").environ.get("RUNPOD_API_KEY") or "").strip()),
        "api_base": runpod_render.RUNPOD_API_BASE,
    })


@wave_e_bp.route("/cloud/runpod/submit", methods=["POST"])
@require_csrf
def route_runpod_submit():
    try:
        from opencut.core import runpod_render
        from opencut.security import safe_bool, safe_float
        data = request.get_json(force=True) or {}
        endpoint_id = str(data.get("endpoint_id") or "").strip()
        payload = data.get("payload")
        if not isinstance(payload, dict):
            return jsonify({
                "error": "'payload' must be a JSON object",
                "code": "INVALID_INPUT",
            }), 400
        api_key = data.get("api_key") if isinstance(data.get("api_key"), str) else None
        sync = safe_bool(data.get("sync", False), False)
        sync_timeout = safe_float(data.get("sync_timeout", 300.0), 300.0,
                                   min_val=10.0, max_val=3600.0)
        result = runpod_render.submit(
            endpoint_id=endpoint_id, payload=payload,
            api_key=api_key, sync=sync, sync_timeout=sync_timeout,
        )
        return jsonify(dict(result))
    except ValueError as exc:
        return jsonify({"error": str(exc), "code": "INVALID_INPUT"}), 400
    except Exception as exc:  # noqa: BLE001
        return safe_error(exc, "runpod_submit")


@wave_e_bp.route("/cloud/runpod/status/<endpoint_id>/<job_id>", methods=["GET"])
def route_runpod_status(endpoint_id: str, job_id: str):
    try:
        from opencut.core import runpod_render
        api_key = request.args.get("api_key") or None
        result = runpod_render.status_of(
            endpoint_id=endpoint_id, job_id=job_id, api_key=api_key,
        )
        return jsonify(dict(result))
    except ValueError as exc:
        return jsonify({"error": str(exc), "code": "INVALID_INPUT"}), 400
    except Exception as exc:  # noqa: BLE001
        return safe_error(exc, "runpod_status")


@wave_e_bp.route("/cloud/runpod/cancel", methods=["POST"])
@require_csrf
def route_runpod_cancel():
    try:
        from opencut.core import runpod_render
        data = request.get_json(force=True) or {}
        endpoint_id = str(data.get("endpoint_id") or "").strip()
        job_id = str(data.get("job_id") or "").strip()
        api_key = data.get("api_key") if isinstance(data.get("api_key"), str) else None
        cancelled = runpod_render.cancel(endpoint_id, job_id, api_key=api_key)
        return jsonify({"cancelled": bool(cancelled), "job_id": job_id})
    except ValueError as exc:
        return jsonify({"error": str(exc), "code": "INVALID_INPUT"}), 400
    except Exception as exc:  # noqa: BLE001
        return safe_error(exc, "runpod_cancel")


# ---------------------------------------------------------------------------
# Plausible telemetry
# ---------------------------------------------------------------------------

@wave_e_bp.route("/telemetry/plausible/info", methods=["GET"])
def route_plausible_info():
    from opencut.core import telemetry_plausible
    cfg = telemetry_plausible.get_config()
    return jsonify({
        "enabled": cfg.enabled,
        "host": cfg.host,
        "domain": cfg.domain,
        "user_agent": cfg.user_agent,
        "queue_depth": telemetry_plausible.queue_depth(),
        "hint": (
            "Set PLAUSIBLE_HOST and PLAUSIBLE_DOMAIN env vars "
            "(self-hosted: https://github.com/plausible/analytics)."
        ),
    })


@wave_e_bp.route("/telemetry/plausible/track", methods=["POST"])
@require_csrf
def route_plausible_track():
    try:
        from opencut.core import telemetry_plausible
        data = request.get_json(force=True) or {}
        event_name = str(data.get("event_name") or "").strip()
        if not event_name:
            return jsonify({
                "error": "'event_name' is required",
                "code": "INVALID_INPUT",
            }), 400
        props = data.get("props") if isinstance(data.get("props"), dict) else None
        url_path = str(data.get("url_path") or "/event")[:200]
        # Always fire-and-forget from the route; sync mode is for tests.
        queued = telemetry_plausible.track(
            event_name, props=props, url_path=url_path, sync=False,
        )
        return jsonify({
            "queued": bool(queued),
            "enabled": telemetry_plausible.check_plausible_available(),
            "queue_depth": telemetry_plausible.queue_depth(),
        })
    except Exception as exc:  # noqa: BLE001
        return safe_error(exc, "plausible_track")
