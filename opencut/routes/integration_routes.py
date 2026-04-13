"""
OpenCut Integration Routes

Blueprint: integration_bp

Endpoints for all 11 integration features:
  30.3 - Adjustment Layers
  30.4 - Nested Sequence Detection
  41.2 - Flight Path Map
  41.3 - Aerial Hyperlapse
  50.2 - Notion/PM Sync
  50.3 - Slack/Discord Notifications
  50.4 - Zapier/Make Webhooks
  39.1 - Stream Deck Integration
  39.2 - MIDI Controller
  39.3 - Shuttle/Jog Wheel
  39.4 - Touch/Pen Optimization
"""

import logging
import os
import tempfile

from flask import Blueprint, jsonify, request

from opencut.errors import safe_error
from opencut.jobs import _update_job, async_job
from opencut.security import (
    require_csrf,
    safe_bool,
    safe_float,
    safe_int,
    validate_filepath,
    validate_path,
)

logger = logging.getLogger("opencut")

integration_bp = Blueprint("integration", __name__)


# ===================================================================
# 30.3 — Adjustment Layers
# ===================================================================

@integration_bp.route("/api/adjustment-layers/create", methods=["POST"])
@require_csrf
def create_adjustment_layer_route():
    """Create an adjustment layer clip from a correction stack."""
    try:
        from opencut.core.adjustment_layers import create_adjustment_layer

        data = request.get_json(silent=True) or {}
        corrections = data.get("corrections", [])
        if not corrections:
            return jsonify({"error": "corrections list is required"}), 400

        duration = safe_float(data.get("duration", 10.0), 10.0, min_val=0.1, max_val=3600.0)
        width = safe_int(data.get("width", 1920), 1920, min_val=320, max_val=7680)
        height = safe_int(data.get("height", 1080), 1080, min_val=240, max_val=4320)
        fps = safe_float(data.get("fps", 30.0), 30.0, min_val=1.0, max_val=120.0)

        out = data.get("output_path")
        if out:
            out = validate_path(out)
        else:
            out = os.path.join(tempfile.gettempdir(), "opencut_adj_layer.mp4")

        result = create_adjustment_layer(
            corrections=corrections,
            duration=duration,
            out_path=out,
            width=width,
            height=height,
            fps=fps,
        )
        return jsonify({
            "output_path": result.output_path,
            "corrections_applied": result.corrections_applied,
            "duration": result.duration,
            "filter_chain": result.filter_chain,
        })
    except Exception as exc:
        return safe_error(exc, "create_adjustment_layer")


@integration_bp.route("/api/adjustment-layers/apply", methods=["POST"])
@require_csrf
@async_job("adjustment_apply")
def apply_adjustment_route(job_id, filepath, data):
    """Apply an adjustment layer to a video range."""
    from opencut.core.adjustment_layers import apply_adjustment_to_range

    layer_path = data.get("layer_path", "")
    if not layer_path:
        raise ValueError("layer_path is required")
    validate_filepath(layer_path)

    start = safe_float(data.get("start", 0.0), 0.0, min_val=0.0)
    end = safe_float(data.get("end", 10.0), 10.0, min_val=0.0)
    opacity = safe_float(data.get("opacity", 1.0), 1.0, min_val=0.0, max_val=1.0)

    out = data.get("output_path", "")

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = apply_adjustment_to_range(
        video_path=filepath,
        layer_path=layer_path,
        start=start,
        end=end,
        out_path=out,
        opacity=opacity,
        on_progress=_p,
    )
    return {
        "output_path": result.output_path,
        "corrections_applied": result.corrections_applied,
        "duration": result.duration,
    }


@integration_bp.route("/api/adjustment-layers/presets", methods=["GET"])
def list_presets_route():
    """List saved adjustment presets."""
    try:
        from opencut.core.adjustment_layers import list_adjustment_presets
        presets = list_adjustment_presets()
        return jsonify({"presets": presets})
    except Exception as exc:
        return safe_error(exc, "list_adjustment_presets")


@integration_bp.route("/api/adjustment-layers/presets", methods=["POST"])
@require_csrf
def save_preset_route():
    """Save an adjustment preset."""
    try:
        from opencut.core.adjustment_layers import save_adjustment_preset

        data = request.get_json(silent=True) or {}
        corrections = data.get("corrections", [])
        name = str(data.get("name", "")).strip()
        if not name:
            return jsonify({"error": "name is required"}), 400

        result = save_adjustment_preset(
            corrections=corrections,
            name=name,
            description=str(data.get("description", "")),
        )
        return jsonify(result)
    except Exception as exc:
        return safe_error(exc, "save_adjustment_preset")


# ===================================================================
# 30.4 — Nested Sequence Detection
# ===================================================================

@integration_bp.route("/api/timeline/detect-patterns", methods=["POST"])
@require_csrf
def detect_patterns_route():
    """Detect repeated patterns in timeline items."""
    try:
        from opencut.core.nested_sequence import detect_repeated_patterns

        data = request.get_json(silent=True) or {}
        items = data.get("timeline_items", [])
        if not items:
            return jsonify({"error": "timeline_items is required"}), 400

        min_length = safe_int(data.get("min_length", 2), 2, min_val=2, max_val=50)
        min_occ = safe_int(data.get("min_occurrences", 2), 2, min_val=2, max_val=100)

        patterns = detect_repeated_patterns(
            timeline_items=items,
            min_length=min_length,
            min_occurrences=min_occ,
        )
        return jsonify({
            "patterns": [p.to_dict() for p in patterns],
            "count": len(patterns),
        })
    except Exception as exc:
        return safe_error(exc, "detect_patterns")


@integration_bp.route("/api/timeline/create-nested", methods=["POST"])
@require_csrf
def create_nested_route():
    """Create a nested sequence from a detected pattern."""
    try:
        from opencut.core.nested_sequence import (
            PatternMatch,
            create_nested_sequence,
        )

        data = request.get_json(silent=True) or {}
        pattern_data = data.get("pattern")
        if not pattern_data:
            return jsonify({"error": "pattern is required"}), 400

        pattern = PatternMatch(
            pattern_id=pattern_data.get("pattern_id", ""),
            items=pattern_data.get("items", []),
            length=pattern_data.get("length", 0),
            occurrences=pattern_data.get("occurrences", 0),
            positions=pattern_data.get("positions", []),
        )

        name = str(data.get("name", ""))

        seq = create_nested_sequence(pattern=pattern, name=name)
        return jsonify(seq.to_dict())
    except Exception as exc:
        return safe_error(exc, "create_nested")


@integration_bp.route("/api/timeline/replace-nested", methods=["POST"])
@require_csrf
def replace_nested_route():
    """Replace pattern occurrences with nested sequence references."""
    try:
        from opencut.core.nested_sequence import (
            NestedSequence,
            PatternMatch,
            replace_with_nested,
        )

        data = request.get_json(silent=True) or {}
        items = data.get("timeline_items", [])
        pattern_data = data.get("pattern", {})
        nested_data = data.get("nested_ref", {})

        if not items or not pattern_data:
            return jsonify({"error": "timeline_items and pattern are required"}), 400

        pattern = PatternMatch(
            pattern_id=pattern_data.get("pattern_id", ""),
            items=pattern_data.get("items", []),
            length=pattern_data.get("length", 0),
            occurrences=pattern_data.get("occurrences", 0),
            positions=pattern_data.get("positions", []),
        )

        nested_ref = NestedSequence(
            sequence_id=nested_data.get("sequence_id", ""),
            name=nested_data.get("name", ""),
            items=nested_data.get("items", []),
            duration=nested_data.get("duration", 0),
            item_count=nested_data.get("item_count", 0),
        )

        result = replace_with_nested(items, pattern, nested_ref)
        return jsonify({
            "timeline_items": result,
            "item_count": len(result),
        })
    except Exception as exc:
        return safe_error(exc, "replace_nested")


# ===================================================================
# 41.2 — Flight Path Map
# ===================================================================

@integration_bp.route("/api/drone/parse-gps", methods=["POST"])
@require_csrf
def parse_gps_route():
    """Parse GPS track from SRT or GPX file."""
    try:
        from opencut.core.flight_path_map import parse_gps_track

        data = request.get_json(silent=True) or {}
        srt_path = data.get("filepath", data.get("srt_path", ""))
        if not srt_path:
            return jsonify({"error": "filepath/srt_path is required"}), 400
        validate_filepath(srt_path)

        points = parse_gps_track(srt_path)
        return jsonify({
            "points": [p.to_dict() for p in points],
            "count": len(points),
        })
    except Exception as exc:
        return safe_error(exc, "parse_gps")


@integration_bp.route("/api/drone/flight-map", methods=["POST"])
@require_csrf
@async_job("flight_map", filepath_required=False)
def flight_map_route(job_id, filepath, data):
    """Render an animated flight path map."""
    from opencut.core.flight_path_map import GpsPoint, render_flight_map

    gps_data = data.get("gps_points", [])
    if not gps_data:
        raise ValueError("gps_points is required")

    gps_points = [
        GpsPoint(
            latitude=float(p.get("latitude", 0)),
            longitude=float(p.get("longitude", 0)),
            altitude=float(p.get("altitude", 0)),
            timestamp=float(p.get("timestamp", 0)),
        )
        for p in gps_data
    ]

    out = data.get("output_path")
    if not out:
        out = os.path.join(tempfile.gettempdir(), "opencut_flight_map.mp4")

    w = safe_int(data.get("width", 640), 640, min_val=160, max_val=3840)
    h = safe_int(data.get("height", 480), 480, min_val=120, max_val=2160)
    duration = safe_float(data.get("duration", 0), 0, min_val=0.0, max_val=300.0)

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = render_flight_map(
        gps_points=gps_points,
        out_path=out,
        resolution=(w, h),
        duration=duration,
        on_progress=_p,
    )
    return {
        "output_path": result.output_path,
        "total_points": result.total_points,
        "total_distance_m": result.total_distance_m,
        "duration": result.duration,
        "bounds": result.bounds,
    }


@integration_bp.route("/api/drone/map-overlay", methods=["POST"])
@require_csrf
@async_job("map_overlay")
def map_overlay_route(job_id, filepath, data):
    """Overlay animated flight map on a video."""
    from opencut.core.flight_path_map import GpsPoint, create_map_overlay

    gps_data = data.get("gps_points", [])
    if not gps_data:
        raise ValueError("gps_points is required")

    gps_track = [
        GpsPoint(
            latitude=float(p.get("latitude", 0)),
            longitude=float(p.get("longitude", 0)),
            altitude=float(p.get("altitude", 0)),
        )
        for p in gps_data
    ]

    position = str(data.get("position", "bottom_right"))
    size_frac = safe_float(data.get("size_fraction", 0.25), 0.25, min_val=0.1, max_val=0.5)
    opacity = safe_float(data.get("opacity", 0.8), 0.8, min_val=0.0, max_val=1.0)

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = create_map_overlay(
        video_path=filepath,
        gps_track=gps_track,
        position=position,
        size_fraction=size_frac,
        opacity=opacity,
        on_progress=_p,
    )
    return {
        "output_path": result.output_path,
        "total_points": result.total_points,
        "total_distance_m": result.total_distance_m,
    }


# ===================================================================
# 41.3 — Aerial Hyperlapse
# ===================================================================

@integration_bp.route("/api/drone/hyperlapse", methods=["POST"])
@require_csrf
@async_job("aerial_hyperlapse")
def aerial_hyperlapse_route(job_id, filepath, data):
    """Create a GPS-sampled stabilised aerial hyperlapse."""
    from opencut.core.aerial_hyperlapse import create_aerial_hyperlapse

    srt_path = data.get("srt_path", "")
    if not srt_path:
        raise ValueError("srt_path is required")
    validate_filepath(srt_path)

    interval = safe_float(data.get("interval_meters", 10.0), 10.0, min_val=1.0, max_val=1000.0)
    speed = safe_float(data.get("speed_factor", 0.0), 0.0, min_val=0.0, max_val=100.0)
    do_stabilize = safe_bool(data.get("stabilize", True), True)
    fps = safe_float(data.get("fps", 30.0), 30.0, min_val=1.0, max_val=120.0)

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = create_aerial_hyperlapse(
        video_path=filepath,
        srt_path=srt_path,
        interval_meters=interval,
        speed_factor=speed,
        stabilize=do_stabilize,
        fps=fps,
        on_progress=_p,
    )
    return {
        "output_path": result.output_path,
        "total_frames": result.total_frames,
        "sampled_frames": result.sampled_frames,
        "total_distance_m": result.total_distance_m,
        "speed_factor": result.speed_factor,
        "stabilised": result.stabilised,
    }


@integration_bp.route("/api/drone/gps-sample", methods=["POST"])
@require_csrf
def gps_sample_route():
    """Sample GPS points by distance interval."""
    try:
        from opencut.core.aerial_hyperlapse import sample_by_gps_distance

        data = request.get_json(silent=True) or {}
        gps_points = data.get("gps_points", [])
        if not gps_points:
            return jsonify({"error": "gps_points is required"}), 400

        interval = safe_float(data.get("interval_meters", 10.0), 10.0, min_val=1.0, max_val=1000.0)

        indices = sample_by_gps_distance(gps_points, interval_meters=interval)
        return jsonify({
            "selected_indices": indices,
            "total_points": len(gps_points),
            "selected_count": len(indices),
        })
    except Exception as exc:
        return safe_error(exc, "gps_sample")


# ===================================================================
# 50.2 — Notion/PM Sync
# ===================================================================

@integration_bp.route("/api/integrations/notion/sync", methods=["POST"])
@require_csrf
def notion_sync_route():
    """Sync project data to Notion."""
    try:
        from opencut.core.notion_sync import sync_to_notion

        data = request.get_json(silent=True) or {}
        project_data = data.get("project_data", {})
        config = data.get("config", {})

        result = sync_to_notion(project_data=project_data, config=config)
        status = 200 if result.success else 400
        return jsonify({
            "success": result.success,
            "pages_created": result.pages_created,
            "pages_updated": result.pages_updated,
            "errors": result.errors,
            "page_ids": result.page_ids,
        }), status
    except Exception as exc:
        return safe_error(exc, "notion_sync")


@integration_bp.route("/api/integrations/notion/page", methods=["PATCH"])
@require_csrf
def notion_update_route():
    """Update a Notion page."""
    try:
        from opencut.core.notion_sync import update_notion_page

        data = request.get_json(silent=True) or {}
        page_id = str(data.get("page_id", "")).strip()
        properties = data.get("properties", {})
        api_key = str(data.get("api_key", "")).strip()

        if not page_id or not api_key:
            return jsonify({"error": "page_id and api_key are required"}), 400

        update_notion_page(page_id, properties, api_key)
        return jsonify({"updated": True, "page_id": page_id})
    except Exception as exc:
        return safe_error(exc, "notion_update")


@integration_bp.route("/api/integrations/notion/entry", methods=["POST"])
@require_csrf
def notion_create_route():
    """Create a new Notion database entry."""
    try:
        from opencut.core.notion_sync import create_notion_entry

        data = request.get_json(silent=True) or {}
        database_id = str(data.get("database_id", "")).strip()
        entry_data = data.get("data", {})
        api_key = str(data.get("api_key", "")).strip()

        if not database_id or not api_key:
            return jsonify({"error": "database_id and api_key are required"}), 400

        result = create_notion_entry(database_id, entry_data, api_key)
        return jsonify({"created": True, "page_id": result.get("id", "")})
    except Exception as exc:
        return safe_error(exc, "notion_create")


# ===================================================================
# 50.3 — Slack/Discord Notifications
# ===================================================================

@integration_bp.route("/api/integrations/slack/send", methods=["POST"])
@require_csrf
def slack_send_route():
    """Send a Slack notification."""
    try:
        from opencut.core.slack_notify import send_slack_notification

        data = request.get_json(silent=True) or {}
        webhook_url = str(data.get("webhook_url", "")).strip()
        message = str(data.get("message", "")).strip()

        if not webhook_url or not message:
            return jsonify({"error": "webhook_url and message are required"}), 400

        result = send_slack_notification(
            webhook_url=webhook_url,
            message=message,
            title=str(data.get("title", "OpenCut")),
            color=str(data.get("color", "#36a64f")),
            fields=data.get("fields"),
        )
        status = 200 if result.success else 502
        return jsonify({
            "success": result.success,
            "platform": result.platform,
            "status_code": result.status_code,
            "error": result.error,
        }), status
    except Exception as exc:
        return safe_error(exc, "slack_send")


@integration_bp.route("/api/integrations/discord/send", methods=["POST"])
@require_csrf
def discord_send_route():
    """Send a Discord notification."""
    try:
        from opencut.core.slack_notify import send_discord_notification

        data = request.get_json(silent=True) or {}
        webhook_url = str(data.get("webhook_url", "")).strip()
        message = str(data.get("message", "")).strip()

        if not webhook_url or not message:
            return jsonify({"error": "webhook_url and message are required"}), 400

        result = send_discord_notification(
            webhook_url=webhook_url,
            message=message,
            title=str(data.get("title", "OpenCut")),
            color=safe_int(data.get("color", 0x36A64F), 0x36A64F),
            fields=data.get("fields"),
        )
        status = 200 if result.success else 502
        return jsonify({
            "success": result.success,
            "platform": result.platform,
            "status_code": result.status_code,
            "error": result.error,
        }), status
    except Exception as exc:
        return safe_error(exc, "discord_send")


@integration_bp.route("/api/integrations/notify/format", methods=["POST"])
@require_csrf
def format_notification_route():
    """Format a job notification for a platform."""
    try:
        from opencut.core.slack_notify import format_job_notification

        data = request.get_json(silent=True) or {}
        job_data = data.get("job_data", {})
        fmt = str(data.get("format", "slack"))

        result = format_job_notification(job_data, format=fmt)
        return jsonify(result)
    except Exception as exc:
        return safe_error(exc, "format_notification")


# ===================================================================
# 50.4 — Zapier/Make Webhooks
# ===================================================================

@integration_bp.route("/api/integrations/webhooks/send", methods=["POST"])
@require_csrf
def webhook_send_route():
    """Send an outbound webhook."""
    try:
        from opencut.core.webhook_integrations import send_webhook

        data = request.get_json(silent=True) or {}
        url = str(data.get("url", "")).strip()
        event_type = str(data.get("event_type", "")).strip()
        payload = data.get("payload", {})

        if not url or not event_type:
            return jsonify({"error": "url and event_type are required"}), 400

        result = send_webhook(url=url, event_type=event_type, payload=payload)
        status = 200 if result.success else 502
        return jsonify({
            "success": result.success,
            "status_code": result.status_code,
            "error": result.error,
        }), status
    except Exception as exc:
        return safe_error(exc, "webhook_send")


@integration_bp.route("/api/integrations/webhooks/register", methods=["POST"])
@require_csrf
def webhook_register_route():
    """Register a webhook trigger."""
    try:
        from opencut.core.webhook_integrations import register_webhook_trigger

        data = request.get_json(silent=True) or {}
        event = str(data.get("event", "")).strip()
        url = str(data.get("url", "")).strip()

        if not event or not url:
            return jsonify({"error": "event and url are required"}), 400

        trigger = register_webhook_trigger(event=event, url=url)
        return jsonify(trigger.to_dict())
    except Exception as exc:
        return safe_error(exc, "webhook_register")


@integration_bp.route("/api/integrations/webhooks", methods=["GET"])
def webhook_list_route():
    """List registered webhook triggers."""
    try:
        from opencut.core.webhook_integrations import list_registered_webhooks
        triggers = list_registered_webhooks()
        return jsonify({"webhooks": triggers, "count": len(triggers)})
    except Exception as exc:
        return safe_error(exc, "webhook_list")


@integration_bp.route("/api/integrations/webhooks/inbound", methods=["POST"])
@require_csrf
def webhook_inbound_route():
    """Handle an inbound webhook from Zapier/Make."""
    try:
        from opencut.core.webhook_integrations import handle_inbound_webhook

        data = request.get_json(silent=True) or {}
        result = handle_inbound_webhook(data)
        status = 200 if result.get("accepted") else 400
        return jsonify(result), status
    except Exception as exc:
        return safe_error(exc, "webhook_inbound")


# ===================================================================
# 39.1 — Stream Deck Integration
# ===================================================================

@integration_bp.route("/api/hardware/stream-deck/profiles", methods=["GET"])
def stream_deck_list_route():
    """List available Stream Deck profiles."""
    try:
        from opencut.core.stream_deck import list_profiles
        profiles = list_profiles()
        return jsonify({"profiles": profiles, "count": len(profiles)})
    except Exception as exc:
        return safe_error(exc, "stream_deck_list")


@integration_bp.route("/api/hardware/stream-deck/profile", methods=["GET"])
def stream_deck_get_route():
    """Get a Stream Deck profile by name."""
    try:
        from opencut.core.stream_deck import get_stream_deck_profile

        name = request.args.get("name", "editing")
        profile = get_stream_deck_profile(name)
        return jsonify(profile)
    except Exception as exc:
        return safe_error(exc, "stream_deck_get")


@integration_bp.route("/api/hardware/stream-deck/profile", methods=["POST"])
@require_csrf
def stream_deck_save_route():
    """Save a Stream Deck profile."""
    try:
        from opencut.core.stream_deck import save_profile

        data = request.get_json(silent=True) or {}
        if not data.get("name"):
            return jsonify({"error": "name is required"}), 400

        path = save_profile(data)
        return jsonify({"saved": True, "path": path})
    except Exception as exc:
        return safe_error(exc, "stream_deck_save")


@integration_bp.route("/api/hardware/stream-deck/button", methods=["POST"])
@require_csrf
def stream_deck_button_route():
    """Create a single button mapping."""
    try:
        from opencut.core.stream_deck import create_button_mapping

        data = request.get_json(silent=True) or {}
        button_id = safe_int(data.get("button_id", 0), 0, min_val=0, max_val=31)
        operation = str(data.get("operation", "")).strip()
        if not operation:
            return jsonify({"error": "operation is required"}), 400

        result = create_button_mapping(
            button_id=button_id,
            operation=operation,
            params=data.get("params"),
            label=str(data.get("label", "")),
            icon=str(data.get("icon", "")),
            color=str(data.get("color", "#333333")),
        )
        return jsonify(result)
    except Exception as exc:
        return safe_error(exc, "stream_deck_button")


@integration_bp.route("/api/hardware/stream-deck/export", methods=["POST"])
@require_csrf
def stream_deck_export_route():
    """Export a Stream Deck profile to a file."""
    try:
        from opencut.core.stream_deck import export_stream_deck_profile

        data = request.get_json(silent=True) or {}
        profile = data.get("profile", {})
        output = str(data.get("output_path", "")).strip()

        if not output:
            return jsonify({"error": "output_path is required"}), 400
        output = validate_path(output)

        path = export_stream_deck_profile(profile, output)
        return jsonify({"exported": True, "path": path})
    except Exception as exc:
        return safe_error(exc, "stream_deck_export")


# ===================================================================
# 39.2 — MIDI Controller
# ===================================================================

@integration_bp.route("/api/hardware/midi/mapping", methods=["POST"])
@require_csrf
def midi_mapping_route():
    """Create a MIDI CC/note mapping."""
    try:
        from opencut.core.midi_controller import create_midi_mapping

        data = request.get_json(silent=True) or {}
        channel = safe_int(data.get("channel", 0), 0, min_val=0, max_val=15)
        parameter = str(data.get("parameter", "")).strip()
        if not parameter:
            return jsonify({"error": "parameter is required"}), 400

        result = create_midi_mapping(
            cc_channel=channel,
            parameter=parameter,
            range_config=data.get("range_config"),
            midi_type=str(data.get("midi_type", "cc")),
            cc_number=safe_int(data.get("cc_number", 0), 0, min_val=0, max_val=127),
        )
        return jsonify(result)
    except Exception as exc:
        return safe_error(exc, "midi_mapping")


@integration_bp.route("/api/hardware/midi/save", methods=["POST"])
@require_csrf
def midi_save_route():
    """Save a MIDI mapping file."""
    try:
        from opencut.core.midi_controller import save_midi_map

        data = request.get_json(silent=True) or {}
        mappings = data.get("mappings", [])
        output = str(data.get("output_path", "")).strip()
        if not output:
            return jsonify({"error": "output_path is required"}), 400
        output = validate_path(output)

        path = save_midi_map(
            mappings=mappings,
            output_path=output,
            name=str(data.get("name", "")),
            device_name=str(data.get("device_name", "")),
        )
        return jsonify({"saved": True, "path": path})
    except Exception as exc:
        return safe_error(exc, "midi_save")


@integration_bp.route("/api/hardware/midi/load", methods=["POST"])
@require_csrf
def midi_load_route():
    """Load a MIDI mapping file."""
    try:
        from opencut.core.midi_controller import load_midi_map

        data = request.get_json(silent=True) or {}
        path = str(data.get("path", "")).strip()
        if not path:
            return jsonify({"error": "path is required"}), 400
        validate_filepath(path)

        result = load_midi_map(path)
        return jsonify(result)
    except Exception as exc:
        return safe_error(exc, "midi_load")


@integration_bp.route("/api/hardware/midi/devices", methods=["GET"])
def midi_devices_route():
    """List known MIDI device templates."""
    try:
        from opencut.core.midi_controller import list_midi_devices
        devices = list_midi_devices()
        return jsonify({"devices": devices, "count": len(devices)})
    except Exception as exc:
        return safe_error(exc, "midi_devices")


# ===================================================================
# 39.3 — Shuttle/Jog Wheel
# ===================================================================

@integration_bp.route("/api/hardware/jog/mapping", methods=["POST"])
@require_csrf
def jog_mapping_route():
    """Create a jog/shuttle controller mapping."""
    try:
        from opencut.core.jog_wheel import create_jog_mapping

        data = request.get_json(silent=True) or {}
        device_type = str(data.get("device_type", "generic_hid"))
        mappings = data.get("mappings", [])

        result = create_jog_mapping(
            device_type=device_type,
            mappings=mappings,
            name=str(data.get("name", "")),
        )
        return jsonify(result)
    except Exception as exc:
        return safe_error(exc, "jog_mapping")


@integration_bp.route("/api/hardware/jog/devices", methods=["GET"])
def jog_devices_route():
    """List supported jog/shuttle devices."""
    try:
        from opencut.core.jog_wheel import list_supported_devices
        devices = list_supported_devices()
        return jsonify({"devices": devices, "count": len(devices)})
    except Exception as exc:
        return safe_error(exc, "jog_devices")


@integration_bp.route("/api/hardware/jog/default", methods=["GET"])
def jog_default_route():
    """Get default mapping for a device type."""
    try:
        from opencut.core.jog_wheel import get_default_mapping

        device_type = request.args.get("device_type", "shuttlepro_v2")
        result = get_default_mapping(device_type)
        return jsonify(result)
    except Exception as exc:
        return safe_error(exc, "jog_default")


@integration_bp.route("/api/hardware/jog/save", methods=["POST"])
@require_csrf
def jog_save_route():
    """Save a jog mapping to file."""
    try:
        from opencut.core.jog_wheel import save_jog_mapping

        data = request.get_json(silent=True) or {}
        output = str(data.get("output_path", "")).strip()

        path = save_jog_mapping(mapping=data, output_path=output)
        return jsonify({"saved": True, "path": path})
    except Exception as exc:
        return safe_error(exc, "jog_save")


# ===================================================================
# 39.4 — Touch/Pen Optimization
# ===================================================================

@integration_bp.route("/api/hardware/touch/config", methods=["GET"])
def touch_config_route():
    """Get touch optimization config."""
    try:
        from opencut.core.touch_optimize import get_touch_config
        config = get_touch_config()
        return jsonify(config)
    except Exception as exc:
        return safe_error(exc, "touch_config")


@integration_bp.route("/api/hardware/touch/config", methods=["POST"])
@require_csrf
def touch_save_route():
    """Save touch optimization config."""
    try:
        from opencut.core.touch_optimize import save_touch_config

        data = request.get_json(silent=True) or {}
        path = save_touch_config(data)
        return jsonify({"saved": True, "path": path})
    except Exception as exc:
        return safe_error(exc, "touch_save")


@integration_bp.route("/api/hardware/touch/gesture", methods=["POST"])
@require_csrf
def touch_gesture_route():
    """Create a gesture mapping."""
    try:
        from opencut.core.touch_optimize import create_gesture_mapping

        data = request.get_json(silent=True) or {}
        gesture = str(data.get("gesture", "")).strip()
        action = str(data.get("action", "")).strip()

        if not gesture or not action:
            return jsonify({"error": "gesture and action are required"}), 400

        result = create_gesture_mapping(gesture=gesture, action=action, params=data.get("params"))
        return jsonify(result)
    except Exception as exc:
        return safe_error(exc, "touch_gesture")


@integration_bp.route("/api/hardware/touch/pen", methods=["GET"])
def touch_pen_route():
    """Get pen pressure configuration."""
    try:
        from opencut.core.touch_optimize import get_pen_pressure_config
        config = get_pen_pressure_config()
        return jsonify(config)
    except Exception as exc:
        return safe_error(exc, "touch_pen")


@integration_bp.route("/api/hardware/touch/optimize-layout", methods=["POST"])
@require_csrf
def touch_optimize_route():
    """Optimize a UI layout for touch interaction."""
    try:
        from opencut.core.touch_optimize import optimize_layout_for_touch

        data = request.get_json(silent=True) or {}
        layout = data.get("layout", data)
        result = optimize_layout_for_touch(layout)
        return jsonify(result)
    except Exception as exc:
        return safe_error(exc, "touch_optimize")
