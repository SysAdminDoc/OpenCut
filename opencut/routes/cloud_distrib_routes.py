"""
OpenCut Cloud & Distribution Routes

Cloud render dispatch, platform publish preparation, content fingerprinting,
render farm management, and distribution analytics endpoints.
"""

import logging

from flask import Blueprint, jsonify, request

from opencut.errors import safe_error
from opencut.helpers import _resolve_output_dir
from opencut.jobs import _update_job, async_job
from opencut.security import require_csrf, safe_float, safe_int, validate_filepath

logger = logging.getLogger("opencut")

cloud_distrib_bp = Blueprint("cloud_distrib", __name__)


# ============================================================================
# Cloud Render
# ============================================================================

@cloud_distrib_bp.route("/cloud/render", methods=["POST"])
@require_csrf
@async_job("cloud_render")
def cloud_render(job_id, filepath, data):
    """Submit a render job to a cloud/remote node."""
    from opencut.core.cloud_render import dispatch_render

    render_config = data.get("render_config", {})
    required_capabilities = data.get("required_capabilities", None)
    timeout = safe_int(data.get("timeout", 3600), 3600, min_val=60, max_val=14400)

    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = _resolve_output_dir(filepath, output_dir)
        render_config["output_dir"] = output_dir

    def _progress(pct):
        _update_job(job_id, progress=pct, message=f"Cloud render... {pct}%")

    result = dispatch_render(
        filepath,
        render_config=render_config,
        required_capabilities=required_capabilities,
        on_progress=_progress,
        timeout=timeout,
    )

    return result.to_dict()


@cloud_distrib_bp.route("/cloud/nodes", methods=["GET"])
def cloud_nodes_list():
    """List all configured render nodes with health status."""
    try:
        from opencut.core.cloud_render import (
            check_all_nodes,
            get_node_summary,
            load_nodes,
        )

        refresh = request.args.get("refresh", "false").lower() == "true"
        nodes = load_nodes()

        statuses = []
        if refresh:
            statuses = [s.to_dict() for s in check_all_nodes()]
        else:
            from opencut.core.cloud_render import get_cached_statuses
            cached = get_cached_statuses()
            for node in nodes:
                if node.name in cached:
                    statuses.append(cached[node.name].to_dict())
                else:
                    statuses.append({
                        "name": node.name,
                        "online": False,
                        "error": "Not checked yet",
                    })

        return jsonify({
            "nodes": [n.to_dict() for n in nodes],
            "statuses": statuses,
            "summary": get_node_summary(),
        })
    except Exception as exc:
        return safe_error(exc, "cloud_nodes_list")


@cloud_distrib_bp.route("/cloud/nodes", methods=["POST"])
@require_csrf
def cloud_nodes_add():
    """Add or update a render node configuration."""
    try:
        from opencut.core.cloud_render import RenderNode, add_node

        data = request.get_json(force=True) or {}
        name = (data.get("name", "") or "").strip()
        host = (data.get("host", "") or "").strip()

        if not name:
            return jsonify({"error": "Node name is required"}), 400
        if not host:
            return jsonify({"error": "Node host is required"}), 400

        node = RenderNode(
            name=name,
            host=host,
            port=safe_int(data.get("port", 9090), 9090, min_val=1, max_val=65535),
            capabilities=data.get("capabilities", ["cpu"]),
            max_concurrent=safe_int(data.get("max_concurrent", 2), 2, min_val=1, max_val=100),
            enabled=data.get("enabled", True),
            priority=safe_int(data.get("priority", 0), 0, min_val=0, max_val=100),
            auth_token=data.get("auth_token", ""),
        )

        nodes = add_node(node)
        return jsonify({
            "message": f"Node '{name}' saved",
            "nodes": [n.to_dict() for n in nodes],
        })
    except Exception as exc:
        return safe_error(exc, "cloud_nodes_add")


@cloud_distrib_bp.route("/cloud/nodes/<name>", methods=["DELETE"])
@require_csrf
def cloud_nodes_remove(name):
    """Remove a render node by name."""
    try:
        from opencut.core.cloud_render import remove_node

        nodes = remove_node(name)
        return jsonify({
            "message": f"Node '{name}' removed",
            "nodes": [n.to_dict() for n in nodes],
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as exc:
        return safe_error(exc, "cloud_nodes_remove")


# ============================================================================
# Platform Publish
# ============================================================================

@cloud_distrib_bp.route("/publish/prepare", methods=["POST"])
@require_csrf
@async_job("platform_publish")
def publish_prepare(job_id, filepath, data):
    """Generate platform-optimized publish packages."""
    from opencut.core.platform_publish import batch_prepare, prepare_publish_package

    platforms = data.get("platforms", [])
    metadata = data.get("metadata", {})
    output_dir = data.get("output_dir", "")
    caption_path = data.get("caption_path", "")

    if output_dir:
        output_dir = _resolve_output_dir(filepath, output_dir)

    def _progress(pct):
        _update_job(job_id, progress=pct, message=f"Preparing publish packages... {pct}%")

    if platforms:
        manifest = batch_prepare(
            filepath, platforms, metadata, output_dir, caption_path,
            on_progress=_progress,
        )
        return manifest.to_dict()
    else:
        # Single platform from "platform" field
        platform = data.get("platform", "youtube")
        pkg = prepare_publish_package(
            filepath, platform, metadata, output_dir, caption_path,
            on_progress=_progress,
        )
        return pkg.to_dict()


@cloud_distrib_bp.route("/publish/platforms", methods=["GET"])
def publish_platforms():
    """List all supported platforms with their specifications."""
    try:
        from opencut.core.platform_publish import list_platforms
        return jsonify({"platforms": list_platforms()})
    except Exception as exc:
        return safe_error(exc, "publish_platforms")


@cloud_distrib_bp.route("/publish/validate", methods=["POST"])
@require_csrf
def publish_validate():
    """Validate metadata for a specific platform."""
    try:
        from opencut.core.platform_publish import validate_metadata, validate_video_for_platform

        data = request.get_json(force=True) or {}
        platform = data.get("platform", "")
        metadata = data.get("metadata", {})
        filepath = data.get("filepath", "")

        if not platform:
            return jsonify({"error": "Platform is required"}), 400

        errors = validate_metadata(platform, metadata)

        if filepath:
            try:
                filepath = validate_filepath(filepath)
                from opencut.helpers import get_video_info
                video_info = get_video_info(filepath)
                video_errors = validate_video_for_platform(platform, video_info)
                errors.extend(video_errors)
            except ValueError:
                pass  # Skip video validation if path invalid

        return jsonify({
            "platform": platform,
            "valid": all(e.severity != "error" for e in errors),
            "errors": [e.to_dict() for e in errors],
        })
    except Exception as exc:
        return safe_error(exc, "publish_validate")


# ============================================================================
# Content Fingerprint
# ============================================================================

@cloud_distrib_bp.route("/fingerprint/generate", methods=["POST"])
@require_csrf
@async_job("fingerprint")
def fingerprint_generate(job_id, filepath, data):
    """Generate a content fingerprint for a media file."""
    from opencut.core.content_fingerprint import generate_fingerprint, index_fingerprint

    auto_index = data.get("auto_index", True)

    def _progress(pct):
        _update_job(job_id, progress=pct, message=f"Fingerprinting... {pct}%")

    result = generate_fingerprint(filepath, on_progress=_progress)

    if auto_index:
        index_fingerprint(result)

    return result.to_dict()


@cloud_distrib_bp.route("/fingerprint/search", methods=["POST"])
@require_csrf
@async_job("fingerprint_search")
def fingerprint_search(job_id, filepath, data):
    """Search the fingerprint index for similar content."""
    from opencut.core.content_fingerprint import (
        generate_fingerprint,
        get_indexed_count,
        search_similar,
    )

    threshold = safe_float(data.get("threshold", 85), 85.0, min_val=0, max_val=100)
    max_results = safe_int(data.get("max_results", 20), 20, min_val=1, max_val=100)

    _update_job(job_id, progress=10, message="Generating query fingerprint...")

    def _progress(pct):
        scaled = 10 + int(pct * 0.6)
        _update_job(job_id, progress=scaled, message=f"Fingerprinting... {scaled}%")

    query_fp = generate_fingerprint(filepath, on_progress=_progress)

    _update_job(job_id, progress=75, message="Searching index...")

    matches = search_similar(query_fp, threshold=threshold, max_results=max_results)

    return {
        "query": query_fp.to_dict(),
        "matches": [m.to_dict() for m in matches],
        "total_indexed": get_indexed_count(),
        "threshold": threshold,
    }


# ============================================================================
# Render Farm
# ============================================================================

@cloud_distrib_bp.route("/farm/render", methods=["POST"])
@require_csrf
@async_job("farm_render")
def farm_render_submit(job_id, filepath, data):
    """Submit a render farm job."""
    from opencut.core.render_farm import farm_render

    strategy = data.get("strategy", "equal_duration")
    if strategy not in ("equal_duration", "scene_based", "chapter_based"):
        raise ValueError(f"Invalid strategy: {strategy}")

    num_segments = safe_int(data.get("num_segments", 4), 4, min_val=2, max_val=64)
    render_config = data.get("render_config", {})
    output_dir = data.get("output_dir", "")
    output_file = data.get("output_file", "")
    use_remote = data.get("use_remote", False)

    if output_dir:
        output_dir = _resolve_output_dir(filepath, output_dir)

    def _progress(pct):
        _update_job(job_id, progress=pct, message=f"Farm render... {pct}%")

    result = farm_render(
        filepath,
        strategy=strategy,
        num_segments=num_segments,
        render_config=render_config,
        output_dir=output_dir,
        output_file=output_file,
        use_remote=use_remote,
        on_progress=_progress,
    )

    return result.to_dict()


@cloud_distrib_bp.route("/farm/status", methods=["GET"])
def farm_status():
    """Get render farm overview and segment estimation."""
    try:
        filepath = request.args.get("filepath", "")
        strategy = request.args.get("strategy", "equal_duration")
        num_segments = safe_int(request.args.get("num_segments", 4), 4, min_val=2, max_val=64)

        result = {}

        if filepath:
            try:
                filepath = validate_filepath(filepath)
                from opencut.core.render_farm import estimate_segments
                result["estimate"] = estimate_segments(filepath, strategy, num_segments)
            except ValueError:
                pass

        # Node summary if cloud render is available
        try:
            from opencut.core.cloud_render import get_node_summary
            result["nodes"] = get_node_summary()
        except Exception:
            result["nodes"] = {"total_nodes": 0}

        return jsonify(result)
    except Exception as exc:
        return safe_error(exc, "farm_status")


# ============================================================================
# Distribution Analytics
# ============================================================================

@cloud_distrib_bp.route("/analytics/record", methods=["POST"])
@require_csrf
def analytics_record():
    """Record publish metrics for a video."""
    try:
        from opencut.core.distribution_analytics import (
            MetricsEntry,
            PublishRecord,
            add_publish_record,
            record_metrics,
        )

        data = request.get_json(force=True) or {}
        action = data.get("action", "metrics")

        if action == "publish":
            record = PublishRecord(
                video_title=data.get("video_title", ""),
                platform=data.get("platform", ""),
                publish_date=data.get("publish_date", ""),
                url=data.get("url", ""),
                file_path=data.get("file_path", ""),
                resolution=data.get("resolution", ""),
                duration_sec=safe_float(data.get("duration_sec", 0), 0),
                file_size_mb=safe_float(data.get("file_size_mb", 0), 0),
                category=data.get("category", ""),
                tags=data.get("tags", []),
            )
            if not record.video_title:
                return jsonify({"error": "video_title is required"}), 400
            if not record.platform:
                return jsonify({"error": "platform is required"}), 400
            record_id = add_publish_record(record)
            return jsonify({"record_id": record_id, "message": "Publish record created"})

        else:
            record_id = safe_int(data.get("record_id", 0), 0, min_val=1)
            if not record_id:
                return jsonify({"error": "record_id is required"}), 400
            entry = MetricsEntry(
                record_id=record_id,
                views=safe_int(data.get("views", 0), 0, min_val=0),
                likes=safe_int(data.get("likes", 0), 0, min_val=0),
                comments=safe_int(data.get("comments", 0), 0, min_val=0),
                shares=safe_int(data.get("shares", 0), 0, min_val=0),
                watch_time_avg=safe_float(data.get("watch_time_avg", 0), 0, min_val=0),
                ctr=safe_float(data.get("ctr", 0), 0, min_val=0, max_val=1),
            )
            metric_id = record_metrics(entry)
            return jsonify({"metric_id": metric_id, "message": "Metrics recorded"})

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as exc:
        return safe_error(exc, "analytics_record")


@cloud_distrib_bp.route("/analytics/report", methods=["GET"])
def analytics_report():
    """Get a comprehensive analytics report."""
    try:
        from opencut.core.distribution_analytics import (
            content_type_analysis,
            generate_report,
        )

        days = safe_int(request.args.get("days", 30), 30, min_val=1, max_val=365)
        report = generate_report(days=days)

        result = report.to_dict()
        result["content_analysis"] = content_type_analysis()

        return jsonify(result)
    except Exception as exc:
        return safe_error(exc, "analytics_report")
