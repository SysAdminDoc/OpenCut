"""
OpenCut Platform & Infrastructure Routes

Routes for review links, Resolve integration, plugin marketplace,
ONNX runtime, AMD GPU, stock media, FFmpeg filter builder,
smart render, render cache, timeline diff, edit branches,
Frame.io integration, waveform timeline, and preview server.
"""

import logging
import os

from flask import Blueprint, jsonify, request

from opencut.errors import safe_error
from opencut.helpers import _resolve_output_dir, _unique_output_path
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

platform_infra_bp = Blueprint("platform_infra", __name__)


# ===================================================================
# 1. Review & Approval Links
# ===================================================================

@platform_infra_bp.route("/review/create", methods=["POST"])
@require_csrf
@async_job("review-create")
def review_create(job_id, filepath, data):
    """Create a shareable review link for a video."""
    title = data.get("title", "")
    expires_hours = safe_float(data.get("expires_hours"), default=0)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.review_links import create_review_link
    result = create_review_link(
        video_path=filepath,
        title=title,
        expires_hours=expires_hours if expires_hours > 0 else None,
        on_progress=_on_progress,
    )
    return {
        "review_id": result.review_id,
        "token": result.token,
        "status": result.status,
        "title": result.title,
    }


@platform_infra_bp.route("/review/comment", methods=["POST"])
@require_csrf
def review_add_comment():
    """Add a timestamped comment to a review."""
    try:
        data = request.get_json(force=True)
        review_id = data.get("review_id", "")
        timestamp = safe_float(data.get("timestamp"), default=0)
        text = data.get("text", "")
        author = data.get("author", "Anonymous")

        from opencut.core.review_links import add_review_comment
        comment = add_review_comment(review_id, timestamp, text, author)
        return jsonify({
            "comment_id": comment.comment_id,
            "review_id": comment.review_id,
            "timestamp": comment.timestamp,
            "text": comment.text,
            "author": comment.author,
        })
    except Exception as exc:
        return safe_error(exc, "review_comment")


@platform_infra_bp.route("/review/comments", methods=["POST"])
@require_csrf
def review_get_comments():
    """Get all comments for a review."""
    try:
        data = request.get_json(force=True)
        review_id = data.get("review_id", "")

        from opencut.core.review_links import get_review_comments
        comments = get_review_comments(review_id)
        return jsonify({
            "review_id": review_id,
            "comments": [
                {
                    "comment_id": c.comment_id,
                    "timestamp": c.timestamp,
                    "text": c.text,
                    "author": c.author,
                    "created_at": c.created_at,
                }
                for c in comments
            ],
        })
    except Exception as exc:
        return safe_error(exc, "review_comments")


@platform_infra_bp.route("/review/status", methods=["POST"])
@require_csrf
def review_update_status():
    """Update review approval status."""
    try:
        data = request.get_json(force=True)
        review_id = data.get("review_id", "")
        status = data.get("status", "")

        from opencut.core.review_links import update_review_status
        result = update_review_status(review_id, status)
        return jsonify({
            "review_id": result.review_id,
            "status": result.status,
        })
    except Exception as exc:
        return safe_error(exc, "review_status")


# ===================================================================
# 2. DaVinci Resolve Integration
# ===================================================================

@platform_infra_bp.route("/resolve/marker", methods=["POST"])
@require_csrf
def resolve_add_marker_route():
    """Add a marker to the Resolve timeline."""
    try:
        data = request.get_json(force=True)
        timestamp = safe_float(data.get("timestamp"), default=0)
        name = data.get("name", "Marker")
        color = data.get("color", "Blue")
        note = data.get("note", "")
        duration = safe_int(data.get("duration"), default=1, min_val=1)

        from opencut.core.resolve_integration import resolve_add_marker
        marker = resolve_add_marker(
            timestamp=timestamp, name=name, color=color,
            note=note, duration=duration,
        )
        return jsonify({
            "frame": marker.frame,
            "name": marker.name,
            "color": marker.color,
        })
    except Exception as exc:
        return safe_error(exc, "resolve_marker")


@platform_infra_bp.route("/resolve/cuts", methods=["POST"])
@require_csrf
def resolve_apply_cuts_route():
    """Apply cuts to the Resolve timeline."""
    try:
        data = request.get_json(force=True)
        cut_list = data.get("cuts", [])

        from opencut.core.resolve_integration import resolve_apply_cuts
        result = resolve_apply_cuts(cut_list=cut_list)
        return jsonify(result)
    except Exception as exc:
        return safe_error(exc, "resolve_cuts")


@platform_infra_bp.route("/resolve/import", methods=["POST"])
@require_csrf
def resolve_import_route():
    """Import media into Resolve media pool."""
    try:
        data = request.get_json(force=True)
        paths = data.get("paths", [])

        from opencut.core.resolve_integration import resolve_import_media
        result = resolve_import_media(paths)
        return jsonify(result)
    except Exception as exc:
        return safe_error(exc, "resolve_import")


@platform_infra_bp.route("/resolve/timeline", methods=["GET"])
def resolve_timeline_info():
    """Get current Resolve timeline info."""
    try:
        from opencut.core.resolve_integration import resolve_get_timeline_info
        info = resolve_get_timeline_info()
        return jsonify({
            "name": info.name,
            "frame_rate": info.frame_rate,
            "width": info.width,
            "height": info.height,
            "duration_frames": info.duration_frames,
            "track_count": info.track_count,
            "marker_count": info.marker_count,
        })
    except Exception as exc:
        return safe_error(exc, "resolve_timeline")


@platform_infra_bp.route("/resolve/render", methods=["POST"])
@require_csrf
def resolve_render_route():
    """Start a Resolve render."""
    try:
        data = request.get_json(force=True)
        settings = data.get("settings", {})

        from opencut.core.resolve_integration import resolve_render
        result = resolve_render(settings=settings)
        return jsonify({
            "success": result.success,
            "output_path": result.output_path,
            "message": result.message,
        })
    except Exception as exc:
        return safe_error(exc, "resolve_render")


# ===================================================================
# 3. Plugin Marketplace
# ===================================================================

@platform_infra_bp.route("/plugins/registry", methods=["GET"])
def plugins_registry():
    """Fetch the plugin registry."""
    try:
        force = safe_bool(request.args.get("force"))
        from opencut.core.plugin_marketplace import fetch_plugin_registry
        plugins = fetch_plugin_registry(force=force)
        return jsonify({
            "plugins": [
                {
                    "plugin_id": p.plugin_id,
                    "name": p.name,
                    "version": p.version,
                    "author": p.author,
                    "description": p.description,
                    "tags": p.tags,
                    "installed": p.installed,
                    "installed_version": p.installed_version,
                }
                for p in plugins
            ],
        })
    except Exception as exc:
        return safe_error(exc, "plugins_registry")


@platform_infra_bp.route("/plugins/search", methods=["GET"])
def plugins_search():
    """Search plugins by keyword."""
    try:
        query = request.args.get("q", "")
        from opencut.core.plugin_marketplace import search_plugins
        results = search_plugins(query)
        return jsonify({
            "query": query,
            "results": [
                {
                    "plugin_id": p.plugin_id,
                    "name": p.name,
                    "version": p.version,
                    "description": p.description,
                    "installed": p.installed,
                }
                for p in results
            ],
        })
    except Exception as exc:
        return safe_error(exc, "plugins_search")


@platform_infra_bp.route("/plugins/install", methods=["POST"])
@require_csrf
@async_job("plugin-install", filepath_required=False)
def plugins_install(job_id, filepath, data):
    """Install a plugin from the marketplace."""
    plugin_id = data.get("plugin_id", "")
    if not plugin_id:
        raise ValueError("plugin_id is required")

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.plugin_marketplace import install_plugin
    result = install_plugin(plugin_id, on_progress=_on_progress)
    return {
        "plugin_id": result.plugin_id,
        "name": result.name,
        "version": result.version,
        "installed": result.installed,
    }


@platform_infra_bp.route("/plugins/update", methods=["POST"])
@require_csrf
@async_job("plugin-update", filepath_required=False)
def plugins_update(job_id, filepath, data):
    """Update an installed plugin."""
    plugin_id = data.get("plugin_id", "")
    if not plugin_id:
        raise ValueError("plugin_id is required")

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.plugin_marketplace import update_plugin
    result = update_plugin(plugin_id, on_progress=_on_progress)
    return {
        "plugin_id": result.plugin_id,
        "name": result.name,
        "version": result.version,
    }


@platform_infra_bp.route("/plugins/installed", methods=["GET"])
def plugins_installed():
    """List installed plugins."""
    try:
        from opencut.core.plugin_marketplace import list_installed_plugins
        plugins = list_installed_plugins()
        return jsonify({
            "plugins": [
                {
                    "plugin_id": p.plugin_id,
                    "name": p.name,
                    "version": p.version,
                }
                for p in plugins
            ],
        })
    except Exception as exc:
        return safe_error(exc, "plugins_installed")


# ===================================================================
# 4. ONNX Runtime
# ===================================================================

@platform_infra_bp.route("/onnx/providers", methods=["GET"])
def onnx_providers():
    """Check available ONNX execution providers."""
    try:
        from opencut.core.onnx_runtime import check_onnx_providers
        providers = check_onnx_providers()
        return jsonify({
            "providers": [
                {
                    "name": p.name,
                    "available": p.available,
                    "priority": p.priority,
                    "details": p.details,
                }
                for p in providers
            ],
        })
    except Exception as exc:
        return safe_error(exc, "onnx_providers")


@platform_infra_bp.route("/onnx/optimal-provider", methods=["GET"])
def onnx_optimal_provider():
    """Get the best available ONNX provider."""
    try:
        from opencut.core.onnx_runtime import get_optimal_provider
        provider = get_optimal_provider()
        return jsonify({"provider": provider})
    except Exception as exc:
        return safe_error(exc, "onnx_optimal_provider")


@platform_infra_bp.route("/onnx/inference", methods=["POST"])
@require_csrf
@async_job("onnx-inference", filepath_param="model_path")
def onnx_inference(job_id, filepath, data):
    """Run ONNX model inference."""
    input_data = data.get("input_data")
    provider = data.get("provider")
    input_name = data.get("input_name", "input")

    if input_data is None:
        raise ValueError("input_data is required")

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.onnx_runtime import run_onnx_inference
    result = run_onnx_inference(
        model_path=filepath,
        input_data=input_data,
        provider=provider,
        input_name=input_name,
        on_progress=_on_progress,
    )
    return result


# ===================================================================
# 5. AMD GPU Support
# ===================================================================

@platform_infra_bp.route("/amd/detect", methods=["GET"])
def amd_detect():
    """Detect AMD GPUs on the system."""
    try:
        from opencut.core.amd_gpu import detect_amd_gpu
        gpus = detect_amd_gpu()
        return jsonify({
            "gpus": [
                {
                    "name": g.name,
                    "vram_mb": g.vram_mb,
                    "architecture": g.architecture,
                    "supports_directml": g.supports_directml,
                    "supports_rocm": g.supports_rocm,
                }
                for g in gpus
            ],
            "count": len(gpus),
        })
    except Exception as exc:
        return safe_error(exc, "amd_detect")


@platform_infra_bp.route("/amd/directml", methods=["GET"])
def amd_directml():
    """Check DirectML availability."""
    try:
        from opencut.core.amd_gpu import get_directml_device
        result = get_directml_device()
        return jsonify(result)
    except Exception as exc:
        return safe_error(exc, "amd_directml")


@platform_infra_bp.route("/amd/rocm", methods=["GET"])
def amd_rocm():
    """Check ROCm availability."""
    try:
        from opencut.core.amd_gpu import check_rocm_available
        available = check_rocm_available()
        return jsonify({"rocm_available": available})
    except Exception as exc:
        return safe_error(exc, "amd_rocm")


@platform_infra_bp.route("/amd/capabilities", methods=["GET"])
def amd_capabilities():
    """Get comprehensive AMD GPU capabilities."""
    try:
        from opencut.core.amd_gpu import get_amd_capabilities
        result = get_amd_capabilities()
        return jsonify(result)
    except Exception as exc:
        return safe_error(exc, "amd_capabilities")


# ===================================================================
# 6. Stock Media Search
# ===================================================================

@platform_infra_bp.route("/stock/video", methods=["GET"])
def stock_video_search():
    """Search for stock videos."""
    try:
        query = request.args.get("q", "")
        source = request.args.get("source", "pexels")
        page = safe_int(request.args.get("page"), default=1, min_val=1)
        per_page = safe_int(request.args.get("per_page"), default=15, min_val=1, max_val=80)

        from opencut.core.stock_search import search_stock_video
        results = search_stock_video(query, source, page, per_page)
        return jsonify({
            "query": query,
            "source": source,
            "results": [
                {
                    "media_id": r.media_id,
                    "title": r.title,
                    "preview_url": r.preview_url,
                    "download_url": r.download_url,
                    "width": r.width,
                    "height": r.height,
                    "duration": r.duration,
                    "author": r.author,
                }
                for r in results
            ],
        })
    except Exception as exc:
        return safe_error(exc, "stock_video")


@platform_infra_bp.route("/stock/photo", methods=["GET"])
def stock_photo_search():
    """Search for stock photos."""
    try:
        query = request.args.get("q", "")
        source = request.args.get("source", "pexels")
        page = safe_int(request.args.get("page"), default=1, min_val=1)
        per_page = safe_int(request.args.get("per_page"), default=15, min_val=1, max_val=80)

        from opencut.core.stock_search import search_stock_photo
        results = search_stock_photo(query, source, page, per_page)
        return jsonify({
            "query": query,
            "source": source,
            "results": [
                {
                    "media_id": r.media_id,
                    "title": r.title,
                    "preview_url": r.preview_url,
                    "download_url": r.download_url,
                    "width": r.width,
                    "height": r.height,
                    "author": r.author,
                }
                for r in results
            ],
        })
    except Exception as exc:
        return safe_error(exc, "stock_photo")


@platform_infra_bp.route("/stock/download", methods=["POST"])
@require_csrf
@async_job("stock-download", filepath_required=False)
def stock_download(job_id, filepath, data):
    """Download a stock media file."""
    media_id = data.get("media_id", "")
    source = data.get("source", "pexels")
    url = data.get("url", "")
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)

    if not media_id:
        raise ValueError("media_id is required")
    if not url:
        raise ValueError("download URL is required")
    if not output_dir:
        output_dir = os.path.join(os.path.expanduser("~"), "Downloads")

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.stock_search import download_stock_media
    return download_stock_media(
        media_id=media_id,
        source=source,
        output_dir=output_dir,
        url=url,
        on_progress=_on_progress,
    )


# ===================================================================
# 7. FFmpeg Filter Chain Builder
# ===================================================================

@platform_infra_bp.route("/filter/build", methods=["POST"])
@require_csrf
def filter_build():
    """Build a filter_complex string from a node graph."""
    try:
        data = request.get_json(force=True)
        nodes = data.get("nodes", [])
        connections = data.get("connections")

        from opencut.core.ffmpeg_builder import build_filter_chain
        chain = build_filter_chain(nodes, connections)
        return jsonify({"filter_chain": chain})
    except Exception as exc:
        return safe_error(exc, "filter_build")


@platform_infra_bp.route("/filter/validate", methods=["POST"])
@require_csrf
def filter_validate():
    """Validate a filter graph."""
    try:
        data = request.get_json(force=True)
        from opencut.core.ffmpeg_builder import validate_filter_graph
        result = validate_filter_graph(data)
        return jsonify(result)
    except Exception as exc:
        return safe_error(exc, "filter_validate")


@platform_infra_bp.route("/filter/preview", methods=["POST"])
@require_csrf
@async_job("filter-preview")
def filter_preview(job_id, filepath, data):
    """Preview a filter chain on a video frame."""
    chain = data.get("filter_chain", "")
    timestamp = safe_float(data.get("timestamp"), default=0)

    if not chain:
        raise ValueError("filter_chain is required")

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.ffmpeg_builder import preview_filter
    output = preview_filter(filepath, chain, timestamp, on_progress=_on_progress)
    return {"output_path": output}


@platform_infra_bp.route("/filter/presets", methods=["GET"])
def filter_presets_list():
    """List saved filter presets."""
    try:
        from opencut.core.ffmpeg_builder import load_filter_presets
        presets = load_filter_presets()
        return jsonify({"presets": presets})
    except Exception as exc:
        return safe_error(exc, "filter_presets")


@platform_infra_bp.route("/filter/presets/save", methods=["POST"])
@require_csrf
def filter_presets_save():
    """Save a filter chain as a preset."""
    try:
        data = request.get_json(force=True)
        chain = data.get("chain", "")
        name = data.get("name", "")
        description = data.get("description", "")

        from opencut.core.ffmpeg_builder import save_filter_preset
        result = save_filter_preset(chain, name, description)
        return jsonify(result)
    except Exception as exc:
        return safe_error(exc, "filter_preset_save")


# ===================================================================
# 8. Smart Render
# ===================================================================

@platform_infra_bp.route("/smart-render", methods=["POST"])
@require_csrf
@async_job("smart-render")
def smart_render_route(job_id, filepath, data):
    """Smart render with partial re-encode."""
    changes = data.get("changes", [])
    codec = data.get("codec", "libx264")
    crf = safe_int(data.get("crf"), default=18, min_val=0, max_val=51)
    preset = data.get("preset", "medium")
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)

    if not changes:
        raise ValueError("changes list is required")

    out = None
    if output_dir:
        resolved_dir = _resolve_output_dir(filepath, output_dir)
        base = os.path.splitext(os.path.basename(filepath))[0]
        out = _unique_output_path(os.path.join(resolved_dir, f"{base}_smart.mp4"))

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.smart_render import smart_render
    return smart_render(
        video_path=filepath,
        changes=changes,
        output_path_str=out,
        codec=codec,
        crf=crf,
        preset=preset,
        on_progress=_on_progress,
    )


@platform_infra_bp.route("/smart-render/estimate", methods=["POST"])
@require_csrf
def smart_render_estimate():
    """Estimate smart render savings."""
    try:
        data = request.get_json(force=True)
        filepath = validate_filepath(data.get("filepath", ""))
        changes = data.get("changes", [])

        from opencut.core.smart_render import estimate_smart_render_savings
        result = estimate_smart_render_savings(filepath, changes)
        return jsonify(result)
    except Exception as exc:
        return safe_error(exc, "smart_render_estimate")


# ===================================================================
# 9. Render Cache
# ===================================================================

@platform_infra_bp.route("/cache/stats", methods=["GET"])
def cache_stats():
    """Get render cache statistics."""
    try:
        from opencut.core.render_cache import get_cache_stats
        stats = get_cache_stats()
        return jsonify(stats)
    except Exception as exc:
        return safe_error(exc, "cache_stats")


@platform_infra_bp.route("/cache/cleanup", methods=["POST"])
@require_csrf
def cache_cleanup():
    """Clean up the render cache."""
    try:
        data = request.get_json(force=True)
        max_size = safe_float(data.get("max_size_gb"), default=5.0, min_val=0.1)

        from opencut.core.render_cache import cleanup_cache
        result = cleanup_cache(max_size_gb=max_size)
        return jsonify(result)
    except Exception as exc:
        return safe_error(exc, "cache_cleanup")


@platform_infra_bp.route("/cache/invalidate", methods=["POST"])
@require_csrf
def cache_invalidate():
    """Invalidate downstream cache entries."""
    try:
        data = request.get_json(force=True)
        input_hash = data.get("input_hash", "")
        operation = data.get("operation", "")

        from opencut.core.render_cache import invalidate_downstream
        result = invalidate_downstream(input_hash, operation)
        return jsonify(result)
    except Exception as exc:
        return safe_error(exc, "cache_invalidate")


# ===================================================================
# 10. Timeline Diff
# ===================================================================

@platform_infra_bp.route("/timeline/diff", methods=["POST"])
@require_csrf
def timeline_diff_route():
    """Compare two timeline snapshots."""
    try:
        data = request.get_json(force=True)
        snapshot_a = data.get("snapshot_a", {})
        snapshot_b = data.get("snapshot_b", {})

        from opencut.core.timeline_diff import diff_timelines
        diff = diff_timelines(snapshot_a, snapshot_b)
        return jsonify({
            "summary": diff.summary,
            "total_changes": diff.total_changes,
            "added": diff.added_count,
            "removed": diff.removed_count,
            "modified": diff.modified_count,
            "moved": diff.moved_count,
            "changes": [
                {
                    "change_type": c.change_type,
                    "clip_id": c.clip_id,
                    "clip_name": c.clip_name,
                    "description": c.description,
                }
                for c in diff.changes
            ],
        })
    except Exception as exc:
        return safe_error(exc, "timeline_diff")


@platform_infra_bp.route("/timeline/diff/export", methods=["POST"])
@require_csrf
def timeline_diff_export():
    """Export a timeline diff report."""
    try:
        data = request.get_json(force=True)
        snapshot_a = data.get("snapshot_a", {})
        snapshot_b = data.get("snapshot_b", {})
        fmt = data.get("format", "json")

        from opencut.core.timeline_diff import diff_timelines, export_diff_report
        diff = diff_timelines(snapshot_a, snapshot_b)
        output = export_diff_report(diff, format=fmt)
        return jsonify({"output_path": output, "format": fmt})
    except Exception as exc:
        return safe_error(exc, "timeline_diff_export")


# ===================================================================
# 11. Branching Edit Workflows
# ===================================================================

@platform_infra_bp.route("/branches/create", methods=["POST"])
@require_csrf
def branch_create():
    """Create a new edit branch."""
    try:
        data = request.get_json(force=True)
        name = data.get("name", "")
        snapshot = data.get("snapshot", {})
        project_id = data.get("project_id", "default")
        parent = data.get("parent_branch", "")

        from opencut.core.edit_branches import create_branch
        branch = create_branch(name, snapshot, project_id, parent)
        return jsonify({
            "name": branch.name,
            "project_id": branch.project_id,
            "created_at": branch.created_at,
        })
    except Exception as exc:
        return safe_error(exc, "branch_create")


@platform_infra_bp.route("/branches/switch", methods=["POST"])
@require_csrf
def branch_switch():
    """Switch to a named branch."""
    try:
        data = request.get_json(force=True)
        name = data.get("name", "")
        project_id = data.get("project_id", "default")

        from opencut.core.edit_branches import switch_branch
        branch = switch_branch(name, project_id)
        return jsonify({
            "name": branch.name,
            "is_active": branch.is_active,
            "snapshot": branch.snapshot,
        })
    except Exception as exc:
        return safe_error(exc, "branch_switch")


@platform_infra_bp.route("/branches/merge", methods=["POST"])
@require_csrf
def branch_merge():
    """Merge two branches."""
    try:
        data = request.get_json(force=True)
        source = data.get("source", "")
        target = data.get("target", "")
        project_id = data.get("project_id", "default")

        from opencut.core.edit_branches import merge_branches
        result = merge_branches(source, target, project_id)
        return jsonify({
            "success": result.success,
            "message": result.message,
            "conflicts": result.conflicts,
            "auto_resolved": result.auto_resolved,
        })
    except Exception as exc:
        return safe_error(exc, "branch_merge")


@platform_infra_bp.route("/branches/list", methods=["GET"])
def branch_list():
    """List branches for a project."""
    try:
        project_id = request.args.get("project_id", "default")

        from opencut.core.edit_branches import list_branches
        branches = list_branches(project_id)
        return jsonify({
            "branches": [
                {
                    "name": b.name,
                    "parent_branch": b.parent_branch,
                    "is_active": b.is_active,
                    "commit_count": b.commit_count,
                    "created_at": b.created_at,
                }
                for b in branches
            ],
        })
    except Exception as exc:
        return safe_error(exc, "branch_list")


@platform_infra_bp.route("/branches/graph", methods=["GET"])
def branch_graph():
    """Get branch graph for visualization."""
    try:
        project_id = request.args.get("project_id", "default")

        from opencut.core.edit_branches import get_branch_graph
        graph = get_branch_graph(project_id)
        return jsonify(graph)
    except Exception as exc:
        return safe_error(exc, "branch_graph")


# ===================================================================
# 12. Frame.io Integration
# ===================================================================

@platform_infra_bp.route("/frameio/upload", methods=["POST"])
@require_csrf
@async_job("frameio-upload")
def frameio_upload(job_id, filepath, data):
    """Upload a video to Frame.io."""
    project_id = data.get("project_id", "")
    api_key = data.get("api_key", "")
    name = data.get("name")

    if not project_id:
        raise ValueError("Frame.io project_id is required")
    if not api_key:
        raise ValueError("Frame.io api_key is required")

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.frameio_integration import upload_to_frameio
    result = upload_to_frameio(
        video_path=filepath,
        project_id=project_id,
        api_key=api_key,
        name=name,
        on_progress=_on_progress,
    )
    return {
        "asset_id": result.asset_id,
        "name": result.name,
        "status": result.status,
    }


@platform_infra_bp.route("/frameio/comments", methods=["POST"])
@require_csrf
def frameio_comments():
    """Get comments from a Frame.io asset."""
    try:
        data = request.get_json(force=True)
        asset_id = data.get("asset_id", "")
        api_key = data.get("api_key", "")

        from opencut.core.frameio_integration import get_frameio_comments
        comments = get_frameio_comments(asset_id, api_key)
        return jsonify({
            "comments": [
                {
                    "comment_id": c.comment_id,
                    "text": c.text,
                    "author": c.author,
                    "timestamp": c.timestamp,
                    "completed": c.completed,
                }
                for c in comments
            ],
        })
    except Exception as exc:
        return safe_error(exc, "frameio_comments")


@platform_infra_bp.route("/frameio/resolve", methods=["POST"])
@require_csrf
def frameio_resolve_comment():
    """Resolve a Frame.io comment."""
    try:
        data = request.get_json(force=True)
        comment_id = data.get("comment_id", "")
        api_key = data.get("api_key", "")

        from opencut.core.frameio_integration import resolve_frameio_comment
        result = resolve_frameio_comment(comment_id, api_key)
        return jsonify(result)
    except Exception as exc:
        return safe_error(exc, "frameio_resolve")


@platform_infra_bp.route("/frameio/sync", methods=["POST"])
@require_csrf
def frameio_sync():
    """Two-way sync Frame.io comments."""
    try:
        data = request.get_json(force=True)
        asset_id = data.get("asset_id", "")
        api_key = data.get("api_key", "")
        local_comments = data.get("local_comments", [])

        from opencut.core.frameio_integration import sync_frameio_comments
        result = sync_frameio_comments(asset_id, api_key, local_comments)
        return jsonify(result)
    except Exception as exc:
        return safe_error(exc, "frameio_sync")


# ===================================================================
# 13. Waveform Timeline
# ===================================================================

@platform_infra_bp.route("/waveform/data", methods=["POST"])
@require_csrf
@async_job("waveform-data")
def waveform_data(job_id, filepath, data):
    """Generate waveform data for frontend rendering."""
    samples_per_second = safe_int(data.get("samples_per_second"), default=100, min_val=10, max_val=1000)
    normalize = safe_bool(data.get("normalize", True))

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.waveform_timeline import generate_waveform_data
    return generate_waveform_data(
        filepath, samples_per_second, normalize, on_progress=_on_progress,
    )


@platform_infra_bp.route("/waveform/image", methods=["POST"])
@require_csrf
@async_job("waveform-image")
def waveform_image(job_id, filepath, data):
    """Generate a waveform image."""
    width = safe_int(data.get("width"), default=1920, min_val=100, max_val=7680)
    height = safe_int(data.get("height"), default=200, min_val=50, max_val=1000)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.waveform_timeline import generate_waveform_image
    output = generate_waveform_image(
        filepath, width, height, on_progress=_on_progress,
    )
    return {"output_path": output}


@platform_infra_bp.route("/waveform/region", methods=["POST"])
@require_csrf
@async_job("waveform-region")
def waveform_region(job_id, filepath, data):
    """Get waveform data for a time region."""
    start = safe_float(data.get("start"), default=0)
    end = safe_float(data.get("end"), default=0)
    sample_count = safe_int(data.get("samples"), default=500, min_val=10, max_val=5000)

    if end <= start:
        raise ValueError("End time must be greater than start time")

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.waveform_timeline import get_waveform_region
    return get_waveform_region(
        filepath, start, end, sample_count, on_progress=_on_progress,
    )


# ===================================================================
# 14. Preview Server
# ===================================================================

@platform_infra_bp.route("/preview/frame", methods=["POST"])
@require_csrf
@async_job("preview-frame")
def preview_frame(job_id, filepath, data):
    """Extract a preview frame at a timestamp."""
    timestamp = safe_float(data.get("timestamp"), default=0)
    width = safe_int(data.get("width"), default=0)
    height = safe_int(data.get("height"), default=0)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.preview_server import extract_preview_frame
    output = extract_preview_frame(
        filepath, timestamp, width=width, height=height,
        on_progress=_on_progress,
    )
    return {"output_path": output}


@platform_infra_bp.route("/preview/clip", methods=["POST"])
@require_csrf
@async_job("preview-clip")
def preview_clip(job_id, filepath, data):
    """Generate a lightweight preview clip."""
    start = safe_float(data.get("start"), default=0)
    end = safe_float(data.get("end"), default=0)
    max_width = safe_int(data.get("max_width"), default=854)
    max_height = safe_int(data.get("max_height"), default=480)

    if end <= start:
        raise ValueError("End time must be greater than start time")

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.preview_server import generate_preview_clip
    return generate_preview_clip(
        filepath, start, end,
        max_width=max_width, max_height=max_height,
        on_progress=_on_progress,
    )


@platform_infra_bp.route("/preview/thumbnails", methods=["POST"])
@require_csrf
@async_job("preview-thumbs")
def preview_thumbnails(job_id, filepath, data):
    """Generate a strip of thumbnail images."""
    count = safe_int(data.get("count"), default=10, min_val=1, max_val=100)
    width = safe_int(data.get("width"), default=160, min_val=32, max_val=640)
    height = safe_int(data.get("height"), default=90, min_val=18, max_val=360)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.preview_server import generate_thumbnail_strip
    return generate_thumbnail_strip(
        filepath, count, width, height, on_progress=_on_progress,
    )
