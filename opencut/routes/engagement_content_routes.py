"""
OpenCut Engagement & Content Routes

Endpoints for engagement prediction, caption styles, hook generation,
A/B variant testing, and Essential Graphics export.
"""

import logging

from flask import Blueprint, jsonify, request

from opencut.jobs import _update_job, async_job
from opencut.security import require_csrf, safe_float, safe_int

logger = logging.getLogger("opencut")

engagement_content_bp = Blueprint("engagement_content", __name__)


# ---------------------------------------------------------------------------
# POST /engagement/predict
# ---------------------------------------------------------------------------
@engagement_content_bp.route("/engagement/predict", methods=["POST"])
@require_csrf
@async_job("engagement_predict")
def engagement_predict(job_id, filepath, data):
    """Predict engagement and retention for a video."""
    from opencut.core.engagement_predict import predict_engagement

    transcript = data.get("transcript", "").strip() or None

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = predict_engagement(
        video_path=filepath,
        transcript=transcript,
        on_progress=_progress,
    )
    return result


# ---------------------------------------------------------------------------
# GET /captions/styles
# ---------------------------------------------------------------------------
@engagement_content_bp.route("/captions/styles", methods=["GET"])
def list_caption_styles():
    """List all available caption styles (synchronous)."""
    from opencut.core.caption_styles import CATEGORIES, get_available_styles

    styles = get_available_styles()
    category = request.args.get("category", "").strip()

    if category:
        styles = [s for s in styles if s.category == category]

    return jsonify({
        "styles": [s.to_dict() for s in styles],
        "count": len(styles),
        "categories": CATEGORIES,
    })


# ---------------------------------------------------------------------------
# POST /captions/style/preview
# ---------------------------------------------------------------------------
@engagement_content_bp.route("/captions/style/preview", methods=["POST"])
@require_csrf
@async_job("caption_style_preview", filepath_required=False)
def caption_style_preview(job_id, filepath, data):
    """Generate a PNG preview of a caption style."""
    from opencut.core.caption_styles import generate_style_preview

    style_id = data.get("style_id", "").strip()
    if not style_id:
        raise ValueError("style_id is required")

    sample_text = data.get("sample_text", "Hello World").strip()

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    _progress(10, f"Generating preview for style: {style_id}")
    preview_bytes = generate_style_preview(style_id, sample_text)
    _progress(100, "Preview generated")

    import base64
    return {
        "style_id": style_id,
        "preview_png_base64": base64.b64encode(preview_bytes).decode("ascii"),
        "sample_text": sample_text,
    }


# ---------------------------------------------------------------------------
# POST /captions/style/apply
# ---------------------------------------------------------------------------
@engagement_content_bp.route("/captions/style/apply", methods=["POST"])
@require_csrf
@async_job("caption_style_apply")
def caption_style_apply(job_id, filepath, data):
    """Apply a caption style to a video."""
    from opencut.core.caption_styles import apply_caption_style

    style_id = data.get("style_id", "").strip()
    if not style_id:
        raise ValueError("style_id is required")

    captions_data = data.get("captions", [])
    if not captions_data:
        raise ValueError("captions list is required")

    out_path = data.get("output_path", "").strip() or None

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result_path = apply_caption_style(
        video_path=filepath,
        captions_data=captions_data,
        style_id=style_id,
        output=out_path,
        on_progress=_progress,
    )
    return {"output_path": result_path, "style_id": style_id}


# ---------------------------------------------------------------------------
# POST /content/generate-hook
# ---------------------------------------------------------------------------
@engagement_content_bp.route("/content/generate-hook", methods=["POST"])
@require_csrf
@async_job("generate_hook")
def content_generate_hook(job_id, filepath, data):
    """Generate a compelling hook for a video."""
    from opencut.core.hook_generator import generate_hook

    transcript = data.get("transcript", "").strip() or None
    hook_type = data.get("hook_type", "auto").strip()

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = generate_hook(
        video_path=filepath,
        transcript=transcript,
        hook_type=hook_type,
        on_progress=_progress,
    )
    return result.to_dict()


# ---------------------------------------------------------------------------
# POST /content/apply-hook
# ---------------------------------------------------------------------------
@engagement_content_bp.route("/content/apply-hook", methods=["POST"])
@require_csrf
@async_job("apply_hook")
def content_apply_hook(job_id, filepath, data):
    """Apply a generated hook to a video."""
    from opencut.core.hook_generator import HookResult, apply_hook

    hook_data = data.get("hook", {})
    if not hook_data:
        raise ValueError("hook data is required")

    hook_result = HookResult(
        hook_text=hook_data.get("hook_text", ""),
        hook_type=hook_data.get("hook_type", "auto"),
        insertion_method=hook_data.get("insertion_method", "caption_overlay"),
        preview_text=hook_data.get("preview_text", ""),
        teaser_start=safe_float(hook_data.get("teaser_start"), 0.0),
        teaser_end=safe_float(hook_data.get("teaser_end"), 0.0),
    )
    out_path = data.get("output_path", "").strip() or None
    tts_voice = data.get("tts_voice", "").strip() or None

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result_path = apply_hook(
        video_path=filepath,
        hook_result=hook_result,
        output=out_path,
        tts_voice=tts_voice,
        on_progress=_progress,
    )
    return {"output_path": result_path}


# ---------------------------------------------------------------------------
# POST /content/ab-variants
# ---------------------------------------------------------------------------
@engagement_content_bp.route("/content/ab-variants", methods=["POST"])
@require_csrf
@async_job("ab_variants")
def content_ab_variants(job_id, filepath, data):
    """Generate A/B test variants of a video."""
    from opencut.core.ab_variant import generate_variants

    variant_count = safe_int(data.get("variant_count"), 3, min_val=1, max_val=5)
    vary_elements = data.get("vary_elements")  # None = default

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = generate_variants(
        video_path=filepath,
        variant_count=variant_count,
        vary_elements=vary_elements,
        on_progress=_progress,
    )
    return result.to_dict()


# ---------------------------------------------------------------------------
# POST /captions/export/essential-graphics
# ---------------------------------------------------------------------------
@engagement_content_bp.route("/captions/export/essential-graphics", methods=["POST"])
@require_csrf
@async_job("essential_graphics_export", filepath_required=False)
def captions_export_essential_graphics(job_id, filepath, data):
    """Export captions as Essential Graphics JSON for Premiere Pro."""
    from opencut.core.essential_graphics import export_as_mogrt_data

    captions = data.get("captions", [])
    if not captions:
        raise ValueError("captions list is required")

    style = data.get("style")  # Optional dict
    out_path = data.get("output_path", "").strip() or None

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = export_as_mogrt_data(
        captions=captions,
        style=style,
        out_path=out_path,
        on_progress=_progress,
    )
    return result.to_dict()


# ---------------------------------------------------------------------------
# POST /captions/export/premiere-xml
# ---------------------------------------------------------------------------
@engagement_content_bp.route("/captions/export/premiere-xml", methods=["POST"])
@require_csrf
@async_job("premiere_xml_export", filepath_required=False)
def captions_export_premiere_xml(job_id, filepath, data):
    """Export captions as Premiere Pro XML."""
    from opencut.core.essential_graphics import (
        export_as_srt_for_premiere,
        generate_premiere_caption_xml,
    )

    captions = data.get("captions", [])
    if not captions:
        raise ValueError("captions list is required")

    export_format = data.get("format", "xml").strip()
    style = data.get("style")
    out_path = data.get("output_path", "").strip() or None
    fps = safe_float(data.get("fps"), 30.0, min_val=1.0, max_val=240.0)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    if export_format == "srt":
        _progress(10, "Exporting SRT for Premiere...")
        result_path = export_as_srt_for_premiere(
            captions=captions,
            out_path=out_path,
            on_progress=_progress,
        )
        return {"output_path": result_path, "format": "srt"}

    # Default: XML
    _progress(10, "Generating Premiere XML...")
    xml_str = generate_premiere_caption_xml(
        captions=captions,
        style=style,
        fps=fps,
    )
    _progress(80, "Writing XML file...")

    import os
    if not out_path:
        out_path = os.path.join(
            os.path.expanduser("~"), ".opencut", "premiere_captions.xml"
        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xml_str)

    _progress(100, "Premiere XML export complete")
    return {"output_path": out_path, "format": "xml"}
