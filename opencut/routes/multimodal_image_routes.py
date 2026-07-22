"""Image generation/editing and multimodal analysis routes formerly grouped in Wave L."""

from __future__ import annotations

from flask import jsonify

from opencut.errors import safe_error
from opencut.jobs import _update_job, async_job
from opencut.security import require_csrf, safe_float, safe_int

from .wave_l_contract import wave_l_bp

# =============================================================
# M2.4 — FLUX.1 Kontext Image Editing
# =============================================================

@wave_l_bp.route("/image/edit/kontext", methods=["POST"])
@require_csrf
@async_job("kontext_edit", filepath_required=True, filepath_param="image_path")
def route_kontext_edit(job_id, filepath, data):
    """Edit image using FLUX.1 Kontext natural language instructions.

    Body params:
      image_path           str    Path to source image (required).
      instruction          str    Edit instruction (required).
      num_inference_steps  int    Diffusion steps 4-100 (default 28).
      guidance_scale       float  CFG scale 1-15 (default 7.5).
      seed                 int    Random seed (-1 for random).
      output               str    Output path (optional).
    """
    from opencut.core import image_edit_kontext
    if not image_edit_kontext.check_kontext_available():
        raise RuntimeError(
            "FLUX Kontext not available. " + image_edit_kontext.INSTALL_HINT
        )

    instruction = str(data.get("instruction") or "").strip()
    if not instruction:
        raise ValueError("instruction is required")

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = image_edit_kontext.edit_image(
        image_path=filepath,
        instruction=instruction,
        num_inference_steps=safe_int(data.get("num_inference_steps"), 28, min_val=4, max_val=100),
        guidance_scale=safe_float(data.get("guidance_scale"), 7.5, min_val=1.0, max_val=15.0),
        seed=safe_int(data.get("seed"), -1),
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {
        "output": result.output,
        "instruction": result.instruction,
        "width": result.width,
        "height": result.height,
        "model": result.model,
        "generation_seconds": result.generation_seconds,
        "notes": result.notes,
    }


@wave_l_bp.route("/image/edit/kontext/info", methods=["GET"])
def route_kontext_info():
    """Return FLUX Kontext model availability and download status."""
    try:
        from opencut.core import image_edit_kontext
        return jsonify(image_edit_kontext.get_model_info())
    except Exception as exc:
        return safe_error(exc, "kontext_info")

# =============================================================
# N3.2 — Qwen2.5-VL Smart Timeline Analysis
# =============================================================

@wave_l_bp.route("/analyze/video/vl", methods=["POST"])
@require_csrf
@async_job("qwen_vl_analyze")
def route_vl_analyze(job_id, filepath, data):
    """Analyze video content using Qwen2.5-VL vision-language model."""
    from opencut.core import analyze_vl_qwen
    if not analyze_vl_qwen.check_qwen_vl_available():
        raise RuntimeError("Qwen VL deps not installed. " + analyze_vl_qwen.INSTALL_HINT)

    analysis_type = str(data.get("analysis_type") or "describe_scenes").strip()
    if analysis_type not in analyze_vl_qwen.ANALYSIS_TYPES:
        analysis_type = "describe_scenes"

    model = str(data.get("model") or "Qwen2.5-VL-7B").strip()
    if model not in analyze_vl_qwen.VL_MODELS:
        model = "Qwen2.5-VL-7B"

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = analyze_vl_qwen.analyze_video(
        video_path=filepath, analysis_type=analysis_type,
        custom_query=str(data.get("query") or data.get("custom_query") or ""),
        model=model,
        max_frames=safe_int(data.get("max_frames"), 16, min_val=1, max_val=64),
        on_progress=_prog,
    )
    return {k: getattr(result, k) for k in result.keys()}


@wave_l_bp.route("/analyze/video/vl/info", methods=["GET"])
def route_vl_info():
    try:
        from opencut.core import analyze_vl_qwen
        return jsonify({
            "available": analyze_vl_qwen.check_qwen_vl_available(),
            "models": analyze_vl_qwen.VL_MODELS,
            "analysis_types": analyze_vl_qwen.ANALYSIS_TYPES,
            "licence": "Apache-2.0",
            "install_hint": analyze_vl_qwen.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "vl_info")

# =============================================================
# P2.1 — HiDream-I1 SOTA T2I
# =============================================================

@wave_l_bp.route("/image/generate/hidream", methods=["POST"])
@require_csrf
@async_job("hidream_t2i", filepath_required=False)
def route_hidream_t2i(job_id, filepath, data):
    """Generate SOTA image using HiDream-I1 (17B)."""
    from opencut.core import t2i_hidream
    if not t2i_hidream.check_hidream_available():
        raise RuntimeError("HiDream deps not installed. " + t2i_hidream.INSTALL_HINT)
    prompt = str(data.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("prompt is required")
    variant = str(data.get("variant") or "fast").strip()
    if variant not in t2i_hidream.HIDREAM_VARIANTS:
        variant = "fast"
    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))
    result = t2i_hidream.generate(
        prompt=prompt, variant=variant,
        width=safe_int(data.get("width"), 1024, min_val=256, max_val=2048),
        height=safe_int(data.get("height"), 1024, min_val=256, max_val=2048),
        seed=safe_int(data.get("seed"), -1),
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {k: getattr(result, k) for k in result.keys()}


@wave_l_bp.route("/image/generate/hidream/info", methods=["GET"])
def route_hidream_info():
    try:
        from opencut.core import t2i_hidream
        return jsonify({"available": t2i_hidream.check_hidream_available(),
                        "variants": t2i_hidream.HIDREAM_VARIANTS,
                        "licence": "MIT + Meta Community Licence (Llama backbone)",
                        "install_hint": t2i_hidream.INSTALL_HINT})
    except Exception as exc:
        return safe_error(exc, "hidream_info")

# =============================================================
# P2.2 — HiDream-E1 Instruction Image Editing
# =============================================================

@wave_l_bp.route("/image/edit/hidream", methods=["POST"])
@require_csrf
@async_job("hidream_edit", filepath_required=True, filepath_param="image_path")
def route_hidream_edit(job_id, filepath, data):
    """Edit image using natural language instruction via HiDream-E1."""
    from opencut.core import img_edit_hidream
    if not img_edit_hidream.check_hidream_edit_available():
        raise RuntimeError("HiDream-E1 not installed. " + img_edit_hidream.INSTALL_HINT)
    instruction = str(data.get("instruction") or "").strip()
    if not instruction:
        raise ValueError("instruction is required")
    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))
    result = img_edit_hidream.edit(
        image_path=filepath, instruction=instruction,
        seed=safe_int(data.get("seed"), -1),
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {k: getattr(result, k) for k in result.keys()}


@wave_l_bp.route("/image/edit/hidream/info", methods=["GET"])
def route_hidream_edit_info():
    try:
        from opencut.core import img_edit_hidream
        return jsonify({"available": img_edit_hidream.check_hidream_edit_available(),
                        "licence": "MIT", "install_hint": img_edit_hidream.INSTALL_HINT})
    except Exception as exc:
        return safe_error(exc, "hidream_edit_info")

# =============================================================
# P2.3 — CogView4-6B Bilingual T2I
# =============================================================

@wave_l_bp.route("/image/generate/cogview4", methods=["POST"])
@require_csrf
@async_job("cogview4_t2i", filepath_required=False)
def route_cogview4_t2i(job_id, filepath, data):
    """Generate bilingual (EN+ZH) image using CogView4-6B."""
    from opencut.core import t2i_cogview4
    if not t2i_cogview4.check_cogview4_available():
        raise RuntimeError("CogView4 not installed. " + t2i_cogview4.INSTALL_HINT)
    prompt = str(data.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("prompt is required")
    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))
    result = t2i_cogview4.generate(
        prompt=prompt,
        width=safe_int(data.get("width"), 1024, min_val=256, max_val=2048),
        height=safe_int(data.get("height"), 1024, min_val=256, max_val=2048),
        guidance_scale=safe_float(data.get("guidance_scale"), 3.5, min_val=1.0, max_val=15.0),
        seed=safe_int(data.get("seed"), -1),
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {k: getattr(result, k) for k in result.keys()}


@wave_l_bp.route("/image/generate/cogview4/info", methods=["GET"])
def route_cogview4_info():
    try:
        from opencut.core import t2i_cogview4
        return jsonify({"available": t2i_cogview4.check_cogview4_available(),
                        "licence": "Apache-2.0", "min_vram_gb": 13,
                        "languages": ["en", "zh"],
                        "install_hint": t2i_cogview4.INSTALL_HINT})
    except Exception as exc:
        return safe_error(exc, "cogview4_info")

# =============================================================
# P3.1 — Qwen2.5-Omni Multimodal Video Narrator
# =============================================================

@wave_l_bp.route("/analyze/video/narrate", methods=["POST"])
@require_csrf
@async_job("omni_narrate")
def route_omni_narrate(job_id, filepath, data):
    """Watch video and generate written + spoken narration."""
    from opencut.core import multimodal_omni
    if not multimodal_omni.check_omni_available():
        raise RuntimeError("Qwen2.5-Omni not installed. " + multimodal_omni.INSTALL_HINT)
    style = str(data.get("style") or "documentary").strip()
    if style not in multimodal_omni.NARRATION_STYLES:
        style = "documentary"
    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))
    result = multimodal_omni.narrate_video(
        video_path=filepath, style=style,
        output_audio=str(data.get("output_audio") or "") or "",
        on_progress=_prog,
    )
    return {k: getattr(result, k) for k in result.keys()}


@wave_l_bp.route("/analyze/video/qa", methods=["POST"])
@require_csrf
@async_job("omni_qa")
def route_omni_qa(job_id, filepath, data):
    """Ask a question about a video."""
    from opencut.core import multimodal_omni
    if not multimodal_omni.check_omni_available():
        raise RuntimeError("Qwen2.5-Omni not installed. " + multimodal_omni.INSTALL_HINT)
    question = str(data.get("question") or "").strip()
    if not question:
        raise ValueError("question is required")
    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))
    result = multimodal_omni.video_qa(
        video_path=filepath, question=question, on_progress=_prog,
    )
    return {k: getattr(result, k) for k in result.keys()}


@wave_l_bp.route("/analyze/video/omni/info", methods=["GET"])
def route_omni_info():
    try:
        from opencut.core import multimodal_omni
        return jsonify({"available": multimodal_omni.check_omni_available(),
                        "narration_styles": multimodal_omni.NARRATION_STYLES,
                        "licence": "Apache-2.0",
                        "install_hint": multimodal_omni.INSTALL_HINT})
    except Exception as exc:
        return safe_error(exc, "omni_info")
