"""Primary image-to-video and Wan generation routes formerly grouped in Wave L."""

from __future__ import annotations

from flask import jsonify

from opencut.errors import safe_error
from opencut.jobs import _update_job, async_job
from opencut.security import require_csrf, safe_bool, safe_float, safe_int

from .wave_l_contract import wave_l_bp

# =============================================================
# L2.1 — FramePack Image-to-Video
# =============================================================

@wave_l_bp.route("/generate/framepack", methods=["POST"])
@require_csrf
@async_job("framepack_i2v", filepath_required=True, filepath_param="image_path")
def route_framepack_generate(job_id, filepath, data):
    """Generate video from a single image using FramePack (6 GB VRAM).

    Body params:
      image_path       str    Path to source image (required).
      prompt           str    Text description of motion/action.
      duration         float  Target seconds, max 60 (default 5).
      fps              float  Output frame rate (default 24).
      model            str    framepack-standard or framepack-fast.
      negative_prompt  str    Things to avoid.
      seed             int    Random seed (-1 for random).
      output           str    Output MP4 path (optional).
    """
    from opencut.core import gen_video_framepack
    if not gen_video_framepack.check_framepack_available():
        raise RuntimeError(
            "FramePack is not installed. " + gen_video_framepack.INSTALL_HINT
        )

    model = str(data.get("model") or "framepack-standard").strip()
    if model not in gen_video_framepack.FRAMEPACK_MODELS:
        model = "framepack-standard"

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = gen_video_framepack.generate(
        image_path=filepath,
        prompt=str(data.get("prompt") or ""),
        duration=safe_float(data.get("duration"), 5.0, min_val=1.0, max_val=60.0),
        fps=safe_float(data.get("fps"), 24.0, min_val=8.0, max_val=60.0),
        model=model,
        negative_prompt=str(data.get("negative_prompt") or ""),
        seed=safe_int(data.get("seed"), -1),
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {
        "output": result.output,
        "duration_seconds": result.duration_seconds,
        "fps": result.fps,
        "width": result.width,
        "height": result.height,
        "model": result.model,
        "generation_seconds": result.generation_seconds,
        "prompt": result.prompt,
        "notes": result.notes,
    }


@wave_l_bp.route("/generate/framepack/info", methods=["GET"])
def route_framepack_info():
    """Return FramePack availability and model catalogue."""
    try:
        from opencut.core import gen_video_framepack
        return jsonify({
            "available": gen_video_framepack.check_framepack_available(),
            "models": gen_video_framepack.FRAMEPACK_MODELS,
            "licence": "Apache-2.0",
            "min_vram_gb": 6,
            "max_duration_seconds": 60,
            "install_hint": gen_video_framepack.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "framepack_info")

# =============================================================
# M2.1 — Wan2.2 T2V / I2V / TI2V Video Generation
# =============================================================

@wave_l_bp.route("/generate/wan2.2/t2v", methods=["POST"])
@require_csrf
@async_job("wan22_t2v", filepath_required=False)
def route_wan22_t2v(job_id, filepath, data):
    """Generate video from text using Wan2.2 (MoE, cinematic aesthetics).

    Body params:
      prompt           str    Video description (required).
      duration         float  Target seconds, max 16 (default 5).
      model            str    ti2v-5b (consumer) / t2v-14b / i2v-14b.
      negative_prompt  str    Things to avoid.
      fps              float  Frame rate (default 24).
      width            int    Output width (default 1280).
      height           int    Output height (default 720).
      seed             int    Random seed (-1 for random).
      offload_model    bool   CPU offload for VRAM savings (default true).
      output           str    Output MP4 path (optional).
    """
    from opencut.core import gen_video_wan22
    if not gen_video_wan22.check_wan22_available():
        raise RuntimeError(
            "Wan2.2 not installed. " + gen_video_wan22.INSTALL_HINT
        )

    prompt = str(data.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("prompt is required")

    model = str(data.get("model") or "ti2v-5b").strip()
    if model not in gen_video_wan22.WAN22_MODELS:
        model = "ti2v-5b"

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = gen_video_wan22.generate_t2v(
        prompt=prompt,
        duration=safe_float(data.get("duration"), 5.0, min_val=1.0, max_val=16.0),
        model=model,
        negative_prompt=str(data.get("negative_prompt") or ""),
        fps=safe_float(data.get("fps"), 24.0, min_val=8.0, max_val=60.0),
        width=safe_int(data.get("width"), 1280, min_val=256, max_val=2048),
        height=safe_int(data.get("height"), 720, min_val=256, max_val=2048),
        seed=safe_int(data.get("seed"), -1),
        offload_model=safe_bool(data.get("offload_model"), True),
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {
        "output": result.output,
        "mode": result.mode,
        "model": result.model,
        "duration_seconds": result.duration_seconds,
        "fps": result.fps,
        "width": result.width,
        "height": result.height,
        "generation_seconds": result.generation_seconds,
        "prompt": result.prompt,
        "notes": result.notes,
    }


@wave_l_bp.route("/generate/wan2.2/i2v", methods=["POST"])
@require_csrf
@async_job("wan22_i2v", filepath_required=True, filepath_param="image_path")
def route_wan22_i2v(job_id, filepath, data):
    """Generate video from image + optional text using Wan2.2.

    Body params:
      image_path     str    Source image (required).
      prompt         str    Optional motion description.
      duration       float  Target seconds, max 16.
      model          str    Model variant.
      seed           int    Random seed.
      offload_model  bool   CPU offload.
      output         str    Output MP4 path.
    """
    from opencut.core import gen_video_wan22
    if not gen_video_wan22.check_wan22_available():
        raise RuntimeError(
            "Wan2.2 not installed. " + gen_video_wan22.INSTALL_HINT
        )

    model = str(data.get("model") or "ti2v-5b").strip()
    if model not in gen_video_wan22.WAN22_MODELS:
        model = "ti2v-5b"

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = gen_video_wan22.generate_i2v(
        image_path=filepath,
        prompt=str(data.get("prompt") or ""),
        duration=safe_float(data.get("duration"), 5.0, min_val=1.0, max_val=16.0),
        model=model,
        seed=safe_int(data.get("seed"), -1),
        offload_model=safe_bool(data.get("offload_model"), True),
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {
        "output": result.output,
        "mode": result.mode,
        "model": result.model,
        "duration_seconds": result.duration_seconds,
        "fps": result.fps,
        "generation_seconds": result.generation_seconds,
        "prompt": result.prompt,
        "notes": result.notes,
    }


@wave_l_bp.route("/generate/wan2.2/info", methods=["GET"])
def route_wan22_info():
    """Return Wan2.2 model availability and variants."""
    try:
        from opencut.core import gen_video_wan22
        return jsonify({
            "available": gen_video_wan22.check_wan22_available(),
            "models": gen_video_wan22.WAN22_MODELS,
            "modes": gen_video_wan22.WAN22_MODES,
            "licence": "Apache-2.0",
            "max_duration_seconds": 16,
            "consumer_model": "ti2v-5b",
            "install_hint": gen_video_wan22.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "wan22_info")

# =============================================================
# N1.1 — FastVideo Inference Acceleration
# =============================================================

@wave_l_bp.route("/generate/wan2.2/fast", methods=["POST"])
@require_csrf
@async_job("fastvideo", filepath_required=False)
def route_fastvideo(job_id, filepath, data):
    """Generate video using FastVideo distilled Wan2.2 (>50x speedup)."""
    from opencut.core import gen_video_fastvideo
    if not gen_video_fastvideo.check_fastvideo_available():
        raise RuntimeError("FastVideo not installed. " + gen_video_fastvideo.INSTALL_HINT)

    prompt = str(data.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("prompt is required")

    model = str(data.get("model") or "FastWan2.2-TI2V-5B").strip()
    if model not in gen_video_fastvideo.FAST_MODELS:
        model = "FastWan2.2-TI2V-5B"

    image = str(data.get("image_path") or "").strip()
    if image:
        from opencut.security import validate_filepath
        image = validate_filepath(image)

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = gen_video_fastvideo.generate(
        prompt=prompt, model=model, image_path=image,
        duration=safe_float(data.get("duration"), 5.0, min_val=1.0, max_val=10.0),
        negative_prompt=str(data.get("negative_prompt") or ""),
        seed=safe_int(data.get("seed"), -1),
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {k: getattr(result, k) for k in result.keys()}


@wave_l_bp.route("/generate/wan2.2/fast/info", methods=["GET"])
def route_fastvideo_info():
    try:
        from opencut.core import gen_video_fastvideo
        return jsonify({
            "available": gen_video_fastvideo.check_fastvideo_available(),
            "models": gen_video_fastvideo.FAST_MODELS,
            "licence": "Apache-2.0",
            "install_hint": gen_video_fastvideo.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "fastvideo_info")

# =============================================================
# N1.2 — LightX2V Quantized I2V
# =============================================================

@wave_l_bp.route("/generate/wan2.2/i2v/quantized", methods=["POST"])
@require_csrf
@async_job("lightx2v", filepath_required=True, filepath_param="image_path")
def route_lightx2v(job_id, filepath, data):
    """Generate I2V with quantized/distilled Wan2.2 A14B (24 GB VRAM)."""
    from opencut.core import gen_video_lightx2v
    if not gen_video_lightx2v.check_lightx2v_available():
        raise RuntimeError("LightX2V not installed. " + gen_video_lightx2v.INSTALL_HINT)

    quant = str(data.get("quant") or "fp8").strip()
    if quant not in gen_video_lightx2v.QUANT_MODES:
        quant = "fp8"

    steps = safe_int(data.get("steps"), 4)
    if steps not in gen_video_lightx2v.STEP_PRESETS:
        steps = 4

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = gen_video_lightx2v.generate_i2v(
        image_path=filepath, prompt=str(data.get("prompt") or ""),
        duration=safe_float(data.get("duration"), 5.0, min_val=1.0, max_val=16.0),
        quant=quant, steps=steps, seed=safe_int(data.get("seed"), -1),
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {k: getattr(result, k) for k in result.keys()}


@wave_l_bp.route("/generate/wan2.2/i2v/quantization/info", methods=["GET"])
def route_lightx2v_info():
    try:
        from opencut.core import gen_video_lightx2v
        return jsonify({
            "available": gen_video_lightx2v.check_lightx2v_available(),
            "quant_modes": gen_video_lightx2v.QUANT_MODES,
            "step_presets": {str(k): v for k, v in gen_video_lightx2v.STEP_PRESETS.items()},
            "licence": "Apache-2.0",
            "install_hint": gen_video_lightx2v.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "lightx2v_info")
