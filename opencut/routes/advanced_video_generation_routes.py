"""Advanced local video generation backends formerly grouped in Wave L."""

from __future__ import annotations

from flask import jsonify

from opencut.errors import safe_error
from opencut.jobs import _update_job, async_job
from opencut.security import require_csrf, safe_float, safe_int

from .wave_l_contract import wave_l_bp

# =============================================================
# N3.1 — CogVideoX T2V + I2V
# =============================================================

@wave_l_bp.route("/generate/cogvideox", methods=["POST"])
@require_csrf
@async_job("cogvideox_t2v", filepath_required=False)
def route_cogvideox_t2v(job_id, filepath, data):
    """Generate video from text using CogVideoX (12 GB VRAM)."""
    from opencut.core import gen_video_cogvideox
    if not gen_video_cogvideox.check_cogvideox_available():
        raise RuntimeError("CogVideoX deps not installed. " + gen_video_cogvideox.INSTALL_HINT)

    prompt = str(data.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("prompt is required")

    model = str(data.get("model") or "CogVideoX-5B").strip()
    if model not in gen_video_cogvideox.COGVIDEOX_MODELS:
        model = "CogVideoX-5B"

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = gen_video_cogvideox.generate_t2v(
        prompt=prompt, model=model,
        num_frames=safe_int(data.get("num_frames"), 49, min_val=8, max_val=81),
        guidance_scale=safe_float(data.get("guidance_scale"), 6.0, min_val=1.0, max_val=15.0),
        seed=safe_int(data.get("seed"), -1),
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {k: getattr(result, k) for k in result.keys()}


@wave_l_bp.route("/generate/cogvideox/i2v", methods=["POST"])
@require_csrf
@async_job("cogvideox_i2v", filepath_required=True, filepath_param="image_path")
def route_cogvideox_i2v(job_id, filepath, data):
    """Generate video from image using CogVideoX I2V."""
    from opencut.core import gen_video_cogvideox
    if not gen_video_cogvideox.check_cogvideox_available():
        raise RuntimeError("CogVideoX deps not installed. " + gen_video_cogvideox.INSTALL_HINT)

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = gen_video_cogvideox.generate_i2v(
        image_path=filepath, prompt=str(data.get("prompt") or ""),
        model=str(data.get("model") or "CogVideoX1.5-5B"),
        seed=safe_int(data.get("seed"), -1),
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {k: getattr(result, k) for k in result.keys()}


@wave_l_bp.route("/generate/cogvideox/info", methods=["GET"])
def route_cogvideox_info():
    try:
        from opencut.core import gen_video_cogvideox
        return jsonify({
            "available": gen_video_cogvideox.check_cogvideox_available(),
            "models": gen_video_cogvideox.COGVIDEOX_MODELS,
            "licence": "Apache-2.0",
            "min_vram_gb": 12,
            "install_hint": gen_video_cogvideox.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "cogvideox_info")

# =============================================================
# O2.1 — LTX-Video T2V + I2V + Extension
# =============================================================

@wave_l_bp.route("/generate/ltxv/t2v", methods=["POST"])
@require_csrf
@async_job("ltxv_t2v", filepath_required=False)
def route_ltxv_t2v(job_id, filepath, data):
    """Generate video from text using LTX-Video (fastest Apache-2 DiT)."""
    from opencut.core import gen_video_ltxv
    if not gen_video_ltxv.check_ltxv_available():
        raise RuntimeError("LTX-Video deps not installed. " + gen_video_ltxv.INSTALL_HINT)

    prompt = str(data.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("prompt is required")

    model = str(data.get("model") or "LTXV-2B").strip()
    if model not in gen_video_ltxv.LTXV_MODELS:
        model = "LTXV-2B"

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = gen_video_ltxv.generate_t2v(
        prompt=prompt, model=model,
        duration=safe_float(data.get("duration"), 5.0, min_val=1.0, max_val=60.0),
        negative_prompt=str(data.get("negative_prompt") or ""),
        seed=safe_int(data.get("seed"), -1),
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {k: getattr(result, k) for k in result.keys()}


@wave_l_bp.route("/generate/ltxv/i2v", methods=["POST"])
@require_csrf
@async_job("ltxv_i2v", filepath_required=True, filepath_param="image_path")
def route_ltxv_i2v(job_id, filepath, data):
    """Generate video from image using LTX-Video I2V."""
    from opencut.core import gen_video_ltxv
    if not gen_video_ltxv.check_ltxv_available():
        raise RuntimeError("LTX-Video deps not installed. " + gen_video_ltxv.INSTALL_HINT)

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = gen_video_ltxv.generate_i2v(
        image_path=filepath, prompt=str(data.get("prompt") or ""),
        model=str(data.get("model") or "LTXV-2B"),
        duration=safe_float(data.get("duration"), 5.0, min_val=1.0, max_val=60.0),
        seed=safe_int(data.get("seed"), -1),
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {k: getattr(result, k) for k in result.keys()}


@wave_l_bp.route("/generate/ltxv/extend", methods=["POST"])
@require_csrf
@async_job("ltxv_extend")
def route_ltxv_extend(job_id, filepath, data):
    """Extend a video forward or backward in time via LTX-Video."""
    from opencut.core import gen_video_ltxv
    if not gen_video_ltxv.check_ltxv_available():
        raise RuntimeError("LTX-Video deps not installed. " + gen_video_ltxv.INSTALL_HINT)

    direction = str(data.get("direction") or "forward").strip()
    if direction not in ("forward", "backward"):
        direction = "forward"

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = gen_video_ltxv.extend_video(
        video_path=filepath, direction=direction,
        duration_sec=safe_float(data.get("duration"), 3.0, min_val=1.0, max_val=30.0),
        model=str(data.get("model") or "LTXV-2B"),
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {k: getattr(result, k) for k in result.keys()}


@wave_l_bp.route("/generate/ltxv/info", methods=["GET"])
def route_ltxv_info():
    try:
        from opencut.core import gen_video_ltxv
        return jsonify({
            "available": gen_video_ltxv.check_ltxv_available(),
            "models": gen_video_ltxv.LTXV_MODELS,
            "licence": "Apache-2.0",
            "max_duration_seconds": 60,
            "install_hint": gen_video_ltxv.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "ltxv_info")

# =============================================================
# P1.2 — Allegro Lightweight T2V + TI2V
# =============================================================

@wave_l_bp.route("/generate/allegro/t2v", methods=["POST"])
@require_csrf
@async_job("allegro_t2v", filepath_required=False)
def route_allegro_t2v(job_id, filepath, data):
    """Generate video using Allegro (9.3 GB VRAM)."""
    from opencut.core import gen_video_allegro
    if not gen_video_allegro.check_allegro_available():
        raise RuntimeError("Allegro not installed. " + gen_video_allegro.INSTALL_HINT)
    prompt = str(data.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("prompt is required")
    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))
    result = gen_video_allegro.generate_t2v(
        prompt=prompt, seed=safe_int(data.get("seed"), -1),
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {k: getattr(result, k) for k in result.keys()}


@wave_l_bp.route("/generate/allegro/ti2v", methods=["POST"])
@require_csrf
@async_job("allegro_ti2v", filepath_required=True, filepath_param="first_frame")
def route_allegro_ti2v(job_id, filepath, data):
    """Generate video from first+last frame interpolation."""
    from opencut.core import gen_video_allegro
    if not gen_video_allegro.check_allegro_available():
        raise RuntimeError("Allegro not installed. " + gen_video_allegro.INSTALL_HINT)
    last = str(data.get("last_frame") or "").strip()
    if last:
        from opencut.security import validate_filepath
        last = validate_filepath(last)
    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))
    result = gen_video_allegro.generate_ti2v(
        first_frame=filepath, last_frame=last,
        prompt=str(data.get("prompt") or ""),
        seed=safe_int(data.get("seed"), -1),
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {k: getattr(result, k) for k in result.keys()}


@wave_l_bp.route("/generate/allegro/info", methods=["GET"])
def route_allegro_info():
    try:
        from opencut.core import gen_video_allegro
        return jsonify({"available": gen_video_allegro.check_allegro_available(),
                        "models": gen_video_allegro.ALLEGRO_MODELS,
                        "licence": "Apache-2.0", "min_vram_gb": 9.3,
                        "install_hint": gen_video_allegro.INSTALL_HINT})
    except Exception as exc:
        return safe_error(exc, "allegro_info")

# =============================================================
# P3.2 — Open-Sora 2.0 High-Quality T2V
# =============================================================

@wave_l_bp.route("/generate/opensora2", methods=["POST"])
@require_csrf
@async_job("opensora2_t2v", filepath_required=False)
def route_opensora2(job_id, filepath, data):
    """Generate SOTA quality video using Open-Sora 2.0 (11B)."""
    from opencut.core import gen_video_opensora2
    if not gen_video_opensora2.check_opensora2_available():
        raise RuntimeError("Open-Sora not installed. " + gen_video_opensora2.INSTALL_HINT)
    prompt = str(data.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("prompt is required")
    model = str(data.get("model") or "11b").strip()
    if model not in gen_video_opensora2.OPENSORA_MODELS:
        model = "11b"
    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))
    result = gen_video_opensora2.generate(
        prompt=prompt, model=model,
        duration=safe_float(data.get("duration"), 5.0, min_val=1.0, max_val=10.0),
        seed=safe_int(data.get("seed"), -1),
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {k: getattr(result, k) for k in result.keys()}


@wave_l_bp.route("/generate/opensora2/info", methods=["GET"])
def route_opensora2_info():
    try:
        from opencut.core import gen_video_opensora2
        return jsonify({"available": gen_video_opensora2.check_opensora2_available(),
                        "models": gen_video_opensora2.OPENSORA_MODELS,
                        "licence": "Apache-2.0",
                        "install_hint": gen_video_opensora2.INSTALL_HINT})
    except Exception as exc:
        return safe_error(exc, "opensora2_info")
