"""
OpenCut Body Effects, Motion Transfer, Foley & Face Restoration Routes

Blueprint ``body_transfer_bp`` provides endpoints for:
  - Body-driven visual effects and keypoint detection
  - Full-body motion transfer (AnimateAnyone / stick figure)
  - AI foley sound generation from video
  - AI face restoration and enhancement
"""

import logging
import os

from flask import Blueprint

from opencut.helpers import _resolve_output_dir
from opencut.jobs import _update_job, async_job
from opencut.security import require_csrf, safe_float, safe_int, validate_filepath

logger = logging.getLogger("opencut")

body_transfer_bp = Blueprint("body_transfer", __name__)


# ===================================================================
# BODY EFFECTS
# ===================================================================


# ---------------------------------------------------------------------------
# POST /video/body-effects -- apply body-driven effects
# ---------------------------------------------------------------------------
@body_transfer_bp.route("/video/body-effects", methods=["POST"])
@require_csrf
@async_job("body_effects")
def body_effects_route(job_id, filepath, data):
    """Apply a body-driven visual effect to a video."""
    from opencut.core.body_effects import apply_body_effect

    effect_type = data.get("effect_type", "glow").strip()
    body_part = data.get("body_part", "nose").strip()
    output_dir = data.get("output_dir", "")

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    effective_dir = _resolve_output_dir(filepath, output_dir)
    base = os.path.splitext(os.path.basename(filepath))[0]
    ext = os.path.splitext(filepath)[1] or ".mp4"
    out_path = os.path.join(effective_dir, f"{base}_body_{effect_type}{ext}")

    result_path = apply_body_effect(
        video_path=filepath,
        effect_type=effect_type,
        body_part=body_part,
        output=out_path,
        on_progress=_progress,
    )

    return {
        "output_path": result_path,
        "effect_type": effect_type,
        "body_part": body_part,
    }


# ---------------------------------------------------------------------------
# POST /video/body-effects/detect -- detect body keypoints only
# ---------------------------------------------------------------------------
@body_transfer_bp.route("/video/body-effects/detect", methods=["POST"])
@require_csrf
@async_job("body_keypoints")
def body_effects_detect_route(job_id, filepath, data):
    """Detect body keypoints across all frames of a video."""
    from opencut.core.body_effects import detect_body_keypoints

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = detect_body_keypoints(filepath, on_progress=_progress)

    return {
        "fps": result.fps,
        "total_frames": result.total_frames,
        "frame_count": len(result.frames),
        "sample_frame": (
            {
                "frame_index": result.frames[0].frame_index,
                "timestamp": result.frames[0].timestamp,
                "keypoints": result.frames[0].keypoints,
            }
            if result.frames
            else None
        ),
    }


# ===================================================================
# MOTION TRANSFER
# ===================================================================


# ---------------------------------------------------------------------------
# POST /video/motion-transfer -- transfer motion between videos
# ---------------------------------------------------------------------------
@body_transfer_bp.route("/video/motion-transfer", methods=["POST"])
@require_csrf
@async_job("motion_transfer")
def motion_transfer_route(job_id, filepath, data):
    """Transfer motion from source video to a target person image."""
    from opencut.core.motion_transfer import transfer_motion

    target_image = data.get("target_image", "").strip()
    if not target_image:
        raise ValueError("No target_image path provided")
    target_image = validate_filepath(target_image)

    model = data.get("model", "auto").strip()
    output_dir = data.get("output_dir", "")

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    effective_dir = _resolve_output_dir(filepath, output_dir)
    base = os.path.splitext(os.path.basename(filepath))[0]
    ext = os.path.splitext(filepath)[1] or ".mp4"
    out_path = os.path.join(effective_dir, f"{base}_motion_transfer{ext}")

    result = transfer_motion(
        source_video=filepath,
        target_image=target_image,
        output=out_path,
        model=model,
        on_progress=_progress,
    )

    return {
        "output_path": result.output_path,
        "source_duration": result.source_duration,
        "frames_generated": result.frames_generated,
        "model_used": result.model_used,
    }


# ---------------------------------------------------------------------------
# POST /video/motion-transfer/preview -- preview motion transfer on frame
# ---------------------------------------------------------------------------
@body_transfer_bp.route("/video/motion-transfer/preview", methods=["POST"])
@require_csrf
@async_job("motion_transfer_preview")
def motion_transfer_preview_route(job_id, filepath, data):
    """Extract pose sequence for preview (first step of motion transfer)."""
    from opencut.core.motion_transfer import extract_pose_sequence

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = extract_pose_sequence(filepath, on_progress=_progress)

    return {
        "fps": result.fps,
        "duration": result.duration,
        "frame_count": len(result.poses),
        "sample_pose": result.poses[0] if result.poses else {},
        "preview": True,
    }


# ===================================================================
# FOLEY GENERATION
# ===================================================================


# ---------------------------------------------------------------------------
# POST /ai/foley/detect-cues -- detect foley cue points
# ---------------------------------------------------------------------------
@body_transfer_bp.route("/ai/foley/detect-cues", methods=["POST"])
@require_csrf
@async_job("foley_detect_cues")
def foley_detect_cues_route(job_id, filepath, data):
    """Detect foley cue points in a video."""
    from opencut.core.foley_gen import detect_foley_cues

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    cues = detect_foley_cues(filepath, on_progress=_progress)

    return {
        "cue_count": len(cues),
        "cues": [
            {
                "sound_type": c.sound_type,
                "start_time": c.start_time,
                "duration": c.duration,
                "confidence": c.confidence,
            }
            for c in cues
        ],
    }


# ---------------------------------------------------------------------------
# POST /ai/foley/generate -- generate and mix foley sounds
# ---------------------------------------------------------------------------
@body_transfer_bp.route("/ai/foley/generate", methods=["POST"])
@require_csrf
@async_job("foley_generate")
def foley_generate_route(job_id, filepath, data):
    """Generate and mix foley sounds onto a video."""
    from opencut.core.foley_gen import generate_foley

    output_dir = data.get("output_dir", "")
    mix_level = safe_float(data.get("mix_level", 0.3), 0.3, min_val=0.0, max_val=1.0)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    effective_dir = _resolve_output_dir(filepath, output_dir)
    base = os.path.splitext(os.path.basename(filepath))[0]
    ext = os.path.splitext(filepath)[1] or ".mp4"
    out_path = os.path.join(effective_dir, f"{base}_foley{ext}")

    result = generate_foley(
        video_path=filepath,
        output_path=out_path,
        mix_level=mix_level,
        on_progress=_progress,
    )

    return {
        "output_path": result.output_path,
        "sounds_generated": [
            {
                "sound_type": s.sound_type,
                "start_time": s.start_time,
                "duration": s.duration,
                "confidence": s.confidence,
            }
            for s in result.sounds_generated
        ],
        "duration": result.duration,
        "mix_level": result.mix_level,
    }


# ===================================================================
# FACE RESTORATION
# ===================================================================


# ---------------------------------------------------------------------------
# POST /video/face-restore -- restore/enhance faces in video
# ---------------------------------------------------------------------------
@body_transfer_bp.route("/video/face-restore", methods=["POST"])
@require_csrf
@async_job("face_restore")
def face_restore_route(job_id, filepath, data):
    """Restore and enhance faces across all frames of a video."""
    from opencut.core.face_restore import restore_faces

    output_dir = data.get("output_dir", "")
    strength = safe_float(data.get("strength", 1.0), 1.0, min_val=0.0, max_val=2.0)
    method = data.get("method", "ffmpeg").strip()

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    effective_dir = _resolve_output_dir(filepath, output_dir)
    base = os.path.splitext(os.path.basename(filepath))[0]
    ext = os.path.splitext(filepath)[1] or ".mp4"
    out_path = os.path.join(effective_dir, f"{base}_face_restored{ext}")

    result = restore_faces(
        video_path=filepath,
        output_path=out_path,
        strength=strength,
        method=method,
        on_progress=_progress,
    )

    return {
        "output_path": result.output_path,
        "faces_detected": result.faces_detected,
        "faces_restored": result.faces_restored,
        "method": result.method,
    }


# ---------------------------------------------------------------------------
# POST /video/face-restore/detect -- detect faces only
# ---------------------------------------------------------------------------
@body_transfer_bp.route("/video/face-restore/detect", methods=["POST"])
@require_csrf
@async_job("face_detect")
def face_restore_detect_route(job_id, filepath, data):
    """Detect faces in a video and return counts + bounding boxes."""
    from opencut.core.face_restore import detect_faces

    sample_frames = safe_int(data.get("sample_frames", 10), 10, min_val=1, max_val=100)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = detect_faces(
        video_path=filepath,
        sample_frames=sample_frames,
        on_progress=_progress,
    )

    return {
        "faces_detected": result.faces_detected,
        "avg_face_size": result.avg_face_size,
        "sample_frames": result.sample_frames,
        "boxes": [
            {
                "x": b.x,
                "y": b.y,
                "width": b.width,
                "height": b.height,
                "confidence": b.confidence,
            }
            for b in result.boxes
        ],
    }


# ---------------------------------------------------------------------------
# POST /video/face-restore/preview -- preview on single frame
# ---------------------------------------------------------------------------
@body_transfer_bp.route("/video/face-restore/preview", methods=["POST"])
@require_csrf
@async_job("face_restore_preview")
def face_restore_preview_route(job_id, filepath, data):
    """Preview face restoration on a single image frame."""
    from opencut.core.face_restore import restore_single_frame

    output_dir = data.get("output_dir", "")
    strength = safe_float(data.get("strength", 1.0), 1.0, min_val=0.0, max_val=2.0)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    _progress(10, "Restoring faces in preview frame...")

    effective_dir = _resolve_output_dir(filepath, output_dir)
    base = os.path.splitext(os.path.basename(filepath))[0]
    out_path = os.path.join(effective_dir, f"{base}_face_preview.jpg")

    result = restore_single_frame(
        image_path=filepath,
        output_path=out_path,
        strength=strength,
    )

    _progress(95, "Preview complete")

    return {
        "output_path": result.output_path,
        "faces_detected": result.faces_detected,
        "faces_restored": result.faces_restored,
        "method": result.method,
        "strength": strength,
        "preview": True,
    }
