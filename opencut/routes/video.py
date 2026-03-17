"""
OpenCut Video Routes

Watermark, scenes, export, effects, AI, face tools, style transfer,
speed, LUT, compositing, titles, transitions, particles, reframe,
merge, trim, preview.
"""

import json
import logging
import os
import subprocess as _sp
import tempfile
import threading
import time
import uuid

from flask import Blueprint, jsonify, request

from opencut.checks import check_watermark_available
from opencut.helpers import (
    _get_file_duration,
    _resolve_output_dir,
    _run_ffmpeg_with_progress,
    _unique_output_path,
)
from opencut.jobs import (
    MAX_BATCH_FILES,
    _is_cancelled,
    _new_job,
    _register_job_process,
    _safe_error,
    _unregister_job_process,
    _update_job,
    job_lock,
    jobs,
)
from opencut.security import (
    require_csrf,
    require_rate_limit,
    safe_float,
    safe_int,
    safe_pip_install,
    validate_filepath,
    validate_path,
)

logger = logging.getLogger("opencut")

video_bp = Blueprint("video", __name__)

# ---------------------------------------------------------------------------
# Module-level model directory env vars (mirrored from server.py)
# ---------------------------------------------------------------------------
FLORENCE_MODEL_DIR = os.environ.get("OPENCUT_FLORENCE_DIR", None)
LAMA_MODEL_DIR = os.environ.get("OPENCUT_LAMA_DIR", None)


# ---------------------------------------------------------------------------
# Watermark Removal (Florence-2 + LaMA)
# ---------------------------------------------------------------------------
@video_bp.route("/video/watermark", methods=["POST"])
@require_csrf
def video_watermark():
    """Remove watermarks from video or image using AI."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")
    max_bbox_percent = safe_int(data.get("max_bbox_percent", 10), 10)
    detection_prompt = data.get("detection_prompt", "watermark")
    detection_skip = safe_int(data.get("detection_skip", 3), 3)
    transparent = data.get("transparent", False)
    preview = data.get("preview", False)
    auto_import = data.get("auto_import", True)

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400

    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if not check_watermark_available():
        return jsonify({"error": "Watermark remover not installed. Please install it first."}), 400

    job_id = _new_job("watermark", filepath)

    def _process():
        try:
            import subprocess

            effective_dir = _resolve_output_dir(filepath, output_dir)
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            ext = os.path.splitext(filepath)[1].lower()
            temp_video = None

            _update_job(job_id, progress=5, message="Initializing AI models...")

            # Determine if video or image
            video_exts = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v', '.flv', '.wmv'}
            is_video = ext in video_exts

            # Build output path
            suffix = "_preview" if preview else "_clean"
            if is_video:
                out_path = os.path.join(effective_dir, f"{base_name}{suffix}.mp4")
            else:
                out_ext = ".png" if transparent else ext
                out_path = os.path.join(effective_dir, f"{base_name}{suffix}{out_ext}")

            _update_job(job_id, progress=10, message="Running watermark detection...")

            # Build command for the watermark remover
            try:
                import cv2
                import numpy as np
                import torch
                from PIL import Image
                from simple_lama_inpainting import SimpleLama
                from transformers import AutoModelForCausalLM, AutoProcessor

                device = "cuda" if torch.cuda.is_available() else "cpu"

                # Load Florence-2 model for detection
                _update_job(job_id, progress=15, message="Loading Florence-2 model...")

                # Use bundled model path if available
                if FLORENCE_MODEL_DIR and os.path.isdir(FLORENCE_MODEL_DIR):
                    model_id = FLORENCE_MODEL_DIR
                    logger.info(f"Using bundled Florence model: {model_id}")
                else:
                    model_id = "microsoft/Florence-2-base"

                florence_model = AutoModelForCausalLM.from_pretrained(
                    model_id, trust_remote_code=True
                ).to(device)
                florence_processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

                # Load LaMA model for inpainting
                _update_job(job_id, progress=25, message="Loading LaMA inpainting model...")

                # SimpleLama uses its own model path, but we can set TORCH_HOME
                if LAMA_MODEL_DIR and os.path.isdir(LAMA_MODEL_DIR):
                    os.environ["LAMA_MODEL"] = os.path.join(LAMA_MODEL_DIR, "big-lama.pt")
                    logger.info(f"Using bundled LaMA model: {LAMA_MODEL_DIR}")

                lama = SimpleLama()

                def detect_watermarks(image):
                    """Detect watermarks using Florence-2."""
                    task = "<OPEN_VOCABULARY_DETECTION>"
                    prompt = task + detection_prompt

                    inputs = florence_processor(text=prompt, images=image, return_tensors="pt").to(device)  # noqa: F821

                    with torch.no_grad():
                        generated_ids = florence_model.generate(  # noqa: F821
                            input_ids=inputs["input_ids"],
                            pixel_values=inputs["pixel_values"],
                            max_new_tokens=1024,
                            do_sample=False,
                            num_beams=3
                        )

                    generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]  # noqa: F821
                    parsed = florence_processor.post_process_generation(  # noqa: F821
                        generated_text, task=task, image_size=(image.width, image.height)
                    )

                    bboxes = []
                    if task in parsed:
                        result = parsed[task]
                        if "bboxes" in result:
                            img_area = image.width * image.height
                            max_area = img_area * (max_bbox_percent / 100.0)

                            for bbox in result["bboxes"]:
                                x1, y1, x2, y2 = bbox
                                bbox_area = (x2 - x1) * (y2 - y1)
                                if bbox_area <= max_area:
                                    bboxes.append([int(x1), int(y1), int(x2), int(y2)])

                    return bboxes

                def create_mask(image_size, bboxes, expand=5):
                    """Create binary mask from bounding boxes."""
                    mask = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
                    for bbox in bboxes:
                        x1, y1, x2, y2 = bbox
                        x1 = max(0, x1 - expand)
                        y1 = max(0, y1 - expand)
                        x2 = min(image_size[0], x2 + expand)
                        y2 = min(image_size[1], y2 + expand)
                        mask[y1:y2, x1:x2] = 255
                    return mask

                def process_image(img_pil):
                    """Process a single image."""
                    bboxes = detect_watermarks(img_pil)

                    if not bboxes:
                        return img_pil, False

                    if preview:
                        # Draw detection boxes
                        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                        for bbox in bboxes:
                            cv2.rectangle(img_cv, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                        return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)), True

                    if transparent:
                        # Make watermark area transparent
                        img_rgba = img_pil.convert("RGBA")
                        img_array = np.array(img_rgba)
                        mask = create_mask((img_pil.width, img_pil.height), bboxes)
                        img_array[:, :, 3] = np.where(mask > 0, 0, 255)
                        return Image.fromarray(img_array), True

                    # Inpaint using LaMA
                    mask = create_mask((img_pil.width, img_pil.height), bboxes)
                    mask_pil = Image.fromarray(mask)
                    result = lama(img_pil, mask_pil)  # noqa: F821
                    return result, True

                if is_video:
                    # Process video
                    _update_job(job_id, progress=30, message="Processing video frames...")

                    cap = cv2.VideoCapture(filepath)
                    try:
                        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                        total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                        # Create temp output
                        _ntf = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
                        temp_video = _ntf.name
                        _ntf.close()
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out_video = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))

                        # Two-pass: detect watermarks on key frames, then apply to all
                        detected_masks = {}
                        frame_idx = 0

                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break

                            # Detect on every N frames
                            if frame_idx % detection_skip == 0:
                                img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                                bboxes = detect_watermarks(img_pil)
                                if bboxes:
                                    detected_masks[frame_idx] = bboxes

                            progress = 30 + int((frame_idx / total_frames) * 30)
                            if frame_idx % 30 == 0:
                                _update_job(job_id, progress=progress, message=f"Detecting... frame {frame_idx}/{total_frames}")

                            frame_idx += 1
                    finally:
                        cap.release()

                    # Second pass: process all frames
                    _update_job(job_id, progress=60, message="Inpainting frames...")

                    cap = cv2.VideoCapture(filepath)
                    try:
                        frame_idx = 0
                        current_mask = None

                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break

                            # Find nearest detected mask
                            if frame_idx in detected_masks:
                                current_mask = detected_masks[frame_idx]

                            if current_mask and not preview:
                                img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                                mask = create_mask((width, height), current_mask)
                                mask_pil = Image.fromarray(mask)
                                result = lama(img_pil, mask_pil)
                                frame = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
                            elif current_mask and preview:
                                for bbox in current_mask:
                                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

                            out_video.write(frame)

                            progress = 60 + int((frame_idx / total_frames) * 35)
                            if frame_idx % 30 == 0:
                                _update_job(job_id, progress=progress, message=f"Processing... frame {frame_idx}/{total_frames}")

                            frame_idx += 1
                    finally:
                        cap.release()
                        out_video.release()

                    # Merge audio if ffmpeg available
                    _update_job(job_id, progress=95, message="Merging audio...")
                    try:
                        subprocess.run([
                            'ffmpeg', '-y',
                            '-i', temp_video,
                            '-i', filepath,
                            '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
                            '-c:a', 'aac', '-b:a', '192k',
                            '-map', '0:v', '-map', '1:a?',
                            '-shortest',
                            out_path
                        ], check=True, capture_output=True, timeout=600)
                        try:
                            os.unlink(temp_video)
                        except OSError:
                            pass
                    except Exception:
                        # No ffmpeg or no audio, just rename
                        import shutil
                        shutil.move(temp_video, out_path)

                else:
                    # Process image
                    _update_job(job_id, progress=50, message="Processing image...")
                    img_pil = Image.open(filepath).convert("RGB")
                    result, had_watermark = process_image(img_pil)

                    if transparent and not preview:
                        result.save(out_path, "PNG")
                    else:
                        result.save(out_path)

                # Clean up models
                del florence_model, florence_processor, lama
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                _update_job(
                    job_id, status="complete", progress=100,
                    message="Watermark removal complete!" if not preview else "Preview complete!",
                    result={
                        "output_path": out_path,
                        "auto_import": auto_import
                    },
                )

            except Exception as e:
                raise Exception(f"Processing error: {str(e)}")

        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Watermark removal error")
            # Cleanup leaked temp video file
            try:
                if temp_video and os.path.exists(temp_video):
                    os.unlink(temp_video)
            except (OSError, NameError):
                pass

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Scene Detection
# ---------------------------------------------------------------------------
@video_bp.route("/video/scenes", methods=["POST"])
@require_csrf
def video_scenes():
    """Detect scene changes in a video."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400

    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    threshold = safe_float(data.get("threshold", 0.3), 0.3)
    min_scene = safe_float(data.get("min_scene_length", 2.0), 2.0)

    job_id = _new_job("scenes", filepath)

    method = data.get("method", "ffmpeg").strip().lower()

    def _process():
        try:
            from opencut.core.scene_detect import detect_scenes, generate_chapter_markers

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            if method == "ml":
                from opencut.core.scene_detect import detect_scenes_ml
                info = detect_scenes_ml(
                    filepath, threshold=threshold,
                    min_scene_length=min_scene,
                    on_progress=_on_progress,
                )
            else:
                info = detect_scenes(
                    filepath, threshold=threshold,
                    min_scene_length=min_scene,
                    on_progress=_on_progress,
                )

            # Generate YouTube chapters
            chapters_yt = generate_chapter_markers(info, format="youtube")

            _update_job(
                job_id, status="complete", progress=100,
                message=f"Found {info.total_scenes} scenes",
                result={
                    "total_scenes": info.total_scenes,
                    "duration": info.duration,
                    "avg_scene_length": round(info.avg_scene_length, 2),
                    "boundaries": [
                        {
                            "time": round(b.time, 3),
                            "frame": b.frame,
                            "score": round(b.score, 3),
                            "label": b.label,
                        }
                        for b in info.boundaries
                    ],
                    "youtube_chapters": chapters_yt,
                },
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Scene detection error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Export Rendered Video (FFmpeg concat)
# ---------------------------------------------------------------------------
@video_bp.route("/export-video", methods=["POST"])
@require_csrf
def export_video():
    """Render a new video file with silences removed (no visible cuts)."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")
    segments_data = data.get("segments", [])
    if len(segments_data) > 10000:
        return jsonify({"error": "Too many segments (max 10000)"}), 400

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400

    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if not segments_data:
        return jsonify({"error": "No segments provided"}), 400

    # Options — allowlisted codecs to prevent argument injection
    _VALID_VIDEO_CODECS = {"libx264", "libx265", "copy", "vp9", "libvpx-vp9", "libaom-av1", "prores_ks"}
    _VALID_AUDIO_CODECS = {"aac", "libmp3lame", "flac", "pcm_s16le", "copy", "libopus", "libvorbis"}
    video_codec = data.get("video_codec", "libx264")
    if video_codec not in _VALID_VIDEO_CODECS:
        video_codec = "libx264"
    audio_codec = data.get("audio_codec", "aac")
    if audio_codec not in _VALID_AUDIO_CODECS:
        audio_codec = "aac"
    quality = data.get("quality", "medium")  # low, medium, high, lossless
    output_format = data.get("output_format", "mp4")
    audio_only = data.get("audio_only", False)

    # Audio-only: override format and extension
    audio_format_ext = data.get("audio_format", "mp3")  # mp3, wav, flac
    if audio_only:
        output_format = audio_format_ext

    job_id = _new_job("export-video", filepath)

    def _process():

        try:
            _update_job(job_id, progress=5, message="Preparing export...")

            try:
                from opencut.utils.media import probe as _probe_media
            except ImportError:
                _probe_media = None

            if _probe_media:
                _probe_media(filepath)

            effective_dir = _resolve_output_dir(filepath, output_dir)
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            suffix = "_audio" if audio_only else "_opencut"
            output_path = _unique_output_path(
                os.path.join(effective_dir, f"{base_name}{suffix}.{output_format}")
            )

            # Build filter_complex for segment extraction and concatenation
            total_segments = len(segments_data)

            if total_segments == 0:
                _update_job(job_id, status="error", error="No segments to export", message="No segments")
                return

            # Determine CRF / quality
            crf_map = {"low": "28", "medium": "23", "high": "18", "lossless": "0"}
            crf = crf_map.get(quality, "23")

            if audio_only:
                # Audio-only: just use atrim + concat (no video)
                filter_parts = []
                concat_inputs = []

                for i, seg in enumerate(segments_data):
                    start = seg.get("start", 0)
                    end = seg.get("end", 0)
                    if end <= start:
                        continue
                    filter_parts.append(
                        f"[0:a]atrim=start={start:.6f}:end={end:.6f},asetpts=PTS-STARTPTS[a{i}]"
                    )
                    concat_inputs.append(f"[a{i}]")

                if not concat_inputs:
                    _update_job(job_id, status="error", error="No valid segments", message="No valid segments")
                    return

                n = len(concat_inputs)
                concat_str = "".join(concat_inputs) + f"concat=n={n}:v=0:a=1[outa]"
                filter_parts.append(concat_str)
                filter_complex = ";\n".join(filter_parts)

                _update_job(job_id, progress=15, message=f"Rendering {n} segments to {output_format.upper()} (audio only)...")

                cmd = [
                    "ffmpeg", "-hide_banner", "-y",
                    "-i", filepath,
                    "-filter_complex", filter_complex,
                    "-map", "[outa]", "-vn",
                ]

                # Audio codec by format
                if output_format == "wav":
                    cmd.extend(["-acodec", "pcm_s16le"])
                elif output_format == "flac":
                    cmd.extend(["-acodec", "flac"])
                else:
                    cmd.extend(["-acodec", "libmp3lame", "-b:a", "192k"])

                cmd.append(output_path)

            else:
                # Video + audio export
                filter_parts = []
                concat_inputs = []

                for i, seg in enumerate(segments_data):
                    start = seg.get("start", 0)
                    end = seg.get("end", 0)
                    if end <= start:
                        continue
                    filter_parts.append(
                        f"[0:v]trim=start={start:.6f}:end={end:.6f},setpts=PTS-STARTPTS[v{i}]"
                    )
                    filter_parts.append(
                        f"[0:a]atrim=start={start:.6f}:end={end:.6f},asetpts=PTS-STARTPTS[a{i}]"
                    )
                    concat_inputs.append(f"[v{i}][a{i}]")

                    if i % 10 == 0:
                        pct = int(5 + (i / total_segments) * 10)
                        _update_job(job_id, progress=pct, message=f"Building filter graph ({i+1}/{total_segments} segments)...")

                if not concat_inputs:
                    _update_job(job_id, status="error", error="No valid segments after filtering", message="No valid segments")
                    return

                n = len(concat_inputs)
                concat_str = "".join(concat_inputs) + f"concat=n={n}:v=1:a=1[outv][outa]"
                filter_parts.append(concat_str)
                filter_complex = ";\n".join(filter_parts)

                _update_job(job_id, progress=15, message=f"Rendering {n} segments to {output_format.upper()}...")

                cmd = [
                    "ffmpeg", "-hide_banner", "-y",
                    "-i", filepath,
                    "-filter_complex", filter_complex,
                    "-map", "[outv]", "-map", "[outa]",
                ]

                if video_codec == "copy":
                    # Stream copy is incompatible with filter_complex concat;
                    # use low-CRF libx264 for near-lossless re-encode
                    cmd.extend(["-c:v", "libx264", "-crf", "18", "-preset", "fast"])
                elif video_codec == "libx264":
                    cmd.extend(["-c:v", "libx264", "-crf", crf, "-preset", "fast"])
                elif video_codec == "libx265":
                    cmd.extend(["-c:v", "libx265", "-crf", crf, "-preset", "fast"])
                else:
                    cmd.extend(["-c:v", video_codec])

                cmd.extend(["-c:a", audio_codec, "-b:a", "192k"])
                cmd.extend(["-pix_fmt", "yuv420p"])
                cmd.append(output_path)

            _update_job(job_id, progress=20, message="FFmpeg encoding in progress...")

            # Run FFmpeg with progress monitoring
            proc = _sp.Popen(
                cmd, stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True
            )
            _register_job_process(job_id, proc)

            # Calculate expected output duration for progress estimation
            total_kept = sum(
                seg.get("end", 0) - seg.get("start", 0) for seg in segments_data
                if seg.get("end", 0) > seg.get("start", 0)
            )

            import re as _re
            for line in proc.stdout:
                line = line.strip()
                # Parse FFmpeg time= progress
                time_match = _re.search(r"time=(\d+):(\d+):(\d+\.?\d*)", line)
                if time_match and total_kept > 0:
                    h, m, s = float(time_match.group(1)), float(time_match.group(2)), float(time_match.group(3))
                    current_time = h * 3600 + m * 60 + s
                    pct = min(95, int(20 + (current_time / total_kept) * 75))
                    elapsed_str = f"{int(current_time // 60)}:{int(current_time % 60):02d}"
                    _update_job(job_id, progress=pct, message=f"Encoding... {elapsed_str} / {int(total_kept // 60)}:{int(total_kept % 60):02d}")

                if _is_cancelled(job_id):
                    proc.kill()
                    _unregister_job_process(job_id)
                    # Clean up partial file
                    if os.path.exists(output_path):
                        try:
                            os.unlink(output_path)
                        except OSError:
                            pass
                    return

            proc.wait()
            _unregister_job_process(job_id)

            if proc.returncode != 0:
                _update_job(
                    job_id, status="error",
                    error=f"FFmpeg exited with code {proc.returncode}",
                    message="Encoding failed",
                )
                return

            # Get output file size
            output_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
            size_mb = output_size / (1024 * 1024)

            _update_job(
                job_id, status="complete", progress=100,
                message="Export complete!",
                result={
                    "output_path": output_path,
                    "segments": n,
                    "duration": round(total_kept, 2),
                    "file_size_mb": round(size_mb, 1),
                    "codec": video_codec,
                },
            )

        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Export video error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread

    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Video Effects (FFmpeg-based, always available)
# ---------------------------------------------------------------------------
@video_bp.route("/video/fx/list", methods=["GET"])
def video_fx_list():
    """Return available FFmpeg-based video effects."""
    try:
        from opencut.core.video_fx import get_available_video_effects
        return jsonify({"effects": get_available_video_effects()})
    except Exception as e:
        return _safe_error(e)


@video_bp.route("/video/fx/apply", methods=["POST"])
@require_csrf
def video_fx_apply():
    """Apply a video effect."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")
    effect = data.get("effect", "").strip()
    params = data.get("params", {})

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400

    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if not effect:
        return jsonify({"error": "No effect specified"}), 400

    job_id = _new_job("video-fx", filepath)

    def _process():
        try:
            from opencut.core import video_fx

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(filepath, output_dir)

            if effect == "stabilize":
                out = video_fx.stabilize_video(
                    filepath, output_dir=effective_dir,
                    smoothing=params.get("smoothing", 10),
                    crop=params.get("crop", "keep"),
                    zoom=params.get("zoom", 0),
                    on_progress=_on_progress,
                )
            elif effect == "chromakey":
                out = video_fx.chromakey(
                    filepath, output_dir=effective_dir,
                    color=params.get("color", "0x00FF00"),
                    similarity=params.get("similarity", 0.3),
                    blend=params.get("blend", 0.1),
                    background=params.get("background", ""),
                    on_progress=_on_progress,
                )
            elif effect == "lut":
                lut_path = params.get("lut_path", "")
                if not lut_path:
                    raise ValueError("No LUT file path provided")
                lut_path = validate_filepath(lut_path)
                out = video_fx.apply_lut(
                    filepath, lut_path, output_dir=effective_dir,
                    intensity=params.get("intensity", 1.0),
                    on_progress=_on_progress,
                )
            elif effect == "vignette":
                out = video_fx.apply_vignette(
                    filepath, output_dir=effective_dir,
                    intensity=params.get("intensity", 0.5),
                    on_progress=_on_progress,
                )
            elif effect == "film_grain":
                out = video_fx.apply_film_grain(
                    filepath, output_dir=effective_dir,
                    intensity=params.get("intensity", 0.5),
                    on_progress=_on_progress,
                )
            elif effect == "letterbox":
                out = video_fx.apply_letterbox(
                    filepath, output_dir=effective_dir,
                    aspect=params.get("aspect", "2.39:1"),
                    on_progress=_on_progress,
                )
            elif effect == "color_match":
                ref_path = params.get("reference_path", "")
                if not ref_path:
                    raise ValueError("No reference file path provided")
                ref_path = validate_filepath(ref_path)
                out = video_fx.color_match(
                    filepath, output_dir=effective_dir,
                    reference_path=ref_path,
                    on_progress=_on_progress,
                )
            else:
                raise ValueError(f"Unknown video effect: {effect}")

            _update_job(
                job_id, status="complete", progress=100,
                message=f"Effect '{effect}' applied!",
                result={"output_path": out, "effect": effect},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Video FX error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Video AI (upscale, bg removal, interpolation, denoise)
# ---------------------------------------------------------------------------
@video_bp.route("/video/ai/capabilities", methods=["GET"])
def video_ai_capabilities():
    """Return available AI video capabilities."""
    try:
        from opencut.core.video_ai import get_ai_capabilities
        return jsonify(get_ai_capabilities())
    except Exception as e:
        return _safe_error(e)


@video_bp.route("/video/ai/upscale", methods=["POST"])
@require_csrf
def video_ai_upscale():
    """AI upscale video using Real-ESRGAN."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")
    scale = safe_int(data.get("scale", 2), 2, min_val=1, max_val=4)
    model = data.get("model", "realesrgan-x4plus")

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400

    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    job_id = _new_job("ai-upscale", filepath)

    def _process():
        try:
            from opencut.core.video_ai import upscale_video

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(filepath, output_dir)
            out = upscale_video(
                filepath, output_dir=effective_dir,
                scale=scale, model=model,
                on_progress=_on_progress,
            )
            _update_job(
                job_id, status="complete", progress=100,
                message=f"Upscaled {scale}x!",
                result={"output_path": out},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("AI upscale error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@video_bp.route("/video/ai/rembg", methods=["POST"])
@require_csrf
def video_ai_rembg():
    """AI background removal using rembg."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")
    model = data.get("model", "u2net")
    bg_color = data.get("bg_color", "")
    alpha_only = data.get("alpha_only", False)

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400

    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    job_id = _new_job("ai-rembg", filepath)

    def _process():
        try:
            from opencut.core.video_ai import remove_background

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(filepath, output_dir)
            out = remove_background(
                filepath, output_dir=effective_dir,
                model=model, bg_color=bg_color,
                alpha_only=alpha_only,
                on_progress=_on_progress,
            )
            _update_job(
                job_id, status="complete", progress=100,
                message="Background removed!",
                result={"output_path": out},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("AI rembg error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@video_bp.route("/video/ai/interpolate", methods=["POST"])
@require_csrf
def video_ai_interpolate():
    """AI frame interpolation."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")
    multiplier = safe_int(data.get("multiplier", 2), 2, min_val=2, max_val=8)

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400

    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    job_id = _new_job("ai-interpolate", filepath)

    def _process():
        try:
            from opencut.core.video_ai import frame_interpolate

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(filepath, output_dir)
            out = frame_interpolate(
                filepath, output_dir=effective_dir,
                multiplier=multiplier,
                on_progress=_on_progress,
            )
            _update_job(
                job_id, status="complete", progress=100,
                message=f"Interpolated {multiplier}x!",
                result={"output_path": out},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("AI interpolation error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@video_bp.route("/video/ai/denoise", methods=["POST"])
@require_csrf
def video_ai_denoise():
    """AI video noise reduction."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")
    method = data.get("method", "nlmeans")
    strength = safe_float(data.get("strength", 0.5), 0.5, min_val=0.0, max_val=1.0)

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400

    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    job_id = _new_job("video-denoise", filepath)

    def _process():
        try:
            from opencut.core.video_ai import video_denoise

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(filepath, output_dir)
            out = video_denoise(
                filepath, output_dir=effective_dir,
                method=method, strength=strength,
                on_progress=_on_progress,
            )
            _update_job(
                job_id, status="complete", progress=100,
                message="Video denoised!",
                result={"output_path": out},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Video denoise error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@video_bp.route("/video/ai/install", methods=["POST"])
@require_csrf
@require_rate_limit("model_install")
def video_ai_install():
    """Install AI video dependencies."""
    data = request.get_json(force=True)
    component = data.get("component", "upscale")

    packages = {
        "upscale": ["realesrgan", "basicsr", "opencv-python-headless"],
        "rembg": ["rembg[gpu]", "opencv-python-headless"],
        "rembg_cpu": ["rembg", "opencv-python-headless"],
    }

    pkgs = packages.get(component, [])
    if not pkgs:
        return jsonify({"error": f"Unknown component: {component}"}), 400

    job_id = _new_job("install", component)

    def _process():
        try:
            for i, pkg in enumerate(pkgs):
                pct = int((i / len(pkgs)) * 90)
                _update_job(job_id, progress=pct, message=f"Installing {pkg}...")
                safe_pip_install(pkg, timeout=600)
            _update_job(
                job_id, status="complete", progress=100,
                message=f"Installed {component}!",
                result={"component": component},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("AI install error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Face Tools (detection, blur, auto-censor)
# ---------------------------------------------------------------------------
@video_bp.route("/video/face/capabilities", methods=["GET"])
def face_capabilities():
    """Return face tool capabilities."""
    try:
        from opencut.core.face_tools import check_face_tools_available
        return jsonify(check_face_tools_available())
    except Exception as e:
        return _safe_error(e)


@video_bp.route("/video/face/detect", methods=["POST"])
@require_csrf
def face_detect():
    """Detect faces in image or first frame of video."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    detector = data.get("detector", "mediapipe")

    if not filepath:
        return jsonify({"error": "File not found"}), 400

    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        from opencut.core.face_tools import detect_faces_in_frame
        result = detect_faces_in_frame(filepath, detector=detector)
        return jsonify(result)
    except Exception as e:
        return _safe_error(e)


@video_bp.route("/video/face/blur", methods=["POST"])
@require_csrf
def face_blur():
    """Auto-detect and blur faces in video."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")
    method = data.get("method", "gaussian")
    strength = safe_int(data.get("strength", 51), 51, min_val=1, max_val=99)
    detector = data.get("detector", "mediapipe")

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400

    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    job_id = _new_job("face-blur", filepath)

    def _process():
        try:
            from opencut.core.face_tools import blur_faces

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(filepath, output_dir)
            out = blur_faces(
                filepath, output_dir=effective_dir,
                method=method, strength=strength,
                detector=detector, on_progress=_on_progress,
            )
            _update_job(
                job_id, status="complete", progress=100,
                message="Faces blurred!",
                result={"output_path": out},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Face blur error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@video_bp.route("/video/face/install", methods=["POST"])
@require_csrf
@require_rate_limit("model_install")
def face_install():
    """Install MediaPipe for face detection."""
    job_id = _new_job("install", "mediapipe")

    def _process():
        try:
            _update_job(job_id, progress=20, message="Installing MediaPipe...")
            safe_pip_install("mediapipe", timeout=600)
            safe_pip_install("opencv-python-headless", timeout=600)
            _update_job(
                job_id, status="complete", progress=100,
                message="MediaPipe installed!",
                result={"component": "mediapipe"},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Style Transfer
# ---------------------------------------------------------------------------
@video_bp.route("/video/style/list", methods=["GET"])
def style_list():
    """Return available style transfer models."""
    try:
        from opencut.core.style_transfer import get_available_styles
        return jsonify({"styles": get_available_styles()})
    except Exception as e:
        return _safe_error(e)


@video_bp.route("/video/style/apply", methods=["POST"])
@require_csrf
def style_apply():
    """Apply neural style transfer to video."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")
    style_name = data.get("style", "candy")
    intensity = safe_float(data.get("intensity", 1.0), 1.0, min_val=0.0, max_val=2.0)

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400

    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    job_id = _new_job("style-transfer", filepath)

    def _process():
        try:
            from opencut.core.style_transfer import style_transfer_video

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(filepath, output_dir)
            out = style_transfer_video(
                filepath, style_name=style_name,
                output_dir=effective_dir,
                intensity=intensity,
                on_progress=_on_progress,
            )
            _update_job(
                job_id, status="complete", progress=100,
                message=f"Style '{style_name}' applied!",
                result={"output_path": out, "style": style_name},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Style transfer error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Export Presets
# ---------------------------------------------------------------------------
@video_bp.route("/export/presets", methods=["GET"])
def export_presets_list():
    """Return available export presets."""
    try:
        from opencut.core.export_presets import get_export_presets, get_preset_categories
        return jsonify({
            "presets": get_export_presets(),
            "categories": get_preset_categories(),
        })
    except Exception as e:
        return _safe_error(e)


@video_bp.route("/export/preset", methods=["POST"])
@require_csrf
def export_with_preset_route():
    """Export video using a preset profile."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")
    preset_name = data.get("preset", "youtube_1080p")

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400

    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    job_id = _new_job("export-preset", filepath)

    def _process():
        try:
            from opencut.core.export_presets import export_with_preset

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(filepath, output_dir)
            out = export_with_preset(
                filepath, preset_name,
                output_dir=effective_dir,
                on_progress=_on_progress,
            )
            size_mb = os.path.getsize(out) / (1024 * 1024)
            _update_job(
                job_id, status="complete", progress=100,
                message=f"Exported ({preset_name}, {size_mb:.1f}MB)",
                result={"output_path": out, "preset": preset_name, "size_mb": round(size_mb, 1)},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Export preset error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Auto-Thumbnail Generation
# ---------------------------------------------------------------------------
@video_bp.route("/export/thumbnails", methods=["POST"])
@require_csrf
def generate_thumbnails_route():
    """Generate thumbnail candidates from video."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")
    count = safe_int(data.get("count", 5), 5, min_val=1, max_val=20)
    width = safe_int(data.get("width", 1920), 1920, min_val=160, max_val=3840)
    use_faces = data.get("use_faces", True)

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400

    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    job_id = _new_job("thumbnails", filepath)

    def _process():
        try:
            from opencut.core.thumbnail import generate_thumbnails

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(filepath, output_dir)
            results = generate_thumbnails(
                filepath, output_dir=effective_dir,
                count=count, width=width,
                use_faces=use_faces,
                on_progress=_on_progress,
            )
            _update_job(
                job_id, status="complete", progress=100,
                message=f"Generated {len(results)} thumbnails",
                result={"thumbnails": results},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Thumbnail generation error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Batch Processing
# ---------------------------------------------------------------------------
@video_bp.route("/batch/create", methods=["POST"])
@require_csrf
def batch_create():
    """Create a batch processing job."""
    data = request.get_json(force=True)
    operation = data.get("operation", "")
    filepaths = data.get("filepaths", [])
    params = data.get("params", {})

    if not operation:
        return jsonify({"error": "No operation specified"}), 400
    if not filepaths:
        return jsonify({"error": "No files specified"}), 400
    if len(filepaths) > MAX_BATCH_FILES:
        return jsonify({"error": f"Too many files. Maximum is {MAX_BATCH_FILES}."}), 400

    # Validate all file paths
    validated_paths = []
    for fp in filepaths:
        try:
            validated_paths.append(validate_filepath(fp))
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
    filepaths = validated_paths

    try:
        from opencut.core.batch_process import create_batch, finalize_batch, update_batch_item

        batch_id = str(uuid.uuid4())[:8]
        batch = create_batch(batch_id, operation, filepaths, params)

        # Process items sequentially in a background thread
        def _process_batch():
            for idx, item in enumerate(batch.items):
                if batch.status == "cancelled":
                    break
                if item.status == "skipped":
                    continue

                update_batch_item(batch_id, idx,
                                  status="running", started_at=time.time(),
                                  message="Processing...")

                try:
                    # Route the operation to the appropriate function
                    result = _execute_batch_item(
                        operation, item.filepath, params,
                        lambda pct, msg: update_batch_item(
                            batch_id, idx, progress=pct, message=msg
                        ),
                    )
                    update_batch_item(batch_id, idx,
                                      status="complete", progress=100,
                                      output_path=result,
                                      message="Done",
                                      finished_at=time.time())
                except Exception as e:
                    update_batch_item(batch_id, idx,
                                      status="error", error=str(e),
                                      message=f"Error: {e}",
                                      finished_at=time.time())

            finalize_batch(batch_id)

        thread = threading.Thread(target=_process_batch, daemon=True)
        thread.start()

        return jsonify({"batch_id": batch_id, "status": "running", "total": batch.total})
    except Exception as e:
        return _safe_error(e)


@video_bp.route("/batch/<batch_id>", methods=["GET"])
def batch_status(batch_id):
    """Get batch job status."""
    try:
        from opencut.core.batch_process import get_batch
        batch = get_batch(batch_id)
        if not batch:
            return jsonify({"error": "Batch not found"}), 404
        return jsonify(batch.summary)
    except Exception as e:
        return _safe_error(e)


@video_bp.route("/batch/<batch_id>/cancel", methods=["POST"])
@require_csrf
def batch_cancel(batch_id):
    """Cancel a running batch."""
    try:
        from opencut.core.batch_process import cancel_batch
        if cancel_batch(batch_id):
            return jsonify({"status": "cancelled"})
        return jsonify({"error": "Batch not found"}), 404
    except Exception as e:
        return _safe_error(e)


@video_bp.route("/batch/list", methods=["GET"])
def batch_list():
    """List recent batch jobs."""
    try:
        from opencut.core.batch_process import list_batches
        return jsonify({"batches": list_batches()})
    except Exception as e:
        return _safe_error(e)


def _execute_batch_item(operation, filepath, params, on_progress):
    """Execute a single batch item based on operation type."""
    output_dir = params.get("output_dir", os.path.dirname(filepath))

    if operation == "denoise":
        from opencut.core.audio_suite import denoise_audio
        return denoise_audio(
            filepath, output_dir=output_dir,
            strength=safe_float(params.get("strength", 0.5), 0.5, min_val=0.0, max_val=1.0),
            on_progress=on_progress,
        )
    elif operation == "normalize":
        from opencut.core.audio_suite import normalize_audio
        return normalize_audio(
            filepath, output_dir=output_dir,
            target_lufs=safe_float(params.get("target_lufs", -16), -16.0, min_val=-70.0, max_val=0.0),
            on_progress=on_progress,
        )
    elif operation == "stabilize":
        from opencut.core.video_fx import stabilize_video
        return stabilize_video(
            filepath, output_dir=output_dir,
            smoothing=safe_int(params.get("smoothing", 10), 10, min_val=1, max_val=100),
            on_progress=on_progress,
        )
    elif operation == "face_blur":
        from opencut.core.face_tools import blur_faces
        _valid_blur = {"gaussian", "pixelate", "fill"}
        _blur_method = params.get("method", "gaussian")
        if _blur_method not in _valid_blur:
            _blur_method = "gaussian"
        return blur_faces(
            filepath, output_dir=output_dir,
            method=_blur_method,
            strength=safe_int(params.get("strength", 51), 51, min_val=1, max_val=99),
            on_progress=on_progress,
        )
    elif operation == "export_preset":
        from opencut.core.export_presets import export_with_preset
        return export_with_preset(
            filepath, params.get("preset", "youtube_1080p"),
            output_dir=output_dir,
            on_progress=on_progress,
        )
    elif operation == "vignette":
        from opencut.core.video_fx import apply_vignette
        return apply_vignette(
            filepath, output_dir=output_dir,
            intensity=safe_float(params.get("intensity", 0.5), 0.5, min_val=0.0, max_val=2.0),
            on_progress=on_progress,
        )
    elif operation == "film_grain":
        from opencut.core.video_fx import apply_film_grain
        return apply_film_grain(
            filepath, output_dir=output_dir,
            intensity=safe_float(params.get("intensity", 0.5), 0.5, min_val=0.0, max_val=2.0),
            on_progress=on_progress,
        )
    elif operation == "letterbox":
        from opencut.core.video_fx import apply_letterbox
        _valid_aspects = {"2.39:1", "2.35:1", "1.85:1", "16:9", "4:3", "1:1", "21:9"}
        _aspect = params.get("aspect", "2.39:1")
        if _aspect not in _valid_aspects:
            _aspect = "2.39:1"
        return apply_letterbox(
            filepath, output_dir=output_dir,
            aspect=_aspect,
            on_progress=on_progress,
        )
    elif operation == "style_transfer":
        from opencut.core.style_transfer import style_transfer_video
        _valid_styles = {"candy", "mosaic", "rain_princess", "udnie", "starry_night", "la_muse", "the_scream", "feathers", "composition_vii"}
        _style = params.get("style", "candy")
        if _style not in _valid_styles:
            _style = "candy"
        return style_transfer_video(
            filepath, style_name=_style,
            output_dir=output_dir,
            intensity=safe_float(params.get("intensity", 1.0), 1.0, min_val=0.0, max_val=2.0),
            on_progress=on_progress,
        )
    else:
        raise ValueError(f"Unsupported batch operation: {operation}")


# ---------------------------------------------------------------------------
# Speed Ramp
# ---------------------------------------------------------------------------
@video_bp.route("/video/speed/presets", methods=["GET"])
def speed_ramp_presets():
    """Return available speed ramp presets."""
    try:
        from opencut.core.speed_ramp import EASING_FUNCTIONS, get_speed_ramp_presets
        return jsonify({
            "presets": get_speed_ramp_presets(),
            "easings": list(EASING_FUNCTIONS.keys()),
        })
    except Exception as e:
        return _safe_error(e)


@video_bp.route("/video/speed/change", methods=["POST"])
@require_csrf
def speed_change_route():
    """Apply constant speed change."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    speed = safe_float(data.get("speed", 2.0), 2.0, min_val=0.1, max_val=100.0)
    output_dir = data.get("output_dir", "")

    if not filepath:
        return jsonify({"error": "File not found"}), 400

    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    job_id = _new_job("speed", filepath)

    def _process():
        try:
            from opencut.core.speed_ramp import change_speed

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(filepath, output_dir)
            out = change_speed(
                filepath, speed=speed, output_dir=effective_dir,
                maintain_pitch=data.get("maintain_pitch", False),
                on_progress=_on_progress,
            )
            _update_job(
                job_id, status="complete", progress=100,
                message=f"Speed changed to {speed}x",
                result={"output_path": out, "speed": speed},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Speed change error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@video_bp.route("/video/speed/reverse", methods=["POST"])
@require_csrf
def speed_reverse_route():
    """Reverse video playback."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")

    if not filepath:
        return jsonify({"error": "File not found"}), 400

    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    job_id = _new_job("reverse", filepath)

    def _process():
        try:
            from opencut.core.speed_ramp import reverse_video

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(filepath, output_dir)
            out = reverse_video(
                filepath, output_dir=effective_dir,
                reverse_audio=data.get("reverse_audio", True),
                on_progress=_on_progress,
            )
            _update_job(
                job_id, status="complete", progress=100,
                message="Video reversed!",
                result={"output_path": out},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@video_bp.route("/video/speed/ramp", methods=["POST"])
@require_csrf
def speed_ramp_route():
    """Apply keyframe-based speed ramp or preset."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    preset = data.get("preset", "")
    keyframes = data.get("keyframes", [])
    output_dir = data.get("output_dir", "")

    if not filepath:
        return jsonify({"error": "File not found"}), 400

    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    job_id = _new_job("speed-ramp", filepath)

    def _process():
        try:
            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(filepath, output_dir)

            if preset:
                from opencut.core.speed_ramp import apply_speed_ramp_preset
                out = apply_speed_ramp_preset(
                    filepath, preset, output_dir=effective_dir,
                    on_progress=_on_progress,
                )
            else:
                from opencut.core.speed_ramp import speed_ramp
                _VALID_EASING = {"linear", "ease_in", "ease_out", "ease_in_out"}
                easing_val = data.get("easing", "ease_in_out")
                if easing_val not in _VALID_EASING:
                    easing_val = "ease_in_out"
                out = speed_ramp(
                    filepath, keyframes, output_dir=effective_dir,
                    easing=easing_val,
                    on_progress=_on_progress,
                )

            _update_job(
                job_id, status="complete", progress=100,
                message="Speed ramp applied!",
                result={"output_path": out},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Speed ramp error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# LUT Library
# ---------------------------------------------------------------------------
@video_bp.route("/video/lut/list", methods=["GET"])
def lut_list():
    """Return available LUTs."""
    try:
        from opencut.core.lut_library import get_lut_list
        return jsonify({"luts": get_lut_list()})
    except Exception as e:
        return _safe_error(e)


@video_bp.route("/video/lut/apply", methods=["POST"])
@require_csrf
def lut_apply():
    """Apply a color LUT to video."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    lut_name = data.get("lut", "teal_orange")
    intensity = safe_float(data.get("intensity", 1.0), 1.0, min_val=0.0, max_val=2.0)
    output_dir = data.get("output_dir", "")

    if not filepath:
        return jsonify({"error": "File not found"}), 400

    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    job_id = _new_job("lut", filepath)

    def _process():
        try:
            from opencut.core.lut_library import apply_lut

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(filepath, output_dir)
            out = apply_lut(
                filepath, lut_name, intensity=intensity,
                output_dir=effective_dir,
                on_progress=_on_progress,
            )
            _update_job(
                job_id, status="complete", progress=100,
                message=f"LUT applied: {lut_name}",
                result={"output_path": out, "lut": lut_name},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("LUT apply error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@video_bp.route("/video/lut/generate-all", methods=["POST"])
@require_csrf
def lut_generate_all():
    """Pre-generate all built-in LUT .cube files."""
    job_id = _new_job("lut-gen", "all")

    def _process():
        try:
            from opencut.core.lut_library import generate_all_luts

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            count = generate_all_luts(on_progress=_on_progress)
            _update_job(
                job_id, status="complete", progress=100,
                message=f"Generated {count} LUT files",
                result={"count": count},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Chromakey & Compositing
# ---------------------------------------------------------------------------
@video_bp.route("/video/chromakey", methods=["POST"])
@require_csrf
def chromakey_route():
    """Chromakey (green/blue screen) removal + compositing."""
    data = request.get_json(force=True)
    fg = data.get("filepath", "").strip()
    bg = data.get("background", "").strip()
    output_dir = data.get("output_dir", "")

    if not fg:
        return jsonify({"error": "Foreground file not found"}), 400
    if not bg:
        return jsonify({"error": "Background file not found"}), 400

    try:
        fg = validate_filepath(fg)
    except ValueError as e:
        return jsonify({"error": f"Foreground: {e}"}), 400

    try:
        bg = validate_filepath(bg)
    except ValueError as e:
        return jsonify({"error": f"Background: {e}"}), 400

    job_id = _new_job("chromakey", fg)

    def _process():
        try:
            from opencut.core.chromakey import chromakey_video

            def _p(pct, msg):
                _update_job(job_id, progress=pct, message=msg)
            d = _resolve_output_dir(fg, output_dir)
            _valid_chroma = {"green", "blue", "red", "custom"}
            _chroma_color = data.get("color", "green")
            if _chroma_color not in _valid_chroma:
                _chroma_color = "green"
            out = chromakey_video(fg, bg, output_dir=d, color=_chroma_color,
                                  tolerance=safe_float(data.get("tolerance", 0.5), 0.5, min_val=0.0, max_val=1.0),
                                  spill_suppress=safe_float(data.get("spill_suppress", 0.5), 0.5, min_val=0.0, max_val=1.0),
                                  edge_blur=safe_int(data.get("edge_blur", 3), 3, min_val=0, max_val=20), on_progress=_p)
            _update_job(job_id, status="complete", progress=100, message="Chromakey complete!",
                        result={"output_path": out})
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@video_bp.route("/video/pip", methods=["POST"])
@require_csrf
def pip_route():
    """Picture-in-picture overlay."""
    data = request.get_json(force=True)
    main = data.get("filepath", "").strip()
    pip = data.get("pip_path", "").strip()

    if not main:
        return jsonify({"error": "Main file not found"}), 400
    if not pip:
        return jsonify({"error": "PiP file not found"}), 400

    try:
        main = validate_filepath(main)
    except ValueError as e:
        return jsonify({"error": f"Main file: {e}"}), 400

    try:
        pip = validate_filepath(pip)
    except ValueError as e:
        return jsonify({"error": f"PiP file: {e}"}), 400

    job_id = _new_job("pip", main)

    def _process():
        try:
            from opencut.core.chromakey import picture_in_picture

            def _p(pct, msg):
                _update_job(job_id, progress=pct, message=msg)
            d = _resolve_output_dir(main, data.get("output_dir", ""))
            out = picture_in_picture(main, pip, output_dir=d,
                                      position=data.get("position", "bottom_right"),
                                      scale=safe_float(data.get("scale", 0.25), 0.25, min_val=0.05, max_val=1.0), on_progress=_p)
            _update_job(job_id, status="complete", progress=100, result={"output_path": out})
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@video_bp.route("/video/blend", methods=["POST"])
@require_csrf
def blend_route():
    """Blend two videos with a blend mode."""
    data = request.get_json(force=True)
    base = data.get("filepath", "").strip()
    overlay = data.get("overlay_path", "").strip()

    if not base:
        return jsonify({"error": "Base file not found"}), 400
    if not overlay:
        return jsonify({"error": "Overlay not found"}), 400

    try:
        base = validate_filepath(base)
    except ValueError as e:
        return jsonify({"error": f"Base file: {e}"}), 400

    try:
        overlay = validate_filepath(overlay)
    except ValueError as e:
        return jsonify({"error": f"Overlay: {e}"}), 400

    job_id = _new_job("blend", base)

    def _process():
        try:
            from opencut.core.chromakey import blend_videos

            def _p(pct, msg):
                _update_job(job_id, progress=pct, message=msg)
            d = _resolve_output_dir(base, data.get("output_dir", ""))
            out = blend_videos(base, overlay, output_dir=d,
                                mode=data.get("mode", "overlay"),
                                opacity=safe_float(data.get("opacity", 0.5), 0.5, min_val=0.0, max_val=1.0), on_progress=_p)
            _update_job(job_id, status="complete", progress=100, result={"output_path": out})
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Object Removal
# ---------------------------------------------------------------------------
@video_bp.route("/video/remove/capabilities", methods=["GET"])
def removal_capabilities():
    try:
        from opencut.core.object_removal import get_removal_capabilities
        return jsonify(get_removal_capabilities())
    except Exception as e:
        return _safe_error(e)


@video_bp.route("/video/remove/watermark", methods=["POST"])
@require_csrf
def remove_watermark_route():
    """Remove watermark using delogo or LaMA."""
    data = request.get_json(force=True)
    fp = data.get("filepath", "").strip()
    region = data.get("region", {})
    method = data.get("method", "delogo")

    if not fp:
        return jsonify({"error": "File not found"}), 400

    try:
        fp = validate_filepath(fp)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if not region:
        return jsonify({"error": "No region specified"}), 400
    job_id = _new_job("watermark", fp)

    def _process():
        try:
            def _p(pct, msg):
                _update_job(job_id, progress=pct, message=msg)
            d = _resolve_output_dir(fp, data.get("output_dir", ""))
            if method == "lama":
                from opencut.core.object_removal import remove_watermark_lama
                out = remove_watermark_lama(fp, region, output_dir=d, on_progress=_p)
            else:
                from opencut.core.object_removal import remove_watermark_delogo
                out = remove_watermark_delogo(fp, region, output_dir=d, on_progress=_p)
            _update_job(job_id, status="complete", progress=100, result={"output_path": out})
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Motion Graphics / Titles
# ---------------------------------------------------------------------------
@video_bp.route("/video/title/presets", methods=["GET"])
def title_presets():
    try:
        from opencut.core.motion_graphics import get_title_presets
        return jsonify({"presets": get_title_presets()})
    except Exception as e:
        return _safe_error(e)


@video_bp.route("/video/title/render", methods=["POST"])
@require_csrf
def title_render():
    """Render a standalone title card video."""
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text"}), 400
    if len(text) > 500:
        return jsonify({"error": "Title text too long (max 500 chars)"}), 400
    subtitle = data.get("subtitle", "")
    if len(subtitle) > 500:
        return jsonify({"error": "Subtitle too long (max 500 chars)"}), 400
    job_id = _new_job("title", text[:40])

    def _process():
        try:
            from opencut.core.motion_graphics import render_title_card

            def _p(pct, msg):
                _update_job(job_id, progress=pct, message=msg)
            d = data.get("output_dir", "")
            if d:
                valid, msg = validate_path(d)
                if not valid:
                    _update_job(job_id, status="error", message=msg)
                    return
            else:
                d = tempfile.gettempdir()
            out = render_title_card(text, output_dir=d,
                                     preset=data.get("preset", "fade_center"),
                                     duration=safe_float(data.get("duration", 5.0), 5.0, min_val=0.5, max_val=60.0),
                                     font_size=safe_int(data.get("font_size", 72), 72, min_val=8, max_val=500),
                                     subtitle=subtitle,
                                     on_progress=_p)
            _update_job(job_id, status="complete", progress=100, result={"output_path": out})
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@video_bp.route("/video/title/overlay", methods=["POST"])
@require_csrf
def title_overlay():
    """Overlay animated title onto existing video."""
    data = request.get_json(force=True)
    fp = data.get("filepath", "").strip()
    text = data.get("text", "").strip()

    if not fp:
        return jsonify({"error": "File not found"}), 400

    try:
        fp = validate_filepath(fp)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if not text:
        return jsonify({"error": "No text"}), 400
    job_id = _new_job("title-overlay", fp)

    def _process():
        try:
            from opencut.core.motion_graphics import overlay_title

            def _p(pct, msg):
                _update_job(job_id, progress=pct, message=msg)
            d = _resolve_output_dir(fp, data.get("output_dir", ""))
            out = overlay_title(fp, text, output_dir=d,
                                 preset=data.get("preset", "fade_center"),
                                 font_size=safe_int(data.get("font_size", 72), 72, min_val=8, max_val=500),
                                 start_time=safe_float(data.get("start_time", 0), 0.0, min_val=0.0, max_val=86400.0),
                                 duration=safe_float(data.get("duration", 5.0), 5.0, min_val=0.5, max_val=60.0),
                                 subtitle=data.get("subtitle", ""),
                                 on_progress=_p)
            _update_job(job_id, status="complete", progress=100, result={"output_path": out})
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------
@video_bp.route("/video/transitions/list", methods=["GET"])
def transitions_list():
    try:
        from opencut.core.transitions_3d import get_transition_list
        return jsonify({"transitions": get_transition_list()})
    except Exception as e:
        return _safe_error(e)


@video_bp.route("/video/transitions/apply", methods=["POST"])
@require_csrf
def transitions_apply():
    """Apply transition between two clips."""
    data = request.get_json(force=True)
    a = data.get("clip_a", "").strip()
    b = data.get("clip_b", "").strip()

    if not a:
        return jsonify({"error": "Clip A not found"}), 400
    if not b:
        return jsonify({"error": "Clip B not found"}), 400

    try:
        a = validate_filepath(a)
    except ValueError as e:
        return jsonify({"error": f"Clip A: {e}"}), 400

    try:
        b = validate_filepath(b)
    except ValueError as e:
        return jsonify({"error": f"Clip B: {e}"}), 400

    job_id = _new_job("transition", a)

    def _process():
        try:
            from opencut.core.transitions_3d import apply_transition

            def _p(pct, msg):
                _update_job(job_id, progress=pct, message=msg)
            d = _resolve_output_dir(a, data.get("output_dir", ""))
            out = apply_transition(a, b, output_dir=d,
                                    transition=data.get("transition", "fade"),
                                    duration=safe_float(data.get("duration", 1.0), 1.0, min_val=0.1, max_val=10.0),
                                    on_progress=_p)
            _update_job(job_id, status="complete", progress=100, result={"output_path": out})
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@video_bp.route("/video/transitions/join", methods=["POST"])
@require_csrf
def transitions_join():
    """Join multiple clips with transitions."""
    data = request.get_json(force=True)
    clips = data.get("clips", [])
    if len(clips) < 2:
        return jsonify({"error": "Need 2+ clips"}), 400

    # Validate all clip paths
    validated_clips = []
    for c in clips:
        try:
            validated_clips.append(validate_filepath(c))
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
    clips = validated_clips

    job_id = _new_job("join", f"{len(clips)} clips")

    def _process():
        try:
            from opencut.core.transitions_3d import join_with_transitions

            def _p(pct, msg):
                _update_job(job_id, progress=pct, message=msg)
            d = _resolve_output_dir(clips[0], data.get("output_dir", ""))
            out = join_with_transitions(clips, output_dir=d,
                                         transition=data.get("transition", "fade"),
                                         duration=safe_float(data.get("duration", 1.0), 1.0, min_val=0.1, max_val=10.0),
                                         on_progress=_p)
            _update_job(job_id, status="complete", progress=100, result={"output_path": out})
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Particles
# ---------------------------------------------------------------------------
@video_bp.route("/video/particles/presets", methods=["GET"])
def particle_presets():
    try:
        from opencut.core.particles import get_particle_presets
        return jsonify({"presets": get_particle_presets()})
    except Exception as e:
        return _safe_error(e)


@video_bp.route("/video/particles/apply", methods=["POST"])
@require_csrf
def particle_apply():
    """Overlay particle effects on video."""
    data = request.get_json(force=True)
    fp = data.get("filepath", "").strip()

    if not fp:
        return jsonify({"error": "File not found"}), 400

    try:
        fp = validate_filepath(fp)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    job_id = _new_job("particles", fp)

    def _process():
        try:
            from opencut.core.particles import overlay_particles

            def _p(pct, msg):
                _update_job(job_id, progress=pct, message=msg)
            d = _resolve_output_dir(fp, data.get("output_dir", ""))
            out = overlay_particles(fp, output_dir=d,
                                     preset=data.get("preset", "confetti"),
                                     density=safe_float(data.get("density", 1.0), 1.0, min_val=0.1, max_val=5.0),
                                     on_progress=_p)
            _update_job(job_id, status="complete", progress=100, result={"output_path": out})
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Face Swap & Enhancement
# ---------------------------------------------------------------------------
@video_bp.route("/video/face/swap/capabilities", methods=["GET"])
def face_swap_capabilities():
    try:
        from opencut.core.face_swap import get_face_capabilities
        return jsonify(get_face_capabilities())
    except Exception as e:
        return _safe_error(e)


@video_bp.route("/video/face/enhance", methods=["POST"])
@require_csrf
def face_enhance_route():
    """Enhance/restore faces in video using GFPGAN."""
    data = request.get_json(force=True)
    fp = data.get("filepath", "").strip()

    if not fp:
        return jsonify({"error": "File not found"}), 400

    try:
        fp = validate_filepath(fp)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    job_id = _new_job("face-enhance", fp)

    def _process():
        try:
            from opencut.core.face_swap import enhance_faces

            def _p(pct, msg):
                _update_job(job_id, progress=pct, message=msg)
            d = _resolve_output_dir(fp, data.get("output_dir", ""))
            out = enhance_faces(fp, output_dir=d, upscale=safe_int(data.get("upscale", 2), 2, min_val=1, max_val=4), on_progress=_p)
            _update_job(job_id, status="complete", progress=100, result={"output_path": out})
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@video_bp.route("/video/face/swap", methods=["POST"])
@require_csrf
def face_swap_route():
    """Swap faces in video with a reference image."""
    data = request.get_json(force=True)
    fp = data.get("filepath", "").strip()
    ref = data.get("reference_face", "").strip()

    if not fp:
        return jsonify({"error": "Video not found"}), 400
    if not ref:
        return jsonify({"error": "Reference face not found"}), 400

    try:
        fp = validate_filepath(fp)
    except ValueError as e:
        return jsonify({"error": f"Video: {e}"}), 400

    try:
        ref = validate_filepath(ref)
    except ValueError as e:
        return jsonify({"error": f"Reference face: {e}"}), 400

    job_id = _new_job("face-swap", fp)

    def _process():
        try:
            from opencut.core.face_swap import swap_face

            def _p(pct, msg):
                _update_job(job_id, progress=pct, message=msg)
            d = _resolve_output_dir(fp, data.get("output_dir", ""))
            out = swap_face(fp, ref, output_dir=d, on_progress=_p)
            _update_job(job_id, status="complete", progress=100, result={"output_path": out})
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Pro Upscaling
# ---------------------------------------------------------------------------
@video_bp.route("/video/upscale/capabilities", methods=["GET"])
def upscale_capabilities():
    try:
        from opencut.core.upscale_pro import get_upscale_capabilities
        return jsonify(get_upscale_capabilities())
    except Exception as e:
        return _safe_error(e)


@video_bp.route("/video/upscale/run", methods=["POST"])
@require_csrf
def upscale_run():
    """Upscale video with quality preset."""
    data = request.get_json(force=True)
    fp = data.get("filepath", "").strip()

    if not fp:
        return jsonify({"error": "File not found"}), 400

    try:
        fp = validate_filepath(fp)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    job_id = _new_job("upscale", fp)

    def _process():
        try:
            from opencut.core.upscale_pro import upscale_with_preset

            def _p(pct, msg):
                _update_job(job_id, progress=pct, message=msg)
            d = _resolve_output_dir(fp, data.get("output_dir", ""))
            out = upscale_with_preset(fp, preset=data.get("preset", "fast"),
                                       scale=safe_int(data.get("scale", 2), 2, min_val=1, max_val=4),
                                       output_dir=d, on_progress=_p)
            _update_job(job_id, status="complete", progress=100, result={"output_path": out})
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Color Management
# ---------------------------------------------------------------------------
@video_bp.route("/video/color/capabilities", methods=["GET"])
def color_capabilities():
    try:
        from opencut.core.color_management import get_color_capabilities
        return jsonify(get_color_capabilities())
    except Exception as e:
        return _safe_error(e)


@video_bp.route("/video/color/correct", methods=["POST"])
@require_csrf
def color_correct_route():
    """Apply color correction (exposure, contrast, saturation, temperature, etc.)."""
    data = request.get_json(force=True)
    fp = data.get("filepath", "").strip()

    if not fp:
        return jsonify({"error": "File not found"}), 400

    try:
        fp = validate_filepath(fp)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    job_id = _new_job("color", fp)

    def _process():
        try:
            from opencut.core.color_management import color_correct

            def _p(pct, msg):
                _update_job(job_id, progress=pct, message=msg)
            d = _resolve_output_dir(fp, data.get("output_dir", ""))
            out = color_correct(fp, output_dir=d,
                                 exposure=safe_float(data.get("exposure", 0), 0.0, min_val=-5.0, max_val=5.0),
                                 contrast=safe_float(data.get("contrast", 1.0), 1.0, min_val=0.0, max_val=3.0),
                                 saturation=safe_float(data.get("saturation", 1.0), 1.0, min_val=0.0, max_val=3.0),
                                 temperature=safe_float(data.get("temperature", 0), 0.0, min_val=-100.0, max_val=100.0),
                                 tint=safe_float(data.get("tint", 0), 0.0, min_val=-100.0, max_val=100.0),
                                 shadows=safe_float(data.get("shadows", 0), 0.0, min_val=-100.0, max_val=100.0),
                                 midtones=safe_float(data.get("midtones", 0), 0.0, min_val=-100.0, max_val=100.0),
                                 highlights=safe_float(data.get("highlights", 0), 0.0, min_val=-100.0, max_val=100.0),
                                 on_progress=_p)
            _update_job(job_id, status="complete", progress=100, result={"output_path": out})
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@video_bp.route("/video/color/convert", methods=["POST"])
@require_csrf
def color_convert_route():
    """Convert video color space."""
    data = request.get_json(force=True)
    fp = data.get("filepath", "").strip()

    if not fp:
        return jsonify({"error": "File not found"}), 400

    try:
        fp = validate_filepath(fp)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    job_id = _new_job("colorspace", fp)

    def _process():
        try:
            from opencut.core.color_management import convert_colorspace

            def _p(pct, msg):
                _update_job(job_id, progress=pct, message=msg)
            d = _resolve_output_dir(fp, data.get("output_dir", ""))
            _VALID_COLORSPACES = {"rec709", "rec2020", "srgb", "dci_p3", "aces", "bt709", "bt2020"}
            target_cs = data.get("target", "rec709")
            if target_cs not in _VALID_COLORSPACES:
                target_cs = "rec709"
            out = convert_colorspace(fp, target=target_cs,
                                      output_dir=d, on_progress=_p)
            _update_job(job_id, status="complete", progress=100, result={"output_path": out})
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@video_bp.route("/video/color/external-lut", methods=["POST"])
@require_csrf
def color_external_lut_route():
    """Apply external .cube/.3dl LUT file."""
    data = request.get_json(force=True)
    fp = data.get("filepath", "").strip()
    lut = data.get("lut_path", "").strip()

    if not fp:
        return jsonify({"error": "Video not found"}), 400
    if not lut:
        return jsonify({"error": "LUT file not found"}), 400

    try:
        fp = validate_filepath(fp)
    except ValueError as e:
        return jsonify({"error": f"Video: {e}"}), 400

    try:
        lut = validate_filepath(lut)
    except ValueError as e:
        return jsonify({"error": f"LUT file: {e}"}), 400

    job_id = _new_job("ext-lut", fp)

    def _process():
        try:
            from opencut.core.color_management import apply_external_lut

            def _p(pct, msg):
                _update_job(job_id, progress=pct, message=msg)
            d = _resolve_output_dir(fp, data.get("output_dir", ""))
            out = apply_external_lut(fp, lut, intensity=safe_float(data.get("intensity", 1.0), 1.0, min_val=0.0, max_val=2.0),
                                      output_dir=d, on_progress=_p)
            _update_job(job_id, status="complete", progress=100, result={"output_path": out})
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Video Reframe (resize/crop for phone / social media aspect ratios)
# ---------------------------------------------------------------------------
@video_bp.route("/video/reframe/presets", methods=["GET"])
def reframe_presets():
    """Return available reframe presets with target dimensions."""
    return jsonify([
        {"id": "tiktok",          "name": "TikTok / Shorts",        "width": 1080, "height": 1920, "aspect": "9:16"},
        {"id": "instagram_reel",  "name": "Instagram Reel / Story",  "width": 1080, "height": 1920, "aspect": "9:16"},
        {"id": "instagram_post",  "name": "Instagram Post",          "width": 1080, "height": 1080, "aspect": "1:1"},
        {"id": "instagram_land",  "name": "Instagram Landscape",     "width": 1080, "height": 566,  "aspect": "1.91:1"},
        {"id": "youtube_short",   "name": "YouTube Short",           "width": 1080, "height": 1920, "aspect": "9:16"},
        {"id": "youtube",         "name": "YouTube (1080p)",         "width": 1920, "height": 1080, "aspect": "16:9"},
        {"id": "youtube_4k",      "name": "YouTube (4K)",            "width": 3840, "height": 2160, "aspect": "16:9"},
        {"id": "twitter",         "name": "Twitter / X",             "width": 1920, "height": 1080, "aspect": "16:9"},
        {"id": "square",          "name": "Square",                  "width": 1080, "height": 1080, "aspect": "1:1"},
        {"id": "custom",          "name": "Custom",                  "width": 0,    "height": 0,    "aspect": "custom"},
    ])


@video_bp.route("/video/reframe", methods=["POST"])
@require_csrf
def video_reframe():
    """Reframe/resize video for a target aspect ratio using crop, pad, or scale."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    if not filepath:
        return jsonify({"error": "No file path provided"}), 400

    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    target_w = safe_int(data.get("width", 1080), 1080)
    target_h = safe_int(data.get("height", 1920), 1920)
    _VALID_REFRAME_MODES = {"crop", "pad", "stretch"}
    mode = data.get("mode", "crop")
    if mode not in _VALID_REFRAME_MODES:
        mode = "crop"
    _VALID_POSITIONS = {"center", "top", "bottom", "left", "right", "face"}
    position = data.get("position", "center")
    if position not in _VALID_POSITIONS:
        import re as _re2
        if not _re2.match(r'^\d+:\d+$', position):
            position = "center"
    bg_color = data.get("bg_color", "black")  # for pad mode
    # Sanitize bg_color: allow only alphanumeric, #, and hex digits
    import re as _re
    if not _re.match(r'^[a-zA-Z0-9#]+$', bg_color):
        bg_color = "black"
    quality = data.get("quality", "high")  # low, medium, high
    output_dir = data.get("output_dir", "")

    if target_w < 16 or target_h < 16:
        return jsonify({"error": "Target dimensions too small (min 16x16)"}), 400

    job_id = _new_job("reframe", filepath)

    def _process():
        try:
            _update_job(job_id, progress=5, message="Probing video...")

            # Get source dimensions
            probe_cmd = [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "json", filepath
            ]
            probe_result = _sp.run(probe_cmd, capture_output=True, text=True, timeout=15)
            probe_data = json.loads(probe_result.stdout)
            streams = probe_data.get("streams", [])
            if not streams:
                _update_job(job_id, status="error", error="No video stream found")
                return
            src_w = int(streams[0]["width"])
            src_h = int(streams[0]["height"])

            _update_job(job_id, progress=10, message=f"Source: {src_w}x{src_h} -> Target: {target_w}x{target_h}")

            # Build the video filter chain based on mode
            if mode == "stretch":
                # Simple scale to exact target (distorts if aspect differs)
                vf = f"scale={target_w}:{target_h}"

            elif mode == "pad":
                # Scale to fit inside target, then pad with bg_color
                vf = (
                    f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
                    f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:color={bg_color}"
                )

            else:
                # Crop mode (default) -- scale to cover target, then crop
                src_aspect = src_w / src_h
                tgt_aspect = target_w / target_h

                if abs(src_aspect - tgt_aspect) < 0.01:
                    # Aspects match -- just scale
                    vf = f"scale={target_w}:{target_h}"
                else:
                    # Scale to cover target dimensions, then crop
                    vf = f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"

                    # Determine crop position
                    _effective_pos = position
                    if position == "auto":
                        # Auto-detect content region via cropdetect
                        try:
                            import re as _re
                            cd_cmd = [
                                "ffmpeg", "-i", filepath,
                                "-vf", "cropdetect=24:16:0",
                                "-frames:v", "120", "-f", "null", "-"
                            ]
                            cd_result = _sp.run(cd_cmd, capture_output=True, text=True, timeout=30)
                            crops = _re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", cd_result.stderr)
                            if crops:
                                # Use the last (most stable) detection
                                cw, ch, cx, cy = [int(v) for v in crops[-1]]
                                # Center of the detected content region
                                content_cx = cx + cw // 2
                                content_cy = cy + ch // 2
                                prop_x = content_cx / src_w if src_w else 0.5
                                prop_y = content_cy / src_h if src_h else 0.5
                                # Clamp so the crop window stays inside the frame
                                crop_x = f"max(0,min(iw-{target_w},int(iw*{prop_x:.4f}-{target_w}/2)))"
                                crop_y = f"max(0,min(ih-{target_h},int(ih*{prop_y:.4f}-{target_h}/2)))"
                                _effective_pos = None  # skip named-position branch
                        except Exception:
                            pass
                        if _effective_pos == "auto":
                            _effective_pos = "center"  # fallback

                    if _effective_pos == "top":
                        crop_x, crop_y = "(ow-iw)/2", "0"
                    elif _effective_pos == "bottom":
                        crop_x, crop_y = "(ow-iw)/2", "(oh-ih)"
                    elif _effective_pos == "left":
                        crop_x, crop_y = "0", "(oh-ih)/2"
                    elif _effective_pos == "right":
                        crop_x, crop_y = "(ow-iw)", "(oh-ih)/2"
                    elif _effective_pos is None:
                        pass  # crop_x/crop_y already set by auto-detect
                    else:
                        # center (default)
                        crop_x, crop_y = "(ow-iw)/2", "(oh-ih)/2"

                    vf += f"crop={target_w}:{target_h}:{crop_x}:{crop_y}"

            # Quality mapping
            crf_map = {"low": "28", "medium": "23", "high": "18"}
            crf = crf_map.get(quality, "18")

            effective_dir = _resolve_output_dir(filepath, output_dir)
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            output_path = _unique_output_path(
                os.path.join(effective_dir, f"{base_name}_reframed_{target_w}x{target_h}.mp4")
            )

            _update_job(job_id, progress=20, message=f"Reframing ({mode})...")

            cmd = [
                "ffmpeg", "-i", filepath,
                "-vf", vf,
                "-c:v", "libx264", "-crf", crf, "-preset", "medium",
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart",
                "-y", output_path
            ]
            proc = _sp.run(cmd, capture_output=True, text=True, timeout=600)
            if proc.returncode != 0:
                err_msg = (proc.stderr or "")[-500:]
                _update_job(job_id, status="error", error=f"FFmpeg failed: {err_msg}")
                return

            if not os.path.isfile(output_path):
                _update_job(job_id, status="error", error="Output file not created")
                return

            size_mb = round(os.path.getsize(output_path) / (1024 * 1024), 2)
            _update_job(job_id, status="complete", progress=100,
                        message=f"Reframed to {target_w}x{target_h}",
                        result={
                            "output_path": output_path,
                            "summary": f"{src_w}x{src_h} -> {target_w}x{target_h} ({mode}), {size_mb} MB",
                            "width": target_w,
                            "height": target_h,
                            "size_mb": size_mb,
                            "mode": mode,
                        })
        except Exception as e:
            _update_job(job_id, status="error", error=str(e))

    threading.Thread(target=_process, daemon=True).start()
    return jsonify({"job_id": job_id})


# ---------------------------------------------------------------------------
# Video Merge / Concatenate
# ---------------------------------------------------------------------------
@video_bp.route("/video/merge", methods=["POST"])
@require_csrf
def video_merge():
    """Merge / concatenate multiple video files into one."""
    data = request.get_json(force=True)
    files = data.get("files", [])
    if not files or len(files) < 2:
        return jsonify({"error": "At least 2 files required"}), 400

    # Validate all file paths
    validated_files = []
    for f in files:
        try:
            validated_files.append(validate_filepath(f))
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
    files = validated_files

    merge_mode = data.get("mode", "concat_demux")  # concat_demux | concat_filter
    quality = data.get("quality", "high")
    output_dir = data.get("output_dir", "")

    job_id = _new_job("merge", files[0])

    def _process():
        try:
            _update_job(job_id, progress=5, message="Preparing merge...")

            effective_dir = _resolve_output_dir(files[0], output_dir)
            base_name = os.path.splitext(os.path.basename(files[0]))[0]
            output_path = _unique_output_path(
                os.path.join(effective_dir, f"{base_name}_merged.mp4")
            )

            # Get total duration for progress
            total_dur = 0
            for f in files:
                total_dur += _get_file_duration(f)

            if merge_mode == "concat_filter":
                # Re-encoded merge via filter_complex
                inputs = []
                for f in files:
                    inputs.extend(["-i", f])

                n = len(files)
                filter_parts = "".join(f"[{i}:v][{i}:a]" for i in range(n))
                filter_parts += f"concat=n={n}:v=1:a=1[outv][outa]"

                crf_map = {"low": "28", "medium": "23", "high": "18"}
                crf = crf_map.get(quality, "18")

                cmd = ["ffmpeg"] + inputs + [
                    "-filter_complex", filter_parts,
                    "-map", "[outv]", "-map", "[outa]",
                    "-c:v", "libx264", "-crf", crf, "-preset", "medium",
                    "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart",
                    "-y", output_path
                ]

                _update_job(job_id, progress=10, message="Merging (re-encode)...")
                rc, stderr = _run_ffmpeg_with_progress(job_id, cmd, total_dur)

            else:
                # concat demux (stream copy, fast)
                concat_file = os.path.join(tempfile.gettempdir(), f"opencut_concat_{job_id}.txt")
                try:
                    with open(concat_file, "w", encoding="utf-8") as cf:
                        for f in files:
                            safe = f.replace("'", "'\\''")
                            cf.write(f"file '{safe}'\n")

                    cmd = [
                        "ffmpeg", "-f", "concat", "-safe", "0",
                        "-i", concat_file, "-c", "copy",
                        "-movflags", "+faststart",
                        "-y", output_path
                    ]

                    _update_job(job_id, progress=10, message="Merging (stream copy)...")
                    rc, stderr = _run_ffmpeg_with_progress(job_id, cmd, total_dur)
                finally:
                    try:
                        os.unlink(concat_file)
                    except OSError:
                        pass

            if _is_cancelled(job_id):
                return

            if rc != 0:
                err_msg = (stderr or "")[-500:]
                _update_job(job_id, status="error", error=f"FFmpeg failed: {err_msg}")
                return

            if not os.path.isfile(output_path):
                _update_job(job_id, status="error", error="Output file not created")
                return

            size_mb = round(os.path.getsize(output_path) / (1024 * 1024), 2)
            _update_job(job_id, status="complete", progress=100,
                        message=f"Merged {len(files)} files",
                        result={
                            "output_path": output_path,
                            "summary": f"{len(files)} files merged ({merge_mode}), {size_mb} MB",
                            "size_mb": size_mb,
                            "mode": merge_mode,
                        })
        except Exception as e:
            _update_job(job_id, status="error", error=str(e))

    threading.Thread(target=_process, daemon=True).start()
    return jsonify({"job_id": job_id})


# ---------------------------------------------------------------------------
# Video Trim
# ---------------------------------------------------------------------------
@video_bp.route("/video/trim", methods=["POST"])
@require_csrf
def video_trim():
    """Trim a video to a start/end time range."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()

    if not filepath:
        return jsonify({"error": "File not found"}), 400

    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    start_time = data.get("start", "00:00:00")
    end_time = data.get("end", "")
    quality = data.get("quality", "high")  # low, medium, high, copy
    output_dir = data.get("output_dir", "")

    if not end_time:
        return jsonify({"error": "End time is required"}), 400

    job_id = _new_job("trim", filepath)

    def _process():
        try:
            _update_job(job_id, progress=5, message="Preparing trim...")

            effective_dir = _resolve_output_dir(filepath, output_dir)
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            ext = os.path.splitext(filepath)[1] or ".mp4"
            output_path = _unique_output_path(
                os.path.join(effective_dir, f"{base_name}_trimmed{ext}")
            )

            # Estimate duration for progress
            total_dur = _get_file_duration(filepath)

            if quality == "copy":
                cmd = [
                    "ffmpeg", "-ss", str(start_time), "-to", str(end_time),
                    "-i", filepath,
                    "-c", "copy",
                    "-movflags", "+faststart",
                    "-y", output_path
                ]
            else:
                crf_map = {"low": "28", "medium": "23", "high": "18"}
                crf = crf_map.get(quality, "18")
                cmd = [
                    "ffmpeg", "-ss", str(start_time), "-to", str(end_time),
                    "-i", filepath,
                    "-c:v", "libx264", "-crf", crf,
                    "-c:a", "aac",
                    "-movflags", "+faststart",
                    "-y", output_path
                ]

            _update_job(job_id, progress=10, message="Trimming...")
            rc, stderr = _run_ffmpeg_with_progress(job_id, cmd, total_dur)

            if _is_cancelled(job_id):
                return

            if rc != 0:
                err_msg = (stderr or "")[-500:]
                _update_job(job_id, status="error", error=f"FFmpeg failed: {err_msg}")
                return

            if not os.path.isfile(output_path):
                _update_job(job_id, status="error", error="Output file not created")
                return

            size_mb = round(os.path.getsize(output_path) / (1024 * 1024), 2)
            _update_job(job_id, status="complete", progress=100,
                        message=f"Trimmed {start_time} - {end_time}",
                        result={
                            "output_path": output_path,
                            "summary": f"Trimmed {start_time} to {end_time}, {size_mb} MB",
                            "size_mb": size_mb,
                        })
        except Exception as e:
            _update_job(job_id, status="error", error=str(e))

    threading.Thread(target=_process, daemon=True).start()
    return jsonify({"job_id": job_id})


# ---------------------------------------------------------------------------
# Video Frame Preview (job-based async)
# ---------------------------------------------------------------------------
@video_bp.route("/video/preview-frame", methods=["POST"])
@require_csrf
def preview_frame():
    """Extract a single frame from video at given timestamp, return as base64 JPEG via job."""
    data = request.get_json(force=True)
    file_path = data.get("file", "")
    timestamp = str(data.get("timestamp", "00:00:01"))
    width = safe_int(data.get("width", 640), 640)

    # Validate timestamp format (e.g. "00:01:30", "90", "1:30.5")
    import re as _re
    if not _re.match(r'^[\d]+(?::[\d]+)*(?:\.[\d]+)?$', timestamp):
        return jsonify({"error": "Invalid timestamp format"}), 400

    if not file_path:
        return jsonify({"error": "File not found"}), 400

    try:
        file_path = validate_filepath(file_path)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    job_id = _new_job("preview-frame", file_path)

    def _process():
        import base64
        tmp = os.path.join(tempfile.gettempdir(), f"opencut_preview_{uuid.uuid4().hex[:8]}.jpg")
        try:
            _update_job(job_id, progress=10, message="Extracting frame...")
            cmd = [
                "ffmpeg", "-ss", str(timestamp), "-i", file_path,
                "-vframes", "1", "-vf", f"scale={width}:-1",
                "-q:v", "2", "-y", tmp
            ]
            _sp.run(cmd, capture_output=True, timeout=30)
            if not os.path.isfile(tmp):
                _update_job(job_id, status="error", error="Frame extraction failed")
                return
            with open(tmp, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")
            _update_job(
                job_id, status="complete", progress=100,
                message="Frame extracted",
                result={"image": img_data, "format": "jpeg", "timestamp": str(timestamp)},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e))
        finally:
            if os.path.isfile(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    threading.Thread(target=_process, daemon=True).start()
    return jsonify({"job_id": job_id})


# ---------------------------------------------------------------------------
# Social Platform Quick Presets (loaded from data/social_presets.json)
# ---------------------------------------------------------------------------
@video_bp.route("/export/social-presets", methods=["GET"])
def social_presets():
    import json as _json
    presets_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "social_presets.json")
    try:
        with open(presets_path, "r") as f:
            return jsonify({"presets": _json.load(f)})
    except (FileNotFoundError, _json.JSONDecodeError) as e:
        logger.warning("Could not load social_presets.json: %s", e)
        return jsonify({"presets": []})


# ---------------------------------------------------------------------------
# Auto-Edit (Motion-Based Editing via auto-editor)
# ---------------------------------------------------------------------------
@video_bp.route("/video/auto-edit", methods=["POST"])
@require_csrf
def video_auto_edit():
    """Run auto-editor to detect interesting segments by motion/audio."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400

    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    from opencut.checks import check_auto_editor_available
    if not check_auto_editor_available():
        return jsonify({"error": "auto-editor not installed. Install with: pip install auto-editor"}), 400

    method = data.get("method", "motion").strip()
    if method not in ("motion", "audio", "both"):
        method = "motion"
    threshold = safe_float(data.get("threshold", 0.02), 0.02, min_val=0.001, max_val=1.0)
    margin = safe_float(data.get("margin", 0.2), 0.2, min_val=0.0, max_val=5.0)
    min_clip = safe_float(data.get("min_clip_length", 0.5), 0.5, min_val=0.1, max_val=10.0)
    output_dir = _resolve_output_dir(filepath, data.get("output_dir", ""))

    job_id = _new_job("auto-edit", filepath)

    def _process():
        try:
            from opencut.core.auto_edit import auto_edit

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            result = auto_edit(
                filepath,
                method=method,
                threshold=threshold,
                margin=margin,
                min_clip_length=min_clip,
                export_xml=True,
                output_dir=output_dir,
                on_progress=_on_progress,
            )

            _update_job(
                job_id, status="complete", progress=100,
                message=f"Done: {result.reduction_percent:.0f}% removed",
                result={
                    "total_duration": result.total_duration,
                    "kept_duration": result.kept_duration,
                    "removed_duration": result.removed_duration,
                    "reduction_percent": result.reduction_percent,
                    "segments_count": len([s for s in result.segments if s.action == "keep"]),
                    "xml_path": result.xml_path,
                    "segments": [
                        {"start": s.start, "end": s.end, "action": s.action}
                        for s in result.segments
                    ],
                },
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Auto-edit error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Face-Tracking Auto-Reframe
# ---------------------------------------------------------------------------
@video_bp.route("/video/reframe/face", methods=["POST"])
@require_csrf
def video_reframe_face():
    """Auto-reframe video to keep face centered using MediaPipe face detection."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400

    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    from opencut.checks import check_mediapipe_available
    if not check_mediapipe_available():
        return jsonify({"error": "mediapipe not installed. Install with: pip install mediapipe"}), 400

    target_w = safe_int(data.get("width", 1080), 1080, min_val=100, max_val=7680)
    target_h = safe_int(data.get("height", 1920), 1920, min_val=100, max_val=7680)
    smoothing = safe_float(data.get("smoothing", 0.15), 0.15, min_val=0.01, max_val=1.0)
    face_padding = safe_float(data.get("face_padding", 0.3), 0.3, min_val=0.0, max_val=1.0)
    output_dir = _resolve_output_dir(filepath, data.get("output_dir", ""))

    job_id = _new_job("face-reframe", filepath)

    def _process():
        try:
            from opencut.core.face_reframe import face_reframe

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            output_path = face_reframe(
                filepath,
                target_w=target_w,
                target_h=target_h,
                smoothing=smoothing,
                face_padding=face_padding,
                output_dir=output_dir,
                on_progress=_on_progress,
            )

            _update_job(
                job_id, status="complete", progress=100,
                message="Face reframe complete!",
                result={"output_path": output_path},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Face reframe error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# LLM Highlight Extraction
# ---------------------------------------------------------------------------
@video_bp.route("/video/highlights", methods=["POST"])
@require_csrf
def video_highlights():
    """Extract highlight clips using LLM analysis of transcript."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400

    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    max_highlights = safe_int(data.get("max_highlights", 5), 5, min_val=1, max_val=20)
    min_duration = safe_float(data.get("min_duration", 15.0), 15.0, min_val=5.0, max_val=300.0)
    max_duration = safe_float(data.get("max_duration", 60.0), 60.0, min_val=10.0, max_val=600.0)
    transcript = data.get("transcript", None)

    # LLM config from request
    llm_provider = data.get("llm_provider", "ollama")
    llm_model = data.get("llm_model", "")
    llm_api_key = data.get("llm_api_key", "")
    llm_base_url = data.get("llm_base_url", "")

    job_id = _new_job("highlights", filepath)

    def _process():
        try:
            from opencut.core.highlights import extract_highlights
            from opencut.core.llm import LLMConfig

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            llm_config = LLMConfig(
                provider=llm_provider,
                model=llm_model,
                api_key=llm_api_key,
                base_url=llm_base_url,
            )

            # If no transcript provided, transcribe first
            transcript_segments = transcript
            if not transcript_segments:
                _update_job(job_id, progress=5, message="Transcribing video first...")
                try:
                    from opencut.core.captions import transcribe as _transcribe
                    t_result = _transcribe(filepath)
                    transcript_segments = [
                        {"start": s.start, "end": s.end, "text": s.text}
                        for s in t_result.segments
                    ]
                except Exception as te:
                    raise RuntimeError(f"Transcription failed: {te}")

            result = extract_highlights(
                transcript_segments,
                max_highlights=max_highlights,
                min_duration=min_duration,
                max_duration=max_duration,
                llm_config=llm_config,
                on_progress=_on_progress,
            )

            _update_job(
                job_id, status="complete", progress=100,
                message=f"Found {result.total_found} highlights",
                result={
                    "total_found": result.total_found,
                    "highlights": [
                        {
                            "start": h.start,
                            "end": h.end,
                            "score": h.score,
                            "reason": h.reason,
                            "title": h.title,
                        }
                        for h in result.highlights
                    ],
                },
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Highlight extraction error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# AI LUT Generation from Reference Image
# ---------------------------------------------------------------------------
@video_bp.route("/video/lut/generate-from-ref", methods=["POST"])
@require_csrf
def video_lut_from_ref():
    """Generate a .cube LUT from a reference image's color palette."""
    data = request.get_json(force=True)
    reference_path = data.get("reference_path", "").strip()

    if not reference_path:
        return jsonify({"error": "No reference image path provided"}), 400

    try:
        reference_path = validate_filepath(reference_path)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    lut_name = data.get("lut_name", "").strip() or "custom_ref_lut"
    method = data.get("method", "histogram").strip()
    if method not in ("histogram", "average"):
        method = "histogram"
    strength = safe_float(data.get("strength", 0.8), 0.8, min_val=0.1, max_val=1.0)

    job_id = _new_job("lut-gen-ref", reference_path)

    def _process():
        try:
            from opencut.core.lut_library import generate_lut_from_reference

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            cube_path = generate_lut_from_reference(
                reference_path,
                lut_name=lut_name,
                method=method,
                strength=strength,
                on_progress=_on_progress,
            )

            _update_job(
                job_id, status="complete", progress=100,
                message=f"LUT generated: {os.path.basename(cube_path)}",
                result={"lut_path": cube_path, "lut_name": lut_name},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("LUT generation error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# One-Click Shorts Pipeline
# ---------------------------------------------------------------------------
@video_bp.route("/video/shorts-pipeline", methods=["POST"])
@require_csrf
def video_shorts_pipeline():
    """Generate short-form clips from a long video (transcribe + highlight + reframe + captions)."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400

    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    output_dir = _resolve_output_dir(filepath, data.get("output_dir", ""))

    job_id = _new_job("shorts-pipeline", filepath)

    def _process():
        try:
            from opencut.core.llm import LLMConfig
            from opencut.core.shorts_pipeline import ShortsPipelineConfig, generate_shorts

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            llm_config = LLMConfig(
                provider=data.get("llm_provider", "ollama"),
                model=data.get("llm_model", ""),
                api_key=data.get("llm_api_key", ""),
                base_url=data.get("llm_base_url", ""),
            )

            config = ShortsPipelineConfig(
                whisper_model=data.get("whisper_model", "base"),
                max_shorts=safe_int(data.get("max_shorts", 5), 5, min_val=1, max_val=20),
                min_duration=safe_float(data.get("min_duration", 15.0), 15.0, min_val=5.0, max_val=300.0),
                max_duration=safe_float(data.get("max_duration", 60.0), 60.0, min_val=10.0, max_val=600.0),
                target_w=safe_int(data.get("width", 1080), 1080, min_val=100, max_val=7680),
                target_h=safe_int(data.get("height", 1920), 1920, min_val=100, max_val=7680),
                face_track=bool(data.get("face_track", True)),
                burn_captions=bool(data.get("burn_captions", True)),
                caption_style=data.get("caption_style", "default"),
                llm_provider=llm_config.provider,
                llm_model=llm_config.model,
                llm_api_key=llm_config.api_key,
                llm_base_url=llm_config.base_url,
            )

            clips = generate_shorts(
                filepath,
                config=config,
                output_dir=output_dir,
                on_progress=_on_progress,
            )

            _update_job(
                job_id, status="complete", progress=100,
                message=f"Generated {len(clips)} short clips!",
                result={
                    "clips": [
                        {
                            "index": c.index,
                            "output_path": c.output_path,
                            "start": c.start,
                            "end": c.end,
                            "duration": c.duration,
                            "title": c.title,
                        }
                        for c in clips
                    ],
                    "total_clips": len(clips),
                },
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Shorts pipeline error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})
