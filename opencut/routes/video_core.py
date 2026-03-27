"""
OpenCut Video Core Routes

Watermark detection/removal, scenes, export, merge, trim, preview,
batch processing, social presets.
"""

import logging
import os
import re
import subprocess as _sp
import tempfile
import threading
import time
import uuid

from flask import Blueprint, jsonify, request

from opencut.checks import check_watermark_available
from opencut.errors import safe_error
from opencut.helpers import (
    _get_file_duration,
    _resolve_output_dir,
    _run_ffmpeg_with_progress,
    _unique_output_path,
    get_ffmpeg_path,
)
from opencut.jobs import (
    MAX_BATCH_FILES,
    _is_cancelled,
    _new_job,
    _register_job_process,
    _unregister_job_process,
    _update_job,
    async_job,
)
from opencut.security import (
    require_csrf,
    safe_float,
    safe_int,
    validate_filepath,
)

logger = logging.getLogger("opencut")

video_core_bp = Blueprint("video_core", __name__)

# ---------------------------------------------------------------------------
# Module-level model directory env vars (mirrored from server.py)
# ---------------------------------------------------------------------------
FLORENCE_MODEL_DIR = os.environ.get("OPENCUT_FLORENCE_DIR", None)
LAMA_MODEL_DIR = os.environ.get("OPENCUT_LAMA_DIR", None)


# ---------------------------------------------------------------------------
# Watermark Removal (Florence-2 + LaMA)
# ---------------------------------------------------------------------------
@video_core_bp.route("/video/auto-detect-watermark", methods=["POST"])
@require_csrf
def video_auto_detect_watermark():
    """Auto-detect watermark region using Florence-2 vision model (or edge fallback)."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    prompt = data.get("prompt", "watermark")[:200]

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        from opencut.core.object_removal import detect_watermark_region
        result = detect_watermark_region(filepath, prompt=prompt)
        if result:
            return jsonify(result)
        return jsonify({"error": "No watermark detected", "suggestion": "Try adjusting the prompt or manually specify the region"}), 200
    except ImportError as e:
        return jsonify({"error": str(e), "suggestion": "Install: pip install transformers torch opencv-python-headless"}), 400
    except Exception as exc:
        return safe_error(exc, "video_auto_detect_watermark")


@video_core_bp.route("/video/watermark", methods=["POST"])
@require_csrf
@async_job("watermark")
def video_watermark(job_id, filepath, data):
    """Remove watermarks from video or image using AI."""
    output_dir = data.get("output_dir", "")
    max_bbox_percent = safe_int(data.get("max_bbox_percent", 10), 10, min_val=1, max_val=100)
    detection_prompt = data.get("detection_prompt", "watermark")[:200]
    detection_skip = safe_int(data.get("detection_skip", 3), 3, min_val=1, max_val=30)
    transparent = data.get("transparent", False)
    preview = data.get("preview", False)
    auto_import = data.get("auto_import", True)
    if not check_watermark_available():
        raise ValueError("Watermark remover not installed. Please install it first.")

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

        device = "cpu"
        if torch.cuda.is_available():
            try:
                free_vram = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                if free_vram >= 2 * 1024 ** 3:  # need ~2GB for Florence-2 + LaMA
                    device = "cuda"
                else:
                    logger.info("Low GPU VRAM (%.0f MB free), falling back to CPU for watermark removal",
                                free_vram / 1024 ** 2)
            except Exception:
                device = "cuda"  # can't query, try GPU anyway

        # Load Florence-2 model for detection
        _update_job(job_id, progress=15, message=f"Loading Florence-2 model ({device})...")

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
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {filepath}")
            out_video = None
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
                if not out_video.isOpened():
                    raise RuntimeError("Failed to create video writer for watermark removal")

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
                if out_video is not None:
                    out_video.release()

            # Second pass: process all frames
            _update_job(job_id, progress=60, message="Inpainting frames...")

            cap = cv2.VideoCapture(filepath)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot reopen video: {filepath}")
            out_video = None
            try:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_video = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
                if not out_video.isOpened():
                    raise RuntimeError("Failed to create video writer for inpainting pass")
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
                if out_video is not None:
                    out_video.release()

            # Merge audio if ffmpeg available
            _update_job(job_id, progress=95, message="Merging audio...")
            try:
                _sp.run([
                    get_ffmpeg_path(), '-y',
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

        return {
                "output_path": out_path,
                "auto_import": auto_import
            }

    except Exception as e:
        raise Exception(f"Processing error: {str(e)}")


# ---------------------------------------------------------------------------
# Scene Detection
# ---------------------------------------------------------------------------
@video_core_bp.route("/video/scenes", methods=["POST"])
@require_csrf
@async_job("scenes")
def video_scenes(job_id, filepath, data):
    """Detect scene changes in a video."""
    threshold = safe_float(data.get("threshold", 0.3), 0.3)
    min_scene = safe_float(data.get("min_scene_length", 2.0), 2.0)
    method = data.get("method", "ffmpeg").strip().lower()
    if method not in ("ffmpeg", "ml", "pyscenedetect"):
        method = "ffmpeg"

    from opencut.core.scene_detect import detect_scenes, generate_chapter_markers

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    if method == "ml":
        from opencut.core.scene_detect import detect_scenes_ml
        info = detect_scenes_ml(
            filepath, threshold=threshold,
            min_scene_length=min_scene,
            on_progress=_on_progress,
        )
    elif method == "pyscenedetect":
        from opencut.core.scene_detect import detect_scenes_pyscenedetect
        info = detect_scenes_pyscenedetect(
            filepath, threshold=safe_float(data.get("threshold", 27.0), 27.0, min_val=5.0, max_val=80.0),
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

    return {
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
        }


# ---------------------------------------------------------------------------
# Export Rendered Video (FFmpeg concat)
# ---------------------------------------------------------------------------
@video_core_bp.route("/export-video", methods=["POST"])
@require_csrf
@async_job("export")
def export_video(job_id, filepath, data):
    """Render a new video file with silences removed (no visible cuts)."""
    output_dir = data.get("output_dir", "")
    segments_data = data.get("segments", [])
    if len(segments_data) > 10000:
        raise ValueError("Too many segments (max 10000)")
    if not segments_data:
        raise ValueError("No segments provided")
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
    audio_format_ext = data.get("audio_format", "mp3")  # mp3, wav, flac
    _VALID_VIDEO_FORMATS = {"mp4", "mov", "mkv", "webm", "avi"}
    _VALID_AUDIO_FORMATS = {"mp3", "wav", "flac", "aac", "ogg"}
    if output_format not in _VALID_VIDEO_FORMATS:
        output_format = "mp4"
    if audio_format_ext not in _VALID_AUDIO_FORMATS:
        audio_format_ext = "mp3"
    if audio_only:
        output_format = audio_format_ext

    _update_job(job_id, progress=5, message="Preparing export...")

    try:
        from opencut.utils.media import probe as _probe_media
    except ImportError:
        _probe_media = None

    if _probe_media:
        _probe_media(filepath)

    effective_dir = _resolve_output_dir(filepath, output_dir)

    # Disk space preflight check (need at least 500 MB free)
    from opencut.helpers import check_disk_space
    disk = check_disk_space(effective_dir)
    if not disk["ok"]:
        raise ValueError(f"Not enough disk space. {disk['free_mb']} MB free, need at least {disk['required_mb']} MB.")

    base_name = os.path.splitext(os.path.basename(filepath))[0]
    suffix = "_audio" if audio_only else "_opencut"
    output_path = _unique_output_path(
        os.path.join(effective_dir, f"{base_name}{suffix}.{output_format}")
    )

    # Build filter_complex for segment extraction and concatenation
    total_segments = len(segments_data)

    if total_segments == 0:
        raise ValueError("No segments to export")

    # Determine CRF / quality
    crf_map = {"low": "28", "medium": "23", "high": "18", "lossless": "0"}
    crf = crf_map.get(quality, "23")

    if audio_only:
        # Audio-only: just use atrim + concat (no video)
        filter_parts = []
        concat_inputs = []

        for i, seg in enumerate(segments_data):
            start = safe_float(seg.get("start", 0), 0.0, min_val=0.0)
            end = safe_float(seg.get("end", 0), 0.0, min_val=0.0)
            if end <= start:
                continue
            filter_parts.append(
                f"[0:a]atrim=start={start:.6f}:end={end:.6f},asetpts=PTS-STARTPTS[a{i}]"
            )
            concat_inputs.append(f"[a{i}]")

        if not concat_inputs:
            raise ValueError("No valid segments")

        n = len(concat_inputs)
        concat_str = "".join(concat_inputs) + f"concat=n={n}:v=0:a=1[outa]"
        filter_parts.append(concat_str)
        filter_complex = ";\n".join(filter_parts)

        _update_job(job_id, progress=15, message=f"Rendering {n} segments to {output_format.upper()} (audio only)...")

        cmd = [
            get_ffmpeg_path(), "-hide_banner", "-y",
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
            start = safe_float(seg.get("start", 0), 0.0, min_val=0.0)
            end = safe_float(seg.get("end", 0), 0.0, min_val=0.0)
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
            raise ValueError("No valid segments after filtering")

        n = len(concat_inputs)
        concat_str = "".join(concat_inputs) + f"concat=n={n}:v=1:a=1[outv][outa]"
        filter_parts.append(concat_str)
        filter_complex = ";\n".join(filter_parts)

        _update_job(job_id, progress=15, message=f"Rendering {n} segments to {output_format.upper()}...")

        cmd = [
            get_ffmpeg_path(), "-hide_banner", "-y",
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
    # Use separate stderr pipe so error diagnostics aren't lost
    proc = _sp.Popen(
        cmd + ["-progress", "pipe:1"],
        stdout=_sp.PIPE, stderr=_sp.PIPE, text=True
    )
    # Drain stderr in background to prevent pipe deadlock
    _stderr_chunks = []

    def _drain_stderr():
        try:
            for _line in proc.stderr:
                _stderr_chunks.append(_line)
                if len(_stderr_chunks) > 200:
                    _stderr_chunks.pop(0)
        except Exception:
            pass

    _stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    _stderr_thread.start()
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

    # Timeout: 2x expected duration + 5min base, capped at 4 hours
    wait_timeout = min(14400, max(300, int(total_kept * 2) + 300)) if total_kept > 0 else 7200
    try:
        proc.wait(timeout=wait_timeout)
    except Exception:
        proc.kill()
        proc.wait(timeout=10)
    _stderr_thread.join(timeout=5)
    _unregister_job_process(job_id)

    if proc.returncode != 0:
        stderr_tail = "".join(_stderr_chunks[-20:]).strip()
        # Clean up partial/corrupt output file
        if os.path.exists(output_path):
            try:
                os.unlink(output_path)
            except OSError:
                pass
        raise ValueError(f"FFmpeg exited with code {proc.returncode}: {stderr_tail[:500]}")

    # Get output file size
    output_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
    size_mb = output_size / (1024 * 1024)

    return {
            "output_path": output_path,
            "segments": n,
            "duration": round(total_kept, 2),
            "file_size_mb": round(size_mb, 1),
            "codec": video_codec,
        }


# ---------------------------------------------------------------------------
# Export Presets
# ---------------------------------------------------------------------------
@video_core_bp.route("/export/presets", methods=["GET"])
def export_presets_list():
    """Return available export presets."""
    try:
        from opencut.core.export_presets import get_export_presets, get_preset_categories
        return jsonify({
            "presets": get_export_presets(),
            "categories": get_preset_categories(),
        })
    except Exception as e:
        return safe_error(e, "export_presets_list")


@video_core_bp.route("/export/preset", methods=["POST"])
@require_csrf
@async_job("export_preset")
def export_with_preset_route(job_id, filepath, data):
    """Export video using a preset profile."""
    output_dir = data.get("output_dir", "")
    preset_name = data.get("preset", "youtube_1080p")

    from opencut.core.export_presets import export_with_preset

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    effective_dir = _resolve_output_dir(filepath, output_dir)
    out = export_with_preset(
        filepath, preset_name,
        output_dir=effective_dir,
        on_progress=_on_progress,
    )
    size_mb = os.path.getsize(out) / (1024 * 1024)
    return {"output_path": out, "preset": preset_name, "size_mb": round(size_mb, 1)}


# ---------------------------------------------------------------------------
# Auto-Thumbnail Generation
# ---------------------------------------------------------------------------
@video_core_bp.route("/export/thumbnails", methods=["POST"])
@require_csrf
@async_job("thumbnails")
def generate_thumbnails_route(job_id, filepath, data):
    """Generate thumbnail candidates from video."""
    output_dir = data.get("output_dir", "")
    count = safe_int(data.get("count", 5), 5, min_val=1, max_val=20)
    width = safe_int(data.get("width", 1920), 1920, min_val=160, max_val=3840)
    use_faces = data.get("use_faces", True)

    from opencut.core.thumbnail import generate_thumbnails

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    effective_dir = _resolve_output_dir(filepath, output_dir)
    results = generate_thumbnails(
        filepath, output_dir=effective_dir,
        count=count, width=width,
        use_faces=use_faces,
        on_progress=_on_progress,
    )
    return {"thumbnails": results}


# ---------------------------------------------------------------------------
# Batch Processing
# ---------------------------------------------------------------------------
@video_core_bp.route("/batch/create", methods=["POST"])
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

    parallel = data.get("parallel", True)

    try:
        from opencut.core.batch_process import (
            create_batch,
            finalize_batch,
            process_batch_parallel,
            update_batch_item,
        )

        batch_id = str(uuid.uuid4()).replace("-", "")[:16]
        batch = create_batch(batch_id, operation, filepaths, params)

        if parallel:
            # Process items concurrently via ThreadPoolExecutor
            def _process_batch_parallel():
                process_batch_parallel(
                    batch_id, filepaths, operation, params,
                )

            thread = threading.Thread(target=_process_batch_parallel, daemon=True)
        else:
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

        return jsonify({
            "batch_id": batch_id,
            "status": "running",
            "total": batch.total,
            "parallel": parallel,
        })
    except Exception as e:
        return safe_error(e, "batch_create")


@video_core_bp.route("/batch/<batch_id>", methods=["GET"])
def batch_status(batch_id):
    """Get batch job status."""
    try:
        from opencut.core.batch_process import get_batch
        batch = get_batch(batch_id)
        if not batch:
            return jsonify({"error": "Batch not found"}), 404
        return jsonify(batch.summary)
    except Exception as e:
        return safe_error(e, "batch_status")


@video_core_bp.route("/batch/<batch_id>/cancel", methods=["POST"])
@require_csrf
def batch_cancel(batch_id):
    """Cancel a running batch."""
    try:
        from opencut.core.batch_process import cancel_batch
        if cancel_batch(batch_id):
            return jsonify({"status": "cancelled"})
        return jsonify({"error": "Batch not found"}), 404
    except Exception as e:
        return safe_error(e, "batch_cancel")


@video_core_bp.route("/batch/list", methods=["GET"])
def batch_list():
    """List recent batch jobs."""
    try:
        from opencut.core.batch_process import list_batches
        return jsonify({"batches": list_batches()})
    except Exception as e:
        return safe_error(e, "batch_list")


# ---------------------------------------------------------------------------
# Parallel Batch Processing (Phase 5.4)
# ---------------------------------------------------------------------------
@video_core_bp.route("/batch/parallel", methods=["POST"])
@require_csrf
def batch_parallel():
    """Run multiple operations in parallel.

    Accepts a list of operations, each with an endpoint path and payload.
    Operations are executed concurrently using a configurable thread pool.

    Request body::

        {
            "operations": [
                {"endpoint": "/video/stabilize", "payload": {"filepath": "...", ...}},
                {"endpoint": "/audio/denoise",   "payload": {"filepath": "...", ...}},
            ],
            "max_workers": 2
        }
    """
    from opencut.core.batch_executor import BatchExecutor, OperationSpec
    from opencut.jobs import TooManyJobsError

    data = request.get_json(force=True)
    operations = data.get("operations", [])
    max_workers = safe_int(data.get("max_workers", 2), 2, min_val=1, max_val=8)

    if not operations:
        return jsonify({"error": "No operations specified"}), 400
    if len(operations) > MAX_BATCH_FILES:
        return jsonify({"error": f"Too many operations. Maximum is {MAX_BATCH_FILES}."}), 400

    # Build operation specs and validate filepaths
    specs = []
    for i, op in enumerate(operations):
        endpoint = op.get("endpoint", "")
        payload = op.get("payload", {})
        if not endpoint:
            return jsonify({"error": f"Operation {i}: missing endpoint"}), 400
        fp = payload.get("filepath", "")
        if fp:
            try:
                payload["filepath"] = validate_filepath(fp)
            except ValueError as e:
                return jsonify({"error": f"Operation {i}: {e}"}), 400
        specs.append(OperationSpec(endpoint=endpoint, payload=payload))

    # Create a parent job to track overall progress
    try:
        job_id = _new_job("batch_parallel", f"{len(specs)} operations")
    except TooManyJobsError:
        return jsonify({"error": "Too many concurrent jobs. Please wait for existing jobs to finish."}), 429

    executor = BatchExecutor(
        operations=specs,
        max_workers=max_workers,
        job_id=job_id,
    )

    def _run():
        try:
            results = executor.run(_dispatch_parallel_op)
            success = sum(1 for r in results if r.status == "complete")
            failed = sum(1 for r in results if r.status == "error")
            _update_job(
                job_id,
                status="complete",
                progress=100,
                result=executor.summary,
                message=f"Done: {success} succeeded, {failed} failed",
            )
        except Exception as e:
            logger.exception("Parallel batch %s failed", job_id)
            _update_job(job_id, status="error", error=str(e),
                        message=f"Error: {e}")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return jsonify({"job_id": job_id, "total": len(specs), "max_workers": max_workers})


def _dispatch_parallel_op(op, progress_cb):
    """Dispatch a single parallel batch operation to its handler.

    Maps the endpoint path to the appropriate core function, similar
    to ``_execute_batch_item`` but using endpoint + payload style.
    """
    endpoint = op.endpoint
    payload = op.payload
    filepath = payload.get("filepath", "")
    output_dir = payload.get("output_dir", os.path.dirname(filepath) if filepath else "")

    if endpoint in ("/video/stabilize",):
        from opencut.core.video_fx import stabilize_video
        return stabilize_video(
            filepath, output_dir=output_dir,
            smoothing=safe_int(payload.get("smoothing", 10), 10, min_val=1, max_val=100),
            on_progress=progress_cb,
        )
    elif endpoint in ("/audio/denoise",):
        from opencut.core.audio_suite import denoise_audio
        out_path = os.path.join(
            output_dir,
            os.path.splitext(os.path.basename(filepath))[0] + "_denoised" + os.path.splitext(filepath)[1],
        )
        return denoise_audio(
            filepath, output_path=out_path,
            strength=safe_float(payload.get("strength", 0.5), 0.5, min_val=0.0, max_val=1.0),
            on_progress=progress_cb,
        )
    elif endpoint in ("/audio/normalize",):
        from opencut.core.audio_suite import normalize_loudness
        out_path = os.path.join(
            output_dir,
            os.path.splitext(os.path.basename(filepath))[0] + "_normalized" + os.path.splitext(filepath)[1],
        )
        return normalize_loudness(
            filepath, output_path=out_path,
            target_lufs=safe_float(payload.get("target_lufs", -16), -16.0, min_val=-70.0, max_val=0.0),
            on_progress=progress_cb,
        )
    elif endpoint in ("/video/face-blur",):
        from opencut.core.face_tools import blur_faces
        _valid_blur = {"gaussian", "pixelate", "black"}
        _blur_method = payload.get("method", "gaussian")
        if _blur_method not in _valid_blur:
            _blur_method = "gaussian"
        return blur_faces(
            filepath, output_dir=output_dir,
            method=_blur_method,
            strength=safe_int(payload.get("strength", 51), 51, min_val=1, max_val=99),
            on_progress=progress_cb,
        )
    elif endpoint in ("/video/export-preset",):
        from opencut.core.export_presets import export_with_preset
        return export_with_preset(
            filepath, payload.get("preset", "youtube_1080p"),
            output_dir=output_dir,
            on_progress=progress_cb,
        )
    elif endpoint in ("/video/vignette",):
        from opencut.core.video_fx import apply_vignette
        return apply_vignette(
            filepath, output_dir=output_dir,
            intensity=safe_float(payload.get("intensity", 0.5), 0.5, min_val=0.0, max_val=2.0),
            on_progress=progress_cb,
        )
    elif endpoint in ("/video/film-grain",):
        from opencut.core.video_fx import apply_film_grain
        return apply_film_grain(
            filepath, output_dir=output_dir,
            intensity=safe_float(payload.get("intensity", 0.5), 0.5, min_val=0.0, max_val=2.0),
            on_progress=progress_cb,
        )
    elif endpoint in ("/video/letterbox",):
        from opencut.core.video_fx import apply_letterbox
        _valid_aspects = {"2.39:1", "2.35:1", "1.85:1", "16:9", "4:3", "1:1", "21:9"}
        _aspect = payload.get("aspect", "2.39:1")
        if _aspect not in _valid_aspects:
            _aspect = "2.39:1"
        return apply_letterbox(
            filepath, output_dir=output_dir,
            aspect=_aspect,
            on_progress=progress_cb,
        )
    elif endpoint in ("/video/style-transfer",):
        from opencut.core.style_transfer import style_transfer_video
        _valid_styles = {"candy", "mosaic", "rain_princess", "udnie", "starry_night",
                         "la_muse", "the_scream", "pointilism"}
        _style = payload.get("style", "candy")
        if _style not in _valid_styles:
            _style = "candy"
        return style_transfer_video(
            filepath, style_name=_style,
            output_dir=output_dir,
            intensity=safe_float(payload.get("intensity", 1.0), 1.0, min_val=0.0, max_val=2.0),
            on_progress=progress_cb,
        )
    elif endpoint in ("/video/watermark",):
        from opencut.core.object_removal import remove_watermark
        return remove_watermark(
            filepath, output_dir=output_dir,
            on_progress=progress_cb,
        )
    else:
        raise ValueError(f"Unsupported parallel batch endpoint: {endpoint}")


def _execute_batch_item(operation, filepath, params, on_progress):
    """Execute a single batch item based on operation type."""
    output_dir = params.get("output_dir", os.path.dirname(filepath))

    if operation == "denoise":
        from opencut.core.audio_suite import denoise_audio
        out_path = os.path.join(output_dir, os.path.splitext(os.path.basename(filepath))[0] + "_denoised" + os.path.splitext(filepath)[1])
        return denoise_audio(
            filepath, output_path=out_path,
            strength=safe_float(params.get("strength", 0.5), 0.5, min_val=0.0, max_val=1.0),
            on_progress=on_progress,
        )
    elif operation == "normalize":
        from opencut.core.audio_suite import normalize_loudness
        out_path = os.path.join(output_dir, os.path.splitext(os.path.basename(filepath))[0] + "_normalized" + os.path.splitext(filepath)[1])
        return normalize_loudness(
            filepath, output_path=out_path,
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
        _valid_blur = {"gaussian", "pixelate", "black"}
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
        _valid_styles = {"candy", "mosaic", "rain_princess", "udnie", "starry_night", "la_muse", "the_scream", "pointilism"}
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
# Video Merge / Concatenate
# ---------------------------------------------------------------------------
@video_core_bp.route("/video/merge", methods=["POST"])
@require_csrf
@async_job("merge", filepath_required=False)
def video_merge(job_id, filepath, data):
    """Merge / concatenate multiple video files into one."""
    files = data.get("files", [])
    if not files or len(files) < 2:
        raise ValueError("At least 2 files required")
    if len(files) > MAX_BATCH_FILES:
        raise ValueError(f"Too many files (max {MAX_BATCH_FILES})")
    validated_files = []
    for f in files:
        validated_files.append(validate_filepath(f))
    files = validated_files
    merge_mode = data.get("mode", "concat_demux")  # concat_demux | concat_filter
    quality = data.get("quality", "high")
    output_dir = data.get("output_dir", "")

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

        cmd = [get_ffmpeg_path()] + inputs + [
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
                get_ffmpeg_path(), "-f", "concat", "-safe", "0",
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
        raise ValueError(f"FFmpeg failed: {err_msg}")

    if not os.path.isfile(output_path):
        raise ValueError("Output file not created")

    size_mb = round(os.path.getsize(output_path) / (1024 * 1024), 2)
    return {
                    "output_path": output_path,
                    "summary": f"{len(files)} files merged ({merge_mode}), {size_mb} MB",
                    "size_mb": size_mb,
                    "mode": merge_mode,
                }


# ---------------------------------------------------------------------------
# Video Trim
# ---------------------------------------------------------------------------
@video_core_bp.route("/video/trim", methods=["POST"])
@require_csrf
@async_job("trim")
def video_trim(job_id, filepath, data):
    """Trim a video to a start/end time range."""
    start_time = data.get("start", "00:00:00")
    end_time = data.get("end", "")
    quality = data.get("quality", "high")  # low, medium, high, copy
    output_dir = data.get("output_dir", "")
    if not end_time:
        raise ValueError("End time is required")
    _time_re = re.compile(r"^\d{1,3}:\d{2}:\d{2}(?:\.\d+)?$")
    if not _time_re.match(str(start_time)) or not _time_re.match(str(end_time)):
        raise ValueError("Invalid time format. Use HH:MM:SS or HH:MM:SS.xxx")

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
            get_ffmpeg_path(), "-ss", str(start_time), "-to", str(end_time),
            "-i", filepath,
            "-c", "copy",
            "-movflags", "+faststart",
            "-y", output_path
        ]
    else:
        crf_map = {"low": "28", "medium": "23", "high": "18"}
        crf = crf_map.get(quality, "18")
        cmd = [
            get_ffmpeg_path(), "-ss", str(start_time), "-to", str(end_time),
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
        raise ValueError(f"FFmpeg failed: {err_msg}")

    if not os.path.isfile(output_path):
        raise ValueError("Output file not created")

    size_mb = round(os.path.getsize(output_path) / (1024 * 1024), 2)
    return {
                    "output_path": output_path,
                    "summary": f"Trimmed {start_time} to {end_time}, {size_mb} MB",
                    "size_mb": size_mb,
                }


# ---------------------------------------------------------------------------
# Video Frame Preview (job-based async)
# ---------------------------------------------------------------------------
@video_core_bp.route("/video/preview-frame", methods=["POST"])
@require_csrf
@async_job("preview-frame")
def preview_frame(job_id, filepath, data):
    """Extract a single frame from video at given timestamp, return as base64 JPEG via job."""
    timestamp = str(data.get("timestamp", "00:00:01"))
    width = safe_int(data.get("width", 640), 640, min_val=32, max_val=3840)
    import re as _re
    if not _re.match(r'^[\d]+(?::[\d]+)*(?:\.[\d]+)?$', timestamp):
        raise ValueError("Invalid timestamp format")

    import base64
    _fd, tmp = tempfile.mkstemp(suffix=".jpg", prefix="opencut_preview_")
    os.close(_fd)
    try:
        _update_job(job_id, progress=10, message="Extracting frame...")
        cmd = [
            get_ffmpeg_path(), "-ss", str(timestamp), "-i", filepath,
            "-vframes", "1", "-vf", f"scale={width}:-1",
            "-q:v", "2", "-y", tmp
        ]
        _sp.run(cmd, capture_output=True, timeout=30)
        if not os.path.isfile(tmp):
            raise ValueError("Frame extraction failed")
        with open(tmp, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")
        return {"image": img_data, "format": "jpeg", "timestamp": str(timestamp)}
    finally:
        if os.path.isfile(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Social Platform Quick Presets (loaded from data/social_presets.json)
# ---------------------------------------------------------------------------
@video_core_bp.route("/export/social-presets", methods=["GET"])
def social_presets():
    import json as _json
    presets_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "social_presets.json")
    try:
        with open(presets_path, "r", encoding="utf-8") as f:
            return jsonify({"presets": _json.load(f)})
    except (FileNotFoundError, _json.JSONDecodeError) as e:
        logger.warning("Could not load social_presets.json: %s", e)
        return jsonify({"presets": []})
