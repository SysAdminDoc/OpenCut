"""
OpenCut Education & Tutorial Routes

Routes for click/keystroke overlays, callout annotations,
screenshot-to-video, slide detection, PiP lecture processing,
and auto-quiz overlays.
"""

import logging
import os

from flask import Blueprint, jsonify, request

from opencut.errors import safe_error
from opencut.jobs import _update_job, async_job
from opencut.security import (
    require_csrf,
    safe_bool,
    safe_float,
    safe_int,
    validate_filepath,
)

logger = logging.getLogger("opencut")

education_bp = Blueprint("education", __name__)


# =========================================================================
# Feature 11.2 - Click & Keystroke Overlay
# =========================================================================

@education_bp.route("/api/tutorial/click-overlay", methods=["POST"])
@require_csrf
@async_job("click_overlay")
def click_overlay(job_id, filepath, data):
    """Render click ripple animations on a video."""
    from opencut.core.click_overlay import ClickEvent, render_click_overlay

    raw_events = data.get("click_events", [])
    if not raw_events:
        raise ValueError("No click_events provided")

    events = []
    for e in raw_events:
        events.append(ClickEvent(
            timestamp=float(e["timestamp"]),
            x=int(e["x"]),
            y=int(e["y"]),
            button=str(e.get("button", "left")),
            duration=float(e.get("duration", 0.4)),
        ))

    max_radius = safe_int(data.get("max_radius", 30), 30, min_val=5, max_val=100)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    return render_click_overlay(
        video_path=filepath,
        click_events=events,
        output_path_str=data.get("output_path"),
        max_radius=max_radius,
        on_progress=_progress,
    )


@education_bp.route("/api/tutorial/keystroke-overlay", methods=["POST"])
@require_csrf
@async_job("keystroke_overlay")
def keystroke_overlay(job_id, filepath, data):
    """Render keystroke badge overlays on a video."""
    from opencut.core.click_overlay import KeystrokeEvent, render_keystroke_overlay

    raw_events = data.get("keystroke_events", [])
    if not raw_events:
        raise ValueError("No keystroke_events provided")

    events = []
    for e in raw_events:
        events.append(KeystrokeEvent(
            timestamp=float(e["timestamp"]),
            keys=str(e["keys"]),
            duration=float(e.get("duration", 1.5)),
        ))

    position = (data.get("position") or "bottom-left").strip()
    font_size = safe_int(data.get("font_size", 28), 28, min_val=12, max_val=120)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    return render_keystroke_overlay(
        video_path=filepath,
        keystroke_events=events,
        output_path_str=data.get("output_path"),
        position=position,
        font_size=font_size,
        font_color=data.get("font_color", "white"),
        bg_color=data.get("bg_color", "black"),
        bg_opacity=safe_float(data.get("bg_opacity", 0.7), 0.7, min_val=0.0, max_val=1.0),
        on_progress=_progress,
    )


@education_bp.route("/api/tutorial/parse-click-log", methods=["POST"])
@require_csrf
def parse_click_log_route():
    """Parse a click/keystroke log file (sync)."""
    data = request.get_json(force=True) or {}
    log_path = (data.get("log_path") or "").strip()
    if not log_path:
        return jsonify({"error": "No log_path provided"}), 400

    try:
        log_path = validate_filepath(log_path)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        from opencut.core.click_overlay import parse_click_log
        result = parse_click_log(log_path)
        return jsonify({
            "clicks": [
                {"timestamp": c.timestamp, "x": c.x, "y": c.y,
                 "button": c.button, "duration": c.duration}
                for c in result["clicks"]
            ],
            "keystrokes": [
                {"timestamp": k.timestamp, "keys": k.keys, "duration": k.duration}
                for k in result["keystrokes"]
            ],
        })
    except (FileNotFoundError, ValueError) as e:
        return jsonify({"error": str(e)}), 400
    except Exception as exc:
        return safe_error(exc, "parse_click_log")


# =========================================================================
# Feature 11.3 - Callout & Annotation Generator
# =========================================================================

@education_bp.route("/api/tutorial/callout", methods=["POST"])
@require_csrf
@async_job("callout")
def generate_callout_route(job_id, filepath, data):
    """Generate callout annotations on a video."""
    from opencut.core.callout_gen import Annotation, Region, generate_callout

    raw_annotations = data.get("annotations", [])
    if not raw_annotations:
        raise ValueError("No annotations provided")

    annotations = []
    for a in raw_annotations:
        region = None
        if a.get("region"):
            r = a["region"]
            region = Region(
                x=int(r["x"]), y=int(r["y"]),
                w=int(r["w"]), h=int(r["h"]),
            )

        arrow_from = None
        arrow_to = None
        if a.get("arrow_from"):
            arrow_from = (int(a["arrow_from"][0]), int(a["arrow_from"][1]))
        if a.get("arrow_to"):
            arrow_to = (int(a["arrow_to"][0]), int(a["arrow_to"][1]))

        annotations.append(Annotation(
            type=str(a.get("type", "callout")),
            start_time=float(a["start_time"]),
            end_time=float(a["end_time"]),
            region=region,
            text=str(a.get("text", "")),
            number=int(a.get("number", 0)),
            color=str(a.get("color", "yellow")),
            font_size=safe_int(a.get("font_size", 24), 24, min_val=8, max_val=200),
            arrow_from=arrow_from,
            arrow_to=arrow_to,
        ))

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    return generate_callout(
        video_path=filepath,
        annotations=annotations,
        output_path_str=data.get("output_path"),
        on_progress=_progress,
    )


@education_bp.route("/api/tutorial/spotlight", methods=["POST"])
@require_csrf
@async_job("spotlight")
def create_spotlight_route(job_id, filepath, data):
    """Create a spotlight effect on a video region."""
    from opencut.core.callout_gen import Region, create_spotlight

    region_data = data.get("region")
    if not region_data:
        raise ValueError("No region provided")

    region = Region(
        x=int(region_data["x"]),
        y=int(region_data["y"]),
        w=int(region_data["w"]),
        h=int(region_data["h"]),
    )

    start = safe_float(data.get("start_time", 0), 0, min_val=0.0)
    end = safe_float(data.get("end_time"), None, min_val=0.0)
    if end is None or end <= start:
        raise ValueError("end_time must be greater than start_time")

    darkness = safe_float(data.get("darkness", 0.6), 0.6, min_val=0.0, max_val=1.0)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    return create_spotlight(
        video_path=filepath,
        region=region,
        timestamp_range=(start, end),
        output_path_str=data.get("output_path"),
        border_color=data.get("border_color", "yellow"),
        darkness=darkness,
        on_progress=_progress,
    )


@education_bp.route("/api/tutorial/step-callout", methods=["POST"])
@require_csrf
def create_step_callout_route():
    """Create a step callout configuration (sync, no job)."""
    data = request.get_json(force=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    number = safe_int(data.get("number", 1), 1, min_val=1)
    style = (data.get("style") or "circle").strip()

    try:
        from opencut.core.callout_gen import create_step_callout
        result = create_step_callout(
            text=text,
            number=number,
            style=style,
            color=data.get("color", "yellow"),
            bg_color=data.get("bg_color", "black"),
            font_size=safe_int(data.get("font_size", 24), 24, min_val=8, max_val=200),
        )
        return jsonify(result.to_dict())
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as exc:
        return safe_error(exc, "create_step_callout")


# =========================================================================
# Feature 11.4 - Screenshot-to-Video with Ken Burns
# =========================================================================

@education_bp.route("/api/tutorial/screenshot-video", methods=["POST"])
@require_csrf
@async_job("screenshot_video", filepath_required=False)
def screenshot_video(job_id, filepath, data):
    """Create a video from screenshots with Ken Burns effect."""
    from opencut.core.screenshot_video import create_screenshot_video

    image_paths = data.get("image_paths", [])
    if not image_paths:
        raise ValueError("No image_paths provided")

    # Validate all image paths
    validated = []
    for p in image_paths:
        p = p.strip()
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Image not found: {p}")
        validated.append(p)

    output = data.get("output_path")
    if not output:
        raise ValueError("No output_path provided")

    duration_per = safe_float(data.get("duration_per_image", 5.0), 5.0, min_val=1.0, max_val=60.0)
    width = safe_int(data.get("width", 1920), 1920, min_val=320, max_val=7680)
    height = safe_int(data.get("height", 1080), 1080, min_val=240, max_val=4320)
    fps = safe_int(data.get("fps", 30), 30, min_val=15, max_val=60)
    transition = (data.get("transition") or "fade").strip()
    transition_dur = safe_float(data.get("transition_duration", 0.5), 0.5, min_val=0.0, max_val=3.0)
    enable_kb = safe_bool(data.get("enable_ken_burns", True), True)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    return create_screenshot_video(
        image_paths=validated,
        output_path_str=output,
        duration_per_image=duration_per,
        resolution=(width, height),
        fps=fps,
        transition=transition,
        transition_duration=transition_dur,
        enable_ken_burns=enable_kb,
        on_progress=_progress,
    )


@education_bp.route("/api/tutorial/detect-roi", methods=["POST"])
@require_csrf
def detect_roi_route():
    """Detect regions of interest in an image (sync)."""
    data = request.get_json(force=True) or {}
    image_path = (data.get("image_path") or "").strip()
    if not image_path:
        return jsonify({"error": "No image_path provided"}), 400

    try:
        image_path = validate_filepath(image_path)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        from opencut.core.screenshot_video import detect_roi
        rois = detect_roi(image_path)
        return jsonify({"rois": rois, "count": len(rois)})
    except (FileNotFoundError, ValueError) as e:
        return jsonify({"error": str(e)}), 400
    except Exception as exc:
        return safe_error(exc, "detect_roi")


@education_bp.route("/api/tutorial/ken-burns-keyframes", methods=["POST"])
@require_csrf
def ken_burns_keyframes_route():
    """Generate Ken Burns keyframes from ROIs (sync)."""
    data = request.get_json(force=True) or {}
    rois = data.get("rois", [])
    if not rois:
        return jsonify({"error": "No rois provided"}), 400

    duration = safe_float(data.get("duration", 5.0), 5.0, min_val=0.5)
    min_zoom = safe_float(data.get("min_zoom", 1.0), 1.0, min_val=0.5, max_val=3.0)
    max_zoom = safe_float(data.get("max_zoom", 1.5), 1.5, min_val=1.0, max_val=5.0)

    try:
        from opencut.core.screenshot_video import generate_ken_burns_keyframes
        keyframes = generate_ken_burns_keyframes(
            rois=rois,
            duration=duration,
            min_zoom=min_zoom,
            max_zoom=max_zoom,
        )
        return jsonify({"keyframes": keyframes, "count": len(keyframes)})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as exc:
        return safe_error(exc, "ken_burns_keyframes")


# =========================================================================
# Feature 33.1 - Slide Change Detection
# =========================================================================

@education_bp.route("/api/education/slide-detect", methods=["POST"])
@require_csrf
@async_job("slide_detect")
def slide_detect(job_id, filepath, data):
    """Detect slide transitions in a screen recording."""
    from opencut.core.slide_detect import detect_slide_changes

    threshold = safe_float(data.get("threshold", 0.3), 0.3, min_val=0.05, max_val=0.95)
    min_interval = safe_float(data.get("min_interval", 1.0), 1.0, min_val=0.1, max_val=30.0)
    max_slides = safe_int(data.get("max_slides", 200), 200, min_val=1, max_val=1000)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    return detect_slide_changes(
        video_path=filepath,
        threshold=threshold,
        min_interval=min_interval,
        max_slides=max_slides,
        on_progress=_progress,
    )


@education_bp.route("/api/education/slide-extract", methods=["POST"])
@require_csrf
@async_job("slide_extract")
def slide_extract(job_id, filepath, data):
    """Extract slide images at specific timestamps."""
    from opencut.core.slide_detect import extract_slide_images

    timestamps = data.get("timestamps", [])
    if not timestamps:
        raise ValueError("No timestamps provided")

    output_dir = (data.get("output_dir") or "").strip()
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(filepath), "slides")

    image_format = (data.get("image_format") or "png").strip().lower()

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    return extract_slide_images(
        video_path=filepath,
        timestamps=[float(t) for t in timestamps],
        output_dir=output_dir,
        image_format=image_format,
        on_progress=_progress,
    )


@education_bp.route("/api/education/slide-chapters", methods=["POST"])
@require_csrf
def slide_chapters_route():
    """Generate chapter markers from slide timestamps (sync)."""
    data = request.get_json(force=True) or {}
    timestamps = data.get("timestamps", [])
    if not timestamps:
        return jsonify({"error": "No timestamps provided"}), 400

    try:
        timestamps = [float(t) for t in timestamps]
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid timestamp value: {e}"}), 400

    video_duration = safe_float(data.get("video_duration", 0), 0, min_val=0.0)
    title_prefix = (data.get("title_prefix") or "Slide").strip()
    custom_titles = data.get("custom_titles")

    try:
        from opencut.core.slide_detect import generate_slide_chapters
        chapters = generate_slide_chapters(
            timestamps=timestamps,
            video_duration=video_duration,
            title_prefix=title_prefix,
            custom_titles=custom_titles,
        )
        return jsonify({"chapters": chapters, "count": len(chapters)})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as exc:
        return safe_error(exc, "slide_chapters")


# =========================================================================
# Feature 33.2 - PiP Lecture Processing
# =========================================================================

@education_bp.route("/api/education/pip-detect", methods=["POST"])
@require_csrf
@async_job("pip_detect")
def pip_detect(job_id, filepath, data):
    """Detect PiP camera region in a video."""
    from opencut.core.pip_lecture import detect_pip_region

    sample_count = safe_int(data.get("sample_count", 5), 5, min_val=1, max_val=20)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    return detect_pip_region(
        video_path=filepath,
        sample_count=sample_count,
        on_progress=_progress,
    )


@education_bp.route("/api/education/pip-extract", methods=["POST"])
@require_csrf
@async_job("pip_extract")
def pip_extract(job_id, filepath, data):
    """Extract speaker and screen streams from PiP video."""
    from opencut.core.pip_lecture import extract_pip_streams

    pip_region = data.get("pip_region")
    if not pip_region:
        raise ValueError("No pip_region provided")

    output_dir = (data.get("output_dir") or "").strip()
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(filepath), "pip_output")

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    return extract_pip_streams(
        video_path=filepath,
        pip_region=pip_region,
        output_dir=output_dir,
        on_progress=_progress,
    )


@education_bp.route("/api/education/pip-side-by-side", methods=["POST"])
@require_csrf
@async_job("pip_side_by_side", filepath_required=False)
def pip_side_by_side(job_id, filepath, data):
    """Create side-by-side layout from speaker and screen videos."""
    from opencut.core.pip_lecture import create_side_by_side

    speaker_path = (data.get("speaker_path") or "").strip()
    screen_path = (data.get("screen_path") or "").strip()
    output = (data.get("output_path") or "").strip()

    if not speaker_path:
        raise ValueError("No speaker_path provided")
    if not screen_path:
        raise ValueError("No screen_path provided")
    if not output:
        raise ValueError("No output_path provided")

    # Validate paths
    if not os.path.isfile(speaker_path):
        raise FileNotFoundError(f"Speaker video not found: {speaker_path}")
    if not os.path.isfile(screen_path):
        raise FileNotFoundError(f"Screen video not found: {screen_path}")

    layout = (data.get("layout") or "speaker-left").strip()
    speaker_scale = safe_float(data.get("speaker_scale", 0.33), 0.33, min_val=0.1, max_val=0.5)
    width = safe_int(data.get("width", 1920), 1920, min_val=640, max_val=7680)
    height = safe_int(data.get("height", 1080), 1080, min_val=480, max_val=4320)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    return create_side_by_side(
        speaker_path=speaker_path,
        screen_path=screen_path,
        output_path_str=output,
        layout=layout,
        speaker_scale=speaker_scale,
        output_width=width,
        output_height=height,
        on_progress=_progress,
    )


# =========================================================================
# Feature 33.3 - Auto-Quiz Overlay
# =========================================================================

@education_bp.route("/api/education/quiz-generate", methods=["POST"])
@require_csrf
def quiz_generate_route():
    """Generate quiz questions from a transcript (sync)."""
    data = request.get_json(force=True) or {}
    transcript = (data.get("transcript") or "").strip()
    if not transcript:
        return jsonify({"error": "No transcript provided"}), 400

    count = safe_int(data.get("count", 5), 5, min_val=1, max_val=50)
    difficulty = (data.get("difficulty") or "medium").strip().lower()

    try:
        from opencut.core.quiz_overlay import generate_quiz_questions
        questions = generate_quiz_questions(
            transcript=transcript,
            count=count,
            difficulty=difficulty,
        )
        return jsonify({"questions": questions, "count": len(questions)})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as exc:
        return safe_error(exc, "quiz_generate")


@education_bp.route("/api/education/quiz-render", methods=["POST"])
@require_csrf
@async_job("quiz_render", filepath_required=False)
def quiz_render(job_id, filepath, data):
    """Render quiz questions as an overlay video."""
    from opencut.core.quiz_overlay import render_quiz_overlay

    questions = data.get("questions", [])
    if not questions:
        raise ValueError("No questions provided")

    output = (data.get("output_path") or "").strip()
    if not output:
        raise ValueError("No output_path provided")

    width = safe_int(data.get("width", 1920), 1920, min_val=640, max_val=7680)
    height = safe_int(data.get("height", 1080), 1080, min_val=480, max_val=4320)
    display_dur = safe_float(data.get("display_duration", 10.0), 10.0, min_val=3.0, max_val=60.0)
    font_size = safe_int(data.get("font_size", 32), 32, min_val=12, max_val=120)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    return render_quiz_overlay(
        question_data=questions,
        output_path_str=output,
        resolution=(width, height),
        display_duration=display_dur,
        bg_color=data.get("bg_color", "black"),
        text_color=data.get("text_color", "white"),
        highlight_color=data.get("highlight_color", "00FF00"),
        font_size=font_size,
        on_progress=_progress,
    )


@education_bp.route("/api/education/quiz-insert", methods=["POST"])
@require_csrf
@async_job("quiz_insert")
def quiz_insert(job_id, filepath, data):
    """Insert quiz overlays at chapter boundaries in a video."""
    from opencut.core.quiz_overlay import insert_quiz_at_chapters

    questions = data.get("questions", [])
    if not questions:
        raise ValueError("No questions provided")

    chapters = data.get("chapters", [])
    if not chapters:
        raise ValueError("No chapters provided")

    output = (data.get("output_path") or "").strip()
    if not output:
        from opencut.helpers import output_path as _output_path
        output = _output_path(filepath, "quiz")

    quiz_dur = safe_float(data.get("quiz_duration", 10.0), 10.0, min_val=3.0, max_val=60.0)
    position = (data.get("position") or "after").strip().lower()

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    return insert_quiz_at_chapters(
        video_path=filepath,
        questions=questions,
        chapters=chapters,
        output_path_str=output,
        quiz_duration=quiz_dur,
        position=position,
        on_progress=_progress,
    )
