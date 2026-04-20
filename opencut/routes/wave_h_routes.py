"""
OpenCut Wave H Routes (v1.25.0) — Commercial Parity & Content-Creator Polish

Tier 1 (fully working):
  POST /analyze/virality              — multimodal 0-100 virality score
  POST /analyze/virality/rank         — rank a batch of candidate clips
  POST /video/cursor-zoom/resolve     — resolve click events for cursor zoom
  GET  /system/changelog/latest       — fetch recent GitHub releases
  GET  /system/changelog/unseen       — releases newer than last-seen tag
  POST /system/changelog/mark-seen    — persist last-seen release tag
  GET  /system/issue-report/bundle    — pre-filled GitHub issue URL
  POST /system/issue-report/bundle    — same, with custom title + description
  GET  /system/demo/list              — list bundled demo assets
  GET  /system/demo/sample            — return primary sample.mp4 path
  POST /system/demo/download          — download sample from GitHub release
  POST /settings/gist/push            — export presets to a GitHub Gist
  POST /settings/gist/pull            — import presets from a GitHub Gist
  GET  /settings/gist/info            — gist metadata without pulling files
  GET  /settings/onboarding           — read onboarding state
  POST /settings/onboarding           — patch onboarding state

Tier 2 (stubbed — 503 MISSING_DEPENDENCY):
  POST /video/upscale/flashvsr        — FlashVSR VSR
  GET  /video/upscale/flashvsr/info
  POST /video/inpaint/rose            — ROSE shadow-preserving inpaint
  POST /video/matte/sammie            — Sammie-Roto-2 temporal rotoscope
  POST /audio/tts/omnivoice           — OmniVoice 600+ language TTS
  GET  /audio/tts/omnivoice/models
  POST /video/style/reezsynth         — ReEzSynth temporal style transfer
  POST /audio/music/vidmuse           — VidMuse video-to-music

Tier 3 (stubbed — 501 ROUTE_STUBBED):
  POST /agent/search-footage          — VideoAgent concept search
  POST /agent/storyboard              — ViMax script-to-storyboard
  POST /generate/cloud/submit         — Hailuo / Seedance cloud gen-video
  GET  /generate/cloud/status/<id>
  GET  /generate/cloud/backends
  POST /lipsync/gaussian              — GaussianHeadTalk lip-sync
  POST /lipsync/fantasy2              — FantasyTalking2 lip-sync
  GET  /lipsync/advanced/backends
  GET  /system/qe-reflect             — QE API reflection probe result
"""

from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request

from opencut.errors import error_response, safe_error
from opencut.jobs import _update_job, async_job
from opencut.security import require_csrf, safe_bool, safe_int, validate_path

logger = logging.getLogger("opencut")

wave_h_bp = Blueprint("wave_h", __name__)


# ===========================================================================
# Shared helpers
# ===========================================================================

def _stub_503(name: str, hint: str):
    return error_response(
        "MISSING_DEPENDENCY",
        f"{name} backend is not installed on this server.",
        status=503,
        suggestion=hint,
    )


def _stub_501(name: str, hint: str = ""):
    return error_response(
        "ROUTE_STUBBED",
        f"{name} is a Wave H Tier 3 strategic stub in v1.25.0.",
        status=501,
        suggestion=hint or "Track the ROADMAP-NEXT.md Wave H section.",
    )


# ===========================================================================
# Tier 1.1 — Virality scoring
# ===========================================================================

@wave_h_bp.route("/analyze/virality", methods=["POST"])
@require_csrf
@async_job("virality_score")
def route_virality_score(job_id, filepath, data):
    """Score a single clip 0-100 across three heuristic signals."""
    from opencut.core import virality_score

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=int(pct), message=str(msg))

    weights = data.get("weights") if isinstance(data.get("weights"), dict) else None
    segments = data.get("segments") if isinstance(data.get("segments"), list) else None
    transcript = str(data.get("transcript") or "")
    skip_visual = safe_bool(data.get("skip_visual"), False)

    llm_config = None
    cfg = data.get("llm_config")
    if isinstance(cfg, dict):
        try:
            from opencut.core.llm import LLMConfig
            llm_config = LLMConfig(
                provider=str(cfg.get("provider") or "ollama"),
                model=str(cfg.get("model") or "llama3"),
                api_key=str(cfg.get("api_key") or ""),
            )
        except Exception:  # noqa: BLE001
            llm_config = None

    result = virality_score.score(
        filepath=filepath,
        segments=segments,
        transcript=transcript,
        llm_config=llm_config,
        weights=weights,
        skip_visual=skip_visual,
        on_progress=_on_progress,
    )
    return {
        "score": result.score,
        "signals": dict(result.signals.__dict__),
        "weights": dict(result.weights),
        "hook_phrase": result.hook_phrase,
        "duration": result.duration,
        "notes": list(result.notes),
    }


@wave_h_bp.route("/analyze/virality/rank", methods=["POST"])
@require_csrf
@async_job("virality_rank", filepath_required=False)
def route_virality_rank(job_id, filepath, data):
    """Rank a batch of candidate clips by virality score, score-desc."""
    from opencut.core import virality_score

    candidates = data.get("candidates") or []
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidates must be a non-empty list")

    # Validate each candidate's filepath up-front.
    cleaned = []
    for i, c in enumerate(candidates):
        if not isinstance(c, dict):
            raise ValueError(f"candidate {i} must be an object")
        fp = validate_path(str(c.get("filepath") or ""))
        cleaned.append({**c, "filepath": fp})

    weights = data.get("weights") if isinstance(data.get("weights"), dict) else None
    skip_visual = safe_bool(data.get("skip_visual"), False)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=int(pct), message=str(msg))

    ranked = virality_score.rank(
        cleaned,
        weights=weights,
        skip_visual=skip_visual,
        on_progress=_on_progress,
    )
    return {"ranked": ranked, "count": len(ranked)}


# ===========================================================================
# Tier 1.2 — Cursor zoom event resolution
# ===========================================================================

@wave_h_bp.route("/video/cursor-zoom/resolve", methods=["POST"])
@require_csrf
@async_job("cursor_zoom_resolve")
def route_cursor_zoom_resolve(job_id, filepath, data):
    """Resolve click events via sidecar → inline events → frame-diff."""
    from opencut.core import cursor_zoom

    sidecar = data.get("sidecar_path") or ""
    if sidecar:
        sidecar = validate_path(str(sidecar))
    events = data.get("events") if isinstance(data.get("events"), list) else None
    allow_framediff = safe_bool(data.get("allow_framediff"), True)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=int(pct), message=str(msg))

    result = cursor_zoom.resolve_click_regions(
        input_path=filepath,
        sidecar_path=sidecar or None,
        events=events,
        allow_framediff=allow_framediff,
        on_progress=_on_progress,
    )
    # ClickRegion is a dataclass — expand for jsonify.
    clicks = [
        {"t": cr.timestamp, "x": cr.x, "y": cr.y, "confidence": cr.confidence}
        for cr in result["click_regions"]
    ]
    return {
        "click_regions": clicks,
        "source": result["source"],
        "width": result["width"],
        "height": result["height"],
        "duration": result["duration"],
        "fps": result["fps"],
        "notes": result["notes"],
    }


# ===========================================================================
# Tier 1.4 — Changelog feed
# ===========================================================================

@wave_h_bp.route("/system/changelog/latest", methods=["GET"])
def route_changelog_latest():
    try:
        from opencut.core import changelog_feed
        limit = safe_int(request.args.get("limit", 5), 5, min_val=1, max_val=25)
        return jsonify(changelog_feed.fetch_releases(limit=limit))
    except Exception as exc:  # noqa: BLE001
        return safe_error(exc, "changelog_latest")


@wave_h_bp.route("/system/changelog/unseen", methods=["GET"])
def route_changelog_unseen():
    try:
        from opencut.core import changelog_feed
        limit = safe_int(request.args.get("limit", 3), 3, min_val=1, max_val=10)
        tag = str(request.args.get("last_seen") or "").strip()
        return jsonify(changelog_feed.latest_unseen(last_seen_tag=tag, limit=limit))
    except Exception as exc:  # noqa: BLE001
        return safe_error(exc, "changelog_unseen")


@wave_h_bp.route("/system/changelog/mark-seen", methods=["POST"])
@require_csrf
def route_changelog_mark_seen():
    try:
        from opencut.core import changelog_feed
        data = request.get_json(force=True) or {}
        tag = str(data.get("tag") or "").strip()
        if not tag:
            return error_response("INVALID_INPUT", "'tag' is required", status=400)
        return jsonify(changelog_feed.mark_seen(tag))
    except ValueError as exc:
        return error_response("INVALID_INPUT", str(exc), status=400)
    except Exception as exc:  # noqa: BLE001
        return safe_error(exc, "changelog_mark_seen")


# ===========================================================================
# Tier 1.5 — Issue report bundle
# ===========================================================================

@wave_h_bp.route("/system/issue-report/bundle", methods=["GET"])
def route_issue_report_bundle_get():
    try:
        from opencut.core import issue_report
        return jsonify(issue_report.bundle())
    except Exception as exc:  # noqa: BLE001
        return safe_error(exc, "issue_report_bundle_get")


@wave_h_bp.route("/system/issue-report/bundle", methods=["POST"])
@require_csrf
def route_issue_report_bundle_post():
    try:
        from opencut.core import issue_report
        data = request.get_json(force=True, silent=True) or {}
        return jsonify(issue_report.bundle(
            title=str(data.get("title") or ""),
            description=str(data.get("description") or ""),
            log_tail_lines=safe_int(data.get("log_tail_lines", 200), 200,
                                     min_val=10, max_val=2000),
            include_crash=safe_bool(data.get("include_crash"), True),
            include_logs=safe_bool(data.get("include_logs"), True),
        ))
    except Exception as exc:  # noqa: BLE001
        return safe_error(exc, "issue_report_bundle_post")


# ===========================================================================
# Tier 1.6 — Demo bundle
# ===========================================================================

@wave_h_bp.route("/system/demo/list", methods=["GET"])
def route_demo_list():
    try:
        from opencut.core import demo_bundle
        return jsonify(demo_bundle.list_assets())
    except Exception as exc:  # noqa: BLE001
        return safe_error(exc, "demo_list")


@wave_h_bp.route("/system/demo/sample", methods=["GET"])
def route_demo_sample():
    try:
        from opencut.core import demo_bundle
        return jsonify(demo_bundle.get_sample())
    except Exception as exc:  # noqa: BLE001
        return safe_error(exc, "demo_sample")


@wave_h_bp.route("/system/demo/download", methods=["POST"])
@require_csrf
def route_demo_download():
    try:
        from opencut.core import demo_bundle
        data = request.get_json(force=True, silent=True) or {}
        return jsonify(demo_bundle.download_sample(
            version=str(data.get("version") or ""),
        ))
    except Exception as exc:  # noqa: BLE001
        return safe_error(exc, "demo_download")


# ===========================================================================
# Tier 1.7 — Gist preset sync
# ===========================================================================

@wave_h_bp.route("/settings/gist/push", methods=["POST"])
@require_csrf
def route_gist_push():
    try:
        from opencut.core import gist_sync
        data = request.get_json(force=True) or {}
        files = data.get("files")
        if not isinstance(files, dict) or not files:
            return error_response(
                "INVALID_INPUT",
                "'files' must be a non-empty object of {filename: data}",
                status=400,
            )
        return jsonify(gist_sync.push(
            files=files,
            description=str(data.get("description") or "OpenCut preset export"),
            public=safe_bool(data.get("public"), False),
            update_id=str(data.get("update_id") or ""),
        ))
    except ValueError as exc:
        return error_response("INVALID_INPUT", str(exc), status=400)
    except RuntimeError as exc:
        return error_response("GIST_FAILED", str(exc), status=502)
    except Exception as exc:  # noqa: BLE001
        return safe_error(exc, "gist_push")


@wave_h_bp.route("/settings/gist/pull", methods=["POST"])
@require_csrf
def route_gist_pull():
    try:
        from opencut.core import gist_sync
        data = request.get_json(force=True) or {}
        url_or_id = str(data.get("gist") or data.get("url") or data.get("id") or "")
        if not url_or_id:
            return error_response(
                "INVALID_INPUT",
                "'gist' (URL or ID) is required",
                status=400,
            )
        return jsonify(gist_sync.pull(url_or_id))
    except ValueError as exc:
        return error_response("INVALID_INPUT", str(exc), status=400)
    except RuntimeError as exc:
        return error_response("GIST_FAILED", str(exc), status=502)
    except Exception as exc:  # noqa: BLE001
        return safe_error(exc, "gist_pull")


@wave_h_bp.route("/settings/gist/info", methods=["GET"])
def route_gist_info():
    try:
        from opencut.core import gist_sync
        gid = str(request.args.get("gist") or request.args.get("id") or "")
        if not gid:
            return error_response(
                "INVALID_INPUT",
                "'gist' query param is required",
                status=400,
            )
        return jsonify(gist_sync.info(gid))
    except ValueError as exc:
        return error_response("INVALID_INPUT", str(exc), status=400)
    except RuntimeError as exc:
        return error_response("GIST_FAILED", str(exc), status=502)
    except Exception as exc:  # noqa: BLE001
        return safe_error(exc, "gist_info")


# ===========================================================================
# Tier 1.8 — Onboarding state
# ===========================================================================

@wave_h_bp.route("/settings/onboarding", methods=["GET"])
def route_onboarding_get():
    try:
        from opencut.core import onboarding
        return jsonify(onboarding.get_state())
    except Exception as exc:  # noqa: BLE001
        return safe_error(exc, "onboarding_get")


@wave_h_bp.route("/settings/onboarding", methods=["POST"])
@require_csrf
def route_onboarding_post():
    try:
        from opencut.core import onboarding
        data = request.get_json(force=True, silent=True) or {}
        seen = data.get("seen")
        if isinstance(seen, bool) or seen is None:
            seen_val = seen
        else:
            seen_val = safe_bool(seen, False)
        step = data.get("step")
        try:
            step_val = None if step is None else int(step)
        except (TypeError, ValueError):
            step_val = None
        return jsonify(onboarding.set_state(seen=seen_val, step=step_val))
    except Exception as exc:  # noqa: BLE001
        return safe_error(exc, "onboarding_post")


# ===========================================================================
# Tier 2 — AI model stubs (503 MISSING_DEPENDENCY)
# ===========================================================================

@wave_h_bp.route("/video/upscale/flashvsr", methods=["POST"])
@require_csrf
def route_flashvsr():
    from opencut.core import upscale_flashvsr
    if not upscale_flashvsr.check_flashvsr_available():
        return _stub_503("FlashVSR", upscale_flashvsr.INSTALL_HINT)
    return error_response(
        "NOT_IMPLEMENTED",
        "FlashVSR wiring ships in v1.26.0.",
        status=501,
        suggestion="Track ROADMAP-NEXT.md Wave H2.1 for the release.",
    )


@wave_h_bp.route("/video/upscale/flashvsr/info", methods=["GET"])
def route_flashvsr_info():
    from opencut.core import upscale_flashvsr
    return jsonify({
        "available": upscale_flashvsr.check_flashvsr_available(),
        "install_hint": upscale_flashvsr.INSTALL_HINT,
    })


@wave_h_bp.route("/video/inpaint/rose", methods=["POST"])
@require_csrf
def route_rose():
    from opencut.core import inpaint_rose
    if not inpaint_rose.check_rose_available():
        return _stub_503("ROSE inpainting", inpaint_rose.INSTALL_HINT)
    return error_response(
        "NOT_IMPLEMENTED",
        "ROSE inpainting wiring ships in a future release.",
        status=501,
    )


@wave_h_bp.route("/video/matte/sammie", methods=["POST"])
@require_csrf
def route_sammie():
    from opencut.core import matte_sammie
    if not matte_sammie.check_sammie_available():
        return _stub_503("Sammie-Roto-2", matte_sammie.INSTALL_HINT)
    return error_response(
        "NOT_IMPLEMENTED",
        "Sammie-Roto-2 wiring ships in a future release.",
        status=501,
    )


@wave_h_bp.route("/audio/tts/omnivoice", methods=["POST"])
@require_csrf
def route_omnivoice():
    from opencut.core import tts_omnivoice
    if not tts_omnivoice.check_omnivoice_available():
        return _stub_503("OmniVoice TTS", tts_omnivoice.INSTALL_HINT)
    return error_response(
        "NOT_IMPLEMENTED",
        "OmniVoice wiring ships in a future release.",
        status=501,
    )


@wave_h_bp.route("/audio/tts/omnivoice/models", methods=["GET"])
def route_omnivoice_models():
    from opencut.core import tts_omnivoice
    return jsonify(tts_omnivoice.list_models())


@wave_h_bp.route("/video/style/reezsynth", methods=["POST"])
@require_csrf
def route_reezsynth():
    from opencut.core import style_reezsynth
    if not style_reezsynth.check_reezsynth_available():
        return _stub_503("ReEzSynth", style_reezsynth.INSTALL_HINT)
    return error_response(
        "NOT_IMPLEMENTED",
        "ReEzSynth wiring ships in a future release.",
        status=501,
    )


@wave_h_bp.route("/audio/music/vidmuse", methods=["POST"])
@require_csrf
def route_vidmuse():
    from opencut.core import music_vidmuse
    if not music_vidmuse.check_vidmuse_available():
        return _stub_503("VidMuse", music_vidmuse.INSTALL_HINT)
    return error_response(
        "NOT_IMPLEMENTED",
        "VidMuse wiring ships in a future release.",
        status=501,
    )


# ===========================================================================
# Tier 3 — Strategic stubs (501 ROUTE_STUBBED)
# ===========================================================================

@wave_h_bp.route("/agent/search-footage", methods=["POST"])
@require_csrf
def route_agent_search():
    from opencut.core import video_agent
    return _stub_501("VideoAgent search", video_agent.INSTALL_HINT)


@wave_h_bp.route("/agent/storyboard", methods=["POST"])
@require_csrf
def route_agent_storyboard():
    from opencut.core import video_agent
    return _stub_501("ViMax storyboard", video_agent.INSTALL_HINT)


@wave_h_bp.route("/generate/cloud/submit", methods=["POST"])
@require_csrf
def route_cloud_submit():
    from opencut.core import gen_video_cloud
    return _stub_501("Cloud gen-video submit", gen_video_cloud.INSTALL_HINT)


@wave_h_bp.route("/generate/cloud/status/<eid>", methods=["GET"])
def route_cloud_status(eid):
    from opencut.core import gen_video_cloud
    return _stub_501("Cloud gen-video status", gen_video_cloud.INSTALL_HINT)


@wave_h_bp.route("/generate/cloud/backends", methods=["GET"])
def route_cloud_backends():
    from opencut.core import gen_video_cloud
    return jsonify(gen_video_cloud.list_backends())


@wave_h_bp.route("/lipsync/gaussian", methods=["POST"])
@require_csrf
def route_lipsync_gaussian():
    from opencut.core import lipsync_advanced
    return _stub_501("GaussianHeadTalk lip-sync", lipsync_advanced.INSTALL_HINT)


@wave_h_bp.route("/lipsync/fantasy2", methods=["POST"])
@require_csrf
def route_lipsync_fantasy2():
    from opencut.core import lipsync_advanced
    return _stub_501("FantasyTalking2 lip-sync", lipsync_advanced.INSTALL_HINT)


@wave_h_bp.route("/lipsync/advanced/backends", methods=["GET"])
def route_lipsync_backends():
    from opencut.core import lipsync_advanced
    return jsonify(lipsync_advanced.list_backends())


# ===========================================================================
# Tier 2.8 — QE reflection probe (host-side stub endpoint)
# ===========================================================================

@wave_h_bp.route("/system/qe-reflect", methods=["GET"])
def route_qe_reflect():
    """Return the most recent QE API reflection result cached by the panel.

    The panel calls ``ocQeReflect()`` in the host JSX at startup, POSTs
    the JSON result to ``/system/qe-reflect`` (future route), and this
    endpoint replays the cached dump. In v1.25.0 the cache is empty
    until the panel writes its first probe — a 200 with
    ``probed=false`` is the expected cold-start response.
    """
    try:
        from opencut.user_data import read_user_file
        state = read_user_file("qe_reflect.json", default={}) or {}
        return jsonify({
            "probed": bool(state.get("probed")),
            "methods": state.get("methods") or [],
            "probed_at": state.get("probed_at") or 0.0,
            "premiere_version": str(state.get("premiere_version") or ""),
            "note": "panel must POST a probe result before this endpoint returns data",
        })
    except Exception as exc:  # noqa: BLE001
        return safe_error(exc, "qe_reflect")


@wave_h_bp.route("/system/qe-reflect", methods=["POST"])
@require_csrf
def route_qe_reflect_write():
    """Cache a QE reflection probe result from the panel."""
    try:
        from opencut.user_data import write_user_file
        data = request.get_json(force=True) or {}
        methods = data.get("methods") or []
        if not isinstance(methods, list):
            return error_response(
                "INVALID_INPUT",
                "'methods' must be a list of strings",
                status=400,
            )
        state = {
            "probed": True,
            "methods": [str(m) for m in methods if isinstance(m, (str, bytes))][:500],
            "probed_at": float(data.get("probed_at") or 0.0),
            "premiere_version": str(data.get("premiere_version") or ""),
        }
        write_user_file("qe_reflect.json", state)
        return jsonify({"saved": True, "method_count": len(state["methods"])})
    except Exception as exc:  # noqa: BLE001
        return safe_error(exc, "qe_reflect_write")
