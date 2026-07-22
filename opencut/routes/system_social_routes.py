"""System social routes registered on the shared system blueprint."""

from .system import (
    _update_job,
    async_job,
    get_json_dict,
    jsonify,
    require_csrf,
    safe_bool,
    safe_error,
    system_bp,
    validate_filepath,
)

# ---------------------------------------------------------------------------
# Social Media Direct Posting
# ---------------------------------------------------------------------------
_VALID_SOCIAL_UPLOAD_PLATFORMS = {"youtube", "tiktok", "instagram", "twitter", "linkedin", "snapchat", "facebook", "pinterest"}


def _magic_clips_bundle_manifest_path(data):
    return str(
        data.get("magic_clips_bundle_manifest")
        or data.get("bundle_manifest_path")
        or ""
    ).strip()


def _validate_social_upload(data):
    if _magic_clips_bundle_manifest_path(data):
        if not safe_bool(data.get("dry_run", False), False):
            return "Magic Clips bundle uploads require dry_run=true"
        return None
    platform = str(data.get("platform") or "").strip().lower()
    if not platform:
        return "No platform specified"
    if platform not in _VALID_SOCIAL_UPLOAD_PLATFORMS:
        return f"Invalid platform. Use one of: {', '.join(sorted(_VALID_SOCIAL_UPLOAD_PLATFORMS))}"
    if not str(data.get("filepath") or "").strip():
        return "No file path provided"
    return None


@system_bp.route("/social/platforms", methods=["GET"])
def social_platforms():
    """List connected social media platforms."""
    try:
        from opencut.core.social_post import get_connected_platforms
        return jsonify({"platforms": get_connected_platforms()})
    except Exception as e:
        return safe_error(e, "social_platforms")


@system_bp.route("/social/auth-url", methods=["POST"])
@require_csrf
def social_auth_url():
    """Get OAuth authorization URL for a platform."""
    data = get_json_dict()
    platform = data.get("platform", "").strip().lower()

    if platform not in ("youtube", "tiktok", "instagram"):
        return jsonify({"error": "Unsupported platform. Use: youtube, tiktok, instagram"}), 400

    try:
        from opencut.core.social_post import get_oauth_url
        url = get_oauth_url(platform)
        if not url:
            return jsonify({"error": f"OAuth not configured for {platform}. Set API credentials in env vars."}), 400
        return jsonify({"auth_url": url, "platform": platform})
    except Exception as e:
        return safe_error(e, "social_auth_url")


@system_bp.route("/social/connect", methods=["POST"])
@require_csrf
def social_connect():
    """Store OAuth credentials after authorization callback."""
    data = get_json_dict()
    platform = data.get("platform", "").strip().lower()
    access_token = data.get("access_token", "").strip()

    if not platform or not access_token:
        return jsonify({"error": "Platform and access_token required"}), 400

    if platform not in ("youtube", "tiktok", "instagram"):
        return jsonify({"error": "Unsupported platform. Use: youtube, tiktok, instagram"}), 400

    try:
        from opencut.core.social_post import store_auth
        store_auth(
            platform=platform,
            access_token=access_token,
            refresh_token=data.get("refresh_token"),
            expires_in=data.get("expires_in"),
            user_id=data.get("user_id"),
            username=data.get("username"),
        )
        return jsonify({"success": True, "message": f"Connected to {platform}"})
    except Exception as e:
        return safe_error(e, "social_connect")


@system_bp.route("/social/disconnect", methods=["POST"])
@require_csrf
def social_disconnect():
    """Remove stored credentials for a platform."""
    data = get_json_dict()
    platform = data.get("platform", "").strip().lower()

    if not platform:
        return jsonify({"error": "Platform required"}), 400

    if platform not in ("youtube", "tiktok", "instagram"):
        return jsonify({"error": "Unsupported platform. Use: youtube, tiktok, instagram"}), 400

    try:
        from opencut.core.social_post import disconnect_platform
        disconnect_platform(platform)
        return jsonify({"success": True, "message": f"Disconnected from {platform}"})
    except Exception as e:
        return safe_error(e, "social_disconnect")


@system_bp.route("/social/upload", methods=["POST"])
@require_csrf
@async_job("social-upload", filepath_required=False, pre_validate=_validate_social_upload)
def social_upload(job_id, filepath, data):
    """Upload a video to a social media platform."""
    bundle_manifest_path = _magic_clips_bundle_manifest_path(data)
    if bundle_manifest_path:
        bundle_manifest_path = validate_filepath(bundle_manifest_path)
        platform_filter = str(data.get("platform") or "").strip().lower()
        privacy = str(data.get("privacy") or "private").strip() or "private"
        from opencut.core.social_post import build_magic_clips_social_upload_plan

        _update_job(job_id, progress=20, message="Reading Magic Clips bundle manifest...")
        plan = build_magic_clips_social_upload_plan(
            bundle_manifest_path,
            platform=platform_filter,
            candidate_ids=data.get("candidate_ids", data.get("candidate_id")),
            privacy=privacy,
        )
        _update_job(job_id, progress=100, message=f"Prepared {plan['upload_count']} social upload item(s).")
        plan["dry_run"] = True
        return plan

    platform = data.get("platform", "").strip().lower()

    title = data.get("title", "")[:100]
    description = data.get("description", "")[:5000]
    tags = data.get("tags", [])
    if isinstance(tags, list):
        tags = tags[:30]
    privacy = data.get("privacy", "private")

    from opencut.core.social_post import upload_to_platform

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = upload_to_platform(
        filepath=filepath,
        platform=platform,
        title=title,
        description=description,
        tags=tags,
        privacy=privacy,
        on_progress=_on_progress,
    )

    if result.success:
        return {
            "platform": result.platform,
            "video_id": result.video_id,
            "url": result.url,
            "upload_time": result.upload_time,
        }
    else:
        raise RuntimeError(f"Upload failed: {result.error}")
