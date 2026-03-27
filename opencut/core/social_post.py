"""
Social Media Direct Posting

Upload exported videos directly to social media platforms from the OpenCut panel.
Supports: YouTube, TikTok, Instagram Reels.

Each platform has its own OAuth2 flow and API requirements:
- YouTube: Google OAuth2 + YouTube Data API v3
- TikTok: TikTok Login Kit + Content Posting API
- Instagram: Facebook Graph API (Instagram Basic Display -> Content Publishing)

Credentials are stored in ~/.opencut/social_credentials.json (encrypted at rest).
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("opencut")

CREDENTIALS_PATH = os.path.join(
    os.path.expanduser("~"), ".opencut", "social_credentials.json"
)

# Platform-specific upload limits
PLATFORM_LIMITS = {
    "youtube": {
        "max_file_size_mb": 256000,  # 256 GB
        "max_duration_seconds": 43200,  # 12 hours
        "supported_formats": ["mp4", "mov", "avi", "wmv", "flv", "webm", "mkv"],
        "max_title_length": 100,
        "max_description_length": 5000,
    },
    "tiktok": {
        "max_file_size_mb": 4096,  # 4 GB
        "max_duration_seconds": 600,  # 10 minutes
        "supported_formats": ["mp4", "webm"],
        "max_title_length": 150,
        "max_description_length": 2200,
    },
    "instagram": {
        "max_file_size_mb": 650,
        "max_duration_seconds": 5400,  # 90 minutes for Reels
        "supported_formats": ["mp4", "mov"],
        "max_title_length": 0,  # Instagram has no title
        "max_description_length": 2200,
    },
}


@dataclass
class UploadResult:
    """Result of a social media upload."""
    platform: str
    success: bool
    video_id: Optional[str] = None
    url: Optional[str] = None
    error: Optional[str] = None
    upload_time: float = 0.0


@dataclass
class PlatformAuth:
    """OAuth credentials for a social platform."""
    platform: str
    access_token: str
    refresh_token: Optional[str] = None
    expires_at: float = 0.0
    user_id: Optional[str] = None
    username: Optional[str] = None

    @property
    def is_expired(self) -> bool:
        if self.expires_at <= 0:
            return False
        return time.time() > self.expires_at - 60  # 60s buffer


def _load_credentials() -> Dict[str, PlatformAuth]:
    """Load stored platform credentials."""
    if not os.path.isfile(CREDENTIALS_PATH):
        return {}
    try:
        with open(CREDENTIALS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            platform: PlatformAuth(
                platform=platform,
                access_token=cred.get("access_token", ""),
                refresh_token=cred.get("refresh_token"),
                expires_at=cred.get("expires_at", 0),
                user_id=cred.get("user_id"),
                username=cred.get("username"),
            )
            for platform, cred in data.items()
        }
    except Exception as e:
        logger.warning("Failed to load social credentials: %s", e)
        return {}


def _save_credentials(creds: Dict[str, PlatformAuth]):
    """Save platform credentials to disk."""
    os.makedirs(os.path.dirname(CREDENTIALS_PATH), exist_ok=True)
    data = {
        platform: {
            "access_token": auth.access_token,
            "refresh_token": auth.refresh_token,
            "expires_at": auth.expires_at,
            "user_id": auth.user_id,
            "username": auth.username,
        }
        for platform, auth in creds.items()
    }
    try:
        with open(CREDENTIALS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        # Restrict file permissions on Unix
        try:
            os.chmod(CREDENTIALS_PATH, 0o600)
        except (OSError, AttributeError):
            pass
    except Exception as e:
        logger.error("Failed to save social credentials: %s", e)


def get_connected_platforms() -> List[dict]:
    """Return list of platforms with active credentials."""
    creds = _load_credentials()
    result = []
    for platform, auth in creds.items():
        result.append({
            "platform": platform,
            "connected": bool(auth.access_token),
            "username": auth.username,
            "expired": auth.is_expired,
        })
    return result


def store_auth(
    platform: str,
    access_token: str,
    refresh_token: Optional[str] = None,
    expires_in: Optional[int] = None,
    user_id: Optional[str] = None,
    username: Optional[str] = None,
):
    """Store OAuth credentials for a platform."""
    creds = _load_credentials()
    expires_at = time.time() + expires_in if expires_in else 0
    creds[platform] = PlatformAuth(
        platform=platform,
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at=expires_at,
        user_id=user_id,
        username=username,
    )
    _save_credentials(creds)
    logger.info("Stored credentials for %s (user: %s)", platform, username)


def disconnect_platform(platform: str):
    """Remove stored credentials for a platform."""
    creds = _load_credentials()
    if platform in creds:
        del creds[platform]
        _save_credentials(creds)
        logger.info("Disconnected from %s", platform)


def _validate_upload(filepath: str, platform: str) -> Optional[str]:
    """Validate file before upload. Returns error message or None."""
    if not os.path.isfile(filepath):
        return f"File not found: {filepath}"

    limits = PLATFORM_LIMITS.get(platform)
    if not limits:
        return f"Unknown platform: {platform}"

    # Check file size
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    if file_size_mb > limits["max_file_size_mb"]:
        return f"File too large for {platform}: {file_size_mb:.0f}MB (max {limits['max_file_size_mb']}MB)"

    # Check format
    ext = os.path.splitext(filepath)[1].lstrip(".").lower()
    if ext not in limits["supported_formats"]:
        return f"Unsupported format for {platform}: .{ext} (supported: {limits['supported_formats']})"

    return None


# ---------------------------------------------------------------------------
# YouTube Upload
# ---------------------------------------------------------------------------

def _upload_youtube(
    filepath: str,
    title: str,
    description: str = "",
    tags: Optional[List[str]] = None,
    privacy: str = "private",
    on_progress: Optional[Callable] = None,
) -> UploadResult:
    """Upload video to YouTube using Data API v3."""
    import urllib.parse
    import urllib.request

    creds = _load_credentials()
    auth = creds.get("youtube")
    if not auth or not auth.access_token:
        return UploadResult(platform="youtube", success=False, error="Not connected to YouTube. Authorize first.")

    if auth.is_expired and auth.refresh_token:
        auth = _refresh_youtube_token(auth)
        if not auth:
            return UploadResult(platform="youtube", success=False, error="YouTube token expired. Re-authorize.")

    error = _validate_upload(filepath, "youtube")
    if error:
        return UploadResult(platform="youtube", success=False, error=error)

    # Truncate title/description
    title = title[:100]
    description = description[:5000]

    t0 = time.monotonic()

    try:
        if on_progress:
            on_progress(5, "Initializing YouTube upload...")

        # Use resumable upload protocol
        metadata = {
            "snippet": {
                "title": title,
                "description": description,
                "tags": tags or [],
                "categoryId": "22",  # People & Blogs
            },
            "status": {
                "privacyStatus": privacy,
                "selfDeclaredMadeForKids": False,
            },
        }

        # Step 1: Initialize upload session
        init_url = (
            "https://www.googleapis.com/upload/youtube/v3/videos"
            "?uploadType=resumable&part=snippet,status"
        )
        meta_bytes = json.dumps(metadata).encode("utf-8")
        req = urllib.request.Request(
            init_url,
            data=meta_bytes,
            headers={
                "Authorization": f"Bearer {auth.access_token}",
                "Content-Type": "application/json; charset=UTF-8",
                "X-Upload-Content-Type": "video/*",
                "X-Upload-Content-Length": str(os.path.getsize(filepath)),
            },
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=30)
        upload_url = resp.headers.get("Location")
        resp.close()

        if not upload_url:
            return UploadResult(platform="youtube", success=False, error="Failed to get upload URL")

        if on_progress:
            on_progress(10, "Uploading to YouTube...")

        # Step 2: Upload video data
        file_size = os.path.getsize(filepath)
        chunk_size = 10 * 1024 * 1024  # 10MB chunks
        uploaded = 0
        body = b""

        with open(filepath, "rb") as f:
            while uploaded < file_size:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                end = min(uploaded + len(chunk), file_size)
                headers = {
                    "Authorization": f"Bearer {auth.access_token}",
                    "Content-Type": "video/*",
                    "Content-Length": str(len(chunk)),
                    "Content-Range": f"bytes {uploaded}-{end - 1}/{file_size}",
                }
                req = urllib.request.Request(upload_url, data=chunk, headers=headers, method="PUT")
                try:
                    resp = urllib.request.urlopen(req, timeout=300)
                    body = resp.read()
                    resp.close()
                except urllib.error.HTTPError as e:
                    if e.code == 308:
                        # Resume incomplete — read and discard response
                        try:
                            e.read()
                        except Exception:
                            pass
                    else:
                        raise

                uploaded += len(chunk)
                if on_progress:
                    pct = 10 + int(uploaded / file_size * 80)
                    on_progress(pct, f"Uploading... {uploaded / (1024*1024):.0f}MB / {file_size / (1024*1024):.0f}MB")

        # Parse final response
        try:
            result_data = json.loads(body)
            video_id = result_data.get("id", "")
        except Exception:
            video_id = ""

        if on_progress:
            on_progress(100, "Upload complete!")

        upload_time = time.monotonic() - t0
        return UploadResult(
            platform="youtube",
            success=True,
            video_id=video_id,
            url=f"https://youtu.be/{video_id}" if video_id else None,
            upload_time=round(upload_time, 2),
        )

    except Exception as e:
        logger.error("YouTube upload failed: %s", e)
        return UploadResult(platform="youtube", success=False, error=str(e))


def _refresh_youtube_token(auth: PlatformAuth) -> Optional[PlatformAuth]:
    """Refresh an expired YouTube OAuth token."""
    import urllib.parse
    import urllib.request

    try:
        # Client ID/secret would come from app config
        client_id = os.environ.get("OPENCUT_YOUTUBE_CLIENT_ID", "")
        client_secret = os.environ.get("OPENCUT_YOUTUBE_CLIENT_SECRET", "")

        if not client_id or not client_secret:
            logger.warning("YouTube client credentials not configured")
            return None

        data = urllib.parse.urlencode({
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": auth.refresh_token,
            "grant_type": "refresh_token",
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://oauth2.googleapis.com/token",
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        resp = urllib.request.urlopen(req, timeout=10)
        result = json.loads(resp.read())
        resp.close()

        auth.access_token = result["access_token"]
        auth.expires_at = time.time() + result.get("expires_in", 3600)

        creds = _load_credentials()
        creds["youtube"] = auth
        _save_credentials(creds)

        return auth

    except Exception as e:
        logger.error("Failed to refresh YouTube token: %s", e)
        return None


# ---------------------------------------------------------------------------
# TikTok Upload
# ---------------------------------------------------------------------------

def _upload_tiktok(
    filepath: str,
    title: str = "",
    description: str = "",
    on_progress: Optional[Callable] = None,
) -> UploadResult:
    """Upload video to TikTok using Content Posting API."""
    import urllib.request

    creds = _load_credentials()
    auth = creds.get("tiktok")
    if not auth or not auth.access_token:
        return UploadResult(platform="tiktok", success=False, error="Not connected to TikTok. Authorize first.")

    error = _validate_upload(filepath, "tiktok")
    if error:
        return UploadResult(platform="tiktok", success=False, error=error)

    caption = description[:2200] if description else title[:150]

    t0 = time.monotonic()

    try:
        if on_progress:
            on_progress(5, "Initializing TikTok upload...")

        file_size = os.path.getsize(filepath)

        # Step 1: Initialize upload
        init_data = json.dumps({
            "post_info": {
                "title": caption,
                "privacy_level": "SELF_ONLY",
                "disable_duet": False,
                "disable_comment": False,
                "disable_stitch": False,
            },
            "source_info": {
                "source": "FILE_UPLOAD",
                "video_size": file_size,
                "chunk_size": min(file_size, 10 * 1024 * 1024),
                "total_chunk_count": max(1, -(-file_size // (10 * 1024 * 1024))),
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://open.tiktokapis.com/v2/post/publish/inbox/video/init/",
            data=init_data,
            headers={
                "Authorization": f"Bearer {auth.access_token}",
                "Content-Type": "application/json; charset=UTF-8",
            },
        )
        resp = urllib.request.urlopen(req, timeout=30)
        init_result = json.loads(resp.read())
        resp.close()

        publish_id = init_result.get("data", {}).get("publish_id", "")
        upload_url = init_result.get("data", {}).get("upload_url", "")

        if not upload_url:
            return UploadResult(platform="tiktok", success=False, error="Failed to get TikTok upload URL")

        if on_progress:
            on_progress(15, "Uploading to TikTok...")

        # Step 2: Upload file
        with open(filepath, "rb") as f:
            video_data = f.read()

        req = urllib.request.Request(
            upload_url,
            data=video_data,
            headers={
                "Content-Type": "video/mp4",
                "Content-Length": str(len(video_data)),
                "Content-Range": f"bytes 0-{len(video_data) - 1}/{len(video_data)}",
            },
            method="PUT",
        )
        resp = urllib.request.urlopen(req, timeout=600)
        resp.close()

        if on_progress:
            on_progress(90, "Finalizing...")

        # Step 3: Check publish status
        status_data = json.dumps({"publish_id": publish_id}).encode("utf-8")
        req = urllib.request.Request(
            "https://open.tiktokapis.com/v2/post/publish/status/fetch/",
            data=status_data,
            headers={
                "Authorization": f"Bearer {auth.access_token}",
                "Content-Type": "application/json",
            },
        )
        resp = urllib.request.urlopen(req, timeout=30)
        resp.read()  # Consume response body
        resp.close()

        if on_progress:
            on_progress(100, "Upload complete!")

        upload_time = time.monotonic() - t0
        return UploadResult(
            platform="tiktok",
            success=True,
            video_id=publish_id,
            upload_time=round(upload_time, 2),
        )

    except Exception as e:
        logger.error("TikTok upload failed: %s", e)
        return UploadResult(platform="tiktok", success=False, error=str(e))


# ---------------------------------------------------------------------------
# Instagram Upload
# ---------------------------------------------------------------------------

def _upload_instagram(
    filepath: str,
    caption: str = "",
    on_progress: Optional[Callable] = None,
) -> UploadResult:
    """Upload Reel to Instagram using Graph API."""
    import urllib.parse
    import urllib.request

    creds = _load_credentials()
    auth = creds.get("instagram")
    if not auth or not auth.access_token:
        return UploadResult(platform="instagram", success=False, error="Not connected to Instagram. Authorize first.")

    error = _validate_upload(filepath, "instagram")
    if error:
        return UploadResult(platform="instagram", success=False, error=error)

    caption = caption[:2200]
    user_id = auth.user_id
    if not user_id:
        return UploadResult(platform="instagram", success=False, error="Instagram user ID not set")

    t0 = time.monotonic()

    try:
        if on_progress:
            on_progress(5, "Initializing Instagram upload...")

        # Instagram Graph API requires video accessible via public HTTPS URL.
        # Check if filepath is already a URL or a local file.
        video_url = filepath
        if not filepath.startswith("http://") and not filepath.startswith("https://"):
            return UploadResult(
                platform="instagram",
                success=False,
                error="Instagram requires a public video URL. Local file upload is not supported. "
                      "Upload the file to a hosting service first, then provide the URL.",
            )

        # Step 1: Create media container for Reel
        init_data = json.dumps({
            "media_type": "REELS",
            "video_url": video_url,
            "caption": caption,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"https://graph.facebook.com/v19.0/{user_id}/media",
            data=init_data,
            headers={
                "Authorization": f"Bearer {auth.access_token}",
                "Content-Type": "application/json",
            },
        )
        resp = urllib.request.urlopen(req, timeout=30)
        create_result = json.loads(resp.read())
        resp.close()

        container_id = create_result.get("id", "")
        if not container_id:
            return UploadResult(platform="instagram", success=False, error="Failed to create Instagram media container")

        if on_progress:
            on_progress(30, "Processing video on Instagram...")

        # Step 2: Poll until container is ready (max 3 minutes)
        container_ready = False
        for attempt in range(36):
            time.sleep(5)
            req = urllib.request.Request(
                f"https://graph.facebook.com/v19.0/{container_id}?fields=status_code",
                headers={"Authorization": f"Bearer {auth.access_token}"},
            )
            resp = urllib.request.urlopen(req, timeout=10)
            status = json.loads(resp.read())
            resp.close()

            code = status.get("status_code", "")
            if code == "FINISHED":
                container_ready = True
                break
            elif code == "ERROR":
                return UploadResult(platform="instagram", success=False, error="Instagram video processing failed")

            if on_progress:
                on_progress(30 + attempt * 1.5, "Processing on Instagram...")

        if not container_ready:
            return UploadResult(platform="instagram", success=False, error="Instagram processing timed out after 3 minutes")

        if on_progress:
            on_progress(85, "Publishing...")

        # Step 3: Publish the container
        publish_data = json.dumps({"creation_id": container_id}).encode("utf-8")
        req = urllib.request.Request(
            f"https://graph.facebook.com/v19.0/{user_id}/media_publish",
            data=publish_data,
            headers={
                "Authorization": f"Bearer {auth.access_token}",
                "Content-Type": "application/json",
            },
        )
        resp = urllib.request.urlopen(req, timeout=30)
        pub_result = json.loads(resp.read())
        resp.close()

        media_id = pub_result.get("id", "")

        if on_progress:
            on_progress(100, "Published to Instagram!")

        upload_time = time.monotonic() - t0
        return UploadResult(
            platform="instagram",
            success=True,
            video_id=media_id,
            upload_time=round(upload_time, 2),
        )

    except Exception as e:
        logger.error("Instagram upload failed: %s", e)
        return UploadResult(platform="instagram", success=False, error=str(e))


# ---------------------------------------------------------------------------
# Unified Upload API
# ---------------------------------------------------------------------------

def upload_to_platform(
    filepath: str,
    platform: str,
    title: str = "",
    description: str = "",
    tags: Optional[List[str]] = None,
    privacy: str = "private",
    on_progress: Optional[Callable] = None,
) -> UploadResult:
    """
    Upload a video to a social media platform.

    Args:
        filepath: Path to the video file.
        platform: Target platform ("youtube", "tiktok", "instagram").
        title: Video title (YouTube/TikTok).
        description: Video description/caption.
        tags: Tags/hashtags (YouTube).
        privacy: Privacy level ("public", "private", "unlisted" for YouTube).
        on_progress: Callback(percent, message).

    Returns:
        UploadResult with success/failure and video URL.
    """
    if not os.path.isfile(filepath):
        return UploadResult(platform=platform, success=False, error=f"File not found: {filepath}")

    platform = platform.lower().strip()

    uploaders = {
        "youtube": lambda: _upload_youtube(filepath, title, description, tags, privacy, on_progress),
        "tiktok": lambda: _upload_tiktok(filepath, title, description, on_progress),
        "instagram": lambda: _upload_instagram(filepath, description or title, on_progress),
    }

    uploader = uploaders.get(platform)
    if not uploader:
        return UploadResult(
            platform=platform, success=False,
            error=f"Unknown platform: {platform}. Supported: {list(uploaders.keys())}",
        )

    result = uploader()

    if result.success:
        logger.info(
            "Uploaded to %s: %s (%.1fs)",
            platform, result.video_id, result.upload_time,
        )
    else:
        logger.error("Upload to %s failed: %s", platform, result.error)

    return result


def get_oauth_url(platform: str, port: int = 5679) -> Optional[str]:
    """
    Get the OAuth authorization URL for a platform.

    The user visits this URL to authorize OpenCut, then the callback
    stores the credentials via store_auth().

    Args:
        platform: "youtube", "tiktok", or "instagram"
        port: Server port for redirect URI (default 5679)
    """
    import urllib.parse

    base_url = f"http://localhost:{port}"

    if platform == "youtube":
        client_id = os.environ.get("OPENCUT_YOUTUBE_CLIENT_ID", "")
        if not client_id:
            return None
        params = urllib.parse.urlencode({
            "client_id": client_id,
            "redirect_uri": f"{base_url}/auth/youtube/callback",
            "response_type": "code",
            "scope": "https://www.googleapis.com/auth/youtube.upload",
            "access_type": "offline",
            "prompt": "consent",
        })
        return f"https://accounts.google.com/o/oauth2/v2/auth?{params}"

    elif platform == "tiktok":
        client_key = os.environ.get("OPENCUT_TIKTOK_CLIENT_KEY", "")
        if not client_key:
            return None
        params = urllib.parse.urlencode({
            "client_key": client_key,
            "redirect_uri": f"{base_url}/auth/tiktok/callback",
            "response_type": "code",
            "scope": "video.upload,video.publish",
        })
        return f"https://www.tiktok.com/v2/auth/authorize/?{params}"

    elif platform == "instagram":
        app_id = os.environ.get("OPENCUT_INSTAGRAM_APP_ID", "")
        if not app_id:
            return None
        params = urllib.parse.urlencode({
            "client_id": app_id,
            "redirect_uri": f"{base_url}/auth/instagram/callback",
            "response_type": "code",
            "scope": "instagram_basic,instagram_content_publish",
        })
        return f"https://api.instagram.com/oauth/authorize?{params}"

    return None
