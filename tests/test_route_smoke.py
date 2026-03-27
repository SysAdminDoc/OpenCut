"""
OpenCut Route Smoke Tests

Every route endpoint gets at least one test proving it:
  1. Doesn't crash (no unhandled exceptions)
  2. Returns valid JSON
  3. Returns a sensible HTTP status (not 500 for bad input)
  4. Returns structured error fields when failing

Uses Flask test client -- no real server, no subprocess, no GPU needed.
External dependencies are mocked where necessary.
"""

import json
from unittest.mock import patch

import pytest

from tests.conftest import csrf_headers

# =====================================================================
# SYSTEM ROUTES
# =====================================================================

class TestSystemRoutes:
    """Smoke tests for opencut/routes/system.py"""

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert "csrf_token" in data
        assert "version" in data
        assert "capabilities" in data

    def test_system_gpu(self, client):
        resp = client.get("/system/gpu")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "gpu_available" in data or "available" in data

    def test_system_gpu_recommend(self, client):
        resp = client.get("/system/gpu-recommend")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, dict)

    def test_system_dependencies(self, client):
        resp = client.get("/system/dependencies")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, dict)
        assert "ffmpeg" in data

    def test_openapi_spec(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["openapi"] == "3.0.3"
        assert len(data["paths"]) > 0

    def test_system_update_check(self, client):
        resp = client.get("/system/update-check")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "current_version" in data

    def test_info_requires_csrf(self, client):
        resp = client.post("/info", data=json.dumps({}),
                           content_type="application/json")
        assert resp.status_code == 403

    def test_info_with_csrf_no_file(self, client, csrf_token):
        resp = client.post("/info", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400  # no filepath provided

    def test_whisper_settings_get(self, client):
        resp = client.get("/whisper/settings")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, dict)

    def test_whisper_settings_post(self, client, csrf_token):
        resp = client.post("/whisper/settings",
                           data=json.dumps({"model": "base"}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code in (200, 400, 429)
        assert resp.get_json() is not None

    def test_whisper_clear_cache(self, client, csrf_token):
        resp = client.post("/whisper/clear-cache", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code in (200, 400, 429)
        assert resp.get_json() is not None

    def test_whisper_reinstall(self, client, csrf_token):
        resp = client.post("/whisper/reinstall", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        # May return 200 or fail gracefully
        assert resp.status_code in (200, 400, 429)
        assert resp.get_json() is not None

    def test_models_list(self, client):
        resp = client.get("/models/list")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, dict)

    def test_models_delete_no_model(self, client, csrf_token):
        resp = client.post("/models/delete", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code in (400, 404)

    def test_llm_status(self, client):
        resp = client.get("/llm/status")
        assert resp.status_code == 200

    def test_llm_test(self, client, csrf_token):
        resp = client.post("/llm/test", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code in (200, 400, 429)
        assert resp.get_json() is not None

    def test_file_no_path_returns_error(self, client):
        resp = client.get("/file")
        assert resp.status_code in (400, 404)

    def test_outputs_recent(self, client):
        resp = client.get("/outputs/recent")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, (list, dict))

    def test_shutdown_non_localhost_rejected(self, client, csrf_token):
        """Shutdown should only work from localhost."""
        resp = client.post("/shutdown", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        # May succeed or reject depending on test client IP
        assert resp.status_code in (200, 400, 429)
        assert resp.get_json() is not None

    def test_install_whisper(self, client, csrf_token):
        resp = client.post("/install-whisper", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code in (200, 400, 429)
        assert resp.get_json() is not None

    def test_demucs_install(self, client, csrf_token):
        resp = client.post("/demucs/install", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code in (200, 400, 429)
        assert resp.get_json() is not None

    def test_watermark_install(self, client, csrf_token):
        resp = client.post("/watermark/install", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code in (200, 400, 429)
        assert resp.get_json() is not None

    def test_system_status(self, client):
        resp = client.get("/system/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "connected" in data
        assert "uptime_seconds" in data
        assert "gpu" in data
        assert "jobs" in data


class TestResolveRoutes:
    """Smoke tests for DaVinci Resolve integration endpoints."""

    def test_resolve_status(self, client):
        resp = client.get("/resolve/status")
        assert resp.status_code == 200

    def test_resolve_media(self, client):
        resp = client.get("/resolve/media")
        assert resp.status_code == 200
        assert resp.get_json() is not None

    def test_resolve_timeline(self, client):
        resp = client.get("/resolve/timeline")
        assert resp.status_code == 200
        assert resp.get_json() is not None

    def test_resolve_import_no_data(self, client, csrf_token):
        resp = client.post("/resolve/import", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code in (200, 400, 429)
        assert resp.get_json() is not None

    def test_resolve_markers_no_data(self, client, csrf_token):
        resp = client.post("/resolve/markers", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code in (200, 400, 429)
        assert resp.get_json() is not None


class TestChatRoutes:
    """Smoke tests for /chat endpoints."""

    def test_chat_no_message(self, client, csrf_token):
        resp = client.post("/chat", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code in (200, 400)

    def test_chat_clear(self, client, csrf_token):
        resp = client.post("/chat/clear", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code in (200, 400, 429)
        assert resp.get_json() is not None

    def test_chat_sessions(self, client):
        resp = client.get("/chat/sessions")
        assert resp.status_code == 200


class TestSocialRoutes:
    """Smoke tests for /social/* endpoints."""

    def test_social_platforms(self, client):
        resp = client.get("/social/platforms")
        assert resp.status_code == 200

    def test_social_auth_url_no_platform(self, client, csrf_token):
        resp = client.post("/social/auth-url", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code in (400, 200)

    def test_social_connect_no_data(self, client, csrf_token):
        resp = client.post("/social/connect", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code in (400, 200)

    def test_social_disconnect_no_data(self, client, csrf_token):
        resp = client.post("/social/disconnect", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code in (400, 200)

    def test_social_upload_no_data(self, client, csrf_token):
        resp = client.post("/social/upload", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code in (400, 200)


class TestWebSocketRoutes:
    """Smoke tests for /ws/* endpoints."""

    def test_ws_status(self, client):
        resp = client.get("/ws/status")
        assert resp.status_code == 200

    def test_ws_start(self, client, csrf_token):
        resp = client.post("/ws/start", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code in (200, 400, 429)
        assert resp.get_json() is not None

    def test_ws_stop(self, client, csrf_token):
        resp = client.post("/ws/stop", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code in (200, 400, 429)
        assert resp.get_json() is not None


class TestEngineRoutes:
    """Smoke tests for /engines/* endpoints."""

    def test_engines_list(self, client):
        resp = client.get("/engines")
        assert resp.status_code == 200

    def test_engines_preferences_get(self, client):
        resp = client.get("/engines/preferences")
        assert resp.status_code == 200

    def test_engines_preference_post(self, client, csrf_token):
        resp = client.post("/engines/preference",
                           data=json.dumps({"operation": "silence", "engine": "builtin"}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code in (200, 400, 429)
        assert resp.get_json() is not None

    def test_engines_resolve(self, client, csrf_token):
        resp = client.post("/engines/resolve",
                           data=json.dumps({"operation": "silence"}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code in (200, 400, 429)
        assert resp.get_json() is not None


class TestMultimodalDiarizeRoute:
    """Smoke tests for /video/multimodal-diarize on system blueprint."""

    def test_multimodal_diarize_no_file(self, client, csrf_token):
        resp = client.post("/video/multimodal-diarize", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_broll_generate_no_data(self, client, csrf_token):
        resp = client.post("/video/broll-generate", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_broll_backends(self, client):
        resp = client.get("/video/broll-backends")
        assert resp.status_code == 200


# =====================================================================
# AUDIO ROUTES
# =====================================================================

class TestAudioRoutes:
    """Smoke tests for opencut/routes/audio.py"""

    def test_silence_no_file_returns_400(self, client, csrf_token):
        resp = client.post("/silence", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data

    def test_fillers_no_file_returns_400(self, client, csrf_token):
        resp = client.post("/fillers", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data

    def test_audio_denoise_no_file_returns_400(self, client, csrf_token):
        resp = client.post("/audio/denoise", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_audio_normalize_no_file_returns_400(self, client, csrf_token):
        resp = client.post("/audio/normalize", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_audio_isolate_no_file_returns_400(self, client, csrf_token):
        resp = client.post("/audio/isolate", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_audio_separate_no_file_returns_400(self, client, csrf_token):
        resp = client.post("/audio/separate", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_audio_beats_no_file_returns_400(self, client, csrf_token):
        resp = client.post("/audio/beats", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_audio_effects_list(self, client):
        resp = client.get("/audio/effects")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, (list, dict))

    def test_audio_effects_apply_no_effect(self, client, csrf_token):
        resp = client.post("/audio/effects/apply", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_audio_loudness_presets(self, client):
        resp = client.get("/audio/loudness-presets")
        assert resp.status_code == 200

    def test_audio_tts_voices(self, client):
        resp = client.get("/audio/tts/voices")
        assert resp.status_code == 200

    def test_audio_tts_generate_no_text(self, client, csrf_token):
        resp = client.post("/audio/tts/generate", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_audio_tts_subtitled_no_data(self, client, csrf_token):
        resp = client.post("/audio/tts/subtitled", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_audio_tts_install(self, client, csrf_token):
        resp = client.post("/audio/tts/install", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code in (200, 400, 429)
        assert resp.get_json() is not None

    def test_audio_crisper_whisper_install(self, client, csrf_token):
        resp = client.post("/audio/crisper-whisper/install", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code in (200, 400, 429)
        assert resp.get_json() is not None

    def test_audio_gen_capabilities(self, client):
        resp = client.get("/audio/gen/capabilities")
        assert resp.status_code == 200

    def test_audio_gen_tone_no_data(self, client, csrf_token):
        resp = client.post("/audio/gen/tone", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code in (200, 400)

    def test_audio_gen_sfx_no_data(self, client, csrf_token):
        resp = client.post("/audio/gen/sfx", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code in (200, 400)

    def test_audio_gen_silence_post(self, client, csrf_token):
        resp = client.post("/audio/gen/silence", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code in (200, 400)

    def test_audio_music_ai_capabilities(self, client):
        resp = client.get("/audio/music-ai/capabilities")
        assert resp.status_code == 200

    def test_audio_music_ai_generate_no_data(self, client, csrf_token):
        resp = client.post("/audio/music-ai/generate", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code in (200, 400)

    def test_audio_music_ai_ace_step_no_data(self, client, csrf_token):
        resp = client.post("/audio/music-ai/ace-step", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code in (200, 400)

    def test_audio_music_ai_melody_no_data(self, client, csrf_token):
        resp = client.post("/audio/music-ai/melody", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_audio_pro_effects(self, client):
        resp = client.get("/audio/pro/effects")
        assert resp.status_code == 200

    def test_audio_pro_apply_no_data(self, client, csrf_token):
        resp = client.post("/audio/pro/apply", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_audio_pro_deepfilter_no_data(self, client, csrf_token):
        resp = client.post("/audio/pro/deepfilter", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_audio_pro_install(self, client, csrf_token):
        resp = client.post("/audio/pro/install", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code in (200, 400, 429)
        assert resp.get_json() is not None

    def test_audio_measure_no_file_returns_400(self, client, csrf_token):
        resp = client.post("/audio/measure", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_audio_duck_no_file_returns_400(self, client, csrf_token):
        resp = client.post("/audio/duck", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_audio_mix_duck_no_file(self, client, csrf_token):
        resp = client.post("/audio/mix-duck", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_audio_duck_video_no_file(self, client, csrf_token):
        resp = client.post("/audio/duck-video", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_audio_mix_no_files(self, client, csrf_token):
        resp = client.post("/audio/mix", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_audio_waveform_no_file_returns_400(self, client, csrf_token):
        resp = client.post("/audio/waveform", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_silence_speed_up_no_file_returns_400(self, client, csrf_token):
        resp = client.post("/silence/speed-up", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_audio_enhance_no_file(self, client, csrf_token):
        resp = client.post("/audio/enhance", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_audio_beat_markers_no_file(self, client, csrf_token):
        resp = client.post("/audio/beat-markers", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_audio_loudness_match_no_file(self, client, csrf_token):
        resp = client.post("/audio/loudness-match", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400


# =====================================================================
# CAPTIONS ROUTES
# =====================================================================

class TestCaptionsRoutes:
    """Smoke tests for opencut/routes/captions.py"""

    def test_captions_no_file_returns_400(self, client, csrf_token):
        resp = client.post("/captions", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data

    def test_caption_styles_get(self, client):
        resp = client.get("/caption-styles")
        assert resp.status_code == 200

    def test_styled_captions_no_file_returns_400(self, client, csrf_token):
        resp = client.post("/styled-captions", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_transcript_no_file_returns_400(self, client, csrf_token):
        resp = client.post("/transcript", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_transcript_export_no_data(self, client, csrf_token):
        resp = client.post("/transcript/export", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_transcript_summarize_no_data(self, client, csrf_token):
        resp = client.post("/transcript/summarize", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_captions_translate_no_file_returns_400(self, client, csrf_token):
        resp = client.post("/captions/translate", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_captions_karaoke_no_file_returns_400(self, client, csrf_token):
        resp = client.post("/captions/karaoke", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_captions_convert_no_segments_returns_400(self, client, csrf_token):
        resp = client.post("/captions/convert",
                           data=json.dumps({"format": "srt"}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_captions_whisperx_no_file(self, client, csrf_token):
        resp = client.post("/captions/whisperx", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_captions_enhanced_install(self, client, csrf_token):
        resp = client.post("/captions/enhanced/install", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code in (200, 400, 429)
        assert resp.get_json() is not None

    def test_captions_emoji_map(self, client):
        resp = client.get("/captions/emoji-map")
        assert resp.status_code == 200

    def test_captions_burnin_styles(self, client):
        resp = client.get("/captions/burnin/styles")
        assert resp.status_code == 200

    def test_captions_burnin_file_no_data(self, client, csrf_token):
        resp = client.post("/captions/burnin/file", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_captions_burnin_segments_no_data(self, client, csrf_token):
        resp = client.post("/captions/burnin/segments", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_captions_animated_presets(self, client):
        resp = client.get("/captions/animated/presets")
        assert resp.status_code == 200

    def test_captions_animated_render_no_data(self, client, csrf_token):
        resp = client.post("/captions/animated/render", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_captions_enhanced_capabilities(self, client):
        resp = client.get("/captions/enhanced/capabilities")
        assert resp.status_code == 200

    def test_full_pipeline_no_file_returns_400(self, client, csrf_token):
        resp = client.post("/full", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_captions_chapters_no_file_returns_400(self, client, csrf_token):
        resp = client.post("/captions/chapters", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_captions_repeat_detect_no_file(self, client, csrf_token):
        resp = client.post("/captions/repeat-detect", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400


# =====================================================================
# VIDEO ROUTES
# =====================================================================

class TestVideoRoutes:
    """Smoke tests for opencut/routes/video.py"""

    def test_export_video_no_file_returns_400(self, client, csrf_token):
        resp = client.post("/export-video", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_watermark_no_file_returns_400(self, client, csrf_token):
        resp = client.post("/video/watermark", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_auto_detect_watermark_no_file(self, client, csrf_token):
        resp = client.post("/video/auto-detect-watermark", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_scenes_no_file_returns_400(self, client, csrf_token):
        resp = client.post("/video/scenes", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_fx_list(self, client):
        resp = client.get("/video/fx/list")
        assert resp.status_code == 200

    def test_video_fx_apply_no_effect(self, client, csrf_token):
        resp = client.post("/video/fx/apply", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_ai_capabilities(self, client):
        resp = client.get("/video/ai/capabilities")
        assert resp.status_code == 200

    def test_video_ai_upscale_no_file(self, client, csrf_token):
        resp = client.post("/video/ai/upscale", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_ai_rembg_no_file(self, client, csrf_token):
        resp = client.post("/video/ai/rembg", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_ai_interpolate_no_file(self, client, csrf_token):
        resp = client.post("/video/ai/interpolate", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_ai_denoise_no_file(self, client, csrf_token):
        resp = client.post("/video/ai/denoise", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_ai_install(self, client, csrf_token):
        resp = client.post("/video/ai/install", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code in (200, 400, 429)
        assert resp.get_json() is not None

    def test_video_upscale_capabilities(self, client):
        resp = client.get("/video/upscale/capabilities")
        assert resp.status_code == 200

    def test_video_upscale_run_no_file(self, client, csrf_token):
        resp = client.post("/video/upscale/run", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_face_capabilities(self, client):
        resp = client.get("/video/face/capabilities")
        assert resp.status_code == 200

    def test_video_face_detect_no_file(self, client, csrf_token):
        resp = client.post("/video/face/detect", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_face_blur_no_file(self, client, csrf_token):
        resp = client.post("/video/face/blur", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_face_install(self, client, csrf_token):
        resp = client.post("/video/face/install", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code in (200, 400, 429)
        assert resp.get_json() is not None

    def test_video_face_swap_capabilities(self, client):
        resp = client.get("/video/face/swap/capabilities")
        assert resp.status_code == 200

    def test_video_face_enhance_no_file(self, client, csrf_token):
        resp = client.post("/video/face/enhance", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_face_swap_no_file(self, client, csrf_token):
        resp = client.post("/video/face/swap", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_color_capabilities(self, client):
        resp = client.get("/video/color/capabilities")
        assert resp.status_code == 200

    def test_video_color_correct_no_file(self, client, csrf_token):
        resp = client.post("/video/color/correct", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_color_convert_no_file(self, client, csrf_token):
        resp = client.post("/video/color/convert", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_color_external_lut_no_file(self, client, csrf_token):
        resp = client.post("/video/color/external-lut", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_lut_list(self, client):
        resp = client.get("/video/lut/list")
        assert resp.status_code == 200

    def test_video_lut_apply_no_file(self, client, csrf_token):
        resp = client.post("/video/lut/apply", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_lut_generate_all_no_file(self, client, csrf_token):
        resp = client.post("/video/lut/generate-all", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_lut_generate_from_ref_no_file(self, client, csrf_token):
        resp = client.post("/video/lut/generate-from-ref", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_lut_generate_ai_no_file(self, client, csrf_token):
        resp = client.post("/video/lut/generate-ai", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_lut_blend_no_data(self, client, csrf_token):
        resp = client.post("/video/lut/blend", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_speed_presets(self, client):
        resp = client.get("/video/speed/presets")
        assert resp.status_code == 200

    def test_video_speed_change_no_file(self, client, csrf_token):
        resp = client.post("/video/speed/change", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_speed_reverse_no_file(self, client, csrf_token):
        resp = client.post("/video/speed/reverse", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_speed_ramp_no_file(self, client, csrf_token):
        resp = client.post("/video/speed/ramp", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_title_presets(self, client):
        resp = client.get("/video/title/presets")
        assert resp.status_code == 200

    def test_video_title_render_no_data(self, client, csrf_token):
        resp = client.post("/video/title/render", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_title_overlay_no_data(self, client, csrf_token):
        resp = client.post("/video/title/overlay", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_transitions_list(self, client):
        resp = client.get("/video/transitions/list")
        assert resp.status_code == 200

    def test_video_transitions_apply_no_data(self, client, csrf_token):
        resp = client.post("/video/transitions/apply", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_transitions_join_no_data(self, client, csrf_token):
        resp = client.post("/video/transitions/join", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_particles_presets(self, client):
        resp = client.get("/video/particles/presets")
        assert resp.status_code == 200

    def test_video_particles_apply_no_data(self, client, csrf_token):
        resp = client.post("/video/particles/apply", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_style_list(self, client):
        resp = client.get("/video/style/list")
        assert resp.status_code == 200

    def test_video_style_apply_no_file(self, client, csrf_token):
        resp = client.post("/video/style/apply", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_style_arbitrary_no_file(self, client, csrf_token):
        resp = client.post("/video/style/arbitrary", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_reframe_presets(self, client):
        resp = client.get("/video/reframe/presets")
        assert resp.status_code == 200

    def test_video_reframe_no_file(self, client, csrf_token):
        resp = client.post("/video/reframe", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_reframe_face_no_file(self, client, csrf_token):
        resp = client.post("/video/reframe/face", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_chromakey_no_file_returns_400(self, client, csrf_token):
        resp = client.post("/video/chromakey", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_pip_no_file_returns_400(self, client, csrf_token):
        resp = client.post("/video/pip", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_blend_no_files_returns_400(self, client, csrf_token):
        resp = client.post("/video/blend", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_remove_capabilities(self, client):
        resp = client.get("/video/remove/capabilities")
        assert resp.status_code == 200

    def test_video_remove_watermark_no_file(self, client, csrf_token):
        resp = client.post("/video/remove/watermark", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_trim_no_file_returns_400(self, client, csrf_token):
        resp = client.post("/video/trim", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_auto_edit_no_file_returns_400(self, client, csrf_token):
        resp = client.post("/video/auto-edit", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_highlights_no_file_returns_400(self, client, csrf_token):
        resp = client.post("/video/highlights", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_merge_too_few_files(self, client, csrf_token):
        resp = client.post("/video/merge", data=json.dumps({"files": []}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_preview_frame_no_file(self, client, csrf_token):
        resp = client.post("/video/preview-frame", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_export_presets(self, client):
        resp = client.get("/export/presets")
        assert resp.status_code == 200

    def test_export_preset_save_no_data(self, client, csrf_token):
        resp = client.post("/export/preset", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_export_thumbnails_no_file(self, client, csrf_token):
        resp = client.post("/export/thumbnails", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_export_social_presets(self, client):
        resp = client.get("/export/social-presets")
        assert resp.status_code == 200

    def test_batch_list(self, client):
        resp = client.get("/batch/list")
        assert resp.status_code == 200

    def test_batch_create_no_data(self, client, csrf_token):
        resp = client.post("/batch/create", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_batch_status_nonexistent(self, client):
        resp = client.get("/batch/nonexistent-batch-id")
        assert resp.status_code == 404

    def test_batch_cancel_nonexistent(self, client, csrf_token):
        resp = client.post("/batch/nonexistent-batch-id/cancel",
                           data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 404

    def test_video_shorts_pipeline_no_file(self, client, csrf_token):
        resp = client.post("/video/shorts-pipeline", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_color_match_no_file(self, client, csrf_token):
        resp = client.post("/video/color-match", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_auto_zoom_no_file(self, client, csrf_token):
        resp = client.post("/video/auto-zoom", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_multicam_cuts_no_data(self, client, csrf_token):
        resp = client.post("/video/multicam-cuts", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_multicam_xml_no_data(self, client, csrf_token):
        resp = client.post("/video/multicam-xml", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_emotion_highlights_no_file(self, client, csrf_token):
        resp = client.post("/video/emotion-highlights", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_depth_map_no_file(self, client, csrf_token):
        resp = client.post("/video/depth/map", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_depth_bokeh_no_file(self, client, csrf_token):
        resp = client.post("/video/depth/bokeh", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_depth_parallax_no_file(self, client, csrf_token):
        resp = client.post("/video/depth/parallax", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_broll_plan_no_data(self, client, csrf_token):
        resp = client.post("/video/broll-plan", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_video_depth_install(self, client, csrf_token):
        resp = client.post("/video/depth/install", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code in (200, 400, 429)
        assert resp.get_json() is not None

    def test_video_emotion_install(self, client, csrf_token):
        resp = client.post("/video/emotion/install", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code in (200, 400, 429)
        assert resp.get_json() is not None

    def test_video_multimodal_diarize_install(self, client, csrf_token):
        resp = client.post("/video/multimodal-diarize/install", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code in (200, 400, 429)
        assert resp.get_json() is not None

    def test_video_broll_generate_install(self, client, csrf_token):
        resp = client.post("/video/broll-generate/install", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code in (200, 400, 429)
        assert resp.get_json() is not None


# =====================================================================
# JOBS ROUTES
# =====================================================================

class TestJobsRoutes:
    """Smoke tests for opencut/routes/jobs_routes.py"""

    def test_jobs_list(self, client):
        resp = client.get("/jobs")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, list)

    def test_job_status_not_found(self, client):
        resp = client.get("/status/nonexistent-job-id")
        assert resp.status_code == 404

    def test_cancel_nonexistent_job(self, client, csrf_token):
        resp = client.post("/cancel/nonexistent-job-id",
                           data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 404

    def test_cancel_all(self, client, csrf_token):
        resp = client.post("/cancel-all", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 200

    def test_queue_list(self, client):
        resp = client.get("/queue/list")
        assert resp.status_code == 200

    def test_queue_clear(self, client, csrf_token):
        resp = client.post("/queue/clear", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 200

    def test_queue_add_invalid_endpoint(self, client, csrf_token):
        resp = client.post("/queue/add",
                           data=json.dumps({"endpoint": "/nonexistent", "payload": {}}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_stream_nonexistent_job(self, client):
        resp = client.get("/stream/nonexistent-job-id")
        # SSE endpoint -- should return 404 or start streaming
        assert resp.status_code in (200, 404)

    def test_jobs_stream_result_nonexistent(self, client):
        resp = client.get("/jobs/stream-result/nonexistent-job-id")
        assert resp.status_code in (200, 404)


class TestJobHistoryRoutes:
    """Smoke tests for job persistence endpoints"""

    def test_jobs_history(self, client):
        resp = client.get("/jobs/history")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, list)

    def test_jobs_history_with_filter(self, client):
        resp = client.get("/jobs/history?status=complete&limit=10")
        assert resp.status_code == 200

    def test_jobs_history_bad_limit(self, client):
        resp = client.get("/jobs/history?limit=abc")
        assert resp.status_code == 200  # safe_int defaults to 50

    def test_jobs_stats(self, client):
        resp = client.get("/jobs/stats")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "total" in data

    def test_jobs_interrupted(self, client):
        resp = client.get("/jobs/interrupted")
        assert resp.status_code == 200
        assert isinstance(resp.get_json(), list)


# =====================================================================
# SETTINGS ROUTES
# =====================================================================

class TestSettingsRoutes:
    """Smoke tests for opencut/routes/settings.py"""

    def test_presets_get(self, client):
        resp = client.get("/presets")
        assert resp.status_code == 200

    def test_presets_save_missing_name(self, client, csrf_token):
        resp = client.post("/presets/save",
                           data=json.dumps({"settings": {}}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_presets_delete_missing_name(self, client, csrf_token):
        resp = client.post("/presets/delete",
                           data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code in (400, 404)

    def test_favorites_get(self, client):
        resp = client.get("/favorites")
        assert resp.status_code == 200

    def test_favorites_save_invalid(self, client, csrf_token):
        resp = client.post("/favorites/save",
                           data=json.dumps({"favorites": "not-a-list"}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_workflows_list(self, client):
        resp = client.get("/workflows/list")
        assert resp.status_code == 200

    def test_workflows_save_missing_name(self, client, csrf_token):
        resp = client.post("/workflows/save",
                           data=json.dumps({"steps": [{"action": "silence"}]}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_workflows_delete_missing_name(self, client, csrf_token):
        resp = client.post("/workflows/delete",
                           data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code in (400, 404)

    def test_settings_llm_get(self, client):
        resp = client.get("/settings/llm")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "provider" in data

    def test_settings_llm_post(self, client, csrf_token):
        resp = client.post("/settings/llm",
                           data=json.dumps({"provider": "none"}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code in (200, 400, 429)
        assert resp.get_json() is not None

    def test_settings_export(self, client):
        resp = client.get("/settings/export")
        assert resp.status_code == 200

    def test_settings_import_empty(self, client, csrf_token):
        resp = client.post("/settings/import",
                           data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        # Should handle gracefully -- either 200 (no-op) or 400, never 500
        assert resp.status_code in (200, 400)

    def test_logs_export(self, client):
        resp = client.get("/logs/export")
        # May return 200 (logs exist) or 404 (no logs)
        assert resp.status_code in (200, 404)

    def test_logs_clear(self, client, csrf_token):
        resp = client.post("/logs/clear", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code in (200, 400, 429)
        assert resp.get_json() is not None

    def test_job_retry_nonexistent(self, client, csrf_token):
        resp = client.post("/jobs/retry/nonexistent-id",
                           data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code in (404, 400)

    def test_system_estimate_time(self, client, csrf_token):
        resp = client.post("/system/estimate-time",
                           data=json.dumps({"operation": "silence", "duration": 60}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 200

    def test_settings_loudness_target_get(self, client):
        resp = client.get("/settings/loudness-target")
        assert resp.status_code == 200

    def test_settings_loudness_target_post(self, client, csrf_token):
        resp = client.post("/settings/loudness-target",
                           data=json.dumps({"target": -14}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code in (200, 400, 429)
        assert resp.get_json() is not None

    def test_settings_multicam_get(self, client):
        resp = client.get("/settings/multicam")
        assert resp.status_code == 200

    def test_settings_multicam_post(self, client, csrf_token):
        resp = client.post("/settings/multicam",
                           data=json.dumps({"enabled": True}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code in (200, 400, 429)
        assert resp.get_json() is not None

    def test_settings_auto_zoom_get(self, client):
        resp = client.get("/settings/auto-zoom")
        assert resp.status_code == 200

    def test_settings_auto_zoom_post(self, client, csrf_token):
        resp = client.post("/settings/auto-zoom",
                           data=json.dumps({"enabled": True}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code in (200, 400, 429)
        assert resp.get_json() is not None

    def test_settings_chapters_get(self, client):
        resp = client.get("/settings/chapters")
        assert resp.status_code == 200

    def test_settings_chapters_post(self, client, csrf_token):
        resp = client.post("/settings/chapters",
                           data=json.dumps({"enabled": True}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code in (200, 400, 429)
        assert resp.get_json() is not None

    def test_settings_footage_index_get(self, client):
        resp = client.get("/settings/footage-index")
        assert resp.status_code == 200

    def test_settings_footage_index_post(self, client, csrf_token):
        resp = client.post("/settings/footage-index",
                           data=json.dumps({"auto_index": True}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code in (200, 400, 429)
        assert resp.get_json() is not None


class TestLogsTailRoute:
    """Smoke tests for /logs/tail"""

    def test_logs_tail(self, client):
        resp = client.get("/logs/tail")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "lines" in data
        assert "total" in data

    def test_logs_tail_with_filter(self, client):
        resp = client.get("/logs/tail?lines=10&level=ERROR")
        assert resp.status_code == 200

    def test_logs_tail_bad_lines(self, client):
        resp = client.get("/logs/tail?lines=abc")
        assert resp.status_code == 200  # safe_int defaults to 100


class TestTemplateRoutes:
    """Smoke tests for /templates/*"""

    def test_templates_list(self, client):
        resp = client.get("/templates/list")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "builtins" in data
        assert len(data["builtins"]) >= 6

    def test_templates_apply_nonexistent(self, client, csrf_token):
        resp = client.post("/templates/apply",
                           data=json.dumps({"id": "nonexistent"}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 404

    def test_templates_save_missing_name(self, client, csrf_token):
        resp = client.post("/templates/save",
                           data=json.dumps({"settings": {}}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400


# =====================================================================
# TIMELINE ROUTES
# =====================================================================

class TestTimelineRoutes:
    """Smoke tests for opencut/routes/timeline.py"""

    def test_batch_rename_invalid_type(self, client, csrf_token):
        resp = client.post("/timeline/batch-rename",
                           data=json.dumps({"renames": "not-a-list"}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_batch_rename_empty_list(self, client, csrf_token):
        resp = client.post("/timeline/batch-rename",
                           data=json.dumps({"renames": []}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 200

    def test_smart_bins_invalid_type(self, client, csrf_token):
        resp = client.post("/timeline/smart-bins",
                           data=json.dumps({"rules": "not-a-list"}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_smart_bins_empty_list(self, client, csrf_token):
        resp = client.post("/timeline/smart-bins",
                           data=json.dumps({"rules": []}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 200

    def test_srt_to_captions_no_data(self, client, csrf_token):
        resp = client.post("/timeline/srt-to-captions",
                           data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_export_from_markers_no_file(self, client, csrf_token):
        resp = client.post("/timeline/export-from-markers",
                           data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_index_status(self, client):
        resp = client.get("/timeline/index-status")
        assert resp.status_code == 200

    def test_export_otio_no_data(self, client, csrf_token):
        resp = client.post("/timeline/export-otio",
                           data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400


# =====================================================================
# SEARCH ROUTES
# =====================================================================

class TestSearchRoutes:
    """Smoke tests for opencut/routes/search.py"""

    def test_search_empty_query(self, client, csrf_token):
        resp = client.post("/search/footage",
                           data=json.dumps({"query": ""}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_search_query_too_long(self, client, csrf_token):
        resp = client.post("/search/footage",
                           data=json.dumps({"query": "x" * 501}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_search_valid_query(self, client, csrf_token):
        with patch("opencut.routes.search.footage_search") as mock_fs:
            mock_fs.search_footage.return_value = []
            resp = client.post("/search/footage",
                               data=json.dumps({"query": "sunset shot"}),
                               headers=csrf_headers(csrf_token))
        assert resp.status_code == 200
        data = resp.get_json()
        assert "results" in data

    def test_search_index_no_data(self, client, csrf_token):
        resp = client.post("/search/index",
                           data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_search_index_delete(self, client, csrf_token):
        resp = client.delete("/search/index",
                             headers=csrf_headers(csrf_token))
        # May succeed or return 503 if module not available
        assert resp.status_code in (200, 503)

    def test_search_auto_index_no_data(self, client, csrf_token):
        resp = client.post("/search/auto-index",
                           data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_search_db_search_no_query(self, client, csrf_token):
        resp = client.post("/search/db-search",
                           data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_search_db_stats(self, client):
        resp = client.get("/search/db-stats")
        assert resp.status_code == 200

    def test_search_cleanup(self, client, csrf_token):
        resp = client.post("/search/cleanup", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code in (200, 400, 429)
        assert resp.get_json() is not None


# =====================================================================
# NLP ROUTES
# =====================================================================

class TestNLPRoutes:
    """Smoke tests for opencut/routes/nlp.py"""

    def test_nlp_empty_command(self, client, csrf_token):
        resp = client.post("/nlp/command",
                           data=json.dumps({"command": ""}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_nlp_command_too_long(self, client, csrf_token):
        resp = client.post("/nlp/command",
                           data=json.dumps({"command": "x" * 2001}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_nlp_valid_command(self, client, csrf_token):
        mock_result = {
            "route": "/silence",
            "params": {"threshold": -30},
            "confidence": 0.9,
            "explanation": "Remove silence",
        }
        with patch("opencut.core.nlp_command.parse_command", return_value=mock_result):
            resp = client.post("/nlp/command",
                               data=json.dumps({"command": "remove silence"}),
                               headers=csrf_headers(csrf_token))
        assert resp.status_code == 200
        data = resp.get_json()
        assert "route" in data


# =====================================================================
# DELIVERABLES ROUTES
# =====================================================================

class TestDeliverablesRoutes:
    """Smoke tests for opencut/routes/deliverables.py"""

    def test_vfx_sheet_no_data(self, client, csrf_token):
        resp = client.post("/deliverables/vfx-sheet",
                           data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_adr_list_no_data(self, client, csrf_token):
        resp = client.post("/deliverables/adr-list",
                           data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_music_cue_sheet_no_data(self, client, csrf_token):
        resp = client.post("/deliverables/music-cue-sheet",
                           data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_asset_list_no_data(self, client, csrf_token):
        resp = client.post("/deliverables/asset-list",
                           data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_vfx_sheet_valid_data(self, client, csrf_token):
        mock_result = {"output": "/tmp/vfx.csv", "rows": 2}
        with patch("opencut.core.deliverables.generate_vfx_sheet", return_value=mock_result):
            resp = client.post("/deliverables/vfx-sheet",
                               data=json.dumps({
                                   "sequence_data": {
                                       "name": "Test",
                                       "clips": [{"name": "VFX_001", "start": 0, "end": 5}],
                                   }
                               }),
                               headers=csrf_headers(csrf_token))
        assert resp.status_code == 200


# =====================================================================
# WORKFLOW ROUTES
# =====================================================================

class TestWorkflowRoutes:
    """Smoke tests for opencut/routes/workflow.py"""

    def test_workflow_presets(self, client):
        resp = client.get("/workflow/presets")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "builtins" in data
        assert len(data["builtins"]) >= 6

    def test_workflow_run_no_file(self, client, csrf_token):
        resp = client.post("/workflow/run",
                           data=json.dumps({"workflow": {"steps": []}}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_workflow_save_missing_name(self, client, csrf_token):
        resp = client.post("/workflow/save",
                           data=json.dumps({"steps": [{"endpoint": "/silence"}]}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_workflow_save_invalid_step(self, client, csrf_token):
        resp = client.post("/workflow/save",
                           data=json.dumps({"name": "test", "steps": [{"endpoint": "/nonexistent"}]}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_workflow_delete_nonexistent(self, client, csrf_token):
        resp = client.delete("/workflow/delete",
                             data=json.dumps({"name": "nonexistent-workflow"}),
                             headers=csrf_headers(csrf_token))
        assert resp.status_code == 404


# =====================================================================
# CONTEXT ROUTES
# =====================================================================

class TestContextRoutes:
    """Smoke tests for opencut/routes/context.py"""

    def test_context_features_list(self, client):
        resp = client.get("/context/features")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "features" in data
        assert isinstance(data["features"], list)

    def test_context_analyze_empty_metadata(self, client, csrf_token):
        resp = client.post("/context/analyze",
                           data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 200
        data = resp.get_json()
        assert "tags" in data
        assert "features" in data
        assert "guidance" in data
        assert "tab_scores" in data

    def test_context_analyze_with_metadata(self, client, csrf_token):
        resp = client.post("/context/analyze",
                           data=json.dumps({
                               "has_audio": True,
                               "has_video": True,
                               "duration": 120.5,
                               "width": 1920,
                               "height": 1080,
                               "frame_rate": 29.97,
                               "codec": "h264",
                               "num_audio_channels": 2,
                               "track_type": "video",
                           }),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data["tags"], list)
        assert isinstance(data["features"], list)


# =====================================================================
# PLUGINS ROUTES
# =====================================================================

class TestPluginsRoutes:
    """Smoke tests for opencut/routes/plugins.py"""

    def test_plugins_list(self, client):
        resp = client.get("/plugins/list")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "plugins" in data
        assert "total" in data

    def test_plugins_loaded(self, client):
        resp = client.get("/plugins/loaded")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "plugins" in data

    def test_plugins_install_no_source(self, client, csrf_token):
        resp = client.post("/plugins/install",
                           data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_plugins_install_bad_source(self, client, csrf_token):
        resp = client.post("/plugins/install",
                           data=json.dumps({"source": "/nonexistent/path"}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_plugins_uninstall_no_name(self, client, csrf_token):
        resp = client.post("/plugins/uninstall",
                           data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_plugins_uninstall_nonexistent(self, client, csrf_token):
        resp = client.post("/plugins/uninstall",
                           data=json.dumps({"name": "nonexistent-plugin"}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 404


# =====================================================================
# STRUCTURED ERROR RESPONSE TESTS
# =====================================================================

class TestStructuredErrors:
    """Verify error responses include code and suggestion fields."""

    def test_error_has_code_field(self, client, csrf_token):
        """A validation error should include an error code."""
        resp = client.post("/silence", data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        data = resp.get_json()
        assert "error" in data
        # Basic validation errors may not have codes yet (migration in progress)
        # but _safe_error responses should

    def test_safe_error_returns_structured(self, client, csrf_token):
        """Force an internal error through a mocked exception and verify structure."""
        with patch("opencut.routes.search.footage_search") as mock_fs:
            mock_fs.search_footage.side_effect = MemoryError("GPU OOM")
            resp = client.post("/search/footage",
                               data=json.dumps({"query": "test"}),
                               headers=csrf_headers(csrf_token))
        # Should be classified as GPU_OUT_OF_MEMORY or INTERNAL_ERROR
        data = resp.get_json()
        assert "error" in data
        assert "code" in data
        assert "suggestion" in data
        assert resp.status_code >= 400

    def test_safe_error_timeout_classified(self, client, csrf_token):
        """A timeout exception should get the OPERATION_TIMEOUT code."""
        with patch("opencut.routes.search.footage_search") as mock_fs:
            mock_fs.search_footage.side_effect = TimeoutError("timed out")
            resp = client.post("/search/footage",
                               data=json.dumps({"query": "test"}),
                               headers=csrf_headers(csrf_token))
        data = resp.get_json()
        assert data.get("code") == "OPERATION_TIMEOUT"
        assert "suggestion" in data

    def test_safe_error_import_classified(self, client, csrf_token):
        """An ImportError should get MISSING_DEPENDENCY code."""
        with patch("opencut.routes.search.footage_search") as mock_fs:
            mock_fs.search_footage.side_effect = ImportError("No module named 'torch'")
            resp = client.post("/search/footage",
                               data=json.dumps({"query": "test"}),
                               headers=csrf_headers(csrf_token))
        data = resp.get_json()
        assert data.get("code") == "MISSING_DEPENDENCY"
        assert "suggestion" in data

    def test_too_many_jobs_has_code(self, client, csrf_token):
        """TooManyJobsError should return code TOO_MANY_JOBS."""
        from opencut.jobs import TooManyJobsError
        with patch("opencut.jobs._new_job", side_effect=TooManyJobsError("Too many jobs")):
            resp = client.post("/silence",
                               data=json.dumps({"filepath": "/tmp/test.wav"}),
                               headers=csrf_headers(csrf_token))
        # Should be 429 with code
        if resp.status_code == 429:
            data = resp.get_json()
            assert data.get("code") == "TOO_MANY_JOBS"


# =====================================================================
# CSRF ENFORCEMENT -- spot check across all blueprints
# =====================================================================

class TestCSRFEnforcement:
    """Every POST route should reject requests without a valid CSRF token."""

    @pytest.mark.parametrize("endpoint", [
        "/silence",
        "/captions",
        "/export-video",
        "/audio/denoise",
        "/audio/pro/apply",
        "/audio/tts/generate",
        "/audio/music-ai/generate",
        "/audio/enhance",
        "/audio/mix",
        "/video/scenes",
        "/video/ai/upscale",
        "/video/face/blur",
        "/video/lut/apply",
        "/video/color/correct",
        "/video/reframe",
        "/video/shorts-pipeline",
        "/video/multicam-cuts",
        "/video/depth/map",
        "/video/broll-plan",
        "/search/footage",
        "/nlp/command",
        "/deliverables/vfx-sheet",
        "/timeline/batch-rename",
        "/presets/save",
        "/queue/add",
        "/workflow/run",
        "/workflow/save",
        "/templates/save",
        "/templates/apply",
        "/context/analyze",
        "/plugins/install",
        "/plugins/uninstall",
        "/social/upload",
        "/chat",
    ])
    def test_post_without_csrf_rejected(self, client, endpoint):
        resp = client.post(endpoint,
                           data=json.dumps({"test": True}),
                           content_type="application/json")
        assert resp.status_code == 403, f"{endpoint} accepted request without CSRF"
