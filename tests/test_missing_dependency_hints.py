import pytest
from flask import Flask


TOP_12_HINTS = [
    ("Silero VAD requires PyTorch", "depth", "torch"),
    ("No Whisper backend installed", "captions", "faster-whisper"),
    ("Demucs not installed", "audio", "demucs"),
    ("DeepFilterNet not installed", "audio", "deepfilternet"),
    ("Depth Anything dependencies not installed", "depth", "transformers"),
    ("Robust Video Matting requires PyTorch", "depth", "torchvision"),
    ("PySceneDetect not installed", "video", "scenedetect"),
    ("RIFE neural interpolation not installed", "video", "rife"),
    ("AudioCraft MusicGen not installed", "music", "audiocraft"),
    ("f5-tts not installed", "tts", "f5-tts"),
    ("chatterbox-tts is not installed", "tts", "chatterbox-tts"),
    ("GFPGAN not installed", "ai", "gfpgan"),
]


@pytest.mark.parametrize(("message", "extra", "package"), TOP_12_HINTS)
def test_top_dependency_hints_include_extra_and_package(message, extra, package):
    from opencut.core.install_hints import build_install_suggestion

    suggestion = build_install_suggestion(message, message=message)

    assert f"pip install 'opencut[{extra}]'" in suggestion
    assert package in suggestion


def test_missing_dependency_accepts_explicit_extra_and_gpu_metadata():
    from opencut.errors import missing_dependency

    exc = missing_dependency("F5-TTS", extra="tts", gpu=True, vram_mb=8192)

    assert exc.code == "MISSING_DEPENDENCY"
    assert "pip install 'opencut[tts]'" in exc.suggestion
    assert "GPU-recommended" in exc.suggestion
    assert "8 GB" in exc.suggestion


def test_safe_error_infers_dependency_install_suggestion():
    from opencut.errors import safe_error

    app = Flask(__name__)
    with app.app_context():
        response, status = safe_error(
            RuntimeError("Demucs not installed. Please install it first."),
            context="audio_separate",
        )

    payload = response.get_json()
    assert status == 503
    assert payload["code"] == "MISSING_DEPENDENCY"
    assert "pip install 'opencut[audio]'" in payload["suggestion"]
    assert "demucs" in payload["suggestion"]


def test_safe_error_classifies_dependencies_not_installed_phrase():
    from opencut.errors import safe_error

    app = Flask(__name__)
    with app.app_context():
        response, status = safe_error(
            RuntimeError("Depth Anything dependencies not installed."),
            context="video_depth",
        )

    payload = response.get_json()
    assert status == 503
    assert payload["code"] == "MISSING_DEPENDENCY"
    assert "pip install 'opencut[depth]'" in payload["suggestion"]


def test_async_job_dependency_errors_keep_install_suggestion(monkeypatch):
    import opencut.jobs as jobs

    monkeypatch.setattr(jobs, "_persist_job", lambda job_dict, **kwargs: None)
    job_id = jobs._new_job("deepfilter", "clip.wav")

    try:
        suggestion = ""
        try:
            from opencut.core.install_hints import suggestion_for_exception

            suggestion = suggestion_for_exception(
                RuntimeError("DeepFilterNet not installed. Run: pip install deepfilternet"),
                context="deepfilter",
            )
        finally:
            jobs._update_job(
                job_id,
                status="error",
                error="DeepFilterNet not installed. Run: pip install deepfilternet",
                message="Error: DeepFilterNet not installed.",
                suggestion=suggestion,
            )

        safe = jobs._get_job_copy(job_id)
        assert safe["status"] == "error"
        assert "pip install 'opencut[audio]'" in safe["suggestion"]
        assert "deepfilternet" in safe["suggestion"]
    finally:
        with jobs.job_lock:
            jobs.jobs.pop(job_id, None)
