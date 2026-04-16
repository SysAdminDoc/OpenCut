"""
OpenCut Privacy & Spectral Routes

Endpoints for privacy redaction (plates, PII, profanity, documents, speaker
anonymization) and spectral audio tools (spectrogram editing, frequency
repair, noise classification, room tone fill).
"""

import logging

from flask import Blueprint

from opencut.jobs import _update_job, async_job
from opencut.security import require_csrf, safe_bool, safe_float, safe_int

logger = logging.getLogger("opencut")

privacy_spectral_bp = Blueprint("privacy_spectral", __name__)


# ===========================================================================
# Privacy Routes
# ===========================================================================

# ---------------------------------------------------------------------------
# License Plate Detection & Blur
# ---------------------------------------------------------------------------
@privacy_spectral_bp.route("/privacy/plate-blur", methods=["POST"])
@require_csrf
@async_job("plate_blur")
def plate_blur(job_id, filepath, data):
    """Detect and blur license plates in a video."""
    from opencut.core.plate_blur import blur_plates

    blur_strength = safe_int(data.get("blur_strength", 30), 30, min_val=1, max_val=100)
    sample_fps = safe_float(data.get("sample_fps", 2.0), 2.0, min_val=0.5, max_val=10.0)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = blur_plates(
        filepath,
        blur_strength=blur_strength,
        sample_fps=sample_fps,
        on_progress=_on_progress,
    )

    return result


# ---------------------------------------------------------------------------
# OCR-Based PII Redaction
# ---------------------------------------------------------------------------
@privacy_spectral_bp.route("/privacy/pii-redact", methods=["POST"])
@require_csrf
@async_job("pii_redact")
def pii_redact(job_id, filepath, data):
    """Detect and redact PII (SSN, phone, email, etc.) in video."""
    from opencut.core.pii_redact import redact_pii

    pii_types = data.get("pii_types", None)
    if isinstance(pii_types, str):
        pii_types = [t.strip() for t in pii_types.split(",") if t.strip()]
    blur_strength = safe_int(data.get("blur_strength", 30), 30, min_val=1, max_val=100)
    sample_fps = safe_float(data.get("sample_fps", 1.0), 1.0, min_val=0.25, max_val=5.0)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = redact_pii(
        filepath,
        pii_types=pii_types,
        blur_strength=blur_strength,
        sample_fps=sample_fps,
        on_progress=_on_progress,
    )

    return result


# ---------------------------------------------------------------------------
# Profanity Bleep Automation
# ---------------------------------------------------------------------------
@privacy_spectral_bp.route("/privacy/profanity-bleep", methods=["POST"])
@require_csrf
@async_job("profanity_bleep")
def profanity_bleep(job_id, filepath, data):
    """Auto-bleep profanity in audio/video based on transcript."""
    from opencut.core.profanity_bleep import bleep_profanity

    transcript_path = data.get("transcript_path", "")
    if not transcript_path:
        raise ValueError("transcript_path is required for profanity bleeping")

    custom_words = data.get("custom_words", None)
    bleep_frequency = safe_float(data.get("bleep_frequency", 1000.0), 1000.0,
                                 min_val=200.0, max_val=4000.0)
    bleep_volume = safe_float(data.get("bleep_volume", 0.5), 0.5,
                              min_val=0.1, max_val=1.0)
    mouth_blur = safe_bool(data.get("mouth_blur", False))

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = bleep_profanity(
        filepath,
        transcript_path=transcript_path,
        custom_words=custom_words,
        bleep_frequency=bleep_frequency,
        bleep_volume=bleep_volume,
        mouth_blur=mouth_blur,
        on_progress=_on_progress,
    )

    return result


# ---------------------------------------------------------------------------
# Document & Screen Redaction
# ---------------------------------------------------------------------------
@privacy_spectral_bp.route("/privacy/doc-redact", methods=["POST"])
@require_csrf
@async_job("doc_redact")
def doc_redact(job_id, filepath, data):
    """Detect and redact screens, documents, and whiteboards in video."""
    from opencut.core.doc_redact import redact_surfaces

    surface_types = data.get("surface_types", None)
    if isinstance(surface_types, str):
        surface_types = [t.strip() for t in surface_types.split(",") if t.strip()]
    redaction_mode = data.get("redaction_mode", "full")
    blur_strength = safe_int(data.get("blur_strength", 30), 30, min_val=1, max_val=100)
    sample_fps = safe_float(data.get("sample_fps", 1.0), 1.0, min_val=0.25, max_val=5.0)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = redact_surfaces(
        filepath,
        surface_types=surface_types,
        redaction_mode=redaction_mode,
        blur_strength=blur_strength,
        sample_fps=sample_fps,
        on_progress=_on_progress,
    )

    return result


# ---------------------------------------------------------------------------
# Audio Redaction & Speaker Anonymization
# ---------------------------------------------------------------------------
@privacy_spectral_bp.route("/privacy/anonymize-speaker", methods=["POST"])
@require_csrf
@async_job("anonymize_speaker")
def anonymize_speaker(job_id, filepath, data):
    """Anonymize a target speaker's voice via pitch/formant shift."""
    from opencut.core.audio_anon import anonymize_speaker as _anonymize

    target_speaker = data.get("target_speaker", "speaker_0")
    pitch_semitones = safe_float(data.get("pitch_semitones", 4.0), 4.0,
                                 min_val=-12.0, max_val=12.0)
    num_speakers = safe_int(data.get("num_speakers", 2), 2, min_val=1, max_val=10)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = _anonymize(
        filepath,
        target_speaker=target_speaker,
        pitch_semitones=pitch_semitones,
        num_speakers=num_speakers,
        on_progress=_on_progress,
    )

    return result


# ===========================================================================
# Spectral Routes
# ===========================================================================

# ---------------------------------------------------------------------------
# Visual Spectrogram Editor
# ---------------------------------------------------------------------------
@privacy_spectral_bp.route("/spectral/edit", methods=["POST"])
@require_csrf
@async_job("spectral_edit")
def spectral_edit(job_id, filepath, data):
    """Apply time-frequency mask via spectrogram editing."""
    from opencut.core.spectrogram_edit import apply_spectrogram_mask

    mask = data.get("mask", [])
    if not isinstance(mask, list):
        raise ValueError("mask must be a list of region objects")

    n_fft = safe_int(data.get("n_fft", 2048), 2048, min_val=256, max_val=8192)
    hop_length = safe_int(data.get("hop_length", 512), 512, min_val=64, max_val=n_fft)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = apply_spectrogram_mask(
        filepath,
        mask=mask,
        n_fft=n_fft,
        hop_length=hop_length,
        on_progress=_on_progress,
    )

    return result


# ---------------------------------------------------------------------------
# Spectral Repair / Frequency Removal
# ---------------------------------------------------------------------------
@privacy_spectral_bp.route("/spectral/repair", methods=["POST"])
@require_csrf
@async_job("spectral_repair")
def spectral_repair(job_id, filepath, data):
    """Remove persistent frequency peaks (hum, buzz) from audio."""
    from opencut.core.spectral_repair import repair_frequencies

    target_frequencies = data.get("target_frequencies", None)
    if isinstance(target_frequencies, str):
        target_frequencies = [float(f.strip()) for f in target_frequencies.split(",") if f.strip()]
    auto_detect = safe_bool(data.get("auto_detect", True), True)
    attenuation_db = safe_float(data.get("attenuation_db", -60.0), -60.0,
                                min_val=-120.0, max_val=-6.0)
    bandwidth = safe_float(data.get("bandwidth", 5.0), 5.0, min_val=1.0, max_val=100.0)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = repair_frequencies(
        filepath,
        target_frequencies=target_frequencies,
        auto_detect=auto_detect,
        attenuation_db=attenuation_db,
        bandwidth=bandwidth,
        on_progress=_on_progress,
    )

    return result


# ---------------------------------------------------------------------------
# AI Environmental Noise Classifier
# ---------------------------------------------------------------------------
@privacy_spectral_bp.route("/spectral/classify-noise", methods=["POST"])
@require_csrf
@async_job("classify_noise")
def classify_noise(job_id, filepath, data):
    """Classify environmental noise types in audio."""
    from opencut.core.noise_classify import classify_noise as _classify

    segment_duration = safe_float(data.get("segment_duration", 1.0), 1.0,
                                  min_val=0.25, max_val=10.0)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = _classify(
        filepath,
        segment_duration=segment_duration,
        on_progress=_on_progress,
    )

    return result


# ---------------------------------------------------------------------------
# Room Tone Auto-Generation & Fill
# ---------------------------------------------------------------------------
@privacy_spectral_bp.route("/spectral/room-tone-fill", methods=["POST"])
@require_csrf
@async_job("room_tone_fill")
def room_tone_fill(job_id, filepath, data):
    """Analyze room tone, synthesize matching fill, and apply to cut points."""
    from opencut.core.room_tone import fill_cuts_with_room_tone

    cut_points = data.get("cut_points", [])
    if not isinstance(cut_points, list):
        raise ValueError("cut_points must be a list of {time, duration} objects")

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = fill_cuts_with_room_tone(
        filepath,
        cut_points=cut_points,
        on_progress=_on_progress,
    )

    return result
