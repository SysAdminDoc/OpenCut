"""Static guardrails for the UXP panel i18n foundation."""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UXP_ROOT = REPO_ROOT / "extension" / "com.opencut.uxp"
UXP_HTML = UXP_ROOT / "index.html"
UXP_JS = UXP_ROOT / "main.js"
UXP_LOCALE = UXP_ROOT / "locales" / "en.json"

I18N_ATTRIBUTES = (
    "data-i18n",
    "data-i18n-title",
    "data-i18n-label",
    "data-i18n-alt",
    "data-i18n-placeholder",
    "data-i18n-aria-label",
)


def _html() -> str:
    return UXP_HTML.read_text(encoding="utf-8")


def _js() -> str:
    return UXP_JS.read_text(encoding="utf-8")


def _locale() -> dict[str, str]:
    return json.loads(UXP_LOCALE.read_text(encoding="utf-8"))


def _html_i18n_keys() -> set[str]:
    html = _html()
    keys: set[str] = set()
    for attribute in I18N_ATTRIBUTES:
        keys.update(re.findall(rf"\s{re.escape(attribute)}=\"([^\"]+)\"", html))
    return keys


def _js_i18n_keys() -> set[str]:
    js = _js()
    keys = set(re.findall(r'(?<![A-Za-z0-9_$])t\(\s*"([^"]+)"', js))
    keys.update(
        re.findall(
            r'(?<![A-Za-z0-9_$])setStatus\(\s*"([a-z0-9_.-]+\.[a-z0-9_.-]+)"',
            js,
        )
    )
    keys.update(
        re.findall(
            r"(?:titleKey|subtitleKey|kickerKey|textKey|actionLabelKey):\s*\"([^\"]+)\"",
            js,
        )
    )
    keys.update(re.findall(r'formatI18n\(\s*"([^"]+)"', js))
    return keys


def test_uxp_locale_file_is_valid_and_local_to_panel():
    locale = _locale()

    assert UXP_LOCALE.exists()
    assert "UXP_LOCALE_PATH  = \"locales/en.json\";" in _js()
    assert locale["uxp.document_title"] == "OpenCut UXP"


def test_uxp_i18n_loader_supports_dom_text_and_attributes():
    js = _js()

    assert "function t(key, fallback)" in js
    assert "function applyI18nToDOM(root = document)" in js
    assert "async function loadLocale(lang = \"en\")" in js
    for data_attribute, dom_attribute in (
        ("data-i18n-title", "title"),
        ("data-i18n-label", "label"),
        ("data-i18n-alt", "alt"),
        ("data-i18n-placeholder", "placeholder"),
        ("data-i18n-aria-label", "aria-label"),
    ):
        assert f'["{data_attribute}", "{dom_attribute}"]' in js

    assert js.index("await loadLocale();") < js.index("bindEvents();")


def test_uxp_shell_i18n_attributes_are_present_and_covered():
    html_keys = _html_i18n_keys()
    locale = _locale()

    assert len(re.findall(r"\sdata-i18n(?:-[a-z-]+)?=", _html())) >= 660
    assert {
        "common.skip_to_main",
        "conn.backend_status",
        "nav.feature_tabs",
        "processing.progress",
        "uxp.audio.clip_path",
        "uxp.audio.ai_noise_reduction",
        "uxp.audio.select_clip_placeholder",
        "uxp.audio.denoise_afftdn",
        "uxp.audio.limit_true_peak",
        "uxp.audio.reference_audio_placeholder",
        "uxp.audio.detect_beats_add_markers",
        "uxp.captions.transcription",
        "uxp.captions.workflow_readiness",
        "uxp.captions.select_clip_placeholder",
        "uxp.captions.model_turbo",
        "uxp.captions.language_auto",
        "uxp.captions.current_plan",
        "uxp.captions.result_details",
        "uxp.captions.result_placeholder",
        "uxp.cut.clip_input",
        "uxp.cut.clip_path_placeholder",
        "uxp.cut.detect_auto",
        "uxp.cut.filler_detection_backend",
        "uxp.cut.detect_remove_fillers",
        "uxp.cut.apply_cuts_to_timeline",
        "uxp.fcc.caption_display_settings",
        "uxp.fcc.compliance_notice_prefix",
        "uxp.fcc.source_link",
        "uxp.fcc.text_color",
        "uxp.fcc.preview",
        "uxp.fcc.loading_tokens",
        "uxp.fcc.caption_preview_sample",
        "uxp.video.clip_path",
        "uxp.video.color_grading_match",
        "uxp.video.reference_video_placeholder",
        "uxp.video.aspect_shorts_tiktok",
        "uxp.video.generate_multicam_cuts",
        "uxp.video.number_of_speakers",
        "uxp.video.broll_description_placeholder",
        "uxp.video.depth_not_installed",
        "uxp.video.apply_depth_effect",
        "uxp.video.emotion_highlights",
        "uxp.video.chat_placeholder",
        "uxp.video.realesrgan_video_upscale",
        "uxp.video.detect_scene_boundaries",
        "uxp.video.style_transfer",
        "uxp.video.apply_style",
        "uxp.video.shorts_pipeline",
        "uxp.video.minimum_short_duration",
        "uxp.video.face_tracking_reframe",
        "uxp.video.social_media_upload",
        "uxp.video.video_title_placeholder",
        "uxp.video.privacy_level",
        "uxp.video.connect_account",
        "uxp.timeline.preview_notice",
        "uxp.timeline.workspace_readiness",
        "uxp.timeline.apply_cuts_to_sequence",
        "uxp.timeline.export_otio_file",
        "uxp.timeline.batch_export_readiness",
        "uxp.timeline.cep_panel_required",
        "uxp.timeline.smart_bins",
        "uxp.timeline.srt_file_placeholder",
        "uxp.timeline.validate_srt_file",
        "uxp.search.library_index",
        "uxp.search.index_folder_placeholder",
        "uxp.search.suggested_search_prompts",
        "uxp.search.empty_search_hint",
        "uxp.search.nlp_command",
        "uxp.search.command_placeholder",
        "uxp.search.suggested_edit_commands",
        "uxp.search.apply_to_timeline",
        "uxp.deliverables.sequence_info",
        "uxp.deliverables.sequence_readiness",
        "uxp.deliverables.output_folder_placeholder",
        "uxp.deliverables.vfx_sheet_desc",
        "uxp.deliverables.project_report_summary",
        "uxp.deliverables.report_format_xlsx_unavailable",
        "uxp.deliverables.generate_full_report",
        "uxp.agent.chat_conductor_f143",
        "uxp.agent.intent_placeholder",
        "uxp.agent.conductor_plan_steps",
        "uxp.agent.one_click_enhance_q3",
        "uxp.agent.style_cinematic",
        "uxp.agent.shorts_ab_variants_q8",
        "uxp.agent.sequence_index_q7_f273",
        "uxp.agent.mcp_bridge_f146",
        "uxp.settings.engine_routing",
        "uxp.settings.engine_routing_summary",
        "uxp.settings.refresh_availability",
        "uxp.settings.live_updates_bridge",
        "uxp.settings.connect_live_updates",
        "uxp.settings.migration_risk",
        "uxp.settings.migration_risk_summary",
        "uxp.settings.keyboard_shortcuts",
        "uxp.settings.cancel_active_job_text",
        "uxp.settings.panel_premiere_uxp",
        "uxp.tabs.cut",
        "uxp.tabs.deliverables",
        "uxp.workspace.current_context",
        "uxp.workspace.choose_media",
        "uxp.guide.choose_media_title",
    }.issubset(html_keys)

    missing = sorted(key for key in html_keys if key not in locale)
    assert missing == []


def test_uxp_dynamic_i18n_keys_are_covered_by_locale():
    js_keys = _js_i18n_keys()
    locale = _locale()

    assert {
        "conn.online",
        "conn.connecting",
        "conn.offline",
        "uxp.status.backend_connected",
        "uxp.status.backend_offline",
        "uxp.guide.backend_offline_title",
        "uxp.guide.writeback_ready_title",
        "uxp.fcc.rendering_preview",
        "uxp.fcc.preview_failed",
        "uxp.fcc.defaults_loaded",
        "uxp.fcc.token_schema_failed",
        "uxp.workspace.library_clip_count_many",
        "uxp.settings.bridge_unavailable",
        "uxp.settings.engine_availability_failed",
        "uxp.settings.migration_dashboard_unavailable_status",
        "uxp.runtime.select_clip_first",
        "uxp.runtime.clipboard_unavailable",
        "uxp.runtime.copied_to_clipboard",
        "uxp.runtime.file_browser_unavailable",
        "uxp.runtime.folder_browser_unavailable",
        "uxp.runtime.executing_actions",
        "uxp.runtime.premiere_api_available",
        "uxp.cut.runtime.detecting_silences",
        "uxp.cut.runtime.silences_removed",
        "uxp.cut.runtime.no_silences_found",
        "uxp.cut.runtime.filler_detected",
        "uxp.cut.runtime.no_changes_yet",
        "uxp.captions.runtime.copy_unavailable_title",
        "uxp.captions.runtime.word_timing_on",
        "uxp.captions.runtime.reconnect_backend",
        "uxp.captions.runtime.review_output_ready",
        "uxp.captions.runtime.repeat_ranges_flagged",
        "uxp.captions.runtime.transcript_ready",
        "uxp.captions.runtime.chapter_generation_done_status",
        "uxp.captions.runtime.repeat_detection_done_status",
        "uxp.captions.runtime.transcribing_long",
        "uxp.captions.runtime.transcription_complete",
        "uxp.captions.runtime.generated_chapters",
        "uxp.captions.runtime.detected_repeats",
        "uxp.captions.runtime.timeline_import_needs_srt",
        "uxp.audio.runtime.applying_noise_reduction",
        "uxp.audio.runtime.denoise_complete_output",
        "uxp.audio.runtime.normalization_complete_output",
        "uxp.audio.runtime.beats_detected_add_markers",
        "uxp.video.runtime.matching_color_grading",
        "uxp.video.runtime.color_match_complete_output",
        "uxp.video.runtime.auto_zoom_complete_output",
        "uxp.video.runtime.multicam_cuts_generated",
        "uxp.video.runtime.generating_ai_broll",
        "uxp.video.runtime.diarization_complete_summary",
        "uxp.video.runtime.uploading_to_platform",
        "uxp.video.runtime.depth_effect_complete",
        "uxp.video.runtime.broll_analysis_complete",
        "uxp.video.runtime.upscaled_output",
        "uxp.video.runtime.scene_boundaries_found",
        "uxp.video.runtime.style_applied_output",
    }.issubset(js_keys)

    missing = sorted(key for key in js_keys if key not in locale)
    assert missing == []


def test_uxp_runtime_toasts_use_locale_helpers():
    js = _js()

    assert "function showSelectClipWarning()" in js
    assert 'UIController.showToast("Select a clip first."' not in js
    assert 'UIController.showToast("Clipboard is unavailable' not in js
    assert 'UIController.showToast("File browser not available' not in js
    assert 'UIController.showToast("Folder browser not available' not in js
    assert 'UIController.showToast("UXP Premiere Pro API available.' not in js


def test_uxp_cut_runtime_feedback_uses_locale_helpers():
    js = _js()
    cut_js = js[js.index("async function runSilenceRemoval") : js.index("async function runTranscribe")]

    assert 'UIController.showToast("Please select a clip first."' not in cut_js
    assert 'UIController.showProcessing("Detecting silences' not in cut_js
    assert 'UIController.setStatus("Running silence removal' not in cut_js
    assert "Removed ${cuts.length} silence region" not in cut_js
    assert 'UIController.showToast("No silences found with current settings.' not in cut_js
    assert 'UIController.showProcessing("Detecting filler words' not in cut_js
    assert "Detected ${count} filler word" not in cut_js
    assert "Filler detection done" in _locale()["uxp.cut.runtime.filler_done_status"]


def test_uxp_captions_runtime_feedback_uses_locale_helpers():
    js = _js()
    captions_state_js = js[
        js.index("function syncCaptionsActionButtons")
        : js.index("function rememberTimelineCuts")
    ]
    captions_jobs_js = js[
        js.index("async function runTranscribe")
        : js.index("/** ── DENOISE")
    ]

    assert 'copyBtn.title = copyBtn.disabled\n      ? "Copy becomes available' not in captions_state_js
    assert 'const note = `${wordLevel ? "Word timing is on"' not in captions_state_js
    assert 'setCaptionsSessionState(\n      "Offline"' not in captions_state_js
    assert '|| "Review output";' not in captions_state_js
    assert 'const headerParts = [`Repeat ${index + 1}`];' not in captions_state_js
    assert 'UIController.showToast("Please select a clip first."' not in captions_jobs_js
    assert 'UIController.showProcessing("Transcribing' not in captions_jobs_js
    assert 'UIController.showToast("Transcription complete.' not in captions_jobs_js
    assert "UIController.showToast(`Transcription error:" not in captions_jobs_js
    assert 'UIController.showProcessing("Generating chapters' not in captions_jobs_js
    assert "UIController.setStatus(`Chapter generation complete" not in captions_jobs_js
    assert "UIController.showToast(`Detected ${count} repeat" not in captions_jobs_js
    assert "UIController.setStatus(`Repeat detection done" not in captions_jobs_js
    assert "Repeat review is ready" in _locale()["uxp.captions.runtime.repeat_review_ready_status"]
    assert "Transcription complete" in _locale()["uxp.captions.runtime.transcription_complete"]


def test_uxp_audio_runtime_feedback_uses_locale_helpers():
    js = _js()
    audio_js = js[js.index("/** ── DENOISE ──") : js.index("/** ── COLOR MATCH ──")]

    assert 'UIController.showToast("Please select a clip first."' not in audio_js
    assert 'UIController.showProcessing("Applying noise reduction' not in audio_js
    assert 'UIController.showProcessing("Normalizing audio' not in audio_js
    assert 'UIController.showToast("Please select both input and reference clips.' not in audio_js
    assert 'UIController.showProcessing("Matching loudness to reference' not in audio_js
    assert 'UIController.showToast("Please select an audio/music file.' not in audio_js
    assert "Detected ${beats.length} beats" not in audio_js
    assert "Denoise complete" in _locale()["uxp.audio.runtime.denoise_complete_status"]


def test_uxp_video_core_runtime_feedback_uses_locale_helpers():
    js = _js()
    video_js = js[
        js.index("/** ── COLOR MATCH ──")
        : js.index("/** ── APPLY TIMELINE CUTS")
    ]

    assert 'UIController.showToast("Please select both input and reference clips.' not in video_js
    assert 'UIController.showProcessing("Matching color grading' not in video_js
    assert "UIController.showToast(`Color match complete. Output:" not in video_js
    assert 'UIController.showToast("Please select a clip first."' not in video_js
    assert 'UIController.showProcessing("Applying auto zoom' not in video_js
    assert "UIController.showToast(`Auto zoom complete. Output:" not in video_js
    assert 'UIController.showToast("Please select both camera files.' not in video_js
    assert 'UIController.showProcessing("Generating multicam cuts' not in video_js
    assert "Generated ${cuts.length} multicam cut point" not in video_js
    assert "Multicam cuts ready" in _locale()["uxp.video.runtime.multicam_cuts_ready_status"]


def test_uxp_video_ai_effects_runtime_feedback_uses_locale_helpers():
    js = _js()
    broll_diarize_js = js[
        js.index("// AI B-Roll Generation")
        : js.index("// Helpers")
    ]
    effects_js = js[
        js.index("// Depth Effects")
        : js.index("// Shorts Pipeline")
    ]
    combined = broll_diarize_js + effects_js

    assert 'UIController.showToast("Enter a B-roll description.' not in combined
    assert 'UIController.showProcessing("Generating AI B-roll' not in combined
    assert "UIController.showToast(`B-roll generated:" not in combined
    assert "UIController.showToast(`B-roll generation failed:" not in combined
    assert 'UIController.showToast("Select a video clip first.' not in combined
    assert 'UIController.showProcessing("Running multimodal diarization' not in combined
    assert "UIController.showToast(`Diarization complete:" not in combined
    assert "UIController.showToast(`Diarization failed:" not in combined
    assert 'UIController.showToast("Select a video to upload.' not in combined
    assert "UIController.showProcessing(`Uploading to" not in combined
    assert "UIController.showToast(`Uploaded! View at:" not in combined
    assert "UIController.showToast(`Uploaded to" not in combined
    assert "UIController.showToast(`Upload failed:" not in combined
    assert "UIController.showToast(`Opening ${platform} authorization" not in combined
    assert "UIController.showToast(`OAuth not configured" not in combined
    assert 'UIController.showProcessing("Running depth effect' not in combined
    assert "UIController.showToast(`Depth effect complete:" not in combined
    assert "UIController.showToast(`Depth effect failed:" not in combined
    assert 'UIController.showProcessing("Analyzing emotions' not in combined
    assert "UIController.showToast(`Emotion analysis complete:" not in combined
    assert "UIController.showToast(`Emotion analysis failed:" not in combined
    assert 'UIController.showProcessing("Analyzing B-roll points' not in combined
    assert "UIController.showToast(`B-roll analysis complete:" not in combined
    assert "UIController.showToast(`B-roll analysis failed:" not in combined
    assert "UIController.showToast(`Upscaled:" not in combined
    assert "UIController.showToast(`Upscale failed:" not in combined
    assert "UIController.showToast(`Found ${count} scene boundaries." not in combined
    assert "UIController.showToast(`Scene detection failed:" not in combined
    assert "UIController.showToast(`Style applied:" not in combined
    assert "UIController.showToast(`Style transfer failed:" not in combined
    assert "B-roll analysis complete" in _locale()["uxp.video.runtime.broll_analysis_complete"]
    assert "Style applied" in _locale()["uxp.video.runtime.style_applied_output"]


def test_uxp_connection_state_does_not_depend_on_visible_english_label():
    js = _js()

    assert "function isBackendConnected()" in js
    assert 'textContent?.trim() === "Online"' not in js
    assert 'dataset.state === "connected"' in js
