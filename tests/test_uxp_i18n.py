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
UXP_ES_LOCALE = UXP_ROOT / "locales" / "es.json"

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


def _locale(path: Path = UXP_LOCALE) -> dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    for one_key, many_key in re.findall(
        r'formatCountI18n\(\s*[^,]+,\s*"([^"]+)"\s*,\s*"[^"]*"\s*,\s*"([^"]+)"',
        js,
        re.DOTALL,
    ):
        keys.update((one_key, many_key))
    return keys


def test_uxp_locale_file_is_valid_and_local_to_panel():
    locale = _locale()

    assert UXP_LOCALE.exists()
    assert 'const UXP_DEFAULT_LOCALE = "en";' in _js()
    assert 'const UXP_LOCALE_DIR   = "locales";' in _js()
    assert "UXP_LOCALE_PATH  = `${UXP_LOCALE_DIR}/${UXP_DEFAULT_LOCALE}.json`;" in _js()
    assert locale["uxp.document_title"] == "OpenCut UXP"


def test_uxp_i18n_loader_supports_dom_text_and_attributes():
    js = _js()

    assert "function t(key, fallback)" in js
    assert "function applyI18nToDOM(root = document)" in js
    assert "function normalizeLocaleTag(lang = UXP_DEFAULT_LOCALE)" in js
    assert "function getLocaleOverride()" in js
    assert "function getPreferredLocale()" in js
    assert "function getLocaleCandidates(lang)" in js
    assert "async function fetchLocaleJson(path)" in js
    assert "async function loadLocale(lang = getPreferredLocale())" in js
    assert 'new URLSearchParams(window.location.search).get("lang")' in js
    assert "navigator.languages" in js
    assert "const baseLocale = await fetchLocaleJson(UXP_LOCALE_PATH) || {};" in js
    assert "_i18n = { ...baseLocale, ...activeLocale };" in js
    for data_attribute, dom_attribute in (
        ("data-i18n-title", "title"),
        ("data-i18n-label", "label"),
        ("data-i18n-alt", "alt"),
        ("data-i18n-placeholder", "placeholder"),
        ("data-i18n-aria-label", "aria-label"),
    ):
        assert f'["{data_attribute}", "{dom_attribute}"]' in js

    assert js.index("await loadLocale();") < js.index("bindEvents();")


def test_uxp_partial_spanish_locale_pack_uses_english_fallback():
    english = _locale()
    spanish = _locale(UXP_ES_LOCALE)

    assert UXP_ES_LOCALE.exists()
    assert 750 <= len(spanish) < len(english)
    assert sorted(key for key in spanish if key not in english) == []
    assert spanish["conn.online"] == "En linea"
    assert spanish["uxp.tabs.cut"] == "Corte"
    assert sorted(key for key in english if key.startswith("uxp.cut.") and key not in spanish) == []
    assert spanish["uxp.cut.clip_input"] == "Entrada de clip"
    assert spanish["uxp.cut.remove_silence"] == "Eliminar silencio"
    assert spanish["uxp.cut.runtime.cut_index"] == "Corte {index}"
    assert spanish["uxp.cut.runtime.cut_windows_summary"] == "{count} ventana{plural} de corte - {removed} eliminado{longest}"
    assert sorted(key for key in english if key.startswith("uxp.audio.") and key not in spanish) == []
    assert spanish["uxp.audio.denoise"] == "Reducir ruido"
    assert spanish["uxp.audio.match_loudness"] == "Igualar sonoridad"
    assert spanish["uxp.audio.runtime.beat_detection_done"] == "Deteccion de ritmo lista - {count} ritmos."
    assert sorted(key for key in english if key.startswith("uxp.fcc.") and key not in spanish) == []
    assert spanish["uxp.fcc.caption_display_settings"] == "Ajustes de visualizacion de subtitulos (FCC)"
    assert spanish["uxp.fcc.preview"] == "Vista previa"
    assert spanish["uxp.fcc.schema_unavailable"] == "No se pudo cargar el esquema de tokens FCC. La tarjeta quedara vacia."
    shared_prefixes = ("uxp.guide.", "uxp.runtime.", "uxp.status.", "uxp.workspace.")
    assert sorted(
        key
        for key in english
        if key.startswith(shared_prefixes) and key not in spanish
    ) == []
    assert spanish["uxp.guide.timeline_title"] == "Envia cambios aprobados de vuelta a Premiere con menos friccion."
    assert spanish["uxp.runtime.invalid_https_authorization_url"] == "El servidor envio una URL de autorizacion HTTPS no valida."
    assert spanish["uxp.status.workspace"] == "Espacio {workspace}"
    assert spanish["uxp.workspace.choose_source_title"] == "Elige un clip o pega una ruta para empezar"
    assert spanish["uxp.runtime.select_clip_first"] == "Selecciona primero un clip."
    assert spanish["uxp.guide.backend_offline_title"] == "Reconecta el backend local de OpenCut antes de ejecutar trabajos."
    assert spanish["uxp.settings.engine_routing"] == "Enrutamiento de motores"
    assert spanish["uxp.settings.live_updates_bridge"] == "Puente de actualizaciones en vivo"
    assert spanish["uxp.settings.migration_risk"] == "Riesgo de migracion"
    assert sorted(key for key in english if key.startswith("uxp.settings.") and key not in spanish) == []
    assert spanish["uxp.settings.engine_routing_auto_healthy"].startswith("El enrutamiento de motores esta saludable.")
    assert spanish["uxp.settings.generate_captions"] == "Generar subtitulos"
    assert spanish["uxp.settings.pinned_title_one"] == "{count} dominio tiene una preferencia de motor fijada."
    assert sorted(key for key in english if key.startswith("uxp.agent.") and key not in spanish) == []
    assert spanish["uxp.agent.chat_conductor_f143"] == "Conductor de chat (F143)"
    assert spanish["uxp.agent.runtime.plan_ready"] == "Plan: {count} paso(s) via {source}. Sesion {session}."
    assert spanish["uxp.agent.runtime.sequence_index_built"].startswith("Indice creado:")
    assert spanish["nav.feature_tabs"] == "Pestanas de funciones"
    assert sorted(key for key in english if key.startswith("uxp.deliverables.") and key not in spanish) == []
    assert spanish["uxp.deliverables.generate_deliverable_documents"] == "Generar documentos de entrega"
    assert spanish["uxp.deliverables.session_temp_folder"] == "Carpeta temporal de sesion"
    assert spanish["uxp.deliverables.session_temp_folder_title"].startswith("Los entregables se guardaran")
    assert spanish["uxp.deliverables.runtime.generated_csv_handoff_many"] == "Se generaron {count} documentos de entrega CSV."
    assert spanish["uxp.deliverables.status_line"] == "Carga informacion de secuencia, elige un destino si hace falta y genera los documentos de entrega que necesitas."
    assert sorted(key for key in english if key.startswith("uxp.search.") and key not in spanish) == []
    assert spanish["uxp.search.library_index"] == "Indice de biblioteca"
    assert spanish["uxp.search.command_placeholder"] == 'p. ej. "elimina todos los silencios de mas de 1 segundo"'
    assert spanish["uxp.search.runtime.searching_for_query"] == 'Buscando "{query}"...'
    assert spanish["uxp.search.runtime.loaded_match_toast"] == "{label} cargado en el espacio de trabajo."
    assert english["uxp.video.clip_path"] == "Clip Path"
    assert "uxp.video.clip_path" not in spanish


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
        "uxp.status.server_reconnected",
        "uxp.status.update_available",
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
        "uxp.runtime.job_cancelled",
        "uxp.runtime.just_now",
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
        "uxp.timeline.runtime.no_cuts_to_apply",
        "uxp.timeline.runtime.applied_cuts_sequence",
        "uxp.timeline.runtime.added_markers_sequence",
        "uxp.timeline.runtime.exported_segments",
        "uxp.timeline.runtime.srt_validated_segments",
        "uxp.timeline.runtime.otio_exported_output",
        "uxp.search.runtime.kind_visual",
        "uxp.search.runtime.indexing_media_library",
        "uxp.search.runtime.library_indexed_toast_many",
        "uxp.search.runtime.loaded_match_toast",
        "uxp.search.runtime.search_ready_many",
        "uxp.search.runtime.searching_for_query",
        "uxp.deliverables.runtime.no_documents_selected",
        "uxp.deliverables.runtime.sequence_ready_status",
        "uxp.deliverables.runtime.deliverable_ready_output",
        "uxp.deliverables.runtime.generated_csv_with_gaps",
        "uxp.deliverables.runtime.package_ready_status_many",
        "uxp.agent.runtime.plan_ready",
        "uxp.agent.runtime.reviewed_status",
        "uxp.agent.runtime.enhance_plan_ready",
        "uxp.agent.runtime.sequence_index_built",
        "uxp.agent.runtime.mcp_bridge_info",
        "uxp.settings.listener_count_many",
        "uxp.settings.engine_option_label",
        "uxp.settings.migration_row_summary",
        "uxp.status.server_reconnected",
        "uxp.video.runtime.matching_color_grading",
        "uxp.video.runtime.color_match_complete_output",
        "uxp.video.runtime.auto_zoom_complete_output",
        "uxp.video.runtime.multicam_cuts_generated",
        "uxp.video.runtime.generating_ai_broll",
        "uxp.video.runtime.diarization_complete_summary",
        "uxp.video.runtime.uploading_to_platform",
        "uxp.video.runtime.uploaded_to_platform",
        "uxp.video.runtime.depth_effect_complete",
        "uxp.video.runtime.depth_install_failed",
        "uxp.video.runtime.installing_depth_anything",
        "uxp.video.runtime.broll_analysis_complete",
        "uxp.video.runtime.upscaled_output",
        "uxp.video.runtime.scene_boundaries_found",
        "uxp.video.runtime.style_applied_output",
        "uxp.video.runtime.shorts_plan_state_candidates",
        "uxp.video.runtime.generated_short_form_clips",
        "uxp.video.runtime.shorts_bundle_state_outputs",
    }.issubset(js_keys)

    missing = sorted(key for key in js_keys if key not in locale)
    assert missing == []


def test_uxp_runtime_toasts_use_locale_helpers():
    js = _js()

    assert "function showSelectClipWarning()" in js
    assert 'UIController.showToast("' not in js
    assert 'UIController.showToast("Select a clip first."' not in js
    assert 'UIController.showToast("Please select a clip first."' not in js
    assert 'UIController.showToast("Server reconnected.' not in js
    assert 'UIController.showToast("Job cancelled.' not in js
    assert 'UIController.showToast("Clipboard is unavailable' not in js
    assert 'UIController.showToast("File browser not available' not in js
    assert 'UIController.showToast("Folder browser not available' not in js
    assert 'UIController.showToast("UXP Premiere Pro API available.' not in js
    assert "OpenCut v${ur.data.latest_version} available" not in js


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


def test_uxp_timeline_runtime_feedback_uses_locale_helpers():
    js = _js()
    timeline_jobs_js = js[
        js.index("/** ── APPLY TIMELINE CUTS")
        : js.index("/** ── INDEX LIBRARY")
    ]
    otio_js = js[
        js.index("// ── OTIO Export ──")
        : js.index("// ── Search ──")
    ]
    combined = timeline_jobs_js + otio_js

    assert 'UIController.showToast("No cuts to apply.' not in combined
    assert 'UIController.setStatus("Applying cuts to timeline' not in combined
    assert "UIController.showToast(`Applied ${result.applied} cut" not in combined
    assert "UIController.setStatus(`Applied ${result.applied} cut" not in combined
    assert 'UIController.setStatus("Timeline write failed' not in combined
    assert 'UIController.setStatus("UXP timeline API unavailable' not in combined
    assert 'UIController.showToast("No markers to add.' not in combined
    assert 'UIController.setStatus("Adding markers to sequence' not in combined
    assert "UIController.showToast(`Added ${result.count} marker" not in combined
    assert 'UIController.showToast("Please select an output folder.' not in combined
    assert 'UIController.showToast("No markers or cuts to export.' not in combined
    assert 'UIController.showProcessing("Starting batch export' not in combined
    assert "UIController.showToast(`Exported ${count} segment" not in combined
    assert "UIController.showToast(`Export error:" not in combined
    assert 'UIController.showToast("Batch rename still runs' not in combined
    assert 'UIController.showToast("Smart bins still execute' not in combined
    assert 'UIController.showToast("Please select an SRT file.' not in combined
    assert 'UIController.showProcessing("Validating SRT' not in combined
    assert "UIController.showToast(`Validated ${count} caption" not in combined
    assert "UIController.showToast(`SRT validation error:" not in combined
    assert 'UIController.showToast("No cuts available.' not in combined
    assert 'UIController.showToast("No markers available.' not in combined
    assert "UIController.showToast(`OTIO exported:" not in combined
    assert "UIController.showToast(`OTIO export failed:" not in combined
    assert "SRT ready" in _locale()["uxp.timeline.runtime.srt_ready_status"]
    assert "OTIO exported" in _locale()["uxp.timeline.runtime.otio_exported_output"]


def test_uxp_search_runtime_feedback_uses_locale_helpers():
    js = _js()
    search_helpers_js = js[
        js.index("function getSearchResultPath")
        : js.index("function getDeliverablesSelectionSummary")
    ]
    search_jobs_js = js[
        js.index("/** ── INDEX LIBRARY")
        : js.index("/** ── DELIVERABLES")
    ]
    combined = search_helpers_js + search_jobs_js

    assert 'return "Visual";' not in combined
    assert "return `${pct}% match`;" not in combined
    assert "Result ${index + 1}" not in combined
    assert 'UIController.showToast("Please select a media folder to index.' not in combined
    assert 'UIController.showProcessing("Indexing media library' not in combined
    assert "UIController.showToast(`Index error:" not in combined
    assert 'UIController.showToast("Please enter a search query.' not in combined
    assert 'UIController.showProcessing("Searching footage' not in combined
    assert "UIController.showToast(`Search error:" not in combined
    assert 'UIController.showToast("No NLP result to apply.' not in combined
    assert 'UIController.showToast("Unknown NLP action type.' not in combined
    assert "Searching for" in _locale()["uxp.search.runtime.searching_for_query"]
    assert "Loaded" in _locale()["uxp.search.runtime.loaded_match_toast"]


def test_uxp_deliverables_runtime_feedback_uses_locale_helpers():
    js = _js()
    deliverables_state_js = js[
        js.index("function getDeliverablesSelectionSummary")
        : js.index("/** ── INDEX LIBRARY")
    ]
    deliverables_jobs_js = js[
        js.index("/** ── DELIVERABLES")
        : js.index("function bindEvents")
    ]
    combined = deliverables_state_js + deliverables_jobs_js

    assert 'label: "No documents selected"' not in combined
    assert 'label: "Session temp folder"' not in combined
    assert 'title: "Deliverables will be saved to the session temp folder' not in combined
    assert 'title: "Select at least one handoff document' not in combined
    assert 'reportBtn.textContent = "Select Documents First"' not in combined
    assert 'const resolution = (info.width && info.height) ? `${info.width} × ${info.height}` : "Unknown size";' not in combined
    assert 'info.name || "Active Sequence"' not in combined
    assert 'UIController.showProcessing("Loading sequence info' not in combined
    assert 'UIController.showToast("No active sequence loaded.' not in combined
    assert "UIController.showProcessing(`Generating ${label}" not in combined
    assert "UIController.showToast(`Deliverable error:" not in combined
    assert 'UIController.showToast("Load sequence info before generating' not in combined
    assert 'UIController.showToast("Select at least one handoff document' not in combined
    assert "UIController.showToast(`Generated ${generated} CSV" not in combined
    assert "Sequence ready" in _locale()["uxp.deliverables.runtime.sequence_ready_status"]
    assert "Generated" in _locale()["uxp.deliverables.runtime.generated_csv_with_gaps"]


def test_uxp_agent_runtime_feedback_uses_locale_helpers():
    js = _js()
    agent_js = js[
        js.index("function initAgentTab")
        : js.index("// =============================================================\n// F236 FCC")
    ]

    assert 'li.textContent = "No steps matched' not in agent_js
    assert 'const matched = review.matched ? "Matched" : "Drift detected";' not in agent_js
    assert "summary.textContent = `${matched} (drift score" not in agent_js
    assert "Suggested retry: ${review.suggested_retry.label}" not in agent_js
    assert 'setStatus("agentChatStatus", "Enter an intent first.' not in agent_js
    assert 'setStatus("agentChatStatus", "Building plan' not in agent_js
    assert '"Plan failed: " + (resp?.error' not in agent_js
    assert "renderPlan(resp.plan)" not in agent_js
    assert "renderReview(resp)" not in agent_js
    assert 'setStatus("agentChatStatus", "Run Plan first' not in agent_js
    assert 'setStatus("agentChatStatus", "Running self-review' not in agent_js
    assert 'setStatus("agentChatStatus", "Cleared.' not in agent_js
    assert 'setStatus("enhanceStatus", "Enter a clip path first.' not in agent_js
    assert 'setStatus("enhanceStatus", "Failed: " + (resp?.error' not in agent_js
    assert "resp.steps || []" not in agent_js
    assert 'setStatus("variantsStatus", "End must be greater than start.' not in agent_js
    assert "Generated ${variants.length} variant" not in agent_js
    assert 'setStatus("sequenceIndexStatus", "Reading active sequence' not in agent_js
    assert '"No active sequence: " + (sequence?.error' not in agent_js
    assert "resp.sequence_name || \"Sequence\"" not in agent_js
    assert "info?.available" not in agent_js
    assert "info.version" not in agent_js
    assert "resp.tools || []" not in agent_js
    assert "const data = responseData(resp);" in agent_js
    assert "renderPlan(data.plan)" in agent_js
    assert "renderReview(data)" in agent_js
    assert "Plan:" in _locale()["uxp.agent.runtime.plan_ready"]
    assert "Index built" in _locale()["uxp.agent.runtime.sequence_index_built"]


def test_uxp_settings_runtime_feedback_uses_locale_helpers():
    js = _js()
    shared_runtime_js = js[
        js.index("async function checkConnection")
        : js.index("// ─────────────────────────────────────────────────────────────\n// Periodic media scan")
    ]
    settings_js = js[
        js.index("async function uxpUpdateWsStatus")
        : js.index("// ─────────────────────────────────────────────────────────────\n// Application init")
    ]
    combined = shared_runtime_js + settings_js

    assert 'UIController.showToast("Server reconnected.' not in combined
    assert 'UIController.showToast("Job cancelled.' not in combined
    assert 'return "just now";' not in combined
    assert 'countEl.textContent = `${clients} ${listenerLabel}`;' not in combined
    assert "? `${clients} ${clients === 1" not in combined
    assert "const label = `${eng.display_name} - ${eng.quality}/${eng.speed}${avail}${activeSuffix}`;" not in combined
    assert "throw new Error(`HTTP ${response.status}`);" not in combined
    assert '].filter(Boolean).join(" | ");' not in combined
    assert 'const summaryText = `${row.role || t("uxp.settings.host_action", "Host action")} ${row.replacement_plan || row.uxp_path || ""}`.trim();' not in combined
    assert "Server reconnected" in _locale()["uxp.status.server_reconnected"]
    assert "{count} listeners" in _locale()["uxp.settings.listener_count_many"]
    assert "{name}" in _locale()["uxp.settings.engine_option_label"]


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
    assert 'UIController.showToast("Installing Depth Anything V2' not in combined
    assert 'UIController.showToast("Install failed:' not in combined
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


def test_uxp_video_shorts_runtime_feedback_uses_locale_helpers():
    js = _js()
    shorts_js = js[
        js.index("// Shorts Pipeline")
        : js.index("async function initApp")
    ]

    assert 'summary.textContent = "Plan state: building candidate review board."' not in shorts_js
    assert ': "Plan state: approve at least one candidate before rendering."' not in shorts_js
    assert '? "Analyze state: cached transcript or highlight data is required before review."' not in shorts_js
    assert 'steps || "No candidate windows are available yet."' not in shorts_js
    assert "Targets: ${(candidate.platform_presets" not in shorts_js
    assert "Caption style: ${candidate.caption_style" not in shorts_js
    assert "Thumbnail: first frame at ${candidate.start" not in shorts_js
    assert "Bundle state: ${outputCount}" not in shorts_js
    assert 'UIController.showToast("Approve at least one Magic Clips candidate first."' not in shorts_js
    assert "UIController.showToast(`Rendered ${count} approved short-form clips." not in shorts_js
    assert "UIController.showToast(`Approved render failed:" not in shorts_js
    assert "UIController.showToast(`Generated ${count} short-form clips." not in shorts_js
    assert "UIController.showToast(`Shorts pipeline failed:" not in shorts_js
    assert "Generated" in _locale()["uxp.video.runtime.generated_short_form_clips"]
    assert "Bundle state" in _locale()["uxp.video.runtime.shorts_bundle_state_outputs"]


def test_uxp_connection_state_does_not_depend_on_visible_english_label():
    js = _js()

    assert "function isBackendConnected()" in js
    assert 'textContent?.trim() === "Online"' not in js
    assert 'dataset.state === "connected"' in js
