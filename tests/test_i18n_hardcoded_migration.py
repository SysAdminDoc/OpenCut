"""
Regression test for the E6 hardcoded-English migration pass.

`RESEARCH_FEATURE_PLAN_2026-05-25.md` E6 identified four user-visible
``showToast(...)`` strings that bypassed the i18n layer:

  - main.js:2100  "Server reconnected"
  - main.js:2125  "OpenCut v{X} available — visit GitHub to update"
  - main.js:2264  "Refreshing Premiere project media…"
  - main.js:2554  "Tip: Save your project before processing"

The follow-on commit replaced each with ``t("key", fallback)``. This
test asserts:

  * the four ``toast.*`` keys are present in ``en.json``;
  * the four sites in ``main.js`` route through ``t("toast.…", …)``;
  * the bare-English form is *no longer* used as the toast message
    argument (catches an accidental revert).
"""
from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EN_JSON = REPO_ROOT / "extension" / "com.opencut.panel" / "client" / "locales" / "en.json"
MAIN_JS = REPO_ROOT / "extension" / "com.opencut.panel" / "client" / "main.js"


WORKSPACE_METADATA_CALLS = (
    # Twenty-ninth batch (workspace header/stage metadata).
    ("tabs.cut_desc", re.compile(r'cut:\s*"Remove dead space')),
    ("tabs.captions_desc", re.compile(r'captions:\s*"Transcribe, edit')),
    ("tabs.audio_desc", re.compile(r'audio:\s*"Repair dialogue')),
    ("tabs.video_desc", re.compile(r'video:\s*"Analyze, repair')),
    ("tabs.export_desc", re.compile(r'export:\s*"Package deliverables')),
    ("tabs.timeline_desc", re.compile(r'timeline:\s*"Send markers')),
    ("tabs.search_desc", re.compile(r'nlp:\s*"Search footage')),
    ("tabs.settings_desc", re.compile(r'settings:\s*"Manage backend health')),
    (
        "tabs.default_desc",
        re.compile(r'\|\|\s*"Focused tools for the current editing workflow\."'),
    ),
    ("workspace.cut_kicker", re.compile(r'kicker:\s*"Cut Pass"')),
    ("workspace.cut_idle_title", re.compile(r'idleTitle:\s*"Select media to start the cut pass\."')),
    ("workspace.cut_idle_copy", re.compile(r'idleCopy:\s*"Choose a source')),
    ("workspace.cut_ready_title", re.compile(r'readyTitle:\s*"Cut tools are ready')),
    ("workspace.cut_ready_copy", re.compile(r'readyCopy:\s*"Run cleanup')),
    ("workspace.captions_kicker", re.compile(r'kicker:\s*"Transcript Flow"')),
    ("workspace.captions_idle_title", re.compile(r'idleTitle:\s*"Select media to build transcript')),
    ("workspace.captions_idle_copy", re.compile(r'idleCopy:\s*"Create captions')),
    ("workspace.captions_ready_title", re.compile(r'readyTitle:\s*"Caption tools are ready')),
    ("workspace.captions_ready_copy", re.compile(r'readyCopy:\s*"Transcribe once')),
    ("workspace.audio_kicker", re.compile(r'kicker:\s*"Audio Pass"')),
    ("workspace.audio_idle_title", re.compile(r'idleTitle:\s*"Select media to start the audio pass\."')),
    ("workspace.audio_idle_copy", re.compile(r'idleCopy:\s*"Denoise, normalize')),
    ("workspace.audio_ready_title", re.compile(r'readyTitle:\s*"Audio tools are ready')),
    ("workspace.audio_ready_copy", re.compile(r'readyCopy:\s*"Move from repair')),
    ("workspace.video_kicker", re.compile(r'kicker:\s*"Finishing"')),
    ("workspace.video_idle_title", re.compile(r'idleTitle:\s*"Select media to start finishing\."')),
    ("workspace.video_idle_copy", re.compile(r'idleCopy:\s*"Analyze scenes')),
    ("workspace.video_ready_title", re.compile(r'readyTitle:\s*"Video tools are ready')),
    ("workspace.video_ready_copy", re.compile(r'readyCopy:\s*"Explore crops')),
    ("workspace.export_kicker", re.compile(r'kicker:\s*"Delivery"')),
    ("workspace.export_idle_title", re.compile(r'idleTitle:\s*"Prepare delivery settings')),
    ("workspace.export_idle_copy", re.compile(r'idleCopy:\s*"Set platform presets')),
    ("workspace.export_ready_title", re.compile(r'readyTitle:\s*"Delivery tools are ready')),
    ("workspace.export_ready_copy", re.compile(r'readyCopy:\s*"Package exports')),
    ("workspace.timeline_kicker", re.compile(r'kicker:\s*"Write-Back"')),
    ("workspace.timeline_idle_title", re.compile(r'idleTitle:\s*"Prepare timeline changes')),
    ("workspace.timeline_idle_copy", re.compile(r'idleCopy:\s*"Review cuts')),
    ("workspace.timeline_ready_title", re.compile(r'readyTitle:\s*"Timeline tools are ready')),
    ("workspace.timeline_ready_copy", re.compile(r'readyCopy:\s*"Send markers')),
    ("workspace.search_kicker", re.compile(r'kicker:\s*"Library Search"')),
    ("workspace.search_idle_title", re.compile(r'idleTitle:\s*"Search footage and route commands\."')),
    ("workspace.search_idle_copy", re.compile(r'idleCopy:\s*"Index media')),
    ("workspace.search_ready_title", re.compile(r'readyTitle:\s*"Search tools are ready')),
    ("workspace.search_ready_copy", re.compile(r'readyCopy:\s*"Keep the current clip')),
    ("workspace.settings_kicker", re.compile(r'kicker:\s*"Studio Control"')),
    ("workspace.settings_idle_title", re.compile(r'idleTitle:\s*"Review studio health')),
    ("workspace.settings_idle_copy", re.compile(r'idleCopy:\s*"Check backend status')),
    ("workspace.settings_ready_title", re.compile(r'readyTitle:\s*"Settings are ready')),
    ("workspace.settings_ready_copy", re.compile(r'readyCopy:\s*"Keep backend health')),
)


MIGRATED_KEYS = (
    # First batch (the explicit RESEARCH_FEATURE_PLAN E6 list)
    "toast.server_reconnected",
    "toast.update_available",
    "toast.refreshing_media",
    "toast.save_first",
    # Second batch (high-impact missing-input alerts).
    "toast.no_clip_selected",
    "toast.choose_source_first",
    "toast.cancel_failed",
    "toast.select_clip_first",
    # Third batch (selection-failure + missing-input prompts).
    "toast.clip_path_unavailable",
    "toast.selection_unreadable",
    "toast.choose_stem_types",
    "toast.enter_broll_prompt",
    "toast.enter_tts_text",
    # Fourth batch (install/status feedback).
    "toast.demucs_installed",
    "toast.depth_installed",
    "toast.emotion_installed",
    "toast.crisper_whisper_installed",
    "toast.broll_generation_installed",
    "toast.multimodal_diarization_installed",
    "toast.watermark_remover_installed",
    "toast.watermark_region_autofilled",
    # Fifth batch (OAuth/live-update bridge feedback).
    "toast.oauth_invalid_url",
    "toast.oauth_opening_auth_page",
    "toast.oauth_not_configured",
    "toast.live_updates_already_connected",
    "toast.live_updates_bridge_open_failed",
    "toast.live_updates_connected",
    "toast.live_updates_bridge_started",
    "toast.live_updates_bridge_start_failed",
    "toast.live_updates_bridge_stopped",
    # Sixth batch (clip/workflow input prompts).
    "toast.enter_lut_path",
    "toast.transcribing_then_translating",
    "toast.no_project_clips",
    "toast.batch_requires_two_clips",
    "toast.select_preset_and_clip",
    "toast.invalid_workflow_preset",
    "toast.transcribing_then_burnin",
    "toast.enter_music_file",
    "toast.enter_pip_video",
    "toast.enter_overlay_path",
    # Seventh batch (media/caption chain prompts).
    "toast.enter_background_path",
    "toast.enter_second_clip_path",
    "toast.enter_title_text",
    "toast.enter_reference_face_path",
    "toast.transcribing_with_word_timing",
    "toast.enter_music_prompt",
    "toast.burning_in_captions_step",
    "toast.rendering_animated_captions_step",
    "toast.translating_captions_step",
    # Eighth batch (Whisper/settings status feedback).
    "toast.installing_faster_whisper",
    "toast.reinstalling_whisper",
    "toast.clearing_whisper_cache",
    "toast.whisper_cache_cleared",
    "toast.whisper_cache_clear_errors",
    "toast.whisper_cache_clear_failed",
    "toast.whisper_cpu_mode_enabled",
    "toast.whisper_cpu_mode_disabled",
    "toast.settings_update_failed",
    "toast.restarting_backend",
    # Ninth batch (import/export result feedback).
    "toast.import_error",
    "toast.opened_sequence",
    "toast.overlay_import_error",
    "toast.stem_import_error",
    "toast.caption_import_error",
    "toast.exported_to",
    "toast.export_failed",
    "toast.unknown_error",
    "toast.imported_sequence",
    "toast.import_failed",
    # Tenth batch (session replay/journal apply feedback).
    "toast.job_rerun_missing_params",
    "toast.job_rerunning",
    "toast.job_rerun_failed",
    "toast.job_replay_missing_params",
    "toast.applying_to_selection",
    "toast.selection_target",
    "toast.premiere_connection_required",
    "toast.no_markers_to_replay",
    "toast.apply_failed",
    "toast.markers_readded",
    # Eleventh batch (settings/live-update utility feedback).
    "toast.oauth_error",
    "toast.ws_start_error",
    "toast.engine_preference_error",
    "toast.preference_save_failed",
    "toast.log_file_path",
    "toast.settings_saved",
    "toast.refreshed",
    # Twelfth batch (path open and journal revert feedback).
    "toast.path_action_failed",
    "toast.path_revealed",
    "toast.path_opened",
    "toast.premiere_revert_connection_required",
    "toast.action_not_revertible",
    "toast.parse_error",
    "toast.revert_failed",
    "toast.journal_revert_update_failed",
    "toast.reverted_action",
    # Thirteenth batch (transcript cache, preview, and polish feedback).
    "toast.used_cached_transcript",
    "toast.preview_failed",
    "toast.preview_failed_http",
    "toast.preview_network_error",
    "toast.running_suggestion",
    "toast.polish_cache_clear_failed",
    "toast.polish_cache_cleared",
    "toast.polish_cache_empty",
    "toast.polish_batch_requires_two",
    "toast.polish_batch_progress",
    # Fourteenth batch (polish finish, journal clear, cut review/history feedback).
    "toast.polish_batch_done",
    "toast.polish_batch_failed_part",
    "toast.journal_clear_failed",
    "toast.journal_cleared",
    "toast.no_cuts_detected",
    "toast.no_cuts_selected",
    "toast.history_missing_output_path",
    # Fifteenth batch (preset save/load/delete feedback).
    "toast.enter_preset_name",
    "toast.preset_saved",
    "toast.preset_saved_toast",
    "toast.preset_save_failed",
    "toast.select_preset_first",
    "toast.preset_loaded",
    "toast.preset_loaded_toast",
    "toast.preset_deleted",
    "toast.preset_deleted_toast",
    "toast.select_preset_export_first",
    # Sixteenth batch (preset file import/export feedback).
    "toast.preset_export_load_failed",
    "toast.preset_exported",
    "toast.preset_invalid_missing_fields",
    "toast.preset_invalid_settings_object",
    "toast.preset_imported",
    "toast.preset_import_failed",
    "toast.preset_invalid_format",
    # Seventeenth batch (project template feedback and labels).
    "toast.select_template_first",
    "toast.template_apply_failed",
    "toast.template_applied",
    "toast.enter_template_name",
    "templates.custom_template_description",
    "toast.template_saved",
    "toast.template_save_failed",
    "templates.select",
    "templates.builtin_group",
    "templates.custom_group",
    # Eighteenth batch (transcript editor and summarize feedback).
    "toast.playhead_sync_unavailable",
    "transcript.summary_topics",
    "transcript.summary_empty",
    "transcript.timeline_editor",
    "transcript.timeline_awaiting_segments",
    "transcript.timeline_segments",
    "transcript.timeline_runtime",
    "transcript.timeline_pace",
    "transcript.timeline_avg_seconds",
    "transcript.timeline_longest",
    "transcript.timeline_select_segment",
    "transcript.timeline_segment_status",
    "transcript.timeline_empty",
    "transcript.timeline_empty_ready",
    "transcript.segment_label",
    "transcript.segment_time_label",
    "transcript.editor_info",
    "transcript.editor_empty",
    "transcript.segment_textarea_label",
    # Nineteenth batch (caption display-settings status feedback).
    "captions.display_preview_placeholder",
    "captions.display_rendering_preview",
    "captions.display_sample_text",
    "captions.display_preview_failed",
    "captions.display_preview_updated",
    "captions.display_reset_defaults",
    "captions.display_loading_tokens",
    "captions.display_schema_load_failed",
    "captions.display_defaults_loaded",
    # Twentieth batch (model management, GPU recommendation, and queue feedback).
    "models.delete_missing_path",
    "models.deleting",
    "models.deleted",
    "models.delete",
    "models.delete_failed",
    "models.scanning_title",
    "models.scanning_desc",
    "models.scanning_status",
    "models.inventory_unavailable_title",
    "models.inventory_unavailable_desc",
    "models.inventory_unavailable_status",
    "models.none_found_title",
    "models.none_found_desc",
    "models.none_found_status",
    "models.unknown_model",
    "models.unknown_source",
    "models.delete_model",
    "models.delete_model_aria",
    "models.detected_status",
    "gpu.checking",
    "gpu.recommendation_failed",
    "gpu.recommendations_applied",
    "queue.cleared",
    "queue.status",
    # Twenty-first batch (recent output browser and batch picker feedback).
    "output.ready_recent",
    "output.path_unavailable",
    "output.open",
    "output.open_title",
    "output.reveal",
    "output.reveal_title",
    "output.import_to_premiere",
    "output.import_title",
    "output.missing_path",
    "output.import_unavailable",
    "output.imported",
    "batch.no_files",
    "batch.add_hint",
    "batch.remove",
    # Twenty-fifth batch (batch run status/progress feedback).
    "batch.starting_status",
    "batch.starting_summary",
    "batch.error_status",
    "batch.unknown",
    "batch.unknown_error",
    "batch.start_failed_summary",
    "batch.running_status",
    "batch.running_summary",
    "batch.poll_failed_status",
    "batch.poll_failed_summary",
    "batch.progress_status",
    "batch.processing_summary",
    "batch.finished_summary",
    "batch.complete",
    # Twenty-sixth batch (batch summary default-state feedback).
    "batch.choose_operation",
    "batch.selected_operation",
    "batch.queue_count",
    "batch.queue_count_title",
    "batch.queue_available",
    "batch.queue_available_title",
    "batch.queue_empty",
    "batch.queue_empty_title",
    "batch.operation_ready_title",
    "batch.operation_choose_title",
    "batch.status_reconnect",
    "batch.status_load_clips",
    "batch.status_add_clips",
    "batch.status_add_one_more",
    "batch.status_ready",
    # Twenty-seventh batch (workspace shell status feedback).
    "workspace.backend_offline",
    "workspace.reconnect_title",
    "workspace.reconnect_copy",
    "workspace.awaiting_media",
    "workspace.choose_clip_title",
    "workspace.default_suite",
    "workspace.status_reconnect",
    "workspace.status_reconnect_title",
    "workspace.status_source_ready",
    "workspace.status_source_ready_title",
    "workspace.status_settings_ready",
    "workspace.status_settings_ready_title",
    "workspace.status_select_media",
    "workspace.status_select_media_title",
    "workspace.clip_choose",
    # Twenty-eighth batch (connection and project-media scan feedback).
    "conn.connected",
    "conn.connected_port",
    "conn.connected_gpu",
    "conn.disconnected",
    "conn.dot_connected",
    "conn.dot_disconnected",
    "conn.port",
    "conn.reconnecting",
    "media.scanning_premiere",
    "media.load_failed",
    "media.no_importable",
    "media.select_clip_placeholder",
    "media.no_project_media",
    "media.untitled_clip",
    "media.read_failed",
    # Thirtieth batch (central job lifecycle feedback).
    "progress.busy",
    "progress.step_prefix",
    "progress.preparing",
    "progress.start_failed",
    "progress.start_failed_prefix",
    "progress.processing",
    "progress.run_failed",
    "progress.unknown_error",
    "progress.finished",
    "progress.success_summary",
    # Thirty-first batch (central cleaned error guidance).
    "error.route_not_found",
    "error.backend_problem",
    "error.request_retry",
    "error.backend_unreachable",
    "error.source_missing",
    "error.file_permission",
    "error.timeout_retry",
    "error.install_from_settings",
    "error.memory_retry",
    "error.permission_check",
    "error.file_moved",
    "error.operation_timeout",
    "error.server_running",
    # Thirty-second batch (install/action helper feedback).
    "alert.open_target",
    "common.working",
    "install.default_start",
    "install.cancelled",
    "install.failed",
    "install.demucs_start",
    "install.depth_start",
    "install.emotion_start",
    "install.crisper_whisper_start",
    "install.broll_generation_start",
    "install.multimodal_diarization_start",
    "install.watermark_start",
    "error.gpu_out_of_memory",
    "error.file_not_found_reselect",
    "error.too_many_jobs",
    # Thirty-third batch (helper/status microcopy).
    "alert.issue_bundle_url",
    "alert.no_demo_footage",
    "audio.measuring",
    "common.auto",
    "interview.server_disconnected",
    "interview.select_clip_to_run",
    "interview.runs_on",
    "social.view_on",
    "social.connected_as",
    "social.uploaded_to",
    "toast.bridgetalk_async_ready",
    "toast.demo_fetch_failed",
    "toast.demo_loaded",
    "toast.issue_report_opened",
    "video.realesrgan_not_installed",
    "video.rembg_not_installed",
    "video.watermark_detected_region",
    "video.watermark_not_detected",
    # Thirty-fourth batch (shortcut labels and numeric validation).
    "silence.detect",
    "shortcuts.captions",
    "shortcuts.normalize",
    "shortcuts.denoise",
    "shortcuts.export",
    "shortcuts.command_palette",
    "shortcuts.cancel",
    "shortcuts.workflow",
    "common.value",
    "forms.highlight_duration",
    "forms.short_duration",
    "toast.number_adjusted",
    "toast.number_kept_in_range",
    # Thirty-fifth batch (status bar and language fallback).
    "status.uptime",
    "status.cpu_usage",
    "status.ram_usage",
    "status.gpu_unavailable",
    "status.jobs_running",
    "status.jobs_queued",
    "status.jobs_done_today",
    "status.jobs_summary",
    "status.jobs_none",
    "toast.language_unavailable",
    # Twenty-second batch (custom workflow builder feedback).
    "workflow.step_count",
    "workflow.enter_name",
    "workflow.add_step_first",
    "workflow.save_failed",
    "workflow.saved",
    "workflow.saved_summary",
    "workflow.loaded_summary",
    "workflow.deleted",
    "workflow.deleted_summary",
    "workflow.custom_default",
    "workflow.running_on",
    "workflow.empty_title",
    "workflow.empty_hint",
    "workflow.remove",
    "workflow.saved_unavailable",
    "workflow.load_saved_failed",
    "workflow.no_custom_workflows",
    # Twenty-third batch (workflow preset loading/run/completion feedback).
    "workflow.preset_unavailable",
    "workflow.load_presets_failed",
    "workflow.builtin_group",
    "workflow.custom_group",
    "workflow.no_presets",
    "workflow.preset_running_on",
    "workflow.complete",
    "workflow.complete_output_suffix",
    "workflow.failed",
    "workflow.unknown_error",
    "workflow.cancelled",
    # Twenty-fourth batch (workflow preset summary/status feedback).
    "workflow.preset_desc_choose",
    "workflow.preset_desc_steps",
    "workflow.preset_loading_pill",
    "workflow.preset_loading_title",
    "workflow.preset_loading_summary",
    "workflow.preset_empty_pill",
    "workflow.preset_empty_title",
    "workflow.preset_empty_summary",
    "workflow.preset_empty_hint",
    "workflow.preset_choose_pill",
    "workflow.preset_choose_title",
    "workflow.preset_count_summary",
    "workflow.preset_count_title",
    "workflow.preset_ready_pill",
    "workflow.preset_ready_title",
    "workflow.preset_summary_label",
    "workflow.preset_summary_title",
    "workflow.preset_status_loading",
    "workflow.preset_status_reconnect",
    "workflow.preset_status_empty",
    "workflow.preset_status_choose",
    "workflow.preset_status_choose_clip",
    "workflow.preset_status_ready_on",
)


# Each tuple: (i18n_key, regex of expected `t("key", ...)` invocation,
# regex of the now-banned bare-English `showToast("...")` form that the
# original code shipped).
EXPECTED_CALLS = (
    (
        "toast.server_reconnected",
        re.compile(r't\(\s*"toast\.server_reconnected"'),
        re.compile(r'showToast\(\s*"Server reconnected"'),
    ),
    (
        "toast.refreshing_media",
        re.compile(r't\(\s*"toast\.refreshing_media"'),
        re.compile(r'showToast\(\s*"Refreshing Premiere project media'),
    ),
    (
        "toast.save_first",
        re.compile(r't\(\s*"toast\.save_first"'),
        re.compile(r'showToast\(\s*"Tip: Save your project before processing"'),
    ),
    (
        # The update-available site interpolates {version}; we only assert
        # the i18n lookup happens — the bare form check is loose because
        # the fallback path still uses the English literal as a safety
        # net when the locale lacks the templated form.
        "toast.update_available",
        re.compile(r't\(\s*"toast\.update_available"'),
        None,
    ),
    # --- Second batch -----------------------------------------------
    (
        "toast.no_clip_selected",
        re.compile(r't\(\s*"toast\.no_clip_selected"'),
        re.compile(r'showAlert\(\s*"No clip selected in timeline\."\s*\)'),
    ),
    (
        "toast.choose_source_first",
        re.compile(r't\(\s*"toast\.choose_source_first"'),
        re.compile(r'showAlert\(\s*"Choose a source in Media before running this tool\."\s*\)'),
    ),
    (
        "toast.cancel_failed",
        re.compile(r't\(\s*"toast\.cancel_failed"'),
        # Use a relaxed bare-form check because the smart quote in the
        # i18n value won't match the original ASCII apostrophe.
        re.compile(r"showToast\(\s*\"Couldn't cancel "),
    ),
    (
        "toast.select_clip_first",
        re.compile(r't\(\s*"toast\.select_clip_first"'),
        re.compile(r'showAlert\(\s*"Select a clip first\."\s*\)'),
    ),
    # --- Third batch ------------------------------------------------
    (
        "toast.clip_path_unavailable",
        re.compile(r't\(\s*"toast\.clip_path_unavailable"'),
        re.compile(r'showAlert\(\s*"Could not get clip path\."\s*\)'),
    ),
    (
        "toast.selection_unreadable",
        re.compile(r't\(\s*"toast\.selection_unreadable"'),
        re.compile(r'showAlert\(\s*"Could not read selection\."\s*\)'),
    ),
    (
        "toast.choose_stem_types",
        re.compile(r't\(\s*"toast\.choose_stem_types"'),
        re.compile(r'showAlert\(\s*"Choose at least one stem type to extract'),
    ),
    (
        "toast.enter_broll_prompt",
        re.compile(r't\(\s*"toast\.enter_broll_prompt"'),
        re.compile(r'showAlert\(\s*"Enter a description for the B-roll clip\."\s*\)'),
    ),
    (
        "toast.enter_tts_text",
        re.compile(r't\(\s*"toast\.enter_tts_text"'),
        re.compile(r'showAlert\(\s*"Enter text to generate speech\."\s*\)'),
    ),
    # --- Fourth batch -----------------------------------------------
    (
        "toast.demucs_installed",
        re.compile(r't\(\s*"toast\.demucs_installed"'),
        re.compile(r'showAlert\(\s*"Demucs installed successfully!"\s*\)'),
    ),
    (
        "toast.depth_installed",
        re.compile(r't\(\s*"toast\.depth_installed"'),
        re.compile(r'showAlert\(\s*"Depth Anything V2 installed successfully!"\s*\)'),
    ),
    (
        "toast.emotion_installed",
        re.compile(r't\(\s*"toast\.emotion_installed"'),
        re.compile(r'showAlert\(\s*"Emotion analysis installed successfully!"\s*\)'),
    ),
    (
        "toast.crisper_whisper_installed",
        re.compile(r't\(\s*"toast\.crisper_whisper_installed"'),
        re.compile(r'showAlert\(\s*"CrisperWhisper installed successfully!"\s*\)'),
    ),
    (
        "toast.broll_generation_installed",
        re.compile(r't\(\s*"toast\.broll_generation_installed"'),
        re.compile(r'showAlert\(\s*"AI B-roll generation installed successfully!"\s*\)'),
    ),
    (
        "toast.multimodal_diarization_installed",
        re.compile(r't\(\s*"toast\.multimodal_diarization_installed"'),
        re.compile(r'showAlert\(\s*"Multimodal diarization installed successfully!"\s*\)'),
    ),
    (
        "toast.watermark_remover_installed",
        re.compile(r't\(\s*"toast\.watermark_remover_installed"'),
        re.compile(r'showAlert\(\s*"Watermark remover installed successfully!"\s*\)'),
    ),
    (
        "toast.watermark_region_autofilled",
        re.compile(r't\(\s*"toast\.watermark_region_autofilled"'),
        re.compile(r'showToast\(\s*"Watermark detected'),
    ),
    # --- Fifth batch ------------------------------------------------
    (
        "toast.oauth_invalid_url",
        re.compile(r't\(\s*"toast\.oauth_invalid_url"'),
        re.compile(r'showAlert\(\s*"Invalid authorization URL received from server\."\s*\)'),
    ),
    (
        "toast.oauth_opening_auth_page",
        re.compile(r't\(\s*"toast\.oauth_opening_auth_page"'),
        re.compile(r'showToast\(\s*"Opening "'),
    ),
    (
        "toast.oauth_not_configured",
        re.compile(r't\(\s*"toast\.oauth_not_configured"'),
        re.compile(r'showAlert\(\s*"OAuth not configured for '),
    ),
    (
        "toast.live_updates_already_connected",
        re.compile(r't\(\s*"toast\.live_updates_already_connected"'),
        re.compile(r'showToast\(\s*"Live updates are already connected"'),
    ),
    (
        "toast.live_updates_bridge_open_failed",
        re.compile(r't\(\s*"toast\.live_updates_bridge_open_failed"'),
        re.compile(r'showToast\(\s*"Could not open the live-updates bridge"'),
    ),
    (
        "toast.live_updates_connected",
        re.compile(r't\(\s*"toast\.live_updates_connected"'),
        re.compile(r'showToast\(\s*"Live updates connected"'),
    ),
    (
        "toast.live_updates_bridge_started",
        re.compile(r't\(\s*"toast\.live_updates_bridge_started"'),
        re.compile(r'showToast\(\s*"Live-updates bridge started"'),
    ),
    (
        "toast.live_updates_bridge_start_failed",
        re.compile(r't\(\s*"toast\.live_updates_bridge_start_failed"'),
        re.compile(r':\s*"Failed to start WebSocket bridge"'),
    ),
    (
        "toast.live_updates_bridge_stopped",
        re.compile(r't\(\s*"toast\.live_updates_bridge_stopped"'),
        re.compile(r'showToast\(\s*"Live-updates bridge stopped"'),
    ),
    # --- Sixth batch ------------------------------------------------
    (
        "toast.enter_lut_path",
        re.compile(r't\(\s*"toast\.enter_lut_path"'),
        re.compile(r'showAlert\(\s*"Please enter a LUT file path"\s*\)'),
    ),
    (
        "toast.transcribing_then_translating",
        re.compile(r't\(\s*"toast\.transcribing_then_translating"'),
        re.compile(r'showAlert\(\s*"Step 1/2: Transcribing first, then translating'),
    ),
    (
        "toast.no_project_clips",
        re.compile(r't\(\s*"toast\.no_project_clips"'),
        re.compile(r'showAlert\(\s*"No clips found in project\. Load clips first\."\s*\)'),
    ),
    (
        "toast.batch_requires_two_clips",
        re.compile(r't\(\s*"toast\.batch_requires_two_clips"'),
        re.compile(r'showAlert\(\s*"Only 1 clip found\. Batch requires 2\+ files\."\s*\)'),
    ),
    (
        "toast.select_preset_and_clip",
        re.compile(r't\(\s*"toast\.select_preset_and_clip"'),
        re.compile(r'showAlert\(\s*"Select a preset and a clip first\."\s*\)'),
    ),
    (
        "toast.invalid_workflow_preset",
        re.compile(r't\(\s*"toast\.invalid_workflow_preset"'),
        re.compile(r'showAlert\(\s*"Invalid workflow preset\."\s*\)'),
    ),
    (
        "toast.transcribing_then_burnin",
        re.compile(r't\(\s*"toast\.transcribing_then_burnin"'),
        re.compile(r'showAlert\(\s*"Step 1/2: Transcribing first, then burning'),
    ),
    (
        "toast.enter_music_file",
        re.compile(r't\(\s*"toast\.enter_music_file"'),
        re.compile(r'showAlert\(\s*"Enter a music file path\."\s*\)'),
    ),
    (
        "toast.enter_pip_video",
        re.compile(r't\(\s*"toast\.enter_pip_video"'),
        re.compile(r'showAlert\(\s*"Enter PiP video path\."\s*\)'),
    ),
    (
        "toast.enter_overlay_path",
        re.compile(r't\(\s*"toast\.enter_overlay_path"'),
        re.compile(r'showAlert\(\s*"Enter overlay path\."\s*\)'),
    ),
    # --- Seventh batch ----------------------------------------------
    (
        "toast.enter_background_path",
        re.compile(r't\(\s*"toast\.enter_background_path"'),
        re.compile(r'showAlert\(\s*"Enter background path\."\s*\)'),
    ),
    (
        "toast.enter_second_clip_path",
        re.compile(r't\(\s*"toast\.enter_second_clip_path"'),
        re.compile(r'showAlert\(\s*"Enter second clip path\."\s*\)'),
    ),
    (
        "toast.enter_title_text",
        re.compile(r't\(\s*"toast\.enter_title_text"'),
        re.compile(r'showAlert\(\s*"Enter title text\."\s*\)'),
    ),
    (
        "toast.enter_reference_face_path",
        re.compile(r't\(\s*"toast\.enter_reference_face_path"'),
        re.compile(r'showAlert\(\s*"Enter reference face image path\."\s*\)'),
    ),
    (
        "toast.transcribing_with_word_timing",
        re.compile(r't\(\s*"toast\.transcribing_with_word_timing"'),
        re.compile(r'showAlert\(\s*"Step 1/2: Transcribing with word-level timing'),
    ),
    (
        "toast.enter_music_prompt",
        re.compile(r't\(\s*"toast\.enter_music_prompt"'),
        re.compile(r'showAlert\(\s*"Enter a music prompt\."\s*\)'),
    ),
    (
        "toast.burning_in_captions_step",
        re.compile(r't\(\s*"toast\.burning_in_captions_step"'),
        re.compile(r'showAlert\(\s*"Step 2/2: Burning in captions'),
    ),
    (
        "toast.rendering_animated_captions_step",
        re.compile(r't\(\s*"toast\.rendering_animated_captions_step"'),
        re.compile(r'showAlert\(\s*"Step 2/2: Rendering animated captions'),
    ),
    (
        "toast.translating_captions_step",
        re.compile(r't\(\s*"toast\.translating_captions_step"'),
        re.compile(r'showAlert\(\s*"Step 2/2: Translating captions'),
    ),
    # --- Eighth batch -----------------------------------------------
    (
        "toast.installing_faster_whisper",
        re.compile(r't\(\s*"toast\.installing_faster_whisper"'),
        re.compile(r'showAlert\(\s*"Installing faster-whisper'),
    ),
    (
        "toast.reinstalling_whisper",
        re.compile(r't\(\s*"toast\.reinstalling_whisper"'),
        re.compile(r'showAlert\(\s*"Reinstalling Whisper"'),
    ),
    (
        "toast.clearing_whisper_cache",
        re.compile(r't\(\s*"toast\.clearing_whisper_cache"'),
        re.compile(r'showAlert\(\s*"Clearing Whisper cache'),
    ),
    (
        "toast.whisper_cache_cleared",
        re.compile(r't\(\s*"toast\.whisper_cache_cleared"'),
        re.compile(r'showAlert\(\s*"Cache cleared! Cleared "'),
    ),
    (
        "toast.whisper_cache_clear_errors",
        re.compile(r't\(\s*"toast\.whisper_cache_clear_errors"'),
        re.compile(r'showAlert\(\s*"Cache clear had errors: "'),
    ),
    (
        "toast.whisper_cache_clear_failed",
        re.compile(r't\(\s*"toast\.whisper_cache_clear_failed"'),
        re.compile(r'showAlert\(\s*"Failed to clear cache\."\s*\)'),
    ),
    (
        "toast.whisper_cpu_mode_enabled",
        re.compile(r't\(\s*"toast\.whisper_cpu_mode_enabled"'),
        re.compile(r'showAlert\(\s*"CPU mode enabled\. Transcription may be slower but more stable\."\s*\)'),
    ),
    (
        "toast.whisper_cpu_mode_disabled",
        re.compile(r't\(\s*"toast\.whisper_cpu_mode_disabled"'),
        re.compile(r'showAlert\(\s*"CPU mode disabled\. Whisper will try to use GPU\."\s*\)'),
    ),
    (
        "toast.settings_update_failed",
        re.compile(r't\(\s*"toast\.settings_update_failed"'),
        re.compile(r'showAlert\(\s*"Failed to update settings\."\s*\)'),
    ),
    (
        "toast.restarting_backend",
        re.compile(r't\(\s*"toast\.restarting_backend"'),
        re.compile(r'showAlert\(\s*"Restarting backend'),
    ),
    # --- Ninth batch ------------------------------------------------
    (
        "toast.import_error",
        re.compile(r't\(\s*"toast\.import_error"'),
        re.compile(r'showAlert\(\s*"Import error: "'),
    ),
    (
        "toast.opened_sequence",
        re.compile(r't\(\s*"toast\.opened_sequence"'),
        re.compile(r'showAlert\(\s*"Opened: "'),
    ),
    (
        "toast.overlay_import_error",
        re.compile(r't\(\s*"toast\.overlay_import_error"'),
        re.compile(r'showAlert\(\s*"Overlay import error: "'),
    ),
    (
        "toast.stem_import_error",
        re.compile(r't\(\s*"toast\.stem_import_error"'),
        re.compile(r'showAlert\(\s*"Stem import error: "'),
    ),
    (
        "toast.caption_import_error",
        re.compile(r't\(\s*"toast\.caption_import_error"'),
        re.compile(r'showAlert\(\s*"Caption import error: "'),
    ),
    (
        "toast.exported_to",
        re.compile(r't\(\s*"toast\.exported_to"'),
        re.compile(r'showAlert\(\s*"Exported to: "'),
    ),
    (
        "toast.export_failed",
        re.compile(r't\(\s*"toast\.export_failed"'),
        re.compile(r'showAlert\(\s*"Export failed: "'),
    ),
    (
        "toast.unknown_error",
        re.compile(r't\(\s*"toast\.unknown_error"'),
        None,
    ),
    (
        "toast.imported_sequence",
        re.compile(r't\(\s*"toast\.imported_sequence"'),
        re.compile(r'showToast\(\s*"Imported \''),
    ),
    (
        "toast.import_failed",
        re.compile(r't\(\s*"toast\.import_failed"'),
        re.compile(r'showAlert\(\s*"Import failed: "'),
    ),
    # --- Tenth batch ------------------------------------------------
    (
        "toast.job_rerun_missing_params",
        re.compile(r't\(\s*"toast\.job_rerun_missing_params"'),
        re.compile(r'showToast\(\s*"Nothing to re-run'),
    ),
    (
        "toast.job_rerunning",
        re.compile(r't\(\s*"toast\.job_rerunning"'),
        re.compile(r'showToast\(\s*"Re-running "'),
    ),
    (
        "toast.job_rerun_failed",
        re.compile(r't\(\s*"toast\.job_rerun_failed"'),
        re.compile(r'showAlert\(\s*"Re-run failed: "'),
    ),
    (
        "toast.job_replay_missing_params",
        re.compile(r't\(\s*"toast\.job_replay_missing_params"'),
        re.compile(r'showToast\(\s*"Job params aren\'t recorded'),
    ),
    (
        "toast.applying_to_selection",
        re.compile(r't\(\s*"toast\.applying_to_selection"'),
        re.compile(r'showToast\(\s*"Applying "\s*\+\s*_(?:sessionCtxOpText|journalActionLabel)'),
    ),
    (
        "toast.selection_target",
        re.compile(r't\(\s*"toast\.selection_target"'),
        None,
    ),
    (
        "toast.premiere_connection_required",
        re.compile(r't\(\s*"toast\.premiere_connection_required"'),
        re.compile(r'showAlert\(\s*"Premiere connection required\."\s*\)'),
    ),
    (
        "toast.no_markers_to_replay",
        re.compile(r't\(\s*"toast\.no_markers_to_replay"'),
        re.compile(r'showAlert\(\s*"No markers to replay\."\s*\)'),
    ),
    (
        "toast.apply_failed",
        re.compile(r't\(\s*"toast\.apply_failed"'),
        re.compile(r'showAlert\(\s*"Apply failed: "'),
    ),
    (
        "toast.markers_readded",
        re.compile(r't\(\s*"toast\.markers_readded"'),
        re.compile(r'showToast\(\s*"Re-added "'),
    ),
    # --- Eleventh batch ---------------------------------------------
    (
        "toast.oauth_error",
        re.compile(r't\(\s*"toast\.oauth_error"'),
        re.compile(r'showAlert\(\s*"OAuth error: "'),
    ),
    (
        "toast.ws_start_error",
        re.compile(r't\(\s*"toast\.ws_start_error"'),
        re.compile(r'showAlert\(\s*"WS start error: "'),
    ),
    (
        "toast.engine_preference_error",
        re.compile(r't\(\s*"toast\.engine_preference_error"'),
        re.compile(r'showAlert\(\s*"Error: "\s*\+\s*perr\.message'),
    ),
    (
        "toast.preference_save_failed",
        re.compile(r't\(\s*"toast\.preference_save_failed"'),
        re.compile(r':\s*"Failed to save preference"'),
    ),
    (
        "toast.log_file_path",
        re.compile(r't\(\s*"toast\.log_file_path"'),
        re.compile(r'showAlert\(\s*"Log file: "'),
    ),
    (
        "toast.settings_saved",
        re.compile(r't\(\s*"toast\.settings_saved"'),
        re.compile(r'showToast\(\s*"Settings saved"'),
    ),
    (
        "toast.refreshed",
        re.compile(r't\(\s*"toast\.refreshed"'),
        re.compile(r'showAlert\(\s*"Refreshed"\s*\)'),
    ),
    # --- Twelfth batch ----------------------------------------------
    (
        "toast.path_action_failed",
        re.compile(r't\(\s*"toast\.path_action_failed"'),
        re.compile(r'showToast\(\s*"Couldn\'t "\s*\+\s*mode'),
    ),
    (
        "toast.path_revealed",
        re.compile(r't\(\s*"toast\.path_revealed"'),
        re.compile(r'"Revealed in file manager"\s*:\s*"Opened"'),
    ),
    (
        "toast.path_opened",
        re.compile(r't\(\s*"toast\.path_opened"'),
        re.compile(r'"Revealed in file manager"\s*:\s*"Opened"'),
    ),
    (
        "toast.premiere_revert_connection_required",
        re.compile(r't\(\s*"toast\.premiere_revert_connection_required"'),
        re.compile(r'showAlert\(\s*"Premiere Pro connection required to revert\."\s*\)'),
    ),
    (
        "toast.action_not_revertible",
        re.compile(r't\(\s*"toast\.action_not_revertible"'),
        re.compile(r'showAlert\(\s*"This action can\'t be reverted automatically\."\s*\)'),
    ),
    (
        "toast.parse_error",
        re.compile(r't\(\s*"toast\.parse_error"'),
        re.compile(r'error:\s*result\s*\|\|\s*"Parse error"'),
    ),
    (
        "toast.revert_failed",
        re.compile(r't\(\s*"toast\.revert_failed"'),
        re.compile(r'showAlert\(\s*"Revert failed: "'),
    ),
    (
        "toast.journal_revert_update_failed",
        re.compile(r't\(\s*"toast\.journal_revert_update_failed"'),
        re.compile(r'showToast\(\s*"Reverted in Premiere but couldn\'t update the journal"'),
    ),
    (
        "toast.reverted_action",
        re.compile(r't\(\s*"toast\.reverted_action"'),
        re.compile(r'showToast\(\s*"Reverted: "'),
    ),
    # --- Thirteenth batch -------------------------------------------
    (
        "toast.used_cached_transcript",
        re.compile(r't\(\s*"toast\.used_cached_transcript"'),
        re.compile(r'showToast\(\s*"Used cached transcript'),
    ),
    (
        "toast.preview_failed",
        re.compile(r't\(\s*"toast\.preview_failed"'),
        re.compile(r'showAlert\(\s*"Preview failed: "'),
    ),
    (
        "toast.preview_failed_http",
        re.compile(r't\(\s*"toast\.preview_failed_http"'),
        re.compile(r'showAlert\(\s*"Preview failed \(HTTP "'),
    ),
    (
        "toast.preview_network_error",
        re.compile(r't\(\s*"toast\.preview_network_error"'),
        re.compile(r'showAlert\(\s*"Preview network error"\s*\)'),
    ),
    (
        "toast.running_suggestion",
        re.compile(r't\(\s*"toast\.running_suggestion"'),
        re.compile(r'showToast\(\s*"Running "\s*\+\s*sug\.title'),
    ),
    (
        "toast.polish_cache_clear_failed",
        re.compile(r't\(\s*"toast\.polish_cache_clear_failed"'),
        re.compile(r'showAlert\(\s*"Couldn\'t clear cache: "'),
    ),
    (
        "toast.polish_cache_cleared",
        re.compile(r't\(\s*"toast\.polish_cache_cleared"'),
        re.compile(r'showToast\(\s*"Cached transcript cleared for this clip\."'),
    ),
    (
        "toast.polish_cache_empty",
        re.compile(r't\(\s*"toast\.polish_cache_empty"'),
        re.compile(r'showToast\(\s*"No cached transcript to clear\."'),
    ),
    (
        "toast.polish_batch_requires_two",
        re.compile(r't\(\s*"toast\.polish_batch_requires_two"'),
        re.compile(r'showAlert\(\s*"Add at least 2 files to the batch picker first\."'),
    ),
    (
        "toast.polish_batch_progress",
        re.compile(r't\(\s*"toast\.polish_batch_progress"'),
        re.compile(r'showToast\(\s*"Polishing "'),
    ),
    # --- Fourteenth batch -------------------------------------------
    (
        "toast.polish_batch_done",
        re.compile(r't\(\s*"toast\.polish_batch_done"'),
        re.compile(r'showAlert\(\s*"Batch polish done: "'),
    ),
    (
        "toast.polish_batch_failed_part",
        re.compile(r't\(\s*"toast\.polish_batch_failed_part"'),
        re.compile(r'", "\s*\+\s*failed\s*\+\s*" failed"'),
    ),
    (
        "toast.journal_clear_failed",
        re.compile(r't\(\s*"toast\.journal_clear_failed"'),
        re.compile(r'showAlert\(\s*"Could not clear: "'),
    ),
    (
        "toast.journal_cleared",
        re.compile(r't\(\s*"toast\.journal_cleared"'),
        re.compile(r'showToast\(\s*"Journal cleared"'),
    ),
    (
        "toast.no_cuts_detected",
        re.compile(r't\(\s*"toast\.no_cuts_detected"'),
        re.compile(r'showAlert\(\s*"No cuts detected in this clip\."\s*\)'),
    ),
    (
        "toast.no_cuts_selected",
        re.compile(r't\(\s*"toast\.no_cuts_selected"'),
        re.compile(r'showAlert\(\s*"No cuts selected\. Select at least one cut to apply\."\s*\)'),
    ),
    (
        "toast.history_missing_output_path",
        re.compile(r't\(\s*"toast\.history_missing_output_path"'),
        re.compile(r'showToast\(\s*"This history item is missing an output path\."'),
    ),
    (
        "toast.applying_to_selection",
        re.compile(r't\(\s*"toast\.applying_to_selection"'),
        re.compile(r'showToast\(\s*"Applying \'"\s*\+\s*entry\.type'),
    ),
    # --- Fifteenth batch --------------------------------------------
    (
        "toast.enter_preset_name",
        re.compile(r't\(\s*"toast\.enter_preset_name"'),
        re.compile(r'showAlert\(\s*"Enter a preset name\."\s*\)'),
    ),
    (
        "toast.preset_saved",
        re.compile(r't\(\s*"toast\.preset_saved"'),
        re.compile(r'showAlert\(\s*"Preset saved: "'),
    ),
    (
        "toast.preset_saved_toast",
        re.compile(r't\(\s*"toast\.preset_saved_toast"'),
        re.compile(r'showToast\(\s*"Preset \'"'),
    ),
    (
        "toast.preset_save_failed",
        re.compile(r't\(\s*"toast\.preset_save_failed"'),
        re.compile(r'showAlert\(\s*"Failed to save preset\."\s*\)'),
    ),
    (
        "toast.select_preset_first",
        re.compile(r't\(\s*"toast\.select_preset_first"'),
        re.compile(r'showAlert\(\s*"Select a preset first\."\s*\)'),
    ),
    (
        "toast.preset_loaded",
        re.compile(r't\(\s*"toast\.preset_loaded"'),
        re.compile(r'showAlert\(\s*"Preset loaded: "'),
    ),
    (
        "toast.preset_loaded_toast",
        re.compile(r't\(\s*"toast\.preset_loaded_toast"'),
        re.compile(r'showToast\(\s*"Preset loaded"'),
    ),
    (
        "toast.preset_deleted",
        re.compile(r't\(\s*"toast\.preset_deleted"'),
        re.compile(r'showAlert\(\s*"Preset deleted: "'),
    ),
    (
        "toast.preset_deleted_toast",
        re.compile(r't\(\s*"toast\.preset_deleted_toast"'),
        re.compile(r'showToast\(\s*"Preset deleted"'),
    ),
    (
        "toast.select_preset_export_first",
        re.compile(r't\(\s*"toast\.select_preset_export_first"'),
        re.compile(r'showToast\(\s*"Select a preset to export first"'),
    ),
    # --- Sixteenth batch --------------------------------------------
    (
        "toast.preset_export_load_failed",
        re.compile(r't\(\s*"toast\.preset_export_load_failed"'),
        re.compile(r'showToast\(\s*"Could not load preset for export"'),
    ),
    (
        "toast.preset_exported",
        re.compile(r't\(\s*"toast\.preset_exported"'),
        re.compile(r'showToast\(\s*"Preset exported: "'),
    ),
    (
        "toast.preset_invalid_missing_fields",
        re.compile(r't\(\s*"toast\.preset_invalid_missing_fields"'),
        re.compile(r'showToast\(\s*"Invalid preset file: missing required fields"'),
    ),
    (
        "toast.preset_invalid_settings_object",
        re.compile(r't\(\s*"toast\.preset_invalid_settings_object"'),
        re.compile(r'showToast\(\s*"Invalid preset file: settings must be an object"'),
    ),
    (
        "toast.preset_imported",
        re.compile(r't\(\s*"toast\.preset_imported"'),
        re.compile(r'showToast\(\s*"Preset imported: "'),
    ),
    (
        "toast.preset_import_failed",
        re.compile(r't\(\s*"toast\.preset_import_failed"'),
        re.compile(r'showToast\(\s*"Failed to import preset"'),
    ),
    (
        "toast.preset_invalid_format",
        re.compile(r't\(\s*"toast\.preset_invalid_format"'),
        re.compile(r'showToast\(\s*"Invalid preset file format"'),
    ),
    # --- Seventeenth batch ------------------------------------------
    (
        "toast.select_template_first",
        re.compile(r't\(\s*"toast\.select_template_first"'),
        re.compile(r'showToast\(\s*"Select a template first"'),
    ),
    (
        "toast.template_apply_failed",
        re.compile(r't\(\s*"toast\.template_apply_failed"'),
        re.compile(r'showToast\(\s*"Failed to apply template"'),
    ),
    (
        "toast.template_applied",
        re.compile(r't\(\s*"toast\.template_applied"'),
        re.compile(r'showToast\(\s*"Template applied: "'),
    ),
    (
        "toast.enter_template_name",
        re.compile(r't\(\s*"toast\.enter_template_name"'),
        re.compile(r'showToast\(\s*"Enter a template name"'),
    ),
    (
        "templates.custom_template_description",
        re.compile(r't\(\s*"templates\.custom_template_description"'),
        re.compile(r'description:\s*"Custom template"'),
    ),
    (
        "toast.template_saved",
        re.compile(r't\(\s*"toast\.template_saved"'),
        re.compile(r'showToast\(\s*"Template saved: "'),
    ),
    (
        "toast.template_save_failed",
        re.compile(r't\(\s*"toast\.template_save_failed"'),
        re.compile(r'showToast\(\s*"Failed to save template"'),
    ),
    (
        "templates.select",
        re.compile(r't\(\s*"templates\.select"'),
        re.compile(r'var html = \'<option value="" disabled selected>Select a template'),
    ),
    (
        "templates.builtin_group",
        re.compile(r't\(\s*"templates\.builtin_group"'),
        re.compile(r'html\s*\+=\s*\'<optgroup label="Built-in">\''),
    ),
    (
        "templates.custom_group",
        re.compile(r't\(\s*"templates\.custom_group"'),
        re.compile(r'html\s*\+=\s*\'<optgroup label="Custom">\''),
    ),
    # --- Eighteenth batch -------------------------------------------
    (
        "toast.playhead_sync_unavailable",
        re.compile(r't\(\s*"toast\.playhead_sync_unavailable"'),
        re.compile(r'showToast\(\s*"Playhead sync unavailable: "'),
    ),
    (
        "transcript.summary_topics",
        re.compile(r't\(\s*"transcript\.summary_topics"'),
        re.compile(r'text\s*\+=\s*"\\nTopics: "\s*\+'),
    ),
    (
        "transcript.summary_empty",
        re.compile(r't\(\s*"transcript\.summary_empty"'),
        re.compile(r'textContent\s*=\s*text\s*\|\|\s*"No summary generated\."'),
    ),
    (
        "transcript.timeline_editor",
        re.compile(r't\(\s*"transcript\.timeline_editor"'),
        re.compile(r'<span class="transcript-timeline-stat-label">Editor</span>'),
    ),
    (
        "transcript.timeline_awaiting_segments",
        re.compile(r't\(\s*"transcript\.timeline_awaiting_segments"'),
        re.compile(r'<span class="transcript-timeline-stat-value">Awaiting transcript segments</span>'),
    ),
    (
        "transcript.timeline_segments",
        re.compile(r't\(\s*"transcript\.timeline_segments"'),
        re.compile(r'<span class="transcript-timeline-stat-label">Segments</span>'),
    ),
    (
        "transcript.timeline_runtime",
        re.compile(r't\(\s*"transcript\.timeline_runtime"'),
        re.compile(r'<span class="transcript-timeline-stat-label">Runtime</span>'),
    ),
    (
        "transcript.timeline_pace",
        re.compile(r't\(\s*"transcript\.timeline_pace"'),
        re.compile(r'<span class="transcript-timeline-stat-label">Pace</span>'),
    ),
    (
        "transcript.timeline_avg_seconds",
        re.compile(r't\(\s*"transcript\.timeline_avg_seconds"'),
        re.compile(r"safeFixed\(avgDuration,\s*1\)\s*\+\s*'s avg"),
    ),
    (
        "transcript.timeline_longest",
        re.compile(r't\(\s*"transcript\.timeline_longest"'),
        re.compile(r'<span class="transcript-timeline-stat-label">Longest</span>'),
    ),
    (
        "transcript.timeline_select_segment",
        re.compile(r't\(\s*"transcript\.timeline_select_segment"'),
        re.compile(r'textContent\s*=\s*"Select a segment to focus the edit\."'),
    ),
    (
        "transcript.timeline_segment_status",
        re.compile(r't\(\s*"transcript\.timeline_segment_status"'),
        re.compile(r'textContent\s*=\s*"Segment "\s*\+'),
    ),
    (
        "transcript.timeline_empty",
        re.compile(r't\(\s*"transcript\.timeline_empty"'),
        re.compile(r'message\s*\|\|\s*"Transcript segments will appear here\."'),
    ),
    (
        "transcript.timeline_empty_ready",
        re.compile(r't\(\s*"transcript\.timeline_empty_ready"'),
        re.compile(r'setTranscriptTimelineEmptyState\("Transcript segments will appear here once the clip is ready\."\)'),
    ),
    (
        "transcript.segment_label",
        re.compile(r't\(\s*"transcript\.segment_label"'),
        re.compile(r"preview\s*\|\|\s*\('Segment '\s*\+"),
    ),
    (
        "transcript.segment_time_label",
        re.compile(r't\(\s*"transcript\.segment_time_label"'),
        re.compile(r'aria-label="Segment \'\s*\+\s*\(i \+ 1\)'),
    ),
    (
        "transcript.editor_info",
        re.compile(r't\(\s*"transcript\.editor_info"'),
        re.compile(r'el\.transcriptInfo\.textContent\s*=\s*wordCount\s*\+'),
    ),
    (
        "transcript.editor_empty",
        re.compile(r't\(\s*"transcript\.editor_empty"'),
        re.compile(r'<div class="transcript-empty-state">Transcribe a clip to start shaping dialogue'),
    ),
    (
        "transcript.segment_textarea_label",
        re.compile(r't\(\s*"transcript\.segment_textarea_label"'),
        re.compile(r'aria-label="Transcript segment \'\s*\+\s*\(i \+ 1\)'),
    ),
    # --- Nineteenth batch -------------------------------------------
    (
        "captions.display_preview_placeholder",
        re.compile(r't\(\s*"captions\.display_preview_placeholder"'),
        re.compile(r'\|\|\s*"Caption preview"'),
    ),
    (
        "captions.display_rendering_preview",
        re.compile(r't\(\s*"captions\.display_rendering_preview"'),
        re.compile(r'setCaptionDisplayStatus\("Rendering preview\.\.\."'),
    ),
    (
        "captions.display_sample_text",
        re.compile(r't\(\s*"captions\.display_sample_text"'),
        re.compile(r'sample_text:\s*"The quick brown fox jumps over the lazy dog\."'),
    ),
    (
        "captions.display_preview_failed",
        re.compile(r't\(\s*"captions\.display_preview_failed"'),
        re.compile(r'setCaptionDisplayStatus\("Preview failed: "\s*\+'),
    ),
    (
        "captions.display_preview_updated",
        re.compile(r't\(\s*"captions\.display_preview_updated"'),
        re.compile(r'setCaptionDisplayStatus\("Preview updated\. These display tokens apply to caption burn-in\."'),
    ),
    (
        "captions.display_reset_defaults",
        re.compile(r't\(\s*"captions\.display_reset_defaults"'),
        re.compile(r'setCaptionDisplayStatus\("Reset to FCC defaults\. Click Preview to re-render\."'),
    ),
    (
        "captions.display_loading_tokens",
        re.compile(r't\(\s*"captions\.display_loading_tokens"'),
        re.compile(r'setCaptionDisplayStatus\("Loading tokens\.\.\."'),
    ),
    (
        "captions.display_schema_load_failed",
        re.compile(r't\(\s*"captions\.display_schema_load_failed"'),
        re.compile(r'setCaptionDisplayStatus\("Could not load FCC token schema\."'),
    ),
    (
        "captions.display_defaults_loaded",
        re.compile(r't\(\s*"captions\.display_defaults_loaded"'),
        re.compile(r'setCaptionDisplayStatus\("Defaults loaded\. Adjust tokens then Preview\."'),
    ),
    # --- Twentieth batch --------------------------------------------
    (
        "models.delete_missing_path",
        re.compile(r't\(\s*"models\.delete_missing_path"'),
        re.compile(r'showAlert\("Couldn\'t determine which model to delete\."'),
    ),
    (
        "models.deleting",
        re.compile(r't\(\s*"models\.deleting"'),
        re.compile(r'btn\.textContent\s*=\s*"Deleting'),
    ),
    (
        "models.deleted",
        re.compile(r't\(\s*"models\.deleted"'),
        re.compile(r'showToast\("Model deleted"'),
    ),
    (
        "models.delete",
        re.compile(r't\(\s*"models\.delete"'),
        re.compile(r'(?:deleteBtn|btn)\.textContent\s*=\s*"Delete"'),
    ),
    (
        "models.delete_failed",
        re.compile(r't\(\s*"models\.delete_failed"'),
        re.compile(r'showAlert\("Failed to delete model\."'),
    ),
    (
        "models.scanning_title",
        re.compile(r't\(\s*"models\.scanning_title"'),
        re.compile(r'buildEmptyHintMarkup\(\s*"Scanning local models'),
    ),
    (
        "models.scanning_desc",
        re.compile(r't\(\s*"models\.scanning_desc"'),
        re.compile(r'^\s*"Reviewing local checkpoints and downloaded assets on this machine\.",', re.MULTILINE),
    ),
    (
        "models.scanning_status",
        re.compile(r't\(\s*"models\.scanning_status"'),
        re.compile(r'^\s*"Scanning local models and checkpoints for the current machine\.",', re.MULTILINE),
    ),
    (
        "models.inventory_unavailable_title",
        re.compile(r't\(\s*"models\.inventory_unavailable_title"'),
        re.compile(r'^\s*"Model inventory unavailable",', re.MULTILINE),
    ),
    (
        "models.inventory_unavailable_desc",
        re.compile(r't\(\s*"models\.inventory_unavailable_desc"'),
        re.compile(r'^\s*"Reconnect the backend or refresh again to inspect local model storage\.",', re.MULTILINE),
    ),
    (
        "models.inventory_unavailable_status",
        re.compile(r't\(\s*"models\.inventory_unavailable_status"'),
        re.compile(r'^\s*"Couldn\'t read the local model inventory\. Reconnect the backend or try again\.",', re.MULTILINE),
    ),
    (
        "models.none_found_title",
        re.compile(r't\(\s*"models\.none_found_title"'),
        re.compile(r'^\s*"No local models found",', re.MULTILINE),
    ),
    (
        "models.none_found_desc",
        re.compile(r't\(\s*"models\.none_found_desc"'),
        re.compile(r'^\s*"Add local checkpoints here, or rely on hosted providers for LLM-driven features\.",', re.MULTILINE),
    ),
    (
        "models.none_found_status",
        re.compile(r't\(\s*"models\.none_found_status"'),
        re.compile(r'^\s*"No local models are installed yet\. Hosted providers can still power supported workflows\.",', re.MULTILINE),
    ),
    (
        "models.unknown_model",
        re.compile(r't\(\s*"models\.unknown_model"'),
        re.compile(r'm\.name\s*\|\|\s*"Unknown model"'),
    ),
    (
        "models.unknown_source",
        re.compile(r't\(\s*"models\.unknown_source"'),
        re.compile(r'm\.source\s*\|\|\s*"Unknown source"'),
    ),
    (
        "models.delete_model",
        re.compile(r't\(\s*"models\.delete_model"'),
        re.compile(r'deleteBtn\.title\s*=\s*"Delete model"'),
    ),
    (
        "models.delete_model_aria",
        re.compile(r't\(\s*"models\.delete_model_aria"'),
        re.compile(r'setAttribute\("aria-label",\s*"Delete model "\s*\+'),
    ),
    (
        "models.detected_status",
        re.compile(r't\(\s*"models\.detected_status"'),
        re.compile(r'data\.models\.length\s*\+\s*" local model"'),
    ),
    (
        "gpu.checking",
        re.compile(r't\(\s*"gpu\.checking"'),
        re.compile(r'setButtonText\(el\.getGpuRecBtn,\s*"Checking'),
    ),
    (
        "gpu.recommendation_failed",
        re.compile(r't\(\s*"gpu\.recommendation_failed"'),
        re.compile(r'showAlert\("Failed to get GPU recommendation\."'),
    ),
    (
        "gpu.recommendations_applied",
        re.compile(r't\(\s*"gpu\.recommendations_applied"'),
        re.compile(r'showToast\("GPU recommendations applied"'),
    ),
    (
        "queue.cleared",
        re.compile(r't\(\s*"queue\.cleared"'),
        re.compile(r'showAlert\("Queue cleared: "\s*\+'),
    ),
    (
        "queue.status",
        re.compile(r't\(\s*"queue\.status"'),
        re.compile(r'textContent\s*=\s*"Queue: "\s*\+'),
    ),
    # --- Twenty-first batch -----------------------------------------
    (
        "output.ready_recent",
        re.compile(r't\(\s*"output\.ready_recent"'),
        re.compile(r'\|\|\s*"Ready in recent outputs"'),
    ),
    (
        "output.path_unavailable",
        re.compile(r't\(\s*"output\.path_unavailable"'),
        re.compile(r'pathEl\.textContent\s*=\s*path\s*\|\|\s*"Path unavailable"'),
    ),
    (
        "output.open",
        re.compile(r't\(\s*"output\.open"'),
        re.compile(r'createOutputActionButton\("Open",\s*"output-item-btn",\s*"Open output file"'),
    ),
    (
        "output.open_title",
        re.compile(r't\(\s*"output\.open_title"'),
        re.compile(r'createOutputActionButton\("Open",\s*"output-item-btn",\s*"Open output file"'),
    ),
    (
        "output.reveal",
        re.compile(r't\(\s*"output\.reveal"'),
        re.compile(r'createOutputActionButton\("Reveal",\s*"output-item-btn",\s*"Reveal in file manager"'),
    ),
    (
        "output.reveal_title",
        re.compile(r't\(\s*"output\.reveal_title"'),
        re.compile(r'createOutputActionButton\("Reveal",\s*"output-item-btn",\s*"Reveal in file manager"'),
    ),
    (
        "output.import_to_premiere",
        re.compile(r't\(\s*"output\.import_to_premiere"'),
        re.compile(r'createOutputActionButton\("Import to Premiere"'),
    ),
    (
        "output.import_title",
        re.compile(r't\(\s*"output\.import_title"'),
        re.compile(
            r'createOutputActionButton\("Import to Premiere",'
            r'\s*"output-item-btn output-item-btn-primary",'
            r'\s*"Import into the current Premiere project"'
        ),
    ),
    (
        "output.missing_path",
        re.compile(r't\(\s*"output\.missing_path"'),
        re.compile(r'showToast\("This output is missing a file path\."'),
    ),
    (
        "output.import_unavailable",
        re.compile(r't\(\s*"output\.import_unavailable"'),
        re.compile(r'showToast\("Premiere isn\'t connected right now, so import is unavailable\."'),
    ),
    (
        "output.imported",
        re.compile(r't\(\s*"output\.imported"'),
        re.compile(r'showToast\("Imported "\s*\+\s*outputPath'),
    ),
    (
        "batch.no_files",
        re.compile(r't\(\s*"batch\.no_files"'),
        re.compile(r'buildEmptyHintMarkup\("No files added"'),
    ),
    (
        "batch.add_hint",
        re.compile(r't\(\s*"batch\.add_hint"'),
        re.compile(r'buildEmptyHintMarkup\("No files added",\s*\'Use "Add Selected" or drag files\.\''),
    ),
    (
        "batch.remove",
        re.compile(r't\(\s*"batch\.remove"'),
        re.compile(
            r'class="batch-file-remove"\s+data-idx="\'\s*\+\s*i\s*\+\s*\'">Remove</button>'
        ),
    ),
    # --- Twenty-fifth batch ----------------------------------------
    (
        "batch.starting_status",
        re.compile(r't\(\s*"batch\.starting_status"'),
        re.compile(r'textContent\s*=\s*"Starting batch: "\s*\+\s*paths\.length'),
    ),
    (
        "batch.starting_summary",
        re.compile(r't\(\s*"batch\.starting_summary"'),
        re.compile(r'updateBatchSummary\("Starting batch processing for "\s*\+\s*paths\.length'),
    ),
    (
        "batch.error_status",
        re.compile(r't\(\s*"batch\.error_status"'),
        re.compile(r'textContent\s*=\s*"Batch error: "\s*\+'),
    ),
    (
        "batch.unknown",
        re.compile(r't\(\s*"batch\.unknown"'),
        re.compile(r'\(data\s*&&\s*data\.error\)\s*\|\|\s*"Unknown"'),
    ),
    (
        "batch.unknown_error",
        re.compile(r't\(\s*"batch\.unknown_error"'),
        re.compile(r'Batch couldn\'t start:[\s\S]{0,180}\|\|\s*"Unknown error"'),
    ),
    (
        "batch.start_failed_summary",
        re.compile(r't\(\s*"batch\.start_failed_summary"'),
        re.compile(r'updateBatchSummary\(\s*"Batch couldn\'t start: "\s*\+'),
    ),
    (
        "batch.running_status",
        re.compile(r't\(\s*"batch\.running_status"'),
        re.compile(r'textContent\s*=\s*"Batch running: 0/"\s*\+\s*data\.total'),
    ),
    (
        "batch.running_summary",
        re.compile(r't\(\s*"batch\.running_summary"'),
        re.compile(r'updateBatchSummary\("Batch is running across "\s*\+\s*data\.total'),
    ),
    (
        "batch.poll_failed_status",
        re.compile(r't\(\s*"batch\.poll_failed_status"'),
        re.compile(r'textContent\s*=\s*"Batch poll failed after 10 errors"'),
    ),
    (
        "batch.poll_failed_summary",
        re.compile(r't\(\s*"batch\.poll_failed_summary"'),
        re.compile(r'updateBatchSummary\(\s*"Batch status polling failed repeatedly\.'),
    ),
    (
        "batch.progress_status",
        re.compile(r't\(\s*"batch\.progress_status"'),
        re.compile(r'textContent\s*=\s*\r?\n?\s*"Batch "\s*\+\s*d2\.status'),
    ),
    (
        "batch.processing_summary",
        re.compile(r't\(\s*"batch\.processing_summary"'),
        re.compile(r'\?\s*"Batch is processing "\s*\+'),
    ),
    (
        "batch.finished_summary",
        re.compile(r't\(\s*"batch\.finished_summary"'),
        re.compile(r':\s*"Batch finished: "\s*\+'),
    ),
    (
        "batch.complete",
        re.compile(r't\(\s*"batch\.complete"'),
        re.compile(r'showAlert\("Batch complete: "\s*\+'),
    ),
    # --- Twenty-sixth batch ----------------------------------------
    (
        "batch.choose_operation",
        re.compile(r't\(\s*"batch\.choose_operation"'),
        re.compile(r'getSelectOptionLabel\(el\.batchOperation,\s*"Choose an operation"\)'),
    ),
    (
        "batch.selected_operation",
        re.compile(r't\(\s*"batch\.selected_operation"'),
        re.compile(r'\|\|\s*"the selected operation"'),
    ),
    (
        "batch.queue_count",
        re.compile(r't\(\s*"batch\.queue_count"'),
        re.compile(r'queueLabel\s*=\s*queuedCount\s*\+\s*" clip"'),
    ),
    (
        "batch.queue_count_title",
        re.compile(r't\(\s*"batch\.queue_count_title"'),
        re.compile(r'queueTitle\s*=\s*queueLabel\s*\+\s*" for the next batch run\."'),
    ),
    (
        "batch.queue_available",
        re.compile(r't\(\s*"batch\.queue_available"'),
        re.compile(r'queueLabel\s*=\s*"0 queued • "\s*\+\s*availableCount'),
    ),
    (
        "batch.queue_available_title",
        re.compile(r't\(\s*"batch\.queue_available_title"'),
        re.compile(r'queueTitle\s*=\s*availableCount\s*\+\s*" project clip"'),
    ),
    (
        "batch.queue_empty",
        re.compile(r't\(\s*"batch\.queue_empty"'),
        re.compile(r'queueLabel\s*=\s*"No clips queued"'),
    ),
    (
        "batch.queue_empty_title",
        re.compile(r't\(\s*"batch\.queue_empty_title"'),
        re.compile(r'queueTitle\s*=\s*"Load clips into the project, then add the ones you want'),
    ),
    (
        "batch.operation_ready_title",
        re.compile(r't\(\s*"batch\.operation_ready_title"'),
        re.compile(r'opLabel\s*\?\s*opLabel\s*\+\s*" will run across the queued clips\."'),
    ),
    (
        "batch.operation_choose_title",
        re.compile(r't\(\s*"batch\.operation_choose_title"'),
        re.compile(r':\s*"Choose the process you want to apply across the queue\."'),
    ),
    (
        "batch.status_reconnect",
        re.compile(r't\(\s*"batch\.status_reconnect"'),
        re.compile(r'setStatusLine\(\s*"batchStatusLine",\s*"Reconnect the backend before running'),
    ),
    (
        "batch.status_load_clips",
        re.compile(r't\(\s*"batch\.status_load_clips"'),
        re.compile(r'setStatusLine\(\s*"batchStatusLine",\s*"Load clips into the project, then add two'),
    ),
    (
        "batch.status_add_clips",
        re.compile(r't\(\s*"batch\.status_add_clips"'),
        re.compile(r'"Add clips to the queue, then run "\s*\+\s*\(opLabel\s*\|\|\s*"the selected operation"\)'),
    ),
    (
        "batch.status_add_one_more",
        re.compile(r't\(\s*"batch\.status_add_one_more"'),
        re.compile(r'"Add one more clip to enable batch processing for "\s*\+\s*\(opLabel\s*\|\|\s*"the selected operation"\)'),
    ),
    (
        "batch.status_ready",
        re.compile(r't\(\s*"batch\.status_ready"'),
        re.compile(r'"Batch is ready to run "\s*\+\s*\(opLabel\s*\|\|\s*"the selected operation"\)'),
    ),
    # --- Twenty-seventh batch --------------------------------------
    (
        "workspace.backend_offline",
        re.compile(r't\(\s*"workspace\.backend_offline"'),
        re.compile(r'stageKicker\s*=\s*"Backend Offline"'),
    ),
    (
        "workspace.reconnect_title",
        re.compile(r't\(\s*"workspace\.reconnect_title"'),
        re.compile(r'stageTitle\s*=\s*"Reconnect OpenCut to run processing jobs\."'),
    ),
    (
        "workspace.reconnect_copy",
        re.compile(r't\(\s*"workspace\.reconnect_copy"'),
        re.compile(r'stageCopy\s*=\s*"The workspace is still available, but processing'),
    ),
    (
        "workspace.awaiting_media",
        re.compile(r't\(\s*"workspace\.awaiting_media"'),
        re.compile(r'workspaceStageSource\.textContent\s*=\s*selectedName\s*\|\|\s*"Awaiting media"'),
    ),
    (
        "workspace.choose_clip_title",
        re.compile(r't\(\s*"workspace\.choose_clip_title"'),
        re.compile(r'workspaceStageSource\.title\s*=\s*selectedPath\s*\|\|\s*"Choose a clip or drop media to start"'),
    ),
    (
        "workspace.default_suite",
        re.compile(r't\(\s*"workspace\.default_suite"'),
        re.compile(
            r'workspaceStageSuite\.textContent\s*=\s*activeTitle\s*\|\|\s*'
            r'\(el\.contentTitle\s*\?\s*el\.contentTitle\.textContent\s*:\s*"Cut & Clean"\)'
        ),
    ),
    (
        "workspace.status_reconnect",
        re.compile(r't\(\s*"workspace\.status_reconnect"'),
        re.compile(r'workspaceStageStatus\.textContent\s*=\s*"Reconnect backend"'),
    ),
    (
        "workspace.status_reconnect_title",
        re.compile(r't\(\s*"workspace\.status_reconnect_title"'),
        re.compile(r'workspaceStageStatus\.title\s*=\s*"Start or reconnect the local OpenCut backend service"'),
    ),
    (
        "workspace.status_source_ready",
        re.compile(r't\(\s*"workspace\.status_source_ready"'),
        re.compile(r'workspaceStageStatus\.textContent\s*=\s*"Source ready"'),
    ),
    (
        "workspace.status_source_ready_title",
        re.compile(r't\(\s*"workspace\.status_source_ready_title"'),
        re.compile(r'workspaceStageStatus\.title\s*=\s*"The active source is selected and ready for processing"'),
    ),
    (
        "workspace.status_settings_ready",
        re.compile(r't\(\s*"workspace\.status_settings_ready"'),
        re.compile(r'workspaceStageStatus\.textContent\s*=\s*"Settings ready"'),
    ),
    (
        "workspace.status_settings_ready_title",
        re.compile(r't\(\s*"workspace\.status_settings_ready_title"'),
        re.compile(r'workspaceStageStatus\.title\s*=\s*"Settings does not require a source clip"'),
    ),
    (
        "workspace.status_select_media",
        re.compile(r't\(\s*"workspace\.status_select_media"'),
        re.compile(r'workspaceStageStatus\.textContent\s*=\s*"Select media"'),
    ),
    (
        "workspace.status_select_media_title",
        re.compile(r't\(\s*"workspace\.status_select_media_title"'),
        re.compile(r'workspaceStageStatus\.title\s*=\s*"Select a clip from Premiere or browse'),
    ),
    (
        "workspace.clip_choose",
        re.compile(r't\(\s*"workspace\.clip_choose"'),
        re.compile(r'workspaceClipStatus\.textContent\s*=\s*"Choose media to begin"'),
    ),
    # --- Twenty-eighth batch ---------------------------------------
    (
        "conn.connected",
        re.compile(r't\(\s*"conn\.connected"'),
        re.compile(r'setConnectionBadge\("online",\s*"Connected"\s*\)'),
    ),
    (
        "conn.connected_port",
        re.compile(r't\(\s*"conn\.connected_port"'),
        re.compile(r'setConnectionBadge\("online",\s*"Connected"\s*\+\s*\(port'),
    ),
    (
        "conn.connected_gpu",
        re.compile(r't\(\s*"conn\.connected_gpu"'),
        re.compile(r'setConnectionBadge\("online",\s*"Connected \("\s*\+\s*data\.gpu_name'),
    ),
    (
        "conn.disconnected",
        re.compile(r't\(\s*"conn\.disconnected"'),
        re.compile(r'setConnectionBadge\("offline",\s*"Disconnected"\s*\)'),
    ),
    (
        "conn.dot_connected",
        re.compile(r't\(\s*"conn\.dot_connected"'),
        re.compile(r'state\s*===\s*"online"\s*\?\s*"Server connected"'),
    ),
    (
        "conn.dot_disconnected",
        re.compile(r't\(\s*"conn\.dot_disconnected"'),
        re.compile(r':\s*"Server disconnected"\s*\)'),
    ),
    (
        "conn.port",
        re.compile(r't\(\s*"conn\.port"'),
        re.compile(r'textContent\s*=\s*"Port "\s*\+\s*port'),
    ),
    (
        "conn.reconnecting",
        re.compile(r't\(\s*"conn\.reconnecting"'),
        re.compile(r'serverStatusMsg\.textContent\s*=\s*"Server disconnected\. Reconnecting'),
    ),
    (
        "media.scanning_premiere",
        re.compile(r't\(\s*"media\.scanning_premiere"'),
        re.compile(r'setProjectMediaPlaceholder\("Scanning Premiere project media'),
    ),
    (
        "media.load_failed",
        re.compile(r't\(\s*"media\.load_failed"'),
        re.compile(r'setProjectMediaPlaceholder\("Couldn\'t load Premiere project media"'),
    ),
    (
        "media.no_importable",
        re.compile(r't\(\s*"media\.no_importable"'),
        re.compile(r'setProjectMediaPlaceholder\("No importable project media found"'),
    ),
    (
        "media.select_clip_placeholder",
        re.compile(r't\(\s*"media\.select_clip_placeholder"'),
        re.compile(r'placeholder\.textContent\s*=\s*files\.length\s*\?\s*"-- Select a clip --"'),
    ),
    (
        "media.no_project_media",
        re.compile(r't\(\s*"media\.no_project_media"'),
        re.compile(r':\s*"No project media found"'),
    ),
    (
        "media.untitled_clip",
        re.compile(r't\(\s*"media\.untitled_clip"'),
        re.compile(r':\s*"Untitled clip"\)'),
    ),
    (
        "media.read_failed",
        re.compile(r't\(\s*"media\.read_failed"'),
        re.compile(r'showAlert\("Couldn\'t read project media\. Make sure a project is open'),
    ),
    # --- Thirtieth batch -------------------------------------------
    (
        "progress.busy",
        re.compile(r't\(\s*"progress\.busy"'),
        re.compile(r'showAlert\("OpenCut is already processing another task\.'),
    ),
    (
        "progress.step_prefix",
        re.compile(r't\(\s*"progress\.step_prefix"'),
        re.compile(r'"Step "\s*\+\s*jobStepCurrent\s*\+\s*"/"\s*\+\s*jobStepTotal'),
    ),
    (
        "progress.preparing",
        re.compile(r't\(\s*"progress\.preparing"'),
        re.compile(r'textContent\s*=\s*stepPrefix\s*\+\s*"Preparing run'),
    ),
    (
        "progress.start_failed",
        re.compile(r't\(\s*"progress\.start_failed"'),
        re.compile(r'data\s*\?\s*data\.error\s*:\s*"Failed to start job"'),
    ),
    (
        "progress.start_failed_prefix",
        re.compile(r't\(\s*"progress\.start_failed_prefix"'),
        re.compile(r'showAlert\("Failed to start job: "\s*\+\s*e\.message'),
    ),
    (
        "progress.processing",
        re.compile(r't\(\s*"progress\.processing"'),
        re.compile(r'job\.message\s*\|\|\s*"Processing'),
    ),
    (
        "progress.run_failed",
        re.compile(r't\(\s*"progress\.run_failed"'),
        re.compile(r'resultsTitle\.textContent\s*=\s*"Run failed"'),
    ),
    (
        "progress.unknown_error",
        re.compile(r't\(\s*"progress\.unknown_error"'),
        re.compile(
            r'resultsStats\.textContent\s*=\s*enhanceError\(\s*'
            r'job\.error\s*\|\|\s*job\.message\s*\|\|\s*"Unknown error"'
        ),
    ),
    (
        "progress.finished",
        re.compile(r't\(\s*"progress\.finished"'),
        re.compile(r'resultsTitle\.textContent\s*=\s*"Finished"'),
    ),
    (
        "progress.success_summary",
        re.compile(r't\(\s*"progress\.success_summary"'),
        re.compile(r'stats\s*\|\|\s*"The run finished successfully\."'),
    ),
    # --- Thirty-first batch ----------------------------------------
    (
        "error.route_not_found",
        re.compile(r't\(\s*"error\.route_not_found"'),
        re.compile(r'return\s+"OpenCut couldn\'t find the local route'),
    ),
    (
        "error.backend_problem",
        re.compile(r't\(\s*"error\.backend_problem"'),
        re.compile(r'return\s+"OpenCut hit a local backend problem'),
    ),
    (
        "error.request_retry",
        re.compile(r't\(\s*"error\.request_retry"'),
        re.compile(r'return\s+"OpenCut couldn\'t complete that request'),
    ),
    (
        "error.backend_unreachable",
        re.compile(r't\(\s*"error\.backend_unreachable"'),
        re.compile(r'return\s+"OpenCut couldn\'t reach the local backend'),
    ),
    (
        "progress.busy",
        re.compile(r't\(\s*"progress\.busy"'),
        re.compile(r'return\s+"OpenCut is already processing another task\.'),
    ),
    (
        "toast.choose_source_first",
        re.compile(r't\(\s*"toast\.choose_source_first"'),
        re.compile(r'return\s+"Choose a source in Media before running this tool\."'),
    ),
    (
        "error.source_missing",
        re.compile(r't\(\s*"error\.source_missing"'),
        re.compile(r'return\s+"OpenCut couldn\'t find that source anymore'),
    ),
    (
        "error.file_permission",
        re.compile(r't\(\s*"error\.file_permission"'),
        re.compile(r'return\s+"OpenCut doesn\'t have permission to read or write'),
    ),
    (
        "error.timeout_retry",
        re.compile(r't\(\s*"error\.timeout_retry"'),
        re.compile(r'return\s+"That run took too long to finish'),
    ),
    (
        "error.install_from_settings",
        re.compile(r't\(\s*"error\.install_from_settings"'),
        re.compile(r'return\s+normalizedMsg\s*\+\s*" \\u2014 You can install'),
    ),
    (
        "error.memory_retry",
        re.compile(r't\(\s*"error\.memory_retry"'),
        re.compile(r'return\s+normalizedMsg\s*\+\s*" \\u2014 Try a smaller file'),
    ),
    (
        "error.permission_check",
        re.compile(r't\(\s*"error\.permission_check"'),
        re.compile(r'return\s+normalizedMsg\s*\+\s*" \\u2014 The file may be locked'),
    ),
    (
        "error.file_moved",
        re.compile(r't\(\s*"error\.file_moved"'),
        re.compile(r'return\s+normalizedMsg\s*\+\s*" \\u2014 The file may have been moved'),
    ),
    (
        "error.operation_timeout",
        re.compile(r't\(\s*"error\.operation_timeout"'),
        re.compile(r'return\s+normalizedMsg\s*\+\s*" \\u2014 The operation took too long'),
    ),
    (
        "error.server_running",
        re.compile(r't\(\s*"error\.server_running"'),
        re.compile(r'return\s+normalizedMsg\s*\+\s*" \\u2014 Make sure the OpenCut server is running'),
    ),
    # --- Thirty-second batch ---------------------------------------
    (
        "alert.open_target",
        re.compile(r't\(\s*"alert\.open_target"'),
        re.compile(r'link\.textContent\s*=\s*"Open "\s*\+'),
    ),
    (
        "common.working",
        re.compile(r't\(\s*"common\.working"'),
        re.compile(r'setButtonText\(btn,\s*"Working'),
    ),
    (
        "install.default_start",
        re.compile(r't\(\s*"install\.default_start"'),
        re.compile(r'config\.startMessage\s*\|\|\s*"Installing'),
    ),
    (
        "install.cancelled",
        re.compile(r't\(\s*"install\.cancelled"'),
        re.compile(r'error\s*=\s*"Cancelled"'),
    ),
    (
        "install.failed",
        re.compile(r't\(\s*"install\.failed"'),
        re.compile(r'"Installation failed: "\s*\+'),
    ),
    (
        "install.demucs_start",
        re.compile(r't\(\s*"install\.demucs_start"'),
        re.compile(r'setHintState\(el\.separateHint,\s*"Installing Demucs'),
    ),
    (
        "install.depth_start",
        re.compile(r't\(\s*"install\.depth_start"'),
        re.compile(r'startMessage:\s*"Installing Depth Anything'),
    ),
    (
        "install.emotion_start",
        re.compile(r't\(\s*"install\.emotion_start"'),
        re.compile(r'startMessage:\s*"Installing emotion analysis'),
    ),
    (
        "install.crisper_whisper_start",
        re.compile(r't\(\s*"install\.crisper_whisper_start"'),
        re.compile(r'startMessage:\s*"Installing CrisperWhisper'),
    ),
    (
        "install.broll_generation_start",
        re.compile(r't\(\s*"install\.broll_generation_start"'),
        re.compile(r'startMessage:\s*"Installing AI B-roll generation'),
    ),
    (
        "install.multimodal_diarization_start",
        re.compile(r't\(\s*"install\.multimodal_diarization_start"'),
        re.compile(r'startMessage:\s*"Installing multimodal diarization'),
    ),
    (
        "install.watermark_start",
        re.compile(r't\(\s*"install\.watermark_start"'),
        re.compile(r'setHintState\(el\.watermarkHint,\s*"Installing watermark remover'),
    ),
    (
        "error.gpu_out_of_memory",
        re.compile(r't\(\s*"error\.gpu_out_of_memory"'),
        re.compile(r'msg:\s*"GPU ran out of memory'),
    ),
    (
        "error.file_not_found_reselect",
        re.compile(r't\(\s*"error\.file_not_found_reselect"'),
        re.compile(r'msg:\s*"File not found\. Re-select your clip\.'),
    ),
    (
        "error.too_many_jobs",
        re.compile(r't\(\s*"error\.too_many_jobs"'),
        re.compile(r'msg:\s*"Too many jobs running'),
    ),
    # --- Thirty-third batch -----------------------------------------
    (
        "alert.issue_bundle_url",
        re.compile(r't\(\s*"alert\.issue_bundle_url"'),
        re.compile(r'showAlert\(\s*"Issue bundle URL'),
    ),
    (
        "alert.no_demo_footage",
        re.compile(r't\(\s*"alert\.no_demo_footage"'),
        re.compile(r'showAlert\(\s*"No demo footage found'),
    ),
    (
        "audio.measuring",
        re.compile(r't\(\s*"audio\.measuring"'),
        re.compile(r'meterLUFS\.textContent\s*=\s*"Measuring'),
    ),
    (
        "common.auto",
        re.compile(r't\(\s*"common\.auto"'),
        re.compile(r'data\.method\s*\|\|\s*"auto"'),
    ),
    (
        "interview.server_disconnected",
        re.compile(r't\(\s*"interview\.server_disconnected"'),
        re.compile(r'interviewPolishHint\.textContent\s*=\s*"Server disconnected'),
    ),
    (
        "interview.select_clip_to_run",
        re.compile(r't\(\s*"interview\.select_clip_to_run"'),
        re.compile(r'interviewPolishHint\.textContent\s*=\s*"Select a clip to run'),
    ),
    (
        "interview.runs_on",
        re.compile(r't\(\s*"interview\.runs_on"'),
        re.compile(r'interviewPolishHint\.textContent\s*=\s*"Runs on'),
    ),
    (
        "social.view_on",
        re.compile(r't\(\s*"social\.view_on"'),
        re.compile(r'textContent\s*=\s*"View on "\s*\+'),
    ),
    (
        "social.connected_as",
        re.compile(r't\(\s*"social\.connected_as"'),
        re.compile(r'textContent\s*=\s*"Connected as "\s*\+'),
    ),
    (
        "social.uploaded_to",
        re.compile(r't\(\s*"social\.uploaded_to"'),
        re.compile(r'showToast\(\s*"Uploaded to "\s*\+'),
    ),
    (
        "toast.bridgetalk_async_ready",
        re.compile(r't\(\s*"toast\.bridgetalk_async_ready"'),
        re.compile(r'showToast\(\s*"BridgeTalk async ready"'),
    ),
    (
        "toast.demo_fetch_failed",
        re.compile(r't\(\s*"toast\.demo_fetch_failed"'),
        re.compile(r'showToast\(\s*"Demo fetch failed"'),
    ),
    (
        "toast.demo_loaded",
        re.compile(r't\(\s*"toast\.demo_loaded"'),
        re.compile(r'showToast\(\s*"Loaded demo footage'),
    ),
    (
        "toast.issue_report_opened",
        re.compile(r't\(\s*"toast\.issue_report_opened"'),
        re.compile(r'showToast\(\s*"Issue report opened'),
    ),
    (
        "video.realesrgan_not_installed",
        re.compile(r't\(\s*"video\.realesrgan_not_installed"'),
        re.compile(r'textContent\s*=\s*"Real-ESRGAN not installed'),
    ),
    (
        "video.rembg_not_installed",
        re.compile(r't\(\s*"video\.rembg_not_installed"'),
        re.compile(r'textContent\s*=\s*"rembg not installed'),
    ),
    (
        "video.watermark_detected_region",
        re.compile(r't\(\s*"video\.watermark_detected_region"'),
        re.compile(r'setHintState\(resEl,\s*"Detected at'),
    ),
    (
        "video.watermark_not_detected",
        re.compile(r't\(\s*"video\.watermark_not_detected"'),
        re.compile(r'setHintState\(resEl,\s*"No watermark detected'),
    ),
    # --- Thirty-fourth batch ----------------------------------------
    (
        "silence.detect",
        re.compile(r't\(\s*"silence\.detect"'),
        re.compile(r'"silence-detect"[^\n]*label:\s*"Detect Silence"'),
    ),
    (
        "shortcuts.captions",
        re.compile(r't\(\s*"shortcuts\.captions"'),
        re.compile(r'"caption-generate"[^\n]*label:\s*"Generate Captions"'),
    ),
    (
        "shortcuts.normalize",
        re.compile(r't\(\s*"shortcuts\.normalize"'),
        re.compile(r'"audio-normalize"[^\n]*label:\s*"Normalize Audio"'),
    ),
    (
        "shortcuts.denoise",
        re.compile(r't\(\s*"shortcuts\.denoise"'),
        re.compile(r'"audio-denoise"[^\n]*label:\s*"Denoise Audio"'),
    ),
    (
        "shortcuts.export",
        re.compile(r't\(\s*"shortcuts\.export"'),
        re.compile(r'"export-video"[^\n]*label:\s*"Export Video"'),
    ),
    (
        "shortcuts.command_palette",
        re.compile(r't\(\s*"shortcuts\.command_palette"'),
        re.compile(r'"command-palette"[^\n]*label:\s*"Command Palette"'),
    ),
    (
        "shortcuts.cancel",
        re.compile(r't\(\s*"shortcuts\.cancel"'),
        re.compile(r'"cancel-job"[^\n]*label:\s*"Cancel Current Job"'),
    ),
    (
        "shortcuts.workflow",
        re.compile(r't\(\s*"shortcuts\.workflow"'),
        re.compile(r'"quick-workflow"[^\n]*label:\s*"Run Quick Workflow"'),
    ),
    (
        "common.value",
        re.compile(r't\(\s*"common\.value"'),
        re.compile(r'normalized\.label\s*\|\|\s*"Value"'),
    ),
    (
        "forms.highlight_duration",
        re.compile(r't\(\s*"forms\.highlight_duration"'),
        re.compile(r'label:\s*"Highlight duration"'),
    ),
    (
        "forms.short_duration",
        re.compile(r't\(\s*"forms\.short_duration"'),
        re.compile(r'label:\s*"Short duration"'),
    ),
    (
        "toast.number_adjusted",
        re.compile(r't\(\s*"toast\.number_adjusted"'),
        re.compile(r'showToast\(\s*\(normalized\.label'),
    ),
    (
        "toast.number_kept_in_range",
        re.compile(r't\(\s*"toast\.number_kept_in_range"'),
        re.compile(r'showToast\(\s*paired\.label\s*\+'),
    ),
    # --- Thirty-fifth batch -----------------------------------------
    (
        "status.uptime",
        re.compile(r't\(\s*"status\.uptime"'),
        re.compile(r'text\.textContent\s*=\s*"Up "\s*\+'),
    ),
    (
        "status.cpu_usage",
        re.compile(r't\(\s*"status\.cpu_usage"'),
        re.compile(r'text\.textContent\s*\+=\s*" \\u00B7 CPU "\s*\+'),
    ),
    (
        "status.ram_usage",
        re.compile(r't\(\s*"status\.ram_usage"'),
        re.compile(r'text\.textContent\s*\+=\s*" \\u00B7 RAM "\s*\+'),
    ),
    (
        "status.gpu_unavailable",
        re.compile(r't\(\s*"status\.gpu_unavailable"'),
        re.compile(r'gpu\.textContent\s*=\s*"GPU: N/A"'),
    ),
    (
        "status.jobs_running",
        re.compile(r't\(\s*"status\.jobs_running"'),
        re.compile(r'parts\.push\(j\.running\s*\+\s*" running"'),
    ),
    (
        "status.jobs_queued",
        re.compile(r't\(\s*"status\.jobs_queued"'),
        re.compile(r'parts\.push\(j\.queued\s*\+\s*" queued"'),
    ),
    (
        "status.jobs_done_today",
        re.compile(r't\(\s*"status\.jobs_done_today"'),
        re.compile(r'parts\.push\(j\.completed_today\s*\+\s*" done today"'),
    ),
    (
        "status.jobs_summary",
        re.compile(r't\(\s*"status\.jobs_summary"'),
        re.compile(r'jobsEl\.textContent\s*=\s*parts\.length\s*\?\s*"Jobs: "\s*\+'),
    ),
    (
        "status.jobs_none",
        re.compile(r't\(\s*"status\.jobs_none"'),
        re.compile(r':\s*"Jobs: 0"'),
    ),
    (
        "toast.language_unavailable",
        re.compile(r't\(\s*"toast\.language_unavailable"'),
        re.compile(r'showToast\(\s*"Language \'"'),
    ),
    # --- Twenty-second batch ----------------------------------------
    (
        "workflow.step_count",
        re.compile(r't\(\s*"workflow\.step_count"'),
        re.compile(r'return\s+count\s*\+\s*" step"'),
    ),
    (
        "workflow.enter_name",
        re.compile(r't\(\s*"workflow\.enter_name"'),
        re.compile(r'showToast\("Enter a workflow name"'),
    ),
    (
        "workflow.add_step_first",
        re.compile(r't\(\s*"workflow\.add_step_first"'),
        re.compile(r'showToast\("Add at least one step"'),
    ),
    (
        "workflow.save_failed",
        re.compile(r't\(\s*"workflow\.save_failed"'),
        re.compile(r'data\s*\?\s*data\.error\s*:\s*"Save failed"'),
    ),
    (
        "workflow.saved",
        re.compile(r't\(\s*"workflow\.saved"'),
        re.compile(r'showToast\("Workflow saved: "\s*\+\s*name'),
    ),
    (
        "workflow.saved_summary",
        re.compile(r't\(\s*"workflow\.saved_summary"'),
        re.compile(r'updateCustomWorkflowSummary\("Saved "\s*\+\s*name'),
    ),
    (
        "workflow.loaded_summary",
        re.compile(r't\(\s*"workflow\.loaded_summary"'),
        re.compile(r'updateCustomWorkflowSummary\(\s*"Loaded "\s*\+\s*data\[i\]\.name'),
    ),
    (
        "workflow.deleted",
        re.compile(r't\(\s*"workflow\.deleted"'),
        re.compile(r'showToast\("Workflow deleted"'),
    ),
    (
        "workflow.deleted_summary",
        re.compile(r't\(\s*"workflow\.deleted_summary"'),
        re.compile(r'updateCustomWorkflowSummary\("Deleted "\s*\+\s*sel\.value'),
    ),
    (
        "workflow.custom_default",
        re.compile(r't\(\s*"workflow\.custom_default"'),
        re.compile(r'\|\|\s*"Custom workflow"'),
    ),
    (
        "workflow.running_on",
        re.compile(r't\(\s*"workflow\.running_on"'),
        re.compile(r'updateCustomWorkflowSummary\(\s*"Running "\s*\+\s*draftName'),
    ),
    (
        "workflow.empty_title",
        re.compile(r't\(\s*"workflow\.empty_title"'),
        re.compile(r'buildEmptyHintMarkup\("Workflow is empty"'),
    ),
    (
        "workflow.empty_hint",
        re.compile(r't\(\s*"workflow\.empty_hint"'),
        re.compile(r'buildEmptyHintMarkup\("Workflow is empty",\s*"Add steps to build a custom workflow\."'),
    ),
    (
        "workflow.remove",
        re.compile(r't\(\s*"workflow\.remove"'),
        re.compile(
            r'class="workflow-step-remove"\s+data-idx="\'\s*\+\s*i\s*\+\s*\'">Remove</button>'
        ),
    ),
    (
        "workflow.saved_unavailable",
        re.compile(r't\(\s*"workflow\.saved_unavailable"'),
        re.compile(r"innerHTML\s*=\s*'<option value=\"\" disabled selected>Saved workflows unavailable</option>'"),
    ),
    (
        "workflow.load_saved_failed",
        re.compile(r't\(\s*"workflow\.load_saved_failed"'),
        re.compile(r'updateCustomWorkflowSummary\(\s*"Couldn\'t load saved workflows\.'),
    ),
    (
        "workflow.no_custom_workflows",
        re.compile(r't\(\s*"workflow\.no_custom_workflows"'),
        re.compile(r"innerHTML\s*=\s*'<option value=\"\" disabled selected>No custom workflows</option>'"),
    ),
    # --- Twenty-third batch -----------------------------------------
    (
        "workflow.preset_unavailable",
        re.compile(r't\(\s*"workflow\.preset_unavailable"'),
        re.compile(r"innerHTML\s*=\s*'<option value=\"\" disabled selected>Preset library unavailable</option>'"),
    ),
    (
        "workflow.load_presets_failed",
        re.compile(r't\(\s*"workflow\.load_presets_failed"'),
        re.compile(r'updateWorkflowPresetSummary\(\s*"Couldn\'t load workflow presets\.'),
    ),
    (
        "workflow.builtin_group",
        re.compile(r't\(\s*"workflow\.builtin_group"'),
        re.compile(r'optg\.label\s*=\s*"Built-in"'),
    ),
    (
        "workflow.custom_group",
        re.compile(r't\(\s*"workflow\.custom_group"'),
        re.compile(r'optg2\.label\s*=\s*"Custom"'),
    ),
    (
        "workflow.no_presets",
        re.compile(r't\(\s*"workflow\.no_presets"'),
        re.compile(r"innerHTML\s*=\s*'<option value=\"\" disabled selected>No presets available</option>'"),
    ),
    (
        "workflow.preset_running_on",
        re.compile(r't\(\s*"workflow\.preset_running_on"'),
        re.compile(r'updateWorkflowPresetSummary\(\s*"Running "\s*\+\s*preset\.name'),
    ),
    (
        "workflow.complete",
        re.compile(r't\(\s*"workflow\.complete"'),
        re.compile(r'var\s+msg\s*=\s*"Workflow complete: "\s*\+'),
    ),
    (
        "workflow.complete_output_suffix",
        re.compile(r't\(\s*"workflow\.complete_output_suffix"'),
        re.compile(r'msg\s*\+=\s*" Output: "\s*\+'),
    ),
    (
        "workflow.failed",
        re.compile(r't\(\s*"workflow\.failed"'),
        re.compile(r'var\s+errorMsg\s*=\s*"Workflow failed: "\s*\+'),
    ),
    (
        "workflow.unknown_error",
        re.compile(r't\(\s*"workflow\.unknown_error"'),
        re.compile(
            r'var\s+errorMsg\s*=\s*"Workflow failed: "\s*\+\s*'
            r'\(job\.error\s*\|\|\s*job\.message\s*\|\|\s*"Unknown error"\)'
        ),
    ),
    (
        "workflow.cancelled",
        re.compile(r't\(\s*"workflow\.cancelled"'),
        re.compile(r'var\s+cancelMsg\s*=\s*"Workflow cancelled before all steps finished\."'),
    ),
    # --- Twenty-fourth batch ---------------------------------------
    (
        "workflow.preset_desc_choose",
        re.compile(r't\(\s*"workflow\.preset_desc_choose"'),
        re.compile(r'setHintState\(\s*el\.workflowPresetDesc,\s*"Choose a preset to preview'),
    ),
    (
        "workflow.preset_desc_steps",
        re.compile(r't\(\s*"workflow\.preset_desc_steps"'),
        re.compile(
            r'workflowStepCountLabel\(\(preset\.steps \|\| \[\]\)\.length\)\s*\+\s*" in sequence\."'
        ),
    ),
    (
        "workflow.preset_loading_pill",
        re.compile(r't\(\s*"workflow\.preset_loading_pill"'),
        re.compile(r'setStatusPill\("workflowPresetPill",\s*"Loading\.\.\."'),
    ),
    (
        "workflow.preset_loading_title",
        re.compile(r't\(\s*"workflow\.preset_loading_title"'),
        re.compile(r'summaryTitle\s*=\s*"Loading workflow presets\."'),
    ),
    (
        "workflow.preset_loading_summary",
        re.compile(r't\(\s*"workflow\.preset_loading_summary"'),
        re.compile(r'summaryLabel\s*=\s*"Checking built-in and custom workflow presets\.\.\."'),
    ),
    (
        "workflow.preset_empty_pill",
        re.compile(r't\(\s*"workflow\.preset_empty_pill"'),
        re.compile(r'setStatusPill\("workflowPresetPill",\s*"Empty"'),
    ),
    (
        "workflow.preset_empty_title",
        re.compile(r't\(\s*"workflow\.preset_empty_title"'),
        re.compile(
            r'setStatusPill\("workflowPresetPill",\s*"Empty",\s*"warning",\s*"No built-in or custom presets'
        ),
    ),
    (
        "workflow.preset_empty_summary",
        re.compile(r't\(\s*"workflow\.preset_empty_summary"'),
        re.compile(r'summaryLabel\s*=\s*"No workflow presets available"'),
    ),
    (
        "workflow.preset_empty_hint",
        re.compile(r't\(\s*"workflow\.preset_empty_hint"'),
        re.compile(r'summaryTitle\s*=\s*"Save a custom workflow or refresh the preset library\."'),
    ),
    (
        "workflow.preset_choose_pill",
        re.compile(r't\(\s*"workflow\.preset_choose_pill"'),
        re.compile(r'setStatusPill\("workflowPresetPill",\s*"Choose one"'),
    ),
    (
        "workflow.preset_choose_title",
        re.compile(r't\(\s*"workflow\.preset_choose_title"'),
        re.compile(
            r'setStatusPill\("workflowPresetPill",\s*"Choose one",\s*"idle",\s*"Choose a workflow preset'
        ),
    ),
    (
        "workflow.preset_count_summary",
        re.compile(r't\(\s*"workflow\.preset_count_summary"'),
        re.compile(r'summaryLabel\s*=\s*availableCount\s*\+\s*" presets ready"'),
    ),
    (
        "workflow.preset_count_title",
        re.compile(r't\(\s*"workflow\.preset_count_title"'),
        re.compile(
            r'summaryTitle\s*=\s*availableCount\s*\+\s*" built-in or custom workflow presets are available\."'
        ),
    ),
    (
        "workflow.preset_ready_pill",
        re.compile(r't\(\s*"workflow\.preset_ready_pill"'),
        re.compile(r'setStatusPill\("workflowPresetPill",\s*"Ready"'),
    ),
    (
        "workflow.preset_ready_title",
        re.compile(r't\(\s*"workflow\.preset_ready_title"'),
        re.compile(r'preset\.name\s*\+\s*" is ready to run\."'),
    ),
    (
        "workflow.preset_summary_label",
        re.compile(r't\(\s*"workflow\.preset_summary_label"'),
        re.compile(r'summaryLabel\s*=\s*preset\.name\s*\+\s*" • "'),
    ),
    (
        "workflow.preset_summary_title",
        re.compile(r't\(\s*"workflow\.preset_summary_title"'),
        re.compile(r'summaryTitle\s*=\s*preset\.name\s*\+\s*" runs "'),
    ),
    (
        "workflow.preset_status_loading",
        re.compile(r't\(\s*"workflow\.preset_status_loading"'),
        re.compile(r'setStatusLine\(\s*"workflowPresetStatus",\s*"Loading workflow presets for repeatable'),
    ),
    (
        "workflow.preset_status_reconnect",
        re.compile(r't\(\s*"workflow\.preset_status_reconnect"'),
        re.compile(r'setStatusLine\(\s*"workflowPresetStatus",\s*"Reconnect the backend before running'),
    ),
    (
        "workflow.preset_status_empty",
        re.compile(r't\(\s*"workflow\.preset_status_empty"'),
        re.compile(r'setStatusLine\(\s*"workflowPresetStatus",\s*"No workflow presets are available yet\.'),
    ),
    (
        "workflow.preset_status_choose",
        re.compile(r't\(\s*"workflow\.preset_status_choose"'),
        re.compile(r'setStatusLine\(\s*"workflowPresetStatus",\s*"Choose a workflow preset to preview its step order'),
    ),
    (
        "workflow.preset_status_choose_clip",
        re.compile(r't\(\s*"workflow\.preset_status_choose_clip"'),
        re.compile(r'preset\.name\s*\+\s*" is ready\. Choose a clip before starting the workflow\."'),
    ),
    (
        "workflow.preset_status_ready_on",
        re.compile(r't\(\s*"workflow\.preset_status_ready_on"'),
        re.compile(r'preset\.name\s*\+\s*" is ready to run on "'),
    ),
)


class TestI18nHardcodedMigration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.en = json.loads(EN_JSON.read_text(encoding="utf-8"))
        cls.js = MAIN_JS.read_text(encoding="utf-8")

    def test_all_migrated_keys_in_en_json(self):
        for key in MIGRATED_KEYS:
            with self.subTest(key=key):
                self.assertIn(key, self.en,
                              f"en.json missing migrated key {key!r}")
                self.assertTrue(self.en[key].strip(),
                                f"en.json {key!r} has empty value")

    def test_main_js_routes_through_t_for_each_key(self):
        for key, present_re, banned_re in EXPECTED_CALLS:
            with self.subTest(key=key):
                self.assertRegex(
                    self.js, present_re,
                    f"main.js does not invoke t({key!r}, …) — migration may have reverted",
                )
                if banned_re is None:
                    continue
                # Bare-English form must not appear as a showToast argument.
                self.assertNotRegex(
                    self.js, banned_re,
                    f"main.js still contains bare-English showToast for {key!r}",
                )

    def test_workspace_metadata_routes_through_t(self):
        for key, banned_re in WORKSPACE_METADATA_CALLS:
            with self.subTest(key=key):
                self.assertIn(key, self.en, f"en.json missing migrated key {key!r}")
                self.assertTrue(self.en[key].strip(), f"en.json {key!r} has empty value")
                self.assertRegex(
                    self.js,
                    re.compile(r't\(\s*["\']' + re.escape(key) + r'["\']'),
                    f"main.js does not invoke t({key!r}, …) — migration may have reverted",
                )
                self.assertNotRegex(
                    self.js,
                    banned_re,
                    f"main.js still contains bare-English workspace metadata for {key!r}",
                )


if __name__ == "__main__":
    unittest.main()
