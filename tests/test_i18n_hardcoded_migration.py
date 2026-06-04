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


if __name__ == "__main__":
    unittest.main()
