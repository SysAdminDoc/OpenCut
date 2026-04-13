"""
Tests for OpenCut Platform UX features.

Covers:
  - Web UI Backend (9.2): session lifecycle, file upload, operation catalog
  - After Effects Extension (9.3): project parsing, manifest generation, ExtendScript
  - Panel UX stubs:
      6.1  Drag-and-Drop handler
      6.2  Workspace Layouts (save/load/list/delete, built-ins)
      6.6  Right-Click Context Menu actions
      6.7  Quick Previews
      6.8  Theme Toggle
      37.1 Guided Walkthroughs
      37.2 Session State persistence
      37.5 Offline Documentation search/retrieval
  - Route smoke tests for all endpoints
"""

import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from dataclasses import asdict
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# 1. Web UI — Session Management
# ============================================================
class TestWebUISessions(unittest.TestCase):
    """Tests for opencut.core.web_ui session lifecycle."""

    def setUp(self):
        from opencut.core.web_ui import _session_lock, _sessions
        with _session_lock:
            _sessions.clear()

    def tearDown(self):
        from opencut.core.web_ui import _session_lock, _sessions
        with _session_lock:
            _sessions.clear()

    def test_create_session(self):
        from opencut.core.web_ui import create_session
        session = create_session()
        self.assertTrue(session.session_id)
        self.assertEqual(len(session.session_id), 16)
        self.assertIsInstance(session.uploaded_files, list)
        self.assertEqual(len(session.uploaded_files), 0)
        self.assertIsInstance(session.created_at, float)

    def test_create_multiple_sessions(self):
        from opencut.core.web_ui import create_session
        s1 = create_session()
        s2 = create_session()
        self.assertNotEqual(s1.session_id, s2.session_id)

    def test_get_session(self):
        from opencut.core.web_ui import create_session, get_session
        session = create_session()
        fetched = get_session(session.session_id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.session_id, session.session_id)

    def test_get_session_not_found(self):
        from opencut.core.web_ui import get_session
        result = get_session("nonexistent_session")
        self.assertIsNone(result)

    def test_get_session_expired(self):
        from opencut.core.web_ui import SESSION_MAX_AGE, _session_lock, _sessions, create_session, get_session
        session = create_session()
        # Manually expire it
        with _session_lock:
            _sessions[session.session_id].created_at = time.time() - SESSION_MAX_AGE - 1
        result = get_session(session.session_id)
        self.assertIsNone(result)

    def test_cleanup_session(self):
        from opencut.core.web_ui import cleanup_session, create_session, get_session
        session = create_session()
        result = cleanup_session(session.session_id)
        self.assertTrue(result)
        self.assertIsNone(get_session(session.session_id))

    def test_cleanup_session_not_found(self):
        from opencut.core.web_ui import cleanup_session
        result = cleanup_session("nonexistent")
        self.assertFalse(result)

    def test_cleanup_expired_sessions(self):
        from opencut.core.web_ui import (
            SESSION_MAX_AGE,
            _cleanup_expired_sessions,
            _session_lock,
            _sessions,
            create_session,
        )
        s1 = create_session()
        s2 = create_session()
        with _session_lock:
            _sessions[s1.session_id].created_at = time.time() - SESSION_MAX_AGE - 1
        removed = _cleanup_expired_sessions()
        self.assertEqual(removed, 1)
        self.assertNotIn(s1.session_id, _sessions)
        self.assertIn(s2.session_id, _sessions)

    def test_session_recent_operations_default_empty(self):
        from opencut.core.web_ui import create_session
        session = create_session()
        self.assertEqual(session.recent_operations, [])


# ============================================================
# 2. Web UI — File Upload
# ============================================================
class TestWebUIUpload(unittest.TestCase):
    """Tests for opencut.core.web_ui file upload."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        from opencut.core.web_ui import _session_lock, _sessions
        with _session_lock:
            _sessions.clear()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        from opencut.core.web_ui import _session_lock, _sessions
        with _session_lock:
            _sessions.clear()

    @patch("opencut.core.web_ui.WEB_UPLOADS_DIR")
    def test_upload_file(self, mock_dir):
        mock_dir.__str__ = lambda s: self.tmp
        with patch("opencut.core.web_ui.WEB_UPLOADS_DIR", self.tmp):
            from opencut.core.web_ui import create_session, upload_file
            session = create_session()
            uploaded = upload_file(session.session_id, "test.mp4", b"\x00" * 100)
            self.assertEqual(uploaded.filename, "test.mp4")
            self.assertEqual(uploaded.size, 100)
            self.assertEqual(uploaded.mime_type, "video/mp4")
            self.assertTrue(os.path.isfile(uploaded.path))

    @patch("opencut.core.web_ui.WEB_UPLOADS_DIR")
    def test_upload_file_session_not_found(self, mock_dir):
        mock_dir.__str__ = lambda s: self.tmp
        from opencut.core.web_ui import upload_file
        with self.assertRaises(ValueError):
            upload_file("nonexistent", "test.mp4", b"\x00")

    @patch("opencut.core.web_ui.WEB_UPLOADS_DIR")
    def test_upload_sanitizes_filename(self, mock_dir):
        mock_dir.__str__ = lambda s: self.tmp
        with patch("opencut.core.web_ui.WEB_UPLOADS_DIR", self.tmp):
            from opencut.core.web_ui import create_session, upload_file
            session = create_session()
            uploaded = upload_file(session.session_id, "../../../etc/passwd", b"test")
            self.assertNotIn("..", uploaded.filename)
            self.assertNotIn("/", uploaded.filename)

    @patch("opencut.core.web_ui.WEB_UPLOADS_DIR")
    def test_upload_empty_filename_gets_default(self, mock_dir):
        mock_dir.__str__ = lambda s: self.tmp
        with patch("opencut.core.web_ui.WEB_UPLOADS_DIR", self.tmp):
            from opencut.core.web_ui import create_session, upload_file
            session = create_session()
            uploaded = upload_file(session.session_id, "", b"test")
            self.assertTrue(uploaded.filename.startswith("upload_"))

    @patch("opencut.core.web_ui.WEB_UPLOADS_DIR")
    def test_list_uploads(self, mock_dir):
        mock_dir.__str__ = lambda s: self.tmp
        with patch("opencut.core.web_ui.WEB_UPLOADS_DIR", self.tmp):
            from opencut.core.web_ui import create_session, list_uploads, upload_file
            session = create_session()
            upload_file(session.session_id, "a.mp4", b"\x00" * 50)
            upload_file(session.session_id, "b.wav", b"\x00" * 30)
            uploads = list_uploads(session.session_id)
            self.assertEqual(len(uploads), 2)

    def test_list_uploads_empty_session(self):
        from opencut.core.web_ui import create_session, list_uploads
        session = create_session()
        uploads = list_uploads(session.session_id)
        self.assertEqual(uploads, [])

    def test_list_uploads_nonexistent_session(self):
        from opencut.core.web_ui import list_uploads
        uploads = list_uploads("nonexistent")
        self.assertEqual(uploads, [])

    @patch("opencut.core.web_ui.WEB_UPLOADS_DIR")
    def test_upload_mime_detection(self, mock_dir):
        mock_dir.__str__ = lambda s: self.tmp
        with patch("opencut.core.web_ui.WEB_UPLOADS_DIR", self.tmp):
            from opencut.core.web_ui import create_session, upload_file
            session = create_session()
            wav = upload_file(session.session_id, "audio.wav", b"\x00" * 10)
            self.assertIn("audio", wav.mime_type)
            png = upload_file(session.session_id, "image.png", b"\x89PNG")
            self.assertIn("image", png.mime_type)


# ============================================================
# 3. Web UI — Operation Catalog
# ============================================================
class TestWebUIOperationCatalog(unittest.TestCase):
    """Tests for opencut.core.web_ui operation catalog."""

    def test_get_operation_catalog(self):
        from opencut.core.web_ui import get_operation_catalog
        catalog = get_operation_catalog()
        self.assertIsInstance(catalog, dict)
        self.assertTrue(len(catalog) > 0)

    def test_catalog_has_categories(self):
        from opencut.core.web_ui import get_operation_catalog
        catalog = get_operation_catalog()
        categories = set(catalog.keys())
        self.assertIn("Video AI", categories)
        self.assertIn("Audio", categories)

    def test_catalog_operations_have_required_fields(self):
        from opencut.core.web_ui import get_operation_catalog
        catalog = get_operation_catalog()
        for cat, ops in catalog.items():
            for op in ops:
                self.assertIn("id", op)
                self.assertIn("name", op)
                self.assertIn("category", op)
                self.assertIn("description", op)
                self.assertIn("endpoint", op)
                self.assertIn("params_schema", op)

    def test_catalog_operation_count(self):
        from opencut.core.web_ui import _OPERATION_CATALOG, get_operation_catalog
        catalog = get_operation_catalog()
        total = sum(len(ops) for ops in catalog.values())
        self.assertEqual(total, len(_OPERATION_CATALOG))

    def test_serve_web_ui_returns_html(self):
        from opencut.core.web_ui import serve_web_ui
        html = serve_web_ui()
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("OpenCut", html)
        self.assertIn("upload", html.lower())

    def test_serve_web_ui_has_script(self):
        from opencut.core.web_ui import serve_web_ui
        html = serve_web_ui()
        self.assertIn("<script>", html)

    def test_uploaded_file_dataclass(self):
        from opencut.core.web_ui import UploadedFile
        f = UploadedFile(
            session_id="abc", filename="test.mp4",
            path="/tmp/test.mp4", size=100, mime_type="video/mp4"
        )
        d = asdict(f)
        self.assertEqual(d["filename"], "test.mp4")
        self.assertEqual(d["size"], 100)

    def test_operation_card_dataclass(self):
        from opencut.core.web_ui import OperationCard
        card = OperationCard(
            id="test_op", name="Test", category="Test",
            description="A test", endpoint="/test"
        )
        d = asdict(card)
        self.assertEqual(d["id"], "test_op")
        self.assertEqual(d["params_schema"], {})


# ============================================================
# 4. After Effects Extension
# ============================================================
class TestAEExtension(unittest.TestCase):
    """Tests for opencut.core.ae_extension."""

    def test_ae_supported_operations(self):
        from opencut.core.ae_extension import ae_supported_operations
        ops = ae_supported_operations()
        self.assertIsInstance(ops, list)
        self.assertTrue(len(ops) >= 7)

    def test_ae_supported_ops_have_required_fields(self):
        from opencut.core.ae_extension import ae_supported_operations
        for op in ae_supported_operations():
            self.assertIn("id", op)
            self.assertIn("name", op)
            self.assertIn("description", op)
            self.assertIn("ae_context", op)

    def test_ae_ops_include_bg_removal(self):
        from opencut.core.ae_extension import ae_supported_operations
        ids = [op["id"] for op in ae_supported_operations()]
        self.assertIn("bg_removal", ids)

    def test_ae_ops_include_upscale(self):
        from opencut.core.ae_extension import ae_supported_operations
        ids = [op["id"] for op in ae_supported_operations()]
        self.assertIn("upscale", ids)

    def test_ae_ops_include_denoise(self):
        from opencut.core.ae_extension import ae_supported_operations
        ids = [op["id"] for op in ae_supported_operations()]
        self.assertIn("denoise", ids)

    def test_get_ae_project_info(self):
        from opencut.core.ae_extension import get_ae_project_info
        data = {
            "project_path": "/path/to/project.aep",
            "comps": [
                {"name": "Main Comp", "width": 1920, "height": 1080, "fps": 30, "duration": 60, "num_layers": 5},
                {"name": "Precomp", "width": 1920, "height": 1080, "fps": 30, "duration": 10, "num_layers": 2},
            ],
            "active_comp": {"name": "Main Comp", "width": 1920, "height": 1080, "fps": 30, "duration": 60, "num_layers": 5},
        }
        project = get_ae_project_info(data)
        self.assertEqual(len(project.comps), 2)
        self.assertEqual(project.comps[0].name, "Main Comp")
        self.assertIsNotNone(project.active_comp)
        self.assertEqual(project.project_path, "/path/to/project.aep")

    def test_get_ae_project_info_empty(self):
        from opencut.core.ae_extension import get_ae_project_info
        project = get_ae_project_info({})
        self.assertEqual(len(project.comps), 0)
        self.assertIsNone(project.active_comp)
        self.assertEqual(project.project_path, "")

    def test_get_ae_project_info_no_active_comp(self):
        from opencut.core.ae_extension import get_ae_project_info
        data = {"comps": [{"name": "Comp1"}], "active_comp": None}
        project = get_ae_project_info(data)
        self.assertEqual(len(project.comps), 1)
        self.assertIsNone(project.active_comp)

    def test_get_comp_info(self):
        from opencut.core.ae_extension import get_comp_info
        comp = get_comp_info({"name": "Test", "width": 3840, "height": 2160, "fps": 24, "duration": 30, "num_layers": 10})
        self.assertEqual(comp.name, "Test")
        self.assertEqual(comp.width, 3840)
        self.assertEqual(comp.height, 2160)
        self.assertEqual(comp.fps, 24.0)
        self.assertEqual(comp.num_layers, 10)

    def test_get_comp_info_defaults(self):
        from opencut.core.ae_extension import get_comp_info
        comp = get_comp_info({})
        self.assertEqual(comp.name, "Untitled")
        self.assertEqual(comp.width, 1920)
        self.assertEqual(comp.height, 1080)

    def test_generate_ae_manifest(self):
        from opencut.core.ae_extension import AE_EXTENSION_ID, generate_ae_manifest
        manifest = generate_ae_manifest()
        self.assertIn("<?xml", manifest)
        self.assertIn("ExtensionManifest", manifest)
        self.assertIn(AE_EXTENSION_ID, manifest)
        self.assertIn("AEFT", manifest)

    def test_manifest_has_panel_type(self):
        from opencut.core.ae_extension import generate_ae_manifest
        manifest = generate_ae_manifest()
        self.assertIn("<Type>Panel</Type>", manifest)

    def test_manifest_has_csxs_runtime(self):
        from opencut.core.ae_extension import generate_ae_manifest
        manifest = generate_ae_manifest()
        self.assertIn("CSXS", manifest)

    def test_manifest_has_nodejs(self):
        from opencut.core.ae_extension import generate_ae_manifest
        manifest = generate_ae_manifest()
        self.assertIn("--enable-nodejs", manifest)

    def test_generate_ae_extendscript(self):
        from opencut.core.ae_extension import generate_ae_extendscript
        script = generate_ae_extendscript()
        self.assertIn("getProjectInfo", script)
        self.assertIn("getActiveCompLayers", script)
        self.assertIn("exportFrameForProcessing", script)

    def test_extendscript_returns_json(self):
        from opencut.core.ae_extension import generate_ae_extendscript
        script = generate_ae_extendscript()
        self.assertIn("JSON.stringify", script)

    def test_ae_comp_info_dataclass(self):
        from opencut.core.ae_extension import AECompInfo
        comp = AECompInfo(name="Test", width=1920, height=1080, fps=30, duration=10, num_layers=3)
        d = asdict(comp)
        self.assertEqual(d["name"], "Test")
        self.assertEqual(d["num_layers"], 3)

    def test_ae_layer_info_dataclass(self):
        from opencut.core.ae_extension import AELayerInfo
        layer = AELayerInfo(index=1, name="Background", type="footage", in_point=0.0, out_point=10.0)
        d = asdict(layer)
        self.assertEqual(d["type"], "footage")

    def test_ae_project_dataclass(self):
        from opencut.core.ae_extension import AEProject
        proj = AEProject(project_path="/test.aep")
        self.assertEqual(proj.comps, [])
        self.assertIsNone(proj.active_comp)


# ============================================================
# 5. Panel UX — Drag-and-Drop (6.1)
# ============================================================
class TestDragAndDrop(unittest.TestCase):
    """Tests for opencut.core.panel_ux drop handler."""

    def setUp(self):
        from opencut.core.panel_ux import clear_drop_registry
        clear_drop_registry()

    def test_register_drop_handler(self):
        from opencut.core.panel_ux import register_drop_handler
        action = register_drop_handler("/path/to/video.mp4", "silence_remove")
        self.assertEqual(action.file_path, "/path/to/video.mp4")
        self.assertEqual(action.operation, "silence_remove")

    def test_register_drop_handler_multiple(self):
        from opencut.core.panel_ux import get_drop_registry, register_drop_handler
        register_drop_handler("/a.mp4", "silence_remove")
        register_drop_handler("/b.mp4", "upscale")
        registry = get_drop_registry()
        self.assertEqual(len(registry), 2)

    def test_register_drop_handler_empty_path_raises(self):
        from opencut.core.panel_ux import register_drop_handler
        with self.assertRaises(ValueError):
            register_drop_handler("", "silence_remove")

    def test_register_drop_handler_empty_op_raises(self):
        from opencut.core.panel_ux import register_drop_handler
        with self.assertRaises(ValueError):
            register_drop_handler("/path.mp4", "")

    def test_clear_drop_registry(self):
        from opencut.core.panel_ux import clear_drop_registry, get_drop_registry, register_drop_handler
        register_drop_handler("/a.mp4", "test")
        clear_drop_registry()
        self.assertEqual(len(get_drop_registry()), 0)


# ============================================================
# 6. Panel UX — Workspace Layouts (6.2)
# ============================================================
class TestWorkspaceLayouts(unittest.TestCase):
    """Tests for opencut.core.panel_ux layouts."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._orig_layouts_dir = None

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_list_layouts_includes_builtins(self):
        from opencut.core.panel_ux import list_layouts
        with patch("opencut.core.panel_ux.LAYOUTS_DIR", os.path.join(self.tmp, "layouts")):
            layouts = list_layouts()
            names = [lt["name"] for lt in layouts]
            self.assertIn("Assembly", names)
            self.assertIn("Audio", names)
            self.assertIn("Color", names)
            self.assertIn("Delivery", names)

    def test_builtin_layouts_are_marked(self):
        from opencut.core.panel_ux import list_layouts
        with patch("opencut.core.panel_ux.LAYOUTS_DIR", os.path.join(self.tmp, "layouts")):
            for layout in list_layouts():
                if layout["name"] in ("Assembly", "Audio", "Color", "Delivery"):
                    self.assertTrue(layout["builtin"])

    def test_load_builtin_layout(self):
        from opencut.core.panel_ux import load_layout
        layout = load_layout("Assembly")
        self.assertIsNotNone(layout)
        self.assertEqual(layout["name"], "Assembly")
        self.assertTrue(layout["builtin"])

    def test_load_nonexistent_layout(self):
        from opencut.core.panel_ux import load_layout
        with patch("opencut.core.panel_ux.LAYOUTS_DIR", os.path.join(self.tmp, "layouts")):
            layout = load_layout("nonexistent")
            self.assertIsNone(layout)

    def test_save_and_load_custom_layout(self):
        from opencut.core.panel_ux import load_layout, save_layout
        layouts_dir = os.path.join(self.tmp, "layouts")
        with patch("opencut.core.panel_ux.LAYOUTS_DIR", layouts_dir):
            state = {"panels": {"timeline": {"visible": True}}}
            save_layout("MyLayout", state)
            loaded = load_layout("MyLayout")
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["name"], "MyLayout")
            self.assertFalse(loaded["builtin"])

    def test_save_layout_empty_name_raises(self):
        from opencut.core.panel_ux import save_layout
        with self.assertRaises(ValueError):
            save_layout("", {})

    def test_delete_custom_layout(self):
        from opencut.core.panel_ux import delete_layout, load_layout, save_layout
        layouts_dir = os.path.join(self.tmp, "layouts")
        with patch("opencut.core.panel_ux.LAYOUTS_DIR", layouts_dir):
            save_layout("ToDelete", {"test": True})
            result = delete_layout("ToDelete")
            self.assertTrue(result)
            self.assertIsNone(load_layout("ToDelete"))

    def test_delete_builtin_layout_raises(self):
        from opencut.core.panel_ux import delete_layout
        with self.assertRaises(ValueError):
            delete_layout("Assembly")

    def test_delete_nonexistent_layout(self):
        from opencut.core.panel_ux import delete_layout
        layouts_dir = os.path.join(self.tmp, "layouts")
        with patch("opencut.core.panel_ux.LAYOUTS_DIR", layouts_dir):
            os.makedirs(layouts_dir, exist_ok=True)
            result = delete_layout("nonexistent")
            self.assertFalse(result)

    def test_list_layouts_includes_custom(self):
        from opencut.core.panel_ux import list_layouts, save_layout
        layouts_dir = os.path.join(self.tmp, "layouts")
        with patch("opencut.core.panel_ux.LAYOUTS_DIR", layouts_dir):
            save_layout("Custom1", {"test": True})
            layouts = list_layouts()
            names = [lt["name"] for lt in layouts]
            self.assertIn("Custom1", names)


# ============================================================
# 7. Panel UX — Context Menu (6.6)
# ============================================================
class TestContextMenu(unittest.TestCase):
    """Tests for opencut.core.panel_ux context menu actions."""

    def test_get_video_actions(self):
        from opencut.core.panel_ux import get_context_menu_actions
        actions = get_context_menu_actions("video")
        self.assertTrue(len(actions) > 0)
        action_names = [a["action"] for a in actions]
        self.assertIn("silence_remove", action_names)

    def test_get_audio_actions(self):
        from opencut.core.panel_ux import get_context_menu_actions
        actions = get_context_menu_actions("audio")
        action_names = [a["action"] for a in actions]
        self.assertIn("audio_enhance", action_names)

    def test_get_image_actions(self):
        from opencut.core.panel_ux import get_context_menu_actions
        actions = get_context_menu_actions("image")
        action_names = [a["action"] for a in actions]
        self.assertIn("bg_removal", action_names)

    def test_get_subtitle_actions(self):
        from opencut.core.panel_ux import get_context_menu_actions
        actions = get_context_menu_actions("subtitle")
        self.assertTrue(len(actions) > 0)

    def test_unknown_type_falls_back_to_video(self):
        from opencut.core.panel_ux import get_context_menu_actions
        video_actions = get_context_menu_actions("video")
        unknown_actions = get_context_menu_actions("unknown_type")
        self.assertEqual(len(video_actions), len(unknown_actions))

    def test_actions_have_required_fields(self):
        from opencut.core.panel_ux import get_context_menu_actions
        for action in get_context_menu_actions("video"):
            self.assertIn("action", action)
            self.assertIn("label", action)
            self.assertIn("icon", action)

    def test_case_insensitive_clip_type(self):
        from opencut.core.panel_ux import get_context_menu_actions
        upper = get_context_menu_actions("VIDEO")
        lower = get_context_menu_actions("video")
        self.assertEqual(len(upper), len(lower))


# ============================================================
# 8. Panel UX — Quick Previews (6.7)
# ============================================================
class TestQuickPreviews(unittest.TestCase):
    """Tests for opencut.core.panel_ux quick previews."""

    def test_generate_preview(self):
        from opencut.core.panel_ux import generate_operation_preview
        result = generate_operation_preview("denoise", "/tmp/frame.png")
        self.assertEqual(result.operation, "denoise")
        self.assertEqual(result.source_frame, "/tmp/frame.png")
        self.assertIn("preview_denoise", result.preview_path)
        self.assertEqual(result.status, "generated")

    def test_preview_empty_operation_raises(self):
        from opencut.core.panel_ux import generate_operation_preview
        with self.assertRaises(ValueError):
            generate_operation_preview("", "/tmp/frame.png")

    def test_preview_empty_frame_raises(self):
        from opencut.core.panel_ux import generate_operation_preview
        with self.assertRaises(ValueError):
            generate_operation_preview("upscale", "")

    def test_preview_result_dataclass(self):
        from opencut.core.panel_ux import generate_operation_preview
        result = generate_operation_preview("style_transfer", "/tmp/frame.jpg")
        d = asdict(result)
        self.assertIn("operation", d)
        self.assertIn("preview_path", d)
        self.assertIn("generated_at", d)


# ============================================================
# 9. Panel UX — Themes (6.8)
# ============================================================
class TestThemes(unittest.TestCase):
    """Tests for opencut.core.panel_ux theme toggle."""

    def test_list_themes(self):
        from opencut.core.panel_ux import list_themes
        themes = list_themes()
        self.assertEqual(len(themes), 3)
        names = [t["name"] for t in themes]
        self.assertIn("dark", names)
        self.assertIn("light", names)
        self.assertIn("system", names)

    def test_get_dark_theme(self):
        from opencut.core.panel_ux import get_theme
        theme = get_theme("dark")
        self.assertIsNotNone(theme)
        self.assertEqual(theme["name"], "dark")
        self.assertIn("--bg-primary", theme)
        self.assertIn("--accent", theme)

    def test_get_light_theme(self):
        from opencut.core.panel_ux import get_theme
        theme = get_theme("light")
        self.assertIsNotNone(theme)
        self.assertEqual(theme["name"], "light")

    def test_get_system_theme(self):
        from opencut.core.panel_ux import get_theme
        theme = get_theme("system")
        self.assertIsNotNone(theme)
        self.assertEqual(theme["name"], "system")

    def test_get_nonexistent_theme(self):
        from opencut.core.panel_ux import get_theme
        theme = get_theme("neon_pink")
        self.assertIsNone(theme)

    def test_themes_have_css_properties(self):
        from opencut.core.panel_ux import get_theme
        for name in ("dark", "light"):
            theme = get_theme(name)
            self.assertIn("--text-primary", theme)
            self.assertIn("--bg-secondary", theme)
            self.assertIn("--error", theme)


# ============================================================
# 10. Panel UX — Guided Walkthroughs (37.1)
# ============================================================
class TestWalkthroughs(unittest.TestCase):
    """Tests for opencut.core.panel_ux walkthroughs."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_list_walkthroughs(self):
        from opencut.core.panel_ux import list_walkthroughs
        with patch("opencut.core.panel_ux.WALKTHROUGH_FILE",
                   os.path.join(self.tmp, "wt.json")):
            wts = list_walkthroughs()
            self.assertTrue(len(wts) >= 6)
            for wt in wts:
                self.assertIn("feature_id", wt)
                self.assertIn("title", wt)
                self.assertIn("num_steps", wt)

    def test_get_walkthrough(self):
        from opencut.core.panel_ux import get_walkthrough
        with patch("opencut.core.panel_ux.WALKTHROUGH_FILE",
                   os.path.join(self.tmp, "wt.json")):
            wt = get_walkthrough("first_import")
            self.assertIsNotNone(wt)
            self.assertEqual(wt["feature_id"], "first_import")
            self.assertTrue(len(wt["steps"]) > 0)
            self.assertFalse(wt["completed"])

    def test_get_walkthrough_not_found(self):
        from opencut.core.panel_ux import get_walkthrough
        with patch("opencut.core.panel_ux.WALKTHROUGH_FILE",
                   os.path.join(self.tmp, "wt.json")):
            wt = get_walkthrough("nonexistent_feature")
            self.assertIsNone(wt)

    def test_mark_walkthrough_completed(self):
        from opencut.core.panel_ux import get_walkthrough, mark_walkthrough_completed
        wt_file = os.path.join(self.tmp, "wt.json")
        with patch("opencut.core.panel_ux.WALKTHROUGH_FILE", wt_file), \
             patch("opencut.core.panel_ux.PANEL_DIR", self.tmp):
            result = mark_walkthrough_completed("first_import")
            self.assertTrue(result)
            wt = get_walkthrough("first_import")
            self.assertTrue(wt["completed"])
            self.assertIsNotNone(wt["completed_at"])

    def test_mark_nonexistent_walkthrough(self):
        from opencut.core.panel_ux import mark_walkthrough_completed
        result = mark_walkthrough_completed("nonexistent")
        self.assertFalse(result)

    def test_walkthrough_steps_have_fields(self):
        from opencut.core.panel_ux import get_walkthrough
        with patch("opencut.core.panel_ux.WALKTHROUGH_FILE",
                   os.path.join(self.tmp, "wt.json")):
            wt = get_walkthrough("first_operation")
            for step in wt["steps"]:
                self.assertIn("step_number", step)
                self.assertIn("title", step)
                self.assertIn("description", step)
                self.assertIn("target_element", step)
                self.assertIn("action", step)

    def test_walkthrough_completion_persists(self):
        from opencut.core.panel_ux import _load_completed_walkthroughs, mark_walkthrough_completed
        wt_file = os.path.join(self.tmp, "wt.json")
        with patch("opencut.core.panel_ux.WALKTHROUGH_FILE", wt_file), \
             patch("opencut.core.panel_ux.PANEL_DIR", self.tmp):
            mark_walkthrough_completed("workspace_setup")
            completed = _load_completed_walkthroughs()
            self.assertIn("workspace_setup", completed)


# ============================================================
# 11. Panel UX — Session State (37.2)
# ============================================================
class TestSessionState(unittest.TestCase):
    """Tests for opencut.core.panel_ux session state persistence."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_save_session_state(self):
        from opencut.core.panel_ux import save_session_state
        state_file = os.path.join(self.tmp, "state.json")
        with patch("opencut.core.panel_ux.STATE_FILE", state_file), \
             patch("opencut.core.panel_ux.PANEL_DIR", self.tmp):
            result = save_session_state({"active_layout": "Color", "zoom": 1.5})
            self.assertIn("state", result)
            self.assertIn("saved_at", result)
            self.assertTrue(os.path.isfile(state_file))

    def test_restore_session_state(self):
        from opencut.core.panel_ux import restore_session_state, save_session_state
        state_file = os.path.join(self.tmp, "state.json")
        with patch("opencut.core.panel_ux.STATE_FILE", state_file), \
             patch("opencut.core.panel_ux.PANEL_DIR", self.tmp):
            save_session_state({"active_layout": "Audio", "zoom": 2.0})
            restored = restore_session_state()
            self.assertIsNotNone(restored)
            self.assertEqual(restored["state"]["active_layout"], "Audio")

    def test_restore_no_saved_state(self):
        from opencut.core.panel_ux import restore_session_state
        state_file = os.path.join(self.tmp, "nonexistent_state.json")
        with patch("opencut.core.panel_ux.STATE_FILE", state_file):
            result = restore_session_state()
            self.assertIsNone(result)

    def test_save_overwrites_previous(self):
        from opencut.core.panel_ux import restore_session_state, save_session_state
        state_file = os.path.join(self.tmp, "state.json")
        with patch("opencut.core.panel_ux.STATE_FILE", state_file), \
             patch("opencut.core.panel_ux.PANEL_DIR", self.tmp):
            save_session_state({"v": 1})
            save_session_state({"v": 2})
            restored = restore_session_state()
            self.assertEqual(restored["state"]["v"], 2)


# ============================================================
# 12. Panel UX — Offline Docs (37.5)
# ============================================================
class TestOfflineDocs(unittest.TestCase):
    """Tests for opencut.core.panel_ux offline documentation."""

    def test_get_documentation(self):
        from opencut.core.panel_ux import get_documentation
        doc = get_documentation("getting_started")
        self.assertIsNotNone(doc)
        self.assertEqual(doc["topic"], "getting_started")
        self.assertIn("content", doc)

    def test_get_documentation_not_found(self):
        from opencut.core.panel_ux import get_documentation
        doc = get_documentation("nonexistent_topic")
        self.assertIsNone(doc)

    def test_list_doc_topics(self):
        from opencut.core.panel_ux import list_doc_topics
        topics = list_doc_topics()
        self.assertTrue(len(topics) >= 10)
        for t in topics:
            self.assertIn("topic", t)
            self.assertIn("title", t)

    def test_search_docs(self):
        from opencut.core.panel_ux import search_docs
        results = search_docs("caption")
        self.assertTrue(len(results) > 0)
        self.assertIn("score", results[0])

    def test_search_docs_empty_query(self):
        from opencut.core.panel_ux import search_docs
        results = search_docs("")
        self.assertEqual(results, [])

    def test_search_docs_multiple_terms(self):
        from opencut.core.panel_ux import search_docs
        results = search_docs("color grading LUT")
        self.assertTrue(len(results) > 0)
        # The color grading doc should score highest
        self.assertIn("color", results[0]["topic"].lower())

    def test_search_docs_no_results(self):
        from opencut.core.panel_ux import search_docs
        results = search_docs("xyzzy_nonexistent_gibberish")
        self.assertEqual(results, [])

    def test_doc_has_content(self):
        from opencut.core.panel_ux import get_documentation
        doc = get_documentation("silence_removal")
        self.assertTrue(len(doc["content"]) > 50)

    def test_search_results_sorted_by_score(self):
        from opencut.core.panel_ux import search_docs
        results = search_docs("video export")
        if len(results) >= 2:
            self.assertGreaterEqual(results[0]["score"], results[1]["score"])

    def test_search_results_have_snippet(self):
        from opencut.core.panel_ux import search_docs
        results = search_docs("FFmpeg")
        for r in results:
            self.assertIn("snippet", r)

    def test_all_topics_are_documented(self):
        from opencut.core.panel_ux import get_documentation, list_doc_topics
        for t in list_doc_topics():
            doc = get_documentation(t["topic"])
            self.assertIsNotNone(doc, f"Topic {t['topic']} has no documentation")


# ============================================================
# 13. Route Smoke Tests
# ============================================================
class TestPlatformUXRoutes(unittest.TestCase):
    """Smoke tests for all platform UX routes."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        config = OpenCutConfig()
        cls.app = create_app(config=config)
        cls.app.config["TESTING"] = True
        # Register the platform_ux blueprint if not already registered
        from opencut.routes.platform_ux_routes import platform_ux_bp
        if "platform_ux" not in cls.app.blueprints:
            cls.app.register_blueprint(platform_ux_bp)
        cls.client = cls.app.test_client()
        # Get CSRF token
        resp = cls.client.get("/health")
        data = resp.get_json()
        cls.csrf = data.get("csrf_token", "")
        cls.headers = {
            "X-OpenCut-Token": cls.csrf,
            "Content-Type": "application/json",
        }

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # -- Web UI routes --

    def test_route_create_session(self):
        resp = self.client.post("/web-ui/session/create", headers=self.headers, data="{}")
        self.assertIn(resp.status_code, (200, 201))
        data = resp.get_json()
        self.assertIn("session_id", data)

    def test_route_operations_catalog(self):
        resp = self.client.get("/web-ui/operations")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("catalog", data)

    def test_route_list_files_nonexistent_session(self):
        resp = self.client.get("/web-ui/session/nonexistent/files")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["files"], [])

    def test_route_cleanup_nonexistent_session(self):
        resp = self.client.delete("/web-ui/session/nonexistent", headers=self.headers)
        self.assertEqual(resp.status_code, 404)

    def test_route_session_lifecycle(self):
        # Create
        resp = self.client.post("/web-ui/session/create", headers=self.headers, data="{}")
        sid = resp.get_json()["session_id"]
        # List files (empty)
        resp = self.client.get(f"/web-ui/session/{sid}/files")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.get_json()["files"]), 0)
        # Delete
        resp = self.client.delete(f"/web-ui/session/{sid}", headers=self.headers)
        self.assertEqual(resp.status_code, 200)

    # -- AE routes --

    def test_route_ae_supported_operations(self):
        resp = self.client.get("/ae/supported-operations")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("operations", data)
        self.assertTrue(len(data["operations"]) >= 7)

    def test_route_ae_manifest(self):
        resp = self.client.get("/ae/manifest")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("xml", resp.content_type)
        self.assertIn(b"ExtensionManifest", resp.data)

    def test_route_ae_project_info(self):
        resp = self.client.get("/ae/project-info",
                               headers=self.headers,
                               data=json.dumps({"comps": [{"name": "Main"}]}))
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("comps", data)

    # -- Layout routes --

    def test_route_list_layouts(self):
        resp = self.client.get("/panel/layouts")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("layouts", data)
        names = [lt["name"] for lt in data["layouts"]]
        self.assertIn("Assembly", names)

    def test_route_load_builtin_layout(self):
        resp = self.client.get("/panel/layout/Assembly")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["name"], "Assembly")

    def test_route_load_nonexistent_layout(self):
        resp = self.client.get("/panel/layout/nonexistent_layout_xyz")
        self.assertEqual(resp.status_code, 404)

    def test_route_save_layout(self):
        resp = self.client.post("/panel/layout/save", headers=self.headers,
                                data=json.dumps({"name": "TestLayout", "state": {"test": True}}))
        self.assertIn(resp.status_code, (200, 201))
        data = resp.get_json()
        self.assertEqual(data["name"], "TestLayout")

    # -- Drop handler route --

    def test_route_drop_handler(self):
        resp = self.client.post("/panel/drop-handler", headers=self.headers,
                                data=json.dumps({"file_path": "/tmp/test.mp4", "operation": "denoise"}))
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["operation"], "denoise")

    def test_route_drop_handler_missing_fields(self):
        resp = self.client.post("/panel/drop-handler", headers=self.headers,
                                data=json.dumps({"file_path": "/tmp/test.mp4"}))
        self.assertEqual(resp.status_code, 400)

    # -- Context menu route --

    def test_route_context_menu_default(self):
        resp = self.client.get("/panel/context-menu")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("actions", data)

    def test_route_context_menu_audio(self):
        resp = self.client.get("/panel/context-menu?clip_type=audio")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["clip_type"], "audio")

    # -- Theme routes --

    def test_route_list_themes(self):
        resp = self.client.get("/panel/themes")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("themes", data)
        self.assertEqual(len(data["themes"]), 3)

    def test_route_get_dark_theme(self):
        resp = self.client.get("/panel/theme/dark")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["name"], "dark")

    def test_route_get_nonexistent_theme(self):
        resp = self.client.get("/panel/theme/neon_pink")
        self.assertEqual(resp.status_code, 404)

    # -- Walkthrough routes --

    def test_route_get_walkthrough(self):
        resp = self.client.get("/panel/walkthrough/first_import")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["feature_id"], "first_import")

    def test_route_get_nonexistent_walkthrough(self):
        resp = self.client.get("/panel/walkthrough/nonexistent")
        self.assertEqual(resp.status_code, 404)

    def test_route_complete_walkthrough(self):
        resp = self.client.post("/panel/walkthrough/first_import/complete",
                                headers=self.headers, data="{}")
        self.assertIn(resp.status_code, (200, 201))
        data = resp.get_json()
        self.assertTrue(data["completed"])

    def test_route_complete_nonexistent_walkthrough(self):
        resp = self.client.post("/panel/walkthrough/nonexistent/complete",
                                headers=self.headers, data="{}")
        self.assertEqual(resp.status_code, 404)

    # -- State routes --

    def test_route_save_state(self):
        resp = self.client.post("/panel/state/save", headers=self.headers,
                                data=json.dumps({"state": {"layout": "Color"}}))
        self.assertEqual(resp.status_code, 200)

    def test_route_restore_state(self):
        # Save first
        self.client.post("/panel/state/save", headers=self.headers,
                         data=json.dumps({"state": {"layout": "Delivery"}}))
        resp = self.client.get("/panel/state/restore")
        self.assertEqual(resp.status_code, 200)

    # -- Docs routes --

    def test_route_get_doc(self):
        resp = self.client.get("/panel/docs/getting_started")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["topic"], "getting_started")

    def test_route_get_nonexistent_doc(self):
        resp = self.client.get("/panel/docs/nonexistent_topic")
        self.assertEqual(resp.status_code, 404)

    def test_route_search_docs(self):
        resp = self.client.get("/panel/docs/search?q=caption")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("results", data)

    def test_route_search_docs_empty(self):
        resp = self.client.get("/panel/docs/search?q=")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["results"], [])


if __name__ == "__main__":
    unittest.main()
