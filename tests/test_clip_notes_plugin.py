"""Tests for the Clip Notes plugin."""

import os

import pytest


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Use a temp database for all tests."""
    # Need to patch BEFORE import
    db_path = str(tmp_path / "test_clip_notes.db")

    # We need to patch the module's _DB_PATH
    # Import the routes module and patch it
    import importlib.util
    plugin_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "opencut", "data", "example_plugins", "clip-notes", "routes.py"
    )
    spec = importlib.util.spec_from_file_location("clip_notes_routes", plugin_path)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setattr(mod, "_DB_PATH", db_path) if hasattr(mod, "_DB_PATH") else None

    # Alternative simpler approach: just test the plugin via HTTP using Flask test client
    yield db_path


class TestClipNotesPlugin:

    @pytest.fixture
    def app(self, tmp_path):
        """Create a minimal Flask app with the plugin loaded."""
        import importlib.util

        from flask import Flask

        app = Flask(__name__)
        app.config["TESTING"] = True

        plugin_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "opencut", "data", "example_plugins", "clip-notes", "routes.py"
        )

        spec = importlib.util.spec_from_file_location("clip_notes_test", plugin_path)
        mod = importlib.util.module_from_spec(spec)

        # Patch DB path before exec
        db_path = str(tmp_path / "notes.db")

        # We'll patch after loading
        spec.loader.exec_module(mod)
        mod._DB_PATH = db_path
        mod._thread_local.conn = None  # Reset connection
        mod._init_db()

        app.register_blueprint(mod.plugin_bp, url_prefix="/plugins/clip-notes")
        return app

    @pytest.fixture
    def client(self, app):
        with app.test_client() as c:
            yield c

    def test_list_empty(self, client):
        r = client.get("/plugins/clip-notes/list")
        assert r.status_code == 200
        assert r.get_json()["count"] == 0

    def test_add_note(self, client):
        r = client.post("/plugins/clip-notes/add", json={
            "clip_path": "/project/footage/interview.mp4",
            "clip_name": "interview.mp4",
            "note": "Great quote at 2:30",
            "tags": ["quote", "highlight"],
            "color": "green",
        })
        assert r.status_code == 200
        data = r.get_json()
        assert data["success"]
        assert data["note"]["note"] == "Great quote at 2:30"
        assert data["note"]["color"] == "green"

    def test_add_and_list(self, client):
        client.post("/plugins/clip-notes/add", json={
            "clip_path": "/a.mp4", "note": "note 1"
        })
        client.post("/plugins/clip-notes/add", json={
            "clip_path": "/b.mp4", "note": "note 2"
        })
        r = client.get("/plugins/clip-notes/list")
        assert r.get_json()["count"] == 2

    def test_filter_by_clip(self, client):
        client.post("/plugins/clip-notes/add", json={
            "clip_path": "/a.mp4", "note": "first"
        })
        client.post("/plugins/clip-notes/add", json={
            "clip_path": "/b.mp4", "note": "second"
        })
        r = client.get("/plugins/clip-notes/list?clip_path=/a.mp4")
        assert r.get_json()["count"] == 1

    def test_update_note(self, client):
        r = client.post("/plugins/clip-notes/add", json={
            "clip_path": "/a.mp4", "note": "original"
        })
        note_id = r.get_json()["note"]["id"]

        r2 = client.post("/plugins/clip-notes/update", json={
            "id": note_id, "note": "updated text", "color": "red"
        })
        assert r2.status_code == 200
        assert r2.get_json()["note"]["note"] == "updated text"
        assert r2.get_json()["note"]["color"] == "red"

    def test_delete_note(self, client):
        r = client.post("/plugins/clip-notes/add", json={
            "clip_path": "/a.mp4", "note": "to delete"
        })
        note_id = r.get_json()["note"]["id"]

        r2 = client.post("/plugins/clip-notes/delete", json={"id": note_id})
        assert r2.status_code == 200
        assert r2.get_json()["success"]

        r3 = client.get("/plugins/clip-notes/list")
        assert r3.get_json()["count"] == 0

    def test_search_notes(self, client):
        client.post("/plugins/clip-notes/add", json={
            "clip_path": "/a.mp4", "note": "The quick brown fox"
        })
        client.post("/plugins/clip-notes/add", json={
            "clip_path": "/b.mp4", "note": "A lazy dog sleeps"
        })

        r = client.post("/plugins/clip-notes/search", json={"query": "brown fox"})
        assert r.status_code == 200
        assert r.get_json()["count"] >= 1

    def test_stats(self, client):
        client.post("/plugins/clip-notes/add", json={
            "clip_path": "/a.mp4", "note": "test"
        })
        r = client.get("/plugins/clip-notes/stats")
        assert r.status_code == 200
        data = r.get_json()
        assert data["total_notes"] == 1
        assert data["clips_with_notes"] == 1

    def test_add_missing_clip_path(self, client):
        r = client.post("/plugins/clip-notes/add", json={"note": "no path"})
        assert r.status_code == 400

    def test_delete_nonexistent(self, client):
        r = client.post("/plugins/clip-notes/delete", json={"id": 99999})
        assert r.status_code == 404
