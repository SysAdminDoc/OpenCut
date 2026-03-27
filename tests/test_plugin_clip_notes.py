"""Tests for the Clip Notes example plugin."""

import csv
import importlib.util
import io
import os

import pytest
from flask import Flask


@pytest.fixture
def plugin_module(tmp_path):
    """Load the clip-notes routes module with a temp notes file."""
    plugin_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "opencut", "data", "example_plugins", "clip-notes", "routes.py",
    )
    spec = importlib.util.spec_from_file_location("clip_notes_routes", plugin_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Redirect storage to temp directory
    notes_path = str(tmp_path / "notes.json")
    mod._NOTES_PATH = notes_path

    return mod


@pytest.fixture
def client(plugin_module):
    """Create a Flask test client with the plugin blueprint registered."""
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(plugin_module.plugin_bp, url_prefix="/plugins/clip-notes")
    with app.test_client() as c:
        yield c


class TestSaveNote:
    """Tests for POST /note."""

    def test_save_note_success(self, client):
        r = client.post("/plugins/clip-notes/note", json={
            "filepath": "/footage/interview.mp4",
            "timestamp": "00:02:30",
            "text": "Great quote here",
        })
        assert r.status_code == 200
        data = r.get_json()
        assert data["success"] is True
        assert data["note"]["filepath"] == "/footage/interview.mp4"
        assert data["note"]["timestamp"] == "00:02:30"
        assert data["note"]["text"] == "Great quote here"
        assert "id" in data["note"]

    def test_save_note_missing_filepath(self, client):
        r = client.post("/plugins/clip-notes/note", json={
            "timestamp": "00:01:00",
            "text": "missing filepath",
        })
        assert r.status_code == 400
        assert "filepath" in r.get_json()["error"]

    def test_save_note_missing_timestamp(self, client):
        r = client.post("/plugins/clip-notes/note", json={
            "filepath": "/a.mp4",
            "text": "missing timestamp",
        })
        assert r.status_code == 400
        assert "timestamp" in r.get_json()["error"]

    def test_save_note_missing_text(self, client):
        r = client.post("/plugins/clip-notes/note", json={
            "filepath": "/a.mp4",
            "timestamp": "00:00:00",
        })
        assert r.status_code == 400
        assert "text" in r.get_json()["error"]

    def test_save_multiple_notes(self, client):
        for i in range(3):
            r = client.post("/plugins/clip-notes/note", json={
                "filepath": "/a.mp4",
                "timestamp": f"00:0{i}:00",
                "text": f"Note {i}",
            })
            assert r.status_code == 200


class TestGetNotes:
    """Tests for GET /notes."""

    def test_get_notes_empty(self, client):
        r = client.get("/plugins/clip-notes/notes?filepath=/a.mp4")
        assert r.status_code == 200
        data = r.get_json()
        assert data["notes"] == []
        assert data["count"] == 0

    def test_get_notes_for_clip(self, client):
        client.post("/plugins/clip-notes/note", json={
            "filepath": "/a.mp4", "timestamp": "00:01:00", "text": "Note A",
        })
        client.post("/plugins/clip-notes/note", json={
            "filepath": "/b.mp4", "timestamp": "00:02:00", "text": "Note B",
        })
        client.post("/plugins/clip-notes/note", json={
            "filepath": "/a.mp4", "timestamp": "00:03:00", "text": "Note A2",
        })

        r = client.get("/plugins/clip-notes/notes?filepath=/a.mp4")
        data = r.get_json()
        assert data["count"] == 2
        texts = [n["text"] for n in data["notes"]]
        assert "Note A" in texts
        assert "Note A2" in texts

    def test_get_notes_missing_filepath(self, client):
        r = client.get("/plugins/clip-notes/notes")
        assert r.status_code == 400
        assert "filepath" in r.get_json()["error"]


class TestDeleteNote:
    """Tests for DELETE /note."""

    def test_delete_note_success(self, client):
        r = client.post("/plugins/clip-notes/note", json={
            "filepath": "/a.mp4", "timestamp": "00:01:00", "text": "To delete",
        })
        note_id = r.get_json()["note"]["id"]

        r2 = client.delete(f"/plugins/clip-notes/note?note_id={note_id}")
        assert r2.status_code == 200
        assert r2.get_json()["success"] is True
        assert r2.get_json()["deleted_id"] == note_id

        # Verify it's gone
        r3 = client.get("/plugins/clip-notes/notes?filepath=/a.mp4")
        assert r3.get_json()["count"] == 0

    def test_delete_note_not_found(self, client):
        r = client.delete("/plugins/clip-notes/note?note_id=nonexistent")
        assert r.status_code == 404

    def test_delete_note_missing_id(self, client):
        r = client.delete("/plugins/clip-notes/note")
        assert r.status_code == 400


class TestExportNotes:
    """Tests for GET /export."""

    def _add_sample_notes(self, client):
        client.post("/plugins/clip-notes/note", json={
            "filepath": "/a.mp4", "timestamp": "00:01:00", "text": "First note",
        })
        client.post("/plugins/clip-notes/note", json={
            "filepath": "/b.mp4", "timestamp": "00:05:30", "text": "Second note",
        })

    def test_export_text(self, client):
        self._add_sample_notes(client)
        r = client.get("/plugins/clip-notes/export?format=text")
        assert r.status_code == 200
        assert r.content_type.startswith("text/plain")
        body = r.data.decode()
        assert "[00:01:00]" in body
        assert "First note" in body
        assert "/a.mp4" in body

    def test_export_csv(self, client):
        self._add_sample_notes(client)
        r = client.get("/plugins/clip-notes/export?format=csv")
        assert r.status_code == 200
        assert r.content_type.startswith("text/csv")
        reader = csv.reader(io.StringIO(r.data.decode()))
        rows = list(reader)
        # Header + 2 data rows
        assert len(rows) == 3
        assert rows[0] == ["id", "filepath", "timestamp", "text"]
        assert rows[1][3] == "First note"

    def test_export_default_format_is_text(self, client):
        r = client.get("/plugins/clip-notes/export")
        assert r.status_code == 200
        assert r.content_type.startswith("text/plain")

    def test_export_empty(self, client):
        r = client.get("/plugins/clip-notes/export?format=text")
        assert r.status_code == 200
        assert "No notes." in r.data.decode()

    def test_export_csv_empty(self, client):
        r = client.get("/plugins/clip-notes/export?format=csv")
        assert r.status_code == 200
        reader = csv.reader(io.StringIO(r.data.decode()))
        rows = list(reader)
        # Only header row
        assert len(rows) == 1
