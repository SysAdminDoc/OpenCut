"""Path-safety for model-download metadata filenames.

``model_name`` is attacker-controllable through the download API, so the
metadata path derived from it must stay inside DOWNLOADS_META_DIR regardless of
separators, traversal sequences, or dotfile tricks.
"""

import os

import pytest

from opencut.core import model_manager as mm


@pytest.mark.parametrize(
    "name",
    [
        "../../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\evil",
        "/etc/shadow",
        "a/b/c",
        "....//....//x",
        ".hidden",
        "",
    ],
)
def test_meta_path_stays_within_downloads_dir(name):
    base = os.path.abspath(mm.DOWNLOADS_META_DIR)
    resolved = mm._meta_path(name)
    assert resolved == base or resolved.startswith(base + os.sep)
    # The result is a single file directly inside the metadata dir — no
    # separators survived sanitization, so it cannot be a nested/traversed path.
    assert os.path.dirname(resolved) == base


def test_safe_model_key_neutralizes_separators_and_dots():
    assert "/" not in mm._safe_model_key("a/b")
    assert "\\" not in mm._safe_model_key("a\\b")
    assert not mm._safe_model_key("...evil").startswith(".")
    assert mm._safe_model_key("") == "model"


def test_save_and_load_roundtrip_uses_same_key(tmp_path, monkeypatch):
    monkeypatch.setattr(mm, "DOWNLOADS_META_DIR", str(tmp_path))
    progress = mm.DownloadProgress(model_name="whisper/tiny", status="completed", percent=100.0)
    mm._save_download_meta("whisper/tiny", progress)
    loaded = mm._load_download_meta("whisper/tiny")
    assert loaded is not None
    assert loaded["status"] == "completed"
    # Written file is inside the metadata dir, not a traversed location.
    written = list(tmp_path.iterdir())
    assert len(written) == 1
    assert written[0].suffix == ".json"


def test_save_meta_does_not_raise_on_bad_name(tmp_path, monkeypatch):
    monkeypatch.setattr(mm, "DOWNLOADS_META_DIR", str(tmp_path))
    # A name that sanitizes to something writable must still round-trip; a
    # persistence failure must be swallowed-with-warning, never propagated.
    mm._save_download_meta("../../escape", mm.DownloadProgress(model_name="x"))
    # File landed inside tmp_path, not two levels up.
    assert not (tmp_path.parent.parent / "escape.json").exists()
