"""Portable semantic-search sidecar (opencut.core.semantic_video_search).

The sidecar stores CLIP embeddings keyed by a move-stable content signature, so
a clip that is relocated or relinked is deterministically reused, while a clip
whose content changed is deterministically invalidated. Pure file I/O — no CLIP
model, GPU, or Adobe host required.
"""

import os
import shutil

import numpy as np

from opencut.core import semantic_video_search as svs


def _write_clip(path, payload: bytes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(payload)
    return path


def _fake_embeddings(clip_path, frames=3, dim=512):
    return {
        "clip_path": os.path.abspath(clip_path),
        "frame_count": frames,
        "fps": 30.0,
        "duration": 4.0,
        "timestamps": [0.0, 1.0, 2.0],
        "embeddings": np.ones((frames, dim), dtype="float32"),
        "engine": svs.DEFAULT_ENGINE,
    }


# --- content signature ----------------------------------------------------

def test_signature_is_move_stable(tmp_path):
    a = _write_clip(str(tmp_path / "a.mp4"), b"OpenCut clip bytes " * 100)
    moved = str(tmp_path / "sub" / "renamed.mp4")
    os.makedirs(os.path.dirname(moved))
    shutil.copyfile(a, moved)
    assert svs.content_signature(a) == svs.content_signature(moved)


def test_signature_changes_with_content(tmp_path):
    a = _write_clip(str(tmp_path / "a.mp4"), b"original bytes " * 100)
    sig1 = svs.content_signature(a)
    _write_clip(a, b"different bytes entirely " * 100)
    assert svs.content_signature(a) != sig1


# --- save / load / relink -------------------------------------------------

def test_moved_clip_is_deterministically_reused(tmp_path):
    project = str(tmp_path / "proj")
    os.makedirs(project)
    clip = _write_clip(str(tmp_path / "media" / "shot.mp4"), b"shot data " * 500)

    svs.save_sidecar_embeddings(project, clip, _fake_embeddings(clip), svs.DEFAULT_ENGINE)

    # Relocate the clip to a new folder (relink) — same bytes, new path.
    moved = str(tmp_path / "media2" / "shot_final.mp4")
    os.makedirs(os.path.dirname(moved))
    shutil.move(clip, moved)

    loaded = svs.load_sidecar_embeddings(project, moved, svs.DEFAULT_ENGINE)
    assert loaded is not None
    assert loaded["frame_count"] == 3
    assert np.array_equal(loaded["embeddings"], np.ones((3, 512), dtype="float32"))
    assert svs.sidecar_relink_status(project, moved) == "reused"


def test_changed_content_is_invalidated(tmp_path):
    project = str(tmp_path / "proj")
    os.makedirs(project)
    clip = _write_clip(str(tmp_path / "shot.mp4"), b"take one " * 500)
    svs.save_sidecar_embeddings(project, clip, _fake_embeddings(clip), svs.DEFAULT_ENGINE)

    # Re-shoot: same filename, new content.
    _write_clip(clip, b"take two totally different " * 500)
    assert svs.load_sidecar_embeddings(project, clip, svs.DEFAULT_ENGINE) is None
    assert svs.sidecar_relink_status(project, clip) == "invalidated"


def test_unknown_clip_is_missing(tmp_path):
    project = str(tmp_path / "proj")
    os.makedirs(project)
    clip = _write_clip(str(tmp_path / "never_indexed.mp4"), b"bytes " * 100)
    assert svs.load_sidecar_embeddings(project, clip, svs.DEFAULT_ENGINE) is None
    assert svs.sidecar_relink_status(project, clip) == "missing"


# --- portability / versioning ---------------------------------------------

def test_sidecar_lives_under_project_and_is_versioned(tmp_path):
    project = str(tmp_path / "proj")
    os.makedirs(project)
    clip = _write_clip(str(tmp_path / "clip.mp4"), b"data " * 200)
    svs.save_sidecar_embeddings(project, clip, _fake_embeddings(clip), svs.DEFAULT_ENGINE)

    root = os.path.join(project, svs.SIDECAR_DIRNAME)
    assert os.path.isdir(root)  # travels with the project
    manifest = svs._read_sidecar_manifest(project)
    assert manifest["version"] == svs.SIDECAR_VERSION
    assert len(manifest["entries"]) == 1
    assert manifest["entries"][0]["clip_name"] == "clip.mp4"


def test_different_engines_use_separate_stores(tmp_path):
    project = str(tmp_path / "proj")
    os.makedirs(project)
    clip = _write_clip(str(tmp_path / "clip.mp4"), b"data " * 200)
    svs.save_sidecar_embeddings(project, clip, _fake_embeddings(clip), "clip-vit-b32")

    # A different engine has no entry for this clip yet.
    assert svs.load_sidecar_embeddings(project, clip, "clip-vit-l14") is None
    assert svs.load_sidecar_embeddings(project, clip, "clip-vit-b32") is not None


def test_resave_replaces_entry_not_duplicates(tmp_path):
    project = str(tmp_path / "proj")
    os.makedirs(project)
    clip = _write_clip(str(tmp_path / "clip.mp4"), b"data " * 200)
    svs.save_sidecar_embeddings(project, clip, _fake_embeddings(clip), svs.DEFAULT_ENGINE)
    svs.save_sidecar_embeddings(project, clip, _fake_embeddings(clip), svs.DEFAULT_ENGINE)
    manifest = svs._read_sidecar_manifest(project)
    assert len(manifest["entries"]) == 1  # same signature -> single entry
