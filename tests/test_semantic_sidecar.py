"""Portable semantic-search sidecar (opencut.core.semantic_video_search).

The sidecar stores CLIP embeddings keyed by a move-stable content signature, so
a clip that is relocated or relinked is deterministically reused, while a clip
whose content changed is deterministically invalidated. Pure file I/O — no CLIP
model, GPU, or Adobe host required.
"""

import os
import shutil

import numpy as np
import pytest

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


# --- index and search with a stubbed CLIP pipeline -------------------------
# No model, torch or ffmpeg: frame extraction writes real JPEGs into the work
# dir it is given (which the indexer deletes), and embeddings are fixed.

_FRAME_EMBEDDINGS = np.array([[1, 0, 0, 0], [0.6, 0.8, 0, 0]], dtype="float32")
_QUERY_EMBEDDING = np.array([[1, 0, 0, 0]], dtype="float32")
_JPEG_PREFIX = "data:image/jpeg;base64,"


def _stub_clip_pipeline(monkeypatch, cache_dir, extract_calls):
    from PIL import Image

    monkeypatch.setattr(svs, "CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(
        svs, "_load_clip_model",
        lambda engine=svs.DEFAULT_ENGINE, on_progress=None: (None, None, None, {"embed_dim": 4}),
    )

    def fake_extract(video_path, out_dir, count=svs.FRAMES_PER_CLIP):
        extract_calls.append(video_path)
        os.makedirs(out_dir, exist_ok=True)
        frames = []
        for idx in range(len(_FRAME_EMBEDDINGS)):
            frame = os.path.join(out_dir, f"kf_{idx + 1:06d}.jpg")
            Image.new("RGB", (320, 180), (60 * idx, 90, 150)).save(frame, "JPEG")
            frames.append(frame)
        return frames, [0.0, 2.5]

    monkeypatch.setattr(svs, "_extract_key_frames", fake_extract)
    monkeypatch.setattr(
        svs, "_compute_frame_embeddings",
        lambda frames, *args, **kwargs: _FRAME_EMBEDDINGS[: len(frames)].copy(),
    )
    monkeypatch.setattr(svs, "_compute_text_embedding", lambda *args, **kwargs: _QUERY_EMBEDDING)
    monkeypatch.setattr(svs, "get_video_info", lambda _path: {"duration": 5.0, "fps": 25.0})


def _assert_live_thumbnail(result):
    import base64

    from PIL import Image

    assert result["thumbnail_path"], result
    assert os.path.isfile(result["thumbnail_path"])
    with Image.open(result["thumbnail_path"]) as img:
        assert img.width == svs.PERSISTED_THUMB_WIDTH
    assert result["thumbnail"].startswith(_JPEG_PREFIX)
    assert base64.b64decode(result["thumbnail"][len(_JPEG_PREFIX):]).startswith(b"\xff\xd8")


def test_index_then_search_returns_live_thumbnails(tmp_path, monkeypatch):
    calls = []
    _stub_clip_pipeline(monkeypatch, tmp_path / "cache", calls)
    clip = _write_clip(str(tmp_path / "media" / "shot.mp4"), b"shot " * 400)

    svs.build_clip_index([clip])
    found = svs.semantic_search([clip], query="a dog", auto_index=False)

    assert [r["timestamp"] for r in found["results"]] == [0.0, 2.5]
    for result in found["results"]:
        _assert_live_thumbnail(result)


def test_search_auto_index_returns_live_thumbnails(tmp_path, monkeypatch):
    calls = []
    _stub_clip_pipeline(monkeypatch, tmp_path / "cache", calls)
    clip = _write_clip(str(tmp_path / "media" / "shot.mp4"), b"shot " * 400)

    found = svs.semantic_search([clip], query="a dog")

    assert calls == [clip]
    assert len(found["results"]) == 2
    for result in found["results"]:
        _assert_live_thumbnail(result)


def test_old_cache_with_deleted_frames_reports_no_thumbnail(tmp_path, monkeypatch):
    calls = []
    _stub_clip_pipeline(monkeypatch, tmp_path / "cache", calls)
    clip = _write_clip(str(tmp_path / "media" / "shot.mp4"), b"shot " * 400)
    gone = str(tmp_path / "clipidx_0_gone")
    svs._save_cached_embeddings(svs._clip_file_hash(clip), {
        "clip_path": clip,
        "frame_count": 2,
        "fps": 25.0,
        "timestamps": [0.0, 2.5],
        "frame_paths": [os.path.join(gone, "kf_000001.jpg"), os.path.join(gone, "kf_000002.jpg")],
        "embeddings": _FRAME_EMBEDDINGS.copy(),
    })

    found = svs.semantic_search([clip], query="a dog", auto_index=False)

    assert calls == []
    assert len(found["results"]) == 2
    assert all(r["thumbnail_path"] == "" and r["thumbnail"] == "" for r in found["results"])


def test_search_reuses_sidecar_for_moved_clip_without_reindexing(tmp_path, monkeypatch):
    calls = []
    _stub_clip_pipeline(monkeypatch, tmp_path / "cache_a", calls)
    project = str(tmp_path / "proj")
    os.makedirs(project)
    clip = _write_clip(str(tmp_path / "media" / "shot.mp4"), b"shot " * 400)
    svs.build_clip_index([clip], project_dir=project)
    assert calls == [clip]

    # The project travels to another machine (empty local cache) and the
    # media is relinked from a new folder under a new name.
    monkeypatch.setattr(svs, "CACHE_DIR", str(tmp_path / "cache_b"))
    moved = str(tmp_path / "relinked" / "shot_final.mp4")
    os.makedirs(os.path.dirname(moved))
    shutil.move(clip, moved)

    found = svs.semantic_search([moved], query="a dog", auto_index=True, project_dir=project)

    assert calls == [clip]  # no re-index at search time
    assert found["index_cached"] is True
    assert [r["clip_path"] for r in found["results"]] == [moved, moved]
    for result in found["results"]:
        _assert_live_thumbnail(result)
        assert result["thumbnail_path"].startswith(str(tmp_path / "cache_b"))


def test_search_auto_index_saves_to_sidecar(tmp_path, monkeypatch):
    calls = []
    _stub_clip_pipeline(monkeypatch, tmp_path / "cache", calls)
    project = str(tmp_path / "proj")
    os.makedirs(project)
    clip = _write_clip(str(tmp_path / "media" / "shot.mp4"), b"shot " * 400)

    svs.semantic_search([clip], query="a dog", project_dir=project)

    entry = svs.load_sidecar_embeddings(project, clip, svs.DEFAULT_ENGINE)
    assert entry is not None
    assert entry["frame_count"] == 2
    sidecar_root = os.path.join(project, svs.SIDECAR_DIRNAME)
    assert len(entry["frame_paths"]) == 2
    for path in entry["frame_paths"]:
        assert os.path.isfile(path)
        assert path.startswith(sidecar_root)


def test_sidecar_thumbnail_names_cannot_escape_the_sidecar(tmp_path):
    import json

    project = str(tmp_path / "proj")
    os.makedirs(project)
    clip = _write_clip(str(tmp_path / "clip.mp4"), b"data " * 200)
    path = svs.save_sidecar_embeddings(project, clip, _fake_embeddings(clip), svs.DEFAULT_ENGINE)
    with np.load(path, allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata"].item()))
        embeddings = cache["embeddings"]
    metadata["thumb_names"] = ["../../../secret.jpg", "thumb_0001.jpg"]
    metadata["frame_paths"] = ["C:/Users/someone/secret.jpg"]
    np.savez_compressed(path, metadata=json.dumps(metadata), embeddings=embeddings)

    loaded = svs.load_sidecar_embeddings(project, clip, svs.DEFAULT_ENGINE)

    assert loaded["frame_paths"][0] == ""
    assert os.path.basename(loaded["frame_paths"][1]) == "thumb_0001.jpg"
    assert os.path.dirname(loaded["frame_paths"][1]).startswith(os.path.join(project, svs.SIDECAR_DIRNAME))


def test_thumbnail_data_uri_only_reads_cache_jpegs(tmp_path, monkeypatch):
    from PIL import Image

    monkeypatch.setattr(svs, "CACHE_DIR", str(tmp_path / "cache"))
    inside = tmp_path / "cache" / "thumb_0000.jpg"
    outside = tmp_path / "elsewhere.jpg"
    os.makedirs(inside.parent)
    Image.new("RGB", (16, 9)).save(inside, "JPEG")
    Image.new("RGB", (16, 9)).save(outside, "JPEG")
    not_jpeg = tmp_path / "cache" / "notes.jpg"
    not_jpeg.write_bytes(b"plain text, not an image")

    assert svs._thumbnail_data_uri(str(inside)).startswith(_JPEG_PREFIX)
    assert svs._thumbnail_data_uri(str(outside)) == ""
    assert svs._thumbnail_data_uri(str(not_jpeg)) == ""
    assert svs._thumbnail_data_uri("") == ""


# --- one unreadable clip or an unwritable project ---------------------------
# A project with one offline, damaged or unsupported clip, or a project folder
# the index can't be written to, still searches every clip it can read.

def _break_extract_for(monkeypatch, bad_clip):
    working = svs._extract_key_frames

    def extract(video_path, out_dir, count=svs.FRAMES_PER_CLIP):
        if video_path == bad_clip:
            raise RuntimeError("FFmpeg error (code 1): moov atom not found")
        return working(video_path, out_dir, count=count)

    monkeypatch.setattr(svs, "_extract_key_frames", extract)


def _refuse_sidecar_writes(monkeypatch):
    attempts = []

    def refuse(project_dir, clip_path, data, engine=svs.DEFAULT_ENGINE):
        attempts.append(clip_path)
        raise PermissionError(13, "Access is denied", os.path.join(project_dir, svs.SIDECAR_DIRNAME))

    monkeypatch.setattr(svs, "save_sidecar_embeddings", refuse)
    return attempts


def test_index_skips_a_clip_ffmpeg_cannot_read(tmp_path, monkeypatch):
    _stub_clip_pipeline(monkeypatch, tmp_path / "cache", [])
    bad = _write_clip(str(tmp_path / "media" / "bad.mp4"), b"broken " * 400)
    good = _write_clip(str(tmp_path / "media" / "good.mp4"), b"shot " * 400)
    _break_extract_for(monkeypatch, bad)

    result = svs.build_clip_index([bad, good])

    assert result["clip_count"] == 1
    assert result["rebuilt_count"] == 1
    assert [s["clip_name"] for s in result["skipped_clips"]] == ["bad.mp4"]
    assert result["skipped_clips"][0]["clip_path"] == os.path.abspath(bad)
    assert "moov atom not found" in result["skipped_clips"][0]["reason"]


def test_search_skips_a_clip_ffmpeg_cannot_read(tmp_path, monkeypatch):
    _stub_clip_pipeline(monkeypatch, tmp_path / "cache", [])
    bad = _write_clip(str(tmp_path / "media" / "bad.mp4"), b"broken " * 400)
    good = _write_clip(str(tmp_path / "media" / "good.mp4"), b"shot " * 400)
    _break_extract_for(monkeypatch, bad)

    found = svs.semantic_search([bad, good], query="a dog")

    assert [r["clip_path"] for r in found["results"]] == [good, good]
    assert found["total_clips_searched"] == 1
    assert [s["clip_name"] for s in found["skipped_clips"]] == ["bad.mp4"]


@pytest.mark.parametrize("run", [
    lambda clip: svs.build_clip_index([clip]),
    lambda clip: svs.semantic_search([clip], query="a dog"),
], ids=["index", "search"])
def test_job_fails_with_the_reason_when_no_clip_can_be_read(tmp_path, monkeypatch, run):
    _stub_clip_pipeline(monkeypatch, tmp_path / "cache", [])
    bad = _write_clip(str(tmp_path / "media" / "bad.mp4"), b"broken " * 400)
    _break_extract_for(monkeypatch, bad)

    with pytest.raises(RuntimeError, match=r"None of the clips could be read\. bad\.mp4: .*moov atom not found"):
        run(bad)


def test_read_only_project_folder_still_builds_the_local_index(tmp_path, monkeypatch):
    _stub_clip_pipeline(monkeypatch, tmp_path / "cache", [])
    project = str(tmp_path / "proj")
    os.makedirs(project)
    first = _write_clip(str(tmp_path / "media" / "a.mp4"), b"first " * 400)
    second = _write_clip(str(tmp_path / "media" / "b.mp4"), b"second " * 400)
    attempts = _refuse_sidecar_writes(monkeypatch)

    result = svs.build_clip_index([first, second], project_dir=project)

    assert result["clip_count"] == 2
    assert "Access is denied" in result["sidecar_error"]
    assert attempts == [first]  # one refusal stops the retries for this run
    found = svs.semantic_search([first, second], query="a dog", auto_index=False)
    assert len(found["results"]) == 4  # both clips came from the local cache


def test_read_only_project_folder_still_returns_search_results(tmp_path, monkeypatch):
    _stub_clip_pipeline(monkeypatch, tmp_path / "cache", [])
    project = str(tmp_path / "proj")
    os.makedirs(project)
    clip = _write_clip(str(tmp_path / "media" / "shot.mp4"), b"shot " * 400)
    _refuse_sidecar_writes(monkeypatch)

    found = svs.semantic_search([clip], query="a dog", project_dir=project)

    assert len(found["results"]) == 2
    assert "Access is denied" in found["sidecar_error"]


def _route_body(name):
    from opencut.routes import object_intel_routes

    return getattr(object_intel_routes, name).__wrapped__.__wrapped__


@pytest.mark.parametrize("route,core,data", [
    ("semantic_search_route", "semantic_search", {"query": "a dog"}),
    ("semantic_index_route", "build_clip_index", {}),
], ids=["search", "index"])
def test_semantic_routes_skip_a_missing_clip(tmp_path, monkeypatch, route, core, data):
    from opencut.security import validate_filepath

    present = _write_clip(str(tmp_path / "media" / "here.mp4"), b"shot " * 400)
    offline = str(tmp_path / "offline" / "gone.mp4")
    seen = []

    def fake_core(clip_paths, **kwargs):
        seen.append(clip_paths)
        return {"skipped_clips": [{"clip_path": "damaged.mp4", "clip_name": "damaged.mp4", "reason": "core"}]}

    monkeypatch.setattr(svs, core, fake_core)

    result = _route_body(route)("job_test", "", {"clip_paths": [offline, present], **data})

    assert seen == [[validate_filepath(present)]]
    assert [s["clip_name"] for s in result["skipped_clips"]] == ["gone.mp4", "damaged.mp4"]
    assert result["skipped_clips"][0]["clip_path"] == offline
    assert "File not found" in result["skipped_clips"][0]["reason"]


@pytest.mark.parametrize("route,data", [
    ("semantic_search_route", {"query": "a dog"}),
    ("semantic_index_route", {}),
], ids=["search", "index"])
def test_semantic_routes_name_the_reason_when_no_clip_exists(tmp_path, route, data):
    offline = str(tmp_path / "offline" / "gone.mp4")

    with pytest.raises(ValueError, match=r"None of the clips can be read\. File not found"):
        _route_body(route)("job_test", "", {"clip_paths": [offline], **data})
