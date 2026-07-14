"""Bounded, path-safe, transactional ZIP handling (opencut.core.archive_safety)."""

import os
import stat
import zipfile

import pytest

from opencut.core import archive_safety
from opencut.core.archive_safety import (
    ArchiveError,
    inspect_members,
    normalize_member,
    safe_extract_all,
    safe_read_member,
)


# --- normalize_member ------------------------------------------------------

@pytest.mark.parametrize("name", ["/etc/passwd", "\\windows", "C:/x", "../evil", "a/../../b"])
def test_normalize_member_rejects_unsafe(name):
    with pytest.raises(ArchiveError):
        normalize_member(name)


def test_normalize_member_allows_nested_relative():
    assert normalize_member("a/b/c.txt") == "a/b/c.txt"
    assert normalize_member("./") == ""


# --- inspect_members -------------------------------------------------------

def _write_zip(path, entries):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in entries:
            zf.writestr(name, data)


def test_inspect_rejects_too_many_members(tmp_path):
    z = tmp_path / "many.zip"
    _write_zip(z, [(f"f{i}.txt", b"x") for i in range(10)])
    with zipfile.ZipFile(z) as zf:
        with pytest.raises(ArchiveError):
            inspect_members(zf, max_members=5)


def test_inspect_rejects_expansion_bomb(tmp_path):
    z = tmp_path / "bomb.zip"
    _write_zip(z, [("big.txt", b"x" * 4000)])
    with zipfile.ZipFile(z) as zf:
        with pytest.raises(ArchiveError):
            inspect_members(zf, max_total_bytes=1000)


def test_inspect_rejects_oversized_member(tmp_path):
    z = tmp_path / "big.zip"
    _write_zip(z, [("big.txt", b"x" * 4000)])
    with zipfile.ZipFile(z) as zf:
        with pytest.raises(ArchiveError):
            inspect_members(zf, max_member_bytes=1000)


def test_inspect_rejects_symlink_entry(tmp_path):
    z = tmp_path / "link.zip"
    with zipfile.ZipFile(z, "w") as zf:
        info = zipfile.ZipInfo("link")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(info, "/etc/passwd")
    with zipfile.ZipFile(z) as zf:
        with pytest.raises(ArchiveError):
            inspect_members(zf)


# --- safe_read_member ------------------------------------------------------

def test_safe_read_member_enforces_ceiling(tmp_path):
    z = tmp_path / "r.zip"
    _write_zip(z, [("doc.json", b"y" * 5000)])
    with zipfile.ZipFile(z) as zf:
        assert safe_read_member(zf, "doc.json", max_bytes=10000) == b"y" * 5000
        with pytest.raises(ArchiveError):
            safe_read_member(zf, "doc.json", max_bytes=100)


# --- safe_extract_all ------------------------------------------------------

def test_safe_extract_happy_path(tmp_path):
    z = tmp_path / "ok.zip"
    _write_zip(z, [("a.txt", b"1"), ("sub/b.txt", b"2")])
    dest = tmp_path / "out"
    count = safe_extract_all(str(z), str(dest))
    assert count == 2
    assert (dest / "a.txt").read_bytes() == b"1"
    assert (dest / "sub" / "b.txt").read_bytes() == b"2"


def test_safe_extract_rejects_traversal_without_partial_dest(tmp_path):
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("good.txt", b"ok")
        zf.writestr("../escape.txt", b"bad")
    dest = tmp_path / "out"
    with pytest.raises(ArchiveError):
        safe_extract_all(str(z), str(dest))
    # No partial destination, and no leftover staging dir.
    assert not dest.exists()
    leftovers = [p for p in os.listdir(tmp_path) if p.startswith(".octmp_restore_")]
    assert leftovers == []


def test_safe_extract_promotes_atomically_over_existing(tmp_path):
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "old.txt").write_text("stale")
    z = tmp_path / "new.zip"
    _write_zip(z, [("fresh.txt", b"new")])
    safe_extract_all(str(z), str(dest))
    assert (dest / "fresh.txt").read_bytes() == b"new"
    assert not (dest / "old.txt").exists()  # replaced, not merged


def test_safe_extract_bomb_leaves_dest_untouched(tmp_path):
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "keep.txt").write_text("keep")
    z = tmp_path / "bomb.zip"
    _write_zip(z, [("big.txt", b"x" * 4000)])
    with pytest.raises(ArchiveError):
        safe_extract_all(str(z), str(dest), max_total_bytes=1000)
    assert (dest / "keep.txt").read_text() == "keep"


def test_safe_extract_missing_archive(tmp_path):
    with pytest.raises(FileNotFoundError):
        safe_extract_all(str(tmp_path / "nope.zip"), str(tmp_path / "out"))


# --- integration: project restore + lottie use the shared guard ------------

def test_project_restore_rejects_bomb(tmp_path, monkeypatch):
    from opencut.core import project_archive

    z = tmp_path / "proj.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr(project_archive._MANIFEST_NAME, '{"version": 1}')
        zf.writestr("payload.bin", b"x" * 4000)
    dest = tmp_path / "restored"
    # Shrink the ceiling so the payload trips the guard.
    monkeypatch.setattr(archive_safety, "DEFAULT_MAX_TOTAL_BYTES", 1000)

    def tiny_extract(archive, d, **kw):
        kw["max_total_bytes"] = 1000
        return _orig(archive, d, **kw)

    _orig = archive_safety.safe_extract_all
    monkeypatch.setattr(project_archive.archive_safety, "safe_extract_all", tiny_extract)
    with pytest.raises(ArchiveError):
        project_archive.restore_archive(str(z), str(dest))
    assert not dest.exists()
