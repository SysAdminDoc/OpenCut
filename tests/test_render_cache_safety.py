"""Regression tests for render-cache forged-index containment."""

from __future__ import annotations

import os


def _isolate_cache(monkeypatch, tmp_path):
    import opencut.core.render_cache as cache

    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(cache, "CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(cache, "CACHE_INDEX", str(cache_dir / "index.json"))
    return cache, cache_dir


def test_get_cached_rejects_forged_output_path(monkeypatch, tmp_path):
    cache, _cache_dir = _isolate_cache(monkeypatch, tmp_path)
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"outside")
    key = cache._cache_key("input", "op", {"q": 1})
    cache._save_index({
        key: {
            "cache_key": key,
            "input_hash": "input",
            "operation": "op",
            "params_hash": cache._compute_hash({"q": 1}),
            "output_path": str(outside),
            "file_size": outside.stat().st_size,
            "created_at": 1,
            "last_accessed": 1,
            "hit_count": 0,
            "dependencies": [],
        }
    })

    assert cache.get_cached("input", "op", {"q": 1}) is None
    assert outside.exists()
    assert cache._load_index() == {}


def test_cleanup_cache_skips_forged_output_path(monkeypatch, tmp_path):
    cache, cache_dir = _isolate_cache(monkeypatch, tmp_path)
    outside = tmp_path / "outside.mov"
    outside.write_bytes(b"outside")
    valid_key = "a" * 32
    invalid_key = "b" * 32
    valid = cache_dir / f"{valid_key}.mp4"
    cache_dir.mkdir(parents=True)
    valid.write_bytes(b"valid")
    cache._save_index({
        invalid_key: {
            "cache_key": invalid_key,
            "output_path": str(outside),
            "file_size": outside.stat().st_size,
            "last_accessed": 1,
        },
        valid_key: {
            "cache_key": valid_key,
            "output_path": str(valid),
            "file_size": valid.stat().st_size,
            "last_accessed": 2,
        },
    })

    result = cache.cleanup_cache(max_size_gb=0)

    assert result["removed"] == 2
    assert result["invalid_entries"] == 1
    assert outside.exists()
    assert not valid.exists()
    assert cache._load_index() == {}


def test_invalidate_downstream_skips_forged_seed_path(monkeypatch, tmp_path):
    cache, cache_dir = _isolate_cache(monkeypatch, tmp_path)
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"outside")
    seed_key = "c" * 32
    child_key = "d" * 32
    child = cache_dir / f"{child_key}.mp4"
    cache_dir.mkdir(parents=True)
    child.write_bytes(b"child")
    cache._save_index({
        seed_key: {
            "cache_key": seed_key,
            "input_hash": "clip-hash",
            "operation": "encode",
            "output_path": str(outside),
            "file_size": outside.stat().st_size,
            "dependencies": [],
        },
        child_key: {
            "cache_key": child_key,
            "input_hash": "child-hash",
            "operation": "thumbnail",
            "output_path": str(child),
            "file_size": child.stat().st_size,
            "dependencies": [seed_key],
        },
    })

    result = cache.invalidate_downstream("clip-hash", "encode")

    assert result["invalidated"] == 2
    assert result["invalid_entries"] == 1
    assert outside.exists()
    assert not child.exists()
    assert cache._load_index() == {}


def test_cache_file_must_match_cache_key_basename(monkeypatch, tmp_path):
    cache, cache_dir = _isolate_cache(monkeypatch, tmp_path)
    key = "e" * 32
    wrong_name = cache_dir / "different.mp4"
    cache_dir.mkdir(parents=True)
    wrong_name.write_bytes(b"inside but wrong")

    size, safe = cache._safe_unlink_cached_file(key, os.fspath(wrong_name))

    assert (size, safe) == (0, False)
    assert wrong_name.exists()


def test_zero_byte_cache_file_is_removed_not_marked_missing(monkeypatch, tmp_path):
    cache, cache_dir = _isolate_cache(monkeypatch, tmp_path)
    key = "f" * 32
    cached = cache_dir / f"{key}.mp4"
    cache_dir.mkdir(parents=True)
    cached.write_bytes(b"")
    cache._save_index({
        key: {
            "cache_key": key,
            "output_path": str(cached),
            "file_size": 1,
            "last_accessed": 1,
        }
    })

    result = cache.cleanup_cache(max_size_gb=0)

    assert result["removed"] == 1
    assert result["freed_bytes"] == 0
    assert result["missing_entries"] == 0
    assert not cached.exists()
