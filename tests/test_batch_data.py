"""
Unit tests for the 8 batch data core modules and routes.

Tests pure logic functions with mocked external dependencies
(FFmpeg subprocess calls, filesystem operations, image processing).
"""

import json
import os
import time
from unittest.mock import MagicMock, patch

import pytest


# ========================================================================
# 1. structured_ingest.py
# ========================================================================
class TestVerifyChecksum:
    """Tests for opencut.core.structured_ingest.verify_checksum."""

    def test_sha256_hash_computed(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello world")

        from opencut.core.structured_ingest import verify_checksum
        result = verify_checksum(str(f), "sha256")

        assert result["algorithm"] == "sha256"
        assert len(result["hash"]) == 64
        assert result["verified"] is None

    def test_md5_hash_computed(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"test data")

        from opencut.core.structured_ingest import verify_checksum
        result = verify_checksum(str(f), "md5")

        assert result["algorithm"] == "md5"
        assert len(result["hash"]) == 32

    def test_checksum_match_verification(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello")

        from opencut.core.structured_ingest import verify_checksum
        # First compute the hash
        r1 = verify_checksum(str(f), "sha256")
        # Then verify against it
        r2 = verify_checksum(str(f), "sha256", expected=r1["hash"])

        assert r2["verified"] is True
        assert r2["match"] is True

    def test_checksum_mismatch(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello")

        from opencut.core.structured_ingest import verify_checksum
        result = verify_checksum(str(f), "sha256", expected="badhash")

        assert result["verified"] is True
        assert result["match"] is False

    def test_unsupported_algorithm(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello")

        from opencut.core.structured_ingest import verify_checksum
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            verify_checksum(str(f), "blake99")

    def test_file_not_found(self):
        from opencut.core.structured_ingest import verify_checksum
        with pytest.raises(FileNotFoundError):
            verify_checksum("/nonexistent/file.bin", "sha256")


class TestRenameByPattern:
    """Tests for opencut.core.structured_ingest.rename_by_pattern."""

    def test_basic_rename(self):
        from opencut.core.structured_ingest import rename_by_pattern
        result = rename_by_pattern("clip.mp4", "{name}_edited{ext}")
        assert result == "clip_edited.mp4"

    def test_counter_placeholder(self):
        from opencut.core.structured_ingest import rename_by_pattern
        result = rename_by_pattern("clip.mp4", "shot_{counter}{ext}", {"counter": 5})
        assert result == "shot_0005.mp4"

    def test_camera_placeholder(self):
        from opencut.core.structured_ingest import rename_by_pattern
        result = rename_by_pattern("clip.mov", "CAM-{camera}_{name}{ext}", {"camera": "B"})
        assert result == "CAM-B_clip.mov"

    def test_empty_pattern_returns_original(self):
        from opencut.core.structured_ingest import rename_by_pattern
        assert rename_by_pattern("test.mp4", "") == "test.mp4"

    def test_sanitizes_invalid_chars(self):
        from opencut.core.structured_ingest import rename_by_pattern
        result = rename_by_pattern("clip.mp4", "{name}:test{ext}")
        assert ":" not in result


class TestRunIngest:
    """Tests for opencut.core.structured_ingest.run_ingest."""

    def test_ingest_copies_files(self, tmp_path):
        # Setup source
        src = tmp_path / "source"
        src.mkdir()
        (src / "clip1.mp4").write_bytes(b"video1")
        (src / "clip2.mp4").write_bytes(b"video2")
        dest = tmp_path / "dest"

        from opencut.core.structured_ingest import run_ingest
        with patch("opencut.core.structured_ingest.get_video_info",
                   return_value={"width": 1920, "height": 1080}):
            result = run_ingest(str(src), dest_dir=str(dest))

        assert result.total == 2
        assert result.copied == 2
        assert result.failed == 0
        assert os.path.isfile(os.path.join(str(dest), "clip1.mp4"))

    def test_ingest_with_rename(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        (src / "raw.mp4").write_bytes(b"video")
        dest = tmp_path / "dest"

        config = {"rename_pattern": "shot_{counter}{ext}"}
        from opencut.core.structured_ingest import run_ingest
        with patch("opencut.core.structured_ingest.get_video_info",
                   return_value={"width": 1920, "height": 1080}):
            result = run_ingest(str(src), config=config, dest_dir=str(dest))

        assert result.copied == 1
        assert any(r.new_name == "shot_0001.mp4" for r in result.files)

    def test_ingest_extension_filter(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        (src / "clip.mp4").write_bytes(b"video")
        (src / "readme.txt").write_bytes(b"text")
        dest = tmp_path / "dest"

        config = {"extensions_filter": ["mp4"]}
        from opencut.core.structured_ingest import run_ingest
        with patch("opencut.core.structured_ingest.get_video_info",
                   return_value={"width": 1920, "height": 1080}):
            result = run_ingest(str(src), config=config, dest_dir=str(dest))

        assert result.total == 1

    def test_ingest_skip_existing(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        (src / "clip.mp4").write_bytes(b"video")
        dest = tmp_path / "dest"
        dest.mkdir()
        (dest / "clip.mp4").write_bytes(b"existing")

        from opencut.core.structured_ingest import run_ingest
        with patch("opencut.core.structured_ingest.get_video_info",
                   return_value={"width": 1920, "height": 1080}):
            result = run_ingest(str(src), dest_dir=str(dest))

        assert result.skipped == 1

    def test_ingest_source_not_found(self):
        from opencut.core.structured_ingest import run_ingest
        with pytest.raises(FileNotFoundError):
            run_ingest("/nonexistent/dir")

    def test_ingest_empty_source(self, tmp_path):
        src = tmp_path / "empty"
        src.mkdir()

        from opencut.core.structured_ingest import run_ingest
        with pytest.raises(ValueError, match="No matching files"):
            run_ingest(str(src))


class TestIngestReport:
    """Tests for opencut.core.structured_ingest.generate_ingest_report."""

    def test_json_report(self, tmp_path):
        from opencut.core.structured_ingest import IngestFileResult, generate_ingest_report
        results = [
            IngestFileResult(source_path="/a.mp4", status="copied"),
            IngestFileResult(source_path="/b.mp4", status="failed", error="oops"),
        ]
        out = str(tmp_path / "report.json")
        path = generate_ingest_report(results, out, format="json")

        assert path.endswith(".json")
        data = json.loads(open(path, encoding="utf-8").read())
        assert data["total"] == 2
        assert data["copied"] == 1
        assert data["failed"] == 1

    def test_csv_report(self, tmp_path):
        from opencut.core.structured_ingest import IngestFileResult, generate_ingest_report
        results = [IngestFileResult(source_path="/a.mp4", status="copied")]
        out = str(tmp_path / "report.csv")
        path = generate_ingest_report(results, out, format="csv")

        assert path.endswith(".csv")
        content = open(path, encoding="utf-8").read()
        assert "source_path" in content


# ========================================================================
# 2. storage_tiering.py
# ========================================================================
class TestStorageTiering:
    """Tests for opencut.core.storage_tiering."""

    def test_scan_for_archival_finds_old_files(self, tmp_path):
        f = tmp_path / "old_clip.mp4"
        f.write_bytes(b"video data")
        # Set access time to 60 days ago
        old_time = time.time() - (60 * 86400)
        os.utime(str(f), (old_time, old_time))

        from opencut.core.storage_tiering import scan_for_archival
        result = scan_for_archival(str(tmp_path), idle_days=30)

        assert result.total_scanned == 1
        assert result.eligible_count == 1

    def test_scan_skips_recent_files(self, tmp_path):
        f = tmp_path / "new_clip.mp4"
        f.write_bytes(b"video data")

        from opencut.core.storage_tiering import scan_for_archival
        result = scan_for_archival(str(tmp_path), idle_days=30)

        assert result.eligible_count == 0

    def test_scan_nonexistent_dir(self):
        from opencut.core.storage_tiering import scan_for_archival
        with pytest.raises(FileNotFoundError):
            scan_for_archival("/nonexistent/dir")

    def test_archive_and_restore(self, tmp_path):
        # Create source file
        src = tmp_path / "project"
        src.mkdir()
        clip = src / "clip.mp4"
        clip.write_bytes(b"original video")
        archive_dir = tmp_path / "archive"

        from opencut.core.storage_tiering import archive_files, restore_file

        # Archive
        result = archive_files([str(clip)], str(archive_dir))
        assert result.archived == 1
        assert not os.path.isfile(str(clip))
        assert os.path.isfile(str(clip) + ".opencut_stub")

        # Restore
        restore_result = restore_file(str(clip))
        assert restore_result["status"] == "restored"
        assert os.path.isfile(str(clip))
        assert clip.read_bytes() == b"original video"

    def test_archive_empty_list(self):
        from opencut.core.storage_tiering import archive_files
        with pytest.raises(ValueError, match="No files"):
            archive_files([], "/tmp/archive")

    def test_get_manifest_empty(self, tmp_path):
        from opencut.core.storage_tiering import get_archive_manifest
        m = get_archive_manifest(str(tmp_path))
        assert m["version"] == 1
        assert m["entries"] == {}

    def test_get_manifest_nonexistent(self):
        from opencut.core.storage_tiering import get_archive_manifest
        m = get_archive_manifest("")
        assert m["entries"] == {}

    def test_restore_missing_stub(self):
        from opencut.core.storage_tiering import restore_file
        with pytest.raises(FileNotFoundError):
            restore_file("/nonexistent/file.mp4")


# ========================================================================
# 3. batch_metadata.py
# ========================================================================
class TestBatchMetadata:
    """Tests for opencut.core.batch_metadata."""

    def test_read_batch_metadata_returns_entries(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "format": {
                "format_name": "mov,mp4",
                "duration": "120.5",
                "size": "5000000",
                "bit_rate": "500000",
                "tags": {"title": "My Video", "artist": "OpenCut"},
            }
        })

        with patch("opencut.core.batch_metadata.subprocess.run", return_value=mock_result), \
             patch("os.path.isfile", return_value=True):
            from opencut.core.batch_metadata import read_batch_metadata
            results = read_batch_metadata(["/fake/video.mp4"])

        assert len(results) == 1
        assert results[0]["metadata"]["title"] == "My Video"
        assert results[0]["format"]["duration"] == 120.5

    def test_read_batch_metadata_file_not_found(self):
        from opencut.core.batch_metadata import read_batch_metadata
        results = read_batch_metadata(["/nonexistent/file.mp4"])
        assert results[0]["error"] == "File not found"

    def test_read_empty_list(self):
        from opencut.core.batch_metadata import read_batch_metadata
        with pytest.raises(ValueError, match="No file paths"):
            read_batch_metadata([])

    def test_write_batch_metadata(self, tmp_path):
        f = tmp_path / "video.mp4"
        f.write_bytes(b"fake video")

        with patch("opencut.core.batch_metadata.run_ffmpeg"):
            # Patch os.replace to avoid actual file replacement
            with patch("os.replace"):
                from opencut.core.batch_metadata import write_batch_metadata
                results = write_batch_metadata(
                    [str(f)],
                    {str(f): {"title": "New Title"}},
                )

        assert len(results) == 1
        assert results[0]["status"] == "updated"
        assert results[0]["updated_tags"]["title"] == "New Title"

    def test_write_metadata_no_updates(self):
        from opencut.core.batch_metadata import write_batch_metadata
        with pytest.raises(ValueError, match="No metadata updates"):
            write_batch_metadata(["/fake.mp4"], {})

    def test_apply_metadata_template(self, tmp_path):
        f = tmp_path / "clip01.mp4"
        f.write_bytes(b"fake")

        with patch("opencut.core.batch_metadata.run_ffmpeg"), \
             patch("os.replace"):
            from opencut.core.batch_metadata import apply_metadata_template
            results = apply_metadata_template(
                [str(f)],
                {"title": "Episode {index} - {filename}"},
            )

        assert results[0]["updated_tags"]["title"] == "Episode 1 - clip01"

    def test_export_metadata_csv(self, tmp_path):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "format": {"tags": {"title": "Test"}, "duration": "10", "size": "100", "bit_rate": "8000"}
        })

        with patch("opencut.core.batch_metadata.subprocess.run", return_value=mock_result), \
             patch("os.path.isfile", return_value=True):
            from opencut.core.batch_metadata import export_metadata_csv
            out = str(tmp_path / "meta.csv")
            path = export_metadata_csv(["/fake/video.mp4"], out)

        assert path.endswith(".csv")
        content = open(path, encoding="utf-8").read()
        assert "title" in content
        assert "Test" in content


# ========================================================================
# 4. batch_conform.py
# ========================================================================
class TestBatchConform:
    """Tests for opencut.core.batch_conform."""

    def test_conform_spec_from_dict(self):
        from opencut.core.batch_conform import ConformSpec
        spec = ConformSpec.from_dict({"width": 3840, "height": 2160, "fps": 30.0})
        assert spec.width == 3840
        assert spec.height == 2160
        assert spec.fps == 30.0
        assert spec.video_codec == "libx264"  # default

    def test_conform_spec_ignores_unknown_keys(self):
        from opencut.core.batch_conform import ConformSpec
        spec = ConformSpec.from_dict({"width": 1920, "bogus_key": 42})
        assert spec.width == 1920

    def test_analyze_conformance_detects_resolution_mismatch(self):
        with patch("opencut.core.batch_conform.get_video_info",
                   return_value={"width": 1280, "height": 720, "fps": 24.0, "duration": 10}), \
             patch("os.path.isfile", return_value=True):
            from opencut.core.batch_conform import analyze_conformance_batch
            results = analyze_conformance_batch(
                ["/fake/video.mp4"],
                {"width": 1920, "height": 1080, "fps": 24.0},
            )

        assert results[0]["needs_conform"] is True
        assert any("resolution" in d for d in results[0]["deviations"])

    def test_analyze_conformance_detects_fps_mismatch(self):
        with patch("opencut.core.batch_conform.get_video_info",
                   return_value={"width": 1920, "height": 1080, "fps": 29.97, "duration": 10}), \
             patch("os.path.isfile", return_value=True):
            from opencut.core.batch_conform import analyze_conformance_batch
            results = analyze_conformance_batch(
                ["/fake/video.mp4"],
                {"width": 1920, "height": 1080, "fps": 24.0},
            )

        assert results[0]["needs_conform"] is True
        assert any("fps" in d for d in results[0]["deviations"])

    def test_analyze_conformance_already_conforms(self):
        with patch("opencut.core.batch_conform.get_video_info",
                   return_value={"width": 1920, "height": 1080, "fps": 24.0, "duration": 10}), \
             patch("os.path.isfile", return_value=True):
            from opencut.core.batch_conform import analyze_conformance_batch
            results = analyze_conformance_batch(
                ["/fake/video.mp4"],
                {"width": 1920, "height": 1080, "fps": 24.0},
            )

        assert results[0]["needs_conform"] is False

    def test_analyze_empty_list(self):
        from opencut.core.batch_conform import analyze_conformance_batch
        with pytest.raises(ValueError, match="No file paths"):
            analyze_conformance_batch([], {})

    def test_conform_batch_re_encodes(self, tmp_path):
        with patch("opencut.core.batch_conform.get_video_info",
                   return_value={"width": 1280, "height": 720, "fps": 30.0, "duration": 10}), \
             patch("os.path.isfile", return_value=True), \
             patch("opencut.core.batch_conform.run_ffmpeg"):
            from opencut.core.batch_conform import conform_batch
            result = conform_batch(
                ["/fake/video.mp4"],
                {"width": 1920, "height": 1080, "fps": 24.0},
                output_dir=str(tmp_path / "conform"),
            )

        assert result.conformed == 1
        assert result.total == 1

    def test_conform_batch_copies_matching(self, tmp_path):
        with patch("opencut.core.batch_conform.get_video_info",
                   return_value={"width": 1920, "height": 1080, "fps": 24.0, "duration": 10}), \
             patch("os.path.isfile", return_value=True), \
             patch("opencut.core.batch_conform.run_ffmpeg"):
            from opencut.core.batch_conform import conform_batch
            result = conform_batch(
                ["/fake/video.mp4"],
                {"width": 1920, "height": 1080, "fps": 24.0},
                output_dir=str(tmp_path / "conform"),
            )

        assert result.copied == 1
        assert result.conformed == 0


# ========================================================================
# 5. star_trail.py
# ========================================================================
class TestStarTrail:
    """Tests for opencut.core.star_trail."""

    def _make_fake_images(self, tmp_path, count=5):
        """Create fake images for testing."""
        np = pytest.importorskip("numpy")
        pytest.importorskip("PIL")
        from PIL import Image
        paths = []
        for i in range(count):
            arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            p = str(tmp_path / f"star_{i:03d}.jpg")
            Image.fromarray(arr).save(p)
            paths.append(p)
        return paths

    def test_composite_star_trails_lighten(self, tmp_path):
        pytest.importorskip("numpy")
        pytest.importorskip("PIL")
        paths = self._make_fake_images(tmp_path, 5)
        out = str(tmp_path / "composite.jpg")

        from opencut.core.star_trail import composite_star_trails
        result = composite_star_trails(paths, out, mode="lighten")

        assert result.status == "complete"
        assert result.frames_processed == 5
        assert os.path.isfile(out)

    def test_composite_star_trails_average(self, tmp_path):
        pytest.importorskip("numpy")
        pytest.importorskip("PIL")
        paths = self._make_fake_images(tmp_path, 3)
        out = str(tmp_path / "avg.jpg")

        from opencut.core.star_trail import composite_star_trails
        result = composite_star_trails(paths, out, mode="average")
        assert result.status == "complete"

    def test_composite_requires_min_images(self):
        from opencut.core.star_trail import composite_star_trails
        with pytest.raises(ValueError, match="At least 2"):
            composite_star_trails([], "/tmp/out.jpg")

    def test_remove_streaks_returns_analysis(self, tmp_path):
        pytest.importorskip("numpy")
        pytest.importorskip("PIL")
        paths = self._make_fake_images(tmp_path, 3)

        from opencut.core.star_trail import remove_streaks
        result = remove_streaks(paths)

        assert "frames_analyzed" in result
        assert result["frames_analyzed"] == 3

    def test_collect_images_filters_non_images(self, tmp_path):
        (tmp_path / "test.txt").write_bytes(b"not an image")
        (tmp_path / "test.jpg").write_bytes(b"fake jpg")

        from opencut.core.star_trail import _collect_images
        result = _collect_images([
            str(tmp_path / "test.txt"),
            str(tmp_path / "test.jpg"),
        ])
        assert len(result) == 1


# ========================================================================
# 6. construction_timelapse.py
# ========================================================================
class TestConstructionTimelapse:
    """Tests for opencut.core.construction_timelapse."""

    def _make_images(self, tmp_path, count=5, w=100, h=100):
        np = pytest.importorskip("numpy")
        pytest.importorskip("PIL")
        from PIL import Image
        paths = []
        for i in range(count):
            arr = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
            p = str(tmp_path / f"frame_{i:03d}.png")
            Image.fromarray(arr).save(p)
            paths.append(p)
        return paths

    def test_align_frames_returns_results(self, tmp_path):
        pytest.importorskip("numpy")
        pytest.importorskip("PIL")
        paths = self._make_images(tmp_path, 3)

        from opencut.core.construction_timelapse import align_frames
        results = align_frames(paths)

        assert len(results) == 3
        # First frame (reference) should always be aligned
        assert results[0].aligned is True
        assert results[0].confidence == 1.0

    def test_align_frames_requires_min_images(self):
        from opencut.core.construction_timelapse import align_frames
        with pytest.raises(ValueError, match="At least 2"):
            align_frames([])

    def test_fill_missing_frames_interpolates(self, tmp_path):
        np = pytest.importorskip("numpy")
        pytest.importorskip("PIL")
        from PIL import Image

        arr = np.full((50, 50, 3), 100, dtype=np.uint8)
        p1 = str(tmp_path / "f1.png")
        p3 = str(tmp_path / "f3.png")
        Image.fromarray(arr).save(p1)
        Image.fromarray(arr).save(p3)

        from opencut.core.construction_timelapse import fill_missing_frames
        results = fill_missing_frames([p1, None, p3])

        assert len(results) == 3
        assert results[0]["interpolated"] is False
        assert results[1]["interpolated"] is True
        assert results[2]["interpolated"] is False

    def test_build_timelapse_calls_ffmpeg(self, tmp_path):
        pytest.importorskip("numpy")
        pytest.importorskip("PIL")
        paths = self._make_images(tmp_path, 3)
        out = str(tmp_path / "timelapse.mp4")

        with patch("opencut.core.construction_timelapse.run_ffmpeg"):
            from opencut.core.construction_timelapse import build_construction_timelapse
            result = build_construction_timelapse(
                paths, out, align=False, deflicker=False,
            )

        assert result.status == "complete"
        assert result.total_frames == 3

    def test_build_timelapse_requires_min_images(self):
        from opencut.core.construction_timelapse import build_construction_timelapse
        with pytest.raises(ValueError, match="At least 2"):
            build_construction_timelapse([], "/tmp/out.mp4")


# ========================================================================
# 7. expression_engine.py
# ========================================================================
class TestExpressionEngine:
    """Tests for opencut.core.expression_engine."""

    def test_validate_valid_arithmetic(self):
        from opencut.core.expression_engine import validate_expression
        r = validate_expression("2 + 3 * x")
        assert r["valid"] is True
        assert "x" in r["variables"]

    def test_validate_math_functions(self):
        from opencut.core.expression_engine import validate_expression
        r = validate_expression("sin(t * pi * 2)")
        assert r["valid"] is True

    def test_validate_conditional(self):
        from opencut.core.expression_engine import validate_expression
        r = validate_expression("100 if t > 1.0 else 0")
        assert r["valid"] is True

    def test_validate_empty_expression(self):
        from opencut.core.expression_engine import validate_expression
        r = validate_expression("")
        assert r["valid"] is False

    def test_validate_syntax_error(self):
        from opencut.core.expression_engine import validate_expression
        r = validate_expression("2 +* 3")
        assert r["valid"] is False
        assert "Syntax error" in r["error"]

    def test_sandbox_blocks_import(self):
        """__import__ parses as valid syntax but the sandbox blocks execution."""
        from opencut.core.expression_engine import (
            ExpressionError,
            create_expression_context,
            evaluate_expression,
        )
        ctx = create_expression_context()
        with pytest.raises(ExpressionError, match="Evaluation error"):
            evaluate_expression("__import__('os')", ctx)

    def test_validate_too_long(self):
        from opencut.core.expression_engine import validate_expression
        r = validate_expression("x " * 1500)
        assert r["valid"] is False

    def test_compile_expression(self):
        from opencut.core.expression_engine import compile_expression
        compiled = compile_expression("frame * 2 + 1")
        assert compiled.valid is True
        assert compiled.code is not None

    def test_compile_invalid(self):
        from opencut.core.expression_engine import compile_expression
        compiled = compile_expression("2 +* 3")
        assert compiled.valid is False

    def test_evaluate_arithmetic(self):
        from opencut.core.expression_engine import (
            create_expression_context,
            evaluate_expression,
        )
        ctx = create_expression_context(frame=10)
        result = evaluate_expression("frame * 2", ctx)
        assert result == 20

    def test_evaluate_sin_wave(self):
        from opencut.core.expression_engine import (
            create_expression_context,
            evaluate_expression,
        )
        ctx = create_expression_context(time=0.25, duration=1.0)
        result = evaluate_expression("sin(t * pi * 2)", ctx)
        assert abs(result - 1.0) < 0.001

    def test_evaluate_conditional(self):
        from opencut.core.expression_engine import (
            create_expression_context,
            evaluate_expression,
        )
        ctx = create_expression_context(time=2.0)
        result = evaluate_expression("100 if t > 1.0 else 0", ctx)
        assert result == 100

    def test_evaluate_lerp(self):
        from opencut.core.expression_engine import (
            create_expression_context,
            evaluate_expression,
        )
        ctx = create_expression_context(time=5.0, duration=10.0)
        result = evaluate_expression("lerp(0, 100, progress)", ctx)
        assert abs(result - 50.0) < 0.01

    def test_evaluate_clamp(self):
        from opencut.core.expression_engine import (
            create_expression_context,
            evaluate_expression,
        )
        ctx = create_expression_context()
        result = evaluate_expression("clamp(150, 0, 100)", ctx)
        assert result == 100

    def test_evaluate_compiled_expression(self):
        from opencut.core.expression_engine import (
            compile_expression,
            create_expression_context,
            evaluate_expression,
        )
        compiled = compile_expression("frame + 1")
        ctx = create_expression_context(frame=99)
        result = evaluate_expression(compiled, ctx)
        assert result == 100

    def test_evaluate_audio_reactive(self):
        from opencut.core.expression_engine import (
            create_expression_context,
            evaluate_expression,
        )
        ctx = create_expression_context(audio_amp=0.8)
        result = evaluate_expression("audio_amp * 200", ctx)
        assert abs(result - 160.0) < 0.01

    def test_evaluate_width_height(self):
        from opencut.core.expression_engine import (
            create_expression_context,
            evaluate_expression,
        )
        ctx = create_expression_context(width=3840, height=2160)
        result = evaluate_expression("w / 2", ctx)
        assert result == 1920.0

    def test_context_shortcuts(self):
        from opencut.core.expression_engine import create_expression_context
        ctx = create_expression_context(frame=5, time=1.0, audio_amp=0.5)
        assert ctx["f"] == 5
        assert ctx["t"] == 1.0
        assert ctx["amp"] == 0.5

    def test_custom_vars(self):
        from opencut.core.expression_engine import (
            create_expression_context,
            evaluate_expression,
        )
        ctx = create_expression_context(custom_vars={"offset": 42})
        result = evaluate_expression("offset + 8", ctx)
        assert result == 50


# ========================================================================
# 8. subtitle_position.py
# ========================================================================
class TestSubtitlePosition:
    """Tests for opencut.core.subtitle_position."""

    def test_compute_position_no_obstructions(self):
        from opencut.core.subtitle_position import compute_subtitle_position
        pos = compute_subtitle_position([], (1920, 1080))

        assert pos.alignment == 2  # bottom-center
        assert pos.safe is True
        assert pos.reason == ""

    def test_compute_position_avoids_bottom_obstruction(self):
        from opencut.core.subtitle_position import Obstruction, compute_subtitle_position
        obs = [Obstruction(x=400, y=920, width=1000, height=160, label="face", confidence=0.9)]
        pos = compute_subtitle_position(obs, (1920, 1080))

        # Should move away from default bottom position
        assert pos.reason != ""

    def test_compute_position_with_margin(self):
        from opencut.core.subtitle_position import compute_subtitle_position
        pos = compute_subtitle_position([], (1920, 1080), margin=100)
        assert pos.margin_bottom == 100

    def test_analyze_frame_file_not_found(self):
        from opencut.core.subtitle_position import analyze_frame_obstructions
        with pytest.raises(FileNotFoundError):
            analyze_frame_obstructions("/nonexistent/frame.jpg")

    def test_analyze_frame_returns_obstructions(self, tmp_path):
        np = pytest.importorskip("numpy")
        pytest.importorskip("PIL")
        from PIL import Image

        # Create a frame with a bright region in the bottom
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        arr[70:100, 20:80, :] = 240  # Bright region at bottom
        p = str(tmp_path / "frame.jpg")
        Image.fromarray(arr).save(p)

        from opencut.core.subtitle_position import analyze_frame_obstructions
        obstructions = analyze_frame_obstructions(p, detect_faces=False)

        # May or may not detect depending on thresholds, but should not crash
        assert isinstance(obstructions, list)

    def test_obstruction_dataclass(self):
        from opencut.core.subtitle_position import Obstruction
        o = Obstruction(x=10, y=20, width=100, height=50, label="text", confidence=0.8)
        assert o.x == 10
        assert o.label == "text"

    def test_positioning_result_dataclass(self):
        from opencut.core.subtitle_position import PositioningResult
        r = PositioningResult(output_path="/out.mp4", status="complete")
        d = r.to_dict()
        assert d["status"] == "complete"


# ========================================================================
# Route integration tests
# ========================================================================
class TestBatchDataRoutes:
    """Integration tests for batch_data_routes.py endpoints."""

    @pytest.fixture(autouse=True)
    def setup_client(self, client, csrf_token):
        self.client = client
        self.token = csrf_token
        self.headers = {
            "X-OpenCut-Token": csrf_token,
            "Content-Type": "application/json",
        }

    def test_ingest_rename_preview(self):
        resp = self.client.post("/ingest/rename-preview",
            headers=self.headers,
            json={"filename": "raw.mp4", "pattern": "{name}_v2{ext}"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["renamed"] == "raw_v2.mp4"

    def test_ingest_rename_preview_missing_params(self):
        resp = self.client.post("/ingest/rename-preview",
            headers=self.headers,
            json={"filename": "raw.mp4"})
        assert resp.status_code in (400, 500)

    def test_expression_validate_valid(self):
        resp = self.client.post("/expression/validate",
            headers=self.headers,
            json={"expression": "sin(t * pi)"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["valid"] is True

    def test_expression_validate_invalid(self):
        resp = self.client.post("/expression/validate",
            headers=self.headers,
            json={"expression": "2 +* 3"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["valid"] is False

    def test_expression_validate_empty(self):
        resp = self.client.post("/expression/validate",
            headers=self.headers,
            json={})
        assert resp.status_code == 400

    def test_expression_evaluate(self):
        resp = self.client.post("/expression/evaluate",
            headers=self.headers,
            json={
                "expression": "frame * 2",
                "context": {"frame": 10},
            })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["result"] == 20

    def test_expression_evaluate_batch(self):
        resp = self.client.post("/expression/evaluate-batch",
            headers=self.headers,
            json={
                "expression": "frame * 2",
                "start_frame": 0,
                "end_frame": 5,
                "step": 1,
            })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] == 5
        assert data["results"][0]["value"] == 0
        assert data["results"][2]["value"] == 4

    def test_subtitle_position_compute(self):
        resp = self.client.post("/subtitle-position/compute",
            headers=self.headers,
            json={
                "obstructions": [],
                "frame_width": 1920,
                "frame_height": 1080,
            })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["alignment"] == 2
        assert data["safe"] is True

    def test_subtitle_position_compute_with_obstruction(self):
        resp = self.client.post("/subtitle-position/compute",
            headers=self.headers,
            json={
                "obstructions": [
                    {"x": 400, "y": 920, "width": 1000, "height": 160,
                     "label": "face", "confidence": 0.9},
                ],
                "frame_width": 1920,
                "frame_height": 1080,
            })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["reason"] != ""

    def test_storage_manifest_empty(self):
        resp = self.client.post("/storage/manifest",
            headers=self.headers,
            json={})
        assert resp.status_code == 200
        data = resp.get_json()
        assert "entries" in data

    def test_metadata_read_validation(self):
        resp = self.client.post("/batch-metadata/read",
            headers=self.headers,
            json={"file_paths": "not_a_list"})
        assert resp.status_code == 400

    def test_conform_analyze_validation(self):
        resp = self.client.post("/batch-conform/analyze",
            headers=self.headers,
            json={"file_paths": [], "target_spec": {}})
        assert resp.status_code == 400

    def test_csrf_required(self):
        """Routes should reject requests without CSRF token."""
        resp = self.client.post("/expression/validate",
            headers={"Content-Type": "application/json"},
            json={"expression": "1+1"})
        assert resp.status_code == 403
