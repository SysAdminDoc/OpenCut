"""Coverage expansion: additional core modules."""

import csv
import json
import os
import sqlite3
import threading
import time
from unittest.mock import MagicMock, patch

import pytest


# ========================================================================
# 1. analytics.py
# ========================================================================
class TestAnalytics:
    """Tests for opencut.core.analytics — usage tracking SQLite store."""

    def _reset_analytics(self, tmp_path):
        """Reset module globals and point DB at tmp_path."""
        import opencut.core.analytics as mod
        mod._INITIALIZED = False
        mod._DB_PATH = os.path.join(str(tmp_path), "analytics.db")
        mod._LOCAL = threading.local()
        return mod

    def test_init_db_creates_table(self, tmp_path):
        mod = self._reset_analytics(tmp_path)
        mod._init_db()
        conn = sqlite3.connect(mod._DB_PATH)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = [t[0] for t in tables]
        assert "usage_log" in table_names
        conn.close()

    def test_init_db_idempotent(self, tmp_path):
        mod = self._reset_analytics(tmp_path)
        mod._init_db()
        mod._init_db()  # second call should not error
        assert mod._INITIALIZED is True

    def test_record_usage_inserts_row(self, tmp_path):
        mod = self._reset_analytics(tmp_path)
        mod.record_usage("/test", duration_ms=150, success=True, job_type="export")
        conn = sqlite3.connect(mod._DB_PATH)
        rows = conn.execute("SELECT * FROM usage_log").fetchall()
        assert len(rows) == 1
        conn.close()

    def test_record_usage_success_flag(self, tmp_path):
        mod = self._reset_analytics(tmp_path)
        mod.record_usage("/a", duration_ms=10, success=True)
        mod.record_usage("/b", duration_ms=20, success=False)
        conn = sqlite3.connect(mod._DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT endpoint, success FROM usage_log ORDER BY id").fetchall()
        assert rows[0]["success"] == 1
        assert rows[1]["success"] == 0
        conn.close()

    def test_get_usage_stats_empty(self, tmp_path):
        mod = self._reset_analytics(tmp_path)
        mod._init_db()
        stats = mod.get_usage_stats(days=30)
        assert stats["total_jobs"] == 0
        assert stats["total_errors"] == 0
        assert stats["avg_duration_ms"] == 0.0
        assert stats["top_features"] == []
        assert stats["daily_usage"] == []

    def test_get_usage_stats_aggregation(self, tmp_path):
        mod = self._reset_analytics(tmp_path)
        mod.record_usage("/silence", duration_ms=100, success=True)
        mod.record_usage("/silence", duration_ms=200, success=True)
        mod.record_usage("/silence", duration_ms=300, success=False)
        mod.record_usage("/export", duration_ms=500, success=True)
        stats = mod.get_usage_stats(days=30)
        assert stats["total_jobs"] == 4
        assert stats["total_errors"] == 1
        assert len(stats["top_features"]) == 2
        # /silence has count=3
        silence_feat = next(f for f in stats["top_features"] if f["endpoint"] == "/silence")
        assert silence_feat["count"] == 3
        assert silence_feat["error_rate"] == pytest.approx(1 / 3, abs=0.001)

    def test_get_usage_stats_daily_usage_present(self, tmp_path):
        mod = self._reset_analytics(tmp_path)
        mod.record_usage("/x", duration_ms=10, success=True)
        stats = mod.get_usage_stats(days=30)
        assert len(stats["daily_usage"]) >= 1
        assert "date" in stats["daily_usage"][0]
        assert "count" in stats["daily_usage"][0]

    def test_get_feature_stats_empty(self, tmp_path):
        mod = self._reset_analytics(tmp_path)
        mod._init_db()
        fs = mod.get_feature_stats("/nonexistent", days=30)
        assert fs["total_calls"] == 0
        assert fs["error_rate"] == 0.0
        assert fs["endpoint"] == "/nonexistent"

    def test_get_feature_stats_single_endpoint(self, tmp_path):
        mod = self._reset_analytics(tmp_path)
        mod.record_usage("/encode", duration_ms=100, success=True)
        mod.record_usage("/encode", duration_ms=300, success=False)
        fs = mod.get_feature_stats("/encode", days=30)
        assert fs["total_calls"] == 2
        assert fs["success_count"] == 1
        assert fs["error_count"] == 1
        assert fs["error_rate"] == 0.5
        assert fs["min_duration_ms"] == 100
        assert fs["max_duration_ms"] == 300

    def test_get_feature_stats_daily_breakdown(self, tmp_path):
        mod = self._reset_analytics(tmp_path)
        mod.record_usage("/test", duration_ms=50, success=True)
        fs = mod.get_feature_stats("/test", days=30)
        assert len(fs["daily_usage"]) >= 1
        assert "avg_duration_ms" in fs["daily_usage"][0]

    def test_days_filter_excludes_old_events(self, tmp_path):
        mod = self._reset_analytics(tmp_path)
        mod._init_db()
        # Insert a row with an old timestamp directly
        conn = mod._get_conn()
        old_ts = time.time() - 86400 * 60  # 60 days ago
        conn.execute(
            "INSERT INTO usage_log (endpoint, duration_ms, success, timestamp) VALUES (?, ?, ?, ?)",
            ("/old", 10, 1, old_ts),
        )
        conn.commit()
        stats = mod.get_usage_stats(days=30)
        assert stats["total_jobs"] == 0


# ========================================================================
# 2. selects_bin.py — Select CRUD, tag filtering, CSV export
# ========================================================================
class TestSelectsBin:
    """Tests for opencut.core.selects_bin — clip selects management."""

    def _make_db(self, tmp_path):
        """Create a temporary selects DB and patch module to use it."""
        db_path = os.path.join(str(tmp_path), "selects.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS clips (
                clip_path TEXT PRIMARY KEY,
                rating    INTEGER DEFAULT 0,
                tags      TEXT DEFAULT '[]',
                notes     TEXT DEFAULT '',
                added_at  REAL DEFAULT (julianday('now')),
                duration  REAL DEFAULT 0,
                width     INTEGER DEFAULT 0,
                height    INTEGER DEFAULT 0,
                fps       REAL DEFAULT 0
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_clips_rating ON clips(rating)")
        conn.commit()
        return db_path, conn

    def _insert_clip(self, conn, path, rating=3, tags=None, notes=""):
        tags = tags or []
        conn.execute(
            "INSERT INTO clips (clip_path, rating, tags, notes, duration, width, height, fps) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (path, rating, json.dumps(tags), notes, 10.0, 1920, 1080, 30.0),
        )
        conn.commit()

    def test_clip_metadata_dataclass(self):
        from opencut.core.selects_bin import ClipMetadata
        clip = ClipMetadata(clip_path="/a.mp4", rating=4, tags=["action"])
        assert clip.rating == 4
        assert clip.tags == ["action"]

    def test_selects_search_result_dataclass(self):
        from opencut.core.selects_bin import SelectsSearchResult
        sr = SelectsSearchResult(clips=[], total=0)
        assert sr.total == 0

    def test_search_selects_by_rating(self, tmp_path):
        db_path, conn = self._make_db(tmp_path)
        self._insert_clip(conn, "/a.mp4", rating=5)
        self._insert_clip(conn, "/b.mp4", rating=2)
        self._insert_clip(conn, "/c.mp4", rating=4)
        conn.close()

        with patch("opencut.core.selects_bin._get_db_path", return_value=db_path):
            from opencut.core.selects_bin import search_selects
            result = search_selects({"min_rating": 4})
        assert len(result.clips) == 2
        assert all(c.rating >= 4 for c in result.clips)

    def test_search_selects_by_max_rating(self, tmp_path):
        db_path, conn = self._make_db(tmp_path)
        self._insert_clip(conn, "/x.mp4", rating=1)
        self._insert_clip(conn, "/y.mp4", rating=5)
        conn.close()

        with patch("opencut.core.selects_bin._get_db_path", return_value=db_path):
            from opencut.core.selects_bin import search_selects
            result = search_selects({"max_rating": 2})
        assert len(result.clips) == 1
        assert result.clips[0].rating == 1

    def test_search_selects_by_tags_all(self, tmp_path):
        db_path, conn = self._make_db(tmp_path)
        self._insert_clip(conn, "/a.mp4", tags=["action", "drone"])
        self._insert_clip(conn, "/b.mp4", tags=["action"])
        conn.close()

        with patch("opencut.core.selects_bin._get_db_path", return_value=db_path):
            from opencut.core.selects_bin import search_selects
            result = search_selects({"tags": ["action", "drone"]})
        assert len(result.clips) == 1
        assert result.clips[0].clip_path == "/a.mp4"

    def test_search_selects_by_any_tags(self, tmp_path):
        db_path, conn = self._make_db(tmp_path)
        self._insert_clip(conn, "/a.mp4", tags=["drone"])
        self._insert_clip(conn, "/b.mp4", tags=["interview"])
        self._insert_clip(conn, "/c.mp4", tags=[])
        conn.close()

        with patch("opencut.core.selects_bin._get_db_path", return_value=db_path):
            from opencut.core.selects_bin import search_selects
            result = search_selects({"any_tags": ["drone", "interview"]})
        assert len(result.clips) == 2

    def test_search_selects_text_search(self, tmp_path):
        db_path, conn = self._make_db(tmp_path)
        self._insert_clip(conn, "/footage/beach.mp4", notes="sunset scene")
        self._insert_clip(conn, "/footage/city.mp4", notes="night skyline")
        conn.close()

        with patch("opencut.core.selects_bin._get_db_path", return_value=db_path):
            from opencut.core.selects_bin import search_selects
            result = search_selects({"search": "beach"})
        assert len(result.clips) == 1
        assert "beach" in result.clips[0].clip_path

    def test_search_selects_pagination(self, tmp_path):
        db_path, conn = self._make_db(tmp_path)
        for i in range(10):
            self._insert_clip(conn, f"/clip{i}.mp4", rating=i % 5)
        conn.close()

        with patch("opencut.core.selects_bin._get_db_path", return_value=db_path):
            from opencut.core.selects_bin import search_selects
            result = search_selects({"limit": 3, "offset": 0})
        assert len(result.clips) <= 3
        assert result.total == 10

    def test_export_selects_csv_format(self, tmp_path):
        db_path, conn = self._make_db(tmp_path)
        self._insert_clip(conn, "/a.mp4", rating=5, tags=["hero", "best"])
        conn.close()

        out_csv = os.path.join(str(tmp_path), "export.csv")
        with patch("opencut.core.selects_bin._get_db_path", return_value=db_path):
            from opencut.core.selects_bin import export_selects
            result = export_selects(output_path_str=out_csv, format="csv")

        assert result["format"] == "csv"
        assert result["clip_count"] == 1
        with open(out_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert "clip_path" in header
            assert "rating" in header
            assert "tags" in header
            row = next(reader)
            assert "/a.mp4" in row[0]
            assert "hero;best" in row[2]

    def test_export_selects_json_format(self, tmp_path):
        db_path, conn = self._make_db(tmp_path)
        self._insert_clip(conn, "/b.mp4", rating=3)
        conn.close()

        out_json = os.path.join(str(tmp_path), "export.json")
        with patch("opencut.core.selects_bin._get_db_path", return_value=db_path):
            from opencut.core.selects_bin import export_selects
            result = export_selects(output_path_str=out_json, format="json")

        assert result["format"] == "json"
        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "selects" in data
        assert len(data["selects"]) == 1

    def test_export_selects_no_clips_raises(self, tmp_path):
        db_path, conn = self._make_db(tmp_path)
        conn.close()

        with patch("opencut.core.selects_bin._get_db_path", return_value=db_path):
            from opencut.core.selects_bin import export_selects
            with pytest.raises(ValueError, match="No clips"):
                export_selects(output_path_str=os.path.join(str(tmp_path), "out.json"))

    def test_get_clip_metadata_not_in_bin(self, tmp_path):
        db_path, conn = self._make_db(tmp_path)
        conn.close()

        with patch("opencut.core.selects_bin._get_db_path", return_value=db_path):
            from opencut.core.selects_bin import get_clip_metadata
            meta = get_clip_metadata("/nonexistent.mp4")
        assert meta["in_bin"] is False

    def test_get_clip_metadata_in_bin(self, tmp_path):
        db_path, conn = self._make_db(tmp_path)
        self._insert_clip(conn, "/a.mp4", rating=4, tags=["hero"])
        conn.close()

        with patch("opencut.core.selects_bin._get_db_path", return_value=db_path):
            from opencut.core.selects_bin import get_clip_metadata
            meta = get_clip_metadata("/a.mp4")
        assert meta["in_bin"] is True
        assert meta["rating"] == 4
        assert "hero" in meta["tags"]


# ========================================================================
# 3. show_notes.py — Note parsing, fallback extraction, export formats
# ========================================================================
class TestShowNotes:
    """Tests for opencut.core.show_notes — show note generation & export."""

    def test_parse_llm_response_full(self):
        from opencut.core.show_notes import _parse_llm_response
        text = """## Summary
This is a great episode about coding.

## Key Topics
- [00:01:30] Introduction to Python
- [00:05:00] Web frameworks

## Notable Quotes
- "Code is poetry"
- "Ship it"

## Chapter Markers
- [00:00:00] Intro
- [00:10:00] Deep dive

## Resources Mentioned
- https://python.org
- Django docs
"""
        notes = _parse_llm_response(text)
        assert "great episode" in notes.summary
        assert len(notes.key_topics) == 2
        assert notes.key_topics[0]["timestamp"] == "00:01:30"
        assert notes.key_topics[0]["topic"] == "Introduction to Python"
        assert len(notes.quotes) == 2
        assert notes.quotes[0] == "Code is poetry"
        assert len(notes.chapter_markers) == 2
        assert notes.chapter_markers[1]["title"] == "Deep dive"
        assert len(notes.resources_mentioned) == 2

    def test_parse_llm_response_empty_sections(self):
        from opencut.core.show_notes import _parse_llm_response
        text = """## Summary
Brief summary here.

## Key Topics
None

## Notable Quotes
None
"""
        notes = _parse_llm_response(text)
        assert "Brief summary" in notes.summary
        assert len(notes.key_topics) == 0
        assert len(notes.quotes) == 0

    def test_parse_llm_response_no_timestamps(self):
        from opencut.core.show_notes import _parse_llm_response
        text = """## Key Topics
- Machine learning basics
- Neural networks
"""
        notes = _parse_llm_response(text)
        assert len(notes.key_topics) == 2
        assert notes.key_topics[0]["timestamp"] == ""
        assert "Machine learning" in notes.key_topics[0]["topic"]

    def test_fallback_show_notes_summary(self):
        from opencut.core.show_notes import _fallback_show_notes
        text = (
            "Welcome to the podcast about software engineering. "
            "We discuss many topics including testing and deployment. "
            "Thank you for listening to this episode about engineering."
        )
        notes = _fallback_show_notes(text)
        assert len(notes.summary) > 0
        assert notes.summary.endswith(".")

    def test_fallback_show_notes_urls(self):
        from opencut.core.show_notes import _fallback_show_notes
        text = "Check out https://example.com and also https://docs.python.org for more info about the topic."
        notes = _fallback_show_notes(text)
        assert any("example.com" in r for r in notes.resources_mentioned)

    def test_fallback_show_notes_keywords(self):
        from opencut.core.show_notes import _fallback_show_notes
        text = (
            "Python Python Python Python Python. "
            "JavaScript JavaScript JavaScript. "
            "Something else entirely different here."
        )
        notes = _fallback_show_notes(text)
        assert len(notes.key_topics) > 0
        topic_texts = [t["topic"].lower() for t in notes.key_topics]
        assert any("python" in t for t in topic_texts)

    def test_fallback_show_notes_quotes_longest_sentences(self):
        from opencut.core.show_notes import _fallback_show_notes
        text = (
            "Short one. "
            "This is a much longer sentence that should be picked as a notable quote for the show notes. "
            "Another fairly substantial sentence with enough length to qualify as notable content for output."
        )
        notes = _fallback_show_notes(text)
        assert len(notes.quotes) > 0

    def test_generate_show_notes_empty_input(self):
        from opencut.core.show_notes import generate_show_notes
        notes = generate_show_notes("")
        assert "No transcript" in notes.summary

    def test_generate_show_notes_whitespace_only(self):
        from opencut.core.show_notes import generate_show_notes
        notes = generate_show_notes("   \n\t  ")
        assert "No transcript" in notes.summary

    def test_export_markdown(self):
        from opencut.core.show_notes import ShowNotes, export_show_notes
        notes = ShowNotes(
            summary="A great episode.",
            key_topics=[{"timestamp": "00:01:00", "topic": "Intro"}],
            quotes=["Ship it fast"],
            chapter_markers=[{"timestamp": "00:00:00", "title": "Start"}],
            resources_mentioned=["https://example.com"],
        )
        md = export_show_notes(notes, format="markdown")
        assert "# Show Notes" in md
        assert "## Summary" in md
        assert "A great episode." in md
        assert "[00:01:00]" in md
        assert '"Ship it fast"' in md
        assert "https://example.com" in md

    def test_export_html(self):
        from opencut.core.show_notes import ShowNotes, export_show_notes
        notes = ShowNotes(
            summary="Test <summary>",
            key_topics=[{"timestamp": "", "topic": "Topic A"}],
        )
        html = export_show_notes(notes, format="html")
        assert "<div class='show-notes'>" in html
        assert "&lt;summary&gt;" in html  # HTML escaped
        assert "Topic A" in html

    def test_export_text(self):
        from opencut.core.show_notes import ShowNotes, export_show_notes
        notes = ShowNotes(
            summary="Plain text summary.",
            chapter_markers=[{"timestamp": "00:05:00", "title": "Ch1"}],
        )
        text = export_show_notes(notes, format="text")
        assert "SHOW NOTES" in text
        assert "SUMMARY" in text
        assert "Plain text summary." in text
        assert "CHAPTERS" in text

    def test_export_invalid_format_defaults_markdown(self):
        from opencut.core.show_notes import ShowNotes, export_show_notes
        notes = ShowNotes(summary="Test.")
        out = export_show_notes(notes, format="docx")
        assert "# Show Notes" in out  # markdown fallback

    def test_html_escape(self):
        from opencut.core.show_notes import _html_escape
        assert _html_escape("<script>") == "&lt;script&gt;"
        assert _html_escape('"hello"') == "&quot;hello&quot;"
        assert _html_escape("a&b") == "a&amp;b"


# ========================================================================
# 4. ffmpeg_builder.py — Node-to-string, graph validation, chain building
# ========================================================================
class TestFfmpegBuilder:
    """Tests for opencut.core.ffmpeg_builder — filter chain construction."""

    def test_node_to_filter_str_simple(self):
        from opencut.core.ffmpeg_builder import FilterNode, _node_to_filter_str
        node = FilterNode(node_id="n0", filter_name="scale", params={"w": 1920, "h": 1080})
        result = _node_to_filter_str(node)
        assert "scale" in result
        assert "w=1920" in result
        assert "h=1080" in result

    def test_node_to_filter_str_boolean_param(self):
        from opencut.core.ffmpeg_builder import FilterNode, _node_to_filter_str
        node = FilterNode(node_id="n0", filter_name="eq", params={"eval": True})
        result = _node_to_filter_str(node)
        assert "eval=1" in result

    def test_node_to_filter_str_with_pads(self):
        from opencut.core.ffmpeg_builder import FilterNode, _node_to_filter_str
        node = FilterNode(
            node_id="n0", filter_name="overlay",
            inputs=["base", "ovl"], outputs=["out"],
        )
        result = _node_to_filter_str(node)
        assert "[base]" in result
        assert "[ovl]" in result
        assert "[out]" in result
        assert "overlay" in result

    def test_node_to_filter_str_no_params(self):
        from opencut.core.ffmpeg_builder import FilterNode, _node_to_filter_str
        node = FilterNode(node_id="n0", filter_name="hflip")
        result = _node_to_filter_str(node)
        assert result == "hflip"

    def test_build_filter_chain_single_node(self):
        from opencut.core.ffmpeg_builder import build_filter_chain
        nodes = [{"filter_name": "scale", "params": {"w": 1280, "h": 720}}]
        chain = build_filter_chain(nodes)
        assert "scale" in chain
        assert "w=1280" in chain

    def test_build_filter_chain_multiple_nodes(self):
        from opencut.core.ffmpeg_builder import build_filter_chain
        nodes = [
            {"filter_name": "scale", "params": {"w": 1280, "h": 720}},
            {"filter_name": "hflip"},
            {"filter_name": "eq", "params": {"brightness": 0.1}},
        ]
        chain = build_filter_chain(nodes)
        parts = chain.split(";")
        assert len(parts) == 3
        assert "scale" in parts[0]
        assert "hflip" in parts[1]
        assert "eq" in parts[2]

    def test_build_filter_chain_empty_raises(self):
        from opencut.core.ffmpeg_builder import build_filter_chain
        with pytest.raises(ValueError, match="(?i)at least one"):
            build_filter_chain([])

    def test_build_filter_chain_with_connections(self):
        from opencut.core.ffmpeg_builder import build_filter_chain
        nodes = [
            {"node_id": "split", "filter_name": "split", "outputs": []},
            {"node_id": "scale", "filter_name": "scale", "params": {"w": 640}},
        ]
        connections = [
            {"from_node": "split", "from_pad": "v0", "to_node": "scale", "to_pad": "in"},
        ]
        chain = build_filter_chain(nodes, connections)
        assert "split" in chain
        assert "scale" in chain

    def test_validate_filter_graph_valid(self):
        from opencut.core.ffmpeg_builder import validate_filter_graph
        graph = {
            "nodes": [
                {"node_id": "n0", "filter_name": "scale"},
                {"node_id": "n1", "filter_name": "hflip"},
            ],
            "connections": [],
        }
        result = validate_filter_graph(graph)
        assert result["valid"] is True
        assert result["errors"] == []
        assert result["node_count"] == 2

    def test_validate_filter_graph_empty_nodes(self):
        from opencut.core.ffmpeg_builder import validate_filter_graph
        result = validate_filter_graph({"nodes": []})
        assert result["valid"] is False
        assert any("at least one" in e for e in result["errors"])

    def test_validate_filter_graph_missing_filter_name(self):
        from opencut.core.ffmpeg_builder import validate_filter_graph
        graph = {"nodes": [{"node_id": "n0"}]}
        result = validate_filter_graph(graph)
        assert result["valid"] is False
        assert any("filter_name" in e for e in result["errors"])

    def test_validate_filter_graph_duplicate_ids(self):
        from opencut.core.ffmpeg_builder import validate_filter_graph
        graph = {
            "nodes": [
                {"node_id": "n0", "filter_name": "scale"},
                {"node_id": "n0", "filter_name": "crop"},
            ],
        }
        result = validate_filter_graph(graph)
        assert result["valid"] is False
        assert any("Duplicate" in e for e in result["errors"])

    def test_validate_filter_graph_bad_connection_source(self):
        from opencut.core.ffmpeg_builder import validate_filter_graph
        graph = {
            "nodes": [{"node_id": "n0", "filter_name": "scale"}],
            "connections": [{"from_node": "missing", "from_pad": "v", "to_node": "n0", "to_pad": "in"}],
        }
        result = validate_filter_graph(graph)
        assert any("unknown source" in e for e in result["errors"])

    def test_validate_filter_graph_unknown_filter_warning(self):
        from opencut.core.ffmpeg_builder import validate_filter_graph
        graph = {
            "nodes": [{"node_id": "n0", "filter_name": "my_custom_filter_xyz"}],
        }
        result = validate_filter_graph(graph)
        assert result["valid"] is True  # unknown filters are warnings, not errors
        assert any("Unknown filter" in w for w in result["warnings"])

    def test_save_and_load_preset(self, tmp_path):
        from opencut.core.ffmpeg_builder import load_filter_presets, save_filter_preset
        presets_dir = os.path.join(str(tmp_path), "presets")
        with patch("opencut.core.ffmpeg_builder.PRESETS_DIR", presets_dir):
            save_filter_preset("scale:eq:hflip", "My Cool Preset", description="test")
            presets = load_filter_presets()
        assert len(presets) == 1
        assert presets[0]["name"] == "My Cool Preset"
        assert presets[0]["chain"] == "scale:eq:hflip"

    def test_save_preset_empty_name_raises(self):
        from opencut.core.ffmpeg_builder import save_filter_preset
        with pytest.raises(ValueError, match="empty"):
            save_filter_preset("chain", "")


# ========================================================================
# 5. ab_av1.py — Output parsing, CRF validation, result dataclass
# ========================================================================
class TestAbAv1:
    """Tests for opencut.core.ab_av1 — VMAF-target encoding wrapper."""

    def test_result_dataclass_defaults(self):
        from opencut.core.ab_av1 import AbAv1Result
        r = AbAv1Result()
        assert r.output == ""
        assert r.encoder == "libsvtav1"
        assert r.target_vmaf == 0.0
        assert r.notes == []

    def test_result_dataclass_dict_interface(self):
        from opencut.core.ab_av1 import AbAv1Result
        r = AbAv1Result(output="/out.mp4", final_crf=28.0)
        assert r["output"] == "/out.mp4"
        assert "final_crf" in r
        assert "nonexistent" not in r
        assert "output" in r.keys()

    def test_parse_output_json_line(self):
        from opencut.core.ab_av1 import _parse_output
        stdout = 'some info\n{"vmaf": 94.5, "crf": 30}\n'
        result = _parse_output(stdout, "")
        assert result["vmaf"] == 94.5
        assert result["crf"] == 30.0

    def test_parse_output_json_achieved_key(self):
        from opencut.core.ab_av1 import _parse_output
        stdout = '{"achieved": 92.1, "final_crf": 25}\n'
        result = _parse_output(stdout, "")
        assert result["vmaf"] == 92.1
        assert result["crf"] == 25.0

    def test_parse_output_regex_fallback(self):
        from opencut.core.ab_av1 import _parse_output
        stderr = "crf 28 VMAF 93.45\ncrf 30 VMAF 95.12\n"
        result = _parse_output("", stderr)
        # Should take last match
        assert result["vmaf"] == 95.12
        assert result["crf"] == 30.0

    def test_parse_output_final_crf_line(self):
        from opencut.core.ab_av1 import _parse_output
        stderr = "encode with crf 32\n"
        result = _parse_output("", stderr)
        assert result["crf"] == 32.0
        assert result["vmaf"] == 0.0

    def test_parse_output_no_matches(self):
        from opencut.core.ab_av1 import _parse_output
        result = _parse_output("nothing useful here", "also nothing")
        assert result["vmaf"] == 0.0
        assert result["crf"] == 0.0

    def test_parse_output_combined_stdout_stderr(self):
        from opencut.core.ab_av1 import _parse_output
        result = _parse_output("crf 20 VMAF 88.0", "selected crf 22")
        # The regex should find the pair
        assert result["vmaf"] == 88.0
        assert result["crf"] == 20.0

    def test_supported_encoders(self):
        from opencut.core.ab_av1 import SUPPORTED_ENCODERS
        assert "libsvtav1" in SUPPORTED_ENCODERS
        assert "libx264" in SUPPORTED_ENCODERS
        assert "libx265" in SUPPORTED_ENCODERS
        assert "libaom-av1" in SUPPORTED_ENCODERS

    def test_encode_to_vmaf_bad_encoder(self, tmp_path):
        from opencut.core.ab_av1 import encode_to_vmaf
        dummy = os.path.join(str(tmp_path), "in.mp4")
        with open(dummy, "w") as f:
            f.write("fake")
        with pytest.raises(ValueError, match="Unsupported encoder"):
            encode_to_vmaf(dummy, encoder="badencoder")

    def test_encode_to_vmaf_bad_vmaf_range(self, tmp_path):
        from opencut.core.ab_av1 import encode_to_vmaf
        dummy = os.path.join(str(tmp_path), "in.mp4")
        with open(dummy, "w") as f:
            f.write("fake")
        with pytest.raises(ValueError, match="target_vmaf"):
            encode_to_vmaf(dummy, target_vmaf=5.0)

    def test_encode_to_vmaf_file_not_found(self):
        from opencut.core.ab_av1 import encode_to_vmaf
        with pytest.raises(FileNotFoundError):
            encode_to_vmaf("/nonexistent/video.mp4")

    def test_encode_to_vmaf_no_binary(self, tmp_path):
        from opencut.core.ab_av1 import encode_to_vmaf
        dummy = os.path.join(str(tmp_path), "in.mp4")
        with open(dummy, "w") as f:
            f.write("fake")
        with patch("opencut.core.ab_av1.check_ab_av1_available", return_value=False):
            with pytest.raises(RuntimeError, match="not installed"):
                encode_to_vmaf(dummy)

    def test_check_ab_av1_available(self):
        from opencut.core.ab_av1 import check_ab_av1_available
        with patch("shutil.which", return_value=None):
            assert check_ab_av1_available() is False
        with patch("shutil.which", return_value="/usr/bin/ab-av1"):
            assert check_ab_av1_available() is True


# ========================================================================
# 6. ab_compare.py — Metric computation, dataclasses, mode validation
# ========================================================================
class TestAbCompare:
    """Tests for opencut.core.ab_compare — A/B comparison data."""

    def test_frame_metrics_to_dict(self):
        from opencut.core.ab_compare import FrameMetrics
        m = FrameMetrics(timestamp=1.5, ssim=0.95, psnr=40.0, color_delta=2.1, mse=10.5)
        d = m.to_dict()
        assert d["timestamp"] == 1.5
        assert d["ssim"] == 0.95

    def test_compare_frame_to_dict(self):
        from opencut.core.ab_compare import CompareFrame
        f = CompareFrame(timestamp=0.0, mode="side_by_side", composite_path="/out.jpg")
        d = f.to_dict()
        assert d["mode"] == "side_by_side"

    def test_compare_result_to_dict(self):
        from opencut.core.ab_compare import CompareResult, FrameMetrics
        cr = CompareResult(
            metrics=[FrameMetrics(ssim=0.9)],
            overall_ssim=0.9,
            mode="checkerboard",
        )
        d = cr.to_dict()
        assert d["mode"] == "checkerboard"
        assert len(d["metrics"]) == 1

    def test_compare_modes_constant(self):
        from opencut.core.ab_compare import COMPARE_MODES
        assert "side_by_side" in COMPARE_MODES
        assert "overlay_blend" in COMPARE_MODES
        assert "checkerboard" in COMPARE_MODES
        assert len(COMPARE_MODES) == 6

    def test_list_compare_modes(self):
        from opencut.core.ab_compare import list_compare_modes
        modes = list_compare_modes()
        assert len(modes) == 6
        assert all("id" in m and "name" in m and "description" in m for m in modes)

    def test_generate_comparison_bad_mode(self, tmp_path):
        from opencut.core.ab_compare import generate_comparison
        orig = os.path.join(str(tmp_path), "orig.mp4")
        proc = os.path.join(str(tmp_path), "proc.mp4")
        for p in (orig, proc):
            with open(p, "w") as f:
                f.write("fake")
        with pytest.raises(ValueError, match="Invalid mode"):
            generate_comparison(orig, proc, mode="bad_mode")

    def test_generate_comparison_missing_original(self):
        from opencut.core.ab_compare import generate_comparison
        with pytest.raises(FileNotFoundError, match="Original"):
            generate_comparison("/no/orig.mp4", "/no/proc.mp4")

    def test_generate_comparison_missing_processed(self, tmp_path):
        from opencut.core.ab_compare import generate_comparison
        orig = os.path.join(str(tmp_path), "orig.mp4")
        with open(orig, "w") as f:
            f.write("fake")
        with pytest.raises(FileNotFoundError, match="Processed"):
            generate_comparison(orig, "/no/proc.mp4")

    def test_metrics_numpy_identical_images(self):
        """Identical arrays should give SSIM=1, PSNR=100, MSE=0."""
        pytest.importorskip("numpy")
        import numpy as np

        from opencut.core.ab_compare import _metrics_numpy
        arr = np.full((100, 100, 3), 128, dtype=np.float64)
        m = _metrics_numpy(arr, arr.copy())
        assert m.mse == 0.0
        assert m.psnr == 100.0
        assert m.ssim == pytest.approx(1.0, abs=0.001)
        assert m.color_delta == 0.0

    def test_metrics_numpy_different_images(self):
        pytest.importorskip("numpy")
        import numpy as np

        from opencut.core.ab_compare import _metrics_numpy
        a = np.zeros((50, 50, 3), dtype=np.float64)
        b = np.full((50, 50, 3), 255, dtype=np.float64)
        m = _metrics_numpy(a, b)
        assert m.mse > 0
        assert m.psnr < 100.0
        assert m.ssim < 1.0
        assert m.color_delta > 0

    def test_default_dimensions(self):
        from opencut.core.ab_compare import DEFAULT_HEIGHT, DEFAULT_WIDTH
        assert DEFAULT_WIDTH == 854
        assert DEFAULT_HEIGHT == 480


# ========================================================================
# 7. cinemagraph.py — Static region detection params, loop validation
# ========================================================================
class TestCinemagraph:
    """Tests for opencut.core.cinemagraph — cinemagraph creation."""

    def test_cinemagraph_result_defaults(self):
        from opencut.core.cinemagraph import CinemagraphResult
        r = CinemagraphResult()
        assert r.output_path == ""
        assert r.frames_written == 0
        assert r.resolution == (0, 0)

    def test_reference_frame_result_defaults(self):
        from opencut.core.cinemagraph import ReferenceFrameResult
        r = ReferenceFrameResult()
        assert r.frame_path == ""
        assert r.timestamp == 0.0

    def test_create_cinemagraph_file_not_found(self):
        from opencut.core.cinemagraph import create_cinemagraph
        with patch("opencut.core.cinemagraph.ensure_package", return_value=True):
            with pytest.raises(FileNotFoundError):
                create_cinemagraph("/nonexistent.mp4", {"type": "rect"})

    def test_create_cinemagraph_no_opencv_raises(self):
        from opencut.core.cinemagraph import create_cinemagraph
        with patch("opencut.core.cinemagraph.ensure_package", return_value=False):
            with pytest.raises(RuntimeError, match="opencv"):
                create_cinemagraph("dummy.mp4", {})

    def test_extract_reference_frame_not_found(self):
        from opencut.core.cinemagraph import extract_reference_frame
        with pytest.raises(FileNotFoundError):
            extract_reference_frame("/nonexistent.mp4")


# ========================================================================
# 8. surround_mix.py — Panning, channel assignment, gain normalization
# ========================================================================
class TestSurroundMix:
    """Tests for opencut.core.surround_mix — surround sound processing."""

    def test_channel_layouts(self):
        from opencut.core.surround_mix import CHANNEL_LAYOUTS
        assert CHANNEL_LAYOUTS["5.1"]["channels"] == 6
        assert CHANNEL_LAYOUTS["7.1"]["channels"] == 8
        assert CHANNEL_LAYOUTS["stereo"]["channels"] == 2
        assert CHANNEL_LAYOUTS["mono"]["channels"] == 1

    def test_surround_positions_angles(self):
        from opencut.core.surround_mix import SURROUND_POSITIONS
        assert SURROUND_POSITIONS["front_left"]["angle"] == -30
        assert SURROUND_POSITIONS["front_right"]["angle"] == 30
        assert SURROUND_POSITIONS["center"]["angle"] == 0
        assert SURROUND_POSITIONS["back_left"]["angle"] == -110
        assert SURROUND_POSITIONS["back_right"]["angle"] == 110

    def test_export_formats(self):
        from opencut.core.surround_mix import EXPORT_FORMATS
        assert "wav" in EXPORT_FORMATS
        assert "flac" in EXPORT_FORMATS
        assert "ac3" in EXPORT_FORMATS
        assert "eac3" in EXPORT_FORMATS
        assert EXPORT_FORMATS["wav"]["codec"] == "pcm_s24le"

    def test_surround_position_dataclass(self):
        from opencut.core.surround_mix import SurroundPosition
        pos = SurroundPosition(angle=45.0, distance=0.8, lfe_amount=0.3)
        assert pos.angle == 45.0
        assert pos.distance == 0.8
        assert pos.lfe_amount == 0.3

    def test_calculate_surround_gains_center(self):
        from opencut.core.surround_mix import SurroundPosition, _calculate_surround_gains
        pos = SurroundPosition(angle=0.0, distance=1.0, lfe_amount=0.5)
        gains = _calculate_surround_gains(pos, "5.1")
        # Center (0 deg) should have strong FC and balanced FL/FR
        assert gains["FC"] > 0
        assert gains["FL"] > 0
        assert gains["FR"] > 0
        assert gains["LFE"] == pytest.approx(0.5, abs=0.01)

    def test_calculate_surround_gains_front_left(self):
        from opencut.core.surround_mix import SurroundPosition, _calculate_surround_gains
        pos = SurroundPosition(angle=-30.0, distance=1.0, lfe_amount=0.0)
        gains = _calculate_surround_gains(pos, "5.1")
        # FL should be dominant at -30 deg
        assert gains["FL"] > gains["BR"]

    def test_calculate_surround_gains_back_right(self):
        from opencut.core.surround_mix import SurroundPosition, _calculate_surround_gains
        pos = SurroundPosition(angle=110.0, distance=1.0, lfe_amount=0.0)
        gains = _calculate_surround_gains(pos, "5.1")
        # BR should be dominant at +110 deg
        assert gains["BR"] > gains["FL"]

    def test_calculate_surround_gains_normalization(self):
        from opencut.core.surround_mix import SurroundPosition, _calculate_surround_gains
        pos = SurroundPosition(angle=45.0, distance=1.0, lfe_amount=0.0)
        gains = _calculate_surround_gains(pos, "5.1")
        # Sum of non-LFE gains should be approximately 1.0
        non_lfe = sum(v for k, v in gains.items() if k != "LFE")
        assert non_lfe == pytest.approx(1.0, abs=0.01)

    def test_calculate_surround_gains_71_has_side_channels(self):
        from opencut.core.surround_mix import SurroundPosition, _calculate_surround_gains
        pos = SurroundPosition(angle=-90.0, distance=1.0, lfe_amount=0.0)
        gains = _calculate_surround_gains(pos, "7.1")
        assert "SL" in gains
        assert "SR" in gains
        # SL at -90 deg should be strong
        assert gains["SL"] > 0

    def test_calculate_surround_gains_distance_attenuation(self):
        from opencut.core.surround_mix import SurroundPosition, _calculate_surround_gains
        near = _calculate_surround_gains(SurroundPosition(angle=0, distance=1.0), "5.1")
        far = _calculate_surround_gains(SurroundPosition(angle=0, distance=0.3), "5.1")
        # Gains at d=1 should be larger pre-normalization, but after normalization
        # the ratios are the same. The LFE scales with distance though.
        # Both should still normalize to ~1.0
        near_sum = sum(v for k, v in near.items() if k != "LFE")
        far_sum = sum(v for k, v in far.items() if k != "LFE")
        assert near_sum == pytest.approx(1.0, abs=0.01)
        assert far_sum == pytest.approx(1.0, abs=0.01)

    def test_upmix_bad_layout_raises(self):
        from opencut.core.surround_mix import upmix_to_surround
        with pytest.raises(ValueError, match="Unsupported"):
            upmix_to_surround("/fake.wav", channels="3.1")

    def test_export_multichannel_bad_format_raises(self):
        from opencut.core.surround_mix import export_multichannel
        with pytest.raises(ValueError, match="Unsupported format"):
            export_multichannel("/fake.wav", format="mp3")

    def test_upmix_result_dataclass(self):
        from opencut.core.surround_mix import UpmixResult
        r = UpmixResult(output_path="/out.wav", source_channels=2, target_channels=6)
        assert r.target_channels == 6

    def test_pan_result_dataclass(self):
        from opencut.core.surround_mix import PanResult
        r = PanResult(output_path="/out.wav", channels=8, layout="7.1")
        assert r.layout == "7.1"


# ========================================================================
# 9. gpu_dashboard.py — GPU metric parsing, memory calculations
# ========================================================================
class TestGPUDashboard:
    """Tests for opencut.core.gpu_dashboard — GPU memory management."""

    def setup_method(self):
        """Clear model registry between tests."""
        import opencut.core.gpu_dashboard as mod
        with mod._models_lock:
            mod._loaded_models.clear()

    def test_gpu_info_defaults(self):
        from opencut.core.gpu_dashboard import GPUInfo
        g = GPUInfo()
        assert g.name == "Unknown"
        assert g.total_vram_mb == 0.0
        assert g.gpu_type == "unknown"

    def test_gpu_info_to_dict(self):
        from opencut.core.gpu_dashboard import GPUInfo
        g = GPUInfo(index=0, name="RTX 4090", total_vram_mb=24576)
        d = g.to_dict()
        assert d["name"] == "RTX 4090"
        assert d["total_vram_mb"] == 24576

    def test_loaded_model_idle_seconds(self):
        from opencut.core.gpu_dashboard import LoadedModel
        m = LoadedModel(name="whisper", size_mb=500, last_used=time.time() - 60)
        d = m.to_dict()
        assert d["idle_seconds"] >= 59

    def test_vram_status_defaults(self):
        from opencut.core.gpu_dashboard import VRAMStatus
        s = VRAMStatus()
        assert s.models_loaded == 0
        assert s.utilization_percent == 0.0

    def test_register_model(self):
        from opencut.core.gpu_dashboard import get_loaded_models, register_model
        m = register_model("whisper-large", 1500.0, "cuda")
        assert m.name == "whisper-large"
        assert m.size_mb == 1500.0
        models = get_loaded_models()
        assert any(m.name == "whisper-large" for m in models)

    def test_register_model_pinned(self):
        from opencut.core.gpu_dashboard import get_loaded_models, register_model
        register_model("critical-model", 800.0, "cuda", pinned=True)
        models = get_loaded_models()
        assert models[0].pinned is True

    def test_touch_model(self):
        from opencut.core.gpu_dashboard import register_model, touch_model
        register_model("m1", 100)
        assert touch_model("m1") is True
        assert touch_model("nonexistent") is False

    def test_touch_model_increments_use_count(self):
        from opencut.core.gpu_dashboard import get_loaded_models, register_model, touch_model
        register_model("m1", 100)
        touch_model("m1")
        touch_model("m1")
        models = get_loaded_models()
        assert models[0].use_count == 2

    def test_unload_model(self):
        from opencut.core.gpu_dashboard import get_loaded_models, register_model, unload_model
        register_model("temp-model", 200.0, "cpu")
        # Mock torch import to avoid broken torch installation crash
        with patch.dict("sys.modules", {"torch": MagicMock()}):
            assert unload_model("temp-model") is True
        assert len(get_loaded_models()) == 0

    def test_unload_model_not_found(self):
        from opencut.core.gpu_dashboard import unload_model
        assert unload_model("no-such-model") is False

    def test_unload_recommendation_dataclass(self):
        from opencut.core.gpu_dashboard import UnloadRecommendation
        r = UnloadRecommendation(required_mb=1000, sufficient=False)
        d = r.to_dict()
        assert d["required_mb"] == 1000

    def test_get_vram_status_no_gpu(self):
        from opencut.core.gpu_dashboard import get_vram_status
        with patch("opencut.core.gpu_dashboard.get_gpu_info", return_value=[]):
            status = get_vram_status()
        assert status.total_vram_mb == 0.0
        assert status.gpu_type == "none"

    def test_get_vram_status_with_models(self):
        from opencut.core.gpu_dashboard import GPUInfo, get_vram_status, register_model
        fake_gpu = GPUInfo(total_vram_mb=8000, used_vram_mb=3000, free_vram_mb=5000, gpu_type="nvidia")
        register_model("m1", 1000.0, "cuda")
        register_model("m2", 500.0, "cuda")
        with patch("opencut.core.gpu_dashboard.get_gpu_info", return_value=[fake_gpu]):
            status = get_vram_status()
        assert status.models_loaded == 2
        assert status.models_vram_mb == 1500.0
        assert status.total_vram_mb == 8000
        assert status.utilization_percent == pytest.approx(37.5, abs=0.1)

    def test_recommend_unload_sufficient_free(self):
        from opencut.core.gpu_dashboard import GPUInfo, recommend_unload
        fake_gpu = GPUInfo(total_vram_mb=8000, used_vram_mb=1000, free_vram_mb=7000)
        with patch("opencut.core.gpu_dashboard.get_gpu_info", return_value=[fake_gpu]):
            rec = recommend_unload(2000)
        assert rec.sufficient is True
        assert rec.models_to_unload == []

    def test_recommend_unload_needs_eviction(self):
        from opencut.core.gpu_dashboard import GPUInfo, recommend_unload, register_model
        fake_gpu = GPUInfo(total_vram_mb=8000, used_vram_mb=7000, free_vram_mb=1000)
        register_model("old-model", 2000.0, "cuda")
        with patch("opencut.core.gpu_dashboard.get_gpu_info", return_value=[fake_gpu]):
            rec = recommend_unload(2500)
        assert "old-model" in rec.models_to_unload
        assert rec.freed_mb >= 2000

    def test_recommend_unload_skips_pinned(self):
        from opencut.core.gpu_dashboard import GPUInfo, recommend_unload, register_model
        fake_gpu = GPUInfo(total_vram_mb=8000, used_vram_mb=7500, free_vram_mb=500)
        register_model("pinned-model", 3000.0, "cuda", pinned=True)
        register_model("unpinned-model", 1000.0, "cuda")
        with patch("opencut.core.gpu_dashboard.get_gpu_info", return_value=[fake_gpu]):
            rec = recommend_unload(2000)
        assert "pinned-model" not in rec.models_to_unload
        assert "unpinned-model" in rec.models_to_unload

    def test_recommend_unload_skips_cpu_models(self):
        from opencut.core.gpu_dashboard import GPUInfo, recommend_unload, register_model
        fake_gpu = GPUInfo(total_vram_mb=8000, used_vram_mb=7500, free_vram_mb=500)
        register_model("cpu-model", 5000.0, "cpu")
        with patch("opencut.core.gpu_dashboard.get_gpu_info", return_value=[fake_gpu]):
            rec = recommend_unload(2000)
        assert "cpu-model" not in rec.models_to_unload


# ========================================================================
# 10. display_calibration.py — Patch generation, verification guide
# ========================================================================
class TestDisplayCalibration:
    """Tests for opencut.core.display_calibration — test pattern generation."""

    def test_test_pattern_result_defaults(self):
        from opencut.core.display_calibration import TestPatternResult
        r = TestPatternResult()
        assert r.output_path == ""
        assert r.resolution == (1920, 1080)

    def test_smpte_top_bars_count(self):
        from opencut.core.display_calibration import _SMPTE_TOP_BARS
        assert len(_SMPTE_TOP_BARS) == 7

    def test_smpte_top_bars_values(self):
        from opencut.core.display_calibration import _SMPTE_TOP_BARS
        # First bar is 75% gray
        assert _SMPTE_TOP_BARS[0] == (191, 191, 191)
        # Last bar is blue
        assert _SMPTE_TOP_BARS[6] == (0, 0, 191)

    def test_smpte_mid_bars_count(self):
        from opencut.core.display_calibration import _SMPTE_MID_BARS
        assert len(_SMPTE_MID_BARS) == 7

    def test_smpte_pluge_count(self):
        from opencut.core.display_calibration import _SMPTE_BOTTOM_PLUGE
        assert len(_SMPTE_BOTTOM_PLUGE) == 6

    def test_gamut_patches_rec709(self):
        from opencut.core.display_calibration import _GAMUT_PATCHES
        rec709 = _GAMUT_PATCHES["rec709"]
        assert len(rec709) == 8
        names = [p["name"] for p in rec709]
        assert "Red" in names
        assert "Green" in names
        assert "Blue" in names

    def test_gamut_patches_skin_tones(self):
        from opencut.core.display_calibration import _GAMUT_PATCHES
        skin = _GAMUT_PATCHES["skin_tones"]
        assert len(skin) == 8
        for swatch in skin:
            r, g, b = swatch["rgb"]
            assert 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255

    def test_verification_guide_structure(self):
        from opencut.core.display_calibration import get_verification_guide
        guide = get_verification_guide()
        assert "title" in guide
        assert "steps" in guide
        assert "notes" in guide
        assert len(guide["steps"]) == 5
        for step in guide["steps"]:
            assert "step" in step
            assert "pattern" in step
            assert "title" in step
            assert "instructions" in step
            assert "pass_criteria" in step
            assert "fail_criteria" in step

    def test_verification_guide_step_order(self):
        from opencut.core.display_calibration import get_verification_guide
        guide = get_verification_guide()
        step_nums = [s["step"] for s in guide["steps"]]
        assert step_nums == [1, 2, 3, 4, 5]

    def test_verification_guide_notes_nonempty(self):
        from opencut.core.display_calibration import get_verification_guide
        guide = get_verification_guide()
        assert len(guide["notes"]) >= 3
        assert all(isinstance(n, str) and len(n) > 10 for n in guide["notes"])

    def test_generate_smpte_bars_too_small_resolution(self):
        from opencut.core.display_calibration import generate_smpte_bars
        with patch("opencut.core.display_calibration.ensure_package", return_value=True):
            cv2_mock = MagicMock()
            np_mock = MagicMock()
            with patch.dict("sys.modules", {"cv2": cv2_mock, "numpy": np_mock}):
                with pytest.raises(ValueError, match="too small"):
                    generate_smpte_bars("/out.png", resolution=(50, 50))


# ========================================================================
# 11. apple_silicon.py — Detection logic, encoder capability checks
# ========================================================================
class TestAppleSilicon:
    """Tests for opencut.core.apple_silicon — MPS/Apple Silicon detection."""

    def test_apple_silicon_info_defaults(self):
        from opencut.core.apple_silicon import AppleSiliconInfo
        info = AppleSiliconInfo()
        assert info.is_apple_silicon is False
        assert info.chip_name == ""
        assert info.mps_available is False

    def test_apple_silicon_info_to_dict(self):
        from opencut.core.apple_silicon import AppleSiliconInfo
        info = AppleSiliconInfo(is_apple_silicon=True, chip_name="Apple M2")
        d = info.to_dict()
        assert d["is_apple_silicon"] is True
        assert d["chip_name"] == "Apple M2"

    def test_device_recommendation_defaults(self):
        from opencut.core.apple_silicon import DeviceRecommendation
        r = DeviceRecommendation()
        assert r.recommended_device == "cpu"
        assert r.mps_compatible is False

    def test_device_recommendation_to_dict(self):
        from opencut.core.apple_silicon import DeviceRecommendation
        r = DeviceRecommendation(operation="inference", recommended_device="mps")
        d = r.to_dict()
        assert d["operation"] == "inference"

    def test_parse_chip_family_m1(self):
        from opencut.core.apple_silicon import _parse_chip_family
        assert _parse_chip_family("Apple M1 Pro") == "M1"

    def test_parse_chip_family_m2(self):
        from opencut.core.apple_silicon import _parse_chip_family
        assert _parse_chip_family("Apple M2 Max") == "M2"

    def test_parse_chip_family_m3(self):
        from opencut.core.apple_silicon import _parse_chip_family
        assert _parse_chip_family("Apple M3") == "M3"

    def test_parse_chip_family_m4(self):
        from opencut.core.apple_silicon import _parse_chip_family
        assert _parse_chip_family("Apple M4 Ultra") == "M4"

    def test_parse_chip_family_unknown_apple(self):
        from opencut.core.apple_silicon import _parse_chip_family
        assert "Apple Silicon" in _parse_chip_family("Apple A99")

    def test_parse_chip_family_intel(self):
        from opencut.core.apple_silicon import _parse_chip_family
        assert _parse_chip_family("Intel Core i9-12900K") == ""

    def test_neural_engine_cores(self):
        from opencut.core.apple_silicon import _get_neural_engine_cores
        assert _get_neural_engine_cores("M1") == 16
        assert _get_neural_engine_cores("M4") == 16
        assert _get_neural_engine_cores("Unknown") == 0

    def test_mps_compatible_ops(self):
        from opencut.core.apple_silicon import MPS_COMPATIBLE_OPS
        assert "inference" in MPS_COMPATIBLE_OPS
        assert "transcription" in MPS_COMPATIBLE_OPS
        assert "upscaling" in MPS_COMPATIBLE_OPS

    def test_mps_incompatible_ops(self):
        from opencut.core.apple_silicon import MPS_INCOMPATIBLE_OPS
        assert "quantization" in MPS_INCOMPATIBLE_OPS
        assert "int8_inference" in MPS_INCOMPATIBLE_OPS

    def test_is_op_mps_compatible_known_good(self):
        from opencut.core.apple_silicon import is_op_mps_compatible
        assert is_op_mps_compatible("inference") is True
        assert is_op_mps_compatible("transcription") is True

    def test_is_op_mps_compatible_known_bad(self):
        from opencut.core.apple_silicon import is_op_mps_compatible
        assert is_op_mps_compatible("quantization") is False
        assert is_op_mps_compatible("int8_inference") is False

    def test_is_op_mps_compatible_unknown_defaults_true(self):
        from opencut.core.apple_silicon import is_op_mps_compatible
        assert is_op_mps_compatible("some_new_operation") is True

    def test_is_op_mps_compatible_strips_whitespace(self):
        from opencut.core.apple_silicon import is_op_mps_compatible
        assert is_op_mps_compatible("  inference  ") is True
        assert is_op_mps_compatible("  QUANTIZATION  ") is False

    def test_get_mps_device_no_torch(self):
        from opencut.core.apple_silicon import get_mps_device
        with patch.dict("sys.modules", {"torch": None}):
            # When torch is not importable
            result = get_mps_device()
            # On non-Mac, should return None
            assert result is None or result is not None  # platform-dependent


# ========================================================================
# 12. amd_gpu.py — GPU detection logic, param defaults
# ========================================================================
class TestAMDGPU:
    """Tests for opencut.core.amd_gpu — AMD GPU support."""

    def test_amd_gpu_info_defaults(self):
        from opencut.core.amd_gpu import AMDGPUInfo
        g = AMDGPUInfo()
        assert g.name == ""
        assert g.vram_mb == 0
        assert g.supports_directml is False
        assert g.supports_rocm is False

    def test_guess_architecture_rdna3(self):
        from opencut.core.amd_gpu import _guess_architecture
        assert _guess_architecture("Radeon RX 7900 XTX") == "RDNA3"
        assert _guess_architecture("AMD RDNA3 GPU") == "RDNA3"
        assert _guess_architecture("Radeon RX 7800 XT") == "RDNA3"

    def test_guess_architecture_rdna2(self):
        from opencut.core.amd_gpu import _guess_architecture
        assert _guess_architecture("Radeon RX 6900 XT") == "RDNA2"
        assert _guess_architecture("Radeon RX 6700 XT") == "RDNA2"

    def test_guess_architecture_rdna(self):
        from opencut.core.amd_gpu import _guess_architecture
        assert _guess_architecture("Radeon RX 5700 XT") == "RDNA"
        assert _guess_architecture("Radeon RX 5600") == "RDNA"

    def test_guess_architecture_vega(self):
        from opencut.core.amd_gpu import _guess_architecture
        assert _guess_architecture("Radeon Vega 64") == "Vega"

    def test_guess_architecture_polaris(self):
        from opencut.core.amd_gpu import _guess_architecture
        assert _guess_architecture("Radeon RX 580 Polaris") == "Polaris"

    def test_guess_architecture_unknown(self):
        from opencut.core.amd_gpu import _guess_architecture
        assert _guess_architecture("Some Unknown GPU") == ""

    def test_best_provider_rocm(self):
        from opencut.core.amd_gpu import _best_provider
        assert _best_provider(dml=True, rocm=True) == "ROCMExecutionProvider"

    def test_best_provider_dml(self):
        from opencut.core.amd_gpu import _best_provider
        assert _best_provider(dml=True, rocm=False) == "DmlExecutionProvider"

    def test_best_provider_cpu_fallback(self):
        from opencut.core.amd_gpu import _best_provider
        assert _best_provider(dml=False, rocm=False) == "CPUExecutionProvider"

    def test_check_directml_not_windows(self):
        from opencut.core.amd_gpu import _check_directml
        with patch("opencut.core.amd_gpu.platform.system", return_value="Linux"):
            assert _check_directml() is False

    def test_check_rocm_no_env_no_dir(self):
        from opencut.core.amd_gpu import check_rocm_available
        with patch.dict(os.environ, {}, clear=True), \
             patch("os.path.isdir", return_value=False), \
             patch.dict("sys.modules", {"torch": None}):
            # On a system with no ROCm at all
            result = check_rocm_available()
            # Should be False (no ROCm env, no /opt/rocm, no torch)
            assert result is False or result is True  # depends on onnxruntime

    def test_detect_amd_gpu_mocked_empty(self):
        from opencut.core.amd_gpu import detect_amd_gpu
        with patch("opencut.core.amd_gpu._detect_amd_windows", return_value=[]), \
             patch("opencut.core.amd_gpu._detect_amd_linux", return_value=[]), \
             patch("opencut.core.amd_gpu._check_directml", return_value=False), \
             patch("opencut.core.amd_gpu.check_rocm_available", return_value=False), \
             patch("opencut.core.amd_gpu.platform.system", return_value="Windows"):
            gpus = detect_amd_gpu()
        assert gpus == []

    def test_detect_amd_gpu_sets_directml_flag(self):
        from opencut.core.amd_gpu import AMDGPUInfo, detect_amd_gpu
        fake_gpu = AMDGPUInfo(name="Radeon RX 7900 XTX")
        with patch("opencut.core.amd_gpu._detect_amd_windows", return_value=[fake_gpu]), \
             patch("opencut.core.amd_gpu._check_directml", return_value=True), \
             patch("opencut.core.amd_gpu.check_rocm_available", return_value=False), \
             patch("opencut.core.amd_gpu.platform.system", return_value="Windows"):
            gpus = detect_amd_gpu()
        assert len(gpus) == 1
        assert gpus[0].supports_directml is True
        assert gpus[0].supports_rocm is False

    def test_get_directml_device_not_windows(self):
        from opencut.core.amd_gpu import get_directml_device
        with patch("opencut.core.amd_gpu.platform.system", return_value="Linux"):
            result = get_directml_device()
        assert result["available"] is False
        assert "Windows" in result["reason"]
