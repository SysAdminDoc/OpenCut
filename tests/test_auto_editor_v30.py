"""
Tests for auto-editor v30 native binary compatibility.

Validates that v30 JSON output is correctly parsed by _parse_auto_editor_json
and that detect_auto_editor_generation distinguishes v30 from v29.
"""

import json
from unittest.mock import MagicMock, patch

from opencut.core.auto_edit import (
    EditSegment,
    _parse_auto_editor_json,
    check_auto_editor_version,
    detect_auto_editor_generation,
)


class TestV30JsonParsing:
    """Validate parsing of auto-editor v30 (Nim native) JSON output."""

    def test_v30_timeline_format(self, tmp_path):
        """v30 uses timeline.v tracks with tb (timebase) field."""
        data = {
            "timeline": {
                "v": [[
                    {"offset": 0, "dur": 900, "speed": 1.0, "tb": 30},
                    {"offset": 900, "dur": 150, "speed": 99999, "tb": 30},
                    {"offset": 1050, "dur": 450, "speed": 1.0, "tb": 30},
                ]],
                "a": [[
                    {"offset": 0, "dur": 900, "speed": 1.0, "tb": 48000},
                    {"offset": 900, "dur": 150, "speed": 99999, "tb": 48000},
                    {"offset": 1050, "dur": 450, "speed": 1.0, "tb": 48000},
                ]],
            }
        }
        json_file = str(tmp_path / "v30.json")
        with open(json_file, "w") as f:
            json.dump(data, f)

        segments = _parse_auto_editor_json(json_file, 50.0)
        assert len(segments) == 3
        assert segments[0].action == "keep"
        assert abs(segments[0].start - 0.0) < 0.01
        assert abs(segments[0].end - 30.0) < 0.01
        assert segments[1].action == "cut"
        assert abs(segments[1].start - 30.0) < 0.01
        assert abs(segments[1].end - 35.0) < 0.01
        assert segments[2].action == "keep"

    def test_v30_single_keep_segment(self, tmp_path):
        """v30 output with no cuts (all content kept)."""
        data = {
            "timeline": {
                "v": [[
                    {"offset": 0, "dur": 3000, "speed": 1.0, "tb": 30},
                ]],
            }
        }
        json_file = str(tmp_path / "v30_nocut.json")
        with open(json_file, "w") as f:
            json.dump(data, f)

        segments = _parse_auto_editor_json(json_file, 100.0)
        assert len(segments) == 1
        assert segments[0].action == "keep"
        assert abs(segments[0].end - 100.0) < 0.01

    def test_v30_speed_zero_is_cut(self, tmp_path):
        """Speed 0 should be treated as cut (not keep)."""
        data = {
            "timeline": {
                "v": [[
                    {"offset": 0, "dur": 300, "speed": 0, "tb": 30},
                    {"offset": 300, "dur": 300, "speed": 1.0, "tb": 30},
                ]],
            }
        }
        json_file = str(tmp_path / "v30_speed0.json")
        with open(json_file, "w") as f:
            json.dump(data, f)

        segments = _parse_auto_editor_json(json_file, 20.0)
        assert segments[0].action == "cut"
        assert segments[1].action == "keep"

    def test_v30_fractional_timebase(self, tmp_path):
        """Handle non-standard timebase values."""
        data = {
            "timeline": {
                "v": [[
                    {"offset": 0, "dur": 24000, "speed": 1.0, "tb": 24000},
                ]],
            }
        }
        json_file = str(tmp_path / "v30_tb.json")
        with open(json_file, "w") as f:
            json.dump(data, f)

        segments = _parse_auto_editor_json(json_file, 1.0)
        assert len(segments) == 1
        assert abs(segments[0].end - 1.0) < 0.01


class TestDetectAutoEditorGeneration:
    """Test detect_auto_editor_generation() v30/v29 discrimination."""

    def test_v30_native_detected(self):
        mock_result = MagicMock(returncode=0, stdout="30.5.0\n")
        with patch("opencut.core.auto_edit.shutil.which", return_value="/usr/bin/auto-editor"):
            with patch("opencut.core.auto_edit.subprocess.run", return_value=mock_result):
                info = detect_auto_editor_generation()
        assert info["generation"] == "v30"
        assert info["native"] is True
        assert info["version"] == "30.5.0"

    def test_v29_pip_detected(self):
        mock_result = MagicMock(returncode=0, stdout="24w51a\n")
        with patch("opencut.core.auto_edit.shutil.which", return_value=None):
            with patch("opencut.core.auto_edit.subprocess.run", return_value=mock_result):
                info = detect_auto_editor_generation()
        assert info["generation"] == "v29"
        assert info["native"] is False
        assert info["version"] == "24w51a"

    def test_not_installed(self):
        with patch("opencut.core.auto_edit.shutil.which", return_value=None):
            with patch("opencut.core.auto_edit.subprocess.run", side_effect=FileNotFoundError):
                info = detect_auto_editor_generation()
        assert info["generation"] is None
        assert info["version"] is None

    def test_v30_high_version(self):
        mock_result = MagicMock(returncode=0, stdout="31.0.0\n")
        with patch("opencut.core.auto_edit.shutil.which", return_value="C:\\auto-editor.exe"):
            with patch("opencut.core.auto_edit.subprocess.run", return_value=mock_result):
                info = detect_auto_editor_generation()
        assert info["generation"] == "v30"
        assert info["native"] is True
