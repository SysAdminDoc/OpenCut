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


# What auto-editor 31.8.0 wrote for a 9 s, 30 fps clip with a tone from 3 s
# to 6 s (`--edit audio --export v1`). Chunk bounds are frames, not seconds.
REAL_V1_EXPORT = {
    "version": "1",
    "source": "clip.mp4",
    "timebase": "30/1",
    "chunks": [[0, 84, 99999.0], [84, 186, 1.0], [186, 270, 99999.0]],
}


class TestV1Export:
    def test_chunks_are_frames_of_the_timebase(self, tmp_path):
        json_file = tmp_path / "v1.json"
        json_file.write_text(json.dumps(REAL_V1_EXPORT), encoding="utf-8")

        segments = _parse_auto_editor_json(str(json_file), 9.0, 30.0)

        assert [(s.action, round(s.start, 3), round(s.end, 3)) for s in segments] == [
            ("cut", 0.0, 2.8), ("keep", 2.8, 6.2), ("cut", 6.2, 9.0),
        ]

    def test_without_a_timebase_the_source_fps_is_used(self, tmp_path):
        # 29.3.1 and 30.0.0 leave "timebase" out of the v1 export.
        data = {k: v for k, v in REAL_V1_EXPORT.items() if k != "timebase"}
        data["chunks"] = [[0, 50, 99999.0], [50, 100, 1.0]]
        json_file = tmp_path / "v1.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")

        segments = _parse_auto_editor_json(str(json_file), 4.0, 25.0)

        assert [(s.action, s.start, s.end) for s in segments] == [
            ("cut", 0.0, 2.0), ("keep", 2.0, 4.0),
        ]

    def test_ntsc_timebase_fraction(self, tmp_path):
        data = dict(REAL_V1_EXPORT, timebase="30000/1001", chunks=[[0, 30000, 1.0]])
        json_file = tmp_path / "v1.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")

        segments = _parse_auto_editor_json(str(json_file), 1001.0, 29.97)

        assert abs(segments[0].end - 1001.0) < 1e-6


class _FakeAutoEditor:
    """Stands in for subprocess.run: records argv and writes a v1 export."""

    def __init__(self, chunks_by_edit):
        self.chunks_by_edit = chunks_by_edit
        self.calls = []

    def __call__(self, cmd, **kwargs):
        # 31.x draws its progress bar in UTF-8. A cp1252 decode kills the
        # reader thread and loses stderr when a run fails.
        assert kwargs.get("encoding") == "utf-8"
        self.calls.append(cmd)
        edit = cmd[cmd.index("--edit") + 1]
        output = cmd[cmd.index("--output") + 1]
        data = dict(REAL_V1_EXPORT, chunks=self.chunks_by_edit[edit.split(":")[0]])
        with open(output, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return MagicMock(returncode=0, stdout="", stderr="")


def _run(tmp_path, fake, method, version):
    from opencut.core.auto_edit import _run_auto_edit

    with patch("opencut.core.auto_edit._find_auto_editor", return_value=["auto-editor"]):
        with patch("opencut.core.auto_edit.subprocess.run", side_effect=fake):
            return _run_auto_edit(
                "clip.mp4", method, 0.02, 0.2, 0.5, False, str(tmp_path),
                str(tmp_path / "auto_edit_result.json"), 9.0, 30.0, None,
                version=version,
            )


class TestAutoEditorCommand:
    AUDIO = [[0, 84, 99999.0], [84, 186, 1.0], [186, 270, 99999.0]]
    MOTION = [[0, 150, 1.0], [150, 270, 99999.0]]

    def test_current_binary_gets_options_it_accepts(self, tmp_path):
        fake = _FakeAutoEditor({"motion": self.MOTION})

        _run(tmp_path, fake, "motion", "31.8.0")

        (cmd,) = fake.calls
        assert cmd[cmd.index("--export") + 1] == "v1"
        assert cmd[cmd.index("--smooth") + 1] == "0.2s,0.5s"
        assert cmd[cmd.index("--edit") + 1] == "motion:threshold=0.02"
        assert "--min-clip-length" not in cmd

    def test_29x_binary_gets_no_smooth_option(self, tmp_path):
        # 29.3.1 rejects --smooth and --min-clip-length alike.
        fake = _FakeAutoEditor({"audio": self.AUDIO})

        _run(tmp_path, fake, "audio", "29.3.1")

        (cmd,) = fake.calls
        assert "--smooth" not in cmd
        assert "--min-clip-length" not in cmd
        assert cmd[cmd.index("--export") + 1] == "v1"

    def test_both_keeps_what_either_pass_kept(self, tmp_path):
        # Two --edit flags mean the last one wins, so "both" runs one pass
        # per method and keeps the union.
        fake = _FakeAutoEditor({"motion": self.MOTION, "audio": self.AUDIO})

        result = _run(tmp_path, fake, "both", "31.8.0")

        assert sorted(c[c.index("--edit") + 1] for c in fake.calls) == [
            "audio", "motion:threshold=0.02",
        ]
        assert all(c.count("--edit") == 1 for c in fake.calls)
        assert [(s.action, round(s.start, 3), round(s.end, 3)) for s in result.segments] == [
            ("keep", 0.0, 6.2), ("cut", 6.2, 9.0),
        ]
        assert round(result.kept_duration, 3) == 6.2
