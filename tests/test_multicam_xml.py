"""Tests for multicam XML export."""

import os


class TestMulticamXml:
    """Test multicam XML generation."""

    def test_basic_generation(self):
        from opencut.core.multicam_xml import generate_multicam_xml

        cuts = [
            {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00", "track": 1},
            {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_01", "track": 2},
            {"start": 10.0, "end": 15.0, "speaker": "SPEAKER_00", "track": 1},
        ]
        source_files = {
            "SPEAKER_00": "/path/to/cam1.mp4",
            "SPEAKER_01": "/path/to/cam2.mp4",
        }

        result = generate_multicam_xml(cuts, source_files)

        assert result["cuts_count"] == 3
        assert result["duration"] == 15.0
        assert "<xmeml" in result["xml"]
        assert "SPEAKER_00" in result["xml"]
        assert result["output"] is None  # No file written

    def test_write_to_file(self, tmp_path):
        from opencut.core.multicam_xml import generate_multicam_xml

        cuts = [
            {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00", "track": 1},
        ]
        output = str(tmp_path / "test_multicam.xml")

        result = generate_multicam_xml(
            cuts, {"SPEAKER_00": "/cam1.mp4"}, output_path=output
        )

        assert result["output"] == output
        assert os.path.isfile(output)
        with open(output, "r") as f:
            content = f.read()
        assert "<xmeml" in content

    def test_list_source_files(self):
        from opencut.core.multicam_xml import generate_multicam_xml

        cuts = [
            {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00", "track": 1},
        ]
        result = generate_multicam_xml(cuts, ["/cam1.mp4", "/cam2.mp4"])
        assert result["cuts_count"] == 1

    def test_empty_cuts(self):
        from opencut.core.multicam_xml import generate_multicam_xml

        result = generate_multicam_xml([], {})
        assert result["cuts_count"] == 0
        assert result["duration"] == 0

    def test_custom_fps(self):
        from opencut.core.multicam_xml import generate_multicam_xml

        cuts = [{"start": 0, "end": 1, "speaker": "S", "track": 1}]
        result = generate_multicam_xml(cuts, {"S": "/a.mp4"}, fps=24)
        assert "24" in result["xml"]

    def test_path_to_url_windows(self):
        from opencut.core.multicam_xml import _path_to_url

        url = _path_to_url("C:\\Users\\test\\video.mp4")
        assert url.startswith("file://localhost/")
        assert "C:/Users/test/video.mp4" in url

    def test_path_to_url_unix(self):
        from opencut.core.multicam_xml import _path_to_url

        url = _path_to_url("/home/user/video.mp4")
        assert url == "file://localhost/home/user/video.mp4"

    def test_ntsc_flag(self):
        from opencut.core.multicam_xml import generate_multicam_xml

        cuts = [{"start": 0, "end": 1, "speaker": "S", "track": 1}]
        # 29.97 is NTSC
        result = generate_multicam_xml(cuts, {"S": "/a.mp4"}, fps=29.97)
        assert "TRUE" in result["xml"]  # ntsc=TRUE

        # 25.0 is not NTSC
        result2 = generate_multicam_xml(cuts, {"S": "/a.mp4"}, fps=25.0)
        assert "FALSE" in result2["xml"]  # ntsc=FALSE
