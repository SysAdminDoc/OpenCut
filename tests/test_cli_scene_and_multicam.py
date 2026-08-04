"""`opencut scene-detect` and `opencut podcast` crashed on valid input.

`scene-detect` passed a `method=` keyword no backend accepts, so both the
default `ffmpeg` run and `pyscenedetect` raised `TypeError`; `--method ml`
survived only to normalise a `SceneInfo` dataclass as list-or-dict and report
"Scenes found: 0" for every video. `podcast` handed a `list[TimeSegment]` to
`generate_multicam_xml`, which immediately called `.get()` on it and threw the
whole diarization pass away at the last step.
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from opencut import cli as cli_module
from opencut.core.multicam_xml import generate_multicam_xml, normalize_cuts
from opencut.core.scene_detect import SceneBoundary, SceneInfo
from opencut.core.silence import TimeSegment


def _scene_info() -> SceneInfo:
    return SceneInfo(
        boundaries=[
            SceneBoundary(time=0.0, frame=0, score=1.0, label="Start"),
            SceneBoundary(time=4.5, frame=135, score=0.4, label="Scene 2"),
            SceneBoundary(time=9.0, frame=270, score=0.6, label="Scene 3"),
        ],
        total_scenes=3,
        duration=12.0,
        avg_scene_length=4.0,
    )


class TestSceneDetectCommand:
    @pytest.mark.parametrize("method", ["ffmpeg", "ml", "pyscenedetect"])
    def test_every_method_dispatches_to_its_own_backend(self, method, tmp_path, monkeypatch):
        calls: dict[str, object] = {}

        def _record(name):
            def _detector(input_path, **kwargs):
                calls["backend"] = name
                calls["kwargs"] = kwargs
                return _scene_info()
            return _detector

        monkeypatch.setattr(
            "opencut.core.scene_detect.detect_scenes", _record("ffmpeg"), raising=True
        )
        monkeypatch.setattr(
            "opencut.core.scene_detect.detect_scenes_ml", _record("ml"), raising=True
        )
        monkeypatch.setattr(
            "opencut.core.scene_detect.detect_scenes_pyscenedetect",
            _record("pyscenedetect"),
            raising=True,
        )

        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"\x00")
        out = tmp_path / "scenes.json"

        result = CliRunner().invoke(
            cli_module.cli,
            ["scene-detect", str(clip), "--method", method, "-o", str(out)],
        )

        assert result.exit_code == 0, result.output
        assert calls["backend"] == method
        # No `method=` keyword: no backend accepts one.
        assert calls["kwargs"] == {}

    def test_omitting_threshold_leaves_each_backend_on_its_own_scale(self, tmp_path, monkeypatch):
        seen: dict[str, object] = {}

        def _detector(input_path, **kwargs):
            seen.update(kwargs)
            return _scene_info()

        monkeypatch.setattr(
            "opencut.core.scene_detect.detect_scenes_pyscenedetect", _detector, raising=True
        )
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"\x00")

        result = CliRunner().invoke(
            cli_module.cli, ["scene-detect", str(clip), "--method", "pyscenedetect"]
        )
        assert result.exit_code == 0, result.output
        assert "threshold" not in seen

        seen.clear()
        result = CliRunner().invoke(
            cli_module.cli,
            ["scene-detect", str(clip), "--method", "pyscenedetect", "--threshold", "27"],
        )
        assert result.exit_code == 0, result.output
        assert seen == {"threshold": 27.0}

    def test_ml_method_reports_the_real_boundary_count(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "opencut.core.scene_detect.detect_scenes_ml",
            lambda input_path, **kwargs: _scene_info(),
            raising=True,
        )
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"\x00")
        out = tmp_path / "scenes.json"

        result = CliRunner().invoke(
            cli_module.cli, ["scene-detect", str(clip), "--method", "ml", "-o", str(out)]
        )

        assert result.exit_code == 0, result.output
        assert "Scenes found: 3" in result.output

        rows = json.loads(out.read_text(encoding="utf-8"))
        assert [row["start"] for row in rows] == [0.0, 4.5, 9.0]
        # The final scene runs to the end of the clip, not to nothing.
        assert rows[-1]["end"] == 12.0
        assert rows[0]["end"] == 4.5


def test_every_cli_subcommand_is_loadable():
    """A registration or import break in any subcommand fails here first.

    Driving every subcommand against real media is not possible in the suite
    (several need models, a HuggingFace token, or a running backend), so this
    covers the cheap half: every command resolves and renders its own help.
    """
    runner = CliRunner()
    names = sorted(cli_module.cli.commands)
    assert len(names) >= 16, names

    for name in names:
        result = runner.invoke(cli_module.cli, [name, "--help"])
        assert result.exit_code == 0, f"{name} --help failed: {result.output}"
        assert result.output.strip(), f"{name} --help printed nothing"


class TestSceneBoundaryRows:
    def test_boundaries_become_contiguous_ranges(self):
        rows = cli_module._scene_boundaries_to_rows(_scene_info())
        assert [(r["start"], r["end"]) for r in rows] == [(0.0, 4.5), (4.5, 9.0), (9.0, 12.0)]

    def test_empty_result_is_not_a_crash(self):
        assert cli_module._scene_boundaries_to_rows(SceneInfo()) == []


class TestMulticamCutNormalization:
    def test_time_segments_from_diarization_are_accepted(self):
        cuts = [
            TimeSegment(start=0.0, end=3.0, label="camera_0"),
            TimeSegment(start=3.0, end=7.5, label="camera_1"),
        ]

        assert normalize_cuts(cuts) == [
            {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00", "track": 1},
            {"start": 3.0, "end": 7.5, "speaker": "SPEAKER_01", "track": 2},
        ]

    def test_generate_multicam_cuts_shape_is_accepted(self):
        # `multicam.generate_multicam_cuts` emits time/duration, not start/end.
        cuts = [{"time": 2.0, "duration": 4.0, "speaker": "SPEAKER_01", "track": 2}]

        assert normalize_cuts(cuts) == [
            {"start": 2.0, "end": 6.0, "speaker": "SPEAKER_01", "track": 2}
        ]

    def test_canonical_dicts_pass_through_unchanged(self):
        cuts = [{"start": 1.0, "end": 2.0, "speaker": "SPEAKER_00", "track": 1}]
        assert normalize_cuts(cuts) == cuts

    def test_xml_generation_from_time_segments_produces_real_clips(self, tmp_path):
        output = tmp_path / "multicam.xml"
        result = generate_multicam_xml(
            cuts=[
                TimeSegment(start=0.0, end=3.0, label="camera_0"),
                TimeSegment(start=3.0, end=7.5, label="camera_1"),
            ],
            source_files={"SPEAKER_00": "cam1.mp4", "SPEAKER_01": "cam2.mp4"},
            fps=30.0,
            output_path=str(output),
        )

        assert result["cuts_count"] == 2
        assert result["duration"] == pytest.approx(7.5)
        assert output.is_file()
        # 7.5s at 30fps: the sequence is not the zero-length one the old
        # `.get("end", 0)` path produced.
        assert "<duration>225</duration>" in result["xml"]


def test_auto_zoom_cli_apply_uses_shared_tracking_filter(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake")
    keyframes = {
        "fps": 24.0,
        "keyframes": [
            {"time": 0.0, "scale": 1.0, "anchor_x": 0.82, "anchor_y": 0.76},
        ],
    }
    captured = {}

    monkeypatch.setattr(
        "opencut.core.auto_zoom.generate_zoom_keyframes",
        lambda *_args, **_kwargs: keyframes,
    )

    def fake_filter(samples, width, height, fps):
        captured["args"] = (samples, width, height, fps)
        return "tracked-zoompan"

    monkeypatch.setattr("opencut.core.auto_zoom.build_zoompan_filter", fake_filter)
    monkeypatch.setattr(
        "opencut.helpers.get_video_info",
        lambda _path: {"width": 1920, "height": 1080, "fps": 24.0},
    )
    monkeypatch.setattr("opencut.helpers.get_ffmpeg_path", lambda: "ffmpeg")
    monkeypatch.setattr("opencut.helpers.run_ffmpeg", lambda command: captured.update(command=command))

    result = CliRunner().invoke(cli_module.cli, ["auto-zoom", str(clip), "--apply"])

    assert result.exit_code == 0, result.output
    assert captured["args"] == (keyframes["keyframes"], 1920, 1080, keyframes["fps"])
    assert "tracked-zoompan" in captured["command"]
