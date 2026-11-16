"""The concat demuxer flags have to sit in front of the list's ``-i``.

``FFmpegCmd.option()`` writes its flags after every input, where FFmpeg reads
``-f concat`` as the output format and refuses the whole command. Twelve
modules joined their segments that way, and their tests mocked ``run_ffmpeg``,
so the joins failed on every real render while the suite stayed green.
"""
import ast
from pathlib import Path

import pytest

from opencut.helpers import FFmpegCmd
from tests.media_corpus import (
    DURATION_TOLERANCE_S,
    build_one,
    ffmpeg_available,
    format_duration,
)

REPO = Path(__file__).resolve().parents[1]


def _concat_output_options(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "option" or len(node.args) < 2:
            continue
        key, value = node.args[:2]
        if (isinstance(key, ast.Constant) and key.value == "f"
                and isinstance(value, ast.Constant) and value.value == "concat"):
            yield node.lineno


def test_concat_flags_come_before_the_list_input():
    cmd = (FFmpegCmd()
           .input("segments.txt", f="concat", safe="0")
           .copy_streams()
           .output("joined.mp4")
           .build())
    at = cmd.index("segments.txt")
    assert cmd[at - 5:at + 1] == ["-f", "concat", "-safe", "0", "-i", "segments.txt"]


def test_no_module_passes_the_concat_demuxer_as_an_output_option():
    offenders = [
        f"{path.relative_to(REPO).as_posix()}:{line}"
        for path in sorted((REPO / "opencut").rglob("*.py"))
        for line in _concat_output_options(path)
    ]
    assert offenders == [], (
        "FFmpegCmd.option() lands after the inputs, so FFmpeg takes "
        "'-f concat' as the output format. Pass it on the list instead: "
        ".input(list_path, f=\"concat\", safe=\"0\")"
    )


@pytest.mark.skipif(not ffmpeg_available(),
                    reason="needs FFmpeg at or above the version floor")
def test_rough_cut_joins_real_segments(tmp_path):
    from opencut.core.rough_cut import PlannedClip, RoughCutPlan, execute_plan

    source = build_one(tmp_path / "corpus", "cfr_25")
    plan = RoughCutPlan(
        clips=[
            PlannedClip(source_file=source, start=0.0, end=1.0, order=0),
            PlannedClip(source_file=source, start=1.0, end=2.0, order=1),
        ],
        total_duration=2.0,
    )
    out = tmp_path / "rough_cut.mp4"

    result = execute_plan(plan, out_path=str(out))

    assert result.output_path == str(out)
    assert out.is_file()
    assert abs(format_duration(out) - 2.0) <= DURATION_TOLERANCE_S
