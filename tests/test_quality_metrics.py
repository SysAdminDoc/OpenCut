"""Focused unit coverage for objective FFmpeg quality metrics."""
from __future__ import annotations

import json
import os
import subprocess


def test_vmaf_escapes_apostrophe_in_log_path(monkeypatch, tmp_path):
    from opencut.core import quality_metrics
    from opencut.helpers import escape_filter_path

    apostrophe_dir = tmp_path / "O'Brien"
    apostrophe_dir.mkdir()
    json_path = apostrophe_dir / "vmaf.json"
    expected_path = str(json_path)
    fd = os.open(expected_path, os.O_CREAT | os.O_RDWR)

    monkeypatch.setattr(
        quality_metrics.tempfile,
        "mkstemp",
        lambda **_: (fd, expected_path),
    )
    monkeypatch.setattr(quality_metrics, "check_vmaf_available", lambda: True)

    def fake_run(_distorted, _reference, filter_complex, timeout):
        assert f"log_path='{escape_filter_path(expected_path)}'" in filter_complex
        with open(expected_path, "w", encoding="utf-8") as output:
            json.dump(
                {
                    "version": "libvmaf-test",
                    "pooled_metrics": {
                        "vmaf": {"mean": 95.5, "min": 90.0, "harmonic_mean": 94.0},
                    },
                    "frames": [{}],
                },
                output,
            )
        return ""

    monkeypatch.setattr(quality_metrics, "_run_ffmpeg_filter_complex", fake_run)

    result = quality_metrics.measure_vmaf("distorted.mp4", "reference.mp4")

    assert result["mean"] == 95.5
    assert result["model"] == quality_metrics.VMAF_MODEL


def test_compare_videos_isolates_metric_timeout(monkeypatch, tmp_path):
    from opencut.core import quality_metrics

    distorted = tmp_path / "distorted.mp4"
    reference = tmp_path / "reference.mp4"
    distorted.touch()
    reference.touch()

    monkeypatch.setattr(quality_metrics, "check_quality_metrics_available", lambda: True)
    monkeypatch.setattr(
        "opencut.helpers.get_video_info",
        lambda _path: {"duration": 1.0},
    )

    def timed_out(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("ffmpeg", 1)

    monkeypatch.setattr(quality_metrics, "measure_vmaf", timed_out)
    monkeypatch.setattr(quality_metrics, "measure_ssim", lambda *_args, **_kwargs: 0.97)

    report = quality_metrics.compare_videos(
        str(distorted),
        str(reference),
        metrics=["vmaf", "ssim"],
        timeout=1,
    )

    assert report.vmaf is None
    assert report.ssim == 0.97
    assert any(note.startswith("vmaf:") and "timed out" in note for note in report.notes)
