"""F238 standards tests for photosensitive flash checking."""

from __future__ import annotations

from unittest.mock import patch

import pytest


def _video_file(tmp_path):
    path = tmp_path / "flash.mp4"
    path.write_bytes(b"\x00" * 128)
    return path


def _alternating_blocks(low: float, high: float, *, block: int, frames: int) -> list[float]:
    values = []
    for index in range(frames):
        values.append(high if (index // block) % 2 else low)
    return values


def _red_blocks(*, block: int, frames: int) -> list[tuple[float, float, float]]:
    values = []
    for index in range(frames):
        values.append((1.0, 0.0, 0.0) if (index // block) % 2 else (0.0, 0.0, 0.0))
    return values


@patch("opencut.core.accessibility._extract_frame_rgb_average", return_value=[])
@patch("opencut.core.accessibility._extract_frame_luminance")
@patch("opencut.core.accessibility.get_video_info")
def test_bt1702_flags_fast_flash_pairs_with_60hz_gap_rule(mock_info, mock_lum, _mock_rgb, tmp_path):
    from opencut.core.accessibility import detect_flashing

    mock_info.return_value = {"width": 1920, "height": 1080, "fps": 60.0, "duration": 1.5}
    mock_lum.return_value = _alternating_blocks(20.0, 230.0, block=5, frames=90)

    result = detect_flashing(str(_video_file(tmp_path)), min_luminance_change=0.1)

    assert result["standard_profile"] == "bt1702-3"
    assert result["frame_rate_profile"] == "60hz"
    assert result["thresholds"]["applied_safe_gap_ms"] == 334.0
    assert result["general_flash_count"] > 3
    assert result["risk_assessment"] in {"warning", "dangerous"}
    assert result["events"][0]["flash_type"] == "general"
    assert result["events"][0]["min_gap_ms"] < 334.0


@patch("opencut.core.accessibility._extract_frame_rgb_average", return_value=[])
@patch("opencut.core.accessibility._extract_frame_luminance")
@patch("opencut.core.accessibility.get_video_info")
def test_bt1702_allows_flashes_separated_by_334ms_at_60hz(mock_info, mock_lum, _mock_rgb, tmp_path):
    from opencut.core.accessibility import detect_flashing

    mock_info.return_value = {"width": 1920, "height": 1080, "fps": 60.0, "duration": 1.5}
    values = [20.0] * 90
    for frame in (10, 31, 52, 73):
        values[frame] = 230.0
    mock_lum.return_value = values

    result = detect_flashing(str(_video_file(tmp_path)), min_luminance_change=0.1)

    assert result["general_flash_count"] == 4
    assert result["events"] == []
    assert result["risk_assessment"] == "safe"


@patch("opencut.core.accessibility._extract_frame_rgb_average")
@patch("opencut.core.accessibility._extract_frame_luminance")
@patch("opencut.core.accessibility.get_video_info")
def test_japan_isolated_red_threshold_flags_saturated_red_pairs(mock_info, mock_lum, mock_rgb, tmp_path):
    from opencut.core.accessibility import detect_flashing

    mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 2.0}
    mock_lum.return_value = [80.0] * 60
    mock_rgb.return_value = _red_blocks(block=3, frames=60)

    result = detect_flashing(
        str(_video_file(tmp_path)),
        min_luminance_change=0.5,
        standard_profile="japan-animation",
    )

    assert result["standard_profile"] == "japan-animation"
    assert result["red_flash_count"] > 3
    assert any(event["flash_type"] == "red" for event in result["events"])
    assert result["thresholds"]["red_ratio_threshold"] == 0.8
    assert result["thresholds"]["red_signal_delta_threshold"] == 20


@patch("opencut.core.accessibility._extract_frame_rgb_average", return_value=[])
@patch("opencut.core.accessibility._extract_frame_luminance")
@patch("opencut.core.accessibility.get_video_info")
def test_area_threshold_keeps_small_flashes_below_failure(mock_info, mock_lum, _mock_rgb, tmp_path):
    from opencut.core.accessibility import detect_flashing

    mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 2.0}
    mock_lum.return_value = _alternating_blocks(20.0, 230.0, block=2, frames=60)

    result = detect_flashing(
        str(_video_file(tmp_path)),
        min_luminance_change=0.1,
        screen_area_ratio=0.10,
    )

    assert result["general_flash_count"] > 3
    assert result["thresholds"]["area_threshold_ratio"] == 0.25
    assert result["events"] == []
    assert result["risk_assessment"] == "safe"


def test_unknown_flash_profile_rejected(tmp_path):
    from opencut.core.accessibility import detect_flashing

    with pytest.raises(ValueError, match="standard_profile"):
        detect_flashing(str(_video_file(tmp_path)), standard_profile="made-up")
