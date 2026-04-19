"""Tests for the Timecode Watermark example plugin."""

import importlib.util
import os
from unittest.mock import patch

import pytest
from flask import Flask


@pytest.fixture
def plugin_module():
    plugin_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "opencut",
        "data",
        "example_plugins",
        "timecode-watermark",
        "routes.py",
    )
    spec = importlib.util.spec_from_file_location("timecode_watermark_routes", plugin_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def client(plugin_module):
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(plugin_module.plugin_bp, url_prefix="/plugins/timecode-watermark")
    with app.test_client() as c:
        yield c


def test_apply_rejects_non_object_body(client):
    resp = client.post("/plugins/timecode-watermark/apply", json=["bad"])

    assert resp.status_code == 400
    assert "path" in resp.get_json()["error"].lower()


def test_apply_rejects_invalid_filter_color(client, tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")

    with patch("opencut.helpers.run_ffmpeg") as run_ffmpeg:
        resp = client.post(
            "/plugins/timecode-watermark/apply",
            json={"filepath": str(video), "color": "white:box=1"},
        )

    assert resp.status_code == 400
    assert "color" in resp.get_json()["error"]
    run_ffmpeg.assert_not_called()


def test_apply_validates_and_clamps_ffmpeg_options(client, tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")

    with patch("opencut.helpers.run_ffmpeg") as run_ffmpeg:
        resp = client.post(
            "/plugins/timecode-watermark/apply",
            json={
                "filepath": str(video),
                "position": "bottom-right",
                "font_size": 999,
                "color": "#ffffff",
                "start_timecode": "00:00:01:12",
            },
        )

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert data["output"].endswith("clip_tc.mp4")
    cmd = run_ffmpeg.call_args.args[0]
    filter_arg = cmd[cmd.index("-vf") + 1]
    assert "fontsize=200" in filter_arg
    assert "x=w-tw-10:y=h-th-10" in filter_arg
