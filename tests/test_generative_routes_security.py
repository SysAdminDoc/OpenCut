import os
import tempfile
from pathlib import Path

import pytest

from tests.conftest import csrf_headers


def _write_scene_file(path: Path) -> Path:
    path.write_text("ply\nformat ascii 1.0\nend_header\n", encoding="utf-8")
    return path


def _skip_if_repo_is_inside_an_approved_root() -> None:
    """A file in the repo only proves the denial path when the repo is outside.

    A clone into the system temp directory sits inside an approved root, so
    `pyproject.toml` is legitimately servable there and the assertion below
    would report a bug that does not exist.
    """
    repo_root = os.path.realpath(Path(__file__).resolve().parents[1])
    for approved in (
        os.path.realpath(tempfile.gettempdir()),
        os.path.realpath(os.path.join(os.path.expanduser("~"), ".opencut")),
    ):
        if os.path.commonpath([repo_root, approved]) == approved:
            pytest.skip(f"checkout lives under an approved root ({approved})")


def test_gaussian_splat_preview_frame_serves_temp_renderer_output(client, csrf_token, tmp_path, monkeypatch):
    scene_path = _write_scene_file(tmp_path / "scene.ply")
    frame_path = tmp_path / "preview.png"
    frame_bytes = b"\x89PNG\r\n\x1a\nopencut-preview"
    frame_path.write_bytes(frame_bytes)

    monkeypatch.setattr(
        "opencut.core.gaussian_splat.render_splat_frame",
        lambda **_kwargs: str(frame_path),
    )

    resp = client.post(
        "/gaussian-splat/preview-frame",
        json={"ply_path": str(scene_path), "width": 64, "height": 64},
        headers=csrf_headers(csrf_token),
    )

    assert resp.status_code == 200
    assert resp.mimetype == "image/png"
    assert resp.data == frame_bytes


def test_gaussian_splat_preview_frame_rejects_renderer_output_outside_allowed_roots(
    client,
    csrf_token,
    tmp_path,
    monkeypatch,
):
    _skip_if_repo_is_inside_an_approved_root()
    scene_path = _write_scene_file(tmp_path / "scene.ply")
    unconfined_path = Path("pyproject.toml").resolve()

    monkeypatch.setattr(
        "opencut.core.gaussian_splat.render_splat_frame",
        lambda **_kwargs: str(unconfined_path),
    )

    resp = client.post(
        "/gaussian-splat/preview-frame",
        json={"ply_path": str(scene_path), "width": 64, "height": 64},
        headers=csrf_headers(csrf_token),
    )

    assert resp.status_code == 403
    assert resp.get_json() == {"error": "Access denied"}
