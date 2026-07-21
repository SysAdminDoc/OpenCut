"""Download-time integrity verification (opencut.core.model_manager).

A freshly downloaded model must be checksum-verified (when the catalogue row
pins a ``sha256``) and pickle-scanned (for code-executing formats such as the
new Parakeet/Canary ``.nemo`` payloads) before it is marked available. A failure
fails closed and quarantines the file so it is never treated as installed.
"""

import hashlib
import sys
import threading
import types

import pytest

from opencut.core import model_manager as mm
from opencut.core.model_safety import is_pickle_format


def _install_fake_picklescan(monkeypatch, infected):
    module = types.ModuleType("picklescan")
    scanner = types.ModuleType("picklescan.scanner")

    class _Result:
        infected_files = infected

    scanner.scan_file_path = lambda path: _Result()
    module.scanner = scanner
    monkeypatch.setitem(sys.modules, "picklescan", module)
    monkeypatch.setitem(sys.modules, "picklescan.scanner", scanner)


# --- checksum -------------------------------------------------------------

def test_matching_sha256_passes(tmp_path):
    f = tmp_path / "model.safetensors"
    data = b"weights-bytes"
    f.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    mm._verify_download("m", str(f), digest)  # must not raise


def test_sha256_prefix_and_case_are_normalized(tmp_path):
    f = tmp_path / "model.safetensors"
    data = b"abc"
    f.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest().upper()
    mm._verify_download("m", str(f), "sha256:" + digest)  # must not raise


def test_mismatched_sha256_fails_closed(tmp_path):
    f = tmp_path / "model.safetensors"
    f.write_bytes(b"real")
    with pytest.raises(mm.ModelIntegrityError):
        mm._verify_download("m", str(f), "0" * 64)


def test_absent_sha256_skips_checksum(tmp_path, monkeypatch):
    _install_fake_picklescan(monkeypatch, infected=0)
    f = tmp_path / "model.safetensors"
    f.write_bytes(b"anything")
    mm._verify_download("m", str(f), None)  # must not raise


# --- pickle scan ----------------------------------------------------------

def test_nemo_is_pickle_format():
    assert is_pickle_format("/x/parakeet-tdt-0.6b-v3.nemo")
    assert is_pickle_format("/x/canary-1b-flash.nemo")
    assert not is_pickle_format("/x/model.safetensors")


def test_infected_pickle_payload_fails_closed(tmp_path, monkeypatch):
    _install_fake_picklescan(monkeypatch, infected=1)
    f = tmp_path / "parakeet.nemo"
    f.write_bytes(b"\x80\x04.")
    with pytest.raises(mm.ModelIntegrityError):
        mm._verify_download("parakeet", str(f), None)


def test_clean_pickle_payload_passes(tmp_path, monkeypatch):
    _install_fake_picklescan(monkeypatch, infected=0)
    f = tmp_path / "parakeet.nemo"
    f.write_bytes(b"\x80\x04.")
    mm._verify_download("parakeet", str(f), None)  # must not raise


def test_safetensors_is_not_pickle_scanned(tmp_path, monkeypatch):
    # A scanner that flags everything must never see a safetensors file.
    _install_fake_picklescan(monkeypatch, infected=99)
    f = tmp_path / "model.safetensors"
    f.write_bytes(b"\x00")
    mm._verify_download("m", str(f), None)  # must not raise


# --- worker quarantine ----------------------------------------------------

def _run_worker_with_bytes(monkeypatch, tmp_path, payload, model_name, expected_sha256):
    """Drive ``_download_worker`` against an in-memory HTTP response."""
    class _FakeResp:
        status = 200

        def __init__(self, data):
            self._data = data
            self._read = False
            self.headers = {"Content-Length": str(len(data))}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n):
            if self._read:
                return b""
            self._read = True
            return self._data

    monkeypatch.setattr(mm, "urlopen", lambda req, timeout=30: _FakeResp(payload))
    out = tmp_path / f"{model_name}.nemo"
    progress = mm.DownloadProgress(model_name=model_name, output_path=str(out))
    monkeypatch.setattr(mm, "DOWNLOADS_META_DIR", str(tmp_path / "meta"))
    mm._downloads[model_name] = progress
    mm._download_worker(
        model_name, "https://example.test/model.nemo", str(out), 0,
        threading.Event(), None, expected_sha256,
    )
    return progress, out


def test_worker_quarantines_on_checksum_mismatch(tmp_path, monkeypatch):
    payload = b"downloaded-bytes"
    progress, out = _run_worker_with_bytes(
        monkeypatch, tmp_path, payload, "badsum", expected_sha256="0" * 64
    )
    assert progress.status == "failed"
    assert not out.exists()  # tampered payload quarantined
    assert "mismatch" in (progress.error or "").lower()


def test_worker_completes_on_matching_checksum(tmp_path, monkeypatch):
    _install_fake_picklescan(monkeypatch, infected=0)
    payload = b"downloaded-bytes"
    digest = hashlib.sha256(payload).hexdigest()
    progress, out = _run_worker_with_bytes(
        monkeypatch, tmp_path, payload, "goodsum", expected_sha256=digest
    )
    assert progress.status == "completed"
    assert out.exists()


def test_worker_quarantines_infected_pickle(tmp_path, monkeypatch):
    _install_fake_picklescan(monkeypatch, infected=1)
    payload = b"\x80\x04."
    progress, out = _run_worker_with_bytes(
        monkeypatch, tmp_path, payload, "evil", expected_sha256=None
    )
    assert progress.status == "failed"
    assert not out.exists()
