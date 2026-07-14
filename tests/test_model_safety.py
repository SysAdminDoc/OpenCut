"""Model-weight supply-chain guard (opencut.core.model_safety).

CVE-2026-24747: torch.load(weights_only=True) can still reach RCE on a crafted
pickle, so pickle-format checkpoints are scanned with picklescan before load.
picklescan is an optional dependency, so the scan path is exercised via an
injected fake module.
"""

import sys
import types

import pytest

from opencut.core.model_safety import ModelSecurityError, safe_torch_load, scan_model_file


def _install_fake_picklescan(monkeypatch, infected):
    module = types.ModuleType("picklescan")
    scanner = types.ModuleType("picklescan.scanner")

    class _Result:
        infected_files = infected

    scanner.scan_file_path = lambda path: _Result()
    module.scanner = scanner
    monkeypatch.setitem(sys.modules, "picklescan", module)
    monkeypatch.setitem(sys.modules, "picklescan.scanner", scanner)


def test_safetensors_skips_scan(tmp_path, monkeypatch):
    # Even with a scanner that flags everything, safetensors is never scanned.
    _install_fake_picklescan(monkeypatch, infected=99)
    f = tmp_path / "model.safetensors"
    f.write_bytes(b"\x00")
    scan_model_file(str(f))  # must not raise


def test_pickle_clean_passes(tmp_path, monkeypatch):
    _install_fake_picklescan(monkeypatch, infected=0)
    f = tmp_path / "weights.pth"
    f.write_bytes(b"\x80\x04.")
    scan_model_file(str(f))  # must not raise


def test_pickle_infected_rejected(tmp_path, monkeypatch):
    _install_fake_picklescan(monkeypatch, infected=1)
    f = tmp_path / "evil.ckpt"
    f.write_bytes(b"\x80\x04.")
    with pytest.raises(ModelSecurityError):
        scan_model_file(str(f))


def test_missing_picklescan_degrades_gracefully(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "picklescan", None)  # force ImportError
    f = tmp_path / "weights.pt"
    f.write_bytes(b"\x80\x04.")
    scan_model_file(str(f))  # best-effort: warns, does not raise


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        scan_model_file(str(tmp_path / "nope.pth"))


def test_safe_torch_load_scans_and_enforces_weights_only(tmp_path, monkeypatch):
    _install_fake_picklescan(monkeypatch, infected=0)
    f = tmp_path / "sd.pth"
    f.write_bytes(b"\x80\x04.")

    calls = {}

    def fake_load(path, **kwargs):
        calls["path"] = path
        calls["kwargs"] = kwargs
        return {"ok": True}

    fake_torch = types.ModuleType("torch")
    fake_torch.load = fake_load
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    out = safe_torch_load(str(f), map_location="cpu")
    assert out == {"ok": True}
    assert calls["kwargs"]["weights_only"] is True
    assert calls["kwargs"]["map_location"] == "cpu"


def test_safe_torch_load_blocks_before_loading(tmp_path, monkeypatch):
    _install_fake_picklescan(monkeypatch, infected=2)
    f = tmp_path / "bad.pth"
    f.write_bytes(b"\x80\x04.")

    def fake_load(path, **kwargs):
        raise AssertionError("torch.load must not run on a flagged file")

    fake_torch = types.ModuleType("torch")
    fake_torch.load = fake_load
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(ModelSecurityError):
        safe_torch_load(str(f))
