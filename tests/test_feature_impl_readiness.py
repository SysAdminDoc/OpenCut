"""Feature readiness must prove implementation, not just dependency presence.

An adapter whose entrypoint still raises NotImplementedError must never be
reported as `available`, even when its optional dependency is installed.
"""

from opencut import registry
from opencut.core import stub_scan


def test_stub_scanner_detects_known_terminal_stubs():
    stubs = stub_scan.terminal_stub_modules()
    for known in ("asr_parakeet", "asr_canary", "relight_iclight", "upscale_seedvr2"):
        assert stub_scan.is_stub_module(known), known
    # Real, implemented modules must not be flagged.
    for real in ("silence", "scene_detect", "archive_safety", "url_safety"):
        assert not stub_scan.is_stub_module(real), real
    assert not stub_scan.is_stub_module("")
    assert len(stubs) > 0


def test_stub_backed_features_resolve_to_stub():
    for fid in ("captions.nemo-asr", "video.relight.iclight", "video.upscale.seedvr2"):
        record = registry.get_feature(fid)
        assert record is not None, fid
        assert record.resolved_state() == registry.STATE_STUB, fid
        assert "not implemented" in record.state_reason().lower()


def test_release_gate_no_available_feature_is_a_terminal_stub():
    """Release gate: nothing that resolves to `available` may be backed by a
    module whose entrypoint raises NotImplementedError."""
    offenders = []
    for record in registry.list_features():
        if record.is_stub_implementation():
            if record.resolved_state() == registry.STATE_AVAILABLE:
                offenders.append(record.feature_id)
    assert offenders == [], f"stub-backed features reported available: {offenders}"


def test_impl_module_pointing_at_real_module_is_not_forced_stub():
    from opencut.registry import FeatureRecord

    record = FeatureRecord(
        feature_id="test.real",
        label="real",
        category="system",
        state=registry.STATE_AVAILABLE,
        impl_module="silence",  # a real implemented module
    )
    assert not record.is_stub_implementation()
    assert record.resolved_state() == registry.STATE_AVAILABLE


def test_as_dict_exposes_state_reason_for_panels_and_mcp():
    record = registry.get_feature("video.relight.iclight")
    payload = record.as_dict()
    assert payload["state"] == registry.STATE_STUB
    assert payload["state_reason"]
    assert "probe" not in payload
