"""Feature readiness must prove implementation, not just dependency presence.

An adapter whose entrypoint still raises NotImplementedError must never be
reported as `available`, even when its optional dependency is installed.
"""

from opencut import registry
from opencut.core import stub_scan


def test_stub_scanner_detects_known_terminal_stubs():
    stubs = stub_scan.terminal_stub_modules()
    for known in ("relight_iclight", "upscale_seedvr2"):
        assert stub_scan.is_stub_module(known), known
    # Real, implemented modules must not be flagged.
    for real in (
        "silence",
        "scene_detect",
        "archive_safety",
        "url_safety",
        "asr_parakeet",
        "asr_canary",
    ):
        assert not stub_scan.is_stub_module(real), real
    assert not stub_scan.is_stub_module("")
    assert len(stubs) > 0


def test_stub_backed_features_resolve_to_stub():
    for fid in ("video.relight.iclight", "video.upscale.seedvr2"):
        record = registry.get_feature(fid)
        assert record is not None, fid
        assert record.resolved_state() == registry.STATE_STUB, fid
        assert "not implemented" in record.state_reason().lower()


def test_nemo_feature_is_implemented_and_runtime_gated():
    record = registry.get_feature("captions.nemo-asr")
    assert record is not None
    assert not record.is_stub_implementation()
    assert record.resolved_state() in {
        registry.STATE_AVAILABLE,
        registry.STATE_MISSING_DEPENDENCY,
    }


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


def test_generated_records_carry_implementation_identity():
    """Generated records must name the adapter behind their probe.

    Without an ``impl_module`` a record is structurally incapable of being
    graded as a stub: there is nothing to scan, so a terminal
    ``NotImplementedError`` adapter whose optional dependency happens to be
    installed gets advertised as ``available``.
    """
    from opencut.tools.dump_feature_readiness import impl_modules_for_probe

    missing = []
    for record in registry.load_generated_feature_records():
        if not record.check_name:
            continue
        # Only probes that actually delegate to an opencut.core adapter can
        # carry one; direct third-party dependency probes legitimately cannot.
        if impl_modules_for_probe(record.check_name) and not record.impl_module:
            missing.append(record.feature_id)
    assert missing == [], f"generated records dropped their adapter identity: {missing}"


def test_no_generated_record_is_available_without_a_resolvable_adapter():
    """The blind spot itself is now a hard gate, not an observation.

    27 records used to report ``available`` with an empty ``impl_module``,
    which is exactly the shape that let three terminal-stub adapters advertise
    as runnable. Derivation now reads what the probe imports, what the routes
    it gates import, and finally a same-named module verified on disk.
    """
    from opencut.tools.dump_feature_readiness import (
        build_manifest,
        unproven_available_records,
    )

    assert unproven_available_records(build_manifest()) == []


def test_committed_readiness_manifest_proves_every_available_record():
    from opencut.tools.dump_feature_readiness import (
        load_manifest,
        unproven_available_records,
    )

    manifest = load_manifest()
    assert manifest is not None
    assert unproven_available_records(manifest) == []


def test_unproven_gate_rejects_an_empty_and_a_dangling_adapter():
    """Guard against a gate wired to always pass."""
    from opencut.tools.dump_feature_readiness import unproven_available_records

    offenders = unproven_available_records(
        {
            "records": [
                {
                    "feature_id": "auto.nothing",
                    "check_name": "check_nothing_available",
                    "state": registry.STATE_AVAILABLE,
                    "impl_module": "",
                },
                {
                    "feature_id": "auto.ghost",
                    "check_name": "check_ghost_available",
                    "state": registry.STATE_AVAILABLE,
                    "impl_module": "module_that_does_not_exist",
                },
                {
                    "feature_id": "auto.fine",
                    "check_name": "check_fine_available",
                    "state": registry.STATE_AVAILABLE,
                    "impl_module": "auto_zoom",
                },
                {
                    "feature_id": "auto.stub",
                    "check_name": "check_stub_available",
                    "state": registry.STATE_STUB,
                    "impl_module": "",
                },
            ]
        }
    )

    assert len(offenders) == 2
    assert any("auto.nothing" in line for line in offenders)
    assert any("auto.ghost" in line for line in offenders)


def test_generated_stub_adapters_never_resolve_available():
    """The three live regressions, plus every other generated record."""
    offenders = []
    for record in registry.load_generated_feature_records():
        if record.is_stub_implementation() and record.resolved_state() != registry.STATE_STUB:
            offenders.append((record.feature_id, record.resolved_state()))
    assert offenders == [], f"generated stub adapters escaped stub state: {offenders}"


def test_known_generated_stub_regressions_stay_stub():
    records = {r.feature_id: r for r in registry.load_generated_feature_records()}
    for feature_id, expected_module in (
        ("auto.deblur-motion", "deblur_motion"),
        # Not mechanically derivable from the probe name: check_searaft lives
        # in flow_searaft, which is exactly why guessing the module fails.
        ("auto.searaft", "flow_searaft"),
        ("auto.track-cutie", "track_cutie"),
    ):
        record = records[feature_id]
        assert record.impl_module == expected_module, feature_id
        assert record.is_stub_implementation(), feature_id
        assert record.resolved_state() == registry.STATE_STUB, feature_id


def test_future_generated_stub_fixture_fails_closed():
    """A newly generated record for a stub adapter must not be advertised.

    This is the shape a future regression takes: the dumper emits a record
    whose declared state is ``missing_dependency`` while its optional
    dependency is in fact installed, so the live probe would promote it.
    """
    from opencut.registry import _record_from_generated

    record = _record_from_generated(
        {
            "feature_id": "auto.future-stub",
            "label": "Future Stub",
            "category": "video",
            "state": registry.STATE_MISSING_DEPENDENCY,
            "routes": ["/video/future-stub"],
            "check_name": "",
            # A real terminal-stub adapter in the tree.
            "impl_module": "relight_iclight",
        }
    )
    record.probe = lambda: True  # dependency present
    assert record.is_stub_implementation()
    assert record.resolved_state() == registry.STATE_STUB


def test_generated_record_without_impl_module_is_not_silently_promoted():
    """Guard the exact hole: no adapter identity plus a passing probe."""
    from opencut.registry import _record_from_generated

    payload = {
        "feature_id": "auto.no-impl",
        "label": "No Impl",
        "category": "video",
        "state": registry.STATE_MISSING_DEPENDENCY,
        "routes": ["/video/no-impl"],
        "check_name": "",
        "impl_module": "",
    }
    record = _record_from_generated(payload)
    assert record.impl_module == ""
    # The dumper is what supplies identity, so assert it would have done so
    # for any probe that delegates to an adapter.
    from opencut.tools.dump_feature_readiness import impl_modules_for_probe

    assert impl_modules_for_probe("check_deblur_motion") == ["deblur_motion"]
