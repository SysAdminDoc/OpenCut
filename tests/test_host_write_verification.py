"""F319 — Premiere host writes must carry independent read-back evidence."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CEP_HOST = ROOT / "extension" / "com.opencut.panel" / "host" / "index.jsx"
CEP_CLIENT = ROOT / "extension" / "com.opencut.panel" / "client" / "main.js"
CEP_VERIFICATION = (
    ROOT
    / "extension"
    / "com.opencut.panel"
    / "client"
    / "host-write-verification.js"
)
UXP_MAIN = ROOT / "extension" / "com.opencut.uxp" / "main.js"
UXP_VERIFICATION = ROOT / "extension" / "com.opencut.uxp" / "uxp-host-write-verification.js"
UXP_SETTINGS = ROOT / "extension" / "com.opencut.uxp" / "uxp-settings-controller.js"
BOLT_API = (
    ROOT
    / "extension"
    / "com.opencut.uxp"
    / "bolt-webview"
    / "src"
    / "api"
    / "premierepro.ts"
)
I18N_LINT = ROOT / "scripts" / "i18n_lint.py"


def _function_source(source: str, name: str) -> str:
    start = source.index(f"function {name}(")
    next_function = source.find("\nfunction ", start + 1)
    return source[start : next_function if next_function >= 0 else len(source)]


def test_cep_mutations_return_the_f319_verification_contract():
    source = CEP_HOST.read_text(encoding="utf-8")
    mutating_actions = (
        "ocAddSequenceMarkers",
        "ocApplySequenceCuts",
        "ocApplyClipKeyframes",
        "ocBatchRenameProjectItems",
        "ocCreateSmartBins",
        "ocAddNativeCaptionTrack",
        "ocRemoveSequenceMarkers",
        "ocUnrenameItems",
        "ocRemoveImportedSequence",
        "ocSetSequencePlayhead",
        "ocRemoveImportedItem",
    )
    for action in mutating_actions:
        body = _function_source(source, action)
        assert "_ocHostWriteVerification(" in body, action
        assert "_ocAttachHostWriteVerification(" in body, action


def test_legacy_cep_imports_are_verified_in_the_host_bridge():
    source = CEP_HOST.read_text(encoding="utf-8")
    for action in (
        "importXMLToProject",
        "importAndOpenXml",
        "applyEditsToTimeline",
        "importCaptions",
        "importFileToProject",
        "importFilesToProject",
        "importCaptionOverlay",
        "autoImportResult",
        "ocExportSequenceRange",
    ):
        body = _function_source(source, action)
        assert "_ocHostWriteVerification(" in body or "_ocAttachProjectImportVerification(" in body or "_ocAttachInterchangeImportVerification(" in body, action

    overlay_alias = _function_source(source, "importOverlayToProject")
    assert "return importCaptionOverlay(overlayPath)" in overlay_alias


def test_uxp_mutations_use_independent_read_back_helpers():
    source = UXP_MAIN.read_text(encoding="utf-8")
    verifier = UXP_VERIFICATION.read_text(encoding="utf-8")
    for action in (
        "addMarkers",
        "removeSequenceMarkers",
        "applyCuts",
        "importFiles",
        "importTimelineInterchange",
        "batchRenameProjectItems",
        "createSmartBins",
        "removeImportedProjectItem",
        "setSequencePlayhead",
    ):
        body = _function_source(source, action)
        assert "_attachHostWriteVerification(" in body, action

    # Disabled state joined the fingerprint in F339: a non-destructive cut
    # leaves every clip in place, so boundaries alone read back as no change.
    assert "video/audio track-item boundary and disabled-state fingerprint diff per cut" in source
    assert "HOST_WRITE_NOT_APPLIED" in verifier
    assert "ensureHostWriteVerification" in source


def test_bolt_write_paths_fail_closed_on_no_change():
    source = BOLT_API.read_text(encoding="utf-8")
    for action in ("addTimelineMarkers", "applyTimelineCuts", "importFiles"):
        body = _function_source(source, action)
        assert "verificationResult(" in body, action
    assert "HOST_WRITE_NOT_APPLIED" in source
    assert "trackSnapshot(sequence)" in source


def test_panels_surface_and_persist_unverified_host_results():
    cep = CEP_CLIENT.read_text(encoding="utf-8")
    cep_verifier = CEP_VERIFICATION.read_text(encoding="utf-8")
    uxp = UXP_MAIN.read_text(encoding="utf-8")
    uxp_settings = UXP_SETTINGS.read_text(encoding="utf-8")
    i18n_lint = I18N_LINT.read_text(encoding="utf-8")
    for source in (cep_verifier, uxp):
        assert "host_write_verification" in source
        assert "verification_status" in source
    assert "host_diagnostics" in cep
    assert "getHostDiagnostics" in uxp
    assert "host_diagnostics" in uxp_settings
    assert "Premiere reported success but independent read-back" in cep_verifier
    assert 'statusKey: "journal.host_write_no_change"' in cep_verifier
    assert 'hintKey: "journal.host_write_unverified"' in cep_verifier
    assert "HOST_WRITE_VERIFICATION_JS" in i18n_lint
