from opencut.core import loudness_standards as standards


def test_f237_current_loudness_standards_are_source_backed():
    metadata = standards.get_loudness_standards()

    itu = metadata["itu_r_bs1770"]
    assert itu["current_version"] == "BS.1770-5"
    assert itu["status"] == "in_force"
    assert itu["previous_version"] == "BS.1770-4"
    assert itu["previous_status"] == "superseded"
    assert itu["source_url"].startswith("https://www.itu.int/")

    ebu = metadata["ebu_r128"]
    assert ebu["current_version"] == "5.0"
    assert ebu["target_lufs"] == -23.0
    assert ebu["max_true_peak_dbtp"] == -1.0
    assert ebu["measurement_basis"] == "itu_r_bs1770"


def test_f237_loudness_presets_keep_compatibility_fields_and_corrected_targets():
    presets = standards.LOUDNESS_PRESETS

    assert presets["broadcast"]["i"] == -23.0
    assert presets["broadcast"]["tp"] == -1.0
    assert presets["broadcast"]["measurement_standard"] == "ebu_r128"
    assert presets["streaming"]["i"] == -16.0
    assert presets["streaming"]["tp"] == -1.0
    assert presets["podcast"]["i"] == -16.0
    assert presets["podcast"]["tp"] == -1.0
    assert "not source-backed" in presets["podcast"]["notes"]
    assert presets["spotify"]["i"] == -14.0
    assert presets["spotify"]["tp"] == -1.0

    for preset in presets.values():
        assert {"i", "tp", "lra", "source_url", "measurement_standard"} <= set(preset)


def test_f237_audio_suite_exports_the_canonical_preset_table():
    from opencut.core.audio_suite import LOUDNESS_PRESETS

    assert LOUDNESS_PRESETS is standards.LOUDNESS_PRESETS
    assert LOUDNESS_PRESETS["broadcast"]["tp"] == -1.0


def test_f237_platform_targets_preserve_existing_broadcast_semantics():
    from opencut.core.audio_analysis import PLATFORM_TARGETS

    assert PLATFORM_TARGETS["broadcast"] == -24.0
    assert PLATFORM_TARGETS["ebu_broadcast"] == -23.0
    assert PLATFORM_TARGETS["online_video"] == -16.0


def test_f237_loudness_preset_route_exposes_sources(client):
    response = client.get("/audio/loudness-presets")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["standards"]["itu_r_bs1770"]["current_version"] == "BS.1770-5"
    assert payload["standards"]["ebu_r128"]["current_version"] == "5.0"
    assert "BS.1770-4 is superseded" in payload["correction_note"]

    by_name = {preset["name"]: preset for preset in payload["presets"]}
    assert by_name["broadcast"]["target_lufs"] == -23.0
    assert by_name["broadcast"]["target_tp"] == -1.0
    assert by_name["streaming"]["target_lufs"] == -16.0
    assert by_name["spotify"]["target_lufs"] == -14.0
