"""Tests for the marker metadata round-trip schema (F103)."""

from __future__ import annotations

import pytest

from opencut.core import marker_metadata as mm


# ----- Colour mapping ------------------------------------------------------


def test_normalise_color_handles_premiere_palette():
    assert mm.normalise_color("Rose", host="premiere") == "rose"
    assert mm.normalise_color("Magenta", host="premiere") == "purple"
    assert mm.normalise_color("blue", host="premiere") == "blue"


def test_normalise_color_handles_davinci_extras():
    assert mm.normalise_color("Mint", host="davinci") == "green"
    assert mm.normalise_color("Cream", host="davinci") == "white"
    assert mm.normalise_color("Sand", host="davinci") == "orange"
    assert mm.normalise_color("Lavender", host="davinci") == "purple"


def test_normalise_color_handles_avid_palette():
    assert mm.normalise_color("Magenta", host="avid") == "purple"


def test_normalise_color_accepts_explicit_host_prefix():
    assert mm.normalise_color("davinci:Pink") == "rose"
    assert mm.normalise_color("otio:MAGENTA") == "purple"


def test_normalise_color_falls_back_to_green_for_unknown():
    assert mm.normalise_color("not a real colour") == "green"
    assert mm.normalise_color("") == "green"
    assert mm.normalise_color(None) == "green"


def test_denormalise_color_uses_host_specific_names():
    assert mm.denormalise_color("rose", "premiere") == "Rose"
    assert mm.denormalise_color("rose", "davinci") == "Pink"
    assert mm.denormalise_color("rose", "avid") == "Magenta"
    assert mm.denormalise_color("orange", "davinci") == "Sand"
    assert mm.denormalise_color("white", "davinci") == "Cream"


def test_denormalise_color_rejects_unknown_host():
    with pytest.raises(ValueError):
        mm.denormalise_color("rose", "not-an-nle")


def test_supported_hosts_returns_premiere_davinci_avid_otio():
    assert set(mm.supported_hosts()) == {"premiere", "davinci", "avid", "otio"}


# ----- Round-trip ----------------------------------------------------------


def _premiere_payload():
    return [
        {"name": "Hook", "start_seconds": 1.5, "duration_seconds": 0.0, "color": "Rose", "comment": "intro"},
        {"name": "Cut", "start_seconds": 5.0, "duration_seconds": 0.5, "color": "Yellow", "comment": ""},
    ]


def test_round_trip_premiere_to_davinci_and_back():
    """A Premiere marker exported to DaVinci and re-imported must keep meaning."""
    canonical = [
        mm.MarkerMetadata(
            name=m["name"],
            start_seconds=m["start_seconds"],
            duration_seconds=m["duration_seconds"],
            color=mm.normalise_color(m["color"], host="premiere"),
            source="premiere",
            comment=m["comment"],
        )
        for m in _premiere_payload()
    ]
    davinci_payload = [m.for_host("davinci") for m in canonical]

    # Bring it back as if importing from DaVinci.
    reimported = []
    for m in davinci_payload:
        reimported.append(
            mm.MarkerMetadata(
                name=m["name"],
                start_seconds=m["start_seconds"],
                duration_seconds=m["duration_seconds"],
                color=mm.normalise_color(m["color"], host="davinci"),
                source="davinci",
                comment=m["comment"],
            ).as_dict()
        )

    diffs = mm.diff_marker_payloads(
        [m.as_dict() for m in canonical],
        reimported,
    )
    # Source field is expected to differ (premiere vs davinci) so
    # diff_marker_payloads intentionally ignores it.
    assert not diffs, diffs


def test_round_trip_premiere_to_avid_collapses_rose_to_magenta():
    canonical = mm.MarkerMetadata(name="Hook", color="Rose", source="premiere")
    avid_payload = canonical.for_host("avid")
    assert avid_payload["color"] == "Magenta"

    # Re-import — Avid magenta maps back to canonical purple, which is
    # acceptable: Avid doesn't have rose, so we expect colour collapse.
    reimported = mm.MarkerMetadata(
        name=avid_payload["name"],
        start_seconds=avid_payload["start_seconds"],
        color=mm.normalise_color(avid_payload["color"], host="avid"),
        source="avid",
    )
    assert reimported.color == "purple"


def test_diff_marker_payloads_detects_count_mismatch():
    a = [mm.MarkerMetadata().as_dict()]
    b = [mm.MarkerMetadata().as_dict(), mm.MarkerMetadata().as_dict()]
    diffs = mm.diff_marker_payloads(a, b)
    assert diffs and "count differs" in diffs[0]


def test_marker_for_host_renames_source():
    m = mm.MarkerMetadata(name="x", color="green", source="csv")
    rendered = m.for_host("premiere")
    assert rendered["source"] == "premiere"
    assert rendered["color"] == "Green"


# ----- Integration with marker_import -------------------------------------


def test_marker_import_csv_colors_normalise_through_metadata():
    from opencut.core.marker_import import parse_csv

    csv_text = (
        "timecode,name,color\n"
        "00:00:01:00,M1,Rose\n"
        "00:00:02:00,M2,Magenta\n"
        "00:00:03:00,M3,Mint\n"
    )
    result = parse_csv(csv_text, fps=30)

    # CSV parser drops unknown synonyms onto green by default — make sure
    # canonical normalisation upgrades them when we go through the schema.
    canonicals = [
        mm.normalise_color(m.color, host="premiere") for m in result.markers
    ]
    assert canonicals[0] == "rose"
    assert canonicals[1] == "purple"  # magenta → purple via premiere alias
    # Mint isn't a Premiere palette name — comes in as "green" from the CSV
    # but normalising via DaVinci would route it differently. The default
    # premiere lookup keeps it green which is the expected fall-back.
    assert canonicals[2] == "green"
