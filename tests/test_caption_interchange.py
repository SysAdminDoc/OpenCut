"""Conformance corpus for canonical TTML-family caption interchange."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from opencut.core.caption_interchange import (
    IMSC_13_PROFILE,
    TTP_NS,
    CaptionCue,
    CaptionDocument,
    CaptionInterchangeError,
    CaptionRegion,
    CaptionStyle,
    default_regions,
    default_styles,
    export_caption_document,
    parse_ttml,
    serialize_caption_document,
    validate_ttml,
)


def _multilingual_document() -> CaptionDocument:
    styles = default_styles()
    styles["emphasis"] = CaptionStyle(
        id="emphasis",
        properties={"fontStyle": "italic", "color": "#FFFF00"},
    )
    regions = default_regions()
    regions["bottom"] = CaptionRegion(
        id="bottom",
        origin="10% 80%",
        extent="80% 15%",
        writing_mode="rltb",
        direction="rtl",
    )
    regions["vertical"] = CaptionRegion(
        id="vertical",
        origin="75% 10%",
        extent="20% 80%",
        display_align="before",
        writing_mode="tbrl",
        direction="ltr",
    )
    return CaptionDocument(
        language="ar",
        title="اختبار متعدد اللغات",
        frame_rate=29.97,
        styles=styles,
        regions=regions,
        cues=[
            CaptionCue(
                id="arabic-1",
                start=0.25,
                end=2.75,
                text="مرحباً بالعالم\nسطر ثانٍ",
                region="bottom",
                style_refs=("default", "emphasis"),
                language="ar",
                writing_mode="rltb",
                direction="rtl",
            ),
            CaptionCue(
                id="japanese-2",
                start=3.0,
                end=5.5,
                text="縦書きの字幕",
                region="vertical",
                style_refs=("default",),
                language="ja",
                writing_mode="tbrl",
                direction="ltr",
            ),
        ],
    )


def test_imsc13_multilingual_round_trip_preserves_semantics(tmp_path):
    output_path = tmp_path / "captions.imsc.xml"
    report = export_caption_document(
        _multilingual_document(),
        output_path,
        profile="imsc1.3",
    )

    assert report.valid
    assert report.profile == "imsc1.3"
    parsed = parse_ttml(output_path, expected_profile="imsc1.3")
    assert parsed.language == "ar"
    assert parsed.title == "اختبار متعدد اللغات"
    assert [cue.text for cue in parsed.cues] == ["مرحباً بالعالم\nسطر ثانٍ", "縦書きの字幕"]
    assert parsed.cues[0].style_refs == ("default", "emphasis")
    assert parsed.cues[0].direction == "rtl"
    assert parsed.cues[1].writing_mode == "tbrl"
    assert parsed.regions["bottom"].direction == "rtl"
    assert parsed.regions["vertical"].writing_mode == "tbrl"
    assert parsed.styles["emphasis"].properties["color"] == "#FFFF00"
    assert parsed.cues[0].start == pytest.approx(0.25)
    assert parsed.cues[1].end == pytest.approx(5.5)


def test_imsc13_profile_signal_uses_2026_recommendation_designator():
    payload = serialize_caption_document(_multilingual_document(), "imsc1_3")
    root = ET.fromstring(payload)
    assert root.get(f"{{{TTP_NS}}}contentProfiles") == IMSC_13_PROFILE
    assert root.get(f"{{{TTP_NS}}}profile") is None
    assert root.get(f"{{{TTP_NS}}}displayAspectRatio") == "16 9"


@pytest.mark.parametrize(
    ("profile", "expected"),
    [("ttml", "ttml"), ("legacy", "imsc1"), ("imsc1.3", "imsc1.3"), ("ebu_tt", "ebu_tt")],
)
def test_explicit_profiles_validate(profile, expected):
    payload = serialize_caption_document(_multilingual_document(), profile)
    report = validate_ttml(payload, expected_profile=profile)
    assert report.valid, report.to_dict()
    assert report.profile == expected


def _corpus_xml(*, root_attrs: str = "", cue_attrs: str = "", region_id: str = "bottom") -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<tt xmlns="http://www.w3.org/ns/ttml"
    xmlns:ttp="http://www.w3.org/ns/ttml#parameter"
    xmlns:tts="http://www.w3.org/ns/ttml#styling"
    xml:lang="en" ttp:timeBase="media" ttp:cellResolution="32 15"
    ttp:contentProfiles="{IMSC_13_PROFILE}" {root_attrs}>
  <head>
    <styling><style xml:id="default" tts:color="white"/></styling>
    <layout><region xml:id="{region_id}" tts:origin="10% 80%" tts:extent="80% 15%" tts:writingMode="lrtb" tts:direction="ltr"/></layout>
  </head>
  <body><div><p xml:id="cue1" begin="00:00:00.000" end="00:00:02.000" region="bottom" style="default" {cue_attrs}>Hello</p></div></body>
</tt>"""


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        (
            _corpus_xml().replace('ttp:cellResolution="32 15"', 'ttp:cellResolution="0 15"'),
            "imsc13.cell_resolution.invalid",
        ),
        (_corpus_xml(region_id="other"), "cue.region.unknown"),
        (
            _corpus_xml().replace('end="00:00:02.000"', 'end="00:00:00.000"'),
            "cue.timing.invalid",
        ),
        (_corpus_xml(cue_attrs='tts:writingMode="sideways"'), "writing_mode.invalid"),
        (
            _corpus_xml().replace(
                "</div></body>",
                '<p xml:id="cue1" begin="2s" end="3s" region="bottom" style="default">Again</p></div></body>',
            ),
            "cue.id.duplicate",
        ),
    ],
)
def test_invalid_conformance_corpus_is_rejected(payload, code):
    report = validate_ttml(payload, expected_profile="imsc1.3")
    assert not report.valid
    assert code in {issue.code for issue in report.errors}


def test_profile_mismatch_is_reported():
    payload = serialize_caption_document(_multilingual_document(), "ttml")
    report = validate_ttml(payload, expected_profile="imsc1.3")
    assert not report.valid
    assert any(issue.code == "profile.mismatch" for issue in report.errors)


def test_dtd_and_oversized_inputs_fail_closed():
    payload = b'<!DOCTYPE tt [<!ENTITY x "boom">]><tt>&x;</tt>'
    report = validate_ttml(payload)
    assert not report.valid
    assert report.errors[0].code == "xml.invalid"

    oversized = b"<" + b"x" * (5 * 1024 * 1024)
    report = validate_ttml(oversized)
    assert not report.valid
    assert "safety limit" in report.errors[0].message


def test_parse_rejects_nonconformant_input():
    with pytest.raises(CaptionInterchangeError, match="not conformant"):
        parse_ttml(_corpus_xml(region_id="missing"), expected_profile="imsc1.3")


def test_both_legacy_export_surfaces_use_canonical_imsc13(tmp_path):
    from opencut.core.broadcast_caption import CaptionSegment, export_imsc13
    from opencut.core.broadcast_cc import export_ttml

    delivery_path = tmp_path / "delivery.xml"
    result = export_ttml(
        [{"start": 0, "end": 1, "text": "مرحبا", "direction": "rtl"}],
        str(delivery_path),
        language="ar",
        profile="imsc1.3",
    )
    assert result["format"] == "ttml-imsc1.3"
    assert result["conformance"]["valid"]

    pro_path = tmp_path / "pro.xml"
    pro_result = export_imsc13(
        [CaptionSegment(index=1, start=0, end=1, text="Bonjour", language="fr")],
        str(pro_path),
        lang="fr",
    )
    assert pro_result.format == "imsc1_3"
    assert not pro_result.validation_errors
    assert validate_ttml(pro_path, expected_profile="imsc1.3").valid


def test_imsc13_preflight_reports_conformance():
    from opencut.core.caption_export_preflight import run_caption_export_preflight

    result = run_caption_export_preflight(
        segments=[{"start": 0, "end": 2, "text": "שלום", "direction": "rtl"}],
        host_version="26.3",
        target_profile="imsc1.3",
        language="he",
    )
    assert result.ready
    assert result.conformance_profile == "imsc1.3"
    assert result.conformance_valid
    assert any(diagnostic.check == "xml_conformance" for diagnostic in result.diagnostics)
