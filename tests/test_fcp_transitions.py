"""Tests for FCP XML transition writer + validator (F104)."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from opencut.export import fcp_transitions as ft


def _clips(outbound_end: int = 100, inbound_start: int = 100, inbound_dur: int = 100, *,
           outbound_source_end: int = 200, inbound_source_start: int = 0):
    outbound = ft.FcpClip(
        name="A",
        in_frame=0,
        out_frame=outbound_end,
        source_start=0,
        source_end=outbound_source_end,
    )
    inbound = ft.FcpClip(
        name="B",
        in_frame=inbound_start,
        out_frame=inbound_start + inbound_dur,
        source_start=inbound_source_start,
        source_end=inbound_source_start + 1000,
    )
    return outbound, inbound


def test_validate_centre_aligned_transition_returns_placement():
    outbound, inbound = _clips()
    req = ft.TransitionRequest(kind="cross_dissolve", duration_frames=20, cut_frame=100)

    placement = ft.validate_transition_request(req, outbound, inbound)

    assert placement.kind == "cross_dissolve"
    assert placement.pre_trim == 10
    assert placement.post_trim == 10
    assert placement.start_frame == 90
    assert placement.end_frame == 110
    assert placement.media_type == "video"


def test_validate_centre_alignment_biases_toward_outbound_on_odd():
    """duration=5 → 3 pre, 2 post (centre bias toward outbound)."""
    outbound, inbound = _clips()
    req = ft.TransitionRequest(kind="cross_dissolve", duration_frames=5, cut_frame=100)
    placement = ft.validate_transition_request(req, outbound, inbound)
    assert placement.pre_trim == 3
    assert placement.post_trim == 2


def test_validate_start_alignment_puts_everything_after_cut():
    outbound, inbound = _clips()
    req = ft.TransitionRequest(kind="cross_dissolve", duration_frames=10, cut_frame=100, align="start")
    placement = ft.validate_transition_request(req, outbound, inbound)
    assert placement.pre_trim == 0
    assert placement.post_trim == 10
    assert placement.start_frame == 100
    assert placement.end_frame == 110


def test_validate_rejects_transition_past_outbound_tail_handle():
    """Outbound clip's source tail must cover the post half of the transition."""
    # outbound.source_end - out_frame = 102 - 100 = 2 frames of tail handle;
    # a centre-aligned 20-frame transition needs 10.
    outbound = ft.FcpClip(name="A", in_frame=0, out_frame=100, source_start=0, source_end=102)
    inbound = ft.FcpClip(name="B", in_frame=100, out_frame=200, source_start=0, source_end=300)
    req = ft.TransitionRequest(kind="cross_dissolve", duration_frames=20, cut_frame=100)

    with pytest.raises(ft.TransitionConstraintError) as exc:
        ft.validate_transition_request(req, outbound, inbound)
    assert "outbound" in str(exc.value).lower() and "handle" in str(exc.value).lower()


def test_validate_rejects_transition_with_insufficient_inbound_material():
    # Inbound clip is only 4 frames long but transition needs 10 frames
    # of timeline material after the cut. Use a generous head handle so
    # the failure pins to the timeline check, not the head-handle check.
    outbound, inbound = _clips(outbound_end=100, inbound_start=100, inbound_dur=4,
                                inbound_source_start=0)
    req = ft.TransitionRequest(kind="cross_dissolve", duration_frames=20, cut_frame=100)

    with pytest.raises(ft.TransitionConstraintError) as exc:
        ft.validate_transition_request(req, outbound, inbound)
    assert "inbound" in str(exc.value).lower()


def test_validate_rejects_unsupported_kind():
    outbound, inbound = _clips()
    req = ft.TransitionRequest(kind="not_a_transition", duration_frames=10, cut_frame=100)
    with pytest.raises(ft.TransitionConstraintError) as exc:
        ft.validate_transition_request(req, outbound, inbound)
    assert "unsupported" in str(exc.value)


def test_validate_rejects_zero_duration():
    outbound, inbound = _clips()
    req = ft.TransitionRequest(kind="cross_dissolve", duration_frames=0, cut_frame=100)
    with pytest.raises(ft.TransitionConstraintError):
        ft.validate_transition_request(req, outbound, inbound)


def test_build_transitionitem_produces_well_formed_xml():
    outbound, inbound = _clips()
    placement = ft.validate_transition_request(
        ft.TransitionRequest(kind="cross_dissolve", duration_frames=12, cut_frame=100),
        outbound,
        inbound,
    )

    item = ft.build_transitionitem(placement)
    assert item.tag == "transitionitem"
    text_of = {child.tag: child.text for child in item}
    assert text_of["start"] == str(placement.start_frame)
    assert text_of["end"] == str(placement.end_frame)
    assert text_of["alignment"] == "center"

    effect = item.find("effect")
    assert effect is not None
    sub = {child.tag: child.text for child in effect}
    assert sub["effectid"] == "Cross Dissolve"
    assert sub["mediatype"] == "video"


def test_build_transitionitem_supports_audio_kind():
    outbound, inbound = _clips()
    placement = ft.validate_transition_request(
        ft.TransitionRequest(kind="constant_power", duration_frames=20, cut_frame=100),
        outbound,
        inbound,
    )
    item = ft.build_transitionitem(placement)
    effect = item.find("effect")
    assert effect is not None
    assert effect.find("mediatype").text == "audio"


def test_trim_for_transition_adjusts_cut_frames():
    outbound, inbound = _clips(outbound_end=100, inbound_start=100, inbound_dur=100)
    placement = ft.validate_transition_request(
        ft.TransitionRequest(kind="cross_dissolve", duration_frames=20, cut_frame=100),
        outbound,
        inbound,
    )

    new_outbound, new_inbound = ft.trim_for_transition(placement, outbound, inbound)
    assert new_outbound.out_frame == 110
    assert new_inbound.in_frame == 90
    # Source ranges left intact.
    assert new_outbound.source_start == 0
    assert new_inbound.source_end == 1000


def test_validate_rejects_when_outbound_source_handle_is_short():
    """outbound.source_end - outbound.out_frame must >= pre."""
    outbound, inbound = _clips(outbound_source_end=102)  # only 2 frames of tail handle
    req = ft.TransitionRequest(kind="cross_dissolve", duration_frames=20, cut_frame=100)
    with pytest.raises(ft.TransitionConstraintError) as exc:
        ft.validate_transition_request(req, outbound, inbound)
    assert "handle" in str(exc.value).lower()


def test_validate_rejects_when_inbound_source_head_handle_is_short():
    outbound, inbound = _clips(inbound_source_start=95)  # only 5 frames of head handle
    req = ft.TransitionRequest(kind="cross_dissolve", duration_frames=20, cut_frame=100)
    with pytest.raises(ft.TransitionConstraintError):
        ft.validate_transition_request(req, outbound, inbound)
