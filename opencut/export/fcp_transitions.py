"""FCP 7 XML transition writer + validator (F104).

The existing :mod:`opencut.export.premiere` emits a clip-only timeline.
Real edits routinely include cross-dissolves and dip-to-color
transitions, and the most common reason a Premiere import fails is a
transition whose duration exceeds the available source handles. This
module is the focused helper:

* :func:`validate_transition_request` — runs the math and raises
  :class:`TransitionConstraintError` with a structured message when a
  transition is impossible.
* :func:`build_transitionitem` — emits the ``<transitionitem>`` element
  in FCP 7 XML so callers don't have to memorise the schema.
* :func:`trim_for_transition` — adjusts the inbound/outbound clip
  positions when a transition is added so the new effect actually
  overlaps the cut as Premiere expects.

The module is stdlib-only and works without OpenTimelineIO, so it can
be exercised inside the release-gate test set.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Iterable, List, Optional


# FCP 7 XML transition effect ids that Premiere recognises out of the
# box. Keep this list short — the goal is round-trippable fidelity, not
# every conceivable plug-in transition.
SUPPORTED_TRANSITIONS = {
    "cross_dissolve": {
        "effect_id": "Cross Dissolve",
        "effect_name": "Cross Dissolve",
        "effect_category": "Dissolve",
        "effect_type": "transition",
        "media_type": "video",
    },
    "dip_to_black": {
        "effect_id": "Dip to Color Dissolve",
        "effect_name": "Dip to Black",
        "effect_category": "Dissolve",
        "effect_type": "transition",
        "media_type": "video",
    },
    "constant_power": {
        "effect_id": "Audio Cross Fade",
        "effect_name": "Constant Power",
        "effect_category": "Audio Transitions",
        "effect_type": "transition",
        "media_type": "audio",
    },
}


class TransitionConstraintError(ValueError):
    """Raised when a transition cannot be inserted at the requested cut."""


@dataclass
class FcpClip:
    """Minimal projection of an FCP-style clip for transition math."""

    name: str
    in_frame: int
    out_frame: int
    source_start: int = 0        # inbound clip's source-start (head handle anchor)
    source_end: int = 0          # inbound clip's source-end (tail handle anchor)

    def duration(self) -> int:
        return max(0, self.out_frame - self.in_frame)


@dataclass
class TransitionRequest:
    kind: str                    # key into SUPPORTED_TRANSITIONS
    duration_frames: int
    cut_frame: int               # timeline frame at which the transition is centred
    align: str = "centre"        # "centre" | "start" | "end"
    media_type: Optional[str] = None  # override SUPPORTED_TRANSITIONS entry


@dataclass
class TransitionPlacement:
    """Outcome of validating + planning a transition insertion."""

    kind: str
    media_type: str
    start_frame: int
    end_frame: int
    pre_trim: int = 0            # frames the outbound clip must shorten by
    post_trim: int = 0           # frames the inbound clip must shorten by
    warnings: List[str] = field(default_factory=list)


def _entry(kind: str) -> dict:
    if kind not in SUPPORTED_TRANSITIONS:
        raise TransitionConstraintError(
            f"unsupported transition kind {kind!r}; choose from {sorted(SUPPORTED_TRANSITIONS)}"
        )
    return SUPPORTED_TRANSITIONS[kind]


def _half(duration: int) -> tuple:
    """Split a duration into the pre/post halves around a cut.

    The first half is the longer one when duration is odd, mirroring
    Premiere's behaviour of biasing centres toward the outbound clip.
    """
    pre = duration // 2 + (duration % 2)
    post = duration // 2
    return pre, post


def validate_transition_request(
    request: TransitionRequest,
    outbound: FcpClip,
    inbound: FcpClip,
) -> TransitionPlacement:
    """Run the math + raise if Premiere wouldn't accept the transition."""
    entry = _entry(request.kind)
    media_type = request.media_type or entry["media_type"]

    if request.duration_frames <= 0:
        raise TransitionConstraintError("duration_frames must be > 0")

    pre, post = _half(request.duration_frames)
    if request.align == "start":
        pre, post = 0, request.duration_frames
    elif request.align == "end":
        pre, post = request.duration_frames, 0
    elif request.align != "centre":
        raise TransitionConstraintError(
            f"unknown align {request.align!r}; expected centre / start / end"
        )

    # Outbound clip provides material for the *post-cut* portion of the
    # transition. The timeline-side cut sits at ``cut_frame`` and the
    # outbound clip's timeline tail is at ``out_frame``; the post half
    # of a centred transition draws from the source material that
    # extends beyond ``out_frame`` (the tail handle).
    outbound_tail_handle = outbound.source_end - outbound.out_frame if outbound.source_end else None
    if outbound_tail_handle is not None and outbound_tail_handle < post:
        raise TransitionConstraintError(
            f"outbound clip handle is {outbound_tail_handle} frames; "
            f"the post-roll needs {post}"
        )

    # The pre-cut half draws from material the outbound clip already
    # has on its timeline. If the cut is *before* the clip starts, the
    # transition cannot pull frames from thin air.
    outbound_timeline_pre = outbound.out_frame - max(outbound.in_frame, request.cut_frame - pre)
    if outbound_timeline_pre < pre:
        raise TransitionConstraintError(
            f"outbound clip has only {outbound_timeline_pre} frames before the cut; "
            f"the pre-roll needs {pre}"
        )

    # Inbound clip mirrors the above: the pre half draws from the inbound
    # source handle that lives before ``in_frame``; the post half lives
    # on the inbound timeline itself.
    inbound_head_handle = inbound.in_frame - inbound.source_start if inbound.source_start is not None else None
    if inbound_head_handle is not None and inbound_head_handle < pre:
        raise TransitionConstraintError(
            f"inbound clip has only {inbound_head_handle} frames of head handle; "
            f"the pre-roll needs {pre}"
        )

    inbound_timeline_post = min(inbound.out_frame, request.cut_frame + post) - inbound.in_frame
    if inbound_timeline_post < post:
        raise TransitionConstraintError(
            f"inbound clip has only {inbound_timeline_post} frames after the cut; "
            f"the post-roll needs {post}"
        )

    placement = TransitionPlacement(
        kind=request.kind,
        media_type=media_type,
        start_frame=request.cut_frame - pre,
        end_frame=request.cut_frame + post,
        pre_trim=pre,
        post_trim=post,
    )
    if media_type not in {"video", "audio"}:
        placement.warnings.append(
            f"unusual media_type {media_type!r}; Premiere may treat this as a video transition"
        )
    return placement


def build_transitionitem(
    placement: TransitionPlacement,
    *,
    transition_name: Optional[str] = None,
) -> ET.Element:
    """Emit a ``<transitionitem>`` XML element ready to append to a track."""
    entry = SUPPORTED_TRANSITIONS[placement.kind]
    name = transition_name or entry["effect_name"]

    item = ET.Element("transitionitem")
    ET.SubElement(item, "start").text = str(placement.start_frame)
    ET.SubElement(item, "end").text = str(placement.end_frame)
    ET.SubElement(item, "alignment").text = "center"

    effect = ET.SubElement(item, "effect")
    ET.SubElement(effect, "name").text = name
    ET.SubElement(effect, "effectid").text = entry["effect_id"]
    ET.SubElement(effect, "effectcategory").text = entry["effect_category"]
    ET.SubElement(effect, "effecttype").text = entry["effect_type"]
    ET.SubElement(effect, "mediatype").text = placement.media_type

    return item


def trim_for_transition(
    placement: TransitionPlacement,
    outbound: FcpClip,
    inbound: FcpClip,
) -> tuple:
    """Return new ``(outbound, inbound)`` clips with their cut points adjusted.

    Premiere's importer expects the inbound clip's ``start`` to overlap
    the cut by ``post`` frames and the outbound clip's ``end`` to
    overlap by ``pre`` frames. We keep both clips' source ranges intact
    and only move the timeline-side ``in_frame`` / ``out_frame``.
    """
    new_outbound = FcpClip(
        name=outbound.name,
        in_frame=outbound.in_frame,
        out_frame=outbound.out_frame + placement.pre_trim,
        source_start=outbound.source_start,
        source_end=outbound.source_end,
    )
    new_inbound = FcpClip(
        name=inbound.name,
        in_frame=inbound.in_frame - placement.post_trim,
        out_frame=inbound.out_frame,
        source_start=inbound.source_start,
        source_end=inbound.source_end,
    )
    return new_outbound, new_inbound
