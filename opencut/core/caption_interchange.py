"""Canonical TTML-family caption interchange.

This module owns the shared caption model, TTML/EBU-TT/IMSC serialization,
safe parsing, and the deterministic conformance checks used by both legacy
broadcast exporters.  It intentionally has no optional runtime dependency.
"""

from __future__ import annotations

import os
import re
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

TT_NS = "http://www.w3.org/ns/ttml"
TTM_NS = "http://www.w3.org/ns/ttml#metadata"
TTP_NS = "http://www.w3.org/ns/ttml#parameter"
TTS_NS = "http://www.w3.org/ns/ttml#styling"
ITTP_NS = "http://www.w3.org/ns/ttml/profile/imsc1#parameter"
EBUTTM_NS = "urn:ebu:tt:metadata"
XML_NS = "http://www.w3.org/XML/1998/namespace"

IMSC_LEGACY_PROFILE = "http://www.w3.org/ns/ttml/profile/imsc1/text"
IMSC_13_PROFILE = "http://www.w3.org/ns/ttml/profile/imsc1.3/text"
EBU_TT_D_PROFILE = "urn:ebu:tt:distribution:2014-01"

PROFILE_TTML = "ttml"
PROFILE_IMSC_LEGACY = "imsc1"
PROFILE_IMSC_13 = "imsc1.3"
PROFILE_EBU_TT = "ebu_tt"

MAX_XML_BYTES = 5 * 1024 * 1024
_LANGUAGE_RE = re.compile(r"^[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*$")
_CLOCK_TIME_RE = re.compile(
    r"^(?P<hours>\d{2,}):(?P<minutes>[0-5]\d):(?P<seconds>[0-5]\d(?:\.\d+)?)$"
)
_OFFSET_TIME_RE = re.compile(r"^(?P<value>\d+(?:\.\d+)?)(?P<unit>h|m|s|ms|f)$")
_PAIR_RE = re.compile(r"^\d+(?:\.\d+)?(?:%|px|c)\s+\d+(?:\.\d+)?(?:%|px|c)$")
_WRITING_MODES = frozenset({"lrtb", "rltb", "tblr", "tbrl", "lr", "rl", "tb"})

for _prefix, _namespace in (
    ("", TT_NS),
    ("ttm", TTM_NS),
    ("ttp", TTP_NS),
    ("tts", TTS_NS),
    ("ittp", ITTP_NS),
    ("ebuttm", EBUTTM_NS),
):
    ET.register_namespace(_prefix, _namespace)


class CaptionInterchangeError(ValueError):
    """Raised when caption interchange input cannot be represented safely."""


@dataclass
class CaptionStyle:
    """Named TTML style and its ``tts:*`` properties."""

    id: str
    properties: dict[str, str] = field(default_factory=dict)
    style_refs: tuple[str, ...] = ()


@dataclass
class CaptionRegion:
    """Named layout region used by one or more cues."""

    id: str
    origin: str = "10% 80%"
    extent: str = "80% 20%"
    display_align: str = "after"
    writing_mode: str = "lrtb"
    direction: str = "ltr"
    style_refs: tuple[str, ...] = ()


@dataclass
class CaptionCue:
    """A timed text cue in the canonical interchange model."""

    id: str
    start: float
    end: float
    text: str
    region: str = "bottom"
    style_refs: tuple[str, ...] = ("default",)
    language: str = ""
    writing_mode: str = ""
    direction: str = ""


@dataclass
class CaptionDocument:
    """Canonical TTML-family document."""

    cues: list[CaptionCue] = field(default_factory=list)
    language: str = "en"
    title: str = ""
    frame_rate: float = 30.0
    styles: dict[str, CaptionStyle] = field(default_factory=dict)
    regions: dict[str, CaptionRegion] = field(default_factory=dict)


@dataclass(frozen=True)
class ConformanceIssue:
    code: str
    message: str
    location: str = ""
    severity: str = "error"

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
            "location": self.location,
            "severity": self.severity,
        }


@dataclass
class ConformanceReport:
    profile: str
    cue_count: int = 0
    issues: list[ConformanceIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ConformanceIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ConformanceIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "profile": self.profile,
            "cue_count": self.cue_count,
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
        }


def normalize_profile(profile: str) -> str:
    """Normalize public profile aliases while preserving explicit legacy use."""

    normalized = str(profile or "").strip().lower().replace("-", "_")
    aliases = {
        "": PROFILE_IMSC_LEGACY,
        "ttml": PROFILE_TTML,
        "ebu_tt": PROFILE_EBU_TT,
        "ebutt": PROFILE_EBU_TT,
        "legacy": PROFILE_IMSC_LEGACY,
        "imsc": PROFILE_IMSC_LEGACY,
        "imsc1": PROFILE_IMSC_LEGACY,
        "imsc1_legacy": PROFILE_IMSC_LEGACY,
        "imsc1.0": PROFILE_IMSC_LEGACY,
        "imsc1.3": PROFILE_IMSC_13,
        "imsc1_3": PROFILE_IMSC_13,
        "imsc13": PROFILE_IMSC_13,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise CaptionInterchangeError(
            f"Unsupported caption XML profile: {profile!r}. "
            "Use ttml, ebu_tt, imsc1 (legacy), or imsc1.3."
        ) from exc


def default_styles() -> dict[str, CaptionStyle]:
    return {
        "default": CaptionStyle(
            id="default",
            properties={
                "fontFamily": "proportionalSansSerif",
                "fontSize": "100%",
                "color": "white",
                "backgroundColor": "rgba(0,0,0,0.8)",
                "textAlign": "center",
            },
        ),
        "italic": CaptionStyle(id="italic", properties={"fontStyle": "italic"}),
        "bold": CaptionStyle(id="bold", properties={"fontWeight": "bold"}),
    }


def default_regions() -> dict[str, CaptionRegion]:
    return {
        "bottom": CaptionRegion(id="bottom"),
        "top": CaptionRegion(
            id="top",
            origin="10% 5%",
            extent="80% 15%",
            display_align="before",
        ),
    }


def document_from_items(
    items: Sequence[Mapping],
    *,
    language: str = "en",
    title: str = "",
    frame_rate: float = 30.0,
    truncate: tuple[int, int] | None = None,
) -> CaptionDocument:
    """Build a canonical document from route-style caption dictionaries."""

    cues: list[CaptionCue] = []
    styles = default_styles()
    regions = default_regions()
    for index, item in enumerate(items, start=1):
        text = str(item.get("text", item.get("content", "")))
        if truncate:
            text = _truncate_text(text, *truncate)
        position = str(item.get("region", item.get("position", "bottom"))).strip()
        region = "top" if position == "top" else position or "bottom"
        if region not in regions:
            regions[region] = CaptionRegion(id=region)
        raw_styles = item.get("style_refs", item.get("style", "default"))
        if isinstance(raw_styles, str):
            style_refs = tuple(part for part in raw_styles.split() if part)
        else:
            style_refs = tuple(str(part) for part in raw_styles or ())
        style_refs = style_refs or ("default",)
        cues.append(
            CaptionCue(
                id=str(item.get("id", item.get("index", index)) or index),
                start=float(item.get("start", item.get("start_time", 0.0))),
                end=float(item.get("end", item.get("end_time", 0.0))),
                text=text,
                region=region,
                style_refs=style_refs,
                language=str(item.get("language", "")),
                writing_mode=str(item.get("writing_mode", item.get("writingMode", ""))),
                direction=str(item.get("direction", "")),
            )
        )
    return CaptionDocument(
        cues=cues,
        language=language or "en",
        title=title,
        frame_rate=frame_rate,
        styles=styles,
        regions=regions,
    )


def serialize_caption_document(document: CaptionDocument, profile: str) -> bytes:
    """Serialize a canonical document to deterministic UTF-8 TTML-family XML."""

    normalized_profile = normalize_profile(profile)
    _validate_model(document)
    root = ET.Element(_q(TT_NS, "tt"))
    root.set(_q(XML_NS, "lang"), document.language)
    root.set(_q(TTP_NS, "timeBase"), "media")
    root.set(_q(TTP_NS, "frameRate"), _format_number(document.frame_rate))

    if normalized_profile in {PROFILE_IMSC_LEGACY, PROFILE_IMSC_13}:
        root.set(_q(TTP_NS, "cellResolution"), "32 15")
    if normalized_profile == PROFILE_IMSC_LEGACY:
        root.set(_q(TTP_NS, "profile"), IMSC_LEGACY_PROFILE)
        root.set(_q(ITTP_NS, "aspectRatio"), "16 9")
    elif normalized_profile == PROFILE_IMSC_13:
        root.set(_q(TTP_NS, "contentProfiles"), IMSC_13_PROFILE)
        root.set(_q(TTP_NS, "displayAspectRatio"), "16 9")

    head = ET.SubElement(root, _q(TT_NS, "head"))
    _append_metadata(head, document, normalized_profile)
    styling = ET.SubElement(head, _q(TT_NS, "styling"))
    for style in document.styles.values():
        style_el = ET.SubElement(styling, _q(TT_NS, "style"))
        style_el.set(_q(XML_NS, "id"), style.id)
        if style.style_refs:
            style_el.set("style", " ".join(style.style_refs))
        for name, value in sorted(style.properties.items()):
            style_el.set(_q(TTS_NS, name), value)

    layout = ET.SubElement(head, _q(TT_NS, "layout"))
    for region in document.regions.values():
        region_el = ET.SubElement(layout, _q(TT_NS, "region"))
        region_el.set(_q(XML_NS, "id"), region.id)
        region_el.set(_q(TTS_NS, "origin"), region.origin)
        region_el.set(_q(TTS_NS, "extent"), region.extent)
        region_el.set(_q(TTS_NS, "displayAlign"), region.display_align)
        region_el.set(_q(TTS_NS, "writingMode"), region.writing_mode)
        region_el.set(_q(TTS_NS, "direction"), region.direction)
        if region.style_refs:
            region_el.set("style", " ".join(region.style_refs))

    body = ET.SubElement(root, _q(TT_NS, "body"))
    div = ET.SubElement(body, _q(TT_NS, "div"))
    for index, cue in enumerate(document.cues, start=1):
        cue_el = ET.SubElement(div, _q(TT_NS, "p"))
        cue_el.set(_q(XML_NS, "id"), _safe_xml_id(cue.id, index))
        cue_el.set("begin", format_media_time(cue.start))
        cue_el.set("end", format_media_time(cue.end))
        cue_el.set("region", cue.region)
        if cue.style_refs:
            cue_el.set("style", " ".join(cue.style_refs))
        if cue.language:
            cue_el.set(_q(XML_NS, "lang"), cue.language)
        if cue.writing_mode:
            cue_el.set(_q(TTS_NS, "writingMode"), cue.writing_mode)
        if cue.direction:
            cue_el.set(_q(TTS_NS, "direction"), cue.direction)
        _append_multiline_text(cue_el, cue.text)

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def export_caption_document(
    document: CaptionDocument,
    output_path: str | os.PathLike[str],
    *,
    profile: str,
) -> ConformanceReport:
    """Serialize, validate, and atomically write a caption XML document."""

    normalized_profile = normalize_profile(profile)
    payload = serialize_caption_document(document, normalized_profile)
    report = validate_ttml(payload, expected_profile=normalized_profile)
    if not report.valid:
        details = "; ".join(issue.message for issue in report.errors)
        raise CaptionInterchangeError(f"Generated caption XML failed conformance: {details}")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except Exception:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
        raise
    return report


def validate_ttml(
    source: str | bytes | os.PathLike[str],
    *,
    expected_profile: str | None = None,
) -> ConformanceReport:
    """Validate structural and profile constraints for the supported corpus."""

    requested_profile = normalize_profile(expected_profile) if expected_profile else ""
    try:
        payload = _read_xml_source(source)
        root = _parse_xml(payload)
    except CaptionInterchangeError as exc:
        return ConformanceReport(
            profile=requested_profile or "unknown",
            issues=[ConformanceIssue("xml.invalid", str(exc), "/")],
        )

    detected_profile = _detect_profile(root)
    report = ConformanceReport(profile=detected_profile)
    if requested_profile and detected_profile != requested_profile:
        report.issues.append(
            ConformanceIssue(
                "profile.mismatch",
                f"Expected {requested_profile}, found {detected_profile}.",
                "/tt",
            )
        )
    if _local_name(root.tag) != "tt" or _namespace(root.tag) != TT_NS:
        report.issues.append(
            ConformanceIssue("root.invalid", "Root element must be TTML <tt>.", "/")
        )
        return report

    language = root.get(_q(XML_NS, "lang"), "")
    if not _LANGUAGE_RE.fullmatch(language):
        report.issues.append(
            ConformanceIssue(
                "language.invalid",
                "xml:lang must contain a valid BCP 47-style language tag.",
                "/tt/@xml:lang",
            )
        )
    if root.get(_q(TTP_NS, "timeBase")) != "media":
        report.issues.append(
            ConformanceIssue(
                "timebase.invalid",
                "Supported caption XML must use ttp:timeBase='media'.",
                "/tt/@ttp:timeBase",
            )
        )

    if detected_profile == PROFILE_IMSC_13:
        _validate_imsc13_root(root, report)
    elif detected_profile == PROFILE_EBU_TT:
        standards = [
            (node.text or "").strip()
            for node in root.findall(f".//{{{EBUTTM_NS}}}conformsToStandard")
        ]
        if EBU_TT_D_PROFILE not in standards:
            report.issues.append(
                ConformanceIssue(
                    "ebu.profile.missing",
                    "EBU-TT output must declare the EBU-TT-D profile.",
                    "/tt/head/metadata",
                )
            )

    styles = _collect_named_elements(root, "style", report)
    regions = _collect_named_elements(root, "region", report)
    frame_rate = _parse_positive_float(root.get(_q(TTP_NS, "frameRate")), 30.0)
    cue_ids: set[str] = set()
    cues = root.findall(f".//{{{TT_NS}}}p")
    report.cue_count = len(cues)
    if not cues:
        report.issues.append(
            ConformanceIssue("cues.missing", "At least one timed <p> cue is required.", "/tt/body")
        )

    for index, cue in enumerate(cues, start=1):
        location = f"/tt/body//p[{index}]"
        cue_id = cue.get(_q(XML_NS, "id"), "")
        if not cue_id:
            report.issues.append(
                ConformanceIssue("cue.id.missing", "Cue requires xml:id.", location)
            )
        elif cue_id in cue_ids:
            report.issues.append(
                ConformanceIssue("cue.id.duplicate", f"Duplicate cue id {cue_id!r}.", location)
            )
        cue_ids.add(cue_id)

        try:
            start = parse_time_expression(cue.get("begin", ""), frame_rate=frame_rate)
            if cue.get("end"):
                end = parse_time_expression(cue.get("end", ""), frame_rate=frame_rate)
            else:
                end = start + parse_time_expression(cue.get("dur", ""), frame_rate=frame_rate)
            if start < 0 or end <= start:
                raise CaptionInterchangeError("end must be later than begin")
        except CaptionInterchangeError as exc:
            report.issues.append(
                ConformanceIssue("cue.timing.invalid", f"Invalid cue timing: {exc}.", location)
            )

        if not _element_text(cue).strip():
            report.issues.append(
                ConformanceIssue("cue.text.empty", "Cue text cannot be empty.", location)
            )
        region_ref = cue.get("region", "")
        if region_ref and region_ref not in regions:
            report.issues.append(
                ConformanceIssue(
                    "cue.region.unknown",
                    f"Cue references unknown region {region_ref!r}.",
                    location,
                )
            )
        _validate_style_refs(cue, styles, report, location)
        _validate_direction(cue, report, location)

    for region_id, region in regions.items():
        location = f"/tt/head/layout/region[@xml:id='{region_id}']"
        for attr_name in ("origin", "extent"):
            value = region.get(_q(TTS_NS, attr_name), "")
            if not _PAIR_RE.fullmatch(value):
                report.issues.append(
                    ConformanceIssue(
                        f"region.{attr_name}.invalid",
                        f"Region {attr_name} must be a two-value length or percentage pair.",
                        location,
                    )
                )
        _validate_style_refs(region, styles, report, location)
        _validate_direction(region, report, location)

    for style_id, style in styles.items():
        _validate_style_refs(
            style,
            styles,
            report,
            f"/tt/head/styling/style[@xml:id='{style_id}']",
        )
        _validate_direction(
            style,
            report,
            f"/tt/head/styling/style[@xml:id='{style_id}']",
        )
    return report


def parse_ttml(
    source: str | bytes | os.PathLike[str],
    *,
    expected_profile: str | None = None,
) -> CaptionDocument:
    """Parse validated TTML-family XML into the canonical document model."""

    payload = _read_xml_source(source)
    report = validate_ttml(payload, expected_profile=expected_profile)
    if not report.valid:
        details = "; ".join(issue.message for issue in report.errors)
        raise CaptionInterchangeError(f"Caption XML is not conformant: {details}")
    root = _parse_xml(payload)
    frame_rate = _parse_positive_float(root.get(_q(TTP_NS, "frameRate")), 30.0)
    styles: dict[str, CaptionStyle] = {}
    for style_el in root.findall(f".//{{{TT_NS}}}styling/{{{TT_NS}}}style"):
        style_id = style_el.get(_q(XML_NS, "id"), "")
        properties = {
            _local_name(name): value
            for name, value in style_el.attrib.items()
            if _namespace(name) == TTS_NS
        }
        styles[style_id] = CaptionStyle(
            id=style_id,
            properties=properties,
            style_refs=_split_refs(style_el.get("style")),
        )

    regions: dict[str, CaptionRegion] = {}
    for region_el in root.findall(f".//{{{TT_NS}}}layout/{{{TT_NS}}}region"):
        region_id = region_el.get(_q(XML_NS, "id"), "")
        regions[region_id] = CaptionRegion(
            id=region_id,
            origin=region_el.get(_q(TTS_NS, "origin"), "10% 80%"),
            extent=region_el.get(_q(TTS_NS, "extent"), "80% 20%"),
            display_align=region_el.get(_q(TTS_NS, "displayAlign"), "after"),
            writing_mode=region_el.get(_q(TTS_NS, "writingMode"), "lrtb"),
            direction=region_el.get(_q(TTS_NS, "direction"), "ltr"),
            style_refs=_split_refs(region_el.get("style")),
        )

    cues: list[CaptionCue] = []
    for index, cue_el in enumerate(root.findall(f".//{{{TT_NS}}}p"), start=1):
        start = parse_time_expression(cue_el.get("begin", ""), frame_rate=frame_rate)
        if cue_el.get("end"):
            end = parse_time_expression(cue_el.get("end", ""), frame_rate=frame_rate)
        else:
            end = start + parse_time_expression(cue_el.get("dur", ""), frame_rate=frame_rate)
        cues.append(
            CaptionCue(
                id=cue_el.get(_q(XML_NS, "id"), f"cue{index}"),
                start=start,
                end=end,
                text=_element_text(cue_el),
                region=cue_el.get("region", "bottom"),
                style_refs=_split_refs(cue_el.get("style")),
                language=cue_el.get(_q(XML_NS, "lang"), ""),
                writing_mode=cue_el.get(_q(TTS_NS, "writingMode"), ""),
                direction=cue_el.get(_q(TTS_NS, "direction"), ""),
            )
        )

    title_node = root.find(f".//{{{TTM_NS}}}title")
    if title_node is None:
        title_node = root.find(f".//{{{EBUTTM_NS}}}documentOriginalProgrammeTitle")
    return CaptionDocument(
        cues=cues,
        language=root.get(_q(XML_NS, "lang"), "en"),
        title=(title_node.text or "").strip() if title_node is not None else "",
        frame_rate=frame_rate,
        styles=styles,
        regions=regions,
    )


def format_media_time(seconds: float) -> str:
    if seconds < 0:
        raise CaptionInterchangeError("Caption times cannot be negative.")
    milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{millis:03d}"


def parse_time_expression(value: str, *, frame_rate: float = 30.0) -> float:
    raw = str(value or "").strip()
    clock = _CLOCK_TIME_RE.fullmatch(raw)
    if clock:
        return (
            int(clock.group("hours")) * 3600
            + int(clock.group("minutes")) * 60
            + float(clock.group("seconds"))
        )
    offset = _OFFSET_TIME_RE.fullmatch(raw)
    if offset:
        amount = float(offset.group("value"))
        factors = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001, "f": 1 / frame_rate}
        return amount * factors[offset.group("unit")]
    raise CaptionInterchangeError(f"unsupported time expression {value!r}")


def _append_metadata(head: ET.Element, document: CaptionDocument, profile: str) -> None:
    metadata = ET.SubElement(head, _q(TT_NS, "metadata"))
    if document.title:
        ET.SubElement(metadata, _q(TTM_NS, "title")).text = document.title
    if profile == PROFILE_EBU_TT:
        document_metadata = ET.SubElement(metadata, _q(EBUTTM_NS, "documentMetadata"))
        ET.SubElement(
            document_metadata, _q(EBUTTM_NS, "documentEbuttVersion")
        ).text = "v1.0"
        ET.SubElement(
            document_metadata, _q(EBUTTM_NS, "conformsToStandard")
        ).text = EBU_TT_D_PROFILE
        if document.title:
            ET.SubElement(
                document_metadata, _q(EBUTTM_NS, "documentOriginalProgrammeTitle")
            ).text = document.title


def _append_multiline_text(parent: ET.Element, text: str) -> None:
    lines = text.split("\n")
    parent.text = lines[0]
    for line in lines[1:]:
        line_break = ET.SubElement(parent, _q(TT_NS, "br"))
        line_break.tail = line


def _element_text(element: ET.Element) -> str:
    chunks = [element.text or ""]
    for child in element:
        if _namespace(child.tag) == TT_NS and _local_name(child.tag) == "br":
            chunks.append("\n")
        else:
            chunks.append("".join(child.itertext()))
        chunks.append(child.tail or "")
    return "".join(chunks)


def _read_xml_source(source: str | bytes | os.PathLike[str]) -> bytes:
    if isinstance(source, bytes):
        payload = source
    elif isinstance(source, os.PathLike):
        payload = Path(source).read_bytes()
    else:
        raw = str(source)
        if raw.lstrip().startswith("<"):
            payload = raw.encode("utf-8")
        else:
            try:
                path = Path(raw)
                if not path.is_file():
                    raise CaptionInterchangeError(f"Caption XML file not found: {path}")
                payload = path.read_bytes()
            except OSError as exc:
                raise CaptionInterchangeError(f"Caption XML path is invalid: {exc}") from exc
    if len(payload) > MAX_XML_BYTES:
        raise CaptionInterchangeError(
            f"Caption XML exceeds the {MAX_XML_BYTES}-byte safety limit."
        )
    upper_payload = payload.upper()
    if b"<!DOCTYPE" in upper_payload or b"<!ENTITY" in upper_payload:
        raise CaptionInterchangeError("DTD and entity declarations are not allowed.")
    return payload


def _parse_xml(payload: bytes) -> ET.Element:
    try:
        return ET.fromstring(payload)
    except (ET.ParseError, ValueError) as exc:
        raise CaptionInterchangeError(f"Malformed caption XML: {exc}") from exc


def _detect_profile(root: ET.Element) -> str:
    content_profiles = root.get(_q(TTP_NS, "contentProfiles"), "").split()
    if IMSC_13_PROFILE in content_profiles:
        return PROFILE_IMSC_13
    if root.get(_q(TTP_NS, "profile")) == IMSC_LEGACY_PROFILE:
        return PROFILE_IMSC_LEGACY
    standards = {
        (node.text or "").strip()
        for node in root.findall(f".//{{{EBUTTM_NS}}}conformsToStandard")
    }
    if EBU_TT_D_PROFILE in standards or root.find(
        f".//{{{EBUTTM_NS}}}documentEbuttVersion"
    ) is not None:
        return PROFILE_EBU_TT
    return PROFILE_TTML


def _validate_imsc13_root(root: ET.Element, report: ConformanceReport) -> None:
    profiles = root.get(_q(TTP_NS, "contentProfiles"), "").split()
    if profiles.count(IMSC_13_PROFILE) != 1:
        report.issues.append(
            ConformanceIssue(
                "imsc13.profile.missing",
                "ttp:contentProfiles must include the IMSC 1.3 text profile exactly once.",
                "/tt/@ttp:contentProfiles",
            )
        )
    resolution = root.get(_q(TTP_NS, "cellResolution"), "")
    parts = resolution.split()
    if len(parts) != 2 or not all(part.isdigit() and int(part) > 0 for part in parts):
        report.issues.append(
            ConformanceIssue(
                "imsc13.cell_resolution.invalid",
                "ttp:cellResolution must contain two positive integers.",
                "/tt/@ttp:cellResolution",
            )
        )
    aspect_ratio = root.get(_q(TTP_NS, "displayAspectRatio"), "")
    if aspect_ratio and not re.fullmatch(r"[1-9]\d* [1-9]\d*", aspect_ratio):
        report.issues.append(
            ConformanceIssue(
                "imsc13.aspect_ratio.invalid",
                "ttp:displayAspectRatio must contain two positive integers.",
                "/tt/@ttp:displayAspectRatio",
            )
        )
    if root.get(_q(ITTP_NS, "aspectRatio")):
        report.issues.append(
            ConformanceIssue(
                "imsc13.aspect_ratio.deprecated",
                "ittp:aspectRatio is permitted but deprecated; use ttp:displayAspectRatio.",
                "/tt/@ittp:aspectRatio",
                severity="warning",
            )
        )


def _collect_named_elements(
    root: ET.Element, local_name: str, report: ConformanceReport
) -> dict[str, ET.Element]:
    elements: dict[str, ET.Element] = {}
    for element in root.findall(f".//{{{TT_NS}}}{local_name}"):
        element_id = element.get(_q(XML_NS, "id"), "")
        if not element_id:
            report.issues.append(
                ConformanceIssue(
                    f"{local_name}.id.missing",
                    f"<{local_name}> requires xml:id.",
                    f"/tt/head//{local_name}",
                )
            )
        elif element_id in elements:
            report.issues.append(
                ConformanceIssue(
                    f"{local_name}.id.duplicate",
                    f"Duplicate {local_name} id {element_id!r}.",
                    f"/tt/head//{local_name}",
                )
            )
        else:
            elements[element_id] = element
    return elements


def _validate_style_refs(
    element: ET.Element,
    styles: Mapping[str, ET.Element],
    report: ConformanceReport,
    location: str,
) -> None:
    for style_ref in _split_refs(element.get("style")):
        if style_ref not in styles:
            report.issues.append(
                ConformanceIssue(
                    "style.unknown",
                    f"Unknown style reference {style_ref!r}.",
                    location,
                )
            )


def _validate_direction(
    element: ET.Element, report: ConformanceReport, location: str
) -> None:
    writing_mode = element.get(_q(TTS_NS, "writingMode"), "")
    direction = element.get(_q(TTS_NS, "direction"), "")
    if writing_mode and writing_mode not in _WRITING_MODES:
        report.issues.append(
            ConformanceIssue(
                "writing_mode.invalid",
                f"Unsupported tts:writingMode {writing_mode!r}.",
                location,
            )
        )
    if direction and direction not in {"ltr", "rtl"}:
        report.issues.append(
            ConformanceIssue(
                "direction.invalid",
                f"Unsupported tts:direction {direction!r}.",
                location,
            )
        )


def _validate_model(document: CaptionDocument) -> None:
    if not document.cues:
        raise CaptionInterchangeError("At least one caption cue is required.")
    if not _LANGUAGE_RE.fullmatch(document.language):
        raise CaptionInterchangeError(f"Invalid document language: {document.language!r}.")
    for region in document.regions.values():
        if region.writing_mode not in _WRITING_MODES:
            raise CaptionInterchangeError(
                f"Unsupported writing mode for region {region.id!r}: {region.writing_mode!r}."
            )
        if region.direction not in {"ltr", "rtl"}:
            raise CaptionInterchangeError(
                f"Unsupported direction for region {region.id!r}: {region.direction!r}."
            )
    for cue in document.cues:
        if cue.start < 0 or cue.end <= cue.start:
            raise CaptionInterchangeError(f"Cue {cue.id!r} has invalid timing.")
        if not cue.text.strip():
            raise CaptionInterchangeError(f"Cue {cue.id!r} has no text.")
        if cue.region not in document.regions:
            raise CaptionInterchangeError(
                f"Cue {cue.id!r} references unknown region {cue.region!r}."
            )
        unknown_styles = [style for style in cue.style_refs if style not in document.styles]
        if unknown_styles:
            raise CaptionInterchangeError(
                f"Cue {cue.id!r} references unknown styles: {', '.join(unknown_styles)}."
            )
        if cue.language and not _LANGUAGE_RE.fullmatch(cue.language):
            raise CaptionInterchangeError(
                f"Cue {cue.id!r} has invalid language {cue.language!r}."
            )
        if cue.writing_mode and cue.writing_mode not in _WRITING_MODES:
            raise CaptionInterchangeError(
                f"Cue {cue.id!r} has unsupported writing mode {cue.writing_mode!r}."
            )
        if cue.direction and cue.direction not in {"ltr", "rtl"}:
            raise CaptionInterchangeError(
                f"Cue {cue.id!r} has unsupported direction {cue.direction!r}."
            )
        _validate_xml_text(cue.text, cue.id)


def _validate_xml_text(text: str, cue_id: str) -> None:
    for char in text:
        codepoint = ord(char)
        if codepoint in (0x9, 0xA, 0xD):
            continue
        if not (
            0x20 <= codepoint <= 0xD7FF
            or 0xE000 <= codepoint <= 0xFFFD
            or 0x10000 <= codepoint <= 0x10FFFF
        ):
            raise CaptionInterchangeError(
                f"Cue {cue_id!r} contains a character XML 1.0 cannot represent."
            )


def _truncate_text(text: str, max_chars: int, max_lines: int) -> str:
    output: list[str] = []
    for raw_line in text.split("\n"):
        current = ""
        for word in raw_line.split():
            candidate = f"{current} {word}".strip()
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    output.append(current)
                current = word[:max_chars]
            if len(output) >= max_lines:
                break
        if current and len(output) < max_lines:
            output.append(current)
        if len(output) >= max_lines:
            break
    return "\n".join(output[:max_lines])


def _safe_xml_id(value: str, index: int) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("_")
    if not normalized or not re.match(r"[A-Za-z_]", normalized):
        normalized = f"cue{index}_{normalized}" if normalized else f"cue{index}"
    return normalized


def _split_refs(value: str | None) -> tuple[str, ...]:
    return tuple(part for part in str(value or "").split() if part)


def _parse_positive_float(value: str | None, default: float) -> float:
    try:
        parsed = float(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _format_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def _q(namespace: str, local_name: str) -> str:
    return f"{{{namespace}}}{local_name}"


def _namespace(name: str) -> str:
    return name[1:].split("}", 1)[0] if name.startswith("{") else ""


def _local_name(name: str) -> str:
    return name.split("}", 1)[-1]
