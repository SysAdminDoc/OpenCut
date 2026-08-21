"""Reject unsafe XML before stdlib parsers see it.

``xml.etree`` and ``minidom`` do not resolve external entities, but a
crafted DTD can still expand entities in-memory. Caption, Final Draft,
and GPX files are commonly downloaded from strangers, so import paths
refuse a DTD or entity declaration and cap the payload size.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union
from xml.etree import ElementTree as ET

from opencut.errors import OpenCutError

MAX_XML_BYTES = 5 * 1024 * 1024
_UNSAFE_MARKERS = (b"<!DOCTYPE", b"<!ENTITY")

XmlSource = Union[str, bytes, Path]


class UnsafeXmlError(OpenCutError):
    """Raised when XML input carries a DTD, entity, or exceeds the size cap."""

    def __init__(self, message: str):
        super().__init__(
            code="INVALID_INPUT",
            message=message,
            suggestion=(
                "Use a caption, screenplay, or GPS file exported without a "
                "document type declaration."
            ),
            status=400,
        )


def inspect_xml_bytes(payload: bytes) -> bytes:
    """Return *payload* or raise :class:`UnsafeXmlError`."""
    if len(payload) > MAX_XML_BYTES:
        raise UnsafeXmlError(
            f"XML exceeds the {MAX_XML_BYTES}-byte safety limit."
        )
    upper_payload = payload.upper()
    if any(marker in upper_payload for marker in _UNSAFE_MARKERS):
        raise UnsafeXmlError("DTD and entity declarations are not allowed.")
    return payload


def read_xml_path(path: XmlSource) -> bytes:
    """Read a file and reject oversized or DTD-bearing payloads."""
    payload = Path(path).read_bytes()
    return inspect_xml_bytes(payload)


def parse_xml_root(path: XmlSource) -> ET.Element:
    """Parse a trusted-size, DTD-free XML file into its root element."""
    payload = read_xml_path(path)
    try:
        return ET.fromstring(payload)
    except (ET.ParseError, ValueError) as exc:
        raise UnsafeXmlError(f"Malformed XML: {exc}") from exc
