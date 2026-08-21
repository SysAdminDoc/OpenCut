"""Import-boundary XML must refuse DTDs before stdlib parsers see them."""
from __future__ import annotations

import pytest

from opencut.core.xml_safe import UnsafeXmlError, inspect_xml_bytes, parse_xml_root

DTD_BOMB = b'<!DOCTYPE gpx [<!ENTITY x "boom">]><gpx>&x;</gpx>'

VALID_GPX = """<?xml version="1.0"?>
<gpx xmlns="http://www.topografix.com/GPX/1/1">
  <trk><trkseg>
    <trkpt lat="37.7749" lon="-122.4194"><ele>100</ele></trkpt>
  </trkseg></trk>
</gpx>
"""

VALID_FDX = """<?xml version="1.0"?>
<FinalDraft>
  <Content>
    <Paragraph Type="Scene Heading"><Text>INT. OFFICE - DAY</Text></Paragraph>
    <Paragraph Type="Action"><Text>Someone sits.</Text></Paragraph>
  </Content>
</FinalDraft>
"""


def _write(tmp_path, name: str, body: bytes | str) -> str:
    path = tmp_path / name
    if isinstance(body, bytes):
        path.write_bytes(body)
    else:
        path.write_text(body, encoding="utf-8")
    return str(path)


def test_inspect_xml_bytes_rejects_dtd():
    with pytest.raises(UnsafeXmlError, match="DTD"):
        inspect_xml_bytes(DTD_BOMB)


def test_inspect_xml_bytes_rejects_oversized_payload():
    from opencut.core.xml_safe import MAX_XML_BYTES

    with pytest.raises(UnsafeXmlError, match="safety limit"):
        inspect_xml_bytes(b"<" + b"x" * (MAX_XML_BYTES + 1))


def test_parse_xml_root_accepts_plain_xml(tmp_path):
    path = _write(tmp_path, "ok.xml", "<root><item>1</item></root>")
    root = parse_xml_root(path)
    assert root.tag == "root"
    assert root.find("item").text == "1"


@pytest.mark.parametrize(
    ("module_path", "func_name", "filename", "valid_body"),
    [
        ("opencut.core.screenplay_parser", "parse_fdx", "script.fdx", VALID_FDX),
        ("opencut.core.flight_path_map", "_parse_gpx", "track.gpx", VALID_GPX),
    ],
)
def test_third_party_xml_imports_reject_dtds(
    tmp_path, module_path, func_name, filename, valid_body
):
    import importlib

    module = importlib.import_module(module_path)
    parse = getattr(module, func_name)

    safe_path = _write(tmp_path, f"ok-{filename}", valid_body)
    parse(safe_path)

    bomb_path = _write(tmp_path, f"bomb-{filename}", DTD_BOMB)
    with pytest.raises(UnsafeXmlError, match="DTD"):
        parse(bomb_path)
