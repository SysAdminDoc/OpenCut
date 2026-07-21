"""Script-to-timeline assembly: alternates, preview, and reversible write-back.

Covers the parts of the rough-cut assembler that make it a reviewable,
reversible plan: best take + ranked alternate takes on separate tracks, a
preview mode that writes nothing, and a write-back that records a revertible
operation-journal entry. Deterministic — no Whisper, no host, fixture data only.
"""

import os
import tempfile
import xml.etree.ElementTree as ET

from opencut.core import script_to_roughcut as rc


# --- fixtures -------------------------------------------------------------

SCRIPT = """
INT. OFFICE - DAY

ALICE
    We need to ship this today.

BOB
    The build is finally green.
"""


def _transcripts():
    """Two clips per line so line 1 has a distinct alternate take."""
    return [
        # Line 1 — best on take A, weaker alternate on take B.
        rc.TranscriptSegment(clip_path="/takeA.mp4", text="We need to ship this today",
                             start_time=1.0, end_time=4.0, confidence=0.9),
        rc.TranscriptSegment(clip_path="/takeB.mp4", text="We need to ship this today please",
                             start_time=2.0, end_time=5.5, confidence=0.6),
        # Line 2 — single good take on C.
        rc.TranscriptSegment(clip_path="/takeC.mp4", text="The build is finally green",
                             start_time=0.5, end_time=3.0, confidence=0.85),
    ]


def _plan_segments(max_alternates=2):
    script = rc.parse_script_segments(SCRIPT)
    matches = rc.fuzzy_match_segments(script, _transcripts(), threshold=0.4)
    segs = rc.select_takes_with_alternates(matches, script, max_alternates=max_alternates)
    return script, matches, segs


# --- matching / plan generation ------------------------------------------

def test_plan_maps_script_lines_to_best_takes():
    _, _, segs = _plan_segments()
    dialogue = [s for s in segs if s.script_text]
    matched = [s for s in dialogue if s.clip_path]
    assert len(matched) >= 2
    # Line 1 resolves to its best take, not the weaker alternate.
    line1 = next(s for s in matched if "ship this today" in s.script_text)
    assert line1.clip_path == "/takeA.mp4"


def test_alternates_are_distinct_clips_and_ranked():
    _, _, segs = _plan_segments(max_alternates=2)
    line1 = next(s for s in segs if s.clip_path and "ship this today" in s.script_text)
    assert line1.alternates, "expected an alternate take for line 1"
    alt_clips = [a["clip_path"] for a in line1.alternates]
    assert line1.clip_path not in alt_clips  # never repeat the primary clip
    assert "/takeB.mp4" in alt_clips


def test_max_alternates_zero_disables_alternates():
    script = rc.parse_script_segments(SCRIPT)
    matches = rc.fuzzy_match_segments(script, _transcripts(), threshold=0.4)
    segs = rc.select_takes_with_alternates(matches, script, max_alternates=0)
    assert all(not s.alternates for s in segs)


# --- alternates on separate tracks (XML) ----------------------------------

def test_xml_export_places_alternates_on_separate_tracks():
    _, _, segs = _plan_segments(max_alternates=2)
    out = os.path.join(tempfile.mkdtemp(), "rough_cut.xml")
    rc.export_xml_timeline(segs, out)
    tree = ET.parse(out)
    tracks = tree.findall(".//video/track")
    assert len(tracks) >= 2, "alternate takes should add a second video track"
    # The alternate clip appears somewhere in the XML.
    content = open(out, encoding="utf-8").read()
    assert "takeB.mp4" in content


def test_xml_export_single_track_when_no_alternates():
    segs = [
        rc.RoughCutSegment(index=0, clip_path="/a.mp4", in_point=0.0, out_point=5.0),
        rc.RoughCutSegment(index=1, clip_path="/b.mp4", in_point=2.0, out_point=8.0),
    ]
    out = os.path.join(tempfile.mkdtemp(), "rc.xml")
    rc.export_xml_timeline(segs, out)
    tracks = ET.parse(out).findall(".//video/track")
    assert len(tracks) == 1


# --- preview (no side effects) --------------------------------------------

def test_preview_writes_nothing_and_flags_preview():
    out_dir = tempfile.mkdtemp()
    result = rc.assemble_rough_cut(
        SCRIPT,
        transcript_map={},
        output_dir=out_dir,
        write_back=False,
    )
    # Preview still produces the plan...
    assert result.preview is True
    assert result.timeline_path == ""
    assert result.journal_entry_id is None
    # ...but writes no timeline files.
    assert not os.listdir(out_dir)


def test_no_output_dir_is_preview():
    result = rc.assemble_rough_cut(SCRIPT, output_dir="")
    assert result.preview is True
    assert result.timeline_path == ""


# --- reversible write-back via the journal --------------------------------

def _isolate_journal():
    from opencut import journal as jm

    tmpdir = tempfile.mkdtemp(prefix="opencut_rc_journal_")
    try:
        if getattr(jm._thread_local, "conn", None):
            jm._thread_local.conn.close()
            jm._thread_local.conn = None
    except Exception:
        pass
    jm._ALL_CONNECTIONS.clear()
    jm._DB_PATH = os.path.join(tmpdir, "journal.db")
    return jm


def test_write_back_records_revertible_journal_entry(monkeypatch):
    jm = _isolate_journal()
    # Provide transcripts inline so no Whisper is needed.
    script = rc.parse_script_segments(SCRIPT)
    matches = rc.fuzzy_match_segments(script, _transcripts(), threshold=0.4)
    segs = rc.select_takes_with_alternates(matches, script, max_alternates=2)

    out_dir = tempfile.mkdtemp()
    result = rc.RoughCutResult(
        segments=segs,
        matched_segments=sum(1 for s in segs if s.clip_path),
        total_segments=len(segs),
        alternate_takes=sum(len(s.alternates) for s in segs),
    )
    result.timeline_path = rc.export_xml_timeline(segs, os.path.join(out_dir, "rough_cut.xml"))
    result.format = "xml"

    entry_id = rc.record_write_back(result, sequence_name="Fixture Cut")
    assert entry_id is not None

    entries = jm.list_entries()
    assert entries and entries[0]["action"] == "import_sequence"
    assert entries[0]["revertible"] is True
    assert entries[0]["inverse"]["timeline_path"] == result.timeline_path
    jm.close_all_connections()


def test_assemble_write_back_sets_journal_id():
    jm = _isolate_journal()
    out_dir = tempfile.mkdtemp()

    # Persist fixture transcripts to JSON and feed via transcript_map so the
    # full pipeline runs without Whisper.
    tmap = {}
    for clip, segs in {
        "/takeA.mp4": [{"text": "We need to ship this today", "start": 1.0, "end": 4.0}],
        "/takeC.mp4": [{"text": "The build is finally green", "start": 0.5, "end": 3.0}],
    }.items():
        p = os.path.join(out_dir, os.path.basename(clip) + ".json")
        import json
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"segments": segs}, f)
        tmap[clip] = p

    result = rc.assemble_rough_cut(
        SCRIPT,
        transcript_map=tmap,
        output_dir=out_dir,
        output_format="xml",
        write_back=True,
    )
    assert result.preview is False
    assert result.timeline_path and os.path.isfile(result.timeline_path)
    assert result.journal_entry_id is not None
    jm.close_all_connections()
