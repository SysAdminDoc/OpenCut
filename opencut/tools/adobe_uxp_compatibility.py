"""Static Adobe Premiere UXP compatibility metadata and drift checks.

The UXP runtime is only available inside Premiere, so a release check cannot
depend on a live host.  This module keeps the host-version contract in a
generated JSON manifest and checks the UXP source tree against the declared
capabilities.  The scanner is intentionally conservative: it understands the
object aliases used by OpenCut's bridge and reports an undeclared call with a
source file and line number instead of guessing that a new API is safe.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

PACKAGE = "@adobe/premierepro"
API_VERSION = 2
MANIFEST_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "opencut" / "_generated" / "adobe_uxp_compatibility.json"
PACKAGE_JSON_PATH = REPO_ROOT / "extension" / "com.opencut.panel" / "package.json"
UXP_MANIFEST_PATH = REPO_ROOT / "extension" / "com.opencut.uxp" / "manifest.json"

SOURCE_PATHS = (
    REPO_ROOT / "extension" / "com.opencut.uxp" / "main.js",
    REPO_ROOT / "extension" / "com.opencut.uxp" / "uxp-theme.js",
    REPO_ROOT / "extension" / "com.opencut.uxp" / "bolt-webview" / "src" / "api" / "premierepro.ts",
)
SOURCE_RELATIVE_PATHS = tuple(path.relative_to(REPO_ROOT) for path in SOURCE_PATHS)

# Premiere 25.6 was the first supported UXP host in OpenCut.  26.2 is kept as
# its own fixture because Adobe shipped the Hybrid UXP lane there, while 26.3
# is the first band in this catalogue with the observed synchronous
# Sequence.setSelection and locked-access behavior.
HOST_BANDS = (
    {
        "id": "25.6",
        "minimum_host": "25.6",
        "maximum_host": "26.1",
        "api_version": API_VERSION,
    },
    {
        "id": "26.2",
        "minimum_host": "26.2",
        "maximum_host": "26.2",
        "api_version": API_VERSION,
    },
    {
        "id": "26.3",
        "minimum_host": "26.3",
        "maximum_host": None,
        "api_version": API_VERSION,
    },
)

HOST_FIXTURES = (
    {
        "host": "25.6",
        "assertions": [
            {"capability": "Sequence.setSelection", "sync_async": "async"},
        ],
    },
    {
        "host": "26.2",
        "assertions": [
            {"capability": "Sequence.setSelection", "sync_async": "async"},
        ],
    },
    {
        "host": "26.3",
        "assertions": [
            {"capability": "Sequence.setSelection", "sync_async": "sync"},
        ],
    },
)


def _capability(
    capability_id: str,
    owner: str,
    member: str,
    *,
    minimum_host: str = "25.6",
    minimum_package: str = "25.6.0",
    sync_async: str = "async",
    fallback: str,
    aliases: Iterable[str] = (),
    host_behavior: Optional[Iterable[dict]] = None,
    fixture_only: bool = False,
) -> dict:
    """Build one compact source declaration used by the generated manifest."""
    behavior = list(host_behavior or ())
    if not behavior:
        behavior = [{"minimum_host": minimum_host, "sync_async": sync_async}]
    return {
        "id": capability_id,
        "owner": owner,
        "member": member,
        "minimum_host": minimum_host,
        "minimum_package_version": minimum_package,
        "api_version": API_VERSION,
        "sync_async": sync_async,
        "host_behavior": behavior,
        "fallback": fallback,
        "aliases": list(aliases),
        "fixture_only": fixture_only,
    }


# The aliases are the object shapes used by main.js and the Bolt WebView
# helper.  Generic Object.* entries are deliberate: project items and track
# items are returned as opaque UXP objects, so their runtime type cannot be
# established statically without a Premiere host.
API_CATALOGUE = (
    _capability(
        "module.import",
        "premierepro",
        "import",
        sync_async="async",
        fallback="Disable direct host actions and retain the CEP/backend path.",
        aliases=("module.import",),
    ),
    _capability(
        "Application.getProjectList",
        "Application",
        "getProjectList",
        fallback="Return no active project and surface the UXP capability state.",
        aliases=("module.app.getProjectList",),
    ),
    _capability(
        "Project.getActiveSequence",
        "Project",
        "getActiveSequence",
        fallback="Return no active sequence and keep backend-only workflows available.",
        aliases=("Project.getActiveSequence",),
    ),
    _capability(
        "Project.getName",
        "Project",
        "getName",
        fallback="Use an empty project name in the returned summary.",
        aliases=("Project.getName",),
    ),
    _capability(
        "Project.getRootItem",
        "Project",
        "getRootItem",
        fallback="Treat the project tree as unavailable and return an empty result.",
        aliases=("Project.getRootItem",),
    ),
    _capability(
        "Project.importFiles",
        "Project",
        "importFiles",
        fallback="Report import unavailable and leave the source file untouched.",
        aliases=("Project.importFiles",),
    ),
    _capability(
        "Project.createBin",
        "Project",
        "createBin",
        fallback="Skip optional bin creation and continue with the project root.",
        aliases=("Project.createBin",),
    ),
    _capability(
        "Project.deleteItem",
        "Project",
        "deleteItem",
        fallback="Use the project-item delete/remove methods when exposed.",
        aliases=("Project.deleteItem",),
    ),
    _capability(
        "Project.executeTransaction",
        "Project",
        "executeTransaction",
        sync_async="sync",
        fallback="Reject the mutating action and retain the CEP fallback.",
        aliases=("Project.executeTransaction",),
    ),
    _capability(
        "Project.lockedAccess",
        "Project",
        "lockedAccess",
        minimum_host="26.3",
        minimum_package="26.3.0",
        sync_async="sync",
        fallback="Run executeTransaction without the optional lock wrapper on 25.6-26.2.",
        aliases=("Project.lockedAccess",),
    ),
    _capability(
        "CompoundAction.addAction",
        "CompoundAction",
        "addAction",
        sync_async="sync",
        fallback="Reject the transaction rather than partially mutating the sequence.",
        aliases=("CompoundAction.addAction",),
    ),
    _capability(
        "Sequence.getSettings",
        "Sequence",
        "getSettings",
        fallback="Return an incomplete sequence summary and keep backend reports available.",
        aliases=("Sequence.getSettings",),
    ),
    _capability(
        "Sequence.getName",
        "Sequence",
        "getName",
        fallback="Use an empty sequence name in the returned summary.",
        aliases=("Sequence.getName",),
    ),
    _capability(
        "Sequence.getPlayerPosition",
        "Sequence",
        "getPlayerPosition",
        fallback="Mark playhead writes unverified when the position cannot be read back.",
        aliases=("Sequence.getPlayerPosition",),
    ),
    _capability(
        "Sequence.getEnd",
        "Sequence",
        "getEnd",
        fallback="Use a zero duration in the returned summary.",
        aliases=("Sequence.getEnd",),
    ),
    _capability(
        "Sequence.getVideoTrackList",
        "Sequence",
        "getVideoTrackList",
        fallback="Return no video tracks and keep sequence metadata readable.",
        aliases=("Sequence.getVideoTrackList",),
    ),
    _capability(
        "Sequence.getAudioTrackList",
        "Sequence",
        "getAudioTrackList",
        fallback="Return no audio tracks and keep sequence metadata readable.",
        aliases=("Sequence.getAudioTrackList",),
    ),
    _capability(
        "Sequence.getGuid",
        "Sequence",
        "getGuid",
        fallback="Use an empty sequence identifier.",
        aliases=("Sequence.getGuid",),
    ),
    _capability(
        "Sequence.getInPoint",
        "Sequence",
        "getInPoint",
        fallback="Do not create a range export when the original range cannot be read.",
        aliases=("Sequence.getInPoint",),
    ),
    _capability(
        "Sequence.getOutPoint",
        "Sequence",
        "getOutPoint",
        fallback="Do not create a range export when the original range cannot be read.",
        aliases=("Sequence.getOutPoint",),
    ),
    _capability(
        "Sequence.createSetInPointAction",
        "Sequence",
        "createSetInPointAction",
        sync_async="sync",
        fallback="Reject sequence-range export and keep the original range unchanged.",
        aliases=("Sequence.createSetInPointAction",),
    ),
    _capability(
        "Sequence.createSetOutPointAction",
        "Sequence",
        "createSetOutPointAction",
        sync_async="sync",
        fallback="Reject sequence-range export and keep the original range unchanged.",
        aliases=("Sequence.createSetOutPointAction",),
    ),
    _capability(
        "Sequence.createSubsequence",
        "Sequence",
        "createSubsequence",
        fallback="Report range export unavailable and retain OTIO/interchange output.",
        aliases=("Sequence.createSubsequence",),
    ),
    _capability(
        "Sequence.getSelection",
        "Sequence",
        "getSelection",
        fallback="Return an empty clip selection.",
        aliases=("Sequence.getSelection",),
    ),
    # F337 (2026-08-20) — the typed successor to `ocApplySequenceCuts`, verified
    # against the pinned @adobe/premierepro 26.3.0 typings
    # (`src/premierepro.d.ts:3203-3232`). The editor is obtained with
    # `SequenceEditor.getEditor(sequence)`, then
    # `createRemoveItemsAction(trackItemSelection, ripple, mediaType,
    # shiftOverLapping?) : Action`. Unlike the bare `sequence.rippleDelete()`
    # call — which premiere-pro-mcp #21 measured returning success while
    # mutating nothing on 26.3 — this is an undoable Action run through
    # `Project.executeTransaction`, so its effect is observable and reversible.
    # Catalogued as migration evidence, not yet consumed by either panel: the
    # cut path still runs through CEP until F252's live-host lane unblocks, so
    # these are fixture_only exactly like Sequence.setSelection below.
    _capability(
        "SequenceEditor.getEditor",
        "SequenceEditor",
        "getEditor",
        sync_async="sync",
        fallback="Report the UXP cut path unavailable and keep the CEP fallback.",
        aliases=("SequenceEditor.getEditor",),
        fixture_only=True,
    ),
    _capability(
        "SequenceEditor.createRemoveItemsAction",
        "SequenceEditor",
        "createRemoveItemsAction",
        sync_async="sync",
        fallback=(
            "Fall back to the CEP ExtendScript cut path and report the UXP "
            "capability as unavailable."
        ),
        aliases=("SequenceEditor.createRemoveItemsAction",),
        fixture_only=True,
    ),
    _capability(
        "Sequence.setSelection",
        "Sequence",
        "setSelection",
        sync_async="async",
        fallback="Do not mutate selection when the host behavior is unknown.",
        aliases=("Sequence.setSelection",),
        host_behavior=(
            {
                "minimum_host": "25.6",
                "maximum_host": "26.2",
                "sync_async": "async",
            },
            {
                "minimum_host": "26.3",
                "sync_async": "sync",
            },
        ),
        fixture_only=True,
    ),
    _capability(
        "Sequence.getMarkerList",
        "Sequence",
        "getMarkerList",
        fallback="Return no markers and keep marker-based workflows backend-only.",
        aliases=("Sequence.getMarkerList",),
    ),
    _capability(
        "Sequence.rippleDelete",
        "Sequence",
        "rippleDelete",
        fallback="Reject direct cuts and use the CEP timeline action.",
        aliases=("Sequence.rippleDelete",),
    ),
    _capability(
        "Sequence.getCaptionTrackCount",
        "Sequence",
        "getCaptionTrackCount",
        fallback="Return a read-only caption capability error.",
        aliases=("Sequence.getCaptionTrackCount",),
    ),
    _capability(
        "Sequence.getCaptionTrack",
        "Sequence",
        "getCaptionTrack",
        fallback="Return a read-only caption capability error.",
        aliases=("Sequence.getCaptionTrack",),
    ),
    _capability(
        "Sequence.setPlayerPosition",
        "Sequence",
        "setPlayerPosition",
        fallback="Try the legacy playhead alias or report no positioning API.",
        aliases=("Sequence.setPlayerPosition",),
    ),
    _capability(
        "Sequence.setPlayheadPosition",
        "Sequence",
        "setPlayheadPosition",
        fallback="Try setPlayerPosition or the Source Monitor fallback.",
        aliases=("Sequence.setPlayheadPosition",),
    ),
    _capability(
        "TrackList.getTrackCount",
        "TrackList",
        "getTrackCount",
        fallback="Return an empty track list.",
        aliases=("TrackList.getTrackCount",),
    ),
    _capability(
        "TrackList.getTrackAtIndex",
        "TrackList",
        "getTrackAtIndex",
        fallback="Skip the inaccessible track and continue the walk.",
        aliases=("TrackList.getTrackAtIndex",),
    ),
    _capability(
        "Track.getTrackItems",
        "Track",
        "getTrackItems",
        fallback="Return an empty track and preserve the sequence index contract.",
        aliases=("Track.getTrackItems", "CaptionTrack.getTrackItems"),
    ),
    _capability(
        "ProjectItem.getItems",
        "ProjectItem",
        "getItems",
        fallback="Return an empty project tree.",
        aliases=("Object.getItems",),
    ),
    _capability(
        "Object.getName",
        "ProjectItem/TrackItem/Marker",
        "getName",
        fallback="Use an empty label for the unreadable host object.",
        aliases=("Object.getName",),
    ),
    _capability(
        "Object.getMediaPath",
        "ProjectItem",
        "getMediaPath",
        fallback="Return an empty media path and mark the item unresolved.",
        aliases=("Object.getMediaPath",),
    ),
    _capability(
        "Object.getOutPoint",
        "ProjectItem/TrackItem",
        "getOutPoint",
        fallback="Use a zero duration for the unreadable object.",
        aliases=("Object.getOutPoint",),
    ),
    _capability(
        "Object.getTime",
        "Marker",
        "getTime",
        fallback="Use the marker's stored numeric time or zero.",
        aliases=("Object.getTime",),
    ),
    _capability(
        "Object.getNodeId",
        "ProjectItem/TrackItem",
        "getNodeId",
        fallback="Use an empty identifier and do not persist a host locator.",
        aliases=("Object.getNodeId",),
    ),
    _capability(
        "Object.getId",
        "ProjectItem/TrackItem/CaptionItem",
        "getId",
        fallback="Use an empty identifier and do not persist a host locator.",
        aliases=("Object.getId",),
    ),
    _capability(
        "Object.isFolder",
        "ProjectItem",
        "isFolder",
        fallback="Treat the item as a leaf and avoid recursive traversal.",
        aliases=("Object.isFolder",),
    ),
    _capability(
        "Object.setName",
        "ProjectItem",
        "setName",
        sync_async="async",
        fallback="Skip the rename and report the item as unchanged.",
        aliases=("Object.setName",),
    ),
    _capability(
        "Object.delete",
        "ProjectItem/Marker",
        "delete",
        fallback="Try remove() or the project delete-item action.",
        aliases=("Object.delete",),
    ),
    _capability(
        "Object.remove",
        "ProjectItem/Marker",
        "remove",
        fallback="Try delete() or the project delete-item action.",
        aliases=("Object.remove",),
    ),
    _capability(
        "Object.getProjectItem",
        "TrackItem",
        "getProjectItem",
        fallback="Return an unresolved clip without media metadata.",
        aliases=("Object.getProjectItem",),
    ),
    _capability(
        "Object.getComponentChain",
        "TrackItem/ProjectItem",
        "getComponentChain",
        fallback="Return an empty effect list.",
        aliases=("Object.getComponentChain",),
    ),
    _capability(
        "Object.getInPoint",
        "TrackItem/CaptionItem",
        "getInPoint",
        fallback="Use a zero start time for the unreadable item.",
        aliases=("Object.getInPoint",),
    ),
    _capability(
        "Object.getTrackIndex",
        "TrackItem",
        "getTrackIndex",
        fallback="Use track index zero for the returned selection summary.",
        aliases=("Object.getTrackIndex",),
    ),
    _capability(
        "Object.getText",
        "CaptionItem",
        "getText",
        fallback="Return an empty caption segment text.",
        aliases=("Object.getText",),
    ),
    _capability(
        "Object.getTitle",
        "CaptionItem",
        "getTitle",
        fallback="Return an empty caption segment text.",
        aliases=("Object.getTitle",),
    ),
    _capability(
        "Object.getStartTime",
        "TrackItem/CaptionItem",
        "getStartTime",
        fallback="Use the segment's zero start time.",
        aliases=("Object.getStartTime",),
    ),
    _capability(
        "Object.getStart",
        "TrackItem/CaptionItem",
        "getStart",
        fallback="Use the segment's zero start time.",
        aliases=("Object.getStart",),
    ),
    _capability(
        "Object.getEndTime",
        "TrackItem/CaptionItem",
        "getEndTime",
        fallback="Use the segment's start time as its end time.",
        aliases=("Object.getEndTime",),
    ),
    _capability(
        "Object.getEnd",
        "TrackItem/CaptionItem",
        "getEnd",
        fallback="Use the segment's start time as its end time.",
        aliases=("Object.getEnd",),
    ),
    _capability(
        "MarkerList.createMarker",
        "MarkerList",
        "createMarker",
        fallback="Keep markers staged for CEP/backend execution.",
        aliases=("MarkerList.createMarker",),
    ),
    _capability(
        "MarkerList.getFirstMarkerAtTime",
        "MarkerList",
        "getFirstMarkerAtTime",
        fallback="Create the marker without optional label/color enrichment.",
        aliases=("MarkerList.getFirstMarkerAtTime",),
    ),
    _capability(
        "MarkerList.getMarkers",
        "MarkerList",
        "getMarkers",
        fallback="Return an empty marker snapshot.",
        aliases=("MarkerList.getMarkers",),
    ),
    _capability(
        "MarkerList.getFirstMarker",
        "MarkerList",
        "getFirstMarker",
        fallback="Return an empty marker snapshot.",
        aliases=("MarkerList.getFirstMarker",),
    ),
    _capability(
        "MarkerList.getNextMarker",
        "MarkerList",
        "getNextMarker",
        fallback="Stop marker traversal at the last readable marker.",
        aliases=("MarkerList.getNextMarker",),
    ),
    _capability(
        "MarkerList.removeMarker",
        "MarkerList",
        "removeMarker",
        fallback="Try the marker object's delete/remove fallback.",
        aliases=("MarkerList.removeMarker",),
    ),
    _capability(
        "Marker.getName",
        "Marker",
        "getName",
        fallback="Use an empty marker name.",
        aliases=("Marker.getName",),
    ),
    _capability(
        "Marker.getComment",
        "Marker",
        "getComment",
        fallback="Use an empty marker comment.",
        aliases=("Marker.getComment",),
    ),
    _capability(
        "Marker.getColorIndex",
        "Marker",
        "getColorIndex",
        fallback="Use no marker color metadata.",
        aliases=("Marker.getColorIndex",),
    ),
    _capability(
        "Marker.setName",
        "Marker",
        "setName",
        fallback="Keep the default marker name.",
        aliases=("Marker.setName",),
    ),
    _capability(
        "Marker.setColorIndex",
        "Marker",
        "setColorIndex",
        fallback="Keep the host's default marker color.",
        aliases=("Marker.setColorIndex",),
    ),
    _capability(
        "ComponentChain.getComponentCount",
        "ComponentChain",
        "getComponentCount",
        sync_async="sync",
        fallback="Return an empty effect list.",
        aliases=("ComponentChain.getComponentCount",),
    ),
    _capability(
        "ComponentChain.getComponentAtIndex",
        "ComponentChain",
        "getComponentAtIndex",
        sync_async="sync",
        fallback="Skip the unreadable effect component.",
        aliases=("ComponentChain.getComponentAtIndex",),
    ),
    _capability(
        "ClipProjectItem.cast",
        "ClipProjectItem",
        "cast",
        sync_async="sync",
        fallback="Use the opaque project item object without transcript access.",
        aliases=("module.ClipProjectItem.cast",),
    ),
    _capability(
        "Transcript.querySupportedLanguages",
        "Transcript",
        "querySupportedLanguages",
        minimum_host="26.3",
        minimum_package="26.3.0",
        sync_async="sync",
        fallback="Report transcript language discovery unavailable.",
        aliases=("module.Transcript.querySupportedLanguages",),
    ),
    _capability(
        "Transcript.hasTranscript",
        "Transcript",
        "hasTranscript",
        minimum_host="26.3",
        minimum_package="26.3.0",
        sync_async="sync",
        fallback="Report no transcript without changing the project.",
        aliases=("module.Transcript.hasTranscript",),
    ),
    _capability(
        "Transcript.exportToJSON",
        "Transcript",
        "exportToJSON",
        minimum_host="26.3",
        minimum_package="26.3.0",
        fallback="Return transcript state without JSON export.",
        aliases=("module.Transcript.exportToJSON",),
    ),
    _capability(
        "ObjectMaskUtils.hasObjectMask",
        "ObjectMaskUtils",
        "hasObjectMask",
        sync_async="sync",
        fallback="Report object-mask inspection unavailable.",
        aliases=("module.ObjectMaskUtils.hasObjectMask",),
    ),
    _capability(
        "TickTime.createWithSeconds",
        "TickTime",
        "createWithSeconds",
        sync_async="sync",
        fallback="Use a plain seconds/ticks compatibility value.",
        aliases=("module.TickTime.createWithSeconds",),
    ),
    _capability(
        "TickTime.createWithTicks",
        "TickTime",
        "createWithTicks",
        sync_async="sync",
        fallback="Use a plain seconds/ticks compatibility value.",
        aliases=("module.TickTime.createWithTicks",),
    ),
    _capability(
        "SourceMonitor.setPosition",
        "SourceMonitor",
        "setPosition",
        fallback="Report no supported playhead positioning API.",
        aliases=("module.SourceMonitor.setPosition",),
    ),
    _capability(
        "EncoderManager.getManager",
        "EncoderManager",
        "getManager",
        sync_async="sync",
        fallback="Report encoder export unavailable.",
        aliases=("module.EncoderManager.getManager",),
    ),
    _capability(
        "EncoderManager.launchEncoder",
        "EncoderManager",
        "launchEncoder",
        fallback="Leave Adobe Media Encoder closed and report the queue failure.",
        aliases=("EncoderManager.launchEncoder",),
    ),
    _capability(
        "EncoderManager.exportSequence",
        "EncoderManager",
        "exportSequence",
        fallback="Report export unavailable without deleting source media.",
        aliases=("EncoderManager.exportSequence",),
    ),
    _capability(
        "EncoderManager.startBatchEncode",
        "EncoderManager",
        "startBatchEncode",
        fallback="Leave the queue pending for manual AME start.",
        aliases=("EncoderManager.startBatchEncode",),
    ),
    _capability(
        "EncoderManager.isAMEInstalled",
        "EncoderManager",
        "isAMEInstalled",
        sync_async="sync",
        fallback="Report Adobe Media Encoder as unavailable.",
        aliases=("EncoderManager.isAMEInstalled",),
    ),
    _capability(
        "ProjectConverter.exportAAF",
        "ProjectConverter",
        "exportAAF",
        fallback="Keep the sequence available for OTIO/XML export instead.",
        aliases=("module.ProjectConverter.exportAAF",),
    ),
    _capability(
        "AAFExportOptions.constructor",
        "AAFExportOptions",
        "constructor",
        sync_async="sync",
        fallback="Use default AAF options or report AAF export unavailable.",
        aliases=("module.AAFExportOptions",),
    ),
    _capability(
        "AAFExportOptions.setMixdownVideo",
        "AAFExportOptions",
        "setMixdownVideo",
        sync_async="sync",
        fallback="Use the host default for mixdown video.",
        aliases=("AAFExportOptions.setMixdownVideo",),
    ),
    _capability(
        "AAFExportOptions.setExplodeToMono",
        "AAFExportOptions",
        "setExplodeToMono",
        sync_async="sync",
        fallback="Use the host default for channel layout.",
        aliases=("AAFExportOptions.setExplodeToMono",),
    ),
    _capability(
        "AAFExportOptions.setEmbedAudio",
        "AAFExportOptions",
        "setEmbedAudio",
        sync_async="sync",
        fallback="Use the host default for embedded audio.",
        aliases=("AAFExportOptions.setEmbedAudio",),
    ),
    _capability(
        "AAFExportOptions.setTrimSources",
        "AAFExportOptions",
        "setTrimSources",
        sync_async="sync",
        fallback="Use the host default for source trimming.",
        aliases=("AAFExportOptions.setTrimSources",),
    ),
    _capability(
        "AAFExportOptions.setRenderAudioEffects",
        "AAFExportOptions",
        "setRenderAudioEffects",
        sync_async="sync",
        fallback="Use the host default for rendered audio effects.",
        aliases=("AAFExportOptions.setRenderAudioEffects",),
    ),
    _capability(
        "AAFExportOptions.setInterleaveWithoutEffects",
        "AAFExportOptions",
        "setInterleaveWithoutEffects",
        sync_async="sync",
        fallback="Use the host default for interleaving.",
        aliases=("AAFExportOptions.setInterleaveWithoutEffects",),
    ),
    _capability(
        "AAFExportOptions.setPreserveParentFolder",
        "AAFExportOptions",
        "setPreserveParentFolder",
        sync_async="sync",
        fallback="Use the host default for parent-folder preservation.",
        aliases=("AAFExportOptions.setPreserveParentFolder",),
    ),
    _capability(
        "AAFExportOptions.setSampleRate",
        "AAFExportOptions",
        "setSampleRate",
        sync_async="sync",
        fallback="Use the host default sample rate.",
        aliases=("AAFExportOptions.setSampleRate",),
    ),
    _capability(
        "AAFExportOptions.setBitsPerSample",
        "AAFExportOptions",
        "setBitsPerSample",
        sync_async="sync",
        fallback="Use the host default bit depth.",
        aliases=("AAFExportOptions.setBitsPerSample",),
    ),
    _capability(
        "AAFExportOptions.setHandleFrames",
        "AAFExportOptions",
        "setHandleFrames",
        sync_async="sync",
        fallback="Use the host default handle length.",
        aliases=("AAFExportOptions.setHandleFrames",),
    ),
    _capability(
        "AAFExportOptions.setAudioFileFormat",
        "AAFExportOptions",
        "setAudioFileFormat",
        sync_async="sync",
        fallback="Use the host default AAF audio format.",
        aliases=("AAFExportOptions.setAudioFileFormat",),
    ),
    _capability(
        "AAFExportOptions.setVideoMixdownPresetPath",
        "AAFExportOptions",
        "setVideoMixdownPresetPath",
        sync_async="sync",
        fallback="Use the host default video mixdown preset.",
        aliases=("AAFExportOptions.setVideoMixdownPresetPath",),
    ),
    _capability(
        "uxp.document.theme.getCurrent",
        "UXP document.theme",
        "getCurrent",
        sync_async="sync",
        fallback="Use the darkest theme class.",
        aliases=("uxp.document.theme.getCurrent",),
    ),
    _capability(
        "uxp.document.theme.onUpdated.addListener",
        "UXP document.theme.onUpdated",
        "addListener",
        sync_async="sync",
        fallback="Apply the initial theme without live theme updates.",
        aliases=("uxp.document.theme.onUpdated.addListener",),
    ),
    _capability(
        "uxp.document.theme.onUpdated.removeListener",
        "UXP document.theme.onUpdated",
        "removeListener",
        sync_async="sync",
        fallback="Leave the listener lifecycle to the host panel teardown.",
        aliases=("uxp.document.theme.onUpdated.removeListener",),
    ),
)

_CATALOGUE_BY_ALIAS = {
    alias: capability
    for capability in API_CATALOGUE
    for alias in capability["aliases"]
}

_CHAIN_RE = re.compile(
    r"\b(?P<chain>(?:ppro|seq|sequence|proj|project|context|markerList|result|"
    r"track|list|item|child|parent|target|compoundAction|manager|options|"
    r"created|marker|documentRef|themeApi|theme|event)"
    r"(?:\?\.|\.)[A-Za-z_$][\w$]*"
    r"(?:(?:\?\.|\.)[A-Za-z_$][\w$]*)*)"
)
_IMPORT_RE = re.compile(r"\b(?:import|require)\s*\(\s*['\"]premierepro['\"]\s*\)")
_PROPERTY_METHODS = {
    "AAFExportOptions",
    "createBin",
    "createSetInPointAction",
    "createSetOutPointAction",
    "createSubsequence",
    "deleteItem",
    "executeTransaction",
    "exportSequence",
    "exportToJSON",
    "getAttribute",
    "getManager",
    "hasTranscript",
    "isAMEInstalled",
    "launchEncoder",
    "lockedAccess",
    "remove",
    "setAudioFileFormat",
    "setBitsPerSample",
    "setColorIndex",
    "setHandleFrames",
    "setInterleaveWithoutEffects",
    "setMixdownVideo",
    "setName",
    "setPlayheadPosition",
    "setPlayerPosition",
    "setPreserveParentFolder",
    "setRenderAudioEffects",
    "setSampleRate",
    "setTrimSources",
    "setVideoMixdownPresetPath",
    "startBatchEncode",
}
_IGNORED_METHODS = {
    "add",
    "appendChild",
    "catch",
    "click",
    "dispatchEvent",
    "filter",
    "focus",
    "getAttribute",
    "forEach",
    "includes",
    "join",
    "map",
    "pop",
    "push",
    "querySelectorAll",
    "remove",
    "slice",
    "sort",
    "split",
    "trim",
}


def _relative(path: Path, root: Path = REPO_ROOT) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _strip_comments(line: str) -> str:
    """Remove the comment portion while preserving the original line number."""
    return re.sub(r"//.*$", "", line)


def _canonical_chain(chain: str) -> str:
    parts = [part for part in re.split(r"\?\.|\.", chain) if part]
    if not parts:
        return ""
    parts = [part.rstrip("?") for part in parts]
    member = parts[-1]
    if parts[0] == "ppro":
        return "module." + ".".join(parts[1:])
    if parts[0] in {"documentRef", "themeApi", "theme", "event"}:
        if "onUpdated" in parts:
            return "uxp.document.theme.onUpdated." + member
        if "theme" in parts:
            return "uxp.document.theme." + member
    if parts[:2] == ["context", "proj"]:
        return "Project." + member
    if parts[:2] == ["result", "markerList"]:
        return "MarkerList." + member
    if parts[0] in {"seq", "sequence"}:
        return "Sequence." + member
    if parts[0] in {"proj", "project"}:
        return "Project." + member
    if parts[0] == "markerList":
        return "MarkerList." + member
    if parts[0] == "track":
        return "Track." + member
    if parts[0] == "list" and member in {"getTrackCount", "getTrackAtIndex"}:
        return "TrackList." + member
    if parts[0] == "chain":
        return "ComponentChain." + member
    if parts[0] == "compoundAction":
        return "CompoundAction." + member
    if parts[0] == "manager":
        return "EncoderManager." + member
    if parts[0] == "options":
        return "AAFExportOptions." + member
    if member in {"getComment", "getColorIndex", "setColorIndex", "setName"} and parts[0] in {"created", "marker"}:
        return "Marker." + member
    if parts[0] in {"item", "child", "parent", "target", "created", "marker"}:
        return "Object." + member
    return ""


def _candidate_is_used(line: str, match: re.Match[str]) -> bool:
    after = line[match.end():]
    stripped = after.lstrip()
    if stripped.startswith("?."):
        stripped = stripped[2:].lstrip()
    if stripped.startswith("("):
        return True
    canonical = _canonical_chain(match.group("chain"))
    member = canonical.rsplit(".", 1)[-1] if canonical else ""
    return member in _PROPERTY_METHODS


def _source_files(root: Path) -> tuple[Path, ...]:
    root = root.resolve()
    return tuple(
        path
        for path in (root / relative for relative in SOURCE_RELATIVE_PATHS)
        if path.is_file()
    )


def scan_sources(root: Path = REPO_ROOT) -> dict:
    """Return declared uses and precise undeclared-capability diagnostics."""
    uses: dict[str, list[dict]] = {capability["id"]: [] for capability in API_CATALOGUE}
    undeclared: list[dict] = []
    seen_uses: set[tuple[str, str, int]] = set()

    for path in _source_files(root):
        relative = _relative(path, root)
        for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = _strip_comments(raw_line)
            if _IMPORT_RE.search(line):
                capability = _CATALOGUE_BY_ALIAS["module.import"]
                key = (capability["id"], relative, line_number)
                if key not in seen_uses:
                    uses[capability["id"]].append(
                        {"file": relative, "line": line_number, "symbol": "module.import"}
                    )
                    seen_uses.add(key)

            for match in _CHAIN_RE.finditer(line):
                if not _candidate_is_used(line, match):
                    continue
                canonical = _canonical_chain(match.group("chain"))
                if not canonical:
                    continue
                capability = _CATALOGUE_BY_ALIAS.get(canonical)
                if capability is None:
                    if canonical.rsplit(".", 1)[-1] in _IGNORED_METHODS:
                        continue
                    undeclared.append(
                        {
                            "file": relative,
                            "line": line_number,
                            "symbol": canonical,
                            "message": f"undeclared UXP capability {canonical}",
                        }
                    )
                    continue
                key = (capability["id"], relative, line_number)
                if key in seen_uses:
                    continue
                uses[capability["id"]].append(
                    {"file": relative, "line": line_number, "symbol": canonical}
                )
                seen_uses.add(key)

    return {
        "uses": uses,
        "undeclared_capabilities": undeclared,
        "source_files": [_relative(path, root) for path in _source_files(root)],
    }


def _load_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _pinned_package_version() -> str:
    payload = _load_json(PACKAGE_JSON_PATH)
    dependencies = payload.get("devDependencies") or {}
    version = dependencies.get(PACKAGE, "")
    return str(version).lstrip("^~= ")


def _package_drift(pinned_version: str) -> dict:
    snapshot_path = REPO_ROOT / "opencut" / "_generated" / "adobe_premierepro_versions.json"
    snapshot = _load_json(snapshot_path)
    latest = str((snapshot.get("tracked_dist_tags") or {}).get("latest") or "")
    tracked = [str(value) for value in (snapshot.get("tracked_versions") or [])]
    if not pinned_version or not latest:
        status = "unknown"
    elif pinned_version == latest or pinned_version in tracked:
        status = "in_sync"
    else:
        status = "drift"
    return {
        "status": status,
        "pinned_package_version": pinned_version,
        "snapshot_latest": latest,
        "snapshot_path": _relative(snapshot_path),
        "requires_live_premiere": False,
    }


def _manifest_host_minimum() -> str:
    manifest = _load_json(UXP_MANIFEST_PATH)
    hosts = manifest.get("host") or []
    if hosts and isinstance(hosts[0], dict):
        return str(hosts[0].get("minVersion") or "25.6")
    return "25.6"


def _materialise_capabilities(scan: dict, pinned_version: str) -> list[dict]:
    capabilities = []
    for declaration in API_CATALOGUE:
        capability = dict(declaration)
        capability.pop("aliases", None)
        capability["package"] = PACKAGE
        capability["package_version"] = pinned_version
        capability["used"] = bool(scan["uses"].get(declaration["id"]))
        capability["source_refs"] = sorted(
            scan["uses"].get(declaration["id"], []),
            key=lambda ref: (ref["file"], ref["line"], ref["symbol"]),
        )
        capabilities.append(capability)
    return capabilities


def build_manifest(root: Path = REPO_ROOT) -> dict:
    """Build the JSON-safe compatibility manifest without a Premiere host."""
    scan = scan_sources(root)
    pinned_version = _pinned_package_version()
    capabilities = _materialise_capabilities(scan, pinned_version)
    used_count = sum(1 for capability in capabilities if capability["used"])
    return {
        "manifest_version": MANIFEST_VERSION,
        "package": PACKAGE,
        "package_version": pinned_version,
        "api_version": API_VERSION,
        "minimum_host": _manifest_host_minimum(),
        "host_bands": list(HOST_BANDS),
        "host_fixtures": list(HOST_FIXTURES),
        "source_files": scan["source_files"],
        "capabilities": capabilities,
        "capability_count": len(capabilities),
        "used_capability_count": used_count,
        "diagnostics": {
            "undeclared_capabilities": scan["undeclared_capabilities"],
            "live_host_checks": "blocked_separate",
        },
        "package_drift": _package_drift(pinned_version),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def load_manifest(path: Path = MANIFEST_PATH) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def write_manifest(manifest: dict, path: Path = MANIFEST_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _comparable(manifest: Optional[dict]) -> dict:
    payload = dict(manifest or {})
    payload.pop("generated_at", None)
    return payload


def diff_manifests(committed: Optional[dict], live: dict) -> dict:
    """Return a small machine-readable diff suitable for release output."""
    if committed is None:
        return {
            "changed": True,
            "fields": {"manifest": {"from": "<absent>", "to": "present"}},
        }
    if _comparable(committed) == _comparable(live):
        return {"changed": False, "fields": {}}
    fields = {}
    for key in sorted(set(_comparable(committed)) | set(_comparable(live))):
        committed_value = _comparable(committed).get(key)
        live_value = _comparable(live).get(key)
        if committed_value != live_value:
            fields[key] = {"from": committed_value, "to": live_value}
    return {"changed": True, "fields": fields}


def _version_key(value: str) -> tuple[int, ...]:
    parts = []
    for chunk in str(value).split("."):
        match = re.match(r"\d+", chunk)
        parts.append(int(match.group()) if match else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def capability_for_host(manifest: dict, capability_id: str, host_version: str) -> Optional[dict]:
    """Resolve one capability's behavior for a host-band fixture."""
    capability = next(
        (row for row in manifest.get("capabilities", []) if row.get("id") == capability_id),
        None,
    )
    if capability is None:
        return None
    host = _version_key(host_version)
    minimum = _version_key(str(capability.get("minimum_host") or "0"))
    if host < minimum:
        return {
            "capability": capability_id,
            "host": host_version,
            "supported": False,
            "sync_async": None,
            "fallback": capability.get("fallback", ""),
        }
    selected = None
    for behavior in capability.get("host_behavior", []):
        if host < _version_key(str(behavior.get("minimum_host") or "0")):
            continue
        maximum = behavior.get("maximum_host")
        if maximum is not None and host > _version_key(str(maximum)):
            continue
        selected = behavior
    selected = selected or {"sync_async": capability.get("sync_async")}
    return {
        "capability": capability_id,
        "host": host_version,
        "supported": True,
        "sync_async": selected.get("sync_async"),
        "fallback": capability.get("fallback", ""),
    }


def validate_host_fixtures(manifest: dict) -> list[str]:
    errors = []
    for fixture in manifest.get("host_fixtures", []):
        host = str(fixture.get("host") or "")
        for assertion in fixture.get("assertions", []):
            capability_id = str(assertion.get("capability") or "")
            result = capability_for_host(manifest, capability_id, host)
            if result is None:
                errors.append(f"fixture {host}: unknown capability {capability_id}")
                continue
            expected = assertion.get("sync_async")
            if not result["supported"] or result["sync_async"] != expected:
                errors.append(
                    f"fixture {host}: {capability_id} expected {expected}, "
                    f"got {result['sync_async']}"
                )
    return errors


def _format_diagnostics(manifest: dict) -> str:
    diagnostics = manifest.get("diagnostics") or {}
    lines = []
    for issue in diagnostics.get("undeclared_capabilities") or []:
        lines.append(
            f"{issue.get('file')}:{issue.get('line')}: {issue.get('message')}"
        )
    for error in validate_host_fixtures(manifest):
        lines.append(f"fixture: {error}")
    return "\n".join(lines)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate or check static Adobe Premiere UXP compatibility metadata."
    )
    parser.add_argument("--check", action="store_true", help="fail when metadata or source declarations drift")
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    parser.add_argument("--output", type=Path, default=MANIFEST_PATH, help="manifest path")
    args = parser.parse_args(list(argv) if argv is not None else None)

    live = build_manifest()
    diagnostics = _format_diagnostics(live)
    if args.check:
        committed = load_manifest(args.output)
        diff = diff_manifests(committed, live)
        payload = {
            "manifest": live,
            "committed_present": committed is not None,
            "drift": diff,
            "diagnostics": diagnostics.splitlines() if diagnostics else [],
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            if diagnostics:
                print("[adobe-uxp] FAIL - compatibility diagnostics:")
                print(diagnostics)
            elif diff["changed"]:
                print("[adobe-uxp] FAIL - generated compatibility metadata drift")
                for field, change in diff["fields"].items():
                    print(f"  {field}: {change.get('from')!r} -> {change.get('to')!r}")
            else:
                print(
                    "[adobe-uxp] OK - "
                    f"{live['used_capability_count']} used capabilities across "
                    f"{len(live['host_bands'])} host bands; live Premiere not required"
                )
        if diagnostics or diff["changed"]:
            return 1
        return 2 if live["package_drift"]["status"] == "drift" else 0

    write_manifest(live, args.output)
    if args.json:
        print(json.dumps({"path": str(args.output), "manifest": live}, indent=2, sort_keys=True))
    else:
        print(
            f"Wrote {args.output} ({live['used_capability_count']} used capabilities, "
            f"package drift={live['package_drift']['status']})."
        )
    return 1 if diagnostics else 0


if __name__ == "__main__":
    sys.exit(main())
