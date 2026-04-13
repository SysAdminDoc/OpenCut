"""
OpenCut After Effects Extension Backend (9.3)

Backend support for After Effects CEP/UXP panel integration.
Provides AE-specific operations (comp-aware processing, layer operations),
CSXS manifest generation, and basic ExtendScript scaffolding.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("opencut")

# Extension identity
AE_EXTENSION_ID = "com.opencut.ae"
AE_EXTENSION_VERSION = "1.0.0"
AE_EXTENSION_NAME = "OpenCut for After Effects"
AE_MIN_AE_VERSION = "16.0"   # CC 2019+
AE_MAX_AE_VERSION = "99.9"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AECompInfo:
    """Metadata for an After Effects composition."""
    name: str
    width: int
    height: int
    fps: float
    duration: float
    num_layers: int


@dataclass
class AELayerInfo:
    """Metadata for a single layer within a composition."""
    index: int
    name: str
    type: str          # e.g. "footage", "solid", "shape", "text", "camera", "light", "null", "adjustment"
    in_point: float
    out_point: float


@dataclass
class AEProject:
    """High-level After Effects project metadata."""
    comps: List[AECompInfo] = field(default_factory=list)
    active_comp: Optional[AECompInfo] = None
    project_path: str = ""


# ---------------------------------------------------------------------------
# AE-specific operations subset
# ---------------------------------------------------------------------------

_AE_SUPPORTED_OPS = [
    {
        "id": "bg_removal",
        "name": "Background Removal",
        "description": "Remove or replace layer background using AI",
        "ae_context": "Works on selected footage layer in the active comp",
    },
    {
        "id": "upscale",
        "name": "AI Upscale",
        "description": "Upscale footage layer resolution",
        "ae_context": "Replaces source footage with upscaled version",
    },
    {
        "id": "style_transfer",
        "name": "Style Transfer",
        "description": "Apply artistic style to footage layer",
        "ae_context": "Creates a new layer with styled footage",
    },
    {
        "id": "object_removal",
        "name": "Object Removal",
        "description": "Remove unwanted objects from footage",
        "ae_context": "Processes selected footage layer in-place",
    },
    {
        "id": "depth_effects",
        "name": "Depth Effects",
        "description": "Generate depth maps and apply depth-based effects",
        "ae_context": "Adds depth map as auxiliary layer in comp",
    },
    {
        "id": "denoise",
        "name": "Video Denoise",
        "description": "Remove noise from footage layer",
        "ae_context": "Processes selected footage layer with AI denoising",
    },
    {
        "id": "face_enhance",
        "name": "Face Enhancement",
        "description": "Enhance and restore faces in footage",
        "ae_context": "Processes faces in selected footage layer",
    },
]


def ae_supported_operations() -> List[Dict]:
    """Return the subset of OpenCut operations relevant to After Effects."""
    return list(_AE_SUPPORTED_OPS)


# ---------------------------------------------------------------------------
# Project / comp parsing
# ---------------------------------------------------------------------------

def get_ae_project_info(project_data: Dict) -> AEProject:
    """Parse After Effects project metadata from a JSON structure.

    Expected ``project_data`` keys:
        - ``project_path`` (str): file system path to the .aep
        - ``comps`` (list[dict]): composition metadata
        - ``active_comp`` (dict | None): currently-active comp
    """
    comps = []
    for c in project_data.get("comps", []):
        comps.append(AECompInfo(
            name=c.get("name", "Untitled"),
            width=int(c.get("width", 1920)),
            height=int(c.get("height", 1080)),
            fps=float(c.get("fps", 29.97)),
            duration=float(c.get("duration", 10.0)),
            num_layers=int(c.get("num_layers", 0)),
        ))

    active = None
    ac = project_data.get("active_comp")
    if ac:
        active = AECompInfo(
            name=ac.get("name", "Untitled"),
            width=int(ac.get("width", 1920)),
            height=int(ac.get("height", 1080)),
            fps=float(ac.get("fps", 29.97)),
            duration=float(ac.get("duration", 10.0)),
            num_layers=int(ac.get("num_layers", 0)),
        )

    return AEProject(
        comps=comps,
        active_comp=active,
        project_path=project_data.get("project_path", ""),
    )


def get_comp_info(comp_data: Dict) -> AECompInfo:
    """Parse a single composition's details from JSON."""
    return AECompInfo(
        name=comp_data.get("name", "Untitled"),
        width=int(comp_data.get("width", 1920)),
        height=int(comp_data.get("height", 1080)),
        fps=float(comp_data.get("fps", 29.97)),
        duration=float(comp_data.get("duration", 10.0)),
        num_layers=int(comp_data.get("num_layers", 0)),
    )


# ---------------------------------------------------------------------------
# CEP manifest generation
# ---------------------------------------------------------------------------

def generate_ae_manifest() -> str:
    """Generate a CSXS/CEP manifest XML string targeting After Effects.

    Returns a complete ``manifest.xml`` suitable for placing in the
    extension's ``CSXS/`` directory.
    """
    manifest = f"""<?xml version="1.0" encoding="UTF-8"?>
<ExtensionManifest Version="7.0" ExtensionBundleId="{AE_EXTENSION_ID}"
                   ExtensionBundleVersion="{AE_EXTENSION_VERSION}"
                   ExtensionBundleName="{AE_EXTENSION_NAME}">
    <ExtensionList>
        <Extension Id="{AE_EXTENSION_ID}.panel" Version="{AE_EXTENSION_VERSION}"/>
    </ExtensionList>
    <ExecutionEnvironment>
        <HostList>
            <Host Name="AEFT" Version="[{AE_MIN_AE_VERSION},{AE_MAX_AE_VERSION}]"/>
        </HostList>
        <LocaleList>
            <Locale Code="All"/>
        </LocaleList>
        <RequiredRuntimeList>
            <RequiredRuntime Name="CSXS" Version="9.0"/>
        </RequiredRuntimeList>
    </ExecutionEnvironment>
    <DispatchInfoList>
        <Extension Id="{AE_EXTENSION_ID}.panel">
            <DispatchInfo>
                <Resources>
                    <MainPath>./index.html</MainPath>
                    <ScriptPath>./jsx/host.jsx</ScriptPath>
                    <CEFCommandLine>
                        <Parameter>--enable-nodejs</Parameter>
                        <Parameter>--mixed-context</Parameter>
                    </CEFCommandLine>
                </Resources>
                <Lifecycle>
                    <AutoVisible>true</AutoVisible>
                </Lifecycle>
                <UI>
                    <Type>Panel</Type>
                    <Menu>{AE_EXTENSION_NAME}</Menu>
                    <Geometry>
                        <Size>
                            <Height>600</Height>
                            <Width>400</Width>
                        </Size>
                        <MinSize>
                            <Height>400</Height>
                            <Width>300</Width>
                        </MinSize>
                    </Geometry>
                    <Icons/>
                </UI>
            </DispatchInfo>
        </Extension>
    </DispatchInfoList>
</ExtensionManifest>"""
    return manifest


# ---------------------------------------------------------------------------
# ExtendScript generation
# ---------------------------------------------------------------------------

def generate_ae_extendscript() -> str:
    """Generate basic ExtendScript (.jsx) for AE comp and layer access.

    The returned script provides helper functions that the CEP panel can
    call via ``csInterface.evalScript()`` to gather comp/layer metadata
    and send it to the OpenCut backend.
    """
    return r"""/*
 * OpenCut ExtendScript for After Effects
 * Provides helper functions for comp/layer access.
 */

function getProjectInfo() {
    var info = {
        project_path: app.project.file ? app.project.file.fsName : "",
        comps: [],
        active_comp: null
    };
    for (var i = 1; i <= app.project.numItems; i++) {
        var item = app.project.item(i);
        if (item instanceof CompItem) {
            var comp = {
                name: item.name,
                width: item.width,
                height: item.height,
                fps: item.frameRate,
                duration: item.duration,
                num_layers: item.numLayers
            };
            info.comps.push(comp);
        }
    }
    if (app.project.activeItem && app.project.activeItem instanceof CompItem) {
        var ac = app.project.activeItem;
        info.active_comp = {
            name: ac.name,
            width: ac.width,
            height: ac.height,
            fps: ac.frameRate,
            duration: ac.duration,
            num_layers: ac.numLayers
        };
    }
    return JSON.stringify(info);
}

function getActiveCompLayers() {
    var comp = app.project.activeItem;
    if (!comp || !(comp instanceof CompItem)) {
        return JSON.stringify({error: "No active composition"});
    }
    var layers = [];
    for (var i = 1; i <= comp.numLayers; i++) {
        var layer = comp.layer(i);
        var type = "unknown";
        if (layer instanceof AVLayer) type = "footage";
        if (layer instanceof ShapeLayer) type = "shape";
        if (layer instanceof TextLayer) type = "text";
        if (layer instanceof CameraLayer) type = "camera";
        if (layer instanceof LightLayer) type = "light";
        if (layer.nullLayer) type = "null";
        if (layer.adjustmentLayer) type = "adjustment";
        layers.push({
            index: i,
            name: layer.name,
            type: type,
            in_point: layer.inPoint,
            out_point: layer.outPoint
        });
    }
    return JSON.stringify({comp_name: comp.name, layers: layers});
}

function exportFrameForProcessing(layerIndex, time, outputPath) {
    var comp = app.project.activeItem;
    if (!comp || !(comp instanceof CompItem)) {
        return JSON.stringify({error: "No active composition"});
    }
    // Save a frame for OpenCut processing
    var rqItem = app.project.renderQueue.items.add(comp);
    var om = rqItem.outputModule(1);
    om.file = new File(outputPath);
    om.applyTemplate("TIFF Sequence");
    rqItem.timeSpanStart = time;
    rqItem.timeSpanDuration = comp.frameDuration;
    return JSON.stringify({status: "queued", output: outputPath});
}
"""
