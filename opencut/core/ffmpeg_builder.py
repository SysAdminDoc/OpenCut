"""
Custom FFmpeg Filter Chain Builder.

Build complex filter chains as JSON node graphs, validate them,
generate filter_complex strings, and preview results.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from opencut.helpers import OPENCUT_DIR, FFmpegCmd, run_ffmpeg

logger = logging.getLogger("opencut")

PRESETS_DIR = os.path.join(OPENCUT_DIR, "filter_presets")


@dataclass
class FilterNode:
    """A single node in a filter graph."""
    node_id: str
    filter_name: str
    params: Dict[str, Any] = field(default_factory=dict)
    inputs: List[str] = field(default_factory=list)    # input pad labels
    outputs: List[str] = field(default_factory=list)   # output pad labels


@dataclass
class FilterConnection:
    """A connection between two nodes in the filter graph."""
    from_node: str
    from_pad: str
    to_node: str
    to_pad: str


def _node_to_filter_str(node: FilterNode) -> str:
    """Convert a FilterNode to an FFmpeg filter expression."""
    params_str = ""
    if node.params:
        parts = []
        for k, v in node.params.items():
            if isinstance(v, bool):
                parts.append(f"{k}={1 if v else 0}")
            else:
                parts.append(f"{k}={v}")
        params_str = ":" + ":".join(parts)

    input_pads = "".join(f"[{i}]" for i in node.inputs) if node.inputs else ""
    output_pads = "".join(f"[{o}]" for o in node.outputs) if node.outputs else ""

    return f"{input_pads}{node.filter_name}{params_str}{output_pads}"


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def build_filter_chain(
    nodes: List[Dict],
    connections: Optional[List[Dict]] = None,
    on_progress: Optional[Callable] = None,
) -> str:
    """Build a filter_complex string from a node graph.

    Args:
        nodes: List of dicts with node_id, filter_name, params, inputs, outputs.
        connections: Optional list of connection dicts (from_node, from_pad,
                     to_node, to_pad). If None, nodes are chained linearly.

    Returns:
        FFmpeg filter_complex string.
    """
    if not nodes:
        raise ValueError("At least one filter node is required")

    if on_progress:
        on_progress(20, "Parsing filter nodes")

    parsed_nodes = []
    for n in nodes:
        parsed_nodes.append(FilterNode(
            node_id=n.get("node_id", f"n{len(parsed_nodes)}"),
            filter_name=n["filter_name"],
            params=n.get("params", {}),
            inputs=n.get("inputs", []),
            outputs=n.get("outputs", []),
        ))

    if connections:
        # Apply connections to set node inputs/outputs
        conn_map: Dict[str, List[str]] = {}
        for c in connections:
            label = f"{c['from_node']}_{c['from_pad']}"
            conn_map.setdefault(c["to_node"], []).append(label)
            # Ensure output pad exists on source node
            for pn in parsed_nodes:
                if pn.node_id == c["from_node"] and label not in pn.outputs:
                    pn.outputs.append(label)

        for pn in parsed_nodes:
            if pn.node_id in conn_map:
                pn.inputs = conn_map[pn.node_id]

    if on_progress:
        on_progress(60, "Building filter chain")

    filter_parts = [_node_to_filter_str(n) for n in parsed_nodes]
    chain = ";".join(filter_parts)

    if on_progress:
        on_progress(100, "Filter chain built")

    logger.info("Built filter chain with %d nodes", len(parsed_nodes))
    return chain


def validate_filter_graph(
    graph: Dict,
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Validate a filter graph for correctness.

    Args:
        graph: Dict with 'nodes' and optional 'connections' keys.

    Returns:
        Dict with 'valid' bool, 'errors' list, and 'warnings' list.
    """
    errors = []
    warnings = []

    if on_progress:
        on_progress(20, "Validating graph structure")

    nodes = graph.get("nodes", [])
    connections = graph.get("connections", [])

    if not nodes:
        errors.append("Graph must contain at least one node")
        return {"valid": False, "errors": errors, "warnings": warnings}

    # Check for required fields
    node_ids = set()
    for i, n in enumerate(nodes):
        if "filter_name" not in n:
            errors.append(f"Node {i} missing 'filter_name'")
        nid = n.get("node_id", "")
        if nid in node_ids:
            errors.append(f"Duplicate node_id: {nid}")
        node_ids.add(nid)

    if on_progress:
        on_progress(50, "Checking connections")

    # Validate connections reference existing nodes
    for c in connections:
        if c.get("from_node") not in node_ids:
            errors.append(f"Connection references unknown source node: {c.get('from_node')}")
        if c.get("to_node") not in node_ids:
            errors.append(f"Connection references unknown target node: {c.get('to_node')}")

    # Check for known FFmpeg filters
    known_filters = {
        "scale", "crop", "overlay", "drawtext", "colorbalance",
        "curves", "eq", "hue", "lut3d", "pad", "fps", "setpts",
        "trim", "split", "concat", "blend", "chromakey", "colorkey",
        "vflip", "hflip", "rotate", "transpose", "unsharp", "boxblur",
        "gblur", "deband", "deflicker", "denoise", "noise",
        "fade", "zoompan", "format", "null", "anull",
        "volume", "atempo", "aresample", "amix", "adelay",
        "aecho", "acompressor", "equalizer", "highpass", "lowpass",
    }
    for n in nodes:
        fname = n.get("filter_name", "")
        if fname and fname not in known_filters:
            warnings.append(f"Unknown filter '{fname}' - may still work if FFmpeg supports it")

    if on_progress:
        on_progress(100, "Validation complete")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "node_count": len(nodes),
        "connection_count": len(connections),
    }


def preview_filter(
    video_path: str,
    filter_chain: str,
    timestamp: float = 0.0,
    output_path: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> str:
    """Preview a filter chain by extracting a single frame.

    Args:
        video_path: Path to the source video.
        filter_chain: FFmpeg filter_complex string.
        timestamp: Time position to extract (seconds).
        output_path: Output image path (auto-generated if None).

    Returns:
        Path to the preview image.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if on_progress:
        on_progress(20, "Preparing preview")

    if output_path is None:
        import tempfile
        output_path = os.path.join(
            tempfile.gettempdir(),
            f"filter_preview_{os.getpid()}.jpg"
        )

    cmd = (
        FFmpegCmd()
        .pre_input("-ss", str(timestamp))
        .input(video_path)
        .filter_complex(filter_chain, maps=["[out]"] if "[out]" in filter_chain else None)
        .frames(1)
        .output(output_path)
        .build()
    )

    if on_progress:
        on_progress(60, "Rendering preview frame")

    run_ffmpeg(cmd, timeout=30)

    if on_progress:
        on_progress(100, "Preview generated")

    return output_path


def save_filter_preset(
    chain: str,
    name: str,
    description: str = "",
    nodes: Optional[List[Dict]] = None,
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Save a filter chain as a reusable preset.

    Args:
        chain: The filter_complex string.
        name: Preset name.
        description: Optional description.
        nodes: Optional original node graph for editing.

    Returns:
        Dict with preset_id and saved path.
    """
    if not name or not name.strip():
        raise ValueError("Preset name cannot be empty")

    if on_progress:
        on_progress(30, "Saving preset")

    os.makedirs(PRESETS_DIR, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    preset_path = os.path.join(PRESETS_DIR, f"{safe_name}.json")

    preset_data = {
        "name": name.strip(),
        "description": description,
        "chain": chain,
        "nodes": nodes,
    }

    with open(preset_path, "w", encoding="utf-8") as fh:
        json.dump(preset_data, fh, indent=2)

    if on_progress:
        on_progress(100, "Preset saved")

    logger.info("Saved filter preset '%s' to %s", name, preset_path)
    return {
        "preset_id": safe_name,
        "path": preset_path,
        "name": name.strip(),
    }


def load_filter_presets(
    on_progress: Optional[Callable] = None,
) -> List[Dict]:
    """Load all saved filter presets.

    Returns:
        List of preset dicts.
    """
    if not os.path.isdir(PRESETS_DIR):
        return []

    presets = []
    for fname in os.listdir(PRESETS_DIR):
        if fname.endswith(".json"):
            fpath = os.path.join(PRESETS_DIR, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                data["preset_id"] = fname[:-5]
                presets.append(data)
            except (json.JSONDecodeError, OSError):
                continue

    return presets
