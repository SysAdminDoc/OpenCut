"""
OpenCut Filter Chain Builder

Build custom FFmpeg filter chains programmatically.  Node-based graph
system where each node represents an FFmpeg filter with typed parameters.
Validate chains, detect cycles, generate ``-filter_complex`` strings,
preview on a single frame, and save/load as JSON presets.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from opencut.helpers import OPENCUT_DIR

logger = logging.getLogger("opencut")

PRESETS_DIR = os.path.join(OPENCUT_DIR, "filter_presets")

# ---------------------------------------------------------------------------
# Known filter catalogue — 20 common filters + extras
# ---------------------------------------------------------------------------

KNOWN_FILTERS: Dict[str, Dict[str, Any]] = {
    # Video filters
    "scale": {"inputs": 1, "outputs": 1, "params": ["w", "h", "flags"]},
    "crop": {"inputs": 1, "outputs": 1, "params": ["w", "h", "x", "y"]},
    "overlay": {"inputs": 2, "outputs": 1, "params": ["x", "y", "eof_action"]},
    "drawtext": {
        "inputs": 1, "outputs": 1,
        "params": ["text", "fontfile", "fontsize", "fontcolor", "x", "y",
                    "box", "boxcolor", "borderw"],
    },
    "colorbalance": {
        "inputs": 1, "outputs": 1,
        "params": ["rs", "gs", "bs", "rm", "gm", "bm", "rh", "gh", "bh"],
    },
    "eq": {
        "inputs": 1, "outputs": 1,
        "params": ["brightness", "contrast", "saturation", "gamma"],
    },
    "hue": {"inputs": 1, "outputs": 1, "params": ["h", "s", "b"]},
    "unsharp": {
        "inputs": 1, "outputs": 1,
        "params": ["luma_msize_x", "luma_msize_y", "luma_amount",
                    "chroma_msize_x", "chroma_msize_y", "chroma_amount"],
    },
    "noise": {
        "inputs": 1, "outputs": 1,
        "params": ["alls", "allf", "c0s", "c0f"],
    },
    "vignette": {"inputs": 1, "outputs": 1, "params": ["angle", "x0", "y0"]},
    "fade": {
        "inputs": 1, "outputs": 1,
        "params": ["type", "start_frame", "nb_frames", "start_time",
                    "duration", "color", "alpha"],
    },
    "concat": {
        "inputs": -1, "outputs": 1,
        "params": ["n", "v", "a"],
    },
    "split": {"inputs": 1, "outputs": -1, "params": ["outputs"]},
    "hstack": {"inputs": -1, "outputs": 1, "params": ["inputs"]},
    "vstack": {"inputs": -1, "outputs": 1, "params": ["inputs"]},
    "pad": {
        "inputs": 1, "outputs": 1,
        "params": ["w", "h", "x", "y", "color"],
    },
    "transpose": {"inputs": 1, "outputs": 1, "params": ["dir"]},
    "setpts": {"inputs": 1, "outputs": 1, "params": ["expr"]},
    # Audio filters
    "volume": {"inputs": 1, "outputs": 1, "params": ["volume"]},
    "amerge": {"inputs": -1, "outputs": 1, "params": ["inputs"]},
}

# Extended set for validation (known but not in the typed catalogue)
EXTENDED_FILTERS: Set[str] = {
    "curves", "lut3d", "fps", "trim", "blend", "chromakey", "colorkey",
    "vflip", "hflip", "rotate", "boxblur", "gblur", "deband",
    "deflicker", "denoise", "zoompan", "format", "null", "anull",
    "atempo", "aresample", "amix", "adelay", "aecho", "acompressor",
    "equalizer", "highpass", "lowpass",
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

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


@dataclass
class FilterChain:
    """Complete filter chain with nodes and connections."""
    nodes: List[FilterNode] = field(default_factory=list)
    connections: List[FilterConnection] = field(default_factory=list)
    name: str = ""
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "name": self.name,
            "description": self.description,
            "nodes": [asdict(n) for n in self.nodes],
            "connections": [asdict(c) for c in self.connections],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FilterChain":
        """Create FilterChain from a dict."""
        nodes = [
            FilterNode(
                node_id=n.get("node_id", f"n{i}"),
                filter_name=n["filter_name"],
                params=n.get("params", {}),
                inputs=n.get("inputs", []),
                outputs=n.get("outputs", []),
            )
            for i, n in enumerate(data.get("nodes", []))
        ]
        connections = [
            FilterConnection(
                from_node=c["from_node"],
                from_pad=c.get("from_pad", "default"),
                to_node=c["to_node"],
                to_pad=c.get("to_pad", "default"),
            )
            for c in data.get("connections", [])
        ]
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            nodes=nodes,
            connections=connections,
        )


# ---------------------------------------------------------------------------
# Node → filter string conversion
# ---------------------------------------------------------------------------

def _node_to_filter_str(node: FilterNode) -> str:
    """Convert a FilterNode to an FFmpeg filter expression."""
    params_str = ""
    if node.params:
        parts = []
        for k, v in node.params.items():
            if isinstance(v, bool):
                parts.append(f"{k}={1 if v else 0}")
            elif v is not None:
                # Escape colons in text values for drawtext
                val = str(v)
                if node.filter_name == "drawtext" and k == "text":
                    val = val.replace(":", "\\:")
                parts.append(f"{k}={val}")
        if parts:
            params_str = "=" + ":".join(parts)

    input_pads = "".join(f"[{i}]" for i in node.inputs) if node.inputs else ""
    output_pads = "".join(f"[{o}]" for o in node.outputs) if node.outputs else ""

    return f"{input_pads}{node.filter_name}{params_str}{output_pads}"


# ---------------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------------

def _detect_cycles(
    nodes: List[FilterNode],
    connections: List[FilterConnection],
) -> List[str]:
    """Detect cycles in the filter graph via DFS.

    Returns:
        List of error messages (empty if no cycles).
    """
    node_ids = {n.node_id for n in nodes}
    # Build adjacency list
    adj: Dict[str, List[str]] = {nid: [] for nid in node_ids}
    for conn in connections:
        if conn.from_node in adj:
            adj[conn.from_node].append(conn.to_node)

    # DFS-based cycle detection
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {nid: WHITE for nid in node_ids}
    cycles: List[str] = []

    def _dfs(node_id: str, path: List[str]):
        color[node_id] = GRAY
        path.append(node_id)
        for neighbor in adj.get(node_id, []):
            if neighbor not in color:
                continue
            if color[neighbor] == GRAY:
                cycle_start = path.index(neighbor)
                cycle_path = " -> ".join(path[cycle_start:] + [neighbor])
                cycles.append(f"Cycle detected: {cycle_path}")
            elif color[neighbor] == WHITE:
                _dfs(neighbor, path)
        path.pop()
        color[node_id] = BLACK

    for nid in node_ids:
        if color[nid] == WHITE:
            _dfs(nid, [])

    return cycles


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_chain(
    chain: FilterChain,
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Validate a filter chain for correctness.

    Checks:
    - Required fields on nodes
    - Duplicate node IDs
    - Connection references to existing nodes
    - Known/unknown filter names
    - Pad compatibility (input/output counts)
    - Cycle detection

    Args:
        chain: FilterChain to validate.
        on_progress: Optional progress callback (int).

    Returns:
        Dict with 'valid' bool, 'errors' list, and 'warnings' list.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if on_progress:
        on_progress(10)

    nodes = chain.nodes
    connections = chain.connections

    if not nodes:
        errors.append("Chain must contain at least one node")
        return {"valid": False, "errors": errors, "warnings": warnings}

    # Check required fields and duplicate IDs
    node_ids: Set[str] = set()
    for i, node in enumerate(nodes):
        if not node.filter_name:
            errors.append(f"Node {i} missing 'filter_name'")
        if not node.node_id:
            errors.append(f"Node {i} missing 'node_id'")
        elif node.node_id in node_ids:
            errors.append(f"Duplicate node_id: {node.node_id}")
        node_ids.add(node.node_id)

    if on_progress:
        on_progress(30)

    # Validate connections reference existing nodes
    for conn in connections:
        if conn.from_node not in node_ids:
            errors.append(
                f"Connection references unknown source node: {conn.from_node}")
        if conn.to_node not in node_ids:
            errors.append(
                f"Connection references unknown target node: {conn.to_node}")

    if on_progress:
        on_progress(50)

    # Check for known filters and pad compatibility
    all_known = set(KNOWN_FILTERS.keys()) | EXTENDED_FILTERS
    for node in nodes:
        fname = node.filter_name
        if fname and fname not in all_known:
            warnings.append(
                f"Unknown filter '{fname}' - may still work if FFmpeg "
                f"supports it")

        if fname in KNOWN_FILTERS:
            spec = KNOWN_FILTERS[fname]
            expected_inputs = spec["inputs"]
            expected_outputs = spec["outputs"]

            # Check input count (skip if -1 = variable)
            if expected_inputs > 0 and node.inputs:
                if len(node.inputs) != expected_inputs:
                    warnings.append(
                        f"Node '{node.node_id}' ({fname}): expected "
                        f"{expected_inputs} input(s), got {len(node.inputs)}")

            # Check output count
            if expected_outputs > 0 and node.outputs:
                if len(node.outputs) != expected_outputs:
                    warnings.append(
                        f"Node '{node.node_id}' ({fname}): expected "
                        f"{expected_outputs} output(s), got {len(node.outputs)}")

    if on_progress:
        on_progress(70)

    # Cycle detection
    cycle_errors = _detect_cycles(nodes, connections)
    errors.extend(cycle_errors)

    if on_progress:
        on_progress(100)

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "node_count": len(nodes),
        "connection_count": len(connections),
    }


# ---------------------------------------------------------------------------
# Build filter_complex string
# ---------------------------------------------------------------------------

def build_filter_string(
    chain: FilterChain,
    on_progress: Optional[Callable] = None,
) -> str:
    """Generate the FFmpeg ``-filter_complex`` argument from a FilterChain.

    Applies connection information to set node input/output pads, then
    joins all nodes with ``;``.

    Args:
        chain: FilterChain to build from.
        on_progress: Optional progress callback (int).

    Returns:
        FFmpeg filter_complex string.

    Raises:
        ValueError: If the chain is empty or invalid.
    """
    if not chain.nodes:
        raise ValueError("At least one filter node is required")

    if on_progress:
        on_progress(20)

    # Deep copy nodes so we don't mutate the originals
    import copy
    nodes = [copy.deepcopy(n) for n in chain.nodes]
    node_map = {n.node_id: n for n in nodes}

    # Apply connections
    if chain.connections:
        for conn in chain.connections:
            label = f"{conn.from_node}_{conn.from_pad}"
            # Add output pad to source node
            src = node_map.get(conn.from_node)
            if src and label not in src.outputs:
                src.outputs.append(label)
            # Add input pad to destination node
            dst = node_map.get(conn.to_node)
            if dst and label not in dst.inputs:
                dst.inputs.append(label)

    if on_progress:
        on_progress(60)

    filter_parts = [_node_to_filter_str(n) for n in nodes]
    result = ";".join(filter_parts)

    if on_progress:
        on_progress(100)

    logger.info("Built filter chain with %d nodes: %s",
                len(nodes), result[:200])
    return result


# Convenience wrapper matching the old API
def build_filter_chain(
    nodes: List[Dict],
    connections: Optional[List[Dict]] = None,
    on_progress: Optional[Callable] = None,
) -> str:
    """Build a filter_complex string from node/connection dicts.

    This is a convenience wrapper around ``build_filter_string`` that
    accepts raw dicts instead of dataclass instances.

    Args:
        nodes: List of node dicts.
        connections: Optional list of connection dicts.
        on_progress: Optional progress callback.

    Returns:
        FFmpeg filter_complex string.
    """
    chain = FilterChain.from_dict({
        "nodes": nodes,
        "connections": connections or [],
    })
    return build_filter_string(chain, on_progress=on_progress)


# ---------------------------------------------------------------------------
# Validation wrapper (dict API)
# ---------------------------------------------------------------------------

def validate_filter_graph(
    graph: Dict,
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Validate a filter graph dict for correctness.

    Args:
        graph: Dict with 'nodes' and optional 'connections' keys.

    Returns:
        Validation result dict.
    """
    chain = FilterChain.from_dict(graph)
    return validate_chain(chain, on_progress=on_progress)


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

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
    from opencut.helpers import FFmpegCmd, run_ffmpeg

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if on_progress:
        on_progress(20)

    if output_path is None:
        import tempfile
        output_path = os.path.join(
            tempfile.gettempdir(),
            f"filter_preview_{os.getpid()}.jpg",
        )

    cmd = (
        FFmpegCmd()
        .pre_input("-ss", str(timestamp))
        .input(video_path)
        .filter_complex(
            filter_chain,
            maps=["[out]"] if "[out]" in filter_chain else None,
        )
        .frames(1)
        .output(output_path)
        .build()
    )

    if on_progress:
        on_progress(60)

    run_ffmpeg(cmd, timeout=30)

    if on_progress:
        on_progress(100)

    return output_path


# ---------------------------------------------------------------------------
# Preset save/load
# ---------------------------------------------------------------------------

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
        on_progress(30)

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
        on_progress(100)

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
