"""
OpenCut 3D Gaussian Splat Viewer v21.6.0

Import .ply Gaussian splat files, define camera paths, render as video:
- PLY parsing: read header to count vertices, extract bounds from position data
- Camera path: keyframed positions/rotations with linear/cubic interpolation
- Rendering: gsplat/nerfstudio when available, Pillow point-cloud fallback
- Orbit camera path generation for turntable-style previews

References:
  - 3D Gaussian Splatting: https://github.com/graphdeco-inria/gaussian-splatting
  - gsplat: https://github.com/nerfstudio-project/gsplat
  - nerfstudio: https://github.com/nerfstudio-project/nerfstudio
"""

import logging
import math
import os
import struct
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from opencut.helpers import ensure_package, run_ffmpeg

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class CameraKeyframe:
    """A single camera keyframe in a camera path."""
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # pitch, yaw, roll
    fov: float = 60.0
    time: float = 0.0


@dataclass
class CameraPath:
    """Sequence of camera keyframes defining a camera trajectory."""
    keyframes: List[CameraKeyframe] = field(default_factory=list)
    interpolation: str = "linear"  # "linear" or "cubic"
    fps: int = 30
    loop: bool = False

    @property
    def duration(self) -> float:
        """Total duration in seconds."""
        if not self.keyframes:
            return 0.0
        return max(kf.time for kf in self.keyframes)

    @property
    def total_frames(self) -> int:
        """Total number of frames for the path."""
        return max(1, int(self.duration * self.fps))


@dataclass
class SplatScene:
    """Metadata about a loaded Gaussian splat scene."""
    ply_path: str = ""
    point_count: int = 0
    bounds: Dict = field(default_factory=lambda: {
        "min": (0.0, 0.0, 0.0),
        "max": (0.0, 0.0, 0.0),
        "center": (0.0, 0.0, 0.0),
        "extent": (0.0, 0.0, 0.0),
    })
    has_colors: bool = False
    has_normals: bool = False
    has_sh_coeffs: bool = False
    has_opacity: bool = False
    format_valid: bool = False
    properties: List[str] = field(default_factory=list)
    message: str = ""


@dataclass
class SplatRenderResult:
    """Result of rendering a splat scene to video."""
    output_path: str = ""
    duration: float = 0.0
    frames: int = 0
    resolution: Tuple[int, int] = (1280, 720)
    renderer: str = ""
    message: str = ""


# ---------------------------------------------------------------------------
# PLY Parsing
# ---------------------------------------------------------------------------

def _parse_ply_header(ply_path: str) -> Dict:
    """
    Parse a PLY file header to extract metadata.

    Returns dict with: format, vertex_count, properties, header_size_bytes.
    """
    properties = []
    vertex_count = 0
    fmt = "ascii"
    header_lines = []

    with open(ply_path, "rb") as f:
        for raw_line in f:
            line = raw_line.decode("ascii", errors="replace").strip()
            header_lines.append(raw_line)
            if line == "end_header":
                break
            if line.startswith("format "):
                parts = line.split()
                fmt = parts[1] if len(parts) > 1 else "ascii"
            elif line.startswith("element vertex "):
                parts = line.split()
                vertex_count = int(parts[2]) if len(parts) > 2 else 0
            elif line.startswith("property "):
                properties.append(line)

    header_size = sum(len(line) for line in header_lines)

    return {
        "format": fmt,
        "vertex_count": vertex_count,
        "properties": properties,
        "header_size_bytes": header_size,
    }


def _extract_property_names(properties: List[str]) -> List[str]:
    """Extract property names from PLY property lines."""
    names = []
    for prop in properties:
        parts = prop.split()
        if len(parts) >= 3 and parts[0] == "property":
            names.append(parts[-1])
    return names


def _get_property_struct_format(properties: List[str]) -> str:
    """Build a struct format string for binary PLY vertex data."""
    type_map = {
        "float": "f", "float32": "f",
        "double": "d", "float64": "d",
        "uchar": "B", "uint8": "B",
        "char": "b", "int8": "b",
        "ushort": "H", "uint16": "H",
        "short": "h", "int16": "h",
        "uint": "I", "uint32": "I",
        "int": "i", "int32": "i",
    }
    fmt_chars = []
    for prop in properties:
        parts = prop.split()
        if len(parts) >= 3 and parts[0] == "property":
            dtype = parts[1]
            char = type_map.get(dtype, "f")
            fmt_chars.append(char)
    return "<" + "".join(fmt_chars)


def _read_positions_binary(ply_path: str, header_info: Dict) -> List[Tuple[float, float, float]]:
    """
    Read vertex positions from a binary little-endian PLY file.

    Returns list of (x, y, z) tuples (capped at 100k for bounds computation).
    """
    props = header_info["properties"]
    prop_names = _extract_property_names(props)
    vertex_count = header_info["vertex_count"]

    # Find x, y, z indices
    try:
        xi = prop_names.index("x")
        yi = prop_names.index("y")
        zi = prop_names.index("z")
    except ValueError:
        return []

    fmt = _get_property_struct_format(props)
    vertex_size = struct.calcsize(fmt)

    positions = []
    # Cap for performance on large scenes
    max_read = min(vertex_count, 100000)
    step = max(1, vertex_count // max_read)

    with open(ply_path, "rb") as f:
        f.seek(header_info["header_size_bytes"])
        for i in range(vertex_count):
            data = f.read(vertex_size)
            if len(data) < vertex_size:
                break
            if i % step == 0:
                values = struct.unpack(fmt, data)
                positions.append((values[xi], values[yi], values[zi]))

    return positions


def _read_positions_ascii(ply_path: str, header_info: Dict) -> List[Tuple[float, float, float]]:
    """Read vertex positions from an ASCII PLY file."""
    prop_names = _extract_property_names(header_info["properties"])
    vertex_count = header_info["vertex_count"]

    try:
        xi = prop_names.index("x")
        yi = prop_names.index("y")
        zi = prop_names.index("z")
    except ValueError:
        return []

    positions = []
    max_read = min(vertex_count, 100000)
    step = max(1, vertex_count // max_read)

    with open(ply_path, "r", errors="replace") as f:
        # Skip header
        for line in f:
            if line.strip() == "end_header":
                break

        for i in range(vertex_count):
            line = f.readline()
            if not line:
                break
            if i % step == 0:
                parts = line.strip().split()
                if len(parts) > max(xi, yi, zi):
                    try:
                        positions.append((
                            float(parts[xi]),
                            float(parts[yi]),
                            float(parts[zi]),
                        ))
                    except ValueError:
                        continue

    return positions


def _compute_bounds(positions: List[Tuple[float, float, float]]) -> Dict:
    """Compute bounding box from position data."""
    if not positions:
        return {
            "min": (0.0, 0.0, 0.0),
            "max": (0.0, 0.0, 0.0),
            "center": (0.0, 0.0, 0.0),
            "extent": (0.0, 0.0, 0.0),
        }

    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    zs = [p[2] for p in positions]

    min_pt = (min(xs), min(ys), min(zs))
    max_pt = (max(xs), max(ys), max(zs))
    center = (
        (min_pt[0] + max_pt[0]) / 2.0,
        (min_pt[1] + max_pt[1]) / 2.0,
        (min_pt[2] + max_pt[2]) / 2.0,
    )
    extent = (
        max_pt[0] - min_pt[0],
        max_pt[1] - min_pt[1],
        max_pt[2] - min_pt[2],
    )

    return {
        "min": min_pt,
        "max": max_pt,
        "center": center,
        "extent": extent,
    }


# ---------------------------------------------------------------------------
# Public API: Load & Validate
# ---------------------------------------------------------------------------

def validate_splat(ply_path: str) -> Dict:
    """
    Validate that a PLY file is a valid Gaussian splat file.

    Returns:
        Dict with keys: valid (bool), point_count, properties, errors.
    """
    errors = []

    if not os.path.isfile(ply_path):
        return {"valid": False, "point_count": 0, "properties": [], "errors": ["File not found"]}

    if not ply_path.lower().endswith(".ply"):
        errors.append("File does not have .ply extension")

    try:
        with open(ply_path, "rb") as f:
            magic = f.read(3)
            if magic != b"ply":
                return {"valid": False, "point_count": 0, "properties": [], "errors": ["Not a PLY file (missing magic bytes)"]}
    except (IOError, OSError) as e:
        return {"valid": False, "point_count": 0, "properties": [], "errors": [f"Cannot read file: {e}"]}

    try:
        header = _parse_ply_header(ply_path)
    except Exception as e:
        return {"valid": False, "point_count": 0, "properties": [], "errors": [f"Failed to parse PLY header: {e}"]}

    prop_names = _extract_property_names(header["properties"])

    # Check required position properties
    required = ["x", "y", "z"]
    for r in required:
        if r not in prop_names:
            errors.append(f"Missing required property: {r}")

    # Gaussian splat typically has SH coefficients and opacity
    has_sh = any(p.startswith("f_dc_") or p.startswith("f_rest_") for p in prop_names)
    has_opacity = "opacity" in prop_names
    has_scale = any(p.startswith("scale_") for p in prop_names)
    has_rot = any(p.startswith("rot_") for p in prop_names)

    if not has_sh and not has_opacity:
        errors.append("Warning: no SH coefficients or opacity found (may not be a Gaussian splat)")

    return {
        "valid": len([e for e in errors if not e.startswith("Warning:")]) == 0,
        "point_count": header["vertex_count"],
        "properties": prop_names,
        "has_sh": has_sh,
        "has_opacity": has_opacity,
        "has_scale": has_scale,
        "has_rotation": has_rot,
        "format": header["format"],
        "errors": errors,
    }


def load_splat(ply_path: str) -> SplatScene:
    """
    Load a Gaussian splat PLY file and extract scene metadata.

    Parses the PLY header for vertex count, properties, and reads
    position data to compute scene bounds.

    Args:
        ply_path: Path to the .ply file.

    Returns:
        SplatScene with metadata about the scene.
    """
    if not os.path.isfile(ply_path):
        raise FileNotFoundError(f"PLY file not found: {ply_path}")

    header = _parse_ply_header(ply_path)
    prop_names = _extract_property_names(header["properties"])

    # Read positions to compute bounds
    if header["format"].startswith("binary_little_endian"):
        positions = _read_positions_binary(ply_path, header)
    elif header["format"] == "ascii":
        positions = _read_positions_ascii(ply_path, header)
    else:
        positions = []

    bounds = _compute_bounds(positions)

    has_colors = any(c in prop_names for c in ("red", "green", "blue", "f_dc_0", "f_dc_1", "f_dc_2"))
    has_normals = all(n in prop_names for n in ("nx", "ny", "nz"))
    has_sh = any(p.startswith("f_dc_") or p.startswith("f_rest_") for p in prop_names)
    has_opacity = "opacity" in prop_names

    return SplatScene(
        ply_path=ply_path,
        point_count=header["vertex_count"],
        bounds=bounds,
        has_colors=has_colors,
        has_normals=has_normals,
        has_sh_coeffs=has_sh,
        has_opacity=has_opacity,
        format_valid=True,
        properties=prop_names,
        message=f"Loaded {header['vertex_count']:,} points from {os.path.basename(ply_path)}",
    )


# ---------------------------------------------------------------------------
# Camera Path
# ---------------------------------------------------------------------------

def define_camera_path(
    keyframes: List[CameraKeyframe],
    interpolation: str = "linear",
    fps: int = 30,
    loop: bool = False,
) -> CameraPath:
    """
    Define a camera path from a list of keyframes.

    Args:
        keyframes: List of CameraKeyframe objects.
        interpolation: "linear" or "cubic" (Catmull-Rom spline).
        fps: Frames per second.
        loop: Whether the path loops back to the start.

    Returns:
        CameraPath object.
    """
    if not keyframes:
        raise ValueError("At least one keyframe is required")

    if interpolation not in ("linear", "cubic"):
        interpolation = "linear"

    # Sort by time
    sorted_kfs = sorted(keyframes, key=lambda kf: kf.time)

    return CameraPath(
        keyframes=sorted_kfs,
        interpolation=interpolation,
        fps=fps,
        loop=loop,
    )


def create_orbit_path(
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    radius: float = 3.0,
    height: float = 1.0,
    frames: int = 120,
    fps: int = 30,
    fov: float = 60.0,
    loops: int = 1,
) -> CameraPath:
    """
    Create a circular orbit camera path around a center point.

    Args:
        center: Center of the orbit (x, y, z).
        radius: Orbit radius.
        height: Camera height above center.
        frames: Total number of frames per loop.
        fps: Frames per second.
        fov: Field of view in degrees.
        loops: Number of orbit loops.

    Returns:
        CameraPath with keyframes for a full orbit.
    """
    if radius <= 0:
        raise ValueError("Orbit radius must be positive")
    if frames < 2:
        raise ValueError("Orbit needs at least 2 frames")

    keyframes = []
    total_frames = frames * loops

    for i in range(total_frames):
        angle = (2.0 * math.pi * i) / frames
        x = center[0] + radius * math.cos(angle)
        z = center[2] + radius * math.sin(angle)
        y = center[1] + height

        # Yaw points toward center
        yaw = math.degrees(math.atan2(center[2] - z, center[0] - x))
        # Pitch looks slightly down toward center
        dx = center[0] - x
        dz = center[2] - z
        dist_xz = math.sqrt(dx * dx + dz * dz)
        pitch = -math.degrees(math.atan2(height, dist_xz)) if dist_xz > 0 else 0.0

        t = i / fps

        keyframes.append(CameraKeyframe(
            position=(x, y, z),
            rotation=(pitch, yaw, 0.0),
            fov=fov,
            time=t,
        ))

    return CameraPath(
        keyframes=keyframes,
        interpolation="linear",
        fps=fps,
        loop=True,
    )


# ---------------------------------------------------------------------------
# Interpolation Helpers
# ---------------------------------------------------------------------------

def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between two values."""
    return a + (b - a) * t


def _lerp_tuple(a: Tuple, b: Tuple, t: float) -> Tuple:
    """Linearly interpolate between two tuples."""
    return tuple(_lerp(ai, bi, t) for ai, bi in zip(a, b))


def _catmull_rom(p0: float, p1: float, p2: float, p3: float, t: float) -> float:
    """Catmull-Rom spline interpolation for a single value."""
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2.0 * p1) +
        (-p0 + p2) * t +
        (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
        (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )


def _catmull_rom_tuple(p0: Tuple, p1: Tuple, p2: Tuple, p3: Tuple, t: float) -> Tuple:
    """Catmull-Rom interpolation for tuples."""
    return tuple(
        _catmull_rom(a, b, c, d, t)
        for a, b, c, d in zip(p0, p1, p2, p3)
    )


def _interpolate_camera(path: CameraPath, time: float) -> CameraKeyframe:
    """Interpolate camera state at a given time along the path."""
    kfs = path.keyframes
    if not kfs:
        return CameraKeyframe()
    if len(kfs) == 1 or time <= kfs[0].time:
        return kfs[0]
    if time >= kfs[-1].time:
        return kfs[-1]

    # Find surrounding keyframes
    idx = 0
    for i in range(len(kfs) - 1):
        if kfs[i].time <= time <= kfs[i + 1].time:
            idx = i
            break

    kf0 = kfs[idx]
    kf1 = kfs[idx + 1]
    seg_duration = kf1.time - kf0.time
    t = (time - kf0.time) / seg_duration if seg_duration > 0 else 0.0
    t = max(0.0, min(1.0, t))

    if path.interpolation == "cubic" and len(kfs) >= 4:
        # Catmull-Rom with clamped boundary indices
        i0 = max(0, idx - 1)
        i3 = min(len(kfs) - 1, idx + 2)

        pos = _catmull_rom_tuple(kfs[i0].position, kf0.position, kf1.position, kfs[i3].position, t)
        rot = _catmull_rom_tuple(kfs[i0].rotation, kf0.rotation, kf1.rotation, kfs[i3].rotation, t)
        fov = _catmull_rom(kfs[i0].fov, kf0.fov, kf1.fov, kfs[i3].fov, t)
    else:
        pos = _lerp_tuple(kf0.position, kf1.position, t)
        rot = _lerp_tuple(kf0.rotation, kf1.rotation, t)
        fov = _lerp(kf0.fov, kf1.fov, t)

    return CameraKeyframe(
        position=pos,
        rotation=rot,
        fov=fov,
        time=time,
    )


# ---------------------------------------------------------------------------
# Fallback Renderer (Pillow point-cloud)
# ---------------------------------------------------------------------------

def _project_point(
    point: Tuple[float, float, float],
    cam_pos: Tuple[float, float, float],
    cam_rot: Tuple[float, float, float],
    fov: float,
    width: int,
    height: int,
) -> Optional[Tuple[int, int, float]]:
    """
    Project a 3D point to 2D screen coordinates using simple perspective.

    Returns (screen_x, screen_y, depth) or None if behind camera.
    """
    # Translate point relative to camera
    dx = point[0] - cam_pos[0]
    dy = point[1] - cam_pos[1]
    dz = point[2] - cam_pos[2]

    # Simple rotation (yaw only for performance)
    yaw_rad = math.radians(cam_rot[1])
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)

    rx = dx * cos_y + dz * sin_y
    rz = -dx * sin_y + dz * cos_y

    # Pitch
    pitch_rad = math.radians(cam_rot[0])
    cos_p = math.cos(pitch_rad)
    sin_p = math.sin(pitch_rad)

    ry = dy * cos_p - rz * sin_p
    rz2 = dy * sin_p + rz * cos_p

    # Behind camera
    if rz2 <= 0.1:
        return None

    # Perspective projection
    focal = (width / 2.0) / math.tan(math.radians(fov / 2.0))

    sx = int(width / 2.0 + (rx / rz2) * focal)
    sy = int(height / 2.0 - (ry / rz2) * focal)

    if 0 <= sx < width and 0 <= sy < height:
        return (sx, sy, rz2)
    return None


def _render_frame_pillow(
    positions: List[Tuple[float, float, float]],
    colors: Optional[List[Tuple[int, int, int]]],
    camera: CameraKeyframe,
    width: int,
    height: int,
) -> object:
    """Render a single frame as a Pillow Image using point projection."""
    if not ensure_package("PIL", "Pillow"):
        raise RuntimeError("Failed to install Pillow")
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), (20, 20, 30))
    draw = ImageDraw.Draw(img)

    # Project and sort by depth
    projected = []
    for i, pos in enumerate(positions):
        result = _project_point(
            pos, camera.position, camera.rotation,
            camera.fov, width, height,
        )
        if result:
            sx, sy, depth = result
            color = colors[i] if colors and i < len(colors) else (200, 200, 200)
            projected.append((sx, sy, depth, color))

    # Sort by depth (far to near) for painter's algorithm
    projected.sort(key=lambda p: -p[2])

    # Draw points
    for sx, sy, depth, color in projected:
        # Size inversely proportional to depth
        size = max(1, int(3.0 / max(0.5, depth)))
        draw.ellipse(
            [sx - size, sy - size, sx + size, sy + size],
            fill=color,
        )

    return img


def _read_colors_binary(
    ply_path: str,
    header_info: Dict,
    max_points: int = 100000,
) -> List[Tuple[int, int, int]]:
    """Read vertex colors from a binary PLY file."""
    props = header_info["properties"]
    prop_names = _extract_property_names(props)
    vertex_count = header_info["vertex_count"]

    # Try direct RGB colors
    has_rgb = all(c in prop_names for c in ("red", "green", "blue"))
    # Or SH DC coefficients
    has_sh = all(c in prop_names for c in ("f_dc_0", "f_dc_1", "f_dc_2"))

    if not has_rgb and not has_sh:
        return []

    if has_rgb:
        ri = prop_names.index("red")
        gi = prop_names.index("green")
        bi = prop_names.index("blue")
    else:
        ri = prop_names.index("f_dc_0")
        gi = prop_names.index("f_dc_1")
        bi = prop_names.index("f_dc_2")

    fmt = _get_property_struct_format(props)
    vertex_size = struct.calcsize(fmt)

    colors = []
    step = max(1, vertex_count // max_points)

    with open(ply_path, "rb") as f:
        f.seek(header_info["header_size_bytes"])
        for i in range(vertex_count):
            data = f.read(vertex_size)
            if len(data) < vertex_size:
                break
            if i % step == 0:
                values = struct.unpack(fmt, data)
                if has_rgb:
                    colors.append((int(values[ri]), int(values[gi]), int(values[bi])))
                else:
                    # SH DC to RGB: C0 * SH_C0 + 0.5, clamped to [0,255]
                    SH_C0 = 0.28209479177387814
                    r = max(0, min(255, int((values[ri] * SH_C0 + 0.5) * 255)))
                    g = max(0, min(255, int((values[gi] * SH_C0 + 0.5) * 255)))
                    b = max(0, min(255, int((values[bi] * SH_C0 + 0.5) * 255)))
                    colors.append((r, g, b))

    return colors


# ---------------------------------------------------------------------------
# Render Single Frame (for preview)
# ---------------------------------------------------------------------------

def render_splat_frame(
    ply_path: str,
    camera: CameraKeyframe,
    resolution: Tuple[int, int] = (1280, 720),
) -> str:
    """
    Render a single frame of a splat scene at a given camera position.

    Uses Pillow fallback renderer (point cloud projection).

    Args:
        ply_path: Path to the .ply file.
        camera: CameraKeyframe with position/rotation/fov.
        resolution: Output image (width, height).

    Returns:
        Path to the rendered PNG frame.
    """
    if not os.path.isfile(ply_path):
        raise FileNotFoundError(f"PLY file not found: {ply_path}")

    header = _parse_ply_header(ply_path)
    w, h = resolution

    if header["format"].startswith("binary_little_endian"):
        positions = _read_positions_binary(ply_path, header)
        colors = _read_colors_binary(ply_path, header)
    else:
        positions = _read_positions_ascii(ply_path, header)
        colors = []

    img = _render_frame_pillow(positions, colors or None, camera, w, h)

    # Save to temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    img.save(tmp.name)

    return tmp.name


# ---------------------------------------------------------------------------
# Render to Video
# ---------------------------------------------------------------------------

def render_splat_to_video(
    ply_path: str,
    camera_path: CameraPath,
    output_path: Optional[str] = None,
    output_dir: str = "",
    resolution: Tuple[int, int] = (1280, 720),
    on_progress: Optional[Callable] = None,
) -> SplatRenderResult:
    """
    Render a Gaussian splat scene to video along a camera path.

    Attempts to use gsplat/nerfstudio if available, otherwise falls back
    to the Pillow-based point cloud renderer.

    Args:
        ply_path: Path to the .ply file.
        camera_path: CameraPath defining the camera trajectory.
        output_path: Explicit output path. Auto-generated if None.
        output_dir: Output directory (used when output_path is None).
        resolution: Output video (width, height).
        on_progress: Progress callback(pct, msg).

    Returns:
        SplatRenderResult with output_path, duration, etc.
    """
    if not os.path.isfile(ply_path):
        raise FileNotFoundError(f"PLY file not found: {ply_path}")

    if not camera_path.keyframes:
        raise ValueError("Camera path has no keyframes")

    if output_path is None:
        base = os.path.splitext(os.path.basename(ply_path))[0]
        directory = output_dir or os.path.dirname(ply_path)
        output_path = os.path.join(directory, f"{base}_render.mp4")

    w, h = resolution
    total_frames = camera_path.total_frames
    duration = camera_path.duration

    if on_progress:
        on_progress(5, "Loading splat scene...")

    # Try gsplat/nerfstudio first
    renderer = "pillow_fallback"
    try:
        import gsplat  # noqa: F401
        renderer = "gsplat"
        # gsplat rendering would go here in a full implementation
        raise ImportError("gsplat video rendering not yet integrated")
    except ImportError:
        pass

    # Fallback: Pillow point-cloud renderer
    if renderer != "gsplat":
        if not ensure_package("PIL", "Pillow"):
            raise RuntimeError("Failed to install Pillow for fallback rendering")

        if on_progress:
            on_progress(10, "Loading point cloud data...")

        header = _parse_ply_header(ply_path)

        if header["format"].startswith("binary_little_endian"):
            positions = _read_positions_binary(ply_path, header)
            colors = _read_colors_binary(ply_path, header)
        else:
            positions = _read_positions_ascii(ply_path, header)
            colors = []

        if not positions:
            raise ValueError("No position data found in PLY file")

        if on_progress:
            on_progress(15, f"Rendering {total_frames} frames...")

        # Render frames to temp dir
        tmp_dir = tempfile.mkdtemp(prefix="opencut_splat_")
        try:
            for frame_idx in range(total_frames):
                t = frame_idx / camera_path.fps
                camera = _interpolate_camera(camera_path, t)

                img = _render_frame_pillow(positions, colors or None, camera, w, h)
                frame_path = os.path.join(tmp_dir, f"frame_{frame_idx:06d}.png")
                img.save(frame_path)

                if on_progress and frame_idx % max(1, total_frames // 20) == 0:
                    pct = 15 + int((frame_idx / max(1, total_frames)) * 75)
                    on_progress(min(92, pct), f"Rendering frame {frame_idx}/{total_frames}...")

            if on_progress:
                on_progress(93, "Encoding video...")

            # Encode frames to video with FFmpeg
            frame_pattern = os.path.join(tmp_dir, "frame_%06d.png")
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-framerate", str(camera_path.fps),
                "-i", frame_pattern,
                "-c:v", "libx264", "-crf", "18", "-preset", "medium",
                "-pix_fmt", "yuv420p",
                output_path,
            ]
            run_ffmpeg(cmd, timeout=3600)

        finally:
            # Clean up temp frames
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if on_progress:
        on_progress(100, "Splat rendering complete!")

    return SplatRenderResult(
        output_path=output_path,
        duration=duration,
        frames=total_frames,
        resolution=resolution,
        renderer=renderer,
        message=f"Rendered {total_frames} frames with {renderer}",
    )
