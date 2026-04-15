"""
OpenCut Delivery Specification Profile Manager (70.5)

Define, manage, and compare delivery specification profiles for major
platforms and custom requirements. Built-in specs for Netflix, YouTube,
Apple TV+, Amazon, Broadcast EBU, DCP, and Theatrical.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from opencut.helpers import OPENCUT_DIR

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Custom specs persistence directory
# ---------------------------------------------------------------------------
_CUSTOM_SPECS_DIR = os.path.join(OPENCUT_DIR, "delivery_specs")


def _ensure_specs_dir():
    """Create custom specs directory if needed."""
    os.makedirs(_CUSTOM_SPECS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SpecRequirement:
    """A single requirement within a delivery spec."""
    category: str = ""        # "video", "audio", "container", "subtitle"
    field_name: str = ""      # "codec", "resolution", "bitrate_min", etc.
    operator: str = "eq"      # "eq", "neq", "gte", "lte", "in", "range", "regex"
    value: Any = None         # Expected value (string, number, list, tuple)
    unit: str = ""            # "bps", "kbps", "Hz", "LUFS", "px", etc.
    description: str = ""     # Human-readable requirement description
    severity: str = "error"   # "error", "warning", "info"
    required: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        """Generate human-readable description of this requirement."""
        if self.description:
            return self.description
        op_labels = {
            "eq": "must be",
            "neq": "must not be",
            "gte": "must be >=",
            "lte": "must be <=",
            "in": "must be one of",
            "range": "must be in range",
            "regex": "must match pattern",
        }
        op_label = op_labels.get(self.operator, self.operator)
        unit_str = f" {self.unit}" if self.unit else ""
        return f"{self.category}.{self.field_name} {op_label} {self.value}{unit_str}"


@dataclass
class DeliveryProfile:
    """A complete delivery specification profile."""
    name: str = ""
    display_name: str = ""
    description: str = ""
    platform: str = ""         # "netflix", "youtube", "apple_tv_plus", etc.
    version: str = "1.0"
    requirements: List[SpecRequirement] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    built_in: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "platform": self.platform,
            "version": self.version,
            "requirements": [r.to_dict() for r in self.requirements],
            "metadata": self.metadata,
            "built_in": self.built_in,
        }

    def get_requirements_by_category(self, category: str) -> List[SpecRequirement]:
        """Filter requirements by category (video, audio, etc.)."""
        return [r for r in self.requirements if r.category == category]

    def requirement_count(self) -> Dict[str, int]:
        """Count requirements per category."""
        counts: Dict[str, int] = {}
        for r in self.requirements:
            counts[r.category] = counts.get(r.category, 0) + 1
        return counts


@dataclass
class SpecDiff:
    """Difference between two delivery specs."""
    field_path: str = ""
    spec_a_value: Any = None
    spec_b_value: Any = None
    diff_type: str = ""  # "added", "removed", "changed", "same"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SpecCompareResult:
    """Result of comparing two delivery specs."""
    spec_a_name: str = ""
    spec_b_name: str = ""
    total_differences: int = 0
    added: List[SpecDiff] = field(default_factory=list)
    removed: List[SpecDiff] = field(default_factory=list)
    changed: List[SpecDiff] = field(default_factory=list)
    common: List[SpecDiff] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "spec_a_name": self.spec_a_name,
            "spec_b_name": self.spec_b_name,
            "total_differences": self.total_differences,
            "added": [d.to_dict() for d in self.added],
            "removed": [d.to_dict() for d in self.removed],
            "changed": [d.to_dict() for d in self.changed],
            "common": [d.to_dict() for d in self.common],
        }


# ---------------------------------------------------------------------------
# Helper: build SpecRequirement shorthand
# ---------------------------------------------------------------------------

def _req(category, field_name, operator, value, unit="", description="",
         severity="error", required=True):
    """Shorthand factory for SpecRequirement."""
    return SpecRequirement(
        category=category,
        field_name=field_name,
        operator=operator,
        value=value,
        unit=unit,
        description=description,
        severity=severity,
        required=required,
    )


# ---------------------------------------------------------------------------
# Built-In Delivery Specs
# ---------------------------------------------------------------------------

BUILT_IN_SPECS: Dict[str, DeliveryProfile] = {}


def _register_built_in(profile: DeliveryProfile):
    """Register a built-in spec."""
    profile.built_in = True
    BUILT_IN_SPECS[profile.name] = profile


# --- Netflix ---
_register_built_in(DeliveryProfile(
    name="netflix",
    display_name="Netflix",
    description="Netflix delivery specifications for original and licensed content",
    platform="netflix",
    version="2024.1",
    requirements=[
        _req("video", "codec", "in", ["h264", "h265", "prores"],
             description="Video codec must be H.264, H.265, or ProRes"),
        _req("video", "resolution_min_width", "gte", 1920, "px",
             description="Minimum 1920px width (Full HD)"),
        _req("video", "resolution_min_height", "gte", 1080, "px",
             description="Minimum 1080px height (Full HD)"),
        _req("video", "frame_rate", "in", [23.976, 24, 25, 29.97, 30, 50, 59.94, 60],
             description="Frame rate must be standard broadcast/cinema rate"),
        _req("video", "bitrate_min", "gte", 10_000_000, "bps",
             description="Minimum video bitrate 10 Mbps"),
        _req("video", "bitrate_max", "lte", 100_000_000, "bps",
             description="Maximum video bitrate 100 Mbps"),
        _req("video", "color_space", "in", ["bt709", "bt2020"],
             description="Color space must be BT.709 or BT.2020"),
        _req("video", "chroma_subsampling", "in", ["4:2:0", "4:2:2", "4:4:4"],
             description="Chroma subsampling 4:2:0 or higher"),
        _req("video", "bit_depth", "gte", 8, "bits",
             description="Minimum 8-bit depth (10-bit preferred for HDR)"),
        _req("audio", "codec", "in", ["aac", "pcm_s24le", "pcm_s16le", "eac3"],
             description="Audio codec must be AAC, PCM, or E-AC-3"),
        _req("audio", "sample_rate", "in", [48000], "Hz",
             description="Audio sample rate must be 48kHz"),
        _req("audio", "channels_min", "gte", 2,
             description="Minimum stereo audio"),
        _req("audio", "loudness_integrated", "range", [-27, -14], "LUFS",
             description="Integrated loudness between -27 and -14 LUFS"),
        _req("audio", "loudness_true_peak", "lte", -1.0, "dBTP",
             description="True peak must not exceed -1.0 dBTP"),
        _req("container", "format", "in", ["mov", "mxf", "mp4"],
             description="Container must be MOV, MXF, or MP4"),
        _req("container", "timecode", "eq", True,
             description="Timecode track required", severity="warning"),
        _req("subtitle", "format", "in", ["ttml", "srt", "dfxp"],
             description="Subtitle format must be TTML, SRT, or DFXP",
             required=False),
    ],
    metadata={"spec_url": "https://partnerhelp.netflixstudios.com/"},
))

# --- YouTube ---
_register_built_in(DeliveryProfile(
    name="youtube",
    display_name="YouTube",
    description="YouTube recommended upload specifications",
    platform="youtube",
    version="2024.1",
    requirements=[
        _req("video", "codec", "in", ["h264", "h265", "vp9", "av1"],
             description="Video codec must be H.264, H.265, VP9, or AV1"),
        _req("video", "resolution_min_width", "gte", 1280, "px",
             description="Minimum 720p width recommended"),
        _req("video", "frame_rate", "in",
             [24, 25, 30, 48, 50, 60],
             description="Standard frame rate required"),
        _req("video", "bitrate_min", "gte", 5_000_000, "bps",
             description="Minimum 5 Mbps for 1080p", severity="warning"),
        _req("video", "bitrate_max", "lte", 200_000_000, "bps",
             description="Maximum 200 Mbps"),
        _req("audio", "codec", "in", ["aac", "opus", "vorbis"],
             description="Audio codec must be AAC, Opus, or Vorbis"),
        _req("audio", "sample_rate", "in", [44100, 48000], "Hz",
             description="Audio sample rate 44.1kHz or 48kHz"),
        _req("audio", "channels_min", "gte", 1,
             description="At least mono audio"),
        _req("audio", "bitrate_min", "gte", 128_000, "bps",
             description="Minimum audio bitrate 128kbps", severity="warning"),
        _req("container", "format", "in", ["mp4", "mov", "webm", "mkv"],
             description="Container must be MP4, MOV, WebM, or MKV"),
    ],
    metadata={"spec_url": "https://support.google.com/youtube/answer/1722171"},
))

# --- Apple TV+ ---
_register_built_in(DeliveryProfile(
    name="apple_tv_plus",
    display_name="Apple TV+",
    description="Apple TV+ delivery specifications",
    platform="apple_tv_plus",
    version="2024.1",
    requirements=[
        _req("video", "codec", "in", ["h265", "prores", "prores_hq"],
             description="H.265 or ProRes required"),
        _req("video", "resolution_min_width", "gte", 3840, "px",
             description="Minimum 4K UHD width"),
        _req("video", "resolution_min_height", "gte", 2160, "px",
             description="Minimum 4K UHD height"),
        _req("video", "frame_rate", "in", [23.976, 24, 25, 29.97, 50, 59.94],
             description="Standard cinema/broadcast frame rate"),
        _req("video", "bit_depth", "gte", 10, "bits",
             description="10-bit minimum for HDR content"),
        _req("video", "color_space", "in", ["bt2020"],
             description="BT.2020 color space for HDR"),
        _req("video", "hdr_format", "in", ["dolby_vision", "hdr10", "hlg"],
             description="Dolby Vision, HDR10, or HLG", severity="warning"),
        _req("audio", "codec", "in", ["pcm_s24le", "aac", "eac3", "ac4"],
             description="PCM, AAC, E-AC-3, or AC-4"),
        _req("audio", "sample_rate", "eq", 48000, "Hz",
             description="48kHz sample rate required"),
        _req("audio", "channels_min", "gte", 2,
             description="Minimum stereo; 5.1/7.1/Atmos preferred"),
        _req("audio", "loudness_integrated", "range", [-27, -14], "LUFS",
             description="Integrated loudness -27 to -14 LUFS"),
        _req("container", "format", "in", ["mov", "mxf"],
             description="MOV or MXF container"),
        _req("container", "timecode", "eq", True,
             description="Timecode track required"),
    ],
    metadata={"spec_url": "https://help.apple.com/itc/contentspec/"},
))

# --- Amazon ---
_register_built_in(DeliveryProfile(
    name="amazon",
    display_name="Amazon Prime Video",
    description="Amazon Prime Video direct delivery specifications",
    platform="amazon",
    version="2024.1",
    requirements=[
        _req("video", "codec", "in", ["h264", "h265", "prores"],
             description="H.264, H.265, or ProRes"),
        _req("video", "resolution_min_width", "gte", 1920, "px",
             description="Minimum Full HD width"),
        _req("video", "resolution_min_height", "gte", 1080, "px",
             description="Minimum Full HD height"),
        _req("video", "frame_rate", "in", [23.976, 24, 25, 29.97, 30],
             description="Standard frame rate"),
        _req("video", "bitrate_min", "gte", 15_000_000, "bps",
             description="Minimum 15 Mbps for HD"),
        _req("video", "bitrate_max", "lte", 100_000_000, "bps",
             description="Maximum 100 Mbps"),
        _req("audio", "codec", "in", ["aac", "pcm_s24le", "pcm_s16le", "eac3", "ac3"],
             description="AAC, PCM, E-AC-3, or AC-3"),
        _req("audio", "sample_rate", "eq", 48000, "Hz",
             description="48kHz required"),
        _req("audio", "channels_min", "gte", 2,
             description="Minimum stereo"),
        _req("audio", "loudness_integrated", "range", [-27, -14], "LUFS",
             description="Integrated loudness -27 to -14 LUFS"),
        _req("container", "format", "in", ["mov", "mxf", "mp4"],
             description="MOV, MXF, or MP4"),
        _req("container", "timecode", "eq", True,
             description="Timecode preferred", severity="warning"),
        _req("subtitle", "format", "in", ["ttml", "srt", "stl"],
             description="TTML, SRT, or STL subtitles",
             required=False),
    ],
    metadata={"spec_url": "https://videodirect.amazon.com/"},
))

# --- Broadcast EBU ---
_register_built_in(DeliveryProfile(
    name="broadcast_ebu",
    display_name="Broadcast (EBU R128)",
    description="European Broadcasting Union R128 broadcast specifications",
    platform="broadcast",
    version="2024.1",
    requirements=[
        _req("video", "codec", "in", ["h264", "mpeg2", "prores", "dnxhd"],
             description="H.264, MPEG-2, ProRes, or DNxHD"),
        _req("video", "resolution_min_width", "gte", 1920, "px",
             description="Full HD minimum"),
        _req("video", "resolution_min_height", "gte", 1080, "px",
             description="Full HD minimum"),
        _req("video", "frame_rate", "in", [25, 50],
             description="25 or 50fps for PAL broadcast"),
        _req("video", "bitrate_min", "gte", 35_000_000, "bps",
             description="Minimum 35 Mbps for broadcast"),
        _req("video", "interlacing", "in", ["progressive", "tff"],
             description="Progressive or top-field-first"),
        _req("audio", "codec", "in", ["pcm_s24le", "pcm_s16le"],
             description="PCM audio required for broadcast"),
        _req("audio", "sample_rate", "eq", 48000, "Hz",
             description="48kHz required"),
        _req("audio", "channels_min", "gte", 2,
             description="Minimum stereo"),
        _req("audio", "loudness_integrated", "eq", -23.0, "LUFS",
             description="Integrated loudness must be -23 LUFS (EBU R128)"),
        _req("audio", "loudness_range", "lte", 20.0, "LU",
             description="Loudness range max 20 LU"),
        _req("audio", "loudness_true_peak", "lte", -1.0, "dBTP",
             description="True peak max -1.0 dBTP"),
        _req("container", "format", "in", ["mxf"],
             description="MXF container required for broadcast"),
        _req("container", "timecode", "eq", True,
             description="Timecode track required"),
    ],
    metadata={"spec_url": "https://tech.ebu.ch/r128"},
))

# --- DCP ---
_register_built_in(DeliveryProfile(
    name="dcp",
    display_name="DCP (Digital Cinema)",
    description="Digital Cinema Package specifications per SMPTE DCI",
    platform="theatrical",
    version="2024.1",
    requirements=[
        _req("video", "codec", "eq", "jpeg2000",
             description="JPEG2000 required for DCP"),
        _req("video", "resolution", "in",
             ["2048x1080", "4096x2160", "1998x1080", "3996x2160"],
             description="DCI 2K or 4K resolution"),
        _req("video", "frame_rate", "in", [24, 48],
             description="24 or 48fps for theatrical"),
        _req("video", "color_space", "eq", "xyz",
             description="XYZ color space"),
        _req("video", "bit_depth", "eq", 12, "bits",
             description="12-bit JPEG2000"),
        _req("audio", "codec", "eq", "pcm_s24le",
             description="24-bit PCM required"),
        _req("audio", "sample_rate", "eq", 48000, "Hz",
             description="48kHz required"),
        _req("audio", "channels_min", "gte", 6,
             description="Minimum 5.1 surround"),
        _req("container", "format", "eq", "mxf",
             description="MXF container for essence tracks"),
    ],
    metadata={"spec_ref": "SMPTE ST 429"},
))

# --- Theatrical (general) ---
_register_built_in(DeliveryProfile(
    name="theatrical",
    display_name="Theatrical (General)",
    description="General theatrical projection requirements",
    platform="theatrical",
    version="2024.1",
    requirements=[
        _req("video", "codec", "in", ["jpeg2000", "prores_4444", "prores_hq"],
             description="JPEG2000 or ProRes 4444/HQ"),
        _req("video", "resolution_min_width", "gte", 2048, "px",
             description="Minimum 2K DCI width"),
        _req("video", "resolution_min_height", "gte", 1080, "px",
             description="Minimum 2K DCI height"),
        _req("video", "frame_rate", "in", [24, 48],
             description="24 or 48fps"),
        _req("video", "bit_depth", "gte", 10, "bits",
             description="Minimum 10-bit depth"),
        _req("audio", "codec", "in", ["pcm_s24le"],
             description="24-bit PCM"),
        _req("audio", "sample_rate", "eq", 48000, "Hz",
             description="48kHz"),
        _req("audio", "channels_min", "gte", 6,
             description="Minimum 5.1 surround"),
        _req("audio", "loudness_true_peak", "lte", -1.0, "dBTP",
             description="True peak max -1.0 dBTP"),
        _req("container", "format", "in", ["mxf", "dcp"],
             description="MXF or DCP package"),
        _req("container", "timecode", "eq", True,
             description="Timecode required"),
    ],
    metadata={"notes": "General theatrical requirements; DCP spec is stricter"},
))


# ---------------------------------------------------------------------------
# Spec Management Functions
# ---------------------------------------------------------------------------

def get_spec(name: str) -> Optional[dict]:
    """Get a delivery spec by name. Checks built-in then custom specs.

    Args:
        name: Spec name (case-insensitive).

    Returns:
        DeliveryProfile.to_dict() or None if not found.
    """
    name_lower = name.lower().strip()

    # Check built-in specs
    if name_lower in BUILT_IN_SPECS:
        return BUILT_IN_SPECS[name_lower].to_dict()

    # Check custom specs on disk
    custom_path = os.path.join(_CUSTOM_SPECS_DIR, f"{name_lower}.json")
    if os.path.isfile(custom_path):
        try:
            with open(custom_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load custom spec '%s': %s", name, exc)
    return None


def list_specs() -> List[dict]:
    """List all available delivery specs (built-in + custom).

    Returns:
        List of dicts with name, display_name, platform, built_in, description.
    """
    specs = []

    # Built-in specs
    for name, profile in sorted(BUILT_IN_SPECS.items()):
        specs.append({
            "name": profile.name,
            "display_name": profile.display_name,
            "platform": profile.platform,
            "built_in": True,
            "description": profile.description,
            "requirement_count": len(profile.requirements),
        })

    # Custom specs
    _ensure_specs_dir()
    try:
        for filename in sorted(os.listdir(_CUSTOM_SPECS_DIR)):
            if not filename.endswith(".json"):
                continue
            filepath = os.path.join(_CUSTOM_SPECS_DIR, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                specs.append({
                    "name": data.get("name", filename.replace(".json", "")),
                    "display_name": data.get("display_name", ""),
                    "platform": data.get("platform", "custom"),
                    "built_in": False,
                    "description": data.get("description", ""),
                    "requirement_count": len(data.get("requirements", [])),
                })
            except (json.JSONDecodeError, OSError):
                continue
    except OSError:
        pass

    return specs


def create_custom_spec(
    name: str,
    requirements: Optional[List[dict]] = None,
    display_name: str = "",
    description: str = "",
    platform: str = "custom",
    metadata: Optional[Dict] = None,
) -> dict:
    """Create and persist a custom delivery spec.

    Args:
        name: Unique spec name (lowercase, no spaces).
        requirements: List of requirement dicts.
        display_name: Human-readable name.
        description: Spec description.
        platform: Platform identifier.
        metadata: Additional metadata dict.

    Returns:
        The created DeliveryProfile.to_dict().

    Raises:
        ValueError: If name conflicts with built-in spec or is invalid.
    """
    name_clean = name.lower().strip().replace(" ", "_")
    if not name_clean:
        raise ValueError("Spec name cannot be empty")
    if name_clean in BUILT_IN_SPECS:
        raise ValueError(f"Cannot override built-in spec '{name_clean}'")

    # Parse requirements
    parsed_reqs = []
    for r in (requirements or []):
        if isinstance(r, SpecRequirement):
            parsed_reqs.append(r)
        elif isinstance(r, dict):
            parsed_reqs.append(SpecRequirement(**{
                k: v for k, v in r.items()
                if k in SpecRequirement.__dataclass_fields__
            }))

    profile = DeliveryProfile(
        name=name_clean,
        display_name=display_name or name_clean.replace("_", " ").title(),
        description=description,
        platform=platform,
        requirements=parsed_reqs,
        metadata=metadata or {},
        built_in=False,
    )

    # Persist to disk
    _ensure_specs_dir()
    spec_path = os.path.join(_CUSTOM_SPECS_DIR, f"{name_clean}.json")
    try:
        with open(spec_path, "w", encoding="utf-8") as f:
            json.dump(profile.to_dict(), f, indent=2, default=str)
        logger.info("Custom delivery spec created: %s -> %s", name_clean, spec_path)
    except OSError as exc:
        raise RuntimeError(f"Failed to save spec: {exc}") from exc

    return profile.to_dict()


def delete_custom_spec(name: str) -> bool:
    """Delete a custom delivery spec by name.

    Returns:
        True if deleted, False if not found.

    Raises:
        ValueError: If attempting to delete a built-in spec.
    """
    name_lower = name.lower().strip()
    if name_lower in BUILT_IN_SPECS:
        raise ValueError(f"Cannot delete built-in spec '{name_lower}'")

    spec_path = os.path.join(_CUSTOM_SPECS_DIR, f"{name_lower}.json")
    if os.path.isfile(spec_path):
        try:
            os.unlink(spec_path)
            logger.info("Deleted custom spec: %s", name_lower)
            return True
        except OSError:
            return False
    return False


def export_spec(name: str, output_path: str) -> str:
    """Export a delivery spec to a JSON file.

    Args:
        name: Spec name.
        output_path: Destination file path.

    Returns:
        Path to exported file.
    """
    spec = get_spec(name)
    if spec is None:
        raise ValueError(f"Spec not found: {name}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, default=str)
    logger.info("Exported spec '%s' to %s", name, output_path)
    return output_path


def import_spec(filepath: str) -> dict:
    """Import a delivery spec from a JSON file.

    Args:
        filepath: Path to JSON spec file.

    Returns:
        The imported DeliveryProfile.to_dict().
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Spec file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    name = data.get("name", "")
    if not name:
        name = os.path.splitext(os.path.basename(filepath))[0]
        data["name"] = name

    return create_custom_spec(
        name=name,
        requirements=data.get("requirements", []),
        display_name=data.get("display_name", ""),
        description=data.get("description", ""),
        platform=data.get("platform", "custom"),
        metadata=data.get("metadata"),
    )


def compare_specs(spec_a: str, spec_b: str) -> dict:
    """Compare two delivery specs and return differences.

    Args:
        spec_a: Name of first spec.
        spec_b: Name of second spec.

    Returns:
        SpecCompareResult.to_dict()

    Raises:
        ValueError: If either spec is not found.
    """
    a_data = get_spec(spec_a)
    b_data = get_spec(spec_b)
    if a_data is None:
        raise ValueError(f"Spec not found: {spec_a}")
    if b_data is None:
        raise ValueError(f"Spec not found: {spec_b}")

    result = SpecCompareResult(
        spec_a_name=spec_a,
        spec_b_name=spec_b,
    )

    # Build requirement maps keyed by (category, field_name)
    a_reqs = {}
    for r in a_data.get("requirements", []):
        key = (r.get("category", ""), r.get("field_name", ""))
        a_reqs[key] = r

    b_reqs = {}
    for r in b_data.get("requirements", []):
        key = (r.get("category", ""), r.get("field_name", ""))
        b_reqs[key] = r

    all_keys = set(a_reqs.keys()) | set(b_reqs.keys())

    for key in sorted(all_keys):
        field_path = f"{key[0]}.{key[1]}"
        in_a = key in a_reqs
        in_b = key in b_reqs

        if in_a and not in_b:
            result.removed.append(SpecDiff(
                field_path=field_path,
                spec_a_value=a_reqs[key].get("value"),
                spec_b_value=None,
                diff_type="removed",
            ))
        elif in_b and not in_a:
            result.added.append(SpecDiff(
                field_path=field_path,
                spec_a_value=None,
                spec_b_value=b_reqs[key].get("value"),
                diff_type="added",
            ))
        else:
            a_val = a_reqs[key].get("value")
            b_val = b_reqs[key].get("value")
            a_op = a_reqs[key].get("operator")
            b_op = b_reqs[key].get("operator")

            if a_val == b_val and a_op == b_op:
                result.common.append(SpecDiff(
                    field_path=field_path,
                    spec_a_value=a_val,
                    spec_b_value=b_val,
                    diff_type="same",
                ))
            else:
                result.changed.append(SpecDiff(
                    field_path=field_path,
                    spec_a_value={"operator": a_op, "value": a_val},
                    spec_b_value={"operator": b_op, "value": b_val},
                    diff_type="changed",
                ))

    result.total_differences = (len(result.added) + len(result.removed)
                                + len(result.changed))
    return result.to_dict()


def suggest_spec(file_info: dict) -> Optional[str]:
    """Auto-suggest a delivery spec based on detected file properties.

    Args:
        file_info: Dict with keys like 'width', 'height', 'codec',
                   'container', 'audio_codec', etc.

    Returns:
        Suggested spec name or None.
    """
    width = file_info.get("width", 0)
    height = file_info.get("height", 0)
    codec = str(file_info.get("codec", "")).lower()
    container = str(file_info.get("container", "")).lower()
    audio_codec = str(file_info.get("audio_codec", "")).lower()

    # DCP detection
    if codec == "jpeg2000" and container == "mxf":
        return "dcp"

    # Broadcast detection (MXF + PCM + 25fps)
    fps = file_info.get("fps", 0)
    if container == "mxf" and "pcm" in audio_codec and fps in (25, 50):
        return "broadcast_ebu"

    # Apple TV+ detection (4K + HDR indicators)
    if width >= 3840 and height >= 2160:
        hdr = file_info.get("hdr_format", "")
        if hdr or codec in ("hevc", "h265"):
            return "apple_tv_plus"

    # Netflix detection (high quality HD+)
    if width >= 1920 and codec in ("h264", "h265", "prores"):
        return "netflix"

    # Amazon detection
    if container in ("mp4", "mov") and width >= 1920:
        return "amazon"

    # YouTube detection (web-friendly codecs)
    if codec in ("h264", "vp9", "av1") and container in ("mp4", "webm"):
        return "youtube"

    return None
