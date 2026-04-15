"""
OpenCut IMF Package Module (70.2)

Interoperable Master Format (IMF) package generation per SMPTE ST 2067.
Generates CPL, OPL, PKL, video MXF (OP1a), audio MXF,
with support for multiple audio languages as supplemental packages.
"""

import hashlib
import json
import logging
import os
import subprocess as _sp
import uuid
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Callable, List, Optional

from opencut.helpers import (
    get_ffmpeg_path,
    get_ffprobe_path,
)

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# IMF Profile Definitions (SMPTE ST 2067)
# ---------------------------------------------------------------------------

IMF_PROFILES = {
    "application_2": {
        "label": "Application 2 (HD/SD)",
        "max_width": 1920,
        "max_height": 1080,
        "video_codecs": ["jpeg2000"],
        "container": "mxf_op1a",
        "color_spaces": ["bt709", "bt601"],
        "frame_rates": [24, 25, 30, 50, 60],
        "audio_codecs": ["pcm_s24le"],
        "audio_sample_rates": [48000, 96000],
        "smpte_spec": "ST 2067-20",
    },
    "application_2e": {
        "label": "Application 2E (UHD/HDR)",
        "max_width": 3840,
        "max_height": 2160,
        "video_codecs": ["jpeg2000", "hevc"],
        "container": "mxf_op1a",
        "color_spaces": ["bt2020", "bt709"],
        "frame_rates": [24, 25, 30, 48, 50, 60],
        "audio_codecs": ["pcm_s24le"],
        "audio_sample_rates": [48000, 96000],
        "smpte_spec": "ST 2067-21",
    },
    "application_4": {
        "label": "Application 4 (Cinema Mezzanine)",
        "max_width": 4096,
        "max_height": 2160,
        "video_codecs": ["jpeg2000"],
        "container": "mxf_op1a",
        "color_spaces": ["xyz", "bt2020"],
        "frame_rates": [24, 48],
        "audio_codecs": ["pcm_s24le"],
        "audio_sample_rates": [48000, 96000],
        "smpte_spec": "ST 2067-40",
    },
    "application_5": {
        "label": "Application 5 (ACES)",
        "max_width": 4096,
        "max_height": 2160,
        "video_codecs": ["jpeg2000"],
        "container": "mxf_op1a",
        "color_spaces": ["aces"],
        "frame_rates": [24, 25, 30, 48, 50, 60],
        "audio_codecs": ["pcm_s24le"],
        "audio_sample_rates": [48000, 96000],
        "smpte_spec": "ST 2067-50",
    },
}

_IMF_NAMESPACES = {
    "cpl": "http://www.smpte-ra.org/schemas/2067-3/2016",
    "core": "http://www.smpte-ra.org/schemas/2067-2/2016",
    "pkl": "http://www.smpte-ra.org/schemas/2067-2/2016#PKL",
    "opl": "http://www.smpte-ra.org/schemas/2067-100/2014",
    "am": "http://www.smpte-ra.org/schemas/429-9/2007/AM",
    "ds": "http://www.w3.org/2000/09/xmldsig#",
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class IMFAudioTrack:
    """Configuration for a single audio track/language."""
    language: str = "en"
    label: str = "English"
    source_path: str = ""
    channels: int = 6
    sample_rate: int = 48000

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class IMFConfig:
    """Configuration for IMF package export."""
    title: str = "Untitled"
    profile: str = "application_2"
    frame_rate: int = 24
    width: int = 1920
    height: int = 1080
    video_codec: str = "jpeg2000"
    audio_tracks: List[IMFAudioTrack] = field(default_factory=list)
    issuer: str = "OpenCut"
    creator: str = "OpenCut IMF Export"
    annotation: str = ""
    content_kind: str = "feature"
    timecode_start: str = "01:00:00:00"
    audio_bit_depth: int = 24
    supplemental: bool = False
    original_cpl_id: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["audio_tracks"] = [t.to_dict() if isinstance(t, IMFAudioTrack) else t
                             for t in self.audio_tracks]
        return d

    def validate(self) -> List[str]:
        """Return list of validation errors."""
        errors = []
        if self.profile not in IMF_PROFILES:
            errors.append(f"Unknown profile '{self.profile}'. "
                          f"Valid: {list(IMF_PROFILES.keys())}")
            return errors
        p = IMF_PROFILES[self.profile]
        if self.width > p["max_width"]:
            errors.append(f"Width {self.width} exceeds max {p['max_width']} "
                          f"for profile {self.profile}")
        if self.height > p["max_height"]:
            errors.append(f"Height {self.height} exceeds max {p['max_height']} "
                          f"for profile {self.profile}")
        if self.video_codec not in p["video_codecs"]:
            errors.append(f"Codec '{self.video_codec}' not supported for "
                          f"profile {self.profile}. Valid: {p['video_codecs']}")
        if self.frame_rate not in p["frame_rates"]:
            errors.append(f"Frame rate {self.frame_rate} not supported for "
                          f"profile {self.profile}. Valid: {p['frame_rates']}")
        if self.audio_bit_depth not in (16, 24):
            errors.append(f"audio_bit_depth must be 16 or 24, "
                          f"got {self.audio_bit_depth}")
        if self.supplemental and not self.original_cpl_id:
            errors.append("supplemental=True requires original_cpl_id")
        return errors


@dataclass
class IMFAsset:
    """A single asset in the IMF package."""
    uuid: str = ""
    filename: str = ""
    asset_type: str = ""
    file_size: int = 0
    hash_value: str = ""
    original_filename: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class IMFResult:
    """Result of IMF package export."""
    output_dir: str = ""
    title: str = ""
    profile: str = ""
    profile_label: str = ""
    frame_rate: int = 24
    resolution: str = ""
    duration_seconds: float = 0.0
    total_frames: int = 0
    assets: List[IMFAsset] = field(default_factory=list)
    video_mxf_path: str = ""
    audio_mxf_paths: List[str] = field(default_factory=list)
    cpl_path: str = ""
    opl_path: str = ""
    pkl_path: str = ""
    assetmap_path: str = ""
    total_size_bytes: int = 0
    supplemental: bool = False

    def to_dict(self) -> dict:
        return {
            "output_dir": self.output_dir,
            "title": self.title,
            "profile": self.profile,
            "profile_label": self.profile_label,
            "frame_rate": self.frame_rate,
            "resolution": self.resolution,
            "duration_seconds": round(self.duration_seconds, 3),
            "total_frames": self.total_frames,
            "assets": [a.to_dict() for a in self.assets],
            "video_mxf_path": self.video_mxf_path,
            "audio_mxf_paths": self.audio_mxf_paths,
            "cpl_path": self.cpl_path,
            "opl_path": self.opl_path,
            "pkl_path": self.pkl_path,
            "assetmap_path": self.assetmap_path,
            "total_size_bytes": self.total_size_bytes,
            "supplemental": self.supplemental,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _imf_uuid() -> str:
    """Generate a URN-prefixed UUID4 for IMF assets."""
    return f"urn:uuid:{uuid.uuid4()}"


def _sha1_file(filepath: str) -> str:
    """Compute SHA-1 hash of a file as base64."""
    import base64
    sha = hashlib.sha1()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            sha.update(chunk)
    return base64.b64encode(sha.digest()).decode("ascii")


def _file_size(filepath: str) -> int:
    """Return file size in bytes, 0 on error."""
    try:
        return os.path.getsize(filepath)
    except OSError:
        return 0


def _probe_source(video_path: str) -> dict:
    """Probe source video for stream details."""
    cmd = [
        get_ffprobe_path(), "-v", "quiet",
        "-show_entries",
        "stream=width,height,r_frame_rate,codec_name,pix_fmt,duration,"
        "nb_frames,channels,sample_rate,codec_type",
        "-show_entries", "format=duration,nb_streams",
        "-of", "json", video_path,
    ]
    result = _sp.run(cmd, capture_output=True, timeout=60)
    info = {
        "width": 1920, "height": 1080, "fps": 24.0, "duration": 0.0,
        "codec": "unknown", "pix_fmt": "unknown", "nb_frames": 0,
        "has_audio": False, "audio_channels": 2, "audio_sample_rate": 48000,
    }
    if result.returncode != 0:
        return info
    try:
        data = json.loads(result.stdout.decode())
        streams = data.get("streams", [])
        fmt = data.get("format", {})
        for s in streams:
            if s.get("codec_type") == "video":
                info["width"] = int(s.get("width", 1920))
                info["height"] = int(s.get("height", 1080))
                rfr = s.get("r_frame_rate", "24/1")
                if "/" in str(rfr):
                    num, den = rfr.split("/")
                    info["fps"] = float(num) / float(den) if float(den) else 24.0
                else:
                    info["fps"] = float(rfr)
                info["codec"] = s.get("codec_name", "unknown")
                info["pix_fmt"] = s.get("pix_fmt", "unknown")
                info["nb_frames"] = int(s.get("nb_frames", 0) or 0)
                stream_dur = float(s.get("duration", 0) or 0)
                fmt_dur = float(fmt.get("duration", 0) or 0)
                info["duration"] = stream_dur or fmt_dur
            elif s.get("codec_type") == "audio":
                info["has_audio"] = True
                info["audio_channels"] = int(s.get("channels", 2) or 2)
                info["audio_sample_rate"] = int(s.get("sample_rate", 48000) or 48000)
    except (ValueError, KeyError, TypeError) as exc:
        logger.warning("IMF probe parse error: %s", exc)
    return info


def _parse_timecode(tc_str: str, frame_rate: int) -> int:
    """Parse SMPTE timecode string to frame number."""
    parts = tc_str.replace(";", ":").split(":")
    if len(parts) != 4:
        return 0
    try:
        hh, mm, ss, ff = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        return ((hh * 3600 + mm * 60 + ss) * frame_rate) + ff
    except ValueError:
        return 0


def _frames_to_timecode(frames: int, frame_rate: int) -> str:
    """Convert frame number to SMPTE timecode string."""
    fr = max(frame_rate, 1)
    ff = frames % fr
    total_seconds = frames // fr
    ss = total_seconds % 60
    mm = (total_seconds // 60) % 60
    hh = total_seconds // 3600
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


# ---------------------------------------------------------------------------
# Video MXF (OP1a) generation
# ---------------------------------------------------------------------------

def _encode_video_mxf(
    video_path: str,
    output_mxf: str,
    config: IMFConfig,
    source_info: dict,
    on_progress: Optional[Callable] = None,
) -> str:
    """Encode video to MXF OP1a with proper timecode track."""
    if on_progress:
        on_progress(10, "Encoding video to MXF OP1a...")

    IMF_PROFILES[config.profile]

    vf_parts = []
    if source_info["width"] != config.width or source_info["height"] != config.height:
        vf_parts.append(f"scale={config.width}:{config.height}:flags=lanczos")
    vf_parts.append(f"fps={config.frame_rate}")
    vf_chain = ",".join(vf_parts) if vf_parts else None

    codec_map = {
        "jpeg2000": "libopenjpeg",
        "hevc": "libx265",
    }
    ffmpeg_codec = codec_map.get(config.video_codec, "libopenjpeg")

    cmd = [
        get_ffmpeg_path(), "-y",
        "-i", video_path,
        "-an",
    ]
    if vf_chain:
        cmd.extend(["-vf", vf_chain])
    cmd.extend([
        "-c:v", ffmpeg_codec,
        "-r", str(config.frame_rate),
        "-timecode", config.timecode_start,
        "-f", "mxf_opatom" if config.video_codec == "jpeg2000" else "mxf",
        output_mxf,
    ])

    logger.info("IMF video encode: %s -> %s", video_path, output_mxf)
    proc = _sp.run(cmd, capture_output=True, timeout=7200)
    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors="replace")[-2000:]
        raise RuntimeError(f"IMF video MXF encode failed: {stderr}")

    if on_progress:
        on_progress(40, "Video MXF encode complete")
    return output_mxf


# ---------------------------------------------------------------------------
# Audio MXF generation
# ---------------------------------------------------------------------------

def _encode_audio_mxf(
    audio_source: str,
    output_mxf: str,
    config: IMFConfig,
    track: Optional[IMFAudioTrack] = None,
    on_progress: Optional[Callable] = None,
    progress_base: int = 45,
) -> str:
    """Encode audio to MXF OP1a with PCM essence."""
    channels = track.channels if track else 6
    sample_rate = track.sample_rate if track else 48000
    label = track.label if track else "Default"

    if on_progress:
        on_progress(progress_base, f"Encoding audio MXF: {label}...")

    cmd = [
        get_ffmpeg_path(), "-y",
        "-i", audio_source,
        "-vn",
        "-c:a", f"pcm_s{config.audio_bit_depth}le",
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-timecode", config.timecode_start,
        "-f", "mxf",
        output_mxf,
    ]

    proc = _sp.run(cmd, capture_output=True, timeout=3600)
    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors="replace")[-1000:]
        raise RuntimeError(f"IMF audio MXF encode failed ({label}): {stderr}")

    if on_progress:
        on_progress(progress_base + 10, f"Audio MXF complete: {label}")
    return output_mxf


# ---------------------------------------------------------------------------
# CPL XML generation (IMF)
# ---------------------------------------------------------------------------

def _generate_imf_cpl(
    cpl_path: str,
    cpl_uuid: str,
    config: IMFConfig,
    video_uuid: str,
    audio_uuids: List[str],
    duration_frames: int,
) -> str:
    """Generate IMF Composition Playlist XML per SMPTE ST 2067-3."""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    edit_rate_str = f"{config.frame_rate}/1"

    root = ET.Element("CompositionPlaylist", xmlns=_IMF_NAMESPACES["cpl"])
    root.set("xmlns:core", _IMF_NAMESPACES["core"])

    ET.SubElement(root, "Id").text = cpl_uuid
    ET.SubElement(root, "AnnotationText").text = (
        config.annotation or config.title
    )
    ET.SubElement(root, "IssueDate").text = now_str
    ET.SubElement(root, "Issuer").text = config.issuer
    ET.SubElement(root, "Creator").text = config.creator
    ET.SubElement(root, "ContentTitle").text = config.title
    ET.SubElement(root, "ContentKind").text = config.content_kind
    ET.SubElement(root, "EditRate").text = edit_rate_str

    _parse_timecode(config.timecode_start, config.frame_rate)
    ET.SubElement(root, "CompositionTimecode").text = config.timecode_start

    # Segment list
    seg_list = ET.SubElement(root, "SegmentList")
    segment = ET.SubElement(seg_list, "Segment")
    ET.SubElement(segment, "Id").text = _imf_uuid()

    seq_list = ET.SubElement(segment, "SequenceList")

    # Video sequence
    vid_seq = ET.SubElement(seq_list, "MainImageSequence")
    ET.SubElement(vid_seq, "Id").text = _imf_uuid()
    ET.SubElement(vid_seq, "TrackId").text = _imf_uuid()

    vid_res = ET.SubElement(vid_seq, "ResourceList")
    vid_resource = ET.SubElement(vid_res, "Resource")
    ET.SubElement(vid_resource, "Id").text = _imf_uuid()
    ET.SubElement(vid_resource, "TrackFileId").text = video_uuid
    ET.SubElement(vid_resource, "EditRate").text = edit_rate_str
    ET.SubElement(vid_resource, "IntrinsicDuration").text = str(duration_frames)
    ET.SubElement(vid_resource, "EntryPoint").text = "0"
    ET.SubElement(vid_resource, "SourceDuration").text = str(duration_frames)

    # Audio sequences
    for idx, audio_uuid in enumerate(audio_uuids):
        track = (config.audio_tracks[idx]
                 if idx < len(config.audio_tracks) else None)
        lang = track.language if track else "en"

        aud_seq = ET.SubElement(seq_list, "MainAudioSequence")
        ET.SubElement(aud_seq, "Id").text = _imf_uuid()
        ET.SubElement(aud_seq, "TrackId").text = _imf_uuid()
        ET.SubElement(aud_seq, "Language").text = lang

        aud_res = ET.SubElement(aud_seq, "ResourceList")
        aud_resource = ET.SubElement(aud_res, "Resource")
        ET.SubElement(aud_resource, "Id").text = _imf_uuid()
        ET.SubElement(aud_resource, "TrackFileId").text = audio_uuid
        ET.SubElement(aud_resource, "EditRate").text = edit_rate_str
        ET.SubElement(aud_resource, "IntrinsicDuration").text = str(duration_frames)
        ET.SubElement(aud_resource, "EntryPoint").text = "0"
        ET.SubElement(aud_resource, "SourceDuration").text = str(duration_frames)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(cpl_path, xml_declaration=True, encoding="UTF-8")

    logger.info("IMF CPL written: %s", cpl_path)
    return cpl_path


# ---------------------------------------------------------------------------
# OPL XML generation
# ---------------------------------------------------------------------------

def _generate_opl(
    opl_path: str,
    opl_uuid: str,
    cpl_uuid: str,
    config: IMFConfig,
) -> str:
    """Generate Output Profile List XML per SMPTE ST 2067-100."""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    profile = IMF_PROFILES[config.profile]

    root = ET.Element("OutputProfileList", xmlns=_IMF_NAMESPACES["opl"])
    ET.SubElement(root, "Id").text = opl_uuid
    ET.SubElement(root, "AnnotationText").text = f"OPL for {config.title}"
    ET.SubElement(root, "IssueDate").text = now_str
    ET.SubElement(root, "Issuer").text = config.issuer
    ET.SubElement(root, "Creator").text = config.creator

    # Composition reference
    comp_list = ET.SubElement(root, "CompositionPlaylistList")
    comp_ref = ET.SubElement(comp_list, "CompositionPlaylistReference")
    ET.SubElement(comp_ref, "Id").text = cpl_uuid

    # Output profile
    profile_list = ET.SubElement(root, "OutputProfileList")
    out_profile = ET.SubElement(profile_list, "OutputProfile")
    ET.SubElement(out_profile, "Id").text = _imf_uuid()
    ET.SubElement(out_profile, "AnnotationText").text = profile["label"]

    # Video constraints
    vid_char = ET.SubElement(out_profile, "VideoCharacteristic")
    ET.SubElement(vid_char, "MaxWidth").text = str(config.width)
    ET.SubElement(vid_char, "MaxHeight").text = str(config.height)
    ET.SubElement(vid_char, "FrameRate").text = f"{config.frame_rate}/1"

    # Audio constraints
    aud_char = ET.SubElement(out_profile, "AudioCharacteristic")
    ET.SubElement(aud_char, "SampleRate").text = "48000"
    ET.SubElement(aud_char, "BitDepth").text = str(config.audio_bit_depth)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(opl_path, xml_declaration=True, encoding="UTF-8")

    logger.info("IMF OPL written: %s", opl_path)
    return opl_path


# ---------------------------------------------------------------------------
# PKL XML generation (IMF)
# ---------------------------------------------------------------------------

def _generate_imf_pkl(
    pkl_path: str,
    pkl_uuid: str,
    config: IMFConfig,
    assets: List[IMFAsset],
) -> str:
    """Generate IMF Packing List XML."""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    root = ET.Element("PackingList", xmlns=_IMF_NAMESPACES["pkl"])
    ET.SubElement(root, "Id").text = pkl_uuid
    ET.SubElement(root, "AnnotationText").text = config.title
    ET.SubElement(root, "IssueDate").text = now_str
    ET.SubElement(root, "Issuer").text = config.issuer
    ET.SubElement(root, "Creator").text = config.creator

    asset_list = ET.SubElement(root, "AssetList")
    for asset in assets:
        a_el = ET.SubElement(asset_list, "Asset")
        ET.SubElement(a_el, "Id").text = asset.uuid
        ET.SubElement(a_el, "AnnotationText").text = asset.original_filename or asset.filename

        if asset.asset_type in ("video_mxf", "audio_mxf"):
            mime = "application/mxf"
        elif asset.asset_type in ("cpl", "opl"):
            mime = "text/xml"
        else:
            mime = "application/octet-stream"

        ET.SubElement(a_el, "Hash").text = asset.hash_value
        ET.SubElement(a_el, "Size").text = str(asset.file_size)
        ET.SubElement(a_el, "Type").text = mime
        if asset.original_filename:
            ET.SubElement(a_el, "OriginalFileName").text = asset.original_filename

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(pkl_path, xml_declaration=True, encoding="UTF-8")

    logger.info("IMF PKL written: %s", pkl_path)
    return pkl_path


# ---------------------------------------------------------------------------
# ASSETMAP generation (IMF)
# ---------------------------------------------------------------------------

def _generate_imf_assetmap(
    assetmap_path: str,
    pkl_uuid: str,
    assets: List[IMFAsset],
    config: IMFConfig,
) -> str:
    """Generate IMF ASSETMAP XML."""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    root = ET.Element("AssetMap", xmlns=_IMF_NAMESPACES["am"])
    ET.SubElement(root, "Id").text = _imf_uuid()
    ET.SubElement(root, "Creator").text = config.creator
    ET.SubElement(root, "VolumeCount").text = "1"
    ET.SubElement(root, "IssueDate").text = now_str
    ET.SubElement(root, "Issuer").text = config.issuer

    asset_list = ET.SubElement(root, "AssetList")

    for asset in assets:
        a_el = ET.SubElement(asset_list, "Asset")
        ET.SubElement(a_el, "Id").text = asset.uuid
        chunk_list = ET.SubElement(a_el, "ChunkList")
        chunk = ET.SubElement(chunk_list, "Chunk")
        ET.SubElement(chunk, "Path").text = asset.filename
        ET.SubElement(chunk, "VolumeIndex").text = "1"
        ET.SubElement(chunk, "Offset").text = "0"
        ET.SubElement(chunk, "Length").text = str(asset.file_size)

    # PKL asset entry
    pkl_el = ET.SubElement(asset_list, "Asset")
    ET.SubElement(pkl_el, "Id").text = pkl_uuid
    pkl_chunks = ET.SubElement(pkl_el, "ChunkList")
    pkl_chunk = ET.SubElement(pkl_chunks, "Chunk")
    ET.SubElement(pkl_chunk, "Path").text = "PKL.xml"
    ET.SubElement(pkl_chunk, "VolumeIndex").text = "1"
    ET.SubElement(pkl_chunk, "Offset").text = "0"

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(assetmap_path, xml_declaration=True, encoding="UTF-8")

    logger.info("IMF ASSETMAP written: %s", assetmap_path)
    return assetmap_path


# ---------------------------------------------------------------------------
# SMPTE ST 2067 Validation
# ---------------------------------------------------------------------------

def _validate_imf_constraints(config: IMFConfig, source_info: dict) -> List[str]:
    """Validate source media against IMF profile constraints."""
    warnings = []
    profile = IMF_PROFILES.get(config.profile)
    if not profile:
        return [f"Unknown profile: {config.profile}"]

    if source_info["width"] > profile["max_width"]:
        warnings.append(
            f"Source width {source_info['width']} exceeds profile max "
            f"{profile['max_width']}; will be scaled down"
        )
    if source_info["height"] > profile["max_height"]:
        warnings.append(
            f"Source height {source_info['height']} exceeds profile max "
            f"{profile['max_height']}; will be scaled down"
        )

    src_fps = round(source_info["fps"])
    if src_fps not in profile["frame_rates"]:
        warnings.append(
            f"Source frame rate ~{source_info['fps']:.2f} not in profile "
            f"allowed rates {profile['frame_rates']}; will be converted"
        )

    return warnings


# ---------------------------------------------------------------------------
# Main Export Function
# ---------------------------------------------------------------------------

def export_imf(
    video_path: str,
    output_dir: str,
    config: Optional[IMFConfig] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Export a video file as an IMF package.

    Args:
        video_path: Path to source video file.
        output_dir: Directory where IMF package folder will be created.
        config: IMFConfig with export settings. Uses defaults if None.
        on_progress: Optional callback ``(percent, message)``.

    Returns:
        IMFResult.to_dict()
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Source video not found: {video_path}")

    if config is None:
        config = IMFConfig()

    errors = config.validate()
    if errors:
        raise ValueError(f"IMFConfig validation failed: {'; '.join(errors)}")

    profile = IMF_PROFILES[config.profile]
    title_safe = "".join(c if c.isalnum() or c in " _-" else "_"
                         for c in config.title).strip()
    pkg_type = "SUPP" if config.supplemental else "OV"
    imf_dirname = f"{title_safe}_IMF_{pkg_type}"
    imf_dir = os.path.join(output_dir, imf_dirname)
    os.makedirs(imf_dir, exist_ok=True)

    if on_progress:
        on_progress(2, f"Probing source: {os.path.basename(video_path)}")

    source_info = _probe_source(video_path)
    duration = source_info["duration"]
    total_frames = int(duration * config.frame_rate)
    if total_frames < 1:
        total_frames = max(source_info["nb_frames"], 1)

    # Validate against profile constraints
    constraint_warnings = _validate_imf_constraints(config, source_info)
    for w in constraint_warnings:
        logger.warning("IMF constraint: %s", w)

    # Generate UUIDs
    video_uuid = _imf_uuid()
    cpl_uuid = _imf_uuid()
    opl_uuid = _imf_uuid()
    pkl_uuid = _imf_uuid()

    # Encode video MXF
    video_mxf_name = f"{title_safe}_video.mxf"
    video_mxf_path = os.path.join(imf_dir, video_mxf_name)
    _encode_video_mxf(video_path, video_mxf_path, config, source_info, on_progress)

    # Encode audio MXF(s)
    audio_uuids = []
    audio_mxf_paths = []
    assets = []

    assets.append(IMFAsset(
        uuid=video_uuid,
        filename=video_mxf_name,
        asset_type="video_mxf",
        file_size=_file_size(video_mxf_path),
        hash_value=_sha1_file(video_mxf_path),
        original_filename=os.path.basename(video_path),
    ))

    audio_tracks = config.audio_tracks or [IMFAudioTrack()]
    for idx, track in enumerate(audio_tracks):
        audio_uuid = _imf_uuid()
        audio_uuids.append(audio_uuid)

        audio_source = track.source_path if track.source_path else video_path
        if not os.path.isfile(audio_source):
            logger.warning("Audio source not found for track %d: %s", idx, audio_source)
            audio_source = video_path

        lang_tag = track.language.replace("-", "_")
        audio_mxf_name = f"{title_safe}_audio_{lang_tag}.mxf"
        audio_mxf_path = os.path.join(imf_dir, audio_mxf_name)

        progress_base = 45 + (idx * 10)
        _encode_audio_mxf(
            audio_source, audio_mxf_path, config, track,
            on_progress, progress_base,
        )
        audio_mxf_paths.append(audio_mxf_path)

        assets.append(IMFAsset(
            uuid=audio_uuid,
            filename=audio_mxf_name,
            asset_type="audio_mxf",
            file_size=_file_size(audio_mxf_path),
            hash_value=_sha1_file(audio_mxf_path),
            original_filename=os.path.basename(audio_source),
        ))

    if on_progress:
        on_progress(70, "Computing hashes, generating metadata...")

    # Generate CPL
    cpl_path = os.path.join(imf_dir, "CPL.xml")
    _generate_imf_cpl(cpl_path, cpl_uuid, config, video_uuid,
                      audio_uuids, total_frames)

    cpl_asset = IMFAsset(
        uuid=cpl_uuid, filename="CPL.xml", asset_type="cpl",
        file_size=_file_size(cpl_path), hash_value=_sha1_file(cpl_path),
    )
    assets.append(cpl_asset)

    # Generate OPL
    opl_path = os.path.join(imf_dir, "OPL.xml")
    _generate_opl(opl_path, opl_uuid, cpl_uuid, config)

    opl_asset = IMFAsset(
        uuid=opl_uuid, filename="OPL.xml", asset_type="opl",
        file_size=_file_size(opl_path), hash_value=_sha1_file(opl_path),
    )
    assets.append(opl_asset)

    if on_progress:
        on_progress(80, "Generating PKL...")

    # Generate PKL
    pkl_path = os.path.join(imf_dir, "PKL.xml")
    _generate_imf_pkl(pkl_path, pkl_uuid, config, assets)

    if on_progress:
        on_progress(90, "Generating ASSETMAP...")

    # Generate ASSETMAP
    assetmap_path = os.path.join(imf_dir, "ASSETMAP.xml")
    _generate_imf_assetmap(assetmap_path, pkl_uuid, assets, config)

    # Compute totals
    total_size = sum(a.file_size for a in assets)
    total_size += _file_size(pkl_path) + _file_size(assetmap_path)

    if on_progress:
        on_progress(100, "IMF package complete")

    result = IMFResult(
        output_dir=imf_dir,
        title=config.title,
        profile=config.profile,
        profile_label=profile["label"],
        frame_rate=config.frame_rate,
        resolution=f"{config.width}x{config.height}",
        duration_seconds=duration,
        total_frames=total_frames,
        assets=assets,
        video_mxf_path=video_mxf_path,
        audio_mxf_paths=audio_mxf_paths,
        cpl_path=cpl_path,
        opl_path=opl_path,
        pkl_path=pkl_path,
        assetmap_path=assetmap_path,
        total_size_bytes=total_size,
        supplemental=config.supplemental,
    )

    logger.info("IMF export complete: %s (%d bytes)", imf_dir, total_size)
    return result.to_dict()
