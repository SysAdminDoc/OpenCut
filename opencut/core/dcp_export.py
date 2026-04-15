"""
OpenCut DCP Export Module (70.1)

Digital Cinema Package (DCP) export for theatrical projection.
Generates JPEG2000 MXF video, 24-bit PCM audio MXF, CPL, PKL,
ASSETMAP, and VOLINDEX per SMPTE DCP specifications.
Supports 2K/4K DCI, Flat/Scope containers, 24/48fps.
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
# DCI Format Definitions
# ---------------------------------------------------------------------------

DCP_FORMATS = {
    "2K_flat": {
        "label": "2K Flat (1.85:1)",
        "width": 1998,
        "height": 1080,
        "aspect": "flat",
        "resolution": "2K",
    },
    "2K_scope": {
        "label": "2K Scope (2.39:1)",
        "width": 2048,
        "height": 858,
        "aspect": "scope",
        "resolution": "2K",
    },
    "4K_flat": {
        "label": "4K Flat (1.85:1)",
        "width": 3996,
        "height": 2160,
        "aspect": "flat",
        "resolution": "4K",
    },
    "4K_scope": {
        "label": "4K Scope (2.39:1)",
        "width": 4096,
        "height": 1716,
        "aspect": "scope",
        "resolution": "4K",
    },
}

_DCP_FRAME_RATES = {24, 48}

_DCI_MAX_BITRATE_2K = 250_000_000  # 250 Mbps
_DCI_MAX_BITRATE_4K = 500_000_000  # 500 Mbps

_DCP_NAMESPACES = {
    "cpl": "http://www.digicine.com/PROTO-ASDCP-CPL-20040511#",
    "pkl": "http://www.digicine.com/PROTO-ASDCP-PKL-20040511#",
    "am": "http://www.digicine.com/PROTO-ASDCP-AM-20040311#",
    "ds": "http://www.w3.org/2000/09/xmldsig#",
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DCPConfig:
    """Configuration for DCP export."""
    title: str = "Untitled"
    format_key: str = "2K_flat"
    frame_rate: int = 24
    content_kind: str = "feature"
    rating: str = ""
    issuer: str = "OpenCut"
    creator: str = "OpenCut DCP Export"
    annotation: str = ""
    audio_channels: int = 6
    audio_sample_rate: int = 48000
    audio_bit_depth: int = 24
    jpeg2000_bandwidth: int = 0
    encrypt: bool = False
    three_d: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> List[str]:
        """Return list of validation errors, empty if config is valid."""
        errors = []
        if self.format_key not in DCP_FORMATS:
            errors.append(f"Unknown format_key '{self.format_key}'. "
                          f"Valid: {list(DCP_FORMATS.keys())}")
        if self.frame_rate not in _DCP_FRAME_RATES:
            errors.append(f"Frame rate must be one of {_DCP_FRAME_RATES}, "
                          f"got {self.frame_rate}")
        if self.audio_channels < 1 or self.audio_channels > 16:
            errors.append(f"audio_channels must be 1-16, got {self.audio_channels}")
        if self.audio_sample_rate not in (48000, 96000):
            errors.append(f"audio_sample_rate must be 48000 or 96000, "
                          f"got {self.audio_sample_rate}")
        if self.audio_bit_depth not in (16, 24):
            errors.append(f"audio_bit_depth must be 16 or 24, "
                          f"got {self.audio_bit_depth}")
        if self.content_kind not in (
            "feature", "trailer", "teaser", "advertisement",
            "short", "transitional", "test", "rating", "policy",
        ):
            errors.append(f"Unknown content_kind '{self.content_kind}'")
        return errors


@dataclass
class DCPAsset:
    """A single asset in the DCP package."""
    uuid: str = ""
    filename: str = ""
    asset_type: str = ""  # "video_mxf", "audio_mxf", "cpl", "pkl"
    file_size: int = 0
    hash_value: str = ""
    hash_algorithm: str = "SHA-1"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DCPResult:
    """Result of DCP export operation."""
    output_dir: str = ""
    title: str = ""
    format_label: str = ""
    resolution: str = ""
    frame_rate: int = 24
    duration_seconds: float = 0.0
    total_frames: int = 0
    assets: List[DCPAsset] = field(default_factory=list)
    video_mxf_path: str = ""
    audio_mxf_path: str = ""
    cpl_path: str = ""
    pkl_path: str = ""
    assetmap_path: str = ""
    volindex_path: str = ""
    total_size_bytes: int = 0

    def to_dict(self) -> dict:
        return {
            "output_dir": self.output_dir,
            "title": self.title,
            "format_label": self.format_label,
            "resolution": self.resolution,
            "frame_rate": self.frame_rate,
            "duration_seconds": round(self.duration_seconds, 3),
            "total_frames": self.total_frames,
            "assets": [a.to_dict() for a in self.assets],
            "video_mxf_path": self.video_mxf_path,
            "audio_mxf_path": self.audio_mxf_path,
            "cpl_path": self.cpl_path,
            "pkl_path": self.pkl_path,
            "assetmap_path": self.assetmap_path,
            "volindex_path": self.volindex_path,
            "total_size_bytes": self.total_size_bytes,
        }


# ---------------------------------------------------------------------------
# UUID and Hash Helpers
# ---------------------------------------------------------------------------

def _generate_uuid() -> str:
    """Generate a URN-prefixed UUID4 for DCP assets."""
    return f"urn:uuid:{uuid.uuid4()}"


def _sha1_file(filepath: str) -> str:
    """Compute SHA-1 hash of a file, returned as base64."""
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
    """Get file size in bytes."""
    try:
        return os.path.getsize(filepath)
    except OSError:
        return 0


# ---------------------------------------------------------------------------
# Probe source media
# ---------------------------------------------------------------------------

def _probe_source(video_path: str) -> dict:
    """Probe source video for detailed info."""
    cmd = [
        get_ffprobe_path(), "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,codec_name,pix_fmt,duration,"
        "nb_frames,color_space,color_transfer,color_primaries",
        "-show_entries", "format=duration,nb_streams",
        "-of", "json", video_path,
    ]
    result = _sp.run(cmd, capture_output=True, timeout=60)
    info = {
        "width": 1920, "height": 1080, "fps": 24.0, "duration": 0.0,
        "codec": "unknown", "pix_fmt": "unknown", "nb_frames": 0,
        "has_audio": False, "nb_streams": 1,
    }
    if result.returncode != 0:
        logger.warning("DCP probe failed for %s", video_path)
        return info
    try:
        data = json.loads(result.stdout.decode())
        streams = data.get("streams", [])
        fmt = data.get("format", {})
        if streams:
            s = streams[0]
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
        info["nb_streams"] = int(fmt.get("nb_streams", 1) or 1)
        info["has_audio"] = info["nb_streams"] > 1
    except (ValueError, KeyError, TypeError) as exc:
        logger.warning("DCP probe parse error: %s", exc)
    return info


# ---------------------------------------------------------------------------
# Audio probe helper
# ---------------------------------------------------------------------------

def _probe_audio(video_path: str) -> dict:
    """Probe audio stream details from source."""
    cmd = [
        get_ffprobe_path(), "-v", "quiet",
        "-select_streams", "a:0",
        "-show_entries",
        "stream=codec_name,channels,sample_rate,bits_per_raw_sample,"
        "channel_layout,duration",
        "-of", "json", video_path,
    ]
    result = _sp.run(cmd, capture_output=True, timeout=30)
    defaults = {
        "codec": "unknown", "channels": 2, "sample_rate": 48000,
        "bit_depth": 24, "channel_layout": "stereo", "duration": 0.0,
    }
    if result.returncode != 0:
        return defaults
    try:
        data = json.loads(result.stdout.decode())
        streams = data.get("streams", [])
        if not streams:
            return defaults
        s = streams[0]
        defaults["codec"] = s.get("codec_name", "unknown")
        defaults["channels"] = int(s.get("channels", 2) or 2)
        defaults["sample_rate"] = int(s.get("sample_rate", 48000) or 48000)
        defaults["bit_depth"] = int(s.get("bits_per_raw_sample", 24) or 24)
        defaults["channel_layout"] = s.get("channel_layout", "stereo")
        defaults["duration"] = float(s.get("duration", 0) or 0)
    except (ValueError, KeyError, TypeError):
        pass
    return defaults


# ---------------------------------------------------------------------------
# Video MXF (JPEG2000) generation
# ---------------------------------------------------------------------------

def _convert_video_to_j2k_mxf(
    video_path: str,
    output_mxf: str,
    fmt: dict,
    config: DCPConfig,
    source_info: dict,
    on_progress: Optional[Callable] = None,
) -> str:
    """Convert source video to JPEG2000 MXF track.

    Uses FFmpeg to scale/pad to DCI container, encode to JPEG2000,
    and wrap in MXF container.
    """
    target_w = fmt["width"]
    target_h = fmt["height"]
    src_w = source_info["width"]
    src_h = source_info["height"]

    # Calculate scaling to fit within DCI container while maintaining aspect
    scale_x = target_w / max(src_w, 1)
    scale_y = target_h / max(src_h, 1)
    scale = min(scale_x, scale_y)
    scaled_w = int(src_w * scale)
    scaled_h = int(src_h * scale)
    # Ensure even dimensions
    scaled_w = scaled_w if scaled_w % 2 == 0 else scaled_w + 1
    scaled_h = scaled_h if scaled_h % 2 == 0 else scaled_h + 1

    # Padding to center within DCI container
    pad_x = (target_w - scaled_w) // 2
    pad_y = (target_h - scaled_h) // 2

    # JPEG2000 bitrate
    bandwidth = config.jpeg2000_bandwidth
    if bandwidth <= 0:
        max_br = (_DCI_MAX_BITRATE_4K if fmt["resolution"] == "4K"
                  else _DCI_MAX_BITRATE_2K)
        bandwidth = max_br

    if on_progress:
        on_progress(10, "Encoding video to JPEG2000 MXF...")

    vf_chain = (
        f"scale={scaled_w}:{scaled_h}:flags=lanczos,"
        f"pad={target_w}:{target_h}:{pad_x}:{pad_y}:color=black,"
        f"fps={config.frame_rate}"
    )

    cmd = [
        get_ffmpeg_path(), "-y",
        "-i", video_path,
        "-an",  # no audio
        "-vf", vf_chain,
        "-c:v", "libopenjpeg",
        "-b:v", str(bandwidth),
        "-pix_fmt", "xyz12le",
        "-r", str(config.frame_rate),
        "-f", "mxf",
        output_mxf,
    ]

    logger.info("DCP video encode: %s -> %s (%dx%d @ %dfps)",
                video_path, output_mxf, target_w, target_h, config.frame_rate)

    proc = _sp.run(cmd, capture_output=True, timeout=7200)
    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors="replace")[-2000:]
        raise RuntimeError(f"JPEG2000 MXF encode failed: {stderr}")

    if on_progress:
        on_progress(40, "Video MXF encode complete")

    return output_mxf


# ---------------------------------------------------------------------------
# Audio MXF (24-bit PCM) generation
# ---------------------------------------------------------------------------

def _convert_audio_to_pcm_mxf(
    video_path: str,
    output_mxf: str,
    config: DCPConfig,
    source_info: dict,
    on_progress: Optional[Callable] = None,
) -> str:
    """Convert audio to 24-bit PCM WAV then wrap in MXF.

    DCP audio must be 24-bit linear PCM at 48kHz or 96kHz.
    """
    if on_progress:
        on_progress(45, "Encoding audio to PCM MXF...")

    # Step 1: extract to WAV
    wav_tmp = output_mxf.replace(".mxf", "_temp.wav")

    wav_cmd = [
        get_ffmpeg_path(), "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s24le",
        "-ar", str(config.audio_sample_rate),
        "-ac", str(config.audio_channels),
        wav_tmp,
    ]

    proc = _sp.run(wav_cmd, capture_output=True, timeout=3600)
    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors="replace")[-1000:]
        raise RuntimeError(f"Audio WAV extraction failed: {stderr}")

    if on_progress:
        on_progress(55, "Wrapping audio in MXF container...")

    # Step 2: wrap WAV in MXF
    mxf_cmd = [
        get_ffmpeg_path(), "-y",
        "-i", wav_tmp,
        "-c:a", "pcm_s24le",
        "-f", "mxf",
        output_mxf,
    ]

    proc = _sp.run(mxf_cmd, capture_output=True, timeout=3600)

    # Clean up temp WAV
    try:
        if os.path.isfile(wav_tmp):
            os.unlink(wav_tmp)
    except OSError:
        pass

    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors="replace")[-1000:]
        raise RuntimeError(f"Audio MXF wrap failed: {stderr}")

    if on_progress:
        on_progress(60, "Audio MXF encode complete")

    return output_mxf


# ---------------------------------------------------------------------------
# CPL XML Generation
# ---------------------------------------------------------------------------

def _generate_cpl(
    cpl_path: str,
    cpl_uuid: str,
    config: DCPConfig,
    video_uuid: str,
    audio_uuid: str,
    fmt: dict,
    duration_frames: int,
) -> str:
    """Generate Composition Playlist XML per DCP spec."""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    edit_rate = f"{config.frame_rate} 1"

    root = ET.Element("CompositionPlaylist", xmlns=_DCP_NAMESPACES["cpl"])
    ET.SubElement(root, "Id").text = cpl_uuid
    ET.SubElement(root, "AnnotationText").text = (
        config.annotation or config.title
    )
    ET.SubElement(root, "IssueDate").text = now_str
    ET.SubElement(root, "Issuer").text = config.issuer
    ET.SubElement(root, "Creator").text = config.creator
    ET.SubElement(root, "ContentTitleText").text = config.title
    ET.SubElement(root, "ContentKind").text = config.content_kind

    if config.rating:
        rating_list = ET.SubElement(root, "RatingList")
        rating_el = ET.SubElement(rating_list, "Rating")
        ET.SubElement(rating_el, "Agency").text = "http://www.mpaa.org/2003-ratings"
        ET.SubElement(rating_el, "Label").text = config.rating
    else:
        ET.SubElement(root, "RatingList")

    # Reel list
    reel_list = ET.SubElement(root, "ReelList")
    reel = ET.SubElement(reel_list, "Reel")
    reel_id = _generate_uuid()
    ET.SubElement(reel, "Id").text = reel_id
    asset_list = ET.SubElement(reel, "AssetList")

    # Main Picture
    main_pic = ET.SubElement(asset_list, "MainPicture")
    ET.SubElement(main_pic, "Id").text = video_uuid
    ET.SubElement(main_pic, "EditRate").text = edit_rate
    ET.SubElement(main_pic, "IntrinsicDuration").text = str(duration_frames)
    ET.SubElement(main_pic, "EntryPoint").text = "0"
    ET.SubElement(main_pic, "Duration").text = str(duration_frames)
    ET.SubElement(main_pic, "FrameRate").text = edit_rate
    ET.SubElement(main_pic, "ScreenAspectRatio").text = (
        f"{fmt['width']} {fmt['height']}"
    )

    # Main Sound
    main_snd = ET.SubElement(asset_list, "MainSound")
    ET.SubElement(main_snd, "Id").text = audio_uuid
    ET.SubElement(main_snd, "EditRate").text = edit_rate
    ET.SubElement(main_snd, "IntrinsicDuration").text = str(duration_frames)
    ET.SubElement(main_snd, "EntryPoint").text = "0"
    ET.SubElement(main_snd, "Duration").text = str(duration_frames)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(cpl_path, xml_declaration=True, encoding="UTF-8")

    logger.info("DCP CPL written: %s", cpl_path)
    return cpl_path


# ---------------------------------------------------------------------------
# PKL XML Generation
# ---------------------------------------------------------------------------

def _generate_pkl(
    pkl_path: str,
    pkl_uuid: str,
    cpl_uuid: str,
    config: DCPConfig,
    assets: List[DCPAsset],
) -> str:
    """Generate Packing List XML."""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")

    root = ET.Element("PackingList", xmlns=_DCP_NAMESPACES["pkl"])
    ET.SubElement(root, "Id").text = pkl_uuid
    ET.SubElement(root, "AnnotationText").text = config.title
    ET.SubElement(root, "IssueDate").text = now_str
    ET.SubElement(root, "Issuer").text = config.issuer
    ET.SubElement(root, "Creator").text = config.creator

    asset_list = ET.SubElement(root, "AssetList")
    for asset in assets:
        a_el = ET.SubElement(asset_list, "Asset")
        ET.SubElement(a_el, "Id").text = asset.uuid
        ET.SubElement(a_el, "AnnotationText").text = asset.filename

        mime = "application/mxf"
        if asset.asset_type == "cpl":
            mime = "text/xml"
        ET.SubElement(a_el, "Hash").text = asset.hash_value
        ET.SubElement(a_el, "Size").text = str(asset.file_size)
        ET.SubElement(a_el, "Type").text = mime

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(pkl_path, xml_declaration=True, encoding="UTF-8")

    logger.info("DCP PKL written: %s", pkl_path)
    return pkl_path


# ---------------------------------------------------------------------------
# ASSETMAP XML Generation
# ---------------------------------------------------------------------------

def _generate_assetmap(
    assetmap_path: str,
    pkl_uuid: str,
    assets: List[DCPAsset],
    config: DCPConfig,
) -> str:
    """Generate ASSETMAP XML listing all files in the DCP."""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")

    root = ET.Element("AssetMap", xmlns=_DCP_NAMESPACES["am"])
    ET.SubElement(root, "Id").text = _generate_uuid()
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

    # PKL entry
    pkl_el = ET.SubElement(asset_list, "Asset")
    ET.SubElement(pkl_el, "Id").text = pkl_uuid
    ET.SubElement(pkl_el, "PackingList").text = "true"
    pkl_chunks = ET.SubElement(pkl_el, "ChunkList")
    pkl_chunk = ET.SubElement(pkl_chunks, "Chunk")
    pkl_filename = os.path.basename(assetmap_path).replace("ASSETMAP.xml", "PKL.xml")
    ET.SubElement(pkl_chunk, "Path").text = pkl_filename
    ET.SubElement(pkl_chunk, "VolumeIndex").text = "1"
    ET.SubElement(pkl_chunk, "Offset").text = "0"

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(assetmap_path, xml_declaration=True, encoding="UTF-8")

    logger.info("DCP ASSETMAP written: %s", assetmap_path)
    return assetmap_path


# ---------------------------------------------------------------------------
# VOLINDEX XML Generation
# ---------------------------------------------------------------------------

def _generate_volindex(volindex_path: str) -> str:
    """Generate VOLINDEX XML (single-volume DCP)."""
    root = ET.Element("VolumeIndex", xmlns=_DCP_NAMESPACES["am"])
    ET.SubElement(root, "Index").text = "1"

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(volindex_path, xml_declaration=True, encoding="UTF-8")

    logger.info("DCP VOLINDEX written: %s", volindex_path)
    return volindex_path


# ---------------------------------------------------------------------------
# Main Export Function
# ---------------------------------------------------------------------------

def export_dcp(
    video_path: str,
    output_dir: str,
    config: Optional[DCPConfig] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Export a video file as a Digital Cinema Package.

    Args:
        video_path: Path to source video file.
        output_dir: Directory where DCP folder will be created.
        config: DCPConfig with export settings. Uses defaults if None.
        on_progress: Optional callback ``(percent, message)``.

    Returns:
        DCPResult.to_dict()

    Raises:
        FileNotFoundError: If video_path does not exist.
        ValueError: If config validation fails.
        RuntimeError: If FFmpeg encoding fails.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Source video not found: {video_path}")

    if config is None:
        config = DCPConfig()

    errors = config.validate()
    if errors:
        raise ValueError(f"DCPConfig validation failed: {'; '.join(errors)}")

    fmt = DCP_FORMATS[config.format_key]
    title_safe = "".join(c if c.isalnum() or c in " _-" else "_"
                         for c in config.title).strip()
    dcp_dirname = f"{title_safe}_DCP"
    dcp_dir = os.path.join(output_dir, dcp_dirname)
    os.makedirs(dcp_dir, exist_ok=True)

    if on_progress:
        on_progress(2, f"Probing source: {os.path.basename(video_path)}")

    source_info = _probe_source(video_path)
    duration = source_info["duration"]
    total_frames = int(duration * config.frame_rate)
    if total_frames < 1:
        total_frames = max(source_info["nb_frames"], 1)

    # Generate UUIDs for all components
    video_uuid = _generate_uuid()
    audio_uuid = _generate_uuid()
    cpl_uuid = _generate_uuid()
    pkl_uuid = _generate_uuid()

    # File names
    video_mxf_name = f"{title_safe}_video.mxf"
    audio_mxf_name = f"{title_safe}_audio.mxf"
    cpl_name = "CPL.xml"
    pkl_name = "PKL.xml"
    assetmap_name = "ASSETMAP.xml"
    volindex_name = "VOLINDEX.xml"

    video_mxf_path = os.path.join(dcp_dir, video_mxf_name)
    audio_mxf_path = os.path.join(dcp_dir, audio_mxf_name)
    cpl_path = os.path.join(dcp_dir, cpl_name)
    pkl_path = os.path.join(dcp_dir, pkl_name)
    assetmap_path = os.path.join(dcp_dir, assetmap_name)
    volindex_path = os.path.join(dcp_dir, volindex_name)

    # Step 1: Encode video to JPEG2000 MXF
    _convert_video_to_j2k_mxf(
        video_path, video_mxf_path, fmt, config, source_info, on_progress,
    )

    # Step 2: Encode audio to PCM MXF
    if source_info["has_audio"]:
        _convert_audio_to_pcm_mxf(
            video_path, audio_mxf_path, config, source_info, on_progress,
        )
    else:
        # Generate silent audio track
        if on_progress:
            on_progress(50, "No audio in source, generating silence...")
        silence_cmd = [
            get_ffmpeg_path(), "-y",
            "-f", "lavfi",
            "-i", f"anullsrc=channel_layout={'5.1' if config.audio_channels >= 6 else 'stereo'}:sample_rate={config.audio_sample_rate}",
            "-t", str(duration),
            "-c:a", "pcm_s24le",
            "-f", "mxf",
            audio_mxf_path,
        ]
        proc = _sp.run(silence_cmd, capture_output=True, timeout=3600)
        if proc.returncode != 0:
            logger.warning("Silent audio generation failed, DCP may lack audio")

    if on_progress:
        on_progress(65, "Computing asset hashes...")

    # Build asset list
    assets = []

    video_asset = DCPAsset(
        uuid=video_uuid,
        filename=video_mxf_name,
        asset_type="video_mxf",
        file_size=_file_size(video_mxf_path),
        hash_value=_sha1_file(video_mxf_path),
    )
    assets.append(video_asset)

    audio_asset = DCPAsset(
        uuid=audio_uuid,
        filename=audio_mxf_name,
        asset_type="audio_mxf",
        file_size=_file_size(audio_mxf_path),
        hash_value=_sha1_file(audio_mxf_path),
    )
    assets.append(audio_asset)

    if on_progress:
        on_progress(75, "Generating CPL...")

    # Step 3: Generate CPL
    _generate_cpl(
        cpl_path, cpl_uuid, config, video_uuid, audio_uuid,
        fmt, total_frames,
    )

    cpl_asset = DCPAsset(
        uuid=cpl_uuid,
        filename=cpl_name,
        asset_type="cpl",
        file_size=_file_size(cpl_path),
        hash_value=_sha1_file(cpl_path),
    )
    assets.append(cpl_asset)

    if on_progress:
        on_progress(80, "Generating PKL...")

    # Step 4: Generate PKL
    _generate_pkl(pkl_path, pkl_uuid, cpl_uuid, config, assets)

    if on_progress:
        on_progress(85, "Generating ASSETMAP...")

    # Step 5: Generate ASSETMAP
    _generate_assetmap(assetmap_path, pkl_uuid, assets, config)

    if on_progress:
        on_progress(90, "Generating VOLINDEX...")

    # Step 6: Generate VOLINDEX
    _generate_volindex(volindex_path)

    # Compute totals
    total_size = sum(a.file_size for a in assets)
    total_size += _file_size(pkl_path)
    total_size += _file_size(assetmap_path)
    total_size += _file_size(volindex_path)

    if on_progress:
        on_progress(100, "DCP export complete")

    result = DCPResult(
        output_dir=dcp_dir,
        title=config.title,
        format_label=fmt["label"],
        resolution=fmt["resolution"],
        frame_rate=config.frame_rate,
        duration_seconds=duration,
        total_frames=total_frames,
        assets=assets,
        video_mxf_path=video_mxf_path,
        audio_mxf_path=audio_mxf_path,
        cpl_path=cpl_path,
        pkl_path=pkl_path,
        assetmap_path=assetmap_path,
        volindex_path=volindex_path,
        total_size_bytes=total_size,
    )

    logger.info("DCP export complete: %s (%d bytes)", dcp_dir, total_size)
    return result.to_dict()
