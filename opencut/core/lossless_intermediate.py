"""
OpenCut Lossless Intermediate Codec Pipeline

Convert video to and from lossless intermediate codecs for multi-step
processing chains where quality preservation is critical.

Supported codecs:
  - FFV1: Open-source, archival-grade lossless codec
  - HuffYUV: Fast lossless codec, wide compatibility
  - UT Video: Fast decode, moderate file size

Uses FFmpeg for all codec conversions.
"""

import logging
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from opencut.helpers import (
    FFmpegCmd,
    get_video_info,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Codec registry
# ---------------------------------------------------------------------------
@dataclass
class IntermediateCodec:
    """Descriptor for an intermediate codec."""
    id: str
    name: str
    ffmpeg_encoder: str
    ffmpeg_decoder: str
    container: str
    pix_fmts: List[str]
    description: str
    pros: List[str]
    cons: List[str]
    default_pix_fmt: str = "yuv422p"


INTERMEDIATE_CODECS: Dict[str, IntermediateCodec] = {
    "ffv1": IntermediateCodec(
        id="ffv1",
        name="FFV1",
        ffmpeg_encoder="ffv1",
        ffmpeg_decoder="ffv1",
        container=".mkv",
        pix_fmts=["yuv420p", "yuv422p", "yuv444p", "yuv420p10le", "yuv422p10le",
                   "yuv444p10le", "rgb24", "bgr0"],
        description="FFV1 is an open-source, mathematically lossless video codec "
                    "designed for archival use. Used by the Library of Congress "
                    "and many film archives.",
        pros=["True mathematical lossless", "Open standard", "Archival grade",
              "10-bit and higher support", "Multithreaded encoding"],
        cons=["Larger files than lossy codecs", "Slower than HuffYUV"],
        default_pix_fmt="yuv422p10le",
    ),
    "huffyuv": IntermediateCodec(
        id="huffyuv",
        name="HuffYUV",
        ffmpeg_encoder="huffyuv",
        ffmpeg_decoder="huffyuv",
        container=".avi",
        pix_fmts=["yuv422p", "rgb24", "bgr0"],
        description="HuffYUV is a fast lossless codec using Huffman coding. "
                    "Very fast encode/decode but limited to 8-bit.",
        pros=["Very fast encode/decode", "Wide NLE compatibility", "Simple algorithm"],
        cons=["8-bit only", "Large files", "AVI container only"],
        default_pix_fmt="yuv422p",
    ),
    "utvideo": IntermediateCodec(
        id="utvideo",
        name="UT Video",
        ffmpeg_encoder="utvideo",
        ffmpeg_decoder="utvideo",
        container=".avi",
        pix_fmts=["yuv420p", "yuv422p", "rgb24", "rgba"],
        description="UT Video is a fast lossless codec with better compression "
                    "than HuffYUV while maintaining fast decode speed.",
        pros=["Fast decode", "Better compression than HuffYUV",
              "Good NLE compatibility"],
        cons=["8-bit only", "Less archival than FFV1"],
        default_pix_fmt="yuv422p",
    ),
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass
class IntermediateResult:
    """Result from conversion to intermediate codec."""
    output_path: str = ""
    codec: str = ""
    pix_fmt: str = ""
    file_size_mb: float = 0.0
    duration: float = 0.0
    lossless: bool = True


@dataclass
class DeliveryResult:
    """Result from conversion from intermediate to delivery codec."""
    output_path: str = ""
    source_codec: str = ""
    delivery_codec: str = ""
    delivery_preset: str = ""
    file_size_mb: float = 0.0


# ---------------------------------------------------------------------------
# Convert TO intermediate
# ---------------------------------------------------------------------------
def to_intermediate(
    video_path: str,
    codec: str = "ffv1",
    pix_fmt: Optional[str] = None,
    output_path_str: Optional[str] = None,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> IntermediateResult:
    """
    Convert a video to a lossless intermediate codec.

    Used to preserve quality during multi-step processing chains
    (color grading, compositing, effects stacking).

    Args:
        video_path: Input video file.
        codec: Intermediate codec ID: "ffv1", "huffyuv", or "utvideo".
        pix_fmt: Pixel format. Uses codec default if None.
        output_path_str: Output path. Auto-generated if None.
        output_dir: Directory for output.
        on_progress: Progress callback(pct, msg).

    Returns:
        IntermediateResult with output path and codec metadata.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    codec = codec.lower().strip()
    if codec not in INTERMEDIATE_CODECS:
        raise ValueError(
            f"Unknown intermediate codec: {codec}. "
            f"Supported: {', '.join(INTERMEDIATE_CODECS.keys())}"
        )

    codec_info = INTERMEDIATE_CODECS[codec]

    # Validate pixel format
    if pix_fmt is None:
        pix_fmt = codec_info.default_pix_fmt
    if pix_fmt not in codec_info.pix_fmts:
        logger.warning(
            "Pixel format %s not in %s supported list, using default %s",
            pix_fmt, codec, codec_info.default_pix_fmt,
        )
        pix_fmt = codec_info.default_pix_fmt

    if on_progress:
        on_progress(5, f"Converting to {codec_info.name} ({pix_fmt})...")

    # Build output path
    if output_path_str is None:
        directory = output_dir or os.path.dirname(video_path)
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_path_str = os.path.join(
            directory, f"{base}_intermediate{codec_info.container}"
        )

    # Build FFmpeg command
    builder = (FFmpegCmd()
               .input(video_path)
               .audio_codec("copy"))

    # Codec-specific settings
    if codec == "ffv1":
        builder = (builder
                   .option("c:v", "ffv1")
                   .option("level", "3")          # FFV1 version 3 (multithreaded)
                   .option("slicecrc", "1")        # Error detection
                   .option("slices", "4")          # Parallel slices
                   .option("pix_fmt", pix_fmt))
    elif codec == "huffyuv":
        builder = (builder
                   .option("c:v", "huffyuv")
                   .option("pix_fmt", pix_fmt))
    elif codec == "utvideo":
        builder = (builder
                   .option("c:v", "utvideo")
                   .option("pix_fmt", pix_fmt))

    builder = builder.output(output_path_str)
    cmd = builder.build()

    if on_progress:
        on_progress(15, "Encoding lossless intermediate...")

    run_ffmpeg(cmd, timeout=7200)

    # Get file size
    file_size = 0.0
    try:
        file_size = os.path.getsize(output_path_str) / (1024 * 1024)
    except OSError:
        pass

    info = get_video_info(output_path_str)

    if on_progress:
        on_progress(100, f"Intermediate created: {file_size:.1f} MB")

    return IntermediateResult(
        output_path=output_path_str,
        codec=codec,
        pix_fmt=pix_fmt,
        file_size_mb=round(file_size, 2),
        duration=info.get("duration", 0),
        lossless=True,
    )


# ---------------------------------------------------------------------------
# Convert FROM intermediate to delivery
# ---------------------------------------------------------------------------
# Delivery presets
DELIVERY_PRESETS = {
    "h264_web": {
        "codec": "libx264", "crf": "18", "preset": "medium",
        "pix_fmt": "yuv420p", "container": ".mp4",
        "description": "H.264 for web delivery",
    },
    "h264_master": {
        "codec": "libx264", "crf": "14", "preset": "slow",
        "pix_fmt": "yuv420p", "container": ".mp4",
        "description": "H.264 high quality master",
    },
    "h265_web": {
        "codec": "libx265", "crf": "22", "preset": "medium",
        "pix_fmt": "yuv420p", "container": ".mp4",
        "description": "H.265/HEVC for web delivery",
    },
    "h265_master": {
        "codec": "libx265", "crf": "16", "preset": "slow",
        "pix_fmt": "yuv420p10le", "container": ".mp4",
        "description": "H.265/HEVC high quality master",
    },
    "prores_proxy": {
        "codec": "prores_ks", "profile": "0", "preset": None,
        "pix_fmt": "yuv422p10le", "container": ".mov",
        "description": "ProRes Proxy (lightweight editing)",
    },
    "prores_hq": {
        "codec": "prores_ks", "profile": "3", "preset": None,
        "pix_fmt": "yuv422p10le", "container": ".mov",
        "description": "ProRes HQ (broadcast quality)",
    },
    "dnxhd_36": {
        "codec": "dnxhd", "bitrate": "36M", "preset": None,
        "pix_fmt": "yuv422p", "container": ".mxf",
        "description": "DNxHD 36 (offline editing)",
    },
}


def from_intermediate(
    intermediate_path: str,
    delivery_codec: str = "h264_web",
    output_path_str: Optional[str] = None,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> DeliveryResult:
    """
    Convert a lossless intermediate to a delivery codec.

    Args:
        intermediate_path: Input lossless intermediate file.
        delivery_codec: Delivery preset ID (e.g. "h264_web", "prores_hq").
        output_path_str: Output path. Auto-generated if None.
        output_dir: Directory for output.
        on_progress: Progress callback(pct, msg).

    Returns:
        DeliveryResult.
    """
    if not os.path.isfile(intermediate_path):
        raise FileNotFoundError(f"Intermediate file not found: {intermediate_path}")

    delivery_codec = delivery_codec.lower().strip()
    if delivery_codec not in DELIVERY_PRESETS:
        raise ValueError(
            f"Unknown delivery preset: {delivery_codec}. "
            f"Available: {', '.join(DELIVERY_PRESETS.keys())}"
        )

    preset = DELIVERY_PRESETS[delivery_codec]

    if on_progress:
        on_progress(5, f"Converting to {preset['description']}...")

    # Detect source codec
    get_video_info(intermediate_path)
    source_codec = "unknown"
    for cid, ci in INTERMEDIATE_CODECS.items():
        ext = ci.container
        if intermediate_path.lower().endswith(ext):
            source_codec = cid
            break

    # Output path
    if output_path_str is None:
        directory = output_dir or os.path.dirname(intermediate_path)
        base = os.path.splitext(os.path.basename(intermediate_path))[0]
        # Remove _intermediate suffix if present
        if base.endswith("_intermediate"):
            base = base[:-len("_intermediate")]
        output_path_str = os.path.join(
            directory, f"{base}_delivery{preset['container']}"
        )

    # Build FFmpeg command
    builder = (FFmpegCmd()
               .input(intermediate_path)
               .audio_codec("aac", bitrate="192k"))

    encoder = preset["codec"]
    pix_fmt = preset.get("pix_fmt", "yuv420p")

    if encoder in ("libx264", "libx265"):
        crf = preset.get("crf", "18")
        enc_preset = preset.get("preset", "medium")
        builder = (builder
                   .option("c:v", encoder)
                   .option("crf", str(crf))
                   .option("preset", enc_preset)
                   .option("pix_fmt", pix_fmt))
        if encoder == "libx264":
            builder = builder.faststart()
    elif encoder == "prores_ks":
        profile = preset.get("profile", "3")
        builder = (builder
                   .option("c:v", "prores_ks")
                   .option("profile:v", profile)
                   .option("pix_fmt", pix_fmt))
    elif encoder == "dnxhd":
        bitrate = preset.get("bitrate", "36M")
        builder = (builder
                   .option("c:v", "dnxhd")
                   .option("b:v", bitrate)
                   .option("pix_fmt", pix_fmt))

    builder = builder.output(output_path_str)
    cmd = builder.build()

    if on_progress:
        on_progress(15, f"Encoding {preset['description']}...")

    run_ffmpeg(cmd, timeout=7200)

    file_size = 0.0
    try:
        file_size = os.path.getsize(output_path_str) / (1024 * 1024)
    except OSError:
        pass

    if on_progress:
        on_progress(100, f"Delivery file: {file_size:.1f} MB")

    return DeliveryResult(
        output_path=output_path_str,
        source_codec=source_codec,
        delivery_codec=delivery_codec,
        delivery_preset=preset["description"],
        file_size_mb=round(file_size, 2),
    )


# ---------------------------------------------------------------------------
# Info helpers
# ---------------------------------------------------------------------------
def list_intermediate_codecs() -> List[Dict]:
    """Return metadata for all supported intermediate codecs."""
    result = []
    for cid, codec in INTERMEDIATE_CODECS.items():
        result.append({
            "id": codec.id,
            "name": codec.name,
            "container": codec.container,
            "pix_fmts": codec.pix_fmts,
            "default_pix_fmt": codec.default_pix_fmt,
            "description": codec.description,
            "pros": codec.pros,
            "cons": codec.cons,
        })
    return result


def get_recommended_codec(
    use_case: str = "general",
) -> Dict:
    """
    Recommend an intermediate codec based on the use case.

    Args:
        use_case: One of "general", "archive", "speed", "compatibility".

    Returns:
        Dict with codec recommendation and reasoning.
    """
    recommendations = {
        "general": {
            "codec": "ffv1",
            "reason": "FFV1 offers the best balance of lossless quality, "
                      "10-bit support, and reasonable file sizes. Suitable "
                      "for most editing and compositing workflows.",
        },
        "archive": {
            "codec": "ffv1",
            "reason": "FFV1 is the gold standard for archival. Open source, "
                      "mathematically lossless, used by the Library of Congress "
                      "and major film archives worldwide.",
        },
        "speed": {
            "codec": "utvideo",
            "reason": "UT Video has the fastest decode speed, making it ideal "
                      "for real-time playback during editing and previewing "
                      "multi-layer compositions.",
        },
        "compatibility": {
            "codec": "huffyuv",
            "reason": "HuffYUV has the widest NLE compatibility, working in "
                      "virtually all editing applications on all platforms.",
        },
    }

    use_case = use_case.lower().strip()
    if use_case not in recommendations:
        use_case = "general"

    rec = recommendations[use_case]
    codec_info = INTERMEDIATE_CODECS[rec["codec"]]

    return {
        "use_case": use_case,
        "recommended_codec": rec["codec"],
        "codec_name": codec_info.name,
        "reason": rec["reason"],
        "codec_details": {
            "container": codec_info.container,
            "default_pix_fmt": codec_info.default_pix_fmt,
            "pros": codec_info.pros,
            "cons": codec_info.cons,
        },
    }
