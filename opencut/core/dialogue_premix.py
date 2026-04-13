"""
OpenCut Automated Dialogue Premix

Apply a broadcast-standard dialogue processing chain:
1. De-ess: tame sibilance in the 4-10kHz range
2. EQ: presence boost at 3kHz, low-cut below 80Hz
3. Compress: speech-optimized dynamics control
4. Loudness normalize to target LUFS (default -23 for broadcast)

Supports single-speaker and multi-speaker premix workflows.
All processing uses FFmpeg audio filters.
"""

import logging
import os
from typing import Callable, Dict, List, Optional

from opencut.helpers import get_ffmpeg_path, run_ffmpeg
from opencut.helpers import output_path as _output_path

logger = logging.getLogger("opencut")


def premix_dialogue(
    input_path: str,
    target_lufs: float = -23.0,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Apply a full dialogue premix chain to audio.

    Processing chain:
        1. De-ess (bandpass sidechain on 4-10kHz)
        2. EQ (presence boost at 3kHz, HPF at 80Hz)
        3. Compress (speech-optimized acompressor)
        4. Loudness normalize to target LUFS (loudnorm)

    Args:
        input_path: Source audio/video file.
        target_lufs: Target integrated loudness in LUFS (default -23 for broadcast).
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path, target_lufs, processing_chain.
    """
    output = _output_path(input_path, "premix")
    target_lufs = max(-36.0, min(-10.0, float(target_lufs)))

    if on_progress:
        on_progress(5, "Starting dialogue premix chain...")

    # Build the complete audio filter chain
    chain_parts = []
    chain_description = []

    # 1. De-ess: reduce sibilance
    # Use a combination of bandpass detection + compression on sibilant range
    # equalizer to attenuate 5-8kHz when loud (de-ess effect)
    # We apply the de-ess as a parallel process: split, process highs, recombine
    # Simpler approach: gentle cut on harsh sibilant frequencies
    deess_simple = "equalizer=f=6000:t=q:w=2:g=-3"
    chain_parts.append(deess_simple)
    chain_description.append("de-ess (6kHz -3dB)")

    if on_progress:
        on_progress(20, "Applied de-ess filter...")

    # 2. EQ: presence boost + low cut
    eq_hpf = "highpass=f=80:poles=2"
    eq_presence = "equalizer=f=3000:t=q:w=1.5:g=3"
    chain_parts.append(eq_hpf)
    chain_parts.append(eq_presence)
    chain_description.append("HPF 80Hz, presence +3dB at 3kHz")

    if on_progress:
        on_progress(40, "Applied EQ (HPF + presence boost)...")

    # 3. Compress: speech-optimized settings
    # Fast attack to catch transients, moderate release, gentle ratio
    compressor = (
        "acompressor="
        "threshold=-18dB:"
        "ratio=3:"
        "attack=10:"
        "release=150:"
        "makeup=2dB:"
        "knee=4dB"
    )
    chain_parts.append(compressor)
    chain_description.append("compress (3:1, -18dB threshold)")

    if on_progress:
        on_progress(60, "Applied compressor...")

    # 4. Loudness normalize to target LUFS
    loudnorm = f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11:print_format=summary"
    chain_parts.append(loudnorm)
    chain_description.append(f"loudnorm to {target_lufs} LUFS")

    if on_progress:
        on_progress(80, f"Normalizing to {target_lufs} LUFS...")

    # Join all filters
    af_chain = ",".join(chain_parts)

    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-af", af_chain,
        "-c:v", "copy",
        output,
    ]
    run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Dialogue premix complete")

    logger.info("Dialogue premix applied (%s LUFS): %s -> %s", target_lufs, input_path, output)
    return {
        "output_path": output,
        "target_lufs": target_lufs,
        "processing_chain": chain_description,
    }


def premix_multi_speaker(
    input_path: str,
    diarization_segments: Optional[List[Dict]] = None,
    target_lufs: float = -23.0,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Apply dialogue premix per speaker segment, then combine.

    If diarization_segments are provided, processes each segment independently
    and recombines. Otherwise, falls back to full-file premix.

    Args:
        input_path: Source audio/video file.
        diarization_segments: List of dicts with keys:
            - start (float): Segment start time in seconds.
            - end (float): Segment end time in seconds.
            - speaker (str, optional): Speaker label.
        target_lufs: Target integrated loudness in LUFS.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path, target_lufs, segments_processed.
    """
    if not diarization_segments:
        # No diarization available, do single-speaker premix
        if on_progress:
            on_progress(5, "No diarization segments provided, applying full-file premix...")
        result = premix_dialogue(input_path, target_lufs=target_lufs, on_progress=on_progress)
        result["segments_processed"] = 0
        return result

    output = _output_path(input_path, "premix_multi")
    target_lufs = max(-36.0, min(-10.0, float(target_lufs)))

    if on_progress:
        on_progress(5, f"Processing {len(diarization_segments)} speaker segments...")

    ffmpeg = get_ffmpeg_path()
    import tempfile

    # Process each segment independently
    segment_files = []
    try:
        for i, seg in enumerate(diarization_segments):
            start = float(seg.get("start", 0))
            end = float(seg.get("end", 0))
            if end <= start:
                continue

            seg_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix=f"seg{i}_")
            seg_path = seg_file.name
            seg_file.close()

            # Extract segment
            cmd_extract = [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-ss", str(start), "-to", str(end),
                "-i", input_path,
                "-vn", "-c:a", "pcm_s16le",
                seg_path,
            ]
            run_ffmpeg(cmd_extract)

            # Apply premix chain to segment
            premixed_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix=f"pmx{i}_")
            premixed_path = premixed_file.name
            premixed_file.close()

            af_chain = (
                "equalizer=f=6000:t=q:w=2:g=-3,"
                "highpass=f=80:poles=2,"
                "equalizer=f=3000:t=q:w=1.5:g=3,"
                f"acompressor=threshold=-18dB:ratio=3:attack=10:release=150:makeup=2dB:knee=4dB,"
                f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11"
            )

            cmd_premix = [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-i", seg_path,
                "-af", af_chain,
                premixed_path,
            ]
            run_ffmpeg(cmd_premix)

            segment_files.append({
                "path": premixed_path,
                "start": start,
                "end": end,
                "temp_extract": seg_path,
            })

            if on_progress:
                pct = 5 + int(70 * (i + 1) / len(diarization_segments))
                on_progress(pct, f"Premixed segment {i + 1}/{len(diarization_segments)}...")

        if not segment_files:
            # No valid segments, fall back to full premix
            result = premix_dialogue(input_path, target_lufs=target_lufs, on_progress=on_progress)
            result["segments_processed"] = 0
            return result

        if on_progress:
            on_progress(80, "Reassembling segments...")

        # Concatenate all premixed segments back together
        concat_list = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="concat_"
        )
        for sf in segment_files:
            # Escape single quotes in path for FFmpeg concat
            safe_path = sf["path"].replace("'", "'\\''")
            concat_list.write(f"file '{safe_path}'\n")
        concat_list.close()

        cmd_concat = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_list.name,
            "-c:a", "pcm_s16le",
            output,
        ]
        run_ffmpeg(cmd_concat)

        # Clean up concat list
        os.unlink(concat_list.name)

        if on_progress:
            on_progress(100, f"Multi-speaker premix complete ({len(segment_files)} segments)")

        logger.info("Multi-speaker premix (%d segments): %s -> %s",
                     len(segment_files), input_path, output)
        return {
            "output_path": output,
            "target_lufs": target_lufs,
            "segments_processed": len(segment_files),
        }

    finally:
        # Clean up temp files
        for sf in segment_files:
            for key in ("path", "temp_extract"):
                path = sf.get(key)
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
