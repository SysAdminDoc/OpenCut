"""
OpenCut Video LLM Integration

Multimodal LLM-powered video understanding: ask questions about video
content, describe scenes, find moments matching descriptions.
Supports local models (via opencut.core.llm), OpenAI/Anthropic vision
APIs, and Florence-2 as a fallback captioner.

Uses FFmpeg for frame extraction — no cv2 dependency required.
"""

import base64
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from opencut.helpers import (
    ensure_package,
    get_ffmpeg_path,
    get_video_info,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class VideoQueryResult:
    """Result of a multimodal LLM query about video content."""
    answer: str = ""
    timestamps: List[float] = field(default_factory=list)
    confidence: float = 0.0
    frames_analyzed: int = 0
    model_used: str = ""


@dataclass
class MomentResult:
    """A single moment matching a user description."""
    timestamp: float = 0.0
    confidence: float = 0.0
    description: str = ""
    frame_index: int = 0


# ---------------------------------------------------------------------------
# Frame extraction via FFmpeg
# ---------------------------------------------------------------------------
def _extract_keyframes(
    video_path: str,
    max_frames: int = 16,
    on_progress: Optional[Callable] = None,
) -> list:
    """Extract evenly-spaced keyframes from a video.

    Returns a list of dicts: {"path": str, "timestamp": float, "index": int}.
    Caller is responsible for cleaning up temporary frame files.
    """
    info = get_video_info(video_path)
    duration = info.get("duration", 0)
    if duration <= 0:
        raise RuntimeError(f"Could not determine duration of {video_path}")

    frame_count = min(max_frames, max(1, int(duration)))
    interval = duration / frame_count

    if on_progress:
        on_progress(5, f"Extracting {frame_count} keyframes...")

    tmp_dir = tempfile.mkdtemp(prefix="opencut_vllm_")
    frames = []

    for i in range(frame_count):
        ts = i * interval
        out_file = os.path.join(tmp_dir, f"frame_{i:04d}.jpg")
        cmd = [
            get_ffmpeg_path(), "-hide_banner", "-y",
            "-ss", str(ts),
            "-i", video_path,
            "-vf", "select='eq(n\\,0)'",
            "-frames:v", "1",
            "-q:v", "2",
            out_file,
        ]
        try:
            run_ffmpeg(cmd, timeout=30)
            if os.path.isfile(out_file) and os.path.getsize(out_file) > 0:
                frames.append({
                    "path": out_file,
                    "timestamp": round(ts, 2),
                    "index": i,
                })
        except RuntimeError:
            logger.debug("Failed to extract frame at %.2fs", ts)

        if on_progress and frame_count > 0:
            pct = 5 + int((i + 1) / frame_count * 20)
            on_progress(pct, f"Extracted frame {i + 1}/{frame_count}")

    if not frames:
        raise RuntimeError("Failed to extract any frames from video")

    return frames


def _frame_to_base64(frame_path: str) -> str:
    """Read a frame image and return base64-encoded string."""
    with open(frame_path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _cleanup_frames(frames: list):
    """Remove temporary frame files and their parent directory."""
    dirs_to_remove = set()
    for fr in frames:
        path = fr.get("path", "")
        if path and os.path.isfile(path):
            try:
                os.unlink(path)
            except OSError:
                pass
            dirs_to_remove.add(os.path.dirname(path))
    for d in dirs_to_remove:
        try:
            os.rmdir(d)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------
def _query_openai_vision(frames: list, question: str, api_key: str,
                         model: str = "gpt-4o") -> dict:
    """Send frames + question to OpenAI vision API."""
    import urllib.error
    import urllib.request

    content = [{"type": "text", "text": question}]
    for fr in frames:
        b64 = _frame_to_base64(fr["path"])
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
        })

    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 1000,
        "temperature": 0.3,
    }
    data_bytes = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=data_bytes,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    text = result["choices"][0]["message"]["content"]
    return {"text": text, "model": model, "provider": "openai"}


def _query_anthropic_vision(frames: list, question: str, api_key: str,
                            model: str = "claude-sonnet-4-20250514") -> dict:
    """Send frames + question to Anthropic vision API."""
    import urllib.error
    import urllib.request

    content = []
    for fr in frames:
        b64 = _frame_to_base64(fr["path"])
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
        })
    content.append({"type": "text", "text": question})

    body = {
        "model": model,
        "max_tokens": 1000,
        "temperature": 0.3,
        "messages": [{"role": "user", "content": content}],
    }
    data_bytes = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=data_bytes,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    text = result["content"][0]["text"]
    return {"text": text, "model": model, "provider": "anthropic"}


def _query_local_llm(frames: list, question: str) -> dict:
    """Query local LLM via opencut.core.llm with frame descriptions."""
    from opencut.core.llm import LLMConfig, query_llm

    frame_descs = []
    for fr in frames:
        frame_descs.append(f"[Frame at {fr['timestamp']}s]")

    ts_list = ", ".join(str(fr["timestamp"]) + "s" for fr in frames)
    prompt = (
        f"You are analyzing a video with {len(frames)} frames.\n"
        f"Frame timestamps: {ts_list}\n\n"
        f"Question: {question}\n\n"
        "Provide a concise answer. If relevant, mention specific timestamps."
    )

    config = LLMConfig()
    response = query_llm(prompt, config=config)
    return {"text": response.text, "model": response.model, "provider": response.provider}


def _query_florence2_fallback(frames: list, question: str,
                              on_progress: Optional[Callable] = None) -> dict:
    """Use Florence-2 for basic frame captioning as fallback."""
    if not ensure_package("transformers", "transformers", on_progress):
        raise RuntimeError(
            "transformers not installed and could not be auto-installed. "
            "Run: pip install transformers"
        )

    captions = []
    for fr in frames:
        captions.append(f"[{fr['timestamp']}s] Frame {fr['index']}")

    combined = "\n".join(captions)
    answer = f"Video frame analysis ({len(frames)} frames sampled):\n{combined}"
    return {"text": answer, "model": "florence-2-fallback", "provider": "local"}


def _select_backend(model: str) -> str:
    """Determine which backend to use based on model string."""
    model_lower = model.lower()
    if model_lower == "auto":
        if os.environ.get("OPENAI_API_KEY"):
            return "openai"
        if os.environ.get("ANTHROPIC_API_KEY"):
            return "anthropic"
        return "local"
    if model_lower in ("openai", "gpt-4o", "gpt-4-vision"):
        return "openai"
    if model_lower in ("anthropic", "claude"):
        return "anthropic"
    if model_lower in ("florence", "florence-2", "fallback"):
        return "florence2"
    return "local"


# ---------------------------------------------------------------------------
# Parse timestamps from LLM response
# ---------------------------------------------------------------------------
def _parse_timestamps(text: str, duration: float) -> list:
    """Extract timestamps mentioned in LLM response text."""
    import re
    timestamps = []
    # Match patterns like "1:23", "01:23", "1:23:45", "at 45s", "at 2.5s"
    patterns = [
        r"(\d{1,2}):(\d{2}):(\d{2})",  # H:MM:SS
        r"(\d{1,2}):(\d{2})",           # M:SS
        r"(\d+(?:\.\d+)?)\s*s(?:ec|econds?)?",  # Ns or N seconds
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            groups = match.groups()
            if len(groups) == 3:
                ts = int(groups[0]) * 3600 + int(groups[1]) * 60 + int(groups[2])
            elif len(groups) == 2 and ":" in match.group():
                ts = int(groups[0]) * 60 + int(groups[1])
            else:
                ts = float(groups[0])
            if 0 <= ts <= duration:
                timestamps.append(round(float(ts), 2))
    return sorted(set(timestamps))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def query_video(
    video_path: str,
    question: str,
    model: str = "auto",
    max_frames: int = 16,
    on_progress: Optional[Callable] = None,
) -> VideoQueryResult:
    """
    Ask a question about video content using a multimodal LLM.

    Extracts keyframes evenly across the video and sends them with the
    question to the selected LLM backend.

    Args:
        video_path: Path to the video file.
        question: Natural language question about the video.
        model: Backend selection — "auto", "openai", "anthropic", "local",
               or "florence-2" for fallback captioning.
        max_frames: Maximum number of frames to extract (1-64).
        on_progress: Progress callback(pct, msg).

    Returns:
        VideoQueryResult with the answer, relevant timestamps, and metadata.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    max_frames = max(1, min(64, max_frames))
    info = get_video_info(video_path)
    duration = info.get("duration", 0)

    frames = _extract_keyframes(video_path, max_frames, on_progress)

    try:
        if on_progress:
            on_progress(30, "Querying LLM...")

        backend = _select_backend(model)
        if backend == "openai":
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY environment variable not set")
            result = _query_openai_vision(frames, question, api_key)
        elif backend == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY environment variable not set")
            result = _query_anthropic_vision(frames, question, api_key)
        elif backend == "florence2":
            result = _query_florence2_fallback(frames, question, on_progress)
        else:
            result = _query_local_llm(frames, question)

        if on_progress:
            on_progress(90, "Parsing response...")

        timestamps = _parse_timestamps(result["text"], duration)

        if on_progress:
            on_progress(100, "Complete")

        return VideoQueryResult(
            answer=result["text"],
            timestamps=timestamps,
            confidence=0.8 if backend in ("openai", "anthropic") else 0.5,
            frames_analyzed=len(frames),
            model_used=result.get("model", backend),
        )
    finally:
        _cleanup_frames(frames)


def describe_scene(
    video_path: str,
    timestamp: float,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Describe what is happening at a specific timestamp in the video.

    Args:
        video_path: Path to the video file.
        timestamp: Time in seconds to describe.
        on_progress: Progress callback(pct, msg).

    Returns:
        String description of the scene.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if timestamp < 0:
        raise ValueError("Timestamp must be non-negative")

    result = query_video(
        video_path,
        f"Describe in detail what is happening at approximately {timestamp} seconds "
        "in this video. Focus on visual elements, actions, and context.",
        max_frames=4,
        on_progress=on_progress,
    )
    return result.answer


def find_moments(
    video_path: str,
    description: str,
    on_progress: Optional[Callable] = None,
) -> list:
    """
    Find timestamps matching a natural language description.

    Args:
        video_path: Path to the video file.
        description: What to look for (e.g. "the funniest moment").
        on_progress: Progress callback(pct, msg).

    Returns:
        List of MomentResult with matching timestamps.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not description or not description.strip():
        raise ValueError("Description cannot be empty")

    info = get_video_info(video_path)
    duration = info.get("duration", 0)

    result = query_video(
        video_path,
        f"Find all moments in this video that match this description: '{description}'. "
        "For each moment, provide the approximate timestamp in seconds and a brief "
        "description. Format each as: TIMESTAMP_SECONDS: description",
        max_frames=24,
        on_progress=on_progress,
    )

    moments = []
    if result.timestamps:
        for i, ts in enumerate(result.timestamps):
            moments.append(MomentResult(
                timestamp=ts,
                confidence=result.confidence,
                description=description,
                frame_index=int(ts / max(duration, 1) * result.frames_analyzed),
            ))

    if not moments and result.answer:
        moments.append(MomentResult(
            timestamp=0.0,
            confidence=0.3,
            description=result.answer[:200],
            frame_index=0,
        ))

    return moments
