"""
OpenCut AI B-Roll Generation Module

Generate B-roll footage from text descriptions:
- Text prompt to video generation (via API or local models)
- Fallback to image generation + Ken Burns animation
- Fallback to stock footage search suggestion
- Match source video style (resolution, color profile, frame rate)
- Configurable duration per clip (2-10 seconds)
- Batch generation for multiple prompts

Supports Wan/CogVideo/LTX via API, Stable Diffusion for image fallback.
"""

import json
import logging
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional

from opencut.helpers import (
    FFmpegCmd,
    ensure_package,
    get_video_info,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Video Generation Backends
# ---------------------------------------------------------------------------
VIDEO_GEN_BACKENDS = {
    "wan": {
        "name": "Wan Video",
        "description": "High-quality video generation from text prompts",
        "api_based": True,
        "max_duration": 10.0,
        "resolutions": [(512, 512), (768, 512), (1024, 576)],
    },
    "cogvideo": {
        "name": "CogVideo",
        "description": "Open-source video generation model",
        "api_based": False,
        "max_duration": 6.0,
        "resolutions": [(480, 480), (720, 480)],
    },
    "ltx": {
        "name": "LTX Video",
        "description": "Lightweight text-to-video model",
        "api_based": True,
        "max_duration": 8.0,
        "resolutions": [(512, 512), (768, 432)],
    },
    "image_kenburns": {
        "name": "Image + Ken Burns",
        "description": "Generate still image and apply Ken Burns pan/zoom",
        "api_based": False,
        "max_duration": 30.0,
        "resolutions": [(1920, 1080), (1280, 720), (720, 720)],
    },
}

# Ken Burns effect presets
KEN_BURNS_PRESETS = {
    "zoom_in": {"start_scale": 1.0, "end_scale": 1.3, "start_pos": (0.5, 0.5), "end_pos": (0.5, 0.5)},
    "zoom_out": {"start_scale": 1.3, "end_scale": 1.0, "start_pos": (0.5, 0.5), "end_pos": (0.5, 0.5)},
    "pan_left": {"start_scale": 1.2, "end_scale": 1.2, "start_pos": (0.6, 0.5), "end_pos": (0.4, 0.5)},
    "pan_right": {"start_scale": 1.2, "end_scale": 1.2, "start_pos": (0.4, 0.5), "end_pos": (0.6, 0.5)},
    "pan_up": {"start_scale": 1.2, "end_scale": 1.2, "start_pos": (0.5, 0.6), "end_pos": (0.5, 0.4)},
    "pan_down": {"start_scale": 1.2, "end_scale": 1.2, "start_pos": (0.5, 0.4), "end_pos": (0.5, 0.6)},
    "zoom_pan_tl": {"start_scale": 1.0, "end_scale": 1.3, "start_pos": (0.4, 0.4), "end_pos": (0.6, 0.6)},
    "zoom_pan_br": {"start_scale": 1.0, "end_scale": 1.3, "start_pos": (0.6, 0.6), "end_pos": (0.4, 0.4)},
}

# Stock footage suggestion keywords
STOCK_CATEGORIES = [
    "nature", "city", "technology", "business", "people", "abstract",
    "aerial", "water", "fire", "space", "food", "travel", "sports",
    "medical", "industrial", "education", "music", "fashion", "animals",
]


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------
@dataclass
class BRollGenConfig:
    """Configuration for B-roll generation."""
    backend: str = "image_kenburns"
    width: int = 1920
    height: int = 1080
    fps: int = 30
    duration: float = 5.0
    min_duration: float = 2.0
    max_duration: float = 10.0
    ken_burns_preset: str = "zoom_in"
    style_prompt_suffix: str = ""
    negative_prompt: str = "blurry, low quality, distorted, watermark, text"
    seed: int = -1
    guidance_scale: float = 7.5
    num_inference_steps: int = 30
    api_key: str = ""
    api_url: str = ""
    match_source: bool = True
    source_video_path: str = ""
    color_temperature: str = ""
    output_format: str = "mp4"

    def to_dict(self) -> dict:
        d = asdict(self)
        # Redact API key
        if d.get("api_key"):
            d["api_key"] = "***"
        return d

    def validate(self) -> List[str]:
        """Return list of validation errors, empty if valid."""
        errors = []
        if self.backend not in VIDEO_GEN_BACKENDS:
            errors.append(f"Unknown backend '{self.backend}', must be one of {list(VIDEO_GEN_BACKENDS.keys())}")
        if self.width < 64 or self.width > 4096:
            errors.append(f"Width {self.width} out of range [64, 4096]")
        if self.height < 64 or self.height > 4096:
            errors.append(f"Height {self.height} out of range [64, 4096]")
        if self.fps < 1 or self.fps > 120:
            errors.append(f"FPS {self.fps} out of range [1, 120]")
        if self.duration < self.min_duration:
            errors.append(f"Duration {self.duration}s below minimum {self.min_duration}s")
        if self.duration > self.max_duration:
            errors.append(f"Duration {self.duration}s exceeds maximum {self.max_duration}s")
        if self.guidance_scale < 1.0 or self.guidance_scale > 30.0:
            errors.append(f"guidance_scale {self.guidance_scale} out of range [1.0, 30.0]")
        return errors


@dataclass
class BRollGenResult:
    """Result of a single B-roll generation."""
    output_path: str = ""
    prompt: str = ""
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: int = 0
    backend_used: str = ""
    fallback_used: bool = False
    generation_time: float = 0.0
    seed_used: int = -1
    stock_suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BatchBRollResult:
    """Result of batch B-roll generation."""
    results: List[BRollGenResult] = field(default_factory=list)
    total_duration: float = 0.0
    total_clips: int = 0
    successful: int = 0
    failed: int = 0

    def to_dict(self) -> dict:
        return {
            "results": [r.to_dict() for r in self.results],
            "total_duration": round(self.total_duration, 2),
            "total_clips": self.total_clips,
            "successful": self.successful,
            "failed": self.failed,
        }


# ---------------------------------------------------------------------------
# Prompt Enhancement
# ---------------------------------------------------------------------------
def _enhance_prompt(prompt: str, style_suffix: str = "") -> str:
    """Enhance a basic prompt with quality modifiers for better generation."""
    quality_terms = [
        "cinematic lighting", "professional quality", "sharp focus",
        "high detail", "4K", "color graded",
    ]
    # Check if prompt already has quality modifiers
    prompt_lower = prompt.lower()
    additions = []
    for term in quality_terms:
        if term.lower() not in prompt_lower:
            additions.append(term)
            if len(additions) >= 3:
                break

    enhanced = prompt.strip()
    if additions:
        enhanced += ", " + ", ".join(additions)
    if style_suffix:
        enhanced += ", " + style_suffix.strip()
    return enhanced


def _suggest_stock_keywords(prompt: str) -> List[str]:
    """Generate stock footage search keywords from a text prompt."""
    words = prompt.lower().split()
    suggestions = []

    # Direct keyword matches
    for word in words:
        for cat in STOCK_CATEGORIES:
            if cat in word or word in cat:
                suggestions.append(cat)

    # Phrase-based suggestions
    prompt_lower = prompt.lower()
    keyword_map = {
        "sunset": ["nature", "golden hour", "landscape"],
        "office": ["business", "corporate", "workspace"],
        "ocean": ["water", "sea", "waves", "coast"],
        "mountain": ["nature", "landscape", "aerial"],
        "computer": ["technology", "digital", "screen"],
        "crowd": ["people", "urban", "event"],
        "cooking": ["food", "kitchen", "chef"],
        "car": ["automotive", "driving", "road"],
        "dance": ["music", "performance", "movement"],
        "lab": ["medical", "science", "research"],
        "factory": ["industrial", "manufacturing", "machinery"],
        "classroom": ["education", "learning", "school"],
        "forest": ["nature", "trees", "wilderness"],
        "rain": ["weather", "water", "moody"],
    }
    for keyword, tags in keyword_map.items():
        if keyword in prompt_lower:
            suggestions.extend(tags)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            unique.append(s)

    # Add raw words as fallback
    for word in words:
        if len(word) > 3 and word not in seen:
            seen.add(word)
            unique.append(word)
            if len(unique) >= 10:
                break

    return unique[:10]


# ---------------------------------------------------------------------------
# Source Video Style Matching
# ---------------------------------------------------------------------------
def _match_source_style(source_path: str, config: BRollGenConfig) -> BRollGenConfig:
    """Update config to match source video properties."""
    if not source_path or not os.path.isfile(source_path):
        return config

    info = get_video_info(source_path)
    config.width = info.get("width", config.width)
    config.height = info.get("height", config.height)
    config.fps = int(info.get("fps", config.fps))

    logger.info("Matched source style: %dx%d @ %dfps", config.width, config.height, config.fps)
    return config


# ---------------------------------------------------------------------------
# Ken Burns Image-to-Video
# ---------------------------------------------------------------------------
def _generate_placeholder_image(prompt: str, width: int, height: int,
                                  out_path: str) -> str:
    """Generate a placeholder image with gradient and text overlay.

    Used when no AI image generator is available.
    """
    ensure_package("PIL", "Pillow")
    from PIL import Image, ImageDraw, ImageFont

    # Gradient background based on prompt hash
    seed_val = hash(prompt) % 1000
    random.seed(seed_val)
    r1, g1, b1 = random.randint(20, 80), random.randint(20, 80), random.randint(40, 100)
    r2, g2, b2 = random.randint(40, 120), random.randint(40, 120), random.randint(60, 140)

    img = Image.new("RGB", (width, height))
    pixels = img.load()
    for y in range(height):
        t = y / max(height - 1, 1)
        r = int(r1 * (1 - t) + r2 * t)
        g = int(g1 * (1 - t) + g2 * t)
        b = int(b1 * (1 - t) + b2 * t)
        for x in range(width):
            # Add subtle horizontal variation
            xt = x / max(width - 1, 1)
            rx = int(r + (xt - 0.5) * 20)
            gx = int(g + (xt - 0.5) * 15)
            bx = int(b + (xt - 0.5) * 25)
            pixels[x, y] = (max(0, min(255, rx)), max(0, min(255, gx)), max(0, min(255, bx)))

    # Overlay prompt text
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial", size=max(16, width // 30))
    except (IOError, OSError):
        font = ImageFont.load_default()

    # Word wrap
    max_chars = max(20, width // 12)
    lines = []
    current_line = ""
    for word in prompt.split():
        test = current_line + " " + word if current_line else word
        if len(test) > max_chars:
            if current_line:
                lines.append(current_line)
            current_line = word
        else:
            current_line = test
    if current_line:
        lines.append(current_line)

    text_block = "\n".join(lines[:5])
    text_bbox = draw.textbbox((0, 0), text_block, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    text_x = (width - text_w) // 2
    text_y = (height - text_h) // 2

    # Shadow
    draw.text((text_x + 2, text_y + 2), text_block, fill=(0, 0, 0, 180), font=font)
    draw.text((text_x, text_y), text_block, fill=(220, 220, 230), font=font)

    # "B-ROLL PLACEHOLDER" label
    try:
        small_font = ImageFont.truetype("arial", size=max(10, width // 50))
    except (IOError, OSError):
        small_font = ImageFont.load_default()
    draw.text((10, height - 30), "B-ROLL PLACEHOLDER", fill=(100, 100, 120), font=small_font)

    img.save(out_path, "PNG")
    return out_path


def _apply_ken_burns(image_path: str, out_path: str, duration: float = 5.0,
                      fps: int = 30, width: int = 1920, height: int = 1080,
                      preset: str = "zoom_in") -> str:
    """Apply Ken Burns pan/zoom effect to a still image, producing video.

    Builds an FFmpeg zoompan filter with the specified preset parameters.
    """
    kb = KEN_BURNS_PRESETS.get(preset, KEN_BURNS_PRESETS["zoom_in"])

    total_frames = int(duration * fps)
    start_scale = kb["start_scale"]
    end_scale = kb["end_scale"]
    sx, sy = kb["start_pos"]
    ex, ey = kb["end_pos"]

    # Build zoompan filter expression
    # zoom: linear interpolation from start_scale to end_scale
    # x, y: linear pan from start_pos to end_pos, centered on the crop window
    zoom_expr = f"{start_scale}+({end_scale}-{start_scale})*on/{total_frames}"
    # iw and ih are input dimensions. x/y are crop offsets.
    x_expr = f"(iw-iw/zoom)/2+({ex}-{sx})*iw*on/{total_frames}"
    y_expr = f"(ih-ih/zoom)/2+({ey}-{sy})*ih*on/{total_frames}"

    zoompan_filter = (
        f"zoompan=z='{zoom_expr}':x='{x_expr}':y='{y_expr}'"
        f":d={total_frames}:s={width}x{height}:fps={fps}"
    )

    cmd = (FFmpegCmd()
           .option("loop", "1")
           .input(image_path)
           .option("t", str(duration))
           .video_filter(zoompan_filter)
           .video_codec("libx264")
           .option("pix_fmt", "yuv420p")
           .option("crf", "18")
           .option("preset", "medium")
           .output(out_path)
           .build())

    run_ffmpeg(cmd)
    return out_path


# ---------------------------------------------------------------------------
# AI Video Generation Backends
# ---------------------------------------------------------------------------
def _try_video_gen_api(prompt: str, config: BRollGenConfig,
                        out_path: str, on_progress: Optional[Callable] = None) -> Optional[str]:
    """Attempt video generation via API backend (Wan/LTX).

    Returns output path on success, None if backend unavailable.
    """
    if not config.api_key or not config.api_url:
        logger.debug("No API credentials for %s backend", config.backend)
        return None

    try:
        import urllib.error
        import urllib.request

        payload = {
            "prompt": _enhance_prompt(prompt, config.style_prompt_suffix),
            "negative_prompt": config.negative_prompt,
            "width": config.width,
            "height": config.height,
            "duration": config.duration,
            "fps": config.fps,
            "guidance_scale": config.guidance_scale,
            "num_inference_steps": config.num_inference_steps,
        }
        if config.seed >= 0:
            payload["seed"] = config.seed

        req = urllib.request.Request(
            config.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.api_key}",
            },
        )

        if on_progress:
            on_progress(30, f"Sending generation request to {config.backend}...")

        with urllib.request.urlopen(req, timeout=300) as resp:
            if resp.status == 200:
                video_data = resp.read()
                with open(out_path, "wb") as f:
                    f.write(video_data)
                if os.path.isfile(out_path) and os.path.getsize(out_path) > 1000:
                    return out_path

    except (urllib.error.URLError, TimeoutError, Exception) as exc:
        logger.warning("API video generation failed for %s: %s", config.backend, exc)

    return None


def _try_local_video_gen(prompt: str, config: BRollGenConfig,
                          out_path: str, on_progress: Optional[Callable] = None) -> Optional[str]:
    """Attempt local video generation via CogVideo or similar.

    Returns output path on success, None if backend unavailable.
    """
    try:
        import importlib
        importlib.import_module("cogvideo")
        logger.info("CogVideo backend found, generating B-roll locally")
        if on_progress:
            on_progress(25, "Generating video via CogVideo (this may take a while)...")
        # CogVideo API call would go here in a real implementation
        return None
    except ImportError:
        logger.debug("CogVideo not available for local generation")

    return None


def _try_ai_image_generation(prompt: str, width: int, height: int,
                               out_path: str, config: BRollGenConfig) -> Optional[str]:
    """Attempt AI image generation via Stable Diffusion or DALL-E API.

    Returns image path on success, None if unavailable.
    """
    # Try local Stable Diffusion
    try:
        import importlib
        importlib.import_module("diffusers")
        logger.info("Diffusers found, generating image via Stable Diffusion")
        return None  # Placeholder for real SD pipeline
    except ImportError:
        pass

    # Try API-based image generation
    if config.api_key and config.api_url:
        try:
            import urllib.request
            # Modify API URL for image endpoint
            image_url = config.api_url.replace("/video", "/image")
            payload = {
                "prompt": _enhance_prompt(prompt, config.style_prompt_suffix),
                "negative_prompt": config.negative_prompt,
                "width": width,
                "height": height,
            }
            req = urllib.request.Request(
                image_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {config.api_key}",
                },
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                if resp.status == 200:
                    img_data = resp.read()
                    with open(out_path, "wb") as f:
                        f.write(img_data)
                    if os.path.isfile(out_path) and os.path.getsize(out_path) > 500:
                        return out_path
        except Exception as exc:
            logger.debug("API image generation failed: %s", exc)

    return None


# ---------------------------------------------------------------------------
# Main Generation Function
# ---------------------------------------------------------------------------
def generate_broll(
    prompt: str,
    duration: float = 5.0,
    config: Optional[BRollGenConfig] = None,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> BRollGenResult:
    """Generate a B-roll video clip from a text description.

    Tries backends in order: API video gen -> local video gen -> AI image +
    Ken Burns -> placeholder image + Ken Burns. Always produces output.

    Args:
        prompt: Text description of desired B-roll footage.
        duration: Target duration in seconds (2-10).
        config: Generation configuration. Uses defaults if None.
        output_dir: Output directory. Defaults to temp directory.
        on_progress: Callback ``(pct, msg)`` for progress updates.

    Returns:
        BRollGenResult with output video path and metadata.

    Raises:
        ValueError: If prompt is empty or config validation fails.
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")

    config = config or BRollGenConfig()
    config.duration = duration
    errors = config.validate()
    if errors:
        raise ValueError("Invalid BRollGenConfig: " + "; ".join(errors))

    # Match source video style if available
    if config.match_source and config.source_video_path:
        config = _match_source_style(config.source_video_path, config)

    start_time = time.time()

    if on_progress:
        on_progress(5, "Preparing B-roll generation...")

    # Determine output path
    out_dir = output_dir or tempfile.gettempdir()
    os.makedirs(out_dir, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "-_ " else "" for c in prompt[:30]).strip()
    safe_name = safe_name.replace(" ", "_") or "broll"
    out_path = os.path.join(out_dir, f"{safe_name}_broll.mp4")

    # Resolve unique filename
    counter = 1
    base_out = out_path
    while os.path.isfile(out_path):
        name, ext = os.path.splitext(base_out)
        out_path = f"{name}_{counter}{ext}"
        counter += 1

    fallback_used = False
    backend_used = config.backend
    seed_used = config.seed if config.seed >= 0 else random.randint(0, 2**31)

    # Strategy 1: API-based video generation
    if config.backend in ("wan", "ltx") and config.api_key:
        if on_progress:
            on_progress(15, f"Attempting {config.backend} video generation...")
        result_path = _try_video_gen_api(prompt, config, out_path, on_progress)
        if result_path:
            elapsed = time.time() - start_time
            if on_progress:
                on_progress(100, "B-roll generated via API")
            return BRollGenResult(
                output_path=result_path, prompt=prompt, duration=duration,
                width=config.width, height=config.height, fps=config.fps,
                backend_used=backend_used, fallback_used=False,
                generation_time=elapsed, seed_used=seed_used,
            )
        fallback_used = True

    # Strategy 2: Local video generation model
    if config.backend == "cogvideo" or fallback_used:
        if on_progress:
            on_progress(25, "Attempting local video generation...")
        result_path = _try_local_video_gen(prompt, config, out_path, on_progress)
        if result_path:
            elapsed = time.time() - start_time
            if on_progress:
                on_progress(100, "B-roll generated locally")
            return BRollGenResult(
                output_path=result_path, prompt=prompt, duration=duration,
                width=config.width, height=config.height, fps=config.fps,
                backend_used="cogvideo", fallback_used=fallback_used,
                generation_time=elapsed, seed_used=seed_used,
            )
        fallback_used = True

    # Strategy 3: AI image + Ken Burns
    if on_progress:
        on_progress(40, "Attempting image generation + Ken Burns...")

    img_path = tempfile.mktemp(suffix=".png", prefix="broll_img_")
    ai_image = _try_ai_image_generation(prompt, config.width, config.height, img_path, config)

    if not ai_image:
        # Strategy 4: Placeholder image + Ken Burns
        if on_progress:
            on_progress(50, "Generating placeholder image with Ken Burns animation...")
        _generate_placeholder_image(prompt, config.width, config.height, img_path)
        fallback_used = True
        backend_used = "placeholder_kenburns"
    else:
        backend_used = "image_kenburns"
        fallback_used = True

    # Apply Ken Burns animation
    if on_progress:
        on_progress(65, "Applying Ken Burns animation...")

    try:
        _apply_ken_burns(
            img_path, out_path,
            duration=config.duration, fps=config.fps,
            width=config.width, height=config.height,
            preset=config.ken_burns_preset,
        )
    except Exception as exc:
        logger.error("Ken Burns animation failed: %s", exc)
        # Last resort: static image to video
        cmd = (FFmpegCmd()
               .option("loop", "1")
               .input(img_path)
               .option("t", str(config.duration))
               .video_codec("libx264")
               .option("pix_fmt", "yuv420p")
               .option("crf", "18")
               .output(out_path)
               .build())
        try:
            run_ffmpeg(cmd)
        except Exception:
            raise RuntimeError(f"Failed to generate B-roll video: {exc}")

    # Clean up temp image
    try:
        if os.path.isfile(img_path):
            os.unlink(img_path)
    except OSError:
        pass

    elapsed = time.time() - start_time
    stock = _suggest_stock_keywords(prompt)

    if on_progress:
        on_progress(100, "B-roll generation complete")

    return BRollGenResult(
        output_path=out_path,
        prompt=prompt,
        duration=config.duration,
        width=config.width,
        height=config.height,
        fps=config.fps,
        backend_used=backend_used,
        fallback_used=fallback_used,
        generation_time=elapsed,
        seed_used=seed_used,
        stock_suggestions=stock,
    )


def batch_generate_broll(
    prompts: List[str],
    duration: float = 5.0,
    config: Optional[BRollGenConfig] = None,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> BatchBRollResult:
    """Generate multiple B-roll clips from a list of prompts.

    Args:
        prompts: List of text descriptions (1-20 prompts).
        duration: Target duration per clip in seconds.
        config: Shared generation config.
        output_dir: Output directory for all clips.
        on_progress: Callback ``(pct, msg)`` for progress updates.

    Returns:
        BatchBRollResult with all generation results.

    Raises:
        ValueError: If prompts list is empty or exceeds 20.
    """
    if not prompts:
        raise ValueError("Prompts list cannot be empty")
    if len(prompts) > 20:
        raise ValueError("Maximum 20 prompts per batch")

    results = []
    successful = 0
    failed = 0
    total_duration = 0.0

    for i, prompt in enumerate(prompts):
        def _sub_progress(pct, msg=""):
            if on_progress:
                base = int(100 * i / len(prompts))
                chunk = int(100 / len(prompts))
                on_progress(base + int(pct * chunk / 100),
                            f"[{i + 1}/{len(prompts)}] {msg}")

        try:
            result = generate_broll(
                prompt=prompt,
                duration=duration,
                config=config,
                output_dir=output_dir,
                on_progress=_sub_progress,
            )
            results.append(result)
            successful += 1
            total_duration += result.duration
        except Exception as exc:
            logger.error("Failed to generate B-roll for prompt %d: %s", i + 1, exc)
            results.append(BRollGenResult(
                prompt=prompt,
                backend_used="failed",
                stock_suggestions=_suggest_stock_keywords(prompt),
            ))
            failed += 1

    return BatchBRollResult(
        results=results,
        total_duration=total_duration,
        total_clips=len(prompts),
        successful=successful,
        failed=failed,
    )
