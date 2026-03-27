"""
AI B-Roll Generation

Generate short B-roll video clips from text descriptions using text-to-video models.
Supports multiple backends:
- HunyuanVideo (Tencent) — highest quality, requires significant VRAM
- Wan 2.2 (Alibaba) — good balance of quality and speed
- Stable Video Diffusion (Stability AI) — image-to-video variant
- CogVideoX (THUDM) — open source, moderate VRAM requirements

Typical workflow:
1. User describes desired B-roll ("aerial shot of cityscape at sunset")
2. Backend generates short clip (2-6 seconds)
3. Clip saved to output directory for timeline insertion
"""

import logging
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

logger = logging.getLogger("opencut")

# Supported backends in order of preference
BACKENDS = ["cogvideox", "wan", "hunyuan", "svd"]


@dataclass
class GeneratedClip:
    """A generated B-roll clip."""
    output_path: str
    prompt: str
    duration: float      # seconds
    resolution: str      # e.g. "720x480"
    backend: str         # which model was used
    generation_time: float  # seconds to generate
    seed: int


def check_cogvideox_available() -> bool:
    """Check if CogVideoX (lightweight text-to-video) is available."""
    try:
        import diffusers  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def check_wan_available() -> bool:
    """Check if Wan 2.2 text-to-video is available."""
    try:
        import diffusers  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def check_hunyuan_available() -> bool:
    """Check if HunyuanVideo is available."""
    try:
        import diffusers  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def check_svd_available() -> bool:
    """Check if Stable Video Diffusion is available."""
    try:
        import diffusers  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def check_broll_generate_available() -> bool:
    """Check if any text-to-video backend is available."""
    return any([
        check_cogvideox_available(),
        check_wan_available(),
        check_hunyuan_available(),
        check_svd_available(),
    ])


def get_available_backends() -> List[str]:
    """Return list of available text-to-video backends."""
    available = []
    checks = {
        "cogvideox": check_cogvideox_available,
        "wan": check_wan_available,
        "hunyuan": check_hunyuan_available,
        "svd": check_svd_available,
    }
    for name, check in checks.items():
        if check():
            available.append(name)
    return available


def _generate_cogvideox(
    prompt: str,
    output_path: str,
    num_frames: int = 49,
    width: int = 720,
    height: int = 480,
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    seed: Optional[int] = None,
    on_progress: Optional[Callable] = None,
) -> GeneratedClip:
    """Generate video using CogVideoX (THUDM)."""
    import torch
    from diffusers import CogVideoXPipeline
    from diffusers.utils import export_to_video

    if on_progress:
        on_progress(10, "Loading CogVideoX model...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = CogVideoXPipeline.from_pretrained(
        "THUDM/CogVideoX-2b",
        torch_dtype=dtype,
    )
    pipe.to(device)

    try:
        # Enable memory optimizations
        if device == "cuda":
            try:
                pipe.enable_model_cpu_offload()
            except Exception:
                pass
            try:
                pipe.vae.enable_tiling()
            except Exception:
                pass

        if on_progress:
            on_progress(30, "Generating video frames...")

        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)
        else:
            seed = generator.seed()

        t0 = time.monotonic()

        video_frames = pipe(
            prompt=prompt,
            num_frames=num_frames,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).frames[0]

        if on_progress:
            on_progress(85, "Encoding video...")

        export_to_video(video_frames, output_path, fps=8)

        gen_time = time.monotonic() - t0
        duration = num_frames / 8.0

        return GeneratedClip(
            output_path=output_path,
            prompt=prompt,
            duration=round(duration, 2),
            resolution=f"{width}x{height}",
            backend="cogvideox",
            generation_time=round(gen_time, 2),
            seed=seed,
        )
    finally:
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _generate_wan(
    prompt: str,
    output_path: str,
    num_frames: int = 81,
    width: int = 832,
    height: int = 480,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    seed: Optional[int] = None,
    on_progress: Optional[Callable] = None,
) -> GeneratedClip:
    """Generate video using Wan 2.2 (Alibaba)."""
    import torch
    from diffusers import WanPipeline
    from diffusers.utils import export_to_video

    if on_progress:
        on_progress(10, "Loading Wan 2.2 model...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = WanPipeline.from_pretrained(
        "Wan-AI/Wan2.2-T2V-1.3B",
        torch_dtype=dtype,
    )
    pipe.to(device)

    try:
        if device == "cuda":
            try:
                pipe.enable_model_cpu_offload()
            except Exception:
                pass

        if on_progress:
            on_progress(30, "Generating video frames...")

        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)
        else:
            seed = generator.seed()

        t0 = time.monotonic()

        output = pipe(
            prompt=prompt,
            num_frames=num_frames,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        video_frames = output.frames[0]

        if on_progress:
            on_progress(85, "Encoding video...")

        export_to_video(video_frames, output_path, fps=16)

        gen_time = time.monotonic() - t0
        duration = num_frames / 16.0

        return GeneratedClip(
            output_path=output_path,
            prompt=prompt,
            duration=round(duration, 2),
            resolution=f"{width}x{height}",
            backend="wan",
            generation_time=round(gen_time, 2),
            seed=seed,
        )
    finally:
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _generate_hunyuan(
    prompt: str,
    output_path: str,
    num_frames: int = 45,
    width: int = 848,
    height: int = 480,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.0,
    seed: Optional[int] = None,
    on_progress: Optional[Callable] = None,
) -> GeneratedClip:
    """Generate video using HunyuanVideo (Tencent)."""
    import torch
    from diffusers import HunyuanVideoPipeline
    from diffusers.utils import export_to_video

    if on_progress:
        on_progress(10, "Loading HunyuanVideo model...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = HunyuanVideoPipeline.from_pretrained(
        "tencent/HunyuanVideo",
        torch_dtype=dtype,
    )
    pipe.to(device)

    try:
        if device == "cuda":
            try:
                pipe.enable_model_cpu_offload()
            except Exception:
                pass

        if on_progress:
            on_progress(30, "Generating video frames...")

        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)
        else:
            seed = generator.seed()

        t0 = time.monotonic()

        output = pipe(
            prompt=prompt,
            num_frames=num_frames,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        video_frames = output.frames[0]

        if on_progress:
            on_progress(85, "Encoding video...")

        export_to_video(video_frames, output_path, fps=15)

        gen_time = time.monotonic() - t0
        duration = num_frames / 15.0

        return GeneratedClip(
            output_path=output_path,
            prompt=prompt,
            duration=round(duration, 2),
            resolution=f"{width}x{height}",
            backend="hunyuan",
            generation_time=round(gen_time, 2),
            seed=seed,
        )
    finally:
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _generate_svd(
    prompt: str,
    output_path: str,
    reference_image: Optional[str] = None,
    num_frames: int = 25,
    width: int = 1024,
    height: int = 576,
    num_inference_steps: int = 25,
    seed: Optional[int] = None,
    on_progress: Optional[Callable] = None,
) -> GeneratedClip:
    """Generate video using Stable Video Diffusion (image-to-video).

    If no reference_image is provided, first generates a still image from the prompt
    using Stable Diffusion, then animates it with SVD.
    """
    import torch
    from diffusers import StableVideoDiffusionPipeline
    from diffusers.utils import export_to_video
    from PIL import Image

    if on_progress:
        on_progress(5, "Preparing reference image...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Get or generate reference image
    if reference_image and os.path.isfile(reference_image):
        with Image.open(reference_image) as img:
            image = img.resize((width, height)).copy()
    else:
        # Generate still image from prompt first
        if on_progress:
            on_progress(10, "Generating reference image from prompt...")
        sd_pipe = None
        try:
            from diffusers import StableDiffusionPipeline
            sd_pipe = StableDiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1",
                torch_dtype=dtype,
            )
            sd_pipe.to(device)
            gen = torch.Generator(device=device)
            if seed is not None:
                gen.manual_seed(seed)
            image = sd_pipe(prompt, width=width, height=height, generator=gen).images[0]
        except Exception as e:
            raise RuntimeError(
                f"SVD requires a reference image or Stable Diffusion for image generation: {e}"
            )
        finally:
            if sd_pipe is not None:
                del sd_pipe
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if on_progress:
        on_progress(30, "Loading SVD model...")

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=dtype,
    )
    pipe.to(device)

    try:
        if device == "cuda":
            try:
                pipe.enable_model_cpu_offload()
            except Exception:
                pass

        if on_progress:
            on_progress(40, "Generating video from image...")

        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)
        else:
            seed = generator.seed()

        t0 = time.monotonic()

        frames = pipe(
            image,
            num_frames=num_frames,
            decode_chunk_size=8,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).frames[0]

        if on_progress:
            on_progress(85, "Encoding video...")

        export_to_video(frames, output_path, fps=7)

        gen_time = time.monotonic() - t0
        duration = num_frames / 7.0

        return GeneratedClip(
            output_path=output_path,
            prompt=prompt,
            duration=round(duration, 2),
            resolution=f"{width}x{height}",
            backend="svd",
            generation_time=round(gen_time, 2),
            seed=seed,
        )
    finally:
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def generate_broll(
    prompt: str,
    output_dir: Optional[str] = None,
    backend: str = "auto",
    num_frames: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    seed: Optional[int] = None,
    reference_image: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> GeneratedClip:
    """
    Generate a B-roll video clip from a text description.

    Args:
        prompt: Text description of desired B-roll (e.g. "aerial cityscape at sunset").
        output_dir: Directory to save output. Uses temp if None.
        backend: Model backend ("cogvideox", "wan", "hunyuan", "svd", or "auto").
        num_frames: Number of frames to generate (backend-specific defaults if None).
        width: Output width (backend-specific defaults if None).
        height: Output height (backend-specific defaults if None).
        seed: Random seed for reproducibility.
        reference_image: Path to reference image (only used by SVD backend).
        on_progress: Callback(percent, message).

    Returns:
        GeneratedClip with output path and metadata.

    Raises:
        RuntimeError: If no text-to-video backend is available.
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt is required for B-roll generation")

    prompt = prompt.strip()[:500]  # Limit prompt length

    # Resolve output path
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = tempfile.mkdtemp(prefix="opencut_broll_")

    # Sanitize prompt for filename
    safe_name = "".join(c if c.isalnum() or c in " -_" else "" for c in prompt[:40]).strip().replace(" ", "_")
    if not safe_name:
        safe_name = "broll"
    output_path = os.path.join(output_dir, f"{safe_name}_{int(time.time())}.mp4")

    # Resolve backend
    if backend == "auto":
        available = get_available_backends()
        if not available:
            raise RuntimeError(
                "No text-to-video backend available. Install one of:\n"
                "  pip install diffusers torch transformers accelerate\n"
                "Supported models: CogVideoX, Wan 2.2, HunyuanVideo, Stable Video Diffusion"
            )
        backend = available[0]
        logger.info("Auto-selected backend: %s", backend)

    if on_progress:
        on_progress(0, f"Starting B-roll generation ({backend})...")

    generators = {
        "cogvideox": _generate_cogvideox,
        "wan": _generate_wan,
        "hunyuan": _generate_hunyuan,
        "svd": _generate_svd,
    }

    gen_fn = generators.get(backend)
    if gen_fn is None:
        raise ValueError(f"Unknown backend: {backend}. Supported: {list(generators.keys())}")

    kwargs = {"prompt": prompt, "output_path": output_path, "seed": seed, "on_progress": on_progress}
    if num_frames is not None:
        kwargs["num_frames"] = num_frames
    if width is not None:
        kwargs["width"] = width
    if height is not None:
        kwargs["height"] = height
    if backend == "svd" and reference_image:
        kwargs["reference_image"] = reference_image

    result = gen_fn(**kwargs)

    if on_progress:
        on_progress(100, f"B-roll generated: {result.duration}s clip")

    logger.info(
        "Generated B-roll: %s (%.1fs, %s, %s, %.1fs generation time)",
        result.output_path, result.duration, result.resolution,
        result.backend, result.generation_time,
    )

    return result
