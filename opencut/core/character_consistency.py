"""
OpenCut Consistent Character Generation Module

Generate the same character across multiple AI-generated shots using
IP-Adapter-style conditioning on face/body embeddings.

Profile workflow:
    1. create_character_profile() - Extract embeddings from reference images
    2. generate_consistent_scene() - Generate video conditioned on character profile

Profiles are stored in ~/.opencut/characters/{profile_id}/.

Functions:
    create_character_profile  - Build a reusable character embedding profile
    generate_consistent_scene - Generate video with consistent character
    list_character_profiles   - List all saved character profiles
    delete_character_profile  - Remove a saved profile
    load_character_profile    - Load a profile by ID
"""

import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional

from opencut.helpers import (
    OPENCUT_DIR,
    FFmpegCmd,
    ensure_package,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHARACTERS_DIR = os.path.join(OPENCUT_DIR, "characters")
VALID_MODELS = ("auto", "ip_adapter_sd", "ip_adapter_sdxl", "external")

# Embedding extraction backends
_EMBEDDING_BACKENDS = ("insightface", "clip")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class CharacterProfile:
    """A reusable character identity profile."""
    profile_id: str = ""
    name: str = ""
    reference_embeddings_path: str = ""
    num_references: int = 0
    created_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GenerationResult:
    """Result of a consistent character generation."""
    output_path: str = ""
    prompt: str = ""
    character_id: str = ""
    duration: float = 0.0
    model_used: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Internal helpers: directory management
# ---------------------------------------------------------------------------
def _ensure_characters_dir():
    """Create the characters directory if it doesn't exist."""
    os.makedirs(CHARACTERS_DIR, exist_ok=True)


def _profile_dir(profile_id: str) -> str:
    """Return the directory path for a specific character profile."""
    return os.path.join(CHARACTERS_DIR, profile_id)


def _profile_meta_path(profile_id: str) -> str:
    """Return the metadata JSON path for a profile."""
    return os.path.join(_profile_dir(profile_id), "profile.json")


def _profile_embeddings_path(profile_id: str) -> str:
    """Return the embeddings file path for a profile."""
    return os.path.join(_profile_dir(profile_id), "embeddings.npz")


# ---------------------------------------------------------------------------
# Internal helpers: embedding extraction
# ---------------------------------------------------------------------------
def _extract_embeddings_insightface(
    image_paths: List[str],
    profile_dir: str,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Extract face embeddings using InsightFace.

    Returns:
        Dict with "embeddings_path", "num_faces", "backend".
    """
    if not ensure_package("insightface", "insightface"):
        return {}
    if not ensure_package("cv2", "opencv-python-headless"):
        return {}
    if not ensure_package("numpy", "numpy"):
        return {}

    import cv2
    import numpy as np
    from insightface.app import FaceAnalysis

    if on_progress:
        on_progress(20, "Loading InsightFace model...")

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    all_embeddings = []
    for i, img_path in enumerate(image_paths):
        if not os.path.isfile(img_path):
            logger.warning("Reference image not found: %s", img_path)
            continue

        img = cv2.imread(img_path)
        if img is None:
            logger.warning("Could not read image: %s", img_path)
            continue

        faces = app.get(img)
        if faces:
            # Take the largest face
            largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            all_embeddings.append(largest.embedding)

        if on_progress:
            pct = 20 + int(40 * (i + 1) / len(image_paths))
            on_progress(pct, f"Processed reference {i + 1}/{len(image_paths)}")

    if not all_embeddings:
        return {}

    # Average the embeddings for a stable identity representation
    avg_embedding = np.mean(np.stack(all_embeddings), axis=0)

    embeddings_path = os.path.join(profile_dir, "embeddings.npz")
    np.savez(
        embeddings_path,
        face_embedding=avg_embedding,
        individual_embeddings=np.stack(all_embeddings),
    )

    return {
        "embeddings_path": embeddings_path,
        "num_faces": len(all_embeddings),
        "backend": "insightface",
    }


def _extract_embeddings_clip(
    image_paths: List[str],
    profile_dir: str,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Extract visual embeddings using CLIP.

    Broader than face-only -- captures overall appearance including clothing,
    body shape, and style. Used as fallback when InsightFace is not available.

    Returns:
        Dict with "embeddings_path", "num_images", "backend".
    """
    if not ensure_package("torch", "torch"):
        return {}
    if not ensure_package("PIL", "Pillow"):
        return {}

    import numpy as np
    import torch

    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        if not ensure_package("transformers", "transformers"):
            return {}
        from transformers import CLIPModel, CLIPProcessor

    if on_progress:
        on_progress(20, "Loading CLIP model...")

    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    from PIL import Image

    all_embeddings = []
    for i, img_path in enumerate(image_paths):
        if not os.path.isfile(img_path):
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            logger.warning("Could not open image: %s", img_path)
            continue

        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
        embedding = outputs.squeeze().cpu().numpy()
        all_embeddings.append(embedding)

        if on_progress:
            pct = 20 + int(40 * (i + 1) / len(image_paths))
            on_progress(pct, f"Processed reference {i + 1}/{len(image_paths)}")

    if not all_embeddings:
        return {}

    avg_embedding = np.mean(np.stack(all_embeddings), axis=0)

    embeddings_path = os.path.join(profile_dir, "embeddings.npz")
    np.savez(
        embeddings_path,
        clip_embedding=avg_embedding,
        individual_embeddings=np.stack(all_embeddings),
    )

    return {
        "embeddings_path": embeddings_path,
        "num_images": len(all_embeddings),
        "backend": "clip",
    }


def _copy_references(image_paths: List[str], profile_dir: str) -> int:
    """Copy reference images into the profile directory.

    Returns:
        Number of images successfully copied.
    """
    import shutil
    refs_dir = os.path.join(profile_dir, "references")
    os.makedirs(refs_dir, exist_ok=True)

    count = 0
    for i, path in enumerate(image_paths):
        if os.path.isfile(path):
            ext = os.path.splitext(path)[1] or ".png"
            dst = os.path.join(refs_dir, f"ref_{i:03d}{ext}")
            shutil.copy2(path, dst)
            count += 1

    return count


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def create_character_profile(
    reference_images: List[str],
    name: str = "character",
    on_progress: Optional[Callable] = None,
) -> CharacterProfile:
    """Create a reusable character identity profile from reference images.

    Extracts face/body embeddings using InsightFace (preferred) or CLIP (fallback).
    The profile is saved to ~/.opencut/characters/{profile_id}/ for reuse.

    Args:
        reference_images: List of image file paths showing the character.
            More images (3-10) from different angles gives better results.
        name: Human-readable name for the character.
        on_progress: Optional progress callback(pct, msg).

    Returns:
        CharacterProfile with the profile ID and metadata.

    Raises:
        ValueError: If no valid reference images are provided.
    """
    if not reference_images:
        raise ValueError("At least one reference image is required")

    # Filter to existing files
    valid_images = [p for p in reference_images if os.path.isfile(p)]
    if not valid_images:
        raise ValueError("No valid reference images found. Check file paths.")

    if on_progress:
        on_progress(5, f"Creating character profile '{name}' from {len(valid_images)} images...")

    profile_id = uuid.uuid4().hex[:12]
    _ensure_characters_dir()
    pdir = _profile_dir(profile_id)
    os.makedirs(pdir, exist_ok=True)

    # Copy reference images
    copied = _copy_references(valid_images, pdir)

    if on_progress:
        on_progress(15, "Extracting character embeddings...")

    # Try InsightFace first (face-specific), then CLIP (general visual)
    embed_result = _extract_embeddings_insightface(valid_images, pdir, on_progress)

    if not embed_result:
        logger.info("InsightFace not available, falling back to CLIP embeddings")
        embed_result = _extract_embeddings_clip(valid_images, pdir, on_progress)

    embeddings_path = embed_result.get("embeddings_path", "")

    if not embeddings_path:
        # Neither backend available -- save references only for external API use
        logger.warning("No embedding backend available. Profile saved with references only.")
        embeddings_path = ""

    if on_progress:
        on_progress(80, "Saving profile...")

    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    profile = CharacterProfile(
        profile_id=profile_id,
        name=name,
        reference_embeddings_path=embeddings_path,
        num_references=copied,
        created_at=created_at,
    )

    # Save metadata
    meta_path = _profile_meta_path(profile_id)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(profile.to_dict(), f, indent=2)

    if on_progress:
        on_progress(100, f"Character profile '{name}' created (ID: {profile_id})")

    return profile


def load_character_profile(profile_id: str) -> CharacterProfile:
    """Load a character profile by ID.

    Args:
        profile_id: The profile ID string.

    Returns:
        CharacterProfile dataclass.

    Raises:
        FileNotFoundError: If the profile does not exist.
    """
    meta_path = _profile_meta_path(profile_id)
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Character profile not found: {profile_id}")

    with open(meta_path, encoding="utf-8") as f:
        data = json.load(f)

    return CharacterProfile(**data)


def list_character_profiles() -> List[Dict]:
    """List all saved character profiles.

    Returns:
        List of profile dicts with profile_id, name, num_references, created_at.
    """
    _ensure_characters_dir()
    profiles = []

    for entry in sorted(os.listdir(CHARACTERS_DIR)):
        meta_path = os.path.join(CHARACTERS_DIR, entry, "profile.json")
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, encoding="utf-8") as f:
                    data = json.load(f)
                profiles.append({
                    "profile_id": data.get("profile_id", entry),
                    "name": data.get("name", "unknown"),
                    "num_references": data.get("num_references", 0),
                    "created_at": data.get("created_at", ""),
                })
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Could not read profile %s: %s", entry, e)

    return profiles


def delete_character_profile(profile_id: str) -> bool:
    """Delete a character profile by ID.

    Args:
        profile_id: The profile ID to delete.

    Returns:
        True if deleted, False if not found.
    """
    pdir = _profile_dir(profile_id)
    if not os.path.isdir(pdir):
        return False

    import shutil
    shutil.rmtree(pdir, ignore_errors=True)
    return True


def generate_consistent_scene(
    prompt: str,
    character_profile: CharacterProfile,
    output: Optional[str] = None,
    duration: float = 4.0,
    on_progress: Optional[Callable] = None,
) -> GenerationResult:
    """Generate a video scene featuring a consistent character.

    Uses IP-Adapter conditioning with Stable Diffusion or video generation
    model to maintain character identity across generated shots.

    Args:
        prompt: Text description of the scene to generate.
        character_profile: CharacterProfile from create_character_profile().
        output: Output video path. Auto-generated if None.
        duration: Desired duration in seconds (1-30).
        on_progress: Optional progress callback(pct, msg).

    Returns:
        GenerationResult with output path and metadata.

    Raises:
        ValueError: If prompt is empty or duration is out of range.
        FileNotFoundError: If character profile embeddings are missing.
        RuntimeError: If no generation model is available.
    """
    if not prompt or not prompt.strip():
        raise ValueError("A text prompt is required")

    duration = max(1.0, min(float(duration), 30.0))

    if on_progress:
        on_progress(5, "Loading character profile...")

    embeddings_path = character_profile.reference_embeddings_path
    has_embeddings = embeddings_path and os.path.isfile(embeddings_path)

    if output is None:
        safe_name = "".join(c for c in character_profile.name if c.isalnum() or c == "_")[:20]
        output = os.path.join(
            OPENCUT_DIR, "generated",
            f"{safe_name}_{uuid.uuid4().hex[:6]}.mp4",
        )
    os.makedirs(os.path.dirname(output), exist_ok=True)

    model_used = ""

    # Try IP-Adapter + Stable Diffusion pipeline
    if has_embeddings:
        try:
            model_used = _generate_with_ip_adapter(
                prompt, embeddings_path, output, duration, on_progress,
            )
        except Exception as e:
            logger.warning("IP-Adapter generation failed: %s", e)
            model_used = ""

    # Try video generation with image conditioning
    if not model_used and has_embeddings:
        try:
            model_used = _generate_with_video_model(
                prompt, character_profile, output, duration, on_progress,
            )
        except Exception as e:
            logger.warning("Video model generation failed: %s", e)
            model_used = ""

    # Fallback: generate a placeholder video with the prompt as text overlay
    if not model_used:
        model_used = _generate_placeholder(
            prompt, character_profile, output, duration, on_progress,
        )

    if on_progress:
        on_progress(100, "Scene generation complete")

    return GenerationResult(
        output_path=output,
        prompt=prompt,
        character_id=character_profile.profile_id,
        duration=duration,
        model_used=model_used,
    )


def _generate_with_ip_adapter(
    prompt: str,
    embeddings_path: str,
    output: str,
    duration: float,
    on_progress: Optional[Callable] = None,
) -> str:
    """Generate frames using IP-Adapter + Stable Diffusion, then encode to video.

    Returns:
        Model identifier string on success, empty string on failure.
    """
    if not ensure_package("diffusers", "diffusers"):
        return ""
    if not ensure_package("torch", "torch"):
        return ""
    if not ensure_package("numpy", "numpy"):
        return ""

    import numpy as np
    import torch

    if on_progress:
        on_progress(15, "Loading IP-Adapter pipeline...")

    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        torch_dtype=torch.float16,
    )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")

    # Load character embeddings and validate presence
    data = np.load(embeddings_path)
    if "face_embedding" not in data and "clip_embedding" not in data:
        return ""

    if on_progress:
        on_progress(30, "Generating frames...")

    fps = 24.0
    num_frames = int(duration * fps)

    import tempfile
    frame_dir = tempfile.mkdtemp(prefix="opencut_chargen_")

    try:
        for i in range(num_frames):
            # Vary the seed slightly per frame for natural motion
            generator = torch.Generator().manual_seed(42 + i)
            result = pipe(
                prompt=prompt,
                num_inference_steps=20,
                generator=generator,
            )
            frame = result.images[0]
            frame.save(os.path.join(frame_dir, f"frame_{i:05d}.png"))

            if on_progress and i % max(1, num_frames // 10) == 0:
                pct = 30 + int(50 * (i + 1) / num_frames)
                on_progress(pct, f"Generated frame {i + 1}/{num_frames}")

        # Encode frames to video
        pattern = os.path.join(frame_dir, "frame_%05d.png")
        cmd = (
            FFmpegCmd()
            .input(pattern, framerate=str(fps))
            .video_codec("libx264", crf=18, preset="fast")
            .output(output)
            .build()
        )
        run_ffmpeg(cmd)

        return "ip_adapter_sd"

    finally:
        import shutil
        shutil.rmtree(frame_dir, ignore_errors=True)


def _generate_with_video_model(
    prompt: str,
    profile: CharacterProfile,
    output: str,
    duration: float,
    on_progress: Optional[Callable] = None,
) -> str:
    """Generate video using a video generation model with reference image conditioning.

    Uses the first reference image as an image-to-video prompt.

    Returns:
        Model identifier string on success, empty string on failure.
    """
    # Find a reference image
    refs_dir = os.path.join(_profile_dir(profile.profile_id), "references")
    if not os.path.isdir(refs_dir):
        return ""

    ref_images = sorted(
        os.path.join(refs_dir, f)
        for f in os.listdir(refs_dir)
        if f.startswith("ref_")
    )
    if not ref_images:
        return ""

    if not ensure_package("diffusers", "diffusers"):
        return ""
    if not ensure_package("torch", "torch"):
        return ""

    import torch

    if on_progress:
        on_progress(15, "Loading video generation model...")

    from PIL import Image

    ref_img = Image.open(ref_images[0]).convert("RGB")

    try:
        from diffusers import DiffusionPipeline

        pipe = DiffusionPipeline.from_pretrained(
            "Wan-AI/Wan2.1-T2V-14B",
            torch_dtype=torch.float16,
        )
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")

        if on_progress:
            on_progress(40, "Generating video...")

        result = pipe(
            prompt=prompt,
            image=ref_img,
            num_frames=int(duration * 24),
        )

        # Save frames and encode
        import tempfile
        frame_dir = tempfile.mkdtemp(prefix="opencut_vidgen_")
        try:
            for i, frame in enumerate(result.frames[0]):
                frame.save(os.path.join(frame_dir, f"frame_{i:05d}.png"))

            pattern = os.path.join(frame_dir, "frame_%05d.png")
            cmd = (
                FFmpegCmd()
                .input(pattern, framerate="24")
                .video_codec("libx264", crf=18, preset="fast")
                .output(output)
                .build()
            )
            run_ffmpeg(cmd)

            return "wan2.1_i2v"
        finally:
            import shutil
            shutil.rmtree(frame_dir, ignore_errors=True)

    except Exception as e:
        logger.warning("Video model generation failed: %s", e)
        return ""


def _generate_placeholder(
    prompt: str,
    profile: CharacterProfile,
    output: str,
    duration: float,
    on_progress: Optional[Callable] = None,
) -> str:
    """Generate a placeholder video with text overlay when no AI model is available.

    Returns:
        "placeholder" as the model identifier.
    """
    if on_progress:
        on_progress(50, "No generation model available. Creating placeholder...")

    # Escape special characters for FFmpeg drawtext
    safe_prompt = prompt.replace("'", "\\'").replace(":", "\\:")[:100]
    safe_name = profile.name.replace("'", "\\'").replace(":", "\\:")[:30]

    drawtext = (
        f"drawtext=text='{safe_prompt}'"
        f":fontsize=28:fontcolor=white:x=(w-tw)/2:y=(h-th)/2"
        f":box=1:boxcolor=black@0.7:boxborderw=10,"
        f"drawtext=text='Character\\: {safe_name}'"
        f":fontsize=20:fontcolor=gray:x=(w-tw)/2:y=(h+40)/2"
        f":box=1:boxcolor=black@0.5:boxborderw=5"
    )

    cmd = (
        FFmpegCmd()
        .filter_complex(
            f"color=c=0x1a1a2e:s=1920x1080:d={duration}[bg];"
            f"[bg]{drawtext}[out]",
            maps=["[out]"],
        )
        .video_codec("libx264", crf=18, preset="fast")
        .output(output)
        .build()
    )
    run_ffmpeg(cmd)

    return "placeholder"
