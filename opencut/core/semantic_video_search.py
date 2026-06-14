"""
OpenCut Semantic Video Search Module (Category 69.4)

Visual semantic search across project clips using CLIP embeddings.

Pipeline:
    1. Extract key frames from each clip
    2. Compute CLIP embeddings for each frame
    3. Cache embeddings for fast re-query
    4. Search by text query or image similarity
    5. Return ranked results with timestamps and thumbnails

Functions:
    build_clip_index  - Pre-compute CLIP index for a set of clips
    semantic_search   - Search clips by text or image query
"""

import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from opencut.helpers import (
    ensure_package,
    get_ffmpeg_path,
    get_video_info,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# How many key frames to extract per clip for indexing
FRAMES_PER_CLIP = 12

# Default engine — legacy CLIP ViT-B/32
DEFAULT_ENGINE = "clip-vit-b32"

# Engine registry — maps engine IDs to HuggingFace model configs.
# Schema version is bumped when embedding format changes (breaking cache).
SEARCH_ENGINES = {
    "clip-vit-b32": {
        "label": "CLIP ViT-B/32 (OpenAI)",
        "model_name": "openai/clip-vit-base-patch32",
        "family": "clip",
        "embed_dim": 512,
        "schema_version": 1,
    },
    "clip-vit-l14": {
        "label": "CLIP ViT-L/14 (OpenAI)",
        "model_name": "openai/clip-vit-large-patch14",
        "family": "clip",
        "embed_dim": 768,
        "schema_version": 1,
    },
    "siglip-base": {
        "label": "SigLIP ViT-B/16 (Google)",
        "model_name": "google/siglip-base-patch16-224",
        "family": "siglip",
        "embed_dim": 768,
        "schema_version": 1,
    },
    "siglip2-base": {
        "label": "SigLIP 2 ViT-B/16 (Google)",
        "model_name": "google/siglip2-base-patch16-224",
        "family": "siglip",
        "embed_dim": 768,
        "schema_version": 1,
    },
}

# Backward compat alias
CLIP_MODEL_NAME = SEARCH_ENGINES[DEFAULT_ENGINE]["model_name"]

# Cache directory
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".opencut", "clip_cache")

# Maximum results to return
DEFAULT_MAX_RESULTS = 20

# Minimum similarity score for inclusion in results
MIN_SIMILARITY = 0.15

# Thumbnail dimensions
THUMB_WIDTH = 320
THUMB_HEIGHT = 180


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class SearchResult:
    """A single search match."""
    clip_path: str = ""
    clip_name: str = ""
    frame_idx: int = 0
    timestamp: float = 0.0
    score: float = 0.0
    thumbnail_path: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SemanticSearchResult:
    """Aggregated search results."""
    query: str = ""
    query_type: str = "text"
    results: List[Dict] = field(default_factory=list)
    total_clips_searched: int = 0
    total_frames_searched: int = 0
    search_time_ms: int = 0
    index_cached: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ClipIndexEntry:
    """Cached index entry for a single clip."""
    clip_path: str = ""
    clip_hash: str = ""
    frame_count: int = 0
    fps: float = 30.0
    duration: float = 0.0
    frame_timestamps: List[float] = field(default_factory=list)
    frame_paths: List[str] = field(default_factory=list)
    # Embeddings stored separately as numpy arrays in the npz cache

    def to_dict(self) -> dict:
        d = asdict(self)
        # Don't serialise frame_paths in the dict output (internal use)
        d.pop("frame_paths", None)
        return d


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------
def _clip_file_hash(filepath: str) -> str:
    """Compute a fast hash of a clip file for cache invalidation.

    Uses file path + size + mtime to avoid reading the whole file.
    """
    try:
        stat = os.stat(filepath)
        key = f"{os.path.abspath(filepath)}:{stat.st_size}:{stat.st_mtime}"
        return hashlib.sha256(key.encode()).hexdigest()[:24]
    except OSError:
        return hashlib.sha256(filepath.encode()).hexdigest()[:24]


def _cache_path(clip_hash: str, engine: str = DEFAULT_ENGINE) -> str:
    """Return the cache file path for a clip hash, scoped by engine."""
    eng = SEARCH_ENGINES.get(engine, SEARCH_ENGINES[DEFAULT_ENGINE])
    schema_v = eng.get("schema_version", 1)
    engine_dir = os.path.join(CACHE_DIR, f"{engine}_v{schema_v}")
    os.makedirs(engine_dir, exist_ok=True)
    return os.path.join(engine_dir, f"clip_{clip_hash}.npz")


def _load_cached_embeddings(clip_hash: str, engine: str = DEFAULT_ENGINE) -> Optional[Dict]:
    """Load cached embeddings for a clip. Returns None if not cached."""
    path = _cache_path(clip_hash, engine)
    if not os.path.isfile(path):
        return None
    try:
        import numpy as np

        with np.load(path, allow_pickle=False) as cache:
            metadata = json.loads(str(cache["metadata"].item()))
            metadata["embeddings"] = cache["embeddings"]
            return metadata
    except Exception as e:
        logger.warning("Failed to load clip cache %s: %s", path, e)
        return None


def _save_cached_embeddings(clip_hash: str, data: Dict, engine: str = DEFAULT_ENGINE):
    """Save embeddings to cache."""
    path = _cache_path(clip_hash, engine)
    try:
        import numpy as np

        os.makedirs(os.path.dirname(path), exist_ok=True)
        metadata = {key: value for key, value in data.items() if key != "embeddings"}
        embeddings = data.get("embeddings")
        if embeddings is None:
            embeddings = np.zeros((0, 512), dtype="float32")
        np.savez_compressed(
            path,
            metadata=json.dumps(metadata),
            embeddings=embeddings,
        )
    except Exception as e:
        logger.warning("Failed to save clip cache %s: %s", path, e)


def clear_clip_cache():
    """Remove all cached clip embeddings."""
    if os.path.isdir(CACHE_DIR):
        import shutil
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
        logger.info("Cleared CLIP embedding cache at %s", CACHE_DIR)


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------
def _extract_key_frames(
    video_path: str,
    out_dir: str,
    count: int = FRAMES_PER_CLIP,
) -> Tuple[List[str], List[float]]:
    """Extract evenly spaced key frames and return (paths, timestamps).

    Returns frame file paths and corresponding timestamps in seconds.
    """
    info = get_video_info(video_path)
    duration = info.get("duration", 0)
    info.get("fps", 30.0)

    if duration <= 0:
        # Try to get at least one frame
        count = 1

    ffmpeg = get_ffmpeg_path()
    os.makedirs(out_dir, exist_ok=True)
    pattern = os.path.join(out_dir, "kf_%06d.jpg")

    if duration > 0 and count > 1:
        interval = duration / count
        select_expr = f"isnan(prev_selected_t)+gte(t-prev_selected_t\\,{interval:.3f})"
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", video_path,
            "-vf", f"select='{select_expr}',scale={THUMB_WIDTH}:{THUMB_HEIGHT}:force_original_aspect_ratio=decrease",
            "-vsync", "vfr",
            "-frames:v", str(count),
            "-q:v", "4", pattern,
        ]
    else:
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", video_path,
            "-vf", f"scale={THUMB_WIDTH}:{THUMB_HEIGHT}:force_original_aspect_ratio=decrease",
            "-frames:v", "1",
            "-q:v", "4", pattern,
        ]

    run_ffmpeg(cmd, timeout=300)

    frames = sorted(
        os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".jpg")
    )

    # Compute approximate timestamps
    timestamps = []
    if duration > 0 and len(frames) > 0:
        interval = duration / len(frames)
        timestamps = [i * interval for i in range(len(frames))]
    else:
        timestamps = [0.0] * len(frames)

    return frames, timestamps


# ---------------------------------------------------------------------------
# CLIP embedding computation
# ---------------------------------------------------------------------------
def _load_clip_model(engine: str = DEFAULT_ENGINE, on_progress: Optional[Callable] = None):
    """Lazy-load a visual-language model. Returns (model, processor, torch, engine_cfg)."""
    if not ensure_package("transformers", "transformers", on_progress):
        raise RuntimeError("transformers required for semantic search")
    if not ensure_package("torch", "torch", on_progress):
        raise RuntimeError("PyTorch required for semantic search")
    if not ensure_package("PIL", "Pillow", on_progress):
        raise RuntimeError("Pillow required")

    eng = SEARCH_ENGINES.get(engine, SEARCH_ENGINES[DEFAULT_ENGINE])
    model_name = eng["model_name"]
    family = eng.get("family", "clip")

    import torch

    if family == "siglip":
        from transformers import AutoModel, AutoProcessor
        logger.info("Loading SigLIP model: %s", model_name)
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    else:
        from transformers import CLIPModel, CLIPProcessor
        logger.info("Loading CLIP model: %s", model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)

    model.eval()
    return model, processor, torch, eng


def _compute_frame_embeddings(
    frame_paths: List[str],
    model,
    processor,
    torch_mod,
    engine_cfg: Optional[Dict] = None,
    batch_size: int = 16,
    on_progress: Optional[Callable] = None,
) -> "numpy.ndarray":  # noqa: F821
    """Compute image embeddings for a list of frames.

    Returns a numpy array of shape (N, embed_dim).
    """
    import numpy as np
    from PIL import Image

    embed_dim = (engine_cfg or {}).get("embed_dim", 512)
    family = (engine_cfg or {}).get("family", "clip")
    all_embeds = []
    total = len(frame_paths)

    for bi in range(0, total, batch_size):
        batch_paths = frame_paths[bi:bi + batch_size]
        images = []
        for fp in batch_paths:
            try:
                img = Image.open(fp).convert("RGB")
                images.append(img)
            except Exception as e:
                logger.warning("Failed to open frame %s: %s", fp, e)
                images.append(Image.new("RGB", (THUMB_WIDTH, THUMB_HEIGHT), (0, 0, 0)))

        inputs = processor(images=images, return_tensors="pt", padding=True)
        with torch_mod.no_grad():
            if family == "siglip":
                embeds = model.get_image_features(**inputs)
            else:
                embeds = model.get_image_features(**inputs)
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
            all_embeds.append(embeds.cpu().numpy())

        if on_progress:
            done = min(bi + batch_size, total)
            on_progress(0, f"Embedding frames: {done}/{total}")

    return np.concatenate(all_embeds, axis=0) if all_embeds else np.zeros((0, embed_dim))


def _compute_text_embedding(
    query: str, model, processor, torch_mod,
    engine_cfg: Optional[Dict] = None,
) -> "numpy.ndarray":  # noqa: F821
    """Compute text embedding for a query string.

    Uses multiple prompt templates for robustness.
    """

    prompts = [
        query,
        f"a photo of {query}",
        f"a video frame showing {query}",
        f"{query} in a scene",
    ]

    inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
    with torch_mod.no_grad():
        embeds = model.get_text_features(**inputs)
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        avg = embeds.mean(dim=0, keepdim=True)
        avg = avg / avg.norm(dim=-1, keepdim=True)

    return avg.cpu().numpy()


def _compute_image_embedding(
    image_path: str, model, processor, torch_mod,
) -> "numpy.ndarray":  # noqa: F821
    """Compute CLIP embedding for a query image."""
    from PIL import Image

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Query image not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=[img], return_tensors="pt")
    with torch_mod.no_grad():
        embed = model.get_image_features(**inputs)
        embed = embed / embed.norm(dim=-1, keepdim=True)

    return embed.cpu().numpy()


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------
def list_search_engines() -> List[Dict]:
    """Return the list of available search engines with their metadata."""
    engines = []
    for eid, cfg in SEARCH_ENGINES.items():
        engines.append({
            "id": eid,
            "label": cfg["label"],
            "model_name": cfg["model_name"],
            "family": cfg.get("family", "clip"),
            "embed_dim": cfg.get("embed_dim", 512),
            "schema_version": cfg.get("schema_version", 1),
            "default": eid == DEFAULT_ENGINE,
        })
    return engines


def build_clip_index(
    clip_paths: List[str],
    frames_per_clip: int = FRAMES_PER_CLIP,
    force_rebuild: bool = False,
    engine: str = DEFAULT_ENGINE,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Build or update a visual embedding index for a set of video clips.

    Results are cached per-clip and per-engine based on file hash. Re-indexes
    only clips that have changed or that lack cache for the requested engine.

    Args:
        clip_paths: List of video file paths.
        frames_per_clip: Number of key frames per clip.
        force_rebuild: Force re-index even if cache exists.
        engine: Search engine ID from SEARCH_ENGINES registry.
        on_progress: Progress callback(pct, msg).

    Returns:
        Dict with clip_count, total_frames, cached_count, rebuilt_count, engine.
    """
    if engine not in SEARCH_ENGINES:
        engine = DEFAULT_ENGINE

    if not clip_paths:
        raise ValueError("No clip paths provided for indexing")

    valid_paths = [p for p in clip_paths if os.path.isfile(p)]
    if not valid_paths:
        raise FileNotFoundError("None of the provided clip paths exist")

    if on_progress:
        eng_label = SEARCH_ENGINES[engine]["label"]
        on_progress(5, f"Indexing {len(valid_paths)} clips with {eng_label}...")

    model, processor, torch_mod, engine_cfg = _load_clip_model(engine, on_progress)

    cached_count = 0
    rebuilt_count = 0
    total_frames = 0

    for ci, clip_path in enumerate(valid_paths):
        clip_hash = _clip_file_hash(clip_path)

        if not force_rebuild:
            cached = _load_cached_embeddings(clip_hash, engine)
            if cached is not None:
                cached_count += 1
                total_frames += cached.get("frame_count", 0)
                if on_progress:
                    pct = 5 + int(90 * (ci + 1) / len(valid_paths))
                    on_progress(pct, f"Clip {ci+1}/{len(valid_paths)} (cached)")
                continue

        work_dir = tempfile.mkdtemp(prefix=f"clipidx_{ci}_")
        frames, timestamps = _extract_key_frames(clip_path, work_dir, count=frames_per_clip)

        if not frames:
            logger.warning("No frames extracted from %s", clip_path)
            continue

        def _emb_progress(pct_inner, msg=""):
            if on_progress:
                base = 5 + int(90 * ci / len(valid_paths))
                on_progress(base, msg)

        embeddings = _compute_frame_embeddings(
            frames, model, processor, torch_mod,
            engine_cfg=engine_cfg,
            on_progress=_emb_progress,
        )

        info = get_video_info(clip_path)

        cache_data = {
            "clip_path": os.path.abspath(clip_path),
            "clip_hash": clip_hash,
            "frame_count": len(frames),
            "fps": info.get("fps", 30.0),
            "duration": info.get("duration", 0),
            "timestamps": timestamps,
            "frame_paths": frames,
            "embeddings": embeddings,
            "engine": engine,
        }
        _save_cached_embeddings(clip_hash, cache_data, engine)

        rebuilt_count += 1
        total_frames += len(frames)

        if on_progress:
            pct = 5 + int(90 * (ci + 1) / len(valid_paths))
            on_progress(pct, f"Indexed clip {ci+1}/{len(valid_paths)}: {os.path.basename(clip_path)}")

    if on_progress:
        on_progress(100, f"Index complete: {len(valid_paths)} clips, {total_frames} frames")

    return {
        "clip_count": len(valid_paths),
        "total_frames": total_frames,
        "cached_count": cached_count,
        "rebuilt_count": rebuilt_count,
        "engine": engine,
    }


# ---------------------------------------------------------------------------
# Semantic search
# ---------------------------------------------------------------------------
def semantic_search(
    clip_paths: List[str],
    query: str = "",
    query_image: str = "",
    max_results: int = DEFAULT_MAX_RESULTS,
    min_score: float = MIN_SIMILARITY,
    auto_index: bool = True,
    engine: str = DEFAULT_ENGINE,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Search video clips by text query or image similarity.

    If clips haven't been indexed yet and auto_index is True, indexes them first.

    Args:
        clip_paths: List of video file paths to search.
        query: Text search query (e.g., "person laughing").
        query_image: Path to a query image for similarity search.
        max_results: Maximum results to return.
        min_score: Minimum similarity score threshold.
        auto_index: Auto-build index if missing.
        engine: Search engine ID from SEARCH_ENGINES registry.
        on_progress: Progress callback(pct, msg).

    Returns:
        SemanticSearchResult as dict.
    """
    if engine not in SEARCH_ENGINES:
        engine = DEFAULT_ENGINE

    if not query and not query_image:
        raise ValueError("Must provide either query text or query_image")

    if not clip_paths:
        raise ValueError("No clip paths provided for search")

    valid_paths = [p for p in clip_paths if os.path.isfile(p)]
    if not valid_paths:
        raise FileNotFoundError("None of the provided clip paths exist")

    start_time = time.time()
    query_type = "text" if query else "image"

    if on_progress:
        eng_label = SEARCH_ENGINES[engine]["label"]
        on_progress(5, f"Semantic search ({query_type}, {eng_label}): {query or os.path.basename(query_image)}")

    model, processor, torch_mod, engine_cfg = _load_clip_model(engine, on_progress)

    if on_progress:
        on_progress(10, "Computing query embedding...")

    if query:
        query_embed = _compute_text_embedding(query, model, processor, torch_mod, engine_cfg)
    else:
        query_embed = _compute_image_embedding(query_image, model, processor, torch_mod)

    all_results = []
    total_frames_searched = 0
    any_cached = False

    for ci, clip_path in enumerate(valid_paths):
        clip_hash = _clip_file_hash(clip_path)
        cached = _load_cached_embeddings(clip_hash, engine)

        if cached is None:
            if auto_index:
                if on_progress:
                    on_progress(15, f"Indexing {os.path.basename(clip_path)}...")
                work_dir = tempfile.mkdtemp(prefix=f"search_idx_{ci}_")
                frames, timestamps = _extract_key_frames(clip_path, work_dir)
                if not frames:
                    continue
                embeddings = _compute_frame_embeddings(
                    frames, model, processor, torch_mod, engine_cfg=engine_cfg,
                )
                info = get_video_info(clip_path)
                cached = {
                    "clip_path": os.path.abspath(clip_path),
                    "clip_hash": clip_hash,
                    "frame_count": len(frames),
                    "fps": info.get("fps", 30.0),
                    "duration": info.get("duration", 0),
                    "timestamps": timestamps,
                    "frame_paths": frames,
                    "embeddings": embeddings,
                    "engine": engine,
                }
                _save_cached_embeddings(clip_hash, cached, engine)
            else:
                logger.warning("Clip not indexed and auto_index=False: %s", clip_path)
                continue
        else:
            any_cached = True

        embeddings = cached.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            continue

        timestamps = cached.get("timestamps", [])
        frame_paths = cached.get("frame_paths", [])
        clip_fps = cached.get("fps", 30.0)
        frame_count = cached.get("frame_count", 0)
        total_frames_searched += frame_count

        # Compute similarities
        sims = (embeddings @ query_embed.T).squeeze(-1)

        for fi, score in enumerate(sims.tolist()):
            if score < min_score:
                continue

            ts = timestamps[fi] if fi < len(timestamps) else fi / clip_fps
            thumb = frame_paths[fi] if fi < len(frame_paths) else ""

            all_results.append(SearchResult(
                clip_path=clip_path,
                clip_name=os.path.basename(clip_path),
                frame_idx=fi,
                timestamp=round(ts, 3),
                score=round(score, 4),
                thumbnail_path=thumb,
            ))

        if on_progress:
            pct = 15 + int(80 * (ci + 1) / len(valid_paths))
            on_progress(pct, f"Searched {ci+1}/{len(valid_paths)} clips")

    # Sort by score descending
    all_results.sort(key=lambda r: r.score, reverse=True)
    all_results = all_results[:max_results]

    elapsed_ms = int((time.time() - start_time) * 1000)

    if on_progress:
        on_progress(100, f"Found {len(all_results)} results in {elapsed_ms}ms")

    result = SemanticSearchResult(
        query=query or f"image:{os.path.basename(query_image)}",
        query_type=query_type,
        results=[r.to_dict() for r in all_results],
        total_clips_searched=len(valid_paths),
        total_frames_searched=total_frames_searched,
        search_time_ms=elapsed_ms,
        index_cached=any_cached,
    )
    return result.to_dict()
