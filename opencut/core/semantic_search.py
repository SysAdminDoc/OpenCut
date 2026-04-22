"""
OpenCut Semantic Search v1.28.0

Unified CLIP visual + CLAP audio + Whisper transcript search index.
"""
from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, List, Optional

import numpy as np

from opencut.helpers import _try_import, get_ffmpeg_path

logger = logging.getLogger("opencut")
INSTALL_HINT = "pip install torch transformers laion-clap openai-clip"
_INDEX_PATH = os.path.join(os.path.expanduser("~"), ".opencut", "search_index.json")
_model_cache: dict = {}
_model_lock = threading.Lock()
_index_lock = threading.Lock()


def check_semantic_search_available() -> bool:
    return _try_import("torch") is not None


@dataclass
class SearchResult:
    path: str = ""
    score: float = 0.0
    type: str = "visual"
    timestamp: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str):
        return getattr(self, k)

    def keys(self):
        return ("path", "score", "type", "timestamp", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


@dataclass
class IndexStatus:
    indexed: int = 0
    pending: int = 0
    last_updated: str = ""
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str):
        return getattr(self, k)

    def keys(self):
        return ("indexed", "pending", "last_updated", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def _load_index() -> dict:
    try:
        with open(_INDEX_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"embeddings": {}, "last_updated": ""}


def _save_index(idx: dict) -> None:
    os.makedirs(os.path.dirname(_INDEX_PATH), exist_ok=True)
    tmp = _INDEX_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(idx, f, indent=2)
    os.replace(tmp, _INDEX_PATH)


def _text_fallback_search(query: str, media_paths: List[str], top_k: int) -> List[SearchResult]:
    q = query.lower()
    results = []
    for path in media_paths:
        score = 1.0 if q in os.path.basename(path).lower() else 0.1
        results.append(SearchResult(path=path, score=score, type="text_fallback"))
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]


def _load_clip():
    """Load CLIP model once; thread-safe. Returns (model, preprocess, clip_mod)."""
    with _model_lock:
        if "clip_model" not in _model_cache:
            import clip as clip_mod  # type: ignore
            model, preprocess = clip_mod.load("ViT-B/32")
            _model_cache["clip_model"] = model
            _model_cache["clip_preprocess"] = preprocess
            _model_cache["clip_mod"] = clip_mod
    return (
        _model_cache["clip_model"],
        _model_cache["clip_preprocess"],
        _model_cache["clip_mod"],
    )


def _extract_frame_png(path: str) -> Optional[bytes]:
    """Extract the first frame of a video as PNG bytes; returns None on failure."""
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-i", path,
        "-vf", "select=eq(n\\,0)",
        "-vframes", "1",
        "-f", "image2pipe", "-vcodec", "png",
        "pipe:1",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=15)
        return result.stdout if result.stdout else None
    except Exception:
        return None


def search(
    query: str,
    media_paths: List[str],
    mode: str = "all",
    top_k: int = 10,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> List[SearchResult]:
    if not check_semantic_search_available():
        logger.warning("torch not available; falling back to text search")
        return _text_fallback_search(query, media_paths, top_k)

    import torch
    results = []

    if mode in ("visual", "all"):
        clip_mod_avail = _try_import("clip")
        if clip_mod_avail is not None:
            try:
                if on_progress:
                    on_progress(10, "Loading CLIP model")
                model, preprocess, clip_mod = _load_clip()

                # Encode the text query once
                text_tokens = clip_mod.tokenize([query])
                with torch.no_grad():
                    text_features = model.encode_text(text_tokens)
                    text_features /= text_features.norm(dim=-1, keepdim=True)

                # Prefer pre-built embeddings from the index when available
                with _index_lock:
                    idx = _load_index()
                embeddings = idx.get("embeddings", {})

                for i, path in enumerate(media_paths):
                    if on_progress and i % 5 == 0:
                        on_progress(
                            20 + int(i / len(media_paths) * 40),
                            f"Scoring {os.path.basename(path)}",
                        )
                    try:
                        stored = embeddings.get(path, {}).get("embedding")
                        if stored is not None:
                            img_feat = torch.tensor(stored, dtype=torch.float32).unsqueeze(0)
                            img_feat /= img_feat.norm(dim=-1, keepdim=True)
                        else:
                            png = _extract_frame_png(path)
                            if not png:
                                raise ValueError("No frame extracted")
                            from PIL import Image  # type: ignore
                            img = Image.open(io.BytesIO(png)).convert("RGB")
                            img_tensor = preprocess(img).unsqueeze(0)
                            with torch.no_grad():
                                img_feat = model.encode_image(img_tensor)
                                img_feat /= img_feat.norm(dim=-1, keepdim=True)
                        score = float((img_feat @ text_features.T).squeeze())
                        results.append(SearchResult(path=path, score=score, type="visual"))
                    except Exception as e:
                        logger.debug("CLIP frame scoring failed for %s: %s", path, e)
                        results.append(SearchResult(path=path, score=0.0, type="visual"))
            except Exception as e:
                logger.warning("CLIP search failed: %s", e)

    if mode in ("transcript", "all"):
        q_lower = query.lower()
        for path in media_paths:
            score = 0.2 if q_lower in os.path.basename(path).lower() else 0.0
            existing = next((r for r in results if r.path == path), None)
            if existing:
                existing.score = max(existing.score, score)
            elif score > 0:
                results.append(SearchResult(path=path, score=score, type="transcript"))

    if not results:
        results = _text_fallback_search(query, media_paths, top_k)

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]


def build_index(
    media_paths: List[str],
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> IndexStatus:
    """Build or update the CLIP visual embedding index for a list of media files."""
    clip_mod_avail = _try_import("clip")
    import torch

    if clip_mod_avail is not None:
        try:
            model, preprocess, clip_mod = _load_clip()
        except Exception as e:
            logger.warning("CLIP model load failed during indexing: %s — timestamps only", e)
            model = preprocess = clip_mod = None
    else:
        model = preprocess = clip_mod = None

    with _index_lock:
        idx = _load_index()
        embeddings = idx.get("embeddings", {})
        total = len(media_paths)
        newly_indexed = 0

        for i, path in enumerate(media_paths):
            if on_progress:
                on_progress(int(i / max(total, 1) * 95), f"Indexing {os.path.basename(path)}")
            entry = embeddings.get(path, {})
            if "embedding" not in entry and model is not None:
                try:
                    png = _extract_frame_png(path)
                    if png:
                        from PIL import Image  # type: ignore
                        img = Image.open(io.BytesIO(png)).convert("RGB")
                        img_tensor = preprocess(img).unsqueeze(0)
                        with torch.no_grad():
                            feat = model.encode_image(img_tensor)
                            feat /= feat.norm(dim=-1, keepdim=True)
                        entry["embedding"] = feat.squeeze().tolist()
                except Exception as e:
                    logger.debug("Could not embed %s: %s", path, e)
            if path not in embeddings or "embedding" in entry:
                entry.setdefault("indexed_at", datetime.now(timezone.utc).isoformat())
                embeddings[path] = entry
                newly_indexed += 1

        idx["embeddings"] = embeddings
        idx["last_updated"] = datetime.now(timezone.utc).isoformat()
        _save_index(idx)

    if on_progress:
        on_progress(100, "Index built")
    return IndexStatus(
        indexed=len(embeddings),
        pending=max(0, total - newly_indexed),
        last_updated=idx["last_updated"],
    )


def get_index_status() -> IndexStatus:
    with _index_lock:
        idx = _load_index()
    embeddings = idx.get("embeddings", {})
    return IndexStatus(indexed=len(embeddings), pending=0,
                       last_updated=str(idx.get("last_updated", "")))


__all__ = ["check_semantic_search_available", "INSTALL_HINT", "SearchResult", "IndexStatus",
           "search", "build_index", "get_index_status"]
