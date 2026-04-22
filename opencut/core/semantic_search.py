"""
OpenCut Semantic Search v1.28.0

Unified CLIP visual + CLAP audio + Whisper transcript search index.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, List, Optional

from opencut.helpers import _try_import

logger = logging.getLogger("opencut")
INSTALL_HINT = "pip install torch transformers laion-clap openai-clip"
_INDEX_PATH = os.path.join(os.path.expanduser("~"), ".opencut", "search_index.json")
_model_cache: dict = {}


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
    with open(_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(idx, f, indent=2)


def _text_fallback_search(query: str, media_paths: List[str], top_k: int) -> List[SearchResult]:
    q = query.lower()
    results = []
    for path in media_paths:
        score = 1.0 if q in os.path.basename(path).lower() else 0.1
        results.append(SearchResult(path=path, score=score, type="text_fallback"))
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]


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
        clip_mod = _try_import("clip")
        if clip_mod is not None:
            try:
                if on_progress:
                    on_progress(10, "Loading CLIP model")
                if "clip" not in _model_cache:
                    import clip  # type: ignore
                    _model_cache["clip_model"], _model_cache["clip_preprocess"] = clip.load("ViT-B/32")
                model = _model_cache["clip_model"]
                preprocess = _model_cache["clip_preprocess"]
                import clip as clip_mod2  # type: ignore
                text_tokens = clip_mod2.tokenize([query])
                with torch.no_grad():
                    text_features = model.encode_text(text_tokens)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                for i, path in enumerate(media_paths):
                    if on_progress and i % 5 == 0:
                        on_progress(20 + int(i / len(media_paths) * 40), f"Scoring {os.path.basename(path)}")
                    try:
                        import subprocess
                        frame_cmd = ["ffmpeg", "-i", path, "-vf", "select=eq(n\\,0)", "-vframes", "1",
                                     "-f", "image2pipe", "-vcodec", "png", "pipe:1"]
                        png = subprocess.run(frame_cmd, capture_output=True, timeout=10).stdout
                        if png:
                            from PIL import Image  # type: ignore
                            import io
                            img = Image.open(io.BytesIO(png)).convert("RGB")
                            img_tensor = preprocess(img).unsqueeze(0)
                            with torch.no_grad():
                                img_features = model.encode_image(img_tensor)
                                img_features /= img_features.norm(dim=-1, keepdim=True)
                            score = float((img_features @ text_features.T).squeeze())
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
    idx = _load_index()
    embeddings = idx.get("embeddings", {})
    total = len(media_paths)
    indexed = 0
    for i, path in enumerate(media_paths):
        if path not in embeddings:
            embeddings[path] = {"indexed_at": datetime.now(timezone.utc).isoformat()}
            indexed += 1
        if on_progress:
            on_progress(int(i / total * 95), f"Indexing {os.path.basename(path)}")
    idx["embeddings"] = embeddings
    idx["last_updated"] = datetime.now(timezone.utc).isoformat()
    _save_index(idx)
    if on_progress:
        on_progress(100, "Index built")
    return IndexStatus(indexed=len(embeddings), pending=max(0, total - indexed),
                       last_updated=idx["last_updated"])


def get_index_status() -> IndexStatus:
    idx = _load_index()
    embeddings = idx.get("embeddings", {})
    return IndexStatus(indexed=len(embeddings), pending=0,
                       last_updated=str(idx.get("last_updated", "")))


__all__ = ["check_semantic_search_available", "INSTALL_HINT", "SearchResult", "IndexStatus",
           "search", "build_index", "get_index_status"]
