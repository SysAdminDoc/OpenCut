"""
OpenCut Face Tagging & Recognition Module v0.9.0

Detect, cluster, and label faces across project media:
- Face detection via Haar cascades (always available)
- Face embedding via histogram/LBP features (no ML deps)
- Agglomerative clustering for grouping same-person faces
- Name tagging and face-based search

Optional: pip install face-recognition dlib (for higher accuracy)
Requires: pip install opencv-python-headless numpy scikit-learn
"""

import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import ensure_package, get_video_info

logger = logging.getLogger("opencut")

# Module-level tag storage (in production, use a database)
_face_tags: Dict[int, str] = {}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class FaceCluster:
    """A cluster of faces belonging to the same person."""
    cluster_id: int
    name: str = ""
    face_count: int = 0
    representative_frame: int = 0
    representative_bbox: List[int] = field(default_factory=list)
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Face Detection
# ---------------------------------------------------------------------------
def detect_faces(
    video_path: str,
    sample_rate: float = 1.0,
    min_face_size: int = 60,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Detect all faces in a video with frame positions.

    Samples frames at the given rate and runs face detection on each.

    Args:
        video_path: Input video path.
        sample_rate: Frames per second to sample (e.g. 1.0 = 1 fps).
        min_face_size: Minimum face size in pixels.
        on_progress: Progress callback(pct, msg).

    Returns:
        Dict with faces (list of detection dicts), total_faces,
        frames_sampled.
    """
    if not ensure_package("cv2", "opencv-python-headless"):
        raise RuntimeError("Failed to install opencv-python-headless")
    if not ensure_package("numpy", "numpy"):
        raise RuntimeError("Failed to install numpy")
    import cv2

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    info = get_video_info(video_path)
    fps = info.get("fps", 30.0)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_step = max(1, int(fps / max(0.1, sample_rate)))

    if on_progress:
        on_progress(5, "Detecting faces...")

    all_faces = []
    frames_sampled = 0
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(min_face_size, min_face_size),
                )

                for (fx, fy, fw, fh) in faces:
                    # Extract face region for embedding
                    face_roi = gray[fy:fy + fh, fx:fx + fw]
                    if face_roi.size == 0:
                        continue

                    # Compute LBP histogram as a simple face embedding
                    resized = cv2.resize(face_roi, (64, 64))
                    hist = cv2.calcHist([resized], [0], None, [64], [0, 256])
                    hist = cv2.normalize(hist, hist).flatten().tolist()

                    all_faces.append({
                        "frame_idx": frame_idx,
                        "timestamp": round(frame_idx / fps, 3),
                        "bbox": [int(fx), int(fy), int(fw), int(fh)],
                        "embedding": hist,
                    })

                frames_sampled += 1

            frame_idx += 1

            if on_progress and frame_idx % (sample_step * 10) == 0:
                pct = 5 + int((frame_idx / max(1, total_frames)) * 85)
                on_progress(min(pct, 92), f"Scanning frame {frame_idx}/{total_frames}...")
    finally:
        cap.release()

    if on_progress:
        on_progress(100, f"Detected {len(all_faces)} faces in {frames_sampled} frames!")

    return {
        "faces": all_faces,
        "total_faces": len(all_faces),
        "frames_sampled": frames_sampled,
    }


# ---------------------------------------------------------------------------
# Face Clustering
# ---------------------------------------------------------------------------
def cluster_faces(
    embeddings: List[dict],
    distance_threshold: float = 0.6,
    min_cluster_size: int = 2,
    on_progress: Optional[Callable] = None,
) -> List[FaceCluster]:
    """
    Cluster face detections by identity using embeddings.

    Uses agglomerative clustering to group faces by visual similarity.

    Args:
        embeddings: List of face detection dicts (from detect_faces),
            each with 'embedding', 'frame_idx', 'bbox' keys.
        distance_threshold: Clustering distance threshold (0-2).
            Lower = stricter matching.
        min_cluster_size: Minimum faces to form a cluster.
        on_progress: Progress callback(pct, msg).

    Returns:
        List of FaceCluster objects.
    """
    if not ensure_package("numpy", "numpy"):
        raise RuntimeError("Failed to install numpy")
    import numpy as np

    if not embeddings:
        return []

    if on_progress:
        on_progress(10, "Clustering faces...")

    # Extract embedding vectors
    vectors = []
    for face in embeddings:
        emb = face.get("embedding", [])
        if emb:
            vectors.append(emb)
        else:
            vectors.append([0.0] * 64)

    X = np.array(vectors, dtype=np.float32)

    # Try sklearn AgglomerativeClustering
    try:
        if not ensure_package("sklearn", "scikit-learn"):
            raise ImportError("scikit-learn not available")
        from sklearn.cluster import AgglomerativeClustering

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="euclidean",
            linkage="average",
        )
        labels = clustering.fit_predict(X)
    except (ImportError, Exception) as e:
        logger.info("sklearn not available, using simple distance clustering: %s", e)
        # Fallback: simple centroid-based clustering
        labels = _simple_cluster(X, distance_threshold)

    if on_progress:
        on_progress(70, "Building cluster summaries...")

    # Build clusters
    cluster_map: Dict[int, list] = {}
    for i, label in enumerate(labels):
        label = int(label)
        if label not in cluster_map:
            cluster_map[label] = []
        cluster_map[label].append(i)

    clusters = []
    for cid, indices in cluster_map.items():
        if len(indices) < min_cluster_size:
            continue

        # Pick representative face (middle occurrence)
        rep_idx = indices[len(indices) // 2]
        rep_face = embeddings[rep_idx]

        name = _face_tags.get(cid, "")

        clusters.append(FaceCluster(
            cluster_id=cid,
            name=name,
            face_count=len(indices),
            representative_frame=rep_face.get("frame_idx", 0),
            representative_bbox=rep_face.get("bbox", []),
            confidence=round(min(1.0, len(indices) / 10.0), 3),
        ))

    clusters.sort(key=lambda c: c.face_count, reverse=True)

    if on_progress:
        on_progress(100, f"Found {len(clusters)} face clusters!")

    return clusters


def _simple_cluster(X, threshold: float) -> list:
    """Simple greedy centroid-based clustering fallback."""
    import numpy as np

    n = len(X)
    labels = [-1] * n
    centroids = []
    cluster_id = 0

    for i in range(n):
        assigned = False
        for cid, centroid in enumerate(centroids):
            dist = float(np.linalg.norm(X[i] - centroid))
            if dist < threshold:
                labels[i] = cid
                # Update centroid (running mean)
                count = sum(1 for lbl in labels if lbl == cid)
                centroids[cid] = centroid + (X[i] - centroid) / count
                assigned = True
                break

        if not assigned:
            labels[i] = cluster_id
            centroids.append(X[i].copy())
            cluster_id += 1

    return labels


# ---------------------------------------------------------------------------
# Tag & Search
# ---------------------------------------------------------------------------
def tag_face_cluster(cluster_id: int, name: str) -> dict:
    """
    Assign a name/label to a face cluster.

    Args:
        cluster_id: Cluster ID from cluster_faces().
        name: Name to assign.

    Returns:
        Dict confirming the tag assignment.
    """
    if not isinstance(cluster_id, int) or cluster_id < 0:
        raise ValueError(f"Invalid cluster_id: {cluster_id}")
    if not name or not name.strip():
        raise ValueError("Name cannot be empty")

    _face_tags[cluster_id] = name.strip()

    return {
        "cluster_id": cluster_id,
        "name": name.strip(),
        "status": "tagged",
    }


def search_by_face(
    name: str,
    clusters: Optional[List[FaceCluster]] = None,
) -> List[dict]:
    """
    Search for a face by name across tagged clusters.

    Args:
        name: Name to search for (case-insensitive partial match).
        clusters: Optional pre-computed cluster list. If None, searches
            the module-level tag store.

    Returns:
        List of matching cluster dicts.
    """
    name_lower = name.lower().strip()
    if not name_lower:
        raise ValueError("Search name cannot be empty")

    results = []

    if clusters:
        for c in clusters:
            if name_lower in c.name.lower():
                results.append(asdict(c))
    else:
        # Search tag store
        for cid, tag_name in _face_tags.items():
            if name_lower in tag_name.lower():
                results.append({
                    "cluster_id": cid,
                    "name": tag_name,
                })

    return results
