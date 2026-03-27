"""
Multimodal Speaker Diarization

Combines audio speaker embeddings (3D-Speaker / pyannote) with face recognition
(InsightFace / facenet) for robust speaker-to-face mapping in multicam content.

Flow:
1. Audio diarization → speaker segments (who speaks when)
2. Face detection + tracking → face segments (who appears when)
3. Cross-modal alignment → speaker↔face mapping (co-occurrence scoring)
4. Output: enriched diarization with face IDs for accurate multicam switching
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("opencut")


@dataclass
class FaceSegment:
    """A segment where a face is visible."""
    face_id: int          # Cluster ID from face recognition
    start: float          # Start time in seconds
    end: float            # End time in seconds
    confidence: float     # Detection confidence 0-1
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h of last seen position

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class SpeakerFaceMapping:
    """Mapping between an audio speaker and a detected face."""
    speaker: str         # e.g., "SPEAKER_00"
    face_id: int         # Face cluster ID
    confidence: float    # Mapping confidence 0-1
    overlap_seconds: float  # Total co-occurrence time


@dataclass
class MultimodalDiarizationResult:
    """Combined audio + visual diarization output."""
    speaker_segments: List[dict]       # Audio diarization segments
    face_segments: List[FaceSegment]   # Visual face segments
    mappings: List[SpeakerFaceMapping]  # Speaker↔face associations
    num_speakers: int = 0
    num_faces: int = 0
    speakers: List[str] = field(default_factory=list)

    def get_speaker_face(self, speaker: str) -> Optional[int]:
        """Get the face_id most likely belonging to a speaker."""
        best = None
        best_conf = 0.0
        for m in self.mappings:
            if m.speaker == speaker and m.confidence > best_conf:
                best = m.face_id
                best_conf = m.confidence
        return best

    def to_enriched_cuts(
        self,
        speaker_camera_map: Optional[Dict[str, int]] = None,
        min_segment_duration: float = 1.0,
    ) -> List[dict]:
        """Generate multicam cuts enriched with face position data."""
        if not self.speaker_segments:
            return []
        if speaker_camera_map is None:
            speaker_camera_map = {s: i for i, s in enumerate(self.speakers)}

        cuts = []
        for seg in self.speaker_segments:
            spk = seg.get("speaker", "")
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
            dur = end - start
            if dur < min_segment_duration:
                continue
            face_id = self.get_speaker_face(spk)
            cuts.append({
                "time": round(start, 4),
                "duration": round(dur, 4),
                "track": speaker_camera_map.get(spk, 0),
                "speaker": spk,
                "face_id": face_id,
            })
        return cuts


def check_multimodal_diarize_available() -> bool:
    """Check if multimodal diarization dependencies are available."""
    try:
        import insightface  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        from facenet_pytorch import MTCNN  # noqa: F401
        return True
    except ImportError:
        return False


def _extract_face_segments(
    video_path: str,
    sample_fps: float = 2.0,
    min_confidence: float = 0.5,
    on_progress: Optional[Callable] = None,
) -> List[FaceSegment]:
    """
    Extract face appearances from video frames.

    Uses InsightFace (preferred) or facenet-pytorch MTCNN for detection,
    then clusters faces by embedding similarity.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0
        sample_interval = max(1, int(fps / sample_fps))

        # Try InsightFace first, fall back to facenet MTCNN
        detector = None
        embedder = None
        backend = "none"

        try:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            app.prepare(ctx_id=0, det_size=(640, 640))
            detector = app
            backend = "insightface"
            logger.debug("Using InsightFace for face detection + embedding")
        except Exception:
            try:
                import torch
                from facenet_pytorch import MTCNN, InceptionResnetV1
                device = "cuda" if torch.cuda.is_available() else "cpu"
                mtcnn = MTCNN(keep_all=True, device=device, min_face_size=40)
                resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
                detector = mtcnn
                embedder = resnet
                backend = "facenet"
                logger.debug("Using facenet-pytorch for face detection + embedding")
            except Exception:
                logger.warning("No face detection backend available, using OpenCV Haar cascade fallback")
                backend = "haar"

        # Collect per-frame face detections with embeddings
        frame_faces = []  # List of (timestamp, [(bbox, embedding), ...])
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_interval == 0:
                timestamp = frame_idx / fps
                faces_in_frame = []

                if backend == "insightface":
                    results = detector.get(frame)
                    for face in results:
                        if face.det_score < min_confidence:
                            continue
                        bbox = tuple(int(v) for v in face.bbox[:4])
                        emb = face.embedding if hasattr(face, "embedding") else None
                        faces_in_frame.append((bbox, emb, float(face.det_score)))

                elif backend == "facenet":
                    import torch
                    boxes, probs = detector.detect(frame)
                    if boxes is not None:
                        # Get embeddings
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        from PIL import Image
                        pil_img = Image.fromarray(rgb)
                        faces_cropped = detector.extract(pil_img, boxes, save_path=None)
                        for i, (box, prob) in enumerate(zip(boxes, probs)):
                            if prob is None or prob < min_confidence:
                                continue
                            bbox = tuple(int(v) for v in box[:4])
                            emb = None
                            if faces_cropped is not None and i < len(faces_cropped) and faces_cropped[i] is not None:
                                with torch.no_grad():
                                    emb = embedder(faces_cropped[i].unsqueeze(0).to(
                                        "cuda" if torch.cuda.is_available() else "cpu"
                                    )).cpu().numpy().flatten()
                            faces_in_frame.append((bbox, emb, float(prob)))

                else:
                    # Haar cascade fallback — no embeddings, just detection
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    cascade = cv2.CascadeClassifier(
                        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                    )
                    rects = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
                    for (x, y, w, h) in rects:
                        faces_in_frame.append(((x, y, x + w, y + h), None, 0.7))

                if faces_in_frame:
                    frame_faces.append((timestamp, faces_in_frame))

                if on_progress and total_frames > 0:
                    pct = int(frame_idx / total_frames * 60)
                    on_progress(pct, f"Detecting faces... {timestamp:.1f}s / {duration:.1f}s")

            frame_idx += 1

        # Cluster faces by embedding similarity
        face_segments = _cluster_faces(frame_faces, sample_interval / fps, min_confidence)

        if on_progress:
            on_progress(65, f"Found {len(set(f.face_id for f in face_segments))} unique faces")

        return face_segments

    finally:
        cap.release()


def _cluster_faces(
    frame_faces: list,
    sample_period: float,
    min_confidence: float,
) -> List[FaceSegment]:
    """
    Cluster face detections across frames into continuous face segments.

    Uses embedding cosine similarity when available, falls back to
    spatial proximity (IoU) for Haar cascade detections.
    """
    import numpy as np

    if not frame_faces:
        return []

    # Collect all detections with timestamps
    all_detections = []  # (timestamp, bbox, embedding, confidence)
    for ts, faces in frame_faces:
        for bbox, emb, conf in faces:
            all_detections.append((ts, bbox, emb, conf))

    if not all_detections:
        return []

    # Check if we have embeddings
    has_embeddings = any(d[2] is not None for d in all_detections)

    if has_embeddings:
        # Cosine similarity clustering
        embeddings = []
        valid_indices = []
        for i, (ts, bbox, emb, conf) in enumerate(all_detections):
            if emb is not None:
                embeddings.append(emb)
                valid_indices.append(i)

        if embeddings:
            emb_matrix = np.array(embeddings)
            # Normalize
            norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            emb_matrix = emb_matrix / norms

            # Simple agglomerative clustering via similarity threshold
            labels = np.full(len(all_detections), -1, dtype=int)
            cluster_centers = []
            threshold = 0.55  # Cosine similarity threshold

            for idx, emb_idx in zip(valid_indices, range(len(embeddings))):
                emb = emb_matrix[emb_idx]
                best_cluster = -1
                best_sim = threshold

                for c_id, center in enumerate(cluster_centers):
                    sim = float(np.dot(emb, center))
                    if sim > best_sim:
                        best_sim = sim
                        best_cluster = c_id

                if best_cluster >= 0:
                    labels[idx] = best_cluster
                    # Update center with running average
                    cluster_centers[best_cluster] = (
                        cluster_centers[best_cluster] * 0.9 + emb * 0.1
                    )
                    norm = np.linalg.norm(cluster_centers[best_cluster])
                    if norm > 0:
                        cluster_centers[best_cluster] /= norm
                else:
                    new_id = len(cluster_centers)
                    labels[idx] = new_id
                    cluster_centers.append(emb.copy())

            # Assign unlabeled detections to nearest cluster by bbox proximity
            for i in range(len(all_detections)):
                if labels[i] < 0:
                    labels[i] = 0  # Default cluster
    else:
        # Spatial-only clustering (Haar cascade fallback)
        labels = np.zeros(len(all_detections), dtype=int)
        # Simple: assign by x-position (left face = 0, right face = 1, etc.)
        if all_detections:
            xs = [d[1][0] for d in all_detections]
            if len(set(xs)) > 1:
                median_x = np.median(xs)
                labels = np.array([0 if d[1][0] < median_x else 1 for d in all_detections])

    # Convert labeled detections to FaceSegments
    segments = []
    unique_labels = set(labels)

    for label in sorted(unique_labels):
        if label < 0:
            continue
        cluster_dets = [(all_detections[i], i) for i in range(len(labels)) if labels[i] == label]
        if not cluster_dets:
            continue

        # Sort by timestamp
        cluster_dets.sort(key=lambda x: x[0][0])

        # Merge consecutive detections into segments
        seg_start = cluster_dets[0][0][0]
        seg_end = seg_start + sample_period
        last_bbox = cluster_dets[0][0][1]
        avg_conf = cluster_dets[0][0][3]
        count = 1

        for det, _ in cluster_dets[1:]:
            ts = det[0]
            if ts - seg_end <= sample_period * 2.5:
                # Continue segment
                seg_end = ts + sample_period
                last_bbox = det[1]
                avg_conf += det[3]
                count += 1
            else:
                # Finish segment, start new one
                segments.append(FaceSegment(
                    face_id=int(label),
                    start=round(seg_start, 4),
                    end=round(seg_end, 4),
                    confidence=round(avg_conf / count, 4),
                    bbox=last_bbox,
                ))
                seg_start = ts
                seg_end = ts + sample_period
                last_bbox = det[1]
                avg_conf = det[3]
                count = 1

        # Final segment
        segments.append(FaceSegment(
            face_id=int(label),
            start=round(seg_start, 4),
            end=round(seg_end, 4),
            confidence=round(avg_conf / count, 4),
            bbox=last_bbox,
        ))

    return segments


def _align_speakers_to_faces(
    speaker_segments: List[dict],
    face_segments: List[FaceSegment],
) -> List[SpeakerFaceMapping]:
    """
    Align audio speakers to visual faces using temporal co-occurrence.

    For each (speaker, face_id) pair, compute total overlap time.
    The highest-overlap pair becomes the mapping.
    """
    if not speaker_segments or not face_segments:
        return []

    # Collect unique speakers and face IDs
    speakers = sorted(set(s.get("speaker", "") for s in speaker_segments))
    face_ids = sorted(set(f.face_id for f in face_segments))

    mappings = []
    for speaker in speakers:
        spk_segs = [s for s in speaker_segments if s.get("speaker") == speaker]
        best_face = -1
        best_overlap = 0.0

        for fid in face_ids:
            f_segs = [f for f in face_segments if f.face_id == fid]
            total_overlap = 0.0

            for ss in spk_segs:
                ss_start = ss.get("start", 0.0)
                ss_end = ss.get("end", 0.0)
                for fs in f_segs:
                    overlap_start = max(ss_start, fs.start)
                    overlap_end = min(ss_end, fs.end)
                    if overlap_end > overlap_start:
                        total_overlap += overlap_end - overlap_start

            if total_overlap > best_overlap:
                best_overlap = total_overlap
                best_face = fid

        if best_face >= 0 and best_overlap > 0:
            # Confidence = overlap / total speaker duration
            total_spk_dur = sum(s.get("end", 0) - s.get("start", 0) for s in spk_segs)
            confidence = best_overlap / total_spk_dur if total_spk_dur > 0 else 0.0
            mappings.append(SpeakerFaceMapping(
                speaker=speaker,
                face_id=best_face,
                confidence=round(min(confidence, 1.0), 4),
                overlap_seconds=round(best_overlap, 4),
            ))

    return mappings


def multimodal_diarize(
    filepath: str,
    num_speakers: Optional[int] = None,
    sample_fps: float = 2.0,
    min_face_confidence: float = 0.5,
    on_progress: Optional[Callable] = None,
) -> MultimodalDiarizationResult:
    """
    Perform multimodal speaker diarization combining audio + face recognition.

    Args:
        filepath: Path to video file.
        num_speakers: Expected number of speakers (None = auto-detect).
        sample_fps: Frames per second to sample for face detection.
        min_face_confidence: Minimum face detection confidence.
        on_progress: Callback(percent, message).

    Returns:
        MultimodalDiarizationResult with speaker segments, face segments,
        and speaker↔face mappings.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    if on_progress:
        on_progress(0, "Starting multimodal diarization...")

    # Step 1: Audio diarization
    if on_progress:
        on_progress(5, "Running audio diarization...")

    speaker_segments = []
    speakers = []
    num_spk = 0

    try:
        from opencut.core.diarize import check_pyannote_available, diarize
        if check_pyannote_available():
            from opencut.utils.config import DiarizeConfig
            config = DiarizeConfig()
            if num_speakers:
                config.num_speakers = num_speakers
            result = diarize(filepath, config)
            speaker_segments = [
                {"speaker": s.speaker, "start": s.start, "end": s.end}
                for s in result.segments
            ]
            speakers = result.speakers
            num_spk = result.num_speakers
            logger.info("Audio diarization: %d segments, %d speakers", len(speaker_segments), num_spk)
        else:
            logger.warning("pyannote not available, using whisper-based speaker segments")
            # Fallback: use whisper timestamps as single-speaker segments
            try:
                from opencut.core.captions import check_whisper_available, transcribe
                available, backend = check_whisper_available()
                if available:
                    result = transcribe(filepath, backend=backend)
                    segments = result.get("segments", [])
                    speaker_segments = [
                        {"speaker": "SPEAKER_00", "start": s.get("start", 0), "end": s.get("end", 0)}
                        for s in segments
                    ]
                    speakers = ["SPEAKER_00"]
                    num_spk = 1
            except Exception as e:
                logger.warning("Whisper fallback failed: %s", e)

    except Exception as e:
        logger.warning("Audio diarization failed: %s", e)

    if on_progress:
        on_progress(30, f"Audio: {num_spk} speakers, {len(speaker_segments)} segments")

    # Step 2: Face detection + clustering
    if on_progress:
        on_progress(35, "Detecting and clustering faces...")

    def _face_progress(pct, msg=""):
        if on_progress:
            # Map 0-65 from face extraction to 35-75 overall
            mapped = 35 + int(pct * 0.6)
            on_progress(mapped, msg)

    try:
        face_segments = _extract_face_segments(
            filepath,
            sample_fps=sample_fps,
            min_confidence=min_face_confidence,
            on_progress=_face_progress,
        )
    except Exception as e:
        logger.warning("Face extraction failed, returning audio-only results: %s", e)
        face_segments = []

    num_faces = len(set(f.face_id for f in face_segments))

    if on_progress:
        on_progress(80, f"Faces: {num_faces} unique, {len(face_segments)} segments")

    # Step 3: Cross-modal alignment
    if on_progress:
        on_progress(85, "Aligning speakers to faces...")

    mappings = _align_speakers_to_faces(speaker_segments, face_segments)

    for m in mappings:
        logger.info(
            "Speaker %s → Face %d (confidence=%.2f, overlap=%.1fs)",
            m.speaker, m.face_id, m.confidence, m.overlap_seconds,
        )

    if on_progress:
        on_progress(100, "Multimodal diarization complete!")

    return MultimodalDiarizationResult(
        speaker_segments=speaker_segments,
        face_segments=face_segments,
        mappings=mappings,
        num_speakers=num_spk,
        num_faces=num_faces,
        speakers=speakers,
    )
