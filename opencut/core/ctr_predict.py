"""
OpenCut CTR Prediction Module

ML-driven thumbnail Click-Through Rate prediction and analysis:
- Analyze thumbnail images for CTR-relevant features
- Face presence, size, and expression scoring
- Text readability (contrast, size, position)
- Color vibrancy and saturation analysis
- Composition scoring (rule of thirds, visual hierarchy)
- Platform-specific weight profiles
- A/B thumbnail comparison with ranked suggestions

Uses Pillow for image analysis with optional deep learning backends.
"""

import colorsys
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from opencut.helpers import ensure_package

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Platform Weight Profiles
# ---------------------------------------------------------------------------
PLATFORM_WEIGHTS = {
    "youtube": {
        "face_presence": 0.22,
        "face_size": 0.08,
        "expression": 0.12,
        "text_readability": 0.14,
        "text_size": 0.06,
        "color_vibrancy": 0.10,
        "contrast": 0.08,
        "composition": 0.10,
        "clutter": 0.05,
        "brand_consistency": 0.05,
    },
    "tiktok": {
        "face_presence": 0.25,
        "face_size": 0.10,
        "expression": 0.15,
        "text_readability": 0.10,
        "text_size": 0.05,
        "color_vibrancy": 0.12,
        "contrast": 0.08,
        "composition": 0.07,
        "clutter": 0.04,
        "brand_consistency": 0.04,
    },
    "instagram": {
        "face_presence": 0.18,
        "face_size": 0.07,
        "expression": 0.10,
        "text_readability": 0.08,
        "text_size": 0.04,
        "color_vibrancy": 0.18,
        "contrast": 0.12,
        "composition": 0.15,
        "clutter": 0.05,
        "brand_consistency": 0.03,
    },
    "twitter": {
        "face_presence": 0.15,
        "face_size": 0.06,
        "expression": 0.08,
        "text_readability": 0.18,
        "text_size": 0.10,
        "color_vibrancy": 0.10,
        "contrast": 0.12,
        "composition": 0.10,
        "clutter": 0.06,
        "brand_consistency": 0.05,
    },
    "facebook": {
        "face_presence": 0.20,
        "face_size": 0.08,
        "expression": 0.12,
        "text_readability": 0.15,
        "text_size": 0.08,
        "color_vibrancy": 0.08,
        "contrast": 0.10,
        "composition": 0.10,
        "clutter": 0.05,
        "brand_consistency": 0.04,
    },
}


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------
@dataclass
class ThumbnailFeatures:
    """Extracted visual features from a thumbnail image."""
    image_path: str = ""
    width: int = 0
    height: int = 0
    face_count: int = 0
    face_area_ratio: float = 0.0
    face_positions: List[Tuple[int, int, int, int]] = field(default_factory=list)
    expression_score: float = 0.0
    avg_saturation: float = 0.0
    avg_brightness: float = 0.0
    color_vibrancy: float = 0.0
    dominant_colors: List[Tuple[int, int, int]] = field(default_factory=list)
    contrast_ratio: float = 0.0
    text_regions: int = 0
    text_area_ratio: float = 0.0
    text_contrast: float = 0.0
    thirds_score: float = 0.0
    visual_weight_center: Tuple[float, float] = (0.5, 0.5)
    edge_density: float = 0.0
    clutter_score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ImprovementSuggestion:
    """A single CTR improvement suggestion."""
    category: str = ""
    priority: str = "medium"
    message: str = ""
    current_score: float = 0.0
    potential_gain: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CTRPrediction:
    """Complete CTR prediction result."""
    image_path: str = ""
    platform: str = "youtube"
    ctr_score: float = 0.0
    confidence: float = 0.0
    feature_scores: Dict[str, float] = field(default_factory=dict)
    features: Optional[ThumbnailFeatures] = None
    suggestions: List[ImprovementSuggestion] = field(default_factory=list)
    grade: str = ""

    def to_dict(self) -> dict:
        d = {
            "image_path": self.image_path,
            "platform": self.platform,
            "ctr_score": round(self.ctr_score, 1),
            "confidence": round(self.confidence, 2),
            "feature_scores": {k: round(v, 1) for k, v in self.feature_scores.items()},
            "suggestions": [s.to_dict() for s in self.suggestions],
            "grade": self.grade,
        }
        if self.features:
            d["features"] = self.features.to_dict()
        return d


@dataclass
class ComparisonResult:
    """Result of comparing multiple thumbnails."""
    predictions: List[CTRPrediction] = field(default_factory=list)
    winner_index: int = 0
    winner_path: str = ""
    score_delta: float = 0.0

    def to_dict(self) -> dict:
        return {
            "predictions": [p.to_dict() for p in self.predictions],
            "winner_index": self.winner_index,
            "winner_path": self.winner_path,
            "score_delta": round(self.score_delta, 1),
        }


# ---------------------------------------------------------------------------
# Feature Analysis Functions
# ---------------------------------------------------------------------------
def _analyze_color_vibrancy(img: "Image.Image") -> Tuple[float, float, float, List[Tuple[int, int, int]]]:  # noqa: F821
    """Analyze color properties: saturation, brightness, vibrancy, dominants.

    Returns (avg_saturation, avg_brightness, vibrancy, dominant_colors).
    """
    img_small = img.resize((100, 100), resample=1)  # BILINEAR
    pixels = list(img_small.getdata())

    total_s = 0.0
    total_v = 0.0
    total_vibrancy = 0.0
    color_buckets: Dict[Tuple[int, int, int], int] = {}

    for r, g, b in pixels[:10000]:
        h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        total_s += s
        total_v += v
        total_vibrancy += s * v

        # Bucket colors (quantize to 32 levels)
        qr, qg, qb = (r // 32) * 32, (g // 32) * 32, (b // 32) * 32
        color_buckets[(qr, qg, qb)] = color_buckets.get((qr, qg, qb), 0) + 1

    n = len(pixels) or 1
    avg_s = total_s / n
    avg_v = total_v / n
    vibrancy = total_vibrancy / n

    # Top 5 dominant colors
    sorted_colors = sorted(color_buckets.items(), key=lambda x: -x[1])
    dominants = [c for c, _ in sorted_colors[:5]]

    return avg_s, avg_v, vibrancy, dominants


def _analyze_contrast(img: "Image.Image") -> float:  # noqa: F821
    """Compute luminance contrast ratio (max/min regions)."""
    gray = img.convert("L")
    small = gray.resize((50, 50), resample=1)
    pixels = list(small.getdata())
    if not pixels:
        return 1.0

    sorted_px = sorted(pixels)
    n = len(sorted_px)
    # Average of darkest 10% vs brightest 10%
    dark_region = sorted_px[:max(n // 10, 1)]
    bright_region = sorted_px[-max(n // 10, 1):]

    avg_dark = sum(dark_region) / len(dark_region) + 1  # avoid div by zero
    avg_bright = sum(bright_region) / len(bright_region) + 1

    ratio = avg_bright / avg_dark
    return min(ratio, 21.0)  # Cap at WCAG max


def _analyze_composition(img: "Image.Image") -> Tuple[float, Tuple[float, float], float]:  # noqa: F821
    """Analyze composition: rule of thirds score, visual weight center, edge density.

    Returns (thirds_score, weight_center, edge_density).
    """
    ensure_package("PIL", "Pillow")
    from PIL import ImageFilter

    gray = img.convert("L")
    w, h = gray.size

    # Edge detection for interest points
    edges = gray.filter(ImageFilter.FIND_EDGES)
    small_edges = edges.resize((50, 50), resample=1)
    edge_pixels = list(small_edges.getdata())
    edge_density = sum(1 for p in edge_pixels if p > 50) / max(len(edge_pixels), 1)

    # Visual weight center (weighted by edge intensity)
    total_weight = 0.0
    wx_sum = 0.0
    wy_sum = 0.0
    sw, sh = 50, 50
    for y in range(sh):
        for x in range(sw):
            val = edge_pixels[y * sw + x]
            total_weight += val
            wx_sum += val * (x / sw)
            wy_sum += val * (y / sh)

    if total_weight > 0:
        weight_cx = wx_sum / total_weight
        weight_cy = wy_sum / total_weight
    else:
        weight_cx, weight_cy = 0.5, 0.5

    # Rule of thirds: measure how close interest points are to third lines
    thirds_x = [1 / 3, 2 / 3]
    thirds_y = [1 / 3, 2 / 3]

    thirds_score = 0.0
    high_interest_count = 0
    for y in range(sh):
        for x in range(sw):
            val = edge_pixels[y * sw + x]
            if val > 80:
                high_interest_count += 1
                nx, ny = x / sw, y / sh
                min_dx = min(abs(nx - tx) for tx in thirds_x)
                min_dy = min(abs(ny - ty) for ty in thirds_y)
                proximity = 1.0 - min(min_dx, min_dy) / 0.5
                thirds_score += max(0.0, proximity) * val

    if high_interest_count > 0:
        thirds_score = thirds_score / (high_interest_count * 255)
    thirds_score = min(1.0, thirds_score)

    return thirds_score, (weight_cx, weight_cy), edge_density


def _detect_faces_heuristic(img: "Image.Image") -> Tuple[int, float, List[Tuple[int, int, int, int]]]:  # noqa: F821
    """Heuristic face detection using skin-tone pixel clustering.

    Returns (face_count, face_area_ratio, face_bboxes).
    This is a rough approximation; real face detection uses dlib/MediaPipe.
    """
    small = img.resize((100, 100), resample=1)
    pixels = small.load()
    w, h = 100, 100

    # Skin tone detection in RGB
    skin_map = [[False] * w for _ in range(h)]
    skin_pixel_count = 0

    for y in range(h):
        for x in range(w):
            r, g, b = pixels[x, y][:3]
            # Common skin tone heuristic
            is_skin = (
                r > 60 and g > 40 and b > 20 and
                r > g and r > b and
                abs(r - g) > 15 and
                max(r, g, b) - min(r, g, b) > 15
            )
            if is_skin:
                skin_map[y][x] = True
                skin_pixel_count += 1

    # Simple connected component to find face-sized clusters
    visited = [[False] * w for _ in range(h)]
    clusters = []

    def _flood(sx, sy):
        stack = [(sx, sy)]
        pts = []
        while stack:
            cx, cy = stack.pop()
            if cx < 0 or cy < 0 or cx >= w or cy >= h:
                continue
            if visited[cy][cx] or not skin_map[cy][cx]:
                continue
            visited[cy][cx] = True
            pts.append((cx, cy))
            if len(pts) > 2000:
                break
            stack.extend([(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)])
        return pts

    for y in range(h):
        for x in range(w):
            if skin_map[y][x] and not visited[y][x]:
                pts = _flood(x, y)
                if len(pts) >= 30:  # Minimum face cluster size
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    bbox = (min(xs), min(ys), max(xs), max(ys))
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    aspect = (bbox[2] - bbox[0]) / max(bbox[3] - bbox[1], 1)
                    # Faces are roughly square aspect ratio
                    if 0.5 < aspect < 2.0 and area > 100:
                        clusters.append(bbox)

    # Scale bboxes back to original image size
    ow, oh = img.size
    face_bboxes = []
    total_face_area = 0
    for bx1, by1, bx2, by2 in clusters[:5]:
        sx = ow / w
        sy = oh / h
        fb = (int(bx1 * sx), int(by1 * sy), int(bx2 * sx), int(by2 * sy))
        face_bboxes.append(fb)
        total_face_area += (fb[2] - fb[0]) * (fb[3] - fb[1])

    face_area_ratio = total_face_area / max(ow * oh, 1)
    return len(face_bboxes), face_area_ratio, face_bboxes


def _estimate_text_presence(img: "Image.Image") -> Tuple[int, float, float]:  # noqa: F821
    """Estimate text presence in image using high-contrast region detection.

    Returns (text_region_count, text_area_ratio, text_contrast).
    """
    ensure_package("PIL", "Pillow")
    from PIL import ImageFilter

    gray = img.convert("L")
    small = gray.resize((100, 100), resample=1)

    # High-contrast sharp edges suggest text
    edges = small.filter(ImageFilter.Kernel(
        size=(3, 3),
        kernel=[-1, -1, -1, -1, 8, -1, -1, -1, -1],
        scale=1, offset=128,
    ))

    edge_pixels = list(edges.getdata())
    strong_edges = sum(1 for p in edge_pixels if abs(p - 128) > 40)
    text_area_ratio = strong_edges / max(len(edge_pixels), 1)

    # Estimate contrast around text-like regions
    orig_pixels = list(small.getdata())
    high_contrast_vals = []
    for i, p in enumerate(edge_pixels):
        if abs(p - 128) > 40:
            high_contrast_vals.append(orig_pixels[i])

    if high_contrast_vals:
        text_contrast = max(high_contrast_vals) - min(high_contrast_vals)
    else:
        text_contrast = 0.0

    # Rough region count (every 10% of image with text-like activity)
    w, h = 100, 100
    grid_size = 10
    regions_with_text = 0
    for gy in range(0, h, grid_size):
        for gx in range(0, w, grid_size):
            count = 0
            for dy in range(grid_size):
                for dx in range(grid_size):
                    idx = (gy + dy) * w + (gx + dx)
                    if idx < len(edge_pixels) and abs(edge_pixels[idx] - 128) > 40:
                        count += 1
            if count > grid_size:  # More than 10% of cell is text-like
                regions_with_text += 1

    return regions_with_text, text_area_ratio, text_contrast / 255.0


def _score_expression(face_area_ratio: float, brightness: float) -> float:
    """Heuristic expression score based on face area and brightness.

    Real expression scoring uses facial landmark analysis. This approximates
    by assuming larger, well-lit faces have more visible expressions.
    """
    size_factor = min(face_area_ratio / 0.15, 1.0)
    brightness_factor = 0.5 + 0.5 * min(brightness / 0.7, 1.0)
    return min(size_factor * brightness_factor, 1.0)


def _analyze_clutter(edge_density: float, color_count: int) -> float:
    """Estimate visual clutter from edge density and color diversity."""
    # High edge density + many colors = cluttered
    edge_factor = min(edge_density / 0.6, 1.0)
    color_factor = min(color_count / 200, 1.0)
    return min((edge_factor * 0.6 + color_factor * 0.4), 1.0)


# ---------------------------------------------------------------------------
# Feature Extraction Pipeline
# ---------------------------------------------------------------------------
def extract_features(image_path: str) -> ThumbnailFeatures:
    """Extract all CTR-relevant features from a thumbnail image.

    Args:
        image_path: Path to thumbnail image.

    Returns:
        ThumbnailFeatures with all analyzed metrics.

    Raises:
        FileNotFoundError: If image not found.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    ensure_package("PIL", "Pillow")
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    # Color analysis
    avg_sat, avg_bright, vibrancy, dominants = _analyze_color_vibrancy(img)

    # Contrast
    contrast = _analyze_contrast(img)

    # Composition
    thirds, weight_center, edge_density = _analyze_composition(img)

    # Face detection
    face_count, face_area, face_bboxes = _detect_faces_heuristic(img)

    # Text detection
    text_regions, text_area, text_contrast = _estimate_text_presence(img)

    # Expression
    expression = _score_expression(face_area, avg_bright)

    # Clutter
    color_bucket_count = len(set(
        ((r // 64) * 64, (g // 64) * 64, (b // 64) * 64)
        for r, g, b in img.resize((50, 50), resample=1).getdata()
    ))
    clutter = _analyze_clutter(edge_density, color_bucket_count)

    return ThumbnailFeatures(
        image_path=image_path,
        width=w,
        height=h,
        face_count=face_count,
        face_area_ratio=face_area,
        face_positions=face_bboxes,
        expression_score=expression,
        avg_saturation=avg_sat,
        avg_brightness=avg_bright,
        color_vibrancy=vibrancy,
        dominant_colors=dominants,
        contrast_ratio=contrast,
        text_regions=text_regions,
        text_area_ratio=text_area,
        text_contrast=text_contrast,
        thirds_score=thirds,
        visual_weight_center=weight_center,
        edge_density=edge_density,
        clutter_score=clutter,
    )


# ---------------------------------------------------------------------------
# Score Computation
# ---------------------------------------------------------------------------
def _compute_feature_scores(features: ThumbnailFeatures) -> Dict[str, float]:
    """Convert raw features into 0-100 scores per category."""
    scores = {}

    # Face presence: 0 faces=20, 1 face=90, 2+=75 (too many faces is worse)
    if features.face_count == 0:
        scores["face_presence"] = 20.0
    elif features.face_count == 1:
        scores["face_presence"] = 90.0
    elif features.face_count == 2:
        scores["face_presence"] = 75.0
    else:
        scores["face_presence"] = max(40.0, 75.0 - (features.face_count - 2) * 10)

    # Face size: ideal is 10-25% of frame
    if features.face_area_ratio < 0.03:
        scores["face_size"] = 30.0
    elif features.face_area_ratio < 0.10:
        scores["face_size"] = 50.0 + (features.face_area_ratio / 0.10) * 30
    elif features.face_area_ratio < 0.25:
        scores["face_size"] = 90.0
    elif features.face_area_ratio < 0.50:
        scores["face_size"] = 80.0 - (features.face_area_ratio - 0.25) * 80
    else:
        scores["face_size"] = 50.0

    # Expression
    scores["expression"] = min(features.expression_score * 100, 100.0)

    # Text readability
    if features.text_regions == 0:
        scores["text_readability"] = 40.0  # No text = not great for CTR
    else:
        scores["text_readability"] = min(40 + features.text_contrast * 60, 100.0)

    # Text size
    if features.text_area_ratio < 0.02:
        scores["text_size"] = 30.0
    elif features.text_area_ratio < 0.15:
        scores["text_size"] = 80.0
    elif features.text_area_ratio < 0.30:
        scores["text_size"] = 70.0
    else:
        scores["text_size"] = 40.0  # Too much text

    # Color vibrancy
    scores["color_vibrancy"] = min(features.color_vibrancy * 200, 100.0)

    # Contrast
    if features.contrast_ratio < 2.0:
        scores["contrast"] = 30.0
    elif features.contrast_ratio < 4.5:
        scores["contrast"] = 50.0 + (features.contrast_ratio / 4.5) * 30
    elif features.contrast_ratio < 10.0:
        scores["contrast"] = 90.0
    else:
        scores["contrast"] = 85.0

    # Composition (rule of thirds)
    scores["composition"] = min(features.thirds_score * 100, 100.0)

    # Clutter (lower is better)
    scores["clutter"] = max(0.0, 100.0 - features.clutter_score * 100)

    # Brand consistency (placeholder — would need reference data)
    scores["brand_consistency"] = 60.0

    return scores


def _generate_suggestions(feature_scores: Dict[str, float],
                           features: ThumbnailFeatures) -> List[ImprovementSuggestion]:
    """Generate actionable improvement suggestions based on scores."""
    suggestions = []

    if feature_scores.get("face_presence", 0) < 50:
        suggestions.append(ImprovementSuggestion(
            category="face_presence",
            priority="high",
            message="Add a face to the thumbnail. Thumbnails with faces get 30-40% higher CTR.",
            current_score=feature_scores.get("face_presence", 0),
            potential_gain=25.0,
        ))

    if feature_scores.get("face_size", 0) < 60:
        if features.face_count > 0 and features.face_area_ratio < 0.10:
            suggestions.append(ImprovementSuggestion(
                category="face_size",
                priority="medium",
                message="Make the face larger — it should fill 10-25% of the frame.",
                current_score=feature_scores.get("face_size", 0),
                potential_gain=15.0,
            ))

    if feature_scores.get("expression", 0) < 50:
        suggestions.append(ImprovementSuggestion(
            category="expression",
            priority="medium",
            message="Use a more expressive face (surprise, excitement, concern). Exaggerated expressions boost CTR.",
            current_score=feature_scores.get("expression", 0),
            potential_gain=12.0,
        ))

    if feature_scores.get("text_readability", 0) < 50:
        if features.text_regions == 0:
            suggestions.append(ImprovementSuggestion(
                category="text_readability",
                priority="high",
                message="Add bold text overlay with a clear value proposition (3-5 words max).",
                current_score=feature_scores.get("text_readability", 0),
                potential_gain=20.0,
            ))
        else:
            suggestions.append(ImprovementSuggestion(
                category="text_readability",
                priority="medium",
                message="Increase text contrast — use white text with dark outline or colored background.",
                current_score=feature_scores.get("text_readability", 0),
                potential_gain=10.0,
            ))

    if feature_scores.get("color_vibrancy", 0) < 50:
        suggestions.append(ImprovementSuggestion(
            category="color_vibrancy",
            priority="medium",
            message="Increase color saturation. Vibrant colors stand out in feeds and search results.",
            current_score=feature_scores.get("color_vibrancy", 0),
            potential_gain=10.0,
        ))

    if feature_scores.get("contrast", 0) < 60:
        suggestions.append(ImprovementSuggestion(
            category="contrast",
            priority="medium",
            message="Improve contrast between foreground and background for better visibility at small sizes.",
            current_score=feature_scores.get("contrast", 0),
            potential_gain=8.0,
        ))

    if feature_scores.get("composition", 0) < 40:
        suggestions.append(ImprovementSuggestion(
            category="composition",
            priority="low",
            message="Place key elements along rule-of-thirds intersections for stronger composition.",
            current_score=feature_scores.get("composition", 0),
            potential_gain=6.0,
        ))

    if feature_scores.get("clutter", 0) < 50:
        suggestions.append(ImprovementSuggestion(
            category="clutter",
            priority="medium",
            message="Simplify the thumbnail — remove background distractions. Clean thumbnails perform better.",
            current_score=feature_scores.get("clutter", 0),
            potential_gain=8.0,
        ))

    # Sort by potential gain descending
    suggestions.sort(key=lambda s: -s.potential_gain)
    return suggestions


def _score_to_grade(score: float) -> str:
    """Convert a 0-100 score to a letter grade."""
    if score >= 90:
        return "A+"
    elif score >= 80:
        return "A"
    elif score >= 70:
        return "B+"
    elif score >= 60:
        return "B"
    elif score >= 50:
        return "C+"
    elif score >= 40:
        return "C"
    elif score >= 30:
        return "D"
    else:
        return "F"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def predict_ctr(
    image_path: str,
    platform: str = "youtube",
    on_progress: Optional[Callable] = None,
) -> CTRPrediction:
    """Predict thumbnail CTR score with feature analysis and suggestions.

    Args:
        image_path: Path to thumbnail image.
        platform: Target platform for weight profile.
        on_progress: Callback ``(pct, msg)`` for progress updates.

    Returns:
        CTRPrediction with score 0-100, feature breakdown, and suggestions.

    Raises:
        FileNotFoundError: If image not found.
        ValueError: If platform is unknown.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    platform = platform.lower().strip()
    if platform not in PLATFORM_WEIGHTS:
        raise ValueError(
            f"Unknown platform '{platform}'. Supported: {list(PLATFORM_WEIGHTS.keys())}")

    weights = PLATFORM_WEIGHTS[platform]

    if on_progress:
        on_progress(10, "Extracting thumbnail features...")

    features = extract_features(image_path)

    if on_progress:
        on_progress(50, "Computing feature scores...")

    feature_scores = _compute_feature_scores(features)

    # Weighted final score
    total_score = 0.0
    total_weight = 0.0
    for category, weight in weights.items():
        score = feature_scores.get(category, 50.0)
        total_score += score * weight
        total_weight += weight

    ctr_score = total_score / max(total_weight, 0.01)
    ctr_score = max(0.0, min(100.0, ctr_score))

    # Confidence: higher when features are more decisive
    score_variance = sum(
        (feature_scores.get(k, 50) - ctr_score) ** 2 for k in weights
    ) / max(len(weights), 1)
    confidence = max(0.4, min(0.95, 0.7 + (score_variance / 10000)))

    if on_progress:
        on_progress(80, "Generating improvement suggestions...")

    suggestions = _generate_suggestions(feature_scores, features)
    grade = _score_to_grade(ctr_score)

    if on_progress:
        on_progress(100, "CTR prediction complete")

    return CTRPrediction(
        image_path=image_path,
        platform=platform,
        ctr_score=ctr_score,
        confidence=confidence,
        feature_scores=feature_scores,
        features=features,
        suggestions=suggestions,
        grade=grade,
    )


def compare_thumbnails(
    image_paths: List[str],
    platform: str = "youtube",
    on_progress: Optional[Callable] = None,
) -> ComparisonResult:
    """Compare multiple thumbnails and rank by predicted CTR.

    Args:
        image_paths: List of thumbnail image paths (2-10).
        platform: Target platform for weight profile.
        on_progress: Callback ``(pct, msg)`` for progress updates.

    Returns:
        ComparisonResult with ranked predictions and winner.

    Raises:
        ValueError: If fewer than 2 images or more than 10.
    """
    if len(image_paths) < 2:
        raise ValueError("Need at least 2 thumbnails to compare")
    if len(image_paths) > 10:
        raise ValueError("Maximum 10 thumbnails per comparison")

    predictions = []
    for i, path in enumerate(image_paths):
        def _sub_progress(pct, msg=""):
            if on_progress:
                base = int(100 * i / len(image_paths))
                chunk = int(100 / len(image_paths))
                on_progress(base + int(pct * chunk / 100), msg)

        pred = predict_ctr(path, platform=platform, on_progress=_sub_progress)
        predictions.append(pred)

    # Rank by CTR score descending
    predictions.sort(key=lambda p: -p.ctr_score)

    winner_idx = 0
    winner = predictions[0]
    runner_up = predictions[1] if len(predictions) > 1 else predictions[0]
    delta = winner.ctr_score - runner_up.ctr_score

    # Find original index of winner
    for i, path in enumerate(image_paths):
        if path == winner.image_path:
            winner_idx = i
            break

    return ComparisonResult(
        predictions=predictions,
        winner_index=winner_idx,
        winner_path=winner.image_path,
        score_delta=delta,
    )
