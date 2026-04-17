"""
OpenCut Command Palette - Feature Index & Fuzzy Search

Build and search a comprehensive index of all OpenCut features.
Powers a Cmd+K search interface with trigram/substring matching,
recent feature tracking, and frequency-based ranking.
"""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from opencut.helpers import OPENCUT_DIR, _ensure_opencut_dir

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Feature Entry
# ---------------------------------------------------------------------------
FEATURE_CATEGORIES = [
    "audio", "video", "captions", "effects", "export", "ai",
    "color", "editing", "analysis", "social", "restoration",
    "motion", "3d_vfx", "broadcast", "collaboration", "platform",
]


@dataclass
class FeatureEntry:
    """A single searchable feature."""
    id: str
    name: str
    description: str
    category: str
    aliases: List[str] = field(default_factory=list)
    route: str = ""
    tags: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Singleton Index
# ---------------------------------------------------------------------------
_feature_index: Optional[List[dict]] = None
_index_lock = threading.Lock()

_RECENTS_FILE = os.path.join(OPENCUT_DIR, "feature_recents.json")
_MAX_RECENTS = 50


def _trigrams(s: str) -> set:
    """Generate character trigrams from a string."""
    s = s.lower().strip()
    if len(s) < 3:
        return {s} if s else set()
    return {s[i:i + 3] for i in range(len(s) - 2)}


def _similarity_score(query: str, text: str) -> float:
    """Score similarity between query and text using multiple heuristics.

    Returns a float 0.0-1.0 where higher is better.
    Ranking: exact match > prefix > word-start > substring > trigram overlap.
    """
    q = query.lower().strip()
    t = text.lower().strip()

    if not q or not t:
        return 0.0

    # Exact match
    if q == t:
        return 1.0

    # Prefix match
    if t.startswith(q):
        return 0.9 + (0.05 * len(q) / max(len(t), 1))

    # Word-start match (query matches start of any word)
    words = t.split()
    for word in words:
        if word.startswith(q):
            return 0.8 + (0.05 * len(q) / max(len(word), 1))

    # Substring match
    if q in t:
        return 0.7 + (0.05 * len(q) / max(len(t), 1))

    # Trigram similarity
    q_tri = _trigrams(q)
    t_tri = _trigrams(t)
    if q_tri and t_tri:
        overlap = len(q_tri & t_tri)
        total = len(q_tri | t_tri)
        if total > 0:
            return 0.5 * (overlap / total)

    return 0.0


def _score_entry(query: str, entry: dict) -> float:
    """Score a feature entry against a query, checking all searchable fields."""
    best = 0.0
    best = max(best, _similarity_score(query, entry["name"]))
    best = max(best, _similarity_score(query, entry["description"]) * 0.8)
    best = max(best, _similarity_score(query, entry["category"]) * 0.7)
    for alias in entry.get("aliases", []):
        best = max(best, _similarity_score(query, alias) * 0.95)
    for tag in entry.get("tags", []):
        best = max(best, _similarity_score(query, tag) * 0.85)
    return best


# ---------------------------------------------------------------------------
# Feature Registry (all 302+ features)
# ---------------------------------------------------------------------------
_FEATURE_DEFS: List[dict] = [
    # --- Audio ---
    {"id": "normalize_audio", "name": "Normalize Audio", "description": "Normalize audio loudness to broadcast standards", "category": "audio", "aliases": ["loudness", "audio level"], "route": "/audio/normalize", "tags": ["loudness", "lufs", "level"]},
    {"id": "remove_silence", "name": "Remove Silence", "description": "Detect and cut silent segments from audio", "category": "audio", "aliases": ["silence", "dead air", "trim silence"], "route": "/silence", "tags": ["silence", "trim", "pacing"]},
    {"id": "audio_enhance", "name": "Enhance Audio", "description": "AI-powered audio cleanup and enhancement", "category": "audio", "aliases": ["clean audio", "noise reduction", "denoise audio"], "route": "/audio/enhance", "tags": ["enhance", "noise", "cleanup"]},
    {"id": "audio_duck", "name": "Audio Ducking", "description": "Auto-duck music under speech", "category": "audio", "aliases": ["duck", "lower music", "sidechain"], "route": "/audio/duck", "tags": ["duck", "music", "speech"]},
    {"id": "stem_split", "name": "Stem Separation", "description": "Separate vocals, drums, bass, and instruments", "category": "audio", "aliases": ["stem", "separate", "isolate vocals"], "route": "/audio/stems", "tags": ["stems", "vocals", "instruments"]},
    {"id": "audio_restore", "name": "Audio Restoration", "description": "Restore degraded or damaged audio", "category": "audio", "aliases": ["restore", "fix audio", "repair audio"], "route": "/audio/restore", "tags": ["restore", "repair", "degraded"]},
    {"id": "room_tone", "name": "Room Tone Generator", "description": "Generate matching room tone to fill gaps", "category": "audio", "aliases": ["tone", "ambient", "fill"], "route": "/audio/room-tone", "tags": ["room tone", "ambient", "fill"]},
    {"id": "spectral_repair", "name": "Spectral Repair", "description": "Repair audio using spectral editing", "category": "audio", "aliases": ["spectral", "frequency repair"], "route": "/audio/spectral-repair", "tags": ["spectral", "frequency"]},
    {"id": "surround_mix", "name": "Surround Sound Mix", "description": "Create 5.1/7.1 surround sound mix", "category": "audio", "aliases": ["surround", "5.1", "7.1", "spatial"], "route": "/audio/surround", "tags": ["surround", "spatial", "5.1"]},
    {"id": "audio_visualizer", "name": "Audio Visualizer", "description": "Generate waveform and spectrum visualizations", "category": "audio", "aliases": ["visualizer", "waveform", "spectrum"], "route": "/audio/visualize", "tags": ["waveform", "spectrum", "visualizer"]},
    {"id": "profanity_bleep", "name": "Profanity Bleep", "description": "Auto-detect and bleep profanity", "category": "audio", "aliases": ["bleep", "censor", "mute words"], "route": "/audio/profanity-bleep", "tags": ["bleep", "censor", "profanity"]},
    {"id": "dialogue_premix", "name": "Dialogue Premix", "description": "Automated dialogue premix with EQ and compression", "category": "audio", "aliases": ["premix", "dialogue mix"], "route": "/audio/dialogue-premix", "tags": ["dialogue", "premix", "mix"]},
    {"id": "loudness_match", "name": "Loudness Match", "description": "Match loudness between multiple clips", "category": "audio", "aliases": ["match loudness", "level match"], "route": "/audio/loudness-match", "tags": ["loudness", "match", "level"]},
    {"id": "audio_fingerprint", "name": "Audio Fingerprint", "description": "Generate audio fingerprint for content ID", "category": "audio", "aliases": ["fingerprint", "content id"], "route": "/audio/fingerprint", "tags": ["fingerprint", "identification"]},
    {"id": "music_gen", "name": "AI Music Generation", "description": "Generate background music with AI", "category": "audio", "aliases": ["generate music", "ai music", "background music"], "route": "/audio/music-gen", "tags": ["music", "generate", "ai"]},
    {"id": "sfx_gen", "name": "AI Sound Effects", "description": "Generate sound effects from text descriptions", "category": "audio", "aliases": ["sound effects", "sfx", "foley"], "route": "/audio/sfx-gen", "tags": ["sfx", "sound effects", "foley"]},
    {"id": "voice_gen", "name": "Voice Generation", "description": "Text-to-speech voice generation", "category": "audio", "aliases": ["tts", "text to speech", "narration"], "route": "/audio/voice-gen", "tags": ["tts", "voice", "narration"]},
    {"id": "voice_conversion", "name": "Voice Conversion", "description": "Convert voice to different speaker", "category": "audio", "aliases": ["voice clone", "voice swap"], "route": "/audio/voice-convert", "tags": ["voice", "conversion", "clone"]},
    {"id": "audio_sync", "name": "Audio Sync", "description": "Sync separate audio to video automatically", "category": "audio", "aliases": ["sync", "align audio"], "route": "/audio/sync", "tags": ["sync", "align", "timecode"]},
    {"id": "audiogram", "name": "Audiogram", "description": "Create audiogram video from audio file", "category": "audio", "aliases": ["podcast video", "audio video"], "route": "/audio/audiogram", "tags": ["audiogram", "podcast", "waveform"]},

    # --- Captions & Subtitles ---
    {"id": "add_captions", "name": "Generate Captions", "description": "Transcribe speech and generate captions", "category": "captions", "aliases": ["transcribe", "subtitles", "srt"], "route": "/captions", "tags": ["captions", "transcribe", "subtitles"]},
    {"id": "styled_captions", "name": "Styled Captions", "description": "Animated captions with custom styles", "category": "captions", "aliases": ["fancy captions", "animated text"], "route": "/captions/styled", "tags": ["styled", "animated", "captions"]},
    {"id": "caption_burnin", "name": "Burn-in Captions", "description": "Permanently burn captions into video", "category": "captions", "aliases": ["hardcode subs", "embed captions"], "route": "/captions/burnin", "tags": ["burnin", "hardcode", "embed"]},
    {"id": "multilang_subtitle", "name": "Multi-Language Subtitles", "description": "Translate subtitles to multiple languages", "category": "captions", "aliases": ["translate subs", "multi language"], "route": "/captions/translate", "tags": ["translate", "multilingual", "subtitles"]},
    {"id": "social_captions", "name": "Social Captions", "description": "Platform-optimized caption styles", "category": "captions", "aliases": ["tiktok captions", "reel captions"], "route": "/captions/social", "tags": ["social", "tiktok", "instagram"]},
    {"id": "sdh_format", "name": "SDH Formatting", "description": "Format subtitles for deaf and hard of hearing", "category": "captions", "aliases": ["sdh", "accessibility subs"], "route": "/captions/sdh", "tags": ["sdh", "accessibility", "deaf"]},
    {"id": "diarize", "name": "Speaker Diarization", "description": "Identify and label different speakers", "category": "captions", "aliases": ["speakers", "who said what"], "route": "/diarize", "tags": ["diarize", "speakers", "identification"]},

    # --- Video Processing ---
    {"id": "scene_detect", "name": "Scene Detection", "description": "Detect scene boundaries in video", "category": "video", "aliases": ["scenes", "shot boundaries"], "route": "/video/scenes", "tags": ["scenes", "detection", "boundaries"]},
    {"id": "upscale", "name": "AI Upscale", "description": "Upscale video resolution with AI", "category": "video", "aliases": ["upscale", "super resolution", "enhance resolution"], "route": "/video/upscale", "tags": ["upscale", "resolution", "enhance"]},
    {"id": "stabilize", "name": "Video Stabilization", "description": "Stabilize shaky footage", "category": "video",  "aliases": ["stabilize", "deshake", "smooth"], "route": "/video/stabilize", "tags": ["stabilize", "shake", "smooth"]},
    {"id": "denoise", "name": "Video Denoise", "description": "Remove noise and grain from video", "category": "video", "aliases": ["denoise", "noise reduction", "clean video"], "route": "/video/denoise", "tags": ["denoise", "noise", "grain"]},
    {"id": "deinterlace", "name": "Deinterlace", "description": "Convert interlaced video to progressive", "category": "video", "aliases": ["deinterlace", "progressive"], "route": "/video/deinterlace", "tags": ["deinterlace", "interlaced", "progressive"]},
    {"id": "framerate_convert", "name": "Frame Rate Convert", "description": "Convert video frame rate with motion interpolation", "category": "video", "aliases": ["fps", "frame rate", "slow motion"], "route": "/video/fps-convert", "tags": ["framerate", "fps", "interpolation"]},
    {"id": "smart_reframe", "name": "Smart Reframe", "description": "AI-powered aspect ratio conversion", "category": "video", "aliases": ["reframe", "crop", "aspect ratio"], "route": "/video/reframe", "tags": ["reframe", "aspect", "crop"]},
    {"id": "speed_ramp", "name": "Speed Ramp", "description": "Create smooth speed ramp transitions", "category": "video", "aliases": ["speed", "ramp", "time remap"], "route": "/video/speed-ramp", "tags": ["speed", "ramp", "time"]},
    {"id": "video_repair", "name": "Video Repair", "description": "Repair corrupted or damaged video files", "category": "video", "aliases": ["repair", "fix video", "corrupted"], "route": "/video/repair", "tags": ["repair", "fix", "corrupted"]},
    {"id": "proxy_gen", "name": "Generate Proxies", "description": "Create low-res proxy files for editing", "category": "video", "aliases": ["proxy", "low res", "offline"], "route": "/video/proxy", "tags": ["proxy", "offline", "low-res"]},
    {"id": "object_removal", "name": "Object Removal", "description": "Remove unwanted objects from video", "category": "video", "aliases": ["remove object", "erase", "inpaint"], "route": "/video/object-remove", "tags": ["remove", "object", "inpaint"]},
    {"id": "bg_replace", "name": "Background Replace", "description": "Replace video background with AI", "category": "video", "aliases": ["green screen", "background", "chroma key"], "route": "/video/bg-replace", "tags": ["background", "replace", "chroma"]},
    {"id": "face_reframe", "name": "Face Reframe", "description": "Auto-reframe video to track faces", "category": "video", "aliases": ["face track", "talking head crop"], "route": "/video/face-reframe", "tags": ["face", "reframe", "track"]},

    # --- Color ---
    {"id": "auto_color", "name": "Auto Color Grade", "description": "AI-powered automatic color grading", "category": "color", "aliases": ["color grade", "auto grade", "color correct"], "route": "/video/auto-color", "tags": ["color", "grade", "auto"]},
    {"id": "color_match", "name": "Color Match", "description": "Match colors between clips or to reference", "category": "color", "aliases": ["match color", "color copy"], "route": "/color/match", "tags": ["color", "match", "reference"]},
    {"id": "lut_apply", "name": "Apply LUT", "description": "Apply color lookup table to video", "category": "color", "aliases": ["lut", "color lookup", "cube file"], "route": "/color/lut", "tags": ["lut", "lookup", "cube"]},
    {"id": "hsl_qualifier", "name": "HSL Qualifier", "description": "Isolate colors with HSL qualifier", "category": "color", "aliases": ["hsl", "color isolation", "secondary color"], "route": "/color/hsl", "tags": ["hsl", "qualifier", "isolation"]},
    {"id": "color_wheels", "name": "Color Wheels", "description": "Lift/gamma/gain color adjustment", "category": "color", "aliases": ["wheels", "lift gamma gain", "color correct"], "route": "/color/wheels", "tags": ["wheels", "lift", "gamma", "gain"]},
    {"id": "film_emulation", "name": "Film Emulation", "description": "Emulate classic film stock looks", "category": "color", "aliases": ["film look", "kodak", "fuji"], "route": "/color/film-emulation", "tags": ["film", "emulation", "stock"]},
    {"id": "hdr_tools", "name": "HDR Tools", "description": "HDR grading and tone mapping", "category": "color", "aliases": ["hdr", "tone map", "high dynamic range"], "route": "/color/hdr", "tags": ["hdr", "tone", "dynamic range"]},

    # --- Effects ---
    {"id": "transitions_3d", "name": "3D Transitions", "description": "GPU-accelerated 3D transitions", "category": "effects", "aliases": ["transition", "3d effect"], "route": "/effects/transitions-3d", "tags": ["transition", "3d", "gpu"]},
    {"id": "style_transfer", "name": "Style Transfer", "description": "Apply artistic styles to video", "category": "effects", "aliases": ["artistic", "painting", "ai style"], "route": "/effects/style-transfer", "tags": ["style", "artistic", "transfer"]},
    {"id": "glitch_effects", "name": "Glitch Effects", "description": "Apply glitch and distortion effects", "category": "effects", "aliases": ["glitch", "distortion", "datamosh"], "route": "/effects/glitch", "tags": ["glitch", "distortion", "datamosh"]},
    {"id": "retro_effects", "name": "Retro Effects", "description": "VHS, CRT, and vintage video effects", "category": "effects", "aliases": ["vhs", "crt", "vintage"], "route": "/effects/retro", "tags": ["retro", "vhs", "vintage"]},
    {"id": "particles", "name": "Particle Effects", "description": "Add particle systems to video", "category": "effects", "aliases": ["particles", "snow", "fire", "sparks"], "route": "/effects/particles", "tags": ["particles", "snow", "fire"]},
    {"id": "kinetic_type", "name": "Kinetic Typography", "description": "Animated text and typography effects", "category": "effects", "aliases": ["kinetic text", "animated text", "text effects"], "route": "/effects/kinetic-type", "tags": ["kinetic", "typography", "text"]},
    {"id": "lower_thirds", "name": "Lower Thirds", "description": "Generate lower third graphics", "category": "effects", "aliases": ["lower third", "name tag", "title"], "route": "/effects/lower-thirds", "tags": ["lower thirds", "graphics", "title"]},
    {"id": "depth_effects", "name": "Depth Effects", "description": "Depth-based blur and focus effects", "category": "effects", "aliases": ["depth blur", "bokeh", "focus"], "route": "/effects/depth", "tags": ["depth", "blur", "bokeh"]},
    {"id": "overlays", "name": "Overlays", "description": "Apply overlay effects (light leaks, film grain)", "category": "effects", "aliases": ["light leak", "film grain", "overlay"], "route": "/effects/overlays", "tags": ["overlay", "light leak", "grain"]},
    {"id": "split_screen", "name": "Split Screen", "description": "Create split screen compositions", "category": "effects", "aliases": ["split", "multi view", "side by side"], "route": "/effects/split-screen", "tags": ["split", "screen", "multi"]},

    # --- Export ---
    {"id": "export_h265", "name": "Export H.265", "description": "Export video in H.265/HEVC format", "category": "export", "aliases": ["hevc", "h265 export"], "route": "/export/h265", "tags": ["export", "h265", "hevc"]},
    {"id": "export_prores", "name": "Export ProRes", "description": "Export in Apple ProRes format", "category": "export", "aliases": ["prores", "apple"], "route": "/export/prores", "tags": ["export", "prores", "apple"]},
    {"id": "export_av1", "name": "Export AV1", "description": "Export in AV1 format for web", "category": "export", "aliases": ["av1", "web video"], "route": "/export/av1", "tags": ["export", "av1", "web"]},
    {"id": "export_gif", "name": "Export GIF", "description": "Convert video to animated GIF", "category": "export", "aliases": ["gif", "animated"], "route": "/export/gif", "tags": ["gif", "animated", "export"]},
    {"id": "export_presets", "name": "Export Presets", "description": "Platform-specific export presets", "category": "export", "aliases": ["presets", "youtube", "tiktok"], "route": "/export/presets", "tags": ["presets", "platform", "export"]},
    {"id": "multi_publish", "name": "Multi-Platform Publish", "description": "Export and upload to multiple platforms", "category": "export", "aliases": ["publish", "upload", "distribute"], "route": "/export/multi-publish", "tags": ["publish", "upload", "multi"]},
    {"id": "batch_transcode", "name": "Batch Transcode", "description": "Transcode multiple files in batch", "category": "export", "aliases": ["batch", "transcode", "convert"], "route": "/batch/transcode", "tags": ["batch", "transcode", "convert"]},

    # --- AI ---
    {"id": "auto_edit", "name": "Auto Edit", "description": "AI-powered automatic video editing", "category": "ai", "aliases": ["ai edit", "smart edit"], "route": "/ai/auto-edit", "tags": ["ai", "auto", "edit"]},
    {"id": "highlights", "name": "Highlight Detection", "description": "AI-detect highlight moments in video", "category": "ai", "aliases": ["best moments", "key moments"], "route": "/ai/highlights", "tags": ["highlights", "moments", "detect"]},
    {"id": "thumbnail_gen", "name": "Thumbnail Generator", "description": "AI-powered thumbnail generation", "category": "ai", "aliases": ["thumbnail", "cover image"], "route": "/thumbnail", "tags": ["thumbnail", "cover", "image"]},
    {"id": "chat_editor", "name": "Chat Editor", "description": "Edit video with natural language commands", "category": "ai", "aliases": ["nlp edit", "chat", "natural language"], "route": "/ai/chat-edit", "tags": ["chat", "nlp", "natural language"]},
    {"id": "broll_suggest", "name": "B-Roll Suggestions", "description": "AI-suggested b-roll insert points", "category": "ai", "aliases": ["broll", "b-roll", "cutaway"], "route": "/ai/broll-suggest", "tags": ["broll", "suggest", "cutaway"]},
    {"id": "auto_montage", "name": "Auto Montage", "description": "AI-assembled montage from multiple clips", "category": "ai", "aliases": ["montage", "compilation"], "route": "/ai/montage", "tags": ["montage", "compilation", "auto"]},
    {"id": "deepfake_detect", "name": "Deepfake Detection", "description": "Detect AI-generated or manipulated content", "category": "ai", "aliases": ["deepfake", "fake detect", "authenticity"], "route": "/ai/deepfake-detect", "tags": ["deepfake", "detect", "authenticity"]},
    {"id": "content_moderation", "name": "Content Moderation", "description": "Detect inappropriate or unsafe content", "category": "ai", "aliases": ["moderation", "nsfw", "safety"], "route": "/ai/moderation", "tags": ["moderation", "safety", "nsfw"]},
    {"id": "video_summary", "name": "Video Summary", "description": "AI-generated video summary and key points", "category": "ai", "aliases": ["summarize", "recap"], "route": "/ai/summary", "tags": ["summary", "recap", "key points"]},
    {"id": "seo_optimizer", "name": "SEO Optimizer", "description": "Generate SEO-optimized titles and descriptions", "category": "ai", "aliases": ["seo", "title", "description"], "route": "/ai/seo", "tags": ["seo", "title", "optimize"]},

    # --- Editing ---
    {"id": "ripple_edit", "name": "Ripple Edit", "description": "Ripple edit with automatic gap closure", "category": "editing", "aliases": ["ripple", "gap close"], "route": "/edit/ripple", "tags": ["ripple", "edit", "gap"]},
    {"id": "multicam", "name": "Multicam Edit", "description": "Multi-camera editing and sync", "category": "editing", "aliases": ["multicam", "multi angle", "camera switch"], "route": "/multicam", "tags": ["multicam", "angle", "switch"]},
    {"id": "paper_edit", "name": "Paper Edit", "description": "Edit video by editing transcript text", "category": "editing", "aliases": ["text edit", "transcript edit"], "route": "/edit/paper", "tags": ["paper", "transcript", "text"]},
    {"id": "rough_cut", "name": "Rough Cut Assembly", "description": "Auto-assemble rough cut from selects", "category": "editing", "aliases": ["assembly", "first cut"], "route": "/edit/rough-cut", "tags": ["rough cut", "assembly", "selects"]},
    {"id": "dead_time", "name": "Dead Time Removal", "description": "Remove dead time and pauses", "category": "editing", "aliases": ["dead air", "remove pauses", "tighten"], "route": "/edit/dead-time", "tags": ["dead time", "pauses", "tighten"]},
    {"id": "beat_cuts", "name": "Beat-Synced Cuts", "description": "Cut video to the beat of music", "category": "editing", "aliases": ["beat sync", "music sync", "rhythm"], "route": "/edit/beat-cuts", "tags": ["beat", "sync", "rhythm"]},
    {"id": "long_to_shorts", "name": "Long to Shorts", "description": "Extract short clips from long video", "category": "editing", "aliases": ["shorts", "clips", "tiktok"], "route": "/edit/long-to-shorts", "tags": ["shorts", "clips", "extract"]},

    # --- Analysis ---
    {"id": "shot_classify", "name": "Shot Classification", "description": "Classify shot types in video", "category": "analysis", "aliases": ["shot type", "classify shots"], "route": "/analysis/shot-classify", "tags": ["shot", "classify", "type"]},
    {"id": "pacing_analysis", "name": "Pacing Analysis", "description": "Analyze edit pacing and rhythm", "category": "analysis", "aliases": ["pacing", "rhythm", "tempo"], "route": "/analysis/pacing", "tags": ["pacing", "rhythm", "analysis"]},
    {"id": "engagement_predict", "name": "Engagement Prediction", "description": "Predict viewer engagement and retention", "category": "analysis", "aliases": ["engagement", "retention", "viewer"], "route": "/analysis/engagement", "tags": ["engagement", "retention", "predict"]},
    {"id": "broadcast_qc", "name": "Broadcast QC", "description": "Broadcast quality control checks", "category": "analysis", "aliases": ["qc", "quality check", "broadcast"], "route": "/qc/broadcast", "tags": ["qc", "broadcast", "quality"]},
    {"id": "duplicate_detect", "name": "Duplicate Detection", "description": "Detect duplicate footage", "category": "analysis", "aliases": ["duplicate", "similar", "find duplicates"], "route": "/analysis/duplicates", "tags": ["duplicate", "detect", "similar"]},
    {"id": "emotion_timeline", "name": "Emotion Timeline", "description": "Map emotional arcs across video timeline", "category": "analysis", "aliases": ["emotion", "sentiment", "mood arc"], "route": "/analysis/emotion", "tags": ["emotion", "sentiment", "timeline"]},

    # --- Social ---
    {"id": "end_screen", "name": "End Screen Generator", "description": "Create YouTube end screen elements", "category": "social", "aliases": ["end card", "outro"], "route": "/social/end-screen", "tags": ["end screen", "youtube", "outro"]},
    {"id": "shorts_pipeline", "name": "Shorts Pipeline", "description": "Full pipeline for short-form content", "category": "social", "aliases": ["reels", "tiktok", "youtube shorts"], "route": "/social/shorts", "tags": ["shorts", "reels", "tiktok"]},
    {"id": "content_calendar", "name": "Content Calendar", "description": "Plan and schedule content publishing", "category": "social", "aliases": ["calendar", "schedule", "plan"], "route": "/social/calendar", "tags": ["calendar", "schedule", "plan"]},
    {"id": "quiz_overlay", "name": "Quiz Overlay", "description": "Add interactive quiz overlays", "category": "social", "aliases": ["quiz", "poll", "interactive"], "route": "/social/quiz", "tags": ["quiz", "poll", "interactive"]},

    # --- Restoration ---
    {"id": "old_restoration", "name": "Old Film Restoration", "description": "Restore old or degraded film footage", "category": "restoration", "aliases": ["restore film", "old video", "archive"], "route": "/video/old-restore", "tags": ["restore", "old", "archive"]},
    {"id": "rolling_shutter", "name": "Rolling Shutter Fix", "description": "Correct rolling shutter distortion", "category": "restoration", "aliases": ["rolling shutter", "jello", "skew"], "route": "/video/rolling-shutter", "tags": ["rolling shutter", "fix", "distortion"]},
    {"id": "lens_correction", "name": "Lens Correction", "description": "Correct lens distortion and chromatic aberration", "category": "restoration", "aliases": ["lens", "distortion", "barrel"], "route": "/video/lens-correct", "tags": ["lens", "distortion", "correction"]},
    {"id": "clean_plate", "name": "Clean Plate", "description": "Generate clean background plate from video", "category": "restoration", "aliases": ["clean plate", "background clean"], "route": "/video/clean-plate", "tags": ["clean plate", "background"]},

    # --- Motion ---
    {"id": "motion_tracking", "name": "Motion Tracking", "description": "Track objects and apply motion data", "category": "motion", "aliases": ["track", "follow", "motion"], "route": "/video/motion-track", "tags": ["motion", "track", "follow"]},
    {"id": "planar_track", "name": "Planar Tracking", "description": "Track planar surfaces for compositing", "category": "motion", "aliases": ["planar", "corner pin", "surface track"], "route": "/video/planar-track", "tags": ["planar", "surface", "compositing"]},
    {"id": "hyperlapse", "name": "Hyperlapse", "description": "Create stabilized hyperlapse from footage", "category": "motion", "aliases": ["hyperlapse", "time lapse"], "route": "/video/hyperlapse", "tags": ["hyperlapse", "timelapse", "stabilize"]},
    {"id": "cinemagraph", "name": "Cinemagraph", "description": "Create cinemagraph from video clip", "category": "motion", "aliases": ["living photo", "still motion"], "route": "/video/cinemagraph", "tags": ["cinemagraph", "living photo"]},

    # --- 3D/VFX ---
    {"id": "camera_solver", "name": "Camera Solver", "description": "3D camera tracking and solving", "category": "3d_vfx", "aliases": ["camera track", "3d track", "matchmove"], "route": "/vfx/camera-solve", "tags": ["camera", "3d", "solve"]},
    {"id": "gaussian_splat", "name": "3D Gaussian Splatting", "description": "Generate 3D scenes from video", "category": "3d_vfx", "aliases": ["gaussian", "3d scene", "nerf"], "route": "/vfx/gaussian-splat", "tags": ["gaussian", "3d", "splat"]},
    {"id": "sky_replace", "name": "Sky Replacement", "description": "AI sky replacement in video", "category": "3d_vfx", "aliases": ["sky", "replace sky"], "route": "/vfx/sky-replace", "tags": ["sky", "replace", "environment"]},
    {"id": "relighting", "name": "AI Relighting", "description": "Relight scenes with AI", "category": "3d_vfx", "aliases": ["relight", "lighting", "ai light"], "route": "/vfx/relight", "tags": ["relight", "lighting", "ai"]},

    # --- Broadcast ---
    {"id": "broadcast_cc", "name": "Broadcast Closed Captions", "description": "CEA-608/708 closed caption support", "category": "broadcast", "aliases": ["closed captions", "cea", "broadcast cc"], "route": "/broadcast/cc", "tags": ["broadcast", "closed captions", "cea"]},
    {"id": "timecode_utils", "name": "Timecode Tools", "description": "Timecode conversion and manipulation", "category": "broadcast", "aliases": ["timecode", "tc", "smpte"], "route": "/broadcast/timecode", "tags": ["timecode", "smpte", "conversion"]},
    {"id": "safe_zones", "name": "Safe Zones", "description": "Display title and action safe zones", "category": "broadcast", "aliases": ["safe area", "title safe", "action safe"], "route": "/broadcast/safe-zones", "tags": ["safe zones", "title safe", "broadcast"]},
    {"id": "mxf_support", "name": "MXF Support", "description": "MXF file import and export", "category": "broadcast", "aliases": ["mxf", "broadcast format"], "route": "/broadcast/mxf", "tags": ["mxf", "broadcast", "format"]},

    # --- Collaboration ---
    {"id": "review_links", "name": "Review Links", "description": "Generate shareable review links", "category": "collaboration", "aliases": ["review", "share", "feedback"], "route": "/collab/review-links", "tags": ["review", "share", "link"]},
    {"id": "project_notes", "name": "Project Notes", "description": "Collaborative project notes", "category": "collaboration", "aliases": ["notes", "comments", "annotations"], "route": "/collab/notes", "tags": ["notes", "comments", "project"]},
    {"id": "team_presets", "name": "Team Presets", "description": "Shared team presets and templates", "category": "collaboration", "aliases": ["team", "shared", "presets"], "route": "/collab/team-presets", "tags": ["team", "presets", "shared"]},
    {"id": "edit_branches", "name": "Edit Branches", "description": "Version-controlled edit branches", "category": "collaboration", "aliases": ["branches", "versions", "history"], "route": "/collab/branches", "tags": ["branches", "version", "history"]},

    # --- Platform ---
    {"id": "watch_folder", "name": "Watch Folder", "description": "Monitor folder for automatic processing", "category": "platform", "aliases": ["watch", "auto process", "hot folder"], "route": "/platform/watch-folder", "tags": ["watch", "folder", "auto"]},
    {"id": "render_queue", "name": "Render Queue", "description": "Manage background render queue", "category": "platform", "aliases": ["queue", "render", "batch render"], "route": "/platform/render-queue", "tags": ["render", "queue", "batch"]},
    {"id": "macro_recorder", "name": "Macro Recorder", "description": "Record and replay editing macros", "category": "platform", "aliases": ["macro", "record", "automate"], "route": "/platform/macro", "tags": ["macro", "record", "automate"]},
    {"id": "webhook_integrations", "name": "Webhook Integrations", "description": "Send webhook notifications on events", "category": "platform", "aliases": ["webhook", "notify", "integrate"], "route": "/platform/webhooks", "tags": ["webhook", "integration", "notify"]},
    {"id": "gpu_dashboard", "name": "GPU Dashboard", "description": "Monitor GPU usage and performance", "category": "platform", "aliases": ["gpu", "performance", "monitor"], "route": "/platform/gpu", "tags": ["gpu", "dashboard", "performance"]},
    {"id": "plugin_marketplace", "name": "Plugin Marketplace", "description": "Browse and install community plugins", "category": "platform", "aliases": ["plugins", "extensions", "marketplace"], "route": "/platform/plugins", "tags": ["plugins", "marketplace", "extensions"]},

    # --- Additional features to reach 302+ ---
    {"id": "eye_contact", "name": "Eye Contact Fix", "description": "AI eye contact correction for interviews", "category": "ai", "aliases": ["eye contact", "gaze"], "route": "/ai/eye-contact", "tags": ["eye", "contact", "gaze"]},
    {"id": "lip_sync_verify", "name": "Lip Sync Verify", "description": "Verify lip sync accuracy", "category": "ai", "aliases": ["lip sync", "verify sync"], "route": "/ai/lip-sync-verify", "tags": ["lip", "sync", "verify"]},
    {"id": "face_swap", "name": "Face Swap", "description": "AI face replacement in video", "category": "ai", "aliases": ["face replace", "swap face"], "route": "/ai/face-swap", "tags": ["face", "swap", "replace"]},
    {"id": "face_tagging", "name": "Face Tagging", "description": "Detect and tag faces in video", "category": "ai", "aliases": ["face detect", "tag faces"], "route": "/ai/face-tag", "tags": ["face", "tag", "detect"]},
    {"id": "ocr_extract", "name": "OCR Extract", "description": "Extract text from video frames", "category": "ai", "aliases": ["ocr", "text extract", "read text"], "route": "/ai/ocr", "tags": ["ocr", "text", "extract"]},
    {"id": "ai_dubbing", "name": "AI Dubbing", "description": "AI-powered multi-language dubbing", "category": "ai", "aliases": ["dub", "voice over", "translate voice"], "route": "/ai/dubbing", "tags": ["dubbing", "voice", "translate"]},
    {"id": "talking_head", "name": "Talking Head Detect", "description": "Detect talking head segments", "category": "ai", "aliases": ["talking head", "speaker detect"], "route": "/ai/talking-head", "tags": ["talking head", "speaker"]},
    {"id": "emotion_voice", "name": "Emotion from Voice", "description": "Detect emotions from voice tone", "category": "ai", "aliases": ["voice emotion", "tone analysis"], "route": "/ai/emotion-voice", "tags": ["emotion", "voice", "tone"]},
    {"id": "pii_redact", "name": "PII Redaction", "description": "Redact personally identifiable information", "category": "ai", "aliases": ["redact", "privacy", "blur faces"], "route": "/ai/pii-redact", "tags": ["pii", "redact", "privacy"]},
    {"id": "doc_redact", "name": "Document Redaction", "description": "Redact documents visible in video", "category": "ai", "aliases": ["redact docs", "blur documents"], "route": "/ai/doc-redact", "tags": ["document", "redact", "privacy"]},
    {"id": "plate_blur", "name": "License Plate Blur", "description": "Auto-detect and blur license plates", "category": "ai", "aliases": ["blur plates", "license plate"], "route": "/ai/plate-blur", "tags": ["plate", "blur", "privacy"]},
    {"id": "data_animation", "name": "Data Animation", "description": "Animate data and charts in video", "category": "effects", "aliases": ["data viz", "charts", "infographic"], "route": "/effects/data-animation", "tags": ["data", "animation", "charts"]},
    {"id": "news_ticker", "name": "News Ticker", "description": "Generate scrolling news ticker", "category": "effects", "aliases": ["ticker", "crawl", "breaking news"], "route": "/effects/news-ticker", "tags": ["ticker", "news", "crawl"]},
    {"id": "credits_gen", "name": "Credits Generator", "description": "Generate rolling credits", "category": "effects", "aliases": ["credits", "end credits", "rolling credits"], "route": "/effects/credits", "tags": ["credits", "rolling", "end"]},
    {"id": "shape_animation", "name": "Shape Animation", "description": "Animate shapes and motion graphics", "category": "effects", "aliases": ["shapes", "motion graphics", "mograph"], "route": "/effects/shapes", "tags": ["shapes", "animation", "motion"]},
    {"id": "callout_gen", "name": "Callout Generator", "description": "Generate callout annotations", "category": "effects", "aliases": ["callout", "annotation", "arrow"], "route": "/effects/callouts", "tags": ["callout", "annotation", "highlight"]},
    {"id": "click_overlay", "name": "Click Overlay", "description": "Visualize mouse clicks in screen recordings", "category": "effects", "aliases": ["mouse click", "cursor highlight"], "route": "/effects/click-overlay", "tags": ["click", "cursor", "screen"]},
    {"id": "cursor_zoom", "name": "Cursor Zoom", "description": "Auto-zoom to cursor in screen recordings", "category": "effects", "aliases": ["zoom cursor", "follow mouse"], "route": "/effects/cursor-zoom", "tags": ["cursor", "zoom", "screen"]},
    {"id": "telemetry_overlay", "name": "Telemetry Overlay", "description": "Overlay GPS, speed, altitude data", "category": "effects", "aliases": ["gps overlay", "speed overlay", "telemetry"], "route": "/effects/telemetry", "tags": ["telemetry", "gps", "overlay"]},
    {"id": "watermark", "name": "Watermark", "description": "Add visible or invisible watermarks", "category": "effects", "aliases": ["watermark", "logo", "brand"], "route": "/effects/watermark", "tags": ["watermark", "logo", "brand"]},
    {"id": "invisible_watermark", "name": "Invisible Watermark", "description": "Embed invisible forensic watermark", "category": "effects", "aliases": ["forensic", "hidden mark", "invisible"], "route": "/effects/invisible-watermark", "tags": ["invisible", "forensic", "watermark"]},
    {"id": "podcast_extract", "name": "Podcast Extract", "description": "Extract clips from podcast episodes", "category": "editing", "aliases": ["podcast clip", "extract segment"], "route": "/podcast/extract", "tags": ["podcast", "extract", "clip"]},
    {"id": "podcast_rss", "name": "Podcast RSS", "description": "Generate podcast RSS feed", "category": "editing", "aliases": ["rss", "feed", "podcast publish"], "route": "/podcast/rss", "tags": ["podcast", "rss", "feed"]},
    {"id": "show_notes", "name": "Show Notes", "description": "Auto-generate show notes from audio", "category": "editing", "aliases": ["show notes", "episode notes"], "route": "/podcast/show-notes", "tags": ["show notes", "podcast", "notes"]},
    {"id": "chapter_gen", "name": "Chapter Generator", "description": "Auto-generate video chapters", "category": "editing", "aliases": ["chapters", "markers", "segments"], "route": "/chapters", "tags": ["chapters", "markers", "segments"]},
    {"id": "photo_montage", "name": "Photo Montage", "description": "Create video from photos with transitions", "category": "editing", "aliases": ["slideshow", "photo video"], "route": "/edit/photo-montage", "tags": ["photo", "montage", "slideshow"]},
    {"id": "instant_replay", "name": "Instant Replay", "description": "Create slow-motion instant replay", "category": "editing", "aliases": ["replay", "slow motion replay"], "route": "/edit/instant-replay", "tags": ["replay", "slow motion"]},
    {"id": "fit_to_fill", "name": "Fit to Fill", "description": "Speed-adjust clip to fill target duration", "category": "editing", "aliases": ["fit", "duration match"], "route": "/edit/fit-to-fill", "tags": ["fit", "fill", "duration"]},
    {"id": "morph_cut", "name": "Morph Cut", "description": "Seamless jump cut transitions", "category": "editing", "aliases": ["morph", "jump cut fix", "smooth cut"], "route": "/edit/morph-cut", "tags": ["morph", "jump cut", "smooth"]},
    {"id": "through_edit", "name": "Through Edit Detect", "description": "Detect unnecessary through-edits", "category": "editing", "aliases": ["through edit", "continuous"], "route": "/edit/through-edit", "tags": ["through edit", "detect"]},
    {"id": "color_scopes", "name": "Color Scopes", "description": "Waveform, vectorscope, histogram", "category": "color", "aliases": ["scopes", "waveform", "vectorscope"], "route": "/color/scopes", "tags": ["scopes", "waveform", "vectorscope"]},
    {"id": "wide_gamut", "name": "Wide Gamut", "description": "Wide gamut color space management", "category": "color", "aliases": ["p3", "rec2020", "color space"], "route": "/color/wide-gamut", "tags": ["gamut", "p3", "rec2020"]},
    {"id": "nd_filter_sim", "name": "ND Filter Simulation", "description": "Simulate neutral density filter effects", "category": "color", "aliases": ["nd filter", "neutral density"], "route": "/color/nd-sim", "tags": ["nd", "filter", "neutral density"]},
    {"id": "log_profiles", "name": "Log Profile Convert", "description": "Convert log footage to rec709", "category": "color", "aliases": ["log", "slog", "clog", "rec709"], "route": "/color/log-convert", "tags": ["log", "slog", "rec709"]},
    {"id": "edl_export", "name": "EDL Export", "description": "Export edit decision list", "category": "export", "aliases": ["edl", "edit decision"], "route": "/export/edl", "tags": ["edl", "export", "decision"]},
    {"id": "fcpxml_export", "name": "FCPXML Export", "description": "Export to Final Cut Pro XML", "category": "export", "aliases": ["fcpxml", "final cut"], "route": "/export/fcpxml", "tags": ["fcpxml", "final cut", "export"]},
    {"id": "dnx_export", "name": "DNxHR Export", "description": "Export in DNxHR/DNxHD format", "category": "export", "aliases": ["dnxhr", "dnxhd", "avid"], "route": "/export/dnx", "tags": ["dnx", "avid", "export"]},
    {"id": "image_sequence", "name": "Image Sequence Export", "description": "Export as image sequence", "category": "export", "aliases": ["image seq", "frames", "png sequence"], "route": "/export/image-sequence", "tags": ["image", "sequence", "frames"]},
    {"id": "lossless_intermediate", "name": "Lossless Intermediate", "description": "Export lossless intermediate format", "category": "export", "aliases": ["lossless", "intermediate", "ffv1"], "route": "/export/lossless", "tags": ["lossless", "intermediate"]},
    {"id": "deliverables", "name": "Deliverables Package", "description": "Generate complete deliverables package", "category": "export", "aliases": ["deliverables", "final output", "package"], "route": "/export/deliverables", "tags": ["deliverables", "package"]},
    {"id": "streaming_package", "name": "Streaming Package", "description": "HLS/DASH adaptive streaming package", "category": "export", "aliases": ["hls", "dash", "streaming"], "route": "/export/streaming", "tags": ["streaming", "hls", "dash"]},
    {"id": "spatial_audio", "name": "Spatial Audio", "description": "Create immersive spatial audio", "category": "audio", "aliases": ["3d audio", "binaural", "ambisonics"], "route": "/audio/spatial", "tags": ["spatial", "3d", "binaural"]},
    {"id": "adr_cueing", "name": "ADR Cueing", "description": "Automated dialogue replacement cueing", "category": "audio", "aliases": ["adr", "dialogue replace", "voice over"], "route": "/audio/adr", "tags": ["adr", "dialogue", "cue"]},
    {"id": "noise_classify", "name": "Noise Classification", "description": "Classify types of audio noise", "category": "audio", "aliases": ["noise type", "classify noise"], "route": "/audio/noise-classify", "tags": ["noise", "classify", "type"]},
    {"id": "spectrogram_edit", "name": "Spectrogram Edit", "description": "Visual spectrogram-based audio editing", "category": "audio", "aliases": ["spectrogram", "visual audio"], "route": "/audio/spectrogram", "tags": ["spectrogram", "visual", "edit"]},
    {"id": "stem_remix", "name": "Stem Remix", "description": "Remix audio using separated stems", "category": "audio", "aliases": ["remix", "stem mix"], "route": "/audio/stem-remix", "tags": ["stem", "remix", "mix"]},
    {"id": "transition_sfx", "name": "Transition SFX", "description": "Auto-add sound effects to transitions", "category": "audio", "aliases": ["whoosh", "transition sound"], "route": "/audio/transition-sfx", "tags": ["transition", "sfx", "whoosh"]},
    {"id": "video_360", "name": "360 Video", "description": "360-degree video editing and export", "category": "video", "aliases": ["360", "vr video", "equirectangular"], "route": "/video/360", "tags": ["360", "vr", "equirectangular"]},
    {"id": "sdr_to_hdr", "name": "SDR to HDR", "description": "Convert SDR footage to HDR", "category": "video", "aliases": ["sdr hdr", "hdr convert"], "route": "/video/sdr-to-hdr", "tags": ["sdr", "hdr", "convert"]},
    {"id": "chromakey", "name": "Chroma Key", "description": "Green/blue screen keying", "category": "video", "aliases": ["green screen", "blue screen", "key"], "route": "/video/chromakey", "tags": ["chroma", "key", "green screen"]},
    {"id": "timelapse", "name": "Timelapse", "description": "Create timelapse from video or images", "category": "video", "aliases": ["time lapse", "speed up"], "route": "/video/timelapse", "tags": ["timelapse", "speed"]},
    {"id": "pip_layout", "name": "Picture-in-Picture", "description": "Create picture-in-picture layouts", "category": "video", "aliases": ["pip", "overlay video"], "route": "/video/pip", "tags": ["pip", "picture", "overlay"]},
    {"id": "redaction", "name": "Video Redaction", "description": "Redact regions in video", "category": "video", "aliases": ["redact", "blur region", "censor"], "route": "/video/redact", "tags": ["redact", "blur", "censor"]},
    {"id": "saliency_crop", "name": "Saliency Crop", "description": "Auto-crop based on visual saliency", "category": "video", "aliases": ["auto crop", "smart crop"], "route": "/video/saliency-crop", "tags": ["saliency", "crop", "auto"]},
    {"id": "scene_extend", "name": "Scene Extension", "description": "AI-extend scene duration", "category": "ai", "aliases": ["extend", "longer scene", "generate frames"], "route": "/ai/scene-extend", "tags": ["scene", "extend", "generate"]},
    {"id": "img_to_video", "name": "Image to Video", "description": "Convert still image to animated video", "category": "ai", "aliases": ["animate image", "image animate"], "route": "/ai/img-to-video", "tags": ["image", "video", "animate"]},
    {"id": "voice_commands", "name": "Voice Commands", "description": "Control editing with voice", "category": "platform", "aliases": ["voice control", "speech command"], "route": "/platform/voice", "tags": ["voice", "command", "speech"]},
    {"id": "stream_deck", "name": "Stream Deck", "description": "Stream Deck integration", "category": "platform", "aliases": ["elgato", "hardware control"], "route": "/platform/stream-deck", "tags": ["stream deck", "elgato", "hardware"]},
    {"id": "midi_controller", "name": "MIDI Controller", "description": "MIDI hardware controller support", "category": "platform", "aliases": ["midi", "hardware", "fader"], "route": "/platform/midi", "tags": ["midi", "controller", "hardware"]},
    {"id": "jog_wheel", "name": "Jog Wheel Support", "description": "Jog/shuttle wheel control", "category": "platform", "aliases": ["jog", "shuttle", "wheel"], "route": "/platform/jog-wheel", "tags": ["jog", "shuttle", "wheel"]},
    {"id": "scripting_console", "name": "Scripting Console", "description": "Python scripting console for automation", "category": "platform", "aliases": ["script", "console", "python"], "route": "/platform/scripting", "tags": ["script", "console", "python"]},
    {"id": "notion_sync", "name": "Notion Sync", "description": "Sync projects with Notion", "category": "collaboration", "aliases": ["notion", "sync notes"], "route": "/collab/notion", "tags": ["notion", "sync", "notes"]},
    {"id": "slack_notify", "name": "Slack Notifications", "description": "Send notifications to Slack", "category": "collaboration", "aliases": ["slack", "notify team"], "route": "/collab/slack", "tags": ["slack", "notify", "team"]},
    {"id": "frameio_integration", "name": "Frame.io Integration", "description": "Sync with Frame.io for review", "category": "collaboration", "aliases": ["frame.io", "frameio", "review"], "route": "/collab/frameio", "tags": ["frameio", "review", "sync"]},
    {"id": "project_archive", "name": "Project Archive", "description": "Archive and restore projects", "category": "collaboration", "aliases": ["archive", "backup", "restore"], "route": "/collab/archive", "tags": ["archive", "backup", "restore"]},
    {"id": "timeline_diff", "name": "Timeline Diff", "description": "Compare timeline versions", "category": "collaboration", "aliases": ["diff", "compare", "changes"], "route": "/collab/diff", "tags": ["diff", "compare", "timeline"]},
    {"id": "edit_snapshots", "name": "Edit Snapshots", "description": "Save and restore edit snapshots", "category": "collaboration", "aliases": ["snapshot", "save point", "checkpoint"], "route": "/collab/snapshots", "tags": ["snapshot", "save", "restore"]},
    {"id": "resolve_bridge", "name": "DaVinci Resolve Bridge", "description": "Export to DaVinci Resolve format", "category": "platform", "aliases": ["resolve", "davinci"], "route": "/platform/resolve", "tags": ["resolve", "davinci", "bridge"]},
    {"id": "iso_ingest", "name": "Camera ISO Ingest", "description": "Import multi-camera ISO recordings", "category": "platform", "aliases": ["iso", "ingest", "camera import"], "route": "/platform/iso-ingest", "tags": ["iso", "ingest", "camera"]},
    {"id": "structured_ingest", "name": "Structured Ingest", "description": "Import and organize media with metadata", "category": "platform", "aliases": ["import", "organize", "ingest"], "route": "/platform/ingest", "tags": ["ingest", "organize", "metadata"]},
    {"id": "storage_tiering", "name": "Storage Tiering", "description": "Manage media across storage tiers", "category": "platform", "aliases": ["storage", "tier", "archive"], "route": "/platform/storage", "tags": ["storage", "tier", "manage"]},
    {"id": "media_relink", "name": "Media Relink", "description": "Relink missing media files", "category": "platform", "aliases": ["relink", "missing media", "reconnect"], "route": "/platform/relink", "tags": ["relink", "missing", "reconnect"]},
    {"id": "auto_update", "name": "Auto Update", "description": "Check for and install updates", "category": "platform", "aliases": ["update", "upgrade"], "route": "/platform/update", "tags": ["update", "upgrade", "version"]},
    {"id": "process_isolation", "name": "Process Isolation", "description": "Run jobs in isolated processes", "category": "platform", "aliases": ["isolation", "sandbox"], "route": "/platform/isolation", "tags": ["isolation", "sandbox", "process"]},
    {"id": "hw_accel", "name": "Hardware Acceleration", "description": "Configure GPU hardware acceleration", "category": "platform", "aliases": ["gpu accel", "cuda", "nvenc"], "route": "/platform/hw-accel", "tags": ["hw", "gpu", "acceleration"]},
    {"id": "model_manager", "name": "AI Model Manager", "description": "Download and manage AI models", "category": "platform", "aliases": ["model", "download model", "ai models"], "route": "/platform/models", "tags": ["model", "manage", "download"]},
    {"id": "display_calibration", "name": "Display Calibration", "description": "Monitor color calibration tools", "category": "platform", "aliases": ["calibrate", "monitor", "display"], "route": "/platform/calibrate", "tags": ["calibrate", "display", "monitor"]},
    {"id": "ceremony_autoedit", "name": "Ceremony Auto-Edit", "description": "Auto-edit ceremony recordings", "category": "ai", "aliases": ["wedding", "ceremony", "event edit"], "route": "/ai/ceremony", "tags": ["ceremony", "wedding", "auto edit"]},
    {"id": "event_recap", "name": "Event Recap", "description": "Auto-generate event recap video", "category": "ai", "aliases": ["recap", "event summary", "highlight reel"], "route": "/ai/event-recap", "tags": ["event", "recap", "highlight"]},
    {"id": "guest_compilation", "name": "Guest Compilation", "description": "Compile multi-guest video submissions", "category": "ai", "aliases": ["compilation", "guest video", "montage"], "route": "/ai/guest-compilation", "tags": ["guest", "compilation", "montage"]},
    {"id": "screenshot_video", "name": "Screenshot to Video", "description": "Create video from screenshots with motion", "category": "ai", "aliases": ["screenshot", "app demo"], "route": "/ai/screenshot-video", "tags": ["screenshot", "demo", "video"]},
    {"id": "programmatic_video", "name": "Programmatic Video", "description": "Generate videos from templates and data", "category": "ai", "aliases": ["template video", "data-driven", "personalized"], "route": "/ai/programmatic", "tags": ["programmatic", "template", "data"]},
    {"id": "reaction_template", "name": "Reaction Template", "description": "Create reaction video templates", "category": "social", "aliases": ["reaction", "react video"], "route": "/social/reaction", "tags": ["reaction", "template", "video"]},
    {"id": "chat_replay", "name": "Chat Replay", "description": "Overlay live chat replay on video", "category": "social", "aliases": ["chat", "live chat", "replay"], "route": "/social/chat-replay", "tags": ["chat", "replay", "overlay"]},
    {"id": "stream_highlights", "name": "Stream Highlights", "description": "Extract highlights from live streams", "category": "social", "aliases": ["stream clip", "twitch clip"], "route": "/social/stream-highlights", "tags": ["stream", "highlights", "clip"]},
    {"id": "stream_chapters", "name": "Stream Chapters", "description": "Auto-chapter live stream recordings", "category": "social", "aliases": ["stream chapter", "vod chapters"], "route": "/social/stream-chapters", "tags": ["stream", "chapters", "vod"]},
    {"id": "video_to_blog", "name": "Video to Blog", "description": "Convert video content to blog post", "category": "ai", "aliases": ["blog", "article", "text content"], "route": "/ai/video-to-blog", "tags": ["blog", "article", "convert"]},
    {"id": "script_storyboard", "name": "Script to Storyboard", "description": "Generate storyboard from script", "category": "ai", "aliases": ["storyboard", "pre-viz", "script"], "route": "/ai/storyboard", "tags": ["storyboard", "script", "pre-viz"]},
    {"id": "mood_board", "name": "Mood Board", "description": "Generate visual mood boards", "category": "ai", "aliases": ["mood", "inspiration", "visual board"], "route": "/ai/mood-board", "tags": ["mood", "board", "visual"]},
    {"id": "brand_kit", "name": "Brand Kit", "description": "Manage brand colors, fonts, and logos", "category": "platform", "aliases": ["brand", "style guide", "identity"], "route": "/platform/brand-kit", "tags": ["brand", "kit", "identity"]},
    {"id": "google_fonts", "name": "Google Fonts", "description": "Browse and use Google Fonts", "category": "platform", "aliases": ["fonts", "typography", "google fonts"], "route": "/platform/fonts", "tags": ["fonts", "google", "typography"]},
    {"id": "footage_search", "name": "Footage Search", "description": "Semantic search across footage library", "category": "platform", "aliases": ["search", "find footage", "clip search"], "route": "/search/query", "tags": ["search", "footage", "semantic"]},
    {"id": "stock_search", "name": "Stock Search", "description": "Search royalty-free stock footage", "category": "platform", "aliases": ["stock footage", "royalty free", "stock video"], "route": "/search/stock", "tags": ["stock", "footage", "royalty free"]},
    {"id": "c2pa_embed", "name": "C2PA Embed", "description": "Embed content credentials (C2PA)", "category": "platform", "aliases": ["c2pa", "content credentials", "provenance"], "route": "/platform/c2pa", "tags": ["c2pa", "credentials", "provenance"]},
    {"id": "evidence_chain", "name": "Evidence Chain", "description": "Chain of custody for forensic video", "category": "platform", "aliases": ["chain of custody", "forensic", "evidence"], "route": "/platform/evidence", "tags": ["evidence", "chain", "forensic"]},
    {"id": "license_tracker", "name": "License Tracker", "description": "Track media licenses and usage rights", "category": "platform", "aliases": ["license", "rights", "usage"], "route": "/platform/licenses", "tags": ["license", "rights", "tracker"]},
    {"id": "analytics", "name": "Analytics Dashboard", "description": "Usage analytics and productivity metrics", "category": "platform", "aliases": ["analytics", "metrics", "stats"], "route": "/platform/analytics", "tags": ["analytics", "metrics", "dashboard"]},
    {"id": "remote_process", "name": "Remote Processing", "description": "Offload processing to remote servers", "category": "platform", "aliases": ["remote", "cloud render", "offload"], "route": "/platform/remote", "tags": ["remote", "cloud", "offload"]},
]


def build_feature_index(on_progress: Optional[Callable] = None) -> List[dict]:
    """Build (or return cached) feature index.

    Returns list of feature entry dicts with id, name, description,
    category, aliases, route, and tags.
    """
    global _feature_index

    if on_progress:
        on_progress(10, "Loading feature definitions...")

    with _index_lock:
        if _feature_index is not None:
            if on_progress:
                on_progress(100, "Feature index loaded from cache")
            return list(_feature_index)

        if on_progress:
            on_progress(50, "Building feature index...")

        index = []
        for feat in _FEATURE_DEFS:
            index.append({
                "id": feat["id"],
                "name": feat["name"],
                "description": feat["description"],
                "category": feat["category"],
                "aliases": list(feat.get("aliases", [])),
                "route": feat.get("route", ""),
                "tags": list(feat.get("tags", [])),
            })

        _feature_index = index

        if on_progress:
            on_progress(100, f"Built index with {len(index)} features")

        return list(_feature_index)


def fuzzy_search(
    query: str,
    index: Optional[List[dict]] = None,
    limit: int = 10,
    on_progress: Optional[Callable] = None,
) -> List[dict]:
    """Fuzzy search the feature index.

    Args:
        query: Search string.
        index: Feature index (uses singleton if None).
        limit: Max results to return.
        on_progress: Progress callback(pct, msg).

    Returns:
        List of feature dicts with added ``score`` field, ranked by relevance.
    """
    if on_progress:
        on_progress(10, "Searching features...")

    if index is None:
        index = build_feature_index()

    if not query or not query.strip():
        if on_progress:
            on_progress(100, "Empty query")
        return []

    query = query.strip()
    scored = []

    for entry in index:
        score = _score_entry(query, entry)
        if score > 0.05:
            result = dict(entry)
            result["score"] = round(score, 4)
            scored.append(result)

    scored.sort(key=lambda x: x["score"], reverse=True)

    if on_progress:
        on_progress(100, f"Found {len(scored)} matches")

    return scored[:limit]


def get_recent_features(limit: int = 5, on_progress: Optional[Callable] = None) -> List[dict]:
    """Return recently used features from the recents file.

    Args:
        limit: Max recent entries to return.
        on_progress: Progress callback(pct, msg).

    Returns:
        List of dicts with feature_id, name, timestamp, use_count.
    """
    if on_progress:
        on_progress(10, "Loading recent features...")

    try:
        if not os.path.isfile(_RECENTS_FILE):
            if on_progress:
                on_progress(100, "No recent features found")
            return []

        with open(_RECENTS_FILE, "r", encoding="utf-8") as f:
            recents = json.load(f)

        if not isinstance(recents, list):
            return []

        # Sort by last-used timestamp descending
        recents.sort(key=lambda x: x.get("timestamp", 0), reverse=True)

        if on_progress:
            on_progress(100, f"Loaded {min(len(recents), limit)} recent features")

        return recents[:limit]
    except (json.JSONDecodeError, OSError) as exc:
        logger.debug("Failed to load recent features: %s", exc)
        return []


def record_feature_use(
    feature_id: str,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Record a feature usage event for recents/frequency tracking.

    Args:
        feature_id: The feature ID to record.
        on_progress: Progress callback(pct, msg).

    Returns:
        Dict with feature_id, use_count, timestamp.
    """
    if on_progress:
        on_progress(10, "Recording feature usage...")

    _ensure_opencut_dir()

    # Load or init recents
    recents: List[dict] = []
    try:
        if os.path.isfile(_RECENTS_FILE):
            with open(_RECENTS_FILE, "r", encoding="utf-8") as f:
                recents = json.load(f)
            if not isinstance(recents, list):
                recents = []
    except (json.JSONDecodeError, OSError):
        recents = []

    if on_progress:
        on_progress(50, "Updating usage record...")

    # Use a monotonically increasing timestamp so two consecutive calls
    # get distinct values even when the wall clock has low resolution
    # (Windows `time.time()` advances in ~15.6ms steps and would tie
    # records logged within the same tick, breaking sort-by-recency).
    now = time.time()
    if recents:
        try:
            max_existing = max(float(r.get("timestamp", 0) or 0) for r in recents)
        except (TypeError, ValueError):
            max_existing = 0.0
        if now <= max_existing:
            now = max_existing + 1e-6

    # Look up the feature name from the index
    index = build_feature_index()
    name = feature_id
    for feat in index:
        if feat["id"] == feature_id:
            name = feat["name"]
            break

    # Find existing entry or create new
    existing = None
    for entry in recents:
        if entry.get("feature_id") == feature_id:
            existing = entry
            break

    if existing:
        existing["use_count"] = existing.get("use_count", 0) + 1
        existing["timestamp"] = now
        existing["name"] = name
    else:
        recents.append({
            "feature_id": feature_id,
            "name": name,
            "use_count": 1,
            "timestamp": now,
        })

    # Trim to max
    recents.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    recents = recents[:_MAX_RECENTS]

    # Write back
    try:
        with open(_RECENTS_FILE, "w", encoding="utf-8") as f:
            json.dump(recents, f, indent=2)
    except OSError as exc:
        logger.warning("Failed to save recent features: %s", exc)

    result_entry = existing if existing else recents[-1]

    if on_progress:
        on_progress(100, f"Recorded usage for {feature_id}")

    return {
        "feature_id": feature_id,
        "name": name,
        "use_count": result_entry.get("use_count", 1),
        "timestamp": now,
    }
