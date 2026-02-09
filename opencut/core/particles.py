"""
OpenCut Particle Effects Module v0.9.0

Overlay particle effects on video:
- Confetti burst, sparkles, snow, rain, fire embers, smoke, bubbles
- Configurable density, speed, color, and lifetime
- Renders particle frames via Pillow, composites via FFmpeg

Zero ML dependencies - Pillow + FFmpeg only.
"""

import logging
import math
import os
import random
import subprocess
import sys
import tempfile
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("opencut")


def _ensure_package(pkg, pip_name=None, on_progress=None):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", pip_name or pkg,
                        "--break-system-packages", "-q"], capture_output=True, timeout=300)


def _run_ffmpeg(cmd, timeout=7200):
    r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {r.stderr.decode(errors='replace')[-500:]}")


def _get_video_info(fp):
    import json
    r = subprocess.run(["ffprobe", "-v", "quiet", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height,r_frame_rate,duration",
                        "-of", "json", fp], capture_output=True, timeout=30)
    try:
        s = json.loads(r.stdout.decode())["streams"][0]
        fps_p = s.get("r_frame_rate", "30/1").split("/")
        fps = float(fps_p[0]) / float(fps_p[1]) if len(fps_p) == 2 else 30.0
        return {"width": int(s.get("width", 1920)), "height": int(s.get("height", 1080)),
                "fps": fps, "duration": float(s.get("duration", 0))}
    except Exception:
        return {"width": 1920, "height": 1080, "fps": 30.0, "duration": 0}


# ---------------------------------------------------------------------------
# Particle Presets
# ---------------------------------------------------------------------------
PARTICLE_PRESETS = {
    "confetti": {
        "label": "Confetti",
        "description": "Colorful confetti burst falling down",
        "count": 150,
        "colors": [(255, 50, 50), (50, 200, 50), (50, 100, 255),
                   (255, 200, 0), (255, 100, 200), (0, 200, 255)],
        "size_range": (4, 12),
        "speed_y": (2, 6),
        "speed_x": (-2, 2),
        "rotation": True,
        "gravity": 0.15,
        "spawn": "top",
    },
    "sparkles": {
        "label": "Sparkles",
        "description": "Twinkling sparkle particles",
        "count": 80,
        "colors": [(255, 255, 200), (255, 255, 255), (255, 240, 150)],
        "size_range": (2, 6),
        "speed_y": (-1, 1),
        "speed_x": (-1, 1),
        "rotation": False,
        "gravity": 0,
        "spawn": "random",
        "fade": True,
    },
    "snow": {
        "label": "Snow",
        "description": "Gentle snowfall",
        "count": 100,
        "colors": [(255, 255, 255), (220, 230, 255), (240, 245, 255)],
        "size_range": (2, 8),
        "speed_y": (1, 3),
        "speed_x": (-1, 1),
        "rotation": False,
        "gravity": 0.02,
        "spawn": "top",
    },
    "rain": {
        "label": "Rain",
        "description": "Rainfall streaks",
        "count": 200,
        "colors": [(180, 200, 230), (160, 190, 220)],
        "size_range": (1, 3),
        "speed_y": (10, 18),
        "speed_x": (-1, 1),
        "rotation": False,
        "gravity": 0.5,
        "spawn": "top",
        "streak": True,
    },
    "fire_embers": {
        "label": "Fire Embers",
        "description": "Rising glowing embers",
        "count": 60,
        "colors": [(255, 100, 0), (255, 150, 30), (255, 200, 50), (255, 60, 0)],
        "size_range": (2, 6),
        "speed_y": (-4, -1),
        "speed_x": (-2, 2),
        "rotation": False,
        "gravity": -0.05,
        "spawn": "bottom",
        "fade": True,
    },
    "smoke": {
        "label": "Smoke",
        "description": "Rising smoke wisps",
        "count": 40,
        "colors": [(120, 120, 120), (150, 150, 150), (100, 100, 100)],
        "size_range": (8, 20),
        "speed_y": (-2, -0.5),
        "speed_x": (-1, 1),
        "rotation": False,
        "gravity": -0.02,
        "spawn": "bottom",
        "fade": True,
    },
    "bubbles": {
        "label": "Bubbles",
        "description": "Floating soap bubbles",
        "count": 30,
        "colors": [(200, 220, 255), (180, 210, 250), (220, 235, 255)],
        "size_range": (6, 20),
        "speed_y": (-2, -0.5),
        "speed_x": (-1, 1),
        "rotation": False,
        "gravity": -0.03,
        "spawn": "bottom",
    },
}


class Particle:
    def __init__(self, x, y, vx, vy, size, color, lifetime, preset_cfg):
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.size = size
        self.color = color
        self.lifetime = lifetime
        self.age = 0
        self.alive = True
        self.gravity = preset_cfg.get("gravity", 0)
        self.fade = preset_cfg.get("fade", False)
        self.streak = preset_cfg.get("streak", False)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += self.gravity
        self.vx += random.uniform(-0.1, 0.1)
        self.age += 1
        if self.age >= self.lifetime:
            self.alive = False

    def get_alpha(self):
        if self.fade:
            return max(0, int(200 * (1 - self.age / self.lifetime)))
        return 180


def _spawn_particle(w, h, preset_cfg):
    spawn = preset_cfg.get("spawn", "random")
    sr = preset_cfg["size_range"]
    size = random.randint(sr[0], sr[1])
    color = random.choice(preset_cfg["colors"])
    vy = random.uniform(*preset_cfg["speed_y"])
    vx = random.uniform(*preset_cfg["speed_x"])
    lifetime = random.randint(30, 120)

    if spawn == "top":
        x, y = random.randint(0, w), random.randint(-20, 0)
    elif spawn == "bottom":
        x, y = random.randint(0, w), random.randint(h, h + 20)
    else:
        x, y = random.randint(0, w), random.randint(0, h)

    return Particle(x, y, vx, vy, size, color, lifetime, preset_cfg)


def overlay_particles(
    video_path: str,
    output_path: Optional[str] = None,
    output_dir: str = "",
    preset: str = "confetti",
    density: float = 1.0,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Overlay particle effects on video.

    Args:
        preset: Particle preset name from PARTICLE_PRESETS.
        density: Particle density multiplier (0.5 = half, 2.0 = double).
    """
    _ensure_package("cv2", "opencv-python-headless", on_progress)
    _ensure_package("PIL", "Pillow", on_progress)
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw

    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        directory = output_dir or os.path.dirname(video_path)
        output_path = os.path.join(directory, f"{base}_{preset}.mp4")

    preset_cfg = PARTICLE_PRESETS.get(preset, PARTICLE_PRESETS["confetti"])
    info = _get_video_info(video_path)
    w, h, fps = info["width"], info["height"], info["fps"]
    max_particles = int(preset_cfg["count"] * density)

    if on_progress:
        on_progress(5, f"Generating {preset} particles...")

    cap = cv2.VideoCapture(video_path)
    tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_video, fourcc, fps, (w, h))

    particles = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Spawn new particles
            while len(particles) < max_particles:
                particles.append(_spawn_particle(w, h, preset_cfg))

            # Create overlay
            overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            alive = []
            for p in particles:
                p.update()
                if not p.alive or p.x < -50 or p.x > w + 50 or p.y < -50 or p.y > h + 50:
                    continue
                alive.append(p)

                alpha = p.get_alpha()
                c = p.color + (alpha,)

                if p.streak:
                    draw.line([(int(p.x), int(p.y)),
                               (int(p.x - p.vx), int(p.y - p.vy * 2))],
                              fill=c, width=p.size)
                else:
                    x0, y0 = int(p.x - p.size), int(p.y - p.size)
                    x1, y1 = int(p.x + p.size), int(p.y + p.size)
                    draw.ellipse([x0, y0, x1, y1], fill=c)

            particles = alive

            # Composite
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
            composite = Image.alpha_composite(pil_frame, overlay)
            result = cv2.cvtColor(np.array(composite.convert("RGB")), cv2.COLOR_RGB2BGR)
            writer.write(result)

            frame_idx += 1
            if on_progress and frame_idx % 60 == 0:
                pct = 5 + int((frame_idx / total_frames) * 85)
                on_progress(pct, f"Rendering frame {frame_idx}/{total_frames}...")

    finally:
        cap.release()
        writer.release()

    if on_progress:
        on_progress(92, "Encoding with audio...")

    _run_ffmpeg([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", tmp_video, "-i", video_path,
        "-map", "0:v", "-map", "1:a?",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-shortest", output_path,
    ])
    os.unlink(tmp_video)

    if on_progress:
        on_progress(100, f"Particles ({preset}) rendered!")
    return output_path


def get_particle_presets() -> List[Dict]:
    return [
        {"name": k, "label": v["label"], "description": v["description"]}
        for k, v in PARTICLE_PRESETS.items()
    ]
