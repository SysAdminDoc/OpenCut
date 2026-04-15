"""
OpenCut Particle System (Category 79 - Motion Design)

Configurable particle emitter for video overlays. Supports multiple emitter
types, sprite shapes, physics simulation (gravity, wind, turbulence), and
blend modes. Presets for common effects like snow, rain, confetti, fire.

Functions:
    render_particles  - Render particle overlay to video
    preview_frame     - Render a single preview frame
    list_presets      - List available particle presets
    list_emitter_types - List emitter types
    list_sprite_types  - List sprite types
"""

import logging
import math
import os
import random
import tempfile
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Result Dataclass
# ---------------------------------------------------------------------------


@dataclass
class ParticleResult:
    """Result of a particle system render."""

    output_path: str = ""
    frames_rendered: int = 0
    total_particles_spawned: int = 0
    peak_active_particles: int = 0
    duration: float = 0.0
    fps: int = 30

    def to_dict(self) -> dict:
        return {
            "output_path": self.output_path,
            "frames_rendered": self.frames_rendered,
            "total_particles_spawned": self.total_particles_spawned,
            "peak_active_particles": self.peak_active_particles,
            "duration": self.duration,
            "fps": self.fps,
        }


# ---------------------------------------------------------------------------
# Emitter Types
# ---------------------------------------------------------------------------

EMITTER_TYPES = ["point", "line", "circle", "rectangle", "burst"]

SPRITE_TYPES = [
    "circle", "square", "star", "snowflake", "spark",
    "confetti", "smoke", "custom_image",
]


def list_emitter_types() -> List[dict]:
    """List available emitter types."""
    descs = {
        "point": "Emit from a single point",
        "line": "Emit along a line segment",
        "circle": "Emit from points on a circle",
        "rectangle": "Emit from random points within a rectangle",
        "burst": "Emit all particles at once in a burst",
    }
    return [{"type": t, "description": descs.get(t, "")} for t in EMITTER_TYPES]


def list_sprite_types() -> List[dict]:
    """List available sprite types."""
    descs = {
        "circle": "Filled circle",
        "square": "Filled square",
        "star": "5-pointed star",
        "snowflake": "6-armed snowflake pattern",
        "spark": "Bright elongated spark",
        "confetti": "Colorful rectangle confetti",
        "smoke": "Soft smoke puff",
        "custom_image": "Custom image sprite from file",
    }
    return [{"type": t, "description": descs.get(t, "")} for t in SPRITE_TYPES]


# ---------------------------------------------------------------------------
# Noise for Turbulence
# ---------------------------------------------------------------------------


def _noise_hash(x: int, seed: int = 0) -> float:
    """Hash for noise generation."""
    n = x + seed * 131
    n = (n << 13) ^ n
    n = n * (n * n * 15731 + 789221) + 1376312589
    return (n & 0x7FFFFFFF) / 0x7FFFFFFF


def _noise_2d(x: float, y: float, seed: int = 0) -> float:
    """2D value noise, returns -1 to 1."""
    ix = int(math.floor(x))
    iy = int(math.floor(y))
    fx = x - ix
    fy = y - iy
    fx = fx * fx * (3.0 - 2.0 * fx)
    fy = fy * fy * (3.0 - 2.0 * fy)

    n00 = _noise_hash(ix + iy * 57, seed) * 2.0 - 1.0
    n10 = _noise_hash(ix + 1 + iy * 57, seed) * 2.0 - 1.0
    n01 = _noise_hash(ix + (iy + 1) * 57, seed) * 2.0 - 1.0
    n11 = _noise_hash(ix + 1 + (iy + 1) * 57, seed) * 2.0 - 1.0

    nx0 = n00 + (n10 - n00) * fx
    nx1 = n01 + (n11 - n01) * fx
    return nx0 + (nx1 - nx0) * fy


# ---------------------------------------------------------------------------
# Particle Data
# ---------------------------------------------------------------------------


class Particle:
    """Single particle state."""

    __slots__ = (
        "x", "y", "vx", "vy", "ax", "ay",
        "lifetime", "age", "size", "size_start", "size_end",
        "color", "opacity", "opacity_start", "opacity_end",
        "rotation", "rotation_speed",
        "sprite_type", "alive",
    )

    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.ax = 0.0
        self.ay = 0.0
        self.lifetime = 1.0
        self.age = 0.0
        self.size = 5.0
        self.size_start = 5.0
        self.size_end = 5.0
        self.color = (255, 255, 255)
        self.opacity = 1.0
        self.opacity_start = 1.0
        self.opacity_end = 0.0
        self.rotation = 0.0
        self.rotation_speed = 0.0
        self.sprite_type = "circle"
        self.alive = True

    def update(self, dt: float, gravity: float, wind_x: float, wind_y: float,
               turbulence: float, seed: int,
               bounce_bounds: Optional[Tuple[int, int, int, int]]):
        """Advance particle by dt seconds."""
        if not self.alive:
            return

        self.age += dt
        if self.age >= self.lifetime:
            self.alive = False
            return

        t = self.age / max(self.lifetime, 0.001)

        # Turbulence via noise
        turb_x = 0.0
        turb_y = 0.0
        if turbulence > 0:
            turb_x = _noise_2d(self.x * 0.01, self.age * 0.5, seed) * turbulence
            turb_y = _noise_2d(self.y * 0.01, self.age * 0.5, seed + 99) * turbulence

        # Apply forces
        self.vx += (self.ax + wind_x + turb_x) * dt
        self.vy += (self.ay + gravity + wind_y + turb_y) * dt

        self.x += self.vx * dt
        self.y += self.vy * dt

        # Interpolate size and opacity
        self.size = self.size_start + (self.size_end - self.size_start) * t
        self.opacity = self.opacity_start + (self.opacity_end - self.opacity_start) * t
        self.rotation += self.rotation_speed * dt

        # Bounce off bounds
        if bounce_bounds:
            bx1, by1, bx2, by2 = bounce_bounds
            if self.x < bx1:
                self.x = bx1
                self.vx = abs(self.vx) * 0.6
            elif self.x > bx2:
                self.x = bx2
                self.vx = -abs(self.vx) * 0.6
            if self.y < by1:
                self.y = by1
                self.vy = abs(self.vy) * 0.6
            elif self.y > by2:
                self.y = by2
                self.vy = -abs(self.vy) * 0.6


# ---------------------------------------------------------------------------
# Emitter
# ---------------------------------------------------------------------------


class ParticleEmitter:
    """Configurable particle emitter."""

    def __init__(
        self,
        emitter_type: str = "point",
        position: Tuple[float, float] = (960, 540),
        emit_rate: float = 30.0,
        burst_count: int = 50,
        # Emitter geometry
        line_end: Tuple[float, float] = (960, 540),
        circle_radius: float = 100.0,
        rect_size: Tuple[float, float] = (200, 100),
        # Particle properties
        velocity_range: Tuple[float, float] = (50, 150),
        angle_range: Tuple[float, float] = (0, 360),
        lifetime_range: Tuple[float, float] = (1.0, 3.0),
        size_range: Tuple[float, float] = (3, 8),
        size_end_range: Tuple[float, float] = (0, 2),
        colors: Optional[List[Tuple[int, int, int]]] = None,
        opacity_start: float = 1.0,
        opacity_end: float = 0.0,
        rotation_speed_range: Tuple[float, float] = (0, 0),
        sprite_type: str = "circle",
        # Physics
        gravity: float = 0.0,
        wind: Tuple[float, float] = (0, 0),
        turbulence: float = 0.0,
        bounce: bool = False,
        # Limits
        max_particles: int = 1000,
        seed: int = 42,
    ):
        self.emitter_type = emitter_type
        self.position = position
        self.emit_rate = emit_rate
        self.burst_count = burst_count
        self.line_end = line_end
        self.circle_radius = circle_radius
        self.rect_size = rect_size
        self.velocity_range = velocity_range
        self.angle_range = angle_range
        self.lifetime_range = lifetime_range
        self.size_range = size_range
        self.size_end_range = size_end_range
        self.colors = colors or [(255, 255, 255)]
        self.opacity_start = opacity_start
        self.opacity_end = opacity_end
        self.rotation_speed_range = rotation_speed_range
        self.sprite_type = sprite_type
        self.gravity = gravity
        self.wind = wind
        self.turbulence = turbulence
        self.bounce = bounce
        self.max_particles = max_particles
        self.seed = seed

        self.rng = random.Random(seed)
        self.particles: List[Particle] = []
        self.emit_accumulator = 0.0
        self.total_spawned = 0
        self.peak_active = 0
        self._burst_done = False

    def _spawn_position(self) -> Tuple[float, float]:
        """Get spawn position based on emitter type."""
        px, py = self.position
        if self.emitter_type == "point" or self.emitter_type == "burst":
            return (px, py)
        elif self.emitter_type == "line":
            t = self.rng.random()
            lx, ly = self.line_end
            return (px + (lx - px) * t, py + (ly - py) * t)
        elif self.emitter_type == "circle":
            angle = self.rng.uniform(0, 2 * math.pi)
            r = self.circle_radius * math.sqrt(self.rng.random())
            return (px + r * math.cos(angle), py + r * math.sin(angle))
        elif self.emitter_type == "rectangle":
            rw, rh = self.rect_size
            return (px + self.rng.uniform(-rw / 2, rw / 2),
                    py + self.rng.uniform(-rh / 2, rh / 2))
        return (px, py)

    def _create_particle(self) -> Particle:
        """Create a new particle with randomized properties."""
        p = Particle()
        p.x, p.y = self._spawn_position()

        speed = self.rng.uniform(*self.velocity_range)
        angle_deg = self.rng.uniform(*self.angle_range)
        angle_rad = math.radians(angle_deg)
        p.vx = speed * math.cos(angle_rad)
        p.vy = speed * math.sin(angle_rad)

        p.lifetime = self.rng.uniform(*self.lifetime_range)
        p.size_start = self.rng.uniform(*self.size_range)
        p.size_end = self.rng.uniform(*self.size_end_range)
        p.size = p.size_start
        p.color = self.rng.choice(self.colors)
        p.opacity_start = self.opacity_start
        p.opacity_end = self.opacity_end
        p.opacity = self.opacity_start
        p.rotation_speed = self.rng.uniform(*self.rotation_speed_range)
        p.rotation = self.rng.uniform(0, 360)
        p.sprite_type = self.sprite_type
        p.alive = True
        p.age = 0.0

        self.total_spawned += 1
        return p

    def update(self, dt: float, resolution: Tuple[int, int]):
        """Update emitter: spawn new particles and update existing."""
        # Spawn new particles
        if self.emitter_type == "burst":
            if not self._burst_done:
                for _ in range(self.burst_count):
                    if len(self.particles) < self.max_particles:
                        self.particles.append(self._create_particle())
                self._burst_done = True
        else:
            self.emit_accumulator += self.emit_rate * dt
            while self.emit_accumulator >= 1.0:
                if len(self.particles) < self.max_particles:
                    self.particles.append(self._create_particle())
                self.emit_accumulator -= 1.0

        # Update particles
        bounce_bounds = None
        if self.bounce:
            bounce_bounds = (0, 0, resolution[0], resolution[1])

        for p in self.particles:
            p.update(dt, self.gravity, self.wind[0], self.wind[1],
                     self.turbulence, self.seed, bounce_bounds)

        # Remove dead particles
        self.particles = [p for p in self.particles if p.alive]

        # Track peak
        active = len(self.particles)
        if active > self.peak_active:
            self.peak_active = active


# ---------------------------------------------------------------------------
# Particle Presets
# ---------------------------------------------------------------------------

PARTICLE_PRESETS = {
    "snow": {
        "description": "Gentle falling snowflakes",
        "emitter_type": "line",
        "position": (960, -20),
        "line_end": (1920, -20),
        "emit_rate": 25,
        "velocity_range": (20, 60),
        "angle_range": (80, 100),
        "lifetime_range": (4.0, 8.0),
        "size_range": (3, 8),
        "size_end_range": (2, 6),
        "colors": [(255, 255, 255), (220, 230, 255), (240, 248, 255)],
        "sprite_type": "snowflake",
        "gravity": 15,
        "wind": (10, 0),
        "turbulence": 30,
        "opacity_end": 0.3,
    },
    "rain": {
        "description": "Rainfall with streaks",
        "emitter_type": "line",
        "position": (0, -30),
        "line_end": (1920, -30),
        "emit_rate": 80,
        "velocity_range": (300, 500),
        "angle_range": (85, 95),
        "lifetime_range": (0.5, 1.5),
        "size_range": (1, 3),
        "size_end_range": (1, 2),
        "colors": [(180, 200, 220), (160, 180, 210)],
        "sprite_type": "spark",
        "gravity": 400,
        "wind": (20, 0),
        "turbulence": 5,
        "opacity_end": 0.2,
    },
    "confetti": {
        "description": "Colorful celebration confetti",
        "emitter_type": "rectangle",
        "position": (960, 200),
        "rect_size": (800, 50),
        "emit_rate": 40,
        "velocity_range": (30, 120),
        "angle_range": (60, 120),
        "lifetime_range": (3.0, 6.0),
        "size_range": (4, 10),
        "size_end_range": (3, 8),
        "colors": [
            (255, 87, 87), (255, 189, 46), (46, 213, 115),
            (69, 170, 242), (165, 94, 234), (255, 159, 243),
        ],
        "sprite_type": "confetti",
        "gravity": 50,
        "wind": (5, 0),
        "turbulence": 20,
        "rotation_speed_range": (-180, 180),
        "opacity_end": 0.5,
    },
    "fire_sparks": {
        "description": "Rising fire sparks and embers",
        "emitter_type": "line",
        "position": (860, 1080),
        "line_end": (1060, 1080),
        "emit_rate": 45,
        "velocity_range": (80, 200),
        "angle_range": (250, 290),
        "lifetime_range": (1.0, 3.0),
        "size_range": (2, 5),
        "size_end_range": (0, 1),
        "colors": [
            (255, 160, 20), (255, 100, 10), (255, 200, 50),
            (255, 80, 0),
        ],
        "sprite_type": "spark",
        "gravity": -30,
        "wind": (15, 0),
        "turbulence": 40,
        "opacity_end": 0.0,
    },
    "dust": {
        "description": "Floating dust motes",
        "emitter_type": "rectangle",
        "position": (960, 540),
        "rect_size": (1920, 1080),
        "emit_rate": 10,
        "velocity_range": (5, 20),
        "angle_range": (0, 360),
        "lifetime_range": (5.0, 10.0),
        "size_range": (1, 3),
        "size_end_range": (1, 2),
        "colors": [(200, 200, 180), (180, 180, 160)],
        "sprite_type": "circle",
        "gravity": -2,
        "wind": (3, 0),
        "turbulence": 15,
        "opacity_start": 0.3,
        "opacity_end": 0.0,
    },
    "bubbles": {
        "description": "Rising bubbles",
        "emitter_type": "line",
        "position": (200, 1080),
        "line_end": (1720, 1080),
        "emit_rate": 15,
        "velocity_range": (30, 80),
        "angle_range": (260, 280),
        "lifetime_range": (3.0, 7.0),
        "size_range": (5, 15),
        "size_end_range": (8, 20),
        "colors": [(150, 200, 255), (180, 220, 255), (200, 230, 255)],
        "sprite_type": "circle",
        "gravity": -25,
        "wind": (5, 0),
        "turbulence": 20,
        "opacity_start": 0.5,
        "opacity_end": 0.1,
    },
    "magic_sparkle": {
        "description": "Magical sparkle particles",
        "emitter_type": "circle",
        "position": (960, 540),
        "circle_radius": 150,
        "emit_rate": 35,
        "velocity_range": (10, 50),
        "angle_range": (0, 360),
        "lifetime_range": (0.5, 2.0),
        "size_range": (2, 6),
        "size_end_range": (0, 1),
        "colors": [
            (255, 255, 100), (255, 200, 255), (150, 200, 255),
            (255, 255, 255),
        ],
        "sprite_type": "star",
        "gravity": 0,
        "wind": (0, 0),
        "turbulence": 25,
        "opacity_end": 0.0,
    },
    "smoke_rising": {
        "description": "Slowly rising smoke",
        "emitter_type": "line",
        "position": (860, 900),
        "line_end": (1060, 900),
        "emit_rate": 12,
        "velocity_range": (20, 50),
        "angle_range": (260, 280),
        "lifetime_range": (3.0, 6.0),
        "size_range": (10, 25),
        "size_end_range": (30, 60),
        "colors": [(120, 120, 120), (100, 100, 100), (80, 80, 80)],
        "sprite_type": "smoke",
        "gravity": -15,
        "wind": (10, 0),
        "turbulence": 30,
        "opacity_start": 0.4,
        "opacity_end": 0.0,
    },
}


def list_presets() -> List[dict]:
    """Return list of available particle presets."""
    return [
        {"name": name, "description": info.get("description", "")}
        for name, info in PARTICLE_PRESETS.items()
    ]


def _build_emitter_from_preset(preset_name: str,
                                resolution: Tuple[int, int],
                                **overrides) -> ParticleEmitter:
    """Build a ParticleEmitter from a preset name with optional overrides."""
    if preset_name not in PARTICLE_PRESETS:
        raise ValueError(
            f"Unknown preset: {preset_name}. "
            f"Available: {list(PARTICLE_PRESETS.keys())}"
        )
    config = dict(PARTICLE_PRESETS[preset_name])
    config.pop("description", None)
    config.update(overrides)

    return ParticleEmitter(
        emitter_type=config.get("emitter_type", "point"),
        position=tuple(config.get("position", (resolution[0] // 2, resolution[1] // 2))),
        emit_rate=config.get("emit_rate", 30),
        burst_count=config.get("burst_count", 50),
        line_end=tuple(config.get("line_end", (resolution[0], 0))),
        circle_radius=config.get("circle_radius", 100),
        rect_size=tuple(config.get("rect_size", (200, 100))),
        velocity_range=tuple(config.get("velocity_range", (50, 150))),
        angle_range=tuple(config.get("angle_range", (0, 360))),
        lifetime_range=tuple(config.get("lifetime_range", (1.0, 3.0))),
        size_range=tuple(config.get("size_range", (3, 8))),
        size_end_range=tuple(config.get("size_end_range", (0, 2))),
        colors=[tuple(c) for c in config.get("colors", [(255, 255, 255)])],
        opacity_start=config.get("opacity_start", 1.0),
        opacity_end=config.get("opacity_end", 0.0),
        rotation_speed_range=tuple(config.get("rotation_speed_range", (0, 0))),
        sprite_type=config.get("sprite_type", "circle"),
        gravity=config.get("gravity", 0),
        wind=tuple(config.get("wind", (0, 0))),
        turbulence=config.get("turbulence", 0),
        bounce=config.get("bounce", False),
        max_particles=config.get("max_particles", 1000),
        seed=config.get("seed", 42),
    )


# ---------------------------------------------------------------------------
# Sprite Rendering
# ---------------------------------------------------------------------------


def _draw_circle_sprite(draw, x: int, y: int, size: float,
                        color: Tuple[int, int, int], alpha: int):
    """Draw circle sprite."""
    r = max(1, int(size / 2))
    draw.ellipse([x - r, y - r, x + r, y + r],
                 fill=(color[0], color[1], color[2], alpha))


def _draw_square_sprite(draw, x: int, y: int, size: float,
                        color: Tuple[int, int, int], alpha: int):
    """Draw square sprite."""
    r = max(1, int(size / 2))
    draw.rectangle([x - r, y - r, x + r, y + r],
                   fill=(color[0], color[1], color[2], alpha))


def _draw_star_sprite(draw, x: int, y: int, size: float,
                      color: Tuple[int, int, int], alpha: int):
    """Draw 5-pointed star sprite."""
    outer = max(2, int(size / 2))
    inner = max(1, int(outer * 0.4))
    points = []
    for i in range(10):
        angle = math.pi * 2 * i / 10 - math.pi / 2
        r = outer if i % 2 == 0 else inner
        points.append((x + int(r * math.cos(angle)),
                       y + int(r * math.sin(angle))))
    draw.polygon(points, fill=(color[0], color[1], color[2], alpha))


def _draw_snowflake_sprite(draw, x: int, y: int, size: float,
                           color: Tuple[int, int, int], alpha: int):
    """Draw snowflake sprite with 6 arms."""
    r = max(2, int(size / 2))
    fill = (color[0], color[1], color[2], alpha)
    for i in range(6):
        angle = math.pi * 2 * i / 6
        ex = x + int(r * math.cos(angle))
        ey = y + int(r * math.sin(angle))
        draw.line([(x, y), (ex, ey)], fill=fill, width=1)
        # Branch
        br = r // 2
        for bi in (-1, 1):
            ba = angle + bi * 0.5
            bx = x + int(br * math.cos(angle)) + int(br // 2 * math.cos(ba))
            by = y + int(br * math.sin(angle)) + int(br // 2 * math.sin(ba))
            mx = x + int(br * math.cos(angle))
            my = y + int(br * math.sin(angle))
            draw.line([(mx, my), (bx, by)], fill=fill, width=1)


def _draw_spark_sprite(draw, x: int, y: int, size: float,
                       color: Tuple[int, int, int], alpha: int):
    """Draw elongated spark sprite."""
    w = max(1, int(size * 0.3))
    h = max(2, int(size))
    draw.ellipse([x - w, y - h, x + w, y + h],
                 fill=(color[0], color[1], color[2], alpha))


def _draw_confetti_sprite(draw, x: int, y: int, size: float,
                          color: Tuple[int, int, int], alpha: int):
    """Draw confetti rectangle sprite."""
    w = max(1, int(size * 0.6))
    h = max(2, int(size))
    draw.rectangle([x - w, y - h, x + w, y + h],
                   fill=(color[0], color[1], color[2], alpha))


def _draw_smoke_sprite(draw, x: int, y: int, size: float,
                       color: Tuple[int, int, int], alpha: int):
    """Draw soft smoke puff."""
    r = max(2, int(size / 2))
    # Lower alpha for softness
    soft_alpha = max(1, alpha // 3)
    draw.ellipse([x - r, y - r, x + r, y + r],
                 fill=(color[0], color[1], color[2], soft_alpha))
    # Smaller inner puff
    ir = max(1, r // 2)
    draw.ellipse([x - ir, y - ir, x + ir, y + ir],
                 fill=(color[0], color[1], color[2], soft_alpha * 2))


SPRITE_DRAWERS = {
    "circle": _draw_circle_sprite,
    "square": _draw_square_sprite,
    "star": _draw_star_sprite,
    "snowflake": _draw_snowflake_sprite,
    "spark": _draw_spark_sprite,
    "confetti": _draw_confetti_sprite,
    "smoke": _draw_smoke_sprite,
}


# ---------------------------------------------------------------------------
# Frame Rendering
# ---------------------------------------------------------------------------


def _render_particle_frame(particles: List[Particle],
                           resolution: Tuple[int, int],
                           blend_mode: str = "alpha",
                           bg_color: Optional[str] = None):
    """Render all particles to a frame image."""
    from PIL import Image, ImageDraw  # noqa: F821

    if bg_color:
        c = bg_color.lstrip("#")
        if len(c) >= 6:
            rgb = (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))
        else:
            rgb = (0, 0, 0)
        img = Image.new("RGBA", resolution, (*rgb, 255))
    else:
        img = Image.new("RGBA", resolution, (0, 0, 0, 0))

    if blend_mode == "additive":
        # For additive, render to separate layer and add
        layer = Image.new("RGBA", resolution, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
    else:
        draw = ImageDraw.Draw(img)

    for p in particles:
        if not p.alive or p.opacity <= 0.01 or p.size < 0.5:
            continue

        alpha = max(0, min(255, int(p.opacity * 255)))
        px = int(p.x)
        py = int(p.y)

        # Skip offscreen particles
        margin = int(p.size) + 5
        if (px < -margin or px > resolution[0] + margin or
                py < -margin or py > resolution[1] + margin):
            continue

        drawer = SPRITE_DRAWERS.get(p.sprite_type, _draw_circle_sprite)
        drawer(draw, px, py, p.size, p.color, alpha)

    if blend_mode == "additive":
        # Simple additive: alpha_composite works for overlay
        img = Image.alpha_composite(img, layer)

    return img


# ---------------------------------------------------------------------------
# FFmpeg Encode
# ---------------------------------------------------------------------------


def _encode_frames(frame_dir: str, output_path: str,
                   fps: int, resolution: Tuple[int, int]) -> str:
    """Encode PNG frame sequence to video."""
    from opencut.helpers import get_ffmpeg_path, run_ffmpeg

    pattern = os.path.join(frame_dir, "frame_%06d.png")
    cmd = [
        get_ffmpeg_path(), "-y",
        "-framerate", str(fps),
        "-i", pattern,
        "-c:v", "libx264", "-pix_fmt", "yuva420p",
        "-crf", "18", "-preset", "fast",
        output_path,
    ]
    run_ffmpeg(cmd)
    return output_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_particles(
    preset: Optional[str] = None,
    emitter_config: Optional[dict] = None,
    duration: float = 5.0,
    fps: int = 30,
    resolution: Tuple[int, int] = (1920, 1080),
    blend_mode: str = "alpha",
    bg_color: Optional[str] = None,
    output_path: Optional[str] = None,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> ParticleResult:
    """Render particle overlay to video.

    Args:
        preset: Particle preset name (snow, rain, confetti, etc.).
        emitter_config: Custom emitter configuration dict. If provided,
            overrides preset values.
        duration: Duration in seconds.
        fps: Frames per second.
        resolution: Output resolution (width, height).
        blend_mode: 'alpha' (standard) or 'additive' (bright).
        bg_color: Background color hex, or None for transparent.
        output_path: Explicit output path.
        output_dir: Directory for output.
        on_progress: Progress callback taking int percentage.

    Returns:
        ParticleResult with output path and metadata.
    """
    if not preset and not emitter_config:
        raise ValueError("Either preset or emitter_config is required")

    overrides = emitter_config or {}

    if preset:
        emitter = _build_emitter_from_preset(preset, resolution, **overrides)
    else:
        emitter = ParticleEmitter(**overrides)

    total_frames = max(1, int(duration * fps))
    dt = 1.0 / fps

    effective_dir = output_dir or tempfile.gettempdir()
    frame_dir = tempfile.mkdtemp(prefix="opencut_particles_", dir=effective_dir)

    for frame_idx in range(total_frames):
        emitter.update(dt, resolution)

        frame_img = _render_particle_frame(
            emitter.particles, resolution, blend_mode, bg_color
        )
        frame_path = os.path.join(frame_dir, f"frame_{frame_idx:06d}.png")
        frame_img.save(frame_path, "PNG")

        if on_progress and total_frames > 1:
            pct = int((frame_idx + 1) / total_frames * 90)
            on_progress(pct)

    # Encode
    if not output_path:
        label = preset or "custom"
        output_path = os.path.join(effective_dir, f"particles_{label}.mp4")

    if on_progress:
        on_progress(92)

    _encode_frames(frame_dir, output_path, fps, resolution)

    if on_progress:
        on_progress(100)

    return ParticleResult(
        output_path=output_path,
        frames_rendered=total_frames,
        total_particles_spawned=emitter.total_spawned,
        peak_active_particles=emitter.peak_active,
        duration=duration,
        fps=fps,
    )


def preview_frame(
    preset: Optional[str] = None,
    emitter_config: Optional[dict] = None,
    time_s: float = 1.0,
    fps: int = 30,
    resolution: Tuple[int, int] = (1920, 1080),
    blend_mode: str = "alpha",
    bg_color: Optional[str] = None,
    output_path: Optional[str] = None,
) -> str:
    """Render a single preview frame at the given time.

    Returns path to saved PNG.
    """
    if not preset and not emitter_config:
        raise ValueError("Either preset or emitter_config is required")

    overrides = emitter_config or {}
    if preset:
        emitter = _build_emitter_from_preset(preset, resolution, **overrides)
    else:
        emitter = ParticleEmitter(**overrides)

    dt = 1.0 / fps
    target_frame = int(time_s * fps)

    for _ in range(target_frame):
        emitter.update(dt, resolution)

    frame_img = _render_particle_frame(
        emitter.particles, resolution, blend_mode, bg_color
    )

    if not output_path:
        output_path = os.path.join(tempfile.gettempdir(), "particle_preview.png")
    frame_img.save(output_path, "PNG")
    return output_path
