"""
OpenCut Route Blueprints

All Flask route handlers organized by domain.
"""


def register_blueprints(app):
    """Register all route Blueprints with the Flask app."""
    from .system import system_bp
    from .audio import audio_bp
    from .captions import captions_bp
    from .video import video_bp
    from .jobs_routes import jobs_bp
    from .settings import settings_bp

    for bp in [system_bp, audio_bp, captions_bp, video_bp, jobs_bp, settings_bp]:
        app.register_blueprint(bp)
