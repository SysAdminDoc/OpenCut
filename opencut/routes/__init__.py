"""
OpenCut Route Blueprints

All Flask route handlers organized by domain.
"""


def register_blueprints(app):
    """Register all route Blueprints with the Flask app."""
    from .audio import audio_bp
    from .captions import captions_bp
    from .context import context_bp
    from .deliverables import deliverables_bp
    from .jobs_routes import jobs_bp
    from .journal import journal_bp
    from .nlp import nlp_bp
    from .plugins import plugins_bp
    from .search import search_bp
    from .settings import settings_bp
    from .system import system_bp
    from .timeline import timeline_bp
    from .video_ai import video_ai_bp
    from .video_core import video_core_bp
    from .video_editing import video_editing_bp
    from .video_fx import video_fx_bp
    from .video_specialty import video_specialty_bp
    from .workflow import workflow_bp

    for bp in [system_bp, audio_bp, captions_bp,
               video_core_bp, video_fx_bp, video_ai_bp,
               video_editing_bp, video_specialty_bp,
               jobs_bp, settings_bp,
               timeline_bp, search_bp, deliverables_bp, nlp_bp, workflow_bp,
               context_bp, plugins_bp, journal_bp]:
        app.register_blueprint(bp)
