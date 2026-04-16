"""
OpenCut Route Blueprints

All Flask route handlers organized by domain.
"""


def register_blueprints(app):
    """Register all route Blueprints with the Flask app."""
    from .ai_content_routes import ai_content_bp
    from .ai_editing_routes import ai_editing_bp
    from .ai_intelligence_routes import ai_intel_bp
    from .analysis_routes import analysis_bp
    from .architecture_routes import architecture_bp
    from .audio import audio_bp
    from .audio_advanced_routes import audio_adv_bp
    from .audio_expansion_routes import audio_expand_bp
    from .audio_post_routes import audio_post_bp
    from .audio_production_routes import audio_prod_bp
    from .batch_data_routes import batch_data_bp
    from .body_transfer_routes import body_transfer_bp
    from .captions import captions_bp
    from .cloud_distrib_routes import cloud_distrib_bp
    from .collab_review_routes import collab_review_bp
    from .color_mam_routes import color_mam_bp
    from .composition_dubbing_routes import composition_dubbing_bp
    from .content_gen_routes import content_gen_bp
    from .content_routes import content_bp
    from .context import context_bp
    from .creative_routes import creative_bp
    from .deliverables import deliverables_bp
    from .delivery_master_routes import delivery_master_bp
    from .delivery_routes import delivery_bp
    from .dev_scripting_routes import dev_scripting_bp
    from .documentary_routes import documentary_bp
    from .editing_workflow_routes import editing_wf_bp
    from .education_routes import education_bp
    from .encoding_routes import encoding_bp
    from .engagement_content_routes import engagement_content_bp
    from .enhanced_media_routes import enhanced_media_bp
    from .format_routes import format_bp
    from .gaming_routes import gaming_bp
    from .generative_routes import generative_bp
    from .hw_routes import hw_bp
    from .infrastructure_routes import infra_bp
    from .integration_routes import integration_bp
    from .jobs_routes import jobs_bp
    from .journal import journal_bp
    from .motion_design_routes import motion_design_bp
    from .motion_gen_routes import motion_gen_bp
    from .multiview_repurpose_routes import multiview_repurpose_bp
    from .music_safety_routes import music_safety_bp
    from .next_gen_ai_routes import next_gen_ai_bp
    from .nlp import nlp_bp
    from .object_intel_routes import object_intel_bp
    from .overlay_routes import overlay_bp
    from .pipeline_intel_routes import pipeline_intel_bp
    from .platform_infra_routes import platform_infra_bp
    from .platform_ux_routes import platform_ux_bp
    from .plugins import plugins_bp
    from .preproduction_proxy_routes import preproduction_proxy_bp
    from .preview_realtime_routes import preview_realtime_bp
    from .privacy_spectral_routes import privacy_spectral_bp
    from .processing_routes import processing_bp
    from .production_routes import production_bp
    from .professional_routes import professional_bp
    from .qc_routes import qc_bp
    from .remote_realtime_routes import remote_realtime_bp
    from .repair_gen_routes import repair_gen_bp
    from .search import search_bp
    from .settings import settings_bp
    from .solver_agent_routes import solver_agent_bp
    from .sound_music_routes import sound_music_bp
    from .subtitle_pro_routes import subtitle_pro_bp
    from .subtitle_routes import subtitle_bp
    from .system import system_bp
    from .timeline import timeline_bp
    from .timeline_auto_routes import timeline_auto_bp
    from .timeline_intel_routes import timeline_intel_bp
    from .tools_routes import tools_bp
    from .transcript_edit_routes import transcript_edit_bp
    from .utility_routes import utility_bp
    from .ux_intelligence_routes import ux_intel_bp
    from .vfx_advanced_routes import vfx_advanced_bp
    from .video_ai import video_ai_bp
    from .video_core import video_core_bp
    from .video_editing import video_editing_bp
    from .video_effects_routes import video_effects_bp
    from .video_fx import video_fx_bp
    from .video_processing_routes import video_proc_bp
    from .video_specialty import video_specialty_bp
    from .video_vfx_routes import video_vfx_bp
    from .voice_speech_routes import voice_speech_bp
    from .vr_lens_routes import vr_lens_bp
    from .workflow import workflow_bp
    from .workflow_dev_routes import workflow_dev_bp
    from .workflow_routes import workflow_auto_bp

    blueprints = [system_bp, audio_bp, captions_bp,
                  video_core_bp, video_fx_bp, video_ai_bp,
                  video_editing_bp, video_specialty_bp,
                  jobs_bp, settings_bp,
                  timeline_bp, search_bp, deliverables_bp, nlp_bp, workflow_bp,
                  context_bp, plugins_bp, journal_bp, overlay_bp, qc_bp,
                  format_bp, processing_bp, content_bp, hw_bp, creative_bp,
                  subtitle_bp, encoding_bp, utility_bp, production_bp,
                  analysis_bp, workflow_auto_bp, video_proc_bp, audio_prod_bp,
                  ai_content_bp, professional_bp, tools_bp, workflow_dev_bp,
                  infra_bp, video_effects_bp, audio_expand_bp, ai_intel_bp,
                  delivery_bp, gaming_bp, education_bp, documentary_bp,
                  batch_data_bp, music_safety_bp, integration_bp,
                  video_vfx_bp, audio_adv_bp, ai_editing_bp,
                  editing_wf_bp, color_mam_bp, platform_infra_bp,
                  transcript_edit_bp, generative_bp, architecture_bp,
                  remote_realtime_bp, vfx_advanced_bp, solver_agent_bp,
                  platform_ux_bp, vr_lens_bp, repair_gen_bp,
                  privacy_spectral_bp, multiview_repurpose_bp,
                  preproduction_proxy_bp, composition_dubbing_bp,
                  enhanced_media_bp, ux_intel_bp, engagement_content_bp,
                  next_gen_ai_bp, motion_gen_bp, body_transfer_bp,
                  timeline_intel_bp, pipeline_intel_bp,
                  object_intel_bp, delivery_master_bp,
                  content_gen_bp,
                  collab_review_bp, timeline_auto_bp,
                  sound_music_bp, preview_realtime_bp,
                  cloud_distrib_bp,
                  voice_speech_bp, motion_design_bp,
                  subtitle_pro_bp, dev_scripting_bp,
                  audio_post_bp]

    for bp in blueprints:
        app.register_blueprint(bp)

    # Preserve the long-standing /api/motion/* surface alongside the current
    # bare /motion/* routes so older panel builds and tests do not break.
    app.register_blueprint(
        motion_design_bp,
        url_prefix="/api",
        name="motion_design_api",
    )
