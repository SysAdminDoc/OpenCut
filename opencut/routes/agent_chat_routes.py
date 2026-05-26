"""
Routes for the F143 chat-conductor scaffold (RESEARCH_FEATURE_PLAN_2026-05-25 Q2).

This blueprint exposes the backend conductor surface. The full UXP UI
(editable plan + thumb-strip diff + per-step accept) lands under F252
and consumes these endpoints.

Routes:
  POST /agent/chat/plan         — intent → plan (LLM or keyword fallback)
  POST /agent/chat/step-status  — record execution status for a plan step
  POST /agent/chat/review       — self-review the executed plan vs intent
  GET  /agent/chat/session      — replay an existing session JSON
  GET  /agent/chat/info         — capability + system-prompt fingerprint
"""
from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request

from opencut.errors import safe_error
from opencut.security import require_csrf

logger = logging.getLogger("opencut")
agent_chat_bp = Blueprint("agent_chat", __name__)


def _build_llm_config(payload: dict):
    """Optional LLMConfig builder. Returns None when no llm block present."""
    if not isinstance(payload, dict):
        return None
    llm = payload.get("llm")
    if not isinstance(llm, dict):
        return None
    provider = str(llm.get("provider") or "").strip()
    if provider not in ("ollama", "openai", "anthropic", "gemini"):
        return None
    try:
        from opencut.core.llm import LLMConfig
    except Exception:
        return None
    return LLMConfig(
        provider=provider,
        model=str(llm.get("model") or ""),
        api_key=str(llm.get("api_key") or ""),
        base_url=str(llm.get("base_url") or ""),
    )


@agent_chat_bp.route("/agent/chat/plan", methods=["POST"])
@require_csrf
def route_agent_chat_plan():
    """Generate a plan from a natural-language ``intent``.

    Body params:
      intent      str   the user request (required)
      session_id  str   optional; mints a fresh session when absent
      llm         dict  optional ``{provider, model, api_key, base_url}``
    """
    try:
        from opencut.core import agent_chat

        data = request.get_json(silent=True) or {}
        intent = str(data.get("intent") or "").strip()
        if not intent:
            raise ValueError("'intent' is required")
        session_id = str(data.get("session_id") or "").strip() or None
        llm_config = _build_llm_config(data)
        result = agent_chat.plan(intent, llm_config=llm_config, session_id=session_id)
        return jsonify({k: result[k] for k in result.keys()})
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover
        return safe_error(exc, "agent_chat_plan")


@agent_chat_bp.route("/agent/chat/step-status", methods=["POST"])
@require_csrf
def route_agent_chat_step_status():
    """Record execution status for a plan step.

    Body params:
      session_id  str   required
      step_id     str   required
      status      str   one of planned|executing|ok|failed|rejected
      job_id      str   optional (when the step ran via @async_job)
      reason      str   optional (when status is failed/rejected)
    """
    try:
        from opencut.core import agent_chat

        data = request.get_json(silent=True) or {}
        session_id = str(data.get("session_id") or "").strip()
        step_id = str(data.get("step_id") or "").strip()
        status = str(data.get("status") or "").strip()
        if not (session_id and step_id and status):
            raise ValueError("session_id, step_id, status are required")
        sess = agent_chat.mark_step_status(
            session_id=session_id,
            step_id=step_id,
            status=status,
            job_id=str(data.get("job_id") or ""),
            reason=str(data.get("reason") or ""),
        )
        return jsonify({"ok": True, "session": sess})
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover
        return safe_error(exc, "agent_chat_step_status")


@agent_chat_bp.route("/agent/chat/review", methods=["POST"])
@require_csrf
def route_agent_chat_review():
    """Self-review the executed plan vs the original intent.

    Body params:
      session_id  str   required
      llm         dict  optional — when supplied, model writes drift notes;
                       absent, the heuristic reports failed/rejected steps.
    """
    try:
        from opencut.core import agent_chat

        data = request.get_json(silent=True) or {}
        session_id = str(data.get("session_id") or "").strip()
        if not session_id:
            raise ValueError("'session_id' is required")
        llm_config = _build_llm_config(data)
        result = agent_chat.review(session_id, llm_config=llm_config)
        return jsonify({k: result[k] for k in result.keys()})
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover
        return safe_error(exc, "agent_chat_review")


@agent_chat_bp.route("/agent/chat/session", methods=["GET"])
def route_agent_chat_session():
    """Return the persisted session JSON. Read-only — no auth required.

    Query params:
      session_id  str   required
    """
    try:
        from opencut.core import agent_chat
        session_id = str(request.args.get("session_id") or "").strip()
        if not session_id:
            return jsonify({"error": "session_id query param required"}), 400
        sess = agent_chat.load_session(session_id)
        if sess is None:
            return jsonify({"error": "session not found"}), 404
        return jsonify(sess)
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover
        return safe_error(exc, "agent_chat_session")


@agent_chat_bp.route("/agent/chat/info", methods=["GET"])
def route_agent_chat_info():
    """Capability report + system-prompt fingerprints (UI rendering hint)."""
    try:
        from opencut.core import agent_chat
        return jsonify({
            "available": agent_chat.check_agent_chat_available(),
            "modes": ["plan", "step-status", "review", "session"],
            "system_prompt_plan": agent_chat.SYSTEM_PROMPT_PLAN,
            "install_hint": agent_chat.INSTALL_HINT,
        })
    except Exception as exc:  # pragma: no cover
        return safe_error(exc, "agent_chat_info")
