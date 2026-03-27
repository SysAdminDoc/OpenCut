"""
Chat-Driven Editing Assistant

Multi-turn conversational agent for video editing commands.
Uses an LLM to understand editing intent, maintain context across
messages, and execute editing actions.

The agent:
1. Understands natural language editing requests
2. Maintains conversation history (what was done, current clip state)
3. Plans multi-step edits from a single request
4. Confirms destructive actions before executing
5. Reports results conversationally

Example conversation:
    User: "Clean up this interview"
    Agent: "I'll remove silences and filler words. Starting with silence detection..."
    Agent: "Found 23 silent segments (42s total). Removing... Done."
    Agent: "Now detecting filler words..."
    Agent: "Removed 8 fillers (um, uh, like). Saved to interview_clean.mp4."
    User: "Also normalize the audio to -14 LUFS"
    Agent: "Normalizing... Current loudness: -22.3 LUFS. Target: -14 LUFS. Done."

Uses: opencut.core.llm for LLM calls, opencut.core.nlp_command for action mapping.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("opencut")

# System prompt for the editing assistant
_EDITOR_SYSTEM_PROMPT = """You are OpenCut, an AI video editing assistant built into Adobe Premiere Pro.
You help editors with tasks like removing silence, adding captions, denoising audio, and applying effects.

When the user asks you to do something:
1. Acknowledge what they want
2. Plan the steps needed
3. For each step, output a JSON action block: {"action": "<route>", "params": {...}}
4. Explain what you're doing in plain English

Available actions (routes):
- /silence — Remove silent segments (params: threshold, min_duration, method)
- /fillers — Remove filler words (params: filler_backend, custom_words)
- /audio/denoise — Reduce noise (params: method, strength)
- /audio/normalize — Normalize loudness (params: preset)
- /audio/separate — Separate stems (params: model, stems)
- /audio/enhance — Enhance speech quality (params: denoise, enhance)
- /styled-captions — Generate styled captions (params: style, model)
- /captions — Generate subtitles (params: model, format, language)
- /video/scenes — Detect scene changes
- /video/fx/apply — Apply video effect (params: effect, params)
- /video/reframe — Reframe for social (params: width, height, mode)
- /video/speed/change — Change speed (params: speed)
- /video/depth/bokeh — Add depth-of-field effect (params: focus_point, blur_strength)
- /export/preset — Export with platform preset (params: preset)

Current context:
- Clip: {filepath}
- Duration: {duration}s
- Has video: {has_video}
- Has audio: {has_audio}

Respond conversationally. Include JSON action blocks wrapped in ```action markers when you want to execute something. Only include actions you're confident about."""


@dataclass
class ChatMessage:
    """A single message in the conversation."""
    role: str        # "user", "assistant", or "system"
    content: str
    timestamp: float = 0.0
    actions: List[Dict] = field(default_factory=list)  # Parsed action blocks


@dataclass
class ChatSession:
    """A chat editing session with conversation history."""
    session_id: str
    filepath: str = ""
    history: List[ChatMessage] = field(default_factory=list)
    context: Dict = field(default_factory=dict)  # Current clip metadata
    max_history: int = 20  # Keep last N messages for context window

    def add_message(self, role: str, content: str, actions: Optional[List[Dict]] = None):
        self.history.append(ChatMessage(
            role=role,
            content=content,
            timestamp=time.time(),
            actions=actions or [],
        ))
        # Trim history to keep context window manageable
        if len(self.history) > self.max_history:
            # Keep system message + last N messages
            system_msgs = [m for m in self.history if m.role == "system"]
            recent = self.history[-self.max_history:]
            self.history = system_msgs + [m for m in recent if m.role != "system"]

    def get_llm_messages(self) -> List[Dict]:
        """Format history for LLM API call."""
        return [{"role": m.role, "content": m.content} for m in self.history]


# Active sessions (in-memory, keyed by session_id)
_sessions: Dict[str, ChatSession] = {}

# Session TTL: evict sessions inactive for more than 2 hours
_SESSION_TTL = 2 * 60 * 60  # seconds
_MAX_SESSIONS = 100


def _evict_stale_sessions():
    """Remove sessions that haven't been active within the TTL."""
    now = time.time()
    stale = [
        sid for sid, s in _sessions.items()
        if s.history and (now - s.history[-1].timestamp) > _SESSION_TTL
    ]
    for sid in stale:
        del _sessions[sid]
    # Hard cap: if still too many, remove oldest
    if len(_sessions) > _MAX_SESSIONS:
        by_age = sorted(
            _sessions.items(),
            key=lambda kv: kv[1].history[-1].timestamp if kv[1].history else 0,
        )
        for sid, _ in by_age[: len(_sessions) - _MAX_SESSIONS]:
            del _sessions[sid]


def get_or_create_session(
    session_id: str,
    filepath: str = "",
    clip_info: Optional[Dict] = None,
) -> ChatSession:
    """Get an existing chat session or create a new one."""
    _evict_stale_sessions()

    if session_id in _sessions:
        session = _sessions[session_id]
        # Update filepath if changed
        if filepath and filepath != session.filepath:
            session.filepath = filepath
            session.context = clip_info or {}
        return session

    session = ChatSession(
        session_id=session_id,
        filepath=filepath,
        context=clip_info or {},
    )
    _sessions[session_id] = session
    return session


def clear_session(session_id: str):
    """Clear a chat session's history."""
    _sessions.pop(session_id, None)


def chat(
    session_id: str,
    user_message: str,
    filepath: str = "",
    clip_info: Optional[Dict] = None,
    llm_config=None,
) -> Dict:
    """
    Process a chat message and return the assistant's response.

    Args:
        session_id: Unique session identifier.
        user_message: The user's message.
        filepath: Current clip path (for context).
        clip_info: Clip metadata dict (duration, has_video, has_audio).
        llm_config: LLM configuration.

    Returns:
        Dict with: response (str), actions (list of action dicts), session_id.
    """
    from opencut.core.llm import LLMConfig, query_llm

    if llm_config is None:
        llm_config = LLMConfig()

    session = get_or_create_session(session_id, filepath, clip_info)

    # Build system prompt with current context
    ctx = session.context or {}
    system_prompt = _EDITOR_SYSTEM_PROMPT.format(
        filepath=filepath or "none selected",
        duration=ctx.get("duration", "unknown"),
        has_video=ctx.get("has_video", "unknown"),
        has_audio=ctx.get("has_audio", "unknown"),
    )

    # Add system prompt if not already present
    if not session.history or session.history[0].role != "system":
        session.history.insert(0, ChatMessage(role="system", content=system_prompt))
    else:
        # Update system prompt with latest context
        session.history[0].content = system_prompt

    # Add user message
    session.add_message("user", user_message)

    # Query LLM with full conversation history
    messages = session.get_llm_messages()

    try:
        response = query_llm(
            prompt=user_message,
            config=llm_config,
            system_prompt=system_prompt,
            messages=messages[1:],  # Skip system (passed separately)
        )
        response_text = response.text if hasattr(response, "text") else str(response)
    except Exception as e:
        logger.error("LLM query failed in chat_editor: %s", e)
        response_text = f"Sorry, I couldn't process that request. The AI backend returned an error: {e}"

    # Parse action blocks from response
    actions = _parse_actions(response_text)

    # Add assistant response to history
    session.add_message("assistant", response_text, actions=actions)

    return {
        "response": response_text,
        "actions": actions,
        "session_id": session_id,
        "history_length": len(session.history),
    }


def _parse_actions(text: str) -> List[Dict]:
    """Parse JSON action blocks from assistant response.

    Looks for ```action or ```json blocks containing {"action": "/route", "params": {...}}.
    """
    import json
    import re

    actions = []

    # Match code blocks with action/json language tag
    pattern = r'```(?:action|json)\s*\n(.*?)\n```'
    matches = re.findall(pattern, text, re.DOTALL)

    for match in matches:
        try:
            parsed = json.loads(match.strip())
            if isinstance(parsed, dict) and "action" in parsed:
                actions.append(parsed)
            elif isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and "action" in item:
                        actions.append(item)
        except json.JSONDecodeError:
            continue

    # Also try inline JSON patterns: {"action": "/route", ...}
    if not actions:
        inline_pattern = r'\{["\']action["\']\s*:\s*["\']\/[^"\']+["\'][^}]*\}'
        for match in re.finditer(inline_pattern, text):
            try:
                parsed = json.loads(match.group())
                if "action" in parsed:
                    actions.append(parsed)
            except json.JSONDecodeError:
                continue

    return actions


def list_sessions() -> List[Dict]:
    """List all active chat sessions."""
    return [
        {
            "session_id": sid,
            "filepath": s.filepath,
            "message_count": len(s.history),
            "last_activity": s.history[-1].timestamp if s.history else 0,
        }
        for sid, s in _sessions.items()
    ]
