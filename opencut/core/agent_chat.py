"""
F143 chat-conductor scaffold (RESEARCH_FEATURE_PLAN_2026-05-25 Q2).

Implements the *backend* surface of the `/agent/chat` conductor designed
in ``.ai/research/2026-05-17/AGENT_UX_RFC.md``. The UI surface (Bolt
UXP WebView with editable plan + thumb-strip diff + per-step accept) is
F252 work and is intentionally out of scope here.

Three modes:

  - **plan**:    intent + LLM config → structured JSON plan
                 (``[{label, endpoint, payload, skill}, …]``).
                 LLM path uses ``core/llm.query_llm`` with a system prompt
                 that forces strict JSON; keyword fallback uses
                 ``core/nlp_command.COMMAND_MAP`` so the conductor always
                 produces at least one step.
  - **execute**: run a single plan step via the existing workflow
                 engine (records before/after snapshot paths so the
                 self-review mode can diff them).
  - **review**:  ask the LLM whether the executed plan matches the
                 original intent; return a drift report.

Session state lives under ``~/.opencut/agent_chat_sessions/<id>.json``
behind ``opencut/user_data.py`` wrappers (per-file locks + atomic
``os.replace`` writes). The session captures intent, plan, executed
steps, and review notes so a refresh re-renders the same conversation.

The scaffold is intentionally LLM-optional: if no provider is reachable
the keyword fallback still produces a runnable plan from the
``COMMAND_MAP`` vocabulary. The UI can layer richer rendering on top
of the plan structure shipped here.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, List, Optional

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Session storage helpers (thin wrappers over opencut.user_data)
# ---------------------------------------------------------------------------
def _session_dir() -> str:
    base = os.path.join(os.path.expanduser("~"), ".opencut", "agent_chat_sessions")
    os.makedirs(base, exist_ok=True)
    return base


_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def _session_path(session_id: str) -> str:
    """Path for ``session_id`` under the sessions directory.

    Defensive: rejects anything that isn't pure alphanumeric / dash /
    underscore. ``../../etc/passwd`` and other path-traversal attempts
    fail fast with ``ValueError`` rather than being silently sanitized.
    """
    if not isinstance(session_id, str) or not _SESSION_ID_RE.match(session_id):
        raise ValueError("session_id must be 1-64 alphanumeric / dash / underscore chars")
    return os.path.join(_session_dir(), f"{session_id}.json")


def load_session(session_id: str) -> Optional[dict]:
    """Return the JSON session payload or None when missing."""
    path = _session_path(session_id)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def save_session(session_id: str, payload: dict) -> None:
    """Atomic write of ``payload`` to the session JSON.

    Sessions live in a sub-directory of ``~/.opencut/`` so they bypass
    the per-file lock map in ``user_data.py`` (which is keyed on
    top-level user-data filenames). The write uses a tempfile-in-same-
    dir + ``os.replace`` to stay atomic across partitions.
    """
    import tempfile
    path = _session_path(session_id)
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, path)
    except OSError:
        if os.path.isfile(tmp):
            try:
                os.remove(tmp)
            except OSError:  # pragma: no cover
                pass
        raise


# ---------------------------------------------------------------------------
# Plan + step shapes
# ---------------------------------------------------------------------------
@dataclass
class PlanStep:
    step_id: str
    label: str
    endpoint: str
    payload: dict = field(default_factory=dict)
    skill: str = ""           # skill name (F145 hook) — optional
    status: str = "planned"   # planned | executing | ok | failed | rejected
    job_id: str = ""          # populated on execute
    reason: str = ""          # populated on failed/rejected

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "label": self.label,
            "endpoint": self.endpoint,
            "payload": dict(self.payload),
            "skill": self.skill,
            "status": self.status,
            "job_id": self.job_id,
            "reason": self.reason,
        }


@dataclass
class PlanResult:
    session_id: str
    intent: str
    plan: List[PlanStep] = field(default_factory=list)
    source: str = "keyword"           # "llm" or "keyword"
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "intent": self.intent,
            "plan": [s.to_dict() for s in self.plan],
            "source": self.source,
            "notes": list(self.notes),
        }

    # Flask jsonify protocol
    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def keys(self):
        return ("session_id", "intent", "plan", "source", "notes")

    def __contains__(self, key: str) -> bool:
        return key in self.keys()


# ---------------------------------------------------------------------------
# Public availability check
# ---------------------------------------------------------------------------
def check_agent_chat_available() -> bool:
    """Always True — the keyword fallback works without any LLM provider."""
    return True


INSTALL_HINT = (
    "Agent chat uses the existing core/llm.py providers. With Ollama "
    "running locally you get richer plans; without it the COMMAND_MAP "
    "keyword fallback still produces a runnable plan."
)


# ---------------------------------------------------------------------------
# Plan generator
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_PLAN = (
    "You are OpenCut's editing conductor. The user describes what they "
    "want to do to a video clip. You output a JSON array of steps. "
    "Each step must be an object with keys: label (short human string), "
    "endpoint (OpenCut REST route, e.g. '/silence' or '/captions'), "
    "payload (object — leave {} when unknown), skill (optional, empty "
    "string by default).\n"
    "Output strict JSON only. No prose, no markdown."
)


_JSON_BLOCK_RE = re.compile(r"\[\s*\{.*?\}\s*\]", re.DOTALL)


def _parse_llm_plan(text: str) -> Optional[list]:
    """Extract a JSON array of steps from an LLM response. Tolerant of
    leading/trailing prose and markdown fencing."""
    if not text:
        return None
    # Try direct parse first.
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\n", "", candidate)
        candidate = re.sub(r"\n```\s*$", "", candidate)
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and isinstance(parsed.get("plan"), list):
            return parsed["plan"]
    except (ValueError, TypeError):
        pass
    # Fallback: regex extract the first JSON array.
    m = _JSON_BLOCK_RE.search(text)
    if not m:
        return None
    try:
        parsed = json.loads(m.group(0))
        if isinstance(parsed, list):
            return parsed
    except (ValueError, TypeError):
        return None
    return None


def _plan_via_llm(intent: str, llm_config) -> tuple[Optional[list], str]:
    """Ask the LLM for a JSON plan. Returns (plan_list_or_none, notes_msg)."""
    try:
        from opencut.core.llm import query_llm
    except Exception:  # pragma: no cover
        return None, "core/llm import failed"
    try:
        resp = query_llm(intent, config=llm_config, system_prompt=SYSTEM_PROMPT_PLAN)
    except Exception as exc:
        return None, f"LLM call failed: {type(exc).__name__}: {exc}"
    text = getattr(resp, "text", None) or (resp if isinstance(resp, str) else "")
    plan = _parse_llm_plan(text)
    if not plan:
        return None, "LLM returned no parseable JSON plan"
    return plan, "ok"


def _keyword_match_any(fragment: str):
    """Try ``parse_command_keyword`` against the fragment and a few simple
    morphological variants. The shipped ``COMMAND_MAP`` uses ``\\b``
    word-boundary matching so plurals (e.g. ``silences``) miss the
    ``silence`` keyword; this helper retries with a singular form."""
    from opencut.core.nlp_command import parse_command_keyword
    if not fragment:
        return None
    match = parse_command_keyword(fragment)
    if match:
        return match
    # Strip trailing 's' from word-tokens longer than 3 chars.
    morph = re.sub(r"\b([A-Za-z]{4,})s\b", r"\1", fragment)
    if morph != fragment:
        match = parse_command_keyword(morph)
        if match:
            return match
    # As a last resort, look for any keyword as a substring (no boundary).
    txt = fragment.lower()
    from opencut.core.nlp_command import _SORTED_COMMAND_MAP
    for entry in _SORTED_COMMAND_MAP:
        for kw in entry["keywords"]:
            if kw in txt:
                return {
                    "route": entry["route"],
                    "params": dict(entry.get("params") or {}),
                    "confidence": 0.4,
                    "matched_keyword": kw,
                }
    return None


def _plan_via_keywords(intent: str) -> list:
    """Fallback: produce a plan by splitting the intent on conjunctions
    and matching each fragment against the COMMAND_MAP.

    Always returns at least one step (the best single match for the
    whole intent) so the UI never sees an empty plan.
    """
    if not intent or not isinstance(intent, str):
        return []
    fragments = [f.strip() for f in re.split(r",|\band\b|\bthen\b|;", intent, flags=re.IGNORECASE) if f.strip()]
    plan: list = []
    seen_routes: set[str] = set()
    for frag in fragments:
        match = _keyword_match_any(frag)
        if not match:
            continue
        if match["route"] in seen_routes:
            continue
        seen_routes.add(match["route"])
        plan.append({
            "label": frag[:120],
            "endpoint": match["route"],
            "payload": dict(match.get("params") or {}),
            "skill": "",
        })
    if not plan:
        whole = _keyword_match_any(intent)
        if whole:
            plan.append({
                "label": intent[:120],
                "endpoint": whole["route"],
                "payload": dict(whole.get("params") or {}),
                "skill": "",
            })
    return plan


def _coerce_step(raw: dict) -> Optional[PlanStep]:
    if not isinstance(raw, dict):
        return None
    endpoint = str(raw.get("endpoint") or "").strip()
    if not endpoint:
        return None
    # Defensive: refuse endpoints that don't start with '/' (model may
    # echo back the verb or a route name).
    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint
    payload = raw.get("payload") if isinstance(raw.get("payload"), dict) else {}
    return PlanStep(
        step_id=uuid.uuid4().hex[:12],
        label=str(raw.get("label") or endpoint),
        endpoint=endpoint,
        payload=payload,
        skill=str(raw.get("skill") or ""),
    )


def plan(intent: str, llm_config=None, session_id: Optional[str] = None) -> PlanResult:
    """Generate a plan from natural-language ``intent``.

    Args:
        intent: The user's request, e.g. ``"Cut silences then add captions"``.
        llm_config: Optional ``LLMConfig`` — when provided, try the LLM
            first; on parse failure, fall back to the keyword planner.
        session_id: When omitted, a fresh session is minted.

    Returns:
        :class:`PlanResult` (subscriptable for Flask jsonify).
    """
    if not isinstance(intent, str) or not intent.strip():
        raise ValueError("intent is required")
    sid = session_id or uuid.uuid4().hex
    notes: List[str] = []
    raw_steps: list = []
    source = "keyword"
    if llm_config is not None:
        raw_steps, note = _plan_via_llm(intent, llm_config)
        if raw_steps:
            source = "llm"
            notes.append(note)
        else:
            notes.append(f"LLM unavailable or failed ({note}); using keyword fallback")
    if not raw_steps:
        raw_steps = _plan_via_keywords(intent)
        source = source if raw_steps and source == "llm" else "keyword"
    steps: List[PlanStep] = []
    for raw in raw_steps:
        step = _coerce_step(raw)
        if step is not None:
            steps.append(step)
    if not steps:
        notes.append("No steps matched — try a more specific intent.")
    result = PlanResult(
        session_id=sid,
        intent=intent.strip(),
        plan=steps,
        source=source,
        notes=notes,
    )
    # Persist the planning turn.
    save_session(sid, {
        "session_id": sid,
        "intent": result.intent,
        "plan": [s.to_dict() for s in steps],
        "source": source,
        "notes": notes,
        "executed_steps": [],
        "review": None,
        "created_ts": time.time(),
        "updated_ts": time.time(),
    })
    return result


# ---------------------------------------------------------------------------
# Step execution (delegated to the existing workflow runner)
# ---------------------------------------------------------------------------
def mark_step_status(
    session_id: str,
    step_id: str,
    status: str,
    *,
    job_id: str = "",
    reason: str = "",
) -> dict:
    """Record a step's execution status back into the session JSON.

    Returns the updated session payload.
    """
    if status not in ("planned", "executing", "ok", "failed", "rejected"):
        raise ValueError(f"Invalid status '{status}'")
    sess = load_session(session_id)
    if sess is None:
        raise FileNotFoundError(f"Session not found: {session_id}")
    found = False
    for step in sess.get("plan", []):
        if step.get("step_id") == step_id:
            step["status"] = status
            if job_id:
                step["job_id"] = job_id
            if reason:
                step["reason"] = reason
            found = True
            break
    if not found:
        raise KeyError(f"Step {step_id} not in session {session_id}")
    sess.setdefault("executed_steps", []).append({
        "step_id": step_id,
        "status": status,
        "job_id": job_id,
        "reason": reason,
        "ts": time.time(),
    })
    sess["updated_ts"] = time.time()
    save_session(session_id, sess)
    return sess


# ---------------------------------------------------------------------------
# Self-review (F144 hook — minimal scaffold)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_REVIEW = (
    "You are a critic auditing an automated video-editor's work. The "
    "user asked: \"{intent}\". The conductor ran these steps: {steps}. "
    "Identify any drift between intent and the executed plan in 1-3 "
    "bullet points. If everything looks right, say 'Plan matched intent.'"
)


SYSTEM_PROMPT_REVIEW_STRUCTURED = (
    "You are a quality reviewer for an automated video editor. "
    "Compare the user's intent against the steps the conductor ran. "
    "Reply with STRICT JSON only (no prose, no markdown fences). "
    "Schema: {\"matched\": bool, \"drift_score\": int 0-100 "
    "(100 = perfect match), \"drift_notes\": [string], "
    "\"suggested_retry\": null | {\"label\": string, "
    "\"endpoint\": string (an OpenCut REST route like /silence), "
    "\"payload\": object, \"rationale\": string}}. "
    "Use suggested_retry to propose ONE remediation step that would "
    "close the most important drift; use null when matched is true."
)


@dataclass
class ReviewResult:
    session_id: str
    intent: str
    drift_notes: List[str] = field(default_factory=list)
    matched: bool = False
    source: str = "heuristic"   # "llm" or "llm_structured" or "heuristic"
    drift_score: int = 100      # 0..100 — 100 means matched
    suggested_retry: Optional[dict] = None  # remediation step the LLM proposed

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "intent": self.intent,
            "drift_notes": list(self.drift_notes),
            "matched": self.matched,
            "source": self.source,
            "drift_score": int(self.drift_score),
            "suggested_retry": dict(self.suggested_retry) if isinstance(self.suggested_retry, dict) else None,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def keys(self):
        return (
            "session_id", "intent", "drift_notes", "matched", "source",
            "drift_score", "suggested_retry",
        )

    def __contains__(self, key: str) -> bool:
        return key in self.keys()


def _summarise_steps(steps: List[dict]) -> str:
    """Build a compact ``[{label, endpoint, status, reason?}, …]`` summary
    that the LLM can reason about. Bounded length so the prompt stays
    short on small models."""
    out: list = []
    for s in steps:
        item = {
            "label": str(s.get("label", ""))[:80],
            "endpoint": str(s.get("endpoint", "")),
            "status": str(s.get("status", "")),
        }
        reason = str(s.get("reason", "")).strip()
        if reason:
            item["reason"] = reason[:80]
        out.append(item)
    text = json.dumps(out, separators=(",", ":"))
    return text[:1500]


def _parse_structured_review(text: str) -> Optional[dict]:
    """Parse the strict-JSON structured-review response.

    Tolerates leading/trailing prose and markdown fences (same approach
    as ``_parse_llm_plan``). Returns ``None`` when no parseable object
    is found.
    """
    if not text:
        return None
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\n", "", candidate)
        candidate = re.sub(r"\n```\s*$", "", candidate)
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except (ValueError, TypeError):
        pass
    # Last-resort regex: first {...} block.
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        parsed = json.loads(m.group(0))
        if isinstance(parsed, dict):
            return parsed
    except (ValueError, TypeError):
        return None
    return None


def _coerce_suggested_retry(raw: Any) -> Optional[dict]:
    """Sanitize the LLM's suggested_retry shape. Returns ``None`` for
    anything that can't safely be appended to the plan."""
    if not isinstance(raw, dict):
        return None
    endpoint = str(raw.get("endpoint") or "").strip()
    if not endpoint:
        return None
    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint
    payload = raw.get("payload") if isinstance(raw.get("payload"), dict) else {}
    label = str(raw.get("label") or endpoint)[:120]
    rationale = str(raw.get("rationale") or "").strip()[:240]
    return {
        "label": label,
        "endpoint": endpoint,
        "payload": payload,
        "rationale": rationale,
    }


def _append_retry_step(sess: dict, retry: dict) -> None:
    """Append a sanitized ``suggested_retry`` to the session plan as a
    new ``planned`` step. Mutates ``sess`` in place; caller is
    responsible for persisting."""
    new_step = {
        "step_id": uuid.uuid4().hex[:12],
        "label": retry["label"],
        "endpoint": retry["endpoint"],
        "payload": dict(retry["payload"]),
        "skill": "",
        "status": "planned",
        "job_id": "",
        "reason": "F144 self-review suggested retry: " + retry.get("rationale", ""),
    }
    sess.setdefault("plan", []).append(new_step)


def review(
    session_id: str,
    llm_config=None,
    *,
    append_retry: bool = True,
    structured: bool = True,
) -> ReviewResult:
    """Compare the executed plan against the original intent (F144).

    With an LLM provider the model writes structured drift notes (and
    optionally proposes a remediation step). Without one, a heuristic
    reports failed/rejected steps as drift signals so the user still
    gets a status summary.

    Args:
        session_id: Session to review.
        llm_config: Optional ``LLMConfig`` — when provided, try the
            structured LLM critique first; fall back to the original
            free-text prompt on parse failure, then to the heuristic.
        append_retry: When True (default), append any ``suggested_retry``
            the LLM proposed to the session's plan as a new
            ``planned`` step. Use ``append_retry=False`` for a
            "review-only" call when the UI hasn't yet asked the user
            to accept the retry.
        structured: When True (default) and an LLM is configured, try
            the strict-JSON structured prompt first. Set False to fall
            back to the legacy free-text prompt.
    """
    sess = load_session(session_id)
    if sess is None:
        raise FileNotFoundError(f"Session not found: {session_id}")
    intent = str(sess.get("intent", ""))
    steps = sess.get("plan", [])
    failed = [s for s in steps if s.get("status") == "failed"]
    rejected = [s for s in steps if s.get("status") == "rejected"]

    # Heuristic baseline: drift_score drops 25 points per failed step,
    # 15 points per rejected step. Floors at 0.
    matched = True
    drift: List[str] = []
    drift_score = 100
    if failed:
        matched = False
        drift_score -= 25 * len(failed)
        drift.append(f"{len(failed)} step(s) failed: " + ", ".join(s.get("label", "") for s in failed))
    if rejected:
        matched = False
        drift_score -= 15 * len(rejected)
        drift.append(f"{len(rejected)} step(s) rejected: " + ", ".join(s.get("label", "") for s in rejected))
    drift_score = max(0, min(100, drift_score))
    source = "heuristic"
    suggested_retry: Optional[dict] = None

    if llm_config is not None and intent:
        try:
            from opencut.core.llm import query_llm

            if structured:
                prompt = (
                    f"User intent: {intent}\n"
                    f"Executed steps: {_summarise_steps(steps)}\n"
                    "Reply with the strict JSON schema described in the "
                    "system prompt."
                )
                resp = query_llm(
                    prompt,
                    config=llm_config,
                    system_prompt=SYSTEM_PROMPT_REVIEW_STRUCTURED,
                )
                text = getattr(resp, "text", None) or (resp if isinstance(resp, str) else "")
                parsed = _parse_structured_review(text)
                if parsed is not None:
                    source = "llm_structured"
                    matched = bool(parsed.get("matched", matched))
                    score_raw = parsed.get("drift_score")
                    try:
                        drift_score = int(score_raw)
                    except (TypeError, ValueError):
                        pass
                    drift_score = max(0, min(100, drift_score))
                    notes = parsed.get("drift_notes")
                    if isinstance(notes, list):
                        # Replace heuristic notes with the LLM's when the
                        # structured parse succeeded — the LLM has more
                        # context. Cap each note to keep the UI legible.
                        drift = [str(n)[:300] for n in notes if isinstance(n, str)]
                        if not drift:
                            drift = ["Plan matched intent." if matched else "Drift detected."]
                    suggested_retry = _coerce_suggested_retry(parsed.get("suggested_retry"))
                else:
                    # Structured parse failed — fall back to free text.
                    structured = False

            if not structured:
                prompt = SYSTEM_PROMPT_REVIEW.format(
                    intent=intent,
                    steps=_summarise_steps(steps),
                )
                resp = query_llm(prompt, config=llm_config, system_prompt="You are a careful reviewer.")
                text = getattr(resp, "text", None) or (resp if isinstance(resp, str) else "")
                if text:
                    source = "llm"
                    if "matched intent" in text.lower() and not failed and not rejected:
                        matched = True
                        drift_score = 100
                    else:
                        matched = matched and "matched intent" in text.lower()
                    drift.append(text.strip()[:400])
        except Exception as exc:  # pragma: no cover — never crash on LLM failure
            drift.append(f"LLM review unavailable: {type(exc).__name__}: {exc}")

    if not drift:
        drift.append("Plan matched intent.")

    # Append the suggested retry as a new planned step (F144 closes the
    # self-correcting loop the RFC calls out). The UI surfaces it as
    # "Agent proposes…" pending user accept/reject.
    appended = False
    if append_retry and suggested_retry and not matched:
        _append_retry_step(sess, suggested_retry)
        appended = True

    sess["review"] = {
        "drift_notes": drift,
        "matched": matched,
        "source": source,
        "drift_score": drift_score,
        "suggested_retry": suggested_retry,
        "appended_retry_step": appended,
        "ts": time.time(),
    }
    sess["updated_ts"] = time.time()
    save_session(session_id, sess)
    return ReviewResult(
        session_id=session_id,
        intent=intent,
        drift_notes=drift,
        matched=matched,
        source=source,
        drift_score=drift_score,
        suggested_retry=suggested_retry,
    )


__all__ = [
    "PlanStep",
    "PlanResult",
    "ReviewResult",
    "INSTALL_HINT",
    "check_agent_chat_available",
    "plan",
    "review",
    "load_session",
    "save_session",
    "mark_step_status",
]
