"""Contract tests for the local Ollama Qwen3-VL lane."""
from __future__ import annotations

import json
from types import SimpleNamespace


def test_qwen3vl_analyze_scores_transcript_first_and_sends_frames(tmp_path, monkeypatch):
    from opencut.core import llm
    from opencut.core import multimodal_qwen3vl as qwen

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"test video placeholder")
    monkeypatch.setattr(qwen, "_probe_duration", lambda _path: 5.0)
    monkeypatch.setattr(
        qwen,
        "_sample_video_frames",
        lambda _path, timestamps: [
            {"timestamp": timestamp, "base64": f"frame-{timestamp}"}
            for timestamp in timestamps
        ],
    )
    calls = []

    def fake_vision_query(*, prompt, images, config, system_prompt):
        calls.append({
            "prompt": prompt,
            "images": images,
            "config": config,
            "system_prompt": system_prompt,
        })
        return SimpleNamespace(
            text=json.dumps({
                "transcript_score": 0.9,
                "visual_score": 0.2,
                "relevance": 0.2,
                "reason": "The transcript carries the moment.",
                "summary": "A useful segment.",
            })
        )

    monkeypatch.setattr(llm, "query_ollama_vision", fake_vision_query)
    result = qwen.analyze(
        str(video),
        prompt="Find useful moments.",
        transcript_segments=[
            {"start": 0, "end": 2, "text": "The important explanation."},
            {"start": 2, "end": 5, "text": "The visual example."},
        ],
        frames_per_segment=3,
    )

    assert len(result.structured_data) == 2
    assert result.frames_analyzed == 6
    assert result.structured_data[0]["relevance"] == 0.655
    assert result.structured_data[0]["transcript_score"] == 0.9
    assert result.structured_data[0]["visual_score"] == 0.2
    assert len(calls) == 2
    assert all(len(call["images"]) == 3 for call in calls)
    assert all(call["config"].api_key == "" for call in calls)
    assert "TRANSCRIPT (primary signal)" in calls[0]["prompt"]


def test_qwen3vl_probe_requires_a_local_model(monkeypatch):
    from opencut.core import llm
    from opencut.core import multimodal_qwen3vl as qwen

    monkeypatch.setattr(
        llm,
        "list_ollama_models",
        lambda _url, timeout=10: ["llama3.2:latest", "qwen3-vl:8b"],
    )
    assert qwen.check_qwen3vl_available()
    assert not qwen.check_qwen3vl_available(model="qwen3-vl:4b")


def test_ollama_vision_transport_stays_loopback_and_keyless(monkeypatch):
    from opencut.core import llm

    captured = {}

    def fake_http_json(url, data=None, **_kwargs):
        captured["url"] = url
        captured["body"] = data
        return {"response": "{}", "eval_count": 4}

    monkeypatch.setattr(llm, "_http_json", fake_http_json)
    response = llm.query_ollama_vision(
        "Score this frame.",
        ["base64-jpeg"],
        config=llm.LLMConfig(model="qwen3-vl:8b"),
    )

    assert response.provider == "ollama"
    assert captured["url"] == "http://localhost:11434/api/generate"
    assert captured["body"]["images"] == ["base64-jpeg"]
    assert "api_key" not in captured["body"]


def test_highlights_passes_sampled_frames_to_local_ollama(monkeypatch, tmp_path):
    from opencut.core import highlights
    from opencut.core.llm import LLMConfig

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"test video placeholder")
    monkeypatch.setattr(
        highlights,
        "extract_frames_for_vision",
        lambda _path, interval_seconds=10.0: [
            {"timestamp": 1.0, "base64": "frame-one"},
            {"timestamp": 2.0, "base64": "frame-two"},
        ],
    )
    captured = {}

    def fake_query(*, prompt, config, system_prompt, images=None):
        captured["images"] = images
        return SimpleNamespace(
            text=json.dumps([
                {
                    "start": 0,
                    "end": 20,
                    "score": 0.8,
                    "reason": "Clear explanation",
                    "title": "Explanation",
                }
            ]),
            provider="ollama",
            model=config.model,
        )

    monkeypatch.setattr("opencut.core.llm.query_llm", fake_query)
    result = highlights.extract_highlights_with_vision(
        str(video),
        [{"start": 0, "end": 20, "text": "A clear explanation."}],
        llm_config=LLMConfig(provider="ollama", model="qwen3-vl:8b"),
    )

    assert result.total_found == 1
    assert captured["images"] == ["frame-one", "frame-two"]


def _stub_scoring(monkeypatch, qwen, llm, on_segment=None):
    """Wire analyze() to fake frames and a fake Ollama so it runs instantly."""
    monkeypatch.setattr(
        qwen,
        "_sample_video_frames",
        lambda _path, timestamps: [
            {"timestamp": timestamp, "base64": "frame"} for timestamp in timestamps
        ],
    )

    def fake_vision_query(*, prompt, images, config, system_prompt):
        if on_segment is not None:
            on_segment()
        return SimpleNamespace(
            text=json.dumps({"transcript_score": 0.5, "visual_score": 0.5})
        )

    monkeypatch.setattr(llm, "query_ollama_vision", fake_vision_query)


def test_cancelling_stops_the_loop_instead_of_running_the_list_out(tmp_path, monkeypatch):
    """F366 — nothing else can interrupt this worker.

    The frame grabs use subprocess.run, so the job runner registers no process
    to kill and only checks cancellation before start and after return. Without
    a per-segment poll a cancelled job keeps spawning ffmpeg and Ollama calls
    until the whole segment list is exhausted.
    """
    from opencut.core import llm
    from opencut.core import multimodal_qwen3vl as qwen

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"test video placeholder")
    monkeypatch.setattr(qwen, "_probe_duration", lambda _path: 100.0)

    scored = []
    _stub_scoring(monkeypatch, qwen, llm, on_segment=lambda: scored.append(1))
    cancelled = {"value": False}

    segments = [{"start": i, "end": i + 1, "text": f"line {i}"} for i in range(50)]

    try:
        qwen.analyze(
            str(video),
            transcript_segments=segments,
            is_cancelled=lambda: cancelled["value"],
            frames_per_segment=1,
        )
    except InterruptedError:
        raise AssertionError("uncancelled run must not raise")

    assert len(scored) == 50, "control run should score every segment"

    # Now cancel after the first segment and prove the loop stops there.
    scored.clear()

    def cancel_after_first():
        scored.append(1)
        cancelled["value"] = True

    _stub_scoring(monkeypatch, qwen, llm, on_segment=cancel_after_first)

    try:
        qwen.analyze(
            str(video),
            transcript_segments=segments,
            is_cancelled=lambda: cancelled["value"],
            frames_per_segment=1,
        )
    except InterruptedError:
        pass
    else:
        raise AssertionError("a cancelled run must raise InterruptedError")

    assert len(scored) == 1, f"loop kept going after cancel: {len(scored)} segments"


def test_is_cancelled_is_an_explicit_parameter_not_swallowed_by_kwargs():
    """F366 — analyze() ends in **kwargs with `del kwargs`.

    A callback passed by keyword alone would be silently discarded, which is
    exactly the trap a call-site-only fix would fall into.
    """
    import inspect

    from opencut.core import multimodal_qwen3vl as qwen

    assert "is_cancelled" in inspect.signature(qwen.analyze).parameters


def test_the_no_transcript_fallback_coarsens_instead_of_making_thousands(monkeypatch):
    """F366 — a 2-hour video at segment_duration=1.0 used to yield 7,200 windows."""
    from opencut.core import multimodal_qwen3vl as qwen

    two_hours = 7200.0
    segments = qwen._fallback_segments(two_hours, 1.0)

    assert len(segments) <= qwen.MAX_SEGMENTS
    # Still covers the whole file, just at a coarser granularity.
    assert segments[0]["start"] == 0.0
    assert abs(segments[-1]["end"] - two_hours) < 0.01
    # A file that already fits is left exactly as it was.
    short = qwen._fallback_segments(60.0, 30.0)
    assert [(s["start"], s["end"]) for s in short] == [(0.0, 30.0), (30.0, 60.0)]


def test_per_segment_notes_cannot_grow_without_bound(tmp_path, monkeypatch):
    """F366 — notes are one string per skipped segment and land in the job result."""
    from opencut.core import llm
    from opencut.core import multimodal_qwen3vl as qwen

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"test video placeholder")
    monkeypatch.setattr(qwen, "_probe_duration", lambda _path: 500.0)

    # Only the first segment yields frames, so it scores and the run succeeds;
    # every other segment wants to append a note.
    def frames_for_first_only(_path, timestamps):
        if timestamps and timestamps[0] < 1.0:
            return [{"timestamp": timestamps[0], "base64": "frame"}]
        return []

    monkeypatch.setattr(qwen, "_sample_video_frames", frames_for_first_only)
    monkeypatch.setattr(
        llm,
        "query_ollama_vision",
        lambda **_kwargs: SimpleNamespace(
            text=json.dumps({"transcript_score": 0.5, "visual_score": 0.5})
        ),
    )

    segments = [{"start": i, "end": i + 1, "text": f"line {i}"} for i in range(400)]
    result = qwen.analyze(str(video), transcript_segments=segments, frames_per_segment=1)

    assert len(result.structured_data) == 1
    # Two preamble notes plus the capped per-segment run, not 399 of them.
    assert len(result.notes) <= qwen.MAX_NOTES + 3, len(result.notes)
    assert result.notes[-1] == "Further per-segment notes were suppressed."


def test_the_route_refuses_an_oversized_transcript_segment_list(client, csrf_token, tmp_path, monkeypatch):
    """F366 — transcript_segments is the one caller-supplied list on this route.

    A minimal segment is about 20 bytes, so the 100 MB body limit admits
    millions of them, and each costs up to six ffmpeg frame grabs plus an
    Ollama call. The guard has to refuse before any of that work starts.
    """
    import time

    from opencut.core import multimodal_qwen3vl as qwen
    from tests.conftest import csrf_headers

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"test video placeholder")

    monkeypatch.setattr(qwen, "check_qwen3vl_available", lambda *a, **k: True)
    # If the guard ever stops firing, this makes the failure loud instead of
    # letting the test pass on a slow analyse that happens to error later.
    def _must_not_run(*_args, **_kwargs):
        raise AssertionError("analyze() ran despite an oversized segment list")

    monkeypatch.setattr(qwen, "analyze", _must_not_run)

    response = client.post(
        "/analyze/video/qwen3vl",
        json={
            "filepath": str(video),
            "transcript_segments": [
                {"start": i, "end": i + 1} for i in range(qwen.MAX_SEGMENTS + 1)
            ],
        },
        headers=csrf_headers(csrf_token),
    )
    assert response.status_code == 200, response.data
    job_id = json.loads(response.data.decode("utf-8"))["job_id"]

    for _ in range(100):
        status = json.loads(client.get(f"/status/{job_id}").data.decode("utf-8"))
        if status.get("status") in ("complete", "error", "cancelled"):
            break
        time.sleep(0.05)

    assert status["status"] == "error", status
    assert "transcript_segments" in str(status.get("error", "")), status
