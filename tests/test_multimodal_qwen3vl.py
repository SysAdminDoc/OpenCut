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
