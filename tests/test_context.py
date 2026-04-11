"""Tests for context awareness system."""

import pytest


class TestClassifyClip:
    """Test clip classification based on metadata."""

    def test_audio_only_clip(self):
        from opencut.core.context_awareness import classify_clip
        tags = classify_clip({"has_audio": True, "has_video": False, "duration": 60})
        assert "audio_only" in tags

    def test_video_only_clip(self):
        from opencut.core.context_awareness import classify_clip
        tags = classify_clip({"has_audio": False, "has_video": True, "duration": 60, "width": 1920, "height": 1080})
        assert "video_only" in tags

    def test_image_clip(self):
        from opencut.core.context_awareness import classify_clip
        tags = classify_clip({"has_video": True, "has_audio": False, "duration": 0, "width": 1920, "height": 1080})
        assert "image_only" in tags

    def test_long_duration(self):
        from opencut.core.context_awareness import classify_clip
        tags = classify_clip({"has_audio": True, "has_video": True, "duration": 600})
        assert "long_duration" in tags

    def test_short_duration(self):
        from opencut.core.context_awareness import classify_clip
        tags = classify_clip({"has_audio": True, "has_video": True, "duration": 30})
        assert "short_duration" in tags

    def test_low_resolution(self):
        from opencut.core.context_awareness import classify_clip
        tags = classify_clip({"has_video": True, "has_audio": False, "duration": 60, "width": 640, "height": 480})
        assert "low_resolution" in tags

    def test_vertical_video(self):
        from opencut.core.context_awareness import classify_clip
        tags = classify_clip({"has_video": True, "has_audio": True, "duration": 30, "width": 1080, "height": 1920})
        assert "vertical_output" in tags

    def test_talking_head(self):
        from opencut.core.context_awareness import classify_clip
        tags = classify_clip({"has_audio": True, "has_video": True, "duration": 120, "num_audio_channels": 1})
        assert "talking_head" in tags


class TestScoreFeatures:
    """Test feature scoring."""

    def test_talking_head_boosts_silence(self):
        from opencut.core.context_awareness import score_features
        tags = {"talking_head", "long_duration"}
        features = score_features(tags)
        silence = next(f for f in features if f["id"] == "silence_detect")
        assert silence["score"] >= 65
        assert silence["relevant"]

    def test_audio_only_penalizes_video(self):
        from opencut.core.context_awareness import score_features
        tags = {"audio_only"}
        features = score_features(tags)
        stabilize = next(f for f in features if f["id"] == "stabilize")
        assert stabilize["score"] < 30
        assert not stabilize["relevant"]

    def test_image_only_penalizes_audio(self):
        from opencut.core.context_awareness import score_features
        tags = {"image_only"}
        features = score_features(tags)
        denoise = next(f for f in features if f["id"] == "denoise")
        assert denoise["score"] < 30

    def test_all_features_have_scores(self):
        from opencut.core.context_awareness import FEATURE_RELEVANCE, score_features
        tags = {"talking_head"}
        features = score_features(tags)
        assert len(features) == len(FEATURE_RELEVANCE)
        for f in features:
            assert 0 <= f["score"] <= 100

    def test_empty_tags(self):
        from opencut.core.context_awareness import score_features
        features = score_features(set())
        assert len(features) > 0
        for f in features:
            assert "id" in f
            assert "score" in f


class TestGuidance:
    """Test guidance message generation."""

    def test_no_tags(self):
        from opencut.core.context_awareness import get_guidance_message
        msg = get_guidance_message(set(), [])
        assert "Select a clip" in msg

    def test_audio_only(self):
        from opencut.core.context_awareness import get_guidance_message
        msg = get_guidance_message({"audio_only"}, [])
        assert "Audio" in msg or "audio" in msg

    def test_image_only(self):
        from opencut.core.context_awareness import get_guidance_message
        msg = get_guidance_message({"image_only"}, [])
        assert "image" in msg.lower()

    def test_talking_head(self):
        from opencut.core.context_awareness import get_guidance_message
        msg = get_guidance_message({"talking_head"}, [])
        assert "talking" in msg.lower() or "Silence" in msg or "Captions" in msg


class TestContextRoute:
    """Test the /context/analyze route."""

    @pytest.fixture
    def client(self):
        from opencut.server import create_app
        app = create_app()
        app.config["TESTING"] = True
        with app.test_client() as c:
            # Get CSRF token
            resp = c.get("/health")
            token = resp.get_json().get("csrf_token", "")
            c._csrf = token
            yield c

    def _post(self, client, url, data):
        return client.post(url, json=data, headers={"X-OpenCut-Token": client._csrf})

    def test_analyze_talking_head(self, client):
        resp = self._post(client, "/context/analyze", {
            "has_audio": True,
            "has_video": True,
            "duration": 300,
            "width": 1920,
            "height": 1080,
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert "tags" in data
        assert "features" in data
        assert "guidance" in data
        assert "tab_scores" in data
        assert len(data["features"]) > 0

    def test_analyze_empty_metadata(self, client):
        resp = self._post(client, "/context/analyze", {})
        assert resp.status_code == 200
        data = resp.get_json()
        assert "features" in data

    def test_analyze_string_booleans_do_not_turn_true(self, client):
        resp = self._post(client, "/context/analyze", {
            "has_audio": "false",
            "has_video": "false",
            "duration": "30",
            "width": "1920",
            "height": "1080",
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert "audio_only" not in data["tags"]
        assert "video_only" not in data["tags"]

    def test_list_features(self, client):
        resp = client.get("/context/features")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "features" in data
        assert len(data["features"]) > 0
