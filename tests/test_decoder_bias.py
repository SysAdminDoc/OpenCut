"""F323 — bias the decoder with the glossary instead of only repairing after.

The glossary was applied as find/replace over a finished transcript, so a
proper noun the decoder never emitted in a recognisable form could not be
repaired. Biasing the decoder with the glossary's target spellings fixes the
cause; the post-hoc pass stays as the second layer.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opencut.core import captions  # noqa: E402
from opencut.core.transcript_corrections import (  # noqa: E402
    MAX_BIAS_CHARS,
    build_hotwords,
    glossary_bias_terms,
)


class TestBiasTermSelection:
    def test_target_spelling_is_the_bias_term(self):
        """The rule says "when you hear X write Y"; Y is what we want emitted."""
        rules = [{"find": "cloud native", "replace": "CloudNative"}]
        assert glossary_bias_terms(rules) == ["CloudNative"]

    def test_deletion_rules_contribute_nothing(self):
        rules = [{"find": "umm", "replace": ""}]
        assert glossary_bias_terms(rules) == []

    def test_order_is_preserved_and_duplicates_collapse(self):
        rules = [
            {"find": "a", "replace": "Kubernetes"},
            {"find": "b", "replace": "Postgres"},
            {"find": "c", "replace": "kubernetes"},
        ]
        assert glossary_bias_terms(rules) == ["Kubernetes", "Postgres"]

    def test_malformed_entries_are_skipped(self):
        assert glossary_bias_terms([None, "nope", 42, {"find": "x"}]) == []

    def test_empty_and_none_are_safe(self):
        assert glossary_bias_terms([]) == []
        assert glossary_bias_terms(None) == []


class TestHotwordString:
    def test_terms_are_joined(self):
        assert build_hotwords(["Kubernetes", "Postgres"]) == "Kubernetes, Postgres"

    def test_cap_drops_whole_terms_never_fragments(self):
        """A truncated term would bias the decoder toward a fragment."""
        terms = [f"Term{i:03d}" for i in range(200)]
        out = build_hotwords(terms)
        assert len(out) <= MAX_BIAS_CHARS
        for chunk in out.split(", "):
            assert chunk in terms

    def test_whitespace_is_normalised(self):
        assert build_hotwords(["  Cloud   Native "]) == "Cloud Native"

    def test_empty_input_disables_biasing(self):
        assert build_hotwords([]) == ""


class TestResolveFromGlossary:
    def test_glossary_becomes_hotwords(self, monkeypatch):
        monkeypatch.setattr(
            "opencut.user_data.load_transcript_glossary",
            lambda _p: [{"find": "kuber netes", "replace": "Kubernetes"}],
        )
        assert captions.resolve_glossary_hotwords("proj") == "Kubernetes"

    def test_no_glossary_means_no_biasing(self, monkeypatch):
        monkeypatch.setattr("opencut.user_data.load_transcript_glossary", lambda _p: [])
        assert captions.resolve_glossary_hotwords("proj") == ""

    def test_a_broken_glossary_never_fails_the_job(self, monkeypatch):
        def _boom(_p):
            raise OSError("glossary unreadable")

        monkeypatch.setattr("opencut.user_data.load_transcript_glossary", _boom)
        assert captions.resolve_glossary_hotwords("proj") == ""


class TestDecoderBiasKwargs:
    def test_no_terms_sends_nothing(self):
        assert captions._decoder_bias_kwargs("") == {}
        assert captions._decoder_bias_kwargs(None) == {}

    def test_modern_wrapper_gets_hotwords(self, monkeypatch):
        import types

        fake = types.ModuleType("faster_whisper")

        class _Model:
            def transcribe(self, audio, hotwords=None, initial_prompt=None):
                pass

        fake.WhisperModel = _Model
        monkeypatch.setitem(sys.modules, "faster_whisper", fake)
        assert captions._decoder_bias_kwargs("Kubernetes") == {"hotwords": "Kubernetes"}

    def test_older_wrapper_falls_back_to_initial_prompt(self, monkeypatch):
        """Passing hotwords to a wrapper without it would raise TypeError."""
        import types

        fake = types.ModuleType("faster_whisper")

        class _Model:
            def transcribe(self, audio, initial_prompt=None):
                pass

        fake.WhisperModel = _Model
        monkeypatch.setitem(sys.modules, "faster_whisper", fake)
        assert captions._decoder_bias_kwargs("Kubernetes") == {
            "initial_prompt": "Kubernetes"
        }

    def test_wrapper_supporting_neither_biases_nothing(self, monkeypatch):
        import types

        fake = types.ModuleType("faster_whisper")

        class _Model:
            def transcribe(self, audio):
                pass

        fake.WhisperModel = _Model
        monkeypatch.setitem(sys.modules, "faster_whisper", fake)
        assert captions._decoder_bias_kwargs("Kubernetes") == {}


class TestProvenanceRecordsTheLayer:
    def _prov(self, hotwords):
        from opencut.utils.config import CaptionConfig

        return captions._backend_provenance(
            "faster-whisper",
            CaptionConfig(hotwords=hotwords),
            "en",
            device="cpu",
            compute_type="int8",
            decode_mode="sequential",
        )

    def test_biased_run_is_labelled(self):
        opts = self._prov("Kubernetes").deterministic_options
        assert opts["glossary_layer"] == "decoder-bias+post-hoc"
        assert opts["decoder_bias_terms"] == "Kubernetes"

    def test_unbiased_run_is_labelled_post_hoc_only(self):
        opts = self._prov("").deterministic_options
        assert opts["glossary_layer"] == "post-hoc"
        assert "decoder_bias_terms" not in opts
