"""F344 — repeat detection must say which take to keep, not just which repeat.

Detection returned "these repeat" with no ranking, so review had no basis for a
preselection. The heuristic has to work with nothing configured, and an LLM
verdict is an optional override whose absence or failure is recorded rather
than swallowed.
"""

from __future__ import annotations

import pytest

from opencut.core.repeat_detect import (
    _filler_count,
    _take_signals,
    detect_repeated_takes,
    rank_repeat_clusters,
)


def _seg(start, end, text):
    return {"start": start, "end": end, "text": text}


FUMBLED = _seg(0.0, 3.0, "The quarterly numbers are up um twelve")
CLEAN = _seg(3.2, 6.0, "The quarterly numbers are up twelve percent.")
UNRELATED = _seg(20.0, 22.0, "Totally different sentence here.")


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------


def test_filler_count_uses_the_shared_filler_list():
    assert _filler_count("um so like basically yes") >= 4
    assert _filler_count("A clean sentence with no hesitation.") == 0


def test_completion_reads_terminal_punctuation():
    assert _take_signals(CLEAN)["completed"] == 1.0
    assert _take_signals(FUMBLED)["completed"] == 0.0


def test_speech_rate_is_words_per_minute():
    signals = _take_signals(_seg(0.0, 60.0, " ".join(["word"] * 120)))

    assert signals["words_per_minute"] == pytest.approx(120.0)


def test_zero_length_take_does_not_divide_by_zero():
    signals = _take_signals(_seg(5.0, 5.0, "text"))

    assert signals["duration"] == 0.0
    assert signals["words_per_minute"] == 0.0


def test_missing_fields_are_tolerated():
    signals = _take_signals({})

    assert signals["word_count"] == 0.0
    assert signals["completed"] == 0.0


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def test_the_clean_complete_take_is_recommended():
    clusters = rank_repeat_clusters([FUMBLED, CLEAN, UNRELATED], {0})

    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster["indices"] == [0, 1]
    assert cluster["keep_index"] == 1
    assert cluster["cut_indices"] == [0]


def test_takes_are_ordered_best_first():
    cluster = rank_repeat_clusters([FUMBLED, CLEAN, UNRELATED], {0})[0]
    scores = [take["score"] for take in cluster["takes"]]

    assert scores == sorted(scores, reverse=True)
    assert cluster["takes"][0]["index"] == cluster["keep_index"]


def test_every_take_carries_its_signals():
    cluster = rank_repeat_clusters([FUMBLED, CLEAN, UNRELATED], {0})[0]

    for take in cluster["takes"]:
        for key in ("filler_count", "words_per_minute", "completed", "word_count"):
            assert key in take["signals"], take


def test_fillers_push_a_take_down_the_ranking():
    noisy = _seg(0.0, 3.0, "The numbers are um uh like basically up.")
    clean = _seg(3.2, 6.0, "The numbers are up.")
    cluster = rank_repeat_clusters([noisy, clean], {0})[0]

    assert cluster["keep_index"] == 1


def test_a_run_of_fumbles_is_one_cluster():
    """Three attempts at the same line is one decision, not three."""
    segments = [
        _seg(0.0, 2.0, "The numbers are um"),
        _seg(2.1, 4.0, "The numbers are uh"),
        _seg(4.1, 6.0, "The numbers are up twelve percent."),
    ]
    clusters = rank_repeat_clusters(segments, {0, 1})

    assert len(clusters) == 1
    assert clusters[0]["indices"] == [0, 1, 2]
    assert clusters[0]["keep_index"] == 2
    assert clusters[0]["cut_indices"] == [0, 1]


def test_separate_fumbles_stay_separate_clusters():
    segments = [
        _seg(0.0, 2.0, "First line um"),
        _seg(2.1, 4.0, "First line done."),
        _seg(30.0, 32.0, "Second line uh"),
        _seg(32.1, 34.0, "Second line done."),
    ]
    clusters = rank_repeat_clusters(segments, {0, 2})

    assert [cluster["indices"] for cluster in clusters] == [[0, 1], [2, 3]]


def test_no_repeats_means_no_clusters():
    assert rank_repeat_clusters([CLEAN, UNRELATED], set()) == []


def test_a_trailing_repeat_without_a_successor_still_ranks():
    """The last segment being a repeat must not index past the end."""
    clusters = rank_repeat_clusters([FUMBLED], {0})

    assert len(clusters) == 1
    assert clusters[0]["indices"] == [0]
    assert clusters[0]["keep_index"] == 0
    assert clusters[0]["cut_indices"] == []


# ---------------------------------------------------------------------------
# LLM layer
# ---------------------------------------------------------------------------


def test_the_heuristic_runs_with_no_llm_configured():
    cluster = rank_repeat_clusters([FUMBLED, CLEAN], {0})[0]

    assert cluster["decision_source"] == "heuristic"
    assert cluster["fallback_reason"] == "no llm configured"


def test_an_llm_verdict_overrides_the_heuristic():
    cluster = rank_repeat_clusters([FUMBLED, CLEAN], {0}, llm_verdict=lambda takes: 0)[0]

    assert cluster["keep_index"] == 0
    assert cluster["decision_source"] == "llm"


def test_a_failing_llm_falls_back_and_records_why():
    def _boom(_takes):
        raise RuntimeError("model unreachable")

    cluster = rank_repeat_clusters([FUMBLED, CLEAN], {0}, llm_verdict=_boom)[0]

    assert cluster["decision_source"] == "heuristic"
    assert cluster["keep_index"] == 1
    assert "model unreachable" in cluster["fallback_reason"]


def test_an_out_of_range_llm_verdict_is_refused():
    cluster = rank_repeat_clusters([FUMBLED, CLEAN], {0}, llm_verdict=lambda takes: 99)[0]

    assert cluster["decision_source"] == "heuristic"
    assert "out of range" in cluster["fallback_reason"]


# ---------------------------------------------------------------------------
# Detector integration
# ---------------------------------------------------------------------------


def test_detect_returns_clusters_alongside_the_existing_keys():
    result = detect_repeated_takes([FUMBLED, CLEAN, UNRELATED])

    assert {"repeats", "clean_ranges", "clusters"} <= set(result)
    assert result["clusters"][0]["keep_index"] == 1


def test_detect_only_output_is_still_available():
    """Existing callers must not be forced into the new shape."""
    result = detect_repeated_takes([FUMBLED, CLEAN, UNRELATED], rank_takes=False)

    assert "clusters" not in result
    assert result["repeats"]


def test_empty_input_returns_an_empty_cluster_list():
    assert detect_repeated_takes([])["clusters"] == []


def test_cut_indices_never_include_the_keep():
    result = detect_repeated_takes([FUMBLED, CLEAN, UNRELATED])

    for cluster in result["clusters"]:
        assert cluster["keep_index"] not in cluster["cut_indices"]
        assert set(cluster["cut_indices"]) < set(cluster["indices"])
