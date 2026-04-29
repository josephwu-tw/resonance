"""
Reliability tests for Resonance.

Covers three areas:
  1. Consistency  — same input must produce the same output every time
  2. Guardrails   — bad/empty inputs must not raise; agent must degrade gracefully
  3. Scoring      — scoring engine invariants (bounded scores, diversity effect)
"""

from unittest.mock import MagicMock, patch

import pytest

from src.recommender import load_songs, recommend_songs
from src.rag import SongIndex
from src import agent
from src.main import _validate

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def songs():
    return load_songs("data/songs.csv")


@pytest.fixture(scope="module")
def index(songs):
    return SongIndex(songs)


# ── 1. Consistency ────────────────────────────────────────────────────────────

class TestConsistency:
    def test_scoring_is_deterministic(self, songs):
        prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}
        r1 = recommend_songs(prefs, songs, k=5)
        r2 = recommend_songs(prefs, songs, k=5)
        assert [s[0]["id"] for s in r1] == [s[0]["id"] for s in r2]

    def test_rag_search_is_deterministic(self, index):
        r1 = index.search("chill lo-fi study music", top_n=5)
        r2 = index.search("chill lo-fi study music", top_n=5)
        assert [s["id"] for s in r1] == [s["id"] for s in r2]

    def test_different_queries_can_produce_different_results(self, index):
        rock = index.search("heavy metal workout", top_n=5)
        lofi = index.search("peaceful lo-fi studying", top_n=5)
        rock_ids = {s["id"] for s in rock}
        lofi_ids = {s["id"] for s in lofi}
        # At least one candidate should differ between very different queries
        assert rock_ids != lofi_ids


# ── 2. Guardrails ─────────────────────────────────────────────────────────────

class TestGuardrails:
    @pytest.mark.parametrize("query,expected_error", [
        ("", "Please describe"),
        ("  ", "Please describe"),
        ("hi", "too short"),
        ("x" * 501, "500 characters"),
    ])
    def test_validate_rejects_bad_input(self, query, expected_error):
        msg = _validate(query)
        assert msg is not None
        assert expected_error.lower() in msg.lower()

    def test_validate_accepts_valid_query(self):
        assert _validate("chill lo-fi for studying") is None

    def test_parse_preferences_returns_tuple_when_api_fails(self):
        """Agent must return (dict, float) even when Claude is unavailable."""
        with patch("src.agent._get_client") as mock_client:
            mock_client.return_value.messages.create.side_effect = Exception("network error")
            prefs, confidence = agent.parse_preferences("upbeat pop song")
        assert isinstance(prefs, dict)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_parse_preferences_handles_empty_query(self):
        prefs, confidence = agent.parse_preferences("")
        assert isinstance(prefs, dict)
        assert isinstance(confidence, float)

    def test_generate_explanation_returns_string_when_api_fails(self, songs):
        prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}
        recs = recommend_songs(prefs, songs, k=3)
        with patch("src.agent._get_client") as mock_client:
            mock_client.return_value.messages.create.side_effect = Exception("timeout")
            result = agent.generate_explanation("happy pop", recs)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_explanation_handles_no_recommendations(self):
        result = agent.generate_explanation("anything", [])
        assert isinstance(result, str)

    def test_rag_search_handles_empty_query(self, index):
        results = index.search("", top_n=5)
        assert isinstance(results, list)

    def test_rag_search_handles_nonsense_query(self, index):
        results = index.search("zzz123xkcd", top_n=5)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_recommend_songs_with_empty_prefs_does_not_raise(self, songs):
        results = recommend_songs({}, songs, k=5)
        assert isinstance(results, list)
        assert len(results) <= 5


# ── 3. Confidence scoring ─────────────────────────────────────────────────────

class TestConfidence:
    def test_keyword_fallback_confidence_is_in_range(self):
        """Keyword-parsed confidence must always be between 0 and 1."""
        for query in ("chill lofi", "pop happy", "something", ""):
            _, confidence = agent.parse_preferences(query)
            assert 0.0 <= confidence <= 1.0, f"Out-of-range confidence for {query!r}"

    def test_empty_query_gives_low_confidence(self):
        """An empty query should return the lowest confidence tier."""
        _, confidence = agent.parse_preferences("")
        assert confidence <= 0.3

    def test_specific_query_gives_higher_confidence_than_vague(self):
        """A query with explicit genre + mood should score higher than a vague one.
        Tests the keyword parser so no API needed."""
        _, conf_specific = agent.parse_preferences("chill lofi beats for studying")
        _, conf_vague    = agent.parse_preferences("")
        assert conf_specific > conf_vague

    def test_match_quality_labels(self, songs):
        from src.main import _match_quality
        assert _match_quality(0.80) == "strong"
        assert _match_quality(0.60) == "moderate"
        assert _match_quality(0.30) == "weak"

    def test_strong_genre_match_produces_strong_quality(self, songs):
        """A well-matched profile should always hit the 'strong' threshold."""
        from src.main import _match_quality
        prefs = {"genre": "pop", "mood": "happy", "energy": 0.82}
        results = recommend_songs(prefs, songs, k=5)
        assert _match_quality(results[0][1]) == "strong"

    def test_impossible_profile_produces_lower_quality(self, songs):
        """A profile the catalog can't satisfy should score below 'strong'."""
        from src.main import _match_quality
        # No song in the catalog is both 'classical' and energy 0.99
        prefs = {"genre": "classical", "mood": "intense", "energy": 0.99}
        results = recommend_songs(prefs, songs, k=5)
        quality = _match_quality(results[0][1])
        assert quality in ("moderate", "weak")


# ── 4. Scoring invariants ─────────────────────────────────────────────────────

class TestScoring:
    def test_scores_are_non_negative(self, songs):
        prefs = {"genre": "rock", "mood": "intense", "energy": 0.9}
        for _, score, _ in recommend_songs(prefs, songs, k=len(songs)):
            assert score >= 0.0

    def test_top_k_respects_k(self, songs):
        for k in (1, 3, 5):
            results = recommend_songs({"genre": "pop"}, songs, k=k)
            assert len(results) <= k

    def test_results_sorted_descending(self, songs):
        prefs = {"genre": "folk", "mood": "melancholic", "energy": 0.35}
        results = recommend_songs(prefs, songs, k=10)
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_diversity_reduces_artist_repetition(self, songs):
        """diversity=True should produce fewer or equal artist repeats than diversity=False."""
        prefs = {"genre": "lofi", "mood": "chill", "energy": 0.38}
        from collections import Counter

        without_div = recommend_songs(prefs, songs, k=5, diversity=False)
        with_div = recommend_songs(prefs, songs, k=5, diversity=True)

        repeats_without = sum(c - 1 for c in Counter(r[0]["artist"] for r in without_div).values())
        repeats_with = sum(c - 1 for c in Counter(r[0]["artist"] for r in with_div).values())

        assert repeats_with <= repeats_without, (
            f"Diversity made repetition worse: {repeats_with} repeats vs {repeats_without} without"
        )

    def test_genre_match_raises_score(self, songs):
        """Songs matching the requested genre should score higher on average."""
        prefs = {"genre": "jazz", "energy": 0.5}
        all_results = recommend_songs(prefs, songs, k=len(songs))
        genre_scores = [s for s in all_results if s[0]["genre"] == "jazz"]
        non_genre_scores = [s for s in all_results if s[0]["genre"] != "jazz"]
        if genre_scores and non_genre_scores:
            avg_genre = sum(r[1] for r in genre_scores) / len(genre_scores)
            avg_non = sum(r[1] for r in non_genre_scores) / len(non_genre_scores)
            assert avg_genre > avg_non
