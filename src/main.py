"""
Resonance — Agentic Music Recommendation System
CodePath AI110 Final Project

Run from the project root:
    python -m src.main

Pipeline per query (observable steps)
--------------------------------------
1. Validate   — guardrails reject bad input before any AI call
2. Plan       — Claude decides scoring mode + flags catalog gaps  [agentic step 1]
3. Retrieve   — RAG semantic search narrows catalog to candidates  [RAG]
4. Parse      — Claude extracts structured preferences             [agentic step 2]
5. Score      — VibeFinder engine ranks candidates (chosen mode)   [mini-project core]
6. Explain    — Claude explains the final picks in plain English   [agentic step 3]
"""

import json
import logging
import sys
from collections import Counter
from typing import Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.logger_setup import setup_logging
setup_logging()

from src import agent
from src.rag import SongIndex
from src.recommender import load_songs, recommend_songs

logger = logging.getLogger(__name__)

_BANNER = """
╔══════════════════════════════════════════════════╗
║  Resonance  —  Agentic Music Recommender         ║
║  CodePath AI110 Final Project                    ║
╠══════════════════════════════════════════════════╣
║  Tell me what you're in the mood for.            ║
║  Examples:                                       ║
║    "chill lo-fi for late-night studying"         ║
║    "upbeat pop to get ready in the morning"      ║
║    "sad acoustic folk from the 2010s"            ║
║  Type 'quit' to exit.                            ║
╚══════════════════════════════════════════════════╝
"""

_SAFE_DEFAULTS: Dict = {"energy": 0.5, "preferred_tags": []}
_RAG_TOP_N  = 15
_RECOMMEND_K = 5

_PARSE_CONF_WARN  = 0.4
_MATCH_SCORE_WARN = 0.5


def _match_quality(top_score: float) -> str:
    if top_score >= 0.75:
        return "strong"
    if top_score >= 0.5:
        return "moderate"
    return "weak"


# ── Input validation ──────────────────────────────────────────────────────────

def _validate(query: str) -> Optional[str]:
    stripped = query.strip()
    if not stripped:
        return "Please describe what you'd like to hear."
    if len(stripped) < 3:
        return "That's too short — give me a bit more to work with."
    if len(stripped) > 500:
        return "Please keep your request under 500 characters."
    return None


# ── Core pipeline ─────────────────────────────────────────────────────────────

def handle_query(
    query: str,
    songs: List[Dict],
    index: SongIndex,
    genre_counts: Dict[str, int],
) -> None:
    """Run the full observable pipeline for one query."""
    logger.info("=== Query received: %r ===", query)

    # ── Step 1: Plan ──────────────────────────────────────────────────────────
    reasoning, mode, catalog_warning = agent.plan_query(query, genre_counts)

    if reasoning:
        print(f"\n  Planning…")
        print(f"  → {reasoning}")
        if catalog_warning:
            print(f"  ⚠  {catalog_warning}")
        print(f"  → Scoring mode: {mode}")

    # ── Step 2: RAG retrieval ─────────────────────────────────────────────────
    candidates = index.search(query, top_n=_RAG_TOP_N)
    logger.info("RAG returned %d candidates.", len(candidates))

    # ── Step 3: Parse preferences ─────────────────────────────────────────────
    prefs, parse_confidence = agent.parse_preferences(query)
    if not prefs:
        logger.warning("No preferences parsed; applying safe defaults.")
        prefs = _SAFE_DEFAULTS.copy()
        parse_confidence = 0.2
    else:
        prefs.setdefault("preferred_tags", [])

    if parse_confidence < _PARSE_CONF_WARN:
        print(f"\n  Note: query was vague (parse confidence: {parse_confidence:.0%}).")
        print("  Try adding a genre, mood, or energy level for better results.\n")

    # ── Step 4: Score ─────────────────────────────────────────────────────────
    results = recommend_songs(
        prefs, candidates, k=_RECOMMEND_K, mode=mode, diversity=True
    )
    logger.info("Scoring returned %d recommendations (mode=%s).", len(results), mode)

    if not results:
        print("\n  No songs matched — try describing a different mood or genre.\n")
        return

    # ── Step 5: Explain ───────────────────────────────────────────────────────
    explanation = agent.generate_explanation(query, results)

    # ── Display ───────────────────────────────────────────────────────────────
    top_score = results[0][1]
    quality   = _match_quality(top_score)

    print("\n" + "─" * 62)
    print(f'  Your request: "{query}"')
    print(
        f"  Parse confidence: {parse_confidence:.0%}  |  "
        f"Match quality: {quality}  (top score: {top_score:.3f})"
    )
    print("─" * 62)

    for rank, (song, score, _) in enumerate(results, 1):
        decade = song.get("release_decade", "")
        print(
            f"  {rank}. {song['title']:30s}  {song['artist']}\n"
            f"     {song['genre']:10s} / {song['mood']:12s}  "
            f"{decade:6s}  score: {score:.3f}"
        )

    if quality == "weak":
        print("\n  Note: best match score is low — the catalog may not have")
        print("  exactly what you're looking for. Try broadening your request.")

    print(f"\n  {explanation}")
    print("─" * 62 + "\n")

    logger.info(
        "Delivered %d recs (parse_conf=%.2f, quality=%s, top=%.3f, mode=%s).",
        len(results), parse_confidence, quality, top_score, mode,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print(_BANNER)

    try:
        songs = load_songs("data/songs.csv")
    except FileNotFoundError:
        logger.error("data/songs.csv not found. Run from the project root.")
        sys.exit(1)

    # Load genre profiles for RAG augmentation
    genre_profiles: Dict = {}
    try:
        with open("data/genre_profiles.json", encoding="utf-8") as f:
            genre_profiles = json.load(f)
        logger.info("Loaded %d genre profiles for RAG augmentation.", len(genre_profiles))
    except FileNotFoundError:
        logger.warning("data/genre_profiles.json not found; RAG will use plain embeddings.")

    logger.info("Loaded %d songs from catalog.", len(songs))
    genre_counts = dict(Counter(s["genre"] for s in songs))
    index = SongIndex(songs, genre_profiles=genre_profiles)

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if query.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        error = _validate(query)
        if error:
            print(f"  {error}\n")
            continue

        try:
            handle_query(query, songs, index, genre_counts)
        except Exception:
            logger.exception("Unexpected error while handling query.")
            print("  Something went wrong — please try again.\n")


if __name__ == "__main__":
    main()
