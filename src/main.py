"""
Resonance — Agentic Music Recommendation System
CodePath AI110 Final Project

Run from the project root:
    python -m src.main

Flow per query
--------------
1. RAG   — semantic search narrows the catalog to the most relevant candidates
2. Agent — Claude parses the natural-language query into structured preferences
3. Score — content-based scoring engine (mini-project core) ranks candidates
4. Agent — Claude explains the final picks in natural language
"""

import sys
import logging

# Load .env before any other project imports so ANTHROPIC_API_KEY is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional; user can export the variable manually

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

_SAFE_DEFAULTS = {"energy": 0.5, "preferred_tags": []}
_RAG_TOP_N = 15
_RECOMMEND_K = 5


# ── Input validation ──────────────────────────────────────────────────────────

def _validate(query: str) -> "str | None":
    """Return an error message if the query should be rejected, else None."""
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
    songs: list,
    index: SongIndex,
    mode: str = "balanced",
) -> None:
    """Run the full RAG → parse → score → explain pipeline for one query."""
    logger.info("=== Query received: %r ===", query)

    # 1. RAG: semantic retrieval
    candidates = index.search(query, top_n=_RAG_TOP_N)
    logger.info("RAG returned %d candidates.", len(candidates))

    # 2. Agent: parse natural-language preferences
    prefs = agent.parse_preferences(query)
    if not prefs:
        logger.warning("No preferences parsed; applying safe defaults.")
        prefs = _SAFE_DEFAULTS.copy()
    else:
        prefs.setdefault("preferred_tags", [])

    # 3. Score: content-based ranking with diversity re-ranking
    results = recommend_songs(
        prefs, candidates, k=_RECOMMEND_K, mode=mode, diversity=True
    )
    logger.info("Scoring engine returned %d recommendations.", len(results))

    if not results:
        print("\n  No songs matched — try describing a different mood or genre.\n")
        return

    # 4. Agent: natural-language explanation
    explanation = agent.generate_explanation(query, results)

    # 5. Display
    print("\n" + "─" * 62)
    print(f'  Your request: "{query}"')
    print("─" * 62)
    for rank, (song, score, _reasons) in enumerate(results, 1):
        decade = song.get("release_decade", "")
        print(
            f"  {rank}. {song['title']:30s}  {song['artist']}\n"
            f"     {song['genre']:10s} / {song['mood']:12s}  "
            f"{decade:6s}  score: {score:.3f}"
        )
    print(f"\n  {explanation}")
    print("─" * 62 + "\n")
    logger.info("Delivered %d recommendations.", len(results))


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print(_BANNER)

    try:
        songs = load_songs("data/songs.csv")
    except FileNotFoundError:
        logger.error("data/songs.csv not found. Run from the project root.")
        sys.exit(1)

    logger.info("Loaded %d songs from catalog.", len(songs))
    index = SongIndex(songs)

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
            handle_query(query, songs, index)
        except Exception:
            logger.exception("Unexpected error while handling query.")
            print("  Something went wrong — please try again.\n")


if __name__ == "__main__":
    main()
