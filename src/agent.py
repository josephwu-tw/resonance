"""
Agentic layer — Claude-powered intent parsing and explanation generation.

Two responsibilities:
  1. parse_preferences()  — turn a free-text user query into a structured
                            preference dict that the scoring engine understands.
  2. generate_explanation() — given the final recommendations, produce a
                              concise natural-language explanation of why
                              these songs were chosen.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import anthropic as _anthropic
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False
    logger.warning(
        "anthropic package not installed; agent will use rule-based fallbacks. "
        "Install it with: pip install anthropic"
    )

_MODEL = "claude-haiku-4-5-20251001"
_client: Optional["_anthropic.Anthropic"] = None

_PARSE_SYSTEM = """\
You are a music preference parser. Given a user's natural-language music request,
extract structured preferences and return them as a single JSON object — no extra text.

Fields (use null when the user didn't imply a value):
{
  "genre":             string or null,   // pop, rock, lofi, folk, jazz, hip-hop, classical, r&b, edm, metal
  "mood":              string or null,   // happy, chill, melancholic, energetic, intense, romantic, focused
  "energy":            float or null,    // 0.0 (very calm) to 1.0 (very energetic)
  "likes_acoustic":    bool or null,
  "preferred_decade":  string or null,   // 1960s … 2020s
  "preferred_tags":    [string],         // vibe words: uplifting, nostalgic, focused, peaceful, euphoric …
  "target_popularity": int or null       // 0–100; how mainstream (100 = very popular)
}"""

_EXPLAIN_SYSTEM = """\
You are a friendly music recommendation assistant. Given a user's request and the
songs selected for them, write a 2–3 sentence explanation of why these specific songs
match. Be specific about musical qualities (genre, mood, energy, era). No bullet points."""


def _get_client() -> "_anthropic.Anthropic":
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. "
                "Add it to a .env file or export it in your shell."
            )
        _client = _anthropic.Anthropic(api_key=api_key)
    return _client


def _keyword_parse(query: str) -> Dict:
    """Rule-based fallback when the Anthropic SDK is unavailable."""
    q = query.lower()
    prefs: Dict = {}

    for genre in ("pop", "rock", "lofi", "folk", "jazz", "hip-hop", "classical", "r&b", "edm", "metal"):
        if genre in q:
            prefs["genre"] = genre
            break

    for mood in ("happy", "chill", "sad", "melancholic", "energetic", "intense", "romantic", "focused"):
        if mood in q:
            prefs["mood"] = mood
            break

    if any(w in q for w in ("calm", "relax", "sleep", "study", "quiet", "peaceful")):
        prefs.setdefault("energy", 0.3)
        prefs.setdefault("mood", "chill")
    elif any(w in q for w in ("hype", "workout", "pump", "intense", "fast")):
        prefs.setdefault("energy", 0.85)

    if any(w in q for w in ("acoustic", "guitar", "unplugged")):
        prefs["likes_acoustic"] = True

    return prefs


def parse_preferences(user_query: str) -> Dict:
    """Extract a structured preference dict from a natural-language query.

    Uses Claude when available; falls back to keyword heuristics.
    Never raises — returns an empty dict on total failure.
    """
    logger.info("Parsing preferences from query: %r", user_query)

    if _HAS_ANTHROPIC:
        try:
            response = _get_client().messages.create(
                model=_MODEL,
                max_tokens=256,
                system=_PARSE_SYSTEM,
                messages=[{"role": "user", "content": user_query}],
            )
            raw = response.content[0].text.strip()

            # Strip markdown code fences if the model added them
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            prefs = json.loads(raw)
            # Drop null values so scoring engine uses its own defaults
            prefs = {k: v for k, v in prefs.items() if v is not None}
            # Ensure preferred_tags is always a list
            prefs.setdefault("preferred_tags", [])
            logger.info("Claude parsed preferences: %s", prefs)
            return prefs

        except json.JSONDecodeError as exc:
            logger.warning("Claude returned non-JSON (%s); falling back to keywords.", exc)
        except Exception as exc:
            logger.warning("Claude preference parsing failed (%s); falling back to keywords.", exc)

    # Fallback
    prefs = _keyword_parse(user_query)
    logger.info("Keyword-parsed preferences: %s", prefs)
    return prefs


def generate_explanation(
    user_query: str,
    recommendations: List[Tuple[Dict, float, List[str]]],
) -> str:
    """Generate a natural-language explanation for the recommendations.

    Uses Claude when available; falls back to a template string.
    Never raises.
    """
    if not recommendations:
        return "No songs matched your request."

    songs_block = "\n".join(
        f'{i+1}. "{r[0]["title"]}" by {r[0]["artist"]} '
        f'(genre: {r[0]["genre"]}, mood: {r[0]["mood"]}, '
        f'energy: {r[0].get("energy", "?")})'
        for i, r in enumerate(recommendations)
    )
    user_content = (
        f'User request: "{user_query}"\n\nRecommended songs:\n{songs_block}'
    )

    logger.info("Generating explanation for %d recommendations.", len(recommendations))

    if _HAS_ANTHROPIC:
        try:
            response = _get_client().messages.create(
                model=_MODEL,
                max_tokens=200,
                system=_EXPLAIN_SYSTEM,
                messages=[{"role": "user", "content": user_content}],
            )
            explanation = response.content[0].text.strip()
            logger.info("Explanation generated (%d chars).", len(explanation))
            return explanation
        except Exception as exc:
            logger.warning("Claude explanation failed (%s); using template.", exc)

    # Fallback template
    titles = ", ".join(
        f'"{r[0]["title"]}" by {r[0]["artist"]}' for r in recommendations[:3]
    )
    return f"Based on your request, we found: {titles}."
