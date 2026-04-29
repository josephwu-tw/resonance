"""
Agentic layer — Claude-powered intent parsing, multi-step planning, and explanation.

Stretch features implemented here:
  - Fine-Tuning/Specialization: few-shot examples in _PARSE_SYSTEM teach Claude to
    handle activity-based ("studying"), metaphorical ("vinyl crackle"), and
    contradictory ("happy but sad") queries that the baseline prompt handled poorly.
  - Agentic Workflow Enhancement: plan_query() adds an observable planning step —
    Claude decides which scoring mode fits the query and flags catalog gaps before
    any song is scored, making the reasoning chain visible.
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

# ── Prompts ───────────────────────────────────────────────────────────────────

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
  "target_popularity": int or null,      // 0–100; how mainstream (100 = very popular)
  "confidence":        float             // 0.0–1.0: how clearly the query expressed preferences.
                                         // 1.0 = every field is explicit; 0.0 = too vague to parse.
}

Few-shot examples — use these to handle activity-based, metaphorical, and conflicting queries:

User: "vinyl crackle lo-fi beats for late-night studying"
{"genre":"lofi","mood":"chill","energy":0.3,"likes_acoustic":false,"preferred_decade":null,"preferred_tags":["focused","peaceful","nostalgic"],"target_popularity":null,"confidence":0.92}

User: "music to drive to with the windows down at night"
{"genre":null,"mood":"chill","energy":0.55,"likes_acoustic":false,"preferred_decade":null,"preferred_tags":["atmospheric","nocturnal"],"target_popularity":null,"confidence":0.65}

User: "happy but kind of sad, indie folk feel"
{"genre":"folk","mood":"melancholic","energy":0.4,"likes_acoustic":true,"preferred_decade":null,"preferred_tags":["bittersweet","nostalgic"],"target_popularity":40,"confidence":0.78}

User: "pump up songs for the gym"
{"genre":null,"mood":"energetic","energy":0.9,"likes_acoustic":false,"preferred_decade":null,"preferred_tags":["powerful","uplifting"],"target_popularity":null,"confidence":0.82}"""

_PLAN_SYSTEM = """\
You are a music recommendation strategist. Analyze the user's request and decide the
best approach. Return a single JSON object — no extra text.

Scoring modes:
  "balanced"       — all features contribute equally (default)
  "genre_first"    — genre is the dominant filter
  "mood_first"     — mood label and vibe tags dominate
  "energy_focused" — energy proximity is the primary signal

Return:
{
  "reasoning":       string,        // 1 sentence: what is the user's primary intent?
  "scoring_mode":    string,        // one of the four modes above
  "catalog_warning": string or null // flag if the requested genre appears sparse or absent
}"""

_EXPLAIN_SYSTEM = """\
You are a friendly music recommendation assistant. Given a user's request and the
songs selected for them, write a 2–3 sentence explanation of why these specific songs
match. Be specific about musical qualities (genre, mood, energy, era). No bullet points."""


# ── Client ────────────────────────────────────────────────────────────────────

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


def _parse_json(raw: str) -> dict:
    """Strip markdown fences and parse JSON."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


# ── Keyword fallback ──────────────────────────────────────────────────────────

_CONFIDENCE_EMPTY = 0.2

def _keyword_parse(query: str) -> Tuple[Dict, float]:
    """Enhanced rule-based fallback inspired by few-shot patterns.

    Handles activity context, metaphorical language, and emotional inference
    beyond simple genre/mood keyword matching. Confidence is derived from
    field completeness rather than a fixed value.
    """
    q = query.lower()
    prefs: Dict = {"preferred_tags": []}

    # Genre — direct keyword match
    for genre in ("pop", "rock", "lofi", "folk", "jazz", "hip-hop", "classical", "r&b", "edm", "metal", "ambient", "synthwave", "soul", "country"):
        if genre in q:
            prefs["genre"] = genre
            break

    # Metaphorical genre inference (few-shot inspired)
    if "genre" not in prefs:
        if any(w in q for w in ("vinyl", "crackle", "tape hiss", "lo-fi", "lofi")):
            prefs["genre"] = "lofi"
        elif any(w in q for w in ("piano", "orchestra", "symphony", "classical", "sonata")):
            prefs["genre"] = "classical"
            prefs["likes_acoustic"] = True
        elif any(w in q for w in ("beats", "flow", "bars", "rap", "trap")):
            prefs["genre"] = "hip-hop"
        elif any(w in q for w in ("neon", "synthwave", "80s synth", "retro synth")):
            prefs["genre"] = "synthwave"

    # Mood — direct keyword match
    for mood in ("happy", "chill", "sad", "melancholic", "energetic", "intense", "romantic", "focused", "peaceful"):
        if mood in q:
            prefs["mood"] = mood
            break

    # Activity-based energy + mood inference (few-shot inspired)
    if any(w in q for w in ("study", "studying", "focus", "concentrate", "reading", "working", "homework")):
        prefs.setdefault("energy", 0.3)
        prefs.setdefault("mood", "chill")
        prefs["preferred_tags"] = list(set(prefs["preferred_tags"] + ["focused", "peaceful"]))
    elif any(w in q for w in ("gym", "workout", "running", "exercise", "sport", "pump", "hype")):
        prefs.setdefault("energy", 0.9)
        prefs.setdefault("mood", "energetic")
        prefs["preferred_tags"] = list(set(prefs["preferred_tags"] + ["powerful", "uplifting"]))
    elif any(w in q for w in ("sleep", "sleeping", "bedtime", "falling asleep")):
        prefs.setdefault("energy", 0.15)
        prefs.setdefault("mood", "chill")
        prefs["preferred_tags"] = list(set(prefs["preferred_tags"] + ["peaceful"]))
    elif any(w in q for w in ("party", "dance", "dancing", "club", "rave")):
        prefs.setdefault("energy", 0.85)
        prefs.setdefault("mood", "energetic")
        prefs["preferred_tags"] = list(set(prefs["preferred_tags"] + ["euphoric", "uplifting"]))
    elif any(w in q for w in ("drive", "driving", "road trip", "commute", "windows down")):
        prefs.setdefault("energy", 0.55)
        prefs["preferred_tags"] = list(set(prefs["preferred_tags"] + ["atmospheric"]))
    elif any(w in q for w in ("relax", "calm", "unwind", "chill out", "quiet", "peaceful")):
        prefs.setdefault("energy", 0.3)
        prefs.setdefault("mood", "chill")

    # Time-of-day context
    if any(w in q for w in ("late night", "midnight", "2am", "after hours", "night time")):
        prefs.setdefault("energy", 0.35)
        prefs["preferred_tags"] = list(set(prefs["preferred_tags"] + ["nocturnal", "peaceful"]))
    elif any(w in q for w in ("morning", "wake up", "sunrise", "early")):
        prefs.setdefault("energy", 0.65)
        prefs.setdefault("mood", "happy")

    # Emotional context
    if any(w in q for w in ("heartbreak", "crying", "breakup", "lonely", "grief", "loss")):
        prefs.setdefault("mood", "melancholic")
        prefs["preferred_tags"] = list(set(prefs["preferred_tags"] + ["melancholic", "nostalgic"]))
    elif any(w in q for w in ("nostalgic", "throwback", "retro", "old school", "memory")):
        prefs["preferred_tags"] = list(set(prefs["preferred_tags"] + ["nostalgic"]))

    # Acoustic preference
    if any(w in q for w in ("acoustic", "guitar", "unplugged", "live", "campfire")):
        prefs["likes_acoustic"] = True

    # Confidence: proportion of key fields filled (few-shot inspired weighting)
    key_fields = ["genre", "mood", "energy", "likes_acoustic"]
    filled = sum(1 for f in key_fields if f in prefs)
    tag_bonus = 0.1 if prefs.get("preferred_tags") else 0.0
    confidence = min(1.0, filled * 0.2 + tag_bonus) if filled > 0 else _CONFIDENCE_EMPTY

    return prefs, round(confidence, 2)


# ── Public API ────────────────────────────────────────────────────────────────

def parse_preferences(user_query: str) -> Tuple[Dict, float]:
    """Extract structured preferences and a confidence score from a natural-language query.

    Returns:
        prefs      — musical preference dict (genre, mood, energy, …)
        confidence — 0.0–1.0 from Claude's self-assessment or field-count heuristic.
    """
    logger.info("Parsing preferences from query: %r", user_query)

    if _HAS_ANTHROPIC:
        try:
            response = _get_client().messages.create(
                model=_MODEL,
                max_tokens=300,
                system=_PARSE_SYSTEM,
                messages=[{"role": "user", "content": user_query}],
            )
            parsed = _parse_json(response.content[0].text)
            confidence = float(parsed.pop("confidence", 0.7))
            confidence = max(0.0, min(1.0, confidence))
            prefs = {k: v for k, v in parsed.items() if v is not None}
            prefs.setdefault("preferred_tags", [])
            logger.info("Claude parsed (confidence=%.2f): %s", confidence, prefs)
            return prefs, confidence
        except json.JSONDecodeError as exc:
            logger.warning("Claude returned non-JSON (%s); using keyword fallback.", exc)
        except Exception as exc:
            logger.warning("Claude parse failed (%s); using keyword fallback.", exc)

    prefs, confidence = _keyword_parse(user_query)
    logger.info("Keyword-parsed (confidence=%.2f): %s", confidence, prefs)
    return prefs, confidence


def plan_query(
    user_query: str,
    genre_counts: Dict[str, int],
) -> Tuple[str, str, Optional[str]]:
    """Agentic planning step — observable intermediate reasoning before scoring.

    Asks Claude to decide:
      - What is the user's primary intent? (reasoning)
      - Which scoring mode fits best?     (scoring_mode)
      - Is the requested genre sparse?    (catalog_warning)

    Returns (reasoning, scoring_mode, catalog_warning).
    Falls back to ("", "balanced", None) on any failure.
    """
    catalog_summary = ", ".join(
        f"{genre}({count})" for genre, count in sorted(genre_counts.items())
    )
    user_content = (
        f'User query: "{user_query}"\n'
        f"Catalog genre counts: {catalog_summary}"
    )

    logger.info("Planning query strategy for: %r", user_query)

    if _HAS_ANTHROPIC:
        try:
            response = _get_client().messages.create(
                model=_MODEL,
                max_tokens=200,
                system=_PLAN_SYSTEM,
                messages=[{"role": "user", "content": user_content}],
            )
            parsed = _parse_json(response.content[0].text)
            reasoning = parsed.get("reasoning", "")
            mode = parsed.get("scoring_mode", "balanced")
            warning = parsed.get("catalog_warning") or None
            if mode not in ("balanced", "genre_first", "mood_first", "energy_focused"):
                mode = "balanced"
            logger.info("Plan: mode=%s, warning=%s", mode, warning)
            return reasoning, mode, warning
        except Exception as exc:
            logger.warning("plan_query failed (%s); using defaults.", exc)

    return "", "balanced", None


def generate_explanation(
    user_query: str,
    recommendations: List[Tuple[Dict, float, List[str]]],
) -> str:
    """Generate a natural-language explanation for the final recommendations."""
    if not recommendations:
        return "No songs matched your request."

    songs_block = "\n".join(
        f'{i+1}. "{r[0]["title"]}" by {r[0]["artist"]} '
        f'(genre: {r[0]["genre"]}, mood: {r[0]["mood"]}, '
        f'energy: {r[0].get("energy", "?")})'
        for i, r in enumerate(recommendations)
    )
    user_content = f'User request: "{user_query}"\n\nRecommended songs:\n{songs_block}'

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
            logger.warning("Explanation failed (%s); using template.", exc)

    titles = ", ".join(
        f'"{r[0]["title"]}" by {r[0]["artist"]}' for r in recommendations[:3]
    )
    return f"Based on your request, we found: {titles}."
