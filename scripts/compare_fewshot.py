"""
Few-Shot Specialization Comparison — Fine-Tuning Stretch Feature
================================================================
Demonstrates measurable improvement from adding few-shot examples to the
intent parser. Compares a minimal baseline parser against the enhanced
keyword parser (which encodes the same inference patterns as the few-shot
examples in Claude's system prompt).

No API key required — runs entirely on the deterministic keyword parsers.

Run from the project root:
    python scripts/compare_fewshot.py

Metrics:
  - Fields extracted per query (field completeness)
  - Confidence score (0.0–1.0)
  - Correct genre/mood/tags for each query (qualitative check)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import _keyword_parse  # enhanced (few-shot inspired)


# ── Baseline parser (pre-few-shot behaviour) ──────────────────────────────────

def _baseline_parse(query: str):
    """Minimal keyword parser — the state of the system before few-shot patterns.
    Only matches literal genre/mood words and a handful of energy heuristics.
    """
    q = query.lower()
    prefs = {}

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

    confidence = 0.5 if prefs else 0.2
    return prefs, confidence


# ── Test queries — chosen to expose the baseline's gaps ──────────────────────

TEST_QUERIES = [
    {
        "query": "vinyl crackle lo-fi beats for late-night studying",
        "description": "Metaphorical language + activity context",
        "expected_genre": "lofi",
        "expected_mood": "chill",
        "expected_tag": "focused",
    },
    {
        "query": "music to drive to with the windows down at night",
        "description": "Metaphorical activity with no explicit genre",
        "expected_genre": None,
        "expected_mood": None,
        "expected_tag": "atmospheric",
    },
    {
        "query": "pump up songs for the gym, high energy",
        "description": "Activity-based energy inference",
        "expected_genre": None,
        "expected_mood": "energetic",
        "expected_tag": "powerful",
    },
    {
        "query": "something for a rainy heartbreak evening",
        "description": "Emotional context + weather metaphor",
        "expected_genre": None,
        "expected_mood": "melancholic",
        "expected_tag": "melancholic",
    },
    {
        "query": "classical piano for deep focus",
        "description": "Instrument-based genre inference + activity",
        "expected_genre": "classical",
        "expected_mood": None,
        "expected_tag": "focused",
    },
]

_KEY_FIELDS = ["genre", "mood", "energy", "likes_acoustic"]


def _field_completeness(prefs: dict) -> int:
    """Count how many key preference fields were extracted."""
    return sum(1 for f in _KEY_FIELDS if f in prefs)


def _has_tag(prefs: dict, tag: str) -> bool:
    return tag in prefs.get("preferred_tags", [])


# ── Run comparison ────────────────────────────────────────────────────────────

print("\n" + "═" * 70)
print("  Few-Shot Specialization Comparison")
print("  Baseline parser  vs.  Enhanced parser (few-shot inspired)")
print("═" * 70)

total_baseline_fields = 0
total_enhanced_fields = 0
total_baseline_conf   = 0.0
total_enhanced_conf   = 0.0
baseline_wins = 0
enhanced_wins = 0

for i, case in enumerate(TEST_QUERIES, 1):
    q    = case["query"]
    desc = case["description"]

    b_prefs, b_conf = _baseline_parse(q)
    e_prefs, e_conf = _keyword_parse(q)

    b_fields = _field_completeness(b_prefs)
    e_fields = _field_completeness(e_prefs)

    b_genre_ok = (case["expected_genre"] is None) or (b_prefs.get("genre") == case["expected_genre"])
    e_genre_ok = (case["expected_genre"] is None) or (e_prefs.get("genre") == case["expected_genre"])
    b_mood_ok  = (case["expected_mood"] is None) or (b_prefs.get("mood") == case["expected_mood"])
    e_mood_ok  = (case["expected_mood"] is None) or (e_prefs.get("mood") == case["expected_mood"])
    b_tag_ok   = (case["expected_tag"] is None) or _has_tag(b_prefs, case["expected_tag"])
    e_tag_ok   = (case["expected_tag"] is None) or _has_tag(e_prefs, case["expected_tag"])

    b_correct = sum([b_genre_ok, b_mood_ok, b_tag_ok])
    e_correct = sum([e_genre_ok, e_mood_ok, e_tag_ok])

    total_baseline_fields += b_fields
    total_enhanced_fields += e_fields
    total_baseline_conf   += b_conf
    total_enhanced_conf   += e_conf
    if e_correct > b_correct:
        enhanced_wins += 1
    elif b_correct > e_correct:
        baseline_wins += 1

    print(f"\n  [{i}] {desc}")
    print(f"  Query: \"{q}\"")
    print(f"  {'Metric':<24} {'Baseline':<28} {'Enhanced (few-shot)'}")
    print(f"  {'─'*24} {'─'*28} {'─'*20}")
    print(f"  {'Fields extracted':<24} {b_fields:<28} {e_fields}")
    print(f"  {'Confidence':<24} {b_conf:<28.2f} {e_conf:.2f}")
    print(f"  {'Genre correct':<24} {'✓' if b_genre_ok else '✗':<28} {'✓' if e_genre_ok else '✗'}")
    print(f"  {'Mood correct':<24} {'✓' if b_mood_ok else '✗':<28} {'✓' if e_mood_ok else '✗'}")
    print(f"  {'Tag present':<24} {'✓' if b_tag_ok else '✗':<28} {'✓' if e_tag_ok else '✗'}")
    print(f"  {'Checks correct':<24} {b_correct}/3{'':<25} {e_correct}/3")

n = len(TEST_QUERIES)
avg_b_fields = total_baseline_fields / n
avg_e_fields = total_enhanced_fields / n
avg_b_conf   = total_baseline_conf / n
avg_e_conf   = total_enhanced_conf / n
field_gain   = avg_e_fields - avg_b_fields
conf_gain    = avg_e_conf - avg_b_conf

print("\n" + "═" * 70)
print("  Aggregate Results")
print("═" * 70)
print(f"  {'Metric':<30} {'Baseline':>12} {'Enhanced':>12} {'Δ':>8}")
print(f"  {'─'*30} {'─'*12} {'─'*12} {'─'*8}")
print(f"  {'Avg fields extracted':<30} {avg_b_fields:>12.1f} {avg_e_fields:>12.1f} {field_gain:>+8.1f}")
print(f"  {'Avg confidence':<30} {avg_b_conf:>12.2f} {avg_e_conf:>12.2f} {conf_gain:>+8.2f}")
print(f"  {'Queries enhanced wins':<30} {baseline_wins:>12} {enhanced_wins:>12}")
print()
print(f"  Field completeness improvement: {field_gain:+.1f} fields/query  "
      f"({'%.0f' % (field_gain / max(avg_b_fields, 0.01) * 100)}% more)")
print(f"  Confidence improvement:         {conf_gain:+.2f}  "
      f"({'%.0f' % (conf_gain / max(avg_b_conf, 0.01) * 100)}% higher)")
print()
