"""
Resonance Evaluation Harness — Test Harness Stretch Feature
============================================================
Runs predefined inputs through the pipeline and prints a structured
pass/fail report. No API key required — uses only the deterministic
scoring engine, keyword parser, and RAG index.

Run from the project root:
    python scripts/evaluate.py

Covers four stretch-feature areas:
  Section A: Core scoring correctness  (8 cases)
  Section B: Guardrail validation      (4 cases)
  Section C: RAG retrieval quality     (3 comparison queries)
  Section D: Confidence scoring        (4 cases)
"""

import json
import sys
import os
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

# Run from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.recommender import load_songs, recommend_songs
from src.rag import SongIndex
from src.agent import _keyword_parse
from src.main import _validate, _match_quality

# ── Helpers ───────────────────────────────────────────────────────────────────

PASS = "PASS"
FAIL = "FAIL"

def run_case(description: str, check: Callable[[], bool]) -> Tuple[str, str]:
    try:
        result = PASS if check() else FAIL
    except Exception as exc:
        result = FAIL
        description = f"{description} [ERROR: {exc}]"
    return result, description


def separator(title: str) -> None:
    width = 62
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


# ── Load shared fixtures ──────────────────────────────────────────────────────

songs = load_songs("data/songs.csv")
genre_counts = dict(Counter(s["genre"] for s in songs))

try:
    with open("data/genre_profiles.json", encoding="utf-8") as f:
        genre_profiles = json.load(f)
except FileNotFoundError:
    genre_profiles = {}

index_augmented = SongIndex(songs, genre_profiles=genre_profiles)
index_plain     = SongIndex(songs)   # no profiles — for RAG comparison

results_log: List[Tuple[str, str]] = []

# ══════════════════════════════════════════════════════════════════════════════
# Section A — Core scoring correctness
# ══════════════════════════════════════════════════════════════════════════════

separator("Section A: Core Scoring Correctness")

cases_A = [
    (
        "Pop/happy profile → top song is pop genre",
        lambda: recommend_songs({"genre": "pop", "mood": "happy", "energy": 0.82}, songs, k=5)[0][0]["genre"] == "pop",
    ),
    (
        "Pop/happy profile → strong match quality (score ≥ 0.75)",
        lambda: recommend_songs({"genre": "pop", "mood": "happy", "energy": 0.82}, songs, k=5)[0][1] >= 0.75,
    ),
    (
        "Lo-fi/chill profile → top-3 all chill or lofi",
        lambda: all(
            r[0]["genre"] == "lofi" or r[0]["mood"] == "chill"
            for r in recommend_songs({"genre": "lofi", "mood": "chill", "energy": 0.38}, songs, k=3)
        ),
    ),
    (
        "Jazz profile → top song is jazz genre",
        lambda: recommend_songs({"genre": "jazz", "mood": "relaxed", "energy": 0.37}, songs, k=5)[0][0]["genre"] == "jazz",
    ),
    (
        "Results are sorted descending by score",
        lambda: (lambda r: [x[1] for x in r] == sorted([x[1] for x in r], reverse=True))(
            recommend_songs({"genre": "folk", "energy": 0.4}, songs, k=10)
        ),
    ),
    (
        "k=3 returns exactly 3 results",
        lambda: len(recommend_songs({"genre": "pop"}, songs, k=3)) == 3,
    ),
    (
        "Diversity mode produces ≤ same artist repeats as no-diversity",
        lambda: (
            lambda nd, wd: (
                sum(c - 1 for c in Counter(r[0]["artist"] for r in wd).values()) <=
                sum(c - 1 for c in Counter(r[0]["artist"] for r in nd).values())
            )
        )(
            recommend_songs({"genre": "lofi", "mood": "chill"}, songs, k=5, diversity=False),
            recommend_songs({"genre": "lofi", "mood": "chill"}, songs, k=5, diversity=True),
        ),
    ),
    (
        "Impossible profile (classical + energy 0.99) → moderate or weak quality",
        lambda: _match_quality(
            recommend_songs({"genre": "classical", "mood": "intense", "energy": 0.99}, songs, k=5)[0][1]
        ) in ("moderate", "weak"),
    ),
]

for desc, check in cases_A:
    status, label = run_case(desc, check)
    results_log.append((status, label))
    icon = "✓" if status == PASS else "✗"
    print(f"  {icon} {status}  {label}")

# ══════════════════════════════════════════════════════════════════════════════
# Section B — Guardrail validation
# ══════════════════════════════════════════════════════════════════════════════

separator("Section B: Guardrail Validation")

cases_B = [
    ("Empty string rejected", lambda: _validate("") is not None),
    ("Whitespace-only rejected", lambda: _validate("   ") is not None),
    ("2-char query rejected", lambda: _validate("hi") is not None),
    (f"501-char query rejected", lambda: _validate("x" * 501) is not None),
    ("Valid query accepted", lambda: _validate("chill lo-fi for studying") is None),
    ("Nonsense query still returns 5 results",
     lambda: len(recommend_songs({}, songs, k=5)) == 5),
]

for desc, check in cases_B:
    status, label = run_case(desc, check)
    results_log.append((status, label))
    icon = "✓" if status == PASS else "✗"
    print(f"  {icon} {status}  {label}")

# ══════════════════════════════════════════════════════════════════════════════
# Section C — RAG Retrieval Quality (augmented vs plain)
# ══════════════════════════════════════════════════════════════════════════════

separator("Section C: RAG Retrieval Quality — Augmented vs Plain")

rag_queries = [
    (
        "vinyl crackle late night studying",
        "lofi",
        "Conceptual lo-fi query: augmented index retrieves lofi in top-3",
    ),
    (
        "neon synth night drive 80s",
        "synthwave",
        "Metaphorical synthwave query: augmented retrieves synthwave in top-3",
    ),
    (
        "coffee shop jazz background",
        "jazz",
        "Activity-based jazz query: augmented retrieves jazz in top-3",
    ),
]

print(f"\n  {'Query':<36} {'Plain top-3 genres':<28} {'Aug top-3 genres':<28} {'Aug wins?'}")
print(f"  {'─'*36} {'─'*28} {'─'*28} {'─'*9}")

for query, target_genre, desc in rag_queries:
    comparison = index_augmented.compare_retrieval(query, top_n=3)
    plain_genres   = [g for _, g, _ in comparison["plain"]]
    aug_genres     = [g for _, g, _ in comparison["augmented"]]
    aug_wins = target_genre in aug_genres
    plain_wins = target_genre in plain_genres

    aug_label   = ", ".join(aug_genres)
    plain_label = ", ".join(plain_genres)
    win_label   = "YES" if (aug_wins and not plain_wins) else ("BOTH" if (aug_wins and plain_wins) else "NO")

    print(f"  {query:<36} {plain_label:<28} {aug_label:<28} {win_label}")

    status = PASS if aug_wins else FAIL
    results_log.append((status, desc))
    icon = "✓" if status == PASS else "✗"
    print(f"  {icon} {status}  {desc}")

# ══════════════════════════════════════════════════════════════════════════════
# Section D — Confidence Scoring
# ══════════════════════════════════════════════════════════════════════════════

separator("Section D: Confidence Scoring")

cases_D = [
    (
        "Empty query → confidence ≤ 0.2",
        lambda: _keyword_parse("")[1] <= 0.2,
    ),
    (
        "Specific genre + activity query → confidence > 0.4",
        lambda: _keyword_parse("chill lofi for studying")[1] > 0.4,
    ),
    (
        "Specific query scores higher confidence than empty query",
        lambda: _keyword_parse("upbeat pop morning energy")[1] > _keyword_parse("")[1],
    ),
    (
        "match_quality labels: 0.8→strong, 0.6→moderate, 0.3→weak",
        lambda: (
            _match_quality(0.80) == "strong"
            and _match_quality(0.60) == "moderate"
            and _match_quality(0.30) == "weak"
        ),
    ),
]

for desc, check in cases_D:
    status, label = run_case(desc, check)
    results_log.append((status, label))
    icon = "✓" if status == PASS else "✗"
    print(f"  {icon} {status}  {label}")

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════

separator("Summary")

passed = sum(1 for s, _ in results_log if s == PASS)
total  = len(results_log)
pct    = passed / total * 100

# Average top score across core scoring cases
sample_scores = [
    recommend_songs({"genre": "pop", "mood": "happy", "energy": 0.82}, songs, k=5)[0][1],
    recommend_songs({"genre": "lofi", "mood": "chill", "energy": 0.38}, songs, k=5)[0][1],
    recommend_songs({"genre": "jazz", "mood": "relaxed", "energy": 0.37}, songs, k=5)[0][1],
    recommend_songs({"genre": "folk", "mood": "melancholic", "energy": 0.35}, songs, k=5)[0][1],
]
avg_score = sum(sample_scores) / len(sample_scores)

# Average confidence from keyword parser on representative queries
sample_confs = [
    _keyword_parse("chill lofi for studying")[1],
    _keyword_parse("upbeat pop morning energy")[1],
    _keyword_parse("sad acoustic folk")[1],
    _keyword_parse("pump up workout gym")[1],
    _keyword_parse("")[1],
]
avg_conf = sum(sample_confs) / len(sample_confs)

print(f"\n  Result:            {passed}/{total} passed ({pct:.0f}%)")
print(f"  Avg top score:     {avg_score:.3f}  (above 0.75 = strong match)")
print(f"  Avg confidence:    {avg_conf:.2f}   (keyword parser; Claude scores higher)")
print()

if passed == total:
    print("  All checks passed.")
else:
    failed = [(s, d) for s, d in results_log if s == FAIL]
    print(f"  {len(failed)} check(s) failed:")
    for _, desc in failed:
        print(f"    ✗ {desc}")

print()
sys.exit(0 if passed == total else 1)
