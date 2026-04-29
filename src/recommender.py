import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


# ── Challenge 2: Scoring mode weight presets ─────────────────────────────────
# Strategy pattern: each mode is a named dict of feature → weight.
# All weights in a mode must sum to 1.0.
SCORING_MODES: Dict[str, Dict[str, float]] = {
    # Balanced: all signals contribute; default mode.
    "balanced": {
        "genre": 0.22, "energy": 0.18, "mood": 0.14,
        "mood_tags": 0.10, "popularity": 0.09, "valence": 0.07,
        "acousticness": 0.07, "release_decade": 0.06,
        "danceability": 0.03, "language": 0.03, "is_instrumental": 0.01,
    },
    # Genre-First: genre is the dominant filter.
    "genre_first": {
        "genre": 0.40, "energy": 0.15, "mood": 0.10,
        "mood_tags": 0.08, "popularity": 0.07, "valence": 0.06,
        "acousticness": 0.06, "release_decade": 0.04,
        "danceability": 0.01, "language": 0.02, "is_instrumental": 0.01,
    },
    # Mood-First: mood label + detailed tags dominate.
    "mood_first": {
        "genre": 0.10, "energy": 0.14, "mood": 0.28,
        "mood_tags": 0.20, "popularity": 0.07, "valence": 0.08,
        "acousticness": 0.06, "release_decade": 0.03,
        "danceability": 0.01, "language": 0.02, "is_instrumental": 0.01,
    },
    # Energy-Focused: closeness to target energy is the primary signal.
    "energy_focused": {
        "genre": 0.10, "energy": 0.40, "mood": 0.12,
        "mood_tags": 0.08, "popularity": 0.08, "valence": 0.08,
        "acousticness": 0.06, "release_decade": 0.03,
        "danceability": 0.02, "language": 0.02, "is_instrumental": 0.01,
    },
}

# Ordered list of decades used for proximity scoring.
_DECADE_ORDER = ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s"]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Song:
    """A single song and its audio feature attributes."""
    # Original fields
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float
    # Challenge 1: new feature fields (defaults preserve backward compatibility)
    popularity: int = 70
    release_decade: str = "2020s"
    mood_tags: List[str] = field(default_factory=list)
    language: str = "english"
    is_instrumental: bool = False


@dataclass
class UserProfile:
    """A user's explicit music taste preferences used for content-based matching."""
    # Original fields
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool
    # Challenge 1: new preference fields (all optional with safe defaults)
    target_popularity: int = 70
    preferred_decade: str = ""
    preferred_tags: List[str] = field(default_factory=list)
    preferred_language: str = "english"
    wants_instrumental: Optional[bool] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _decade_proximity(user_decade: str, song_decade: str) -> float:
    """Return 0–1 score based on how many decades apart two era strings are."""
    if not user_decade or user_decade not in _DECADE_ORDER:
        return 0.5  # neutral when user has no era preference
    if song_decade not in _DECADE_ORDER:
        return 0.5
    gap = abs(_DECADE_ORDER.index(user_decade) - _DECADE_ORDER.index(song_decade))
    return max(0.0, 1.0 - gap * 0.25)


def _tag_overlap(user_tags: List[str], song_tags: List[str]) -> float:
    """Return 0–1 Jaccard-like overlap between two mood-tag lists."""
    if not user_tags or not song_tags:
        return 0.0
    user_set = set(t.lower() for t in user_tags)
    song_set = set(t.lower() for t in song_tags)
    return len(user_set & song_set) / max(len(user_set), len(song_set))


# ── Core scoring ──────────────────────────────────────────────────────────────

def score_song(
    user_prefs: Dict, song: Dict, mode: str = "balanced"
) -> Tuple[float, List[str]]:
    """Score one song against user_prefs under the chosen mode; return (score, reasons)."""
    w = SCORING_MODES.get(mode, SCORING_MODES["balanced"])
    score = 0.0
    reasons: List[str] = []

    # ── Original features ──────────────────────────────────────────────────

    # Genre — binary match
    if song["genre"] == user_prefs.get("genre", ""):
        pts = w["genre"]
        score += pts
        reasons.append(f"genre match: '{song['genre']}' (+{pts:.2f})")

    # Energy — proximity to user target
    target_energy = user_prefs.get("energy", 0.5)
    energy_pts = round((1.0 - abs(target_energy - song["energy"])) * w["energy"], 3)
    score += energy_pts
    reasons.append(f"energy {song['energy']:.2f} vs {target_energy:.2f} (+{energy_pts:.2f})")

    # Mood — binary match
    if song["mood"] == user_prefs.get("mood", ""):
        pts = w["mood"]
        score += pts
        reasons.append(f"mood match: '{song['mood']}' (+{pts:.2f})")

    # Valence — raw positivity signal, no user target
    score += round(song["valence"] * w["valence"], 3)

    # Acousticness — boolean user preference converted to proximity target
    acoustic_target = 0.8 if user_prefs.get("likes_acoustic", False) else 0.2
    score += round((1.0 - abs(acoustic_target - song["acousticness"])) * w["acousticness"], 3)

    # Danceability — minor background signal
    score += round(song["danceability"] * w["danceability"], 3)

    # ── Challenge 1: new features ──────────────────────────────────────────

    # Mood tags — Jaccard overlap between user's preferred tags and song's tags
    user_tags = user_prefs.get("preferred_tags", [])
    raw_tags = song.get("mood_tags", [])
    song_tags = raw_tags if isinstance(raw_tags, list) else [
        t.strip() for t in raw_tags.split(";") if t.strip()
    ]
    tag_score = _tag_overlap(user_tags, song_tags)
    tag_pts = round(tag_score * w["mood_tags"], 3)
    score += tag_pts
    if user_tags and tag_score > 0:
        matched = sorted(set(t.lower() for t in user_tags) & set(t.lower() for t in song_tags))
        reasons.append(f"tag overlap {matched} (+{tag_pts:.2f})")

    # Popularity — proximity to user's target popularity (0–100 scale)
    target_pop = user_prefs.get("target_popularity", 70)
    song_pop = int(song.get("popularity", 70))
    pop_pts = round((1.0 - abs(target_pop - song_pop) / 100) * w["popularity"], 3)
    score += pop_pts
    if w["popularity"] >= 0.06:
        reasons.append(f"popularity {song_pop} vs target {target_pop} (+{pop_pts:.2f})")

    # Release decade — ordered proximity (each decade step costs 0.25)
    preferred_decade = user_prefs.get("preferred_decade", "")
    song_decade = str(song.get("release_decade", "2020s"))
    decade_pts = round(_decade_proximity(preferred_decade, song_decade) * w["release_decade"], 3)
    score += decade_pts
    if preferred_decade and w["release_decade"] >= 0.04:
        reasons.append(f"decade {song_decade} vs {preferred_decade} (+{decade_pts:.2f})")

    # Language — binary match
    preferred_lang = user_prefs.get("preferred_language", "english")
    if str(song.get("language", "english")) == preferred_lang:
        pts = w["language"]
        score += pts
        if w["language"] >= 0.02:
            reasons.append(f"language match: {preferred_lang} (+{pts:.2f})")

    # Instrumental preference — binary match when user expresses a preference
    wants_inst = user_prefs.get("wants_instrumental")
    if wants_inst is not None:
        song_inst = bool(int(song.get("is_instrumental", 0)))
        if song_inst == wants_inst:
            pts = w["is_instrumental"]
            score += pts
            label = "instrumental" if wants_inst else "vocal"
            reasons.append(f"{label} match (+{pts:.2f})")

    return round(score, 3), reasons


# ── Challenge 3: Diversity-aware re-ranking ───────────────────────────────────

def _greedy_diverse(
    base_scores: Dict[str, Tuple[float, List[str]]],
    songs: List[Dict],
    k: int,
    artist_penalty: float,
    genre_penalty: float,
) -> List[Tuple[Dict, float, List[str]]]:
    """Greedy selection that applies artist/genre penalties after each pick."""
    results: List[Tuple[Dict, float, List[str]]] = []
    remaining = list(songs)
    artist_counts: Dict[str, int] = {}
    genre_counts: Dict[str, int] = {}

    while len(results) < k and remaining:
        # Compute penalized score for each remaining candidate
        candidates = []
        for song in remaining:
            base_score, base_reasons = base_scores[song["id"]]
            penalty = 0.0
            extra: List[str] = []

            if artist_counts.get(song["artist"], 0) >= 1:
                penalty += artist_penalty
                extra.append(
                    f"diversity: '{song['artist']}' already picked (-{artist_penalty:.2f})"
                )
            if genre_counts.get(song["genre"], 0) >= 2:
                penalty += genre_penalty
                extra.append(
                    f"diversity: '{song['genre']}' already 2× (-{genre_penalty:.2f})"
                )

            adjusted = round(max(0.0, base_score - penalty), 3)
            candidates.append((song, adjusted, base_reasons + extra))

        # Pick the highest penalized-score candidate
        candidates.sort(key=lambda x: x[1], reverse=True)
        best = candidates[0]
        results.append(best)

        artist_counts[best[0]["artist"]] = artist_counts.get(best[0]["artist"], 0) + 1
        genre_counts[best[0]["genre"]] = genre_counts.get(best[0]["genre"], 0) + 1
        remaining = [s for s in remaining if s["id"] != best[0]["id"]]

    return results


# ── Public API ────────────────────────────────────────────────────────────────

def load_songs(csv_path: str) -> List[Dict]:
    """Read songs.csv and return a list of dicts with numeric fields cast to correct types."""
    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["energy"] = float(row["energy"])
            row["tempo_bpm"] = float(row["tempo_bpm"])
            row["valence"] = float(row["valence"])
            row["danceability"] = float(row["danceability"])
            row["acousticness"] = float(row["acousticness"])
            if "popularity" in row:
                row["popularity"] = int(row["popularity"])
            if "mood_tags" in row:
                row["mood_tags"] = [t.strip() for t in row["mood_tags"].split(";") if t.strip()]
            if "is_instrumental" in row:
                row["is_instrumental"] = bool(int(row["is_instrumental"]))
            songs.append(row)
    return songs


def recommend_songs(
    user_prefs: Dict,
    songs: List[Dict],
    k: int = 5,
    mode: str = "balanced",
    diversity: bool = False,
) -> List[Tuple[Dict, float, List[str]]]:
    """Score every song and return the top-k as (song, score, reasons) tuples.

    Args:
        user_prefs: user preference dict
        songs: full catalog loaded by load_songs()
        k: number of results to return
        mode: one of 'balanced', 'genre_first', 'mood_first', 'energy_focused'
        diversity: if True, apply artist/genre diversity penalties via greedy re-ranking
    """
    # Pre-compute base scores once (used by both paths)
    base_scores = {song["id"]: score_song(user_prefs, song, mode=mode) for song in songs}

    if diversity:
        return _greedy_diverse(base_scores, songs, k, artist_penalty=0.15, genre_penalty=0.05)

    scored = [(song, *base_scores[song["id"]]) for song in songs]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


# ── OOP wrapper (used by tests) ───────────────────────────────────────────────

class Recommender:
    """OOP wrapper around the scoring logic; operates on typed Song and UserProfile objects."""

    def __init__(self, songs: List[Song]):
        self.songs = songs

    def _score_song(self, user: UserProfile, song: Song) -> float:
        """Return the weighted 0–1 score for a Song against a UserProfile."""
        genre_score = 1.0 if song.genre == user.favorite_genre else 0.0
        mood_score = 1.0 if song.mood == user.favorite_mood else 0.0
        energy_score = 1.0 - abs(user.target_energy - song.energy)
        acoustic_target = 0.8 if user.likes_acoustic else 0.2
        acousticness_score = 1.0 - abs(acoustic_target - song.acousticness)
        tag_score = _tag_overlap(user.preferred_tags, song.mood_tags)
        pop_score = 1.0 - abs(user.target_popularity - song.popularity) / 100
        decade_score = _decade_proximity(user.preferred_decade, song.release_decade)
        return (
            0.22 * genre_score
            + 0.18 * energy_score
            + 0.14 * mood_score
            + 0.10 * tag_score
            + 0.09 * pop_score
            + 0.07 * song.valence
            + 0.07 * acousticness_score
            + 0.06 * decade_score
            + 0.03 * song.danceability
        )

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return the top-k songs ranked highest to lowest by weighted score."""
        return sorted(self.songs, key=lambda s: self._score_song(user, s), reverse=True)[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Return a human-readable sentence explaining why song was recommended."""
        reasons = []
        if song.genre == user.favorite_genre:
            reasons.append(f"genre matches '{user.favorite_genre}'")
        if song.mood == user.favorite_mood:
            reasons.append(f"mood matches '{user.favorite_mood}'")
        if abs(user.target_energy - song.energy) <= 0.15:
            reasons.append(f"energy {song.energy:.2f} close to target {user.target_energy:.2f}")
        if user.preferred_tags:
            overlap = sorted(
                set(t.lower() for t in user.preferred_tags)
                & set(t.lower() for t in song.mood_tags)
            )
            if overlap:
                reasons.append(f"shared tags: {overlap}")
        if not reasons:
            reasons.append("best available match on overall profile")
        return "Recommended because: " + ", ".join(reasons)
