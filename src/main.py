"""
Command line runner for the Music Recommender Simulation.

Run from the project root:
    python -m src.main
"""

from src.recommender import load_songs, recommend_songs, SCORING_MODES

try:
    from tabulate import tabulate
    _HAS_TABULATE = True
except ImportError:
    _HAS_TABULATE = False


# ── Challenge 4: Visual summary table ────────────────────────────────────────

def print_table(
    label: str,
    user_prefs: dict,
    results: list,
    mode: str = "balanced",
    diversity: bool = False,
) -> None:
    """Print a formatted table of recommendations using tabulate (or ASCII fallback)."""
    mode_tag = f"[{mode}{'  diversity' if diversity else ''}]"
    header = f"  {label}  {mode_tag}"
    print(f"\n{'─'*70}")
    print(header)
    print(f"  Prefs: {user_prefs}")
    print(f"{'─'*70}")

    rows = []
    for rank, (song, score, reasons) in enumerate(results, 1):
        reason_text = "\n".join(f"• {r}" for r in reasons)
        rows.append([
            rank,
            song["title"],
            f"{song['genre']} / {song['mood']}",
            f"{song.get('popularity', '?')}  {song.get('release_decade', '?')}",
            f"{score:.3f}",
            reason_text,
        ])

    headers = ["#", "Title", "Genre / Mood", "Pop / Era", "Score", "Reasons"]

    if _HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))
    else:
        # Plain ASCII fallback when tabulate is not installed
        col_w = [3, 22, 22, 12, 6, 48]
        sep = "+" + "+".join("-" * (w + 2) for w in col_w) + "+"

        def fmt_row(cells):
            lines_per_cell = [str(c).split("\n") for c in cells]
            row_height = max(len(lines) for lines in lines_per_cell)
            out = []
            for i in range(row_height):
                parts = []
                for j, lines in enumerate(lines_per_cell):
                    text = lines[i] if i < len(lines) else ""
                    parts.append(f" {text:<{col_w[j]}} ")
                out.append("|" + "|".join(parts) + "|")
            return "\n".join(out)

        print(sep)
        print(fmt_row(headers))
        print(sep.replace("-", "="))
        for row in rows:
            print(fmt_row(row))
            print(sep)


# ── Demo profiles ─────────────────────────────────────────────────────────────

def main() -> None:
    songs = load_songs("data/songs.csv")
    print(f"Loaded {len(songs)} songs  |  Available modes: {list(SCORING_MODES)}")

    # ── Challenge 1: New features in action ───────────────────────────────────
    # Profile uses popularity target, era preference, and mood tags
    pop_profile = {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.8,
        "target_popularity": 80,         # Challenge 1: wants mainstream/popular songs
        "preferred_decade": "2020s",     # Challenge 1: prefers recent music
        "preferred_tags": ["uplifting", "euphoric"],  # Challenge 1: detailed mood tags
    }

    # ── Challenge 2: Same profile, three different scoring modes ─────────────
    for mode in ("balanced", "genre_first", "energy_focused"):
        results = recommend_songs(pop_profile, songs, k=5, mode=mode)
        print_table("Pop / Happy / High Energy", pop_profile, results, mode=mode)

    # ── Challenge 2: Mood-first mode highlights mood-tag power ───────────────
    nostalgic_profile = {
        "genre": "folk",
        "mood": "melancholic",
        "energy": 0.35,
        "preferred_tags": ["melancholic", "nostalgic"],
        "preferred_decade": "2010s",
        "likes_acoustic": True,
        "target_popularity": 45,
    }
    results = recommend_songs(nostalgic_profile, songs, k=5, mode="mood_first")
    print_table("Folk / Melancholic / Nostalgic", nostalgic_profile, results, mode="mood_first")

    # ── Challenge 3: Diversity penalty comparison ─────────────────────────────
    # Lofi profile has 3 lofi songs in catalog → diversity penalty kicks in
    lofi_profile = {
        "genre": "lofi",
        "mood": "chill",
        "energy": 0.38,
        "likes_acoustic": True,
        "preferred_tags": ["peaceful", "focused"],
        "target_popularity": 60,
    }

    no_div = recommend_songs(lofi_profile, songs, k=5, mode="balanced", diversity=False)
    with_div = recommend_songs(lofi_profile, songs, k=5, mode="balanced", diversity=True)

    print_table("Lofi / Chill — no diversity", lofi_profile, no_div, mode="balanced", diversity=False)
    print_table("Lofi / Chill — WITH diversity", lofi_profile, with_div, mode="balanced", diversity=True)

    print()


if __name__ == "__main__":
    main()
