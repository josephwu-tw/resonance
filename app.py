"""
Resonance — Streamlit Web UI
Run from project root:
    streamlit run app.py
"""

import json
import logging
import os
import sys
import warnings
from collections import Counter

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Resonance",
    page_icon="🎵",
    layout="centered",
)

# ── Imports (after path is set) ───────────────────────────────────────────────

from src import agent
from src.rag import SongIndex
from src.recommender import load_songs, recommend_songs
from src.main import _validate, _match_quality

# ── Cached resources (loaded once) ───────────────────────────────────────────

@st.cache_resource(show_spinner="Loading song catalog and building index…")
def load_resources():
    songs = load_songs("data/songs.csv")
    genre_profiles = {}
    try:
        with open("data/genre_profiles.json", encoding="utf-8") as f:
            genre_profiles = json.load(f)
    except FileNotFoundError:
        pass
    index = SongIndex(songs, genre_profiles=genre_profiles)
    genre_counts = dict(Counter(s["genre"] for s in songs))
    return songs, index, genre_counts

songs, index, genre_counts = load_resources()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎵 Resonance")
    st.markdown("AI-powered music discovery. Describe your mood in plain English — Claude plans, retrieves, and explains the right songs for you.")
    st.divider()

    st.markdown("#### How it works")
    st.markdown(
        "1. **Plan** — Claude picks a scoring strategy\n"
        "2. **Retrieve** — semantic search finds candidates\n"
        "3. **Parse** — Claude extracts your preferences\n"
        "4. **Score** — VibeFinder ranks the results\n"
        "5. **Explain** — Claude explains the picks"
    )
    st.divider()

    st.markdown("#### Try these")
    examples = [
        "chill lo-fi for late night studying",
        "upbeat pop to get ready in the morning",
        "sad acoustic folk from the 2010s",
        "pump up songs for the gym",
        "something to drive to at night",
        "happy but kind of melancholic indie",
    ]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state["query_input"] = ex

    st.divider()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        st.success("Claude API connected")
    else:
        st.warning("No API key — using keyword fallback")

# ── Main UI ───────────────────────────────────────────────────────────────────

st.title("🎵 Resonance")
st.caption("Tell me what you're in the mood for — I'll find the right songs.")

query = st.text_input(
    label="Your request",
    placeholder="e.g. chill lo-fi for late night studying…",
    key="query_input",
    label_visibility="collapsed",
)

search_clicked = st.button("Find songs →", type="primary", use_container_width=True)

# ── Pipeline ──────────────────────────────────────────────────────────────────

if search_clicked and query:
    error = _validate(query)
    if error:
        st.error(error)
        st.stop()

    # Step 1: Plan
    with st.spinner("Planning…"):
        reasoning, mode, catalog_warning = agent.plan_query(query, genre_counts)

    # Step 2: RAG
    with st.spinner("Searching catalog…"):
        candidates = index.search(query, top_n=15)

    # Step 3: Parse
    with st.spinner("Parsing your preferences…"):
        prefs, parse_confidence = agent.parse_preferences(query)
        if not prefs:
            prefs = {"energy": 0.5, "preferred_tags": []}
            parse_confidence = 0.2
        else:
            prefs.setdefault("preferred_tags", [])

    # Step 4: Score
    results = recommend_songs(prefs, candidates, k=5, mode=mode, diversity=True)

    # Step 5: Explain
    with st.spinner("Generating explanation…"):
        explanation = agent.generate_explanation(query, results)

    # ── Planning callout ──────────────────────────────────────────────────────
    if reasoning:
        with st.expander("🧠 Planning step (agentic reasoning)", expanded=True):
            st.markdown(f"**Intent:** {reasoning}")
            st.markdown(f"**Scoring mode chosen:** `{mode}`")
            if catalog_warning:
                st.warning(catalog_warning, icon="⚠")

    # ── Metrics row ───────────────────────────────────────────────────────────
    if results:
        top_score = results[0][1]
        quality   = _match_quality(top_score)

        quality_color = {"strong": "🟢", "moderate": "🟡", "weak": "🔴"}

        col1, col2, col3 = st.columns(3)
        col1.metric("Parse confidence", f"{parse_confidence:.0%}")
        col2.metric("Match quality", f"{quality_color[quality]} {quality.capitalize()}")
        col3.metric("Top score", f"{top_score:.3f}")

        if quality == "weak":
            st.info(
                "Best match score is low — the catalog may not have exactly what you're "
                "looking for. Try broadening your request.",
                icon="ℹ",
            )

    # ── Results cards ─────────────────────────────────────────────────────────
    st.divider()
    st.markdown(f"**Results for:** *\"{query}\"*")

    if not results:
        st.warning("No songs matched. Try a different mood or genre.")
    else:
        for rank, (song, score, _) in enumerate(results, 1):
            with st.container(border=True):
                left, right = st.columns([4, 1])
                with left:
                    st.markdown(
                        f"**{rank}. {song['title']}** &nbsp; "
                        f"<span style='color:gray'>— {song['artist']}</span>",
                        unsafe_allow_html=True,
                    )
                    tags = song.get("mood_tags", [])
                    if isinstance(tags, str):
                        tags = [t.strip() for t in tags.split(";") if t.strip()]
                    tag_str = "  ".join(f"`{t}`" for t in tags) if tags else ""
                    st.caption(
                        f"{song['genre'].capitalize()} · {song['mood'].capitalize()} · "
                        f"{song.get('release_decade', '')}  {tag_str}"
                    )
                with right:
                    st.metric(label="Score", value=f"{score:.3f}")

        # ── Explanation ───────────────────────────────────────────────────────
        st.divider()
        st.markdown("#### Why these songs?")
        st.markdown(f"> {explanation}")

elif search_clicked and not query:
    st.warning("Please enter a request first.")

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Resonance · CodePath AI110 Final Project · "
    "Powered by Claude (Anthropic) + sentence-transformers"
)
