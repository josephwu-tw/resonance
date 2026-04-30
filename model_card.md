# Model Card: Resonance — Agentic Music Recommendation System

> CodePath AI110 Final Project | Extended from **VibeFinder 1.0** (Modules 1–3)

---

## 1. Model / System Name

**Resonance v1.0**

An agentic music recommendation system that extends VibeFinder 1.0 (the Module 1–3 mini project) into a full AI pipeline. Users describe what they want to hear in plain English; the system retrieves semantically relevant songs, parses structured preferences using Claude, scores and re-ranks candidates using the original VibeFinder scoring engine, and explains the results in natural language.

---

## 2. Base Project: VibeFinder 1.0

VibeFinder 1.0 was a content-based music recommender built in Modules 1–3. Given four explicit preferences (genre, mood, energy, acoustic taste), it scored every song in a 20-track catalog using a weighted formula and returned the top-5 matches. It proved that a transparent scoring algorithm could produce sensible recommendations while revealing clear failure modes: genre-lock bias (30% genre weight dominated), silent failure when preferences couldn't be satisfied, and poor fallback when a genre had only one song in the catalog.

Resonance inherits VibeFinder's scoring engine (`src/recommender.py`) unchanged and wraps it with a Claude agentic layer and semantic retrieval.

---

## 3. Intended Use

- **Primary use:** Interactive music discovery via natural-language queries in a Streamlit web UI (`streamlit run app.py`) or CLI (`python -m src.main`).
- **Audience:** Individual users exploring the system locally with their own API key.
- **Scope:** Educational / portfolio demonstration. Not intended for production deployment, real-user data collection, or commercial music recommendation.

---

## 4. How the System Works

### Pipeline (in order)

| Step | Component | What it does |
|---|---|---|
| 1 | Input Guardrails | Rejects empty, too-short, or over-long queries before any AI call |
| 2 | `plan_query()` | Claude reads the query + catalog genre counts; returns reasoning, scoring mode, and catalog warnings |
| 3 | `SongIndex.search()` | Sentence-transformers embeds the query; returns top-15 semantically similar songs |
| 4 | `parse_preferences()` | Claude extracts structured preferences (genre, mood, energy, tags) with a self-reported confidence score |
| 5 | `recommend_songs()` | VibeFinder weighted scorer + diversity re-ranker produces top-5 |
| 6 | `generate_explanation()` | Claude explains the picks in 2–3 natural-language sentences |

### AI Models Used

| Model | Role |
|---|---|
| `claude-haiku-4-5-20251001` | Planning, preference parsing, explanation generation |
| `all-MiniLM-L6-v2` (sentence-transformers) | Song embedding and semantic retrieval |

---

## 5. Data

**Primary catalog:** `data/songs.csv` — 20 songs, 15 features each (genre, mood, energy, valence, danceability, acousticness, tempo, popularity, decade, mood tags, language, instrumental flag, artist, title, id).

**Second data source (RAG Enhancement):** `data/genre_profiles.json` — 15 genre descriptions providing rich conceptual vocabulary (e.g., "vinyl crackle", "neon-lit synths") augmented into song embeddings at index time.

**Known data limitations:**
- All 20 songs represent Western, English-language genres only. No Afrobeats, K-pop, reggaeton, or global music is represented.
- Song feature values (energy, valence, etc.) were hand-estimated for simulation — not measured from real audio.
- 20 songs is far too small for production use. Spotify's catalog has 100M+ tracks.

---

## 6. Limitations and Biases

### Catalog cultural bias
The most significant bias is in the catalog itself. All 20 songs reflect Western, English-language musical culture. A user from outside this frame would receive recommendations with no connection to their actual musical references, and the system would give no indication it had failed them.

### Genre-lock (inherited from VibeFinder)
Genre and mood matching are binary — "indie pop" scores zero against a "pop" request despite being semantically close. Genre carries 22% weight in balanced mode, meaning a genre label mismatch is a heavy penalty regardless of how well the song matches in every other dimension.

### Claude's inherited associations
The intent parser inherits whatever associations Claude learned during training. "Aggressive" music may default to rock or metal even when the user means aggressive jazz. The system cannot flag when this interpretation drift occurs.

### Popularity bias
Popularity weighting pushes results toward mainstream songs. In a larger real-world catalog this would systematically disadvantage independent and international artists.

### Explanation authority
Claude's explanations sound authoritative and personalized. A user is unlikely to question "these songs match your late-night focused energy" even if the underlying scoring weights produced a poor result. Fluent explanation can mask bad recommendations.

---

## 7. Evaluation and Testing Results

### Automated test suite
**28/28 unit tests pass** across four categories:

| Category | Tests | Focus |
|---|---|---|
| Consistency | 3 | Same input → same output every time |
| Guardrails | 9 | Bad input rejected; API failures degrade gracefully |
| Confidence scoring | 6 | Confidence bounds, quality labels, field completeness |
| Scoring invariants | 5 | Bounded scores, sorted output, diversity effect, genre lift |

### Evaluation harness
**`scripts/evaluate.py` — 21/21 predefined checks pass:**
- Avg top recommendation score: **0.770** (above 0.75 = strong match)
- Avg parse confidence (keyword fallback): **0.52**

### Few-shot specialization improvement
`scripts/compare_fewshot.py` compares the baseline and enhanced keyword parsers on 5 ambiguous queries:
- Field completeness: **+1.2 fields/query (+150%)**
- Confidence: **+0.12 (+32%)**
- Enhanced parser won on **5/5 queries**

### What the tests revealed
- The diversity re-ranker *reduces* artist repetition but cannot eliminate it in a 20-song catalog — an initial test assertion assumed zero repeats and had to be corrected to match the actual contract.
- Confidence scoring showed that queries a user considers reasonable ("play me something good") can score as low as 0.25 — revealing a gap between user expectations and parse signal.
- The "classical + aggressive + high energy" profile correctly landed on moderate/weak match quality, confirming the impossible-preference detection works.

---

## 8. Responsible AI Considerations

### Could this be misused?
Resonance is low-stakes — the worst outcome is a bad playlist. However, the same architecture pattern (natural-language intake → LLM parsing → scored retrieval → explanation generation) scales directly to higher-stakes domains: hiring, loan decisions, medical triage. In those contexts, the "explanation laundering" risk is real — fluent explanations could make biased decisions appear well-reasoned to users who cannot inspect the underlying weights.

### What is built in
- All inputs are validated and logged locally.
- No user data is stored or transmitted beyond the single-session Claude API call.
- Scoring logic is fully readable in `src/recommender.py` — no hidden model layers.
- Fallback to deterministic keyword parsing when API is unavailable.

---

## 9. AI Collaboration — Reflection

### Helpful instance
When adding confidence scoring, the suggestion was to include a `"confidence"` field directly inside the JSON Claude was already returning for preference parsing — so a single API call extracts both the structured preferences *and* a self-assessment of how clearly the query expressed them. This was cleaner than a separate API call and required only two lines of change. It worked exactly as described and became a core part of the system's reliability output.

### Flawed instance
The initial diversity test was written with the assertion `all(c == 1 for c in artist_counts.values())` — meaning no artist should appear more than once in the top-5 with diversity enabled. The reasoning was that the greedy penalty algorithm would prevent any artist from being selected twice. That reasoning was wrong: the penalty reduces the probability of repetition but doesn't guarantee uniqueness. In a catalog with only two lo-fi artists, the penalized score of the second LoRoom song was still higher than any non-lo-fi alternative. The test passed on the first profile tested, then failed on a lo-fi-heavy profile. The fix required understanding *why* the algorithm works rather than trusting the initial assertion.

### What I learned about AI-assisted development
The Claude API call in `src/agent.py` is six lines. What took real thought was deciding what to pass in, what to do when it fails, how to validate the JSON output, and how to connect the result to a system designed before Claude was part of it. The hardest part of AI system development is not the AI call — it's the architecture around it: the guardrails, the fallbacks, the logging, and the honest measurement of what the system actually does versus what it appears to do.

---

## 10. Future Work

- **Larger, diverse catalog:** Extending to 500+ songs with global genres would immediately fix the cultural bias and sparse-genre fallback problem.
- **Soft genre/mood matching:** Replace binary genre match with a similarity score so "indie pop" scores 0.7 against "pop" rather than 0.0.
- **Conflict detection:** Before scoring, check whether the user's preferences are satisfiable (e.g., "classical + aggressive + energy 0.99") and explain the tradeoff rather than silently dropping a signal.
- **Implicit feedback loop:** Let users rate or skip recommendations so the system can refine preference weights over the session — turning it from declarative ("you told me") to adaptive ("I learned from you").
