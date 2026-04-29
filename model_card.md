# Model Card: VibeFinder 1.0

---

## 1. Model Name

**VibeFinder 1.0**

A content-based music recommender that matches songs to a user's stated taste profile using a weighted scoring formula.

---

## 2. Goal / Task

VibeFinder's job is to answer one question: *given what a user says they like, which songs in the catalog are the closest match?*

It does this by scoring every song against the user's preferences and returning the top 5 highest-scoring tracks. It does **not** learn from what users skip or replay — it only uses the preferences the user explicitly provides before any music plays.

The task is essentially a ranking problem: turn a user's four preferences (genre, mood, energy, acoustic taste) into a sorted list of songs.

---

## 3. Data Used

**Catalog size:** 20 songs in `data/songs.csv`

**Features stored per song:**

| Feature | Type | What it measures |
|---|---|---|
| genre | text | Musical style label (pop, rock, lofi, etc.) |
| mood | text | Emotional character (happy, chill, intense, etc.) |
| energy | 0.0–1.0 | How loud and active a track feels |
| valence | 0.0–1.0 | Musical positivity; higher = more upbeat |
| acousticness | 0.0–1.0 | How acoustic vs. electronically produced it sounds |
| danceability | 0.0–1.0 | How suitable for dancing |
| tempo_bpm | number | Beats per minute (stored but not used in scoring) |

**Genres in the catalog:** pop, lofi, rock, ambient, jazz, synthwave, indie pop, hip-hop, electronic, classical, country, r&b, metal, folk, soul, drum & bass, bossa nova

**Known data limits:**
- Most genres have only 1 song. Lofi has 3; pop has 2. This makes the catalog uneven.
- All songs and genres reflect Western, English-language music. No Afrobeats, K-pop, reggaeton, or other global genres are included.
- Songs were invented for this simulation. Their feature values are estimated, not measured from real audio.
- 20 songs is far too small for a real recommender. Spotify's catalog has over 100 million tracks.

---

## 4. Algorithm Summary

Here is how VibeFinder decides which songs to recommend, in plain language:

**Step 1 — Score each song.** For every song in the catalog, the system asks: how well does this song match what the user asked for? It checks six things and awards points for each:

- **Genre match (30 points max):** Does the song's genre match the user's favorite? This is all-or-nothing — full points for a match, zero for a miss.
- **Energy proximity (25 points max):** How close is the song's energy to the user's target? A song at exactly the right energy gets full points. The further away it is, the fewer points it earns.
- **Mood match (20 points max):** Does the song's mood match the user's favorite? Also all-or-nothing.
- **Valence (10 points max):** How upbeat/positive is the song? No user preference is set for this — the system just rewards higher valence as a background signal.
- **Acousticness proximity (10 points max):** If the user likes acoustic music, the system rewards acoustic songs. If they don't, it rewards produced/electronic songs.
- **Danceability (5 points max):** How danceable is the song? Another background signal with no user target.

The total is a number between 0 and 1, where 1.0 would be a perfect match on everything.

**Step 2 — Rank all songs.** Sort all 20 scores from highest to lowest.

**Step 3 — Return the top 5.** The five songs with the highest scores become the recommendations.

---

## 5. Observed Behavior / Biases

**Genre lock — the biggest bias found:**
Because genre is worth 30% of the total score, a genre match alone can outrank songs that are emotionally much closer to what the user wants. In testing, "Gym Hero" (pop/intense) ranked #2 for a pop/happy user — not because it's a happy song, but because it shares the "pop" label. "Rooftop Lights" (indie pop/happy), which actually matches the user's mood and energy, ranked lower just because its genre label says "indie pop" instead of "pop."

**Silent failure on conflicting preferences:**
When the user asked for "rock + melancholic + high energy," no melancholic song appeared in the top 5. The only melancholic track in the catalog (Empty Porch, energy 0.31) is too quiet to compete with the energy score, so the mood preference is simply dropped. The system never tells the user this happened.

**Sparse fallback after niche genres:**
For a jazz/relaxed profile, the system correctly surfaces Coffee Shop Stories (#1, score 0.939). But positions 2–5 are all lofi songs with no connection to jazz. Once the single jazz song is used, the system runs out of genre-relevant options and fills with "closest energy" matches instead.

**Neutral users get arbitrary results:**
When no genre or mood preference is set, the two strongest scoring signals disappear. Rankings fall back on valence, acousticness, and danceability — small signals that were never designed to carry the recommendation alone. The "winner" in this case (Velvet Moonlight, r&b/romantic) had the closest energy to the neutral default of 0.5, which is essentially an accident.

---

## 6. Evaluation Process

Seven user profiles were run through the recommender to check whether results made intuitive sense.

**Profiles tested:**

| Profile | Expected behavior | What actually happened |
|---|---|---|
| Pop / Happy / energy 0.8 | Sunrise City at #1 | ✓ Score 0.967, felt correct |
| Rock / Intense / energy 0.92 | Storm Runner at #1 | ✓ Score 0.918, felt correct |
| Lofi / Chill / energy 0.38 (acoustic) | Lofi songs dominate | ✓ All top-3 were lofi |
| Adversarial: High Energy + Melancholic | System should struggle | ✓ It did — melancholic mood was silently dropped |
| Adversarial: No preferences set | Unpredictable output | ✓ Arbitrary winner based on minor signals |
| Adversarial: Jazz / Relaxed (1 song) | Good #1, poor fallback | ✓ Only Coffee Shop Stories was relevant |
| Adversarial: Classical + Aggressive + energy 0.97 | Impossible to satisfy | ✓ Metal song beat the classical song — energy overwhelmed genre |

**Weight experiment run:**
Genre weight was halved (0.30 → 0.15) and energy weight was doubled (0.25 → 0.50). The key change: "Rooftop Lights" jumped from #3 to #2 on the Pop/Happy profile, overtaking "Gym Hero." Rooftop Lights is a better emotional match for a happy pop listener, suggesting the original genre weight is slightly too aggressive.

**What the experiment confirmed:**
Changing a single weight shifts rankings in predictable and explainable ways. The system is transparent enough that you can trace exactly why a song moved up or down.

---

## 7. Intended Use and Non-Intended Use

### Intended Use

- **Classroom exploration:** Learning how content-based filtering works by reading the code and running it.
- **Demonstrating tradeoffs:** Showing how weight choices affect recommendations and where simple scoring logic breaks down.
- **Testing edge cases:** Running adversarial profiles to see how the algorithm handles conflicting or sparse preferences.

### Not Intended For

- **Real music discovery:** The 20-song catalog is far too small to be genuinely useful. A real listener would exhaust the relevant results immediately.
- **Personalization over time:** The system has no memory. It cannot learn from what you skip, replay, or save. Every session starts from scratch.
- **Non-Western or global music:** The catalog only contains styles common in Western markets. Using this system to recommend music for a user whose tastes fall outside pop, rock, lofi, or jazz would produce meaningless results.
- **Production deployment:** There is no authentication, rate limiting, or error handling. It is a simulation script, not a service.
- **Evaluating real songs:** Feature values were estimated by hand for a simulation. They do not reflect actual audio analysis and should not be used to characterize real tracks.

---

## 8. Ideas for Improvement

**1. Soft mood and genre matching**
Replace the binary match (it matches or it doesn't) with a similarity score. For example, "chill" and "relaxed" could score 0.7 instead of 0.0, and "indie pop" could score 0.6 against "pop." This would dramatically improve results for users whose preferred genre is underrepresented in the catalog.

**2. Conflict detection before scoring**
Before running the algorithm, check whether the user's preferences are likely contradictory — for example, if the catalog has no songs with both high energy and melancholic mood, warn the user and explain the tradeoff being made. Right now, the system silently favors one signal over another with no indication that it happened.

**3. Result diversity enforcement**
After scoring, add a rule that prevents the same genre from filling all five slots. If three lofi songs are in the top 5, replace the weakest one with the top-scoring song from a different genre. This would give users exposure to songs they might not have known to ask for.

---

## 9. Personal Reflection

**Biggest learning moment:**
The weight experiment was the clearest moment of insight. I assumed that energy would feel like the dominant feature — that's what makes a song "feel" intense or chill. But when I looked at the actual scores, genre was doing more work. "Gym Hero" kept appearing in the Pop/Happy results not because it felt happy, but because the word "pop" matched. That gap — between what a label says and what a song actually feels like — is exactly the problem that deep learning on raw audio tries to solve. A weighted formula can only see what's written in the spreadsheet; it has no idea what the music actually sounds like.

**How AI tools helped, and when I had to double-check them:**
AI was most useful for generating the expanded song catalog and drafting the initial scoring formula structure. Both saved significant time. But I had to verify both carefully. The generated songs needed energy and valence values that were internally consistent — a "frenetic drum & bass" track should have high energy and low acousticness, and I had to check that those numbers reflected that. For the scoring formula, the AI suggested a reasonable starting structure, but the specific weights came from reasoning about what features matter most to a listener — that judgment call couldn't be delegated.

**What surprised me about how simple algorithms can still "feel" like recommendations:**
The most surprising thing was that three numbers — genre match, energy proximity, mood match — were enough to produce results that felt largely correct for normal profiles. Sunrise City (#1 for pop/happy) and Storm Runner (#1 for rock/intense) both "felt right" immediately. The system had no understanding of music whatsoever; it just compared labels and numbers. That result felt like it should require more. What it actually revealed is that listeners use genre and mood as shortcuts too — and when the shortcuts align, the math looks surprisingly smart.

**What I'd try next:**
The most interesting extension would be adding implicit feedback: instead of asking the user to declare their preferences upfront, present them with a few songs and ask "does this match your mood right now?" — then infer the weights from their responses. That would turn this from a declarative system (you tell it what you want) into an adaptive one (it learns what you want), which is much closer to how Spotify's Discover Weekly actually works. Technically, it would require treating the weights as variables and updating them based on user ratings — a simple form of gradient descent in miniature.