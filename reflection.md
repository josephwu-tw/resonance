# Reflection: Profile Comparisons

## Pop/Happy vs. Rock/Intense

These two profiles share zero top-5 results, which is exactly what you would hope for — they represent genuinely different listening moods. The pop profile surfaces bright, high-valence songs with close energy around 0.8, while the rock profile surfaces loud, fast songs near energy 0.92.

The one thing that stands out: "Gym Hero" (pop/intense) appears in *both* lists. For the pop user it ranks #2 because of the genre match; for the rock user it ranks #2 because of the mood match. It's a crossover song that the system correctly identifies as relevant to both contexts, just for different reasons. This makes intuitive sense — Gym Hero is a high-energy pop track, which genuinely could appeal to someone pumping iron whether they prefer pop or rock.

---

## Rock/Intense vs. Lofi/Chill

These profiles are essentially opposites on the energy axis (0.92 vs. 0.38) and the acoustic axis (electric vs. acoustic), and their results reflect that completely. Not a single song overlaps. The rock list is all high-BPM, loud tracks; the lofi list is all quiet, acoustic, mid-tempo songs.

The lofi list feels *more satisfying* than the rock list, and that's a data problem, not a logic problem. The catalog has 3 lofi songs but only 1 rock song. After Storm Runner takes the #1 slot for rock, the system has to reach into pop and electronic to fill the remaining spots — genres the user didn't ask for. A real catalog would have dozens of rock songs to choose from, and the rankings would make more sense.

---

## Lofi/Chill vs. Jazz/Relaxed (Adversarial)

These two profiles are close in energy (0.38 vs. 0.37) and acoustic preference (both like acoustic), but differ in genre and mood. The lofi profile gets a rich, varied top-5 entirely within its genre. The jazz profile gets one perfect match (#1: Coffee Shop Stories at 0.939) and then four lofi songs filling the gaps.

This comparison reveals a core weakness: **the system can only recommend what exists in the catalog.** If your genre is well-represented, you get good variety. If you're a jazz listener and there's only one jazz song, you get one jazz recommendation and then a list of "closest other things," which may not feel related at all. A real recommender would expand its catalog or tell you "we don't have much jazz — here's what's most similar."

---

## High Energy + Melancholic vs. Classical + Aggressive (Both Adversarial)

Both of these profiles ask for combinations that the catalog can't satisfy, but they fail in different ways.

- **High Energy + Melancholic:** The system silently ignores the melancholic mood and returns high-energy songs. The single melancholic track in the catalog (Empty Porch, energy 0.31) is too quiet to compete with the energy score, so it never shows up. The user gets what they half-asked for (high energy rock) with no explanation for why the mood preference was dropped.

- **Classical + Aggressive:** The system has to choose between a classical song that's too quiet (Morning Sonata, energy 0.22 vs. target 0.97) and a metal song that's the right energy but wrong genre (Shatter Glass). Shatter Glass wins because an energy mismatch of 0.75 costs more points than not having the genre match at all. So the user who asked for "classical" gets metal as their top recommendation. Again, no warning.

Both cases show the same lesson: **a system that scores and ranks without checking whether any result actually fits the request will always return *something*, even when the real answer is "we can't help you with that."** Knowing when *not* to recommend is just as important as knowing what to recommend.

---

## Why Does "Gym Hero" Keep Appearing?

"Gym Hero" (pop/intense, energy 0.93, danceability 0.88, valence 0.77) shows up in the top-5 for four out of seven profiles tested. Here's the plain-language reason: Gym Hero is a high-energy pop song. "High energy" is the second most important feature in the scoring formula (25%), and many user profiles ask for high energy. Even when the genre or mood doesn't match, the energy score alone is enough to keep Gym Hero near the top.

Think of it like a restaurant menu: if most customers say they want "something filling," the kitchen will keep putting the large entrée on the daily specials, even for tables that asked for something light. The recommendation is technically responding to the request, but it's missing the spirit of it.

The fix is either to lower the energy weight relative to mood and genre, or to add a diversity rule that prevents the same song from dominating multiple profiles. The weight experiment confirmed this — when genre weight was halved and energy doubled, Gym Hero dropped from #2 to #3 on the Pop/Happy profile, replaced by Rooftop Lights, which actually matches the "happy" mood. Smaller genre weight, paradoxically, produced a recommendation that felt more emotionally correct.
