"""
RAG layer — semantic retrieval over the song catalog.

RAG Enhancement (stretch feature):
  Accepts an optional genre_profiles dict (from data/genre_profiles.json).
  When provided, each song's embedding text is augmented with its genre's
  rich conceptual description — adding vocabulary like "vinyl crackle",
  "neon-lit synths", or "tape hiss" that never appears in songs.csv.
  This lets queries using conceptual language retrieve the right genre
  even when no song label directly matches the user's words.

  Retrieval improvement is measurable via compare_retrieval(), which runs
  the same query against both plain and augmented indexes and returns scores.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except ImportError:
    _HAS_ST = False
    logger.warning(
        "sentence-transformers not installed; RAG will return the full catalog. "
        "Install it with: pip install sentence-transformers"
    )

_MODEL_NAME = "all-MiniLM-L6-v2"


def _song_to_text(song: Dict, genre_profile: Optional[Dict] = None) -> str:
    """Render a song dict as a descriptive sentence for embedding.

    If genre_profile is provided its description is appended, giving the
    embedding richer conceptual vocabulary beyond the raw song labels.
    """
    tags = song.get("mood_tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(";") if t.strip()]
    tag_str = ", ".join(tags) if tags else "none"

    base = (
        f"{song['title']} by {song['artist']}. "
        f"Genre: {song['genre']}. Mood: {song['mood']}. "
        f"Energy: {float(song['energy']):.2f}. "
        f"Decade: {song.get('release_decade', 'unknown')}. "
        f"Tags: {tag_str}."
    )

    if genre_profile:
        base = f"{base} {genre_profile.get('description', '')}"

    return base


class SongIndex:
    """Semantic search index over the song catalog.

    RAG Enhancement: pass genre_profiles to augment song embeddings with
    rich genre descriptions, improving retrieval for conceptual queries.
    """

    def __init__(
        self,
        songs: List[Dict],
        genre_profiles: Optional[Dict] = None,
    ) -> None:
        self.songs = songs
        self._genre_profiles = genre_profiles or {}
        self._embeddings: Optional[np.ndarray] = None
        self._model: Optional["SentenceTransformer"] = None
        self._augmented = bool(genre_profiles)

        if _HAS_ST:
            logger.info(
                "Loading sentence-transformer model '%s' (augmented=%s)…",
                _MODEL_NAME, self._augmented,
            )
            self._model = SentenceTransformer(_MODEL_NAME)
            texts = [
                _song_to_text(s, self._genre_profiles.get(s.get("genre", "")))
                for s in songs
            ]
            self._embeddings = self._model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
            logger.info(
                "SongIndex built: %d songs × %d-dim embeddings (profiles: %s)",
                len(songs),
                self._embeddings.shape[1],
                "yes" if self._augmented else "no",
            )
        else:
            logger.info("SongIndex running in keyword-fallback mode (%d songs).", len(songs))

    def search(self, query: str, top_n: int = 15) -> List[Dict]:
        """Return the top-n songs most semantically similar to query."""
        if not query or not query.strip():
            logger.debug("Empty query received; returning full catalog.")
            return self.songs

        if self._model is not None and self._embeddings is not None:
            q_emb = self._model.encode([query], normalize_embeddings=True)
            scores = (self._embeddings @ q_emb.T).flatten()
            indices = np.argsort(scores)[::-1][:top_n]
            results = [self.songs[i] for i in indices]
            logger.info(
                "RAG: retrieved %d candidates (augmented=%s, top_score=%.3f) for %r",
                len(results), self._augmented, float(scores[indices[0]]), query,
            )
            return results

        logger.info("RAG fallback: returning full catalog of %d songs.", len(self.songs))
        return self.songs

    def compare_retrieval(
        self, query: str, top_n: int = 5
    ) -> Dict[str, List[Tuple[str, str, float]]]:
        """Compare retrieval quality with vs without genre profile augmentation.

        Returns a dict with keys 'plain' and 'augmented', each containing a list
        of (title, genre, similarity_score) tuples for the given query.
        Only meaningful when sentence-transformers is available.
        """
        if self._model is None or self._embeddings is None:
            return {"plain": [], "augmented": [], "note": "sentence-transformers not available"}

        q_emb = self._model.encode([query], normalize_embeddings=True)

        # Augmented results (this index)
        aug_scores = (self._embeddings @ q_emb.T).flatten()
        aug_idx = np.argsort(aug_scores)[::-1][:top_n]
        augmented = [
            (self.songs[i]["title"], self.songs[i]["genre"], float(aug_scores[i]))
            for i in aug_idx
        ]

        # Plain results (rebuild texts without profiles)
        plain_texts = [_song_to_text(s) for s in self.songs]
        plain_emb = self._model.encode(plain_texts, normalize_embeddings=True, show_progress_bar=False)
        plain_scores = (plain_emb @ q_emb.T).flatten()
        plain_idx = np.argsort(plain_scores)[::-1][:top_n]
        plain = [
            (self.songs[i]["title"], self.songs[i]["genre"], float(plain_scores[i]))
            for i in plain_idx
        ]

        return {"plain": plain, "augmented": augmented}
