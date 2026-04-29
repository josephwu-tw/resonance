"""
RAG layer — semantic retrieval over the song catalog.

Uses sentence-transformers to embed song descriptions and a user query,
then returns the closest candidates by cosine similarity so the scoring
engine only re-ranks a relevant shortlist rather than the full catalog.
Falls back to returning the full catalog if the model is unavailable.
"""

import logging
from typing import Dict, List

import numpy as np

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


def _song_to_text(song: Dict) -> str:
    """Render a song dict as a descriptive sentence for embedding."""
    tags = song.get("mood_tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(";") if t.strip()]
    tag_str = ", ".join(tags) if tags else "none"
    return (
        f"{song['title']} by {song['artist']}. "
        f"Genre: {song['genre']}. Mood: {song['mood']}. "
        f"Energy: {float(song['energy']):.2f}. "
        f"Decade: {song.get('release_decade', 'unknown')}. "
        f"Tags: {tag_str}."
    )


class SongIndex:
    """Semantic search index over the song catalog.

    Build once at startup, then call .search() for every user query.
    """

    def __init__(self, songs: List[Dict]) -> None:
        self.songs = songs
        self._embeddings: np.ndarray | None = None
        self._model: "SentenceTransformer | None" = None

        if _HAS_ST:
            logger.info("Loading sentence-transformer model '%s'…", _MODEL_NAME)
            self._model = SentenceTransformer(_MODEL_NAME)
            texts = [_song_to_text(s) for s in songs]
            self._embeddings = self._model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
            logger.info(
                "SongIndex built: %d songs × %d-dim embeddings",
                len(songs),
                self._embeddings.shape[1],
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
                "RAG: retrieved %d candidates for query %r (top score=%.3f)",
                len(results),
                query,
                float(scores[indices[0]]),
            )
            return results

        # Fallback: return entire catalog; scoring engine handles ranking
        logger.info("RAG fallback: returning full catalog of %d songs.", len(self.songs))
        return self.songs
