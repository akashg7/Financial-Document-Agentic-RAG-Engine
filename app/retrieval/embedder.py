"""
Embedding service using sentence-transformers.

WHY sentence-transformers over raw HuggingFace transformers:
- sentence-transformers handles pooling, normalization, and batching automatically.
  With raw transformers, you'd need to manually apply mean pooling on the last
  hidden state, which is error-prone and verbose.
- The library is the de facto standard for embedding in RAG pipelines.

WHY all-MiniLM-L6-v2:
- 384 dimensions (small payload, fast search).
- Trained on 1B sentence pairs — strong semantic understanding.
- ~80MB model, runs on CPU in <100ms per batch of 32.
- Free, no API key needed.

WHY lazy loading:
- Importing sentence_transformers eagerly pulls in torch (~2GB), which slows
  down app startup and breaks fast test iteration. By loading on first use,
  tests that don't need embeddings (e.g., chunker tests) stay fast.
"""

import logging
import time
from typing import Optional

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

# Module-level singleton — initialized on first call to get_embedding_model()
_model = None


def get_embedding_model():
    """
    Lazily load and cache the sentence-transformers model.

    WHY a function instead of a global constant:
    - Avoids importing torch at module load time, which would slow down
      every test file that imports anything from app.retrieval.
    - The model is cached after first load, so subsequent calls are free.

    Returns:
        SentenceTransformer model instance.
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", settings.embedding_model_name)
        start = time.time()
        _model = SentenceTransformer(settings.embedding_model_name)
        elapsed = time.time() - start
        dim = (
            _model.get_embedding_dimension()
            if hasattr(_model, "get_embedding_dimension")
            else _model.get_sentence_embedding_dimension()
        )
        logger.info(
            "Embedding model loaded in %.2fs (dim=%d)",
            elapsed,
            dim,
        )
    return _model


def embed_texts(
    texts: list[str],
    batch_size: int = 64,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Encode a list of texts into dense vectors.

    Args:
        texts: List of strings to embed.
        batch_size: How many texts to encode in one forward pass.
                    Larger = faster (GPU utilization) but more memory.
                    64 is a safe default for CPU.
        show_progress: Show a tqdm progress bar (useful for large ingestions).

    Returns:
        numpy array of shape (len(texts), embedding_dim).

    WHY batch encoding instead of one-at-a-time:
    - The transformer model processes a batch in a single forward pass.
      Encoding 100 texts one-at-a-time does 100 forward passes.
      Encoding 100 texts in 2 batches of 50 does 2 forward passes.
      The speedup is roughly proportional to batch_size.
    """
    model = get_embedding_model()

    logger.debug("Embedding %d texts with batch_size=%d", len(texts), batch_size)
    start = time.time()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalize for cosine similarity
    )

    elapsed = time.time() - start
    logger.info(
        "Embedded %d texts in %.2fs (%.1f texts/sec)",
        len(texts),
        elapsed,
        len(texts) / elapsed if elapsed > 0 else 0,
    )

    return embeddings


def embed_query(query: str) -> np.ndarray:
    """
    Encode a single query string into a dense vector.

    This is separated from embed_texts for clarity:
    - At search time, we embed exactly one query.
    - At ingestion time, we embed many chunks in batches.

    Returns:
        1D numpy array of shape (embedding_dim,).
    """
    model = get_embedding_model()
    embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embedding


def get_embedding_dimension() -> int:
    """Return the dimensionality of the embedding model's output vectors."""
    model = get_embedding_model()
    if hasattr(model, "get_embedding_dimension"):
        return model.get_embedding_dimension()
    return model.get_sentence_embedding_dimension()
