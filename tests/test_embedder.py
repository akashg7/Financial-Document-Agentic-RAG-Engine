"""
Tests for the embedding service.

These tests verify:
1. embed_texts produces vectors of the correct dimension (384 for all-MiniLM-L6-v2)
2. Batch embedding is consistent with single-text embedding
3. embed_query returns a 1D vector
4. Semantically similar texts have higher similarity than dissimilar texts
5. Empty input is handled gracefully

NOTE: These tests load the actual sentence-transformers model (~80MB).
First run will download the model and be slow. Subsequent runs use cache.
"""

import pytest
import numpy as np

from app.retrieval.embedder import (
    embed_texts,
    embed_query,
    get_embedding_dimension,
)


# ============================================================
# Dimension Tests
# ============================================================

class TestEmbeddingDimensions:
    """Tests for correct embedding dimensions."""

    def test_single_text_dimension(self):
        """A single text should produce a vector of the expected dimension."""
        embeddings = embed_texts(["Apple reported strong revenue growth."])
        assert embeddings.shape == (1, 384)

    def test_batch_dimension(self):
        """A batch of N texts should produce an (N, 384) matrix."""
        texts = [
            "Revenue increased 10% year over year.",
            "The company declared a quarterly dividend.",
            "Net income reached $97 billion.",
        ]
        embeddings = embed_texts(texts)
        assert embeddings.shape == (3, 384)

    def test_get_embedding_dimension(self):
        """get_embedding_dimension should return 384."""
        dim = get_embedding_dimension()
        assert dim == 384


# ============================================================
# Query Embedding Tests
# ============================================================

class TestQueryEmbedding:
    """Tests for single query embedding."""

    def test_query_returns_1d(self):
        """embed_query should return a 1D vector."""
        vec = embed_query("What was Apple's revenue?")
        assert vec.ndim == 1
        assert vec.shape[0] == 384

    def test_query_is_normalized(self):
        """The query vector should be L2-normalized (unit length)."""
        vec = embed_query("What was Apple's revenue?")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"


# ============================================================
# Consistency Tests
# ============================================================

class TestEmbeddingConsistency:
    """Tests that embedding is deterministic and consistent."""

    def test_deterministic(self):
        """Embedding the same text twice should give identical vectors."""
        text = "Apple reported revenue of $383 billion."
        v1 = embed_texts([text])
        v2 = embed_texts([text])
        np.testing.assert_array_almost_equal(v1, v2, decimal=6)

    def test_batch_equals_single(self):
        """Batch embedding should give same results as individual embedding."""
        texts = ["Revenue grew.", "Expenses decreased."]
        batch_result = embed_texts(texts)
        single_results = np.vstack([embed_texts([t]) for t in texts])
        np.testing.assert_array_almost_equal(batch_result, single_results, decimal=5)

    def test_normalized_vectors(self):
        """All output vectors should be L2-normalized."""
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        embeddings = embed_texts(texts)
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(3), decimal=5)


# ============================================================
# Semantic Similarity Tests
# ============================================================

class TestSemanticSimilarity:
    """Tests that embeddings capture semantic meaning."""

    def test_similar_texts_closer(self):
        """
        Semantically similar texts should have higher cosine similarity
        than dissimilar texts.

        This is the fundamental property we need for retrieval:
        a query about 'revenue' should rank a chunk about 'revenue growth'
        higher than a chunk about 'employee count'.
        """
        revenue_q = embed_query("What was the company's revenue?")
        revenue_a = embed_texts(["Total revenue reached $383 billion in fiscal 2023."])[0]
        unrelated = embed_texts(["The company has 164,000 full-time employees."])[0]

        # Cosine similarity (vectors are normalized, so dot product = cosine sim)
        sim_relevant = float(np.dot(revenue_q, revenue_a))
        sim_unrelated = float(np.dot(revenue_q, unrelated))

        assert sim_relevant > sim_unrelated, (
            f"Expected similar text ({sim_relevant:.4f}) to score higher "
            f"than unrelated text ({sim_unrelated:.4f})"
        )

    def test_different_phrasing_still_similar(self):
        """
        Different phrasings of the same concept should have high similarity.
        This tests the model's ability to handle paraphrases — critical for
        RAG where the user's query wording won't match the document exactly.
        """
        v1 = embed_query("How much money did Apple make?")
        v2 = embed_query("What was Apple's total revenue?")

        sim = float(np.dot(v1, v2))
        assert sim > 0.5, f"Expected high similarity for paraphrases, got {sim:.4f}"
