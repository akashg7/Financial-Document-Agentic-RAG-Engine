"""
Tests for the BM25 sparse encoder.

These tests verify:
1. Tokenization produces expected tokens
2. BM25 fitting computes corpus-level statistics
3. Sparse document encoding produces valid SparseVectors
4. Sparse query encoding uses IDF weights
5. Term-to-index mapping is deterministic
6. Edge cases: empty text, single-word docs, unknown query terms
"""

import pytest
from qdrant_client.http.models import SparseVector

from app.retrieval.sparse import (
    tokenize,
    fit_bm25,
    encode_sparse_document,
    encode_sparse_query,
    _term_to_index,
    get_bm25_model,
    reset_bm25,
)


@pytest.fixture(autouse=True)
def clean_bm25_state():
    """Reset BM25 model state before each test."""
    reset_bm25()
    yield
    reset_bm25()


# Sample corpus for testing
SAMPLE_CORPUS = [
    "Apple reported total revenue of 383 billion dollars for fiscal year 2023",
    "The company revenue grew by 12 percent compared to the prior year",
    "Microsoft cloud revenue exceeded expectations in the third quarter",
    "Risk factors include supply chain disruptions and currency fluctuations",
    "Operating expenses increased due to research and development investments",
]


class TestTokenization:
    """Test the tokenizer."""

    def test_basic_tokenization(self):
        """Should lowercase and split on non-alphanumeric boundaries."""
        tokens = tokenize("Apple reported Revenue of $383B")
        assert "apple" in tokens
        assert "reported" in tokens
        assert "revenue" in tokens
        assert "383b" in tokens

    def test_removes_short_tokens(self):
        """Should filter out single-character tokens."""
        tokens = tokenize("I am a big company")
        assert "i" not in tokens
        assert "a" not in tokens
        assert "am" in tokens
        assert "big" in tokens
        assert "company" in tokens

    def test_empty_text(self):
        """Empty text should produce empty token list."""
        assert tokenize("") == []

    def test_punctuation_only(self):
        """Punctuation-only text should produce empty token list."""
        assert tokenize("!@#$%^&*()") == []

    def test_numbers_preserved(self):
        """Numeric tokens should be preserved (financial data matters)."""
        tokens = tokenize("Revenue was 383 billion in Q3 2023")
        assert "383" in tokens
        assert "2023" in tokens
        assert "q3" in tokens


class TestTermToIndex:
    """Test the hash-based term-to-index mapping."""

    def test_deterministic(self):
        """Same term should always map to the same index."""
        idx1 = _term_to_index("revenue")
        idx2 = _term_to_index("revenue")
        assert idx1 == idx2

    def test_different_terms_different_indices(self):
        """Different terms should (almost certainly) map to different indices."""
        idx_revenue = _term_to_index("revenue")
        idx_expenses = _term_to_index("expenses")
        assert idx_revenue != idx_expenses

    def test_index_is_positive_integer(self):
        """Index should be a positive integer within 32-bit range."""
        idx = _term_to_index("test")
        assert isinstance(idx, int)
        assert idx >= 0
        assert idx < 2**32


class TestBM25Fitting:
    """Test BM25 model fitting on a corpus."""

    def test_fit_creates_model(self):
        """Fitting should create a BM25 model."""
        assert get_bm25_model() is None
        fit_bm25(SAMPLE_CORPUS)
        assert get_bm25_model() is not None

    def test_fit_computes_avgdl(self):
        """Fitted model should have average document length."""
        model = fit_bm25(SAMPLE_CORPUS)
        assert model.avgdl > 0

    def test_fit_with_single_doc(self):
        """Should work with a single document (edge case)."""
        model = fit_bm25(["hello world"])
        assert model is not None
        assert model.avgdl > 0


class TestSparseDocumentEncoding:
    """Test document → sparse vector encoding."""

    def test_produces_sparse_vector(self):
        """Encoding should produce a SparseVector with indices and values."""
        fit_bm25(SAMPLE_CORPUS)
        sv = encode_sparse_document("Apple reported total revenue of 383 billion dollars")
        assert isinstance(sv, SparseVector)
        assert len(sv.indices) > 0
        assert len(sv.values) > 0
        assert len(sv.indices) == len(sv.values)

    def test_values_are_positive(self):
        """All TF-based weights should be positive."""
        fit_bm25(SAMPLE_CORPUS)
        sv = encode_sparse_document("Revenue grew in the fiscal year")
        for val in sv.values:
            assert val > 0

    def test_empty_text_returns_empty_vector(self):
        """Empty text should produce an empty sparse vector."""
        fit_bm25(SAMPLE_CORPUS)
        sv = encode_sparse_document("")
        assert len(sv.indices) == 0
        assert len(sv.values) == 0

    def test_repeated_terms_have_higher_weights(self):
        """A term appearing multiple times should have higher weight due to TF."""
        fit_bm25(SAMPLE_CORPUS)

        sv_once = encode_sparse_document("revenue was strong")
        sv_twice = encode_sparse_document("revenue revenue was strong")

        # Find the "revenue" term index
        rev_idx = _term_to_index("revenue")

        # Get the weight for "revenue" in each
        weight_once = 0.0
        weight_twice = 0.0
        for i, idx in enumerate(sv_once.indices):
            if idx == rev_idx:
                weight_once = sv_once.values[i]
        for i, idx in enumerate(sv_twice.indices):
            if idx == rev_idx:
                weight_twice = sv_twice.values[i]

        assert weight_twice > weight_once  # TF saturation, but still higher

    def test_raises_without_fitting(self):
        """Should raise RuntimeError if BM25 not fitted yet."""
        with pytest.raises(RuntimeError, match="BM25 model not fitted"):
            encode_sparse_document("test")


class TestSparseQueryEncoding:
    """Test query → sparse vector encoding with IDF weights."""

    def test_produces_sparse_vector(self):
        """Query encoding should produce a SparseVector."""
        fit_bm25(SAMPLE_CORPUS)
        sv = encode_sparse_query("what was the revenue?")
        assert isinstance(sv, SparseVector)
        assert len(sv.indices) > 0

    def test_rare_terms_have_higher_idf(self):
        """Rare terms should get higher IDF weights than common terms."""
        fit_bm25(SAMPLE_CORPUS)

        sv = encode_sparse_query("revenue disruptions")

        # "revenue" appears in 2 docs, "disruptions" in 1 doc
        # "disruptions" should have higher IDF
        rev_idx = _term_to_index("revenue")
        dis_idx = _term_to_index("disruptions")

        idf_revenue = 0.0
        idf_disruptions = 0.0
        for i, idx in enumerate(sv.indices):
            if idx == rev_idx:
                idf_revenue = sv.values[i]
            elif idx == dis_idx:
                idf_disruptions = sv.values[i]

        assert idf_disruptions > idf_revenue

    def test_unknown_query_terms_excluded(self):
        """Terms not in the corpus should be excluded from the query vector."""
        fit_bm25(SAMPLE_CORPUS)
        sv = encode_sparse_query("xyznonexistentterm12345")
        # Unknown term has 0 doc frequency → 0 IDF → excluded
        assert len(sv.indices) == 0

    def test_empty_query_returns_empty_vector(self):
        """Empty query should produce an empty sparse vector."""
        fit_bm25(SAMPLE_CORPUS)
        sv = encode_sparse_query("")
        assert len(sv.indices) == 0

    def test_raises_without_fitting(self):
        """Should raise RuntimeError if BM25 not fitted yet."""
        with pytest.raises(RuntimeError, match="BM25 model not fitted"):
            encode_sparse_query("test")

