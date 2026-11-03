"""
Sparse (BM25) vector encoder for keyword-based retrieval.

WHY BM25 as sparse vectors instead of using a separate search engine (Elasticsearch)?
- Qdrant natively supports sparse vectors in the SAME collection as dense vectors.
  This means we upsert ONCE (with both dense and sparse vectors per chunk) and
  search ONCE (Qdrant handles both in parallel). No second service to maintain.
- BM25 captures exact keyword matches that dense embeddings sometimes miss.
  For example, the query "AAPL 10-K Item 1A" relies on exact terms, not semantics.

HOW BM25 WORKS (intuition):
  BM25 assigns a weight to each term in a document based on:
  1. Term Frequency (TF): How often the term appears in THIS chunk.
     More appearances = more relevant, but with diminishing returns (saturation).
  2. Inverse Document Frequency (IDF): How rare the term is across ALL chunks.
     Rare terms ("amortization") are more discriminative than common ones ("the").
  3. Document Length Normalization: Longer chunks get penalized so they don't
     unfairly dominate just by having more words.

  BM25 score for term t in document d:
    score(t, d) = IDF(t) * (TF(t,d) * (k1 + 1)) / (TF(t,d) + k1 * (1 - b + b * |d|/avgdl))

  k1 controls TF saturation (default 1.5, higher = more sensitivity to repeated terms)
  b  controls length normalization (default 0.75, 0 = no normalization, 1 = full)

HOW WE ENCODE BM25 AS SPARSE VECTORS:
  A sparse vector is just a list of (index, value) pairs.
  - Index: a unique integer ID for each vocabulary term (we hash the term).
  - Value: the BM25 weight for that term in this chunk.
  
  For a query, we compute the IDF-weighted representation.
  
  Qdrant computes dot product between query sparse vector and document sparse
  vectors, which is equivalent to summing BM25 scores for matching terms.

WHY we use rank_bm25 to fit the vocabulary statistics (IDF, avgdl) and then
convert to sparse vectors ourselves:
  - rank_bm25 gives us proven BM25 math without reimplementing the formula.
  - We extract the per-term weights and map them to Qdrant's SparseVector format.
  - This approach is "build BM25 from a library, encode as sparse vectors."
"""

import hashlib
import logging
import re
from collections import Counter
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from qdrant_client.http.models import SparseVector

logger = logging.getLogger(__name__)

# Module-level BM25 model — fitted on the corpus during ingestion
_bm25_model: Optional[BM25Okapi] = None
_tokenized_corpus: Optional[list[list[str]]] = None
_vocabulary: Optional[dict[str, int]] = None


def tokenize(text: str) -> list[str]:
    """
    Simple whitespace + punctuation tokenizer with lowercasing.

    WHY not use NLTK or spaCy tokenizers:
    - We want zero extra dependencies for this step. BM25 is robust to
      simple tokenization — it's the TF-IDF math that matters, not
      sophisticated NLP preprocessing.
    - Financial text has predictable structure (sentences, numbers, tickers).
    - We strip very short tokens (len < 2) to remove noise but keep
      meaningful financial abbreviations (e.g., "Q3", "10").

    Returns:
        List of lowercase tokens.
    """
    # Lowercase and split on non-alphanumeric characters
    tokens = re.findall(r'\b[a-z0-9]+\b', text.lower())
    # Remove very short tokens that are usually noise
    return [t for t in tokens if len(t) >= 2]


def _term_to_index(term: str) -> int:
    """
    Map a term string to a stable integer index for sparse vector encoding.

    WHY hash-based mapping instead of a sequential vocabulary dict:
    - A sequential dict requires maintaining state and grows with vocab size.
    - Hash-based mapping is stateless: any term maps to the same index
      regardless of when or where it's computed.
    - Collision probability with 32-bit range (~4 billion) is negligible
      for our vocabulary size (~50K unique terms).

    We use the first 4 bytes of MD5 as an unsigned 32-bit integer.
    MD5 is NOT used for security here — just as a fast, uniform hash.
    """
    hash_bytes = hashlib.md5(term.encode()).digest()
    return int.from_bytes(hash_bytes[:4], 'big')


def fit_bm25(texts: list[str]) -> BM25Okapi:
    """
    Fit a BM25 model on the given corpus of texts.

    This must be called during ingestion BEFORE encoding sparse vectors,
    because BM25 needs corpus-level statistics (IDF, average document length).

    WHY we store the model globally:
    - BM25 statistics (IDF values, average doc length) are computed once
      over the entire corpus and reused for every query.
    - Re-fitting on every query would be O(n) per request — unacceptable.

    Args:
        texts: List of all chunk texts in the corpus.

    Returns:
        The fitted BM25Okapi model.
    """
    global _bm25_model, _tokenized_corpus, _vocabulary

    logger.info("Fitting BM25 model on %d documents...", len(texts))

    _tokenized_corpus = [tokenize(text) for text in texts]
    _bm25_model = BM25Okapi(_tokenized_corpus)

    # Build vocabulary from all tokens seen in the corpus
    all_tokens = set()
    for tokens in _tokenized_corpus:
        all_tokens.update(tokens)
    _vocabulary = {term: _term_to_index(term) for term in all_tokens}

    logger.info(
        "BM25 model fitted: %d documents, %d unique terms, avgdl=%.1f",
        len(texts),
        len(_vocabulary),
        _bm25_model.avgdl,
    )

    return _bm25_model


def get_bm25_model() -> Optional[BM25Okapi]:
    """Return the fitted BM25 model, or None if not yet fitted."""
    return _bm25_model


def encode_sparse_document(text: str) -> SparseVector:
    """
    Encode a single document/chunk as a BM25 sparse vector.

    For documents, the sparse vector captures TF-based weights:
    each term gets a weight proportional to its frequency in the chunk,
    normalized by document length (the BM25 TF component without IDF).

    WHY we use TF-only for documents and IDF-weighted for queries:
    - BM25 score = sum over query terms of: IDF(t) * TF_component(t, d)
    - If we put TF_component in the document vector and IDF in the query vector,
      the dot product gives us the exact BM25 score.
    - This is the standard decomposition for encoding BM25 as sparse vectors.

    Args:
        text: The chunk text to encode.

    Returns:
        SparseVector with term indices and TF-based weights.
    """
    if _bm25_model is None:
        raise RuntimeError(
            "BM25 model not fitted. Call fit_bm25() first during ingestion."
        )

    tokens = tokenize(text)
    if not tokens:
        return SparseVector(indices=[], values=[])

    # Count term frequencies
    tf_counts = Counter(tokens)
    doc_len = len(tokens)

    indices = []
    values = []

    # Compute BM25 TF component for each unique term
    # TF_component = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avgdl))
    k1 = _bm25_model.k1
    b = _bm25_model.b
    avgdl = _bm25_model.avgdl

    for term, tf in tf_counts.items():
        idx = _term_to_index(term)
        # BM25 TF saturation formula
        tf_weight = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avgdl))
        indices.append(idx)
        values.append(float(tf_weight))

    return SparseVector(indices=indices, values=values)


def encode_sparse_query(query: str) -> SparseVector:
    """
    Encode a query as a BM25-weighted sparse vector.

    For queries, each term gets an IDF weight:
    - Rare terms (high IDF) get high weight → more discriminative.
    - Common terms (low IDF) get low weight → less useful for retrieval.

    WHY IDF for queries:
    - The dot product of (IDF-weighted query) · (TF-weighted document)
      = sum of IDF(t) * TF_component(t, d) for matching terms
      = the BM25 relevance score.

    Args:
        query: The search query string.

    Returns:
        SparseVector with term indices and IDF-based weights.
    """
    if _bm25_model is None:
        raise RuntimeError(
            "BM25 model not fitted. Call fit_bm25() first during ingestion."
        )

    tokens = tokenize(query)
    if not tokens:
        return SparseVector(indices=[], values=[])

    indices = []
    values = []

    # Get IDF values from the fitted BM25 model
    # rank_bm25 stores IDF in the model after fitting
    for term in set(tokens):  # Deduplicate query terms
        idx = _term_to_index(term)
        idf = _get_idf(term)
        if idf > 0:  # Skip terms with zero or negative IDF (too common)
            indices.append(idx)
            values.append(float(idf))

    return SparseVector(indices=indices, values=values)


def _get_idf(term: str) -> float:
    """
    Get the IDF value for a term from the fitted BM25 model.

    WHY we compute IDF manually instead of using rank_bm25's internal arrays:
    - rank_bm25 stores IDF as an array indexed by document position, not by term.
    - We need term-level IDF, so we compute it from the document frequency.

    IDF formula (BM25 variant):
        IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
    where N = total docs, df(t) = docs containing term t.
    """
    if _bm25_model is None or _tokenized_corpus is None:
        return 0.0

    n_docs = len(_tokenized_corpus)

    # Count how many documents contain this term
    doc_freq = sum(1 for doc_tokens in _tokenized_corpus if term in set(doc_tokens))

    if doc_freq == 0:
        return 0.0

    # BM25 IDF formula
    idf = np.log((n_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
    return float(idf)


def reset_bm25() -> None:
    """Reset the BM25 model. Used in tests to ensure clean state."""
    global _bm25_model, _tokenized_corpus, _vocabulary
    _bm25_model = None
    _tokenized_corpus = None
    _vocabulary = None

