"""
Document chunker with metadata preservation and content-hash deduplication.

KEY DESIGN DECISIONS (see /docs/DECISIONS.md for full rationale):

1. Recursive character splitting over fixed-size splitting:
   We try to split on paragraph breaks first (\n\n), then sentences (\n),
   then words (' '). This keeps semantic units together. A fixed 512-char
   split would cut mid-sentence, which hurts retrieval quality.

2. Content-hash deduplication (SHA-256):
   Each chunk is hashed by its text content. On re-ingestion, unchanged
   chunks are skipped. This matters for SEC filings where an amended 10-K/A
   might change only a few sections — we re-embed only what changed.

3. Metadata propagation:
   Every chunk carries the full document metadata (ticker, filing_date, etc.)
   plus chunk-specific fields (chunk_index, section, page, content_hash).
   This enables filtered retrieval ("show me only AAPL 10-K chunks").
"""

import hashlib
import logging
from typing import Optional

from app.schemas import Chunk, ChunkMetadata, DocumentMetadata, DocumentType

logger = logging.getLogger(__name__)


def compute_content_hash(text: str) -> str:
    """
    Compute a SHA-256 hash of the chunk text.

    WHY SHA-256 instead of a faster hash (e.g., xxhash):
    - We need collision resistance, not speed. With ~100K chunks, even MD5
      would be fine, but SHA-256 is the safe default and the stdlib includes it.
    - The hashing cost is negligible compared to embedding time.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def recursive_character_split(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    separators: Optional[list[str]] = None,
) -> list[str]:
    """
    Split text into chunks using a hierarchy of separators.

    Algorithm:
    1. Try the first separator. If any segment is still too long, recurse
       with the next separator.
    2. Merge small segments back together up to chunk_size.
    3. Add overlap from the end of the previous chunk to the start of the next.

    This is similar to LangChain's RecursiveCharacterTextSplitter, but we
    implement it from scratch so you understand every line for interviews.

    Args:
        text: The full document text to split.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of characters to overlap between consecutive chunks.
        separators: Ordered list of separators to try. Defaults to
                    [paragraph, newline, sentence-end, space].

    Returns:
        List of chunk strings.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]

    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    # Pick the first separator that actually appears in the text
    chosen_sep = ""
    for sep in separators:
        if sep in text:
            chosen_sep = sep
            break

    # If no separator found, force-split by character count
    if not chosen_sep:
        return _force_split(text, chunk_size, chunk_overlap)

    # Split on the chosen separator
    raw_splits = text.split(chosen_sep)

    # Recursively split any segment that's still too long, using remaining separators
    remaining_seps = separators[separators.index(chosen_sep) + 1 :]
    segments: list[str] = []
    for segment in raw_splits:
        if len(segment) > chunk_size and remaining_seps:
            segments.extend(
                recursive_character_split(segment, chunk_size, chunk_overlap, remaining_seps)
            )
        else:
            stripped = segment.strip()
            if stripped:
                segments.append(stripped)

    # Merge small segments back up to chunk_size
    return _merge_segments(segments, chunk_size, chunk_overlap, chosen_sep)


def _merge_segments(
    segments: list[str],
    chunk_size: int,
    chunk_overlap: int,
    separator: str,
) -> list[str]:
    """
    Merge small segments into chunks up to chunk_size, then add overlap.

    WHY merge instead of keeping tiny chunks:
    - A 20-character chunk has almost no semantic meaning for retrieval.
    - Merging ensures each chunk is substantive enough to be useful.
    """
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_length = 0

    for segment in segments:
        seg_len = len(segment)

        # If adding this segment exceeds chunk_size, flush the current chunk
        if current_length + seg_len + len(separator) > chunk_size and current_chunk:
            chunk_text = separator.join(current_chunk).strip()
            if chunk_text:
                chunks.append(chunk_text)

            # Start a new chunk, but keep overlap from the end of the previous one
            if chunk_overlap > 0 and chunks:
                overlap_text = chunks[-1][-chunk_overlap:]
                current_chunk = [overlap_text, segment]
                current_length = len(overlap_text) + seg_len + len(separator)
            else:
                current_chunk = [segment]
                current_length = seg_len
        else:
            current_chunk.append(segment)
            current_length += seg_len + len(separator)

    # Flush the last chunk
    if current_chunk:
        chunk_text = separator.join(current_chunk).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks


def _force_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Last resort: split by character count when no separators are found.
    This handles edge cases like a single very long URL or encoded data.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - chunk_overlap if chunk_overlap > 0 else end
    return chunks


def chunk_document(
    text: str,
    doc_metadata: DocumentMetadata,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    section: str = "",
    page: Optional[int] = None,
) -> list[Chunk]:
    """
    Split a document into chunks with full metadata and content hashes.

    This is the main entry point for the chunking pipeline.

    Args:
        text: Full document text.
        doc_metadata: Metadata from the document loader.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks.
        section: Which section of the document this text comes from.
        page: Page number if known.

    Returns:
        List of Chunk objects ready for deduplication and embedding.
    """
    raw_chunks = recursive_character_split(text, chunk_size, chunk_overlap)

    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        content_hash = compute_content_hash(chunk_text)
        chunk_id = f"{doc_metadata.doc_id}_chunk_{i:04d}"

        chunk_meta = ChunkMetadata(
            chunk_id=chunk_id,
            doc_id=doc_metadata.doc_id,
            doc_type=doc_metadata.doc_type,
            ticker=doc_metadata.ticker,
            company_name=doc_metadata.company_name,
            filing_date=doc_metadata.filing_date,
            fiscal_year=doc_metadata.fiscal_year,
            fiscal_quarter=doc_metadata.fiscal_quarter,
            section=section,
            page=page,
            chunk_index=i,
            content_hash=content_hash,
            source_url=doc_metadata.source_url,
        )
        chunks.append(Chunk(text=chunk_text, metadata=chunk_meta))

    logger.info(
        "Chunked doc %s into %d chunks (size=%d, overlap=%d)",
        doc_metadata.doc_id,
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks


def deduplicate_chunks(
    new_chunks: list[Chunk],
    existing_hashes: set[str],
) -> tuple[list[Chunk], list[Chunk]]:
    """
    Filter out chunks whose content_hash already exists.

    Args:
        new_chunks: Chunks from the current ingestion.
        existing_hashes: Set of content hashes already in the vector store.

    Returns:
        Tuple of (chunks_to_insert, chunks_skipped).
        
    WHY content-hash dedup instead of chunk_id dedup:
    - If a document is re-ingested, chunk boundaries might shift (e.g., a
      paragraph was inserted before an existing one). Content-hash catches
      that the *content* is the same even if the chunk_id changed.
    - This also catches identical paragraphs across different documents
      (e.g., boilerplate legal disclaimers).
    """
    to_insert = []
    skipped = []

    for chunk in new_chunks:
        if chunk.metadata.content_hash in existing_hashes:
            skipped.append(chunk)
        else:
            to_insert.append(chunk)
            # Add to existing set so intra-batch duplicates are also caught
            existing_hashes.add(chunk.metadata.content_hash)

    logger.info(
        "Dedup: %d to insert, %d skipped (of %d total)",
        len(to_insert),
        len(skipped),
        len(new_chunks),
    )
    return to_insert, skipped
