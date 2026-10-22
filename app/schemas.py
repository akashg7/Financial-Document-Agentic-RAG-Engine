"""
Pydantic models (schemas) for the Financial Document Intelligence Engine.

WHY separate schemas from ORM/storage models:
- The API contract is decoupled from the internal data representation.
- Pydantic handles request validation, serialization, and OpenAPI docs generation.
- Adding a new field to a response doesn't require changing database logic.
"""

from __future__ import annotations

import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ============================================================
# Document & Chunk Models
# ============================================================

class DocumentType(str, Enum):
    """Types of financial documents we support."""
    SEC_10K = "10-K"
    SEC_10Q = "10-Q"
    EARNINGS_CALL = "earnings_call"


class DocumentMetadata(BaseModel):
    """Metadata attached to every ingested document."""
    doc_id: str = Field(..., description="Unique document identifier")
    doc_type: DocumentType = Field(..., description="Type of financial document")
    ticker: str = Field(..., description="Stock ticker symbol, e.g. AAPL")
    company_name: str = Field(default="", description="Full company name")
    filing_date: Optional[str] = Field(default=None, description="Filing date in YYYY-MM-DD format")
    fiscal_year: Optional[str] = Field(default=None, description="Fiscal year, e.g. 2023")
    fiscal_quarter: Optional[str] = Field(default=None, description="Fiscal quarter, e.g. Q3")
    source_url: str = Field(default="", description="URL where the original document was fetched")


class ChunkMetadata(BaseModel):
    """Metadata attached to every chunk. Superset of document metadata."""
    chunk_id: str = Field(..., description="Unique chunk identifier (doc_id + chunk_index)")
    doc_id: str = Field(..., description="Parent document identifier")
    doc_type: DocumentType = Field(..., description="Type of financial document")
    ticker: str = Field(..., description="Stock ticker symbol")
    company_name: str = Field(default="")
    filing_date: Optional[str] = Field(default=None)
    fiscal_year: Optional[str] = Field(default=None)
    fiscal_quarter: Optional[str] = Field(default=None)
    section: str = Field(default="", description="Section of the document (e.g. 'Item 1A', 'Q&A')")
    page: Optional[int] = Field(default=None, description="Page number in the original document")
    chunk_index: int = Field(..., description="0-based index of this chunk within the document")
    content_hash: str = Field(..., description="SHA-256 hash of the chunk text for deduplication")
    source_url: str = Field(default="")


class Chunk(BaseModel):
    """A chunk of text with its metadata, ready for embedding and storage."""
    text: str = Field(..., description="The chunk text content")
    metadata: ChunkMetadata = Field(..., description="Structured metadata for this chunk")


# ============================================================
# Ingestion Models
# ============================================================

class IngestionStatus(str, Enum):
    """Status of an ingestion job."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class IngestionRequest(BaseModel):
    """Request to ingest documents."""
    source: str = Field(
        ...,
        description="Data source: 'sec_edgar', 'huggingface', or a file path",
    )
    tickers: list[str] = Field(
        default_factory=list,
        description="List of ticker symbols to ingest (for SEC EDGAR source)",
    )
    filing_types: list[DocumentType] = Field(
        default_factory=lambda: [DocumentType.SEC_10K],
        description="Types of filings to fetch",
    )
    max_documents: int = Field(default=10, description="Maximum number of documents to ingest")


class IngestionResponse(BaseModel):
    """Response after starting an ingestion job."""
    job_id: str = Field(..., description="Unique job identifier for tracking")
    status: IngestionStatus = Field(default=IngestionStatus.PENDING)
    message: str = Field(default="Ingestion job created")


class JobStatus(BaseModel):
    """Status of a background job."""
    job_id: str
    status: IngestionStatus
    total_documents: int = 0
    processed_documents: int = 0
    total_chunks: int = 0
    skipped_chunks: int = Field(0, description="Chunks skipped due to deduplication")
    errors: list[str] = Field(default_factory=list)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


# ============================================================
# Search / Query Models
# ============================================================

class SearchRequest(BaseModel):
    """Request for searching chunks."""
    query: str = Field(..., description="Natural language query", min_length=1)
    top_k: int = Field(default=10, description="Number of results to return", ge=1, le=100)
    ticker_filter: Optional[str] = Field(default=None, description="Filter by ticker symbol")
    doc_type_filter: Optional[DocumentType] = Field(default=None, description="Filter by document type")


class SearchResult(BaseModel):
    """A single search result with score and metadata."""
    chunk_id: str
    text: str
    score: float = Field(..., description="Relevance score (higher is better)")
    metadata: dict = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Response containing search results."""
    query: str
    results: list[SearchResult]
    retrieval_method: str = Field(default="dense", description="Method used: dense, sparse, hybrid")
    latency_ms: float = Field(..., description="Search latency in milliseconds")


# ============================================================
# Citation & Answer Models (for later days)
# ============================================================

class Citation(BaseModel):
    """A citation binding a claim to a source chunk."""
    chunk_id: str
    doc_id: str
    ticker: str = ""
    section: str = ""
    page: Optional[int] = None
    quote: str = Field(default="", description="Verbatim quote from the source chunk")


class AnswerResponse(BaseModel):
    """Final answer with citations and confidence metadata."""
    query: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(0.0, description="Confidence score between 0 and 1")
    decision: str = Field(
        default="generate",
        description="Agent decision: 'generate', 'rewrite', or 'abstain'",
    )
    retries: int = Field(0, description="Number of query rewrites performed")
    latency_ms: float = 0.0
    cached: bool = False
