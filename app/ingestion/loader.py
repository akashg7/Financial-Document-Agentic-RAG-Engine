"""
Document loaders for SEC EDGAR filings and HuggingFace earnings transcripts.

DATA SOURCES (see /docs/DECISIONS.md):

1. SEC EDGAR (10-K, 10-Q):
   - Public domain government data. No license restrictions.
   - Fair-use policy requires a User-Agent header with name + email.
   - Rate limit: 10 requests/second. We use 0.15s delay between requests.
   - We fetch the "full submission text" and extract the filing body.

2. HuggingFace: Rogersurf/earnings-call-transcripts
   - Cleaned earnings call transcripts with ticker, quarter, date.
   - We load via the `datasets` library.

WHY two loaders instead of one unified loader:
- The data formats are completely different (HTML vs. structured Parquet).
- Having separate loaders keeps each one simple and testable.
- The ingestion pipeline calls whichever loader matches the request source.
"""

import logging
import time
import re
from typing import Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from app.config import settings
from app.schemas import DocumentMetadata, DocumentType

logger = logging.getLogger(__name__)

# SEC EDGAR base URLs
EDGAR_FULL_TEXT_SEARCH = "https://efts.sec.gov/LATEST/search-index"
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/"

# Mapping of common tickers to CIK numbers (Central Index Key)
# In production, you'd query the EDGAR company search API, but for a
# portfolio project with ~20 companies, a static map is simpler and avoids
# an extra API call per ticker.
TICKER_TO_CIK = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "GOOGL": "0001652044",
    "AMZN": "0001018724",
    "META": "0001326801",
    "TSLA": "0001318605",
    "NVDA": "0001045810",
    "JPM": "0000019617",
    "JNJ": "0000200406",
    "V": "0001403161",
    "WMT": "0000104169",
    "PG": "0000080424",
    "UNH": "0000731766",
    "HD": "0000354950",
    "BAC": "0000070858",
    "KO": "0000021344",
    "PFE": "0000078003",
    "DIS": "0001744489",
    "NFLX": "0001065280",
    "INTC": "0000050863",
}

# Map filing type enum to EDGAR form type strings
DOC_TYPE_TO_FORM = {
    DocumentType.SEC_10K: "10-K",
    DocumentType.SEC_10Q: "10-Q",
}


def _get_edgar_headers() -> dict[str, str]:
    """
    Return headers for SEC EDGAR requests.

    WHY a custom User-Agent:
    SEC EDGAR's fair-use policy (https://www.sec.gov/os/accessing-edgar-data)
    requires automated tools to send a User-Agent with company name and
    contact email. Requests without it get rate-limited or blocked.
    """
    return {
        "User-Agent": settings.sec_edgar_user_agent,
        "Accept-Encoding": "gzip, deflate",
    }


def _polite_sleep(seconds: float = 0.15):
    """
    Sleep between EDGAR requests to respect the 10 req/sec fair-use limit.
    0.15s gives us ~6.7 req/sec with some headroom.
    """
    time.sleep(seconds)


def fetch_edgar_filing_list(
    ticker: str,
    filing_type: DocumentType = DocumentType.SEC_10K,
    max_filings: int = 5,
) -> list[dict]:
    """
    Fetch a list of recent filings for a ticker from SEC EDGAR.

    Returns a list of dicts with keys: accession_number, filing_date,
    primary_document, form_type.

    The SEC submissions API returns all filings for a company. We filter
    to the requested form type and take the most recent max_filings.
    """
    cik = TICKER_TO_CIK.get(ticker.upper())
    if not cik:
        logger.warning("Unknown ticker %s — no CIK mapping found", ticker)
        return []

    url = EDGAR_SUBMISSIONS_URL.format(cik=cik)
    headers = _get_edgar_headers()

    try:
        _polite_sleep()
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error("Failed to fetch EDGAR submissions for %s: %s", ticker, e)
        return []

    recent = data.get("filings", {}).get("recent", {})
    if not recent:
        logger.warning("No filings found for %s", ticker)
        return []

    form_type_str = DOC_TYPE_TO_FORM.get(filing_type, "10-K")

    filings = []
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])

    for i, form in enumerate(forms):
        if form == form_type_str and len(filings) < max_filings:
            filings.append({
                "accession_number": accessions[i].replace("-", ""),
                "accession_number_dashed": accessions[i],
                "filing_date": dates[i],
                "primary_document": primary_docs[i] if i < len(primary_docs) else "",
                "form_type": form,
            })

    logger.info("Found %d %s filings for %s", len(filings), form_type_str, ticker)
    return filings


def fetch_edgar_filing_text(
    ticker: str,
    filing_info: dict,
) -> Optional[tuple[str, str]]:
    """
    Download and extract text from a single EDGAR filing.

    Returns (text_content, source_url) or None if download fails.

    WHY BeautifulSoup for text extraction instead of raw text:
    - SEC filings are HTML with nested tables, inline CSS, and XBRL tags.
    - BS4 with .get_text() strips all markup and gives us clean text.
    - We use separator='\\n' to preserve paragraph structure for chunking.
    """
    cik = TICKER_TO_CIK.get(ticker.upper(), "")
    accession = filing_info["accession_number"]
    primary_doc = filing_info["primary_document"]

    # Build the URL to the primary document
    doc_url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik.lstrip('0')}/{accession}/{primary_doc}"
    )

    headers = _get_edgar_headers()

    try:
        _polite_sleep()
        resp = requests.get(doc_url, headers=headers, timeout=60)
        resp.raise_for_status()
    except Exception as e:
        logger.error("Failed to download filing from %s: %s", doc_url, e)
        return None

    # Parse HTML and extract text
    soup = BeautifulSoup(resp.content, "lxml")

    # Remove script and style elements that pollute the text
    for element in soup(["script", "style", "meta", "link"]):
        element.decompose()

    text = soup.get_text(separator="\n", strip=True)

    # Clean up excessive whitespace while preserving paragraph breaks
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)

    if len(text) < 100:
        logger.warning("Filing text suspiciously short (%d chars) from %s", len(text), doc_url)
        return None

    logger.info(
        "Downloaded %s filing for %s: %d chars from %s",
        filing_info["form_type"],
        ticker,
        len(text),
        doc_url,
    )
    return text, doc_url


def load_sec_edgar_documents(
    tickers: list[str],
    filing_types: list[DocumentType],
    max_per_ticker: int = 3,
) -> list[tuple[str, DocumentMetadata]]:
    """
    Load documents from SEC EDGAR for the given tickers and filing types.

    Returns a list of (text, DocumentMetadata) tuples.
    """
    documents = []

    for ticker in tickers:
        for filing_type in filing_types:
            filings = fetch_edgar_filing_list(ticker, filing_type, max_per_ticker)

            for filing_info in filings:
                result = fetch_edgar_filing_text(ticker, filing_info)
                if result is None:
                    continue

                text, source_url = result
                doc_id = f"{ticker}_{filing_info['form_type']}_{filing_info['filing_date']}"

                metadata = DocumentMetadata(
                    doc_id=doc_id,
                    doc_type=filing_type,
                    ticker=ticker.upper(),
                    company_name=ticker.upper(),  # Simplified; could lookup from EDGAR
                    filing_date=filing_info["filing_date"],
                    source_url=source_url,
                )
                documents.append((text, metadata))

    logger.info("Loaded %d documents from SEC EDGAR", len(documents))
    return documents


def load_huggingface_earnings_transcripts(
    max_documents: int = 100,
    tickers: Optional[list[str]] = None,
) -> list[tuple[str, DocumentMetadata]]:
    """
    Load earnings call transcripts from HuggingFace datasets.

    Uses: Rogersurf/earnings-call-transcripts
    This dataset contains cleaned transcripts with ticker, quarter, date.

    WHY HuggingFace datasets library instead of direct download:
    - Handles caching, streaming, and format conversion automatically.
    - Parquet files are columnar and memory-efficient for large datasets.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("'datasets' library not installed. pip install datasets")
        return []

    try:
        # Load the dataset — streams from HuggingFace Hub
        ds = load_dataset(
            "Rogersurf/earnings-call-transcripts",
            split="train",
            trust_remote_code=False,
        )
    except Exception as e:
        logger.error("Failed to load HuggingFace dataset: %s", e)
        return []

    documents = []
    count = 0

    for row in ds:
        if count >= max_documents:
            break

        # Extract fields from the dataset
        ticker = row.get("ticker", "UNKNOWN")
        if tickers and ticker.upper() not in [t.upper() for t in tickers]:
            continue

        text = row.get("text", "") or row.get("transcript", "") or ""
        if not text or len(text) < 100:
            continue

        quarter = row.get("quarter", "")
        date = row.get("date", "") or row.get("call_date", "")

        doc_id = f"{ticker}_earnings_{quarter}_{date}".replace(" ", "_")

        metadata = DocumentMetadata(
            doc_id=doc_id,
            doc_type=DocumentType.EARNINGS_CALL,
            ticker=ticker.upper(),
            company_name=row.get("company", ticker.upper()),
            filing_date=str(date) if date else None,
            fiscal_quarter=str(quarter) if quarter else None,
            source_url="https://huggingface.co/datasets/Rogersurf/earnings-call-transcripts",
        )
        documents.append((text, metadata))
        count += 1

    logger.info("Loaded %d earnings transcripts from HuggingFace", len(documents))
    return documents
