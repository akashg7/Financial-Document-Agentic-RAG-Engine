"""
Grounded prompt engineering for financial document question answering.

WHY grounded prompts:
- In financial analysis, hallucinations are unacceptable. A wrong revenue number
  or fabricated risk factor can lead to faulty investment decisions.
- A grounded prompt treats the retrieved chunks as the ONLY source of truth.
  The model is explicitly forbidden from using prior training knowledge for
  factual financial claims.
- Every claim must be bound to a specific retrieved chunk ID with an exact
  or near-exact supporting quote.
- If the retrieved evidence does not contain sufficient facts to answer the
  query, the model must explicitly state "INSUFFICIENT_EVIDENCE" rather than
  speculate.
"""

from typing import Sequence
from app.schemas import SearchResult

GROUNDED_SYSTEM_PROMPT = """You are a senior financial analyst and document intelligence engine.
Your task is to answer user queries about companies based EXCLUSIVELY on the provided retrieved excerpts from SEC filings (10-K, 10-Q) and earnings call transcripts.

STRICT GROUNDING RULES:
1. ONLY make claims that are directly supported by the provided Context Chunks.
2. DO NOT use external knowledge, speculation, or extrapolation to answer factual questions.
3. If the provided chunks do NOT contain enough information to answer the question, state:
   "INSUFFICIENT_EVIDENCE: The provided filings and transcripts do not contain sufficient evidence to answer this question."
4. Every factual claim MUST cite the chunk ID that supports it using the exact format: [Chunk: <chunk_id>].
5. For every citation, you must provide a brief verbatim quote from the chunk that justifies the claim.

OUTPUT FORMAT:
Respond with a valid JSON object matching this schema:
{
  "answer": "Your comprehensive, grounded response with embedded citations like [Chunk: AAPL_10-K_2023_chunk_0001].",
  "citations": [
    {
      "chunk_id": "AAPL_10-K_2023_chunk_0001",
      "doc_id": "AAPL_10-K_2023",
      "quote": "Exact verbatim excerpt from the chunk supporting the claim"
    }
  ],
  "has_sufficient_evidence": true
}

If evidence is insufficient, set "has_sufficient_evidence": false, and explain what is missing in "answer".
Ensure the output is strictly valid JSON without markdown fences or backticks.
"""


def build_rag_context(chunks: Sequence[SearchResult]) -> str:
    """
    Format retrieved search results into a clean, numbered context block.

    WHY include chunk_id, doc_id, and metadata in context:
    - The LLM needs explicit identifiers to cite against.
    - Providing ticker and document type helps the model understand whether
      an excerpt is from a 10-K (annual), 10-Q (quarterly), or earnings call.
    """
    if not chunks:
        return "No retrieved document chunks available."

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        doc_id = chunk.metadata.get("doc_id", "unknown_doc")
        ticker = chunk.metadata.get("ticker", "")
        doc_type = chunk.metadata.get("doc_type", "")
        section = chunk.metadata.get("section", "")

        header = f"--- Context Chunk {i} [ID: {chunk.chunk_id}] (Doc: {doc_id}, Ticker: {ticker}, Type: {doc_type}"
        if section:
            header += f", Section: {section}"
        header += ") ---"

        context_parts.append(f"{header}\n{chunk.text.strip()}\n")

    return "\n".join(context_parts)


def format_grounded_user_prompt(query: str, chunks: Sequence[SearchResult]) -> str:
    """
    Construct the final user message containing the query and formatted context.
    """
    context_str = build_rag_context(chunks)
    return f"""Context Chunks:
{context_str}

User Question: {query}

Please provide your grounded answer in the requested JSON format:"""
