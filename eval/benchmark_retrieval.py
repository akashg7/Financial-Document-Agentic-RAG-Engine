#!/usr/bin/env python3
"""
Benchmark retrieval strategies: vector-only vs hybrid vs hybrid+rerank.

This script evaluates retrieval quality and latency by comparing three strategies
across realistic financial benchmark queries with candidate passage pools.

Strategies evaluated:
1. vector_only: Dense semantic embedding using sentence-transformers (all-MiniLM-L6-v2)
   and cosine similarity ranking.
2. hybrid: Fusion of dense semantic search and BM25 lexical keyword search
   using Reciprocal Rank Fusion (RRF, k=60).
3. hybrid_rerank: Two-stage retrieval — hybrid search over-retrieval pool followed
   by cross-encoder reranking (cross-encoder/ms-marco-MiniLM-L-6-v2).

Metrics computed:
- Recall@5: Fraction of relevant chunks found in top 5
- Recall@10: Fraction of relevant chunks found in top 10
- MRR (Mean Reciprocal Rank): 1 / rank of the first relevant chunk
- Latency (ms): mean, p50 (median), p95

Usage:
    python eval/benchmark_retrieval.py

Output:
    results/retrieval_benchmark_<timestamp>.json
"""

import json
import sys
import time
import statistics
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rank_bm25 import BM25Okapi

from app.schemas import SearchResult
from app.retrieval.embedder import embed_query, embed_texts
from app.retrieval.sparse import tokenize
from app.retrieval.rrf import reciprocal_rank_fusion
from app.rerank.reranker import rerank


# ---------------------------------------------------------------------------
# Benchmark test cases — queries with candidate passages and ground truth
# ---------------------------------------------------------------------------

BENCHMARK_CASES = [
    {
        "query": "What was Apple's total revenue in fiscal year 2023?",
        "candidates": [
            SearchResult(chunk_id="aapl_rev_2023", text="Apple reported total net sales of $383,285 million for the fiscal year ended September 30, 2023, compared to $394,328 million in 2022.", score=0.0, metadata={"ticker": "AAPL"}),
            SearchResult(chunk_id="aapl_rev_segment", text="iPhone revenue was $200,583 million in fiscal 2023. Services revenue reached $85,200 million.", score=0.0, metadata={"ticker": "AAPL"}),
            SearchResult(chunk_id="msft_rev_2023", text="Microsoft reported total revenue of $211.9 billion for fiscal year 2023.", score=0.0, metadata={"ticker": "MSFT"}),
            SearchResult(chunk_id="aapl_employees", text="As of September 30, 2023, Apple had approximately 161,000 full-time equivalent employees.", score=0.0, metadata={"ticker": "AAPL"}),
            SearchResult(chunk_id="aapl_risk_supply", text="The Company's business can be impacted by supply chain disruptions and single-source component suppliers.", score=0.0, metadata={"ticker": "AAPL"}),
            SearchResult(chunk_id="tsla_rev_2023", text="Tesla's total revenue was $96.8 billion in fiscal year 2023.", score=0.0, metadata={"ticker": "TSLA"}),
            SearchResult(chunk_id="aapl_dividend", text="The Board of Directors declared a cash dividend of $0.24 per share of common stock.", score=0.0, metadata={"ticker": "AAPL"}),
            SearchResult(chunk_id="googl_rev_2023", text="Alphabet reported total revenue of $307.4 billion in fiscal year 2023.", score=0.0, metadata={"ticker": "GOOGL"}),
            SearchResult(chunk_id="aapl_stores", text="The Company operates 524 retail stores worldwide and sells through its online store.", score=0.0, metadata={"ticker": "AAPL"}),
            SearchResult(chunk_id="amzn_rev_2023", text="Amazon reported total revenue of $574.8 billion in fiscal year 2023.", score=0.0, metadata={"ticker": "AMZN"}),
        ],
        "relevant_ids": {"aapl_rev_2023", "aapl_rev_segment"},
    },
    {
        "query": "What are the key risk factors Apple disclosed regarding supply chain and international operations?",
        "candidates": [
            SearchResult(chunk_id="aapl_risk_supply", text="The Company depends on component and product manufacturing and logistical services provided by outsourcing partners, many of which are located outside the U.S.", score=0.0, metadata={"ticker": "AAPL"}),
            SearchResult(chunk_id="aapl_risk_macro", text="The Company's operations and performance depend significantly on worldwide economic conditions and their impact on levels of consumer spending.", score=0.0, metadata={"ticker": "AAPL"}),
            SearchResult(chunk_id="aapl_risk_competition", text="The markets for the Company's products and services are highly competitive and the Company is confronted by aggressive competition in all areas of its business.", score=0.0, metadata={"ticker": "AAPL"}),
            SearchResult(chunk_id="aapl_risk_regulatory", text="The Company is subject to complex and evolving laws and regulations regarding privacy, data protection, and other matters.", score=0.0, metadata={"ticker": "AAPL"}),
            SearchResult(chunk_id="aapl_rev_2023", text="Apple reported total net sales of $383,285 million for the fiscal year ended September 30, 2023.", score=0.0, metadata={"ticker": "AAPL"}),
            SearchResult(chunk_id="aapl_services", text="Services revenue reached $85.2 billion, driven by growth in App Store, Apple Music, and iCloud.", score=0.0, metadata={"ticker": "AAPL"}),
            SearchResult(chunk_id="msft_risk_ai", text="Microsoft faces risks from the rapidly evolving AI competitive landscape.", score=0.0, metadata={"ticker": "MSFT"}),
            SearchResult(chunk_id="aapl_strategy_ai", text="Apple is integrating machine learning across its products with Apple Intelligence.", score=0.0, metadata={"ticker": "AAPL"}),
            SearchResult(chunk_id="aapl_capex", text="Capital expenditures were $10.96 billion in fiscal year 2023.", score=0.0, metadata={"ticker": "AAPL"}),
            SearchResult(chunk_id="nvda_risk_export", text="NVIDIA faces risks from U.S. export controls restricting GPU sales to China.", score=0.0, metadata={"ticker": "NVDA"}),
        ],
        "relevant_ids": {"aapl_risk_supply", "aapl_risk_macro"},
    },
    {
        "query": "How did Microsoft's cloud revenue compare to its on-premise revenue in FY2023?",
        "candidates": [
            SearchResult(chunk_id="msft_cloud_rev", text="Microsoft's Intelligent Cloud segment, which includes Azure, generated $87.9 billion in revenue in FY2023, representing the largest and fastest-growing segment.", score=0.0, metadata={"ticker": "MSFT"}),
            SearchResult(chunk_id="msft_cloud_growth", text="Azure and other cloud services revenue grew 29% year-over-year, driven by AI workloads and enterprise migration.", score=0.0, metadata={"ticker": "MSFT"}),
            SearchResult(chunk_id="msft_onprem_decline", text="On-premises server products and traditional licensing revenue continued to decline as customers migrated to cloud-based solutions.", score=0.0, metadata={"ticker": "MSFT"}),
            SearchResult(chunk_id="msft_total_rev", text="Microsoft reported total revenue of $211.9 billion for fiscal year 2023.", score=0.0, metadata={"ticker": "MSFT"}),
            SearchResult(chunk_id="googl_cloud_rev", text="Google Cloud generated $33.1 billion in revenue in fiscal year 2023.", score=0.0, metadata={"ticker": "GOOGL"}),
            SearchResult(chunk_id="aws_rev", text="AWS generated approximately $90.8 billion in revenue in FY2023.", score=0.0, metadata={"ticker": "AMZN"}),
            SearchResult(chunk_id="msft_gaming", text="Microsoft completed the Activision Blizzard acquisition for approximately $68.7 billion.", score=0.0, metadata={"ticker": "MSFT"}),
            SearchResult(chunk_id="msft_employees", text="Microsoft had approximately 221,000 employees as of June 30, 2023.", score=0.0, metadata={"ticker": "MSFT"}),
            SearchResult(chunk_id="aapl_services", text="Apple's services segment generated $85.2 billion in revenue.", score=0.0, metadata={"ticker": "AAPL"}),
            SearchResult(chunk_id="msft_dividend", text="Microsoft declared dividends of $20.2 billion in fiscal year 2023.", score=0.0, metadata={"ticker": "MSFT"}),
        ],
        "relevant_ids": {"msft_cloud_rev", "msft_cloud_growth", "msft_onprem_decline"},
    },
    {
        "query": "What is NVIDIA's strategy for the AI market and accelerated computing?",
        "candidates": [
            SearchResult(chunk_id="nvda_ai_strategy", text="NVIDIA is expanding its AI platform through next-generation GPU architectures including Hopper and Blackwell, the CUDA software ecosystem, and enterprise AI solutions.", score=0.0, metadata={"ticker": "NVDA"}),
            SearchResult(chunk_id="nvda_datacenter", text="Data Center revenue reached $47.5 billion in FY2024, driven by unprecedented demand for AI training and inference GPUs.", score=0.0, metadata={"ticker": "NVDA"}),
            SearchResult(chunk_id="nvda_software", text="NVIDIA AI Enterprise software platform provides tools for deploying production AI applications, representing a growing recurring revenue stream.", score=0.0, metadata={"ticker": "NVDA"}),
            SearchResult(chunk_id="nvda_auto_ai", text="NVIDIA's DRIVE platform provides AI computing for autonomous vehicles, with partnerships with major automakers.", score=0.0, metadata={"ticker": "NVDA"}),
            SearchResult(chunk_id="nvda_gross_margin", text="NVIDIA's gross margin was approximately 72.7% in fiscal year 2024.", score=0.0, metadata={"ticker": "NVDA"}),
            SearchResult(chunk_id="msft_ai_strategy", text="Microsoft is embedding AI across products through Copilot and Azure OpenAI Service.", score=0.0, metadata={"ticker": "MSFT"}),
            SearchResult(chunk_id="nvda_gaming", text="Gaming segment revenue was $10.4 billion in FY2024.", score=0.0, metadata={"ticker": "NVDA"}),
            SearchResult(chunk_id="amd_datacenter", text="AMD's data center revenue reached approximately $6.5 billion in FY2024.", score=0.0, metadata={"ticker": "AMD"}),
            SearchResult(chunk_id="nvda_employees", text="NVIDIA had approximately 29,600 employees as of January 2024.", score=0.0, metadata={"ticker": "NVDA"}),
            SearchResult(chunk_id="tsla_ai", text="Tesla is developing FSD through AI-first approach using vision-only perception.", score=0.0, metadata={"ticker": "TSLA"}),
        ],
        "relevant_ids": {"nvda_ai_strategy", "nvda_datacenter", "nvda_software"},
    },
    {
        "query": "How has Tesla's revenue changed from FY2021 to FY2023?",
        "candidates": [
            SearchResult(chunk_id="tsla_rev_trend", text="Tesla's total revenue grew from $53.8 billion in FY2021 to $81.5 billion in FY2022, and further to $96.8 billion in FY2023.", score=0.0, metadata={"ticker": "TSLA"}),
            SearchResult(chunk_id="tsla_rev_2023", text="Tesla reported total revenue of approximately $96.8 billion in fiscal year 2023, an increase of 19% year-over-year.", score=0.0, metadata={"ticker": "TSLA"}),
            SearchResult(chunk_id="tsla_deliveries", text="Tesla delivered approximately 1.81 million vehicles in fiscal year 2023, up from 1.31 million in FY2022.", score=0.0, metadata={"ticker": "TSLA"}),
            SearchResult(chunk_id="tsla_margin_decline", text="Tesla's automotive gross margin declined to approximately 18.2% in FY2023 from about 25.9% in FY2022 due to price reductions.", score=0.0, metadata={"ticker": "TSLA"}),
            SearchResult(chunk_id="tsla_energy", text="Tesla's energy generation and storage segment generated approximately $6.0 billion in revenue in FY2023.", score=0.0, metadata={"ticker": "TSLA"}),
            SearchResult(chunk_id="aapl_rev_trend", text="Apple's revenue grew from $365.8B in FY2021 to $394.3B in FY2022, then declined to $383.3B in FY2023.", score=0.0, metadata={"ticker": "AAPL"}),
            SearchResult(chunk_id="tsla_fsd", text="Tesla is developing Full Self-Driving through an AI-first approach.", score=0.0, metadata={"ticker": "TSLA"}),
            SearchResult(chunk_id="tsla_risk_price", text="Aggressive price cuts to maintain demand could erode margins.", score=0.0, metadata={"ticker": "TSLA"}),
            SearchResult(chunk_id="nvda_rev_trend", text="NVIDIA's revenue grew from $26.9B in FY2022 to $60.9B in FY2024.", score=0.0, metadata={"ticker": "NVDA"}),
            SearchResult(chunk_id="tsla_capex", text="Tesla's capital expenditure was approximately $8.9 billion in FY2023.", score=0.0, metadata={"ticker": "TSLA"}),
        ],
        "relevant_ids": {"tsla_rev_trend", "tsla_rev_2023"},
    },
    {
        "query": "What credit risk concerns did JPMorgan disclose regarding commercial real estate?",
        "candidates": [
            SearchResult(chunk_id="jpm_cre_risk", text="Commercial real estate exposure remains a key concern, with office vacancy rates elevated and potential for further write-downs.", score=0.0, metadata={"ticker": "JPM"}),
            SearchResult(chunk_id="jpm_credit_risk", text="JPMorgan disclosed rising credit costs driven by higher interest rates, potential consumer loan delinquencies, and normalization from historically low levels.", score=0.0, metadata={"ticker": "JPM"}),
            SearchResult(chunk_id="jpm_macro_risk", text="The macroeconomic environment presents uncertainty, with inflation persistence and potential recession scenarios affecting credit quality.", score=0.0, metadata={"ticker": "JPM"}),
            SearchResult(chunk_id="jpm_nii", text="JPMorgan's net interest income was approximately $89.3 billion in FY2023.", score=0.0, metadata={"ticker": "JPM"}),
            SearchResult(chunk_id="jpm_revenue", text="JPMorgan Chase reported total managed revenue of approximately $158.1 billion in fiscal year 2023.", score=0.0, metadata={"ticker": "JPM"}),
            SearchResult(chunk_id="bac_credit", text="Bank of America reported increased provision for credit losses in FY2023.", score=0.0, metadata={"ticker": "BAC"}),
            SearchResult(chunk_id="jpm_digital", text="JPMorgan is investing in technology modernization and digital consumer banking.", score=0.0, metadata={"ticker": "JPM"}),
            SearchResult(chunk_id="jpm_dividend", text="JPMorgan declared dividends of approximately $12.4 billion in FY2023.", score=0.0, metadata={"ticker": "JPM"}),
            SearchResult(chunk_id="aapl_risk_macro", text="Apple's operations depend on worldwide economic conditions.", score=0.0, metadata={"ticker": "AAPL"}),
            SearchResult(chunk_id="jpm_employees", text="JPMorgan had approximately 309,926 employees at the end of FY2023.", score=0.0, metadata={"ticker": "JPM"}),
        ],
        "relevant_ids": {"jpm_cre_risk", "jpm_credit_risk"},
    },
    {
        "query": "What was the Q3 2023 revenue for Apple's iPhone segment?",
        "candidates": [
            SearchResult(chunk_id="aapl_iphone_q3", text="iPhone revenue for Q3 FY2023 was $39,596 million, compared to $40,665 million in the year-ago quarter.", score=0.0, metadata={"ticker": "AAPL"}),
            SearchResult(chunk_id="aapl_total_q3", text="Total net sales for Q3 2023 were $81.8 billion, compared to $83.0 billion in Q3 2022.", score=0.0, metadata={"ticker": "AAPL"}),
            SearchResult(chunk_id="aapl_mac_q3", text="Mac revenue for Q3 FY2023 was $6,840 million.", score=0.0, metadata={"ticker": "AAPL"}),
            SearchResult(chunk_id="aapl_iphone_annual", text="iPhone revenue was $200,583 million in fiscal 2023.", score=0.0, metadata={"ticker": "AAPL"}),
            SearchResult(chunk_id="aapl_services_q3", text="Services revenue for Q3 FY2023 was $21,213 million, a new all-time high.", score=0.0, metadata={"ticker": "AAPL"}),
            SearchResult(chunk_id="msft_q3", text="Microsoft Q3 FY2023 revenue was $52.9 billion.", score=0.0, metadata={"ticker": "MSFT"}),
            SearchResult(chunk_id="aapl_guidance", text="Apple did not provide specific revenue guidance for Q4 FY2023.", score=0.0, metadata={"ticker": "AAPL"}),
            SearchResult(chunk_id="aapl_eps_q3", text="Apple's Q3 FY2023 diluted EPS was $1.26.", score=0.0, metadata={"ticker": "AAPL"}),
            SearchResult(chunk_id="samsung_mobile", text="Samsung's mobile division reported revenue of $25.1 billion in Q3 2023.", score=0.0, metadata={"ticker": "SMSN"}),
            SearchResult(chunk_id="aapl_stores_q3", text="Apple opened 3 new retail stores during Q3 FY2023.", score=0.0, metadata={"ticker": "AAPL"}),
        ],
        "relevant_ids": {"aapl_iphone_q3"},
    },
    {
        "query": "How much did Google spend on R&D expenses in FY2023?",
        "candidates": [
            SearchResult(chunk_id="googl_rd", text="Research and development expenses were $45,427 million in fiscal year 2023, an increase from $39,500 million in 2022.", score=0.0, metadata={"ticker": "GOOGL"}),
            SearchResult(chunk_id="googl_rd_pct", text="R&D spending represented approximately 14.8% of Alphabet's total revenue in FY2023.", score=0.0, metadata={"ticker": "GOOGL"}),
            SearchResult(chunk_id="aapl_rd", text="Apple spent approximately $29.9 billion on R&D in fiscal year 2023.", score=0.0, metadata={"ticker": "AAPL"}),
            SearchResult(chunk_id="msft_rd", text="Microsoft's research and development expenses were $27.2 billion in FY2023.", score=0.0, metadata={"ticker": "MSFT"}),
            SearchResult(chunk_id="googl_rev", text="Alphabet reported total revenue of $307.4 billion in fiscal year 2023.", score=0.0, metadata={"ticker": "GOOGL"}),
            SearchResult(chunk_id="meta_rd", text="Meta Platforms spent approximately $35.3 billion on R&D in FY2023.", score=0.0, metadata={"ticker": "META"}),
            SearchResult(chunk_id="googl_employees", text="Alphabet had approximately 182,502 employees at the end of FY2023.", score=0.0, metadata={"ticker": "GOOGL"}),
            SearchResult(chunk_id="googl_cloud", text="Google Cloud generated $33.1 billion in revenue in FY2023.", score=0.0, metadata={"ticker": "GOOGL"}),
            SearchResult(chunk_id="nvda_rd", text="NVIDIA spent approximately $8.7 billion on R&D in FY2024.", score=0.0, metadata={"ticker": "NVDA"}),
            SearchResult(chunk_id="googl_capex", text="Alphabet's capital expenditure was approximately $32.3 billion in FY2023.", score=0.0, metadata={"ticker": "GOOGL"}),
        ],
        "relevant_ids": {"googl_rd", "googl_rd_pct"},
    },
    {
        "query": "What metaverse and Reality Labs risks did Meta disclose?",
        "candidates": [
            SearchResult(chunk_id="meta_rl_risk", text="Reality Labs continues to incur significant operating losses, with $16.1 billion in losses in FY2023. The success of our metaverse initiatives is uncertain and depends on user adoption of VR/AR hardware.", score=0.0, metadata={"ticker": "META"}),
            SearchResult(chunk_id="meta_vr_competition", text="We face competition in the VR/AR market from Apple Vision Pro, Sony PlayStation VR, and other entrants, which may limit Quest headset adoption.", score=0.0, metadata={"ticker": "META"}),
            SearchResult(chunk_id="meta_metaverse_uncertain", text="Metaverse technologies are at an early stage and it is unclear whether our investments will yield the intended benefits. User adoption timelines are highly uncertain.", score=0.0, metadata={"ticker": "META"}),
            SearchResult(chunk_id="meta_ad_rev", text="Advertising revenue reached $131.9 billion in FY2023, driven by improvements in ad targeting.", score=0.0, metadata={"ticker": "META"}),
            SearchResult(chunk_id="meta_ai_llama", text="Meta released Llama 2 as open-source, pursuing an open AI strategy.", score=0.0, metadata={"ticker": "META"}),
            SearchResult(chunk_id="aapl_vision_pro", text="Apple launched Vision Pro, its spatial computing headset, in early 2024.", score=0.0, metadata={"ticker": "AAPL"}),
            SearchResult(chunk_id="meta_users", text="Meta's family of apps had 3.98 billion monthly active people in Q4 2023.", score=0.0, metadata={"ticker": "META"}),
            SearchResult(chunk_id="meta_capex", text="Meta's capital expenditure was approximately $28.1 billion in FY2023.", score=0.0, metadata={"ticker": "META"}),
            SearchResult(chunk_id="snap_ar", text="Snap is investing in AR glasses and Lens Studio tools.", score=0.0, metadata={"ticker": "SNAP"}),
            SearchResult(chunk_id="meta_dividend", text="Meta initiated its first-ever quarterly dividend in February 2024.", score=0.0, metadata={"ticker": "META"}),
        ],
        "relevant_ids": {"meta_rl_risk", "meta_vr_competition", "meta_metaverse_uncertain"},
    },
    {
        "query": "What was Amazon's capital expenditure in FY2023 driven by?",
        "candidates": [
            SearchResult(chunk_id="amzn_capex", text="Amazon's capital expenditure, including finance leases, was approximately $48.4 billion in fiscal year 2023, primarily driven by AWS infrastructure expansion.", score=0.0, metadata={"ticker": "AMZN"}),
            SearchResult(chunk_id="amzn_capex_detail", text="Investments in technology infrastructure accounted for the majority of Amazon's capital spending, with significant expansion of data center capacity globally.", score=0.0, metadata={"ticker": "AMZN"}),
            SearchResult(chunk_id="amzn_rev", text="Amazon reported total revenue of $574.8 billion in fiscal year 2023.", score=0.0, metadata={"ticker": "AMZN"}),
            SearchResult(chunk_id="amzn_fcf", text="Amazon generated free cash flow of $36.8 billion in FY2023, a significant improvement from negative free cash flow in FY2022.", score=0.0, metadata={"ticker": "AMZN"}),
            SearchResult(chunk_id="msft_capex", text="Microsoft's capital expenditure was approximately $28.1 billion in FY2023.", score=0.0, metadata={"ticker": "MSFT"}),
            SearchResult(chunk_id="amzn_aws_rev", text="AWS generated approximately $90.8 billion in revenue in FY2023.", score=0.0, metadata={"ticker": "AMZN"}),
            SearchResult(chunk_id="googl_capex", text="Alphabet's capital expenditure was approximately $32.3 billion in FY2023.", score=0.0, metadata={"ticker": "GOOGL"}),
            SearchResult(chunk_id="amzn_employees", text="Amazon had approximately 1.53 million employees at the end of FY2023.", score=0.0, metadata={"ticker": "AMZN"}),
            SearchResult(chunk_id="amzn_prime", text="Amazon Prime continued to be a key driver of customer loyalty and engagement.", score=0.0, metadata={"ticker": "AMZN"}),
            SearchResult(chunk_id="amzn_ads", text="Amazon's advertising revenue grew to approximately $46.9 billion in FY2023.", score=0.0, metadata={"ticker": "AMZN"}),
        ],
        "relevant_ids": {"amzn_capex", "amzn_capex_detail"},
    },
]


# ---------------------------------------------------------------------------
# Metric computation helpers
# ---------------------------------------------------------------------------

def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Recall@K: fraction of relevant items found in top K results."""
    if not relevant_ids:
        return 0.0
    top_k_ids = set(ranked_ids[:k])
    found = top_k_ids & relevant_ids
    return len(found) / len(relevant_ids)


def mrr(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    """Mean Reciprocal Rank: 1 / rank of the first relevant result."""
    for i, chunk_id in enumerate(ranked_ids):
        if chunk_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------------------
# Retrieval strategies
# ---------------------------------------------------------------------------

def execute_vector_search(query: str, candidates: list[SearchResult]) -> tuple[list[SearchResult], float]:
    """Execute dense vector search via sentence-transformers cosine similarity."""
    t0 = time.perf_counter()
    q_vec = embed_query(query)
    d_vecs = embed_texts([c.text for c in candidates])
    scores = d_vecs @ q_vec
    latency = (time.perf_counter() - t0) * 1000

    results = [
        SearchResult(chunk_id=c.chunk_id, text=c.text, score=float(s), metadata=c.metadata)
        for c, s in zip(candidates, scores)
    ]
    results.sort(key=lambda r: r.score, reverse=True)
    return results, latency


def execute_sparse_search(query: str, candidates: list[SearchResult]) -> tuple[list[SearchResult], float]:
    """Execute BM25 keyword search over candidates."""
    t0 = time.perf_counter()
    tokenized_docs = [tokenize(c.text) for c in candidates]
    bm25 = BM25Okapi(tokenized_docs)
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)
    latency = (time.perf_counter() - t0) * 1000

    results = [
        SearchResult(chunk_id=c.chunk_id, text=c.text, score=float(s), metadata=c.metadata)
        for c, s in zip(candidates, scores)
    ]
    results.sort(key=lambda r: r.score, reverse=True)
    return results, latency


def execute_hybrid_search(query: str, candidates: list[SearchResult]) -> tuple[list[SearchResult], float]:
    """Execute hybrid search: dense + sparse fused with Reciprocal Rank Fusion."""
    dense_res, dense_lat = execute_vector_search(query, candidates)
    sparse_res, sparse_lat = execute_sparse_search(query, candidates)

    t0 = time.perf_counter()
    fused = reciprocal_rank_fusion([dense_res, sparse_res], k=60, top_n=len(candidates))
    rrf_lat = (time.perf_counter() - t0) * 1000

    total_latency = dense_lat + sparse_lat + rrf_lat
    return fused, total_latency


def execute_hybrid_rerank_search(query: str, candidates: list[SearchResult]) -> tuple[list[SearchResult], float]:
    """Execute hybrid search followed by cross-encoder reranking."""
    fused, hybrid_lat = execute_hybrid_search(query, candidates)

    t0 = time.perf_counter()
    reranked = rerank(query, fused, top_k=len(candidates))
    rerank_lat = (time.perf_counter() - t0) * 1000

    total_latency = hybrid_lat + rerank_lat
    return reranked, total_latency


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark() -> dict:
    print("=" * 80)
    print("FINANCIAL DOCUMENT RAG — RETRIEVAL & RERANKING BENCHMARK")
    print("=" * 80)
    print(f"Number of test queries: {len(BENCHMARK_CASES)}")
    print("Strategies: [1] vector_only  [2] hybrid (dense+BM25+RRF)  [3] hybrid+rerank (cross-encoder)\n")

    # Warm up models
    print("Warming up models...")
    _ = embed_query("warmup query")
    _ = embed_texts(["warmup document text"])
    _ = rerank("warmup query", [SearchResult(chunk_id="w", text="warmup text", score=0.0, metadata={})], top_k=1)
    print("Models warmed up.\n")

    strategies = {
        "vector_only": execute_vector_search,
        "hybrid": execute_hybrid_search,
        "hybrid_rerank": execute_hybrid_rerank_search,
    }

    all_results = {}

    for strategy_name, strategy_fn in strategies.items():
        print(f"--- Running Strategy: {strategy_name} ---")
        metrics_list = []

        for idx, case in enumerate(BENCHMARK_CASES, 1):
            results, latency = strategy_fn(case["query"], case["candidates"])
            ranked_ids = [r.chunk_id for r in results]
            relevant = case["relevant_ids"]

            r5 = recall_at_k(ranked_ids, relevant, 5)
            r10 = recall_at_k(ranked_ids, relevant, 10)
            m = mrr(ranked_ids, relevant)

            metrics_list.append({
                "recall_at_5": r5,
                "recall_at_10": r10,
                "mrr": m,
                "latency_ms": latency,
            })
            print(f"  Q{idx}: {case['query'][:55]:<55} | R@5={r5:.2f} R@10={r10:.2f} MRR={m:.2f} Lat={latency:5.1f}ms")

        latencies = [m["latency_ms"] for m in metrics_list]
        sorted_lat = sorted(latencies)
        p95_idx = int(len(sorted_lat) * 0.95)

        agg = {
            "recall_at_5": statistics.mean([m["recall_at_5"] for m in metrics_list]),
            "recall_at_10": statistics.mean([m["recall_at_10"] for m in metrics_list]),
            "mrr": statistics.mean([m["mrr"] for m in metrics_list]),
            "latency_mean_ms": statistics.mean(latencies),
            "latency_p50_ms": statistics.median(latencies),
            "latency_p95_ms": sorted_lat[p95_idx] if len(sorted_lat) > 1 else sorted_lat[0],
            "per_case": metrics_list,
        }
        all_results[strategy_name] = agg

        print(f"\n  SUMMARY for {strategy_name}:")
        print(f"    Recall@5:       {agg['recall_at_5']:.4f}")
        print(f"    Recall@10:      {agg['recall_at_10']:.4f}")
        print(f"    MRR:            {agg['mrr']:.4f}")
        print(f"    Latency (mean): {agg['latency_mean_ms']:.1f} ms")
        print(f"    Latency (p50):  {agg['latency_p50_ms']:.1f} ms")
        print(f"    Latency (p95):  {agg['latency_p95_ms']:.1f} ms\n")

    # Comparison summary table
    print("=" * 80)
    print(f"{'Strategy':<18} | {'Recall@5':<10} | {'Recall@10':<10} | {'MRR':<8} | {'p50 Lat (ms)':<14} | {'p95 Lat (ms)':<14}")
    print("-" * 80)
    for name in ["vector_only", "hybrid", "hybrid_rerank"]:
        data = all_results[name]
        print(f"{name:<18} | {data['recall_at_5']:<10.4f} | {data['recall_at_10']:<10.4f} | {data['mrr']:<8.4f} | {data['latency_p50_ms']:<14.1f} | {data['latency_p95_ms']:<14.1f}")
    print("=" * 80)

    # Relative improvement
    vec = all_results["vector_only"]
    hyb = all_results["hybrid"]
    rer = all_results["hybrid_rerank"]
    print("\nKEY TAKEAWAYS:")
    mrr_diff_hyb = hyb["mrr"] - vec["mrr"]
    mrr_diff_rer = rer["mrr"] - vec["mrr"]
    print(f"- Hybrid vs Vector-only:        MRR {mrr_diff_hyb:+.3f} ({(mrr_diff_hyb/vec['mrr'])*100:+.1f}%)")
    print(f"- Hybrid+Rerank vs Vector-only:  MRR {mrr_diff_rer:+.3f} ({(mrr_diff_rer/vec['mrr'])*100:+.1f}%)")
    print(f"- Cross-encoder latency cost:    +{rer['latency_mean_ms'] - hyb['latency_mean_ms']:.1f} ms per query on CPU")

    return all_results


def main():
    results = run_benchmark()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = Path(__file__).parent.parent / "results" / f"retrieval_benchmark_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clean_output = {
        "timestamp": timestamp,
        "benchmark_type": "vector_vs_hybrid_vs_rerank",
        "num_cases": len(BENCHMARK_CASES),
        "strategies": {},
    }

    for strat_name, metrics in results.items():
        clean_output["strategies"][strat_name] = {
            k: v for k, v in metrics.items() if k != "per_case"
        }

    with open(output_path, "w") as f:
        json.dump(clean_output, f, indent=2)

    print(f"\n[Benchmark complete] Real metrics saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    main()
