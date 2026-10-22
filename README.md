# Financial Document Intelligence Engine

> A backend-first, agentic RAG system for SEC filings and earnings call transcripts. Every claim is grounded in a source. If evidence is weak, the system retries or abstains.

## Quick Start

```bash
# 1. Clone and configure
git clone <repo-url>
cd Financial-Document-Agentic-RAG-Engine
cp .env.example .env
# Edit .env with your API keys (Groq, Gemini)

# 2. Start everything
docker compose up --build -d

# 3. Verify
curl http://localhost:8000/health
```

## Architecture

See [docs/architecture.md](docs/architecture.md) for diagrams.

## Tech Stack

| Component | Technology |
|---|---|
| API | FastAPI + Pydantic + uvicorn |
| Vector DB | Qdrant (dense + sparse vectors) |
| Cache | Redis |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Groq (primary) + Gemini (fallback) |
| Agent | LangGraph |
| Evaluation | RAGAS |
| Observability | Prometheus + Grafana + LangSmith |

## Project Status

- [x] Day 1: Repo, Docker, config, loaders, chunker, dedup
- [ ] Day 2: Embeddings, Qdrant upsert, /search/dense
- [ ] Day 3: Sparse/BM25, RRF, /search/hybrid
- [ ] Day 4: Cross-encoder rerank, golden set, benchmarks
- [ ] Day 5: LLM client, citations, validation
- [ ] Day 6: Confidence scorer, threshold tuning
- [ ] Day 7: LangGraph agent (generate/rewrite/abstain)
- [ ] Day 8: Caching, lineage, delta re-ingest
- [ ] Day 9: SSE streaming, rate limiting, observability
- [ ] Day 10: Final benchmarks, RAGAS, interview prep

## Results

*All benchmarks and metrics are in [/results](results/). No fabricated numbers.*

## Documentation

- [Architecture](docs/architecture.md) — system diagrams
- [Decisions](docs/DECISIONS.md) — design tradeoffs
- [Version History](docs/versions/) — daily changelogs
- [Learning Guides](docs/learn/) — interview prep material
- [Interview Defense](docs/interview-defense.md) — 5-min walkthrough + hard questions

## License

MIT
