# Day 3 — Learning Guide: BM25 Sparse Vectors, Reciprocal Rank Fusion, and Hybrid Retrieval

---

## 1. What We Built Today in Plain English

Yesterday, we gave our system a "semantic brain" using neural embeddings (`all-MiniLM-L6-v2`). Semantic embeddings are great at recognizing when words *mean* the same thing (e.g., matching "top-line expansion" to "revenue grew").

However, semantic search has a glaring blind spot: it frequently misses **exact, specific keywords**. In financial filings, numbers, ticker symbols, and regulatory headers matter immensely. If an analyst queries for `"Item 1A Risk Factors"` or `"AAPL 10-K/A"`, semantic embeddings might return a generic paragraph discussing general business risks rather than the actual section heading.

Today, we built a **hybrid retrieval engine** that combines the best of both worlds:
1. **The Lexical Engine (BM25 Sparse Vectors)**: We implemented Okapi BM25 keyword matching encoded as sparse vectors in Qdrant. Common words ("the", "and") receive low weights, while distinctive terms ("amortization", "EBITDA") receive high weights.
2. **Co-Located Named Vectors in Qdrant**: Instead of maintaining a separate Elasticsearch or OpenSearch cluster, we configured Qdrant with named vectors: `"dense"` (384-dimensional dense vectors) and `"sparse"` (variable-length BM25 vectors) living side-by-side in the same collection.
3. **Reciprocal Rank Fusion (RRF)**: When dense search returns 10 candidates and sparse search returns 10 candidates, how do you combine them? Raw scores cannot be added because cosine similarity ($0.3 - 0.9$) and BM25 scores ($0.0 - 25.0+$) are on completely different scales. We implemented RRF, an algorithm that blends ranked lists based on their *positions* rather than their raw scores.
4. **Hybrid Search Endpoint (`POST /api/v1/search/hybrid`)**: An endpoint that executes dense and sparse searches in parallel, fuses the candidate lists via RRF, and returns a balanced, high-precision set of chunks.

---

## 2. Why Each Design Decision Was Made

### A. Why Co-Locate Sparse Vectors in Qdrant Instead of Running Elasticsearch?
- **Decision**: Store BM25 vectors directly inside Qdrant as named vectors (`sparse`).
- **Why**: Traditional enterprise RAG architectures run Elasticsearch alongside a vector DB. This requires running two databases, paying for 4GB+ JVM memory overhead, keeping two datastores in sync on updates and deletes, and dealing with split-brain issues. Qdrant natively supports sparse vectors and inverted indexes in Rust, writing both dense embeddings and sparse BM25 weights in a single atomic transaction.
- **Alternatives**:
  - *Elasticsearch/OpenSearch*: Powerful, but heavy, complex, and violates our zero-cost, single-command compose goal.
  - *rank_bm25 purely in Python memory*: Doesn't scale past memory limits, doesn't support Qdrant's payload filtering (`ticker`, `doc_type`), and must be recomputed on service restarts.

### B. Why Reciprocal Rank Fusion (RRF) Instead of Score Normalization?
- **Decision**: Use rank-based fusion with formula $RRF(d) = \sum \frac{1}{k + r(d)}$.
- **Why**: Dense models output bounded similarity scores ($[0.0, 1.0]$), whereas BM25 outputs unbounded scores ($[0.0, \infty)$) that vary drastically depending on document length and term rarity. Min-max normalization fails whenever a single document has an unusually high BM25 score, compressing all other scores toward zero. RRF is immune to scale differences because it only cares about rank positions ($1^{\text{st}}, 2^{\text{nd}}, 3^{\text{rd}}$).
- **Alternatives**:
  - *Linear score blending ($\alpha \cdot S_{\text{dense}} + (1-\alpha) \cdot S_{\text{sparse}}$)*: Requires continuous calibration of $\alpha$ and complex per-query score normalization.
  - *Cross-encoder reranking only*: High latency if run across all documents; RRF efficiently selects the best candidate pool of 50 chunks for downstream cross-encoders (Day 4).

### C. Why Hash-Based Term Indexing?
- **Decision**: Hash each unique word to a 32-bit integer via `md5(term)[:4]` to determine its sparse vector index.
- **Why**: A centralized vocabulary dictionary (`{"revenue": 0, "operating": 1, ...}`) must be synchronized across distributed workers and persisted to disk. Hash-based mapping is stateless: any query or chunk can be vectorized independently on any thread without looking up an external vocabulary table. With a 32-bit address space ($4.29 \times 10^9$ slots), collision probability across 50,000 domain terms is $< 0.03\%$.

---

## 3. Key Concepts Explained from First Principles

### A. What is BM25 (Best Matching 25)?

BM25 is the industry gold standard for keyword retrieval. Given a query term $t$ and document $d$:

$$\text{BM25Score}(D, Q) = \sum_{t \in Q} \text{IDF}(t) \cdot \frac{f(t, D) \cdot (k_1 + 1)}{f(t, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

Where:
1. **$f(t, D)$ (Term Frequency)**: How many times word $t$ appears in chunk $D$. Unlike raw TF, BM25 uses saturation: mentioning "revenue" 10 times is not 10 times more relevant than mentioning it twice. Parameter $k_1$ (typically $1.2 - 1.5$) controls this saturation ceiling.
2. **$\text{IDF}(t)$ (Inverse Document Frequency)**: How rare the term is across the entire filing library:
   $$\text{IDF}(t) = \ln\left(\frac{N - n(t) + 0.5}{n(t) + 0.5} + 1\right)$$
   A word appearing in every document (e.g., "company") receives an IDF near 0. A rare word appearing in only 2 documents (e.g., "subordinated") receives a high IDF.
3. **Length Normalization ($\frac{|D|}{\text{avgdl}}$)**: A 2,000-word chunk mentions terms more often simply by being longer. Parameter $b$ (typically $0.75$) penalizes long chunks so concise, relevant chunks aren't drowned out.

---

### B. Representing BM25 as a Sparse Vector

A dense vector has values at *every* dimension:
$$\vec{v}_{\text{dense}} = [0.12, -0.45, 0.89, \dots] \quad (384 \text{ floats})$$

A **sparse vector** has non-zero values at only a tiny fraction of its dimensions, represented as two aligned arrays of **indices** and **values**:
```python
SparseVector(
    indices = [ 10482,   89201,  449102 ],
    values  = [  2.41,    1.18,    4.55 ]
)
```

#### How the Dot Product Computes BM25:
- **Document Chunk**: Stores indices of its words with their TF length-normalized component:
  $$\text{Value}_d(t) = \frac{f(t, D) \cdot (k_1 + 1)}{f(t, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{\text{avgdl}})}$$
- **Query**: Stores indices of query terms with their corpus IDF values:
  $$\text{Value}_q(t) = \text{IDF}(t)$$
- **Qdrant Dot Product**:
  $$\vec{v}_q \cdot \vec{v}_d = \sum_{t \in Q \cap D} \text{Value}_q(t) \cdot \text{Value}_d(t) \equiv \text{Exact BM25 Score!}$$

---

### C. Reciprocal Rank Fusion (RRF) Formula

When multiple search algorithms rank the same corpus:
- Dense Search Rank: $r_{\text{dense}}(d) \in \{1, 2, 3, \dots\}$
- Sparse Search Rank: $r_{\text{sparse}}(d) \in \{1, 2, 3, \dots\}$

The RRF score for document $d$ across all ranked lists $M$ is:
$$\text{RRF\_Score}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$

Where $k$ is a constant (typically $60$).

#### Concrete Example:
Suppose chunk $A$ is ranked #1 in Dense and #3 in Sparse.
Chunk $B$ is ranked #2 in Dense and #1 in Sparse.
Chunk $C$ is ranked #3 in Dense and absent from Sparse ($r = \infty$).

With $k = 60$:
- **Chunk A**: $\frac{1}{60 + 1} + \frac{1}{60 + 3} = \frac{1}{61} + \frac{1}{63} \approx 0.01639 + 0.01587 = \mathbf{0.03226}$
- **Chunk B**: $\frac{1}{60 + 2} + \frac{1}{60 + 1} = \frac{1}{62} + \frac{1}{61} \approx 0.01613 + 0.01639 = \mathbf{0.03252}$ (Winner!)
- **Chunk C**: $\frac{1}{60 + 3} + 0 = \frac{1}{63} \approx \mathbf{0.01587}$

**Why Chunk B wins**: It ranked near the top in **both** retrieval methods. RRF inherently rewards consensus.

---

## 4. End-to-End Request Walkthrough: `POST /api/v1/search/hybrid`

```
User Request: POST /api/v1/search/hybrid
Payload: {"query": "iPhone revenue headwinds currency volatility", "top_k": 5}
    │
    ▼
1. routes.py: hybrid_search(request)
    ├── Calculates retrieval_k = min(top_k * 2, 50) = 10
    │
    ├── 2. Parallel Dense Retrieval: vector_store.search_dense()
    │      ├── embedder.py: embed_query("iPhone revenue...") → 384-dim dense vector
    │      └── Qdrant.query_points(using="dense", limit=10)
    │          Returns: [AAPL_chunk_01 (0.61), AAPL_chunk_02 (0.58), MSFT_chunk_01 (0.10), ...]
    │
    ├── 3. Parallel Sparse Retrieval: vector_store.search_sparse()
    │      ├── sparse.py: encode_sparse_query("iPhone revenue...")
    │      │   - Tokenize: ["iphone", "revenue", "headwinds", "currency", "volatility"]
    │      │   - Fetch IDFs: {"headwinds": 3.8, "volatility": 3.4, "iphone": 2.9, ...}
    │      │   - Output: SparseVector(indices=[...], values=[3.8, 3.4, 2.9, ...])
    │      └── Qdrant.query_points(using="sparse", limit=10)
    │          Returns: [AAPL_chunk_02 (7.02), AAPL_chunk_01 (5.14), ...]
    │
    └── 4. Rank Fusion: rrf.reciprocal_rank_fusion([dense_results, sparse_results], k=60, top_n=5)
           - Accumulates reciprocal rank scores across lists
           - Identifies that AAPL_chunk_02 ranks #2 dense and #1 sparse
           - Sorts fused results by composite score descending
           - Returns top 5 SearchResult objects with latency_ms
```

---

## 5. Interview Defense & Technical Questions

### Q1: "Why not just use dense embeddings for everything? When does dense retrieval fail?"
**Senior Answer**:
> "Dense retrieval maps entire passages into a continuous latent space. It excels at semantic similarity, but fundamentally struggles with:
> 1. Exact entity and symbol matching: Financial codes like '10-K/A', 'Form 8-K', or 'Item 1A' get diffused across 384 dimensions. A dense model might judge a passage about 'Item 7 MD&A' as semantically close to 'Item 1A Risk Factors' because both discuss financial operations.
> 2. Rare or out-of-vocabulary terms: Specialized acronyms (e.g. 'EBITDAR', 'SOFR transition') or company-specific product codenames that were underrepresented in the pretraining corpus.
> 3. Numbers and financial metrics: To a dense model, '$383 billion' and '$384 billion' have nearly identical embeddings ($>0.99$ cosine similarity) despite being factually distinct. BM25 provides lexical grounding that keeps exact keywords anchored."

### Q2: "Why did you choose Reciprocal Rank Fusion over learned linear combinations like $\alpha \cdot \text{Dense} + (1-\alpha) \cdot \text{Sparse}$?"
**Senior Answer**:
> "Linear score combination requires calibrating the blending weight $\alpha$ and normalizing scores. Dense scores are bounded in $[0, 1]$, while BM25 scores are unbounded and depend on document length and term frequency. If a user enters a query with a very rare keyword, the BM25 score might spike to 35.0, completely overpowering the dense score. Normalization techniques like Min-Max scale the scores of a single query, but fail when one candidate is an extreme outlier.
> RRF replaces raw scores with rank positions ($1 / (k + r)$), making it completely scale-invariant and robust across different query types without needing per-dataset weight tuning."

### Q3: "What is the role of the smoothing constant $k$ in RRF, and why is it usually set to 60?"
**Senior Answer**:
> "The constant $k$ in $1 / (k + r)$ dampens the penalty of ranking lower in a list. If $k=1$, rank #1 gets a score of $0.5$ and rank #2 gets $0.33$ — a massive 34% drop. Under small $k$, a single #1 ranking in one list completely eclipses a document that ranks #2 in both lists.
> By setting $k=60$ (the empirical standard from Cormack et al.), rank #1 gets $1/61 \approx 0.01639$ and rank #2 gets $1/62 \approx 0.01613$ — a subtle difference. This ensures that documents appearing near the top of *multiple* candidate lists receive a compounded boost that outperforms a document that placed first in only one list."

### Q4: "How does Qdrant implement sparse vector search under the hood?"
**Senior Answer**:
> "Under the hood, Qdrant constructs an **inverted index** for sparse vectors. Each unique index dimension (representing a term hash) maps to a posting list containing document IDs and their corresponding term weights.
> When a sparse vector query arrives, Qdrant only reads the posting lists corresponding to the non-zero query indices. It computes dot-product accumulators across the postings using WAND (Weak AND) or Block-Max WAND pruning, skipping documents that cannot mathematically beat the current top-$k$ threshold. This achieves sub-millisecond keyword retrieval directly in Rust without scanning the entire collection."

---

## 6. Self-Assessment: "Explain It Back" Checks

1. **Why does multiplying query IDF by document TF yield the BM25 formula?**
   *(Check: Because the dot product between query sparse vector $[IDF_1, IDF_2, ...]$ and document sparse vector $[TF_1, TF_2, ...]$ sums $IDF_i \times TF_i$ across overlapping terms).*
2. **What happens if a user searches for a term not in the corpus during sparse search?**
   *(Check: Document frequency is 0, so IDF is 0; the term is omitted from the sparse query vector, returning empty results or falling back to dense search in hybrid mode).*
3. **Why do we retrieve $top\_k \times 2$ in hybrid search before calling RRF?**
   *(Check: To ensure sufficient overlap between both candidate lists; a document ranked #11 in dense and #11 in sparse could easily be in the overall top 5 of fused results).*
4. **Why is cosine distance used for dense vectors but dot product for sparse vectors?**
   *(Check: Dense embeddings are unit-normalized directions where cosine measures angle; sparse vectors store non-normalized term weights whose dot product directly computes the BM25 score).*
5. **How does hash-based term indexing prevent state synchronization issues?**
   *(Check: Hashing terms to 32-bit integers is deterministic and stateless, allowing any worker to compute vector indices without querying a shared database).*

---

## 7. Common Failure Modes & How to Debug

1. **Empty Sparse Search Results**:
   - *Cause*: Query terms were stripped by tokenization (e.g., all words had length $< 2$) or none of the query terms exist in the BM25 corpus dictionary.
   - *Debug*: Check `tokenize(query)` output and verify `doc_freq > 0` for query words.
2. **Dense Search Outperforming Sparse on Exact Numbers**:
   - *Cause*: Stopword or token filtering removed numbers, or the chunker split numbers mid-string.
   - *Debug*: Ensure `re.findall(r'\b[a-z0-9]+\b')` preserves digits like `"383"` and `"2023"`.
3. **Qdrant Named Vector Schema Mismatch**:
   - *Cause*: Attempting to query `"sparse"` vector on a collection initialized with an unnamed single vector.
   - *Debug*: Ensure `init_collection()` configured `vectors_config={"dense": ...}` and `sparse_vectors_config={"sparse": ...}`.

---

## 8. Recommended Reading for Day 4 (Reranking & Golden Set Evaluation)

- **Cross-Encoders for Re-Ranking**: Sentence-Transformers Cross-Encoder documentation (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
- **Evaluation Metrics in IR**: Understanding Mean Reciprocal Rank (MRR), Precision@K, and Recall@K.
- **RAG Golden Sets**: Best practices for constructing evaluation datasets with synthetic and human-annotated financial queries.

