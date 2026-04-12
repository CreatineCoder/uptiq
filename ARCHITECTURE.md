# Architecture Overview

This document describes the technical design, data flow, and safety mechanisms of the RAG Benchmarking Framework. The system evaluates and compares Naive RAG (linear, single-pass) against Agentic RAG (self-correcting, iterative) on 1,100 HotpotQA multi-hop reasoning queries.

---

## 1. High-Level System Architecture

The framework is organized into five sequential stages: Data Ingestion → Vector Indexing → Agent Processing → Evaluation → Visualization.

```
┌──────────────────────────────────────────────────────────────────────┐
│                        BENCHMARK PIPELINE                            │
│                                                                      │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────────────┐    │
│  │  HotpotQA   │───▶│    Query     │───▶│  Agent A: Naive RAG  │    │
│  │  Dataset    │    │  Dispatcher  │    └──────────┬───────────┘    │
│  │ (1,100 q's) │    │              │               │                 │
│  └─────────────┘    │              │    ┌──────────▼───────────┐    │
│         │           │              │───▶│  Agent B: Agentic RAG│    │
│         ▼           └──────────────┘    └──────────┬───────────┘    │
│  ┌─────────────┐                                   │                 │
│  │  ChromaDB   │◀──────────────────────────────────┘                │
│  │  + BM25     │          (both agents query the same index)         │
│  │   Index     │                                                      │
│  └─────────────┘         ┌─────────────────────┐                    │
│                           │   Result Collector  │                    │
│                           │  (JSONL + checkpts) │                    │
│                           └──────────┬──────────┘                   │
│                                      │                               │
│                    ┌─────────────────┼──────────────┐               │
│                    │                 │              │                │
│          ┌─────────▼──────┐  ┌───────▼──────┐  ┌───▼────────────┐  │
│          │  Quantitative  │  │ LLM-as-Judge │  │ Failure Mode   │  │
│          │ Metrics        │  │ (GPT-4o-mini)│  │   Analyzer     │  │
│          │ EM, F1, R@5,   │  │ Correctness  │  │ Retrieval /    │  │
│          │ MRR            │  │ Completeness │  │ Reasoning /    │  │
│          │                │  │ Reasoning Q  │  │ Success        │  │
│          └────────┬───────┘  └──────┬───────┘  └──────┬─────────┘  │
│                   │                 │                  │             │
│          ┌────────▼─────────────────▼──────────────────▼──────────┐ │
│          │               Analysis & Visualization                  │ │
│          │   • 12+ chart types (radar, scatter, donut, bar)        │ │
│          │   • Statistical significance tests                      │ │
│          │   • Interactive Streamlit dashboard (5 tabs)            │ │
│          └────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. Dataset and Preprocessing

### Source
**HotpotQA** (`fullwiki` configuration, validation split) via HuggingFace `datasets`. Each question requires synthesizing information from two or more Wikipedia passages — making it ideal for testing multi-hop reasoning.

### Size
1,100 queries selected from the validation split (7,405 total available).

### Preprocessing Pipeline

```
HuggingFace datasets API
         │
         ▼
hotpot_qa 'fullwiki' validation split
         │
         ▼  select(range(1100))
         │
         ▼
For each item:
  context dict → reconstruct passage strings
  (title + sentences[] → "Title: X\n sentence1 sentence2...")
         │
         ▼
JSONL record:
  { id, question, gold_answer, gold_context,
    dataset="hotpotqa", difficulty="multi-hop",
    supporting_facts: {titles, sent_ids} }
         │
         ▼
data/processed/benchmark_dataset.jsonl  (1,100 lines)
         │
         ▼
TextChunker (chunk_size=600, chunk_overlap=150)
         │
         ▼
ChromaDB + BAAI/bge-small-en-v1.5 embeddings
  collection: benchmark_corpus_bge_small
  persist: data/corpus/chroma_db_bge_small
```

---

## 3. Naive RAG Architecture (Baseline)

Naive RAG is the industry-standard baseline: a strictly linear, one-pass retrieve-then-generate pipeline with no ability to verify or self-correct its retrievals.

```
┌─────────────┐     ┌───────────────┐     ┌───────────────┐     ┌──────────────┐
│    User     │────▶│   Retriever   │────▶│   Top-K Docs  │────▶│  LLM (GPT-   │
│  Question   │     │  (Hybrid RRF) │     │  (k=10)       │     │  4o-mini)    │
└─────────────┘     └───────┬───────┘     └───────────────┘     └──────┬───────┘
                            │                                           │
                    ┌───────▼───────┐                          ┌───────▼───────┐
                    │   ChromaDB    │                          │   Predicted   │
                    │  dense search │                          │    Answer     │
                    │  + BM25 (RRF) │                          └───────────────┘
                    └───────────────┘
```

### Flow

1. **Retrieve** — Embed the query with `bge-small-en-v1.5`, run cosine similarity search in ChromaDB (dense), run BM25 keyword search in parallel, merge both result sets via Reciprocal Rank Fusion (RRF) to get top-10 chunks.
2. **Generate** — Format retrieved chunks + question into a prompt and call `gpt-4o-mini`.
3. **Output** — Return whatever the LLM predicts from that single pass.

**Limitation:** If the hybrid search returns semantically similar but factually wrong passages, the LLM has no mechanism to detect or recover from this.

---

## 4. Agentic RAG Architecture

Agentic RAG is implemented as a **LangGraph state machine** (`AgenticRAGAgent`) with five nodes and conditional routing. It evaluates its own retrieval quality and iteratively refines queries when retrieval is poor.

```
                    ┌─────────────────────┐
                    │    User Question     │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │    EXPAND Node      │
                    │  HyDE: generate a   │
                    │  hypothetical       │
                    │  Wikipedia passage  │
                    └──────────┬──────────┘
                               │
              ┌────────────────▼────────────────┐
              │          RETRIEVE Node           │◀──────────────────┐
              │  Pass 1: embed(original query)   │                   │
              │  Pass 2: embed(HyDE passage)     │                   │
              │  → merge via RRF → top-20 chunks │                   │
              └────────────────┬────────────────┘                   │
                               │                                     │
                  top_score >= 0.90?                                 │
                  /              \                                   │
           [Yes] /                \ [No]                            │
                /                  \                                 │
  ┌────────────▼──┐      ┌──────────▼────────────┐                  │
  │  (skip grade) │      │      GRADE Node        │                  │
  │               │      │  CrossEncoder re-rank   │                  │
  │               │      │  ms-marco-MiniLM-L-6-v2 │                  │
  │               │      │  filter: score >= 0.5   │                  │
  │               │      └──────────┬─────────────┘                  │
  │               │                 │                                 │
  │               │    filtered >= min_relevant_docs?                 │
  │               │         /             \                           │
  │               │   [Yes]/               \[No, retries left]        │
  │               │       /                 \                         │
  │     ┌─────────▼───────▼─┐        ┌──────▼──────────┐             │
  │     │   GENERATE Node   │        │   REWRITE Node  │─────────────┘
  │     │  Multi-hop prompt │        │  context-aware  │  (max 1 retry)
  │     │  gpt-4o-mini      │        │  query rewrite  │
  │     └─────────┬─────────┘        └─────────────────┘
  │               │
  └───────────────┘
               │
     ┌─────────▼──────────┐
     │   Final Answer     │
     └────────────────────┘
```

### Node Descriptions

| Node | Description |
|---|---|
| **Expand** | Calls `gpt-4o-mini` with a HyDE prompt to generate a hypothetical Wikipedia passage. Stored separately from the query to avoid diluting the embedding signal. |
| **Retrieve** | Two independent retrieval passes (query + HyDE), fused via RRF. Top-20 unique chunks returned. If top document score >= 0.90, `relevant_docs` is populated immediately and grading is skipped. |
| **Grade** | `cross-encoder/ms-marco-MiniLM-L-6-v2` scores each (question, chunk) pair. Chunks scoring < 0.5 are discarded. Falls back to LLM binary grader if CrossEncoder unavailable. |
| **Route** | Conditional edge: if `len(relevant_docs) >= min_relevant_docs` → Generate. Else if retries remain → Rewrite. Else → Generate with best available docs. |
| **Rewrite** | LLM rewrites the query using a summary of what was retrieved, so it can reason about why retrieval failed. HyDE passage is cleared so the next retrieve uses only the rewritten query. |
| **Generate** | Multi-hop synthesis prompt instructs the LLM to explicitly connect evidence across passages before producing an answer. |

### State Schema (`AgenticRAGState`)

```python
class AgenticRAGState(TypedDict):
    question:            str         # Original user question (never overwritten)
    current_query:       str         # May be rewritten by REWRITE node
    hyde_passage:        str         # HyDE passage for dual-pass retrieval
    retrieved_docs:      List[str]   # Raw top-20 chunks from RETRIEVE node
    relevant_docs:       List[str]   # Chunks that passed GRADE node
    answer:              str         # Final generated answer
    steps:               List[str]   # Execution trace (e.g. ["expand","retrieve","grade","generate"])
    retries:             int         # Current rewrite retry count
    top_retrieval_score: float       # Highest dense relevance score from last retrieve
```

---

## 4. Guardrails and Safety Mechanisms

The Agentic RAG pipeline includes several explicit safety mechanisms to prevent hallucination, control cost, and ensure deterministic evaluation.

```
┌────────────────────────────────────────────────────────────────────┐
│                        GUARDRAIL LAYER                             │
│                                                                    │
│  ┌──────────────────────┐    ┌──────────────────────────────────┐  │
│  │  Retrieval Quality   │    │     Infinite Loop Prevention     │  │
│  │  Gate                │    │                                  │  │
│  │  CrossEncoder score  │    │  max_rewrite_retries = 1         │  │
│  │  threshold >= 0.5    │    │  Hard cap regardless of quality  │  │
│  │  Docs below removed  │    │                                  │  │
│  └──────────────────────┘    └──────────────────────────────────┘  │
│                                                                    │
│  ┌──────────────────────┐    ┌──────────────────────────────────┐  │
│  │  Hallucination       │    │     Closed-Book Evaluation       │  │
│  │  Reduction           │    │                                  │  │
│  │  Only graded-relevant│    │  Web search disabled entirely    │  │
│  │  docs reach the LLM  │    │  Agent confined to indexed docs  │  │
│  │  for generation      │    │                                  │  │
│  └──────────────────────┘    └──────────────────────────────────┘  │
│                                                                    │
│  ┌──────────────────────┐    ┌──────────────────────────────────┐  │
│  │  Generation Fallback │    │      Cost Controls               │  │
│  │                      │    │                                  │  │
│  │  If all retries fail  │    │  gpt-4o-mini throughout         │  │
│  │  and no docs pass    │    │  temperature=0.0 (deterministic) │  │
│  │  grading, top-5 raw  │    │  per-call token tracking         │  │
│  │  retrieved docs used │    │  cost logged per agent type      │  │
│  └──────────────────────┘    └──────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

### Guardrail Details

**1. Retrieval Quality Gate**
The CrossEncoder (`ms-marco-MiniLM-L-6-v2`) scores every (question, chunk) pair as raw logits. A threshold of `>= 0.5` filters out chunks that are topically similar but factually irrelevant. This is the primary mechanism preventing the LLM from generating answers grounded in wrong evidence. If no chunks pass, `relevant_docs` is left empty and the rewrite loop activates.

**2. Infinite Loop Prevention**
`max_rewrite_retries = 1` (configurable in `default.yaml`). The routing logic enforces this hard cap — regardless of retrieval quality, after 1 rewrite the agent is forced to generate with whatever context it has. This bounds worst-case latency and API cost.

**3. Hallucination Reduction**
The grading step acts as a filter between retrieval and generation. Only chunks that pass the CrossEncoder threshold are passed to the LLM prompt. Naive RAG passes all retrieved chunks unfiltered — this is the primary architectural difference between the two agents.

**4. Closed-Book Evaluation Enforcement**
Web search (Tavily or similar) is explicitly disabled. The agent can only use the pre-indexed ChromaDB corpus. This ensures evaluation is fair — both agents operate with identical information access.

**5. Generation Fallback**
If retries are exhausted and `relevant_docs` is still empty, the `GENERATE` node uses `retrieved_docs[:5]` as a last resort. This ensures the agent always produces an answer rather than returning empty output, enabling complete metric computation.

**6. Cost Controls**
All LLM calls use `gpt-4o-mini` at `temperature=0.0`. Token counts are tracked per LLM call and accumulated into `AgentResponse.token_usage`. The `CostTracker` logs per-agent total cost to `run_summary.json` after each run.

---

## 5. Evaluation and Failure Analysis Architecture

```
┌──────────────────────┐
│    Agent Outputs     │   (JSONL: question, gold_answer, predicted_answer,
│  naive_rag + agentic │    retrieved_contexts, steps, latency, token_usage)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────────────────────────┐
│            Quantitative Metrics              │
│  Exact Match  │  F1 Score  │  Recall@5  │ MRR│
│  (normalized: lowercase, strip punctuation)  │
└──────────┬───────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────┐
│            LLM-as-a-Judge                    │
│  GPT-4o-mini evaluates predicted vs gold:    │
│  • Correctness     (1–5)                     │
│  • Completeness    (1–5)                     │
│  • Reasoning Quality (1–5)                   │
└──────────┬───────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────┐
│           Failure Mode Classifier            │
└──────────┬───────────────────────────────────┘
           │
           ├─────────────────────┐  EM = 1.0
           │                     ▼
           │              [SUCCESS]
           │
           ├─────────────────────┐  Recall = 0.0  AND  EM < 1.0
           │                     ▼
           │              [RETRIEVAL FAILURE]  — gold not in retrieved chunks
           │
           ├─────────────────────┐  Recall > 0.0  AND  EM < 1.0
           │                     ▼
           │              [REASONING FAILURE]  — context retrieved, answer wrong
           │
           └─────────────────────┐  Agentic RAG retry loops > 3
                                 ▼
                          [LATENCY SPIKE]
```

---

## 6. Retrieval Layer: Hybrid Search

Both agents use the same `VectorStoreWrapper` which implements hybrid dense + sparse retrieval:

```
Query
  │
  ├──▶ ChromaDB cosine similarity (BAAI/bge-small-en-v1.5)  ──▶ top-K dense docs
  │
  └──▶ BM25 keyword search (BM25Retriever)                   ──▶ top-K sparse docs
                                          │
                                          ▼
                          Reciprocal Rank Fusion (RRF, k=60)
                          score(doc) = Σ 1/(60 + rank_i)
                                          │
                                          ▼
                              Merged, re-ranked top-K results
```

Documents appearing in both dense and sparse results receive additive RRF score boosts, improving recall for queries with both semantic and lexical signal.

The Agentic RAG agent runs this hybrid search **twice** (once for the original query, once for the HyDE passage) and applies a second round of RRF across both 15-doc result sets, yielding up to 20 unique candidate chunks before grading.
