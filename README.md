# Agent Benchmarking: Naive RAG vs Agentic RAG

A benchmarking framework that evaluates two RAG architectures — a standard linear pipeline (Naive RAG) and a self-correcting agentic pipeline (Agentic RAG) — across 1,100 HotpotQA multi-hop queries. Evaluation combines quantitative metrics (Exact Match, F1, Recall@5, MRR) with LLM-as-a-Judge scoring for correctness, completeness, and reasoning quality.

---

## Overview

| | |
|---|---|
| **Agents** | Naive RAG (baseline) vs Agentic RAG |
| **Dataset** | HotpotQA — 1,100 multi-hop reasoning queries |
| **Evaluation** | Exact Match, F1, Recall@5, MRR + LLM-as-a-Judge |
| **Orchestration** | LangGraph state machine with conditional routing |
| **Visualization** | 12+ chart types + interactive Streamlit dashboard |

---

## Architecture

**Naive RAG** follows a strictly linear retrieve-then-generate pipeline with no self-correction.

**Agentic RAG** is a LangGraph state machine with five nodes:

1. **Expand** — HyDE generates a hypothetical Wikipedia-style passage to improve retrieval embedding quality
2. **Retrieve** — Dual-pass retrieval (original query + HyDE passage), merged via Reciprocal Rank Fusion (RRF). High-confidence results (score >= 0.90) skip grading entirely
3. **Grade** — `cross-encoder/ms-marco-MiniLM-L-6-v2` re-ranks and filters documents (threshold >= 0.5)
4. **Route** — If enough relevant docs pass filtering, generate. Otherwise, rewrite the query and retry (up to 1 retry)
5. **Generate** — Multi-hop synthesis prompt that explicitly instructs cross-passage reasoning

See [ARCHITECTURE.md](ARCHITECTURE.md) for full diagrams and flow descriptions.

---

## Dataset

| Property | Details |
|---|---|
| **Source** | [HotpotQA](https://hotpotqa.github.io/) — `fullwiki` configuration, validation split |
| **Size** | 1,100 queries (subset of 7,405 available validation examples) |
| **Task Type** | Multi-hop reasoning — each question requires synthesizing information from 2+ Wikipedia passages |
| **Answer Format** | Short extractive answers (single entity, date, or yes/no) |
| **Gold Context** | Each item includes the supporting Wikipedia passages and their supporting fact annotations |

### Preprocessing

1. The `fullwiki` split is loaded via HuggingFace `datasets`
2. For each item, supporting passages are reconstructed from the `context` dict (`title` + `sentences` lists) into a single `gold_context` string
3. Each record is saved as a JSONL entry with fields: `id`, `question`, `gold_answer`, `gold_context`, `dataset`, `difficulty`, `supporting_facts`
4. All 1,100 records are written to `data/processed/benchmark_dataset.jsonl`
5. Gold contexts are chunked (600 tokens, 150 overlap) and indexed into ChromaDB using `BAAI/bge-small-en-v1.5` embeddings

---

## Tech Stack

| Component | Tool |
|---|---|
| LLM | GPT-4o-mini |
| Embeddings | BAAI/bge-small-en-v1.5 (HuggingFace) |
| Vector Store | ChromaDB + BM25 hybrid retrieval (RRF fusion) |
| Re-ranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Agent Orchestration | LangGraph |
| Evaluation | Custom metrics + LLM-as-a-Judge (GPT-4o-mini) |
| Visualization | Plotly, Streamlit |
| Framework | LangChain |

---

## Project Structure

```
uptiq/
├── src/
│   ├── agents/
│   │   ├── base_agent.py             # Abstract BaseAgent + AgentResponse dataclass
│   │   ├── agentic_rag_agent.py      # Agentic RAG (LangGraph state machine)
│   │   └── naive_rag_agent.py        # Naive RAG (linear retrieve → generate)
│   ├── retrieval/
│   │   ├── vector_store.py           # ChromaDB + BM25 hybrid search wrapper
│   │   └── chunking.py               # Text chunker (600 token chunks, 150 overlap)
│   ├── evaluation/
│   │   ├── metrics.py                # EM, F1, Recall@K, MRR
│   │   ├── llm_judge.py              # LLM-as-a-Judge (correctness/completeness/reasoning)
│   │   └── cost_tracker.py           # Per-agent API cost tracking
│   ├── pipeline/
│   │   ├── benchmark_runner.py       # Orchestrates full benchmark run
│   │   ├── data_loader.py            # HotpotQA download + preprocessing
│   │   └── result_collector.py       # JSONL result writing + checkpointing
│   ├── visualization/
│   │   ├── dashboard.py              # Streamlit interactive dashboard
│   │   ├── charts.py                 # Plotly chart generators (12+ chart types)
│   │   └── report_generator.py       # Markdown/HTML report generator
│   └── analysis/
│       └── analyzer.py               # Failure mode classifier + analysis report
├── configs/
│   └── default.yaml                  # All benchmark settings
├── prompts/
│   ├── agentic_rag_generator.txt     # Multi-hop synthesis prompt
│   ├── agentic_rag_grader.txt        # LLM fallback relevance grader prompt
│   ├── agentic_rag_rewriter.txt      # Context-aware query rewrite prompt
│   └── query_expansion.txt           # HyDE passage generation prompt
├── data/
│   ├── processed/                    # benchmark_dataset.jsonl (1,100 queries)
│   ├── corpus/                       # ChromaDB index (chroma_db_bge_small)
│   └── results/                      # Benchmark outputs, checkpoints, analysis
├── reports/                          # Generated chart images + HTML reports
├── evaluation/scripts/
│   ├── build_index.py                # Builds ChromaDB index from dataset
│   └── run_benchmark.py              # CLI entry point for benchmark runs
└── requirements.txt
```

---

## Setup

### Prerequisites

- Python 3.11+
- OpenAI API key

### Step 1 — Clone and create virtual environment

```bash
git clone <repo-url>
cd uptiq
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Configure environment

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=sk-...
```

### Step 4 — Build the dataset

Downloads 1,100 HotpotQA queries from HuggingFace and writes them to `data/processed/benchmark_dataset.jsonl`:

```bash
python src/pipeline/data_loader.py
```

### Step 5 — Build the vector index

Chunks all gold contexts and indexes them into ChromaDB. Automatically clears any existing index before rebuilding:

```bash
python evaluation/scripts/build_index.py
```

### Step 6 — Run the benchmark

```bash
python evaluation/scripts/run_benchmark.py --config configs/default.yaml
```

For a quick pilot run (e.g. 20 queries):
```bash
python evaluation/scripts/run_benchmark.py --config configs/default.yaml --pilot 20
```

### Step 7 — Launch the dashboard

```bash
streamlit run src/visualization/dashboard.py
```

Open `http://localhost:8501` in your browser.

---

## Configuration

All settings live in `configs/default.yaml`:

```yaml
dataset:
  total_queries: 1100         # Number of queries to run

agents:
  agentic_rag:
    max_rewrite_retries: 1    # Max query rewrite loops before forced generation
    min_relevant_docs: 1      # Minimum docs needed to skip rewrite

evaluation:
  llm_judge:
    enabled: true             # Toggle LLM-as-a-Judge scoring
    sample_size: 1100         # Number of queries to judge
```

---

## Evaluation Framework

### A. Quantitative Metrics
Run on all queries automatically, no additional API cost:

| Metric | Description |
|---|---|
| **Exact Match (EM)** | Binary — 1 if predicted answer matches gold after normalization |
| **F1 Score** | Token-level overlap between predicted and gold answer |
| **Recall@5** | Whether the gold answer text appears in the top-5 retrieved chunks |
| **MRR** | Mean Reciprocal Rank of the first chunk containing the gold answer |

### B. LLM-as-a-Judge
GPT-4o-mini evaluates each answer on a 1–5 scale across three criteria:

| Criterion | Description |
|---|---|
| **Correctness** | Is the answer factually accurate compared to the gold answer? |
| **Completeness** | Does the answer cover all parts of the question? |
| **Reasoning Quality** | For multi-hop questions, does the answer connect evidence correctly? |

### Failure Mode Analysis
Each result is automatically classified into one of four categories:

| Category | Condition |
|---|---|
| **Success** | EM = 1.0 |
| **Retrieval Failure** | Recall = 0.0 and EM < 1.0 — gold answer not in retrieved context |
| **Reasoning Failure** | Recall > 0.0 and EM < 1.0 — context was retrieved but answer was wrong |
| **Latency Spike** | Agentic RAG used more than 3 retry loops |

---

## Guardrails

See [ARCHITECTURE.md — Section 4: Guardrails](ARCHITECTURE.md#4-guardrails-and-safety-mechanisms) for full details.

| Guardrail | Mechanism |
|---|---|
| **Hallucination reduction** | Cross-encoder filters irrelevant context before generation |
| **Retrieval quality gate** | Docs below score 0.5 are excluded; rewrite loop activates |
| **Infinite loop prevention** | Hard cap of 1 rewrite retry (`max_rewrite_retries`) |
| **Generation fallback** | If all retries fail, top retrieved docs used as last-resort context |
| **Cost controls** | gpt-4o-mini used throughout; per-call cost tracked and logged |

---

## License

MIT
