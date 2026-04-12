# Agent Benchmarking: Naive RAG vs Corrective Agentic RAG

A benchmarking framework that evaluates two RAG agent architectures -- standard linear retrieval (Naive RAG) and a self-correcting agentic pipeline (Corrective RAG) -- across 1,000+ queries with multi-dimensional evaluation metrics, failure analysis, and interactive visualizations.

## Overview

| | |
|---|---|
| **Agents** | Naive RAG (baseline) vs Corrective RAG (CRAG) |
| **Dataset** | HotpotQA (multi-hop reasoning) |
| **Evaluation** | Exact Match, F1, Recall@5, MRR + RAGAS + LLM-as-a-Judge |
| **Orchestration** | LangGraph state machine with conditional routing |
| **Visualization** | 12 chart types + Streamlit dashboard |

## Architecture

**Naive RAG** follows a linear retrieve-then-generate pipeline with no self-correction.

**Corrective RAG** is a LangGraph state machine that evaluates its own retrievals and iteratively corrects mistakes:

1. **Query Expansion** -- HyDE generates a hypothetical Wikipedia passage to improve retrieval
2. **Dual-Pass Retrieval** -- Runs two independent retrieval passes (original query + HyDE), merged via Reciprocal Rank Fusion (RRF)
3. **Cross-Encoder Grading** -- `ms-marco-MiniLM-L-6-v2` re-ranks and filters documents (threshold >= 0.5). High-confidence results (>= 0.90) skip grading entirely
4. **Conditional Routing** -- If enough relevant docs pass filtering, generate. Otherwise, rewrite the query with context about why retrieval failed and loop back (up to 2 retries)
5. **Multi-Hop Generation** -- Synthesis prompt that explicitly instructs cross-passage reasoning

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed diagrams and flow descriptions.

## Tech Stack

| Component | Tool |
|---|---|
| LLM | GPT-4o-mini (agents), GPT-4o-mini (judge) |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Vector Store | ChromaDB + BM25 hybrid retrieval |
| Re-ranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Agent Orchestration | LangGraph |
| Evaluation | Custom metrics + RAGAS |
| Visualization | Plotly, Matplotlib, Streamlit |
| Framework | LangChain |

## Project Structure

```
uptiq/
├── src/
│   ├── agents/               # Naive RAG and Corrective RAG implementations
│   ├── retrieval/            # Vector store, chunking, hybrid search
│   ├── evaluation/           # Metrics, RAGAS, LLM judge, cost tracking
│   ├── pipeline/             # Benchmark runner, data loader, config
│   ├── visualization/        # Charts, Streamlit dashboard, report generator
│   └── analysis/             # Failure mode analysis
├── configs/                  # YAML configuration files
├── prompts/                  # Prompt templates for all agent components
├── data/
│   ├── processed/            # Benchmark dataset (JSONL)
│   ├── corpus/               # ChromaDB index
│   └── results/              # Benchmark outputs and checkpoints
├── reports/                  # Generated reports and figures
├── evaluation/scripts/       # CLI entry points
└── notebooks/                # Analysis notebooks
```

## Setup

**Prerequisites:** Python 3.11+, OpenAI API key

```bash
# Clone and install
git clone <repo-url>
cd uptiq
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env: OPENAI_API_KEY=sk-...
```

## Usage

```bash
# Build the vector index
python evaluation/scripts/build_index.py

# Run benchmark (configure query count in configs/default.yaml)
python evaluation/scripts/run_benchmark.py --config configs/default.yaml

# Launch interactive dashboard
streamlit run src/visualization/dashboard.py
```

### Configuration

All settings are in `configs/default.yaml`:
- `dataset.total_queries` -- Number of queries to benchmark
- `agents.corrective_rag.max_rewrite_retries` -- Max query rewrite loops
- `evaluation.ragas.enabled` / `evaluation.llm_judge.enabled` -- Toggle evaluation methods

## Evaluation Framework

**Quantitative Metrics** (all queries, no API cost):
Exact Match, F1 Score, Recall@5, MRR

**RAGAS** (optional, LLM-based):
Faithfulness, Answer Relevancy, Context Precision, Context Recall

**LLM-as-a-Judge** (optional, GPT-4o-mini):
Correctness, Completeness, Reasoning Quality (1-5 scale)

**Failure Mode Analysis** classifies errors into: Retrieval Failure, Comprehension Failure, Incomplete Answer, and Success -- enabling targeted diagnosis of where each agent breaks down.

## License

MIT
