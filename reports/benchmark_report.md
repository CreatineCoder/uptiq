# Benchmark Report: Naive RAG vs Agentic RAG

> Generated on 2026-04-12 22:24:08

---

## 1. Executive Summary

- **Winner: Naive RAG** (by average F1 score)
- Naive RAG achieved **0.3938** avg F1 | Agentic RAG achieved **0.2798** avg F1
- Agentic RAG average latency (3.22s) is **2.8x** that of Naive RAG (1.17s)
- Statistical significance: ❌ Not confirmed for F1 score

---

## 2. Experiment Setup

### 2.1 Objective
Compare a simple retrieve-then-generate pipeline (Naive RAG) against a self-correcting agentic pipeline (Agentic RAG) across accuracy, faithfulness, latency, and cost.

### 2.2 Datasets
| Dataset | Type | Queries | Purpose |
|---------|------|---------|---------|
| HotpotQA | Multi-hop | 1,100 | Tests multi-step reasoning and document synthesis |

### 2.3 Retrieval Configuration
- **Embedding Model:** BAAI/bge-small-en-v1.5 (384-dim)
- **Vector Store:** ChromaDB with BM25 hybrid retrieval
- **Re-ranker:** cross-encoder/ms-marco-MiniLM-L-6-v2
- **LLM:** GPT-4o-mini (temperature=0)

---

## 3. Results

### 3.1 Overall Performance

| Metric | Naive RAG | Agentic RAG | Δ (Agentic - Naive) |
|--------|-----------|-------------|---------------------|
| **Exact Match** | 0.2982 | 0.2000 | -0.0982 |
| **F1 Score**    | 0.3938 | 0.2798 | -0.1140 |
| **Recall@5**    | 0.5209 | 0.4800 | -0.0409 |
| **MRR**         | 0.4022 | 0.3606 | -0.0416 |
| **Avg Latency** | 1.17s | 3.22s | +2.05s |

### 3.2 Performance by Dataset

#### NQ (0 queries)
| Metric | Naive RAG | Agentic RAG |
|--------|-----------|-------------|
| F1 | 0.0000 | 0.0000 |
| EM | 0.0000 | 0.0000 |
| Recall@5 | 0.0000 | 0.0000 |

#### HOTPOTQA (1100 queries)
| Metric | Naive RAG | Agentic RAG |
|--------|-----------|-------------|
| F1 | 0.3938 | 0.2798 |
| EM | 0.2982 | 0.2000 |
| Recall@5 | 0.5209 | 0.4800 |

### 3.3 Performance by Difficulty

#### Single-Hop (0 queries)
| Metric | Naive RAG | Agentic RAG |
|--------|-----------|-------------|
| F1 | 0.0000 | 0.0000 |
| EM | 0.0000 | 0.0000 |

#### Multi-Hop (1100 queries)
| Metric | Naive RAG | Agentic RAG |
|--------|-----------|-------------|
| F1 | 0.3938 | 0.2798 |
| EM | 0.2982 | 0.2000 |

---

## 4. Failure Mode Analysis

### 4.1 Naive RAG Failure Modes
| Mode | Count | % |
|------|-------|---|
| Retrieval Failure | 487 | 44.27% |
| Correct | 328 | 29.82% |
| Comprehension Failure | 179 | 16.27% |
| Incomplete Answer | 102 | 9.27% |
| Wrong Reasoning | 4 | 0.36% |

### 4.2 Agentic RAG Failure Modes
| Mode | Count | % |
|------|-------|---|
| Retrieval Failure | 25 | 50.0% |
| Comprehension Failure | 11 | 22.0% |
| Correct | 10 | 20.0% |
| Incomplete Answer | 3 | 6.0% |
| Overcorrection | 1 | 2.0% |

### 4.3 Agentic RAG Self-Correction Effectiveness
| Component | Triggered | Trigger Rate | Success Rate |
|-----------|-----------|-------------|--------------|
| Query Rewrite | 4 | 8.0% | 0.0% |
| Web Search    | 0 | 0.0% | 0.0% |

---

## 5. Statistical Significance

| Metric | p-value | Significant? | Effect Size | Agentic RAG Wins | Naive Wins | Ties |
|--------|---------|-------------|-------------|------------------|------------|------|
| exact_match | 0.705457 | ❌ | 0.0535 | 4 | 3 | 43 |
| f1 | 0.618522 | ❌ | 0.0704 | 6 | 5 | 39 |
| recall_at_5 | 1.000000 | ❌ | 0.0 | 2 | 2 | 46 |
| mrr | 0.055515 | ❌ | 0.2708 | 11 | 5 | 34 |

---

## 6. Cost Analysis

| Agent | Total Cost | Total Tokens | Avg Cost/Query |
|-------|-----------|-------------|----------------|
| Naive RAG   | $0.1923 | 1,270,903 | $0.000175 |
| Agentic RAG | $0.1805 | 975,218 | $0.000164 |

---

## 7. Charts

All 12 charts are saved to `reports/figures/`:
1. Overall Score Comparison
2. Score Distribution (Box Plot)
3. F1 Score Histogram
4. Latency Distribution
5. Cost Analysis
6. Performance by Difficulty
7. Failure Mode Breakdown
8. Radar Chart
9. Correctness vs Latency
10. Agentic RAG Step Distribution
11. Win/Loss/Tie
12. Metric Correlation Heatmap

---

## 8. Reproducibility

- **Config Hash:** `6d768021d8a515e8`
- **Run Command:** `python evaluation/scripts/run_benchmark.py --config configs/default.yaml`
- **Analysis:** `python src/analysis/analyzer.py`
- **Dashboard:** `streamlit run src/visualization/dashboard.py`
