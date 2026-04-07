# 📊 Benchmark Report: Naive RAG vs Corrective Agentic RAG

> Generated on 2026-04-04 22:48:09

---

## 1. Executive Summary

- **Winner: Corrective RAG** (by average F1 score)
- Naive RAG achieved **0.0000** avg F1 | CRAG achieved **0.1143** avg F1
- CRAG's average latency (20.64s) is **13.2x** that of Naive RAG (1.57s)
- Statistical significance: ❌ Not confirmed for F1 score

---

## 2. Experiment Setup

### 2.1 Objective
Compare a simple retrieve-then-generate pipeline (Naive RAG) against a self-correcting agentic pipeline (Corrective RAG) across accuracy, faithfulness, latency, and cost.

### 2.2 Datasets
| Dataset | Type | Queries | Purpose |
|---------|------|---------|---------|
| Natural Questions (NQ) | Single-hop | 1,000 | Tests query ambiguity and retrieval precision |
| HotpotQA | Multi-hop | 500 | Tests multi-step reasoning and document synthesis |

### 2.3 Retrieval Configuration
- **Embedding Model:** all-MiniLM-L6-v2 (384-dim)
- **Vector Store:** ChromaDB (shared corpus of ~2,000 chunks)
- **Top-K:** 5 documents per query
- **LLM:** GPT-4o-mini (temperature=0)

---

## 3. Results

### 3.1 Overall Performance

| Metric | Naive RAG | CRAG | Δ (CRAG - Naive) |
|--------|-----------|------|-------------------|
| **Exact Match** | 0.0000 | 0.1000 | +0.1000 |
| **F1 Score** | 0.0000 | 0.1143 | +0.1143 |
| **Recall@5** | 0.3000 | 0.5000 | +0.2000 |
| **MRR** | 0.3000 | 0.4333 | +0.1333 |
| **Avg Latency** | 1.57s | 20.64s | +19.07s |

### 3.2 Performance by Dataset

#### NQ (0 queries)
| Metric | Naive RAG | CRAG |
|--------|-----------|------|
| F1 | 0.0000 | 0.0000 |
| EM | 0.0000 | 0.0000 |
| Recall@5 | 0.0000 | 0.0000 |

#### HOTPOTQA (10 queries)
| Metric | Naive RAG | CRAG |
|--------|-----------|------|
| F1 | 0.0000 | 0.1143 |
| EM | 0.0000 | 0.1000 |
| Recall@5 | 0.3000 | 0.5000 |

### 3.3 Performance by Difficulty

#### Single-Hop (0 queries)
| Metric | Naive RAG | CRAG |
|--------|-----------|------|
| F1 | 0.0000 | 0.0000 |
| EM | 0.0000 | 0.0000 |

#### Multi-Hop (10 queries)
| Metric | Naive RAG | CRAG |
|--------|-----------|------|
| F1 | 0.0000 | 0.1143 |
| EM | 0.0000 | 0.1000 |

---

## 4. Failure Mode Analysis

### 4.1 Naive RAG Failure Modes
| Mode | Count | % |
|------|-------|---|
| Retrieval Failure | 7 | 70.0% |
| Comprehension Failure | 3 | 30.0% |

### 4.2 CRAG Failure Modes
| Mode | Count | % |
|------|-------|---|
| Retrieval Failure | 5 | 50.0% |
| Overcorrection | 4 | 40.0% |
| Correct | 1 | 10.0% |

### 4.3 CRAG Self-Correction Effectiveness
| Component | Triggered | Trigger Rate | Success Rate |
|-----------|-----------|-------------|--------------|
| Query Rewrite | 10 | 100.0% | 10.0% |
| Web Search | 10 | 100.0% | 10.0% |

---

## 5. Statistical Significance

| Metric | p-value | Significant? | Effect Size | CRAG Wins | Naive Wins | Ties |
|--------|---------|-------------|-------------|-----------|------------|------|
| exact_match | 1.000000 | ❌ | 0.0 | 1 | 0 | 9 |
| f1 | 0.500000 | ❌ | 0.2133 | 2 | 0 | 8 |
| recall_at_5 | 0.500000 | ❌ | 0.2133 | 2 | 0 | 8 |
| mrr | 0.500000 | ❌ | 0.2133 | 2 | 1 | 7 |

---

## 6. Cost Analysis

| Agent | Total Cost | Total Tokens | Avg Cost/Query |
|-------|-----------|-------------|----------------|
| Naive RAG | $0.0027 | 17,981 | $0.000272 |
| CRAG | $0.0152 | 99,100 | $0.001518 |

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
10. CRAG Step Distribution
11. Win/Loss/Tie
12. Metric Correlation Heatmap

---

## 8. Reproducibility

- **Config Hash:** `63193c54c0c4da22`
- **Run Command:** `python evaluation/scripts/run_benchmark.py --config configs/default.yaml`
- **Analysis:** `python src/analysis/analyzer.py`
- **Dashboard:** `streamlit run src/visualization/dashboard.py`
