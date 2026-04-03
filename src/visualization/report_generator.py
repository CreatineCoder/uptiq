"""
Phase 6.3 — Report Generator.

Auto-generates reports/benchmark_report.md from the benchmark results.
Embeds chart images and fills in tables with actual numbers.
"""
import os
import sys
import json
import logging
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

logger = logging.getLogger(__name__)


def generate_report(results_dir: str = "data/results", output_path: str = "reports/benchmark_report.md"):
    """Auto-generate the benchmark report from saved results and analysis."""
    
    # Load analysis
    analysis_path = os.path.join(results_dir, "analysis_report.json")
    if not os.path.exists(analysis_path):
        logger.error(f"Analysis report not found at {analysis_path}. Run analysis first.")
        return
    
    with open(analysis_path, "r") as f:
        analysis = json.load(f)
    
    summary_path = os.path.join(results_dir, "run_summary.json")
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)
    
    naive = analysis.get("naive_rag", {})
    crag = analysis.get("corrective_rag", {})
    naive_m = naive.get("avg_metrics", {})
    crag_m = crag.get("avg_metrics", {})
    sig = analysis.get("significance_tests", {})
    
    # Determine winner
    naive_f1 = naive_m.get("avg_f1", 0)
    crag_f1 = crag_m.get("avg_f1", 0)
    winner = "Corrective RAG" if crag_f1 > naive_f1 else "Naive RAG" if naive_f1 > crag_f1 else "Tie"
    
    report = f"""# 📊 Benchmark Report: Naive RAG vs Corrective Agentic RAG

> Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. Executive Summary

- **Winner: {winner}** (by average F1 score)
- Naive RAG achieved **{naive_m.get('avg_f1', 0):.4f}** avg F1 | CRAG achieved **{crag_m.get('avg_f1', 0):.4f}** avg F1
- CRAG's average latency ({crag.get('avg_latency', 0):.2f}s) is **{crag.get('avg_latency', 1) / max(naive.get('avg_latency', 1), 0.01):.1f}x** that of Naive RAG ({naive.get('avg_latency', 0):.2f}s)
- Statistical significance: {"✅ Confirmed (p < 0.05)" if sig.get("f1", {}).get("significant") else "❌ Not confirmed"} for F1 score

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
| **Exact Match** | {naive_m.get('avg_exact_match', 0):.4f} | {crag_m.get('avg_exact_match', 0):.4f} | {crag_m.get('avg_exact_match', 0) - naive_m.get('avg_exact_match', 0):+.4f} |
| **F1 Score** | {naive_m.get('avg_f1', 0):.4f} | {crag_m.get('avg_f1', 0):.4f} | {crag_m.get('avg_f1', 0) - naive_m.get('avg_f1', 0):+.4f} |
| **Recall@5** | {naive_m.get('avg_recall_at_5', 0):.4f} | {crag_m.get('avg_recall_at_5', 0):.4f} | {crag_m.get('avg_recall_at_5', 0) - naive_m.get('avg_recall_at_5', 0):+.4f} |
| **MRR** | {naive_m.get('avg_mrr', 0):.4f} | {crag_m.get('avg_mrr', 0):.4f} | {crag_m.get('avg_mrr', 0) - naive_m.get('avg_mrr', 0):+.4f} |
| **Avg Latency** | {naive.get('avg_latency', 0):.2f}s | {crag.get('avg_latency', 0):.2f}s | {crag.get('avg_latency', 0) - naive.get('avg_latency', 0):+.2f}s |

### 3.2 Performance by Dataset
"""
    
    perf_ds = analysis.get("performance_by_dataset", {})
    for ds_name, ds_data in perf_ds.items():
        n = ds_data.get("naive_rag", {})
        c = ds_data.get("corrective_rag", {})
        report += f"""
#### {ds_name.upper()} ({ds_data.get('naive_count', 0)} queries)
| Metric | Naive RAG | CRAG |
|--------|-----------|------|
| F1 | {n.get('avg_f1', 0):.4f} | {c.get('avg_f1', 0):.4f} |
| EM | {n.get('avg_exact_match', 0):.4f} | {c.get('avg_exact_match', 0):.4f} |
| Recall@5 | {n.get('avg_recall_at_5', 0):.4f} | {c.get('avg_recall_at_5', 0):.4f} |
"""

    report += """
### 3.3 Performance by Difficulty
"""
    perf_diff = analysis.get("performance_by_difficulty", {})
    for diff_name, diff_data in perf_diff.items():
        n = diff_data.get("naive_rag", {})
        c = diff_data.get("corrective_rag", {})
        report += f"""
#### {diff_name.title()} ({diff_data.get('naive_count', 0)} queries)
| Metric | Naive RAG | CRAG |
|--------|-----------|------|
| F1 | {n.get('avg_f1', 0):.4f} | {c.get('avg_f1', 0):.4f} |
| EM | {n.get('avg_exact_match', 0):.4f} | {c.get('avg_exact_match', 0):.4f} |
"""

    # Failure modes
    report += """
---

## 4. Failure Mode Analysis

### 4.1 Naive RAG Failure Modes
| Mode | Count | % |
|------|-------|---|
"""
    for mode, data in naive.get("failure_modes", {}).items():
        report += f"| {mode.replace('_', ' ').title()} | {data['count']} | {data['percentage']}% |\n"
    
    report += """
### 4.2 CRAG Failure Modes
| Mode | Count | % |
|------|-------|---|
"""
    for mode, data in crag.get("failure_modes", {}).items():
        report += f"| {mode.replace('_', ' ').title()} | {data['count']} | {data['percentage']}% |\n"

    # CRAG component analysis
    comp = crag.get("component_analysis", {})
    if comp:
        rw = comp.get("rewrites", {})
        ws = comp.get("web_search", {})
        report += f"""
### 4.3 CRAG Self-Correction Effectiveness
| Component | Triggered | Trigger Rate | Success Rate |
|-----------|-----------|-------------|--------------|
| Query Rewrite | {rw.get('triggered', 0)} | {rw.get('trigger_rate', 0)}% | {rw.get('success_rate', 0)}% |
| Web Search | {ws.get('triggered', 0)} | {ws.get('trigger_rate', 0)}% | {ws.get('success_rate', 0)}% |
"""

    # Statistical significance
    report += """
---

## 5. Statistical Significance

| Metric | p-value | Significant? | Effect Size | CRAG Wins | Naive Wins | Ties |
|--------|---------|-------------|-------------|-----------|------------|------|
"""
    for metric, result in sig.items():
        sig_icon = "✅" if result.get("significant") else "❌"
        report += f"| {metric} | {result.get('p_value', 'N/A'):.6f} | {sig_icon} | {result.get('effect_size', 'N/A')} | {result.get('crag_wins', 'N/A')} | {result.get('naive_wins', 'N/A')} | {result.get('ties', 'N/A')} |\n"

    # Cost
    naive_cost = summary.get("naive_rag_cost", {})
    crag_cost = summary.get("crag_cost", {})
    report += f"""
---

## 6. Cost Analysis

| Agent | Total Cost | Total Tokens | Avg Cost/Query |
|-------|-----------|-------------|----------------|
| Naive RAG | ${naive_cost.get('total_cost_usd', 0):.4f} | {naive_cost.get('total_tokens', 0):,} | ${naive_cost.get('avg_cost_per_call', 0):.6f} |
| CRAG | ${crag_cost.get('total_cost_usd', 0):.4f} | {crag_cost.get('total_tokens', 0):,} | ${crag_cost.get('avg_cost_per_call', 0):.6f} |

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

- **Config Hash:** `{summary.get('config_hash', 'N/A')}`
- **Run Command:** `python evaluation/scripts/run_benchmark.py --config configs/default.yaml`
- **Analysis:** `python src/analysis/analyzer.py`
- **Dashboard:** `streamlit run src/visualization/dashboard.py`
"""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    logger.info(f"[Report] Benchmark report generated at {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
    generate_report()
