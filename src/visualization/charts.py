"""
Phase 6.1 — Static Chart Generation (12 Publication-Quality Charts).

Uses Plotly for interactive HTML charts and Matplotlib/Seaborn for static PNGs.
All charts are saved to reports/figures/.
"""
import os
import sys
import json
import logging
import numpy as np
from typing import Dict, List

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

# Color palette
NAIVE_COLOR = "#6366f1"   # Indigo
CRAG_COLOR = "#f97316"    # Orange
BG_COLOR = "#0f172a"      # Dark navy
GRID_COLOR = "#1e293b"    # Slate


def _load_jsonl(path: str) -> List[Dict]:
    results = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    results.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
    return results


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _dark_layout(fig, title: str):
    """Apply consistent dark theme to Plotly figures."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color="white")),
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
        font=dict(color="white", family="Inter"),
        legend=dict(bgcolor="rgba(0,0,0,0.3)", font=dict(color="white")),
        xaxis=dict(gridcolor=GRID_COLOR, color="white"),
        yaxis=dict(gridcolor=GRID_COLOR, color="white")
    )
    return fig


# ─────────────────────────────────────────────────────────────
# Chart 1: Overall Score Comparison (Grouped Bar)
# ─────────────────────────────────────────────────────────────
def chart_overall_comparison(analysis: Dict, output_dir: str):
    naive = analysis["naive_rag"]["avg_metrics"]
    crag = analysis["corrective_rag"]["avg_metrics"]
    
    metrics = ["avg_exact_match", "avg_f1", "avg_recall_at_5", "avg_mrr"]
    labels = ["Exact Match", "F1 Score", "Recall@5", "MRR"]
    
    fig = go.Figure(data=[
        go.Bar(name="Naive RAG", x=labels, y=[naive.get(m, 0) for m in metrics], marker_color=NAIVE_COLOR),
        go.Bar(name="CRAG", x=labels, y=[crag.get(m, 0) for m in metrics], marker_color=CRAG_COLOR)
    ])
    fig.update_layout(barmode='group')
    _dark_layout(fig, "Overall Score Comparison: Naive RAG vs CRAG")
    fig.write_html(os.path.join(output_dir, "01_overall_comparison.html"))
    fig.write_image(os.path.join(output_dir, "01_overall_comparison.png"), width=1200, height=600)
    logger.info("[Charts] Chart 1: Overall comparison saved.")


# ─────────────────────────────────────────────────────────────
# Chart 2: Score Distribution (Box Plot)
# ─────────────────────────────────────────────────────────────
def chart_score_distribution(naive_results: List, crag_results: List, output_dir: str):
    data = []
    for r in naive_results:
        data.append({"Agent": "Naive RAG", "F1": r["metrics"].get("f1", 0), "EM": r["metrics"].get("exact_match", 0)})
    for r in crag_results:
        data.append({"Agent": "CRAG", "F1": r["metrics"].get("f1", 0), "EM": r["metrics"].get("exact_match", 0)})
    
    fig = go.Figure()
    for agent, color in [("Naive RAG", NAIVE_COLOR), ("CRAG", CRAG_COLOR)]:
        vals = [d["F1"] for d in data if d["Agent"] == agent]
        fig.add_trace(go.Box(y=vals, name=agent, marker_color=color, boxmean=True))
    
    _dark_layout(fig, "F1 Score Distribution")
    fig.write_html(os.path.join(output_dir, "02_score_distribution.html"))
    fig.write_image(os.path.join(output_dir, "02_score_distribution.png"), width=1200, height=600)
    logger.info("[Charts] Chart 2: Score distribution saved.")


# ─────────────────────────────────────────────────────────────
# Chart 3: F1 Score Histogram
# ─────────────────────────────────────────────────────────────
def chart_f1_histogram(naive_results: List, crag_results: List, output_dir: str):
    naive_f1 = [r["metrics"].get("f1", 0) for r in naive_results]
    crag_f1 = [r["metrics"].get("f1", 0) for r in crag_results]
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=naive_f1, name="Naive RAG", marker_color=NAIVE_COLOR, opacity=0.7, nbinsx=20))
    fig.add_trace(go.Histogram(x=crag_f1, name="CRAG", marker_color=CRAG_COLOR, opacity=0.7, nbinsx=20))
    fig.update_layout(barmode='overlay')
    _dark_layout(fig, "F1 Score Distribution (Histogram)")
    fig.write_html(os.path.join(output_dir, "03_f1_histogram.html"))
    fig.write_image(os.path.join(output_dir, "03_f1_histogram.png"), width=1200, height=600)
    logger.info("[Charts] Chart 3: F1 histogram saved.")


# ─────────────────────────────────────────────────────────────
# Chart 4: Latency Distribution (Box Plot)
# ─────────────────────────────────────────────────────────────
def chart_latency_distribution(naive_results: List, crag_results: List, output_dir: str):
    fig = go.Figure()
    fig.add_trace(go.Box(y=[r["latency"] for r in naive_results], name="Naive RAG", marker_color=NAIVE_COLOR, boxmean=True))
    fig.add_trace(go.Box(y=[r["latency"] for r in crag_results], name="CRAG", marker_color=CRAG_COLOR, boxmean=True))
    _dark_layout(fig, "Latency Distribution (seconds)")
    fig.update_yaxes(title_text="Latency (s)")
    fig.write_html(os.path.join(output_dir, "04_latency_distribution.html"))
    fig.write_image(os.path.join(output_dir, "04_latency_distribution.png"), width=1200, height=600)
    logger.info("[Charts] Chart 4: Latency distribution saved.")


# ─────────────────────────────────────────────────────────────
# Chart 5: Cost Analysis (Stacked Bar)
# ─────────────────────────────────────────────────────────────
def chart_cost_analysis(analysis: Dict, summary: Dict, output_dir: str):
    naive_cost = summary.get("naive_rag_cost", {})
    crag_cost = summary.get("crag_cost", {})
    
    fig = go.Figure(data=[
        go.Bar(name="Input Tokens Cost",
               x=["Naive RAG", "CRAG"],
               y=[naive_cost.get("total_prompt_tokens", 0) / 1e6 * 0.15,
                  crag_cost.get("total_prompt_tokens", 0) / 1e6 * 0.15],
               marker_color=NAIVE_COLOR),
        go.Bar(name="Output Tokens Cost",
               x=["Naive RAG", "CRAG"],
               y=[naive_cost.get("total_completion_tokens", 0) / 1e6 * 0.60,
                  crag_cost.get("total_completion_tokens", 0) / 1e6 * 0.60],
               marker_color=CRAG_COLOR)
    ])
    fig.update_layout(barmode='stack')
    _dark_layout(fig, "API Cost Breakdown ($USD)")
    fig.update_yaxes(title_text="Cost ($)")
    fig.write_html(os.path.join(output_dir, "05_cost_analysis.html"))
    fig.write_image(os.path.join(output_dir, "05_cost_analysis.png"), width=1200, height=600)
    logger.info("[Charts] Chart 5: Cost analysis saved.")


# ─────────────────────────────────────────────────────────────
# Chart 6: Performance by Difficulty (Grouped Bar)
# ─────────────────────────────────────────────────────────────
def chart_performance_by_difficulty(analysis: Dict, output_dir: str):
    perf = analysis.get("performance_by_difficulty", {})
    
    difficulties = list(perf.keys())
    naive_f1 = [perf[d]["naive_rag"].get("avg_f1", 0) for d in difficulties]
    crag_f1 = [perf[d]["corrective_rag"].get("avg_f1", 0) for d in difficulties]
    
    fig = go.Figure(data=[
        go.Bar(name="Naive RAG", x=difficulties, y=naive_f1, marker_color=NAIVE_COLOR),
        go.Bar(name="CRAG", x=difficulties, y=crag_f1, marker_color=CRAG_COLOR)
    ])
    fig.update_layout(barmode='group')
    _dark_layout(fig, "F1 Score by Difficulty Level")
    fig.write_html(os.path.join(output_dir, "06_perf_by_difficulty.html"))
    fig.write_image(os.path.join(output_dir, "06_perf_by_difficulty.png"), width=1200, height=600)
    logger.info("[Charts] Chart 6: Performance by difficulty saved.")


# ─────────────────────────────────────────────────────────────
# Chart 7: Failure Mode Breakdown (Stacked Bar)
# ─────────────────────────────────────────────────────────────
def chart_failure_modes(analysis: Dict, output_dir: str):
    naive_fm = analysis["naive_rag"]["failure_modes"]
    crag_fm = analysis["corrective_rag"]["failure_modes"]
    
    all_modes = sorted(set(list(naive_fm.keys()) + list(crag_fm.keys())))
    colors = px.colors.qualitative.Set2
    
    fig = go.Figure()
    for i, mode in enumerate(all_modes):
        fig.add_trace(go.Bar(
            name=mode.replace("_", " ").title(),
            x=["Naive RAG", "CRAG"],
            y=[naive_fm.get(mode, {}).get("count", 0), crag_fm.get(mode, {}).get("count", 0)],
            marker_color=colors[i % len(colors)]
        ))
    
    fig.update_layout(barmode='stack')
    _dark_layout(fig, "Failure Mode Breakdown")
    fig.write_html(os.path.join(output_dir, "07_failure_modes.html"))
    fig.write_image(os.path.join(output_dir, "07_failure_modes.png"), width=1200, height=600)
    logger.info("[Charts] Chart 7: Failure modes saved.")


# ─────────────────────────────────────────────────────────────
# Chart 8: Radar / Spider Chart
# ─────────────────────────────────────────────────────────────
def chart_radar(analysis: Dict, output_dir: str):
    naive = analysis["naive_rag"]["avg_metrics"]
    crag = analysis["corrective_rag"]["avg_metrics"]
    
    categories = ["Exact Match", "F1 Score", "Recall@5", "MRR"]
    keys = ["avg_exact_match", "avg_f1", "avg_recall_at_5", "avg_mrr"]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[naive.get(k, 0) for k in keys] + [naive.get(keys[0], 0)],
                                   theta=categories + [categories[0]], name="Naive RAG",
                                   line=dict(color=NAIVE_COLOR, width=3), fill='toself', fillcolor='rgba(99,102,241,0.2)'))
    fig.add_trace(go.Scatterpolar(r=[crag.get(k, 0) for k in keys] + [crag.get(keys[0], 0)],
                                   theta=categories + [categories[0]], name="CRAG",
                                   line=dict(color=CRAG_COLOR, width=3), fill='toself', fillcolor='rgba(249,115,22,0.2)'))
    
    fig.update_layout(polar=dict(bgcolor=BG_COLOR, radialaxis=dict(gridcolor=GRID_COLOR, color="white"),
                                  angularaxis=dict(gridcolor=GRID_COLOR, color="white")))
    _dark_layout(fig, "Multi-Dimensional Performance Radar")
    fig.write_html(os.path.join(output_dir, "08_radar_chart.html"))
    fig.write_image(os.path.join(output_dir, "08_radar_chart.png"), width=800, height=800)
    logger.info("[Charts] Chart 8: Radar chart saved.")


# ─────────────────────────────────────────────────────────────
# Chart 9: Correctness vs Latency (Scatter)
# ─────────────────────────────────────────────────────────────
def chart_correctness_vs_latency(naive_results: List, crag_results: List, output_dir: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[r["latency"] for r in naive_results],
                              y=[r["metrics"].get("f1", 0) for r in naive_results],
                              mode='markers', name="Naive RAG",
                              marker=dict(color=NAIVE_COLOR, size=5, opacity=0.6)))
    fig.add_trace(go.Scatter(x=[r["latency"] for r in crag_results],
                              y=[r["metrics"].get("f1", 0) for r in crag_results],
                              mode='markers', name="CRAG",
                              marker=dict(color=CRAG_COLOR, size=5, opacity=0.6)))
    _dark_layout(fig, "Quality vs Speed Tradeoff")
    fig.update_xaxes(title_text="Latency (seconds)")
    fig.update_yaxes(title_text="F1 Score")
    fig.write_html(os.path.join(output_dir, "09_correctness_vs_latency.html"))
    fig.write_image(os.path.join(output_dir, "09_correctness_vs_latency.png"), width=1200, height=600)
    logger.info("[Charts] Chart 9: Correctness vs latency saved.")


# ─────────────────────────────────────────────────────────────
# Chart 10: CRAG Step Distribution (Histogram)
# ─────────────────────────────────────────────────────────────
def chart_crag_steps(crag_results: List, output_dir: str):
    step_counts = [len(r.get("steps", [])) for r in crag_results]
    
    fig = go.Figure(data=[go.Histogram(x=step_counts, marker_color=CRAG_COLOR, nbinsx=15)])
    _dark_layout(fig, "CRAG: Steps Taken Per Query")
    fig.update_xaxes(title_text="Number of Steps")
    fig.update_yaxes(title_text="Count")
    fig.write_html(os.path.join(output_dir, "10_crag_steps.html"))
    fig.write_image(os.path.join(output_dir, "10_crag_steps.png"), width=1200, height=600)
    logger.info("[Charts] Chart 10: CRAG step distribution saved.")


# ─────────────────────────────────────────────────────────────
# Chart 11: Win/Loss/Tie (Donut)
# ─────────────────────────────────────────────────────────────
def chart_win_loss_tie(naive_results: List, crag_results: List, output_dir: str):
    naive_by_id = {r["query_id"]: r for r in naive_results}
    crag_by_id = {r["query_id"]: r for r in crag_results}
    common = set(naive_by_id.keys()) & set(crag_by_id.keys())
    
    wins, losses, ties = 0, 0, 0
    for qid in common:
        nf1 = naive_by_id[qid]["metrics"].get("f1", 0)
        cf1 = crag_by_id[qid]["metrics"].get("f1", 0)
        if cf1 > nf1:
            wins += 1
        elif nf1 > cf1:
            losses += 1
        else:
            ties += 1
    
    fig = go.Figure(data=[go.Pie(
        labels=["CRAG Wins", "Naive RAG Wins", "Tie"],
        values=[wins, losses, ties],
        hole=0.5,
        marker=dict(colors=[CRAG_COLOR, NAIVE_COLOR, "#64748b"]),
        textinfo="label+percent"
    )])
    _dark_layout(fig, "Per-Query Win/Loss/Tie (F1)")
    fig.write_html(os.path.join(output_dir, "11_win_loss_tie.html"))
    fig.write_image(os.path.join(output_dir, "11_win_loss_tie.png"), width=800, height=800)
    logger.info("[Charts] Chart 11: Win/Loss/Tie saved.")


# ─────────────────────────────────────────────────────────────
# Chart 12: Metric Correlation Heatmap (Matplotlib)
# ─────────────────────────────────────────────────────────────
def chart_metric_correlation(naive_results: List, output_dir: str):
    import pandas as pd
    
    rows = []
    for r in naive_results:
        m = r.get("metrics", {})
        rows.append({
            "EM": m.get("exact_match", 0),
            "F1": m.get("f1", 0),
            "Recall@5": m.get("recall_at_5", 0),
            "MRR": m.get("mrr", 0),
            "Latency": r.get("latency", 0)
        })
    
    df = pd.DataFrame(rows)
    corr = df.corr()
    
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="dark")
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, square=True,
                linewidths=1, fmt=".2f", cbar_kws={"shrink": 0.8})
    plt.title("Metric Correlation Heatmap", fontsize=16, color="white", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "12_metric_correlation.png"), dpi=150, facecolor="#0f172a")
    plt.close()
    logger.info("[Charts] Chart 12: Metric correlation heatmap saved.")


# ─────────────────────────────────────────────────────────────
# Generate ALL Charts
# ─────────────────────────────────────────────────────────────
def generate_all_charts(results_dir: str, output_dir: str = "reports/figures"):
    """Generate all 12 charts from the benchmark results."""
    _ensure_dir(output_dir)
    
    # Load data
    naive_results = _load_jsonl(os.path.join(results_dir, "naive_rag_results.jsonl"))
    crag_results = _load_jsonl(os.path.join(results_dir, "crag_results.jsonl"))
    
    analysis_path = os.path.join(results_dir, "analysis_report.json")
    with open(analysis_path, "r") as f:
        analysis = json.load(f)
    
    summary_path = os.path.join(results_dir, "run_summary.json")
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)
    
    logger.info(f"[Charts] Generating 12 charts from {len(naive_results)} + {len(crag_results)} results...")
    
    try:
        chart_overall_comparison(analysis, output_dir)
    except Exception as e:
        logger.warning(f"[Charts] Chart 1 failed: {e}")
    
    try:
        chart_score_distribution(naive_results, crag_results, output_dir)
    except Exception as e:
        logger.warning(f"[Charts] Chart 2 failed: {e}")
    
    try:
        chart_f1_histogram(naive_results, crag_results, output_dir)
    except Exception as e:
        logger.warning(f"[Charts] Chart 3 failed: {e}")
    
    try:
        chart_latency_distribution(naive_results, crag_results, output_dir)
    except Exception as e:
        logger.warning(f"[Charts] Chart 4 failed: {e}")
    
    try:
        chart_cost_analysis(analysis, summary, output_dir)
    except Exception as e:
        logger.warning(f"[Charts] Chart 5 failed: {e}")
    
    try:
        chart_performance_by_difficulty(analysis, output_dir)
    except Exception as e:
        logger.warning(f"[Charts] Chart 6 failed: {e}")
    
    try:
        chart_failure_modes(analysis, output_dir)
    except Exception as e:
        logger.warning(f"[Charts] Chart 7 failed: {e}")
    
    try:
        chart_radar(analysis, output_dir)
    except Exception as e:
        logger.warning(f"[Charts] Chart 8 failed: {e}")
    
    try:
        chart_correctness_vs_latency(naive_results, crag_results, output_dir)
    except Exception as e:
        logger.warning(f"[Charts] Chart 9 failed: {e}")
    
    try:
        chart_crag_steps(crag_results, output_dir)
    except Exception as e:
        logger.warning(f"[Charts] Chart 10 failed: {e}")
    
    try:
        chart_win_loss_tie(naive_results, crag_results, output_dir)
    except Exception as e:
        logger.warning(f"[Charts] Chart 11 failed: {e}")
    
    try:
        chart_metric_correlation(naive_results, output_dir)
    except Exception as e:
        logger.warning(f"[Charts] Chart 12 failed: {e}")
    
    logger.info(f"[Charts] All charts saved to {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
    generate_all_charts("data/results")
