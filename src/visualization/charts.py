"""
Static Chart Generation (12 Publication-Quality Charts).

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
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

# Color palette
NAIVE_COLOR   = "#6366f1"   # Indigo
AGENTIC_COLOR = "#f97316"   # Orange
BG_COLOR      = "#0f172a"   # Dark navy
GRID_COLOR    = "#1e293b"   # Slate


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
        yaxis=dict(gridcolor=GRID_COLOR, color="white"),
    )
    return fig


# ─────────────────────────────────────────────────────────────
# Chart 1: Overall Score Comparison (Grouped Bar)
# ─────────────────────────────────────────────────────────────
def chart_overall_comparison(analysis: Dict, output_dir: str):
    naive   = analysis["naive_rag"]["avg_metrics"]
    agentic = analysis["agentic_rag"]["avg_metrics"]

    metrics = ["avg_exact_match", "avg_f1", "avg_recall_at_5", "avg_mrr"]
    labels  = ["Exact Match", "F1 Score", "Recall@5", "MRR"]

    fig = go.Figure(data=[
        go.Bar(name="Naive RAG",   x=labels, y=[naive.get(m, 0)   for m in metrics], marker_color=NAIVE_COLOR),
        go.Bar(name="Agentic RAG", x=labels, y=[agentic.get(m, 0) for m in metrics], marker_color=AGENTIC_COLOR),
    ])
    fig.update_layout(barmode="group")
    _dark_layout(fig, "Overall Score Comparison: Naive RAG vs Agentic RAG")
    fig.write_html(os.path.join(output_dir, "01_overall_comparison.html"))
    fig.write_image(os.path.join(output_dir, "01_overall_comparison.png"), width=1200, height=600)
    logger.info("[Charts] Chart 1: Overall comparison saved.")


# ─────────────────────────────────────────────────────────────
# Chart 2: Score Distribution (Box Plot)
# ─────────────────────────────────────────────────────────────
def chart_score_distribution(naive_results: List, agentic_results: List, output_dir: str):
    fig = go.Figure()
    for results, name, color in [
        (naive_results,   "Naive RAG",   NAIVE_COLOR),
        (agentic_results, "Agentic RAG", AGENTIC_COLOR),
    ]:
        vals = [r["metrics"].get("f1", 0) for r in results]
        fig.add_trace(go.Box(y=vals, name=name, marker_color=color, boxmean=True))

    _dark_layout(fig, "F1 Score Distribution")
    fig.write_html(os.path.join(output_dir, "02_score_distribution.html"))
    fig.write_image(os.path.join(output_dir, "02_score_distribution.png"), width=1200, height=600)
    logger.info("[Charts] Chart 2: Score distribution saved.")


# ─────────────────────────────────────────────────────────────
# Chart 3: F1 Score Histogram
# ─────────────────────────────────────────────────────────────
def chart_f1_histogram(naive_results: List, agentic_results: List, output_dir: str):
    fig = go.Figure([
        go.Histogram(x=[r["metrics"].get("f1", 0) for r in naive_results],   name="Naive RAG",   marker_color=NAIVE_COLOR,   opacity=0.7, nbinsx=20),
        go.Histogram(x=[r["metrics"].get("f1", 0) for r in agentic_results], name="Agentic RAG", marker_color=AGENTIC_COLOR, opacity=0.7, nbinsx=20),
    ])
    fig.update_layout(barmode="overlay")
    _dark_layout(fig, "F1 Score Distribution (Histogram)")
    fig.write_html(os.path.join(output_dir, "03_f1_histogram.html"))
    fig.write_image(os.path.join(output_dir, "03_f1_histogram.png"), width=1200, height=600)
    logger.info("[Charts] Chart 3: F1 histogram saved.")


# ─────────────────────────────────────────────────────────────
# Chart 4: Latency Distribution (Box Plot)
# ─────────────────────────────────────────────────────────────
def chart_latency_distribution(naive_results: List, agentic_results: List, output_dir: str):
    fig = go.Figure([
        go.Box(y=[r["latency"] for r in naive_results],   name="Naive RAG",   marker_color=NAIVE_COLOR,   boxmean=True),
        go.Box(y=[r["latency"] for r in agentic_results], name="Agentic RAG", marker_color=AGENTIC_COLOR, boxmean=True),
    ])
    _dark_layout(fig, "Latency Distribution (seconds)")
    fig.update_yaxes(title_text="Latency (s)")
    fig.write_html(os.path.join(output_dir, "04_latency_distribution.html"))
    fig.write_image(os.path.join(output_dir, "04_latency_distribution.png"), width=1200, height=600)
    logger.info("[Charts] Chart 4: Latency distribution saved.")


# ─────────────────────────────────────────────────────────────
# Chart 5: Cost Analysis (Stacked Bar)
# ─────────────────────────────────────────────────────────────
def chart_cost_analysis(analysis: Dict, summary: Dict, output_dir: str):
    naive_cost   = summary.get("naive_rag_cost", {})
    agentic_cost = summary.get("agentic_rag_cost", summary.get("crag_cost", {}))  # backward compat

    fig = go.Figure(data=[
        go.Bar(name="Input Tokens Cost",
               x=["Naive RAG", "Agentic RAG"],
               y=[naive_cost.get("total_prompt_tokens", 0) / 1e6 * 0.15,
                  agentic_cost.get("total_prompt_tokens", 0) / 1e6 * 0.15],
               marker_color=NAIVE_COLOR),
        go.Bar(name="Output Tokens Cost",
               x=["Naive RAG", "Agentic RAG"],
               y=[naive_cost.get("total_completion_tokens", 0) / 1e6 * 0.60,
                  agentic_cost.get("total_completion_tokens", 0) / 1e6 * 0.60],
               marker_color=AGENTIC_COLOR),
    ])
    fig.update_layout(barmode="stack")
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
    naive_f1   = [perf[d]["naive_rag"].get("avg_f1", 0)   for d in difficulties]
    agentic_f1 = [perf[d]["agentic_rag"].get("avg_f1", 0) for d in difficulties]

    fig = go.Figure(data=[
        go.Bar(name="Naive RAG",   x=difficulties, y=naive_f1,   marker_color=NAIVE_COLOR),
        go.Bar(name="Agentic RAG", x=difficulties, y=agentic_f1, marker_color=AGENTIC_COLOR),
    ])
    fig.update_layout(barmode="group")
    _dark_layout(fig, "F1 Score by Difficulty Level")
    fig.write_html(os.path.join(output_dir, "06_perf_by_difficulty.html"))
    fig.write_image(os.path.join(output_dir, "06_perf_by_difficulty.png"), width=1200, height=600)
    logger.info("[Charts] Chart 6: Performance by difficulty saved.")


# ─────────────────────────────────────────────────────────────
# Chart 7: Failure Mode Breakdown (Stacked Bar)
# ─────────────────────────────────────────────────────────────
def chart_failure_modes(analysis: Dict, output_dir: str):
    naive_fm   = analysis["naive_rag"]["failure_modes"]
    agentic_fm = analysis["agentic_rag"]["failure_modes"]

    all_modes = sorted(set(list(naive_fm.keys()) + list(agentic_fm.keys())))
    colors    = px.colors.qualitative.Set2

    fig = go.Figure()
    for i, mode in enumerate(all_modes):
        fig.add_trace(go.Bar(
            name=mode.replace("_", " ").title(),
            x=["Naive RAG", "Agentic RAG"],
            y=[naive_fm.get(mode, {}).get("count", 0), agentic_fm.get(mode, {}).get("count", 0)],
            marker_color=colors[i % len(colors)],
        ))
    fig.update_layout(barmode="stack")
    _dark_layout(fig, "Failure Mode Breakdown")
    fig.write_html(os.path.join(output_dir, "07_failure_modes.html"))
    fig.write_image(os.path.join(output_dir, "07_failure_modes.png"), width=1200, height=600)
    logger.info("[Charts] Chart 7: Failure modes saved.")


# ─────────────────────────────────────────────────────────────
# Chart 8: Radar / Spider Chart
# ─────────────────────────────────────────────────────────────
def chart_radar(analysis: Dict, output_dir: str):
    naive   = analysis["naive_rag"]["avg_metrics"]
    agentic = analysis["agentic_rag"]["avg_metrics"]

    categories = ["Exact Match", "F1 Score", "Recall@5", "MRR"]
    keys       = ["avg_exact_match", "avg_f1", "avg_recall_at_5", "avg_mrr"]

    fig = go.Figure([
        go.Scatterpolar(
            r=[naive.get(k, 0) for k in keys] + [naive.get(keys[0], 0)],
            theta=categories + [categories[0]], name="Naive RAG",
            line=dict(color=NAIVE_COLOR, width=3),
            fill="toself", fillcolor="rgba(99,102,241,0.2)",
        ),
        go.Scatterpolar(
            r=[agentic.get(k, 0) for k in keys] + [agentic.get(keys[0], 0)],
            theta=categories + [categories[0]], name="Agentic RAG",
            line=dict(color=AGENTIC_COLOR, width=3),
            fill="toself", fillcolor="rgba(249,115,22,0.2)",
        ),
    ])
    fig.update_layout(polar=dict(
        bgcolor=BG_COLOR,
        radialaxis=dict(gridcolor=GRID_COLOR, color="white"),
        angularaxis=dict(gridcolor=GRID_COLOR, color="white"),
    ))
    _dark_layout(fig, "Multi-Dimensional Performance Radar")
    fig.write_html(os.path.join(output_dir, "08_radar_chart.html"))
    fig.write_image(os.path.join(output_dir, "08_radar_chart.png"), width=800, height=800)
    logger.info("[Charts] Chart 8: Radar chart saved.")


# ─────────────────────────────────────────────────────────────
# Chart 9: Correctness vs Latency (Scatter)
# ─────────────────────────────────────────────────────────────
def chart_correctness_vs_latency(naive_results: List, agentic_results: List, output_dir: str):
    fig = go.Figure([
        go.Scatter(
            x=[r["latency"] for r in naive_results],
            y=[r["metrics"].get("f1", 0) for r in naive_results],
            mode="markers", name="Naive RAG",
            marker=dict(color=NAIVE_COLOR, size=5, opacity=0.6),
        ),
        go.Scatter(
            x=[r["latency"] for r in agentic_results],
            y=[r["metrics"].get("f1", 0) for r in agentic_results],
            mode="markers", name="Agentic RAG",
            marker=dict(color=AGENTIC_COLOR, size=5, opacity=0.6),
        ),
    ])
    _dark_layout(fig, "Quality vs Speed Tradeoff")
    fig.update_xaxes(title_text="Latency (seconds)")
    fig.update_yaxes(title_text="F1 Score")
    fig.write_html(os.path.join(output_dir, "09_correctness_vs_latency.html"))
    fig.write_image(os.path.join(output_dir, "09_correctness_vs_latency.png"), width=1200, height=600)
    logger.info("[Charts] Chart 9: Correctness vs latency saved.")


# ─────────────────────────────────────────────────────────────
# Chart 10: Agentic RAG Step Distribution (Histogram)
# ─────────────────────────────────────────────────────────────
def chart_agentic_steps(agentic_results: List, output_dir: str):
    step_counts = [len(r.get("steps", [])) for r in agentic_results]
    fig = go.Figure(go.Histogram(x=step_counts, marker_color=AGENTIC_COLOR, nbinsx=15))
    _dark_layout(fig, "Agentic RAG: Steps Taken Per Query")
    fig.update_xaxes(title_text="Number of Steps")
    fig.update_yaxes(title_text="Count")
    fig.write_html(os.path.join(output_dir, "10_agentic_steps.html"))
    fig.write_image(os.path.join(output_dir, "10_agentic_steps.png"), width=1200, height=600)
    logger.info("[Charts] Chart 10: Agentic RAG step distribution saved.")


# ─────────────────────────────────────────────────────────────
# Chart 11: Win/Loss/Tie (Donut)
# ─────────────────────────────────────────────────────────────
def chart_win_loss_tie(naive_results: List, agentic_results: List, output_dir: str):
    naive_by_id   = {r["query_id"]: r for r in naive_results}
    agentic_by_id = {r["query_id"]: r for r in agentic_results}
    common        = set(naive_by_id.keys()) & set(agentic_by_id.keys())

    wins, losses, ties = 0, 0, 0
    for qid in common:
        nf1 = naive_by_id[qid]["metrics"].get("f1", 0)
        af1 = agentic_by_id[qid]["metrics"].get("f1", 0)
        if af1 > nf1:
            wins += 1
        elif nf1 > af1:
            losses += 1
        else:
            ties += 1

    fig = go.Figure(go.Pie(
        labels=["Agentic RAG Wins", "Naive RAG Wins", "Tie"],
        values=[wins, losses, ties],
        hole=0.5,
        marker=dict(colors=[AGENTIC_COLOR, NAIVE_COLOR, "#64748b"]),
        textinfo="label+percent",
    ))
    _dark_layout(fig, "Per-Query Win/Loss/Tie (F1)")
    fig.write_html(os.path.join(output_dir, "11_win_loss_tie.html"))
    fig.write_image(os.path.join(output_dir, "11_win_loss_tie.png"), width=800, height=800)
    logger.info("[Charts] Chart 11: Win/Loss/Tie saved.")


# ─────────────────────────────────────────────────────────────
# Chart 12: Metric Correlation Heatmap (Matplotlib)
# ─────────────────────────────────────────────────────────────
def chart_metric_correlation(naive_results: List, output_dir: str):
    import pandas as pd

    rows = [{
        "EM":       r.get("metrics", {}).get("exact_match", 0),
        "F1":       r.get("metrics", {}).get("f1", 0),
        "Recall@5": r.get("metrics", {}).get("recall_at_5", 0),
        "MRR":      r.get("metrics", {}).get("mrr", 0),
        "Latency":  r.get("latency", 0),
    } for r in naive_results]

    corr = pd.DataFrame(rows).corr()

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

    # Load data — check both new and legacy filenames
    import glob as _glob

    def _latest(pattern):
        files = sorted(_glob.glob(os.path.join(results_dir, pattern)))
        return files[-1] if files else ""

    naive_file   = _latest("naive_rag_results*.jsonl")
    agentic_file = _latest("agentic_rag_results*.jsonl") or _latest("crag_results*.jsonl")

    naive_results   = _load_jsonl(naive_file)   if naive_file   else []
    agentic_results = _load_jsonl(agentic_file) if agentic_file else []

    analysis_path = os.path.join(results_dir, "analysis_report.json")
    with open(analysis_path, "r") as f:
        analysis = json.load(f)

    # Normalize legacy top-level key
    if "corrective_rag" in analysis and "agentic_rag" not in analysis:
        analysis["agentic_rag"] = analysis.pop("corrective_rag")

    summary_path = os.path.join(results_dir, "run_summary.json")
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)

    logger.info(f"[Charts] Generating 12 charts from {len(naive_results)} + {len(agentic_results)} results...")

    for fn, label, *args in [
        (chart_overall_comparison,      "1",  (analysis, output_dir)),
        (chart_score_distribution,      "2",  (naive_results, agentic_results, output_dir)),
        (chart_f1_histogram,            "3",  (naive_results, agentic_results, output_dir)),
        (chart_latency_distribution,    "4",  (naive_results, agentic_results, output_dir)),
        (chart_cost_analysis,           "5",  (analysis, summary, output_dir)),
        (chart_performance_by_difficulty, "6", (analysis, output_dir)),
        (chart_failure_modes,           "7",  (analysis, output_dir)),
        (chart_radar,                   "8",  (analysis, output_dir)),
        (chart_correctness_vs_latency,  "9",  (naive_results, agentic_results, output_dir)),
        (chart_agentic_steps,           "10", (agentic_results, output_dir)),
        (chart_win_loss_tie,            "11", (naive_results, agentic_results, output_dir)),
        (chart_metric_correlation,      "12", (naive_results, output_dir)),
    ]:
        try:
            fn(*args[0])
        except Exception as e:
            logger.warning(f"[Charts] Chart {label} failed: {e}")

    logger.info(f"[Charts] All charts saved to {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%H:%M:%S")
    generate_all_charts("data/results")
