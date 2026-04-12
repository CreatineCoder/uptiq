"""
Interactive Streamlit Dashboard — Naive RAG vs Corrective RAG Benchmark.

Launch with: streamlit run src/visualization/dashboard.py
"""
import os
import sys
import json
import logging

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

RESULTS_DIR = os.path.join(project_root, "data", "results")
NAIVE_COLOR = "#6366f1"
CRAG_COLOR = "#f97316"
BG_COLOR = "#0f172a"
CARD_BG = "#1e293b"
GRID_COLOR = "#334155"
SUCCESS_COLOR = "#22c55e"
WARN_COLOR = "#eab308"

PLOTLY_THEME = dict(
    plot_bgcolor=BG_COLOR,
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e2e8f0", family="Inter, sans-serif", size=13),
    legend=dict(bgcolor="rgba(0,0,0,0.3)", font=dict(color="#e2e8f0"), bordercolor=GRID_COLOR, borderwidth=1),
    xaxis=dict(gridcolor=GRID_COLOR, color="#94a3b8", zerolinecolor=GRID_COLOR),
    yaxis=dict(gridcolor=GRID_COLOR, color="#94a3b8", zerolinecolor=GRID_COLOR),
    margin=dict(t=50, b=40, l=50, r=20),
)

CUSTOM_CSS = """
<style>
    /* Page background */
    .stApp { background-color: #0f172a; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #1e293b; border-right: 1px solid #334155; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMarkdown { color: #94a3b8 !important; }

    /* Agent cards */
    .agent-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px 24px;
        height: 100%;
    }
    .agent-card.naive { border-top: 3px solid #6366f1; }
    .agent-card.crag  { border-top: 3px solid #f97316; }
    .agent-label { font-size: 11px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 6px; }
    .agent-label.naive { color: #818cf8; }
    .agent-label.crag  { color: #fb923c; }
    .metric-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-top: 12px; }
    .metric-item { text-align: center; }
    .metric-value { font-size: 22px; font-weight: 700; color: #f1f5f9; }
    .metric-label { font-size: 10px; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 2px; }
    .metric-delta { font-size: 11px; margin-top: 3px; }
    .delta-pos { color: #22c55e; }
    .delta-neg { color: #ef4444; }
    .delta-neu { color: #64748b; }

    /* Section headers */
    .section-title {
        font-size: 13px;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 1px solid #1e293b;
    }

    /* Query cards */
    .query-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .query-question { font-size: 15px; color: #e2e8f0; font-weight: 500; margin-bottom: 8px; }
    .query-gold { font-size: 13px; color: #94a3b8; margin-bottom: 12px; }
    .answer-block { background: #0f172a; border-radius: 8px; padding: 12px 14px; }
    .answer-label { font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px; }
    .answer-label.naive { color: #818cf8; }
    .answer-label.crag  { color: #fb923c; }
    .answer-text { font-size: 14px; color: #e2e8f0; }
    .score-row { display: flex; gap: 12px; margin-top: 8px; }
    .score-chip { font-size: 11px; background: #1e293b; border: 1px solid #334155; border-radius: 4px; padding: 2px 8px; color: #94a3b8; }
    .score-chip.hit { border-color: #22c55e; color: #22c55e; }

    /* Stat significance badge */
    .sig-badge { display: inline-block; padding: 2px 10px; border-radius: 100px; font-size: 11px; font-weight: 600; }
    .sig-badge.yes { background: rgba(34,197,94,0.15); color: #22c55e; border: 1px solid rgba(34,197,94,0.3); }
    .sig-badge.no  { background: rgba(100,116,139,0.15); color: #64748b; border: 1px solid rgba(100,116,139,0.3); }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; background: #1e293b; border-radius: 10px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; color: #64748b; font-size: 13px; font-weight: 500; }
    .stTabs [aria-selected="true"] { background: #0f172a !important; color: #e2e8f0 !important; }

    /* Dataframe */
    .stDataFrame { border-radius: 8px; overflow: hidden; }

    /* Divider */
    hr { border-color: #1e293b; }

    /* Hide default Streamlit header */
    header[data-testid="stHeader"] { background: transparent; }

    /* Metric delta fix */
    [data-testid="stMetricDelta"] { font-size: 12px; }
</style>
"""


# ─────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────

def _load_jsonl(path: str) -> list:
    results = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    results.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
    return results


def _find_latest_file(directory: str, base_name: str) -> str:
    """Return path to the most recently timestamped file matching base_name."""
    files = [f for f in os.listdir(directory) if f.startswith(base_name) and f.endswith(".jsonl")]
    if not files:
        return os.path.join(directory, f"{base_name}.jsonl")
    files.sort(reverse=True)
    return os.path.join(directory, files[0])


@st.cache_data
def load_run_data(full_path: str) -> tuple:
    naive = _load_jsonl(_find_latest_file(full_path, "naive_rag_results"))
    crag  = _load_jsonl(_find_latest_file(full_path, "crag_results"))

    analysis, summary = {}, {}
    analysis_path = os.path.join(full_path, "analysis_report.json")
    fallback_analysis = os.path.join(RESULTS_DIR, "analysis_report.json")
    if os.path.exists(analysis_path):
        with open(analysis_path) as f:
            analysis = json.load(f)
    elif os.path.exists(fallback_analysis):
        with open(fallback_analysis) as f:
            analysis = json.load(f)

    summary_path = os.path.join(full_path, "run_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)

    ragas  = _load_jsonl(os.path.join(full_path, "ragas_results.jsonl"))
    judge  = _load_jsonl(os.path.join(full_path, "judge_results.jsonl"))
    return naive, crag, analysis, summary, ragas, judge


def results_to_df(results: list) -> pd.DataFrame:
    rows = []
    for r in results:
        m = r.get("metrics", {})
        rows.append({
            "query_id":        r.get("query_id"),
            "agent_type":      r.get("agent_type"),
            "question":        r.get("question"),
            "gold_answer":     r.get("gold_answer"),
            "predicted_answer": r.get("predicted_answer"),
            "dataset":         r.get("dataset"),
            "difficulty":      r.get("difficulty"),
            "exact_match":     m.get("exact_match", 0),
            "f1":              m.get("f1", 0),
            "recall_at_5":     m.get("recall_at_5", 0),
            "mrr":             m.get("mrr", 0),
            "latency":         r.get("latency", 0),
            "cost_usd":        r.get("cost_usd", 0),
            "failure_mode":    r.get("failure_mode", "unknown"),
            "steps":           r.get("steps", []),
            "num_steps":       len(r.get("steps", [])),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# Chart helpers
# ─────────────────────────────────────────────────────────────

def _apply_theme(fig: go.Figure, title: str = "", height: int = 380) -> go.Figure:
    fig.update_layout(height=height, title=dict(text=title, font=dict(size=14, color="#e2e8f0")), **PLOTLY_THEME)
    return fig


def chart_overall_comparison(analysis: dict) -> go.Figure:
    naive = analysis.get("naive_rag", {}).get("avg_metrics", {})
    crag  = analysis.get("corrective_rag", {}).get("avg_metrics", {})
    keys   = ["avg_exact_match", "avg_f1", "avg_recall_at_5", "avg_mrr"]
    labels = ["Exact Match", "F1 Score", "Recall@5", "MRR"]
    fig = go.Figure([
        go.Bar(name="Naive RAG", x=labels, y=[naive.get(k, 0) for k in keys],
               marker_color=NAIVE_COLOR, marker_line_width=0),
        go.Bar(name="CRAG",      x=labels, y=[crag.get(k, 0) for k in keys],
               marker_color=CRAG_COLOR,  marker_line_width=0),
    ])
    fig.update_layout(barmode="group")
    return _apply_theme(fig, "Score Comparison — All Metrics", 360)


def chart_radar(analysis: dict) -> go.Figure:
    naive = analysis.get("naive_rag", {}).get("avg_metrics", {})
    crag  = analysis.get("corrective_rag", {}).get("avg_metrics", {})
    cats  = ["Exact Match", "F1 Score", "Recall@5", "MRR"]
    keys  = ["avg_exact_match", "avg_f1", "avg_recall_at_5", "avg_mrr"]

    fig = go.Figure([
        go.Scatterpolar(
            r=[naive.get(k, 0) for k in keys] + [naive.get(keys[0], 0)],
            theta=cats + [cats[0]], name="Naive RAG",
            line=dict(color=NAIVE_COLOR, width=2.5),
            fill="toself", fillcolor="rgba(99,102,241,0.15)",
        ),
        go.Scatterpolar(
            r=[crag.get(k, 0) for k in keys] + [crag.get(keys[0], 0)],
            theta=cats + [cats[0]], name="CRAG",
            line=dict(color=CRAG_COLOR, width=2.5),
            fill="toself", fillcolor="rgba(249,115,22,0.15)",
        ),
    ])
    fig.update_layout(
        polar=dict(
            bgcolor=BG_COLOR,
            radialaxis=dict(gridcolor=GRID_COLOR, color="#64748b", range=[0, 0.6]),
            angularaxis=dict(gridcolor=GRID_COLOR, color="#94a3b8"),
        ),
        height=380,
        **{k: v for k, v in PLOTLY_THEME.items() if k not in ("xaxis", "yaxis")},
        title=dict(text="Multi-Dimensional Radar", font=dict(size=14, color="#e2e8f0")),
        margin=dict(t=50, b=20, l=40, r=40),
    )
    return fig


def chart_win_loss_tie(naive_df: pd.DataFrame, crag_df: pd.DataFrame) -> go.Figure:
    merged = naive_df[["query_id", "f1"]].merge(crag_df[["query_id", "f1"]], on="query_id", suffixes=("_n", "_c"))
    wins  = (merged["f1_c"] > merged["f1_n"]).sum()
    losses = (merged["f1_n"] > merged["f1_c"]).sum()
    ties  = (merged["f1_n"] == merged["f1_c"]).sum()
    fig = go.Figure(go.Pie(
        labels=["CRAG Wins", "Naive RAG Wins", "Tie"],
        values=[wins, losses, ties],
        hole=0.55,
        marker=dict(colors=[CRAG_COLOR, NAIVE_COLOR, "#475569"]),
        textinfo="label+percent",
        textfont=dict(size=12),
    ))
    _apply_theme(fig, "Per-Query Win / Loss / Tie  (F1)", 360)
    fig.update_traces(hovertemplate="%{label}: %{value} queries")
    return fig


def chart_f1_histogram(naive_df: pd.DataFrame, crag_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure([
        go.Histogram(x=naive_df["f1"], name="Naive RAG", marker_color=NAIVE_COLOR, opacity=0.75, nbinsx=20),
        go.Histogram(x=crag_df["f1"],  name="CRAG",      marker_color=CRAG_COLOR,  opacity=0.75, nbinsx=20),
    ])
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title_text="F1 Score")
    fig.update_yaxes(title_text="Queries")
    return _apply_theme(fig, "F1 Score Distribution", 320)


def chart_latency(naive_df: pd.DataFrame, crag_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure([
        go.Box(y=naive_df["latency"], name="Naive RAG", marker_color=NAIVE_COLOR, boxmean=True),
        go.Box(y=crag_df["latency"],  name="CRAG",      marker_color=CRAG_COLOR,  boxmean=True),
    ])
    fig.update_yaxes(title_text="Latency (s)")
    return _apply_theme(fig, "Latency Distribution", 320)


def chart_scatter_quality_speed(naive_df: pd.DataFrame, crag_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure([
        go.Scatter(x=naive_df["latency"], y=naive_df["f1"], mode="markers", name="Naive RAG",
                   marker=dict(color=NAIVE_COLOR, size=6, opacity=0.65)),
        go.Scatter(x=crag_df["latency"],  y=crag_df["f1"],  mode="markers", name="CRAG",
                   marker=dict(color=CRAG_COLOR, size=6, opacity=0.65)),
    ])
    fig.update_xaxes(title_text="Latency (s)")
    fig.update_yaxes(title_text="F1 Score")
    return _apply_theme(fig, "Quality vs Speed Tradeoff", 320)


def chart_failure_donut(failure_modes: dict, title: str, color_seq) -> go.Figure:
    labels = [k.replace("_", " ").title() for k in failure_modes]
    values = [v["count"] for v in failure_modes.values()]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.55,
        marker=dict(colors=color_seq),
        textinfo="label+percent", textfont=dict(size=11),
    ))
    _apply_theme(fig, title, 300)
    return fig


def chart_crag_steps(crag_df: pd.DataFrame) -> go.Figure:
    counts = crag_df["num_steps"].value_counts().sort_index()
    fig = go.Figure(go.Bar(x=counts.index.tolist(), y=counts.values.tolist(), marker_color=CRAG_COLOR, marker_line_width=0))
    fig.update_xaxes(title_text="Number of Steps", dtick=1)
    fig.update_yaxes(title_text="Queries")
    return _apply_theme(fig, "CRAG Step Distribution", 300)


def chart_significance(sig_tests: dict) -> go.Figure:
    metrics = list(sig_tests.keys())
    pvals   = [sig_tests[m]["p_value"] for m in metrics]
    colors  = [SUCCESS_COLOR if p < 0.05 else NAIVE_COLOR for p in pvals]
    labels  = [m.replace("_", " ").title() for m in metrics]
    fig = go.Figure(go.Bar(x=labels, y=pvals, marker_color=colors, marker_line_width=0))
    fig.add_hline(y=0.05, line_dash="dash", line_color="#ef4444",
                  annotation_text="p=0.05", annotation_font=dict(color="#ef4444"))
    fig.update_yaxes(title_text="p-value", range=[0, max(pvals) * 1.15])
    return _apply_theme(fig, "Statistical Significance  (p-values)", 320)


# ─────────────────────────────────────────────────────────────
# KPI card helpers
# ─────────────────────────────────────────────────────────────

def _delta_html(val, baseline) -> str:
    if baseline == 0:
        return ""
    pct = (val - baseline) / baseline * 100
    cls = "delta-pos" if pct > 0 else ("delta-neg" if pct < 0 else "delta-neu")
    sign = "+" if pct > 0 else ""
    return f'<div class="metric-delta {cls}">{sign}{pct:.1f}%</div>'


def kpi_cards(naive_df: pd.DataFrame, crag_df: pd.DataFrame, show_naive: bool, show_crag: bool):
    def _agg(df):
        return {
            "em":  df["exact_match"].mean() if len(df) else 0,
            "f1":  df["f1"].mean() if len(df) else 0,
            "r5":  df["recall_at_5"].mean() if len(df) else 0,
            "mrr": df["mrr"].mean() if len(df) else 0,
            "lat": df["latency"].mean() if len(df) else 0,
            "cost": df["cost_usd"].sum() if len(df) else 0,
        }

    n = _agg(naive_df)
    c = _agg(crag_df)

    cols = st.columns([1, 1] if (show_naive and show_crag) else [1])

    def _render_card(agg, other, cls, label, col):
        with col:
            st.markdown(f"""
            <div class="agent-card {cls}">
                <div class="agent-label {cls}">{label}</div>
                <div class="metric-grid">
                    <div class="metric-item">
                        <div class="metric-value">{agg['em']:.1%}</div>
                        <div class="metric-label">Exact Match</div>
                        {_delta_html(agg['em'], other['em']) if other['em'] > 0 else ''}
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">{agg['f1']:.1%}</div>
                        <div class="metric-label">F1 Score</div>
                        {_delta_html(agg['f1'], other['f1']) if other['f1'] > 0 else ''}
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">{agg['r5']:.1%}</div>
                        <div class="metric-label">Recall@5</div>
                        {_delta_html(agg['r5'], other['r5']) if other['r5'] > 0 else ''}
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">{agg['mrr']:.3f}</div>
                        <div class="metric-label">MRR</div>
                        {_delta_html(agg['mrr'], other['mrr']) if other['mrr'] > 0 else ''}
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">{agg['lat']:.2f}s</div>
                        <div class="metric-label">Avg Latency</div>
                        {_delta_html(agg['lat'], other['lat']) if other['lat'] > 0 else ''}
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">${agg['cost']:.4f}</div>
                        <div class="metric-label">Total Cost</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    col_idx = 0
    if show_naive:
        _render_card(n, c, "naive", "Naive RAG", cols[col_idx])
        col_idx += 1
    if show_crag:
        _render_card(c, n, "crag", "Corrective RAG", cols[col_idx])


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="RAG Benchmark", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── Sidebar: Run selector ──
    st.sidebar.markdown("### Benchmark Run")
    base_results = RESULTS_DIR
    valid_dirs: dict[str, str] = {}

    for d in sorted(os.listdir(base_results)):
        full_d = os.path.join(base_results, d)
        if os.path.isdir(full_d) and any(f.endswith(".jsonl") for f in os.listdir(full_d)):
            valid_dirs[d] = full_d
    if any(f.endswith(".jsonl") for f in os.listdir(base_results)
           if os.path.isfile(os.path.join(base_results, f))):
        valid_dirs["data/results (default)"] = base_results
    if not valid_dirs:
        valid_dirs["data/results (default)"] = base_results

    selected_run = st.sidebar.selectbox("Run", list(valid_dirs.keys()), label_visibility="collapsed")
    results_path = valid_dirs[selected_run]

    naive_results, crag_results, analysis, summary, ragas_results, judge_results = load_run_data(results_path)

    if not naive_results and not crag_results:
        st.error(f"No results found in **{selected_run}**. Run the benchmark first.")
        return

    naive_df    = results_to_df(naive_results)
    crag_df     = results_to_df(crag_results)
    combined_df = pd.concat([naive_df, crag_df], ignore_index=True)

    # ── Sidebar: Filters ──
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Filters")
    avail_datasets = ["All"] + sorted(combined_df["dataset"].dropna().unique().tolist())
    sel_dataset = st.sidebar.selectbox("Dataset", avail_datasets)
    sel_agent   = st.sidebar.selectbox("Agent",   ["Both", "naive_rag", "corrective_rag"])

    filtered = combined_df.copy()
    if sel_dataset != "All":
        filtered = filtered[filtered["dataset"] == sel_dataset]
    if sel_agent != "Both":
        filtered = filtered[filtered["agent_type"] == sel_agent]

    n_filt = filtered[filtered["agent_type"] == "naive_rag"]
    c_filt = filtered[filtered["agent_type"] == "corrective_rag"]
    show_naive = sel_agent in ("Both", "naive_rag")
    show_crag  = sel_agent in ("Both", "corrective_rag")

    # ── Sidebar: Run info ──
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Run Info")
    n_count = len(naive_df)
    c_count = len(crag_df)
    if n_count or c_count:
        st.sidebar.markdown(f"**Naive RAG:** {n_count} queries")
        st.sidebar.markdown(f"**CRAG:** {c_count} queries")
    if summary:
        naive_cost = summary.get("naive_rag_cost", {}).get("total_cost_usd", 0)
        crag_cost  = summary.get("crag_cost",  {}).get("total_cost_usd", 0)
        st.sidebar.markdown(f"**Total API cost:** ${naive_cost + crag_cost:.4f}")

    # ── Header ──
    st.markdown("## RAG Benchmark Dashboard")
    st.markdown('<p style="color:#64748b;margin-top:-12px;margin-bottom:20px">Naive RAG vs Corrective RAG &nbsp;·&nbsp; HotpotQA</p>', unsafe_allow_html=True)

    # ── KPI Cards ──
    kpi_cards(n_filt if len(n_filt) else naive_df,
              c_filt if len(c_filt) else crag_df,
              show_naive, show_crag)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──
    has_ragas = bool(ragas_results)
    tab_labels = ["Overview", "Query Browser", "Failure Analysis", "Statistics"]
    if has_ragas:
        tab_labels.append("RAGAS")
    tab_labels.append("Export")

    tabs = st.tabs(tab_labels)
    tab_map = {label: tab for label, tab in zip(tab_labels, tabs)}

    # ── Tab: Overview ──
    with tab_map["Overview"]:
        if analysis:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(chart_overall_comparison(analysis), use_container_width=True)
            with col2:
                st.plotly_chart(chart_radar(analysis), use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(chart_f1_histogram(n_filt if len(n_filt) else naive_df,
                                                c_filt if len(c_filt) else crag_df),
                            use_container_width=True)
        with col4:
            st.plotly_chart(chart_latency(n_filt if len(n_filt) else naive_df,
                                           c_filt if len(c_filt) else crag_df),
                            use_container_width=True)

        col5, col6 = st.columns(2)
        with col5:
            st.plotly_chart(chart_scatter_quality_speed(n_filt if len(n_filt) else naive_df,
                                                         c_filt if len(c_filt) else crag_df),
                            use_container_width=True)
        with col6:
            if not naive_df.empty and not crag_df.empty:
                st.plotly_chart(chart_win_loss_tie(naive_df, crag_df), use_container_width=True)

    # ── Tab: Query Browser ──
    with tab_map["Query Browser"]:
        col_search, col_fm, col_result = st.columns([3, 2, 2])
        with col_search:
            search_text = st.text_input("Search questions", placeholder="Type to filter…")
        with col_fm:
            fm_options = ["All"] + sorted(combined_df["failure_mode"].dropna().unique().tolist())
            sel_fm = st.selectbox("Failure mode", fm_options)
        with col_result:
            sel_result = st.selectbox("Result", ["All", "Correct (EM=1)", "Incorrect (EM=0)"])

        naive_ids = set(naive_df["query_id"].tolist())
        crag_ids  = set(crag_df["query_id"].tolist())
        common_ids = sorted(naive_ids & crag_ids)

        # Apply filters
        view_naive = naive_df[naive_df["query_id"].isin(common_ids)].copy()
        if search_text:
            view_naive = view_naive[view_naive["question"].str.contains(search_text, case=False, na=False)]
        if sel_fm != "All":
            view_naive = view_naive[view_naive["failure_mode"] == sel_fm]
        if sel_result == "Correct (EM=1)":
            view_naive = view_naive[view_naive["exact_match"] == 1]
        elif sel_result == "Incorrect (EM=0)":
            view_naive = view_naive[view_naive["exact_match"] == 0]

        display_ids = view_naive["query_id"].tolist()
        st.caption(f"{len(display_ids)} queries matching filters")

        crag_lookup  = crag_df.set_index("query_id").to_dict("index") if not crag_df.empty else {}

        for qid in display_ids[:20]:
            n_row = view_naive[view_naive["query_id"] == qid].iloc[0]
            c_data = crag_lookup.get(qid)

            em_n = int(n_row["exact_match"])
            em_c = int(c_data["exact_match"]) if c_data else None

            with st.expander(f"{qid}  ·  {n_row['question'][:90]}{'…' if len(n_row['question']) > 90 else ''}", expanded=False):
                st.markdown(f"**Question:** {n_row['question']}")
                st.markdown(f"**Gold answer:** `{n_row['gold_answer']}`")
                st.markdown(f"**Dataset:** {n_row['dataset']}  ·  **Difficulty:** {n_row['difficulty']}")
                st.markdown("---")
                col_n, col_c = st.columns(2)
                with col_n:
                    st.markdown(f'<div class="answer-label naive">Naive RAG</div>', unsafe_allow_html=True)
                    st.markdown(n_row["predicted_answer"])
                    chips = f'<span class="score-chip {"hit" if em_n else ""}">EM={em_n}</span>'
                    chips += f' <span class="score-chip">F1={n_row["f1"]:.3f}</span>'
                    chips += f' <span class="score-chip">{n_row["latency"]:.2f}s</span>'
                    st.markdown(f'<div class="score-row">{chips}</div>', unsafe_allow_html=True)
                with col_c:
                    if c_data:
                        st.markdown(f'<div class="answer-label crag">Corrective RAG</div>', unsafe_allow_html=True)
                        st.markdown(c_data["predicted_answer"])
                        chips = f'<span class="score-chip {"hit" if em_c else ""}">EM={em_c}</span>'
                        chips += f' <span class="score-chip">F1={c_data["f1"]:.3f}</span>'
                        chips += f' <span class="score-chip">{c_data["latency"]:.2f}s</span>'
                        steps_str = " → ".join(c_data["steps"]) if isinstance(c_data["steps"], list) else str(c_data["steps"])
                        chips += f' <span class="score-chip">{steps_str}</span>'
                        st.markdown(f'<div class="score-row">{chips}</div>', unsafe_allow_html=True)

        if len(display_ids) > 20:
            st.caption(f"Showing first 20 of {len(display_ids)} results. Use filters to narrow down.")

    # ── Tab: Failure Analysis ──
    with tab_map["Failure Analysis"]:
        if analysis:
            fm_naive = analysis.get("naive_rag", {}).get("failure_modes", {})
            fm_crag  = analysis.get("corrective_rag", {}).get("failure_modes", {})

            FAILURE_COLORS = ["#22c55e", "#ef4444", "#f97316", "#eab308", "#8b5cf6", "#06b6d4", "#64748b"]

            col_d1, col_d2 = st.columns(2)
            with col_d1:
                if fm_naive:
                    st.plotly_chart(chart_failure_donut(fm_naive, "Naive RAG — Failure Modes", FAILURE_COLORS), use_container_width=True)
            with col_d2:
                if fm_crag:
                    st.plotly_chart(chart_failure_donut(fm_crag, "CRAG — Failure Modes", FAILURE_COLORS), use_container_width=True)

            # Side-by-side table comparison
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                if fm_naive:
                    df_n = pd.DataFrame([
                        {"Mode": k.replace("_", " ").title(), "Count": v["count"], "%": f"{v['percentage']:.0f}%"}
                        for k, v in sorted(fm_naive.items(), key=lambda x: -x[1]["count"])
                    ])
                    st.dataframe(df_n, use_container_width=True, hide_index=True)
            with col_t2:
                if fm_crag:
                    df_c = pd.DataFrame([
                        {"Mode": k.replace("_", " ").title(), "Count": v["count"], "%": f"{v['percentage']:.0f}%"}
                        for k, v in sorted(fm_crag.items(), key=lambda x: -x[1]["count"])
                    ])
                    st.dataframe(df_c, use_container_width=True, hide_index=True)

            # CRAG internals
            crag_comp = analysis.get("corrective_rag", {}).get("component_analysis", {})
            if crag_comp:
                st.markdown("---")
                st.markdown("**CRAG Component Activity**")
                col_r, col_w, col_h, col_steps = st.columns(4)
                rw = crag_comp.get("rewrites", {})
                ws = crag_comp.get("web_search", {})
                hr = crag_comp.get("hallucination_retries", {})
                sd = crag_comp.get("step_distribution", {})
                with col_r:
                    st.metric("Query Rewrites", rw.get("triggered", 0),
                              delta=f"{rw.get('trigger_rate', 0):.0f}% trigger rate" if rw.get("trigger_rate") else None)
                with col_w:
                    st.metric("Web Searches", ws.get("triggered", 0),
                              delta=f"{ws.get('trigger_rate', 0):.0f}% trigger rate" if ws.get("trigger_rate") else None)
                with col_h:
                    st.metric("Hallucination Retries", hr.get("triggered", 0))
                with col_steps:
                    st.metric("Avg Steps / Query", f"{sd.get('mean', 0):.1f}",
                              delta=f"max {sd.get('max', 0)}")

                if not crag_df.empty:
                    st.plotly_chart(chart_crag_steps(crag_df), use_container_width=True)

    # ── Tab: Statistics ──
    with tab_map["Statistics"]:
        sig = analysis.get("significance_tests", {}) if analysis else {}
        if sig:
            st.plotly_chart(chart_significance(sig), use_container_width=True)

            rows = []
            for metric, res in sig.items():
                rows.append({
                    "Metric":     metric.replace("_", " ").title(),
                    "p-value":    f"{res['p_value']:.4f}",
                    "Significant": "Yes ✓" if res["significant"] else "No",
                    "Effect Size": res.get("effect_size", "—"),
                    "Naive Mean":  res.get("naive_mean", "—"),
                    "CRAG Mean":   res.get("crag_mean", "—"),
                    "CRAG Wins":   res.get("crag_wins", "—"),
                    "Naive Wins":  res.get("naive_wins", "—"),
                    "Ties":        res.get("ties", "—"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("Run `python src/analysis/analyzer.py` first to generate significance tests.")

    # ── Tab: RAGAS (optional) ──
    if has_ragas:
        with tab_map["RAGAS"]:
            ragas_df = pd.DataFrame(ragas_results)
            if "ragas_scores" in ragas_df.columns:
                ragas_expanded = ragas_df.join(pd.json_normalize(ragas_df["ragas_scores"]))
                score_cols = [c for c in ragas_expanded.columns if c not in ("query_id", "agent_type", "ragas_scores")]

                for agent_type, color in [("naive_rag", NAIVE_COLOR), ("corrective_rag", CRAG_COLOR)]:
                    subset = ragas_expanded[ragas_expanded["agent_type"] == agent_type]
                    if subset.empty:
                        continue
                    label = "Naive RAG" if agent_type == "naive_rag" else "CRAG"
                    means = subset[score_cols].mean()
                    st.markdown(f"**{label}**")
                    cols = st.columns(len(score_cols))
                    for i, (col_name, val) in enumerate(means.items()):
                        with cols[i]:
                            st.metric(col_name.replace("_", " ").title(), f"{val:.3f}")
                    st.markdown("---")

    # ── Tab: Export ──
    with tab_map["Export"]:
        st.markdown("**Download Results**")
        col_e1, col_e2, col_e3 = st.columns(3)
        with col_e1:
            st.download_button("Naive RAG (CSV)", naive_df.drop(columns=["steps"], errors="ignore").to_csv(index=False),
                               "naive_rag_results.csv", "text/csv", use_container_width=True)
        with col_e2:
            st.download_button("CRAG (CSV)", crag_df.drop(columns=["steps"], errors="ignore").to_csv(index=False),
                               "crag_results.csv", "text/csv", use_container_width=True)
        with col_e3:
            st.download_button("Combined (CSV)", combined_df.drop(columns=["steps"], errors="ignore").to_csv(index=False),
                               "combined_results.csv", "text/csv", use_container_width=True)

        if analysis:
            st.markdown("**Download Analysis**")
            st.download_button(
                "Analysis Report (JSON)",
                json.dumps(analysis, indent=2),
                "analysis_report.json", "application/json",
            )


if __name__ == "__main__":
    main()
