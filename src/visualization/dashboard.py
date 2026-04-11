"""
Phase 6.2 — Interactive Streamlit Dashboard.

Launch with: streamlit run src/visualization/dashboard.py

Features:
- Dropdown filters (dataset, difficulty, agent)
- Interactive Plotly charts
- Sample query browser (side-by-side comparison)
- Failure mode drill-down
- Download raw results as CSV
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

# ─────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────

RESULTS_DIR = os.path.join(project_root, "data", "results")
FIGURES_DIR = os.path.join(project_root, "reports", "figures")

@st.cache_data
def load_results():
    naive = _load_jsonl(os.path.join(RESULTS_DIR, "naive_rag_results.jsonl"))
    crag = _load_jsonl(os.path.join(RESULTS_DIR, "crag_results.jsonl"))
    
    analysis = {}
    analysis_path = os.path.join(RESULTS_DIR, "analysis_report.json")
    if os.path.exists(analysis_path):
        with open(analysis_path, "r") as f:
            analysis = json.load(f)
    
    summary = {}
    summary_path = os.path.join(RESULTS_DIR, "run_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)
    
    judge = _load_jsonl(os.path.join(RESULTS_DIR, "judge_results.jsonl"))
    
    return naive, crag, analysis, summary, judge


def _load_jsonl(path):
    results = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    results.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
    return results


def results_to_df(results):
    rows = []
    for r in results:
        m = r.get("metrics", {})
        rows.append({
            "query_id": r.get("query_id"),
            "agent_type": r.get("agent_type"),
            "question": r.get("question"),
            "gold_answer": r.get("gold_answer"),
            "predicted_answer": r.get("predicted_answer"),
            "dataset": r.get("dataset"),
            "difficulty": r.get("difficulty"),
            "exact_match": m.get("exact_match", 0),
            "f1": m.get("f1", 0),
            "recall_at_5": m.get("recall_at_5", 0),
            "mrr": m.get("mrr", 0),
            "latency": r.get("latency", 0),
            "cost_usd": r.get("cost_usd", 0),
            "failure_mode": r.get("failure_mode", "unknown"),
            "steps": str(r.get("steps", []))
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# Dashboard Layout
# ─────────────────────────────────────────────────────────────

NAIVE_COLOR = "#6366f1"
CRAG_COLOR = "#f97316"

def main():
    st.set_page_config(page_title="RAG Benchmark Dashboard", page_icon=None, layout="wide")
    
    st.title("Naive RAG vs Corrective RAG — Benchmark Dashboard")
    st.markdown("---")
    
    # Load data
    st.sidebar.header("Data Sources")
    
    # Auto-detect available results directories
    base_results = os.path.join(project_root, "data", "results")
    
    valid_dirs = {}  # display_name -> full_path
    
    # First, add named subdirectory runs (e.g. nq_10, hotpot_20)
    for d in sorted(os.listdir(base_results)):
        full_d = os.path.join(base_results, d)
        if os.path.isdir(full_d):
            files = os.listdir(full_d)
            if any(f.endswith('.jsonl') for f in files):
                valid_dirs[d] = full_d
    
    # Also add the root results dir if it has result files
    root_files = os.listdir(base_results)
    if any(f.endswith('.jsonl') for f in root_files if os.path.isfile(os.path.join(base_results, f))):
        valid_dirs["data/results (default)"] = base_results
    
    if not valid_dirs:
        valid_dirs["data/results (default)"] = base_results
    
    run_options = list(valid_dirs.keys())
    selected_run = st.sidebar.selectbox("Select Benchmark Run", run_options, index=0)
    selected_results_full_path = valid_dirs[selected_run]
    
    @st.cache_data
    def load_run_data(full_path):
        # Helper to find the latest timestamped file or default
        def find_latest_file(base_name):
            files = [f for f in os.listdir(full_path) if f.startswith(base_name) and f.endswith(".jsonl")]
            if not files:
                return os.path.join(full_path, f"{base_name}.jsonl")
            # Sort by filename which includes timestamp (e.g. naive_rag_results_1712500000.jsonl)
            # This works because timestamps are numeric and fixed length
            files.sort(reverse=True)
            return os.path.join(full_path, files[0])

        naive_path = find_latest_file("naive_rag_results")
        crag_path = find_latest_file("crag_results")
        
        naive = _load_jsonl(naive_path)
        crag = _load_jsonl(crag_path)
        
        analysis = {}
        analysis_path = os.path.join(full_path, "analysis_report.json")
        # Try to find the latest valid analysis report 
        if os.path.exists(analysis_path):
            with open(analysis_path, "r") as f:
                analysis = json.load(f)
        elif os.path.exists(os.path.join(RESULTS_DIR, "analysis_report.json")):
            with open(os.path.join(RESULTS_DIR, "analysis_report.json"), "r") as f:
                analysis = json.load(f)
        
        summary = {}
        summary_path = os.path.join(full_path, "run_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                summary = json.load(f)
        
        judge = _load_jsonl(os.path.join(full_path, "judge_results.jsonl"))
        return naive, crag, analysis, summary, judge

    naive_results, crag_results, analysis, summary, judge_results = load_run_data(selected_results_full_path)
    
    if not naive_results and not crag_results:
        st.error(f"No benchmark results found in '{selected_run}'. Please run the benchmark first.")
        return
    
    naive_df = results_to_df(naive_results)
    crag_df = results_to_df(crag_results)
    combined_df = pd.concat([naive_df, crag_df], ignore_index=True)
    
    # ── Sidebar Filters ──
    st.sidebar.header("Filters")
    
    # Dynamically get available datasets or fallback to known ones
    available_datasets = ["All"] + sorted(combined_df["dataset"].dropna().unique().tolist())
    if len(available_datasets) == 1:
        available_datasets = ["All", "squad_v2", "hotpotqa"]
        
    selected_dataset = st.sidebar.selectbox("Dataset", available_datasets)
    selected_agent = st.sidebar.selectbox("Agent", ["Both", "naive_rag", "corrective_rag"])
    
    # Apply filters
    filtered = combined_df.copy()
    if selected_dataset != "All":
        filtered = filtered[filtered["dataset"] == selected_dataset]
    if selected_agent != "Both":
        filtered = filtered[filtered["agent_type"] == selected_agent]
    
    # ── KPI Row ──
    st.header("Key Metrics")
    
    naive_filtered = filtered[filtered["agent_type"] == "naive_rag"]
    crag_filtered = filtered[filtered["agent_type"] == "corrective_rag"]
    
    n_em = naive_filtered['exact_match'].mean() if len(naive_filtered) > 0 else 0
    n_f1 = naive_filtered['f1'].mean() if len(naive_filtered) > 0 else 0
    n_lat = naive_filtered['latency'].mean() if len(naive_filtered) > 0 else 0
    
    c_em = crag_filtered['exact_match'].mean() if len(crag_filtered) > 0 else 0
    c_f1 = crag_filtered['f1'].mean() if len(crag_filtered) > 0 else 0
    c_lat = crag_filtered['latency'].mean() if len(crag_filtered) > 0 else 0
    
    def pct_change(val, baseline, baseline_name):
        if baseline == 0: return None
        return f"{(val - baseline) / baseline * 100:+.2f}% vs {baseline_name}"
        
    show_naive = selected_agent in ["Both", "naive_rag"]
    show_crag = selected_agent in ["Both", "corrective_rag"]
    
    # Calculate how many display columns we need based on selections
    num_cols = (3 if show_naive else 0) + (3 if show_crag else 0)
    cols = st.columns(max(num_cols, 1)) # Ensure at least 1 column for safety
    
    col_idx = 0
    if show_naive:
        with cols[col_idx]:
            st.metric("Naive Exact Match (EM)", f"{n_em:.2%}", delta=pct_change(n_em, c_em, "CRAG") if c_em > 0 and show_crag else None)
        with cols[col_idx+1]:
            st.metric("Naive F1 Score", f"{n_f1:.2%}", delta=pct_change(n_f1, c_f1, "CRAG") if c_f1 > 0 and show_crag else None)
        with cols[col_idx+2]:
            st.metric("Naive Avg Latency", f"{n_lat:.2f}s", delta=pct_change(n_lat, c_lat, "CRAG") if c_lat > 0 and show_crag else None, delta_color="inverse")
        col_idx += 3
        
    if show_crag:
        with cols[col_idx]:
            st.metric("CRAG Exact Match (EM)", f"{c_em:.2%}", delta=pct_change(c_em, n_em, "Naive RAG") if n_em > 0 and show_naive else None)
        with cols[col_idx+1]:
            st.metric("CRAG F1 Score", f"{c_f1:.2%}", delta=pct_change(c_f1, n_f1, "Naive RAG") if n_f1 > 0 and show_naive else None)
        with cols[col_idx+2]:
            st.metric("CRAG Avg Latency", f"{c_lat:.2f}s", delta=pct_change(c_lat, n_lat, "Naive RAG") if n_lat > 0 and show_naive else None, delta_color="inverse")
    
    st.markdown("---")
    
    # ── Tab Layout ──
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Charts", "Query Browser", "Failure Analysis", "Statistics", "Export"])
    
    with tab1:
        st.subheader("Overall Comparison")
        if analysis:
            naive_m = analysis.get("naive_rag", {}).get("avg_metrics", {})
            crag_m = analysis.get("corrective_rag", {}).get("avg_metrics", {})
            
            metrics = ["avg_exact_match", "avg_f1", "avg_recall_at_5", "avg_mrr"]
            labels = ["Exact Match", "F1 Score", "Recall@5", "MRR"]
            
            fig = go.Figure(data=[
                go.Bar(name="Naive RAG", x=labels, y=[naive_m.get(m, 0) for m in metrics], marker_color=NAIVE_COLOR),
                go.Bar(name="CRAG", x=labels, y=[crag_m.get(m, 0) for m in metrics], marker_color=CRAG_COLOR)
            ])
            fig.update_layout(barmode='group', template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Latency comparison
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("F1 Distribution")
            fig2 = go.Figure()
            if len(naive_filtered) > 0:
                fig2.add_trace(go.Histogram(x=naive_filtered["f1"], name="Naive RAG", marker_color=NAIVE_COLOR, opacity=0.7))
            if len(crag_filtered) > 0:
                fig2.add_trace(go.Histogram(x=crag_filtered["f1"], name="CRAG", marker_color=CRAG_COLOR, opacity=0.7))
            fig2.update_layout(barmode='overlay', template="plotly_dark", height=350)
            st.plotly_chart(fig2, use_container_width=True)
        
        with col_b:
            st.subheader("Latency Distribution")
            fig3 = go.Figure()
            if len(naive_filtered) > 0:
                fig3.add_trace(go.Box(y=naive_filtered["latency"], name="Naive RAG", marker_color=NAIVE_COLOR))
            if len(crag_filtered) > 0:
                fig3.add_trace(go.Box(y=crag_filtered["latency"], name="CRAG", marker_color=CRAG_COLOR))
            fig3.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig3, use_container_width=True)
    
    with tab2:
        st.subheader("Side-by-Side Query Browser")
        
        # Get common query IDs
        naive_ids = set(naive_df["query_id"].tolist())
        crag_ids = set(crag_df["query_id"].tolist())
        common_ids = sorted(naive_ids & crag_ids)
        
        if common_ids:
            selected_ids = st.multiselect("Select Query IDs (up to 5 by default)", common_ids, default=common_ids[:5])
            
            for selected_id in selected_ids:
                n_row = naive_df[naive_df["query_id"] == selected_id].iloc[0] if selected_id in naive_ids else None
                c_row = crag_df[crag_df["query_id"] == selected_id].iloc[0] if selected_id in crag_ids else None
                
                if n_row is not None:
                    st.markdown(f"**Question ID:** {selected_id}")
                    st.markdown(f"**Question:** {n_row['question']}")
                    st.markdown(f"**Gold Answer:** `{n_row['gold_answer']}`")
                    st.markdown(f"**Dataset:** {n_row['dataset']}")
                    
                    col_n, col_c = st.columns(2)
                    with col_n:
                        st.markdown("### Naive RAG")
                        st.markdown(f"**Answer:** {n_row['predicted_answer']}")
                        st.markdown(f"EM: `{n_row['exact_match']}` | F1: `{n_row['f1']:.3f}` | Latency: `{n_row['latency']:.2f}s`")
                    
                    with col_c:
                        if c_row is not None:
                            st.markdown("### Corrective RAG")
                            st.markdown(f"**Answer:** {c_row['predicted_answer']}")
                            st.markdown(f"EM: `{c_row['exact_match']}` | F1: `{c_row['f1']:.3f}` | Latency: `{c_row['latency']:.2f}s`")
                            st.markdown(f"**Steps:** {c_row['steps']}")
                    st.markdown("---")
    
    with tab3:
        st.subheader("Failure Mode Analysis")
        if analysis:
            col_fm1, col_fm2 = st.columns(2)
            with col_fm1:
                st.markdown("### Naive RAG")
                fm_naive = analysis.get("naive_rag", {}).get("failure_modes", {})
                if fm_naive:
                    fm_df = pd.DataFrame([{"Mode": k.replace("_", " ").title(), "Count": v["count"], "%": v["percentage"]} for k, v in fm_naive.items()])
                    st.dataframe(fm_df, use_container_width=True)
            
            with col_fm2:
                st.markdown("### Corrective RAG")
                fm_crag = analysis.get("corrective_rag", {}).get("failure_modes", {})
                if fm_crag:
                    fm_df = pd.DataFrame([{"Mode": k.replace("_", " ").title(), "Count": v["count"], "%": v["percentage"]} for k, v in fm_crag.items()])
                    st.dataframe(fm_df, use_container_width=True)
            
            # CRAG component analysis
            crag_comp = analysis.get("corrective_rag", {}).get("component_analysis", {})
            if crag_comp:
                st.markdown("---")
                st.subheader("CRAG Component Analysis")
                col_r, col_w, col_h = st.columns(3)
                with col_r:
                    rw = crag_comp.get("rewrites", {})
                    st.metric("Query Rewrites Triggered", f"{rw.get('triggered', 0)} ({rw.get('trigger_rate', 0)}%)")
                    st.metric("Rewrite Success Rate", f"{rw.get('success_rate', 0)}%")
                with col_w:
                    ws = crag_comp.get("web_search", {})
                    st.metric("Web Searches Triggered", f"{ws.get('triggered', 0)} ({ws.get('trigger_rate', 0)}%)")
                    st.metric("Web Search Success Rate", f"{ws.get('success_rate', 0)}%")
                with col_h:
                    hr = crag_comp.get("hallucination_retries", {})
                    st.metric("Hallucination Retries", f"{hr.get('triggered', 0)} ({hr.get('trigger_rate', 0)}%)")
    
    with tab4:
        st.subheader("Statistical Significance Tests")
        if analysis and "significance_tests" in analysis:
            for metric, result in analysis["significance_tests"].items():
                with st.expander(f"Metrics: {metric.replace('_', ' ').title()}", expanded=True):
                    sig_status = "Significant" if result.get("significant") else "Not Significant"
                    st.markdown(f"**p-value:** `{result.get('p_value', 'N/A'):.6f}` ({sig_status})")
                    st.markdown(f"**Effect size:** `{result.get('effect_size', 'N/A')}`")
                    st.markdown(f"**Naive Mean:** `{result.get('naive_mean', 'N/A')}` | **CRAG Mean:** `{result.get('crag_mean', 'N/A')}`")
                    if "crag_wins" in result:
                        st.markdown(f"CRAG Wins: `{result['crag_wins']}` | Naive Wins: `{result['naive_wins']}` | Ties: `{result['ties']}`")
        else:
            st.info("Run `python src/analysis/analyzer.py` first to generate significance tests.")
    
    with tab5:
        st.subheader("Download Results")
        st.download_button("Download Naive RAG Results (CSV)", naive_df.to_csv(index=False), "naive_rag_results.csv", "text/csv")
        st.download_button("Download CRAG Results (CSV)", crag_df.to_csv(index=False), "crag_results.csv", "text/csv")
        st.download_button("Download Combined Results (CSV)", combined_df.to_csv(index=False), "combined_results.csv", "text/csv")


if __name__ == "__main__":
    main()
