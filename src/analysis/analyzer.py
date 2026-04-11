"""
Phase 5 — Analysis & Failure Mode Classification.

Classifies WHY each agent fails, runs statistical significance tests,
and performs CRAG-specific component analysis.
"""
import os
import sys
import json
import logging
import numpy as np
from typing import Dict, List
from collections import Counter

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# 1. Failure Mode Classification
# ──────────────────────────────────────────────────────────────

FAILURE_CATEGORIES = {
    "correct": "The answer is correct (EM = 1.0)",
    "retrieval_failure": "Gold answer not found in any retrieved document",
    "hallucination": "Agent fabricated info not grounded in context",
    "incomplete_answer": "Partially correct, missing key details",
    "comprehension_failure": "Answer is in context but agent got it wrong",
    "wrong_reasoning": "Multi-hop logic error (HotpotQA only)",
    "overcorrection": "CRAG rewrote a query that had correct retrieval (CRAG only)",
    "latency_spike": "CRAG entered costly retry loops (CRAG only)"
}


def classify_failure(result: Dict) -> str:
    """
    Classify why an agent failed on a given query.
    
    Args:
        result: A single result dict from naive_rag_results.jsonl or crag_results.jsonl
        
    Returns:
        One of the FAILURE_CATEGORIES keys.
    """
    metrics = result.get("metrics", {})
    em = metrics.get("exact_match", 0.0)
    f1 = metrics.get("f1", 0.0)
    recall = metrics.get("recall_at_5", 0.0)
    agent_type = result.get("agent_type", "")
    steps = result.get("steps", [])
    metadata = result.get("metadata", {})
    predicted = result.get("predicted_answer", "").lower()
    
    # Correct
    if em == 1.0:
        return "correct"
    
    # Retrieval failure — gold answer not in any retrieved context
    if recall == 0.0:
        return "retrieval_failure"
    
    # CRAG-specific: overcorrection
    if agent_type == "corrective_rag":
        rewrites = metadata.get("rewrites", 0)
        if rewrites > 0 and recall == 1.0:
            return "overcorrection"
    
    # CRAG-specific: latency spike (>3 steps beyond basic retrieve+grade+generate)
    if agent_type == "corrective_rag" and len(steps) > 6:
        return "latency_spike"
    
    # Hallucination — predicted something but answer is UNANSWERABLE or very low F1
    if predicted == "unanswerable" or f1 == 0.0:
        return "comprehension_failure"
    
    # Incomplete — partial match
    if f1 > 0.3:
        return "incomplete_answer"
    
    # Wrong reasoning for multi-hop questions
    if result.get("difficulty") == "multi-hop" and f1 < 0.3:
        return "wrong_reasoning"
    
    return "comprehension_failure"


def classify_all_results(results: List[Dict]) -> List[Dict]:
    """Classify failure modes for a list of results."""
    for r in results:
        r["failure_mode"] = classify_failure(r)
    return results


def failure_mode_summary(results: List[Dict]) -> Dict:
    """Generate a summary of failure mode distribution."""
    modes = [r.get("failure_mode", "unknown") for r in results]
    counter = Counter(modes)
    total = len(modes)
    
    summary = {}
    for mode, count in counter.most_common():
        summary[mode] = {
            "count": count,
            "percentage": round(count / total * 100, 2)
        }
    return summary


# ──────────────────────────────────────────────────────────────
# 2. Statistical Significance Testing
# ──────────────────────────────────────────────────────────────

def run_significance_tests(naive_results: List[Dict], crag_results: List[Dict]) -> Dict:
    """
    Run paired Wilcoxon signed-rank tests comparing Naive RAG vs CRAG.
    Tests whether CRAG is statistically significantly better.
    
    Returns:
        Dict with test results for each metric.
    """
    from scipy import stats
    
    # Match results by query_id
    naive_by_id = {r["query_id"]: r for r in naive_results}
    crag_by_id = {r["query_id"]: r for r in crag_results}
    
    common_ids = set(naive_by_id.keys()) & set(crag_by_id.keys())
    logger.info(f"[Analysis] Running significance tests on {len(common_ids)} paired queries.")
    
    metrics_to_test = ["exact_match", "f1", "recall_at_5", "mrr"]
    test_results = {}
    
    for metric in metrics_to_test:
        naive_scores = [naive_by_id[qid]["metrics"].get(metric, 0.0) for qid in common_ids]
        crag_scores = [crag_by_id[qid]["metrics"].get(metric, 0.0) for qid in common_ids]
        
        # Check if there's any difference at all
        diffs = [c - n for c, n in zip(crag_scores, naive_scores)]
        non_zero_diffs = [d for d in diffs if d != 0]
        
        if len(non_zero_diffs) == 0:
            test_results[metric] = {
                "statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
                "effect_size": 0.0,
                "crag_mean": np.mean(crag_scores),
                "naive_mean": np.mean(naive_scores),
                "note": "No difference between agents on this metric"
            }
            continue
        
        # Wilcoxon signed-rank test
        stat, p_value = stats.wilcoxon(naive_scores, crag_scores, alternative='two-sided')
        
        # Effect size (r = Z / sqrt(N))
        n = len(common_ids)
        z_score = stats.norm.ppf(1 - p_value / 2)
        effect_size = z_score / np.sqrt(n)
        
        test_results[metric] = {
            "statistic": float(stat),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
            "effect_size": round(float(effect_size), 4),
            "crag_mean": round(float(np.mean(crag_scores)), 4),
            "naive_mean": round(float(np.mean(naive_scores)), 4),
            "crag_wins": sum(1 for d in diffs if d > 0),
            "naive_wins": sum(1 for d in diffs if d < 0),
            "ties": sum(1 for d in diffs if d == 0),
        }
        
        sig_label = "✅ SIGNIFICANT" if p_value < 0.05 else "❌ NOT significant"
        logger.info(f"[Analysis] {metric}: p={p_value:.6f} ({sig_label}), effect_size={effect_size:.4f}")
    
    return test_results


# ──────────────────────────────────────────────────────────────
# 3. CRAG-Specific Component Analysis
# ──────────────────────────────────────────────────────────────

def crag_component_analysis(crag_results: List[Dict]) -> Dict:
    """
    Analyze CRAG-specific behavior:
    - Grading accuracy
    - Rewrite effectiveness
    - Web search hit rate
    - Step distribution
    """
    total = len(crag_results)
    if total == 0:
        return {}
    
    rewrites_triggered = 0
    web_search_triggered = 0
    hallucination_retries = 0
    step_counts = []
    
    rewrite_helped = 0  # Query was rewritten AND final answer was correct
    web_helped = 0      # Web search was used AND final answer was correct
    
    for r in crag_results:
        meta = r.get("metadata", {})
        steps = r.get("steps", [])
        em = r.get("metrics", {}).get("exact_match", 0.0)
        
        n_rewrites = meta.get("rewrites", 0)
        n_web = meta.get("web_results_used", 0)
        n_halluc = meta.get("hallucination_retries", 0)
        
        if n_rewrites > 0:
            rewrites_triggered += 1
            if em == 1.0:
                rewrite_helped += 1
        
        if n_web > 0:
            web_search_triggered += 1
            if em == 1.0:
                web_helped += 1
        
        if n_halluc > 0:
            hallucination_retries += 1
        
        step_counts.append(len(steps))
    
    analysis = {
        "total_queries": total,
        "rewrites": {
            "triggered": rewrites_triggered,
            "trigger_rate": round(rewrites_triggered / total * 100, 2),
            "success_rate": round(rewrite_helped / max(rewrites_triggered, 1) * 100, 2)
        },
        "web_search": {
            "triggered": web_search_triggered,
            "trigger_rate": round(web_search_triggered / total * 100, 2),
            "success_rate": round(web_helped / max(web_search_triggered, 1) * 100, 2)
        },
        "hallucination_retries": {
            "triggered": hallucination_retries,
            "trigger_rate": round(hallucination_retries / total * 100, 2)
        },
        "step_distribution": {
            "min": int(np.min(step_counts)),
            "max": int(np.max(step_counts)),
            "mean": round(float(np.mean(step_counts)), 2),
            "median": float(np.median(step_counts)),
            "histogram": dict(Counter(step_counts))
        }
    }
    
    logger.info(f"[Analysis] CRAG Rewrites: {rewrites_triggered}/{total} ({analysis['rewrites']['trigger_rate']}%)")
    logger.info(f"[Analysis] CRAG Web Search: {web_search_triggered}/{total} ({analysis['web_search']['trigger_rate']}%)")
    
    return analysis


# ──────────────────────────────────────────────────────────────
# 4. Run Full Analysis Pipeline
# ──────────────────────────────────────────────────────────────

def run_full_analysis(results_dir: str) -> Dict:
    """
    Run the complete analysis pipeline on saved results.
    
    Args:
        results_dir: Path to the data/results directory
        
    Returns:
        Complete analysis dict saved to analysis_report.json
    """
    logger.info(f"[Analysis] {'='*60}")
    logger.info(f"[Analysis] STARTING FULL ANALYSIS")
    logger.info(f"[Analysis] {'='*60}")
    
    # Read run_summary.json to get the latest config hash
    summary_path = os.path.join(results_dir, "run_summary.json")
    config_hash = ""
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
            config_hash = summary.get("config_hash", "")

    import glob
    naive_files = sorted(glob.glob(os.path.join(results_dir, "naive_rag_results*.jsonl")))
    crag_files = sorted(glob.glob(os.path.join(results_dir, "crag_results*.jsonl")))
    
    naive_file = naive_files[-1] if naive_files else ""
    crag_file = crag_files[-1] if crag_files else ""

    # Load results
    naive_results = _load_jsonl(naive_file) if naive_file else []
    crag_results = _load_jsonl(crag_file) if crag_file else []
    
    logger.info(f"[Analysis] Loaded {len(naive_results)} Naive RAG + {len(crag_results)} CRAG results from hash {config_hash}.")
    
    # 1. Classify failure modes
    naive_results = classify_all_results(naive_results)
    crag_results = classify_all_results(crag_results)
    
    naive_failures = failure_mode_summary(naive_results)
    crag_failures = failure_mode_summary(crag_results)
    
    logger.info(f"[Analysis] Naive RAG failure modes: {naive_failures}")
    logger.info(f"[Analysis] CRAG failure modes: {crag_failures}")
    
    # 2. Statistical significance
    significance = run_significance_tests(naive_results, crag_results)
    
    # 3. CRAG component analysis
    crag_analysis = crag_component_analysis(crag_results)
    
    # 4. Performance by dataset
    perf_by_dataset = _performance_by_dataset(naive_results, crag_results)
    
    # 5. Performance by difficulty
    perf_by_difficulty = _performance_by_difficulty(naive_results, crag_results)
    
    # Package everything
    report = {
        "naive_rag": {
            "total_queries": len(naive_results),
            "failure_modes": naive_failures,
            "avg_metrics": _avg_metrics(naive_results),
            "avg_latency": round(np.mean([r["latency"] for r in naive_results]), 4)
        },
        "corrective_rag": {
            "total_queries": len(crag_results),
            "failure_modes": crag_failures,
            "avg_metrics": _avg_metrics(crag_results),
            "avg_latency": round(np.mean([r["latency"] for r in crag_results]), 4),
            "component_analysis": crag_analysis
        },
        "significance_tests": significance,
        "performance_by_dataset": perf_by_dataset,
        "performance_by_difficulty": perf_by_difficulty
    }
    
    # Save
    report_path = os.path.join(results_dir, "analysis_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"[Analysis] Full report saved to {report_path}")
    return report


# ──────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────

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


def _avg_metrics(results: List[Dict]) -> Dict:
    if not results:
        return {}
    keys = ["exact_match", "f1", "recall_at_5", "mrr"]
    avgs = {}
    for k in keys:
        vals = [r.get("metrics", {}).get(k, 0.0) for r in results]
        avgs[f"avg_{k}"] = round(np.mean(vals), 4)
    return avgs


def _performance_by_dataset(naive: List[Dict], crag: List[Dict]) -> Dict:
    result = {}
    for dataset_name in ["nq", "hotpotqa"]:
        n_sub = [r for r in naive if r.get("dataset") == dataset_name]
        c_sub = [r for r in crag if r.get("dataset") == dataset_name]
        result[dataset_name] = {
            "naive_rag": _avg_metrics(n_sub) if n_sub else {},
            "corrective_rag": _avg_metrics(c_sub) if c_sub else {},
            "naive_count": len(n_sub),
            "crag_count": len(c_sub)
        }
    return result


def _performance_by_difficulty(naive: List[Dict], crag: List[Dict]) -> Dict:
    result = {}
    for diff in ["single-hop", "multi-hop"]:
        n_sub = [r for r in naive if r.get("difficulty") == diff]
        c_sub = [r for r in crag if r.get("difficulty") == diff]
        result[diff] = {
            "naive_rag": _avg_metrics(n_sub) if n_sub else {},
            "corrective_rag": _avg_metrics(c_sub) if c_sub else {},
            "naive_count": len(n_sub),
            "crag_count": len(c_sub)
        }
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze RAG Benchmark Results")
    parser.add_argument("--results-dir", type=str, default="data/results", help="Directory containing JSONL results")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
    run_full_analysis(args.results_dir)
