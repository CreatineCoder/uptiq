"""
Phase 4.3 — Benchmark Runner (Main Orchestrator).

Wires together agents, evaluators, and result collection into one executable pipeline.

Flow:
  1. Load config + dataset
  2. Initialize both agents
  3. For each query (with checkpoint/resume):
     a. Run Naive RAG → score → save
     b. Run CRAG → score → save
  4. Run RAGAS evaluation on all results
  5. Run LLM Judge on sampled results
  6. Compute & save aggregate metrics summary
"""
import os
import sys
import json
import time
import random
import logging
from typing import Dict, List, Any

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.pipeline.config import load_config, config_hash
from src.pipeline.result_collector import ResultCollector
from src.retrieval.vector_store import VectorStoreWrapper
from src.agents.naive_rag_agent import NaiveRAGAgent
from src.agents.corrective_rag_agent import CorrectiveRAGAgent
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.cost_tracker import CostTracker

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Orchestrates the full benchmark pipeline."""
    
    def __init__(self, config_path: str, pilot: int = None):
        """
        Args:
            config_path: Path to the YAML config file.
            pilot: If set, only run this many queries (for cost estimation).
        """
        self.config = load_config(config_path)
        self.config_id = config_hash(self.config)
        self.pilot = pilot
        
        logger.info(f"[Runner] Config loaded. Hash: {self.config_id}")
        if pilot:
            logger.info(f"[Runner] PILOT MODE: Running only {pilot} queries.")
        
        # Initialize components
        self.result_collector = ResultCollector(self.config["output"]["results_dir"])
        self.cost_tracker_naive = CostTracker()
        self.cost_tracker_crag = CostTracker()
        
        # Initialize vector store
        vs_config = self.config["vector_store"]
        self.vector_store = VectorStoreWrapper(
            persist_directory=vs_config["persist_directory"],
            embedding_model_name=vs_config["embedding_model"]
        )
        
        # Initialize agents
        self.naive_agent = None
        self.crag_agent = None
        
        agents_config = self.config["agents"]
        if agents_config["naive_rag"]["enabled"]:
            self.naive_agent = NaiveRAGAgent(
                vector_store=self.vector_store,
                model_name=agents_config["naive_rag"]["model"],
                temperature=agents_config["naive_rag"]["temperature"]
            )
            logger.info("[Runner] Naive RAG Agent initialized.")
        
        if agents_config["corrective_rag"]["enabled"]:
            self.crag_agent = CorrectiveRAGAgent(
                vector_store=self.vector_store,
                model_name=agents_config["corrective_rag"]["model"],
                temperature=agents_config["corrective_rag"]["temperature"],
                tavily_api_key=agents_config["corrective_rag"].get("tavily_api_key")
            )
            logger.info("[Runner] Corrective RAG Agent initialized.")
    
    def _load_dataset(self) -> List[Dict]:
        """Load the benchmark dataset from JSONL."""
        dataset_path = self.config["dataset"]["path"]
        data = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        
        # Apply pilot limit
        total = self.pilot if self.pilot else self.config["dataset"]["total_queries"]
        data = data[:total]
        
        logger.info(f"[Runner] Loaded {len(data)} queries from {dataset_path}")
        return data
    
    def _run_agent_on_query(self, agent, query_item: Dict, agent_type: str, cost_tracker: CostTracker) -> Dict:
        """Run a single agent on a single query and return the scored result."""
        query_id = query_item["id"]
        question = query_item["question"]
        gold_answer = query_item["gold_answer"]
        
        # Run the agent
        response = agent.answer(question)
        
        # Compute quantitative metrics (free, no API cost)
        quant_metrics = compute_all_metrics(
            prediction=response.answer,
            gold=gold_answer,
            retrieved_contexts=response.retrieved_contexts
        )
        
        # Track cost
        cost = cost_tracker.calculate_cost(response.token_usage, model=self.config["agents"][agent_type.replace("corrective_rag", "corrective_rag").replace("naive_rag", "naive_rag")]["model"])
        
        # Package result
        result = {
            "query_id": query_id,
            "agent_type": response.agent_type,
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": response.answer,
            "dataset": query_item.get("dataset", "unknown"),
            "difficulty": query_item.get("difficulty", "unknown"),
            "retrieved_contexts": response.retrieved_contexts,
            "latency": response.latency,
            "token_usage": response.token_usage,
            "cost_usd": cost,
            "steps": response.steps,
            "metrics": quant_metrics,
            "metadata": response.metadata,
            "config_hash": self.config_id
        }
        
        return result
    
    def run(self):
        """Execute the full benchmark pipeline."""
        start_time = time.time()
        dataset = self._load_dataset()
        
        checkpoint_interval = self.config["output"].get("checkpoint_interval", 10)
        
        logger.info(f"[Runner] {'='*60}")
        logger.info(f"[Runner] STARTING BENCHMARK RUN")
        logger.info(f"[Runner] Queries: {len(dataset)} | Config: {self.config_id}")
        logger.info(f"[Runner] {'='*60}")
        
        for i, query_item in enumerate(dataset):
            query_id = query_item["id"]
            progress = f"[{i+1}/{len(dataset)}]"
            
            # --- Naive RAG ---
            if self.naive_agent and not self.result_collector.is_processed(query_id, "naive_rag"):
                try:
                    logger.info(f"{progress} Running Naive RAG on: '{query_item['question'][:50]}...'")
                    result = self._run_agent_on_query(self.naive_agent, query_item, "naive_rag", self.cost_tracker_naive)
                    self.result_collector.save_result(result)
                except Exception as e:
                    logger.error(f"{progress} Naive RAG FAILED on {query_id}: {e}")
            
            # --- Corrective RAG ---
            if self.crag_agent and not self.result_collector.is_processed(query_id, "corrective_rag"):
                try:
                    logger.info(f"{progress} Running CRAG on: '{query_item['question'][:50]}...'")
                    result = self._run_agent_on_query(self.crag_agent, query_item, "corrective_rag", self.cost_tracker_crag)
                    self.result_collector.save_result(result)
                except Exception as e:
                    logger.error(f"{progress} CRAG FAILED on {query_id}: {e}")
            
            # --- Checkpoint log ---
            if (i + 1) % checkpoint_interval == 0:
                progress_info = self.result_collector.get_progress()
                elapsed = time.time() - start_time
                logger.info(f"[Runner] ---- CHECKPOINT {i+1}/{len(dataset)} ----")
                logger.info(f"[Runner] Naive: {progress_info['naive_rag_completed']} | CRAG: {progress_info['corrective_rag_completed']}")
                logger.info(f"[Runner] Elapsed: {elapsed:.1f}s | Naive Cost: ${self.cost_tracker_naive.get_summary()['total_cost_usd']:.4f} | CRAG Cost: ${self.cost_tracker_crag.get_summary()['total_cost_usd']:.4f}")
        
        # --- Run RAGAS evaluation on all results ---
        if self.config["evaluation"]["ragas"]["enabled"]:
            self._run_ragas_evaluation()
        
        # --- Run LLM Judge on sampled results ---
        if self.config["evaluation"]["llm_judge"]["enabled"]:
            self._run_llm_judge(dataset)
        
        # --- Final Summary ---
        total_time = time.time() - start_time
        logger.info(f"\n[Runner] {'='*60}")
        logger.info(f"[Runner] BENCHMARK COMPLETE")
        logger.info(f"[Runner] Total Time: {total_time:.1f}s ({total_time/60:.1f} min)")
        logger.info(f"[Runner] Naive RAG Cost: ${self.cost_tracker_naive.get_summary()['total_cost_usd']:.4f}")
        logger.info(f"[Runner] CRAG Cost: ${self.cost_tracker_crag.get_summary()['total_cost_usd']:.4f}")
        logger.info(f"[Runner] Results saved to: {self.config['output']['results_dir']}")
        logger.info(f"[Runner] {'='*60}")
        
        # Save cost + aggregate metrics summaries
        self._save_summary()
    
    def _run_llm_judge(self, dataset: List[Dict]):
        """Run LLM-as-a-Judge on a stratified sample of results."""
        from src.evaluation.llm_judge import LLMJudge
        
        judge_config = self.config["evaluation"]["llm_judge"]
        sample_size = judge_config.get("sample_size", 200)
        
        logger.info(f"[Runner] Running LLM Judge on {sample_size} sampled queries...")
        
        judge = LLMJudge(model_name=judge_config.get("model", "gpt-4o"))
        
        # Load completed results
        naive_results = self.result_collector.load_all_results("naive_rag")
        crag_results = self.result_collector.load_all_results("corrective_rag")
        
        # Stratified sampling: 50% from NQ, 50% from HotpotQA
        def stratified_sample(results, n):
            nq = [r for r in results if r.get("dataset") == "nq"]
            hotpot = [r for r in results if r.get("dataset") == "hotpotqa"]
            half = n // 2
            random.seed(self.config["experiment"].get("seed", 42))
            sampled = random.sample(nq, min(half, len(nq))) + random.sample(hotpot, min(half, len(hotpot)))
            return sampled
        
        # Judge Naive RAG
        sampled_naive = stratified_sample(naive_results, sample_size)
        judge_results_path = os.path.join(self.config["output"]["results_dir"], "judge_results.jsonl")
        
        for r in sampled_naive:
            scores = judge.judge(r["question"], r["gold_answer"], r["predicted_answer"])
            judge_entry = {
                "query_id": r["query_id"],
                "agent_type": "naive_rag",
                "judge_scores": scores
            }
            with open(judge_results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(judge_entry) + "\n")
        
        # Judge CRAG
        sampled_crag = stratified_sample(crag_results, sample_size)
        for r in sampled_crag:
            scores = judge.judge(r["question"], r["gold_answer"], r["predicted_answer"])
            judge_entry = {
                "query_id": r["query_id"],
                "agent_type": "corrective_rag",
                "judge_scores": scores
            }
            with open(judge_results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(judge_entry) + "\n")
        
        logger.info(f"[Runner] LLM Judge complete. Results saved to {judge_results_path}")
        logger.info(f"[Runner] Judge token usage: {judge.get_token_usage()}")
    
    def _run_ragas_evaluation(self):
        """Run RAGAS evaluation (Faithfulness, Answer Relevancy, Context Precision/Recall) on all results."""
        from src.evaluation.ragas_evaluator import RagasEvaluator
        
        logger.info(f"[Runner] Running RAGAS evaluation on all results...")
        ragas = RagasEvaluator()
        
        ragas_results_path = os.path.join(self.config["output"]["results_dir"], "ragas_results.jsonl")
        
        for agent_type in ["naive_rag", "corrective_rag"]:
            results = self.result_collector.load_all_results(agent_type)
            if not results:
                continue
            
            logger.info(f"[Runner] RAGAS: Evaluating {len(results)} {agent_type} results...")
            
            for r in results:
                ragas_scores = ragas.evaluate_single(
                    question=r["question"],
                    answer=r["predicted_answer"],
                    contexts=r.get("retrieved_contexts", []),
                    gold_answer=r["gold_answer"]
                )
                
                ragas_entry = {
                    "query_id": r["query_id"],
                    "agent_type": agent_type,
                    "ragas_scores": ragas_scores
                }
                with open(ragas_results_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(ragas_entry) + "\n")
            
            logger.info(f"[Runner] RAGAS: {agent_type} evaluation complete.")
        
        logger.info(f"[Runner] RAGAS results saved to {ragas_results_path}")
    
    def _compute_aggregate_metrics(self) -> Dict:
        """Compute average EM, F1, Recall@5, MRR across all queries for each agent."""
        aggregates = {}
        
        for agent_type in ["naive_rag", "corrective_rag"]:
            results = self.result_collector.load_all_results(agent_type)
            if not results:
                continue
            
            metrics_keys = ["exact_match", "f1", "recall_at_5", "mrr"]
            totals = {k: 0.0 for k in metrics_keys}
            latencies = []
            
            for r in results:
                m = r.get("metrics", {})
                for k in metrics_keys:
                    totals[k] += m.get(k, 0.0)
                latencies.append(r.get("latency", 0.0))
            
            n = len(results)
            aggregates[agent_type] = {
                "count": n,
                "avg_exact_match": round(totals["exact_match"] / n, 4),
                "avg_f1": round(totals["f1"] / n, 4),
                "avg_recall_at_5": round(totals["recall_at_5"] / n, 4),
                "avg_mrr": round(totals["mrr"] / n, 4),
                "avg_latency_s": round(sum(latencies) / n, 4),
                "total_latency_s": round(sum(latencies), 2)
            }
            
            logger.info(f"[Runner] {agent_type} Aggregate Metrics: {aggregates[agent_type]}")
        
        return aggregates
    
    def _save_summary(self):
        """Save cost, progress, and aggregate metrics summaries to disk."""
        aggregate_metrics = self._compute_aggregate_metrics()
        
        summary = {
            "config_hash": self.config_id,
            "naive_rag_cost": self.cost_tracker_naive.get_summary(),
            "crag_cost": self.cost_tracker_crag.get_summary(),
            "progress": self.result_collector.get_progress(),
            "aggregate_metrics": aggregate_metrics
        }
        
        summary_path = os.path.join(self.config["output"]["results_dir"], "run_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"[Runner] Summary saved to {summary_path}")
