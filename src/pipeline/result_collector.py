"""
Phase 4.2 — Result Collector with Checkpointing.

Saves each query result to disk as a JSONL file.
Supports resume: if the pipeline crashes, it picks up where it left off.
"""
import os
import json
import logging
from typing import Dict, Set, List

logger = logging.getLogger(__name__)


class ResultCollector:
    """Manages saving benchmark results with checkpoint/resume support."""
    
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        self.naive_path = os.path.join(results_dir, "naive_rag_results.jsonl")
        self.crag_path = os.path.join(results_dir, "crag_results.jsonl")
        
        # Load already-processed query IDs for resume
        self.processed_naive = self._load_processed_ids(self.naive_path)
        self.processed_crag = self._load_processed_ids(self.crag_path)
        
        logger.info(f"[ResultCollector] Loaded {len(self.processed_naive)} existing Naive RAG results.")
        logger.info(f"[ResultCollector] Loaded {len(self.processed_crag)} existing CRAG results.")
    
    def _load_processed_ids(self, filepath: str) -> Set[str]:
        """Read an existing results file and return the set of already-processed query IDs."""
        ids = set()
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        ids.add(entry.get("query_id", ""))
                    except json.JSONDecodeError:
                        continue
        return ids
    
    def is_processed(self, query_id: str, agent_type: str) -> bool:
        """Check if a query has already been processed by a specific agent."""
        if agent_type == "naive_rag":
            return query_id in self.processed_naive
        elif agent_type == "corrective_rag":
            return query_id in self.processed_crag
        return False
    
    def save_result(self, result: Dict):
        """
        Append a single result to the appropriate JSONL file.
        
        Args:
            result: dict containing query_id, agent_type, and all metrics/response data.
        """
        agent_type = result.get("agent_type", "unknown")
        
        if agent_type == "naive_rag":
            filepath = self.naive_path
            self.processed_naive.add(result["query_id"])
        elif agent_type == "corrective_rag":
            filepath = self.crag_path
            self.processed_crag.add(result["query_id"])
        else:
            filepath = os.path.join(self.results_dir, f"{agent_type}_results.jsonl")
        
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
    
    def load_all_results(self, agent_type: str) -> List[Dict]:
        """Load all saved results for a given agent type."""
        if agent_type == "naive_rag":
            filepath = self.naive_path
        elif agent_type == "corrective_rag":
            filepath = self.crag_path
        else:
            filepath = os.path.join(self.results_dir, f"{agent_type}_results.jsonl")
        
        results = []
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        results.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
        return results
    
    def get_progress(self) -> Dict:
        """Return current progress stats."""
        return {
            "naive_rag_completed": len(self.processed_naive),
            "corrective_rag_completed": len(self.processed_crag),
        }
