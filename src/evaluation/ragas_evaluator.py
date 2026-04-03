"""
Phase 3.3 — RAGAS Evaluation Wrapper.

Wraps the RAGAS library to compute:
- Faithfulness: Is the answer grounded in the retrieved context?
- Answer Relevancy: Does the answer address the question?
- Context Precision: What fraction of retrieved docs are relevant?
- Context Recall: Did retrieval capture all ground truth information?
"""
import os
import sys
import logging
from typing import List, Dict

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

logger = logging.getLogger(__name__)


class RagasEvaluator:
    """
    Wrapper around the RAGAS evaluation library.
    Converts our benchmark results into the RAGAS-expected format and runs evaluation.
    """
    
    def __init__(self):
        """Initialize RAGAS metrics. Imports are deferred to avoid issues if RAGAS is not installed."""
        try:
            from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
            from ragas import evaluate
            from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
            
            self.faithfulness = faithfulness
            self.answer_relevancy = answer_relevancy
            self.context_precision = context_precision
            self.context_recall = context_recall
            self.evaluate = evaluate
            self.SingleTurnSample = SingleTurnSample
            self.EvaluationDataset = EvaluationDataset
            self._available = True
            logger.info("[RAGAS] RAGAS library loaded successfully.")
        except ImportError as e:
            logger.warning(f"[RAGAS] RAGAS library not available: {e}. Using fallback scoring.")
            self._available = False
    
    def evaluate_single(self, question: str, answer: str, contexts: List[str], gold_answer: str) -> Dict[str, float]:
        """
        Evaluate a single query result using RAGAS metrics.
        
        Args:
            question: The original question
            answer: The agent's generated answer
            contexts: The retrieved context documents
            gold_answer: The ground truth answer
            
        Returns:
            dict with keys: faithfulness, answer_relevancy, context_precision, context_recall
        """
        if not self._available:
            return self._fallback_scores()
        
        try:
            sample = self.SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
                reference=gold_answer
            )
            dataset = self.EvaluationDataset(samples=[sample])
            
            result = self.evaluate(
                dataset=dataset,
                metrics=[self.faithfulness, self.answer_relevancy, self.context_precision, self.context_recall]
            )
            
            scores = result.to_pandas().iloc[0].to_dict()
            
            return {
                "faithfulness": float(scores.get("faithfulness", 0.0)),
                "answer_relevancy": float(scores.get("answer_relevancy", 0.0)),
                "context_precision": float(scores.get("context_precision", 0.0)),
                "context_recall": float(scores.get("context_recall", 0.0))
            }
        except Exception as e:
            logger.warning(f"[RAGAS] Evaluation failed for query: {e}. Using fallback.")
            return self._fallback_scores()
    
    def evaluate_batch(self, results: List[Dict]) -> List[Dict[str, float]]:
        """
        Evaluate a batch of results.
        
        Args:
            results: List of dicts, each with keys: question, answer, contexts, gold_answer
            
        Returns:
            List of score dicts.
        """
        if not self._available:
            logger.warning("[RAGAS] RAGAS not available. Returning fallback scores for all.")
            return [self._fallback_scores() for _ in results]
        
        try:
            samples = []
            for r in results:
                samples.append(self.SingleTurnSample(
                    user_input=r["question"],
                    response=r["answer"],
                    retrieved_contexts=r["contexts"],
                    reference=r["gold_answer"]
                ))
            
            dataset = self.EvaluationDataset(samples=samples)
            
            result = self.evaluate(
                dataset=dataset,
                metrics=[self.faithfulness, self.answer_relevancy, self.context_precision, self.context_recall]
            )
            
            df = result.to_pandas()
            scores_list = []
            for _, row in df.iterrows():
                scores_list.append({
                    "faithfulness": float(row.get("faithfulness", 0.0)),
                    "answer_relevancy": float(row.get("answer_relevancy", 0.0)),
                    "context_precision": float(row.get("context_precision", 0.0)),
                    "context_recall": float(row.get("context_recall", 0.0))
                })
            
            return scores_list
        except Exception as e:
            logger.warning(f"[RAGAS] Batch evaluation failed: {e}. Using fallback.")
            return [self._fallback_scores() for _ in results]
    
    @staticmethod
    def _fallback_scores() -> Dict[str, float]:
        """Return zero scores when RAGAS is not available or fails."""
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0
        }
