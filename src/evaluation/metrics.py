"""
Phase 3.1 — Quantitative Metrics for RAG Evaluation.

Implements: Exact Match (EM), F1 Score, Recall@K, Mean Reciprocal Rank (MRR).
All metrics follow the standard SQuAD evaluation protocol for normalization.
"""
import re
import string
from collections import Counter
from typing import List


def normalize_answer(text: str) -> str:
    """
    Normalize an answer string for fair comparison.
    Steps: lowercase → remove articles → remove punctuation → collapse whitespace.
    """
    text = text.lower()
    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Collapse whitespace
    text = ' '.join(text.split())
    return text.strip()


def exact_match(prediction: str, gold: str) -> float:
    """
    Returns 1.0 if the normalized prediction exactly matches the normalized gold answer, else 0.0.
    """
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def f1_score(prediction: str, gold: str) -> float:
    """
    Computes token-level F1 score between prediction and gold answer.
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    
    if num_common == 0:
        return 0.0
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    
    return 2 * (precision * recall) / (precision + recall)


def recall_at_k(retrieved_contexts: List[str], gold_answer: str, k: int = 5) -> float:
    """
    Returns 1.0 if the gold answer appears (normalized) within any of the top-K retrieved contexts.
    This checks whether the retrieval stage successfully found the answer.
    """
    normalized_gold = normalize_answer(gold_answer)
    
    for context in retrieved_contexts[:k]:
        if normalized_gold in normalize_answer(context):
            return 1.0
    return 0.0


def mean_reciprocal_rank(retrieved_contexts: List[str], gold_answer: str) -> float:
    """
    Returns 1/rank of the first retrieved context that contains the gold answer.
    If the answer isn't found in any context, returns 0.0.
    """
    normalized_gold = normalize_answer(gold_answer)
    
    for i, context in enumerate(retrieved_contexts):
        if normalized_gold in normalize_answer(context):
            return 1.0 / (i + 1)
    return 0.0


def compute_all_metrics(prediction: str, gold: str, retrieved_contexts: List[str]) -> dict:
    """
    Convenience function to compute all metrics at once for a single query.
    
    Returns:
        dict with keys: exact_match, f1, recall_at_5, mrr
    """
    return {
        "exact_match": exact_match(prediction, gold),
        "f1": f1_score(prediction, gold),
        "recall_at_5": recall_at_k(retrieved_contexts, gold, k=5),
        "mrr": mean_reciprocal_rank(retrieved_contexts, gold)
    }
