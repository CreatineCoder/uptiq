"""
Phase 3.2 — LLM-as-a-Judge Evaluator.

Uses GPT-4o (a DIFFERENT, more powerful model than the agents which use GPT-4o-mini)
to score each agent's answer on 3 criteria with a 1-5 scale.
This avoids self-assessment bias.
"""
import os
import sys
import json
import logging
from typing import Dict

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """You are an expert evaluator for question-answering systems. 
You will be given a question, a gold (correct) answer, and a predicted answer from an AI system.

Score the predicted answer on 3 criteria using a 1-5 scale:

1. **Correctness** (1-5): Is the predicted answer factually correct compared to the gold answer?
   - 5: Perfectly correct
   - 3: Partially correct (some facts right, some wrong)
   - 1: Completely wrong

2. **Completeness** (1-5): Does the predicted answer cover all key aspects of the gold answer?
   - 5: Fully complete, covers everything
   - 3: Covers the main point but misses details
   - 1: Misses the core answer entirely

3. **Reasoning Quality** (1-5): Is the answer well-structured and logically coherent?
   - 5: Clear, concise, and well-reasoned
   - 3: Acceptable but could be clearer
   - 1: Incoherent or nonsensical

Respond with ONLY a valid JSON object in this exact format (no markdown, no explanation):
{{"correctness": <int>, "completeness": <int>, "reasoning_quality": <int>}}

Question: {question}
Gold Answer: {gold_answer}
Predicted Answer: {predicted_answer}

JSON Score:"""


class LLMJudge:
    """Uses GPT-4o to evaluate agent answers on correctness, completeness, and reasoning quality."""
    
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.0):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.prompt = ChatPromptTemplate.from_template(JUDGE_PROMPT)
        self.chain = self.prompt | self.llm
        self._token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    def _accumulate_tokens(self, response_msg):
        """Track token usage across all judge calls."""
        if hasattr(response_msg, "usage_metadata") and response_msg.usage_metadata:
            self._token_usage["prompt_tokens"] += response_msg.usage_metadata.get("input_tokens", 0)
            self._token_usage["completion_tokens"] += response_msg.usage_metadata.get("output_tokens", 0)
            self._token_usage["total_tokens"] += response_msg.usage_metadata.get("total_tokens", 0)
    
    def judge(self, question: str, gold_answer: str, predicted_answer: str) -> Dict:
        """
        Score a single prediction.
        
        Returns:
            dict with keys: correctness, completeness, reasoning_quality (each 1-5)
            On failure, returns default scores of 1.
        """
        logger.info(f"[LLMJudge] Judging answer for: '{question[:60]}...'")
        
        try:
            response = self.chain.invoke({
                "question": question,
                "gold_answer": gold_answer,
                "predicted_answer": predicted_answer
            })
            self._accumulate_tokens(response)
            
            # Parse the JSON response
            raw = response.content.strip()
            # Handle cases where the LLM wraps JSON in markdown code blocks
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            
            scores = json.loads(raw)
            
            # Validate scores are in range
            for key in ["correctness", "completeness", "reasoning_quality"]:
                scores[key] = max(1, min(5, int(scores[key])))
            
            logger.info(f"[LLMJudge] Scores: {scores}")
            return scores
            
        except Exception as e:
            logger.warning(f"[LLMJudge] Failed to parse judge response: {e}. Defaulting to 1s.")
            return {"correctness": 1, "completeness": 1, "reasoning_quality": 1}
    
    def get_token_usage(self) -> Dict:
        """Return accumulated token usage from all judge calls."""
        return self._token_usage.copy()
