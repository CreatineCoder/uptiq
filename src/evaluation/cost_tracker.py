"""
Phase 2.4 / Phase 3 — Cost Tracker.

Calculates the dollar cost of each agent run based on OpenAI's token pricing.
Supports multiple models with configurable pricing.
"""
from typing import Dict


# Pricing per 1 million tokens (as of 2024-2025)
MODEL_PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
}


class CostTracker:
    """Tracks and calculates API costs based on token usage."""
    
    def __init__(self):
        self.total_cost = 0.0
        self.total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.calls = 0
    
    def calculate_cost(self, token_usage: Dict[str, int], model: str = "gpt-4o-mini") -> float:
        """
        Calculate the cost for a single API call.
        
        Args:
            token_usage: dict with prompt_tokens, completion_tokens, total_tokens
            model: The model name used for pricing lookup
            
        Returns:
            Cost in USD
        """
        pricing = MODEL_PRICING.get(model, MODEL_PRICING["gpt-4o-mini"])
        
        input_cost = (token_usage.get("prompt_tokens", 0) / 1_000_000) * pricing["input"]
        output_cost = (token_usage.get("completion_tokens", 0) / 1_000_000) * pricing["output"]
        
        cost = input_cost + output_cost
        
        # Accumulate
        self.total_cost += cost
        self.total_tokens["prompt_tokens"] += token_usage.get("prompt_tokens", 0)
        self.total_tokens["completion_tokens"] += token_usage.get("completion_tokens", 0)
        self.total_tokens["total_tokens"] += token_usage.get("total_tokens", 0)
        self.calls += 1
        
        return cost
    
    def get_summary(self) -> Dict:
        """Return a summary of all tracked costs."""
        return {
            "total_cost_usd": round(self.total_cost, 6),
            "total_calls": self.calls,
            "total_prompt_tokens": self.total_tokens["prompt_tokens"],
            "total_completion_tokens": self.total_tokens["completion_tokens"],
            "total_tokens": self.total_tokens["total_tokens"],
            "avg_cost_per_call": round(self.total_cost / max(self.calls, 1), 6)
        }
    
    def reset(self):
        """Reset all counters."""
        self.total_cost = 0.0
        self.total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.calls = 0
