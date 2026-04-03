from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class AgentResponse:
    """Standardized output format for all agents in the benchmark."""
    answer: str
    retrieved_contexts: List[str]
    latency: float
    token_usage: Dict[str, int] # Custom format: {"prompt_tokens": x, "completion_tokens": y, "total_tokens": z}
    steps: List[str]            # Execution trace (e.g., ["retrieve", "grade", "generate"])
    agent_type: str             # E.g., 'naive_rag' or 'corrective_rag'
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseAgent(ABC):
    """Abstract base class that all benchmarked agents must implement."""
    
    @abstractmethod
    def answer(self, query: str) -> AgentResponse:
        """
        Process a user query and return a standard AgentResponse.
        
        Args:
            query (str): The user's question.
            
        Returns:
            AgentResponse: The standardized response containing the generated answer, metrics, and tracing info.
        """
        pass
