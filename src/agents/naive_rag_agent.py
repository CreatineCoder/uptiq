import time
import os
import sys
import logging

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage

from src.agents.base_agent import BaseAgent, AgentResponse
from src.retrieval.vector_store import VectorStoreWrapper

logger = logging.getLogger(__name__)

class NaiveRAGAgent(BaseAgent):
    """
    A simple retrieve-then-generate pipeline (Baseline).
    It takes a query, retrieves the top K documents, and passes them to an LLM to generate an answer.
    """
    def __init__(self, vector_store: VectorStoreWrapper, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        self.vector_store = vector_store
        
        # Initialize OpenAI Chat Model
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        
        # Load prompt template
        prompt_path = os.path.join(project_root, "prompts", "naive_rag.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            template_str = f.read()
            
        self.prompt_template = ChatPromptTemplate.from_template(template_str)
        self.chain = self.prompt_template | self.llm
        
    def answer(self, query: str) -> AgentResponse:
        logger.info(f"[NaiveRAG] Starting query processing: '{query}'")
        start_time = time.time()
        
        # Step 1: Retrieve context
        logger.info(f"[NaiveRAG] Retrieving top-5 documents from ChromaDB...")
        retrieved_docs = self.vector_store.retrieve(query, top_k=5)
        context_texts = [doc.page_content for doc in retrieved_docs]
        context_str = "\n\n".join(context_texts)
        logger.info(f"[NaiveRAG] Successfully retrieved {len(context_texts)} documents.")
        
        # Step 2: Generate answer
        logger.info(f"[NaiveRAG] Passing context and query to LLM ({self.llm.model_name})...")
        response_msg: AIMessage = self.chain.invoke({
            "context": context_str,
            "question": query
        })
        
        end_time = time.time()
        latency = end_time - start_time
        logger.info(f"[NaiveRAG] Answer generated in {latency:.2f} seconds.")
        
        # Step 3: Extract token usage safely across different LangChain versions
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        if hasattr(response_msg, "usage_metadata") and response_msg.usage_metadata:
            token_usage = {
                "prompt_tokens": response_msg.usage_metadata.get("input_tokens", 0),
                "completion_tokens": response_msg.usage_metadata.get("output_tokens", 0),
                "total_tokens": response_msg.usage_metadata.get("total_tokens", 0)
            }
        elif hasattr(response_msg, "response_metadata") and "token_usage" in response_msg.response_metadata:
            usage = response_msg.response_metadata["token_usage"]
            token_usage = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            }
            
        return AgentResponse(
            answer=response_msg.content.strip(),
            retrieved_contexts=context_texts,
            latency=end_time - start_time,
            token_usage=token_usage,
            steps=["retrieve", "generate"],
            agent_type="naive_rag"
        )
