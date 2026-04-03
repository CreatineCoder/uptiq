import time
import os
import sys
import logging
from typing import TypedDict, List

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from src.agents.base_agent import BaseAgent, AgentResponse
from src.retrieval.vector_store import VectorStoreWrapper

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Define the shared state used across all LangGraph nodes
# ---------------------------------------------------------------------------
class CRAGState(TypedDict):
    question: str               # Original user question
    current_query: str          # May be rewritten
    retrieved_docs: List[str]   # Raw retrieved document texts
    relevant_docs: List[str]    # Docs that passed the relevance grader
    web_results: List[str]      # Results from web search fallback
    answer: str                 # Final generated answer
    steps: List[str]            # Execution trace
    retries: int                # Number of query-rewrite retries
    hallucination_retries: int  # Number of hallucination re-generation retries


class CorrectiveRAGAgent(BaseAgent):
    """
    A self-correcting RAG agent built with LangGraph.
    
    Flow:
        RETRIEVE → GRADE → (GENERATE | REWRITE → RETRIEVE | WEB SEARCH)
                                        ↓
                              HALLUCINATION CHECK → (DONE | RE-GENERATE)
    """
    
    MAX_REWRITE_RETRIES = 2
    MAX_HALLUCINATION_RETRIES = 1
    MIN_RELEVANT_DOCS = 2
    
    def __init__(self, vector_store: VectorStoreWrapper, model_name: str = "gpt-4o-mini", temperature: float = 0.0, tavily_api_key: str = None):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        
        # Load all prompt templates
        self.grader_prompt = self._load_prompt("crag_grader.txt")
        self.generator_prompt = self._load_prompt("crag_generator.txt")
        self.rewriter_prompt = self._load_prompt("crag_rewriter.txt")
        self.hallucination_prompt = self._load_prompt("hallucination_check.txt")
        
        # Build the LangGraph state machine
        self.graph = self._build_graph()
        
        # Track token usage across all LLM calls within a single .answer() invocation
        self._token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    def _load_prompt(self, filename: str) -> ChatPromptTemplate:
        path = os.path.join(project_root, "prompts", filename)
        with open(path, "r", encoding="utf-8") as f:
            return ChatPromptTemplate.from_template(f.read())
    
    def _accumulate_tokens(self, response_msg):
        """Extract and accumulate token usage from an LLM response."""
        if hasattr(response_msg, "usage_metadata") and response_msg.usage_metadata:
            self._token_usage["prompt_tokens"] += response_msg.usage_metadata.get("input_tokens", 0)
            self._token_usage["completion_tokens"] += response_msg.usage_metadata.get("output_tokens", 0)
            self._token_usage["total_tokens"] += response_msg.usage_metadata.get("total_tokens", 0)
        elif hasattr(response_msg, "response_metadata") and "token_usage" in response_msg.response_metadata:
            usage = response_msg.response_metadata["token_usage"]
            self._token_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
            self._token_usage["completion_tokens"] += usage.get("completion_tokens", 0)
            self._token_usage["total_tokens"] += usage.get("total_tokens", 0)

    # -----------------------------------------------------------------------
    # 2. Define each node in the graph
    # -----------------------------------------------------------------------
    def _retrieve(self, state: CRAGState) -> CRAGState:
        """Node 1: Retrieve top-5 documents from ChromaDB."""
        query = state["current_query"]
        logger.info(f"[CRAG] RETRIEVE — Searching ChromaDB for: '{query}'")
        
        docs = self.vector_store.retrieve(query, top_k=5)
        doc_texts = [doc.page_content for doc in docs]
        
        logger.info(f"[CRAG] RETRIEVE — Got {len(doc_texts)} documents.")
        state["retrieved_docs"] = doc_texts
        state["steps"].append("retrieve")
        return state
    
    def _grade_documents(self, state: CRAGState) -> CRAGState:
        """Node 2: Grade each retrieved document as relevant or irrelevant."""
        logger.info(f"[CRAG] GRADE — Grading {len(state['retrieved_docs'])} documents...")
        
        relevant = []
        chain = self.grader_prompt | self.llm
        
        for i, doc in enumerate(state["retrieved_docs"]):
            response = chain.invoke({"document": doc, "question": state["question"]})
            self._accumulate_tokens(response)
            grade = response.content.strip().lower()
            is_relevant = grade == "yes"
            logger.info(f"[CRAG] GRADE — Doc {i+1}: {'✅ Relevant' if is_relevant else '❌ Irrelevant'}")
            if is_relevant:
                relevant.append(doc)
        
        state["relevant_docs"] = relevant
        state["steps"].append(f"grade({len(relevant)}_relevant)")
        logger.info(f"[CRAG] GRADE — {len(relevant)}/{len(state['retrieved_docs'])} documents deemed relevant.")
        return state
    
    def _rewrite_query(self, state: CRAGState) -> CRAGState:
        """Node 3: Rewrite the query for better retrieval."""
        logger.info(f"[CRAG] REWRITE — Rewriting query (attempt {state['retries'] + 1})...")
        
        chain = self.rewriter_prompt | self.llm
        response = chain.invoke({"question": state["current_query"]})
        self._accumulate_tokens(response)
        
        new_query = response.content.strip()
        logger.info(f"[CRAG] REWRITE — New query: '{new_query}'")
        
        state["current_query"] = new_query
        state["retries"] += 1
        state["steps"].append("rewrite")
        return state
    
    def _web_search(self, state: CRAGState) -> CRAGState:
        """Node 4: Fallback to Tavily web search."""
        logger.info(f"[CRAG] WEB SEARCH — Searching the web for: '{state['current_query']}'")
        
        web_results = []
        if self.tavily_api_key:
            try:
                from tavily import TavilyClient
                client = TavilyClient(api_key=self.tavily_api_key)
                results = client.search(query=state["current_query"], max_results=3)
                web_results = [r["content"] for r in results.get("results", [])]
                logger.info(f"[CRAG] WEB SEARCH — Got {len(web_results)} web results.")
            except Exception as e:
                logger.warning(f"[CRAG] WEB SEARCH — Failed: {e}")
        else:
            logger.warning("[CRAG] WEB SEARCH — No TAVILY_API_KEY set. Skipping web search.")
        
        state["web_results"] = web_results
        # Add web results to relevant docs so the generator can use them
        state["relevant_docs"].extend(web_results)
        state["steps"].append("web_search")
        return state
    
    def _generate(self, state: CRAGState) -> CRAGState:
        """Node 5: Generate an answer from the relevant context."""
        context = "\n\n".join(state["relevant_docs"])
        logger.info(f"[CRAG] GENERATE — Generating answer from {len(state['relevant_docs'])} context pieces...")
        
        chain = self.generator_prompt | self.llm
        response = chain.invoke({"context": context, "question": state["question"]})
        self._accumulate_tokens(response)
        
        state["answer"] = response.content.strip()
        state["steps"].append("generate")
        logger.info(f"[CRAG] GENERATE — Answer: '{state['answer'][:100]}...'")
        return state
    
    def _hallucination_check(self, state: CRAGState) -> CRAGState:
        """Node 6: Verify the answer is grounded in context."""
        logger.info(f"[CRAG] HALLUCINATION CHECK — Verifying answer is grounded...")
        
        context = "\n\n".join(state["relevant_docs"])
        chain = self.hallucination_prompt | self.llm
        response = chain.invoke({"context": context, "answer": state["answer"]})
        self._accumulate_tokens(response)
        
        is_grounded = response.content.strip().lower() == "yes"
        logger.info(f"[CRAG] HALLUCINATION CHECK — Grounded: {'✅ Yes' if is_grounded else '❌ No'}")
        
        if is_grounded:
            state["steps"].append("hallucination_check(passed)")
        else:
            state["steps"].append("hallucination_check(failed)")
            state["hallucination_retries"] += 1
        
        return state

    # -----------------------------------------------------------------------
    # 3. Define routing logic (conditional edges)
    # -----------------------------------------------------------------------
    def _route_after_grading(self, state: CRAGState) -> str:
        """Decide what to do after grading documents."""
        relevant_count = len(state["relevant_docs"])
        
        if relevant_count >= self.MIN_RELEVANT_DOCS:
            logger.info(f"[CRAG] ROUTE — Enough relevant docs ({relevant_count}). → GENERATE")
            return "generate"
        elif state["retries"] < self.MAX_REWRITE_RETRIES:
            logger.info(f"[CRAG] ROUTE — Not enough docs ({relevant_count}). → REWRITE (retry {state['retries'] + 1})")
            return "rewrite"
        else:
            logger.info(f"[CRAG] ROUTE — Rewrites exhausted. → WEB SEARCH")
            return "web_search"
    
    def _route_after_hallucination_check(self, state: CRAGState) -> str:
        """Decide what to do after hallucination check."""
        last_step = state["steps"][-1]
        
        if "passed" in last_step:
            logger.info(f"[CRAG] ROUTE — Hallucination check passed. → DONE")
            return "done"
        elif state["hallucination_retries"] <= self.MAX_HALLUCINATION_RETRIES:
            logger.info(f"[CRAG] ROUTE — Hallucination detected. → RE-GENERATE")
            return "regenerate"
        else:
            logger.info(f"[CRAG] ROUTE — Max hallucination retries hit. → DONE (with warning)")
            return "done"

    # -----------------------------------------------------------------------
    # 4. Build the LangGraph state machine
    # -----------------------------------------------------------------------
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(CRAGState)
        
        # Add nodes
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("grade", self._grade_documents)
        workflow.add_node("rewrite", self._rewrite_query)
        workflow.add_node("web_search", self._web_search)
        workflow.add_node("generate", self._generate)
        workflow.add_node("hallucination_check", self._hallucination_check)
        
        # Set entry point
        workflow.set_entry_point("retrieve")
        
        # Define edges
        workflow.add_edge("retrieve", "grade")                      # Always grade after retrieval
        
        workflow.add_conditional_edges("grade", self._route_after_grading, {
            "generate": "generate",
            "rewrite": "rewrite",
            "web_search": "web_search"
        })
        
        workflow.add_edge("rewrite", "retrieve")                    # Rewrite loops back to retrieve
        workflow.add_edge("web_search", "generate")                 # Web search leads to generate
        workflow.add_edge("generate", "hallucination_check")        # Always check after generating
        
        workflow.add_conditional_edges("hallucination_check", self._route_after_hallucination_check, {
            "done": END,
            "regenerate": "generate"
        })
        
        return workflow.compile()

    # -----------------------------------------------------------------------
    # 5. Public interface (implements BaseAgent)
    # -----------------------------------------------------------------------
    def answer(self, query: str) -> AgentResponse:
        logger.info(f"[CRAG] ========== Processing query: '{query}' ==========")
        
        # Reset token tracker for this invocation
        self._token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        start_time = time.time()
        
        # Initialize state
        initial_state: CRAGState = {
            "question": query,
            "current_query": query,
            "retrieved_docs": [],
            "relevant_docs": [],
            "web_results": [],
            "answer": "",
            "steps": [],
            "retries": 0,
            "hallucination_retries": 0
        }
        
        # Execute the graph
        final_state = self.graph.invoke(initial_state)
        
        end_time = time.time()
        latency = end_time - start_time
        logger.info(f"[CRAG] ========== Completed in {latency:.2f}s | Steps: {final_state['steps']} ==========")
        
        return AgentResponse(
            answer=final_state["answer"],
            retrieved_contexts=final_state["relevant_docs"],
            latency=latency,
            token_usage=self._token_usage,
            steps=final_state["steps"],
            agent_type="corrective_rag",
            metadata={
                "rewrites": final_state["retries"],
                "hallucination_retries": final_state["hallucination_retries"],
                "web_results_used": len(final_state["web_results"]),
                "original_query": query,
                "final_query": final_state["current_query"]
            }
        )
