import time
import os
import sys
import logging
from typing import TypedDict, List, Optional

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
    answer: str                 # Final generated answer
    steps: List[str]            # Execution trace
    retries: int                # Number of query-rewrite retries
    top_retrieval_score: float  # Highest relevance score from the last retrieve call


class CorrectiveRAGAgent(BaseAgent):
    """
    A self-correcting RAG agent built with LangGraph.
    
    Flow:
        EXPAND → RETRIEVE → GRADE → (GENERATE | REWRITE → RETRIEVE)
    """

    # Sub-task B: if the top retrieved doc scores above this threshold (0-1),
    # the retrieval is highly confident and we skip the expensive grading step.
    HIGH_CONFIDENCE_THRESHOLD = 0.80

    # Sub-task C: cross-encoder relevance scores are raw logits; anything above
    # this value is treated as relevant. -3.0 is a lenient default for ms-marco models 
    # to allow partial multi-hop contexts to pass.
    CROSS_ENCODER_THRESHOLD = -3.0
    
    def __init__(self, 
                 vector_store: VectorStoreWrapper, 
                 model_name: str = "gpt-4o-mini", 
                 temperature: float = 0.0, 
                 max_rewrite_retries: int = 2,
                 min_relevant_docs: int = 1):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        
        # Configuration injected from YAML
        self.max_rewrite_retries = max_rewrite_retries
        self.min_relevant_docs = min_relevant_docs
        
        # Load all prompt templates
        self.expander_prompt = self._load_prompt("query_expansion.txt")
        self.grader_prompt = self._load_prompt("crag_grader.txt")
        self.generator_prompt = self._load_prompt("crag_generator.txt")
        self.rewriter_prompt = self._load_prompt("crag_rewriter.txt")

        # Sub-task C: load a local cross-encoder re-ranker to replace LLM-based grading.
        # Falls back silently to the LLM grader if sentence-transformers is missing.
        try:
            from sentence_transformers import CrossEncoder
            self.cross_encoder: Optional[CrossEncoder] = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
            logger.info("[CRAG] CrossEncoder re-ranker loaded: ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            self.cross_encoder = None
            logger.warning(f"[CRAG] CrossEncoder unavailable ({e}). Falling back to LLM grader.")
        
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
    def _expand_query(self, state: CRAGState) -> CRAGState:
        """Node 0: Expand the initial query using a generic hypothetical answer (HyDE approach)."""
        logger.info(f"[CRAG] EXPAND — Generating hypothetical answer for: '{state['question']}'")
        
        chain = self.expander_prompt | self.llm
        response = chain.invoke({"question": state["question"]})
        self._accumulate_tokens(response)
        
        hypothetical_answer = response.content.strip()
        expanded_query = f"{state['question']}\n{hypothetical_answer}"
        
        logger.info(f"[CRAG] EXPAND — Expanded query length: {len(expanded_query)} chars")
        
        state["current_query"] = expanded_query
        state["steps"].append("expand")
        return state

    def _retrieve(self, state: CRAGState) -> CRAGState:
        """Node 1: Retrieve top-10 documents from ChromaDB with relevance scores.

        Sub-task B: if the top document's relevance score exceeds HIGH_CONFIDENCE_THRESHOLD,
        we trust the retrieval and populate relevant_docs immediately, allowing
        _grade_documents to skip its expensive grading loop entirely.
        """
        query = state["current_query"]
        logger.info(f"[CRAG] RETRIEVE — Searching ChromaDB for: '{query}'")

        # Reset relevant_docs so stale data from a previous rewrite loop is cleared
        state["relevant_docs"] = []

        # Retrieve top 15 chunks based on the expanded query
        docs_with_scores = self.vector_store.retrieve_with_scores(query, top_k=20)

        if not docs_with_scores:
            state["retrieved_docs"] = []
            state["top_retrieval_score"] = 0.0
            state["steps"].append("retrieve(no_results)")
            logger.info("[CRAG] RETRIEVE — ChromaDB returned no documents.")
            return state

        top_score = docs_with_scores[0][1]
        doc_texts = [doc.page_content for doc, _ in docs_with_scores]

        state["retrieved_docs"] = doc_texts
        state["top_retrieval_score"] = top_score

        logger.info(f"[CRAG] RETRIEVE — Got {len(doc_texts)} docs. Top similarity score: {top_score:.3f}")

        # Sub-task B: high-confidence skip — bypass the grading node entirely
        if top_score >= self.HIGH_CONFIDENCE_THRESHOLD:
            # Keep top 5 docs (already sorted by similarity, best first)
            state["relevant_docs"] = doc_texts[:5]
            state["steps"].append(f"retrieve(high_conf={top_score:.2f}_grade_skipped)")
            logger.info(
                f"[CRAG] RETRIEVE — High confidence hit (score={top_score:.3f} >= {self.HIGH_CONFIDENCE_THRESHOLD}). "
                f"Grading bypassed. Using top {len(state['relevant_docs'])} docs directly."
            )
        else:
            state["steps"].append(f"retrieve(top_score={top_score:.2f})")

        return state
    
    def _grade_documents(self, state: CRAGState) -> CRAGState:
        """Node 2: Re-rank retrieved documents and keep the most relevant ones.

        Sub-task B: if _retrieve already populated relevant_docs via the high-confidence
        skip, this node exits immediately — zero extra cost.

        Sub-task C: uses a local CrossEncoder (ms-marco-MiniLM-L-6-v2) for re-ranking
        instead of calling the LLM 10 times per query. The LLM grader is kept as a
        fallback in case the CrossEncoder is unavailable.
        """
        # High-confidence skip — relevant_docs already populated by _retrieve
        if state["relevant_docs"]:
            logger.info(
                f"[CRAG] GRADE — Skipped (high-confidence retrieve already provided "
                f"{len(state['relevant_docs'])} relevant docs)."
            )
            state["steps"].append(f"grade(skipped_{len(state['relevant_docs'])}_relevant)")
            return state

        docs = state["retrieved_docs"]
        question = state["question"]
        logger.info(f"[CRAG] GRADE — Evaluating {len(docs)} documents for: '{question[:60]}...'")

        if self.cross_encoder and docs:
            # ── Sub-task C: CrossEncoder re-ranking (local, free, fast) ──────────
            pairs = [(question, doc) for doc in docs]
            scores = self.cross_encoder.predict(pairs)          # raw logits, list[float]

            scored = sorted(zip(scores, docs), reverse=True)   # best scores first
            # Keep top 10 from the 15 retrieved
            top_k = min(10, len(scored))
            relevant = [doc for _, doc in scored[:top_k]]
            
            logger.info(
                f"[CRAG] GRADE (CrossEncoder) — Kept top {top_k}/{len(docs)} docs. "
                f"Top score: {scored[0][0]:.3f}"
            )
        else:
            # ── Fallback: LLM-based binary grader (original behaviour) ───────────
            logger.info("[CRAG] GRADE (LLM fallback) — CrossEncoder unavailable, using LLM grader.")
            relevant = []
            chain = self.grader_prompt | self.llm
            for i, doc in enumerate(docs):
                response = chain.invoke({"document": doc, "question": question})
                self._accumulate_tokens(response)
                is_relevant = response.content.strip().lower() == "yes"
                logger.info(f"[CRAG] GRADE — Doc {i+1}: {'✅ Relevant' if is_relevant else '❌ Irrelevant'}")
                if is_relevant:
                    relevant.append(doc)

        state["relevant_docs"] = relevant
        state["steps"].append(f"grade({len(relevant)}_relevant)")
        logger.info(f"[CRAG] GRADE — Final relevant docs: {len(relevant)}/{len(docs)}")
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
    
    def _generate(self, state: CRAGState) -> CRAGState:
        """Node 4: Generate an answer from the relevant context."""
        context = "\n\n".join(state["relevant_docs"])
        logger.info(f"[CRAG] GENERATE — Generating answer from {len(state['relevant_docs'])} context pieces...")
        
        chain = self.generator_prompt | self.llm
        response = chain.invoke({"context": context, "question": state["question"]})
        self._accumulate_tokens(response)
        
        state["answer"] = response.content.strip()
        state["steps"].append("generate")
        logger.info(f"[CRAG] GENERATE — Answer: '{state['answer'][:100]}...'")
        return state

    # -----------------------------------------------------------------------
    # 3. Define routing logic (conditional edges)
    # -----------------------------------------------------------------------
    def _route_after_grading(self, state: CRAGState) -> str:
        """Decide what to do after grading documents."""
        relevant_count = len(state["relevant_docs"])
        
        if relevant_count >= self.min_relevant_docs:
            logger.info(f"[CRAG] ROUTE — Enough relevant docs ({relevant_count}). → GENERATE")
            return "generate"
        elif state["retries"] < self.max_rewrite_retries:
            logger.info(f"[CRAG] ROUTE — Not enough docs ({relevant_count}). → REWRITE (retry {state['retries'] + 1})")
            return "rewrite"
        else:
            logger.info(f"[CRAG] ROUTE — Rewrites exhausted. → GENERATE (skip web search)")
            return "generate"

    # -----------------------------------------------------------------------
    # 4. Build the LangGraph state machine
    # -----------------------------------------------------------------------
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(CRAGState)
        
        # Add nodes
        workflow.add_node("expand", self._expand_query)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("grade", self._grade_documents)
        workflow.add_node("rewrite", self._rewrite_query)
        workflow.add_node("generate", self._generate)
        
        # Set entry point
        workflow.set_entry_point("expand")
        
        # Define edges
        workflow.add_edge("expand", "retrieve")                     # Expand and then retrieve
        workflow.add_edge("retrieve", "grade")                      # Always grade after retrieval
        
        # After grading: enough docs → generate, else rewrite up to max_rewrite_retries
        # then fall back to generate with whatever docs are available (web search disabled
        # for closed-corpus benchmarks where live internet results hurt accuracy)
        workflow.add_conditional_edges("grade", self._route_after_grading, {
            "generate": "generate",
            "rewrite": "rewrite",
        })
        
        workflow.add_edge("rewrite", "retrieve")                    # Rewrite loops back to retrieve
        workflow.add_edge("generate", END)                          # Generation is the last step
        
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
            "answer": "",
            "steps": [],
            "retries": 0,
            "top_retrieval_score": 0.0
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
                "original_query": query,
                "final_query": final_state["current_query"]
            }
        )
