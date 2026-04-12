import time
import os
import sys
import logging
from typing import TypedDict, List, Optional

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
class AgenticRAGState(TypedDict):
    question: str               # Original user question
    current_query: str          # May be rewritten
    hyde_passage: str           # HyDE hypothetical passage (used for dual-pass retrieval)
    retrieved_docs: List[str]   # Raw retrieved document texts
    relevant_docs: List[str]    # Docs that passed the relevance grader
    answer: str                 # Final generated answer
    steps: List[str]            # Execution trace
    retries: int                # Number of query-rewrite retries
    top_retrieval_score: float  # Highest relevance score from the last retrieve call


class AgenticRAGAgent(BaseAgent):
    """
    A self-correcting RAG agent built with LangGraph.

    Flow:
        EXPAND → RETRIEVE → GRADE → (GENERATE | REWRITE → RETRIEVE)
    """

    # If the top retrieved doc scores above this threshold (0-1),
    # the retrieval is highly confident and we skip the expensive grading step.
    # Set to 0.90 — only bypass grading when retrieval is near-certain,
    # which is critical for multi-hop queries that need diverse evidence.
    HIGH_CONFIDENCE_THRESHOLD = 0.90

    # Cross-encoder relevance scores are raw logits (ms-marco-MiniLM).
    # Typical score distribution: irrelevant docs score < 0, relevant docs score > 1.
    # A threshold of 0.5 filters out clearly irrelevant docs while keeping
    # partial matches useful for multi-hop reasoning.
    CROSS_ENCODER_THRESHOLD = 0.5

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
        self.grader_prompt   = self._load_prompt("agentic_rag_grader.txt")
        self.generator_prompt = self._load_prompt("agentic_rag_generator.txt")
        self.rewriter_prompt  = self._load_prompt("agentic_rag_rewriter.txt")

        # Load a local cross-encoder re-ranker to replace LLM-based grading.
        # Falls back silently to the LLM grader if sentence-transformers is missing.
        try:
            from sentence_transformers import CrossEncoder
            self.cross_encoder: Optional[CrossEncoder] = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
            logger.info("[AgenticRAG] CrossEncoder re-ranker loaded: ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            self.cross_encoder = None
            logger.warning(f"[AgenticRAG] CrossEncoder unavailable ({e}). Falling back to LLM grader.")

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
            self._token_usage["prompt_tokens"]    += response_msg.usage_metadata.get("input_tokens", 0)
            self._token_usage["completion_tokens"] += response_msg.usage_metadata.get("output_tokens", 0)
            self._token_usage["total_tokens"]      += response_msg.usage_metadata.get("total_tokens", 0)
        elif hasattr(response_msg, "response_metadata") and "token_usage" in response_msg.response_metadata:
            usage = response_msg.response_metadata["token_usage"]
            self._token_usage["prompt_tokens"]    += usage.get("prompt_tokens", 0)
            self._token_usage["completion_tokens"] += usage.get("completion_tokens", 0)
            self._token_usage["total_tokens"]      += usage.get("total_tokens", 0)

    # -----------------------------------------------------------------------
    # 2. Define each node in the graph
    # -----------------------------------------------------------------------
    def _expand_query(self, state: AgenticRAGState) -> AgenticRAGState:
        """Node 0: Generate a HyDE hypothetical passage for dual-pass retrieval.

        Instead of concatenating the passage with the query (which dilutes the
        embedding), we store it separately. The retrieve node will run two
        retrieval passes — one with the original query, one with the HyDE
        passage — and merge results via Reciprocal Rank Fusion (RRF).
        """
        logger.info(f"[AgenticRAG] EXPAND — Generating HyDE passage for: '{state['question']}'")

        chain = self.expander_prompt | self.llm
        response = chain.invoke({"question": state["question"]})
        self._accumulate_tokens(response)

        hyde_passage = response.content.strip()
        logger.info(f"[AgenticRAG] EXPAND — HyDE passage ({len(hyde_passage)} chars): '{hyde_passage[:120]}...'")

        # Store HyDE passage separately; current_query stays as the original question
        state["hyde_passage"] = hyde_passage
        state["steps"].append("expand")
        return state

    def _retrieve(self, state: AgenticRAGState) -> AgenticRAGState:
        """Node 1: Dual-pass retrieval with RRF merge.

        Pass 1 — retrieve using the current query (original or rewritten).
        Pass 2 — retrieve using the HyDE passage (if available).
        Merge both result sets via Reciprocal Rank Fusion so that documents
        appearing in both passes get a significant boost.

        If the top document's relevance score exceeds HIGH_CONFIDENCE_THRESHOLD,
        we trust the retrieval and populate relevant_docs immediately,
        allowing _grade_documents to skip.
        """
        query = state["current_query"]
        hyde  = state.get("hyde_passage", "")
        logger.info(f"[AgenticRAG] RETRIEVE — Dual-pass search. Query: '{query[:80]}...'")

        # Reset relevant_docs so stale data from a previous rewrite loop is cleared
        state["relevant_docs"] = []

        # ── Pass 1: original / rewritten query ──────────────────────────────
        pass1 = self.vector_store.retrieve_with_scores(query, top_k=15)

        # ── Pass 2: HyDE passage (skip if empty, e.g. after a rewrite) ──────
        pass2 = []
        if hyde:
            pass2 = self.vector_store.retrieve_with_scores(hyde, top_k=15)

        # ── Reciprocal Rank Fusion ──────────────────────────────────────────
        RRF_K = 60  # standard RRF constant
        doc_scores: dict[str, float] = {}
        doc_objects: dict[str, tuple] = {}  # content → (Document, best_raw_score)

        for rank, (doc, raw_score) in enumerate(pass1):
            content = doc.page_content
            doc_scores[content] = doc_scores.get(content, 0) + 1 / (RRF_K + rank)
            if content not in doc_objects or raw_score > doc_objects[content][1]:
                doc_objects[content] = (doc, raw_score)

        for rank, (doc, raw_score) in enumerate(pass2):
            content = doc.page_content
            doc_scores[content] = doc_scores.get(content, 0) + 1 / (RRF_K + rank)
            if content not in doc_objects or raw_score > doc_objects[content][1]:
                doc_objects[content] = (doc, raw_score)

        # Sort by fused score, keep top 20
        sorted_contents = sorted(doc_scores.keys(), key=lambda c: doc_scores[c], reverse=True)
        top_contents = sorted_contents[:20]

        if not top_contents:
            state["retrieved_docs"] = []
            state["top_retrieval_score"] = 0.0
            state["steps"].append("retrieve(no_results)")
            logger.info("[AgenticRAG] RETRIEVE — No documents from either pass.")
            return state

        # Use the best raw relevance score (from pass1) for the high-confidence check
        top_score = doc_objects[top_contents[0]][1]
        doc_texts  = top_contents

        state["retrieved_docs"]       = doc_texts
        state["top_retrieval_score"]  = top_score

        logger.info(
            f"[AgenticRAG] RETRIEVE — Merged {len(pass1)}+{len(pass2)} → {len(doc_texts)} unique docs. "
            f"Top score: {top_score:.3f}"
        )

        # High-confidence skip — bypass grading only when near-certain
        if top_score >= self.HIGH_CONFIDENCE_THRESHOLD:
            state["relevant_docs"] = doc_texts[:5]
            state["steps"].append(f"retrieve(high_conf={top_score:.2f}_grade_skipped)")
            logger.info(
                f"[AgenticRAG] RETRIEVE — High confidence hit (score={top_score:.3f} >= "
                f"{self.HIGH_CONFIDENCE_THRESHOLD}). Grading bypassed."
            )
        else:
            state["steps"].append(f"retrieve(top_score={top_score:.2f})")

        return state

    def _grade_documents(self, state: AgenticRAGState) -> AgenticRAGState:
        """Node 2: Re-rank retrieved documents and keep the most relevant ones.

        If _retrieve already populated relevant_docs via the high-confidence
        skip, this node exits immediately — zero extra cost.

        Uses a local CrossEncoder (ms-marco-MiniLM-L-6-v2) for re-ranking
        instead of calling the LLM 10 times per query. The LLM grader is kept
        as a fallback in case the CrossEncoder is unavailable.
        """
        # High-confidence skip — relevant_docs already populated by _retrieve
        if state["relevant_docs"]:
            logger.info(
                f"[AgenticRAG] GRADE — Skipped (high-confidence retrieve already provided "
                f"{len(state['relevant_docs'])} relevant docs)."
            )
            state["steps"].append(f"grade(skipped_{len(state['relevant_docs'])}_relevant)")
            return state

        docs     = state["retrieved_docs"]
        question = state["question"]
        logger.info(f"[AgenticRAG] GRADE — Evaluating {len(docs)} documents for: '{question[:60]}...'")

        if self.cross_encoder and docs:
            # ── CrossEncoder re-ranking with threshold filtering ─────────────
            pairs  = [(question, doc) for doc in docs]
            scores = self.cross_encoder.predict(pairs)          # raw logits, list[float]
            scored = sorted(zip(scores, docs), reverse=True)    # best scores first

            # Filter by threshold — only keep genuinely relevant docs
            relevant = [doc for score, doc in scored if score >= self.CROSS_ENCODER_THRESHOLD]

            # If nothing passes the threshold, leave relevant empty so routing
            # triggers a rewrite. The best doc is still available in
            # retrieved_docs for a last-resort generate after retries exhaust.
            if not relevant:
                logger.info(
                    f"[AgenticRAG] GRADE (CrossEncoder) — All docs below threshold "
                    f"({self.CROSS_ENCODER_THRESHOLD}). Best score: "
                    f"{scored[0][0]:.3f}. Will trigger rewrite."
                )
            else:
                logger.info(
                    f"[AgenticRAG] GRADE (CrossEncoder) — Kept {len(relevant)}/{len(docs)} docs "
                    f"above threshold {self.CROSS_ENCODER_THRESHOLD}. "
                    f"Top score: {scored[0][0]:.3f}"
                )
        else:
            # ── Fallback: LLM-based binary grader ───────────────────────────
            logger.info("[AgenticRAG] GRADE (LLM fallback) — CrossEncoder unavailable, using LLM grader.")
            relevant = []
            chain = self.grader_prompt | self.llm
            for i, doc in enumerate(docs):
                response = chain.invoke({"document": doc, "question": question})
                self._accumulate_tokens(response)
                is_relevant = response.content.strip().lower() == "yes"
                logger.info(f"[AgenticRAG] GRADE — Doc {i+1}: {'✅ Relevant' if is_relevant else '❌ Irrelevant'}")
                if is_relevant:
                    relevant.append(doc)

        state["relevant_docs"] = relevant
        state["steps"].append(f"grade({len(relevant)}_relevant)")
        logger.info(f"[AgenticRAG] GRADE — Final relevant docs: {len(relevant)}/{len(docs)}")
        return state

    def _rewrite_query(self, state: AgenticRAGState) -> AgenticRAGState:
        """Node 3: Rewrite the query for better retrieval.

        Passes a summary of the top retrieved docs so the rewriter can
        understand WHY retrieval failed and adjust accordingly.
        """
        logger.info(f"[AgenticRAG] REWRITE — Rewriting query (attempt {state['retries'] + 1})...")

        top_docs = state["retrieved_docs"][:3]
        retrieval_summary = "\n".join(
            f"- {doc[:150]}..." for doc in top_docs
        ) if top_docs else "(no documents retrieved)"

        chain    = self.rewriter_prompt | self.llm
        response = chain.invoke({
            "question": state["question"],  # Use original question, not the mangled current_query
            "retrieval_summary": retrieval_summary,
        })
        self._accumulate_tokens(response)

        new_query = response.content.strip()
        logger.info(f"[AgenticRAG] REWRITE — New query: '{new_query}'")

        state["current_query"] = new_query
        # Clear HyDE passage so the next retrieve uses only the rewritten query
        state["hyde_passage"] = ""
        state["retries"] += 1
        state["steps"].append("rewrite")
        return state

    def _generate(self, state: AgenticRAGState) -> AgenticRAGState:
        """Node 4: Generate an answer from the relevant context."""
        docs = state["relevant_docs"]
        if not docs:
            # Fallback: retries exhausted, use top retrieved docs as best effort
            docs = state["retrieved_docs"][:5]
            state["relevant_docs"] = docs
            logger.info(f"[AgenticRAG] GENERATE — No relevant docs; falling back to top {len(docs)} retrieved docs.")
        context = "\n\n".join(docs)
        logger.info(f"[AgenticRAG] GENERATE — Generating answer from {len(docs)} context pieces...")

        chain    = self.generator_prompt | self.llm
        response = chain.invoke({"context": context, "question": state["question"]})
        self._accumulate_tokens(response)

        state["answer"] = response.content.strip()
        state["steps"].append("generate")
        logger.info(f"[AgenticRAG] GENERATE — Answer: '{state['answer'][:100]}...'")
        return state

    # -----------------------------------------------------------------------
    # 3. Define routing logic (conditional edges)
    # -----------------------------------------------------------------------
    def _route_after_grading(self, state: AgenticRAGState) -> str:
        """Decide what to do after grading documents."""
        relevant_count = len(state["relevant_docs"])

        if relevant_count >= self.min_relevant_docs:
            logger.info(f"[AgenticRAG] ROUTE — Enough relevant docs ({relevant_count}). → GENERATE")
            return "generate"
        elif state["retries"] < self.max_rewrite_retries:
            logger.info(f"[AgenticRAG] ROUTE — Not enough docs ({relevant_count}). → REWRITE (retry {state['retries'] + 1})")
            return "rewrite"
        else:
            logger.info(f"[AgenticRAG] ROUTE — Rewrites exhausted. → GENERATE")
            return "generate"

    # -----------------------------------------------------------------------
    # 4. Build the LangGraph state machine
    # -----------------------------------------------------------------------
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgenticRAGState)

        workflow.add_node("expand",   self._expand_query)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("grade",    self._grade_documents)
        workflow.add_node("rewrite",  self._rewrite_query)
        workflow.add_node("generate", self._generate)

        workflow.set_entry_point("expand")

        workflow.add_edge("expand",   "retrieve")
        workflow.add_edge("retrieve", "grade")
        workflow.add_conditional_edges("grade", self._route_after_grading, {
            "generate": "generate",
            "rewrite":  "rewrite",
        })
        workflow.add_edge("rewrite",  "retrieve")
        workflow.add_edge("generate", END)

        return workflow.compile()

    # -----------------------------------------------------------------------
    # 5. Public interface (implements BaseAgent)
    # -----------------------------------------------------------------------
    def answer(self, query: str) -> AgentResponse:
        logger.info(f"[AgenticRAG] ========== Processing query: '{query}' ==========")

        self._token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        start_time = time.time()

        initial_state: AgenticRAGState = {
            "question":           query,
            "current_query":      query,
            "hyde_passage":       "",
            "retrieved_docs":     [],
            "relevant_docs":      [],
            "answer":             "",
            "steps":              [],
            "retries":            0,
            "top_retrieval_score": 0.0,
        }

        final_state = self.graph.invoke(initial_state)

        latency = time.time() - start_time
        logger.info(f"[AgenticRAG] ========== Completed in {latency:.2f}s | Steps: {final_state['steps']} ==========")

        # Use the full retrieved pool for recall/MRR evaluation (retrieval quality),
        # not just the filtered relevant_docs (which measures grading quality).
        eval_contexts = final_state["retrieved_docs"] or final_state["relevant_docs"]

        return AgentResponse(
            answer=final_state["answer"],
            retrieved_contexts=eval_contexts,
            latency=latency,
            token_usage=self._token_usage,
            steps=final_state["steps"],
            agent_type="agentic_rag",
            metadata={
                "rewrites":      final_state["retries"],
                "original_query": query,
                "final_query":   final_state["current_query"],
            }
        )
