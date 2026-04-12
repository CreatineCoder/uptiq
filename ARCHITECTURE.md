# Architecture Overview

This document outlines the technical design, system components, and architectural workflows of the RAG (Retrieval-Augmented Generation) Benchmarking Framework. The system is designed to evaluate and compare the performance of standard (Naive) RAG against an autonomous, self-correcting (Agentic) RAG.

---

## 1. High-Level System Architecture

The benchmarking framework is organized into a modular pipeline, consisting of Data Ingestion, Vector Retrieval, Agent Processing, Evaluation, and Visualization components.

```
┌─────────────────────────────────────────────────────────────────┐
│                      BENCHMARK PIPELINE                         │
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────────┐   │
│  │ Dataset   │───▶│    Query     │───▶│  Agent A: Naive RAG │   │
│  │ Loader    │    │  Dispatcher  │    └─────────┬───────────┘   │
│  │           │    │              │              │                │
│  │ • NQ      │    │              │    ┌─────────▼───────────┐   │
│  │ • HotpotQA│    │              │───▶│  Agent B: CRAG       │   │
│  └──────────┘    └──────────────┘    └─────────┬───────────┘   │
│                                                │                │
│                                     ┌──────────▼──────────┐    │
│                                     │  Result Collector    │    │
│                                     │  (with checkpoints)  │    │
│                                     └──────────┬──────────┘    │
│                                                │                │
│                              ┌─────────────────┼────────────┐  │
│                              │                 │            │   │
│                    ┌─────────▼──┐  ┌───────────▼┐  ┌───────▼─┐│
│                    │Quantitative│  │   RAGAS     │  │LLM-as-a ││
│                    │ Metrics    │  │ Evaluation  │  │  Judge   ││
│                    │ (EM, F1)   │  │(Faith/Rel)  │  │(GPT-4o) ││
│                    └─────┬──────┘  └──────┬──────┘  └────┬────┘│
│                          │               │              │      │
│                    ┌─────▼───────────────▼──────────────▼────┐ │
│                    │      Analysis & Visualization            │ │
│                    │  • Failure modes  • Statistical tests    │ │
│                    │  • 12 chart types • Streamlit dashboard  │ │
│                    └─────────────────┬────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Components:
*   **Data Pipeline:** Loads queries from SQuAD/HotpotQA benchmark datasets.
*   **Vector Store:** ChromaDB paired with HuggingFace `bge-small-en-v1.5` embeddings, additionally re-ranked by a Cross-Encoder for high-precision retrieval.
*   **Benchmarking Runner:** Orchestrates the execution of queries across both agent types concurrently, handling checkpoints and cost-tracking.
*   **Evaluation Engine:** Computes deterministic metrics (Exact Match, F1, Recall), LLM Judge evaluations, and RAGAS metrics.
*   **Analysis & Visualization:** Post-run analysis scripts dynamically categorize failure modes, which are then rendered on an interactive Streamlit dashboard.

---

## 2. Naive RAG Architecture (Baseline)

The Naive Retrieval-Augmented Generation approach represents the industry-standard baseline. It follows a strictly linear, one-pass workflow without any ability to self-correct retrieval hallucinations.

```
       ┌──────────────┐
       │              │
 ┌─────┴─────┐  ┌─────▼─────┐  ┌───────────┐  ┌──────────────┐
 │   User    │──▶ Retriever │──▶ Top-K     │──▶     LLM      │
 │ Question  │  │   Node    │  │ Documents │  │  Generator   │
 └───────────┘  └─────┬─────┘  └───────────┘  └──────┬───────┘
                      │                              │
                ┌─────▼─────┐                  ┌─────▼───────┐
                │  Vector   │                  │  Predicted  │
                │ Database  │                  │   Answer    │
                └───────────┘                  └─────────────┘
```

### Flow description:
1.  **Retrieve:** Embed the user query and fetch the top $K$ documents from ChromaDB via standard cosine similarity.
2.  **Generate:** Format the raw retrieved context and question into a prompt and feed it to the language model (e.g., `gpt-4o-mini`).
3.  **Output:** Return whatever the LLM predicts based *only* on that single pass.

*Limitations:* Vulnerable to semantic mismatches. If the vector search returns irrelevant data (e.g., wrong financial year or unrelated entity), the LLM is forced to either hallucinate or reply "Unanswerable".

---

## 3. Agentic RAG Architecture (Corrective RAG)

The Agentic RAG implementation relies on an autonomous architecture called **Corrective RAG (CRAG)**, built using **LangGraph**. It acts as a state machine that evaluates its own retrievals and iteratively corrects mistakes using query rewriting loops. Web search fallbacks are intentionally disabled to enforce strict closed-book evaluation constraints.

```text
                       ┌─────────────────┐
                       │  User Question  │
                       └────────┬────────┘
                                │
                       ┌────────▼────────┐
                       │   Expand Node   │ (HyDE — Wikipedia-style passage)
                       └────────┬────────┘
                                │
                       ┌────────▼────────┐
     ┌────────────────▶│  Retrieve Node  │ (Dual-pass: Query + HyDE → RRF merge)
     │                 └────────┬────────┘
     │                          │
     │                    Top Doc >= 0.90?
     │                   /                \
     │            [Yes] /                  \ [No]
     │                 /                    \
     │  ┌─────────────▼┐               ┌─────▼─────────┐
     │  │              │               │ Cross-Encoder │
     │  │   Generate   │               │  Grader +     │
     │  │     Node     │               │  Filter (≥0.5)│
     │  │ (multi-hop   │               └─────┬─────────┘
     │  │  synthesis)  │                     │
     │  │              │            Filtered Docs >= Min?
     │  │              │              /            \
     │  │              │       [Yes] /              \ [No]
     │  │              │            /                \
     │  │              │◀──────────┘             ┌────▼───────┐
     │  │              │                         │   Query    │
     └──│              │                         │  Rewriter  │
        └──────┬───────┘                         │(w/ context)│
               │                                 └────────────┘
       ┌───────▼───────┐
       │ Curated Final │
       │    Answer     │
       └───────────────┘
```

### Flow description:
1.  **Expand Node:** Uses HyDE (Hypothetical Document Embeddings) to generate a concise, Wikipedia-style hypothetical passage with specific entities and terms. The passage is stored separately — not concatenated with the query — to avoid diluting the embedding signal.
2.  **Retrieve Node (Dual-Pass with RRF):** Runs two independent retrieval passes — one with the original/rewritten query, one with the HyDE passage — then merges results via Reciprocal Rank Fusion (RRF). Documents appearing in both passes get a significant score boost. Fetches up to 20 unique chunks. Includes a **High Confidence Skip**: if the top document's relevance score exceeds **0.90**, the expensive grading step is bypassed (raised from 0.80 to prevent premature skips on multi-hop queries).
3.  **Document Grader Node:** A fast local cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-ranks retrieved chunks and **filters out irrelevant context** using a logit threshold of **≥ 0.5** (raised from -3.0). Only documents exceeding this threshold are kept. If all documents fall below the threshold, the single best document is preserved as a fallback.
4.  **Conditional Routing:**
    *   If top doc was highly confident or Grader finds `Filtered Docs >= Minimum`: Route directly to the **Generation Node**.
    *   If Grader finds `Filtered Docs < Minimum`: Route to the **Rewriter Node**.
5.  **Rewrite & Loop:** The agent rewrites the query, informed by a summary of what was actually retrieved (so it can understand *why* retrieval failed). The rewrite targets the original question (not a previously mangled query). It then loops back to the **Retrieve Node** for a new search pass (up to 2 retries).
6.  **Generate:** The LLM produces the final answer using a **multi-hop synthesis prompt** that explicitly instructs cross-passage reasoning — distinct from the Naive RAG prompt.

*Advantage:* The dual-pass retrieval, threshold-based filtering, and context-aware rewriting work together to dramatically reduce retrieval failures and hallucinations on multi-hop datasets. The cross-encoder threshold ensures the rewrite loop actually activates when retrieval quality is poor, while the high-confidence skip preserves low latency for easy queries.

---

## 4. Evaluation and Failure Analysis Architecture

Post-generation, the system relies on a multi-pronged evaluation strategy to prove the efficacy of the Agentic architecture.

```
┌─────────────────┐
│  Agent Outputs  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│     Performance Metrics         │
│  [Exact Match] [F1 Score]       │
│  [Recall@5]    [LLM Correct]    │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Failure Mode   │
│    Analyzer     │
└────────┬────────┘
         │
         ├──────────────────────────┐ (EM = 1.0)
         │                          ▼
         │                 [✅ Success]
         │
         ├──────────────────────────┐ (Recall = 0.0 & EM < 1.0)
         │                          ▼
         │                 [❌ Retrieval Failure]
         │
         ├──────────────────────────┐ (Recall > 0.0 & EM < 1.0)
         │                          ▼
         │                 [❌ Reasoning Failure]
         │
         └──────────────────────────┐ (CRAG Retry loops > 3)
                                    ▼
                           [⚠️ Latency Spike]
```
