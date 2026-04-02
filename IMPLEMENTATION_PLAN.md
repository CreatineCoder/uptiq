# 📋 Implementation Plan — Naive RAG vs Corrective Agentic RAG Benchmark

> This is the living document for the entire project. Follow this phase-by-phase during development.
> Each phase lists the exact files to create, the logic inside them, and a checklist to track progress.

---

## Progress Tracker

| Phase | Name | Status | Files Created |
|-------|------|--------|---------------|
| 0 | Repository Setup | ✅ Complete | `.gitignore`, `README.md`, `requirements.txt`, `ARCHITECTURE.md` |
| 1 | Dataset & Vector Index | ✅ Complete | `src/pipeline/data_loader.py`, `src/retrieval/chunking.py`, `src/retrieval/vector_store.py`, `evaluation/scripts/build_index.py` |
| 2 | Agent Implementation | ⬜ Next | `src/agents/base_agent.py`, `src/agents/naive_rag_agent.py`, `src/agents/corrective_rag_agent.py` |
| 3 | Evaluation Framework | ⬜ Pending | `src/evaluation/metrics.py`, `src/evaluation/llm_judge.py`, `src/evaluation/ragas_evaluator.py`, `src/evaluation/cost_tracker.py` |
| 4 | Benchmark Pipeline | ⬜ Pending | `src/pipeline/benchmark_runner.py`, `src/pipeline/result_collector.py`, `src/pipeline/config.py`, `evaluation/scripts/run_benchmark.py` |
| 5 | Analysis & Failure Modes | ⬜ Pending | `notebooks/03_results_analysis.ipynb` |
| 6 | Visualization & Dashboard | ⬜ Pending | `src/visualization/charts.py`, `src/visualization/dashboard.py`, `src/visualization/report_generator.py` |
| 7 | Report & Documentation | ⬜ Pending | `reports/benchmark_report.md`, `ARCHITECTURE.md` (finalize) |
| 8 | Demo Video | ⬜ Pending | `DEMO_VIDEO.md` (link) |

---

## ✅ Phase 0 — Repository Setup (COMPLETE)

**What was done:**
- [x] Created full directory structure (`src/`, `data/`, `evaluation/`, `configs/`, `notebooks/`, `reports/`, `tests/`)
- [x] Wrote comprehensive `README.md` with full project documentation
- [x] Created `requirements.txt` with all dependencies
- [x] Created `.gitignore` with Python and project-specific rules
- [x] Created placeholder files: `ARCHITECTURE.md`, `ANY_OTHER_DIAGRAMS.md`, `DEMO_VIDEO.md`

**Files:**
```
.gitignore
README.md
requirements.txt
ARCHITECTURE.md
ANY_OTHER_DIAGRAMS.md
DEMO_VIDEO.md
```

---

## ✅ Phase 1 — Dataset & Vector Index (COMPLETE)

**What was done:**
- [x] Downloaded HotpotQA (`fullwiki`, validation split, 500 samples)
- [x] Downloaded Natural Questions (`nq_open`, validation split, 1,000 samples)
- [x] Unified into `data/processed/benchmark_dataset.jsonl` (1,500 queries)
- [x] Built text chunking utility (`RecursiveCharacterTextSplitter`, 2000 chars, 200 overlap)
- [x] Built ChromaDB vector store wrapper with `all-MiniLM-L6-v2` embeddings
- [x] Indexed 499 unique gold contexts → 2,193 chunks into `data/corpus/chroma_db`

**Files:**
```
src/pipeline/data_loader.py          # Downloads and normalizes both datasets
src/retrieval/chunking.py            # Text splitting logic
src/retrieval/vector_store.py        # ChromaDB wrapper (index + retrieve)
evaluation/scripts/build_index.py    # Orchestrates chunking → embedding → indexing
data/processed/benchmark_dataset.jsonl   # 1,500 unified queries
data/corpus/chroma_db/               # Persisted ChromaDB index
```

**Unified Data Schema (each line in benchmark_dataset.jsonl):**
```json
{
    "id": "hotpot_xxx or nq_xxx",
    "question": "The actual question text",
    "gold_answer": "The ground truth answer",
    "gold_context": "Wikipedia text containing the answer",
    "dataset": "hotpotqa or nq",
    "difficulty": "multi-hop or single-hop",
    "supporting_facts": {"titles": [...], "sent_ids": [...]} or null
}
```

**Key Decisions Made:**
- **Why NQ + HotpotQA (not SQuAD):** SQuAD questions are near-perfect lexical matches to their passages, meaning Naive RAG would almost always succeed. NQ provides real-world ambiguity (query rewriting stress test), HotpotQA provides multi-hop reasoning (relevance grading stress test).
- **Why fullwiki gold context:** We mix all 1,500 gold contexts into one shared ChromaDB corpus. This creates natural retrieval noise — the agent must find the right 2-3 passages out of thousands.
- **Why nq_open:** The full `natural_questions` dataset requires downloading ~20GB of HTML. `nq_open` provides clean question-answer pairs without the bloat.

---

## ⬜ Phase 2 — Agent Implementation (NEXT)

**Goal:** Build both RAG agents with a shared interface so they can be swapped in the benchmark pipeline.

**Duration:** 6-8 hours

### Step 2.1 — Base Agent Interface

**File:** `src/agents/base_agent.py`

```python
# Contains:
# 1. AgentResponse dataclass — standardized output from every agent
#    Fields: answer, retrieved_contexts, latency, token_usage, steps, agent_type, metadata
#
# 2. BaseAgent abstract class — defines the .answer(query) interface
#    Both agents MUST implement this method
```

- [ ] Create `AgentResponse` dataclass with all tracking fields
- [ ] Create `BaseAgent` ABC with abstract `answer(query: str) -> AgentResponse`

### Step 2.2 — Naive RAG Agent

**File:** `src/agents/naive_rag_agent.py`

**Logic (3 steps, linear, no branching):**
```
Input: query (string)
  │
  ├─ Step 1: Retrieve top-5 documents from ChromaDB
  ├─ Step 2: Concatenate retrieved docs into a prompt context
  ├─ Step 3: Send prompt to GPT-4o-mini → Get answer
  │
Output: AgentResponse
```

**Implementation Details:**
- Uses `VectorStoreWrapper.retrieve(query, top_k=5)` from Phase 1
- Uses `ChatOpenAI(model="gpt-4o-mini", temperature=0)`
- Prompt template: simple QA format ("Given the following context, answer the question...")
- Tracks: latency, token usage, retrieved contexts

- [ ] Implement `NaiveRAGAgent` class extending `BaseAgent`
- [ ] Create prompt template `prompts/naive_rag.txt`
- [ ] Test with 5 sample queries to verify it works end-to-end

### Step 2.3 — Corrective RAG Agent (CRAG)

**File:** `src/agents/corrective_rag_agent.py`

**Logic (6 nodes, conditional branching via LangGraph):**
```
Input: query (string)
  │
  ├─ Node 1: RETRIEVE — Get top-5 docs from ChromaDB
  ├─ Node 2: GRADE DOCUMENTS — LLM grades each doc as relevant/irrelevant
  │     ├─ If ≥2 relevant docs → Go to GENERATE
  │     ├─ If 1 relevant doc → Go to WEB SEARCH (supplement)
  │     └─ If 0 relevant docs → Go to REWRITE QUERY
  ├─ Node 3: REWRITE QUERY — LLM rewrites the original query for better retrieval
  │     └─ Loop back to RETRIEVE (max 2 retries)
  ├─ Node 4: WEB SEARCH — Tavily API fetches external results
  ├─ Node 5: GENERATE — LLM generates answer from context
  ├─ Node 6: HALLUCINATION CHECK — LLM verifies answer is grounded in context
  │     ├─ If grounded → DONE
  │     └─ If hallucinated → RE-GENERATE (max 1 retry)
  │
Output: AgentResponse (with full step trace)
```

**Implementation Details:**
- Built with `langgraph.graph.StateGraph` for conditional routing
- Same embedding model and ChromaDB as Naive RAG (fair comparison)
- Same LLM (GPT-4o-mini) for generation (fair comparison)
- Additional LLM calls for: grading, rewriting, hallucination checking
- Tracks: all steps taken, which nodes were visited, retry counts

**Prompt Templates to Create:**
```
prompts/naive_rag.txt              # "Answer this question given the context..."
prompts/crag_grader.txt            # "Is this document relevant to the question? (yes/no)"
prompts/crag_generator.txt         # "Using ONLY the provided context, answer..."
prompts/crag_rewriter.txt          # "Rewrite this question for better search results..."
prompts/hallucination_check.txt    # "Is the answer supported by the context? (yes/no)"
```

- [ ] Implement `CorrectiveRAGAgent` class extending `BaseAgent`
- [ ] Create all 4 CRAG prompt templates
- [ ] Build LangGraph state machine with conditional edges
- [ ] Test with 5 sample queries (verify grading, rewriting, and fallback work)

### Step 2.4 — Shared Utilities

**File:** `src/evaluation/cost_tracker.py`

```python
# Tracks per-query costs based on OpenAI pricing:
# GPT-4o-mini: $0.15/M input tokens, $0.60/M output tokens
# GPT-4o:      $2.50/M input tokens, $10.00/M output tokens
```

- [ ] Implement `CostTracker` class

### Step 2.5 — Environment Setup

**File:** `.env.example`

```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

- [ ] Create `.env.example` with placeholder API keys
- [ ] Update `.gitignore` to exclude `.env` (already done)

### Phase 2 Checklist
- [ ] `src/agents/base_agent.py`
- [ ] `src/agents/naive_rag_agent.py`
- [ ] `src/agents/corrective_rag_agent.py`
- [ ] `src/evaluation/cost_tracker.py`
- [ ] `prompts/naive_rag.txt`
- [ ] `prompts/crag_grader.txt`
- [ ] `prompts/crag_generator.txt`
- [ ] `prompts/crag_rewriter.txt`
- [ ] `prompts/hallucination_check.txt`
- [ ] `.env.example`
- [ ] Both agents tested on 5 sample queries

---

## ⬜ Phase 3 — Evaluation Framework

**Goal:** Build all the scoring functions so we can measure how well each agent performs.

**Duration:** 4-6 hours

### Step 3.1 — Quantitative Metrics

**File:** `src/evaluation/metrics.py`

**Metrics to Implement:**
| Metric | Formula | Applied To |
|--------|---------|------------|
| **Exact Match (EM)** | `normalize(pred) == normalize(gold)` → 0 or 1 | All 1,500 queries |
| **F1 Score** | Token-level precision × recall harmonic mean | All 1,500 queries |
| **Recall@K** | Is gold answer in any of the top-K retrieved docs? | All 1,500 queries |
| **MRR** | 1 / rank of first relevant retrieved document | All 1,500 queries |

**Normalization:** lowercase, remove articles (a/an/the), remove punctuation, collapse whitespace.

- [ ] Implement `normalize_answer(text)`
- [ ] Implement `exact_match(prediction, gold)`
- [ ] Implement `f1_score(prediction, gold)`
- [ ] Implement `recall_at_k(retrieved_contexts, gold_answer, k)`
- [ ] Implement `mean_reciprocal_rank(retrieved_contexts, gold_answer)`

### Step 3.2 — LLM-as-a-Judge

**File:** `src/evaluation/llm_judge.py`

**What it does:** Uses GPT-4o (a DIFFERENT model from the agents) to score each agent's answer on 3 criteria (1-5 scale each):
1. **Correctness** — Is the answer factually right?
2. **Completeness** — Does it cover all key aspects?
3. **Reasoning Quality** — Is the logic coherent?

**Applied to:** 200 sampled queries (100 per dataset) × 2 agents = 400 judge calls.

**Why GPT-4o as judge:** Using a different, more powerful model than the agents (GPT-4o-mini) avoids self-assessment bias.

- [ ] Implement `LLMJudge` class with structured JSON output
- [ ] Implement sampling logic (stratified by dataset)
- [ ] Add retry/error handling for API failures

### Step 3.3 — RAGAS Evaluation

**File:** `src/evaluation/ragas_evaluator.py`

**Metrics from RAGAS:**
| Metric | What It Measures |
|--------|-----------------|
| **Faithfulness** | Is the answer grounded ONLY in the retrieved context? (anti-hallucination) |
| **Answer Relevancy** | Does the answer directly address the question? |
| **Context Precision** | What fraction of retrieved docs are actually relevant? |
| **Context Recall** | Did the retrieval capture all the ground truth information? |

**Applied to:** All 1,500 queries × 2 agents.

- [ ] Implement `RagasEvaluator` wrapper class
- [ ] Handle RAGAS dataset format conversion
- [ ] Test on 10 sample results

### Phase 3 Checklist
- [ ] `src/evaluation/metrics.py`
- [ ] `src/evaluation/llm_judge.py`
- [ ] `src/evaluation/ragas_evaluator.py`
- [ ] All metrics unit tested in `tests/test_evaluation.py`

---

## ⬜ Phase 4 — Benchmark Pipeline & Execution

**Goal:** Wire everything together into one executable pipeline with checkpointing.

**Duration:** 4-6 hours

### Step 4.1 — Configuration System

**File:** `src/pipeline/config.py`

- Loads YAML configs from `configs/` directory
- Supports environment variable substitution (`${OPENAI_API_KEY}`)
- Generates SHA-256 hash of config for reproducibility tracking

**Config files to create:**
```
configs/default.yaml          # Full experiment configuration
configs/evaluation.yaml       # Evaluation-specific parameters
```

- [ ] Implement YAML config loader with env var substitution
- [ ] Create `configs/default.yaml`
- [ ] Create `configs/evaluation.yaml`

### Step 4.2 — Result Collector

**File:** `src/pipeline/result_collector.py`

- Saves each query result as a JSON line in `data/results/`
- Supports **checkpointing**: if the pipeline crashes at query #743, it resumes from #744
- Tracks which queries have been processed

- [ ] Implement `ResultCollector` with checkpoint/resume logic
- [ ] Test resume functionality

### Step 4.3 — Benchmark Runner (Main Orchestrator)

**File:** `src/pipeline/benchmark_runner.py`

**Execution Flow:**
```
1. Load config
2. Load dataset (1,500 queries)
3. Initialize both agents
4. For each query (with checkpoint):
   a. Run Naive RAG → save AgentResponse
   b. Run CRAG → save AgentResponse
   c. Checkpoint to disk
5. Run quantitative evaluation (EM, F1) on all results
6. Run RAGAS evaluation on all results
7. Run LLM Judge on 200 sampled results
8. Aggregate and save final scores
```

- [ ] Implement `BenchmarkRunner` class
- [ ] Wire up agents, evaluators, and result collector

### Step 4.4 — CLI Entry Point

**File:** `evaluation/scripts/run_benchmark.py`

```bash
# Full benchmark (1,500 queries)
python evaluation/scripts/run_benchmark.py --config configs/default.yaml

# Pilot run (50 queries for cost estimation)
python evaluation/scripts/run_benchmark.py --config configs/default.yaml --pilot 50

# Evaluation only (on existing results)
python evaluation/scripts/run_benchmark.py --evaluate-only --results-dir data/results
```

- [ ] Implement CLI with Click/argparse
- [ ] Add `--pilot` flag for small test runs
- [ ] Add `--evaluate-only` flag

### Step 4.5 — Execution

- [ ] Run 10-query test to verify pipeline works end-to-end
- [ ] Run 50-query pilot to validate API cost estimates
- [ ] Execute full 1,500-query benchmark run
- [ ] Verify all results saved and checkpointed

### Phase 4 Checklist
- [ ] `src/pipeline/config.py`
- [ ] `src/pipeline/result_collector.py`
- [ ] `src/pipeline/benchmark_runner.py`
- [ ] `evaluation/scripts/run_benchmark.py`
- [ ] `configs/default.yaml`
- [ ] `configs/evaluation.yaml`
- [ ] Full benchmark run completed

---

## ⬜ Phase 5 — Analysis & Failure Modes

**Goal:** Go beyond aggregate scores. Classify WHY each agent fails.

**Duration:** 4-6 hours

### Step 5.1 — Failure Mode Classification

**7 Failure Categories:**

| # | Category | Description | Applies To |
|---|----------|-------------|------------|
| 1 | Retrieval Failure | Gold answer not in any retrieved document | Both |
| 2 | Comprehension Failure | Answer is in context but agent got it wrong | Both |
| 3 | Hallucination | Agent fabricated info not in context | Both |
| 4 | Incomplete Answer | Partially correct, missing key details | Both |
| 5 | Wrong Reasoning | Multi-hop logic error | Both (HotpotQA) |
| 6 | Overcorrection | CRAG rewrites a query that had correct retrieval | CRAG only |
| 7 | Latency Spike | CRAG enters costly retry loops | CRAG only |

**Classification Logic:**
```python
def classify_failure(item):
    if em_score == 1.0: return "correct"
    if gold_answer not in any retrieved_context: return "retrieval_failure"
    if faithfulness_score < 0.3: return "hallucination"
    if f1_score > 0.5: return "incomplete_answer"
    return "comprehension_failure"
```

- [ ] Implement failure classification function
- [ ] Apply to all 1,500 × 2 agent results

### Step 5.2 — Statistical Significance

- Paired **Wilcoxon signed-rank test** (non-parametric, no normality assumption)
- Test: "Is CRAG statistically significantly better than Naive RAG?" (p < 0.05)
- Compute effect size

- [ ] Implement statistical significance testing
- [ ] Run on EM, F1, Faithfulness, and Judge scores

### Step 5.3 — CRAG-Specific Analysis

- Grading accuracy: How often does the relevance grader correctly classify documents?
- Rewrite effectiveness: Do rewritten queries produce better retrieval?
- Web search hit rate: When fallback triggers, does it actually help?
- Step distribution: Histogram of steps taken per query

- [ ] Implement CRAG component-level analysis
- [ ] Create analysis notebook: `notebooks/03_results_analysis.ipynb`

### Phase 5 Checklist
- [ ] Failure mode classification implemented and applied
- [ ] Statistical significance tests run
- [ ] CRAG-specific analysis complete
- [ ] `notebooks/03_results_analysis.ipynb` created with full analysis

---

## ⬜ Phase 6 — Visualization & Dashboard

**Goal:** Create 12 publication-quality charts + an interactive Streamlit dashboard.

**Duration:** 4-6 hours

### Step 6.1 — Static Charts

**File:** `src/visualization/charts.py`

| # | Chart | Type | What It Shows |
|---|-------|------|---------------|
| 1 | Overall Score Comparison | Grouped bar | Side-by-side all metrics |
| 2 | Score Distribution | Violin/Box plot | Variance per metric |
| 3 | F1 Score Distribution | Histogram | Density comparison |
| 4 | Latency Distribution | Box plot | Response time comparison |
| 5 | Cost Analysis | Stacked bar | Token breakdown & costs |
| 6 | Performance by Difficulty | Grouped bar | Single-hop vs Multi-hop |
| 7 | Failure Mode Breakdown | Stacked bar | Categorized failures |
| 8 | Radar Chart | Radar/Spider | Multi-dimensional view |
| 9 | Correctness vs Latency | Scatter | Quality-speed tradeoff |
| 10 | CRAG Step Distribution | Histogram | Steps taken per query |
| 11 | Win/Loss/Tie | Donut | Per-query verdict |
| 12 | Metric Correlation | Heatmap | How metrics relate |

**Tools:** Plotly (interactive) + Matplotlib/Seaborn (publication static)

- [ ] Implement all 12 chart functions
- [ ] Save all charts to `reports/figures/` (PNG + HTML)

### Step 6.2 — Interactive Dashboard

**File:** `src/visualization/dashboard.py`

**Streamlit dashboard with:**
- Dropdown filters (dataset, difficulty, agent)
- Interactive Plotly charts
- Sample query browser (view individual responses side-by-side)
- Download raw results as CSV
- Failure mode drill-down

```bash
streamlit run src/visualization/dashboard.py
```

- [ ] Build Streamlit dashboard
- [ ] Test all interactive features

### Step 6.3 — Report Generator

**File:** `src/visualization/report_generator.py`

- Auto-generates `reports/benchmark_report.md` from the results data
- Embeds chart images
- Fills in tables with actual numbers

- [ ] Implement auto-report generator

### Phase 6 Checklist
- [ ] `src/visualization/charts.py` (all 12 charts)
- [ ] `src/visualization/dashboard.py` (Streamlit)
- [ ] `src/visualization/report_generator.py`
- [ ] All figures saved to `reports/figures/`

---

## ⬜ Phase 7 — Report & Documentation

**Goal:** Write the final publication-grade benchmark report.

**Duration:** 4-5 hours

### Report Structure (`reports/benchmark_report.md`)

```
1. Executive Summary
   - Key findings (3-4 bullet points)
   - Winner declaration with caveats

2. Experiment Setup
   2.1 Objective & Scope
   2.2 Agent Architectures (with diagrams)
   2.3 Dataset Description
   2.4 Retrieval Configuration
   2.5 Evaluation Methodology

3. Results
   3.1 Overall Performance Summary (table)
   3.2 Retrieval Quality
   3.3 Generation Quality (EM, F1, Faithfulness)
   3.4 LLM-as-a-Judge Scores
   3.5 Operational Metrics (Latency, Cost)
   3.6 Performance by Dataset
   3.7 Performance by Difficulty

4. Analysis
   4.1 Failure Mode Deep Dive
   4.2 CRAG Self-Correction Effectiveness
   4.3 Statistical Significance
   4.4 Quality-Cost-Latency Tradeoffs

5. Insights & Recommendations
   5.1 When to Use Naive RAG
   5.2 When to Use Corrective RAG
   5.3 Architectural Improvements

6. Limitations & Future Work

7. Appendix
   - Full metric tables
   - Sample outputs (5-10 examples per agent)
   - Configuration details
   - Reproducibility instructions
```

- [ ] Write `reports/benchmark_report.md`
- [ ] Finalize `ARCHITECTURE.md` with actual system diagrams
- [ ] Update `README.md` with final results and key findings
- [ ] Include 5-10 example outputs per agent in appendix

### Phase 7 Checklist
- [ ] `reports/benchmark_report.md`
- [ ] `ARCHITECTURE.md` finalized
- [ ] `README.md` updated with results

---

## ⬜ Phase 8 — Demo Video

**Goal:** Record a 3-5 minute demo video showcasing the entire project.

**Duration:** 2-3 hours

### Demo Script

| Timestamp | Content |
|-----------|---------|
| 0:00 - 0:30 | Introduction: Problem statement & approach |
| 0:30 - 1:00 | Architecture overview (show diagrams) |
| 1:00 - 1:30 | Dataset walkthrough (show data samples) |
| 1:30 - 2:30 | Live demo: Run benchmark on 5-10 queries |
| 2:30 - 3:30 | Results walkthrough: Charts & key findings |
| 3:30 - 4:00 | Interactive dashboard demo |
| 4:00 - 4:30 | Failure analysis examples |
| 4:30 - 5:00 | Conclusion & key takeaways |

- [ ] Prepare demo script
- [ ] Record with Loom or OBS
- [ ] Upload and add link to `DEMO_VIDEO.md`

### Phase 8 Checklist
- [ ] Demo video recorded
- [ ] `DEMO_VIDEO.md` updated with link

---

## 🔧 Environment & API Keys

**Required `.env` file:**
```
OPENAI_API_KEY=sk-...          # Required for both agents + LLM Judge
TAVILY_API_KEY=tvly-...        # Required for CRAG web search fallback
```

**Estimated API Costs:**

| Component | Model | Queries | Est. Cost |
|-----------|-------|---------|-----------|
| Naive RAG (generation) | GPT-4o-mini | 1,500 | ~$1.50 |
| CRAG (grading + rewriting + generation) | GPT-4o-mini | 1,500 × ~3 calls | ~$5.00 |
| CRAG (web search) | Tavily | ~300 (20% fallback) | Free tier |
| RAGAS evaluation | GPT-4o-mini | 3,000 | ~$3.00 |
| LLM-as-a-Judge | GPT-4o | 400 | ~$8.00 |
| **Total** | | | **~$17.50** |

---

## 📂 Final File Map

```
uptiq/
├── src/
│   ├── agents/
│   │   ├── base_agent.py                  # Phase 2 — Abstract interface + AgentResponse
│   │   ├── naive_rag_agent.py             # Phase 2 — Linear retrieve → generate
│   │   └── corrective_rag_agent.py        # Phase 2 — LangGraph CRAG
│   ├── retrieval/
│   │   ├── chunking.py                    # Phase 1 ✅ — Text splitting
│   │   └── vector_store.py                # Phase 1 ✅ — ChromaDB wrapper
│   ├── evaluation/
│   │   ├── metrics.py                     # Phase 3 — EM, F1, Recall@K, MRR
│   │   ├── llm_judge.py                   # Phase 3 — GPT-4o judge
│   │   ├── ragas_evaluator.py             # Phase 3 — RAGAS wrapper
│   │   └── cost_tracker.py                # Phase 2 — Token/dollar tracking
│   ├── pipeline/
│   │   ├── data_loader.py                 # Phase 1 ✅ — Dataset download
│   │   ├── benchmark_runner.py            # Phase 4 — Main orchestrator
│   │   ├── result_collector.py            # Phase 4 — Checkpointed result storage
│   │   └── config.py                      # Phase 4 — YAML config loader
│   └── visualization/
│       ├── charts.py                      # Phase 6 — 12 chart types
│       ├── dashboard.py                   # Phase 6 — Streamlit app
│       └── report_generator.py            # Phase 6 — Auto-report
├── data/
│   ├── processed/benchmark_dataset.jsonl  # Phase 1 ✅ — 1,500 queries
│   ├── corpus/chroma_db/                  # Phase 1 ✅ — Vector index
│   └── results/                           # Phase 4 — Benchmark outputs
├── configs/
│   ├── default.yaml                       # Phase 4
│   └── evaluation.yaml                    # Phase 4
├── prompts/
│   ├── naive_rag.txt                      # Phase 2
│   ├── crag_grader.txt                    # Phase 2
│   ├── crag_generator.txt                 # Phase 2
│   ├── crag_rewriter.txt                  # Phase 2
│   └── hallucination_check.txt            # Phase 2
├── evaluation/scripts/
│   ├── build_index.py                     # Phase 1 ✅
│   └── run_benchmark.py                   # Phase 4
├── reports/
│   ├── benchmark_report.md                # Phase 7
│   └── figures/                           # Phase 6
├── notebooks/
│   └── 03_results_analysis.ipynb          # Phase 5
├── tests/
│   ├── test_agents.py                     # Phase 2
│   └── test_evaluation.py                 # Phase 3
├── README.md                              # Phase 0 ✅
├── IMPLEMENTATION_PLAN.md                 # This file
├── ARCHITECTURE.md                        # Phase 7
├── DEMO_VIDEO.md                          # Phase 8
├── requirements.txt                       # Phase 0 ✅
├── .env.example                           # Phase 2
└── .gitignore                             # Phase 0 ✅
```
