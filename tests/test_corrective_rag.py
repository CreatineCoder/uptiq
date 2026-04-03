import os
import sys
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.vector_store import VectorStoreWrapper
from src.agents.corrective_rag_agent import CorrectiveRAGAgent

# Configure logging to show all CRAG step traces
logging.basicConfig(level=logging.INFO, format='%(message)s')

def load_env_manual():
    """Manually read and set .env variables."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    if not os.path.exists(env_path):
        print(f"❌ .env file not found at: {env_path}")
        return
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ[key.strip()] = value.strip()

def run_test():
    # 1. Load environment variables
    load_env_manual()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY is missing. Please add it to your .env file.")
        return

    # 2. Connect to ChromaDB
    print("\n⏳ Loading local ChromaDB corpus...")
    vector_store = VectorStoreWrapper(persist_directory="data/corpus/chroma_db")
    
    # 3. Initialize the Corrective RAG Agent
    print("🤖 Initializing Corrective RAG Agent (LangGraph)...")
    agent = CorrectiveRAGAgent(
        vector_store=vector_store,
        tavily_api_key=os.getenv("TAVILY_API_KEY")
    )
    
    # 4. Test Queries (one from HotpotQA that requires multi-hop reasoning)
    queries = [
        "Were Scott Derrickson and Ed Wood of the same nationality?",
    ]
    
    for q in queries:
        print(f"\n{'='*60}")
        print(f"❓ Query: {q}")
        print(f"{'='*60}")
        
        response = agent.answer(q)
        
        print(f"\n{'─'*60}")
        print(f"✅ GENERATED ANSWER:\n{response.answer}\n")
        print(f"⏱️  Latency: {response.latency:.2f} seconds")
        print(f"🪙  Token Usage: {response.token_usage}")
        print(f"📝  Steps Taken: {response.steps}")
        print(f"📚  Relevant Docs Used: {len(response.retrieved_contexts)}")
        print(f"🔄  Query Rewrites: {response.metadata.get('rewrites', 0)}")
        print(f"🌐  Web Results Used: {response.metadata.get('web_results_used', 0)}")
        print(f"🔍  Final Query: {response.metadata.get('final_query', q)}")
        
        # Print each relevant context snippet
        if response.retrieved_contexts:
            print(f"\n{'─'*60}")
            print("📄 RELEVANT CONTEXTS (after grading):")
            for idx, ctx in enumerate(response.retrieved_contexts, 1):
                print(f"\n  Context {idx}:\n  {ctx[:200]}")
                print(f"  {'·'*40}")

if __name__ == "__main__":
    run_test()
