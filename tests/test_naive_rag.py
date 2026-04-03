import os
import sys
import logging
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.vector_store import VectorStoreWrapper
from src.agents.naive_rag_agent import NaiveRAGAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def load_env_manual():
    """Manually read and set .env variables to avoid dotenv edge cases."""
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
    # 1. Load environment variables manually
    load_env_manual()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY is missing. Please add it to your .env file.")
        return

    # 2. Connect to our localized ChromaDB
    print("\n⏳ Loading local ChromaDB corpus...")
    vector_store = VectorStoreWrapper(persist_directory="data/corpus/chroma_db")
    
    # 3. Initialize the Agent
    print("🤖 Initializing Naive RAG Agent...")
    agent = NaiveRAGAgent(vector_store=vector_store)
    
    # 4. Test Queries
    queries = [
        "What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?", # Simple factual query
    ]
    
    for q in queries:
        print(f"\n{'='*50}\n❓ Query: {q}\n{'-'*50}")
        
        # This will trigger our logger inside the agent
        response = agent.answer(q)
        
        # Print the final standardized AgentResponse
        print(f"\n✅ GENERATED ANSWER:\n{response.answer}\n")
        print(f"⏱️ Latency: {response.latency:.2f} seconds")
        print(f"🪙 Token Usage: {response.token_usage}")
        print(f"📚 Context Docs Retrieved: {len(response.retrieved_contexts)}")
        if response.retrieved_contexts:
            for idx, context in enumerate(response.retrieved_contexts, 1):
                print(f"🔍 Context {idx}:\n{context[:200]}\n{'-'*30}")

if __name__ == "__main__":
    run_test()
