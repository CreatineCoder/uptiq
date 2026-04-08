import json
import logging
import os
import sys
from tqdm import tqdm

# Add the project root to the Python path so we can import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.retrieval.chunking import TextChunker
from src.retrieval.vector_store import VectorStoreWrapper

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_index():
    data_path = "data/processed/benchmark_dataset.jsonl"
    persist_dir = "data/corpus/chroma_db_bge_small"
    
    if not os.path.exists(data_path):
        logging.error(f"Dataset not found at {data_path}. Run data_loader.py first.")
        return
        
    chunker = TextChunker(chunk_size=600, chunk_overlap=150) 
    vector_store = VectorStoreWrapper(persist_directory=persist_dir)
    
    # CLEAR EXISTING COLLECTION to prevent duplicates
    try:
        vector_store.vector_store.delete_collection()
        # Re-initialize after deletion
        vector_store = VectorStoreWrapper(persist_directory=persist_dir)
        logging.info("Cleared existing ChromaDB collection for a clean rebuild.")
    except Exception as e:
        logging.warning(f"Could not clear collection: {e}")

    unique_contexts = set()
    logging.info("Reading dataset and extracting unique contexts...")
    
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            context = item.get("gold_context", "").strip()
            # Ignore placeholder contexts (e.g., from nq_open which we didn't download full text for)
            if context and not context.startswith("Context not"):
                unique_contexts.add(context)
                
    logging.info(f"Found {len(unique_contexts)} unique contexts. Starting chunking...")
    
    all_chunks = []
    for context in tqdm(unique_contexts, desc="Chunking"):
        chunks = chunker.chunk_document(context)
        all_chunks.extend(chunks)
        
    logging.info(f"Generated {len(all_chunks)} total chunks. Indexing into ChromaDB...")
    
    # Add in batches to avoid taking too much memory at once and hitting API limits
    batch_size = 500
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Indexing Batches"):
        batch_chunks = all_chunks[i:i+batch_size]
        # Adding metadata to identify where this chunk came from
        metadatas = [{"source": "benchmark_gold_context"} for _ in batch_chunks]
        vector_store.add_texts(batch_chunks, metadatas=metadatas)
        
    logging.info("Vector store index built successfully! 🎉")

if __name__ == "__main__":
    build_index()
