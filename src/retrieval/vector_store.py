import os
from typing import List, Dict, Any, Tuple
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

class VectorStoreWrapper:
    """Wrapper around ChromaDB for indexing and searching documents."""
    
    def __init__(self, persist_directory: str, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.persist_directory = persist_directory
        # HuggingFaceEmbeddings uses the sentence-transformers library locally
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
        # Initialize the vector store connection
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name="benchmark_corpus"
        )
        
    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] = None):
        """Add string chunks to the vector database."""
        if not texts:
            return
        self.vector_store.add_texts(texts=texts, metadatas=metadatas)
        
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """Retrieve the top_k most similar documents for a given query."""
        return self.vector_store.similarity_search(query, k=top_k)

    def retrieve_with_scores(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        """Retrieve top_k documents with normalised relevance scores in [0, 1].
        
        Higher score = more similar to query. Uses Chroma's built-in relevance
        score normalisation so callers can apply confidence thresholds directly.
        """
        try:
            return self.vector_store.similarity_search_with_relevance_scores(query, k=top_k)
        except Exception:
            # Fallback: return docs without scores (score=0.5 as neutral)
            docs = self.vector_store.similarity_search(query, k=top_k)
            return [(doc, 0.5) for doc in docs]
