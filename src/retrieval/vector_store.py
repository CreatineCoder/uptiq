import os
from typing import List, Dict, Any
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
