import os
import logging
from typing import List, Dict, Any, Tuple
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class VectorStoreWrapper:
    """Wrapper around ChromaDB and BM25 for indexing and hybrid searching documents."""
    
    def __init__(self, persist_directory: str, embedding_model_name: str = "BAAI/bge-small-en-v1.5"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Use a new collection name for the BGE model to prevent dimension mismatch
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name="benchmark_corpus_bge_small"
        )
        
        self.bm25_retriever = None
        self._init_bm25()
        
    def _init_bm25(self):
        """Initialize the BM25 retriever using texts currently in the database."""
        try:
            data = self.vector_store.get()
            texts = data.get("documents", [])
            if texts:
                from langchain_community.retrievers import BM25Retriever
                self.bm25_retriever = BM25Retriever.from_texts(texts)
                logger.info(f"[VectorStore] BM25 completely initialized with {len(texts)} chunks.")
        except Exception as e:
            logger.warning(f"[VectorStore] Could not initialize BM25: {e}")

    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] = None):
        """Add string chunks to the vector database and re-initialize BM25."""
        if not texts:
            return
        self.vector_store.add_texts(texts=texts, metadatas=metadatas)
        self._init_bm25()
        
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """Hybrid retrieval for standard (Naive RAG) usage."""
        if self.bm25_retriever:
            dense_docs = self.vector_store.similarity_search(query, k=top_k)
            
            self.bm25_retriever.k = top_k
            bm25_docs = self.bm25_retriever.invoke(query)
            
            # Reciprocal Rank Fusion (RRF)
            doc_scores = {}
            for rank, doc in enumerate(dense_docs):
                doc_scores[doc.page_content] = doc_scores.get(doc.page_content, 0) + 1 / (60 + rank)
            for rank, doc in enumerate(bm25_docs):
                doc_scores[doc.page_content] = doc_scores.get(doc.page_content, 0) + 1 / (60 + rank)
                
            all_docs = {d.page_content: d for d in dense_docs + bm25_docs}
            sorted_contents = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
            return [all_docs[c] for c in sorted_contents[:top_k]]
            
        return self.vector_store.similarity_search(query, k=top_k)

    def retrieve_with_scores(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        """Retrieve top_k documents with normalised relevance scores.
        
        Performs dense search for precise scoring, then injects BM25 hits
        with a fallback score so they don't get wrongly flagged as high-confidence
        but still get graded.
        """
        try:
            dense_docs = self.vector_store.similarity_search_with_relevance_scores(query, k=top_k)
            
            if self.bm25_retriever:
                self.bm25_retriever.k = top_k
                bm25_docs = self.bm25_retriever.invoke(query)
                
                seen_content = set(doc.page_content for doc, _ in dense_docs)
                blended = list(dense_docs)
                
                for doc in bm25_docs:
                    if doc.page_content not in seen_content:
                        # Give pure BM25 hits a conservative relevance score (0.75) 
                        # This prevents auto-skipping the grade node, forcing evaluation
                        blended.append((doc, 0.75))
                        seen_content.add(doc.page_content)
                        
                blended.sort(key=lambda x: x[1], reverse=True)
                return blended[:top_k]
                
            return dense_docs
        except Exception:
            docs = self.vector_store.similarity_search(query, k=top_k)
            return [(doc, 0.5) for doc in docs]
