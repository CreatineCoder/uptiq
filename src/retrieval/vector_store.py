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
        """Retrieve top_k documents via RRF-fused dense + BM25 retrieval.

        Both retrieval passes contribute to a Reciprocal Rank Fusion score.
        The raw dense relevance score of the best-scoring document is preserved
        for the Agentic RAG agent's high-confidence skip logic.
        """
        try:
            dense_docs = self.vector_store.similarity_search_with_relevance_scores(query, k=top_k)

            if not self.bm25_retriever:
                return dense_docs

            self.bm25_retriever.k = top_k
            bm25_docs = self.bm25_retriever.invoke(query)

            # ── Reciprocal Rank Fusion ──────────────────────────────────
            RRF_K = 60
            rrf_scores: dict[str, float] = {}
            doc_map: dict[str, tuple] = {}  # content → (Document, best_dense_score)

            for rank, (doc, dense_score) in enumerate(dense_docs):
                content = doc.page_content
                rrf_scores[content] = rrf_scores.get(content, 0) + 1 / (RRF_K + rank)
                doc_map[content] = (doc, dense_score)

            for rank, doc in enumerate(bm25_docs):
                content = doc.page_content
                rrf_scores[content] = rrf_scores.get(content, 0) + 1 / (RRF_K + rank)
                if content not in doc_map:
                    # BM25-only hit: assign a conservative dense score so it
                    # never triggers the high-confidence skip but still gets graded
                    doc_map[content] = (doc, 0.50)

            # Sort by fused score, return with the original dense relevance score
            sorted_contents = sorted(rrf_scores.keys(), key=lambda c: rrf_scores[c], reverse=True)
            results = [(doc_map[c][0], doc_map[c][1]) for c in sorted_contents[:top_k]]
            return results

        except Exception:
            docs = self.vector_store.similarity_search(query, k=top_k)
            return [(doc, 0.5) for doc in docs]
