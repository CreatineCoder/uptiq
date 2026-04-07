from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextChunker:
    """
    Handles splitting full documents into smaller chunks for vector database indexing.
    """
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 150):
        # Optimized for performance: chunk_size=600 (~150 tokens) and overlap=150 (~35 tokens).
        # Smaller chunks allow for more precise retrieval and higher top_k diversity.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def chunk_document(self, text: str) -> list[str]:
        """Split a single text document into a list of string chunks."""
        if not text or text.strip() == "":
            return []
        return self.splitter.split_text(text)
